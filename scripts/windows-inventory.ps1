# Read-only inventory. No enrollment, remote execution, SDK install, or GPU qualification.
# Records GPU model, VRAM, driver, OS, and CUDA/HIP availability as JSON for a person
# preparing native Windows qualification. Windows PowerShell 5.1 and PowerShell 7 compatible.
# UNTESTED: written without access to a Windows machine; run it and review the JSON.
[CmdletBinding()]
param(
  [string]$OutputPath = "paralyn-windows-inventory.json",
  # Optional: a built paralyn.exe. Only `paralyn devices --json` (enumeration, no GPU work) is run.
  [string]$ParalynPath = ""
)
$ErrorActionPreference = "Stop"
if (Test-Path -LiteralPath $OutputPath) { throw "Output already exists: $OutputPath" }

function File-Info([string]$Path) {
  if (-not (Test-Path -LiteralPath $Path)) { return $null }
  $item = Get-Item -LiteralPath $Path
  return [ordered]@{
    path = $item.FullName
    file_version = $item.VersionInfo.FileVersion
    product_version = $item.VersionInfo.ProductVersion
    bytes = $item.Length
  }
}
function Invoke-Native([string]$Path, [string[]]$Arguments) {
  # Windows PowerShell 5.1 turns redirected native stderr into terminating errors
  # under "Stop"; native tools report through exit codes, so relax it locally.
  $ErrorActionPreference = "Continue"
  $output = @(& $Path @Arguments 2>&1 | ForEach-Object { "$_" })
  return [ordered]@{ exit_code = $LASTEXITCODE; output = $output }
}
function Run-Tool([string]$Name, [string[]]$Arguments) {
  $command = Get-Command $Name -ErrorAction SilentlyContinue
  if (-not $command) { return [ordered]@{ found = $false } }
  $run = Invoke-Native $command.Source $Arguments
  return [ordered]@{ found = $true; path = $command.Source; exit_code = $run.exit_code; output = $run.output }
}
function To-UInt64([string]$Text) {
  $value = [uint64]0
  if ([uint64]::TryParse($Text, [ref]$value)) { return $value }
  return $null
}

$os = Get-CimInstance Win32_OperatingSystem
$currentVersion = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion" -ErrorAction SilentlyContinue
$computer = Get-CimInstance Win32_ComputerSystem

# Display adapters. WDDM records a 64-bit memory size in the adapter's class key;
# Win32_VideoController.AdapterRAM is a 32-bit value that saturates near 4 GiB.
$classRoot = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
$classKeys = @(Get-ChildItem $classRoot -ErrorAction SilentlyContinue | Where-Object { $_.PSChildName -match '^\d{4}$' })
$gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object {
  $controller = $_
  $vram = $null
  $vramSource = "unavailable"
  foreach ($key in $classKeys) {
    $properties = Get-ItemProperty $key.PSPath -ErrorAction SilentlyContinue
    if ($properties -and $properties.DriverDesc -eq $controller.Name) {
      $size = $properties."HardwareInformation.qwMemorySize"
      if ($size) { $vram = [uint64]$size; $vramSource = "registry HardwareInformation.qwMemorySize" }
      break
    }
  }
  $driverDate = $null
  if ($controller.DriverDate) { $driverDate = $controller.DriverDate.ToString("o") }
  [ordered]@{
    name = $controller.Name
    vendor = $controller.AdapterCompatibility
    pnp_device_id = $controller.PNPDeviceID
    driver_version = $controller.DriverVersion
    driver_date = $driverDate
    vram_bytes = $vram
    vram_source = $vramSource
    adapter_ram_reported_bytes = $controller.AdapterRAM
    adapter_ram_note = "Win32_VideoController.AdapterRAM is 32-bit and can be truncated; not authoritative."
    status = $controller.Status
  }
})

# NVIDIA: driver-reported per-GPU facts (authoritative VRAM in MiB, driver, compute capability).
$nvidia = [ordered]@{ nvidia_smi_found = $false; gpus = @() }
$smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($smi) {
  $nvidia["nvidia_smi_found"] = $true
  $nvidia["nvidia_smi_path"] = $smi.Source
  $query = Invoke-Native $smi.Source @("--query-gpu=index,name,uuid,memory.total,driver_version,pci.bus_id", "--format=csv,noheader,nounits")
  $nvidia["exit_code"] = $query.exit_code
  if ($query.exit_code -eq 0) {
    $nvidia["gpus"] = @($query.output | Where-Object { $_.Trim() } | ForEach-Object {
      $f = @($_.Split(",") | ForEach-Object { $_.Trim() })
      [ordered]@{ index = [int]$f[0]; name = $f[1]; uuid = $f[2]; memory_total_mib = (To-UInt64 $f[3]);
                  driver_version = $f[4]; pci_bus_id = $f[5] }
    })
    # compute_cap is only understood by newer drivers; record failure instead of guessing.
    $caps = Invoke-Native $smi.Source @("--query-gpu=index,compute_cap", "--format=csv,noheader")
    if ($caps.exit_code -eq 0) {
      foreach ($line in $caps.output) {
        $f = @($line.Split(",") | ForEach-Object { $_.Trim() })
        foreach ($gpu in $nvidia["gpus"]) { if ("$($gpu["index"])" -eq $f[0]) { $gpu["compute_capability"] = $f[1] } }
      }
    } else { $nvidia["compute_capability_query_failed"] = $caps.output }
    $header = Invoke-Native $smi.Source @()
    $match = [regex]::Match((($header.output | Select-Object -First 6) -join "`n"), "CUDA Version:\s*([0-9.]+)")
    if ($match.Success) { $nvidia["driver_max_cuda_version"] = $match.Groups[1].Value }
  } else { $nvidia["raw_output"] = $query.output }
}

# CUDA availability as Paralyn's dynamically loaded backend sees it.
$system32 = Join-Path $env:SystemRoot "System32"
$cudaPath = $env:CUDA_PATH
$nvrtc = @()
if ($cudaPath) {
  foreach ($dir in @("bin\x64", "bin")) {
    $nvrtc += @(Get-ChildItem -Path (Join-Path $cudaPath $dir) -Filter "nvrtc64_*.dll" -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -notmatch "builtins" } | ForEach-Object { File-Info $_.FullName })
  }
}
$cuda = [ordered]@{
  driver_library = File-Info (Join-Path $system32 "nvcuda.dll")
  cuda_path = $cudaPath
  nvrtc_libraries = $nvrtc
  nvcc = Run-Tool "nvcc" @("--version")
  paralyn_backend_prerequisites_present = [bool]((Test-Path (Join-Path $system32 "nvcuda.dll")) -and $nvrtc.Count -gt 0)
}

# HIP/ROCm on Windows (HIP SDK). Paralyn has no HIP backend; this only records presence.
$hipPath = $env:HIP_PATH
$hipRuntime = @(Get-ChildItem -Path $system32 -Filter "amdhip64*.dll" -ErrorAction SilentlyContinue | ForEach-Object { File-Info $_.FullName })
$hiprtc = @()
if ($hipPath) {
  $hiprtc = @(Get-ChildItem -Path (Join-Path $hipPath "bin") -Filter "hiprtc*.dll" -ErrorAction SilentlyContinue | ForEach-Object { File-Info $_.FullName })
}
$hip = [ordered]@{
  hip_path = $hipPath
  runtime_libraries = $hipRuntime
  hiprtc_libraries = $hiprtc
  hipcc = Run-Tool "hipcc" @("--version")
  hipinfo = Run-Tool "hipInfo" @()
  paralyn_backend = "not_implemented"
}

$paralyn = $null
if ($ParalynPath) {
  $run = Invoke-Native $ParalynPath @("devices", "--json")
  $paralyn = [ordered]@{ exit_code = $run.exit_code; output = $run.output }
}

$result = [ordered]@{
  schema = "paralyn.hardware-inventory"
  schema_version = 2
  platform = "windows-native"
  captured_at = [DateTime]::UtcNow.ToString("o")
  os = [ordered]@{
    caption = $os.Caption
    version = $os.Version
    build = $os.BuildNumber
    update_build_revision = $currentVersion.UBR
    display_version = $currentVersion.DisplayVersion
    architecture = $os.OSArchitecture
  }
  machine = [ordered]@{
    manufacturer = $computer.Manufacturer
    model = $computer.Model
    physical_memory_bytes = $computer.TotalPhysicalMemory
  }
  gpu_adapters = $gpus
  nvidia = $nvidia
  cuda = $cuda
  hip = $hip
  paralyn_devices = $paralyn
  qualification = "not_run"
  note = "Inventory is a prerequisite, not evidence that a Paralyn backend is implemented or qualified."
}
$json = $result | ConvertTo-Json -Depth 10
$utf8 = New-Object System.Text.UTF8Encoding($false)
$file = [System.IO.File]::Open($OutputPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
try { $bytes = $utf8.GetBytes($json + "`n"); $file.Write($bytes, 0, $bytes.Length) } finally { $file.Dispose() }
Write-Output "Inventory written to $OutputPath"
