# Read-only inventory. No enrollment, remote execution, SDK install, or GPU qualification.
[CmdletBinding()]
param([string]$OutputPath = "paralyn-windows-inventory.json")
$ErrorActionPreference = "Stop"
if (Test-Path -LiteralPath $OutputPath) { throw "Output already exists: $OutputPath" }
$os = Get-CimInstance Win32_OperatingSystem
$gpus = @(Get-CimInstance Win32_VideoController | ForEach-Object {
  [ordered]@{
    name = $_.Name
    pnp_device_id = $_.PNPDeviceID
    driver_version = $_.DriverVersion
    driver_date = $_.DriverDate.ToString("o")
    adapter_ram_reported_bytes = $_.AdapterRAM
    adapter_ram_note = "Win32_VideoController.AdapterRAM can be truncated; this is not an authoritative VRAM measurement."
    status = $_.Status
  }
})
$nvidia = $null
$nvidiaCommand = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidiaCommand) {
  $nvidia = @(& $nvidiaCommand.Source --query-gpu=name,uuid,memory.total,driver_version --format=csv,noheader,nounits 2>&1)
  if ($LASTEXITCODE -ne 0) { $nvidia = @("nvidia-smi failed", $nvidia) }
}
$result = [ordered]@{
  schema = "paralyn.hardware-inventory"
  schema_version = 1
  platform = "windows-native"
  captured_at = [DateTime]::UtcNow.ToString("o")
  os_caption = $os.Caption
  os_version = $os.Version
  os_build = $os.BuildNumber
  architecture = $os.OSArchitecture
  gpu_adapters = $gpus
  nvidia_smi = $nvidia
  qualification = "not_run"
  note = "Inventory is a prerequisite, not evidence that a Paralyn backend is implemented or qualified."
}
$json = $result | ConvertTo-Json -Depth 8
$utf8 = New-Object System.Text.UTF8Encoding($false)
$file = [System.IO.File]::Open($OutputPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
try { $bytes = $utf8.GetBytes($json + "`n"); $file.Write($bytes, 0, $bytes.Length) } finally { $file.Dispose() }
Write-Output "Inventory written to $OutputPath"
