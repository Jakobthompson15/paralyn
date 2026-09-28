#!/usr/bin/env python3
"""CUDA backend absence must be reported honestly by the CLI and native ABI.

The CUDA driver/NVRTC are forced absent through explicit nonexistent library
paths, so this CPU-only test is deterministic even on an NVIDIA machine. No GPU
work is submitted and nothing here is CUDA qualification evidence.
"""
import argparse
import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FORCED = {
    "PARALYN_CUDA_DRIVER_LIBRARY": "/nonexistent/paralyn-test/libcuda-forced-absent",
    "PARALYN_NVRTC_LIBRARY": "/nonexistent/paralyn-test/libnvrtc-forced-absent",
}


class BackendStatus(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("version", ctypes.c_uint32),
                ("backend", ctypes.c_char * 32), ("implemented", ctypes.c_uint32),
                ("available", ctypes.c_uint32), ("device_count", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32), ("driver_version", ctypes.c_int32),
                ("compiler_version", ctypes.c_int32), ("driver_library", ctypes.c_char * 512),
                ("compiler_library", ctypes.c_char * 512), ("reason", ctypes.c_char * 1024)]


class ErrorInfo(ctypes.Structure):
    _fields_ = [("code", ctypes.c_int), ("operation", ctypes.c_char * 64),
                ("message", ctypes.c_char * 1024)]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def run(paralyn, *arguments, code=0, cwd=None):
    environment = dict(os.environ, **FORCED)
    completed = subprocess.run([str(paralyn), *map(str, arguments)], capture_output=True, text=True,
                               env=environment, cwd=cwd, timeout=60)
    require(code is None or completed.returncode == code,
            f"{arguments}: exit {completed.returncode}\n{completed.stdout}\n{completed.stderr}")
    require("Verification: PASS" not in completed.stdout + completed.stderr,
            "an unavailable backend must never print a verification pass")
    return completed


def native_abi(library):
    for key, value in FORCED.items():
        os.environ[key] = value
    lib = ctypes.CDLL(str(library))
    lib.pr_backend_status_get.argtypes = [ctypes.c_char_p, ctypes.POINTER(BackendStatus)]
    lib.pr_context_create.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_uint64)]
    lib.pr_last_error.argtypes = [ctypes.POINTER(ErrorInfo)]
    status = BackendStatus(struct_size=ctypes.sizeof(BackendStatus), version=1)
    require(lib.pr_backend_status_get(b"cuda", ctypes.byref(status)) == 0, "cuda status query failed")
    require(status.implemented == 1 and status.available == 0 and status.device_count == 0,
            "forced-absent CUDA reported available")
    require(b"CUDA driver library not found" in status.reason, status.reason)
    hip = BackendStatus(struct_size=ctypes.sizeof(BackendStatus), version=1)
    require(lib.pr_backend_status_get(b"hip", ctypes.byref(hip)) == 0 and hip.implemented == 0,
            "HIP must not be reported as implemented")
    bad = BackendStatus(struct_size=ctypes.sizeof(BackendStatus), version=2)
    require(lib.pr_backend_status_get(b"cuda", ctypes.byref(bad)) == 5, "unknown query version accepted")
    require(lib.pr_backend_status_get(b"opencl", ctypes.byref(status)) == 1, "unknown backend accepted")
    for selector, text in ((b"cuda:0", b"CUDA backend unavailable"), (b"cuda:x", b"cuda:INDEX"),
                           (b"hip:0", b"not implemented")):
        handle = ctypes.c_uint64(0)
        result = lib.pr_context_create(selector, ctypes.byref(handle))
        error = ErrorInfo()
        lib.pr_last_error(ctypes.byref(error))
        require(result == 9 and handle.value == 0, f"{selector!r}: expected PR_DEVICE_UNAVAILABLE, got {result}")
        require(text in error.message, f"{selector!r}: {error.message!r}")


PLANTED = ("libcuda.dylib", "libnvrtc.dylib", "libcuda.so.1", "libcuda.so", "libnvrtc.so.13",
           "libnvrtc.so.12", "libnvrtc.so", "nvcuda.dll", "nvrtc64_130_0.dll", "nvrtc64_120_0.dll")
MARKER = "PARALYN_CWD_PROBE_LOADED"


def current_directory_not_searched(paralyn, library, probe):
    """Regression: device queries must not load CUDA libraries from the CWD.

    A harmless probe library is planted under every CUDA library name in the
    working directory with the PARALYN_* overrides unset (the default search).
    Loading it would create MARKER; a positive control proves the probe works.
    """
    environment = {k: v for k, v in os.environ.items() if k not in FORCED}
    with tempfile.TemporaryDirectory() as work:
        work = Path(work)
        for name in PLANTED:
            shutil.copyfile(probe, work / name)
        commands = (["devices", "--json"], ["devices"], ["support", "--json"],
                    ["doctor", "--device", "cuda:0", "--json"])
        for arguments in commands:
            completed = subprocess.run([str(paralyn), *arguments], capture_output=True, text=True,
                                       env=environment, cwd=work, timeout=60)
            output = completed.stdout + completed.stderr
            require(not (work / MARKER).exists(),
                    f"{arguments}: a library was loaded from the current working directory")
            require("lacks required symbols" not in output and str(work) not in output,
                    f"{arguments}: a planted library was opened:\n{output}")
            require("Verification: PASS" not in output, "no verification may pass here")
        # Native ABI and Python devices() in a process whose CWD is the planted directory.
        script = ("import ctypes, sys\n"
                  "lib = ctypes.CDLL(sys.argv[1])\n"
                  "n = ctypes.c_uint32()\n"
                  "assert lib.pr_device_count(ctypes.byref(n)) == 0\n"
                  "sys.path.insert(0, sys.argv[2])\n"
                  "import paralyn\n"
                  "paralyn.devices(sys.argv[1])\n")
        bindings = Path(__file__).resolve().parents[2] / "bindings" / "python"
        completed = subprocess.run([sys.executable, "-c", script, str(library), str(bindings)],
                                   capture_output=True, text=True, env=environment, cwd=work, timeout=60)
        require(completed.returncode == 0, f"native device query failed:\n{completed.stderr}")
        require(not (work / MARKER).exists(), "native ABI/Python loaded a library from the CWD")
        # Positive control: the probe does signal when loaded by absolute path.
        control = dict(environment, PARALYN_CUDA_DRIVER_LIBRARY=str(work / "libcuda.dylib"),
                       PARALYN_NVRTC_LIBRARY=str(work / "libnvrtc.dylib"))
        completed = subprocess.run([str(paralyn), "devices", "--json"], capture_output=True, text=True,
                                   env=control, cwd=work, timeout=60)
        require((work / MARKER).exists() and "lacks required symbols" in completed.stdout,
                "probe positive control failed; the CWD regression check would be meaningless")
        # A relative explicit override is refused, never resolved against the CWD.
        (work / MARKER).unlink()
        relative = dict(environment, PARALYN_CUDA_DRIVER_LIBRARY="libcuda.dylib",
                        PARALYN_NVRTC_LIBRARY="libnvrtc.dylib")
        completed = subprocess.run([str(paralyn), "devices", "--json"], capture_output=True, text=True,
                                   env=relative, cwd=work, timeout=60)
        require(not (work / MARKER).exists(), "relative override loaded a library from the CWD")
        cuda = json.loads(completed.stdout)["backends"]["cuda"]
        require(cuda["available"] is False and "not found" in cuda["reason"], f"relative override: {cuda}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paralyn", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    args = parser.parse_args()
    args.paralyn, args.library = args.paralyn.resolve(), args.library.resolve()
    current_directory_not_searched(args.paralyn, args.library, args.probe.resolve())

    support = json.loads(run(args.paralyn, "support", "--json").stdout)["support"]
    require(support["backends"]["cuda"] == "implemented_unqualified", "support backend state")
    cuda = support["backend_availability"]["cuda"]
    require(cuda["implemented"] is True and cuda["available"] is False and cuda["qualification"] == "unavailable",
            "support must report CUDA as implemented but unavailable/unqualified")
    require(support["backend_availability"]["hip"]["implemented"] is False, "HIP stub reported implemented")
    require(support["complete_portfolio"] is False, "portfolio claim")

    devices = json.loads(run(args.paralyn, "devices", "--json").stdout)
    require(all(d["backend"] != "CUDA" for d in devices["devices"]), "fabricated CUDA device")
    require(devices["backends"]["cuda"]["available"] is False, "devices reports CUDA available")
    human = run(args.paralyn, "devices").stdout
    require("CUDA backend: unavailable" in human, "human devices output hides CUDA absence")

    with tempfile.TemporaryDirectory() as work:
        evidence = Path(work) / "evidence"
        failed = run(args.paralyn, "doctor", "--device", "cuda:0", "--json", "--artifacts", evidence,
                     code=1, cwd=work)
        report = json.loads(failed.stdout)
        require(report["status"] == "failed" and report["diagnostic"]["id"] == "P-BACKEND-UNAVAILABLE",
                f"doctor cuda:0 diagnostic: {report}")
        require("never emulated" in report["diagnostic"]["message"], "absence message")
        require(not evidence.exists(), "an unavailable backend must not produce execution evidence")
        hip = json.loads(run(args.paralyn, "doctor", "--device", "hip:0", "--json", code=1, cwd=work).stdout)
        require(hip["diagnostic"]["id"] == "P-BACKEND-UNIMPLEMENTED", "HIP selector diagnostic")

    native_abi(args.library)
    print("CUDA unavailable reporting (CLI support/devices/doctor, native ABI selectors, current directory "
          "never searched for CUDA libraries; no GPU work): PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"CUDA unavailable-reporting test failed: {error}", file=sys.stderr)
        sys.exit(1)
