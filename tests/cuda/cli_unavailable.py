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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paralyn", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    args = parser.parse_args()
    args.paralyn, args.library = args.paralyn.resolve(), args.library.resolve()

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
    print("CUDA unavailable reporting (CLI support/devices/doctor, native ABI selectors; no GPU work): PASS")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as error:
        print(f"CUDA unavailable-reporting test failed: {error}", file=sys.stderr)
        sys.exit(1)
