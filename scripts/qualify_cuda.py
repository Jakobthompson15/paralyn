#!/usr/bin/env python3
"""CUDA backend qualification harness for a machine with a real NVIDIA GPU.

Runs verified-IR add/affine kernels through the native C ABI on cuda:N, compares
every output value against an independent FP32 CPU reference, and records the
device, driver, NVRTC and evidence. It exits 2 with status "unavailable" (never a
pass) when the CUDA backend is not available. Only real comparisons print
"Verification: PASS".

--harness-self-test-metal runs the same comparisons on metal:0 solely to test this
harness on a Mac; its summary is marked cuda_qualification=false.
"""
import argparse
import ctypes
import json
import os
import random
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bindings/python"))


class BackendStatus(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("version", ctypes.c_uint32),
                ("backend", ctypes.c_char * 32), ("implemented", ctypes.c_uint32),
                ("available", ctypes.c_uint32), ("device_count", ctypes.c_uint32),
                ("reserved", ctypes.c_uint32), ("driver_version", ctypes.c_int32),
                ("compiler_version", ctypes.c_int32), ("driver_library", ctypes.c_char * 512),
                ("compiler_library", ctypes.c_char * 512), ("reason", ctypes.c_char * 1024)]


class Timing(ctypes.Structure):
    _fields_ = [("struct_size", ctypes.c_uint32), ("version", ctypes.c_uint32),
                ("completed", ctypes.c_uint32), ("duration_valid", ctypes.c_uint32),
                ("timestamps_valid", ctypes.c_uint32), ("clock_domain", ctypes.c_int),
                ("duration_seconds", ctypes.c_double), ("start_seconds", ctypes.c_double),
                ("end_seconds", ctypes.c_double)]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def timing(pr, event):
    record = Timing(struct_size=ctypes.sizeof(Timing), version=1)
    lib = event._lib
    lib.api.pr_event_timing.argtypes = [ctypes.c_uint64, ctypes.POINTER(Timing)]
    lib.check(lib.api.pr_event_timing(event.handle, ctypes.byref(record)))
    return record


def launch_case(pr, context, queue, kernels, name, n, seed, offset_elements, index_type):
    """One launch on fresh buffers with guard sentinels; returns a result record."""
    rng = random.Random(seed)
    guard = 5
    total = n + offset_elements + guard
    a = [f32(rng.randint(-4096, 4096) * 0.125) for _ in range(total)]
    b = [f32(rng.randint(-4096, 4096) * 0.0625) for _ in range(total)]
    sentinel = f32(-7777.5)
    scale = f32(rng.choice([0.5, -1.25, 3.0, 0.1]))
    pack = lambda values: struct.pack(f"<{len(values)}f", *values)
    buffers = []
    try:
        da, db, dc = (context.buffer(total * 4) for _ in range(3))
        buffers += [da, db, dc]
        da.write(pack(a)), db.write(pack(b)), dc.write(pack([sentinel] * total))
        start, size = offset_elements * 4, max(n, 1) * 4
        views = [da.view(offset=start, size=size, access=pr.Access.READ),
                 db.view(offset=start, size=size, access=pr.Access.READ),
                 dc.view(offset=start, size=size, access=pr.Access.WRITE)]
        buffers += views
        count = pr.i32(n) if index_type == "i32" else pr.u32(n)
        arguments = views + [count] + ([pr.f32(scale)] if name.endswith("affine") else [])
        block = 128
        event = queue.launch(kernels[name], arguments, grid=(max(1, (n + block - 1) // block), 1, 1),
                             block=(block, 1, 1))
        buffers.append(event)
        info = timing(pr, event)
        raw = dc.read()
        actual = list(struct.unpack(f"<{total}f", raw))
        for i in range(total):
            inside = offset_elements <= i < offset_elements + n
            if name.endswith("affine"):
                expected = f32(f32(a[i] * scale) + b[i]) if inside else sentinel
            else:
                expected = f32(a[i] + b[i]) if inside else sentinel
            if struct.pack("<f", actual[i]) != struct.pack("<f", expected):
                raise AssertionError(f"{name} n={n} offset={offset_elements}: element {i} is "
                                     f"{actual[i]!r}, independent reference {expected!r}")
        return {"kernel": name, "elements": n, "view_offset_elements": offset_elements,
                "checked_values": total, "completed": bool(info.completed),
                "duration_valid": bool(info.duration_valid),
                "timestamps_valid": bool(info.timestamps_valid),
                "clock_domain": info.clock_domain,
                "gpu_duration_seconds": info.duration_seconds if info.duration_valid else None}
    finally:
        for owned in reversed(buffers):
            owned.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paralyn", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True, help="built paralyn native library")
    parser.add_argument("--module", type=Path, required=True,
                        help="verified-IR module with array_add/array_affine (build/operators.prk) "
                             "or vector_add/affine (build/native-kernels.prk)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True, help="new evidence directory")
    parser.add_argument("--harness-self-test-metal", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit(f"{args.output} exists; evidence is never overwritten")
    selector = "metal:0" if args.harness_self_test_metal else args.device
    os.environ["PARALYN_LIBRARY"] = str(args.library.resolve())
    import paralyn as pr

    library = pr._library(str(args.library.resolve()))
    status = BackendStatus(struct_size=ctypes.sizeof(BackendStatus), version=1)
    library.api.pr_backend_status_get.argtypes = [ctypes.c_char_p, ctypes.POINTER(BackendStatus)]
    library.check(library.api.pr_backend_status_get(b"cuda", ctypes.byref(status)))
    cuda_status = {"available": bool(status.available), "device_count": status.device_count,
                   "driver_version": status.driver_version, "nvrtc_version": status.compiler_version,
                   "driver_library": status.driver_library.decode(),
                   "nvrtc_library": status.compiler_library.decode(), "reason": status.reason.decode()}
    summary = {"schema": "paralyn.cuda-qualification", "schema_version": 1, "selector": selector,
               "cuda_qualification": not args.harness_self_test_metal, "cuda_backend": cuda_status}
    if not args.harness_self_test_metal and not status.available:
        summary["status"] = "unavailable"
        print(json.dumps(summary, indent=2))
        print("CUDA qualification: UNAVAILABLE (not a pass)", file=sys.stderr)
        return 2

    args.output.mkdir(parents=True)
    devices = subprocess.run([str(args.paralyn), "devices", "--json"], capture_output=True, text=True)
    (args.output / "devices.json").write_text(devices.stdout)
    doctor = subprocess.run([str(args.paralyn), "doctor", "--device", selector, "--json",
                             "--artifacts", str(args.output / "doctor")], capture_output=True, text=True)
    (args.output / "doctor.stdout").write_text(doctor.stdout)
    (args.output / "doctor.stderr").write_text(doctor.stderr)
    require(doctor.returncode == 0, f"doctor failed: {doctor.stdout}{doctor.stderr}")
    doctor_report = json.loads(doctor.stdout)
    require(doctor_report["verification"]["status"] == "passed", "doctor verification")

    results = []
    with pr.Context(selector) as context:
        device = context.device
        expected_backend = "Metal" if args.harness_self_test_metal else "CUDA"
        require(device.backend == expected_backend, f"selected {device.backend}, expected {expected_backend}")
        with pr.Module.load_file(context, str(args.module)) as module, context.queue() as queue:
            names = [p for p in ("array_add", "array_affine", "vector_add", "affine")]
            kernels = {}
            for name in names:
                try:
                    kernels[name] = module.kernel(name)
                except pr.Error:
                    pass
            require(kernels, "module contains none of the add/affine kernels")
            seed = 20260928
            for name, kernel in kernels.items():
                index_type = "u32" if name.startswith("array_") else "i32"
                for n in (1, 127, 128, 129, 1000, 65537, 1000003):
                    for offset in (0, 3):
                        results.append(launch_case(pr, context, queue, kernels, name, n, seed, offset, index_type))
                        seed += 1
            for kernel in kernels.values():
                kernel.close()
        context.write_evidence(str(args.output / "native"))
    execution = json.loads((args.output / "native/execution.json").read_text())
    require(execution["cpu_fallback"] is False, "execution evidence cpu_fallback")
    if not args.harness_self_test_metal:
        require(execution["backend"] == "CUDA" and execution["test_double"] is False, "CUDA evidence identity")
        for event in results:
            require(event["completed"] and event["duration_valid"] and not event["timestamps_valid"]
                    and event["clock_domain"] == 1, "CUDA duration-only timing contract")
    summary.update({"status": "passed", "device": {"name": device.name, "backend": device.backend,
                                                   "os": device.os},
                    "launches": len(results), "checked_values": sum(r["checked_values"] for r in results),
                    "results": results, "doctor_report": "doctor.stdout",
                    "evidence": "native/execution.json"})
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    label = "harness self-test on Metal (not CUDA qualification)" if args.harness_self_test_metal else selector
    print(f"{label}: {len(results)} launches, {summary['checked_values']} values independently compared")
    print("Verification: PASS")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except AssertionError as error:
        print(f"CUDA qualification failed: {error}", file=sys.stderr)
        sys.exit(1)
