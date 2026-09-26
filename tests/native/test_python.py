#!/usr/bin/env python3
"""Native Python/C ABI qualification; requires real Metal hardware and a compiled .prk."""
import argparse
from array import array
from contextlib import ExitStack
import ctypes as c
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys

import paralyn as pr


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rejects(function, codes=None, kinds=(pr.Error,)):
    try:
        function()
    except kinds as error:
        if isinstance(error, pr.Error):
            require(error.operation and error.message, "native error lost its operation/detail")
            if codes is not None:
                require(error.code in codes, f"wrong error: {error}")
        return
    raise RuntimeError("invalid operation unexpectedly succeeded")


def floats(data):
    result = array("f")
    result.frombytes(data)
    return list(result)


def checked_event(event, records, label):
    info = event.wait()
    require(info.completed and math.isfinite(info.gpu_start_seconds) and
            math.isfinite(info.gpu_end_seconds) and
            0 < info.gpu_start_seconds < info.gpu_end_seconds,
            "missing actual GPU completion/timing evidence")
    again = event.wait()
    require(info == again, "repeated wait changed event evidence")
    records.append(dict(label=label, **asdict(info)))
    return info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    args = parser.parse_args()
    require(not args.artifacts.exists(), "artifacts directory must not already exist")
    args.artifacts.mkdir(parents=True)
    require(array("f").itemsize == 4, "host array('f') is not FP32")
    require(pr.ABI_VERSION == 1 and pr.devices(), "missing native ABI or real device")
    records = []
    with ExitStack() as owners:
        context = owners.enter_context(pr.Context())
        device = context.device
        require(device.backend == "Metal" and device.registry_id > 0, "not a physical Metal device")
        module = owners.enter_context(pr.Module.load_file(context, args.module))
        kernel = owners.enter_context(module.kernel("vector_add"))
        affine = owners.enter_context(module.kernel("affine"))
        queue = owners.enter_context(context.queue())
        queue2 = owners.enter_context(context.queue())
        require(queue.handle != queue2.handle, "acquired queue handles must have independent ownership")
        parameters = kernel.parameters
        require(len(parameters) == 4 and [p.type for p in parameters] ==
                [pr.Type.F32, pr.Type.F32, pr.Type.F32, pr.Type.I32], "wrong parameter reflection")
        require([p.is_buffer for p in parameters] == [True, True, True, False], "wrong argument kinds")
        require(parameters[0].access == pr.Access.READ and parameters[1].access == pr.Access.READ
                and parameters[2].access == pr.Access.WRITE, "access reflection does not describe IR effects")
        n, prefix, suffix = 1003, 5, 7
        a = [(i % 31 - 15) * 0.5 for i in range(n)]
        b = [(i % 17 - 8) * 0.25 for i in range(n)]
        sentinel = -9999.0
        da = owners.enter_context(context.buffer((n + prefix + suffix) * 4))
        db = owners.enter_context(context.buffer((n + prefix + suffix) * 4))
        output = owners.enter_context(context.buffer((n + prefix + suffix) * 4))
        da.write(array("f", [sentinel] * prefix + a + [sentinel] * suffix).tobytes())
        db.write(array("f", [sentinel] * prefix + b + [sentinel] * suffix).tobytes())
        av = owners.enter_context(da.view(offset=prefix * 4, size=n * 4, access=pr.Access.READ))
        bv = owners.enter_context(db.view(offset=prefix * 4, size=n * 4, access=pr.Access.READ))
        cv = owners.enter_context(output.view(offset=prefix * 4, size=n * 4, access=pr.Access.WRITE))
        require(da.size == (n + prefix + suffix) * 4, "buffer size changed")
        for selected, count, scale, label in (
            (kernel, n, None, "vector_add_offset_partial_block"),
            (affine, n, -2.0, "affine_negative_scalar"),
            (affine, 257, 0.25, "affine_short_scalar_bound"),
            (kernel, 0, None, "zero_logical_length"),
            (kernel, n, None, "repeated_vector_add"),
        ):
            output.write(array("f", [sentinel] * (n + prefix + suffix)).tobytes())
            arguments = [av, bv, cv, pr.i32(count)]
            if scale is not None:
                arguments.append(pr.f32(scale))
            with queue.launch(selected, arguments, grid=((n + 63) // 64, 1, 1), block=(64, 1, 1)) as event:
                checked_event(event, records, label)
                rejects(event.cancel, {pr.Status.UNSUPPORTED})
            expected = [sentinel] * prefix + [
                (a[i] + b[i] if scale is None else a[i] * scale + b[i])
                if i < count else sentinel for i in range(n)] + [sentinel] * suffix
            require(floats(output.read()) == expected, f"CPU comparison/canaries failed: {label}")
        require(floats(da.read()) == [sentinel] * prefix + a + [sentinel] * suffix,
                "read-only input changed")
        # Same-allocation read/write views exercise alias grouping with byte offsets.
        alias_out = owners.enter_context(da.view(offset=prefix * 4, size=n * 4,
                                                 access=pr.Access.READ_WRITE))
        with queue2.launch(kernel, [av, bv, alias_out, pr.i32(n)],
                           grid=(16, 1, 1), block=(64, 1, 1)) as event:
            checked_event(event, records, "same_allocation_alias")
        require(floats(da.read()) == [sentinel] * prefix + [x + y for x, y in zip(a, b)] +
                [sentinel] * suffix, "alias output/canaries differ from CPU")

        with context.buffer(0) as empty:
            require(empty.size == 0 and empty.read() == b"", "zero buffer contract failed")
            empty.write(b"")
            with empty.view(size=0) as empty_view:
                require(empty_view.handle != 0, "zero view has invalid ownership handle")
        invalid = {pr.Status.INVALID_ARGUMENT}
        bounds = {pr.Status.OUT_OF_BOUNDS}
        rejects(lambda: da.view(offset=da.size + 1, size=0), bounds)
        rejects(lambda: da.view(offset=2, size=4, alignment=4), invalid)
        rejects(lambda: da.view(offset=0, size=4, alignment=3), {pr.Status.INVALID_ARGUMENT, pr.Status.UNSUPPORTED})
        rejects(lambda: output.read(4, offset=output.size), bounds)
        rejects(lambda: output.write(b"bad!", offset=output.size), bounds)
        rejects(lambda: output.view(offset=(1 << 64) - 1, size=4), bounds)
        rejects(lambda: queue.launch(kernel, [av, bv, cv, pr.u32(n)], grid=16, block=64), invalid)
        rejects(lambda: queue.launch(kernel, [av, bv, cv], grid=16, block=64), invalid)
        rejects(lambda: queue.launch(kernel, [pr.i32(1), bv, cv, pr.i32(n)], grid=16, block=64), invalid)
        rejects(lambda: queue.launch(kernel, [av, bv, cv, pr.i32(n)], grid=0, block=64), invalid)
        rejects(lambda: queue.launch(kernel, [av, bv, cv, pr.i32(n)], grid=1, block=(1 << 32) - 1), invalid)
        output_readonly = owners.enter_context(output.view(offset=prefix * 4, size=n * 4,
                                                           access=pr.Access.READ))
        rejects(lambda: queue.launch(kernel, [av, bv, output_readonly, pr.i32(n)], grid=16, block=64),
                {pr.Status.INVALID_ARGUMENT, pr.Status.UNSUPPORTED})
        with pr.Context() as other:
            with other.buffer(n * 4) as foreign, foreign.view(access=pr.Access.READ) as foreign_view:
                rejects(lambda: queue.launch(kernel, [foreign_view, bv, cv, pr.i32(n)],
                                              grid=16, block=64), {pr.Status.CONTEXT_MISMATCH})
        rejects(lambda: module.kernel("missing_kernel"), {pr.Status.INVALID_ARGUMENT})
        rejects(lambda: pr.Module.load(context, b"not a Paralyn artifact"),
                {pr.Status.INVALID_ARGUMENT, pr.Status.COMPILATION_FAILED})
        # Standard-library memoryviews exercise the non-C-contiguous rejection.
        # This standard-library suite does not qualify a Fortran-only provider; that layout is rejected
        # by the same explicit c_contiguous boundary, without converting its order.
        strided = memoryview(array("f", [1, 2, 3, 4]))[::2]
        before_bad_upload = output.read()
        rejects(lambda: output.write(strided), kinds=(ValueError,))
        require(output.read() == before_bad_upload, "rejected strided upload changed the buffer")
        rejects(lambda: pr.Module.load(context, memoryview(args.module.read_bytes())[::2]),
                kinds=(ValueError,))
        for factory, value in ((pr.i32, 1 << 31), (pr.i32, -(1 << 31) - 1),
                               (pr.u32, -1), (pr.u32, 1 << 32), (pr.f32, 1e40)):
            rejects(lambda factory=factory, value=value: factory(value), kinds=(ValueError,))
        rejects(lambda: pr.i32(1.5), kinds=(TypeError,))
        rejects(lambda: pr.f32("1.0"), kinds=(TypeError,))
        rejects(lambda: queue.launch(kernel, [av, bv, cv, n], grid=16, block=64), kinds=(TypeError,))
        rejects(lambda: context.buffer(-1), kinds=(ValueError,))
        rejects(lambda: context.buffer(1 << 64), kinds=(ValueError,))
        rejects(lambda: queue.launch(kernel, [av, bv, cv, pr.i32(n)], grid=-1, block=64), kinds=(ValueError,))
        require(pr.f32(float("inf")).value == float("inf") and math.isnan(pr.f32(float("nan")).value),
                "explicit IEEE exceptional scalar conversion changed")
        # Direct calls verify that C ABI guards survive bypassing Python-side validation.
        lib = context._lib
        native_size = c.c_uint64()
        rejects(lambda: lib.check(lib.api.pr_buffer_size(kernel.handle, c.byref(native_size))),
                {pr.Status.INVALID_HANDLE})
        rejects(lambda: lib.check(lib.api.pr_buffer_read(output.handle, 0, None, 4)), invalid)
        temp = context.buffer(4)
        stale = temp.handle
        temp.close()
        temp.close()  # Python close is idempotent; the C ownership identity is not reusable.
        rejects(lambda: lib.check(lib.api.pr_release(stale)), {pr.Status.INVALID_HANDLE})
        rejects(lambda: temp.read(), {pr.Status.INVALID_HANDLE})
        temp2 = context.buffer(4)
        require(temp2.handle != stale, "freed identity was reused")
        temp2.close()
        # C ABI scalars are copied at launch, not borrowed from ctypes storage.
        output.write(array("f", [sentinel] * (n + prefix + suffix)).tobytes())
        native_arguments = (pr._Argument * 4)()
        for i, view in enumerate((av, bv, cv)):
            native_arguments[i].type, native_arguments[i].view = pr.Type.BUFFER, view.handle
        native_arguments[3].type, native_arguments[3].i32 = pr.Type.I32, n
        native_event = c.c_uint64()
        lib.check(lib.api.pr_launch(queue.handle, kernel.handle, pr._Dim3(16, 1, 1),
                                   pr._Dim3(64, 1, 1), native_arguments, 4, c.byref(native_event)))
        native_arguments[3].i32 = 0
        with pr.Event._adopt(lib, native_event.value) as event:
            checked_event(event, records, "ctypes_scalar_storage_copied")
        expected_twice = [sentinel] * prefix + [x + y + y for x, y in zip(a, b)] + [sentinel] * suffix
        require(floats(output.read()) == expected_twice, "launch borrowed scalar storage")
        # Allocation handles and module handles can be consumed before retained children launch.
        da.close()
        db.close()
        module.close()
        with queue.launch(kernel, [av, bv, cv, pr.i32(n)], grid=16, block=64) as event:
            kernel.close()
            checked_event(event, records, "released_buffer_module_kernel_owners")
        require(floats(output.read()) == expected_twice, "children lost allocation/module lifetime")
        queue2.synchronize()
        context.synchronize()
        context.write_evidence(args.artifacts / "primary")
        # A context handle is not the lifetime of its retained queue/kernel/allocation children.
        context.close()
        with queue.launch(affine, [av, bv, cv, pr.i32(n), pr.f32(0.5)], grid=16, block=64) as event:
            affine.close()
            queue.close()
            checked_event(event, records, "released_context_queue_kernel_owners")
        expected_lifetime = [sentinel] * prefix + [(x + y) * 0.5 + y for x, y in zip(a, b)] + [sentinel] * suffix
        require(floats(output.read()) == expected_lifetime, "context release invalidated children")
    report = {"status": "verified", "api": "Python ctypes over Paralyn C ABI 1", "cpu_fallback": False,
              "device": asdict(device), "module": str(args.module.resolve()),
              "native_evidence_directory": "primary",
              "evidence_scope": "first eight events have full native command/source records; the final ownership test intentionally consumes the context handle before launching and records its actual pr_event_wait completion/timestamps here",
              "events": records, "checks": "vector_add/affine, offset views/canaries, explicit scalars, repeated/zero logical lengths, aliasing, ownership, invalid inputs/handles/contexts, C ABI scalar copy"}
    (args.artifacts / "python-qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.artifacts / "verification.txt").write_text("Verification: PASS\nAll native Python outputs independently compared with CPU references.\n")
    print(f"Native Python qualification: PASS ({device.name}; {len(records)} completed GPU events)")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Native Python qualification failed: {error}", file=sys.stderr)
        sys.exit(1)
