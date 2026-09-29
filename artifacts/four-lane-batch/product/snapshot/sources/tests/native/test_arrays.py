#!/usr/bin/env python3
"""Real-GPU FP32 array qualification, including ownership and rejected inputs."""
from array import array
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import paralyn as p


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def rejects(function, exception=(ValueError, TypeError, p.Error)):
    try:
        function()
    except exception:
        return
    raise RuntimeError("invalid array operation unexpectedly succeeded")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--module", type=Path)
    args = parser.parse_args()
    if args.module:
        os.environ["PARALYN_OPERATORS"] = str(args.module.resolve())
    require(not args.artifacts.exists(), "evidence directory must be new")
    events = []
    with p.Context() as context:
        for n in (0, 1, 17, 256, 257, 1003, 1000003):
            a_host = array("f", ((i % 61 - 30) * 0.5 for i in range(n)))
            b_host = array("f", ((i % 37 - 18) * 0.25 for i in range(n)))
            expected = [(x + y) * 0.5 + (x * -2 + y) for x, y in zip(a_host, b_host)]
            with ExitStack() as owners:
                a = owners.enter_context(p.asarray(a_host, context=context, shape=(n,)))
                b = owners.enter_context(p.asarray(b_host, context=context))
                require(a.shape == (n,) and a.dtype == p.float32 and a.nbytes == 4 * n,
                        "array metadata is wrong")
                require(a.wait() is None, "upload fabricated GPU work")
                if n:
                    a_host[0] = 9999
                    require(a.to_host()[0] != 9999, "asarray borrowed mutable host memory")
                summed = owners.enter_context(p.add(a, b))
                scaled = owners.enter_context(p.affine(a, b, -2))
                chained = owners.enter_context(p.affine(summed, scaled, 0.5))
                a.close()
                b.close()
                for label, value in (("add", summed), ("affine", scaled), ("dependent_affine", chained)):
                    info = value.wait()
                    if n:
                        require(info and info.completed and 0 < info.gpu_start_seconds < info.gpu_end_seconds,
                                "missing physical GPU event")
                        events.append(dict(length=n, operation=label, **asdict(info)))
                    else:
                        require(info is None, "empty operation fabricated a GPU event")
                summed.close()
                scaled.close()
                require(list(chained.to_host()) == expected, f"independent CPU mismatch at length {n}")
                snapshot = chained.to_host()
                if n:
                    snapshot[0] = 9999
                    require(chained.to_host()[0] != 9999, "to_host returned borrowed device storage")
        with p.asarray([1, 2.5, -4], context=context) as a:
            with p.add(a, a) as aliased:
                require(list(aliased.to_host()) == [2, 5, -8], "same-input alias result failed")
                events.append(dict(length=3, operation="same_input_alias", **asdict(aliased.wait())))
            with p.asarray([1], context=context) as short:
                rejects(lambda: p.add(a, short), (ValueError,))
            with p.Context() as other, p.asarray([1, 2, 3], context=other) as foreign:
                rejects(lambda: p.add(a, foreign), (p.Error,))
            rejects(lambda: p.affine(a, a, 1e40), (ValueError,))
            rejects(lambda: p.affine(a, a, "2"), (TypeError,))
            rejects(lambda: p.affine(a, a, None), (TypeError,))
            rejects(lambda: p.affine(a, a, True), (TypeError,))
        rejects(lambda: a.to_host(), (p.Error,))
        rejects(lambda: p.add(a, a), (p.Error,))
        for options in ({"shape": (1, 2)}, {"shape": (3,)}, {"shape": (-1,)},
                        {"shape": (True,)}, {"dtype": "float64"}):
            rejects(lambda options=options: p.asarray([1, 2], context=context, **options))
        rejects(lambda: p.asarray([True], context=context), (TypeError,))
        rejects(lambda: p.asarray([[1, 2]], context=context), (TypeError,))
        rejects(lambda: p.asarray(array("d", [1, 2]), context=context), (TypeError,))
        rejects(lambda: p.asarray(memoryview(array("f", [1, 2, 3, 4]))[::2], context=context), (ValueError,))
        rejects(lambda: p.asarray(b"\0\0\0\0", context=context), (TypeError,))
        # A failure after output allocation must unwind the newly allocated buffer.
        with p.asarray([1, 2], context=context) as broken, p.asarray([3, 4], context=context) as valid:
            broken._buffer.close()
            rejects(lambda: p.add(broken, valid), (p.Error,))
        # The operator module is a real dependency; no missing-module CPU path exists.
        old_path = os.environ.get("PARALYN_OPERATORS")
        os.environ["PARALYN_OPERATORS"] = str(args.artifacts / "missing.prk")
        try:
            with p.Context() as isolated, p.asarray([1], context=isolated) as one:
                rejects(lambda: p.add(one, one), (FileNotFoundError,))
        finally:
            if old_path is None:
                os.environ.pop("PARALYN_OPERATORS", None)
            else:
                os.environ["PARALYN_OPERATORS"] = old_path
        context.write_evidence(args.artifacts)
        device = context.device
    evidence = json.loads((args.artifacts / "execution.json").read_text())
    require(len(evidence["launches"]) == len(events) == 19, "wrong array GPU launch count")
    require(evidence["runtime_owned_current_buffer_bytes"] == 0, "array failure/cleanup leaked buffers")
    require(evidence["backend"] == "Metal" and evidence["cpu_fallback"] is False, "wrong execution policy")
    for event, launch in zip(events, evidence["launches"]):
        require(launch["command_status"] == "completed" and not launch["error"], "array command failed")
        require((event["gpu_start_seconds"], event["gpu_end_seconds"]) ==
                (launch["gpu_start_seconds"], launch["gpu_end_seconds"]), "event/native evidence differs")
        require((args.artifacts / launch["source_file"]).is_file(), "missing dispatched array source")
    # Context close drains submitted work; its surviving output can still materialize.
    context = p.Context()
    with p.asarray([1, 2, 3], context=context) as a, p.asarray([4, 5, 6], context=context) as b:
        with p.add(a, b) as result:
            context.close()
            require(list(result.to_host()) == [5, 7, 9], "context close invalidated retained result")
            info = result.wait()
            require(info.completed and 0 < info.gpu_start_seconds < info.gpu_end_seconds,
                    "retained result lacks GPU completion")
            lifetime_event = asdict(info)
            rejects(lambda: p.add(a, b), (p.Error,))
            rejects(lambda: p.asarray([1], context=context), (p.Error,))
    module = p.array._operator_path()
    report = {"status": "verified", "dtype": "float32", "shape_contract": "one-dimensional contiguous",
              "device": asdict(device), "python_version": sys.version, "events": events,
              "post_context_close_event": lifetime_event, "cpu_fallback": False,
              "operator_module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
              "native_library_sha256": hashlib.sha256(Path(p._library().path).read_bytes()).hexdigest(),
              "evidence_scope": "19 source-linked commands plus one actual retained-output event after context close"}
    (args.artifacts / "array-qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.artifacts / "verification.txt").write_text("Verification: PASS native Python arrays\n")
    print(f"Verification: PASS native Python arrays ({device.name}; 20 GPU events, empty operations skipped)")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Array qualification failed: {error}", file=sys.stderr)
        sys.exit(1)
