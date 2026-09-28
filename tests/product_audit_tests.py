#!/usr/bin/env python3
"""Read-only positive audit and tamper rejection using retained real GPU evidence.

No fixture fabricates a successful hardware run. --execution tests the common
auditor against an existing execution directory; --capture audits the whole
product capture and rejects corruption of a copied artifact. Neither submits GPU
work or alters the original evidence.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qualify_product", ROOT / "scripts/qualify_product.py")
Q = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(Q)


def rejects(call, label):
    try:
        call()
    except (RuntimeError, OSError, ValueError, KeyError):
        return
    raise RuntimeError(f"auditor accepted {label}")


def execution_cases(source):
    original = Q.read_json(source / "execution.json")
    recorded = {key: original[key] for key in ("paralyn_commit", "paralyn_dirty", "llvm_version")}
    count = len(original["launches"])
    Q.require(count > 0, "test input has no actual retained GPU command")
    Q.audit_execution(source, count, recorded, generated=False)
    with tempfile.TemporaryDirectory(prefix="paralyn-audit-tests-") as temporary:
        copied = Path(temporary) / "evidence"
        shutil.copytree(source, copied)
        path = copied / "execution.json"
        def bad(mutator, label):
            value = json.loads(json.dumps(original))
            mutator(value)
            Q.write_json(path, value)
            rejects(lambda: Q.audit_execution(copied, count, recorded, generated=False), label)
        bad(lambda x: x.update(cpu_fallback=True), "CPU fallback")
        bad(lambda x: x["launches"][0].update(gpu_start_seconds=0), "zero GPU timestamp")
        bad(lambda x: x["launches"][0].update(command_status="failed", error="failure"), "failed GPU command")
        bad(lambda x: x["launches"][0].update(source_file="../escaped.metal"), "escaping source path")
        bad(lambda x: x["launches"].pop(), "missing GPU record")
        bad(lambda x: x.update(paralyn_commit="incorrect-revision"), "different execution provenance")
        bad(lambda x: x["launches"][0].update(gpu_duration_valid=False), "invalid GPU duration flag")
        Q.write_json(path, original)
        shader = copied / original["launches"][0]["source_file"]
        shader.write_text("")
        rejects(lambda: Q.audit_execution(copied, count, recorded, generated=False), "empty dispatched source")
    print("Execution auditor: PASS (real positive evidence; eight corruption checks; no GPU work)")


def capture_cases(source):
    Q.audit_capture(source)
    with tempfile.TemporaryDirectory(prefix="paralyn-capture-audit-") as temporary:
        copied = Path(temporary) / "capture"
        shutil.copytree(source, copied)
        Q.audit_capture(copied)
        library = copied / "snapshot/binaries/libparalyn_native.dylib"
        with library.open("ab") as stream:
            stream.write(b"corruption")
        rejects(lambda: Q.audit_capture(copied), "modified retained native library")
    print("Product auditor: PASS (real complete capture survives relocation; corruption rejected; no GPU work)")


def main():
    parser = argparse.ArgumentParser()
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--execution", type=Path)
    choice.add_argument("--capture", type=Path)
    args = parser.parse_args()
    if args.execution:
        execution_cases(args.execution.resolve())
    else:
        capture_cases(args.capture.resolve())


if __name__ == "__main__":
    main()
