#!/usr/bin/env python3
"""CTest entry point with fresh retained native evidence and independent auditing."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
from qualify_native import (KINDS, gather_provenance, require, run_case,
                            run_finalizer_policy, sha256, verify_unchanged)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=KINDS + ("python-finalizer",))
    parser.add_argument("--build", required=True, type=Path)
    parser.add_argument("--module", type=Path)
    parser.add_argument("--artifacts-root", type=Path)
    args = parser.parse_args()
    build = args.build.resolve()
    module = (args.module or build / "native-kernels.prk").resolve()
    root = (args.artifacts_root or build / "test-artifacts").resolve()
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=args.kind + "-", dir=root))
    provenance, environment = gather_provenance(REPO, build, require_all=False)
    report = {"status": "running", "provenance": provenance}
    try:
        if args.kind == "python-finalizer":
            result = run_finalizer_policy(REPO, build, directory / "case", environment)
        else:
            require(module.is_file(), "native kernel module must be built before CTest")
            before = sha256(module)
            result = run_case(args.kind, REPO, build, module, directory / "case", provenance, environment)
            require(sha256(module) == before, "module changed during CTest")
            report["module_sha256"] = before
        verify_unchanged(REPO, build, provenance, directory)
        report.update(status="verified", result=result)
        print(f"{args.kind}: PASS; retained evidence: {directory}")
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        (directory / "capture.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"Native CTest failed: {error}", file=sys.stderr)
        sys.exit(1)
