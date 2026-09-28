#!/usr/bin/env python3
"""Build a deterministic, dependency-free ctypes wheel from a verified native build.

The compiler is used only at packaging time to reproduce/verify the operator
artifact. Neither LLVM nor a compiler is included or needed by installed clients.
"""
import argparse
import ast
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile


def digest(data):
    return hashlib.sha256(data).hexdigest()


def command(*args):
    return subprocess.check_output([str(a) for a in args], text=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--module", required=True, type=Path)
    parser.add_argument("--compiler", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new wheel path or output directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    library, module, compiler = (getattr(args, k).resolve() for k in ("library", "module", "compiler"))
    source = Path(__file__).resolve().parent / "operators.cu"
    with tempfile.TemporaryDirectory(prefix="paralyn-wheel-verify-") as temporary:
        rebuilt = Path(temporary) / "operators.prk"
        result = command(compiler, "compile", source, "--output", rebuilt)
        if "Compiled 2 verified kernel(s)" not in result or rebuilt.read_bytes() != module.read_bytes():
            raise RuntimeError("operator artifact does not reproduce from the verified current source")
    arches = command("lipo", "-archs", library).split()
    if arches != ["arm64"]:
        raise RuntimeError("This wheel builder currently qualifies only a macOS arm64 native library")
    load_commands = command("otool", "-l", library)
    minimum = re.search(r"\bminos\s+(\d+)\.(\d+)", load_commands)
    if not minimum:
        raise RuntimeError("Cannot determine the native library's actual minimum macOS version")
    tag = f"py3-none-macosx_{minimum[1]}_{minimum[2]}_arm64"
    dependencies = command("otool", "-L", library).splitlines()[2:]
    for line in dependencies:
        dependency = line.strip().split(" (", 1)[0]
        if not dependency.startswith(("/usr/lib/", "/System/Library/")):
            raise RuntimeError(f"Unbundled native dependency would break isolated installation: {dependency}")
    package = Path(__file__).resolve().parent / "paralyn"
    tree = ast.parse((package / "__init__.py").read_text())
    version = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__"
                                                         for t in node.targets))
    dist = f"paralyn-{version}.dist-info"
    entries = {"paralyn/" + path.name: path.read_bytes() for path in package.glob("*.py")}
    entries["paralyn/_native/libparalyn_native.dylib"] = library.read_bytes()
    entries["paralyn/_data/operators.prk"] = module.read_bytes()
    entries["paralyn/_data/operators.cu"] = source.read_bytes()
    provenance = {"schema": 1, "abi_version": 1, "platform_tag": tag,
                  "native_library_sha256": digest(library.read_bytes()),
                  "operator_module_sha256": digest(module.read_bytes()),
                  "operator_source_sha256": digest(source.read_bytes()),
                  "compiler_version": command(compiler, "--version"),
                  "packaging_source_commit": command("git", "-C", root, "rev-parse", "HEAD"),
                  "packaging_source_dirty": bool(command("git", "-C", root, "status", "--porcelain")),
                  "native_dependencies": [line.strip() for line in dependencies]}
    entries["paralyn/_build.json"] = (json.dumps(provenance, sort_keys=True, indent=2) + "\n").encode()
    entries[f"{dist}/METADATA"] = (
        f"Metadata-Version: 2.1\nName: paralyn\nVersion: {version}\n"
        "Summary: Explicit GPU arrays and native runtime bindings\n"
        "License: Apache-2.0\nRequires-Python: >=3.9\n"
        "Project-URL: Source, https://github.com/Jakobthompson15/paralyn\n\n"
        "Contiguous one-dimensional FP32 arrays and native C ABI bindings. "
        "The bundled backend requires a physical supported Metal GPU; no CPU fallback.\n").encode()
    entries[f"{dist}/WHEEL"] = ("Wheel-Version: 1.0\nGenerator: paralyn-native-wheel\n"
                               f"Root-Is-Purelib: false\nTag: {tag}\n").encode()
    entries[f"{dist}/licenses/LICENSE"] = (root / "LICENSE").read_bytes()
    entries[f"{dist}/licenses/THIRD_PARTY_LICENSES.md"] = (root / "THIRD_PARTY_LICENSES.md").read_bytes()
    for license_path in (root / "third_party").glob("*/LICENSE*"):
        entries[f"{dist}/licenses/{license_path.parent.name}/{license_path.name}"] = license_path.read_bytes()
    record = io.StringIO(newline="")
    writer = csv.writer(record, lineterminator="\n")
    for name, data in sorted(entries.items()):
        encoded = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
        writer.writerow((name, "sha256=" + encoded, len(data)))
    writer.writerow((f"{dist}/RECORD", "", ""))
    entries[f"{dist}/RECORD"] = record.getvalue().encode()
    wheel = args.output.resolve()
    if wheel.suffix != ".whl":
        wheel.mkdir(parents=True, exist_ok=True)
        wheel = wheel / f"paralyn-{version}-{tag}.whl"
    wheel.parent.mkdir(parents=True, exist_ok=True)
    with wheel.open("xb") as stream, zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED,
                                                     compresslevel=9) as archive:
        for name, data in sorted(entries.items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    print(wheel)


if __name__ == "__main__":
    main()
