#!/usr/bin/env python3
"""Verify (and optionally fetch) the pinned GLSL/HLSL compiler source archives.

glslang (vulkan-sdk-1.4.363.0) is vendored in third_party/glslang/. The DXC
set (DirectXShaderCompiler v1.9.2607 and the exact submodule commits of that
tag) is about 29 MB and is therefore not committed: `--restore` downloads each
archive by immutable commit into third_party/dxc/ and keeps it only when its
SHA-256 equals the pinned value. CMake re-checks the same hashes and never
downloads. Upstream-generated archive bytes are not contractually stable; a
mismatch is reported, never silently accepted, and an existing file is never
overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

REPO = Path(__file__).resolve().parents[1]
HOST = "codeload." + "github.com"
# (directory, owner, project, commit, sha256, vendored_in_repository, note)
PINS = [
    ("glslang", "KhronosGroup", "glslang", "e1b562a8bed273a02f30b59b66a5d499793cede5",
     "907174a24713c6202c146f164bf81783f1fbc79c8cb821a30f18f159eb980312", True,
     "tag vulkan-sdk-1.4.363.0"),
    ("dxc", "microsoft", "DirectXShaderCompiler", "0d3ee6b551b8fa768fbf825300ebab81047ef6a8",
     "36d9383cfcb1a189efbadf181c81719ab329bd940b2f417cc5ba2d39a2ab5aea", False, "tag v1.9.2607"),
    ("dxc", "KhronosGroup", "SPIRV-Headers", "29981f65241605e08b0ede4cfeb999fe3b723c6a",
     "232899f1ad4104fb5bc377b94596c7621575eee62ad9a9e8f929b63a7dd8a7ad", False,
     "DXC v1.9.2607 external/SPIRV-Headers"),
    ("dxc", "KhronosGroup", "SPIRV-Tools", "b707790a898e44038547df54580022fc1cf89c3d",
     "05d8af89737bde57571c48dbd36714c9f520a69623e14de72c3be6b600e277d6", False,
     "DXC v1.9.2607 external/SPIRV-Tools"),
    ("dxc", "microsoft", "DirectX-Headers", "980971e835876dc0cde415e8f9bc646e64667bf7",
     "b5a4b6d8806ff7f29f19879f83d015dbe8740676d4ca0b48647a789cc7773c4e", False,
     "DXC v1.9.2607 external/DirectX-Headers"),
]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true",
                        help="re-download every pinned archive and compare hashes (writes nothing)")
    parser.add_argument("--restore", action="store_true",
                        help="download only missing archives and write them when the SHA-256 matches")
    parser.add_argument("--only", choices=["glslang", "dxc"], help="limit to one compiler")
    args = parser.parse_args()
    report, ok = {"archives": {}}, True
    for directory, owner, project, commit, expected, vendored, note in PINS:
        if args.only and args.only != directory:
            continue
        path = REPO / "third_party" / directory / f"{project}-{commit}.tar.gz"
        entry = {"commit": commit, "note": note, "expected_sha256": expected,
                 "path": str(path.relative_to(REPO)), "committed_to_repository": vendored}
        present = path.is_file()
        if present:
            entry["local_sha256"] = digest(path.read_bytes())
            entry["local_matches"] = entry["local_sha256"] == expected
            ok &= entry["local_matches"]
        else:
            entry["local_matches"] = False
            ok = False
        if args.download or (args.restore and not present):
            url = f"https://{HOST}/{owner}/{project}/tar.gz/{commit}"
            data = urllib.request.urlopen(url, timeout=600).read()
            entry.update(url=url, downloaded_sha256=digest(data),
                         downloaded_matches=digest(data) == expected)
            ok &= entry["downloaded_matches"]
            if args.restore and not present and entry["downloaded_matches"]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
                entry["restored"] = True
                entry["local_matches"] = True
        report["archives"][f"{directory}/{project}"] = entry
    report["status"] = "verified" if ok else "mismatch_or_missing"
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
