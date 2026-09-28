#!/usr/bin/env python3
"""Verify (and optionally re-fetch) the pinned SPIR-V toolchain source archives.

The archives under third_party/spirv/ are the authoritative build inputs; CMake
checks the same SHA-256 values before extracting them and never downloads.
This script lets a reviewer (a) verify the vendored bytes and (b) re-download
each archive by exact upstream commit and compare. Upstream-generated archive
bytes are not contractually stable; a download mismatch is reported, never
silently accepted, and the vendored file is never overwritten by default.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

REPO = Path(__file__).resolve().parents[1]
RELEASE = "vulkan-sdk-1.4.363.0"
PINS = {
    "SPIRV-Headers": ("496543121ce6419f23d6fa5d7194ba66c36212d2",
                      "a9bb9c48713245eacf97cc539b6f1d45405a92d8813f8b82e635f0a085ee9898"),
    "SPIRV-Tools": ("ef96ed763b43b59b33b31b362f09a02b729fa1c9",
                    "82c62146083fd558735a3171cf97cfc47903ca7d368482e87f94bd44883c0f00"),
    "SPIRV-Cross": ("f11ba9f0b21ba8fc15153d50a2a1ae31ab1cf8f7",
                    "92b0458889ed77eac9892b3be5a41e3cf716849fd90f39632e159e555abf3d03"),
}
HOST = "codeload." + "github.com"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--download", action="store_true",
                        help="re-download each pinned commit archive and compare hashes")
    parser.add_argument("--restore", action="store_true",
                        help="write a downloaded archive only when the vendored file is missing "
                             "and the download matches the pinned SHA-256")
    args = parser.parse_args()
    report, ok = {"release": RELEASE, "archives": {}}, True
    for name, (commit, expected) in PINS.items():
        path = REPO / "third_party" / "spirv" / f"{name}-{commit}.tar.gz"
        entry = {"commit": commit, "expected_sha256": expected, "path": str(path.relative_to(REPO))}
        if path.is_file():
            entry["vendored_sha256"] = digest(path.read_bytes())
            entry["vendored_matches"] = entry["vendored_sha256"] == expected
            ok &= entry["vendored_matches"]
        else:
            entry["vendored_matches"] = False
            ok = False
        if args.download or args.restore:
            url = f"https://{HOST}/KhronosGroup/{name}/tar.gz/{commit}"
            data = urllib.request.urlopen(url, timeout=120).read()
            entry.update(url=url, downloaded_sha256=digest(data),
                         downloaded_matches=digest(data) == expected)
            ok &= entry["downloaded_matches"]
            if args.restore and not path.is_file() and entry["downloaded_matches"]:
                path.write_bytes(data)
                entry["restored"] = True
        report["archives"][name] = entry
    report["status"] = "verified" if ok else "mismatch"
    print(json.dumps(report, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
