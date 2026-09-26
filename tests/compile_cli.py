#!/usr/bin/env python3
"""Exercise actual Clang import and module output without executing host code."""
import argparse
import pathlib
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paralyn", required=True)
    parser.add_argument("--source", required=True)
    args = parser.parse_args()

    def run(*arguments):
        return subprocess.run(
            [args.paralyn, *map(str, arguments)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    with tempfile.TemporaryDirectory(prefix="paralyn-module-") as temporary:
        directory = pathlib.Path(temporary)
        module = directory / "kernels.plyn"
        result = run("compile", args.source, "--output", module)
        assert result.returncode == 0, result.stdout
        assert "Compiled 2 verified kernel(s)" in result.stdout, result.stdout
        assert "ERROR: native module" not in result.stdout, result.stdout
        compiled = module.read_bytes()
        assert compiled[:20] == b"PARALYN\0\1\0\0\0\1\0\0\0\2\0\0\0"
        assert b"vector_add" in compiled and b"affine" in compiled

        result = run("compile", args.source, "--output", module)
        assert result.returncode != 0 and "never overwritten" in result.stdout, result.stdout
        assert module.read_bytes() == compiled, "compile overwrote an existing module"
        second = directory / "second.plyn"
        result = run("compile", args.source, "--output", second)
        assert result.returncode == 0, result.stdout
        assert second.read_bytes() == compiled, "compiler artifact is nondeterministic"

        # This would leave a filesystem marker if compilation ran host main.
        marker = directory / "host-ran"
        source = directory / "host_side_effect.cu"
        source.write_text(
            '#include <cuda_runtime.h>\n#include <cstdio>\n'
            '__global__ void copy(float* out) { out[threadIdx.x] = 3.0f; }\n'
            f'int main() {{ std::FILE* f = std::fopen("{marker}", "w"); '
            'if (f) { std::fputs("bad", f); std::fclose(f); } return 93; }\n'
        )
        result = run("compile", source, "--output", directory / "side_effect.plyn")
        assert result.returncode == 0, result.stdout
        assert not marker.exists(), "compile executed source host code"

        unsupported = directory / "unsupported.cu"
        unsupported.write_text(
            '#include <cuda_runtime.h>\n'
            '__global__ void bad(float* out) { out[threadIdx.x] = 3.0f / 2.0f; }\n'
        )
        missing = directory / "unsupported.plyn"
        result = run("compile", unsupported, "--output", missing)
        assert result.returncode != 0, "compile accepted unsupported division"
        assert not missing.exists(), "failed compilation left a module"

        empty = directory / "empty.cu"
        empty.write_text("int main() { return 0; }\n")
        result = run("compile", empty, "--output", directory / "empty.plyn")
        assert result.returncode != 0, "compile accepted an empty module"
        for options in [[], ["--device", "auto"], ["--output"],
                        ["--output", missing, "--output", missing]]:
            result = run("compile", args.source, *options)
            assert result.returncode != 0, f"compile accepted invalid options: {options}"
        result = run("inspect", args.source)
        assert result.returncode == 0 and "Detected kernels:" in result.stdout, result.stdout
        assert "vector_add(" in result.stdout and "affine(" in result.stdout, result.stdout
    print("CLI module compilation, non-execution, diagnostics and output protection: PASS")


if __name__ == "__main__":
    main()
