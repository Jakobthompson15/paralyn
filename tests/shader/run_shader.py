#!/usr/bin/env python3
"""CTest entry for the GLSL/HLSL compute frontends.

--mode cli:   paralyn compile/check/inspect/support/run on the examples and the
              negative fixtures (Metal compilation/reflection only; no dispatch).
--mode gpu:   compile the examples with `paralyn compile`, then run the C ABI
              driver and the Python binding test on the physical GPU with
              independent CPU references.
--mode cases: `paralyn verify` of every GLSL/HLSL kernel case in
              examples/cases (direct and through examples/cases/shaders.toml),
              plus kernel-case negative checks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

GLSL_PIN = "glslang vulkan-sdk-1.4.363.0 (e1b562a8bed273a02f30b59b66a5d499793cede5)"
DXC_PIN = "DXC v1.9.2607 (0d3ee6b551b8fa768fbf825300ebab81047ef6a8"
# (fixture, extra arguments, diagnostic id, stage, located line or None)
NEGATIVE = [
    ("compile_error.comp", [], "P-GLSL-COMPILE", "compilation", 9),
    ("include.comp", [], "P-GLSL-INCLUDE", "input", 4),
    ("include_comment.comp", [], "P-GLSL-INCLUDE", "input", 4),
    ("include_splice.comp", [], "P-GLSL-INCLUDE", "input", 4),
    ("include_extension.comp", [], "P-GLSL-INCLUDE", "input", 3),
    ("image.comp", [], "P-SPIRV-RESOURCE", "import", None),
    ("float64.comp", [], "P-SPIRV-CAPABILITY", "import", None),
    ("subgroup.comp", [], "P-SPIRV-CAPABILITY", "import", None),
    ("atomic.comp", [], "P-SPIRV-INSTRUCTION", "import", None),
    ("compile_error.hlsl", ["--entry", "main_cs", "--profile", "cs_6_0"], "P-HLSL-COMPILE", "compilation", 7),
    ("include.hlsl", ["--entry", "main_cs", "--profile", "cs_6_0"], "P-HLSL-INCLUDE", "input", 2),
    ("include_comment.hlsl", ["--entry", "main_cs", "--profile", "cs_6_0"], "P-HLSL-INCLUDE", "input", 2),
    ("include_splice.hlsl", ["--entry", "main_cs", "--profile", "cs_6_0"], "P-HLSL-INCLUDE", "input", 2),
    ("include_literal.hlsl", ["--entry", "main_cs", "--profile", "cs_6_0"], "P-HLSL-INCLUDE", "input", 3),
    ("vertex.hlsl", ["--entry", "vs_main", "--profile", "cs_6_0"], "P-HLSL-COMPILE", "compilation", 3),
    ("vertex.hlsl", ["--entry", "vs_main", "--profile", "vs_6_0"], "P-HLSL-PROFILE", "input", None),
    ("texture.hlsl", ["--entry", "main_cs", "--profile", "cs_6_0"], "P-SPIRV-RESOURCE", "import", None),
    ("float64.hlsl", ["--entry", "main_cs", "--profile", "cs_6_0"], "P-SPIRV-CAPABILITY", "import", None),
    ("int64.hlsl", ["--entry", "main_cs", "--profile", "cs_6_0"], "P-SPIRV-CAPABILITY", "import", None),
    ("wave.hlsl", ["--entry", "main_cs", "--profile", "cs_6_0"], "P-SPIRV-CAPABILITY", "import", None),
    ("atomic.hlsl", ["--entry", "main_cs", "--profile", "cs_6_0"], "P-SPIRV-INSTRUCTION", "import", None),
]
# name -> (example path relative to examples/, compile arguments, parameter names, workgroup)
EXAMPLES = {
    "glsl_vector_add": ("glsl/vector_add.comp", ["--entry", "vector_add"], ["a", "b", "c", "n"], [64, 1, 1]),
    "blur_rows": ("glsl/blur_rows.comp", ["--entry", "blur_rows"],
                  ["src", "weights", "dst", "width", "height", "radius"], [16, 16, 1]),
    "blur_columns": ("glsl/blur_columns.comp", ["--entry", "blur_columns"],
                     ["src", "weights", "dst", "width", "height", "radius"], [16, 16, 1]),
    "hlsl_vector_add": ("hlsl/vector_add.hlsl", ["--entry", "vector_add", "--profile", "cs_6_0"],
                        ["a", "b", "c", "n"], [64, 1, 1]),
    "transpose": ("hlsl/transpose.hlsl", ["--entry", "transpose_tiled", "--profile", "cs_6_0"],
                  ["src", "dst", "width", "height"], [16, 16, 1]),
    "reduce_sum": ("hlsl/reduce_sum.hlsl", ["--entry", "reduce_sum", "--profile", "cs_6_0"],
                   ["input", "partial", "n", "scale"], [256, 1, 1]),
}
CASES = ["glsl-vector-add", "glsl-blur-rows", "glsl-blur-columns", "hlsl-vector-add", "hlsl-transpose",
         "hlsl-reduce-sum"]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def cli(paralyn, *args, expect=0, env=None):
    result = subprocess.run([paralyn, *args, "--json"], capture_output=True, text=True, env=env)
    require(result.returncode == expect,
            f"{' '.join(args)} exited {result.returncode}, expected {expect}: {result.stderr}{result.stdout}")
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    require(len(lines) == 1, f"{' '.join(args)} did not emit exactly one JSON document: {result.stdout}")
    return json.loads(lines[0])




def language(path):
    return "hlsl" if str(path).endswith(".hlsl") else "glsl"


def built_examples(args):
    return {name: value for name, value in EXAMPLES.items() if language(value[0]) in args.languages}


def compile_examples(args, root):
    modules = {}
    for name, (path, extra, parameters, block) in built_examples(args).items():
        source = args.examples / path
        output = root / f"{name}.prx"
        report = cli(args.paralyn, "compile", str(source), *extra, "--output", str(output))
        hlsl = path.endswith(".hlsl")
        require(report["status"] == "compiled" and report["frontend"] == ("hlsl_compute" if hlsl else "glsl_compute"),
                f"{name}: unexpected compile report")
        shader = report["shader"]
        require((DXC_PIN if hlsl else GLSL_PIN) in shader["toolchain"] and
                "SPIRV-Tools vulkan-sdk-1.4.363.0" in report["toolchain"], f"{name}: toolchain identity")
        require("files read: source only" in shader["toolchain"] and "files read: source only" in report["toolchain"],
                f"{name}: the compiler-reported dependency check is not recorded")
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        require(f"sha256 {source_hash}" in report["toolchain"] and report["source_sha256"] == source_hash,
                f"{name}: container does not record the exact source hash")
        require((shader["normalization"] is not None) == hlsl, f"{name}: normalization record")
        require(shader["worker_arguments"][-1] == source.name and
                ("-fspv-target-env=vulkan1.1" in shader["worker_arguments"] if hlsl
                 else "vulkan1.1" in shader["worker_arguments"] and "--depfile" in shader["worker_arguments"]),
                f"{name}: worker arguments")
        e = report["entries"][0]
        require([p["name"] for p in e["parameters"]] == parameters and e["required_block"] == block,
                f"{name}: reflected descriptor {e}")
        require(output.is_file(), f"{name}: no module written")
        modules[name] = output
    return modules


def include_bypass(args, root):
    """The reviewer's end-to-end forms: a one-line HLSL wrapper whose `#/**/include`
    names an absolute path, and a GLSL `#\\<newline>include` of a file next to
    the worker's private temporary directory. Both must be refused before
    anything is compiled from the other file, and nothing is written."""
    rejected = 0
    tmp = root / "tmpdir"
    tmp.mkdir()
    environment = dict(os.environ, TMPDIR=str(tmp))
    if "hlsl" in args.languages:
        secret = root / "secret.hlsl"
        secret.write_text("[[vk::binding(0, 0)]] RWStructuredBuffer<float> hidden_buffer;\n"
                          "[numthreads(64, 1, 1)] void main_cs(uint3 id : SV_DispatchThreadID) "
                          "{ hidden_buffer[id.x] = 1; }\n")
        wrapper = root / "e2e.hlsl"
        wrapper.write_text(f'#/**/include "{secret}"\n')
        for command in ("inspect", "check", "compile"):
            extra = ["--output", str(root / "e2e.prx")] if command == "compile" else []
            report = cli(args.paralyn, command, str(wrapper), "--entry", "main_cs", "--profile", "cs_6_0", *extra,
                         expect=1, env=environment)
            d = report["diagnostic"]
            require(report["status"] == "failed" and d["id"] == "P-HLSL-INCLUDE" and "entries" not in report and
                    "hidden_buffer" not in json.dumps(report), f"HLSL #/**/include wrapper: {report}")
            rejected += 1
        require(not (root / "e2e.prx").exists(), "a refused HLSL wrapper wrote a module")
    if "glsl" in args.languages:
        (tmp.parent / "secret2.glsl").write_text("layout(set = 0, binding = 0) buffer H { float d[]; } hidden;\n")
        wrapper = root / "e2e.comp"
        wrapper.write_text('#version 450\n#extension GL_GOOGLE_include_directive : enable\n#\\\ninclude '
                           '"../secret2.glsl"\nlayout(local_size_x = 64) in;\n'
                           'void main() { hidden.d[gl_GlobalInvocationID.x] = 1.0; }\n')
        for command in ("inspect", "check", "compile"):
            extra = ["--output", str(root / "e2e-glsl.prx")] if command == "compile" else []
            report = cli(args.paralyn, command, str(wrapper), *extra, expect=1, env=environment)
            d = report["diagnostic"]
            require(report["status"] == "failed" and d["id"] == "P-GLSL-INCLUDE" and "entries" not in report and
                    '"hidden"' not in json.dumps(report), f"GLSL spliced #include: {report}")
            located = [x for x in d.get("source_diagnostics", []) if x["line"] == 3]
            require(located and located[0]["file"] == "e2e.comp", f"GLSL spliced #include not located: {d}")
            rejected += 1
        require(not (root / "e2e-glsl.prx").exists(), "a refused GLSL wrapper wrote a module")
    return rejected


def cli_mode(args, root):
    modules = compile_examples(args, root)
    for name, (path, extra, _, _) in built_examples(args).items():
        for target, flags in ((args.examples / path, extra), (modules[name], [])):
            report = cli(args.paralyn, "check", str(target), *flags)
            require(report["status"] == "checked" and report["backend_compilation"] == "passed" and
                    report["gpu_work_submitted"] is False and report["host_code_executed"] is False,
                    f"check {target} did not pass Metal compilation/reflection")
        inspected = cli(args.paralyn, "inspect", str(args.examples / path), *extra)
        require(inspected["status"] == "checked" and "backend_compilation" not in inspected,
                "inspect must not compile for a device")
    if "glsl" in args.languages:
        # GLSL without --entry keeps glslang's `main`, which SPIRV-Cross renames main0.
        plain = cli(args.paralyn, "inspect", str(args.examples / "glsl/vector_add.comp"))
        require(plain["entries"][0]["name"] == "main0", "GLSL default entry name")
    rejected = 0
    for fixture, extra, diagnostic, stage, line in NEGATIVE:
        if language(fixture) not in args.languages:
            continue
        output = root / (fixture + ".prx")
        for command in (["compile", str(args.negative / fixture), *extra, "--output", str(output)],
                        ["check", str(args.negative / fixture), *extra]):
            report = cli(args.paralyn, *command, expect=1)
            d = report["diagnostic"]
            require(report["status"] == "failed" and d["id"] == diagnostic and d["stage"] == stage,
                    f"{fixture}: expected {diagnostic}/{stage}, got {d}")
            if line is not None:
                located = [x for x in d.get("source_diagnostics", []) if x["line"] == line]
                require(located and located[0]["file"] == fixture and located[0]["severity"] == "error",
                        f"{fixture}: diagnostic not located at line {line}: {d}")
            require(not output.exists(), f"{fixture}: a failed compile wrote an output module")
            rejected += 1
    rejected += include_bypass(args, root)
    if "hlsl" in args.languages:
        add_hlsl = str(args.examples / "hlsl/vector_add.hlsl")
        for flags, diagnostic in ((["--entry", "vector_add"], "P-HLSL-PROFILE"),
                                  (["--entry", "vector_add", "--profile", "cs_5_0"], "P-HLSL-PROFILE"),
                                  (["--entry", "vector_add", "--profile", "ps_6_0"], "P-HLSL-PROFILE"),
                                  (["--profile", "cs_6_0"], "P-HLSL-ENTRY"),
                                  (["--entry", "missing", "--profile", "cs_6_0"], "P-HLSL-COMPILE")):
            report = cli(args.paralyn, "inspect", add_hlsl, *flags, expect=1)
            require(report["diagnostic"]["id"] == diagnostic,
                    f"{flags}: expected {diagnostic}, got {report['diagnostic']}")
            rejected += 1
    report = cli(args.paralyn, "inspect", str(args.examples / "glsl/vector_add.comp"), "--profile", "cs_6_0", expect=1)
    require(report["diagnostic"]["id"] == "P-PROFILE-NOT-APPLICABLE", "GLSL --profile must be rejected")
    any_module = next(iter(modules.values()))
    report = cli(args.paralyn, "inspect", str(any_module), "--profile", "cs_6_0", expect=1)
    require(report["diagnostic"]["id"] == "P-PROFILE-NOT-APPLICABLE", ".prx --profile must be rejected")
    rejected += 2
    for target, flags in ((args.examples / "glsl/vector_add.comp", ["--entry", "vector_add"]),
                          (args.examples / "hlsl/vector_add.hlsl", ["--entry", "vector_add", "--profile", "cs_6_0"])):
        if language(target) not in args.languages:
            continue
        for command in ("run", "verify"):
            report = cli(args.paralyn, command, str(target), *flags, expect=1)
            require(report["diagnostic"]["id"] == "P-KERNEL-CASE-REQUIRED",
                    f"{command} {target.name} without a case: {report['diagnostic']}")
            rejected += 1
    support = {r["name"]: r for r in cli(args.paralyn, "support")["support"]["inputs"]}
    for row, lang in (("GLSL compute", "glsl"), ("HLSL compute", "hlsl")):
        if lang in args.languages:
            require(support[row]["implementation"] == "partial" and support[row]["full_profile_qualified"] is False and
                    "no CUDA/HIP" in support[row]["scope"], f"support row for {row} is wrong")
        else:
            require(support[row]["implementation"] == "not_implemented" and
                    support[row]["available_in_this_build"] is False, f"support row for unbuilt {row} is wrong")
    # A frontend that is not in this build is reported as unavailable, not attempted.
    for lang, target, flags in (("glsl", "glsl/vector_add.comp", ["--entry", "vector_add"]),
                                ("hlsl", "hlsl/vector_add.hlsl", ["--entry", "vector_add", "--profile", "cs_6_0"])):
        if lang in args.languages:
            continue
        report = cli(args.paralyn, "inspect", str(args.examples / target), *flags, expect=1)
        require(report["diagnostic"]["id"] == f"P-{lang.upper()}-UNAVAILABLE",
                f"unbuilt {lang} frontend: {report['diagnostic']}")
        rejected += 1
    print(f"GLSL/HLSL CLI ({','.join(sorted(args.languages))}): {len(modules)} example compilations, "
          f"{rejected} stable diagnostics, no GPU dispatch")


def gpu_mode(args, root):
    modules = compile_examples(args, root)
    evidence = root / "evidence"
    order = ["glsl_vector_add", "blur_rows", "blur_columns", "hlsl_vector_add", "transpose", "reduce_sum"]
    languages = ",".join(lang for lang in ("glsl", "hlsl") if lang in args.languages)
    subprocess.run([args.driver, languages, *[str(modules[m]) if m in modules else "-" for m in order],
                    str(evidence)], check=True)
    require((evidence / "execution.json").is_file(), "driver did not export runtime evidence")
    environment = dict(os.environ, PARALYN_LIBRARY=str(args.library),
                       PYTHONPATH=str(args.python_path) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    python_args = []
    if "glsl" in args.languages:
        python_args += ["--glsl-blur-rows", str(modules["blur_rows"]),
                        "--glsl-blur-columns", str(modules["blur_columns"])]
    if "hlsl" in args.languages:
        python_args += ["--hlsl-vector-add", str(modules["hlsl_vector_add"])]
    subprocess.run([sys.executable, str(Path(__file__).with_name("test_python.py")), *python_args],
                   check=True, env=environment)


def cases_mode(args, root):
    cases = args.examples / "cases"
    project = cases / "shaders.toml"
    glsl, hlsl = "glsl" in args.languages, "hlsl" in args.languages
    passed = 0
    for index, name in enumerate(CASES):
        if name.split("-")[0] not in args.languages:
            continue
        report = cli(args.paralyn, "verify", "--project", str(project), "--case", name,
                     "--artifacts", str(root / f"project-{index}"))
        require(report["status"] == "passed" and report["verification"]["status"] == "passed" and
                report["project"]["case"] == name, f"{name}: {report.get('diagnostic')}")
        require((root / f"project-{index}" / "module.prx").is_file() and
                any((root / f"project-{index}").glob("source.*")), f"{name}: evidence lacks module/source")
        passed += 1
    # Direct (non-project) selection with --entry/--profile.
    direct = [("glsl/vector_add.comp", ["--entry", "vector_add"], "glsl_vector_add.toml"),
              ("glsl/blur_rows.comp", ["--entry", "blur_rows"], "glsl_blur_rows.toml"),
              ("hlsl/transpose.hlsl", ["--entry", "transpose_tiled", "--profile", "cs_6_0"], "hlsl_transpose.toml"),
              ("hlsl/reduce_sum.hlsl", ["--entry", "reduce_sum", "--profile", "cs_6_0"], "hlsl_reduce_sum.toml")]
    for index, (source, flags, case) in enumerate(direct):
        if language(source) not in args.languages:
            continue
        report = cli(args.paralyn, "verify", str(args.examples / source), *flags, "--case", str(cases / case),
                     "--artifacts", str(root / f"direct-{index}"))
        require(report["status"] == "passed" and report["frontend_module"] in ("glsl_compute", "hlsl_compute"),
                f"{source}: {report.get('diagnostic')}")
        passed += 1
    # `check --case` binds without dispatch; `run` executes without verifying.
    report = cli(args.paralyn, "check", "--project", str(project), "--case",
                 "hlsl-reduce-sum" if hlsl else "glsl-blur-rows")
    require(report["status"] == "checked" and report.get("gpu_work_submitted") is False, "check --case")
    report = cli(args.paralyn, "run", "--project", str(project), "--case",
                 "glsl-blur-columns" if glsl else "hlsl-vector-add", "--artifacts", str(root / "run"))
    require(report["verification"]["status"] == "not_requested", "run must not verify")
    rejected = 0
    # A mismatching --entry, a conflicting --profile, and a wrong-entry case are refused.
    if hlsl:
        report = cli(args.paralyn, "verify", "--project", str(project), "--case", "hlsl-transpose",
                     "--entry", "other", expect=1)
        require(report["diagnostic"]["id"] == "P-CASE-ENTRY-MISMATCH", f"entry mismatch: {report['diagnostic']}")
        report = cli(args.paralyn, "verify", "--project", str(project), "--case", "hlsl-transpose",
                     "--profile", "cs_6_6", expect=1)
        require(report["diagnostic"]["id"] == "P-TARGET-AMBIGUOUS", f"profile conflict: {report['diagnostic']}")
        report = cli(args.paralyn, "verify", str(args.examples / "hlsl/vector_add.hlsl"), "--entry", "vector_add",
                     "--case", str(cases / "hlsl_vector_add.toml"), expect=1)
        require(report["diagnostic"]["id"] == "P-HLSL-PROFILE", f"HLSL case without profile: {report['diagnostic']}")
        rejected += 3
    else:
        # The project still names HLSL modules; selecting one reports the frontend as unavailable.
        report = cli(args.paralyn, "verify", "--project", str(project), "--case", "hlsl-transpose", expect=1)
        require(report["diagnostic"]["id"] == "P-HLSL-UNAVAILABLE", f"unbuilt HLSL case: {report['diagnostic']}")
        rejected += 1
    if glsl:
        report = cli(args.paralyn, "verify", str(args.examples / "glsl/vector_add.comp"), "--case",
                     str(cases / "glsl_vector_add.toml"), expect=1)
        require(report["diagnostic"]["id"] == "P-CASE-ENTRY-UNKNOWN",
                f"GLSL without --entry exposes main0, not vector_add: {report['diagnostic']}")
        report = cli(args.paralyn, "verify", "--project", str(project), "--case", "glsl-blur-rows",
                     "--entry", "other", expect=1)
        require(report["diagnostic"]["id"] == "P-CASE-ENTRY-MISMATCH", f"GLSL entry mismatch: {report['diagnostic']}")
        rejected += 2
    else:
        report = cli(args.paralyn, "verify", "--project", str(project), "--case", "glsl-blur-rows", expect=1)
        require(report["diagnostic"]["id"] == "P-GLSL-UNAVAILABLE", f"unbuilt GLSL case: {report['diagnostic']}")
        rejected += 1
    # Project schema: entry/profile only where they apply; HLSL needs both.
    ex = args.examples.resolve()
    bad = {
        "metal-entry": f'[modules.m]\nsource = "{ex}/metal/kernels.metal"\nmanifest = "{ex}/metal/kernels.json"\nentry = "x"\n',
        "glsl-profile": f'[modules.m]\nsource = "{ex}/glsl/vector_add.comp"\nprofile = "cs_6_0"\n',
        "hlsl-no-profile": f'[modules.m]\nsource = "{ex}/hlsl/vector_add.hlsl"\nentry = "vector_add"\n',
    }
    expected = {"metal-entry": "P-PROJECT-UNKNOWN-FIELD", "glsl-profile": "P-PROJECT-UNKNOWN-FIELD",
                "hlsl-no-profile": "P-PROJECT-MISSING-FIELD"}
    for label, module in bad.items():
        path = root / f"{label}.toml"
        path.write_text('schema = "paralyn.project"\nschema_version = 1\n[project]\nname = "x"\n' + module +
                        f'[cases.c]\nmodule = "m"\nfile = "{cases.resolve()}/glsl_vector_add.toml"\n')
        report = cli(args.paralyn, "check", "--project", str(path), "--case", "c", expect=1)
        require(report["diagnostic"]["id"] == expected[label], f"{label}: {report['diagnostic']}")
        rejected += 1
    print(f"GLSL/HLSL kernel cases ({','.join(sorted(args.languages))}): {passed} verified on the GPU, "
          f"{rejected} refusals")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("cli", "gpu", "cases"), required=True)
    parser.add_argument("--paralyn", required=True)
    parser.add_argument("--examples", required=True, type=Path)
    parser.add_argument("--negative", required=True, type=Path)
    parser.add_argument("--languages", default="glsl,hlsl",
                        help="comma-separated frontends in this build (glsl, hlsl); the others must be unavailable")
    parser.add_argument("--driver")
    parser.add_argument("--library", type=Path)
    parser.add_argument("--python-path", type=Path)
    parser.add_argument("--keep", type=Path, help="retain modules/evidence in this new directory")
    args = parser.parse_args()
    args.languages = set(filter(None, args.languages.split(",")))
    require(args.languages and args.languages <= {"glsl", "hlsl"}, f"bad --languages {args.languages}")
    run = {"cli": cli_mode, "gpu": gpu_mode, "cases": cases_mode}[args.mode]
    if args.keep:
        args.keep.mkdir(parents=True, exist_ok=False)
        run(args, args.keep.resolve())
        return
    with tempfile.TemporaryDirectory(prefix=f"paralyn-shader-{args.mode}-") as temporary:
        run(args, Path(temporary).resolve())


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, KeyError, subprocess.SubprocessError) as error:
        print(f"GLSL/HLSL test failed: {error}", file=sys.stderr)
        sys.exit(1)
