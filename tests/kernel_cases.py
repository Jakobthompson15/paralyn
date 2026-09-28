#!/usr/bin/env python3
"""Kernel cases and paralyn.toml projects: real CLI processes, real GPU dispatch,
independent re-audit of every saved output, and strict negative schema checks.

Nothing here trusts the CLI's own PASS: each verified output file is re-read and
compared in Python against the shipped reference bytes or an independent
Python recomputation, and runtime evidence must show a completed Metal command.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import tempfile


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def floats(path):
    data = Path(path).read_bytes()
    return list(struct.unpack(f"<{len(data) // 4}f", data))


class Suite:
    def __init__(self, paralyn, source, prk, work):
        self.paralyn, self.source, self.prk, self.work = paralyn, source, prk, work
        self.cases = source / "examples/cases"
        self.metal = source / "examples/metal/kernels.metal"
        self.manifest = source / "examples/metal/kernels.json"
        self.negative = 0
        self.gpu_events = 0
        self.environment = os.environ.copy()
        for key in ("PARALYN_RUNTIME_LOG", "PARALYN_ARTIFACT_DIR", "PARALYN_EVENT_LOG",
                    "PARALYN_COMMIT", "PARALYN_SOURCE_DIRTY"):
            self.environment.pop(key, None)
        self.counter = 0

    def call(self, *arguments, cwd=None):
        return subprocess.run([str(self.paralyn), *map(str, arguments)], cwd=cwd or self.work,
                              env=self.environment, text=True, capture_output=True, timeout=120,
                              check=False)

    def json(self, *arguments, code=0, cwd=None):
        arguments = list(arguments)
        # Everything after "--" belongs to the application, so --json goes before it.
        split = arguments.index("--") if "--" in arguments else len(arguments)
        result = self.call(*arguments[:split], "--json", *arguments[split:], cwd=cwd)
        assert result.returncode == code, (arguments, result.returncode, result.stdout, result.stderr)
        value = json.loads(result.stdout)
        assert value["schema"] == "paralyn.report" and value["schema_version"] == 1, value
        return value

    def fails(self, error_id, *arguments, cwd=None):
        """A rejected request: stable id, failed status, and no GPU evidence directory."""
        runs = (cwd or self.work) / ".paralyn"
        before = sorted(runs.rglob("*")) if runs.exists() else []
        value = self.json(*arguments, code=1, cwd=cwd)
        assert value["status"] == "failed", value
        assert value["diagnostic"]["id"] == error_id, (error_id, arguments, value["diagnostic"])
        assert value["diagnostic"]["message"], value
        after = sorted(runs.rglob("*")) if runs.exists() else []
        assert before == after, f"rejected request created evidence: {arguments}"
        self.negative += 1
        return value

    def artifacts(self):
        self.counter += 1
        return self.work / f"evidence-{self.counter}"

    def case(self, name, text):
        path = self.work / name
        path.write_text(text, encoding="utf-8")
        return path

    def base(self):
        return (self.cases / "vector_add_builtin.toml").read_text(encoding="utf-8")

    def gpu_report(self, value, entry, verified):
        assert value["cpu_fallback"] is False and value["frontend"] == "kernel_case", value
        assert value["case"]["entry"] == entry and re.fullmatch(r"[0-9a-f]{64}", value["case"]["sha256"])
        build = value["build"]
        assert re.fullmatch(r"[0-9a-f]{40}", build["revision"]) and type(build["dirty"]) is bool, build
        assert value["source_revision"] == build["revision"] and build["captured"] == "build"
        timing = value["timing"]
        assert timing["completed"] and timing["duration_valid"] and timing["duration_seconds"] > 0, timing
        evidence = json.loads(Path(value["runtime_evidence_path"]).read_text())
        assert evidence["cpu_fallback"] is False and evidence["paralyn_runtime_build"]["revision"] == build["revision"]
        assert evidence["paralyn_commit"] == build["revision"], evidence
        launches = evidence["launches"]
        assert len(launches) == 1 and launches[0]["kernel"] == entry, launches
        assert launches[0]["command_status"] == "completed" and launches[0]["gpu_duration_valid"], launches
        assert launches[0]["grid"] == value["launch"]["grid"] and launches[0]["block"] == value["launch"]["block"]
        events = [json.loads(line) for line in Path(value["runtime_events_path"]).read_text().splitlines() if line]
        assert [e["detail"] for e in events if e["category"] == "completion" and e["status"] == "completed"] == [entry], events
        assert not any(e["status"] == "failed" for e in events), events
        for output in value["outputs"]:
            data = Path(output["path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == output["sha256"], output
            assert len(data) == output["elements"] * 4
        # The evidence copy is exactly the bytes whose digest the report records.
        copied = Path(value["artifacts"], "case.toml").read_bytes()
        assert hashlib.sha256(copied).hexdigest() == value["case"]["sha256"], value["case"]
        assert value["verification"]["status"] == ("passed" if verified else "not_requested"), value
        self.gpu_events += 1

    # ---------------------------------------------------------------- negative
    def missing_case(self):
        prx = self.work / "kernels.prx"
        compiled = self.json("compile", self.metal, "--manifest", self.manifest, "--output", prx)
        assert compiled["status"] == "compiled"
        for command in ("run", "verify"):
            for target in ([prx], [self.metal, "--manifest", self.manifest], [self.prk]):
                value = self.fails("P-KERNEL-CASE-REQUIRED", command, *target, "--device", "metal:0")
                message = value["diagnostic"]["message"]
                assert "--case CASE.toml" in message and "--entry NAME" in message, message
        self.fails("P-REFERENCE-REQUIRED", "verify", self.source / "examples/vector_add.cu")
        self.fails("P-CASE-NOT-APPLICABLE", "run", self.source / "examples/vector_add.cu",
                   "--case", self.cases / "vector_add.toml")
        self.fails("P-CASE-FILE", "run", prx, "--entry", "vector_add", "--case", self.work / "absent.toml")
        self.fails("P-CASE-ARGUMENTS", "run", prx, "--case", self.cases / "vector_add.toml", "--", "x")
        self.prx = prx
        # Selection flags that do not apply to the target are rejected, never ignored.
        cu = self.source / "examples/vector_add.cu"
        app = self.work / "app.py"
        app.write_text("print('never executed')\n")
        self.fails("P-CASE-NOT-APPLICABLE", "run", cu, "--entry", "vector_add")
        self.fails("P-CASE-NOT-APPLICABLE", "run", app, "--entry", "main")
        self.fails("P-CASE-NOT-APPLICABLE", "check", prx, "--entry", "vector_add", "--device", "metal:0")
        self.fails("P-CASE-NOT-APPLICABLE", "run", "--project", self.cases / "paralyn.toml",
                   "--program", "cuda-vector-add", "--entry", "vector_add")
        self.fails("P-MANIFEST-NOT-APPLICABLE", "run", cu, "--manifest", self.manifest)
        self.fails("P-MANIFEST-NOT-APPLICABLE", "run", app, "--manifest", self.manifest)
        self.fails("P-MANIFEST-NOT-APPLICABLE", "inspect", cu, "--manifest", self.manifest)
        self.fails("P-MANIFEST-NOT-APPLICABLE", "run", prx, "--manifest", self.manifest,
                   "--case", self.cases / "vector_add.toml")
        self.fails("P-MANIFEST-NOT-APPLICABLE", "verify", self.prk, "--manifest", self.manifest,
                   "--case", self.cases / "ir_affine.toml")

    def malformed_cases(self):
        base = self.base()
        good = self.case("good.toml", base)
        run = lambda error, path, *extra: self.fails(error, "run", self.prx, "--case", path, *extra)
        run("P-CASE-SYNTAX", self.case("syntax.toml", base + "\n[launch\n"))
        run("P-CASE-SYNTAX", self.case("duplicate.toml", base.replace("[scalars.n]", "[scalars.n]\ntype = \"u32\"", 1)))
        run("P-CASE-UNKNOWN-FIELD", self.case("unknown-top.toml", "inputs = 3\n" + base))
        run("P-CASE-UNKNOWN-FIELD", self.case("unknown-nested.toml", base.replace('dtype = "f32"\nlength = 7', 'dtype = "f32"\nlength = 7\ncanary = true', 1)))
        run("P-CASE-UNKNOWN-FIELD", self.case("unknown-launch.toml", base.replace("block = [64, 1, 1]", "block = [64, 1, 1]\nshared = 0")))
        run("P-CASE-MISSING-FIELD", self.case("no-launch.toml", base.replace("[launch]\ngrid = [1, 1, 1]\nblock = [64, 1, 1]\n", "")))
        run("P-CASE-MISSING-FIELD", self.case("no-block.toml", base.replace("block = [64, 1, 1]\n", "")))
        run("P-CASE-MISSING-FIELD", self.case("no-schema.toml", base.replace('schema = "paralyn.kernel-case"\n', "")))
        run("P-CASE-VERSION", self.case("version.toml", base.replace("schema_version = 1", "schema_version = 2")))
        run("P-CASE-SCHEMA", self.case("schema.toml", base.replace("paralyn.kernel-case", "paralyn.project")))
        run("P-CASE-TYPE", self.case("grid-shape.toml", base.replace("grid = [1, 1, 1]", "grid = [1, 1]")))
        run("P-CASE-VALUE", self.case("grid-zero.toml", base.replace("grid = [1, 1, 1]", "grid = [0, 1, 1]")))
        run("P-CASE-TYPE", self.case("grid-float.toml", base.replace("grid = [1, 1, 1]", "grid = [1.0, 1, 1]")))
        run("P-CASE-DATA-SOURCE", self.case("no-data.toml", base.replace("fill = -65536.0\n", "")))
        run("P-CASE-DATA-SOURCE", self.case("two-sources.toml", base.replace("fill = -65536.0", "fill = -65536.0\nvalues = [1, 2, 3, 4, 5, 6, 7]")))
        run("P-CASE-DATA-SIZE", self.case("short-values.toml", base.replace("[1.5, -2.0, 0.25, 1024.0, -0.0]", "[1.5, -2.0]")))
        run("P-CASE-VALUE", self.case("inexact.toml", base.replace("0.25, 1024.0", "0.1, 1024.0")))
        run("P-CASE-VALUE", self.case("negative-u32.toml", base.replace("value = 5", "value = -5")))
        run("P-CASE-TYPE", self.case("float-u32.toml", base.replace("value = 5", "value = 5.0")))
        run("P-CASE-VALUE", self.case("dtype.toml", base.replace('[buffers.a]\ndtype = "f32"', '[buffers.a]\ndtype = "f64"')))
        run("P-CASE-OUTPUT-REQUIRED", self.case("no-output.toml", base.replace("output = true\n", "")))
        run("P-CASE-DUPLICATE-ARGUMENT", self.case("dup-arg.toml", base.replace("[buffers.a]", '[scalars.a]\ntype = "u32"\nvalue = 1\n\n[buffers.a]')))
        # Hashed data files.
        data = self.cases / "data/vector_add_a.f32"
        text = (self.cases / "vector_add.toml").read_text()
        wrong_hash = text.replace("daed10ac0acd21c0624c83ee0373c0827d04d991260ff0e7b6d5bdd3799aea33", "0" * 64)
        local = self.work / "data"
        local.mkdir(exist_ok=True)
        for name in ("vector_add_a.f32", "vector_add_b.f32", "vector_add_expected.f32"):
            shutil.copy(self.cases / "data" / name, local / name)
        run("P-CASE-DATA-HASH", self.case("hash.toml", wrong_hash))
        run("P-CASE-MISSING-FIELD", self.case("no-hash.toml", re.sub(r'file = "data/vector_add_a.f32"\nsha256 = "[0-9a-f]+"', 'file = "data/vector_add_a.f32"', text)))
        run("P-CASE-VALUE", self.case("bad-hash.toml", text.replace("daed10ac0acd21c0624c83ee0373c0827d04d991260ff0e7b6d5bdd3799aea33", "DAED")))
        run("P-CASE-DATA-SIZE", self.case("length.toml", text.replace("length = 1003\nfile = \"data/vector_add_a.f32\"", "length = 1002\nfile = \"data/vector_add_a.f32\"")))
        run("P-CASE-DATA-FILE", self.case("missing-data.toml", text.replace("data/vector_add_a.f32", "data/absent.f32")))
        assert data.read_bytes() == (local / "vector_add_a.f32").read_bytes()
        # Verification contracts: explicit reference and tolerance, strict roles.
        run("P-CASE-MISSING-FIELD", self.case("no-tolerance.toml", base.replace('tolerance = { kind = "exact" }\n', "")))
        run("P-CASE-VALUE", self.case("tolerance-kind.toml", base.replace('kind = "exact"', 'kind = "close"')))
        run("P-CASE-UNKNOWN-FIELD", self.case("tolerance-field.toml", base.replace('kind = "exact"', 'kind = "exact", ulp = 1')))
        run("P-CASE-VALUE", self.case("tolerance-negative.toml", base.replace('kind = "exact"', 'kind = "absolute_relative", absolute = -1.0, relative = 0.0')))
        run("P-REFERENCE-BUILTIN-UNKNOWN", self.case("builtin.toml", base.replace("vector-add-f32", "vector-mul-f32")))
        run("P-CASE-MISSING-FIELD", self.case("role.toml", base.replace(', count = "n"', "")))
        run("P-CASE-VERIFY-ROLE", self.case("role-type.toml", base.replace('count = "n"', 'count = "a"')))
        # Builtin shape contracts are static: rejected by run as well as verify and check,
        # before any dispatch could read past the declared a/b allocations.
        oversized = self.case("shape.toml", base.replace("value = 5", "value = 8"))
        assert oversized.read_text() != base
        run("P-REFERENCE-SHAPE", oversized)
        self.fails("P-REFERENCE-SHAPE", "verify", self.prx, "--case", oversized)
        self.fails("P-REFERENCE-SHAPE", "check", self.prx, "--case", oversized, "--device", "metal:0")
        shutil.copy(self.cases / "data/transpose_input.f32", local / "transpose_input.f32")
        transpose = (self.cases / "transpose_builtin.toml").read_text()
        wide = self.case("shape-transpose.toml", re.sub(r"(\[scalars\.width\][^\[]*value = )(\d+)",
                                                         lambda m: m.group(1) + str(int(m.group(2)) + 1), transpose))
        assert wide.read_text() != transpose
        self.fails("P-REFERENCE-SHAPE", "run", self.prx, "--case", wide)
        run("P-CASE-UNKNOWN-FIELD", self.case("role-extra.toml", base.replace('count = "n"', 'count = "n", scale = "n"')))
        run("P-CASE-VERIFY-BUFFER", self.case("verify-input.toml", base.replace('buffer = "out"', 'buffer = "a"')))
        run("P-CASE-DATA-SOURCE", self.case("reference-empty.toml", base.replace('{ builtin = "vector-add-f32", lhs = "a", rhs = "b", count = "n" }', "{}")))
        run("P-CASE-TYPE", self.case("verify-table.toml", base.replace("[[verify]]", "[verify]")))
        # Entry selection is never guessed, even for a single-entry case.
        no_entry = self.case("no-entry.toml", base.replace('entry = "vector_add"\n', ""))
        run("P-CASE-ENTRY-REQUIRED", no_entry)
        run("P-CASE-ENTRY-MISMATCH", good, "--entry", "block_reduce")
        run("P-CASE-ENTRY-UNKNOWN", no_entry, "--entry", "vector_sub")
        # Binding against actual reflected parameters (module load, no dispatch).
        run("P-CASE-ARGUMENT-MISSING", self.case("arg-missing.toml", base.replace('[scalars.n]\ntype = "u32"\nvalue = 5\n', "").replace(', count = "n"', "").replace("vector-add-f32", "vector-add-f32").replace('reference = { builtin = "vector-add-f32", lhs = "a", rhs = "b" }', 'reference = { values = [0, 0, 0, 0, 0, 0, 0] }')))
        run("P-CASE-ARGUMENT-TYPE", self.case("arg-type.toml", base.replace('type = "u32"\nvalue = 5', 'type = "i32"\nvalue = 5').replace('count = "n"', 'count = "n"')))
        run("P-CASE-ARGUMENT-UNKNOWN", self.case("arg-extra.toml", base.replace("[buffers.out]", '[buffers.extra]\ndtype = "f32"\nlength = 1\nfill = 0\n\n[buffers.out]')))
        run("P-CASE-OUTPUT-ACCESS", self.case("output-read.toml", base.replace('[buffers.a]\ndtype = "f32"\nlength = 5', '[buffers.a]\ndtype = "f32"\noutput = true\nlength = 5')))
        self.fails("P-REFERENCE-REQUIRED", "verify", self.prx, "--case",
                   self.case("no-verify.toml", base.split("[[verify]]")[0]))

    def two_outputs(self):
        """verify must cover every declared output; a partial contract never prints PASS."""
        metal = self.work / "two_outputs.metal"
        metal.write_text(
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void two_outputs(device const float* a [[buffer(0)]],\n"
            "                        device float* x [[buffer(1)]],\n"
            "                        device float* y [[buffer(2)]],\n"
            "                        constant uint& n [[buffer(3)]],\n"
            "                        uint i [[thread_position_in_grid]]) {\n"
            "  if (i < n) { x[i] = a[i] + 1.0f; y[i] = a[i] * 2.0f; }\n}\n")

        def parameter(name, kind, access, binding):
            return {"name": name, "type": kind, "buffer": kind == "f32", "access": access,
                    "binding": binding, "alignment": 4, "minimum_bytes": 4}
        manifest = self.work / "two_outputs.json"
        manifest.write_text(json.dumps({
            "target": "metal-msl3.1", "numerical_policy": 1,
            "entries": [{"name": "two_outputs", "required_block": [0, 0, 0], "parameters": [
                parameter("a", "f32", "read", 0), parameter("x", "f32", "read_write", 1),
                parameter("y", "f32", "read_write", 2), parameter("n", "u32", "read", 3)]}]}))
        body = ('schema = "paralyn.kernel-case"\nschema_version = 1\n'
                '[case]\nname = "two"\nentry = "two_outputs"\n'
                '[launch]\ngrid = [1, 1, 1]\nblock = [4, 1, 1]\n'
                '[scalars.n]\ntype = "u32"\nvalue = 4\n'
                '[buffers.a]\ndtype = "f32"\nlength = 4\nvalues = [1.0, 2.0, 3.0, 4.0]\n'
                '[buffers.x]\ndtype = "f32"\nlength = 4\nfill = 0.0\noutput = true\n'
                '[buffers.y]\ndtype = "f32"\nlength = 4\nfill = 0.0\noutput = true\n'
                '[[verify]]\nbuffer = "x"\nreference = { values = [2.0, 3.0, 4.0, 5.0] }\n'
                'tolerance = { kind = "exact" }\n')
        partial = self.case("two-partial.toml", body)
        target = (metal, "--manifest", manifest)
        value = self.fails("P-CASE-VERIFY-UNCOVERED", "verify", *target, "--case", partial, "--device", "metal:0")
        assert "output buffer(s) y " in value["diagnostic"]["message"], value
        # run performs no comparison, so a partial contract is still executable there.
        ran = self.json("run", *target, "--case", partial, "--device", "metal:0", "--artifacts", self.artifacts())
        self.gpu_report(ran, "two_outputs", False)
        # With both outputs covered, a wrong y reference must fail.
        wrong = self.case("two-wrong.toml", body + '[[verify]]\nbuffer = "y"\n'
                          'reference = { values = [2.0, 4.0, 6.0, 12345.0] }\ntolerance = { kind = "exact" }\n')
        destination = self.artifacts()
        result = self.call("verify", *target, "--case", wrong, "--device", "metal:0", "--artifacts", destination)
        assert result.returncode == 1 and "PASS" not in result.stdout, (result.stdout, result.stderr)
        assert json.loads((destination / "report.json").read_text())["diagnostic"]["id"] == "P-VERIFY-MISMATCH"
        self.gpu_events += 1
        full = self.case("two-full.toml", body + '[[verify]]\nbuffer = "y"\n'
                         'reference = { values = [2.0, 4.0, 6.0, 8.0] }\ntolerance = { kind = "exact" }\n')
        value = self.json("verify", *target, "--case", full, "--device", "metal:0", "--artifacts", self.artifacts())
        self.gpu_report(value, "two_outputs", True)
        assert value["verification"]["compared"] == 8 and len(value["verification"]["checks"]) == 2, value
        outputs = {o["name"]: floats(o["path"]) for o in value["outputs"]}
        assert outputs == {"x": [2.0, 3.0, 4.0, 5.0], "y": [2.0, 4.0, 6.0, 8.0]}, outputs

    def malformed_projects(self):
        project = self.cases / "paralyn.toml"
        text = project.read_text()
        self.fails("P-PROJECT-SELECTION-AMBIGUOUS", "run", "--project", project)
        self.fails("P-PROJECT-SELECTION-AMBIGUOUS", "run", project)
        self.fails("P-PROJECT-SELECTION-UNKNOWN", "run", "--project", project, "--case", "vector-sub")
        self.fails("P-PROJECT-SELECTION-UNKNOWN", "run", "--project", project, "--program", "vector-add")
        self.fails("P-PROJECT-SELECTION-AMBIGUOUS", "run", "--project", project, "--case", "vector-add",
                   "--program", "cuda-vector-add")
        self.fails("P-TARGET-AMBIGUOUS", "run", self.prx, "--project", project, "--case", "vector-add")
        self.fails("P-TARGET-AMBIGUOUS", "run", "--project", project, "--case", "vector-add",
                   "--manifest", self.manifest)
        # verify selects cases only; a program name is not a case.
        self.fails("P-PROJECT-SELECTION-UNKNOWN", "verify", "--project", project, "--case", "cuda-vector-add")
        empty = self.work / "empty-dir"
        empty.mkdir()
        self.fails("P-TARGET-REQUIRED", "run", cwd=empty)
        self.fails("P-PROJECT-REQUIRED", "run", self.prx, "--program", "x")
        root = self.work / "projects"
        root.mkdir()

        def project_file(name, body):
            directory = root / name
            directory.mkdir()
            (directory / "paralyn.toml").write_text(body)
            return directory / "paralyn.toml"
        rel = os.path.relpath(self.cases, root / "x")
        header = 'schema = "paralyn.project"\nschema_version = 1\n[project]\nname = "p"\n'
        module = f'[modules.k]\nsource = "{rel}/../metal/kernels.metal"\nmanifest = "{rel}/../metal/kernels.json"\n'
        self.fails("P-PROJECT-UNKNOWN-FIELD", "run", project_file("unknown", header + "jobs = 2\n" + module))
        self.fails("P-PROJECT-UNKNOWN-FIELD", "run", project_file("unknown-case", header + module + f'[cases.a]\nmodule = "k"\nfile = "{rel}/vector_add.toml"\ndevice = "metal:0"\n'))
        self.fails("P-PROJECT-MISSING-FIELD", "run", project_file("no-manifest", header + f'[modules.k]\nsource = "{rel}/../metal/kernels.metal"\n[cases.a]\nmodule = "k"\nfile = "{rel}/vector_add.toml"\n'))
        self.fails("P-PROJECT-REFERENCE", "run", project_file("bad-module", header + module + f'[cases.a]\nmodule = "missing"\nfile = "{rel}/vector_add.toml"\n'))
        self.fails("P-PROJECT-REFERENCE", "run", project_file("bad-default", header.replace('name = "p"', 'name = "p"\ndefault = "zzz"') + module + f'[cases.a]\nmodule = "k"\nfile = "{rel}/vector_add.toml"\n'))
        self.fails("P-PROJECT-AMBIGUOUS-NAME", "run", project_file("same-name", header + module + f'[cases.a]\nmodule = "k"\nfile = "{rel}/vector_add.toml"\n[programs.a]\nsource = "{rel}/../vector_add.cu"\n'))
        self.fails("P-PROJECT-EMPTY", "run", project_file("empty", header + module))
        self.fails("P-PROJECT-VALUE", "run", project_file("kind", header + f'[modules.k]\nsource = "{rel}/../vector_add.cu"\n[cases.a]\nmodule = "k"\nfile = "{rel}/vector_add.toml"\n'))
        self.fails("P-PROJECT-SYNTAX", "run", project_file("syntax", header + "[cases.a\n"))
        self.fails("P-PROJECT-VERSION", "run", project_file("version", header.replace("schema_version = 1", "schema_version = 3") + module))
        args = project_file("args", header + f'[programs.a]\nsource = "{rel}/../vector_add.cu"\narguments = ["x"]\n')
        self.fails("P-PROJECT-ARGUMENTS-AMBIGUOUS", "run", args, "--", "y")
        # A program has no declared reference, even when it is the only target.
        self.fails("P-REFERENCE-REQUIRED", "verify", args)
        # Case files named by a project resolve relative to the project file.
        self.single = project_file("single", header + module + f'[cases.only]\nmodule = "k"\nfile = "{rel}/vector_add_builtin.toml"\n')
        self.defaulted = project_file("defaulted", header.replace('name = "p"', 'name = "p"\ndefault = "t"') + module +
                                      f'[cases.v]\nmodule = "k"\nfile = "{rel}/vector_add.toml"\n[cases.t]\nmodule = "k"\nfile = "{rel}/transpose.toml"\n')

    # ---------------------------------------------------------------- positive GPU
    def verify(self, *target, entry, reference=None, expected=None, human=False):
        destination = self.artifacts()
        if human:
            result = self.call("verify", *target, "--device", "metal:0", "--artifacts", destination)
            assert result.returncode == 0, (result.stdout, result.stderr)
            assert result.stdout.count("Verification: PASS") == 1, result.stdout
            value = json.loads((destination / "report.json").read_text())
        else:
            value = self.json("verify", *target, "--device", "metal:0", "--artifacts", destination)
        self.gpu_report(value, entry, True)
        verification = value["verification"]
        assert verification["status"] == "passed" and all(c["mismatches"] == 0 for c in verification["checks"])
        output = Path(value["outputs"][0]["path"])
        actual = floats(output)
        # Independent re-audit, outside the CLI's comparison.
        if reference is not None:
            assert output.read_bytes() == reference.read_bytes(), f"{output} differs from {reference}"
        if expected is not None:
            assert actual == expected, (actual[:8], expected[:8])
        assert verification["compared"] == len(actual)
        return value

    def positive(self):
        data = self.cases / "data"
        a, b = floats(data / "vector_add_a.f32"), floats(data / "vector_add_b.f32")
        # Public MSL source + manifest, direct kernel-only execution.
        self.verify(self.metal, "--manifest", self.manifest, "--entry", "vector_add", "--case",
                    self.cases / "vector_add.toml", entry="vector_add",
                    reference=data / "vector_add_expected.f32",
                    expected=[f32(x + y) for x, y in zip(a, b)], human=True)
        # Packaged .prx, builtin reference with sentinels beyond n.
        inline_a, inline_b = [1.5, -2.0, 0.25, 1024.0, -0.0], [2.5, 2.0, -0.75, 0.5, 3.0]
        self.verify(self.prx, "--case", self.cases / "vector_add_builtin.toml", entry="vector_add",
                    expected=[f32(x + y) for x, y in zip(inline_a, inline_b)] + [-65536.0, -65536.0])
        reduce_input = floats(data / "reduce_input.f32")
        sums = [float(sum(reduce_input[g * 64:(g + 1) * 64])) for g in range(16)] + [-65536.0, -65536.0]
        self.verify(self.prx, "--entry", "block_reduce", "--case", self.cases / "block_reduce.toml",
                    entry="block_reduce", expected=sums)
        t = floats(data / "transpose_input.f32")
        transposed = [t[y * 33 + x] for x in range(33) for y in range(19)]
        self.verify(self.prx, "--case", self.cases / "transpose.toml", entry="tiled_transpose",
                    reference=data / "transpose_expected.f32", expected=transposed)
        self.verify("--project", self.cases / "paralyn.toml", "--case", "transpose-builtin",
                    entry="tiled_transpose", expected=transposed + [-65536.0, -65536.0])
        project = self.verify(self.cases / "paralyn.toml", "--case", "block-reduce", entry="block_reduce",
                              expected=sums)
        assert project["project"]["case"] == "block-reduce" and project["project"]["module"] == "kernels"
        # Existing verified-IR .prk module compiled from CUDA source.
        self.verify(self.prk, "--case", self.cases / "ir_affine.toml", entry="affine",
                    reference=data / "affine_expected.f32",
                    expected=[f32(f32(x * 0.5) + y) for x, y in zip(a, b)])
        # Selection: discovered ./paralyn.toml with a single target, and a declared default.
        value = self.json("verify", "--device", "metal:0", "--artifacts", self.artifacts(),
                          cwd=self.single.parent)
        self.gpu_report(value, "vector_add", True)
        value = self.json("verify", self.defaulted, "--artifacts", self.artifacts())
        self.gpu_report(value, "tiled_transpose", True)
        assert value["project"]["case"] == "t"

    def run_is_not_verify(self):
        destination = self.artifacts()
        result = self.call("run", self.prx, "--case", self.cases / "vector_add.toml", "--device",
                           "metal:0", "--artifacts", destination)
        assert result.returncode == 0 and "PASS" not in result.stdout + result.stderr, result.stdout
        value = json.loads((destination / "report.json").read_text())
        self.gpu_report(value, "vector_add", False)
        assert value["verification"] == {"status": "not_requested", "contract_declared": True}
        assert (destination / "outputs/out.bin").read_bytes() == (self.cases / "data/vector_add_expected.f32").read_bytes()
        checked = self.json("check", self.prx, "--case", self.cases / "block_reduce.toml", "--device", "metal:0")
        assert checked["gpu_work_submitted"] is False and checked["status"] == "checked"
        assert "run_id" not in checked
        # Build provenance is also reported for single-file check/inspect.
        for command in (("inspect", self.prx), ("check", self.prx, "--device", "metal:0")):
            inspected = self.json(*command)
            assert re.fullmatch(r"[0-9a-f]{40}", inspected["build"]["revision"]), inspected
            assert inspected["source_revision"] == inspected["build"]["revision"]
            assert type(inspected["build_dirty"]) is bool and inspected["gpu_work_submitted"] is False
        existing = self.artifacts()
        existing.mkdir()
        (existing / "keep").write_text("x")
        self.fails("P-OUTPUT-EXISTS", "run", self.prx, "--case", self.cases / "vector_add.toml",
                   "--artifacts", existing)

    def detects_mismatch(self):
        """A wrong reference must fail: proves verify compares rather than asserts."""
        wrong = self.base().replace('{ builtin = "vector-add-f32", lhs = "a", rhs = "b", count = "n" }',
                                    "{ values = [4.0, 0.0, -0.5, 1024.5, 3.0, -65536.0, -65535.99609375] }  # adjacent f32: 1 ULP")
        path = self.case("wrong-reference.toml", wrong)
        destination = self.artifacts()
        result = self.call("verify", self.prx, "--case", path, "--artifacts", destination)
        assert result.returncode == 1, (result.stdout, result.stderr)
        assert "Verification: FAIL" in result.stdout and "PASS" not in result.stdout, result.stdout
        value = json.loads((destination / "report.json").read_text())
        assert value["status"] == "failed" and value["diagnostic"]["id"] == "P-VERIFY-MISMATCH"
        check = value["verification"]["checks"][0]
        assert check["mismatches"] == 1 and check["first_mismatches"][0]["index"] == 6, check
        assert check["first_mismatches"][0]["actual"] == -65536.0
        self.gpu_events += 1
        # Toleranced comparison accepts the same one-ULP difference only when declared.
        tolerant = self.case("tolerant.toml", wrong.replace('kind = "exact"', 'kind = "ulp", ulp = 1'))
        value = self.json("verify", self.prx, "--case", tolerant, "--artifacts", self.artifacts())
        assert value["verification"]["checks"][0]["max_ulp_distance"] == 1
        self.gpu_events += 1

    def program_through_project(self):
        destination = self.artifacts()
        result = self.call("run", "--project", self.cases / "paralyn.toml", "--program", "cuda-vector-add",
                           "--device", "metal:0", "--artifacts", destination)
        assert result.returncode == 0, (result.stdout, result.stderr)
        assert "Verification: PASS" in result.stdout  # printed by the CUDA program's own comparison
        value = json.loads((destination / "report.json").read_text())
        assert value["project"]["program"] == "cuda-vector-add" and value["verification"]["status"] == "not_requested"
        assert value["build"]["revision"] == value["source_revision"]
        self.gpu_events += 1

    def bounded_capture(self):
        """Program streams are bounded sidecars; a partial report exists while the app runs."""
        app = self.work / "noisy.py"
        app.write_text(
            "import json, os, pathlib, sys\n"
            "report = pathlib.Path(os.environ['PARALYN_ARTIFACT_DIR']).parent / 'report.json'\n"
            "state = json.loads(report.read_text())\n"
            "sys.stderr.write('partial=%s status=%s\\n' % (state['partial'], state['status']))\n"
            "sys.stdout.write('x' * 5000)\n"
            "sys.exit(3)\n")
        destination = self.artifacts()
        environment = dict(self.environment, PARALYN_CAPTURE_LIMIT_BYTES="1000",
                           PARALYN_PYTHON=sys.executable)
        result = subprocess.run([str(self.paralyn), "run", str(app), "--json", "--artifacts", str(destination)],
                                cwd=self.work, env=environment, text=True, capture_output=True, timeout=60)
        assert result.returncode == 3, (result.stdout, result.stderr)
        value = json.loads(result.stdout)
        capture = value["application"]["capture"]
        assert capture["streamed"] and capture["stdout_bytes"] == 5000 and capture["stdout_persisted_bytes"] == 1000
        assert capture["stdout_truncated"] and not capture["stderr_truncated"], capture
        assert (destination / "application.stdout").read_bytes() == b"x" * 1000
        assert (destination / "application.stderr").read_text() == "partial=True status=running\n"
        final = json.loads((destination / "report.json").read_text())
        assert final["status"] == "failed" and "partial" not in final and final["exit_code"] == 3
        environment["PARALYN_CAPTURE_LIMIT_BYTES"] = "12x"
        rejected = self.artifacts()
        for target in (app, self.source / "examples/vector_add.cu"):
            bad = subprocess.run([str(self.paralyn), "run", str(target), "--json", "--artifacts", str(rejected)],
                                 cwd=self.work, env=environment, text=True, capture_output=True, timeout=60)
            assert bad.returncode == 1, (bad.stdout, bad.stderr)
            diagnostic = json.loads(bad.stdout)["diagnostic"]
            assert diagnostic["id"] == "P-CAPTURE-LIMIT", diagnostic
            assert "PARALYN_CAPTURE_LIMIT_BYTES" in diagnostic["message"], diagnostic
            assert not rejected.exists(), f"rejected capture limit created evidence in {rejected}"
            self.negative += 1

    def doctor_provenance(self):
        value = self.json("doctor", "--device", "metal:0", "--artifacts", self.artifacts())
        assert re.fullmatch(r"[0-9a-f]{40}", value["build"]["revision"]) and type(value["build_dirty"]) is bool
        evidence = json.loads((Path(value["artifacts"]) / "execution.json").read_text())
        assert evidence["paralyn_commit"] == value["build"]["revision"], evidence
        assert evidence["paralyn_revision_source"] == "embedded_build"
        assert evidence["paralyn_dirty"] == value["build_dirty"] and evidence["llvm_version"] != "unknown"
        self.gpu_events += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paralyn", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--prk", required=True)
    args = parser.parse_args()
    source = Path(args.source).resolve()
    check = subprocess.run([sys.executable, str(source / "examples/cases/generate_data.py"), "--check"],
                           capture_output=True, text=True)
    assert check.returncode == 0, check.stderr
    with tempfile.TemporaryDirectory(prefix="paralyn-cases-") as temporary:
        suite = Suite(Path(args.paralyn).resolve(), source, Path(args.prk).resolve(), Path(temporary).resolve())
        suite.missing_case()
        suite.malformed_cases()
        suite.malformed_projects()
        suite.two_outputs()
        suite.positive()
        suite.run_is_not_verify()
        suite.detects_mismatch()
        suite.program_through_project()
        suite.bounded_capture()
        suite.doctor_provenance()
        print(json.dumps({"negative_checks": suite.negative, "gpu_events": suite.gpu_events}))


if __name__ == "__main__":
    main()
