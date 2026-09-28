#!/usr/bin/env python3
"""Black-box CLI qualification: real processes, actual GPU probe, independent audit."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


class Suite:
    def __init__(self, executable, source, work):
        self.executable, self.source, self.work = executable, source, work
        self.checks = 0
        self.environment = os.environ.copy()
        self.environment["PARALYN_PYTHON"] = sys.executable
        # Inherited destinations must not mingle unrelated runtime logs/evidence.
        for key in ("PARALYN_RUNTIME_LOG", "PARALYN_ARTIFACT_DIR"):
            self.environment.pop(key, None)

    def run(self, *arguments, code=0, structured=True):
        result = subprocess.run(
            [str(self.executable), *map(str, arguments)], cwd=self.work,
            env=self.environment, text=True, encoding="utf-8", capture_output=True,
            timeout=90, check=False,
        )
        self.checks += 1
        assert (result.returncode == code if code is not None else result.returncode != 0), (
            arguments, result.returncode, result.stdout, result.stderr)
        if not structured:
            return result
        try:
            value = json.loads(result.stdout)
        except json.JSONDecodeError as error:
            raise AssertionError(f"stdout is not exactly one JSON document: {arguments}\n{result.stdout}\n{result.stderr}") from error
        assert value["schema"] == "paralyn.report" and value["schema_version"] == 1, value
        if result.returncode:
            assert value["status"] == "failed", value
        return value, result

    def cpu(self):
        support, _ = self.run("support", "--json")
        rows = support["support"]["inputs"]
        assert len(rows) == 17 and all(row["required"] for row in rows)
        assert support["support"]["complete_portfolio"] is False
        for command in [("compile", "--json"), ("support", "--json", "--unknown"),
                        ("inspect", "missing.wgsl", "--json")]:
            failure, _ = self.run(*command, code=None)
            assert failure["failure_origin"] == "paralyn" and failure["diagnostic"]["message"]

        saved = self.work / "support-report.json"
        value, _ = self.run("support", "--json", "--report-json", saved)
        assert json.loads(saved.read_text()) == value
        before = saved.read_bytes()
        failure, _ = self.run("support", "--json", "--report-json", saved, code=None)
        assert failure["diagnostic"]["id"] == "P-OUTPUT-EXISTS" and saved.read_bytes() == before
        displayed, _ = self.run("report", saved, "--json")
        assert displayed == value

        self.marker = self.work / "host-was-executed"
        self.cuda = self.work / "ordinary source ü.cu"
        self.cuda.write_text(
            '#include <cuda_runtime.h>\n#include <cstdio>\n'
            '__global__ void fill(float* out) { out[threadIdx.x] = 3.0f; }\n'
            f'int main() {{ auto f=std::fopen({json.dumps(str(self.marker))}, "w"); '
            'if(f){std::fputs("unexpected",f);std::fclose(f);} return 93; }\n',
            encoding="utf-8")
        inspected, _ = self.run("inspect", self.cuda, "--json")
        assert inspected["host_code_executed"] is False and inspected["gpu_work_submitted"] is False
        assert not self.marker.exists(), "inspect executed host main"
        output = self.work / "ordinary.prk"
        compiled, _ = self.run("compile", self.cuda, "--output", output, "--json")
        assert compiled["status"] == "compiled" and output.read_bytes().startswith(b"PARALYN\0")
        assert not self.marker.exists(), "compile executed host main"
        original = output.read_bytes()
        self.run("compile", self.cuda, "--output", output, "--json", code=None)
        assert output.read_bytes() == original

        self.metal = self.source / "examples/metal/kernels.metal"
        self.manifest = json.loads((self.source / "examples/metal/kernels.json").read_text())
        self.manifest_path = self.work / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest))
        self.msl_output = self.work / "kernels.prx"
        compiled, _ = self.run("compile", self.metal, "--manifest", self.manifest_path,
                               "--output", self.msl_output, "--json")
        assert self.msl_output.read_bytes().startswith(b"PARALYNX")
        inspected, _ = self.run("inspect", self.msl_output, "--json")
        assert {e["name"] for e in inspected["entries"]} == {"vector_add", "block_reduce", "tiled_transpose"}

        cases = []
        def change(path, value):
            item = copy.deepcopy(self.manifest)
            target = item
            for part in path[:-1]:
                target = target[part]
            target[path[-1]] = value
            cases.append(json.dumps(item))

        for value in (-1, 1.5, True, "1", 2**32 + 1, 2**64):
            change(["numerical_policy"], value)
        for field, values in {
            "binding": (-1, 0.5, True, "0", 2**32),
            "alignment": (-1, 4.5, True, "4", 2**32 + 4),
            "minimum_bytes": (-1, 4.5, True, "4", 2**64 + 4),
            "buffer": (0, "true", None),
        }.items():
            for value in values:
                change(["entries", 0, "parameters", 0, field], value)
        for value in ([0, 0], [0, 0, 0, 0], [2**32, 0, 0], [0.5, 0, 0], {"x": 0, "y": 0, "z": 0}):
            change(["entries", 0, "required_block"], value)
        change(["entries"], {"first": self.manifest["entries"][0]})
        change(["entries", 0, "parameters"], {"first": self.manifest["entries"][0]["parameters"][0]})
        cases.append(json.dumps(self.manifest).replace('"numerical_policy": 1', '"numerical_policy": 9, "numerical_policy": 1'))
        cases.append(json.dumps(self.manifest).replace('"binding": 0', '"binding": 12, "binding": 0', 1))
        for index, text in enumerate(cases):
            path = self.work / f"invalid-{index}.json"
            path.write_text(text)
            destination = self.work / f"invalid-{index}.prx"
            failed, _ = self.run("compile", self.metal, "--manifest", path,
                                  "--output", destination, "--json", code=None)
            assert failed["diagnostic"]["message"] and not destination.exists()
        # Large and malformed source inputs cannot be mistaken for a successful import.
        unsupported = self.work / "unsupported.wgsl"
        unsupported.write_text("@compute @workgroup_size(1) fn main() {}")
        failed, _ = self.run("inspect", unsupported, "--json", code=None)
        assert failed["diagnostic"]["id"] == "P-FRONTEND-UNIMPLEMENTED"

    def gpu(self):
        checked, _ = self.run("check", self.cuda, "--json")
        assert checked["backend_compilation"] == "passed" and not self.marker.exists()
        assert checked["host_code_executed"] is False and checked["gpu_work_submitted"] is False
        checked, _ = self.run("check", self.metal, "--manifest", self.manifest_path, "--json")
        assert checked["backend_compilation"] == "passed" and checked["gpu_work_submitted"] is False

        directory = self.work / "doctor"
        doctor, _ = self.run("doctor", "--json", "--artifacts", directory)
        assert doctor["verification"]["status"] == "passed" and doctor["cpu_fallback"] is False
        assert doctor["device"]["backend"].lower() == "metal"
        timing = doctor["timing"]
        assert timing["completed"] and timing["duration_valid"] and timing["timestamps_valid"]
        assert timing["duration_seconds"] > 0 and timing["start_seconds"] > 0
        assert timing["end_seconds"] > timing["start_seconds"]
        reference = json.loads((directory / "reference.json").read_text())
        assert len(reference["actual"]) == doctor["verification"]["elements"] >= 257
        assert len(reference["a"]) == len(reference["b"]) == len(reference["actual"])
        assert len(set(reference["actual"])) > 20
        for a, b, actual in zip(reference["a"], reference["b"], reference["actual"]):
            assert actual == a + b, "Independent report readback did not match CPU arithmetic"
        evidence = json.loads((directory / "execution.json").read_text())
        assert evidence["cpu_fallback"] is False and len(evidence["launches"]) == 1
        command = evidence["launches"][0]
        assert command["command_status"] == "completed" and not command["error"]
        assert command["gpu_start_seconds"] == timing["start_seconds"]
        assert command["gpu_end_seconds"] == timing["end_seconds"]
        assert (directory / command["source_file"]).stat().st_size > 0
        assert (directory / "probe.prk").read_bytes().startswith(b"PARALYN\0")
        assert json.loads((directory / "report.json").read_text()) == doctor
        displayed, _ = self.run("report", directory / "report.json", "--json")
        assert displayed == doctor
        self.run("doctor", "--device", "cuda:0", "--json", code=None)

        application = self.work / "app ü.py"
        application.write_text(
            "import json,sys\n"
            "print(json.dumps(sys.argv[1:], ensure_ascii=False), flush=True)\n"
            "print('Verification: PASS', flush=True)\n"
            "print('child stderr Ω', file=sys.stderr, flush=True)\n"
            "raise SystemExit(int(sys.argv[1]))\n", encoding="utf-8")
        literal = ["hello world", "雪 / café", "$(touch forbidden)", "`uname`", 'a"b', "", "--json", "--"]
        for status in (0, 37):
            run_dir = self.work / f"python-{status}"
            value, result = self.run("run", application, "--json", "--artifacts", run_dir,
                                     "--", str(status), *literal, code=status)
            expected_args = [str(status), *literal]
            expected_stdout = json.dumps(expected_args, ensure_ascii=False) + "\nVerification: PASS\n"
            assert value["application"]["arguments"] == expected_args
            assert value["application"]["exit_code"] == status and value["exit_code"] == status
            assert value["failure_origin"] == ("application" if status else None)
            assert Path(value["application"]["stdout"]).read_text() == expected_stdout
            assert Path(value["application"]["stderr"]).read_text() == "child stderr Ω\n"
            assert value["verification"]["status"] == "not_requested"
            assert value["runtime_evidence"] is None
            assert "child stderr" not in result.stderr and "Verification: PASS" not in result.stdout
            assert not (self.work / "forbidden").exists(), "Application arguments went through a shell"
        run_dir = self.work / "python-live"
        result = self.run("run", application, "--artifacts", run_dir, "--", "0", *literal, structured=False)
        assert result.stdout == json.dumps(["0", *literal], ensure_ascii=False) + "\nVerification: PASS\n"
        assert "child stderr Ω\n" in result.stderr
        value = json.loads((run_dir / "report.json").read_text())
        assert value["verification"]["status"] == "not_requested"

        large = self.work / "both_streams.py"
        large.write_text("import sys,threading\n"
                         "def send(stream,letter):\n"
                         "    stream.write(letter*262144); stream.flush()\n"
                         "a=threading.Thread(target=send,args=(sys.stdout,'O'))\n"
                         "b=threading.Thread(target=send,args=(sys.stderr,'E'))\n"
                         "a.start(); b.start(); a.join(); b.join()\n")
        value, _ = self.run("run", large, "--json", "--artifacts", self.work / "large-streams")
        assert Path(value["application"]["stdout"]).read_bytes() == b"O" * 262144
        assert Path(value["application"]["stderr"]).read_bytes() == b"E" * 262144

        # C source must remain C; merely selecting -std=c11 with a C++ driver is insufficient.
        c_source = self.work / "native language.c"
        c_source.write_text('#ifdef __cplusplus\n#error C source compiled as C++\n#endif\n'
                            '#include <stdio.h>\n#include <paralyn/native.h>\n'
                            'int main(void) { printf("C ABI %u\\n",pr_abi_version()); return 0; }\n')
        value, _ = self.run("run", c_source, "--artifacts", self.work / "native-c", "--json")
        assert Path(value["application"]["stdout"]).read_text() == "C ABI 1\n"
        assert value["verification"]["status"] == "not_requested"

        # A recovered API error is not proof that it caused a later application exit.
        recovery = self.work / "recovered.c"
        recovery.write_text('#include <paralyn/native.h>\n'
                            'int main(void) { uint64_t n=0; pr_status s=pr_buffer_size(0,&n); '
                            'return s==PR_INVALID_HANDLE ? 37 : 99; }\n')
        value, _ = self.run("run", recovery, "--json", "--artifacts", self.work / "recovered", code=37)
        assert value["runtime_errors_observed"] is True and value["failure_origin"] == "application"

        # Runtime-forced process exits have their own causal terminal event.
        fatal = self.work / "unobserved.cu"
        fatal.write_text('#include <cuda_runtime.h>\n'
                         '__global__ void unused(float* p) { p[threadIdx.x]=1.0f; }\n'
                         'int main() { cudaFree(reinterpret_cast<void*>(1)); return 0; }\n')
        value, _ = self.run("run", fatal, "--json", "--artifacts", self.work / "unobserved", code=None)
        assert value["runtime_errors_observed"] is True and value["failure_origin"] == "runtime"
        assert any(e["category"]=="process_failure" and e["status"]=="failed" for e in value["runtime_events"])

        # Refusing an existing report must happen before starting arbitrary application code.
        side_effect = self.work / "side_effect.py"
        side_effect.write_text(f"from pathlib import Path\nPath({str(self.marker)!r}).write_text('bad')\n")
        report = self.work / "do-not-overwrite.json"
        report.write_text("original report\n")
        failed, _ = self.run("run", side_effect, "--json", "--report-json", report,
                              "--artifacts", self.work / "must-not-run", code=None)
        assert failed["diagnostic"]["id"] == "P-OUTPUT-EXISTS"
        assert not self.marker.exists() and report.read_text() == "original report\n"

        if os.name == "posix":
            ready = self.work / "interrupt-ready"
            interrupted = self.work / "interrupt.py"
            interrupted.write_text(
                "import signal,sys\nfrom pathlib import Path\n"
                "def stop(number,frame):\n"
                "    print('child observed SIGINT',flush=True)\n"
                "    raise SystemExit(77)\n"
                "signal.signal(signal.SIGINT,stop)\n"
                f"Path({str(ready)!r}).write_text('ready')\n"
                "while True: signal.pause()\n")
            process = subprocess.Popen(
                [str(self.executable), "run", str(interrupted), "--json", "--artifacts",
                 str(self.work / "interrupted")], cwd=self.work, env=self.environment,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                start_new_session=True,
            )
            try:
                deadline = time.monotonic() + 15
                while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert ready.exists(), "Child did not reach installed signal handler"
                process.send_signal(signal.SIGINT)
                stdout, stderr = process.communicate(timeout=15)
                self.checks += 1
                assert process.returncode == 77, (stdout, stderr, process.returncode)
                value = json.loads(stdout)
                assert value["application"]["interrupted"] and value["application"]["exit_code"] == 77
                assert value["failure_origin"] == "application" and value["verification"]["status"] == "not_requested"
                assert "not claimed" in value["interruption"]
                assert Path(value["application"]["stdout"]).read_text() == "child observed SIGINT\n"
            finally:
                if process.poll() is None:
                    process.kill()
                    process.wait(timeout=15)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--paralyn", required=True)
    parser.add_argument("--source-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--artifacts")
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    def execute(work):
        suite = Suite(Path(args.paralyn).resolve(), Path(args.source_root).resolve(), work)
        suite.cpu()
        if not args.cpu_only:
            suite.gpu()
        result = {"verification": "PASS", "commands": suite.checks,
                  "physical_gpu_probe": not args.cpu_only,
                  "cli_sha256": hashlib.sha256(suite.executable.read_bytes()).hexdigest()}
        (work / "cli-test-summary.json").write_text(json.dumps(result, indent=2) + "\n")
        print(f"CLI {'CPU contracts' if args.cpu_only else 'real GPU probe, process semantics, JSON and rejection contracts'}: PASS ({suite.checks} commands)")
    if args.artifacts:
        work = Path(args.artifacts).resolve()
        work.mkdir(parents=True, exist_ok=False)
        execute(work)
    else:
        with tempfile.TemporaryDirectory(prefix="paralyn-cli-") as temporary:
            execute(Path(temporary))


if __name__ == "__main__":
    main()
