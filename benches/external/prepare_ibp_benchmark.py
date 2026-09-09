#!/usr/bin/env python3
"""Generate independent IBP coefficient inputs with pinned Kira and Ratracer."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("family", choices=["box1l", "box2l", "diamond2l", "diamond3l", "tth2l_b16", "tth2l_b25", "xbox2l2m"])
parser.add_argument("integral", help="one integral in the author's mandatory list, e.g. basis[1,1,1,1,1,1,1,-1,-1]")
parser.add_argument("--count", type=int, default=4)
parser.add_argument("--label", help="separate work/input label for another selection of the same family")
args = parser.parse_args()
assert args.count > 0
label = args.label or args.family
assert re.fullmatch(r"[A-Za-z0-9_]+", label), "label must be a simple directory name"
root = Path(__file__).resolve().parents[2]
external = root / "target/reconstruction-external"
source = external / "ibp-benchmark"
pins = {"ibp-benchmark": "f518de1f4f89cc716a31d5d9a9a9ba0a3b72f458", "kira": "aadf0671ed090b8427e14a3d908d6c7033d46b2c", "ratracer": "88646ca7b65c24bfce3a8be6e1093a9b89731f23", "ratracer-firefly": "27d4bdec27436bbfb000ce47fd5dbc1ff08ae8a7", "ratracer-flintxx": "0be0a5f4da4dcf475eff00e6adbf5a728fb0153b"}
for name, revision in pins.items():
    assert subprocess.check_output(["git", "-C", str(external/name), "rev-parse", "HEAD"], text=True).strip() == revision
template = source / "problems" / (args.family + ".kira-ratracer")
work = external / "ibp-work" / label
work.mkdir(parents=True, exist_ok=True)
logs = work / "preparation-logs"
logs.mkdir(exist_ok=True)
# The upstream generator writes every family, so protect generation and copying
# when independent family preparations run concurrently.
with (source / ".symbolica-preparation.lock").open("w") as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    subprocess.run([sys.executable, "generate.py"], cwd=source, check=True, stdout=subprocess.DEVNULL)
    assert args.integral in (template / "integrals").read_text().splitlines()
    assert args.integral + "*-1" not in (template / "master-eqns").read_text().splitlines(), \
        "selected integral is already a preferred master; choose a non-master benchmark"
    shutil.copytree(template / "config", work / "config", dirs_exist_ok=True)
    for name in ["integrals", "master-eqns", "output-list"]:
        shutil.copyfile(template / name, work / name)
    job = (template / "export.yaml").read_text()
    assert job.count("    run_firefly: false\n") == 1
    (work / "export-equations.yaml").write_text(job.replace("    run_firefly: false\n", ""))
    (work / "target-list").write_text(f"CO[{args.integral},*]\n")
    limits = re.search(r"--maxr=(\d+) --maxs=(\d+) --maxd=(\d+)", (template / "run.sh").read_text()).groups()
loader = ([os.environ["EXTERNAL_LOADER"], "--library-path", os.environ["EXTERNAL_LIBRARY_PATH"]]
          if "EXTERNAL_LOADER" in os.environ else [])
affinity = ["taskset", "-c", os.environ["BENCH_CPU_SET"]] if "BENCH_CPU_SET" in os.environ else []


def run(label, executable, arguments):
    command = affinity + loader + [str(executable)] + list(map(str, arguments))
    (logs / (label + ".command.json")).write_text(json.dumps(command, indent=2) + "\n")
    try:
        result = subprocess.run(command, cwd=work, capture_output=True, text=True,
                                timeout=float(os.environ.get("IBP_PREPARE_TIMEOUT", "600")))
    except subprocess.TimeoutExpired as e:
        (logs / (label + ".log")).write_bytes((e.stdout or b"") + (e.stderr or b""))
        raise
    (logs / (label + ".log")).write_text(result.stdout + result.stderr)
    result.check_returncode()
    return result.stdout


run("kira-export", external / "kira-build/src/kira/kira", ["--parallel=" + os.environ.get("IBP_THREADS", "4"), "export-equations.yaml"])
equations = sorted(work.glob("input_kira/basis/*.kira.gz"))
assert equations
trace_args = []
if (work / "master-eqns").stat().st_size:
    trace_args += ["define-family", "master", "load-equations", "master-eqns"]
for equation in equations:
    trace_args += ["load-equations", equation]
trace_args += ["solve-equations", "choose-equation-outputs", "--family=basis"]
trace_args += ["--max" + k + "=" + v for k, v in zip("rsd", limits)]
trace_args += ["drop-equations", "keep-outputs", "target-list", "optimize", "finalize", "save-trace", "top.trace", "list-outputs", "--to=top.outputs", "list-inputs"]
tracer = external / "ratracer-tool"
variables = [line.split(" ", 1)[1] for line in run("trace", tracer, trace_args).splitlines()]
assert variables and all(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", v) for v in variables)
coefficients = work / "ranked-coefficients"
coefficients.mkdir(exist_ok=True)
entries = []
for line in (work / "top.outputs").read_text().splitlines():
    index, name = line.split(" ", 1)
    pick, trace = coefficients / (index + ".output"), coefficients / (index + ".trace")
    pick.write_text(name + "\n")
    # Finalized instructions are immutable to the optimizer. Convert back first.
    run("coefficient-" + index, tracer, ["load-trace", "top.trace", "unfinalize", "keep-outputs", pick, "optimize", "finalize", "save-trace", trace])
    data = trace.read_bytes()
    magic, nv, no, nc, nl, size, hi_size = struct.unpack("<QIIIIQQ", data[:40])
    assert magic == 0x3430303043524052 and no == 1 and hi_size == 0
    entries.append(dict(output=int(index), name=name, instruction_bytes=size,
                        trace_sha256=hashlib.sha256(data).hexdigest()))
entries.sort(key=lambda e: (-e["instruction_bytes"], e["output"]))
assert len(entries) >= args.count
selected = entries[:args.count]
for rank, entry in enumerate(selected, 1):
    entry["case"] = f"ibp_{label}_rank{rank:04d}"
(work / "selected.outputs").write_text("\n".join(e["name"] for e in selected) + "\n")
(work / "selected.names").write_text("".join(f"{i} {e['case']}\n" for i, e in enumerate(sorted(selected, key=lambda e: e["output"]))))
run("reference", tracer, ["load-trace", "top.trace", "unfinalize", "keep-outputs", "selected.outputs", "rename-outputs", "selected.names", "optimize", "finalize", "save-trace", "selected.trace", "reconstruct", "--inmem", "--threads=" + os.environ.get("IBP_THREADS", "4"), "--factor-scan", "--shift-scan", "--to=selected.results"])
expressions = dict(re.findall(r"(\w+)\s*=\s*(.*?);", (work / "selected.results").read_text(), flags=re.S))
assert set(expressions) == {e["case"] for e in selected}
output = external / "ibp-inputs" / label
output.mkdir(parents=True, exist_ok=True)
for entry in selected:
    expression = expressions[entry["case"]].strip() + "\n"
    (output / entry["case"]).write_text(expression)
    (output / (entry["case"] + ".variables")).write_text(" ".join(variables) + "\n")
    trace_link = output / (entry["case"] + ".trace")
    if trace_link.is_symlink():
        trace_link.unlink()
    trace_link.symlink_to(os.path.relpath(coefficients / (str(entry["output"]) + ".trace"), output))
    entry["expression_sha256"] = hashlib.sha256(expression.encode()).hexdigest()
manifest = dict(revisions=pins, family=args.family, label=label, integral=args.integral,
                limits=dict(zip("rsd", map(int, limits))), variables=variables,
                selection="largest finalized single-output instruction byte counts after unfinalizing and optimizing; output index breaks ties",
                entries=entries, selected_cases=[e["case"] for e in selected],
                equations={str(f.relative_to(work)): hashlib.sha256(f.read_bytes()).hexdigest() for f in equations},
                validation="reference expressions reconstructed by the authors' FireFly fork; independent trace checks still required")
(output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(output / "suite-cases.txt").write_text("\n".join(e["case"] for e in selected) + "\n")
print(f"Prepared {len(selected)} {args.family} coefficients; independent validation is still required.")
