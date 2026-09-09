#!/usr/bin/env python3
"""Audit the factor-lifting checkpoint and retain its measured builds and inputs."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tarfile

root = Path(__file__).resolve().parents[2]
base = root / "target/reconstruction-external"
out = root / "benches/results/reconstruction/rational-factors"
out.mkdir(parents=True, exist_ok=True)
read = lambda path: list(csv.DictReader(path.open()))
required = {"nb0": 3319, "graph5": 945, "q": 5, "sheared": 4, "sparse": 4,
            "box2l": 12, "xbox2l2m": 12, "diamond3l": 12, "tth2l_b16": 12,
            "b25-final": 24, "b25-prototype": 4, "joint-b25": 6, "joint-b25-all": 2,
            "joint-b25-primes": 2, "joint-b25-all-primes": 1, "joint-xbox-primes": 1,
            "joint-b16-all": 1, "clean-b25": 4, "clean-q": 5}
tables = {}
for name, count in required.items():
    path = base / f"rational-factor-{name}.csv"
    rows = read(path)
    assert len(rows) == count and all(r["status"] == "ok" for r in rows), name
    for r in rows:
        if r.get("probes_by_prime"):
            assert sum(int(pair.split(":")[1]) for pair in r["probes_by_prime"].split(";")) == int(r["probes"])
        if "outputs" in r: assert r["completed"] == r["outputs"]
    tables[path.stem] = rows
    shutil.copyfile(path, out / path.name)
old = root / "benches/results/reconstruction/ordering-traces"
regressions = {"nb0":"nb0", "graph5":"graph5", "q":"q", "sheared":"sheared", "sparse":"sparse", "box2l":"box2l", "xbox2l2m":"xbox2l2m", "diamond3l":"diamond", "tth2l_b16":"b16"}
for name, previous in regressions.items():
    refs = {(r["case"], r["method"], r["seed"]): r for r in read(old / f"ordering-final-{previous}.csv")}
    for r in tables["rational-factor-" + name]:
        ref = refs[r["case"], r["method"], r["seed"]]
        for field in ["probes", "primes", "images", "support_reuses", "support_fallbacks", "probes_by_prime", "selected_methods"]:
            if field in ref and field in r: assert r[field] == ref[field], (name, r["case"], field)
for name in ["b25", "xbox"]:
    path = base / f"joint-{name}.csv"
    rows = read(path)
    assert len(rows) == 9 and all(r["status"] == "ok" for r in rows)
    tables[path.stem] = rows
    shutil.copyfile(path, out / ("baseline-" + path.name))
medians = {}
for name in ["rational-factor-joint-b25", "joint-b25", "joint-xbox"]:
    for method in {r["method"] for r in tables[name]}:
        rows = [r for r in tables[name] if r["method"] == method]
        assert len({r["probes"] for r in rows}) == 1
        medians[name + ":" + method] = statistics.median(float(r["elapsed_us"]) for r in rows) / 1e6
for name in ["b25", "q"]:
    refs = {(r["case"], r["method"]): r for r in tables["rational-factor-" + ("b25-final" if name == "b25" else "q")]}
    for r in tables["rational-factor-clean-" + name]:
        ref = refs[r["case"], r["method"]]
        assert r["probes"] == ref["probes"] and r["probes_by_prime"] == ref["probes_by_prime"]
with tarfile.open(out / "benchmark-logs.tar.gz", "w:gz", compresslevel=6) as tar:
    for name, rows in tables.items():
        folder = base / (name + "-logs")
        paths = set(folder.glob("*.q-oracle")) | set(folder.glob("*.json"))
        paths.add(folder / "cases.txt")
        for row in rows:
            prefix = row.get("case", "")
            prefix = prefix + "." if prefix else ""
            paths.add(folder / f"{prefix}{row['method']}.{row['seed']}.log")
        for path in sorted(paths):
            if path.exists(): tar.add(path, arcname=str(path.relative_to(base)))
    for path in (base / "joint-driver-controls").iterdir():
        if path.is_file() and path.suffix != ".so": tar.add(path, arcname=str(path.relative_to(base)))
logs = ["rational-factor-build", "rational-factor-tests", "rational-factor-final-build", "rational-factor-final-tests",
        "rational-factor-clean-build", "rational-factor-clean-tests", "rational-factor-clean-clippy", "rational-factor-fmt",
        "rational-factor-joint-driver-build", "rational-factor-joint-histogram-build", "rational-factor-joint-controls",
        "prepare-tth2l_b25_all", "validate-tth2l_b25_all", "joint-trace-b25-all-check"]
for name in logs:
    shutil.copyfile(base / (name + ".log"), out / (name + ".log"))
with tarfile.open(out / "full-target-inputs.tar.gz", "w:gz", compresslevel=6) as tar:
    family = "tth2l_b25_all"
    inputs = base / "ibp-inputs" / family
    manifest = json.loads((inputs / "manifest.json").read_text())
    assert len(manifest["selected_cases"]) == 147
    for name in ["manifest.json", "validation.json", "joint-trace-validation.json", "suite-cases.txt"]:
        path = inputs / name
        tar.add(path, arcname=str(path.relative_to(base)))
    for case in manifest["selected_cases"]:
        for suffix in ["", ".variables"]:
            path = inputs / (case + suffix)
            tar.add(path, arcname=str(path.relative_to(base)))
    work = base / "ibp-work" / family
    for name in ["selected.trace", "selected.names", "selected.outputs", "top.outputs", "config", "master-eqns", "integrals", "target-list", "export-equations.yaml", "preparation-logs"]:
        path = work / name
        tar.add(path, arcname=str(path.relative_to(base)))
sources = [root / "src/poly/reconstruction.rs", *sorted((root / "src/poly/reconstruction").glob("*.rs")),
           root / "tests/rational_reconstruction.rs", root / "examples/support/reconstruction_q.rs", root / "examples/reconstruction_joint_benchmark.rs"]
sources += [root / "benches/external" / name for name in ["archive_rational_factors.py", "run_q_stress.py", "run_joint_stress.py", "firefly_joint_stress.cpp", "check_joint_driver.py"]]
binaries = [root / "target/release/examples" / name for name in ["reconstruction_stress_benchmark", "reconstruction_joint_benchmark"]]
binaries += [base / name for name in ["rational-factor-baseline", "rational-factor-joint-baseline", "rational-factor-prototype", "firefly-joint-stress", "libtrace-oracle.so"]]
for name in tables:
    folder = base / (name + "-logs")
    binaries += list((folder / "binaries").glob("*"))
    if (folder / "symbolica-stress").exists(): binaries.append(folder / "symbolica-stress")
(out / "sources-and-binaries.sha256").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n" for p in sources + binaries))
summary = dict(baseline="5221207f", unchanged_regression_runs=sum(required[name] for name in regressions),
               successful_recorded_runs=sum(len(rows) for rows in tables.values()), median_seconds=medians,
               tests=49, scalar_b25_probes=[14242,6027,1727,1778], selected_joint_b25_probes=14694,
               full_joint_b25_probes=18491, full_joint_b25_reference_probes=16107,
               remaining=["Improve initial interpolation sharing for the nonplanar family and full 147-output b25 target."],
               notes=["Main repeated measurements precede the arithmetic-only no-factor shortcut and the added joint prime histogram; follow-up runs verify unchanged probes.",
                      "rational-factor-tests.log is the development run with the shorter performance control; both later 49-test runs pass.",
                      "The full target archive retains the joint trace; individual optimized traces can be regenerated from the pinned preparation.",
                      "All full output sets in this checkpoint belong to one integral, not a complete IBP reduction table."])
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
