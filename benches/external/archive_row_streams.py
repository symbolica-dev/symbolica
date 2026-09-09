#!/usr/bin/env python3
"""Archive completed row-stream measurements, interpreter inputs and provenance."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tarfile

root = Path(__file__).resolve().parents[2]
base = root / "target/reconstruction-external"
out = root / "benches/results/reconstruction/row-streams"
out.mkdir(parents=True, exist_ok=True)
assert "test result: ok. 50 passed; 0 failed" in (base / "row-stream-tests.log").read_text()
tables = {}
for path in sorted(base.glob("row-stream-*.csv")):
    rows = list(csv.DictReader(path.open()))
    assert rows and all(row["status"] == "ok" for row in rows), path
    for row in rows:
        if row.get("probes_by_prime"):
            assert sum(int(pair.split(":")[1]) for pair in row["probes_by_prime"].split(";")) == int(row["probes"])
        if "outputs" in row:
            assert row["completed"] == row["outputs"]
    tables[path.stem] = rows
    shutil.copyfile(path, out / path.name)
assert sum(len(tables["row-stream-" + name]) for name in
           ["nb0", "graph5", "q", "sheared", "sparse", "box2l", "xbox2l2m", "diamond3l", "tth2l_b16"]) == 4325
for name, count in {"joint-xbox": 1, "joint-b25": 1, "joint-xbox-all": 3, "b25-scalar": 12,
                    "xbox-all-reference": 3, "joint-b25-all": 6, "joint-b16-all": 3}.items():
    assert len(tables["row-stream-" + name]) == count, name
with tarfile.open(out / "benchmark-logs.tar.gz", "w:gz", compresslevel=6) as tar:
    for name, rows in tables.items():
        folder = base / (name + "-logs")
        for path in sorted(folder.glob("*")):
            if path.is_file() and path.suffix in {".json", ".log", ".txt", ".q-oracle"}:
                tar.add(path, arcname=str(path.relative_to(base)))
for path in sorted(base.glob("row-stream-*")):
    if path.is_file() and path.suffix in {".json", ".log"} and path.name != "row-stream-archive.log":
        shutil.copyfile(path, out / path.name)
with tarfile.open(out / "full-target-inputs.tar.gz", "w:gz", compresslevel=6) as tar:
    family = "xbox2l2m_all"
    inputs = base / "ibp-inputs" / family
    manifest = json.loads((inputs / "manifest.json").read_text())
    assert len(manifest["selected_cases"]) == 71
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
           root / "tests/rational_reconstruction.rs", root / "examples/reconstruction_joint_benchmark.rs",
           root / "examples/support/reconstruction_q.rs"]
sources += [root / "benches/external" / name for name in ["archive_row_streams.py", "run_row_stream_regressions.py", "run_q_stress.py", "run_stress.py", "run_joint_stress.py", "fire7_q_stress.cpp", "firefly_q_stress.cpp"]]
binaries = [root / "target/release/examples" / name for name in ["reconstruction_stress_benchmark", "reconstruction_joint_benchmark"]]
binaries += [base / name for name in ["row-stream-joint-baseline", "row-stream-scalar-baseline", "firefly-joint-stress", "libtrace-oracle.so", "fire7-q-stress", "firefly-q-stress"]]
for name in tables:
    folder = base / (name + "-logs")
    binaries += list((folder / "binaries").glob("*"))
    if (folder / "symbolica-stress").exists():
        binaries.append(folder / "symbolica-stress")
(out / "sources-and-binaries.sha256").write_text("".join(
    f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}\n"
    for path in sources + binaries))
joint = {}
for name, rows in tables.items():
    if "outputs" not in rows[0]:
        continue
    for method in sorted({row["method"] for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        counts = [int(row["probes"]) for row in selected]
        joint[name + ":" + method] = {
            "runs": len(selected), "outputs": int(selected[0]["outputs"]),
            "median_probes": statistics.median(counts),
            "min_probes": min(counts), "max_probes": max(counts),
            "median_seconds": statistics.median(float(row["elapsed_us"]) for row in selected) / 1e6,
        }
summary = {"baseline": "1fbc3a6f", "tests": 50,
           "successful_recorded_runs": sum(map(len, tables.values())),
           "tables": {name: len(rows) for name, rows in tables.items()}, "joint": joint}
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
