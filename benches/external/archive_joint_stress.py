#!/usr/bin/env python3
"""Archive completed comparisons and clearly labeled snapshots of ongoing runs."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tarfile

root = Path(__file__).resolve().parents[2]
base = root / "target/reconstruction-external"
out = root / "benches/results/reconstruction/joint-ibp"
out.mkdir(parents=True, exist_ok=True)
read = lambda path: list(csv.DictReader(path.open()))
required = {"joint-b16": 9, "joint-b16-all": 9, "joint-b25-scalar-final": 20}
summary = {"complete_suites": {}, "pending_suites": {}, "median_seconds": {}}
tables = {}
for name, count in required.items():
    rows = read(base / (name + ".csv"))
    assert len(rows) == count
    if "scalar" in name:
        assert sum(r["status"] == "ok" for r in rows) == 19
        failed = [r for r in rows if r["status"] != "ok"]
        assert len(failed) == 1 and failed[0]["method"] == "Rare_scaling" and failed[0]["status"] == "prime_limit"
    else:
        assert all(r["status"] == "ok" and r["completed"] == r["outputs"] for r in rows)
        for method in {r["method"] for r in rows}:
            selected = [r for r in rows if r["method"] == method]
            assert len({r["probes"] for r in selected}) == 1
            summary["median_seconds"][name + ":" + method] = statistics.median(float(r["elapsed_us"]) for r in selected) / 1e6
    tables[name] = rows
    summary["complete_suites"][name] = {"runs": count, "successful": sum(r["status"] == "ok" for r in rows)}
    shutil.copyfile(base / (name + ".csv"), out / (name + ".csv"))
for name in ["joint-b25", "joint-xbox"]:
    path = base / (name + ".csv")
    rows = read(path) if path.exists() else []
    tables[name] = rows
    complete = len(rows) == 9
    summary["complete_suites" if complete else "pending_suites"][name] = {"recorded_runs": len(rows), "planned_runs": 9}
    if path.exists():
        shutil.copyfile(path, out / (name + (".csv" if complete else ".partial.csv")))
for name in ["joint-b25-scalar", "joint-b25-128", "joint-b25-fire7-127", "joint-b25-rare-114"]:
    tables[name] = read(base / (name + ".csv"))
    shutil.copyfile(base / (name + ".csv"), out / (name + ".csv"))
logs = ["joint-trace-build", "joint-rust-build", "joint-rust-final-build", "joint-firefly-build", "joint-rare-build", "joint-clippy", "joint-fmt", "joint-trace-checks", "joint-trace-b16-all-check", "joint-driver-checks", "joint-rare-controls", "joint-scalar-driver-checks", "joint-scalar-abi-b16", "joint-scalar-abi-b25", "prepare-tth2l_b25", "validate-tth2l_b25", "prepare-tth2l_b16_all", "validate-tth2l_b16_all"]
for name in logs:
    shutil.copyfile(base / (name + ".log"), out / (name + ".log"))
with tarfile.open(out / "benchmark-logs.tar.gz", "w:gz") as tar:
    for name, rows in tables.items():
        folder = base / (name + "-logs")
        if not folder.exists():
            continue
        # Per-run logs are written only after process completion. The generic
        # FireFly log may still be active and is deliberately not snapshotted.
        files = set(folder.glob("*.q-oracle")) | set(folder.glob("*.command.json"))
        files |= {folder / "cases.txt", folder / "metadata.json"}
        for row in rows:
            prefix = f"{row['case']}." if "case" in row else ""
            files.add(folder / f"{prefix}{row['method']}.{row['seed']}.log")
            files.add(folder / f"{prefix}{row['method']}.{row['seed']}.result")
        for path in sorted(files):
            if path.exists(): tar.add(path, arcname=str(path.relative_to(base)))
    for name in ["joint-controls", "joint-driver-controls", "rare-controls"]:
        for path in (base / name).iterdir():
            if path.is_file() and path.suffix != ".so": tar.add(path, arcname=str(path.relative_to(base)))
with tarfile.open(out / "inputs-traces-and-provenance.tar.gz", "w:gz", dereference=True) as tar:
    for family in ["tth2l_b16", "tth2l_b16_all", "tth2l_b25", "xbox2l2m"]:
        inputs = base / "ibp-inputs" / family
        manifest = json.loads((inputs / "manifest.json").read_text())
        for name in ["manifest.json", "validation.json", "joint-trace-validation.json", "trace-abi-validation.json", "suite-cases.txt"]:
            path = inputs / name
            if path.exists(): tar.add(path, arcname=str(path.relative_to(base)))
        for case in manifest["selected_cases"]:
            for suffix in ["", ".variables", ".trace"]:
                path = inputs / (case + suffix)
                tar.add(path, arcname=str(path.relative_to(base)))
        work = base / "ibp-work" / family
        for name in ["selected.trace", "selected.names", "selected.outputs", "top.outputs"]:
            path = work / name
            tar.add(path, arcname=str(path.relative_to(base)))
        if family in ["tth2l_b16_all", "tth2l_b25"]:
            for name in ["preparation-logs", "config", "target-list", "export-equations.yaml", "master-eqns", "integrals"]:
                path = work / name
                tar.add(path, arcname=str(path.relative_to(base)))
sources = [root / "Cargo.toml", root / "examples/reconstruction_joint_benchmark.rs", root / "benches/external/rare/src/main.rs"]
sources += [path for path in (root / "benches/external").iterdir() if path.is_file() and ("joint" in path.name or "trace_oracle" in path.name)]
sources += [root / "benches/external/prepare_ibp_benchmark.py", root / "benches/external/check_rare_adapter.py"]
sources += [root / "target/release/examples/reconstruction_joint_benchmark", root / "target/release/examples/reconstruction_stress_benchmark"]
sources += [base / name for name in ["firefly-joint-stress", "firefly-q-stress", "fire7-q-stress", "rare-q-stress", "libtrace-oracle.so"]]
(out / "sources-and-binaries.sha256").write_text("".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(root)}\n" for p in sources))
summary["limitations"] = ["Complete output sets refer to a single target integral, not a complete IBP reduction.", "Rare's largest b25 coefficient exhausts its full 114-prime table.", "The 128-prime FIRE7 diagnostic is an invalid configuration, superseded by the 114-prime final comparison.", "Pending trace suites are snapshots, not completed evidence."]
(out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
