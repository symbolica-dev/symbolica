#!/usr/bin/env python3
"""Replay the preceding scalar screen after changing row sample streams."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
base = root / "target/reconstruction-external"
groups = {
    "nb0": "fire-table-inputs/all",
    "graph5": "fire-table-inputs/graph5/all",
    "q": "fire-table-inputs",
    "sheared": "fire-table-inputs/sheared",
    "sparse": None,
    **{family: "ibp-inputs/" + family for family in
       ["box2l", "xbox2l2m", "diamond3l", "tth2l_b16"]},
}
summary = {}
for name, inputs in groups.items():
    before = list(csv.DictReader((base / f"rational-factor-{name}.csv").open()))
    assert {r["method"] for r in before} == {"Automatic"}
    cases = list(dict.fromkeys(r["case"] for r in before))
    repeats = max(int(r["seed"]) for r in before)
    destination = base / f"row-stream-{name}.csv"
    env = {**os.environ, "BENCH_METHODS": "Automatic", "RECONSTRUCTION_REPEATS": str(repeats)}
    for key in ["TRACE_ORACLE_DIR", "TRACE_ORACLE_PATH", "RECONSTRUCTION_OVER_Q", "BENCH_INPUT_DIR"]:
        env.pop(key, None)
    if inputs:
        env["BENCH_INPUT_DIR"] = str(base / inputs)
    runner = "run_stress.py" if name == "sparse" else "run_q_stress.py"
    command = [sys.executable, str(root / "benches/external" / runner), str(destination), *cases]
    (base / f"row-stream-{name}.command.json").write_text(json.dumps(command, indent=2) + "\n")
    with (base / f"row-stream-{name}.log").open("w") as log:
        subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    after = list(csv.DictReader(destination.open()))
    key = lambda row: (row["case"], row["method"], row["seed"])
    refs = {key(row): row for row in before}
    assert len(after) == len(before) and set(map(key, after)) == set(refs)
    assert all(row["status"] == "ok" for row in after), name
    deltas = [int(row["probes"]) - int(refs[key(row)]["probes"]) for row in after]
    summary[name] = {
        "exact_successes": len(after),
        "previous_probes": sum(int(row["probes"]) for row in before),
        "current_probes": sum(int(row["probes"]) for row in after),
        "lower": sum(delta < 0 for delta in deltas),
        "equal": sum(delta == 0 for delta in deltas),
        "higher": sum(delta > 0 for delta in deltas),
        "changes": [{"case": row["case"], "seed": row["seed"], "delta": delta}
                    for row, delta in zip(after, deltas) if delta],
    }
    print(name, summary[name], flush=True)
    (base / "row-stream-regressions.json").write_text(json.dumps(summary, indent=2) + "\n")
