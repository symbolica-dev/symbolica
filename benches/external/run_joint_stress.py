#!/usr/bin/env python3
"""Compare actual vector trace evaluations, retaining output order and failures."""
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys

root = Path(__file__).resolve().parents[2]
external = root / "target/reconstruction-external"
family, destination = sys.argv[1:]
inputs = external / "ibp-inputs" / family
work = external / "ibp-work" / family
output = root / destination
logs = output.parent / (output.stem + "-logs")
logs.mkdir(parents=True, exist_ok=True)
run_lock = (logs / ".run.lock").open("w")
fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
cases = [line.split()[1] for line in (work / "selected.names").read_text().splitlines()]
case_list = logs / "cases.txt"
case_list.write_text("\n".join(cases) + "\n")
manifest = json.loads((inputs / "manifest.json").read_text())
assert set(cases) == set(manifest["selected_cases"])
trace = work / "selected.trace"
env = {**os.environ, "BENCH_INPUT_DIR": str(inputs), "RECONSTRUCTION_OVER_Q": "1",
       "TRACE_ORACLE_PATH": str(trace), "TRACE_ORACLE_LIBRARY": str(external / "libtrace-oracle.so")}
for key in ["TRACE_ORACLE_DIR", "EXPORT_ORACLE"]:
    env.pop(key, None)
loader = [env["EXTERNAL_LOADER"], "--library-path", env["EXTERNAL_LIBRARY_PATH"]] if "EXTERNAL_LOADER" in env else []
affinity = ["taskset", "-c", env["BENCH_CPU"]] if "BENCH_CPU" in env else []
rust_source = root / os.environ.get("SYMBOLICA_JOINT_BINARY", "target/release/examples/reconstruction_joint_benchmark")
firefly_source = root / os.environ.get("FIREFLY_JOINT_BINARY", "target/reconstruction-external/firefly-joint-stress")
library_source = Path(os.environ.get("TRACE_ORACLE_LIBRARY", external / "libtrace-oracle.so"))
snapshots = logs / "binaries"
snapshots.mkdir(exist_ok=True)
rust, firefly, library = [snapshots / name for name in ["symbolica-joint", "firefly-joint", "libtrace-oracle.so"]]
for source, snapshot in [(rust_source, rust), (firefly_source, firefly), (library_source, library)]:
    shutil.copy2(source, snapshot)
env["TRACE_ORACLE_LIBRARY"] = str(library)
for case in cases:
    oracle = logs / (case + ".q-oracle")
    subprocess.run([str(root / "target/release/examples/reconstruction_stress_benchmark"), case, "Automatic", "1"],
                   env={**env, "EXPORT_ORACLE": str(oracle)}, cwd=root, check=True, stdout=subprocess.DEVNULL, timeout=180)
metadata = {"family": family, "cases_in_trace_order": cases, "oracle": "ratracer_joint_trace",
            "baseline": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "max_primes": env.get("MAX_PRIMES", "32"), "max_total_probes": env.get("MAX_TOTAL_PROBES", "2000000"),
            "sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [trace, rust, firefly, library]},
            "binary_sources": [str(p) for p in [rust_source, firefly_source, library_source]],
            "limitations": ["Selected coefficients of one integral, not a complete IBP reduction.",
                            "Symbolica sequential scalar reconstructions share a vector-value cache; FireFly reconstructs outputs jointly.",
                            "Native prime policies differ. Timings exclude trace loading and exact identity checks."]}
(logs / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
fields = "family,method,seed,status,elapsed_us,probes,scalar_requests,cache_hits,outputs,completed,incremental_probes,probes_by_prime".split(",")
methods = os.environ.get("JOINT_METHODS", "Symbolica_joint_cache,FireFly_joint_scan,FireFly_joint_default").split(",")
assert set(methods) <= {"Symbolica_joint_cache", "FireFly_joint_scan", "FireFly_joint_default"}
with output.open("w") as out:
    writer = csv.DictWriter(out, fields, lineterminator="\n")
    writer.writeheader()
    for seed in range(1, int(env.get("RECONSTRUCTION_REPEATS", "1")) + 1):
        for method in methods[seed % len(methods):] + methods[:seed % len(methods)]:
            args = ([str(rust), str(inputs), str(case_list), str(trace), str(seed)] if method == "Symbolica_joint_cache"
                    else [str(firefly), str(logs), str(case_list), str(seed), method.removeprefix("FireFly_joint_")])
            command = affinity + loader + args
            (logs / f"{method}.{seed}.command.json").write_text(json.dumps(command, indent=2) + "\n")
            row = dict(family=family, method=method, seed=seed)
            try:
                result = subprocess.run(command, cwd=logs, env=env, capture_output=True, text=True, timeout=float(env.get("PROCESS_TIMEOUT", "600")))
                (logs / f"{method}.{seed}.log").write_text(result.stdout + result.stderr)
                if result.returncode:
                    row["status"] = f"exit_{result.returncode}"
                else:
                    lines = result.stdout.splitlines()
                    start = next(i for i, line in enumerate(lines) if line.startswith("method,seed,status,"))
                    parsed = list(csv.DictReader(lines[start:]))
                    assert len(parsed) == 1 and parsed[0]["method"] == method
                    row.update(parsed[0])
                    if row.get("probes_by_prime"):
                        assert sum(int(pair.split(":")[1]) for pair in row["probes_by_prime"].split(";")) == int(row["probes"])
                    if row["status"] == "ok":
                        assert int(row["completed"]) == int(row["outputs"]) == len(cases)
            except subprocess.TimeoutExpired as e:
                row["status"] = "process_timeout"
                (logs / f"{method}.{seed}.log").write_bytes((e.stdout or b"") + (e.stderr or b""))
            writer.writerow(row)
            out.flush()
            print(family, method, seed, row["status"], row.get("probes", ""), flush=True)
