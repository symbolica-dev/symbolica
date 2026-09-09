#!/usr/bin/env python3
"""Matched modular stress runs against actual FireFly; keep incomplete outcomes."""
import csv
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
external = root / "target/reconstruction-external"
output = root / sys.argv[1]
output.parent.mkdir(parents=True, exist_ok=True)
run_dir = output.parent / (output.stem + "-logs")
run_dir.mkdir(exist_ok=True)
cases = sys.argv[2:] or ["firefly_f1", "firefly_f2", "firefly_f3", "firefly_f4", "coeff_prop_4l", "aajamp"]
rust = root / os.environ.get("SYMBOLICA_STRESS_BINARY", "target/release/examples/reconstruction_stress_benchmark")
affinity = ["taskset", "-c", os.environ["BENCH_CPU"]] if "BENCH_CPU" in os.environ else []
loader = ([os.environ["EXTERNAL_LOADER"], "--library-path", os.environ["EXTERNAL_LIBRARY_PATH"]]
          if "EXTERNAL_LOADER" in os.environ else [])
env = {**os.environ, "BENCH_PRIME": "9223372036854775783", "CACHED_ORACLE": "1"}
env.pop("EXPORT_ORACLE", None)
env.pop("RECONSTRUCTION_DEGREE_RACE", None)
env.pop("BENCH_ORDER", None)
env.pop("FIRE7_LEARN_BATCH", None)
methods = os.environ.get("BENCH_METHODS", "BalancedZippel,FireFly_default").split(",")
assert set(methods) <= {"Automatic", "BalancedZippel", "BalancedZippelSeparated", "CuytLee", "CuytLeePruned", "CuytLeePrunedRace", "BalancedZippelRace", "FireFly_default", "FIRE7_balanced_adapter", "FIRE7_learned_batch"}
fields = "case,method,seed,status,elapsed_us,probes,prime,oracle,degree_race,num_terms,den_terms,setup_ms,selected_methods,selection_probes".split(",")
with output.open("w") as out:
    writer = csv.DictWriter(out, fields, lineterminator="\n")
    writer.writeheader()
    for case in cases:
        oracle = external / f"{case}.oracle"
        subprocess.run([str(rust), case, "BalancedZippel", "1"], cwd=root,
                       env={**env, "EXPORT_ORACLE": str(oracle)}, check=True, timeout=120,
                       stdout=subprocess.DEVNULL)
        with oracle.open() as inp:
            nv, prime, ns, ds = inp.readline().split()
        assert prime == env["BENCH_PRIME"]
        for seed in range(1, int(os.environ.get("RECONSTRUCTION_REPEATS", "1")) + 1):
            for method in methods[seed % len(methods):] + methods[:seed % len(methods)]:
                if method.startswith("FIRE7_") and nv != "2":
                    continue
                external_method = method == "FireFly_default" or method.startswith("FIRE7_")
                race = method in {"BalancedZippelRace", "CuytLeePrunedRace"}
                job_env = {**env, "FIREFLY_BENCH_SEED": str(seed)}
                if race:
                    job_env["RECONSTRUCTION_DEGREE_RACE"] = "1"
                if method == "FIRE7_learned_batch":
                    job_env["FIRE7_LEARN_BATCH"] = "1"
                binary = "firefly-stress" if method == "FireFly_default" else "fire7-stress"
                args = (loader + [str(external / binary), str(oracle), case, str(seed)]
                        if external_method else
                        [str(rust), case, method.removesuffix("Race") if race else method, str(seed)])
                row = dict(case=case, method=method, seed=seed, prime=prime, oracle="cached_powers",
                           degree_race=int(race), num_terms=ns, den_terms=ds)
                try:
                    result = subprocess.run(affinity + args, cwd=run_dir if external_method else root, env=job_env, text=True,
                                            capture_output=True, timeout=float(os.environ.get("PROCESS_TIMEOUT", "240")))
                    (run_dir / f"{case}.{method}.{seed}.log").write_text(result.stdout + result.stderr)
                    if result.returncode:
                        row["status"] = f"exit_{result.returncode}"
                    else:
                        # FireFly may print diagnostics before the CSV header.
                        lines = result.stdout.splitlines()
                        start = next(i for i, line in enumerate(lines) if line.startswith("case,method,"))
                        parsed = list(csv.DictReader(lines[start:]))
                        if len(parsed) != 1:
                            raise ValueError(result.stdout)
                        for key in ["status", "elapsed_us", "probes", "setup_ms", "selected_methods", "selection_probes"]:
                            if key in parsed[0]:
                                row[key] = parsed[0][key]
                except subprocess.TimeoutExpired as e:
                    row["status"] = "process_timeout"
                    (run_dir / f"{case}.{method}.{seed}.log").write_bytes((e.stdout or b"") + (e.stderr or b""))
                writer.writerow(row)
                out.flush()
                print(case, method, seed, row["status"], row.get("probes", ""), flush=True)
