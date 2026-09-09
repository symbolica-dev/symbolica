#!/usr/bin/env python3
"""Full Q reconstruction of shared exact inputs; retain prime costs and failures."""
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
cases = sys.argv[2:] or ["coeff_prop_4l", "aajamp"]
rust = root / os.environ.get("SYMBOLICA_STRESS_BINARY", "target/release/examples/reconstruction_stress_benchmark")
affinity = ["taskset", "-c", os.environ["BENCH_CPU"]] if "BENCH_CPU" in os.environ else []
loader = ([os.environ["EXTERNAL_LOADER"], "--library-path", os.environ["EXTERNAL_LIBRARY_PATH"]]
          if "EXTERNAL_LOADER" in os.environ else [])
env = {**os.environ, "RECONSTRUCTION_OVER_Q": "1", "CACHED_ORACLE": "1"}
for key in ["EXPORT_ORACLE", "RECONSTRUCTION_DEGREE_RACE", "RECONSTRUCTION_NO_REUSE", "BENCH_ORDER"]:
    env.pop(key, None)
methods = os.environ.get("BENCH_METHODS", "CuytLeePruned,BalancedZippel,FireFly_default,FireFly_scan").split(",")
assert set(methods) <= {"Automatic", "CuytLee", "CuytLeePruned", "CuytLeePrunedRace", "BalancedZippel", "BalancedZippelRace", "BalancedZippelSeparated", "FireFly_default", "FireFly_scan", "BalancedZippelNoReuse", "CuytLeePrunedNoReuse", "FIRE7_Q", "FIRE7_Q_learned"}
fields = "case,method,seed,status,elapsed_us,probes,primes,images,support_reuses,support_fallbacks,probes_by_prime,prime_policy,oracle,num_terms,den_terms,setup_ms,selected_methods".split(",")
with output.open("w") as out:
    writer = csv.DictWriter(out, fields, lineterminator="\n")
    writer.writeheader()
    for case in cases:
        # Concurrent benchmark invocations must not overwrite one another's input.
        oracle = run_dir / f"{case}.q-oracle"
        subprocess.run([str(rust), case, "BalancedZippel", "1"], cwd=root,
                       env={**env, "EXPORT_ORACLE": str(oracle)}, check=True, timeout=180,
                       stdout=subprocess.DEVNULL)
        with oracle.open() as inp:
            nv, marker, ns, ds = inp.readline().split()
        assert marker == "0"
        for seed in range(1, int(os.environ.get("RECONSTRUCTION_REPEATS", "1")) + 1):
            for method in methods[seed % len(methods):] + methods[:seed % len(methods)]:
                is_fire7 = method.startswith("FIRE7_")
                is_external = method.startswith("FireFly_") or is_fire7
                job_env = {**env, "FIREFLY_BENCH_SEED": str(seed)}
                race = method.endswith("Race")
                if race:
                    job_env["RECONSTRUCTION_DEGREE_RACE"] = "1"
                if method.endswith("NoReuse"):
                    job_env["RECONSTRUCTION_NO_REUSE"] = "1"
                args = (loader + [str(external / "fire7-q-stress"), str(oracle), case, str(seed), "learned" if method.endswith("learned") else "default"]
                        if is_fire7 else loader + [str(external / "firefly-q-stress"), str(oracle), case, str(seed), method.removeprefix("FireFly_")]
                        if is_external else [str(rust), case, method.removesuffix("NoReuse").removesuffix("Race"), str(seed)])
                row = dict(case=case, method=method, seed=seed, oracle="cached_powers_Q",
                           prime_policy="FIRE7_native_64" if is_fire7 else "FireFly_default" if is_external else "Symbolica_default",
                           num_terms=ns, den_terms=ds)
                try:
                    result = subprocess.run(affinity + args, cwd=run_dir if is_external else root,
                                            env=job_env, text=True, capture_output=True,
                                            timeout=float(os.environ.get("PROCESS_TIMEOUT", "600")))
                    (run_dir / f"{case}.{method}.{seed}.log").write_text(result.stdout + result.stderr)
                    if result.returncode:
                        row["status"] = f"exit_{result.returncode}"
                    else:
                        lines = result.stdout.splitlines()
                        start = next(i for i, line in enumerate(lines) if line.startswith("case,method,"))
                        parsed = list(csv.DictReader(lines[start:]))
                        if len(parsed) != 1:
                            raise ValueError(result.stdout)
                        for key in ["status", "elapsed_us", "probes", "primes", "images", "support_reuses", "support_fallbacks", "probes_by_prime", "setup_ms", "selected_methods"]:
                            if key in parsed[0]:
                                row[key] = parsed[0][key]
                except subprocess.TimeoutExpired as e:
                    row["status"] = "process_timeout"
                    (run_dir / f"{case}.{method}.{seed}.log").write_bytes((e.stdout or b"") + (e.stderr or b""))
                writer.writerow(row)
                out.flush()
                print(case, method, seed, row["status"], row.get("probes", ""), flush=True)
