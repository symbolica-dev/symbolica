#!/usr/bin/env python3
"""Run bounded public stress cases; retain failures and time limits in the CSV."""
import csv
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
binary = (root / sys.argv[1]).resolve()
output = root / sys.argv[2]
output.parent.mkdir(parents=True, exist_ok=True)
cases = sys.argv[3:] or ["firefly_f1", "firefly_f2", "firefly_f3", "firefly_f4", "aajamp", "coeff_prop_4l", "mixed_sparse5", "separated_dense4"]
methods = os.environ.get("BENCH_METHODS", "BalancedZippel,BalancedZippelSeparated,CuytLee").split(",")
repeats = int(os.environ.get("RECONSTRUCTION_REPEATS", "1"))
affinity = ["taskset", "-c", os.environ["BENCH_CPU"]] if "BENCH_CPU" in os.environ else []
fields = "case,method,seed,order,status,setup_ms,elapsed_us,probes,attempts,num_terms,den_terms".split(",")
with output.open("w") as stream:
    writer = csv.DictWriter(stream, fields + ["degree_race", "prime", "oracle"], lineterminator="\n")
    writer.writeheader()
    for seed in range(1, repeats + 1):
        for case in cases:
            order = os.environ.get("BENCH_ORDER", "original")
            for method in methods[seed % len(methods):] + methods[:seed % len(methods)]:
                try:
                    result = subprocess.run(affinity + [str(binary),case,method,str(seed),order], cwd=root,
                                            text=True,capture_output=True,timeout=float(os.environ.get("PROCESS_TIMEOUT","180")))
                    output.with_suffix(f".{case}.{method}.{seed}.log").write_text(result.stdout+result.stderr)
                    if result.returncode:
                        row = dict(case=case,method=method,seed=seed,order=order,status=f"exit_{result.returncode}")
                    else:
                        rows = list(csv.DictReader(result.stdout.splitlines()))
                        if len(rows)!=1 or set(rows[0])!=set(fields): raise ValueError(result.stdout)
                        row = rows[0]
                except subprocess.TimeoutExpired:
                    row = dict(case=case,method=method,seed=seed,order=order,status="process_timeout")
                row["degree_race"] = int("RECONSTRUCTION_DEGREE_RACE" in os.environ)
                row["prime"] = os.environ.get("BENCH_PRIME", "2305843009213693951")
                row["oracle"] = "cached_powers" if "CACHED_ORACLE" in os.environ else "expanded"
                writer.writerow(row)
                stream.flush()
                print(case,method,seed,row["status"],row.get("probes",""),flush=True)
