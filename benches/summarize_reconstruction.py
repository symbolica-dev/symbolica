#!/usr/bin/env python3
"""Summarize CSV from reconstruction_benchmark; only Python's standard library."""
import csv
import statistics
import sys
from collections import defaultdict

rows = defaultdict(lambda: defaultdict(list))
with open(sys.argv[1], newline="") as stream:
    reader = csv.DictReader(stream)
    over_q = "primes" in reader.fieldnames
    for row in reader:
        rows[row["case"]][row["method"]].append(row)

extra = " Primes C/B |" if over_q else ""
print("| Case | Cuyt–Lee probes | Balanced probes | Cuyt–Lee µs | Balanced µs | Speedup C/B |" + extra)
print("|---|---:|---:|---:|---:|---:|" + ("---:|" if over_q else ""))
for case, methods in rows.items():
    assert set(methods) == {"CuytLee", "BalancedZippel"}, (case, methods.keys())
    a, b = methods["CuytLee"], methods["BalancedZippel"]
    assert sorted(r["seed"] for r in a) == sorted(r["seed"] for r in b), case
    if not over_q:
        assert len({r["probe_work"] for r in a + b}) == 1, case
    def median(rs, key):
        return statistics.median(float(r[key]) for r in rs)
    ta, tb = median(a, "elapsed_us"), median(b, "elapsed_us")
    extra = f" {median(a, 'primes'):g}/{median(b, 'primes'):g} |" if over_q else ""
    print(f"| {case} | {median(a, 'probes'):g} | {median(b, 'probes'):g} | "
          f"{ta:.1f} | {tb:.1f} | {ta / tb:.2f}× |" + extra)
print("\nMedians across matching seeds; ratios greater than one favor balanced Zippel.")
