#!/usr/bin/env python3
"""Summarize all methods, including denominator separation, from benchmark CSV."""
import csv
import statistics
import sys
from collections import defaultdict

groups = defaultdict(list)
with open(sys.argv[1], newline="") as stream:
    for row in csv.DictReader(stream):
        groups[row["case"], row["method"]].append(row)

print("| Case | Method | Runs | Probes | ms | Separation fallbacks |")
print("|---|---|---:|---:|---:|---:|")
for (case, method), rows in sorted(groups.items()):
    def median(key):
        return statistics.median(float(row[key]) for row in rows)
    assert len({row["seed"] for row in rows}) == len(rows)
    assert len({row.get("mode", "" ) for row in rows}) == 1
    assert len({row.get("probe_work", "") for row in rows}) == 1
    fallbacks = f"{median('separation_fallbacks'):g}" if "separation_fallbacks" in rows[0] else "—"
    print(f"| {case} | {method} | {len(rows)} | {median('probes'):g} | "
          f"{median('elapsed_us') / 1000:.3f} | {fallbacks} |")
print("\nMedians, including unsuccessful candidates and validation.")
