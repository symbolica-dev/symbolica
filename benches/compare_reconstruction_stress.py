#!/usr/bin/env python3
"""Compare completed stress runs; keep incomplete runs visibly marked."""
import csv
import sys

def read(path):
    with open(path, newline="") as stream:
        return {(r["case"], r["method"], r["seed"], r["order"]): r for r in csv.DictReader(stream)}

before, after = read(sys.argv[1]), read(sys.argv[2])
print("| Case | Method | Seed | Before probes | After probes | Reduction | After ms |")
print("|---|---|---:|---:|---:|---:|---:|")
for key in sorted(after):
    a, b = before.get(key), after[key]
    old = a["probes"] if a and a["status"] == "ok" else (a["status"] if a else "—")
    new = b["probes"] if b["status"] == "ok" else b["status"]
    reduction = f"{100*(1-int(b['probes'])/int(a['probes'])):.1f}%" if a and a["status"] == b["status"] == "ok" else "—"
    time = f"{float(b['elapsed_us'])/1000:.3f}" if b["status"] == "ok" else "—"
    print(f"| {key[0]} | {key[1]} | {key[2]} | {old} | {new} | {reduction} | {time} |")
print("\nTime-limit probe prefixes are deliberately not compared with completed reconstructions.")
