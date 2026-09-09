#!/usr/bin/env python3
"""Exercise output order, vector-cache reuse, pole and budget accounting."""
import csv
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
external = root / "target/reconstruction-external"
out = external / "joint-driver-controls"
out.mkdir(exist_ok=True)
source = out / "fixture.c"
source.write_text(r'''
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
static uint64_t calls;
void* rr_open_many(const char* p) { (void)p; calls=0; return &calls; }
void rr_close(void* p) { (void)p; }
size_t rr_inputs(void* p) { (void)p; return 1; }
size_t rr_outputs(void* p) { (void)p; return 2; }
const char* rr_input_name(void* p, size_t i) { (void)p; return i==0 ? "x" : NULL; }
const char* rr_output_name(void* p, size_t i) { (void)p; return i==0 ? "left" : i==1 ? "right" : NULL; }
uint64_t rr_calls(void* p) { (void)p; return calls; }
int rr_evaluate(void* h, uint64_t p, const uint64_t* x, size_t n, uint64_t* y) { return -1; }
int rr_evaluate_many(void* h, uint64_t p, const uint64_t* x, size_t n, uint64_t* y, size_t m) {
    (void)h; (void)x; (void)p;
    if(n!=1 || m!=2) return -1;
    ++calls;
    int status=atoi(getenv("TRACE_FIXTURE_STATUS"));
    if(status) return status;
    y[0]=1; y[1]=2;
    return 0;
}
''')
library = out / "fixture.so"
subprocess.run([os.environ.get("CC", "cc"), "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
cases = out / "cases.txt"
cases.write_text("left\nright\n")
reverse = out / "reversed.txt"
reverse.write_text("right\nleft\n")
for name, value in [("left", 1), ("right", 2)]:
    (out / name).write_text(str(value))
    (out / (name + ".variables")).write_text("x\n")
    (out / (name + ".q-oracle")).write_text(f"1 0 1 1\nx\n{value} 0\n1 0\n")
loader = [os.environ["EXTERNAL_LOADER"], "--library-path", os.environ["EXTERNAL_LIBRARY_PATH"]] if "EXTERNAL_LOADER" in os.environ else []
rows = []
for method in ["rust", "firefly"]:
    for control in ["valid", "budget", "reversed", "pole", "error"]:
        # A persistent pole has no finite values for Symbolica to reconstruct;
        # cap its oracle work and check that attempted calls remain visible.
        listing = reverse if control == "reversed" else cases
        args = ([str(root / "target/release/examples/reconstruction_joint_benchmark"), str(out), str(listing), "fixture", "1"] if method == "rust"
                else [str(external / "firefly-joint-stress"), str(out), str(listing), "1", "scan"])
        env = {**os.environ, "TRACE_ORACLE_PATH": "fixture", "TRACE_ORACLE_LIBRARY": str(library),
               "TRACE_FIXTURE_STATUS": "1" if control == "pole" else "-1" if control == "error" else "0",
               "MAX_TOTAL_PROBES": "1" if control in ["budget", "pole"] else "10000"}
        result = subprocess.run(loader + args, env=env, cwd=out, capture_output=True, text=True, timeout=30)
        (out / f"{method}-{control}.log").write_text(result.stdout + result.stderr)
        if control == "reversed" or (method == "rust" and control == "error"):
            assert result.returncode != 0
            assert "assertion" in result.stderr if control == "reversed" and method == "rust" else "trace" in result.stderr.lower()
            continue
        assert result.returncode == 0, result.stderr
        lines = result.stdout.splitlines()
        start = next(i for i, line in enumerate(lines) if line.startswith("method,seed,status,"))
        parsed = list(csv.DictReader(lines[start:]))
        assert len(parsed) == 1
        row = parsed[0]
        assert sum(int(pair.split(":")[1]) for pair in row["probes_by_prime"].split(";") if pair) == int(row["probes"])
        expected = "ok" if control == "valid" else "probe_limit" if control == "budget" or method == "rust" else "trace_" + control
        assert row["status"] == expected, row
        if control != "valid":
            assert row["probes"] == "1", row
        else:
            assert row["outputs"] == row["completed"] == "2", row
            if method == "rust":
                assert int(row["cache_hits"]) > 0
                assert int(row["scalar_requests"]) == int(row["probes"]) + int(row["cache_hits"])
        row["control"] = control
        rows.append(row)
        print(method, control, row["status"], row["probes"], flush=True)
(out / "summary.txt").write_text(repr(rows) + "\n")
