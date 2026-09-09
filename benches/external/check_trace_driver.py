#!/usr/bin/env python3
"""Ensure FireFly trace failures retain their status and attempted probe count."""
import csv
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
external = root / "target/reconstruction-external"
output = external / "trace-driver-controls"
output.mkdir(exist_ok=True)
source = output / "failure.c"
source.write_text(r'''
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
static uint64_t calls;
void* rr_open(const char* p) { (void)p; calls=0; return &calls; }
void rr_close(void* p) { (void)p; }
size_t rr_inputs(void* p) { (void)p; return 1; }
const char* rr_input_name(void* p, size_t i) { (void)p; (void)i; return "x"; }
uint64_t rr_calls(void* p) { (void)p; return calls; }
int rr_evaluate(void* h, uint64_t p, const uint64_t* x, size_t n, uint64_t* y) {
    (void)h; (void)p; (void)x; (void)n; (void)y; ++calls;
    return atoi(getenv("TRACE_FIXTURE_STATUS"));
}
''')
library = output / "failure.so"
subprocess.run([os.environ.get("CC", "cc"), "-shared", "-fPIC", str(source), "-o", str(library)], check=True)
oracle = output / "constant.q-oracle"
oracle.write_text("1 0 1 1\nx\n1 0\n1 0\n")
loader = ([os.environ["EXTERNAL_LOADER"], "--library-path", os.environ["EXTERNAL_LIBRARY_PATH"]]
          if "EXTERNAL_LOADER" in os.environ else [])
for status, expected in [(1, "trace_pole"), (-1, "trace_error")]:
    result = subprocess.run(loader + [str(external / "firefly-q-stress"), str(oracle), "failure", "1", "scan"],
                            cwd=output, env={**os.environ, "TRACE_ORACLE_PATH": "fixture",
                            "TRACE_ORACLE_LIBRARY": str(library), "TRACE_FIXTURE_STATUS": str(status)},
                            capture_output=True, text=True, timeout=30)
    (output / (expected + ".log")).write_text(result.stdout + result.stderr)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("case,method,"))
    rows = list(csv.DictReader(lines[start:]))
    assert len(rows) == 1 and rows[0]["status"] == expected
    assert rows[0]["probes"] == "1" and rows[0]["primes"] == "1"
    print(expected, "preserved one attempted probe", flush=True)
