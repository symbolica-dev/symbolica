#!/usr/bin/env python3
"""Compare the shared Ratracer ABI to expanded exact inputs over several fields."""
import argparse
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys

# A dynamically loaded FLINT must use the same libc as its Python host.
if "EXTERNAL_LOADER" in os.environ and "TRACE_ORACLE_CHECK_LOADED" not in os.environ:
    os.execve(os.environ["EXTERNAL_LOADER"],
              [os.environ["EXTERNAL_LOADER"], "--library-path", os.environ["EXTERNAL_LIBRARY_PATH"],
               sys.executable, __file__, *sys.argv[1:]],
              {**os.environ, "TRACE_ORACLE_CHECK_LOADED": "1"})

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("family")
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
external = root / "target/reconstruction-external"
inputs = external / "ibp-inputs" / args.family
manifest = json.loads((inputs / "manifest.json").read_text())
lib = c.CDLL(str(external / "libtrace-oracle.so"))
lib.rr_open.argtypes, lib.rr_open.restype = [c.c_char_p], c.c_void_p
lib.rr_close.argtypes = [c.c_void_p]
lib.rr_inputs.argtypes, lib.rr_inputs.restype = [c.c_void_p], c.c_size_t
lib.rr_input_name.argtypes, lib.rr_input_name.restype = [c.c_void_p, c.c_size_t], c.c_char_p
lib.rr_calls.argtypes, lib.rr_calls.restype = [c.c_void_p], c.c_uint64
lib.rr_evaluate.argtypes = [c.c_void_p, c.c_uint64, c.POINTER(c.c_uint64), c.c_size_t, c.POINTER(c.c_uint64)]
lib.rr_evaluate.restype = c.c_int
rng = random.Random(0x5452414345)
report = []
for entry in manifest["entries"]:
    if "case" not in entry:
        continue
    case = entry["case"]
    trace = inputs / (case + ".trace")
    assert hashlib.sha256(trace.read_bytes()).hexdigest() == entry["trace_sha256"]
    oracle = inputs / (case + ".check-oracle")
    subprocess.run([str(root / "target/release/examples/reconstruction_stress_benchmark"), case, "Automatic", "1"],
                   env={**os.environ, "BENCH_INPUT_DIR": str(inputs), "EXPORT_ORACLE": str(oracle), "RECONSTRUCTION_OVER_Q": "1"},
                   cwd=root, check=True, stdout=subprocess.DEVNULL, timeout=180)
    lines = oracle.read_text().splitlines()
    nv, marker, ns, ds = map(int, lines[0].split())
    names = lines[1].split()
    assert marker == 0 and len(names) == nv and len(lines) == ns + ds + 2
    terms = [list(map(int, line.split())) for line in lines[2:]]
    handle = lib.rr_open(os.fsencode(trace))
    assert handle
    try:
        assert lib.rr_inputs(handle) == nv
        assert [lib.rr_input_name(handle, i).decode() for i in range(nv)] == names
        comparisons, poles, removable, calls = 0, 0, 0, 0
        for prime in [97, 1152921504606846883, 2305843009213693967, 9223372036854775783]:
            points = [[rng.randrange(prime) for _ in names] for _ in range(16)]
            points += [[0] * nv]
            for i in range(nv):
                point = [rng.randrange(1, prime) for _ in names]
                point[i] = 0
                points.append(point)
            for point in points:
                def polynomial(rows):
                    value = 0
                    for row in rows:
                        term = row[0] % prime
                        for x, exponent in zip(point, row[1:]):
                            term = term * pow(x, exponent, prime) % prime
                        value = (value + term) % prime
                    return value
                numerator, denominator = polynomial(terms[:ns]), polynomial(terms[ns:])
                output = c.c_uint64()
                status = lib.rr_evaluate(handle, prime, (c.c_uint64 * nv)(*point), nv, c.byref(output))
                calls += 1
                if status == 1:
                    poles += 1
                    removable += denominator != 0
                else:
                    assert status == 0 and denominator != 0
                    assert output.value == numerator * pow(denominator, -1, prime) % prime
                    comparisons += 1
        assert comparisons >= 48 and poles > 0
        assert lib.rr_calls(handle) == calls
        # Native FIRE7's 64-bit primes must be explicitly rejected.
        assert lib.rr_evaluate(handle, 18446744073709551557, (c.c_uint64 * nv)(*([1] * nv)), nv, c.byref(c.c_uint64())) == -1
        assert lib.rr_calls(handle) == calls
        report.append(dict(case=case, finite_agreements=comparisons, intermediate_poles=poles, removable_poles=removable, interpreter_calls=calls))
        print(report[-1], flush=True)
    finally:
        lib.rr_close(handle)
(inputs / "trace-abi-validation.json").write_text(json.dumps(report, indent=2) + "\n")
