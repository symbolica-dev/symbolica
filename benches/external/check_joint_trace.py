#!/usr/bin/env python3
"""Check named joint outputs against separately optimized scalar traces."""
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import random
import sys

if "EXTERNAL_LOADER" in os.environ and "TRACE_ORACLE_CHECK_LOADED" not in os.environ:
    os.execve(os.environ["EXTERNAL_LOADER"], [os.environ["EXTERNAL_LOADER"], "--library-path", os.environ["EXTERNAL_LIBRARY_PATH"], sys.executable, __file__, *sys.argv[1:]], {**os.environ, "TRACE_ORACLE_CHECK_LOADED": "1"})
external = Path(__file__).resolve().parents[2] / "target/reconstruction-external"
family = sys.argv[1]
work = external / "ibp-work" / family
inputs = external / "ibp-inputs" / family
cases = [line.split()[1] for line in (work / "selected.names").read_text().splitlines()]
manifest = json.loads((inputs / "manifest.json").read_text())
names = manifest["variables"]
lib = c.CDLL(str(external / "libtrace-oracle.so"))
for name in ["rr_open", "rr_open_many"]:
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = [c.c_char_p], c.c_void_p
lib.rr_close.argtypes = [c.c_void_p]
for name in ["rr_inputs", "rr_outputs", "rr_calls"]:
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = [c.c_void_p], c.c_uint64
for name in ["rr_input_name", "rr_output_name"]:
    fn = getattr(lib, name)
    fn.argtypes, fn.restype = [c.c_void_p, c.c_size_t], c.c_char_p
lib.rr_evaluate.argtypes = [c.c_void_p, c.c_uint64, c.POINTER(c.c_uint64), c.c_size_t, c.POINTER(c.c_uint64)]
lib.rr_evaluate_many.argtypes = lib.rr_evaluate.argtypes + [c.c_size_t]
trace = work / "selected.trace"
assert len(cases) > 1 and not lib.rr_open(os.fsencode(trace))
joint = lib.rr_open_many(os.fsencode(trace))
scalars = [lib.rr_open(os.fsencode(inputs / (case + ".trace"))) for case in cases]
assert joint and all(scalars)
try:
    assert lib.rr_inputs(joint) == len(names) and lib.rr_outputs(joint) == len(cases)
    assert [lib.rr_input_name(joint, i).decode() for i in range(len(names))] == names
    assert [lib.rr_output_name(joint, i).decode() for i in range(len(cases))] == cases
    rng = random.Random(731)
    calls = finite = poles = 0
    for prime in [97, 1152921504606846883, 2305843009213693967, 9223372036854775783]:
        points = [[rng.randrange(prime) for _ in names] for _ in range(32)] + [[0] * len(names)]
        for values in points:
            point = (c.c_uint64 * len(names))(*values)
            output = (c.c_uint64 * len(cases))()
            status = lib.rr_evaluate_many(joint, prime, point, len(names), output, len(cases))
            assert status in [0, 1]
            calls += 1
            if status == 1:
                poles += 1
            else:
                for i, scalar in enumerate(scalars):
                    value = c.c_uint64()
                    assert lib.rr_evaluate(scalar, prime, point, len(names), c.byref(value)) == 0
                    assert output[i] == value.value
                finite += 1
    assert finite >= 90 and lib.rr_calls(joint) == calls
    point = (c.c_uint64 * len(names))(*([1] * len(names)))
    output = (c.c_uint64 * len(cases))()
    assert lib.rr_evaluate_many(joint, 97, point, len(names), output, len(cases) - 1) == -1
    assert lib.rr_evaluate(joint, 97, point, len(names), output) == -1
    assert lib.rr_evaluate_many(joint, 18446744073709551557, point, len(names), output, len(cases)) == -1
    assert lib.rr_calls(joint) == calls
    report = dict(family=family, cases=cases, joint_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(), finite_vectors=finite, intermediate_poles=poles, interpreter_calls=calls)
    (inputs / "joint-trace-validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report)
finally:
    lib.rr_close(joint)
    for scalar in scalars:
        lib.rr_close(scalar)
