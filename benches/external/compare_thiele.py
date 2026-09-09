#!/usr/bin/env python3
"""Interleave before/after Thiele timings on deterministic dense rational functions."""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
external = root / "target/reconstruction-external"
output = root / sys.argv[1]
output.parent.mkdir(parents=True, exist_ok=True)
logs = output.parent / (output.stem + "-logs")
logs.mkdir(parents=True, exist_ok=True)
inputs = external / "thiele-inputs"
inputs.mkdir(exist_ok=True)
cases = []
manifest = {"coefficient_rule": "SHA256('Thiele:20260909:SIDE:POWER'), first 8 bytes big-endian modulo 1000003 minus 500001; replace zero with one", "variables": ["x"], "prime": 9223372036854775783, "method": "BalancedZippel", "entries": []}
for degree in [32, 64, 128, 256, 512]:
    case = f"thiele_dense_{degree}"
    polys = []
    for side in [0, 1]:
        terms = []
        for i in range(degree + 1):
            raw = hashlib.sha256(f"Thiele:20260909:{side}:{i}".encode()).digest()
            c = int.from_bytes(raw[:8], "big") % 1000003 - 500001
            terms.append(f"({c or 1})*x^{i}")
        polys.append("+".join(terms))
    data = f"({polys[0]})/({polys[1]})\n".encode()
    (inputs / case).write_bytes(data)
    (inputs / f"{case}.variables").write_text("x\n")
    manifest["entries"].append({"case": case, "degree": degree, "sha256": hashlib.sha256(data).hexdigest()})
    cases.append(case)
binaries = {
    "baseline": root / os.environ.get("SYMBOLICA_BASELINE", "target/reconstruction-external/division-free-baseline"),
    "current": root / os.environ.get("SYMBOLICA_STRESS_BINARY", "target/release/examples/reconstruction_stress_benchmark"),
}
manifest["binaries"] = {name: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in binaries.items()}
assert manifest["binaries"]["baseline"]["sha256"] != manifest["binaries"]["current"]["sha256"]
(logs / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
env = {**os.environ, "BENCH_INPUT_DIR": str(inputs), "CACHED_ORACLE": "1", "BENCH_PRIME": str(manifest["prime"])}
for key in ["RECONSTRUCTION_OVER_Q", "EXPORT_ORACLE", "RECONSTRUCTION_DEGREE_RACE", "RECONSTRUCTION_DENSE_ROWS"]:
    env.pop(key, None)
affinity = ["taskset", "-c", env["BENCH_CPU"]] if "BENCH_CPU" in env else []
fields = "case,version,seed,status,elapsed_us,probes,setup_ms,num_terms,den_terms".split(",")
with output.open("w") as fp:
    writer = csv.DictWriter(fp, fields, lineterminator="\n")
    writer.writeheader()
    for case in cases:
        for seed in range(1, int(env.get("RECONSTRUCTION_REPEATS", "9")) + 1):
            for version in (["baseline", "current"] if seed % 2 else ["current", "baseline"]):
                row = dict(case=case, version=version, seed=seed)
                log = logs / f"{case}.{version}.{seed}.log"
                try:
                    result = subprocess.run(affinity + [str(binaries[version]), case, "BalancedZippel", str(seed)],
                                            env=env, cwd=root, text=True, capture_output=True,
                                            timeout=float(env.get("PROCESS_TIMEOUT", "180")))
                    log.write_text(result.stdout + result.stderr)
                    if result.returncode:
                        row["status"] = f"exit_{result.returncode}"
                    else:
                        lines = result.stdout.splitlines()
                        start = next(i for i, line in enumerate(lines) if line.startswith("case,method,"))
                        parsed = list(csv.DictReader(lines[start:]))
                        assert len(parsed) == 1
                        row.update({k: parsed[0][k] for k in fields if k not in row})
                except subprocess.TimeoutExpired as e:
                    row["status"] = "process_timeout"
                    log.write_bytes((e.stdout or b"") + (e.stderr or b""))
                writer.writerow(row)
                fp.flush()
                print(case, version, seed, row["status"], row.get("elapsed_us"), flush=True)
