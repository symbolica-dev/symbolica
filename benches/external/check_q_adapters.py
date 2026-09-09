#!/usr/bin/env python3
"""Exercise exact Q adapter checks and resource limits on small known inputs."""
import csv
from math import comb
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
external = root / "target/reconstruction-external"
output = external / "q-adapter-controls"
output.mkdir(exist_ok=True)
loader = ([os.environ["EXTERNAL_LOADER"], "--library-path", os.environ["EXTERNAL_LIBRARY_PATH"]]
          if "EXTERNAL_LOADER" in os.environ else [])
env = {**os.environ, "BENCH_TIMEOUT": "60", "MAX_TOTAL_PROBES": "100000", "MAX_PRIMES": "32"}
for key in ["FIRE7_LEARN_BATCH", "FIREFLY_BENCH_SEED"]:
    env.pop(key, None)


def export(name, numerator, denominator, names=("x", "y")):
    path = output / f"{name}.q-oracle"
    with path.open("w") as out:
        out.write(f"{len(names)} 0 {len(numerator)} {len(denominator)}\n" + " ".join(names) + "\n")
        for poly in [numerator, denominator]:
            for exponents, coefficient in sorted(poly.items(), reverse=True):
                assert len(exponents) == len(names)
                out.write(str(coefficient) + " " + " ".join(map(str, exponents)) + "\n")
    return path


cases = {
    "eq3": export("eq3", {(1, 1): 1, (0, 0): 2}, {(1, 1): 1, (1, 0): -2, (0, 0): 4}),
    "large": export("large", {(1, 1): 10**50 + 13, (0, 0): 2}, {(1, 1): 1, (1, 0): -2, (0, 0): 4}),
    "origin_pole": export("origin_pole", {(2, 0): 10**50 + 13, (0, 1): -7, (0, 0): 1}, {(1, 0): 1, (0, 1): 3}),
}
cases["univariate"] = export("univariate", {(3,): 10**40+7, (0,): 2}, {(1,): 1, (0,): 3}, ("d",))
cases["trivariate"] = export("trivariate", {(1,1,0): 3, (0,0,1): 1, (0,0,0): 1},
    {(1,1,0): 1, (0,1,1): 1, (1,0,1): 1, (0,0,0): 2}, ("x","y","z"))
cases["four_variable_order"] = export("four_variable_order", {(2,0,0,1): 5, (0,1,1,0): 10**40+7, (0,0,0,0): 1},
    {(0,0,0,0): 3, (1,0,1,0): 1, (0,1,0,1): -2}, ("z","a","m","b"))
n = {(2*i, j): comb(7, i)*9**(7-i)*comb(30, j)*13**(30-j) for i in range(8) for j in range(31)}
n[0, 0] += 1
d = {(2*i, j): comb(5, i)*(-1)**(5-i)*comb(29, j)*(-4)**(29-j) for i in range(6) for j in range(30)}
cases["eq28"] = export("eq28", n, d)

rows = []
for name, oracle in cases.items():
    for binary, modes in [("firefly-q-stress", ["default", "scan"]), ("fire7-q-stress", ["default", "learned"])]:
        for mode in modes:
            result = subprocess.run(loader + [str(external/binary), str(oracle), name, "1", mode],
                                    env=env, cwd=output, text=True, capture_output=True, timeout=120)
            (output / f"{name}.{binary}.{mode}.log").write_text(result.stdout + result.stderr)
            assert result.returncode == 0, (name, binary, mode, result.stderr)
            lines = result.stdout.splitlines()
            start = next(i for i, line in enumerate(lines) if line.startswith("case,method,"))
            parsed = list(csv.DictReader(lines[start:]))
            assert len(parsed) == 1 and parsed[0]["status"] == "ok", parsed
            row = parsed[0]
            assert sum(int(pair.split(":")[1]) for pair in row["probes_by_prime"].split(";")) == int(row["probes"])
            rows.append(row)
            print(name, row["method"], row["probes"], "exact check passed", flush=True)

# Both adapters must expose exhausted budgets as incomplete outcomes.
for binary in ["firefly-q-stress", "fire7-q-stress"]:
    for label, limits, expected in [
        ("probes", {"MAX_TOTAL_PROBES": "1"}, "probe_limit"),
        ("primes", {"MAX_PRIMES": "2"}, "prime_limit"),
    ]:
        result = subprocess.run(loader + [str(external/binary), str(cases["large"]), "large", "1", "default"],
                                env={**env, **limits}, cwd=output, text=True, capture_output=True, timeout=120)
        (output / f"limit-{label}.{binary}.log").write_text(result.stdout + result.stderr)
        assert result.returncode == 0 and f",{expected}," in result.stdout, result
        print(binary, label, expected, flush=True)

# Reject dimensions that exceed the authors' fixed coefficient-lifting buffers.
names = tuple(f"x{i}" for i in range(17))
unsupported = export("too_many_variables", {(0,)*17: 1}, {(0,)*17: 1}, names)
result = subprocess.run(loader + [str(external/"fire7-q-stress"), str(unsupported), "dimension_limit", "1", "default"],
                        env=env, cwd=output, text=True, capture_output=True, timeout=120)
(output / "limit-variables.fire7-q-stress.log").write_text(result.stdout + result.stderr)
assert result.returncode != 0 and "at most 16 variables" in result.stderr, result
print("fire7-q-stress variables rejected", flush=True)

fields = "case,method,seed,status,elapsed_us,probes,primes,images,probes_by_prime".split(",")
with (output / "results.csv").open("w") as out:
    writer = csv.DictWriter(out, fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
