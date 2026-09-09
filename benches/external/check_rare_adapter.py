#!/usr/bin/env python3
"""Check the rare adapter's exact results, declared variable order and limits."""
import csv
import os
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parents[2]
external = root / "target/reconstruction-external"
output = external / "rare-controls"
output.mkdir(exist_ok=True)
env = {**os.environ, "BENCH_TIMEOUT": "60", "MAX_TOTAL_PROBES": "100000", "MAX_PRIMES": "16"}
loader = ([env["EXTERNAL_LOADER"], "--library-path", env["EXTERNAL_LIBRARY_PATH"]]
          if "EXTERNAL_LOADER" in env else [])


def export(name, names, numerator, denominator):
    path = output / f"{name}.q-oracle"
    with path.open("w") as fp:
        fp.write(f"{len(names)} 0 {len(numerator)} {len(denominator)}\n" + " ".join(names) + "\n")
        for poly in [numerator, denominator]:
            for ex, c in sorted(poly.items(), reverse=True):
                assert len(ex) == len(names)
                fp.write(str(c) + " " + " ".join(map(str, ex)) + "\n")
    return path


cases = {
    "zero": export("zero", ["x", "y"], {}, {(0, 0): 1}),
    "constant": export("constant", ["x", "y"], {(0, 0): -7}, {(0, 0): 11}),
    "univariate": export("univariate", ["d"], {(3,): 13, (0,): 2}, {(1,): 1, (0,): 3}),
    "mixed": export("mixed", ["x", "y"], {(1, 1): 1, (0, 0): 2}, {(1, 1): 1, (1, 0): -2, (0, 0): 4}),
    "large": export("large", ["x", "y"], {(2, 0): 10**50+13, (0, 1): -7, (0, 0): 1}, {(1, 0): 1, (0, 1): 3}),
    "trivariate": export("trivariate", ["x", "y", "z"], {(1,1,0): 3, (0,0,1): 1, (0,0,0): 1}, {(1,1,0): 1, (0,1,1): 1, (1,0,1): 1, (0,0,0): 2}),
    "order": export("order", ["z", "a", "m", "b"], {(2,0,0,1): 5, (0,1,1,0): 10**40+7, (0,0,0,0): 1}, {(0,0,0,0): 3, (1,0,1,0): 1, (0,1,0,1): -2}),
}
for n in range(5, 9):
    names = [f"x{i}" for i in reversed(range(n))]
    numerator = {tuple([0] * n): 11}
    denominator = {tuple([0] * n): 13}
    for i in range(n):
        ex = [0] * n
        ex[i] = 1
        denominator[tuple(ex)] = 2 * i + 3
        ex[(i + 1) % n] = 1
        numerator[tuple(ex)] = 3 * i + 5
    cases[f"variables{n}"] = export(f"variables{n}", names, numerator, denominator)
rows = []


def run(name, seed, limits=None):
    result = subprocess.run([str(external/"rare-q-stress"), str(cases[name]), name, str(seed), str(output/f"{name}.{seed}.result")],
                            cwd=root, env={**env, **(limits or {})}, text=True, capture_output=True, timeout=120)
    label = name + "." + str(seed) + ("." + "-".join(limits) if limits else "")
    (output/f"{label}.log").write_text(result.stdout + result.stderr)
    assert result.returncode == 0, (label, result.stderr)
    lines = result.stdout.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("case,method,"))
    parsed = list(csv.DictReader(lines[start:]))
    assert len(parsed) == 1
    row = parsed[0]
    assert sum(int(pair.split(":")[1]) for pair in row["probes_by_prime"].split(";") if pair) == int(row["probes"])
    assert int(row["primes"]) == len([x for x in row["probes_by_prime"].split(";") if x])
    row["control"] = label
    rows.append(row)
    print(label, row["status"], row["probes"], flush=True)
    return row, result.stderr


for name in cases:
    for seed in [1, 7, 19]:
        row, diagnostic = run(name, seed)
        assert row["status"] == "ok", row
        assert "exact Q identity verified" in diagnostic

for limits, status, probes in [
    ({"MAX_TOTAL_PROBES": "1"}, "probe_limit", 1),
    ({"MAX_PRIMES": "0"}, "prime_limit", 0),
    ({"MAX_PRIMES": "2"}, "prime_limit", None),
    ({"BENCH_TIMEOUT": "0"}, "time_limit", 0),
    ({"RARE_Q_CHECKER": str(output/"missing-checker")}, "identity_check_failed", None),
]:
    row, _ = run("large", 1, limits)
    assert row["status"] == status, row
    if probes is not None:
        assert int(row["probes"]) == probes, row

# The independent checker must reject plausible-looking but incorrect output.
for label, expression in [("wrong", "(0)/(1)"), ("zero_denominator", "(1)/(0)")]:
    result = output/f"{label}.result"
    result.write_text(expression)
    check = subprocess.run(loader + [str(external/"check-q-result"), str(cases["mixed"]), str(result)],
                           text=True, capture_output=True, env=env, timeout=60)
    (output/f"{label}.log").write_text(check.stdout + check.stderr)
    assert check.returncode != 0, label

with (output/"controls.csv").open("w") as fp:
    writer = csv.DictWriter(fp, list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
print(f"{3 * len(cases)} exact reconstructions, 5 adapter failure controls, 2 checker rejection controls passed")
