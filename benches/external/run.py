#!/usr/bin/env python3
"""Run checked external comparisons; stop on any failed reconstruction."""
import csv
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL = ROOT / "target/reconstruction-external"
OUT = ROOT / "benches/results/reconstruction/improved"
OUT.mkdir(parents=True, exist_ok=True)
RUN = EXTERNAL / "runs"
RUN.mkdir(exist_ok=True)
repeats = int(os.environ.get("RECONSTRUCTION_REPEATS", "9"))
affinity = ["taskset", "-c", os.environ["BENCH_CPU"]] if "BENCH_CPU" in os.environ else []
# For Nix installations with a newer FLINT libc than the compiler's default.
loader = []
if os.environ.get("EXTERNAL_LOADER"):
    loader = [os.environ["EXTERNAL_LOADER"], "--library-path", os.environ["EXTERNAL_LIBRARY_PATH"]]

for mode in ["ff", "q"]:
    with (OUT / f"external-symbolica-{mode}.csv").open("w") as out:
        subprocess.run(affinity + [str(ROOT / "target/release/examples/reconstruction_external_benchmark"), mode],
                       cwd=RUN, stdout=out, check=True, timeout=300)

jobs = [(case, "firefly", mode, config) for case in ["eq3", "eq28"]
        for mode, config in [("ff", "default"), ("q", "default"), ("q", "scan")]]
jobs += [(case, "fire7", "ff", "balanced_adapter") for case in ["eq3", "eq28"]]
with (OUT / "external-cpp.csv").open("w") as out:
    writer = csv.writer(out)
    writer.writerow(["case", "method", "mode", "seed", "elapsed_us", "probes"])
    for seed in range(repeats + 1):
        for case, engine, mode, config in jobs[seed % len(jobs):] + jobs[:seed % len(jobs)]:
            args = [case, mode, config, str(seed)] if engine == "firefly" else [case, str(seed)]
            result = subprocess.run(affinity + loader + [str(EXTERNAL / f"{engine}-bench")] + args,
                                    env={**os.environ, "FIREFLY_BENCH_SEED": str(seed)},
                                    cwd=RUN, text=True, capture_output=True, timeout=300)
            (RUN / f"{engine}-{case}-{mode}-{config}-{seed}.log").write_text(result.stdout + result.stderr)
            result.check_returncode()
            rows = [row for row in csv.reader(result.stdout.splitlines()) if len(row) == 6 and row[0].startswith("paper_")]
            if len(rows) != 1:
                sys.exit(f"Invalid CSV from {engine}: {result.stdout}")
            if seed:
                writer.writerow(rows[0])
                out.flush()
