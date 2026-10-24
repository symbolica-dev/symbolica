#!/usr/bin/env python3
"""Benchmark the unchanged public issue #104 caller and its two closed inputs.

Download gist revision 1e2c24e606cdf95c934e51d6e8168dd842a15390 and build its
Rust caller with the desired local Numerica dependency before running this file.
No input values, precision settings, solver options, or verification bounds are
changed. Mathematica uses the gist's one-thread launcher and original script.
"""

import argparse
import csv
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time


HASHES = {
    1500: "58bc5f61d0e24c54cad864d8fe269a6d1019ad103e205330d021ae3ee6db2dd7",
    1484: "0a1ea3e15a5ff6816ead654cd82b62f842fed9c45110aed29da5804ef7aefcdb",
}


def verify(request, answer, reference):
    vector = [int(value) for value in answer["integer_vector"]]
    if vector != reference or len(vector) != 61 or vector[0] <= 0:
        raise ValueError("The primitive vector differs from the recorded result")
    if math.gcd(*vector) != 1 or max(map(abs, vector)) > int(request["max_coeff"]):
        raise ValueError("The primitive vector violates the coefficient bound")
    inputs = [Fraction(request["target_decimal_prefix"])] + [
        Fraction(value["decimal_prefix"]) for value in request["basis"]
    ]
    residual = abs(sum((a * x for a, x in zip(vector, inputs)), Fraction(0)))
    if residual >= Fraction(request["raw_input_error_tolerance"]):
        raise ValueError("The exact absolute residual check failed")
    if residual / abs(inputs[0]) >= Fraction(request["fit_relative_tolerance"]):
        raise ValueError("The exact target-relative residual check failed")


def timed_run(command, environment, log_path, timeout):
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    started = time.perf_counter()
    with log_path.open("wb") as log:
        process = subprocess.Popen(
            command, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            raise
    wall = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    if code:
        raise RuntimeError(f"Exit code {code}; inspect {log_path}")
    return wall, after.ru_utime - before.ru_utime, after.ru_stime - before.ru_stime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reproducer", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--wolfram", type=Path)
    parser.add_argument("--cpu", type=int, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    reproducer, native = args.reproducer.resolve(), args.native.resolve()
    records = json.loads((reproducer / "recorded-results.json").read_text())["results"]
    requests, references = {}, {}
    for profile, digest in HASHES.items():
        raw = (reproducer / f"request-{profile}.json").read_bytes()
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError(f"Input {profile} differs from the published input")
        requests[profile] = json.loads(raw)
        vectors = [
            [int(value) for value in row["relation_vector"]]
            for row in records if row["physical_digits"] == profile
        ]
        if not vectors or any(vector != vectors[0] for vector in vectors):
            raise ValueError("Inconsistent recorded reference vectors")
        references[profile] = vectors[0]

    args.output.mkdir(parents=True, exist_ok=False)
    environment = os.environ.copy()
    for name in ("RAYON_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        environment[name] = "1"
    if args.wolfram:
        environment["RUSTFLOW_WOLFRAM_KERNEL"] = str(args.wolfram.absolute())
    engines = ["Numerica"] + (["Mathematica"] if args.wolfram else [])
    rows = []
    with (args.output / "runs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "profile", "engine", "repetition", "wall_seconds", "user_seconds", "system_seconds",
        ], lineterminator="\n")
        writer.writeheader()
        for repetition in range(1, args.repeats + 1):
            # Alternate engine order; every measured process uses the same CPU.
            for profile in HASHES:
                for engine in engines if repetition % 2 else reversed(engines):
                    stem = f"{engine.lower()}-{profile}-{repetition}"
                    output = (args.output / f"{stem}.json").resolve()
                    request = reproducer / f"request-{profile}.json"
                    command = ["taskset", "-c", str(args.cpu)]
                    if engine == "Numerica":
                        command += [str(native), str(request), str(output)]
                    else:
                        command += [
                            sys.executable, str(reproducer / "wolfram_one_thread.py"),
                            "-noinit", "-noprompt", "-script", str(reproducer / "discover.wl"),
                            str(request), str(output),
                        ]
                    wall, user, system = timed_run(
                        command, environment, args.output / f"{stem}.log", args.timeout,
                    )
                    verify(requests[profile], json.loads(output.read_text()), references[profile])
                    row = dict(profile=profile, engine=engine, repetition=repetition,
                               wall_seconds=wall, user_seconds=user, system_seconds=system)
                    rows.append(row)
                    writer.writerow(row)
                    stream.flush()
                    print(f"{engine} {profile} run {repetition}: {wall:.3f}s; exact checks passed", flush=True)

    summaries = []
    for profile in HASHES:
        for engine in engines:
            times = [row["wall_seconds"] for row in rows if row["profile"] == profile and row["engine"] == engine]
            summaries.append(dict(profile=profile, engine=engine, median=statistics.median(times),
                                  minimum=min(times), maximum=max(times)))
    summary = dict(platform=platform.platform(), cpu=args.cpu, native=str(native),
                   native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
                   repeats=args.repeats, input_sha256=HASHES, results=summaries)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
