#!/usr/bin/env python3
r"""Run prebuilt alias benchmarks sequentially, rotating order between paired rounds.

Example:
  python3 run_alias_performance.py \
    --baseline /tmp/bin/main \
    --global-baseline /tmp/bin/global-baseline \
    --optimized /tmp/bin/optimized \
    --global-baseline-alias /tmp/bin/global-baseline-alias \
    --optimized-alias /tmp/bin/optimized-alias \
    --output-dir /tmp/alias-results

Build binaries before running this script. Defaults are three paired rounds of
seven approximately 100 ms samples per workload. Each round runs all variants
consecutively, rotating the order in each suite. Optional --previous and
--previous-alias binaries add a comparison against an earlier optimization.
The summary compares medians within matching rounds before taking the median ratio.
Negative percentage changes mean less time per operation.
"""

import argparse
import csv
import datetime
import io
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys


INPUT_FIELDS = (
    "workload", "sample", "iterations", "total_ns", "ns_per_iteration", "result_bytes"
)
RAW_FIELDS = ("suite", "variant", "round", "position", *INPUT_FIELDS)
SUMMARY_FIELDS = (
    "suite", "workload", "variant", "samples", "median_ns", "min_round_median_ns",
    "max_round_median_ns", "paired_change_vs_main_pct",
    "paired_change_vs_global_baseline_pct", "paired_change_vs_previous_pct",
)


def binary_path(value):
    path = Path(value).expanduser().resolve()
    if not path.is_file() or not os.access(path, os.X_OK):
        raise argparse.ArgumentTypeError(f"not an executable file: {path}")
    return path


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def affinity_command(cpu):
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else []
    if cpu == "none":
        return [], None, allowed
    if cpu == "auto":
        selected = allowed[0] if allowed else None
    else:
        try:
            selected = int(cpu)
        except ValueError as error:
            raise ValueError("--cpu must be auto, none, or a CPU number") from error
        if selected < 0 or (allowed and selected not in allowed):
            raise ValueError(f"CPU {selected} is outside allowed CPUs: {allowed}")
    taskset = shutil.which("taskset")
    if selected is not None and taskset:
        return [taskset, "--cpu-list", str(selected)], selected, allowed
    if cpu not in ("auto", "none"):
        raise ValueError("explicit CPU pinning requires taskset")
    return [], None, allowed


def parse_samples(output, expected_samples):
    reader = csv.DictReader(io.StringIO(output))
    if tuple(reader.fieldnames or ()) != INPUT_FIELDS:
        raise ValueError(f"unexpected benchmark CSV header: {reader.fieldnames}")
    rows = list(reader)
    seen = {}
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"malformed benchmark row: {row}")
        for field in ("sample", "iterations", "total_ns", "result_bytes"):
            row[field] = int(row[field])
        row["ns_per_iteration"] = float(row["ns_per_iteration"])
        if row["iterations"] <= 0 or row["total_ns"] <= 0:
            raise ValueError(f"nonpositive sample measurement: {row}")
        if not math.isfinite(row["ns_per_iteration"]) or row["ns_per_iteration"] <= 0:
            raise ValueError(f"invalid per-iteration measurement: {row}")
        seen.setdefault(row["workload"], []).append(row["sample"])
    if not rows:
        raise ValueError("benchmark produced no matching workloads")
    for workload, samples in seen.items():
        if sorted(samples) != list(range(expected_samples)):
            raise ValueError(f"unexpected sample numbers for {workload}: {samples}")
    return rows


def summarize(rows):
    grouped = {}
    for row in rows:
        key = row["suite"], row["workload"], row["variant"]
        grouped.setdefault(key, {}).setdefault(row["round"], []).append(row["ns_per_iteration"])
    round_medians = {
        key: {round_number: statistics.median(samples) for round_number, samples in rounds.items()}
        for key, rounds in grouped.items()
    }

    def change(key, reference):
        baseline = round_medians.get((key[0], key[1], reference))
        if baseline is None:
            return ""
        comparisons = [value / baseline[number] for number, value in round_medians[key].items()]
        return 100.0 * (statistics.median(comparisons) - 1.0)

    summary = []
    for key, rounds in grouped.items():
        all_samples = [sample for samples in rounds.values() for sample in samples]
        summary.append(dict(zip(SUMMARY_FIELDS, (
            *key, len(all_samples), statistics.median(all_samples),
            min(round_medians[key].values()), max(round_medians[key].values()),
            change(key, "main"), change(key, "global-baseline"), change(key, "previous"),
        ))))
    return summary


def format_percent(value):
    return "—" if value == "" else f"{value:+.1f}%"


def markdown_summary(rows, rounds, samples, target_ms, cpu):
    lines = [
        f"{rounds} paired rounds × {samples} samples × target {target_ms} ms; "
        + (f"CPU {cpu}." if cpu is not None else "CPU affinity unchanged."),
        "", "Changes use paired round medians; negative values mean faster.", "",
        "| Suite | Workload | Variant | Median ns/op | vs main | vs global baseline | vs previous |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['suite']} | {row['workload']} | {row['variant']} | "
            f"{row['median_ns']:,.1f} | {format_percent(row['paired_change_vs_main_pct'])} | "
            f"{format_percent(row['paired_change_vs_global_baseline_pct'])} | "
            f"{format_percent(row['paired_change_vs_previous_pct'])} |"
        )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--baseline", type=binary_path, required=True, help="main alias_performance binary")
    parser.add_argument("--global-baseline", type=binary_path, required=True, help="merged unoptimized alias_performance binary")
    parser.add_argument("--previous", type=binary_path, help="previous optimized alias_performance binary")
    parser.add_argument("--optimized", type=binary_path, required=True, help="optimized alias_performance binary")
    parser.add_argument("--global-baseline-alias", type=binary_path, help="merged unoptimized global_alias_performance binary")
    parser.add_argument("--previous-alias", type=binary_path, help="previous optimized global_alias_performance binary (requires --previous and both other actual-alias binaries)")
    parser.add_argument("--optimized-alias", type=binary_path, help="optimized global_alias_performance binary")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--samples", type=positive_int, default=7)
    parser.add_argument("--target-ms", type=positive_int, default=100)
    parser.add_argument("--filter", default="", help="workload name substring")
    parser.add_argument("--cpu", default="auto", help="auto (first allowed CPU), none, or a CPU number")
    args = parser.parse_args()
    if bool(args.global_baseline_alias) != bool(args.optimized_alias):
        parser.error("provide both actual-alias binaries, or neither")
    if args.previous_alias and not (args.previous and args.global_baseline_alias):
        parser.error("--previous-alias requires --previous and both other actual-alias binaries")
    try:
        prefix, cpu, allowed_cpus = affinity_command(args.cpu)
    except ValueError as error:
        parser.error(str(error))
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("samples.csv", "summary.csv", "summary.md", "metadata.json"):
        if (output_dir / filename).exists():
            parser.error(f"output already exists: {output_dir / filename}; select a fresh directory")

    ordinary_variants = [("main", args.baseline), ("global-baseline", args.global_baseline)]
    if args.previous:
        ordinary_variants.append(("previous", args.previous))
    ordinary_variants.append(("optimized", args.optimized))
    suites = [("no-aliases", ordinary_variants)]
    if args.global_baseline_alias:
        alias_variants = [("global-baseline", args.global_baseline_alias)]
        if args.previous_alias:
            alias_variants.append(("previous", args.previous_alias))
        alias_variants.append(("optimized", args.optimized_alias))
        suites.append(("live-aliases", alias_variants))
    metadata = {
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "rounds": args.rounds, "samples": args.samples, "target_ms": args.target_ms,
        "filter": args.filter, "cpu": cpu, "allowed_cpus": allowed_cpus,
        "comparison": "median of within-round median ratios", "runs": [],
    }
    environment = dict(os.environ, SYMBOLICA_HIDE_BANNER="1")
    rows = []
    workload_names = {}
    with (output_dir / "samples.csv").open("w", newline="") as raw_file:
        writer = csv.DictWriter(raw_file, fieldnames=RAW_FIELDS)
        writer.writeheader()
        for round_number in range(args.rounds):
            for suite, variants in suites:
                shift = round_number % len(variants)
                order = variants[shift:] + variants[:shift]
                for position, (variant, binary) in enumerate(order):
                    command = [*prefix, str(binary), "--samples", str(args.samples), "--target-ms", str(args.target_ms)]
                    if args.filter:
                        command.extend(["--filter", args.filter])
                    print(f"Round {round_number + 1}/{args.rounds}: {suite}, {variant}", file=sys.stderr, flush=True)
                    completed = subprocess.run(command, env=environment, text=True, capture_output=True)
                    stem = f"round-{round_number + 1}-{suite}-{variant}"
                    (output_dir / f"{stem}.stdout.csv").write_text(completed.stdout)
                    (output_dir / f"{stem}.stderr.log").write_text(completed.stderr)
                    metadata["runs"].append({
                        "round": round_number, "suite": suite, "variant": variant,
                        "position": position, "command": command, "exit_code": completed.returncode,
                    })
                    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
                    if completed.returncode:
                        raise RuntimeError(f"{variant} exited {completed.returncode}; see {stem}.stderr.log")
                    run_rows = parse_samples(completed.stdout, args.samples)
                    names = {row["workload"] for row in run_rows}
                    if suite in workload_names and workload_names[suite] != names:
                        raise ValueError(f"workloads differ between runs in {suite}")
                    workload_names[suite] = names
                    for row in run_rows:
                        row.update(suite=suite, variant=variant, round=round_number, position=position)
                        writer.writerow(row)
                        rows.append(row)
                    raw_file.flush()
    summary = summarize(rows)
    with (output_dir / "summary.csv").open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    markdown = markdown_summary(summary, args.rounds, args.samples, args.target_ms, cpu)
    (output_dir / "summary.md").write_text(markdown)
    metadata["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(markdown, end="")
    print(f"Results: {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
