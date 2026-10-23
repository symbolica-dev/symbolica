#!/usr/bin/env python3
"""Compare prebuilt binaries with adjacent per-workload measurements.

Interleave variants for each workload, preserving raw samples and paired ratios. Workload
names are passed as full --filter strings and the returned workload is checked
for exact equality, since the Rust harness implements substring filtering.
With --quiet, retry complete variant groups using predeclared interference
criteria only. Rejected raw output and telemetry remain available for audit.
"""
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

DEFAULT_WORKLOADS = {
    "expansion": ["expand_power_small", "expand_product_small", "expand_power_large", "expand_product_large_repeated"],
    "ordinary": ["construct", "add_pair", "add_chain", "add_many", "normalize_product", "expand_power", "expand_product", "differentiate", "replace"],
    "live": ["construct_with_live_alias", "copy_alias_power", "add_alias_plain", "multiply_alias_powers", "copy_large_alias_power", "multiply_large_alias_powers", "copy_repeated_multi_aliases", "copy_alias_child_from_multi_owner"],
    "keys": ["hash_small", "hash_large", "hash_view_small", "hash_view_large", "eq_small_equal", "eq_large_equal", "eq_large_unequal_early", "eq_large_unequal_late", "eq_view_small_equal", "eq_view_large_equal", "ord_owned_small_equal", "ord_owned_large_equal", "ord_owned_large_unequal_late"],
    "alias-keys": ["hash_plain_body", "hash_transparent_alias", "hash_nested_plain", "hash_nested_alias", "eq_plain_body", "eq_transparent_alias_plain", "eq_nested_plain", "eq_nested_alias_plain"],
}
CPU_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal", "guest", "guest_nice")
RUSAGE_FIELDS = ("ru_utime", "ru_stime", "ru_minflt", "ru_majflt", "ru_inblock", "ru_oublock", "ru_nvcsw", "ru_nivcsw")
QUIET_CRITERIA = {
    "pre_group_window_seconds": 1.0,
    "pre_group_cpu_busy_fraction_exclusive_max": 0.15,
    "pre_group_wait_timeout_seconds": 900.0,
    "run_process_cpu_fraction_min": 0.95,
    "run_sibling_busy_fraction_max": 0.20,
    "rustc_pids_before_and_after": 0,
    "max_group_attempts": 10,
    "selection": "Accept or reject an entire workload variant group using telemetry only; never use operation timings.",
    "limitation": "Compiler detection samples /proc before and after each observation/run; a compiler that starts and exits between samples may be missed.",
}


def read_text(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cpu_list(value):
    values = []
    for token in (value or "").split(","):
        if not token:
            continue
        ends = [int(x) for x in token.split("-")]
        values.extend(range(ends[0], ends[-1] + 1))
    return values


def cpu_info(cpu):
    base = Path(f"/sys/devices/system/cpu/cpu{cpu}")
    return {
        "cpu": cpu,
        "allowed_cpus": sorted(os.sched_getaffinity(0)),
        "siblings": cpu_list(read_text(base / "topology/thread_siblings_list")),
        "core_id": read_text(base / "topology/core_id"),
        "package_id": read_text(base / "topology/physical_package_id"),
        "llc_shared_cpu_list": read_text(base / "cache/index3/shared_cpu_list"),
        "driver": read_text(base / "cpufreq/scaling_driver"),
        "governor": read_text(base / "cpufreq/scaling_governor"),
        "energy_performance_preference": read_text(base / "cpufreq/energy_performance_preference"),
        "scaling_min_freq": read_text(base / "cpufreq/scaling_min_freq"),
        "scaling_max_freq": read_text(base / "cpufreq/scaling_max_freq"),
        "boost": read_text(base / "cpufreq/boost"),
        "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
    }


def snapshot(cpu, siblings):
    stats = {}
    try:
        for line in Path("/proc/stat").read_text().splitlines():
            fields = line.split()
            if fields and fields[0].startswith("cpu") and fields[0][3:].isdigit():
                number = int(fields[0][3:])
                if number in siblings:
                    stats[str(number)] = dict(zip(CPU_FIELDS, map(int, fields[1:])))
    except OSError:
        pass
    return {
        "cpu_ticks": stats,
        "instantaneous_scaling_cur_freq": read_text(f"/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"),
        "loadavg": read_text("/proc/loadavg"),
        "cpu_pressure": read_text("/proc/pressure/cpu"),
    }


def cpu_deltas(before, after):
    result = {}
    for cpu in before["cpu_ticks"].keys() & after["cpu_ticks"].keys():
        delta = {field: after["cpu_ticks"][cpu][field] - before["cpu_ticks"][cpu][field] for field in CPU_FIELDS}
        # Linux guest time is already counted in user/nice. Do not count it twice.
        total = sum(delta[field] for field in CPU_FIELDS[:8])
        busy = total - delta["idle"] - delta["iowait"]
        result[cpu] = {"ticks": delta, "busy_fraction": busy / total if total else None}
    return result


def rustc_pids():
    pids = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit() and read_text(entry / "comm") == "rustc":
            pids.append(int(entry.name))
    return sorted(pids)


def wait_for_quiet(cpu, siblings, journal, group):
    started = time.monotonic()
    deadline = started + QUIET_CRITERIA["pre_group_wait_timeout_seconds"]
    checks = 0
    while True:
        if time.monotonic() + QUIET_CRITERIA["pre_group_window_seconds"] > deadline:
            raise TimeoutError(f"No quiet CPU window within 15 minutes for {group}")
        compilers_before = rustc_pids()
        before = snapshot(cpu, siblings)
        time.sleep(QUIET_CRITERIA["pre_group_window_seconds"])
        after = snapshot(cpu, siblings)
        compilers_after = rustc_pids()
        deltas = cpu_deltas(before, after)
        quiet = not compilers_before and not compilers_after and all(
            str(number) in deltas
            and deltas[str(number)]["busy_fraction"] is not None
            and deltas[str(number)]["busy_fraction"] < QUIET_CRITERIA["pre_group_cpu_busy_fraction_exclusive_max"]
            for number in siblings
        )
        checks += 1
        check = {
            **group, "check": checks, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - started, "quiet": quiet,
            "rustc_before": compilers_before, "rustc_after": compilers_after,
            "cpu_deltas": deltas, "before": before, "after": after,
        }
        journal.write(json.dumps(check) + "\n")
        journal.flush()
        if quiet:
            return {"checks": checks, "elapsed_seconds": check["elapsed_seconds"], "passed_check": check}
        if checks == 1 or checks % 30 == 0:
            print(f"Waiting for quiet CPUs {siblings} and no rustc ({checks}s): {group}", file=sys.stderr, flush=True)


def interference_reasons(run, cpu, siblings):
    reasons = []
    if run["process_cpu_fraction"] < QUIET_CRITERIA["run_process_cpu_fraction_min"]:
        reasons.append("process CPU/wall below 0.95")
    for number in siblings:
        if number == cpu:
            continue
        busy = run["cpu_deltas"].get(str(number), {}).get("busy_fraction")
        if busy is None:
            reasons.append(f"CPU {number} sibling telemetry unavailable")
        elif busy > QUIET_CRITERIA["run_sibling_busy_fraction_max"]:
            reasons.append(f"CPU {number} sibling busy above 0.20")
    if run["rustc_before"] or run["rustc_after"]:
        reasons.append("rustc present before or after run")
    return reasons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--target-ms", type=int, default=100)
    parser.add_argument("--cpu", type=int, default=6)
    parser.add_argument("--quiet", action="store_true", help="retry whole variant groups when CPU/compiler telemetry exceeds predeclared limits")
    parser.add_argument("--suite", action="append", required=True, help="suite:variant=path,variant=path")
    parser.add_argument("--workloads", action="append", default=[], help="override suite:workload,workload")
    parser.add_argument("--filter", default="", help="select workload names containing this substring")
    parser.add_argument("--runner", type=Path, default=Path(__file__).with_name("run_alias_performance.py"))
    args = parser.parse_args()
    if min(args.rounds, args.samples, args.target_ms) <= 0:
        parser.error("rounds, samples, and target-ms must be positive")
    spec = importlib.util.spec_from_file_location("alias_benchmark_parser", args.runner)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    try:
        prefix, cpu, allowed = runner.affinity_command(str(args.cpu))
        suites = {}
        for entry in args.suite:
            suite, variants = entry.split(":", 1)
            if suite in suites:
                raise ValueError(f"duplicate suite: {suite}")
            suites[suite] = [(name, runner.binary_path(path)) for name, path in (part.split("=", 1) for part in variants.split(","))]
            names = [name for name, _ in suites[suite]]
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate variant names in {suite}")
            if args.rounds % len(names):
                raise ValueError(f"rounds must be a multiple of the number of {suite} variants")
        workloads = dict(DEFAULT_WORKLOADS)
        for entry in args.workloads:
            suite, names = entry.split(":", 1)
            workloads[suite] = names.split(",")
        workloads = {suite: [name for name in workloads[suite] if args.filter in name] for suite in suites}
        if any(not names for names in workloads.values()):
            raise ValueError("a suite has no selected workloads")
        for suite, names in workloads.items():
            if len(names) != len(set(names)):
                raise ValueError(f"duplicate workloads in {suite}")
            if any(name != other and name in other for name in names for other in names):
                raise ValueError(f"ambiguous substring filters within {suite}")
    except (ValueError, KeyError, argparse.ArgumentTypeError) as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=False)
    info = cpu_info(cpu)
    siblings = sorted(set([cpu, *info["siblings"]]))
    meta = {
        "invocation": [sys.executable, *sys.argv],
        "rounds": args.rounds, "samples": args.samples, "target_ms": args.target_ms,
        "cpu": cpu, "filter": args.filter, "workloads": workloads,
        "order": "round, suite, workload, variants rotated by round plus workload index",
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "binary_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for variants in suites.values() for _, path in variants},
        "runner_sha256": hashlib.sha256(args.runner.read_bytes()).hexdigest(),
        "cpu_info": info, "runs": [],
        "quiet": args.quiet, "quiet_criteria": QUIET_CRITERIA if args.quiet else None,
        "diagnostics_note": "Process rusage and elapsed time include initialization and calibration. CPU snapshots include other processes on the sampled logical CPUs. Instantaneous frequency is not an average. With --quiet only accepted complete groups enter samples.csv; every rejected run retains raw output and telemetry. No operation timings inform acceptance and no measurements are rescaled.",
    }
    (args.out / "reproduce.py").write_text(Path(__file__).read_text())
    (args.out / "run_alias_performance.py").write_text(args.runner.read_text())
    rows = []
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    # A complete group gets one acceptance decision. Per-attempt raw files are
    # written immediately; final run records are journaled after that decision.
    with (args.out / "samples.csv").open("w", newline="") as output, (args.out / "runs.jsonl").open("w") as journal, (args.out / "quiet_checks.jsonl").open("w") as quiet_journal:
        writer = csv.DictWriter(output, fieldnames=runner.RAW_FIELDS)
        writer.writeheader()
        output.flush()
        try:
            for number in range(args.rounds):
                for suite, variants in suites.items():
                    for workload_index, workload in enumerate(workloads[suite]):
                        shift = (number + workload_index) % len(variants)
                        ordered = variants[shift:] + variants[:shift]
                        max_attempts = QUIET_CRITERIA["max_group_attempts"] if args.quiet else 1
                        for attempt in range(1, max_attempts + 1):
                            group = {"round": number, "suite": suite, "workload": workload, "attempt": attempt}
                            quiet_wait = wait_for_quiet(cpu, siblings, quiet_journal, group) if args.quiet else None
                            group_runs = []
                            group_samples = []
                            group_error = None
                            try:
                                for position, (name, path) in enumerate(ordered):
                                    command = [*prefix, str(path), "--samples", str(args.samples), "--target-ms", str(args.target_ms), "--filter", workload]
                                    print(f"Round {number + 1}/{args.rounds}: {suite}, {workload}, {name}, attempt {attempt}", file=sys.stderr, flush=True)
                                    compilers_before = rustc_pids() if args.quiet else None
                                    before = snapshot(cpu, siblings)
                                    usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
                                    started_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
                                    started_ns = time.monotonic_ns()
                                    completed = subprocess.run(command, env=dict(os.environ, SYMBOLICA_HIDE_BANNER="1"), capture_output=True, text=True)
                                    elapsed_seconds = (time.monotonic_ns() - started_ns) / 1e9
                                    usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
                                    after = snapshot(cpu, siblings)
                                    compilers_after = rustc_pids() if args.quiet else None
                                    usage = {field: getattr(usage_after, field) - getattr(usage_before, field) for field in RUSAGE_FIELDS}
                                    process_seconds = usage["ru_utime"] + usage["ru_stime"]
                                    stem = f"{suite}-{number + 1}-{workload}-{name}" + (f"-attempt-{attempt}" if args.quiet else "")
                                    stdout_file = f"{stem}.stdout.csv"
                                    stderr_file = f"{stem}.stderr.log"
                                    (args.out / stdout_file).write_text(completed.stdout)
                                    (args.out / stderr_file).write_text(completed.stderr)
                                    run = {
                                        **group, "variant": name, "position": position, "command": command,
                                        "started_utc": started_utc, "stdout_file": stdout_file, "stderr_file": stderr_file,
                                        "exit_code": completed.returncode, "elapsed_seconds": elapsed_seconds,
                                        "rusage": usage, "process_cpu_seconds": process_seconds,
                                        "process_cpu_fraction": process_seconds / elapsed_seconds,
                                        "before": before, "after": after, "cpu_deltas": cpu_deltas(before, after),
                                        "rustc_before": compilers_before, "rustc_after": compilers_after,
                                    }
                                    run["interference_reasons"] = interference_reasons(run, cpu, siblings) if args.quiet else []
                                    group_runs.append(run)
                                    completed.check_returncode()
                                    samples = runner.parse_samples(completed.stdout, args.samples)
                                    if {sample["workload"] for sample in samples} != {workload}:
                                        raise ValueError(f"filter {workload!r} returned unexpected workloads")
                                    for sample in samples:
                                        sample.update(suite=suite, variant=name, round=number, position=position)
                                        group_samples.append(sample)
                            except BaseException as error:
                                group_error = f"{type(error).__name__}: {error}"
                                raise
                            finally:
                                rejected = [
                                    {"variant": run["variant"], "reasons": run["interference_reasons"]}
                                    for run in group_runs if run["interference_reasons"]
                                ]
                                accepted = group_error is None and len(group_runs) == len(ordered) and not rejected
                                for run in group_runs:
                                    run.update(accepted=accepted, group_rejections=rejected, group_error=group_error, quiet_wait=quiet_wait)
                                    meta["runs"].append(run)
                                    journal.write(json.dumps(run) + "\n")
                                journal.flush()
                            if accepted:
                                for sample in group_samples:
                                    writer.writerow(sample)
                                    rows.append(sample)
                                output.flush()
                                break
                            print(f"Rejected complete group {group}: {rejected}", file=sys.stderr, flush=True)
                        else:
                            raise RuntimeError(f"No acceptable complete group after {max_attempts} attempts: {suite}, {workload}, round {number + 1}")
                (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        except BaseException as error:
            meta["aborted_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            meta["error"] = f"{type(error).__name__}: {error}"
            (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
            raise
    summary = []
    for suite, variants in suites.items():
        names = [name for name, _ in variants]
        for workload in workloads[suite]:
            data = [row for row in rows if row["suite"] == suite and row["workload"] == workload]
            medians = {(name, number): statistics.median(row["ns_per_iteration"] for row in data if row["variant"] == name and row["round"] == number) for name in names for number in range(args.rounds)}
            item = {"suite": suite, "workload": workload, "median_ns": {name: statistics.median(medians[name, number] for number in range(args.rounds)) for name in names}, "paired_ratios": {}}
            for index, name in enumerate(names):
                for reference in names[:index]:
                    ratios = [medians[name, number] / medians[reference, number] for number in range(args.rounds)]
                    item["paired_ratios"][f"{name}/{reference}"] = {"median": statistics.median(ratios), "min": min(ratios), "max": max(ratios)}
            item["result_bytes"] = {name: sorted({row["result_bytes"] for row in data if row["variant"] == name}) for name in names}
            summary.append(item)
            print(suite, workload, json.dumps(item["median_ns"]), json.dumps({key: round((value["median"] - 1) * 100, 2) for key, value in item["paired_ratios"].items()}))
        for pair in summary[-1]["paired_ratios"]:
            ratios = [item["paired_ratios"][pair]["median"] for item in summary if item["suite"] == suite]
            print(f"{suite} geomean {pair}: {(math.prod(ratios) ** (1 / len(ratios)) - 1) * 100:+.2f}%")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    meta["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta["cpu_info_at_finish"] = cpu_info(cpu)
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (args.out / "reproduce.py").write_text(Path(__file__).read_text())
    (args.out / "run_alias_performance.py").write_text(args.runner.read_text())


if __name__ == "__main__":
    main()
