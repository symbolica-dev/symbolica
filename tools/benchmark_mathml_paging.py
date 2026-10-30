#!/usr/bin/env python3
"""Repeatable native/Pyodide-backend paging timings, excluding browser layout.

Run with the same Python build and arguments before and after a change. Optional
real tensor cases use the two captured diagram JSON fixtures; no notebook or
widget dependencies are loaded. Example:

    python tools/benchmark_mathml_paging.py --iterations 30 --output before.json \
        --tensor-fixtures /path/to/compact-display-wasm-repro
"""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time

from symbolica import E, S


def fixtures(tensor_directory):
    yield "core-sum", E("(a+b+c+d)^20").expand(), False
    yield "core-product", E("*".join(f"f{i}((a+b+c)^5)" for i in range(34))), False
    if tensor_directory is None:
        return
    from symbolica.community.hepkit import Model, FeynmanDiagram
    from symbolica.community.tensor import TensorExpression

    model = Model.qcd()
    for name, filename in [("projected", "actual-four-loop-diagram.json"),
                           ("four-loop-projected", "four-loop-diagram.json")]:
        diagram = FeynmanDiagram.from_json(model, (tensor_directory / filename).read_text())
        numerator = model.expand_couplings(
            diagram.numerator_expression(in_lmb=True)
            * diagram.overall_factor_expression(evaluate=True)
        ).with_lorentz_dimension(S("D")).collect_factors()
        slots = numerator.structure.slots()
        projected = (TensorExpression.g(slots[0], slots[1])
                     * TensorExpression.g(slots[2], slots[3]) / 8 * numerator)
        yield name, projected, True
        yield name.replace("projected", "reduced"), projected.simplify_algebra(contract="dots"), True


def measure(function, count):
    samples = []
    for _ in range(count):
        started = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    return {"median_ms": statistics.median(samples), "min_ms": min(samples),
            "max_ms": max(samples), "samples_ms": samples}


def benchmark(expression, tensor, count):
    # Explicit25 keeps initial item requests identical across versions and
    # makes the projected-product omission available for popup measurements.
    make = (lambda: expression.paged(25)) if tensor else (lambda: expression.mathml_viewer(25))
    original_state = bytes(expression.__getstate__())

    def cold_page():
        viewer = make()
        try:
            return viewer.page()
        finally:
            viewer.close()

    def cold_static():
        viewer = make()
        try:
            return viewer._repr_html_()
        finally:
            viewer.close()

    cold_page()  # Load module-level lazy state before collecting samples.
    result = {"cold_page": measure(cold_page, count),
              "cold_static": measure(cold_static, count)}
    result["static_sha256"] = hashlib.sha256(cold_static().encode()).hexdigest()
    viewer = make()
    try:
        page = viewer.page()
        result["payload"] = {
            "html_bytes": len(page["html"].encode()),
            "serialized_bytes": len(json.dumps(page).encode()),
            "layouts": len(page.get("layouts", [])) + 1,
            "markers": len(page["markers"]),
            "sha256": hashlib.sha256(json.dumps(page, sort_keys=True).encode()).hexdigest(),
        }
        result["warm_page"] = measure(viewer.page, count * 10)
        result["warm_layout"] = measure(lambda: viewer._render_page(0), count * 10)
        if page["markers"]:
            path = page["markers"][0][0]
            request = dict(action="preview", value=path, layout=0)
            viewer.handle_message(request)
            result["warm_hover"] = measure(lambda: viewer.handle_message(request), count * 10)
            request = dict(request, popup=True)
            viewer.handle_message(request)
            result["warm_popup"] = measure(lambda: viewer.handle_message(request), count * 10)
        if page.get("next") is not None:
            def round_trip():
                viewer.handle_message(dict(action="next"))
                viewer.handle_message(dict(action="previous"))
            round_trip()
            result["warm_next_previous"] = measure(round_trip, count * 10)
    finally:
        viewer.close()

    # Measure validation's JSON allocation separately from normal timings.
    # This wrapper records only a fresh page request, not benchmark output.
    original_dumps = json.dumps
    calls = []
    def counted_dumps(*args, **kwargs):
        started = time.perf_counter_ns()
        value = original_dumps(*args, **kwargs)
        calls.append({"ms": (time.perf_counter_ns() - started) / 1_000_000,
                      "bytes": len(value.encode())})
        return value
    try:
        json.dumps = counted_dumps
        cold_page()
    finally:
        json.dumps = original_dumps
    result["validation_json"] = {"calls": len(calls),
                                 "total_ms": sum(item["ms"] for item in calls),
                                 "total_bytes": sum(item["bytes"] for item in calls)}
    assert bytes(expression.__getstate__()) == original_state, "Paging mutated the expression"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-fixtures", type=Path)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    report = {"python": sys.version, "platform": sys.platform,
              "distribution_version": importlib.metadata.version("symbolica"),
              "iterations": args.iterations, "cases": {}}
    for name, expression, tensor in fixtures(args.tensor_fixtures):
        report["cases"][name] = benchmark(expression, tensor, args.iterations)
        print(name, {key: round(value["median_ms"], 4)
                     for key, value in report["cases"][name].items()
                     if "median_ms" in value}, flush=True)
    report["anywidget_imported"] = "anywidget" in sys.modules
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
