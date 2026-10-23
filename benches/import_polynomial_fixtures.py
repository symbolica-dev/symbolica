#!/usr/bin/env python3
"""Import pinned polynomial benchmark inputs; requires git and the brotli CLI.

Usage: python benches/import_polynomial_fixtures.py /path/to/source [--brotli brotli]
The checkout must contain SOURCE_COMMIT. No timings are imported as expectations.
"""
import argparse
import csv
import hashlib
import json
import lzma
from pathlib import Path
import subprocess
import sys

SOURCE_COMMIT = "07806e55609d999f4103a33a0ee2d2aa1ba2ae57"
SOURCE = "benchmark/FLINT_vs_symbolica/"

# Published segmentation operand, reconstructed from its compact parent. The
# hash below is checked against the upstream manifest, not a newly chosen input.
SEGMENTATION_ID = "g33S2-res-02-41f64e7684-segenc"
SEGMENTATION_WEIGHTS = [0, 217, 0, 0, 1]
SEGMENTATION_BITS = 652
FIXTURE_FORMAT = "polynomial-bench-op 1"


def fixture_operand(raw):
    header, newline, body = raw.partition(b"\n")
    assert newline and header.endswith(b"-bench-op 1")
    return FIXTURE_FORMAT.encode() + b"\n" + body


def segmented_operand(raw):
    sys.set_int_max_str_digits(0)
    lines = iter(raw.decode().splitlines())
    header = next(lines)
    assert header.endswith("-bench-op 1")
    assert next(lines) == "operation resultant"
    assert len(next(lines).split()) == 6
    variable = int(next(lines).split()[1])
    assert next(lines) == "polynomials 2"
    output = [header, "operation resultant", "variables X",
              "variable 0", "polynomials 2"]
    for _ in range(2):
        coefficients = {}
        for _ in range(int(next(lines).split()[1])):
            coefficient, *exponents = map(int, next(lines).split())
            shift = sum(e * w for e, w in zip(exponents, SEGMENTATION_WEIGHTS)) * SEGMENTATION_BITS
            degree = exponents[variable]
            coefficients[degree] = coefficients.get(degree, 0) + (coefficient << shift)
        terms = [f"{c} {e}" for e, c in sorted(coefficients.items()) if c]
        output.extend([f"terms {len(terms)}", *terms])
    return ("\n".join(output) + "\n").encode()


def operand_metadata(op):
    """Keep source identity and the data needed to reproduce derived operands."""
    if op is None:
        return None
    keys = ("id", "operation", "sha256", "bytes", "committed", "parent", "derivative_variable")
    return {key: op[key] for key in keys if key in op}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    parser.add_argument("--brotli", default="brotli")
    args = parser.parse_args()

    def read(name):
        return subprocess.check_output(
            ["git", "-C", str(args.checkout), "show", f"{SOURCE_COMMIT}:{SOURCE}{name}"]
        )

    manifest = json.loads(read("operands/manifest.json"))
    factors = [o for o in manifest["ops"] if o["operation"] == "factor"]
    gcds = {o["parent"]: o for o in manifest["ops"] if o.get("derived") == "gcd-squarefree"}
    dest = Path(__file__).parent / "fixtures" / "polynomial"
    dest.mkdir(parents=True, exist_ok=True)
    fixtures = {}

    def store_operand(op, raw):
        normalized = fixture_operand(raw)
        (dest / f"{op['id']}.op.br").write_bytes(
            subprocess.check_output([args.brotli, "-q", "9", "-c"], input=normalized))
        fixtures[op["id"]] = {
            "sha256": hashlib.sha256(normalized).hexdigest(), "bytes": len(normalized),
        }

    imported, missing, cases, gcd_cases = [], [], [], []
    for op in factors:
        if not op.get("committed"):
            missing.append(op["id"])
            continue
        raw = lzma.decompress(read(f"operands/sources/{op['id']}.op.xz"))
        assert hashlib.sha256(raw).hexdigest() == op["sha256"], op["id"]
        store_operand(op, raw)
        names = raw.decode().splitlines()[2].split()[1:]
        gcd = gcds.get(op["id"])
        variable = names.index(gcd["derivative_variable"]) if gcd else None
        if gcd:
            # Reproduce the published two-polynomial operand byte-for-byte,
            # although the Rust benchmark constructs this derivative at setup.
            lines = raw.decode().splitlines()
            derivative = []
            for line in lines[6:]:
                coefficient, *exponents = map(int, line.split())
                degree = exponents[variable]
                if degree:
                    exponents[variable] -= 1
                    derivative.append(" ".join(map(str, [coefficient * degree, *exponents])))
            lines[1] = "operation gcd"
            lines[3] = f"variable {variable}"
            lines[4] = "polynomials 2"
            derived = ("\n".join(lines + [f"terms {len(derivative)}", *derivative]) + "\n").encode()
            assert hashlib.sha256(derived).hexdigest() == gcd["sha256"], gcd["id"]
        case = 'Case { id: "%s", terms: %d, gcd_variable: %s }' % (
            op["id"], op["sizes"]["terms"][0], f"Some({variable})" if gcd else "None")
        cases.append(case)
        if gcd:
            gcd_cases.append(case)
        imported.append({"factor": operand_metadata(op), "gcd": operand_metadata(gcd)})
    segmented = next(o for o in manifest["ops"] if o["id"] == SEGMENTATION_ID)
    parent = next(o for o in manifest["ops"] if o["id"] == segmented["parent"])
    raw = lzma.decompress(read(f"operands/{parent['committed']}"))
    assert hashlib.sha256(raw).hexdigest() == parent["sha256"]
    assert hashlib.sha256(segmented_operand(raw)).hexdigest() == segmented["sha256"]
    store_operand(parent, raw)
    slow_elimination, elimination_cases = [], []
    ops = {o["id"]: o for o in manifest["ops"]}
    for row in csv.DictReader(read("results/per_op.csv").decode().splitlines()):
        ratio = row["ratio_symbolica_production_over_flint"]
        if row["kind"] not in ("resultant", "discriminant") or not ratio or float(ratio) <= 1:
            continue
        op = ops[row["id"]]
        raw = lzma.decompress(read(f"operands/{op['committed']}"))
        assert hashlib.sha256(raw).hexdigest() == op["sha256"], op["id"]
        store_operand(op, raw)
        discriminant = str(op["operation"] == "discriminant").lower()
        elimination_cases.append(f'EliminationCase {{ id: "{op["id"]}", discriminant: {discriminant} }}')
        slow_elimination.append(operand_metadata(op))
    (dest / "cases.rs").write_text(
        "// Generated by benches/import_polynomial_fixtures.py; do not edit.\n"
        + "pub const FACTOR_CASES: &[Case] = &[\n    " + ",\n    ".join(cases) + ",\n];\n"
        + "pub const GCD_CASES: &[Case] = &[\n    " + ",\n    ".join(gcd_cases) + ",\n];\n"
        + f'pub const RESULTANT_CASES: &[SegmentedCase] = &[SegmentedCase {{ id: "{segmented["id"]}", '
        + f'parent: "{parent["id"]}", weights: &{SEGMENTATION_WEIGHTS}, digit_bits: {SEGMENTATION_BITS} }}];\n'
        + "pub const ELIMINATION_CASES: &[EliminationCase] = &[\n    " + ",\n    ".join(elimination_cases) + ",\n];\n")
    (dest / "manifest.json").write_text(json.dumps({
        "commit": SOURCE_COMMIT, "source": SOURCE,
        "operands": imported, "omitted_upstream": missing,
        "segmented_resultants": [{"operand": operand_metadata(segmented),
                                  "parent": operand_metadata(parent),
                                  "weights": SEGMENTATION_WEIGHTS, "digit_bits": SEGMENTATION_BITS}],
        "slow_elimination": slow_elimination,
        "format": FIXTURE_FORMAT,
        "fixtures": dict(sorted(fixtures.items())),
    }, indent=2) + "\n")
    print(f"Imported {len(cases)} factorization, {len(gcd_cases)} derived GCD, "
          f"{len(elimination_cases)} slow elimination, and 1 encoded resultant cases; "
          f"{len(missing)} source operands unavailable upstream.")


if __name__ == "__main__":
    main()
