#!/usr/bin/env python3
"""Select slow captured operations and deduplicate their compressed operands.

Usage: python3 benches/extract_slow_fac.py OUTPUT line2=/tmp/capture-line2 ...
Then run slow_fac_operations with SYMBOLICA_SLOW_FAC_DESCRIBE=1 to fill in
the last column (expected result terms, or `none` for failed divisibility).
"""

import argparse
import hashlib
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("captures", nargs="+", help="source_label=capture_directory")
    parser.add_argument("--per-operation", type=int, default=3)
    args = parser.parse_args()
    if args.per_operation < 1:
        parser.error("--per-operation must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = ["# name\toperation\tnvars\tleft\tright\tleft_terms\tright_terms\tresult_terms"]
    provenance = ["# case\tcapture\tleft_sha256\tright_sha256"]
    for source in args.captures:
        label, directory = source.split("=", 1)
        grouped = {}
        for metadata in sorted(Path(directory).glob("*.meta")):
            info = dict(line.split("=", 1) for line in metadata.read_text().splitlines())
            grouped.setdefault(info["operation"], []).append((metadata, info))
        if not grouped:
            parser.error(f"no captures in {directory}")
        for operation, candidates in sorted(grouped.items()):
            candidates.sort(key=lambda pair: float(pair[1]["seconds"]), reverse=True)
            for metadata, info in candidates[:args.per_operation]:
                name = f"{label}_{metadata.stem.replace('-', '_')}"
                files, hashes = [], []
                for side in ("left", "right"):
                    operand = metadata.with_suffix(f".{side}.txt.br")
                    digest = hashlib.sha256(operand.read_bytes()).hexdigest()
                    filename = f"{digest}.txt.br"
                    if not (args.output / filename).exists():
                        shutil.copyfile(operand, args.output / filename)
                    files.append(filename)
                    hashes.append(digest)
                rows.append("\t".join([name, operation, info["variables"], *files,
                                       info["left_terms"], info["right_terms"], "?"]))
                provenance.append("\t".join([name, f"{label}/{metadata.stem}", *hashes]))
    (args.output / "cases.tsv").write_text("\n".join(rows) + "\n")
    (args.output / "provenance.tsv").write_text("\n".join(provenance) + "\n")


if __name__ == "__main__":
    main()
