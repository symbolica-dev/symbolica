#!/usr/bin/env python3
"""Extract a deterministic coefficient from a pinned public FIRE7 IBP table."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import argparse

parser = argparse.ArgumentParser(description=__doc__)
selection = parser.add_mutually_exclusive_group()
selection.add_argument("--suite", action="store_true", help="also export 32 distinct coefficients spanning the table")
selection.add_argument("--all", dest="all_coefficients", action="store_true", help="export every distinct coefficient string into the all subdirectory")
args = parser.parse_args()

root = Path(__file__).resolve().parents[2]
fire = root / "target/reconstruction-external/fire"
revision = "d132e5365dd2a13db9cd9dbaf5c200b53d489cfd"
assert subprocess.check_output(["git", "-C", str(fire), "rev-parse", "HEAD"], text=True).strip() == revision
source = fire / "FIRE7/examples/nb0/intsde-nb0.tables"
raw = source.read_bytes()
digest = hashlib.sha256(raw).hexdigest()
assert digest == "cebddb9e7b6695a3dccb0e91ebe6e821f5093b060107a957a70a60cb6090ea42"
# This table contains lists, integer identifiers and quoted polynomial strings.
# Polynomial strings contain no braces, so converting the list delimiters is exact.
assert not re.search(r'"[^"\n]*[{}][^"\n]*"', raw.decode())
table = json.loads(raw.decode().replace("{", "[").replace("}", "]"))
coefficients = [(row, master, expression) for row, terms in table[0] for master, expression in terms]
row, master, expression = max(coefficients, key=lambda c: len(c[2]))
assert set(re.findall(r"[A-Za-z]+", expression)) == {"u", "v", "w", "d"}
out = root / "target/reconstruction-external/fire-table-inputs"
out.mkdir(exist_ok=True)
data = (expression + "\n").encode()
(out / "fire7_nb0_largest").write_bytes(data)
manifest = dict(revision=revision, source=str(source.relative_to(fire)), source_sha256=digest,
                selection="longest coefficient string, first in table order on ties",
                coefficient_count=len(coefficients), row=str(row), master=str(master),
                variables=["u", "v", "w", "d"], expression_bytes=len(expression),
                output_sha256=hashlib.sha256(data).hexdigest())
(out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(json.dumps(manifest, indent=2))

if args.suite or args.all_coefficients:
    # Deduplicate exact source strings, preserving the first table occurrence.
    unique = {}
    for item in coefficients:
        unique.setdefault(item[2], item)
    ranked = sorted(unique.values(), key=lambda item: -len(item[2]))
    assert len(ranked) >= 32
    # Largest 16, then 16 evenly spaced ranks in the remaining size range.
    ranks = list(range(16)) + [16 + i * (len(ranked) - 17) // 15 for i in range(16)]
    rule = "first 16 descending string-length ranks, then 16 evenly spaced remaining ranks; first table occurrence breaks ties"
    suite_out = out
    if args.all_coefficients:
        ranks = range(len(ranked))
        rule = "all distinct coefficient strings, descending string-length ranks; first table occurrence breaks ties"
        suite_out = out / "all"
        suite_out.mkdir(exist_ok=True)
    entries = []
    for rank in ranks:
        row, master, expression = ranked[rank]
        assert set(re.findall(r"[A-Za-z]+", expression)) <= {"u", "v", "w", "d"}
        case = f"fire7_nb0_rank{rank + 1:04d}"
        data = (expression + "\n").encode()
        (suite_out / case).write_bytes(data)
        (suite_out / f"{case}.variables").write_text("u v w d\n")
        entries.append(dict(case=case, rank=rank + 1, row=str(row), master=str(master),
                            expression_bytes=len(expression), output_sha256=hashlib.sha256(data).hexdigest()))
    suite = dict(revision=revision, source=manifest["source"], source_sha256=digest,
                 coefficient_count=len(coefficients), distinct_source_strings=len(ranked),
                 selection=rule,
                 variables=["u", "v", "w", "d"], entries=entries)
    (suite_out / "suite-manifest.json").write_text(json.dumps(suite, indent=2) + "\n")
    (suite_out / "suite-cases.txt").write_text("\n".join(entry["case"] for entry in entries) + "\n")
    print(f"Exported {len(entries)} suite cases from {len(ranked)} distinct source strings")
