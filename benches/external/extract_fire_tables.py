#!/usr/bin/env python3
"""Extract a deterministic coefficient from a pinned public FIRE7 IBP table."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

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
