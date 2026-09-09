#!/usr/bin/env python3
"""Invertibly mix d and u in the four largest pinned nb0 coefficients."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
subprocess.run([sys.executable, str(root / "benches/external/extract_fire_tables.py"), "--all"], check=True)
source = root / "target/reconstruction-external/fire-table-inputs/all"
manifest = json.loads((source / "suite-manifest.json").read_text())
out = root / "target/reconstruction-external/fire-table-inputs/sheared"
out.mkdir(parents=True, exist_ok=True)
entries = []
for entry in manifest["entries"][:4]:
    data = (source / entry["case"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["output_sha256"]
    expression, replacements = re.subn(r"\bd\b", "(d+u)", data.decode())
    assert replacements > 0
    case = entry["case"] + "_shear_d"
    (out / case).write_text(expression)
    (out / (case + ".variables")).write_text("u v w d\n")
    entries.append({"case": case, "original": entry, "replacements": replacements,
                    "output_sha256": hashlib.sha256(expression.encode()).hexdigest()})
manifest = {k: v for k, v in manifest.items() if k != "entries"}
manifest.update(selection="first four descending source-length ranks from the full nb0 table",
                substitution={"d": "d+u"}, inverse_substitution={"d": "d-u"}, entries=entries)
(out / "suite-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
(out / "suite-cases.txt").write_text("\n".join(e["case"] for e in entries) + "\n")
print(f"Exported {len(entries)} invertibly transformed nb0 coefficients to {out}")
