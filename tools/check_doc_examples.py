"""Run Markdown/Quarto Python blocks in separate temporary directories.

Use the Python interpreter containing the wheel under review. Each block must
be self-contained. Historical examples should not be passed to this checker.
"""
import argparse
from pathlib import Path
import re
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--allow-unavailable-integration", action="store_true")
    args = parser.parse_args()
    passed = skipped = failed = 0
    for path in args.paths:
        source = path.read_text()
        for match in re.finditer(r"^```(?:python|\{python\})\n(.*?)^```", source, re.M | re.S):
            line = source[:match.start()].count("\n") + 1
            label = f"{path}:{line}"
            with tempfile.TemporaryDirectory(prefix="symbolica-example-") as directory:
                try:
                    result = subprocess.run(
                        [sys.executable, "-c", match[1]], cwd=directory,
                        capture_output=True, text=True, timeout=args.timeout,
                    )
                except subprocess.TimeoutExpired:
                    print(f"FAIL {label}: exceeded {args.timeout}s")
                    failed += 1
                    continue
            if result.returncode == 0:
                passed += 1
            elif args.allow_unavailable_integration and (
                result.stderr.rstrip().endswith(
                    "NotImplementedError: No symbolic integration backend is linked into this Symbolica build"
                )
            ):
                print(f"SKIP {label}: integration backend unavailable")
                skipped += 1
            else:
                print(f"FAIL {label}:\n{result.stderr[-4000:]}")
                failed += 1
    print(f"{passed} passed, {skipped} skipped, {failed} failed")
    if passed + skipped + failed == 0:
        raise SystemExit("No Python examples found")
    raise SystemExit(bool(failed))


if __name__ == "__main__":
    main()
