"""Compare reviewed overloads in the shipped stub with generated binding metadata.

Usage: python tools/check_api_stubs.py /tmp/symbolica-stubs/symbolica/core.pyi
Runtime and static example tests separately check whether these calls are usable.
"""
import argparse
import ast
from pathlib import Path


TARGETS = {
    (None, "S"), ("Expression", "symbol"), ("Expression", "to_polynomial"),
    ("Series", "get_coefficient"), ("Evaluator", "load"),
    ("Evaluator", "evaluate_complex_with_prec"),
}


def signatures(path):
    tree = ast.parse(path.read_text())
    found = {}
    scopes = [(None, tree.body)] + [
        (node.name, node.body) for node in tree.body if isinstance(node, ast.ClassDef)
    ]
    for scope, nodes in scopes:
        for node in nodes:
            key = (scope, getattr(node, "name", None))
            if key not in TARGETS or not isinstance(node, ast.FunctionDef):
                continue
            args = node.args
            positional = args.posonlyargs + args.args
            required_count = len(positional) - len(args.defaults)
            parameters = []
            for i, arg in enumerate(positional):
                if scope is not None and i == 0:
                    continue  # self/cls/_cls are implementation spelling choices.
                kind = "positional" if i < len(args.posonlyargs) else "positional_or_keyword"
                parameters.append((arg.arg, kind, i < required_count))
            if args.vararg:
                parameters.append((args.vararg.arg, "varargs", False))
            parameters.extend(
                (arg.arg, "keyword", default is None)
                for arg, default in zip(args.kwonlyargs, args.kw_defaults)
            )
            if args.kwarg:
                parameters.append((args.kwarg.arg, "kwargs", False))
            # Keyword-only argument order has no effect on valid calls.
            signature = (
                tuple(p for p in parameters if p[1] != "keyword"),
                tuple(sorted(p for p in parameters if p[1] == "keyword")),
                ast.unparse(node.returns).replace("decimal.", "").replace("typing.", "").replace("builtins.", ""),
            )
            found.setdefault(key, set()).add(signature)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", type=Path)
    parser.add_argument("--shipped", type=Path, default=Path(__file__).resolve().parents[1] / "symbolica.pyi")
    args = parser.parse_args()
    actual, generated = signatures(args.shipped), signatures(args.generated)
    failures = []
    for target in sorted(TARGETS, key=str):
        if target not in actual or target not in generated or actual[target] != generated[target]:
            failures.append(f"{target}:\n  shipped: {actual.get(target)}\n  generated: {generated.get(target)}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"{len(TARGETS)} reviewed API signatures agree with generated metadata")


if __name__ == "__main__":
    main()
