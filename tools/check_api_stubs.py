"""Compare reviewed overloads in the shipped stub with generated binding metadata.

Usage: python tools/check_api_stubs.py /tmp/symbolica-stubs/symbolica/core.pyi
Runtime and static example tests separately check whether these calls are usable.
"""
import argparse
import ast
import re
from pathlib import Path


TARGETS = {
    ("Expression", "solve"), ("Solution", "as_dict"), ("SolutionSet", "dimension"),
    (None, "S"), ("Expression", "symbol"), ("Expression", "to_polynomial"),
    ("Series", "get_coefficient"), ("Evaluator", "load"),
    ("Evaluator", "evaluate_complex_with_prec"), ("Evaluator", "evaluate_with_prec"),
    ("Expression", "nsolve"), ("Expression", "nsolve_system"),
    ("Float", "__new__"), ("Float", "from_ratio"), ("Float", "with_precision"),
    ("Float", "to_decimal"), ("Float", "as_integer_ratio"),
    ("ComplexFloat", "__new__"), ("ComplexFloat", "with_precision"),
    ("ComplexFloat", "to_decimal_tuple"), ("ComplexFloat", "as_tuple"),
    *((name, "load") for name in (
        "CompiledRealEvaluator", "CompiledComplexEvaluator",
        "CompiledSimdRealEvaluator", "CompiledSimdComplexEvaluator",
        "CompiledCudaRealEvaluator", "CompiledCudaComplexEvaluator",
    )),
}


# Cover every newly exposed operation and constant, including argument defaults.
FLOAT_METHODS = {
    "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh", "tan",
    "pi", "e", "euler", "euler_gamma", "phi", "i", "new_zero", "new_one",
    "conj", "conjugate", "neg", "zero", "one", "nan", "inv", "norm",
    "is_zero", "is_one", "is_fully_zero", "fixed_precision", "get_precision", "get_epsilon",
    "from_usize", "from_i64", "from_rational", "from_ratio", "sample_unit", "set_from",
    "pow", "powf", "atan2", "mul_add",
}
TARGETS.update((cls, method) for cls in ("Float", "ComplexFloat") for method in FLOAT_METHODS)
TARGETS.update(("Float", method) for method in ("to_f64", "round_to_nearest_integer", "to_usize_clamped"))


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
                if scope is not None and i == 0 and arg.arg in {"self", "cls", "_cls"}:
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
            return_type = ast.unparse(node.returns).replace("decimal.", "").replace("typing.", "").replace("builtins.", "")
            return_type = re.sub(r"^Optional\[(.*)\]$", r"\1 | None", return_type)
            signature = (
                tuple(p for p in parameters if p[1] != "keyword"),
                tuple(sorted(p for p in parameters if p[1] == "keyword")),
                return_type,
            )
            found.setdefault(key, set()).add(signature)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated", type=Path)
    parser.add_argument("--shipped", type=Path, default=Path(__file__).resolve().parents[1] / "symbolica.pyi")
    args = parser.parse_args()
    actual, generated = signatures(args.shipped), signatures(args.generated)
    # Numerica registers its classes in the consuming project's default module.
    package_stub = args.generated.with_name("__init__.pyi")
    if package_stub != args.generated and package_stub.exists():
        generated.update(signatures(package_stub))
    failures = []
    # Decimal is an explicit interoperability format, never an implicit result.
    for node in ast.walk(ast.parse(args.shipped.read_text())):
        if (isinstance(node, ast.FunctionDef) and node.returns
                and "Decimal" in ast.unparse(node.returns)
                and node.name not in {"to_decimal", "to_decimal_tuple"}):
            failures.append(f"{node.name} still returns Decimal implicitly")
    for target in sorted(TARGETS, key=str):
        if target not in actual or target not in generated or actual[target] != generated[target]:
            failures.append(f"{target}:\n  shipped: {actual.get(target)}\n  generated: {generated.get(target)}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"{len(TARGETS)} reviewed API signatures agree with generated metadata")


if __name__ == "__main__":
    main()
