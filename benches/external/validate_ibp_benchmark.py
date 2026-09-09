#!/usr/bin/env python3
"""Check prepared IBP expressions against their traces at eight exact Q points.

These independent point checks validate input provenance probabilistically;
the benchmark drivers separately check reconstruction identities exactly.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import re
import random
import subprocess


def parse(expression):
    # Iterative shunting-yard parsing also handles polynomials too large for
    # Python's AST construction recursion limit. Never execute input as code.
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+|[()+*/^\-]", expression)
    assert "".join(tokens) == re.sub(r"\s+", "", expression)
    output, operators = [], []
    precedence = {"+": 1, "-": 1, "*": 2, "/": 2, "u+": 3, "u-": 3, "^": 4}
    operand = True
    for token in tokens:
        if token == "(":
            operators.append(token)
            operand = True
        elif token == ")":
            while operators[-1] != "(":
                output.append(operators.pop())
            operators.pop()
            operand = False
        elif token in precedence:
            if operand and token in ("+", "-"):
                operators.append("u" + token)
                continue
            assert not operand
            while operators and operators[-1] != "(" and (
                precedence[operators[-1]] > precedence[token] or
                precedence[operators[-1]] == precedence[token] and token != "^"
            ):
                output.append(operators.pop())
            operators.append(token)
            operand = True
        else:
            assert operand
            output.append(int(token) if token.isdigit() else token)
            operand = False
    assert not operand and "(" not in operators
    return output + operators[::-1]


def evaluate(tokens, values):
    stack = []
    for token in tokens:
        if type(token) is int:
            stack.append(Fraction(token))
        elif token in values:
            stack.append(Fraction(values[token]))
        elif token in ("u+", "u-"):
            stack[-1] *= -1 if token == "u-" else 1
        else:
            b, a = stack.pop(), stack.pop()
            if token == "+":
                stack.append(a + b)
            elif token == "-":
                stack.append(a - b)
            elif token == "*":
                stack.append(a * b)
            elif token == "/":
                stack.append(a / b)
            elif token == "^":
                assert b.denominator == 1
                stack.append(a ** b.numerator)
            else:
                raise ValueError(token)
    assert len(stack) == 1
    return stack[0]


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("family")
args = parser.parse_args()
external = Path(__file__).resolve().parents[2] / "target/reconstruction-external"
work = external / "ibp-work" / args.family
inputs = external / "ibp-inputs" / args.family
manifest = json.loads((inputs / "manifest.json").read_text())
trees = {}
for entry in manifest["entries"]:
    if "case" not in entry:
        continue
    data = (inputs / entry["case"]).read_bytes()
    assert hashlib.sha256(data).hexdigest() == entry["expression_sha256"]
    trees[entry["case"]] = parse(data.decode())
loader = ([os.environ["EXTERNAL_LOADER"], "--library-path", os.environ["EXTERNAL_LIBRARY_PATH"]]
          if "EXTERNAL_LOADER" in os.environ else [])
checks = []
seed = 0x5241545241434552
rng = random.Random(seed)
for index in range(8):
    values = {v: rng.randrange(11, 2000) for v in manifest["variables"]}
    command = loader + [str(external / "ratracer-tool")]
    for variable, value in values.items():
        command += ["set", variable, str(value)]
    command += ["load-trace", "selected.trace", "finalize", "evaluate"]
    result = subprocess.run(command, cwd=work, capture_output=True, text=True, timeout=120)
    (work / "preparation-logs" / f"validate-{index}.log").write_text(result.stdout + result.stderr)
    result.check_returncode()
    actual = dict(re.findall(r"(ibp_\w+)\s*=\s*([^;]+);", result.stdout))
    assert set(actual) == set(trees), result.stdout
    expected = {case: evaluate(tree, values) for case, tree in trees.items()}
    assert all(Fraction(actual[case].strip()) == value for case, value in expected.items())
    checks.append(dict(point=values, values={case: str(value) for case, value in expected.items()}))
    print(f"{args.family}: exact trace check {index + 1}/8 passed", flush=True)
report = dict(method="eight exact rational point checks against Ratracer; not a symbolic identity certificate",
              seed=seed, trace_sha256=hashlib.sha256((work / "selected.trace").read_bytes()).hexdigest(), checks=checks)
(inputs / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
manifest["validation"] = report["method"] + "; all passed (validation.json)"
(inputs / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
