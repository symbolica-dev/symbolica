"""Scenario driver representing user code that has not opted out of license checks."""

import sys

import pysecdec
from symbolica import E


def plain_callback() -> int:
    return 42


def symbolica_callback() -> object:
    # The active Python globals at the license check belong to this file, not pysecdec.py.
    return E("user_callback_value + 1")


def main(scenario: str) -> None:
    if scenario == "library":
        print(pysecdec.library_expression(), flush=True)
    elif scenario == "plain-callback":
        print(pysecdec.use_plain_callback(plain_callback), flush=True)
    elif scenario == "symbolica-callback":
        pysecdec.use_symbolica_callback(symbolica_callback)
    elif scenario == "direct-user":
        E("direct_user_value + 1")
    else:
        raise ValueError(f"unknown scenario: {scenario}")


if __name__ == "__main__":
    main(sys.argv[1])
