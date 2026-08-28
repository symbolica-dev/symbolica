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
        print(pysecdec.use_symbolica_callback(symbolica_callback), flush=True)
    elif scenario == "threads-at-limit":
        print(len(pysecdec.run_threads(8)), flush=True)
    elif scenario == "threads-over-limit":
        pysecdec.run_threads(9)
    elif scenario == "copied-token":
        try:
            from symbolica import oem_scope

            with oem_scope(pysecdec.OEM_TOKEN, 0):
                E("copied_token")
        except PermissionError as error:
            print(error, flush=True)
        else:
            raise AssertionError("user.py unexpectedly activated the pysecdec OEM token")
    elif scenario == "library-then-user":
        pysecdec.library_expression()
        E("direct_user_after_library")
    elif scenario == "direct-user":
        E("direct_user_value + 1")
    else:
        raise ValueError(f"unknown scenario: {scenario}")


if __name__ == "__main__":
    main(sys.argv[1])
