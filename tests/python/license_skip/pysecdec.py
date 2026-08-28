"""Minimal stand-in for a Python library whose own Symbolica calls skip licensing."""

SKIP_LICENSE = True

from collections.abc import Callable

from symbolica import E

__all__ = ["library_expression", "use_plain_callback", "use_symbolica_callback"]


def library_expression() -> str:
    """Run and format a Symbolica operation entirely inside this module."""
    return str(E("library_value + 1"))


def use_plain_callback(callback: Callable[[], int]) -> tuple[int, str]:
    """Get plain user input, then resume skipped Symbolica work in this module."""
    value = callback()
    return value, str(E("library_after_callback + 1"))


def use_symbolica_callback(callback: Callable[[], object]) -> object:
    """Invoke user code without extending this module's license opt-out into it."""
    return callback()
