"""Minimal stand-in for a pure-Python library with a signed Symbolica OEM allowance."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

from symbolica import E, oem_scope

OEM_TOKEN = (
    "eyJ2ZXJzaW9uIjoxLCJwYWNrYWdlIjoicHlzZWNkZWMiLCJtYXhfcHJvY2Vzc2VzIjo0LCJtYXhfdGhyZWFkc19wZXJfcHJvY2VzcyI6OH0K."
    "ce4TV4ICSXSNBESisJSbAheXXL5T5Fts6FPdMlSlHVCA0bJFIlLRj3s2iyobmIZk7yOR0Ix_vjVnufkULUjwCw"
)

__all__ = [
    "library_expression",
    "run_threads",
    "use_plain_callback",
    "use_symbolica_callback",
]


def library_expression() -> str:
    """Run and format a Symbolica operation entirely inside this module."""
    with oem_scope(OEM_TOKEN):
        return str(E("library_value + 1"))


def use_plain_callback(callback: Callable[[], int]) -> tuple[int, str]:
    """Get plain user input, then resume skipped Symbolica work in this module."""
    with oem_scope(OEM_TOKEN):
        value = callback()
        return value, str(E("library_after_callback + 1"))


def use_symbolica_callback(callback: Callable[[], object]) -> object:
    """Invoke user code within this library operation's OEM allowance."""
    with oem_scope(OEM_TOKEN):
        return callback()


def run_threads(count: int) -> list[str]:
    """Make `count` Python threads enter Symbolica concurrently under one OEM scope."""
    barrier = Barrier(count)

    def work(index: int) -> str:
        barrier.wait()
        return str(E(f"thread_{index} + 1"))

    with oem_scope(OEM_TOKEN):
        with ThreadPoolExecutor(max_workers=count) as executor:
            return list(executor.map(work, range(count)))


def hold_oem_process(ready_path: str) -> None:
    """Hold one token process slot until stdin closes."""
    with oem_scope(OEM_TOKEN):
        E("oem_process_holder")
        Path(ready_path).write_text("ready", encoding="utf-8")
        input()
