"""Minimal stand-in for a pure-Python library with a signed Symbolica OEM allowance."""

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Event

from symbolica import E, oem_scope

OEM_TOKEN = (
    "eyJ2ZXJzaW9uIjoxLCJwYWNrYWdlIjoicHlzZWNkZWMiLCJtYXhfcHJvY2Vzc2VzIjo0fQo."
    "BlpNuJqgYtiWDgokAZO2Q8udQUB9WTIJn6EueXmm35G85NSMCngcpMGkxdzFNRXGVP3m6nR-KyhOK64Y7sZHAw"
)
__all__ = [
    "library_expression",
    "run_threads",
    "use_plain_callback",
    "use_symbolica_callback",
]


def library_expression() -> str:
    """Run and format a Symbolica operation entirely inside this module."""
    with oem_scope(OEM_TOKEN, 0):
        return str(E("library_value + 1"))


def use_plain_callback(callback: Callable[[], int]) -> tuple[int, str]:
    """Get plain user input, then resume skipped Symbolica work in this module."""
    with oem_scope(OEM_TOKEN, 0):
        value = callback()
        return value, str(E("library_after_callback + 1"))


def use_symbolica_callback(callback: Callable[[], object]) -> object:
    """Invoke user code within this library operation's OEM allowance."""
    with oem_scope(OEM_TOKEN, 0):
        return callback()


def run_threads(count: int, callback: Callable[[], None] | None = None) -> list[str]:
    """Make `count` Python threads enter Symbolica concurrently under one OEM scope."""
    all_workers_registered = Barrier(count + 1)
    release_workers = Event()

    def work(index: int) -> str:
        result = str(E(f"thread_{index} + 1"))
        all_workers_registered.wait()
        release_workers.wait()
        return result

    with oem_scope(OEM_TOKEN, count):
        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = [executor.submit(work, index) for index in range(count)]
            all_workers_registered.wait()
            try:
                if callback is not None:
                    callback()
            finally:
                release_workers.set()
            return [future.result() for future in futures]


def hold_oem_process(ready_path: str) -> None:
    """Hold one token process slot until stdin closes."""
    with oem_scope(OEM_TOKEN, 0):
        E("oem_process_holder")
        Path(ready_path).write_text("ready", encoding="utf-8")
        input()
