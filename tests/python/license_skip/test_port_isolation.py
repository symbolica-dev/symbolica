"""End-to-end test for file-local license skipping across Python callback boundaries."""

import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


FIXTURE_DIR = Path(__file__).resolve().parent
PORT_COLLISION_MESSAGE = "Cannot start new unlicensed Symbolica instance"


def find_available_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class LicenseSkipPortIsolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        ready_path = Path(cls._temporary_directory.name) / "ready"

        cls.environment = os.environ.copy()
        cls.environment.pop("SYMBOLICA_LICENSE", None)
        cls.environment["SYMBOLICA_HIDE_BANNER"] = "1"
        cls.environment["SYMBOLICA_PORT"] = str(find_available_port())

        cls.holder = subprocess.Popen(
            [sys.executable, str(FIXTURE_DIR / "hold_license_port.py"), str(ready_path)],
            cwd=FIXTURE_DIR,
            env=cls.environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        deadline = time.monotonic() + 10
        while not ready_path.exists():
            if cls.holder.poll() is not None:
                output = cls.holder.stdout.read() if cls.holder.stdout else ""
                raise RuntimeError(f"port holder exited early ({cls.holder.returncode}):\n{output}")
            if time.monotonic() >= deadline:
                cls.holder.terminate()
                raise TimeoutError("timed out waiting for the Symbolica port holder")
            time.sleep(0.02)

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.holder.stdin:
            cls.holder.stdin.close()
        try:
            cls.holder.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.holder.terminate()
            cls.holder.wait(timeout=5)
        cls._temporary_directory.cleanup()

    def run_user_scenario(self, scenario: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(FIXTURE_DIR / "user.py"), scenario],
            cwd=FIXTURE_DIR,
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )

    def assert_succeeds(self, scenario: str) -> str:
        result = self.run_user_scenario(scenario)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result.stdout

    def assert_port_collision(self, scenario: str) -> None:
        result = self.run_user_scenario(scenario)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(PORT_COLLISION_MESSAGE, result.stdout)

    def test_symbolica_inside_library_does_not_collide(self) -> None:
        self.assertIn("library_value", self.assert_succeeds("library"))

    def test_plain_callback_returns_to_skipped_library_scope(self) -> None:
        output = self.assert_succeeds("plain-callback")
        self.assertIn("42", output)
        self.assertIn("library_after_callback", output)

    def test_symbolica_inside_user_callback_collides(self) -> None:
        self.assert_port_collision("symbolica-callback")

    def test_symbolica_directly_inside_user_file_collides(self) -> None:
        self.assert_port_collision("direct-user")


if __name__ == "__main__":
    unittest.main()
