"""End-to-end test for signed OEM scopes, lockfiles, callbacks, and concurrency limits."""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path


FIXTURE_DIR = Path(__file__).resolve().parent
UNLICENSED_COLLISION_MESSAGE = "Cannot start new unlicensed Symbolica instance"
OEM_CONCURRENCY_MESSAGE = "OEM concurrency exceeds the library's declared allowance"


class OemScopeConcurrencyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._temporary_directory = tempfile.TemporaryDirectory()
        ready_path = Path(cls._temporary_directory.name) / "ready"

        cls.environment = os.environ.copy()
        cls.environment.pop("SYMBOLICA_LICENSE", None)
        cls.environment["SYMBOLICA_HIDE_BANNER"] = "1"
        cls.environment["SYMBOLICA_LOCK_DIR"] = cls._temporary_directory.name

        cls.holder = subprocess.Popen(
            [sys.executable, str(FIXTURE_DIR / "hold_unlicensed_process.py"), str(ready_path)],
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
                raise RuntimeError(
                    f"unlicensed holder exited early ({cls.holder.returncode}):\n{output}"
                )
            if time.monotonic() >= deadline:
                cls.holder.terminate()
                raise TimeoutError("timed out waiting for the unlicensed Symbolica holder")
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
        if cls.holder.stdout:
            cls.holder.stdout.close()
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

    def assert_unlicensed_collision(self, scenario: str) -> None:
        result = self.run_user_scenario(scenario)
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(UNLICENSED_COLLISION_MESSAGE, result.stdout)

    def test_symbolica_inside_library_does_not_collide(self) -> None:
        self.assertIn("library_value", self.assert_succeeds("library"))

    def test_plain_callback_returns_to_skipped_library_scope(self) -> None:
        output = self.assert_succeeds("plain-callback")
        self.assertIn("42", output)
        self.assertIn("library_after_callback", output)

    def test_symbolica_inside_user_callback_inherits_oem_scope(self) -> None:
        self.assertIn("user_callback_value", self.assert_succeeds("symbolica-callback"))

    def test_symbolica_directly_inside_user_file_collides(self) -> None:
        self.assert_unlicensed_collision("direct-user")

    def test_user_code_after_library_scope_collides(self) -> None:
        self.assert_unlicensed_collision("library-then-user")

    def test_declared_thread_allowance_succeeds(self) -> None:
        self.assertIn("8", self.assert_succeeds("threads-at-limit"))

    def test_extra_thread_exceeds_oem_allowance(self) -> None:
        result = self.run_user_scenario("threads-over-limit")
        self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn(OEM_CONCURRENCY_MESSAGE, result.stdout)

    def test_copied_token_is_rejected_outside_package(self) -> None:
        self.assertIn("cannot be activated", self.assert_succeeds("copied-token"))

    def test_extra_oem_process_exceeds_allowance(self) -> None:
        processes: list[subprocess.Popen[str]] = []
        try:
            for index in range(4):
                ready_path = Path(self._temporary_directory.name) / f"oem-ready-{index}"
                process = subprocess.Popen(
                    [
                        sys.executable,
                        str(FIXTURE_DIR / "hold_oem_process.py"),
                        str(ready_path),
                    ],
                    cwd=FIXTURE_DIR,
                    env=self.environment,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
                processes.append(process)

                deadline = time.monotonic() + 10
                while not ready_path.exists():
                    if process.poll() is not None:
                        output = process.stdout.read() if process.stdout else ""
                        self.fail(f"OEM holder exited early ({process.returncode}):\n{output}")
                    if time.monotonic() >= deadline:
                        self.fail("timed out waiting for OEM process holder")
                    time.sleep(0.02)

            result = subprocess.run(
                [
                    sys.executable,
                    str(FIXTURE_DIR / "hold_oem_process.py"),
                    str(Path(self._temporary_directory.name) / "oem-ready-extra"),
                ],
                cwd=FIXTURE_DIR,
                env=self.environment,
                input="",
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn(OEM_CONCURRENCY_MESSAGE, result.stdout)
        finally:
            for process in processes:
                if process.stdin:
                    process.stdin.close()
            for process in processes:
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait(timeout=5)
                if process.stdout:
                    process.stdout.close()


if __name__ == "__main__":
    unittest.main()
