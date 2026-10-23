"""Exact floating expression text and portable, versioned pickle regression tests."""
import base64
import pickle
import subprocess
import sys
import unittest

from symbolica import E, N, S, Float


class TrackedRoundtripTests(unittest.TestCase):
    def test_repr_preserves_tracking(self):
        for expression in [
            E("1.25`20"), E("0.``-4000"), E("1e-4000`80"),
            E("1e4000`80"), N(Float(1, precision=24)) ** 0,
            E("1.25`20*x+0.``8*y"),
        ]:
            self.assertEqual(E(repr(expression)), expression)
            self.assertEqual(E(expression.format_plain()), expression)

    def test_pickle_preserves_tracking(self):
        expression = E("1.25`20*pickle_roundtrip::x+0.``-4000")
        self.assertEqual(pickle.loads(pickle.dumps(expression)), expression)
        reconstruct, args = expression.__reduce__()
        self.assertTrue(args[0].startswith(b"SYMPKL\x01"))
        for invalid in [b"", b"\x01\x00", b"SYMPKL\x00", args[0][:-1], args[0] + b"trailing"]:
            with self.assertRaises(ValueError):
                reconstruct(invalid)

    def test_pickle_restores_symbols_in_a_fresh_process(self):
        expression = E("1.25`20*pickle_fresh::x+0.``-4000*pickle_fresh::y")
        payload = base64.b64encode(pickle.dumps(expression)).decode("ascii")
        script = """
import base64, pickle, sys
from symbolica import S
S('unrelated_symbol_registered_before_import')
print(repr(pickle.loads(base64.b64decode(sys.argv[1]))))
"""
        restored = subprocess.check_output([sys.executable, "-c", script, payload], text=True).strip()
        self.assertEqual(E(restored), expression)


if __name__ == "__main__":
    unittest.main()
