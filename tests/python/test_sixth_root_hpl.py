"""Exact endpoint tags and precision-aware sixth-root HPL evaluation."""

import unittest
from decimal import Decimal, localcontext

from symbolica import E, Expression, Symbol


class SixthRootHplTests(unittest.TestCase):
    def test_constructor_and_parsing_in_the_python_namespace(self):
        value = Expression.hpl_at_sixth_root([1, 0])
        self.assertEqual(value, E("hpl_at_sixth_root(hpl_indices(1,0),1)"))
        self.assertEqual(str(Symbol.HPL_AT_SIXTH_ROOT), "hpl_at_sixth_root")

    def test_precision_escalation_and_conjugation(self):
        value = Expression.hpl_at_sixth_root([1, 0])
        low = value.evaluate({}, 20)
        high = value.evaluate({}, 80)
        self.assertGreater(high.precision, low.precision)
        self.assertGreaterEqual(high.precision, 260)
        with localcontext() as context:
            context.prec = 100
            reference = Decimal(
                "-1.01494160640965362502120255427452028594168930753029979201748910677"
            )
            self.assertLess(abs(high.imag.to_decimal() - reference), Decimal("1e-62"))
        negative = Expression.hpl_at_sixth_root([1, 0], -1)
        self.assertEqual(high.real, negative.evaluate({}, 80).real)
        self.assertEqual(high.imag, -negative.evaluate({}, 80).imag)

    def test_invalid_words_and_signs(self):
        for word, sign in [([2], 1), ([1] * 7, 1), ([1], 0)]:
            with self.subTest(word=word, sign=sign), self.assertRaises(ValueError):
                Expression.hpl_at_sixth_root(word, sign)
        with self.assertRaisesRegex(ValueError, "sign"):
            E("hpl_at_sixth_root(hpl_indices(1),0)").evaluate({})


if __name__ == "__main__":
    unittest.main()
