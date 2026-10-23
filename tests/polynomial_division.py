"""Polynomial division regressions; run with the built extension on PYTHONPATH."""
import unittest

from symbolica import E, P, S


class PolynomialDivision(unittest.TestCase):
    def polynomial_factories(self):
        return [
            ("rational", P),
            ("prime_two", lambda s: E(s).to_polynomial(modulus=2)),
            ("finite_field", lambda s: E(s).to_polynomial(modulus=5)),
            ("galois_two", lambda s: E(s).to_polynomial(
                modulus=2, minimal_poly=P("a^2+a+1"))),
            ("galois", lambda s: E(s).to_polynomial(
                modulus=5, minimal_poly=P("a^2+2"))),
            ("number_field", lambda s: P(s).to_number_field(P("a^2-2"))),
        ]

    def test_exact_division_rejection_and_zero_in_all_polynomial_classes(self):
        for name, make in self.polynomial_factories():
            with self.subTest(domain=name):
                quotient, divisor = make("x^3+x+1"), make("x^2+x+1")
                dividend = quotient * divisor
                self.assertEqual(dividend / divisor, quotient)
                self.assertEqual(make("0") / divisor, make("0"))
                self.assertEqual(dividend / make("1"), dividend)
                with self.assertRaisesRegex(ValueError, "nonzero remainder"):
                    (dividend + make("1")) / divisor
                with self.assertRaisesRegex(ValueError, "Division by zero"):
                    dividend / make("0")
                with self.assertRaisesRegex(ValueError, "Division by zero"):
                    make("0") / make("0")
                q, r = (dividend + make("1")).quot_rem(divisor)
                self.assertEqual(q, quotient)
                self.assertEqual(r, make("1"))
                self.assertEqual((dividend + make("1")) // divisor, quotient)

    def test_division_rejects_incompatible_fields_even_for_zero_and_one(self):
        pairs = [
            (lambda s: P(s).to_number_field(P("a^2-2")),
             lambda s: P(s).to_number_field(P("a^2-3"))),
            (lambda s: E(s).to_polynomial(modulus=5),
             lambda s: E(s).to_polynomial(modulus=7)),
            (lambda s: E(s).to_polynomial(modulus=5, minimal_poly=P("a^2+2")),
             lambda s: E(s).to_polynomial(modulus=5, minimal_poly=P("a^2+3"))),
            (lambda s: E(s).to_polynomial(modulus=2, minimal_poly=P("a^3+a+1")),
             lambda s: E(s).to_polynomial(modulus=2, minimal_poly=P("a^3+a^2+1"))),
        ]
        for index, (left, right) in enumerate(pairs):
            for numerator in ("a", "0"):
                for denominator in ("a", "1", "0"):
                    with self.subTest(pair=index, numerator=numerator, denominator=denominator):
                        with self.assertRaisesRegex(ValueError, "different rings"):
                            left(numerator) / right(denominator)
            for operation in [lambda a, b: a // b, lambda a, b: a % b,
                              lambda a, b: a.quot_rem(b)]:
                with self.subTest(pair=index, operation=operation):
                    with self.assertRaisesRegex(ValueError, "different rings"):
                        operation(left("a"), right("a"))

    def test_nonmonic_number_field_division_and_variable_unification(self):
        make = lambda s: P(s).to_number_field(P("a^2-2"))
        quotient, divisor = make("x^3+a*x+1"), make("(a+1)*x^2+a*x+1")
        dividend = quotient * divisor
        divisor.reorder([S("a"), S("unused"), S("x")])
        actual = dividend / divisor
        actual.unify_variables(quotient)
        self.assertEqual(actual, quotient)

    def test_multivariate_rational_division_with_fractional_content(self):
        quotient = P("(x+y+z+1)^5/33")
        divisor = P("(x-2*y+3*z+1)^3*7/11")
        self.assertEqual(quotient * divisor / divisor, quotient)
        with self.assertRaisesRegex(ValueError, "nonzero remainder"):
            (quotient * divisor + P("1")) / divisor


if __name__ == "__main__":
    unittest.main()
