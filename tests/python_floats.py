"""Integration regressions for Numerica scalars in Symbolica's Python API."""
import unittest
from decimal import Decimal

from symbolica import ComplexFloat, E, Expression, Float, N, S


class ScalarIntegration(unittest.TestCase):
    def test_expression_conversion_preserves_precision(self):
        x = Float("1.23456789012345678901234567890123456789", decimal_digits=80)
        z = ComplexFloat(x, -x)
        for value in [x, z]:
            expression = N(value)
            self.assertEqual(expression, value)
            self.assertEqual(value, expression)
            result = expression.evaluate({}, 80)
            self.assertIsInstance(result, ComplexFloat)
            expected = ComplexFloat(value)
            self.assertEqual(result.to_decimal_tuple(), expected.to_decimal_tuple())
        self.assertEqual(N(Float("1.25")), N(5)/4)
        self.assertEqual(N(1) + x, N(x) + 1)
        self.assertEqual(x + N(1), N(x) + 1)
        self.assertEqual(N(1) + z, N(z) + 1)
        self.assertEqual(z + N(1), N(z) + 1)
        self.assertEqual(N(1), Float(1))
        self.assertNotEqual(N(1), Float("1.0000000000000000000001"))
        self.assertEqual(N(ComplexFloat("0.3333333", "0.5"), 1e-5), N(1)/3 + Expression.I/2)

    def test_evaluation_and_legacy_inputs(self):
        x = S("scalar_evaluation_x")
        for value in [Float("1.25"), Decimal("1.25"), 1.25]:
            self.assertEqual((x*x).evaluate({x: value}, 60), ComplexFloat("1.5625", 0))
        z = ComplexFloat("1.234567890123456789012345678901", "2.5", decimal_digits=70)
        self.assertEqual(x.evaluate({x: z}, 70).to_decimal_tuple(), z.to_decimal_tuple())
        self.assertEqual(x.evaluate({x: (Decimal("1.25"), Decimal("2.5"))}, 60), ComplexFloat("1.25", "2.5"))
        self.assertIsInstance(x.evaluate({x: z}), complex)
        with self.assertRaises(ValueError):
            x.evaluate({x: z}, 0)

    def test_evaluator_returns_scalars_for_both_precision_backends(self):
        x = S("scalar_evaluator_x")
        for digits in [32, 60]:
            ev = (x*x).evaluator([x])
            for value in [Float("1.25", decimal_digits=digits), Decimal("1.25")]:
                result = ev.evaluate_with_prec([value], digits)
                self.assertIsInstance(result[0], Float)
                self.assertEqual(result[0], Float("1.5625"))
            for value in [ComplexFloat("1.25", "2.5", decimal_digits=digits),
                          (Decimal("1.25"), Decimal("2.5"))]:
                result = ev.evaluate_complex_with_prec([value], digits)
                self.assertIsInstance(result[0], ComplexFloat)
                self.assertEqual(result[0], ComplexFloat("-4.6875", "6.25"))
            self.assertEqual(ev.evaluate([1.25]).dtype.name, "float64")
            self.assertEqual(ev.evaluate_complex([1.25+2.5j]).dtype.name, "complex128")

    def test_root_finders_return_scalars(self):
        x = S("scalar_root_x")
        for init in [1.0, Float(1, decimal_digits=60), Decimal("1.00000000000000000000")]:
            root = (x*x - 2).nsolve(x, init, 1e-12)
            self.assertIsInstance(root, Float)
            self.assertLess(abs(float(root)**2 - 2), 1e-10)
            roots = Expression.nsolve_system([x*x - 2], [x], [init], 1e-12)
            self.assertIsInstance(roots[0], Float)
        polynomial = (x*x-2).to_polynomial()
        for root, multiplicity in polynomial.approximate_roots(100, 1e-20, 60):
            self.assertIsInstance(root, ComplexFloat)
            self.assertEqual(multiplicity, 1)
            self.assertLess(abs(complex(root)**2 - 2), 1e-12)
        self.assertIsInstance(polynomial.approximate_roots(100, 1e-10)[0][0], complex)
        with self.assertRaises(ValueError):
            polynomial.approximate_roots(100, 1e-10, 0)

    def test_arbitrary_precision_callbacks_receive_scalars(self):
        seen = []
        def real(args):
            self.assertIsInstance(args[0], Float)
            seen.append("real")
            return args[0] * 2
        def complex_(args):
            self.assertIsInstance(args[0], ComplexFloat)
            seen.append("complex")
            return args[0] * 2
        f = S("scalar_callback", eval={"decimal": real, "decimal_complex": complex_})
        x = S("scalar_callback_x")
        evaluator = f(x).evaluator([x])
        self.assertEqual(evaluator.evaluate_with_prec([Float("1.25", decimal_digits=60)], 60), [Float("2.5")])
        self.assertEqual(evaluator.evaluate_complex_with_prec([ComplexFloat("1.25", "2.5", decimal_digits=60)], 60), [ComplexFloat("2.5", "5")])
        self.assertIn("real", seen)
        self.assertIn("complex", seen)
        # Existing callback return values remain accepted.
        legacy = S("scalar_callback_legacy", eval={"decimal_complex": lambda args: (Decimal("1.25"), Decimal("2.5"))})
        self.assertEqual(legacy(x).evaluate({x: 1}, 60), ComplexFloat("1.25", "2.5"))

    def test_constant_and_tagged_callbacks(self):
        value = ComplexFloat("1.2345678901234567890123456789", "2.5", decimal_digits=80)
        f = S("scalar_constant", eval={"constant": lambda tags, digits: value})
        self.assertEqual(f().evaluate({}, 80).to_decimal_tuple(), value.to_decimal_tuple())
        calls = []
        def tagged(tags):
            def evaluate(args):
                self.assertIsInstance(args[0], ComplexFloat)
                calls.append(tags)
                return args[0] + 1
            return evaluate
        f = S("scalar_tagged", eval={"tag_count": 1, "decimal_complex": tagged})
        x, tag = S("scalar_tagged_x", "scalar_tag")
        self.assertEqual(f(tag, x).evaluate({x: value}, 80), value + 1)
        self.assertTrue(calls)


if __name__ == "__main__":
    unittest.main()
