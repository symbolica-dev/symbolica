"""Uncertainty survives Python inspection, substitution and evaluator construction."""
import math
import pickle
import unittest

from symbolica import E, N, S, Float, ComplexFloat, Magnitude, TrackedComplexFloat, Evaluator


class TrackedEvaluationTests(unittest.TestCase):
    def test_wide_magnitudes_and_pickle(self):
        for value in [Magnitude(), Magnitude(0.1), Magnitude(1, -50000),
                      Magnitude(1, 50000), Magnitude(float("inf")), Magnitude(float("nan"))]:
            copied = pickle.loads(pickle.dumps(value))
            self.assertEqual((value.significand, value.exponent),
                             (copied.significand, copied.exponent))
            self.assertEqual((eval(repr(value)).significand, eval(repr(value)).exponent),
                             (value.significand, value.exponent))
        for exponent in [-50000, 50000]:
            value = Magnitude(1, exponent)
            self.assertIsNone(value.to_float())
            with self.assertRaises(OverflowError):
                float(value)
        self.assertEqual(float(Magnitude()), 0.)
        self.assertTrue(math.isnan(Magnitude(float("nan")).to_float()))
        with self.assertRaises(ValueError):
            Magnitude.from_parts(1, 0)

    def test_stored_error_and_cancellation(self):
        a = E("1.0``1")
        self.assertGreaterEqual(a.get_error()[0].to_float(), 0.1)
        self.assertTrue(a.get_error()[1].is_zero())
        result = (a-a).evaluate_with_error({}, 80)
        self.assertGreaterEqual(result.real_error.to_float(), 0.2)
        self.assertTrue(result.is_real())
        self.assertIsNone(result.decimal_precision[0])
        self.assertEqual(complex(result), 0j)
        self.assertTrue(N(1).get_error()[0].is_zero())
        with self.assertRaises(TypeError):
            S("error_metadata_symbol").get_error()

    def test_input_errors_and_compiled_constants(self):
        x = S("tracked_input_x")
        value = TrackedComplexFloat(ComplexFloat("1", precision=160), Magnitude(.1), Magnitude())
        expression = E("1.0``1") * x
        direct = expression.evaluate_with_error({x: value}, 60)
        evaluator = expression.evaluator([x])
        built = evaluator.evaluate_with_error([value], 60)[0]
        loaded = Evaluator.load(evaluator.save()).evaluate_with_error([value], 60)[0]
        self.assertEqual(loaded.to_expression(), built.to_expression())
        self.assertEqual(evaluator.evaluate_with_error([value], 60)[0].to_expression(), built.to_expression())
        for result in [direct, built]:
            self.assertGreaterEqual(result.real_error.to_float(), .21)
            self.assertTrue(result.imaginary_error.is_zero())
            self.assertEqual(result.value, ComplexFloat(1))
        again = x.evaluate_with_error({x: direct}, 100)
        self.assertEqual(again.real_error, direct.real_error)
        self.assertEqual(again.precision, direct.precision)

    def test_uncertain_imaginary_zero_and_result_pickle(self):
        value = TrackedComplexFloat(ComplexFloat("2", precision=137),
                                   Magnitude(1, -50000), Magnitude(1, -50000))
        expression = value.to_expression()
        self.assertFalse(value.is_real())
        self.assertEqual(expression.get_error(), (value.real_error, value.imaginary_error))
        copied = pickle.loads(pickle.dumps(value))
        self.assertEqual(copied.to_expression(), expression)
        self.assertEqual(eval(repr(value)).to_expression(), expression)
        result = expression.evaluate_with_error({}, 100)
        self.assertFalse(result.is_real())
        self.assertEqual(result.real_error, value.real_error)
        self.assertEqual(result.imaginary_error, value.imaginary_error)
        self.assertIsNone(result.imaginary_error.to_float())

    def test_real_evaluators_reject_complex_float_coefficients_cleanly(self):
        x = S("tracked_real_guard_x")
        evaluator = (N(ComplexFloat("1", "2")) * x).evaluator([x])
        with self.assertRaises(ValueError):
            evaluator.evaluate([1.])
        for digits in [32, 80]:
            with self.assertRaises(ValueError):
                evaluator.evaluate_with_prec([1.], digits)
        self.assertEqual(evaluator.evaluate_with_error([1], 80)[0].value,
                         ComplexFloat("1", "2"))

    def test_domain_and_argument_errors(self):
        x = S("tracked_error_x")
        with self.assertRaises(ValueError):
            x.evaluate_with_error({}, 50)
        with self.assertRaises(ValueError):
            N(1).evaluate_with_error({}, 0)
        with self.assertRaises(ValueError):
            x.evaluator([x]).evaluate_with_error([], 50)
        with self.assertRaises(TypeError):
            x.evaluate_with_error({x: x+1}, 50)


if __name__ == "__main__":
    unittest.main()
