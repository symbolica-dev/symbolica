"""Run against the standalone Numerica module or an embedding application's module.

A native Python operand of a ComplexFloat whose components have different precisions
must use the larger component precision, like an explicit Float operand at that precision.
"""
import importlib
import os
import unittest
from decimal import Decimal

host = importlib.import_module(os.environ.get("NUMERICA_PYTHON_MODULE", "numerica"))
Float, ComplexFloat = host.Float, host.ComplexFloat


def represented(z):
    return tuple((v.as_integer_ratio(), v.precision) for v in (z.real, z.imag))


class NativeOperandPrecision(unittest.TestCase):
    def examples(self):
        low = Float(0, precision=2)
        high = Float.from_ratio(1, 7, precision=160)
        return (ComplexFloat(low, high), ComplexFloat(high, low))

    def test_native_multiply_preserves_more_precise_component(self):
        coefficient = Float(3, precision=160)
        for z in self.examples():
            self.assertEqual(represented(z * 3), represented(z * coefficient))
            self.assertEqual(represented(3 * z), represented(coefficient * z))

    def test_native_division_and_reverse_preserve_component_precision(self):
        coefficient = Float(3, precision=160)
        for z in self.examples():
            self.assertEqual(represented(z / 3), represented(z / coefficient))
            self.assertEqual(represented(3 / z), represented(coefficient / z))

    def test_native_float_decimal_and_complex_use_same_working_precision(self):
        for z in self.examples():
            for native in (0.25, Decimal("0.25"), 0.25j):
                coefficient = ComplexFloat(native, precision=160)
                self.assertEqual(represented(z * native), represented(z * coefficient))

    def test_numeric_method_operand_uses_more_precise_component(self):
        coefficient = Float(3, precision=160)
        zero = Float(0, precision=160)
        for z in self.examples():
            self.assertEqual(represented(z.mul_add(3, 0)), represented(z.mul_add(coefficient, zero)))

    def test_existing_float_operand_is_not_promoted(self):
        coefficient = Float.from_ratio(1, 3, precision=20)
        for z in self.examples():
            product = z * coefficient
            nonzero = [v for v in (product.real, product.imag) if v != 0]
            self.assertTrue(nonzero)
            self.assertTrue(all(v.precision <= 20 for v in nonzero))
            self.assertEqual(coefficient.precision, 20)

    def test_precision_property_and_input_components_stay_unchanged(self):
        for z in self.examples():
            before = represented(z)
            self.assertEqual(z.precision, 2)
            z * 3
            self.assertEqual(represented(z), before)


if __name__ == "__main__":
    unittest.main()
