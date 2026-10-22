"""Run against the standalone Numerica module or an embedding application's module."""
import cmath
import copy
import importlib
import math
import os
import random
import re
from pathlib import Path
import unittest
from decimal import Decimal, localcontext, ROUND_DOWN

module = importlib.import_module(os.environ.get("NUMERICA_PYTHON_MODULE", "numerica"))
Float, ComplexFloat = module.Float, module.ComplexFloat


class PythonFloats(unittest.TestCase):
    def test_rich_display_preserves_precision(self):
        class Pretty:
            def __init__(self):
                self.output = ""

            def text(self, value):
                self.output += value

        real = Float.from_ratio(1, 3, decimal_digits=80)
        imag = Float.from_ratio(-1, 7, decimal_digits=60)
        self.assertEqual(str(real), "0." + "3" * 80)
        real_text, imag_text = str(real), str(imag)
        values = [
            (real, f"$${real_text}$$"),
            (ComplexFloat(real, imag), f"$${real_text}{imag_text}\\,i$$"),
        ]
        with localcontext() as context:
            context.prec = 3
            for value, latex in values:
                with self.subTest(value=type(value).__name__):
                    self.assertEqual(value._repr_html_(), f"<pre>{str(value)}</pre>")
                    self.assertEqual(value._repr_latex_(), latex)
                    pretty = Pretty()
                    value._repr_pretty_(pretty, False)
                    self.assertEqual(pretty.output, str(value))
                    pretty = Pretty()
                    value._repr_pretty_(pretty, True)
                    self.assertEqual(pretty.output, "...")

    def test_rich_display_scientific_notation_and_special_values(self):
        for text, expected in [
            ("1.25e100", r"1.25\times 10^{100}"),
            ("-1.25e-100", r"-1.25\times 10^{-100}"),
            ("-0", "-0"),
            ("Infinity", r"\infty"),
            ("-Infinity", r"-\infty"),
            ("NaN", r"\mathrm{NaN}"),
        ]:
            value = Float(text, decimal_digits=80)
            with self.subTest(value=text):
                self.assertEqual(value._repr_latex_(), f"$${expected}$$")
                self.assertEqual(value._repr_html_(), f"<pre>{str(value)}</pre>")
                complex_value = ComplexFloat(2, value)
                sign = "" if expected.startswith("-") else "+"
                self.assertEqual(complex_value._repr_latex_(), f"$$2{sign}{expected}\\,i$$")

    def test_constructors_and_precision(self):
        self.assertEqual(Float(), 0)
        for value in [123, 123.0, "123", Decimal("123")]:
            self.assertEqual(Float(value), 123)
        n = 10**100 + 123
        self.assertEqual(Float(n).as_integer_ratio(), (n, 1))
        x = Float("1.2345678901234567890123456789", decimal_digits=80)
        self.assertEqual(x.precision, 266)
        self.assertEqual(Float(x).precision, 266)
        self.assertEqual(Float(x).as_integer_ratio(), x.as_integer_ratio())
        self.assertEqual(x.with_precision(precision=120).precision, 120)
        self.assertEqual(Float(1.25, precision=80).to_decimal(), Decimal("1.25"))
        for kwargs in [{"precision": 0}, {"decimal_digits": 0}, {"precision": 80, "decimal_digits": 20}]:
            with self.assertRaises(ValueError):
                Float(1, **kwargs)
        for invalid in ["1e", "not a number", "1.2.3"]:
            with self.assertRaises(ValueError):
                Float(invalid)
        with self.assertRaises(TypeError):
            Float(object())
        with self.assertRaises(AttributeError):
            x.precision = 53

    def test_exact_context_independent_decimal_conversion(self):
        x = Float(0.1)
        expected = Decimal.from_float(0.1)
        with localcontext() as context:
            context.prec = 3
            context.rounding = ROUND_DOWN
            self.assertEqual(x.to_decimal(), expected)
            self.assertEqual(x.to_decimal(10), Decimal("0.1"))
            self.assertEqual(x.as_integer_ratio(), (3602879701896397, 36028797018963968))
        with self.assertRaises(ValueError):
            x.to_decimal(0)
        x = Float.from_ratio(1, 3, decimal_digits=80)
        with localcontext() as context:
            context.prec = 100
            self.assertLess(abs(x.to_decimal() - Decimal(1)/Decimal(3)), Decimal("1e-80"))
        with self.assertRaises(ZeroDivisionError):
            Float.from_ratio(1, 0)

    def test_printing_repr_and_copy(self):
        self.assertEqual(str(Float("1.25", precision=100)), "1.25")
        self.assertEqual(format(Float("1.25"), ".3f"), "1.250")
        for text in ["0.1", "-0", "1.234567890123456789012345678901", "1e100", "-1e-100"]:
            x = Float(text, precision=200)
            with localcontext() as context:
                context.prec = 3
                y = eval(repr(x), {"Float": Float})
                self.assertEqual(x.as_integer_ratio(), y.as_integer_ratio())
                self.assertEqual(x.precision, y.precision)
            self.assertEqual(copy.copy(x), x)
            self.assertEqual(copy.deepcopy(x), x)
        for zero in ["-0", -0.0, Decimal("-0")]:
            self.assertTrue(Float(zero).to_decimal().is_signed())
            self.assertEqual(math.copysign(1, float(Float(zero))), -1)

    def test_arithmetic_keeps_precision_for_native_operands(self):
        x = Float("1.25", precision=200)
        for value, expected in [(x+2, "3.25"), (2+x, "3.25"), (x*2, "2.5"),
                                (2-x, "0.75"), (x/2, "0.625"), (Decimal("2")+x, "3.25")]:
            self.assertIsInstance(value, Float)
            self.assertGreaterEqual(value.precision, 199)
            self.assertEqual(value.to_decimal(), Decimal(expected))
        self.assertEqual(x**2, Float("1.5625"))
        self.assertEqual(Float(2, precision=200)**-3, Float("0.125"))
        self.assertEqual(abs(-x), x)
        self.assertEqual(int(Float("-3.9")), -3)
        self.assertEqual(float(x), 1.25)
        self.assertFalse(bool(Float(0)))
        self.assertTrue(bool(x))
        with self.assertRaises(ZeroDivisionError):
            x / 0
        with self.assertRaises(TypeError):
            x + "1.25"
        self.assertEqual(Float(4, precision=100).sqrt(), 2)
        self.assertEqual(Float(0).exp(), 1)

    def test_zero_to_negative_zero_power(self):
        for exponent in [-0.0, Decimal("-0"), Float("-0")]:
            with self.subTest(exponent=exponent):
                self.assertEqual(Float(0).powf(exponent), 1)
                self.assertEqual(Float(0) ** exponent, 1)
                self.assertEqual(ComplexFloat(0).powf(exponent), 1)
        for exponent in [-1, Float(-1)]:
            with self.assertRaises(ZeroDivisionError):
                Float(0).powf(exponent)
            with self.assertRaises(ZeroDivisionError):
                Float(0) ** exponent
            with self.assertRaises(ZeroDivisionError):
                ComplexFloat(0).powf(exponent)

    def test_numeric_equality_nan_and_hashing(self):
        self.assertEqual(Float(1, precision=53), Float(1, precision=200))
        self.assertEqual(Float(1), Decimal(1))
        self.assertEqual(Float(1), 1+0j)
        self.assertEqual(1+0j, Float(1))
        self.assertNotEqual(Float(1), 1+1j)
        self.assertEqual(Float(0.1), 0.1)
        self.assertNotEqual(Float(0.1), Decimal("0.1"))
        self.assertLess(Float("1.25"), Decimal("1.2500000000000000000001"))
        self.assertFalse(Float(1) == "1")
        with self.assertRaises(TypeError):
            hash(Float(1))
        nan = Float("NaN", precision=100)
        self.assertTrue(nan.is_nan())
        self.assertFalse(nan.is_finite())
        self.assertNotEqual(nan, nan)
        self.assertFalse(nan < Float(1))
        self.assertTrue(nan.to_decimal().is_nan())
        self.assertTrue(Float("Infinity").is_infinite())
        self.assertEqual(float(Float("-Infinity")), -math.inf)
        with self.assertRaises(ValueError):
            nan.as_integer_ratio()

    def test_complex_constructors_components_and_conversions(self):
        for z in [ComplexFloat(1.25, -2.5), ComplexFloat(1.25-2.5j),
                  ComplexFloat((Decimal("1.25"), Decimal("-2.5"))), ComplexFloat("1.25-2.5j")]:
            self.assertIsInstance(z.real, Float)
            self.assertIsInstance(z.imag, Float)
            self.assertEqual(z.to_decimal_tuple(), (Decimal("1.25"), Decimal("-2.5")))
            self.assertEqual(complex(z), 1.25-2.5j)
            self.assertEqual(z.as_tuple(), (Float("1.25"), Float("-2.5")))
        self.assertEqual(ComplexFloat("-j"), ComplexFloat(0, -1))
        self.assertEqual(ComplexFloat("1e-20+2e-20j").real, Float("1e-20"))
        z = ComplexFloat(Float(1, precision=100), Float(2, precision=200))
        self.assertEqual((z.real.precision, z.imag.precision), (100, 200))
        self.assertEqual(ComplexFloat(z).imag.precision, 200)
        self.assertEqual(z.with_precision(decimal_digits=60).real.precision, 200)
        with self.assertRaises(AttributeError):
            z.real = Float(2)
        with self.assertRaises(ValueError):
            ComplexFloat((1, 2, 3))

    def test_complex_printing_arithmetic_and_comparisons(self):
        z = ComplexFloat("1.25", "-2.5", precision=180)
        self.assertEqual(str(z), "(1.25-2.5j)")
        self.assertEqual(format(z, ".2f"), "(1.25-2.50j)")
        r = eval(repr(z), {"Float": Float, "ComplexFloat": ComplexFloat})
        self.assertEqual(r.to_decimal_tuple(), z.to_decimal_tuple())
        self.assertEqual(r.precision, z.precision)
        self.assertEqual(z.conjugate(), ComplexFloat("1.25", "2.5"))
        self.assertEqual(ComplexFloat(3, 4).__abs__(), Float(5))
        self.assertEqual(z + 2, ComplexFloat("3.25", "-2.5"))
        self.assertEqual(2 - z, ComplexFloat("0.75", "2.5"))
        self.assertEqual(ComplexFloat(0, 1)**2, ComplexFloat(-1, 0))
        self.assertEqual(ComplexFloat(0, 1)**-1, ComplexFloat(0, -1))
        self.assertEqual(ComplexFloat(1, 2), 1+2j)
        self.assertEqual(ComplexFloat(1, 0), 1)
        self.assertNotEqual(ComplexFloat(1, 0), (1, 0))
        self.assertNotEqual(ComplexFloat(0.1, 0), Decimal("0.1"))
        with self.assertRaises(TypeError):
            z < z
        with self.assertRaises(TypeError):
            hash(z)
        with self.assertRaises(ZeroDivisionError):
            z / 0
        self.assertEqual(Float(1, precision=100) + 2j, ComplexFloat(1, 2))


class CompleteFloatAPI(unittest.TestCase):
    def test_trait_surface_is_exposed(self):
        source = (Path(__file__).parents[1] / "src/domains/float.rs").read_text()
        for cls, traits in [(Float, ["FloatLike", "SingleFloat", "RealLike", "Real"]),
                            (ComplexFloat, ["FloatLike", "SingleFloat", "Real"])]:
            for trait in traits:
                body = source.split(f"pub trait {trait}:", 1)[1].split("\n}", 1)[0]
                for name in re.findall(r"    fn (\w+)", body):
                    # These Rust trait methods are not part of the Python API.
                    if name in {"real_cmp", "needs_rescaling", "copy_sign"}:
                        continue
                    with self.subTest(cls=cls.__name__, trait=trait, method=name):
                        self.assertTrue(callable(getattr(cls, name, None)))

    def test_constants_have_requested_accuracy(self):
        values = {
            "pi": "3.141592653589793238462643383279502884197169399375105820974944592307",
            "e": "2.718281828459045235360287471352662497757247093699959574966967627724",
            "euler": "0.577215664901532860606512090082402431042159335939923598805767234885",
            "phi": "1.618033988749894848204586834365638117720309179805762862135448622705",
        }
        with localcontext() as ctx:
            ctx.prec = 90
            for cls in [Float, ComplexFloat]:
                for name, text in values.items():
                    with self.subTest(cls=cls.__name__, constant=name):
                        value = getattr(cls, name)(decimal_digits=60)
                        self.assertEqual(value.precision, 200)
                        self.assertLess(abs(value.real.to_decimal() - Decimal(text)), Decimal("1e-59"))
                        self.assertEqual(value.imag, 0)
                        self.assertEqual(getattr(cls, name)(precision=120).precision, 120)
                        with self.assertRaises(ValueError):
                            getattr(cls, name)(decimal_digits=0)
                        with self.assertRaises(ValueError):
                            getattr(cls, name)(precision=80, decimal_digits=20)
                self.assertEqual(cls.euler_gamma(precision=100), cls.euler(precision=100))
        self.assertEqual(ComplexFloat.i(decimal_digits=60), 1j)
        self.assertEqual(ComplexFloat.i(decimal_digits=60).precision, 200)
        self.assertIsNone(Float.i())

    def test_real_elementary_functions(self):
        names = ["sqrt", "exp", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "atanh"]
        x = Float("0.375", precision=200)
        for name in names:
            with self.subTest(method=name):
                result = getattr(x, name)()
                self.assertIsInstance(result, Float)
                self.assertAlmostEqual(float(result), getattr(math, name)(0.375), places=14)
        self.assertAlmostEqual(float(Float(2).acosh()), math.acosh(2), places=14)
        for name, inverse in [("sin", "asin"), ("tan", "atan"), ("sinh", "asinh"), ("tanh", "atanh")]:
            self.assertLess(abs(getattr(getattr(x, name)(), inverse)() - x), Decimal("1e-57"))
        for y, other in [(1, 1), (1, -1), (-1, -1), (-1, 1), (1, 0)]:
            self.assertAlmostEqual(float(Float(y).atan2(other)), math.atan2(y, other), places=14)
        for y, other in [(math.inf, 1), (1, -math.inf), (-math.inf, math.inf), (math.inf, -math.inf)]:
            self.assertAlmostEqual(float(Float(y).atan2(other)), math.atan2(y, other), places=14)
        self.assertTrue(Float(1).atan2(Float("NaN")).is_nan())
        self.assertTrue(Float(2).asin().is_nan())
        self.assertEqual(Float(4, precision=200).powf(Decimal("0.5")), 2)
        self.assertEqual(Float(4, precision=200).pow(3), 64)
        for exponent in [-1.0, Float(-1)]:
            with self.assertRaises(ZeroDivisionError):
                Float(0).powf(exponent)
            with self.assertRaises(ZeroDivisionError):
                Float(0) ** exponent

    def test_stable_elementary_functions(self):
        for cls, value, reference in [
            (Float, 0.375, math),
            (ComplexFloat, 0.375 + 0.25j, cmath),
            (ComplexFloat, -2 + 0.25j, cmath),
            (ComplexFloat, -2 - 0.25j, cmath),
        ]:
            x = cls(value, precision=200)
            for name, expected in [
                ("log1p", reference.log(1 + value)),
                ("sech", 1 / reference.cosh(value)),
                ("csch", 1 / reference.sinh(value)),
            ]:
                with self.subTest(cls=cls.__name__, method=name, value=value):
                    result = getattr(x, name)()
                    self.assertIsInstance(result, cls)
                    self.assertLess(abs((float(result) if cls is Float else complex(result)) - expected), 2e-14)
                    self.assertGreater(result.precision, 180)

        for cls in [Float, ComplexFloat]:
            tiny = cls("1e-100", precision=200)
            self.assertLess(abs(tiny.log1p() / tiny - 1), Decimal("1e-55"))
            self.assertLess(abs(tiny.csch() * tiny - 1), Decimal("1e-55"))
            for value in ["1e13", "-1e13"]:
                x = cls(value, precision=200)
                self.assertEqual(x.sech(), 0)
                self.assertEqual(x.csch(), 0)
            self.assertEqual(cls(0).sech(), 1)

        for x in [-2.0, -0.5, 0.0, 2.0]:
            for y in [0.0, -0.0]:
                z = complex(x, y)
                actual = complex(ComplexFloat(z, precision=200).log1p())
                expected = cmath.log(complex(1 + x, y))
                self.assertLess(abs(actual - expected), 2e-14)
                self.assertEqual(math.copysign(1, actual.imag), math.copysign(1, y))
        for y in [0.0, -0.0]:
            result = complex(ComplexFloat(complex(-1, y), precision=200).log1p())
            self.assertEqual(result.real, -math.inf)
            self.assertEqual(result.imag, 0)
            self.assertEqual(math.copysign(1, result.imag), math.copysign(1, y))

    def test_hypot_numeric_operands_and_scaling(self):
        x = Float(3, precision=200)
        for other in [4, 4.0, Decimal(4), Float(4, precision=200)]:
            result = x.hypot(other)
            self.assertIsInstance(result, Float)
            self.assertEqual(result, 5)
            self.assertGreater(result.precision, 180)
        z = ComplexFloat(3, 4, precision=200)
        for other in [12, 12.0, 12j, Decimal(12), Float(12, precision=200),
                      ComplexFloat(0, 12, precision=200)]:
            result = z.hypot(other)
            self.assertIsInstance(result, Float)
            self.assertLess(abs(result - 13), Decimal("1e-55"))
            self.assertGreater(result.precision, 180)
        for cls in [Float, ComplexFloat]:
            for scale in ["1e1000", "1e-1000"]:
                x = cls(scale, precision=200)
                self.assertLess(abs(x.hypot(x) / abs(x) - Float(2, precision=200).sqrt()),
                                Decimal("1e-55"))
            for invalid in ["4", (3, 4), object()]:
                with self.assertRaises(TypeError):
                    cls(3).hypot(invalid)
        with self.assertRaises(TypeError):
            Float(3).hypot(4j)

    def test_complex_elementary_functions_and_powers(self):
        names = ["sqrt", "exp", "sin", "cos", "tan", "asin", "acos", "atan", "sinh", "cosh", "tanh", "asinh", "acosh", "atanh"]
        for native in [0.375+0.25j, -0.375+0.25j, -0.375-0.25j, 0.375-0.25j]:
            z = ComplexFloat(native, precision=200)
            for name in names:
                with self.subTest(method=name, value=native):
                    result = getattr(z, name)()
                    self.assertIsInstance(result, ComplexFloat)
                    self.assertLess(abs(complex(result) - getattr(cmath, name)(native)), 2e-14)
            self.assertLess(abs(complex(z.powf(0.5+0.25j)) - native**(0.5+0.25j)), 2e-14)
            self.assertLess(abs(complex(z**ComplexFloat(0.5, 0.25)) - native**(0.5+0.25j)), 2e-14)
            self.assertLess(abs(complex(z.atan2(2)) - cmath.atan(native/2)), 2e-14)
            self.assertEqual(z.pow(3), z**3)
        self.assertAlmostEqual(complex(ComplexFloat(1).atan2(-1)).real, math.atan2(1, -1), places=14)
        z = ComplexFloat("0.375", "0.25", precision=200)
        self.assertLess(abs(z.sinh().asinh() - z), Decimal("1e-55"))
        self.assertEqual(ComplexFloat(0)**0, 1)
        with self.assertRaises(ZeroDivisionError):
            ComplexFloat(0).powf(1j)
        with self.assertRaises(TypeError):
            z.powf("0.5")

    def test_signed_branch_cuts_preserve_accuracy(self):
        for name in ["asin", "acos", "atan", "asinh", "acosh", "atanh", "sqrt", "log"]:
            for native in [complex(2, 0), complex(-2, 0), complex(2, -0.0), complex(-2, -0.0),
                           complex(0, 2), complex(0, -2), complex(-0.0, 2), complex(-0.0, -2)]:
                with self.subTest(method=name, value=native):
                    result = getattr(ComplexFloat(native, precision=200), name)()
                    expected = getattr(cmath, name)(native)
                    self.assertLess(abs(complex(result)-expected), 2e-14)
                    self.assertGreater(result.precision, 180)
        self.assertTrue(ComplexFloat(1).atanh().real.is_infinite())
        self.assertEqual(ComplexFloat(1).atanh().imag, 0)
        self.assertTrue(ComplexFloat(0, 1).atan().imag.is_infinite())
        for y in [0.0, -0.0]:
            for x in [0.0, -0.0, -1.0, 1.0]:
                result = float(Float(y).atan2(x))
                self.assertAlmostEqual(result, math.atan2(y, x), places=14)
                self.assertEqual(math.copysign(1, result), math.copysign(1, y))
        for base, exponent, expected in [(4, "0.5", 2), (16, "0.25", 2), (16, "-0.25", 0.5), (4, "1.5", 8)]:
            self.assertEqual(Float(base, precision=200).powf(Float(exponent, precision=200)), expected)
        tiny = Float("1e-60", precision=200)
        self.assertEqual((tiny + Float(0, precision=200)).precision, tiny.precision)

    def test_precision_arithmetic_and_conversion_helpers(self):
        for cls in [Float, ComplexFloat]:
            x = cls(3, precision=200)
            self.assertEqual(x.get_precision(), x.precision)
            self.assertEqual(x.get_epsilon(), 2**-200)
            self.assertFalse(x.fixed_precision())
            for value in [x.zero(), x.one(), x.nan(), x.from_usize(2), x.from_i64(-2)]:
                self.assertEqual(value.precision, 200)
            self.assertTrue(x.zero().is_zero())
            self.assertTrue(x.zero().is_fully_zero())
            self.assertTrue(x.one().is_one())
            self.assertTrue(x.nan().is_nan())
            self.assertEqual(x.neg(), -x)
            self.assertEqual(x.conj(), x.conjugate())
            self.assertEqual(x.norm(), abs(x))
            self.assertEqual(x.mul_add(2, -1), 5)
            self.assertEqual(x.inv()*3, 1)
            self.assertEqual(x.from_rational(1, 4), 0.25)
            self.assertEqual(cls.from_ratio(1, 4, precision=200).precision, 200)
            self.assertEqual(cls.new_zero(precision=80).precision, 80)
            self.assertEqual(cls.new_one(decimal_digits=60).precision, 200)
            other = cls(5, precision=120)
            self.assertEqual(x.set_from(other).precision, 120)
            self.assertEqual(x, 3)
            with self.assertRaises(ZeroDivisionError):
                x.zero().inv()
            with self.assertRaises(ZeroDivisionError):
                x.from_rational(1, 0)
        self.assertEqual(Float("1.25").to_f64(), 1.25)
        for text, expected in [("2.5", 2), ("3.5", 4), ("-2.5", -2), ("-3.5", -4), ("3.9", 4)]:
            self.assertEqual(Float(text).round_to_nearest_integer(), expected)
        self.assertEqual(Float(-3).to_usize_clamped(), 0)
        self.assertEqual(Float("3.9").to_usize_clamped(), 4)
        self.assertGreater(Float("Infinity").to_usize_clamped(), 2**31)
        for bad in ["NaN", "Infinity"]:
            with self.assertRaises(ValueError):
                Float(bad).round_to_nearest_integer()
        with self.assertRaises(ValueError):
            Float("NaN").to_usize_clamped()

    def test_sampling_uses_full_precision_and_rng(self):
        for cls in [Float, ComplexFloat]:
            x = cls(precision=200)
            a = x.sample_unit(random.Random(123))
            b = x.sample_unit(random.Random(123))
            self.assertEqual(a, b)
            self.assertEqual(a.precision, 200)
            self.assertEqual(a.imag, 0)
            self.assertGreaterEqual(a.real, 0)
            self.assertLess(a.real, 1)
            self.assertGreater(a.real.as_integer_ratio()[1].bit_length(), 100)
            self.assertIsInstance(x.sample_unit(), cls)
        class BadRng:
            def getrandbits(self, bits): return 1 << bits
        with self.assertRaises(ValueError):
            Float().sample_unit(BadRng())


if __name__ == "__main__":
    unittest.main()
