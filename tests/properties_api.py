"""Mathematical property queries preserve unknown results at the Python boundary."""
import unittest
import ast
from pathlib import Path

from symbolica import Expression, S


class PropertyQueries(unittest.TestCase):
    def test_expression_stub_exposes_optional_property_results(self):
        stub = ast.parse((Path(__file__).resolve().parents[1] / "symbolica.pyi").read_text())
        expression = next(n for n in stub.body if isinstance(n, ast.ClassDef) and n.name == "Expression")
        returns = {n.name: ast.unparse(n.returns) for n in expression.body
                   if isinstance(n, ast.FunctionDef) and n.returns}
        for name in ("is_real", "is_integer", "is_scalar", "is_positive", "is_nonnegative"):
            self.assertEqual(returns[name], "bool | None")
        for name in ("is_finite", "is_constant"):
            self.assertEqual(returns[name], "bool")

    def test_unknown_is_not_false(self):
        x = S("python_property_unknown")
        for name in ("is_real", "is_integer", "is_scalar", "is_positive", "is_nonnegative"):
            self.assertIsNone(getattr(x, name)(), name)
        self.assertIsNone(S("python_property_no_assumption", is_real=False).is_real())
        self.assertIs((x + 1j).is_real(), None)
        r = S("python_property_real", is_real=True)
        self.assertIs(r.is_real(), True)
        self.assertIs((r + 1j).is_real(), False)
        self.assertIs(r.is_integer(), None)

    def test_strict_and_weak_inequalities(self):
        zero = Expression.parse("0")
        self.assertIs(zero.is_positive(), False)
        self.assertIs(zero.is_nonnegative(), True)
        r = S("python_property_sign_real", is_real=True)
        self.assertIs((r**2).is_positive(), None)
        self.assertIs((r**2).is_nonnegative(), True)
        self.assertIs((r**2 + 1).is_positive(), True)
        self.assertIs((-r**2).is_positive(), False)
        self.assertIs((-r**2).is_nonnegative(), None)

    def test_integer_reciprocals_and_numeric_answers(self):
        n = S("python_property_integer", is_integer=True)
        self.assertIs((n**2).is_integer(), True)
        self.assertIs((1/n).is_integer(), None)
        self.assertIs(Expression.parse("1/2").is_integer(), False)
        self.assertIs(Expression.parse("2").is_integer(), True)
        self.assertIs(Expression.parse("-2").is_positive(), False)
        self.assertIs(Expression.num(2.0).is_integer(), True)
        self.assertIs(Expression.num(0.5).is_integer(), False)

    def test_structural_queries_stay_boolean(self):
        x = S("python_property_structural")
        self.assertIs(x.is_constant(), False)
        self.assertIs((1/x).is_finite(), True)

    def test_solver_uses_negative_proofs_and_preserves_unknown_conditions(self):
        r = S("python_property_solver_real", is_real=True)
        p = S("python_property_solver_positive", is_positive=True)
        self.assertTrue(Expression.solve(p + r**2, [p]).is_empty())
        self.assertFalse(Expression.solve(p - r**2 - 1, [p])[0].is_conditional())
        self.assertTrue(Expression.solve(p - r**2, [p])[0].is_conditional())


if __name__ == "__main__":
    unittest.main()
