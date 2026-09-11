"""Solution-set protocol tests. Run with the built symbolica extension on PYTHONPATH."""
import unittest
import ast
import doctest
import io
from pathlib import Path

import symbolica
from symbolica import (
    Expression, S, Reals, Complexes, Integers, Rationals, SolutionSet,
    SolveError, UnsupportedProblem, IncompleteCoverage,
)


class SolutionSetContract(unittest.TestCase):
    def setUp(self):
        self.x, self.y = S("x", "y")
        self.a = S("a", is_real=True)

    def test_booleans_and_empty_assignment(self):
        x = self.x
        for variables in ([], [x]):
            for system in (True, [True], []):
                result = Expression.solve(system, variables, domain=Reals)
                self.assertIsInstance(result, SolutionSet)
                self.assertEqual(len(result), 1)
                self.assertTrue(result)
                self.assertEqual(len(result[0]), 0)
                self.assertEqual(result[0].free_variables(), variables)
                self.assertEqual(result[0].variables, variables)
                self.assertEqual(dict(result[0]), {})
                self.assertEqual(result.coverage, "complete")
                self.assertEqual(result.variables, variables)
                self.assertEqual(result.domain, Reals)
            for system in (False, [False], [True, False]):
                result = Expression.solve(system, variables, domain=Reals)
                self.assertEqual(len(result), 0)
                self.assertTrue(result.is_empty())
                self.assertFalse(result)
                self.assertEqual(result.dimension(), -1)
        self.assertEqual(dict(Expression.solve(True, [])[0]), {})
        self.assertTrue(Expression.solve([1], []).is_empty())

    def test_points_map_variables_directly_to_expressions(self):
        x, y = self.x, self.y
        result = Expression.solve([x.eq(2), y.eq(3)], [x, y])
        b = result[0]
        self.assertIsInstance(b[x], Expression)
        self.assertEqual(b[x], 2)
        self.assertEqual(b[x] + b[y], 5)
        self.assertEqual(dict(b), {x: 2, y: 3})
        self.assertEqual(b.as_dict(), dict(b))
        self.assertEqual(b.get(x), b[x])
        self.assertEqual(b.free_variables(), [])
        self.assertEqual(dict(b), {x: 2, y: 3})
        self.assertEqual(set(b), {x, y})
        self.assertEqual(len(b), 2)
        self.assertTrue(all(isinstance(v, Expression) for v in b.values()))
        self.assertEqual(dict(b.items()), {x: 2, y: 3})
        self.assertEqual(set(b.as_dict()), {x, y})
        self.assertEqual(b.variables, [x, y])
        self.assertEqual(b.dimension(), 0)
        self.assertEqual(b.codimension(), 2)
        self.assertTrue(b.is_point())
        self.assertEqual(dict(result[-1]), dict(b))
        self.assertEqual(len(list(result)), len(result))
        with self.assertRaises(IndexError):
            result[1]
        with self.assertRaises(IndexError):
            result[-2]
        with self.assertRaises(KeyError):
            b[self.a]

    def test_free_family_and_dependency_order(self):
        x, y = self.x, self.y
        result = Expression.solve(x + y - 1, [x, y], domain=Reals)
        b = result[0]
        self.assertEqual(result.variables, [x, y])
        self.assertEqual(b.variables, [x, y])
        self.assertEqual(b.free_variables(), [y])
        self.assertIsNone(b.get(y))
        self.assertNotIn(y, b)
        self.assertEqual(b[x], 1-y)
        self.assertEqual(len(b), 1)
        self.assertEqual(list(b), [x])
        self.assertEqual(b.keys(), [x])
        self.assertEqual(b.values(), [1-y])
        self.assertEqual(b.items(), [(x, 1-y)])
        self.assertEqual(b.as_dict(), {x: 1-y})
        self.assertEqual(dict(b), b.as_dict())
        with self.assertRaises(KeyError):
            b[y]
        self.assertEqual(b.dimension(), 1)
        self.assertFalse(b.is_conditional())
        self.assertFalse(b.is_point())
        ordered = Expression.solve(x+y-1, [y, x], domain=Reals)
        self.assertEqual(ordered[0].variables, [y, x])
        self.assertEqual(ordered[0].free_variables(), [x])
        self.assertNotIn(x, ordered[0])
        self.assertEqual(ordered[0][y], 1-x)
        with self.assertRaises(SolveError):
            Expression.solve(x+y, [x, x])

    def test_unused_unknowns_extend_each_root_with_free_coordinates(self):
        x, y = self.x, self.y
        for domain, count in ((Complexes, 3), (Reals, 1)):
            roots = Expression.solve(x**3 - 1, [x], domain=domain)
            for variables in ([x, y], [y, x]):
                result = Expression.solve([x**3 - 1], variables, domain=domain)
                self.assertEqual(len(result), count)
                self.assertEqual(result.coverage, "complete")
                self.assertEqual(result.parameters, [])
                self.assertEqual(result.dimension(), 1)
                self.assertEqual([dict(b) for b in result], [dict(b) for b in roots])
                for branch in result:
                    self.assertEqual(branch.variables, variables)
                    self.assertEqual(branch.free_variables(), [y])
                    self.assertEqual(branch.dimension(), 1)
                    self.assertEqual(branch.codimension(), 1)
                    self.assertNotIn(y, branch)
                    self.assertFalse(branch.is_point())
                    self.assertNotIn("free:", str(branch))
        self.assertTrue(Expression.solve([x**2 + 1], [x, y], domain=Reals).is_empty())
        a = self.a
        generic = Expression.solve(a*x - 1, [x, y], domain=Reals)
        self.assertEqual(generic.coverage, "generic")
        self.assertEqual(generic.parameters, [a])
        self.assertEqual(dict(generic[0]), {x: 1/a})
        self.assertEqual(generic[0].free_variables(), [y])
        self.assertTrue(generic[0].is_conditional())

    def test_formula_lowering_and_symbolic_truth(self):
        x, y = self.x, self.y
        for condition in (x < 0, x <= 0, x > 0, x >= 0, x.eq(0), x.ne(0)):
            with self.assertRaises(TypeError):
                bool(condition)
            self.assertIsNone(condition.eval())
        with self.assertRaises(UnsupportedProblem):
            Expression.solve([x.eq(1), y < 0], [x, y])
        self.assertTrue(bool(Expression.parse("1.5") < 2))
        self.assertFalse(bool(Expression.parse("1.5").eq(2)))
        self.assertIs(x == y, False)
        self.assertIs(x != y, True)
        for formula in ((x*x+y*y < 1) & y.ne(0), (x < 0) | (y > 0), ~(x < 0), (x >= 0) & (x <= 1), x*y > 0, (x-1)/(x+1) > 0):
            with self.assertRaises(UnsupportedProblem):
                Expression.solve(formula, [x, y], domain=Reals)
        for domain in (Complexes, Integers, Rationals):
            with self.assertRaisesRegex(UnsupportedProblem, "equalities only"):
                Expression.solve(x < 0, [x], domain=domain)
        # Definedness is retained before comparison subtraction cancels 1/x.
        self.assertIsNone((1/x).eq(1/x).eval())
        defined = Expression.solve((1/x).eq(1/x), [x], domain=Reals)
        self.assertEqual(dict(defined[0]), {})
        self.assertEqual(defined[0].free_variables(), [x])
        self.assertTrue(defined[0].is_conditional())
        with self.assertRaises(TypeError):
            Expression.solve(S("x_").eq(0), [x])

    def test_coverage_parameters_and_errors(self):
        x, y, a = self.x, self.y, self.a
        family = Expression.solve(x*y, [x, y], domain=Reals)
        self.assertEqual(dict(family[0]), {x: 0})
        self.assertEqual(family[0].free_variables(), [y])
        self.assertEqual(family.coverage, "generic")
        self.assertIn("y != 0", str(family.coverage_guard))
        for name, value in (("parameters", [a]), ("parameter_domain", Reals),
                            ("variable_order", [x]), ("options", None)):
            with self.assertRaises(TypeError):
                Expression.solve(x-a, [x], **{name: value})
        whole = Expression.solve(True, [x])
        self.assertEqual(whole.parameters, [])
        self.assertFalse(hasattr(whole, "parameter_domain"))
        self.assertEqual(whole.dimension(), 1)
        for system in ([x-1, x-2], [1], x*x+1):
            self.assertTrue(Expression.solve(system, [x], domain=Reals).is_empty())
        with self.assertRaises(UnsupportedProblem):
            Expression.solve(0.1*x, [x])

    def test_rational_equations_keep_operand_definedness(self):
        x, y = self.x, self.y
        for equation, expected in (((x/(x-1)).eq(0), 0), ((1/x).eq(2), Expression.num(1)/2)):
            result = Expression.solve(equation, [x, y])
            self.assertEqual(dict(result[0]), {x: expected})
            self.assertEqual(result[0].free_variables(), [y])
            self.assertEqual(result.coverage, "complete")
        # Subtracting the two sides cancels their common pole, but x=0 is invalid.
        self.assertTrue(Expression.solve((1/x).eq(1/x+x), [x]).is_empty())
        self.assertEqual(dict(Expression.solve(1/x-2, [x])[0]), {x: Expression.num(1)/2})
        result = Expression.solve([(1/x).eq(1/x), y.eq(1)], [x, y])
        self.assertEqual(dict(result[0]), {y: 1})
        self.assertEqual(result[0].free_variables(), [x])
        self.assertTrue(any("x != 0" in str(c) for c in result[0].conditions()))
        for equation in ((1/x).ne(0), (1/x) > 0):
            with self.assertRaises(UnsupportedProblem):
                Expression.solve(equation, [x])

    def test_symbol_domains_filter_roots_and_retain_family_conditions(self):
        r = S("solve_unknown_real", is_real=True)
        p = S("solve_unknown_positive", is_positive=True)
        n = S("solve_unknown_integer", is_integer=True)
        for variable, equation in ((r, r*r+1), (p, p+1), (p, p), (n, 2*n-1)):
            self.assertTrue(Expression.solve(equation, [variable]).is_empty())
        for variable, equation, expected in ((p, p*p-4, 2), (n, (n-2)*(2*n-1), 2)):
            self.assertEqual(dict(Expression.solve(equation, [variable])[0]), {variable: expected})
        q = S("solve_other_positive_unknown", is_positive=True)
        self.assertTrue(Expression.solve(p+q, [p, q]).is_empty())
        for variable in (r, p, n):
            branch = Expression.solve(True, [variable])[0]
            self.assertEqual(dict(branch), {})
            self.assertEqual(branch.free_variables(), [variable])
            self.assertTrue(branch.is_conditional())
            self.assertTrue(branch.conditions())
            self.assertIsNone(branch.dimension())
        branch = Expression.solve(p-self.a, [p])[0]
        self.assertEqual(dict(branch), {p: self.a})
        self.assertTrue(any("a > 0" in str(c) for c in branch.conditions()))
        branch = Expression.solve(r-self.x, [r])[0]
        condition = next(c for c in branch.conditions() if c.kind == "domain_membership")
        self.assertEqual(condition.variable, r)
        self.assertEqual(condition.value, self.x)
        self.assertEqual(condition.domain, Reals)

    def test_redundant_parameterized_linear_equations(self):
        x, y, a = self.x, self.y, self.a
        for equations in ([a*x-1, 2*a*x-2], [2*a*x-2, a*x-1]):
            result = Expression.solve(equations, [x, y])
            self.assertEqual(dict(result[0]), {x: 1/a})
            self.assertEqual(result[0].free_variables(), [y])
            self.assertEqual(result.coverage, "generic")
            self.assertTrue(result[0].is_conditional())
        result = Expression.solve([x-a, 2*x-2*a], [x])
        self.assertEqual(dict(result[0]), {x: a})
        self.assertEqual(result.coverage, "complete")
        # An additional equation must never be silently dropped as redundant.
        conditional = Expression.solve([x-a, x-y], [x])
        self.assertEqual(dict(conditional[0]), {x: a})
        self.assertTrue(conditional[0].is_conditional())
        self.assertEqual(conditional.coverage, "complete")

    def test_complete_parameter_affine_family(self):
        x, y, a = self.x, self.y, self.a
        point = Expression.solve(x-a*a, [x], domain=Reals)
        self.assertEqual(point.coverage, "complete")
        self.assertEqual(point.parameters, [a])
        self.assertEqual(dict(point[0]), {x: a*a})
        family = Expression.solve(x+y-a, [x, y], domain=Reals)
        self.assertEqual(family.dimension(), 1)
        self.assertFalse(family[0].is_conditional())
        conditional = Expression.solve([x-a, x-y], [x], domain=Reals)
        self.assertTrue(conditional[0].is_conditional())

    def test_parameter_domains_combine_context_and_symbol_attributes(self):
        x = self.x
        real = S("solve_real_parameter", is_real=True)
        integer = S("solve_integer_parameter", is_integer=True)
        positive = S("solve_positive_parameter", is_positive=True)
        unrestricted = S("solve_complex_parameter")
        result = Expression.solve(x - real - integer, [x], domain=Reals)
        self.assertEqual(dict(result[0]), {x: real + integer})
        self.assertEqual(set(result.parameters), {real, integer})
        # Complex unknowns can depend on individually restricted parameters.
        result = Expression.solve(x - real - unrestricted, [x])
        self.assertEqual(dict(result[0]), {x: real + unrestricted})
        self.assertEqual(set(result.parameters), {real, unrestricted})
        # The real context is local: it must not change the symbol's attributes.
        self.assertFalse(unrestricted.is_real())
        result = Expression.solve(x - unrestricted, [x], domain=Reals)
        self.assertEqual(dict(result[0]), {x: unrestricted})
        self.assertEqual(result.parameters, [unrestricted])
        self.assertFalse(unrestricted.is_real())
        result = Expression.solve(unrestricted*x - 1, [x], domain=Reals)
        self.assertEqual(dict(result[0]), {x: 1/unrestricted})
        self.assertEqual(result.coverage, "generic")
        self.assertIn("solve_complex_parameter != 0", str(result.coverage_guard))
        result = Expression.solve(x - unrestricted, [x])
        self.assertEqual(dict(result[0]), {x: unrestricted})
        self.assertFalse(unrestricted.is_real())
        # Positivity proves that this coefficient cannot vanish.
        result = Expression.solve(positive*x, [x], domain=Reals)
        self.assertEqual(dict(result[0]), {x: 0})
        self.assertEqual(result.coverage, "complete")
        self.assertTrue(bool(result.coverage_guard))

    def test_python_solve_has_only_implemented_arguments(self):
        x = self.x
        for keyword in ("assumptions", "options"):
            with self.assertRaises(TypeError):
                Expression.solve(x-1, [x], **{keyword: None})
        self.assertFalse(hasattr(Expression.solve(x-1, [x]), "assumptions"))
        self.assertFalse(hasattr(Expression.solve(x-1, [x]), "formula"))
        self.assertFalse(hasattr(symbolica, "SolveOptions"))

    def test_parametric_linear_point_and_coverage_guard(self):
        v1, v2, vin, alpha = S("V1", "V2", "Vin", "α")
        system = [(2 + alpha)*v1 - v2 - vin, -v1 + (1 + alpha)*v2]
        determinant = alpha**2 + 3*alpha + 1
        result = Expression.solve(system, [v1, v2])
        self.assertEqual(result.coverage, "generic")
        solution = dict(result[0])
        self.assertEqual((solution[v1] - (1 + alpha)*vin/determinant).together(), 0)
        self.assertEqual(solution[v2], vin/determinant)
        self.assertIn(str(determinant), str(result.coverage_guard))
        self.assertIn(str(result.coverage_guard), str(result))
        self.assertIn("!= 0", result._repr_html_())
        gain = (solution[v2] / solution[v1]).cancel()
        self.assertEqual(gain, 1/(1 + alpha))
        self.assertAlmostEqual(float(gain.evaluate({alpha: 0.1}, 30).real), 10/11)
        with self.assertRaises(IncompleteCoverage):
            result.is_empty()

    def test_generic_guard_survives_cancellation(self):
        x, a = self.x, self.a
        # The solution has no denominator, but a=0 has infinitely many solutions.
        result = Expression.solve(a*x, [x])
        self.assertEqual(dict(result[0]), {x: 0})
        self.assertEqual(result.coverage, "generic")
        self.assertIn("a != 0", str(result.coverage_guard))
        # A pole in the input remains excluded even when the answer is zero.
        result = Expression.solve(x/a, [x])
        self.assertEqual(dict(result[0]), {x: 0})
        self.assertEqual(result.coverage, "complete")
        self.assertIn("a != 0", str(result[0].conditions()))
        Expression.solve(x/a, [x])

    def test_nonnegative_square_does_not_imply_nonzero(self):
        x, a = self.x, self.a
        result = Expression.solve(a*a*x, [x])
        self.assertEqual(dict(result[0]), {x: 0})
        self.assertEqual(result.coverage, "generic")
        self.assertIn("a != 0", str(result.coverage_guard))
        result = Expression.solve((a*a+1)*x, [x])
        self.assertEqual(result.coverage, "complete")
        positive = S("solve_square_positive", is_positive=True)
        branch = Expression.solve(positive-a*a, [positive])[0]
        self.assertTrue(any("a^2 > 0" in str(c) for c in branch.conditions()))
        result = Expression.solve((1/a**2).eq(1/a**2), [x])
        self.assertTrue(any("a != 0" in str(c) for c in result[0].conditions()))

    def test_parameterized_univariate_equations_reach_the_existing_backend(self):
        x, y = self.x, self.y
        z = S("solve_univariate_parameter")
        for variables in ([x, y], [y, x]):
            for equation in (1/x-z, (1/x).eq(z)):
                result = Expression.solve(equation, variables)
                self.assertEqual(dict(result[0]), {x: 1/z})
                self.assertEqual(result[0].free_variables(), [y])
                self.assertEqual(result[0].variables, variables)
                self.assertEqual(result.parameters, [z])
                self.assertEqual(result.coverage, "generic")
                self.assertIn("solve_univariate_parameter != 0", str(result.coverage_guard))
            result = Expression.solve(x*x-x-z, variables)
            self.assertEqual(len(result), 2)
            self.assertEqual(result.coverage, "complete")
            self.assertEqual({b[x].replace(z, 2) for b in result}, {Expression.num(-1), Expression.num(2)})
            for branch in result:
                self.assertEqual(branch.free_variables(), [y])
                self.assertEqual(branch.variables, variables)
                self.assertFalse(branch.is_conditional())
        # This uses the existing polynomial solver, not a quadratic-only formula.
        cubic = Expression.solve(x**3-z, [x, y])
        self.assertEqual(len(cubic), 3)
        for branch in cubic:
            value = branch[x].replace(z, 8).evaluate({}, 30)
            self.assertLess(abs(complex(value)**3-8), 1e-12)
        real = Expression.solve(1/x-z, [x, y], domain=Reals)
        self.assertEqual(dict(real[0]), {x: 1/z})
        self.assertEqual(len(real[0].conditions()), 1)
        real = Expression.solve(x*x-x-z, [x, y], domain=Reals)
        for branch in real:
            self.assertTrue(any(c.kind == "domain_membership" for c in branch.conditions()))
        # Keep an input pole even if it disappears from the returned assignment.
        pole = Expression.solve((x-z)/(x-1), [x, y])[0]
        self.assertEqual(pole[x], z)
        self.assertTrue(any("!= 0" in str(c) for c in pole.conditions()))

    def test_quadratic_root_collisions_do_not_exclude_valid_solutions(self):
        x, y = self.x, self.y
        z = S("solve_collision_parameter")
        quarter = Expression.num(1)/4
        for equation, collision, expected in ((x*x-x+z, quarter, Expression.num(1)/2),
                                               (x*x-z, Expression.num(0), Expression.num(0)),
                                               (x*x-z*z, Expression.num(0), Expression.num(0))):
            result = Expression.solve(equation, [x, y])
            self.assertEqual(result.coverage, "complete")
            self.assertTrue(bool(result.coverage_guard))
            self.assertEqual({b[x].replace(z, collision) for b in result}, {expected})
            for branch in result:
                self.assertEqual(branch.conditions(), [])
                self.assertEqual(branch.free_variables(), [y])
            self.assertNotIn("not covered", result._repr_html_())
            specialized = Expression.solve(equation.replace(z, collision), [x, y])
            self.assertEqual(len(specialized), 1)
            self.assertEqual(specialized[0][x], expected)
        # A zero leading coefficient changes the equation's degree, not just
        # whether two valid root formulas happen to coincide.
        result = Expression.solve(z*x*x-x+1, [x, y])
        self.assertEqual(result.coverage, "generic")
        self.assertIn("solve_collision_parameter != 0", str(result.coverage_guard))
        # A pole remains excluded even when it coincides with a repeated root.
        result = Expression.solve((x*x-x+z)/(x-Expression.num(1)/2), [x, y])
        self.assertTrue(all(branch.is_conditional() for branch in result))

    def test_assignment_dicts_use_explicit_branch_selection(self):
        x = self.x
        result = Expression.solve(x*x - 1, [x])
        self.assertEqual([dict(b) for b in result], [b.as_dict() for b in result])
        self.assertFalse(hasattr(result, "as_point_dict"))
        self.assertFalse(hasattr(result[0], "as_point_dict"))

    def test_parameterized_linear_families_follow_coordinate_order(self):
        x, y, a = self.x, self.y, self.a
        result = Expression.solve(a*x+y-1, [y, x])
        self.assertEqual(dict(result[0]), {y: 1-a*x})
        self.assertEqual(result[0].free_variables(), [x])
        self.assertEqual(result.dimension(), 1)
        reverse = Expression.solve(a*x+y-1, [x, y])
        self.assertEqual((reverse[0][x]-(1-y)/a).together(), 0)
        self.assertEqual(reverse[0].free_variables(), [y])
        self.assertEqual(reverse.coverage, "generic")
        self.assertIn("a != 0", str(reverse.coverage_guard))
        # An intermediate symbolic pivot must not exclude a nonsingular case.
        result = Expression.solve([a*x+y-1, x+y-2], [y, x])
        self.assertEqual({k: v.replace(a, 0) for k,v in result[0].items()}, {x: 1, y: 1})
        self.assertEqual(str(result.coverage_guard), f"{a-1} != 0")

    def test_linear_consistency_conditions_and_parameter_only_equations(self):
        x, a = self.x, self.a
        b = S("solve_consistency_parameter")
        equations = [x-a, x-b]
        result = Expression.solve(equations, [x])
        self.assertEqual(dict(result[0]), {x: a})
        self.assertTrue(result[0].is_conditional())
        self.assertEqual(result.coverage, "complete")
        self.assertIn(str(b), str(result[0].conditions()))
        for query in (result.is_empty, lambda: bool(result)):
            with self.assertRaises(IncompleteCoverage): query()
        self.assertIsNone(result.dimension())
        self.assertTrue(Expression.solve([x-a, x-a-1], [x]).is_empty())
        self.assertEqual(dict(Expression.solve([e.replace(b,a) for e in equations], [x])[0]), {x:a})
        for variables in ([], [x]):
            result = Expression.solve(a, variables)
            self.assertEqual(dict(result[0]), {})
            self.assertEqual(result[0].free_variables(), variables)
            self.assertTrue(result[0].is_conditional())

    def test_rational_families_preserve_free_coordinates_and_poles(self):
        x, y, a = self.x, self.y, self.a
        for equation in (x/y, (x/y).eq(0)):
            result = Expression.solve(equation, [x, y])
            self.assertEqual(dict(result[0]), {x: 0})
            self.assertEqual(result[0].free_variables(), [y])
            self.assertIn("y != 0", str(result[0].conditions()))
        for equation, roots in (((x*x-1)/y, {Expression.num(-1),Expression.num(1)}),):
            result = Expression.solve(equation, [x, y])
            self.assertEqual({b[x] for b in result}, roots)
            for branch in result:
                self.assertEqual(branch.free_variables(), [y])
                self.assertIn("y != 0", str(branch.conditions()))
        result = Expression.solve((x*x-a)/y, [x,y])
        self.assertEqual(len(result), 2)
        self.assertTrue(all(b.free_variables()==[y] for b in result))
        for equation, value in ((x/a, Expression.num(0)), (x-1/a, 1/a)):
            result = Expression.solve(equation, [x])
            self.assertEqual(dict(result[0]), {x:value})
            self.assertIn("a != 0", str(result[0].conditions()))

    def test_cancellation_cannot_restore_a_root_at_an_input_pole(self):
        x = self.x
        for expression in ((x*x-2*x+1)/(x-1), 1/(1+1/x)):
            for equation in (expression, expression.eq(0)):
                self.assertTrue(Expression.solve(equation, [x]).is_empty())
        result = Expression.solve((x*x-1)/(x-1), [x])
        self.assertEqual(dict(result[0]), {x:-1})

    def test_linear_families_over_rationals_and_integers(self):
        x, y = self.x, self.y
        for domain in (Rationals, Integers):
            result = Expression.solve(2*x+y-1, [y,x], domain=domain)
            self.assertEqual(dict(result[0]), {y:1-2*x})
            self.assertEqual(result[0].free_variables(), [x])
            self.assertFalse(result[0].is_conditional())
        result = Expression.solve(x+2*y-1, [y,x], domain=Integers)
        condition = next(c for c in result[0].conditions() if c.kind=="domain_membership")
        self.assertEqual(condition.variable, y)
        self.assertEqual(condition.domain, Integers)
        self.assertEqual((condition.value-(1-x)/2).expand(), 0)
        self.assertTrue(Expression.solve(2*x-1, [x], domain=Integers).is_empty())

    def test_conditional_real_roots_do_not_claim_unconditional_nonemptiness(self):
        x, a = self.x, self.a
        result = Expression.solve(x*x+a, [x], domain=Reals)
        self.assertEqual(result.coverage, "complete")
        with self.assertRaises(IncompleteCoverage): result.is_empty()
        self.assertTrue(Expression.solve(x*x+1, [x], domain=Reals).is_empty())

    def test_uniform_symbolic_matrix_keeps_complete_coverage(self):
        x, y, a = self.x, self.y, self.a
        result = Expression.solve([a*x + y - 1, x - 2], [x, y])
        self.assertEqual(result.coverage, "complete")
        self.assertTrue(bool(result.coverage_guard))
        self.assertEqual(dict(result[0]), {x: 2, y: 1 - 2*a})

    def test_exact_solution_expression_outlives_the_set(self):
        x = self.x
        result = Expression.solve(x**5-x-1, [x], domain=Reals)
        value = result[0][x]
        self.assertIsInstance(value, Expression)
        before = str(value)
        del result
        self.assertEqual(str(value), before)
        self.assertIsInstance(value + 1, Expression)

    def test_detached_branches_retain_parameter_conditions(self):
        x, a = self.x, self.a
        result = Expression.solve(a*x-1, [x])
        guard = str(result.coverage_guard)
        branches = [result[0], result[-1], next(iter(result))]
        del result
        for branch in branches:
            self.assertEqual(branch[x], 1/a)
            self.assertTrue(branch.is_point())
            self.assertTrue(branch.is_conditional())
            self.assertEqual(dict(branch), {x: 1/a})
            self.assertEqual(branch.as_dict(), {x: 1/a})
            self.assertEqual(len(branch.conditions()), 1)
            condition = branch.conditions()[0]
            self.assertEqual(condition.kind, "formula")
            self.assertEqual(str(condition.formula), guard)
            self.assertIn(guard, str(branch))
            self.assertIn("!= 0", branch._repr_html_())
            self.assertEqual(branch.dimension(), 0)

    def test_solution_display_shows_answers(self):
        x, y, a = self.x, self.y, self.a
        points = Expression.solve(x*x - 1, [x], domain=Reals)
        for output in (str(points), repr(points)):
            self.assertIn("x = -1", output)
            self.assertIn("x = 1", output)
            self.assertNotIn("SolutionSet over", output)
            self.assertNotIn("2 branches", output)
            self.assertNotIn("coverage:", output)
        self.assertNotIn("<thead>", points._repr_html_())
        for branch in points:
            self.assertIn(branch[x]._repr_html_(), points._repr_html_())
        generic = Expression.solve(a*x-1, [x])
        for output in (str(generic), repr(generic), generic._repr_html_()):
            self.assertNotIn("Parameters:", output)
            self.assertIn("a != 0", output)
            self.assertIn("other parameter values are not covered", output)
            self.assertNotIn("coverage: generic", output)
        family = Expression.solve(x + y - a, [x, y], domain=Reals)
        self.assertEqual(family.parameters, [a])
        self.assertEqual(family[0].free_variables(), [y])
        self.assertEqual(family[0].domain, Reals)
        for output in (str(family), repr(family), family._repr_html_()):
            self.assertNotIn("Parameters:", output)
            self.assertNotIn("free:", output)
        self.assertIn(f"x = {family[0][x]}", str(family))
        self.assertIn(family[0][x]._repr_html_(), family._repr_html_())
        self.assertNotIn("free:", str(family[0]))
        self.assertNotIn("free:", family[0]._repr_html_())

    def test_empty_set_display_differs_from_empty_assignment(self):
        empty = Expression.solve(False, [])
        assignment = Expression.solve(True, [])
        for output in (str(empty), empty._repr_html_()):
            self.assertIn("No solutions", output)
            self.assertNotIn("{}", output)
        for output in (str(assignment), assignment._repr_html_()):
            self.assertIn("{}", output)
            self.assertNotIn("No solutions", output)
            self.assertNotIn("1 branch (", output)

    def test_python_api_does_not_expose_cad_wrappers(self):
        for name in ("SolutionValue", "VariableSolution", "SolutionBound", "ExactValue", "SolveOptions", "SolveFormula", "UnsupportedDomain"):
            self.assertFalse(hasattr(symbolica, name), name)

    def test_notebook_display_escapes_symbols(self):
        x = S("display<tag>")
        result = Expression.solve(x - 2, [x])
        for html in (result._repr_html_(), result[0]._repr_html_()):
            self.assertNotIn("display<tag>", html)
            self.assertIn("display&lt;tag&gt;", html)
            self.assertIn("<span> = </span>", html)
            self.assertIn(result[0][x]._repr_html_(), html)

    def test_notebook_display_formats_aligned_assignments(self):
        x, y = self.x, self.y
        result = Expression.solve(x*x-x-3, [x, y], domain=Reals)
        self.assertIn("(0) ", result._repr_html_())
        self.assertIn("(1) ", result._repr_html_())
        for branch in result:
            self.assertNotIn("(0) ", branch._repr_html_())
            self.assertNotIn("(1) ", branch._repr_html_())
            for html in (branch._repr_html_(), result._repr_html_()):
                self.assertNotIn("<table", html)
                self.assertIn("display: grid", html)
                self.assertIn(branch[x]._repr_html_(), html)
                self.assertNotIn("$$", html)
                self.assertIn("<span> = </span>", html)
                self.assertNotIn("free:", html)
                self.assertNotIn("<pre>", html)
                self.assertNotIn("13^(1/2)", html)
        multiple = Expression.solve([x-2, y-3], [x, y])
        self.assertEqual(multiple._repr_html_().count("<span> = </span>"), 2)
        for variable, value in multiple[0].items():
            self.assertIn(variable._repr_html_(), multiple._repr_html_())
            self.assertIn(value._repr_html_(), multiple._repr_html_())

    def test_ipython_display(self):
        try:
            from IPython.core.formatters import DisplayFormatter
            from IPython.lib.pretty import pretty
        except ImportError:
            self.skipTest("IPython is not installed")
        result = Expression.solve(self.x - 2, [self.x], domain=Reals)
        for value in (result, result[0]):
            self.assertEqual(pretty(value), str(value))
            data, _ = DisplayFormatter().format(value)
            self.assertIn("x = 2", data["text/plain"])
            self.assertEqual(data["text/html"], value._repr_html_())
            self.assertIn(result[0][self.x]._repr_html_(), data["text/html"])
            output = io.StringIO()
            class Printer:
                text = output.write
            value._repr_pretty_(Printer(), True)
            self.assertEqual(output.getvalue(), f"{type(value).__name__}(...)")

    def test_rich_output_formats_all_embedded_expressions(self):
        x, y, z = S("x", "y", "display_parameter")
        radical = Expression.solve([x*x-x-1,y.sqrt()-z], [x,y])
        branch = radical[0]
        condition = branch.conditions()[0]
        for value in (radical, branch):
            output = io.StringIO()
            class Printer:
                text = output.write
            value._repr_pretty_(Printer(), False)
            self.assertIn("display_parameter²", output.getvalue())
            self.assertNotIn("display_parameter^2", output.getvalue())
        for html in (radical._repr_html_(), branch._repr_html_(), condition._repr_html_(),
                     condition.formula._repr_html_()):
            self.assertIn("display_parameter²", html)
            self.assertNotIn("display_parameter^2", html)
            self.assertNotIn("$$", html)
        real = Expression.solve(x*x-z, [x], domain=Reals)
        for branch in real:
            condition = next(c for c in branch.conditions() if c.kind=="domain_membership")
            value_html = condition.value._repr_html_()
            inline_value = value_html.removeprefix('<div style="white-space: pre-wrap; margin: 0;">').removesuffix('</div>')
            self.assertIn(inline_value, condition._repr_html_())
            self.assertIn(inline_value, branch._repr_html_())
        generic = Expression.solve((z*z-1)*x-1, [x])
        for html in (generic._repr_html_(), generic.coverage_guard._repr_html_()):
            self.assertIn("display_parameter²", html)
            self.assertNotIn("display_parameter^2", html)
        empty = Expression.solve((z*z-1)*(x*x+1), [x], domain=Reals)
        self.assertEqual(len(empty), 0)
        self.assertIn("display_parameter²", empty._repr_html_())

    def test_later_variables_are_preferred_as_free_coordinates(self):
        from itertools import permutations
        x, y, z = S("x", "y", "z")
        equations = [x-y-z-2, 3*x+2*y-z-4]
        for variables in permutations([x, y, z]):
            branch = Expression.solve(equations, variables)[0]
            self.assertEqual(branch.variables, list(variables))
            self.assertEqual(branch.free_variables(), [variables[-1]])
            self.assertEqual(branch.keys(), list(variables[:-1]))
            for equation in equations:
                for variable, value in branch.items():
                    equation = equation.replace(variable, value)
                self.assertEqual(equation.expand().together(), 0)
        branch = Expression.solve(equations, [x, y, z])[0]
        self.assertEqual((branch[x]-(8+3*z)/5).expand(), 0)
        self.assertEqual((branch[y]+(2+2*z)/5).expand(), 0)
        for variables in ([x, y], [y, x]):
            result = Expression.solve(x*x-y, variables)
            self.assertEqual(len(result), 2 if variables[0]==x else 1)
            for branch in result:
                self.assertEqual(branch.free_variables(), [variables[-1]])
                residual = (x*x-y).replace(variables[0], branch[variables[0]])
                self.assertEqual(residual.expand(), 0)

    def test_mixed_polynomial_and_symbolic_radical_system(self):
        x, y, z = S("x", "y", "z")
        for variables in ([x,y], [y,x]):
            result = Expression.solve([x*x-x-1, y.sqrt()-z], variables)
            self.assertEqual(result.coverage, "complete")
            self.assertEqual(result.parameters, [z])
            self.assertEqual(len(result), 2)
            self.assertEqual(len({b[x] for b in result}), 2)
            for branch in result:
                self.assertEqual(branch.variables, variables)
                self.assertEqual((branch[x]**2-branch[x]-1).expand(), 0)
                self.assertEqual(branch[y], z*z)
                self.assertEqual(branch.free_variables(), [])
                self.assertIn(f"{(z*z).sqrt()-z} = 0", str(branch))
                self.assertNotIn("solve_aux_", str(branch))
        # Squaring must not admit the wrong principal square-root branch.
        for parameter, count in [(0,2), (2,2), (-2,0)]:
            result = Expression.solve([x*x-x-1, y.sqrt()-parameter], [x,y])
            self.assertEqual(len(result), count)

    def test_independent_and_triangular_parameterized_equations(self):
        x, y, z, w = S("x", "y", "z", "w")
        independent = Expression.solve([x*x-z, y*y-1], [x,y,w])
        self.assertEqual(len(independent), 4)
        for branch in independent:
            self.assertEqual((branch[x]**2-z).expand(), 0)
            self.assertIn(branch[y], [-1,1])
            self.assertEqual(branch.free_variables(), [w])
        systems = ([x*x-z, x-y, y*y-z], [x*y-z, x+y-1],
                   [x*x+y*y-z, x-y])
        for equations in systems:
            for variables in ([x,y], [y,x]):
                result = Expression.solve(equations, variables)
                self.assertEqual(len(result), 2)
                self.assertEqual(result.coverage, "complete")
                for branch in result:
                    for equation in equations:
                        for variable, value in branch.items():
                            equation = equation.replace(variable, value)
                        self.assertEqual(equation.expand().together(), 0)
        coupled = Expression.solve([x.sqrt()-z, y-x*x], [x,y])
        self.assertEqual(dict(coupled[0]), {x:z*z, y:z**4})
        for domain in (Integers, Rationals):
            self.assertTrue(Expression.solve([x-y*y,y*y-2], [x,y], domain=domain).is_empty())
        self.assertTrue(Expression.solve([x*x-z, y*y+1], [x,y], domain=Reals).is_empty())

    def test_general_parameterized_polynomial_system_reaches_the_backend(self):
        x, y, a, b = S("x", "y", "a", "b")
        equations = [x*x+y*y-a, x*y-b]
        for variables in ([x,y], [y,x]):
            result = Expression.solve(equations, variables)
            self.assertEqual(len(result), 4)
            self.assertEqual(result.coverage, "generic")
            for precision in (None,30):
                points = []
                for branch in result:
                    values = tuple(complex(branch[v].evaluate({a:5,b:2}, precision)) for v in [x,y])
                    self.assertLess(abs(values[0]**2+values[1]**2-5), 1e-12)
                    self.assertLess(abs(values[0]*values[1]-2), 1e-12)
                    points.append(tuple(round(v.real) for v in values))
                self.assertEqual(set(points), {(-1,-2),(-2,-1),(1,2),(2,1)})
        # Explicit substitutions for a larger expression must still take precedence.
        root = result[0][x]
        f = S("evaluate_root_override")
        self.assertEqual(f(root).evaluate({f(root):7,a:5,b:2}), 7)
        flag, missing = S("flag", "missing")
        lazy = Expression.parse("if(flag,missing,7)").replace(missing, root)
        self.assertEqual(lazy.evaluate({flag:0,a:5}), 7)
        for branch in result:
            v, w = (complex(branch[k].evaluate({a:2+1j,b:1j},30)) for k in [x,y])
            self.assertLess(abs(v*v+w*w-(2+1j)), 1e-12)
            self.assertLess(abs(v*w-1j), 1e-12)
        z = S("z")
        family = Expression.solve([x*y-z,x*x+y*y-1],[x,y,z])
        self.assertEqual(len(family), 4)
        for branch in family:
            self.assertEqual(branch.free_variables(), [z])
            v, w = (complex(branch[k].evaluate({z:0.25},30)) for k in [x,y])
            self.assertLess(abs(v*v+w*w-1), 1e-12)
            self.assertLess(abs(v*w-0.25), 1e-12)

    def test_triangular_binomial_roots_include_collisions(self):
        x, y, z = S("x", "y", "z")
        cubic = Expression.solve(x**3-z, [x])
        self.assertEqual(cubic.coverage, "complete")
        self.assertEqual(len(cubic), 3)
        self.assertTrue(all(b[x].replace(z, 0)==0 for b in cubic))
        result = Expression.solve([x**3+y+1, y*y-z], [x,y])
        self.assertEqual(result.coverage, "complete")
        self.assertEqual(len(result), 6)
        for branch in result:
            self.assertNotIn("root(", str(branch[x]))
            for parameter in (0,1,2):
                a = complex(branch[x].evaluate({z:parameter}, 30))
                b = complex(branch[y].evaluate({z:parameter}, 30))
                self.assertLess(abs(a**3+b+1), 1e-12)
                self.assertLess(abs(b*b-parameter), 1e-12)

    def test_symbolic_algebraic_roots_retain_rational_and_integer_restrictions(self):
        x, y, z = S("x", "y", "z")
        for domain in (Integers, Rationals):
            for variables in ([x], [x,y]):
                result = Expression.solve(x*x-z, variables, domain=domain)
                self.assertEqual(len(result), 2)
                self.assertEqual(result.coverage, "complete")
                for branch in result:
                    self.assertEqual((branch[x]**2-z).expand(), 0)
                    self.assertTrue(any(c.kind=="domain_membership" and c.domain==domain
                                        for c in branch.conditions()))
                    self.assertEqual(branch.free_variables(), variables[1:])
            family = Expression.solve(x*x-y, [x,y], domain=domain)
            self.assertTrue(all(b.free_variables()==[y] for b in family))
        self.assertTrue(Expression.solve([z*x, 1], [x]).is_empty())
        self.assertTrue(Expression.solve([z*x, y*y+1], [x,y], domain=Reals).is_empty())

    def test_symbolic_radicals_use_general_power_polynomialization(self):
        x, z = S("x", "z")
        for degree in (2,3,4):
            for shift in (0,1,-2):
                exponent = Expression.num(1)/degree
                equation = (x+shift)**exponent-z
                result = Expression.solve(equation, [x])
                self.assertEqual(len(result), 1)
                self.assertEqual((result[0][x]-(z**degree-shift)).expand(), 0)
                self.assertEqual(result.coverage, "complete")
                self.assertTrue(result[0].is_conditional())
                self.assertNotIn("solve_aux_", str(result))
        nested = Expression.solve((x.sqrt()+1).sqrt()-z, [x])
        self.assertEqual((nested[0][x]-(z*z-1)**2).expand(), 0)
        reciprocal = Expression.solve(1/x.sqrt()-z, [x])
        self.assertEqual(reciprocal[0][x], 1/z**2)
        self.assertEqual(reciprocal.coverage, "generic")
        self.assertIn("z != 0", str(reciprocal.coverage_guard))
        power = Expression.solve(x**(Expression.num(2)/3)-z, [x])
        self.assertEqual(len(power), 2)
        self.assertEqual({b[x] for b in power}, {z**(Expression.num(3)/2), -z**(Expression.num(3)/2)})
        self.assertTrue(all(b.is_conditional() for b in power))
        self.assertTrue(Expression.solve((x.sqrt()-z)/(x-z*z), [x]).is_empty())

    def test_solve_documentation_examples(self):
        """Run examples from both help() docstrings and editor-visible stubs."""
        parser = doctest.DocTestParser()
        runner = doctest.DocTestRunner()
        targets = ("Solution", "SolutionSet")
        docs = [(name, getattr(symbolica, name).__doc__) for name in targets]
        docs.append(("Expression.solve", Expression.solve.__doc__))
        stub_path = Path(__file__).resolve().parents[1] / "symbolica.pyi"
        for node in ast.parse(stub_path.read_text()).body:
            if isinstance(node, ast.ClassDef):
                if node.name in targets:
                    docs.append((f"stub.{node.name}", ast.get_docstring(node)))
                elif node.name == "Expression":
                    method = next(n for n in node.body
                                  if isinstance(n, ast.FunctionDef) and n.name == "solve")
                    docs.append(("stub.Expression.solve", ast.get_docstring(method)))
        output = io.StringIO()
        for name, doc in docs:
            with self.subTest(doc=name):
                self.assertIn("Examples", doc)
                result = runner.run(parser.get_doctest(doc, {}, name, name, 0), out=output.write)
                self.assertEqual(result.failed, 0, output.getvalue())
                self.assertGreater(result.attempted, 0)


if __name__ == "__main__":
    unittest.main()
