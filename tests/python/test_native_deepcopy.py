"""Native deep copies preserve values and isolate mutable object state."""

import copy
from decimal import Decimal
import unittest

from symbolica import Graph, Matrix, NumericalIntegrator, RandomNumberGenerator, S


class NativeDeepcopyTests(unittest.TestCase):
    def test_all_polynomial_variants_and_container_aliases(self):
        x, a = S('deepcopy_poly::x', 'deepcopy_poly::a')
        expression = x**2 + a*x + 1
        rational = (1/(x+1)).to_rational_polynomial()
        polynomials = [
            expression.to_polynomial(),
            expression.to_polynomial(modulus=17),
            expression.to_polynomial(modulus=2),
            expression.to_polynomial(modulus=17, power=(2, a)),
            expression.to_polynomial(modulus=2, power=(2, a)),
            expression.to_polynomial(minimal_poly=(a**2-2).to_polynomial()),
            rational,
            rational.to_finite_field(17),
        ]
        self.assertEqual(len({type(p) for p in polynomials}), 8)
        for original in polynomials:
            with self.subTest(type=type(original).__name__):
                memo = {}
                result = copy.deepcopy({'polynomial': original, 'aliases': [original]}, memo)
                cloned = result['polynomial']
                self.assertIs(type(cloned), type(original))
                self.assertIsNot(cloned, original)
                self.assertIs(result['aliases'][0], cloned)
                self.assertIs(memo[id(original)], cloned)
                self.assertEqual(cloned, original)
                self.assertEqual(cloned + cloned, original + original)

    def test_matrix_storage_is_independent(self):
        x = S('deepcopy_matrix::x')
        original = Matrix.from_nested([[x, 1], [2, x+1]])
        cloned = copy.deepcopy(original)
        self.assertIsNot(cloned, original)
        self.assertEqual(cloned, original)
        cloned.swap_rows(0, 1)
        self.assertEqual(original, Matrix.from_nested([[x, 1], [2, x+1]]))
        self.assertEqual(cloned, Matrix.from_nested([[2, x+1], [x, 1]]))

    def test_graph_storage_is_independent(self):
        x = S('deepcopy_graph::x')
        original = Graph()
        original.add_node(x)
        original.add_node(x+1)
        original.add_edge(0, 1, data=x**2)
        cloned = copy.deepcopy(original)
        self.assertIsNot(cloned, original)
        self.assertEqual(cloned, original)
        cloned.add_node(x+2)
        cloned.add_edge(1, 2, data=x**3)
        self.assertEqual(len(original), 2)
        self.assertEqual(len(cloned), 3)
        self.assertNotEqual(cloned, original)

    def test_rng_preserves_current_position_with_independent_state(self):
        original = RandomNumberGenerator(17, 2)
        for _ in range(5):
            original.next()
        cloned = copy.deepcopy(original)
        self.assertIsNot(cloned, original)
        state = original.save()
        sequence = [cloned.next() for _ in range(10)]
        self.assertEqual(original.save(), state)
        self.assertEqual([original.next() for _ in range(10)], sequence)

    def test_evaluator_copies_initialized_backends(self):
        x = S('deepcopy_evaluator::x')
        for jit in (False, True):
            with self.subTest(jit=jit):
                original = (x**2 + 1).evaluator([x], jit_compile=jit)
                self.assertEqual([list(row) for row in original.evaluate([[2]])], [[5]])
                self.assertEqual(
                    [list(row) for row in original.evaluate_complex([[2+1j]])], [[4+4j]])
                self.assertEqual(original.evaluate_with_prec([Decimal(2)], 40), [Decimal(5)])
                cloned = copy.deepcopy(original)
                self.assertIsNot(cloned, original)
                self.assertEqual([list(row) for row in cloned.evaluate([[3]])], [[10]])
                self.assertEqual(
                    [list(row) for row in cloned.evaluate_complex([[3+1j]])], [[9+6j]])
                self.assertEqual(cloned.evaluate_with_prec([Decimal(3)], 64), [Decimal(10)])
                self.assertEqual([list(row) for row in original.evaluate([[2]])], [[5]])
                self.assertEqual(original.evaluate_with_prec([Decimal(2)], 40), [Decimal(5)])

    def test_integrator_preserves_pending_samples_and_independent_state(self):
        def continuous():
            return NumericalIntegrator.continuous(1, n_bins=4, min_samples_for_update=2)

        grids = [
            continuous(),
            NumericalIntegrator.discrete([continuous(), continuous()]),
            NumericalIntegrator.uniform([2], continuous()),
        ]
        for index, original in enumerate(grids):
            with self.subTest(grid=index):
                samples = original.sample(40, RandomNumberGenerator(17, index))
                original.add_training_samples(samples, [1 + sum(s.c) + sum(s.d) for s in samples])
                state = original.export_grid()
                cloned = copy.deepcopy(original)
                self.assertIsNot(cloned, original)
                self.assertEqual(cloned.export_grid(), state)
                self.assertNotEqual(copy.copy(original).export_grid(), state)
                cloned.update(1.5, 1.5)
                self.assertEqual(original.export_grid(), state)
                self.assertNotEqual(cloned.export_grid(), state)
                original.update(1.5, 1.5)
                self.assertEqual(cloned.export_grid(), original.export_grid())


if __name__ == '__main__':
    unittest.main()
