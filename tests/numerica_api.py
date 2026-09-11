"""Numerica integration API regressions; run with the built extension on PYTHONPATH."""
import unittest

from symbolica import NumericalIntegrator


class GridConstruction(unittest.TestCase):
    def test_empty_grids_raise_value_error(self):
        with self.assertRaises(ValueError):
            NumericalIntegrator.discrete([])
        for dimensions, bins in [(0, 10), (1, 0), (0, 0)]:
            with self.subTest(dimensions=dimensions, bins=bins):
                with self.assertRaises(ValueError):
                    NumericalIntegrator.continuous(dimensions, n_bins=bins)

    def test_bin_evolution_cannot_be_empty_or_contain_zero(self):
        for evolution in [[], [0], [10, 0]]:
            with self.subTest(evolution=evolution):
                with self.assertRaises(ValueError):
                    NumericalIntegrator.continuous(1, bin_number_evolution=evolution)

    def test_uniform_layers_cannot_have_zero_bins(self):
        with self.assertRaises(ValueError):
            NumericalIntegrator.uniform([2, 0], NumericalIntegrator.continuous(1))

    def test_valid_grids_still_sample(self):
        continuous = NumericalIntegrator.continuous(1, n_bins=1)
        discrete = NumericalIntegrator.discrete([None, continuous])
        uniform = NumericalIntegrator.uniform([2], continuous)
        rng = NumericalIntegrator.rng(0, 0)
        for grid in [continuous, discrete, uniform]:
            self.assertEqual(len(grid.sample(3, rng)), 3)


if __name__ == "__main__":
    unittest.main()
