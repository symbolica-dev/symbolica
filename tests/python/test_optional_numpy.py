"""The same wheel must support evaluation with and without NumPy installed."""

import subprocess
import sys
import textwrap

import pytest


def run_python(source):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_evaluation_without_numpy():
    # Use a fresh interpreter so rust-numpy cannot reuse an initialized C API.
    run_python("""
        import importlib.abc
        import shutil
        import sys
        import tempfile
        from pathlib import Path

        class NoNumpy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "numpy" or fullname.startswith("numpy."):
                    raise ModuleNotFoundError("No module named 'numpy'", name="numpy")

        sys.meta_path.insert(0, NoNumpy())
        from symbolica import E, Expression, N, P, S

        x, y = S("optional_numpy_x", "optional_numpy_y")
        for jit in (False, True):
            ev = Expression.evaluator_multiple([x + y, x * y], [x, y], jit_compile=jit)
            for inputs in ([1, 2, 3, 4], [[1, 2], [3, 4]], ((1, 2), (3, 4))):
                assert ev.evaluate(inputs) == [[3.0, 2.0], [7.0, 12.0]]
                assert ev.evaluate_complex(inputs) == [[3+0j, 2+0j], [7+0j, 12+0j]]
            assert ev.evaluate_complex([[1+2j, 3-1j]]) == [[4+1j, 5+5j]]
            assert ev.evaluate([]) == []
            assert ev.evaluate_complex([]) == []

            for evaluate in (ev.evaluate, ev.evaluate_complex):
                for bad in ([[1]], [[1, 2], [3]], [1, 2, 3]):
                    try:
                        evaluate(bad)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError(f"Accepted invalid input: {bad}")
                assert evaluate([[1, 2]]) == [[3, 2]]

            constant = N(3).evaluator([], jit_compile=jit)
            assert constant.evaluate([]) == [[3.0]]
            assert constant.evaluate([[], []]) == [[3.0], [3.0]]
            assert constant.evaluate_complex([[], []]) == [[3+0j], [3+0j]]
            complex_constant = N(1j).evaluator([], jit_compile=jit)
            assert complex_constant.evaluate_complex([]) == [[1j]]
            try:
                complex_constant.evaluate([])
            except ValueError:
                pass
            else:
                raise AssertionError("Accepted complex coefficient for real evaluation")

        poly = P("x*y+2*x+x^2")
        assert poly.evaluate([2, 3]) == 14.0
        assert poly.evaluate_complex((2+1j, 3+2j)) == 11+13j
        try:
            poly.evaluate([2])
        except ValueError:
            pass
        else:
            raise AssertionError("Accepted wrong polynomial input length")

        import symbolica
        if hasattr(symbolica, "CompiledRealEvaluator") and shutil.which("c++"):
            with tempfile.TemporaryDirectory() as directory:
                for backend in ("real", "complex", "real_4x", "complex_4x"):
                    base = Path(directory) / backend
                    compiled = ev.compile("optional_numpy", str(base.with_suffix(".cpp")),
                                          str(base.with_suffix(".so")), backend,
                                          inline_asm="none", optimization_level=0)
                    assert compiled.evaluate([[1, 2], [3, 4]]) == [[3, 2], [7, 12]]
                    assert compiled.evaluate([]) == []
                    if "complex" in backend:
                        assert compiled.evaluate([[1+2j, 3-1j]]) == [[4+1j, 5+5j]]
        assert "numpy" not in sys.modules
    """)


def test_numpy_import_errors_are_not_hidden():
    run_python("""
        import importlib.abc
        import sys

        class BrokenNumpy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "numpy":
                    raise ModuleNotFoundError("Broken NumPy installation", name="numpy_dependency")

        sys.meta_path.insert(0, BrokenNumpy())
        from symbolica import N
        try:
            N(3).evaluator([]).evaluate([])
        except ModuleNotFoundError as error:
            assert error.name == "numpy_dependency"
        else:
            raise AssertionError("Suppressed a broken NumPy installation")
    """)


@pytest.mark.parametrize("jit", [False, True])
def test_numpy_results_are_preserved(jit):
    np = pytest.importorskip("numpy")
    from symbolica import Expression, P, S

    x, y = S("numpy_preserved_x", "numpy_preserved_y")
    ev = Expression.evaluator_multiple([x + y, x * y], [x, y], jit_compile=jit)
    inputs = np.array([[1, 2], [3, 4]], dtype=float)
    for values in (inputs, inputs.tolist(), inputs.astype(int)):
        result = ev.evaluate(values)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [[3, 2], [7, 12]])
        result = ev.evaluate_complex(values)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.complex128
        np.testing.assert_array_equal(result, [[3, 2], [7, 12]])
    assert ev.evaluate(np.empty((0, 2))).shape == (0, 2)
    assert P("x*y+2*x+x^2").evaluate(np.array([2, 3])) == 14
