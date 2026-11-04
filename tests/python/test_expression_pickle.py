import pickle
import subprocess
import sys
import tempfile
import textwrap
import unittest

from symbolica import E, Expression, S


def test_expression_pickle_roundtrip_preserves_terms():
    expr = E("x+y+z")

    restored = pickle.loads(pickle.dumps(expr))

    assert str(restored) == str(expr)
    assert [str(term) for term in restored.terms()] == ["x", "y", "z"]


class ExpressionPickleTests(unittest.TestCase):
    def run_python(self, source, *args):
        result = subprocess.run(
            [sys.executable, "-c", textwrap.dedent(source), *args],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_roundtrip_preserves_terms(self):
        test_expression_pickle_roundtrip_preserves_terms()

    def test_pickle_uses_regular_save_and_load(self):
        expression = E('pickle_bytes::f(pickle_bytes::x)+1/3')
        data = expression.save()
        self.assertIsInstance(data, bytes)
        self.assertEqual(Expression.load(data), expression)
        self.assertEqual(expression.__getstate__(), data)
        reconstruct, arguments = expression.__reduce__()
        self.assertIs(reconstruct.__self__, Expression)
        self.assertEqual(reconstruct.__name__, 'load')
        self.assertEqual(arguments, (data,))

    def test_file_and_byte_saves_use_the_same_format(self):
        from pathlib import Path

        expression = E('pickle_file::f(pickle_file::x)+1/3')
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'expression.dat'
            for compression_level in (0, 9, 11):
                with self.subTest(compression_level=compression_level):
                    self.assertIsNone(expression.save(str(filename), compression_level))
                    data = expression.save(compression_level=compression_level)
                    self.assertEqual(filename.read_bytes(), data)
                    self.assertEqual(Expression.load(data), expression)
                    self.assertEqual(Expression.load(str(filename)), expression)

    def test_unpickle_imports_the_owning_module_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = directory + "/autoload.pkl"
            with open(filename, 'wb') as stream:
                pickle.dump(E('pickle_autoload::x^2+1'), stream)
            self.run_python("""
                import pickle, sys

                assert not any(name.startswith('symbolica') for name in sys.modules)
                with open(sys.argv[1], 'rb') as stream:
                    restored = pickle.load(stream)
                assert type(restored).__name__ == 'Expression'
                assert len(list(restored.terms())) == 2
                module = sys.modules[type(restored).__module__]
                assert module.Expression is type(restored)
                assert pickle.loads(pickle.dumps(restored)) == restored
            """, filename)

    def test_roundtrip_preserves_state_in_another_process(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = directory + "/expression.pkl"
            self.run_python("""
                import pickle, sys
                from symbolica import S

                x, y = S('pickle_state::x', 'pickle_state::y')
                dependency = S('pickle_state::dependency', is_positive=True)
                f = S('pickle_state::f', is_symmetric=True, tags=['pickle_tag'],
                      data={'dependency': dependency + 1})
                S('pickle_state::unused', is_symmetric=True)
                coefficient = ((x + y) / (x - y)).set_coefficient_ring([x, y])
                with open(sys.argv[1], 'wb') as stream:
                    pickle.dump([f(x, y)**2 + x/3 + y**-16, coefficient], stream)
            """, filename)
            self.run_python("""
                import pickle, sys
                from symbolica import E, S

                # Allocate different symbol IDs and a different coefficient variable list.
                a, b = S('pickle_state::a', 'pickle_state::b')
                (a + b).set_coefficient_ring([b, a])
                y = S('pickle_state::y')
                S('pickle_state::unused', is_symmetric=False)
                with open(sys.argv[1], 'rb') as stream:
                    restored, coefficient = pickle.load(stream)

                x = S('pickle_state::x')
                f = S('pickle_state::f')
                dependency = S('pickle_state::dependency', is_positive=True)
                expected = f(x, y)**2 + x/3 + y**-16
                assert restored == expected
                assert hash(restored) == hash(expected)
                assert {expected: 'found'}[restored] == 'found'
                assert f(y, x) == f(x, y)
                assert f.get_tags() == ['python::pickle_tag']
                assert f.get_symbol_data('dependency') == dependency + 1
                assert coefficient == ((x + y) / (x - y)).set_coefficient_ring([x, y])
            """, filename)

    def test_state_conflicts_raise_value_error(self):
        for conflicting_options in (
            "{'is_symmetric': False, 'tags': ['saved'], 'data': 7}",
            "{'is_symmetric': True, 'tags': ['different'], 'data': 7}",
            "{'is_symmetric': True, 'tags': ['saved'], 'data': 8}",
        ):
            with self.subTest(options=conflicting_options), tempfile.TemporaryDirectory() as directory:
                filename = directory + "/conflict.pkl"
                self.run_python("""
                    import pickle, sys
                    from symbolica import S
                    f = S('pickle_conflict::f', is_symmetric=True, tags=['saved'], data=7)
                    with open(sys.argv[1], 'wb') as stream:
                        pickle.dump(f(1, 2), stream)
                """, filename)
                self.run_python("""
                    import ast, pickle, sys
                    from symbolica import E, S
                    S('pickle_conflict::f', **ast.literal_eval(sys.argv[2]))
                    try:
                        with open(sys.argv[1], 'rb') as stream:
                            pickle.load(stream)
                    except ValueError as error:
                        assert 'Symbol conflict' in str(error), str(error)
                    else:
                        raise AssertionError('Unpickling accepted conflicting symbol state')
                    # The interpreter remains usable after the failed import.
                    assert E('1+2') == 3
                """, filename, conflicting_options)

    def test_same_process_roundtrip_accepts_matching_state(self):
        f = S('pickle_matching::f', is_symmetric=True, data={'value': 7})
        expression = f(1, 2)
        self.assertEqual(pickle.loads(pickle.dumps(expression)), expression)


if __name__ == "__main__":
    unittest.main()
