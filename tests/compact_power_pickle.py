"""State-aware pickles preserve canonical compact numbers, powers, and viewer paths."""

import pickle
import re
import unittest
import xml.etree.ElementTree as ET

from symbolica import ComplexFloat, E, Expression, Float, PrintMode


def reconstruct(state):
    constructor, _ = E('0').__reduce__()
    return constructor(state)


class CompactPowerPickleTests(unittest.TestCase):
    def assert_same_expression(self, restored, expected):
        self.assertEqual(restored, expected)
        self.assertEqual(hash(restored), hash(expected))
        self.assertEqual({expected: 'found'}[restored], 'found')
        self.assertEqual(bytes(restored.__getstate__()), bytes(expected.__getstate__()))
        restored_html = restored._repr_html_()
        expected_html = expected._repr_html_()
        restored_math = re.search(r'<math\b.*?</math>', restored_html, re.S)
        expected_math = re.search(r'<math\b.*?</math>', expected_html, re.S)
        if restored_math is not None and expected_math is not None:
            self.assertEqual(ET.tostring(ET.fromstring(restored_math.group())),
                             ET.tostring(ET.fromstring(expected_math.group())))
        else:
            self.assertEqual(restored_html, expected_html)

    def test_current_pickle_roundtrip(self):
        for source in (
            '-16', '-15', '0', '1', '15', '16', '1/2', '2+3*𝑖',
            'x^2', 'x^-16', 'f(x)^2',
            'pickle_f(x^2,pickle_g(y^-16),z^16,2)',
        ):
            with self.subTest(source=source):
                expected = E(source)
                self.assert_same_expression(pickle.loads(pickle.dumps(expected)), expected)

    def test_nested_powers_and_mathml(self):
        for exponent in (-17, -16, -15, -2, -1, 2, 15, 16):
            with self.subTest(exponent=exponent):
                expected = E(f'pickle_f(x^({exponent}),pickle_g(y^-16),2)')
                restored = pickle.loads(pickle.dumps(expected))
                self.assert_same_expression(restored, expected)
                power = ET.fromstring(
                    '<math>' + restored[0].format(mode=PrintMode.MathML) + '</math>')[0]
                if exponent < 0:
                    self.assertEqual(power.tag, 'mfrac')
                    self.assertEqual(''.join(power[0].itertext()), '1')
                    power = power[1]
                if exponent == -1:
                    self.assertEqual(power.tag, 'mi')
                    self.assertEqual(power.text, 'x')
                else:
                    self.assertEqual(power.tag, 'msup')
                    self.assertEqual(''.join(power[0].itertext()), 'x')
                    self.assertEqual(''.join(power[1].itertext()), str(abs(exponent)))

    def test_compact_floats_preserve_value_and_component_precision(self):
        for imaginary, precision in (('0', 128), ('0', 192), ('-2.5', 192)):
            with self.subTest(imaginary=imaginary, precision=precision):
                expected = Expression.num(ComplexFloat(
                    Float('1.25', precision=128), Float(imaginary, precision=precision)))
                restored = pickle.loads(pickle.dumps(expected))
                self.assert_same_expression(restored, expected)

    def test_repeated_numbers_keep_child_and_viewer_paths(self):
        values = (-15, -15, 0, 15, 15)
        expected = E('pickle_f(-15,-15,0,15,15,pickle_g(2)^2)')
        restored = pickle.loads(pickle.dumps(expected))
        self.assert_same_expression(restored, expected)
        self.assertEqual(restored[:], expected[:])
        self.assertEqual(restored[::-1], expected[::-1])
        self.assertEqual(restored[5][0][0], E('2'))
        self.assertEqual(restored[5][1], E('2'))
        viewers = (restored.mathml_viewer(page_size=2), expected.mathml_viewer(page_size=2))
        try:
            for index, value in enumerate(values):
                with self.subTest(index=index):
                    self.assertEqual(restored[index], E(str(value)))
            for start in (0, 2, 4):
                pages = [viewer.page() for viewer in viewers]
                self.assertEqual((pages[0]['start'], pages[0]['end']), (start, start + 2))
                self.assertEqual(pages[0]['html'], pages[1]['html'])
                for viewer in viewers:
                    viewer.handle_message({'action': 'next'})
        finally:
            for viewer in viewers:
                viewer.close()

        expected = E('pickle_g(' + ','.join(['2'] * 30) + ')')
        restored = pickle.loads(pickle.dumps(E(f'pickle_f({expected},{expected})')))
        viewer = restored.mathml_viewer(page_size=2)
        try:
            for index in (0, 1):
                self.assert_same_expression(restored[index], expected)
                _, page = viewer.handle_message({
                    'action': 'open', 'value': [index], 'layout': 4, 'start': 0})
                self.assertNotIn('error', page)
                self.assertEqual((page['start'], page['end'], page['total']), (0, 2, 30))
                root = ET.fromstring(page['html'])
                self.assertEqual([node.text for node in root.iter()
                                  if node.tag.rsplit('}', 1)[-1] == 'mn'], ['2', '2'])
                _, following = viewer.handle_message({'action': 'next'})
                self.assertEqual((following['start'], following['end']), (2, 4))
                viewer.handle_message({'action': 'back'})
        finally:
            viewer.close()

    def test_empty_and_old_stateless_pickles_raise_value_error(self):
        legacy_number = b'\x01\x01\x02'
        for state in (
            b'', b'\x00', legacy_number,
            b'\x00\x05' + len(legacy_number).to_bytes(8, 'little') + legacy_number,
            b'\x00\x06' + (1).to_bytes(8, 'little') + b'\x19',
        ):
            with self.subTest(state=state), self.assertRaises(ValueError):
                reconstruct(state)
        state = bytes(E('x+1').__getstate__())
        with self.assertRaises(ValueError):
            reconstruct(state[:-1])


if __name__ == '__main__':
    unittest.main()
