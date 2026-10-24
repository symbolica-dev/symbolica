"""Round-trip compact atoms through tracked format-7/state-12 expression pickles."""

import pickle
import re
import unittest
import xml.etree.ElementTree as ET

from symbolica import ComplexFloat, E, Expression, Float, PrintMode


PICKLE_HEADER = b'SYMPKL\x01'
STATE_VERSION_OFFSET = len(PICKLE_HEADER) + 4


def packed_pair(numerator, denominator):
    def integer(value):
        for tag, width in enumerate((1, 2, 4, 8), 1):
            if value < 1 << (8 * width):
                return tag, value.to_bytes(width, 'little')
        raise ValueError(value)

    numerator_tag, numerator_bytes = integer(numerator)
    if denominator == 1:
        return bytes([numerator_tag]) + numerator_bytes
    denominator_tag, denominator_bytes = integer(denominator)
    return bytes([numerator_tag | denominator_tag << 4]) + numerator_bytes + denominator_bytes


def legacy_integer(value):
    # Format 5 used a numeric tag followed by a packed coefficient.
    packed = bytearray(packed_pair(abs(value), 1))
    if value < 0:
        packed[0] |= 0x80
    return b'\x01' + packed


def legacy_versioned_state(atom_format, raw):
    return bytes([0, atom_format]) + len(raw).to_bytes(8, 'little') + raw


def reconstruct(state):
    constructor, _ = E('0').__reduce__()
    return constructor(state)


class SerializedPickle:
    def __init__(self, state):
        self.state = state

    def __reduce__(self):
        constructor, _ = E('0').__reduce__()
        return constructor, (self.state,)


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
            'x^18446744073709551616', 'x^-18446744073709551617',
            'f(x)^18446744073709551616',
            'pickle_f(x^2,pickle_g(y^-16),z^16,2)',
        ):
            with self.subTest(source=source):
                expected = E(source)
                state = bytes(expected.__getstate__())
                self.assertTrue(state.startswith(PICKLE_HEADER))
                self.assertEqual(int.from_bytes(
                    state[STATE_VERSION_OFFSET:STATE_VERSION_OFFSET + 2], 'little'), 12)
                self.assert_same_expression(pickle.loads(pickle.dumps(expected)), expected)

        # A single inline zero is the final atom: its format, length and one-byte payload.
        state = bytes(E('0').__getstate__())
        self.assertEqual(state[-10], 7)
        self.assertEqual(int.from_bytes(state[-9:-1], 'little'), 1)

    def test_nested_powers_keep_child_paths(self):
        expected = E('pickle_f(x^2,pickle_g(y^-16),z^16)')
        restored = pickle.loads(pickle.dumps(expected))
        self.assert_same_expression(restored, expected)
        self.assertEqual(restored + expected, E('2*pickle_f(x^2,pickle_g(y^-16),z^16)'))
        self.assertEqual(restored[0][1], E('2'))
        self.assertEqual(restored[1][0][1], E('-16'))
        self.assertEqual(restored[2][1], E('16'))

    def test_nested_powers_keep_mathml(self):
        for exponent in (-17, -16, -15, -2, -1, 2, 15, 16, 18446744073709551616):
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

    def test_inline_and_noninline_numbers_roundtrip(self):
        for value in (*range(-17, 18), -(1 << 80), 1 << 80):
            with self.subTest(value=value):
                expected = E(str(value))
                self.assert_same_expression(pickle.loads(pickle.dumps(expected)), expected)

    def test_term_order_and_coefficients_roundtrip(self):
        expected = E('x^2+x^16+x^-15+2*x^3')
        restored = pickle.loads(pickle.dumps(expected))
        self.assert_same_expression(restored, expected)
        self.assertEqual(restored + expected, 2 * expected)

    def test_compact_floats_preserve_value_and_component_precision(self):
        for imaginary, precision in (('0', 128), ('0', 192), ('-2.5', 192)):
            with self.subTest(imaginary=imaginary, precision=precision):
                expected = Expression.num(ComplexFloat(
                    Float('1.25', precision=128), Float(imaginary, precision=precision)))
                restored = pickle.loads(pickle.dumps(expected))
                self.assert_same_expression(restored, expected)
                self.assert_same_expression(pickle.loads(pickle.dumps(restored)), expected)

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
        restored = pickle.loads(pickle.dumps(E('pickle_f(' + str(expected) + ',' + str(expected) + ')')))
        viewer = restored.mathml_viewer(page_size=2)
        try:
            for index in (0, 1):
                self.assert_same_expression(restored[index], expected)
                _, page = viewer.handle_message({'action': 'open', 'value': [index], 'layout': 4})
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

    def test_legacy_raw_and_stateless_pickles_are_rejected(self):
        # Plain and historical tracked formats reused 6/11 with different layouts.
        # No raw or old stateless pickle may be guessed into the current format.
        real = (128).to_bytes(4, 'little') + b'\x01\x02\xa0'
        imaginary = (192).to_bytes(4, 'little')
        legacy_float = b'\x01\x0b' + packed_pair(len(real), len(imaginary)) + real + imaginary
        for raw in (legacy_integer(-15), legacy_integer(0), legacy_float):
            for atom_format in (None, 5, 6):
                state = raw if atom_format is None else legacy_versioned_state(atom_format, raw)
                with self.subTest(raw=raw, atom_format=atom_format):
                    with self.assertRaisesRegex(ValueError, 'Unsupported expression pickle'):
                        pickle.loads(pickle.dumps(SerializedPickle(state)))

    def test_unsupported_state_versions_are_rejected_before_payload(self):
        state = bytearray(E('x^2').__getstate__())
        for version in (0, 6, 10, 11, 13, 65535):
            with self.subTest(version=version):
                state[STATE_VERSION_OFFSET:STATE_VERSION_OFFSET + 2] = version.to_bytes(2, 'little')
                # A valid header followed by no payload must report the version first.
                with self.assertRaisesRegex(ValueError, f'Unsupported export format version {version}'):
                    reconstruct(bytes(state[:STATE_VERSION_OFFSET + 2]))

    def test_empty_truncated_and_malformed_pickles_raise_value_error(self):
        valid = bytes(E('pickle_f(x^2,3)').__getstate__())
        malformed = (
            b'', b'\x00', b'\x00\x05', b'\x00\x06', b'SYMPKL\x02', PICKLE_HEADER,
            PICKLE_HEADER + b'\x00' + valid[len(PICKLE_HEADER) + 1:],
            valid[:-1], valid + b'trailing',
        )
        for state in malformed:
            with self.subTest(state=state), self.assertRaises(ValueError):
                reconstruct(state)


if __name__ == '__main__':
    unittest.main()
