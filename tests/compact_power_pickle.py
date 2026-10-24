"""Migrate format-5 pickles to canonical inline numbers in atom format 6."""

import pickle
import re
import unittest
import xml.etree.ElementTree as ET

from symbolica import ComplexFloat, E, Expression, Float, PrintMode


def raw_atom(expression):
    state = bytes(expression.__getstate__())
    assert state[0] == 0
    assert int.from_bytes(state[2:10], 'little') == len(state) - 10
    return state[10:]


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


def read_pair(data, offset):
    tag = data[offset]
    offset += 1
    width = 1 << ((tag & 15) - 1)
    numerator = int.from_bytes(data[offset:offset + width], 'little')
    offset += width
    denominator = 1
    if tag & 0x70:
        width = 1 << (((tag >> 4) & 7) - 1)
        denominator = int.from_bytes(data[offset:offset + width], 'little')
        offset += width
    return numerator, denominator, offset


def legacy_integer(value):
    # Format 5 used a numeric tag followed by a packed coefficient.
    packed = bytearray(packed_pair(abs(value), 1))
    if value < 0:
        packed[0] |= 0x80
    return b'\x01' + packed


def legacy_power(variable, exponent):
    # Format 5 stored the complete number and had no compact-power header flag.
    return b'\x06' + raw_atom(E(variable)) + legacy_integer(exponent)


def legacy_function(name, *arguments):
    empty = raw_atom(E(name + '()'))
    _, _, metadata = read_pair(empty, 1)
    symbol, _, _ = read_pair(empty, metadata)
    payload = packed_pair(symbol, len(arguments)) + b''.join(arguments)
    return empty[:1] + packed_pair(len(payload), 1) + payload


def legacy_list(tag, *arguments):
    payload = b''.join(arguments)
    return bytes([tag]) + packed_pair(len(arguments), len(payload)) + payload


def versioned_state(atom_format, raw):
    return bytes([0, atom_format]) + len(raw).to_bytes(8, 'little') + raw


def reconstruct(state):
    constructor, _ = E('0').__reduce__()
    return constructor(state)


class LegacyPickle:
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
            'pickle_f(x^2,pickle_g(y^-16),z^16,2)',
        ):
            with self.subTest(source=source):
                expected = E(source)
                self.assertEqual(bytes(expected.__getstate__())[:2], bytes([0, 6]))
                self.assert_same_expression(pickle.loads(pickle.dumps(expected)), expected)

    def test_legacy_nested_powers_are_canonicalized(self):
        raw = legacy_function(
            'pickle_f', legacy_power('x', 2),
            legacy_function('pickle_g', legacy_power('y', -16)),
            legacy_power('z', 16),
        )
        restored = pickle.loads(pickle.dumps(LegacyPickle(raw)))
        self.assert_same_expression(restored, E('pickle_f(x^2,pickle_g(y^-16),z^16)'))
        self.assertEqual(restored + E('pickle_f(x^2,pickle_g(y^-16),z^16)'),
                         E('2*pickle_f(x^2,pickle_g(y^-16),z^16)'))
        self.assertEqual(restored[0][1], E('2'))
        self.assertEqual(restored[1][0][1], E('-16'))
        self.assertEqual(restored[2][1], E('16'))

    def test_versioned_legacy_nested_powers_are_canonicalized(self):
        for exponent in (-17, -16, -15, -2, -1, 2, 15, 16):
            with self.subTest(exponent=exponent):
                raw = legacy_function(
                    'pickle_f', legacy_power('x', exponent),
                    legacy_function('pickle_g', legacy_power('y', -16)),
                    legacy_integer(2),
                )
                restored = pickle.loads(pickle.dumps(
                    LegacyPickle(versioned_state(5, raw))))
                self.assert_same_expression(
                    restored, E(f'pickle_f(x^({exponent}),pickle_g(y^-16),2)'))
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

    def test_legacy_standalone_numbers_are_canonicalized(self):
        for value in range(-17, 17):
            for atom_format in (None, 5):
                with self.subTest(atom_format=atom_format, value=value):
                    state = legacy_integer(value)
                    if atom_format is not None:
                        state = versioned_state(atom_format, state)
                    self.assert_same_expression(reconstruct(state), E(str(value)))

    def test_legacy_term_order_and_coefficients_are_canonicalized(self):
        for atom_format in (None, 5):
            with self.subTest(atom_format=atom_format):
                raw = legacy_list(
                    5, legacy_power('x', 16),
                    legacy_power('x', 2),
                    legacy_list(0x44, legacy_integer(2), legacy_power('x', 3)),
                    legacy_power('x', -15),
                )
                state = raw if atom_format is None else versioned_state(atom_format, raw)
                restored = reconstruct(state)
                self.assert_same_expression(restored, E('x^2+x^16+x^-15+2*x^3'))

    def test_legacy_compact_floats_preserve_value_and_component_precision(self):
        # Format 5 introduced this binary float layout, unchanged by format 6.
        # 0xa0 * 2**(1-8) = 1.25; the imaginary payload encodes -2.5 similarly.
        real = (128).to_bytes(4, 'little') + b'\x01\x02\xa0'
        for imaginary, precision, payload in (
            ('0', 128, b''),
            ('0', 192, (192).to_bytes(4, 'little')),
            ('-2.5', 192, (192).to_bytes(4, 'little') + b'\x02\x04\xa0'),
        ):
            expected = Expression.num(ComplexFloat(
                Float('1.25', precision=128), Float(imaginary, precision=precision)))
            raw = b'\x01\x0b' + packed_pair(len(real), len(payload)) + real + payload
            self.assertEqual(raw_atom(expected), raw)
            for atom_format in (None, 5):
                with self.subTest(imaginary=imaginary, precision=precision,
                                  atom_format=atom_format):
                    state = raw if atom_format is None else versioned_state(atom_format, raw)
                    restored = pickle.loads(pickle.dumps(LegacyPickle(state)))
                    self.assert_same_expression(restored, expected)
                    self.assert_same_expression(pickle.loads(pickle.dumps(restored)), expected)

    def test_legacy_repeated_numbers_keep_child_and_viewer_paths(self):
        values = (-15, -15, 0, 15, 15)
        power = b'\x06' + legacy_function('pickle_g', legacy_integer(2)) + legacy_integer(2)
        raw = legacy_function('pickle_f', *(legacy_integer(v) for v in values), power)
        expected = E('pickle_f(-15,-15,0,15,15,pickle_g(2)^2)')
        restored = reconstruct(versioned_state(5, raw))
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

        repeated = legacy_function('pickle_g', *([legacy_integer(2)] * 30))
        restored = reconstruct(versioned_state(5, legacy_function('pickle_f', repeated, repeated)))
        expected = E('pickle_g(' + ','.join(['2'] * 30) + ')')
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

    def test_empty_and_unsupported_pickle_formats_raise_value_error(self):
        with self.assertRaisesRegex(ValueError, 'empty'):
            reconstruct(b'')
        for version in (0, 1, 2, 3, 4, 7, 255):
            with self.subTest(version=version), self.assertRaisesRegex(ValueError, 'format'):
                reconstruct(bytes([0, version]))
        with self.assertRaises(ValueError):
            reconstruct(b'\x00')


if __name__ == '__main__':
    unittest.main()
