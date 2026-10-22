"""Matrix frames preserve coordinates while native cell pages stay independent."""
import json
import sys
import unittest
from xml.etree import ElementTree as ET
from symbolica import E, S, Matrix, RationalPolynomial
from test_mathml_algebraic import root, fragment
from test_mathml_semantics import mathml_expression


def cells(html):
    return [n for n in ET.fromstring(html).iter() if 'data-symbolica-entry' in n.attrib
            and any(c.tag.rsplit('}', 1)[-1] == 'math' for c in n.iter())]


class MatrixMathMLTests(unittest.TestCase):
    def test_static_cells_preserve_signs_fractions_and_coordinates(self):
        entries = [[E('-2*x'), E('(x+1)/(y+2)')], [E('0'), E('x^2-y')]]
        matrix = Matrix.from_nested(entries)
        html = matrix._repr_html_()
        self.assertIn('<table', html)
        self.assertNotIn('data-symbolica-marker', html)
        self.assertNotIn('overflow', html)
        rendered = cells(html)
        self.assertEqual(len(rendered), 4)
        for index, cell in enumerate(rendered):
            self.assertEqual(int(cell.get('data-symbolica-entry')), index)
            self.assertEqual(mathml_expression(fragment(root(ET.tostring(cell, encoding='unicode')))), entries[index // 2][index % 2])

    def test_each_long_cell_has_an_independent_pager_and_expansion(self):
        x, y = S('x', 'y')
        large = E('(1+x)^100').expand()
        matrix = Matrix.from_nested([[large, E('(1+y)^90').expand()], [1, 2]])
        viewer = matrix.mathml_viewer(page_size=7)
        original = str(matrix)
        try:
            _, page = viewer.handle_message({'action': 'attach', 'view': 'matrix'})
            self.assertEqual(page['total'], 2)
            self.assertEqual([m[0] for m in page['markers']], [[0], [1]])
            self.assertEqual(page['marker_ids'], [0, 1])
            _, expanded = viewer.handle_message({'action': 'expand', 'value': [0], 'size': 14})
            expanded_cells = cells(expanded['html'])
            self.assertGreater(len(list(expanded_cells[0].iter())), len(list(cells(page['html'])[0].iter())))
            self.assertEqual(ET.tostring(expanded_cells[1]), ET.tostring(cells(page['html'])[1]))
            viewer.handle_message({'action': 'collapse'})
            _, part = viewer.handle_message({'action': 'open', 'value': [0], 'layout': 2})
            self.assertEqual(part['total'], 101)
            self.assertEqual(part['end'], 7)
            _, following = viewer.handle_message({'action': 'next'})
            self.assertEqual((following['start'], following['end']), (7, 14))
            viewer.handle_message({'action': 'back'})
            _, other = viewer.handle_message({'action': 'open', 'value': [1]})
            self.assertEqual(other['total'], 91)
            self.assertEqual(str(matrix), original)
        finally:
            viewer.close()

    def test_rational_cell_parts_keep_distinct_paths(self):
        value = RationalPolynomial.parse('(1+x)^40/(1+y)^35', ['x','y'])
        viewer = Matrix.from_nested([[1, value]]).mathml_viewer(page_size=7)
        try:
            page = viewer.page()
            self.assertEqual([m[0] for m in page['markers']], [[1, 0], [1, 1]])
            for part, count in [(0, 41), (1, 36)]:
                _, page = viewer.handle_message({'action': 'open', 'value': [1, part]})
                self.assertEqual(page['total'], count)
                self.assertEqual(page['end'], 7)
                viewer.handle_message({'action': 'back'})
        finally:
            viewer.close()

    def test_large_matrix_pages_cover_every_entry_once(self):
        matrix = Matrix.from_linear(12, 11, list(range(132)))
        viewer = matrix.mathml_viewer(page_size=7)
        try:
            page = viewer.page()
            seen = []
            while True:
                self.assertLess(len(json.dumps(page)), 256 * 1024)
                self.assertEqual(page['unit'], 'rows')
                self.assertEqual(len(cells(page['html'])), (page['end'] - page['start']) * 11)
                self.assertNotIn('Outside this page', page['html'])
                for cell in cells(page['html']):
                    index = int(cell.get('data-symbolica-entry'))
                    value = mathml_expression(fragment(root(ET.tostring(cell, encoding='unicode'))))
                    self.assertEqual(value, E(str(index)))
                    seen.append(index)
                if page['next'] is None: break
                _, page = viewer.handle_message({'action': 'next'})
            self.assertEqual(seen, list(range(132)))
        finally:
            viewer.close()

    def test_wide_rows_and_last_page_have_no_placeholder_cells(self):
        matrix = Matrix.from_linear(8, 40, list(range(320)))
        viewer = matrix.mathml_viewer(page_size=3)
        try:
            page = viewer.page()
            self.assertEqual((page['start'], page['end'], page['total']), (0, 3, 8))
            self.assertEqual(len(cells(page['html'])), 120)
            _, page = viewer.handle_message({'action': 'expand', 'value': [], 'size': 3})
            self.assertEqual((page['start'], page['end']), (0, 6))
            self.assertEqual(len(cells(page['html'])), 240)
            _, page = viewer.handle_message({'action': 'next'})
            self.assertEqual((page['start'], page['end']), (6, 8))
            self.assertEqual(len(cells(page['html'])), 80)
            self.assertIsNone(page['next'])
            self.assertNotIn('Outside this page', page['html'])
        finally:
            viewer.close()

    def test_coordinates_only_label_truncated_dimensions(self):
        for rows, cols, expected_rows, expected_cols in [(32, 11, True, False), (1, 40, False, False), (2, 2, False, False)]:
            viewer = Matrix.from_linear(rows, cols, list(range(rows * cols))).mathml_viewer()
            try:
                html = ET.fromstring(viewer.page()['html'])
                self.assertEqual(bool(html.findall('.//th[@scope="row"]')), expected_rows)
                self.assertEqual(bool(html.findall('.//th[@scope="col"]')), expected_cols)
            finally:
                viewer.close()

    def test_dense_large_cells_stay_within_the_transfer_budget(self):
        large = E('(1+x)^100').expand()
        matrix = Matrix.from_nested([[large] * 8 for _ in range(32)])
        viewer = matrix.mathml_viewer()
        try:
            page = viewer.page()
            self.assertEqual((page['start'], page['end'], page['total']), (0, 25, 32))
            self.assertLess(len(json.dumps(page)), 512 * 1024)
            self.assertTrue(any(path == [] and label == 'Open remaining matrix rows'
                                for path, label in page['markers']))
            self.assertEqual(len(page['marker_ids']), len(set(page['marker_ids'])))
            table = ET.fromstring(page['html']).find('.//table')
            omission = list(table)[-1]
            self.assertIn('data-symbolica-matrix-omission', omission.attrib)
            self.assertEqual(omission[0].get('colspan'), '9')
            self.assertEqual(len(list(omission.iter('mtext'))), 1)

        finally:
            viewer.close()

    def test_viewer_snapshot_and_bounds_without_notebook_imports(self):
        matrix = Matrix.from_nested([[1,2],[3,4]])
        viewer = matrix.mathml_viewer()
        try:
            original = viewer.page()['html']
            matrix.swap_rows(0, 1)
            self.assertEqual(viewer.page()['html'], original)
            _, rejected = viewer.handle_message({'action': 'open', 'value': [4]})
            self.assertIn('error', rejected)
            self.assertEqual(rejected['html'], original)
        finally:
            viewer.close()
        for size in (0, 501):
            with self.assertRaises(ValueError): matrix.mathml_viewer(page_size=size)
        for name in ['anywidget','ipywidgets','IPython','marimo','traitlets']:
            self.assertNotIn(name, sys.modules)


if __name__ == '__main__':
    unittest.main()
