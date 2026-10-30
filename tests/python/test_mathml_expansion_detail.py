"""Expanding a sum reveals complete terms in every responsive alternative."""

import unittest
from xml.etree import ElementTree

from symbolica import E, S


class ExpansionDetailTests(unittest.TestCase):
    def test_initial_wrapped_sums_offer_shorter_complete_prefixes(self):
        polynomial = E("(x+y+z+w)^20").expand()
        for expression, prefix in ((polynomial, []), (polynomial * E("1+a"), [1])):
            with self.subTest(prefix=prefix):
                viewer = expression.mathml_viewer()
                self.addCleanup(viewer.close)
                initial = viewer.page()

                def terms(page):
                    result = {}
                    for node in ElementTree.fromstring(page["html"]).iter():
                        path = node.get("data-symbolica-path")
                        if path is None:
                            continue
                        path = [int(p) for p in path.split("/")]
                        if len(path) == len(prefix) + 1 and path[:-1] == prefix:
                            result[path[-1]] = "".join(node.itertext())
                    return result

                complete = terms(initial)
                lengths = []
                for page in [initial, *initial["layouts"]]:
                    self.assertEqual([path for path, _ in page["markers"]], [prefix])
                    shown = terms(page)
                    self.assertTrue(shown)
                    self.assertTrue(all(complete.get(k) == v for k, v in shown.items()))
                    if prefix and page.get("layout") == 4:
                        self.assertIn('data-symbolica-sum=', page["html"])
                    lengths.append(len(shown))
                self.assertLess(min(lengths), max(lengths))

    def test_show_more_preserves_polynomial_factors_in_compact_layouts(self):
        x, y, z, w, a = S("x", "y", "z", "w", "a")
        expression = ((x + y + z + w) ** 20).expand() * (1 + a)
        original = bytes(expression.__getstate__())
        viewer = expression.formatted().mathml_viewer()
        self.addCleanup(viewer.close)
        initial = viewer.page()
        for page in [initial, *initial["layouts"]]:
            with self.subTest(initial_layout=page["layout"]):
                self.assertIn("<mi>a</mi>", page["html"])
                self.assertFalse(any(path[:1] == [0] for path, _ in page["markers"]))
        compact = next(page for page in initial["layouts"] if page["layout"] == 1)
        self.assertEqual([path for path, _ in compact["markers"]], [[1]])
        expected = compact["marker_starts"][0] + 25
        field, expanded = viewer.handle_message(
            {"action": "expand", "value": [1], "layout": 1, "size": 25}
        )
        self.assertEqual(field, "page")
        self.assertNotIn("error", expanded)
        for page in [expanded, *expanded["layouts"]]:
            with self.subTest(layout=page["layout"]):
                self.assertEqual([path for path, _ in page["markers"]], [[1]])
                self.assertEqual(page["marker_starts"], [expected])
                self.assertIn('data-symbolica-sum=""', page["html"])
                self.assertLessEqual(len(page["html"].encode()), 256 * 1024)
        _, collapsed = viewer.handle_message({"action": "collapse"})
        self.assertEqual(collapsed["html"], initial["html"])
        self.assertEqual(bytes(expression.__getstate__()), original)

    def test_short_native_power_tails_stay_visible(self):
        viewer = E("19380*x*y^3*z^16+q").mathml_viewer()
        self.addCleanup(viewer.close)
        page = viewer.page()
        for layout in [page, *page["layouts"]]:
            with self.subTest(layout=layout["layout"]):
                self.assertFalse(layout["markers"])
                self.assertIn("<mn>16</mn>", layout["html"])

    def test_parentheses_allow_taller_detail_to_replace_an_omission(self):
        viewer = (E("(x+y+z+w)^20").expand() * E("1+a")).mathml_viewer()
        self.addCleanup(viewer.close)
        initial = viewer.page()
        for page in [initial, *initial["layouts"]]:
            with self.subTest(layout=page["layout"]):
                tree = ElementTree.fromstring(page["html"])
                for row in tree.iter("{http://www.w3.org/1998/Math/MathML}mrow"):
                    if len(row) < 2 or row[0].text != "(" or row[-1].text != ")":
                        continue
                    if any("data-symbolica-marker" in child.attrib for child in row.iter()):
                        self.assertNotEqual(row[0].get("stretchy"), "false")
                        self.assertNotEqual(row[-1].get("stretchy"), "false")

    def test_larger_root_sum_page_keeps_complete_terms(self):
        viewer = E("(x+y+z+w)^20").expand().mathml_viewer()
        self.addCleanup(viewer.close)
        viewer.page()
        _, expanded = viewer.handle_message({"action": "size", "value": 50})
        self.assertNotIn("error", expanded)
        for page in [expanded, *expanded["layouts"]]:
            with self.subTest(layout=page["layout"]):
                self.assertEqual(page["end"], 50)
                self.assertEqual([path for path, _ in page["markers"]], [[]])
                self.assertEqual(page["marker_starts"], [50])


if __name__ == "__main__":
    unittest.main()
