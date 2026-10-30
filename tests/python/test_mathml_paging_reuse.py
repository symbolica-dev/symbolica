"""Reuse cached pages without extra rendering or exposing mutable cache state."""

import unittest

from symbolica import MathMLViewer


class LayoutSource:
    def __init__(self):
        self.calls = []

    def page(self, path, start, size, layout, expansions):
        self.calls.append((tuple(path), start, size, layout))
        end = min(start + size, 100)
        # Layout1 deliberately repeats0, while every later layout is distinct.
        label = 0 if layout == 1 else layout
        return dict(html=(f"<math><mi>layout_{label}</mi>"
                          '<mtext data-symbolica-marker="0">[1]</mtext></math>'),
                    start=start, end=end, total=100, kind="Sum",
                    markers=[(list(path), "Inspect sum")], marker_starts=[end])


class MathMLPageReuseTests(unittest.TestCase):
    def viewer(self, source, **kwargs):
        viewer = MathMLViewer(source, layout_count=8, supports_expansion=True, **kwargs)
        self.addCleanup(viewer.close)
        return viewer

    def test_static_output_does_not_render_unused_responsive_layouts(self):
        source = LayoutSource()
        viewer = self.viewer(source)
        self.assertIn("layout_0", viewer._repr_html_())
        self.assertEqual(source.calls, [((), 0, 25, 0)])
        page = viewer.page()
        self.assertEqual([p["layout"] for p in page["layouts"]], [2, 3, 4, 5, 6, 7])
        self.assertEqual([call[3] for call in source.calls], list(range(8)))

    def test_hover_needs_only_the_selected_main_capability(self):
        source = LayoutSource()
        viewer = self.viewer(source)
        field, preview = viewer.handle_message(dict(action="preview", value=[], layout=0))
        self.assertEqual(field, "preview")
        self.assertNotIn("error", preview)
        self.assertEqual((preview["start"], preview["end"]), (25, 30))
        self.assertEqual([call for call in source.calls if call[1] == 0], [((), 0, 25, 0)])
        self.assertEqual([call[3] for call in source.calls if call[1] == 25], list(range(8)))
        calls = list(source.calls)
        viewer.handle_message(dict(action="preview", value=[], layout=0))
        self.assertEqual(source.calls, calls)

    def test_selected_layouts_keep_deduplication_and_provider_order(self):
        source = LayoutSource()
        viewer = self.viewer(source)
        with self.assertRaisesRegex(ValueError, "Unknown layout"):
            viewer._selected_page(dict(layout=1))
        self.assertEqual([call[3] for call in source.calls], [0, 1])
        selected = viewer._selected_page(dict(layout=3))
        self.assertEqual(selected["layout"], 3)
        self.assertIn("layout_3", selected["html"])
        self.assertEqual([call[3] for call in source.calls], [0, 1, 2, 3])
        calls = list(source.calls)
        with self.assertRaisesRegex(ValueError, "Unknown layout"):
            viewer._selected_page(dict(layout=8))
        self.assertEqual(source.calls, calls)

    def test_response_mutation_cannot_change_cached_pages_or_capabilities(self):
        viewer = self.viewer(LayoutSource())
        original = viewer.page()
        original["html"] = "poisoned"
        original["markers"][0][0].append(99)
        original["marker_ids"][0] = 42
        original["marker_starts"][0] = 0
        original["layouts"][0]["markers"][0][0].append(98)
        original["layouts"][0]["html"] = "also poisoned"

        fresh = viewer.page()
        self.assertIn("layout_0", fresh["html"])
        self.assertEqual(fresh["markers"][0][0], [])
        self.assertEqual(fresh["markers"][0][1], "Inspect sum")
        self.assertEqual(fresh["marker_ids"], [0])
        self.assertEqual(fresh["marker_starts"], [25])
        self.assertIn("layout_2", fresh["layouts"][0]["html"])
        self.assertEqual(fresh["layouts"][0]["markers"][0][0], [])
        _, error = viewer.handle_message(dict(action="preview", value=[99]))
        self.assertIn("Unknown subexpression", error["error"])
        _, preview = viewer.handle_message(dict(action="preview", value=[]))
        self.assertNotIn("error", preview)

    def test_selected_layout_uses_one_automatic_preview_snapshot(self):
        class TransientProduct(LayoutSource):
            attempts = 0

            def page(self, path, start, size, layout, expansions):
                if start == 0 and size == 34 and layout == 0:
                    self.attempts += 1
                    if self.attempts == 1:
                        raise ValueError("temporary optional probe failure")
                result = super().page(path, start, size, layout, expansions)
                result.update(kind="Product", total=34, end=min(start + size, 34))
                if result["end"] == 34:
                    result["markers"] = []
                    result["marker_starts"] = []
                    result["html"] = f"<math><mi>complete_{layout}</mi></math>"
                return result

        for action, layout in (("preview", 3), ("selected", 3), ("selected", 0)):
            with self.subTest(action=action, layout=layout):
                source = TransientProduct()
                viewer = self.viewer(source, _automatic_product_preview=True)
                if action == "preview":
                    _, preview = viewer.handle_message(dict(action=action, value=[], layout=layout))
                    self.assertNotIn("error", preview)
                    self.assertEqual((preview["start"], preview["end"]), (25, 30))
                else:
                    page = viewer._selected_page(dict(layout=layout))
                    self.assertEqual((page["layout"], page["end"]), (layout, 25))
                self.assertEqual(source.attempts, 1)
                expected_layouts = 8 if layout == 0 else 4
                self.assertEqual([call[2] for call in source.calls if call[1] == 0], [25] * expected_layouts)
                # A later request can retry the failed optional probe and adopt
                # the complete product without changing sizes during selection.
                self.assertEqual(viewer.page()["end"], 34)
                self.assertEqual(source.attempts, 2)


if __name__ == "__main__":
    unittest.main()
