"""Automatic product previews use rendered cost without changing explicit paging."""

import unittest

from symbolica import E, MathMLViewer


class ProductSource:
    def __init__(self, total=34, candidate="complete", kind="Product"):
        self.total = total
        self.candidate = candidate
        self.kind = kind
        self.calls = []

    def page(self, path, start, size, layout, expansions):
        self.calls.append((tuple(path), start, size, layout, tuple(expansions)))
        probing = size == self.total and size > 25
        if probing and self.candidate == "error":
            raise ValueError("temporary candidate failure")
        end = min(start + size, self.total)
        if probing and self.candidate == "partial":
            end -= 1
        elif probing and self.candidate == "no_progress":
            end = 25
        # Indexed factors are small in the rendered output even when their
        # provider-side expressions contain much larger tensor descriptors.
        parts = [f"<msup><mi>g</mi><mrow><mi>a</mi><mn>{i}</mn></mrow></msup>"
                 for i in range(start, end)]
        markers, starts = [], []
        if end < self.total:
            markers.append((list(path), "Inspect product"))
            starts.append(end)
            parts.append('<mtext data-symbolica-marker="0">[1]</mtext>')
        if probing:
            if self.candidate == "bytes":
                parts.append(f"<mrow title=\"{'x' * 262_144}\"/>")
            elif self.candidate == "elements":
                parts.extend("<mrow/>" for _ in range(2000))
            elif self.candidate == "visible":
                parts.append(f"<mtext>{'x' * 768}</mtext>")
            elif self.candidate == "markup":
                # Rich notation can carry long source labels while displaying
                # only a compact indexed tensor. Metadata is not visible cost.
                parts = [f'<mrow title="{"tensor source label " * 40}">{part}</mrow>'
                         for part in parts]
            elif self.candidate == "hidden":
                parts.append('<mtext data-symbolica-marker="99">[100]</mtext>')
        return dict(html=f"<math><mrow>{''.join(parts)}</mrow></math>",
                    start=start, end=end, total=self.total, kind=self.kind,
                    markers=markers, marker_starts=starts)


class AutomaticProductPreviewTests(unittest.TestCase):
    def viewer(self, source, automatic=True):
        viewer = MathMLViewer(source, supports_expansion=True,
                              _automatic_product_preview=automatic)
        self.addCleanup(viewer.close)
        return viewer

    def send(self, viewer, action, **kwargs):
        field, page = viewer.handle_message(dict(action=action, **kwargs))
        self.assertEqual(field, "page")
        self.assertNotIn("error", page)
        return page

    def test_small_complete_product_uses_one_cached_candidate(self):
        source = ProductSource()
        viewer = self.viewer(source)
        page = viewer.page()
        self.assertEqual((page["end"], page["total"], page["page_size"]), (34, 34, 34))
        self.assertFalse(page["markers"])
        self.assertFalse(page["can_collapse"])
        self.assertEqual([call[2] for call in source.calls], [25, 34])
        calls = list(source.calls)
        self.assertEqual(viewer.page()["html"], page["html"])
        self.assertEqual(viewer._render_page(0)["html"], page["html"])
        self.assertEqual(source.calls, calls)

    def test_explicit_size_and_later_navigation_remain_exact(self):
        explicit_source = ProductSource()
        explicit = self.viewer(explicit_source, automatic=False)
        self.assertEqual(explicit.page()["end"], 25)
        self.assertEqual([call[2] for call in explicit_source.calls], [25])

        source = ProductSource()
        automatic = self.viewer(source)
        self.assertEqual(automatic.page()["end"], 34)
        selected = self.send(automatic, "size", value=25)
        self.assertEqual((selected["end"], selected["page_size"]), (25, 25))
        self.assertEqual(automatic.page()["end"], 25)
        later = self.send(automatic, "next")
        self.assertEqual((later["start"], later["end"]), (25, 34))
        self.assertEqual(self.send(automatic, "previous")["end"], 25)
        self.assertEqual(self.send(automatic, "collapse")["end"], 25)
        self.assertEqual(sum(call[2] == 34 for call in source.calls), 1)

    def test_short_visible_product_can_exceed_one_page_of_markup(self):
        source = ProductSource(total=50, candidate="markup")
        viewer = self.viewer(source)
        page = viewer.page()
        self.assertEqual((page["end"], page["total"], page["page_size"]), (50, 50, 50))
        self.assertFalse(page["markers"])
        self.assertGreater(len(page["html"].encode()), 32_768)
        self.assertLessEqual(len(page["html"].encode()), 262_144)
        self.assertEqual([call[2] for call in source.calls], [25, 50])
        calls = list(source.calls)
        self.assertEqual(viewer.page()["html"], page["html"])
        self.assertEqual(source.calls, calls)

        explicit = self.viewer(ProductSource(total=50, candidate="markup"), automatic=False)
        self.assertEqual((explicit.page()["end"], explicit.page()["page_size"]), (25, 25))

    def test_incomplete_or_large_candidates_keep_the_original_page(self):
        for candidate in ["no_progress", "elements", "visible", "hidden"]:
            with self.subTest(candidate=candidate):
                source = ProductSource(candidate=candidate)
                viewer = self.viewer(source)
                page = viewer.page()
                self.assertEqual((page["end"], page["page_size"]), (25, 25))
                self.assertTrue(page["markers"])
                self.assertEqual([call[2] for call in source.calls], [25, 34])
                calls = list(source.calls)
                self.assertEqual(viewer.page()["html"], page["html"])
                self.assertEqual(source.calls, calls)

    def test_factor_count_does_not_replace_the_visible_budget(self):
        for total in (51, 100):
            with self.subTest(total=total):
                source = ProductSource(total=total)
                viewer = self.viewer(source)
                self.assertEqual(viewer.page()["end"], total)
                self.assertEqual([call[2] for call in source.calls], [25, total])
        source = ProductSource(kind="Sum")
        viewer = self.viewer(source)
        self.assertEqual(viewer.page()["end"], 25)
        self.assertEqual([call[2] for call in source.calls], [25])

    def test_longer_bounded_prefix_can_keep_the_product_tail(self):
        source = ProductSource(candidate="partial")
        viewer = self.viewer(source)
        page = viewer.page()
        self.assertEqual((page["end"], page["total"]), (33, 34))
        self.assertEqual(page["marker_starts"], [33])
        self.assertEqual([call[2] for call in source.calls], [25, 34])

        class ManyTinyFactors(ProductSource):
            def page(self, *args):
                page = super().page(*args)
                factors = "<mi>x</mi>" * page["end"]
                tail = '<mtext data-symbolica-marker="0">[1]</mtext>'
                page["html"] = f"<math><mrow>{factors}{tail}</mrow></math>"
                return page

        source = ManyTinyFactors(total=100_000)
        viewer = self.viewer(source)
        page = viewer.page()
        self.assertEqual((page["end"], page["total"], page["page_size"]),
                         (500, 100_000, 500))
        self.assertEqual(page["marker_starts"], [500])
        self.assertEqual([call[2] for call in source.calls], [25, 500])
        self.assertEqual(viewer.page()["end"], 500)
        self.assertEqual([call[2] for call in source.calls], [25, 500])

    def test_more_factors_preserve_existing_nested_detail(self):
        class NestedProduct(ProductSource):
            def page(self, path, start, size, layout, expansions):
                self.calls.append((tuple(path), start, size, layout, tuple(expansions)))
                end = 13 if size == 25 else 37
                marker_path, marker_start = self.candidate if size > 25 else ([4], 2)
                markers = [(marker_path, "Inspect sum")]
                starts = [marker_start]
                if end < 37:
                    markers.append(([], "Inspect product"))
                    starts.append(end)
                factors = "<mi>x</mi>" * end
                holes = "".join(f'<mtext data-symbolica-marker="{i}">[{i + 1}]</mtext>'
                                for i in range(len(markers)))
                return dict(html=f"<math><mrow>{factors}{holes}</mrow></math>",
                            start=0, end=end, total=37, kind="Product",
                            markers=markers, marker_starts=starts)

        for candidate, accepted in (
            (([4], 2), True), (([4], 3), True),
            (([4, 3], 0), True), (([20], 0), True),
            (([4], 1), False), (([4, 1], 0), False),
            (([3], 0), False), (([], 4), False),
        ):
            with self.subTest(candidate=candidate):
                source = NestedProduct(total=37, candidate=candidate)
                viewer = self.viewer(source)
                page = viewer.page()
                self.assertEqual(page["end"], 37 if accepted else 13)
                if accepted:
                    self.assertEqual(page["markers"][0][0], candidate[0])
                    self.assertEqual(page["marker_starts"][0], candidate[1])
                self.assertEqual([call[2] for call in source.calls], [25, 37])

    def test_optional_candidate_errors_are_not_cached(self):
        for candidate in ("error", "bytes"):
            with self.subTest(candidate=candidate):
                source = ProductSource(candidate=candidate)
                viewer = self.viewer(source)
                self.assertEqual(viewer.page()["end"], 25)
                source.candidate = "complete"
                self.assertEqual(viewer.page()["end"], 34)
                self.assertEqual([call[2] for call in source.calls], [25, 34, 34])

    def test_native_expression_default_and_explicit_sizes(self):
        expression = E("*".join(f"x{i}" for i in range(34)))
        automatic = expression.mathml_viewer()
        explicit = expression.mathml_viewer(25)
        self.addCleanup(automatic.close)
        self.addCleanup(explicit.close)
        self.assertEqual(automatic.page()["end"], 34)
        self.assertFalse(automatic.page()["markers"])
        self.assertEqual(explicit.page()["end"], 25)

        # More markup is affordable for short indexed labels, not for long
        # visible factor names which would make the automatic preview unwieldy.
        small_names = E("*".join(f"x{i}" for i in range(100)))
        automatic = small_names.mathml_viewer()
        self.addCleanup(automatic.close)
        self.assertEqual(automatic.page()["end"], 100)
        long_names = E("*".join(f"long_visible_factor_name_{i}" for i in range(100)))
        automatic = long_names.mathml_viewer()
        complete = long_names.mathml_viewer(100)
        self.addCleanup(automatic.close)
        self.addCleanup(complete.close)
        self.assertEqual(automatic.page()["end"], 25)
        self.assertEqual(complete.page()["end"], 100)
        self.assertFalse(complete.page()["markers"])

        # Increasing the item allowance cannot make a genuinely large product
        # eligible: its complete output still contains budget omissions.
        large_sum = "+".join(f"y{i}" for i in range(100))
        complex_product = E("*".join(f"f{i}({large_sum})" for i in range(34)))
        automatic = complex_product.mathml_viewer()
        explicit = complex_product.mathml_viewer(25)
        self.addCleanup(automatic.close)
        self.addCleanup(explicit.close)
        self.assertEqual(automatic.page()["html"], explicit.page()["html"])
        self.assertTrue(automatic.page()["markers"])


if __name__ == "__main__":
    unittest.main()
