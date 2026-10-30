"""Navigation and bounded reuse for the widget-independent MathML pager."""

import unittest

from symbolica import E, MathMLViewer, Matrix, RationalPolynomial


class CountingSource:
    def __init__(self, total=80, branches=None, padding=0, inline_limit=None):
        self.total = total if branches is None else len(branches)
        self.branches = branches
        self.padding = padding
        self.inline_limit = inline_limit
        self.calls = []
        self.fail_start = None
        self.hidden_root = False

    def page(self, path, start, size, layout, expansions):
        key = (tuple(path), start, size, layout,
               tuple((tuple(p), count) for p, count in expansions))
        self.calls.append(key)
        if start == self.fail_start:
            raise ValueError("temporary provider failure")
        total = self.branches[path[0]] if path and self.branches else self.total
        end = min(start + size, total)
        markers, starts = [], []
        parts = [f"<mtext>{'x' * self.padding}</mtext>"] if self.padding else []

        def marker(target, offset):
            parts.append(f'<mtext data-symbolica-marker="{len(markers)}">[?]</mtext>')
            markers.append((list(target), "Inspect omitted items"))
            starts.append(offset)

        if self.branches and not path:
            expanded = {tuple(p): count for p, count in expansions}
            for child in range(start, end):
                shown = min(expanded.get((child,), 2), self.branches[child])
                if self.inline_limit is not None:
                    shown = min(shown, self.inline_limit)
                parts.append(f"<mi>branch_{child}_{shown}</mi>")
                if shown < self.branches[child]:
                    marker([child], shown)
        else:
            parts.extend(f"<mi>x_{i}</mi>" for i in range(start, end))
        if end < total:
            marker(path, end)
        if self.hidden_root:
            # Providers can omit an unnavigable root capability, while retaining
            # its marker in HTML. This still is not a complete Show all result.
            parts.append('<mtext data-symbolica-marker="999">[?]</mtext>')
        return dict(html=f"<math><mrow>{''.join(parts)}</mrow></math>",
                    start=start, end=end, total=total,
                    markers=markers, marker_starts=starts)


class MathMLPagingTests(unittest.TestCase):
    def viewer(self, source, **kwargs):
        viewer = MathMLViewer(source, supports_expansion=True, **kwargs)
        self.addCleanup(viewer.close)
        return viewer

    def send(self, viewer, action, **kwargs):
        field, result = viewer.handle_message(dict(action=action, **kwargs))
        self.assertEqual(field, "preview" if action == "preview" else "page")
        self.assertNotIn("error", result)
        return result

    def test_inspect_and_popup_start_at_the_omitted_items(self):
        source = CountingSource()
        viewer = self.viewer(source, page_size=10)
        initial = viewer.page()
        preview = self.send(viewer, "preview", value=[], popup=True)
        self.assertEqual((preview["start"], preview["end"]), (10, 20))
        inspected = self.send(viewer, "open", value=[])
        self.assertEqual((inspected["start"], inspected["end"]), (10, 20))
        self.assertTrue(inspected["previous"])
        earlier = self.send(viewer, "previous")
        self.assertEqual(earlier["start"], 0)
        restored = self.send(viewer, "back")
        self.assertEqual(restored["html"], initial["html"])
        inspected = self.send(viewer, "open", value=[], start=35)
        self.assertEqual((inspected["start"], inspected["end"]), (35, 45))

    def test_invalid_offsets_leave_the_main_position_unchanged(self):
        viewer = self.viewer(CountingSource())
        initial = viewer.page()
        for action in ["preview", "open"]:
            for offset in [-1, True, 2**63]:
                with self.subTest(action=action, offset=offset):
                    _, result = viewer.handle_message(dict(
                        action=action, value=[], popup=True, start=offset))
                    self.assertIn("error", result)
                    self.assertEqual(viewer.page()["html"], initial["html"])
                    self.assertFalse(viewer.page()["back"])

    def test_expansion_and_targeted_collapse_preserve_page_position(self):
        viewer = self.viewer(CountingSource(), page_size=10)
        self.send(viewer, "next")
        expanded = self.send(viewer, "expand", value=[], size=10)
        self.assertEqual((expanded["start"], expanded["end"]), (10, 30))
        collapsed = self.send(viewer, "collapse", value=[])
        self.assertEqual((collapsed["start"], collapsed["end"]), (10, 20))
        self.assertTrue(collapsed["previous"])
        self.assertFalse(collapsed["can_collapse"])

    def test_targeted_collapse_keeps_other_expanded_branches(self):
        source = CountingSource(branches=[30, 40])
        viewer = self.viewer(source)
        self.send(viewer, "expand", value=[0], size=10)
        self.send(viewer, "expand", value=[1], size=10)
        preview = self.send(viewer, "preview", value=[0], popup=True)
        self.assertTrue(preview["can_collapse"])
        page = self.send(viewer, "collapse", value=[0])
        self.assertEqual(page["marker_starts"], [2, 12])
        self.assertTrue(page["can_collapse"])
        inspected = self.send(viewer, "open", value=[1])
        self.assertEqual(inspected["start"], 12)
        self.assertFalse(inspected["can_collapse"])
        self.send(viewer, "back")
        restored = self.send(viewer, "collapse")
        self.assertEqual(restored["marker_starts"], [2, 2])

    def test_show_all_reuses_its_probe_and_is_terminal(self):
        source = CountingSource()
        viewer = self.viewer(source)
        preview = self.send(viewer, "preview", value=[], popup=True)
        self.assertTrue(preview["can_show_all"])
        self.assertFalse(preview["can_collapse"])
        calls = list(source.calls)
        again = self.send(viewer, "preview", value=[], popup=True)
        self.assertTrue(again["can_show_all"])
        self.assertEqual(source.calls, calls)
        page = self.send(viewer, "expand_all", value=[])
        self.assertEqual((page["start"], page["end"], page["total"]), (0, 80, 80))
        self.assertFalse(page["markers"])
        self.assertTrue(page["can_collapse"])
        self.assertEqual(source.calls, calls)
        # A stale popup cannot apply a second, ineffective Show all request.
        _, error = viewer.handle_message(dict(action="expand_all", value=[],
                                               context=preview["context"]))
        self.assertIn("error", error)
        self.assertEqual(viewer.page()["html"], page["html"])

    def test_show_all_checks_the_actual_inline_page(self):
        for inline_limit, allowed in [(None, True), (8, False)]:
            with self.subTest(inline_limit=inline_limit):
                source = CountingSource(branches=[30, 40], inline_limit=inline_limit)
                viewer = self.viewer(source)
                compact = viewer.page()
                preview = self.send(viewer, "preview", value=[0], popup=True)
                self.assertEqual(preview["can_show_all"], allowed)
                calls = list(source.calls)
                _, page = viewer.handle_message(dict(action="expand_all", value=[0]))
                self.assertEqual(source.calls, calls)
                if allowed:
                    self.assertEqual([p for p, _ in page["markers"]], [[1]])
                    self.assertIn("branch_0_30", page["html"])
                else:
                    self.assertIn("error", page)
                    self.assertEqual(viewer.page()["html"], compact["html"])

    def test_show_all_allows_an_unrelated_parent_tail(self):
        viewer = self.viewer(CountingSource(branches=[30, 40]), page_size=1)
        preview = self.send(viewer, "preview", value=[0], popup=True)
        self.assertTrue(preview["can_show_all"])
        page = self.send(viewer, "expand_all", value=[0])
        self.assertEqual([p for p, _ in page["markers"]], [[]])
        self.assertEqual(page["marker_starts"], [1])

    def test_popup_capabilities_expire_when_paging_away(self):
        viewer = self.viewer(CountingSource(branches=[80, 80]), page_size=1)
        preview = self.send(viewer, "preview", value=[0], popup=True)
        self.assertTrue(preview["markers"])
        current = self.send(viewer, "next")
        _, failure = viewer.handle_message(dict(action="expand_all", value=[0],
                                                context=preview["context"]))
        self.assertIn("Unknown or expired popup", failure["error"])
        self.assertEqual(viewer.page()["html"], current["html"])

    def test_ancestor_tail_cannot_grow_and_hide_an_opaque_child_path(self):
        class OpaqueRows(CountingSource):
            def page(self, path, start, size, layout, expansions):
                if path:
                    return CountingSource(total=20).page(path, start, size, layout, expansions)
                end = 50 if expansions and self.inline_limit else 60
                markers = [] if expansions else [([0], "Inspect cell in row 55")]
                starts = [] if expansions else [2]
                markers.append(([], "Inspect remaining rows"))
                starts.append(end)
                html = ''.join(f'<mtext data-symbolica-marker="{i}">[?]</mtext>'
                               for i in range(len(markers)))
                return dict(html=f"<math>{html}</math>", start=0, end=end, total=100,
                            markers=markers, marker_starts=starts)

        for grows in [False, True]:
            with self.subTest(grows=grows):
                viewer = self.viewer(OpaqueRows(inline_limit=grows), page_size=100)
                preview = self.send(viewer, "preview", value=[0], popup=True)
                self.assertEqual(preview["can_show_all"], not grows)

    def test_show_all_rejects_hidden_or_oversized_results(self):
        for total, hidden in [(501, False), (80, True)]:
            with self.subTest(total=total, hidden=hidden):
                source = CountingSource(total=total)
                source.hidden_root = hidden
                viewer = self.viewer(source)
                preview = self.send(viewer, "preview", value=[], popup=True)
                self.assertFalse(preview["can_show_all"])
                if total > 500:
                    self.assertFalse(any(call[2] == 500 for call in source.calls))
                _, response = viewer.handle_message(dict(action="expand_all", value=[]))
                self.assertIn("error", response)

    def test_hover_does_not_probe_show_all_and_reuses_rendering(self):
        source = CountingSource()
        viewer = self.viewer(source)
        first = self.send(viewer, "preview", value=[])
        calls = list(source.calls)
        self.assertFalse(first["can_show_all"])
        self.assertFalse(any(call[2] == 500 for call in calls))
        second = self.send(viewer, "preview", value=[])
        self.assertEqual(second["html"], first["html"])
        self.assertEqual(source.calls, calls)
        self.assertNotEqual(second["context"], first["context"])

    def test_cache_evicts_by_entry_and_byte_limits(self):
        for padding, previews in [(0, 30), (200_000, 12)]:
            with self.subTest(padding=padding):
                source = CountingSource(total=1000, padding=padding)
                viewer = self.viewer(source)
                for start in range(1, previews + 1):
                    self.send(viewer, "preview", value=[], popup=True, start=start, size=1)
                first_key = ((), 1, 1, 0, ())
                self.assertEqual(source.calls.count(first_key), 1)
                self.send(viewer, "preview", value=[], popup=True, start=1, size=1)
                self.assertEqual(source.calls.count(first_key), 2)

    def test_provider_errors_are_not_cached(self):
        source = CountingSource()
        viewer = self.viewer(source)
        initial = viewer.page()
        source.fail_start = 25
        _, failure = viewer.handle_message(dict(action="preview", value=[]))
        self.assertIn("temporary provider failure", failure["error"])
        source.fail_start = None
        preview = self.send(viewer, "preview", value=[])
        self.assertEqual(preview["start"], 25)
        self.assertEqual(viewer.page()["html"], initial["html"])
        self.assertEqual(source.calls.count(((), 25, 5, 0, ())), 2)

    def test_real_expression_show_all_remains_bounded(self):
        for degree, allowed in [(80, True), (600, False)]:
            with self.subTest(degree=degree):
                viewer = E(f"(1+pager_navigation_x)^{degree}").expand().mathml_viewer()
                self.addCleanup(viewer.close)
                compact = viewer.page()
                preview = self.send(viewer, "preview", value=[], popup=True)
                self.assertEqual(preview["can_show_all"], allowed)
                if allowed:
                    complete = self.send(viewer, "expand_all", value=[])
                    self.assertEqual(complete["end"], degree + 1)
                    self.assertFalse(complete["markers"])
                    self.assertLessEqual(len(complete["html"].encode()), 256 * 1024)
                    self.assertEqual(self.send(viewer, "collapse")["html"], compact["html"])

    def test_rational_denominator_show_all_uses_child_paths_not_item_indices(self):
        poly = RationalPolynomial.parse('(1+x)^40/(1+y)^35', ['x', 'y'])
        viewer = poly.mathml_viewer(page_size=7)
        self.addCleanup(viewer.close)
        self.assertEqual(viewer.page()["total"], 1)
        preview = self.send(viewer, "preview", value=[1], popup=True)
        self.assertTrue(preview["can_show_all"])
        page = self.send(viewer, "expand_all", value=[1])
        self.assertNotIn([1], [p for p, _ in page["markers"]])
        self.assertIn([0], [p for p, _ in page["markers"]])

    def test_matrix_cell_show_all_keeps_an_unrelated_row_tail(self):
        large = E('(1+pager_matrix_x)^20').expand()
        viewer = Matrix.from_nested([[1, large], [2, 3]]).mathml_viewer(page_size=1)
        self.addCleanup(viewer.close)
        page = viewer.page()
        self.assertEqual([p for p, _ in page['markers']], [[1], []])
        preview = self.send(viewer, 'preview', value=[1], popup=True)
        self.assertTrue(preview['can_show_all'])
        page = self.send(viewer, 'expand_all', value=[1])
        self.assertEqual([p for p, _ in page['markers']], [[]])
        self.assertEqual(page['marker_starts'], [1])


if __name__ == "__main__":
    unittest.main()
