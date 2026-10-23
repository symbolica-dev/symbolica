"""Static MathML embeds remain usable in constrained notebook layouts."""

from html.parser import HTMLParser
from fractions import Fraction
import builtins
import json
import unittest
from unittest.mock import patch
from xml.etree import ElementTree

from symbolica import E, S, Expression, FormattedOutput, MathMLViewer, PrintMode


def mathml_value(html, variables):
    """Evaluate the arithmetic MathML subset used by these sign regressions."""
    def tokens(node):
        tag = node.tag.rsplit("}", 1)[-1]
        if tag == "mn":
            yield Fraction(node.text)
        elif tag == "mi":
            yield Fraction(variables[node.text])
        elif tag == "mspace":
            pass
        elif tag == "mo":
            operator = node.text
            if operator in ("(", ")", "+", "−"):
                yield operator
            elif operator not in ("*", "·", "×", "\u2062"):
                raise AssertionError(f"Unsupported arithmetic operator: {operator}")
        elif tag == "mfrac":
            yield value(node[0]) / value(node[1])
        elif tag == "msup":
            yield value(node[0]) ** value(node[1])
        elif tag in ("math", "mrow", "mtable", "mtr", "mtd"):
            for child in node:
                yield from tokens(child)
        else:
            raise AssertionError(f"Unsupported arithmetic element: {tag}")

    def value(node):
        items = list(tokens(node))
        index = 0

        def factor():
            nonlocal index
            sign = 1
            while items[index] in ("+", "−"):
                if items[index] == "−":
                    sign = -sign
                index += 1
            item = items[index]
            index += 1
            if item == "(":
                item = expression()
                assert items[index] == ")"
                index += 1
            assert isinstance(item, Fraction)
            return sign * item

        def product():
            nonlocal index
            result = factor()
            while index < len(items) and items[index] not in ("+", "−", ")"):
                result *= factor()
            return result

        def expression():
            nonlocal index
            result = product()
            while index < len(items) and items[index] in ("+", "−"):
                operator = items[index]
                index += 1
                term = product()
                result += term if operator == "+" else -term
            return result

        result = expression()
        assert index == len(items)
        return result

    return value(ElementTree.fromstring(f"<math>{html}</math>"))


class MathDisplay(HTMLParser):
    def __init__(self, html):
        super().__init__()
        self.stack = []
        self.math = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "math":
            self.math.append((attrs, list(self.stack)))
        self.stack.append((tag, attrs))

    def handle_endtag(self, tag):
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break


class MathMLDisplayTests(unittest.TestCase):
    def test_scientific_floats_preserve_displayed_digits_and_value(self):
        for text in ["4.21359549995700e+1", "-4.21359549995700e+1",
                     "1.2300e-7", "-1.2300e-7", "1.25e100", "1.25e-100"]:
            expression = E(text)
            original = bytes(expression.__getstate__())
            for precision in [None, 6, 15, 25]:
                options = {} if precision is None else {"precision": precision}
                plain = expression.format(**options)
                html = expression.format(mode=PrintMode.MathML, **options)
                with self.subTest(text=text, precision=precision, plain=plain):
                    root = ElementTree.fromstring("<math>" + html + "</math>")
                    powers = list(root.iter("msup"))
                    self.assertEqual(mathml_value(html, {}), Fraction(plain))
                    if "e" in plain.lower():
                        mantissa, exponent = plain.lower().split("e")
                        self.assertEqual(len(powers), 1)
                        self.assertEqual(powers[0][0].text, "10")
                        self.assertIn("<mn>" + mantissa.lstrip("-") + "</mn>", html)
                        self.assertIn("<mo>×</mo>", html)
                        self.assertEqual(mathml_value(ElementTree.tostring(powers[0][1], encoding="unicode"), {}), int(exponent))
                    else:
                        self.assertFalse(powers)
                        self.assertIn("<mn>" + plain.lstrip("-") + "</mn>", html)
            self.assertEqual(bytes(expression.__getstate__()), original)

    def test_scientific_coefficients_keep_product_power_and_complex_grouping(self):
        scalar = E("-4.21359549995700e+1")
        imaginary = E("2.5e-3")
        real_value = Fraction(scalar.format())
        imag_value = Fraction(imaginary.format())
        for expression, expected in [
            (scalar * S("x"), real_value * 2),
            (scalar / S("x"), real_value / 2),
            (scalar ** S("n"), real_value ** 2),
            (S("x") + scalar, 2 + real_value),
            (scalar + imaginary * S("𝑖"), real_value + 3 * imag_value),
            (scalar - imaginary * S("𝑖"), real_value - 3 * imag_value),
        ]:
            viewer = expression.mathml_viewer()
            try:
                for html in [expression.format(mode=PrintMode.MathML), viewer.page()["html"]]:
                    with self.subTest(expression=str(expression)):
                        self.assertEqual(mathml_value(html, {"x": 2, "n": 2, "𝑖": 3}), expected)
            finally:
                viewer.close()

    def test_default_viewer_juxtaposes_symbolic_product_factors(self):
        expression = E("*".join(f"product_spacing::x{i}" for i in range(10)))
        original = bytes(expression.__getstate__())
        viewer = expression.mathml_viewer()
        try:
            root = ElementTree.fromstring(viewer.page()["html"])
            operators = [
                node.text for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "mo"
            ]
            self.assertEqual(operators.count("\u2062"), 9)
            self.assertNotIn("·", operators)
            explicit = expression.format(
                mode=PrintMode.MathML, multiplication_operator="×"
            )
            self.assertEqual(explicit.count("<mo>×</mo>"), 9)
            self.assertEqual(bytes(expression.__getstate__()), original)
        finally:
            viewer.close()

    def test_adjacent_numeric_custom_fragments_keep_a_separator(self):
        expression = E("1")
        for index in range(10):
            expression *= S(
                f"numeric_product_spacing::n{index}",
                print={"mathml": f"<mn>{index + 2}</mn>"},
            )
        original = bytes(expression.__getstate__())
        viewer = expression.mathml_viewer()
        try:
            root = ElementTree.fromstring(viewer.page()["html"])
            operators = [
                node.text for node in root.iter() if node.tag.rsplit("}", 1)[-1] == "mo"
            ]
            self.assertEqual(operators.count("·"), 9)
            self.assertEqual(bytes(expression.__getstate__()), original)
        finally:
            viewer.close()

    def test_rendered_signs_preserve_the_expression_value(self):
        polynomial = E("x^4+4*y^4")
        factored = polynomial.factor()
        self.assertEqual(factored.expand(), polynomial)
        expressions = [factored, *map(E, [
            "(-x+y)*(x+y)",
            "-(-x+y)*(x+y)",
            "(-x+y)*(-x+2*y)",
            "(-x-y)*(x-y)",
            "(-x+y)/(x+y)",
            "(-x+y)*(x+y)/(x+2*y)",
            "(-x+y)*(x+y)+3*x",
            "-3*x/7",
            "x-3*y/7",
            "(-x+y)^2",
            "(-x+y)^-2",
            "(-2*x+y)^-3/7",
        ])]
        for expression in expressions:
            viewer = expression.mathml_viewer()
            try:
                page = viewer.page()
                self.assertFalse(page["markers"])
                outputs = [expression.format(mode=PrintMode.MathML), page["html"]]
                for x, y in [(2, 3), (-3, 2), (5, -1)]:
                    expected = expression.replace(S("x"), E(str(x))).replace(S("y"), E(str(y)))
                    for html in outputs:
                        with self.subTest(expression=str(expression), x=x, y=y, html=html):
                            displayed = mathml_value(html, {"x": x, "y": y})
                            self.assertEqual(E(str(displayed)), expected)
            finally:
                viewer.close()

    def test_two_large_factors_retain_all_terms_and_reflow_boundaries(self):
        left = E("+".join(f"product_left_{i}" for i in range(12)))
        right = E("+".join(f"product_right_{i}" for i in range(12)))
        expression = left * right
        before = expression.format()
        viewer = expression.mathml_viewer()
        try:
            page = viewer.page()
            self.assertFalse(page["markers"])
            self.assertEqual((page["start"], page["end"], page["total"]), (0, 2, 2))
            math = ElementTree.fromstring(page["html"])
            self.assertEqual(sum("data-symbolica-factor" in node.attrib for node in math.iter()), 2)
            self.assertEqual(sum(node.tag.endswith("mi") for node in math.iter()), 24)
            self.assertEqual(sum("data-symbolica-term" in node.attrib for node in math.iter()), 24)
            self.assertEqual(expression.format(), before)
        finally:
            viewer.close()

    def test_standalone_viewer_never_imports_notebook_packages(self):
        expression = E("(1+x)^100").expand()
        original_import = builtins.__import__

        def checked_import(name, *args, **kwargs):
            if name.split(".")[0] in {"anywidget", "ipywidgets", "traitlets", "marimo", "IPython"}:
                raise AssertionError(f"Unexpected notebook dependency: {name}")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", checked_import):
            viewer = expression.mathml_viewer()
            self.assertIsInstance(viewer, MathMLViewer)
            page = json.loads(json.dumps(viewer.page()))
            self.assertEqual((page["start"], page["end"], page["total"]), (0, 25, 101))
            self.assertLess(len(page["html"].encode()), 256 * 1024)
            self.assertTrue(page["markers"])
            self.assertIsInstance(MathMLViewer.javascript(), str)
            formatted = expression.formatted(page_size=7)
            other = formatted.mathml_viewer()
            self.assertIs(other, formatted.mathml_viewer())
            self.assertEqual(other.page()["end"], 7)
            self.assertIsNone(FormattedOutput("plain").mathml_viewer())
            viewer.close()
            other.close()
            with self.assertRaises(RuntimeError):
                viewer.page()

    def test_worker_messages_page_forward_and_back_without_changing_expression(self):
        expression = E("(1+x)^100").expand()
        before = expression.format()
        viewer = expression.mathml_viewer(page_size=10)
        field, first = viewer.handle_message({"action": "attach", "view": "worker"})
        self.assertEqual(field, "page")
        ranges = [(first["start"], first["end"])]
        while first["next"] is not None:
            field, first = json.loads(json.dumps(viewer.handle_message({"action": "next"})))
            self.assertEqual(field, "page")
            ranges.append((first["start"], first["end"]))
        self.assertEqual(ranges[-1][1], 101)
        self.assertTrue(all(a[1] == b[0] for a, b in zip(ranges, ranges[1:])))
        for start, end in reversed(ranges[:-1]):
            field, page = viewer.handle_message({"action": "previous"})
            self.assertEqual((page["start"], page["end"]), (start, end))
        self.assertEqual(expression.format(), before)
        self.assertEqual(viewer.handle_message({"action": "dispose", "view": "worker"})[0], "")
        viewer.close()

    def test_standalone_page_size_is_bounded(self):
        for size in (0, 501):
            with self.assertRaises(ValueError):
                E("x").mathml_viewer(page_size=size)

    def test_static_output_scrolls_html_ancestors_instead_of_math(self):
        for expression in [E("sin(1/x)"), E("sqrt(1+x^2)"), E("x+y+z")]:
            for html in [expression._repr_html_(), expression.formatted()._repr_html_()]:
                with self.subTest(expression=str(expression), html=html):
                    display = MathDisplay(html)
                    self.assertEqual(len(display.math), 3 if expression == E("x+y+z") else 1)
                    for attrs, ancestors in display.math:
                        self.assertNotIn("overflow", attrs.get("style", ""))
                        self.assertNotIn("max-width", attrs.get("style", ""))
                        self.assertTrue(any(
                            tag in ("span", "div") and "overflow-x:auto" in attrs.get("style", "")
                            for tag, attrs in ancestors
                        ))

    def test_solution_cells_leave_scrolling_to_the_result_container(self):
        x, y = S('solution_scroll::x', 'solution_scroll::y')
        solutions = Expression.solve([x**2 + y**2 - 25, x - 2*y], [x, y])
        self.assertEqual(len(solutions), 2)
        for result in [solutions, *solutions]:
            html = result._repr_html_()
            self.assertEqual(html.count('overflow-x:auto'), 1, html)
            for attrs, ancestors in MathDisplay(html).math:
                scrollers = [(tag, attrs) for tag, attrs in ancestors
                             if 'overflow-x:auto' in attrs.get('style', '')]
                self.assertEqual(len(scrollers), 1, html)
                self.assertEqual(scrollers[0][0], 'div', html)
                self.assertNotIn('overflow', attrs.get('style', ''))
        polynomial = E('(1+solution_scroll::x)^100').expand()
        original = bytes(polynomial.__getstate__())
        large = Expression.solve([y - polynomial], [y])
        html = large._repr_html_()
        self.assertEqual(large[0][y], polynomial)
        self.assertEqual(bytes(polynomial.__getstate__()), original)
        self.assertEqual(html.count('overflow-x:auto'), 1, html)
        self.assertIn('symbolica-math-flow', html)
        self.assertIn('data-symbolica-marker', html)
        self.assertIn('1–25 of 101', html)
        self.assertLess(len(MathDisplay(html).math), 30)

    def test_mathml_fragments_stay_embeddable_in_math(self):
        fragment = E("sin(1/x)").format(mode=PrintMode.MathML)
        self.assertIn("<mfrac>", fragment)
        self.assertNotIn("<span", fragment)
        self.assertNotIn("<div", fragment)
        self.assertNotIn("<math", fragment)


if __name__ == "__main__":
    unittest.main()
