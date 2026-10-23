"""MathML for namespaced symbols, polynomial domains and truncated series."""
import sys
import unittest
from xml.etree import ElementTree as ET

from symbolica import E, S, P, PrintMode, RationalPolynomial, FiniteFieldRationalPolynomial
from test_mathml_semantics import mathml_expression


def root(html):
    tree = ET.fromstring(html)
    flow = next((n for n in tree.iter() if n.get('class') == 'symbolica-math-flow'), None)
    if flow is not None:
        # Recombine the HTML flow's independent math roots for the same
        # semantic assertions used for a single MathML sum.
        math = ET.Element('math')
        row = ET.SubElement(math, 'mrow')
        for term in flow[-1]:
            term.tag = 'mrow'
            row.append(term)
        return math
    return next(n for n in tree.iter() if kind(n) == 'math')


def kind(node):
    return node.tag.rsplit('}', 1)[-1]


def fragment(node):
    # Strip the namespace URI when passing to the independent semantic reader.
    for child in node.iter():
        child.tag = kind(child)
    return ET.tostring(node, encoding='unicode')


class AlgebraicMathMLTests(unittest.TestCase):
    def test_namespaces_are_upright_subscripts(self):
        expr = E('physics::x + geometry::x + physics::f(geometry::x)^2')
        tree = ET.fromstring('<math>' + expr.format(mode=PrintMode.MathML, show_namespaces=True) + '</math>')
        subscripts = [n for n in tree.iter() if kind(n) == 'msub']
        self.assertEqual({n.attrib['title'] for n in subscripts}, {'physics::x', 'geometry::x', 'physics::f'})
        for sub in subscripts:
            self.assertEqual(kind(sub[0]), 'mi')
            self.assertEqual(kind(sub[1]), 'mtext')
            self.assertNotIn('::', sub[0].text)
        hidden = expr.format(mode=PrintMode.MathML, show_namespaces=True, hide_namespace='physics')
        self.assertNotIn('>physics<', hidden)
        self.assertIn('>geometry<', hidden)
        self.assertNotIn('<msub', expr.format(mode=PrintMode.MathML))
        # Interpret each annotated identifier using the complete symbol name,
        # independently of the visual namespace label.
        for sub in subscripts:
            full_name = sub.attrib['title']
            sub.clear()
            sub.tag = 'mi'
            sub.text = full_name
        self.assertEqual(mathml_expression(fragment(tree)), expr)
        builtin = E('sin(physics::x)').format(mode=PrintMode.MathML, show_namespaces=True)
        self.assertNotIn('>symbolica<', builtin)

    def test_rational_polynomial_meaning(self):
        for text in ['0', '1', '-1', 'x', '-x', '2/3*x^2-7/5*y+1', '(x+y)^3', 'x^4+4*y^4']:
            with self.subTest(text=text):
                poly = E(text).to_polynomial()
                rendered = mathml_expression(fragment(root(poly._repr_html_())))
                self.assertEqual((rendered - poly.to_expression()).expand(), E('0'))
                self.assertIn('<math', poly.formatted()._repr_html_())
                self.assertNotIn('<math', poly.formatted(mathml=False)._repr_html_())
                self.assertNotIn('<math', str(poly.formatted()))
        html = E('2/3*x^2').to_polynomial()._repr_html_()
        self.assertIn('<msup', html)
        self.assertIn('<mfrac', html)
        tree = root(E('1-x+x^2').to_polynomial()._repr_html_())
        minus = [n for n in tree.iter() if kind(n) == 'mo' and n.text == '−']
        self.assertEqual(minus[0].get('form'), 'infix')

    def test_function_and_power_polynomial_variables(self):
        for text in ['f(x)^2 + 2*f(x) + 1', 'x^(1/2)+x^(3/2)']:
            poly = E(text).to_polynomial()
            rendered = mathml_expression(fragment(root(poly._repr_html_())))
            self.assertEqual((rendered - poly.to_expression()).expand(), E('0'))

    def test_rational_function_uses_a_fraction(self):
        for text in ['(x-1)/(x+2)', '-(x+1)/(x^2+2)', '(1+x*y)/(1+x+y)']:
            poly = RationalPolynomial.parse(text, ['x', 'y'])
            tree = root(poly._repr_html_())
            self.assertIn('mfrac', [kind(n) for n in tree.iter()])
            rendered = mathml_expression(fragment(tree))
            self.assertEqual(rendered, poly.to_expression())

    def test_all_finite_polynomial_domains_preserve_domain(self):
        for modulus, power in [(7, None), (2, None), (7, (2, S('a'))), (2, (3, S('a')))]:
            with self.subTest(modulus=modulus, power=power):
                poly = P('x^2+2*x+1', modulus=modulus, power=power)
                html = poly._repr_html_()
                self.assertIn('<msup', html)
                self.assertIn('mtext', [kind(n) for n in root(html).iter()])
                ET.fromstring('<math>' + poly.format(mode=PrintMode.MathML) + '</math>')
        poly = FiniteFieldRationalPolynomial.parse('(x+1)/(x+2)', ['x'], 7)
        self.assertIn('<mfrac', poly._repr_html_())
        root(poly._repr_html_())
        a = S('a')
        poly = E('x^2+a*x+1').to_polynomial().to_number_field(E('a^2-2').to_polynomial())
        self.assertIn('<msup', poly._repr_html_())
        root(poly._repr_html_())

    def test_series_terms_preserve_meaning_and_order(self):
        x = S('x')
        for text, point, order in [
            ('exp(x)', 0, 6), ('sin(x)', 0, 7), ('exp(x)', 1, 5),
            ('1/x+x+x^2', 0, 4), ('x^(1/2)+x^(3/2)', 0, 4),
            ('(a+b)*x+(a-b)*x^2', 0, 4), ('(-2-3*𝑖)*x+x^2', 0, 4),
            ('x^10', 0, 3), ('exp(x)', -2, 4), ('exp(x)', E('a+b'), 3),
        ]:
            with self.subTest(text=text, point=point):
                series = E(text).series(x, point, order)
                tree = root(series._repr_html_())
                # The order term stays together when the surrounding sum wraps.
                row = tree[0]
                order_term = row[-1]
                self.assertEqual(order_term[-3].text, '𝒪')
                order_math = ET.Element('math')
                order_math.extend(list(order_term)[-1][1:-1])
                rendered_order = mathml_expression(fragment(order_math))
                self.assertEqual((rendered_order - (x-E(str(point)))**E('/'.join(map(str, series.get_absolute_order())))).expand(), E('0'))
                known = ET.Element('math')
                children = list(row)[:-1]
                known.extend(children)
                rendered = mathml_expression(fragment(known)) if children else E('0')
                self.assertEqual((rendered - series.to_expression()).expand(), E('0'))
                self.assertIn('<math', series.formatted()._repr_html_())
                self.assertNotIn('<math', series.formatted(mathml=False)._repr_html_())
                self.assertNotIn('<math', str(series.formatted()))

    def test_algebraic_pagers_visit_every_term(self):
        x = S('x')
        objects = [E('(1+x)^40').expand().to_polynomial(),
                   E('exp(x)').series(x, 0, 40), P('+'.join(f'x^{i}' for i in range(41)), modulus=7)]
        for obj in objects:
            original = str(obj)
            viewer = obj.mathml_viewer(page_size=7)
            try:
                _, page = viewer.handle_message({'action': 'attach', 'view': 'algebraic'})
                self.assertTrue(page['markers'], (type(obj).__name__, page['start'], page['end'], page['total'], str(obj)[:120]))
                self.assertIn('data-symbolica-marker', page['html'])
                def page_value(page):
                    math = root(page['html'])
                    known = ET.Element('math')
                    for term in math[0]:
                        if kind(term) == 'mtext': continue  # coefficient domain
                        if any('data-symbolica-marker' in n.attrib or n.text == '𝒪' for n in term.iter()): continue
                        known.append(term)
                    return mathml_expression(fragment(known))
                rendered = page_value(page)
                end = page['end']
                while page['next'] is not None:
                    _, page = viewer.handle_message({'action': 'next'})
                    self.assertEqual(page['start'], end)
                    end = page['end']
                    rendered += page_value(page)
                self.assertEqual((rendered - obj.to_expression()).expand(), E('0'))
                self.assertEqual(end, page['total'])
                self.assertFalse(page['markers'])
                self.assertEqual(str(obj), original)
            finally:
                viewer.close()
        formatted = objects[0].formatted(page_size=7)
        viewer = formatted.mathml_viewer()
        self.assertEqual(viewer.page()['end'], 7)
        viewer.close()
        viewer = objects[1].formatted(page_size=7).mathml_viewer()
        self.assertEqual(viewer.page()['end'], 7)
        viewer.close()

    def test_rational_pager_exposes_numerator_and_denominator(self):
        poly = RationalPolynomial.parse('((1+x)^40)/((1+y)^35)', ['x','y'])
        viewer = poly.mathml_viewer(page_size=7)
        original = str(poly)
        try:
            _, page = viewer.handle_message({'action': 'attach', 'view': 'rational'})
            self.assertEqual([m[0] for m in page['markers']], [[0], [1]])
            for part, total in [(0,41),(1,36)]:
                viewer.handle_message({'action': 'back'})
                _, page = viewer.handle_message({'action': 'open', 'value': [part]})
                self.assertEqual(page['total'], total)
                self.assertEqual(page['end'], 7)
                self.assertEqual(page['markers'][0][0], [part])
                while page['next'] is not None:
                    _, page = viewer.handle_message({'action': 'next'})
                self.assertEqual(page['end'], total)
            self.assertEqual(str(poly), original)
        finally:
            viewer.close()

    def test_bounds_and_no_optional_imports(self):
        poly = E('(1+x)^150').expand().to_polynomial()
        self.assertIn('⋯', poly._repr_html_())
        self.assertIn('data-symbolica-flow', poly._repr_html_())
        self.assertIn('flex-wrap:wrap', poly._repr_html_())
        self.assertLess(len(poly._repr_html_()), 100_000)
        self.assertIn('⋯', E('exp(x)').series(S('x'), 0, 150)._repr_html_())
        for name in ['anywidget', 'ipywidgets', 'IPython', 'marimo', 'traitlets']:
            self.assertNotIn(name, sys.modules)

    def test_show_more_and_less_restore_compact_display(self):
        x = S('x')
        objects = [E('(1+x)^20').expand(),
                   E('(1+x)^20').expand().to_polynomial(),
                   E('exp(x)').series(x, 0, 20)]
        for obj in objects:
            original = str(obj)
            viewer = obj.mathml_viewer(page_size=7)
            try:
                compact = viewer.page()
                self.assertTrue(compact['supports_expansion'])
                self.assertFalse(compact['can_collapse'])
                self.assertEqual(compact['end'], 7)
                for end in [14, 21]:
                    _, page = viewer.handle_message({'action': 'expand', 'value': [], 'size': 7})
                    self.assertEqual(page['end'], end)
                    self.assertTrue(page['can_collapse'])
                # The final omission disappears; the main view must still offer
                # Show less, without requiring a marker to open a popup.
                self.assertFalse(page['markers'])
                _, restored = viewer.handle_message({'action': 'collapse'})
                self.assertEqual(restored['html'], compact['html'])
                self.assertEqual(restored['end'], 7)
                self.assertFalse(restored['can_collapse'])
                self.assertEqual(str(obj), original)
            finally:
                viewer.close()

    def test_rational_parts_expand_independently_and_collapse_together(self):
        poly = RationalPolynomial.parse('(1+x)^40/(1+y)^35', ['x', 'y'])
        viewer = poly.mathml_viewer(page_size=7)
        def counts(page):
            fraction = next(n for n in root(page['html']).iter() if kind(n) == 'mfrac')
            return [sum(not any('data-symbolica-marker' in n.attrib for n in term.iter())
                        for term in part) for part in fraction]
        try:
            compact = viewer.page()
            self.assertEqual(counts(compact), [7, 7])
            _, page = viewer.handle_message({'action': 'expand', 'value': [0], 'size': 7})
            self.assertEqual(counts(page), [14, 7])
            _, inspected = viewer.handle_message({'action': 'open', 'value': [0]})
            self.assertFalse(inspected['can_collapse'])
            _, page = viewer.handle_message({'action': 'back'})
            self.assertTrue(page['can_collapse'])
            self.assertEqual(counts(page), [14, 7])
            _, page = viewer.handle_message({'action': 'expand', 'value': [1], 'size': 7})
            self.assertEqual(counts(page), [14, 14])
            self.assertTrue(page['can_collapse'])
            _, page = viewer.handle_message({'action': 'collapse'})
            self.assertEqual(page['html'], compact['html'])
            self.assertFalse(page['can_collapse'])
        finally:
            viewer.close()


if __name__ == '__main__':
    unittest.main()
