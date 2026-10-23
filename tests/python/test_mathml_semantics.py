"""Reconstruct presentation MathML independently to verify algebraic meaning.

Layout rows do not imply algebraic parentheses. This matters for signs in
products, fractions, complex coefficients, and powers.
"""

from dataclasses import dataclass
import random
import unittest
from xml.etree import ElementTree

from symbolica import E, S, Expression, PrintMode

@dataclass
class DerivativeHead:
    head: Expression
    orders: list


def mathml_expression(html):
    def tag(node):
        return node.tag.rsplit('}', 1)[-1]

    def operator(item, choices):
        return isinstance(item, str) and item in choices

    def tokens(node):
        kind = tag(node)
        if kind in ('mn', 'mi', 'mtext'):
            yield E((node.text or '').replace('_', '') if kind == 'mn' else node.text)
        elif kind == 'mspace':
            pass
        elif kind == 'mo':
            text = node.text
            if text in ('(', ')', '+', '−', ',', '\u2061'):
                yield text
            elif text not in ('*', '·', '×', '\u2062'):
                raise ValueError(f'operator: {text}')
        elif kind == 'mfrac':
            assert len(node) == 2
            yield value(node[0]) / value(node[1])
        elif kind == 'msqrt':
            assert len(node) == 1
            yield value(node[0]) ** E('1/2')
        elif kind == 'msup':
            if tag(node[0]) == 'mi' and len(node[1]) and tag(node[1][0]) == 'mo' and node[1][0].text == '(':
                orders = [item for item in tokens(node[1]) if not isinstance(item, str)]
                yield DerivativeHead(E(node[0].text), orders)
            else:
                yield value(node[0]) ** value(node[1])
        elif kind in ('math', 'mrow', 'mtable', 'mtr', 'mtd'):
            for child in node:
                yield from tokens(child)
        else:
            raise ValueError(f'element: {kind}')

    def value(node):
        items = list(tokens(node))
        index = 0

        def arguments():
            nonlocal index
            assert operator(items[index], ('(',))
            index += 1
            result = []
            if not operator(items[index], (')',)):
                while True:
                    result.append(expression())
                    if not operator(items[index], (',',)):
                        break
                    index += 1
            assert operator(items[index], (')',))
            index += 1
            return result

        def factor():
            nonlocal index
            sign = 1
            while operator(items[index], ('+', '−')):
                if items[index] == '−':
                    sign = -sign
                index += 1
            if operator(items[index], ('(',)):
                group = arguments()
                assert len(group) == 1
                result = group[0]
            else:
                result = items[index]
                index += 1
            if index < len(items) and operator(items[index], ('\u2061',)):
                index += 1
                args = arguments()
                if isinstance(result, DerivativeHead):
                    result = S('symbolica::der')(*result.orders, result.head, *args)
                else:
                    result = result(*args)
            assert isinstance(result, Expression), type(result)
            return result * sign

        def product():
            nonlocal index
            result = factor()
            while index < len(items) and not operator(items[index], ('+', '−', ')', ',')):
                result *= factor()
            return result

        def expression():
            nonlocal index
            result = product()
            while index < len(items) and operator(items[index], ('+', '−')):
                operation = items[index]
                index += 1
                term = product()
                result = result + term if operation == '+' else result - term
            return result

        result = expression()
        assert index == len(items), (index, len(items))
        return result

    return value(ElementTree.fromstring(f'<math>{html}</math>'))


cases = [
    '0', '-1', '1/7', '-3/7', '123456789012345678901234567890/7',
    '1i', '-1i', '81i/64', '-81i/64', '2+3i', '-2+3i', '-2-3i', '2-3i',
    '(-2+3i)*x', '(-2-3i)*x', '(2-3i)*x', '(2+3i)*x', 'x-1i/2', '-1i*x/2',
    '(-2+3i)^x', 'x^(-2+3i)', 'x^(-2-3i)', 'x^(2-3i)',
    'x/((a+b)*(c+d))', '(a+b)/(c+d)', '(-a+b)*(x-y)/(c+d)',
    'x/(-a+b)', 'x/(-a-b)', '(x-y)/(-a-b)', '(-x-y)*(-a-b)',
    'x^(1/2)', 'x^(-1/2)', 'x^(2/3)', 'x^(-2/3)',
    '(x+y)^(-3/2)', '(x+y)^(a+b)', '(x^y)^z', 'x^(y^z)', '(-2)^x', '(2/3)^x',
    '(-x+y)^(-3/2)', 'x/(a*b+c*d)^2', '(a+b)*(c+d)/(x-y)/(x+y)',
    'sin(alpha)^2', 'sin(1/x)', 'sin(sqrt(x))', 'sin(x+y)',
    'f()', 'f(x,y,z)', 'f(x+y,(a+b)/(c+d),(-x+y)^2)', 'f(g(x),h(y,z))',
    'f(x)^2', '1/f(x,y)', 'der(1,f,x)', 'der(1,2,f,x,y)',
    '1.25*x', '-1.25*x', '(-1.25+2.5i)*x',
]
# Fixed-seed compositions exercise signs at different nesting levels.
randomizer = random.Random(1789)
leaves = ['x', 'y', 'z', '-2', '3/7', '-3/7', '1i', '-2+3i', '2-3i']
def compose(depth):
    if depth == 0:
        return randomizer.choice(leaves)
    left, right = compose(depth-1), compose(depth-1)
    return randomizer.choice([
        f'({left})+({right})', f'({left})-({right})',
        f'({left})*({right})', f'({left})/((x^2+y^2+1)*({right})^2+1)',
        f'({left})^2', f'f({left},{right})',
    ])
cases += [compose(3) for _ in range(200)]
cases += ['pi', 'exp(1)', '∞', '-∞', '⧞', '¿']
product = '*'.join(f'(x+{i})' for i in range(1, 13))
cases += [product, f'({product})/8', f'({product})/(y+1)', f'-({product})/(y+1)']

class MathMLSemanticsTests(unittest.TestCase):
    def test_native_and_paged_mathml_preserve_algebra(self):
        for text in cases:
            original = E(text)
            before = original.format()
            expected = original.expand()
            viewer = original.mathml_viewer()
            try:
                page = viewer.page()
                self.assertFalse(page["markers"], text)
                outputs = [("formatter", original.format(mode=PrintMode.MathML))]
                # Responsive alternatives can omit subtrees; every complete
                # alternative must still represent exactly the same expression.
                for candidate in [page, *page["layouts"]]:
                    if not candidate["markers"]:
                        outputs.append(("viewer", candidate["html"]))
                for kind, html in outputs:
                    with self.subTest(expression=text, output=kind, html=html):
                        self.assertEqual(mathml_expression(html).expand(), expected)
                self.assertEqual(original.format(), before)
            finally:
                viewer.close()

    def test_complex_components_use_math_operators(self):
        for text in ["-2-3i", "-2+3i", "2-3i", "2+3i", "-1i", "-81i/64",
                     "(-1.25+2.5i)*x", "(-123456789012345678901+3i)*x"]:
            with self.subTest(expression=text):
                html = E(text).format(mode=PrintMode.MathML)
                self.assertNotIn("<mtext>", html)
                self.assertIn("<mi>𝑖</mi>", html)
                if "-" in text:
                    self.assertIn("<mo>−</mo>", html)
                self.assertEqual(mathml_expression(html).expand(), E(text).expand())


if __name__ == "__main__":
    unittest.main()
