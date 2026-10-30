"""Exercise the shared viewer in real browser engines, without a native build.

Install playwright and its chromium, firefox and webkit browsers, then run:
    python tests/browser_mathml_wrapping.py
Optional *_EXECUTABLE environment variables select existing browser binaries.
SYMBOLICA_TEST_BROWSERS selects engines, comma-separated (default: all three).
Set SYMBOLICA_TEST_STATIC=1 to also check static previews from a locally built
Symbolica module (selected with PYTHONPATH).
"""

import math
import os
from pathlib import Path
import unittest

BROWSERS = os.environ.get('SYMBOLICA_TEST_BROWSERS', 'chromium,firefox,webkit').split(',')

JAVASCRIPT = (Path(__file__).resolve().parents[1]
              / "src/api/python/display/paging.js").read_text()


def static_previews():
    if os.environ.get('SYMBOLICA_TEST_STATIC') != '1':
        return []
    from symbolica import E, S
    expression = E('(1+x)^100').expand()
    return [expression._repr_html_(), expression.formatted()._repr_html_(),
            expression.mathml_viewer()._repr_html_(),
            expression.to_polynomial()._repr_html_(),
            E('exp(x)').series(S('x'), 0, 40)._repr_html_()]


# Initialize the native allocator before loading the browser driver's TLS users.
STATIC_PREVIEWS = static_previews()

def short_sum_previews():
    if os.environ.get('SYMBOLICA_TEST_STATIC') != '1':
        return []
    from symbolica import E
    pages = []
    for expression in (
        'c*(aaaa*bbbb*cccc+dddd*eeee*ffff+gggg*hhhh*iiii)',
        'c*(aaaa*bbbb*cccc*dddd+eeee*ffff*gggg*hhhh)',
    ):
        viewer = E(expression).formatted().mathml_viewer()
        try:
            pages.append(dict(viewer.page(), layouts=[], can_collapse=True))
        finally:
            viewer.close()
    return pages


SHORT_SUM_PREVIEWS = short_sum_previews()

from playwright.async_api import async_playwright


def sum_page(display="inline-flex", start=0):
    """The bounded polynomial sum emitted for E('(1+x)^100').expand()."""
    terms = []
    for i in range(start, start + 25):
        sign = '<mo form="infix">+</mo>' if i > start else ''
        power = f'<msup><mi>x</mi><mn>{i}</mn></msup>' if i else ''
        terms.append(f'<mrow style="flex:0 0 auto">{sign}'
                     f'<mrow data-symbolica-path="{i}"><mn>{math.comb(100, i)}</mn>'
                     f'{power}</mrow></mrow>')
    prefix = '<mtext>⋯</mtext><mspace width="0.5em"/>' if start else ''
    html = ('<math xmlns="http://www.w3.org/1998/Math/MathML" '
            'style="font-family:serif;font-size:21px">'
            f'<mrow style="display:{display};flex-wrap:wrap;align-items:baseline;'
            'max-width:var(--symbolica-wrap-width,min(650px,80vw));row-gap:.5em">'
            + prefix + ''.join(terms)
            + '<mrow><mo form="infix">+</mo>'
            '<mtext data-symbolica-marker="0">[1]</mtext></mrow></mrow></math>')
    return dict(html=html, start=start, end=start + 25, total=101,
                kind="Expression", unit="terms", markers=[[[], "Inspect sum [1]"]],
                marker_ids=[0], marker_starts=[start + 25], layout=0,
                previous=start > 0, next=start + 25, back=False, page_size=25,
                supports_expansion=True, can_collapse=False)


def product_page(location="root", separator="\u2062", omission=False):
    """Bounded product rows, including products on either side of a fraction."""
    def table(prefix):
        factors = []
        if location == 'coefficient':
            factors.append('<mrow data-symbolica-factor="" data-test-factor="coefficient">'
                           '<mo>−</mo><mfrac><mn>1</mn><mn>8</mn></mfrac></mrow>')
        for index in range(16):
            sign = ''
            if index or location == 'coefficient':
                sign = f'<mo>{separator}</mo>'
                if separator == '\u2062':
                    sign += '<mspace width="0.167em"/>'
            factors.append('<mrow data-symbolica-factor="" '
                           f'data-test-factor="{prefix}{index}">{sign}'
                           f'<mi>{prefix}{index}</mi></mrow>')
        rows = '<mtr><mtd style="padding:0"><mrow>' + ''.join(factors) + '</mrow></mtd></mtr>'
        if omission:
            rows += ('<mtr><mtd style="padding:0.5em 0 0"><mrow>'
                     '<mo form="infix">·</mo><mtext data-symbolica-marker="0">[1]</mtext>'
                     '</mrow></mtd></mtr>')
        return ('<mtable data-symbolica-product="" data-symbolica-row-gap="0.5em" '
                'columnalign="left" rowspacing="0">' + rows + '</mtable>')

    if location in ('root', 'coefficient'):
        body = table('x')
    elif location == 'nested':
        body = ('<mtable data-symbolica-product="" data-symbolica-row-gap="0.5em" '
                'columnalign="left" rowspacing="0"><mtr><mtd style="padding:0"><mrow>'
                '<mrow data-symbolica-factor="" data-test-factor="outer0">'
                '<mi>q</mi><mspace width="80px"/></mrow>'
                '<mrow data-symbolica-factor="" data-test-factor="outer1">'
                '<mo>&#x2062;</mo><mspace width="0.167em"/>'
                '<mrow><mi>f</mi><mo>(</mo>' + table('x') + '<mo>)</mo></mrow>'
                '</mrow></mrow></mtd></mtr></mtable>')
    else:
        numerator = table('x') if location in ('numerator', 'both') else '<mi>a</mi>'
        denominator = table('y') if location in ('denominator', 'both') else '<mi>b</mi>'
        body = f'<mfrac><mrow>{numerator}</mrow><mrow>{denominator}</mrow></mfrac>'
    sign = '' if location == 'coefficient' else '<mo>−</mo>'
    return dict(html='<math displaystyle="true" style="font-size:21px">'
                '<mrow>' + sign + body + '</mrow></math>',
                start=0, end=16, total=20 if omission else 16,
                kind="Product", unit="factors", layout=0,
                markers=[[[], "Inspect product [1]"]] if omission else [],
                marker_ids=[0] if omission else [], marker_starts=[16] if omission else [],
                previous=False, next=16 if omission else None, back=False, page_size=25,
                supports_expansion=True, can_collapse=False)


def fraction_page(denominator='number', short=False, coefficient=False):
    """Canonical product fraction, before the browser chooses a narrow layout."""
    def table(parts, role):
        factors = ''.join(
            f'<mrow data-symbolica-factor="" data-test-{role}="{index}">'
            + ('' if index == 0 else '<mo>&#x2062;</mo><mspace width="0.167em"/>')
            + part + '</mrow>' for index, part in enumerate(parts))
        if coefficient and role == 'numerator':
            factors = factors.replace('data-symbolica-factor=""',
                                      'data-symbolica-factor="" data-symbolica-coefficient=""', 1)
        return ('<mrow><mtable data-symbolica-product="" data-symbolica-row-gap="0.5em" '
                f'data-test-table="{role}" columnalign="left" rowspacing="0">'
                '<mtr><mtd style="padding:0"><mrow>' + factors
                + '</mrow></mtd></mtr></mtable></mrow>')

    numerator = ([f'<mrow><mo>(</mo><mi>x</mi><mo>+</mo><mn>{value}</mn><mo>)</mo></mrow>'
                  for value in (6, 5)] if short else
                 [f'<msub><mi>a</mi><mn>{index}</mn></msub>' for index in range(16)])
    if coefficient:
        if coefficient == 'large':
            # Each original factor fits alone, but the proposed compact
            # coefficient and first remaining factor would overflow together.
            numerator[0] = '<mi>momentum</mi>'
        numerator.insert(0, '<mn>12345678901</mn>' if coefficient == 'large' else '<mn>3</mn>')
    denominators = {'number': ['<mn>8</mn>'], 'symbol': ['<mi>x</mi>'],
                    'power': ['<msup><mi>x</mi><mn>2</mn></msup>'],
                    'large': [f'<msub><mi>b</mi><mn>{index}</mn></msub>' for index in range(8)],
                    'ratio': ['<mn>12345</mn>']}
    html = ('<math displaystyle="true" style="font-size:21px"><mrow><mo>−</mo>'
            '<mfrac data-symbolica-product-fraction="">' + table(numerator, 'numerator')
            + table(denominators[denominator], 'denominator') + '</mfrac></mrow></math>')
    return dict(html=html, start=0, end=len(numerator), total=len(numerator),
                kind='Product', unit='factors', layout=0, markers=[], marker_ids=[], marker_starts=[],
                previous=False, next=None, back=False, page_size=25,
                supports_expansion=True, can_collapse=False)


class MathMLWrappingTests(unittest.IsolatedAsyncioTestCase):
    @unittest.skipUnless(SHORT_SUM_PREVIEWS, 'Set SYMBOLICA_TEST_STATIC=1 for native previews')
    async def test_short_wide_sums_wrap_and_stretch_their_fences(self):
        async with async_playwright() as playwright:
            for engine in BROWSERS:
                with self.subTest(engine=engine):
                    options = {'headless': True}
                    if executable := os.environ.get(f'{engine.upper()}_EXECUTABLE'):
                        options['executable_path'] = executable
                    browser = await getattr(playwright, engine).launch(**options)
                    try:
                        for fixture in SHORT_SUM_PREVIEWS:
                            page = await browser.new_page(viewport={'width': 1100, 'height': 900})
                            font = os.environ.get('SYMBOLICA_TEST_FONT')
                            await page.route('https://cdn.jsdelivr.net/**', lambda route:
                                             route.fulfill(path=font, content_type='font/woff2') if font else route.abort())
                            await page.set_content('<div id="host" style="width:900px"></div>')
                            await page.evaluate('''async ({source,fixture}) => {
                              const url=URL.createObjectURL(new Blob([source],{type:'text/javascript'}));
                              const {mount}=await import(url);URL.revokeObjectURL(url);
                              window.viewer=mount({el:host,page:fixture,send:()=>{}});
                            }''', {'source': JAVASCRIPT, 'fixture': fixture})
                            await page.evaluate('document.fonts.ready')
                            wide = None
                            for width in (900, 350, 900):
                                await page.locator('#host').evaluate('(el,w)=>el.style.width=`${w}px`', width)
                                await self.settle(page)
                                state = await page.locator('#host').evaluate('''host => {
                                  const root=host.shadowRoot, content=root.querySelector('.math-content');
                                  const table=content.querySelector('mtable[data-symbolica-sum]');
                                  const fence=table.closest('mrow[data-symbolica-path]').parentElement;
                                  return {rows:table.children.length, text:table.textContent,
                                    height:table.getBoundingClientRect().height,
                                    fence:fence.firstElementChild.getBoundingClientRect().height,
                                    scroll:content.scrollWidth, width:root.querySelector('.math').clientWidth};
                                }''')
                                self.assertLessEqual(state['scroll'], state['width'] + 1, state)
                                if width == 900:
                                    self.assertEqual(state['rows'], 1, state)
                                    if wide is not None:
                                        self.assertEqual(state, wide)
                                    wide = state
                                else:
                                    self.assertGreater(state['rows'], 1, state)
                                    self.assertGreater(state['fence'], wide['fence'], state)
                                    self.assertGreaterEqual(state['fence'], state['height'] - 2, state)
                                    self.assertEqual(state['text'], wide['text'])
                            await page.close()
                    finally:
                        await browser.close()

    async def test_product_fraction_layout_responds_to_available_width(self):
        async with async_playwright() as playwright:
            for engine in BROWSERS:
                with self.subTest(engine=engine):
                    options = {'headless': True}
                    executable = os.environ.get(f'{engine.upper()}_EXECUTABLE')
                    if executable:
                        options['executable_path'] = executable
                    browser = await getattr(playwright, engine).launch(**options)
                    try:
                        await self.check_responsive_product_fractions(browser)
                    finally:
                        await browser.close()

    async def check_responsive_product_fractions(self, browser):
        page = await browser.new_page(viewport={'width': 1200, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        font = os.environ.get('SYMBOLICA_TEST_FONT')
        await page.route('https://cdn.jsdelivr.net/**', lambda route:
                         route.fulfill(path=font, content_type='font/woff2') if font else route.abort())
        await page.set_content('<div id="host" style="width:900px"></div>')
        await page.evaluate('''async source => {
          const url = URL.createObjectURL(new Blob([source], {type:'text/javascript'}));
          window.mount = (await import(url)).mount;
          URL.revokeObjectURL(url);
        }''', JAVASCRIPT)
        cases = [('number', False, 260, False), ('symbol', False, 260, False),
                 ('power', False, 260, False), ('large', False, 260, False),
                 ('symbol', True, 260, False), ('ratio', True, 145, False),
                 ('number', False, 260, True), ('number', False, 190, 'large')]
        for denominator, short, narrow, coefficient in cases:
            with self.subTest(denominator=denominator, short=short, coefficient=coefficient):
                fixture = fraction_page(denominator, short, coefficient)
                await page.evaluate('''fixture => {
                  window.viewer?.close();
                  window.fixture = fixture;
                  window.originalPayload = JSON.stringify(fixture);
                  window.viewer = mount({el:document.querySelector('#host'),page:fixture,send:()=>{}});
                }''', fixture)
                await page.evaluate('document.fonts.ready')
                original = None
                for width in (900, narrow, 900, narrow, 900):
                    await page.locator('#host').evaluate('(host,width)=>host.style.width=`${width}px`', width)
                    await self.settle(page)
                    states = []
                    for _ in range(3):
                        states.append(await page.locator('#host').evaluate('''host => {
                          const root = host.shadowRoot;
                          const content = root.querySelector('.math-content');
                          const table = content.querySelector('[data-test-table="numerator"]');
                          const ownRows = [...table.children].filter(row =>
                            [...row.querySelectorAll('[data-test-numerator]')].some(n=>n.closest('mtable')===table));
                          const fraction = content.querySelector('mfrac');
                          const sign = [...content.querySelectorAll('mo')].filter(n=>n.textContent==='−');
                          return {
                            compact:!!content.querySelector('[data-symbolica-product-prefactor]'),
                            fractions:content.querySelectorAll('mfrac').length,
                            fractionWidth:fraction.getBoundingClientRect().width,
                            fractionNumerator:fraction.children[0].textContent,
                            fractionDenominator:fraction.children[1].textContent,
                            numerator:[...content.querySelectorAll('[data-test-numerator]')].map(n=>n.dataset.testNumerator),
                            denominator:[...content.querySelectorAll('[data-test-denominator]')].map(n=>n.textContent),
                            rows:ownRows.length,
                            continuations:ownRows.map(row=>[...row.querySelectorAll('[data-symbolica-continuation]')]
                              .filter(n=>n.closest('mtable')===table).map(n=>n.textContent)),
                            signs:sign.length,
                            signOnFirstRow:sign[0]?.closest('mtr')===ownRows[0],
                            viewport:root.querySelector('.math').clientWidth,
                            scroll:content.scrollWidth,
                            unchanged:JSON.stringify(fixture)===originalPayload
                          };
                        }'''))
                        await self.settle(page)
                    self.assertEqual(states[0], states[1], states)
                    self.assertEqual(states[1], states[2], states)
                    state = states[0]
                    self.assertTrue(state['unchanged'], state)
                    self.assertEqual(state['signs'], 1, state)
                    self.assertEqual(state['fractions'], 1, state)
                    self.assertEqual(sorted(state['numerator'], key=int),
                                     [str(i) for i in range((2 if short else 16) + int(bool(coefficient)))], state)
                    self.assertEqual([i for i in state['numerator'] if not coefficient or i != '0'],
                                     [str(i) for i in range(int(bool(coefficient)),
                                                           (2 if short else 16) + int(bool(coefficient)))], state)
                    self.assertLessEqual(state['scroll'], state['viewport'] + 2, state)
                    compact = (width == narrow and not short and denominator != 'large'
                               and coefficient != 'large')
                    self.assertEqual(state['compact'], compact, state)
                    if compact:
                        self.assertEqual(state['fractionNumerator'], '3' if coefficient else '1', state)
                        self.assertTrue(state['signOnFirstRow'], state)
                        self.assertLess(state['fractionWidth'], 21 * 4, state)
                    if width == 900:
                        self.assertEqual(state['rows'], 1, state)
                        if original is None:
                            original = state
                        else:
                            self.assertEqual(state, original)
                    elif not short or denominator == 'ratio':
                        self.assertGreater(state['rows'], 1, state)
                    else:
                        self.assertEqual(state['rows'], 1, state)
                    self.assertEqual(state['continuations'], [[]] + [['×']] * (state['rows'] - 1), state)
                    self.assertEqual(state['denominator'], original['denominator'], state)
        await page.evaluate('viewer.close()')
        await page.close()
        self.assertFalse(errors, errors)

    async def test_product_continuations_resize_in_each_engine(self):
        async with async_playwright() as playwright:
            for engine in BROWSERS:
                with self.subTest(engine=engine):
                    options = {"headless": True}
                    executable = os.environ.get(f"{engine.upper()}_EXECUTABLE")
                    if executable:
                        options["executable_path"] = executable
                    browser = await getattr(playwright, engine).launch(**options)
                    try:
                        await self.check_product_continuations(browser)
                    finally:
                        await browser.close()

    async def check_product_continuations(self, browser):
        page = await browser.new_page(viewport={"width": 1200, "height": 1000})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        font = os.environ.get('SYMBOLICA_TEST_FONT')
        await page.route("https://cdn.jsdelivr.net/**", lambda route:
                         route.fulfill(path=font, content_type='font/woff2') if font else route.abort())
        await page.set_content('<div id="host" style="width:900px"></div>')
        await page.evaluate("""async source => {
          const url = URL.createObjectURL(new Blob([source], {type:'text/javascript'}));
          window.mount = (await import(url)).mount;
          URL.revokeObjectURL(url);
        }""", JAVASCRIPT)
        cases = [(location, '\u2062', False)
                 for location in ('root', 'numerator', 'denominator', 'both', 'nested', 'coefficient')]
        cases.extend([('root', '×', False), ('root', '·', False),
                      ('root', '\u2062', True), ('root', '·', True)])
        for location, separator, omission in cases:
            with self.subTest(location=location, separator=separator, omission=omission):
                fixture = product_page(location, separator, omission)
                await page.evaluate("""fixture => {
                  window.viewer?.close();
                  window.fixture = fixture;
                  window.originalPayload = JSON.stringify(fixture);
                  window.viewer = mount({el:document.querySelector('#host'),
                    page:fixture, send:()=>{}});
                }""", fixture)
                await page.evaluate('document.fonts.ready')
                for width, action in ((900, None), (190, None), (190, '='),
                                      (190, '-'), (190, 'font'), (900, None)):
                    await page.locator('#host').evaluate('(host,width)=>host.style.width=`${width}px`', width)
                    if action == 'font':
                        await page.evaluate("""() => {
                          const event = new Event('loadingdone');
                          Object.defineProperty(event,'fontfaces',{value:[{family:'STIX Two Math'}]});
                          document.fonts.dispatchEvent(event);
                        }""")
                    elif action:
                        await page.locator('.math').press(action)
                    await self.settle(page)
                    measured = await page.locator('#host').evaluate("""host => {
                      const root = host.shadowRoot;
                      const content = root.querySelector('.math-content');
                      const visible = node => node.getBoundingClientRect().width > .01
                        && getComputedStyle(node).visibility !== 'hidden';
                      const coefficient = content.querySelector('[data-test-factor="coefficient"]');
                      const coefficientFraction = coefficient?.querySelector('mfrac');
                      return {
                        viewport: root.querySelector('.math').clientWidth,
                        scroll: content.scrollWidth,
                        unchanged: JSON.stringify(fixture) === originalPayload,
                        fractionChildren: [...content.querySelectorAll('mfrac')].map(n=>n.children.length),
                        signs: [...content.querySelectorAll('mo')].filter(n=>n.textContent==='−').length,
                        coefficient: coefficient && {
                          width:coefficientFraction.getBoundingClientRect().width,
                          font:parseFloat(getComputedStyle(coefficient).fontSize),
                          signOnFirstRow:coefficient.querySelector('mo').closest('mtr')
                            === content.querySelector('mtable[data-symbolica-product] > mtr'),
                          firstFactor:content.querySelector('[data-test-factor]')===coefficient
                        },
                        tables: [...content.querySelectorAll('mtable[data-symbolica-product]')].map(table=>({
                          factors: [...table.querySelectorAll('[data-test-factor]')]
                            .filter(n=>n.closest('mtable[data-symbolica-product]')===table).map(n=>n.dataset.testFactor),
                          rows: [...table.children].map(row=>{
                            const cell = row.firstElementChild;
                            const factors = [...row.querySelectorAll('[data-test-factor]')]
                              .filter(n=>n.closest('mtable[data-symbolica-product]')===table);
                            const token = factors[0]?.querySelector('mi,mfrac');
                            const operators = [...row.querySelectorAll('mo')].filter(visible);
                            const leading = token ? operators.filter(n=>
                              n.compareDocumentPosition(token) & Node.DOCUMENT_POSITION_FOLLOWING) : [];
                            const cross = leading.find(n=>n.textContent==='×');
                            return {count:factors.length, operators:operators.map(n=>n.textContent),
                              leading:leading.map(n=>n.textContent),
                              indent:cross ? cross.getBoundingClientRect().left-cell.getBoundingClientRect().left : 0,
                              font:parseFloat(getComputedStyle(cell).fontSize),
                              buttons:row.querySelectorAll('button.omission').length};
                          })
                        }))
                      };
                    }""")
                    self.assertTrue(measured['unchanged'], measured)
                    self.assertEqual(measured['signs'], 1, measured)
                    self.assertEqual(measured['fractionChildren'], [] if location in ('root', 'nested') else [2])
                    self.assertLessEqual(measured['scroll'], measured['viewport'] + 2, measured)
                    if location == 'coefficient':
                        self.assertTrue(measured['coefficient']['signOnFirstRow'], measured)
                        self.assertTrue(measured['coefficient']['firstFactor'], measured)
                        self.assertLess(measured['coefficient']['width'],
                                        measured['coefficient']['font'] * 2, measured)
                    prefixes = ('x', 'y') if location == 'both' else ('y',) if location == 'denominator' else ('x',)
                    expected = [[f'{prefix}{i}' for i in range(16)] for prefix in prefixes]
                    if location == 'nested':
                        expected.insert(0, ['outer0', 'outer1'])
                    elif location == 'coefficient':
                        expected[0].insert(0, 'coefficient')
                    self.assertEqual(len(measured['tables']), len(expected))
                    for table, factors in zip(measured['tables'], expected):
                        self.assertEqual(table['factors'], factors)
                        rows = [row for row in table['rows'] if row['count']]
                        if width == 900:
                            self.assertEqual(len(rows), 1, measured)
                        else:
                            self.assertGreater(len(rows), 1, measured)
                        for index, row in enumerate(rows):
                            leading = ['×'] if index else ['−'] if location == 'coefficient' else []
                            self.assertEqual(row['leading'], leading, measured)
                            if index:
                                self.assertGreaterEqual(row['indent'], row['font'] * .5, measured)
                            if separator != '\u2062':
                                expected = ([separator] * (row['count'] - 1))
                                if index:
                                    expected.insert(0, '×')
                                self.assertEqual(row['operators'], expected, measured)
                        tails = [row for row in table['rows'] if not row['count']]
                        self.assertEqual(len(tails), int(omission), measured)
                        if omission:
                            self.assertEqual(tails[0]['operators'], ['·'], measured)
                            self.assertEqual(tails[0]['buttons'], 1, measured)

        # Sum rows keep their ordinary additive signs during the same resizing.
        terms = ''.join('<mrow data-symbolica-term="">' + ('<mo>+</mo>' if i else '')
                        + f'<mi>s{i}</mi></mrow>' for i in range(16))
        fixture = dict(product_page(), kind='Sum', unit='terms', html=(
            '<math><mtable data-symbolica-sum="" data-symbolica-row-gap="0.5em">'
            '<mtr><mtd><mrow>' + terms + '</mrow></mtd></mtr></mtable></math>'))
        await page.evaluate('fixture => viewer.update(["page",fixture])', fixture)
        for width in (900, 190, 900):
            await page.locator('#host').evaluate('(host,width)=>host.style.width=`${width}px`', width)
            await self.settle(page)
            self.assertEqual(await page.locator('.math-content mi').all_text_contents(),
                             [f's{i}' for i in range(16)])
            self.assertEqual(await page.locator('.math-content mo').all_text_contents(), ['+'] * 15)
        await page.evaluate('viewer.close()')
        await page.close()
        self.assertFalse(errors, errors)

    async def test_composite_cell_overflow_selects_a_compact_page(self):
        async with async_playwright() as playwright:
            for engine in BROWSERS:
                with self.subTest(engine=engine):
                    options = {"headless": True}
                    executable = os.environ.get(f"{engine.upper()}_EXECUTABLE")
                    if executable:
                        options["executable_path"] = executable
                    browser = await getattr(playwright, engine).launch(**options)
                    try:
                        page = await browser.new_page()
                        await page.route("https://cdn.jsdelivr.net/**", lambda route: route.abort())
                        await page.set_content('<div id="host" style="width:480px"></div>')
                        html = ('<div style="width:var(--symbolica-wrap-width)"><table '
                                'style="width:100%;table-layout:fixed"><tr><td>'
                                # A zero-height box does not contribute scroll overflow in Chromium.
                                '<div data-symbolica-cell=""><math><mspace width="300px" height="20px"/></math></div>'
                                '</td><td><div data-symbolica-cell=""><math><mi>x</mi></math></div>'
                                '</td></tr></table></div>')
                        fixture = dict(html=html, start=0, end=2, total=2,
                                       kind="Matrix", unit="entries", markers=[], layout=0,
                                       previous=False, next=None, back=False, page_size=25,
                                       layouts=[dict(html=html.replace('300px', '80px'), layout=1)])
                        await page.evaluate("""async ({source, fixture}) => {
                          const url = URL.createObjectURL(new Blob([source], {type:'text/javascript'}));
                          const {mount} = await import(url);
                          URL.revokeObjectURL(url);
                          mount({el:document.querySelector('#host'),page:fixture,send:()=>{}});
                        }""", {"source": JAVASCRIPT, "fixture": fixture})
                        await page.wait_for_function("""() => document.querySelector('#host')
                          .shadowRoot.querySelector('.math-content').dataset.layout === '1'""")
                        self.assertTrue(await page.locator('#host').evaluate("""host =>
                          [...host.shadowRoot.querySelectorAll('[data-symbolica-cell]')]
                            .every(cell => cell.scrollWidth <= cell.clientWidth + 1)"""))
                    finally:
                        await browser.close()

    async def test_fractional_cell_width_does_not_wrap_a_fitting_sum(self):
        async with async_playwright() as playwright:
            for engine in BROWSERS:
                with self.subTest(engine=engine):
                    options = {"headless": True}
                    executable = os.environ.get(f"{engine.upper()}_EXECUTABLE")
                    if executable:
                        options["executable_path"] = executable
                    browser = await getattr(playwright, engine).launch(**options)
                    try:
                        page = await browser.new_page()
                        await page.route("https://cdn.jsdelivr.net/**", lambda route: route.abort())
                        await page.set_content('<div id="host" style="width:300px"></div>')
                        html = ('<div data-symbolica-cell="" style="width:70.125px">'
                                '<math><mrow style="display:flex;flex-wrap:wrap">'
                                '<mrow><mspace width="35.25px" height="10px"/></mrow>'
                                '<mrow><mspace width="34.875px" height="10px"/></mrow>'
                                '</mrow></math></div>')
                        fixture = dict(html=html, start=0, end=1, total=1,
                                       kind="Matrix", unit="rows", markers=[], layout=0,
                                       previous=False, next=None, back=False, page_size=25)
                        await page.evaluate("""async ({source, fixture}) => {
                          const url = URL.createObjectURL(new Blob([source], {type:'text/javascript'}));
                          const {mount} = await import(url);
                          URL.revokeObjectURL(url);
                          mount({el:document.querySelector('#host'),page:fixture,send:()=>{}});
                        }""", {"source": JAVASCRIPT, "fixture": fixture})
                        await page.wait_for_function("""() => document.querySelector('#host')
                          .shadowRoot.querySelectorAll('.math-flow > math').length === 2""")
                        for _ in range(2):
                            bottoms = await page.locator('#host').evaluate("""host =>
                              [...host.shadowRoot.querySelectorAll('.math-flow > math')]
                                .map(term => term.getBoundingClientRect().bottom)""")
                            self.assertLess(max(bottoms) - min(bottoms), 1, bottoms)
                            await page.locator('#host').locator('.math').press('=')
                    finally:
                        await browser.close()

    async def test_product_cell_padding_survives_measured_repacking(self):
        async with async_playwright() as playwright:
            for engine in BROWSERS:
                with self.subTest(engine=engine):
                    options = {"headless": True}
                    executable = os.environ.get(f"{engine.upper()}_EXECUTABLE")
                    if executable:
                        options["executable_path"] = executable
                    browser = await getattr(playwright, engine).launch(**options)
                    try:
                        page = await browser.new_page()
                        await page.route("https://cdn.jsdelivr.net/**", lambda route: route.abort())
                        await page.set_content('<div id="host" style="width:260px"></div>')
                        factors = ''.join(f'<mrow data-symbolica-factor=""><mi>x{i}</mi></mrow>'
                                          for i in range(20))
                        html = ('<math><mrow><mo>−</mo><mtable data-symbolica-product="" '
                                'data-symbolica-row-gap="0.5em" rowspacing="0">'
                                '<mtr><mtd style="padding:0"><mrow>' + factors
                                + '</mrow></mtd></mtr></mtable></mrow></math>')
                        fixture = dict(html=html, start=0, end=20, total=20,
                                       kind="Product", unit="factors", markers=[], layout=0,
                                       previous=False, next=None, back=False, page_size=25)
                        await page.evaluate("""async ({source, fixture}) => {
                          const url = URL.createObjectURL(new Blob([source], {type:'text/javascript'}));
                          const {mount} = await import(url);
                          URL.revokeObjectURL(url);
                          mount({el:document.querySelector('#host'),page:fixture,send:()=>{}});
                        }""", {"source": JAVASCRIPT, "fixture": fixture})
                        await page.wait_for_function("""() => document.querySelector('#host')
                          .shadowRoot.querySelectorAll('mtable mtr').length > 1""")
                        padding = await page.locator('#host').evaluate("""host => {
                          const table = host.shadowRoot.querySelector('mtable');
                          return [...table.querySelectorAll('mtd')].map(cell => {
                            const s = getComputedStyle(cell);
                            return [s.paddingTop,s.paddingRight,s.paddingBottom,s.paddingLeft].map(parseFloat);
                          });
                        }""")
                        self.assertEqual(padding[0], [0, 0, 0, 0], padding)
                        for top, right, bottom, left in padding[1:]:
                            self.assertGreater(top, 0, padding)
                            self.assertEqual([right, bottom, left], [0, 0, 0], padding)
                    finally:
                        await browser.close()

    async def test_root_sums_resize_and_preview_in_each_engine(self):
        async with async_playwright() as playwright:
            for engine in BROWSERS:
                with self.subTest(engine=engine):
                    options = {"headless": True}
                    executable = os.environ.get(f"{engine.upper()}_EXECUTABLE")
                    if executable:
                        options["executable_path"] = executable
                    browser = await getattr(playwright, engine).launch(**options)
                    try:
                        await self.check_browser(browser)
                    finally:
                        await browser.close()

    async def check_browser(self, browser):
        page = await browser.new_page(viewport={"width": 1400, "height": 1100})
        page.set_default_timeout(10_000)
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        await page.route("https://cdn.jsdelivr.net/**", lambda route: route.abort())
        await page.set_content('<div id="host" style="width:900px"></div>')
        await page.evaluate("""async source => {
          const url = URL.createObjectURL(new Blob([source], {type:'text/javascript'}));
          window.mount = (await import(url)).mount;
          URL.revokeObjectURL(url);
        }""", JAVASCRIPT)

        # Algebraic containers use display:flex; expressions use inline-flex.
        for display in ("inline-flex", "flex"):
            fixture = sum_page(display)
            if display == "flex":
                # Frontend sanitizers may strip inline CSS variable references.
                fixture['html'] = fixture['html'].replace(
                    'max-width:var(--symbolica-wrap-width,min(650px,80vw));', '')
            await page.evaluate("""async fixture => {
              await window.viewer?.close();
              window.fixture = fixture;
              window.originalPayload = JSON.stringify(fixture);
              window.sent = [];
              const template = document.createElement('template');
              template.innerHTML = fixture.html;
              window.originalText = template.content.textContent;
              window.originalNodes = template.content.querySelectorAll('*').length;
              document.querySelector('#host').style.width = '900px';
              window.viewer = mount({el:document.querySelector('#host'), page:fixture,
                send: message => {
                  sent.push(message);
                  if (message.action === 'preview') return ['preview', {
                    ...fixture, request:message.request, context:'popup', can_expand:true
                  }];
                }});
              window.initialRoot = document.querySelector('#host').shadowRoot
                .querySelector('.math-content').firstElementChild;
            }""", fixture)
            await self.settle(page)
            wide = await self.assert_fits(page, ".math-content")
            self.assertTrue(await page.evaluate("""() => {
              const root = document.querySelector('#host').shadowRoot;
              const content = root.querySelector('.math-content');
              return content.textContent === originalText.replace('[1]', '1')
                && content.querySelectorAll('*').length <= originalNodes + 1
                && JSON.stringify(fixture) === originalPayload
                && content.firstElementChild === initialRoot
                && [...content.querySelector('.math-flow').children].every(node =>
                  node.localName === 'math'
                  && node.namespaceURI === 'http://www.w3.org/1998/Math/MathML');
            }"""))

            await page.locator('#host').evaluate("el => el.style.width = '480px'")
            await self.settle(page)
            narrow = await self.assert_fits(page, ".math-content")
            self.assertGreater(narrow, wide)
            await page.locator('.math-content button.omission').click()
            await page.wait_for_function("""() => {
              const popup = document.querySelector('#host').shadowRoot.querySelector('.preview');
              return !popup.hidden && !popup.hasAttribute('aria-busy');
            }""")
            await self.assert_fits(page, ".preview-math")
            self.assertTrue(await page.evaluate("""() =>
              sent.some(message => message.action === 'preview'
                && JSON.stringify(message.value) === '[]')"""))
            await page.keyboard.press('Escape')

            # Subsequent pages also contain a leading continuation/space.
            await page.evaluate("fixture => viewer.update(['page', fixture])",
                                sum_page(display, start=25))
            await self.settle(page)
            await self.assert_fits(page, ".math-content")
            self.assertIn('⋯', await page.locator('.math-content').text_content())
            await page.locator('#host').evaluate("el => el.style.width = '1200px'")
            await self.settle(page)
            await self.assert_fits(page, ".math-content")

        # Repacking a sum into separate math roots must keep fraction sizing.
        for wrapped in (False, True):
            fraction = '<mrow><mfrac><mi>x</mi><mn>8</mn></mfrac></mrow>'
            body = ('<mrow style="display:inline-flex;flex-wrap:wrap">'
                    + fraction + '<mrow><mo>+</mo><mi>y</mi></mrow></mrow>'
                    if wrapped else fraction)
            fixture = dict(sum_page(), markers=[], marker_ids=[], marker_starts=[],
                           next=None, html='<math displaystyle="true" '
                           'style="font-size:21px">' + body + '</math>')
            await page.evaluate("fixture => viewer.update(['page', fixture])", fixture)
            await self.settle(page)
            sizes = await page.locator('.math-content mn').evaluate("""node => ({
              denominator:getComputedStyle(node).fontSize,
              root:getComputedStyle(node.closest('math')).fontSize
            })""")
            self.assertEqual(sizes['denominator'], sizes['root'], (wrapped, sizes))

        # A nested sum used as a power base must remain one fenced math tree.
        nested = dict(sum_page(), markers=[], marker_ids=[], marker_starts=[])
        nested['html'] = ('<math xmlns="http://www.w3.org/1998/Math/MathML">'
                          '<msup><mrow><mo>(</mo><mi>x</mi><mo>+</mo><mn>1</mn>'
                          '<mo>)</mo></mrow><mn>2</mn></msup></math>')
        await page.evaluate("fixture => viewer.update(['page', fixture])", nested)
        await self.settle(page)
        self.assertEqual(await page.locator('.math-content math').count(), 1)
        self.assertEqual(await page.locator('.math-content msup > mrow > mo').all_text_contents(),
                         ['(', '+', ')'])
        self.assertEqual(await page.locator('.math-content .math-flow').count(), 0)
        # If no candidate fits, the earlier smallest DOM must be restored. If
        # a later candidate fits, keep that candidate and its layout capability.
        candidates = dict(sum_page(), markers=[], marker_ids=[], marker_starts=[], next=None)
        def oversized(count):
            return '<math><mrow><mi>' + 'W' * count + '</mi></mrow></math>'
        candidates['html'] = oversized(100)
        candidates['layouts'] = [dict(html=oversized(200), layout=1)]
        await page.evaluate("fixture => viewer.update(['page', fixture])", candidates)
        self.assertEqual(await page.locator('.math-content mi').text_content(), 'W' * 100)
        self.assertEqual(await page.locator('.math-content').get_attribute('data-layout'), '0')
        candidates['layouts'] = [dict(html='<math><mi>x</mi></math>', layout=1)]
        await page.evaluate("fixture => viewer.update(['page', fixture])", candidates)
        self.assertEqual(await page.locator('.math-content mi').text_content(), 'x')
        self.assertEqual(await page.locator('.math-content').get_attribute('data-layout'), '1')
        await page.evaluate("() => viewer.close()")
        await self.check_popup_lifetime(page)
        await self.check_hover_refitting(page)
        for html in STATIC_PREVIEWS:
            await page.set_content('<div id="static" style="width:900px">' + html + '</div>')
            for width in (900, 480):
                await page.locator('#static').evaluate('(el, width) => el.style.width = `${width}px`', width)
                result = await page.locator('.symbolica-math-flow > div').evaluate("""flow => ({
                  width:flow.clientWidth, scroll:flow.scrollWidth,
                  first:flow.firstElementChild.getBoundingClientRect().bottom,
                  last:flow.lastElementChild.getBoundingClientRect().top,
                  namespaces:[...flow.children].every(n =>
                    n.namespaceURI === 'http://www.w3.org/1998/Math/MathML')
                })""")
                self.assertLessEqual(result['scroll'], width + 2, result)
                self.assertGreater(result['last'], result['first'], result)
                self.assertTrue(result['namespaces'])
        self.assertFalse(errors, errors)

    async def check_popup_lifetime(self, page):
        await page.set_viewport_size({'width': 1400, 'height': 1600})
        await page.evaluate("""fixture => {
          const host = document.querySelector('#host');
          host.style.cssText = 'width:900px;margin-top:200px';
          document.body.style.minHeight = '3000px';
          window.sent = [];
          window.viewer = mount({el:host, page:fixture, send:message => {
            sent.push(message);
            if (message.action === 'preview') return ['preview', {
              ...fixture, request:message.request, context:'popup', can_expand:true
            }];
          }});
        }""", sum_page())
        await self.settle(page)
        await page.locator('.math-content button.omission').click()
        await page.wait_for_function("""() => {
          const popup = document.querySelector('#host').shadowRoot.querySelector('.preview');
          return !popup.hidden && !popup.hasAttribute('aria-busy');
        }""")
        popup = page.locator('.preview')
        before = await popup.bounding_box()
        await page.evaluate('() => window.scrollBy(0, 120)')
        await self.settle(page)
        after = await popup.bounding_box()
        self.assertAlmostEqual(after['y'] - before['y'], -120, delta=2)

        # Unrelated document fonts do not rebuild the expression at all.
        for family in ('Unrelated UI Font', 'STIX Two Math'):
            await page.evaluate("""family => {
              const root = document.querySelector('#host').shadowRoot;
              window.beforeFont = root.querySelector('.math-content').firstElementChild;
              const event = new Event('loadingdone');
              Object.defineProperty(event, 'fontfaces', {value:[{family}]});
              document.fonts.dispatchEvent(event);
            }""", family)
            await self.settle(page)
            self.assertTrue(await popup.is_visible())
            if family == 'Unrelated UI Font':
                self.assertTrue(await page.evaluate("""() => beforeFont ===
                  document.querySelector('#host').shadowRoot.querySelector('.math-content').firstElementChild"""))

        # A resize reanchors/refits the popup even if the host has a fixed width.
        await page.set_viewport_size({'width': 660, 'height': 1200})
        await self.settle(page)
        box = await popup.bounding_box()
        self.assertGreaterEqual(box['x'], 11)
        self.assertLessEqual(box['x'] + box['width'], 649)
        await self.assert_fits(page, '.preview-math')
        # Refitting the main view replaces its marker; Escape must focus the new one.
        await page.locator('#host').evaluate("el => el.style.width = '480px'")
        await self.settle(page)
        self.assertTrue(await popup.is_visible())
        self.assertEqual(await page.evaluate("() => sent.filter(m => m.action === 'preview').length"), 1)
        await page.keyboard.press('Escape')
        self.assertFalse(await popup.is_visible())
        self.assertTrue(await page.evaluate("""() => {
          const root = document.querySelector('#host').shadowRoot;
          return root.activeElement === root.querySelector('.math-content button.omission');
        }"""))
        await page.evaluate('() => viewer.close()')

    async def check_hover_refitting(self, page):
        # The same source path can hide a different tail in each layout. An
        # unpinned hover follows that tail; a pinned popup keeps its own page.
        await page.set_viewport_size({'width': 1400, 'height': 1100})
        fixture = dict(sum_page(), kind='Product', unit='factors', total=15,
                       markers=[[[0], 'Inspect product [1]']], marker_starts=[8])
        fixture['html'] = ('<math><mrow><mspace width="650px"/>'
                           '<mtext data-symbolica-marker="0">[1]</mtext></mrow></math>')
        fixture['layouts'] = [dict(layout=1, marker_starts=[4],
                                   html=fixture['html'].replace('650px', '80px'))]
        for scenario in ('visible', 'pending', 'inflight', 'pinned'):
            with self.subTest(hover_refit=scenario):
                await page.evaluate("""async ({fixture, hold}) => {
                  await window.viewer?.close();
                  const host = document.querySelector('#host');
                  host.style.cssText = 'width:900px';
                  window.sent = [];
                  window.responses = [];
                  window.viewer = mount({el:host, page:fixture, send:message => {
                    sent.push(message);
                    if (message.action !== 'preview') return;
                    const start = message.popup ? message.start
                      : (message.layout === 1 ? 4 : 8);
                    const response = ['preview', {...fixture, layouts:[], markers:[],
                      marker_ids:[], marker_starts:[], request:message.request,
                      context:message.request, start, end:start+5, next:start+5,
                      html:`<math><mn>${start}</mn></math>`}];
                    responses.push(response);
                    return hold ? undefined : response;
                  }});
                }""", {'fixture': fixture, 'hold': scenario == 'inflight'})
                await self.settle(page)
                self.assertEqual(await page.locator('.math-content').get_attribute('data-layout'), '0')
                button = page.locator('.math-content button.omission')
                if scenario == 'pending':
                    # Resize in the focus event's turn, before the 180ms timer.
                    await button.evaluate("""button => {
                      button.focus();
                      document.querySelector('#host').style.width = '480px';
                    }""")
                else:
                    if scenario == 'pinned':
                        await button.click()
                    else:
                        await button.focus()
                    await page.wait_for_function("() => responses.length === 1")
                    if scenario != 'inflight':
                        await page.wait_for_function("""() => {
                          const popup = document.querySelector('#host').shadowRoot.querySelector('.preview');
                          return !popup.hidden && !popup.hasAttribute('aria-busy');
                        }""")
                    await page.locator('#host').evaluate("host => host.style.width = '480px'")
                await self.settle(page)
                if scenario == 'pinned':
                    self.assertTrue(await page.locator('.preview').is_visible())
                    self.assertEqual(await page.evaluate('responses.length'), 1)
                    self.assertIn('factors 9–13', await page.locator('.preview-heading').text_content())
                    continue
                expected = 1 if scenario == 'pending' else 2
                await page.wait_for_function('n => responses.length === n', arg=expected)
                if scenario == 'inflight':
                    # A response for the old layout must not overwrite the new
                    # hover even when it arrives after the replacement response.
                    await page.evaluate("() => {viewer.update(responses[1]); viewer.update(responses[0]);}")
                await page.wait_for_function("""() => {
                  const popup = document.querySelector('#host').shadowRoot.querySelector('.preview');
                  return !popup.hidden && !popup.hasAttribute('aria-busy')
                    && popup.querySelector('.preview-heading').textContent.includes('factors 5–9');
                }""")
                self.assertEqual(await button.get_attribute('data-layout'), '1')
                self.assertEqual(await page.evaluate('sent.filter(m=>m.action==="preview").at(-1).layout'), 1)
                self.assertTrue(await button.evaluate('b=>b.getRootNode().activeElement===b'))
        await page.evaluate('() => viewer.close()')

    async def settle(self, page):
        await page.evaluate("""() => new Promise(resolve =>
          requestAnimationFrame(() => requestAnimationFrame(resolve)))""")

    async def assert_fits(self, page, selector):
        measurements = await page.locator(selector).evaluate("""element => {
          const flow = element.firstElementChild;
          const terms = [...flow.children];
          return {
            width:element.clientWidth, scroll:element.scrollWidth,
            height:flow.getBoundingClientRect().height,
            firstBottom:terms[0].getBoundingClientRect().bottom,
            lastTop:terms.at(-1).getBoundingClientRect().top
          };
        }""")
        self.assertLessEqual(measurements['scroll'], measurements['width'] + 2,
                             measurements)
        self.assertGreater(measurements['lastTop'], measurements['firstBottom'],
                           measurements)
        return measurements['height']


if __name__ == "__main__":
    unittest.main()
