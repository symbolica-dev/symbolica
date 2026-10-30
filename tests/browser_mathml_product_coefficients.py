"""Browser regressions for coefficients preceding a wrapped, fenced sum.

Run with Playwright's browser engines installed, or set *_EXECUTABLE to local
browser binaries. SYMBOLICA_TEST_FONT optionally supplies the STIX WOFF2 file.
"""
import os
from pathlib import Path
import unittest

from playwright.async_api import async_playwright

JAVASCRIPT = (Path(__file__).resolve().parents[1]
              / 'src/api/python/display/paging.js').read_text()


def coefficient_page(location='root', leaf=False, large=False, separator='\u2062'):
    terms = ''.join(
        f'<mrow data-symbolica-term="" data-test-term="{index}">'
        + ('<mo>−</mo>' if index % 3 == 0 else '<mo>+</mo>' if index else '')
        + f'<mn>{index + 1}</mn><msup><mi>m</mi><mn>4</mn></msup>'
        '<msup><mi>t</mi><mn>2</mn></msup><mi>u</mi></mrow>'
        for index in range(12))
    next_factor = ('<mspace data-test-leaf="" width="100px"/>' if leaf else
                   '<mrow><mo>(</mo><mtable data-symbolica-sum="" '
                   'data-symbolica-row-gap="0.5em" columnalign="left" rowspacing="0">'
                   '<mtr><mtd style="padding:0"><mrow>' + terms
                   + '</mrow></mtd></mtr></mtable><mo>)</mo></mrow>')
    table = ('<mtable data-symbolica-product="" data-test-product="" '
             'data-symbolica-row-gap="0.5em" columnalign="left" rowspacing="0">'
             '<mtr><mtd style="padding:0"><mrow>'
             '<mrow data-symbolica-factor="" data-symbolica-coefficient="" data-test-factor="coefficient">'
             + ('<mn>12345678901</mn>' if large else '<mn>2</mn>')
             + '</mrow><mrow data-symbolica-factor="" data-test-factor="next">'
             f'<mo>{separator}</mo><mspace width="0.167em"/>' + next_factor
             + '</mrow></mrow></mtd></mtr></mtable>')
    if location == 'numerator':
        denominator = ('<mrow><msup><mrow><mo>(</mo><mi>d</mi><mo>+</mo><mi>e</mi><mo>)</mo></mrow><mn>2</mn></msup>'
                       '<mo>&#x2062;</mo><mspace width="0.167em"/>'
                       '<msup><mrow><mo>(</mo><mi>f</mi><mo>+</mo><mi>g</mi><mo>)</mo></mrow><mn>2</mn></msup></mrow>')
        table = '<mfrac data-symbolica-product-fraction=""><mrow>' + table + '</mrow>' + denominator + '</mfrac>'
    return dict(html='<math displaystyle="true" style="font-size:21px"><mrow><mo>−</mo>'
                + table + '</mrow></math>', start=0, end=2, total=2, kind='Product', unit='factors',
                layout=0, markers=[], marker_ids=[], marker_starts=[], previous=False, next=None,
                back=False, page_size=25, supports_expansion=True, can_collapse=False)


class ProductCoefficientTests(unittest.IsolatedAsyncioTestCase):
    async def test_coefficients_stay_with_wrapped_factors(self):
        async with async_playwright() as playwright:
            for engine in ('chromium', 'firefox', 'webkit'):
                with self.subTest(engine=engine):
                    options = {'headless': True}
                    executable = os.environ.get(f'{engine.upper()}_EXECUTABLE')
                    if executable:
                        options['executable_path'] = executable
                    browser = await getattr(playwright, engine).launch(**options)
                    try:
                        await self.check_coefficients(browser)
                    finally:
                        await browser.close()

    async def check_coefficients(self, browser):
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
        cases = [('root', False, False, 290, '\u2062'), ('numerator', False, False, 290, '\u2062'),
                 ('root', True, False, 190, '\u2062'), ('root', True, True, 190, '\u2062'),
                 ('root', False, False, 290, '*')]
        for location, leaf, large, narrow, separator in cases:
            with self.subTest(location=location, leaf=leaf, large=large, separator=separator):
                fixture = coefficient_page(location, leaf, large, separator)
                await page.evaluate('''fixture => {
                  window.viewer?.close();
                  window.originalPayload = JSON.stringify(fixture);
                  window.fixture = fixture;
                  window.viewer = mount({el:document.querySelector('#host'),page:fixture,send:()=>{}});
                }''', fixture)
                await page.evaluate('document.fonts.ready')
                wide = None
                for width in (900, narrow, 900, narrow, 900):
                    await page.locator('#host').evaluate('(host,width)=>host.style.width=`${width}px`', width)
                    await self.settle(page)
                    measured = await page.locator('#host').evaluate('''host => {
                      const content = host.shadowRoot.querySelector('.math-content');
                      const product = content.querySelector('[data-test-product]');
                      const sum = content.querySelector('mtable[data-symbolica-sum]');
                      const separator = content.querySelector('[data-test-factor="next"] > mo');
                      const rows = [...product.children].map(row =>
                        [...row.querySelectorAll('[data-test-factor]')]
                          .filter(factor=>factor.closest('mtable[data-symbolica-product]')===product)
                          .map(factor=>factor.dataset.testFactor)).filter(factors=>factors.length);
                      return {
                        rows,
                        terms:[...content.querySelectorAll('[data-test-term]')].map(term=>term.dataset.testTerm),
                        termSigns:[...content.querySelectorAll('[data-test-term]')].map(term=>
                          [...term.querySelectorAll('mo')].map(operator=>operator.textContent)),
                        signs:[...content.querySelectorAll('mo')].filter(operator=>operator.textContent==='−').length,
                        sumRows:sum ? [...sum.children].map(row=>{
                          const term=row.querySelector('[data-test-term]');
                          return {terms:row.querySelectorAll('[data-test-term]').length,
                            firstTermLeft:term.getBoundingClientRect().left,
                            rowLeft:row.getBoundingClientRect().left,
                            font:parseFloat(getComputedStyle(term).fontSize)};
                        }) : [],
                        viewport:content.parentElement.clientWidth,scroll:content.scrollWidth,
                        coefficient:content.querySelector('[data-test-factor="coefficient"]').textContent,
                        separator:{text:separator.textContent,visible:separator.getBoundingClientRect().width>0},
                        unchanged:JSON.stringify(fixture)===originalPayload,
                        fractions:content.querySelectorAll('mfrac').length,
                        productContinuations:[...product.querySelectorAll('[data-symbolica-continuation]')]
                          .filter(node=>node.closest('mtable[data-symbolica-product],mtable[data-symbolica-sum]')===product)
                          .map(node=>node.textContent)
                      };
                    }''')
                    self.assertTrue(measured['unchanged'], measured)
                    self.assertEqual([factor for row in measured['rows'] for factor in row],
                                     ['coefficient', 'next'], measured)
                    self.assertLessEqual(measured['scroll'], measured['viewport'] + 2, measured)
                    self.assertEqual(measured['coefficient'], '12345678901' if large else '2', measured)
                    self.assertEqual(measured['fractions'], int(location == 'numerator'), measured)
                    self.assertEqual(measured['separator']['text'], separator, measured)
                    if separator == '*':
                        self.assertTrue(measured['separator']['visible'], measured)
                    if large and width == narrow:
                        self.assertEqual(measured['rows'], [['coefficient'], ['next']], measured)
                        self.assertEqual(measured['productContinuations'], ['×'], measured)
                    else:
                        self.assertEqual(measured['rows'], [['coefficient', 'next']], measured)
                        self.assertEqual(measured['productContinuations'], [], measured)
                    if leaf:
                        self.assertEqual(measured['terms'], [], measured)
                        self.assertEqual(measured['signs'], 1, measured)
                    else:
                        self.assertEqual(measured['terms'], [str(i) for i in range(12)], measured)
                        self.assertEqual(measured['termSigns'], [['−'] if i % 3 == 0 else ['+']
                                                                for i in range(12)], measured)
                        self.assertEqual(measured['signs'], 5, measured)
                        if width == narrow:
                            self.assertGreater(len(measured['sumRows']), 1, measured)
                        for row in measured['sumRows'][1:]:
                            self.assertGreaterEqual(row['firstTermLeft'] - row['rowLeft'],
                                                    row['font'] * .5, measured)
                    if width == 900:
                        if wide is None:
                            wide = measured
                        else:
                            self.assertEqual(measured, wide)
        await page.evaluate('viewer.close()')
        await page.close()
        self.assertFalse(errors, errors)

    async def settle(self, page):
        await page.evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
        await page.wait_for_timeout(80)


if __name__ == '__main__':
    unittest.main()
