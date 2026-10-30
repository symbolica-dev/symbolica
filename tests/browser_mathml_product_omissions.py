"""Browser regressions for terminal omissions in measured product rows.

These fixtures use the renderer's product-factor and fraction markup. They
exercise browser packing; the Rust tests separately check that the renderer
places the terminal marker inside the numerator product with this metadata.
Run with Playwright browsers installed, or set *_EXECUTABLE to local binaries.
SYMBOLICA_TEST_FONT optionally supplies the STIX WOFF2 file.
"""
import os
from pathlib import Path
import unittest

from playwright.async_api import async_playwright

JAVASCRIPT = (Path(__file__).resolve().parents[1]
              / 'src/api/python/display/paging.js').read_text()


def omission_page(case):
    """A renderer-shaped root product, with its tail in the numerator join."""
    nested = case == 'nested'
    reciprocal = case == 'reciprocal'
    fraction = case in ('rational', 'large_denominator', 'reciprocal')
    count = 16
    factors = []
    if reciprocal:
        factors.append('<mrow data-symbolica-factor="" data-test-factor="one"><mn>1</mn></mrow>')
    elif fraction:
        factors.append('<mrow data-symbolica-factor="" data-symbolica-coefficient="" '
                       'data-test-factor="coefficient"><mn>3</mn></mrow>')
    for index in range(0 if reciprocal else count):
        separator = '<mo>&#x2062;</mo><mspace width="0.167em"/>' if index or fraction else ''
        value = f'<msub><mi>a</mi><mn>{index}</mn></msub>'
        if nested and index == 4:
            # This marker is part of a visible function argument. It is not a
            # terminal factor and must remain inside its original fences.
            value = ('<mrow data-test-nested=""><mi>f</mi><mo>(</mo><mi>q</mi>'
                     '<mo>+</mo><mtext data-symbolica-marker="1">[2]</mtext><mo>)</mo></mrow>')
        factors.append(f'<mrow data-symbolica-factor="" data-test-factor="{index}">'
                       + separator + value + '</mrow>')
    tail = ('<mrow data-symbolica-factor="" data-test-factor="tail">'
            '<mo form="infix">×</mo><mspace width="0.25em"/><mtext data-symbolica-marker="0">[1]</mtext></mrow>')
    # Static output may already have an omission row. The browser must treat
    # it as the last factor, including when it fits beside the preceding one.
    product = ('<mtable data-symbolica-product="" data-test-product="" '
               'data-symbolica-row-gap="0.5em" columnalign="left" rowspacing="0">'
               '<mtr><mtd style="padding:0"><mrow>' + ''.join(factors)
               + '</mrow></mtd></mtr><mtr><mtd style="padding:0.5em 0 0"><mrow>'
               + tail + '</mrow></mtd></mtr></mtable>')
    body = product
    if fraction:
        values = (['<mn>8</mn>'] if case == 'rational' else
                  [f'<msub><mi>b</mi><mn>{index}</mn></msub>'
                   for index in range(count if reciprocal else 6)])
        denominator = ('<mtable data-symbolica-product="" data-symbolica-row-gap="0.5em" '
                       'columnalign="left" rowspacing="0"><mtr><mtd style="padding:0"><mrow>'
                       + ''.join(f'<mrow data-symbolica-factor="" data-test-denominator-factor="{index}">'
                                 + ('<mo>&#x2062;</mo><mspace width="0.167em"/>' if index else '')
                                 + value + '</mrow>' for index, value in enumerate(values))
                       + '</mrow></mtd></mtr></mtable>')
        body = ('<mfrac data-symbolica-product-fraction=""><mrow>' + product
                + '</mrow><mrow data-test-denominator="">' + denominator + '</mrow></mfrac>')
    return dict(html='<math xmlns="http://www.w3.org/1998/Math/MathML" displaystyle="true" '
                'style="font-size:21px"><mrow>' + ('' if reciprocal else '<mo>−</mo>')
                + body + '</mrow></math>',
                start=0, end=count, total=24, kind='Product', unit='factors', layout=2,
                markers=[[[], 'Inspect product [1]']] + ([[[4, 1], 'Inspect sum [2]']] if nested else []),
                marker_ids=[0, 1] if nested else [0], marker_starts=[count, 0] if nested else [count],
                previous=False, next=count, back=False, page_size=25,
                supports_expansion=True, can_collapse=False)


MEASURE = '''host => {
  const content = host.shadowRoot.querySelector('.math-content');
  const product = content.querySelector('[data-test-product]');
  const rows = [...product.children].map(row => ({
    row,
    factors:[...row.querySelectorAll(':scope > mtd > mrow > [data-test-factor]')]
  })).filter(({factors}) => factors.length);
  const tail = content.querySelector('[data-test-factor="tail"]');
  const button = tail.querySelector('button.omission');
  const last = rows.at(-1);
  const prefix = last.row.querySelector(':scope > mtd > mrow > [data-symbolica-continuation]');
  const visible = node => node.getBoundingClientRect().width > 0;
  const nested = content.querySelector('[data-test-nested]');
  const fraction = content.querySelector('mfrac');
  const cross = [...last.row.querySelectorAll('mo')].filter(node=>
    node.textContent==='×' && visible(node)).at(-1);
  const markerText = button.parentElement;
  const rect = node => {
    const r=node.getBoundingClientRect();
    return {left:r.left,right:r.right,top:r.top,bottom:r.bottom,width:r.width};
  };
  return {
    compact:!!content.querySelector('[data-symbolica-product-prefactor]'),
    rows:rows.map(({factors})=>factors.map(node=>node.dataset.testFactor)),
    markerInsideProduct:button.closest('mtable[data-symbolica-product]')===product,
    markerIsLast:tail===last.factors.at(-1),
    tailFirst:tail===last.factors[0],
    tailOperator:tail.firstElementChild.textContent,
    tailOperatorVisible:visible(tail.firstElementChild),
    markerGap:markerText.getBoundingClientRect().left-cross.getBoundingClientRect().right,
    markerGapEm:parseFloat(getComputedStyle(cross).fontSize),
    prefixOperators:prefix ? [...prefix.querySelectorAll('mo')].filter(visible).map(n=>n.textContent) : [],
    tailRowCrosses:[...last.row.querySelectorAll('mo')]
      .filter(node=>node.closest('mtable[data-symbolica-product]')===product
        && node.textContent==='×' && visible(node)).length,
    marker:rect(button),tail:rect(tail),lastRow:rect(last.row),
    markerCount:content.querySelectorAll('button.omission').length,
    handlers:[...content.querySelectorAll('button.omission')].map(button=>
      typeof button.onclick==='function' && typeof button.onfocus==='function'),
    signs:[...content.querySelectorAll('mo')].filter(n=>n.textContent==='−').length,
    coefficient:content.querySelector('[data-symbolica-coefficient]')?.textContent || null,
    fractions:content.querySelectorAll('mfrac').length,
    fractionNumerator:fraction?.firstElementChild.textContent || null,
    denominator:content.querySelector('[data-test-denominator]')?.textContent.replace(/\u2062/g,'') || null,
    denominatorFactors:[...content.querySelectorAll('[data-test-denominator-factor]')]
      .map(node=>node.dataset.testDenominatorFactor),
    nested:nested ? {
      factor:nested.closest('[data-test-factor]').dataset.testFactor,
      marker:nested.querySelector('[data-symbolica-marker]').dataset.symbolicaMarker,
      operators:[...nested.querySelectorAll('mo')].map(n=>n.textContent),
      factors:nested.querySelectorAll('[data-symbolica-factor]').length
    } : null,
    unchanged:JSON.stringify(fixture)===originalPayload,
    viewport:content.parentElement.clientWidth,scroll:content.scrollWidth
  };
}'''


class ProductOmissionTests(unittest.IsolatedAsyncioTestCase):
    async def test_product_omissions_follow_visible_factors(self):
        async with async_playwright() as playwright:
            for engine in ('chromium', 'firefox', 'webkit'):
                with self.subTest(engine=engine):
                    options = {'headless': True}
                    executable = os.environ.get(f'{engine.upper()}_EXECUTABLE')
                    if executable:
                        options['executable_path'] = executable
                    browser = await getattr(playwright, engine).launch(**options)
                    try:
                        await self.check_product_omissions(browser)
                    finally:
                        await browser.close()

    async def check_product_omissions(self, browser):
        page = await browser.new_page(viewport={'width': 1200, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        font = os.environ.get('SYMBOLICA_TEST_FONT')
        await page.route('https://cdn.jsdelivr.net/**', lambda route:
                         route.fulfill(path=font, content_type='font/woff2') if font else route.abort())
        await page.set_content('<div id="host" style="width:900px"></div>')
        await page.evaluate('''async source => {
          const url=URL.createObjectURL(new Blob([source],{type:'text/javascript'}));
          window.mount=(await import(url)).mount;
          URL.revokeObjectURL(url);
        }''', JAVASCRIPT)
        for case in ('plain', 'rational', 'large_denominator', 'tail_alone', 'nested', 'reciprocal'):
            with self.subTest(case=case):
                await page.evaluate('''fixture => {
                  window.viewer?.close();
                  window.fixture=fixture;
                  window.originalPayload=JSON.stringify(fixture);
                  window.calls=[];
                  window.viewer=mount({el:document.querySelector('#host'),page:fixture,
                    send:message=>{ calls.push(message); }});
                }''', omission_page(case))
                await page.evaluate('document.fonts.ready')
                narrow = 260
                if case == 'tail_alone':
                    # Find a real boundary with the current engine/font rather
                    # than inflating a marker to force a particular line break.
                    narrow = None
                    for width in range(180, 361, 5):
                        await self.resize(page, width)
                        state = await page.locator('#host').evaluate(MEASURE)
                        if state['rows'][-1] == ['tail']:
                            narrow = width
                            break
                    self.assertIsNotNone(narrow, 'No marker-only continuation found')
                wide_state = None
                for width in (900, narrow, 900, narrow, 900):
                    await self.resize(page, width)
                    state = await page.locator('#host').evaluate(MEASURE)
                    self.assertTrue(state['unchanged'], state)
                    self.assertTrue(state['markerInsideProduct'], state)
                    self.assertTrue(state['markerIsLast'], state)
                    self.assertEqual(state['tailOperator'], '×', state)
                    # Inline omissions and marker-only continuations use the
                    # same explicit quarter-em gap, without counting the
                    # button's independent one-pixel focus/border margin.
                    self.assertAlmostEqual(state['markerGap'], .25 * state['markerGapEm'],
                                           delta=.5, msg=state)
                    self.assertLessEqual(state['scroll'], state['viewport'] + 2, state)
                    self.assertGreaterEqual(state['marker']['left'], state['lastRow']['left'] - 1, state)
                    self.assertLessEqual(state['marker']['right'], state['lastRow']['right'] + 1, state)
                    self.assertEqual(state['signs'], 0 if case == 'reciprocal' else 1, state)
                    self.assertTrue(all(state['handlers']), state)
                    self.assertEqual(state['markerCount'], 2 if case == 'nested' else 1, state)
                    compact = case == 'rational' and width == narrow
                    self.assertEqual(state['compact'], compact, state)
                    expected = ['one', 'tail'] if case == 'reciprocal' else [str(i) for i in range(16)] + ['tail']
                    if case in ('rational', 'large_denominator') and not compact:
                        expected.insert(0, 'coefficient')
                    self.assertEqual([factor for row in state['rows'] for factor in row], expected, state)
                    self.assertEqual(state['fractions'], int(case in ('rational', 'large_denominator', 'reciprocal')), state)
                    self.assertEqual(state['coefficient'], '3' if case in ('rational', 'large_denominator') else None, state)
                    if case == 'rational':
                        self.assertEqual(state['denominator'], '8', state)
                        if compact:
                            self.assertEqual(state['fractionNumerator'], '3', state)
                    elif case == 'large_denominator':
                        self.assertEqual(state['denominator'], 'b0b1b2b3b4b5', state)
                    elif case == 'reciprocal':
                        self.assertEqual(state['denominator'].replace('×', ''), ''.join(f'b{i}' for i in range(16)), state)
                        self.assertEqual(state['denominatorFactors'], [str(i) for i in range(16)], state)
                        self.assertEqual(state['rows'], [['one', 'tail']], state)
                        self.assertEqual(state['fractionNumerator'], '1×1', state)
                    if state['tailFirst'] and len(state['rows']) > 1:
                        self.assertFalse(state['tailOperatorVisible'], state)
                        self.assertEqual(state['prefixOperators'], ['×'], state)
                        self.assertEqual(state['tailRowCrosses'], 1, state)
                    else:
                        self.assertTrue(state['tailOperatorVisible'], state)
                        self.assertEqual(state['tailRowCrosses'], 1 + len(state['prefixOperators']), state)
                    if case == 'nested':
                        self.assertEqual(state['nested'], dict(factor='4', marker='1',
                                                              operators=['(', '+', ')'], factors=0), state)
                    if width == narrow:
                        if case == 'reciprocal':
                            self.assertEqual(len(state['rows']), 1, state)
                        else:
                            self.assertGreater(len(state['rows']), 1, state)
                        if case == 'tail_alone':
                            self.assertEqual(state['rows'][-1], ['tail'], state)
                    else:
                        self.assertEqual(len(state['rows']), 1, state)
                        if wide_state is None:
                            wide_state = state
                        else:
                            self.assertEqual(state, wide_state)
                    # update(true) reparses candidates on a resize. Verify the
                    # currently decorated button remains bound after packing,
                    # including the path/start used for its backend request.
                    for marker, path, start in [(0, [], 16)] + ([(1, [4, 1], 0)] if case == 'nested' else []):
                        await self.check_routing(page, marker, path, start)
        await page.evaluate('viewer.close()')
        await page.close()
        self.assertFalse(errors, errors)

    async def check_routing(self, page, marker, path, start):
        before = await page.evaluate('calls.length')
        await page.locator('#host').evaluate('''(host, marker) => {
          host.shadowRoot.querySelector(`.math-content [data-symbolica-marker="${marker}"] button`).click();
        }''', marker)
        await page.wait_for_function('before=>calls.slice(before).some(c=>c.action==="preview"&&c.popup)', arg=before)
        request = await page.evaluate('before=>calls.slice(before).find(c=>c.action==="preview"&&c.popup)', before)
        self.assertEqual(request['value'], path, request)
        self.assertEqual(request['start'], start, request)
        self.assertEqual(request['layout'], 2, request)
        await page.keyboard.press('Escape')

    async def resize(self, page, width):
        await page.locator('#host').evaluate('(host,width)=>host.style.width=`${width}px`', width)
        await page.evaluate('new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
        await page.wait_for_timeout(80)


if __name__ == '__main__':
    unittest.main()
