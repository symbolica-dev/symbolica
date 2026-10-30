"""Render real backend pages and keep a product's sign on its first row.

Requires the built Symbolica Python module and Playwright. Set *_EXECUTABLE
for local browsers and SYMBOLICA_TEST_FONT for a local STIX WOFF2 file.
SYMBOLICA_SIGN_ARTIFACTS optionally saves the actual pages and PNG evidence.
"""
import copy
import json
import os
from pathlib import Path
import unittest

from playwright.async_api import async_playwright

JAVASCRIPT = (Path(__file__).resolve().parents[1]
              / 'src/api/python/display/paging.js').read_text()


def backend_pages():
    from symbolica import E

    factors = 'G^8*' + '*'.join(f'(k{i}+p{i})' for i in range(12))
    imaginary = E('-1i/16*' + factors).mathml_viewer(8)
    # Complete-term previews retain modest products even in narrow alternatives.
    # Use a product beyond the per-layout factor bound to exercise expansion.
    negative_sum = E('a-1i/16*G^8*' + '*'.join(
        f'(k{i}+p{i})' for i in range(600))).mathml_viewer(25)
    rational = E('-3/16*' + factors).mathml_viewer(8)
    unit = E('-x').mathml_viewer(1)
    try:
        preview = imaginary.page()
        _, expanded = imaginary.handle_message(dict(action='expand_all', value=[]))
        # Expanding the nested product must not add a '+' before its infix minus.
        initial_sum = negative_sum.page()
        compact = next(p for p in [initial_sum, *initial_sum['layouts']]
                       if any(path == [1] for path, _ in p['markers']))
        _, expanded_sum = negative_sum.handle_message(dict(
            action='expand', value=[1], size=25, layout=compact['layout']))
        assert 'error' not in expanded and 'error' not in expanded_sum
        return {'imaginary-preview': preview,
                'imaginary-expanded': expanded,
                'expanded-negative-sum': expanded_sum,
                'canonical-rational': rational.page(),
                'negative-unit-tail': unit.page()}
    finally:
        for viewer in (imaginary, negative_sum, rational, unit):
            viewer.close()


MEASURE = '''host => {
  const content = host.shadowRoot.querySelector('.math-content');
  const product = content.querySelector('mtable[data-symbolica-product]');
  const ownFactors = [...product.querySelectorAll('[data-symbolica-factor]')]
    .filter(node => node.closest('mtable[data-symbolica-product]') === product);
  const minus = [...content.querySelectorAll('mo')].filter(node => node.textContent === '−');
  const sign = minus[0];
  const first = ownFactors[0];
  const signBox = sign.getBoundingClientRect();
  const firstBox = first.getBoundingClientRect();
  const rows = [...product.children].filter(row => ownFactors.some(f => row.contains(f)));
  const rect = node => {
    const r = node.getBoundingClientRect();
    return {left:r.left, top:r.top, right:r.right, bottom:r.bottom, width:r.width};
  };
  return {
    signs:minus.length,
    owned:first.contains(sign),
    signOnFirstRow:rows[0].contains(sign),
    signVisible:signBox.width > 0 && getComputedStyle(sign).display !== 'none',
    signSeparator:sign.hasAttribute('data-symbolica-continuation-separator'),
    signForm:sign.getAttribute('form'),
    sign:rect(sign), first:rect(first), product:rect(product),
    rowCount:rows.length,
    compact:!!content.querySelector('[data-symbolica-product-prefactor]'),
    canonical:!!content.querySelector('mfrac[data-symbolica-product-fraction]'),
    leadingSumText:content.textContent.replaceAll('\\u2062','').slice(0,8),
    factors:ownFactors.map(f => f.textContent.replaceAll('\\u2062','').replace(/^×/,'')),
    continuations:[...product.querySelectorAll('[data-symbolica-continuation]')]
      .filter(node => node.closest('mtable[data-symbolica-product]') === product)
      .map(node => node.textContent),
    scroll:content.scrollWidth, viewport:content.parentElement.clientWidth,
    unchanged:JSON.stringify(fixture) === originalPayload
  };
}'''


class ProductSignTests(unittest.IsolatedAsyncioTestCase):
    async def test_product_signs_stay_on_first_row(self):
        async with async_playwright() as playwright:
            for engine in ('chromium', 'firefox', 'webkit'):
                with self.subTest(engine=engine):
                    options = {'headless': True}
                    executable = os.environ.get(f'{engine.upper()}_EXECUTABLE')
                    if executable:
                        options['executable_path'] = executable
                    browser = await getattr(playwright, engine).launch(**options)
                    try:
                        await self.check_product_signs(browser)
                    finally:
                        await browser.close()

    async def check_product_signs(self, browser):
        fixtures = backend_pages()
        artifacts = os.environ.get('SYMBOLICA_SIGN_ARTIFACTS')
        artifacts = Path(artifacts) if artifacts else None
        if artifacts:
            artifacts.mkdir(parents=True, exist_ok=True)
            (artifacts / 'backend-pages.json').write_text(json.dumps(fixtures, indent=2))
        page = await browser.new_page(viewport={'width': 2000, 'height': 1000})
        errors = []
        page.on('pageerror', lambda error: errors.append(str(error)))
        font = os.environ.get('SYMBOLICA_TEST_FONT')
        await page.route('https://cdn.jsdelivr.net/**', lambda route:
                         route.fulfill(path=font, content_type='font/woff2') if font else route.abort())
        await page.set_content('<div id="host" style="width:1800px"></div>')
        await page.evaluate('''async source => {
          const url = URL.createObjectURL(new Blob([source], {type:'text/javascript'}));
          window.mount = (await import(url)).mount;
          URL.revokeObjectURL(url);
        }''', JAVASCRIPT)

        async def mount(fixture, update=False):
            await page.evaluate('''async ({value, update}) => {
              window.fixture = value;
              window.originalPayload = JSON.stringify(value);
              if (update) window.viewer.update(['page', value]);
              else {
                await window.viewer?.close();
                window.viewer = mount({el:document.querySelector('#host'),page:value,send:()=>{}});
              }
            }''', {'value': fixture, 'update': update})
            await page.evaluate('document.fonts.ready')

        async def resize(width):
            await page.locator('#host').evaluate('(host,w)=>host.style.width=`${w}px`', width)
            await page.evaluate('new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))')
            await page.wait_for_timeout(100)
            return await page.locator('#host').evaluate(MEASURE)

        report = {}
        for name, fixture in fixtures.items():
            with self.subTest(case=name):
                await mount(fixture, update=name == 'imaginary-expanded')
                measurements = []
                for width in (1800, 360, 900, 360, 1800):
                    measured = await resize(width)
                    measurements.append(dict(width=width, **measured))
                    self.assertEqual(measured['signs'], 1, (name, width, measured))
                    self.assertTrue(measured['signVisible'], (name, width, measured))
                    self.assertFalse(measured['signSeparator'], (name, width, measured))
                    self.assertTrue(measured['unchanged'], measured)
                    self.assertLessEqual(measured['scroll'], measured['viewport'] + 2, measured)
                    if name == 'canonical-rational' and measured['canonical']:
                        self.assertFalse(measured['owned'], measured)
                        self.assertFalse(measured['compact'], measured)
                    else:
                        self.assertTrue(measured['owned'], (name, width, measured))
                        self.assertTrue(measured['signOnFirstRow'], measured)
                        self.assertLessEqual(measured['first']['top'], measured['sign']['top'] + 1, measured)
                        self.assertGreaterEqual(measured['first']['bottom'], measured['sign']['bottom'] - 1, measured)
                    self.assertEqual(measured['continuations'], ['×'] * (measured['rowCount'] - 1), measured)
                    if name == 'expanded-negative-sum':
                        self.assertEqual(measured['signForm'], 'infix', measured)
                        self.assertTrue(measured['leadingSumText'].startswith('a−'), measured)
                    if width == 360 and name != 'negative-unit-tail':
                        self.assertGreater(measured['rowCount'], 1, measured)
                    if artifacts and width == 360:
                        await page.locator('#host').screenshot(path=str(artifacts / f'{name}-360.png'))
                self.assertEqual(measurements[0], measurements[-1])
                self.assertEqual(measurements[1], measurements[3])
                if name == 'canonical-rational':
                    self.assertTrue(measurements[0]['canonical'], measurements)
                    self.assertTrue(measurements[1]['compact'], measurements)
                report[name] = measurements

        # Negative control: reconstruct the old external sign from a real
        # backend page. It floats beside the multiline table, failing exactly
        # the owned-first-row invariant used above (not merely a text check).
        old = copy.deepcopy(fixtures['imaginary-expanded'])
        for payload in [old, *old.get('layouts', [])]:
            payload['html'] = payload['html'].replace(
                '<mrow data-symbolica-factor=""><mo>−</mo>',
                '<mrow data-symbolica-factor="">', 1).replace(
                '<mtable data-symbolica-product=""',
                '<mo>−</mo><mtable data-symbolica-product=""', 1)
        await mount(old)
        measured = await resize(360)
        self.assertFalse(measured['owned'], measured)
        self.assertFalse(measured['signOnFirstRow'], measured)
        self.assertGreater(measured['sign']['top'], measured['first']['bottom'], measured)
        report['old-external-sign-negative-control'] = measured
        if artifacts:
            await page.locator('#host').screenshot(path=str(artifacts / 'old-external-sign-360.png'))
            (artifacts / 'browser-report.json').write_text(json.dumps(report, indent=2))
        await page.evaluate('viewer.close()')
        await page.close()
        self.assertFalse(errors, errors)


if __name__ == '__main__':
    unittest.main()
