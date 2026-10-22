"""Exercise the shared viewer in real browser engines, without a native build.

Install playwright and its chromium, firefox and webkit browsers, then run:
    python tests/browser_mathml_wrapping.py
Optional *_EXECUTABLE environment variables select existing browser binaries.
Set SYMBOLICA_TEST_STATIC=1 to also check static previews from a locally built
Symbolica module (selected with PYTHONPATH).
"""

import math
import os
from pathlib import Path
import unittest

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


class MathMLWrappingTests(unittest.IsolatedAsyncioTestCase):
    async def test_composite_cell_overflow_selects_a_compact_page(self):
        async with async_playwright() as playwright:
            for engine in ("chromium", "firefox", "webkit"):
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
                                '<div data-symbolica-cell=""><math><mspace width="300px"/></math></div>'
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
            for engine in ("chromium", "firefox", "webkit"):
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
            for engine in ("chromium", "firefox", "webkit"):
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
            for engine in ("chromium", "firefox", "webkit"):
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
                    self.assertIn('factors 1–5', await page.locator('.preview-heading').text_content())
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
