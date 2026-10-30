"""Native pager regressions at mobile widths.

Requires a locally built Symbolica and Playwright. Run this file directly;
CHROMIUM_EXECUTABLE and SYMBOLICA_TEST_FONT can select local browser/font files.
"""

import os
from pathlib import Path
import unittest

from symbolica import S

# Initialize the native allocator before the browser driver's TLS users.
x, y, z, w, a = S("x", "y", "z", "w", "a")
EXPRESSION = ((x + y + z + w) ** 20).expand() * (1 + a)

from playwright.async_api import async_playwright

JAVASCRIPT = (Path(__file__).resolve().parents[1]
              / "src/api/python/display/paging.js").read_text()
ANCHOR = "  let lastWidth = -1, frame;"
assert JAVASCRIPT.count(ANCHOR) == 1
# Exercise the existing refinement operation on actual kernel alternatives,
# independently of which alternative a browser/font initially selects.
JAVASCRIPT = JAVASCRIPT.replace(ANCHOR, """
  el.refine = layouts => revealFittingParts(layouts, math.clientWidth);
""" + ANCHOR)


class ExpansionDetailBrowserTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.viewer = EXPRESSION.mathml_viewer()
        self.playwright = await async_playwright().start()
        engine = os.environ.get("SYMBOLICA_TEST_BROWSER", "chromium")
        options = {"headless": True}
        if executable := os.environ.get(f"{engine.upper()}_EXECUTABLE"):
            options["executable_path"] = executable
        if engine == "chromium":
            options["ignore_default_args"] = ["--hide-scrollbars"]
        self.browser = await getattr(self.playwright, engine).launch(**options)
        self.page = await self.browser.new_page(viewport={"width": 390, "height": 844})
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        font = os.environ.get("SYMBOLICA_TEST_FONT")
        await self.page.route("https://cdn.jsdelivr.net/**", lambda route:
                              route.fulfill(path=font, content_type="font/woff2")
                              if font else route.abort())
        await self.page.expose_function("backend", lambda message: self.viewer.handle_message(message))
        await self.page.set_content('<div id="host" style="width:300px;max-height:420px;overflow:auto"></div>')
        await self.page.evaluate("""async source => {
          const url=URL.createObjectURL(new Blob([source],{type:'text/javascript'}));
          window.mount=(await import(url)).mount;URL.revokeObjectURL(url);
        }""", JAVASCRIPT)

    async def asyncTearDown(self):
        await self.page.evaluate("window.pager?.close()")
        await self.browser.close()
        await self.playwright.stop()
        self.viewer.close()
        self.assertFalse(self.errors, self.errors)

    async def settle(self):
        await self.page.evaluate("""async () => {
          await document.fonts.ready;
          await new Promise(r=>requestAnimationFrame(()=>requestAnimationFrame(r)));
        }""")

    async def test_show_more_keeps_complete_terms_and_the_small_factor(self):
        await self.page.evaluate("""initial => {
          window.responses=[];
          window.pager=mount({el:document.querySelector('#host'),page:initial,send:async m=>{
            const result=await backend(m);responses.push(result);return result;
          }});
        }""", self.viewer.page())
        await self.page.wait_for_function("responses.some(r=>r[0]==='page')")
        await self.settle()

        async def state():
            return await self.page.locator('.math-content').evaluate(r"""el => ({
              markers:el.querySelectorAll('button.omission').length,
              small:el.querySelector('[data-symbolica-path="0"]').textContent,
              terms:Object.fromEntries([...el.querySelectorAll('[data-symbolica-path]')]
                .filter(n=>/^1\/\d+$/.test(n.dataset.symbolicaPath))
                .map(n=>[n.dataset.symbolicaPath,n.textContent]))
            })""")

        before = await state()
        await self.page.locator('.math-content button.omission').last.click()
        await self.page.locator('.preview-expand').click()
        await self.page.wait_for_function("responses.filter(r=>r[0]==='page').length>=2")
        await self.settle()
        after = await state()
        self.assertEqual(after['markers'], 1)
        self.assertEqual(after['small'], '1+a')
        self.assertGreater(len(after['terms']), len(before['terms']))
        self.assertTrue(all(after['terms'].get(k) == v for k, v in before['terms'].items()))

    async def test_refinement_stretches_the_original_parentheses(self):
        # Leave room for complete terms even when STIX is unavailable and the
        # fallback font has wider glyphs. The sum still wraps on this width.
        await self.page.locator('#host').evaluate("""el=>{
          el.style.width='360px';el.style.maxHeight='none';
        }""")
        initial = self.viewer.page()
        layouts = [dict(initial, **p, layouts=[]) for p in initial['layouts']]
        compact = next(p for p in layouts if p['layout'] == 6)
        detailed = next(p for p in layouts if p['layout'] == 2)
        await self.page.evaluate("""compact => {
          window.pager=mount({el:document.querySelector('#host'),page:compact,send:()=>{}});
        }""", compact)
        await self.settle()
        result = await self.page.evaluate("""layouts => {
          const host=document.querySelector('#host');host.refine(layouts);
          const part=host.shadowRoot.querySelector('[data-symbolica-path="1"]');
          const fence=part.parentElement;
          return {rows:part.querySelectorAll('mtable[data-symbolica-sum]>mtr').length,
            height:part.getBoundingClientRect().height,
            left:fence.firstElementChild.getBoundingClientRect().height,
            right:fence.lastElementChild.getBoundingClientRect().height};
        }""", [detailed, compact])
        self.assertGreater(result['rows'], 1)
        self.assertGreaterEqual(result['left'], result['height'] * .9)
        self.assertGreaterEqual(result['right'], result['height'] * .9)

    async def test_initial_height_and_explicit_expansion(self):
        for factored in (False, True):
            self.viewer.close()
            self.viewer = (EXPRESSION if factored else
                           ((x + y + z + w) ** 20).expand()).mathml_viewer()
            for theme in ('light', 'dark'):
                await self.page.emulate_media(color_scheme=theme)
                await self.page.evaluate("""initial => {
                  window.pager?.close();window.responses=[];
                  window.pager=mount({el:document.querySelector('#host'),page:initial,send:async m=>{
                    if(m.action==='dispose') return null;
                    const result=await backend(m);responses.push(result);return result;
                  }});
                }""", self.viewer.page())
                await self.page.wait_for_function("responses.some(r=>r[0]==='page')")
                for width in (1060, 760, 480, 350, 300, 1060, 300):
                    with self.subTest(factored=factored, theme=theme, width=width):
                        await self.page.set_viewport_size({'width': max(390, width + 40), 'height': 900})
                        await self.page.locator('#host').evaluate('(el,w)=>el.style.width=w+"px"', width)
                        await self.settle()
                        state = await self.height_state(factored)
                        self.assertLessEqual(state['height'], state['limit'] + 1, state)
                        self.assertLessEqual(state['scroll'], state['width'] + 1, state)
                        self.assertEqual(state['markers'], 1, state)
                        self.assertGreaterEqual(len(state['terms']), 4, state)
                        self.assertTrue(all('sum' in label.lower() for label in state['labels']))
                        if factored:
                            self.assertEqual(state['small'], '1+a')
                before = await self.height_state(factored)
                await self.page.locator('.math-content button.omission').last.click()
                await self.page.locator('.preview-expand').click()
                await self.page.wait_for_function("responses.filter(r=>r[0]==='page').length>=2")
                await self.settle()
                after = await self.height_state(factored)
                self.assertGreater(after['height'], after['limit'])
                self.assertGreater(len(after['terms']), len(before['terms']))
                self.assertTrue(all(after['terms'].get(k) == v for k, v in before['terms'].items()))
                self.assertEqual(after['markers'], 1)
                await self.page.locator('#host').evaluate('el=>el.style.width="350px"')
                await self.settle()
                self.assertEqual((await self.height_state(factored))['terms'], after['terms'])
                await self.page.locator('.collapse-all').click()
                await self.page.wait_for_function("responses.filter(r=>r[0]==='page').at(-1)?.[1].can_collapse===false")
                await self.settle()
                collapsed = await self.height_state(factored)
                self.assertLessEqual(collapsed['height'], collapsed['limit'] + 1)

    async def mount_expression(self):
        await self.page.evaluate("""initial => {
          window.pager?.close();window.responses=[];
          window.pager=mount({el:document.querySelector('#host'),page:initial,send:async m=>{
            if(m.action==='dispose') return null;
            const result=await backend(m);responses.push(result);return result;
          }});
        }""", self.viewer.page())
        await self.page.wait_for_function("responses.some(r=>r[0]==='page')")
        await self.settle()

    async def test_widening_does_not_remove_terms(self):
        await self.page.set_viewport_size({'width': 1250, 'height': 950})
        await self.mount_expression()
        # Include the boundaries where the short prefix used to move onto a
        # separate line, and where a rejected candidate adds a native scrollbar.
        widths = [350, 400, 500, 540, 546, 580, 606, 631, 666, 766,
                  866, 900, 916, 926, 951, 971, 986, 1001, 1066]
        counts = {}
        for width in widths + widths[::-1]:
            await self.page.locator('#host').evaluate('(el,w)=>el.style.width=w+"px"', width)
            await self.settle()
            state = await self.height_state(True)
            with self.subTest(width=width):
                self.assertLessEqual(state['scroll'], state['width'] + 1, state)
                self.assertLessEqual(state['height'], state['limit'] + 1, state)
                count = len(state['terms'])
                if width in counts:
                    self.assertEqual(count, counts[width], state)
                counts[width] = count
        for previous, current in zip(widths, widths[1:]):
            self.assertGreaterEqual(counts[current], counts[previous], counts)

    async def test_scaled_host_keeps_detail_and_wraps_expansion(self):
        await self.page.set_viewport_size({'width': 1600, 'height': 1100})
        baseline = None
        for css in ('', 'zoom:0.8', 'zoom:1.25',
                    'transform:scale(0.8)', 'transform:scale(1.25)'):
            with self.subTest(css=css):
                self.viewer.close()
                self.viewer = EXPRESSION.mathml_viewer()
                # Scale the surrounding notebook container, not the viewer's
                # own zoom control. Native scrollbars stay enabled throughout.
                await self.page.evaluate("""css => {
                  const host=document.querySelector('#host');
                  if (!host.parentElement.classList.contains('output')) {
                    const output=document.createElement('div');output.className='output';
                    host.replaceWith(output);output.append(host);
                  }
                  host.style.cssText='';
                  host.parentElement.style.cssText=
                    'width:700px;max-height:420px;overflow:auto;transform-origin:top left;'+css;
                }""", css)
                await self.mount_expression()
                before = await self.height_state(True)
                if baseline is None:
                    baseline = before['terms']
                self.assertEqual(before['terms'], baseline, before)
                for step in range(3):
                    await self.page.locator('.math-content button.omission').last.click()
                    await self.page.locator('.preview-expand').click()
                    await self.page.wait_for_function(
                        "n=>responses.filter(r=>r[0]==='page').length>=n", arg=step + 2)
                    await self.settle()
                    after = await self.height_state(True)
                    self.assertGreater(len(after['terms']), len(before['terms']))
                    self.assertTrue(all(after['terms'].get(k) == v for k, v in before['terms'].items()))
                    self.assertLessEqual(after['scroll'], after['width'] + 1, after)
                    overflow = await self.page.locator('.output').evaluate(
                        'el=>el.scrollWidth-el.clientWidth')
                    self.assertLessEqual(overflow, 1)
                    before = after

    async def height_state(self, factored):
        return await self.page.locator('.math-content').evaluate(r"""(el,factored) => ({
          height:el.getBoundingClientRect().height,limit:12*parseFloat(getComputedStyle(el).fontSize),
          width:el.parentElement.clientWidth,scroll:el.scrollWidth,
          markers:el.querySelectorAll('button.omission').length,
          labels:[...el.querySelectorAll('button.omission')].map(b=>b.getAttribute('aria-label')),
          termCount:el.querySelectorAll(factored ? '[data-symbolica-path="1"] [data-symbolica-term]'
            : '.math-flow > [data-symbolica-term]').length,
          small:el.querySelector('[data-symbolica-path="0"]')?.textContent,
          terms:Object.fromEntries([...el.querySelectorAll('[data-symbolica-path]')]
            .filter(n=>(factored ? /^1\/\d+$/ : /^\d+$/).test(n.dataset.symbolicaPath))
            .map(n=>[n.dataset.symbolicaPath,n.textContent]))
        })""", factored)

    async def test_show_all_bypasses_height_target(self):
        self.viewer.close()
        self.viewer = (((x + y + z + w) ** 5).expand() * (1 + a)).mathml_viewer()
        await self.page.evaluate("""initial => {
          window.responses=[];
          window.pager=mount({el:document.querySelector('#host'),page:initial,send:async m=>{
            if(m.action==='dispose') return null;
            const result=await backend(m);responses.push(result);return result;
          }});
        }""", self.viewer.page())
        await self.page.wait_for_function("responses.some(r=>r[0]==='page')")
        await self.settle()
        before = await self.height_state(True)
        self.assertLessEqual(before['height'], before['limit'] + 1)
        await self.page.locator('.math-content button.omission').last.click()
        await self.page.get_by_role('button', name='Show all', exact=True).click()
        await self.page.wait_for_function("responses.filter(r=>r[0]==='page').length>=2")
        await self.settle()
        after = await self.height_state(True)
        self.assertEqual(after['markers'], 0)
        self.assertEqual(after['termCount'], 56)
        self.assertGreater(after['height'], after['limit'])
        self.assertEqual(after['small'], '1+a')


if __name__ == "__main__":
    unittest.main()
