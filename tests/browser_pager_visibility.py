"""Refit only nearby outputs, retaining dirty width/font changes until visible.

Run with Playwright; SYMBOLICA_TEST_ENGINES and *_EXECUTABLE select browsers.
"""

import os
import unittest
from pathlib import Path

from browser_mathml_wrapping import product_page
from playwright.async_api import async_playwright

SOURCE = (
    Path(__file__).resolve().parents[1] / "src/api/python/display/paging.js"
).read_text()
ANCHOR = "  function update(preservePreview = false) {"
assert SOURCE.count(ANCHOR) == 1
SOURCE = SOURCE.replace(
    ANCHOR,
    ANCHOR + "\n    el.refits = (el.refits || 0) + (preservePreview === true ? 1 : 0);",
)
# Expose DOM measurements, not the visibility or scheduling implementation.
ANCHOR = "  let lastWidth = -1, frame;"
assert SOURCE.count(ANCHOR) == 1
SOURCE = SOURCE.replace(
    ANCHOR,
    """
  el.geometry = () => [...content.querySelectorAll('[data-test-factor]')].map(factor => {
    const rect = factor.getBoundingClientRect(), origin = math.getBoundingClientRect();
    return [rect.x - origin.x, rect.y - origin.y, rect.width, rect.height];
  });
"""
    + ANCHOR,
)
SETTLE = """async () => {
  for (let i = 0; i < 8; ++i) await new Promise(requestAnimationFrame);
}"""


class VisibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_deferred_refits_and_catchup(self):
        async with async_playwright() as playwright:
            for engine in os.environ.get(
                "SYMBOLICA_TEST_ENGINES", "chromium,firefox,webkit"
            ).split(","):
                browser = await getattr(playwright, engine).launch(
                    executable_path=os.environ.get(f"{engine.upper()}_EXECUTABLE")
                )
                try:
                    # Include ancestor clipping and an iframe-owned document.
                    for container in ("document", "scroller", "iframe", "fallback"):
                        with self.subTest(engine=engine, container=container):
                            page = await browser.new_page(
                                viewport={"width": 1100, "height": 600}
                            )
                            errors = []
                            page.on(
                                "pageerror",
                                lambda error, errors=errors: errors.append(str(error)),
                            )
                            await page.route(
                                "https://cdn.jsdelivr.net/**",
                                lambda route: route.abort(),
                            )
                            inner = '<style>body{margin:0}#host{width:900px}</style><div id="space" style="height:2000px"></div><div id="host"></div><div style="height:2000px"></div>'
                            if container == "iframe":
                                await page.set_content(
                                    '<iframe style="width:1000px;height:500px"></iframe>'
                                )
                                frame = page.frames[1]
                                await frame.set_content(inner)
                            else:
                                await page.set_content(
                                    inner
                                    if container != "scroller"
                                    else '<div id="scroller" style="height:500px;overflow:auto">'
                                    + inner
                                    + "</div>"
                                )
                                frame = page
                            await frame.evaluate(
                                """async ({source, fixture, fallback}) => {
                              if (fallback) window.IntersectionObserver = undefined;
                              const url = URL.createObjectURL(new Blob([source], {type:'text/javascript'}));
                              window.mount = (await import(url)).mount; URL.revokeObjectURL(url);
                              window.viewer = mount({el:host, page:fixture, send:()=>null});
                            }""",
                                {
                                    "source": SOURCE,
                                    "fixture": product_page(),
                                    "fallback": container == "fallback",
                                },
                            )
                            await frame.evaluate(SETTLE)
                            before = await frame.evaluate(
                                "({calls:host.refits, geometry:host.geometry(), height:host.offsetHeight})"
                            )
                            await frame.evaluate("""() => {
                              host.refits = 0; host.style.width = '400px';
                              document.fonts.dispatchEvent(new FontFaceSetLoadEvent('loadingdone', {
                                fontfaces:[new FontFace('STIX Two Math', 'local(serif)')]}));
                            }""")
                            await frame.evaluate(SETTLE)
                            after = await frame.evaluate(
                                "({calls:host.refits, geometry:host.geometry(), height:host.offsetHeight})"
                            )
                            if container == "fallback":
                                self.assertGreater(after["calls"], 0)
                            else:
                                self.assertEqual(after["calls"], 0)
                                # Cached rows and output height remain intact offscreen.
                                self.assertEqual(before["geometry"], after["geometry"])
                                self.assertEqual(before["height"], after["height"])
                            # Coalesce another width and a later font notification.
                            await frame.evaluate("""() => {
                              host.style.width = '300px';
                              document.fonts.dispatchEvent(new FontFaceSetLoadEvent('loadingdone', {
                                fontfaces:[new FontFace('STIX Two Math', 'local(serif)')]}));
                            }""")
                            await frame.evaluate(SETTLE)
                            await frame.evaluate("""() => {
                              const scroller = document.querySelector('#scroller');
                              if (scroller) scroller.scrollTop = 1800;
                              else window.scrollTo(0, 1800);
                            }""")
                            await frame.evaluate(SETTLE)
                            caught_up = await frame.evaluate("host.geometry()")
                            self.assertNotEqual(caught_up, before["geometry"])
                            if container != "fallback":
                                self.assertEqual(await frame.evaluate("host.refits"), 1)
                            # Compare with a newly mounted visible viewer at the latest width.
                            await frame.evaluate(
                                """fixture => {
                              const el = document.createElement('div');el.id='fresh';el.style.width='300px';
                              host.after(el);window.other=mount({el,page:fixture,send:()=>null});
                            }""",
                                product_page(),
                            )
                            await frame.evaluate(SETTLE)
                            self.assertEqual(
                                caught_up, await frame.evaluate("fresh.geometry()")
                            )
                            await frame.evaluate("""async () => {
                              await other.close();fresh.remove();host.refits=0;
                              if (document.querySelector('#scroller')) scroller.scrollTop=0;
                              else window.scrollTo(0,0);
                            }""")
                            await frame.evaluate(SETTLE)
                            # Scrolling out and back without a change needs no new fit.
                            await frame.evaluate("host.scrollIntoView()")
                            await frame.evaluate(SETTLE)
                            self.assertEqual(await frame.evaluate("host.refits"), 0)
                            await frame.evaluate("""() => {
                              if (document.querySelector('#scroller')) scroller.scrollTop=0;
                              else window.scrollTo(0,0);
                            }""")
                            await frame.evaluate(SETTLE)
                            # A late font alone must also be remembered offscreen.
                            await frame.evaluate("""() => {
                              document.fonts.dispatchEvent(new FontFaceSetLoadEvent('loadingdone', {
                                fontfaces:[new FontFace('STIX Two Math','local(serif)')]}));
                            }""")
                            await frame.evaluate(SETTLE)
                            if container != "fallback":
                                self.assertEqual(await frame.evaluate("host.refits"), 0)
                            await frame.evaluate("host.scrollIntoView()")
                            await frame.evaluate(SETTLE)
                            self.assertEqual(await frame.evaluate("host.refits"), 1)
                            # Incoming explicit pages still render while out of view.
                            await frame.evaluate("""() => {
                              if (document.querySelector('#scroller')) scroller.scrollTop=0;
                              else window.scrollTo(0,0);
                            }""")
                            await frame.evaluate(SETTLE)
                            updated = product_page()
                            updated["html"] = updated["html"].replace(
                                "<mi>x0</mi>", "<mi>changed</mi>"
                            )
                            await frame.evaluate(
                                'fixture=>viewer.update(["page", fixture])', updated
                            )
                            self.assertIn(
                                "changed",
                                await frame.evaluate("host.shadowRoot.textContent"),
                            )
                            await frame.evaluate("""async () => {
                              await viewer.close();host.refits=0;host.style.width='450px';
                              document.fonts.dispatchEvent(new FontFaceSetLoadEvent('loadingdone', {
                                fontfaces:[new FontFace('STIX Two Math','local(serif)')]}));
                              host.scrollIntoView();
                            }""")
                            await frame.evaluate(SETTLE)
                            self.assertEqual(await frame.evaluate("host.refits"), 0)
                            self.assertFalse(errors, errors)
                            await page.close()
                finally:
                    await browser.close()


if __name__ == "__main__":
    unittest.main()
