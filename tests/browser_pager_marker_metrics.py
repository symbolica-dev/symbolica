"""Renumber omissions without refitting unless their measured size changes.

Run directly with Playwright; SYMBOLICA_TEST_ENGINES, *_EXECUTABLE and
SYMBOLICA_TEST_FONT select installed browsers and an optional local STIX font.
"""

import os
import unittest
from pathlib import Path

from playwright.async_api import async_playwright

JAVASCRIPT = (
    Path(__file__).resolve().parents[1] / "src/api/python/display/paging.js"
).read_text()
HOOK = """
  el.testNumbering = ({texts,size,family,zoom}) => {
    const markers=[...content.querySelectorAll('button.omission')];
    markers.forEach((marker,i)=>{
      marker.textContent=texts[i];
      marker.style.fontSize=`${size}px`;
      marker.style.fontFamily=family;
    });
    content.style.zoom=zoom;
    setWrapWidth(content,math.clientWidth);
    const boxes=()=>markers.map(marker=>{
      const rect=marker.getBoundingClientRect();return [rect.width,rect.height];
    });
    const rows=()=>[...content.querySelectorAll('mtr')].map(row=>{
      const rect=row.getBoundingClientRect();return [rect.x,rect.y,rect.width,rect.height];
    });
    const bindings=markers.map(marker=>inspection.get(marker));
    const before=boxes();
    const changed=numberMarkers(content);
    const after=boxes();
    if(changed) setWrapWidth(content,math.clientWidth);
    const fitted=rows();
    setWrapWidth(content,math.clientWidth);
    return {before,after,changed,fitted,forced:rows(),
      texts:markers.map(marker=>marker.textContent),
      labels:markers.map(marker=>marker.getAttribute('aria-label')),
      sameBindings:markers.every((marker,i)=>inspection.get(marker)===bindings[i])};
  };
"""
ANCHOR = "  let lastWidth = -1, frame;"
assert JAVASCRIPT.count(ANCHOR) == 1
JAVASCRIPT = JAVASCRIPT.replace(ANCHOR, HOOK + "\n" + ANCHOR)


def fixture(count):
    factors = "".join(
        '<mrow data-symbolica-factor=""><mi>x</mi>'
        f'<mtext data-symbolica-marker="{i}">[{i + 1}]</mtext></mrow>'
        for i in range(count)
    )
    html = (
        '<math displaystyle="true"><mtable data-symbolica-product="" '
        'data-symbolica-row-gap="0.5em"><mtr><mtd style="padding:0">'
        "<mrow>" + factors + "</mrow></mtd></mtr></mtable></math>"
    )
    return {
        "html": html,
        "start": 0,
        "end": count,
        "total": count,
        "kind": "Expression",
        "unit": "factors",
        "markers": [[[i], f"Inspect factor [{i + 1}]"] for i in range(count)],
        "marker_ids": list(range(count)),
        "marker_starts": [0] * count,
        "layout": 0,
        "previous": False,
        "next": None,
        "back": False,
        "page_size": 25,
        "can_collapse": False,
    }


class MarkerMetricsTests(unittest.IsolatedAsyncioTestCase):
    async def test_actual_marker_dimensions_control_refitting(self):
        async with async_playwright() as playwright:
            for engine in os.environ.get(
                "SYMBOLICA_TEST_ENGINES", "chromium,firefox,webkit"
            ).split(","):
                with self.subTest(engine=engine):
                    browser = await getattr(playwright, engine).launch(
                        executable_path=os.environ.get(f"{engine.upper()}_EXECUTABLE")
                    )
                    try:
                        page = await browser.new_page()
                        errors = []
                        page.on(
                            "pageerror",
                            lambda error, errors=errors: errors.append(str(error)),
                        )
                        font = os.environ.get("SYMBOLICA_TEST_FONT")

                        async def font_handler(route, _request, font=font):
                            if font:
                                await route.fulfill(
                                    path=font, content_type="font/woff2"
                                )
                            else:
                                await route.abort()

                        await page.route("https://cdn.jsdelivr.net/**", font_handler)
                        await page.set_content(
                            '<div id="host" style="width:260px"></div>'
                        )
                        await page.evaluate(
                            """async source=>{
                          const url=URL.createObjectURL(new Blob([source],{type:'text/javascript'}));
                          window.mount=(await import(url)).mount;URL.revokeObjectURL(url);
                        }""",
                            JAVASCRIPT,
                        )
                        for texts, expected in (
                            (["8", "9"], False),
                            (["888", "9"], True),
                            (["1"] * 10, True),
                        ):
                            for family in (
                                '"STIX Two Math",math',
                                "missing-test-font,serif",
                            ):
                                for size, zoom in ((13, 1), (19.5, 1), (19.5, 1.25)):
                                    with self.subTest(
                                        texts=texts, family=family, size=size, zoom=zoom
                                    ):
                                        await page.evaluate(
                                            """async fixture=>{
                                          await window.viewer?.close();
                                          window.viewer=mount({el:document.querySelector('#host'),
                                            page:fixture,send:()=>null});
                                          await document.fonts.ready;
                                          await new Promise(resolve=>requestAnimationFrame(
                                            ()=>requestAnimationFrame(resolve)));
                                        }""",
                                            fixture(len(texts)),
                                        )
                                        result = await page.evaluate(
                                            """args=>
                                          document.querySelector('#host').testNumbering(args)""",
                                            {
                                                "texts": texts,
                                                "family": family,
                                                "size": size,
                                                "zoom": zoom,
                                            },
                                        )
                                        self.assertEqual(result["changed"], expected)
                                        self.assertEqual(
                                            result["changed"],
                                            result["before"] != result["after"],
                                        )
                                        self.assertEqual(
                                            result["fitted"], result["forced"]
                                        )
                                        self.assertEqual(
                                            result["texts"],
                                            [str(i + 1) for i in range(len(texts))],
                                        )
                                        self.assertEqual(
                                            result["labels"],
                                            [
                                                f"Open factor [{i + 1}]"
                                                for i in range(len(texts))
                                            ],
                                        )
                                        self.assertTrue(result["sameBindings"])
                        self.assertFalse(errors, errors)
                    finally:
                        await browser.close()


if __name__ == "__main__":
    unittest.main()
