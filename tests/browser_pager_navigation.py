"""Pager interaction regressions, using real browser focus and layout.

Run with python tests/browser_pager_navigation.py. CHROMIUM_EXECUTABLE can select
an installed Chromium. No Symbolica module or native build is required.
"""

import os
from pathlib import Path
import unittest

from playwright.async_api import async_playwright

JAVASCRIPT = (Path(__file__).resolve().parents[1]
              / 'src/api/python/display/paging.js').read_text()


def fixture():
    return dict(html='<math displaystyle="true"><mrow data-symbolica-path="2">'
                     '<mi>x</mi><mo>+</mo><mtext data-symbolica-marker="0">[1]</mtext>'
                     '</mrow></math>',
                start=0, end=1, total=3, kind='Sum', unit='terms',
                markers=[[[2], 'Inspect sum [1]']], marker_ids=[0], marker_starts=[7],
                layout=0, previous=False, next=None, back=False, page_size=5,
                supports_expansion=True, can_collapse=False)


class PagerNavigationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.playwright = await async_playwright().start()
        executable = os.environ.get('CHROMIUM_EXECUTABLE')
        self.browser = await self.playwright.chromium.launch(
            executable_path=executable or None, headless=True)
        self.page = await self.browser.new_page(viewport={'width': 1000, 'height': 800})
        self.page.set_default_timeout(4000)
        self.errors = []
        self.page.on('pageerror', lambda error: self.errors.append(str(error)))
        await self.page.route('https://cdn.jsdelivr.net/**', lambda route: route.abort())
        await self.page.set_content('<input id="editor"><div id="host" style="width:800px"></div>')
        await self.page.evaluate('''async ({source, fixture}) => {
          const url = URL.createObjectURL(new Blob([source], {type:'text/javascript'}));
          const {mount} = await import(url);
          URL.revokeObjectURL(url);
          window.fixture = fixture;
          window.sent = [];
          window.allowAll = true;
          window.branchExpanded = false;
          window.viewer = mount({el:document.querySelector('#host'), page:fixture, send:message => {
            sent.push(message);
            if (message.action === 'preview') {
              const nested = message.context != null;
              return ['preview', {...fixture, request:message.request, context:'popup-origin',
                start:message.start, end:message.start+3, total:40, next:message.start+3,
                can_expand:true, can_show_all:allowAll, can_collapse:branchExpanded,
                html:nested ? '<math><mi>nested</mi></math>' :
                  '<math><mrow data-symbolica-path="1"><mi>p</mi><mo>+</mo>'
                  +'<mtext data-symbolica-marker="0">[1]</mtext></mrow></math>',
                markers:nested ? [] : [[[1], 'Inspect nested sum [1]']],
                marker_ids:nested ? [] : [0], marker_starts:nested ? [] : [11]}];
            }
            if (['expand', 'expand_all', 'collapse'].includes(message.action)) {
              if (message.action === 'collapse' && message.value == null) {
                branchExpanded = false;
                return ['page', window.initialCollapsePage || fixture];
              }
              branchExpanded = message.action !== 'collapse';
              const response = ['page', {...fixture, can_collapse:branchExpanded,
                html:'<math><mrow data-symbolica-path="2"><mi>x</mi><mo>+</mo>'
                  +'<mrow data-symbolica-path="2/7"><mi>new_part</mi></mrow>'
                  +'<mo>+</mo><mtext data-symbolica-marker="0">[1]</mtext></mrow></math>',
                marker_starts:[10]}];
              if (window.compactExpansion) {
                response[1].html = '<math><mi>' + 'fully_visible_detail_'.repeat(8) + '</mi></math>';
                response[1].layouts = [{layout:1, html:'<math><mi>compact</mi></math>',
                  markers:[], marker_ids:[], marker_starts:[]}];
                response[1].markers = response[1].marker_ids = response[1].marker_starts = [];
              }
              if (window.expansionError) response[1].error = 'Expansion failed';
              return window.holdExpansion ? new Promise(resolve => {
                window.finishExpansion = () => resolve(response);
              }) : response;
            }
          }});
        }''', {'source': JAVASCRIPT, 'fixture': fixture()})
        await self.settle()

    async def asyncTearDown(self):
        await self.page.evaluate('() => viewer.close()')
        await self.browser.close()
        await self.playwright.stop()
        self.assertEqual(self.errors, [])

    async def settle(self):
        await self.page.evaluate('''() => new Promise(resolve =>
          requestAnimationFrame(() => requestAnimationFrame(resolve)))''')

    async def open_popup(self):
        await self.page.locator('.math-content button.omission').click()
        await self.page.wait_for_function('''() => {
          const popup = document.querySelector('#host').shadowRoot.querySelector('.preview');
          return !popup.hidden && !popup.hasAttribute('aria-busy');
        }''')

    async def zoom(self):
        return await self.page.locator('.math-content').evaluate(
            "node => Number(node.style.getPropertyValue('--expression-zoom'))")

    async def test_main_view_keyboard_zoom_and_editor_isolation(self):
        # Clicking actual MathML must focus the pager, including complete outputs.
        await self.page.evaluate('''() => viewer.update(['page', {...fixture,
          html:'<math><mi>complete_expression</mi></math>', markers:[], marker_ids:[], marker_starts:[]}])''')
        await self.page.locator('.math-content mi').click()
        await self.page.keyboard.press('+')
        self.assertEqual(await self.zoom(), 1.25)
        await self.page.keyboard.press('-')
        self.assertEqual(await self.zoom(), 1)
        await self.page.keyboard.press('=')
        await self.page.keyboard.press('0')
        self.assertEqual(await self.zoom(), 1)
        await self.page.keyboard.press('Control++')
        self.assertEqual(await self.zoom(), 1)
        await self.page.locator('#editor').fill('a')
        await self.page.keyboard.type('+-0')
        self.assertEqual(await self.page.locator('#editor').input_value(), 'a+-0')
        self.assertEqual(await self.zoom(), 1)
        self.assertIsNone(await self.page.locator('.math').get_attribute('title'))

    async def test_popup_opens_omitted_tail_and_nested_tail(self):
        await self.open_popup()
        self.assertEqual(await self.page.evaluate(
            "sent.filter(m => m.action === 'preview' && m.popup).at(-1).start"), 7)
        self.assertIn('terms 8–10', await self.page.locator('.preview-heading').text_content())
        await self.page.locator('.preview-math button.omission').click()
        await self.page.wait_for_function("() => sent.filter(m => m.action === 'preview').at(-1).context != null")
        self.assertEqual(await self.page.evaluate("sent.filter(m => m.action === 'preview').at(-1).start"), 11)
        self.assertTrue(await self.page.locator('.preview').evaluate(
            'node => node.getRootNode().activeElement === node'))

    async def test_page_size_keeps_omitted_position_and_previous_page_is_nearby(self):
        await self.open_popup()
        await self.page.locator('.preview-size').select_option('10')
        await self.page.wait_for_function("() => sent.filter(m => m.action === 'preview').at(-1).size === 10")
        self.assertEqual(await self.page.evaluate("sent.filter(m => m.action === 'preview').at(-1).start"), 7)
        await self.page.locator('.preview-size').select_option('5')
        await self.page.locator('.preview-previous').click()
        self.assertEqual(await self.page.evaluate("sent.filter(m => m.action === 'preview').at(-1).start"), 2)

    async def test_popup_keyboard_zoom_preserves_dialog_and_focus(self):
        await self.open_popup()
        await self.page.keyboard.press('+')
        self.assertEqual(await self.zoom(), 1.25)
        self.assertTrue(await self.page.locator('.preview').is_visible())
        await self.page.keyboard.press('-')
        self.assertEqual(await self.zoom(), 1)
        await self.page.locator('.preview-size').focus()
        await self.page.keyboard.press('+')
        self.assertEqual(await self.zoom(), 1)
        await self.page.keyboard.press('Escape')
        self.assertTrue(await self.page.locator('.math-content button.omission').evaluate(
            'node => node.getRootNode().activeElement === node'))

    async def test_bounded_show_all_and_branch_collapse(self):
        await self.open_popup()
        await self.page.locator('.preview-show-all').click()
        await self.page.wait_for_function("() => sent.some(m => m.action === 'expand_all')")
        message = await self.page.evaluate("sent.find(m => m.action === 'expand_all')")
        self.assertEqual(message['value'], [2])
        await self.open_popup()
        await self.page.locator('.preview-collapse').click()
        message = await self.page.evaluate("sent.find(m => m.action === 'collapse')")
        self.assertEqual(message['value'], [2])
        self.assertIn('layout', message)
        await self.page.evaluate('() => { window.allowAll = false; }')
        await self.open_popup()
        self.assertFalse(await self.page.locator('.preview-show-all').is_visible())

    async def test_show_all_retains_complete_layout_when_narrow_and_resized(self):
        await self.page.evaluate('''() => {
          window.compactExpansion = true;
          document.querySelector('#host').style.width = '220px';
        }''')
        await self.open_popup()
        await self.page.locator('.preview-show-all').click()
        await self.page.wait_for_function("() => sent.some(m => m.action === 'expand_all')")
        await self.settle()
        self.assertIn('fully_visible_detail_', await self.page.locator('.math-content').text_content())
        self.assertEqual(await self.page.locator('.math-content').get_attribute('data-layout'), '0')
        await self.page.locator('#host').evaluate("node => node.style.width = '180px'")
        await self.settle()
        self.assertIn('fully_visible_detail_', await self.page.locator('.math-content').text_content())
        # A fresh unrelated page returns to normal width-based selection.
        await self.page.evaluate('''() => viewer.update(['page', {...fixture,
          html:'<math><mi>' + 'wide_item_'.repeat(20) + '</mi></math>',
          layouts:[{layout:1, html:'<math><mi>compact</mi></math>', markers:[], marker_ids:[]}]}])''')
        self.assertEqual(await self.page.locator('.math-content').get_attribute('data-layout'), '1')

    async def test_failed_show_all_does_not_force_full_layout(self):
        await self.page.evaluate('''() => {
          window.compactExpansion = window.expansionError = true;
          document.querySelector('#host').style.width = '220px';
        }''')
        await self.open_popup()
        await self.page.locator('.preview-show-all').click()
        await self.page.wait_for_function("() => sent.some(m => m.action === 'expand_all')")
        await self.settle()
        self.assertEqual(await self.page.locator('.math-content').get_attribute('data-layout'), '1')

    async def test_collapse_all_remains_available_after_last_marker_disappears(self):
        await self.page.evaluate('''() => {
          window.compactExpansion = true;
          document.querySelector('#host').style.width = '220px';
          window.initialCollapsePage = {...fixture,
            html:'<math><mi>' + 'wide_initial_detail_'.repeat(8) + '</mi></math>',
            layouts:[{...fixture, layout:1}]};
          viewer.update(['page', initialCollapsePage]);
        }''')
        await self.settle()
        self.assertEqual(await self.page.locator('.math-content').get_attribute('data-layout'), '1')
        self.assertFalse(await self.page.locator('.collapse-all').is_visible())
        await self.open_popup()
        await self.page.locator('.preview-show-all').click()
        await self.page.wait_for_function("() => sent.some(m => m.action === 'expand_all')")
        await self.settle()
        self.assertEqual(await self.page.locator('.math-content button.omission').count(), 0)
        self.assertEqual(await self.page.locator('.math-content').get_attribute('data-layout'), '0')
        await self.page.locator('.collapse-all').click()
        await self.page.wait_for_function("() => sent.some(m => m.action === 'collapse')")
        await self.settle()
        message = await self.page.evaluate("sent.find(m => m.action === 'collapse')")
        self.assertIsNone(message.get('value'))
        self.assertEqual(await self.page.locator('.math-content').get_attribute('data-layout'), '1')
        self.assertEqual(await self.page.locator('.math-content button.omission').count(), 1)
        self.assertFalse(await self.page.locator('.collapse-all').is_visible())
        self.assertTrue(await self.page.locator('.math').evaluate(
            'node => node.getRootNode().activeElement === node'))

    async def test_inspect_follows_current_popup_page(self):
        await self.open_popup()
        await self.page.locator('.preview-next').click()
        await self.page.wait_for_function("() => sent.filter(m => m.action === 'preview').at(-1).start === 10")
        await self.page.locator('.preview-inspect').click()
        message = await self.page.evaluate("sent.find(m => m.action === 'open')")
        self.assertEqual(message['start'], 10)
        self.assertEqual(message['value'], [2])

    async def test_expansion_restores_focus_and_scroll_context(self):
        await self.page.evaluate('''() => {
          document.body.style.height = '3000px';
          document.querySelector('#host').style.marginTop = '1200px';
          window.scrollTo(0, 1000);
        }''')
        await self.open_popup()
        before = await self.page.locator('.math-content button.omission').bounding_box()
        await self.page.locator('.preview-expand').click()
        await self.page.wait_for_function("() => sent.some(m => m.action === 'expand')")
        await self.settle()
        self.assertTrue(await self.page.evaluate('''() => {
          const root = document.querySelector('#host').shadowRoot;
          return root.querySelector('.math').contains(root.activeElement);
        }'''))
        after = await self.page.locator('[data-symbolica-path="2/7"]').bounding_box()
        self.assertAlmostEqual(after['y'], before['y'], delta=4)

    async def test_delayed_expansion_does_not_steal_editor_focus_or_scroll(self):
        await self.page.evaluate('''() => {
          window.holdExpansion = true;
          document.body.style.height = '3000px';
        }''')
        await self.open_popup()
        await self.page.locator('.preview-expand').click()
        await self.page.wait_for_function('() => !!window.finishExpansion')
        await self.page.locator('#editor').focus()
        await self.page.evaluate('() => window.scrollTo(0, 200)')
        await self.page.evaluate('() => finishExpansion()')
        await self.settle()
        self.assertTrue(await self.page.locator('#editor').evaluate(
            'node => node.ownerDocument.activeElement === node'))
        self.assertEqual(await self.page.evaluate('scrollY'), 200)


if __name__ == '__main__':
    unittest.main()
