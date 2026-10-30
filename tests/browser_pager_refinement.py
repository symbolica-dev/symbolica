"""Regression tests for repeated fitting and rejected detail refinements.

Run with python tests/browser_pager_refinement.py. No native build is needed.
SYMBOLICA_TEST_ENGINES selects comma-separated Playwright engines (default:
chromium,firefox,webkit); *_EXECUTABLE and SYMBOLICA_TEST_FONT select local browsers/font.
"""

import os
import unittest
from pathlib import Path

from browser_mathml_wrapping import fraction_page, product_page
from playwright.async_api import async_playwright

JAVASCRIPT = (Path(__file__).resolve().parents[1]
              / 'src/api/python/display/paging.js').read_text()

# Resize updates rebuild the payload. Expose only the existing fitter and
# refinement operation so these tests also exercise the same decorated DOM.
HOOK = '''
  el.testPager = (() => {
    const target = scope => scope === 'preview' ? previewMath : content;
    const width = scope => scope === 'preview' ? previewMath.clientWidth - 4 : math.clientWidth;
    const rect = node => {
      const r = node.getBoundingClientRect();
      return [r.x, r.y, r.width, r.height];
    };
    const snapshot = scope => {
      const node = target(scope);
      return {html:node.innerHTML, rect:rect(node), scroll:[node.scrollWidth,node.scrollHeight],
        prefactors:node.querySelectorAll('[data-symbolica-product-prefactor]').length,
        signs:[...node.querySelectorAll('mo')].filter(n=>n.textContent==='−').length,
        fractions:[...node.querySelectorAll('mfrac')].map(n=>[...n.children].map(c=>c.textContent)),
        rows:[...node.querySelectorAll('mtable')].map(t=>[...t.children].map(row=>({
          rect:rect(row), text:row.textContent}))),
        markers:[...node.querySelectorAll('button.omission')].map(b=>({
          ...inspection.get(b), text:b.textContent, layout:b.dataset.layout, rect:rect(b)}))};
    };
    const descendants = node => {
      const walker = doc.createTreeWalker(node, NodeFilter.SHOW_ALL), nodes=[node];
      while (walker.nextNode()) nodes.push(walker.currentNode);
      return nodes;
    };
    return {
      prepare(scope) {setWrapWidth(target(scope), width(scope));},
      reuseFittingRows() {
        const before=snapshot('root');
        const rows=[...content.querySelectorAll('mtable > mtr')];
        setWrapWidth(content,width('root'));
        const after=snapshot('root');
        const kept=[...content.querySelectorAll('mtable > mtr')];
        el.style.width='110px';setWrapWidth(content,width('root'));
        const narrow=content.querySelectorAll('mtable > mtr').length;
        el.style.width='1100px';setWrapWidth(content,width('root'));
        return {before,after,narrow,wide:snapshot('root'),
          reused:rows.length===kept.length&&rows.every((row,i)=>row===kept[i])};
      },
      repeat(scope) {
        const states=[snapshot(scope)];
        for(let i=0;i<3;i++) {setWrapWidth(target(scope),width(scope));states.push(snapshot(scope));}
        return states;
      },
      resizeRoundTrip(scope) {
        const frame=scope==='preview'?preview:el, previous=frame.style.width;
        frame.style.width='900px';setWrapWidth(target(scope),width(scope));
        const wide=snapshot(scope);
        frame.style.width=previous;setWrapWidth(target(scope),width(scope));
        return {wide,restored:snapshot(scope)};
      },
      reject(layouts, scope, context, narrow=false) {
        const node=target(scope), nodes=descendants(node);
        const buttons=[...node.querySelectorAll('button.omission')];
        const bindings=buttons.map(b=>[b.onclick,b.onmouseenter,b.onfocus,inspection.get(b)]);
        const before=snapshot(scope);
        const observer=new MutationObserver(()=>{});
        observer.observe(node,{childList:true,subtree:true});
        // Model an enclosing scrollbar reducing the available width during
        // the first rejected trial, independently of OS scrollbar settings.
        const frame=scope==='preview'?previewMath:math;
        const initialWidth=frame.clientWidth;
        let reads=0;
        if(narrow) Object.defineProperty(frame,'clientWidth',{configurable:true,
          get:()=>initialWidth-(reads++?20:0)});
        revealFittingParts(layouts,width(scope),node,context);
        if(narrow) delete frame.clientWidth;
        const trialNodes=new Set(observer.takeRecords().flatMap(record=>[...record.addedNodes])
          .filter(n=>n.nodeType===1&&n.matches('[data-test-detail]')));
        const trialInserted=trialNodes.size>0;
        observer.disconnect();
        const restored=descendants(node);
        return {before,after:snapshot(scope),trialInserted,trialCount:trialNodes.size,
          sameNodes:nodes.length===restored.length&&nodes.every((n,i)=>n===restored[i]),
          sameBindings:buttons.every((b,i)=>b.isConnected&&b.onclick===bindings[i][0]
            &&b.onmouseenter===bindings[i][1]&&b.onfocus===bindings[i][2]
            &&inspection.get(b)===bindings[i][3])};
      }
    };
  })();
'''
ANCHOR = '  let lastWidth = -1, frame;'
assert JAVASCRIPT.count(ANCHOR) == 1
JAVASCRIPT = JAVASCRIPT.replace(ANCHOR, HOOK + '\n' + ANCHOR)


def nested_sum_page():
    """A narrow continuation changes how many terms fit in the middle factor."""
    widths = [220.671875, 255.515625, 246.46875, 256.875, 311.625,
              302.59375, 311.625, 312.984375, 314.34375, 37.125]
    terms = ''.join(f'<mrow data-symbolica-term=""><mspace width="{width}px" '
                    'height="30px"/></mrow>' for width in widths)
    inner = ('<mtable data-symbolica-sum="" columnalign="left" rowspacing="0.35em">'
             '<mtr><mtd><mrow>' + terms + '</mrow></mtd></mtr></mtable>')
    separator = '<mo>&#x2062;</mo><mspace width="0.167em"/>'
    factors = ['<msup><mi>G</mi><mn>8</mn></msup>',
               separator + '<mrow><mo>(</mo><mrow>' + inner + '</mrow><mo>)</mo></mrow>',
               separator + '<msup><msub><mi>C</mi><mi>A</mi></msub><mn>4</mn></msup>']
    fixture = product_page()
    fixture.update(html='<math displaystyle="true"><mrow><mo>−</mo>'
                   '<mtable data-symbolica-product="" data-symbolica-row-gap="0.5em" rowspacing="0">'
                   '<mtr><mtd style="padding:0"><mrow>'
                   + ''.join('<mrow data-symbolica-factor="">' + factor + '</mrow>' for factor in factors)
                   + '</mrow></mtd></mtr></mtable></mrow></math>', end=3, total=3)
    return fixture


def refinement_pages():
    compact = fraction_page(coefficient=True)
    leaf = '<msub><mi>a</mi><mn>4</mn></msub>'
    part = ('<mrow data-symbolica-path="4/1"><mi>q</mi><mo>+</mo>'
            '<mtext data-symbolica-marker="0">[1]</mtext></mrow>')
    compact.update(html=compact['html'].replace(leaf, part), layout=3,
                   markers=[[[4, 1], 'Inspect sum [1]']], marker_ids=[0], marker_starts=[7])
    detailed = dict(compact, layout=0, html=compact['html'].replace(
        part, '<mrow data-symbolica-path="4/1" data-test-detail=""><mspace width="1800px" height="20px"/>'
        '<mtext data-symbolica-marker="0">[1]</mtext></mrow>'), marker_starts=[11])
    return compact, detailed


def short_factor_product(count=6):
    """Several short binomials: finishing the product costs just one more row."""
    def html(compact):
        factors = []
        for i in range(count):
            detail = (f'<mtext data-symbolica-marker="{i}">[{i + 1}]</mtext>' if compact else
                      '<mspace width="180px" height="21px"/><mi>k</mi><mi>γ</mi>')
            factors.append('<mrow data-symbolica-factor=""><mrow><mo>(</mo>'
                           f'<mrow data-symbolica-path="{i}"><mi>M</mi><mi>g</mi>'
                           f'<mo>+</mo>{detail}</mrow><mo>)</mo></mrow></mrow>')
        return ('<math displaystyle="true"><mtable data-symbolica-product="" '
                'data-symbolica-row-gap="0.5em" columnalign="left" rowspacing="0">'
                '<mtr><mtd style="padding:0"><mrow>' + ''.join(factors)
                + '</mrow></mtd></mtr></mtable></math>')

    page = dict(product_page(), html=html(False), end=count, total=count)
    page['layouts'] = [{'html': html(True), 'layout': 1,
                        'markers': [[[i], f'Inspect sum [{i + 1}]'] for i in range(count)],
                        'marker_ids': list(range(count)), 'marker_starts': [1] * count}]
    return page


def font_handler(font):
    return lambda route: (route.fulfill(path=font, content_type='font/woff2')
                          if font else route.abort())


class PagerRefinementTests(unittest.IsolatedAsyncioTestCase):
    async def browser_cases(self, check):
        async with async_playwright() as playwright:
            for engine in os.environ.get('SYMBOLICA_TEST_ENGINES', 'chromium,firefox,webkit').split(','):
                with self.subTest(engine=engine):
                    browser = await getattr(playwright, engine).launch(
                        headless=True, executable_path=os.environ.get(f'{engine.upper()}_EXECUTABLE'))
                    try:
                        page = await browser.new_page(viewport={'width': 1200, 'height': 1000})
                        errors = []
                        page.on('pageerror', lambda error, errors=errors: errors.append(str(error)))
                        font = os.environ.get('SYMBOLICA_TEST_FONT')
                        await page.route('https://cdn.jsdelivr.net/**', font_handler(font))
                        await page.set_content('<div id="host"></div>')
                        await page.evaluate('''async source => {
                          const url=URL.createObjectURL(new Blob([source],{type:'text/javascript'}));
                          window.mount=(await import(url)).mount;URL.revokeObjectURL(url);
                        }''', JAVASCRIPT)
                        await check(page)
                        await page.evaluate('window.viewer?.close()')
                        self.assertFalse(errors, errors)
                    finally:
                        await browser.close()

    async def mount(self, page, fixture, width, preview=None):
        await page.evaluate('''async ({fixture,width,preview}) => {
          await window.viewer?.close();window.sent=[];
          const host=document.querySelector('#host');host.style.width=`${width}px`;
          window.viewer=mount({el:host,page:fixture,send:message=>{
            sent.push(message);
            if(message.action==='preview'&&preview) return ['preview',{
              ...preview,request:message.request,context:'popup-origin'}];
          }});
          await document.fonts.ready;
          await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
        }''', {'fixture': fixture, 'width': width, 'preview': preview})

    async def test_repeated_fits_keep_the_same_nested_rows_fractions_and_cells(self):
        await self.browser_cases(self.check_repeated_fits)

    async def test_complete_products_get_a_small_height_allowance(self):
        async def check(page):
            for count, total, whole in [(6, 6, True), (6, 100, False), (12, 12, False)]:
                fixture = dict(short_factor_product(count), total=total)
                await self.mount(page, fixture, 490)
                state = await page.evaluate("document.querySelector('#host').testPager.repeat('root')[0]")
                with self.subTest(count=count, total=total):
                    self.assertEqual(not state['markers'], whole, state)
                    self.assertLessEqual(state['scroll'][0], 491)
                    if whole:
                        self.assertGreater(state['rect'][3], 252)
                        self.assertLessEqual(state['rect'][3], 315)
                    else:
                        self.assertLessEqual(state['rect'][3], 253)

        await self.browser_cases(check)

    async def test_fitting_single_rows_are_retained_but_still_reflow_on_resize(self):
        async def check(page):
            product = product_page()
            terms = ''.join('<mrow data-symbolica-term=""><mo>+</mo>'
                            f'<mi>x</mi><mn>{i}</mn></mrow>' for i in range(8))
            total = dict(product, html='<math displaystyle="true"><mrow><mo>(</mo>'
                         '<mtable data-symbolica-sum=""><mtr><mtd><mrow>' + terms
                         + '</mrow></mtd></mtr></mtable><mo>)</mo></mrow></math>')
            for name, fixture in [('product', product), ('nested sum', total)]:
                with self.subTest(fixture=name):
                    await self.mount(page, fixture, 1100)
                    result = await page.evaluate("document.querySelector('#host').testPager.reuseFittingRows()")
                    self.assertTrue(result['reused'], 'An already fitting row was rebuilt')
                    self.assertEqual(result['after'], result['before'])
                    # A narrow rebuild normalizes CSS serialization (padding:0
                    # becomes padding: 0px); all geometry and notation match.
                    self.assertEqual({k: v for k, v in result['wide'].items() if k != 'html'},
                                     {k: v for k, v in result['before'].items() if k != 'html'})
                    self.assertGreater(result['narrow'], 1)

        await self.browser_cases(check)

    async def check_repeated_fits(self, page):
        matrix = fraction_page(coefficient=True)
        matrix['html'] = ('<div data-symbolica-cell="" style="width:254.125px">'
                          + matrix['html'] + '</div>')
        cases = [('nested sum', nested_sum_page(), (1060, 430)),
                 ('nested product', product_page('nested'), (430, 190)),
                 ('rational coefficient', fraction_page(coefficient=True), (900, 260)),
                 ('large coefficient', fraction_page(coefficient='large'), (900, 190)),
                 ('fractional matrix cell', matrix, (1060, 430))]
        for name, fixture, widths in cases:
            for width in widths:
                with self.subTest(fixture=name, width=width):
                    await self.mount(page, fixture, width)
                    states = await page.evaluate("document.querySelector('#host').testPager.repeat('root')")
                    for state in states[1:]:
                        self.assertEqual(state, states[0])
                    self.assertLessEqual(states[0]['scroll'][0], width + 1)

    async def test_identical_rejected_details_are_tried_once_per_refinement(self):
        await self.browser_cases(self.check_identical_rejected_details)

    async def check_identical_rejected_details(self, page):
        compact, detailed = refinement_pages()
        # No marker inside the replacement: its actual decorated DOM is the
        # same in both layouts, including its labels and navigation context.
        detailed['html'] = detailed['html'].replace(
            '<mtext data-symbolica-marker="0">[1]</mtext>', '')
        duplicate = dict(detailed, layout=1)
        fitting = dict(detailed, layout=2, html=detailed['html'].replace('1800px', '40px'))
        anchor = dict(compact, html='<math><mrow data-symbolica-path="9"><mi>x</mi><mo>+</mo>'
                      '<mtext data-symbolica-marker="0">[1]</mtext></mrow></math>',
                      markers=[[[9], 'Inspect outer [1]']], marker_starts=[2])
        for scope in ('root', 'preview'):
            with self.subTest(scope=scope):
                await self.mount(page, compact if scope == 'root' else anchor, 260,
                                 preview=compact if scope == 'preview' else None)
                if scope == 'preview':
                    await page.locator('#host').evaluate('''host =>
                      host.shadowRoot.querySelector('.math-content button.omission').click()''')
                    await page.wait_for_function('''() => {
                      const popup=document.querySelector('#host').shadowRoot.querySelector('.preview');
                      return !popup.hidden&&!popup.hasAttribute('aria-busy');
                    }''')
                await page.evaluate("scope=>document.querySelector('#host').testPager.prepare(scope)", scope)
                # Each new refinement must retry: no persistent measurement
                # cache may outlive a width/font or surrounding-DOM change.
                for _ in range(2):
                    result = await page.evaluate('''({layouts,scope}) =>
                      document.querySelector('#host').testPager.reject(layouts,scope,
                        scope==='preview'?'popup-origin':null)''',
                                                {'layouts': [detailed, duplicate, compact], 'scope': scope})
                    self.assertEqual(result['trialCount'], 1)
                    self.assertEqual(result['after'], result['before'])
                    self.assertTrue(result['sameNodes'])
                    self.assertTrue(result['sameBindings'])
                # Identical source markup with different decorated marker
                # metadata is not an interchangeable navigation candidate.
                _, marked = refinement_pages()
                result = await page.evaluate('''({layouts,scope}) =>
                  document.querySelector('#host').testPager.reject(layouts,scope,
                    scope==='preview'?'popup-origin':null)''',
                                            {'layouts': [marked, dict(marked, layout=1), compact],
                                             'scope': scope})
                self.assertEqual(result['trialCount'], 2)
                self.assertEqual(result['after'], result['before'])
                result = await page.evaluate('''({layouts,scope}) =>
                  document.querySelector('#host').testPager.reject(layouts,scope,
                    scope==='preview'?'popup-origin':null)''',
                                            {'layouts': [detailed, duplicate, fitting, compact], 'scope': scope})
                self.assertEqual(result['trialCount'], 2)
                self.assertFalse(result['after']['markers'])
                self.assertIn('width="40px"', result['after']['html'])
                await self.mount(page, compact if scope == 'root' else anchor, 260,
                                 preview=compact if scope == 'preview' else None)
                if scope == 'preview':
                    await page.locator('#host').evaluate('''host =>
                      host.shadowRoot.querySelector('.math-content button.omission').click()''')
                    await page.wait_for_function('''() => {
                      const popup=document.querySelector('#host').shadowRoot.querySelector('.preview');
                      return !popup.hidden&&!popup.hasAttribute('aria-busy');
                    }''')
                await page.evaluate("scope=>document.querySelector('#host').testPager.prepare(scope)", scope)
                result = await page.evaluate('''({layouts,scope}) =>
                  document.querySelector('#host').testPager.reject(layouts,scope,
                    scope==='preview'?'popup-origin':null,true)''',
                                            {'layouts': [detailed, duplicate, compact], 'scope': scope})
                self.assertEqual(result['trialCount'], 2,
                                 'A changed width must not reuse the earlier rejection')

    async def test_rejected_refinements_restore_nodes_bindings_and_popup_context(self):
        await self.browser_cases(self.check_rejected_refinements)

    async def check_rejected_refinements(self, page):
        compact, detailed = refinement_pages()
        anchor = dict(compact, html='<math><mrow data-symbolica-path="9"><mi>x</mi><mo>+</mo>'
                      '<mtext data-symbolica-marker="0">[1]</mtext></mrow></math>',
                      markers=[[[9], 'Inspect outer [1]']], marker_starts=[2])
        for scope in ('root', 'preview'):
            with self.subTest(scope=scope):
                await self.mount(page, compact if scope == 'root' else anchor, 260,
                                 preview=compact if scope == 'preview' else None)
                if scope == 'preview':
                    await page.locator('#host').evaluate('''host =>
                      host.shadowRoot.querySelector('.math-content button.omission').click()''')
                    await page.wait_for_function('''() => {
                      const popup=document.querySelector('#host').shadowRoot.querySelector('.preview');
                      return !popup.hidden&&!popup.hasAttribute('aria-busy');
                    }''')
                    await page.locator('#host').evaluate('''host =>
                      host.shadowRoot.querySelector('.preview').style.width='310px' ''')
                # Popup refinement uses its inner budget, four pixels below
                # clientWidth. Match that budget before testing its rollback.
                await page.evaluate("scope=>document.querySelector('#host').testPager.prepare(scope)", scope)
                result = await page.evaluate('''({layouts,scope}) =>
                  document.querySelector('#host').testPager.reject(layouts,scope,
                    scope==='preview'?'popup-origin':null)''',
                                            {'layouts': [detailed, compact], 'scope': scope})
                self.assertEqual(result['after'], result['before'])
                self.assertTrue(result['trialInserted'], 'The richer subtree was never tried')
                self.assertTrue(result['sameNodes'], 'Rejected refinement replaced original nodes')
                self.assertTrue(result['sameBindings'], 'Rejected refinement lost a marker binding')
                self.assertEqual(result['after']['prefactors'], 1)
                self.assertEqual(result['after']['signs'], 1)
                self.assertEqual(result['after']['fractions'], [['3', '8']])
                self.assertEqual([(m['path'], m['layout'], m['start']) for m in result['after']['markers']],
                                 [([4, 1], '3', 7)])
                # A later refit must still restore the original rational
                # fraction correctly; its bookkeeping includes detached nodes.
                states = await page.evaluate("scope=>document.querySelector('#host').testPager.repeat(scope)", scope)
                for state in states:
                    self.assertEqual(state, result['before'])
                # Change representation on these same nodes, without a resize
                # update rebuilding the payload between the two measurements.
                resized = await page.evaluate("scope=>document.querySelector('#host').testPager.resizeRoundTrip(scope)", scope)
                self.assertEqual(resized['wide']['prefactors'], 0)
                self.assertEqual(resized['wide']['signs'], 1)
                self.assertEqual(len(resized['wide']['fractions']), 1)
                self.assertEqual(resized['wide']['fractions'][0][1], '8')
                self.assertTrue(resized['wide']['fractions'][0][0].startswith('3'))
                self.assertEqual([(m['path'], m['layout'], m['start']) for m in resized['wide']['markers']],
                                 [([4, 1], '3', 7)])
                self.assertEqual(resized['restored'], result['before'])
                before = await page.evaluate('sent.length')
                selector = '.preview-math' if scope == 'preview' else '.math-content'
                await page.locator('#host').evaluate('''(host,selector) =>
                  host.shadowRoot.querySelector(`${selector} button.omission`).click()''', selector)
                await page.wait_for_function('before=>sent.slice(before).some(m=>m.action==="preview")', arg=before)
                request = await page.evaluate('before=>sent.slice(before).find(m=>m.action==="preview")', before)
                self.assertEqual((request['value'], request['layout'], request['start']), ([4, 1], 3, 7))
                self.assertEqual(request.get('context'), 'popup-origin' if scope == 'preview' else None)


    async def test_refined_matrix_entries_do_not_overlap_neighbouring_cells(self):
        await self.browser_cases(self.check_refined_matrix_entries)

    async def check_refined_matrix_entries(self, page):
        marker = '<mrow data-symbolica-path="0"><mtext data-symbolica-marker="0">[1]</mtext></mrow>'
        html = ('<div style="width:var(--symbolica-wrap-width)"><table '
                'style="width:100%;table-layout:fixed"><tr><td><div data-symbolica-cell="">'
                '<math>' + marker + '</math></div></td><td><div data-symbolica-cell="">'
                '<math><mn>123456789</mn></math></div></td></tr></table></div>')
        compact = {'html': html, 'start': 0, 'end': 2, 'total': 2, 'kind': 'Matrix', 'unit': 'entries',
                       'markers': [[[0], 'Inspect first cell [1]']], 'marker_ids': [0], 'marker_starts': [0],
                       'layout': 1, 'previous': False, 'next': None, 'back': False, 'page_size': 25}
        detailed = dict(compact, layout=0, markers=[], marker_ids=[], marker_starts=[],
                        html=html.replace(marker, '<mrow data-symbolica-path="0" data-test-detail="">'
                                          '<mspace width="300px" height="20px"/><mi>y</mi></mrow>'))
        fitting = dict(detailed, html=detailed['html'].replace('300px', '80px'))
        anchor = dict(compact, html='<math><mrow data-symbolica-path="9">'
                      '<mtext data-symbolica-marker="0">[1]</mtext></mrow></math>',
                      markers=[[[9], 'Inspect matrix [1]']])
        for scope in ('root', 'preview'):
            for width in (480, 360):
                with self.subTest(scope=scope, width=width):
                    await self.mount(page, compact if scope == 'root' else anchor, width,
                                     preview=compact if scope == 'preview' else None)
                    if scope == 'preview':
                        await page.locator('#host').evaluate("""host =>
                          host.shadowRoot.querySelector('.math-content button.omission').click()""")
                        await page.wait_for_function("""() => {
                          const popup=document.querySelector('#host').shadowRoot.querySelector('.preview');
                          return !popup.hidden&&!popup.hasAttribute('aria-busy');
                        }""")
                        await page.locator('#host').evaluate("""(host,width) =>
                          host.shadowRoot.querySelector('.preview').style.width=`${width}px`""", width)
                    await page.evaluate("scope=>document.querySelector('#host').testPager.prepare(scope)", scope)
                    result = await page.evaluate("""({layouts,scope}) =>
                      document.querySelector('#host').testPager.reject(layouts,scope,
                        scope==='preview'?'matrix-preview':null)""",
                                                 {'layouts': [detailed, compact], 'scope': scope})
                    self.assertEqual(result['trialCount'], 1)
                    self.assertEqual(result['after'], result['before'])
                    self.assertTrue(result['sameNodes'])
                    self.assertTrue(result['sameBindings'])
                    # A genuinely fitting richer entry remains eligible; the
                    # check must not disable refinement in composite outputs.
                    result = await page.evaluate("""({layouts,scope}) =>
                      document.querySelector('#host').testPager.reject(layouts,scope,
                        scope==='preview'?'matrix-preview':null)""",
                                                 {'layouts': [fitting, compact], 'scope': scope})
                    self.assertEqual(result['trialCount'], 1)
                    self.assertFalse(result['after']['markers'])
                    self.assertIn('<mi>y</mi>', result['after']['html'])
                    selector = '.preview-math' if scope == 'preview' else '.math-content'
                    self.assertTrue(await page.locator('#host').evaluate("""(host,selector) =>
                      [...host.shadowRoot.querySelector(selector).querySelectorAll('[data-symbolica-cell]')]
                        .every(cell=>cell.scrollWidth<=cell.clientWidth+1)""", selector))


if __name__ == '__main__':
    unittest.main()
