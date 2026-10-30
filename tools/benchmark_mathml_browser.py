"""Measure pager DOM/layout time using saved viewer.page() JSON fixtures.

Install Playwright and Chromium, then compare two JS revisions with identical
fixtures, font and browser. Example:
    python tools/benchmark_mathml_browser.py fixtures --javascript old.js --output before.json
    python tools/benchmark_mathml_browser.py fixtures --output after.json

Fixtures are named NAME-page.json; --case NAME selects a subset. This measures
synchronous browser work, excluding backend/Pyodide generation and GPU painting.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixtures', type=Path)
    parser.add_argument('--javascript', type=Path, default=Path(__file__).resolve().parents[1]
                        / 'src/api/python/display/paging.js')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--browser', help='Optional Chromium executable path')
    parser.add_argument('--font', type=Path, help='Local STIX Two Math WOFF2; otherwise use fallback fonts')
    parser.add_argument('--case', action='append', help='Fixture NAME, without -page.json; repeatable')
    parser.add_argument('--widths', default='1060,430,1060')
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--profile', action='store_true', help='Also count/timestamp wrapping and refinement calls')
    args = parser.parse_args()
    widths = [int(width) for width in args.widths.split(',')]
    if args.samples < 1 or not widths or min(widths) < 1:
        parser.error('samples and widths must be positive')
    paths = ([args.fixtures / f'{name}-page.json' for name in args.case] if args.case
             else sorted(args.fixtures.glob('*-page.json')))
    if not paths:
        parser.error('no NAME-page.json fixtures found')
    source = args.javascript.read_text()
    report = {'javascript_sha256': hashlib.sha256(source.encode()).hexdigest(),
              'font_sha256': hashlib.sha256(args.font.read_bytes()).hexdigest() if args.font else None,
              'samples': args.samples, 'profile': args.profile,
              'note': 'Synchronous Chromium DOM/style/layout; excludes backend generation and GPU painting.',
              'cases': {}, 'errors': []}
    if args.profile:
        for name in ('setWrapWidth', 'revealFittingParts', 'fitTable'):
            declaration = f'function {name}('
            assert source.count(declaration) == 1, f'Cannot instrument {name}'
            source = source.replace(declaration, f'''function {name}(...args) {{
              const start=performance.now();
              try {{ return __benchmark_{name}(...args); }} finally {{
                const entry=(globalThis.__pagerBench ||= {{}})['{name}'] ||= {{calls:0,ms:0}};
                entry.calls++;entry.ms+=performance.now()-start;
              }}
            }}
            function __benchmark_{name}(''')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(executable_path=args.browser)
        report['browser_version'] = browser.version
        page = browser.new_page(viewport={'width': max(widths) + 100, 'height': 1000})
        page.on('pageerror', lambda error: report['errors'].append(str(error)))
        page.route('https://cdn.jsdelivr.net/**', lambda route:
                   route.fulfill(path=args.font, content_type='font/woff2') if args.font else route.abort())
        session = page.context.new_cdp_session(page)
        session.send('Performance.enable')

        def metrics():
            return {metric['name']: metric['value'] for metric in
                    session.send('Performance.getMetrics')['metrics']}

        page.set_content('<style>body{margin:0;padding:20px}</style><div id="host"></div>')
        page.evaluate('''async source => {
          const url=URL.createObjectURL(new Blob([source],{type:'text/javascript'}));
          window.mount=(await import(url)).mount;URL.revokeObjectURL(url);
        }''', source)
        for path in paths:
            fixture_bytes = path.read_bytes()
            fixture = json.loads(fixture_bytes)
            results = []
            for index, width in enumerate(widths):
                page.evaluate('''async ({fixture,width,first,font}) => {
                  const host=document.querySelector('#host');
                  if(first) await window.viewer?.close();
                  host.style.width=`${width}px`;
                  if(first) window.viewer=mount({el:host,page:fixture,send:()=>null});
                  await document.fonts.ready;
                  if(font&&!document.fonts.check('21px "STIX Two Math"')) throw Error('STIX not loaded');
                  await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(()=>requestAnimationFrame(resolve))));
                }''', {'fixture': fixture, 'width': width, 'first': index == 0, 'font': bool(args.font)})
                before = metrics()
                samples = page.evaluate('''({fixture,count}) => {
                  globalThis.__pagerBench={};const samples=[];
                  for(let i=0;i<count;i++) {
                    const start=performance.now();viewer.update(['page',fixture]);
                    document.querySelector('#host').getBoundingClientRect();
                    samples.push(performance.now()-start);
                  }
                  return samples;
                }''', {'fixture': fixture, 'count': args.samples})
                after = metrics()
                state = page.evaluate('''() => {
                  const root=document.querySelector('#host').shadowRoot;
                  const content=root.querySelector('.math-content');
                  return {layout:Number(content.dataset.layout),height:content.getBoundingClientRect().height,
                    scroll_width:content.scrollWidth,available:root.querySelector('.math').clientWidth,
                    dom_elements:content.querySelectorAll('*').length,
                    markers:[...content.querySelectorAll('button.omission')].map(b=>({
                      text:b.textContent,layout:b.dataset.layout,label:b.getAttribute('aria-label')})),
                    text:content.textContent,profile:globalThis.__pagerBench};
                }''')
                results.append({'width': width, 'samples_ms': samples, 'median_ms': statistics.median(samples),
                                'browser_metrics': {name: after[name] - before[name] for name in
                                                    ('LayoutCount', 'RecalcStyleCount', 'LayoutDuration',
                                                     'RecalcStyleDuration', 'TaskDuration')}, **state})
            report['cases'][path.name.removesuffix('-page.json')] = {
                'fixture_sha256': hashlib.sha256(fixture_bytes).hexdigest(), 'widths': results}
        page.evaluate('viewer.close()')
        browser.close()
    output = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end='')
    raise SystemExit(bool(report['errors']))


if __name__ == '__main__':
    main()
