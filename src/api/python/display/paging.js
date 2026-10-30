function render({ model, el }) {
  // Register the full math font once in the document: @font-face declarations
  // inside a shadow root are not available to the browser's font loader.
  const doc = el.ownerDocument;
  if (!doc.getElementById('symbolica-notebook-math-font')) {
    const style = doc.createElement('style');
    style.id = 'symbolica-notebook-math-font';
    style.textContent = '@font-face{font-family:"STIX Two Math";src:local("STIX Two Math"),url("https://cdn.jsdelivr.net/gh/stipub/stixfonts@2.13b171/fonts/static_otf_woff2/STIXTwoMath-Regular.woff2") format("woff2");font-display:swap}';
    doc.head.append(style);
  }
  // Notebook themes reset native controls. Keep the pager's controls and layout
  // isolated while inheriting the host's light/dark theme colours.
  const root = el.shadowRoot || el.attachShadow({ mode: 'open' });
  root.innerHTML = `<style>
    :host {
      display:block; position:relative;
      contain:inline-size; width:100%; min-width:0; max-width:100%;
      color:var(--foreground,var(--jp-ui-font-color1,#202624));
      font:13px/1.5 system-ui,sans-serif;
    }
    * { box-sizing:border-box; }
    [hidden] { display:none!important; }
    .toolbar { display:flex; flex-wrap:wrap; align-items:center; gap:8px; }
    .toolbar { padding:8px 0; }
    button,select {
      font:inherit; color:inherit;
      background:var(--background,var(--jp-layout-color1,#fff));
      border:1px solid var(--border,var(--jp-border-color2,#dce2df));
      border-radius:6px; min-height:30px; padding:4px 10px;
      transition:background .12s,border-color .12s;
    }
    button {
      display:inline-flex; align-items:center; justify-content:center;
      cursor:pointer; font-weight:500;
    }
    button.page-arrow { width:30px; padding:5px; }
    button:hover:not(:disabled),select:hover:not(:disabled) {
      background:var(--muted,var(--jp-layout-color2,#f3f5f4));
    }
    button:disabled,select:disabled { opacity:.4; cursor:default; }
    button:focus-visible,select:focus-visible {
      outline:2px solid var(--primary,var(--jp-brand-color1,#16846b)); outline-offset:2px;
    }
    .status {
      color:var(--muted-foreground,var(--jp-ui-font-color2,#68716c));
      font-size:12px; padding-bottom:8px;
    }
    .math {
      overflow-x:auto; padding:12px 0;
      border-block:1px solid var(--border,var(--jp-border-color2,#e2e7e4));
    }
    .math.compact { border:0; padding:0; }
    .math:focus-visible {
      outline:2px solid var(--primary,var(--jp-brand-color1,#16846b)); outline-offset:3px;
    }
    /* The container scrolls; an inline MathML root must not get its own
       miniature scrollbars around fractions and tall delimiters. */
    math { font-family:"STIX Two Math",math; font-size:21px; font-synthesis:none; }
    /* MathML glyph ink can extend beyond its line box. Keep that space inside
       the zoomed content so fractions do not create a vertical scrollbar. */
    .math-content {
      width:fit-content; box-sizing:border-box; padding:.6em .1em;
      font-size:21px; zoom:var(--expression-zoom,1);
    }
    .math-content > math { overflow:visible!important; }
    .math-flow { max-width:var(--symbolica-wrap-width,min(650px,80vw)); }
    .math-flow > math {
      flex:0 0 auto; overflow:visible!important;
      font-family:inherit; font-size:inherit; font-synthesis:inherit;
    }
    /* MathML Core uses CSS alignment; columnalign alone is not portable. */
    mtable[data-symbolica-product] > mtr > mtd,
    mtable[data-symbolica-sum] > mtr > mtd { text-align:left; }
    [data-symbolica-continuation-separator] { display:none; }
    .zoom-feedback {
      position:absolute; top:4px; right:4px; padding:3px 8px; border-radius:4px;
      background:var(--background,var(--jp-layout-color1,#fff));
      box-shadow:0 1px 6px #0002; pointer-events:none; font-size:12px;
    }
    button.omission {
      font:500 13px/18px system-ui,sans-serif;
      box-sizing:border-box; min-width:20px; min-height:20px; padding:0 4px;
      color:var(--primary,var(--jp-brand-color1,#16846b));
      background:color-mix(in srgb,currentColor 8%,transparent);
      border:1px solid color-mix(in srgb,currentColor 40%,transparent);
      border-radius:3px; margin:0 1px;
      white-space:nowrap; vertical-align:baseline; transform:translateY(-1px);
      font-variant-numeric:tabular-nums;
    }
    button.omission:hover:not(:disabled),button.omission[aria-expanded="true"] {
      background:color-mix(in srgb,currentColor 17%,transparent);
    }
    button.omission:focus-visible {
      background:color-mix(in srgb,currentColor 17%,transparent);
    }
    .preview {
      position:fixed; inset:auto; margin:0; padding:12px;
      width:360px; max-width:calc(100vw - 24px);
      max-height:calc(100vh - 24px); overflow:auto;
      color:var(--foreground,var(--jp-ui-font-color1,#202624));
      background:var(--background,var(--jp-layout-color1,#fff));
      border:1px solid var(--border,var(--jp-border-color2,#dce2df));
      border-radius:8px; box-shadow:0 4px 20px #0002;
      font:13px/1.5 system-ui,sans-serif; z-index:10000;
    }
    .preview.pinned { width:720px; }
    .preview-header { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:8px; }
    .preview-heading { font-weight:600; }
    .preview-controls { display:flex; align-items:center; flex-wrap:wrap; gap:8px; margin-top:12px; }
    .preview.pinned .preview-math { max-height: min(55vh, 480px); }
    /* Math glyphs can extend beyond the inline line box. Leave room for that
       ink so a short preview doesn't acquire a tiny vertical scrollbar. */
    .preview-math { overflow:auto; max-height:160px; font-size:21px; padding:6px 2px; }
    .preview-math > math { overflow:visible!important; }
    .preview-hint { margin-top:8px; color:var(--muted-foreground,var(--jp-ui-font-color2,#68716c)); font-size:12px; }
  </style>
  <nav class="toolbar" aria-label="Expression controls"></nav>
  <div class="status" role="status" aria-live="polite"></div>
  <div class="math" tabindex="0" role="region" aria-label="Expression"
    aria-keyshortcuts="+ = - 0"
    ><div class="math-content"></div></div>
  <span class="zoom-feedback" aria-live="polite" hidden></span>
  <div class="preview" role="tooltip" popover="manual" hidden>
    <div class="preview-header">
      <div class="preview-heading" aria-live="polite"></div>
      <div><button class="preview-back" type="button" hidden>Back</button>
      <button class="preview-close" type="button" hidden>Close</button></div>
    </div>
    <div class="preview-math"></div>
    <nav class="preview-controls" aria-label="Subexpression pages" hidden>
      <button class="preview-previous" type="button">Previous</button>
      <button class="preview-next" type="button">Next</button>
      <select class="preview-size" aria-label="Items per page"></select>
      <button class="preview-expand" type="button">Show more</button>
      <button class="preview-show-all" type="button" hidden>Show all</button>
      <button class="preview-collapse" type="button" hidden>Show less</button>
      <button class="preview-inspect" type="button">Inspect</button>
    </nav>
    <div class="preview-hint">Click to explore this part. Escape closes this preview.</div>
  </div>`;
  const view = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
  const controls = root.querySelector('.toolbar');
  const math = root.querySelector('.math');
  const content = root.querySelector('.math-content');
  const status = root.querySelector('.status');
  const zoomFeedback = root.querySelector('.zoom-feedback');
  const scales = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32];
  const defaultScaleIndex = scales.indexOf(1);
  let scaleIndex = defaultScaleIndex;
  let zoomTimer;
  function setZoom(index) {
    scaleIndex = Math.max(0, Math.min(scales.length - 1, index));
    content.style.setProperty('--expression-zoom', scales[scaleIndex]);
    previewMath.style.zoom = scales[scaleIndex];
    zoomFeedback.textContent = `Zoom ${Math.round(scales[scaleIndex] * 100)}% · 0 to reset`;
    zoomFeedback.hidden = false;
    clearTimeout(zoomTimer);
    zoomTimer = setTimeout(() => { zoomFeedback.hidden = true; }, 1800);
    if (activePreview?.pinned) positionPreview();
    else hidePreview();
  }
  const preview = root.querySelector('.preview');
  const previewHeading = root.querySelector('.preview-heading');
  const previewMath = root.querySelector('.preview-math');
  const previewControls = root.querySelector('.preview-controls');
  const previewClose = root.querySelector('.preview-close');
  const previewBack = root.querySelector('.preview-back');
  const previewPrevious = root.querySelector('.preview-previous');
  const previewNext = root.querySelector('.preview-next');
  const previewSize = root.querySelector('.preview-size');
  const previewInspect = root.querySelector('.preview-inspect');
  const previewExpand = root.querySelector('.preview-expand');
  const previewShowAll = root.querySelector('.preview-show-all');
  const previewCollapse = root.querySelector('.preview-collapse');
  pageArrow(previewPrevious, 'Previous', 'previous');
  pageArrow(previewNext, 'Next', 'next');
  const previewHint = root.querySelector('.preview-hint');
  preview.id = `symbolica-preview-${view}`;
  preview.tabIndex = -1;
  preview.setAttribute('aria-keyshortcuts', '+ = - 0');
  previewHeading.id = `${preview.id}-heading`;
  const popupSizes = new Map();
  let activePreview = null;
  const inspection = new WeakMap();
  let refitting = false;
  let previewSequence = 0;
  let hoverTimer, hideTimer, restoringFocus = false;
  let positionFrame, previewNeedsFit = false;
  const pendingExpansions = [];
  let showAllLayout = false;
  // Parsed source trees are immutable. Keep a small per-view cache, never
  // measured DOM: fitting moves nodes and depends on the current font/width.
  const templates = new Map();
  let templateBytes = 0;
  function sourceTemplate(html) {
    let entry = templates.get(html);
    if (entry) {
      templates.delete(html);
      templates.set(html, entry);
      return entry;
    }
    const template = doc.createElement('template');
    template.innerHTML = html;
    entry = { template, parts: new Map([...template.content.querySelectorAll('[data-symbolica-path]')]
      .map(part => [part.dataset.symbolicaPath, part])) };
    const bytes = html.length * 2;
    if (bytes <= 2 * 1024 * 1024) {
      while (templates.size && (templates.size >= 32 || templateBytes + bytes > 2 * 1024 * 1024)) {
        const oldest = templates.keys().next().value;
        templateBytes -= oldest.length * 2;
        templates.delete(oldest);
      }
      templates.set(html, entry);
      templateBytes += bytes;
    }
    return entry;
  }
  function putHtml(container, html) {
    container.replaceChildren(sourceTemplate(html).template.content.cloneNode(true));
  }
  function hidePreview(restoreFocus = false) {
    clearTimeout(hoverTimer);
    clearTimeout(hideTimer);
    const anchor = activePreview?.button;
    anchor?.removeAttribute('aria-describedby');
    anchor?.setAttribute('aria-expanded', 'false');
    activePreview = null;
    if (preview.hidePopover && preview.matches(':popover-open')) preview.hidePopover();
    preview.hidden = true;
    previewMath.replaceChildren();
    if (restoreFocus && anchor?.isConnected) {
      restoringFocus = true;
      anchor.focus({ preventScroll: true });
      restoringFocus = false;
    }
  }
  function positionPreview() {
    if (!activePreview) return;
    const box = activePreview.button.getBoundingClientRect();
    const width = preview.offsetWidth, height = preview.offsetHeight;
    preview.style.left = `${Math.max(12, Math.min(box.left, window.innerWidth - width - 12))}px`;
    const top = box.bottom + 8 + height <= window.innerHeight - 12
      ? box.bottom + 8 : box.top - height - 8;
    preview.style.top = `${Math.max(12, top)}px`;
  }
  function refitPreview() {
    if (!activePreview?.lastValue) return positionPreview();
    const top = previewMath.scrollTop, left = previewMath.scrollLeft;
    displayPreview(activePreview.lastValue);
    previewMath.scrollTop = top;
    previewMath.scrollLeft = left;
  }
  function schedulePosition(event) {
    if (!activePreview) return;
    previewNeedsFit ||= event?.type === 'resize';
    if (positionFrame) return;
    positionFrame = requestAnimationFrame(() => {
      positionFrame = null;
      if (previewNeedsFit) refitPreview();
      else positionPreview();
      previewNeedsFit = false;
    });
  }
  function requestPreview() {
    const state = activePreview;
    if (!state) return;
    state.request = `${view}-${++previewSequence}`;
    previewPrevious.disabled = previewNext.disabled = previewSize.disabled = true;
    previewExpand.disabled = previewShowAll.disabled = previewCollapse.disabled = true;
    preview.setAttribute('aria-busy', 'true');
    model.send({ action: 'preview', value: state.path, request: state.request,
      layout: state.layout, context: state.context, popup: state.pinned, start: state.start, size: state.size });
  }
  function showPreview(button, path, label, pinned = false, layout = currentLayout, markerStart = 0) {
    clearTimeout(hideTimer);
    if (!pinned && activePreview?.pinned) return;
    if (activePreview?.button === button && activePreview.pinned === pinned) return;
    hidePreview();
    const key = JSON.stringify(path);
    const size = popupSizes.get(key) || model.get('page').page_size;
    activePreview = { button, path, layout, markerStart, pinned, size, start: markerStart, previous: [] };
    const show = () => {
      preview.classList.toggle('pinned', pinned);
      preview.setAttribute('role', pinned ? 'dialog' : 'tooltip');
      preview.setAttribute('aria-labelledby', previewHeading.id);
      previewControls.hidden = previewClose.hidden = !pinned;
      previewBack.hidden = true;
      previewPrevious.hidden = previewNext.hidden = previewSize.hidden = true;
      previewExpand.hidden = !model.get('page').supports_expansion;
      previewShowAll.hidden = previewCollapse.hidden = true;
      previewHint.textContent = pinned
        ? (model.get('page').supports_expansion
          ? 'Show more expands this part in the main view. Show less compacts this part. Inspect opens it separately.'
          : 'Inspect opens this part in the main view.')
        : 'Click to explore this part. Escape closes this preview.';
      previewHeading.textContent = label.replace('Inspect ', '');
      previewMath.textContent = 'Loading preview…';
      previewSize.replaceChildren();
      for (const n of [...new Set([5, 10, 25, 50, 100, 250, 500, size])].sort((a,b) => a-b)) {
        previewSize.add(new Option(`${n} per page`, n));
      }
      previewSize.value = size;
      preview.hidden = false;
      preview.showPopover?.();
      positionPreview();
      button.setAttribute(pinned ? 'aria-controls' : 'aria-describedby', preview.id);
      button.setAttribute('aria-expanded', String(pinned));
      requestPreview();
      if (pinned) preview.focus({ preventScroll: true });
    };
    if (pinned) show();
    else hoverTimer = setTimeout(show, 180);
  }
  function scheduleHide() {
    if (activePreview?.pinned) return;
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => hidePreview(), 150);
  }
  function updatePreview() {
    displayPreview(model.get('preview'));
  }
  const productPrefactors = new WeakMap();
  const wrappedWidths = new WeakMap();
  function wrappingCheckpoint(container) {
    // Refinement trials are synchronous. Retain the actual nodes and layout
    // state so a rejected trial can restore a fitted page without measuring
    // every table again. Node identity preserves inspection handlers/context.
    const nodes = [container, ...container.querySelectorAll('*')];
    const saved = nodes.map(node => ({ node, children: [...node.childNodes],
      attributes:[...node.attributes].map(a => [a.namespaceURI, a.name, a.value]),
      prefactor: productPrefactors.get(node), width: wrappedWidths.get(node) }));
    return () => {
      for (const node of container.querySelectorAll('[data-symbolica-product-prefactor]')) {
        productPrefactors.delete(node);
      }
      for (const { node, children, attributes, prefactor, width } of saved) {
        const current = node.childNodes;
        if (current.length !== children.length || children.some((child, i) => current[i] !== child)) {
          node.replaceChildren(...children);
        }
        const actual = [...node.attributes];
        if (actual.length !== attributes.length || actual.some((a, i) =>
            a.namespaceURI !== attributes[i][0] || a.name !== attributes[i][1] || a.value !== attributes[i][2])) {
          for (const attribute of actual) node.removeAttributeNode(attribute);
          for (const [namespace, name, value] of attributes) node.setAttributeNS(namespace, name, value);
        }
        productPrefactors.delete(node);
        if (prefactor) productPrefactors.set(node, prefactor);
        if (width === undefined) wrappedWidths.delete(node);
        else wrappedWidths.set(node, width);
      }
    };
  }
  function restoreProductFraction(numerator) {
    const state = productPrefactors.get(numerator);
    if (!state) return;
    state.prefix.remove();
    numerator.removeAttribute('data-symbolica-product-prefactor');
    numerator.replaceWith(state.fraction);
    state.fraction.append(numerator, state.denominator);
    if (state.sign) state.signPlaceholder.replaceWith(state.sign);
    if (state.coefficientFactor) {
      state.table.querySelector(':scope > mtr > mtd > mrow > [data-symbolica-factor]')
        ?.before(state.coefficientFactor);
    }
    productPrefactors.delete(numerator);
  }
  function layoutMetrics(container) {
    // clientWidth/scrollWidth and the wrap budget use CSS layout pixels.
    // DOMRects include ancestor CSS zoom/transforms, even inside a shadow root.
    // Our HTML containers use border-box sizing and computed pixel dimensions.
    const style = getComputedStyle(container);
    const width = parseFloat(style.width), height = parseFloat(style.height);
    const scale = width > 0 ? container.getBoundingClientRect().width / width : 1;
    return { width, height, scale: scale > 0 ? scale : 1 };
  }
  function setWrapWidth(container, width) {
    let budget = width;
    for (let attempt = 0; attempt < 3; attempt++) {
      wrapToWidth(container, budget);
      // MathML can add spacing/italic overhang outside the tables we packed.
      // A few pixels of root overflow should trigger a tighter wrap, not a
      // switch to a drastically shorter preview (or an expansion scrollbar).
      const metrics = layoutMetrics(container);
      const overflow = Math.max(container.scrollWidth, metrics.width) - width;
      if (overflow <= 1 || budget === 0) break;
      const overhang = Math.max(0, container.scrollWidth - metrics.width);
      budget = Math.max(0, Math.min(budget - overflow - .5, width - overhang - 1));
    }
    wrappedWidths.set(container, width);
  }
  function wrapToWidth(container, width) {
    const style = getComputedStyle(container);
    const {scale} = layoutMetrics(container);
    const widthOf = node => node ? node.getBoundingClientRect().width / scale : 0;
    const padding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
    const available = Math.max(0, width - padding);
    container.style.setProperty('--symbolica-wrap-width', `${available}px`);
    // Always decide between a stacked fraction and a leading coefficient in
    // the original stacked layout. Measuring the previously chosen layout
    // would let the coefficient's own width change the decision on a resize.
    for (const numerator of [...container.querySelectorAll('[data-symbolica-product-prefactor]')].reverse()) {
      const cell = numerator.closest('[data-symbolica-cell]');
      if (cell && cell !== container) continue;
      restoreProductFraction(numerator);
    }
    // Gecko and WebKit ignore flex layout on MathML rows. Keep the compact
    // MathML transport, but use HTML flex layout with one math root per term.
    // Only convert a root flow: nested math must retain its grouping/fences.
    for (const math of container.querySelectorAll(':scope > math')) {
      const row = math.children.length === 1 ? math.firstElementChild : null;
      if (row?.localName !== 'mrow' || row.style.flexWrap !== 'wrap'
          || !['flex', 'inline-flex'].includes(row.style.display)) continue;
      const flow = doc.createElement('div');
      flow.className = 'math-flow';
      flow.style.cssText = math.style.cssText;
      for (const name of row.style) {
        flow.style.setProperty(name, row.style.getPropertyValue(name));
      }
      for (const child of [...row.children]) {
        const term = doc.createElementNS(math.namespaceURI, 'math');
        const displaystyle = math.getAttribute('displaystyle');
        if (displaystyle !== null) term.setAttribute('displaystyle', displaystyle);
        if (child.localName === 'mrow') {
          for (const { name, value } of child.attributes) term.setAttribute(name, value);
          term.append(...child.childNodes);
        } else {
          term.append(child);
        }
        flow.append(term);
      }
      math.replaceWith(flow);
    }
    // The kernel's product rows are a static fallback. Fit the final notation,
    // including decorated omissions, using the font actually loaded here.
    // Nested sums in a long factor use the same measured packing policy.
    const tables = 'mtable[data-symbolica-product],mtable[data-symbolica-sum]';
    // Descendants must not reserve the previous pass's outer continuation.
    // Clear stale row starts before measuring any table; recursive fits below
    // still reserve the continuations created during this pass.
    for (const node of container.querySelectorAll('[data-symbolica-continuation], [data-symbolica-continuation-separator]')) {
      const cell = node.closest('[data-symbolica-cell]');
      if (cell && cell !== container) continue;
      if (node.hasAttribute('data-symbolica-continuation')) node.remove();
      else node.removeAttribute('data-symbolica-continuation-separator');
    }
    const refitted = new Map();
    const tableLimits = new Map();
    function fitTable(table) {
      // This denominator already passed the compact, single-line check in
      // canonical layout. Keep the resulting coefficient indivisible.
      if (table.closest('[data-symbolica-fraction-prefix]')) return;
      const cell = table.closest('[data-symbolica-cell]');
      if (cell && cell !== container) return;
      const product = table.hasAttribute('data-symbolica-product');
      const rows = [...table.children].map(row => {
        const cell = row.children.length === 1 ? row.firstElementChild : null;
        const body = cell?.localName === 'mtd' && cell.children.length === 1
          ? cell.firstElementChild : null;
        if (body?.localName === 'mrow') {
          // Previous line starts can become inline factors after a resize.
          // Restore their original separators before measuring the new rows.
          for (const child of [...body.children]) {
            if (child.hasAttribute('data-symbolica-continuation')) child.remove();
            if (child.hasAttribute('data-symbolica-factor')) {
              for (const separator of child.children) {
                separator.removeAttribute('data-symbolica-continuation-separator');
              }
            }
          }
        }
        const factors = body?.localName === 'mrow'
          && [...body.childNodes].every(node => node.nodeType === 1
            ? node.hasAttribute('data-symbolica-factor') || node.hasAttribute('data-symbolica-term')
            : node.nodeType === 3 && !node.textContent.trim())
          ? [...body.children] : [];
        return { row, factors };
      });
      const factorCount = rows.reduce((n, row) => n + row.factors.length, 0);
      if (factorCount < 2) return;
      // Reserve surrounding signs, fences and fraction padding. Products
      // have zero outer cell padding, including after measured row packing.
      // Other table rows and wrapping sums/products do not restrict this table
      // to their current arrangement.
      const rowWidth = Math.max(0, ...rows.map(({ row }) =>
        widthOf(row.querySelector(':scope > mtd > mrow'))));
      const cellOverhead = Math.max(0, widthOf(table) - rowWidth);
      let overhead = cellOverhead;
      for (let child = table, parent = child.parentElement;
          parent && parent !== container; child = parent, parent = parent.parentElement) {
        const children = [...parent.children];
        if (parent.localName === 'mrow'
            && children.some(node => node.hasAttribute('data-symbolica-factor') || node.hasAttribute('data-symbolica-term'))) {
          // The surrounding product can move this factor between rows, but
          // a continuation's fixed prefix still takes space from its children.
          // Keep a small leading factor with the next factor: otherwise
          // a nested sum first fills the entire row and strands the prefix
          // above it. Reserve the prefix while fitting that sum, even if
          // a previous pass put it on a separate continuation row.
          const owner = child.hasAttribute('data-symbolica-factor')
            ? parent.closest('mtable[data-symbolica-product]') : null;
          const factors = owner ? [...owner.querySelectorAll(':scope > mtr > mtd > mrow > [data-symbolica-factor]')] : [];
          // Include short symbolic factors such as (1+a), not just numbers.
          // Moving such a prefix to its own line can otherwise increase height
          // when the viewport widens and make height fitting drop detail.
          const prefix = factors[0];
          const prefixWidth = widthOf(prefix);
          const keepPrefix = child === factors[1] && prefixWidth > 0
            && prefixWidth <= 4 * parseFloat(getComputedStyle(prefix).fontSize);
          let separatorWidth = 0;
          if (keepPrefix) {
            // Recursive fitting can reach an old continuation whose original
            // multiplication separator is still hidden. Account for that
            // separator too: it returns when these factors share a row.
            const hidden = [...child.children].filter(node =>
              node.hasAttribute('data-symbolica-continuation-separator'));
            if (hidden.length) {
              const before = widthOf(child);
              for (const node of hidden) node.removeAttribute('data-symbolica-continuation-separator');
              separatorWidth = Math.max(0, widthOf(child) - before);
              for (const node of hidden) node.setAttribute('data-symbolica-continuation-separator', '');
            }
          }
          overhead += keepPrefix ? prefixWidth + separatorWidth
            : widthOf(children.find(node => node.hasAttribute('data-symbolica-continuation')));
          continue;
        }
        if (['mtable', 'mtr', 'mtd'].includes(parent.localName)
            || (['flex', 'inline-flex'].includes(parent.style.display) && parent.style.flexWrap === 'wrap')) continue;
        const childWidth = parent.localName === 'mfrac'
          ? Math.max(0, ...children.map(node => widthOf(node)))
          : widthOf(child);
        overhead += Math.max(0, widthOf(parent) - childWidth);
      }
      const tableLimit = Math.max(0, available - (overhead - cellOverhead));
      let limit = Math.max(0, available - overhead);
      // A single canonical row that already fits needs no continuation
      // measurement, separator toggles, or replacement rows. Check the same
      // additive factor widths the packer uses, as well as the rendered table.
      if (rows.length === 1 && rows[0].factors.length === factorCount
          && (!table.hasAttribute('data-symbolica-row-gap')
            || rows[0].row.firstElementChild.style.padding === '0px')
          && rows[0].factors.reduce((sum, factor) => sum + widthOf(factor), 0) <= limit + .5
          && widthOf(table) <= tableLimit + .5) {
        tableLimits.set(table, limit);
        return;
      }
      let continuation, continuationWidth = 0;
      {
        continuation = doc.createElementNS(table.namespaceURI, 'mrow');
        continuation.setAttribute('data-symbolica-continuation', '');
        const indent = doc.createElementNS(table.namespaceURI, 'mspace');
        indent.setAttribute('width', '0.75em');
        continuation.append(indent);
        if (product) {
          const operator = doc.createElementNS(table.namespaceURI, 'mo');
          operator.setAttribute('form', 'infix');
          operator.setAttribute('lspace', '0');
          operator.setAttribute('rspace', '0');
          operator.textContent = '×';
          const gap = doc.createElementNS(table.namespaceURI, 'mspace');
          gap.setAttribute('width', '0.25em');
          continuation.append(operator, gap);
        } else {
          // The term itself already carries its + or −. Only indent it.
          continuation.setAttribute('data-symbolica-sum-continuation', '');
        }
      }
      if (!refitted.has(table)) refitted.set(table, new Set());
      const descendants = refitted.get(table);
      for (let attempt = 0; attempt < factorCount; attempt++) {
        const measured = rows.map(({ row, factors }) => ({
          row, factors: factors.map(node => {
            // The kernel puts the multiplication separator inside each factor.
            // A continuation replaces that separator instead of doubling it.
            const leading = product && node.firstElementChild;
            const separators = leading?.localName === 'mo' ? [leading] : [];
            if ((leading?.textContent === '\u2062' || leading?.textContent === '×')
                && leading.nextElementSibling?.localName === 'mspace') {
              separators.push(leading.nextElementSibling);
            }
            for (const separator of separators) separator.removeAttribute('data-symbolica-continuation-separator');
            return { node, separators };
          })
        }));
        const factors = measured.flatMap(row => row.factors);
        for (const factor of factors) factor.width = widthOf(factor.node);
        if (!attempt) {
          // Read the original factors before inserting the probe. Removing it
          // and hiding separators can then share one layout invalidation.
          // Measure in this table's math style, including fraction/script size.
          factors[0].node.before(continuation);
          continuationWidth = widthOf(continuation);
          continuation.remove();
        }
        for (const factor of factors) {
          for (const separator of factor.separators) separator.setAttribute('data-symbolica-continuation-separator', '');
        }
        for (const factor of factors) {
          factor.continuationWidth = product ? widthOf(factor.node) : factor.width;
        }
        for (const factor of factors) {
          for (const separator of factor.separators) separator.removeAttribute('data-symbolica-continuation-separator');
        }
        const packed = [];
        let line, lineWidth = 0, seenFactors = 0;
        for (const { row, factors } of measured) {
          if (!factors.length) {
            // In particular, keep a terminal product omission on its own row.
            packed.push(row);
            line = null;
            continue;
          }
          for (const factor of factors) {
            let factorWidth = factor.width;
            if (!line || lineWidth + factor.width > limit + 0.5) {
              const row = doc.createElementNS(table.namespaceURI, 'mtr');
              const cell = doc.createElementNS(table.namespaceURI, 'mtd');
              line = doc.createElementNS(table.namespaceURI, 'mrow');
              cell.append(line);
              row.append(cell);
              packed.push(row);
              lineWidth = 0;
              if (seenFactors) {
                line.append(continuation.cloneNode(true));
                for (const separator of factor.separators) separator.setAttribute('data-symbolica-continuation-separator', '');
                // Include both the indent and operator when fitting this row.
                lineWidth = continuationWidth;
                factorWidth = factor.continuationWidth;
              }
            }
            line.append(factor.node);
            lineWidth += factorWidth;
            seenFactors++;
          }
        }
        table.replaceChildren(...packed);
        const rowGap = table.getAttribute('data-symbolica-row-gap');
        if (rowGap) {
          // Repacking creates cells and can move the omission row. Apply the
          // same internal-only spacing as the kernel's static product layout.
          for (const [index, row] of packed.entries()) {
            for (const cell of row.children) {
              cell.style.padding = index ? `${rowGap} 0 0` : '0';
            }
          }
        }
        // An inner product was initially fitted before its outer row existed.
        // If that factor now starts a continuation, refit its contents with
        // the newly reserved prefix width. Share the retry sets across this
        // whole pass: an indivisible wide leaf must not trigger exponentially
        // many retries through its ancestors. Each ancestor/descendant pair
        // is retried once, including across recursive calls.
        let nestedRefit = false;
        for (const row of packed) {
          if (!row.querySelector(':scope > mtd > mrow > [data-symbolica-continuation]')
              || widthOf(row) <= tableLimit + .5) continue;
          for (const nested of [...row.querySelectorAll(tables)].reverse()) {
            if (!descendants.has(nested)) {
              descendants.add(nested);
              nestedRefit = true;
              fitTable(nested);
            }
          }
        }
        // WebKit can change intrinsic cell widths when factors move between
        // rows. Check the packed table too, and only reduce this pass's budget.
        const overflow = widthOf(table) - tableLimit;
        if (overflow <= 0.5) {
          // An inner fit can free enough room to combine outer rows. Repack
          // with those final widths instead of keeping a stale line break.
          if (nestedRefit && limit !== 0) continue;
          break;
        }
        if (limit === 0) break;
        // A single wide leaf cannot become narrower by moving other terms to
        // another row. In particular, a sum's hanging indent can expose this
        // limit. Avoid retrying every term when no nested table can reflow it.
        const indivisibleOverflow = packed.some(row => {
          const body = row.querySelector(':scope > mtd > mrow');
          const factors = body ? [...body.children].filter(node =>
            node.hasAttribute('data-symbolica-factor') || node.hasAttribute('data-symbolica-term')) : [];
          return factors.length === 1 && !factors[0].querySelector(tables)
            && widthOf(body) > limit + .5;
        });
        if (indivisibleOverflow) break;
        limit = Math.max(0, limit - overflow);
      }
      tableLimits.set(table, limit);
    }
    for (const table of [...container.querySelectorAll(tables)].reverse()) fitTable(table);
    // A short denominator is more readable as a leading coefficient only
    // when a substantially longer numerator has actual product continuations.
    // Leave single-line quotients, sums and large denominators stacked.
    const compactFractions = [];
    for (const fraction of container.querySelectorAll('mfrac[data-symbolica-product-fraction]')) {
      const cell = fraction.closest('[data-symbolica-cell]');
      if (cell && cell !== container) continue;
      const [numerator, denominator] = fraction.children;
      const table = numerator?.children.length === 1 ? numerator.firstElementChild : null;
      if (!denominator || table?.localName !== 'mtable'
          || !table.hasAttribute('data-symbolica-product')) continue;
      const factorRows = [...table.children].map(row =>
        [...row.querySelectorAll(':scope > mtd > mrow > [data-symbolica-factor]')]);
      if (factorRows.filter(factors => factors.length).length < 2) continue;
      // Restrict this change to simple, fully visible denominator notation.
      // In particular, a tiny omission button must not disguise a large sum.
      const simple = new Set(['mrow', 'mtable', 'mtr', 'mtd', 'mi', 'mn', 'mo', 'mspace', 'msub', 'msup', 'msubsup']);
      if ([denominator, ...denominator.querySelectorAll('*')].some(node =>
        !simple.has(node.localName) || node.hasAttribute('data-symbolica-marker')
        || node.hasAttribute('data-symbolica-sum')
        || (node.localName === 'mo' && !['\u2062', '×', '·'].includes(node.textContent)))) continue;
      const denominatorTables = [...denominator.querySelectorAll('mtable')];
      if (denominatorTables.length > 1
          || denominatorTables.some(table => table.children.length !== 1)) continue;
      const rect = denominator.getBoundingClientRect();
      const box = {width:rect.width / scale, height:rect.height / scale};
      const em = parseFloat(getComputedStyle(denominator).fontSize);
      const factors = factorRows.flat();
      const numeratorWidth = factors.reduce((total, factor) => total + widthOf(factor), 0);
      if (!(box.width > 0 && em > 0) || box.width > 3 * em || box.height > 2 * em
          || numeratorWidth < 4 * box.width) continue;
      // The kernel marks its real numeric coefficient, which is ordered first.
      // Keep that exact node so 3*A/8 becomes (3/8)*A rather than (1/8)*3*A.
      const coefficientFactor = factors[0].hasAttribute('data-symbolica-coefficient') ? factors[0] : null;
      compactFractions.push({ fraction, numerator, denominator, table, coefficientFactor,
        first: factors[coefficientFactor ? 1 : 0], limit: tableLimits.get(table) ?? 0 });
    }
    // Freeze every choice before changing any DOM. In particular, moving an
    // outer coefficient must not change the eligibility of a nested fraction.
    for (const { fraction, numerator, denominator, table, coefficientFactor, first, limit } of compactFractions.reverse()) {
      const prefix = doc.createElementNS(fraction.namespaceURI, 'mrow');
      prefix.setAttribute('data-symbolica-fraction-prefix', '');
      const previous = fraction.previousElementSibling;
      const sign = previous?.localName === 'mo' && previous.textContent === '−' ? previous : null;
      const signPlaceholder = sign ? doc.createComment('product sign') : null;
      if (sign) {
        sign.replaceWith(signPlaceholder);
        prefix.append(sign);
      }
      const coefficient = doc.createElementNS(fraction.namespaceURI, 'mfrac');
      const one = doc.createElementNS(fraction.namespaceURI, 'mn');
      one.textContent = '1';
      const coefficientRow = coefficientFactor?.parentElement.children.length === 1
        ? coefficientFactor.closest('mtr') : null;
      coefficient.append(coefficientFactor || one, denominator);
      coefficientRow?.remove();
      const separator = doc.createElementNS(fraction.namespaceURI, 'mo');
      separator.textContent = '\u2062';
      const gap = doc.createElementNS(fraction.namespaceURI, 'mspace');
      gap.setAttribute('width', '0.15em');
      prefix.append(coefficient);
      // Moving the original first coefficient leaves its next factor's
      // configured multiplication separator in place. Otherwise this was the
      // first factor already and needs a new implicit separator.
      if (!coefficientFactor) {
        prefix.append(separator, gap);
      }
      first.prepend(prefix);
      fraction.replaceWith(numerator);
      numerator.setAttribute('data-symbolica-product-prefactor', '');
      productPrefactors.set(numerator, { fraction, denominator, table, coefficientFactor,
        prefix, sign, signPlaceholder });
      // Keep the coefficient attached to a real factor, but do not create a
      // newly indivisible wide item. A large numeric coefficient may have fit
      // on its own canonical row. The original row budget is conservative:
      // removing the outer fraction/sign only makes more room available.
      if (widthOf(first) > limit + .5) restoreProductFraction(numerator);
    }
    if (compactFractions.length) {
      // The selected coefficient consumes width only on the first row. Fit
      // once more without reconsidering the canonical decision in this pass.
      refitted.clear();
      for (const table of [...container.querySelectorAll(tables)].reverse()) fitTable(table);
    }
    // Composite outputs retain a static frame while each value is fitted to
    // its own cell width. Root-flow conversion stays outside nested MathML.
    for (const cell of container.querySelectorAll('[data-symbolica-cell]')) {
      const parent = cell.parentElement.closest('[data-symbolica-cell]');
      if (!parent || parent === container) {
        // clientWidth rounds to integer CSS pixels. Rounding down a column by
        // even a fraction of a pixel can wrap a sum that fits exactly. Keep
        // its used width (independent of the ancestor's expression zoom).
        // A subpixel guard covers computed-style serialization precision.
        const used = parseFloat(getComputedStyle(cell).width);
        setWrapWidth(cell, Number.isFinite(used) ? used + .05 : cell.clientWidth);
      }
    }
    wrappedWidths.set(container, width);
  }
  function widthWithCells(container, width, available) {
    // A wide term can overlap the next column without overflowing the entire
    // matrix. Treat overflow within a value cell as a reason to try a more
    // compact native page too.
    let excess = 0;
    for (const cell of container.querySelectorAll('[data-symbolica-cell]')) {
      excess = Math.max(excess, cell.scrollWidth - cell.clientWidth);
    }
    return excess > 1 ? Math.max(width, available + excess) : width;
  }
  function displayPreview(value) {
    if (!activePreview || value.request !== activePreview.request) return;
    activePreview.lastValue = value;
    previewBack.hidden = !activePreview.parent;
    preview.removeAttribute('aria-busy');
    if (value.error) {
      previewMath.textContent = value.error;
      previewPrevious.hidden = previewNext.hidden = previewSize.hidden = true;
    } else {
      const layouts = [value, ...(value.layouts || []).map(p => ({...value, ...p}))];
      const zoom = previewMath.style.zoom;
      previewMath.style.zoom = 1;
      let selected = value, smallest = Infinity, selectedNodes, shown;
      for (const candidate of layouts) {
        putHtml(previewMath, candidate.html);
        shown = candidate;
        decorateMarkers(candidate, previewMath, value.context);
        setWrapWidth(previewMath, previewMath.clientWidth);
        const width = widthWithCells(previewMath,
          Math.max(previewMath.scrollWidth - 4, (previewMath.firstElementChild?.getBoundingClientRect().width || 0) / layoutMetrics(previewMath).scale),
          previewMath.clientWidth - 4);
        if (width < smallest) {
          selected = candidate;
          smallest = width;
          selectedNodes = [...previewMath.childNodes];
        }
        // Rendering a tall candidate may introduce a vertical scrollbar and
        // reduce the available width on platforms with classic scrollbars.
        if (width <= previewMath.clientWidth - 4 + 1) { selected = candidate; break; }
      }
      if (shown !== selected) {
        previewMath.replaceChildren(...selectedNodes);
        setWrapWidth(previewMath, previewMath.clientWidth);
      }
      revealFittingParts(layouts, previewMath.clientWidth - 4, previewMath, value.context);
      numberMarkers(previewMath);
      setWrapWidth(previewMath, previewMath.clientWidth);
      previewMath.style.zoom = zoom || scales[scaleIndex];
      activePreview.page = selected;
      activePreview.start = selected.start;
      previewMath.scrollTop = previewMath.scrollLeft = 0;
      previewHeading.textContent = `${selected.kind} · ${selected.unit} ${selected.total ? selected.start + 1 : 0}–${selected.end} of ${selected.total}`;
      previewPrevious.disabled = selected.start === 0;
      previewNext.disabled = selected.next === null;
      const paginated = selected.start > 0 || selected.next !== null;
      previewPrevious.hidden = previewNext.hidden = previewSize.hidden = !paginated;
      previewSize.disabled = false;
      previewExpand.disabled = value.kind === 'Value' || value.can_expand === false;
      previewExpand.title = previewExpand.disabled ? 'Inline display limit reached; use pagination or Inspect.' : '';
      previewShowAll.hidden = !value.can_show_all;
      previewShowAll.disabled = !value.can_show_all;
      previewCollapse.hidden = !(value.can_collapse ?? model.get('page').can_collapse);
      previewCollapse.disabled = false;
    }
    positionPreview();
  }
  function showSubPreview(path, label, layout, context, markerStart = 0) {
    const parent = activePreview;
    if (!parent) return;
    parent.pinned = true;
    parent.scrollTop = previewMath.scrollTop;
    parent.scrollLeft = previewMath.scrollLeft;
    activePreview = {button:parent.button, parent, path, layout, context,
      markerStart, pinned:true, start:markerStart,
      size:popupSizes.get(JSON.stringify(path)) || parent.size, previous:[]};
    preview.classList.add('pinned');
    preview.setAttribute('role', 'dialog');
    previewControls.hidden = previewClose.hidden = false;
    previewExpand.hidden = !model.get('page').supports_expansion;
    previewShowAll.hidden = previewCollapse.hidden = true;
    previewHint.textContent = 'Back returns to the previous popup. Inspect opens this part in the main view.';
    previewHeading.textContent = label.replace('Inspect ', '');
    previewMath.textContent = 'Loading preview…';
    previewBack.hidden = false;
    previewSize.value = activePreview.size;
    requestPreview();
    preview.focus({preventScroll:true});
  }
  previewBack.onclick = () => {
    if (!activePreview?.parent) return;
    activePreview = activePreview.parent;
    previewSize.value = activePreview.size;
    displayPreview(activePreview.lastValue);
    previewMath.scrollTop = activePreview.scrollTop || 0;
    previewMath.scrollLeft = activePreview.scrollLeft || 0;
    if (!activePreview.parent) {
      previewHint.textContent = model.get('page').supports_expansion
        ? 'Show more expands this part in the main view. Show less compacts this part. Inspect opens it separately.'
        : 'Inspect opens this part in the main view.';
    }
    preview.focus({preventScroll:true});
  };
  previewPrevious.onclick = () => {
    activePreview.start = activePreview.previous.pop() ?? Math.max(0, activePreview.start - activePreview.size);
    requestPreview();
  };
  previewNext.onclick = () => {
    activePreview.previous.push(activePreview.start);
    activePreview.start = activePreview.page.next;
    requestPreview();
  };
  previewSize.onchange = () => {
    activePreview.size = Number(previewSize.value);
    popupSizes.set(JSON.stringify(activePreview.path), activePreview.size);
    activePreview.previous = [];
    requestPreview();
  };
  function rememberExpansion(action, button, target, rect = button.getBoundingClientRect()) {
    const scrollers = [];
    // Include nested notebook scroll panes as well as the document. Snapshot
    // actual offsets so a user scroll during the request is not undone.
    for (let node = math; node; node = node.parentElement || node.getRootNode()?.host) {
      if (node === doc.scrollingElement || node.scrollHeight > node.clientHeight
          || node.scrollWidth > node.clientWidth) {
        scrollers.push({ node, top: node.scrollTop, left: node.scrollLeft });
      }
    }
    if (!scrollers.some(({node}) => node === doc.scrollingElement)) {
      const node = doc.scrollingElement;
      scrollers.push({node, top:node.scrollTop, left:node.scrollLeft});
    }
    pendingExpansions.push({ action, path:target.path, start:target.start, rect, scrollers, button });
  }
  function changeExpansion(action) {
    const { path, layout, size, context, button } = activePreview;
    rememberExpansion(action, button, inspection.get(button));
    hidePreview(true);
    model.send({ action, value:path, layout, size, context });
  }
  previewExpand.onclick = () => changeExpansion('expand');
  previewShowAll.onclick = () => changeExpansion('expand_all');
  previewCollapse.onclick = () => changeExpansion('collapse');
  previewInspect.onclick = () => {
    const { path, layout, size, context, start } = activePreview;
    hidePreview();
    model.send({ action: 'open', value: path, layout, size, context, start });
  };
  previewClose.onclick = () => hidePreview(true);
  function bindInspection(button, path, label, layout, context = null, start = 0) {
    inspection.set(button, { path, label, layout, start });
    button.setAttribute('aria-haspopup', 'dialog');
    button.setAttribute('aria-expanded', 'false');
    if (context) {
      button.onclick = () => showSubPreview(path, label, layout, context, start);
      return;
    }
    button.onclick = () => showPreview(button, path, label, true, layout, start);
    button.onmouseenter = () => showPreview(button, path, label, false, layout, start);
    button.onmouseleave = scheduleHide;
    button.onfocus = () => { if (!restoringFocus) showPreview(button, path, label, false, layout, start); };
    button.onblur = () => { if (!refitting && !activePreview?.pinned) hidePreview(); };
  }
  preview.onmouseenter = () => clearTimeout(hideTimer);
  preview.onmouseleave = scheduleHide;
  const outsidePointer = event => {
    if (activePreview?.pinned && !event.composedPath().includes(preview)
      && !event.composedPath().includes(activePreview.button)) hidePreview();
  };
  doc.addEventListener('pointerdown', outsidePointer, true);
  const onKeyDown = event => {
    if (event.key === 'Escape' && activePreview) {
      event.preventDefault();
      event.stopPropagation();
      hidePreview(true);
    }
    // Local shortcuts only: leave browser zoom and notebook/editor inputs alone.
    const target = event.composedPath()[0];
    if ((!math.contains(target) && !preview.contains(target)) || event.ctrlKey || event.metaKey || event.altKey
      || event.isComposing || target.closest('input,textarea,select,[contenteditable]')) return;
    let index;
    if (event.key === '+' || event.key === '=') index = scaleIndex + 1;
    else if (event.key === '-' || event.key === '−') index = scaleIndex - 1;
    else if (event.key === '0') index = defaultScaleIndex;
    else return;
    event.preventDefault();
    event.stopPropagation();
    setZoom(index);
  };
  root.addEventListener('keydown', onKeyDown);
  math.addEventListener('pointerdown', event => {
    if (!event.target.closest('button')) math.focus({ preventScroll: true });
  });
  // Scrolling a marker into view can finish after mouseenter. Keep its pending
  // hover request alive and reposition the preview with its anchor.
  math.addEventListener('scroll', schedulePosition, { passive: true });
  doc.addEventListener('scroll', schedulePosition, { capture: true, passive: true });
  doc.defaultView.addEventListener('resize', schedulePosition, { passive: true });
  doc.defaultView.visualViewport?.addEventListener('resize', schedulePosition, { passive: true });
  doc.defaultView.visualViewport?.addEventListener('scroll', schedulePosition, { passive: true });
  let currentLayout = 0;
  const send = (action, value) => model.send({
    action, ...(value === undefined ? {} : { value }), layout: currentLayout
  });
  function pageArrow(b, label, action) {
    b.classList.add('page-arrow');
    b.title = label;
    b.setAttribute('aria-label', label);
    const path = action === 'previous' ? 'M15 18l-6-6 6-6' : 'M9 6l6 6-6 6';
    b.innerHTML = `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="${path}"/></svg>`;
  }
  function button(label, action) {
    const b = document.createElement('button');
    b.textContent = label;
    b.type = 'button';
    if (action === 'previous' || action === 'next') {
      pageArrow(b, label, action);
    }
    b.onclick = () => send(action);
    controls.append(b);
    return b;
  }
  const back = button('Back to parent', 'back');
  const previous = button('Previous', 'previous');
  const next = button('Next', 'next');
  const collapseAll = button('Collapse all', 'collapse');
  collapseAll.classList.add('collapse-all');
  collapseAll.onclick = () => {
    rememberExpansion('collapse', collapseAll, {path:[], start:0}, math.getBoundingClientRect());
    send('collapse');
  };
  function initialHeightLimit(page) {
    // About eight ordinary math lines. This is a soft target: keep a taller
    // expression when no shorter alternative retains useful complete terms.
    // Explicit expansion and inspection must not immediately fold again.
    return page.can_collapse || page.back || showAllLayout
      ? Infinity : 12 * parseFloat(getComputedStyle(content).fontSize);
  }
  function keepsCompleteTerms(candidate, detailed) {
    const template = sourceTemplate(candidate.html);
    return candidate.markers.every(([path, label], index) => {
      // Replacing a sum suffix removes complete terms. Existing omissions
      // may also remain; height fitting must not add holes inside those terms.
      if (label.toLowerCase().includes('sum')) return true;
      if (detailed.markers.some(([parent], i) => parent.every((p, n) => path[n] === p)
          && (path[parent.length] ?? candidate.marker_starts[index]) >= detailed.marker_starts[i])) return true;
      const marker = template.template.content.querySelector(`[data-symbolica-marker="${candidate.marker_ids[index]}"]`);
      return label.toLowerCase().includes('product') && marker
        && !marker.closest('[data-symbolica-term]');
    });
  }
  function update(preservePreview = false) {
    const expansion = preservePreview === true ? null : pendingExpansions.shift();
    const restoreExpansionFocus = expansion && root.activeElement === expansion.button;
    const restoreExpansionScroll = expansion && expansion.scrollers.every(({node, top, left}) =>
      Math.abs(node.scrollTop - top) < 1 && Math.abs(node.scrollLeft - left) < 1);
    const retainedPreview = preservePreview === true ? activePreview : null;
    const restoreMarkerFocus = retainedPreview && root.activeElement === retainedPreview.button;
    if (!retainedPreview) hidePreview();
    else {
      clearTimeout(hoverTimer);
      clearTimeout(hideTimer);
    }
    refitting = true;
    const base = model.get('page');
    if (preservePreview !== true) {
      if (expansion?.action === 'expand_all' && !base.error) showAllLayout = true;
      else if (!expansion || expansion.action === 'collapse') showAllLayout = false;
    }
    // An explicit Show all must not immediately hide the same detail again
    // through a narrower alternative. Keep measured wrapping, with horizontal
    // scrolling for an indivisible wide item, until navigation or collapse.
    const layouts = showAllLayout ? [base]
      : [base, ...(base.layouts || []).map(candidate => ({ ...base, ...candidate }))];
    // Measure native MathML with the actual font and notebook width. Keep the
    // first (most detailed) layout that fits; no resize messages to the kernel.
    // Zoom magnifies the chosen expression, rather than hiding detail again.
    const zoom = content.style.getPropertyValue('--expression-zoom');
    content.style.setProperty('--expression-zoom', 1);
    let available = math.clientWidth;
    let page = base, smallestWidth = Infinity, selectedNodes, shown;
    let widthFit;
    const heightLimit = initialHeightLimit(base);
    // Finishing a modest expression is worth another couple of lines. Keep
    // the ordinary target for partial previews: giving every prefix this
    // allowance would just move the cutoff for arbitrarily large output.
    const complete = candidate => candidate.start === 0 && candidate.end === candidate.total
      && !candidate.markers.length;
    const candidateHeightLimit = candidate => complete(candidate) ? heightLimit * 1.25 : heightLimit;
    const measured = new Map();
    function measure(candidate) {
      const saved = measured.get(candidate);
      if (saved && saved.available === available) return saved;
      putHtml(content, candidate.html);
      shown = candidate;
      decorateMarkers(candidate);
      // A taller candidate can give an enclosing notebook cell a scrollbar.
      // Keep the narrowest width observed during this pass, including changes
      // caused by trial layouts, so compacting cannot reopen a wider layout.
      available = Math.min(available, math.clientWidth);
      setWrapWidth(content, available);
      const metrics = layoutMetrics(content);
      const width = widthWithCells(content,
        Math.max(content.scrollWidth, metrics.width), available);
      const result = {candidate, width, height:metrics.height,
        available, nodes:[...content.childNodes]};
      measured.set(candidate, result);
      return result;
    }
    let heightFit;
    // The complete page has a little more room than its partial alternatives,
    // so it can fit even when a shorter prefix misses the ordinary target.
    // Check it before the binary search, whose cutoffs must be monotone.
    if (Number.isFinite(heightLimit) && complete(base)) {
      const result = measure(base);
      if (result.width <= available + 1 && result.height <= candidateHeightLimit(base) + 1) heightFit = result;
    }
    if (!heightFit && Number.isFinite(heightLimit) && base.html.length > 16 * 1024 && layouts.length > 1) {
      // Alternatives progressively shorten the preview. Search the complete
      // prefixes first, instead of laying out a very long sum just to reject
      // its height. This threshold only chooses the search order; it does not
      // cap displayed markup. A fitting 64 KiB expression can still win.
      const candidates = layouts.filter(candidate => candidate === base || keepsCompleteTerms(candidate, base));
      for (let pass = 0; pass < 2; pass++) {
        heightFit = null;
        let low = 0, high = candidates.length - 1;
        while (low <= high) {
          const middle = (low + high) >> 1;
          const result = measure(candidates[middle]);
          if (result.width <= available + 1 && result.height <= candidateHeightLimit(result.candidate) + 1) {
            heightFit = result;
            high = middle - 1;
          } else low = middle + 1;
        }
        // Testing a taller candidate may introduce a native scrollbar after
        // the winner was measured. Repeat the bounded search at that narrower
        // width rather than restoring a now-too-tall winner and flickering on
        // the subsequent ResizeObserver callback.
        if (!heightFit || heightFit.available === available) break;
      }
      if (heightFit && heightFit.available !== available) heightFit = null;
    }
    if (heightFit) {
      page = heightFit.candidate;
      selectedNodes = heightFit.nodes;
    } else for (const candidate of layouts) {
      const result = measure(candidate);
      const {width, height, nodes} = result;
      if (!widthFit && width < smallestWidth) {
        page = candidate;
        smallestWidth = width;
        selectedNodes = nodes;
      }
      if (width <= available + 1) {
        if (!widthFit) {
          widthFit = candidate;
          page = candidate;
          selectedNodes = nodes;
        }
        if (height <= candidateHeightLimit(candidate) + 1
            && (candidate === widthFit || keepsCompleteTerms(candidate, widthFit))) {
          page = candidate;
          selectedNodes = nodes;
          break;
        }
      }
    }
    currentLayout = page.layout || 0;
    content.dataset.layout = currentLayout;
    // Retain the winning DOM and its marker handlers instead of parsing,
    // decorating and fitting the same page a second time.
    if (shown !== page) {
      content.replaceChildren(...selectedNodes);
      setWrapWidth(content, available);
    }
    if (!expansion && preservePreview !== true) math.scrollLeft = 0;
    status.textContent = page.error || `Items ${page.total ? page.start + 1 : 0}–${page.end} of ${page.total} (${page.unit})${page.markers.length ? ' · Detail fitted to width; open a marker for more' : ''}`;
    previous.disabled = !page.previous;
    next.disabled = page.next === null;
    back.hidden = !page.back;
    const paginated = page.previous || page.next !== null;
    previous.hidden = next.hidden = !page.back || !paginated;
    collapseAll.hidden = !page.can_collapse;
    controls.hidden = !page.back && !page.can_collapse;
    math.classList.toggle('compact', !page.back);
    status.hidden = !page.error && !page.back;
    const finalWidth = Math.min(available, math.clientWidth);
    if (finalWidth < available - .5) setWrapWidth(content, finalWidth);
    available = finalWidth;
    // Height fitting already rejected the longer prefixes. Refining them
    // again repeats expensive layout work and cannot add another full page
    // of terms within the same target. Width-only fits retain local refinement.
    if (!Number.isFinite(heightLimit) || ((page === base || page === widthFit)
        && layoutMetrics(content).height <= heightLimit + 1)) {
      revealFittingParts(layouts, available, content, null,
        Math.max(heightLimit, layoutMetrics(content).height));
    }
    if (numberMarkers(content)) setWrapWidth(content, available);
    content.style.setProperty('--expression-zoom', zoom || 1);
    refitting = false;
    if (expansion) {
      const path = expansion.path.join('/');
      const firstRevealed = [...expansion.path, expansion.start].join('/');
      const parts = [...content.querySelectorAll('[data-symbolica-path]')];
      const marker = [...content.querySelectorAll('button.omission')].find(button =>
        JSON.stringify(inspection.get(button)?.path) === JSON.stringify(expansion.path));
      const target = parts.find(part => part.dataset.symbolicaPath === firstRevealed)
        || parts.find(part => part.dataset.symbolicaPath === path) || marker || math;
      if (restoreExpansionScroll) {
        for (const {node, top, left} of expansion.scrollers) {
          node.scrollTop = top;
          node.scrollLeft = left;
        }
        for (const {node} of expansion.scrollers) {
          const rect = target.getBoundingClientRect();
          node.scrollTop += rect.top - expansion.rect.top;
          node.scrollLeft += rect.left - expansion.rect.left;
        }
      }
      if (restoreExpansionFocus) {
        restoringFocus = true;
        (marker || math).focus({preventScroll:true});
        restoringFocus = false;
      }
    }
    if (retainedPreview) {
      let anchor = retainedPreview;
      while (anchor.parent) anchor = anchor.parent;
      const button = [...content.querySelectorAll('button.omission')].find(button =>
        JSON.stringify(inspection.get(button)?.path) === JSON.stringify(anchor.path));
      if (!button) hidePreview();
      else {
        const target = inspection.get(button);
        // A hover follows the marker's current omitted tail, not just its path.
        // Restart a cancelled debounce or changed target; the new request ID
        // also prevents an in-flight response from restoring the old tail.
        if (!anchor.pinned && (!anchor.request || anchor.layout !== target.layout
            || anchor.markerStart !== target.start)) {
          showPreview(button, target.path, target.label, false, target.layout, target.start);
        } else {
          for (let state = retainedPreview; state; state = state.parent) state.button = button;
          button.setAttribute('aria-expanded', String(retainedPreview.pinned));
          button.setAttribute(retainedPreview.pinned ? 'aria-controls' : 'aria-describedby', preview.id);
          refitPreview();
        }
        if (restoreMarkerFocus) {
          restoringFocus = true;
          button.focus({ preventScroll: true });
          restoringFocus = false;
        }
      }
    }
  }
  function numberMarkers(container) {
    const changes = [];
    let number = 0;
    for (const marker of container.querySelectorAll('button.omission[data-numbered]')) {
      const text = String(++number);
      if (marker.textContent !== text) changes.push({marker, text});
      marker.setAttribute('aria-label', marker.getAttribute('aria-label').replace(/\[\d+\]$/, `[${number}]`));
    }
    // Numbering often changes only a tabular digit, leaving its button the
    // same size. Refit the expression only when actual dimensions change,
    // including multi-digit labels or fallback fonts with different metrics.
    // Batch reads and writes to avoid a layout flush for each marker.
    for (const change of changes) change.before = change.marker.getBoundingClientRect();
    for (const {marker, text} of changes) marker.textContent = text;
    return changes.some(({marker, before}) => {
      const after = marker.getBoundingClientRect();
      return after.width !== before.width || after.height !== before.height;
    });
  }
  function decorateMarkers(page, container = content, context = null) {
    for (const marker of container.querySelectorAll('[data-symbolica-marker]')) {
      const index = page.marker_ids.indexOf(Number(marker.dataset.symbolicaMarker));
      if (index === -1) continue;
      const [path, label] = page.markers[index];
      const b = document.createElement('button');
      b.type = 'button';
      b.className = 'omission';
      b.dataset.layout = page.layout || 0;
      b.textContent = marker.textContent.replace(/^\[(\d+)\]$/, '$1');
      if (/^\[\d+\]$/.test(marker.textContent)) b.dataset.numbered = '';
      b.setAttribute('aria-label', label.replace('Inspect ', 'Open '));
      bindInspection(b, path, label, page.layout || 0, context, page.marker_starts?.[index] || 0);
      if (context && JSON.stringify(path) === JSON.stringify(activePreview?.path)) {
        b.onclick = () => {
          activePreview.previous.push(activePreview.start);
          activePreview.start = page.marker_starts[index];
          requestPreview();
        };
      }
      marker.replaceChildren(b);
    }
  }
  function revealFittingParts(layouts, available, container = content, context = null, heightLimit = Infinity) {
    // Reuse only the bounded layouts already in this page. Providers can opt
    // in by marking corresponding MathML subtrees with data-symbolica-path.
    // Each marker keeps its source layout for valid kernel navigation.
    if (layouts.length < 2 || !container.querySelector('button.omission')) return;
    const sources = layouts.map(page => ({page, parts:sourceTemplate(page.html).parts}));
    let attempts = 0;
    let fittingWidth = available;
    const tried = new Set();
    for (let pass = 0; pass < layouts.length; pass++) {
      let changed = false;
      for (const marker of container.querySelectorAll('button.omission')) {
        if (!marker.isConnected) continue;
        const part = marker.closest('[data-symbolica-path]');
        if (!part?.dataset.symbolicaPath) continue;
        // Alternative pages can contain the exact same subtree. Compare the
        // decorated candidate so marker labels/context still participate in
        // the decision. A rejected candidate need not be fitted twice while
        // this marker's surroundings and width are unchanged. Keep this local
        // to one marker: another accepted refinement, resize or font load can
        // change whether the same content fits.
        const candidates = new Set();
        for (const { page, parts } of sources) {
          if ((page.layout || 0) >= Number(marker.dataset.layout)) continue;
          const key = `${part.dataset.symbolicaPath}:${marker.dataset.layout}:${page.layout || 0}`;
          if (tried.has(key)) continue;
          tried.add(key);
          const original = parts.get(part.dataset.symbolicaPath);
          if (!original) continue;
          const replacement = original.cloneNode(true);
          decorateMarkers(page, replacement, context);
          const candidate = `${fittingWidth}:${replacement.outerHTML}`;
          if (candidates.has(candidate)) continue;
          candidates.add(candidate);
          if (++attempts > 64) return;
          const trialWidth = fittingWidth;
          const rollback = wrappedWidths.get(container) === trialWidth ? wrappingCheckpoint(container) : null;
          part.replaceWith(replacement);
          setWrapWidth(container, fittingWidth);
          // Richer detail must fit each matrix/tensor cell too. A wide value
          // can overlap its neighbour without overflowing the outer frame.
          const metrics = layoutMetrics(container);
          const width = widthWithCells(container, context
            ? Math.max(container.scrollWidth - 4, (container.firstElementChild?.getBoundingClientRect().width || 0) / metrics.scale)
            : Math.max(container.scrollWidth, metrics.width), fittingWidth);
          // Keep DOM/transport-sized work bounded even when mixing layouts.
          fittingWidth = Math.min(fittingWidth, context ? container.clientWidth - 4 : math.clientWidth);
          if (width <= fittingWidth + 1 && metrics.height <= heightLimit + 1
              && container.querySelectorAll('*').length <= 10000
              && container.innerHTML.length <= 256 * 1024) {
            changed = true;
            break;
          }
          // A scrollbar can narrow the available width during a trial. Such
          // a change needs a fresh fit instead of the saved wider arrangement.
          if (rollback && fittingWidth === trialWidth) rollback();
          else {
            replacement.replaceWith(part);
            setWrapWidth(container, fittingWidth);
          }
        }
      }
      if (!changed) break;
    }
  }
  let lastWidth = -1, frame;
  let nearViewport = true, fitNeeded = false;
  const scheduleFit = () => {
    fitNeeded = true;
    cancelAnimationFrame(frame);
    if (!nearViewport) return;
    frame = requestAnimationFrame(() => {
      fitNeeded = false;
      update(true);
    });
  };
  // A notebook resize affects every output, including those far offscreen.
  // Keep their current layout and height until they approach the viewport.
  // Navigation still updates immediately; only width/font refits are deferred.
  const visibility = doc.defaultView.IntersectionObserver
    ? new doc.defaultView.IntersectionObserver(entries => {
      nearViewport = entries[entries.length - 1].isIntersecting;
      if (!nearViewport) cancelAnimationFrame(frame);
      else if (fitNeeded) scheduleFit();
    }, { rootMargin: '300px' }) : null;
  visibility?.observe(math);
  const observer = new ResizeObserver(entries => {
    const width = entries[0].contentRect.width;
    if (Math.abs(width - lastWidth) > 1) {
      lastWidth = width;
      scheduleFit();
    }
  });
  // STIX can arrive after the first layout. Refit when its metrics are known.
  const fontsLoaded = event => {
    if ([...(event.fontfaces || [])].some(face =>
        face.family.replace(/['"]/g, '') === 'STIX Two Math')) scheduleFit();
  };
  doc.fonts?.addEventListener('loadingdone', fontsLoaded);
  model.on('change:page', update);
  model.on('change:preview', updatePreview);
  update();
  lastWidth = math.clientWidth;
  observer.observe(math);
  model.send({ action: 'attach', view });
  return () => {
    doc.removeEventListener('pointerdown', outsidePointer, true);
    doc.removeEventListener('scroll', schedulePosition, true);
    doc.defaultView.removeEventListener('resize', schedulePosition);
    doc.defaultView.visualViewport?.removeEventListener('resize', schedulePosition);
    doc.defaultView.visualViewport?.removeEventListener('scroll', schedulePosition);
    observer.disconnect();
    visibility?.disconnect();
    cancelAnimationFrame(frame);
    cancelAnimationFrame(positionFrame);
    doc.fonts?.removeEventListener('loadingdone', fontsLoaded);
    model.off('change:page', update);
    model.off('change:preview', updatePreview);
    hidePreview();
    clearTimeout(zoomTimer);
    templates.clear();
    popupSizes.clear();
    pendingExpansions.length = 0;
    root.removeEventListener('keydown', onKeyDown);
    model.send({ action: 'dispose', view });
    root.replaceChildren();
  };
}
/** Mount the shared pager without a notebook or widget library.
 * `send(message)` returns the viewer's handle_message() response, synchronously
 * or as a Promise. Messages and responses contain only JSON-compatible data.
 * Await close() before releasing the corresponding viewer in the Python worker.
 */
export function mount({ el, page, send }) {
  if (typeof send !== 'function') throw new TypeError('send must be a function');
  const state = { page, preview: {} };
  const listeners = new Map();
  let closed = false;
  let pending = Promise.resolve();
  let queuedHover = null;
  const publish = response => {
    if (closed || response == null) return;
    const [field, value] = response;
    if (field === '') return;
    if (field !== 'page' && field !== 'preview') throw new Error('Unknown pager response');
    state[field] = value;
    for (const callback of listeners.get(`change:${field}`) || []) callback();
  };
  const model = {
    get: name => state[name],
    on(name, callback) {
      if (!listeners.has(name)) listeners.set(name, new Set());
      listeners.get(name).add(callback);
    },
    off(name, callback) { listeners.get(name)?.delete(callback); },
    send(message) {
      const disposing = message.action === 'dispose';
      if (closed && !disposing) return;
      // A moving pointer can queue several previews behind a busy worker.
      // Only the newest unsent hover remains useful. Pinned popup requests
      // and navigation are authoritative and must keep their original order.
      const hover = message.action === 'preview' && message.popup === false;
      const ticket = hover ? {} : null;
      queuedHover = ticket;
      pending = pending.then(async () => {
        if (closed && !disposing) return;
        if (hover && queuedHover !== ticket) return;
        if (hover) queuedHover = null;
        try {
          publish(await send(message));
        } catch (error) {
          const detail = String(error).slice(0, 1024);
          // Failed previews belong to their request, including a hover that
          // became stale while running. Never replace the main page with one.
          publish(message.action === 'preview'
            ? ['preview', { request: message.request, error: detail }]
            : ['page', { ...state.page, error: detail }]);
        }
      });
    },
  };
  const cleanup = render({ model, el });
  return {
    // Also supports transports that deliver responses through an event listener.
    update: publish,
    close() {
      if (!closed) {
        closed = true;
        cleanup();
        listeners.clear();
      }
      return pending;
    },
  };
}
export default { render };
