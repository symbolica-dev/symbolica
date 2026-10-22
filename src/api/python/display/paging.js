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
  <nav class="toolbar" aria-label="Expression pages"></nav>
  <div class="status" role="status" aria-live="polite"></div>
  <div class="math" tabindex="0" role="region" aria-label="Expression"
    aria-keyshortcuts="+ = - 0"
    title="Click or Tab to focus. + or = zooms in, − zooms out, 0 resets."
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
    zoomFeedback.textContent = `Zoom ${Math.round(scales[scaleIndex] * 100)}% · 0 to reset`;
    zoomFeedback.hidden = false;
    clearTimeout(zoomTimer);
    zoomTimer = setTimeout(() => { zoomFeedback.hidden = true; }, 1800);
    hidePreview();
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
  const previewCollapse = root.querySelector('.preview-collapse');
  pageArrow(previewPrevious, 'Previous', 'previous');
  pageArrow(previewNext, 'Next', 'next');
  const previewHint = root.querySelector('.preview-hint');
  preview.id = `symbolica-preview-${view}`;
  preview.tabIndex = -1;
  previewHeading.id = `${preview.id}-heading`;
  const popupSizes = new Map();
  let activePreview = null;
  const inspection = new WeakMap();
  let refitting = false;
  let previewSequence = 0;
  let hoverTimer, hideTimer, restoringFocus = false;
  let positionFrame, previewNeedsFit = false;
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
    previewExpand.disabled = previewCollapse.disabled = true;
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
    activePreview = { button, path, layout, markerStart, pinned, size, start: 0, previous: [] };
    const show = () => {
      preview.classList.toggle('pinned', pinned);
      preview.setAttribute('role', pinned ? 'dialog' : 'tooltip');
      preview.setAttribute('aria-labelledby', previewHeading.id);
      previewControls.hidden = previewClose.hidden = !pinned;
      previewBack.hidden = true;
      previewPrevious.hidden = previewNext.hidden = previewSize.hidden = true;
      previewExpand.hidden = !model.get('page').supports_expansion;
      previewCollapse.hidden = !model.get('page').can_collapse;
      previewHint.textContent = pinned
        ? (model.get('page').supports_expansion
          ? 'Show more expands the main view. Show less restores its compact display. Inspect opens this part separately.'
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
  function setWrapWidth(container, width) {
    const style = getComputedStyle(container);
    const padding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
    const available = Math.max(0, width - padding);
    container.style.setProperty('--symbolica-wrap-width', `${available}px`);
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
    for (const table of [...container.querySelectorAll('mtable[data-symbolica-product],mtable[data-symbolica-sum]')].reverse()) {
      const cell = table.closest('[data-symbolica-cell]');
      if (cell && cell !== container) continue;
      const rows = [...table.children].map(row => {
        const cell = row.children.length === 1 ? row.firstElementChild : null;
        const body = cell?.localName === 'mtd' && cell.children.length === 1
          ? cell.firstElementChild : null;
        const factors = body?.localName === 'mrow'
          && [...body.childNodes].every(node => node.nodeType === 1
            ? node.hasAttribute('data-symbolica-factor') || node.hasAttribute('data-symbolica-term')
            : node.nodeType === 3 && !node.textContent.trim())
          ? [...body.children] : [];
        return { row, factors };
      });
      const factorCount = rows.reduce((n, row) => n + row.factors.length, 0);
      if (factorCount < 2) continue;
      // Reserve surrounding signs, fences and fraction padding. Products
      // have zero outer cell padding, including after measured row packing.
      // Other table rows and wrapping sums/products do not restrict this table
      // to their current arrangement.
      const rowWidth = Math.max(0, ...rows.map(({ row }) =>
        row.querySelector(':scope > mtd > mrow')?.getBoundingClientRect().width || 0));
      const cellOverhead = Math.max(0, table.getBoundingClientRect().width - rowWidth);
      let overhead = cellOverhead;
      for (let child = table, parent = child.parentElement;
          parent && parent !== container; child = parent, parent = parent.parentElement) {
        if (['mtable', 'mtr', 'mtd'].includes(parent.localName)
            || (['flex', 'inline-flex'].includes(parent.style.display) && parent.style.flexWrap === 'wrap')
            || (parent.localName === 'mrow'
              && [...parent.children].some(node => node.hasAttribute('data-symbolica-factor') || node.hasAttribute('data-symbolica-term')))) continue;
        const children = [...parent.children];
        const childWidth = parent.localName === 'mfrac'
          ? Math.max(0, ...children.map(node => node.getBoundingClientRect().width))
          : child.getBoundingClientRect().width;
        overhead += Math.max(0, parent.getBoundingClientRect().width - childWidth);
      }
      const tableLimit = Math.max(0, available - (overhead - cellOverhead));
      let limit = Math.max(0, available - overhead);
      for (let attempt = 0; attempt < factorCount; attempt++) {
        const measured = rows.map(({ row, factors }) => ({
          row, factors: factors.map(node => ({ node, width: node.getBoundingClientRect().width }))
        }));
        const packed = [];
        let line, lineWidth = 0;
        for (const { row, factors } of measured) {
          if (!factors.length) {
            // In particular, keep a terminal product omission on its own row.
            packed.push(row);
            line = null;
            continue;
          }
          for (const factor of factors) {
            if (!line || lineWidth + factor.width > limit + 0.5) {
              const row = doc.createElementNS(table.namespaceURI, 'mtr');
              const cell = doc.createElementNS(table.namespaceURI, 'mtd');
              line = doc.createElementNS(table.namespaceURI, 'mrow');
              cell.append(line);
              row.append(cell);
              packed.push(row);
              lineWidth = 0;
            }
            line.append(factor.node);
            lineWidth += factor.width;
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
        // WebKit can change intrinsic cell widths when factors move between
        // rows. Check the packed table too, and only reduce this pass's budget.
        const overflow = table.getBoundingClientRect().width - tableLimit;
        if (overflow <= 0.5 || limit === 0) break;
        limit = Math.max(0, limit - overflow);
      }
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
      let selected = value, smallest = Infinity;
      for (const candidate of layouts) {
        previewMath.innerHTML = candidate.html;
        decorateMarkers(candidate, previewMath, value.context);
        setWrapWidth(previewMath, previewMath.clientWidth);
        const width = widthWithCells(previewMath,
          Math.max(previewMath.scrollWidth - 4, previewMath.firstElementChild?.getBoundingClientRect().width || 0),
          previewMath.clientWidth - 4);
        if (width < smallest) { selected = candidate; smallest = width; }
        // Rendering a tall candidate may introduce a vertical scrollbar and
        // reduce the available width on platforms with classic scrollbars.
        if (width <= previewMath.clientWidth - 4 + 1) { selected = candidate; break; }
      }
      previewMath.innerHTML = selected.html;
      decorateMarkers(selected, previewMath, value.context);
      setWrapWidth(previewMath, previewMath.clientWidth);
      revealFittingParts(layouts, previewMath.clientWidth - 4, previewMath, value.context);
      numberMarkers(previewMath);
    setWrapWidth(previewMath, previewMath.clientWidth);
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
      previewCollapse.disabled = false;
    }
    positionPreview();
  }
  function showSubPreview(path, label, layout, context) {
    const parent = activePreview;
    if (!parent) return;
    parent.pinned = true;
    parent.scrollTop = previewMath.scrollTop;
    parent.scrollLeft = previewMath.scrollLeft;
    activePreview = {button:parent.button, parent, path, layout, context,
      pinned:true, start:0, size:popupSizes.get(JSON.stringify(path)) || parent.size, previous:[]};
    preview.classList.add('pinned');
    preview.setAttribute('role', 'dialog');
    previewControls.hidden = previewClose.hidden = false;
    previewExpand.hidden = !model.get('page').supports_expansion;
    previewCollapse.hidden = !model.get('page').can_collapse;
    previewHint.textContent = 'Back returns to the previous popup. Inspect opens this part in the main view.';
    previewHeading.textContent = label.replace('Inspect ', '');
    previewMath.textContent = 'Loading preview…';
    previewBack.hidden = false;
    previewSize.value = activePreview.size;
    requestPreview();
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
        ? 'Show more expands the main view. Show less restores its compact display. Inspect opens this part separately.'
        : 'Inspect opens this part in the main view.';
    }
    preview.focus({preventScroll:true});
  };
  previewPrevious.onclick = () => {
    activePreview.start = activePreview.previous.pop() ?? 0;
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
    activePreview.start = 0;
    activePreview.previous = [];
    requestPreview();
  };
  previewExpand.onclick = () => {
    const { path, layout, size, context } = activePreview;
    hidePreview();
    model.send({ action: 'expand', value: path, layout, size, context });
  };
  previewCollapse.onclick = () => {
    hidePreview();
    model.send({ action: 'collapse' });
  };
  previewInspect.onclick = () => {
    const { path, layout, size, context } = activePreview;
    hidePreview();
    model.send({ action: 'open', value: path, layout, size, context });
  };
  previewClose.onclick = () => hidePreview(true);
  function bindInspection(button, path, label, layout, context = null, start = 0) {
    inspection.set(button, { path, label, layout, start });
    button.setAttribute('aria-haspopup', 'dialog');
    button.setAttribute('aria-expanded', 'false');
    if (context) {
      button.onclick = () => showSubPreview(path, label, layout, context);
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
    if (!math.contains(event.target) || event.ctrlKey || event.metaKey || event.altKey
      || event.isComposing || event.target.closest('input,textarea,select,[contenteditable]')) return;
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
  const send = (action, value) => model.send({ action, value, layout: currentLayout });
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
  function update(preservePreview = false) {
    const retainedPreview = preservePreview === true ? activePreview : null;
    const restoreMarkerFocus = retainedPreview && root.activeElement === retainedPreview.button;
    if (!retainedPreview) hidePreview();
    else {
      clearTimeout(hoverTimer);
      clearTimeout(hideTimer);
    }
    refitting = true;
    const base = model.get('page');
    const layouts = [base, ...(base.layouts || []).map(candidate => ({ ...base, ...candidate }))];
    // Measure native MathML with the actual font and notebook width. Keep the
    // first (most detailed) layout that fits; no resize messages to the kernel.
    // Zoom magnifies the chosen expression, rather than hiding detail again.
    const zoom = content.style.getPropertyValue('--expression-zoom');
    content.style.setProperty('--expression-zoom', 1);
    let available = math.clientWidth;
    let page = base, smallestWidth = Infinity, selectedNodes, shown;
    for (const candidate of layouts) {
      content.innerHTML = candidate.html;
      shown = candidate;
      decorateMarkers(candidate);
      // A taller candidate can give an enclosing notebook cell a scrollbar.
      // Keep the narrowest width observed during this pass, including changes
      // caused by trial layouts, so compacting cannot reopen a wider layout.
      available = Math.min(available, math.clientWidth);
      setWrapWidth(content, available);
      const width = widthWithCells(content,
        Math.max(content.scrollWidth, content.getBoundingClientRect().width), available);
      if (width < smallestWidth) {
        page = candidate;
        smallestWidth = width;
        selectedNodes = [...content.childNodes];
      }
      if (width <= available + 1) { page = candidate; break; }
    }
    currentLayout = page.layout || 0;
    content.dataset.layout = currentLayout;
    // Retain the winning DOM and its marker handlers instead of parsing,
    // decorating and fitting the same page a second time.
    if (shown !== page) {
      content.replaceChildren(...selectedNodes);
      setWrapWidth(content, available);
    }
    math.scrollLeft = 0;
    status.textContent = page.error || `Items ${page.total ? page.start + 1 : 0}–${page.end} of ${page.total} (${page.unit})${page.markers.length ? ' · Detail fitted to width; open a marker for more' : ''}`;
    previous.disabled = !page.previous;
    next.disabled = page.next === null;
    back.hidden = !page.back;
    const paginated = page.previous || page.next !== null;
    previous.hidden = next.hidden = !page.back || !paginated;
    controls.hidden = !page.back;
    math.classList.toggle('compact', controls.hidden);
    status.hidden = !page.error && !page.back;
    const finalWidth = Math.min(available, math.clientWidth);
    if (finalWidth < available - .5) setWrapWidth(content, finalWidth);
    available = finalWidth;
    revealFittingParts(layouts, available);
    if (numberMarkers(content)) setWrapWidth(content, available);
    content.style.setProperty('--expression-zoom', zoom || 1);
    refitting = false;
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
    let number = 0, changed = false;
    for (const marker of container.querySelectorAll('button.omission[data-numbered]')) {
      const text = String(++number);
      if (marker.textContent !== text) {
        marker.textContent = text;
        changed = true;
      }
      marker.setAttribute('aria-label', marker.getAttribute('aria-label').replace(/\[\d+\]$/, `[${number}]`));
    }
    return changed;
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
  function revealFittingParts(layouts, available, container = content, context = null) {
    // Reuse only the bounded layouts already in this page. Providers can opt
    // in by marking corresponding MathML subtrees with data-symbolica-path.
    // Each marker keeps its source layout for valid kernel navigation.
    if (layouts.length < 2 || !container.querySelector('button.omission')) return;
    const sources = layouts.map(page => {
      const template = document.createElement('template');
      template.innerHTML = page.html;
      return { page, parts: new Map([...template.content.querySelectorAll('[data-symbolica-path]')]
        .map(part => [part.dataset.symbolicaPath, part])) };
    });
    let attempts = 0;
    let fittingWidth = available;
    const tried = new Set();
    for (let pass = 0; pass < layouts.length; pass++) {
      let changed = false;
      for (const marker of container.querySelectorAll('button.omission')) {
        if (!marker.isConnected) continue;
        const part = marker.closest('[data-symbolica-path]');
        if (!part?.dataset.symbolicaPath) continue;
        for (const { page, parts } of sources) {
          if ((page.layout || 0) >= Number(marker.dataset.layout)) continue;
          const key = `${part.dataset.symbolicaPath}:${marker.dataset.layout}:${page.layout || 0}`;
          if (tried.has(key)) continue;
          tried.add(key);
          const original = parts.get(part.dataset.symbolicaPath);
          if (!original) continue;
          if (++attempts > 64) return;
          const replacement = original.cloneNode(true);
          decorateMarkers(page, replacement, context);
          part.replaceWith(replacement);
          setWrapWidth(container, fittingWidth);
          const width = context
            ? Math.max(container.scrollWidth - 4, container.firstElementChild?.getBoundingClientRect().width || 0)
            : Math.max(container.scrollWidth, container.getBoundingClientRect().width);
          // Keep DOM/transport-sized work bounded even when mixing layouts.
          fittingWidth = Math.min(fittingWidth, context ? container.clientWidth - 4 : math.clientWidth);
          if (width <= fittingWidth + 1 && container.querySelectorAll('*').length <= 10000
              && container.innerHTML.length <= 256 * 1024) {
            changed = true;
            break;
          }
          replacement.replaceWith(part);
          setWrapWidth(container, fittingWidth);
        }
      }
      if (!changed) break;
    }
  }
  let lastWidth = -1, frame;
  const scheduleFit = () => {
    cancelAnimationFrame(frame);
    frame = requestAnimationFrame(() => update(true));
  };
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
    cancelAnimationFrame(frame);
    cancelAnimationFrame(positionFrame);
    doc.fonts?.removeEventListener('loadingdone', fontsLoaded);
    model.off('change:page', update);
    model.off('change:preview', updatePreview);
    hidePreview();
    clearTimeout(zoomTimer);
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
      pending = pending.then(async () => {
        if (closed && !disposing) return;
        try {
          publish(await send(message));
        } catch (error) {
          publish(['page', { ...state.page, error: String(error).slice(0, 1024) }]);
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
