//! Bounded native MathML presentation. Omitted subtrees are display metadata,
//! never substitutions in the underlying expression.
use super::*;
mod compact;
mod cursor;
mod structured;
use cursor::MathMlChildren;
pub(crate) use cursor::MathMlCursorCache;
pub(crate) use structured::{
    AlgebraicMathMl, atom_coefficient, polynomial, rational_polynomial, series,
};

pub(crate) const DEFAULT_PAGE_SIZE: usize = 25;
// Bounded alternatives let each notebook view fit locally without kernel traffic.
pub(crate) const LAYOUTS: [(usize, usize); 8] = [
    (500, 32),
    (32, 24),
    (16, 20),
    (8, 16),
    (4, 12),
    (2, 8),
    (1, 4),
    (1, 2),
];
const MAX_NODES: usize = 1_000;
const MAX_DEPTH: usize = 32;
const MAX_LEAF_BYTES: usize = 4_096;
const MAX_TEXT_BYTES: usize = 16_384;
const MIN_BUDGET_PAGES: usize = 4;
const PRODUCT_ROW_COST: usize = 48;
// Scroll HTML containers, not the MathML root: a scrollport on inline math can
// clip glyph ink and create miniature scrollbars in grid layouts.
const MATH_OPEN: &str = "<math xmlns=\"http://www.w3.org/1998/Math/MathML\" displaystyle=\"true\" style=\"font-family:'STIX Two Math',math;font-size:21px;font-synthesis:none\">";
const MATH_CLOSE: &str = "</math>";

pub(crate) struct MathMlPage {
    pub html: String,
    pub start: usize,
    pub end: usize,
    pub total: usize,
    pub markers: Vec<MathMlMarker>,
    pub custom: bool,
}

pub(crate) struct MathMlMarker {
    pub path: Vec<usize>,
    pub label: String,
    pub start: usize,
}

impl MathMlPage {
    pub(crate) fn fragment(&self) -> std::borrow::Cow<'_, str> {
        let body = &self.html[MATH_OPEN.len()..self.html.len() - MATH_CLOSE.len()];
        if self.markers.is_empty() {
            return std::borrow::Cow::Borrowed(body);
        }
        // A plain formatter has no live page metadata. Keep omissions visible,
        // without creating marker links that cannot be resolved by its caller.
        let mut body = body.to_owned();
        for index in 0..self.markers.len() {
            body = body.replace(
                &format!(
                    "<mtext data-symbolica-marker=\"{index}\">[{}]</mtext>",
                    index + 1
                ),
                "<mtext>⋯</mtext>",
            );
        }
        std::borrow::Cow::Owned(body)
    }

    pub fn is_partial(&self) -> bool {
        self.start != 0 || self.end < self.total || !self.markers.is_empty()
    }
}

// Keep Python errors in the page source; the native renderer only needs to stop.
pub(crate) type SemanticPrinter<'a> =
    dyn FnMut(AtomView<'_>, &PrintOptions, &PrintState) -> Result<Option<String>, ()> + 'a;

struct Renderer<'a, 'r> {
    opts: &'a PrintOptions,
    state: PrintState,
    cursors: Option<&'a MathMlCursorCache<'a>>,
    nodes: usize,
    bytes: usize,
    budget_exhausted: bool,
    // Direct budget refusals still present in the current fragment. Sequence
    // rollback restores this counter when it replaces a rejected atom with a
    // meaningful suffix omission after a complete prefix.
    budget_holes: usize,
    nested_limit: usize,
    depth_limit: usize,
    expansions: &'a [(Vec<usize>, usize)],
    multiline_sums: bool,
    markers: Vec<MathMlMarker>,
    semantic: Option<&'r mut SemanticPrinter<'r>>,
    probing_compact_tail: bool,
    compact_probe_remaining: usize,
    failed: bool,
}

struct ProductFactor {
    html: String,
    needs_fence: bool,
    coefficient: bool,
}

struct FractionParts {
    numerator: String,
    denominator: ProductFactor,
}

// Estimate the visible token count, independently of the size of MathML markup.
// Stop as soon as the caller's budget is exceeded. Whitespace inside text tokens
// and character entities retain their existing row-packing cost.
pub(crate) fn visible_cost(fragment: &str, budget: usize) -> usize {
    let mut cost = 0;
    let mut tag = false;
    let mut tag_start = 0;
    let mut token_depth = 0usize;
    let mut entity = false;
    for (position, c) in fragment.char_indices() {
        match c {
            '<' => {
                tag = true;
                tag_start = position + 1;
            }
            '>' => {
                let markup = &fragment[tag_start..position];
                let name = markup
                    .trim_start_matches('/')
                    .split(|c: char| c.is_ascii_whitespace() || c == '/')
                    .next()
                    .unwrap_or_default();
                if matches!(name, "mi" | "mn" | "mo" | "mtext" | "ms") {
                    if markup.starts_with('/') {
                        token_depth = token_depth.saturating_sub(1);
                    } else if !markup.trim_end().ends_with('/') {
                        token_depth += 1;
                    }
                }
                tag = false;
            }
            _ if tag => (),
            '&' => entity = true,
            ';' if entity => {
                entity = false;
                cost += 1;
            }
            _ if entity => (),
            _ if token_depth == 0 && c.is_ascii_whitespace() => (),
            _ => cost += 1,
        }
        if cost > budget {
            break;
        }
    }
    cost
}

impl Renderer<'_, '_> {
    fn row(body: String) -> String {
        format!("<mrow>{body}</mrow>")
    }
    fn fence(body: String) -> String {
        // Natural-size parentheses avoid the extra advance reserved for
        // stretchy operators in browsers. Keep stretching for tall content
        // and omissions: browser refinement can replace a compact subtree
        // with a multiline one without replacing its surrounding fences.
        let tall = [
            "<mfrac",
            "<msqrt",
            "<mroot",
            "<msup",
            "<msub",
            "<munder",
            "<mover",
            "<mtable",
            "<mmultiscripts",
        ]
        .iter()
        .any(|tag| body.contains(tag))
            || body.contains("data-symbolica-marker=");
        let size = if tall { "" } else { " stretchy=\"false\"" };
        Self::row(format!("<mo{size}>(</mo>{body}<mo{size}>)</mo>"))
    }
    fn token(tag: &str, text: &str) -> String {
        format!("<{tag}>{}</{tag}>", AnsiHtmlFormatter::escape_html(text))
    }

    fn real_number(&self, text: &str, parts: &mut Option<FractionParts>) -> String {
        let (sign, value) = text
            .strip_prefix('-')
            .map_or(("", text), |v| ("<mo>−</mo>", v));
        let body = if let Some((n, d)) = value.split_once('/')
            && n.chars()
                .all(|c| c.is_ascii_digit() || Some(c) == self.opts.number_thousands_separator)
            && d.chars()
                .all(|c| c.is_ascii_digit() || Some(c) == self.opts.number_thousands_separator)
        {
            *parts = Some(FractionParts {
                numerator: Self::row(format!("{sign}{}", Self::token("mn", n))),
                denominator: ProductFactor {
                    html: Self::token("mn", d),
                    needs_fence: false,
                    coefficient: false,
                },
            });
            format!(
                "<mfrac>{}{}</mfrac>",
                Self::token("mn", n),
                Self::token("mn", d)
            )
        } else if let Some(number) = super::mathml_number(value) {
            number
        } else {
            // Exotic coefficient domains retain the existing exact notation.
            Self::token("mtext", value)
        };
        Self::row(format!("{sign}{body}"))
    }

    fn fraction(
        numerator: String,
        denominator: String,
        parts: &mut Option<FractionParts>,
    ) -> String {
        *parts = Some(FractionParts {
            numerator: numerator.clone(),
            denominator: ProductFactor {
                html: denominator.clone(),
                needs_fence: false,
                coefficient: false,
            },
        });
        format!("<mfrac>{numerator}{denominator}</mfrac>")
    }

    fn leading_minus(atom: AtomView<'_>, fragment: &str) -> Option<usize> {
        // A sum's first term can be negative without negating the whole sum.
        // Only our number and product renderers emit a sign for the whole atom.
        if !matches!(atom, AtomView::Num(_) | AtomView::Mul(_)) || atom.mathml_complex_coefficient()
        {
            return None;
        }
        let mut offset = 0;
        while fragment[offset..].starts_with("<mrow>") || fragment[offset..].starts_with("<mrow ") {
            offset += fragment[offset..].find('>')? + 1;
        }
        if fragment[offset..].starts_with("<mtable data-symbolica-product=\"\"") {
            // A wrapped product keeps its global sign in its first factor.
            // Follow only our table structure, not arbitrary nested content:
            // a custom factor can itself begin with a negative summand.
            for prefix in [
                "<mtable data-symbolica-product=\"\"",
                "<mtr>",
                "<mtd ",
                "<mrow>",
                "<mrow data-symbolica-factor=\"\"",
            ] {
                if !fragment[offset..].starts_with(prefix) {
                    return None;
                }
                offset += fragment[offset..].find('>')? + 1;
            }
        }
        fragment[offset..]
            .starts_with("<mo>−</mo>")
            .then_some(offset)
    }

    fn product_sign(atom: AtomView<'_>, mut fragment: String, negative: &mut bool) -> String {
        if let Some(offset) = Self::leading_minus(atom, &fragment) {
            *negative = !*negative;
            fragment.replace_range(offset..offset + "<mo>−</mo>".len(), "");
        }
        fragment
    }

    fn checkpoint(&self) -> (usize, usize, usize, bool, usize) {
        (
            self.nodes,
            self.bytes,
            self.markers.len(),
            self.budget_exhausted,
            self.budget_holes,
        )
    }

    fn restore(&mut self, state: (usize, usize, usize, bool, usize)) {
        self.nodes = state.0;
        self.bytes = state.1;
        self.markers.truncate(state.2);
        self.budget_exhausted = state.3;
        self.budget_holes = state.4;
    }

    fn tagged(body: String, path: &[usize]) -> String {
        let path = path
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join("/");
        format!("<mrow data-symbolica-path=\"{path}\">{body}</mrow>")
    }

    // A marker is roughly four short glyphs wide, including its padding. Probe
    // only that much structure: large trees and coefficients are never printed
    // just to decide whether abbreviating them would save space.
    fn compact_cost(&self, atom: AtomView<'_>, budget: usize) -> Option<usize> {
        if budget == 0 {
            return None;
        }
        let name_cost = |symbol: Symbol| {
            if symbol.get_name().len() > MAX_LEAF_BYTES {
                return None;
            }
            let mut name = String::new();
            symbol
                .format_with_custom(self.opts, PrintState::new(), false, &mut name)
                .ok()?;
            // MathML symbol formatting includes tags and escaped entities.
            // Only its displayed text contributes to this cheap estimate.
            let cost = visible_cost(&name, budget).max(1);
            (cost <= budget).then_some(cost)
        };
        let mut cost = match atom {
            AtomView::Var(v) => return name_cost(v.get_symbol()),
            AtomView::Num(n) => {
                let CoefficientView::Natural(r, 1, 0, 1) = n.get_coeff_view() else {
                    return None;
                };
                let cost = r.to_string().len();
                return (cost <= budget).then_some(cost);
            }
            AtomView::Fun(f) => name_cost(f.get_symbol())? + 2,
            _ => 1,
        };
        if cost > budget || atom.mathml_child_count() > budget {
            return None;
        }
        for (i, child) in atom.children().enumerate() {
            if i > 0 && !matches!(atom, AtomView::Pow(_)) {
                cost += 1;
            }
            cost += self.compact_cost(child, budget.checked_sub(cost)?)?;
        }
        (cost <= budget).then_some(cost)
    }

    fn compact_tail_limit(&self, atom: AtomView<'_>, start: usize, limit: usize) -> usize {
        let total = atom.mathml_child_count();
        let end = start.saturating_add(limit);
        if total <= end || total - end > 4 || total - start > 500 {
            return limit;
        }
        let mut cost = 0;
        for (i, child) in MathMlChildren::new(atom, end, self.cursors).enumerate() {
            let (_, child) = child;
            if i > 0 {
                cost += 1;
            }
            let Some(child_cost) = self.compact_cost(child, 4_usize.saturating_sub(cost)) else {
                return limit;
            };
            cost += child_cost;
        }
        total - start
    }

    // Probe only after rendering the retained prefix: scoped formatters may
    // allocate index aliases in visitation order. Keep the rendered fragments
    // and their budget charges on success, so callbacks are never repeated.
    // The few-glyph allowance favors readable notation; the browser still
    // selects the responsive layout using actual measured widths.
    fn compact_tail(
        &mut self,
        atom: AtomView<'_>,
        path: &mut Vec<usize>,
        depth: usize,
        start: usize,
        state: PrintState,
    ) -> Vec<(String, Option<FractionParts>)> {
        if depth == 0 {
            return vec![];
        }
        self.compact_rendered_tail(atom, path, depth, start..atom.mathml_child_count(), state)
    }

    fn compact_rendered_tail(
        &mut self,
        atom: AtomView<'_>,
        path: &mut Vec<usize>,
        depth: usize,
        range: std::ops::Range<usize>,
        state: PrintState,
    ) -> Vec<(String, Option<FractionParts>)> {
        if self.probing_compact_tail
            || self.compact_probe_remaining == 0
            || self.failed
            || self.budget_exhausted
            || range.len() > 4
            || self.hard_limit(atom, depth)
        {
            return vec![];
        }
        let checkpoint = self.checkpoint();
        self.probing_compact_tail = true;
        let mut tail = Vec::new();
        let mut visible = String::new();
        // A product omission includes a cross, spacing and a numbered button.
        // Keep short powers visible when replacing them saves little width.
        let product = matches!(atom, AtomView::Mul(_));
        let glyph_budget = if product { 6 } else { 4 };
        for (i, child) in MathMlChildren::new(atom, range.start, self.cursors).take(range.len()) {
            path.push(i);
            let mut parts = None;
            let html = self.render_factor(child, path, depth + 1, &mut parts, state);
            path.pop();
            if !tail.is_empty() {
                visible.push_str(if product {
                    "<mspace width=\"0.167em\"/>"
                } else {
                    "<mo>,</mo>"
                });
            }
            visible.push_str(&html);
            if self.failed
                || self.budget_exhausted
                || self.markers.len() != checkpoint.2
                || !compact::fits(&visible, glyph_budget)
            {
                self.restore(checkpoint);
                self.probing_compact_tail = false;
                return vec![];
            }
            tail.push((html, parts));
        }
        self.probing_compact_tail = false;
        tail
    }

    fn hole(&mut self, atom: AtomView<'_>, path: &[usize]) -> String {
        let marker = self.hole_at(atom, path, 0);
        // A collapsed call still exposes its head, while the marker opens its
        // arguments. Reserve the same name budget as an ordinary function.
        if let AtomView::Fun(f) = atom
            && f.get_symbol().get_name().len() <= MAX_LEAF_BYTES
            && self.bytes >= MAX_LEAF_BYTES
        {
            return Self::tagged(
                Self::row(format!(
                    "{}<mo>&#x2061;</mo>{}",
                    self.name(f.get_symbol()),
                    Self::fence(marker)
                )),
                path,
            );
        }
        marker
    }

    fn hole_at(&mut self, atom: AtomView<'_>, path: &[usize], start: usize) -> String {
        // Every omission retains a path, including an oversized indivisible leaf.
        let label = match atom {
            AtomView::Add(_) => "sum",
            AtomView::Mul(_) => "product",
            AtomView::Fun(_) => "function",
            AtomView::Pow(_) => "power",
            _ => "value exceeds display budget",
        };
        let index = self.markers.len();
        self.markers.push(MathMlMarker {
            path: path.to_vec(),
            label: format!("Inspect {label} [{}]", index + 1),
            start,
        });
        // The surrounding call supplies the function's type and name. An
        // abbreviated argument list reads F(x, [1]), without a generic label.
        if matches!(atom, AtomView::Fun(_))
            || (start == 0 && matches!(atom, AtomView::Add(_) | AtomView::Mul(_)))
        {
            return format!(
                "<mtext data-symbolica-marker=\"{index}\">[{}]</mtext>",
                index + 1
            );
        }
        let marker_type = match atom {
            AtomView::Add(_) => "+",
            AtomView::Mul(_) => "×",
            _ => {
                if atom.mathml_child_count() > 0 {
                    label
                } else {
                    "⋯"
                }
            }
        };
        // Leading MathML operators do not reliably honor rspace in browsers.
        // Use the same explicit gap as a product continuation in the viewer.
        let gap = if matches!(atom, AtomView::Mul(_)) {
            "<mspace width=\"0.25em\"/>"
        } else {
            ""
        };
        format!(
            "<mrow><mo form=\"infix\">{marker_type}</mo>{gap}<mtext data-symbolica-marker=\"{index}\">[{}]</mtext></mrow>",
            index + 1
        )
    }

    fn name(&mut self, symbol: Symbol) -> String {
        let name = symbol_html(symbol, self.opts);
        self.bytes = self.bytes.saturating_sub(name.len());
        name
    }

    fn render(&mut self, atom: AtomView<'_>, path: &mut Vec<usize>, depth: usize) -> String {
        self.render_factor(atom, path, depth, &mut None, self.state_at(depth))
    }

    fn state_at(&self, depth: usize) -> PrintState {
        if depth == 0 {
            return self.state;
        }
        PrintState {
            level: self.state.level.saturating_add(depth as u16),
            bracket_level: self.state.bracket_level,
            top_level_add_child: depth == 1,
            ..PrintState::new()
        }
    }

    fn semantic(
        &mut self,
        atom: AtomView<'_>,
        path: &[usize],
        state: PrintState,
    ) -> Option<String> {
        if !matches!(atom, AtomView::Var(_) | AtomView::Fun(_)) {
            return None;
        }
        let semantic = self.semantic.as_mut()?;
        match semantic(atom, self.opts, &state) {
            Ok(Some(fragment)) => self.custom_fragment(atom, path, fragment, true),
            Ok(None) => None,
            Err(()) => {
                self.failed = true;
                Some(String::new())
            }
        }
    }

    fn custom(&mut self, atom: AtomView<'_>, path: &[usize], state: PrintState) -> Option<String> {
        let symbol = atom.get_symbol()?;
        let callback = symbol.get_global_data().custom_print.as_ref()?;
        let fragment = callback(atom, self.opts, &state)?;
        self.custom_fragment(atom, path, fragment, false)
    }

    fn custom_fragment(
        &mut self,
        atom: AtomView<'_>,
        path: &[usize],
        fragment: String,
        semantic: bool,
    ) -> Option<String> {
        // Custom callbacks are responsible for bounded work. Their returned
        // fragment is an atomic presentation. Charge the generated grouping and
        // path too, including UTF-8 bytes and every opening/closing delimiter.
        let fragment = if semantic {
            Self::tagged(fragment, path)
        } else {
            Self::row(fragment)
        };
        let nodes = fragment.bytes().filter(|c| *c == b'<').count().max(1);
        // The per-leaf ceiling applies even when a page has a larger allowance.
        // Decline an oversized scoped fragment so its native arguments remain
        // navigable instead of replacing it with a marker for itself.
        if semantic && (fragment.len() > MAX_LEAF_BYTES || nodes > MAX_NODES) {
            return None;
        }
        if fragment.len() > MAX_LEAF_BYTES || fragment.len() > self.bytes || nodes > self.nodes {
            // A bounded semantic leaf only rejected by the remaining budget
            // must not fall back to raw tensor arguments. Let its containing
            // sum keep complete terms and omit the unfinished suffix instead.
            self.budget_exhausted = true;
            self.budget_holes += 1;
            return Some(self.hole(atom, path));
        }
        self.bytes -= fragment.len();
        self.nodes -= nodes;
        Some(fragment)
    }

    fn hard_limit(&self, atom: AtomView<'_>, depth: usize) -> bool {
        self.bytes < MAX_LEAF_BYTES || self.hard_semantic_limit(atom, depth)
    }

    fn hard_semantic_limit(&self, atom: AtomView<'_>, depth: usize) -> bool {
        depth >= 64
            || self.nodes == 0
            || self.bytes == 0
            || atom
                .get_symbol()
                .is_some_and(|s| s.get_name().len() > MAX_LEAF_BYTES)
            || (matches!(atom, AtomView::Num(_)) && atom.get_byte_size() > MAX_LEAF_BYTES)
    }

    fn render_factor(
        &mut self,
        atom: AtomView<'_>,
        path: &mut Vec<usize>,
        depth: usize,
        parts: &mut Option<FractionParts>,
        state: PrintState,
    ) -> String {
        if self.failed {
            return String::new();
        }
        if self.probing_compact_tail {
            // Rejected probes restore output budgets, but must not refund work.
            if self.compact_probe_remaining == 0 {
                return self.hole(atom, path);
            }
            self.compact_probe_remaining -= 1;
        }
        // Explicit expansion reveals a little more depth along this branch.
        // Both the absolute depth and the existing node/text budgets still apply.
        let depth_limit = self
            .expansions
            .iter()
            .filter(|(p, _)| p.starts_with(path) || path.starts_with(p))
            .fold(self.depth_limit, |limit, (p, _)| {
                limit.max((p.len() + 4).min(64))
            });
        if self.hard_semantic_limit(atom, depth) {
            self.budget_exhausted |= self.nodes == 0 || self.bytes < MAX_LEAF_BYTES;
            if self.nodes == 0 || self.bytes == 0 {
                self.budget_holes += 1;
            }
            return self.hole(atom, path);
        }
        if let Some(custom) = self.semantic(atom, path, state) {
            return custom;
        }
        // The native formatter reserves enough room for its largest leaf.
        // Scoped notation already has an exact byte/node check, so a small
        // tensor or scalar product can still use the remaining space.
        if self.bytes < MAX_LEAF_BYTES {
            self.budget_exhausted = true;
            self.budget_holes += 1;
            return self.hole(atom, path);
        }
        // A tail probe may render an opaque semantic leaf or a small native
        // expression, including powers of semantic leaves. Never walk a large
        // native subtree merely to decide whether its marker saves space.
        if self.probing_compact_tail
            && !matches!(atom, AtomView::Pow(_))
            && self.compact_cost(atom, 4).is_none()
        {
            return self.hole(atom, path);
        }
        if depth >= depth_limit
            && (atom.mathml_child_count() > 0)
            && (if matches!(atom, AtomView::Fun(_)) {
                // F([1]) retains the head, so only the arguments count
                // toward the space saved, even for a long function name.
                self.compact_tail_limit(atom, 0, 0) < atom.mathml_child_count()
            } else {
                self.compact_cost(atom, 4).is_none()
            })
        {
            return self.hole(atom, path);
        }
        self.nodes -= 1;
        if let Some(custom) = self.custom(atom, path, state) {
            return custom;
        }
        match atom {
            AtomView::Num(_) => {
                if self.opts.precision.is_some_and(|p| p > MAX_LEAF_BYTES) {
                    return self.hole(atom, path);
                }
                let text = atom
                    .printer(PrintOptions {
                        mode: PrintMode::Symbolica,
                        ..self.opts.clone()
                    })
                    .to_string();
                if text.len() > MAX_LEAF_BYTES {
                    return self.hole(atom, path);
                }
                self.bytes = self.bytes.saturating_sub(text.len());
                // Render the components separately: a leading real minus must
                // never negate the imaginary part, and signs need MathML spacing.
                let components = match atom {
                    AtomView::Num(n) => match n.get_coeff_view() {
                        CoefficientView::Natural(r, d, i, di) if i != 0 => Some((
                            Rational::from_int_unchecked(r, d).into(),
                            Rational::from_int_unchecked(i, di).into(),
                        )),
                        CoefficientView::Large(r, i) if !i.is_zero() => {
                            Some((r.to_rat().into(), i.to_rat().into()))
                        }
                        CoefficientView::Float(r, i) if !i.is_zero() => {
                            Some((r.to_float().into(), i.to_float().into()))
                        }
                        _ => None,
                    },
                    _ => unreachable!(),
                };
                if let Some((real, imag)) = components {
                    let real: crate::coefficient::Coefficient = real;
                    let imag: crate::coefficient::Coefficient = imag;
                    let real_zero = real.is_zero();
                    let negative = imag.is_negative();
                    let imag = if negative { -imag } else { imag };
                    let scalar = |coefficient| {
                        let scalar = Atom::num(coefficient);
                        let text = scalar
                            .as_view()
                            .printer(PrintOptions {
                                mode: PrintMode::Symbolica,
                                ..self.opts.clone()
                            })
                            .to_string();
                        self.real_number(&text, &mut None)
                    };
                    let unit = "<mi>𝑖</mi>";
                    let imaginary = if imag.is_one() {
                        unit.to_owned()
                    } else {
                        Self::row(format!("{}<mo>&#x2062;</mo>{unit}", scalar(imag)))
                    };
                    let real = if real_zero {
                        String::new()
                    } else {
                        scalar(real)
                    };
                    let sign = if negative {
                        "<mo>−</mo>"
                    } else if !real_zero {
                        "<mo>+</mo>"
                    } else {
                        ""
                    };
                    return Self::row(format!("{real}{sign}{imaginary}"));
                }
                self.real_number(&text, parts)
            }
            AtomView::Var(v) => self.name(v.get_symbol()),
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                path.push(0);
                let mut b = self.render_factor(
                    base,
                    path,
                    depth + 1,
                    &mut None,
                    PrintState {
                        in_exp_base: true,
                        ..self.state_at(depth + 1)
                    },
                );
                path.pop();
                if let AtomView::Num(n) = exp {
                    if let CoefficientView::Natural(1, 2, 0, 1) = n.get_coeff_view() {
                        return format!("<msqrt>{b}</msqrt>");
                    }
                    if let CoefficientView::Natural(-1, 1, 0, 1) = n.get_coeff_view() {
                        let html = Self::fraction("<mn>1</mn>".to_owned(), b, parts);
                        parts.as_mut().unwrap().denominator.needs_fence =
                            matches!(base, AtomView::Add(_));
                        return html;
                    }
                    if let CoefficientView::Natural(-1, 2, 0, 1) = n.get_coeff_view() {
                        return Self::fraction(
                            "<mn>1</mn>".to_owned(),
                            format!("<msqrt>{b}</msqrt>"),
                            parts,
                        );
                    }
                }
                let positive_integer = matches!(base, AtomView::Num(n)
                    if matches!(n.get_coeff_view(), CoefficientView::Natural(r, 1, 0, 1) if r >= 0));
                if !matches!(base, AtomView::Var(_) | AtomView::Fun(_)) && !positive_integer {
                    b = Self::fence(b);
                }
                path.push(1);
                let e = self.render_factor(
                    exp,
                    path,
                    depth + 1,
                    &mut None,
                    PrintState {
                        in_exp: true,
                        ..self.state_at(depth + 1)
                    },
                );
                path.pop();
                let real_exponent = if let AtomView::Num(n) = exp {
                    match n.get_coeff_view() {
                        CoefficientView::Natural(_, _, i, _) => i == 0,
                        CoefficientView::Large(_, i) => i.is_zero(),
                        CoefficientView::Float(_, i) => i.is_zero(),
                        _ => false,
                    }
                } else {
                    false
                };
                if real_exponent && let Some(magnitude) = e.strip_prefix("<mrow><mo>−</mo>") {
                    return Self::fraction(
                        "<mn>1</mn>".to_owned(),
                        format!("<msup>{b}<mrow>{magnitude}</msup>"),
                        parts,
                    );
                }
                format!("<msup>{b}{e}</msup>")
            }
            _ => self.sequence(atom, path, depth, 0, self.nested_limit).0,
        }
    }

    fn sequence(
        &mut self,
        atom: AtomView<'_>,
        path: &mut Vec<usize>,
        depth: usize,
        start: usize,
        limit: usize,
    ) -> (String, usize) {
        let (body, end) = self.sequence_inner(atom, path, depth, start, limit);
        (
            if depth > 0 {
                Self::tagged(body, path)
            } else {
                body
            },
            end,
        )
    }

    fn sequence_inner(
        &mut self,
        atom: AtomView<'_>,
        path: &mut Vec<usize>,
        depth: usize,
        start: usize,
        limit: usize,
    ) -> (String, usize) {
        let mut limit = self
            .expansions
            .iter()
            .find(|(p, _)| p == path)
            .map_or(limit, |(_, count)| *count);
        // A compact alternative must retain the ancestors of an expanded part.
        for (p, _) in self.expansions {
            if depth > 0 && p.starts_with(path) && p.len() > path.len() {
                limit = limit.max((p[path.len()] + 1).saturating_sub(start));
            }
        }
        // Responsive abbreviation should not replace a short trailing argument
        // or factor with a wider marker. Explicit root page sizes stay exact.
        let limit = if depth > 0 && atom.mathml_derivative_head().is_none() {
            self.compact_tail_limit(atom, start, limit.min(500))
        } else {
            limit.min(500)
        };
        if matches!(atom, AtomView::Mul(_)) {
            return self.product(atom, path, depth, start, limit);
        }
        if let Some((count, symbol)) = atom.mathml_derivative_head() {
            if symbol.get_name().len() > MAX_LEAF_BYTES {
                return (self.hole(atom, path), count);
            }
            let AtomView::Fun(f) = atom else {
                unreachable!()
            };
            let head = self.name(symbol);
            let mut orders = String::new();
            let mut arguments = String::new();
            let mut end = start;
            for i in start..count.min(start.saturating_add(limit)) {
                if end > start && (self.nodes == 0 || self.bytes < MAX_LEAF_BYTES) {
                    self.budget_exhausted = true;
                    break;
                }
                let checkpoint = self.checkpoint();
                path.push(i);
                let mut order = self.render(f.get(i), path, depth + 1);
                path.pop();
                path.push(count + 1 + i);
                let mut argument = self.render(f.get(count + 1 + i), path, depth + 1);
                path.pop();
                if depth == 0 && self.budget_exhausted {
                    self.restore(checkpoint);
                    if end > start {
                        break;
                    }
                    path.push(i);
                    order = if self.compact_cost(f.get(i), 4).is_some() {
                        self.render(f.get(i), path, depth + 1)
                    } else {
                        self.hole(f.get(i), path)
                    };
                    path.pop();
                    path.push(count + 1 + i);
                    argument = self.hole(f.get(count + 1 + i), path);
                    path.pop();
                    self.nodes = 0;
                }
                if i > start {
                    orders.push_str("<mo>,</mo>");
                    arguments.push_str("<mo>,</mo>");
                }
                orders.push_str(&order);
                arguments.push_str(&argument);
                end = i + 1;
            }
            if start > 0 {
                orders.insert_str(0, "<mtext>⋯</mtext><mo>,</mo>");
                arguments.insert_str(0, "<mtext>⋯</mtext><mo>,</mo>");
            }
            if end < count {
                orders.push_str("<mo>,</mo><mtext>⋯</mtext>");
                arguments.push_str("<mo>,</mo>");
                arguments.push_str(&self.hole_at(atom, path, end));
            }
            return (
                Self::row(format!(
                    "<msup>{head}{}</msup><mo>&#x2061;</mo>{}",
                    Self::fence(orders),
                    Self::fence(arguments)
                )),
                end,
            );
        }
        let mut body = String::new();
        let expanded_sum = depth > 0
            && matches!(atom, AtomView::Add(_))
            && (self.multiline_sums || self.expansions.iter().any(|(p, _)| p == path));
        let previous_nested_limit = self.nested_limit;
        let previous_depth_limit = self.depth_limit;
        if matches!(atom, AtomView::Add(_)) && (depth == 0 || expanded_sum) {
            // Wrapped sums vary the number of complete terms, rather than
            // paying for additional rows by folding each term's factors.
            // Keep the ordinary node/text/depth bounds: if a term exhausts
            // them, the loop retains one navigable sum suffix.
            self.nested_limit = LAYOUTS[0].0;
            self.depth_limit = self.depth_limit.max((depth + 4).min(MAX_DEPTH));
        }
        let mut expanded_terms = Vec::new();
        let markers_before = self.markers.len();
        let mut arguments = Vec::new();
        let wrap = depth == 0 && matches!(atom, AtomView::Add(_));
        // Reserve the function head before rendering its arguments, so nested
        // long names cannot evade the text budget while unwinding recursion.
        let head = if let AtomView::Fun(f) = atom {
            Some(self.name(f.get_symbol()))
        } else {
            None
        };
        let mut end = start;
        let total = atom.mathml_child_count();
        let mut tail = vec![].into_iter();
        for (i, child) in MathMlChildren::new(atom, start, self.cursors).take(limit + 4) {
            if self.failed {
                break;
            }
            if i == start + limit {
                tail = self
                    .compact_tail(atom, path, depth, i, self.state_at(depth + 1))
                    .into_iter();
            }
            let preview = if i >= start + limit {
                let Some((html, _)) = tail.next() else {
                    break;
                };
                Some(html)
            } else {
                None
            };
            // Stop a page before the budget runs out. Nested lists become one
            // navigable omission instead of emitting an unbounded set of markers.
            if preview.is_none() && end > start && (self.nodes == 0 || self.bytes < MAX_LEAF_BYTES)
            {
                self.budget_exhausted = true;
                break;
            }
            let checkpoint = self.checkpoint();
            path.push(i);
            let mut fragment = preview.unwrap_or_else(|| self.render(child, path, depth + 1));
            // Root pages keep their first bounded preview so opening an
            // oversized item always makes progress. A nested sum with no
            // complete first term should instead omit that whole sum, rather
            // than exposing a tiny tensor as g([n]). Native calls can likewise
            // retain complete arguments before a rejected atomic argument.
            let keep_complete_prefix = if end > start {
                depth == 0 || matches!(atom, AtomView::Add(_)) || self.budget_holes > checkpoint.4
            } else {
                depth > 0 && matches!(atom, AtomView::Add(_))
            };
            if self.budget_exhausted && keep_complete_prefix {
                self.restore(checkpoint);
                self.budget_exhausted = true;
                path.pop();
                break;
            }
            path.pop();
            let term_start = body.len();
            let separator = match atom {
                AtomView::Add(_) if wrap || expanded_sum => "<mo form=\"infix\">+</mo>".to_owned(),
                AtomView::Add(_) => "<mo>+</mo>".to_owned(),
                _ => "<mo>,</mo>".to_owned(),
            };
            if wrap {
                body.push_str("<mrow data-symbolica-term=\"\" style=\"flex:0 0 auto\">");
            }
            if i > start {
                if matches!(atom, AtomView::Add(_))
                    && let Some(offset) = Self::leading_minus(child, &fragment)
                {
                    let sign = if wrap || expanded_sum {
                        "<mo form=\"infix\">−</mo>"
                    } else {
                        "<mo>−</mo>"
                    };
                    if fragment[..offset].contains("<mtable data-symbolica-product=") {
                        // Keep an expanded negative term's sign beside its
                        // first factor instead of centering it beside the table.
                        fragment.replace_range(offset..offset + "<mo>−</mo>".len(), sign);
                    } else {
                        body.push_str(sign);
                        fragment.replace_range(offset..offset + "<mo>−</mo>".len(), "");
                    }
                } else if !body.ends_with("<mo>−</mo>") {
                    body.push_str(&separator);
                }
            }
            body.push_str(&fragment);
            if head.is_some() {
                arguments.push(fragment);
            }
            if wrap {
                body.push_str("</mrow>");
            }
            if expanded_sum {
                expanded_terms.push(body[term_start..].to_owned());
            }
            end = i + 1;
        }
        if start > 0 {
            body.insert_str(0, "<mtext>⋯</mtext><mspace width=\"0.5em\"/>");
            if head.is_some() {
                arguments.insert(0, "<mtext>⋯</mtext>".to_owned());
            }
        }
        let nested_markers = self.markers.len() > markers_before;
        if end < total {
            // Root omissions also open a popup, so ordinary output needs no
            // global pager. The inspection view still offers Previous/Next.
            if !matches!(atom, AtomView::Add(_)) {
                body.push_str("<mo>,</mo>");
            }
            let omission = self.hole_at(atom, path, end);
            body.push_str(&omission);
            if expanded_sum {
                expanded_terms.push(omission.clone());
            }
            if head.is_some() {
                arguments.push(omission);
            }
        }
        // The larger child budget keeps each term complete; it must not
        // widen the compact layout's fallback rows and prevent mobile wraps.
        let columns = previous_nested_limit.clamp(2, if self.multiline_sums { 6 } else { 8 });
        self.nested_limit = previous_nested_limit;
        self.depth_limit = previous_depth_limit;
        // A few wide terms can overflow just as easily as a long sum. Keep
        // their boundaries available to the browser even when the static
        // fallback fits them on one row. Zero cell padding preserves inline
        // spacing; the row gap applies only when the sum actually wraps.
        if expanded_sum && expanded_terms.len() > 1 {
            body = "<mtable data-symbolica-sum=\"\" data-symbolica-row-gap=\"0.35em\" columnalign=\"left\" rowspacing=\"0\">"
                .to_owned();
            for (row, terms) in expanded_terms.chunks(columns).enumerate() {
                let padding = if row == 0 { "0" } else { "0.35em 0 0" };
                body.push_str(&format!(
                    "<mtr><mtd style=\"padding:{padding}\">{}</mtd></mtr>",
                    Self::row(
                        terms
                            .iter()
                            .map(|term| { format!("<mrow data-symbolica-term=\"\">{term}</mrow>") })
                            .collect()
                    )
                ));
            }
            body.push_str("</mtable>");
        }
        if let Some(head) = head {
            // Keep sibling arguments visible when one contains an omission.
            // Native MathML table rows preserve math layout inside each argument.
            if arguments.len() > 1 && (nested_markers || (end < total && arguments.len() > 3)) {
                let last = arguments.len() - 1;
                body = "<mtable columnalign=\"left\" rowspacing=\"0.45em\">".to_owned();
                for (index, argument) in arguments.into_iter().enumerate() {
                    let comma = if index < last { "<mo>,</mo>" } else { "" };
                    body.push_str(&format!("<mtr><mtd>{argument}{comma}</mtd></mtr>"));
                }
                body.push_str("</mtable>");
            }
            body = format!("{}<mo>&#x2061;</mo>{}", head, Self::fence(body));
        }
        let body = if wrap {
            format!(
                "<mrow style=\"display:inline-flex;flex-wrap:wrap;align-items:baseline;max-width:var(--symbolica-wrap-width,min(650px,80vw));row-gap:.5em\">{body}</mrow>"
            )
        } else {
            Self::row(body)
        };
        (body, end)
    }

    fn product(
        &mut self,
        atom: AtomView<'_>,
        path: &mut Vec<usize>,
        depth: usize,
        start: usize,
        limit: usize,
    ) -> (String, usize) {
        // Every root product needs factor boundaries: even two factors may be
        // wider than the available notebook cell. The browser repacks these
        // rows with the loaded font, while static output keeps a bounded fallback.
        let expanded = self.expansions.iter().any(|(p, _)| p == path);
        let multiline = depth == 0 || expanded;
        let previous_nested_limit = self.nested_limit;
        if expanded {
            // Reveal factors before spending the entire page budget inside the
            // first few sums. Detailed subexpressions remain independently
            // inspectable. All ordinary traversal limits still apply.
            let count = atom
                .mathml_child_count()
                .saturating_sub(start)
                .min(limit)
                .max(1);
            self.nested_limit = self.nested_limit.min((self.nodes / count / 8).max(1));
        }
        let mut numerator = Vec::new();
        let mut denominator = Vec::new();
        let mut negative = false;
        let mut end = start;
        let state = PrintState {
            in_product: true,
            suppress_one: true,
            ..self.state_at(depth + 1)
        };
        let mut tail = vec![].into_iter();
        let requested_end = start.saturating_add(limit).min(atom.mathml_child_count());
        let mut reserved_tail: Option<std::vec::IntoIter<(String, Option<FractionParts>)>> = None;
        for (i, child) in MathMlChildren::new(atom, start, self.cursors).take(limit + 4) {
            if self.failed {
                break;
            }
            if i == start + limit {
                tail = self.compact_tail(atom, path, depth, i, state).into_iter();
            }
            let preview = if let Some(tail) = &mut reserved_tail {
                let Some(fragment) = tail.next() else {
                    break;
                };
                Some(fragment)
            } else if i >= start + limit {
                let Some(fragment) = tail.next() else {
                    break;
                };
                Some(fragment)
            } else {
                None
            };
            if preview.is_none() && end > start && (self.nodes == 0 || self.bytes < MAX_LEAF_BYTES)
            {
                self.budget_exhausted = true;
                break;
            }
            let checkpoint = self.checkpoint();
            path.push(i);
            let mut parts = None;
            let previous_multiline_sums = self.multiline_sums;
            self.multiline_sums |= multiline;
            // Keep a small part of the existing page budget for a few following
            // factors. A large sum must not spend the resources needed to show
            // an otherwise compact scalar, such as C_A^4. Defer its formatting
            // until after the prefix so scoped aliases retain visitation order.
            let remaining = requested_end.saturating_sub(i + 1);
            let reserve = if depth == 0 && preview.is_none() && (1..=4).contains(&remaining) {
                (
                    (64 * remaining).min(self.nodes / 4),
                    (MAX_LEAF_BYTES + 512 * remaining)
                        .min(self.bytes.saturating_sub(MAX_LEAF_BYTES)),
                )
            } else {
                (0, 0)
            };
            self.nodes -= reserve.0;
            self.bytes -= reserve.1;
            let mut html = if let Some((html, preview_parts)) = preview {
                parts = preview_parts;
                html
            } else {
                self.render_factor(child, path, depth + 1, &mut parts, state)
            };
            self.nodes += reserve.0;
            self.bytes += reserve.1;
            self.multiline_sums = previous_multiline_sums;
            path.pop();
            // Keep useful prefixes of large sums, functions and denominators.
            // Only roll back a later factor when it still contains a direct
            // atom-level budget refusal; its enclosing product tail is the
            // meaningful omission. Do this before changing signs or fractions.
            if end > start && self.budget_holes > checkpoint.4 {
                self.restore(checkpoint);
                self.budget_exhausted = true;
                break;
            }
            if let Some(parts) = parts {
                let n = Self::product_sign(child, parts.numerator, &mut negative);
                if n != "<mn>1</mn>" && n != "<mrow><mn>1</mn></mrow>" {
                    numerator.push(ProductFactor {
                        html: n,
                        needs_fence: false,
                        coefficient: matches!(child, AtomView::Num(_)),
                    });
                }
                denominator.push(parts.denominator);
            } else {
                html = Self::product_sign(child, html, &mut negative);
                if html != "<mrow><mn>1</mn></mrow>" {
                    numerator.push(ProductFactor {
                        html,
                        needs_fence: matches!(child, AtomView::Add(_))
                            || child.mathml_complex_coefficient(),
                        coefficient: matches!(child, AtomView::Num(n) if n.get_coeff_view().is_real()),
                    });
                }
            }
            end = i + 1;
            // Spend the reserve only on complete, compact factors within the
            // requested page. A larger remainder keeps its navigable omission.
            if self.budget_exhausted {
                if depth > 0 {
                    break;
                }
                self.budget_exhausted = false;
                let retained =
                    self.compact_rendered_tail(atom, path, depth, end..requested_end, state);
                if retained.is_empty() {
                    self.budget_exhausted = true;
                    break;
                }
                reserved_tail = Some(retained.into_iter());
            }
        }
        self.nested_limit = previous_nested_limit;
        let mut omission = (end < atom.mathml_child_count()).then(|| self.hole_at(atom, path, end));
        let fraction = !denominator.is_empty();
        let sign_in_factor = negative && multiline && !fraction;
        let join = |factors: Vec<ProductFactor>, tail: Option<String>, leading_sign: bool| {
            let mut result = String::new();
            let mut rows = Vec::new();
            let mut row_cost = 0;
            let mut previous_numeric = false;
            // A fraction bar already groups its numerator and denominator.
            // Fence a sum only when it is multiplied by another factor.
            let multiple = !fraction || factors.len() > 1 || tail.is_some();
            for (i, factor) in factors.into_iter().enumerate() {
                let coefficient = factor.coefficient;
                let factor = if multiple && factor.needs_fence {
                    Self::fence(factor.html)
                } else {
                    factor.html
                };
                if multiline {
                    let cost = visible_cost(&factor, PRODUCT_ROW_COST)
                        + usize::from(i > 0)
                        + usize::from(leading_sign && i == 0);
                    if !result.is_empty() && row_cost + cost > PRODUCT_ROW_COST {
                        rows.push(std::mem::take(&mut result));
                        row_cost = 0;
                    }
                    row_cost += cost;
                    result.push_str(if coefficient {
                        "<mrow data-symbolica-factor=\"\" data-symbolica-coefficient=\"\">"
                    } else {
                        "<mrow data-symbolica-factor=\"\">"
                    });
                }
                let mut leading = factor.as_str();
                while let Some(rest) = leading
                    .strip_prefix("<mrow>")
                    .or_else(|| leading.strip_prefix("<msup>"))
                {
                    leading = rest;
                }
                let numeric = leading.starts_with("<mn>");
                if i > 0 && !result.ends_with("<mo>−</mo>") {
                    if self.opts.multiplication_operator == '·' && !(previous_numeric && numeric) {
                        result.push_str("<mo>&#x2062;</mo><mspace width=\"0.167em\"/>");
                    } else {
                        result.push_str(&Self::token(
                            "mo",
                            &self.opts.multiplication_operator.to_string(),
                        ));
                    }
                }
                if leading_sign && i == 0 {
                    result.push_str("<mo>−</mo>");
                }
                result.push_str(&factor);
                if multiline {
                    result.push_str("</mrow>");
                }
                previous_numeric = numeric;
            }
            if result.is_empty() {
                if multiline && (tail.is_some() || leading_sign) {
                    // A reciprocal-only prefix still needs a packable
                    // numerator: 1 and the omitted product share a row.
                    result.push_str("<mrow data-symbolica-factor=\"\">");
                    if leading_sign {
                        result.push_str("<mo>−</mo>");
                    }
                    result.push_str("<mn>1</mn></mrow>");
                } else {
                    result.push_str("<mn>1</mn>");
                }
            }
            if multiline {
                rows.push(result);
                if let Some(tail) = tail {
                    // The omitted suffix is a product factor too. Keep its
                    // leading operator directly inside the factor so browser
                    // packing can replace it with a continuation operator.
                    let tail = tail
                        .strip_prefix("<mrow>")
                        .and_then(|body| body.strip_suffix("</mrow>"))
                        .unwrap_or(&tail);
                    rows.push(format!("<mrow data-symbolica-factor=\"\">{tail}</mrow>"));
                }
                let rows = rows
                    .into_iter()
                    .enumerate()
                    .map(|(index, r)| {
                        // Table cells have browser-default padding on every
                        // edge. Products need no outer space: add only the
                        // intended gap above continuation rows.
                        let padding = if index == 0 { "0" } else { "0.5em 0 0" };
                        format!(
                            "<mtr><mtd style=\"padding:{padding}\">{}</mtd></mtr>",
                            Self::row(r)
                        )
                    })
                    .collect::<String>();
                format!(
                    "<mtable data-symbolica-product=\"\" data-symbolica-row-gap=\"0.5em\" columnalign=\"left\" rowspacing=\"0\">{rows}</mtable>"
                )
            } else {
                Self::row(result)
            }
        };
        let n = join(
            numerator,
            if multiline {
                // The omitted product multiplies the known quotient. Keeping
                // it in the numerator also keeps it in the final visible row.
                omission.take()
            } else {
                None
            },
            sign_in_factor,
        );
        let mut body = if denominator.is_empty() {
            n
        } else {
            let denominator = join(denominator, None, false);
            if multiline {
                // WebKit mislays fractions with direct table arguments. Group
                // each table without changing the numerator or denominator.
                // The browser can move a small denominator into a prefactor
                // when this canonical stacked layout needs continuation rows.
                format!(
                    "<mfrac data-symbolica-product-fraction=\"\">{}{}</mfrac>",
                    Self::row(n),
                    Self::row(denominator)
                )
            } else {
                format!("<mfrac>{n}{denominator}</mfrac>")
            }
        };
        if negative && !sign_in_factor {
            body.insert_str(0, "<mo>−</mo>");
        }
        if start > 0 {
            body.insert_str(0, "<mtext>⋯</mtext><mspace width=\"0.5em\"/>");
        }
        if let Some(omission) = omission {
            body.push_str(&omission);
        }
        (Self::row(body), end)
    }
}

impl<'a> AtomView<'a> {
    fn mathml_children(self) -> crate::atom::ListIterator<'a> {
        match self {
            AtomView::Add(a) => a.iter(),
            AtomView::Mul(a) => a.iter(),
            AtomView::Fun(a) => a.iter(),
            AtomView::Pow(a) => a.iter(),
            _ => unreachable!("MathML sequence iterator requested for a leaf"),
        }
    }

    pub(super) fn format_mathml<W: std::fmt::Write>(
        self,
        f: &mut W,
        options: &PrintOptions,
        state: PrintState,
    ) -> Result<bool, std::fmt::Error> {
        let page = self
            .render_mathml_layout_with_state(
                options,
                0,
                options.max_terms.unwrap_or(usize::MAX),
                0,
                &[],
                state,
                None,
                None,
            )
            .unwrap();
        f.write_str(&page.fragment())?;
        Ok(false)
    }

    fn mathml_child_count(self) -> usize {
        match self {
            AtomView::Add(a) => a.get_nargs(),
            AtomView::Mul(a) => a.get_nargs(),
            AtomView::Fun(a) => a.get_nargs(),
            AtomView::Pow(_) => 2,
            _ => 0,
        }
    }

    fn mathml_complex_coefficient(self) -> bool {
        let AtomView::Num(n) = self else {
            return false;
        };
        match n.get_coeff_view() {
            CoefficientView::Natural(r, _, i, _) => r != 0 && i != 0,
            CoefficientView::Large(r, i) => !r.is_zero() && !i.is_zero(),
            CoefficientView::Float(r, i) => !r.is_zero() && !i.is_zero(),
            _ => false,
        }
    }

    // der(d1,...,dn,f,x1,...,xn) is displayed as f^(d1,...,dn)(x1,...,xn).
    // Page derivative orders and their corresponding arguments together.
    fn mathml_derivative_head(self) -> Option<(usize, Symbol)> {
        let AtomView::Fun(f) = self else { return None };
        let count = f.get_nargs();
        if f.get_symbol() == Symbol::DERIVATIVE && count >= 3 && count % 2 == 1 {
            f.get(count / 2).get_symbol().map(|s| (count / 2, s))
        } else {
            None
        }
    }

    fn mathml_item_count(self) -> usize {
        self.mathml_derivative_head()
            .map_or_else(|| self.mathml_child_count(), |(n, _)| n)
    }

    /// Render one page, retaining paths to omitted nested expressions. Rendering
    /// and output are bounded; locating a page scans the preceding children.
    pub(crate) fn render_mathml_page(
        self,
        options: &PrintOptions,
        start: usize,
        limit: usize,
    ) -> MathMlPage {
        self.render_mathml_layout(options, start, limit, 0, &[])
    }

    pub(crate) fn render_mathml_layout(
        self,
        options: &PrintOptions,
        start: usize,
        limit: usize,
        layout: usize,
        expansions: &[(Vec<usize>, usize)],
    ) -> MathMlPage {
        self.render_mathml_layout_with_state(
            options,
            start,
            limit,
            layout,
            expansions,
            PrintState::new(),
            None,
            None,
        )
        .unwrap()
    }

    pub(crate) fn render_mathml_layout_cached<'r>(
        self,
        options: &PrintOptions,
        start: usize,
        limit: usize,
        layout: usize,
        expansions: &[(Vec<usize>, usize)],
        cursors: &MathMlCursorCache<'_>,
        semantic: Option<&'r mut SemanticPrinter<'r>>,
    ) -> Result<MathMlPage, ()> {
        self.render_mathml_layout_with_state(
            options,
            start,
            limit,
            layout,
            expansions,
            PrintState::new(),
            Some(cursors),
            semantic,
        )
    }

    fn render_mathml_layout_with_state<'r>(
        self,
        options: &PrintOptions,
        start: usize,
        limit: usize,
        layout: usize,
        expansions: &[(Vec<usize>, usize)],
        state: PrintState,
        cursors: Option<&MathMlCursorCache<'_>>,
        semantic: Option<&'r mut SemanticPrinter<'r>>,
    ) -> Result<MathMlPage, ()> {
        let atom = self;
        let (nested_limit, depth_limit) = LAYOUTS[layout];
        let opts = PrintOptions {
            mode: PrintMode::MathML,
            color_mode: ColorMode::Never,
            color_builtin_symbols: false,
            num_exp_as_superscript: false,
            max_line_length: None,
            max_terms: None,
            ..options.clone()
        };
        // Explicit paging/expansion requests may reveal more detail than the
        // compact initial preview. Scale both budgets together, retaining a
        // fixed ceiling even for many independently expanded subexpressions.
        let requested = expansions
            .iter()
            .fold(limit, |total, (_, count)| total.saturating_add(*count));
        // Scripted tensors and fractions can exceed 16 KiB of markup while
        // occupying only a few lines. Give the initial preview 64 KiB as well
        // as the corresponding node allowance; keep the 256 KiB ceiling.
        let budget_pages = requested
            .div_ceil(DEFAULT_PAGE_SIZE)
            .clamp(MIN_BUDGET_PAGES, 16);
        let mut renderer = Renderer {
            opts: &opts,
            state,
            cursors,
            nodes: MAX_NODES * budget_pages,
            bytes: MAX_TEXT_BYTES * budget_pages,
            budget_exhausted: false,
            budget_holes: 0,
            nested_limit,
            depth_limit: depth_limit.min(MAX_DEPTH),
            expansions,
            multiline_sums: false,
            markers: vec![],
            semantic,
            probing_compact_tail: false,
            compact_probe_remaining: MAX_NODES,
            failed: false,
        };
        let sequence = matches!(atom, AtomView::Add(_) | AtomView::Mul(_) | AtomView::Fun(_));
        let custom = if matches!(atom, AtomView::Fun(_)) {
            if renderer.semantic.is_some() && renderer.hard_limit(atom, 0) {
                Some(renderer.hole(atom, &[]))
            } else {
                renderer
                    .semantic(atom, &[], state)
                    .or_else(|| renderer.custom(atom, &[], state))
            }
        } else {
            None
        };
        let total = if custom.is_none() && sequence {
            atom.mathml_item_count().max(1)
        } else {
            1
        };
        let has_custom = custom.is_some();
        let start = start.min(total - 1);
        let (body, end) = if let Some(fragment) = custom {
            (fragment, 1)
        } else if sequence
            && atom.mathml_child_count() > 0
            && !atom
                .get_symbol()
                .is_some_and(|s| s.get_name().len() > MAX_LEAF_BYTES)
        {
            // Initial root sums need shorter complete prefixes for the
            // browser's height target. Explicit larger pages and expansions
            // retain their requested term counts in every alternative.
            let limit = if matches!(atom, AtomView::Add(_))
                && limit == DEFAULT_PAGE_SIZE
                && expansions.is_empty()
            {
                limit.min(nested_limit.max(4))
            } else {
                limit
            };
            renderer.sequence(atom, &mut vec![], 0, start, limit.max(1))
        } else if matches!(atom, AtomView::Fun(_)) && !renderer.hard_limit(atom, 0) {
            // A zero-argument function has already tried custom dispatch above.
            // Do not call a stateful scoped renderer twice for the same atom.
            (
                renderer.sequence(atom, &mut vec![], 0, 0, limit.max(1)).0,
                1,
            )
        } else {
            (renderer.render(atom, &mut vec![], 0), 1)
        };
        if renderer.failed {
            return Err(());
        }
        Ok(MathMlPage {
            html: format!("{MATH_OPEN}{body}{MATH_CLOSE}"),
            start,
            end,
            total,
            markers: renderer.markers,
            custom: has_custom,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn scoped_page<'r>(
        input: &str,
        layout: usize,
        semantic: &'r mut SemanticPrinter<'r>,
    ) -> Result<MathMlPage, ()> {
        let expression = crate::parse!(input);
        let cursors = MathMlCursorCache::new(expression.as_view());
        expression.as_view().render_mathml_layout_cached(
            &PrintOptions::new(),
            0,
            25,
            layout,
            &[],
            &cursors,
            Some(semantic),
        )
    }

    #[test]
    fn native_polynomial_tails_use_rendered_script_width() {
        let expression = parse!("19380*x*y^3*z^16+q");
        for layout in 0..LAYOUTS.len() {
            let page =
                expression
                    .as_view()
                    .render_mathml_layout(&PrintOptions::new(), 0, 25, layout, &[]);
            assert!(page.markers.is_empty(), "layout {layout}: {}", page.html);
            assert!(page.html.contains("<mn>16</mn>"));
        }
        let long = parse!("f(19380*x*long_variable_y^3*long_variable_z^16+q)");
        let page = long
            .as_view()
            .render_mathml_layout(&PrintOptions::new(), 0, 25, 5, &[]);
        assert!(!page.markers.is_empty());
    }

    #[test]
    fn expanded_sum_keeps_complete_terms_in_every_responsive_layout() {
        let sum = parse!("(x+y+z+w)^20").expand();
        let expression = &sum * parse!("1+a");
        let cursors = MathMlCursorCache::new(expression.as_view());
        let initial =
            expression
                .as_view()
                .render_mathml_layout(&PrintOptions::new(), 0, 25, 1, &[]);
        let omitted = initial
            .markers
            .iter()
            .find(|marker| matches!(cursors.locate(&marker.path), Some(AtomView::Add(_))))
            .unwrap();
        let count = omitted.start + 25;
        for layout in 0..LAYOUTS.len() {
            let page = expression.as_view().render_mathml_layout(
                &PrintOptions::new(),
                0,
                25,
                layout,
                &[(omitted.path.clone(), count)],
            );
            assert!(
                page.markers
                    .iter()
                    .all(|marker| marker.path == omitted.path),
                "layout {layout}: {}",
                page.html
            );
            assert_eq!(page.markers.len(), 1);
            assert_eq!(page.markers[0].start, count);
            assert!(page.html.contains("data-symbolica-sum"));
        }
    }

    #[test]
    fn short_nested_sums_retain_wrappable_terms() {
        for input in [
            "c*(a*b*d*e*f*g+h*i*j*k*l*m)",
            "c*(a*b*d*e*f*g+h*i*j*k*l*m+n*o*p*q*r*s)",
        ] {
            let html = page(input);
            assert!(html.contains("data-symbolica-sum=\"\""), "{html}");
            assert!(html.contains("data-symbolica-term=\"\""), "{html}");
            // Reflow can make these sums tall, so their fences must stretch.
            assert!(!html.contains("stretchy=\"false\""), "{html}");
            assert!(!html.contains("data-symbolica-marker="), "{html}");
        }
    }

    #[test]
    fn scoped_atoms_precede_soft_abbreviation_and_keep_power_groups() {
        let fragment = "<mi>u</mi><mi>v</mi>";
        let target = crate::symbol!("mathml_scoped::target");
        let mut semantic = |atom: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
            Ok((atom.get_symbol() == Some(target)).then(|| fragment.to_owned()))
        };
        let root = scoped_page("mathml_scoped::target(a,b,c,d,e)", 7, &mut semantic).unwrap();
        assert_eq!((root.start, root.end, root.total), (0, 1, 1));
        assert!(root.custom);
        assert!(root.markers.is_empty());
        let nested =
            scoped_page("f(g(mathml_scoped::target(a,b,c,d,e)))", 7, &mut semantic).unwrap();
        assert!(
            nested
                .html
                .contains(&Renderer::tagged(fragment.into(), &[0, 0]))
        );
        assert!(nested.markers.is_empty(), "{}", nested.html);

        let mut states = Vec::new();
        let mut semantic = |atom: AtomView<'_>, _: &PrintOptions, state: &PrintState| {
            states.push((atom.get_symbol().unwrap(), *state));
            Ok(Some(fragment.to_owned()))
        };
        let power = scoped_page(
            "mathml_scoped::base^mathml_scoped::exponent",
            0,
            &mut semantic,
        )
        .unwrap();
        assert!(power.html.contains(&format!(
            "<msup>{}{}</msup>",
            Renderer::tagged(fragment.into(), &[0]),
            Renderer::tagged(fragment.into(), &[1])
        )));
        assert_eq!(states.len(), 2);
        assert!(states[0].1.in_exp_base);
        assert!(states[1].1.in_exp);
    }

    #[test]
    fn compact_tails_use_tensor_notation_and_preserve_callback_order() {
        let metric = crate::symbol!("mathml_compact::metric");
        let fragment = concat!(
            "<mmultiscripts><mi>g</mi><none/>",
            "<msub><mi>ρ</mi><mn>2</mn></msub><none/>",
            "<msub><mi>ν</mi><mn>2</mn></msub></mmultiscripts>"
        );
        for input in [
            "a*b*mathml_compact::metric(mink(4,rho),mink(4,nu)) + c*d*mathml_compact::metric(mink(4,rho),mink(4,nu))",
            "outer(inner(a,b,mathml_compact::metric(mink(4,rho),mink(4,nu))))",
            "outer(a+b+mathml_compact::metric(mink(4,rho),mink(4,nu)))",
        ] {
            let mut full_order = Vec::new();
            let mut full = |atom: AtomView<'_>, _: &PrintOptions, state: &PrintState| {
                full_order.push((atom.get_symbol().unwrap(), state.level, state.in_product));
                Ok((atom.get_symbol() == Some(metric)).then(|| fragment.to_owned()))
            };
            let expected = scoped_page(input, 0, &mut full).unwrap();
            assert!(expected.markers.is_empty());
            let mut compact_order = Vec::new();
            let mut semantic = |atom: AtomView<'_>, _: &PrintOptions, state: &PrintState| {
                compact_order.push((atom.get_symbol().unwrap(), state.level, state.in_product));
                Ok((atom.get_symbol() == Some(metric)).then(|| fragment.to_owned()))
            };
            let actual = scoped_page(input, 5, &mut semantic).unwrap();
            assert!(actual.markers.is_empty(), "{}", actual.html);
            assert!(actual.html.contains(fragment));
            assert_eq!(compact_order, full_order);
        }
    }

    #[test]
    fn compact_semantic_powers_keep_context_and_fraction_parts() {
        let casimir = crate::symbol!("mathml_compact_power::casimir");
        let fragment = "<msub><mi>C</mi><mi>A</mi></msub>";
        for exponent in ["2", "-1"] {
            let input = format!("outer(a*b*mathml_compact_power::casimir(adj(8))^{exponent})");
            let mut states = Vec::new();
            let mut semantic = |atom: AtomView<'_>, _: &PrintOptions, state: &PrintState| {
                if atom.get_symbol() == Some(casimir) {
                    states.push(*state);
                    Ok(Some(fragment.to_owned()))
                } else {
                    Ok(None)
                }
            };
            let actual = scoped_page(&input, 5, &mut semantic).unwrap();
            assert!(actual.markers.is_empty(), "{}", actual.html);
            assert!(actual.html.contains(fragment));
            assert_eq!(states.len(), 1);
            assert!(states[0].in_exp_base);
            if exponent == "-1" {
                assert_eq!(actual.html.matches("<mfrac>").count(), 1);
                assert!(!actual.html.contains("<mn>1</mn>"));
            }
        }
    }

    #[test]
    fn semantic_tail_probes_reject_wide_partial_and_oversized_fragments() {
        let target = crate::symbol!("mathml_compact_reject::target");
        for fragment in [
            "<mi>long_visible_name</mi>".to_owned(),
            format!("<mi>{}</mi>", "x".repeat(MAX_LEAF_BYTES)),
            format!(
                "{}<mi>g</mi>{}",
                "<mrow>".repeat(500),
                "</mrow>".repeat(500)
            ),
        ] {
            let mut semantic = |atom: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
                Ok((atom.get_symbol() == Some(target)).then(|| fragment.clone()))
            };
            let actual = scoped_page(
                "outer(a*b*mathml_compact_reject::target(a,b,c,d,e))",
                5,
                &mut semantic,
            )
            .unwrap();
            assert_eq!(actual.markers.len(), 1);
            assert_eq!(actual.markers[0].start, 2);
            assert_eq!(actual.markers[0].path, [0]);
            assert!(!actual.html.contains(&fragment));
        }

        let mut calls = 0;
        let mut failing = |atom: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
            if atom.get_symbol() == Some(target) {
                calls += 1;
                Err(())
            } else {
                Ok(None)
            }
        };
        assert!(
            scoped_page(
                "outer(a*b*mathml_compact_reject::target(a,b,c,d,e))",
                5,
                &mut failing
            )
            .is_err()
        );
        assert_eq!(calls, 1);
    }

    #[test]
    fn semantic_tail_probes_restore_output_but_bound_speculative_work() {
        let target = crate::symbol!("mathml_compact_budget::target");
        let expression = parse!("f(mathml_compact_budget::target(a,b,c,d,e))");
        let options = PrintOptions::new();
        let mut calls = 0;
        let fragment = "<msup><mi>g</mi><mi>μ</mi></msup>";
        let mut semantic = |atom: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
            assert_eq!(atom.get_symbol(), Some(target));
            calls += 1;
            Ok(Some(fragment.to_owned()))
        };
        let mut renderer = Renderer {
            opts: &options,
            state: PrintState::new(),
            cursors: None,
            nodes: 2,
            bytes: MAX_TEXT_BYTES,
            budget_exhausted: false,
            budget_holes: 0,
            nested_limit: 2,
            depth_limit: MAX_DEPTH,
            expansions: &[],
            multiline_sums: false,
            markers: vec![],
            semantic: Some(&mut semantic),
            probing_compact_tail: false,
            compact_probe_remaining: 2,
            failed: false,
        };
        let before = renderer.checkpoint();
        for _ in 0..3 {
            assert!(
                renderer
                    .compact_tail(expression.as_view(), &mut vec![], 1, 0, PrintState::new(),)
                    .is_empty()
            );
            assert_eq!(renderer.checkpoint(), before);
        }
        assert_eq!(renderer.compact_probe_remaining, 0);
        drop(renderer);
        assert_eq!(calls, 2);

        // If a later sibling is too wide, the earlier accepted fragment must
        // not consume output budget or leave a stray marker behind.
        let expression =
            parse!("f(mathml_compact_budget::target(a),mathml_compact_budget::target(b))");
        let mut calls = 0;
        let mut semantic = |_: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
            calls += 1;
            Ok(Some(
                if calls == 1 {
                    fragment
                } else {
                    "<mi>very_wide</mi>"
                }
                .to_owned(),
            ))
        };
        let mut renderer = Renderer {
            opts: &options,
            state: PrintState::new(),
            cursors: None,
            nodes: MAX_NODES,
            bytes: MAX_TEXT_BYTES,
            budget_exhausted: false,
            budget_holes: 0,
            nested_limit: 2,
            depth_limit: MAX_DEPTH,
            expansions: &[],
            multiline_sums: false,
            markers: vec![],
            semantic: Some(&mut semantic),
            probing_compact_tail: false,
            compact_probe_remaining: MAX_NODES,
            failed: false,
        };
        let before = renderer.checkpoint();
        assert!(
            renderer
                .compact_tail(expression.as_view(), &mut vec![], 1, 0, PrintState::new(),)
                .is_empty()
        );
        assert_eq!(renderer.checkpoint(), before);
        assert_eq!(renderer.compact_probe_remaining, MAX_NODES - 2);
    }

    #[test]
    fn compact_semantic_tails_preserve_explicit_root_page_sizes() {
        let target = crate::symbol!("mathml_compact_root::target");
        let expression = parse!("a*b*mathml_compact_root::target(a,b,c,d,e)");
        let cursors = MathMlCursorCache::new(expression.as_view());
        let mut semantic = |atom: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
            assert_ne!(atom.get_symbol(), Some(target));
            Ok(None)
        };
        for layout in 0..LAYOUTS.len() {
            let page = expression
                .as_view()
                .render_mathml_layout_cached(
                    &PrintOptions::new(),
                    0,
                    1,
                    layout,
                    &[],
                    &cursors,
                    Some(&mut semantic),
                )
                .unwrap();
            assert_eq!((page.start, page.end, page.total), (0, 1, 3));
            assert_eq!(page.markers.len(), 1);
            assert_eq!(page.markers[0].start, 1);
        }
    }

    #[test]
    fn compact_semantic_leaves_use_the_actual_remaining_byte_budget() {
        for fragment in [
            "<mmultiscripts><mi>g</mi><none/><msub><mi>μ</mi><mn>3</mn></msub><none/><msub><mi>ν</mi><mn>3</mn></msub></mmultiscripts>",
            "<mrow><mo>(</mo><msub><mi>k</mi><mn>0</mn></msub><mo>·</mo><msub><mi>k</mi><mn>1</mn></msub><mo>)</mo></mrow>",
        ] {
            let expression = parse!("mathml_compact_bytes::leaf(mink(4,mu),mink(4,nu))");
            let options = PrintOptions::new();
            let mut calls = 0;
            let mut semantic = |_: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
                calls += 1;
                Ok(Some(fragment.to_owned()))
            };
            let mut renderer = Renderer {
                opts: &options,
                state: PrintState::new(),
                cursors: None,
                nodes: 100,
                bytes: 512,
                budget_exhausted: false,
                budget_holes: 0,
                nested_limit: 1,
                depth_limit: 2,
                expansions: &[],
                multiline_sums: false,
                markers: vec![],
                semantic: Some(&mut semantic),
                probing_compact_tail: false,
                compact_probe_remaining: MAX_NODES,
                failed: false,
            };
            let html = renderer.render(expression.as_view(), &mut vec![0, 1], 3);
            assert!(html.contains(fragment), "{html}");
            assert!(renderer.markers.is_empty());
            assert!(!renderer.budget_exhausted);
            assert_eq!(renderer.bytes, 512 - html.len());
            assert_eq!(
                renderer.nodes,
                100 - html.bytes().filter(|c| *c == b'<').count()
            );
            drop(renderer);
            assert_eq!(calls, 1);
        }
    }

    #[test]
    fn semantic_node_exhaustion_keeps_complete_nested_sum_terms() {
        for (name, fragment) in [
            (
                "g",
                "<mmultiscripts><mi>g</mi><none/><msub><mi>μ</mi><mn>3</mn></msub><none/><msub><mi>ν</mi><mn>3</mn></msub></mmultiscripts>",
            ),
            (
                "dot",
                "<mrow><mo>(</mo><msub><mi>k</mi><mn>0</mn></msub><mo>·</mo><msub><mi>k</mi><mn>1</mn></msub><mo>)</mo></mrow>",
            ),
        ] {
            let expression = parse!(&format!(
                "{name}(mink(4,mu),mink(4,nu))+{name}(mink(4,rho),mink(4,sigma))"
            ));
            let leaf_cost = fragment.bytes().filter(|c| *c == b'<').count() + 2;
            let options = PrintOptions::new();
            let mut calls = 0;
            let mut semantic = |atom: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
                assert!(
                    atom.get_symbol()
                        .unwrap()
                        .get_name()
                        .ends_with(&format!("::{name}"))
                );
                calls += 1;
                Ok(Some(fragment.to_owned()))
            };
            let mut renderer = Renderer {
                opts: &options,
                state: PrintState::new(),
                cursors: None,
                nodes: 2 * leaf_cost - 1,
                bytes: MAX_TEXT_BYTES,
                budget_exhausted: false,
                budget_holes: 0,
                nested_limit: 500,
                depth_limit: 2,
                expansions: &[],
                multiline_sums: false,
                markers: vec![],
                semantic: Some(&mut semantic),
                probing_compact_tail: false,
                compact_probe_remaining: MAX_NODES,
                failed: false,
            };
            let (html, end) = renderer.sequence(expression.as_view(), &mut vec![0], 1, 0, 25);
            assert_eq!(end, 1, "{html}");
            assert_eq!(html.matches(fragment).count(), 1, "{html}");
            assert!(!html.contains("&#x2061;"), "{html}");
            assert_eq!(renderer.markers.len(), 1);
            assert_eq!(renderer.markers[0].path, [0]);
            assert_eq!(renderer.markers[0].start, 1);
            assert_eq!(renderer.nodes, leaf_cost - 1);
            assert!(renderer.budget_exhausted);
            drop(renderer);
            assert_eq!(calls, 2);
        }
    }

    #[test]
    fn incomplete_first_nested_sum_term_becomes_one_sum_omission() {
        let expression = parse!("g(mink(4,mu),mink(4,nu))+g(mink(4,rho),mink(4,sigma))");
        let fragment = "<msup><mi>g</mi><mrow><mi>μ</mi><mi>ν</mi></mrow></msup>";
        let tagged = Renderer::tagged(fragment.to_owned(), &[0, 0]);
        let leaf_nodes = tagged.bytes().filter(|c| *c == b'<').count();
        for (nodes, bytes) in [
            (leaf_nodes - 1, MAX_TEXT_BYTES),
            (MAX_NODES, tagged.len() - 1),
        ] {
            let options = PrintOptions::new();
            let mut calls = 0;
            let mut semantic = |_: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
                calls += 1;
                Ok(Some(fragment.to_owned()))
            };
            let mut renderer = Renderer {
                opts: &options,
                state: PrintState::new(),
                cursors: None,
                nodes,
                bytes,
                budget_exhausted: false,
                budget_holes: 0,
                nested_limit: 500,
                depth_limit: 32,
                expansions: &[],
                multiline_sums: true,
                markers: vec![],
                semantic: Some(&mut semantic),
                probing_compact_tail: false,
                compact_probe_remaining: MAX_NODES,
                failed: false,
            };
            let (html, end) = renderer.sequence(expression.as_view(), &mut vec![0], 1, 0, 25);
            assert_eq!(end, 0, "{html}");
            assert!(!html.contains("&#x2061;"), "{html}");
            assert_eq!(renderer.markers.len(), 1);
            assert_eq!(renderer.markers[0].path, [0]);
            assert_eq!(renderer.markers[0].start, 0);
            assert_eq!((renderer.nodes, renderer.bytes), (nodes, bytes));
            assert!(renderer.budget_exhausted);
            drop(renderer);
            assert_eq!(calls, 1);
        }
    }

    #[test]
    fn exhausted_atomic_product_factor_moves_to_the_product_tail() {
        let fragment = "<msup><mi>g</mi><mrow><mi>μ</mi><mi>ν</mi></mrow></msup>";
        let tagged = Renderer::tagged(fragment.to_owned(), &[0]);
        let leaf_nodes = tagged.bytes().filter(|c| *c == b'<').count();
        for power in [false, true] {
            let input = (0..7)
                .map(|i| format!("g(k{i}){}", if power { "^2" } else { "" }))
                .collect::<Vec<_>>()
                .join("*");
            let expression = parse!(&input);
            let cost = leaf_nodes + if power { 2 } else { 0 };
            let byte_limit = MAX_LEAF_BYTES + if power { tagged.len() + 16 } else { 1 };
            for (nodes, bytes) in [(2 * cost - 1, MAX_TEXT_BYTES), (MAX_NODES, byte_limit)] {
                let options = PrintOptions::new();
                let mut calls = 0;
                let mut semantic = |atom: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
                    assert!(matches!(atom, AtomView::Fun(_)));
                    calls += 1;
                    Ok(Some(fragment.to_owned()))
                };
                let mut renderer = Renderer {
                    opts: &options,
                    state: PrintState::new(),
                    cursors: None,
                    nodes,
                    bytes,
                    budget_exhausted: false,
                    budget_holes: 0,
                    nested_limit: 500,
                    depth_limit: 32,
                    expansions: &[],
                    multiline_sums: false,
                    markers: vec![],
                    semantic: Some(&mut semantic),
                    probing_compact_tail: false,
                    compact_probe_remaining: MAX_NODES,
                    failed: false,
                };
                let (html, end) = renderer.product(expression.as_view(), &mut vec![], 0, 0, 25);
                assert_eq!(end, 1, "{html}");
                assert_eq!(html.matches(fragment).count(), 1, "{html}");
                assert!(!html.contains("&#x2061;"), "{html}");
                assert_eq!(renderer.markers.len(), 1);
                assert!(renderer.markers[0].path.is_empty());
                assert_eq!(renderer.markers[0].start, 1);
                assert_eq!(renderer.nodes, nodes - cost);
                assert!(renderer.bytes <= bytes);
                assert!(renderer.budget_exhausted);
                drop(renderer);
                assert_eq!(
                    calls,
                    if bytes == MAX_TEXT_BYTES || power {
                        2
                    } else {
                        1
                    }
                );
            }
        }
    }

    #[test]
    fn native_function_keeps_complete_arguments_before_a_budget_refusal() {
        for count in [200, 400] {
            let arguments = (0..count)
                .map(|i| format!("g(k{i})"))
                .collect::<Vec<_>>()
                .join(",");
            let expression = parse!(&format!("a*f({arguments})"));
            let metric = crate::symbol!("g");
            let fragment = "<msup><mi>g</mi><mrow><mi>μ</mi><mi>ν</mi></mrow></msup>";
            let cursors = MathMlCursorCache::new(expression.as_view());
            let mut calls = 0;
            let mut semantic = |atom: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
                Ok((atom.get_symbol() == Some(metric)).then(|| {
                    calls += 1;
                    fragment.to_owned()
                }))
            };
            let page = expression
                .as_view()
                .render_mathml_layout_cached(
                    &PrintOptions::new(),
                    0,
                    25,
                    0,
                    &[],
                    &cursors,
                    Some(&mut semantic),
                )
                .unwrap();
            assert_eq!((page.end, page.total), (2, 2));
            assert_eq!(page.html.matches("&#x2061;").count(), 1, "{}", page.html);
            assert!(page.html.matches(fragment).count() > 1);
            if count == 200 {
                // Moderate scripted output fits the initial preview, although its
                // markup exceeded the old allowance.
                assert!(page.markers.is_empty());
                assert_eq!(page.html.matches(fragment).count(), count);
            } else {
                assert_eq!(page.markers.len(), 1);
                assert_eq!(page.markers[0].path, [1]);
                assert!(page.markers[0].start > 0);
            }
            assert!(calls <= count);
        }
    }

    #[test]
    fn compound_function_and_power_factors_keep_useful_sum_previews() {
        let sum = (0..1600)
            .map(|i| format!("x{i}^2"))
            .collect::<Vec<_>>()
            .join("+");
        for input in [
            format!("a*f({sum})"),
            format!("a*({sum})^2"),
            format!("a/({sum})"),
        ] {
            let expression = parse!(&input);
            let page = expression
                .as_view()
                .render_mathml_page(&PrintOptions::new(), 0, 25);
            assert_eq!((page.end, page.total), (2, 2), "{}", page.html);
            assert!(page.html.contains("<mi>x"), "{}", page.html);
            assert!(page.markers.iter().any(|m| m.path == [1, 0] && m.start > 0));
            assert!(page.markers.iter().all(|m| !m.path.is_empty()));
        }
    }

    #[test]
    fn semantic_node_exhaustion_does_not_leave_raw_heads_in_polynomials() {
        let terms = (0..60)
            .map(|i| format!("dot(p,k{i})^2*dot(k{i},k{i})^2*dot(p,p)"))
            .collect::<Vec<_>>()
            .join("+");
        let expression = parse!(&format!("a*({terms})"));
        let dot = crate::symbol!("dot");
        let fragment = "<mrow><mo>(</mo><msub><mi>k</mi><mn>0</mn></msub><mo>·</mo><msub><mi>k</mi><mn>1</mn></msub><mo>)</mo></mrow>";
        let cursors = MathMlCursorCache::new(expression.as_view());
        let mut calls = 0;
        let mut semantic = |atom: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
            calls += 1;
            Ok((atom.get_symbol() == Some(dot)).then(|| fragment.to_owned()))
        };
        let page = expression
            .as_view()
            .render_mathml_layout_cached(
                &PrintOptions::new(),
                0,
                25,
                0,
                &[],
                &cursors,
                Some(&mut semantic),
            )
            .unwrap();
        assert!(page.html.contains(fragment));
        assert!(!page.html.contains("&#x2061;"), "{}", page.html);
        assert_eq!(page.markers.len(), 1);
        assert_eq!(page.markers[0].path, [1]);
        assert!(page.markers[0].start > 0);
        assert_eq!((page.end, page.total), (2, 2));
        assert!(calls < MAX_NODES);
    }

    #[test]
    fn scoped_errors_stop_dispatch_and_hard_guards_precede_callbacks() {
        let mut calls = 0;
        let mut fail = |_: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
            calls += 1;
            Err(())
        };
        assert!(scoped_page("f(a,b,c)", 0, &mut fail).is_err());
        assert_eq!(calls, 1);

        let long_name = "x".repeat(MAX_LEAF_BYTES + 1);
        let mut forbidden =
            |_: AtomView<'_>, _: &PrintOptions, _: &PrintState| -> Result<Option<String>, ()> {
                panic!("hard limit must run before scoped dispatch")
            };
        assert!(
            !scoped_page(&long_name, 0, &mut forbidden)
                .unwrap()
                .markers
                .is_empty()
        );

        let input = format!("{}x{}", "f(".repeat(64), ")".repeat(64));
        let expression = crate::parse!(&input);
        let cursors = MathMlCursorCache::new(expression.as_view());
        let mut semantic = |_: AtomView<'_>, _: &PrintOptions, state: &PrintState| {
            assert!(state.level < 64);
            Ok(None)
        };
        let page = expression
            .as_view()
            .render_mathml_layout_cached(
                &PrintOptions::new(),
                0,
                25,
                0,
                &[(vec![0; 64], 1)],
                &cursors,
                Some(&mut semantic),
            )
            .unwrap();
        assert!(page.markers.iter().any(|marker| marker.path.len() == 64));
    }

    #[test]
    fn declining_scoped_printer_preserves_global_callback_abbreviation() {
        let _ = crate::symbol!(
            "mathml_scoped_decline::target",
            print = |_, _, _| { Some("<mi>global</mi>".into()) }
        );
        let input = "f(g(mathml_scoped_decline::target(a,b,c,d,e)))";
        let expression = crate::parse!(input);
        for layout in 0..LAYOUTS.len() {
            let expected =
                expression
                    .as_view()
                    .render_mathml_layout(&PrintOptions::new(), 0, 25, layout, &[]);
            let mut decline = |_: AtomView<'_>, _: &PrintOptions, _: &PrintState| Ok(None);
            let actual = scoped_page(input, layout, &mut decline).unwrap();
            assert_eq!(actual.html, expected.html);
            assert_eq!(
                (actual.start, actual.end, actual.total),
                (expected.start, expected.end, expected.total)
            );
            assert_eq!(actual.markers.len(), expected.markers.len());
            for (actual, expected) in actual.markers.iter().zip(&expected.markers) {
                assert_eq!(
                    (&actual.path, actual.start),
                    (&expected.path, expected.start)
                );
            }
        }
    }

    #[test]
    fn scoped_budget_rejection_restores_native_function_paging() {
        for fragment in [format!("<mi>{}</mi>", "μ".repeat(2040)), "<m/>".repeat(999)] {
            let mut semantic = |atom: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
                Ok(matches!(atom, AtomView::Fun(_)).then(|| fragment.clone()))
            };
            let page = scoped_page("f(a,b,c)", 0, &mut semantic).unwrap();
            assert!(!page.custom);
            assert_eq!((page.start, page.end, page.total), (0, 3, 3));
            assert!(page.markers.is_empty(), "{}", page.html);
            assert!(page.html.contains("<mi>a</mi>"));
            assert!(!page.html.contains("μ"));
        }
    }

    #[test]
    fn custom_output_accounts_for_utf8_elements_and_its_wrapper() {
        let options = PrintOptions::new();
        let atom = crate::parse!("x");
        let mut renderer = Renderer {
            opts: &options,
            state: PrintState::new(),
            cursors: None,
            nodes: MAX_NODES,
            bytes: MAX_TEXT_BYTES,
            budget_exhausted: false,
            budget_holes: 0,
            nested_limit: 500,
            depth_limit: MAX_DEPTH,
            expansions: &[],
            multiline_sums: false,
            markers: vec![],
            semantic: None,
            probing_compact_tail: false,
            compact_probe_remaining: MAX_NODES,
            failed: false,
        };
        let before = renderer.checkpoint();
        let utf8 = format!("<mi>{}</mi>", "μ".repeat(2040));
        assert!(utf8.len() <= MAX_LEAF_BYTES);
        assert!(
            renderer
                .custom_fragment(atom.as_view(), &[], utf8, true)
                .is_none()
        );
        assert_eq!(renderer.checkpoint(), before);
        assert!(
            renderer
                .custom_fragment(atom.as_view(), &[], "<m/>".repeat(999), true)
                .is_none()
        );
        assert_eq!(renderer.checkpoint(), before);
        let raw = format!("<mi>{}</mi>", "x".repeat(4000));
        assert!(
            renderer
                .custom_fragment(atom.as_view(), &vec![usize::MAX; 64], raw.clone(), true)
                .is_none()
        );
        assert_eq!(renderer.checkpoint(), before);
        let accepted = renderer
            .custom_fragment(atom.as_view(), &[1, 2], raw, true)
            .unwrap();
        assert_eq!(renderer.bytes, MAX_TEXT_BYTES - accepted.len());
        assert_eq!(renderer.nodes, MAX_NODES - 4);
        assert!(!renderer.budget_exhausted);
    }

    fn page(input: &str) -> String {
        crate::parse!(input)
            .as_view()
            .render_mathml_page(&PrintOptions::new(), 0, 25)
            .html
    }

    #[test]
    fn large_sum_after_small_factor_keeps_its_preview() {
        let sum = parse!("(x+y+z+w)^20").expand();
        let expression = &sum * parse!("1+x");
        let before = expression.clone();
        let cursors = MathMlCursorCache::new(expression.as_view());
        let page = expression
            .as_view()
            .render_mathml_page(&PrintOptions::new(), 0, 25);

        assert_eq!((page.end, page.total), (2, 2));
        assert_eq!(page.html.matches("data-symbolica-factor").count(), 2);
        for name in ["x", "y", "z", "w"] {
            assert!(page.html.contains(&format!("<mi>{name}</mi>")));
        }
        assert!(page.markers.iter().all(|marker| !marker.path.is_empty()));
        let omission = page
            .markers
            .iter()
            .find(|marker| marker.path == [1])
            .expect("the omitted terms belong to the large sum, not the product");
        assert!(omission.start > 0);
        let target = cursors.locate(&omission.path).unwrap();
        assert_eq!(target, sum.as_view());
        let next = target.render_mathml_page(&PrintOptions::new(), omission.start, 25);
        assert_eq!(next.start, omission.start);
        assert_eq!(next.end, omission.start + 25);
        assert_eq!(next.total, 1771);
        assert_eq!(expression, before);
    }

    #[test]
    fn compact_root_factor_after_large_sum_keeps_its_display_budget() {
        let casimir = crate::symbol!("mathml_root_reserve::casimir");
        // Short powers exhaust nodes; long names exhaust the text budget.
        for prefix in ["x".to_owned(), "long".repeat(20)] {
            let sum = (0..1600)
                .map(|i| format!("{prefix}{i}^2"))
                .collect::<Vec<_>>()
                .join("+");
            let expression = parse!(&format!(
                "a*({sum})*mathml_root_reserve::casimir(adjoint)^4"
            ));
            let cursors = MathMlCursorCache::new(expression.as_view());
            let mut calls = 0;
            let mut semantic = |atom: AtomView<'_>, _: &PrintOptions, state: &PrintState| {
                if atom.get_symbol() == Some(casimir) {
                    calls += 1;
                    assert!(state.in_exp_base);
                    Ok(Some("<msub><mi>C</mi><mi>A</mi></msub>".to_owned()))
                } else {
                    Ok(None)
                }
            };
            let page = expression
                .as_view()
                .render_mathml_layout_cached(
                    &PrintOptions::new(),
                    0,
                    25,
                    0,
                    &[],
                    &cursors,
                    Some(&mut semantic),
                )
                .unwrap();
            assert_eq!((page.end, page.total), (3, 3));
            assert!(page.html.contains("<msub><mi>C</mi><mi>A</mi></msub>"));
            assert_eq!(calls, 1);
            assert!(page.markers.iter().all(|marker| !marker.path.is_empty()));
            assert!(page.markers.iter().any(|marker| marker.path == [1]));
        }
    }

    #[test]
    fn root_reserve_preserves_requested_pages_and_following_factors() {
        let casimir = crate::symbol!("mathml_root_reserve_limits::casimir");
        let sum = (0..1600)
            .map(|i| format!("x{i}^2"))
            .collect::<Vec<_>>()
            .join("+");
        let expression = parse!(&format!(
            "a*({sum})*mathml_root_reserve_limits::casimir(adjoint)^4"
        ));
        let cursors = MathMlCursorCache::new(expression.as_view());
        for (start, size, fragment, expected_end, expected_calls) in [
            (0, 2, "<msub><mi>C</mi><mi>A</mi></msub>", 2, 0),
            (1, 2, "<msub><mi>C</mi><mi>A</mi></msub>", 3, 1),
            (0, 25, "<mi>long_visible_casimir_name</mi>", 3, 1),
        ] {
            let mut calls = 0;
            let mut semantic = |atom: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
                if atom.get_symbol() == Some(casimir) {
                    calls += 1;
                    Ok(Some(fragment.to_owned()))
                } else {
                    Ok(None)
                }
            };
            let page = expression
                .as_view()
                .render_mathml_layout_cached(
                    &PrintOptions::new(),
                    start,
                    size,
                    0,
                    &[],
                    &cursors,
                    Some(&mut semantic),
                )
                .unwrap();
            assert_eq!((page.start, page.end, page.total), (start, expected_end, 3));
            assert_eq!(calls, expected_calls);
            if expected_end == 2 {
                assert!(
                    page.markers
                        .iter()
                        .any(|marker| marker.path.is_empty() && marker.start == 2)
                );
                assert!(!page.html.contains(fragment));
            } else {
                assert!(page.html.contains(fragment));
            }
        }
    }

    #[test]
    fn later_factor_previews_respect_budgets_and_keep_remaining_factors_navigable() {
        // Powers exhaust the node budget; long identifiers exhaust the text budget.
        for prefix in ["x".to_owned(), "long".repeat(20)] {
            let sum = (0..1600)
                .map(|i| format!("{prefix}{i}^2"))
                .collect::<Vec<_>>()
                .join("+");
            for (input, fraction) in [
                (format!("(a+b)*({sum})*({sum}+1)"), false),
                (format!("-2*(a+b)/({sum})"), true),
            ] {
                let expression = parse!(&input);
                let cursors = MathMlCursorCache::new(expression.as_view());
                let mut nodes = 0;
                let mut bytes = 0;
                let mut observe = |atom: AtomView<'_>, opts: &PrintOptions, _: &PrintState| {
                    nodes += 1;
                    bytes += symbol_html(atom.get_symbol().unwrap(), opts).len();
                    Ok(None)
                };
                let page = expression
                    .as_view()
                    .render_mathml_layout_cached(
                        &PrintOptions::new(),
                        0,
                        25,
                        0,
                        &[],
                        &cursors,
                        Some(&mut observe),
                    )
                    .unwrap();
                assert!(nodes <= MAX_NODES * MIN_BUDGET_PAGES);
                assert!(bytes <= MAX_TEXT_BYTES * MIN_BUDGET_PAGES);
                assert_eq!(page.total, 3);
                assert_eq!(page.end, 3);
                assert!(page.html.contains(&format!("<mi>{prefix}")));
                assert_eq!(page.html.contains("<mfrac"), fraction);
                if fraction {
                    assert!(page.html.contains("<mo>−</mo>"));
                }
                let path = if fraction { vec![2, 0] } else { vec![1] };
                assert!(page.markers.iter().any(|m| m.path == path && m.start > 0));
                for marker in &page.markers {
                    let target = cursors.locate(&marker.path).unwrap();
                    let next = target.render_mathml_page(&PrintOptions::new(), marker.start, 25);
                    assert!(next.end > next.start);
                    assert!(next.html.contains("<mi>") || next.html.contains("<mn>"));
                    assert!(
                        !next
                            .markers
                            .iter()
                            .any(|m| m.path.is_empty() && m.start == 0)
                    );
                }
            }
        }
    }

    #[test]
    fn oversized_first_root_item_keeps_bounded_preview() {
        // Short names exhaust the node budget; long names exhaust the text budget.
        for prefix in ["x".to_owned(), "long".repeat(20)] {
            let sum = (0..1600)
                .map(|i| format!("{prefix}{i}"))
                .collect::<Vec<_>>()
                .join("+");
            let product = format!("({sum})*({sum}+a)*({sum}+b)");
            for input in [
                format!("{product}+c*({product})"),
                format!("1/(({product}+c)*({product}+d))"),
            ] {
                let expression = parse!(&input);
                let cursors = MathMlCursorCache::new(expression.as_view());
                for start in 0..2 {
                    let mut nodes = 0;
                    let mut bytes = 0;
                    let mut observe = |atom: AtomView<'_>, opts: &PrintOptions, _: &PrintState| {
                        nodes += 1;
                        bytes += symbol_html(atom.get_symbol().unwrap(), opts).len();
                        Ok(None)
                    };
                    let page = expression
                        .as_view()
                        .render_mathml_layout_cached(
                            &PrintOptions::new(),
                            start,
                            25,
                            0,
                            &[],
                            &cursors,
                            Some(&mut observe),
                        )
                        .unwrap();
                    assert!(nodes <= MAX_NODES * MIN_BUDGET_PAGES);
                    assert!(bytes <= MAX_TEXT_BYTES * MIN_BUDGET_PAGES);
                    assert!(page.html.contains("<mi>"), "{}", page.html);
                    assert!(page.is_partial());
                    assert_eq!((page.start, page.total), (start, 2));
                    assert_eq!(page.end, if prefix == "x" { 2 } else { start + 1 });
                    assert!(page.markers.iter().any(|marker| marker.path.len() > 1));
                    for marker in &page.markers {
                        let target = cursors.locate(&marker.path).unwrap();
                        assert!(!marker.path.is_empty() || marker.start > start);
                        let opened =
                            target.render_mathml_page(&PrintOptions::new(), marker.start, 25);
                        assert!(opened.end > opened.start);
                        assert!(opened.html.contains("<mi>"));
                        assert!(
                            !opened
                                .markers
                                .iter()
                                .any(|m| m.path.is_empty() && m.start == 0)
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn larger_pages_reveal_factors_past_the_compact_render_budget() {
        let factors = (0..34)
            .map(|i| {
                let arguments = (0..60)
                    .map(|j| format!("x{j}"))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("f{i}({arguments})")
            })
            .collect::<Vec<_>>()
            .join("*");
        let expression = crate::parse!(&factors);
        let atom = expression.as_view();
        let compact = atom.render_mathml_page(&PrintOptions::new(), 0, 25);
        assert!(compact.end < compact.total);
        let expanded = atom.render_mathml_page(&PrintOptions::new(), 0, 50);
        assert!(expanded.end > compact.end);
        let complete = atom.render_mathml_page(&PrintOptions::new(), 0, 100);
        assert_eq!(complete.end, complete.total);
        assert_eq!(complete.total, 34);
        assert!(complete.html.len() < 256 * 1024);
    }

    #[test]
    fn few_large_factors_keep_reflow_boundaries() {
        let left = (0..12)
            .map(|i| format!("x{i}"))
            .collect::<Vec<_>>()
            .join("+");
        let right = (0..12)
            .map(|i| format!("y{i}"))
            .collect::<Vec<_>>()
            .join("+");
        let expression = crate::parse!(&format!("({left})*({right})"));
        let before = expression.clone();
        let html = expression
            .as_view()
            .render_mathml_page(&PrintOptions::new(), 0, 25);
        assert!(html.markers.is_empty(), "{}", html.html);
        assert!(
            html.html.contains("<mtable data-symbolica-product"),
            "{}",
            html.html
        );
        assert_eq!(html.html.matches("data-symbolica-factor").count(), 2);
        assert_eq!(html.html.matches("data-symbolica-sum").count(), 2);
        assert_eq!(html.html.matches("data-symbolica-term").count(), 24);
        assert_eq!(expression, before);
        // Small products use the same scalar spacing as their inline form.
        let short = crate::parse!("x*y")
            .as_view()
            .render_mathml_page(&PrintOptions::new().multiplication_operator('·'), 0, 25)
            .html;
        assert_eq!(short.matches("data-symbolica-factor").count(), 2);
        assert!(short.contains("<mo>&#x2062;</mo>"), "{short}");
        assert!(!short.contains("<mo>·</mo>"), "{short}");
    }

    #[test]
    fn product_tables_have_only_deliberate_internal_row_spacing() {
        for input in ["-2*sqrt(5)", "-sqrt(5)", "x*y", "x/(a*b)"] {
            let html = page(input);
            assert!(html.contains("data-symbolica-product"), "{input}: {html}");
            assert!(
                html.contains("rowspacing=\"0\"><mtr><mtd style=\"padding:0\">"),
                "{input}: {html}"
            );
            assert!(!html.contains("<mtd>"), "{input}: {html}");
            assert!(!html.contains("padding:0.5em"), "{input}: {html}");
        }
        let product = (0..40)
            .map(|i| format!("x{i}"))
            .collect::<Vec<_>>()
            .join("*");
        let html = page(&product);
        assert!(html.contains("<mtd style=\"padding:0.5em 0 0\">"), "{html}");
        assert_eq!(
            html.matches("<mtd style=\"padding:0\">").count(),
            1,
            "{html}"
        );
        assert!(!html.contains("<mtd>"), "{html}");
    }

    #[test]
    fn multiline_product_signs_stay_in_the_first_factor() {
        let product = (0..40)
            .map(|i| format!("x{i}"))
            .collect::<Vec<_>>()
            .join("*");
        for input in [
            "-2*x*y".to_owned(),
            "-sqrt(5)".to_owned(),
            format!("-1i/16*({product})"),
        ] {
            let expression = parse!(&input);
            let before = expression.clone();
            let page = expression
                .as_view()
                .render_mathml_page(&PrintOptions::new(), 0, 25);
            let first = page.html.find("<mrow data-symbolica-factor=").unwrap();
            let content = first + page.html[first..].find('>').unwrap() + 1;
            assert!(
                page.html[content..].starts_with("<mo>−</mo>"),
                "{}",
                page.html
            );
            assert_eq!(page.html.matches("<mo>−</mo>").count(), 1, "{}", page.html);
            assert!(
                Renderer::leading_minus(expression.as_view(), &page.fragment()).is_some(),
                "{}",
                page.html
            );
            assert_eq!(expression, before);
        }
        // An explicit one-item page can contain just the negative unit and
        // the omitted suffix. Its sign must still belong to a packed factor.
        let tail = parse!("-x")
            .as_view()
            .render_mathml_page(&PrintOptions::new(), 0, 1);
        assert_eq!(tail.end, 1);
        assert_eq!(tail.markers.len(), 1);
        assert!(
            tail.html
                .contains("<mrow data-symbolica-factor=\"\"><mo>−</mo><mn>1</mn></mrow>")
        );
    }

    #[test]
    fn expanded_negative_products_keep_their_sum_sign_in_the_first_factor() {
        let expression = parse!("a-2*b*c");
        let product = expression
            .as_view()
            .children()
            .position(|term| matches!(term, AtomView::Mul(_)))
            .unwrap();
        let page = expression.as_view().render_mathml_layout(
            &PrintOptions::new(),
            0,
            25,
            0,
            &[(vec![product], 25)],
        );
        let first = page.html.find("<mrow data-symbolica-factor=").unwrap();
        let content = first + page.html[first..].find('>').unwrap() + 1;
        assert!(
            page.html[content..].starts_with("<mo form=\"infix\">−</mo>"),
            "{}",
            page.html
        );
        assert_eq!(page.html.matches("<mo form=\"infix\">−</mo>").count(), 1);
        assert!(
            !page.html.contains("<mo form=\"infix\">+</mo>"),
            "{}",
            page.html
        );
    }

    #[test]
    fn product_sign_detection_does_not_enter_custom_factor_contents() {
        let target = crate::symbol!("mathml_product_sign::target");
        let mut semantic = |atom: AtomView<'_>, _: &PrintOptions, _: &PrintState| {
            Ok((atom.get_symbol() == Some(target))
                .then(|| "<mo>−</mo><mi>a</mi><mo>+</mo><mi>b</mi>".to_owned()))
        };
        let expression = parse!("mathml_product_sign::target(x)*y");
        let page = scoped_page("mathml_product_sign::target(x)*y", 0, &mut semantic).unwrap();
        assert!(page.html.contains("<mo>−</mo>"));
        assert!(Renderer::leading_minus(expression.as_view(), &page.fragment()).is_none());
    }

    #[test]
    fn complex_components_keep_their_own_signs() {
        for (input, real, sign, imag) in [
            ("-2-3i", "<mo>−</mo><mn>2</mn>", "−", "3"),
            ("-2+3i", "<mo>−</mo><mn>2</mn>", "+", "3"),
            ("2-3i", "<mn>2</mn>", "−", "3"),
            ("2+3i", "<mn>2</mn>", "+", "3"),
        ] {
            let html = page(input);
            assert!(
                html.contains(&format!("<mrow>{real}</mrow><mo>{sign}</mo>")),
                "{html}"
            );
            assert!(html.contains(&format!("<mn>{imag}</mn>")), "{html}");
            assert!(html.contains("<mi>𝑖</mi>"), "{html}");
            assert!(!html.contains("<mtext>"), "{html}");
        }
        let pure_imaginary = page("-81i/64");
        assert!(pure_imaginary.contains("<mo>−</mo>"), "{pure_imaginary}");
        assert!(
            pure_imaginary.contains("<mfrac><mn>81</mn><mn>64</mn></mfrac>"),
            "{pure_imaginary}"
        );
        assert!(pure_imaginary.contains("<mi>𝑖</mi>"), "{pure_imaginary}");
    }

    #[test]
    fn visible_cost_ignores_indentation_but_preserves_token_spaces() {
        let compact = "<msubsup><mi>k</mi><mn>0</mn><mi>μ</mi></msubsup>";
        let pretty = "<msubsup>\n    <mi>k</mi>\n    <mn>0</mn>\n    <mi>μ</mi>\n</msubsup>\n";
        assert_eq!(visible_cost(compact, PRODUCT_ROW_COST), 3);
        assert_eq!(visible_cost(pretty, PRODUCT_ROW_COST), 3);
        assert_eq!(visible_cost("<mtext>a b</mtext>", PRODUCT_ROW_COST), 3);
        assert_eq!(visible_cost("<mtext>a\u{a0}b</mtext>", PRODUCT_ROW_COST), 3);
        assert_eq!(
            visible_cost(
                "<mrow>\n<mi>x</mi>\n<mo>&#x2062;</mo>\n</mrow>",
                PRODUCT_ROW_COST
            ),
            2
        );
    }

    #[test]
    fn visible_cost_separates_markup_and_stops_at_the_requested_budget() {
        let token = format!("<mi data-label=\"{}\">μ</mi>", "x".repeat(16_384));
        assert_eq!(visible_cost(&token, 768), 1);
        assert_eq!(visible_cost(&token.repeat(50), 768), 50);
        assert_eq!(visible_cost(&"<mi>μ</mi>".repeat(1000), 768), 769);
        assert_eq!(
            visible_cost(&"<mi>μ</mi>".repeat(100), PRODUCT_ROW_COST),
            49
        );
    }

    #[test]
    fn pretty_custom_leaves_pack_the_same_product_rows() {
        let _ = crate::symbol!(
            "mathml_row_cost::compact",
            print = |_, options, _| {
                options
                    .mode
                    .is_mathml()
                    .then(|| "<msubsup><mi>k</mi><mn>0</mn><mi>μ</mi></msubsup>".into())
            }
        );
        let _ = crate::symbol!(
            "mathml_row_cost::pretty",
            print = |_, options, _| {
                options.mode.is_mathml().then(|| {
                    "<msubsup>\n    <mi>k</mi>\n    <mn>0</mn>\n    <mi>μ</mi>\n</msubsup>\n".into()
                })
            }
        );
        for head in ["mathml_row_cost::compact", "mathml_row_cost::pretty"] {
            let product = (0..10)
                .map(|i| format!("{head}({i})"))
                .collect::<Vec<_>>()
                .join("*");
            let html = page(&format!("({product})/8"));
            // Ten short factors share one numerator row; the denominator has
            // its own row. Pretty XML must not turn this into eleven rows.
            assert_eq!(html.matches("<mtr>").count(), 2, "{html}");
            assert!(!html.contains("data-symbolica-marker"), "{html}");
        }
    }

    #[test]
    fn multiline_fraction_tables_are_grouped_for_webkit() {
        let product = (0..10)
            .map(|i| format!("x{i}"))
            .collect::<Vec<_>>()
            .join("*");
        let html = page(&format!("({product})/8"));
        assert!(
            html.contains("<mfrac data-symbolica-product-fraction=\"\"><mrow><mtable"),
            "{html}"
        );
        assert!(html.contains("</mtable></mrow><mrow><mtable"), "{html}");
        assert!(html.contains("</mtable></mrow></mfrac>"), "{html}");
    }

    #[test]
    fn product_fractions_keep_canonical_layout_and_mark_numeric_coefficients() {
        for input in ["-3*x*y/8", "-3*x*y/z", "-x*y/8", "(x+5)*(x+6)/x"] {
            let html = page(input);
            assert!(
                html.contains("<mfrac data-symbolica-product-fraction=\"\"><mrow><mtable"),
                "{input}: {html}"
            );
            if input.starts_with('-') {
                assert!(
                    html.contains("<mo>−</mo><mfrac data-symbolica-product-fraction="),
                    "{html}"
                );
            }
            assert_eq!(html.matches("<mfrac").count(), 1, "{input}: {html}");
            assert_eq!(
                html.matches("data-symbolica-coefficient").count(),
                usize::from(input.contains("-3")),
                "{input}: {html}"
            );
        }
        // An imaginary or mixed complex factor must not be treated as a real
        // numeric numerator when the viewer forms a compact coefficient.
        for input in ["2i*x/y", "(2+3i)*x/y"] {
            assert!(!page(input).contains("data-symbolica-coefficient"));
        }
        // Inline products in sums keep their normal fraction; an explicitly
        // expanded product gets responsive rows and the opt-in marker.
        let expression = crate::parse!("a-b*c/8");
        let product = expression
            .as_view()
            .children()
            .position(|term| matches!(term, AtomView::Mul(_)))
            .unwrap();
        let inline = page("a-b*c/8");
        assert!(!inline.contains("data-symbolica-product-fraction"));
        let expanded = expression.as_view().render_mathml_layout(
            &PrintOptions::new(),
            0,
            25,
            0,
            &[(vec![product], 25)],
        );
        assert!(expanded.html.contains("data-symbolica-product-fraction"));
        assert!(expanded.html.contains("<mo form=\"infix\">−</mo>"));
    }

    #[test]
    fn product_omissions_are_packed_factors_inside_fraction_numerators() {
        let product = (0..40)
            .map(|i| format!("x{i}"))
            .collect::<Vec<_>>()
            .join("*");
        let tail = "<mrow data-symbolica-factor=\"\"><mo form=\"infix\">×</mo><mspace width=\"0.25em\"/><mtext data-symbolica-marker=\"0\">[1]</mtext></mrow>";
        for (input, fraction, negative) in [
            (product.clone(), false, false),
            (format!("-3*({product})/8"), true, true),
            (format!("-3*({product})/12345678901"), true, true),
            (format!("1/({product})"), true, false),
        ] {
            let expression = parse!(&input);
            let before = expression.clone();
            let page = expression
                .as_view()
                .render_mathml_page(&PrintOptions::new(), 0, 25);
            assert_eq!(page.end, 25);
            assert_eq!(page.markers.len(), 1);
            assert!(page.markers[0].path.is_empty());
            assert_eq!(page.markers[0].start, 25);
            assert_eq!(page.html.matches(tail).count(), 1, "{}", page.html);
            assert!(!page.html.contains("<mo form=\"infix\">·</mo>"));
            if fraction {
                // The omitted product multiplies the numerator. Leaving it
                // after the fraction places it beside the center of a wrapped
                // numerator instead of after the final visible factor.
                let numerator_end = page
                    .html
                    .find("</mtable></mrow><mrow><mtable")
                    .expect("canonical numerator and denominator tables");
                assert!(
                    page.html.find(tail).unwrap() < numerator_end,
                    "{}",
                    page.html
                );
                assert_eq!(page.html.matches("<mfrac").count(), 1);
                if !negative {
                    assert!(
                        page.html
                            .contains("<mrow data-symbolica-factor=\"\"><mn>1</mn></mrow>"),
                        "{}",
                        page.html
                    );
                }
            }
            assert_eq!(
                page.html.matches("<mo>−</mo>").count(),
                usize::from(negative)
            );
            let rest = expression
                .as_view()
                .render_mathml_page(&PrintOptions::new(), page.end, 25);
            assert_eq!(rest.start, 25);
            assert_eq!(rest.end, rest.total);
            assert!(rest.markers.is_empty());
            assert!(rest.html.contains("<mtext>⋯</mtext>"));
            assert_eq!(expression, before);
        }
    }

    #[test]
    fn inline_product_omissions_use_a_single_cross_after_the_fraction() {
        let product = (0..40)
            .map(|i| format!("x{i}"))
            .collect::<Vec<_>>()
            .join("*");
        let expression = parse!(&format!("outer(-3*({product})/8)"));
        let page = expression
            .as_view()
            .render_mathml_layout(&PrintOptions::new(), 0, 25, 5, &[]);
        assert!(!page.html.contains("data-symbolica-product"));
        assert_eq!(page.markers.len(), 1);
        assert_eq!(page.markers[0].path, [0]);
        assert_eq!(page.markers[0].start, 2);
        assert!(
            page.html
                .contains("</mfrac><mrow><mo form=\"infix\">×</mo><mspace width=\"0.25em\"/>"),
            "{}",
            page.html
        );
        assert_eq!(page.html.matches("<mo form=\"infix\">×</mo>").count(), 1);
        assert_eq!(page.html.matches("<mo>−</mo>").count(), 1);
    }

    #[test]
    fn fraction_bars_group_single_sums_without_parentheses() {
        for input in ["x/(1+z^2)", "(a+b)/(c+d)", "1/(a+b)"] {
            let html = page(input);
            assert!(html.contains("<mfrac"), "{input}: {html}");
            assert!(!html.contains(">(</mo>"), "{input}: {html}");
        }
    }

    #[test]
    fn products_and_power_bases_keep_required_parentheses() {
        for (input, pairs) in [
            ("x/((a+b)*(c+d))", 2),
            ("x/((a+b)*z)", 1),
            ("(a+b)*x/(c+d)", 1),
            ("(a+b)*(c+d)", 2),
            ("(a+b)^2/(c+d)", 1),
            ("(a+b)/(c+d)^2", 1),
        ] {
            let html = page(input);
            assert_eq!(html.matches(">(</mo>").count(), pairs, "{input}: {html}");
            assert_eq!(html.matches(">)</mo>").count(), pairs, "{input}: {html}");
        }
        let partial =
            crate::parse!("(a+b)*(c+d)")
                .as_view()
                .render_mathml_page(&PrintOptions::new(), 0, 1);
        assert_eq!(
            partial.html.matches(">(</mo>").count(),
            1,
            "{}",
            partial.html
        );
        assert!(!partial.markers.is_empty());
    }

    #[test]
    fn function_delimiters_use_natural_glyphs_for_flat_arguments() {
        for input in ["sin(alpha)^2", "f(x+y,2)"] {
            let html = page(input);
            assert!(
                html.contains("<mo stretchy=\"false\">(</mo>"),
                "{input}: {html}"
            );
            assert!(
                html.contains("<mo stretchy=\"false\">)</mo>"),
                "{input}: {html}"
            );
        }
        for input in ["f(1/(1+x^2))", "f(sqrt(x))", "f(x^2)"] {
            let html = page(input);
            assert!(html.contains("<mo>(</mo>"), "{input}: {html}");
            assert!(html.contains("<mo>)</mo>"), "{input}: {html}");
        }
    }
}

/// A namespace qualifies the identifier, rather than becoming part of its glyph.
/// Keep it upright and subordinate, while retaining the complete name on hover.
pub(crate) fn symbol_html(symbol: Symbol, opts: &PrintOptions) -> String {
    if symbol.get_name().len() > MAX_LEAF_BYTES {
        return Renderer::token("mtext", "⋯");
    }
    let mut plain = opts.clone();
    plain.mode = PrintMode::Symbolica;
    plain.color_mode = ColorMode::Never;
    let mut text = String::new();
    symbol
        .format_with_custom(&plain, PrintState::new(), false, &mut text)
        .unwrap();
    let name = symbol.get_stripped_name();
    let prefix = text
        .strip_suffix(name)
        .unwrap_or_default()
        .trim_end_matches("::");
    let identifier = Renderer::token("mi", name);
    if prefix.is_empty() {
        identifier
    } else {
        format!(
            "<msub title=\"{}\">{identifier}<mtext style=\"font-size:.85em\">{}</mtext></msub>",
            AnsiHtmlFormatter::escape_html(symbol.get_name()),
            AnsiHtmlFormatter::escape_html(prefix)
        )
    }
}

pub(crate) fn document(fragment: &str) -> String {
    let open = MATH_OPEN
        .replace("<math ", "<math display=\"block\" ")
        .replace(
            "font-synthesis:none",
            "font-synthesis:none;display:block;text-align:left",
        );
    format!("{open}{fragment}{MATH_CLOSE}")
}
