//! Bounded native MathML presentation. Omitted subtrees are display metadata,
//! never substitutions in the underlying expression.
use super::*;
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
    nested_limit: usize,
    depth_limit: usize,
    expansions: &'a [(Vec<usize>, usize)],
    multiline_sums: bool,
    markers: Vec<MathMlMarker>,
    semantic: Option<&'r mut SemanticPrinter<'r>>,
    failed: bool,
}

struct ProductFactor {
    html: String,
    needs_fence: bool,
}

struct FractionParts {
    numerator: String,
    denominator: ProductFactor,
}

impl Renderer<'_, '_> {
    fn row(body: String) -> String {
        format!("<mrow>{body}</mrow>")
    }
    fn fence(body: String) -> String {
        // Natural-size parentheses avoid the extra advance reserved for
        // stretchy operators in browsers. Keep stretching for tall content.
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
        .any(|tag| body.contains(tag));
        let size = if tall { "" } else { " stretchy=\"false\"" };
        Self::row(format!("<mo{size}>(</mo>{body}<mo{size}>)</mo>"))
    }
    fn token(tag: &str, text: &str) -> String {
        format!("<{tag}>{}</{tag}>", AnsiHtmlFormatter::escape_html(text))
    }

    // A bounded estimate for grouping already-rendered factors into native MathML
    // rows. Ignore XML indentation, but retain whitespace inside text tokens.
    fn visible_cost(fragment: &str) -> usize {
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
            if cost > PRODUCT_ROW_COST {
                break;
            }
        }
        cost
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

    fn checkpoint(&self) -> (usize, usize, usize, bool) {
        (
            self.nodes,
            self.bytes,
            self.markers.len(),
            self.budget_exhausted,
        )
    }

    fn restore(&mut self, state: (usize, usize, usize, bool)) {
        self.nodes = state.0;
        self.bytes = state.1;
        self.markers.truncate(state.2);
        self.budget_exhausted = state.3;
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
            let cost = name.chars().take(budget + 1).count().max(1);
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
            AtomView::Mul(_) => "·",
            _ => {
                if atom.mathml_child_count() > 0 {
                    label
                } else {
                    "⋯"
                }
            }
        };
        format!(
            "<mrow><mo form=\"infix\">{marker_type}</mo><mtext data-symbolica-marker=\"{index}\">[{}]</mtext></mrow>",
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
        if fragment.len() > MAX_LEAF_BYTES || fragment.len() > self.bytes || nodes > self.nodes {
            // An opaque scoped fragment that cannot fit should not hide the
            // native arguments behind a root marker that opens itself forever.
            if semantic {
                return None;
            }
            self.budget_exhausted = true;
            return Some(self.hole(atom, path));
        }
        self.bytes -= fragment.len();
        self.nodes -= nodes;
        Some(fragment)
    }

    fn hard_limit(&self, atom: AtomView<'_>, depth: usize) -> bool {
        depth >= 64
            || self.nodes == 0
            || self.bytes < MAX_LEAF_BYTES
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
        // Explicit expansion reveals a little more depth along this branch.
        // Both the absolute depth and the existing node/text budgets still apply.
        let depth_limit = self
            .expansions
            .iter()
            .filter(|(p, _)| p.starts_with(path) || path.starts_with(p))
            .fold(self.depth_limit, |limit, (p, _)| {
                limit.max((p.len() + 4).min(64))
            });
        if self.hard_limit(atom, depth) {
            self.budget_exhausted |= self.nodes == 0 || self.bytes < MAX_LEAF_BYTES;
            return self.hole(atom, path);
        }
        if let Some(custom) = self.semantic(atom, path, state) {
            return custom;
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
        for (i, child) in MathMlChildren::new(atom, start, self.cursors).take(limit) {
            if self.failed {
                break;
            }
            // Stop a page before the budget runs out. Nested lists become one
            // navigable omission instead of emitting an unbounded set of markers.
            if end > start && (self.nodes == 0 || self.bytes < MAX_LEAF_BYTES) {
                self.budget_exhausted = true;
                break;
            }
            let checkpoint = self.checkpoint();
            path.push(i);
            let mut fragment = self.render(child, path, depth + 1);
            // The first item already contains bounded, navigable omissions.
            // Keep that preview; later items can start on a fresh page budget.
            if depth == 0 && self.budget_exhausted && end > start {
                self.restore(checkpoint);
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
                body.push_str("<mrow style=\"flex:0 0 auto\">");
            }
            if i > start {
                if matches!(atom, AtomView::Add(_))
                    && let Some(offset) = Self::leading_minus(child, &fragment)
                {
                    body.push_str(if wrap || expanded_sum {
                        "<mo form=\"infix\">−</mo>"
                    } else {
                        "<mo>−</mo>"
                    });
                    fragment.replace_range(offset..offset + "<mo>−</mo>".len(), "");
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
        let columns = self
            .nested_limit
            .clamp(2, if self.multiline_sums { 6 } else { 8 });
        if expanded_sum && expanded_terms.len() > columns {
            body = "<mtable data-symbolica-sum=\"\" columnalign=\"left\" rowspacing=\"0.35em\">"
                .to_owned();
            for terms in expanded_terms.chunks(columns) {
                body.push_str(&format!(
                    "<mtr><mtd>{}</mtd></mtr>",
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
        for (i, child) in MathMlChildren::new(atom, start, self.cursors).take(limit) {
            if end > start && (self.nodes == 0 || self.bytes < MAX_LEAF_BYTES) {
                self.budget_exhausted = true;
                break;
            }
            path.push(i);
            let mut parts = None;
            let previous_multiline_sums = self.multiline_sums;
            self.multiline_sums |= multiline;
            let mut html = self.render_factor(
                child,
                path,
                depth + 1,
                &mut parts,
                PrintState {
                    in_product: true,
                    suppress_one: true,
                    ..self.state_at(depth + 1)
                },
            );
            self.multiline_sums = previous_multiline_sums;
            path.pop();
            if let Some(parts) = parts {
                let n = Self::product_sign(child, parts.numerator, &mut negative);
                if n != "<mn>1</mn>" && n != "<mrow><mn>1</mn></mrow>" {
                    numerator.push(ProductFactor {
                        html: n,
                        needs_fence: false,
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
                    });
                }
            }
            end = i + 1;
            // A factor may have used the remaining budget to show part of a
            // large sum. Keep that bounded preview, fraction parts and markers;
            // only the following factors need a fresh page budget.
            if depth == 0 && self.budget_exhausted {
                break;
            }
        }
        self.nested_limit = previous_nested_limit;
        let mut omission = (end < atom.mathml_child_count()).then(|| self.hole_at(atom, path, end));
        let fraction = !denominator.is_empty();
        let join = |factors: Vec<ProductFactor>, tail: Option<String>| {
            let mut result = String::new();
            let mut rows = Vec::new();
            let mut row_cost = 0;
            let mut previous_numeric = false;
            // A fraction bar already groups its numerator and denominator.
            // Fence a sum only when it is multiplied by another factor.
            let multiple = !fraction || factors.len() > 1 || tail.is_some();
            for (i, factor) in factors.into_iter().enumerate() {
                let factor = if multiple && factor.needs_fence {
                    Self::fence(factor.html)
                } else {
                    factor.html
                };
                if multiline {
                    let cost = Self::visible_cost(&factor) + usize::from(i > 0);
                    if !result.is_empty() && row_cost + cost > PRODUCT_ROW_COST {
                        rows.push(std::mem::take(&mut result));
                        row_cost = 0;
                    }
                    row_cost += cost;
                    result.push_str("<mrow data-symbolica-factor=\"\">");
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
                result.push_str(&factor);
                if multiline {
                    result.push_str("</mrow>");
                }
                previous_numeric = numeric;
            }
            if result.is_empty() {
                result.push_str("<mn>1</mn>");
            }
            if multiline {
                rows.push(result);
                if let Some(tail) = tail {
                    rows.push(tail);
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
            if multiline && denominator.is_empty() {
                omission.take()
            } else {
                None
            },
        );
        let mut body = if denominator.is_empty() {
            n
        } else {
            let denominator = join(denominator, None);
            if multiline {
                // WebKit mislays fractions with direct table arguments. Group
                // each table without changing the numerator or denominator.
                format!("<mfrac>{}{}</mfrac>", Self::row(n), Self::row(denominator))
            } else {
                format!("<mfrac>{n}{denominator}</mfrac>")
            }
        };
        if negative {
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
        let budget_pages = requested.div_ceil(DEFAULT_PAGE_SIZE).clamp(1, 16);
        let mut renderer = Renderer {
            opts: &opts,
            state,
            cursors,
            nodes: MAX_NODES * budget_pages,
            bytes: MAX_TEXT_BYTES * budget_pages,
            budget_exhausted: false,
            nested_limit,
            depth_limit: depth_limit.min(MAX_DEPTH),
            expansions,
            multiline_sums: false,
            markers: vec![],
            semantic,
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
            nested_limit: 500,
            depth_limit: MAX_DEPTH,
            expansions: &[],
            multiline_sums: false,
            markers: vec![],
            semantic: None,
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
    fn later_factor_previews_respect_budgets_and_keep_remaining_factors_navigable() {
        // Powers exhaust the node budget; long identifiers exhaust the text budget.
        for prefix in ["x".to_owned(), "long".repeat(20)] {
            let sum = (0..400)
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
                assert!(nodes <= MAX_NODES);
                assert!(bytes <= MAX_TEXT_BYTES);
                assert_eq!(page.total, 3);
                assert_eq!(page.end, if fraction { 3 } else { 2 });
                assert!(page.html.contains(&format!("<mi>{prefix}")));
                assert_eq!(page.html.contains("<mfrac>"), fraction);
                if fraction {
                    assert!(page.html.contains("<mo>−</mo>"));
                } else {
                    assert!(
                        page.markers
                            .iter()
                            .any(|m| m.path.is_empty() && m.start == 2)
                    );
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
            let sum = (0..400)
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
                    assert!(nodes <= MAX_NODES);
                    assert!(bytes <= MAX_TEXT_BYTES);
                    assert!(page.html.contains("<mi>"), "{}", page.html);
                    assert!(page.is_partial());
                    assert_eq!((page.start, page.end, page.total), (start, start + 1, 2));
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
        assert_eq!(Renderer::visible_cost(compact), 3);
        assert_eq!(Renderer::visible_cost(pretty), 3);
        assert_eq!(Renderer::visible_cost("<mtext>a b</mtext>"), 3);
        assert_eq!(Renderer::visible_cost("<mtext>a\u{a0}b</mtext>"), 3);
        assert_eq!(
            Renderer::visible_cost("<mrow>\n<mi>x</mi>\n<mo>&#x2062;</mo>\n</mrow>"),
            2
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
        assert!(html.contains("<mfrac><mrow><mtable"), "{html}");
        assert!(html.contains("</mtable></mrow><mrow><mtable"), "{html}");
        assert!(html.contains("</mtable></mrow></mfrac>"), "{html}");
    }

    #[test]
    fn fraction_bars_group_single_sums_without_parentheses() {
        for input in ["x/(1+z^2)", "(a+b)/(c+d)", "1/(a+b)"] {
            let html = page(input);
            assert!(html.contains("<mfrac>"), "{input}: {html}");
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
