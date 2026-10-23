//! Structural formatting for algebraic containers. Coefficients are delegated
//! to their own ring, so symbols, signs and coefficient domains stay intact.
use super::*;
use crate::{
    domains::{Ring, SelfRing, rational::Q, rational_polynomial::RationalPolynomial},
    poly::{
        Exponent, MonomialOrder, PolyVariable, PositiveExponent,
        polynomial::MultivariatePolynomial, series::Series,
    },
};

fn variable(v: &PolyVariable, opts: &PrintOptions) -> String {
    match v {
        PolyVariable::Symbol(s) => symbol_html(*s, opts),
        PolyVariable::Temporary(i) => format!("<msub><mi>T</mi><mn>{i}</mn></msub>"),
        PolyVariable::Function(_, a) | PolyVariable::Power(a) => {
            let mut html = String::new();
            a.as_view()
                .format_mathml(&mut html, opts, PrintState::new())
                .unwrap();
            // A polynomial variable can itself be a non-polynomial power.
            if matches!(v, PolyVariable::Power(_)) {
                Renderer::fence(html)
            } else {
                html
            }
        }
    }
}

fn finish<W: std::fmt::Write>(
    body: String,
    terms: usize,
    state: PrintState,
    f: &mut W,
) -> Result<bool, std::fmt::Error> {
    let body = if terms > 1 && (state.in_product || state.in_exp || state.in_exp_base) {
        Renderer::fence(body)
    } else if state.level == 0 && terms > 1 {
        format!(
            "<mrow data-symbolica-flow=\"\" style=\"display:flex;flex-wrap:wrap;align-items:baseline;max-width:var(--symbolica-wrap-width,100%);row-gap:.45em\">{body}</mrow>"
        )
    } else {
        Renderer::row(body)
    };
    f.write_str(&body)?;
    Ok(false)
}

fn domain<R: Ring>(ring: &R, opts: &PrintOptions, f: &mut String) {
    if opts.print_ring && !ring.has_independent_elements() {
        let plain = PrintOptions {
            mode: PrintMode::Symbolica,
            color_mode: ColorMode::Never,
            ..opts.clone()
        };
        let mut text = String::new();
        ring.format_ring(&plain, PrintState::new(), &mut text)
            .unwrap();
        f.push_str("<mspace width=\".5em\"/>");
        f.push_str(&Renderer::token("mtext", &text));
    }
}

pub(crate) fn polynomial<R: Ring, E: Exponent, O: MonomialOrder, W: std::fmt::Write>(
    poly: &MultivariatePolynomial<R, E, O>,
    opts: &PrintOptions,
    state: PrintState,
    f: &mut W,
) -> Result<bool, std::fmt::Error> {
    polynomial_range(
        poly,
        opts,
        state,
        f,
        0,
        opts.max_terms.unwrap_or(100).clamp(1, 500),
        false,
    )
    .map(|(suppressed, _)| suppressed)
}

pub(crate) fn polynomial_range<R: Ring, E: Exponent, O: MonomialOrder, W: std::fmt::Write>(
    poly: &MultivariatePolynomial<R, E, O>,
    opts: &PrintOptions,
    state: PrintState,
    f: &mut W,
    start: usize,
    limit: usize,
    live: bool,
) -> Result<(bool, usize), std::fmt::Error> {
    let print_ring = opts.print_ring && !poly.ring().has_independent_elements();
    if poly.is_constant() && !poly.is_zero() && (!print_ring || state.level > 0) {
        return poly
            .ring()
            .format_mathml(&poly.get_constant(), opts, state, f)
            .map(|suppressed| (suppressed, 1));
    }
    let mut body = String::new();
    let mut end = start;
    for (i, term) in poly.into_iter().enumerate().skip(start).take(limit) {
        let has_var = term.exponents.iter().any(|e| !e.is_zero());
        let mut html = String::new();
        let mut suppressed = poly.ring().format_mathml(
            term.coefficient,
            opts,
            PrintState {
                in_sum: i > 0 || (state.in_sum && !(poly.nterms() > 1 && state.in_product)),
                in_product: has_var,
                suppress_one: has_var,
                level: state.level + 1,
                ..PrintState::new()
            },
            &mut html,
        )?;
        for (v, exponent) in poly.variables().iter().zip(term.exponents) {
            if exponent.is_zero() {
                continue;
            }
            if !suppressed {
                html.push_str("<mo>\u{2062}</mo><mspace width=\"thinmathspace\"/>");
            }
            suppressed = false;
            if html.len() > MAX_TEXT_BYTES {
                html.push_str("<mtext>⋯</mtext>");
                break;
            }
            let v = variable(v, opts);
            if exponent.to_i32() == 1 {
                html.push_str(&v);
            } else {
                html.push_str(&format!(
                    "<msup>{v}{}</msup>",
                    Renderer::token("mn", &exponent.to_string())
                ));
            }
        }
        body.push_str(&format!("<mrow>{html}</mrow>"));
        end = i + 1;
        if body.len() > MAX_TEXT_BYTES {
            break;
        }
    }
    if poly.is_zero() {
        if state.in_sum {
            body.push_str("<mo form=\"infix\">+</mo>");
        }
        body.push_str("<mn>0</mn>");
    } else if end < poly.nterms() {
        body.push_str(&omission(live));
    }
    if state.level == 0 {
        domain(poly.ring(), opts, &mut body);
    }
    // A sum used as a coefficient needs one sign outside its fences.
    if state.in_sum && state.in_product && poly.nterms() > 1 {
        f.write_str("<mo form=\"infix\">+</mo>")?;
    }
    finish(body, poly.nterms(), state, f).map(|suppressed| (suppressed, end.max(1)))
}

pub(crate) fn rational_polynomial<R: Ring, E: PositiveExponent, W: std::fmt::Write>(
    poly: &RationalPolynomial<R, E>,
    opts: &PrintOptions,
    state: PrintState,
    f: &mut W,
) -> Result<bool, std::fmt::Error> {
    if poly.denominator.is_one() {
        return polynomial(&poly.numerator, opts, state, f);
    }
    if state.in_sum {
        f.write_str("<mo form=\"infix\">+</mo>")?;
    }
    let inner = PrintOptions {
        print_ring: false,
        ..opts.clone()
    };
    let num = poly.numerator.format_string(
        &inner,
        PrintState {
            level: state.level + 1,
            ..PrintState::new()
        },
    );
    let den = poly.denominator.format_string(
        &inner,
        PrintState {
            level: state.level + 1,
            ..PrintState::new()
        },
    );
    let mut body = format!("<mfrac>{num}{den}</mfrac>");
    if state.level == 0 {
        domain(poly.numerator.ring(), opts, &mut body);
    }
    finish(body, 1, state, f)
}

pub(crate) fn series<R: Ring, W: std::fmt::Write>(
    series: &Series<R>,
    opts: &PrintOptions,
    state: PrintState,
    f: &mut W,
) -> Result<bool, std::fmt::Error> {
    series_range(
        series,
        opts,
        state,
        f,
        0,
        opts.max_terms.unwrap_or(100).clamp(1, 500),
        false,
    )
    .map(|(suppressed, _)| suppressed)
}

pub(crate) fn series_range<R: Ring, W: std::fmt::Write>(
    series: &Series<R>,
    opts: &PrintOptions,
    state: PrintState,
    f: &mut W,
    start: usize,
    limit: usize,
    live: bool,
) -> Result<(bool, usize), std::fmt::Error> {
    let ring = series.get_field();
    let mut base = variable(series.get_vars_ref(), opts);
    let point = series.get_expansion_point();
    if !ring.is_zero(&point) {
        let mut offset = String::new();
        ring.format_mathml(
            &ring.neg(&point),
            opts,
            PrintState {
                in_sum: true,
                ..PrintState::new()
            },
            &mut offset,
        )?;
        base = Renderer::fence(format!("{base}{offset}"));
    }
    let power = |exponent: &crate::domains::rational::Rational| {
        if exponent.is_one() {
            return base.clone();
        }
        let mut exp = String::new();
        Q.format_mathml(exponent, opts, PrintState::new(), &mut exp)
            .unwrap();
        format!("<msup>{base}<mrow>{exp}</mrow></msup>")
    };
    let mut body = String::new();
    let mut terms = 0;
    let mut omitted = false;
    for (exponent, c) in series
        .terms_for_display()
        .filter(|(_, c)| !ring.is_zero(c))
        .skip(start)
    {
        if terms >= limit || body.len() > MAX_TEXT_BYTES {
            omitted = true;
            break;
        }
        let mut html = String::new();
        let suppressed = ring.format_mathml(
            c,
            opts,
            PrintState {
                in_sum: terms > 0 || start > 0,
                in_product: !exponent.is_zero(),
                suppress_one: !exponent.is_zero(),
                level: state.level + 1,
                ..PrintState::new()
            },
            &mut html,
        )?;
        if !exponent.is_zero() {
            if !suppressed {
                html.push_str("<mo>\u{2062}</mo><mspace width=\"thinmathspace\"/>");
            }
            html.push_str(&power(&exponent));
        }
        body.push_str(&Renderer::row(html));
        terms += 1;
    }
    if state.in_sum {
        f.write_str("<mo form=\"infix\">+</mo>")?;
    }
    if omitted {
        body.push_str(&omission(live));
    }
    let separator = if terms > 0 {
        "<mo form=\"infix\">+</mo>"
    } else {
        ""
    };
    body.push_str(&Renderer::row(format!(
        "{separator}<mi>𝒪</mi><mo>\u{2061}</mo>{}",
        Renderer::fence(power(&series.absolute_order()))
    )));
    finish(body, terms + 1, state, f).map(|suppressed| (suppressed, (start + terms).max(1)))
}

pub(crate) fn atom_coefficient<W: std::fmt::Write>(
    atom: AtomView<'_>,
    opts: &PrintOptions,
    state: PrintState,
    f: &mut W,
) -> Result<bool, std::fmt::Error> {
    if state.suppress_one && atom.is_one() {
        if state.in_sum {
            f.write_str("<mo form=\"infix\">+</mo>")?;
        }
        return Ok(true);
    }
    if state.suppress_one && atom == Atom::num(-1).as_view() {
        f.write_str(if state.in_sum {
            "<mo form=\"infix\">−</mo>"
        } else {
            "<mo>−</mo>"
        })?;
        return Ok(true);
    }
    let mut body = String::new();
    atom.format_mathml(&mut body, opts, PrintState::new())?;
    let atom = atom.mathml_shown_view(opts);
    let fence =
        state.in_product && (matches!(atom, AtomView::Add(_)) || atom.mathml_complex_coefficient());
    if state.in_sum && (fence || Renderer::leading_minus(atom, &body).is_none()) {
        f.write_str("<mo form=\"infix\">+</mo>")?;
    }
    if state.in_sum
        && !fence
        && let Some(offset) = Renderer::leading_minus(atom, &body)
    {
        body.replace_range(
            offset..offset + "<mo>−</mo>".len(),
            "<mo form=\"infix\">−</mo>",
        );
    }
    if fence {
        body = Renderer::fence(body);
    }
    f.write_str(&body)?;
    Ok(false)
}

fn omission(live: bool) -> String {
    if live {
        "<mrow><mo form=\"infix\">+</mo><mtext data-symbolica-marker=\"0\">[1]</mtext></mrow>"
            .to_owned()
    } else {
        "<mrow><mo form=\"infix\">+</mo><mtext>⋯</mtext></mrow>".to_owned()
    }
}

/// Owned algebraic sources page their native storage, without converting the
/// full object to an expression or serializing hidden terms.
pub(crate) trait AlgebraicMathMl: SelfRing {
    fn page(
        &self,
        opts: &PrintOptions,
        path: &[usize],
        start: usize,
        size: usize,
    ) -> Option<MathMlPage>;
    fn page_expanded(
        &self,
        opts: &PrintOptions,
        path: &[usize],
        start: usize,
        size: usize,
        _expansions: &[(Vec<usize>, usize)],
    ) -> Option<MathMlPage> {
        self.page(opts, path, start, size)
    }
    fn kind(&self) -> &'static str;
}

fn sequence_page(body: String, start: usize, end: usize, total: usize) -> MathMlPage {
    MathMlPage {
        html: format!("{MATH_OPEN}{body}{MATH_CLOSE}"),
        start,
        end,
        total,
        custom: false,
        markers: if end < total {
            vec![MathMlMarker {
                path: vec![],
                label: "Inspect omitted terms [1]".into(),
                start: end,
            }]
        } else {
            vec![]
        },
    }
}

impl<R: Ring, E: Exponent, O: MonomialOrder> AlgebraicMathMl for MultivariatePolynomial<R, E, O> {
    fn kind(&self) -> &'static str {
        "Polynomial"
    }
    fn page(
        &self,
        opts: &PrintOptions,
        path: &[usize],
        start: usize,
        size: usize,
    ) -> Option<MathMlPage> {
        if !path.is_empty() {
            return None;
        }
        let total = self.nterms().max(1);
        let start = start.min(total - 1);
        let mut body = String::new();
        let (_, end) =
            polynomial_range(self, opts, PrintState::new(), &mut body, start, size, true).ok()?;
        Some(sequence_page(body, start, end, total))
    }
}

impl<R: Ring> AlgebraicMathMl for Series<R> {
    fn kind(&self) -> &'static str {
        "Series"
    }
    fn page(
        &self,
        opts: &PrintOptions,
        path: &[usize],
        start: usize,
        size: usize,
    ) -> Option<MathMlPage> {
        if !path.is_empty() {
            return None;
        }
        let total = self
            .terms_for_display()
            .filter(|(_, c)| !self.get_field().is_zero(c))
            .count()
            .max(1);
        let start = start.min(total - 1);
        let mut body = String::new();
        let (_, end) =
            series_range(self, opts, PrintState::new(), &mut body, start, size, true).ok()?;
        Some(sequence_page(body, start, end, total))
    }
}

impl<R: Ring, E: PositiveExponent> AlgebraicMathMl for RationalPolynomial<R, E> {
    fn kind(&self) -> &'static str {
        "Rational polynomial"
    }
    fn page(
        &self,
        opts: &PrintOptions,
        path: &[usize],
        start: usize,
        size: usize,
    ) -> Option<MathMlPage> {
        self.page_expanded(opts, path, start, size, &[])
    }
    fn page_expanded(
        &self,
        opts: &PrintOptions,
        path: &[usize],
        start: usize,
        size: usize,
        expansions: &[(Vec<usize>, usize)],
    ) -> Option<MathMlPage> {
        if let Some((part, rest)) = path.split_first() {
            return match part {
                0 => self.numerator.page(opts, rest, start, size),
                1 => self.denominator.page(opts, rest, start, size),
                _ => None,
            };
        }
        if self.denominator.is_one() {
            return self.numerator.page(opts, path, start, size);
        }
        let inner = PrintOptions {
            print_ring: false,
            ..opts.clone()
        };
        let limit = |part| {
            expansions
                .iter()
                .filter(|(path, _)| path.as_slice() == [part])
                .map(|(_, count)| *count)
                .max()
                .unwrap_or(size)
                .max(size)
                .min(500)
        };
        let num = self.numerator.page(&inner, &[], 0, limit(0))?;
        let den = self.denominator.page(&inner, &[], 0, limit(1))?;
        let mut markers = vec![];
        let mut parts = vec![];
        for (part, page) in [num, den].into_iter().enumerate() {
            let mut body =
                page.html[MATH_OPEN.len()..page.html.len() - MATH_CLOSE.len()].to_owned();
            if !page.markers.is_empty() {
                let id = markers.len();
                body = body
                    .replace(
                        "data-symbolica-marker=\"0\"",
                        &format!("data-symbolica-marker=\"{id}\""),
                    )
                    .replace(">[1]</mtext>", &format!(">[{}]</mtext>", id + 1));
                markers.push(MathMlMarker {
                    path: vec![part],
                    label: format!(
                        "Inspect {} [{}]",
                        if part == 0 {
                            "numerator"
                        } else {
                            "denominator"
                        },
                        id + 1
                    ),
                    start: page.end,
                });
            }
            parts.push(body);
        }
        let mut body = format!("<mfrac>{}{}</mfrac>", parts[0], parts[1]);
        domain(self.numerator.ring(), opts, &mut body);
        Some(MathMlPage {
            html: format!("{MATH_OPEN}{body}{MATH_CLOSE}"),
            start: 0,
            end: 1,
            total: 1,
            markers,
            custom: false,
        })
    }
}
