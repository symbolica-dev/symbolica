//! Conventional notation for built-in functions, such as `Γ(z)`, `ψ⁽ⁿ⁾(z)`,
//! `Liₛ(z)`, `Jₙ(z)`, `|z|` and `z*`, shared by the MathML, LaTeX and Typst
//! printers. Other symbols keep their names.
use std::{collections::HashMap, sync::LazyLock};

use crate::{atom::Symbol, transcendental};

/// The name of a function in each output format.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Head {
    pub mathml: &'static str,
    pub latex: &'static str,
    pub typst: &'static str,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Notation {
    /// A conventional function name, such as `Γ` for `gamma`.
    Head(Head),
    /// `head⁽ⁿ⁾(z)` for `f(n, z)`, written `head(z)` when `n = 0`.
    DerivativeOrder(Head),
    /// `headₛ(z)` for `f(s, z)`.
    Indexed(Head),
    /// `|z|`.
    Abs,
    /// `z*`.
    Conjugate,
}

impl Notation {
    /// The function name used when an argument count has no special form.
    pub(crate) fn head(self) -> Option<Head> {
        match self {
            Self::Head(head) | Self::DerivativeOrder(head) | Self::Indexed(head) => Some(head),
            Self::Abs | Self::Conjugate => None,
        }
    }
}

const fn head(mathml: &'static str, latex: &'static str, typst: &'static str) -> Head {
    Head {
        mathml,
        latex,
        typst,
    }
}

/// A multi-letter, upright function name.
macro_rules! operator {
    ($name: literal) => {
        head(
            concat!("<mi>", $name, "</mi>"),
            concat!("\\operatorname{", $name, "}"),
            concat!("op(\"", $name, "\")"),
        )
    };
    ($name: literal, $latex: literal) => {
        head(
            concat!("<mi>", $name, "</mi>"),
            $latex,
            concat!("op(\"", $name, "\")"),
        )
    };
}

static NOTATION: LazyLock<HashMap<Symbol, Notation>> = LazyLock::new(|| {
    use Notation::*;
    use transcendental as t;
    // Uppercase Greek and multi-letter names are upright; single Latin and
    // lowercase Greek letters keep the default italic of an identifier.
    let mut table = HashMap::from([
        (
            t::gamma(),
            Head(head(
                "<mi mathvariant=\"normal\">Γ</mi>",
                "\\Gamma",
                "Gamma",
            )),
        ),
        (t::zeta(), Head(head("<mi>ζ</mi>", "\\zeta", "zeta"))),
        (
            t::polygamma(),
            DerivativeOrder(head("<mi>ψ</mi>", "\\psi", "psi")),
        ),
        (t::polylog(), Indexed(operator!("Li"))),
        (t::bessel_j(), Indexed(head("<mi>J</mi>", "J", "J"))),
        (t::bessel_y(), Indexed(head("<mi>Y</mi>", "Y", "Y"))),
        (t::bessel_i(), Indexed(head("<mi>I</mi>", "I", "I"))),
        (t::bessel_k(), Indexed(head("<mi>K</mi>", "K", "K"))),
        (t::erf(), Head(operator!("erf"))),
        (t::tan(), Head(operator!("tan", "\\tan"))),
        (t::cot(), Head(operator!("cot", "\\cot"))),
        (t::sec(), Head(operator!("sec", "\\sec"))),
        (t::csc(), Head(operator!("csc", "\\csc"))),
        (t::sinh(), Head(operator!("sinh", "\\sinh"))),
        (t::cosh(), Head(operator!("cosh", "\\cosh"))),
        (t::tanh(), Head(operator!("tanh", "\\tanh"))),
        (t::coth(), Head(operator!("coth", "\\coth"))),
        (t::sech(), Head(operator!("sech"))),
        (t::csch(), Head(operator!("csch"))),
        // Inverse functions use the ISO 80000-2 names.
        (t::asin(), Head(operator!("arcsin", "\\arcsin"))),
        (t::acos(), Head(operator!("arccos", "\\arccos"))),
        (t::atan(), Head(operator!("arctan", "\\arctan"))),
        (t::acot(), Head(operator!("arccot"))),
        (t::asec(), Head(operator!("arcsec"))),
        (t::acsc(), Head(operator!("arccsc"))),
        (t::asinh(), Head(operator!("arsinh"))),
        (t::acosh(), Head(operator!("arcosh"))),
        (t::atanh(), Head(operator!("artanh"))),
        (t::acoth(), Head(operator!("arcoth"))),
        (t::asech(), Head(operator!("arsech"))),
        (t::acsch(), Head(operator!("arcsch"))),
        (Symbol::ABS, Abs),
        (Symbol::CONJ, Conjugate),
    ]);
    if let Some(dzeta) = crate::get_symbol!("dzeta") {
        table.insert(dzeta, DerivativeOrder(head("<mi>ζ</mi>", "\\zeta", "zeta")));
    }
    table
});

/// The conventional notation of a built-in function symbol, if any.
pub(crate) fn notation(symbol: Symbol) -> Option<Notation> {
    NOTATION.get(&symbol).copied()
}
