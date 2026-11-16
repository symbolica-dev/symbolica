//! Conventional notation for built-in functions, such as `Γ(z)`, `ψ⁽ⁿ⁾(z)`,
//! `Liₛ(z)`, `Jₙ(z)`, `|z|` and `z*`. Other symbols keep their names.
use std::{collections::HashMap, sync::LazyLock};

use crate::{atom::Symbol, transcendental};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Notation {
    /// A conventional function name, such as `Γ` for `gamma`.
    Head(&'static str),
    /// `head⁽ⁿ⁾(z)` for `f(n, z)`, written `head(z)` when `n = 0`.
    DerivativeOrder(&'static str),
    /// `headₛ(z)` for `f(s, z)`.
    Indexed(&'static str),
    /// `|z|`.
    Abs,
    /// `z*`.
    Conjugate,
}

impl Notation {
    /// The function name used when an argument count has no special form.
    pub(super) fn head(self) -> Option<&'static str> {
        match self {
            Self::Head(head) | Self::DerivativeOrder(head) | Self::Indexed(head) => Some(head),
            Self::Abs | Self::Conjugate => None,
        }
    }
}

static NOTATION: LazyLock<HashMap<Symbol, Notation>> = LazyLock::new(|| {
    use Notation::*;
    use transcendental as t;
    // Uppercase Greek and multi-letter names are upright; single Latin and
    // lowercase Greek letters keep the default italic of an identifier.
    let mut table = HashMap::from([
        (t::gamma(), Head("<mi mathvariant=\"normal\">Γ</mi>")),
        (t::zeta(), Head("<mi>ζ</mi>")),
        (t::polygamma(), DerivativeOrder("<mi>ψ</mi>")),
        (t::polylog(), Indexed("<mi>Li</mi>")),
        (t::bessel_j(), Indexed("<mi>J</mi>")),
        (t::bessel_y(), Indexed("<mi>Y</mi>")),
        (t::bessel_i(), Indexed("<mi>I</mi>")),
        (t::bessel_k(), Indexed("<mi>K</mi>")),
        // Inverse functions use the ISO 80000-2 names.
        (t::asin(), Head("<mi>arcsin</mi>")),
        (t::acos(), Head("<mi>arccos</mi>")),
        (t::atan(), Head("<mi>arctan</mi>")),
        (t::acot(), Head("<mi>arccot</mi>")),
        (t::asec(), Head("<mi>arcsec</mi>")),
        (t::acsc(), Head("<mi>arccsc</mi>")),
        (t::asinh(), Head("<mi>arsinh</mi>")),
        (t::acosh(), Head("<mi>arcosh</mi>")),
        (t::atanh(), Head("<mi>artanh</mi>")),
        (t::acoth(), Head("<mi>arcoth</mi>")),
        (t::asech(), Head("<mi>arsech</mi>")),
        (t::acsch(), Head("<mi>arcsch</mi>")),
        (Symbol::ABS, Abs),
        (Symbol::CONJ, Conjugate),
    ]);
    if let Some(dzeta) = crate::get_symbol!("dzeta") {
        table.insert(dzeta, DerivativeOrder("<mi>ζ</mi>"));
    }
    table
});

/// The conventional notation of a built-in function symbol, if any.
pub(super) fn notation(symbol: Symbol) -> Option<Notation> {
    NOTATION.get(&symbol).copied()
}
