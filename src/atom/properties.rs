//! Cheap, conservative mathematical property inference.
//!
//! A missing proof is inconclusive, not false. These queries do not expand,
//! normalize, sample, or invoke the algebraic solver, since normalization itself
//! calls them. Properties of expressions with poles apply where they are defined.

use super::{AtomView, Symbol};
use crate::{coefficient::CoefficientView, domains::rational::Rational, id::ConditionResult};
use ConditionResult::{False, Inconclusive, True};

fn proven(value: bool) -> ConditionResult {
    if value { True } else { Inconclusive }
}

/// Closure under addition/multiplication proves membership when every operand
/// belongs. Failure of closure's premise does not prove nonmembership.
fn closed(results: impl Iterator<Item = ConditionResult>) -> ConditionResult {
    proven(results.into_iter().all(|r| r.is_true()))
}

fn branches(a: ConditionResult, b: ConditionResult) -> ConditionResult {
    if a == b { a } else { Inconclusive }
}

// Both the integers and the reals are closed under subtraction: one summand
// outside the domain plus summands inside stays outside. Two may cancel.
fn sum_membership(results: impl Iterator<Item = ConditionResult>) -> ConditionResult {
    let mut outside = 0;
    for result in results {
        match result {
            True => {}
            False => outside += 1,
            Inconclusive => return Inconclusive,
        }
    }
    match outside {
        0 => True,
        1 => False,
        _ => Inconclusive,
    }
}

impl AtomView<'_> {
    pub(crate) fn is_scalar(&self) -> ConditionResult {
        match self {
            Self::Num(n) => match n.get_coeff_view() {
                CoefficientView::Indeterminate | CoefficientView::Infinity(_) => Inconclusive,
                _ => True,
            },
            Self::Var(v) => proven(v.get_symbol().is_scalar()),
            Self::Fun(f) => {
                let s = f.get_symbol();
                if s.get_id() == Symbol::IF_ID && f.get_nargs() == 3 {
                    return closed(f.iter().skip(1).map(|a| a.is_scalar()));
                }
                if f.get_nargs() == 1
                    && matches!(
                        s.get_id(),
                        Symbol::EXP_ID
                            | Symbol::SIN_ID
                            | Symbol::COS_ID
                            | Symbol::SQRT_ID
                            | Symbol::LOG_ID
                            | Symbol::CONJ_ID
                    )
                {
                    return proven(f.iter().next().unwrap().is_scalar().is_true());
                }
                proven(s.is_scalar())
            }
            Self::Pow(p) => closed([p.get_base().is_scalar(), p.get_exp().is_scalar()].into_iter()),
            Self::Mul(m) => closed(m.iter().map(|a| a.is_scalar())),
            Self::Add(a) => closed(a.iter().map(|a| a.is_scalar())),
        }
    }

    pub(crate) fn is_integer(&self) -> ConditionResult {
        match self {
            Self::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(_, _, _, _) | CoefficientView::Large(_, _) => {
                    n.get_coeff_view().is_integer().into()
                }
                CoefficientView::Float(_, _) => crate::id::exact_numeric_value(*self)
                    .map(|n| (n.im.is_zero() && n.re.is_integer()).into())
                    .unwrap_or(Inconclusive),
                _ => Inconclusive,
            },
            Self::Var(v) => proven(v.get_symbol().is_integer()),
            Self::Fun(f) => {
                if f.get_symbol_id() == Symbol::IF_ID && f.get_nargs() == 3 {
                    let mut args = f.iter().skip(1);
                    return branches(
                        args.next().unwrap().is_integer(),
                        args.next().unwrap().is_integer(),
                    );
                }
                proven(f.get_symbol().is_integer())
            }
            Self::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                proven(
                    base.is_integer().is_true()
                        && exp.is_integer().is_true()
                        && exp.is_nonnegative().is_true(),
                )
            }
            Self::Mul(m) => closed(m.iter().map(|a| a.is_integer())),
            Self::Add(a) => sum_membership(a.iter().map(|a| a.is_integer())),
        }
    }

    pub(crate) fn is_real(&self) -> ConditionResult {
        match self {
            Self::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(_, _, _, _) | CoefficientView::Large(_, _) => {
                    n.get_coeff_view().is_real().into()
                }
                CoefficientView::Float(_, _) => crate::id::exact_numeric_value(*self)
                    .map(|n| n.im.is_zero().into())
                    .unwrap_or(Inconclusive),
                // Formal coefficient rings and undefined values are not evidence
                // either for realness or for nonrealness.
                _ => Inconclusive,
            },
            Self::Var(v) => proven(v.get_symbol().is_real()),
            Self::Fun(f) => {
                let s = f.get_symbol();
                if s.get_id() == Symbol::IF_ID {
                    if f.get_nargs() != 3 {
                        return Inconclusive;
                    }
                    let mut args = f.iter().skip(1);
                    return branches(
                        args.next().unwrap().is_real(),
                        args.next().unwrap().is_real(),
                    );
                }
                match s.get_id() {
                    Symbol::EXP_ID
                    | Symbol::SIN_ID
                    | Symbol::COS_ID
                    | Symbol::SQRT_ID
                    | Symbol::LOG_ID
                    | Symbol::CONJ_ID
                    | Symbol::ABS_ID => {
                        if f.get_nargs() != 1 {
                            return Inconclusive;
                        }
                        let arg = f.iter().next().unwrap();
                        match s.get_id() {
                            Symbol::CONJ_ID => arg.is_real(),
                            Symbol::ABS_ID => True,
                            Symbol::SQRT_ID => proven(arg.is_nonnegative().is_true()),
                            Symbol::LOG_ID => proven(arg.is_positive().is_true()),
                            _ => proven(arg.is_real().is_true()),
                        }
                    }
                    _ => proven(s.is_real()),
                }
            }
            Self::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                proven(
                    base.is_real().is_true()
                        && (exp.is_integer().is_true()
                            || base.is_nonnegative().is_true() && exp.is_real().is_true()),
                )
            }
            Self::Mul(m) => closed(m.iter().map(|a| a.is_real())),
            Self::Add(a) => sum_membership(a.iter().map(|a| a.is_real())),
        }
    }

    pub(crate) fn is_positive(&self) -> ConditionResult {
        match self.sign_hint() {
            Sign::Positive => True,
            Sign::Negative | Sign::Nonpositive | Sign::Zero => False,
            _ if self.is_real().is_false() => False,
            _ => Inconclusive,
        }
    }

    pub(crate) fn is_nonnegative(&self) -> ConditionResult {
        match self.sign_hint() {
            Sign::Positive | Sign::Nonnegative | Sign::Zero => True,
            Sign::Negative => False,
            _ if self.is_real().is_false() => False,
            _ => Inconclusive,
        }
    }

    fn sign_hint(&self) -> Sign {
        use Sign::*;
        match self {
            Self::Num(_) => {
                if let Some(n) = crate::id::exact_numeric_value(*self) {
                    if !n.im.is_zero() {
                        return Unknown;
                    }
                    if n.re.is_zero() {
                        Zero
                    } else if n.re.is_negative() {
                        Negative
                    } else {
                        Positive
                    }
                } else {
                    Unknown
                }
            }
            Self::Var(v) => {
                if v.get_symbol().is_positive() {
                    Positive
                } else {
                    Unknown
                }
            }
            Self::Fun(f) => {
                let s = f.get_symbol();
                if matches!(
                    s.get_id(),
                    Symbol::ABS_ID | Symbol::EXP_ID | Symbol::SQRT_ID | Symbol::CONJ_ID
                ) && f.get_nargs() != 1
                {
                    return Unknown;
                }
                if s.get_id() == Symbol::IF_ID && f.get_nargs() == 3 {
                    let mut args = f.iter().skip(1);
                    return args
                        .next()
                        .unwrap()
                        .sign_hint()
                        .union(args.next().unwrap().sign_hint());
                }
                if f.get_nargs() == 1 {
                    let arg = f.iter().next().unwrap();
                    match s.get_id() {
                        Symbol::ABS_ID => {
                            return match arg.sign_hint() {
                                Positive | Negative => Positive,
                                Zero => Zero,
                                _ => Nonnegative,
                            };
                        }
                        Symbol::EXP_ID if arg.is_real().is_true() => return Positive,
                        Symbol::SQRT_ID => {
                            return match arg.sign_hint() {
                                Positive => Positive,
                                Zero => Zero,
                                Nonnegative => Nonnegative,
                                _ => Unknown,
                            };
                        }
                        Symbol::CONJ_ID => return arg.sign_hint(),
                        _ => {}
                    }
                }
                if s.is_positive() { Positive } else { Unknown }
            }
            Self::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                let sign = base.sign_hint();
                if matches!(sign, Positive | Nonnegative) && exp.is_real().is_true() {
                    return sign;
                }
                if let Ok(e) = Rational::try_from(exp) {
                    if e.is_integer() && !e.is_zero() {
                        if e.numerator_ref() % 2 == 0 && base.is_real().is_true() {
                            return if matches!(sign, Positive | Negative) {
                                Positive
                            } else {
                                Nonnegative
                            };
                        }
                        if matches!(sign, Positive | Negative) || !e.is_negative() {
                            return sign;
                        }
                    }
                    if !e.is_negative() && !e.is_zero() && matches!(sign, Zero | Nonnegative) {
                        return sign;
                    }
                }
                Unknown
            }
            Self::Mul(m) => m.iter().fold(Positive, |s, a| s.mul(a.sign_hint())),
            Self::Add(a) => a.iter().fold(Zero, |s, a| s.add(a.sign_hint())),
        }
    }
}

/// Sign bounds, including the distinction between strict and weak inequalities.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Sign {
    Negative,
    Nonpositive,
    Zero,
    Nonnegative,
    Positive,
    Unknown,
}

impl Sign {
    fn negate(self) -> Self {
        use Sign::*;
        match self {
            Negative => Positive,
            Nonpositive => Nonnegative,
            Zero => Zero,
            Nonnegative => Nonpositive,
            Positive => Negative,
            Unknown => Unknown,
        }
    }

    fn union(self, rhs: Self) -> Self {
        use Sign::*;
        if self == rhs {
            return self;
        }
        match (self, rhs) {
            (Zero | Nonnegative | Positive, Zero | Nonnegative | Positive) => Nonnegative,
            (Zero | Nonpositive | Negative, Zero | Nonpositive | Negative) => Nonpositive,
            _ => Unknown,
        }
    }

    fn add(self, rhs: Self) -> Self {
        use Sign::*;
        match (self, rhs) {
            (Zero, s) | (s, Zero) => s,
            (Positive, Positive | Nonnegative) | (Nonnegative, Positive) => Positive,
            (Negative, Negative | Nonpositive) | (Nonpositive, Negative) => Negative,
            (Nonnegative, Nonnegative) => Nonnegative,
            (Nonpositive, Nonpositive) => Nonpositive,
            _ => Unknown,
        }
    }

    fn mul(self, rhs: Self) -> Self {
        use Sign::*;
        match (self, rhs) {
            (Unknown, _) | (_, Unknown) => Unknown,
            (Zero, _) | (_, Zero) => Zero,
            (Positive, s) | (s, Positive) => s,
            (Negative, s) | (s, Negative) => s.negate(),
            (Nonnegative, Nonnegative) | (Nonpositive, Nonpositive) => Nonnegative,
            _ => Nonpositive,
        }
    }
}
