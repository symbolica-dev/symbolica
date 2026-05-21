use std::{
    fmt::{self, Display, Formatter, Write},
    hash::Hash,
};

use rand::Rng;
use xprec::Df64;

use crate::{
    domains::{RingOps, Set, integer::Integer},
    printer::{self, PrintMode},
};

use super::{Complex, DoubleFloat, F64, Float, SingleFloat};
use crate::domains::{EuclideanDomain, Field, InternalOrdering, Ring, RingPrinter, SelfRing};

/// A field of floating point type `T`. For `f64` fields, use [`FloatField<F64>`].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FloatField<T> {
    rep: T,
}

impl<T> FloatField<T> {
    pub fn from_rep(rep: T) -> Self {
        FloatField { rep }
    }

    pub fn get_rep(&self) -> &T {
        &self.rep
    }
}

impl Default for FloatField<F64> {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for FloatField<DoubleFloat> {
    fn default() -> Self {
        Self::new()
    }
}

impl Default for FloatField<Complex<F64>> {
    fn default() -> Self {
        Self::new()
    }
}

impl FloatField<F64> {
    pub const fn new() -> Self {
        FloatField { rep: F64(0.) }
    }
}

impl FloatField<DoubleFloat> {
    pub const fn new() -> Self {
        FloatField {
            rep: DoubleFloat(Df64::new(0.)),
        }
    }
}

impl FloatField<Complex<F64>> {
    pub const fn new() -> Self {
        FloatField {
            rep: Complex::new(F64(0.), F64(0.)),
        }
    }
}

impl FloatField<Float> {
    pub fn new(prec: u32) -> Self {
        FloatField {
            rep: Float::new(prec),
        }
    }
}

impl<T> Display for FloatField<T> {
    fn fmt(&self, _: &mut Formatter<'_>) -> fmt::Result {
        Ok(())
    }
}

impl<T: SingleFloat + Hash + Eq + InternalOrdering> Set for FloatField<T> {
    type Element = T;

    #[inline(always)]
    fn size(&self) -> Option<Integer> {
        None
    }
}

impl<T: SingleFloat + Hash + Eq + InternalOrdering> RingOps<T> for FloatField<T> {
    #[inline(always)]
    fn add(&self, a: T, b: T) -> Self::Element {
        a + b
    }

    #[inline(always)]
    fn sub(&self, a: T, b: T) -> Self::Element {
        a - b
    }

    #[inline(always)]
    fn mul(&self, a: T, b: T) -> Self::Element {
        a * b
    }

    #[inline(always)]
    fn add_assign(&self, a: &mut Self::Element, b: T) {
        *a += b;
    }

    #[inline(always)]
    fn sub_assign(&self, a: &mut Self::Element, b: T) {
        *a -= b;
    }

    #[inline(always)]
    fn mul_assign(&self, a: &mut Self::Element, b: T) {
        *a *= b;
    }

    #[inline(always)]
    fn add_mul_assign(&self, a: &mut Self::Element, b: T, c: T) {
        // a += b * c
        *a = b.mul_add(&c, a);
    }

    #[inline(always)]
    fn sub_mul_assign(&self, a: &mut Self::Element, b: T, c: T) {
        // a -= b * c
        *a = b.mul_add(&(-c), a);
    }

    #[inline(always)]
    fn neg(&self, a: T) -> Self::Element {
        -a
    }
}

impl<T: SingleFloat + Hash + Eq + InternalOrdering> RingOps<&T> for FloatField<T> {
    #[inline(always)]
    fn add(&self, a: &T, b: &T) -> Self::Element {
        a.clone() + b.clone()
    }

    #[inline(always)]
    fn sub(&self, a: &T, b: &T) -> Self::Element {
        a.clone() - b.clone()
    }

    #[inline(always)]
    fn mul(&self, a: &Self::Element, b: &T) -> Self::Element {
        a.clone() * b.clone()
    }

    #[inline(always)]
    fn add_assign(&self, a: &mut Self::Element, b: &T) {
        *a += b;
    }

    #[inline(always)]
    fn sub_assign(&self, a: &mut Self::Element, b: &T) {
        *a -= b;
    }

    #[inline(always)]
    fn mul_assign(&self, a: &mut Self::Element, b: &T) {
        *a *= b;
    }

    #[inline(always)]
    fn add_mul_assign(&self, a: &mut Self::Element, b: &T, c: &T) {
        // a += b * c
        *a = b.mul_add(c, a);
    }

    #[inline(always)]
    fn sub_mul_assign(&self, a: &mut Self::Element, b: &T, c: &T) {
        // a -= b * c
        *a = b.mul_add(&-c.clone(), a);
    }

    #[inline(always)]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        -a.clone()
    }
}

impl<T: SingleFloat + Hash + Eq + InternalOrdering> Ring for FloatField<T> {
    #[inline(always)]
    fn zero(&self) -> Self::Element {
        self.rep.zero()
    }

    #[inline(always)]
    fn one(&self) -> Self::Element {
        self.rep.one()
    }

    #[inline(always)]
    fn nth(&self, n: Integer) -> Self::Element {
        self.rep.from_rational(&n.into())
    }

    #[inline(always)]
    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        b.pow(e)
    }

    #[inline(always)]
    fn is_zero(&self, a: &Self::Element) -> bool {
        a.is_zero()
    }

    #[inline(always)]
    fn is_one(&self, a: &Self::Element) -> bool {
        a.is_one()
    }

    #[inline(always)]
    fn one_is_gcd_unit() -> bool {
        true
    }

    #[inline(always)]
    fn characteristic(&self) -> Integer {
        0.into()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if a.is_zero() { None } else { Some(a.inv()) }
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        Some(a.clone() / b)
    }

    #[inline(always)]
    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        self.rep.from_i64(rng.random_range(range.0..range.1))
    }

    #[inline(always)]
    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &printer::PrintOptions,
        state: printer::PrintState,
        f: &mut W,
    ) -> Result<bool, fmt::Error> {
        if opts.mode.is_mathematica() {
            let mut s = String::new();
            if let Some(p) = opts.precision {
                if state.in_sum {
                    s.write_fmt(format_args!("{self:+.p$}"))?
                } else {
                    s.write_fmt(format_args!("{self:.p$}"))?
                }
            } else if state.in_sum {
                s.write_fmt(format_args!("{self:+}"))?
            } else {
                s.write_fmt(format_args!("{self}"))?
            }

            f.write_str(&s.replace('e', "*^"))?;
            return Ok(false);
        }

        if let Some(p) = opts.precision {
            if state.in_sum {
                f.write_fmt(format_args!("{element:+.p$}"))?
            } else {
                f.write_fmt(format_args!("{element:.p$}"))?
            }
        } else if state.in_sum {
            f.write_fmt(format_args!("{element:+}"))?
        } else {
            f.write_fmt(format_args!("{element}"))?
        }

        Ok(false)
    }

    #[inline(always)]
    fn printer<'a>(&'a self, element: &'a Self::Element) -> RingPrinter<'a, Self> {
        RingPrinter::new(self, element)
    }
}

impl SelfRing for F64 {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        SingleFloat::is_zero(self)
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        SingleFloat::is_one(self)
    }

    #[inline(always)]
    fn format<W: std::fmt::Write>(
        &self,
        opts: &printer::PrintOptions,
        state: printer::PrintState,
        f: &mut W,
    ) -> Result<bool, fmt::Error> {
        if opts.mode.is_mathematica() || opts.mode.is_latex() || opts.mode.is_typst() {
            let mut s = String::new();
            if let Some(p) = opts.precision {
                if state.in_sum {
                    s.write_fmt(format_args!("{self:+.p$}"))?
                } else {
                    s.write_fmt(format_args!("{self:.p$}"))?
                }
            } else if state.in_sum {
                s.write_fmt(format_args!("{self:+}"))?
            } else {
                s.write_fmt(format_args!("{self}"))?
            }

            if s.contains('e') {
                match opts.mode {
                    PrintMode::Mathematica => s = s.replace('e', "*^"),
                    PrintMode::Latex => s = s.replace('e', "\\cdot 10^{") + "}",
                    PrintMode::Typst => s = s.replace('e', " dot 10^(") + ")",
                    _ => {
                        unreachable!()
                    }
                }
            }

            f.write_str(&s)?;

            return Ok(false);
        }

        if let Some(p) = opts.precision {
            if state.in_sum {
                f.write_fmt(format_args!("{self:+.p$}"))?
            } else {
                f.write_fmt(format_args!("{self:.p$}"))?
            }
        } else if state.in_sum {
            f.write_fmt(format_args!("{self:+}"))?
        } else {
            f.write_fmt(format_args!("{self}"))?
        }

        Ok(false)
    }
}

impl SelfRing for DoubleFloat {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        SingleFloat::is_zero(self)
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        SingleFloat::is_one(self)
    }

    #[inline(always)]
    fn format<W: std::fmt::Write>(
        &self,
        opts: &printer::PrintOptions,
        state: printer::PrintState,
        f: &mut W,
    ) -> Result<bool, fmt::Error> {
        Float::from(*self).format(opts, state, f)
    }
}

impl SelfRing for Float {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        SingleFloat::is_zero(self)
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        SingleFloat::is_one(self)
    }

    #[inline(always)]
    fn format<W: std::fmt::Write>(
        &self,
        opts: &printer::PrintOptions,
        state: printer::PrintState,
        f: &mut W,
    ) -> Result<bool, fmt::Error> {
        if opts.mode.is_mathematica() || opts.mode.is_latex() || opts.mode.is_typst() {
            let mut s = String::new();
            if let Some(p) = opts.precision {
                if state.in_sum {
                    s.write_fmt(format_args!("{self:+.p$}"))?
                } else {
                    s.write_fmt(format_args!("{self:.p$}"))?
                }
            } else if state.in_sum {
                s.write_fmt(format_args!("{self:+}"))?
            } else {
                s.write_fmt(format_args!("{self}"))?
            }

            if s.contains('e') {
                match opts.mode {
                    PrintMode::Mathematica => s = s.replace('e', "*^"),
                    PrintMode::Latex => s = s.replace('e', "\\cdot 10^{") + "}",
                    PrintMode::Typst => s = s.replace('e', " dot 10^(") + ")",
                    _ => {
                        unreachable!()
                    }
                }
            }

            f.write_str(&s)?;

            return Ok(false);
        }

        if let Some(p) = opts.precision {
            if state.in_sum {
                f.write_fmt(format_args!("{self:+.p$}"))?
            } else {
                f.write_fmt(format_args!("{self:.p$}"))?
            }
        } else if state.in_sum {
            f.write_fmt(format_args!("{self:+}"))?
        } else {
            f.write_fmt(format_args!("{self}"))?
        }

        Ok(false)
    }
}

impl SelfRing for Complex<Float> {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        SingleFloat::is_zero(&self.re) && SingleFloat::is_zero(&self.im)
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        SingleFloat::is_one(&self.re) && SingleFloat::is_zero(&self.im)
    }

    #[inline(always)]
    fn format<W: std::fmt::Write>(
        &self,
        opts: &printer::PrintOptions,
        mut state: printer::PrintState,
        f: &mut W,
    ) -> Result<bool, fmt::Error> {
        let re_zero = SingleFloat::is_zero(&self.re);
        let im_zero = SingleFloat::is_zero(&self.im);
        let add_paren =
            (state.in_product || state.in_exp || state.in_exp_base) && !re_zero && !im_zero
                || (state.in_exp || state.in_exp_base) && !im_zero;
        if add_paren {
            f.write_char('(')?;
            state.in_sum = false;
        }

        if !re_zero || im_zero {
            self.re.format(opts, state, f)?;
        }

        if !re_zero && !im_zero {
            state.in_sum = true;
        }

        if !im_zero {
            self.im.format(opts, state, f)?;

            if opts.mode.is_symbolica() && opts.color_builtin_symbols {
                f.write_str("\u{1b}\u{5b}\u{33}\u{35}\u{6d}\u{1d456}\u{1b}\u{5b}\u{30}\u{6d}")?;
            } else if opts.mode.is_mathematica() {
                f.write_char('I')?;
            } else {
                f.write_char('𝑖')?;
            }
        }

        if add_paren {
            f.write_char(')')?;
        }

        Ok(false)
    }
}

impl<T: SingleFloat + Hash + Eq + InternalOrdering> EuclideanDomain for FloatField<T> {
    #[inline(always)]
    fn rem(&self, a: &Self::Element, _: &Self::Element) -> Self::Element {
        a.zero()
    }

    #[inline(always)]
    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (a.clone() / b, a.zero())
    }

    #[inline(always)]
    fn gcd(&self, a: &Self::Element, _: &Self::Element) -> Self::Element {
        a.one()
    }
}

impl<T: SingleFloat + Hash + Eq + InternalOrdering> Field for FloatField<T> {
    #[inline(always)]
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone() / b
    }

    #[inline(always)]
    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a /= b;
    }

    #[inline(always)]
    fn inv(&self, a: &Self::Element) -> Self::Element {
        a.inv()
    }
}
