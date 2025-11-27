//! Defines polynomials and series.

pub mod evaluate;
pub mod factor;
pub mod gcd;
pub mod groebner;
pub mod polynomial;
mod resultant;
pub mod series;
pub mod univariate;

use std::borrow::Cow;
use std::cmp::Ordering::{self, Equal};
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::iter::Sum;
use std::ops::{Add as OpAdd, AddAssign, DerefMut, Div, Mul as OpMul, Neg, Rem, Sub};
use std::sync::Arc;

use ahash::HashMap;
use smallvec::{SmallVec, smallvec};
use smartstring::{LazyCompact, SmartString};

use crate::atom::{Atom, AtomCore, AtomView, Indeterminate, Symbol};
use crate::coefficient::{Coefficient, CoefficientView, ConvertToRing};
use crate::domains::atom::AtomField;
use crate::domains::factorized_rational_polynomial::{
    FactorizedRationalPolynomial, FromNumeratorAndFactorizedDenominator,
};
use crate::domains::integer::{Integer, gcd_signed, gcd_unsigned};
use crate::domains::rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial};
use crate::domains::{EuclideanDomain, Ring, SelfRing};
use crate::parser::{Operator, Token};
use crate::printer::{PrintOptions, PrintState};
use crate::state::Workspace;

use self::factor::Factorize;
use self::gcd::PolynomialGCD;
use self::polynomial::MultivariatePolynomial;

pub(crate) const INLINED_EXPONENTS: usize = 6;

/// Describes an exponent of a variable in a polynomial.
///
/// The recommended type is `u16` for polynomials
/// and `i16` for negative exponents. For size optimizations
/// `u8` can be used.
pub trait Exponent:
    Hash
    + Debug
    + Display
    + Ord
    + OpMul<Output = Self>
    + Div<Output = Self>
    + Rem<Output = Self>
    + Sub<Output = Self>
    + OpAdd<Output = Self>
    + Sum<Self>
    + AddAssign
    + Clone
    + Copy
    + PartialEq
    + Eq
    + TryFrom<i32>
{
    fn zero() -> Self;
    fn one() -> Self;
    /// Convert the exponent to `i32`. This is always possible, as `i32` is the largest supported exponent type.
    fn to_i32(&self) -> i32;
    /// Convert from `i32`. This function may panic if the exponent is too large.
    fn from_i32(n: i32) -> Self;
    fn is_zero(&self) -> bool;
    fn checked_add(&self, other: &Self) -> Option<Self>;
    fn gcd(&self, other: &Self) -> Self;

    /// Pack a list of exponents into a number, such that arithmetic and
    /// comparisons can be performed. The caller must guarantee that:
    /// - the list is no longer than 8 entries
    /// - each entry is not larger than 255
    fn pack(list: &[Self]) -> u64;
    fn unpack(n: u64, out: &mut [Self]);

    /// Pack a list of exponents into a number, such that arithmetic and
    /// comparisons can be performed. The caller must guarantee that:
    /// - the list is no longer than 4 entries
    /// - each entry is not larger than 2^16 - 1
    fn pack_u16(list: &[Self]) -> u64;
    fn unpack_u16(n: u64, out: &mut [Self]);
}

impl Exponent for u32 {
    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }

    #[inline]
    fn to_i32(&self) -> i32 {
        *self as i32
    }

    #[inline]
    fn from_i32(n: i32) -> Self {
        if n < 0 {
            panic!("Exponent {n} is negative");
        }
        n as u32
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0
    }

    #[inline]
    fn checked_add(&self, other: &Self) -> Option<Self> {
        i32::checked_add(*self as i32, *other as i32).map(|x| x as u32)
    }

    #[inline]
    fn gcd(&self, other: &Self) -> Self {
        gcd_unsigned(*self as u64, *other as u64) as Self
    }

    fn pack(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 8) + (*x as u8 as u64);
        }
        num.swap_bytes()
    }

    fn unpack(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u8, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = *ss as u32;
        }
    }

    fn pack_u16(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 16) + ((*x as u16).to_be() as u64);
        }
        num.swap_bytes()
    }

    fn unpack_u16(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u16, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = ss.swap_bytes() as u32;
        }
    }
}

impl Exponent for i32 {
    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }

    #[inline]
    fn to_i32(&self) -> i32 {
        *self
    }

    #[inline]
    fn from_i32(n: i32) -> Self {
        n
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0
    }

    #[inline]
    fn checked_add(&self, other: &Self) -> Option<Self> {
        i32::checked_add(*self, *other)
    }

    #[inline]
    fn gcd(&self, other: &Self) -> Self {
        gcd_signed(*self as i64, *other as i64) as Self
    }

    // Pack a list of positive exponents.
    fn pack(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 8) + (*x as u8 as u64);
        }
        num.swap_bytes()
    }

    fn unpack(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u8, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = *ss as i32;
        }
    }

    // Pack a list of positive exponents.
    fn pack_u16(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 16) + ((*x as u16).to_be() as u64);
        }
        num.swap_bytes()
    }

    fn unpack_u16(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u16, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = ss.swap_bytes() as i32;
        }
    }
}

impl Exponent for u16 {
    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }

    #[inline]
    fn to_i32(&self) -> i32 {
        *self as i32
    }

    #[inline]
    fn from_i32(n: i32) -> Self {
        if n >= 0 && n <= u16::MAX as i32 {
            n as u16
        } else {
            panic!("Exponent {n} too large for u16");
        }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0
    }

    #[inline]
    fn checked_add(&self, other: &Self) -> Option<Self> {
        u16::checked_add(*self, *other)
    }

    #[inline]
    fn gcd(&self, other: &Self) -> Self {
        gcd_unsigned(*self as u64, *other as u64) as Self
    }

    fn pack(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 8) + (*x as u8 as u64);
        }
        num.swap_bytes()
    }

    fn unpack(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u8, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = *ss as u16;
        }
    }

    fn pack_u16(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 16) + x.to_be() as u64;
        }
        num.swap_bytes()
    }

    fn unpack_u16(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u16, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = ss.swap_bytes();
        }
    }
}

impl Exponent for i16 {
    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }

    #[inline]
    fn to_i32(&self) -> i32 {
        *self as i32
    }

    #[inline]
    fn from_i32(n: i32) -> Self {
        if n >= i16::MIN as i32 && n <= i16::MAX as i32 {
            n as i16
        } else {
            panic!("Exponent {n} too large for i16");
        }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0
    }

    #[inline]
    fn checked_add(&self, other: &Self) -> Option<Self> {
        i16::checked_add(*self, *other)
    }

    #[inline]
    fn gcd(&self, other: &Self) -> Self {
        gcd_signed(*self as i64, *other as i64) as Self
    }

    // Pack a list of positive exponents.
    fn pack(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 8) + (*x as u8 as u64);
        }
        num.swap_bytes()
    }

    fn unpack(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u8, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = *ss as i16;
        }
    }

    // Pack a list of positive exponents.
    fn pack_u16(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 16) + ((*x as u16).to_be() as u64);
        }
        num.swap_bytes()
    }

    fn unpack_u16(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u16, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = ss.swap_bytes() as i16;
        }
    }
}

/// An exponent limited to 255 for efficiency
impl Exponent for u8 {
    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }

    #[inline]
    fn to_i32(&self) -> i32 {
        *self as i32
    }

    #[inline]
    fn from_i32(n: i32) -> Self {
        if n >= 0 && n <= u8::MAX as i32 {
            n as u8
        } else {
            panic!("Exponent {n} too large for u8");
        }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0
    }

    #[inline]
    fn checked_add(&self, other: &Self) -> Option<Self> {
        u8::checked_add(*self, *other)
    }

    #[inline]
    fn gcd(&self, other: &Self) -> Self {
        gcd_unsigned(*self as u64, *other as u64) as Self
    }

    fn pack(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 8) + (*x as u64);
        }
        num.swap_bytes()
    }

    fn unpack(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u8, out.len()) };
        out.copy_from_slice(s);
    }

    fn pack_u16(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 16) + ((*x as u16).to_be() as u64);
        }
        num.swap_bytes()
    }

    fn unpack_u16(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u16, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = ss.swap_bytes() as u8;
        }
    }
}

impl Exponent for i8 {
    #[inline]
    fn zero() -> Self {
        0
    }

    #[inline]
    fn one() -> Self {
        1
    }

    #[inline]
    fn to_i32(&self) -> i32 {
        *self as i32
    }

    #[inline]
    fn from_i32(n: i32) -> Self {
        if n >= i8::MIN as i32 && n <= i8::MAX as i32 {
            n as i8
        } else {
            panic!("Exponent {n} too large for i8");
        }
    }

    #[inline]
    fn is_zero(&self) -> bool {
        *self == 0
    }

    #[inline]
    fn checked_add(&self, other: &Self) -> Option<Self> {
        i8::checked_add(*self, *other)
    }

    #[inline]
    fn gcd(&self, other: &Self) -> Self {
        gcd_signed(*self as i64, *other as i64) as Self
    }

    // Pack a list of positive exponents.
    fn pack(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 8) + (*x as u8 as u64);
        }
        num.swap_bytes()
    }

    fn unpack(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u8, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = *ss as i8;
        }
    }

    // Pack a list of positive exponents.
    fn pack_u16(list: &[Self]) -> u64 {
        let mut num: u64 = 0;
        for x in list.iter().rev() {
            num = (num << 16) + ((*x as u16).to_be() as u64);
        }
        num.swap_bytes()
    }

    fn unpack_u16(mut n: u64, out: &mut [Self]) {
        n = n.swap_bytes();
        let s = unsafe { std::slice::from_raw_parts(&n as *const u64 as *const u16, out.len()) };
        for (o, ss) in out.iter_mut().zip(s) {
            *o = ss.swap_bytes() as i8;
        }
    }
}

/// An exponent that must be zero or higher.
pub trait PositiveExponent: Exponent {
    fn from_u32(n: u32) -> Self {
        if n > i32::MAX as u32 {
            panic!("Exponent {n} too large for i32");
        }
        Self::from_i32(n as i32)
    }
    fn to_u32(&self) -> u32;
}

impl PositiveExponent for u8 {
    #[inline]
    fn to_u32(&self) -> u32 {
        *self as u32
    }
}
impl PositiveExponent for u16 {
    #[inline]
    fn to_u32(&self) -> u32 {
        *self as u32
    }
}
impl PositiveExponent for u32 {
    #[inline]
    fn to_u32(&self) -> u32 {
        *self
    }
}

macro_rules! to_positive {
    ($neg: ty, $pos: ty) => {
        impl<R: Ring> MultivariatePolynomial<R, $neg> {
            /// Convert a polynomial with positive exponents to its unsigned type equivalent
            /// by a safe and almost zero-cost cast.
            ///
            /// Panics if the polynomial has negative exponents.
            pub fn to_positive(self) -> MultivariatePolynomial<R, $pos> {
                if !self.is_polynomial() {
                    panic!("Polynomial has negative exponent");
                }

                unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(self)) }
            }
        }

        impl<R: Ring> MultivariatePolynomial<R, $pos> {
            /// Convert a polynomial with positive exponents to its signed type equivalent
            /// by a safe and almost zero-cost cast.
            ///
            /// Panics if the polynomial has exponents that are too large.
            pub fn to_signed(self) -> MultivariatePolynomial<R, $neg> {
                if self
                    .exponents
                    .iter()
                    .any(|x| x.to_i32() > <$neg>::MAX as i32)
                {
                    panic!("Polynomial has exponents that are too large");
                }

                unsafe { std::mem::transmute_copy(&std::mem::ManuallyDrop::new(self)) }
            }
        }
    };
}

to_positive!(i8, u8);
to_positive!(i16, u16);
to_positive!(i32, u32);

/// A well-order of monomials.
pub trait MonomialOrder: Clone {
    fn cmp<E: Exponent>(a: &[E], b: &[E]) -> Ordering;
}

/// Graded reverse lexicographic ordering of monomials.
#[derive(Clone)]
pub struct GrevLexOrder {}

impl MonomialOrder for GrevLexOrder {
    #[inline]
    fn cmp<E: Exponent>(a: &[E], b: &[E]) -> Ordering {
        let deg: E = a.iter().cloned().sum();
        let deg2: E = b.iter().cloned().sum();

        match deg.cmp(&deg2) {
            Equal => {}
            x => {
                return x;
            }
        }

        for (a1, a2) in a.iter().rev().zip(b.iter().rev()) {
            match a1.cmp(a2) {
                Equal => {}
                x => {
                    return x.reverse();
                }
            }
        }

        Equal
    }
}

/// Lexicographic ordering of monomials.
#[derive(Clone)]
pub struct LexOrder {}

impl MonomialOrder for LexOrder {
    #[inline]
    fn cmp<E: Exponent>(a: &[E], b: &[E]) -> Ordering {
        a.cmp(b)
    }
}

/// A polynomial variable. It is either a (global) symbol
/// a temporary variable (for internal use), an array entry,
/// a function or any non-polynomial power.
///
/// Variables should be constructed using [From] or [Into] on
/// symbols and atoms. Variables can be
/// converted into an atom using [PolyVariable::to_atom].
#[derive(Clone, Hash, PartialEq, Eq, PartialOrd, Ord, Debug)]
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::Encode),
    derive(bincode_trait_derive::Decode),
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
    trait_decode(trait = crate::state::HasStateMap)
)]
pub enum PolyVariable {
    /// A symbol, for example x, y, z, etc.
    Symbol(Symbol),
    /// A function, for example f(x), sin(x), etc.
    Function(Symbol, Atom),
    /// A non-polynomial power, for example x^-1, x^y, etc.
    Power(Atom),
    /// A temporary variable, for internal use.
    #[doc(hidden)]
    Temporary(usize),
}

impl std::fmt::Display for PolyVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PolyVariable::Symbol(v) => f.write_str(v.get_stripped_name()),
            PolyVariable::Temporary(t) => f.write_fmt(format_args!("_TMP_{}", *t)),
            PolyVariable::Function(_, a) | PolyVariable::Power(a) => std::fmt::Display::fmt(a, f),
        }
    }
}

impl From<Symbol> for PolyVariable {
    fn from(i: Symbol) -> PolyVariable {
        PolyVariable::Symbol(i)
    }
}

impl From<Indeterminate> for PolyVariable {
    fn from(i: Indeterminate) -> PolyVariable {
        match i {
            Indeterminate::Symbol(s, _) => PolyVariable::Symbol(s),
            Indeterminate::Function(_, a) => PolyVariable::Function(a.get_symbol().unwrap(), a),
        }
    }
}

impl PartialEq<Symbol> for PolyVariable {
    fn eq(&self, other: &Symbol) -> bool {
        match self {
            PolyVariable::Symbol(s) => *s == *other,
            _ => false,
        }
    }
}

impl<T: AtomCore> PartialEq<T> for PolyVariable {
    fn eq(&self, other: &T) -> bool {
        match self {
            PolyVariable::Symbol(s) => match other.as_atom_view() {
                AtomView::Var(v) => *s == v.get_symbol(),
                _ => false,
            },
            PolyVariable::Function(_, a) | PolyVariable::Power(a) => {
                a.as_view() == other.as_atom_view()
            }
            PolyVariable::Temporary(_) => false,
        }
    }
}

impl TryFrom<Atom> for PolyVariable {
    type Error = String;

    fn try_from(a: Atom) -> Result<PolyVariable, Self::Error> {
        match a {
            Atom::Var(v) => Ok(PolyVariable::Symbol(v.get_symbol())),
            Atom::Fun(f) => Ok(PolyVariable::Function(f.get_symbol(), Atom::Fun(f))),
            Atom::Pow(p) => {
                let (_, exp) = p.to_pow_view().get_base_exp();
                if exp.is_positive() {
                    Err(format!(
                        "Cannot convert {} to a variable as it can be decomposed into a polynomial part",
                        Atom::Pow(p)
                    ))
                } else {
                    Ok(PolyVariable::Power(Atom::Pow(p)))
                }
            }
            _ => Err(format!(
                "Cannot convert {a} to a variable as it can be decomposed into a polynomial part"
            )),
        }
    }
}

impl From<PolyVariable> for Atom {
    fn from(val: PolyVariable) -> Self {
        match val {
            PolyVariable::Symbol(s) => Atom::var(s),
            PolyVariable::Function(_, a) | PolyVariable::Power(a) => a.as_ref().clone(),
            PolyVariable::Temporary(x) => {
                panic!("Cannot convert a temporary variable {x} to an atom")
            }
        }
    }
}

impl PolyVariable {
    /// Get the symbol if this variable is a symbol.
    pub fn get_id(&self) -> Option<Symbol> {
        match self {
            PolyVariable::Symbol(s) => Some(*s),
            _ => None,
        }
    }

    fn format_string(&self, opts: &PrintOptions, state: PrintState) -> String {
        match self {
            PolyVariable::Symbol(v) => v.get_stripped_name().to_string(),
            PolyVariable::Temporary(t) => format!("_TMP_{}", *t),
            PolyVariable::Function(_, a) | PolyVariable::Power(a) => a.format_string(opts, state),
        }
    }

    pub fn to_atom(&self) -> Atom {
        match self {
            PolyVariable::Symbol(s) => Atom::var(*s),
            PolyVariable::Function(_, a) | PolyVariable::Power(a) => a.as_ref().clone(),
            PolyVariable::Temporary(_) => panic!("Cannot convert a temporary variable to an atom"),
        }
    }

    /// Check if the symbol `symbol` appears at most once in the variable map.
    /// For example, `[x,f(x)]` is not independent in `x`, but `[x,y]` is.
    pub fn is_independent_symbol(variables: &[PolyVariable], symbol: Symbol) -> bool {
        let mut seen = false;

        for v in variables {
            match v {
                PolyVariable::Symbol(s) => {
                    if *s == symbol {
                        if seen {
                            return false;
                        }
                        seen = true;
                    }
                }
                PolyVariable::Function(_, f) | PolyVariable::Power(f) => {
                    if f.contains_symbol(symbol) {
                        if seen {
                            return false;
                        }
                        seen = true;
                    }
                }
                PolyVariable::Temporary(_) => {}
            }
        }

        true
    }
}

impl AtomView<'_> {
    /// Convert an expanded expression to a polynomial.
    fn to_polynomial_expanded<R: Ring + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: Option<&Arc<Vec<PolyVariable>>>,
        allow_new_vars: bool,
    ) -> Result<MultivariatePolynomial<R, E>, &'static str> {
        fn check_factor(
            factor: &AtomView<'_>,
            vars: &mut Vec<PolyVariable>,
            allow_new_vars: bool,
        ) -> Result<(), &'static str> {
            match factor {
                AtomView::Num(n) => match n.get_coeff_view() {
                    CoefficientView::FiniteField(_, _) => {
                        Err("Finite field not supported in conversion routine")
                    }
                    _ => Ok(()),
                },
                AtomView::Var(v) => {
                    let name = v.get_symbol();
                    if !vars.contains(&name.into()) {
                        if !allow_new_vars {
                            return Err("Expression contains variable that is not in variable map");
                        } else {
                            vars.push(v.get_symbol().into());
                        }
                    }
                    Ok(())
                }
                AtomView::Fun(_) => Err("function not supported in polynomial"),
                AtomView::Pow(p) => {
                    let (base, exp) = p.get_base_exp();
                    match base {
                        AtomView::Var(v) => {
                            let name = v.get_symbol();
                            if !vars.contains(&name.into()) {
                                if !allow_new_vars {
                                    return Err(
                                        "Expression contains variable that is not in variable map",
                                    );
                                } else {
                                    vars.push(v.get_symbol().into());
                                }
                            }
                        }
                        _ => return Err("base must be a variable"),
                    }

                    match exp {
                        AtomView::Num(n) => match n.get_coeff_view() {
                            CoefficientView::Natural(n, d, ni, _di) => {
                                if d == 1 && ni == 0 && n >= 0 && n <= u32::MAX as i64 {
                                    Ok(())
                                } else {
                                    Err("Exponent negative or a fraction")
                                }
                            }
                            CoefficientView::Large(r, ri) => {
                                let r = r.to_rat();
                                if ri.is_zero()
                                    && r.is_integer()
                                    && !r.is_negative()
                                    && r.numerator_ref() <= &u32::MAX
                                {
                                    Ok(())
                                } else {
                                    Err("Exponent too large or negative or a fraction")
                                }
                            }
                            CoefficientView::Indeterminate => Err("Indeterminate exponent"),
                            CoefficientView::Infinity(_) => Err("Infinite exponent"),
                            CoefficientView::Float(_, _) => {
                                Err("Float is not supported in conversion routine")
                            }
                            CoefficientView::FiniteField(_, _) => {
                                Err("Finite field not supported in conversion routine")
                            }
                            CoefficientView::RationalPolynomial(_) => {
                                Err("Rational polynomial not supported in conversion routine")
                            }
                        },
                        _ => Err("base must be a variable"),
                    }
                }
                AtomView::Add(_) => Err("Expression may not contain subexpressions"),
                AtomView::Mul(_) => unreachable!("Mul inside mul found"),
            }
        }

        fn check_term(
            term: &AtomView<'_>,
            vars: &mut Vec<PolyVariable>,
            allow_new_vars: bool,
        ) -> Result<(), &'static str> {
            match term {
                AtomView::Mul(m) => {
                    for factor in m {
                        check_factor(&factor, vars, allow_new_vars)?;
                    }
                    Ok(())
                }
                _ => check_factor(term, vars, allow_new_vars),
            }
        }

        // get all variables and check structure
        let mut vars = var_map.map(|v| (**v).clone()).unwrap_or_default();
        let mut n_terms = 0;
        match self {
            AtomView::Add(a) => {
                for term in a {
                    check_term(&term, &mut vars, allow_new_vars)?;
                    n_terms += 1;
                }
            }
            _ => {
                check_term(self, &mut vars, allow_new_vars)?;
                n_terms += 1;
            }
        }

        fn parse_factor<R: Ring + ConvertToRing, E: Exponent>(
            factor: &AtomView<'_>,
            vars: &[PolyVariable],
            coefficient: &mut R::Element,
            exponents: &mut SmallVec<[E; INLINED_EXPONENTS]>,
            field: &R,
        ) -> Result<(), &'static str> {
            match factor {
                AtomView::Num(n) => {
                    field.mul_assign(
                        coefficient,
                        &field
                            .try_element_from_coefficient_view(n.get_coeff_view())
                            .map_err(|_| "Conversion error")?,
                    );

                    Ok(())
                }
                AtomView::Var(v) => {
                    let id = v.get_symbol();
                    exponents[vars.iter().position(|v| *v == id).unwrap()] += E::one();
                    Ok(())
                }
                AtomView::Pow(p) => {
                    let (base, exp) = p.get_base_exp();

                    let var_index = match base {
                        AtomView::Var(v) => {
                            let id = v.get_symbol();
                            vars.iter().position(|v| *v == id).unwrap()
                        }
                        _ => unreachable!(),
                    };

                    match exp {
                        AtomView::Num(n) => match n.get_coeff_view() {
                            CoefficientView::Natural(r, _, _, _) => {
                                exponents[var_index] += E::from_i32(r as i32)
                            }
                            CoefficientView::Large(r, _) => {
                                exponents[var_index] +=
                                    E::from_i32(r.to_rat().numerator_ref().to_i64().unwrap() as i32)
                            }
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    }

                    Ok(())
                }
                _ => unreachable!("Unsupported expression"),
            }
        }

        fn parse_term<R: Ring + ConvertToRing, E: Exponent>(
            term: &AtomView<'_>,
            vars: &[PolyVariable],
            poly: &mut MultivariatePolynomial<R, E>,
            field: &R,
        ) -> Result<(), &'static str> {
            let mut coefficient = poly.ring.one();
            let mut exponents = smallvec![E::zero(); vars.len()];

            match term {
                AtomView::Mul(m) => {
                    for factor in m {
                        parse_factor(&factor, vars, &mut coefficient, &mut exponents, field)?;
                    }
                }
                _ => parse_factor(term, vars, &mut coefficient, &mut exponents, field)?,
            }

            poly.append_monomial(coefficient, &exponents);
            Ok(())
        }

        let mut poly =
            MultivariatePolynomial::<R, E>::new(field, Some(n_terms), Arc::new(vars.clone()));

        match self {
            AtomView::Add(a) => {
                for term in a {
                    parse_term(&term, &vars, &mut poly, field)?;
                }
            }
            _ => parse_term(self, &vars, &mut poly, field)?,
        }

        Ok(poly)
    }

    /// Convert the atom to a polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-polynomial parts are automatically
    /// defined as a new independent variable in the polynomial.
    pub(crate) fn try_to_polynomial<R: EuclideanDomain + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: Option<Arc<Vec<PolyVariable>>>,
    ) -> Result<MultivariatePolynomial<R, E>, String> {
        self.to_polynomial_impl(field, var_map.as_ref().unwrap_or(&Arc::new(Vec::new())))
    }

    pub(crate) fn to_polynomial_impl<R: EuclideanDomain + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: &Arc<Vec<PolyVariable>>,
    ) -> Result<MultivariatePolynomial<R, E>, String> {
        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial_expanded(field, Some(var_map), true) {
            return Ok(num);
        }

        match self {
            AtomView::Num(n) => {
                field.try_element_from_coefficient_view(n.get_coeff_view())?; // must fail
                unreachable!("This case should have been handled by the fast routine")
            }
            AtomView::Var(_) => {
                unreachable!("This case should have been handled by the fast routine")
            }
            AtomView::Pow(p) => {
                // the case var^exp is already treated, so this must be a case that requires a map
                // check if the exponent is a positive integer, if so the base must be mapped
                // otherwise, map the entire power

                // TODO: make sure that this coefficient does not depend on any of the variables in var_map

                let (base, exp) = p.get_base_exp();

                if let AtomView::Num(n) = exp {
                    let num_n = n.get_coeff_view();
                    if let CoefficientView::Natural(nn, nd, ni, _di) = num_n
                        && nd == 1
                        && ni == 0
                    {
                        if nn > 0 && nn < i32::MAX as i64 {
                            return Ok(base.to_polynomial_impl(field, var_map)?.pow(nn as usize));
                        } else if nn < 0 && nn > i32::MIN as i64 {
                            // allow x^-2 as a term if supported by the exponent
                            if let Ok(e) = (nn as i32).try_into()
                                && let AtomView::Var(v) = base
                            {
                                let s = PolyVariable::Symbol(v.get_symbol());
                                if let Some(id) = var_map.iter().position(|v| v == &s) {
                                    let mut exp = vec![E::zero(); var_map.len()];
                                    exp[id] = e;
                                    return Ok(MultivariatePolynomial::new(
                                        field,
                                        None,
                                        var_map.clone(),
                                    )
                                    .monomial(field.one(), exp));
                                } else {
                                    let mut var_map = var_map.as_ref().clone();
                                    var_map.push(s);
                                    let mut exp = vec![E::zero(); var_map.len()];
                                    exp[var_map.len() - 1] = e;

                                    return Ok(MultivariatePolynomial::new(
                                        field,
                                        None,
                                        Arc::new(var_map),
                                    )
                                    .monomial(field.one(), exp));
                                }
                            }
                        }
                    }
                }

                // check if we have seen this variable before
                if let Some(id) = var_map.iter().position(|v| match v {
                    PolyVariable::Power(vv) => vv.as_view() == *self,
                    _ => false,
                }) {
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[id] = E::one();
                    Ok(MultivariatePolynomial::new(field, None, var_map.clone())
                        .monomial(field.one(), exp))
                } else {
                    let mut var_map = var_map.as_ref().clone();
                    var_map.push(PolyVariable::Power(self.to_owned()));
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[var_map.len() - 1] = E::one();

                    Ok(MultivariatePolynomial::new(field, None, Arc::new(var_map))
                        .monomial(field.one(), exp))
                }
            }
            AtomView::Fun(f) => {
                // TODO: make sure that this coefficient does not depend on any of the variables in var_map

                // check if we have seen this variable before
                if let Some(id) = var_map.iter().position(|v| match v {
                    PolyVariable::Function(_, vv) => vv.as_view() == *self,
                    _ => false,
                }) {
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[id] = E::one();
                    Ok(MultivariatePolynomial::new(field, None, var_map.clone())
                        .monomial(field.one(), exp))
                } else {
                    let mut var_map = var_map.as_ref().clone();
                    var_map.push(PolyVariable::Function(f.get_symbol(), self.to_owned()));
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[var_map.len() - 1] = E::one();

                    Ok(MultivariatePolynomial::new(field, None, Arc::new(var_map))
                        .monomial(field.one(), exp))
                }
            }
            AtomView::Mul(m) => {
                let mut r =
                    MultivariatePolynomial::new(field, None, var_map.clone()).constant(field.one());
                for arg in m {
                    let mut arg_r = arg.to_polynomial_impl(field, &r.variables)?;
                    r.unify_variables(&mut arg_r);
                    r = &r * &arg_r;
                }
                Ok(r)
            }
            AtomView::Add(a) => {
                let mut r = MultivariatePolynomial::new(field, None, var_map.clone());
                for arg in a {
                    let mut arg_r = arg.to_polynomial_impl(field, &r.variables)?;
                    r.unify_variables(&mut arg_r);
                    r = &r + &arg_r;
                }
                Ok(r)
            }
        }
    }

    /// Convert the atom to a polynomial in specific variables.
    /// All other parts will be collected into the coefficient, which
    /// is a general expression.
    ///
    /// This routine does not perform expansions.
    pub(crate) fn to_polynomial_in_vars<E: Exponent>(
        &self,
        var_map: &Arc<Vec<PolyVariable>>,
    ) -> MultivariatePolynomial<AtomField, E> {
        let poly = MultivariatePolynomial::<_, E>::new(&AtomField::new(), None, var_map.clone());
        self.to_polynomial_in_vars_impl(var_map, &poly)
    }

    /// Convert the atom to a polynomial in specific variables.
    /// All other parts will be collected into the coefficient, which
    /// is a general expression.
    ///
    /// This routine does not perform expansions.
    fn to_polynomial_in_vars_impl<E: Exponent>(
        &self,
        var_map: &Arc<Vec<PolyVariable>>,
        poly: &MultivariatePolynomial<AtomField, E>,
    ) -> MultivariatePolynomial<AtomField, E> {
        let field = AtomField::new();
        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial_expanded(&field, Some(var_map), false) {
            return num;
        }

        match self {
            AtomView::Num(_) | AtomView::Var(_) => poly.constant(self.to_owned()),
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                if let AtomView::Num(n) = exp {
                    let num_n = n.get_coeff_view();
                    if let CoefficientView::Natural(nn, nd, ni, _) = num_n {
                        if ni == 0 && nd == 1 && nn > 0 && nn < i32::MAX as i64 {
                            return base
                                .to_polynomial_in_vars_impl(var_map, poly)
                                .pow(nn as usize);
                        } else if ni == 0 && nd == 1 && nn < 0 && nn > i32::MIN as i64 {
                            // allow x^-2 as a term if supported by the exponent
                            if let Ok(e) = (nn as i32).try_into()
                                && let AtomView::Var(v) = base
                            {
                                let s = PolyVariable::Symbol(v.get_symbol());
                                if let Some(id) = var_map.iter().position(|v| v == &s) {
                                    let mut exp = vec![E::zero(); var_map.len()];
                                    exp[id] = e;
                                    return poly.monomial(field.one(), exp);
                                } else {
                                    return poly.constant(self.to_owned());
                                }
                            }
                        }
                    }
                }

                if let Some(id) = var_map.iter().position(|v| match v {
                    PolyVariable::Power(vv) => vv.as_view() == *self,
                    _ => false,
                }) {
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[id] = E::one();
                    poly.monomial(field.one(), exp)
                } else {
                    poly.constant(self.to_owned())
                }
            }
            AtomView::Fun(_) => {
                if let Some(id) = var_map.iter().position(|v| match v {
                    PolyVariable::Function(_, vv) => vv.as_view() == *self,
                    _ => false,
                }) {
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[id] = E::one();
                    poly.monomial(field.one(), exp)
                } else {
                    poly.constant(self.to_owned())
                }
            }
            AtomView::Mul(m) => {
                let mut r = poly.one();
                for arg in m {
                    let arg_r = arg.to_polynomial_in_vars_impl(&r.variables, poly);
                    r = &r * &arg_r;
                }
                r
            }
            AtomView::Add(a) => {
                let mut r = poly.zero();
                for arg in a {
                    let arg_r = arg.to_polynomial_in_vars_impl(&r.variables, poly);
                    r = &r + &arg_r;
                }
                r
            }
        }
    }

    /// Convert the atom to a rational polynomial, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    pub(crate) fn try_to_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<PolyVariable>>>,
    ) -> Result<RationalPolynomial<RO, E>, String>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        self.to_rational_polynomial_impl(
            field,
            out_field,
            var_map.as_ref().unwrap_or(&Arc::new(Vec::new())),
        )
    }

    fn to_rational_polynomial_impl<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: &Arc<Vec<PolyVariable>>,
    ) -> Result<RationalPolynomial<RO, E>, String>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial_expanded(field, Some(var_map), true) {
            let den = num.one();
            return Ok(RationalPolynomial::from_num_den(num, den, out_field, false));
        }

        match self {
            AtomView::Num(n) => {
                field.try_element_from_coefficient_view(n.get_coeff_view())?; // must fail
                unreachable!("This case should have been handled by the fast routine")
            }
            AtomView::Var(_) => {
                unreachable!("This case should have been handled by the fast routine")
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                if let AtomView::Num(n) = exp {
                    let num_n = n.get_coeff_view();

                    if let CoefficientView::Natural(nn, nd, ni, _) = num_n
                        && ni == 0
                        && nd == 1
                    {
                        let b = base.to_rational_polynomial_impl(field, out_field, var_map)?;

                        return if nn < 0 {
                            let b_inv = b.inv();
                            Ok(b_inv.pow(-nn as u64))
                        } else {
                            Ok(b.pow(nn as u64))
                        };
                    }
                }

                // non-integer exponent, convert to new variable
                if let Some(id) = var_map.iter().position(|v| match v {
                    PolyVariable::Power(vv) => vv.as_view() == *self,
                    _ => false,
                }) {
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[id] = E::one();
                    let r = MultivariatePolynomial::new(field, None, var_map.clone())
                        .monomial(field.one(), exp);
                    let den = r.one();
                    Ok(RationalPolynomial::from_num_den(r, den, out_field, false))
                } else {
                    let mut var_map = var_map.as_ref().clone();
                    var_map.push(PolyVariable::Power(self.to_owned()));
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[var_map.len() - 1] = E::one();

                    let r = MultivariatePolynomial::new(field, None, Arc::new(var_map))
                        .monomial(field.one(), exp);
                    let den = r.one();
                    Ok(RationalPolynomial::from_num_den(r, den, out_field, false))
                }
            }
            AtomView::Fun(f) => {
                // check if we have seen this variable before
                if let Some(id) = var_map.iter().position(|v| match v {
                    PolyVariable::Function(_, vv) => vv.as_view() == *self,
                    _ => false,
                }) {
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[id] = E::one();
                    let r = MultivariatePolynomial::new(field, None, var_map.clone())
                        .monomial(field.one(), exp);
                    let den = r.one();
                    Ok(RationalPolynomial::from_num_den(r, den, out_field, false))
                } else {
                    let mut var_map = var_map.as_ref().clone();
                    var_map.push(PolyVariable::Function(f.get_symbol(), self.to_owned()));
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[var_map.len() - 1] = E::one();

                    let r = MultivariatePolynomial::new(field, None, Arc::new(var_map))
                        .monomial(field.one(), exp);

                    let den = r.one();
                    Ok(RationalPolynomial::from_num_den(r, den, out_field, false))
                }
            }
            AtomView::Mul(m) => {
                let mut r = RationalPolynomial::new(out_field, var_map.clone());
                r.numerator = r.numerator.add_constant(out_field.one());
                for arg in m {
                    let mut arg_r =
                        arg.to_rational_polynomial_impl(field, out_field, &r.numerator.variables)?;
                    r.unify_variables(&mut arg_r);
                    r = &r * &arg_r;
                }
                Ok(r)
            }
            AtomView::Add(a) => {
                let mut r = RationalPolynomial::new(out_field, var_map.clone());
                for arg in a {
                    let mut arg_r =
                        arg.to_rational_polynomial_impl(field, out_field, &r.numerator.variables)?;
                    r.unify_variables(&mut arg_r);
                    r = &r + &arg_r;
                }
                Ok(r)
            }
        }
    }

    /// Convert the atom to a rational polynomial with factorized denominators, optionally in the variable ordering
    /// specified by `var_map`. If new variables are encountered, they are
    /// added to the variable map. Similarly, non-rational polynomial parts are automatically
    /// defined as a new independent variable in the rational polynomial.
    pub(crate) fn try_to_factorized_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: Option<Arc<Vec<PolyVariable>>>,
    ) -> Result<FactorizedRationalPolynomial<RO, E>, String>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize,
    {
        self.to_factorized_rational_polynomial_impl(
            field,
            out_field,
            var_map.as_ref().unwrap_or(&Arc::new(Vec::new())),
        )
    }

    pub fn to_factorized_rational_polynomial_impl<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: &Arc<Vec<PolyVariable>>,
    ) -> Result<FactorizedRationalPolynomial<RO, E>, String>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize,
    {
        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial_expanded(field, Some(var_map), true) {
            let den = vec![(num.one(), 1)];
            return Ok(FactorizedRationalPolynomial::from_num_den(
                num, den, out_field, false,
            ));
        }

        match self {
            AtomView::Num(n) => {
                field.try_element_from_coefficient_view(n.get_coeff_view())?; // must fail
                unreachable!("This case should have been handled by the fast routine")
            }
            AtomView::Var(_) => {
                unreachable!("This case should have been handled by the fast routine")
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();
                if let AtomView::Num(n) = exp {
                    let num_n = n.get_coeff_view();

                    if let CoefficientView::Natural(nn, nd, ni, _) = num_n
                        && ni == 0
                        && nd == 1
                    {
                        let b =
                            base.to_factorized_rational_polynomial_impl(field, out_field, var_map)?;

                        return if nn < 0 {
                            let b_inv = b.inv();
                            Ok(b_inv.pow(-nn as u64))
                        } else {
                            Ok(b.pow(nn as u64))
                        };
                    }
                }

                // non-integer exponent, convert to new variable
                if let Some(id) = var_map.iter().position(|v| match v {
                    PolyVariable::Power(vv) => vv.as_view() == *self,
                    _ => false,
                }) {
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[id] = E::one();
                    let r = MultivariatePolynomial::new(field, None, var_map.clone())
                        .monomial(field.one(), exp);
                    Ok(FactorizedRationalPolynomial::from_num_den(
                        r,
                        vec![],
                        out_field,
                        false,
                    ))
                } else {
                    let mut var_map = var_map.as_ref().clone();
                    var_map.push(PolyVariable::Power(self.to_owned()));
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[var_map.len() - 1] = E::one();

                    let r = MultivariatePolynomial::new(field, None, Arc::new(var_map))
                        .monomial(field.one(), exp);
                    Ok(FactorizedRationalPolynomial::from_num_den(
                        r,
                        vec![],
                        out_field,
                        false,
                    ))
                }
            }
            AtomView::Fun(f) => {
                // check if we have seen this variable before
                if let Some(id) = var_map.iter().position(|v| match v {
                    PolyVariable::Function(_, vv) => vv.as_view() == *self,
                    _ => false,
                }) {
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[id] = E::one();
                    let r = MultivariatePolynomial::new(field, None, var_map.clone())
                        .monomial(field.one(), exp);
                    Ok(FactorizedRationalPolynomial::from_num_den(
                        r,
                        vec![],
                        out_field,
                        false,
                    ))
                } else {
                    let mut var_map = var_map.as_ref().clone();
                    var_map.push(PolyVariable::Function(f.get_symbol(), self.to_owned()));
                    let mut exp = vec![E::zero(); var_map.len()];
                    exp[var_map.len() - 1] = E::one();

                    let r = MultivariatePolynomial::new(field, None, Arc::new(var_map))
                        .monomial(field.one(), exp);
                    Ok(FactorizedRationalPolynomial::from_num_den(
                        r,
                        vec![],
                        out_field,
                        false,
                    ))
                }
            }
            AtomView::Mul(m) => {
                let mut r = FactorizedRationalPolynomial::new(out_field, var_map.clone());
                r.numerator = r.numerator.add_constant(out_field.one());
                r.numer_coeff = out_field.one();
                for arg in m {
                    let mut arg_r = arg.to_factorized_rational_polynomial_impl(
                        field,
                        out_field,
                        &r.numerator.variables,
                    )?;
                    r.unify_variables(&mut arg_r);
                    r = &r * &arg_r;
                }
                Ok(r)
            }
            AtomView::Add(a) => {
                let mut r = FactorizedRationalPolynomial::new(out_field, var_map.clone());
                for arg in a {
                    let mut arg_r = arg.to_factorized_rational_polynomial_impl(
                        field,
                        out_field,
                        &r.numerator.variables,
                    )?;
                    r.unify_variables(&mut arg_r);
                    r = &r + &arg_r;
                }
                Ok(r)
            }
        }
    }
}

impl<E: Exponent, O: MonomialOrder> MultivariatePolynomial<AtomField, E, O> {
    /// Convert the polynomial to an expression, optionally distributing the polynomial variables over coefficient sums.
    pub fn flatten(&self, distribute: bool) -> Atom {
        let mut out = Atom::default();
        Workspace::get_local().with(|ws| self.flatten_impl(distribute, ws, &mut out));
        out
    }

    fn flatten_impl(&self, expand: bool, ws: &Workspace, out: &mut Atom) {
        if self.is_zero() {
            out.set_from_view(&ws.new_num(0).as_view());
            return;
        }

        let add = out.to_add();

        let mut mul_h = ws.new_atom();
        let mut num_h = ws.new_atom();
        let mut pow_h = ws.new_atom();

        let vars: Vec<Atom> = self.variables.iter().map(|v| v.clone().into()).collect();

        let mut sorted_vars = (0..vars.len()).collect::<Vec<_>>();
        sorted_vars.sort_by_key(|&i| vars[i].clone());

        for monomial in self {
            let mul = mul_h.to_mul();

            for i in &sorted_vars {
                let var = &vars[*i];
                let pow = monomial.exponents[*i];
                if pow != E::zero() {
                    if pow != E::one() {
                        num_h.to_num((pow.to_i32() as i64).into());
                        pow_h.to_pow(var.as_view(), num_h.as_view());
                        mul.extend(pow_h.as_view());
                    } else {
                        mul.extend(var.as_view());
                    }
                }
            }

            if expand {
                if let AtomView::Add(a) = monomial.coefficient.as_view() {
                    let mut tmp = ws.new_atom();
                    for term in a {
                        term.mul_with_ws_into(ws, mul_h.as_view(), &mut tmp);
                        add.extend(tmp.as_view());
                    }
                } else {
                    mul.extend(monomial.coefficient.as_view());
                    add.extend(mul_h.as_view());
                }
            } else {
                mul.extend(monomial.coefficient.as_view());
                add.extend(mul_h.as_view());
            }
        }

        let mut norm = ws.new_atom();
        out.as_view().normalize(ws, &mut norm);
        std::mem::swap(norm.deref_mut(), out);
    }
}

impl<R: Ring, E: Exponent, O: MonomialOrder> MultivariatePolynomial<R, E, O> {
    pub fn to_expression(&self) -> Atom
    where
        R::Element: Into<Coefficient>,
    {
        let mut out = Atom::default();
        self.to_expression_into(&mut out);
        out
    }

    pub fn to_expression_into(&self, out: &mut Atom)
    where
        R::Element: Into<Coefficient>,
    {
        Workspace::get_local().with(|ws| self.to_expression_with_map(ws, &HashMap::default(), out));
    }

    pub(crate) fn to_expression_with_map(
        &self,
        workspace: &Workspace,
        map: &HashMap<PolyVariable, AtomView>,
        out: &mut Atom,
    ) where
        R::Element: Into<Coefficient>,
    {
        if self.is_zero() {
            out.set_from_view(&workspace.new_num(0).as_view());
            return;
        }

        let add = out.to_add();

        let mut mul_h = workspace.new_atom();
        let mut num_h = workspace.new_atom();
        let mut pow_h = workspace.new_atom();

        let vars: Vec<_> = self
            .variables
            .iter()
            .map(|v| {
                if let PolyVariable::Temporary(_) = v {
                    let a = map.get(v).expect("Variable missing from map");
                    a.to_owned()
                } else {
                    v.clone().into()
                }
            })
            .collect();

        let mut sorted_vars = (0..vars.len()).collect::<Vec<_>>();
        sorted_vars.sort_by_key(|&i| vars[i].clone());

        for monomial in self {
            let mul = mul_h.to_mul();

            for i in &sorted_vars {
                let var = &vars[*i];
                let pow = monomial.exponents[*i];
                if pow != E::zero() {
                    if pow != E::one() {
                        num_h.to_num((pow.to_i32() as i64).into());
                        pow_h.to_pow(var.as_view(), num_h.as_view());
                        mul.extend(pow_h.as_view());
                    } else {
                        mul.extend(var.as_view());
                    }
                }
            }

            let number = monomial.coefficient.clone().into();
            num_h.to_num(number);
            mul.extend(num_h.as_view());
            add.extend(mul_h.as_view());
        }

        let mut norm = workspace.new_atom();
        out.as_view().normalize(workspace, &mut norm);
        std::mem::swap(norm.deref_mut(), out);
    }

    pub fn to_expression_with_coeff_map<F: Fn(&R, &R::Element, &mut Atom)>(&self, f: F) -> Atom {
        let mut out = Atom::default();
        self.to_expression_with_coeff_map_into(f, &mut out);
        out
    }

    pub fn to_expression_with_coeff_map_into<F: Fn(&R, &R::Element, &mut Atom)>(
        &self,
        f: F,
        out: &mut Atom,
    ) {
        Workspace::get_local().with(|ws| self.to_expression_coeff_map_impl(ws, f, out));
    }

    pub(crate) fn to_expression_coeff_map_impl<F: Fn(&R, &R::Element, &mut Atom)>(
        &self,
        workspace: &Workspace,
        f: F,
        out: &mut Atom,
    ) {
        if self.is_zero() {
            out.set_from_view(&workspace.new_num(0).as_view());
            return;
        }

        let add = out.to_add();

        let mut mul_h = workspace.new_atom();
        let mut var_h = workspace.new_atom();
        let mut num_h = workspace.new_atom();
        let mut pow_h = workspace.new_atom();

        let mut coeff = workspace.new_atom();
        for monomial in self {
            let mul = mul_h.to_mul();

            for (var_id, &pow) in self.variables.iter().zip(monomial.exponents) {
                if pow != E::zero() {
                    match var_id {
                        PolyVariable::Symbol(v) => {
                            var_h.to_var(*v);
                        }
                        PolyVariable::Temporary(_) => {
                            unreachable!("Temporary variables not supported");
                        }
                        PolyVariable::Function(_, a) | PolyVariable::Power(a) => {
                            var_h.set_from_view(&a.as_view());
                        }
                    }

                    if pow != E::one() {
                        num_h.to_num((pow.to_i32() as i64).into());
                        pow_h.to_pow(var_h.as_view(), num_h.as_view());
                        mul.extend(pow_h.as_view());
                    } else {
                        mul.extend(var_h.as_view());
                    }
                }
            }

            f(&self.ring, monomial.coefficient, &mut coeff);
            mul.extend(coeff.as_view());
            add.extend(mul_h.as_view());
        }

        let mut norm = workspace.new_atom();
        out.as_view().normalize(workspace, &mut norm);
        std::mem::swap(norm.deref_mut(), out);
    }
}

impl<R: Ring, E: PositiveExponent> RationalPolynomial<R, E> {
    pub fn to_expression(&self) -> Atom
    where
        R::Element: Into<Coefficient>,
    {
        let mut out = Atom::default();
        self.to_expression_into(&mut out);
        out
    }

    pub fn to_expression_into(&self, out: &mut Atom)
    where
        R::Element: Into<Coefficient>,
    {
        Workspace::get_local().with(|ws| self.to_expression_with_map(ws, &HashMap::default(), out));
    }

    /// Convert from a rational polynomial to an atom. The `map` maps all
    /// temporary variables back to atoms.
    pub(crate) fn to_expression_with_map(
        &self,
        workspace: &Workspace,
        map: &HashMap<PolyVariable, AtomView>,
        out: &mut Atom,
    ) where
        R::Element: Into<Coefficient>,
    {
        if self.denominator.is_one() {
            self.numerator.to_expression_with_map(workspace, map, out);
            return;
        }

        let mul = out.to_mul();

        let mut poly = workspace.new_atom();
        self.numerator
            .to_expression_with_map(workspace, map, &mut poly);
        mul.extend(poly.as_view());

        self.denominator
            .to_expression_with_map(workspace, map, &mut poly);

        let mut pow_h = workspace.new_atom();
        pow_h.to_pow(poly.as_view(), workspace.new_num(-1).as_view());
        mul.extend(pow_h.as_view());

        let mut norm = workspace.new_atom();
        out.as_view().normalize(workspace, &mut norm);
        std::mem::swap(norm.deref_mut(), out);
    }
}

impl Token {
    pub fn to_polynomial<R: Ring + ConvertToRing, E: Exponent>(
        &self,
        field: &R,
        var_map: &Arc<Vec<PolyVariable>>,
        var_name_map: &[SmartString<LazyCompact>],
    ) -> Result<MultivariatePolynomial<R, E>, Cow<'static, str>> {
        fn parse_factor<R: Ring + ConvertToRing, E: Exponent>(
            factor: &Token,
            var_name_map: &[SmartString<LazyCompact>],
            coefficient: &mut R::Element,
            exponents: &mut SmallVec<[E; INLINED_EXPONENTS]>,
            field: &R,
        ) -> Result<(), Cow<'static, str>> {
            match factor {
                Token::Number(n, false) => match n.parse::<Integer>() {
                    Ok(x) => {
                        field.mul_assign(coefficient, &field.element_from_integer(x));
                    }
                    Err(e) => Err(format!("Could not parse number: {e}"))?,
                },
                Token::ID(x) => {
                    let Some(index) = var_name_map.iter().position(|v| v == x) else {
                        Err(format!("Variable {x} not specified in variable map"))?
                    };
                    exponents[index] += E::one();
                }
                Token::Op(_, _, Operator::Neg, args) => {
                    if args.len() != 1 {
                        Err("Wrong args for neg")?;
                    }

                    *coefficient = field.neg(&*coefficient);
                    parse_factor(&args[0], var_name_map, coefficient, exponents, field)?;
                }
                Token::Op(_, _, Operator::Pow, args) => {
                    if args.len() != 2 {
                        Err("Wrong args for pow")?;
                    }

                    let var_index = match &args[0] {
                        Token::ID(v) => match var_name_map.iter().position(|v1| v == v1) {
                            Some(p) => p,
                            None => Err(format!("Variable {v} not specified in variable map"))?,
                        },
                        _ => Err("Unsupported base")?,
                    };

                    match &args[1] {
                        Token::Number(n, false) => {
                            if let Ok(x) = n.parse::<i32>() {
                                exponents[var_index] += E::from_i32(x);
                            } else {
                                Err("Invalid exponent")?
                            };
                        }
                        _ => Err("Unsupported exponent")?,
                    }
                }
                _ => Err("Unsupported expression")?,
            }

            Ok(())
        }

        fn parse_term<R: Ring + ConvertToRing, E: Exponent>(
            term: &Token,
            var_name_map: &[SmartString<LazyCompact>],
            poly: &mut MultivariatePolynomial<R, E>,
            field: &R,
        ) -> Result<(), Cow<'static, str>> {
            let mut coefficient = poly.ring.one();
            let mut exponents = smallvec![E::zero(); var_name_map.len()];

            match term {
                Token::Op(_, _, Operator::Mul, args) => {
                    for factor in args {
                        parse_factor(
                            factor,
                            var_name_map,
                            &mut coefficient,
                            &mut exponents,
                            field,
                        )?;
                    }
                }
                Token::Op(_, _, Operator::Neg, args) => {
                    if args.len() != 1 {
                        Err("Wrong args for neg")?;
                    }

                    coefficient = field.neg(&coefficient);

                    match &args[0] {
                        Token::Op(_, _, Operator::Mul, args) => {
                            for factor in args {
                                parse_factor(
                                    factor,
                                    var_name_map,
                                    &mut coefficient,
                                    &mut exponents,
                                    field,
                                )?;
                            }
                        }
                        _ => parse_factor(
                            &args[0],
                            var_name_map,
                            &mut coefficient,
                            &mut exponents,
                            field,
                        )?,
                    }
                }
                _ => parse_factor(term, var_name_map, &mut coefficient, &mut exponents, field)?,
            }

            poly.append_monomial(coefficient, &exponents);
            Ok(())
        }

        match self {
            Token::Op(_, _, Operator::Add, args) => {
                let mut poly =
                    MultivariatePolynomial::<R, E>::new(field, Some(args.len()), var_map.clone());

                for term in args {
                    parse_term(term, var_name_map, &mut poly, field)?;
                }
                Ok(poly)
            }
            _ => {
                let mut poly = MultivariatePolynomial::<R, E>::new(field, Some(1), var_map.clone());
                parse_term(self, var_name_map, &mut poly, field)?;
                Ok(poly)
            }
        }
    }

    /// Convert a parsed expression to a rational polynomial if possible,
    /// skipping the conversion to a Symbolica expression. This method
    /// is faster if the parsed expression is already in the same format
    /// i.e. the ordering is the same
    pub fn to_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + ConvertToRing + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: &Arc<Vec<PolyVariable>>,
        var_name_map: &[SmartString<LazyCompact>],
    ) -> Result<RationalPolynomial<RO, E>, Cow<'static, str>>
    where
        RationalPolynomial<RO, E>:
            FromNumeratorAndDenominator<R, RO, E> + FromNumeratorAndDenominator<RO, RO, E>,
    {
        // use a faster routine to parse the rational polynomial
        if let Token::RationalPolynomial(r) = self {
            let mut iter = r.split(',');
            let Some(num) = iter.next() else {
                Err("Empty [] in input")?
            };

            let num = Token::parse_polynomial(num.as_bytes(), var_map, var_name_map, field).1;
            let den = if let Some(den) = iter.next() {
                Token::parse_polynomial(den.as_bytes(), var_map, var_name_map, field).1
            } else {
                num.one()
            };

            // in the fast format [a,b], the gcd of a and b should always be 1
            return Ok(RationalPolynomial::from_num_den(num, den, out_field, false));
        }

        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial(field, var_map, var_name_map) {
            let den = num.one();
            return Ok(RationalPolynomial::from_num_den(num, den, out_field, false));
        }

        match self {
            Token::Number(_, false) | Token::ID(_) => {
                let num = self.to_polynomial(field, var_map, var_name_map)?;
                let den = num.one();
                Ok(RationalPolynomial::from_num_den(num, den, out_field, false))
            }
            Token::Op(_, _, Operator::Inv, args) => {
                assert!(args.len() == 1);
                let r = args[0].to_rational_polynomial(field, out_field, var_map, var_name_map)?;
                Ok(r.inv())
            }
            Token::Op(_, _, Operator::Pow, args) => {
                // we have a pow that could not be parsed by to_polynomial
                // if the exponent is not -1, we pass the subexpression to
                // the general routine
                if Token::Number("-1".into(), false) == args[1] {
                    let r =
                        args[0].to_rational_polynomial(field, out_field, var_map, var_name_map)?;
                    Ok(r.inv())
                } else {
                    Workspace::get_local().with(|ws| {
                        let mut atom = ws.new_atom();
                        self.to_atom_with_output_and_var_map(ws, var_map, var_name_map, &mut atom)?;
                        Ok(atom
                            .as_view()
                            .to_rational_polynomial_impl(field, out_field, var_map)?)
                    })
                }
            }
            Token::Op(_, _, Operator::Mul, args) => {
                let mut r = RationalPolynomial::new(out_field, var_map.clone());
                r.numerator = r.numerator.add_constant(out_field.one());
                for arg in args {
                    let mut arg_r =
                        arg.to_rational_polynomial(field, out_field, var_map, var_name_map)?;
                    r.unify_variables(&mut arg_r);
                    r = &r * &arg_r;
                }
                Ok(r)
            }
            Token::Op(_, _, Operator::Add, args) => {
                let mut r = RationalPolynomial::new(out_field, var_map.clone());
                for arg in args {
                    let mut arg_r =
                        arg.to_rational_polynomial(field, out_field, var_map, var_name_map)?;
                    r.unify_variables(&mut arg_r);
                    r = &r + &arg_r;
                }
                Ok(r)
            }
            Token::Op(_, _, Operator::Neg, args) => {
                let r = args[0].to_rational_polynomial(field, out_field, var_map, var_name_map)?;

                Ok(r.neg())
            }
            _ => Workspace::get_local().with(|ws| {
                let mut atom = ws.new_atom();
                self.to_atom_with_output_and_var_map(ws, var_map, var_name_map, &mut atom)?;
                Ok(atom
                    .as_view()
                    .to_rational_polynomial_impl(field, out_field, var_map)?)
            }),
        }
    }

    /// Convert a parsed expression to a rational polynomial if possible,
    /// skipping the conversion to a Symbolica expression. This method
    /// is faster if the parsed expression is already in the same format
    /// i.e. the ordering is the same
    pub fn to_factorized_rational_polynomial<
        R: EuclideanDomain + ConvertToRing,
        RO: EuclideanDomain + ConvertToRing + PolynomialGCD<E>,
        E: PositiveExponent,
    >(
        &self,
        field: &R,
        out_field: &RO,
        var_map: &Arc<Vec<PolyVariable>>,
        var_name_map: &[SmartString<LazyCompact>],
    ) -> Result<FactorizedRationalPolynomial<RO, E>, Cow<'static, str>>
    where
        FactorizedRationalPolynomial<RO, E>: FromNumeratorAndFactorizedDenominator<R, RO, E>
            + FromNumeratorAndFactorizedDenominator<RO, RO, E>,
        MultivariatePolynomial<RO, E>: Factorize,
    {
        // use a faster routine to parse the rational polynomial
        if let Token::RationalPolynomial(r) = self {
            let mut iter = r.split(',');
            let Some(num) = iter.next() else {
                Err("Empty [] in input")?
            };

            let num = Token::parse_polynomial(num.as_bytes(), var_map, var_name_map, field).1;

            let mut dens = vec![];

            let den = if let Some(den) = iter.next() {
                Token::parse_polynomial(den.as_bytes(), var_map, var_name_map, field).1
            } else {
                num.one()
            };

            if let Some(p1) = iter.next() {
                if !den.is_one() {
                    dens.push((
                        den,
                        p1.parse::<usize>()
                            .map_err(|e| format!("Could not parse power: {e}"))?,
                    ));
                }

                while let Some(p) = iter.next() {
                    let den = Token::parse_polynomial(p.as_bytes(), var_map, var_name_map, field).1;

                    let p = iter.next().ok_or("Missing power")?;
                    let p = p
                        .parse::<usize>()
                        .map_err(|e| format!("Could not parse power: {e}"))?;

                    dens.push((den, p));
                }
            } else if !den.is_one() {
                dens.push((den, 1));
            }

            // in the fast format [n,d1,p1,d2,p2,...] every denominator is irreducible and unique
            // TODO: set do_factor to true for [n,d] as this may have just the gcd being 1
            return Ok(FactorizedRationalPolynomial::from_num_den(
                num, dens, out_field, false,
            ));
        }

        // see if the current term can be cast into a polynomial using a fast routine
        if let Ok(num) = self.to_polynomial(field, var_map, var_name_map) {
            let den = vec![(num.one(), 1)];
            return Ok(FactorizedRationalPolynomial::from_num_den(
                num, den, out_field, false,
            ));
        }

        match self {
            Token::Number(_, false) | Token::ID(_) => {
                let num = self.to_polynomial(field, var_map, var_name_map)?;
                let den = vec![(num.one(), 1)];
                Ok(FactorizedRationalPolynomial::from_num_den(
                    num, den, out_field, false,
                ))
            }
            Token::Op(_, _, Operator::Inv, args) => {
                assert!(args.len() == 1);
                let r = args[0].to_factorized_rational_polynomial(
                    field,
                    out_field,
                    var_map,
                    var_name_map,
                )?;
                Ok(r.inv())
            }
            Token::Op(_, _, Operator::Pow, args) => {
                // we have a pow that could not be parsed by to_polynomial
                // if the exponent is not -1, we pass the subexpression to
                // the general routine
                if Token::Number("-1".into(), false) == args[1] {
                    let r = args[0].to_factorized_rational_polynomial(
                        field,
                        out_field,
                        var_map,
                        var_name_map,
                    )?;
                    Ok(r.inv())
                } else {
                    Workspace::get_local().with(|ws| {
                        let mut atom = ws.new_atom();
                        self.to_atom_with_output_and_var_map(ws, var_map, var_name_map, &mut atom)?;
                        Ok(atom
                            .as_view()
                            .to_factorized_rational_polynomial_impl(field, out_field, var_map)?)
                    })
                }
            }
            Token::Op(_, _, Operator::Mul, args) => {
                let mut r = FactorizedRationalPolynomial::new(out_field, var_map.clone());
                r.numerator = r.numerator.add_constant(out_field.one());
                r.numer_coeff = out_field.one();
                for arg in args {
                    if let Token::Op(_, _, Operator::Inv, inv_args) = arg {
                        debug_assert!(inv_args.len() == 1);
                        let mut arg_r = inv_args[0].to_factorized_rational_polynomial(
                            field,
                            out_field,
                            var_map,
                            var_name_map,
                        )?;

                        r.unify_variables(&mut arg_r);
                        r = &r / &arg_r;
                    } else {
                        let mut arg_r = arg.to_factorized_rational_polynomial(
                            field,
                            out_field,
                            var_map,
                            var_name_map,
                        )?;
                        r.unify_variables(&mut arg_r);
                        r = &r * &arg_r;
                    }
                }
                Ok(r)
            }
            Token::Op(_, _, Operator::Add, args) => {
                let mut r = FactorizedRationalPolynomial::new(out_field, var_map.clone());

                // sort based on length, as this may improve performance
                let mut polys: Vec<FactorizedRationalPolynomial<_, _>> = args
                    .iter()
                    .map(|arg| {
                        arg.to_factorized_rational_polynomial(
                            field,
                            out_field,
                            var_map,
                            var_name_map,
                        )
                    })
                    .collect::<Result<_, _>>()?;

                polys.sort_by_key(|p| {
                    p.numerator.nterms()
                        + p.denominators
                            .iter()
                            .map(|(x, _)| x.nterms())
                            .sum::<usize>()
                });

                for mut p in polys {
                    r.unify_variables(&mut p);
                    r = &r + &p;
                }
                Ok(r)
            }
            Token::Op(_, _, Operator::Neg, args) => {
                let r = args[0].to_factorized_rational_polynomial(
                    field,
                    out_field,
                    var_map,
                    var_name_map,
                )?;

                Ok(r.neg())
            }
            _ => Workspace::get_local().with(|ws| {
                let mut atom = ws.new_atom();
                self.to_atom_with_output_and_var_map(ws, var_map, var_name_map, &mut atom)?;
                Ok(atom
                    .as_view()
                    .to_factorized_rational_polynomial_impl(field, out_field, var_map)?)
            }),
        }
    }
}
