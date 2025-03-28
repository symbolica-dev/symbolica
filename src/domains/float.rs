//! Floating-point numbers and traits.

use std::{
    f64::consts::{LOG10_2, LOG2_10},
    fmt::{self, Debug, Display, Formatter, LowerExp, Write},
    hash::Hash,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use rand::Rng;
use wide::{f64x2, f64x4};

use crate::domains::integer::Integer;

use super::{rational::Rational, EuclideanDomain, Field, InternalOrdering, Ring, SelfRing};

#[derive(Debug, Clone, PartialEq, Hash)]
pub struct MultiPrecisionFloat;

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
}

impl FloatField<F64> {
    pub fn new() -> Self {
        FloatField { rep: (0.).into() }
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

impl<T: NumericalFloatLike + SingleFloat + Hash + Eq + InternalOrdering> Ring for FloatField<T> {
    type Element = T;

    #[inline(always)]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone() + b.clone()
    }

    #[inline(always)]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone() - b.clone()
    }

    #[inline(always)]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone() * b.clone()
    }

    #[inline(always)]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a += b;
    }

    #[inline(always)]
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a -= b;
    }

    #[inline(always)]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a *= b;
    }

    #[inline(always)]
    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        // a += b * c
        *a = b.mul_add(c, a);
    }

    #[inline(always)]
    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        // a -= b * c
        *a = b.mul_add(&-c.clone(), a);
    }

    #[inline(always)]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        -a.clone()
    }

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

    #[inline(always)]
    fn size(&self) -> Integer {
        0.into()
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        Some(a.clone() / b)
    }

    #[inline(always)]
    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        self.rep.from_i64(rng.gen_range(range.0..range.1))
    }

    #[inline(always)]
    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &crate::printer::PrintOptions,
        state: crate::printer::PrintState,
        f: &mut W,
    ) -> Result<bool, fmt::Error> {
        if let Some(p) = opts.precision {
            if state.in_sum {
                f.write_fmt(format_args!("{:+.*}", p, element))?
            } else {
                f.write_fmt(format_args!("{:.*}", p, element))?
            }
        } else if state.in_sum {
            f.write_fmt(format_args!("{:+}", element))?
        } else {
            f.write_fmt(format_args!("{}", element))?
        }

        Ok(false)
    }

    #[inline(always)]
    fn printer<'a>(&'a self, element: &'a Self::Element) -> super::RingPrinter<'a, Self> {
        super::RingPrinter::new(self, element)
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
        opts: &crate::printer::PrintOptions,
        state: crate::printer::PrintState,
        f: &mut W,
    ) -> Result<bool, fmt::Error> {
        if let Some(p) = opts.precision {
            if state.in_sum {
                f.write_fmt(format_args!("{:+.*}", p, self))?
            } else {
                f.write_fmt(format_args!("{:.*}", p, self))?
            }
        } else if state.in_sum {
            f.write_fmt(format_args!("{:+}", self))?
        } else {
            f.write_fmt(format_args!("{}", self))?
        }

        Ok(false)
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
        opts: &crate::printer::PrintOptions,
        state: crate::printer::PrintState,
        f: &mut W,
    ) -> Result<bool, fmt::Error> {
        if let Some(p) = opts.precision {
            if state.in_sum {
                f.write_fmt(format_args!("{:+.*}", p, self))?
            } else {
                f.write_fmt(format_args!("{:.*}", p, self))?
            }
        } else if state.in_sum {
            f.write_fmt(format_args!("{:+}", self))?
        } else {
            f.write_fmt(format_args!("{}", self))?
        }

        Ok(false)
    }
}

impl<T: NumericalFloatLike + SingleFloat + Hash + Eq + InternalOrdering> EuclideanDomain
    for FloatField<T>
{
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

impl<T: NumericalFloatLike + SingleFloat + Hash + Eq + InternalOrdering> Field for FloatField<T> {
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

/// A number, that is potentially floating point.
pub trait NumericalFloatLike:
    PartialEq
    + Clone
    + Debug
    + LowerExp
    + Display
    + std::ops::Neg<Output = Self>
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + for<'a> Div<&'a Self, Output = Self>
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + for<'a> DivAssign<&'a Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
{
    /// Perform `(self * a) + b`.
    fn mul_add(&self, a: &Self, b: &Self) -> Self;
    fn neg(&self) -> Self;
    fn zero(&self) -> Self;
    /// Create a zero that should only be used as a temporary value,
    /// as for some types it may have wrong precision information.
    fn new_zero() -> Self;
    fn one(&self) -> Self;
    fn pow(&self, e: u64) -> Self;
    fn inv(&self) -> Self;

    fn from_usize(&self, a: usize) -> Self;
    fn from_i64(&self, a: i64) -> Self;

    /// Get the number of precise binary digits.
    fn get_precision(&self) -> u32;
    fn get_epsilon(&self) -> f64;
    /// Return true iff the precision is fixed, or false
    /// if the precision is changed dynamically.
    fn fixed_precision(&self) -> bool;

    /// Sample a point on the interval [0, 1].
    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self;
}

/// A number that behaves like a single number.
pub trait SingleFloat: NumericalFloatLike {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn is_finite(&self) -> bool;
    /// Convert a rational to a float with the same precision as the current float.
    fn from_rational(&self, rat: &Rational) -> Self;
}

/// A number that can be converted to a usize, f64, or rounded to the nearest integer.
pub trait RealNumberLike: SingleFloat {
    fn to_usize_clamped(&self) -> usize;
    fn to_f64(&self) -> f64;
    fn round_to_nearest_integer(&self) -> Integer;
}

/// A float that can be constructed without any parameters, such as f64.
pub trait ConstructibleFloat: NumericalFloatLike {
    fn new_one() -> Self;
    fn new_from_usize(a: usize) -> Self;
    fn new_from_i64(a: i64) -> Self;
    /// Sample a point on the interval [0, 1].
    fn new_sample_unit<R: Rng + ?Sized>(rng: &mut R) -> Self;
}

/// A number that behaves like a real number, with constants like π and e
/// and functions like sine and cosine.
pub trait Real: NumericalFloatLike {
    /// The constant π, 3.1415926535...
    fn pi(&self) -> Self;
    /// Euler's number, 2.7182818...
    fn e(&self) -> Self;
    /// The Euler-Mascheroni constant, 0.5772156649...
    fn euler(&self) -> Self;
    /// The golden ratio, 1.6180339887...
    fn phi(&self) -> Self;
    /// The imaginary unit, if it exists.
    fn i(&self) -> Option<Self>;

    fn norm(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn log(&self) -> Self;
    fn exp(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn tan(&self) -> Self;
    fn asin(&self) -> Self;
    fn acos(&self) -> Self;
    fn atan2(&self, x: &Self) -> Self;
    fn sinh(&self) -> Self;
    fn cosh(&self) -> Self;
    fn tanh(&self) -> Self;
    fn asinh(&self) -> Self;
    fn acosh(&self) -> Self;
    fn atanh(&self) -> Self;
    fn powf(&self, e: &Self) -> Self;
}

impl NumericalFloatLike for f64 {
    #[inline(always)]
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        f64::mul_add(*self, *a, *b)
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        -self
    }

    #[inline(always)]
    fn zero(&self) -> Self {
        0.
    }

    #[inline(always)]
    fn new_zero() -> Self {
        0.
    }

    #[inline(always)]
    fn one(&self) -> Self {
        1.
    }

    #[inline]
    fn pow(&self, e: u64) -> Self {
        // FIXME: use binary exponentiation
        debug_assert!(e <= i32::MAX as u64);
        self.powi(e as i32)
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        1. / self
    }

    #[inline(always)]
    fn from_usize(&self, a: usize) -> Self {
        a as f64
    }

    #[inline(always)]
    fn from_i64(&self, a: i64) -> Self {
        a as f64
    }

    #[inline(always)]
    fn get_precision(&self) -> u32 {
        53
    }

    #[inline(always)]
    fn get_epsilon(&self) -> f64 {
        f64::EPSILON / 2.
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        true
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        rng.gen()
    }
}

impl SingleFloat for f64 {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        *self == 0.
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        *self == 1.
    }

    #[inline(always)]
    fn is_finite(&self) -> bool {
        (*self).is_finite()
    }

    #[inline(always)]
    fn from_rational(&self, rat: &Rational) -> Self {
        rat.to_f64()
    }
}

impl RealNumberLike for f64 {
    fn to_usize_clamped(&self) -> usize {
        *self as usize
    }

    fn to_f64(&self) -> f64 {
        *self
    }

    #[inline(always)]
    fn round_to_nearest_integer(&self) -> Integer {
        if *self < 0. {
            Integer::from_f64(*self - 0.5)
        } else {
            Integer::from_f64(*self + 0.5)
        }
    }
}

impl ConstructibleFloat for f64 {
    #[inline(always)]
    fn new_one() -> Self {
        1.
    }

    #[inline(always)]
    fn new_from_usize(a: usize) -> Self {
        a as f64
    }

    #[inline(always)]
    fn new_from_i64(a: i64) -> Self {
        a as f64
    }

    #[inline(always)]
    fn new_sample_unit<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.gen()
    }
}

impl Real for f64 {
    #[inline(always)]
    fn pi(&self) -> Self {
        std::f64::consts::PI
    }

    #[inline(always)]
    fn e(&self) -> Self {
        std::f64::consts::E
    }

    #[inline(always)]
    fn euler(&self) -> Self {
        0.57721566490153286
    }

    #[inline(always)]
    fn phi(&self) -> Self {
        1.6180339887498948
    }

    #[inline(always)]
    fn i(&self) -> Option<Self> {
        None
    }

    #[inline(always)]
    fn norm(&self) -> Self {
        f64::abs(*self)
    }

    #[inline(always)]
    fn sqrt(&self) -> Self {
        (*self).sqrt()
    }

    #[inline(always)]
    fn log(&self) -> Self {
        (*self).ln()
    }

    #[inline(always)]
    fn exp(&self) -> Self {
        (*self).exp()
    }

    #[inline(always)]
    fn sin(&self) -> Self {
        (*self).sin()
    }

    #[inline(always)]
    fn cos(&self) -> Self {
        (*self).cos()
    }

    #[inline(always)]
    fn tan(&self) -> Self {
        (*self).tan()
    }

    #[inline(always)]
    fn asin(&self) -> Self {
        (*self).asin()
    }

    #[inline(always)]
    fn acos(&self) -> Self {
        (*self).acos()
    }

    #[inline(always)]
    fn atan2(&self, x: &Self) -> Self {
        (*self).atan2(*x)
    }

    #[inline(always)]
    fn sinh(&self) -> Self {
        (*self).sinh()
    }

    #[inline(always)]
    fn cosh(&self) -> Self {
        (*self).cosh()
    }

    #[inline(always)]
    fn tanh(&self) -> Self {
        (*self).tanh()
    }

    #[inline(always)]
    fn asinh(&self) -> Self {
        (*self).asinh()
    }

    #[inline(always)]
    fn acosh(&self) -> Self {
        (*self).acosh()
    }

    #[inline(always)]
    fn atanh(&self) -> Self {
        (*self).atanh()
    }

    #[inline]
    fn powf(&self, e: &f64) -> Self {
        (*self).powf(*e)
    }
}

impl From<&Rational> for f64 {
    fn from(value: &Rational) -> Self {
        value.to_f64()
    }
}

impl From<Rational> for f64 {
    fn from(value: Rational) -> Self {
        value.to_f64()
    }
}

/// A wrapper around `f64` that implements `Eq`, `Ord`, and `Hash`.
/// All `NaN` values are considered equal, and `-0` is considered equal to `0`.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Copy, Clone)]
pub struct F64(f64);

impl F64 {
    pub fn into_inner(self) -> f64 {
        self.0
    }
}

impl NumericalFloatLike for F64 {
    #[inline(always)]
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        self.0.mul_add(a.0, b.0).into()
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        (-self.0).into()
    }

    #[inline(always)]
    fn zero(&self) -> Self {
        (0.).into()
    }

    #[inline(always)]
    fn new_zero() -> Self {
        (0.).into()
    }

    #[inline(always)]
    fn one(&self) -> Self {
        (1.).into()
    }

    #[inline(always)]
    fn pow(&self, e: u64) -> Self {
        NumericalFloatLike::pow(&self.0, e).into()
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        self.0.inv().into()
    }

    #[inline(always)]
    fn from_usize(&self, a: usize) -> Self {
        self.0.from_usize(a).into()
    }

    #[inline(always)]
    fn from_i64(&self, a: i64) -> Self {
        self.0.from_i64(a).into()
    }

    #[inline(always)]
    fn get_precision(&self) -> u32 {
        self.0.get_precision().into()
    }

    #[inline(always)]
    fn get_epsilon(&self) -> f64 {
        self.0.get_epsilon().into()
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        self.0.fixed_precision().into()
    }

    #[inline(always)]
    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        self.0.sample_unit(rng).into()
    }
}

impl Neg for F64 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        self.0.neg().into()
    }
}

impl Add<&F64> for F64 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        (self.0 + rhs.0).into()
    }
}

impl Add<F64> for F64 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        (self.0 + rhs.0).into()
    }
}

impl Sub<&F64> for F64 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        (self.0 - rhs.0).into()
    }
}

impl Sub<F64> for F64 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        (self.0 - rhs.0).into()
    }
}

impl Mul<&F64> for F64 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        (self.0 * rhs.0).into()
    }
}

impl Mul<F64> for F64 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        (self.0 * rhs.0).into()
    }
}

impl Div<&F64> for F64 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &Self) -> Self::Output {
        (self.0 / rhs.0).into()
    }
}

impl Div<F64> for F64 {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        (self.0 / rhs.0).into()
    }
}

impl AddAssign<&F64> for F64 {
    #[inline]
    fn add_assign(&mut self, rhs: &F64) {
        self.0 += rhs.0;
    }
}

impl AddAssign<F64> for F64 {
    #[inline]
    fn add_assign(&mut self, rhs: F64) {
        self.0 += rhs.0;
    }
}

impl SubAssign<&F64> for F64 {
    #[inline]
    fn sub_assign(&mut self, rhs: &F64) {
        self.0 -= rhs.0;
    }
}

impl SubAssign<F64> for F64 {
    #[inline]
    fn sub_assign(&mut self, rhs: F64) {
        self.0 -= rhs.0;
    }
}

impl MulAssign<&F64> for F64 {
    #[inline]
    fn mul_assign(&mut self, rhs: &F64) {
        self.0 *= rhs.0;
    }
}

impl MulAssign<F64> for F64 {
    #[inline]
    fn mul_assign(&mut self, rhs: F64) {
        self.0 *= rhs.0;
    }
}

impl DivAssign<&F64> for F64 {
    #[inline]
    fn div_assign(&mut self, rhs: &F64) {
        self.0 /= rhs.0
    }
}

impl DivAssign<F64> for F64 {
    #[inline]
    fn div_assign(&mut self, rhs: F64) {
        self.0 /= rhs.0
    }
}

impl SingleFloat for F64 {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.0.is_one()
    }

    #[inline(always)]
    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }

    #[inline(always)]
    fn from_rational(&self, rat: &Rational) -> Self {
        rat.to_f64().into()
    }
}

impl RealNumberLike for F64 {
    fn to_usize_clamped(&self) -> usize {
        self.0.to_usize_clamped()
    }

    fn to_f64(&self) -> f64 {
        self.0.to_f64()
    }

    #[inline(always)]
    fn round_to_nearest_integer(&self) -> Integer {
        self.0.round_to_nearest_integer()
    }
}

impl ConstructibleFloat for F64 {
    #[inline(always)]
    fn new_one() -> Self {
        f64::new_one().into()
    }

    #[inline(always)]
    fn new_from_usize(a: usize) -> Self {
        f64::new_from_usize(a).into()
    }

    #[inline(always)]
    fn new_from_i64(a: i64) -> Self {
        f64::new_from_i64(a).into()
    }

    #[inline(always)]
    fn new_sample_unit<R: Rng + ?Sized>(rng: &mut R) -> Self {
        f64::new_sample_unit(rng).into()
    }
}

impl Real for F64 {
    #[inline(always)]
    fn pi(&self) -> Self {
        std::f64::consts::PI.into()
    }

    #[inline(always)]
    fn e(&self) -> Self {
        std::f64::consts::E.into()
    }

    #[inline(always)]
    fn euler(&self) -> Self {
        0.57721566490153286.into()
    }

    #[inline(always)]
    fn phi(&self) -> Self {
        1.6180339887498948.into()
    }

    #[inline(always)]
    fn i(&self) -> Option<Self> {
        None
    }

    #[inline(always)]
    fn norm(&self) -> Self {
        self.0.norm().into()
    }

    #[inline(always)]
    fn sqrt(&self) -> Self {
        self.0.sqrt().into()
    }

    #[inline(always)]
    fn log(&self) -> Self {
        self.0.ln().into()
    }

    #[inline(always)]
    fn exp(&self) -> Self {
        self.0.exp().into()
    }

    #[inline(always)]
    fn sin(&self) -> Self {
        self.0.sin().into()
    }

    #[inline(always)]
    fn cos(&self) -> Self {
        self.0.cos().into()
    }

    #[inline(always)]
    fn tan(&self) -> Self {
        self.0.tan().into()
    }

    #[inline(always)]
    fn asin(&self) -> Self {
        self.0.asin().into()
    }

    #[inline(always)]
    fn acos(&self) -> Self {
        self.0.acos().into()
    }

    #[inline(always)]
    fn atan2(&self, x: &Self) -> Self {
        self.0.atan2(x.0).into()
    }

    #[inline(always)]
    fn sinh(&self) -> Self {
        self.0.sinh().into()
    }

    #[inline(always)]
    fn cosh(&self) -> Self {
        self.0.cosh().into()
    }

    #[inline(always)]
    fn tanh(&self) -> Self {
        self.0.tanh().into()
    }

    #[inline(always)]
    fn asinh(&self) -> Self {
        self.0.asinh().into()
    }

    #[inline(always)]
    fn acosh(&self) -> Self {
        self.0.acosh().into()
    }

    #[inline(always)]
    fn atanh(&self) -> Self {
        self.0.atanh().into()
    }

    #[inline(always)]
    fn powf(&self, e: &Self) -> Self {
        self.0.powf(e.0).into()
    }
}

impl From<f64> for F64 {
    #[inline(always)]
    fn from(value: f64) -> Self {
        F64(value)
    }
}

impl PartialEq for F64 {
    fn eq(&self, other: &Self) -> bool {
        if self.0.is_nan() && other.0.is_nan() {
            true
        } else {
            self.0 == other.0
        }
    }
}

impl PartialOrd for F64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl InternalOrdering for F64 {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl Display for F64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.0, f)
    }
}

impl LowerExp for F64 {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        LowerExp::fmt(&self.0, f)
    }
}

impl Eq for F64 {}

impl Hash for F64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        if self.0.is_nan() {
            state.write_u64(0x7ff8000000000000);
        } else if self.0 == 0. {
            state.write_u64(0);
        } else {
            state.write_u64(self.0.to_bits());
        }
    }
}

/// A multi-precision floating point type. Operations on this type
/// loosely track the precision of the result, but always overestimate.
/// Some operations may improve precision, such as `sqrt` or adding an
/// infinite-precision integer.
///
/// Floating point output with less than five significant binary digits
/// should be considered unreliable.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone)]
pub struct Float(MultiPrecisionFloat);

#[cfg(feature = "bincode")]
impl bincode::Encode for Float {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        //self.0.prec().encode(encoder)?;
        //self.0.to_string_radix(16, None).encode(encoder)
        todo!()
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(Float);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for Float {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let prec = u32::decode(decoder)?;
        let r = String::decode(decoder)?;
        /*let val = MultiPrecisionFloat::parse_radix(&r, 16)
            .map_err(|_| bincode::error::DecodeError::Other("Failed to parse float from string"))?
            .complete(prec);
        Ok(Float(val))*/
        todo!()
    }
}

impl Debug for Float {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.0, f)
    }
}

impl PartialEq for Float {
    fn eq(&self, other: &Self) -> bool {
        todo!()
    }
}

impl Eq for Float {}

impl Hash for Float {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        todo!()
    }
}

impl InternalOrdering for Float {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl Display for Float {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

impl LowerExp for Float {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

impl PartialOrd for Float {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        todo!()
    }
}

impl Neg for Float {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        todo!()
    }
}

impl Add<&Float> for Float {
    type Output = Self;

    /// Add two floats, while keeping loose track of the precision.
    /// The precision of the output will be at most 2 binary digits too high.
    #[inline]
    fn add(mut self, rhs: &Self) -> Self::Output {
        todo!()
    }
}

impl Add<Float> for Float {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        if rhs.prec() > self.prec() {
            rhs + &self
        } else {
            self + &rhs
        }
    }
}

impl Sub<&Float> for Float {
    type Output = Self;

    #[inline]
    fn sub(mut self, rhs: &Self) -> Self::Output {
        todo!()
    }
}

impl Sub<Float> for Float {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        if rhs.prec() > self.prec() {
            -rhs + &self
        } else {
            self - &rhs
        }
    }
}

impl Mul<&Float> for Float {
    type Output = Self;

    #[inline]
    fn mul(mut self, rhs: &Self) -> Self::Output {
        todo!()
    }
}

impl Mul<Float> for Float {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl Div<&Float> for Float {
    type Output = Self;

    #[inline]
    fn div(mut self, rhs: &Self) -> Self::Output {
        todo!()
    }
}

impl Div<Float> for Float {
    type Output = Self;

    #[inline]
    fn div(mut self, rhs: Self) -> Self::Output {
        todo!()
    }
}

impl AddAssign<&Float> for Float {
    #[inline]
    fn add_assign(&mut self, rhs: &Float) {
        todo!()
    }
}

impl AddAssign<Float> for Float {
    #[inline]
    fn add_assign(&mut self, rhs: Float) {
        self.add_assign(&rhs)
    }
}

impl SubAssign<&Float> for Float {
    #[inline]
    fn sub_assign(&mut self, rhs: &Float) {
        todo!()
    }
}

impl SubAssign<Float> for Float {
    #[inline]
    fn sub_assign(&mut self, rhs: Float) {
        self.sub_assign(&rhs)
    }
}

impl MulAssign<&Float> for Float {
    #[inline]
    fn mul_assign(&mut self, rhs: &Float) {
        todo!()
    }
}

impl MulAssign<Float> for Float {
    #[inline]
    fn mul_assign(&mut self, rhs: Float) {
        todo!()
    }
}

impl DivAssign<&Float> for Float {
    #[inline]
    fn div_assign(&mut self, rhs: &Float) {
        todo!()
    }
}

impl DivAssign<Float> for Float {
    #[inline]
    fn div_assign(&mut self, rhs: Float) {
        todo!()
    }
}

impl Add<i64> for Float {
    type Output = Self;

    /// Add an infinite-precision `i64` to the float.
    #[inline]
    fn add(mut self, rhs: i64) -> Self::Output {
        todo!()
    }
}

impl Add<Float> for i64 {
    type Output = Float;

    /// Add a float to an infinite-precision `i64`.
    #[inline]
    fn add(self, rhs: Float) -> Self::Output {
        rhs + self
    }
}

impl Sub<i64> for Float {
    type Output = Self;

    /// Subtract an infinite-precision `i64` from a float.
    #[inline]
    fn sub(self, rhs: i64) -> Self::Output {
        self + -rhs
    }
}

impl Sub<Float> for i64 {
    type Output = Float;

    /// Subtract a float from an infinite-precision `i64`.
    #[inline]
    fn sub(self, rhs: Float) -> Self::Output {
        -rhs + self
    }
}

impl Mul<i64> for Float {
    type Output = Self;

    /// Multiply an infinite-precision `i64` to a float.
    #[inline]
    fn mul(self, rhs: i64) -> Self::Output {
        todo!()
    }
}

impl Mul<Float> for i64 {
    type Output = Float;

    /// Multiply a float to an infinite-precision `i64`.
    #[inline]
    fn mul(self, rhs: Float) -> Self::Output {
        todo!()
    }
}

impl Div<i64> for Float {
    type Output = Self;

    /// Divide an infinite-precision `i64` from a float.
    #[inline]
    fn div(self, rhs: i64) -> Self::Output {
        todo!()
    }
}

impl Div<Float> for i64 {
    type Output = Float;

    /// Divide a float from an infinite-precision `i64`.
    #[inline]
    fn div(self, rhs: Float) -> Self::Output {
        todo!()
    }
}

impl Add<Rational> for Float {
    type Output = Self;

    /// Add an infinite-precision rational to the float.
    #[inline]
    fn add(mut self, rhs: Rational) -> Self::Output {
        todo!()
    }
}

impl Sub<Rational> for Float {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Rational) -> Self::Output {
        self + -rhs
    }
}

impl Mul<Rational> for Float {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Rational) -> Self::Output {
        todo!() //(self.0 * rhs.to_multi_prec()).into()
    }
}

impl Div<Rational> for Float {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Rational) -> Self::Output {
        todo!() //(self.0 / rhs.to_multi_prec()).into()
    }
}

impl From<f64> for Float {
    fn from(value: f64) -> Self {
        Float::with_val(53, value)
    }
}

impl Float {
    pub fn new(prec: u32) -> Self {
        todo!()
    }

    pub fn with_val<T>(prec: u32, val: T) -> Self {
        todo!()
    }

    pub fn prec(&self) -> u32 {
        todo!()
    }

    pub fn set_prec(&mut self, prec: u32) {
        todo!()
    }

    pub fn is_finite(&self) -> bool {
        todo!()
    }

    pub fn is_negative(&self) -> bool {
        todo!()
    }

    /// Parse a float from a string.
    /// Precision can be specified by a trailing backtick followed by the precision.
    /// For example: ```1.234`20``` for a precision of 20 decimal digits.
    /// The precision is allowed to be a floating point number.
    ///  If `prec` is `None` and no precision is specified, the precision is derived from the string, with
    /// a minimum of 53 bits (`f64` precision).
    pub fn parse(s: &str, prec: Option<u32>) -> Result<Self, String> {
        todo!()
    }

    pub fn serialize(&self) -> Vec<u8> {
        todo!()
    }

    pub fn deserialize(d: &[u8], prec: u32) -> Float {
        todo!()
    }

    pub fn to_rational(&self) -> Rational {
        todo!()
    }

    pub fn into_inner(self) -> MultiPrecisionFloat {
        self.0
    }
}

impl From<MultiPrecisionFloat> for Float {
    fn from(value: MultiPrecisionFloat) -> Self {
        Float(value)
    }
}

impl NumericalFloatLike for Float {
    #[inline(always)]
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        self.clone() * a + b
    }

    #[inline(always)]
    fn neg(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn zero(&self) -> Self {
        Float::new(self.prec())
    }

    #[inline(always)]
    fn new_zero() -> Self {
        Float::new(1)
    }

    #[inline(always)]
    fn one(&self) -> Self {
        Float::with_val(self.prec(), 1.)
    }

    #[inline]
    fn pow(&self, e: u64) -> Self {
        todo!()
    }

    #[inline(always)]
    fn inv(&self) -> Self {
        todo!()
    }

    /// Convert from a `usize`. This may involve a loss of precision.
    #[inline(always)]
    fn from_usize(&self, a: usize) -> Self {
        Float::with_val(self.prec(), a)
    }

    /// Convert from a `i64`. This may involve a loss of precision.
    #[inline(always)]
    fn from_i64(&self, a: i64) -> Self {
        Float::with_val(self.prec(), a)
    }

    fn get_precision(&self) -> u32 {
        self.prec()
    }

    #[inline(always)]
    fn get_epsilon(&self) -> f64 {
        2.0f64.powi(-(self.prec() as i32))
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        false
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        let f: f64 = rng.gen();
        Float::with_val(self.prec(), f)
    }
}

impl SingleFloat for Float {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        todo!()
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        todo!()
    }

    #[inline(always)]
    fn is_finite(&self) -> bool {
        todo!()
    }

    #[inline(always)]
    fn from_rational(&self, rat: &Rational) -> Self {
        rat.to_multi_prec_float(self.prec())
    }
}

impl RealNumberLike for Float {
    fn to_usize_clamped(&self) -> usize {
        todo!()
    }

    fn to_f64(&self) -> f64 {
        todo!()
    }

    #[inline(always)]
    fn round_to_nearest_integer(&self) -> Integer {
        todo!()
    }
}

impl Real for Float {
    #[inline(always)]
    fn pi(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn e(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn euler(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn phi(&self) -> Self {
        (self.one() + self.from_i64(5).sqrt()) / 2
    }

    #[inline(always)]
    fn i(&self) -> Option<Self> {
        None
    }

    #[inline(always)]
    fn norm(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn sqrt(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn log(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn exp(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn sin(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn cos(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn tan(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn asin(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn acos(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn atan2(&self, x: &Self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn sinh(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn cosh(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn tanh(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn asinh(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn acosh(&self) -> Self {
        todo!()
    }

    #[inline(always)]
    fn atanh(&self) -> Self {
        todo!()
    }

    #[inline]
    fn powf(&self, e: &Self) -> Self {
        todo!()
    }
}

impl Rational {
    // Convert the rational number to a multi-precision float with precision `prec`.
    pub fn to_multi_prec_float(&self, prec: u32) -> Float {
        todo!()
    }
}

/// A float that does linear error propagation.
#[derive(Copy, Clone)]
pub struct ErrorPropagatingFloat<T: NumericalFloatLike> {
    value: T,
    abs_err: f64,
}

impl From<f64> for ErrorPropagatingFloat<f64> {
    fn from(value: f64) -> Self {
        if value == 0. {
            ErrorPropagatingFloat {
                value,
                abs_err: f64::EPSILON,
            }
        } else {
            ErrorPropagatingFloat {
                value,
                abs_err: f64::EPSILON * value.abs(),
            }
        }
    }
}

impl<T: NumericalFloatLike> Neg for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        ErrorPropagatingFloat {
            value: -self.value,
            abs_err: self.abs_err,
        }
    }
}

impl<T: RealNumberLike> Add<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        ErrorPropagatingFloat {
            abs_err: self.abs_err + rhs.abs_err,
            value: self.value + &rhs.value,
        }
    }
}

impl<T: RealNumberLike> Add<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<T: RealNumberLike> Sub<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        self - rhs.clone()
    }
}

impl<T: RealNumberLike> Sub<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        self + (-rhs)
    }
}

impl<T: RealNumberLike> Mul<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        let value = self.value.clone() * &rhs.value;
        let r = rhs.value.to_f64().abs();
        let s = self.value.to_f64().abs();

        if s == 0. && r == 0. {
            return ErrorPropagatingFloat {
                value,
                abs_err: self.abs_err * rhs.abs_err,
            };
        } else {
            ErrorPropagatingFloat {
                value,
                abs_err: self.abs_err * r + rhs.abs_err * s,
            }
        }
    }
}

impl<T: RealNumberLike + Add<Rational, Output = T>> Add<Rational> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Rational) -> Self::Output {
        ErrorPropagatingFloat {
            abs_err: self.abs_err,
            value: self.value + rhs,
        }
    }
}

impl<T: RealNumberLike + Add<Rational, Output = T>> Sub<Rational> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Rational) -> Self::Output {
        self + -rhs
    }
}

impl<T: RealNumberLike + Mul<Rational, Output = T>> Mul<Rational> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Rational) -> Self::Output {
        ErrorPropagatingFloat {
            abs_err: self.abs_err * rhs.to_f64().abs(),
            value: self.value * rhs,
        }
        .truncate()
    }
}

impl<T: RealNumberLike + Div<Rational, Output = T>> Div<Rational> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Rational) -> Self::Output {
        ErrorPropagatingFloat {
            abs_err: self.abs_err * rhs.inv().to_f64().abs(),
            value: self.value.clone() / rhs,
        }
        .truncate()
    }
}

impl<T: RealNumberLike> Mul<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<T: RealNumberLike> Div<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &Self) -> Self::Output {
        self * rhs.inv()
    }
}

impl<T: RealNumberLike> Div<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self / &rhs
    }
}

impl<T: RealNumberLike> AddAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() + rhs;
    }
}

impl<T: RealNumberLike> AddAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn add_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.add_assign(&rhs)
    }
}

impl<T: RealNumberLike> SubAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() - rhs;
    }
}

impl<T: RealNumberLike> SubAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.sub_assign(&rhs)
    }
}

impl<T: RealNumberLike> MulAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() * rhs;
    }
}

impl<T: RealNumberLike> MulAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.mul_assign(&rhs)
    }
}

impl<T: RealNumberLike> DivAssign<&ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn div_assign(&mut self, rhs: &ErrorPropagatingFloat<T>) {
        // TODO: optimize
        *self = self.clone() / rhs;
    }
}

impl<T: RealNumberLike> DivAssign<ErrorPropagatingFloat<T>> for ErrorPropagatingFloat<T> {
    #[inline]
    fn div_assign(&mut self, rhs: ErrorPropagatingFloat<T>) {
        self.div_assign(&rhs)
    }
}

impl<T: RealNumberLike> ErrorPropagatingFloat<T> {
    /// Create a new precision tracking float with a number of precise decimal digits `prec`.
    /// The `prec` must be smaller than the precision of the underlying float.
    ///
    /// If the value provided is 0, the precision argument is interpreted as an accuracy (
    /// the number of digits of the absolute error).
    pub fn new(value: T, prec: f64) -> Self {
        let r = value.to_f64().abs();

        if r == 0. {
            ErrorPropagatingFloat {
                abs_err: 10f64.powf(-prec),
                value,
            }
        } else {
            ErrorPropagatingFloat {
                abs_err: 10f64.powf(-prec) * r,
                value,
            }
        }
    }

    pub fn get_absolute_error(&self) -> f64 {
        self.abs_err
    }

    pub fn get_relative_error(&self) -> f64 {
        self.abs_err / self.value.to_f64().abs()
    }

    /// Get the precision in number of decimal digits.
    #[inline(always)]
    pub fn get_precision(&self) -> Option<f64> {
        let r = self.value.to_f64().abs();
        if r == 0. {
            return None;
        } else {
            Some(-(self.abs_err / r).log10())
        }
    }

    /// Get the accuracy in number of decimal digits.
    #[inline(always)]
    pub fn get_accuracy(&self) -> f64 {
        -self.abs_err.log10()
    }

    /// Truncate the precision to the maximal number of stable decimal digits
    /// of the underlying float.
    #[inline(always)]
    pub fn truncate(mut self) -> Self {
        if self.value.fixed_precision() {
            self.abs_err = self
                .abs_err
                .max(self.value.get_epsilon() * self.value.to_f64());
        }
        self
    }
}

impl<T: NumericalFloatLike> ErrorPropagatingFloat<T> {
    pub fn new_with_accuracy(value: T, acc: f64) -> Self {
        ErrorPropagatingFloat {
            value,
            abs_err: 10f64.powf(-acc),
        }
    }

    /// Get the number.
    #[inline(always)]
    pub fn get_num(&self) -> &T {
        &self.value
    }
}

impl<T: RealNumberLike> fmt::Display for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Some(p) = self.get_precision() {
            if p < 0. {
                return f.write_char('0');
            } else {
                f.write_fmt(format_args!("{0:.1$}", self.value, p as usize))
            }
        } else {
            f.write_char('0')
        }
    }
}

impl<T: RealNumberLike> Debug for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.value, f)?;

        if let Some(p) = self.get_precision() {
            f.write_fmt(format_args!("`{:.2}", p))
        } else {
            f.write_fmt(format_args!("``{:.2}", -self.abs_err.log10()))
        }
    }
}

impl<T: RealNumberLike> LowerExp for ErrorPropagatingFloat<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl<T: NumericalFloatLike> PartialEq for ErrorPropagatingFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        // TODO: ignore precision for partial equality?
        self.value == other.value
    }
}

impl<T: NumericalFloatLike + PartialOrd> PartialOrd for ErrorPropagatingFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<T: RealNumberLike> NumericalFloatLike for ErrorPropagatingFloat<T> {
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        self.clone() * a + b
    }

    fn neg(&self) -> Self {
        -self.clone()
    }

    fn zero(&self) -> Self {
        ErrorPropagatingFloat {
            value: self.value.zero(),
            abs_err: 2f64.powf(-(self.value.get_precision() as f64)),
        }
    }

    fn new_zero() -> Self {
        ErrorPropagatingFloat {
            value: T::new_zero(),
            abs_err: 2f64.powi(-53),
        }
    }

    fn one(&self) -> Self {
        ErrorPropagatingFloat {
            value: self.value.one(),
            abs_err: 2f64.powf(-(self.value.get_precision() as f64)),
        }
    }

    fn pow(&self, e: u64) -> Self {
        let i = self.to_f64().abs();

        if i == 0. {
            return ErrorPropagatingFloat {
                value: self.value.pow(e),
                abs_err: self.abs_err.powf(e as f64),
            };
        }

        let r = self.value.pow(e);
        ErrorPropagatingFloat {
            abs_err: self.abs_err * e as f64 * r.to_f64().abs() / i,
            value: r,
        }
    }

    fn inv(&self) -> Self {
        let r = self.value.inv();
        let rr = r.to_f64().abs();
        ErrorPropagatingFloat {
            abs_err: self.abs_err * rr * rr,
            value: r,
        }
    }

    /// Convert from a `usize`.
    fn from_usize(&self, a: usize) -> Self {
        let v = self.value.from_usize(a);
        let r = v.to_f64().abs();
        if r == 0. {
            ErrorPropagatingFloat {
                value: v,
                abs_err: 2f64.powf(-(self.value.get_precision() as f64)),
            }
        } else {
            ErrorPropagatingFloat {
                value: v,
                abs_err: 2f64.powf(-(self.value.get_precision() as f64)) * r,
            }
        }
    }

    /// Convert from a `i64`.
    fn from_i64(&self, a: i64) -> Self {
        let v = self.value.from_i64(a);
        let r = v.to_f64().abs();
        if r == 0. {
            ErrorPropagatingFloat {
                value: v,
                abs_err: 2f64.powf(-(self.value.get_precision() as f64)),
            }
        } else {
            ErrorPropagatingFloat {
                value: v,
                abs_err: 2f64.powf(-(self.value.get_precision() as f64)) * r,
            }
        }
    }

    fn get_precision(&self) -> u32 {
        // return the precision of the underlying float instead
        // of the current tracked precision
        self.value.get_precision()
    }

    fn get_epsilon(&self) -> f64 {
        2.0f64.powi(-(self.value.get_precision() as i32))
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        self.value.fixed_precision()
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        let v = self.value.sample_unit(rng);
        ErrorPropagatingFloat {
            abs_err: self.abs_err * v.to_f64().abs(),
            value: v,
        }
    }
}

impl<T: RealNumberLike> SingleFloat for ErrorPropagatingFloat<T> {
    fn is_zero(&self) -> bool {
        self.value.is_zero()
    }

    fn is_one(&self) -> bool {
        self.value.is_one()
    }

    fn is_finite(&self) -> bool {
        self.value.is_finite()
    }

    fn from_rational(&self, rat: &Rational) -> Self {
        if rat.is_zero() {
            ErrorPropagatingFloat {
                value: self.value.from_rational(rat),
                abs_err: self.abs_err,
            }
        } else {
            ErrorPropagatingFloat {
                value: self.value.from_rational(rat),
                abs_err: self.abs_err * rat.to_f64(),
            }
        }
    }
}

impl<T: RealNumberLike> RealNumberLike for ErrorPropagatingFloat<T> {
    fn to_usize_clamped(&self) -> usize {
        self.value.to_usize_clamped()
    }

    fn to_f64(&self) -> f64 {
        self.value.to_f64()
    }

    fn round_to_nearest_integer(&self) -> Integer {
        // TODO: what does this do with the error?
        self.value.round_to_nearest_integer()
    }
}

impl<T: Real + RealNumberLike> Real for ErrorPropagatingFloat<T> {
    fn pi(&self) -> Self {
        let v = self.value.pi();
        ErrorPropagatingFloat {
            abs_err: 2f64.powf(-(self.value.get_precision() as f64)) * v.to_f64(),
            value: v,
        }
    }

    fn e(&self) -> Self {
        let v = self.value.e();
        ErrorPropagatingFloat {
            abs_err: 2f64.powf(-(self.value.get_precision() as f64)) * v.to_f64(),
            value: v,
        }
    }

    fn euler(&self) -> Self {
        let v = self.value.euler();
        ErrorPropagatingFloat {
            abs_err: 2f64.powf(-(self.value.get_precision() as f64)) * v.to_f64(),
            value: v,
        }
    }

    fn phi(&self) -> Self {
        let v = self.value.phi();
        ErrorPropagatingFloat {
            abs_err: 2f64.powf(-(self.value.get_precision() as f64)) * v.to_f64(),
            value: v,
        }
    }

    #[inline(always)]
    fn i(&self) -> Option<Self> {
        Some(ErrorPropagatingFloat {
            value: self.value.i()?,
            abs_err: 2f64.powf(-(self.value.get_precision() as f64)),
        })
    }

    fn norm(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: self.abs_err,
            value: self.value.norm(),
        }
    }

    fn sqrt(&self) -> Self {
        let v = self.value.sqrt();
        let r = v.to_f64().abs();

        ErrorPropagatingFloat {
            abs_err: self.abs_err / (2. * r),
            value: v,
        }
        .truncate()
    }

    fn log(&self) -> Self {
        let r = self.value.log();
        ErrorPropagatingFloat {
            abs_err: self.abs_err / self.value.to_f64().abs(),
            value: r,
        }
        .truncate()
    }

    fn exp(&self) -> Self {
        let v = self.value.exp();
        ErrorPropagatingFloat {
            abs_err: v.to_f64().abs() * self.abs_err,
            value: v,
        }
        .truncate()
    }

    fn sin(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: self.abs_err * self.value.to_f64().cos().abs(),
            value: self.value.sin(),
        }
        .truncate()
    }

    fn cos(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: self.abs_err * self.value.to_f64().sin().abs(),
            value: self.value.cos(),
        }
        .truncate()
    }

    fn tan(&self) -> Self {
        let t = self.value.tan();
        let tt = t.to_f64().abs();

        ErrorPropagatingFloat {
            abs_err: self.abs_err * (1. + tt * tt),
            value: t,
        }
        .truncate()
    }

    fn asin(&self) -> Self {
        let v = self.value.to_f64();
        let t = self.value.asin();
        let tt = (1. - v * v).sqrt();
        ErrorPropagatingFloat {
            abs_err: self.abs_err / tt,
            value: t,
        }
        .truncate()
    }

    fn acos(&self) -> Self {
        let v = self.value.to_f64();
        let t = self.value.acos();
        let tt = (1. - v * v).sqrt();
        ErrorPropagatingFloat {
            abs_err: self.abs_err / tt,
            value: t,
        }
        .truncate()
    }

    fn atan2(&self, x: &Self) -> Self {
        let t = self.value.atan2(&x.value);
        let r = self.clone() / x;
        let r2 = r.value.to_f64().abs();

        let tt = 1. + r2 * r2;
        ErrorPropagatingFloat {
            abs_err: r.abs_err / tt,
            value: t,
        }
        .truncate()
    }

    fn sinh(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: self.abs_err * self.value.cosh().to_f64().abs(),
            value: self.value.sinh(),
        }
        .truncate()
    }

    fn cosh(&self) -> Self {
        ErrorPropagatingFloat {
            abs_err: self.abs_err * self.value.sinh().to_f64().abs(),
            value: self.value.cosh(),
        }
        .truncate()
    }

    fn tanh(&self) -> Self {
        let t = self.value.tanh();
        let tt = t.clone().to_f64().abs();
        ErrorPropagatingFloat {
            abs_err: self.abs_err * (1. - tt * tt),
            value: t,
        }
        .truncate()
    }

    fn asinh(&self) -> Self {
        let v = self.value.to_f64();
        let t = self.value.asinh();
        let tt = (1. + v * v).sqrt();
        ErrorPropagatingFloat {
            abs_err: self.abs_err / tt,
            value: t,
        }
        .truncate()
    }

    fn acosh(&self) -> Self {
        let v = self.value.to_f64();
        let t = self.value.acosh();
        let tt = (v * v - 1.).sqrt();
        ErrorPropagatingFloat {
            abs_err: self.abs_err / tt,
            value: t,
        }
        .truncate()
    }

    fn atanh(&self) -> Self {
        let v = self.value.to_f64();
        let t = self.value.atanh();
        let tt = 1. - v * v;
        ErrorPropagatingFloat {
            abs_err: self.abs_err / tt,
            value: t,
        }
        .truncate()
    }

    fn powf(&self, e: &Self) -> Self {
        let i = self.to_f64().abs();

        if i == 0. {
            return ErrorPropagatingFloat {
                value: self.value.powf(&e.value),
                abs_err: 0.,
            };
        }

        let r = self.value.powf(&e.value);
        ErrorPropagatingFloat {
            abs_err: (self.abs_err * e.value.to_f64() + i * e.abs_err * i.ln().abs())
                * r.to_f64().abs()
                / i,
            value: r,
        }
        .truncate()
    }
}

macro_rules! simd_impl {
    ($t:ty, $p:ident) => {
        impl NumericalFloatLike for $t {
            #[inline(always)]
            fn mul_add(&self, a: &Self, b: &Self) -> Self {
                *self * *a + b
            }

            #[inline(always)]
            fn neg(&self) -> Self {
                -self
            }

            #[inline(always)]
            fn zero(&self) -> Self {
                Self::ZERO
            }

            #[inline(always)]
            fn new_zero() -> Self {
                Self::ZERO
            }

            #[inline(always)]
            fn one(&self) -> Self {
                Self::ONE
            }

            #[inline]
            fn pow(&self, e: u64) -> Self {
                // FIXME: use binary exponentiation
                debug_assert!(e <= i32::MAX as u64);
                (*self).powf(e as f64)
            }

            #[inline(always)]
            fn inv(&self) -> Self {
                Self::ONE / *self
            }

            #[inline(always)]
            fn from_usize(&self, a: usize) -> Self {
                Self::from(a as f64)
            }

            #[inline(always)]
            fn from_i64(&self, a: i64) -> Self {
                Self::from(a as f64)
            }

            #[inline(always)]
            fn get_precision(&self) -> u32 {
                53
            }

            #[inline(always)]
            fn get_epsilon(&self) -> f64 {
                f64::EPSILON / 2.
            }

            #[inline(always)]
            fn fixed_precision(&self) -> bool {
                true
            }

            fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
                Self::from(rng.gen::<f64>())
            }
        }

        impl Real for $t {
            #[inline(always)]
            fn pi(&self) -> Self {
                std::f64::consts::PI.into()
            }

            #[inline(always)]
            fn e(&self) -> Self {
                std::f64::consts::E.into()
            }

            #[inline(always)]
            fn euler(&self) -> Self {
                0.57721566490153286.into()
            }

            #[inline(always)]
            fn phi(&self) -> Self {
                1.6180339887498948.into()
            }

            #[inline(always)]
            fn i(&self) -> Option<Self> {
                None
            }

            #[inline(always)]
            fn norm(&self) -> Self {
                (*self).abs()
            }

            #[inline(always)]
            fn sqrt(&self) -> Self {
                (*self).sqrt()
            }

            #[inline(always)]
            fn log(&self) -> Self {
                (*self).ln()
            }

            #[inline(always)]
            fn exp(&self) -> Self {
                (*self).exp()
            }

            #[inline(always)]
            fn sin(&self) -> Self {
                (*self).sin()
            }

            #[inline(always)]
            fn cos(&self) -> Self {
                (*self).cos()
            }

            #[inline(always)]
            fn tan(&self) -> Self {
                (*self).tan()
            }

            #[inline(always)]
            fn asin(&self) -> Self {
                (*self).asin()
            }

            #[inline(always)]
            fn acos(&self) -> Self {
                (*self).acos()
            }

            #[inline(always)]
            fn atan2(&self, x: &Self) -> Self {
                (*self).atan2(*x)
            }

            #[inline(always)]
            fn sinh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn cosh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn tanh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn asinh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn acosh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn atanh(&self) -> Self {
                unimplemented!("Hyperbolic geometric functions are not supported with SIMD");
            }

            #[inline(always)]
            fn powf(&self, e: &Self) -> Self {
                (*self).$p(*e)
            }
        }

        impl From<&Rational> for $t {
            fn from(value: &Rational) -> Self {
                value.to_f64().into()
            }
        }
    };
}

simd_impl!(f64x2, pow_f64x2);
simd_impl!(f64x4, pow_f64x4);

impl From<Float> for Rational {
    fn from(value: Float) -> Self {
        value.to_rational()
    }
}

impl LowerExp for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // lower-exp is ignored for integers
        f.write_fmt(format_args!("{}", self))
    }
}

impl NumericalFloatLike for Rational {
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        self * a + b
    }

    fn neg(&self) -> Self {
        self.neg()
    }

    fn zero(&self) -> Self {
        Self::zero()
    }

    fn new_zero() -> Self {
        Self::zero()
    }

    fn one(&self) -> Self {
        Self::one()
    }

    fn pow(&self, e: u64) -> Self {
        self.pow(e)
    }

    fn inv(&self) -> Self {
        self.inv()
    }

    fn from_usize(&self, a: usize) -> Self {
        a.into()
    }

    fn from_i64(&self, a: i64) -> Self {
        a.into()
    }

    #[inline(always)]
    fn get_precision(&self) -> u32 {
        u32::MAX
    }

    #[inline(always)]
    fn get_epsilon(&self) -> f64 {
        0.
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        true
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        let rng1 = rng.gen::<i64>();
        let rng2 = rng.gen::<i64>();

        if rng1 > rng2 {
            (rng2, rng1).into()
        } else {
            (rng1, rng2).into()
        }
    }
}

impl ConstructibleFloat for Rational {
    fn new_one() -> Self {
        Rational::one()
    }

    fn new_from_usize(a: usize) -> Self {
        (a, 1).into()
    }

    fn new_from_i64(a: i64) -> Self {
        (a, 1).into()
    }

    fn new_sample_unit<R: Rng + ?Sized>(rng: &mut R) -> Self {
        let rng1 = rng.gen::<i64>();
        let rng2 = rng.gen::<i64>();

        if rng1 > rng2 {
            (rng2, rng1).into()
        } else {
            (rng1, rng2).into()
        }
    }
}

impl SingleFloat for Rational {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.is_zero()
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.is_one()
    }

    #[inline(always)]
    fn is_finite(&self) -> bool {
        true
    }

    #[inline(always)]
    fn from_rational(&self, rat: &Rational) -> Self {
        rat.clone()
    }
}

impl RealNumberLike for Rational {
    fn to_usize_clamped(&self) -> usize {
        f64::from(self).to_usize_clamped()
    }

    fn to_f64(&self) -> f64 {
        f64::from(self)
    }

    #[inline(always)]
    fn round_to_nearest_integer(&self) -> Integer {
        self.round_to_nearest_integer()
    }
}

/// A complex number, `re + i * im`, where `i` is the imaginary unit.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Copy, Clone, PartialEq, Eq, Hash)]
#[repr(C)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

impl<T: Default> Default for Complex<T> {
    fn default() -> Self {
        Complex {
            re: T::default(),
            im: T::default(),
        }
    }
}

impl<T: InternalOrdering> InternalOrdering for Complex<T> {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.re
            .internal_cmp(&other.re)
            .then_with(|| self.im.internal_cmp(&other.im))
    }
}

impl<T: ConstructibleFloat + Real> ConstructibleFloat for Complex<T> {
    fn new_from_i64(a: i64) -> Self {
        Complex {
            re: T::new_from_i64(a),
            im: T::new_zero(),
        }
    }

    fn new_from_usize(a: usize) -> Self {
        Complex {
            re: T::new_from_usize(a),
            im: T::new_zero(),
        }
    }

    fn new_one() -> Self {
        Complex {
            re: T::new_one(),
            im: T::new_zero(),
        }
    }

    fn new_sample_unit<R: Rng + ?Sized>(rng: &mut R) -> Self {
        Complex {
            re: T::new_sample_unit(rng),
            im: T::new_zero(),
        }
    }
}

impl<T: NumericalFloatLike> Complex<T> {
    #[inline]
    pub fn new(re: T, im: T) -> Complex<T> {
        Complex { re, im }
    }

    #[inline]
    pub fn new_zero() -> Self
    where
        T: ConstructibleFloat,
    {
        Complex {
            re: T::new_zero(),
            im: T::new_zero(),
        }
    }

    #[inline]
    pub fn new_i() -> Self
    where
        T: ConstructibleFloat,
    {
        Complex {
            re: T::new_zero(),
            im: T::new_one(),
        }
    }

    #[inline]
    pub fn one(&self) -> Self {
        Complex {
            re: self.re.one(),
            im: self.im.zero(),
        }
    }

    #[inline]
    pub fn conj(&self) -> Self {
        Complex {
            re: self.re.clone(),
            im: -self.im.clone(),
        }
    }

    #[inline]
    pub fn zero(&self) -> Self {
        Complex {
            re: self.re.zero(),
            im: self.im.zero(),
        }
    }

    #[inline]
    pub fn i(&self) -> Complex<T> {
        Complex {
            re: self.re.zero(),
            im: self.im.one(),
        }
    }

    #[inline]
    pub fn norm_squared(&self) -> T {
        self.re.clone() * &self.re + self.im.clone() * &self.im
    }
}

impl<T: Real> Complex<T> {
    #[inline]
    pub fn arg(&self) -> T {
        self.im.atan2(&self.re)
    }

    #[inline]
    pub fn to_polar_coordinates(self) -> (T, T) {
        (self.norm_squared().sqrt(), self.arg())
    }

    #[inline]
    pub fn from_polar_coordinates(r: T, phi: T) -> Complex<T> {
        Complex::new(r.clone() * phi.cos(), r.clone() * phi.sin())
    }
}

impl<T: NumericalFloatLike> Add<Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<T: NumericalFloatLike> Add<T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Complex::new(self.re + rhs, self.im)
    }
}

impl<T: NumericalFloatLike> Add<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        Complex::new(self.re + &rhs.re, self.im + &rhs.im)
    }
}

impl<T: NumericalFloatLike> Add<&T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: &T) -> Self::Output {
        Complex::new(self.re + rhs, self.im)
    }
}

impl<'a, 'b, T: NumericalFloatLike> Add<&'a Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() + rhs
    }
}

impl<'a, T: NumericalFloatLike> Add<&T> for &'a Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: &T) -> Self::Output {
        self.clone() + rhs
    }
}

impl<'b, T: NumericalFloatLike> Add<Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: Complex<T>) -> Self::Output {
        self.clone() + rhs
    }
}

impl<'b, T: NumericalFloatLike> Add<T> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        self.clone() + rhs
    }
}

impl<T: NumericalFloatLike> AddAssign for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs)
    }
}

impl<T: NumericalFloatLike> AddAssign<T> for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.re += rhs;
    }
}

impl<T: NumericalFloatLike> AddAssign<&Complex<T>> for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        self.re += &rhs.re;
        self.im += &rhs.im;
    }
}

impl<T: NumericalFloatLike> AddAssign<&T> for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &T) {
        self.re += rhs;
    }
}

impl<T: NumericalFloatLike> Sub for Complex<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<T: NumericalFloatLike> Sub<T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Complex::new(self.re - rhs, self.im)
    }
}

impl<T: NumericalFloatLike> Sub<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        Complex::new(self.re - &rhs.re, self.im - &rhs.im)
    }
}

impl<T: NumericalFloatLike> Sub<&T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: &T) -> Self::Output {
        Complex::new(self.re - rhs, self.im)
    }
}

impl<'a, 'b, T: NumericalFloatLike> Sub<&'a Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() - rhs
    }
}

impl<'a, T: NumericalFloatLike> Sub<&T> for &'a Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: &T) -> Self::Output {
        self.clone() - rhs
    }
}

impl<'b, T: NumericalFloatLike> Sub<Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: Complex<T>) -> Self::Output {
        self.clone() - rhs
    }
}

impl<'b, T: NumericalFloatLike> Sub<T> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        self.clone() - rhs
    }
}

impl<T: NumericalFloatLike> SubAssign for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs)
    }
}

impl<T: NumericalFloatLike> SubAssign<T> for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        self.re -= rhs;
    }
}

impl<T: NumericalFloatLike> SubAssign<&Complex<T>> for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        self.re -= &rhs.re;
        self.im -= &rhs.im;
    }
}

impl<T: NumericalFloatLike> SubAssign<&T> for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &T) {
        self.re -= rhs;
    }
}

impl<T: NumericalFloatLike> Mul for Complex<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<T: NumericalFloatLike> Mul<T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Complex::new(self.re * &rhs, self.im * &rhs)
    }
}

impl<T: NumericalFloatLike> Mul<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        Complex::new(
            self.re.clone() * &rhs.re - self.im.clone() * &rhs.im,
            self.re.clone() * &rhs.im + self.im.clone() * &rhs.re,
        )
    }
}

impl<T: NumericalFloatLike> Mul<&T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: &T) -> Self::Output {
        Complex::new(self.re * rhs, self.im * rhs)
    }
}

impl<'a, 'b, T: NumericalFloatLike> Mul<&'a Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<'a, T: NumericalFloatLike> Mul<&T> for &'a Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: &T) -> Self::Output {
        self.clone() * rhs
    }
}

impl<'b, T: NumericalFloatLike> Mul<Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: Complex<T>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<'b, T: NumericalFloatLike> Mul<T> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        self.clone() * rhs
    }
}

impl<T: NumericalFloatLike> MulAssign for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone().mul(rhs);
    }
}

impl<T: NumericalFloatLike> MulAssign<T> for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = self.clone().mul(rhs);
    }
}

impl<T: NumericalFloatLike> MulAssign<&Complex<T>> for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        *self = self.clone().mul(rhs);
    }
}

impl<T: NumericalFloatLike> MulAssign<&T> for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &T) {
        *self = self.clone().mul(rhs);
    }
}

impl<T: NumericalFloatLike> Div for Complex<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl<T: NumericalFloatLike> Div<T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Complex::new(self.re / &rhs, self.im / &rhs)
    }
}

impl<T: NumericalFloatLike> Div<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &Self) -> Self::Output {
        let n = rhs.norm_squared();
        let re = self.re.clone() * &rhs.re + self.im.clone() * &rhs.im;
        let im = self.im.clone() * &rhs.re - self.re.clone() * &rhs.im;
        Complex::new(re / &n, im / &n)
    }
}

impl<T: NumericalFloatLike> Div<&T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &T) -> Self::Output {
        Complex::new(self.re / rhs, self.im / rhs)
    }
}

impl<'a, 'b, T: NumericalFloatLike> Div<&'a Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() / rhs
    }
}

impl<'a, T: NumericalFloatLike> Div<&T> for &'a Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &T) -> Self::Output {
        self.clone() / rhs
    }
}

impl<'b, T: NumericalFloatLike> Div<Complex<T>> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: Complex<T>) -> Self::Output {
        self.clone() / rhs
    }
}

impl<'b, T: NumericalFloatLike> Div<T> for &'b Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        self.clone() / rhs
    }
}

impl<T: NumericalFloatLike> DivAssign for Complex<T> {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone().div(rhs);
    }
}

impl<T: NumericalFloatLike> DivAssign<T> for Complex<T> {
    fn div_assign(&mut self, rhs: T) {
        *self = self.clone().div(rhs);
    }
}

impl<T: NumericalFloatLike> DivAssign<&Complex<T>> for Complex<T> {
    fn div_assign(&mut self, rhs: &Self) {
        *self = self.clone().div(rhs);
    }
}

impl<T: NumericalFloatLike> DivAssign<&T> for Complex<T> {
    fn div_assign(&mut self, rhs: &T) {
        *self = self.clone().div(rhs);
    }
}

impl<T: NumericalFloatLike> Neg for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn neg(self) -> Complex<T> {
        Complex::new(-self.re, -self.im)
    }
}

impl<T: NumericalFloatLike> Display for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('(')?;
        Display::fmt(&self.re, f)?;
        f.write_char('+')?;
        Display::fmt(&self.im, f)?;
        f.write_str("i)")
    }
}

impl<T: NumericalFloatLike> Debug for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('(')?;
        Debug::fmt(&self.re, f)?;
        f.write_char('+')?;
        Debug::fmt(&self.im, f)?;
        f.write_str("i)")
    }
}

impl<T: NumericalFloatLike> LowerExp for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('(')?;
        LowerExp::fmt(&self.re, f)?;
        f.write_char('+')?;
        LowerExp::fmt(&self.im, f)?;
        f.write_str("i)")
    }
}

impl<T: SingleFloat> SingleFloat for Complex<T> {
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.re.is_zero() && self.im.is_zero()
    }

    #[inline(always)]
    fn is_one(&self) -> bool {
        self.re.is_one() && self.im.is_zero()
    }

    #[inline(always)]
    fn is_finite(&self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }

    #[inline(always)]
    fn from_rational(&self, rat: &Rational) -> Self {
        Complex {
            re: self.re.from_rational(rat),
            im: self.im.zero(),
        }
    }
}

impl<T: NumericalFloatLike> NumericalFloatLike for Complex<T> {
    #[inline]
    fn mul_add(&self, a: &Self, b: &Self) -> Self {
        self.clone() * a + b
    }

    #[inline]
    fn neg(&self) -> Self {
        Complex {
            re: -self.re.clone(),
            im: -self.im.clone(),
        }
    }

    #[inline]
    fn zero(&self) -> Self {
        Complex {
            re: self.re.zero(),
            im: self.im.zero(),
        }
    }

    fn new_zero() -> Self {
        Complex {
            re: T::new_zero(),
            im: T::new_zero(),
        }
    }

    fn one(&self) -> Self {
        Complex {
            re: self.re.one(),
            im: self.im.zero(),
        }
    }

    fn pow(&self, e: u64) -> Self {
        // TODO: use binary exponentiation
        let mut r = self.one();
        for _ in 0..e {
            r *= self;
        }
        r
    }

    fn inv(&self) -> Self {
        let n = self.norm_squared();
        Complex::new(self.re.clone() / &n, -self.im.clone() / &n)
    }

    fn from_usize(&self, a: usize) -> Self {
        Complex {
            re: self.re.from_usize(a),
            im: self.im.zero(),
        }
    }

    fn from_i64(&self, a: i64) -> Self {
        Complex {
            re: self.re.from_i64(a),
            im: self.im.zero(),
        }
    }

    #[inline(always)]
    fn get_precision(&self) -> u32 {
        self.re.get_precision().min(self.im.get_precision())
    }

    #[inline(always)]
    fn get_epsilon(&self) -> f64 {
        (2.0f64).powi(-(self.get_precision() as i32))
    }

    #[inline(always)]
    fn fixed_precision(&self) -> bool {
        self.re.fixed_precision() || self.im.fixed_precision()
    }

    fn sample_unit<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {
        Complex {
            re: self.re.sample_unit(rng),
            im: self.im.zero(),
        }
    }
}

/// Following the same conventions and formulas as num::Complex.
impl<T: Real> Real for Complex<T> {
    #[inline]
    fn pi(&self) -> Self {
        Complex::new(self.re.pi(), self.im.zero())
    }

    #[inline]
    fn e(&self) -> Self {
        Complex::new(self.re.e(), self.im.zero())
    }

    #[inline]
    fn euler(&self) -> Self {
        Complex::new(self.re.euler(), self.im.zero())
    }

    #[inline]
    fn phi(&self) -> Self {
        Complex::new(self.re.phi(), self.im.zero())
    }

    #[inline(always)]
    fn i(&self) -> Option<Self> {
        Some(self.i())
    }

    #[inline]
    fn norm(&self) -> Self {
        Complex::new(self.norm_squared().sqrt(), self.im.zero())
    }

    #[inline]
    fn sqrt(&self) -> Self {
        let (r, phi) = self.clone().to_polar_coordinates();
        Complex::from_polar_coordinates(r.sqrt(), phi / self.re.from_usize(2))
    }

    #[inline]
    fn log(&self) -> Self {
        Complex::new(self.norm().re.log(), self.arg())
    }

    #[inline]
    fn exp(&self) -> Self {
        let r = self.re.exp();
        Complex::new(r.clone() * self.im.cos(), r * self.im.sin())
    }

    #[inline]
    fn sin(&self) -> Self {
        Complex::new(
            self.re.sin() * self.im.cosh(),
            self.re.cos() * self.im.sinh(),
        )
    }

    #[inline]
    fn cos(&self) -> Self {
        Complex::new(
            self.re.cos() * self.im.cosh(),
            -self.re.sin() * self.im.sinh(),
        )
    }

    #[inline]
    fn tan(&self) -> Self {
        let (r, i) = (self.re.clone() + &self.re, self.im.clone() + &self.im);
        let m = r.cos() + i.cosh();
        Self::new(r.sin() / &m, i.sinh() / m)
    }

    #[inline]
    fn asin(&self) -> Self {
        let i = self.i();
        -i.clone() * ((self.one() - self.clone() * self).sqrt() + i * self).log()
    }

    #[inline]
    fn acos(&self) -> Self {
        let i = self.i();
        -i.clone() * (i * (self.one() - self.clone() * self).sqrt() + self).log()
    }

    #[inline]
    fn atan2(&self, x: &Self) -> Self {
        // TODO: pick proper branch
        let r = self.clone() / x;
        let i = self.i();
        let one = self.one();
        let two = one.clone() + &one;
        // TODO: add edge cases
        ((&one + &i * &r).log() - (&one - &i * r).log()) / (two * i)
    }

    #[inline]
    fn sinh(&self) -> Self {
        Complex::new(
            self.re.sinh() * self.im.cos(),
            self.re.cosh() * self.im.sin(),
        )
    }

    #[inline]
    fn cosh(&self) -> Self {
        Complex::new(
            self.re.cosh() * self.im.cos(),
            self.re.sinh() * self.im.sin(),
        )
    }

    #[inline]
    fn tanh(&self) -> Self {
        let (two_re, two_im) = (self.re.clone() + &self.re, self.im.clone() + &self.im);
        let m = two_re.cosh() + two_im.cos();
        Self::new(two_re.sinh() / &m, two_im.sin() / m)
    }

    #[inline]
    fn asinh(&self) -> Self {
        let one = self.one();
        (self.clone() + (one + self.clone() * self).sqrt()).log()
    }

    #[inline]
    fn acosh(&self) -> Self {
        let one = self.one();
        let two = one.clone() + &one;
        &two * (((self.clone() + &one) / &two).sqrt() + ((self.clone() - one) / &two).sqrt()).log()
    }

    #[inline]
    fn atanh(&self) -> Self {
        let one = self.one();
        let two = one.clone() + &one;
        // TODO: add edge cases
        ((&one + self).log() - (one - self).log()) / two
    }

    #[inline]
    fn powf(&self, e: &Self) -> Self {
        if e.re == self.re.zero() && e.im == self.im.zero() {
            self.one()
        } else if e.im == self.im.zero() {
            let (r, phi) = self.clone().to_polar_coordinates();
            Self::from_polar_coordinates(r.powf(&e.re), phi * e.re.clone())
        } else {
            (e * self.log()).exp()
        }
    }
}

impl<T: NumericalFloatLike> From<T> for Complex<T> {
    fn from(value: T) -> Self {
        let zero = value.zero();
        Complex::new(value, zero)
    }
}

impl<'a, T: NumericalFloatLike + From<&'a Rational>> From<&'a Rational> for Complex<T> {
    fn from(value: &'a Rational) -> Self {
        let c: T = value.into();
        let zero = c.zero();
        Complex::new(c, zero)
    }
}

#[cfg(test)]
mod test {

    use super::*;

    #[test]
    fn double() {
        let a = 5.;
        let b = 7.;

        let r = a.sqrt() + b.log() + b.sin() - a.cos() + b.tan() - 0.3.asin() + 0.5.acos()
            - a.atan2(b)
            + b.sinh()
            - a.cosh()
            + b.tanh()
            - 0.7.asinh()
            + b.acosh() / 0.4.atanh()
            + b.powf(a);
        assert_eq!(r, 17293.219725825093);
    }

    #[test]
    fn error_propagation() {
        let a = ErrorPropagatingFloat::new(5., 16.);
        let b = ErrorPropagatingFloat::new(7., 16.);
        let c = ErrorPropagatingFloat::new(0.3, 16.);
        let d = ErrorPropagatingFloat::new(0.5, 16.);
        let e = ErrorPropagatingFloat::new(0.7, 16.);
        let f = ErrorPropagatingFloat::new(0.4, 16.);

        let r = a.sqrt() + b.log() + b.sin() - a.cos() + b.tan() - c.asin() + d.acos()
            - a.atan2(&b)
            + b.sinh()
            - a.cosh()
            + b.tanh()
            - e.asinh()
            + b.acosh() / f.atanh()
            + b.powf(&a);
        assert_eq!(r.value, 17293.219725825093);
        // error is 14.836811363436391 when the f64 could have theoretically grown in between
        assert_eq!(r.get_precision(), Some(14.836795991431746));
    }

    #[test]
    fn error_truncation() {
        let a = ErrorPropagatingFloat::new(0.0000000123456789, 9.)
            .exp()
            .log();
        assert_eq!(a.get_precision(), Some(8.046104745509947));
    }

    #[test]
    fn large_cancellation() {
        let a = ErrorPropagatingFloat::new(Float::with_val(200, 1e-50), 60.);
        let r = (a.exp() - a.one()) / a;
        assert_eq!(format!("{}", r), "1.000000000");
        assert_eq!(r.get_precision(), Some(10.205999132796238));
    }

    #[test]
    fn complex() {
        let a = Complex::new(1., 2.);
        let b: Complex<f64> = Complex::new(3., 4.);

        let r = &a.sqrt() + &b.log() - a.exp() + b.sin() - a.cos() + b.tan() - a.asin() + b.acos()
            - a.atan2(&b)
            + b.sinh()
            - a.cosh()
            + b.tanh()
            - a.asinh()
            + &b.acosh() / a.atanh()
            + b.powf(&a);
        assert_eq!(r, Complex::new(0.1924131450685842, -39.83285329561913));
    }

    #[test]
    fn float_int() {
        let a = Float::with_val(53, 0.123456789123456);
        let b = a / 10i64 * 1300;
        assert_eq!(b.get_precision(), 53);

        let a = Float::with_val(53, 12345.6789);
        let b = a - 12345;
        assert_eq!(b.get_precision(), 40);
    }

    #[test]
    fn float_cancellation() {
        let a = Float::with_val(10, 1000);
        let b = a + 10i64;
        assert_eq!(b.get_precision(), 11);

        let a = Float::with_val(53, -1001);
        let b = a + 1000i64;
        assert_eq!(b.get_precision(), 45); // tight bound is 44 digits

        let a = Float::with_val(53, 1000);
        let b = Float::with_val(100, -1001);
        let c = a + b;
        assert_eq!(c.get_precision(), 45); // tight bound is 44 digits

        let a = Float::with_val(20, 1000);
        let b = Float::with_val(40, 1001);
        let c = a + b;
        assert_eq!(c.get_precision(), 22);

        let a = Float::with_val(4, 18.0);
        let b = Float::with_val(24, -17.9199009);
        let c = a + b;
        assert_eq!(c.get_precision(), 1); // capped at 1

        let a = Float::with_val(24, 18.00000);
        let b = Float::with_val(24, -17.992);
        let c = a + b;
        assert_eq!(c.get_precision(), 14);
    }

    #[test]
    fn float_growth() {
        let a = Float::with_val(53, 0.01);
        let b = a.exp();
        assert_eq!(b.get_precision(), 60);

        let a = Float::with_val(53, 0.8);
        let b = a.exp();
        assert_eq!(b.get_precision(), 54);

        let a = Float::with_val(53, 200);
        let b = a.exp();
        assert_eq!(b.get_precision(), 46);

        let a = Float::with_val(53, 0.8);
        let b = a.log();
        assert_eq!(b.get_precision(), 53);

        let a = Float::with_val(53, 300.0);
        let b = a.log();
        assert_eq!(b.get_precision(), 57);

        let a = Float::with_val(53, 1.5709);
        let b = a.sin();
        assert_eq!(b.get_precision(), 53);

        let a = Float::with_val(53, 14.);
        let b = a.tanh();
        assert_eq!(b.get_precision(), 66);

        let a = Float::with_val(53, 1.);
        let b = Float::with_val(53, 0.1);
        let b = a.powf(&b);
        assert_eq!(b.get_precision(), 57);

        let a = Float::with_val(53, 1.);
        let b = Float::with_val(200, 0.1);
        let b = a.powf(&b);
        assert_eq!(b.get_precision(), 57);
    }

    #[test]
    fn powf_prec() {
        let a = Float::with_val(53, 10.);
        let b = Float::with_val(200, 0.1);
        let c = a.powf(&b);
        assert_eq!(c.get_precision(), 57);

        let a = Float::with_val(200, 2.);
        let b = Float::with_val(53, 0.1);
        let c = a.powf(&b);
        assert_eq!(c.get_precision(), 58);

        let a = Float::with_val(53, 3.);
        let b = Float::with_val(200, 20.);
        let c = a.powf(&b);
        assert_eq!(c.get_precision(), 49);

        let a = Float::with_val(200, 1.);
        let b = Float::with_val(53, 0.1);
        let c = a.powf(&b);
        assert_eq!(c.get_precision(), 57); // a=1 is anomalous

        let a = Float::with_val(200, 0.4);
        let b = Float::with_val(53, 0.1);
        let c = a.powf(&b);
        assert_eq!(c.get_precision(), 57);
    }

    #[cfg(feature = "bincode")]
    #[test]
    fn bincode_export() {
        let a = Float::with_val(15, 1.127361273);
        let encoded = bincode::encode_to_vec(&a, bincode::config::standard()).unwrap();
        let b: Float = bincode::decode_from_slice(&encoded, bincode::config::standard())
            .unwrap()
            .0;
        assert_eq!(a, b);
    }
}
