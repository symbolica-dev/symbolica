//! Floating-point numbers and traits.

use std::{
    fmt::{Debug, Display, LowerExp},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Sub, SubAssign},
};

use rand::Rng;

use super::{integer::Integer, rational::Rational};

mod complex;
mod double;
mod error;
mod field;
mod multiprecision;
mod native;
#[cfg(feature = "python")]
mod python;
mod rational;
mod simd;

#[cfg(test)]
mod tests;

pub use complex::Complex;
pub use double::DoubleFloat;
pub use error::ErrorPropagatingFloat;
pub use field::FloatField;
pub use multiprecision::Float;
pub use native::F64;
#[cfg(feature = "python")]
pub use python::PythonMultiPrecisionFloat;

pub trait FloatLike:
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
    /// Set this value from another value. May reuse memory.
    fn set_from(&mut self, other: &Self);

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

    /// Return true if the number is exactly equal to zero (in all components).
    fn is_fully_zero(&self) -> bool;
}

/// A number that behaves like a single number (excluding simd-like types).
pub trait SingleFloat: FloatLike {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn is_finite(&self) -> bool;
    /// Convert a rational to a float with the same precision as the current float.
    fn from_rational(&self, rat: &Rational) -> Self;
}

/// A number that can be converted to a `usize`, `f64`, or rounded to the nearest integer (excluding complex numbers).
pub trait RealLike: SingleFloat {
    fn to_usize_clamped(&self) -> usize;
    fn to_f64(&self) -> f64;
    fn round_to_nearest_integer(&self) -> Integer;
}

/// A float that can be constructed without any parameters, such as `f64` (excluding multi-precision floats).
pub trait Constructible: FloatLike {
    fn new_one() -> Self;
    fn new_from_usize(a: usize) -> Self;
    fn new_from_i64(a: i64) -> Self;
    /// Sample a point on the interval [0, 1].
    fn new_sample_unit<R: Rng + ?Sized>(rng: &mut R) -> Self;
}

/// A float that has a fixed finite precision, such as `f64` (excluding multi-precision floats).
pub trait FixedPrecision {
    /// The number of binary digits in the mantissa.
    const BINARY_PRECISION: usize;
    /// The (rounded) number of decimal digits in the mantissa.
    const DECIMAL_PRECISION: usize = Self::BINARY_PRECISION
        .saturating_mul(30103)
        .saturating_add(99999)
        / 100000;
}

/// A number that behaves like a real number, with constants like π and e
/// and functions like sine and cosine.
///
/// It may also have a notion of an imaginary unit.
pub trait Real: FloatLike {
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

    fn conj(&self) -> Self;
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
