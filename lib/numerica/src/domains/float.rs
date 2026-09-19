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
mod interval;
mod multiprecision;
mod native;
#[cfg(feature = "python")]
mod python;
mod rational;
mod simd;

#[cfg(test)]
mod complex_tests;
#[cfg(test)]
mod tests;

pub use super::backend::float::RoundingDirection;
pub use complex::Complex;
pub use double::DoubleFloat;
pub use error::ErrorPropagatingFloat;
pub use field::{FloatComparisonError, FloatField};
pub use interval::{ComplexBall, RealBall};
pub use multiprecision::Float;
pub use native::F64;
#[cfg(feature = "python")]
pub use python::{
    PythonComplexFloat, PythonFloat, PythonMultiPrecisionComplex, PythonMultiPrecisionFloat,
    register_python_floats,
};

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
    /// Compare ordered scalar values. Non-scalar types may return `None`.
    #[inline]
    fn real_cmp(&self, _other: &Self) -> Option<std::cmp::Ordering> {
        None
    }

    /// Classify a scalar float for exceptional-value handling.
    /// Return `None` for domains without a scalar floating-point classification,
    /// such as complex numbers, SIMD vectors, or exact types.
    #[inline]
    fn real_classify(&self) -> Option<std::num::FpCategory> {
        None
    }

    /// Request scaled arithmetic for zero, subnormal or non-finite intermediates.
    /// The default opts out of scalar range guards (for example for exact types
    /// or types with multiple independently scaled components).
    #[inline]
    fn needs_rescaling(&self) -> bool {
        false
    }

    /// Set this value from another value. May reuse memory.
    fn set_from(&mut self, other: &Self);

    /// Perform `(self * a) + b`.
    fn mul_add(&self, a: &Self, b: &Self) -> Self;
    fn neg(&self) -> Self;
    fn zero(&self) -> Self;
    /// Construct a NaN, preserving this value's precision and component shape.
    /// Returns `None` for exact types, such as rationals, that cannot represent NaN.
    fn nan(&self) -> Option<Self> {
        None
    }
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
    /// Magnitude of a pair, avoiding unnecessary overflow and underflow.
    #[inline]
    fn hypot(&self, other: &Self) -> Self {
        let (mut a, mut b) = (self.norm(), other.norm());
        match a.real_cmp(&b) {
            Some(std::cmp::Ordering::Less) => std::mem::swap(&mut a, &mut b),
            Some(_) => {}
            None => return (self.clone() * self + other.clone() * other).sqrt(),
        }
        if b.is_fully_zero() {
            return a;
        }
        let r = b / &a;
        a * (r.one() + r.clone() * r).sqrt()
    }

    /// Absolute value with the sign of `sign`, including signed zero where supported.
    #[inline]
    fn copy_sign(&self, sign: &Self) -> Self {
        if sign.real_cmp(&sign.zero()) == Some(std::cmp::Ordering::Less) {
            -self.norm()
        } else {
            self.norm()
        }
    }

    /// Compute log(1 + self), retaining small increments lost when adding one.
    #[inline]
    fn log1p(&self) -> Self {
        if self.is_fully_zero() {
            return self.clone();
        }
        if self.needs_rescaling() && self.real_cmp(&self.one()) == Some(std::cmp::Ordering::Greater)
        {
            return self.log();
        }
        // log(1+x) = 2 asinh(x / (2 sqrt(1+x))). This also avoids
        // cancellation-induced precision loss in dynamically sized floats.
        let two = self.from_usize(2);
        (self.clone() / (self.one() + self).sqrt() / &two).asinh() * two
    }

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
    /// Reciprocal hyperbolic cosine, without overflowing an intermediate cosh.
    #[inline]
    fn sech(&self) -> Self {
        // Split the exponential so that rounding exp(-|x|) to zero does not
        // discard a representable subnormal value of 2 exp(-|x|).
        let e = (-self.norm() / self.from_usize(2)).exp();
        let e2 = e.clone() * &e;
        (e.clone() + &e) * e / (e2.one() + e2.clone() * e2)
    }

    /// Reciprocal hyperbolic sine, retaining accuracy near zero and at infinity.
    #[inline]
    fn csch(&self) -> Self {
        self.sech() / self.tanh()
    }

    fn asinh(&self) -> Self;
    fn acosh(&self) -> Self;
    fn atanh(&self) -> Self;
    fn powf(&self, e: &Self) -> Self;
}
