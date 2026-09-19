use std::{
    cmp::Ordering,
    fmt::{Debug, Display, LowerExp, Write},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use rand::Rng;

use super::{Constructible, FixedPrecision, FloatLike, Real, SingleFloat};
use super::{DoubleFloat, Float, RealLike};
use crate::domains::{InternalOrdering, integer::Integer, rational::Rational};

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

impl<T> From<(T, T)> for Complex<T> {
    fn from((re, im): (T, T)) -> Self {
        Complex { re, im }
    }
}

impl<T: Constructible> Constructible for Complex<T> {
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
            im: T::new_sample_unit(rng),
        }
    }
}

impl<T> Complex<T> {
    #[inline]
    pub const fn new(re: T, im: T) -> Complex<T> {
        Complex { re, im }
    }
}

impl<T: FloatLike> Complex<T> {
    #[inline]
    pub fn new_zero() -> Self
    where
        T: Constructible,
    {
        Complex {
            re: T::new_zero(),
            im: T::new_zero(),
        }
    }

    #[inline]
    pub fn new_i() -> Self
    where
        T: Constructible,
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
    // Used only after an unscaled exponential/hyperbolic factor overflowed.
    #[cold]
    fn exp_product(exponent: &T, factor: T, half: bool) -> T {
        if factor.is_fully_zero() {
            return factor;
        }
        let two = exponent.from_usize(2);
        let e = (exponent.clone() / &two).exp();
        if !e.needs_rescaling() {
            let other = if half { e.clone() / two } else { e.clone() };
            return (factor * e) * other;
        }
        let mut log = exponent.clone() + factor.norm().log();
        if half {
            log -= two.log();
        }
        log.exp().copy_sign(&factor)
    }

    #[inline]
    fn hyperbolic(&self, cosine: bool) -> Self {
        let sh = self.re.sinh();
        let ch = self.re.cosh();
        let s = self.im.sin();
        let c = self.im.cos();
        if !ch.needs_rescaling() || self.re.real_cmp(&self.re.zero()).is_none() {
            return if cosine {
                Self::new(ch * c, sh * s)
            } else {
                Self::new(sh * c, ch * s)
            };
        }
        let exponent = self.re.norm();
        let sign = self.re.one().copy_sign(&self.re);
        if cosine {
            Self::new(
                Self::exp_product(&exponent, c, true),
                Self::exp_product(&exponent, s, true) * sign,
            )
        } else {
            Self::new(
                Self::exp_product(&exponent, c, true) * sign,
                Self::exp_product(&exponent, s, true),
            )
        }
    }

    #[inline]
    fn large_inverse_argument(&self) -> bool {
        let limit = self
            .re
            .from_usize(2)
            .pow((self.get_precision() as u64 + 1) / 2 + 2);
        self.re.norm().real_cmp(&limit) == Some(Ordering::Greater)
            || self.im.norm().real_cmp(&limit) == Some(Ordering::Greater)
    }

    #[inline]
    pub fn arg(&self) -> T {
        self.im.atan2(&self.re)
    }

    #[inline]
    pub fn to_polar_coordinates(self) -> (T, T) {
        (self.re.hypot(&self.im), self.arg())
    }

    #[inline]
    pub fn from_polar_coordinates(r: T, phi: T) -> Complex<T> {
        let c = phi.cos();
        let s = phi.sin();
        if r.needs_rescaling() && r.real_cmp(&r).is_some() {
            let product = |factor: T| {
                if factor.is_fully_zero() {
                    r.one().copy_sign(&r) * factor
                } else {
                    r.clone() * factor
                }
            };
            Self::new(product(c), product(s))
        } else {
            Self::new(r.clone() * c, r * s)
        }
    }
}

impl<T: SingleFloat> Complex<T> {
    pub fn is_real(&self) -> bool {
        self.im.is_zero()
    }

    #[inline]
    pub fn to_real(&self) -> Option<&T> {
        if self.im.is_zero() {
            Some(&self.re)
        } else {
            None
        }
    }
}

impl<T: FloatLike> Add<Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Complex::new(self.re + rhs.re, self.im + rhs.im)
    }
}

impl<T: FloatLike> Add<T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        Complex::new(self.re + rhs, self.im)
    }
}

impl<T: FloatLike> Add<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: &Self) -> Self::Output {
        Complex::new(self.re + &rhs.re, self.im + &rhs.im)
    }
}

impl<T: FloatLike> Add<&T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: &T) -> Self::Output {
        Complex::new(self.re + rhs, self.im)
    }
}

impl<'a, T: FloatLike> Add<&'a Complex<T>> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() + rhs
    }
}

impl<T: FloatLike> Add<&T> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: &T) -> Self::Output {
        self.clone() + rhs
    }
}

impl<T: FloatLike> Add<Complex<T>> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: Complex<T>) -> Self::Output {
        self.clone() + rhs
    }
}

impl<T: FloatLike> Add<T> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        self.clone() + rhs
    }
}

impl<T: FloatLike> AddAssign for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        self.add_assign(&rhs)
    }
}

impl<T: FloatLike> AddAssign<T> for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: T) {
        self.re += rhs;
    }
}

impl<T: FloatLike> AddAssign<&Complex<T>> for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &Self) {
        self.re += &rhs.re;
        self.im += &rhs.im;
    }
}

impl<T: FloatLike> AddAssign<&T> for Complex<T> {
    #[inline]
    fn add_assign(&mut self, rhs: &T) {
        self.re += rhs;
    }
}

impl<T: FloatLike> Sub for Complex<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Complex::new(self.re - rhs.re, self.im - rhs.im)
    }
}

impl<T: FloatLike> Sub<T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        Complex::new(self.re - rhs, self.im)
    }
}

impl<T: FloatLike> Sub<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: &Self) -> Self::Output {
        Complex::new(self.re - &rhs.re, self.im - &rhs.im)
    }
}

impl<T: FloatLike> Sub<&T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: &T) -> Self::Output {
        Complex::new(self.re - rhs, self.im)
    }
}

impl<'a, T: FloatLike> Sub<&'a Complex<T>> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() - rhs
    }
}

impl<T: FloatLike> Sub<&T> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: &T) -> Self::Output {
        self.clone() - rhs
    }
}

impl<T: FloatLike> Sub<Complex<T>> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: Complex<T>) -> Self::Output {
        self.clone() - rhs
    }
}

impl<T: FloatLike> Sub<T> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn sub(self, rhs: T) -> Self::Output {
        self.clone() - rhs
    }
}

impl<T: FloatLike> SubAssign for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        self.sub_assign(&rhs)
    }
}

impl<T: FloatLike> SubAssign<T> for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: T) {
        self.re -= rhs;
    }
}

impl<T: FloatLike> SubAssign<&Complex<T>> for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &Self) {
        self.re -= &rhs.re;
        self.im -= &rhs.im;
    }
}

impl<T: FloatLike> SubAssign<&T> for Complex<T> {
    #[inline]
    fn sub_assign(&mut self, rhs: &T) {
        self.re -= rhs;
    }
}

impl<T: FloatLike> Mul for Complex<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

impl<T: FloatLike> Mul<T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        Complex::new(self.re * &rhs, self.im * &rhs)
    }
}

impl<T: FloatLike> Mul<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: &Self) -> Self::Output {
        Complex::new(
            self.re.clone() * &rhs.re - self.im.clone() * &rhs.im,
            self.re.clone() * &rhs.im + self.im.clone() * &rhs.re,
        )
    }
}

impl<T: FloatLike> Mul<&T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: &T) -> Self::Output {
        Complex::new(self.re * rhs, self.im * rhs)
    }
}

impl<'a, T: FloatLike> Mul<&'a Complex<T>> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<T: FloatLike> Mul<&T> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: &T) -> Self::Output {
        self.clone() * rhs
    }
}

impl<T: FloatLike> Mul<Complex<T>> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: Complex<T>) -> Self::Output {
        self.clone() * rhs
    }
}

impl<T: FloatLike> Mul<T> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn mul(self, rhs: T) -> Self::Output {
        self.clone() * rhs
    }
}

impl<T: FloatLike> MulAssign for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = self.clone().mul(rhs);
    }
}

impl<T: FloatLike> MulAssign<T> for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: T) {
        *self = self.clone().mul(rhs);
    }
}

impl<T: FloatLike> MulAssign<&Complex<T>> for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &Self) {
        *self = self.clone().mul(rhs);
    }
}

impl<T: FloatLike> MulAssign<&T> for Complex<T> {
    #[inline]
    fn mul_assign(&mut self, rhs: &T) {
        *self = self.clone().mul(rhs);
    }
}

impl<T: FloatLike> Div for Complex<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl<T: FloatLike> Div<T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        Complex::new(self.re / &rhs, self.im / &rhs)
    }
}

impl<T: FloatLike> Div<&Complex<T>> for Complex<T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: &Self) -> Self::Output {
        let n = rhs.norm_squared();
        let re = self.re.clone() * &rhs.re + self.im.clone() * &rhs.im;
        let im = self.im.clone() * &rhs.re - self.re.clone() * &rhs.im;
        if !n.needs_rescaling()
            && (!re.needs_rescaling() || re.is_fully_zero())
            && (!im.needs_rescaling() || im.is_fully_zero())
        {
            return Complex::new(re / &n, im / &n);
        }

        if rhs.im.is_fully_zero() {
            return Self::new(self.re / &rhs.re, self.im / &rhs.re);
        }
        if rhs.re.is_fully_zero() {
            return Self::new(self.im / &rhs.im, -self.re / &rhs.im);
        }

        // Keep the usual two-division path for ordinary inputs. Only rescale
        // when a squared norm or numerator overflowed or became subnormal.
        let abs = |x: &T| {
            if x.real_cmp(&x.zero()) == Some(Ordering::Less) {
                -x.clone()
            } else {
                x.clone()
            }
        };
        let (a, b) = (abs(&rhs.re), abs(&rhs.im));
        let scale = match a.real_cmp(&b) {
            Some(Ordering::Less) => b,
            Some(_) => a,
            None => return Complex::new(re / &n, im / &n),
        };
        if scale.is_fully_zero() {
            return Complex::new(re / &n, im / &n);
        }
        let (c, d) = (rhs.re.clone() / &scale, rhs.im.clone() / &scale);
        let denominator = c.clone() * &c + d.clone() * &d;
        let (a, b) = (abs(&self.re), abs(&self.im));
        let numerator_scale = if a.real_cmp(&b) == Some(Ordering::Less) {
            b
        } else {
            a
        };
        if numerator_scale.is_fully_zero() {
            return self;
        }
        let (a, b) = (self.re / &numerator_scale, self.im / &numerator_scale);
        let ratio = numerator_scale.clone() / &scale;
        let rescale = |q: T| {
            if !ratio.needs_rescaling() {
                q * &ratio
            } else {
                let product = q.clone() * &numerator_scale;
                if !product.needs_rescaling() || q.is_fully_zero() {
                    product / &scale
                } else {
                    (q / &scale) * &numerator_scale
                }
            }
        };
        let mut result = Complex::new(
            rescale((a.clone() * &c + b.clone() * &d) / &denominator),
            rescale((b * c - a * d) / &denominator),
        );
        // Keep a component that was already safe: normalizing a very small
        // component alongside a large one can otherwise discard its contribution.
        if !n.needs_rescaling() {
            if !re.needs_rescaling() || re.is_fully_zero() {
                result.re = re / &n;
            }
            if !im.needs_rescaling() || im.is_fully_zero() {
                result.im = im / n;
            }
        } else {
            if !re.needs_rescaling() {
                result.re = re / &scale / &denominator / &scale;
            }
            if !im.needs_rescaling() {
                result.im = im / &scale / &denominator / &scale;
            }
        }
        result
    }
}

impl<T: FloatLike> Div<&T> for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &T) -> Self::Output {
        Complex::new(self.re / rhs, self.im / rhs)
    }
}

impl<'a, T: FloatLike> Div<&'a Complex<T>> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &'a Complex<T>) -> Self::Output {
        self.clone() / rhs
    }
}

impl<T: FloatLike> Div<&T> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: &T) -> Self::Output {
        self.clone() / rhs
    }
}

impl<T: FloatLike> Div<Complex<T>> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: Complex<T>) -> Self::Output {
        self.clone() / rhs
    }
}

impl<T: FloatLike> Div<T> for &Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn div(self, rhs: T) -> Self::Output {
        self.clone() / rhs
    }
}

impl<T: FloatLike> DivAssign for Complex<T> {
    fn div_assign(&mut self, rhs: Self) {
        *self = self.clone().div(rhs);
    }
}

impl<T: FloatLike> DivAssign<T> for Complex<T> {
    fn div_assign(&mut self, rhs: T) {
        *self = self.clone().div(rhs);
    }
}

impl<T: FloatLike> DivAssign<&Complex<T>> for Complex<T> {
    fn div_assign(&mut self, rhs: &Self) {
        *self = self.clone().div(rhs);
    }
}

impl<T: FloatLike> DivAssign<&T> for Complex<T> {
    fn div_assign(&mut self, rhs: &T) {
        *self = self.clone().div(rhs);
    }
}

impl<T: FloatLike> Neg for Complex<T> {
    type Output = Complex<T>;

    #[inline]
    fn neg(self) -> Complex<T> {
        Complex::new(-self.re, -self.im)
    }
}

impl<T: FloatLike> Display for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('(')?;
        Display::fmt(&self.re, f)?;
        f.write_char('+')?;
        Display::fmt(&self.im, f)?;
        f.write_str("i)")
    }
}

impl<T: FloatLike> Debug for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_char('(')?;
        Debug::fmt(&self.re, f)?;
        f.write_char('+')?;
        Debug::fmt(&self.im, f)?;
        f.write_str("i)")
    }
}

impl<T: FloatLike> LowerExp for Complex<T> {
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

impl<T: FloatLike> FloatLike for Complex<T> {
    fn nan(&self) -> Option<Self> {
        Some(Self {
            re: self.re.nan()?,
            im: self.im.nan()?,
        })
    }

    #[inline]
    fn set_from(&mut self, other: &Self) {
        self.re.set_from(&other.re);
        self.im.set_from(&other.im);
    }

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
        if e != 0 {
            // Avoid multiplying by a weak-precision identity or changing a
            // signed zero before the first actual multiplication.
            let mut result = self.clone();
            for _ in 1..e {
                result *= self;
            }
            return result;
        }
        let prototype = if self.re.get_precision() >= self.im.get_precision() {
            &self.re
        } else {
            &self.im
        };
        Self::new(prototype.one(), prototype.zero())
    }

    fn inv(&self) -> Self {
        let n = self.norm_squared();
        if !n.needs_rescaling() {
            Complex::new(self.re.clone() / &n, -self.im.clone() / &n)
        } else {
            let abs = |x: &T| {
                if x.real_cmp(&x.zero()) == Some(Ordering::Less) {
                    -x.clone()
                } else {
                    x.clone()
                }
            };
            let (a, b) = (abs(&self.re), abs(&self.im));
            let scale = match a.real_cmp(&b) {
                Some(Ordering::Less) => b,
                Some(_) => a,
                None => return self.one() / self,
            };
            if scale.is_fully_zero() {
                return self.one() / self;
            }
            let (c, d) = (self.re.clone() / &scale, self.im.clone() / &scale);
            let n = c.clone() * &c + d.clone() * &d;
            Self::new(c / &n / &scale, -d / n / scale)
        }
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

    #[inline(always)]
    fn is_fully_zero(&self) -> bool {
        self.re.is_fully_zero() && self.im.is_fully_zero()
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

    #[inline(always)]
    fn conj(&self) -> Self {
        Complex::new(self.re.clone(), -self.im.clone())
    }

    #[inline]
    fn norm(&self) -> Self {
        Complex::new(self.re.hypot(&self.im), self.im.zero())
    }

    #[inline]
    fn sqrt(&self) -> Self {
        use std::num::FpCategory;

        let fixed = self.re.fixed_precision() && self.im.fixed_precision();
        // Only constants borrow the stronger component's precision. Computed
        // values keep their own precision, including uncertain small components.
        let prototype = if fixed || self.re.get_precision() >= self.im.get_precision() {
            &self.re
        } else {
            &self.im
        };

        // Resolve infinite limits before scaling (infinity / infinity is NaN).
        let im_class = self.im.real_classify();
        if im_class == Some(FpCategory::Infinite) {
            return Self::new(self.im.norm(), self.im.clone());
        }
        let re_class = self.re.real_classify();
        if re_class == Some(FpCategory::Infinite) {
            let other = if im_class == Some(FpCategory::Nan) {
                self.im.clone()
            } else {
                prototype.zero()
            };
            return if self.re.real_cmp(&self.re.zero()) == Some(Ordering::Less) {
                Self::new(other, self.re.norm().copy_sign(&self.im))
            } else {
                Self::new(self.re.clone(), other.copy_sign(&self.im))
            };
        }
        if re_class == Some(FpCategory::Nan) || im_class == Some(FpCategory::Nan) {
            let nan = prototype.nan().unwrap_or_else(|| {
                if re_class == Some(FpCategory::Nan) {
                    self.re.clone()
                } else {
                    self.im.clone()
                }
            });
            return Self::new(nan.clone(), nan);
        }

        let (a, b) = (self.re.norm(), self.im.norm());
        let order = a.real_cmp(&b);
        if order.is_none() {
            let (r, phi) = self.clone().to_polar_coordinates();
            return Self::from_polar_coordinates(r.sqrt(), phi / prototype.from_usize(2));
        }
        // Keep exact axes and signed-zero branch lips without a division by zero.
        if self.im.is_fully_zero() {
            let root = a.sqrt();
            return if self.re.real_cmp(&self.re.zero()) == Some(Ordering::Less) {
                Self::new(prototype.zero(), root.copy_sign(&self.im))
            } else {
                Self::new(root, self.im.clone())
            };
        }

        let two = prototype.from_usize(2);
        let s = if fixed {
            let h = a.hypot(&b);
            if !h.needs_rescaling() {
                (a.clone() / &two + h / &two).sqrt()
            } else {
                let scale = if order == Some(Ordering::Less) {
                    &b
                } else {
                    &a
                };
                let x = a.clone() / scale;
                let y = b.clone() / scale;
                scale.sqrt() * ((x.hypot(&y) + x) / &two).sqrt()
            }
        } else {
            // Unlike hypot's ratio formula, this sum retains the precision of
            // the dominant squared component when the other is tiny and uncertain.
            let scale = if order == Some(Ordering::Less) {
                &b
            } else {
                &a
            };
            let x = a.clone() / scale;
            let y = b.clone() / scale;
            let radius = (x.clone() * &x + y.clone() * &y).sqrt();
            scale.sqrt() * ((radius + x) / &two).sqrt()
        };
        let d = self.im.clone() / &s / two;
        if self.re.real_cmp(&self.re.zero()) == Some(Ordering::Less) {
            Self::new(d.norm(), s.copy_sign(&self.im))
        } else {
            Self::new(s, d)
        }
    }

    #[inline]
    fn log(&self) -> Self {
        let (mut a, mut b) = (self.re.norm(), self.im.norm());
        if a.real_cmp(&b) == Some(Ordering::Less) {
            std::mem::swap(&mut a, &mut b);
        }
        let one = a.one();
        let two = a.from_usize(2);
        let h = a.hypot(&b);
        let re = if a.real_cmp(&(one.clone() / &two)) == Some(Ordering::Greater)
            && a.real_cmp(&(one.clone() + one.clone() / &two)) == Some(Ordering::Less)
        {
            // Retain the small real part near the unit circle.
            ((a.clone() - &one) * (a.clone() + one) + b.clone() * b).log1p() / two
        } else if h.needs_rescaling() && !a.is_fully_zero() && a.real_cmp(&b).is_some() {
            let r = b / &a;
            a.log() + (r.clone() * r).log1p() / two
        } else {
            h.log()
        };
        Complex::new(re, self.arg())
    }

    #[inline]
    fn exp(&self) -> Self {
        let r = self.re.exp();
        let c = self.im.cos();
        let s = self.im.sin();
        if !r.needs_rescaling() || r.is_fully_zero() || self.re.real_cmp(&self.re.zero()).is_none()
        {
            Self::new(r.clone() * c, r * s)
        } else {
            Self::new(
                Self::exp_product(&self.re, c, false),
                Self::exp_product(&self.re, s, false),
            )
        }
    }

    #[inline]
    fn sin(&self) -> Self {
        let t = Self::new(-self.im.clone(), self.re.clone()).sinh();
        Self::new(t.im, -t.re)
    }

    #[inline]
    fn cos(&self) -> Self {
        Self::new(-self.im.clone(), self.re.clone()).cosh()
    }

    #[inline]
    fn tan(&self) -> Self {
        let t = Self::new(self.im.clone(), self.re.clone()).tanh();
        Self::new(t.im, t.re)
    }

    #[inline]
    fn asin(&self) -> Self {
        let t = Self::new(-self.im.clone(), self.re.clone()).asinh();
        Self::new(t.im, -t.re)
    }

    #[inline]
    fn acos(&self) -> Self {
        let two = self.re.from_usize(2);
        if self.large_inverse_argument() {
            return Self::new(
                self.im.norm().atan2(&self.re),
                -(self.log().re + two.log()).copy_sign(&self.im),
            );
        }
        let s1 = Self::new(self.re.one() - &self.re, -self.im.clone()).sqrt();
        let s2 = Self::new(self.re.one() + &self.re, self.im.clone()).sqrt();
        Self::new(
            s1.re.atan2(&s2.re) * two,
            (s2.re * s1.im - s2.im * s1.re).asinh(),
        )
    }

    #[inline]
    fn atan2(&self, x: &Self) -> Self {
        // Preserve the existing atan(self / x) branch convention.
        let r = self.clone() / x;
        let t = Self::new(-r.im, r.re).atanh();
        Self::new(t.im, -t.re)
    }

    #[inline]
    fn sinh(&self) -> Self {
        self.hyperbolic(false)
    }

    #[inline]
    fn cosh(&self) -> Self {
        self.hyperbolic(true)
    }

    #[inline]
    fn sech(&self) -> Self {
        let ch = self.re.cosh();
        if !ch.needs_rescaling() && ch.real_cmp(&ch.zero()).is_some() {
            return Self::new(ch * self.im.cos(), self.re.sinh() * self.im.sin()).inv();
        }
        // cosh(x + iy) / cosh(x) = cos(y) + i tanh(x) sin(y).
        // Both the numerator and denominator stay bounded at large |x|.
        let inverse = Self::new(self.im.cos(), self.re.tanh() * self.im.sin()).inv();
        let scale = self.re.sech();
        Self::new(scale.clone() * inverse.re, scale * inverse.im)
    }

    #[inline]
    fn csch(&self) -> Self {
        let ch = self.re.cosh();
        if !ch.needs_rescaling() && ch.real_cmp(&ch.zero()).is_some() {
            return Self::new(self.re.sinh() * self.im.cos(), ch * self.im.sin()).inv();
        }
        // sinh(x + iy) / cosh(x) = tanh(x) cos(y) + i sin(y).
        // Complex inversion also scales tiny denominators near the poles.
        let inverse = Self::new(self.re.tanh() * self.im.cos(), self.im.sin()).inv();
        let scale = self.re.sech();
        Self::new(scale.clone() * inverse.re, scale * inverse.im)
    }

    #[inline]
    fn tanh(&self) -> Self {
        // Divide sinh(x) cosh(x) + i sin(y) cos(y) and its denominator
        // sinh(x)^2 + cos(y)^2 by cosh(x)^2. All intermediates are bounded,
        // and the sum of squares avoids cancellation near the poles.
        // Compute sech(x) from exp(-|x|), avoiding overflow in cosh(x) and
        // cancellation in 1 - tanh(x)^2 (which would lose the imaginary tail).
        let t = self.re.tanh();
        let e = (-self.re.norm()).exp();
        let sech = (e.clone() + &e) / (e.one() + e.clone() * &e);
        let s = self.im.sin() * &sech;
        let c = self.im.cos() * sech;
        let m = t.clone() * &t + c.clone() * &c;
        Self::new(t / &m, s * c / m)
    }

    #[inline]
    fn asinh(&self) -> Self {
        if self.large_inverse_argument() {
            let r = self.log().re + self.re.from_usize(2).log();
            return Self::new(r.copy_sign(&self.re), self.im.atan2(&self.re.norm()));
        }
        // Square-root-product identities are also used by CPython's cmath:
        // https://github.com/python/cpython/blob/main/Modules/cmathmodule.c
        // Products of square roots avoid z^2 and the cancellation in
        // log(z + sqrt(1 + z^2)), particularly on the negative real axis.
        let s1 = Self::new(self.re.one() + &self.im, -self.re.clone()).sqrt();
        let s2 = Self::new(self.re.one() - &self.im, self.re.clone()).sqrt();
        Self::new(
            (s1.re.clone() * &s2.im - s2.re.clone() * &s1.im).asinh(),
            self.im.atan2(&(s1.re * s2.re - s1.im * s2.im)),
        )
    }

    #[inline]
    fn acosh(&self) -> Self {
        let two = self.re.from_usize(2);
        if self.large_inverse_argument() {
            return Self::new(self.log().re + two.log(), self.arg());
        }
        let s1 = Self::new(self.re.clone() - self.re.one(), self.im.clone()).sqrt();
        let s2 = Self::new(self.re.clone() + self.re.one(), self.im.clone()).sqrt();
        Self::new(
            (s1.re * &s2.re + s1.im.clone() * s2.im).asinh(),
            s1.im.atan2(&s2.re) * two,
        )
    }

    #[inline]
    fn atanh(&self) -> Self {
        if self.re.real_cmp(&self.re).is_none() {
            return ((self.one() + self).log() - (self.one() - self).log()) / self.re.from_usize(2);
        }
        let one = self.re.one();
        let two = self.re.from_usize(2);
        if self.large_inverse_argument() {
            let reciprocal = self.inv();
            return Self::new(
                reciprocal.re,
                (self.re.pi() / two).copy_sign(&self.im) + reciprocal.im,
            );
        }
        let x = self.re.norm();
        let y = self.im.norm();
        let minus = one.clone() - &x;
        let plus = one + &x;
        let d = minus.clone() * &minus + y.clone() * &y;
        let re = if !d.needs_rescaling() {
            (x * self.re.from_usize(4) / d).log1p() / self.re.from_usize(4)
        } else {
            (Self::new(plus.clone(), y.clone()).log().re
                - Self::new(minus.clone(), y.clone()).log().re)
                / &two
        };
        let im = if minus.is_fully_zero() && y.is_fully_zero() {
            self.im.clone()
        } else {
            (self.im.clone() * &two).atan2(&(minus * plus - y.clone() * y)) / two
        };
        Self::new(re.copy_sign(&self.re), im)
    }

    #[inline]
    fn powf(&self, e: &Self) -> Self {
        if e.re == self.re.zero() && e.im == self.im.zero() {
            self.pow(0)
        } else if e.im == self.im.zero() {
            // Exact half powers retain Cartesian root precision and branch lips.
            let half = e.re.one() / e.re.from_usize(2);
            if e.re == half {
                return self.sqrt();
            }
            if e.re == -half {
                return self.sqrt().inv();
            }
            let three_halves = e.re.from_usize(3) / e.re.from_usize(2);
            if e.re == three_halves {
                return self.clone() * self.sqrt();
            }
            if e.re == -three_halves {
                // Invert before cubing to avoid overflowing a representable result.
                return self.sqrt().inv().pow(3);
            }
            let (r, phi) = self.clone().to_polar_coordinates();
            let radius = if r.needs_rescaling() && !self.is_fully_zero() {
                (self.log().re * &e.re).exp()
            } else {
                r.powf(&e.re)
            };
            Self::from_polar_coordinates(radius, phi * e.re.clone())
        } else {
            (e * self.log()).exp()
        }
    }
}

impl<T: FixedPrecision> FixedPrecision for Complex<T> {
    const BINARY_PRECISION: usize = T::BINARY_PRECISION;
    const DECIMAL_PRECISION: usize = T::DECIMAL_PRECISION;
}

impl<T: FloatLike> From<T> for Complex<T> {
    fn from(value: T) -> Self {
        let zero = value.zero();
        Complex::new(value, zero)
    }
}

impl<'a, T: FloatLike + From<&'a Rational>> From<&'a Rational> for Complex<T> {
    fn from(value: &'a Rational) -> Self {
        let c: T = value.into();
        let zero = c.zero();
        Complex::new(c, zero)
    }
}

impl Add<&Complex<Integer>> for &Complex<Integer> {
    type Output = Complex<Integer>;

    fn add(self, rhs: &Complex<Integer>) -> Self::Output {
        Complex::new(&self.re + &rhs.re, &self.im + &rhs.im)
    }
}

impl Sub<&Complex<Integer>> for &Complex<Integer> {
    type Output = Complex<Integer>;

    fn sub(self, rhs: &Complex<Integer>) -> Self::Output {
        Complex::new(&self.re - &rhs.re, &self.im - &rhs.im)
    }
}

impl Mul<&Complex<Integer>> for &Complex<Integer> {
    type Output = Complex<Integer>;

    fn mul(self, rhs: &Complex<Integer>) -> Self::Output {
        Complex::new(
            &self.re * &rhs.re - &self.im * &rhs.im,
            &self.re * &rhs.im + &self.im * &rhs.re,
        )
    }
}

impl Div<&Complex<Integer>> for &Complex<Integer> {
    type Output = Complex<Integer>;

    fn div(self, rhs: &Complex<Integer>) -> Self::Output {
        let n = &rhs.re * &rhs.re + &rhs.im * &rhs.im;
        let re = &self.re * &rhs.re + &self.im * &rhs.im;
        let im = &self.im * &rhs.re - &self.re * &rhs.im;
        Complex::new(&re / &n, &im / &n)
    }
}

impl Complex<Integer> {
    pub fn gcd(mut self, mut other: Self) -> Self {
        if self.re.is_zero() && self.im.is_zero() {
            return other.clone();
        }
        if other.re.is_zero() && other.im.is_zero() {
            return self.clone();
        }

        while !other.re.is_zero() || !other.im.is_zero() {
            let q = &self / &other;
            let r = &self - &(&q * &other);
            (self, other) = (other, r);
        }
        self
    }
}

impl Complex<Rational> {
    pub fn gcd(&self, other: &Self) -> Self {
        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }

        let scaling = Rational::from(
            self.re
                .denominator_ref()
                .lcm(&other.re.denominator_ref())
                .lcm(&self.im.denominator_ref())
                .lcm(other.im.denominator_ref()),
        );

        let c1_i = Complex {
            re: (&self.re * &scaling).numerator(),
            im: (&self.im * &scaling).numerator(),
        };

        let c2_i = Complex {
            re: (&other.re * &scaling).numerator(),
            im: (&other.im * &scaling).numerator(),
        };

        let gcd = c1_i.gcd(c2_i);

        Complex {
            re: Rational::from(gcd.re) / &scaling,
            im: Rational::from(gcd.im) / &scaling,
        }
    }
}

impl Complex<Float> {
    pub fn to_f64(&self) -> Complex<f64> {
        Complex::new(self.re.to_f64(), self.im.to_f64())
    }

    pub fn to_double_float(&self) -> Complex<DoubleFloat> {
        Complex::new(self.re.to_double_float(), self.im.to_double_float())
    }
}
