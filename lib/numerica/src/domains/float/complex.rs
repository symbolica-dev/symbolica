use std::{
    fmt::{Debug, Display, LowerExp, Write},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use rand::Rng;

use super::{
    Constructible, DoubleFloat, FixedPrecision, Float, FloatLike, Real, RealLike, SingleFloat,
};
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
        Complex::new(re / &n, im / &n)
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
