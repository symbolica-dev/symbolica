use std::fmt::LowerExp;

use rand::Rng;

use super::{Constructible, Float, FloatLike, RealLike, SingleFloat};
use crate::domains::{integer::Integer, rational::Rational};

impl TryFrom<Float> for Rational {
    type Error = &'static str;

    fn try_from(value: Float) -> Result<Self, Self::Error> {
        value
            .try_to_rational()
            .ok_or("Cannot convert Float to Rational")
    }
}

impl LowerExp for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // lower-exp is ignored for integers
        f.write_fmt(format_args!("{self}"))
    }
}

impl FloatLike for Rational {
    fn set_from(&mut self, other: &Self) {
        *self = other.clone();
    }

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
        let rng1 = rng.random::<i64>();
        let rng2 = rng.random::<i64>();

        if rng1 > rng2 {
            (rng2, rng1).into()
        } else {
            (rng1, rng2).into()
        }
    }

    fn is_fully_zero(&self) -> bool {
        self.is_zero()
    }
}

impl Constructible for Rational {
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
        let rng1 = rng.random::<i64>();
        let rng2 = rng.random::<i64>();

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

impl RealLike for Rational {
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
