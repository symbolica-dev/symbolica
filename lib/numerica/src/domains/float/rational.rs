use std::fmt::LowerExp;

use rand::Rng;

use super::{Constructible, FloatLike, RealLike, SingleFloat};
use super::{Float, Magnitude, RoundingDirection};
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
    fn set_precision(&mut self, _precision: u32) {}

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
    fn absolute_magnitude(&self) -> Magnitude {
        // Rounding away from zero before reducing the significand preserves a
        // conservative bound even when the rational is outside binary64 range.
        let direction = if self.is_negative() {
            RoundingDirection::Down
        } else {
            RoundingDirection::Up
        };
        Float::from_rational_round(self, 32, direction).absolute_magnitude()
    }

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

#[cfg(test)]
mod magnitude_tests {
    use super::{Integer, Magnitude, Rational, RealLike};

    #[test]
    fn absolute_magnitude_preserves_rational_range_and_discarded_bits() {
        let large: Rational = Integer::from(2).pow(4000).into();
        let small = large.inv();
        assert_eq!(large.absolute_magnitude(), Magnitude::ONE.scale_pow2(4000));
        assert_eq!(small.absolute_magnitude(), Magnitude::ONE.scale_pow2(-4000));
        assert_eq!(small.absolute_magnitude(), small.neg().absolute_magnitude());

        let tiny: Rational = (Integer::one(), Integer::from(2).pow(100)).into();
        let just_above_one = Rational::one() + tiny;
        assert_eq!(just_above_one.to_f64(), 1.0);
        assert_eq!(
            just_above_one.absolute_magnitude(),
            Magnitude::from_parts((1 << 31) + 1, 0)
        );
        assert_eq!(
            just_above_one.neg().absolute_magnitude(),
            just_above_one.absolute_magnitude()
        );
        assert_eq!(Rational::zero().absolute_magnitude(), Magnitude::ZERO);
    }
}
