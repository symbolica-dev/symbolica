//! Compact nonnegative magnitudes for error estimates with a wide exponent range.

use std::{
    cmp::Ordering,
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign},
};

/// An eight-byte nonnegative floating-point magnitude, rounded upward.
///
/// Finite nonzero values are `significand * 2^(exponent - 31)`, with the top
/// significand bit set. Arithmetic has 32 significant bits and an `i32` binary
/// exponent. Positive underflow saturates at [`Self::MIN_POSITIVE`], and overflow
/// saturates at infinity. Zero, infinity and NaN have canonical representations.
/// This is an error magnitude, not a signed general-purpose floating-point type.
#[repr(C)]
#[derive(Clone, Copy, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(try_from = "[u8; 8]", into = "[u8; 8]"))]
pub struct Magnitude {
    significand: u32,
    exponent: i32,
}

impl Magnitude {
    pub const ZERO: Self = Self {
        significand: 0,
        exponent: 0,
    };
    pub const ONE: Self = Self {
        significand: 1 << 31,
        exponent: 0,
    };
    pub const INFINITY: Self = Self {
        significand: 0,
        exponent: i32::MAX,
    };
    pub const NAN: Self = Self {
        significand: 0,
        exponent: i32::MIN,
    };
    pub const MIN_POSITIVE: Self = Self {
        significand: 1 << 31,
        exponent: i32::MIN,
    };
    pub const MAX: Self = Self {
        significand: u32::MAX,
        exponent: i32::MAX,
    };

    /// Normalize `significand * 2^(exponent - 31)`, preserving exact zero.
    pub fn from_parts(significand: u32, exponent: i32) -> Self {
        Self::from_scaled(significand as u64, exponent as i64 - 31)
    }

    /// Convert the absolute value of an `f64`, rounding upward when necessary.
    pub fn from_f64(value: f64) -> Self {
        Self::from_f64_exp(value, 0)
    }

    /// Convert `abs(mantissa) * 2^exponent` without narrowing the exponent to `f64`.
    pub fn from_f64_exp(mantissa: f64, exponent: i64) -> Self {
        let bits = mantissa.to_bits() & !(1 << 63);
        let fraction = bits & ((1 << 52) - 1);
        let encoded_exponent = (bits >> 52) as i64;
        if encoded_exponent == 0x7ff {
            return if fraction == 0 {
                Self::INFINITY
            } else {
                Self::NAN
            };
        }
        if encoded_exponent == 0 {
            Self::from_scaled(fraction, exponent.saturating_sub(1074))
        } else {
            Self::from_scaled(
                (1 << 52) | fraction,
                exponent.saturating_add(encoded_exponent - 1023 - 52),
            )
        }
    }

    /// Construct an upper-rounded value of `2^logarithm`.
    pub fn from_log2(logarithm: f64) -> Self {
        if logarithm.is_nan() {
            return Self::NAN;
        }
        if logarithm == f64::NEG_INFINITY {
            return Self::ZERO;
        }
        if logarithm >= i32::MAX as f64 + 1. {
            return Self::INFINITY;
        }
        if logarithm < i32::MIN as f64 {
            return Self::MIN_POSITIVE;
        }
        let exponent = logarithm.floor() as i64;
        let fraction = logarithm - exponent as f64;
        if fraction == 0. {
            Self::ONE.scale_pow2(exponent)
        } else {
            // Guard the binary64 transcendental rounding before rounding to the
            // much coarser 32-bit magnitude significand.
            Self::from_f64_exp(fraction.exp2().next_up(), exponent)
        }
    }

    /// Construct `10^-accuracy` without passing through binary64's exponent range.
    pub fn from_decimal_accuracy(accuracy: f64) -> Self {
        if accuracy.is_nan() {
            return Self::NAN;
        }
        if accuracy == f64::INFINITY {
            return Self::ZERO;
        }
        if accuracy == f64::NEG_INFINITY {
            return Self::INFINITY;
        }
        if accuracy == 0. {
            return Self::ONE;
        }
        let power = -accuracy;
        let factor = if power > 0. {
            std::f64::consts::LOG2_10.next_up()
        } else {
            std::f64::consts::LOG2_10.next_down()
        };
        Self::from_log2((power * factor).next_up())
    }

    fn from_scaled(significand: u64, exponent: i64) -> Self {
        if significand == 0 {
            return Self::ZERO;
        }
        let top = 63 - significand.leading_zeros();
        let mut result_exponent = exponent.saturating_add(top as i64);
        let mut result_significand = if top > 31 {
            let shift = top - 31;
            let discarded = significand & ((1u64 << shift) - 1);
            (significand >> shift) + u64::from(discarded != 0)
        } else {
            significand << (31 - top)
        };
        if result_significand == 1u64 << 32 {
            result_significand >>= 1;
            result_exponent = result_exponent.saturating_add(1);
        }
        Self::from_normalized(result_significand as u32, result_exponent)
    }

    /// Finish an already normalized significand without repeating bit scans.
    #[inline]
    fn from_normalized(significand: u32, exponent: i64) -> Self {
        debug_assert!(significand >= 1 << 31);
        if exponent > i32::MAX as i64 {
            return Self::INFINITY;
        }
        if exponent < i32::MIN as i64 {
            return Self::MIN_POSITIVE;
        }
        Self {
            significand,
            exponent: exponent as i32,
        }
    }

    pub const fn significand(self) -> u32 {
        self.significand
    }
    pub const fn exponent(self) -> i32 {
        self.exponent
    }
    pub const fn is_zero(self) -> bool {
        self.significand == 0 && self.exponent == 0
    }
    pub const fn is_nan(self) -> bool {
        self.significand == 0 && self.exponent == i32::MIN
    }
    pub const fn is_infinite(self) -> bool {
        self.significand == 0 && self.exponent == i32::MAX
    }
    pub const fn is_finite(self) -> bool {
        self.significand != 0 || self.exponent == 0
    }

    /// Convert upward to `f64`, returning `None` for a finite nonzero magnitude
    /// outside the binary64 exponent range. Zero and the explicitly stored
    /// infinity/NaN special values return `Some` of their binary64 counterparts.
    /// In-range subnormal values round upward when not exactly representable.
    pub fn to_f64(self) -> Option<f64> {
        if self.significand != 0 && !(-1074..=1023).contains(&self.exponent) {
            None
        } else {
            Some(self.to_f64_saturating())
        }
    }

    /// Convert upward to `f64`. Positive underflow returns its smallest positive
    /// subnormal, and overflow returns infinity; the stored wide magnitude remains
    /// available through [`Self::significand`] and [`Self::exponent`].
    pub fn to_f64_saturating(self) -> f64 {
        if self.significand == 0 {
            return if self.is_nan() {
                f64::NAN
            } else if self.is_zero() {
                0.
            } else {
                f64::INFINITY
            };
        }
        if self.exponent > 1023 {
            return f64::INFINITY;
        }
        if self.exponent >= -1022 {
            return f64::from_bits(
                (((self.exponent + 1023) as u64) << 52)
                    | ((self.significand as u64 - (1 << 31)) << 21),
            );
        }
        if self.exponent < -1074 {
            return f64::from_bits(1);
        }
        let shift = -1043 - self.exponent;
        let units = if shift > 0 {
            (self.significand as u64).div_ceil(1 << shift)
        } else {
            (self.significand as u64) << -shift
        };
        f64::from_bits(units)
    }

    pub fn max(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            Self::NAN
        } else if self >= other {
            self
        } else {
            other
        }
    }

    /// Return the preceding nonnegative representable magnitude. This gives a
    /// lower bound after a single correctly upward-rounded extraction, but not
    /// after arithmetic that may have accumulated several rounding errors.
    pub fn next_down(self) -> Self {
        if self.is_zero() || self.is_nan() {
            self
        } else if self.is_infinite() {
            Self::MAX
        } else if self.significand > 1 << 31 {
            Self {
                significand: self.significand - 1,
                exponent: self.exponent,
            }
        } else if self.exponent == i32::MIN {
            Self::ZERO
        } else {
            Self {
                significand: u32::MAX,
                exponent: self.exponent - 1,
            }
        }
    }

    pub fn scale_pow2(self, power: i64) -> Self {
        if self.significand == 0 {
            return self;
        }
        Self::from_normalized(
            self.significand,
            (self.exponent as i64).saturating_add(power),
        )
    }

    /// Return a lower bound for `max(self - other, 0)`, rounding downward.
    ///
    /// Unlike the upward-rounded arithmetic operators, a positive result below
    /// [`Self::MIN_POSITIVE`] becomes zero. NaN and infinity minus infinity
    /// return NaN; finite minus infinity returns zero.
    pub fn sub_lower(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() || (self.is_infinite() && other.is_infinite()) {
            return Self::NAN;
        }
        if self <= other {
            return Self::ZERO;
        }
        if self.is_infinite() || other.is_zero() {
            return self;
        }

        let gap = self.exponent as i64 - other.exponent as i64;
        if gap > 32 {
            // The subtrahend is smaller than a predecessor ulp, even just
            // below a power of two where the spacing is halved.
            return self.next_down();
        }
        let difference = ((self.significand as u64) << gap) - other.significand as u64;
        let top = 63 - difference.leading_zeros();
        let exponent = other.exponent as i64 - 31 + top as i64;
        if exponent < i32::MIN as i64 {
            return Self::ZERO;
        }
        let significand = if top > 31 {
            difference >> (top - 31)
        } else {
            difference << (31 - top)
        };
        Self {
            significand: significand as u32,
            exponent: exponent as i32,
        }
    }

    pub fn sqrt(self) -> Self {
        if self.significand == 0 {
            return self;
        }
        let radicand = (self.significand as u64) << (31 + self.exponent.rem_euclid(2));
        let lower = radicand.isqrt();
        let upper = lower + u64::from(lower * lower != radicand);
        Self::from_scaled(upper, self.exponent.div_euclid(2) as i64 - 31)
    }

    pub fn powf(self, power: f64) -> Self {
        if power == 0. || self == Self::ONE {
            return Self::ONE;
        }
        if self.is_nan() || power.is_nan() {
            return Self::NAN;
        }
        if self.is_zero() {
            return if power > 0. {
                Self::ZERO
            } else {
                Self::INFINITY
            };
        }
        if self.is_infinite() {
            return if power > 0. {
                Self::INFINITY
            } else {
                Self::ZERO
            };
        }
        if power.is_infinite() {
            return if (self > Self::ONE) == power.is_sign_positive() {
                Self::INFINITY
            } else {
                Self::ZERO
            };
        }
        if power.fract() == 0. && power.abs() < u64::MAX as f64 {
            let mut exponent = power.abs() as u64;
            let mut factor = if power < 0. { Self::ONE / self } else { self };
            let mut result = Self::ONE;
            while exponent != 0 {
                if exponent & 1 != 0 {
                    result *= factor;
                }
                exponent >>= 1;
                if exponent != 0 {
                    factor *= factor;
                }
            }
            return result;
        }
        let fractional_log = (self.significand as f64 / (1u64 << 31) as f64).log2();
        let logarithm = if power > 0. {
            (self.exponent as f64 + fractional_log.next_up().next_up()).next_up()
        } else {
            (self.exponent as f64 + fractional_log.next_down().next_down()).next_down()
        };
        Self::from_log2((logarithm * power).next_up())
    }

    pub fn log2(self) -> f64 {
        if self.significand == 0 {
            if self.is_zero() {
                f64::NEG_INFINITY
            } else if self.is_nan() {
                f64::NAN
            } else {
                f64::INFINITY
            }
        } else {
            self.exponent as f64 + (self.significand as f64 / (1u64 << 31) as f64).log2()
        }
    }

    pub fn log10(self) -> f64 {
        self.log2() * std::f64::consts::LOG10_2
    }

    /// Compute `log2(self / other)` without first rounding the ratio or
    /// subtracting two large, nearly equal logarithms.
    pub fn log2_ratio(self, other: Self) -> f64 {
        if self.significand == 0 || other.significand == 0 {
            self.log2() - other.log2()
        } else {
            (self.exponent as i64 - other.exponent as i64) as f64
                + (self.significand as f64 / other.significand as f64).log2()
        }
    }

    pub fn to_le_bytes(self) -> [u8; 8] {
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&self.significand.to_le_bytes());
        bytes[4..].copy_from_slice(&self.exponent.to_le_bytes());
        bytes
    }

    /// Decode a canonical magnitude, rejecting non-normalized significands and
    /// unassigned zero-significand special encodings.
    pub fn from_le_bytes(bytes: [u8; 8]) -> Result<Self, &'static str> {
        let significand = u32::from_le_bytes(bytes[..4].try_into().unwrap());
        let exponent = i32::from_le_bytes(bytes[4..].try_into().unwrap());
        if significand == 0 {
            if ![0, i32::MIN, i32::MAX].contains(&exponent) {
                return Err("Invalid magnitude special value");
            }
        } else if significand & (1 << 31) == 0 {
            return Err("Magnitude significand is not normalized");
        }
        Ok(Self {
            significand,
            exponent,
        })
    }
}

impl Default for Magnitude {
    fn default() -> Self {
        Self::ZERO
    }
}
impl From<f64> for Magnitude {
    fn from(value: f64) -> Self {
        Self::from_f64(value)
    }
}
impl From<Magnitude> for [u8; 8] {
    fn from(value: Magnitude) -> Self {
        value.to_le_bytes()
    }
}
impl TryFrom<[u8; 8]> for Magnitude {
    type Error = &'static str;
    fn try_from(bytes: [u8; 8]) -> Result<Self, Self::Error> {
        Self::from_le_bytes(bytes)
    }
}

impl PartialEq for Magnitude {
    fn eq(&self, other: &Self) -> bool {
        !self.is_nan()
            && !other.is_nan()
            && self.significand == other.significand
            && self.exponent == other.exponent
    }
}

impl PartialOrd for Magnitude {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.is_nan() || other.is_nan() {
            return None;
        }
        if self == other {
            return Some(Ordering::Equal);
        }
        if self.is_zero() || other.is_infinite() {
            return Some(Ordering::Less);
        }
        if self.is_infinite() || other.is_zero() {
            return Some(Ordering::Greater);
        }
        Some(
            self.exponent
                .cmp(&other.exponent)
                .then_with(|| self.significand.cmp(&other.significand)),
        )
    }
}

impl Add for Magnitude {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        if self.is_nan() || other.is_nan() {
            return Self::NAN;
        }
        if self.is_infinite() || other.is_infinite() {
            return Self::INFINITY;
        }
        if self.is_zero() {
            return other;
        }
        if other.is_zero() {
            return self;
        }
        let (larger, smaller) = if self.exponent >= other.exponent {
            (self, other)
        } else {
            (other, self)
        };
        let difference = larger.exponent as i64 - smaller.exponent as i64;
        if difference >= 32 {
            Self::from_scaled(larger.significand as u64 + 1, larger.exponent as i64 - 31)
        } else {
            Self::from_scaled(
                ((larger.significand as u64) << difference) + smaller.significand as u64,
                smaller.exponent as i64 - 31,
            )
        }
    }
}

impl Mul for Magnitude {
    type Output = Self;
    fn mul(self, other: Self) -> Self {
        if self.significand == 0 || other.significand == 0 {
            if self.is_nan()
                || other.is_nan()
                || (self.is_zero() && other.is_infinite())
                || (other.is_zero() && self.is_infinite())
            {
                return Self::NAN;
            }
            return if self.is_zero() || other.is_zero() {
                Self::ZERO
            } else {
                Self::INFINITY
            };
        }

        // Two normalized 32-bit significands have a product with its top bit
        // at 62 or 63; a general normalization bit scan is unnecessary.
        let product = self.significand as u64 * other.significand as u64;
        let shift = 31 + (product >> 63);
        let discarded = product & ((1 << shift) - 1);
        let mut significand = (product >> shift) + u64::from(discarded != 0);
        let mut exponent = self.exponent as i64 + other.exponent as i64 + shift as i64 - 31;
        if significand == 1 << 32 {
            significand >>= 1;
            exponent += 1;
        }
        Self::from_normalized(significand as u32, exponent)
    }
}

impl Div for Magnitude {
    type Output = Self;
    fn div(self, other: Self) -> Self {
        if self.significand == 0 || other.significand == 0 {
            if self.is_nan()
                || other.is_nan()
                || (self.is_zero() && other.is_zero())
                || (self.is_infinite() && other.is_infinite())
            {
                return Self::NAN;
            }
            return if self.is_infinite() || other.is_zero() {
                Self::INFINITY
            } else {
                Self::ZERO
            };
        }
        let shift = if self.significand >= other.significand {
            31
        } else {
            32
        };
        let quotient = ((self.significand as u64) << shift).div_ceil(other.significand as u64);
        // The shift puts the rounded quotient directly in [2^31, 2^32).
        Self::from_normalized(
            quotient as u32,
            self.exponent as i64 - other.exponent as i64 + 31 - shift,
        )
    }
}

macro_rules! borrowed_operator {
    ($trait:ident, $method:ident, $assign:ident, $assign_method:ident) => {
        impl $trait<&Magnitude> for Magnitude {
            type Output = Self;
            fn $method(self, other: &Magnitude) -> Self {
                self.$method(*other)
            }
        }
        impl $trait<Magnitude> for &Magnitude {
            type Output = Magnitude;
            fn $method(self, other: Magnitude) -> Magnitude {
                (*self).$method(other)
            }
        }
        impl $trait<&Magnitude> for &Magnitude {
            type Output = Magnitude;
            fn $method(self, other: &Magnitude) -> Magnitude {
                (*self).$method(*other)
            }
        }
        impl $assign for Magnitude {
            fn $assign_method(&mut self, other: Self) {
                *self = (*self).$method(other);
            }
        }
    };
}
borrowed_operator!(Add, add, AddAssign, add_assign);
borrowed_operator!(Mul, mul, MulAssign, mul_assign);
borrowed_operator!(Div, div, DivAssign, div_assign);

impl fmt::Debug for Magnitude {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.significand == 0 {
            self.to_f64_saturating().fmt(formatter)
        } else {
            write!(
                formatter,
                "Magnitude({} * 2^{})",
                self.significand,
                self.exponent as i64 - 31
            )
        }
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for Magnitude {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        self.to_le_bytes().encode(encoder)
    }
}
#[cfg(feature = "bincode")]
impl<C> bincode::Decode<C> for Magnitude {
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Self::from_le_bytes(<[u8; 8]>::decode(decoder)?)
            .map_err(|error| bincode::error::DecodeError::OtherString(error.into()))
    }
}
#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(Magnitude);

#[cfg(test)]
mod tests {
    use super::Magnitude as M;

    #[test]
    fn compact_layout_and_validated_encoding() {
        assert_eq!(std::mem::size_of::<M>(), 8);
        for magnitude in [
            M::ZERO,
            M::ONE,
            M::MIN_POSITIVE,
            M::INFINITY,
            M::NAN,
            M::from_f64(1.234),
        ] {
            assert_eq!(
                M::from_le_bytes(magnitude.to_le_bytes())
                    .unwrap()
                    .to_le_bytes(),
                magnitude.to_le_bytes()
            );
        }
        assert!(M::from_le_bytes([1, 0, 0, 0, 0, 0, 0, 0]).is_err());
        assert!(M::from_le_bytes([0, 0, 0, 0, 1, 0, 0, 0]).is_err());
        assert_eq!(M::from_parts(1, 31), M::ONE);
        assert_ne!(M::NAN, M::NAN);
        assert_eq!(M::NAN.partial_cmp(&M::ONE), None);
    }

    #[cfg(feature = "bincode")]
    #[test]
    fn bincode_roundtrip_validates_magnitude_bytes() {
        let config = bincode::config::standard();
        for value in [
            M::ZERO,
            M::ONE,
            M::MIN_POSITIVE,
            M::MAX,
            M::INFINITY,
            M::NAN,
        ] {
            let bytes = bincode::encode_to_vec(value, config).unwrap();
            assert_eq!(bytes, value.to_le_bytes());
            let (decoded, consumed): (M, _) = bincode::decode_from_slice(&bytes, config).unwrap();
            assert_eq!(decoded.to_le_bytes(), value.to_le_bytes());
            assert_eq!(consumed, 8);
        }
        assert!(bincode::decode_from_slice::<M, _>(&[1, 0, 0, 0, 0, 0, 0, 0], config).is_err());
        assert!(bincode::decode_from_slice::<M, _>(&[0, 0, 0, 0, 1, 0, 0, 0], config).is_err());
    }

    #[test]
    fn wide_exponents_and_conservative_endpoints() {
        for exponent in [-1000000, -4000, -1074, -1023, -1022, 0, 1023, 4000, 1000000] {
            let magnitude = M::from_f64_exp(1., exponent);
            assert!(magnitude.is_finite());
            assert!(!magnitude.is_zero());
            assert_eq!(magnitude.exponent(), exponent as i32);
            assert_eq!(magnitude.log2(), exponent as f64);
            assert!(magnitude.sqrt() * magnitude.sqrt() >= magnitude);
            if exponent % 2 == 0 {
                assert_eq!(magnitude.sqrt() * magnitude.sqrt(), magnitude);
            }
        }
        assert_eq!(M::ONE.scale_pow2(i64::MAX), M::INFINITY);
        assert_eq!(M::ONE.scale_pow2(i64::MIN), M::MIN_POSITIVE);
        assert_eq!(M::MIN_POSITIVE / M::from_f64(2.), M::MIN_POSITIVE);
        assert_eq!(
            M::from_f64_exp(1., -4000).to_f64_saturating(),
            f64::from_bits(1)
        );
        assert!(M::from_f64_exp(1., 4000).to_f64_saturating().is_infinite());
        assert_eq!(M::from_f64_exp(1., -4000).to_f64(), None);
        assert_eq!(M::from_f64_exp(1., 4000).to_f64(), None);
        assert_eq!(M::from_f64_exp(1., -1075).to_f64(), None);
        assert_eq!(M::from_f64_exp(1., -1074).to_f64(), Some(f64::from_bits(1)));
        assert_eq!(M::from_f64_exp(1., 1023).to_f64(), Some(2f64.powi(1023)));
        assert_eq!(M::from_f64_exp(1., 1024).to_f64(), None);
        assert_eq!(M::ZERO.to_f64(), Some(0.));
        assert_eq!(M::INFINITY.to_f64(), Some(f64::INFINITY));
        assert!(M::NAN.to_f64().unwrap().is_nan());
        assert_eq!((M::ONE + M::MIN_POSITIVE).significand(), (1 << 31) + 1);
        assert!((M::ZERO * M::INFINITY).is_nan());
        assert!((M::ZERO / M::ZERO).is_nan());
        assert_eq!(M::MIN_POSITIVE.next_down(), M::ZERO);
        assert_eq!(M::INFINITY.next_down(), M::MAX);
        assert_eq!(M::ONE.next_down(), M::from_parts(u32::MAX, -1));
        assert_eq!((M::ONE + M::MIN_POSITIVE).next_down(), M::ONE);
    }

    #[test]
    fn binary64_conversion_rounds_up() {
        let values = [
            f64::from_bits(1),
            f64::from_bits(2),
            f64::MIN_POSITIVE,
            1.,
            1. + f64::EPSILON,
            1.234567890123,
            f64::MAX,
        ];
        for value in values {
            assert!(M::from_f64(value).to_f64_saturating() >= value);
            assert_eq!(M::from_f64(-value), M::from_f64(value));
        }
        for bits in 1..2048u64 {
            let value = f64::from_bits(bits);
            assert_eq!(M::from_f64(value).to_f64(), Some(value));
        }
        let subnormal = M::from_parts((1 << 31) + 1, -1074);
        assert_eq!(subnormal.to_f64().unwrap().to_bits(), 2);
    }

    #[test]
    fn arithmetic_bounds_small_integer_examples() {
        for a in 1..128 {
            for b in 1..128 {
                let x = M::from_f64(a as f64);
                let y = M::from_f64(b as f64);
                assert_eq!((x + y).to_f64(), Some((a + b) as f64));
                assert_eq!((x * y).to_f64(), Some((a * b) as f64));
                assert_eq!(x.sub_lower(y).to_f64(), Some((a - b).max(0) as f64));
                let quotient = (x / y).to_f64().unwrap();
                assert!(quotient >= a as f64 / b as f64);
                assert!(quotient <= (a as f64 / b as f64) * (1. + 2f64.powi(-30)));
                assert!(x.sqrt().to_f64().unwrap() >= (a as f64).sqrt());
            }
        }
        assert_eq!(M::from_f64(2.).powf(-1000.).log2(), -1000.);
        assert_eq!(
            M::from_f64_exp(1., 1000000).log2_ratio(M::from_f64_exp(1., 999999)),
            1.
        );
        assert!(M::from_decimal_accuracy(1000.).is_finite());
        assert!(!M::from_decimal_accuracy(1000.).is_zero());
    }

    #[test]
    fn subtraction_lower_preserves_clearance_and_saturates_downward() {
        let adjacent = M::from_parts((1 << 31) + 1, -4000);
        assert_eq!(
            adjacent.sub_lower(M::ONE.scale_pow2(-4000)),
            M::ONE.scale_pow2(-4031)
        );
        assert_eq!(M::ONE.sub_lower(M::ONE.next_down()), M::ONE.scale_pow2(-32));
        assert_eq!(
            M::ONE.sub_lower(M::from_parts(u32::MAX, -32)),
            M::from_parts(u32::MAX - 1, -1)
        );
        assert_eq!(M::ONE.sub_lower(M::ONE.scale_pow2(-32)), M::ONE.next_down());
        assert_eq!(M::ONE.sub_lower(M::ONE.scale_pow2(-33)), M::ONE.next_down());
        assert_eq!(M::MIN_POSITIVE.sub_lower(M::MIN_POSITIVE), M::ZERO);
        assert_eq!(
            M::from_parts((1 << 31) + 1, i32::MIN).sub_lower(M::MIN_POSITIVE),
            M::ZERO
        );
        assert_eq!(M::ONE.sub_lower(M::from_f64(2.)), M::ZERO);
        assert_eq!(M::MAX.sub_lower(M::ZERO), M::MAX);
        assert_eq!(M::INFINITY.sub_lower(M::ONE), M::INFINITY);
        assert_eq!(M::ONE.sub_lower(M::INFINITY), M::ZERO);
        assert!(M::INFINITY.sub_lower(M::INFINITY).is_nan());
        assert!(M::NAN.sub_lower(M::ZERO).is_nan());
        assert!(M::ZERO.sub_lower(M::NAN).is_nan());
    }

    #[test]
    fn normalized_arithmetic_matches_general_normalization() {
        let mut seed = 0x4321_abcd_9876_1234u64;
        for index in 0..10000 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let left = (seed as u32) | (1 << 31);
            let right = ((seed >> 32) as u32) | (1 << 31);
            let exponent = match index % 5 {
                0 => i32::MIN,
                1 => i32::MAX,
                _ => (seed >> 32) as i32,
            };
            let x = M::from_parts(left, exponent);
            let y = M::from_parts(right, -(index % 8192));
            assert_eq!(
                x * y,
                M::from_scaled(
                    left as u64 * right as u64,
                    x.exponent as i64 + y.exponent as i64 - 62
                )
            );
            let shift = if left >= right { 31 } else { 32 };
            assert_eq!(
                x / y,
                M::from_scaled(
                    ((left as u64) << shift).div_ceil(right as u64),
                    x.exponent as i64 - y.exponent as i64 - shift,
                )
            );
            let power = index as i64 - 5000;
            assert_eq!(
                x.scale_pow2(power),
                M::from_scaled(left as u64, (x.exponent as i64 - 31) + power)
            );
        }
        for x in [M::ZERO, M::ONE, M::INFINITY, M::NAN] {
            for y in [M::ZERO, M::ONE, M::INFINITY, M::NAN] {
                for (actual, expected) in [
                    (x * y, x.to_f64().unwrap() * y.to_f64().unwrap()),
                    (x / y, x.to_f64().unwrap() / y.to_f64().unwrap()),
                ] {
                    if expected.is_nan() {
                        assert!(actual.is_nan());
                    } else {
                        assert_eq!(actual.to_f64(), Some(expected));
                    }
                }
            }
        }
    }

    #[cfg(feature = "float-mpfr")]
    #[test]
    fn operations_bound_high_precision_values() {
        use rug::{Float, ops::Pow};
        fn float(value: M) -> Float {
            let mut result = Float::with_val(192, value.significand());
            result <<= value.exponent() - 31;
            result
        }
        fn bound(actual: M, expected: Float) {
            let upper = float(actual);
            assert!(upper >= expected, "{actual:?} underestimated {expected}");
            assert!(upper <= expected * Float::with_val(192, 1. + 2f64.powi(-28)));
        }
        let mantissas = [1 << 31, (1 << 31) + 1, 0xaaaa_aaaa, u32::MAX];
        for accuracy in [-1000., -20.25, 0., 20.25, 1000.] {
            bound(
                M::from_decimal_accuracy(accuracy),
                Float::with_val(192, 10).pow(-accuracy),
            );
        }
        for difference in [0, 1, 30, 31, 32, 33, 64, 4000] {
            let larger = M::from_parts(u32::MAX, 0);
            let smaller = M::from_parts(u32::MAX, -difference);
            bound(
                larger + smaller,
                Float::with_val(192, float(larger) + float(smaller)),
            );
            for significand in mantissas {
                let larger = M::from_parts(significand, 0);
                if larger > smaller {
                    let actual = float(larger.sub_lower(smaller));
                    let expected = Float::with_val(192, float(larger) - float(smaller));
                    assert!(
                        actual <= expected,
                        "subtraction overestimated the clearance"
                    );
                    assert!(actual >= expected * Float::with_val(192, 1. - 2f64.powi(-30)));
                }
            }
        }
        for a in mantissas {
            for b in mantissas {
                for exponent in [-4000, -3, 0, 3, 4000] {
                    let x = M::from_parts(a, exponent);
                    let y = M::from_parts(b, -exponent);
                    bound(x + y, Float::with_val(192, float(x) + float(y)));
                    bound(x * y, Float::with_val(192, float(x) * float(y)));
                    bound(x / y, Float::with_val(192, float(x) / float(y)));
                    if x > y {
                        let actual = float(x.sub_lower(y));
                        let expected = Float::with_val(192, float(x) - float(y));
                        assert!(actual <= expected);
                        assert!(actual >= expected * Float::with_val(192, 1. - 2f64.powi(-30)));
                    }
                    bound(x.sqrt(), float(x).sqrt());
                    for power in [-2., -0.5, 0.5, 1.5, 3.] {
                        bound(x.powf(power), float(x).pow(power));
                    }
                }
            }
        }
    }
}
