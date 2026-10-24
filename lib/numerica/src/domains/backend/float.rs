use crate::domains::backend::integer::MultiPrecisionInteger;

/// The rounding direction for a `Float` operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoundingDirection {
    /// Round to the nearest representable value, breaking ties toward an even
    /// significand.
    Nearest,
    /// Round toward positive infinity.
    Up,
    /// Round toward negative infinity.
    Down,
}

#[cfg(feature = "float-mpfr")]
pub use rug::{
    Assign, Float as MultiPrecisionFloat, Rational as BackendRational,
    float::Constant,
    ops::{CompleteRound, Pow},
};

/// A compact upper estimate of the absolute value, preserving the backend exponent.
pub(crate) trait MultiPrecisionFloatMagnitude {
    fn absolute_magnitude(&self) -> crate::domains::float::Magnitude;
}

#[cfg(feature = "float-mpfr")]
impl MultiPrecisionFloatMagnitude for MultiPrecisionFloat {
    fn absolute_magnitude(&self) -> crate::domains::float::Magnitude {
        // Directed rounding must be away from zero before discarding the sign.
        let rounding = if self.is_sign_negative() {
            rug::float::Round::Down
        } else {
            rug::float::Round::Up
        };
        let (mantissa, exponent) = self.to_f64_exp_round(rounding);
        crate::domains::float::Magnitude::from_f64_exp(mantissa, i64::from(exponent))
    }
}

pub(crate) trait MultiPrecisionFloatRounding {
    fn add_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self;
    fn sub_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self;
    fn mul_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self;
    fn div_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self;
}

/// Integer operations shared by independently selected integer and float backends.
pub(crate) trait MultiPrecisionFloatInteger {
    fn add_integer(self, value: MultiPrecisionInteger) -> Self;
    fn mul_integer(self, value: MultiPrecisionInteger) -> Self;
    fn div_integer(self, value: MultiPrecisionInteger) -> Self;
    fn from_integer(prec: u32, value: MultiPrecisionInteger) -> Self;
    fn to_integer_exact(&self) -> Option<MultiPrecisionInteger>;
}

/// Exact rational operations shared by the independently selected integer and float backends.
pub(crate) trait MultiPrecisionFloatRational {
    fn mul_integer_ratio(self, num: MultiPrecisionInteger, den: MultiPrecisionInteger) -> Self;
    fn div_integer_ratio(self, num: MultiPrecisionInteger, den: MultiPrecisionInteger) -> Self;
    fn from_integer_ratio_round(
        num: MultiPrecisionInteger,
        den: MultiPrecisionInteger,
        prec: u32,
        direction: RoundingDirection,
    ) -> Self;
    fn to_integer_ratio(&self) -> Option<(MultiPrecisionInteger, MultiPrecisionInteger)>;
}

#[cfg(feature = "float-mpfr")]
fn rug_rounding_direction(direction: RoundingDirection) -> rug::float::Round {
    match direction {
        RoundingDirection::Nearest => rug::float::Round::Nearest,
        RoundingDirection::Up => rug::float::Round::Up,
        RoundingDirection::Down => rug::float::Round::Down,
    }
}

#[cfg(feature = "float-mpfr")]
impl MultiPrecisionFloatRounding for MultiPrecisionFloat {
    fn add_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self {
        Self::with_val_round(prec, self + rhs, rug_rounding_direction(direction)).0
    }

    fn sub_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self {
        Self::with_val_round(prec, self - rhs, rug_rounding_direction(direction)).0
    }

    fn mul_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self {
        Self::with_val_round(prec, self * rhs, rug_rounding_direction(direction)).0
    }

    fn div_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self {
        Self::with_val_round(prec, self / rhs, rug_rounding_direction(direction)).0
    }
}

#[cfg(all(feature = "float-mpfr", feature = "integer-gmp"))]
fn rug_integer_from_backend(value: MultiPrecisionInteger) -> rug::Integer {
    value.into_raw()
}

#[cfg(all(feature = "float-mpfr", feature = "integer-malachite"))]
fn rug_integer_from_backend(value: MultiPrecisionInteger) -> rug::Integer {
    value.to_string().parse().unwrap()
}

#[cfg(all(feature = "float-mpfr", feature = "integer-gmp"))]
fn backend_integer_from_rug(value: rug::Integer) -> MultiPrecisionInteger {
    MultiPrecisionInteger::from_raw(value)
}

#[cfg(all(feature = "float-mpfr", feature = "integer-malachite"))]
fn backend_integer_from_rug(value: rug::Integer) -> MultiPrecisionInteger {
    value.to_string().parse().unwrap()
}

#[cfg(feature = "float-mpfr")]
impl MultiPrecisionFloatInteger for MultiPrecisionFloat {
    fn add_integer(self, value: MultiPrecisionInteger) -> Self {
        self + rug_integer_from_backend(value)
    }

    fn mul_integer(self, value: MultiPrecisionInteger) -> Self {
        self * rug_integer_from_backend(value)
    }

    fn div_integer(self, value: MultiPrecisionInteger) -> Self {
        self / rug_integer_from_backend(value)
    }

    fn from_integer(prec: u32, value: MultiPrecisionInteger) -> Self {
        Self::with_val(prec, rug_integer_from_backend(value))
    }

    fn to_integer_exact(&self) -> Option<MultiPrecisionInteger> {
        self.to_integer().map(backend_integer_from_rug)
    }
}

#[cfg(feature = "float-mpfr")]
impl MultiPrecisionFloatRational for MultiPrecisionFloat {
    fn mul_integer_ratio(self, num: MultiPrecisionInteger, den: MultiPrecisionInteger) -> Self {
        self * rug::Rational::from((rug_integer_from_backend(num), rug_integer_from_backend(den)))
    }

    fn div_integer_ratio(self, num: MultiPrecisionInteger, den: MultiPrecisionInteger) -> Self {
        self / rug::Rational::from((rug_integer_from_backend(num), rug_integer_from_backend(den)))
    }

    fn from_integer_ratio_round(
        num: MultiPrecisionInteger,
        den: MultiPrecisionInteger,
        prec: u32,
        direction: RoundingDirection,
    ) -> Self {
        let value =
            rug::Rational::from((rug_integer_from_backend(num), rug_integer_from_backend(den)));
        Self::with_val_round(prec, value, rug_rounding_direction(direction)).0
    }

    fn to_integer_ratio(&self) -> Option<(MultiPrecisionInteger, MultiPrecisionInteger)> {
        self.to_rational().map(|value| {
            let (num, den) = value.into_numer_denom();
            (backend_integer_from_rug(num), backend_integer_from_rug(den))
        })
    }
}

#[cfg(feature = "float-astro")]
mod astro {
    use std::{
        cell::RefCell,
        cmp::Ordering,
        fmt::{self, Debug, Display, Formatter},
        hash::Hash,
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    };

    use astro_float::{BigFloat, Consts, INF_NEG, INF_POS, NAN, Radix, RoundingMode, Sign};
    use malachite_q::Rational as MalachiteRational;

    use super::RoundingDirection;
    use crate::domains::backend::integer::MultiPrecisionInteger;

    pub type BackendRational = MalachiteRational;

    pub enum Constant {
        Pi,
        Euler,
    }

    pub trait Assign<T> {
        fn assign_into(prec: u32, val: T) -> MultiPrecisionFloat;
    }

    pub trait CompleteRound {
        type Completed;

        fn complete(self, prec: u32) -> Self::Completed;
    }

    pub trait Pow<Rhs> {
        type Output;

        fn pow(self, rhs: Rhs) -> Self::Output;
    }

    pub struct ParseIncomplete {
        text: String,
        radix: u8,
    }

    impl CompleteRound for ParseIncomplete {
        type Completed = MultiPrecisionFloat;

        fn complete(self, prec: u32) -> Self::Completed {
            MultiPrecisionFloat::parse_at_prec_radix(self.text.as_bytes(), self.radix, prec)
                .unwrap()
        }
    }

    #[derive(Clone)]
    pub struct MultiPrecisionFloat {
        value: BigFloat,
        prec: u32,
    }

    impl MultiPrecisionFloat {
        /// Borrow nonzero finite magnitude limbs in least-significant-first order.
        /// The value is `sign * integer(words) * 2^(exponent - words.len()*WORD_BITS)`.
        pub(crate) fn binary_parts(&self) -> Option<(&[astro_float::Word], i64)> {
            let (words, bits, _, exponent, _) = self.value.as_raw_parts()?;
            (bits != 0).then_some((words, i64::from(exponent)))
        }

        /// Decode an exact finite value from normalized little-endian bytes.
        /// The value is `sign * integer(significand) * 2^(normalized_exponent - 8*len)`.
        pub(crate) fn from_binary_parts(
            precision: u32,
            negative: bool,
            normalized_exponent: i64,
            significand: &[u8],
        ) -> Result<Self, String> {
            if precision == 0 {
                return Err("Binary float precision must be positive".into());
            }
            let Some((&first, rest)) = significand.split_first() else {
                return Err("Binary float significand is empty".into());
            };
            if first == 0 || rest.last().unwrap_or(&first) & 0x80 == 0 {
                return Err("Binary float significand is not canonical".into());
            }
            if normalized_exponent > i64::from(astro_float::EXPONENT_MAX) {
                return Err("Binary float exponent exceeds the Astro range".into());
            }

            let word_bits = astro_float::WORD_BIT_SIZE;
            let source_bits = significand
                .len()
                .checked_mul(8)
                .ok_or("Binary float significand is too large")?;
            let significant_bits = source_bits - first.trailing_zeros() as usize;
            // Astro arithmetic can retain a full word of value bits beyond its
            // requested precision. Preserve those bits and the requested hint
            // independently rather than silently rounding during import.
            let required_bits = (precision as usize).max(significant_bits);
            let word_count = required_bits
                .checked_add(word_bits - 1)
                .ok_or("Binary float precision is too large")?
                / word_bits;
            let storage_bits = word_count
                .checked_mul(word_bits)
                .ok_or("Binary float precision is too large")?;
            if storage_bits >= (isize::MAX / 2 + astro_float::EXPONENT_MIN as isize) as usize {
                return Err("Binary float precision exceeds the Astro range".into());
            }

            let exponent = normalized_exponent.max(i64::from(astro_float::EXPONENT_MIN));
            let padding_bits = storage_bits
                .checked_sub(source_bits)
                .ok_or("Binary float significand does not fit its storage")?;
            let shift = normalized_exponent
                .checked_sub(exponent)
                .and_then(|shift| shift.checked_add(padding_bits as i64))
                .ok_or("Binary float exponent is out of range")?;
            // Below emin, the exact value must fit Astro's subnormal spacing.
            // Canonical bytes have at most seven low zero bits, so this also
            // rejects extreme negative exponents before allocating a buffer.
            if shift < -i64::from(first.trailing_zeros()) {
                return Err("Binary float value is below the exact Astro range".into());
            }

            let mut words = smallvec::SmallVec::<[astro_float::Word; 4]>::new();
            words
                .try_reserve_exact(word_count)
                .map_err(|_| "Cannot allocate binary float significand")?;
            words.resize(word_count, 0);
            if significand.len() >= std::mem::size_of::<astro_float::Word>()
                && shift >= 0
                && shift % 8 == 0
            {
                // Normal values need only byte padding. Import whole words
                // instead of updating each destination word once per byte.
                const WORD_BYTES: usize = std::mem::size_of::<astro_float::Word>();
                let byte_offset = shift as usize / 8;
                let mut word = byte_offset / WORD_BYTES;
                let offset = byte_offset % WORD_BYTES;
                let mut source = significand;
                if offset != 0 {
                    let count = source.len().min(WORD_BYTES - offset);
                    let mut bytes = [0; WORD_BYTES];
                    bytes[offset..offset + count].copy_from_slice(&source[..count]);
                    words[word] = astro_float::Word::from_le_bytes(bytes);
                    source = &source[count..];
                    word += 1;
                }
                let mut chunks = source.chunks_exact(WORD_BYTES);
                let count = chunks.len();
                for (destination, bytes) in
                    words[word..word + count].iter_mut().zip(chunks.by_ref())
                {
                    *destination = astro_float::Word::from_le_bytes(bytes.try_into().unwrap());
                }
                if !chunks.remainder().is_empty() {
                    let mut bytes = [0; WORD_BYTES];
                    bytes[..chunks.remainder().len()].copy_from_slice(chunks.remainder());
                    words[word + count] = astro_float::Word::from_le_bytes(bytes);
                }
            } else {
                // Short significands avoid chunk setup. Subnormal values may
                // need a non-byte-aligned shift, including dropping up to seven
                // low bits already checked to be zero.
                for (index, &byte) in significand.iter().enumerate() {
                    let bit = index as i64 * 8 + shift;
                    if bit < 0 {
                        words[0] |= (byte >> -bit as u32) as astro_float::Word;
                    } else {
                        let word = bit as usize / word_bits;
                        let offset = bit as usize % word_bits;
                        words[word] |= (byte as astro_float::Word) << offset;
                        if offset > word_bits - 8 {
                            words[word + 1] |= (byte as astro_float::Word) >> (word_bits - offset);
                        }
                    }
                }
            }
            let value = BigFloat::from_raw_parts(
                &words,
                (source_bits as i64 + shift) as usize,
                if negative { Sign::Neg } else { Sign::Pos },
                exponent as i32,
                false,
            );
            if value.is_nan() {
                return Err("Cannot construct exact binary float in Astro".into());
            }
            Ok(Self {
                value,
                prec: precision,
            })
        }
    }

    impl super::MultiPrecisionFloatMagnitude for MultiPrecisionFloat {
        fn absolute_magnitude(&self) -> crate::domains::float::Magnitude {
            use crate::domains::float::Magnitude;

            let Some((words, _, _, exponent, _)) = self.value.as_raw_parts() else {
                return Magnitude::from_f64(if self.is_nan() {
                    f64::NAN
                } else {
                    f64::INFINITY
                });
            };
            let Some(index) = words.iter().rposition(|&word| word != 0) else {
                return Magnitude::ZERO;
            };
            let top = words[index] as u64;
            let significant = u64::BITS - top.leading_zeros();
            let word_bits = astro_float::WORD_BIT_SIZE;
            let (mut significand, discarded) = if significant >= 32 {
                let shift = significant - 32;
                (
                    top >> shift,
                    top & ((1u64 << shift) - 1) != 0 || words[..index].iter().any(|&w| w != 0),
                )
            } else {
                let needed = 32 - significant;
                let mut significand = top << needed;
                let discarded = if index == 0 {
                    false
                } else {
                    let next = words[index - 1] as u64;
                    let shift = word_bits as u32 - needed;
                    significand |= next >> shift;
                    next & ((1u64 << shift) - 1) != 0 || words[..index - 1].iter().any(|&w| w != 0)
                };
                (significand, discarded)
            };
            significand += u64::from(discarded);
            let scale = i64::from(exponent) - (words.len() - index) as i64 * word_bits as i64
                + i64::from(significant)
                - 32;
            // A 33-bit rounded significand is exactly representable as f64.
            // The full exponent stays separate; no logarithms or allocation.
            Magnitude::from_f64_exp(significand as f64, scale)
        }
    }

    impl PartialEq for MultiPrecisionFloat {
        fn eq(&self, other: &Self) -> bool {
            (self.is_nan() && other.is_nan()) || self.value == other.value
        }
    }

    impl Eq for MultiPrecisionFloat {}

    impl Hash for MultiPrecisionFloat {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            if self.is_nan() {
                state.write_u64(0x7ff8000000000000);
            } else if self.is_zero() {
                state.write_u64(0);
            } else {
                self.is_sign_negative().hash(state);
                self.get_exp().hash(state);
                if let Some((words, _, _, _, _)) = self.value.as_raw_parts() {
                    // Equal values can have extra low zero words at higher precision.
                    let first = words.iter().position(|&word| word != 0).unwrap();
                    words[first..].hash(state);
                }
            }
        }
    }

    #[cfg(feature = "serde")]
    impl serde::Serialize for MultiPrecisionFloat {
        fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
        {
            (self.prec, self.value.to_string()).serialize(serializer)
        }
    }

    #[cfg(feature = "serde")]
    impl<'de> serde::Deserialize<'de> for MultiPrecisionFloat {
        fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            let (prec, value) = <(u32, String)>::deserialize(deserializer)?;
            Self::parse_at_prec(&value, prec).map_err(serde::de::Error::custom)
        }
    }

    pub trait IntoMultiPrecisionFloat {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat;
    }

    const ROUNDING_MODE: RoundingMode = RoundingMode::ToEven;

    fn rounding_mode(direction: RoundingDirection) -> RoundingMode {
        match direction {
            RoundingDirection::Nearest => RoundingMode::ToEven,
            RoundingDirection::Up => RoundingMode::Up,
            RoundingDirection::Down => RoundingMode::Down,
        }
    }

    fn precision(prec: u32) -> usize {
        prec.max(1) as usize
    }

    fn guard_precision(prec: u32) -> usize {
        precision(prec).saturating_add(64)
    }

    thread_local! {
        static CONSTANTS: RefCell<Consts> =
            RefCell::new(Consts::new().expect("astro-float constants cache initialized"));
    }

    fn with_constants<T>(f: impl FnOnce(&mut Consts) -> T) -> T {
        CONSTANTS.with(|constants| {
            let mut constants = constants.borrow_mut();
            f(&mut constants)
        })
    }

    fn radix(radix: u8) -> Result<Radix, String> {
        match radix {
            2 => Ok(Radix::Bin),
            8 => Ok(Radix::Oct),
            10 => Ok(Radix::Dec),
            16 => Ok(Radix::Hex),
            _ => Err(format!("unsupported radix {radix}")),
        }
    }

    fn format_value(value: &BigFloat, radix: Radix) -> String {
        with_constants(|constants| value.format(radix, ROUNDING_MODE, constants))
            .unwrap_or_else(|_| value.to_string())
    }

    fn format_lower_exp_value(value: &BigFloat, precision: Option<usize>) -> String {
        let value = format_value(value, Radix::Dec);
        let Some(precision) = precision else {
            return value;
        };
        let Some((mantissa, exponent)) = value.split_once('e') else {
            return value;
        };

        let (sign, mantissa) = mantissa
            .strip_prefix('-')
            .map_or(("", mantissa), |mantissa| ("-", mantissa));
        let mut digits = mantissa.chars().filter(|c| *c != '.').collect::<String>();
        if digits.is_empty() {
            digits.push('0');
        }
        let kept_digits = precision + 1;
        while digits.len() <= kept_digits {
            digits.push('0');
        }
        let discarded = &digits.as_bytes()[kept_digits..];
        let round_up = discarded[0] > b'5'
            || (discarded[0] == b'5'
                && (discarded[1..].iter().any(|&d| d != b'0')
                    || (digits.as_bytes()[kept_digits - 1] - b'0') % 2 != 0));
        let mut digits = digits.as_bytes()[..kept_digits].to_vec();
        let mut exponent = exponent.to_owned();

        if round_up {
            let mut carry = true;
            for digit in digits.iter_mut().rev() {
                if *digit == b'9' {
                    *digit = b'0';
                } else {
                    *digit += 1;
                    carry = false;
                    break;
                }
            }

            if carry {
                digits[0] = b'1';
                for digit in &mut digits[1..] {
                    *digit = b'0';
                }
                if let Ok(parsed_exponent) = exponent.parse::<i32>() {
                    exponent = format!("{:+}", parsed_exponent + 1);
                }
            }
        }

        let digits = String::from_utf8(digits).unwrap();

        if precision == 0 {
            format!("{sign}{}e{exponent}", &digits[..1])
        } else {
            format!(
                "{sign}{}.{}e{exponent}",
                &digits[..1],
                &digits[1..=precision]
            )
        }
    }

    fn format_decimal_value(value: &BigFloat, digits: usize, scientific: bool) -> String {
        if value.is_zero() {
            return if value.is_negative() { "-0" } else { "0" }.to_owned();
        }
        if value.is_inf() {
            return if value.is_negative() { "-inf" } else { "inf" }.to_owned();
        }

        let text = format_lower_exp_value(value, Some(digits.saturating_sub(1)));
        let Some((mantissa, exponent)) = text.split_once('e') else {
            return text;
        };
        let Ok(exponent) = exponent.parse::<i32>() else {
            return text;
        };

        // Like MPFR, use decimal notation only when the rounded significant
        // digits contain every digit before the point, and the value is >= 1.
        if !scientific && exponent >= 0 && (exponent as usize) < digits {
            let mut text = mantissa.replace('.', "");
            let point = exponent as usize + 1 + usize::from(text.starts_with('-'));
            if point < text.len() {
                text.insert(point, '.');
            }
            text
        } else {
            format!("{mantissa}e{exponent}")
        }
    }

    fn parse_integer_at_precision(value: impl Display, prec: usize) -> BigFloat {
        with_constants(|constants| {
            BigFloat::parse(
                &value.to_string(),
                Radix::Dec,
                prec.max(1),
                ROUNDING_MODE,
                constants,
            )
        })
    }

    fn rounded_value(mut value: BigFloat, prec: u32) -> MultiPrecisionFloat {
        let _ = value.set_precision(precision(prec), ROUNDING_MODE);
        MultiPrecisionFloat { value, prec }
    }

    fn ratio_to_float(
        num: MultiPrecisionInteger,
        den: MultiPrecisionInteger,
        prec: u32,
    ) -> BigFloat {
        let p = precision(prec);
        let work_precision = guard_precision(prec)
            .max(num.significant_bits() as usize)
            .max(den.significant_bits() as usize);
        let num = parse_integer_at_precision(num, work_precision);
        let den = parse_integer_at_precision(den, work_precision);
        num.div(&den, p, ROUNDING_MODE)
    }

    impl super::MultiPrecisionFloatRounding for MultiPrecisionFloat {
        fn add_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self {
            MultiPrecisionFloat {
                value: self
                    .value
                    .add(&rhs.value, precision(prec), rounding_mode(direction)),
                prec,
            }
        }

        fn sub_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self {
            MultiPrecisionFloat {
                value: self
                    .value
                    .sub(&rhs.value, precision(prec), rounding_mode(direction)),
                prec,
            }
        }

        fn mul_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self {
            MultiPrecisionFloat {
                value: self
                    .value
                    .mul(&rhs.value, precision(prec), rounding_mode(direction)),
                prec,
            }
        }

        fn div_round(&self, rhs: &Self, prec: u32, direction: RoundingDirection) -> Self {
            MultiPrecisionFloat {
                value: self
                    .value
                    .div(&rhs.value, precision(prec), rounding_mode(direction)),
                prec,
            }
        }
    }

    impl super::MultiPrecisionFloatInteger for MultiPrecisionFloat {
        fn add_integer(self, value: MultiPrecisionInteger) -> Self {
            self + value
        }

        fn mul_integer(self, value: MultiPrecisionInteger) -> Self {
            self * value
        }

        fn div_integer(self, value: MultiPrecisionInteger) -> Self {
            self / value
        }

        fn from_integer(prec: u32, value: MultiPrecisionInteger) -> Self {
            Self::with_val(prec, value)
        }

        fn to_integer_exact(&self) -> Option<MultiPrecisionInteger> {
            self.to_integer()
        }
    }

    impl super::MultiPrecisionFloatRational for MultiPrecisionFloat {
        fn mul_integer_ratio(self, num: MultiPrecisionInteger, den: MultiPrecisionInteger) -> Self {
            self * rational_from_integer_ratio(num, den)
        }

        fn div_integer_ratio(self, num: MultiPrecisionInteger, den: MultiPrecisionInteger) -> Self {
            self / rational_from_integer_ratio(num, den)
        }

        fn from_integer_ratio_round(
            num: MultiPrecisionInteger,
            den: MultiPrecisionInteger,
            prec: u32,
            direction: RoundingDirection,
        ) -> Self {
            let p = precision(prec);
            let work_precision = guard_precision(prec)
                .max(num.significant_bits() as usize)
                .max(den.significant_bits() as usize);
            let num = parse_integer_at_precision(num, work_precision);
            let den = parse_integer_at_precision(den, work_precision);
            MultiPrecisionFloat {
                value: num.div(&den, p, rounding_mode(direction)),
                prec,
            }
        }

        fn to_integer_ratio(&self) -> Option<(MultiPrecisionInteger, MultiPrecisionInteger)> {
            finite_to_rational(&self.value).map(rational_into_integer_ratio)
        }
    }

    fn integer_to_float(value: impl Display, prec: u32) -> BigFloat {
        parse_integer_at_precision(value, precision(prec))
    }

    fn rational_from_integer_ratio(
        num: MultiPrecisionInteger,
        den: MultiPrecisionInteger,
    ) -> MalachiteRational {
        format!("{num}/{den}").parse().unwrap()
    }

    fn rational_into_integer_ratio(
        value: MalachiteRational,
    ) -> (MultiPrecisionInteger, MultiPrecisionInteger) {
        let value = value.to_string();
        if let Some((num, den)) = value.split_once('/') {
            (num.parse().unwrap(), den.parse().unwrap())
        } else {
            (value.parse().unwrap(), MultiPrecisionInteger::from(1))
        }
    }

    fn finite_to_rational(value: &BigFloat) -> Option<MalachiteRational> {
        let (words, _, sign, exponent, _) = value.as_raw_parts()?;
        let mut mantissa = MultiPrecisionInteger::from(0);
        for word in words.iter().rev() {
            mantissa =
                (mantissa << astro_float::WORD_BIT_SIZE) + MultiPrecisionInteger::from(*word);
        }

        if sign == Sign::Neg {
            mantissa = -mantissa;
        }

        let shift = exponent - (words.len() * astro_float::WORD_BIT_SIZE) as i32;
        if shift >= 0 {
            let num = mantissa << shift as usize;
            Some(rational_from_integer_ratio(
                num,
                MultiPrecisionInteger::from(1),
            ))
        } else {
            let den = MultiPrecisionInteger::from(1) << (-shift) as usize;
            Some(rational_from_integer_ratio(mantissa, den))
        }
    }

    fn euler_mascheroni(prec: u32) -> BigFloat {
        let target_precision = precision(prec);
        let work_precision = guard_precision(prec);
        let x_value = (target_precision / 4).max(8);
        let x = BigFloat::from_u64(x_value as u64, work_precision);
        let x_squared = x.mul(&x, work_precision, ROUNDING_MODE);

        let mut term = BigFloat::from_u32(1, work_precision);
        let mut i_sum = term.clone();
        let mut s_sum = BigFloat::new(work_precision);
        let mut harmonic = BigFloat::new(work_precision);

        for k in 1..=(4 * x_value + 64) {
            let k_float = BigFloat::from_u64(k as u64, work_precision);
            harmonic = harmonic.add(
                &k_float.reciprocal(work_precision, ROUNDING_MODE),
                work_precision,
                ROUNDING_MODE,
            );

            let k_squared = k_float.mul(&k_float, work_precision, ROUNDING_MODE);
            term = term.mul(&x_squared, work_precision, ROUNDING_MODE).div(
                &k_squared,
                work_precision,
                ROUNDING_MODE,
            );

            i_sum = i_sum.add(&term, work_precision, ROUNDING_MODE);
            s_sum = s_sum.add(
                &harmonic.mul(&term, work_precision, ROUNDING_MODE),
                work_precision,
                ROUNDING_MODE,
            );
        }

        let quotient = s_sum.div(&i_sum, work_precision, ROUNDING_MODE);
        let log_x = with_constants(|constants| x.ln(work_precision, ROUNDING_MODE, constants));
        quotient.sub(&log_x, target_precision, ROUNDING_MODE)
    }

    impl MultiPrecisionFloat {
        fn fmt_decimal(&self, f: &mut Formatter<'_>, scientific: bool) -> fmt::Result {
            // Zero precision, like omitted precision, requests enough digits
            // to reconstruct the binary value, following MPFR's convention.
            let digits = f
                .precision()
                .filter(|&digits| digits != 0)
                .unwrap_or_else(|| {
                    1 + (f64::from(self.prec) * std::f64::consts::LOG10_2).ceil() as usize
                });
            let text = format_decimal_value(&self.value, digits, scientific);
            f.pad_integral(
                !text.starts_with('-'),
                "",
                text.strip_prefix('-').unwrap_or(&text),
            )
        }

        pub fn new(prec: u32) -> Self {
            Self {
                value: BigFloat::new(precision(prec)),
                prec,
            }
        }

        pub fn with_val<T>(prec: u32, val: T) -> Self
        where
            Self: Assign<T>,
        {
            <Self as Assign<T>>::assign_into(prec, val)
        }

        pub fn parse(s: &str) -> Result<ParseIncomplete, String> {
            Ok(ParseIncomplete {
                text: s.to_owned(),
                radix: 10,
            })
        }

        pub fn parse_radix(s: impl AsRef<[u8]>, radix: u8) -> Result<ParseIncomplete, String> {
            let text = std::str::from_utf8(s.as_ref())
                .map_err(|e| e.to_string())?
                .to_owned();
            Ok(ParseIncomplete { text, radix })
        }

        pub(crate) fn parse_at_prec(s: &str, prec: u32) -> Result<Self, String> {
            Self::parse_at_prec_radix(s.as_bytes(), 10, prec)
        }

        pub(crate) fn parse_at_prec_radix(s: &[u8], radix: u8, prec: u32) -> Result<Self, String> {
            let s = std::str::from_utf8(s).map_err(|e| e.to_string())?;
            let radix = self::radix(radix)?;
            let mut value = with_constants(|constants| {
                BigFloat::parse(s, radix, precision(prec), ROUNDING_MODE, constants)
            });
            // Astro normalizes parsed zeroes; retain the literal sign.
            if value.is_zero() && s.starts_with('-') {
                value.set_sign(Sign::Neg);
            }
            if value.is_nan() {
                Err(format!("failed to parse float `{s}`"))
            } else {
                Ok(Self { value, prec })
            }
        }

        pub(crate) fn from_f64(prec: u32, value: f64) -> Self {
            let value = if value.is_nan() {
                NAN
            } else if value == f64::INFINITY {
                INF_POS
            } else if value == f64::NEG_INFINITY {
                INF_NEG
            } else {
                let negative = value.is_sign_negative();
                // Astro 0.9 misinterprets the exponent of subnormal f64s.
                // Normalize exactly before conversion, then restore the exponent.
                let subnormal = value.is_subnormal();
                let normalized = if subnormal {
                    value * 4503599627370496.0
                } else {
                    value
                };
                let mut value = BigFloat::from_f64(
                    normalized,
                    precision(prec).max(f64::MANTISSA_DIGITS as usize),
                );
                if subnormal {
                    value.set_exponent(value.exponent().unwrap() - 52);
                }
                let _ = value.set_precision(precision(prec), ROUNDING_MODE);
                if value.is_zero() && negative {
                    value.set_sign(Sign::Neg);
                }
                value
            };
            Self { value, prec }
        }

        pub(crate) fn from_rational(prec: u32, value: MalachiteRational) -> Self {
            let (num, den) = rational_into_integer_ratio(value);
            let value = ratio_to_float(num, den, prec);
            Self { value, prec }
        }

        pub fn prec(&self) -> u32 {
            self.prec
        }

        pub fn set_prec(&mut self, prec: u32) {
            self.prec = prec;
            let _ = self.value.set_precision(precision(prec), ROUNDING_MODE);
        }

        pub fn is_nan(&self) -> bool {
            self.value.is_nan()
        }

        pub fn is_zero(&self) -> bool {
            self.value.is_zero()
        }

        pub fn is_finite(&self) -> bool {
            !self.value.is_inf() && !self.value.is_nan()
        }

        pub fn is_sign_negative(&self) -> bool {
            self.value.sign().is_some_and(|s| s.is_negative())
        }

        pub fn to_f64(&self) -> f64 {
            if self.value.is_nan() {
                f64::NAN
            } else if self.value.is_inf_pos() {
                f64::INFINITY
            } else if self.value.is_inf_neg() {
                f64::NEG_INFINITY
            } else if self.value.is_zero() {
                if self.is_sign_negative() { -0.0 } else { 0.0 }
            } else {
                format_value(&self.value, Radix::Dec)
                    .parse::<f64>()
                    .unwrap_or_else(|_| {
                        if self.is_sign_negative() {
                            f64::NEG_INFINITY
                        } else {
                            f64::INFINITY
                        }
                    })
            }
        }

        pub fn abs(self) -> Self {
            Self {
                value: if self.value.is_negative() {
                    self.value.neg()
                } else {
                    self.value
                },
                prec: self.prec,
            }
        }

        pub fn recip(self) -> Self {
            let value = self.value.reciprocal(precision(self.prec), ROUNDING_MODE);
            Self {
                value,
                prec: self.prec,
            }
        }

        pub fn pow_u64(&self, mut exp: u64) -> Self {
            if let Ok(exp) = usize::try_from(exp) {
                return Self {
                    value: self.value.powi(exp, precision(self.prec), ROUNDING_MODE),
                    prec: self.prec,
                };
            }

            let mut base = self.clone();
            let mut result = Self::with_val(self.prec, 1u32);
            while exp != 0 {
                if exp & 1 == 1 {
                    result *= &base;
                }
                exp >>= 1;
                if exp != 0 {
                    let square = base.clone() * &base;
                    base = square;
                }
            }
            result
        }

        pub fn pow(&self, rhs: &Self) -> Self {
            let prec = self.prec;
            // Astro's general power can loop while trying to prove rounding of
            // exact roots (for example 4**0.5). Small dyadic exponents can be
            // evaluated directly by square roots and integer exponentiation.
            if !self.is_sign_negative()
                && self.is_finite()
                && rhs.get_exp().is_some_and(|e| (-32..=64).contains(&e))
                && let Some(ratio) = finite_to_rational(&rhs.value)
            {
                let (num, den) = rational_into_integer_ratio(ratio);
                let num = if rhs.is_sign_negative() { -num } else { num };
                if let (Some(mut power), Some(den)) = (num.to_u64(), den.to_u64())
                    && den.is_power_of_two()
                    && den.trailing_zeros() <= 32
                {
                    let work = guard_precision(prec).saturating_add(64);
                    let mut base = self.value.clone();
                    let _ = base.set_precision(work, ROUNDING_MODE);
                    for _ in 0..den.trailing_zeros() {
                        base = base.sqrt(work, ROUNDING_MODE);
                    }
                    let mut value = BigFloat::from_u8(1, work);
                    while power != 0 {
                        if power & 1 != 0 {
                            value = value.mul(&base, work, ROUNDING_MODE);
                        }
                        power >>= 1;
                        if power != 0 {
                            base = base.mul(&base, work, ROUNDING_MODE);
                        }
                    }
                    if rhs.is_sign_negative() {
                        value = value.reciprocal(work, ROUNDING_MODE);
                    }
                    let _ = value.set_precision(precision(prec), ROUNDING_MODE);
                    return Self { value, prec };
                }
            }
            let value = with_constants(|constants| {
                self.value
                    .pow(&rhs.value, precision(prec), ROUNDING_MODE, constants)
            });
            Self { value, prec }
        }

        fn unary(
            &self,
            f: impl FnOnce(&BigFloat, usize, RoundingMode, &mut Consts) -> BigFloat,
        ) -> Self {
            let value = with_constants(|constants| {
                f(&self.value, precision(self.prec), ROUNDING_MODE, constants)
            });
            Self {
                value,
                prec: self.prec,
            }
        }

        fn cmp_zero(&self) -> Option<Ordering> {
            self.value
                .cmp(&BigFloat::new(precision(self.prec)))
                .map(|c| {
                    if c < 0 {
                        Ordering::Less
                    } else if c > 0 {
                        Ordering::Greater
                    } else {
                        Ordering::Equal
                    }
                })
        }

        pub fn to_rational(&self) -> Option<MalachiteRational> {
            finite_to_rational(&self.value)
        }

        pub fn to_string_radix(&self, radix: i32, _digits: Option<usize>) -> String {
            format_value(
                &self.value,
                self::radix(radix.try_into().unwrap_or(10)).unwrap_or(Radix::Dec),
            )
        }

        pub fn get_exp(&self) -> Option<i32> {
            // Match MPFR: zero has no significant-bit exponent. Treating it as
            // an ordinary exponent discards precision when adding an exact zero.
            if self.is_zero() || !self.is_finite() {
                None
            } else {
                self.value.exponent()
            }
        }

        pub fn get_significand(&self) -> Option<String> {
            if self.is_finite() && !self.is_zero() {
                Some(format_value(&self.value, Radix::Dec))
            } else {
                None
            }
        }

        pub fn to_integer(&self) -> Option<MultiPrecisionInteger> {
            let value = finite_to_rational(&self.value)?;
            let (num, den) = rational_into_integer_ratio(value);
            Some(num / den)
        }

        pub fn sqrt_ref(&self) -> Self {
            self.unary(|x, p, rm, _| x.sqrt(p, rm))
        }

        pub fn ln_ref(&self) -> Self {
            self.unary(BigFloat::ln)
        }

        pub fn exp_ref(&self) -> Self {
            self.unary(BigFloat::exp)
        }

        pub fn tanh_ref(&self) -> Self {
            self.unary(BigFloat::tanh)
        }

        pub fn ln(self) -> Self {
            self.ln_ref()
        }

        pub fn exp(self) -> Self {
            self.exp_ref()
        }

        pub fn sin(self) -> Self {
            self.unary(BigFloat::sin)
        }

        pub fn cos(self) -> Self {
            self.unary(BigFloat::cos)
        }

        pub fn tan(self) -> Self {
            self.unary(BigFloat::tan)
        }

        pub fn asin(self) -> Self {
            self.unary(BigFloat::asin)
        }

        pub fn acos(self) -> Self {
            self.unary(BigFloat::acos)
        }

        pub fn atan(self) -> Self {
            self.unary(BigFloat::atan)
        }

        pub fn atan2(self, rhs: &Self) -> Self {
            let prec = self.prec.min(rhs.prec);
            if self.is_nan() || rhs.is_nan() {
                return Self::from_f64(prec, f64::NAN);
            }
            if !self.is_finite() || !rhs.is_finite() {
                let quarters = if !self.is_finite() {
                    if !rhs.is_finite() {
                        if rhs.is_sign_negative() { 3 } else { 1 }
                    } else {
                        2
                    }
                } else if rhs.is_sign_negative() {
                    4
                } else {
                    0
                };
                let angle = if quarters == 0 {
                    Self::new(prec)
                } else {
                    Self::with_val(prec, Constant::Pi) * quarters as i64 / 4i64
                };
                return if self.is_sign_negative() {
                    -angle
                } else {
                    angle
                };
            }
            let x_cmp = rhs.cmp_zero();
            let y_cmp = self.cmp_zero();

            if x_cmp == Some(Ordering::Equal) {
                let pi = Self::with_val(prec, Constant::Pi);
                let half_pi = pi.clone() / 2i64;
                return if y_cmp == Some(Ordering::Less) {
                    -half_pi
                } else if y_cmp == Some(Ordering::Greater) {
                    half_pi
                } else {
                    let angle = if rhs.is_sign_negative() {
                        pi
                    } else {
                        Self::new(prec)
                    };
                    if self.is_sign_negative() {
                        -angle
                    } else {
                        angle
                    }
                };
            }

            if y_cmp == Some(Ordering::Equal) {
                let angle = if x_cmp == Some(Ordering::Less) {
                    Self::with_val(prec, Constant::Pi)
                } else {
                    Self::new(prec)
                };
                return if self.is_sign_negative() {
                    -angle
                } else {
                    angle
                };
            }

            let work_precision = guard_precision(prec);
            let mut y = self.value.clone();
            let mut x = rhs.value.clone();
            let _ = y.set_precision(work_precision, ROUNDING_MODE);
            let _ = x.set_precision(work_precision, ROUNDING_MODE);

            let x_squared = x.mul(&x, work_precision, ROUNDING_MODE);
            let y_squared = y.mul(&y, work_precision, ROUNDING_MODE);
            let radius = x_squared
                .add(&y_squared, work_precision, ROUNDING_MODE)
                .sqrt(work_precision, ROUNDING_MODE);

            let value = with_constants(|constants| {
                let two = BigFloat::from_u32(2, work_precision);
                if x_cmp == Some(Ordering::Greater) {
                    let denominator = radius.add(&x, work_precision, ROUNDING_MODE);
                    let angle = y.div(&denominator, work_precision, ROUNDING_MODE).atan(
                        work_precision,
                        ROUNDING_MODE,
                        constants,
                    );
                    return two.mul(&angle, work_precision, ROUNDING_MODE);
                }

                let denominator = radius.sub(&x, work_precision, ROUNDING_MODE);
                let angle = y.div(&denominator, work_precision, ROUNDING_MODE).atan(
                    work_precision,
                    ROUNDING_MODE,
                    constants,
                );
                let correction = two.mul(&angle, work_precision, ROUNDING_MODE);
                let pi = constants.pi(work_precision, ROUNDING_MODE);
                if y_cmp == Some(Ordering::Less) {
                    pi.neg().sub(&correction, work_precision, ROUNDING_MODE)
                } else {
                    pi.sub(&correction, work_precision, ROUNDING_MODE)
                }
            });

            rounded_value(value, prec)
        }

        pub fn sinh(self) -> Self {
            self.unary(BigFloat::sinh)
        }

        pub fn cosh(self) -> Self {
            self.unary(BigFloat::cosh)
        }

        pub fn tanh(self) -> Self {
            self.tanh_ref()
        }

        pub fn asinh(self) -> Self {
            self.unary(BigFloat::asinh)
        }

        pub fn acosh(self) -> Self {
            self.unary(BigFloat::acosh)
        }

        pub fn atanh(self) -> Self {
            self.unary(BigFloat::atanh)
        }

        pub fn serialize(&self) -> Vec<u8> {
            if self.is_zero() {
                b"0".to_vec()
            } else {
                format_value(&self.value, Radix::Dec).into_bytes()
            }
        }
    }

    impl IntoMultiPrecisionFloat for MultiPrecisionFloat {
        fn into_float(mut self, prec: u32) -> MultiPrecisionFloat {
            self.set_prec(prec);
            self
        }
    }

    impl IntoMultiPrecisionFloat for &MultiPrecisionFloat {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            self.clone().into_float(prec)
        }
    }

    impl IntoMultiPrecisionFloat for f64 {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            MultiPrecisionFloat::from_f64(prec, self)
        }
    }

    impl IntoMultiPrecisionFloat for &f64 {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            (*self).into_float(prec)
        }
    }

    impl IntoMultiPrecisionFloat for f32 {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            MultiPrecisionFloat::from_f64(prec, self as f64)
        }
    }

    impl IntoMultiPrecisionFloat for i64 {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            let value = integer_to_float(self, prec);
            MultiPrecisionFloat { value, prec }
        }
    }

    impl IntoMultiPrecisionFloat for &i64 {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            (*self).into_float(prec)
        }
    }

    impl IntoMultiPrecisionFloat for i128 {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            let value = integer_to_float(self, prec);
            MultiPrecisionFloat { value, prec }
        }
    }

    impl IntoMultiPrecisionFloat for i32 {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            (self as i64).into_float(prec)
        }
    }

    impl IntoMultiPrecisionFloat for u64 {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            let value = integer_to_float(self, prec);
            MultiPrecisionFloat { value, prec }
        }
    }

    impl IntoMultiPrecisionFloat for u32 {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            (self as u64).into_float(prec)
        }
    }

    impl IntoMultiPrecisionFloat for usize {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            let value = integer_to_float(self, prec);
            MultiPrecisionFloat { value, prec }
        }
    }

    impl IntoMultiPrecisionFloat for MultiPrecisionInteger {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            let value = integer_to_float(self, prec);
            MultiPrecisionFloat { value, prec }
        }
    }

    impl IntoMultiPrecisionFloat for &MultiPrecisionInteger {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            let value = integer_to_float(self, prec);
            MultiPrecisionFloat { value, prec }
        }
    }

    impl IntoMultiPrecisionFloat for MalachiteRational {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            MultiPrecisionFloat::from_rational(prec, self)
        }
    }

    impl IntoMultiPrecisionFloat for &MalachiteRational {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            self.clone().into_float(prec)
        }
    }

    impl IntoMultiPrecisionFloat for Constant {
        fn into_float(self, prec: u32) -> MultiPrecisionFloat {
            match self {
                Constant::Pi => {
                    let value =
                        with_constants(|constants| constants.pi(precision(prec), ROUNDING_MODE));
                    MultiPrecisionFloat { value, prec }
                }
                Constant::Euler => MultiPrecisionFloat {
                    value: euler_mascheroni(prec),
                    prec,
                },
            }
        }
    }

    impl<T: IntoMultiPrecisionFloat> Assign<T> for MultiPrecisionFloat {
        fn assign_into(prec: u32, val: T) -> MultiPrecisionFloat {
            val.into_float(prec)
        }
    }

    impl Debug for MultiPrecisionFloat {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            Debug::fmt(&self.value, f)
        }
    }

    impl Display for MultiPrecisionFloat {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            self.fmt_decimal(f, false)
        }
    }

    impl std::fmt::LowerExp for MultiPrecisionFloat {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            self.fmt_decimal(f, true)
        }
    }

    impl PartialOrd for MultiPrecisionFloat {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            self.value.partial_cmp(&other.value)
        }
    }

    impl PartialEq<f64> for MultiPrecisionFloat {
        fn eq(&self, other: &f64) -> bool {
            self == &Self::with_val(self.prec, *other)
        }
    }

    impl PartialEq<i32> for MultiPrecisionFloat {
        fn eq(&self, other: &i32) -> bool {
            self == &Self::with_val(self.prec, *other)
        }
    }

    impl PartialEq<u32> for MultiPrecisionFloat {
        fn eq(&self, other: &u32) -> bool {
            self == &Self::with_val(self.prec, *other)
        }
    }

    impl Neg for MultiPrecisionFloat {
        type Output = Self;

        fn neg(self) -> Self::Output {
            Self {
                value: -self.value,
                prec: self.prec,
            }
        }
    }

    impl Pow<u64> for &MultiPrecisionFloat {
        type Output = MultiPrecisionFloat;

        fn pow(self, rhs: u64) -> Self::Output {
            self.pow_u64(rhs)
        }
    }

    impl Pow<u64> for MultiPrecisionFloat {
        type Output = MultiPrecisionFloat;

        fn pow(self, rhs: u64) -> Self::Output {
            self.pow_u64(rhs)
        }
    }

    impl Pow<&MultiPrecisionFloat> for MultiPrecisionFloat {
        type Output = MultiPrecisionFloat;

        fn pow(self, rhs: &MultiPrecisionFloat) -> Self::Output {
            MultiPrecisionFloat::pow(&self, rhs)
        }
    }

    impl Add<&MultiPrecisionFloat> for MultiPrecisionFloat {
        type Output = Self;

        fn add(self, rhs: &Self) -> Self::Output {
            let prec = self.prec.max(rhs.prec).max(1);
            let value = self.value.add(&rhs.value, precision(prec), ROUNDING_MODE);
            Self { value, prec }
        }
    }

    impl Add<MultiPrecisionFloat> for MultiPrecisionFloat {
        type Output = Self;

        fn add(self, rhs: Self) -> Self::Output {
            self + &rhs
        }
    }

    impl Add<f64> for MultiPrecisionFloat {
        type Output = Self;

        fn add(self, rhs: f64) -> Self::Output {
            let prec = self.prec;
            self + &Self::with_val(prec, rhs)
        }
    }

    impl Add<i64> for MultiPrecisionFloat {
        type Output = Self;

        fn add(self, rhs: i64) -> Self::Output {
            let prec = self.prec;
            self + &Self::with_val(prec, rhs)
        }
    }

    impl Add<i128> for MultiPrecisionFloat {
        type Output = Self;

        fn add(self, rhs: i128) -> Self::Output {
            let prec = self.prec;
            self + &Self::with_val(prec, rhs)
        }
    }

    impl Add<&MultiPrecisionInteger> for MultiPrecisionFloat {
        type Output = Self;

        fn add(self, rhs: &MultiPrecisionInteger) -> Self::Output {
            let prec = self.prec;
            self + &Self::parse_at_prec(&rhs.to_string(), prec).unwrap()
        }
    }

    impl Add<MultiPrecisionInteger> for MultiPrecisionFloat {
        type Output = Self;

        fn add(self, rhs: MultiPrecisionInteger) -> Self::Output {
            self + &rhs
        }
    }

    impl Sub<&MultiPrecisionFloat> for MultiPrecisionFloat {
        type Output = Self;

        fn sub(self, rhs: &Self) -> Self::Output {
            let prec = self.prec.max(rhs.prec).max(1);
            let value = self.value.sub(&rhs.value, precision(prec), ROUNDING_MODE);
            Self { value, prec }
        }
    }

    impl Sub<MultiPrecisionFloat> for MultiPrecisionFloat {
        type Output = Self;

        fn sub(self, rhs: Self) -> Self::Output {
            self - &rhs
        }
    }

    impl Sub<f64> for MultiPrecisionFloat {
        type Output = Self;

        fn sub(self, rhs: f64) -> Self::Output {
            let prec = self.prec;
            self - &Self::with_val(prec, rhs)
        }
    }

    impl Mul<&MultiPrecisionFloat> for MultiPrecisionFloat {
        type Output = Self;

        fn mul(self, rhs: &Self) -> Self::Output {
            let prec = self.prec.min(rhs.prec).max(1);
            let value = self.value.mul(&rhs.value, precision(prec), ROUNDING_MODE);
            Self { value, prec }
        }
    }

    impl Mul<MultiPrecisionFloat> for MultiPrecisionFloat {
        type Output = Self;

        fn mul(self, rhs: Self) -> Self::Output {
            self * &rhs
        }
    }

    impl Mul<f64> for MultiPrecisionFloat {
        type Output = Self;

        fn mul(self, rhs: f64) -> Self::Output {
            let prec = self.prec;
            self * &Self::with_val(prec, rhs)
        }
    }

    impl Mul<i64> for MultiPrecisionFloat {
        type Output = Self;

        fn mul(self, rhs: i64) -> Self::Output {
            let prec = self.prec;
            self * &Self::with_val(prec, rhs)
        }
    }

    impl Mul<i128> for MultiPrecisionFloat {
        type Output = Self;

        fn mul(self, rhs: i128) -> Self::Output {
            let prec = self.prec;
            self * &Self::with_val(prec, rhs)
        }
    }

    impl Mul<&MultiPrecisionInteger> for MultiPrecisionFloat {
        type Output = Self;

        fn mul(self, rhs: &MultiPrecisionInteger) -> Self::Output {
            let prec = self.prec;
            self * &Self::parse_at_prec(&rhs.to_string(), prec).unwrap()
        }
    }

    impl Mul<MultiPrecisionInteger> for MultiPrecisionFloat {
        type Output = Self;

        fn mul(self, rhs: MultiPrecisionInteger) -> Self::Output {
            self * &rhs
        }
    }

    impl Mul<MalachiteRational> for MultiPrecisionFloat {
        type Output = Self;

        fn mul(self, rhs: MalachiteRational) -> Self::Output {
            let prec = self.prec;
            self * &Self::with_val(prec, rhs)
        }
    }

    impl Div<&MultiPrecisionFloat> for MultiPrecisionFloat {
        type Output = Self;

        fn div(self, rhs: &Self) -> Self::Output {
            let prec = self.prec.min(rhs.prec).max(1);
            let value = self.value.div(&rhs.value, precision(prec), ROUNDING_MODE);
            Self { value, prec }
        }
    }

    impl Div<MultiPrecisionFloat> for MultiPrecisionFloat {
        type Output = Self;

        fn div(self, rhs: Self) -> Self::Output {
            self / &rhs
        }
    }

    impl Div<f64> for MultiPrecisionFloat {
        type Output = Self;

        fn div(self, rhs: f64) -> Self::Output {
            let prec = self.prec;
            self / &Self::with_val(prec, rhs)
        }
    }

    impl Div<i64> for MultiPrecisionFloat {
        type Output = Self;

        fn div(self, rhs: i64) -> Self::Output {
            let prec = self.prec;
            self / &Self::with_val(prec, rhs)
        }
    }

    impl Div<i128> for MultiPrecisionFloat {
        type Output = Self;

        fn div(self, rhs: i128) -> Self::Output {
            let prec = self.prec;
            self / &Self::with_val(prec, rhs)
        }
    }

    impl Div<&MultiPrecisionInteger> for MultiPrecisionFloat {
        type Output = Self;

        fn div(self, rhs: &MultiPrecisionInteger) -> Self::Output {
            let prec = self.prec;
            self / &Self::parse_at_prec(&rhs.to_string(), prec).unwrap()
        }
    }

    impl Div<MultiPrecisionInteger> for MultiPrecisionFloat {
        type Output = Self;

        fn div(self, rhs: MultiPrecisionInteger) -> Self::Output {
            self / &rhs
        }
    }

    impl Div<MalachiteRational> for MultiPrecisionFloat {
        type Output = Self;

        fn div(self, rhs: MalachiteRational) -> Self::Output {
            let prec = self.prec;
            self / &Self::with_val(prec, rhs)
        }
    }

    impl AddAssign<&MultiPrecisionFloat> for MultiPrecisionFloat {
        fn add_assign(&mut self, rhs: &MultiPrecisionFloat) {
            *self = self.clone() + rhs;
        }
    }

    impl AddAssign<MultiPrecisionFloat> for MultiPrecisionFloat {
        fn add_assign(&mut self, rhs: MultiPrecisionFloat) {
            *self += &rhs;
        }
    }

    impl AddAssign<f64> for MultiPrecisionFloat {
        fn add_assign(&mut self, rhs: f64) {
            *self = self.clone() + rhs;
        }
    }

    impl SubAssign<&MultiPrecisionFloat> for MultiPrecisionFloat {
        fn sub_assign(&mut self, rhs: &MultiPrecisionFloat) {
            *self = self.clone() - rhs;
        }
    }

    impl SubAssign<MultiPrecisionFloat> for MultiPrecisionFloat {
        fn sub_assign(&mut self, rhs: MultiPrecisionFloat) {
            *self -= &rhs;
        }
    }

    impl SubAssign<f64> for MultiPrecisionFloat {
        fn sub_assign(&mut self, rhs: f64) {
            *self = self.clone() - rhs;
        }
    }

    impl MulAssign<&MultiPrecisionFloat> for MultiPrecisionFloat {
        fn mul_assign(&mut self, rhs: &MultiPrecisionFloat) {
            *self = self.clone() * rhs;
        }
    }

    impl MulAssign<MultiPrecisionFloat> for MultiPrecisionFloat {
        fn mul_assign(&mut self, rhs: MultiPrecisionFloat) {
            *self *= &rhs;
        }
    }

    impl DivAssign<&MultiPrecisionFloat> for MultiPrecisionFloat {
        fn div_assign(&mut self, rhs: &MultiPrecisionFloat) {
            *self = self.clone() / rhs;
        }
    }

    impl DivAssign<MultiPrecisionFloat> for MultiPrecisionFloat {
        fn div_assign(&mut self, rhs: MultiPrecisionFloat) {
            *self /= &rhs;
        }
    }

    impl Add<MultiPrecisionFloat> for i64 {
        type Output = MultiPrecisionFloat;

        fn add(self, rhs: MultiPrecisionFloat) -> Self::Output {
            rhs + self
        }
    }

    impl Sub<MultiPrecisionFloat> for i64 {
        type Output = MultiPrecisionFloat;

        fn sub(self, rhs: MultiPrecisionFloat) -> Self::Output {
            MultiPrecisionFloat::with_val(rhs.prec, self) - rhs
        }
    }

    impl Mul<MultiPrecisionFloat> for i64 {
        type Output = MultiPrecisionFloat;

        fn mul(self, rhs: MultiPrecisionFloat) -> Self::Output {
            rhs * self
        }
    }

    impl Div<MultiPrecisionFloat> for i64 {
        type Output = MultiPrecisionFloat;

        fn div(self, rhs: MultiPrecisionFloat) -> Self::Output {
            MultiPrecisionFloat::with_val(rhs.prec, self) / rhs
        }
    }

    #[cfg(test)]
    mod binary_tests {
        use super::*;

        #[test]
        fn binary_parts_borrow_finite_limbs_and_exclude_special_values() {
            let value = MultiPrecisionFloat::with_val(53, -1.25);
            let (words, exponent) = value.binary_parts().unwrap();
            assert_eq!(
                words.as_ptr(),
                value.value.mantissa_digits().unwrap().as_ptr()
            );
            assert_eq!(exponent, 1);
            for value in [0., -0., f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
                assert!(
                    MultiPrecisionFloat::with_val(53, value)
                        .binary_parts()
                        .is_none()
                );
            }
            // Astro and the shared wire format both use one unsigned NaN.
            assert!(!MultiPrecisionFloat::with_val(53, -f64::NAN).is_sign_negative());
        }

        #[test]
        fn binary_decode_preserves_extra_astro_bits_at_requested_precision() {
            // Astro division uses 64 storage bits even with a 53-bit precision
            // hint. The final nonzero bits must survive binary decoding.
            let third = MultiPrecisionFloat::with_val(53, 1) / MultiPrecisionFloat::with_val(53, 3);
            let significand = 0xaaaa_aaaa_aaaa_aaabu64.to_le_bytes();
            for negative in [false, true] {
                let decoded =
                    MultiPrecisionFloat::from_binary_parts(53, negative, -1, &significand).unwrap();
                let expected = if negative {
                    -third.clone()
                } else {
                    third.clone()
                };
                assert_eq!(decoded, expected);
                assert_eq!(decoded.prec(), 53);
                assert_eq!(decoded.value.precision(), Some(64));
            }
        }

        #[test]
        fn binary_decode_preserves_dense_byte_boundaries() {
            use crate::domains::float::Float;

            for length in [1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 513] {
                let mut bytes: Vec<_> = (0..length).map(|i| (i * 37 + 1) as u8).collect();
                *bytes.last_mut().unwrap() |= 0x80;
                for precision in [length as u32 * 8, length as u32 * 8 + 64] {
                    for negative in [false, true] {
                        let decoded =
                            MultiPrecisionFloat::from_binary_parts(precision, negative, 1, &bytes)
                                .unwrap();
                        let mut expected = vec![if negative { 2 } else { 1 }, 2];
                        expected.extend_from_slice(&bytes);
                        assert_eq!(Float::from_raw(decoded).serialize(), expected);
                    }
                }
            }

            // An exactly byte-aligned subnormal also uses the word-copy path.
            let exponent = i64::from(astro_float::EXPONENT_MIN) - 8;
            let decoded = MultiPrecisionFloat::from_binary_parts(
                128,
                false,
                exponent,
                &[1, 2, 3, 4, 5, 6, 7, 8, 0x80],
            )
            .unwrap();
            let (words, bits, _, raw_exponent, _) = decoded.value.as_raw_parts().unwrap();
            assert_eq!(raw_exponent, astro_float::EXPONENT_MIN);
            assert_eq!(bits, 120);
            let stored: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
            let mut expected = vec![0; 16];
            expected[6..15].copy_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 0x80]);
            assert_eq!(stored, expected);
        }

        #[test]
        fn binary_decode_preserves_precision_at_word_boundaries() {
            for precision in [1, 7, 8, 24, 31, 32, 33, 53, 63, 64, 65, 127, 128, 129, 256] {
                for negative in [false, true] {
                    let decoded =
                        MultiPrecisionFloat::from_binary_parts(precision, negative, 1, &[0x80])
                            .unwrap();
                    assert_eq!(decoded.prec(), precision);
                    assert_eq!(decoded.to_f64(), if negative { -1. } else { 1. });
                    let storage_bits = (precision as usize).div_ceil(astro_float::WORD_BIT_SIZE)
                        * astro_float::WORD_BIT_SIZE;
                    assert_eq!(decoded.value.mantissa_max_bit_len(), Some(storage_bits));
                }
            }
        }

        #[test]
        fn binary_decode_preserves_astro_subnormals_and_exponent_endpoints() {
            for precision in [24, 53, 65, 128, 256] {
                let storage_bits = (precision as usize).div_ceil(astro_float::WORD_BIT_SIZE)
                    * astro_float::WORD_BIT_SIZE;
                let minimum_exponent = i64::from(astro_float::EXPONENT_MIN);
                for negative in [false, true] {
                    for (exponent, significand, mut expected) in [
                        (
                            minimum_exponent - storage_bits as i64 + 1,
                            vec![0x80],
                            BigFloat::min_positive(precision as usize),
                        ),
                        (
                            minimum_exponent,
                            vec![0x80],
                            BigFloat::min_positive_normal(precision as usize),
                        ),
                        (
                            i64::from(astro_float::EXPONENT_MAX),
                            vec![0xff; storage_bits / 8],
                            BigFloat::max_value(precision as usize),
                        ),
                    ] {
                        if negative {
                            expected.set_sign(Sign::Neg);
                        }
                        let decoded = MultiPrecisionFloat::from_binary_parts(
                            precision,
                            negative,
                            exponent,
                            &significand,
                        )
                        .unwrap();
                        assert_eq!(decoded.value, expected);
                        assert_eq!(decoded.prec(), precision);
                    }
                }

                // The smallest exact subnormals require discarding only zero
                // padding bits from the canonical byte, not rounding its value.
                for (significand, adjustment, expected_word) in [(0xc0, 2, 3), (0xa0, 3, 5)] {
                    let decoded = MultiPrecisionFloat::from_binary_parts(
                        precision,
                        false,
                        minimum_exponent - storage_bits as i64 + adjustment,
                        &[significand],
                    )
                    .unwrap();
                    let (words, exponent) = decoded.binary_parts().unwrap();
                    assert_eq!(exponent, minimum_exponent);
                    assert_eq!(words[0], expected_word);
                    assert!(words[1..].iter().all(|&word| word == 0));
                }
            }
        }

        #[test]
        fn binary_decode_shifts_subnormal_bytes_across_limb_boundaries() {
            let significand = [1, 2, 3, 4, 5, 6, 7, 8, 0x80];
            let decoded = MultiPrecisionFloat::from_binary_parts(
                128,
                false,
                i64::from(astro_float::EXPONENT_MIN) - 3,
                &significand,
            )
            .unwrap();
            let mut expected = ((0x80u128 << 64) | 0x0807_0605_0403_0201) << 53;
            let (words, exponent) = decoded.binary_parts().unwrap();
            assert_eq!(exponent, i64::from(astro_float::EXPONENT_MIN));
            for &word in words {
                assert_eq!(word, expected as astro_float::Word);
                expected >>= astro_float::WORD_BIT_SIZE;
            }
            assert_eq!(expected, 0);
        }

        #[test]
        fn binary_decode_rejects_invalid_or_inexact_finite_values() {
            for significand in [&[][..], &[0][..], &[1][..], &[0, 0x80][..], &[1, 0x7f][..]] {
                assert!(MultiPrecisionFloat::from_binary_parts(53, false, 0, significand).is_err());
            }
            assert!(MultiPrecisionFloat::from_binary_parts(0, false, 1, &[0x80]).is_err());
            for exponent in [i64::MAX, i64::MIN, i64::from(astro_float::EXPONENT_MAX) + 1] {
                assert!(
                    MultiPrecisionFloat::from_binary_parts(53, false, exponent, &[0x80]).is_err()
                );
            }
            let storage_bits =
                53usize.div_ceil(astro_float::WORD_BIT_SIZE) * astro_float::WORD_BIT_SIZE;
            let minimum_exponent = i64::from(astro_float::EXPONENT_MIN) - storage_bits as i64 + 1;
            assert!(
                MultiPrecisionFloat::from_binary_parts(53, false, minimum_exponent - 1, &[0x80])
                    .is_err()
            );
            assert!(
                MultiPrecisionFloat::from_binary_parts(53, false, minimum_exponent, &[0x81])
                    .is_err()
            );
        }
    }
}

#[cfg(feature = "float-astro")]
pub use astro::{
    Assign, BackendRational, CompleteRound, Constant, IntoMultiPrecisionFloat, MultiPrecisionFloat,
    Pow,
};
