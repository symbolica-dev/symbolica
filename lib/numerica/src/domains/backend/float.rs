use crate::domains::backend::integer::MultiPrecisionInteger;

#[cfg(feature = "gmp")]
pub use rug::{
    Assign, Float as MultiPrecisionFloat, Rational as BackendRational,
    float::Constant,
    ops::{CompleteRound, Pow},
};

pub trait BackendRationalExt {
    fn from_integer_ratio(num: MultiPrecisionInteger, den: MultiPrecisionInteger) -> Self;
    fn into_integer_ratio(self) -> (MultiPrecisionInteger, MultiPrecisionInteger);
}

#[cfg(feature = "gmp")]
impl BackendRationalExt for BackendRational {
    fn from_integer_ratio(num: MultiPrecisionInteger, den: MultiPrecisionInteger) -> Self {
        Self::from((num, den))
    }

    fn into_integer_ratio(self) -> (MultiPrecisionInteger, MultiPrecisionInteger) {
        self.into_numer_denom()
    }
}

#[cfg(feature = "no_gmp")]
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

    use super::BackendRationalExt;
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

    #[derive(Clone, PartialEq)]
    pub struct MultiPrecisionFloat {
        value: BigFloat,
        prec: u32,
    }

    impl Eq for MultiPrecisionFloat {}

    impl Hash for MultiPrecisionFloat {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.prec.hash(state);
            self.value.to_string().hash(state);
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
        let round_up = digits.as_bytes()[kept_digits] >= b'5';
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

    fn integer_to_float(value: impl Display, prec: u32) -> BigFloat {
        parse_integer_at_precision(value, precision(prec))
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
            Some(MalachiteRational::from_integer_ratio(
                num,
                MultiPrecisionInteger::from(1),
            ))
        } else {
            let den = MultiPrecisionInteger::from(1) << (-shift) as usize;
            Some(MalachiteRational::from_integer_ratio(mantissa, den))
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
            let value = with_constants(|constants| {
                BigFloat::parse(s, radix, precision(prec), ROUNDING_MODE, constants)
            });
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
                #[cfg(target_arch = "wasm32")]
                let mut value = with_constants(|constants| {
                    BigFloat::parse(
                        &value.to_string(),
                        Radix::Dec,
                        precision(prec).max(f64::MANTISSA_DIGITS as usize),
                        ROUNDING_MODE,
                        constants,
                    )
                });

                #[cfg(not(target_arch = "wasm32"))]
                let mut value =
                    BigFloat::from_f64(value, precision(prec).max(f64::MANTISSA_DIGITS as usize));

                let _ = value.set_precision(precision(prec), ROUNDING_MODE);
                value
            };
            Self { value, prec }
        }

        pub(crate) fn from_rational(prec: u32, value: MalachiteRational) -> Self {
            let (num, den) = value.into_integer_ratio();
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
            self.value.exponent()
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
            let (num, den) = value.into_integer_ratio();
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
            let x_cmp = rhs.cmp_zero();
            let y_cmp = self.cmp_zero();

            if x_cmp == Some(Ordering::Equal) {
                let pi = Self::with_val(prec, Constant::Pi);
                let half_pi = pi / 2i64;
                return if y_cmp == Some(Ordering::Less) {
                    -half_pi
                } else if y_cmp == Some(Ordering::Greater) {
                    half_pi
                } else {
                    Self::new(prec)
                };
            }

            if y_cmp == Some(Ordering::Equal) {
                return if x_cmp == Some(Ordering::Less) {
                    Self::with_val(prec, Constant::Pi)
                } else {
                    Self::new(prec)
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

    impl super::BackendRationalExt for BackendRational {
        fn from_integer_ratio(num: MultiPrecisionInteger, den: MultiPrecisionInteger) -> Self {
            format!("{num}/{den}").parse().unwrap()
        }

        fn into_integer_ratio(self) -> (MultiPrecisionInteger, MultiPrecisionInteger) {
            let value = self.to_string();
            if let Some((num, den)) = value.split_once('/') {
                (num.parse().unwrap(), den.parse().unwrap())
            } else {
                (value.parse().unwrap(), MultiPrecisionInteger::from(1))
            }
        }
    }

    impl Debug for MultiPrecisionFloat {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            Debug::fmt(&self.value, f)
        }
    }

    impl Display for MultiPrecisionFloat {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            Display::fmt(&format_value(&self.value, Radix::Dec), f)
        }
    }

    impl std::fmt::LowerExp for MultiPrecisionFloat {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            f.write_str(&format_lower_exp_value(&self.value, f.precision()))
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
}

#[cfg(feature = "no_gmp")]
pub use astro::{
    Assign, BackendRational, CompleteRound, Constant, IntoMultiPrecisionFloat, MultiPrecisionFloat,
    Pow,
};
