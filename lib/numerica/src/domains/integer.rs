//! Arbitrary precision integers.

use std::{
    cmp::Ordering,
    fmt::{Display, Error, Formatter},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, Sub, SubAssign},
    str::FromStr,
};

use rand::Rng;
use rug::{
    Complete, Integer as MultiPrecisionInteger,
    ops::{Pow, RemRounding},
};

use crate::{
    domains::{RingOps, Set},
    printer::{PrintOptions, PrintState},
    tensors::matrix::Matrix,
};

use super::{
    EuclideanDomain, InternalOrdering, Ring, SelfRing,
    finite_field::{
        FiniteField, FiniteFieldCore, FiniteFieldWorkspace, Mersenne64, ToFiniteField, Two, Z2, Zp,
        Zp64,
    },
    float::{FloatField, FloatLike, Real, RealLike, SingleFloat},
    rational::Rational,
};

/// The first 100 primes.
pub const SMALL_PRIMES: [i64; 100] = [
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
    101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
    197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307,
    311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421,
    431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541,
];

/// The integer ring.
pub type Z = IntegerRing;
/// The integer ring.
pub const Z: IntegerRing = IntegerRing::new();

/// The integer ring.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct IntegerRing;

impl Default for IntegerRing {
    fn default() -> Self {
        Self::new()
    }
}

impl IntegerRing {
    pub const fn new() -> IntegerRing {
        IntegerRing
    }
}

/// An arbitrary-precision integer that automatically upgrades and downgrades to the most efficient
/// representation.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Eq, Hash)]
pub enum Integer {
    /// Single machine-width integer (`i64`), hardware-accelerated on most platforms.
    Single(i64),
    /// Double machine-width integer (`i128`), partially hardware accelerated on some platforms.
    Double(i128),
    /// Multi-precision integer (using the `rug` crate).
    Large(MultiPrecisionInteger),
}

impl InternalOrdering for Integer {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        Ord::cmp(self, other)
    }
}

#[cfg(feature = "python")]
use pyo3::{
    Borrowed, Bound, FromPyObject, IntoPyObject, PyErr, PyResult, Python, exceptions, types::PyInt,
};

#[cfg(feature = "python_stubgen")]
impl_stub_type!(Integer = PyInt);

#[cfg(feature = "python")]
impl<'py> FromPyObject<'_, 'py> for Integer {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(num) = ob.extract::<i64>() {
            Ok(num.into())
        } else if let Ok(num) = ob.cast::<PyInt>() {
            let a = num.to_string();
            Ok(Integer::from(rug::Integer::parse(&a).unwrap().complete()))
        } else {
            Err(exceptions::PyValueError::new_err("Not a valid integer"))
        }
    }
}

#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for Integer {
    type Target = PyInt;
    type Output = Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            Integer::Single(n) => n.into_pyobject(py),
            Integer::Double(d) => d.into_pyobject(py),
            Integer::Large(l) => unsafe {
                Ok(Bound::from_owned_ptr(
                    py,
                    pyo3::ffi::PyLong_FromString(
                        l.to_string().as_str().as_ptr() as *const i8,
                        std::ptr::null_mut(),
                        10,
                    ),
                )
                .cast_into::<PyInt>()
                .unwrap())
            },
        }
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for Integer {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        match self {
            Integer::Single(val) => {
                0u8.encode(encoder)?;
                val.encode(encoder)
            }
            Integer::Double(val) => {
                1u8.encode(encoder)?;
                val.encode(encoder)
            }
            Integer::Large(val) => {
                2u8.encode(encoder)?;
                let bytes = val.to_digits::<u8>(rug::integer::Order::MsfBe);
                bytes.encode(encoder)
            }
        }
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(Integer);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for Integer {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let variant = u8::decode(decoder)?;
        match variant {
            0 => {
                let val = i64::decode(decoder)?;
                Ok(Integer::Single(val))
            }
            1 => {
                let val = i128::decode(decoder)?;
                Ok(Integer::Double(val))
            }
            2 => {
                let b = Vec::<u8>::decode(decoder)?;
                let val = MultiPrecisionInteger::from_digits(&b, rug::integer::Order::MsfBe);
                Ok(Integer::Large(val))
            }
            _ => Err(bincode::error::DecodeError::OtherString(format!(
                "Invalid variant for Integer: {}",
                variant
            ))),
        }
    }
}

/// An error that can occur when performing integer-reconstruction operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum IntegerRelationError {
    PrecisionLimit,
    IterationLimit(Vec<Integer>),
    CoefficientLimit,
}

macro_rules! from_with_i64_cast {
    ($base: ty) => {
        impl From<$base> for Integer {
            #[inline]
            fn from(value: $base) -> Self {
                Integer::Single(value as i64)
            }
        }

        impl PartialEq<$base> for Integer {
            #[inline]
            fn eq(&self, other: &$base) -> bool {
                match self {
                    Integer::Single(n) => *n == *other as i64,
                    _ => false,
                }
            }
        }

        impl PartialEq<Integer> for $base {
            #[inline]
            fn eq(&self, other: &Integer) -> bool {
                other == self
            }
        }

        impl PartialOrd<$base> for Integer {
            #[inline]
            fn partial_cmp(&self, other: &$base) -> Option<Ordering> {
                match self {
                    Integer::Single(n) => n.partial_cmp(&(*other as i64)),
                    x => {
                        if x.is_negative() {
                            Some(Ordering::Less)
                        } else {
                            Some(Ordering::Greater)
                        }
                    }
                }
            }
        }

        impl PartialOrd<Integer> for $base {
            #[inline]
            fn partial_cmp(&self, other: &Integer) -> Option<Ordering> {
                other.partial_cmp(self).map(|x| x.reverse())
            }
        }
    };
}

from_with_i64_cast!(i8);
from_with_i64_cast!(i16);
from_with_i64_cast!(i32);
from_with_i64_cast!(i64);
from_with_i64_cast!(u8);
from_with_i64_cast!(u16);
from_with_i64_cast!(u32);

macro_rules! cmp_with_conv {
    ($base: ty) => {
        impl PartialEq<$base> for Integer {
            #[inline]
            fn eq(&self, other: &$base) -> bool {
                self == &Integer::from(*other)
            }
        }

        impl PartialOrd<$base> for Integer {
            #[inline]
            fn partial_cmp(&self, other: &$base) -> Option<Ordering> {
                self.partial_cmp(&Integer::from(*other))
            }
        }

        impl PartialEq<Integer> for $base {
            #[inline]
            fn eq(&self, other: &Integer) -> bool {
                other == self
            }
        }

        impl PartialOrd<Integer> for $base {
            #[inline]
            fn partial_cmp(&self, other: &Integer) -> Option<Ordering> {
                other.partial_cmp(self).map(|x| x.reverse())
            }
        }
    };
}

impl TryFrom<Integer> for i64 {
    type Error = &'static str;

    #[inline]
    fn try_from(value: Integer) -> Result<Self, Self::Error> {
        match value {
            Integer::Single(n) => Ok(n),
            _ => Err("Could not convert to i64 as it is too large"),
        }
    }
}

impl TryFrom<Integer> for i128 {
    type Error = &'static str;

    #[inline]
    fn try_from(value: Integer) -> Result<Self, Self::Error> {
        match value {
            Integer::Single(n) => Ok(n as i128),
            Integer::Double(n) => Ok(n),
            _ => Err("Could not convert to i64 as it is too large"),
        }
    }
}

macro_rules! try_from_int_smaller_i64 {
    ($base: ty) => {
        impl TryFrom<Integer> for $base {
            type Error = &'static str;

            #[inline]
            fn try_from(value: Integer) -> Result<Self, Self::Error> {
                match value {
                    Integer::Single(n) => {
                        if n >= <$base>::MIN as i64 && n <= <$base>::MAX as i64 {
                            Ok(n as $base)
                        } else {
                            Err(concat!(
                                "Could not convert to ",
                                stringify!($base),
                                " as the integer is out of range"
                            ))
                        }
                    }
                    _ => Err(concat!(
                        "Could not convert to ",
                        stringify!($base),
                        " as the integer is too large"
                    )),
                }
            }
        }
    };
}

try_from_int_smaller_i64!(i8);
try_from_int_smaller_i64!(i16);
try_from_int_smaller_i64!(i32);
try_from_int_smaller_i64!(u8);
try_from_int_smaller_i64!(u16);
try_from_int_smaller_i64!(u32);

impl TryFrom<Integer> for isize {
    type Error = &'static str;

    #[inline]
    fn try_from(value: Integer) -> Result<Self, Self::Error> {
        if isize::BITS == 32 {
            i32::try_from(value).map(|x| x as isize)
        } else if usize::BITS == 64 {
            i64::try_from(value).map(|x| x as isize)
        } else if usize::BITS == 128 {
            i128::try_from(value).map(|x| x as isize)
        } else {
            Err("Could not convert to isize on this platform")
        }
    }
}

impl TryFrom<Integer> for usize {
    type Error = &'static str;

    #[inline]
    fn try_from(value: Integer) -> Result<Self, Self::Error> {
        if isize::BITS == 32 {
            u32::try_from(value).map(|x| x as usize)
        } else if usize::BITS == 64 {
            u64::try_from(value).map(|x| x as usize)
        } else if usize::BITS == 128 {
            u128::try_from(value).map(|x| x as usize)
        } else {
            Err("Could not convert to isize on this platform")
        }
    }
}

impl TryFrom<Integer> for u64 {
    type Error = &'static str;

    #[inline]
    fn try_from(value: Integer) -> Result<Self, Self::Error> {
        match value {
            Integer::Single(n) => {
                if n >= 0 {
                    Ok(n as u64)
                } else {
                    Err("Could not convert to u64 as it is negative")
                }
            }
            Integer::Double(n) => {
                if n >= 0 {
                    if n <= u64::MAX as i128 {
                        Ok(n as u64)
                    } else {
                        Err("Could not convert to u64 as it is too large")
                    }
                } else {
                    Err("Could not convert to u64 as it is negative")
                }
            }
            Integer::Large(_) => Err("Could not convert to u64 as it is too large"),
        }
    }
}

impl TryFrom<Integer> for u128 {
    type Error = &'static str;

    #[inline]
    fn try_from(value: Integer) -> Result<Self, Self::Error> {
        match value {
            Integer::Single(n) => {
                if n >= 0 {
                    Ok(n as u128)
                } else {
                    Err("Could not convert to u128 as it is negative")
                }
            }
            Integer::Double(n) => {
                if n >= 0 {
                    Ok(n as u128)
                } else {
                    Err("Could not convert to u128 as it is negative")
                }
            }
            Integer::Large(l) => l
                .to_u128()
                .ok_or("Could not convert to u128 as it is too large"),
        }
    }
}

impl From<i128> for Integer {
    #[inline]
    fn from(value: i128) -> Self {
        Integer::from_double(value)
    }
}

impl From<isize> for Integer {
    #[inline]
    fn from(value: isize) -> Self {
        if usize::BITS == 32 {
            Integer::Single(value as i64)
        } else if usize::BITS == 64 {
            Integer::Single(value as i64)
        } else if usize::BITS == 128 {
            Integer::from(value as i128)
        } else {
            Integer::from(MultiPrecisionInteger::from(value))
        }
    }
}

impl From<u64> for Integer {
    #[inline]
    fn from(value: u64) -> Self {
        if value <= i64::MAX as u64 {
            Integer::Single(value as i64)
        } else {
            Integer::Double(value as i128)
        }
    }
}

impl From<u128> for Integer {
    #[inline]
    fn from(value: u128) -> Self {
        if value <= i128::MAX as u128 {
            Integer::from_double(value as i128)
        } else {
            Integer::Large(value.into())
        }
    }
}

impl From<usize> for Integer {
    #[inline]
    fn from(value: usize) -> Self {
        if usize::BITS == 32 {
            Integer::Single(value as i64)
        } else if usize::BITS == 64 {
            Integer::from(value as u64)
        } else if usize::BITS == 128 {
            Integer::from(value as u128)
        } else {
            Integer::from(MultiPrecisionInteger::from(value))
        }
    }
}

impl From<MultiPrecisionInteger> for Integer {
    /// Convert from a multi-precision integer to an Integer, potentially
    /// downcasting the number.
    #[inline]
    fn from(n: MultiPrecisionInteger) -> Self {
        if let Some(n) = n.to_i64() {
            Integer::Single(n)
        } else if let Some(n) = n.to_i128() {
            Integer::Double(n)
        } else {
            Integer::Large(n)
        }
    }
}

cmp_with_conv!(i128);
cmp_with_conv!(isize);
cmp_with_conv!(u64);
cmp_with_conv!(u128);
cmp_with_conv!(usize);

impl FromStr for Integer {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        if s.len() <= 20
            && let Ok(n) = s.parse::<i64>()
        {
            return Ok(Integer::Single(n));
        }

        if s.len() <= 40
            && let Ok(n) = s.parse::<i128>()
        {
            return Ok(Integer::Double(n));
        }

        if let Ok(n) = s.parse::<MultiPrecisionInteger>() {
            Ok(Integer::Large(n))
        } else {
            Err("Could not parse integer")
        }
    }
}

impl std::fmt::Debug for Integer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(n) => std::fmt::Display::fmt(n, f),
            Self::Double(n) => std::fmt::Display::fmt(n, f),
            Self::Large(n) => std::fmt::Display::fmt(n, f),
        }
    }
}

impl ToFiniteField<u32> for Integer {
    fn to_finite_field(&self, field: &Zp) -> <Zp as Set>::Element {
        match self {
            Integer::Single(n) => field.to_element(n.rem_euclid(field.get_prime() as i64) as u32),
            Integer::Double(n) => field.to_element(n.rem_euclid(field.get_prime() as i128) as u32),
            Integer::Large(r) => field.to_element(r.mod_u(field.get_prime())),
        }
    }
}

impl ToFiniteField<u64> for Integer {
    fn to_finite_field(&self, field: &Zp64) -> <Zp64 as Set>::Element {
        match self {
            &Integer::Single(n) => {
                if field.get_prime() > i64::MAX as u64 {
                    field.to_element((n as i128).rem_euclid(field.get_prime() as i128) as u64)
                } else {
                    field.to_element(n.rem_euclid(field.get_prime() as i64) as u64)
                }
            }
            &Integer::Double(n) => field.to_element(n.rem_euclid(field.get_prime() as i128) as u64),
            Integer::Large(r) => {
                field.to_element(r.rem_euc(field.get_prime()).complete().to_u64().unwrap())
            }
        }
    }
}

impl ToFiniteField<Two> for Integer {
    fn to_finite_field(&self, field: &Z2) -> <Z2 as Set>::Element {
        match self {
            &Integer::Single(n) => field.to_element(Two(n.rem_euclid(2) as u8)),
            &Integer::Double(n) => field.to_element(Two(n.rem_euclid(2) as u8)),
            Integer::Large(r) => field.to_element(Two(r.mod_u(2) as u8)),
        }
    }
}

impl ToFiniteField<Integer> for Integer {
    fn to_finite_field(
        &self,
        field: &FiniteField<Integer>,
    ) -> <FiniteField<Integer> as Set>::Element {
        field.to_element(self.clone())
    }
}

impl ToFiniteField<Mersenne64> for Integer {
    fn to_finite_field(
        &self,
        _field: &FiniteField<Mersenne64>,
    ) -> <FiniteField<Mersenne64> as Set>::Element {
        match self {
            &Integer::Single(n) => n.rem_euclid(Mersenne64::PRIME as i64) as u64,
            &Integer::Double(n) => n.rem_euclid(Mersenne64::PRIME as i128) as u64,
            Integer::Large(r) => r.rem_euc(Mersenne64::PRIME).complete().to_u64().unwrap(),
        }
    }
}

pub trait FromFiniteField<UField: FiniteFieldWorkspace>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
{
    fn from_finite_field(
        field: &FiniteField<UField>,
        element: <FiniteField<UField> as Set>::Element,
    ) -> Self;
    fn from_prime(field: &FiniteField<UField>) -> Self;
}

impl FromFiniteField<u32> for Integer {
    fn from_finite_field(field: &Zp, element: <Zp as Set>::Element) -> Self {
        Integer::Single(field.from_element(&element) as i64)
    }

    fn from_prime(field: &Zp) -> Self {
        Integer::Single(field.get_prime() as i64)
    }
}

impl FromFiniteField<u64> for Integer {
    fn from_finite_field(field: &Zp64, element: <Zp64 as Set>::Element) -> Self {
        let r = field.from_element(&element);
        if r <= i64::MAX as u64 {
            Integer::Single(r as i64)
        } else {
            Integer::Double(r as i128)
        }
    }

    fn from_prime(field: &Zp64) -> Self {
        let r = field.get_prime();
        if r <= i64::MAX as u64 {
            Integer::Single(r as i64)
        } else {
            Integer::Double(r as i128)
        }
    }
}

impl FromFiniteField<Mersenne64> for Integer {
    fn from_finite_field(_field: &FiniteField<Mersenne64>, element: u64) -> Self {
        Integer::Single(element as i64)
    }

    fn from_prime(_field: &FiniteField<Mersenne64>) -> Self {
        Integer::Single(Mersenne64::PRIME as i64)
    }
}

impl Integer {
    pub fn new(num: i64) -> Integer {
        Integer::Single(num)
    }

    #[inline]
    fn simplify(&mut self) -> &mut Self {
        match self {
            Integer::Double(n) => {
                *self = Integer::from_double(*n);
            }
            Integer::Large(l) => {
                if let Some(n) = l.to_i64() {
                    *self = Integer::Single(n);
                } else if let Some(n) = l.to_i128() {
                    *self = Integer::Double(n);
                }
            }
            _ => {}
        }
        self
    }

    #[inline]
    pub fn from_double(n: i128) -> Integer {
        if n >= i64::MIN as i128 && n <= i64::MAX as i128 {
            Integer::Single(n as i64)
        } else {
            Integer::Double(n)
        }
    }

    #[inline]
    pub fn from_f64(f: f64) -> Integer {
        Self::from(MultiPrecisionInteger::from_f64(f).unwrap())
    }

    pub fn to_rational(&self) -> Rational {
        self.into()
    }

    pub fn to_multi_prec(self) -> MultiPrecisionInteger {
        match self {
            Integer::Single(n) => n.into(),
            Integer::Double(d) => d.into(),
            Integer::Large(l) => l,
        }
    }

    #[inline]
    pub const fn is_zero(&self) -> bool {
        match self {
            Integer::Single(n) => *n == 0,
            _ => false,
        }
    }

    #[inline]
    pub const fn is_one(&self) -> bool {
        match self {
            Integer::Single(n) => *n == 1,
            _ => false,
        }
    }

    #[inline]
    pub fn is_negative(&self) -> bool {
        match self {
            Integer::Single(n) => *n < 0,
            Integer::Double(n) => *n < 0,
            Integer::Large(r) => MultiPrecisionInteger::from(r.signum_ref()) == -1,
        }
    }

    #[inline]
    pub const fn zero() -> Integer {
        Integer::Single(0)
    }

    #[inline]
    pub const fn one() -> Integer {
        Integer::Single(1)
    }

    #[inline]
    pub fn to_i64(&self) -> Option<i64> {
        match self {
            Integer::Single(n) => Some(*n),
            _ => None,
        }
    }

    pub fn abs(&self) -> Integer {
        match self {
            Integer::Single(n) => {
                if *n == i64::MIN {
                    Integer::Double((*n as i128).abs())
                } else {
                    Integer::Single(n.abs())
                }
            }
            Integer::Double(n) => {
                if *n == i128::MIN {
                    Integer::Large(MultiPrecisionInteger::from(*n).abs())
                } else {
                    Integer::Double(n.abs())
                }
            }
            Integer::Large(n) => Integer::Large(n.clone().abs()),
        }
    }

    pub fn abs_cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (Integer::Large(n1), Integer::Large(n2)) => n1.as_abs().cmp(&n2.as_abs()),
            (Integer::Single(n1), Integer::Large(n2)) => n2
                .as_abs()
                .partial_cmp(&n1.unsigned_abs())
                .unwrap_or(Ordering::Equal)
                .reverse(),
            (Integer::Double(n1), Integer::Large(n2)) => n2
                .as_abs()
                .partial_cmp(&n1.unsigned_abs())
                .unwrap_or(Ordering::Equal)
                .reverse(),
            (Integer::Large(n1), Integer::Single(n2)) => n1
                .as_abs()
                .partial_cmp(&n2.unsigned_abs())
                .unwrap_or(Ordering::Equal),
            (Integer::Large(n1), Integer::Double(n2)) => n1
                .as_abs()
                .partial_cmp(&n2.unsigned_abs())
                .unwrap_or(Ordering::Equal),
            (_, _) => Ord::cmp(&self.abs(), &other.abs()),
        }
    }

    /// Compute `n` factorial (`n!`).
    pub fn factorial(n: u32) -> Integer {
        if n <= 20 {
            let mut f: i64 = 1;
            for x in 2..=n as i64 {
                f *= x;
            }
            Integer::Single(f)
        } else {
            Integer::Large(rug::Integer::factorial(n).complete())
        }
    }

    /// Compute the binomial coefficient `(n k) = n!/(k!(n-k)!)`.
    ///
    /// The implementation does not to overflow.
    pub fn binom(n: i64, mut k: i64) -> Integer {
        if n < 0 || k < 0 || k > n {
            return Integer::zero();
        }
        if k > n / 2 {
            k = n - k
        }
        let mut res = Integer::one();
        for i in 1..=k {
            res *= n - k + i;
            res /= i;
        }
        res
    }

    /// Compute the multinomial coefficient `(k_1+...+k_n)!/(k_1!*...*k_n!)`
    ///
    /// The implementation does not to overflow.
    pub fn multinom(k: &[u32]) -> Integer {
        let mut mcr = Integer::one();
        let mut accum = 0i64;
        for v in k {
            if let Some(res) = accum.checked_add(*v as i64) {
                accum = res;
            } else {
                panic!("Sum of occurrences exceeds i64: {k:?}");
            }

            mcr *= &Self::binom(accum, *v as i64);
        }
        mcr
    }

    pub fn pow(&self, e: u64) -> Integer {
        if e > u32::MAX as u64 {
            panic!("Power of exponentiation is larger than 2^32: {e}");
        }
        let e = e as u32;

        if e == 0 {
            return Integer::one();
        }

        match self {
            Integer::Single(n1) => {
                if let Some(pn) = n1.checked_pow(e) {
                    Integer::Single(pn)
                } else if let Some(pn) = (*n1 as i128).checked_pow(e) {
                    Integer::Double(pn)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*n1).pow(e))
                }
            }
            Integer::Double(n1) => {
                if let Some(pn) = n1.checked_pow(e) {
                    Integer::Double(pn)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*n1).pow(e))
                }
            }
            Integer::Large(r) => Integer::Large(r.pow(e).into()),
        }
    }

    pub fn quot_rem(&self, b: &Integer) -> (Integer, Integer) {
        if b.is_zero() {
            panic!("Cannot divide by zero");
        }

        match (self, b) {
            (Integer::Single(aa), Integer::Single(bb)) => {
                if let Some(q) = aa.checked_div_euclid(*bb) {
                    (Integer::Single(q), self - &(b * &Integer::Single(q)))
                } else {
                    (Integer::Double(-(i64::MIN as i128)), Integer::zero())
                }
            }
            (Integer::Single(a), Integer::Double(b)) => {
                // we always have |a| <= |b|
                if *a < 0 {
                    if *b > 0 {
                        (Integer::Single(-1), Integer::from_double(*a as i128 + *b))
                    } else {
                        (Integer::Single(1), Integer::from_double(*a as i128 - *b))
                    }
                } else {
                    (Integer::zero(), Integer::Single(*a))
                }
            }
            (Integer::Double(aa), Integer::Single(bb)) => {
                if let Some(q) = aa.checked_div_euclid(*bb as i128) {
                    let q = Integer::from_double(q);
                    (q.clone(), self - &(b * &q))
                } else {
                    (
                        Integer::Large(MultiPrecisionInteger::from(i128::MIN).neg()),
                        Integer::zero(),
                    )
                }
            }
            (Integer::Double(aa), Integer::Double(bb)) => {
                let q = Integer::from_double(aa.div_euclid(*bb)); // b != -1
                (q.clone(), self - &(b * &q))
            }
            (Integer::Single(a), Integer::Large(b)) => {
                if *a < 0 {
                    if *b > 0 {
                        (Integer::Single(-1), Integer::from((a + b).complete()))
                    } else {
                        (Integer::Single(1), Integer::from((a - b).complete()))
                    }
                } else {
                    (Integer::zero(), Integer::Single(*a))
                }
            }
            (Integer::Large(a), Integer::Single(b)) => {
                let r = a.clone().div_rem_euc(MultiPrecisionInteger::from(*b));
                (Integer::from(r.0), Integer::from(r.1))
            }
            (Integer::Large(a), Integer::Large(b)) => {
                let r = a.clone().div_rem_euc(b.clone());
                (Integer::from(r.0), Integer::from(r.1))
            }

            (Integer::Double(a), Integer::Large(b)) => {
                if *a < 0 {
                    if *b > 0 {
                        (Integer::Single(-1), Integer::from((a + b).complete()))
                    } else {
                        (Integer::Single(1), Integer::from((a - b).complete()))
                    }
                } else {
                    (Integer::zero(), Integer::Double(*a))
                }
            }
            (Integer::Large(a), Integer::Double(b)) => {
                let r = a.clone().div_rem_euc(MultiPrecisionInteger::from(*b));
                (Integer::from(r.0), Integer::from(r.1))
            }
        }
    }

    pub fn gcd(&self, b: &Integer) -> Integer {
        match (self, b) {
            (Integer::Single(n1), Integer::Single(n2)) => {
                let gcd = gcd_signed(*n1, *n2);
                if gcd == i64::MAX as u64 + 1 {
                    // n1 == n2 == u64::MIN
                    Integer::Double(gcd as i128)
                } else {
                    Integer::Single(gcd as i64)
                }
            }
            (Integer::Single(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Single(n1)) => {
                let r1 = MultiPrecisionInteger::from(*n1);
                Integer::from(r1.gcd(r2))
            }
            (Integer::Large(r1), Integer::Large(r2)) => Integer::from(r1.clone().gcd(r2)),
            (Integer::Single(r1), Integer::Double(r2))
            | (Integer::Double(r2), Integer::Single(r1)) => {
                Integer::from_double(gcd_signed_i128(*r1 as i128, *r2) as i128)
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                let gcd = gcd_signed_i128(*r1, *r2);
                if gcd == i128::MAX as u128 + 1 {
                    Integer::Large(MultiPrecisionInteger::from(gcd))
                } else {
                    Integer::from_double(gcd as i128)
                }
            }
            (Integer::Double(r1), Integer::Large(r2)) => {
                Integer::from(MultiPrecisionInteger::from(*r1).clone().gcd(r2))
            }
            (Integer::Large(r1), Integer::Double(r2)) => {
                Integer::from(r1.clone().gcd(&MultiPrecisionInteger::from(*r2)))
            }
        }
    }

    pub fn extended_gcd(&self, b: &Integer) -> (Integer, Integer, Integer) {
        match (self, b) {
            (Integer::Single(n1), Integer::Single(n2)) => {
                let (gcd, t, s) = extended_gcd(*n1, *n2);
                if gcd == i64::MAX as u64 + 1 {
                    (
                        Integer::Double(gcd as i128),
                        Integer::Single(t),
                        Integer::Single(s),
                    )
                } else {
                    (
                        Integer::Single(gcd as i64),
                        Integer::Single(t),
                        Integer::Single(s),
                    )
                }
            }
            (Integer::Single(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Single(n1)) => {
                let r1 = MultiPrecisionInteger::from(*n1);
                let (g, s, t) = r1.extended_gcd(r2.clone(), MultiPrecisionInteger::new());
                (Integer::from(g), Integer::from(s), Integer::from(t))
            }
            (Integer::Large(r1), Integer::Large(r2)) => {
                let (g, s, t) = r1
                    .clone()
                    .extended_gcd(r2.clone(), MultiPrecisionInteger::new());
                (Integer::from(g), Integer::from(s), Integer::from(t))
            }
            (Integer::Single(r1), Integer::Double(r2))
            | (Integer::Double(r2), Integer::Single(r1)) => {
                let (gcd, t, s) = extended_gcd_i128(*r1 as i128, *r2);
                (
                    Integer::from_double(gcd as i128),
                    Integer::from_double(t),
                    Integer::from_double(s),
                )
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                let (g, t, s) = extended_gcd_i128(*r1, *r2);
                if g == i128::MAX as u128 + 1 {
                    (
                        Integer::Large(MultiPrecisionInteger::from(g)),
                        Integer::from_double(t),
                        Integer::from_double(s),
                    )
                } else {
                    (
                        Integer::from_double(g as i128),
                        Integer::from_double(t),
                        Integer::from_double(s),
                    )
                }
            }
            (Integer::Double(r1), Integer::Large(r2)) => {
                let (g, s, t) = MultiPrecisionInteger::from(*r1)
                    .clone()
                    .extended_gcd(r2.clone(), MultiPrecisionInteger::new());
                (Integer::from(g), Integer::from(s), Integer::from(t))
            }
            (Integer::Large(r1), Integer::Double(r2)) => {
                let (g, s, t) = r1.clone().extended_gcd(
                    MultiPrecisionInteger::from(*r2),
                    MultiPrecisionInteger::new(),
                );
                (Integer::from(g), Integer::from(s), Integer::from(t))
            }
        }
    }

    /// Compute the least common multiple of two integers.
    pub fn lcm(&self, b: &Integer) -> Integer {
        let g = self.gcd(b);
        if g.is_zero() {
            Integer::zero()
        } else {
            (self / &g) * b
        }
    }

    /// Use Garner's algorithm for the Chinese remainder theorem
    /// to reconstruct an `x` that satisfies `n1 = x % p1` and `n2 = x % p2`.
    /// The `x` will be in the range `[-p1*p2/2,p1*p2/2]`.
    pub fn chinese_remainder(
        mut n1: Integer,
        mut n2: Integer,
        p1: Integer,
        p2: Integer,
    ) -> Integer {
        if n1 < 0 {
            n1 += &p1;
        }
        if n2 < 0 {
            n2 += &p2;
        }

        // make sure n1 < n2
        if match (&n1, &n2) {
            (Integer::Single(n1), Integer::Single(n2)) => n1 > n2,
            (Integer::Single(_), Integer::Large(_)) => false,
            (Integer::Single(_), Integer::Double(_)) => false,
            (Integer::Double(_), Integer::Single(_)) => true,
            (Integer::Double(_), Integer::Large(_)) => false,
            (Integer::Large(_), Integer::Single(_)) => true,
            (Integer::Large(_), Integer::Double(_)) => true,
            (Integer::Double(n1), Integer::Double(n2)) => n1 > n2,
            (Integer::Large(r1), Integer::Large(r2)) => r1 > r2,
        } {
            return Self::chinese_remainder(n2, n1, p2, p1);
        }

        let p1 = match p1 {
            Integer::Single(n) => MultiPrecisionInteger::from(n),
            Integer::Double(n) => MultiPrecisionInteger::from(n),
            Integer::Large(r) => r,
        };
        let p2 = match p2 {
            Integer::Single(n) => MultiPrecisionInteger::from(n),
            Integer::Double(n) => MultiPrecisionInteger::from(n),
            Integer::Large(r) => r,
        };

        let n1 = match n1 {
            Integer::Single(n) => MultiPrecisionInteger::from(n),
            Integer::Double(n) => MultiPrecisionInteger::from(n),
            Integer::Large(r) => r,
        };
        let n2 = match n2 {
            Integer::Single(n) => MultiPrecisionInteger::from(n),
            Integer::Double(n) => MultiPrecisionInteger::from(n),
            Integer::Large(r) => r,
        };

        // convert to mixed-radix notation
        let gamma1 = (p1.clone() % p2.clone())
            .invert(&p2)
            .unwrap_or_else(|_| panic!("Could not invert {p1} in {p2}"));

        let v1 = ((n2 - n1.clone()) * gamma1) % p2.clone();

        // convert to standard representation
        let r = v1 * p1.clone() + n1;

        let res = if r.clone() * 2 > p1.clone() * p2.clone() {
            r - p1 * p2
        } else {
            r
        };

        Integer::from(res)
    }

    /// Perform the symmetric mod `p` on `self`.
    #[inline]
    pub fn symmetric_mod(self, p: &Integer) -> Integer {
        let c = self % p;

        if &c + &c > *p { c - p } else { c }
    }

    /// Compute the modular inverse of `self` in the ring with size `n`.
    /// `self` and `n` must be coprime.
    pub fn mod_inverse(&self, n: &Integer) -> Integer {
        let mut t0 = Integer::zero();
        let mut t1 = Integer::one();
        let mut r0 = n.clone();
        let mut r1 = self.clone();

        while !r1.is_zero() {
            let (q, r) = Z.quot_rem(&r0, &r1);
            (t1, t0) = (&t0 - &(&q * &t1), t1);
            (r1, r0) = (r, r1);
        }

        if r0 > Integer::one() {
            panic!("{self} is not invertible in ring {n}");
        }
        if t0.is_negative() {
            t0 += n;
        }

        t0
    }

    /// Use the PSLQ algorithm to find a vector of integers `a` that satisfies `a.x = 0`,
    /// where every element of `a` is less than `max_coeff`, using a specified tolerance and number
    /// of iterations. The parameter `gamma` must be more than or equal to `2/sqrt(3)`.
    ///
    /// If the procedure runs out of iterations, the current best solution is returned.
    pub fn solve_integer_relation<
        T: SelfRing
            + FloatLike
            + RealLike
            + Real
            + SingleFloat
            + std::hash::Hash
            + Eq
            + InternalOrdering
            + PartialOrd,
    >(
        x: &[T],
        tolerance: T,
        max_iter: usize,
        max_coeff: Option<Integer>,
        gamma: Option<T>,
    ) -> Result<Vec<Integer>, IntegerRelationError> {
        let gamma = gamma.unwrap_or_else(|| x[0].from_usize(2) / x[0].from_usize(3).sqrt());
        let field = FloatField::from_rep(x[0].clone());
        let n = x.len();

        let mut s = Vec::with_capacity(n);
        let mut sum = x[0].zero();
        for xx in x.iter().rev() {
            sum += xx.clone() * xx;
            s.push(sum.sqrt());
        }
        s.reverse();

        // normalize the input
        let t = s[0].clone();
        let mut y = x
            .iter()
            .map(|xx| xx.clone() / t.clone())
            .collect::<Vec<_>>();
        for ss in &mut s {
            *ss /= &t;
        }

        // construct orthogonal matrix h
        let mut h = Matrix::new(n as u32, n as u32 - 1, field.clone());
        for i in 0..n {
            if i < n - 1 {
                h[(i as u32, i as u32)] = s[i + 1].clone() / &s[i];
            }

            for j in 0..i {
                h[(i as u32, j as u32)] = -y[i].clone() * &y[j] / (s[j].clone() * &s[j + 1]);
            }
        }

        let mut b = Matrix::identity(n as u32, IntegerRing);

        macro_rules! hermite_reduction {
            ($i: expr, $j: expr) => {
                if SelfRing::is_zero(&h[($j, $j)]) {
                    return Err(IntegerRelationError::PrecisionLimit);
                }

                let t = (h[($i, $j)].clone() / &h[($j, $j)]).round_to_nearest_integer();

                if t.is_zero() {
                    continue;
                }

                let t_f = x[0].from_rational(&(t.clone(), Integer::one()).into());
                let r = t_f.clone() * &y[$i as usize];
                y[$j as usize] += r;

                for k in 0..$j + 1 {
                    let r = t_f.clone() * &h[($j, k)];
                    h[($i, k)] -= r;
                }
                for k in 0..n as u32 {
                    let r = t.clone() * &b[(k, $i)];
                    b[(k, $j)] += r;
                }
            };
        }

        // Hermite reduction
        for i in 1..n as u32 {
            for j in (0..i).rev() {
                hermite_reduction!(i, j);
            }
        }

        for i in 0..max_iter {
            let mut gamma_max = gamma.clone();
            let ms: Vec<_> = (0..n as u32 - 1)
                .map(|i| {
                    if !h[(i, i)].is_finite() {
                        return Err(IntegerRelationError::PrecisionLimit);
                    }

                    let r = h[(i, i)].norm() * &gamma_max;
                    gamma_max *= &gamma;
                    Ok((i, r))
                })
                .collect::<Result<_, _>>()?;
            let m = ms
                .iter()
                .max_by(|(_, x), (_, y)| x.partial_cmp(y).unwrap())
                .unwrap()
                .0;

            y.swap(m as usize, m as usize + 1);
            h.swap_rows(m, m + 1, 0);
            b.swap_cols(m, m + 1);

            // make h lower trapezoidal
            if m < n as u32 - 2 {
                let t0 = (h[(m, m)].clone() * &h[(m, m)] + h[(m, m + 1)].clone() * &h[(m, m + 1)])
                    .sqrt();

                if SelfRing::is_zero(&t0) {
                    return Err(IntegerRelationError::PrecisionLimit);
                }

                let t1 = h[(m, m)].clone() / &t0;
                let t2 = h[(m, m + 1)].clone() / &t0;
                for i in m..n as u32 {
                    let t3 = h[(i, m)].clone();
                    let t4 = h[(i, m + 1)].clone();
                    h[(i, m)] = t1.clone() * t3.clone() + t2.clone() * t4.clone();
                    h[(i, m + 1)] = -t2.clone() * t3.clone() + t1.clone() * t4.clone();
                }
            }

            for i in (m + 1)..n as u32 {
                for j in (0..i.min(m + 2)).rev() {
                    hermite_reduction!(i, j);
                }
            }

            if let Some(i) = y.iter().position(|yy| yy.norm() < tolerance) {
                let res = (0..n as u32)
                    .map(|j| b[(j, i as u32)].clone())
                    .collect::<Vec<_>>();

                if let Some(max_coeff) = &max_coeff {
                    if res.iter().all(|r| &r.abs() <= max_coeff) {
                        return Ok(res);
                    }
                } else {
                    return Ok(res);
                }
            }

            // check the norm of the largest element once in a while
            if let Some(max_coeff) = &max_coeff
                && i % 20 == 0
            {
                let mut norm = x[0].zero();
                for i in 0..n as u32 {
                    let mut row_norm_sq = x[0].zero();
                    for j in 0..n as u32 - 1 {
                        row_norm_sq += h[(i, j)].clone() * &h[(i, j)];
                    }

                    if row_norm_sq > norm {
                        norm = row_norm_sq;
                    }
                }

                if &norm.sqrt().inv().round_to_nearest_integer() > max_coeff {
                    return Err(IntegerRelationError::CoefficientLimit);
                }
            }
        }

        // return the best estimate
        let min = y
            .iter()
            .enumerate()
            .min_by(|(_, xx), (_, yy)| {
                xx.norm()
                    .partial_cmp(&yy.norm())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap()
            .0;
        let res = (0..n as u32)
            .map(|j| b[(j, min as u32)].clone())
            .collect::<Vec<_>>();

        Err(IntegerRelationError::IterationLimit(res))
    }
}

impl Display for Integer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Integer::Single(n) => n.fmt(f),
            Integer::Double(n) => n.fmt(f),
            Integer::Large(r) => r.fmt(f),
        }
    }
}

impl Display for IntegerRing {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl PartialOrd for Integer {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        match (self, other) {
            (Integer::Single(n1), Integer::Single(n2)) => n1.partial_cmp(n2),
            (Integer::Single(n1), Integer::Large(n2)) => n1.partial_cmp(n2),
            (Integer::Large(n1), Integer::Single(n2)) => n1.partial_cmp(n2),
            (Integer::Large(n1), Integer::Large(n2)) => n1.partial_cmp(n2),
            (Integer::Single(n1), Integer::Double(n2)) => (*n1 as i128).partial_cmp(n2),
            (Integer::Double(n1), Integer::Single(n2)) => n1.partial_cmp(&(*n2 as i128)),
            (Integer::Double(n1), Integer::Double(n2)) => n1.partial_cmp(n2),
            (Integer::Double(n1), Integer::Large(n2)) => n1.partial_cmp(n2),
            (Integer::Large(n1), Integer::Double(n2)) => n1.partial_cmp(n2),
        }
    }
}

impl Ord for Integer {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl Set for IntegerRing {
    type Element = Integer;

    fn size(&self) -> Option<Integer> {
        None
    }
}

impl RingOps<<Self as Set>::Element> for IntegerRing {
    #[inline]
    fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        a + b
    }

    #[inline]
    fn sub(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        a - b
    }

    #[inline]
    fn mul(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        a * b
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a += b;
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a -= b;
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a *= b;
    }

    #[inline(always)]
    fn add_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        a.add_assign(b * c)
    }

    #[inline(always)]
    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        a.sub_assign(b * c)
    }

    #[inline]
    fn neg(&self, a: Self::Element) -> Self::Element {
        a.neg()
    }
}

impl RingOps<&<Self as Set>::Element> for IntegerRing {
    #[inline]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone() + b
    }

    #[inline]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone() - b
    }

    #[inline]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone() * b
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a += b;
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a -= b;
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a *= b;
    }

    #[inline(always)]
    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        if let Integer::Large(l) = a {
            // prevent the creation of a GMP integer b * c
            match (b, c) {
                (Integer::Single(b1), Integer::Large(c1)) => l.add_assign(b1 * c1),
                (Integer::Double(b1), Integer::Large(c1)) => l.add_assign(b1 * c1),
                (Integer::Large(b1), Integer::Single(c1)) => l.add_assign(b1 * c1),
                (Integer::Large(b1), Integer::Double(c1)) => l.add_assign(b1 * c1),
                (Integer::Large(b1), Integer::Large(c1)) => l.add_assign(b1 * c1),
                _ => {
                    return *a += b * c;
                }
            }

            a.simplify();
            return;
        }

        *a += b * c;
    }

    #[inline(always)]
    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        if let Integer::Large(l) = a {
            match (b, c) {
                (Integer::Single(b1), Integer::Large(c1)) => l.sub_assign(b1 * c1),
                (Integer::Double(b1), Integer::Large(c1)) => l.sub_assign(b1 * c1),
                (Integer::Large(b1), Integer::Single(c1)) => l.sub_assign(b1 * c1),
                (Integer::Large(b1), Integer::Double(c1)) => l.sub_assign(b1 * c1),
                (Integer::Large(b1), Integer::Large(c1)) => l.sub_assign(b1 * c1),
                _ => {
                    return *a -= b * c;
                }
            }

            a.simplify();
            return;
        }

        *a -= b * c;
    }

    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        a.clone().neg()
    }
}

impl Ring for IntegerRing {
    #[inline]
    fn zero(&self) -> Self::Element {
        Integer::zero()
    }

    #[inline]
    fn one(&self) -> Self::Element {
        Integer::one()
    }

    #[inline]
    fn nth(&self, n: Integer) -> Self::Element {
        n
    }

    #[inline]
    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        b.pow(e)
    }

    #[inline]
    fn is_zero(&self, a: &Self::Element) -> bool {
        match a {
            Integer::Single(r) => *r == 0,
            Integer::Double(_) => false,
            Integer::Large(_) => false,
        }
    }

    #[inline]
    fn is_one(&self, a: &Self::Element) -> bool {
        match a {
            Integer::Single(r) => *r == 1,
            Integer::Double(_) => false,
            Integer::Large(_) => false,
        }
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        0.into()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        match a {
            Integer::Single(1) | Integer::Single(-1) => Some(a.clone()),
            _ => None,
        }
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if b.is_zero() {
            return None;
        }

        let r = a / b;
        if *a == &r * b { Some(r) } else { None }
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.random_range(range.0..range.1);
        Integer::Single(r)
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        element.format(opts, state, f)
    }

    fn has_independent_elements(&self) -> bool {
        true
    }
}

impl SelfRing for Integer {
    fn is_zero(&self) -> bool {
        self.is_zero()
    }

    fn is_one(&self) -> bool {
        self.is_one()
    }

    fn format<W: std::fmt::Write>(
        &self,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        match self {
            Integer::Single(n) => {
                if state.suppress_one {
                    if *n == 1 {
                        if state.in_sum {
                            write!(f, "+")?;
                            return Ok(true);
                        } else {
                            write!(f, "")?;
                            return Ok(true);
                        }
                    } else if *n == -1 {
                        write!(f, "-")?;
                        return Ok(true);
                    }
                }

                if state.in_sum {
                    write!(f, "{n:+}")?
                } else {
                    write!(f, "{n}")?
                }
            }
            Integer::Double(n) => {
                if state.in_sum {
                    write!(f, "{n:+}")?
                } else {
                    write!(f, "{n}")?
                }
            }
            Integer::Large(r) => {
                if opts.explicit_rational_polynomial {
                    // write the GMP number in hexadecimal representation,
                    // since the conversion is much faster than for the decimal representation
                    if r.is_negative() {
                        write!(f, "-#{:X}", r.as_abs())?
                    } else if state.in_sum {
                        write!(f, "+#{r:X}")?
                    } else {
                        write!(f, "#{r:X}")?
                    }
                } else if state.in_sum {
                    write!(f, "{r:+}")?
                } else {
                    write!(f, "{r}")?
                }
            }
        }

        Ok(false)
    }
}

impl EuclideanDomain for IntegerRing {
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a % b
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        a.quot_rem(b)
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.gcd(b)
    }
}

impl<'b> Add<&'b Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn add(self, rhs: &'b Integer) -> Integer {
        if let Integer::Large(r) = self {
            match rhs {
                Integer::Single(n) => Integer::from(*n + r),
                Integer::Double(n) => Integer::from(*n + r),
                Integer::Large(n) => Integer::from(n + r),
            }
        } else {
            &self + rhs
        }
    }
}

impl Add<Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn add(self, rhs: Integer) -> Integer {
        if let Integer::Large(r) = self {
            match rhs {
                Integer::Single(n) => Integer::from(n + r),
                Integer::Double(n) => Integer::from(n + r),
                Integer::Large(n) => Integer::from(n + r),
            }
        } else if let Integer::Large(r) = rhs {
            match self {
                Integer::Single(n) => Integer::from(n + r),
                Integer::Double(n) => Integer::from(n + r),
                Integer::Large(n) => Integer::from(n + r),
            }
        } else {
            self + &rhs
        }
    }
}

impl Add<Integer> for &Integer {
    type Output = Integer;

    #[inline(always)]
    fn add(self, rhs: Integer) -> Integer {
        rhs + self
    }
}

impl<'b> Add<&'b Integer> for &Integer {
    type Output = Integer;

    #[inline(always)]
    fn add(self, rhs: &'b Integer) -> Integer {
        match (self, rhs) {
            (Integer::Single(n1), Integer::Single(n2)) => {
                if let Some(num) = n1.checked_add(*n2) {
                    Integer::Single(num)
                } else {
                    Integer::Double(*n1 as i128 + *n2 as i128)
                }
            }
            (Integer::Single(n1), Integer::Double(r2))
            | (Integer::Double(r2), Integer::Single(n1)) => {
                if let Some(num) = (*n1 as i128).checked_add(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*r2) + *n1)
                }
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                if let Some(num) = r1.checked_add(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*r1) + *r2)
                }
            }
            (Integer::Single(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Single(n1)) => Integer::from((*n1 + r2).complete()),
            (Integer::Double(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Double(n1)) => Integer::from((*n1 + r2).complete()),
            (Integer::Large(r1), Integer::Large(r2)) => Integer::from((r1 + r2).complete()),
        }
    }
}

impl Sub<&Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn sub(self, rhs: &Integer) -> Integer {
        if let Integer::Large(s) = self {
            match rhs {
                Integer::Single(r) => Integer::from(s - r),
                Integer::Double(r) => Integer::from(s - r),
                Integer::Large(r) => Integer::from(s - r),
            }
        } else {
            &self - rhs
        }
    }
}

impl Sub<Integer> for &Integer {
    type Output = Integer;

    #[inline(always)]
    fn sub(self, rhs: Integer) -> Integer {
        if let Integer::Large(r) = rhs {
            match self {
                Integer::Single(s) => Integer::from(*s - r),
                Integer::Double(s) => Integer::from(*s - r),
                Integer::Large(s) => Integer::from(s - r),
            }
        } else {
            self - &rhs
        }
    }
}

impl Sub<Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn sub(self, rhs: Integer) -> Integer {
        if let Integer::Large(s) = self {
            match rhs {
                Integer::Single(r) => Integer::from(s - r),
                Integer::Double(r) => Integer::from(s - r),
                Integer::Large(r) => Integer::from(s - r),
            }
        } else if let Integer::Large(r) = rhs {
            match self {
                Integer::Single(s) => Integer::from(s - r),
                Integer::Double(s) => Integer::from(s - r),
                Integer::Large(s) => Integer::from(s - r),
            }
        } else {
            self - &rhs
        }
    }
}

impl<'b> Sub<&'b Integer> for &Integer {
    type Output = Integer;

    #[inline(always)]
    fn sub(self, rhs: &'b Integer) -> Integer {
        match (self, rhs) {
            (Integer::Single(n1), Integer::Single(n2)) => {
                if let Some(num) = n1.checked_sub(*n2) {
                    Integer::Single(num)
                } else {
                    Integer::Double(*n1 as i128 - *n2 as i128)
                }
            }
            (Integer::Single(n1), Integer::Double(r2)) => {
                if let Some(num) = (*n1 as i128).checked_sub(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*n1) - *r2)
                }
            }
            (Integer::Double(r1), Integer::Single(r2)) => {
                if let Some(num) = r1.checked_sub(*r2 as i128) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*r1) - *r2)
                }
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                if let Some(num) = r1.checked_sub(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*r1) - *r2)
                }
            }
            (Integer::Single(n1), Integer::Large(r2)) => Integer::from((*n1 - r2).complete()),
            (Integer::Large(r1), Integer::Single(n2)) => Integer::from((r1 - *n2).complete()),
            (Integer::Double(n1), Integer::Large(r2)) => Integer::from((*n1 - r2).complete()),
            (Integer::Large(r1), Integer::Double(n2)) => Integer::from((r1 - *n2).complete()),
            (Integer::Large(r1), Integer::Large(r2)) => Integer::from((r1 - r2).complete()),
        }
    }
}

impl<'a> Mul<&'a Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn mul(self, rhs: &'a Integer) -> Integer {
        if let Integer::Large(r) = self {
            match rhs {
                Integer::Single(n) => Integer::from(*n * r),
                Integer::Double(n) => Integer::from(*n * r),
                Integer::Large(n) => Integer::from(n * r),
            }
        } else {
            &self * rhs
        }
    }
}

impl Mul<Integer> for &Integer {
    type Output = Integer;

    #[inline(always)]
    fn mul(self, rhs: Integer) -> Integer {
        rhs * self
    }
}

impl Mul<Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn mul(self, rhs: Integer) -> Integer {
        if let Integer::Large(r) = self {
            match rhs {
                Integer::Single(n) => Integer::from(n * r),
                Integer::Double(n) => Integer::from(n * r),
                Integer::Large(n) => Integer::from(n * r),
            }
        } else if let Integer::Large(r) = rhs {
            match self {
                Integer::Single(n) => Integer::from(n * r),
                Integer::Double(n) => Integer::from(n * r),
                Integer::Large(n) => Integer::from(n * r),
            }
        } else {
            self * &rhs
        }
    }
}

impl<'b> Mul<&'b Integer> for &Integer {
    type Output = Integer;

    #[inline(always)]
    fn mul(self, rhs: &'b Integer) -> Integer {
        match (self, rhs) {
            (Integer::Single(n1), Integer::Single(n2)) => {
                if let Some(num) = n1.checked_mul(*n2) {
                    Integer::Single(num)
                } else {
                    Integer::Double(*n1 as i128 * *n2 as i128)
                }
            }
            (Integer::Single(n1), Integer::Double(r2))
            | (Integer::Double(r2), Integer::Single(n1)) => {
                if let Some(num) = (*n1 as i128).checked_mul(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*r2) * *n1)
                }
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                if let Some(num) = r1.checked_mul(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*r1) * *r2)
                }
            }
            (Integer::Single(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Single(n1)) => Integer::from((n1 * r2).complete()),
            (Integer::Double(n1), Integer::Large(r2))
            | (Integer::Large(r2), Integer::Double(n1)) => Integer::from((n1 * r2).complete()),
            (Integer::Large(r1), Integer::Large(r2)) => Integer::from((r1 * r2).complete()),
        }
    }
}

impl Div<&Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn div(self, rhs: &Integer) -> Integer {
        if let Integer::Large(s) = self {
            match rhs {
                Integer::Single(r) => Integer::from(s / r),
                Integer::Double(r) => Integer::from(s / r),
                Integer::Large(r) => Integer::from(s / r),
            }
        } else {
            &self / rhs
        }
    }
}

impl Div<Integer> for &Integer {
    type Output = Integer;

    #[inline(always)]
    fn div(self, rhs: Integer) -> Integer {
        if let Integer::Large(r) = rhs {
            match self {
                Integer::Single(s) => Integer::from(*s / r),
                Integer::Double(s) => Integer::from(*s / r),
                Integer::Large(s) => Integer::from(s / r),
            }
        } else {
            self / &rhs
        }
    }
}

impl Div<Integer> for Integer {
    type Output = Integer;

    #[inline(always)]
    fn div(self, rhs: Integer) -> Integer {
        if let Integer::Large(s) = self {
            match rhs {
                Integer::Single(r) => Integer::from(s / r),
                Integer::Double(r) => Integer::from(s / r),
                Integer::Large(r) => Integer::from(s / r),
            }
        } else if let Integer::Large(r) = rhs {
            match self {
                Integer::Single(s) => Integer::from(s / r),
                Integer::Double(s) => Integer::from(s / r),
                Integer::Large(s) => Integer::from(s / r),
            }
        } else {
            self / &rhs
        }
    }
}

impl<'b> Div<&'b Integer> for &Integer {
    type Output = Integer;

    #[inline(always)]
    fn div(self, rhs: &'b Integer) -> Integer {
        match (self, rhs) {
            (Integer::Single(n1), Integer::Single(n2)) => {
                if let Some(num) = n1.checked_div(*n2) {
                    Integer::Single(num)
                } else {
                    Integer::Double(*n1 as i128 / *n2 as i128)
                }
            }
            (Integer::Single(n1), Integer::Double(r2)) => {
                if let Some(num) = (*n1 as i128).checked_div(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*n1) / *r2)
                }
            }
            (Integer::Double(r1), Integer::Single(r2)) => {
                if let Some(num) = r1.checked_div(*r2 as i128) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*r1) / *r2)
                }
            }
            (Integer::Double(r1), Integer::Double(r2)) => {
                if let Some(num) = r1.checked_div(*r2) {
                    Integer::from_double(num)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*r1) / *r2)
                }
            }
            (Integer::Single(n1), Integer::Large(r2)) => Integer::from((*n1 / r2).complete()),
            (Integer::Large(r1), Integer::Single(n2)) => Integer::from((r1 / *n2).complete()),
            (Integer::Double(n1), Integer::Large(r2)) => Integer::from((*n1 / r2).complete()),
            (Integer::Large(r1), Integer::Double(n2)) => Integer::from((r1 / *n2).complete()),
            (Integer::Large(r1), Integer::Large(r2)) => Integer::from((r1 / r2).complete()),
        }
    }
}

macro_rules! bin_op_int {
    ($base: ty) => {
        impl Add<$base> for Integer {
            type Output = Integer;

            #[inline(always)]
            fn add(self, rhs: $base) -> Integer {
                self + Integer::from(rhs)
            }
        }

        impl<'a> Add<$base> for &'a Integer {
            type Output = Integer;

            #[inline(always)]
            fn add(self, rhs: $base) -> Integer {
                self + Integer::from(rhs)
            }
        }

        impl<'a> Add<Integer> for $base {
            type Output = Integer;

            #[inline(always)]
            fn add(self, rhs: Integer) -> Integer {
                rhs + self
            }
        }

        impl<'a> Add<&'a Integer> for $base {
            type Output = Integer;

            #[inline(always)]
            fn add(self, rhs: &'a Integer) -> Integer {
                rhs + self
            }
        }

        impl Sub<$base> for Integer {
            type Output = Integer;

            #[inline(always)]
            fn sub(self, rhs: $base) -> Integer {
                self - Integer::from(rhs)
            }
        }

        impl<'a> Sub<$base> for &'a Integer {
            type Output = Integer;

            #[inline(always)]
            fn sub(self, rhs: $base) -> Integer {
                self - Integer::from(rhs)
            }
        }

        impl<'a> Sub<Integer> for $base {
            type Output = Integer;

            #[inline(always)]
            fn sub(self, rhs: Integer) -> Integer {
                (-rhs) + self
            }
        }

        impl<'a> Sub<&'a Integer> for $base {
            type Output = Integer;

            #[inline(always)]
            fn sub(self, rhs: &'a Integer) -> Integer {
                -(rhs + -(self as i64))
            }
        }

        impl Mul<$base> for Integer {
            type Output = Integer;

            #[inline(always)]
            fn mul(self, rhs: $base) -> Integer {
                self * Integer::from(rhs)
            }
        }

        impl<'a> Mul<$base> for &'a Integer {
            type Output = Integer;

            #[inline(always)]
            fn mul(self, rhs: $base) -> Integer {
                self * Integer::from(rhs)
            }
        }

        impl<'a> Mul<Integer> for $base {
            type Output = Integer;

            #[inline(always)]
            fn mul(self, rhs: Integer) -> Integer {
                rhs * self
            }
        }

        impl<'a> Mul<&'a Integer> for $base {
            type Output = Integer;

            #[inline(always)]
            fn mul(self, rhs: &'a Integer) -> Integer {
                rhs * self
            }
        }

        impl Div<$base> for Integer {
            type Output = Integer;

            #[inline(always)]
            fn div(self, rhs: $base) -> Integer {
                &self / rhs
            }
        }

        impl<'a> Div<$base> for &'a Integer {
            type Output = Integer;

            #[inline(always)]
            fn div(self, rhs: $base) -> Integer {
                self / Integer::from(rhs)
            }
        }

        impl Rem<$base> for Integer {
            type Output = Integer;

            #[inline(always)]
            fn rem(self, rhs: $base) -> Integer {
                &self % rhs
            }
        }

        impl<'a> Rem<$base> for &'a Integer {
            type Output = Integer;

            #[inline(always)]
            fn rem(self, rhs: $base) -> Integer {
                self % Integer::from(rhs)
            }
        }

        impl AddAssign<$base> for Integer {
            #[inline]
            fn add_assign(&mut self, rhs: $base) {
                *self += Integer::from(rhs);
            }
        }

        impl SubAssign<$base> for Integer {
            #[inline]
            fn sub_assign(&mut self, rhs: $base) {
                *self -= Integer::from(rhs);
            }
        }

        impl MulAssign<$base> for Integer {
            #[inline]
            fn mul_assign(&mut self, rhs: $base) {
                *self *= Integer::from(rhs);
            }
        }

        impl DivAssign<$base> for Integer {
            #[inline]
            fn div_assign(&mut self, rhs: $base) {
                *self /= Integer::from(rhs);
            }
        }
    };
}

bin_op_int!(i8);
bin_op_int!(i16);
bin_op_int!(i32);
bin_op_int!(i64);
bin_op_int!(i128);
bin_op_int!(u8);
bin_op_int!(u16);
bin_op_int!(u32);
bin_op_int!(u64);
bin_op_int!(u128);

impl AddAssign<Integer> for Integer {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Single(r) => l.add_assign(r),
                Integer::Double(r) => l.add_assign(r),
                Integer::Large(r) => l.add_assign(r),
            }

            self.simplify();
        } else {
            *self = rhs + &*self;
        }
    }
}

impl<'a> AddAssign<&'a Integer> for Integer {
    #[inline(always)]
    fn add_assign(&mut self, rhs: &'a Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Single(r) => l.add_assign(*r),
                Integer::Double(r) => l.add_assign(*r),
                Integer::Large(r) => l.add_assign(r),
            }

            self.simplify();
        } else {
            *self = &*self + rhs;
        }
    }
}

impl SubAssign<Integer> for Integer {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Single(r) => l.sub_assign(r),
                Integer::Double(r) => l.sub_assign(r),
                Integer::Large(r) => l.sub_assign(r),
            }

            self.simplify();
        } else {
            *self = &*self - rhs;
        }
    }
}

impl<'a> SubAssign<&'a Integer> for Integer {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: &'a Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Single(r) => l.sub_assign(*r),
                Integer::Double(r) => l.sub_assign(*r),
                Integer::Large(r) => l.sub_assign(r),
            }

            self.simplify();
        } else {
            *self = &*self - rhs;
        }
    }
}

impl MulAssign<Integer> for Integer {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Single(r) => l.mul_assign(r),
                Integer::Double(r) => l.mul_assign(r),
                Integer::Large(r) => l.mul_assign(r),
            }

            self.simplify();
        } else {
            *self = &*self * rhs;
        }
    }
}

impl<'a> MulAssign<&'a Integer> for Integer {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: &'a Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Single(r) => l.mul_assign(*r),
                Integer::Double(r) => l.mul_assign(*r),
                Integer::Large(r) => l.mul_assign(r),
            }

            self.simplify();
        } else {
            *self = &*self * rhs;
        }
    }
}

impl DivAssign<Integer> for Integer {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Single(r) => l.div_assign(r),
                Integer::Double(r) => l.div_assign(r),
                Integer::Large(r) => l.div_assign(r),
            }

            self.simplify();
        } else {
            *self = &*self / rhs;
        }
    }
}

impl<'a> DivAssign<&'a Integer> for Integer {
    #[inline(always)]
    fn div_assign(&mut self, rhs: &'a Integer) {
        if let Integer::Large(l) = self {
            match rhs {
                Integer::Single(r) => l.div_assign(*r),
                Integer::Double(r) => l.div_assign(*r),
                Integer::Large(r) => l.div_assign(r),
            }

            self.simplify();
        } else {
            *self = &*self / rhs;
        }
    }
}

impl Neg for Integer {
    type Output = Integer;

    #[inline]
    fn neg(self) -> Self::Output {
        match self {
            Integer::Single(n) => {
                if let Some(neg) = n.checked_neg() {
                    Integer::Single(neg)
                } else {
                    Integer::Double((n as i128).neg())
                }
            }
            Integer::Double(n) => {
                if let Some(neg) = n.checked_neg() {
                    Integer::from_double(neg)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(n).neg())
                }
            }
            Integer::Large(r) => Integer::from(-r),
        }
    }
}

impl Neg for &Integer {
    type Output = Integer;

    #[inline]
    fn neg(self) -> Self::Output {
        match self {
            Integer::Single(n) => {
                if let Some(neg) = n.checked_neg() {
                    Integer::Single(neg)
                } else {
                    Integer::Double((*n as i128).neg())
                }
            }
            Integer::Double(n) => {
                if let Some(neg) = n.checked_neg() {
                    Integer::from_double(neg)
                } else {
                    Integer::Large(MultiPrecisionInteger::from(*n).neg())
                }
            }
            Integer::Large(r) => Integer::from(r.clone().neg()),
        }
    }
}

impl Rem<Integer> for Integer {
    type Output = Integer;

    fn rem(self, rhs: Self) -> Self::Output {
        self % &rhs
    }
}

impl<'a> Rem<&'a Integer> for Integer {
    type Output = Integer;

    fn rem(self, rhs: &'a Self) -> Self::Output {
        if rhs.is_zero() {
            panic!("Cannot divide by zero");
        }

        if !self.is_negative() && self < *rhs {
            return self;
        }

        match (self, rhs) {
            (Integer::Large(a), Integer::Single(b)) => Integer::from(a.rem_euc(b)),
            (Integer::Large(a), Integer::Double(b)) => Integer::from(a.rem_euc(b)),
            (Integer::Large(a), Integer::Large(b)) => Integer::from(a.rem_euc(b)),
            (x, _) => (&x).rem(rhs),
        }
    }
}

impl Rem<Integer> for &Integer {
    type Output = Integer;

    fn rem(self, rhs: Integer) -> Self::Output {
        self % &rhs
    }
}

impl Rem for &Integer {
    type Output = Integer;

    fn rem(self, rhs: Self) -> Self::Output {
        if rhs.is_zero() {
            panic!("Cannot divide by zero");
        }

        if !self.is_negative() && self < rhs {
            return self.clone();
        }

        match (self, rhs) {
            (Integer::Single(a), Integer::Single(b)) => {
                if let Some(r) = a.checked_rem_euclid(*b) {
                    Integer::Single(r)
                } else {
                    Integer::zero()
                }
            }
            (Integer::Single(a), Integer::Double(b)) => {
                // b must be larger than a, so division is never needed
                if *a < 0 {
                    if *b > 0 {
                        Integer::from_double(*a as i128 + *b)
                    } else {
                        Integer::from_double(*a as i128 - *b)
                    }
                } else {
                    Integer::Single(*a)
                }
            }
            (Integer::Single(a), Integer::Large(b)) => {
                if *a < 0 {
                    if *b > 0 {
                        Integer::from((a + b).complete())
                    } else {
                        Integer::from((a - b).complete())
                    }
                } else {
                    Integer::Single(*a)
                }
            }
            (Integer::Double(a), Integer::Large(b)) => {
                if *a < 0 {
                    if *b > 0 {
                        Integer::from((a + b).complete())
                    } else {
                        Integer::from((a - b).complete())
                    }
                } else {
                    Integer::Double(*a)
                }
            }
            (Integer::Double(a), Integer::Single(b)) => {
                if let Some(r) = a.checked_rem_euclid(*b as i128) {
                    Integer::from_double(r)
                } else {
                    Integer::zero()
                }
            }
            (Integer::Double(a), Integer::Double(b)) => {
                if let Some(r) = a.checked_rem_euclid(*b) {
                    Integer::from_double(r)
                } else {
                    Integer::zero()
                }
            }
            (Integer::Large(a), Integer::Single(b)) => Integer::from(a.rem_euc(b).complete()),
            (Integer::Large(a), Integer::Double(b)) => Integer::from(a.rem_euc(b).complete()),
            (Integer::Large(a), Integer::Large(b)) => Integer::from(a.rem_euc(b).complete()),
        }
    }
}

/// A ring for multi-precision integers.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct MultiPrecisionIntegerRing;

impl Display for MultiPrecisionIntegerRing {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl Default for MultiPrecisionIntegerRing {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiPrecisionIntegerRing {
    pub fn new() -> MultiPrecisionIntegerRing {
        MultiPrecisionIntegerRing
    }
}

impl InternalOrdering for MultiPrecisionInteger {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        Ord::cmp(self, other)
    }
}

impl Set for MultiPrecisionIntegerRing {
    type Element = MultiPrecisionInteger;

    fn size(&self) -> Option<Integer> {
        None
    }
}

impl RingOps<<Self as Set>::Element> for MultiPrecisionIntegerRing {
    #[inline]
    fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        a + b
    }

    #[inline]
    fn sub(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        a - b
    }

    #[inline]
    fn mul(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        a * b
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a += b;
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a -= b;
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a *= b;
    }

    #[inline(always)]
    fn add_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        a.add_assign(b * c)
    }

    #[inline(always)]
    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        a.sub_assign(b * c)
    }

    #[inline]
    fn neg(&self, a: Self::Element) -> Self::Element {
        a.neg()
    }
}

impl RingOps<&<Self as Set>::Element> for MultiPrecisionIntegerRing {
    #[inline]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone() + b
    }

    #[inline]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone() - b
    }

    #[inline]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone() * b
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a += b;
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a -= b;
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a *= b;
    }

    #[inline(always)]
    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        a.add_assign(b * c)
    }

    #[inline(always)]
    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        a.sub_assign(b * c)
    }

    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        a.clone().neg()
    }
}

impl Ring for MultiPrecisionIntegerRing {
    #[inline]
    fn zero(&self) -> Self::Element {
        MultiPrecisionInteger::new()
    }

    #[inline]
    fn one(&self) -> Self::Element {
        MultiPrecisionInteger::from(1)
    }

    #[inline]
    fn nth(&self, n: Integer) -> Self::Element {
        n.to_multi_prec()
    }

    #[inline]
    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        if e > u32::MAX as u64 {
            panic!("Power of exponentiation is larger than 2^32: {e}");
        }
        b.clone().pow(e as u32)
    }

    #[inline]
    fn is_zero(&self, a: &Self::Element) -> bool {
        a.is_zero()
    }

    #[inline]
    fn is_one(&self, a: &Self::Element) -> bool {
        *a == self.one()
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn characteristic(&self) -> Integer {
        0.into()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if *a == 1 || *a == -1 {
            Some(a.clone())
        } else {
            None
        }
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if b.is_zero() {
            return None;
        }

        let (r, q) = a.div_rem_ref(b).complete();

        if q.is_zero() { Some(r) } else { None }
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.random_range(range.0..range.1);
        MultiPrecisionInteger::from(r)
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        _opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        if state.in_sum {
            write!(f, "{element:+}")?
        } else {
            write!(f, "{element}")?
        }

        Ok(false)
    }

    fn has_independent_elements(&self) -> bool {
        true
    }
}

impl EuclideanDomain for MultiPrecisionIntegerRing {
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone() % b
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        a.clone().div_rem_euc(b.clone())
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.clone().gcd(b)
    }
}

/// Compute the GCD of two `u64` numbers.
pub fn gcd_unsigned(mut a: u64, mut b: u64) -> u64 {
    let mut c;
    while a != 0 {
        c = a;
        a = b % a;
        b = c;
    }
    b
}

/// Compute the extended GCD of two `i64` numbers.
pub fn extended_gcd(mut a: i64, mut b: i64) -> (u64, i64, i64) {
    if a.unsigned_abs() < b.unsigned_abs() {
        let (g, s, t) = extended_gcd(b, a);
        return (g, t, s);
    }

    if a == i64::MIN {
        if b == -1 {
            return (1, 0, -1);
        } else if b == i64::MAX {
            return (1, -1, -1);
        }
    }

    let mut s0 = 1;
    let mut s1 = 0;
    let mut t0 = 0;
    let mut t1 = 1;

    while b != 0 {
        let q = a / b;
        (a, b) = (b, a - q * b);
        (s0, s1) = (s1, s0 - q * s1);
        (t0, t1) = (t1, t0 - q * t1);
    }

    (a.unsigned_abs(), s0, t0)
}

/// Compute the extended GCD of two `i128` numbers.
pub fn extended_gcd_i128(mut a: i128, mut b: i128) -> (u128, i128, i128) {
    if a.unsigned_abs() < b.unsigned_abs() {
        let (g, s, t) = extended_gcd_i128(b, a);
        return (g, t, s);
    }

    if a == i128::MIN {
        if b == -1 {
            return (1, 0, -1);
        } else if b == i128::MAX {
            return (1, -1, -1);
        }
    }

    let mut s0 = 1;
    let mut s1 = 0;
    let mut t0 = 0;
    let mut t1 = 1;

    while b != 0 {
        let q = a / b;
        (a, b) = (b, a - q * b);
        (s0, s1) = (s1, s0 - q * s1);
        (t0, t1) = (t1, t0 - q * t1);
    }

    (a.unsigned_abs(), s0, t0)
}

/// Compute the signed GCD of two `i64` numbers.
pub fn gcd_signed(mut a: i64, mut b: i64) -> u64 {
    let mut c;
    while a != 0 {
        c = a;
        // only wraps when i64::MIN % -1 and that still yields 0
        a = b.wrapping_rem(a);
        b = c;
    }
    b.unsigned_abs()
}

/// Compute the signed GCD of two `i128` numbers.
pub fn gcd_signed_i128(mut a: i128, mut b: i128) -> u128 {
    let mut c;
    while a != 0 {
        c = a;
        // only wraps when i128::MIN % -1 and that still yields 0
        a = b.wrapping_rem(a);
        b = c;
    }
    b.unsigned_abs()
}

#[cfg(test)]
mod test {
    use std::ops::{Add, Div, Mul, Rem, Sub};

    use rug::Complete;

    use crate::domains::{
        float::{F64, Float},
        integer::{extended_gcd, extended_gcd_i128},
    };

    use super::Integer;

    #[test]
    fn binary_ops() {
        let a = Integer::from(5);
        let b: Integer = 7.into();

        assert!(a >= 5);
        assert!(5 <= a);
        assert!(a >= -891273892173892178922i128);

        let c = 2u32;

        assert_eq!(c + b.clone(), 9);
        assert_eq!(c + &b, 9);
        assert_eq!(c * b.clone(), 14);
        assert_eq!(c * &b, 14);
        assert_eq!(b.clone() + c, 9);
        assert_eq!(&b + c, 9);
        assert_eq!(b.clone() * c, 14);
        assert_eq!(&b * c, 14);
        assert_eq!(b.clone() / c, 3);
        assert_eq!(&b / c, 3);
        assert_eq!(&b % c, 1);
        assert_eq!(b.clone() % c, 1);

        macro_rules! try_variants {
            ($a: expr, $b: expr, $res: expr, $op: tt) => {
                assert_eq!($a.clone().$op(&$b), $res);
                assert_eq!($a.clone().$op($b.clone()), $res);
                assert_eq!((&$a).$op($b.clone()), $res);
                assert_eq!((&$a).$op(&$b), $res);
            };
        }

        try_variants!(a, b, 12, add);
        try_variants!(a, b, -2, sub);
        try_variants!(a, b, 35, mul);
        try_variants!(a, b, 0, div);
        try_variants!(a, b, 5, rem);

        let a = Integer::from(5123123132i64).pow(5);
        let b: Integer = Integer::from(-312223132i64).pow(5);

        try_variants!(
            a,
            b,
            Integer::from(
                rug::Integer::parse("3529178341193418202448766865967598093745792000000")
                    .unwrap()
                    .complete()
            ),
            add
        );
        try_variants!(
            a,
            b,
            Integer::from(
                rug::Integer::parse("3529184275300451286008027827753913822719081764864")
                    .unwrap()
                    .complete()
            ),
            sub
        );
        try_variants!(a, b,  Integer::from(
            rug::Integer::parse("-10471269811147586074167526453409671971439869211545667023340431153576356451994427981246234624")
                .unwrap()
                .complete()
        ), mul);
        try_variants!(a, b, -1189456, div);
        try_variants!(
            a,
            b,
            Integer::from(
                rug::Integer::parse("1700675215712075116094879895131557158845440")
                    .unwrap()
                    .complete()
            ),
            rem
        );
    }

    #[test]
    fn pslq_small() {
        let result = Integer::solve_integer_relation(
            &[F64::from(-32.0177), F64::from(3.1416), F64::from(2.7183)],
            F64::from(1e-4),
            1,
            Some(Integer::from(100000u64)),
            None,
        )
        .unwrap();

        assert_eq!(result, &[1, 5, 6]);
    }

    #[test]
    fn pslq_medium() {
        let pi = Float::with_val(300, rug::float::Constant::Pi);
        let e = Float::with_val(300, rug::float::Constant::Euler);
        let log2 = Float::with_val(300, rug::float::Constant::Log2);
        let r = pi.clone() * 178236781263123i64
            + e.clone() * -712365671253675i64
            + log2.clone() * 712637812361762786i64;

        let result = Integer::solve_integer_relation(
            &[pi, e, log2, -r],
            Float::with_val(300, 1e-90),
            400,
            Some(Integer::from(1000000000000000000000000000000u128)),
            None,
        )
        .unwrap();

        assert_eq!(result.iter().last().unwrap().abs(), 1);
    }

    #[test]
    fn extended_gcds() {
        let (g, s, t) = extended_gcd(123, 456);
        assert_eq!(g, 3);
        assert_eq!(s, -63);
        assert_eq!(t, 17);

        let (g2, s2, t2) = extended_gcd(48, 18);
        assert_eq!(g2, 6);
        assert_eq!(s2, -1);
        assert_eq!(t2, 3);

        let (g3, s3, t3) = extended_gcd_i128(101, 147);
        assert_eq!(g3, 1);
        assert_eq!(s3, -16);
        assert_eq!(t3, 11);
    }

    #[cfg(feature = "bincode")]
    #[test]
    fn bincode_export() {
        let a = Integer::from(rug::Integer::factorial(150).complete());
        let encoded = bincode::encode_to_vec(&a, bincode::config::standard()).unwrap();
        let b: Integer = bincode::decode_from_slice(&encoded, bincode::config::standard())
            .unwrap()
            .0;
        assert_eq!(a, b);
    }
}
