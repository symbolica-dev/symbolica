//! Fraction fields and rational numbers.

use std::{
    borrow::Cow,
    fmt::{Display, Error, Formatter},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{
    domains::{RingOps, Set, finite_field::Zp64},
    printer::{PrintOptions, PrintState},
};

use super::{
    EuclideanDomain, Field, InternalOrdering, Ring, SelfRing, UpgradeToField,
    finite_field::{
        FiniteField, FiniteFieldCore, FiniteFieldWorkspace, PrimeIteratorU64, ToFiniteField, Two,
        Z2, Zp,
    },
    integer::{Integer, IntegerRing, Z},
};

/// The field of rational numbers.
pub type Q = FractionField<IntegerRing>;
pub type RationalField = FractionField<IntegerRing>;
/// The field of rational numbers.
pub const Q: FractionField<IntegerRing> = FractionField::new(Z);

/// The fraction field of `R`.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FractionField<R: Ring> {
    ring: R,
}

impl<R: Ring> FractionField<R> {
    pub const fn new(ring: R) -> FractionField<R> {
        FractionField { ring }
    }

    pub fn ring(&self) -> &R {
        &self.ring
    }
}

impl<R: EuclideanDomain + FractionNormalization> FractionField<R> {
    pub fn to_element_numerator(&self, numerator: R::Element) -> <Self as Set>::Element {
        Fraction {
            numerator,
            denominator: self.ring.one(),
        }
    }
}

impl<R: EuclideanDomain + FractionNormalization> FractionField<R> {
    pub fn to_element(
        &self,
        mut numerator: R::Element,
        mut denominator: R::Element,
        do_gcd: bool,
    ) -> <Self as Set>::Element {
        if self.ring.is_zero(&denominator) {
            panic!("Cannot create a fraction with zero denominator");
        }

        if do_gcd {
            let g = self.ring.gcd(&numerator, &denominator);
            if !self.ring.is_one(&g) {
                numerator = self.ring.quot_rem(&numerator, &g).0;
                denominator = self.ring.quot_rem(&denominator, &g).0;
            }
        }

        let f = self.ring.get_normalization_factor(&denominator);

        if self.ring.is_one(&f) {
            Fraction {
                numerator,
                denominator,
            }
        } else {
            Fraction {
                numerator: self.ring.mul(&numerator, &f),
                denominator: self.ring.mul(&denominator, &f),
            }
        }
    }
}

impl<R: Ring> Display for FractionField<R> {
    fn fmt(&self, _f: &mut Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

pub trait FractionNormalization: Ring {
    /// Get the factor that normalizes the element `a`.
    /// - For a field, this is the inverse of `a`.
    /// - For the integers, this is the sign of `a`.
    /// - For a polynomial ring, this is the normalization factor of the leading coefficient.
    fn get_normalization_factor(&self, a: &Self::Element) -> Self::Element;
}

impl FractionNormalization for Z {
    fn get_normalization_factor(&self, a: &Integer) -> Integer {
        if *a < 0 { (-1).into() } else { 1.into() }
    }
}

impl<T: Field> FractionNormalization for T {
    fn get_normalization_factor(&self, a: &Self::Element) -> Self::Element {
        self.inv(a)
    }
}

/// A fraction of two elements of a ring. Create a new one through [FractionField::to_element].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Fraction<R: Ring> {
    numerator: R::Element,
    denominator: R::Element,
}

#[cfg(feature = "bincode")]
impl<R: Ring> bincode::enc::Encode for Fraction<R>
where
    R::Element: bincode::enc::Encode,
{
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        self.numerator.encode(encoder)?;
        self.denominator.encode(encoder)
    }
}

#[cfg(feature = "bincode")]
impl<C, R: Ring> bincode::de::Decode<C> for Fraction<R>
where
    R::Element: bincode::de::Decode<C>,
{
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Ok(Fraction {
            numerator: R::Element::decode(decoder)?,
            denominator: R::Element::decode(decoder)?,
        })
    }
}

#[cfg(feature = "bincode")]
impl<'de, C, R: Ring> bincode::de::BorrowDecode<'de, C> for Fraction<R>
where
    R::Element: bincode::de::BorrowDecode<'de, C>,
{
    fn borrow_decode<D: bincode::de::BorrowDecoder<'de, Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        Ok(Fraction {
            numerator: R::Element::borrow_decode(decoder)?,
            denominator: R::Element::borrow_decode(decoder)?,
        })
    }
}

impl<R: Ring> Fraction<R> {
    pub fn numerator(&self) -> R::Element {
        self.numerator.clone()
    }

    pub fn denominator(&self) -> R::Element {
        self.denominator.clone()
    }

    pub fn numerator_ref(&self) -> &R::Element {
        &self.numerator
    }

    pub fn denominator_ref(&self) -> &R::Element {
        &self.denominator
    }

    pub fn from_unchecked(numerator: R::Element, denominator: R::Element) -> Self {
        Fraction {
            numerator,
            denominator,
        }
    }
}

impl<R: Ring> InternalOrdering for Fraction<R> {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.numerator
            .internal_cmp(&other.numerator)
            .then_with(|| self.denominator.internal_cmp(&other.denominator))
    }
}

impl<R: EuclideanDomain + FractionNormalization> Set for FractionField<R> {
    type Element = Fraction<R>;

    fn size(&self) -> Option<Integer> {
        self.ring.size().map(|s| &s * (&s - 1))
    }
}

impl<R: EuclideanDomain + FractionNormalization> RingOps<<FractionField<R> as Set>::Element>
    for FractionField<R>
{
    fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        let r = &self.ring;

        if a.denominator == b.denominator {
            let num = r.add(&a.numerator, &b.numerator);
            let g = r.gcd(&num, &a.denominator);
            if !r.is_one(&g) {
                return Fraction {
                    numerator: r.quot_rem(&num, &g).0,
                    denominator: r.quot_rem(&a.denominator, &g).0,
                };
            } else {
                return Fraction {
                    numerator: num,
                    denominator: a.denominator.clone(),
                };
            }
        }

        let denom_gcd = r.gcd(&a.denominator, &b.denominator);

        let mut a_den_red = Cow::Borrowed(&a.denominator);
        let mut b_den_red = Cow::Borrowed(&b.denominator);

        if !r.is_one(&denom_gcd) {
            a_den_red = Cow::Owned(r.quot_rem(&a.denominator, &denom_gcd).0);
            b_den_red = Cow::Owned(r.quot_rem(&b.denominator, &denom_gcd).0);
        }

        let num1 = r.mul(&a.numerator, &b_den_red);
        let num2 = r.mul(&b.numerator, &a_den_red);
        let mut num = r.add(&num1, &num2);

        // TODO: prefer small * large over medium * medium sized operations
        // a_denom_red.as_ref() * &other.denominator may be faster
        // TODO: add size hint trait with default implementation?
        let mut den = r.mul(b_den_red.as_ref(), &a.denominator);

        let g = r.gcd(&num, &denom_gcd);

        if !r.is_one(&g) {
            num = r.quot_rem(&num, &g).0;
            den = r.quot_rem(&den, &g).0;
        }

        Fraction {
            numerator: num,
            denominator: den,
        }
    }

    fn sub(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        self.add(a, self.neg(b))
    }

    fn mul(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        let r = &self.ring;
        let gcd1 = r.gcd(&a.numerator, &b.denominator);
        let gcd2 = r.gcd(&a.denominator, &b.numerator);

        if r.is_one(&gcd1) {
            if r.is_one(&gcd2) {
                Fraction {
                    numerator: r.mul(&a.numerator, &b.numerator),
                    denominator: r.mul(&a.denominator, &b.denominator),
                }
            } else {
                Fraction {
                    numerator: r.mul(&a.numerator, &r.quot_rem(&b.numerator, &gcd2).0),
                    denominator: r.mul(&r.quot_rem(&a.denominator, &gcd2).0, &b.denominator),
                }
            }
        } else if r.is_one(&gcd2) {
            Fraction {
                numerator: r.mul(&r.quot_rem(&a.numerator, &gcd1).0, &b.numerator),
                denominator: r.mul(&a.denominator, &r.quot_rem(&b.denominator, &gcd1).0),
            }
        } else {
            Fraction {
                numerator: r.mul(
                    &r.quot_rem(&a.numerator, &gcd1).0,
                    &r.quot_rem(&b.numerator, &gcd2).0,
                ),
                denominator: r.mul(
                    &r.quot_rem(&a.denominator, &gcd2).0,
                    &r.quot_rem(&b.denominator, &gcd1).0,
                ),
            }
        }
    }

    fn add_assign(&self, a: &mut Self::Element, b: Self::Element) {
        // TODO: optimize
        *a = self.add(&*a, &b);
    }

    fn sub_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.sub(&*a, &b);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.mul(&*a, &b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        self.add_assign(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        self.sub_assign(a, &self.mul(b, c));
    }

    fn neg(&self, a: Self::Element) -> Self::Element {
        Fraction {
            numerator: self.ring.neg(a.numerator),
            denominator: a.denominator,
        }
    }
}

impl<R: EuclideanDomain + FractionNormalization> RingOps<&<FractionField<R> as Set>::Element>
    for FractionField<R>
{
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let r = &self.ring;

        if a.denominator == b.denominator {
            let num = r.add(&a.numerator, &b.numerator);
            let g = r.gcd(&num, &a.denominator);
            if !r.is_one(&g) {
                return Fraction {
                    numerator: r.quot_rem(&num, &g).0,
                    denominator: r.quot_rem(&a.denominator, &g).0,
                };
            } else {
                return Fraction {
                    numerator: num,
                    denominator: a.denominator.clone(),
                };
            }
        }

        let denom_gcd = r.gcd(&a.denominator, &b.denominator);

        let mut a_den_red = Cow::Borrowed(&a.denominator);
        let mut b_den_red = Cow::Borrowed(&b.denominator);

        if !r.is_one(&denom_gcd) {
            a_den_red = Cow::Owned(r.quot_rem(&a.denominator, &denom_gcd).0);
            b_den_red = Cow::Owned(r.quot_rem(&b.denominator, &denom_gcd).0);
        }

        let num1 = r.mul(&a.numerator, &b_den_red);
        let num2 = r.mul(&b.numerator, &a_den_red);
        let mut num = r.add(&num1, &num2);

        // TODO: prefer small * large over medium * medium sized operations
        // a_denom_red.as_ref() * &other.denominator may be faster
        // TODO: add size hint trait with default implementation?
        let mut den = r.mul(b_den_red.as_ref(), &a.denominator);

        let g = r.gcd(&num, &denom_gcd);

        if !r.is_one(&g) {
            num = r.quot_rem(&num, &g).0;
            den = r.quot_rem(&den, &g).0;
        }

        Fraction {
            numerator: num,
            denominator: den,
        }
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        // TODO: optimize
        self.add(a, &self.neg(b))
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let r = &self.ring;
        let gcd1 = r.gcd(&a.numerator, &b.denominator);
        let gcd2 = r.gcd(&a.denominator, &b.numerator);

        if r.is_one(&gcd1) {
            if r.is_one(&gcd2) {
                Fraction {
                    numerator: r.mul(&a.numerator, &b.numerator),
                    denominator: r.mul(&a.denominator, &b.denominator),
                }
            } else {
                Fraction {
                    numerator: r.mul(&a.numerator, &r.quot_rem(&b.numerator, &gcd2).0),
                    denominator: r.mul(&r.quot_rem(&a.denominator, &gcd2).0, &b.denominator),
                }
            }
        } else if r.is_one(&gcd2) {
            Fraction {
                numerator: r.mul(&r.quot_rem(&a.numerator, &gcd1).0, &b.numerator),
                denominator: r.mul(&a.denominator, &r.quot_rem(&b.denominator, &gcd1).0),
            }
        } else {
            Fraction {
                numerator: r.mul(
                    &r.quot_rem(&a.numerator, &gcd1).0,
                    &r.quot_rem(&b.numerator, &gcd2).0,
                ),
                denominator: r.mul(
                    &r.quot_rem(&a.denominator, &gcd2).0,
                    &r.quot_rem(&b.denominator, &gcd1).0,
                ),
            }
        }
    }

    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        // TODO: optimize
        *a = self.add(&*a, b);
    }

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.sub(&*a, b);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(&*a, b);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.add_assign(a, &self.mul(b, c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &self.mul(b, c));
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        Fraction {
            numerator: self.ring.neg(&a.numerator),
            denominator: a.denominator.clone(),
        }
    }
}

impl<R: EuclideanDomain + FractionNormalization> Ring for FractionField<R> {
    fn zero(&self) -> Self::Element {
        Fraction {
            numerator: self.ring.zero(),
            denominator: self.ring.one(),
        }
    }

    fn one(&self) -> Self::Element {
        Fraction {
            numerator: self.ring.one(),
            denominator: self.ring.one(),
        }
    }

    #[inline]
    fn nth(&self, n: Integer) -> Self::Element {
        Fraction {
            numerator: self.ring.nth(n),
            denominator: self.ring.one(),
        }
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        Fraction {
            numerator: self.ring.pow(&b.numerator, e),
            denominator: self.ring.pow(&b.denominator, e),
        }
    }

    fn is_zero(&self, a: &Self::Element) -> bool {
        self.ring.is_zero(&a.numerator)
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        self.ring.is_one(&a.numerator) && self.ring.is_one(&a.denominator)
    }

    fn one_is_gcd_unit() -> bool {
        false
    }

    fn characteristic(&self) -> Integer {
        self.ring.characteristic()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if self.ring.is_zero(&a.numerator) {
            None
        } else {
            Some(self.inv(a))
        }
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if self.is_zero(b) {
            None
        } else {
            Some(self.div(a, b))
        }
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        Fraction {
            numerator: self.ring.sample(rng, range),
            denominator: self.ring.one(),
        }
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        mut state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        let has_denom = !self.ring.is_one(&element.denominator);

        let write_par = has_denom && (state.in_exp || state.in_exp_base);
        if write_par {
            if state.in_sum {
                state.in_sum = false;
                f.write_char('+')?;
            }

            f.write_char('(')?;
            state.in_exp = false;
            state.in_exp_base = false;
        }

        if self.ring.format(
            &element.numerator,
            opts,
            PrintState {
                in_product: state.in_product || has_denom,
                suppress_one: state.suppress_one && !has_denom,
                level: state.level + 1,
                ..state
            },
            f,
        )? {
            return Ok(true);
        };

        if has_denom {
            f.write_char('/')?;
            self.ring.format(
                &element.denominator,
                opts,
                state.step(false, true, false, true),
                f,
            )?;
        }

        if write_par {
            f.write_char(')')?;
        }

        Ok(false)
    }

    fn has_independent_elements(&self) -> bool {
        self.ring.has_independent_elements()
    }
}

impl<R: Ring> SelfRing for Fraction<R>
where
    R::Element: SelfRing,
{
    fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    fn is_one(&self) -> bool {
        self.numerator.is_one() && self.denominator.is_one()
    }

    fn format<W: std::fmt::Write>(
        &self,
        opts: &PrintOptions,
        mut state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        let has_denom = !self.denominator.is_one();

        let write_par = has_denom && (state.in_exp || state.in_exp_base);
        if write_par {
            if state.in_sum {
                state.in_sum = false;
                f.write_char('+')?;
            }

            f.write_char('(')?;
            state.in_exp = false;
            state.in_exp_base = false;
        }

        if self.numerator.format(
            opts,
            PrintState {
                in_product: state.in_product || has_denom,
                suppress_one: state.suppress_one && !has_denom,
                level: state.level + 1,
                ..state
            },
            f,
        )? {
            return Ok(true);
        }

        if has_denom {
            f.write_char('/')?;
            self.denominator
                .format(opts, state.step(false, true, false, true), f)?;
        }

        if write_par {
            f.write_char(')')?;
        }

        Ok(false)
    }
}

impl<R: Ring> Display for Fraction<R>
where
    R::Element: SelfRing,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.format(&PrintOptions::default(), PrintState::new(), f)
            .map(|_| ())
    }
}

impl<R: EuclideanDomain + FractionNormalization> EuclideanDomain for FractionField<R> {
    fn rem(&self, _: &Self::Element, _: &Self::Element) -> Self::Element {
        self.zero()
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.div(a, b), self.zero())
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let gcd_num = self.ring.gcd(&a.numerator, &b.numerator);
        let gcd_den = self.ring.gcd(&a.denominator, &b.denominator);

        let d1 = self.ring.quot_rem(&a.denominator, &gcd_den).0;
        let lcm = self.ring.mul(&d1, &b.denominator);

        Fraction {
            numerator: gcd_num,
            denominator: lcm,
        }
    }
}

impl<R: EuclideanDomain + FractionNormalization> Field for FractionField<R> {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        // TODO: optimize
        self.mul(a, &self.inv(b))
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.div(a, b);
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        if self.ring.is_zero(&a.numerator) {
            panic!("Division by 0");
        }

        let f = self.ring.get_normalization_factor(&a.numerator);

        Fraction {
            numerator: self.ring.mul(&a.denominator, &f),
            denominator: self.ring.mul(&a.numerator, &f),
        }
    }
}

/// A rational number.
pub type Rational = Fraction<IntegerRing>;

impl UpgradeToField for IntegerRing {
    type Upgraded = Q;

    fn upgrade(self) -> Self::Upgraded {
        Q
    }

    fn upgrade_element(&self, element: <Self as Set>::Element) -> <Self::Upgraded as Set>::Element {
        Rational::from(element)
    }
}

impl Default for Rational {
    fn default() -> Self {
        Rational::zero()
    }
}

impl PartialEq<Integer> for Rational {
    fn eq(&self, other: &Integer) -> bool {
        self.denominator.is_one() && &self.numerator == other
    }
}

impl<T: Into<Integer> + Copy> PartialEq<T> for Rational {
    fn eq(&self, other: &T) -> bool {
        self.denominator.is_one() && self.numerator == (*other).into()
    }
}

impl<T: Into<Integer> + Copy> PartialEq<(T, T)> for Rational {
    fn eq(&self, other: &(T, T)) -> bool {
        self == &Rational::from((other.0.into(), other.1.into()))
    }
}

impl PartialOrd<Integer> for Rational {
    fn partial_cmp(&self, other: &Integer) -> Option<std::cmp::Ordering> {
        Some(self.numerator.cmp(&(other * self.denominator_ref())))
    }
}

impl<T: Into<Integer> + Copy> PartialOrd<T> for Rational {
    fn partial_cmp(&self, other: &T) -> Option<std::cmp::Ordering> {
        Some(
            self.numerator
                .cmp(&((*other).into() * self.denominator_ref())),
        )
    }
}

impl<T: Into<Integer> + Copy> PartialOrd<(T, T)> for Rational {
    fn partial_cmp(&self, other: &(T, T)) -> Option<std::cmp::Ordering> {
        Some(self.cmp(&Rational::from((other.0.into(), other.1.into()))))
    }
}

impl TryFrom<f64> for Rational {
    type Error = &'static str;

    /// Convert a floating point number to its exact rational number equivalent.
    /// Use [`Rational::truncate_denominator`] to get an approximation with a smaller denominator.
    #[inline]
    fn try_from(f: f64) -> Result<Self, Self::Error> {
        if !f.is_finite() {
            return Err("Cannot convert non-finite float to rational");
        }

        // taken from num-traits
        let bits: u64 = f.to_bits();
        let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
        let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
        let mantissa = if exponent == 0 {
            (bits & 0xfffffffffffff) << 1
        } else {
            (bits & 0xfffffffffffff) | 0x10000000000000
        };
        // Exponent bias + mantissa shift
        exponent -= 1023 + 52;

        // superfluous factors of 2 will be divided out in the conversion to rational
        if exponent < 0 {
            Ok((
                (sign as i64 * mantissa as i64).into(),
                Integer::from(2).pow(-exponent as u64),
            )
                .into())
        } else {
            Ok((
                &Integer::from(sign as i64 * mantissa as i64)
                    * &Integer::from(2).pow(exponent as u64),
                1.into(),
            )
                .into())
        }
    }
}

impl<T: Into<Integer>> From<T> for Rational {
    #[inline]
    fn from(value: T) -> Self {
        Rational {
            numerator: value.into(),
            denominator: 1.into(),
        }
    }
}

impl From<&Integer> for Rational {
    fn from(value: &Integer) -> Self {
        Rational {
            numerator: value.clone(),
            denominator: 1.into(),
        }
    }
}

impl<T: Into<Integer>> From<(T, T)> for Rational {
    #[inline]
    fn from((num, den): (T, T)) -> Self {
        Q.to_element(num.into(), den.into(), true)
    }
}

impl From<rug::Rational> for Rational {
    fn from(value: rug::Rational) -> Self {
        let (num, den) = value.into_numer_denom();
        Q.to_element(num.into(), den.into(), false)
    }
}

impl ToFiniteField<u32> for Rational {
    fn to_finite_field(&self, field: &Zp) -> <Zp as Set>::Element {
        field.div(
            &self.numerator.to_finite_field(field),
            &self.denominator.to_finite_field(field),
        )
    }
}

impl ToFiniteField<u64> for Rational {
    fn to_finite_field(&self, field: &Zp64) -> <Zp64 as Set>::Element {
        field.div(
            &self.numerator.to_finite_field(field),
            &self.denominator.to_finite_field(field),
        )
    }
}

impl ToFiniteField<Two> for Rational {
    fn to_finite_field(&self, field: &Z2) -> <Z2 as Set>::Element {
        field.div(
            &self.numerator.to_finite_field(field),
            &self.denominator.to_finite_field(field),
        )
    }
}

impl Rational {
    pub fn new<T: Into<Integer>>(num: T, den: T) -> Rational {
        let d = den.into();
        if d.is_zero() {
            panic!("Cannot create a rational number with zero denominator");
        }

        Q.to_element(num.into(), d, true)
    }

    pub fn from_int_unchecked<T: Into<Integer>>(num: T, den: T) -> Rational {
        Q.to_element(num.into(), den.into(), false)
    }

    pub fn is_negative(&self) -> bool {
        self.numerator < 0
    }

    pub fn is_integer(&self) -> bool {
        self.denominator.is_one()
    }

    pub const fn zero() -> Rational {
        Rational {
            numerator: Integer::zero(),
            denominator: Integer::one(),
        }
    }

    pub const fn one() -> Rational {
        Rational {
            numerator: Integer::one(),
            denominator: Integer::one(),
        }
    }

    pub fn abs(&self) -> Rational {
        if self.is_negative() {
            self.clone().neg()
        } else {
            self.clone()
        }
    }

    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    pub fn is_one(&self) -> bool {
        self.numerator.is_one() && self.denominator.is_one()
    }

    pub fn pow(&self, e: u64) -> Rational {
        Q.pow(self, e)
    }

    pub fn inv(&self) -> Rational {
        Q.inv(self)
    }

    pub fn neg(&self) -> Rational {
        Q.neg(self)
    }

    pub fn gcd(&self, other: &Rational) -> Rational {
        Q.gcd(self, other)
    }

    pub fn to_f64(&self) -> f64 {
        rug::Rational::from((
            self.numerator.clone().to_multi_prec(),
            self.denominator.clone().to_multi_prec(),
        ))
        .to_f64()
    }

    pub fn to_multi_prec(self) -> rug::Rational {
        rug::Rational::from((
            self.numerator.to_multi_prec(),
            self.denominator.to_multi_prec(),
        ))
    }

    /// Return a best approximation of the rational number where the denominator
    /// is less than or equal to `max_denominator`.
    pub fn truncate_denominator(&self, max_denominator: &Integer) -> Rational {
        assert!(!max_denominator.is_zero() && !max_denominator.is_negative());

        if self.denominator_ref() < max_denominator {
            return self.clone();
        }

        let (mut p0, mut q0, mut p1, mut q1) = (
            Integer::zero(),
            Integer::one(),
            Integer::one(),
            Integer::zero(),
        );

        let (mut n, mut d) = (self.numerator_ref().abs(), self.denominator());
        loop {
            let a = &n / &d;
            let q2 = &q0 + &(&a * &q1);
            if &q2 > max_denominator {
                break;
            }
            (p1, p0, q0, q1) = (p0 + &(&a * &p1), p1, q1, q2);
            (d, n) = (&n - &a * &d, d);
        }

        let k = &(max_denominator - &q0) / &q1;
        let bound1: Rational = (p0 + &(&k * &p1), &q0 + &(&k * &q1)).into();
        let bound2: Rational = (p1, q1).into();

        let res = if (&bound2 - self).abs() <= (&bound1 - self).abs() {
            bound2
        } else {
            bound1
        };

        if self.is_negative() { res.neg() } else { res }
    }

    /// Round the rational to the one with the smallest numerator or denominator in the interval
    /// `[self * (1-relative_error), self * (1+relative_error)]`, where
    /// `0 < relative_error < 1`.
    pub fn round(&self, relative_error: &Rational) -> Rational {
        if self.is_zero() {
            return Rational::zero();
        }

        if self.is_negative() {
            self.round_in_interval(
                self.clone() * (Rational::one() + relative_error),
                self.clone() * (Rational::one() - relative_error),
            )
        } else {
            self.round_in_interval(
                self.clone() * (Rational::one() - relative_error),
                self.clone() * (Rational::one() + relative_error),
            )
        }
    }

    /// Round the rational to the one with the smallest numerator or denominator in the interval
    /// `[l, u]`, where `l < u`.
    pub fn round_in_interval(&self, mut l: Rational, mut u: Rational) -> Rational {
        assert!(l < u);

        let mut flip = false;
        if l.is_negative() && u.is_negative() {
            flip = true;
            (l, u) = (-u, -l);
        } else if l.is_negative() {
            return Rational::zero();
        }

        // use continued fractions to find the best approximation in an interval
        let (mut ln, mut ld) = (l.numerator(), l.denominator());
        let (mut un, mut ud) = (u.numerator(), u.denominator());

        // h1/k1 accumulates the shared continued fraction terms of l and u
        let (mut h1, mut h0, mut k1, mut k0): (Integer, Integer, Integer, Integer) =
            (1.into(), 0.into(), 0.into(), 1.into());

        loop {
            let a = &(&ln - &1.into()) / &ld; // get next term in continued fraction
            (ld, ud, ln, un) = (&un - &a * &ud, &ln - &a * &ld, ud, ld); // subtract and invert
            (h1, h0) = (&a * &h1 + &h0, h1);
            (k1, k0) = (&a * &k1 + &k0, k1);
            if ln <= ld {
                let res: Rational = (h1 + &h0, k1 + &k0).into();

                if flip {
                    return -res;
                } else {
                    return res;
                }
            }
        }
    }

    /// Round to the nearest integer towards zero.
    pub fn floor(&self) -> Integer {
        self.numerator_ref() / self.denominator_ref()
    }

    /// Round to the nearest integer away from zero.
    pub fn ceil(&self) -> Integer {
        if self.is_negative() {
            (self.numerator().clone() + 1) / self.denominator_ref() - 1
        } else {
            ((self.numerator().clone() - 1) / self.denominator_ref()) + 1
        }
    }

    pub fn round_to_nearest_integer(&self) -> Integer {
        if self.is_negative() {
            (self - &(1, 2).into()).floor()
        } else {
            (self + &(1, 2).into()).floor()
        }
    }

    /// Reconstruct a rational number `q` from a value `v` in a prime field `p`,
    /// such that `q ≡ v mod p`.
    ///
    /// From "Maximal Quotient Rational Reconstruction: An Almost
    /// Optimal Algorithm for Rational Reconstruction" by Monagan.
    pub fn maximal_quotient_reconstruction(
        v: &Integer,
        p: &Integer,
        acceptance_scale: Option<Integer>,
    ) -> Result<Rational, &'static str> {
        let mut acceptance_scale = match acceptance_scale {
            Some(t) => t.clone(),
            None => {
                // set t to 2^20*ceil(log2(m))
                let ceil_log2 = match &p {
                    Integer::Single(n) => u64::BITS as u64 - (*n as u64).leading_zeros() as u64,
                    Integer::Double(n) => u128::BITS as u64 - (*n as u128).leading_zeros() as u64,
                    Integer::Large(n) => {
                        let mut pos = 0;
                        while let Some(p) = n.find_one(pos) {
                            if let Some(p2) = pos.checked_add(p) {
                                if p2 == u32::MAX {
                                    return Err("Could not reconstruct, as the log is too large");
                                }

                                pos += 1;
                            } else {
                                return Err("Could not reconstruct, as the log is too large");
                            }
                        }
                        pos as u64
                    }
                };

                &Integer::new(2i64 << 10) * &Integer::new(ceil_log2 as i64)
            }
        };

        if v.is_zero() {
            return if p > &acceptance_scale {
                Ok(Rational::zero())
            } else {
                Err("Could not reconstruct: u=0 and t <= m")
            };
        }

        let mut n = Integer::zero();
        let mut d = Integer::zero();
        let (mut t, mut old_t) = (Integer::one(), Integer::zero());
        let (mut r, mut old_r) = (if v.is_negative() { v + p } else { v.clone() }, p.clone());

        while !r.is_zero() && old_r > acceptance_scale {
            let q = &old_r / &r;
            if q > acceptance_scale {
                n = r.clone();
                d = t.clone();
                acceptance_scale = q.clone();
            }
            (r, old_r) = (&old_r - &(&q * &r), r);
            (t, old_t) = (&old_t - &(&q * &t), t);
        }

        if d.is_zero() || !Z.gcd(&n, &d).is_one() {
            return Err("Reconstruction failed");
        }
        if d < Integer::zero() {
            n = n.neg();
            d = d.neg();
        }

        Ok((n, d).into())
    }

    /// Return the rational number that corresponds to `f` evaluated at sample point `sample`,
    /// i.e. `f(sample)`, if such a number exists and if the evaluations were not unlucky.
    ///
    /// The procedure can be repeated with a different starting prime, by setting `prime_start`
    /// to a non-zero value.
    pub fn rational_reconstruction<
        F: Fn(&Zp, &[<Zp as Set>::Element]) -> <Zp as Set>::Element,
        R: Ring,
    >(
        f: F,
        sample: &[R::Element],
        prime_start: Option<usize>,
    ) -> Result<Rational, &'static str>
    where
        Zp: FiniteFieldCore<u32>,
        R::Element: ToFiniteField<u32>,
    {
        let mut cur_result = Integer::one();
        let mut prime_accum = Integer::one();
        let mut prime_sample_point = vec![];
        let mut primes =
            PrimeIteratorU64::new(u32::get_large_prime() as u64 + prime_start.unwrap_or(0) as u64);

        let mut last_guess = Rational::zero();
        for i in 0..sample.len() {
            let Some(p) = primes.next() else {
                return Err("Ran out of primes for rational reconstruction");
            };
            let Some(p) = u32::try_from_integer(p.into()) else {
                return Err("Ran out of primes for rational reconstruction");
            };

            let field = FiniteField::<u32>::new(p);
            prime_sample_point.clear();
            for x in sample {
                prime_sample_point.push(x.to_finite_field(&field));
            }

            let eval = f(&field, &prime_sample_point);

            let eval_conv = field.from_element(&eval).to_integer();

            if i == 0 {
                cur_result = eval_conv;
            } else {
                let new_result = Integer::chinese_remainder(
                    eval_conv,
                    cur_result.clone(),
                    Integer::Single(p as i64),
                    prime_accum.clone(),
                );

                if cur_result == new_result {
                    return Ok(last_guess);
                }
                cur_result = new_result;
            }

            prime_accum *= &Integer::Single(p as i64);

            if cur_result < Integer::zero() {
                cur_result += &prime_accum;
            }

            if let Ok(q) =
                Rational::maximal_quotient_reconstruction(&cur_result, &prime_accum, None)
            {
                if q == last_guess {
                    return Ok(q);
                } else {
                    last_guess = q;
                }
            }
        }

        Err("Reconstruction failed")
    }
}

impl PartialOrd for Rational {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rational {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        if self.denominator == other.denominator {
            return self.numerator.cmp(&other.numerator);
        }

        let a = self.numerator_ref() * other.denominator_ref();
        let b = self.denominator_ref() * other.numerator_ref();

        a.cmp(&b)
    }
}

impl Add<Rational> for Rational {
    type Output = Rational;

    fn add(self, other: Rational) -> Self::Output {
        Q.add(&self, &other)
    }
}

impl Sub<Rational> for Rational {
    type Output = Rational;

    fn sub(self, other: Rational) -> Self::Output {
        self.add(&other.neg())
    }
}

impl Mul<Rational> for Rational {
    type Output = Rational;

    fn mul(self, other: Rational) -> Self::Output {
        Q.mul(&self, &other)
    }
}

impl Div<Rational> for Rational {
    type Output = Rational;

    fn div(self, other: Rational) -> Self::Output {
        Q.div(&self, &other)
    }
}

impl<'a> Add<&'a Rational> for Rational {
    type Output = Rational;

    fn add(self, other: &'a Rational) -> Self::Output {
        Q.add(&self, other)
    }
}

impl<'a> Sub<&'a Rational> for Rational {
    type Output = Rational;

    fn sub(self, other: &'a Rational) -> Self::Output {
        self.add(&other.neg())
    }
}

impl<'a> Mul<&'a Rational> for Rational {
    type Output = Rational;

    fn mul(self, other: &'a Rational) -> Self::Output {
        Q.mul(&self, other)
    }
}

impl<'a> Div<&'a Rational> for Rational {
    type Output = Rational;

    fn div(self, other: &'a Rational) -> Self::Output {
        Q.div(&self, other)
    }
}

impl<'a> Add<&'a Rational> for &Rational {
    type Output = Rational;

    fn add(self, other: &'a Rational) -> Self::Output {
        Q.add(self, other)
    }
}

impl<'a> Sub<&'a Rational> for &Rational {
    type Output = Rational;

    fn sub(self, other: &'a Rational) -> Self::Output {
        Q.sub(self, other)
    }
}

impl Neg for Rational {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Q.neg(&self)
    }
}

impl<'a> Mul<&'a Rational> for &Rational {
    type Output = Rational;

    fn mul(self, other: &'a Rational) -> Self::Output {
        Q.mul(self, other)
    }
}

impl<'a> Div<&'a Rational> for &Rational {
    type Output = Rational;

    fn div(self, other: &'a Rational) -> Self::Output {
        Q.div(self, other)
    }
}

impl<'a> AddAssign<&'a Rational> for Rational {
    fn add_assign(&mut self, other: &'a Rational) {
        Q.add_assign(self, other)
    }
}

impl<'a> SubAssign<&'a Rational> for Rational {
    fn sub_assign(&mut self, other: &'a Rational) {
        self.add_assign(&other.neg())
    }
}

impl<'a> MulAssign<&'a Rational> for Rational {
    fn mul_assign(&mut self, other: &'a Rational) {
        Q.mul_assign(self, other)
    }
}

impl<'a> DivAssign<&'a Rational> for Rational {
    fn div_assign(&mut self, other: &'a Rational) {
        Q.div_assign(self, other)
    }
}

impl AddAssign<Rational> for Rational {
    fn add_assign(&mut self, other: Rational) {
        Q.add_assign(self, &other)
    }
}

impl SubAssign<Rational> for Rational {
    fn sub_assign(&mut self, other: Rational) {
        self.add_assign(&other.neg())
    }
}

impl MulAssign<Rational> for Rational {
    fn mul_assign(&mut self, other: Rational) {
        Q.mul_assign(self, &other)
    }
}

impl DivAssign<Rational> for Rational {
    fn div_assign(&mut self, other: Rational) {
        Q.div_assign(self, &other)
    }
}

impl<'a> std::iter::Sum<&'a Self> for Rational {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Rational::zero(), |a, b| a + b)
    }
}

#[cfg(test)]
mod test {
    use crate::domains::{
        Field, Ring, RingOps,
        integer::Z,
        rational::{FractionField, Rational},
    };

    #[test]
    fn rounding() {
        let r: Rational = (11, 10).into();
        let res = r.round_in_interval((1, 1).into(), (12, 10).into());
        assert_eq!(res, (1, 1));

        let r: Rational = (11, 10).into();
        let res = r.round_in_interval((2, 1).into(), (3, 1).into());
        assert_eq!(res, (2, 1));

        let r: Rational = (503, 1500).into();
        let res = r.round(&(1, 10).into());
        assert_eq!(res, (1, 3));

        let r: Rational = (-503, 1500).into();
        let res = r.round(&(1, 10).into());
        assert_eq!(res, (-1, 3));

        let r = crate::domains::float::Float::from(rug::Float::with_val(
            1000,
            rug::float::Constant::Pi,
        ))
        .to_rational();
        let res = r.round(&(1, 100000000).into());
        assert_eq!(res, (93343, 29712));
    }

    #[test]
    fn fraction_int() {
        let f = FractionField::new(Z);
        let b = f.neg(f.nth(3.into()));
        let d = f.div(&f.add(&f.nth(100.into()), &b), &b);
        assert_eq!(d, f.to_element((-97).into(), 3.into(), false));
    }
}
