//! Defines and handles coefficients of Symbolica expressions.
//!
//! It includes implementations for different types of coefficients such as rational numbers,
//! floating-point numbers, finite field elements, rational polynomials and the conversions
//! between them in [Coefficient] and [CoefficientView].
//! These connect with Symbolica's standalone data types such as [Integer],
//! which should be used instead of [Coefficient] if possible.
//!
//! Additionally, the module contains functions for normalizing coefficients, converting
//! coefficients to floats with a specified precision, and setting the coefficient ring
//! for multivariate rational polynomials.
use std::{
    cmp::Ordering,
    f64::consts::LOG2_10,
    ops::{Add, Div, Mul, Neg},
    sync::Arc,
};

use ahash::HashMap;
use bytes::Buf;
use rug::{integer::Order, ops::NegAssign};
use smallvec::{SmallVec, smallvec};

use crate::{
    atom::{Atom, AtomView, Symbol},
    domains::{
        EuclideanDomain, Field, InternalOrdering, Ring, RingOps, Set,
        algebraic_number::AlgebraicExtension,
        atom::AtomField,
        finite_field::{
            FiniteField, FiniteFieldCore, FiniteFieldElement, FiniteFieldWorkspace, ToFiniteField,
            Zp64,
        },
        float::{Complex, F64, Float, FloatField, FloatLike, Real, RealLike, SingleFloat},
        integer::{Integer, IntegerRing, Z},
        rational::{Fraction, Q, Rational},
        rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial},
    },
    error,
    poly::{INLINED_EXPONENTS, PolyVariable, polynomial::MultivariatePolynomial},
    state::{FiniteFieldIndex, State, Workspace},
    warn,
};

pub trait ConvertToRing: Ring {
    /// Convert from an `Integer` to a Ring.
    fn element_from_integer(&self, number: Integer) -> Self::Element;

    /// Convert from a Symbolica `Coefficient` to a Ring.
    /// Panics when the conversion is impossible.
    fn element_from_coefficient(&self, number: Coefficient) -> Self::Element {
        match self.try_element_from_coefficient(number) {
            Ok(e) => e,
            Err(msg) => panic!("Cannot convert coefficient to ring element: {}", msg),
        }
    }

    /// Try to convert from a Symbolica `Coefficient` to a Ring.
    fn try_element_from_coefficient(&self, number: Coefficient) -> Result<Self::Element, String>;

    /// Convert from a Symbolica `CoefficientView` to a Ring.
    /// Panics when the conversion is impossible.
    fn element_from_coefficient_view(&self, number: CoefficientView<'_>) -> Self::Element {
        match self.try_element_from_coefficient_view(number) {
            Ok(e) => e,
            Err(msg) => panic!("Cannot convert coefficient to ring element: {}", msg),
        }
    }

    /// Try to convert from a Symbolica `CoefficientView` to a Ring.
    fn try_element_from_coefficient_view(
        &self,
        number: CoefficientView<'_>,
    ) -> Result<Self::Element, String>;
}

/// A coefficient that can appear in a Symbolica expression.
/// In most cases, this is a rational number but it can also be a finite field element or
/// a rational polynomial. If no additions of mixed-types such as rational numbers
/// or floats is expected, use the standalone structs such as [Rational] and [Float] instead.
///
/// The borrowed version of this is [CoefficientView].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Coefficient {
    /// An indeterminate number, such as `0/0` or `0^0`.
    Indeterminate,
    /// Infinity with an optional complex phase
    Infinity(Option<Complex<Rational>>),
    Complex(Complex<Rational>),
    Float(Complex<Float>),
    FiniteField(FiniteFieldElement<u64>, FiniteFieldIndex),
    RationalPolynomial(RationalPolynomial<IntegerRing, u16>),
}

impl Coefficient {
    /// Construct positive infinity.
    pub fn positive_infinity() -> Self {
        Coefficient::Infinity(Some(Complex::new(Rational::one(), Rational::zero())))
    }

    /// Construct complex infinity.
    pub fn complex_infinity() -> Self {
        Coefficient::Infinity(None)
    }

    /// Construct an indeterminate quantity.
    pub fn indeterminate() -> Self {
        Coefficient::Indeterminate
    }

    /// Construct a coefficient from a finite field element.
    pub fn from_finite_field(field: Zp64, element: FiniteFieldElement<u64>) -> Self {
        let index = State::get_or_insert_finite_field(field);
        Coefficient::FiniteField(element, index)
    }

    /// Normalize the direction of an infinity so that the real part has absolute value 1.
    /// If the real part is zero, the imaginary part is normalized to an absolute value of 1.
    pub(crate) fn normalize_infinity(dir: Complex<Rational>) -> Coefficient {
        if dir.is_zero() {
            Coefficient::Indeterminate
        } else if dir.re.is_zero() {
            Coefficient::Infinity(Some(Complex::new(
                Rational::zero(),
                &dir.im / &dir.im.abs(),
            )))
        } else {
            Coefficient::Infinity(Some(&dir / &dir.re.abs()))
        }
    }

    /// Compute the complex conjugate of the coefficient.
    pub fn conjugate(&self) -> Self {
        match self {
            Coefficient::Indeterminate => Coefficient::Indeterminate,
            Coefficient::Infinity(phase) => match phase {
                Some(p) => Coefficient::Infinity(Some(p.conj())),
                None => Coefficient::Infinity(None),
            },
            Coefficient::Complex(r) => Coefficient::Complex(r.conj()),
            Coefficient::Float(f) => Coefficient::Float(f.conj()),
            Coefficient::FiniteField(n, i) => Coefficient::FiniteField(*n, *i),
            Coefficient::RationalPolynomial(p) => Coefficient::RationalPolynomial(p.clone()),
        }
    }
}

impl std::fmt::Display for Coefficient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Coefficient::Indeterminate => write!(f, "¿"),
            Coefficient::Infinity(Some(phase)) => write!(f, "({})*∞", phase),
            Coefficient::Infinity(None) => write!(f, "⧞"),
            Coefficient::Complex(r) => write!(f, "{}", r),
            Coefficient::Float(fl) => write!(f, "{}", fl),
            Coefficient::FiniteField(n, i) => {
                let field = State::get_finite_field(*i);
                write!(f, "{} (mod {})", n.inner(), field.get_prime())
            }
            Coefficient::RationalPolynomial(rp) => write!(f, "{}", rp),
        }
    }
}

macro_rules! from_via_rational {
    ($type:ty) => {
        impl From<$type> for Coefficient {
            fn from(value: $type) -> Self {
                Coefficient::Complex(Rational::from(value).into())
            }
        }

        impl From<&$type> for Coefficient {
            fn from(value: &$type) -> Self {
                Coefficient::Complex(Rational::from(*value).into())
            }
        }
    };
}

from_via_rational!(i32);
from_via_rational!(i64);
from_via_rational!(i128);
from_via_rational!(isize);
from_via_rational!(u32);
from_via_rational!(u64);
from_via_rational!(u128);
from_via_rational!(usize);
from_via_rational!((i32, i32));
from_via_rational!((i64, i64));
from_via_rational!((i128, i128));
from_via_rational!((isize, isize));
from_via_rational!((u32, u32));
from_via_rational!((u64, u64));
from_via_rational!((u128, u128));
from_via_rational!((usize, usize));

impl From<f64> for Coefficient {
    fn from(value: f64) -> Self {
        Coefficient::Float(Float::with_val(53, value).into())
    }
}

impl From<&f64> for Coefficient {
    fn from(value: &f64) -> Self {
        Coefficient::Float(Float::with_val(53, value).into())
    }
}

impl From<(i64, i64)> for CoefficientView<'_> {
    #[inline]
    fn from(r: (i64, i64)) -> Self {
        CoefficientView::Natural(r.0, r.1, 0, 1)
    }
}

impl From<(i64, i64, i64, i64)> for CoefficientView<'_> {
    #[inline]
    fn from(r: (i64, i64, i64, i64)) -> Self {
        CoefficientView::Natural(r.0, r.1, r.2, r.3)
    }
}

impl From<Integer> for Coefficient {
    fn from(value: Integer) -> Self {
        Coefficient::Complex(Rational::from(value).into())
    }
}

impl From<(Integer, Integer)> for Coefficient {
    fn from(value: (Integer, Integer)) -> Self {
        Coefficient::Complex(Rational::from(value).into())
    }
}

impl From<rug::Integer> for Coefficient {
    fn from(value: rug::Integer) -> Self {
        Coefficient::Complex(Rational::from(value).into())
    }
}

impl From<rug::Rational> for Coefficient {
    fn from(value: rug::Rational) -> Self {
        Coefficient::Complex(Rational::from(value).into())
    }
}

impl From<Rational> for Coefficient {
    fn from(value: Rational) -> Self {
        Coefficient::Complex(value.into())
    }
}

impl From<Float> for Coefficient {
    fn from(value: Float) -> Self {
        Coefficient::Float(value.into())
    }
}

impl From<Complex<Rational>> for Coefficient {
    fn from(value: Complex<Rational>) -> Self {
        Coefficient::Complex(value)
    }
}

impl From<Complex<Float>> for Coefficient {
    fn from(value: Complex<Float>) -> Self {
        Coefficient::Float(value)
    }
}

impl Default for Coefficient {
    fn default() -> Self {
        Coefficient::zero()
    }
}

impl PartialOrd for Coefficient {
    fn partial_cmp(&self, other: &Coefficient) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Coefficient {
    /// A comparison. It considers infinites the same.
    fn cmp(&self, other: &Coefficient) -> Ordering {
        match (self, other) {
            (Coefficient::Indeterminate, Coefficient::Indeterminate) => Ordering::Equal,
            (Coefficient::Infinity(_), Coefficient::Infinity(_)) => Ordering::Equal,
            (Coefficient::Complex(r1), Coefficient::Complex(r2)) => r1
                .re
                .partial_cmp(&r2.re)
                .unwrap_or(Ordering::Equal)
                .then_with(|| r1.im.partial_cmp(&r2.im).unwrap_or(Ordering::Equal)),
            (Coefficient::FiniteField(n1, _), Coefficient::FiniteField(n2, _)) => {
                n1.inner().cmp(n2.inner())
            }
            (Coefficient::Float(f1), Coefficient::Float(f2)) => f1
                .re
                .partial_cmp(&f2.re)
                .unwrap_or(Ordering::Equal)
                .then_with(|| f1.im.partial_cmp(&f2.im).unwrap_or(Ordering::Equal)),
            (Coefficient::RationalPolynomial(n1), Coefficient::RationalPolynomial(n2)) => {
                n1.internal_cmp(n2)
            }
            (Coefficient::Indeterminate, _) => Ordering::Greater,
            (_, Coefficient::Indeterminate) => Ordering::Less,
            (Coefficient::Infinity(_), _) => Ordering::Greater,
            (_, Coefficient::Infinity(_)) => Ordering::Less,
            (Coefficient::Complex(_), _) => Ordering::Less,
            (_, Coefficient::Complex(_)) => Ordering::Greater,
            (Coefficient::Float(_), _) => Ordering::Less,
            (_, Coefficient::Float(_)) => Ordering::Greater,
            (Coefficient::FiniteField(_, _), _) => Ordering::Less,
            (_, Coefficient::FiniteField(_, _)) => Ordering::Greater,
        }
    }
}

impl Coefficient {
    pub fn new() -> Coefficient {
        Coefficient::zero()
    }

    pub fn zero() -> Coefficient {
        Coefficient::Complex(Complex::new_zero())
    }

    pub fn one() -> Coefficient {
        Coefficient::Complex(Complex::new(Rational::one(), Rational::zero()))
    }

    pub fn is_negative(&self) -> bool {
        match self {
            Coefficient::Indeterminate | Coefficient::Infinity(None) => false,
            Coefficient::Infinity(Some(r)) | Coefficient::Complex(r) => {
                r.re.is_negative() && r.im.is_zero() || r.im.is_negative() && r.re.is_zero()
            }
            Coefficient::Float(f) => {
                f.re.is_negative() && f.im.is_zero() || f.im.is_negative() && f.re.is_zero()
            }
            Coefficient::FiniteField(_, _) => false,
            Coefficient::RationalPolynomial(r) => r.numerator.lcoeff().is_negative(),
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Coefficient::Indeterminate | Coefficient::Infinity(_) => false,
            Coefficient::Complex(r) => r.is_zero(),
            Coefficient::Float(f) => f.is_zero(),
            Coefficient::FiniteField(num, _field) => *num.inner() == 0,
            Coefficient::RationalPolynomial(r) => r.numerator.is_zero(),
        }
    }

    pub fn is_one(&self) -> bool {
        match self {
            Coefficient::Indeterminate | Coefficient::Infinity(_) => false,
            Coefficient::Complex(r) => r.is_one(),
            Coefficient::Float(f) => f.is_one(),
            Coefficient::FiniteField(num, field) => {
                let f = State::get_finite_field(*field);
                f.is_one(num)
            }
            Coefficient::RationalPolynomial(r) => r.numerator.is_one(),
        }
    }

    pub fn gcd(&self, rhs: &Self) -> Self {
        match (self, rhs) {
            (Coefficient::Indeterminate | Coefficient::Infinity(_), _) => Self::one(),
            (_, Coefficient::Indeterminate | Coefficient::Infinity(_)) => Self::one(),
            (Coefficient::Complex(r1), Coefficient::Complex(r2)) => {
                Coefficient::Complex(r1.gcd(r2))
            }
            (Coefficient::FiniteField(_n1, i1), Coefficient::FiniteField(_n2, i2)) => {
                if i1 != i2 {
                    error!(
                        "Cannot multiply numbers from different finite fields: p1={}, p2={}",
                        State::get_finite_field(*i1).get_prime(),
                        State::get_finite_field(*i2).get_prime()
                    );
                    return Coefficient::Indeterminate;
                }
                let f = State::get_finite_field(*i1);
                Coefficient::FiniteField(f.one(), *i1)
            }
            (Coefficient::FiniteField(_, _), _) | (_, Coefficient::FiniteField(_, _)) => {
                error!(
                    "Cannot multiply finite field to non-finite number. Convert other number first?"
                );
                Coefficient::Indeterminate
            }
            (Coefficient::Complex(r), Coefficient::RationalPolynomial(rp))
            | (Coefficient::RationalPolynomial(rp), Coefficient::Complex(r)) => {
                if r.is_real() {
                    let p = RationalPolynomial::from_num_den(
                        rp.numerator.constant(r.re.numerator()),
                        rp.numerator.constant(r.re.denominator()),
                        &Z,
                        false,
                    );

                    let g = p.gcd(rp);
                    if g.is_constant() {
                        (g.numerator.lcoeff(), g.denominator.lcoeff()).into()
                    } else {
                        unreachable!()
                    }
                } else {
                    error!("Cannot multiply complex number to rational polynomial");
                    Coefficient::Indeterminate
                }
            }
            (Coefficient::RationalPolynomial(p1), Coefficient::RationalPolynomial(p2)) => {
                let r = if p1.get_variables() != p2.get_variables() {
                    let mut p1 = p1.clone();
                    let mut p2 = p2.clone();
                    p1.unify_variables(&mut p2);
                    p1.gcd(&p2)
                } else {
                    p1.gcd(p2)
                };

                if r.is_constant() {
                    (r.numerator.lcoeff(), r.denominator.lcoeff()).into()
                } else {
                    Coefficient::RationalPolynomial(r)
                }
            }
            (Coefficient::Complex(_), Coefficient::Float(f))
            | (Coefficient::Float(f), Coefficient::Complex(_)) => Coefficient::Float(f.one()),
            (Coefficient::Float(f1), Coefficient::Float(_f2)) => Coefficient::Float(f1.one()),
            (Coefficient::Float(_), _) | (_, Coefficient::Float(_)) => {
                error!("Cannot take gcd of floats and other coefficient types");
                Coefficient::Indeterminate
            }
        }
    }
}

impl Neg for Coefficient {
    type Output = Coefficient;

    fn neg(self) -> Coefficient {
        match self {
            Coefficient::Indeterminate => Coefficient::Indeterminate,
            Coefficient::Infinity(phase) => match phase {
                Some(p) => Coefficient::Infinity(Some(-p)),
                None => Coefficient::Infinity(None),
            },
            Coefficient::Complex(r) => Coefficient::Complex(-r),
            Coefficient::Float(f) => Coefficient::Float(-f),
            Coefficient::FiniteField(n, i) => {
                let f = State::get_finite_field(i);
                Coefficient::FiniteField(f.neg(n), i)
            }
            Coefficient::RationalPolynomial(p) => Coefficient::RationalPolynomial(-p),
        }
    }
}

impl Add for Coefficient {
    type Output = Coefficient;

    fn add(self, rhs: Coefficient) -> Coefficient {
        match (self, rhs) {
            (Coefficient::Complex(r1), Coefficient::Complex(r2)) => Coefficient::Complex(r1 + r2),
            (Coefficient::FiniteField(n1, i1), Coefficient::FiniteField(n2, i2)) => {
                if i1 != i2 {
                    error!(
                        "Cannot add numbers from different finite fields: p1={}, p2={}",
                        State::get_finite_field(i1).get_prime(),
                        State::get_finite_field(i2).get_prime()
                    );
                    return Coefficient::Indeterminate;
                }
                let f = State::get_finite_field(i1);
                Coefficient::FiniteField(f.add(&n1, &n2), i1)
            }
            (Coefficient::FiniteField(_, _), _) | (_, Coefficient::FiniteField(_, _)) => {
                error!("Cannot add finite field to non-finite number. Convert other number first?");
                Coefficient::Indeterminate
            }
            (Coefficient::Complex(r), Coefficient::RationalPolynomial(rp))
            | (Coefficient::RationalPolynomial(rp), Coefficient::Complex(r)) => {
                if r.is_real() {
                    let r2 = RationalPolynomial {
                        numerator: rp.numerator.constant(r.re.numerator()),
                        denominator: rp.denominator.constant(r.re.denominator()),
                    };
                    let res = &rp + &r2;

                    if res.is_constant() {
                        (res.numerator.lcoeff(), res.denominator.lcoeff()).into()
                    } else {
                        Coefficient::RationalPolynomial(res)
                    }
                } else {
                    error!(
                        "Cannot add complex number to rational polynomial. Convert other number first?"
                    );
                    Coefficient::Indeterminate
                }
            }
            (Coefficient::RationalPolynomial(mut p1), Coefficient::RationalPolynomial(mut p2)) => {
                if p1.get_variables() != p2.get_variables() {
                    p1.unify_variables(&mut p2);
                };

                let r = &p1 + &p2;

                if r.is_constant() {
                    (r.numerator.lcoeff(), r.denominator.lcoeff()).into()
                } else {
                    Coefficient::RationalPolynomial(r)
                }
            }
            (Coefficient::Complex(r), Coefficient::Float(f))
            | (Coefficient::Float(f), Coefficient::Complex(r)) => {
                Coefficient::Float(Complex::new(f.re + r.re, f.im + r.im))
            }
            (Coefficient::Float(f1), Coefficient::Float(f2)) => Coefficient::Float(f1 + f2),
            (Coefficient::Float(_), _) | (_, Coefficient::Float(_)) => {
                error!("Cannot add float to finite-field number or rational polynomial");
                Coefficient::Indeterminate
            }
            (Coefficient::Indeterminate, _) => Coefficient::Indeterminate,
            (_, Coefficient::Indeterminate) => Coefficient::Indeterminate,
            (Coefficient::Infinity(phase1), Coefficient::Infinity(phase2)) => {
                match (phase1, phase2) {
                    (Some(p1), Some(p2)) if p1 == p2 => Coefficient::Infinity(Some(p1)),
                    (phase1, phase2) => {
                        warn!(
                            "Created indeterminate by adding {} to {}",
                            Coefficient::Infinity(phase1),
                            Coefficient::Infinity(phase2)
                        );
                        Coefficient::Indeterminate
                    }
                }
            }
            (Coefficient::Infinity(a), _) | (_, Coefficient::Infinity(a)) => {
                Coefficient::Infinity(a)
            }
        }
    }
}

impl Mul for Coefficient {
    type Output = Coefficient;

    fn mul(self, rhs: Coefficient) -> Self::Output {
        match (self, rhs) {
            (Coefficient::Complex(r1), Coefficient::Complex(r2)) => Coefficient::Complex(r1 * r2),
            (Coefficient::FiniteField(n1, i1), Coefficient::FiniteField(n2, i2)) => {
                if i1 != i2 {
                    error!(
                        "Cannot multiply numbers from different finite fields: p1={}, p2={}",
                        State::get_finite_field(i1).get_prime(),
                        State::get_finite_field(i2).get_prime()
                    );
                    return Coefficient::Indeterminate;
                }
                let f = State::get_finite_field(i1);
                Coefficient::FiniteField(f.mul(&n1, &n2), i1)
            }
            (Coefficient::FiniteField(_, _), _) | (_, Coefficient::FiniteField(_, _)) => {
                error!(
                    "Cannot multiply finite field to non-finite number. Convert other number first?"
                );
                Coefficient::Indeterminate
            }
            (Coefficient::Complex(r), Coefficient::RationalPolynomial(mut rp))
            | (Coefficient::RationalPolynomial(mut rp), Coefficient::Complex(r)) => {
                if r.is_real() {
                    let gcd1 = Z.gcd(&r.re.numerator(), &rp.denominator.content());
                    let gcd2 = Z.gcd(&r.re.denominator(), &rp.numerator.content());
                    rp.numerator = rp
                        .numerator
                        .div_coeff(&gcd2)
                        .mul_coeff(r.re.numerator().div(&gcd1));
                    rp.denominator = rp
                        .denominator
                        .div_coeff(&gcd1)
                        .mul_coeff(r.re.denominator().div(&gcd2));
                    Coefficient::RationalPolynomial(rp)
                } else {
                    error!(
                        "Cannot multiply complex number to non-complex number. Convert other number first?"
                    );
                    Coefficient::Indeterminate
                }
            }
            (Coefficient::RationalPolynomial(mut p1), Coefficient::RationalPolynomial(mut p2)) => {
                if p1.get_variables() != p2.get_variables() {
                    p1.unify_variables(&mut p2);
                };

                let r = &p1 * &p2;

                if r.is_constant() {
                    (r.numerator.lcoeff(), r.denominator.lcoeff()).into()
                } else {
                    Coefficient::RationalPolynomial(r)
                }
            }
            (Coefficient::Complex(r), Coefficient::Float(f))
            | (Coefficient::Float(f), Coefficient::Complex(r)) => Coefficient::Float(Complex::new(
                f.re.clone() * r.re.clone() - f.im.clone() * r.im.clone(),
                f.re * r.im + f.im * r.re,
            )),
            (Coefficient::Float(f1), Coefficient::Float(f2)) => Coefficient::Float(f1 * f2),
            (Coefficient::Float(_), _) | (_, Coefficient::Float(_)) => {
                error!("Cannot multiply float to finite-field number or rational polynomial");
                Coefficient::Indeterminate
            }
            (Coefficient::Indeterminate, _) => Coefficient::Indeterminate,
            (_, Coefficient::Indeterminate) => Coefficient::Indeterminate,
            (Coefficient::Infinity(phase1), Coefficient::Infinity(phase2)) => {
                match (phase1, phase2) {
                    (Some(p1), Some(p2)) => Coefficient::normalize_infinity(p1 * p2),
                    _ => Coefficient::Infinity(None),
                }
            }
            (Coefficient::Infinity(a), Coefficient::Complex(r))
            | (Coefficient::Complex(r), Coefficient::Infinity(a)) => {
                if r.is_zero() {
                    warn!(
                        "Created indeterminate by multiplying {} with {}",
                        Coefficient::Infinity(a),
                        Coefficient::Complex(r)
                    );
                    return Coefficient::Indeterminate;
                }
                match a {
                    Some(p) => Coefficient::normalize_infinity(p * r),
                    None => Coefficient::Infinity(None),
                }
            }
            (Coefficient::Infinity(_), _) | (_, Coefficient::Infinity(_)) => {
                error!(
                    "Multiplication of rational polynomial or finite field is not implemented yet"
                );
                Coefficient::Indeterminate
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SerializedRational<'a> {
    Natural(i64, i64),
    Large(SerializedLargeRational<'a>),
}

impl SerializedRational<'_> {
    pub fn is_zero(&self) -> bool {
        match self {
            SerializedRational::Natural(n, _) => *n == 0,
            SerializedRational::Large(r) => r.is_zero(),
        }
    }

    pub fn is_negative(&self) -> bool {
        match self {
            SerializedRational::Natural(n, _) => *n < 0,
            SerializedRational::Large(r) => r.is_negative(),
        }
    }

    pub fn to_rat(&self) -> Rational {
        match self {
            SerializedRational::Natural(n, d) => Rational::from_int_unchecked(*n, *d),
            SerializedRational::Large(r) => r.to_rat(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SerializedLargeRational<'a> {
    pub(crate) is_negative: bool,
    pub(crate) num_digits: &'a [u8],
    pub(crate) den_digits: &'a [u8],
}

impl SerializedLargeRational<'_> {
    pub fn is_zero(&self) -> bool {
        self.num_digits.is_empty()
    }

    pub fn is_negative(&self) -> bool {
        self.is_negative
    }

    pub fn to_rat(&self) -> Rational {
        if self.num_digits.is_empty() {
            return Rational::zero();
        }

        let mut num = rug::Integer::from_digits(self.num_digits, Order::Lsf);
        let den = rug::Integer::from_digits(self.den_digits, Order::Lsf);
        if self.is_negative {
            num.neg_assign();
        }

        Rational::from_int_unchecked(num, den)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SerializedRationalPolynomial<'a>(pub &'a [u8]);

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct SerializedFloat<'a>(pub &'a [u8]);

impl SerializedFloat<'_> {
    pub fn to_float(&self) -> Float {
        let mut d = self.0;
        let prec = d.get_u32_le();
        Float::deserialize(d, prec)
    }

    pub fn is_zero(&self) -> bool {
        self.to_float().is_zero() // TODO: improve
    }
}

/// A view of a coefficient that keeps its complicated variants
/// serialized for efficiency.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum CoefficientView<'a> {
    Indeterminate,
    Infinity(Option<(SerializedRational<'a>, SerializedRational<'a>)>),
    /// A complex number `(n_re, d_re, n_im, d_im)` that represents `n_re/d_re + i * n_im/d_im`.
    Natural(i64, i64, i64, i64),
    /// A floating point number `(re, im)` that represents `re + i * im`.
    Float(SerializedFloat<'a>, SerializedFloat<'a>),
    /// A large complex number `(re, im)` that represents `re + i * im`.
    Large(SerializedRational<'a>, SerializedRational<'a>),
    FiniteField(FiniteFieldElement<u64>, FiniteFieldIndex),
    RationalPolynomial(SerializedRationalPolynomial<'a>),
}

impl ConvertToRing for Q {
    #[inline]
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        number.into()
    }

    #[inline]
    fn try_element_from_coefficient(&self, number: Coefficient) -> Result<Self::Element, String> {
        match number {
            Coefficient::Indeterminate => {
                Err("Cannot convert indeterminate to rational".to_owned())
            }
            Coefficient::Infinity(_) => Err("Cannot convert infinity to rational".to_owned()),
            Coefficient::Complex(r) => {
                if r.is_real() {
                    Ok(r.re)
                } else {
                    Err(format!("Cannot convert {r} to rational"))
                }
            }
            Coefficient::Float(f) => Err(format!("Cannot convert float {f} to rational")),
            Coefficient::FiniteField(_, _) => {
                Err("Cannot convert finite field to rational".to_owned())
            }
            Coefficient::RationalPolynomial(r) => Err(format!(
                "Cannot convert rational polynomial {r} to rational"
            )),
        }
    }

    #[inline]
    fn try_element_from_coefficient_view(
        &self,
        number: CoefficientView<'_>,
    ) -> Result<Rational, String> {
        match number {
            CoefficientView::Indeterminate => {
                Err("Cannot convert indeterminate to rational".to_owned())
            }
            CoefficientView::Infinity(_) => Err("Cannot convert infinity to rational".to_owned()),
            CoefficientView::Natural(r, d, cr, _cd) => {
                if cr == 0 {
                    Ok(Rational::from_int_unchecked(r, d))
                } else {
                    Err(format!(
                        "Cannot convert complex number {} to rational",
                        number.to_owned()
                    ))
                }
            }
            CoefficientView::Large(r, i) => {
                if i.is_zero() {
                    Ok(r.to_rat())
                } else {
                    Err(format!(
                        "Cannot convert complex number {} to rational",
                        number.to_owned()
                    ))
                }
            }
            CoefficientView::Float(_, _) => Err(format!(
                "Cannot convert float {} to rational",
                number.to_owned()
            )),
            CoefficientView::FiniteField(_, _) => {
                Err("Cannot convert finite field to rational".to_owned())
            }
            CoefficientView::RationalPolynomial(_) => Err(format!(
                "Cannot convert rational polynomial {} to rational",
                number.to_owned()
            )),
        }
    }
}

impl ConvertToRing for IntegerRing {
    #[inline]
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        number
    }

    #[inline]
    fn try_element_from_coefficient(&self, number: Coefficient) -> Result<Integer, String> {
        match number {
            Coefficient::Indeterminate => Err("Cannot convert indeterminate to integer".to_owned()),
            Coefficient::Infinity(_) => Err("Cannot convert infinity to integer".to_owned()),
            Coefficient::Complex(r) => {
                if r.is_real() {
                    if r.re.is_integer() {
                        Ok(r.re.numerator())
                    } else {
                        Err("Cannot convert non-integer real number to integer".to_owned())
                    }
                } else {
                    Err(format!("Cannot convert complex number {r} to integer"))
                }
            }
            Coefficient::Float(f) => Err(format!("Cannot convert float {f} to integer")),
            Coefficient::FiniteField(_, _) => {
                Err("Cannot convert finite field to integer".to_owned())
            }
            Coefficient::RationalPolynomial(r) => {
                Err(format!("Cannot convert rational polynomial {r} to integer"))
            }
        }
    }

    #[inline]
    fn try_element_from_coefficient_view(
        &self,
        number: CoefficientView<'_>,
    ) -> Result<Integer, String> {
        match number {
            CoefficientView::Indeterminate => {
                Err("Cannot convert indeterminate to integer".to_owned())
            }
            CoefficientView::Infinity(_) => Err("Cannot convert infinity to integer".to_owned()),
            CoefficientView::Natural(nr, dr, ni, _di) => {
                if dr == 1 && ni == 0 {
                    Ok(Integer::Single(nr))
                } else {
                    Err("Cannot convert rational or complex number to integer".to_owned())
                }
            }
            CoefficientView::Large(r, i) => {
                if !i.is_zero() {
                    return Err(format!(
                        "Cannot convert complex number {} to integer",
                        number.to_owned()
                    ));
                }
                let r = r.to_rat();
                if !r.is_integer() {
                    return Err(format!("Cannot convert rational {r} to integer"));
                }
                Ok(r.numerator())
            }
            CoefficientView::Float(_, _) => Err(format!(
                "Cannot convert float {} to integer",
                number.to_owned()
            )),
            CoefficientView::FiniteField(_, _) => {
                Err("Cannot convert finite field to integer".to_owned())
            }
            CoefficientView::RationalPolynomial(_) => Err(format!(
                "Cannot convert rational polynomial {} to integer",
                number.to_owned()
            )),
        }
    }
}

impl ConvertToRing for AtomField {
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        Atom::num(number)
    }

    fn element_from_coefficient(&self, number: Coefficient) -> Self::Element {
        Atom::num(number)
    }

    fn element_from_coefficient_view(&self, number: CoefficientView<'_>) -> Self::Element {
        Atom::num(number.to_owned())
    }

    fn try_element_from_coefficient(&self, number: Coefficient) -> Result<Self::Element, String> {
        Ok(Atom::num(number))
    }

    fn try_element_from_coefficient_view(
        &self,
        number: CoefficientView<'_>,
    ) -> Result<Self::Element, String> {
        Ok(Atom::num(number.to_owned()))
    }
}

impl<UField: FiniteFieldWorkspace> ConvertToRing for FiniteField<UField>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
    Integer: ToFiniteField<UField>,
{
    #[inline]
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        number.to_finite_field(self)
    }

    #[inline]
    fn try_element_from_coefficient(
        &self,
        number: Coefficient,
    ) -> Result<<FiniteField<UField> as Set>::Element, String> {
        match number {
            Coefficient::Indeterminate => {
                Err("Cannot convert indeterminate to finite field".to_owned())
            }
            Coefficient::Infinity(_) => Err("Cannot convert infinity to finite field".to_owned()),
            Coefficient::Complex(r) => {
                if r.is_real() {
                    Ok(self.div(
                        &r.re.numerator().to_finite_field(self),
                        &r.re.denominator().to_finite_field(self),
                    ))
                } else {
                    Err(format!("Cannot convert complex number {r} to finite field"))
                }
            }
            Coefficient::Float(f) => Err(format!("Cannot convert float {f} to finite field",)),
            Coefficient::FiniteField(_, _) => {
                Err("Cannot convert finite field to other one".to_owned())
            }
            Coefficient::RationalPolynomial(_) => Err(format!(
                "Cannot convert rational polynomial {} to finite field",
                number.to_owned()
            )),
        }
    }

    #[inline]
    fn try_element_from_coefficient_view(
        &self,
        number: CoefficientView<'_>,
    ) -> Result<<FiniteField<UField> as Set>::Element, String> {
        match number {
            CoefficientView::Indeterminate => {
                Err("Cannot convert indeterminate to finite field".to_owned())
            }
            CoefficientView::Infinity(_) => {
                Err("Cannot convert infinity to finite field".to_owned())
            }
            CoefficientView::Natural(n, d, i, _) => {
                if i == 0 {
                    Ok(self.div(
                        &Integer::new(n).to_finite_field(self),
                        &Integer::new(d).to_finite_field(self),
                    ))
                } else {
                    Err("Cannot convert complex number to finite field".to_owned())
                }
            }
            CoefficientView::Large(r, i) => {
                if i.is_zero() {
                    let l = r.to_rat();
                    Ok(self.div(
                        &l.numerator().to_finite_field(self),
                        &l.denominator().to_finite_field(self),
                    ))
                } else {
                    Err(format!(
                        "Cannot convert complex number {} to finite field",
                        number.to_owned()
                    ))
                }
            }
            CoefficientView::Float(_, _) => Err(format!(
                "Cannot convert float {} to finite field",
                number.to_owned()
            )),
            CoefficientView::FiniteField(_, _) => {
                Err("Cannot convert finite field to other one".to_owned())
            }
            CoefficientView::RationalPolynomial(_) => Err(format!(
                "Cannot convert rational polynomial {} to finite field",
                number.to_owned()
            )),
        }
    }
}

impl ConvertToRing for FloatField<F64> {
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        self.get_rep().from_rational(&number.into())
    }

    fn try_element_from_coefficient(
        &self,
        number: crate::coefficient::Coefficient,
    ) -> Result<Self::Element, String> {
        match number {
            crate::coefficient::Coefficient::Complex(complex) => {
                if complex.is_real() {
                    Ok(F64(complex.re.to_f64()))
                } else {
                    Err(format!("Cannot convert {complex} to real float"))
                }
            }
            crate::coefficient::Coefficient::Float(complex) => {
                if complex.is_real() {
                    Ok(F64(complex.re.to_f64()))
                } else {
                    Err(format!("Cannot convert {complex} to real float"))
                }
            }
            _ => Err(format!("Cannot convert {number} to complex")),
        }
    }

    fn try_element_from_coefficient_view(
        &self,
        number: crate::coefficient::CoefficientView<'_>,
    ) -> Result<Self::Element, String> {
        self.try_element_from_coefficient(number.to_owned())
    }
}

impl ConvertToRing for FloatField<Float> {
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        self.get_rep().from_rational(&number.into())
    }

    fn try_element_from_coefficient(
        &self,
        number: crate::coefficient::Coefficient,
    ) -> Result<Self::Element, String> {
        match number {
            crate::coefficient::Coefficient::Complex(complex) => {
                if complex.is_real() {
                    Ok(complex.re.to_multi_prec_float(self.get_rep().prec()))
                } else {
                    Err(format!("Cannot convert {complex} to real float"))
                }
            }
            crate::coefficient::Coefficient::Float(complex) => {
                if complex.is_real() {
                    Ok(complex.re)
                } else {
                    Err(format!("Cannot convert {complex} to real float"))
                }
            }
            _ => Err(format!("Cannot convert {number} to complex")),
        }
    }

    fn try_element_from_coefficient_view(
        &self,
        number: crate::coefficient::CoefficientView<'_>,
    ) -> Result<Self::Element, String> {
        self.try_element_from_coefficient(number.to_owned())
    }
}

impl ConvertToRing for FloatField<Complex<F64>> {
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        self.get_rep().from_rational(&number.into())
    }

    fn try_element_from_coefficient(
        &self,
        number: crate::coefficient::Coefficient,
    ) -> Result<Self::Element, String> {
        match number {
            crate::coefficient::Coefficient::Complex(complex) => Ok(Complex::new(
                complex.re.to_f64().into(),
                complex.im.to_f64().into(),
            )),
            crate::coefficient::Coefficient::Float(complex) => Ok(Complex::new(
                complex.re.to_f64().into(),
                complex.im.to_f64().into(),
            )),
            _ => Err(format!("Cannot convert {number} to complex")),
        }
    }

    fn try_element_from_coefficient_view(
        &self,
        number: crate::coefficient::CoefficientView<'_>,
    ) -> Result<Self::Element, String> {
        self.try_element_from_coefficient(number.to_owned())
    }
}

impl ConvertToRing for FloatField<Complex<Float>> {
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        self.get_rep().from_rational(&number.into())
    }

    fn try_element_from_coefficient(
        &self,
        number: crate::coefficient::Coefficient,
    ) -> Result<Self::Element, String> {
        match number {
            crate::coefficient::Coefficient::Complex(complex) => Ok(Complex::new(
                complex.re.to_multi_prec_float(self.get_rep().re.prec()),
                complex.im.to_multi_prec_float(self.get_rep().im.prec()),
            )),
            crate::coefficient::Coefficient::Float(complex) => Ok(complex),
            _ => Err(format!("Cannot convert {number} to complex")),
        }
    }

    fn try_element_from_coefficient_view(
        &self,
        number: crate::coefficient::CoefficientView<'_>,
    ) -> Result<Self::Element, String> {
        self.try_element_from_coefficient(number.to_owned())
    }
}

impl ConvertToRing for FloatField<Complex<Rational>> {
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        self.get_rep().from_rational(&number.into())
    }

    fn try_element_from_coefficient(
        &self,
        number: crate::coefficient::Coefficient,
    ) -> Result<Self::Element, String> {
        match number {
            crate::coefficient::Coefficient::Complex(complex) => Ok(complex),
            _ => Err(format!("Cannot convert {number} to complex rational")),
        }
    }

    fn try_element_from_coefficient_view(
        &self,
        number: crate::coefficient::CoefficientView<'_>,
    ) -> Result<Self::Element, String> {
        self.try_element_from_coefficient(number.to_owned())
    }
}

impl ConvertToRing for AlgebraicExtension<Q> {
    #[inline]
    fn element_from_integer(&self, number: Integer) -> Self::Element {
        self.constant(number.into())
    }

    #[inline]
    fn try_element_from_coefficient(&self, number: Coefficient) -> Result<Self::Element, String> {
        match number {
            Coefficient::Indeterminate => {
                Err("Cannot convert indeterminate to rational".to_owned())
            }
            Coefficient::Infinity(_) => Err("Cannot convert infinity to rational".to_owned()),
            Coefficient::Complex(r) => {
                if r.is_real() {
                    Ok(self.constant(r.re.clone()))
                } else if self.poly().exponents == [0, 2]
                    && self.poly().get_constant() == Rational::one()
                {
                    Ok(self.to_element(
                        self.poly().monomial(r.im.clone(), vec![1])
                            + self.poly().constant(r.re.clone()),
                    ))
                } else {
                    Err(
                        "Cannot directly convert complex number to this extension. First create a polynomial with extension x^2+1 and then upgrade."
                            .to_owned(),
                    )
                }
            }
            Coefficient::Float(_) => Err(format!("Cannot convert float {} to extension", number)),
            Coefficient::FiniteField(_, _) => Err(format!(
                "Cannot convert finite field {} to extension",
                number
            )),
            Coefficient::RationalPolynomial(_) => {
                // TODO: this may be possible!
                Err("Cannot convert rational polynomial to extension".to_owned())
            }
        }
    }

    #[inline]
    fn try_element_from_coefficient_view(
        &self,
        number: CoefficientView<'_>,
    ) -> Result<Self::Element, String> {
        match number {
            CoefficientView::Indeterminate => {
                Err("Cannot convert indeterminate to rational".to_owned())
            }
            CoefficientView::Infinity(_) => Err("Cannot convert infinity to rational".to_owned()),
            CoefficientView::Natural(r, d, cr, cd) => {
                if cr == 0 {
                    Ok(self.constant(Rational::from_int_unchecked(r, d)))
                } else if self.poly().exponents == [0, 2]
                    && self.poly().get_constant() == Rational::one()
                {
                    Ok(self.to_element(
                        self.poly()
                            .monomial(Rational::from_int_unchecked(cr, cd), vec![1])
                            + self.poly().constant(Rational::from_int_unchecked(r, d)),
                    ))
                } else {
                    Err(
                        "Cannot directly convert complex number to this extension. First create a polynomial with extension x^2+1 and then upgrade."
                            .to_owned(),
                    )
                }
            }
            CoefficientView::Large(r, i) => {
                if i.is_zero() {
                    Ok(self.constant(r.to_rat()))
                } else if self.poly().exponents == [0, 2]
                    && self.poly().get_constant() == Rational::one()
                {
                    Ok(self.to_element(
                        self.poly().monomial(i.to_rat(), vec![1])
                            + self.poly().constant(r.to_rat()),
                    ))
                } else {
                    Err(
                        "Cannot directly convert complex number to this extension. First create a polynomial with extension x^2+1 and then upgrade."
                            .to_owned(),
                    )
                }
            }
            CoefficientView::Float(_, _) => Err(format!(
                "Cannot convert float {} to rational",
                number.to_owned()
            )),
            CoefficientView::FiniteField(_, _) => {
                Err("Cannot convert finite field to rational".to_owned())
            }
            CoefficientView::RationalPolynomial(_) => Err(format!(
                "Cannot convert rational polynomial {} to rational",
                number.to_owned()
            )),
        }
    }
}

impl CoefficientView<'_> {
    pub fn normalize(&self) -> Coefficient {
        match self {
            CoefficientView::Natural(nr, dr, ni, di) => {
                Coefficient::Complex(Complex::new((*nr, *dr).into(), (*ni, *di).into()))
            }
            CoefficientView::Float(_, _)
            | CoefficientView::Large(_, _)
            | CoefficientView::FiniteField(_, _)
            | CoefficientView::RationalPolynomial(_) => self.to_owned(),
            CoefficientView::Indeterminate => Coefficient::Indeterminate,
            CoefficientView::Infinity(None) => Coefficient::Infinity(None),
            CoefficientView::Infinity(Some(_)) => self.to_owned(),
        }
    }

    pub fn to_owned(&self) -> Coefficient {
        match self {
            CoefficientView::Natural(nr, dr, ni, di) => Coefficient::Complex(Complex::new(
                Rational::from_int_unchecked(*nr, *dr),
                Rational::from_int_unchecked(*ni, *di),
            )),
            CoefficientView::Large(rr, ri) => {
                Coefficient::Complex(Complex::new(rr.to_rat(), ri.to_rat()))
            }
            CoefficientView::Float(r, i) => {
                Coefficient::Float(Complex::new(r.to_float(), i.to_float()))
            }
            CoefficientView::FiniteField(num, field) => Coefficient::FiniteField(*num, *field),
            CoefficientView::RationalPolynomial(p) => {
                Coefficient::RationalPolynomial(p.deserialize())
            }
            CoefficientView::Indeterminate => Coefficient::Indeterminate,
            CoefficientView::Infinity(None) => Coefficient::Infinity(None),
            CoefficientView::Infinity(Some((rr, ri))) => {
                Coefficient::Infinity(Some(Complex::new(rr.to_rat(), ri.to_rat())))
            }
        }
    }

    pub fn is_zero(&self) -> bool {
        match self {
            CoefficientView::Natural(n, _, ni, _) => *n == 0 && *ni == 0,
            CoefficientView::Large(r, i) => r.is_zero() && i.is_zero(),
            CoefficientView::Float(r, i) => r.is_zero() && i.is_zero(),
            CoefficientView::FiniteField(num, _) => *num.inner() == 0,
            CoefficientView::RationalPolynomial(p) => p.deserialize().is_zero(),
            CoefficientView::Indeterminate | CoefficientView::Infinity(_) => false,
        }
    }

    pub fn is_one(&self) -> bool {
        match self {
            CoefficientView::Natural(n, d, ni, di) => *n == *d && *ni == 0 && *di == 1,
            CoefficientView::Large(r, i) => i.is_zero() && r.to_rat().is_one(),
            CoefficientView::Float(r, i) => i.is_zero() && r.to_float().is_one(),
            CoefficientView::FiniteField(num, field) => {
                let f = State::get_finite_field(*field);
                f.is_one(num)
            }
            CoefficientView::RationalPolynomial(p) => {
                let r = p.deserialize();
                r.numerator.is_one() && r.denominator.is_one()
            }
            CoefficientView::Indeterminate | CoefficientView::Infinity(_) => false,
        }
    }

    pub fn pow(&self, other: &CoefficientView<'_>) -> (Coefficient, Coefficient, Coefficient) {
        if let CoefficientView::Natural(0, _, 0, _) = self {
            let r = match other {
                CoefficientView::Indeterminate => Coefficient::Indeterminate,
                CoefficientView::Infinity(None) => {
                    warn!(
                        "Created indeterminate by raising 0 to the power of {}",
                        Coefficient::Infinity(None)
                    );
                    Coefficient::Indeterminate
                }
                CoefficientView::Infinity(Some((r, i))) => {
                    if i.is_zero() {
                        if !r.is_negative() {
                            Coefficient::zero()
                        } else {
                            Coefficient::Infinity(None)
                        }
                    } else {
                        warn!(
                            "Created indeterminate by raising 0 to the power of {}",
                            other.to_owned()
                        );
                        Coefficient::Indeterminate
                    }
                }
                CoefficientView::Natural(r, _, _, _) => {
                    if *r < 0 {
                        warn!("Created infinity by raising 0 to the power of {}", r);
                        Coefficient::Infinity(None)
                    } else if *r == 0 {
                        warn!("Created indeterminate by raising 0 to the power of 0");
                        Coefficient::Indeterminate
                    } else {
                        Coefficient::zero()
                    }
                }
                CoefficientView::Float(_, _) => {
                    error!("Cannot exponentiate zero with float exponent");
                    Coefficient::Indeterminate
                }
                CoefficientView::Large(r, _) => {
                    if r.is_negative() {
                        warn!(
                            "Created infinity by raising 0 to the power of {}",
                            r.to_rat()
                        );
                        Coefficient::Infinity(None)
                    } else if r.is_zero() {
                        warn!("Created indeterminate by raising 0 to the power of 0");
                        Coefficient::Indeterminate
                    } else {
                        Coefficient::zero()
                    }
                }
                CoefficientView::FiniteField(_, _) => {
                    error!("Cannot exponentiate zero with finite field exponent");
                    Coefficient::Indeterminate
                }
                CoefficientView::RationalPolynomial(_) => {
                    error!("Cannot exponentiate zero with rational polynomial exponent");
                    Coefficient::Indeterminate
                }
            };

            return (Coefficient::one(), r, Coefficient::one());
        }

        // cannot simplify complex exponent
        if let CoefficientView::Natural(_, _, n, _) = other {
            if *n != 0 {
                return (Coefficient::one(), self.to_owned(), other.to_owned());
            }
        } else if let CoefficientView::Large(_, n) = other
            && !n.is_zero()
        {
            return (Coefficient::one(), self.to_owned(), other.to_owned());
        }

        fn rat_pow(
            mut base: Rational,
            mut exp: Rational,
        ) -> (Complex<Rational>, Rational, Rational) {
            if base.is_negative() && !exp.is_integer() {
                let pow = exp.numerator() / exp.denominator();
                let rest = exp.numerator() - &pow * exp.denominator();

                let base_integer_pow = if pow.is_negative() {
                    base.inv().pow(pow.to_i64().unwrap().unsigned_abs())
                } else {
                    base.pow(pow.to_i64().unwrap().unsigned_abs())
                };

                if exp.denominator_ref() == &2 {
                    (
                        if rest.is_negative() {
                            Complex::new(Rational::zero(), -base_integer_pow)
                        } else {
                            Complex::new(Rational::zero(), base_integer_pow)
                        },
                        base.abs(),
                        Rational::from_int_unchecked(rest, exp.denominator()),
                    )
                } else {
                    (
                        base_integer_pow.into(),
                        base,
                        Rational::from_int_unchecked(rest, exp.denominator()),
                    )
                }
            } else if base.is_one() {
                (Rational::one().into(), Rational::one(), Rational::one())
            } else {
                if exp < 0 {
                    base = base.inv();
                    exp = -exp;
                }

                base = base.pow(exp.numerator().to_i64().unwrap().unsigned_abs());
                (
                    Rational::one().into(),
                    base,
                    Rational::from_int_unchecked(Integer::one(), exp.denominator()),
                )
            }
        }

        // TODO: normalize 4^1/3 to 2^(2/3)?
        match (self, other) {
            (
                &CoefficientView::Natural(n1, d1, ni1, di1),
                &CoefficientView::Natural(n2, d2, _, _),
            ) => {
                if ni1 == 0 {
                    let (coeff, base, exp) = rat_pow(
                        Rational::from_int_unchecked(n1, d1),
                        Rational::from_int_unchecked(n2, d2),
                    );
                    (coeff.into(), base.into(), exp.into())
                } else if d2 == 1 {
                    let r = Complex::new(
                        Rational::from_int_unchecked(n1, d1),
                        Rational::from_int_unchecked(ni1, di1),
                    )
                    .pow(n2.unsigned_abs());

                    (
                        Coefficient::one(),
                        if n2 < 0 { r.inv().into() } else { r.into() },
                        Rational::one().into(),
                    )
                } else {
                    (
                        Coefficient::one(),
                        Complex::new(
                            Rational::from_int_unchecked(n1, d1),
                            Rational::from_int_unchecked(ni1, di1),
                        )
                        .into(),
                        Rational::from_int_unchecked(n2, d2).into(),
                    )
                }
            }
            (&CoefficientView::Natural(n1, d1, ni1, di1), &CoefficientView::Large(n2, _)) => {
                if ni1 == 0 {
                    let (coeff, base, exp) =
                        rat_pow(Rational::from_int_unchecked(n1, d1), n2.to_rat());
                    (coeff.into(), base.into(), exp.into())
                } else {
                    (
                        Coefficient::one(),
                        Complex::new(
                            Rational::from_int_unchecked(n1, d1),
                            Rational::from_int_unchecked(ni1, di1),
                        )
                        .into(),
                        n2.to_rat().into(),
                    )
                }
            }
            (&CoefficientView::RationalPolynomial(r), &CoefficientView::Natural(n2, d2, _, _)) => {
                if n2.unsigned_abs() > u32::MAX as u64 {
                    error!("Power is too large: {n2}");
                    return (
                        Coefficient::one(),
                        Coefficient::Indeterminate,
                        Coefficient::one(),
                    );
                }

                if n2 < 0 {
                    let r = r.deserialize().inv();
                    (
                        Coefficient::one(),
                        Coefficient::RationalPolynomial(r.pow(n2.unsigned_abs())),
                        Rational::from_int_unchecked(1, d2).into(),
                    )
                } else {
                    (
                        Coefficient::one(),
                        Coefficient::RationalPolynomial(r.deserialize().pow(n2 as u64)),
                        Rational::from_int_unchecked(1, d2).into(),
                    )
                }
            }
            (&CoefficientView::Large(r, i), &CoefficientView::Natural(n2, d2, _, _)) => {
                if i.is_zero() {
                    let (coeff, base, exp) =
                        rat_pow(r.to_rat(), Rational::from_int_unchecked(n2, d2));
                    (coeff.into(), base.into(), exp.into())
                } else if d2 == 1 {
                    let r = Complex::new(r.to_rat(), i.to_rat()).pow(n2.unsigned_abs());

                    (
                        Coefficient::one(),
                        if n2 < 0 { r.inv().into() } else { r.into() },
                        Rational::one().into(),
                    )
                } else {
                    (
                        Coefficient::one(),
                        Complex::new(r.to_rat(), i.to_rat()).into(),
                        Rational::from_int_unchecked(n2, d2).into(),
                    )
                }
            }
            (&CoefficientView::Float(fr, fi), &CoefficientView::Natural(n2, d2, ni2, di2)) => {
                if d2 == 1 && ni2 == 0 {
                    let r = Complex::new(fr.to_float(), fi.to_float()).pow(n2.unsigned_abs());
                    (
                        Coefficient::one(),
                        if n2 < 0 { r.inv().into() } else { r.into() },
                        Coefficient::one(),
                    )
                } else {
                    // FIXME: what precision should be used?
                    let f = fr.to_float();
                    let p = f.prec();
                    (
                        Coefficient::one(),
                        Complex::new(f, fi.to_float())
                            .powf(&Complex::new(
                                Rational::from_int_unchecked(n2, d2).to_multi_prec_float(p),
                                Rational::from_int_unchecked(ni2, di2).to_multi_prec_float(p),
                            ))
                            .into(),
                        Coefficient::one(),
                    )
                }
            }
            (&CoefficientView::Float(fr, fi), &CoefficientView::Large(r, d)) => {
                let f = fr.to_float();
                let p = f.prec();
                (
                    Coefficient::one(),
                    Complex::new(f, fi.to_float())
                        .powf(&Complex::new(
                            r.to_rat().to_multi_prec_float(p),
                            d.to_rat().to_multi_prec_float(p),
                        ))
                        .into(),
                    Coefficient::one(),
                )
            }
            (&CoefficientView::Natural(n2, d2, ni2, di2), &CoefficientView::Float(fr, fi)) => {
                let f = fr.to_float();
                let p = f.prec();
                (
                    Coefficient::one(),
                    Complex::new(
                        Rational::from_int_unchecked(n2, d2).to_multi_prec_float(p),
                        Rational::from_int_unchecked(ni2, di2).to_multi_prec_float(p),
                    )
                    .powf(&Complex::new(f, fi.to_float()))
                    .into(),
                    Coefficient::one(),
                )
            }
            (&CoefficientView::Large(r, d), &CoefficientView::Float(fr, fi)) => {
                let f = fr.to_float();
                let p = f.prec();
                (
                    Coefficient::one(),
                    Complex::new(
                        r.to_rat().to_multi_prec_float(p),
                        d.to_rat().to_multi_prec_float(p),
                    )
                    .powf(&Complex::new(f, fi.to_float()))
                    .into(),
                    Coefficient::one(),
                )
            }
            (&CoefficientView::Float(fr1, fi1), &CoefficientView::Float(fr2, fi2)) => {
                let p = if fi1.is_zero() && fi2.is_zero() {
                    fr1.to_float().powf(&fr2.to_float()).into()
                } else {
                    Complex::new(fr1.to_float(), fi1.to_float())
                        .powf(&Complex::new(fr2.to_float(), fi2.to_float()))
                        .into()
                };

                (Coefficient::one(), p, Coefficient::one())
            }
            (&CoefficientView::Large(r, i), &CoefficientView::Large(er, _)) => {
                if i.is_zero() {
                    let (coeff, base, exp) = rat_pow(r.to_rat(), er.to_rat());
                    (coeff.into(), base.into(), exp.into())
                } else {
                    (
                        Coefficient::one(),
                        Complex::new(r.to_rat(), i.to_rat()).into(),
                        er.to_rat().into(),
                    )
                }
            }
            (&CoefficientView::Infinity(d), CoefficientView::Natural(r, rd, i, _)) => {
                if *r < 0 {
                    (Coefficient::one(), Coefficient::zero(), Coefficient::one())
                } else if *r == 0 {
                    warn!(
                        "Created indeterminate by raising {} to the power of {}",
                        self.to_owned(),
                        other.to_owned()
                    );
                    (
                        Coefficient::one(),
                        Coefficient::Indeterminate,
                        Coefficient::one(),
                    )
                } else if *i == 0
                    && let Some((rr, ri)) = d
                {
                    let dir = Complex::new(rr.to_rat(), ri.to_rat());
                    if *rd == 1 {
                        (
                            Coefficient::one(),
                            Coefficient::normalize_infinity(dir.pow(*r as u64)),
                            Coefficient::one(),
                        )
                    } else {
                        error!(
                            "Cannot express infinite direction ({})^({}/{}) as single complex number",
                            dir, r, rd
                        );
                        (
                            Coefficient::one(),
                            Coefficient::Infinity(None),
                            Coefficient::one(),
                        )
                    }
                } else {
                    (
                        Coefficient::one(),
                        Coefficient::Infinity(None),
                        Coefficient::one(),
                    )
                }
            }
            (&CoefficientView::Infinity(d), CoefficientView::Large(r, i)) => {
                if r.is_negative() {
                    (Coefficient::one(), Coefficient::zero(), Coefficient::one())
                } else if r.is_zero() {
                    warn!(
                        "Created indeterminate by raising {} to the power of {}",
                        self.to_owned(),
                        other.to_owned()
                    );
                    (
                        Coefficient::one(),
                        Coefficient::Indeterminate,
                        Coefficient::one(),
                    )
                } else if i.is_zero()
                    && let Some((rr, ri)) = d
                {
                    let r = r.to_rat();
                    let dir = Complex::new(rr.to_rat(), ri.to_rat());
                    if r.is_integer() {
                        error!("Power is too large: {r}");
                        (
                            Coefficient::one(),
                            Coefficient::Infinity(None),
                            Coefficient::one(),
                        )
                    } else {
                        error!(
                            "Cannot express infinite direction ({})^({}) as single complex number",
                            dir, r
                        );
                        (
                            Coefficient::one(),
                            Coefficient::Infinity(None),
                            Coefficient::one(),
                        )
                    }
                } else {
                    (
                        Coefficient::one(),
                        Coefficient::Infinity(None),
                        Coefficient::one(),
                    )
                }
            }
            (&CoefficientView::Infinity(d), CoefficientView::Float(r, i)) => {
                let r = r.to_float();
                if r.is_negative() {
                    (Coefficient::one(), Coefficient::zero(), Coefficient::one())
                } else if r.is_zero() {
                    warn!(
                        "Created indeterminate by raising {} to the power of {}",
                        self.to_owned(),
                        other.to_owned()
                    );
                    (
                        Coefficient::one(),
                        Coefficient::Indeterminate,
                        Coefficient::one(),
                    )
                } else if i.is_zero() && d.is_some() {
                    error!("Cannot express infinite float direction as single complex number");
                    (
                        Coefficient::one(),
                        Coefficient::Infinity(None),
                        Coefficient::one(),
                    )
                } else {
                    (
                        Coefficient::one(),
                        Coefficient::Infinity(None),
                        Coefficient::one(),
                    )
                }
            }
            (&CoefficientView::Indeterminate, _) | (_, &CoefficientView::Indeterminate) => (
                Coefficient::one(),
                Coefficient::Indeterminate,
                Coefficient::one(),
            ),
            (_, CoefficientView::Infinity(None)) => {
                warn!(
                    "Created indeterminate by raising {} to the power of {}",
                    self.to_owned(),
                    other.to_owned()
                );
                (
                    Coefficient::one(),
                    Coefficient::Indeterminate,
                    Coefficient::one(),
                )
            }
            (_, CoefficientView::Infinity(Some((r, i)))) => {
                let r = r.to_rat();
                if i.is_zero() {
                    let c = match self {
                        CoefficientView::Natural(rn, rd, ir, id) => {
                            Complex::new((*rn, *rd).into(), (*ir, *id).into())
                        }
                        CoefficientView::Large(rn, ir) => Complex::new(rn.to_rat(), ir.to_rat()),
                        CoefficientView::Indeterminate => {
                            return (
                                Coefficient::one(),
                                Coefficient::Indeterminate,
                                Coefficient::one(),
                            );
                        }
                        CoefficientView::Infinity(_) => {
                            return (
                                Coefficient::one(),
                                Coefficient::Infinity(None),
                                Coefficient::one(),
                            );
                        }
                        _ => {
                            error!(
                                "Cannot simplify infinite exponent with float, finite field or rational polynomial base"
                            );
                            return (
                                Coefficient::one(),
                                Coefficient::Indeterminate,
                                Coefficient::one(),
                            );
                        }
                    };

                    let norm = c.norm_squared();

                    if norm == Rational::one() {
                        warn!(
                            "Created indeterminate by raising {} to the power of {}",
                            self.to_owned(),
                            other.to_owned()
                        );

                        return (
                            Coefficient::one(),
                            Coefficient::Indeterminate,
                            Coefficient::one(),
                        );
                    }

                    if norm < Rational::one() && r > Rational::zero()
                        || norm > Rational::one() && r < Rational::zero()
                    {
                        (Coefficient::one(), Coefficient::zero(), Coefficient::one())
                    } else {
                        (
                            Coefficient::one(),
                            Coefficient::Infinity(None),
                            Coefficient::one(),
                        )
                    }
                } else {
                    match self {
                        CoefficientView::Natural(rn, _, ir, _) => {
                            return (
                                Coefficient::one(),
                                if *rn > 0 && *ir == 0 {
                                    Coefficient::Infinity(None)
                                } else {
                                    warn!(
                                        "Created indeterminate by raising {} to the power of {}",
                                        self.to_owned(),
                                        other.to_owned()
                                    );

                                    Coefficient::Indeterminate
                                },
                                Coefficient::one(),
                            );
                        }
                        CoefficientView::Large(rn, ir) => {
                            return (
                                Coefficient::one(),
                                if !rn.is_negative() && !rn.is_zero() && ir.is_zero() {
                                    Coefficient::Infinity(None)
                                } else {
                                    warn!(
                                        "Created indeterminate by raising {} to the power of {}",
                                        self.to_owned(),
                                        other.to_owned()
                                    );

                                    Coefficient::Indeterminate
                                },
                                Coefficient::one(),
                            );
                        }
                        CoefficientView::Indeterminate => {
                            return (
                                Coefficient::one(),
                                Coefficient::Indeterminate,
                                Coefficient::one(),
                            );
                        }
                        CoefficientView::Infinity(_) => {
                            return (
                                Coefficient::one(),
                                Coefficient::Infinity(None),
                                Coefficient::one(),
                            );
                        }
                        _ => {
                            error!(
                                "Cannot simplify infinite exponent with float, finite field or rational polynomial base"
                            );
                            (
                                Coefficient::one(),
                                Coefficient::Indeterminate,
                                Coefficient::one(),
                            )
                        }
                    }
                }
            }
            (&CoefficientView::Natural(_, _, _, _), &CoefficientView::RationalPolynomial(_)) => {
                (Coefficient::one(), self.to_owned(), other.to_owned())
            }
            (&CoefficientView::Large(_, _), &CoefficientView::RationalPolynomial(_)) => {
                (Coefficient::one(), self.to_owned(), other.to_owned())
            }
            (&CoefficientView::FiniteField(f, d), CoefficientView::Natural(n, 1, 0, 1)) => {
                let field = State::get_finite_field(d);
                let r = if *n < 0 {
                    let f = field.inv(&f);
                    field.pow(&f, n.unsigned_abs())
                } else {
                    field.pow(&f, n.unsigned_abs())
                };
                (
                    Coefficient::one(),
                    Coefficient::FiniteField(r, d),
                    Coefficient::one(),
                )
            }
            (&CoefficientView::RationalPolynomial(_), _) => {
                (Coefficient::one(), self.to_owned(), other.to_owned())
            }
            (&CoefficientView::FiniteField(_, _), _) => {
                error!(
                    "Cannot exponentiate finite field with non-integer exponent or large {:?}",
                    other
                );
                (
                    Coefficient::one(),
                    Coefficient::Indeterminate,
                    Coefficient::one(),
                )
            }
            (_, CoefficientView::FiniteField(_, _)) => {
                error!("Cannot exponentiate with finite field exponent {:?}", other);
                (
                    Coefficient::one(),
                    Coefficient::Indeterminate,
                    Coefficient::one(),
                )
            }
            (&CoefficientView::Infinity(_), CoefficientView::RationalPolynomial(_)) => {
                warn!(
                    "Created indeterminate by raising {} to the power of {}",
                    self.to_owned(),
                    other.to_owned()
                );
                (
                    Coefficient::one(),
                    Coefficient::Indeterminate,
                    Coefficient::one(),
                )
            }
            (_, CoefficientView::RationalPolynomial(_)) => {
                error!("Cannot exponentiate with rational polynomial exponent");
                (
                    Coefficient::one(),
                    Coefficient::Indeterminate,
                    Coefficient::one(),
                )
            }
        }
    }

    pub fn is_integer(&self) -> bool {
        match self {
            CoefficientView::Natural(_, d, i, _) => *i == 0 && *d == 1,
            CoefficientView::Float(_, _) => false,
            CoefficientView::Large(r, d) => d.is_zero() && r.to_rat().is_integer(),
            CoefficientView::FiniteField(_, _) => true,
            CoefficientView::RationalPolynomial(_) => false,
            CoefficientView::Indeterminate => false,
            CoefficientView::Infinity(_) => false,
        }
    }

    pub fn is_real(&self) -> bool {
        match self {
            CoefficientView::Natural(_, _, i, _) => *i == 0,
            CoefficientView::Float(_, i) => i.is_zero(),
            CoefficientView::Large(_, i) => i.is_zero(),
            CoefficientView::FiniteField(_, _) => true,
            CoefficientView::RationalPolynomial(_) => true,
            CoefficientView::Indeterminate => false,
            CoefficientView::Infinity(None) => false,
            CoefficientView::Infinity(Some((_, i))) => i.is_zero(),
        }
    }
}

impl PartialOrd for CoefficientView<'_> {
    fn partial_cmp(&self, other: &CoefficientView) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for CoefficientView<'_> {
    fn cmp(&self, other: &CoefficientView) -> Ordering {
        match (self, other) {
            (
                CoefficientView::Natural(n1, d1, ni1, di1),
                CoefficientView::Natural(n2, d2, ni2, di2),
            ) => Rational::from_int_unchecked(*n1, *d1)
                .cmp(&Rational::from_int_unchecked(*n2, *d2))
                .then_with(|| {
                    Rational::from_int_unchecked(*ni1, *di1)
                        .cmp(&Rational::from_int_unchecked(*ni2, *di2))
                }),
            (CoefficientView::Large(n1, d1), CoefficientView::Large(n2, d2)) => n1
                .to_rat()
                .cmp(&n2.to_rat())
                .then_with(|| d1.to_rat().cmp(&d2.to_rat())),
            (CoefficientView::FiniteField(n1, _), CoefficientView::FiniteField(n2, _)) => {
                n1.inner().cmp(&n2.inner())
            }
            (CoefficientView::Natural(n1, d1, ni1, di1), CoefficientView::Large(n2, d)) => {
                Rational::from_int_unchecked(*n1, *d1)
                    .cmp(&n2.to_rat())
                    .then_with(|| Rational::from_int_unchecked(*ni1, *di1).cmp(&d.to_rat()))
            }
            (CoefficientView::Large(n1, d), CoefficientView::Natural(n2, d2, ni2, di2)) => n1
                .to_rat()
                .cmp(&Rational::from_int_unchecked(*n2, *d2))
                .then_with(|| d.to_rat().cmp(&Rational::from_int_unchecked(*ni2, *di2))),
            (CoefficientView::Float(fr1, fi1), CoefficientView::Float(fr2, fi2)) => fr1
                .to_float()
                .partial_cmp(&fr2.to_float())
                .unwrap_or(Ordering::Equal)
                .then_with(|| {
                    fi1.to_float()
                        .partial_cmp(&fi2.to_float())
                        .unwrap_or(Ordering::Equal)
                }),
            (CoefficientView::RationalPolynomial(n1), CoefficientView::RationalPolynomial(n2)) => {
                n1.deserialize().internal_cmp(&n2.deserialize())
            }
            (CoefficientView::Indeterminate, CoefficientView::Indeterminate) => Ordering::Equal,
            (CoefficientView::Infinity(_), CoefficientView::Infinity(_)) => Ordering::Equal,
            (CoefficientView::Indeterminate, _) => Ordering::Less,
            (_, CoefficientView::Indeterminate) => Ordering::Greater,
            (CoefficientView::Infinity(_), _) => Ordering::Less,
            (_, CoefficientView::Infinity(_)) => Ordering::Greater,
            (CoefficientView::Natural(_, _, _, _), _) => Ordering::Less,
            (_, CoefficientView::Natural(_, _, _, _)) => Ordering::Greater,
            (CoefficientView::Large(_, _), _) => Ordering::Less,
            (_, CoefficientView::Large(_, _)) => Ordering::Greater,
            (CoefficientView::Float(_, _), _) => Ordering::Less,
            (_, CoefficientView::Float(_, _)) => Ordering::Greater,
            (CoefficientView::FiniteField(_, _), _) => Ordering::Less,
            (_, CoefficientView::FiniteField(_, _)) => Ordering::Greater,
        }
    }
}

impl Add<CoefficientView<'_>> for CoefficientView<'_> {
    type Output = Coefficient;

    fn add(self, other: CoefficientView<'_>) -> Coefficient {
        match (self, other) {
            (
                CoefficientView::Natural(n1, d1, ni1, di1),
                CoefficientView::Natural(n2, d2, ni2, di2),
            ) => Coefficient::Complex(
                Complex::new(
                    Rational::from_int_unchecked(n1, d1),
                    Rational::from_int_unchecked(ni1, di1),
                ) + Complex::new(
                    Rational::from_int_unchecked(n2, d2),
                    Rational::from_int_unchecked(ni2, di2),
                ),
            ),
            (CoefficientView::Natural(n1, d1, ni1, di1), CoefficientView::Large(r2, d))
            | (CoefficientView::Large(r2, d), CoefficientView::Natural(n1, d1, ni1, di1)) => {
                Coefficient::Complex(
                    Complex::new(r2.to_rat(), d.to_rat())
                        + Complex::new(
                            Rational::from_int_unchecked(n1, d1),
                            Rational::from_int_unchecked(ni1, di1),
                        ),
                )
            }
            (CoefficientView::Large(r1, d1), CoefficientView::Large(r2, d2)) => {
                Coefficient::Complex(
                    Complex::new(r1.to_rat(), d1.to_rat()) + Complex::new(r2.to_rat(), d2.to_rat()),
                )
            }
            (CoefficientView::FiniteField(n1, i1), CoefficientView::FiniteField(n2, i2)) => {
                if i1 != i2 {
                    error!(
                        "Cannot add numbers from different finite fields: p1={}, p2={}",
                        State::get_finite_field(i1).get_prime(),
                        State::get_finite_field(i2).get_prime()
                    );
                    return Coefficient::Indeterminate;
                }
                let f = State::get_finite_field(i1);
                Coefficient::FiniteField(f.add(&n1, &n2), i1)
            }
            (CoefficientView::FiniteField(_, _), _) => {
                error!("Cannot add finite field to non-finite number. Convert other number first?");
                Coefficient::Indeterminate
            }
            (_, CoefficientView::FiniteField(_, _)) => {
                error!("Cannot add finite field to non-finite number. Convert other number first?");
                Coefficient::Indeterminate
            }
            (CoefficientView::Natural(n, d, ni, _di), CoefficientView::RationalPolynomial(p))
            | (CoefficientView::RationalPolynomial(p), CoefficientView::Natural(n, d, ni, _di)) => {
                if ni != 0 {
                    error!("Cannot add complex number with polynomial");
                    return Coefficient::Indeterminate;
                }
                let r = p.deserialize();
                let r2 = RationalPolynomial {
                    numerator: r.numerator.constant(Integer::Single(n)),
                    denominator: r.denominator.constant(Integer::Single(d)),
                };
                Coefficient::RationalPolynomial(&r + &r2)
            }
            (CoefficientView::Large(l, d), CoefficientView::RationalPolynomial(p))
            | (CoefficientView::RationalPolynomial(p), CoefficientView::Large(l, d)) => {
                if !d.is_zero() {
                    error!("Cannot add complex number with polynomial");
                    return Coefficient::Indeterminate;
                }
                let r = p.deserialize();
                let l = l.to_rat();
                let r2 = RationalPolynomial {
                    numerator: r.numerator.constant(l.numerator()),
                    denominator: r.denominator.constant(l.denominator()),
                };
                Coefficient::RationalPolynomial(&r + &r2)
            }
            (CoefficientView::RationalPolynomial(p1), CoefficientView::RationalPolynomial(p2)) => {
                let mut p1 = p1.deserialize();
                let mut p2 = p2.deserialize();

                let r = if p1.get_variables() != p2.get_variables() {
                    p1.unify_variables(&mut p2);
                    &p1 + &p2
                } else {
                    &p1 + &p2
                };

                if r.is_constant() {
                    (r.numerator.lcoeff(), r.denominator.lcoeff()).into()
                } else {
                    Coefficient::RationalPolynomial(r)
                }
            }
            (CoefficientView::Natural(n, d, ni, di), CoefficientView::Float(rf, ri))
            | (CoefficientView::Float(rf, ri), CoefficientView::Natural(n, d, ni, di)) => {
                Coefficient::Float(Complex::new(
                    rf.to_float() + Rational::from((n, d)),
                    ri.to_float() + Rational::from((ni, di)),
                ))
            }
            (CoefficientView::Large(r, i), CoefficientView::Float(fr, fi))
            | (CoefficientView::Float(fr, fi), CoefficientView::Large(r, i)) => Coefficient::Float(
                Complex::new(fr.to_float() + r.to_rat(), fi.to_float() + i.to_rat()),
            ),
            (CoefficientView::Float(r1, i1), CoefficientView::Float(r2, i2)) => Coefficient::Float(
                Complex::new(r1.to_float() + r2.to_float(), i1.to_float() + i2.to_float()),
            ),
            (CoefficientView::Float(_, _), CoefficientView::RationalPolynomial(_)) => {
                error!("Cannot add float to rational polynomial");
                Coefficient::Indeterminate
            }
            (CoefficientView::RationalPolynomial(_), CoefficientView::Float(_, _)) => {
                error!("Cannot add float to rational polynomial");
                Coefficient::Indeterminate
            }
            (CoefficientView::Indeterminate, _) => Coefficient::Indeterminate,
            (_, CoefficientView::Indeterminate) => Coefficient::Indeterminate,
            (CoefficientView::Infinity(phase1), CoefficientView::Infinity(phase2)) => {
                match (phase1, phase2) {
                    (Some(p1), Some(p2)) if p1 == p2 => {
                        Coefficient::Infinity(Some(Complex::new(p1.0.to_rat(), p1.1.to_rat())))
                    }
                    _ => {
                        warn!(
                            "Created indeterminate by adding {} to {}",
                            self.to_owned(),
                            other.to_owned()
                        );
                        Coefficient::Indeterminate
                    }
                }
            }
            (CoefficientView::Infinity(a), _) | (_, CoefficientView::Infinity(a)) => {
                Coefficient::Infinity(a.map(|p| Complex::new(p.0.to_rat(), p.1.to_rat())))
            }
        }
    }
}

impl Mul for CoefficientView<'_> {
    type Output = Coefficient;

    fn mul(self, other: CoefficientView<'_>) -> Coefficient {
        match (self, other) {
            (
                CoefficientView::Natural(n1, d1, ni1, di1),
                CoefficientView::Natural(n2, d2, ni2, di2),
            ) => Coefficient::Complex(
                Complex::new(
                    Rational::from_int_unchecked(n1, d1),
                    Rational::from_int_unchecked(ni1, di1),
                ) * Complex::new(
                    Rational::from_int_unchecked(n2, d2),
                    Rational::from_int_unchecked(ni2, di2),
                ),
            ),
            (CoefficientView::Natural(n1, d1, ni1, di1), CoefficientView::Large(r2, d2))
            | (CoefficientView::Large(r2, d2), CoefficientView::Natural(n1, d1, ni1, di1)) => {
                Coefficient::Complex(
                    Complex::new(r2.to_rat(), d2.to_rat())
                        * Complex::new(
                            Rational::from_int_unchecked(n1, d1),
                            Rational::from_int_unchecked(ni1, di1),
                        ),
                )
            }
            (CoefficientView::Large(r1, d1), CoefficientView::Large(r2, d2)) => {
                Coefficient::Complex(
                    Complex::new(r1.to_rat(), d1.to_rat()) * Complex::new(r2.to_rat(), d2.to_rat()),
                )
            }
            (CoefficientView::FiniteField(n1, i1), CoefficientView::FiniteField(n2, i2)) => {
                if i1 != i2 {
                    error!(
                        "Cannot multiply numbers from different finite fields: p1={}, p2={}",
                        State::get_finite_field(i1).get_prime(),
                        State::get_finite_field(i2).get_prime()
                    );
                    return Coefficient::Indeterminate;
                }
                let f = State::get_finite_field(i1);
                Coefficient::FiniteField(f.mul(&n1, &n2), i1)
            }
            (CoefficientView::FiniteField(_, _), _) => {
                error!(
                    "Cannot multiply finite field to non-finite number. Convert other number first?"
                );
                Coefficient::Indeterminate
            }
            (_, CoefficientView::FiniteField(_, _)) => {
                error!(
                    "Cannot multiply finite field to non-finite number. Convert other number first?"
                );
                Coefficient::Indeterminate
            }
            (CoefficientView::Natural(n, d, ni, _di), CoefficientView::RationalPolynomial(p))
            | (CoefficientView::RationalPolynomial(p), CoefficientView::Natural(n, d, ni, _di)) => {
                if ni != 0 {
                    error!("Cannot multiply complex number with polynomial");
                    return Coefficient::Indeterminate;
                }

                let mut r = p.deserialize();
                let (n, d) = (Integer::Single(n), Integer::Single(d));

                let gcd1 = Z.gcd(&n, &r.denominator.content());
                let gcd2 = Z.gcd(&d, &r.numerator.content());
                r.numerator = r.numerator.div_coeff(&gcd2).mul_coeff(n.div(&gcd1));
                r.denominator = r.denominator.div_coeff(&gcd1).mul_coeff(d.div(&gcd2));
                Coefficient::RationalPolynomial(r)
            }
            (CoefficientView::Large(l, d), CoefficientView::RationalPolynomial(p))
            | (CoefficientView::RationalPolynomial(p), CoefficientView::Large(l, d)) => {
                if !d.is_zero() {
                    error!("Cannot multiply complex number with polynomial");
                    return Coefficient::Indeterminate;
                }

                let mut r = p.deserialize();
                let l = l.to_rat();
                let (n, d) = (l.numerator_ref(), l.denominator_ref());

                let gcd1 = Z.gcd(n, &r.denominator.content());
                let gcd2 = Z.gcd(d, &r.numerator.content());
                r.numerator = r.numerator.div_coeff(&gcd2).mul_coeff(n.div(&gcd1));
                r.denominator = r.denominator.div_coeff(&gcd1).mul_coeff(d.div(&gcd2));
                Coefficient::RationalPolynomial(r)
            }
            (CoefficientView::RationalPolynomial(p1), CoefficientView::RationalPolynomial(p2)) => {
                let mut p1 = p1.deserialize();
                let mut p2 = p2.deserialize();
                let r = if p1.get_variables() != p2.get_variables() {
                    p1.unify_variables(&mut p2);
                    &p1 * &p2
                } else {
                    &p1 * &p2
                };

                if r.is_constant() {
                    (r.numerator.lcoeff(), r.denominator.lcoeff()).into()
                } else {
                    Coefficient::RationalPolynomial(r)
                }
            }
            (CoefficientView::Natural(n1, d1, ni1, di1), CoefficientView::Float(fr, fi))
            | (CoefficientView::Float(fr, fi), CoefficientView::Natural(n1, d1, ni1, di1)) => {
                let c2: Complex<Fraction<IntegerRing>> =
                    Complex::new((n1, d1).into(), (ni1, di1).into());
                let c1 = Complex::new(fr.to_float(), fi.to_float());
                Complex::new(
                    c1.re.clone() * c2.re.clone() - c1.im.clone() * c2.im.clone(),
                    c1.re * c2.im + c1.im * c2.re,
                )
                .into()
            }
            (CoefficientView::Large(r, d), CoefficientView::Float(fr, fi))
            | (CoefficientView::Float(fr, fi), CoefficientView::Large(r, d)) => {
                let c2 = Complex::new(r.to_rat(), d.to_rat());
                let c1 = Complex::new(fr.to_float(), fi.to_float());

                Complex::new(
                    c1.re.clone() * c2.re.clone() - c1.im.clone() * c2.im.clone(),
                    c1.re * c2.im + c1.im * c2.re,
                )
                .into()
            }
            (CoefficientView::Float(r1, i1), CoefficientView::Float(r2, i2)) => {
                let c1 = Complex::new(r1.to_float(), i1.to_float());
                let c2 = Complex::new(r2.to_float(), i2.to_float());
                (c1 * c2).into()
            }
            (CoefficientView::Float(_, _), CoefficientView::RationalPolynomial(_)) => {
                error!("Cannot multiply float to rational polynomial");
                Coefficient::Indeterminate
            }
            (CoefficientView::RationalPolynomial(_), CoefficientView::Float(_, _)) => {
                error!("Cannot multiply float to rational polynomial");
                Coefficient::Indeterminate
            }
            (CoefficientView::Indeterminate, _) => Coefficient::Indeterminate,
            (_, CoefficientView::Indeterminate) => Coefficient::Indeterminate,
            (CoefficientView::Infinity(phase1), CoefficientView::Infinity(phase2)) => {
                match (phase1, phase2) {
                    (Some(p1), Some(p2)) => {
                        let r = Complex::new(p1.0.to_rat(), p1.1.to_rat())
                            * Complex::new(p2.0.to_rat(), p2.1.to_rat());
                        Coefficient::normalize_infinity(r)
                    }
                    _ => Coefficient::Infinity(None),
                }
            }
            (CoefficientView::Infinity(a), CoefficientView::Natural(n1, d1, ni1, di1))
            | (CoefficientView::Natural(n1, d1, ni1, di1), CoefficientView::Infinity(a)) => {
                if n1 == 0 && ni1 == 0 {
                    warn!(
                        "Created indeterminate by raising {} to the power of {}",
                        self.to_owned(),
                        other.to_owned()
                    );
                    return Coefficient::Indeterminate;
                }
                match a {
                    Some(p) => {
                        let r = Complex::new(p.0.to_rat(), p.1.to_rat())
                            * Complex::new(Rational::new(n1, d1), Rational::new(ni1, di1));
                        Coefficient::normalize_infinity(r)
                    }
                    None => Coefficient::Infinity(None),
                }
            }
            (CoefficientView::Infinity(a), CoefficientView::Large(r, i))
            | (CoefficientView::Large(r, i), CoefficientView::Infinity(a)) => {
                if r.is_zero() && i.is_zero() {
                    warn!(
                        "Created indeterminate by raising {} to the power of {}",
                        self.to_owned(),
                        other.to_owned()
                    );
                    return Coefficient::Indeterminate;
                }
                match a {
                    Some(p) => {
                        let r = Complex::new(p.0.to_rat(), p.1.to_rat())
                            * Complex::new(r.to_rat(), i.to_rat());
                        Coefficient::normalize_infinity(r)
                    }
                    None => Coefficient::Infinity(None),
                }
            }
            (CoefficientView::Infinity(_), _) | (_, CoefficientView::Infinity(_)) => {
                error!(
                    "Multiplication of rational polynomial or finite field is not implemented yet"
                );
                Coefficient::Indeterminate
            }
        }
    }
}

impl Add<i64> for CoefficientView<'_> {
    type Output = Coefficient;

    fn add(self, other: i64) -> Coefficient {
        match self {
            CoefficientView::Natural(n1, d1, ni1, di1) => Coefficient::Complex(
                Complex::new(
                    Rational::from_int_unchecked(n1, d1),
                    Rational::from_int_unchecked(ni1, di1),
                ) + Rational::from(other),
            ),
            CoefficientView::Float(r, i) => {
                Coefficient::Float(Complex::new(r.to_float() + other, i.to_float()))
            }
            CoefficientView::Large(r1, d1) => {
                Coefficient::Complex(Complex::new(r1.to_rat(), d1.to_rat()) + Rational::from(other))
            }
            CoefficientView::FiniteField(n1, i1) => {
                let f = State::get_finite_field(i1);
                Coefficient::FiniteField(f.add(&n1, &f.element_from_coefficient(other.into())), i1)
            }
            CoefficientView::RationalPolynomial(p) => {
                let p = p.deserialize();
                let a = RationalPolynomial {
                    numerator: p.numerator.constant(Integer::Single(other)),
                    denominator: p.denominator.constant(Integer::Single(1)),
                };

                Coefficient::RationalPolynomial(&p + &a)
            }
            CoefficientView::Indeterminate => Coefficient::Indeterminate,
            CoefficientView::Infinity(a) => {
                Coefficient::Infinity(a.map(|(r, i)| Complex::new(r.to_rat(), i.to_rat())))
            }
        }
    }
}

macro_rules! try_from_atom_int {
    ($t:ty) => {
        impl TryFrom<Atom> for $t {
            type Error = &'static str;

            fn try_from(value: Atom) -> Result<Self, Self::Error> {
                value.as_view().try_into()
            }
        }

        impl TryFrom<&Atom> for $t {
            type Error = &'static str;

            fn try_from(value: &Atom) -> Result<Self, Self::Error> {
                value.as_view().try_into()
            }
        }

        impl<'a> TryFrom<AtomView<'a>> for $t {
            type Error = &'static str;

            fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
                <$t>::try_from(Integer::try_from(value)?)
            }
        }
    };
}

try_from_atom_int!(i8);
try_from_atom_int!(i16);
try_from_atom_int!(i32);
try_from_atom_int!(i64);
try_from_atom_int!(i128);
try_from_atom_int!(isize);
try_from_atom_int!(u8);
try_from_atom_int!(u16);
try_from_atom_int!(u32);
try_from_atom_int!(u64);
try_from_atom_int!(u128);
try_from_atom_int!(usize);

impl TryFrom<Atom> for Integer {
    type Error = &'static str;

    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl TryFrom<&Atom> for Integer {
    type Error = &'static str;

    fn try_from(value: &Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl<'a> TryFrom<AtomView<'a>> for Integer {
    type Error = &'static str;

    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        let r = Rational::try_from(value)?;
        if r.is_integer() {
            Ok(r.numerator())
        } else {
            Err("Cannot convert rational number to integer")
        }
    }
}

impl TryFrom<Atom> for Rational {
    type Error = &'static str;

    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl TryFrom<&Atom> for Rational {
    type Error = &'static str;

    fn try_from(value: &Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl<'a> TryFrom<AtomView<'a>> for Rational {
    type Error = &'static str;

    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        if let AtomView::Num(n) = value {
            match n.get_coeff_view() {
                CoefficientView::Natural(n, d, ni, _di) => {
                    if ni == 0 {
                        Ok(Rational::from_int_unchecked(n, d))
                    } else {
                        Err("Not a rational")
                    }
                }
                CoefficientView::Large(r, d) => {
                    if d.is_zero() {
                        Ok(r.to_rat())
                    } else {
                        Err("Not a rational")
                    }
                }
                _ => Err("Not a rational"),
            }
        } else {
            Err("Not a number")
        }
    }
}

impl TryFrom<Atom> for Float {
    type Error = &'static str;

    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl TryFrom<&Atom> for Float {
    type Error = &'static str;

    fn try_from(value: &Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl<'a> TryFrom<AtomView<'a>> for Float {
    type Error = &'static str;

    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        if let AtomView::Num(n) = value {
            match n.get_coeff_view() {
                CoefficientView::Float(r, i) => {
                    if i.is_zero() {
                        Ok(r.to_float())
                    } else {
                        Err("Cannot convert complex number to float")
                    }
                }
                _ => Err("Not a float"),
            }
        } else {
            Err("Not a number")
        }
    }
}

impl TryFrom<Atom> for Complex<Float> {
    type Error = &'static str;

    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl TryFrom<&Atom> for Complex<Float> {
    type Error = &'static str;

    fn try_from(value: &Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl<'a> TryFrom<AtomView<'a>> for Complex<Float> {
    type Error = &'static str;

    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        if let AtomView::Num(n) = value {
            match n.get_coeff_view() {
                CoefficientView::Float(f, i) => Ok(Complex::new(f.to_float(), i.to_float())),
                _ => Err("Not a float"),
            }
        } else {
            Err("Not a number")
        }
    }
}

impl TryFrom<Atom> for Complex<Rational> {
    type Error = &'static str;

    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl TryFrom<&Atom> for Complex<Rational> {
    type Error = &'static str;

    fn try_from(value: &Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl<'a> TryFrom<AtomView<'a>> for Complex<Rational> {
    type Error = &'static str;

    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        if let AtomView::Num(n) = value {
            match n.get_coeff_view() {
                CoefficientView::Natural(n, d, ni, di) => Ok(Complex::new(
                    Rational::from_int_unchecked(n, d),
                    Rational::from_int_unchecked(ni, di),
                )),
                CoefficientView::Large(r, i) => Ok(Complex::new(r.to_rat(), i.to_rat())),
                _ => Err("Not a rational"),
            }
        } else {
            Err("Not a number")
        }
    }
}

impl TryFrom<Atom> for Coefficient {
    type Error = &'static str;

    fn try_from(value: Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl TryFrom<&Atom> for Coefficient {
    type Error = &'static str;

    fn try_from(value: &Atom) -> Result<Self, Self::Error> {
        value.as_view().try_into()
    }
}

impl<'a> TryFrom<AtomView<'a>> for Coefficient {
    type Error = &'static str;

    fn try_from(value: AtomView<'a>) -> Result<Self, Self::Error> {
        if let AtomView::Num(n) = value {
            Ok(n.get_coeff_view().to_owned())
        } else {
            Err("Not a number")
        }
    }
}

impl AtomView<'_> {
    /// Set the coefficient ring to the multivariate rational polynomial with `vars` variables.
    pub(crate) fn set_coefficient_ring(&self, vars: &Arc<Vec<PolyVariable>>) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut out = ws.new_atom();
            self.set_coefficient_ring_with_ws_into(vars, ws, &mut out);
            out.into_inner()
        })
    }

    /// Set the coefficient ring to the multivariate rational polynomial with `vars` variables.
    pub(crate) fn set_coefficient_ring_with_ws_into(
        &self,
        vars: &Arc<Vec<PolyVariable>>,
        workspace: &Workspace,
        out: &mut Atom,
    ) -> bool {
        match self {
            AtomView::Num(n) => {
                if let CoefficientView::RationalPolynomial(r) = n.get_coeff_view() {
                    let r = r.deserialize();
                    let old_var_map = r.get_variables();
                    if old_var_map != vars {
                        if old_var_map.iter().all(|x| vars.contains(x)) {
                            let n = r.numerator.rearrange_with_growth(vars).unwrap();
                            let d = r.denominator.rearrange_with_growth(vars).unwrap();
                            let r = RationalPolynomial {
                                numerator: n,
                                denominator: d,
                            };
                            out.to_num(Coefficient::RationalPolynomial(r));
                            true
                        } else {
                            let mut n1 = workspace.new_atom();
                            r.numerator.to_expression_with_map(
                                workspace,
                                &HashMap::default(),
                                &mut n1,
                            );

                            let mut n1_conv = workspace.new_atom();
                            n1.as_view().set_coefficient_ring_with_ws_into(
                                vars,
                                workspace,
                                &mut n1_conv,
                            );

                            let mut n2 = workspace.new_atom();
                            r.denominator.to_expression_with_map(
                                workspace,
                                &HashMap::default(),
                                &mut n2,
                            );

                            let mut n2_conv = workspace.new_atom();
                            n2.as_view().set_coefficient_ring_with_ws_into(
                                vars,
                                workspace,
                                &mut n2_conv,
                            );

                            // create n1/n2
                            let mut n3 = workspace.new_atom();
                            let mut exp = workspace.new_atom();
                            exp.to_num(Coefficient::Complex(Rational::from(-1i64).into()));
                            n3.to_pow(n2_conv.as_view(), exp.as_view());

                            let mut m = workspace.new_atom();
                            let mm = m.to_mul();
                            mm.extend(n1_conv.as_view());
                            mm.extend(n3.as_view());
                            m.as_view().normalize(workspace, out);
                            true
                        }
                    } else {
                        out.set_from_view(self);
                        false
                    }
                } else {
                    out.set_from_view(self);
                    false
                }
            }
            AtomView::Var(v) => {
                let id = v.get_symbol();
                if vars.contains(&id.into()) {
                    // change variable into coefficient
                    let mut poly = MultivariatePolynomial::new(&Z, None, vars.clone());
                    let mut e: SmallVec<[u16; INLINED_EXPONENTS]> = smallvec![0; vars.len()];
                    e[vars.iter().position(|x| *x == id).unwrap()] = 1;
                    poly.append_monomial(Integer::one(), &e);
                    let den = poly.one();

                    out.to_num(Coefficient::RationalPolynomial(RationalPolynomial {
                        numerator: poly,
                        denominator: den,
                    }));
                    true
                } else {
                    out.set_from_view(self);
                    false
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut nb = workspace.new_atom();
                if base.set_coefficient_ring_with_ws_into(vars, workspace, &mut nb) {
                    let mut o = workspace.new_atom();
                    o.to_pow(nb.as_view(), exp);

                    o.as_view().normalize(workspace, out);
                    true
                } else {
                    out.set_from_view(self);
                    false
                }
            }
            AtomView::Mul(m) => {
                let mut o = workspace.new_atom();
                let mul = o.to_mul();

                let mut changed = false;

                let mut arg_o = workspace.new_atom();
                for arg in m {
                    changed |= arg.set_coefficient_ring_with_ws_into(vars, workspace, &mut arg_o);
                    mul.extend(arg_o.as_view());
                }

                mul.set_normalized(!changed);

                if !changed {
                    mul.set_has_coefficient(m.has_coefficient());
                    std::mem::swap(out, &mut o);
                    false
                } else {
                    o.as_view().normalize(workspace, out);
                    true
                }
            }
            AtomView::Add(a) => {
                let mut o = workspace.new_atom();
                let mul = o.to_add();

                let mut changed = false;

                let mut arg_o = workspace.new_atom();
                for arg in a {
                    changed |= arg.set_coefficient_ring_with_ws_into(vars, workspace, &mut arg_o);
                    mul.extend(arg_o.as_view());
                }

                mul.set_normalized(!changed);

                if !changed {
                    std::mem::swap(out, &mut o);
                    false
                } else {
                    o.as_view().normalize(workspace, out);
                    true
                }
            }
            AtomView::Fun(_) => {
                // do not propagate into functions
                out.set_from_view(self);
                false
            }
        }
    }

    /// Convert all coefficients and built-in functions to floats with a given precision `decimal_prec`.
    /// The precision of floating point coefficients in the input will be truncated to `decimal_prec`.
    pub(crate) fn to_float_into(&self, decimal_prec: u32, out: &mut Atom) {
        let binary_prec = (decimal_prec as f64 * LOG2_10).ceil() as u32;

        Workspace::get_local().with(|ws| self.to_float_impl(binary_prec, true, false, ws, out))
    }

    fn to_float_impl(
        &self,
        binary_prec: u32,
        enter_function: bool,
        enter_exponent: bool,
        ws: &Workspace,
        out: &mut Atom,
    ) {
        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, d, ni, di) => {
                    out.to_num(Coefficient::Float(Complex::new(
                        Float::with_val(binary_prec, n) / Float::with_val(binary_prec, d),
                        Float::with_val(binary_prec, ni) / Float::with_val(binary_prec, di),
                    )));
                }
                CoefficientView::Float(r, i) => {
                    let mut f = r.to_float();
                    let mut g = i.to_float();
                    if f.prec() > binary_prec || g.prec() > binary_prec {
                        f.set_prec(binary_prec);
                        g.set_prec(binary_prec);
                        out.to_num(Coefficient::Float(Complex::new(f, g)));
                    } else {
                        out.set_from_view(self);
                    }
                }
                CoefficientView::Large(r, d) => {
                    out.to_num(Coefficient::Float(Complex::new(
                        r.to_rat().to_multi_prec_float(binary_prec),
                        d.to_rat().to_multi_prec_float(binary_prec),
                    )));
                }
                CoefficientView::FiniteField(_, _) => {
                    error!("Cannot convert finite field to float");
                    out.to_num(Coefficient::Indeterminate);
                }
                CoefficientView::RationalPolynomial(_) => {
                    error!("Cannot convert rational polynomial to float");
                    out.to_num(Coefficient::Indeterminate);
                }
                CoefficientView::Infinity(None) => {
                    error!("Cannot convert complex infinity to float");
                    out.to_num(Coefficient::Indeterminate);
                }
                CoefficientView::Indeterminate => {
                    error!("Cannot convert indeterminate to float");
                    out.to_num(Coefficient::Indeterminate);
                }
                CoefficientView::Infinity(Some((r, i))) => {
                    if i.is_zero() {
                        if r.is_negative() {
                            out.to_num(
                                Float::with_val(binary_prec, rug::float::Special::Infinity).into(),
                            );
                        } else {
                            out.to_num(
                                Float::with_val(binary_prec, rug::float::Special::NegInfinity)
                                    .into(),
                            );
                        }
                    } else {
                        error!("Cannot convert complex infinity to float");
                        out.to_num(Coefficient::Indeterminate);
                    }
                }
            },
            AtomView::Var(v) => {
                let s = v.get_symbol();

                match s {
                    Symbol::PI => {
                        out.to_num(Coefficient::Float(
                            Float::with_val(binary_prec, rug::float::Constant::Pi).into(),
                        ));
                    }
                    Symbol::E => {
                        out.to_num(Coefficient::Float(
                            Float::with_val(binary_prec, 1).exp().into(),
                        ));
                    }
                    _ => {
                        out.set_from_view(self);
                    }
                }
            }
            AtomView::Fun(f) => {
                if enter_function {
                    let mut o = ws.new_atom();
                    let ff = o.to_fun(f.get_symbol());

                    let mut na = ws.new_atom();
                    for a in f.iter() {
                        a.to_float_impl(binary_prec, enter_function, enter_exponent, ws, &mut na);
                        ff.add_arg(na.as_view());
                    }

                    o.as_view().normalize(ws, out);
                } else {
                    out.set_from_view(self);
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut nb = ws.new_atom();
                base.to_float_impl(binary_prec, enter_function, enter_exponent, ws, &mut nb);

                let mut o = ws.new_atom();

                if enter_exponent {
                    let mut ne = ws.new_atom();
                    exp.to_float_impl(binary_prec, enter_function, enter_exponent, ws, &mut ne);
                    o.to_pow(nb.as_view(), ne.as_view());
                } else {
                    o.to_pow(nb.as_view(), exp);
                }

                o.as_view().normalize(ws, out);
            }
            AtomView::Mul(m) => {
                let mut o = ws.new_atom();
                let mm = o.to_mul();

                let mut na = ws.new_atom();
                for a in m.iter() {
                    a.to_float_impl(binary_prec, enter_function, enter_exponent, ws, &mut na);
                    mm.extend(na.as_view());
                }

                o.as_view().normalize(ws, out);
            }
            AtomView::Add(a) => {
                let mut o = ws.new_atom();
                let aa = o.to_add();

                let mut na = ws.new_atom();
                for a in a.iter() {
                    a.to_float_impl(binary_prec, enter_function, enter_exponent, ws, &mut na);
                    aa.extend(na.as_view());
                }

                o.as_view().normalize(ws, out);
            }
        }
    }

    /// Map all floating point and rational coefficients to the best rational approximation
    /// in the interval `[self*(1-relative_error),self*(1+relative_error)]`.
    pub(crate) fn rationalize_coefficients(&self, relative_error: &Rational) -> Atom {
        let mut a = Atom::new();
        Workspace::get_local().with(|ws| {
            self.map_coefficient_impl(
                |c| match c {
                    CoefficientView::Float(r, i) => Coefficient::Complex(Complex::new(
                        r.to_float().to_rational().round(relative_error),
                        i.to_float().to_rational().round(relative_error),
                    )),
                    CoefficientView::Natural(n, d, ni, di) => {
                        let r = Rational::from_int_unchecked(n, d);
                        let real = r.round(relative_error);

                        let r = Rational::from_int_unchecked(ni, di);
                        let imag = r.round(relative_error);

                        Coefficient::Complex(Complex::new(real, imag))
                    }
                    CoefficientView::Large(r, d) => Coefficient::Complex(Complex::new(
                        r.to_rat().round(relative_error),
                        d.to_rat().round(relative_error),
                    )),
                    _ => c.to_owned(),
                },
                true,
                false,
                ws,
                &mut a,
            )
        });
        a
    }

    /// Map all coefficients using a given function.
    pub(crate) fn map_coefficient<F: Fn(CoefficientView) -> Coefficient + Copy>(
        &self,
        f: F,
    ) -> Atom {
        let mut a = Atom::new();
        self.map_coefficient_into(f, &mut a);
        a
    }

    /// Map all coefficients using a given function.
    pub(crate) fn map_coefficient_into<F: Fn(CoefficientView) -> Coefficient + Copy>(
        &self,
        f: F,
        out: &mut Atom,
    ) {
        Workspace::get_local().with(|ws| self.map_coefficient_impl(f, true, true, ws, out))
    }

    fn map_coefficient_impl<F: Fn(CoefficientView) -> Coefficient + Copy>(
        &self,
        coeff_map: F,
        enter_function: bool,
        enter_exponent: bool,
        ws: &Workspace,
        out: &mut Atom,
    ) {
        match self {
            AtomView::Num(n) => {
                out.to_num(coeff_map(n.get_coeff_view()));
            }
            AtomView::Var(_) => out.set_from_view(self),
            AtomView::Fun(f) => {
                if enter_function {
                    let mut o = ws.new_atom();
                    let ff = o.to_fun(f.get_symbol());

                    let mut na = ws.new_atom();
                    for a in f.iter() {
                        a.map_coefficient_impl(
                            coeff_map,
                            enter_function,
                            enter_exponent,
                            ws,
                            &mut na,
                        );
                        ff.add_arg(na.as_view());
                    }

                    o.as_view().normalize(ws, out);
                } else {
                    out.set_from_view(self);
                }
            }
            AtomView::Pow(p) => {
                let (base, exp) = p.get_base_exp();

                let mut nb = ws.new_atom();
                base.map_coefficient_impl(coeff_map, enter_function, enter_exponent, ws, &mut nb);

                let mut o = ws.new_atom();

                if enter_exponent {
                    let mut ne = ws.new_atom();
                    exp.map_coefficient_impl(
                        coeff_map,
                        enter_function,
                        enter_exponent,
                        ws,
                        &mut ne,
                    );
                    o.to_pow(nb.as_view(), ne.as_view());
                } else {
                    o.to_pow(nb.as_view(), exp);
                }

                o.as_view().normalize(ws, out);
            }
            AtomView::Mul(m) => {
                let mut o = ws.new_atom();
                let mm = o.to_mul();

                let mut na = ws.new_atom();
                for a in m.iter() {
                    a.map_coefficient_impl(coeff_map, enter_function, enter_exponent, ws, &mut na);
                    mm.extend(na.as_view());
                }

                o.as_view().normalize(ws, out);
            }
            AtomView::Add(a) => {
                let mut o = ws.new_atom();
                let aa = o.to_add();

                let mut na = ws.new_atom();
                for a in a.iter() {
                    a.map_coefficient_impl(coeff_map, enter_function, enter_exponent, ws, &mut na);
                    aa.extend(na.as_view());
                }

                o.as_view().normalize(ws, out);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::{
        atom::{Atom, AtomCore},
        domains::float::Float,
        parse,
        printer::{AtomPrinter, PrintOptions},
        symbol,
    };

    use super::Coefficient;

    #[test]
    fn rat_pow() {
        assert_eq!(parse!("(-2)^(-1/2)"), parse!("-1i*(1/2)^(1/2)"));
        assert_eq!(parse!("(-2)^(1/2)"), parse!("1i*2^(1/2)"));
        assert_eq!(parse!("(-2)^(-5/3)"), parse!("-1/2*(-2)^(-2/3)"));
        assert_eq!(parse!("(-2)^(-5/2)"), parse!("-1i/4*(1/2)^(1/2)"));
        assert_eq!(parse!("(2)^(-5/2)"), parse!("(1/32)^(1/2)"));
    }

    #[test]
    fn coeff_conversion() {
        let expr = parse!("v1*coeff(v2+v3/v4)+v1*coeff(v2)");
        let res = parse!("coeff((v3+2*v2*v4)/v4)*v1");
        assert_eq!(expr - &res, Atom::new());
    }

    #[test]
    fn coeff_conversion_large() {
        let expr = parse!(
            "coeff(-8123781237821378123128937128937211238971238*v2-1289378192371289372891378127893)"
        );
        let res = parse!(
            "1289378192371289372891378127893+8123781237821378123128937128937211238971238*coeff(v2)"
        );
        assert_eq!(expr + &res, Atom::new());
    }

    #[test]
    fn coefficient_ring() {
        let expr = parse!("v1*v3+v1*(v2+2)^-1*(v2+v3+1)");

        let v2 = symbol!("v2");
        let expr_yz = expr.set_coefficient_ring(&Arc::new(vec![v2.into(), symbol!("v3").into()]));

        let a = ((&expr_yz + &Atom::num((1, 2))) * &Atom::num((3, 4))).expand();

        let a = (a / &Atom::num((3, 4)) - &Atom::num((1, 2))).expand();

        let a = a.set_coefficient_ring(&Arc::new(vec![]));

        let expr = expr.replace(v2).with(Atom::num(3)).expand();

        let a = a.replace(v2).with(Atom::num(3)).expand();

        assert_eq!(a, expr);
    }

    #[test]
    fn float() {
        let expr = parse!("1/2 x + 5.8912734891723 + sin(1.2334)");
        let c = Coefficient::Float(Float::with_val(200, rug::float::Constant::Pi).into());
        let expr = expr * &Atom::num(c);
        let r = format!(
            "{}",
            AtomPrinter::new_with_options(
                expr.expand().as_view(),
                PrintOptions::file_no_namespace()
            )
        );
        assert_eq!(
            r,
            "1.57079632679489661923132169163975144209858469968755291048747*x+21.4724504210349"
        );
    }

    #[test]
    fn float_convert() {
        let expr = parse!("1/2 x + 238947/128903718927 + sin(3/4)");
        let expr = expr.to_float(60);
        let r = format!(
            "{}",
            AtomPrinter::new_with_options(expr.as_view(), PrintOptions::file_no_namespace())
        );
        assert_eq!(
            r,
            "5.00000000000000000000000000000000000000000000000000000000000e-1*x+6.81640613709185816359170669566511987261485697756222332885129e-1"
        );
    }

    #[test]
    fn float_to_rat() {
        let expr = parse!("1/2 x + 238947/128903718927 + sin(3/4)");
        let expr = expr.to_float(60);
        let expr = expr.rationalize(&(1, 10000).into());
        assert_eq!(expr, parse!("1/2*x+137/201"));
    }

    #[test]
    fn infinity() {
        assert_eq!(parse!("x^0"), 1);
        assert_eq!(parse!("0^0"), Coefficient::Indeterminate);
        assert_eq!(parse!("∞^0"), Coefficient::Indeterminate);
        assert_eq!(parse!("1^3"), 1);
        assert_eq!(parse!("1^∞"), Coefficient::Indeterminate);
        assert_eq!(parse!("(1/2)^∞"), 0);
        assert_eq!(parse!("(1/2 + 1i/2)^∞"), 0);
        assert_eq!(parse!("2^(-∞)"), 0);
        assert_eq!(parse!("(1/2)^((2+1i)∞)"), Coefficient::complex_infinity());
        assert_eq!(parse!("(-1)^∞"), Coefficient::Indeterminate);
        assert_eq!(parse!("∞ + ∞"), Coefficient::positive_infinity());
        assert_eq!(parse!("∞ - ∞"), Coefficient::Indeterminate);
        assert_eq!(parse!("∞/∞"), Coefficient::Indeterminate);
        assert_eq!(parse!("0 * ∞"), Coefficient::Indeterminate);
        assert_eq!(parse!("0/0"), Coefficient::Indeterminate);
        assert_eq!(parse!("∞^-1"), 0);
        assert_eq!(parse!("0/∞"), 0);
        assert_eq!(parse!("∞/0"), Coefficient::complex_infinity());
        assert_eq!(parse!("1/0"), Coefficient::complex_infinity());
        assert_eq!(parse!("⧞^0"), Coefficient::Indeterminate);
        assert_eq!(parse!("¿^0"), Coefficient::Indeterminate);
        assert_eq!(parse!("¿ * x + 5"), Coefficient::Indeterminate);
        assert_eq!(parse!("log(0)"), -Coefficient::positive_infinity());
        assert_eq!(parse!("(-2+1i)∞ (3+2i)∞"), parse!("(-1-1i/8)∞"));
    }
}
