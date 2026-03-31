//! Defines core algebraic traits and data structures.
//!
//! The core trait is [Ring], which has two binary operations, addition and multiplication.
//! Each ring has an associated element type, that should not be confused with the ring type itself.
//! For example:
//! - The ring of integers [Z](type@integer::Z) has elements of type [Integer].
//! - The ring of rational numbers [Q](type@rational::Q) has elements of type [Rational](rational::Rational).
//! - The ring of finite fields [FiniteField](finite_field::FiniteField) has elements of type [FiniteField](finite_field::FiniteFieldElement).
//! - The ring of polynomials [PolynomialRing](super::poly::polynomial::PolynomialRing) has elements of type [MultivariatePolynomial](super::poly::polynomial::MultivariatePolynomial).
//!
//! In general, the ring elements do not implement operations such as addition or multiplication,
//! but rather the ring itself does. Most Symbolica structures are generic over the ring type.
//!
//! An extension of the ring trait is the [`EuclideanDomain`] trait, which adds the ability to compute remainders, quotients, and gcds.
//! Another extension is the [`Field`] trait, which adds the ability to divide and invert elements.
pub mod dual;
pub mod finite_field;
pub mod float;
pub mod integer;
pub mod rational;

use std::borrow::Borrow;
use std::fmt::{Debug, Display, Error, Formatter};
use std::hash::Hash;
use std::ops::{Add, Deref, Div, Mul, Sub};

use integer::Integer;

use crate::printer::{PrintOptions, PrintState};

/// The internal ordering trait is used to compare elements of a ring.
/// This ordering is defined even for rings that do not have a total ordering, such
/// as complex numbers.
pub trait InternalOrdering {
    /// Compare two elements using an internal ordering.
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering;
}

macro_rules! impl_internal_ordering {
    ($($t:ty),*) => {
        $(
            impl InternalOrdering for $t {
                fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
                    self.cmp(other)
                }
            }
        )*
    };
}

impl_internal_ordering!(u8);
impl_internal_ordering!(u64);

macro_rules! impl_internal_ordering_range {
    ($($t:ty),*) => {
        $(
            impl<T: InternalOrdering> InternalOrdering for $t {
                fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
                    match self.len().cmp(&other.len()) {
                        std::cmp::Ordering::Equal => (),
                        ord => return ord,
                    }

                    for (i, j) in self.iter().zip(other) {
                        match i.internal_cmp(&j) {
                            std::cmp::Ordering::Equal => {}
                            ord => return ord,
                        }
                    }

                    std::cmp::Ordering::Equal
                }
            }
        )*
    };
}

impl_internal_ordering_range!([T]);
impl_internal_ordering_range!(Vec<T>);

/// Rings whose elements contain all the knowledge of the ring itself,
/// for example integers. A counterexample would be finite field elements,
/// as they do not store the prime.
pub trait SelfRing: Clone + PartialEq + Eq + Hash + InternalOrdering + Debug + Display {
    fn is_zero(&self) -> bool;
    fn is_one(&self) -> bool;
    fn format<W: std::fmt::Write>(
        &self,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error>;

    fn format_string(&self, opts: &PrintOptions, state: PrintState) -> String {
        let mut s = String::new();
        self.format(opts, state, &mut s)
            .expect("Could not write to string");
        s
    }
}

/// A set is a collection of elements.
pub trait Set: Clone + PartialEq + Eq + Hash + Debug + Display {
    /// The element of a set. For example, the elements of the ring of integers [Z](type@integer::Z), `Z::Element`, are [Integer].
    type Element: Clone + PartialEq + Eq + Hash + InternalOrdering + Debug;

    /// The number of elements in the set. `None` is used for infinite sets.
    fn size(&self) -> Option<Integer>;
}

/// Operations on rings. They should be implemented for `T = <Self as Set>::Element` and `T = &<Self as Set>::Element`.
pub trait RingOps<T>: Set {
    /// Compute `a + b`.
    fn add(&self, a: T, b: T) -> Self::Element;
    /// Compute `a - b`.
    fn sub(&self, a: T, b: T) -> Self::Element;
    /// Compute `a * b`.
    fn mul(&self, a: T, b: T) -> Self::Element;
    /// Compute `-a`.
    fn neg(&self, a: T) -> Self::Element;

    /// In-place addition: `a += b`.
    fn add_assign(&self, a: &mut Self::Element, b: T);
    /// In-place subtraction: `a -= b`.
    fn sub_assign(&self, a: &mut Self::Element, b: T);
    /// In-place multiplication: `a *= b`.
    fn mul_assign(&self, a: &mut Self::Element, b: T);
    /// In-place fused multiply-add: `a += b * c`.
    fn add_mul_assign(&self, a: &mut Self::Element, b: T, c: T);
    /// In-place fused multiply-subtract: `a -= b * c`.
    fn sub_mul_assign(&self, a: &mut Self::Element, b: T, c: T);
}

/// A ring is a set with two binary operations, addition and multiplication.
/// Examples of rings include the integers, rational numbers, and polynomials.
///
/// Each ring has an element type, that should not be confused with the ring type itself.
/// For example:
/// - The ring of integers [Z](type@integer::Z) has elements of type [Integer].
/// - The ring of rational numbers [Q](type@rational::Q) has elements of type [Rational](rational::Rational).
/// - The ring of finite fields [FiniteField](finite_field::FiniteField) has elements of type [FiniteField](finite_field::FiniteFieldElement).
/// - The ring of polynomials [PolynomialRing](super::poly::polynomial::PolynomialRing) has elements of type [MultivariatePolynomial](super::poly::polynomial::MultivariatePolynomial).
///
/// In general, the ring elements do not implement operations such as addition or multiplication,
/// but rather the ring itself does. Most Symbolica structures are generic over the ring type.
///
/// An extension of the ring trait is the [`EuclideanDomain`] trait, which adds the ability to compute remainders, quotients, and gcds.
/// Another extension is the [`Field`] trait, which adds the ability to divide and invert elements.
pub trait Ring:
    Set + RingOps<<Self as Set>::Element> + for<'a> RingOps<&'a <Self as Set>::Element>
{
    /// Return the additive identity `0`.
    fn zero(&self) -> Self::Element;
    /// Return the multiplicative identity `1`.
    fn one(&self) -> Self::Element;
    /// Return the nth element by computing `n * 1`.
    fn nth(&self, n: Integer) -> Self::Element;
    /// Return `b` raised to the power of `e`.
    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element;
    /// Return `true` iff `a` is the additive identity `0`.
    fn is_zero(&self, a: &Self::Element) -> bool;
    /// Return `true` iff `a` is the multiplicative identity `1`.
    fn is_one(&self, a: &Self::Element) -> bool;
    /// Should return `true` iff `gcd(1,x)` returns `1` for any `x`.
    /// For fraction fields, this is most often `false`, as `gcd(1,1/2)` is commonly
    /// defined to be `1/2`.
    fn one_is_gcd_unit() -> bool;
    /// The characteristic of the ring, i.e., the smallest positive integer `n` such that
    /// `n * 1 = 0`. If no such `n` exists, return `0`.
    fn characteristic(&self) -> Integer;

    /// Invert `a` if `a` is a unit in the ring. If is not, return `None`.
    /// For example, in [Z](type@integer::Z), only `1` and `-1` are invertible.
    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element>;

    /// Return the result of dividing `a` by `b`, if possible and if the result is unique.
    /// For example, in [Z](type@integer::Z), `4/2` is possible but `3/2` is not.
    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element>;

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element;

    /// Format a ring element with custom [PrintOptions] and [PrintState].
    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error>;

    /// Whether the ring does not contain additional information
    /// that cannot be inferred from any element of the ring.
    /// For example, [Z](type@integer::Z) and [Q](type@rational::Q) have independent elements,
    /// while [FiniteField](finite_field::FiniteField) does not, as the prime is part of the ring itself
    /// and not part of the elements.
    ///
    /// Types that return `true` can implement [SelfRing].
    fn has_independent_elements(&self) -> bool {
        false
    }

    /// Format the ring itself.
    fn format_ring<W: std::fmt::Write>(
        &self,
        _opts: &PrintOptions,
        _state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        f.write_fmt(format_args!("{}", self)).map(|_| false)
    }

    /// Create a new printer for the given ring element that
    /// can be used in a [format!] macro.
    fn printer<'a>(&'a self, element: &'a Self::Element) -> RingPrinter<'a, Self> {
        RingPrinter::new(self, element)
    }

    /// Wrap an element of the ring together with the ring itself, so that
    /// operators such as `+` and `*` can be used.
    ///
    /// ```
    /// use numerica::domains::{Ring, finite_field::{FiniteFieldCore, Zp}};
    /// let z = Zp::new(5);
    /// let w = z.wrap(z.to_element(3));
    /// let sub = w - &z.to_element(2);
    /// assert_eq!(*sub, z.one());
    /// ```
    fn wrap(&self, element: Self::Element) -> WrappedRingElement<Self, &Self>
    where
        Self: Sized,
    {
        WrappedRingElement::new(self, element)
    }
}

/// A Euclidean domain is a ring that supports division with remainder, quotients, and gcds.
pub trait EuclideanDomain: Ring {
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element);

    fn quot(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.quot_rem(a, b).0
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
}

/// A field is a ring that supports division and inversion.
pub trait Field: EuclideanDomain {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element);
    fn inv(&self, a: &Self::Element) -> Self::Element;

    /// Find the shortest linear recurrence relation for a given series `s`,
    /// using the Berlekamp-Massey algorithm.
    ///
    /// Yields a vector `c` such that `s[i] = sum(j, 0, m, c[j] * s[i-j-1])` for `i > m` and
    /// the number of stable iterations.
    ///
    /// # Example
    /// ```rust
    /// use numerica::domains::{Field, rational::Q};
    /// let (res, s) = Q.find_linear_recurrence_relation(&[0.into(), 1.into(), 1.into(), 3.into(),
    ///                                               5.into(), 11.into(), 21.into()]);
    /// assert_eq!(s, 3);
    /// assert_eq!(res, [1, 2]); // s[i] = 1 * s[i-1] + 2 * s[i-2]
    /// ```
    fn find_linear_recurrence_relation(
        &self,
        series: &[Self::Element],
    ) -> (Vec<Self::Element>, usize) {
        let mut c = vec![self.one()];
        let mut c_old = vec![self.one()];
        let mut tmp = vec![];
        let mut seq_len = 0;
        let mut m = 1;
        let mut stable_count = 0;
        let mut b_inv = self.one();
        for (n, s) in series.iter().enumerate() {
            let mut error = s.clone();
            for i in 1..=seq_len {
                self.add_mul_assign(&mut error, &c[i], &series[n - i]);
            }

            if self.is_zero(&error) {
                m += 1;
                stable_count += 1;
            } else if 2 * seq_len <= n {
                tmp.clone_from(&c);

                let factor = self.mul(&error, &b_inv);

                if c.len() < m + c_old.len() {
                    c.resize(m + c_old.len(), self.zero());
                }

                for (j, c_j) in c_old.iter().enumerate() {
                    self.sub_mul_assign(&mut c[j + m], c_j, &factor);
                }
                seq_len = n + 1 - seq_len;
                std::mem::swap(&mut c_old, &mut tmp);
                b_inv = self.inv(&error);
                m = 1;
                stable_count = 0;
            } else {
                let factor = self.mul(&error, &b_inv);

                if c.len() < m + c_old.len() {
                    c.resize(m + c_old.len(), self.zero());
                }

                for (j, c_j) in c_old.iter().enumerate() {
                    self.sub_mul_assign(&mut c[j + m], c_j, &factor);
                }
                m += 1;
                stable_count = 0;
            }
        }

        c.drain(0..c.len() - seq_len);
        for x in &mut c {
            *x = self.neg(&*x);
        }
        (c, stable_count)
    }
}

/// Rings that can be upgraded to fields, such as `IntegerRing` and `PolynomialRing`.
/// The most common upgrade is by creating a fraction field, such as `Q[x]`.
pub trait UpgradeToField: Ring {
    type Upgraded: Field;

    /// Upgrade the ring to a field.
    fn upgrade(self) -> Self::Upgraded;

    /// Upgrade an element of the ring to an element of the upgraded field.
    fn upgrade_element(&self, element: <Self as Set>::Element) -> <Self::Upgraded as Set>::Element;
}

impl<T: Field> UpgradeToField for T {
    type Upgraded = Self;

    fn upgrade(self) -> Self::Upgraded {
        self
    }

    fn upgrade_element(&self, element: <Self as Set>::Element) -> <Self::Upgraded as Set>::Element {
        element
    }
}

/// Provides an interface for printing elements of a ring with optional customization,
/// suitable as an argument to [format!]. Internally, it will call [Ring::format].
pub struct RingPrinter<'a, R: Ring> {
    pub ring: &'a R,
    pub element: &'a R::Element,
    pub opts: PrintOptions,
    pub state: PrintState,
}

impl<'a, R: Ring> RingPrinter<'a, R> {
    pub fn new(ring: &'a R, element: &'a R::Element) -> RingPrinter<'a, R> {
        RingPrinter {
            ring,
            element,
            opts: PrintOptions::default(),
            state: PrintState::default(),
        }
    }
}

impl<R: Ring> Display for RingPrinter<'_, R> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.ring
            .format(
                self.element,
                &self.opts.update_with_fmt(f),
                self.state.update_with_fmt(f),
                f,
            )
            .map(|_| ())
    }
}

/// A ring element wrapped together with its ring.
///
/// ```
/// # use numerica::domains::{Ring, finite_field::{FiniteFieldCore, Zp}};
/// let z = Zp::new(5);
/// let w = z.wrap(z.to_element(3));
/// let sub = w - &z.to_element(2);
/// assert_eq!(*sub, z.one());
/// ```
#[derive(Clone)]
pub struct WrappedRingElement<R: Ring, C: Clone + Borrow<R>> {
    pub ring: C,
    pub element: R::Element,
}

impl<R: Ring, C: Clone + Borrow<R>> AsRef<R::Element> for WrappedRingElement<R, C> {
    fn as_ref(&self) -> &R::Element {
        &self.element
    }
}

impl<R: Ring, C: Clone + Borrow<R>> Deref for WrappedRingElement<R, C> {
    type Target = R::Element;
    fn deref(&self) -> &R::Element {
        &self.element
    }
}

impl<R: Ring, C: Clone + Borrow<R>> WrappedRingElement<R, C> {
    pub fn new(ring: C, element: R::Element) -> Self {
        WrappedRingElement { ring, element }
    }

    pub fn ring(&self) -> &R {
        self.ring.borrow()
    }
}

impl<R: Ring, C: Clone + Borrow<R>> Debug for WrappedRingElement<R, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.ring()
            .format(
                &self.element,
                &PrintOptions::default(),
                PrintState::default(),
                f,
            )
            .map(|_| ())
    }
}

impl<R: Ring, C: Clone + Borrow<R>> Display for WrappedRingElement<R, C> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.ring()
            .format(
                &self.element,
                &PrintOptions::default(),
                PrintState::default(),
                f,
            )
            .map(|_| ())
    }
}

impl<R: Ring, C: Clone + Borrow<R>> PartialEq for WrappedRingElement<R, C> {
    fn eq(&self, other: &Self) -> bool {
        self.element == other.element
    }
}

impl<R: Ring, C: Clone + Borrow<R>> Eq for WrappedRingElement<R, C> {}

impl<R: Ring, C: Clone + Borrow<R>> Hash for WrappedRingElement<R, C> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.element.hash(state)
    }
}

impl<R: Ring, C: Clone + Borrow<R>> InternalOrdering for WrappedRingElement<R, C> {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.element.internal_cmp(&other.element)
    }
}

impl<R: Ring, C: Clone + Borrow<R>> SelfRing for WrappedRingElement<R, C> {
    fn is_zero(&self) -> bool {
        self.ring().is_zero(&self.element)
    }

    fn is_one(&self) -> bool {
        self.ring().is_one(&self.element)
    }

    fn format<W: std::fmt::Write>(
        &self,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, Error> {
        self.ring().format(&self.element, opts, state, f)
    }
}

impl<R: Ring, C: Clone + Borrow<R>> Add<&R::Element> for WrappedRingElement<R, C> {
    type Output = WrappedRingElement<R, C>;

    fn add(self, rhs: &R::Element) -> Self::Output {
        WrappedRingElement {
            element: self.ring().add(&self.element, rhs),
            ring: self.ring,
        }
    }
}

impl<R: Ring, C: Clone + Borrow<R>> Sub<&R::Element> for WrappedRingElement<R, C> {
    type Output = WrappedRingElement<R, C>;

    fn sub(self, rhs: &R::Element) -> Self::Output {
        WrappedRingElement {
            element: self.ring().sub(&self.element, rhs),
            ring: self.ring,
        }
    }
}

impl<R: Ring, C: Clone + Borrow<R>> Mul<&R::Element> for WrappedRingElement<R, C> {
    type Output = WrappedRingElement<R, C>;

    fn mul(self, rhs: &R::Element) -> Self::Output {
        WrappedRingElement {
            element: self.ring().mul(&self.element, rhs),
            ring: self.ring,
        }
    }
}

impl<R: Field, C: Clone + Borrow<R>> Div<&R::Element> for WrappedRingElement<R, C> {
    type Output = WrappedRingElement<R, C>;

    fn div(self, rhs: &R::Element) -> Self::Output {
        WrappedRingElement {
            element: self.ring().div(&self.element, rhs),
            ring: self.ring,
        }
    }
}

/// A ring that supports a derivative.
pub trait Derivable: Ring {
    type Variable;

    /// Take the derivative of `e` in `x`.
    fn derivative(&self, e: &<Self as Set>::Element, x: &Self::Variable) -> <Self as Set>::Element;
}

#[cfg(test)]
mod tests {
    use crate::domains::{Field, Ring, rational::Q};

    #[test]
    fn linear_recurrence() {
        let series = [
            2.into(),
            2.into(),
            1.into(),
            2.into(),
            1.into(),
            191.into(),
            393.into(),
            132.into(),
        ];

        let (res, it) = Q.find_linear_recurrence_relation(&series);
        assert_eq!(it, 0);

        for (i, x) in series.iter().enumerate().skip(res.len()) {
            let mut c = Q.zero();
            for j in 0..res.len() {
                c += &res[j] * &series[i - j - 1];
            }

            assert_eq!(&c, x);
        }
    }
}
