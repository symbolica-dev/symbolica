//! The field of general expressions.

use crate::{
    atom::{Atom, AtomCore, AtomView},
    domains::{RingOps, Set},
    poly::PolyVariable,
};

use super::{
    Derivable, EuclideanDomain, Field, InternalOrdering, Ring, SelfRing, integer::Integer,
};

use dyn_clone::DynClone;
use rand::Rng;

pub trait Map: Fn(AtomView, &mut Atom) -> bool + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(Map);
impl<T: Clone + Send + Sync + Fn(AtomView<'_>, &mut Atom) -> bool> Map for T {}

/// The field of general expressions.
///
/// # Examples
///
/// ```
/// use symbolica::{
/// atom::Atom,
/// domains::{atom::AtomField, Field},
/// parse,
/// };
///
/// let field = AtomField {
///     statistical_zero_test: false,
///     cancel_check_on_division: true,
///     custom_normalization: None,
/// };
///
/// let r = field.div(
///     &parse!("x^2+2x+1"),
///     &parse!("x+1"),
/// );
/// assert_eq!(r, parse!("x+1"));
/// ```
#[derive(Clone)]
pub struct AtomField {
    /// Use a statistical test to check for zero, instead of only checking for an exact zero.
    pub statistical_zero_test: bool,
    /// Perform a cancellation check of numerators and denominators after a division.
    pub cancel_check_on_division: bool,
    /// A custom normalization function applied after every operation.
    pub custom_normalization: Option<Box<dyn Map>>,
}

impl PartialEq for AtomField {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for AtomField {}

impl std::hash::Hash for AtomField {
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {}
}

impl Default for AtomField {
    fn default() -> Self {
        AtomField::new()
    }
}

impl AtomField {
    pub fn new() -> AtomField {
        AtomField {
            statistical_zero_test: true,
            custom_normalization: None,
            cancel_check_on_division: false,
        }
    }

    #[inline(always)]
    fn normalize(&self, r: Atom) -> Atom {
        if let Some(f) = &self.custom_normalization {
            let mut res = Atom::new();
            if f(r.as_view(), &mut res) { res } else { r }
        } else {
            r
        }
    }

    #[inline(always)]
    fn normalize_mut(&self, r: &mut Atom) {
        if let Some(f) = &self.custom_normalization {
            let mut res = Atom::new();
            if f(r.as_view(), &mut res) {
                std::mem::swap(r, &mut res);
            }
        }
    }
}

impl std::fmt::Display for AtomField {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl std::fmt::Debug for AtomField {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl InternalOrdering for Atom {
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.cmp(other)
    }
}

impl Set for AtomField {
    type Element = Atom;

    fn size(&self) -> Option<Integer> {
        None
    }
}

impl RingOps<Atom> for AtomField {
    fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        self.normalize(a + b)
    }

    fn sub(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        self.normalize(a - b)
    }

    fn mul(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        self.normalize(a * b)
    }

    fn add_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = &*a + b;
        self.normalize_mut(a);
    }

    fn sub_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = &*a - b;
        self.normalize_mut(a);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = self.mul(&*a, &b);
        self.normalize_mut(a);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        *a = &*a + self.mul(b, c);
        self.normalize_mut(a);
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        *a = &*a - self.mul(b, c);
        self.normalize_mut(a);
    }

    fn neg(&self, a: Self::Element) -> Self::Element {
        self.normalize(-a)
    }
}

impl RingOps<&Atom> for AtomField {
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.normalize(a + b)
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.normalize(a - b)
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.normalize(a * b)
    }

    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = &*a + b;
        self.normalize_mut(a);
    }

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = &*a - b;
        self.normalize_mut(a);
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.mul(&*a, b);
        self.normalize_mut(a);
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = &*a + self.mul(b, c);
        self.normalize_mut(a);
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = &*a - self.mul(b, c);
        self.normalize_mut(a);
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        self.normalize(-a)
    }
}

impl Ring for AtomField {
    fn zero(&self) -> Self::Element {
        Atom::num(0)
    }

    fn one(&self) -> Self::Element {
        Atom::num(1)
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        self.normalize(b.pow(e))
    }

    /// Check if the result could be 0 using a statistical method.
    fn is_zero(&self, a: &Self::Element) -> bool {
        if self.statistical_zero_test {
            !a.as_view().zero_test(10, f64::EPSILON).is_false()
        } else {
            a.is_zero()
        }
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        if let AtomView::Num(n) = a.as_view() {
            n.is_one()
        } else {
            false
        }
    }

    fn one_is_gcd_unit() -> bool {
        true
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if SelfRing::is_zero(a) {
            None
        } else {
            Some(self.inv(a))
        }
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if SelfRing::is_zero(b) {
            None
        } else {
            Some(self.div(a, b))
        }
    }

    fn sample(&self, rng: &mut impl rand::RngCore, range: (i64, i64)) -> Self::Element {
        let r = rng.random_range(range.0..range.1);
        Atom::num(r)
    }

    fn nth(&self, n: Integer) -> Self::Element {
        Atom::num(n)
    }

    fn characteristic(&self) -> Integer {
        0.into()
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &crate::printer::PrintOptions,
        state: crate::printer::PrintState,
        f: &mut W,
    ) -> Result<bool, std::fmt::Error> {
        element.as_view().format(f, opts, state)
    }
}

impl SelfRing for Atom {
    fn is_zero(&self) -> bool {
        !self.as_view().zero_test(10, f64::EPSILON).is_false()
    }

    fn is_one(&self) -> bool {
        self.is_one()
    }

    fn format<W: std::fmt::Write>(
        &self,
        opts: &crate::printer::PrintOptions,
        state: crate::printer::PrintState,
        f: &mut W,
    ) -> Result<bool, std::fmt::Error> {
        self.as_view().format(f, opts, state)
    }
}

impl EuclideanDomain for AtomField {
    fn rem(&self, _a: &Self::Element, _b: &Self::Element) -> Self::Element {
        self.zero()
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.div(a, b), self.zero())
    }

    fn gcd(&self, _a: &Self::Element, _b: &Self::Element) -> Self::Element {
        // FIXME: return something else?
        self.one()
    }
}

impl Field for AtomField {
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        let r = a / b;

        self.normalize(if self.cancel_check_on_division {
            r.cancel()
        } else {
            r
        })
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.div(a, b);

        if self.cancel_check_on_division {
            *a = a.cancel();
        }

        self.normalize_mut(a);
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        let one = Atom::num(1);
        self.normalize(self.div(&one, a))
    }
}

impl Derivable for AtomField {
    type Variable = PolyVariable;

    fn derivative(&self, e: &Atom, x: &PolyVariable) -> Atom {
        match x {
            PolyVariable::Symbol(s) => e.derivative(*s),
            _ => panic!("Cannot take derivative of non-symbol"),
        }
    }
}
