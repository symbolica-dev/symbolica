//! Multivariate polynomial structures and methods.

use ahash::{HashMap, HashMapExt};
use std::cell::{Cell, UnsafeCell};
use std::cmp::{Ordering, Reverse};
use std::collections::{BTreeMap, BinaryHeap};
use std::fmt::Display;
use std::marker::PhantomData;
use std::mem;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use crate::domains::algebraic_number::{AlgebraicExtension, AlgebraicNumber};
use crate::domains::float::{C, Complex, FloatLike};
use crate::domains::integer::{Integer, IntegerRing};
use crate::domains::rational::{Fraction, FractionField, FractionNormalization, Q, RationalField};
use crate::domains::{
    Derivable, EuclideanDomain, Field, InternalOrdering, Ring, RingOps, SelfRing, Set,
};
use crate::printer::{PrintOptions, PrintState};

use super::gcd::PolynomialGCD;
use super::univariate::UnivariatePolynomial;
use super::{Exponent, INLINED_EXPONENTS, LexOrder, MonomialOrder, PolyVariable, PositiveExponent};
use smallvec::{SmallVec, smallvec};

const MAX_DENSE_MUL_BUFFER_SIZE: usize = 1 << 24;
thread_local! { static DENSE_MUL_BUFFER: Cell<Vec<u32>> = const { Cell::new(Vec::new()) }; }

/// A ring for multivariate polynomials.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct PolynomialRing<R: Ring, E: Exponent> {
    pub(crate) ring: R,
    _phantom_exp: PhantomData<E>,
}

impl<R: Ring + FractionNormalization, E: Exponent> FractionNormalization for PolynomialRing<R, E> {
    fn get_normalization_factor(&self, a: &Self::Element) -> Self::Element {
        a.constant(a.ring.get_normalization_factor(&a.lcoeff()))
    }
}

impl<R: EuclideanDomain + FractionNormalization, E: Exponent> PolynomialRing<FractionField<R>, E> {
    pub fn to_rational_polynomial(
        &self,
        e: &<Self as Set>::Element,
    ) -> Fraction<PolynomialRing<R, E>> {
        let ring = self.ring.ring();
        let mut lcm = ring.one();
        for x in &e.coefficients {
            let g = ring.gcd(&lcm, x.denominator_ref());
            lcm = ring.mul(&lcm, &ring.quot_rem(x.denominator_ref(), &g).0);
        }

        let e2 = e.map_coeff(
            |c| {
                ring.mul(
                    c.numerator_ref(),
                    &ring.quot_rem(&lcm, c.denominator_ref()).0,
                )
            },
            ring.clone(),
        );

        Fraction::from_unchecked(e2.constant(lcm), e2)
    }
}

impl<R: Ring, E: Exponent> PolynomialRing<R, E> {
    pub fn new(coeff_ring: R) -> PolynomialRing<R, E> {
        PolynomialRing {
            ring: coeff_ring,
            _phantom_exp: PhantomData,
        }
    }

    pub fn from_poly(poly: &MultivariatePolynomial<R, E>) -> PolynomialRing<R, E> {
        PolynomialRing {
            ring: poly.ring.clone(),
            _phantom_exp: PhantomData,
        }
    }

    /// Get the coefficient ring.
    pub fn coefficient_ring(&self) -> &R {
        &self.ring
    }
}

impl<R: Ring, E: Exponent> std::fmt::Display for PolynomialRing<R, E> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl<R: Ring, E: Exponent> Set for PolynomialRing<R, E> {
    type Element = MultivariatePolynomial<R, E>;

    fn size(&self) -> Option<Integer> {
        None
    }
}

impl<R: Ring, E: Exponent> RingOps<MultivariatePolynomial<R, E>> for PolynomialRing<R, E> {
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
        a * &b
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = std::mem::replace(a, b.zero()) + b;
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = std::mem::replace(a, b.zero()) - b;
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = &*a * &b;
    }

    #[inline]
    fn add_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        *a = std::mem::replace(a, b.zero()) + b * &c
    }

    #[inline]
    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        *a = std::mem::replace(a, b.zero()) - b * &c
    }

    #[inline]
    fn neg(&self, a: Self::Element) -> Self::Element {
        a.neg()
    }
}

impl<R: Ring, E: Exponent> RingOps<&MultivariatePolynomial<R, E>> for PolynomialRing<R, E> {
    #[inline]
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a + b
    }

    #[inline]
    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a - b
    }

    #[inline]
    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
    }

    #[inline]
    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = &*a + b;
    }

    #[inline]
    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = &*a - b;
    }

    #[inline]
    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = std::mem::replace(a, b.zero()) * b;
    }

    #[inline]
    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = std::mem::replace(a, b.zero()) + b * c
    }

    #[inline]
    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = std::mem::replace(a, b.zero()) - b * c
    }

    #[inline]
    fn neg(&self, a: &Self::Element) -> Self::Element {
        a.clone().neg()
    }
}

impl<R: Ring, E: Exponent> Ring for PolynomialRing<R, E> {
    #[inline]
    fn zero(&self) -> Self::Element {
        MultivariatePolynomial::new(&self.ring, None, Arc::new(vec![]))
    }

    #[inline]
    fn one(&self) -> Self::Element {
        self.zero().one()
    }

    #[inline]
    fn nth(&self, n: Integer) -> Self::Element {
        self.zero().constant(self.ring.nth(n))
    }

    #[inline]
    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        b.pow(e as usize)
    }

    #[inline]
    fn is_zero(&self, a: &Self::Element) -> bool {
        a.is_zero()
    }

    #[inline]
    fn is_one(&self, a: &Self::Element) -> bool {
        a.is_one()
    }

    fn one_is_gcd_unit() -> bool {
        false
    }

    fn characteristic(&self) -> Integer {
        self.ring.characteristic()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if a.is_constant() {
            let inv = self.ring.try_inv(&a.get_constant())?;
            Some(a.constant(inv))
        } else {
            None
        }
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        a.try_div(b)
    }

    fn sample(&self, _rng: &mut impl rand::RngCore, _range: (i64, i64)) -> Self::Element {
        todo!("Sampling a polynomial is not possible yet")
    }

    fn format<W: std::fmt::Write>(
        &self,
        element: &Self::Element,
        opts: &PrintOptions,
        state: PrintState,
        f: &mut W,
    ) -> Result<bool, std::fmt::Error> {
        element.format(opts, state, f)
    }

    fn has_independent_elements(&self) -> bool {
        // the coefficient ring is stored in the polynomial
        true
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> EuclideanDomain
    for PolynomialRing<R, E>
{
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.rem(b)
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        a.quot_rem(b, false)
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.gcd(b)
    }
}

/// Multivariate polynomial with a sparse degree and dense variable representation.
/// Negative exponents are supported, if they are allowed by the exponent type.
#[derive(Clone)]
#[cfg_attr(feature = "bincode", derive(bincode_trait_derive::Encode))]
pub struct MultivariatePolynomial<F: Ring, E: Exponent = u16, O: MonomialOrder = LexOrder> {
    // Data format: the i-th monomial is stored as coefficients[i] and
    // exponents[i * nvars .. (i + 1) * nvars]. Terms are always expanded and sorted by the exponents via
    // cmp_exponents().
    pub coefficients: Vec<F::Element>,
    pub exponents: Vec<E>,
    /// The coefficient ring.
    pub ring: F,
    pub variables: Arc<Vec<PolyVariable>>,
    pub(crate) _phantom: PhantomData<O>,
}

#[cfg(feature = "bincode")]
impl<
    C: crate::state::HasStateMap,
    F: Ring + bincode::Decode<C>,
    E: Exponent + bincode::Decode<C>,
    O: MonomialOrder + bincode::Decode<C>,
> bincode::Decode<C> for MultivariatePolynomial<F, E, O>
where
    F::Element: for<'a> bincode::Decode<&'a F>,
{
    fn decode<D: bincode::de::Decoder<Context = C>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let ring = F::decode(decoder)?;

        let coefficients = Vec::<F::Element>::decode(&mut decoder.with_context(&ring))?;
        let exponents = Vec::<E>::decode(decoder)?;
        let variables = Arc::<Vec<PolyVariable>>::decode(decoder)?;
        Ok(MultivariatePolynomial {
            coefficients,
            exponents,
            ring,
            variables,
            _phantom: PhantomData,
        })
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> MultivariatePolynomial<F, E, O> {
    /// Constructs a zero polynomial. Instead of using this constructor,
    /// prefer to create new polynomials from existing ones, so that the
    /// variable map and field are inherited.
    #[inline]
    pub fn new(ring: &F, cap: Option<usize>, variables: Arc<Vec<PolyVariable>>) -> Self {
        Self {
            coefficients: Vec::with_capacity(cap.unwrap_or(0)),
            exponents: Vec::with_capacity(cap.unwrap_or(0) * variables.len()),
            ring: ring.clone(),
            variables,
            _phantom: PhantomData,
        }
    }

    /// Constructs a zero polynomial. Instead of using this constructor,
    /// prefer to create new polynomials from existing ones, so that the
    /// variable map is inherited.
    #[inline]
    pub fn new_zero(ring: &F) -> Self {
        Self {
            coefficients: vec![],
            exponents: vec![],
            ring: ring.clone(),
            variables: Arc::new(vec![]),
            _phantom: PhantomData,
        }
    }

    /// Constructs a polynomial that is one. Instead of using this constructor,
    /// prefer to create new polynomials from existing ones, so that the
    /// variable map is inherited.
    #[inline]
    pub fn new_one(ring: &F) -> Self {
        Self {
            coefficients: vec![ring.one()],
            exponents: vec![],
            ring: ring.clone(),
            variables: Arc::new(vec![]),
            _phantom: PhantomData,
        }
    }

    /// Constructs a zero polynomial, inheriting the field and variable map from `self`.
    #[inline]
    pub fn zero(&self) -> Self {
        Self {
            coefficients: vec![],
            exponents: vec![],
            ring: self.ring.clone(),
            variables: self.variables.clone(),
            _phantom: PhantomData,
        }
    }

    /// Constructs a zero polynomial with the given number of variables and capacity,
    /// inheriting the field and variable map from `self`.
    #[inline]
    pub fn zero_with_capacity(&self, cap: usize) -> Self {
        Self {
            coefficients: Vec::with_capacity(cap),
            exponents: Vec::with_capacity(cap * self.nvars()),
            ring: self.ring.clone(),
            variables: self.variables.clone(),
            _phantom: PhantomData,
        }
    }

    /// Constructs a constant polynomial,
    /// inheriting the field and variable map from `self`.
    #[inline]
    pub fn constant(&self, coeff: F::Element) -> Self {
        if self.ring.is_zero(&coeff) {
            return self.zero();
        }

        Self {
            coefficients: vec![coeff],
            exponents: vec![E::zero(); self.nvars()],
            ring: self.ring.clone(),
            variables: self.variables.clone(),
            _phantom: PhantomData,
        }
    }

    /// Constructs a polynomial that is one, inheriting the field and variable map from `self`.
    #[inline]
    pub fn one(&self) -> Self {
        Self {
            coefficients: vec![self.ring.one()],
            exponents: vec![E::zero(); self.nvars()],
            ring: self.ring.clone(),
            variables: self.variables.clone(),
            _phantom: PhantomData,
        }
    }

    /// Constructs a polynomial with a single term.
    #[inline]
    pub fn monomial(&self, coeff: F::Element, exponents: Vec<E>) -> Self {
        debug_assert!(self.nvars() == exponents.len());

        if self.ring.is_zero(&coeff) {
            return self.zero();
        }

        Self {
            coefficients: vec![coeff],
            exponents,
            ring: self.ring.clone(),
            variables: self.variables.clone(),
            _phantom: PhantomData,
        }
    }

    /// Constructs a polynomial with a single term that is a variable.
    #[inline]
    pub fn variable(&self, var: &PolyVariable) -> Result<Self, String> {
        if let Some(pos) = self.variables.iter().position(|v| v == var) {
            let mut exp = vec![E::zero(); self.nvars()];
            exp[pos] = E::one();
            Ok(self.monomial(self.ring.one(), exp))
        } else {
            Err(format!("Variable {} not found", var))
        }
    }

    /// Get the ith monomial
    pub fn to_monomial_view(&self, i: usize) -> MonomialView<'_, F, E> {
        assert!(i < self.nterms());

        MonomialView {
            coefficient: &self.coefficients[i],
            exponents: self.exponents(i),
        }
    }

    #[inline]
    pub fn reserve(&mut self, cap: usize) -> &mut Self {
        self.coefficients.reserve(cap);
        self.exponents.reserve(cap * self.nvars());
        self
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.nterms() == 0
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        self.nterms() == 1
            && self.ring.is_one(&self.coefficients[0])
            && self.exponents.iter().all(|x| x.is_zero())
    }

    /// Returns the number of terms in the polynomial.
    #[inline]
    pub fn nterms(&self) -> usize {
        self.coefficients.len()
    }

    /// Returns the number of variables in the polynomial.
    #[inline]
    pub fn nvars(&self) -> usize {
        self.variables.len()
    }

    /// Returns true if the polynomial is constant.
    #[inline]
    pub fn is_constant(&self) -> bool {
        if self.is_zero() {
            return true;
        }
        if self.nterms() >= 2 {
            return false;
        }
        debug_assert!(!self.ring.is_zero(self.coefficients.first().unwrap()));
        self.exponents.iter().all(|e| e.is_zero())
    }

    /// Get the constant term of the polynomial.
    #[inline]
    pub fn get_constant(&self) -> F::Element {
        if self.is_zero() || !self.exponents(0).iter().all(|e| e.is_zero()) {
            return self.ring.zero();
        }

        self.coefficients[0].clone()
    }

    /// Returns the `index`th monomial, starting from the back.
    #[inline]
    pub fn coefficient_back(&self, index: usize) -> &F::Element {
        &self.coefficients[self.nterms() - index - 1]
    }

    /// Returns the slice for the exponents of the specified monomial.
    #[inline]
    pub fn exponents(&self, index: usize) -> &[E] {
        //&self.exponents[index * self.nvars()..(index + 1) * self.nvars()]
        unsafe {
            self.exponents
                .get_unchecked(index * self.nvars()..(index + 1) * self.nvars())
        }
    }

    /// Returns the slice for the exponents of the specified monomial
    /// starting from the back.
    #[inline]
    pub fn exponents_back(&self, index: usize) -> &[E] {
        let index = self.nterms() - index - 1;
        &self.exponents[index * self.nvars()..(index + 1) * self.nvars()]
    }

    #[inline(always)]
    pub fn last_exponents(&self) -> &[E] {
        //assert!(self.nterms() > 0);
        &self.exponents[(self.nterms() - 1) * self.nvars()..self.nterms() * self.nvars()]
    }

    /// Returns the mutable slice for the exponents of the specified monomial.
    #[inline]
    pub fn exponents_mut(&mut self, index: usize) -> &mut [E] {
        let nvars = self.nvars();
        &mut self.exponents[index * nvars..(index + 1) * nvars]
    }

    /// Returns an iterator over the exponents of every monomial.
    #[inline]
    pub fn exponents_iter(&self) -> std::slice::Chunks<'_, E> {
        self.exponents.chunks(self.nvars())
    }

    /// Returns an iterator over the mutable exponents of every monomial.
    #[inline]
    pub fn exponents_iter_mut(&mut self) -> std::slice::ChunksMut<'_, E> {
        let nvars = self.nvars();
        self.exponents.chunks_mut(nvars)
    }

    /// Reset the polynomial to 0.
    #[inline]
    pub fn clear(&mut self) {
        self.coefficients.clear();
        self.exponents.clear();
    }

    /// Get a copy of the variable list.
    pub fn get_vars(&self) -> Arc<Vec<PolyVariable>> {
        self.variables.clone()
    }

    /// Get a reference to the variables list.
    pub fn get_vars_ref(&self) -> &[PolyVariable] {
        self.variables.as_ref()
    }

    /// Rename a variable.
    pub fn rename_variable(&mut self, old: &PolyVariable, new: &PolyVariable) {
        if let Some(pos) = self.variables.iter().position(|v| v == old) {
            let mut new_vars = self.variables.as_ref().clone();
            new_vars[pos] = new.clone();
            self.variables = Arc::new(new_vars);
        }
    }

    /// Unify the variable maps of two polynomials, i.e.
    /// rewrite a polynomial in `x` and one in `y` to a
    /// two polynomial in `x` and `y`.
    ///
    /// The variable map will be inherited from
    /// `self` and will be extended by variables occurring
    /// in `other`.
    #[inline(always)]
    pub fn unify_variables(&mut self, other: &mut Self) {
        if self.variables == other.variables {
            return;
        }

        self.unify_variables_impl(other)
    }

    fn unify_variables_impl(&mut self, other: &mut Self) {
        let mut new_var_map = self.variables.as_ref().clone();
        let mut new_var_pos_other = vec![0; other.nvars()];
        for (pos, v) in new_var_pos_other.iter_mut().zip(other.variables.as_ref()) {
            if let Some(p) = new_var_map.iter().position(|x| x == v) {
                *pos = p;
            } else {
                *pos = new_var_map.len();
                new_var_map.push(v.clone());
            }
        }

        let mut newexp = vec![E::zero(); new_var_map.len() * self.nterms()];

        for t in 0..self.nterms() {
            newexp[t * new_var_map.len()..t * new_var_map.len() + self.nvars()]
                .copy_from_slice(self.exponents(t));
        }

        self.variables = Arc::new(new_var_map);
        self.exponents = newexp;

        // check if term ordering remains unchanged
        if new_var_pos_other.windows(2).all(|w| w[0] <= w[1]) {
            let mut newexp = vec![E::zero(); self.nvars() * other.nterms()];

            if other.nvars() > 0 {
                for (d, t) in newexp
                    .chunks_mut(self.nvars())
                    .zip(other.exponents.chunks(other.nvars()))
                {
                    for (var, e) in t.iter().enumerate() {
                        d[new_var_pos_other[var]] = *e;
                    }
                }
            }

            other.variables = self.variables.clone();
            other.exponents = newexp;
            return;
        }

        // reconstruct 'other' with correct monomial ordering
        let mut newother = Self::new(&other.ring, other.nterms().into(), self.variables.clone());
        let mut newexp = vec![E::zero(); self.nvars()];
        for t in other.into_iter() {
            for c in &mut newexp {
                *c = E::zero();
            }

            for (var, e) in t.exponents.iter().enumerate() {
                newexp[new_var_pos_other[var]] = *e;
            }
            newother.append_monomial(t.coefficient.clone(), &newexp);
        }
        *other = newother;
    }

    /// Unify the variable maps of all polynomials in the slice.
    pub fn unify_variables_list(polys: &mut [Self]) {
        if polys.len() < 2 {
            return;
        }

        let (first, rest) = polys.split_first_mut().unwrap();
        for _ in 0..2 {
            for p in &mut *rest {
                first.unify_variables(p);
            }
        }
    }

    /// Reverse the coefficients: `1+2x+3x^3` becomes `3+2x^2+x^3`.
    pub fn reverse(&mut self) {
        self.coefficients.reverse();
        let nterms = self.nterms();
        let nvars = self.nvars();
        let degs = (0..nvars).map(|i| self.degree(i)).collect::<Vec<_>>();

        for e in self.exponents.chunks_mut(nvars) {
            for (ee, d) in e.iter_mut().zip(&degs) {
                *ee = *d - *ee;
            }
        }

        let midu = if nterms.is_multiple_of(2) {
            self.nvars() * (nterms / 2)
        } else {
            self.nvars() * (nterms / 2 + 1)
        };

        let (l, r) = self.exponents.split_at_mut(midu);

        let rend = r.len();
        for i in 0..nterms / 2 {
            l[i * nvars..(i + 1) * nvars]
                .swap_with_slice(&mut r[rend - (i + 1) * nvars..rend - i * nvars]);
        }
    }

    /// Reverse the monomial ordering in-place.
    fn reverse_monomials(&mut self) {
        let nterms = self.nterms();
        let nvars = self.nvars();
        if nterms < 2 {
            return;
        }

        self.coefficients.reverse();

        let midu = if nterms.is_multiple_of(2) {
            self.nvars() * (nterms / 2)
        } else {
            self.nvars() * (nterms / 2 + 1)
        };

        let (l, r) = self.exponents.split_at_mut(midu);

        let rend = r.len();
        for i in 0..nterms / 2 {
            l[i * nvars..(i + 1) * nvars]
                .swap_with_slice(&mut r[rend - (i + 1) * nvars..rend - i * nvars]);
        }
    }

    /// Add a variable to the polynomial if it is not already present.
    pub fn add_variable(&mut self, var: &PolyVariable) {
        if self.variables.iter().any(|v| v == var) {
            return;
        }

        let l = self.variables.len();

        let mut new_exp = vec![E::zero(); (l + 1) * self.nterms()];

        if l > 0 {
            for (en, e) in new_exp.chunks_mut(l + 1).zip(self.exponents.chunks(l)) {
                en[..l].copy_from_slice(e);
            }
        }

        let mut new_vars = self.variables.as_ref().clone();
        new_vars.push(var.clone());
        self.variables = Arc::new(new_vars);
        self.exponents = new_exp;
    }

    /// Check if the polynomial is sorted and has only non-zero coefficients
    pub fn check_consistency(&self) {
        assert_eq!(self.coefficients.len(), self.nterms());
        assert_eq!(self.exponents.len(), self.nterms() * self.nvars());

        for c in &self.coefficients {
            if self.ring.is_zero(c) {
                panic!("Inconsistent polynomial (0 coefficient): {self}");
            }
        }

        for t in 1..self.nterms() {
            match O::cmp(self.exponents(t), self.exponents(t - 1)) {
                Ordering::Equal => panic!("Inconsistent polynomial (equal monomials): {self}"),
                Ordering::Less => {
                    panic!("Inconsistent polynomial (wrong monomial ordering): {self}")
                }
                Ordering::Greater => {}
            }
        }
    }

    /// Append a monomial to the back. It merges with the last monomial if the
    /// exponents are equal.
    #[inline]
    pub fn append_monomial_back(&mut self, coefficient: F::Element, exponents: &[E]) {
        if self.ring.is_zero(&coefficient) {
            return;
        }

        let nterms = self.nterms();
        if nterms > 0 && exponents == self.last_exponents() {
            self.ring
                .add_assign(&mut self.coefficients[nterms - 1], &coefficient);

            if self.ring.is_zero(&self.coefficients[nterms - 1]) {
                self.coefficients.pop();
                self.exponents.truncate((nterms - 1) * self.nvars());
            }
        } else {
            self.coefficients.push(coefficient);
            self.exponents.extend_from_slice(exponents);
        }
    }

    /// Appends a monomial to the polynomial.
    pub fn append_monomial(&mut self, coefficient: F::Element, exponents: &[E]) {
        if self.ring.is_zero(&coefficient) {
            return;
        }
        if self.nvars() != exponents.len() {
            panic!(
                "nvars mismatched: got {}, expected {}",
                exponents.len(),
                self.nvars()
            );
        }

        // should we append to the back?
        if self.nterms() == 0 || O::cmp(self.last_exponents(), exponents).is_lt() {
            self.coefficients.push(coefficient);
            self.exponents.extend_from_slice(exponents);
            return;
        }

        if O::cmp(self.exponents(0), exponents).is_gt() {
            self.coefficients.insert(0, coefficient);
            self.exponents.splice(0..0, exponents.iter().cloned());
            return;
        }

        // Binary search to find the insert-point.
        let mut l = 0;
        let mut r = self.nterms();

        while l <= r {
            let m = (l + r) / 2;
            let c = O::cmp(exponents, self.exponents(m)); // note the reversal

            match c {
                Ordering::Equal => {
                    // Add the two coefficients.
                    self.ring
                        .add_assign(&mut self.coefficients[m], &coefficient);
                    if self.ring.is_zero(&self.coefficients[m]) {
                        // The coefficient becomes zero. Remove this monomial.
                        self.coefficients.remove(m);
                        let i = m * self.nvars();
                        self.exponents.splice(i..i + self.nvars(), Vec::new());
                    }
                    return;
                }
                Ordering::Greater => {
                    l = m + 1;

                    if l == self.nterms() {
                        self.coefficients.push(coefficient);
                        self.exponents.extend_from_slice(exponents);
                        return;
                    }
                }
                Ordering::Less => {
                    if m == 0 {
                        self.coefficients.insert(0, coefficient);
                        self.exponents.splice(0..0, exponents.iter().cloned());
                        return;
                    }

                    r = m - 1;
                }
            }
        }

        self.coefficients.insert(l, coefficient);
        let i = l * self.nvars();
        self.exponents.splice(i..i, exponents.iter().cloned());
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> SelfRing for MultivariatePolynomial<F, E, O> {
    #[inline]
    fn is_zero(&self) -> bool {
        self.is_zero()
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.is_one()
    }

    fn format<W: std::fmt::Write>(
        &self,
        opts: &PrintOptions,
        mut state: PrintState,
        f: &mut W,
    ) -> Result<bool, std::fmt::Error> {
        let print_ring = opts.print_ring && !self.ring.has_independent_elements();

        if self.is_constant() {
            if self.is_zero() {
                if state.in_sum {
                    f.write_str("+")?;
                }
                f.write_char('0')?;
                return Ok(false);
            } else if !print_ring || state.level > 0 {
                return self.ring.format(&self.coefficients[0], opts, state, f);
            }
        }

        let add_paren = (self.nterms() > 1 || print_ring) && state.in_product
            || ((state.in_exp || state.in_exp_base)
                && (self.nterms() > 1
                    || print_ring
                    || self.exponents(0).iter().filter(|e| **e > E::zero()).count() > 1
                    || !self.ring.is_one(&self.coefficients[0])));

        if add_paren {
            if state.in_sum {
                f.write_str("+")?;
                state.in_sum = false;
            }

            state.in_product = false;
            state.in_exp = false;
            state.in_exp_base = false;
            f.write_str("(")?;
        }
        let in_product = state.in_product;

        let var_map: Vec<String> = self
            .variables
            .as_ref()
            .iter()
            .map(|v| {
                v.format_string(
                    opts,
                    PrintState {
                        in_exp: true,
                        ..state
                    },
                )
            })
            .collect();

        state.level += 1;
        for monomial in self {
            let has_var = monomial.exponents.iter().any(|e| !e.is_zero());
            state.in_product = in_product || has_var;
            state.suppress_one = has_var; // any products before should not be considered
            state.in_exp |= print_ring; // make sure to add parentheses

            let mut suppressed_one = self.ring.format(monomial.coefficient, opts, state, f)?;

            for (var_id, e) in var_map.iter().zip(monomial.exponents) {
                if e.is_zero() {
                    continue;
                }
                if suppressed_one {
                    suppressed_one = false;
                } else if !opts.mode.is_latex() {
                    f.write_char(opts.multiplication_operator)?;
                }

                f.write_str(var_id)?;

                if e.to_i32() != 1 {
                    if opts.mode.is_latex() {
                        write!(f, "^{{{e}}}")?;
                    } else if opts.double_star_for_exponentiation {
                        write!(f, "**{e}")?;
                    } else {
                        write!(f, "^{e}")?;
                    }
                }
            }

            state.in_sum = true;
        }
        state.level -= 1;

        if self.is_zero() {
            f.write_char('0')?;
        }

        if print_ring && state.level == 0 {
            self.ring.format_ring(opts, state, f)?;
        }

        if add_paren {
            f.write_str(")")?;
        }

        Ok(false)
    }
}

impl<F: Ring + std::fmt::Debug, E: Exponent + std::fmt::Debug, O: MonomialOrder> std::fmt::Debug
    for MultivariatePolynomial<F, E, O>
{
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.is_zero() {
            return write!(f, "[]");
        }
        let mut first = true;
        write!(f, "[ ")?;
        for monomial in self {
            if first {
                first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(
                f,
                "{{ {:?}, {:?} }}",
                monomial.coefficient, monomial.exponents
            )?;
        }
        write!(f, " ]")
    }
}

impl<F: Ring + Display, E: Exponent, O: MonomialOrder> Display for MultivariatePolynomial<F, E, O> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.format(&PrintOptions::from_fmt(f), PrintState::from_fmt(f), f)
            .map(|_| ())
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> PartialEq for MultivariatePolynomial<F, E, O> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.variables != other.variables {
            if self.is_constant() != other.is_constant() {
                return false;
            }

            if self.is_zero() != other.is_zero() {
                return false;
            }

            if self.is_zero() {
                return true;
            }

            if self.is_constant() {
                return self.coefficients[0] == other.coefficients[0];
            }

            return false;
        }
        if self.nterms() != other.nterms() {
            return false;
        }
        self.exponents.eq(&other.exponents) && self.coefficients.eq(&other.coefficients)
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> std::hash::Hash for MultivariatePolynomial<F, E, O> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coefficients.hash(state);
        self.exponents.hash(state);

        if !self.is_constant() {
            self.variables.hash(state);
        }
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> Eq for MultivariatePolynomial<F, E, O> {}

impl<R: Ring, E: Exponent, O: MonomialOrder> InternalOrdering for MultivariatePolynomial<R, E, O> {
    /// An ordering of polynomials that has no intuitive meaning.
    fn internal_cmp(&self, other: &Self) -> Ordering {
        // TODO: what about different variables?
        Ord::cmp(&self.exponents, &other.exponents)
            .then_with(|| self.coefficients.internal_cmp(&other.coefficients))
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> Add for MultivariatePolynomial<F, E, O> {
    type Output = Self;

    fn add(mut self, mut other: Self) -> Self::Output {
        assert_eq!(self.ring, other.ring);

        self.unify_variables(&mut other);

        if self.is_zero() {
            return other;
        }
        if other.is_zero() {
            return self;
        }

        // Merge the two polynomials, which are assumed to be already sorted.

        let mut new_coefficients = vec![self.ring.zero(); self.nterms() + other.nterms()];
        let mut new_exponents: Vec<E> =
            vec![E::zero(); self.nvars() * (self.nterms() + other.nterms())];
        let mut new_nterms = 0;
        let mut i = 0;
        let mut j = 0;

        macro_rules! insert_monomial {
            ($source:expr, $index:expr) => {
                mem::swap(
                    &mut new_coefficients[new_nterms],
                    &mut $source.coefficients[$index],
                );

                new_exponents[new_nterms * $source.nvars()..(new_nterms + 1) * $source.nvars()]
                    .clone_from_slice($source.exponents($index));
                new_nterms += 1;
            };
        }

        while i < self.nterms() && j < other.nterms() {
            let c = O::cmp(self.exponents(i), other.exponents(j));
            match c {
                Ordering::Less => {
                    insert_monomial!(self, i);
                    i += 1;
                }
                Ordering::Greater => {
                    insert_monomial!(other, j);
                    j += 1;
                }
                Ordering::Equal => {
                    self.ring
                        .add_assign(&mut self.coefficients[i], &other.coefficients[j]);
                    if !self.ring.is_zero(&self.coefficients[i]) {
                        insert_monomial!(self, i);
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        while i < self.nterms() {
            insert_monomial!(self, i);
            i += 1;
        }

        while j < other.nterms() {
            insert_monomial!(other, j);
            j += 1;
        }

        new_coefficients.truncate(new_nterms);
        new_exponents.truncate(self.nvars() * new_nterms);

        Self {
            coefficients: new_coefficients,
            exponents: new_exponents,
            ring: self.ring,
            variables: self.variables,
            _phantom: PhantomData,
        }
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> Add<&MultivariatePolynomial<F, E, O>>
    for &MultivariatePolynomial<F, E, O>
{
    type Output = MultivariatePolynomial<F, E, O>;

    fn add(self, other: &MultivariatePolynomial<F, E, O>) -> Self::Output {
        assert_eq!(self.ring, other.ring);

        if self.is_zero() {
            return other.clone();
        }
        if other.is_zero() {
            return self.clone();
        }

        if self.variables != other.variables {
            let mut c1 = self.clone();
            let mut c2 = other.clone();
            c1.unify_variables(&mut c2);
            return c1 + c2;
        }

        // Merge the two polynomials, which are assumed to be already sorted.
        let mut new_coefficients = vec![self.ring.zero(); self.nterms() + other.nterms()];
        let mut new_exponents: Vec<E> =
            vec![E::zero(); self.nvars() * (self.nterms() + other.nterms())];
        let mut new_nterms = 0;
        let mut i = 0;
        let mut j = 0;

        macro_rules! insert_monomial {
            ($source:expr, $index:expr) => {
                new_coefficients[new_nterms] = $source.coefficients[$index].clone();
                new_exponents[new_nterms * $source.nvars()..(new_nterms + 1) * $source.nvars()]
                    .clone_from_slice($source.exponents($index));
                new_nterms += 1;
            };
        }

        while i < self.nterms() && j < other.nterms() {
            let c = O::cmp(self.exponents(i), other.exponents(j));
            match c {
                Ordering::Less => {
                    insert_monomial!(self, i);
                    i += 1;
                }
                Ordering::Greater => {
                    insert_monomial!(other, j);
                    j += 1;
                }
                Ordering::Equal => {
                    let coeff = self.ring.add(&self.coefficients[i], &other.coefficients[j]);
                    if !self.ring.is_zero(&coeff) {
                        new_coefficients[new_nterms] = coeff;
                        new_exponents[new_nterms * self.nvars()..(new_nterms + 1) * self.nvars()]
                            .clone_from_slice(self.exponents(i));
                        new_nterms += 1;
                    }
                    i += 1;
                    j += 1;
                }
            }
        }

        while i < self.nterms() {
            insert_monomial!(self, i);
            i += 1;
        }

        while j < other.nterms() {
            insert_monomial!(other, j);
            j += 1;
        }

        new_coefficients.truncate(new_nterms);
        new_exponents.truncate(self.nvars() * new_nterms);

        MultivariatePolynomial {
            coefficients: new_coefficients,
            exponents: new_exponents,
            ring: self.ring.clone(),
            variables: self.variables.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> Sub for MultivariatePolynomial<F, E, O> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.add(other.neg())
    }
}

impl<'a, F: Ring, E: Exponent, O: MonomialOrder> Sub<&'a MultivariatePolynomial<F, E, O>>
    for &MultivariatePolynomial<F, E, O>
{
    type Output = MultivariatePolynomial<F, E, O>;

    fn sub(self, other: &'a MultivariatePolynomial<F, E, O>) -> Self::Output {
        self + &other.clone().neg() // TODO: improve
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> Neg for MultivariatePolynomial<F, E, O> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        // Negate coefficients of all terms.
        for c in &mut self.coefficients {
            *c = self.ring.neg(&*c);
        }
        self
    }
}

impl<'a, F: Ring, E: Exponent> Mul<&'a MultivariatePolynomial<F, E, LexOrder>>
    for &MultivariatePolynomial<F, E, LexOrder>
{
    type Output = MultivariatePolynomial<F, E, LexOrder>;

    #[inline]
    fn mul(self, rhs: &'a MultivariatePolynomial<F, E, LexOrder>) -> Self::Output {
        assert_eq!(self.ring, rhs.ring);

        if self.nterms() == 0 || rhs.nterms() == 0 {
            return self.zero();
        }

        if self.is_constant() {
            return rhs.clone().mul_coeff(self.coefficients[0].clone());
        }
        if rhs.is_constant() {
            return self.clone().mul_coeff(rhs.coefficients[0].clone());
        }

        if self.variables != rhs.variables {
            let mut c1 = self.clone();
            let mut c2 = rhs.clone();
            c1.unify_variables(&mut c2);
            return c1.mul(&c2);
        }

        if self.nterms() == 1 {
            return rhs
                .clone()
                .mul_monomial(&self.coefficients[0], &self.exponents);
        }

        if rhs.nterms() == 1 {
            return self
                .clone()
                .mul_monomial(&rhs.coefficients[0], &rhs.exponents);
        }

        if let Some(r) = self.mul_dense(rhs) {
            r
        } else {
            self.heap_mul(rhs)
        }
    }
}

impl<'a, F: Ring, E: Exponent> Mul<&'a MultivariatePolynomial<F, E, LexOrder>>
    for MultivariatePolynomial<F, E, LexOrder>
{
    type Output = MultivariatePolynomial<F, E, LexOrder>;

    /// Multiply two polynomials, using either use dense multiplication or heap multiplication.
    #[inline]
    fn mul(self, rhs: &'a MultivariatePolynomial<F, E, LexOrder>) -> Self::Output {
        (&self) * rhs
    }
}

impl<'a, F: EuclideanDomain, E: PositiveExponent> Div<&'a MultivariatePolynomial<F, E, LexOrder>>
    for &MultivariatePolynomial<F, E, LexOrder>
{
    type Output = MultivariatePolynomial<F, E, LexOrder>;

    fn div(self, other: &'a MultivariatePolynomial<F, E, LexOrder>) -> Self::Output {
        self.try_div(other)
            .unwrap_or_else(|| panic!("No clean division of {self} by {other}"))
    }
}

impl<'a, F: EuclideanDomain, E: PositiveExponent> Div<&'a MultivariatePolynomial<F, E, LexOrder>>
    for MultivariatePolynomial<F, E, LexOrder>
{
    type Output = MultivariatePolynomial<F, E, LexOrder>;

    fn div(
        self: MultivariatePolynomial<F, E, LexOrder>,
        other: &'a MultivariatePolynomial<F, E, LexOrder>,
    ) -> Self::Output {
        (&self).div(other)
    }
}

impl<F: Ring, E: Exponent, O: MonomialOrder> MultivariatePolynomial<F, E, O> {
    /// Change the monomial order of the polynomial from `O` to `ON`.
    pub fn reorder<ON: MonomialOrder>(&self) -> MultivariatePolynomial<F, E, ON> {
        let mut sorted_index: Vec<_> = (0..self.nterms()).collect();
        sorted_index.sort_by(|a, b| ON::cmp(self.exponents(*a), self.exponents(*b)));

        let coefficients: Vec<_> = sorted_index
            .iter()
            .map(|i| self.coefficients[*i].clone())
            .collect();
        let exponents: Vec<_> = sorted_index
            .iter()
            .flat_map(|i| self.exponents(*i))
            .cloned()
            .collect();

        MultivariatePolynomial {
            coefficients,
            exponents,
            ring: self.ring.clone(),
            variables: self.variables.clone(),
            _phantom: PhantomData,
        }
    }

    /// Multiply every coefficient with `other`.
    pub fn mul_coeff(mut self, other: F::Element) -> Self {
        if self.ring.is_one(&other) {
            return self;
        }

        for c in &mut self.coefficients {
            self.ring.mul_assign(c, &other);
        }

        for i in (0..self.nterms()).rev() {
            if self.ring.is_zero(&self.coefficients[i]) {
                self.coefficients.remove(i);
                self.exponents
                    .drain(i * self.nvars()..(i + 1) * self.nvars());
            }
        }

        self
    }

    /// Map a coefficient using the function `f`.
    pub fn map_coeff<U: Ring, T: Fn(&F::Element) -> U::Element>(
        &self,
        f: T,
        field: U,
    ) -> MultivariatePolynomial<U, E, O> {
        let mut coefficients = Vec::with_capacity(self.coefficients.len());
        let mut exponents = Vec::with_capacity(self.exponents.len());

        for m in self.into_iter() {
            let nc = f(m.coefficient);
            if !field.is_zero(&nc) {
                coefficients.push(nc);
                exponents.extend(m.exponents);
            }
        }

        MultivariatePolynomial {
            coefficients,
            exponents,
            ring: field,
            variables: self.variables.clone(),
            _phantom: PhantomData,
        }
    }

    /// Add `exponents` to every exponent.
    pub fn mul_exp(mut self, exponents: &[E]) -> Self {
        debug_assert_eq!(self.nvars(), exponents.len());

        if self.nvars() == 0 {
            return self;
        }

        for e in self.exponents_iter_mut() {
            for (e1, e2) in e.iter_mut().zip(exponents) {
                *e1 = e1.checked_add(e2).expect("overflow in adding exponents");
            }
        }

        self
    }

    #[inline]
    pub fn max_coeff(&self) -> &F::Element {
        self.coefficients.last().unwrap()
    }

    #[inline]
    pub fn max_exp(&self) -> &[E] {
        if self.coefficients.is_empty() {
            panic!("Cannot get max exponent of empty polynomial");
        }

        &self.exponents[(self.nterms() - 1) * self.nvars()..self.nterms() * self.nvars()]
    }

    /// Add a new monomial with coefficient `other` and exponent one.
    pub fn add_constant(mut self, other: F::Element) -> Self {
        let nvars = self.nvars();
        self.append_monomial(other, &vec![E::zero(); nvars]);
        self
    }

    #[inline]
    pub fn mul_monomial(self, coefficient: &F::Element, exponents: &[E]) -> Self {
        self.mul_coeff(coefficient.clone()).mul_exp(exponents)
    }

    /// Check if the polynomial contains a variable `x`.
    pub fn contains(&self, x: usize) -> bool {
        if self.nvars() == 0 {
            return false;
        }

        for e in self.exponents.iter().skip(x).step_by(self.nvars()) {
            if *e != E::zero() {
                return true;
            }
        }
        false
    }

    /// Get the degree of the variable `x`.
    /// This operation is O(n).
    pub fn degree(&self, x: usize) -> E {
        if self.nvars() == 0 {
            return E::zero();
        }

        let mut max = None;
        for e in self.exponents.iter().skip(x).step_by(self.nvars()) {
            if max.map(|max| max < *e).unwrap_or(true) {
                max = Some(*e);
            }
        }
        max.unwrap_or(E::zero())
    }

    /// Get the lowest and highest exponent of the variable `x`.
    /// This operation is O(n).
    pub fn degree_bounds(&self, x: usize) -> (E, E) {
        if self.nvars() == 0 {
            return (E::zero(), E::zero());
        }

        let mut min = None;
        let mut max = None;
        for e in self.exponents.iter().skip(x).step_by(self.nvars()) {
            if max.map(|max| max < *e).unwrap_or(true) {
                max = Some(*e);
            }
            if min.map(|min| min > *e).unwrap_or(true) {
                min = Some(*e);
            }
        }
        (min.unwrap_or(E::zero()), max.unwrap_or(E::zero()))
    }

    // Get the highest degree of a variable in the leading monomial.
    pub fn ldegree(&self, v: usize) -> E {
        if self.is_zero() {
            return E::zero();
        }
        self.last_exponents()[v]
    }

    /// Get the highest degree of the leading monomial.
    pub fn ldegree_max(&self) -> E {
        if self.is_zero() {
            return E::zero();
        }
        *self.last_exponents().iter().max().unwrap_or(&E::zero())
    }

    /// Get the leading coefficient.
    pub fn lcoeff(&self) -> F::Element {
        if self.is_zero() {
            return self.ring.zero();
        }
        self.coefficients.last().unwrap().clone()
    }

    /// Perform self % var^pow.
    pub fn mod_var(&self, var: usize, pow: E) -> Self {
        let mut m = self.zero();
        for t in self.into_iter() {
            if t.exponents[var] < pow {
                m.append_monomial(t.coefficient.clone(), t.exponents);
            }
        }
        m
    }

    /// Take the derivative of the polynomial w.r.t the variable `var`.
    pub fn derivative(&self, var: usize) -> Self {
        debug_assert!(var < self.nvars());

        let mut res = self.zero_with_capacity(self.nterms());

        let mut exp = vec![E::zero(); self.nvars()];
        for x in self {
            if x.exponents[var] > E::zero() {
                exp.copy_from_slice(x.exponents);
                let pow = exp[var].to_i32() as u64;
                exp[var] = exp[var] - E::one();
                res.append_monomial(
                    self.ring.mul(x.coefficient, &self.ring.nth(pow.into())),
                    &exp,
                );
            }
        }

        res
    }

    /// Get the coefficient of the monomial with the given exponents if it is present.
    pub fn coefficient(&self, exponents: &[E]) -> Option<F::Element> {
        if self.is_zero() {
            if exponents.iter().all(|e| *e == E::zero()) {
                return Some(self.ring.zero());
            }
            return None;
        }

        let mut low = 0;
        let mut high = self.coefficients.len();

        while low < high {
            let mid = low + (high - low) / 2;

            match O::cmp(
                &self.exponents[mid * self.nvars()..(mid + 1) * self.nvars()],
                exponents,
            ) {
                Ordering::Equal => return Some(self.coefficients[mid].clone()),
                Ordering::Less => {
                    low = mid + 1;
                }
                Ordering::Greater => {
                    high = mid;
                }
            }
        }

        None
    }

    pub fn map_exp<E2: Exponent>(&self, f: impl Fn(&E) -> E2) -> MultivariatePolynomial<F, E2, O> {
        MultivariatePolynomial {
            coefficients: self.coefficients.clone(),
            exponents: self.exponents.iter().map(f).collect::<Vec<_>>(),
            ring: self.ring.clone(),
            variables: self.variables.clone(),
            _phantom: PhantomData,
        }
    }

    /// Kronecker map all variables starting from `start_index` using the powers given in `powers`:
    /// `x_i -> x_{start_index}^powers[i - start_index]` for `i > start_index`.
    pub fn kronecker_map(&self, powers: &[E], start_index: usize) -> Self {
        let mut res = self.zero_with_capacity(self.nterms());
        let mut new_exponents = vec![E::zero(); self.nvars()];
        for a in self {
            new_exponents[..(start_index + 1)].copy_from_slice(&a.exponents[..(start_index + 1)]);

            for (i, e) in a.exponents.iter().skip(start_index + 1).enumerate() {
                new_exponents[start_index] += powers[i] * *e;
            }
            res.append_monomial(a.coefficient.clone(), &new_exponents);
        }
        res
    }

    /// Invert a Kronecker map.
    pub fn kronecker_inv_map(&self, powers: &[E], start_index: usize) -> Self {
        let mut res = self.zero_with_capacity(self.nterms());
        let mut new_exponents = vec![E::zero(); self.nvars()];
        for a in self {
            new_exponents[..start_index].copy_from_slice(&a.exponents[..start_index]);

            let mut total = a.exponents[start_index];
            new_exponents[start_index] = total % powers[0];
            for i in (start_index + 1)..self.nvars() {
                total = total - new_exponents[i - 1];
                new_exponents[i] = total % powers[i - start_index];
                new_exponents[i] = new_exponents[i] / powers[i - start_index - 1];
            }
            res.append_monomial(a.coefficient.clone(), &new_exponents);
        }
        res
    }
}

impl<F: Ring, E: PositiveExponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Remove all non-occurring variables from the polynomial.
    pub fn condense(&mut self) {
        if self.nvars() == 0 {
            return;
        }

        let degrees: Vec<_> = (0..self.nvars())
            .filter(|i| self.degree(*i) > E::zero())
            .collect();

        let mut new_exponents = vec![E::zero(); self.nterms() * degrees.len()];

        if degrees.is_empty() {
            self.exponents = new_exponents;
            self.variables = Arc::new(vec![]);
            return;
        }

        for (d, e) in new_exponents
            .chunks_mut(degrees.len())
            .zip(self.exponents_iter())
        {
            for (dr, s) in d.iter_mut().zip(&degrees) {
                *dr = e[*s];
            }
        }

        self.exponents = new_exponents;
        self.variables = Arc::new(
            degrees
                .into_iter()
                .map(|x| self.variables[x].clone())
                .collect(),
        );
    }

    /// Replace a variable `n` in the polynomial by an element from
    /// the ring `v`.
    pub fn replace(&self, n: usize, v: &F::Element) -> MultivariatePolynomial<F, E, LexOrder> {
        if (n + 1..self.nvars()).all(|i| self.degree(i) == E::zero()) {
            return self.replace_last(n, v);
        }

        let mut res = self.zero_with_capacity(self.nterms());
        let mut e: SmallVec<[E; INLINED_EXPONENTS]> = smallvec![E::zero(); self.nvars()];

        // TODO: cache power taking?
        for t in self {
            if t.exponents[n] == E::zero() {
                res.append_monomial(t.coefficient.clone(), t.exponents);
                continue;
            }

            let c = self.ring.mul(
                t.coefficient,
                &self.ring.pow(v, t.exponents[n].to_i32() as u64),
            );

            e.copy_from_slice(t.exponents);
            e[n] = E::zero();
            res.append_monomial(c, &e);
        }

        res
    }

    /// Replace the last variable `n` in the polynomial by an element from
    /// the ring `v`.
    pub fn replace_last(&self, n: usize, v: &F::Element) -> MultivariatePolynomial<F, E, LexOrder> {
        const MAX_EXP_BUF: usize = 100000;
        let mut res = self.zero_with_capacity(self.nterms());
        let mut e = vec![E::zero(); self.nvars()];
        let mut res_cache =
            vec![self.ring.zero(); (self.degree(n).to_i32() as usize + 1).min(MAX_EXP_BUF)];

        let nvars = self.nvars();

        for t in self {
            if t.exponents[n] == E::zero() {
                res.append_monomial(t.coefficient.clone(), t.exponents);
                continue;
            }

            let ee = t.exponents[n].to_i32() as usize;

            let c = if ee < MAX_EXP_BUF {
                if self.ring.is_zero(&res_cache[ee]) {
                    res_cache[ee] = self.ring.pow(v, ee as u64);
                }

                self.ring.mul(t.coefficient, &res_cache[ee])
            } else {
                self.ring.mul(t.coefficient, &self.ring.pow(v, ee as u64))
            };

            if self.ring.is_zero(&c) {
                continue;
            }

            e.copy_from_slice(t.exponents);
            e[n] = E::zero();

            let mut different = false;
            if !res.is_zero() {
                let start = (res.nterms() - 1) * nvars;

                for i in (0..n).rev() {
                    if unsafe { e.get_unchecked(i) != res.exponents.get_unchecked(start + i) } {
                        different = true;
                        break;
                    }
                }
            } else {
                different = true;
            }

            if different {
                res.coefficients.push(c);
                res.exponents.extend_from_slice(&e);
            } else {
                let l = res.coefficients.last_mut().unwrap();
                self.ring.add_assign(l, &c);

                if self.ring.is_zero(l) {
                    res.coefficients.pop();
                    res.exponents.truncate(res.exponents.len() - self.nvars());
                }
            }
        }

        res
    }

    pub fn evaluate<T: FloatLike, M: Fn(&F::Element) -> T>(&self, map_coeff: M, point: &[T]) -> T {
        let mut res = map_coeff(&self.ring.zero());

        for t in self {
            let mut c = map_coeff(t.coefficient);

            for (i, v) in point.iter().zip(t.exponents) {
                if v != &E::zero() {
                    c *= i.pow(v.to_u32() as u64);
                }
            }

            res += c;
        }

        res
    }

    /// Evaluate the polynomial at the given point, mapping coefficients to the ring `U`.
    pub fn evaluate_with_coeff_map<U: Ring, T: Fn(&F::Element) -> U::Element>(
        &self,
        map_coeff: T,
        point: &[U::Element],
        ring: &U,
    ) -> U::Element {
        let mut res = map_coeff(&self.ring.zero());
        assert_eq!(point.len(), self.nvars());

        for t in self {
            let mut c = map_coeff(t.coefficient);

            for (i, v) in point.iter().zip(t.exponents) {
                if v != &E::zero() {
                    ring.mul_assign(&mut c, &ring.pow(i, v.to_u32() as u64));
                }
            }

            ring.add_assign(&mut res, &c);
        }

        res
    }

    /// Replace all variables in the polynomial by an element from
    /// the ring `v`.
    pub fn replace_all(&self, r: &[F::Element]) -> F::Element {
        let mut res = self.ring.zero();

        // TODO: cache power taking?
        for t in self {
            let mut c = t.coefficient.clone();

            for (i, v) in r.iter().zip(t.exponents) {
                if v != &E::zero() {
                    self.ring
                        .mul_assign(&mut c, &self.ring.pow(i, v.to_u32() as u64));
                }
            }

            self.ring.add_assign(&mut res, &c);
        }

        res
    }

    /// Replace a variable `n` in the polynomial by a polynomial `v`.
    pub fn replace_with_poly(&self, n: usize, v: &Self) -> Self {
        assert_eq!(self.variables, v.variables);

        if v.is_constant() {
            return self.replace(n, &v.lcoeff());
        }

        let mut res = self.zero_with_capacity(self.nterms());
        let mut exp = vec![E::zero(); self.nvars()];
        for t in self {
            if t.exponents[n] == E::zero() {
                res.append_monomial(t.coefficient.clone(), &t.exponents[..self.nvars()]);
                continue;
            }

            exp.copy_from_slice(t.exponents);
            exp[n] = E::zero();

            // TODO: cache v^e
            res = res
                + (&v.pow(t.exponents[n].to_i32() as usize)
                    * &self.monomial(t.coefficient.clone(), exp.clone()));
        }
        res
    }

    /// Replace all variables except `v` in the polynomial by elements from
    /// the ring.
    pub fn replace_except(
        &self,
        v: usize,
        r: &[(usize, F::Element)],
        cache: &mut [Vec<F::Element>],
    ) -> MultivariatePolynomial<F, E, LexOrder> {
        let mut tm: HashMap<E, F::Element> = HashMap::new();

        for t in self {
            let mut c = t.coefficient.clone();
            for (n, vv) in r {
                let p = t.exponents[*n].to_i32() as usize;
                if p > 0 {
                    if p < cache[*n].len() {
                        if self.ring.is_zero(&cache[*n][p]) {
                            cache[*n][p] = self.ring.pow(vv, p as u64);
                        }

                        self.ring.mul_assign(&mut c, &cache[*n][p]);
                    } else {
                        self.ring.mul_assign(&mut c, &self.ring.pow(vv, p as u64));
                    }
                }
            }

            tm.entry(t.exponents[v])
                .and_modify(|e| self.ring.add_assign(e, &c))
                .or_insert(c);
        }

        let mut res = self.zero();
        let mut e = vec![E::zero(); self.nvars()];
        for (k, c) in tm {
            e[v] = k;
            res.append_monomial(c, &e);
            e[v] = E::zero();
        }

        res
    }

    /// Shift a variable `var` to `var+shift`.
    pub fn shift_var(&self, var: usize, shift: &F::Element) -> Self {
        let d = self.degree(var).to_i32() as usize;

        let y_poly = self.to_univariate_polynomial_list(var);

        let mut v = vec![self.zero(); d + 1];
        for (x_poly, p) in y_poly {
            v[p.to_i32() as usize] = x_poly;
        }

        for k in 0..d {
            for j in (k..d).rev() {
                v[j] = &v[j] + &v[j + 1].clone().mul_coeff(shift.clone());
            }
        }

        let mut poly = self.zero();
        for (i, mut v) in v.into_iter().enumerate() {
            for x in v.exponents.chunks_mut(self.nvars()) {
                x[var] = E::from_i32(i as i32);
            }

            for m in &v {
                poly.append_monomial(m.coefficient.clone(), m.exponents);
            }
        }

        poly
    }

    /// Compute the inverse of the univariate polynomial in `var` up until `pow` using Newton's method.
    pub fn inverse_univariate(&self, var: usize, pow: E) -> Self {
        let mut g = self.constant(
            self.ring
                .try_div(&self.ring.one(), &self.get_constant())
                .unwrap(),
        );
        let mut exp = E::one();
        while exp < pow {
            exp = exp * E::from_u32(2);
            let h =
                g.clone().mul_coeff(self.ring.nth(2.into())) - self * &(&g * &g).mod_var(var, exp);
            g = h.mod_var(var, exp);
        }
        g
    }
}

impl<F: Ring, E: Exponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Check if all exponents are positive.
    pub fn is_polynomial(&self) -> bool {
        self.is_zero() || self.exponents.iter().all(|e| *e >= E::zero())
    }

    /// Get the leading coefficient under a given variable ordering.
    /// This operation is O(n) if the variables are out of order.
    pub fn lcoeff_varorder(&self, vars: &[usize]) -> F::Element {
        if vars.windows(2).all(|s| s[0] < s[1]) {
            return self.lcoeff();
        }

        let mut highest = vec![E::zero(); self.nvars()];
        let mut highestc = &self.ring.zero();

        'nextmon: for m in self.into_iter() {
            let mut more = false;
            for &v in vars {
                if more {
                    highest[v] = m.exponents[v];
                } else {
                    match m.exponents[v].cmp(&highest[v]) {
                        Ordering::Less => {
                            continue 'nextmon;
                        }
                        Ordering::Greater => {
                            highest[v] = m.exponents[v];
                            more = true;
                        }
                        Ordering::Equal => {}
                    }
                }
            }
            highestc = m.coefficient;
        }
        debug_assert!(!self.ring.is_zero(highestc));
        highestc.clone()
    }

    /// Get the leading coefficient of a multivariate polynomial viewed as a
    /// univariate polynomial in `x`.
    pub fn univariate_lcoeff(&self, x: usize) -> MultivariatePolynomial<F, E, LexOrder> {
        let d = self.degree(x);
        let mut lcoeff = self.zero();

        if self.coefficients.is_empty() {
            return lcoeff;
        }

        if d == E::zero() {
            return self.clone();
        }

        let mut e = vec![E::zero(); self.nvars()];
        for t in self {
            if t.exponents[x] == d {
                e.copy_from_slice(t.exponents);
                e[x] = E::zero();
                lcoeff.append_monomial(t.coefficient.clone(), &e);
            }
        }

        lcoeff
    }

    /// Get the leading coefficient of a multivariate polynomial viewed as a bivariate polynomial in the first two variables.
    pub fn bivariate_lcoeff(&self) -> MultivariatePolynomial<F, E, LexOrder> {
        let mut lcoeff = self.zero();

        if self.coefficients.is_empty() {
            return lcoeff;
        }

        let d = self.degree(0);
        let mut d1 = E::zero();
        for t in self {
            if t.exponents[0] == d && t.exponents[1] > d1 {
                d1 = t.exponents[1];
            }
        }

        let mut e = vec![E::zero(); self.nvars()];
        for t in self {
            if t.exponents[0] == d && t.exponents[1] == d1 {
                e.copy_from_slice(t.exponents);
                e[0] = E::zero();
                e[1] = E::zero();
                lcoeff.append_monomial(t.coefficient.clone(), &e);
            }
        }

        lcoeff
    }

    /// Get the leading coefficient viewed as a polynomial
    /// in all variables except the last variable `n`.
    pub fn lcoeff_last(&self, n: usize) -> MultivariatePolynomial<F, E, LexOrder> {
        if self.is_zero() {
            return self.clone();
        }
        // the last variable should have the least sorting priority,
        // so the last term should still be the lcoeff
        let last = self.last_exponents();

        let mut res = self.zero();
        let mut e: SmallVec<[E; INLINED_EXPONENTS]> = smallvec![E::zero(); self.nvars()];

        for t in (0..self.nterms()).rev() {
            if (0..self.nvars() - 1).all(|i| self.exponents(t)[i] == last[i] || i == n) {
                e[n] = self.exponents(t)[n];
                res.append_monomial(self.coefficients[t].clone(), &e);
                e[n] = E::zero();
            } else {
                break;
            }
        }

        res
    }

    /// Get the leading coefficient viewed as a polynomial
    /// in all variables with order as described in `vars` except the last variable in `vars`.
    /// This operation is O(n) if the variables are out of order.
    pub fn lcoeff_last_varorder(&self, vars: &[usize]) -> MultivariatePolynomial<F, E, LexOrder> {
        if self.is_zero() {
            return self.clone();
        }

        if vars.windows(2).all(|s| s[0] < s[1]) {
            return self.lcoeff_last(*vars.last().unwrap());
        }

        let (vars, lastvar) = vars.split_at(vars.len() - 1);

        let mut highest = vec![E::zero(); self.nvars()];
        let mut indices = Vec::with_capacity(10);

        'nextmon: for (i, m) in self.into_iter().enumerate() {
            let mut more = false;
            for &v in vars {
                if more {
                    highest[v] = m.exponents[v];
                } else {
                    match m.exponents[v].cmp(&highest[v]) {
                        Ordering::Less => {
                            continue 'nextmon;
                        }
                        Ordering::Greater => {
                            highest[v] = m.exponents[v];
                            indices.clear();
                            more = true;
                        }
                        Ordering::Equal => {}
                    }
                }
            }
            indices.push(i);
        }

        let mut res = self.zero();
        let mut e = vec![E::zero(); self.nvars()];
        for i in indices {
            e[lastvar[0]] = self.exponents(i)[lastvar[0]];
            res.append_monomial(self.coefficients[i].clone(), &e);
            e[lastvar[0]] = E::zero();
        }
        res
    }

    /// Change the order of the variables in the polynomial, using `order`.
    /// The map can also be reversed, by setting `inverse` to `true`.
    pub(crate) fn rearrange_impl(
        &self,
        order: &[usize],
        inverse: bool,
        update_variables: bool,
    ) -> MultivariatePolynomial<F, E, LexOrder> {
        let mut new_exp = vec![E::zero(); self.nterms() * self.nvars()];
        for (e, er) in new_exp.chunks_mut(self.nvars()).zip(self.exponents_iter()) {
            for x in 0..order.len() {
                if !inverse {
                    e[x] = er[order[x]];
                } else {
                    e[order[x]] = er[x];
                }
            }
        }

        let mut indices: Vec<usize> = (0..self.nterms()).collect();
        indices.sort_unstable_by_key(|&i| &new_exp[i * self.nvars()..(i + 1) * self.nvars()]);

        let mut res = self.zero_with_capacity(self.nterms());

        for i in indices {
            res.append_monomial(
                self.coefficients[i].clone(),
                &new_exp[i * self.nvars()..(i + 1) * self.nvars()],
            );
        }

        if update_variables {
            let mut vm = self.variables.as_ref().clone();
            for x in 0..order.len() {
                if !inverse {
                    vm[x] = self.variables[order[x]].clone();
                } else {
                    vm[order[x]] = self.variables[x].clone();
                }
            }

            res.variables = Arc::new(vm);
        }

        res
    }

    /// Change the order of the variables in the polynomial, using `order`.
    /// The map can also be reversed, by setting `inverse` to `true`.
    pub fn rearrange(
        &self,
        order: &[usize],
        inverse: bool,
    ) -> MultivariatePolynomial<F, E, LexOrder> {
        self.rearrange_impl(order, inverse, true)
    }

    /// Change the order of the variables in the polynomial, using `order`.
    /// The order may contain `None`, to signal unmapped indices. This operation
    /// allows the polynomial to grow in size.
    pub fn rearrange_with_growth(
        &self,
        order: &[PolyVariable],
    ) -> Result<MultivariatePolynomial<F, E, LexOrder>, String> {
        let new_order: Vec<_> = order
            .iter()
            .map(|x| self.variables.iter().position(|e| e == x))
            .collect();

        for (i, v) in self.variables.iter().enumerate() {
            if !new_order.contains(&Some(i)) && self.contains(i) {
                return Err(format!("Variable {v} is not in the new order"));
            }
        }

        let mut new_exp = vec![E::zero(); self.nterms() * order.len()];
        for (e, er) in new_exp.chunks_mut(order.len()).zip(self.exponents_iter()) {
            for x in 0..new_order.len() {
                if let Some(v) = new_order[x] {
                    e[x] = er[v];
                }
            }
        }

        let mut indices: Vec<usize> = (0..self.nterms()).collect();
        indices.sort_unstable_by_key(|&i| &new_exp[i * order.len()..(i + 1) * order.len()]);

        let mut res =
            MultivariatePolynomial::new(&self.ring, self.nterms().into(), Arc::new(order.to_vec()));

        for i in indices {
            res.append_monomial(
                self.coefficients[i].clone(),
                &new_exp[i * order.len()..(i + 1) * order.len()],
            );
        }

        Ok(res)
    }

    /// Compute `self^pow`.
    pub fn pow(&self, pow: usize) -> Self {
        if pow == 0 {
            return self.one();
        }
        if pow == 1 {
            return self.clone();
        }

        if self.is_constant() {
            return self.constant(self.ring.pow(&self.lcoeff(), pow as u64));
        }

        if self.coefficients.len() == 1 {
            return self.monomial(
                self.ring.pow(&self.coefficients[0], pow as u64),
                self.exponents
                    .iter()
                    .map(|x| *x * E::from_i32(pow as i32))
                    .collect(),
            );
        }

        // heuristic for when to use heap_pow
        if pow > 10 || (0..self.nvars()).all(|x| self.degree(x) <= E::from_i32(2)) {
            // if the characteristic is non-zero, a division by the exponent in the heap_pow algorithm
            // may cause a division by 0
            if self.ring.characteristic() == 0
                || self.nvars() == 1
                    && self.degree(0).to_i32() as usize + 1 < self.ring.characteristic()
            {
                return self.heap_pow(pow);
            }
        }

        // perform repeated multiplication instead of binary exponentiation, as
        // the latter is often much slower for sparse polynomials
        let mut res = self * self;
        for _ in 2..pow {
            res = &res * self;
        }
        res
    }

    pub fn to_univariate(&self, var: usize) -> UnivariatePolynomial<PolynomialRing<F, E>> {
        let c = self.to_univariate_polynomial_list(var);

        let mut p = UnivariatePolynomial::new(
            &PolynomialRing::from_poly(self),
            None,
            Arc::new(self.variables[var].clone()),
        );

        if c.is_empty() {
            return p;
        }

        p.coefficients = vec![self.zero(); c.last().unwrap().1.to_i32() as usize + 1];
        for (q, e) in c {
            if e < E::zero() {
                panic!("Negative exponent in univariate conversion");
            }

            p.coefficients[e.to_i32() as usize] = q;
        }

        p
    }

    pub fn to_univariate_from_univariate(&self, var: usize) -> UnivariatePolynomial<F> {
        let mut p =
            UnivariatePolynomial::new(&self.ring, None, Arc::new(self.variables[var].clone()));

        if self.is_zero() {
            return p;
        }

        p.coefficients = vec![p.ring.zero(); self.degree(var).to_i32() as usize + 1];
        for (q, e) in self.coefficients.iter().zip(self.exponents_iter()) {
            if e[var] < E::zero() {
                panic!("Negative exponent in univariate conversion");
            }

            p.coefficients[e[var].to_i32() as usize] = q.clone();
        }

        p
    }

    /// Create a univariate polynomial coefficient list out of a multivariate polynomial.
    /// The output is sorted in the degree.
    pub fn to_univariate_polynomial_list(
        &self,
        x: usize,
    ) -> Vec<(MultivariatePolynomial<F, E, LexOrder>, E)> {
        if self.coefficients.is_empty() {
            return vec![];
        }

        // get maximum degree for variable x
        let mut mindeg = E::zero();
        let mut maxdeg = E::zero();
        for t in 0..self.nterms() {
            let d = self.exponents(t)[x];
            if d > maxdeg {
                maxdeg = d;
            }
            if d < mindeg {
                mindeg = d;
            }
        }

        // construct the coefficient per power of x
        let mut result = vec![];
        let mut e: SmallVec<[E; INLINED_EXPONENTS]> = smallvec![E::zero(); self.nvars()];
        for d in mindeg.to_i32()..maxdeg.to_i32() + 1 {
            // TODO: add bounds estimate
            let mut a = self.zero();
            for t in 0..self.nterms() {
                if self.exponents(t)[x].to_i32() == d {
                    for (i, ee) in self.exponents(t).iter().enumerate() {
                        e[i] = *ee;
                    }
                    e[x] = E::zero();
                    a.append_monomial(self.coefficients[t].clone(), &e);
                }
            }

            if !a.is_zero() {
                result.push((a, E::from_i32(d)));
            }
        }

        result
    }

    /// Split the polynomial as a polynomial in `xs` if include is true,
    /// else excluding `xs`.
    pub fn to_multivariate_polynomial_list(
        &self,
        xs: &[usize],
        include: bool,
    ) -> HashMap<SmallVec<[E; INLINED_EXPONENTS]>, MultivariatePolynomial<F, E, LexOrder>> {
        if self.coefficients.is_empty() {
            return HashMap::new();
        }

        let mut tm: HashMap<
            SmallVec<[E; INLINED_EXPONENTS]>,
            MultivariatePolynomial<F, E, LexOrder>,
        > = HashMap::new();
        let mut e_not_in_xs = smallvec![E::zero(); self.nvars()];
        let mut e_in_xs = smallvec![E::zero(); self.nvars()];
        for t in self {
            for (i, ee) in t.exponents.iter().enumerate() {
                e_not_in_xs[i] = *ee;
                e_in_xs[i] = E::zero();
            }

            for x in xs {
                e_in_xs[*x] = e_not_in_xs[*x];
                e_not_in_xs[*x] = E::zero();
            }

            if include {
                tm.entry(e_in_xs.clone())
                    .and_modify(|x| x.append_monomial(t.coefficient.clone(), &e_not_in_xs))
                    .or_insert_with(|| {
                        MultivariatePolynomial::monomial(
                            self,
                            t.coefficient.clone(),
                            e_not_in_xs.to_vec(),
                        )
                    });
            } else {
                tm.entry(e_not_in_xs.clone())
                    .and_modify(|x| x.append_monomial(t.coefficient.clone(), &e_in_xs))
                    .or_insert_with(|| {
                        MultivariatePolynomial::monomial(
                            self,
                            t.coefficient.clone(),
                            e_in_xs.to_vec(),
                        )
                    });
            }
        }

        tm
    }

    pub(crate) fn mul_univariate_dense(&self, rhs: &Self, max_pow: Option<usize>) -> Self {
        if self.is_constant() {
            if let Some(m) = max_pow
                && let Some(var) = rhs.last_exponents().iter().position(|e| *e != E::zero())
                && rhs.degree(var).to_i32() > m as i32
            {
                return rhs
                    .mod_var(var, E::from_i32(m as i32 + 1))
                    .mul_coeff(self.lcoeff());
            }
            return rhs.clone().mul_coeff(self.lcoeff());
        }

        if rhs.is_constant() {
            if let Some(m) = max_pow
                && let Some(var) = self.last_exponents().iter().position(|e| *e != E::zero())
                && self.degree(var).to_i32() > m as i32
            {
                return self
                    .mod_var(var, E::from_i32(m as i32 + 1))
                    .mul_coeff(rhs.lcoeff());
            }
            return self.clone().mul_coeff(rhs.lcoeff());
        }

        let var = self
            .last_exponents()
            .iter()
            .position(|e| *e != E::zero())
            .unwrap();

        let d1 = self.degree(var);
        let d2 = rhs.degree(var);
        let mut max = (d1.to_i32() + d2.to_i32()) as usize;
        if let Some(m) = max_pow {
            max = max.min(m);
        }

        let mut coeffs = vec![self.ring.zero(); max + 1];

        for x in self {
            for y in rhs {
                let pos = x.exponents[var].to_i32() + y.exponents[var].to_i32();
                if pos as usize > max {
                    continue;
                }

                self.ring
                    .add_mul_assign(&mut coeffs[pos as usize], x.coefficient, y.coefficient);
            }
        }

        let mut exp = vec![E::zero(); self.nvars()];
        let mut res = self.zero_with_capacity(coeffs.len());
        for (p, c) in coeffs.into_iter().enumerate() {
            if !self.ring.is_zero(&c) {
                exp[var] = E::from_i32(p as i32);
                res.append_monomial(c, &exp);
            }
        }
        res
    }

    /// Synthetic division for univariate polynomials, where `div` is monic.
    pub(crate) fn quot_rem_univariate_monic(
        &self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        debug_assert_eq!(div.lcoeff(), self.ring.one());
        if self.is_zero() {
            return (self.clone(), self.clone());
        }

        let mut dividendpos = self.nterms() - 1; // work from the back

        let mut q = self.zero_with_capacity(self.nterms());
        let mut r = self.zero();

        // determine the variable
        let mut var = 0;
        for (i, x) in self.last_exponents().iter().enumerate() {
            if !x.is_zero() {
                var = i;
                break;
            }
        }

        let m = div.ldegree_max();
        let mut pow = self.ldegree_max();

        loop {
            // find the power in the dividend if it exists
            let mut coeff = loop {
                if self.exponents(dividendpos)[var] == pow {
                    break self.coefficients[dividendpos].clone();
                }
                if dividendpos == 0 || self.exponents(dividendpos)[var] < pow {
                    break self.ring.zero();
                }
                dividendpos -= 1;
            };

            let mut qindex = 0; // starting from highest
            let mut bindex = 0; // starting from lowest
            while bindex < div.nterms() && qindex < q.nterms() {
                while bindex + 1 < div.nterms()
                    && div.exponents(bindex)[var] + q.exponents(qindex)[var] < pow
                {
                    bindex += 1;
                }

                if div.exponents(bindex)[var] + q.exponents(qindex)[var] == pow {
                    self.ring.sub_mul_assign(
                        &mut coeff,
                        &div.coefficients[bindex],
                        &q.coefficients[qindex],
                    );
                }

                qindex += 1;
            }

            if !self.ring.is_zero(&coeff) {
                // can the division be performed? if not, add to rest
                // TODO: refactor
                let (quot, div) = if pow >= m {
                    (coeff, true)
                } else {
                    (coeff, false)
                };

                if div {
                    let nterms = q.nterms();
                    let nvars = q.nvars();
                    q.coefficients.push(quot);
                    q.exponents.resize((nterms + 1) * nvars, E::zero());
                    q.exponents[nterms * nvars + var] = pow - m;
                } else {
                    let nterms = r.nterms();
                    let nvars = r.nvars();
                    r.coefficients.push(quot);
                    r.exponents.resize((nterms + 1) * nvars, E::zero());
                    r.exponents[nterms * nvars + var] = pow;
                }
            }

            if pow.is_zero() {
                break;
            }

            pow = pow - E::one();
        }

        q.reverse_monomials();
        r.reverse_monomials();

        #[cfg(debug_assertions)]
        {
            if !(&q * div + r.clone() - self.clone()).is_zero() {
                panic!("Division failed: ({self})/({div}): q={q}, r={r}");
            }
        }

        (q, r)
    }

    fn mul_dense(
        &self,
        rhs: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> Option<MultivariatePolynomial<F, E, LexOrder>> {
        if !self.is_polynomial() || !rhs.is_polynomial() {
            return None;
        }

        let max_degs_rev = (0..self.nvars())
            .rev()
            .map(|i| 1 + self.degree(i).to_i32() as usize + rhs.degree(i).to_i32() as usize)
            .collect::<Vec<_>>();

        if max_degs_rev.iter().filter(|x| **x > 1).count() == 1 {
            if max_degs_rev.iter().sum::<usize>() < 10000 {
                return Some(self.mul_univariate_dense(rhs, None));
            }

            return None;
        }

        let mut total: usize = 1;
        for x in &max_degs_rev {
            if *x > MAX_DENSE_MUL_BUFFER_SIZE {
                return None;
            }

            if let Some(r) = total.checked_mul(*x) {
                total = r;
            } else {
                return None;
            }
        }

        if total > MAX_DENSE_MUL_BUFFER_SIZE {
            return None;
        }

        #[inline(always)]
        fn to_uni_var<E: Exponent>(s: &[E], max_degs_rev: &[usize]) -> u32 {
            let mut shift = 1;
            let mut res = s.last().unwrap().to_i32() as u32;
            for (ee, &x) in s.iter().rev().skip(1).zip(max_degs_rev) {
                shift *= x as u32;
                res += ee.to_i32() as u32 * shift;
            }
            res
        }

        #[inline(always)]
        fn from_uni_var<E: Exponent>(mut p: u32, max_degs_rev: &[usize], exp: &mut [E]) {
            for (ee, &x) in exp.iter_mut().rev().zip(max_degs_rev) {
                *ee = E::from_i32((p % x as u32) as i32);
                p /= x as u32;
            }
        }

        let mut uni_exp_self = vec![0; self.coefficients.len()];
        for (es, s) in &mut uni_exp_self.iter_mut().zip(self.exponents_iter()) {
            *es = to_uni_var(s, &max_degs_rev);
        }

        let mut uni_exp_rhs = vec![0; rhs.coefficients.len()];
        for (es, s) in &mut uni_exp_rhs.iter_mut().zip(rhs.exponents_iter()) {
            *es = to_uni_var(s, &max_degs_rev);
        }

        let mut exp = vec![E::zero(); self.nvars()];
        let mut r = self.zero_with_capacity(self.nterms().max(rhs.nterms()));

        // check if we need to use a dense indexing array to save memory
        if total < 1000 {
            let mut coeffs = vec![self.ring.zero(); total];

            for (c1, e1) in self.coefficients.iter().zip(&uni_exp_self) {
                for (c2, e2) in rhs.coefficients.iter().zip(&uni_exp_rhs) {
                    let pos = *e1 as usize + *e2 as usize;
                    self.ring.add_mul_assign(&mut coeffs[pos], c1, c2);
                }
            }

            for (p, c) in coeffs.into_iter().enumerate() {
                if !self.ring.is_zero(&c) {
                    from_uni_var(p as u32, &max_degs_rev, &mut exp);
                    r.append_monomial(c, &exp);
                }
            }

            Some(r)
        } else {
            let mut coeffs = Vec::with_capacity(self.nterms().max(rhs.nterms()));

            let mut coeff_index = DENSE_MUL_BUFFER.take();

            if coeff_index.len() < total {
                coeff_index.resize(total, 0u32);
            }

            for (c1, e1) in self.coefficients.iter().zip(&uni_exp_self) {
                for (c2, e2) in rhs.coefficients.iter().zip(&uni_exp_rhs) {
                    let pos = *e1 as usize + *e2 as usize;
                    if coeff_index[pos] == 0 {
                        coeffs.push(self.ring.mul(c1, c2));
                        coeff_index[pos] = coeffs.len() as u32;
                    } else {
                        self.ring.add_mul_assign(
                            &mut coeffs[coeff_index[pos] as usize - 1],
                            c1,
                            c2,
                        );
                    }
                }
            }

            for (p, c) in coeff_index[..total].iter_mut().enumerate() {
                if *c != 0 {
                    from_uni_var(p as u32, &max_degs_rev, &mut exp);
                    r.append_monomial(
                        std::mem::replace(&mut coeffs[*c as usize - 1], self.ring.zero()),
                        &exp,
                    );
                    *c = 0;
                }
            }

            DENSE_MUL_BUFFER.set(coeff_index);

            Some(r)
        }
    }

    fn heap_mul(
        &self,
        rhs: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> MultivariatePolynomial<F, E, LexOrder> {
        // place the smallest polynomial first, as this is faster
        // in the heap algorithm
        if self.nterms() > rhs.nterms() {
            return rhs.heap_mul(self);
        }

        let degree_sum: Vec<_> = (0..self.nvars())
            .map(|i| self.degree(i).to_i32() as i64 + rhs.degree(i).to_i32() as i64)
            .collect();

        // use a special routine if the exponents can be packed into a u64
        let mut pack_u8 = true;
        if self.nvars() <= 8
            && self.is_polynomial()
            && rhs.is_polynomial()
            && degree_sum.iter().all(|deg| {
                if *deg > 255 {
                    pack_u8 = false;
                }

                *deg <= 255 || self.nvars() <= 4 && *deg <= 65535
            })
        {
            return self.heap_mul_packed_exp(rhs, pack_u8);
        }

        let mut monomials = Vec::with_capacity(self.nterms() * self.nvars());
        monomials.extend(
            self.exponents(0)
                .iter()
                .zip(rhs.exponents(0))
                .map(|(e1, e2)| *e1 + *e2),
        );

        let monomials = UnsafeCell::new((self.nvars(), monomials));

        /// In order to prevent allocations of the exponents, store them in a single
        /// append-only vector and use a key to index into it. For performance,
        /// we use an unsafe cell.
        #[derive(Clone, Copy)]
        struct Key<'a, E: Exponent> {
            index: usize,
            monomials: &'a UnsafeCell<(usize, Vec<E>)>,
        }

        impl<E: Exponent> PartialEq for Key<'_, E> {
            #[inline(always)]
            fn eq(&self, other: &Self) -> bool {
                unsafe {
                    let b1 = &*self.monomials.get();
                    b1.1.get_unchecked(self.index..self.index + b1.0)
                        == b1.1.get_unchecked(other.index..other.index + b1.0)
                }
            }
        }

        impl<E: Exponent> Eq for Key<'_, E> {}

        impl<E: Exponent> PartialOrd for Key<'_, E> {
            #[inline(always)]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl<E: Exponent> Ord for Key<'_, E> {
            #[inline(always)]
            fn cmp(&self, other: &Self) -> Ordering {
                unsafe {
                    let b1 = &*self.monomials.get();
                    b1.1.get_unchecked(self.index..self.index + b1.0)
                        .cmp(b1.1.get_unchecked(other.index..other.index + b1.0))
                }
            }
        }

        impl<E: Exponent> std::hash::Hash for Key<'_, E> {
            #[inline(always)]
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
                unsafe {
                    let b = &*self.monomials.get();
                    b.1.get_unchecked(self.index..self.index + b.0).hash(state);
                }
            }
        }

        let mut res = self.zero_with_capacity(self.nterms().max(rhs.nterms()));

        let mut cache: HashMap<_, Vec<(usize, usize)>> = HashMap::new();
        let mut q_cache: Vec<Vec<(usize, usize)>> = vec![];

        // create a min-heap since our polynomials are sorted smallest to largest
        let mut h: BinaryHeap<Reverse<_>> = BinaryHeap::with_capacity(self.nterms());

        cache.insert(
            Key {
                index: 0,
                monomials: &monomials,
            },
            vec![(0, 0)],
        );
        h.push(Reverse(Key {
            index: 0,
            monomials: &monomials,
        }));

        // i=merged_index[j] signifies that self[i]*other[j] has been merged
        let mut merged_index = vec![0; rhs.nterms()];
        // in_heap[j] signifies that other[j] is in the heap
        let mut in_heap = vec![false; rhs.nterms()];
        in_heap[0] = true;

        while !h.is_empty() {
            let cur_mon = h.pop().unwrap();

            let mut coefficient = self.ring.zero();

            let mut q = cache.remove(&cur_mon.0).unwrap();

            for (i, j) in q.drain(..) {
                self.ring.add_mul_assign(
                    &mut coefficient,
                    &self.coefficients[i],
                    &rhs.coefficients[j],
                );

                merged_index[j] = i + 1;

                if i + 1 < self.nterms() && (j == 0 || merged_index[j - 1] > i + 1) {
                    let m = unsafe {
                        let b = &mut *monomials.get();
                        let index = b.1.len();
                        b.1.extend(
                            self.exponents(i + 1)
                                .iter()
                                .zip(rhs.exponents(j))
                                .map(|(e1, e2)| *e1 + *e2),
                        );

                        Key {
                            index,
                            monomials: &monomials,
                        }
                    };

                    if let Some(e) = cache.get_mut(&m) {
                        e.push((i + 1, j));
                    } else {
                        h.push(Reverse(m)); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((i + 1, j));
                            cache.insert(m, qq);
                        } else {
                            cache.insert(m, vec![(i + 1, j)]);
                        }
                    }
                } else {
                    in_heap[j] = false;
                }

                if j + 1 < rhs.nterms() && !in_heap[j + 1] {
                    let m = unsafe {
                        let b = &mut *monomials.get();
                        let index = b.1.len();
                        b.1.extend(
                            self.exponents(i)
                                .iter()
                                .zip(rhs.exponents(j + 1))
                                .map(|(e1, e2)| *e1 + *e2),
                        );

                        Key {
                            index,
                            monomials: &monomials,
                        }
                    };

                    if let Some(e) = cache.get_mut(&m) {
                        e.push((i, j + 1));
                    } else {
                        h.push(Reverse(m)); // only add when new

                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((i, j + 1));
                            cache.insert(m, qq);
                        } else {
                            cache.insert(m, vec![(i, j + 1)]);
                        }
                    }

                    in_heap[j + 1] = true;
                }
            }

            q_cache.push(q);

            if !self.ring.is_zero(&coefficient) {
                res.coefficients.push(coefficient);

                unsafe {
                    let b = &*monomials.get();
                    res.exponents
                        .extend_from_slice(&b.1[cur_mon.0.index..cur_mon.0.index + b.0]);
                }
            }
        }

        res
    }

    /// Heap multiplication, but with the exponents packed into a `u64`.
    /// Each exponent is limited to 65535 if there are four or fewer variables,
    /// or 255 if there are 8 or fewer variables.
    fn heap_mul_packed_exp(
        &self,
        other: &MultivariatePolynomial<F, E, LexOrder>,
        pack_u8: bool,
    ) -> MultivariatePolynomial<F, E, LexOrder> {
        let mut res = self.zero_with_capacity(self.nterms().max(other.nterms()));

        let pack_a: Vec<_> = if pack_u8 {
            self.exponents_iter().map(|c| E::pack(c)).collect()
        } else {
            self.exponents_iter().map(|c| E::pack_u16(c)).collect()
        };
        let pack_b: Vec<_> = if pack_u8 {
            other.exponents_iter().map(|c| E::pack(c)).collect()
        } else {
            other.exponents_iter().map(|c| E::pack_u16(c)).collect()
        };

        let mut cache: BTreeMap<u64, Vec<(usize, usize)>> = BTreeMap::new();
        let mut q_cache: Vec<Vec<(usize, usize)>> = vec![];

        // create a min-heap since our polynomials are sorted smallest to largest
        let mut h: BinaryHeap<Reverse<u64>> = BinaryHeap::with_capacity(self.nterms());

        let monom: u64 = pack_a[0] + pack_b[0];
        cache.insert(monom, vec![(0, 0)]);
        h.push(Reverse(monom));

        // i=merged_index[j] signifies that self[i]*other[j] has been merged
        let mut merged_index = vec![0; other.nterms()];
        // in_heap[j] signifies that other[j] is in the heap
        let mut in_heap = vec![false; other.nterms()];
        in_heap[0] = true;

        while let Some(cur_mon) = h.pop() {
            let mut coefficient = self.ring.zero();

            let mut q = cache.remove(&cur_mon.0).unwrap();

            for (i, j) in q.drain(..) {
                self.ring.add_mul_assign(
                    &mut coefficient,
                    &self.coefficients[i],
                    &other.coefficients[j],
                );

                merged_index[j] = i + 1;

                if i + 1 < self.nterms() && (j == 0 || merged_index[j - 1] > i + 1) {
                    let m = pack_a[i + 1] + pack_b[j];
                    if let Some(e) = cache.get_mut(&m) {
                        e.push((i + 1, j));
                    } else {
                        h.push(Reverse(m)); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((i + 1, j));
                            cache.insert(m, qq);
                        } else {
                            cache.insert(m, vec![(i + 1, j)]);
                        }
                    }
                } else {
                    in_heap[j] = false;
                }

                if j + 1 < other.nterms() && !in_heap[j + 1] {
                    let m = pack_a[i] + pack_b[j + 1];
                    if let Some(e) = cache.get_mut(&m) {
                        e.push((i, j + 1));
                    } else {
                        h.push(Reverse(m)); // only add when new

                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((i, j + 1));
                            cache.insert(m, qq);
                        } else {
                            cache.insert(m, vec![(i, j + 1)]);
                        }
                    }

                    in_heap[j + 1] = true;
                }
            }

            q_cache.push(q);

            if !self.ring.is_zero(&coefficient) {
                res.coefficients.push(coefficient);
                let len = res.exponents.len();

                res.exponents.resize(len + self.nvars(), E::zero());

                if pack_u8 {
                    E::unpack(cur_mon.0, &mut res.exponents[len..len + self.nvars()]);
                } else {
                    E::unpack_u16(cur_mon.0, &mut res.exponents[len..len + self.nvars()]);
                }
            }
        }
        res
    }

    /// Compute `self^pow` using a heap-based algorithm of "Sparse Polynomial Powering Using Heaps"
    /// by Michael Monagan and Roman Pearce.
    ///
    /// The caller must assure that the ring's cardinality is large enough to contain the exponents
    /// after Kronecker mapping.
    pub fn heap_pow(&self, pow: usize) -> Self {
        if self.is_constant() {
            return self.constant(self.ring.pow(&self.lcoeff(), pow as u64));
        }

        if self.coefficients.len() == 1 {
            return self.monomial(
                self.ring.pow(&self.coefficients[0], pow as u64),
                self.exponents
                    .iter()
                    .map(|x| *x * E::from_i32(pow as i32))
                    .collect(),
            );
        }

        #[inline(always)]
        fn to_uni_var<E: Exponent>(s: &[E], max_degs_rev: &[usize]) -> Integer {
            let mut shift = 1;
            let mut res = Integer::from(s.last().unwrap().to_i32());
            for (ee, &x) in s.iter().rev().skip(1).zip(max_degs_rev) {
                shift *= x as u32;
                res += ee.to_i32() as u32 * shift;
            }
            res
        }

        #[inline(always)]
        fn from_uni_var<E: Exponent>(mut p: Integer, max_degs_rev: &[usize], exp: &mut [E]) {
            for (ee, &x) in exp.iter_mut().rev().zip(max_degs_rev) {
                *ee = E::from_i32(((&p % x as u64).to_i64().unwrap() as u32) as i32);
                p /= x as u32;
            }
        }

        let degree_bounds = (0..self.nvars())
            .map(|v| self.degree_bounds(v))
            .collect::<Vec<_>>();

        let max_degs_rev = degree_bounds
            .iter()
            .rev()
            .map(|v| (v.1 - v.0).to_i32() as usize * pow + 1)
            .collect::<Vec<_>>();

        let mut exp = vec![E::zero(); self.nvars()];
        let mut f_exp: Vec<_> = self
            .exponents_iter()
            .map(|c| {
                for ((ee, x), d) in exp.iter_mut().zip(c.iter()).zip(&degree_bounds) {
                    *ee = *x - d.0;
                }

                to_uni_var(&exp, &max_degs_rev)
            })
            .collect();
        f_exp.reverse(); // descending order

        let mut g_coeff = vec![self.ring.pow(self.coefficients.last().unwrap(), pow as u64)];
        let mut g_exp = vec![f_exp[0].clone() * pow as u64];

        let mut cache: BTreeMap<Integer, Vec<(usize, usize)>> = BTreeMap::new();
        let mut q_cache: Vec<Vec<(usize, usize)>> = vec![];

        // create a min-heap since our polynomials are sorted smallest to largest
        let mut h: BinaryHeap<Integer> = BinaryHeap::with_capacity(self.nterms());

        let monom = f_exp[1].clone() + &g_exp[0];
        cache.insert(monom.clone(), vec![(1, 0)]);
        h.push(monom);

        // i=merged_index[j] signifies that self[i]*g[j] has been merged
        let mut merged_index = vec![0; self.nterms()];
        // in_heap[j] signifies that g[j] is in the heap
        let mut in_heap = vec![false; self.nterms()];
        in_heap[0] = true;

        while let Some(cur_mon) = h.pop() {
            let mut coefficient = self.ring.zero();

            let mut q = cache.remove(&cur_mon).unwrap();

            for (i, j) in q.drain(..) {
                self.ring.add_mul_assign(
                    &mut coefficient,
                    &g_coeff[j],
                    &self.ring.mul(
                        self.coefficient_back(i),
                        &self
                            .ring
                            .nth(g_exp[j].clone() - f_exp[i].clone() * pow as u64),
                    ),
                );

                if j + 1 >= merged_index.len() {
                    merged_index.resize(j + 2, 0);
                    in_heap.resize(j + 2, false);
                }

                merged_index[j] = i + 1;

                if i + 1 < self.nterms() && (j == 0 || merged_index[j - 1] > i + 1) {
                    let m = f_exp[i + 1].clone() + &g_exp[j];
                    if let Some(e) = cache.get_mut(&m) {
                        e.push((i + 1, j));
                    } else {
                        h.push(m.clone()); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((i + 1, j));
                            cache.insert(m, qq);
                        } else {
                            cache.insert(m, vec![(i + 1, j)]);
                        }
                    }
                } else {
                    in_heap[j] = false;
                }

                if j + 1 < g_exp.len() && !in_heap[j + 1] {
                    let m = f_exp[i].clone() + &g_exp[j + 1];
                    if let Some(e) = cache.get_mut(&m) {
                        e.push((i, j + 1));
                    } else {
                        h.push(m.clone()); // only add when new

                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((i, j + 1));
                            cache.insert(m, qq);
                        } else {
                            cache.insert(m, vec![(i, j + 1)]);
                        }
                    }

                    in_heap[j + 1] = true;
                }
            }

            q_cache.push(q);

            if !self.ring.is_zero(&coefficient) {
                g_exp.push(&cur_mon - &f_exp[0]);

                let q = self
                    .ring
                    .try_div(
                        &coefficient,
                        &self.ring.mul(
                            self.coefficient_back(0),
                            &self.ring.nth(g_exp[0].clone() + &f_exp[0] - cur_mon),
                        ),
                    )
                    .unwrap();
                g_coeff.push(q);

                if g_exp.len() >= in_heap.len() {
                    merged_index.resize(g_exp.len(), 0);
                    in_heap.resize(g_exp.len(), false);
                }

                if !in_heap[g_exp.len() - 1] {
                    let m = f_exp[1].clone() + &g_exp[g_exp.len() - 1];
                    if let Some(e) = cache.get_mut(&m) {
                        e.push((1, g_exp.len() - 1));
                    } else {
                        h.push(m.clone());
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((1, g_exp.len() - 1));
                            cache.insert(m, qq);
                        } else {
                            cache.insert(m, vec![(1, g_exp.len() - 1)]);
                        }
                    }

                    in_heap[g_exp.len() - 1] = true;
                }
            }
        }

        let mut res = self.zero();
        for (c, e) in g_coeff.into_iter().zip(g_exp).rev() {
            from_uni_var(e, &max_degs_rev, &mut exp);

            for (ee, d) in exp.iter_mut().zip(&degree_bounds) {
                *ee += d.0 * E::from_i32(pow as i32);
            }

            res.append_monomial(c, &exp);
        }
        res
    }
}

impl<F: EuclideanDomain, E: PositiveExponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Convert the polynomial to one in a number field, where the variable
    /// of the number field is moved into the coefficient.
    pub fn to_number_field(
        &self,
        field: &AlgebraicExtension<F>,
    ) -> MultivariatePolynomial<AlgebraicExtension<F>, E> {
        let var = &field.poly().get_vars_ref()[0];
        let Some(var_index) = self.get_vars_ref().iter().position(|x| x == var) else {
            return self.map_coeff(
                |c| field.to_element(field.poly().constant(c.clone())),
                field.clone(),
            );
        };

        let polys = self.to_multivariate_polynomial_list(&[var_index], false);

        // TODO: remove the variable from the variable map?
        let mut poly =
            MultivariatePolynomial::new(field, self.nterms().into(), self.variables.clone());
        for (e, c) in polys {
            let mut c2 = MultivariatePolynomial::new(
                &self.ring,
                c.nterms().into(),
                Arc::new(vec![self.variables.as_ref()[var_index].clone()]),
            );

            c2.exponents = c
                .exponents_iter()
                .map(|x| x[var_index].to_i32() as u16)
                .collect();
            c2.coefficients = c.coefficients;

            poly.append_monomial(field.to_element(c2), &e);
        }
        poly
    }

    /// Get the content from the coefficients.
    pub fn content(&self) -> F::Element {
        if self.coefficients.is_empty() {
            return self.ring.zero();
        }
        let mut c = self.coefficients.first().unwrap().clone();
        for cc in self.coefficients.iter().skip(1) {
            // early return if possible (not possible for rationals)
            if F::one_is_gcd_unit() && self.ring.is_one(&c) {
                break;
            }

            c = self.ring.gcd(&c, cc);
        }
        c
    }

    /// Divide every coefficient with `other`.
    pub fn div_coeff(mut self, other: &F::Element) -> Self {
        for c in &mut self.coefficients {
            let (quot, rem) = self.ring.quot_rem(c, other);
            debug_assert!(self.ring.is_zero(&rem));
            *c = quot;
        }
        self
    }

    /// Make the polynomial primitive by removing the content.
    pub fn make_primitive(self) -> Self {
        let c = self.content();
        self.div_coeff(&c)
    }
}

impl<F: EuclideanDomain, E: PositiveExponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Compute the remainder `self % div`.
    pub fn rem(&self, div: &MultivariatePolynomial<F, E, LexOrder>) -> Self {
        self.quot_rem(div, false).1
    }

    /// Divide two multivariate polynomials and return the quotient and remainder.
    pub fn quot_rem(
        &self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
        abort_on_remainder: bool,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        self.quot_rem_impl(div, abort_on_remainder)
    }

    /// Compute the p-adic expansion of the polynomial.
    /// It returns `[a0, a1, a2, ...]` such that `a0 + a1 * p^1 + a2 * p^2 + ... = self`.
    pub fn p_adic_expansion(&self, p: &Self) -> Vec<Self> {
        if self.variables != p.variables {
            let mut c1 = self.clone();
            let mut c2 = p.clone();
            c1.unify_variables(&mut c2);
            return c1.p_adic_expansion(&c2);
        }

        let mut res = vec![];
        let mut r = self.clone();
        while !r.is_zero() {
            let (q, rem) = r.quot_rem(p, false);
            res.push(rem);
            r = q;
        }
        res
    }
}

impl<F: Ring, E: Exponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Divide `self` by `div` if there is no remainder, else return `None`.
    pub fn try_div(
        &self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
    ) -> Option<MultivariatePolynomial<F, E, LexOrder>> {
        if div.is_zero() {
            panic!("Cannot divide by 0 polynomial");
        }

        if self.variables != div.variables {
            let mut c1 = self.clone();
            let mut c2 = div.clone();
            c1.unify_variables(&mut c2);
            return c1.try_div(&c2);
        }

        if self.is_zero() {
            return Some(self.clone());
        }

        // check if the leading coefficients divide
        self.ring.try_div(&self.lcoeff(), &div.lcoeff())?;

        if !self.is_polynomial() || !div.is_polynomial() {
            // remove all negative exponents
            let mut c1 = self.clone();
            let mut c2 = div.clone();
            let degrees = (0..self.nvars())
                .map(|v| E::zero() - div.degree_bounds(v).0.min(E::zero()))
                .collect::<Vec<_>>();

            c1 = c1.mul_exp(&degrees);
            c2 = c2.mul_exp(&degrees);

            let mut degrees = (0..self.nvars())
                .map(|v| E::zero() - self.degree_bounds(v).0.min(E::zero()))
                .collect::<Vec<_>>();

            c1 = c1.mul_exp(&degrees);

            let r = c1.try_div(&c2)?;

            for d in &mut degrees {
                *d = E::zero() - *d;
            }

            return Some(r.mul_exp(&degrees));
        }

        if !self.is_polynomial() {
            return None;
        }

        if (0..self.nvars()).any(|v| self.degree(v) < div.degree(v)) {
            return None;
        }

        if self.ring.characteristic().is_zero() {
            // test division of constant term (evaluation at x_i = 0)
            let c = div.get_constant();
            if !self.ring.is_zero(&c)
                && !self.ring.is_one(&c)
                && self.ring.try_div(&self.get_constant(), &c).is_none()
            {
                return None;
            }

            // test division at x_i = 1
            let mut num = self.ring.zero();
            for c in &self.coefficients {
                self.ring.add_assign(&mut num, c);
            }
            let mut den = self.ring.zero();
            for c in &div.coefficients {
                self.ring.add_assign(&mut den, c);
            }

            if !self.ring.is_zero(&den)
                && !self.ring.is_one(&den)
                && self.ring.try_div(&num, &den).is_none()
            {
                return None;
            }
        }

        let (a, b) = self.quot_rem_impl(div, true);
        if b.nterms() == 0 { Some(a) } else { None }
    }

    /// Divide two multivariate polynomials and return the quotient and remainder.
    ///
    /// The input must not have negative exponents.
    fn quot_rem_impl(
        &self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
        abort_on_remainder: bool,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        if div.is_zero() {
            panic!("Cannot divide by 0 polynomial");
        }

        if self.is_zero() {
            return (self.clone(), self.clone());
        }

        if div.is_one() {
            return (self.clone(), self.zero());
        }

        if self.variables != div.variables {
            let mut c1 = self.clone();
            let mut c2 = div.clone();
            c1.unify_variables(&mut c2);
            return c1.quot_rem_impl(&c2, abort_on_remainder);
        }

        if self.nterms() == div.nterms() {
            if self == div {
                return (self.one(), self.zero());
            }

            // check if one is a multiple of the other
            if let Some(q) = self.ring.try_div(&self.lcoeff(), &div.lcoeff())
                && self
                    .into_iter()
                    .zip(div)
                    .all(|(t1, t2)| t1.exponents == t2.exponents)
                && self
                    .into_iter()
                    .zip(div)
                    .all(|(t1, t2)| &self.ring.mul(t2.coefficient, &q) == t1.coefficient)
            {
                return (self.constant(q), self.zero());
            }
        }

        if div.is_constant() {
            let mut q = self.clone();
            let dive = div.to_monomial_view(0);

            if let Some(i) = div.ring.try_inv(dive.coefficient) {
                return (q.mul_coeff(i), self.zero());
            }

            for c in &mut q.coefficients {
                if let Some(quot) = self.ring.try_div(c, dive.coefficient) {
                    *c = quot;
                } else {
                    return (self.zero(), self.clone());
                }
            }

            return (q, self.zero());
        }

        // check if the division is univariate with the same variable
        let degree_sum: Vec<_> = (0..self.nvars())
            .map(|i| self.degree(i).to_i32() as usize + div.degree(i).to_i32() as usize)
            .collect();

        if div.ring.is_one(&div.lcoeff()) && degree_sum.iter().filter(|x| **x > 0).count() == 1 {
            return self.quot_rem_univariate_monic(div);
        }

        let mut pack_u8 = true;
        if self.nvars() <= 8
            && (0..self.nvars()).all(|i| {
                let deg = self.degree(i).to_i32() as u32;
                if deg > 127 {
                    pack_u8 = false;
                }

                deg <= 127 || self.nvars() <= 4 && deg <= 32767
            })
        {
            self.heap_division_packed_exp(div, abort_on_remainder, pack_u8)
        } else {
            self.heap_division(div, abort_on_remainder)
        }
    }

    /// Heap division for multivariate polynomials, using a cache so that only unique
    /// monomial exponents appear in the heap.
    /// Reference: "Sparse polynomial division using a heap" by Monagan, Pearce (2011)
    ///
    /// The input must not have negative exponents.
    fn heap_division(
        &self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
        abort_on_remainder: bool,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        let mut q = self.zero_with_capacity(self.nterms());
        let mut r = self.zero();

        let mut div_monomial_in_heap = vec![false; div.nterms()];
        let mut merged_index_of_div_monomial_in_quotient = vec![0; div.nterms()];

        let mut cache: BTreeMap<Vec<E>, Vec<(usize, usize, bool)>> = BTreeMap::new();

        let mut h: BinaryHeap<Vec<E>> = BinaryHeap::with_capacity(self.nterms());
        let mut q_cache: Vec<Vec<(usize, usize, bool)>> = vec![];

        let mut m = vec![E::zero(); div.nvars()];
        let mut m_cache = vec![E::zero(); div.nvars()];
        let mut c;

        let mut k = 0;
        while !h.is_empty() || k < self.nterms() {
            if k < self.nterms() && (h.is_empty() || self.exponents_back(k) >= h.peek().unwrap()) {
                for (s, e) in m.iter_mut().zip(self.exponents_back(k)) {
                    *s = *e;
                }

                c = self.coefficient_back(k).clone();
                k += 1;
            } else {
                for (s, e) in m.iter_mut().zip(h.peek().unwrap().as_slice()) {
                    *s = *e;
                }
                c = self.ring.zero();
            }

            if let Some(monomial) = h.peek()
                && &m == monomial
            {
                h.pop().unwrap();

                let mut qs = cache.remove(&m).unwrap();
                for (i, j, next_in_divisor) in qs.drain(..) {
                    self.ring
                        .sub_mul_assign(&mut c, &q.coefficients[i], div.coefficient_back(j));

                    if next_in_divisor && j + 1 < div.nterms() {
                        // quotient heap product
                        for ((m, e1), e2) in m_cache
                            .iter_mut()
                            .zip(q.exponents(i))
                            .zip(div.exponents_back(j + 1))
                        {
                            *m = *e1 + *e2;
                        }

                        // TODO: make macro
                        if let Some(e) = cache.get_mut(&m_cache) {
                            e.push((i, j + 1, true));
                        } else {
                            h.push(m_cache.clone()); // only add when new
                            if let Some(mut qq) = q_cache.pop() {
                                qq.push((i, j + 1, true));
                                cache.insert(m_cache.clone(), qq);
                            } else {
                                cache.insert(m_cache.clone(), vec![(i, j + 1, true)]);
                            }
                        }
                    } else if !next_in_divisor {
                        merged_index_of_div_monomial_in_quotient[j] = i + 1;

                        if i + 1 < q.nterms()
                            && (j == 1 // the divisor starts with the sub-leading term in the heap
                                    || merged_index_of_div_monomial_in_quotient[j - 1] > i + 1)
                        {
                            for ((m, e1), e2) in m_cache
                                .iter_mut()
                                .zip(q.exponents(i + 1))
                                .zip(div.exponents_back(j))
                            {
                                *m = *e1 + *e2;
                            }

                            if let Some(e) = cache.get_mut(&m_cache) {
                                e.push((i + 1, j, false));
                            } else {
                                h.push(m_cache.clone()); // only add when new
                                if let Some(mut qq) = q_cache.pop() {
                                    qq.push((i + 1, j, false));
                                    cache.insert(m_cache.clone(), qq);
                                } else {
                                    cache.insert(m_cache.clone(), vec![(i + 1, j, false)]);
                                }
                            }
                        } else {
                            div_monomial_in_heap[j] = false;
                        }

                        if j + 1 < div.nterms() && !div_monomial_in_heap[j + 1] {
                            for ((m, e1), e2) in m_cache
                                .iter_mut()
                                .zip(q.exponents(i))
                                .zip(div.exponents_back(j + 1))
                            {
                                *m = *e1 + *e2;
                            }

                            if let Some(e) = cache.get_mut(&m_cache) {
                                e.push((i, j + 1, false));
                            } else {
                                h.push(m_cache.clone()); // only add when new

                                if let Some(mut qq) = q_cache.pop() {
                                    qq.push((i, j + 1, false));
                                    cache.insert(m_cache.clone(), qq);
                                } else {
                                    cache.insert(m_cache.clone(), vec![(i, j + 1, false)]);
                                }
                            }

                            div_monomial_in_heap[j + 1] = true;
                        }
                    }
                }

                q_cache.push(qs);
            }

            if self.ring.is_zero(&c) {
                continue;
            }

            if div.last_exponents().iter().zip(&m).all(|(ge, me)| me >= ge) {
                if let Some(quot) = self.ring.try_div(&c, &div.lcoeff()) {
                    q.coefficients.push(quot);
                } else if abort_on_remainder {
                    r = self.one();
                    return (q, r);
                } else {
                    return (self.zero(), self.clone());
                }

                q.exponents.extend(
                    div.last_exponents()
                        .iter()
                        .zip(&m)
                        .map(|(ge, me)| *me - *ge),
                );

                if div.nterms() == 1 {
                    continue;
                }

                for ((m, e1), e2) in m_cache
                    .iter_mut()
                    .zip(q.last_exponents())
                    .zip(div.exponents_back(1))
                {
                    *m = *e1 + *e2;
                }

                if q.nterms() < div.nterms() {
                    // using quotient heap

                    if let Some(e) = cache.get_mut(&m_cache) {
                        e.push((q.nterms() - 1, 1, true));
                    } else {
                        h.push(m_cache.clone()); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((q.nterms() - 1, 1, true));
                            cache.insert(m_cache.clone(), qq);
                        } else {
                            cache.insert(m_cache.clone(), vec![(q.nterms() - 1, 1, true)]);
                        }
                    }
                } else if q.nterms() >= div.nterms() {
                    // using divisor heap
                    if !div_monomial_in_heap[1] {
                        div_monomial_in_heap[1] = true;

                        if let Some(e) = cache.get_mut(&m_cache) {
                            e.push((q.nterms() - 1, 1, false));
                        } else {
                            h.push(m_cache.clone()); // only add when new
                            if let Some(mut qq) = q_cache.pop() {
                                qq.push((q.nterms() - 1, 1, false));
                                cache.insert(m_cache.clone(), qq);
                            } else {
                                cache.insert(m_cache.clone(), vec![(q.nterms() - 1, 1, false)]);
                            }
                        }
                    }
                } else {
                    // switch to divisor heap
                    for index in &mut merged_index_of_div_monomial_in_quotient {
                        *index = q.nterms() - 1;
                    }
                    debug_assert!(div_monomial_in_heap.iter().any(|c| !c));
                    div_monomial_in_heap[1] = true;

                    if let Some(e) = cache.get_mut(&m_cache) {
                        e.push((q.nterms() - 1, 1, false));
                    } else {
                        h.push(m_cache.clone()); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((q.nterms() - 1, 1, false));
                            cache.insert(m_cache.clone(), qq);
                        } else {
                            cache.insert(m_cache.clone(), vec![(q.nterms() - 1, 1, false)]);
                        }
                    }
                }
            } else if abort_on_remainder {
                r = self.one();
                return (q, r);
            } else {
                r.coefficients.push(c);
                r.exponents.extend(&m);
            }
        }

        // q and r have the highest monomials first
        q.reverse_monomials();
        r.reverse_monomials();

        #[cfg(debug_assertions)]
        {
            if !(&q * div + r.clone() - self.clone()).is_zero() {
                panic!("Division failed: ({self})/({div}): q={q}, r={r}");
            }
        }

        (q, r)
    }

    /// Heap division, but with the exponents packed into a `u64`.
    /// Each exponent is limited to 32767 if there are 5 or fewer variables,
    /// or 127 if there are 8 or fewer variables, such that the last bit per byte can
    /// be used to check for subtraction overflow, serving as a division test.
    ///
    /// The input must not have negative exponents.
    fn heap_division_packed_exp(
        &self,
        div: &MultivariatePolynomial<F, E, LexOrder>,
        abort_on_remainder: bool,
        pack_u8: bool,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        let mut q = self.zero_with_capacity(self.nterms());
        let mut r = self.zero();

        let pack_a: Vec<_> = if pack_u8 {
            self.exponents_iter().map(|c| E::pack(c)).collect()
        } else {
            self.exponents_iter().map(|c| E::pack_u16(c)).collect()
        };
        let pack_div: Vec<_> = if pack_u8 {
            div.exponents_iter().map(|c| E::pack(c)).collect()
        } else {
            div.exponents_iter().map(|c| E::pack_u16(c)).collect()
        };

        let mut div_monomial_in_heap = vec![false; div.nterms()];
        let mut merged_index_of_div_monomial_in_quotient = vec![0; div.nterms()];

        let mut cache: BTreeMap<u64, Vec<(usize, usize, bool)>> = BTreeMap::new();

        #[inline(always)]
        fn divides(a: u64, b: u64, pack_u8: bool) -> Option<u64> {
            let d = a.overflowing_sub(b).0;
            if pack_u8 && (d & 9259542123273814144u64 == 0)
                || !pack_u8 && (d & 9223512776490647552u64 == 0)
            {
                Some(d)
            } else {
                None
            }
        }

        let mut h: BinaryHeap<u64> = BinaryHeap::with_capacity(self.nterms());
        let mut q_cache: Vec<Vec<(usize, usize, bool)>> = Vec::with_capacity(self.nterms());

        let mut m;
        let mut m_cache;
        let mut c;

        let mut q_exp = Vec::with_capacity(self.nterms());

        let mut k = 0;
        while !h.is_empty() || k < self.nterms() {
            if k < self.nterms()
                && (h.is_empty() || pack_a[self.nterms() - k - 1] >= *h.peek().unwrap())
            {
                m = pack_a[self.nterms() - k - 1];

                c = self.coefficient_back(k).clone();

                k += 1;
            } else {
                m = *h.peek().unwrap();
                c = self.ring.zero();
            }

            if let Some(monomial) = h.peek()
                && &m == monomial
            {
                h.pop().unwrap();

                let mut qs = cache.remove(&m).unwrap();
                for (i, j, next_in_divisor) in qs.drain(..) {
                    // TODO: use fraction-free routines
                    self.ring
                        .sub_mul_assign(&mut c, &q.coefficients[i], div.coefficient_back(j));

                    if next_in_divisor && j + 1 < div.nterms() {
                        // quotient heap product
                        m_cache = q_exp[i] + pack_div[div.nterms() - (j + 1) - 1];

                        // TODO: make macro
                        if let Some(e) = cache.get_mut(&m_cache) {
                            e.push((i, j + 1, true));
                        } else {
                            h.push(m_cache); // only add when new
                            if let Some(mut qq) = q_cache.pop() {
                                qq.push((i, j + 1, true));
                                cache.insert(m_cache, qq);
                            } else {
                                cache.insert(m_cache, vec![(i, j + 1, true)]);
                            }
                        }
                    } else if !next_in_divisor {
                        merged_index_of_div_monomial_in_quotient[j] = i + 1;

                        if i + 1 < q.nterms()
                            && (j == 1 // the divisor starts with the sub-leading term in the heap
                                    || merged_index_of_div_monomial_in_quotient[j - 1] > i + 1)
                        {
                            m_cache = q_exp[i + 1] + pack_div[div.nterms() - j - 1];

                            if let Some(e) = cache.get_mut(&m_cache) {
                                e.push((i + 1, j, false));
                            } else {
                                h.push(m_cache); // only add when new
                                if let Some(mut qq) = q_cache.pop() {
                                    qq.push((i + 1, j, false));
                                    cache.insert(m_cache, qq);
                                } else {
                                    cache.insert(m_cache, vec![(i + 1, j, false)]);
                                }
                            }
                        } else {
                            div_monomial_in_heap[j] = false;
                        }

                        if j + 1 < div.nterms() && !div_monomial_in_heap[j + 1] {
                            m_cache = q_exp[i] + pack_div[div.nterms() - (j + 1) - 1];

                            if let Some(e) = cache.get_mut(&m_cache) {
                                e.push((i, j + 1, false));
                            } else {
                                h.push(m_cache); // only add when new

                                if let Some(mut qq) = q_cache.pop() {
                                    qq.push((i, j + 1, false));
                                    cache.insert(m_cache, qq);
                                } else {
                                    cache.insert(m_cache, vec![(i, j + 1, false)]);
                                }
                            }

                            div_monomial_in_heap[j + 1] = true;
                        }
                    }
                }

                q_cache.push(qs);
            }

            if self.ring.is_zero(&c) {
                continue;
            }

            let q_e = divides(m, pack_div[pack_div.len() - 1], pack_u8);
            if let Some(q_e) = q_e {
                if let Some(quot) = self.ring.try_div(&c, &div.lcoeff()) {
                    q.coefficients.push(quot);
                } else if abort_on_remainder {
                    r = self.one();
                    return (q, r);
                } else {
                    return (self.zero(), self.clone());
                }

                let len = q.exponents.len();
                q.exponents.resize(len + self.nvars(), E::zero());

                if pack_u8 {
                    E::unpack(q_e, &mut q.exponents[len..len + self.nvars()]);
                } else {
                    E::unpack_u16(q_e, &mut q.exponents[len..len + self.nvars()]);
                }
                q_exp.push(q_e);

                if div.nterms() == 1 {
                    continue;
                }

                m_cache = q_exp.last().unwrap() + pack_div[pack_div.len() - 2];

                if q.nterms() < div.nterms() {
                    // using quotient heap

                    if let Some(e) = cache.get_mut(&m_cache) {
                        e.push((q.nterms() - 1, 1, true));
                    } else {
                        h.push(m_cache); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((q.nterms() - 1, 1, true));
                            cache.insert(m_cache, qq);
                        } else {
                            cache.insert(m_cache, vec![(q.nterms() - 1, 1, true)]);
                        }
                    }
                } else if q.nterms() >= div.nterms() {
                    // using divisor heap
                    if !div_monomial_in_heap[1] {
                        div_monomial_in_heap[1] = true;

                        if let Some(e) = cache.get_mut(&m_cache) {
                            e.push((q.nterms() - 1, 1, false));
                        } else {
                            h.push(m_cache); // only add when new
                            if let Some(mut qq) = q_cache.pop() {
                                qq.push((q.nterms() - 1, 1, false));
                                cache.insert(m_cache, qq);
                            } else {
                                cache.insert(m_cache, vec![(q.nterms() - 1, 1, false)]);
                            }
                        }
                    }
                } else {
                    // switch to divisor heap
                    for index in &mut merged_index_of_div_monomial_in_quotient {
                        *index = q.nterms() - 1;
                    }
                    debug_assert!(div_monomial_in_heap.iter().any(|c| !c));
                    div_monomial_in_heap[1] = true;

                    if let Some(e) = cache.get_mut(&m_cache) {
                        e.push((q.nterms() - 1, 1, false));
                    } else {
                        h.push(m_cache); // only add when new
                        if let Some(mut qq) = q_cache.pop() {
                            qq.push((q.nterms() - 1, 1, false));
                            cache.insert(m_cache, qq);
                        } else {
                            cache.insert(m_cache, vec![(q.nterms() - 1, 1, false)]);
                        }
                    }
                }
            } else if abort_on_remainder {
                r = self.one();
                return (q, r);
            } else {
                r.coefficients.push(c);
                let len = r.exponents.len();
                r.exponents.resize(len + self.nvars(), E::zero());

                if pack_u8 {
                    E::unpack(m, &mut r.exponents[len..len + self.nvars()]);
                } else {
                    E::unpack_u16(m, &mut r.exponents[len..len + self.nvars()]);
                }
            }
        }

        // q and r have the highest monomials first
        q.reverse_monomials();
        r.reverse_monomials();

        #[cfg(debug_assertions)]
        {
            if !(&q * div + r.clone() - self.clone()).is_zero() {
                panic!("Division failed: ({self})/({div}): q={q}, r={r}");
            }
        }

        (q, r)
    }
}

impl<F: Field, E: Exponent, O: MonomialOrder> MultivariatePolynomial<F, E, O> {
    /// Make the polynomial monic, i.e., make the leading coefficient `1` by
    /// multiplying all monomials with `1/lcoeff`.
    pub fn make_monic(self) -> Self {
        if self.lcoeff() != self.ring.one() {
            let ci = self.ring.inv(&self.lcoeff());
            self.mul_coeff(ci)
        } else {
            self
        }
    }
}

impl<F: Field, E: PositiveExponent, O: MonomialOrder> MultivariatePolynomial<F, E, O> {
    /// Integrate the polynomial w.r.t the variable `var`,
    /// producing the antiderivative with zero constant.
    pub fn integrate(&self, var: usize) -> Self {
        debug_assert!(var < self.nvars());
        if self.is_zero() {
            return self.zero();
        }

        let mut res = self.zero_with_capacity(self.nterms());

        let mut exp = vec![E::zero(); self.nvars()];
        for x in self {
            exp.copy_from_slice(x.exponents);
            let pow = exp[var].to_u32() as u64;
            exp[var] += E::one();
            res.append_monomial(
                self.ring
                    .div(x.coefficient, &self.ring.nth(Integer::from(pow) + 1)),
                &exp,
            );
        }

        res
    }
}

impl<F: Field, E: PositiveExponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Optimized division routine for univariate polynomials over a field, which
    /// makes the divisor monic first.
    pub fn quot_rem_univariate(
        &self,
        div: &mut MultivariatePolynomial<F, E, LexOrder>,
    ) -> (
        MultivariatePolynomial<F, E, LexOrder>,
        MultivariatePolynomial<F, E, LexOrder>,
    ) {
        if self.is_zero() {
            return (self.clone(), self.clone());
        }

        if div.nterms() == 1 {
            // calculate inverse once
            let inv = self.ring.inv(&div.coefficients[0]);

            if div.is_constant() {
                let mut q = self.clone();
                for c in &mut q.coefficients {
                    self.ring.mul_assign(c, &inv);
                }

                return (q, self.zero());
            }

            let mut q = self.zero_with_capacity(self.nterms());
            let mut r = self.zero();
            let dive = div.exponents(0);

            for m in self.into_iter() {
                if m.exponents.iter().zip(dive).all(|(a, b)| a >= b) {
                    q.coefficients.push(self.ring.mul(m.coefficient, &inv));

                    for (ee, ed) in m.exponents.iter().zip(dive) {
                        q.exponents.push(*ee - *ed);
                    }
                } else {
                    r.coefficients.push(m.coefficient.clone());
                    r.exponents.extend(m.exponents);
                }
            }
            return (q, r);
        }

        // normalize the lcoeff to 1 to prevent a costly inversion
        if !self.ring.is_one(&div.lcoeff()) {
            let o = div.lcoeff();
            let inv = self.ring.inv(&div.lcoeff());

            for c in &mut div.coefficients {
                self.ring.mul_assign(c, &inv);
            }

            let mut res = self.quot_rem_univariate_monic(div);

            for c in &mut res.0.coefficients {
                self.ring.mul_assign(c, &inv);
            }

            for c in &mut div.coefficients {
                self.ring.mul_assign(c, &o);
            }

            return res;
        }

        self.quot_rem_univariate_monic(div)
    }

    /// Compute self^n % m where m is a polynomial
    pub fn exp_mod_univariate(&self, mut n: Integer, m: &mut Self) -> Self {
        if n.is_zero() {
            return self.one();
        }

        // use binary exponentiation and mod at every stage
        let mut x = self.quot_rem_univariate(m).1;
        let mut y = self.one();
        while !n.is_one() {
            if (&n % &Integer::Single(2)).is_one() {
                y = (&y * &x).quot_rem_univariate(m).1;
                n -= &Integer::one();
            }

            x = (&x * &x).quot_rem_univariate(m).1;
            n /= 2;
        }

        (x * &y).quot_rem_univariate(m).1
    }

    /// Perform a fast univariate division of `self` by `div`, using a precomputed inverse of `div` reversed.
    pub fn quot_rem_univariate_fast(&self, div: &Self, var: usize, inv_div: &Self) -> (Self, Self) {
        if div.is_zero() {
            panic!("Cannot divide by 0 polynomial");
        }

        if self.is_zero() {
            return (self.clone(), self.clone());
        }

        let deg_a = *self.last_exponents().iter().max().unwrap();
        let deg_b = *div.last_exponents().iter().max().unwrap();
        if deg_a < deg_b {
            return (self.zero(), self.clone());
        }

        let m = deg_a - deg_b;

        let mut self_i = self.clone();
        self_i.reverse();
        let mut q = (&self_i * &inv_div.mod_var(var, m + E::one())).mod_var(var, m + E::one());

        q.reverse();

        if q.degree(var) < m {
            let mut exp = vec![E::zero(); self.nvars()];
            exp[var] = m - q.degree(var);
            q = q.mul_exp(&exp);
        }

        let r = self - &(div * &q);

        (q, r)
    }

    /// Compute `self^n % m` where `m` is a polynomial, using a precomputed inverse of `m` reversed.
    pub fn exp_mod_univariate_fast(
        &mut self,
        var: usize,
        mut n: Integer,
        m: &Self,
        m_inv: &Self,
    ) -> Self {
        if n.is_zero() {
            return self.one();
        }

        // use binary exponentiation and mod at every stage
        let mut x = self.quot_rem_univariate_fast(m, var, m_inv).1;
        let mut y = self.one();
        while !n.is_one() {
            if (&n % &Integer::Single(2)).is_one() {
                y = (&y * &x).quot_rem_univariate_fast(m, var, m_inv).1;
                n -= &Integer::one();
            }

            x = (&x * &x).quot_rem_univariate_fast(m, var, m_inv).1;
            n /= 2;
        }

        (x * &y).quot_rem_univariate_fast(m, var, m_inv).1
    }

    /// Compute `(g, s, t)` where `self * s + other * t = g`
    /// by means of the extended Euclidean algorithm.
    pub fn eea_univariate(&self, other: &Self) -> (Self, Self, Self) {
        let mut r0 = self.clone().make_monic();
        let mut r1 = other.clone().make_monic();
        let mut s0 = self.constant(self.ring.inv(&self.lcoeff()));
        let mut s1 = self.zero();
        let mut t0 = self.zero();
        let mut t1 = self.constant(self.ring.inv(&other.lcoeff()));

        while !r1.is_zero() {
            let (q, r) = r0.quot_rem_univariate(&mut r1);
            if self.ring.is_zero(&r.lcoeff()) {
                return (r1, s1, t1);
            }

            let a = self.ring.inv(&r.lcoeff());
            (r1, r0) = (r.mul_coeff(a.clone()), r1);
            (s1, s0) = ((s0 - &q * &s1).mul_coeff(a.clone()), s1);
            (t1, t0) = ((t0 - q * &t1).mul_coeff(a), t1);
        }

        (r0, s0, t0)
    }

    /// Compute `(s1,...,n2)` where `A0 * s0 + ... + An * sn = g`
    /// where `Ai = prod(polys[j], j != i)`
    /// by means of the extended Euclidean algorithm.
    ///
    /// The `polys` must be pairwise co-prime.
    pub fn diophantine_univariate(polys: &mut [Self], b: &Self) -> Vec<Self> {
        let mut cur = polys.last().unwrap().clone();
        let mut a = vec![cur.clone()];
        for x in polys[1..].iter().rev().skip(1) {
            cur = cur * x;
            a.push(cur.clone());
        }
        a.reverse();

        let mut ss = vec![];
        let mut cur_s = b.clone();
        for (p, aa) in polys.iter_mut().zip(&mut a) {
            let (g, s, t) = p.eea_univariate(aa);
            debug_assert!(g.is_one());
            let new_s = (t * &cur_s).quot_rem_univariate(p).1;
            ss.push(new_s);
            cur_s = (s * &cur_s).quot_rem_univariate(aa).1;
        }

        ss.push(cur_s);
        ss
    }

    /// Find a rational fraction `n(x)/d(x)`, the Pade approximant,
    ///  such that `d(x)*self-n(x)=0 mod x^(deg_n+deg_d+1)` and
    /// `deg(d(x)) <= deg_d` and `deg(n(x) <= deg_n` using the extended Euclidean algorithm.
    pub fn rational_approximant_univariate(&self, deg_n: u32, deg_d: u32) -> Option<(Self, Self)>
    where
        F: PolynomialGCD<E>,
    {
        let Some(var) = self.last_exponents().iter().position(|x| *x > E::zero()) else {
            return Some((self.clone(), self.one()));
        };

        let mut exp = self.last_exponents().to_vec();
        exp[var] = E::from_u32(deg_n) + E::from_u32(deg_d) + E::one();
        let mut v0 = self.monomial(self.ring.one(), exp);
        let mut v1 = self.zero();

        let mut w0 = self.clone();
        let mut w1 = self.one();

        while w0.degree(var).to_u32() > deg_n {
            let (q, r) = v0.quot_rem_univariate(&mut w0);
            (w1, v1) = (v1 - q * &w1, w1);
            (v0, w0) = (w0, r);
        }

        // TODO: normalize denominator?
        let r = w0.gcd(&w1);

        Some((w0 / &r, w1 / &r))
    }

    /// Shift a variable `var` to `var+shift`, using an optimized routine that
    /// uses a power cache. If working in a finite field, the characteristic
    /// should be larger than the degree of the polynomial.
    pub fn shift_var_cached(&self, var: usize, shift: &F::Element) -> Self {
        let d = self.degree(var).to_u32() as usize;

        let y_poly = self.to_univariate_polynomial_list(var);
        let mut sample_powers = Vec::with_capacity(d + 1);
        let mut accum = self.ring.one();

        sample_powers.push(self.ring.one());
        for _ in 0..d {
            self.ring.mul_assign(&mut accum, shift);
            sample_powers.push(accum.clone());
        }
        let mut v = vec![self.zero(); d + 1];
        for (x_poly, p) in y_poly {
            let i = p.to_u32() as usize;
            v[i] = x_poly.mul_coeff(sample_powers[i].clone());
        }

        for k in 0..d {
            for j in (k..d).rev() {
                v[j] = &v[j] + &v[j + 1];
            }
        }

        let mut poly = self.zero();
        let mut accum_inv = self.ring.one();
        let sample_point_inv = self.ring.inv(shift);
        for (i, mut v) in v.into_iter().enumerate() {
            v = v.mul_coeff(accum_inv.clone());

            for x in v.exponents.chunks_mut(self.nvars()) {
                x[var] = E::from_u32(i as u32);
            }

            for m in &v {
                poly.append_monomial(m.coefficient.clone(), m.exponents);
            }

            self.ring.mul_assign(&mut accum_inv, &sample_point_inv);
        }

        poly
    }
}

impl<R: Ring, E: Exponent> Derivable for PolynomialRing<R, E> {
    type Variable = PolyVariable;

    fn derivative(
        &self,
        p: &MultivariatePolynomial<R, E>,
        x: &PolyVariable,
    ) -> MultivariatePolynomial<R, E> {
        if let Some(pos) = p.get_vars_ref().iter().position(|v| v == x) {
            p.derivative(pos)
        } else {
            self.zero()
        }
    }
}

impl<R: EuclideanDomain, E: Exponent> MultivariatePolynomial<AlgebraicExtension<R>, E> {
    /// Convert the polynomial to a multivariate polynomial that contains the
    /// variable in the number field.
    pub fn from_number_field(&self) -> MultivariatePolynomial<R, E> {
        let var = &self.ring.poly().get_vars_ref()[0];

        let (var_map, var_index) = if let Some(p) =
            self.get_vars_ref().iter().position(|v| v == var)
        {
            if self.degree(p) > E::zero() {
                panic!(
                    "The variable of the minimal polynomial of the coefficient field also appears in the polynomial"
                );
            }
            (self.variables.clone(), p)
        } else {
            let p = self.get_vars_ref().len();
            let mut v = self.get_vars_ref().to_vec();
            v.push(var.clone());
            (Arc::new(v), p)
        };

        let mut poly =
            MultivariatePolynomial::new(&self.ring.poly().ring, self.nterms().into(), var_map);
        let mut exp = vec![E::zero(); poly.nvars()];
        for t in self {
            exp[..self.nvars()].copy_from_slice(t.exponents);
            for t2 in &t.coefficient.poly {
                exp[var_index] = E::from_i32(t2.exponents[0].to_i32());
                poly.append_monomial(t2.coefficient.clone(), &exp);
            }
        }

        poly
    }
}

impl<E: Exponent> From<&MultivariatePolynomial<IntegerRing, E>>
    for MultivariatePolynomial<RationalField, E>
{
    fn from(val: &MultivariatePolynomial<IntegerRing, E>) -> Self {
        MultivariatePolynomial {
            coefficients: val.coefficients.iter().map(|x| x.into()).collect(),
            exponents: val.exponents.clone(),
            ring: Q,
            variables: val.variables.clone(),
            _phantom: PhantomData,
        }
    }
}

/// View object for a term in a multivariate polynomial.
#[derive(Copy, Clone, Debug)]
pub struct MonomialView<'a, F: 'a + Ring, E: 'a + Exponent> {
    pub coefficient: &'a F::Element,
    pub exponents: &'a [E],
}

/// Iterator over terms in a multivariate polynomial.
pub struct MonomialViewIterator<'a, F: Ring, E: Exponent, O: MonomialOrder> {
    poly: &'a MultivariatePolynomial<F, E, O>,
    index: usize,
}

impl<'a, F: Ring, E: Exponent, O: MonomialOrder> Iterator for MonomialViewIterator<'a, F, E, O> {
    type Item = MonomialView<'a, F, E>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.index == self.poly.nterms() {
            None
        } else {
            let view = MonomialView {
                coefficient: &self.poly.coefficients[self.index],
                exponents: self.poly.exponents(self.index),
            };
            self.index += 1;
            Some(view)
        }
    }
}

impl<'a, F: Ring, E: Exponent, O: MonomialOrder> IntoIterator
    for &'a MultivariatePolynomial<F, E, O>
{
    type Item = MonomialView<'a, F, E>;
    type IntoIter = MonomialViewIterator<'a, F, E, O>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            poly: self,
            index: 0,
        }
    }
}

// General implementation for F::Element blocked on https://github.com/rust-lang/rust/issues/20400
impl<E: Exponent, O: MonomialOrder> PartialEq<Integer>
    for MultivariatePolynomial<IntegerRing, E, O>
{
    #[inline]
    fn eq(&self, other: &Integer) -> bool {
        self.is_constant() && self.get_constant() == *other
    }
}

impl<R: Ring + FractionNormalization + EuclideanDomain, E: Exponent, O: MonomialOrder>
    PartialEq<Fraction<R>> for MultivariatePolynomial<FractionField<R>, E, O>
{
    #[inline]
    fn eq(&self, other: &Fraction<R>) -> bool {
        self.is_constant() && self.get_constant() == *other
    }
}

impl<E: Exponent> MultivariatePolynomial<C, E> {
    /// Convert the polynomial to a multivariate polynomial over rationals.
    pub fn to_extension(&self) -> MultivariatePolynomial<AlgebraicExtension<Q>, E> {
        let extension = AlgebraicExtension::new_complex(Q);
        let var = extension.poly().variables.clone();
        self.map_coeff(
            move |c| AlgebraicNumber {
                poly: if c.re.is_zero() {
                    MultivariatePolynomial {
                        coefficients: vec![c.im.clone()],
                        exponents: vec![1],
                        ring: Q,
                        variables: var.clone(),
                        _phantom: PhantomData,
                    }
                } else if c.im.is_zero() {
                    MultivariatePolynomial {
                        coefficients: vec![c.re.clone()],
                        exponents: vec![0],
                        ring: Q,
                        variables: var.clone(),
                        _phantom: PhantomData,
                    }
                } else {
                    MultivariatePolynomial {
                        coefficients: vec![c.re.clone(), c.im.clone()],
                        exponents: vec![0, 1],
                        ring: Q,
                        variables: var.clone(),
                        _phantom: PhantomData,
                    }
                },
            },
            extension,
        )
    }
}

impl<E: Exponent> MultivariatePolynomial<AlgebraicExtension<Q>, E> {
    /// Convert the polynomial to a multivariate polynomial over complex numbers.
    pub fn to_complex(&self) -> MultivariatePolynomial<C, E> {
        if self.ring.poly().nterms() != 2 || self.ring.poly().get_constant() != Q.one() {
            panic!(
                "Cannot convert to complex polynomial: the coefficient field is not complex numbers"
            );
        }

        self.map_coeff(
            |c| match c.poly.coefficients.len() {
                0 => Complex {
                    re: Q.zero(),
                    im: Q.zero(),
                },
                1 => {
                    if c.poly.exponents[0] == 0 {
                        Complex {
                            re: c.poly.coefficients[0].clone(),
                            im: Q.zero(),
                        }
                    } else {
                        Complex {
                            re: Q.zero(),
                            im: c.poly.coefficients[0].clone(),
                        }
                    }
                }
                2 => Complex {
                    re: c.poly.coefficients[0].clone(),
                    im: c.poly.coefficients[1].clone(),
                },
                _ => unreachable!(),
            },
            C,
        )
    }
}

#[cfg(test)]
mod test {
    use crate::{atom::AtomCore, domains::integer::Z, domains::rational::Q, parse, symbol};

    #[test]
    fn mul_packed() {
        let p1 = parse!("v1^2+v2^3*v3*+3*v1^4+4*v2*v3").to_polynomial::<_, u8>(&Z, None);
        let b = &p1 * &p1;
        let r = parse!(
            "16*v2^2*v3^2+8*v1^2*v2*v3+v1^4+24*v1^4*v2^4*v3^2+6*v1^6*v2^3*v3+9*v1^8*v2^6*v3^2"
        );
        assert_eq!(b.to_expression(), r)
    }

    #[test]
    fn mul_full() {
        let p1 = parse!("v1^2+v2^3*v3*+3*v1^4+4*v2*v3+v4+v5+v6*v1*v2+v7*v5+v8+v9*v8")
            .to_polynomial::<_, u8>(&Z, None);
        let b = &p1 * &p1;

        let r = parse!(
            "16*v2^2*v3^2+8*v1*v2^2*v3*v6+8*v1^2*v2*v3+v1^2*v2^2*v6^2+2*v1^3*v2*v6+v1^4+24*v1^4*v2^4*v3^2+6*v1^5*v2^4*v3*v6+6*v1^6*v2^3*v3+9*v1^8*v2^6*v3^2+8*v8*v2*v3+8*v8*v2*v3*v9+2*v8*v1*v2*v6+2*v8*v1*v2*v9*v6+2*v8*v1^2+2*v8*v1^2*v9+6*v8*v1^4*v2^3*v3+6*v8*v1^4*v2^3*v3*v9+v8^2+2*v8^2*v9+v8^2*v9^2+8*v5*v2*v3+8*v5*v2*v3*v7+2*v5*v1*v2*v6+2*v5*v1*v2*v7*v6+2*v5*v1^2+2*v5*v1^2*v7+6*v5*v1^4*v2^3*v3+6*v5*v1^4*v2^3*v3*v7+2*v5*v8+2*v5*v8*v9+2*v5*v8*v7+2*v5*v8*v7*v9+v5^2+2*v5^2*v7+v5^2*v7^2+8*v4*v2*v3+2*v4*v1*v2*v6+2*v4*v1^2+6*v4*v1^4*v2^3*v3+2*v4*v8+2*v4*v8*v9+2*v4*v5+2*v4*v5*v7+v4^2"
        );
        assert_eq!(b.to_expression(), r)
    }

    #[test]
    fn div_packed() {
        let p1 = parse!("(v1+v2*5+v3*v2+v1*v2*v3)(v1+v2+v3)").to_polynomial::<_, u8>(&Z, None);

        let p2 = parse!("v1+v2+v3+1").to_polynomial::<_, u8>(&Z, p1.variables.clone());

        let (q, r) = p1.quot_rem(&p2, false);
        assert_eq!(q.to_expression(), parse!("-1+5*v2+v1+v1*v2*v3"));
        assert_eq!(r.to_expression(), parse!("1+v3-4*v2+v2*v3^2+v2^2*v3"));
    }

    #[test]
    fn div_full() {
        let p1 = parse!("(v1+v2*5+v3*v2+v1*v2*v3+v4+v5+v6+v7+v8+v9*v8)(v1+v2+v3)")
            .to_polynomial::<_, u8>(&Z, None);

        let p2 = parse!("v1+v2+v3+1").to_polynomial::<_, u8>(&Z, p1.variables.clone());

        let (q, r) = p1.quot_rem(&p2, false);
        assert_eq!(
            q.to_expression(),
            parse!("-1+v8+v8*v9+v7+v6+v5+v4+5*v2+v1+v1*v2*v3")
        );
        assert_eq!(
            r.to_expression(),
            parse!("1-v8-v8*v9-v7-v6-v5-v4+v3-4*v2+v2*v3^2+v2^2*v3")
        );
    }

    #[test]
    fn fuse_variables() {
        let p1 = parse!("v1+v2").to_polynomial::<_, u8>(&Z, None);
        let p2 = parse!("v4").to_polynomial::<_, u8>(&Z, None);

        let p3 = parse!("v3").to_polynomial::<_, u8>(&Z, p1.variables.clone());

        let r = p1 * &p2 + p3;

        assert_eq!(
            r.get_vars_ref(),
            &[symbol!("v1"), symbol!("v2"), symbol!("v4"), symbol!("v3")]
        );
    }

    #[test]
    fn fast_exp_mod() {
        let mut a = parse!("x^3 + 2*x + 3").to_polynomial::<_, u8>(&Q, None);
        let b = parse!("x^2 + x + 2").to_polynomial::<_, u8>(&Q, None);
        let mut b_rev = b.clone();
        b_rev.reverse();
        let c = b_rev.inverse_univariate(0, 2.into());
        let d = a.exp_mod_univariate_fast(0, 4.into(), &b, &c);
        assert_eq!(d.to_expression(), parse!("367+333*x"));
    }
}
