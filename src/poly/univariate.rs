//! Univariate polynomials and their ring.

use std::{
    cmp::Ordering,
    collections::HashMap,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::{Arc, LazyLock, OnceLock, RwLock},
};

use numerica::domains::float::{ComplexBall, Float, RealBall};

use crate::{
    atom::Atom,
    domains::{
        EuclideanDomain, Field, InternalOrdering, Ring, RingOps, SelfRing, Set,
        algebraic_number::{AlgebraicExtension, AlgebraicNumber},
        float::{Complex, FloatField, FloatLike, Real, SingleFloat},
        integer::{Integer, IntegerRing, Z},
        rational::{Q, Rational, RationalField},
        rational_polynomial::{
            FromNumeratorAndDenominator, RationalPolynomial, RationalPolynomialField,
        },
    },
    printer::{PrintOptions, PrintState},
    transcendental::{root, root_var},
};

use super::{
    PolyVariable, PositiveExponent,
    factor::Factorize,
    polynomial::{MultivariatePolynomial, PolynomialRing},
};

/// A univariate polynomial ring.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct UnivariatePolynomialRing<R: Ring> {
    ring: R,
    variable: Arc<PolyVariable>,
}

impl<R: Ring> UnivariatePolynomialRing<R> {
    pub fn new(coeff_ring: R, var_map: Arc<PolyVariable>) -> UnivariatePolynomialRing<R> {
        UnivariatePolynomialRing {
            ring: coeff_ring,
            variable: var_map,
        }
    }

    pub fn new_from_poly(poly: &UnivariatePolynomial<R>) -> UnivariatePolynomialRing<R> {
        UnivariatePolynomialRing {
            ring: poly.ring.clone(),
            variable: poly.variable.clone(),
        }
    }
}

impl<R: Ring> std::fmt::Display for UnivariatePolynomialRing<R> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl<R: Ring> Set for UnivariatePolynomialRing<R> {
    type Element = UnivariatePolynomial<R>;

    fn size(&self) -> Option<Integer> {
        None
    }
}

impl<R: Ring> RingOps<UnivariatePolynomial<R>> for UnivariatePolynomialRing<R> {
    fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        a + b
    }

    fn sub(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        a - b
    }

    fn mul(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        a * &b
    }

    fn add_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = std::mem::replace(a, b.zero()) + b;
    }

    fn sub_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = std::mem::replace(a, b.zero()) - b;
    }

    fn mul_assign(&self, a: &mut Self::Element, b: Self::Element) {
        *a = std::mem::replace(a, b.zero()) * &b;
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        *a = std::mem::replace(a, b.zero()) + b * &c
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        *a = std::mem::replace(a, b.zero()) - b * &c
    }

    fn neg(&self, a: Self::Element) -> Self::Element {
        a.neg()
    }
}

impl<R: Ring> RingOps<&UnivariatePolynomial<R>> for UnivariatePolynomialRing<R> {
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a + b
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a - b
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
    }

    fn add_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = std::mem::replace(a, b.zero()) + b.clone();
    }

    fn sub_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = std::mem::replace(a, b.zero()) - b.clone();
    }

    fn mul_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = std::mem::replace(a, b.zero()) * b;
    }

    fn add_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = std::mem::replace(a, b.zero()) + b * c
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        *a = std::mem::replace(a, b.zero()) - b * c
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        a.clone().neg()
    }
}

impl<R: Ring> Ring for UnivariatePolynomialRing<R> {
    fn zero(&self) -> Self::Element {
        UnivariatePolynomial::new(&self.ring, None, self.variable.clone())
    }

    fn one(&self) -> Self::Element {
        self.zero().one()
    }

    fn nth(&self, n: Integer) -> Self::Element {
        self.zero().constant(self.ring.nth(n))
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        b.pow(e as usize)
    }

    fn is_zero(&self, a: &Self::Element) -> bool {
        a.is_zero()
    }

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
        self.ring.has_independent_elements()
    }
}

impl<R: EuclideanDomain> EuclideanDomain for UnivariatePolynomialRing<R> {
    fn rem(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.rem(b)
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        a.quot_rem(b)
    }

    fn gcd(&self, _a: &Self::Element, _b: &Self::Element) -> Self::Element {
        todo!("Implement univariate GCD for non-fields")
    }
}

/// A dense univariate polynomial.
#[derive(Clone)]
pub struct UnivariatePolynomial<F: Ring> {
    pub coefficients: Vec<F::Element>,
    pub variable: Arc<PolyVariable>,
    pub ring: F,
}

impl<R: Ring> InternalOrdering for UnivariatePolynomial<R> {
    /// An ordering of polynomials that has no intuitive meaning.
    fn internal_cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.coefficients.internal_cmp(&other.coefficients)
    }
}

impl<F: Ring + std::fmt::Debug> std::fmt::Debug for UnivariatePolynomial<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.is_zero() {
            return write!(f, "[]");
        }
        let mut first = true;
        write!(f, "[ ")?;
        for c in self.coefficients.iter() {
            if first {
                first = false;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{{ {c:?} }}")?;
        }
        write!(f, " ]")
    }
}

impl<F: Ring + std::fmt::Display> std::fmt::Display for UnivariatePolynomial<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.format(&PrintOptions::from_fmt(f), PrintState::from_fmt(f), f)
            .map(|_| ())
    }
}

impl<F: Ring> UnivariatePolynomial<F> {
    /// Constructs a zero polynomial. Instead of using this constructor,
    /// prefer to create new polynomials from existing ones, so that the
    /// variable map and field are inherited.
    #[inline]
    pub fn new(field: &F, cap: Option<usize>, variable: Arc<PolyVariable>) -> Self {
        Self {
            coefficients: Vec::with_capacity(cap.unwrap_or(0)),
            ring: field.clone(),
            variable,
        }
    }

    /// Constructs a zero polynomial, inheriting the field and variable map from `self`.
    #[inline]
    pub fn zero(&self) -> Self {
        Self {
            coefficients: vec![],
            ring: self.ring.clone(),
            variable: self.variable.clone(),
        }
    }

    /// Constructs a zero polynomial with the given number of variables and capacity,
    /// inheriting the field and variable map from `self`.
    #[inline]
    pub fn zero_with_capacity(&self, cap: usize) -> Self {
        Self {
            coefficients: Vec::with_capacity(cap),
            ring: self.ring.clone(),
            variable: self.variable.clone(),
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
            ring: self.ring.clone(),
            variable: self.variable.clone(),
        }
    }

    /// Constructs a polynomial that is one, inheriting the field and variable map from `self`.
    #[inline]
    pub fn one(&self) -> Self {
        Self {
            coefficients: vec![self.ring.one()],
            ring: self.ring.clone(),
            variable: self.variable.clone(),
        }
    }

    /// Constructs a polynomial with a single term.
    #[inline]
    pub fn monomial(&self, coeff: F::Element, exponent: usize) -> Self {
        if self.ring.is_zero(&coeff) {
            return self.zero();
        }

        let mut coefficients = vec![self.ring.zero(); exponent + 1];
        coefficients[exponent] = coeff;

        Self {
            coefficients,
            ring: self.ring.clone(),
            variable: self.variable.clone(),
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        self.coefficients.len() == 1 && self.ring.is_one(&self.coefficients[0])
    }

    /// Returns true if the polynomial is constant.
    #[inline]
    pub fn is_constant(&self) -> bool {
        self.coefficients.len() <= 1
    }

    /// Get the constant term of the polynomial.
    #[inline]
    pub fn get_constant(&self) -> F::Element {
        if self.is_zero() {
            return self.ring.zero();
        }

        self.coefficients[0].clone()
    }

    /// Get a copy of the variable/
    pub fn get_vars(&self) -> Arc<PolyVariable> {
        self.variable.clone()
    }

    /// Get a reference to the variables
    pub fn get_vars_ref(&self) -> &PolyVariable {
        self.variable.as_ref()
    }

    /// Get the leading coefficient.
    pub fn lcoeff(&self) -> F::Element {
        self.coefficients
            .last()
            .unwrap_or(&self.ring.zero())
            .clone()
    }

    /// Get the degree of the polynomial.
    /// A zero polynomial has degree 0.
    pub fn degree(&self) -> usize {
        if self.is_zero() {
            return 0; // TODO: return None?
        }

        self.coefficients.len() - 1
    }

    /// Compute `self^pow`.
    pub fn pow(&self, mut pow: usize) -> Self {
        if pow == 0 {
            return self.one();
        }

        let mut x = self.clone();
        let mut y = self.one();
        while pow != 1 {
            if pow % 2 == 1 {
                y = &y * &x;
                pow -= 1;
            }

            x = &x * &x;
            pow /= 2;
        }

        x * &y
    }

    /// Multiply by a variable to the power of `exp`.
    pub fn mul_exp(&self, exp: usize) -> Self {
        if exp == 0 {
            return self.clone();
        }

        let mut a = self.zero();
        a.coefficients = vec![self.ring.zero(); self.degree() + exp + 1];

        for (cn, c) in a.coefficients.iter_mut().skip(exp).zip(&self.coefficients) {
            *cn = c.clone();
        }

        a
    }

    /// Divide by a variable to the power of `exp`.
    pub fn div_exp(&self, exp: usize) -> Self {
        if exp == 0 {
            return self.clone();
        }

        let mut a = self.zero();

        if self.degree() < exp {
            return a;
        }

        a.coefficients = vec![self.ring.zero(); self.degree() - exp + 1];

        for (cn, c) in a
            .coefficients
            .iter_mut()
            .zip(self.coefficients.iter().skip(exp))
        {
            *cn = c.clone();
        }

        a
    }

    /// Multiply by a coefficient `coeff`.
    pub fn mul_coeff(mut self, coeff: &F::Element) -> Self {
        for c in &mut self.coefficients {
            if !self.ring.is_zero(c) {
                self.ring.mul_assign(c, coeff);
            }
        }

        self
    }

    /// Map a coefficient using the function `f`.
    pub fn map_coeff<U: Ring, T: Fn(&F::Element) -> U::Element>(
        &self,
        f: T,
        field: U,
    ) -> UnivariatePolynomial<U> {
        let mut r = UnivariatePolynomial::new(&field, None, self.variable.clone());
        r.coefficients = self.coefficients.iter().map(f).collect::<Vec<_>>();
        r.truncate();
        r
    }

    pub(crate) fn truncate(&mut self) {
        let d = self
            .coefficients
            .iter_mut()
            .rev()
            .position(|c| !self.ring.is_zero(c))
            .unwrap_or(self.coefficients.len());

        self.coefficients.truncate(self.coefficients.len() - d);
    }

    /// Evaluate the polynomial, using Horner's method.
    pub fn evaluate(&self, x: &F::Element) -> F::Element {
        if self.is_constant() {
            return self.get_constant();
        }

        let mut res = self.coefficients.last().unwrap().clone();
        for c in self.coefficients.iter().rev().skip(1) {
            if !self.ring.is_zero(c) {
                res = self.ring.add(&self.ring.mul(&res, x), c);
            } else {
                self.ring.mul_assign(&mut res, x);
            }
        }

        res
    }

    /// Take the derivative of the polynomial.
    pub fn derivative(&self) -> Self {
        if self.is_constant() {
            return self.zero();
        }

        let mut res = self.zero();
        res.coefficients
            .resize(self.coefficients.len() - 1, self.ring.zero());

        for (p, (nc, oc)) in res
            .coefficients
            .iter_mut()
            .zip(self.coefficients.iter().skip(1))
            .enumerate()
        {
            if !self.ring.is_zero(oc) {
                *nc = self.ring.mul(oc, &self.ring.nth(Integer::from(p) + 1));
            }
        }

        res
    }

    /// Convert from a univariate polynomial to a multivariate polynomial.
    pub fn to_multivariate<E: PositiveExponent>(self) -> MultivariatePolynomial<F, E> {
        let mut res = MultivariatePolynomial::new(
            &self.ring,
            self.degree().into(),
            Arc::new(vec![self.variable.as_ref().clone()]),
        );

        for (p, c) in self.coefficients.into_iter().enumerate() {
            res.append_monomial(c, &[E::from_u32(p as u32)]);
        }

        res
    }

    /// Shift the variable `var` to `var+shift`.
    pub fn shift_var(&self, shift: &F::Element) -> Self {
        let d = self.degree();
        let mut poly = self.clone();

        // TODO: improve with caching
        for k in 0..d {
            for j in (k..d).rev() {
                let (s, c) = poly.coefficients.split_at_mut(j + 1);
                self.ring.add_mul_assign(&mut s[j], &c[0], shift);
            }
        }

        poly
    }

    pub fn try_div(&self, div: &UnivariatePolynomial<F>) -> Option<UnivariatePolynomial<F>> {
        if div.is_zero() {
            return None;
        }

        if self.is_zero() {
            return Some(self.clone());
        }

        if self.variable != div.variable {
            return None;
        }

        // check if the leading coefficients divide
        self.ring.try_div(&self.lcoeff(), &div.lcoeff())?;

        if self.degree() < div.degree() {
            return None;
        }

        if self.ring.characteristic().is_zero() {
            // test division of constant term (evaluation at x_i = 0)
            let c = div.get_constant();
            if !self.ring.is_zero(&c)
                && !self.ring.is_one(&c)
                && self.ring.try_div(&self.get_constant(), &c).is_some()
            {
                return None;
            }

            // test division at x_i = 1
            let mut num = self.ring.zero();
            for c in &self.coefficients {
                if !self.ring.is_zero(c) {
                    self.ring.add_assign(&mut num, c);
                }
            }
            let mut den = self.ring.zero();
            for c in &div.coefficients {
                if !self.ring.is_zero(c) {
                    self.ring.add_assign(&mut den, c);
                }
            }

            if !self.ring.is_zero(&den)
                && !self.ring.is_one(&den)
                && self.ring.try_div(&num, &den).is_none()
            {
                return None;
            }
        }

        let (a, b) = self.quot_rem_impl(div, true);
        if b.is_zero() { Some(a) } else { None }
    }

    fn quot_rem_impl(&self, div: &Self, early_return: bool) -> (Self, Self) {
        if div.is_zero() {
            panic!("Cannot divide by 0");
        }

        if self.is_zero() {
            return (self.clone(), self.clone());
        }

        if self.variable != div.variable {
            panic!("Cannot divide with different variables");
        }

        let mut n = self.degree();
        let m = div.degree();

        if n < m {
            return (self.zero(), self.clone());
        }

        let mut q = self.zero();
        q.coefficients = vec![self.ring.zero(); n + 1 - m];

        let mut r = self.clone();

        while n >= m {
            if let Some(qq) = self.ring.try_div(&r.coefficients[n], &div.coefficients[m]) {
                r = r - div.mul_exp(n - m).mul_coeff(&qq);
                q.coefficients[n - m] = qq;
            } else if early_return {
                return (self.zero(), r);
            } else {
                break;
            }

            if r.is_zero() {
                break;
            }

            n = r.degree();
        }

        q.truncate();

        (q, r)
    }
}

impl<F: Ring> SelfRing for UnivariatePolynomial<F> {
    fn is_zero(&self) -> bool {
        self.is_zero()
    }

    fn is_one(&self) -> bool {
        self.is_one()
    }

    fn format<W: std::fmt::Write>(
        &self,
        opts: &PrintOptions,
        mut state: PrintState,
        f: &mut W,
    ) -> Result<bool, std::fmt::Error> {
        if self.is_constant() {
            if self.is_zero() {
                if state.in_sum {
                    f.write_str("+")?;
                }
                f.write_char('0')?;
                return Ok(false);
            } else {
                return self.ring.format(&self.coefficients[0], opts, state, f);
            }
        }

        let non_zero = self
            .coefficients
            .iter()
            .filter(|c| !self.ring.is_zero(c))
            .count();

        let add_paren = non_zero > 1 && state.in_product
            || ((state.in_exp || state.in_exp_base)
                && (non_zero > 1
                    || self
                        .coefficients
                        .iter()
                        .filter(|c| !self.ring.is_one(c))
                        .count()
                        > 0));

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

        let v = self.variable.format_string(
            opts,
            PrintState {
                in_exp: true,
                ..state
            },
        );

        for (e, c) in self.coefficients.iter().enumerate() {
            state.suppress_one = e > 0;

            if self.ring.is_zero(c) {
                continue;
            }

            let suppressed_one = self.ring.format(
                c,
                opts,
                state.step(state.in_sum, state.in_product, false, false),
                f,
            )?;

            if !suppressed_one && e > 0 {
                f.write_char(opts.multiplication_operator)?;
            }

            if e == 1 {
                write!(f, "{v}")?;
            } else if e > 1 {
                write!(f, "{v}^{e}")?;
            }

            state.in_sum = true;
            state.in_product = true;
        }

        if self.is_zero() {
            f.write_char('0')?;
        }

        if add_paren {
            f.write_str(")")?;
        }

        Ok(false)
    }
}

type ExactComplexField = FloatField<Complex<Rational>>;
pub type ExactComplexPolynomial = UnivariatePolynomial<FloatField<Complex<Rational>>>;

/// An exact rational disk in the complex plane.
#[derive(Clone, Debug)]
pub struct ComplexDisk {
    center: Complex<Rational>,
    radius: Rational,
}

/// Proven location of an isolated root relative to the coordinate axes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RootLocation {
    Complex,
    Real,
    Imaginary,
    Zero,
}

/// An axis of the complex plane.
#[derive(Clone, Copy, Eq, PartialEq)]
enum CoordinateAxis {
    Real,
    Imaginary,
}

/// An isolated root in the complex plane, together with its exact defining
/// polynomial, certified enclosure, and any proven axis location.
#[derive(Clone, Debug)]
pub struct IsolatedRoot {
    poly: Arc<ExactComplexPolynomial>,
    index: usize,
    enclosure: ComplexDisk,
    location: Option<RootLocation>,
}

#[derive(Clone, Debug)]
struct RootState {
    enclosure: ComplexDisk,
    location: Option<RootLocation>,
}

#[derive(Clone)]
struct RootIdentity {
    poly: Arc<ExactComplexPolynomial>,
    index: usize,
}

#[derive(Clone)]
struct PolynomialRootLayout {
    roots: Vec<(RootIdentity, usize)>,
}

#[derive(Clone)]
struct RealProjection {
    poly: Arc<UnivariatePolynomial<Q>>,
    intervals: Vec<(Rational, Rational)>,
}

struct ProjectedRealRoot {
    poly: Arc<UnivariatePolynomial<Q>>,
    interval: (Rational, Rational),
}

#[derive(Clone, Eq, Hash, PartialEq)]
enum RootPolynomialKey {
    Rational(Vec<Rational>),
    Complex(Vec<Complex<Rational>>),
}

/// A cache for storing root information for univariate polynomials.
pub struct RootCache {
    roots: RwLock<HashMap<RootPolynomialKey, Arc<RootCacheEntry>>>,
    layouts: RwLock<HashMap<RootPolynomialKey, Arc<RootLayoutEntry>>>,
}

struct RootCacheEntry {
    roots: OnceLock<RwLock<Vec<RootState>>>,
}

struct RootLayoutEntry {
    layout: OnceLock<PolynomialRootLayout>,
}

impl RootCacheEntry {
    fn new() -> Self {
        Self {
            roots: OnceLock::new(),
        }
    }
}

impl RootLayoutEntry {
    fn new() -> Self {
        Self {
            layout: OnceLock::new(),
        }
    }
}

impl RootCache {
    fn new() -> Self {
        Self {
            roots: RwLock::new(HashMap::new()),
            layouts: RwLock::new(HashMap::new()),
        }
    }

    fn root_entry(&self, key: RootPolynomialKey) -> Arc<RootCacheEntry> {
        if let Some(entry) = self.roots.read().unwrap().get(&key).cloned() {
            return entry;
        }

        self.roots
            .write()
            .unwrap()
            .entry(key)
            .or_insert_with(|| Arc::new(RootCacheEntry::new()))
            .clone()
    }

    fn layout_entry(&self, key: RootPolynomialKey) -> Arc<RootLayoutEntry> {
        if let Some(entry) = self.layouts.read().unwrap().get(&key).cloned() {
            return entry;
        }

        self.layouts
            .write()
            .unwrap()
            .entry(key)
            .or_insert_with(|| Arc::new(RootLayoutEntry::new()))
            .clone()
    }

    fn states_from_roots(mut roots: Vec<IsolatedRoot>) -> Vec<RootState> {
        UnivariatePolynomial::<Q>::sort_complex_roots_canonical(&mut roots);
        roots
            .into_iter()
            .map(|root| RootState {
                enclosure: root.enclosure,
                location: root.location,
            })
            .collect()
    }

    fn compute_defining_roots(
        poly: &ExactComplexPolynomial,
        target_radius: Option<&Rational>,
    ) -> Vec<IsolatedRoot> {
        if let Some(rational) = poly.try_map_to_rational() {
            rational.isolate_roots_impl(target_radius)
        } else {
            poly.isolate_roots_impl(target_radius)
        }
    }

    fn defining_entry(&self, poly: &ExactComplexPolynomial) -> Arc<RootCacheEntry> {
        let key = if let Some(rational) = poly.try_map_to_rational() {
            RootPolynomialKey::Rational(rational.coefficients)
        } else {
            RootPolynomialKey::Complex(poly.coefficients.clone())
        };
        let entry = self.root_entry(key);
        entry.roots.get_or_init(|| {
            RwLock::new(Self::states_from_roots(Self::compute_defining_roots(
                poly, None,
            )))
        });
        entry
    }

    fn materialize_root(&self, poly: Arc<ExactComplexPolynomial>, index: usize) -> IsolatedRoot {
        let entry = self.defining_entry(&poly);
        let roots = entry.roots.get().unwrap().read().unwrap();
        let state = roots
            .get(index)
            .unwrap_or_else(|| panic!("root index {index} is out of bounds for {poly}"));
        IsolatedRoot {
            poly,
            index,
            enclosure: state.enclosure.clone(),
            location: state.location,
        }
    }

    fn defining_roots(&self, poly: Arc<ExactComplexPolynomial>) -> Vec<IsolatedRoot> {
        let entry = self.defining_entry(&poly);
        let states = entry.roots.get().unwrap().read().unwrap();
        states
            .iter()
            .enumerate()
            .map(|(index, state)| IsolatedRoot {
                poly: poly.clone(),
                index,
                enclosure: state.enclosure.clone(),
                location: state.location,
            })
            .collect()
    }

    fn layout_from_defining_factors(
        &self,
        factors: impl IntoIterator<Item = (Arc<ExactComplexPolynomial>, usize)>,
    ) -> PolynomialRootLayout {
        let mut roots = Vec::new();
        let mut multiplicities = HashMap::new();
        for (poly, multiplicity) in factors {
            multiplicities.insert(poly.coefficients.clone(), multiplicity);
            roots.extend(self.defining_roots(poly));
        }

        UnivariatePolynomial::<Q>::refine_isolated_complex_roots_until_disjoint(&mut roots);
        UnivariatePolynomial::<Q>::sort_complex_roots_canonical(&mut roots);
        for root in &roots {
            self.merge_root(root);
        }

        PolynomialRootLayout {
            roots: roots
                .into_iter()
                .map(|root| {
                    let multiplicity = multiplicities[&root.poly.coefficients];
                    (
                        RootIdentity {
                            poly: root.poly,
                            index: root.index,
                        },
                        multiplicity,
                    )
                })
                .collect(),
        }
    }

    fn refine_root(&self, root: &mut IsolatedRoot, tolerance: &Rational) {
        let entry = self.defining_entry(&root.poly);
        let mut states = entry.roots.get().unwrap().write().unwrap();
        let state = states.get(root.index).unwrap_or_else(|| {
            panic!(
                "root index {} is out of bounds for {}",
                root.index, root.poly
            )
        });
        root.enclosure = state.enclosure.clone();
        root.location = state.location;

        if !tolerance.is_zero()
            && root.enclosure.radius > *tolerance
            && !UnivariatePolynomial::<Q>::refine_isolated_complex_root_to_tolerance(
                root, tolerance,
            )
        {
            let replacement =
                Self::states_from_roots(Self::compute_defining_roots(&root.poly, Some(tolerance)));
            *states = replacement;
            let state = states.get(root.index).unwrap_or_else(|| {
                panic!(
                    "root index {} disappeared while refining {}",
                    root.index, root.poly
                )
            });
            root.enclosure = state.enclosure.clone();
            root.location = state.location;
            return;
        }

        states[root.index] = RootState {
            enclosure: root.enclosure.clone(),
            location: root.location,
        };
    }

    fn merge_root(&self, root: &IsolatedRoot) {
        let entry = self.defining_entry(&root.poly);
        let mut states = entry.roots.get().unwrap().write().unwrap();
        let state = states.get_mut(root.index).unwrap_or_else(|| {
            panic!(
                "root index {} is out of bounds for {}",
                root.index, root.poly
            )
        });
        if root.enclosure.radius < state.enclosure.radius {
            state.enclosure = root.enclosure.clone();
        }
        if state.location.is_none() {
            state.location = root.location;
        }
    }

    fn resolve_root_location(&self, root: &mut IsolatedRoot) {
        let entry = self.defining_entry(&root.poly);
        let mut roots = {
            let states = entry.roots.get().unwrap().read().unwrap();
            let state = states.get(root.index).unwrap_or_else(|| {
                panic!(
                    "root index {} is out of bounds for {}",
                    root.index, root.poly
                )
            });
            if state.location.is_some() {
                root.enclosure = state.enclosure.clone();
                root.location = state.location;
                return;
            }

            states
                .iter()
                .enumerate()
                .map(|(index, state)| IsolatedRoot {
                    poly: root.poly.clone(),
                    index,
                    enclosure: state.enclosure.clone(),
                    location: state.location,
                })
                .collect::<Vec<_>>()
        };

        root.poly.classify_complex_root_locations(&mut roots);

        let mut states = entry.roots.get().unwrap().write().unwrap();
        for resolved in roots {
            let state = &mut states[resolved.index];
            if state.location.is_none() {
                state.location = resolved.location;
            }
        }
        let state = &states[root.index];
        root.enclosure = state.enclosure.clone();
        root.location = state.location;
    }

    fn materialize_layout(&self, layout: &PolynomialRootLayout) -> Vec<(IsolatedRoot, usize)> {
        layout
            .roots
            .iter()
            .map(|(identity, multiplicity)| {
                let root = self.materialize_root(identity.poly.clone(), identity.index);
                (root, *multiplicity)
            })
            .collect()
    }

    fn get_index(&self, layout: &PolynomialRootLayout, index: usize) -> Option<IsolatedRoot> {
        let mut seen = 0;
        for (identity, multiplicity) in &layout.roots {
            if index < seen + multiplicity {
                return Some(self.materialize_root(identity.poly.clone(), identity.index));
            }
            seen += *multiplicity;
        }
        None
    }
}

static ROOT_CACHE: LazyLock<RootCache> = LazyLock::new(RootCache::new);

fn retain_real_roots(roots: Vec<(IsolatedRoot, usize)>) -> Vec<(IsolatedRoot, usize)> {
    roots
        .into_iter()
        .filter_map(|(mut root, multiplicity)| {
            let location = root.location();
            matches!(location, RootLocation::Real | RootLocation::Zero)
                .then_some((root, multiplicity))
        })
        .collect()
}

impl ComplexDisk {
    fn norm_upper_bound(z: &Complex<Rational>) -> Rational {
        z.re.abs() + z.im.abs()
    }

    fn norm_lower_bound(z: &Complex<Rational>) -> Rational {
        z.re.abs().max(z.im.abs())
    }

    pub fn is_disjoint(&self, other: &Self) -> bool {
        &self.radius + &other.radius < Self::norm_lower_bound(&(&self.center - &other.center))
    }

    pub fn center(&self) -> &Complex<Rational> {
        &self.center
    }

    pub fn radius(&self) -> &Rational {
        &self.radius
    }

    /// Convert this exact rational disk to a certified rectangular
    /// floating-point enclosure.
    pub fn to_ball(&self, precision: u32) -> ComplexBall {
        ComplexBall::from_rational_ball(&self.center, &self.radius, precision)
    }
}

impl RootLocation {
    fn with_axis(location: Option<Self>, axis: CoordinateAxis) -> Self {
        match (location, axis) {
            (Some(Self::Imaginary | Self::Zero), CoordinateAxis::Real)
            | (Some(Self::Real | Self::Zero), CoordinateAxis::Imaginary) => Self::Zero,
            (_, CoordinateAxis::Real) => Self::Real,
            (_, CoordinateAxis::Imaginary) => Self::Imaginary,
        }
    }
}

impl CoordinateAxis {
    fn contains_interval(self, interval: &(Rational, Rational), root: &IsolatedRoot) -> bool {
        let (center, distance_to_axis) = match self {
            Self::Real => (&root.enclosure.center.re, root.enclosure.center.im.abs()),
            Self::Imaginary => (&root.enclosure.center.im, root.enclosure.center.re.abs()),
        };
        let lower_distance = (&interval.0 - center).abs() + &distance_to_axis;
        let upper_distance = (&interval.1 - center).abs() + &distance_to_axis;
        if lower_distance <= root.enclosure.radius && upper_distance <= root.enclosure.radius {
            return true;
        }

        self == Self::Imaginary
            && distance_to_axis <= root.enclosure.radius
            && interval.0 <= center.clone() - &root.enclosure.radius
            && center.clone() + &root.enclosure.radius <= interval.1
    }
}

impl IsolatedRoot {
    fn absolute_tolerance(binary_precision: u32) -> Rational {
        Rational::from((
            Integer::one(),
            Integer::from(2).pow(binary_precision as u64),
        ))
    }

    /// Returns a reference to the polynomial defining this root.
    pub fn defining_polynomial(&self) -> &ExactComplexPolynomial {
        &self.poly
    }

    /// Returns the complex disk enclosure of this root.
    pub fn enclosure(&self) -> &ComplexDisk {
        &self.enclosure
    }

    /// Canonical index of this root within its defining polynomial.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Convert this root to the canonical expression-level `root` representation.
    pub fn to_atom(&self) -> Atom {
        let mut polynomial = self.poly.as_ref().clone().to_multivariate::<u16>();
        let variable = polynomial.get_vars_ref()[0].clone();
        let canonical_variable = PolyVariable::Symbol(root_var());
        if variable != canonical_variable {
            polynomial.rename_variable(&variable, &canonical_variable);
        }
        root().call((polynomial.to_expression(), self.index))
    }

    /// Refine this root to an absolute radius, update its cached enclosure,
    /// and return the updated root.
    pub fn refine(mut self, tolerance: &Rational) -> Self {
        ROOT_CACHE.refine_root(&mut self, tolerance);
        self
    }

    /// Determine whether this root lies on the positive real axis. Complex,
    /// imaginary, and zero roots are not positive.
    pub fn is_positive(&mut self) -> bool {
        if self.location() != RootLocation::Real {
            return false;
        }

        let mut binary_precision = 32u32;
        loop {
            if &self.enclosure.center.re - &self.enclosure.radius > Rational::zero() {
                return true;
            }
            if &self.enclosure.center.re + &self.enclosure.radius < Rational::zero() {
                return false;
            }

            let tolerance = Self::absolute_tolerance(binary_precision);
            ROOT_CACHE.refine_root(self, &tolerance);
            binary_precision = binary_precision.saturating_mul(2);
        }
    }

    /// Resolve this root's exact relationship to the coordinate axes and
    /// update both this root and its defining-root cache entry.
    pub fn location(&mut self) -> RootLocation {
        if self.location.is_none() {
            ROOT_CACHE.resolve_root_location(self);
        }
        self.location
            .expect("root location resolution must classify every root")
    }

    pub(crate) fn to_float_center(&self, binary_prec: u32) -> Complex<Float> {
        let mut center = Complex::new(
            self.enclosure.center.re.to_multi_prec_float(binary_prec),
            self.enclosure.center.im.to_multi_prec_float(binary_prec),
        );

        let field = FloatField::from_rep(Complex::new(
            Float::with_val(binary_prec, 1),
            Float::new(binary_prec),
        ));
        let poly = self.poly.map_coeff(
            |c| {
                Complex::new(
                    c.re.to_multi_prec_float(binary_prec),
                    c.im.to_multi_prec_float(binary_prec),
                )
            },
            field,
        );
        let derivative = poly.derivative();
        let tolerance = Rational::from((Integer::one(), Integer::from(2).pow(binary_prec as u64)))
            .to_multi_prec_float(binary_prec);
        let tolerance_squared = tolerance.clone() * tolerance;

        for _ in 0..32 {
            let derivative_at_center = derivative.evaluate(&center);
            if SingleFloat::is_zero(&derivative_at_center) {
                break;
            }

            let correction = poly.evaluate(&center) / derivative_at_center;
            if !correction.is_finite() {
                break;
            }

            center -= correction.clone();
            if correction.norm_squared() < tolerance_squared {
                break;
            }
        }

        center
    }
}

#[derive(Clone, Copy)]
pub(crate) enum RootImage<'a> {
    Identity,
    RationalPolynomial(&'a [Rational]),
}

impl RootImage<'_> {
    fn evaluate(self, root: &IsolatedRoot, precision: u32) -> ComplexBall {
        let root_ball = root.enclosure().to_ball(precision);
        let Self::RationalPolynomial(coefficients) = self else {
            return root_ball;
        };

        let zero = RealBall::exact(Float::new(precision));
        let mut value = ComplexBall::new(zero.clone(), zero);
        for coefficient in coefficients.iter().rev() {
            let coefficient = RealBall::from_rational_bounds(coefficient, coefficient, precision);
            value = value * &root_ball + coefficient;
        }
        value
    }
}

pub(crate) fn match_certified_root_images(
    target: &IsolatedRoot,
    target_image: RootImage<'_>,
    candidates: &mut [IsolatedRoot],
    candidate_image: RootImage<'_>,
    expected_count: usize,
) -> Result<Vec<usize>, String> {
    let mut target = target.clone();
    let mut binary_precision = 32u32;
    for _ in 0..10 {
        let target_ball = target_image.evaluate(&target, binary_precision);
        let matches = candidates
            .iter()
            .enumerate()
            .filter_map(|(index, candidate)| {
                let value = candidate_image.evaluate(candidate, binary_precision);
                (!value.is_disjoint(&target_ball)).then_some(index)
            })
            .collect::<Vec<_>>();

        if matches.len() == expected_count {
            return Ok(matches);
        }

        let tolerance = IsolatedRoot::absolute_tolerance(binary_precision);
        target = target.refine(&tolerance);
        let candidates_to_refine: Vec<_> = if matches.is_empty() {
            (0..candidates.len()).collect()
        } else {
            matches
        };
        for index in candidates_to_refine {
            candidates[index] = candidates[index].clone().refine(&tolerance);
        }
        binary_precision = binary_precision.saturating_mul(2);
    }

    Err(format!(
        "could not identify {expected_count} matching certified root image(s)"
    ))
}

pub(crate) fn rational_polynomial_value_has_positive_real_part(
    root: &IsolatedRoot,
    coefficients: &[Rational],
) -> Result<bool, String> {
    let mut root = root.clone();
    let mut binary_precision = 32u32;
    for _ in 0..10 {
        let value = RootImage::RationalPolynomial(coefficients).evaluate(&root, binary_precision);
        if value.re.is_strictly_positive() {
            return Ok(true);
        }
        if value.re.is_strictly_negative() {
            return Ok(false);
        }

        root = root.refine(&IsolatedRoot::absolute_tolerance(binary_precision));
        binary_precision = binary_precision.saturating_mul(2);
    }

    Err("could not determine the sign of a polynomial value at an isolated root".to_string())
}

impl UnivariatePolynomial<RationalField> {
    fn complex_root_tolerance(num_prec: u32) -> Float {
        let bits = num_prec.saturating_sub(8).max(1);
        Rational::from((Integer::one(), Integer::from(2).pow(bits as u64)))
            .to_multi_prec_float(num_prec)
    }

    fn certify_one_complex_root_disk(
        poly: &UnivariatePolynomial<FloatField<Complex<Rational>>>,
        center: &Complex<Rational>,
        radius: &Rational,
    ) -> bool {
        if radius.is_zero() {
            return false;
        }

        let shifted = poly.shift_var(center);
        let Some(linear) = shifted.coefficients.get(1) else {
            return false;
        };

        let mut eval_higher_powers = Rational::zero();
        for (pow, c) in shifted.coefficients.iter().enumerate().skip(2) {
            eval_higher_powers += radius.pow(pow as u64) * ComplexDisk::norm_upper_bound(c);
        }

        let first_lower_bound = ComplexDisk::norm_lower_bound(linear) * radius;
        let const_upper = shifted
            .coefficients
            .first()
            .map(ComplexDisk::norm_upper_bound)
            .unwrap_or_else(Rational::zero);

        first_lower_bound > const_upper + eval_higher_powers
    }

    fn complex_root_radius(
        centers: &[Complex<Rational>],
        root_index: usize,
        target_radius: Option<&Rational>,
    ) -> Option<Rational> {
        let mut radius = None;

        for (other_index, other_center) in centers.iter().enumerate() {
            if root_index == other_index {
                continue;
            }

            let distance = ComplexDisk::norm_lower_bound(&(&centers[root_index] - other_center));
            radius = Some(match radius {
                Some(r) if r < distance => r,
                _ => distance,
            });
        }

        let mut radius = radius
            .map(|r| r / Rational::from(4))
            .or_else(|| target_radius.cloned())
            .unwrap_or_else(Rational::one);

        if let Some(target_radius) = target_radius {
            if !target_radius.is_zero() && target_radius < &radius {
                radius = target_radius.clone();
            }
        }

        if radius.is_zero() { None } else { Some(radius) }
    }

    /// Gets the `index`-th root of the polynomial. Fails when `index` is out of bounds.
    pub fn root(&self, index: usize) -> Option<IsolatedRoot> {
        if index >= self.degree() {
            return None;
        }

        let entry = ROOT_CACHE.layout_entry(RootPolynomialKey::Rational(self.coefficients.clone()));
        let layout = entry.layout.get_or_init(|| self.compute_root_layout());
        ROOT_CACHE.get_index(layout, index)
    }

    /// Isolate the distinct complex roots of the polynomial. The result contains
    /// canonically sorted `(root, multiplicity)` pairs. Every root enclosure is a
    /// rational ball containing exactly one root of its defining polynomial.
    pub fn isolate_roots(&self) -> Vec<(IsolatedRoot, usize)> {
        let entry = ROOT_CACHE.layout_entry(RootPolynomialKey::Rational(self.coefficients.clone()));
        let layout = entry.layout.get_or_init(|| self.compute_root_layout());
        ROOT_CACHE.materialize_layout(layout)
    }

    /// Isolate the distinct real roots of the polynomial. Resolving whether a
    /// root lies on the real axis may refine its cached enclosure.
    pub fn isolate_real_roots(&self) -> Vec<(IsolatedRoot, usize)> {
        retain_real_roots(self.isolate_roots())
    }

    fn compute_root_layout(&self) -> PolynomialRootLayout {
        let complex_field = FloatField::from_rep(Complex::from(Rational::one()));
        let factors = self
            .clone()
            .to_multivariate::<u16>()
            .factor()
            .into_iter()
            .filter(|(factor, _)| !factor.is_constant())
            .map(|(factor, multiplicity)| {
                let defining_poly = Arc::new(factor.to_univariate_from_univariate(0).map_coeff(
                    |coefficient| Complex::from(coefficient.clone()),
                    complex_field.clone(),
                ));
                (defining_poly, multiplicity)
            });
        ROOT_CACHE.layout_from_defining_factors(factors)
    }

    fn sort_complex_roots_canonical(roots: &mut [IsolatedRoot]) {
        // Establishing the order may strengthen the supplied root snapshots.
        // Cache owners must retain those stronger certificates after this call.
        Self::refine_real_projections_for_sort(roots);
        let known_equal_real_parts = Self::known_equal_real_parts(roots);
        let needs_projection =
            Self::roots_with_unresolved_real_projection_overlaps(roots, &known_equal_real_parts);

        let mut projection_cache = HashMap::new();
        let projected_roots = roots
            .iter_mut()
            .zip(needs_projection)
            .map(|(root, needs_projection)| {
                if !needs_projection {
                    return None;
                }

                let poly = &root.poly;
                let projection = projection_cache
                    .entry(poly.coefficients.clone())
                    .or_insert_with(|| {
                        Self::complex_real_projection_polynomial(poly).map(|projection| {
                            let intervals = projection
                                .isolate_real_root_intervals()
                                .into_iter()
                                .map(|(lower, upper, _)| (lower, upper))
                                .collect();
                            RealProjection {
                                poly: Arc::new(projection),
                                intervals,
                            }
                        })
                    })
                    .as_ref()?;

                Self::projected_real_root_for_complex_root(root, projection)
            })
            .collect::<Vec<_>>();

        let mut order = (0..roots.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            Self::cmp_complex_roots_canonical_with_projected(
                &roots[a],
                &roots[b],
                projected_roots[a].as_ref(),
                projected_roots[b].as_ref(),
                known_equal_real_parts[a * roots.len() + b],
            )
        });

        let sorted = order
            .into_iter()
            .map(|index| roots[index].clone())
            .collect::<Vec<_>>();
        roots.clone_from_slice(&sorted);
    }

    fn refine_real_projections_for_sort(roots: &mut [IsolatedRoot]) {
        // Use bounded-size approximate centers here. Repeated exact rational
        // Newton steps make numerator and denominator sizes grow explosively.
        // Most comparisons resolve at the first target; persistent overlaps
        // are handled exactly by the projection polynomial below.
        for target_radius_bits in [32, 64] {
            let known_equal_real_parts = Self::known_equal_real_parts(roots);
            let overlaps = Self::roots_with_unresolved_real_projection_overlaps(
                roots,
                &known_equal_real_parts,
            );
            if !overlaps.iter().any(|overlaps| *overlaps) {
                return;
            }

            let mut refined_any = false;
            for (root, overlaps) in roots.iter_mut().zip(overlaps) {
                if !overlaps {
                    continue;
                }
                let poly = (*root.poly).clone();
                refined_any |= Self::bounded_refine_complex_root_disk_for_sort(
                    &poly,
                    root,
                    target_radius_bits + 32,
                    target_radius_bits as u64,
                );
            }
            if !refined_any {
                return;
            }
        }
    }

    fn bounded_refine_complex_root_disk_for_sort(
        poly: &ExactComplexPolynomial,
        root: &mut IsolatedRoot,
        binary_precision: u32,
        target_radius_bits: u64,
    ) -> bool {
        let approximate_center = root.to_float_center(binary_precision);
        if !approximate_center.is_finite() {
            return false;
        }
        let center = Complex::new(
            approximate_center.re.to_rational(),
            approximate_center.im.to_rational(),
        );
        let center_distance = ComplexDisk::norm_upper_bound(&(&center - &root.enclosure.center));

        let max_radius = &root.enclosure.radius / &Rational::from(2);
        let mut radius = Rational::from((Integer::one(), Integer::from(2).pow(target_radius_bits)))
            .min(max_radius.clone());

        for _ in 0..64 {
            if &center_distance + &radius <= root.enclosure.radius
                && Self::certify_one_complex_root_disk(poly, &center, &radius)
            {
                root.enclosure.center = center;
                root.enclosure.radius = radius;
                return true;
            }

            if radius >= max_radius {
                return false;
            }
            radius = (radius * Rational::from(2)).min(max_radius.clone());
        }

        false
    }

    fn known_equal_real_parts(roots: &[IsolatedRoot]) -> Vec<bool> {
        let mut equal = vec![false; roots.len() * roots.len()];
        for i in 0..roots.len() {
            equal[i * roots.len() + i] = true;

            let poly = &roots[i].poly;
            if poly
                .coefficients
                .iter()
                .any(|coefficient| !coefficient.im.is_zero())
            {
                continue;
            }

            // Conjugating a certified root ball of a real polynomial produces
            // another certified one-root ball. It cannot be certified disjoint
            // from the ball containing the conjugate root. Only use the match
            // when it is unique: this proves which isolated root is the
            // conjugate without requiring either independently refined ball to
            // contain the other.
            let mut conjugate = None;
            for (j, candidate) in roots.iter().enumerate() {
                let candidate_poly = &candidate.poly;
                if poly.coefficients != candidate_poly.coefficients {
                    continue;
                }

                let center_distance = ComplexDisk::norm_lower_bound(&Complex::new(
                    &roots[i].enclosure.center.re - &candidate.enclosure.center.re,
                    &roots[i].enclosure.center.im + &candidate.enclosure.center.im,
                ));
                if center_distance <= &roots[i].enclosure.radius + &candidate.enclosure.radius {
                    if conjugate.is_some() {
                        conjugate = None;
                        break;
                    }
                    conjugate = Some(j);
                }
            }

            if let Some(j) = conjugate {
                equal[i * roots.len() + j] = true;
                equal[j * roots.len() + i] = true;
            }
        }
        equal
    }

    fn roots_with_unresolved_real_projection_overlaps(
        roots: &[IsolatedRoot],
        known_equal_real_parts: &[bool],
    ) -> Vec<bool> {
        let mut overlaps = vec![false; roots.len()];
        for i in 0..roots.len() {
            let a_lower = &roots[i].enclosure.center.re - &roots[i].enclosure.radius;
            let a_upper = &roots[i].enclosure.center.re + &roots[i].enclosure.radius;

            for j in i + 1..roots.len() {
                if known_equal_real_parts[i * roots.len() + j] {
                    continue;
                }

                let b_lower = &roots[j].enclosure.center.re - &roots[j].enclosure.radius;
                let b_upper = &roots[j].enclosure.center.re + &roots[j].enclosure.radius;
                if a_lower <= b_upper && b_lower <= a_upper {
                    overlaps[i] = true;
                    overlaps[j] = true;
                }
            }
        }
        overlaps
    }

    #[cfg(test)]
    fn cmp_complex_roots_canonical(a: &IsolatedRoot, b: &IsolatedRoot) -> Ordering {
        let a_projected = Self::projected_real_root_for_complex_root_uncached(a);
        let b_projected = Self::projected_real_root_for_complex_root_uncached(b);
        Self::cmp_complex_roots_canonical_with_projected(
            a,
            b,
            a_projected.as_ref(),
            b_projected.as_ref(),
            false,
        )
    }

    fn cmp_complex_roots_canonical_with_projected(
        a: &IsolatedRoot,
        b: &IsolatedRoot,
        a_projected: Option<&ProjectedRealRoot>,
        b_projected: Option<&ProjectedRealRoot>,
        known_equal_real_parts: bool,
    ) -> Ordering {
        let a_re_upper = &a.enclosure.center.re + &a.enclosure.radius;
        let b_re_lower = &b.enclosure.center.re - &b.enclosure.radius;
        if a_re_upper < b_re_lower {
            return Ordering::Less;
        }

        let b_re_upper = &b.enclosure.center.re + &b.enclosure.radius;
        let a_re_lower = &a.enclosure.center.re - &a.enclosure.radius;
        if b_re_upper < a_re_lower {
            return Ordering::Greater;
        }

        if known_equal_real_parts {
            return Self::cmp_complex_roots_by_imaginary_part(a, b);
        }

        if let Some(ordering) =
            Self::cmp_complex_roots_by_projected_real_parts(a_projected, b_projected)
        {
            if ordering != Ordering::Equal {
                return ordering;
            }

            return Self::cmp_complex_roots_by_imaginary_part(a, b);
        }

        match a.enclosure.center.re.cmp(&b.enclosure.center.re) {
            Ordering::Equal => {}
            ordering => return ordering,
        }

        Self::cmp_complex_roots_by_imaginary_part(a, b)
    }

    fn cmp_complex_roots_by_imaginary_part(a: &IsolatedRoot, b: &IsolatedRoot) -> Ordering {
        let a_im_upper = &a.enclosure.center.im + &a.enclosure.radius;
        let b_im_lower = &b.enclosure.center.im - &b.enclosure.radius;
        if a_im_upper < b_im_lower {
            return Ordering::Less;
        }

        let b_im_upper = &b.enclosure.center.im + &b.enclosure.radius;
        let a_im_lower = &a.enclosure.center.im - &a.enclosure.radius;
        if b_im_upper < a_im_lower {
            return Ordering::Greater;
        }

        a.enclosure
            .center
            .im
            .cmp(&b.enclosure.center.im)
            .then_with(|| a.enclosure.radius.cmp(&b.enclosure.radius))
    }

    fn cmp_complex_roots_by_projected_real_parts(
        a: Option<&ProjectedRealRoot>,
        b: Option<&ProjectedRealRoot>,
    ) -> Option<Ordering> {
        let a = a?;
        let b = b?;

        if Arc::ptr_eq(&a.poly, &b.poly) {
            if a.interval == b.interval {
                return Some(Ordering::Equal);
            }

            if a.interval.1 < b.interval.0 {
                return Some(Ordering::Less);
            }

            if b.interval.1 < a.interval.0 {
                return Some(Ordering::Greater);
            }
        }

        Self::cmp_projected_real_roots(a, b)
    }

    #[cfg(test)]
    fn projected_real_root_for_complex_root_uncached(
        root: &IsolatedRoot,
    ) -> Option<ProjectedRealRoot> {
        let poly = &root.poly;
        let projection = Self::complex_real_projection_polynomial(poly)?;
        let intervals = projection
            .isolate_real_root_intervals()
            .into_iter()
            .map(|(lower, upper, _)| (lower, upper))
            .collect();
        let projection = RealProjection {
            poly: Arc::new(projection),
            intervals,
        };
        let mut root = root.clone();
        Self::projected_real_root_for_complex_root(&mut root, &projection)
    }

    fn projected_real_root_for_complex_root(
        root: &mut IsolatedRoot,
        projection: &RealProjection,
    ) -> Option<ProjectedRealRoot> {
        let mut intervals = projection.intervals.clone();

        for _ in 0..1024 {
            let root_interval = Self::complex_root_real_interval(root);
            let mut candidates = intervals
                .iter()
                .enumerate()
                .filter(|(_, interval)| {
                    Self::rational_intervals_intersect(interval, &root_interval)
                })
                .map(|(i, _)| i)
                .collect::<Vec<_>>();

            if candidates.len() == 1 {
                return Some(ProjectedRealRoot {
                    poly: projection.poly.clone(),
                    interval: intervals.swap_remove(candidates[0]),
                });
            }

            if candidates.is_empty() {
                candidates.extend(0..intervals.len());
            }

            let poly_complex = (*root.poly).clone();
            let derivative = poly_complex.derivative();
            let _ = Self::newton_refine_complex_root_disk(&poly_complex, &derivative, root);

            for i in candidates {
                projection
                    .poly
                    .refine_real_root_interval_once(&mut intervals[i]);
            }
        }

        None
    }

    fn complex_real_projection_polynomial(
        poly: &ExactComplexPolynomial,
    ) -> Option<UnivariatePolynomial<Q>> {
        let variables = Arc::new(vec![PolyVariable::Temporary(0), PolyVariable::Temporary(1)]);
        let mut real_part = MultivariatePolynomial::<Q, u16>::new(&Q, None, variables.clone());
        let mut imaginary_part = MultivariatePolynomial::<Q, u16>::new(&Q, None, variables);

        for (pow, coeff) in poly.coefficients.iter().enumerate() {
            for y_pow in 0..=pow {
                let x_pow = pow - y_pow;
                let binom = Self::binomial_rational(pow, y_pow);
                let rotated = Self::mul_complex_rational_by_i_power(coeff, y_pow);
                let exponents = [u16::try_from(x_pow).ok()?, u16::try_from(y_pow).ok()?];

                real_part.append_monomial(rotated.re * &binom, &exponents);
                imaginary_part.append_monomial(rotated.im * binom, &exponents);
            }
        }

        if real_part.is_zero() || imaginary_part.is_zero() {
            return None;
        }

        let rational_function_field = RationalPolynomialField::new(Z);
        let real_in_y = real_part.to_univariate(1).map_coeff(
            |c| RationalPolynomial::from_num_den(c.clone(), c.one(), &Z, false),
            rational_function_field.clone(),
        );
        let imaginary_in_y = imaginary_part.to_univariate(1).map_coeff(
            |c| RationalPolynomial::from_num_den(c.clone(), c.one(), &Z, false),
            rational_function_field,
        );

        let resultant = real_in_y.resultant(&imaginary_in_y);
        let mut projection = resultant
            .numerator
            .map_coeff(|c| c.to_rational(), Q)
            .to_univariate_from_univariate(0);
        projection.truncate();

        if projection.is_constant() {
            return None;
        }

        let derivative = projection.derivative();
        if !derivative.is_zero() {
            let repeated = projection.gcd(&derivative);
            if !repeated.is_constant() {
                projection = projection.quot_rem(&repeated).0;
                projection.truncate();
            }
        }

        Some(projection)
    }

    fn binomial_rational(n: usize, k: usize) -> Rational {
        let k = k.min(n - k);
        let mut result = Rational::one();

        for i in 0..k {
            result *= Rational::from(n - i);
            result /= Rational::from(i + 1);
        }

        result
    }

    fn mul_complex_rational_by_i_power(c: &Complex<Rational>, pow: usize) -> Complex<Rational> {
        match pow % 4 {
            0 => c.clone(),
            1 => Complex::new(-c.im.clone(), c.re.clone()),
            2 => Complex::new(-c.re.clone(), -c.im.clone()),
            _ => Complex::new(c.im.clone(), -c.re.clone()),
        }
    }

    fn complex_root_real_interval(root: &IsolatedRoot) -> (Rational, Rational) {
        (
            root.enclosure.center.re.clone() - &root.enclosure.radius,
            root.enclosure.center.re.clone() + &root.enclosure.radius,
        )
    }

    fn rational_intervals_intersect(a: &(Rational, Rational), b: &(Rational, Rational)) -> bool {
        a.0 <= b.1 && b.0 <= a.1
    }

    fn rational_interval_contains(a: &(Rational, Rational), b: &(Rational, Rational)) -> bool {
        a.0 <= b.0 && b.1 <= a.1
    }

    fn cmp_projected_real_roots(a: &ProjectedRealRoot, b: &ProjectedRealRoot) -> Option<Ordering> {
        // TODO: strip GCD first?
        let mut a_interval = a.interval.clone();
        let mut b_interval = b.interval.clone();
        let gcd = a.poly.gcd(&b.poly);
        let mut common_intervals = if gcd.is_constant() {
            vec![]
        } else {
            gcd.isolate_real_root_intervals()
                .into_iter()
                .map(|(lower, upper, _)| (lower, upper))
                .collect::<Vec<_>>()
        };

        for _ in 0..1024 {
            if a_interval.1 < b_interval.0 {
                return Some(Ordering::Less);
            }

            if b_interval.1 < a_interval.0 {
                return Some(Ordering::Greater);
            }

            if common_intervals.iter().any(|interval| {
                Self::rational_interval_contains(&a_interval, interval)
                    && Self::rational_interval_contains(&b_interval, interval)
            }) {
                return Some(Ordering::Equal);
            }

            a.poly.refine_real_root_interval_once(&mut a_interval);
            b.poly.refine_real_root_interval_once(&mut b_interval);
            for interval in &mut common_intervals {
                gcd.refine_real_root_interval_once(interval);
            }
        }

        None
    }

    fn complex_root_disks_are_disjoint(roots: &[IsolatedRoot]) -> bool {
        for i in 0..roots.len() {
            for j in i + 1..roots.len() {
                if !roots[i].enclosure.is_disjoint(&roots[j].enclosure) {
                    return false;
                }
            }
        }

        true
    }

    fn newton_refine_complex_root_disk(
        poly: &UnivariatePolynomial<FloatField<Complex<Rational>>>,
        derivative: &UnivariatePolynomial<FloatField<Complex<Rational>>>,
        root: &mut IsolatedRoot,
    ) -> bool {
        let derivative_at_center = derivative.evaluate(&root.enclosure.center);
        if derivative_at_center.is_zero() {
            return false;
        }

        let new_center = &root.enclosure.center
            - &(poly.evaluate(&root.enclosure.center) / derivative_at_center);
        let half_radius = root.enclosure.radius.clone() / Rational::from(2);
        let mut candidate_radii = vec![];

        let quadratic_radius = &root.enclosure.radius * &root.enclosure.radius;
        if !quadratic_radius.is_zero()
            && quadratic_radius < half_radius
            && candidate_radii
                .iter()
                .all(|radius| radius != &quadratic_radius)
        {
            candidate_radii.push(quadratic_radius);
        }
        candidate_radii.push(half_radius);

        for mut new_radius in candidate_radii {
            for _ in 0..16 {
                if Self::certify_one_complex_root_disk(poly, &new_center, &new_radius) {
                    root.enclosure.center = new_center;
                    root.enclosure.radius = new_radius;
                    return true;
                }

                new_radius *= Rational::from((1, 2));
                if new_radius.is_zero() {
                    break;
                }
            }
        }

        false
    }

    fn refine_complex_root_disks_until_disjoint(
        poly: &UnivariatePolynomial<FloatField<Complex<Rational>>>,
        derivative: &UnivariatePolynomial<FloatField<Complex<Rational>>>,
        roots: &mut [IsolatedRoot],
    ) -> bool {
        if Self::complex_root_disks_are_disjoint(roots) {
            return true;
        }

        for _ in 0..32 {
            for root in roots.iter_mut() {
                if !Self::newton_refine_complex_root_disk(poly, derivative, root) {
                    return false;
                }
            }

            if Self::complex_root_disks_are_disjoint(roots) {
                return true;
            }
        }

        false
    }

    fn refine_isolated_complex_roots_until_disjoint(roots: &mut [IsolatedRoot]) {
        for _ in 0..32 {
            if Self::complex_root_disks_are_disjoint(roots) {
                return;
            }

            for root in roots.iter_mut() {
                let poly_complex = (*root.poly).clone();

                let derivative = poly_complex.derivative();

                if !Self::newton_refine_complex_root_disk(&poly_complex, &derivative, root) {
                    return;
                }
            }
        }
    }

    fn refine_isolated_complex_root_to_tolerance(
        root: &mut IsolatedRoot,
        refine: &Rational,
    ) -> bool {
        if refine.is_zero() {
            return true;
        }

        if root.enclosure.radius <= *refine {
            return true;
        }

        // Repeated exact rational Newton steps can grow the center's
        // numerator and denominator exponentially. Obtain a bounded-size
        // numerical center and certify the requested rational disk exactly.
        // The old ball supplies the containment proof that it is the same
        // root.
        let integer_bits = |integer: Integer| match integer {
            Integer::Single(value) => i64::BITS - value.unsigned_abs().leading_zeros(),
            Integer::Double(value) => i128::BITS - value.unsigned_abs().leading_zeros(),
            Integer::Large(value) => value.significant_bits(),
        };
        let numerator_bits = integer_bits(refine.numerator());
        let denominator_bits = integer_bits(refine.denominator());
        let mut binary_precision = denominator_bits
            .saturating_sub(numerator_bits)
            .saturating_add(32)
            .max(64);
        let poly = (*root.poly).clone();
        for _ in 0..4 {
            let approximate_center = root.to_float_center(binary_precision);
            if approximate_center.is_finite() {
                let center = Complex::new(
                    approximate_center.re.to_rational(),
                    approximate_center.im.to_rational(),
                );
                let center_distance =
                    ComplexDisk::norm_upper_bound(&(&center - &root.enclosure.center));
                if &center_distance + refine <= root.enclosure.radius
                    && Self::certify_one_complex_root_disk(&poly, &center, refine)
                {
                    root.enclosure.center = center;
                    root.enclosure.radius = refine.clone();
                    return true;
                }
            }
            binary_precision = binary_precision.saturating_mul(2);
        }

        for _ in 0..64 {
            if root.enclosure.radius <= *refine {
                return true;
            }

            let poly_complex = (*root.poly).clone();

            let derivative = poly_complex.derivative();

            if !Self::newton_refine_complex_root_disk(&poly_complex, &derivative, root) {
                return false;
            }
        }

        root.enclosure.radius <= *refine
    }

    fn refine_real_root_interval_once(&self, interval: &mut (Rational, Rational)) {
        if interval.0 == interval.1 {
            return;
        }

        let left_value = self.evaluate(&interval.0);
        if left_value.is_zero() {
            interval.1 = interval.0.clone();
            return;
        }

        let right_value = self.evaluate(&interval.1);
        if right_value.is_zero() {
            interval.0 = interval.1.clone();
            return;
        }

        let left_is_negative = left_value.is_negative();
        let mid = (&interval.0 + &interval.1) / Rational::from(2);
        let mid_value = self.evaluate(&mid);
        if mid_value.is_zero() {
            interval.0 = mid.clone();
            interval.1 = mid;
        } else if mid_value.is_negative() == left_is_negative {
            interval.0 = mid;
        } else {
            interval.1 = mid;
        }
    }

    fn polynomial_at_imaginary_axis_parts(&self) -> (Self, Self) {
        let mut real = self.zero();
        let mut imaginary = self.zero();
        real.coefficients = vec![self.ring.zero(); self.coefficients.len()];
        imaginary.coefficients = vec![self.ring.zero(); self.coefficients.len()];

        for (pow, coeff) in self.coefficients.iter().enumerate() {
            if self.ring.is_zero(coeff) {
                continue;
            }

            let mut transformed = coeff.clone();
            if (pow / 2) % 2 == 1 {
                transformed = -transformed;
            }

            if pow % 2 == 0 {
                real.coefficients[pow] = transformed;
            } else {
                imaginary.coefficients[pow] = transformed;
            }
        }

        real.truncate();
        imaginary.truncate();
        (real, imaginary)
    }

    /// Isolate all roots directly when every root lies on one of the coordinate
    /// axes
    fn isolate_axis_roots(&self, target_radius: Option<&Rational>) -> Option<Vec<IsolatedRoot>> {
        let real_roots = self.isolate_real_root_intervals();
        let (real_part, imaginary_part) = self.polynomial_at_imaginary_axis_parts();
        let imaginary_axis_poly = match (real_part.is_zero(), imaginary_part.is_zero()) {
            (true, true) => return None,
            (true, false) => imaginary_part,
            (false, true) => real_part,
            (false, false) => real_part.gcd(&imaginary_part),
        };
        let imaginary_roots = if imaginary_axis_poly.is_constant() {
            Vec::new()
        } else {
            imaginary_axis_poly.isolate_real_root_intervals()
        };

        let root_count = real_roots
            .iter()
            .chain(&imaginary_roots)
            .map(|(_, _, multiplicity)| *multiplicity)
            .sum::<usize>();
        if root_count != self.degree() {
            return None;
        }

        let complex_field = FloatField::from_rep(Complex::from(Rational::one()));
        let complex_poly = Arc::new(self.map_coeff(|c| Complex::from(c.clone()), complex_field));
        let mut roots = Vec::with_capacity(root_count);

        for (lower, upper, _) in real_roots {
            let center = (&lower + &upper) / Rational::from(2);
            let radius = (&upper - &lower) / Rational::from(2);
            roots.push(IsolatedRoot {
                poly: complex_poly.clone(),
                index: roots.len(),
                enclosure: ComplexDisk {
                    center: Complex::new(center, Rational::zero()),
                    radius,
                },
                location: Some(RootLocation::Real),
            });
        }
        for (lower, upper, _) in imaginary_roots {
            let center = (&lower + &upper) / Rational::from(2);
            let radius = (&upper - &lower) / Rational::from(2);
            roots.push(IsolatedRoot {
                poly: complex_poly.clone(),
                index: roots.len(),
                enclosure: ComplexDisk {
                    center: Complex::new(Rational::zero(), center),
                    radius,
                },
                location: Some(RootLocation::Imaginary),
            });
        }

        let derivative = complex_poly.derivative();
        if !Self::refine_complex_root_disks_until_disjoint(&complex_poly, &derivative, &mut roots) {
            return None;
        }

        if let Some(target_radius) = target_radius {
            for root in &mut roots {
                if !Self::refine_isolated_complex_root_to_tolerance(root, target_radius) {
                    return None;
                }
            }
        }

        Some(roots)
    }

    // Self is square-free.
    fn isolate_roots_impl(&self, target_radius: Option<&Rational>) -> Vec<IsolatedRoot> {
        if let Some(roots) = self.isolate_axis_roots(target_radius) {
            return roots;
        }

        let complex_field = FloatField::from_rep(Complex::from(Rational::one()));
        let complex_poly = self.map_coeff(
            |coefficient| Complex::from(coefficient.clone()),
            complex_field,
        );
        complex_poly.isolate_roots_impl(target_radius)
    }

    /// Isolate the distinct real roots as rational intervals, together with
    /// their multiplicities.
    pub fn isolate_real_root_intervals(&self) -> Vec<(Rational, Rational, usize)> {
        let c = self.content();

        let stripped = self.map_coeff(
            |coeff| {
                let coeff = self.ring.div(coeff, &c);
                debug_assert!(coeff.is_integer());
                coeff.numerator()
            },
            Z,
        );

        stripped.isolate_real_root_intervals()
    }

    /// Approximate the single root of the polynomial in the interval (lower, higher) with a given tolerance
    /// using bisection.
    pub fn refine_root_interval(
        &self,
        mut interval: (Rational, Rational),
        tolerance: &Rational,
    ) -> (Rational, Rational) {
        if interval.0 == interval.1 {
            return interval;
        }

        // make the input square free, so that the derivative is non-zero at the roots
        let mut u = self.one();
        for (f, _pow) in self
            .clone()
            .to_multivariate::<u16>()
            .square_free_factorization()
        {
            if !f.is_constant() {
                u = u * &f.to_univariate_from_univariate(0);
            }
        }

        let left_bound_neg = match u.evaluate(&interval.0).cmp(&(0, 1).into()) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => u.derivative().evaluate(&interval.0).is_negative(),
        };
        debug_assert!(u.evaluate(&interval.1).is_negative() != left_bound_neg);

        while (&interval.1 - &interval.0) / (&interval.0 + &interval.1).abs() > *tolerance {
            let mid = (&interval.0 + &interval.1) / &(2, 1).into();
            let mid_val = u.evaluate(&mid);

            if mid_val.is_negative() == left_bound_neg {
                interval.0 = mid;
            } else {
                interval.1 = mid;
            }
        }

        interval
    }

    /// Refine the intervals of two polynomials until they are disjoint.
    /// The polynomials must be square free.
    fn refine_root_interval_until_disjoint(
        &self,
        mut interval: (Rational, Rational),
        other: &Self,
        mut other_interval: (Rational, Rational),
    ) -> ((Rational, Rational), (Rational, Rational)) {
        if !(interval.0 >= other_interval.0 && interval.0 < other_interval.1
            || interval.1 > other_interval.0 && interval.1 <= other_interval.1)
        {
            return (interval, other_interval);
        }

        let left_bound_neg = match self.evaluate(&interval.0).cmp(&(0, 1).into()) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => self.derivative().evaluate(&interval.0).is_negative(),
        };
        let other_left_bound_neg = match other.evaluate(&other_interval.0).cmp(&(0, 1).into()) {
            Ordering::Less => true,
            Ordering::Greater => false,
            Ordering::Equal => other.derivative().evaluate(&other_interval.0).is_negative(),
        };

        while interval.0 >= other_interval.0 && interval.0 < other_interval.1
            || interval.1 > other_interval.0 && interval.1 <= other_interval.1
        {
            if interval.0 != interval.1 {
                let mid = (&interval.0 + &interval.1) / &(2, 1).into();
                let mid_val = self.evaluate(&mid);

                if mid_val.is_negative() == left_bound_neg {
                    interval.0 = mid;
                } else {
                    interval.1 = mid;
                }
            }

            if other_interval.0 != other_interval.1 {
                let mid = (&other_interval.0 + &other_interval.1) / &(2, 1).into();
                let mid_val = other.evaluate(&mid);

                if mid_val.is_negative() == other_left_bound_neg {
                    other_interval.0 = mid;
                } else {
                    other_interval.1 = mid;
                }
            }
        }

        (interval, other_interval)
    }

    /// Approximate all complex roots of the polynomial.
    /// Returns `Ok(roots)` when all roots were found up to the tolerance, and `Err(roots)` when the number of iterations ran out.
    /// In that case, the current-best estimate for each root is returned.
    pub fn approximate_roots<
        F: Real + SingleFloat + std::hash::Hash + Eq + PartialOrd + InternalOrdering,
    >(
        &self,
        max_iterations: usize,
        tolerance: &F,
    ) -> Result<Vec<(Complex<F>, usize)>, Vec<(Complex<F>, usize)>> {
        let mut roots = vec![];
        let mut iter_bound = false;
        for (f, pow) in self
            .clone()
            .to_multivariate::<u16>()
            .square_free_factorization()
        {
            if f.is_constant() {
                continue;
            }

            // make monic to prevent casting large integers that may overflow the float
            let f = f.to_univariate_from_univariate(0).make_monic();

            match f
                .map_coeff(
                    |c| tolerance.from_rational(c).into(),
                    FloatField::from_rep(tolerance.clone().into()),
                )
                .roots(max_iterations, tolerance)
            {
                Ok(r) => roots.extend(r.into_iter().map(|r| (r, pow))),
                Err(r) => {
                    roots.extend(r.into_iter().map(|r| (r, pow)));
                    iter_bound = true;
                }
            }
        }

        if iter_bound { Err(roots) } else { Ok(roots) }
    }
}

impl UnivariatePolynomial<ExactComplexField> {
    fn coordinate_axis_polynomial(&self, axis: CoordinateAxis) -> Option<UnivariatePolynomial<Q>> {
        let mut real_part = self.map_coeff(|coefficient| coefficient.re.clone(), Q);
        let mut imaginary_part = self.map_coeff(|coefficient| coefficient.im.clone(), Q);
        if axis == CoordinateAxis::Imaginary {
            real_part
                .coefficients
                .resize(self.coefficients.len(), Rational::zero());
            imaginary_part
                .coefficients
                .resize(self.coefficients.len(), Rational::zero());
            for (power, coefficient) in self.coefficients.iter().enumerate() {
                let rotated =
                    UnivariatePolynomial::<Q>::mul_complex_rational_by_i_power(coefficient, power);
                real_part.coefficients[power] = rotated.re;
                imaginary_part.coefficients[power] = rotated.im;
            }
            real_part.truncate();
            imaginary_part.truncate();
        }

        let polynomial = match (real_part.is_zero(), imaginary_part.is_zero()) {
            (true, true) => return None,
            (true, false) => imaginary_part,
            (false, true) => real_part,
            (false, false) => real_part.gcd(&imaginary_part),
        };
        (!polynomial.is_constant()).then_some(polynomial)
    }

    fn mark_coordinate_axis_roots(
        axis_polynomial: &UnivariatePolynomial<Q>,
        roots: &mut [IsolatedRoot],
        axis: CoordinateAxis,
    ) {
        for (lower, upper, _) in axis_polynomial.isolate_real_root_intervals() {
            let mut interval = (lower, upper);
            let mut identified = false;
            for _ in 0..4096 {
                for root in roots.iter_mut() {
                    if axis.contains_interval(&interval, root) {
                        root.location = Some(RootLocation::with_axis(root.location, axis));
                        identified = true;
                        break;
                    }
                }

                if identified {
                    break;
                }
                axis_polynomial.refine_real_root_interval_once(&mut interval);
            }
            assert!(
                identified,
                "could not match an exact coordinate-axis root to its complex enclosure"
            );
        }
    }

    fn classify_complex_root_locations(&self, roots: &mut [IsolatedRoot]) {
        for axis in [CoordinateAxis::Real, CoordinateAxis::Imaginary] {
            if let Some(axis_polynomial) = self.coordinate_axis_polynomial(axis) {
                Self::mark_coordinate_axis_roots(&axis_polynomial, roots, axis);
            }
        }
        for root in roots {
            if root.location.is_none() {
                root.location = Some(RootLocation::Complex);
            }
        }
    }

    fn try_map_to_rational(&self) -> Option<UnivariatePolynomial<Q>> {
        if self.coefficients.iter().any(|c| !c.im.is_zero()) {
            return None;
        }

        Some(self.map_coeff(|c| c.re.clone(), Q))
    }

    fn complex_rational_to_algebraic(
        field: &AlgebraicExtension<Q>,
        c: &Complex<Rational>,
    ) -> AlgebraicNumber<Q> {
        let mut poly = field.poly().constant(c.re.clone());
        if !c.im.is_zero() {
            poly = poly + field.poly().monomial(c.im.clone(), vec![1]);
        }
        field.to_element(poly)
    }

    fn algebraic_to_complex_rational(c: &AlgebraicNumber<Q>) -> Complex<Rational> {
        Complex::new(
            c.poly().coefficient(&[0]).unwrap_or_else(Rational::zero),
            c.poly().coefficient(&[1]).unwrap_or_else(Rational::zero),
        )
    }

    fn certify_complex_roots_from_approximations(
        &self,
        roots: &[Complex<Float>],
        target_radius: Option<&Rational>,
    ) -> Option<Vec<IsolatedRoot>> {
        let defining_polynomial = Arc::new(self.clone());
        let centers = roots
            .iter()
            .map(|root| Complex::new(root.re.to_rational(), root.im.to_rational()))
            .collect::<Vec<_>>();

        let mut complex_roots = Vec::with_capacity(centers.len());
        for (root_index, center) in centers.iter().enumerate() {
            let mut radius = UnivariatePolynomial::<Q>::complex_root_radius(
                &centers,
                root_index,
                target_radius,
            )?;

            let mut certified_radius = None;
            for _ in 0..16 {
                if UnivariatePolynomial::<Q>::certify_one_complex_root_disk(self, center, &radius) {
                    certified_radius = Some(radius);
                    break;
                }

                radius *= Rational::from((1, 2));
                if radius.is_zero() {
                    break;
                }
            }

            complex_roots.push(IsolatedRoot {
                poly: defining_polynomial.clone(),
                index: root_index,
                enclosure: ComplexDisk {
                    center: center.clone(),
                    radius: certified_radius?,
                },
                location: None,
            });
        }

        let derivative = self.derivative();
        if UnivariatePolynomial::<Q>::refine_complex_root_disks_until_disjoint(
            self,
            &derivative,
            &mut complex_roots,
        ) {
            Some(complex_roots)
        } else {
            None
        }
    }

    /// Gets the `index`-th root of the polynomial. Fails when `index` is out of bounds.
    pub fn root(&self, index: usize) -> Option<IsolatedRoot> {
        if let Some(poly) = self.try_map_to_rational() {
            return poly.root(index);
        }
        if index >= self.degree() {
            return None;
        }

        let entry = ROOT_CACHE.layout_entry(RootPolynomialKey::Complex(self.coefficients.clone()));
        let layout = entry.layout.get_or_init(|| self.compute_root_layout());
        ROOT_CACHE.get_index(layout, index)
    }

    /// Isolate the distinct complex roots of a polynomial with exact complex
    /// rational coefficients as canonically sorted `(root, multiplicity)` pairs.
    /// If all coefficients are rational, use the rational polynomial path and
    /// its root cache.
    pub fn isolate_roots(&self) -> Vec<(IsolatedRoot, usize)> {
        if let Some(poly) = self.try_map_to_rational() {
            return poly.isolate_roots();
        }

        let entry = ROOT_CACHE.layout_entry(RootPolynomialKey::Complex(self.coefficients.clone()));
        let layout = entry.layout.get_or_init(|| self.compute_root_layout());
        ROOT_CACHE.materialize_layout(layout)
    }

    /// Isolate the distinct real roots of the polynomial. Resolving whether a
    /// root lies on the real axis may refine its cached enclosure.
    pub fn isolate_real_roots(&self) -> Vec<(IsolatedRoot, usize)> {
        retain_real_roots(self.isolate_roots())
    }

    fn compute_root_layout(&self) -> PolynomialRootLayout {
        let complex_field = FloatField::from_rep(Complex::from(Rational::one()));
        let algebraic_field = AlgebraicExtension::new_complex(Q);
        let algebraic_poly = self.map_coeff(
            |c| Self::complex_rational_to_algebraic(&algebraic_field, c),
            algebraic_field.clone(),
        );

        let factors = algebraic_poly
            .to_multivariate::<u16>()
            .square_free_factorization()
            .into_iter()
            .filter(|(factor, _)| !factor.is_constant())
            .map(|(factor, multiplicity)| {
                let defining_poly = Arc::new(
                    factor
                        .to_univariate_from_univariate(0)
                        .map_coeff(Self::algebraic_to_complex_rational, complex_field.clone()),
                );
                (defining_poly, multiplicity)
            });
        ROOT_CACHE.layout_from_defining_factors(factors)
    }

    // Self is square-free
    fn isolate_roots_impl(&self, target_radius: Option<&Rational>) -> Vec<IsolatedRoot> {
        const ABERTH_CERTIFICATION_BATCH: usize = 64;
        const MAX_ABERTH_ITERATIONS_PER_PRECISION: usize = 256;

        let mut num_prec = 128;
        let mut previous_roots: Option<Vec<Complex<Float>>> = None;

        loop {
            let tolerance = UnivariatePolynomial::<Q>::complex_root_tolerance(num_prec);
            let field = FloatField::from_rep(Complex::from(tolerance.clone()));
            let c = self.map_coeff(
                |c| {
                    Complex::new(
                        c.re.to_multi_prec_float(num_prec),
                        c.im.to_multi_prec_float(num_prec),
                    )
                },
                field,
            );

            let mut roots_at_precision = previous_roots.take().map(|roots| {
                roots
                    .into_iter()
                    .map(|root: Complex<Float>| {
                        Complex::new(
                            root.re.to_rational().to_multi_prec_float(num_prec),
                            root.im.to_rational().to_multi_prec_float(num_prec),
                        )
                    })
                    .collect::<Vec<_>>()
            });

            let mut iterations = 0;
            while iterations < MAX_ABERTH_ITERATIONS_PER_PRECISION {
                let batch = ABERTH_CERTIFICATION_BATCH
                    .min(MAX_ABERTH_ITERATIONS_PER_PRECISION - iterations);
                let roots = if let Some(initial_guesses) = roots_at_precision.take() {
                    c.roots_hot_start(batch, &tolerance, initial_guesses)
                } else {
                    c.roots(batch, &tolerance)
                };
                iterations += batch;

                let aberth_converged = roots.is_ok();
                let roots = match roots {
                    Ok(roots) => roots,
                    Err(roots) => roots,
                };
                if let Some(complex_roots) =
                    self.certify_complex_roots_from_approximations(&roots, target_radius)
                {
                    return complex_roots;
                }

                roots_at_precision = Some(roots);
                if aberth_converged {
                    break;
                }
            }

            previous_roots = roots_at_precision;
            num_prec *= 2;
        }
    }
}

impl UnivariatePolynomial<IntegerRing> {
    /// Approximate all complex roots of the polynomial.
    /// Returns `Ok(roots)` when all roots were found up to the tolerance, and `Err(roots)` when the number of iterations ran out.
    /// In that case, the current-best estimate for each root is returned.
    pub fn approximate_roots<
        F: Real + SingleFloat + std::hash::Hash + Eq + PartialOrd + InternalOrdering,
    >(
        &self,
        max_iterations: usize,
        tolerance: &F,
    ) -> Result<Vec<(Complex<F>, usize)>, Vec<(Complex<F>, usize)>> {
        self.map_coeff(|c| c.into(), Q)
            .approximate_roots(max_iterations, tolerance)
    }

    /// Approximate the single root of the polynomial in the interval (lower, higher) with a given tolerance
    /// using bisection.
    pub fn refine_root_interval(
        &self,
        interval: (Rational, Rational),
        tolerance: &Rational,
    ) -> (Rational, Rational) {
        self.map_coeff(|c| c.into(), Q)
            .refine_root_interval(interval, tolerance)
    }

    /// Get the number of sign changes in the polynomial.
    pub fn sign_changes(&self) -> usize {
        let mut sign_changes = 0;
        let mut last_sign = 0;
        for c in &self.coefficients {
            let sign = if c < &0 {
                -1
            } else if c > &0 {
                1
            } else {
                0
            };

            if sign != 0 && sign != last_sign {
                if last_sign != 0 {
                    sign_changes += 1;
                }
                last_sign = sign;
            }
        }
        sign_changes
    }

    /// Isolate the distinct real roots as rational intervals, together with
    /// their multiplicities.
    pub fn isolate_real_root_intervals(&self) -> Vec<(Rational, Rational, usize)> {
        let fs = self.clone().to_multivariate::<u16>();
        let mut intervals = vec![];

        for (f, pow) in fs.square_free_factorization() {
            if f.is_constant() {
                continue;
            }

            let f = f.to_univariate_from_univariate(0);
            let mut neg_f = f.clone();
            for c in neg_f.coefficients.iter_mut().skip(1).step_by(2) {
                *c = -c.clone();
            }

            let f_rat = f.map_coeff(|c| c.to_rational(), Q);

            for (i, p) in [neg_f, f].into_iter().enumerate() {
                for mut x in p.isolate_real_roots_square_free() {
                    if i == 0 {
                        std::mem::swap(&mut x.0, &mut x.1);
                        x.0 = -x.0;
                        x.1 = -x.1;
                    }

                    if i == 1 || !x.0.is_zero() || !x.1.is_zero() {
                        intervals.push(((x.0, x.1), f_rat.clone(), pow));
                    }
                }
            }
        }

        for i in 0..intervals.len() {
            for j in i + 1..intervals.len() {
                let (a1, p1, _) = &intervals[i];
                let (a2, p2, _) = &intervals[j];

                if p1 == p2 {
                    continue;
                }

                let (a1, a2) = p1.refine_root_interval_until_disjoint(a1.clone(), p2, a2.clone());
                intervals[i].0 = a1;
                intervals[j].0 = a2;
            }
        }

        intervals.sort_by(|a, b| a.0.cmp(&b.0));

        intervals
            .into_iter()
            .map(|(x, _, pow)| (x.0, x.1, pow))
            .collect()
    }

    /// Compute an upper bound for the maximal positive real root of the polynomial.
    pub fn max_real_root_bound(&self) -> Rational {
        // use the local-max root bound
        // TODO: also implement first-lambda bound
        if self.degree() == 0 {
            return Rational::zero();
        }

        let sign_flip = self.coefficients.last().unwrap() < &0;

        let mut j = self.coefficients.len() - 1;
        let mut t = 1;

        let mut bound = Rational::zero();
        for i in (0..self.coefficients.len()).rev() {
            if !sign_flip && self.coefficients[i] < 0 || sign_flip && self.coefficients[i] > 0 {
                // TODO: what if precision is not enough?
                let tmp: f64 = (-2f64.powf(t as f64) * self.coefficients[i].to_rational().to_f64()
                    / self.coefficients[j].to_rational().to_f64())
                .powf(1. / (j - i) as f64);
                let tmp = Rational::try_from(tmp).unwrap();
                if tmp > bound {
                    bound = tmp;
                }

                t += 1;
            } else if !sign_flip && self.coefficients[i] > self.coefficients[j]
                || sign_flip && self.coefficients[i] < self.coefficients[j]
            {
                j = i;
                t = 1;
            }
        }

        bound
    }

    /// Isolate the roots of the polynomial using VAS-CF.
    fn isolate_real_roots_square_free(&self) -> Vec<(Rational, Rational)> {
        let mut roots = vec![];

        let mut p = self.clone();
        if p.coefficients[0] == 0 {
            roots.push((Rational::zero(), Rational::zero()));

            p = p.div_exp(1);
        }

        let max_root = p.max_real_root_bound().ceil();

        // map the polynomial to the interval (0, max_root)
        for c in p.coefficients.iter_mut().enumerate() {
            *c.1 *= max_root.pow(c.0 as u64);
        }
        p.coefficients.reverse();
        p = p.shift_var(&Integer::from(1));

        if p.coefficients[0] == 0 {
            roots.push((
                (max_root.clone(), Integer::one()).into(),
                (max_root.clone(), Integer::one()).into(),
            ));
            p = p.div_exp(1);
        }

        let s = p.sign_changes();
        if s == 0 {
            return roots;
        }

        if s == 1 {
            roots.push((0.into(), max_root.into()));
            return roots;
        }

        struct Interval {
            a: Integer,
            b: Integer,
            c: Integer,
            d: Integer,
            p: UnivariatePolynomial<IntegerRing>,
            s: usize,
        }

        let mut intervals = vec![Interval {
            a: Integer::zero(),
            b: max_root,
            c: Integer::one(),
            d: Integer::one(),
            p,
            s,
        }];

        while let Some(Interval {
            mut a,
            mut b,
            mut c,
            mut d,
            mut p,
            mut s,
        }) = intervals.pop()
        {
            // compute lower bound on root
            p.coefficients.reverse();
            let upper_bound = p.max_real_root_bound();
            p.coefficients.reverse();
            let mut lower_bound = upper_bound.inv().floor();

            // rescale x if the lower bound is large
            if lower_bound > 16 {
                for (i, c) in p.coefficients.iter_mut().enumerate() {
                    if c != &0 {
                        *c *= lower_bound.pow(i as u64);
                    }
                }

                a *= &lower_bound;
                c *= &lower_bound;
                lower_bound = Integer::one();
            }

            // move the lower bound of the interval
            if lower_bound >= 1 {
                p = p.shift_var(&lower_bound);
                b += &a * &lower_bound;
                d += &c * &lower_bound;

                if p.coefficients[0] == 0 {
                    roots.push(((b.clone(), d.clone()).into(), (b.clone(), d.clone()).into()));
                    p = p.div_exp(1);
                }

                s = p.sign_changes();
                if s == 0 {
                    continue;
                } else if s == 1 {
                    let b1 = (b.clone(), d.clone()).into();
                    let b2 = (a.clone(), c.clone()).into();
                    roots.push(if b1 < b2 { (b1, b2) } else { (b2, b1) });
                    continue;
                }
            }

            let mut n1 = Interval {
                a: a.clone(),
                b: &a + &b,
                c: c.clone(),
                d: &c + &d,
                p: p.shift_var(&1.into()),
                s: 0,
            };
            let mut r = 0;
            if n1.p.coefficients[0] == 0 {
                roots.push((
                    (n1.b.clone(), n1.d.clone()).into(),
                    (n1.b.clone(), n1.d.clone()).into(),
                ));

                n1.p = n1.p.div_exp(1);
                r = 1;
            }
            n1.s = n1.p.sign_changes();

            let mut n2 = Interval {
                a: b.clone(),
                b: a + b,
                c: d.clone(),
                d: c + d,
                p: p.zero(),
                s: s - n1.s - r,
            };
            if n2.s > 1 {
                //construct (x+1)^m p(1/(x+1))
                n2.p = p.clone();
                n2.p.coefficients.reverse();
                n2.p = n2.p.shift_var(&Integer::from(1));

                if n2.p.coefficients[0] == 0 {
                    n2.p = n2.p.div_exp(1);
                }

                n2.s = n2.p.sign_changes();
            }

            if n1.s < n2.s {
                std::mem::swap(&mut n1, &mut n2);
            }

            for int in [n1, n2] {
                if int.s == 0 {
                    continue;
                } else if int.s == 1 {
                    let b1 = (int.b.clone(), int.d.clone()).into();
                    let b2 = (int.a.clone(), int.c.clone()).into();
                    roots.push(if b1 < b2 { (b1, b2) } else { (b2, b1) });
                } else {
                    intervals.push(int);
                }
            }
        }

        roots
    }
}

impl<R: Real + SingleFloat + std::hash::Hash + Eq + PartialOrd + InternalOrdering>
    UnivariatePolynomial<FloatField<Complex<R>>>
{
    /// Get an upper bound on the norm of all (complex) roots.
    pub fn get_root_upper_bound(&self) -> R {
        if self.is_zero() {
            return self.ring.zero().re;
        }

        let last = self.coefficients.last().unwrap();
        let mut max = last.zero().re;
        for c in self.coefficients.iter().rev().skip(1) {
            let r = (c / last).norm().re;
            if r > max {
                max = r;
            }
        }

        max + self.ring.one().re
    }

    /// Get a lower bound on the norm of all (complex) roots.
    pub fn get_root_lower_bound(&self) -> R {
        if self.is_zero() {
            return self.ring.zero().re;
        }

        let last = &self.coefficients[0];
        let mut max = last.zero().re;
        for c in self.coefficients.iter().skip(1) {
            let r = (c / last).norm().re;
            if r > max {
                max = r;
            }
        }

        self.ring.one().re / (max + self.ring.one().re)
    }

    /// Compute all complex roots of the polynomial using Aberth's method.
    /// Returns `Ok(roots)` when all roots were found up to the tolerance, and `Err(roots)` when the number of iterations ran out.
    /// In that case, the current-best estimate for each root is returned.
    ///
    /// For better performance, square-free factor the polynomial first.
    pub fn roots(
        &self,
        max_iterations: usize,
        tolerance: &R,
    ) -> Result<Vec<Complex<R>>, Vec<Complex<R>>> {
        if self.get_constant().is_zero() {
            match self.div_exp(1).roots(max_iterations, tolerance) {
                Ok(mut roots) => {
                    roots.push(self.ring.zero());
                    return Ok(roots);
                }
                Err(mut roots) => {
                    roots.push(self.ring.zero());
                    return Err(roots);
                }
            }
        }

        let upper = self.get_root_upper_bound();
        let lower = self.get_root_lower_bound();
        let radius_span = upper.clone() - &lower;
        let golden_angle = upper.pi() * (upper.from_usize(3) - upper.from_usize(5).sqrt());
        let degree = self.degree();

        let n: Vec<_> = (0..degree)
            .map(|i| {
                let radius_fraction = upper.from_usize(i + 1) / upper.from_usize(degree + 1);
                let r = lower.clone() + radius_span.clone() * &radius_fraction;
                let phi = golden_angle.clone() * upper.from_usize(i + 1);
                Complex::from_polar_coordinates(r, phi)
            })
            .collect();

        self.roots_hot_start(max_iterations, tolerance, n)
    }

    /// Compute all complex roots of the polynomial using Aberth's method with an initial guess for each root.
    pub fn roots_hot_start(
        &self,
        max_iterations: usize,
        tolerance: &R,
        initial_guesses: Vec<Complex<R>>,
    ) -> Result<Vec<Complex<R>>, Vec<Complex<R>>> {
        if self.get_constant().is_zero() {
            match self.div_exp(1).roots(max_iterations, tolerance) {
                Ok(mut roots) => {
                    roots.push(self.ring.zero());
                    return Ok(roots);
                }
                Err(mut roots) => {
                    roots.push(self.ring.zero());
                    return Err(roots);
                }
            }
        }

        let df = self.derivative();

        let mut n = initial_guesses;
        let finite_error = |roots: &[Complex<R>]| {
            let mut roots = roots.to_vec();
            roots.sort_unstable_by(|a, b| {
                a.re.partial_cmp(&b.re)
                    .unwrap_or(Ordering::Equal)
                    .then(a.im.partial_cmp(&b.im).unwrap_or(Ordering::Equal))
            });
            Err(roots)
        };

        let t_sq = tolerance.clone() * tolerance;
        for _ in 0..max_iterations {
            for i in 0..n.len() {
                let last_finite = n.clone();
                let p_at_i = self.evaluate(&n[i]);
                let df_at_i = df.evaluate(&n[i]);
                if !p_at_i.is_finite() || !df_at_i.is_finite() || df_at_i.is_zero() {
                    return finite_error(&last_finite);
                }

                let e = p_at_i / df_at_i;
                if !e.is_finite() {
                    return finite_error(&last_finite);
                }

                let mut rep = e.zero();
                for j in 0..n.len() {
                    if i != j && n[i] != n[j] {
                        let diff = n[i].clone() - &n[j];
                        if !diff.is_finite() || diff.is_zero() {
                            return finite_error(&last_finite);
                        }

                        let diff_inv = diff.inv();
                        if !diff_inv.is_finite() {
                            return finite_error(&last_finite);
                        }

                        rep += diff_inv;
                    }
                }
                if !rep.is_finite() {
                    return finite_error(&last_finite);
                }

                let denom = rep.one() - &e * rep;
                if !denom.is_finite() || denom.is_zero() {
                    return finite_error(&last_finite);
                }

                let correction = e / denom;
                if !correction.is_finite() {
                    return finite_error(&last_finite);
                }

                n[i] -= correction;
                if !n[i].is_finite() {
                    return finite_error(&last_finite);
                }
            }
            if n.iter().all(|x| self.evaluate(x).norm_squared() < t_sq) {
                n.sort_unstable_by(|a, b| {
                    a.re.partial_cmp(&b.re)
                        .unwrap_or(Ordering::Equal)
                        .then(a.im.partial_cmp(&b.im).unwrap_or(Ordering::Equal))
                });
                return Ok(n);
            }
        }

        n.sort_unstable_by(|a, b| {
            a.re.partial_cmp(&b.re)
                .unwrap_or(Ordering::Equal)
                .then(a.im.partial_cmp(&b.im).unwrap_or(Ordering::Equal))
        });
        Err(n)
    }
}

impl<F: Ring> PartialEq for UnivariatePolynomial<F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.variable != other.variable {
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

            // TODO: what is expected here?
            unimplemented!(
                "Cannot compare non-constant polynomials with different variable maps yet"
            );
        }

        if self.degree() != other.degree() {
            return false;
        }
        self.coefficients.eq(&other.coefficients)
    }
}

impl<F: Ring> std::hash::Hash for UnivariatePolynomial<F> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coefficients.hash(state);
        self.variable.hash(state);
    }
}

impl<F: Ring> Eq for UnivariatePolynomial<F> {}

impl<R: Ring> PartialOrd for UnivariatePolynomial<R> {
    /// An ordering of polynomials that has no intuitive meaning.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.coefficients.internal_cmp(&other.coefficients))
    }
}

impl<F: Ring> Add for UnivariatePolynomial<F> {
    type Output = Self;

    fn add(mut self, mut other: Self) -> Self::Output {
        assert_eq!(self.ring, other.ring);

        if self.variable != other.variable {
            panic!("Cannot multiply polynomials with different variables");
        }

        if self.is_zero() {
            return other;
        }
        if other.is_zero() {
            return self;
        }

        if self.degree() < other.degree() {
            std::mem::swap(&mut self, &mut other);
        }

        for (i, c) in other.coefficients.iter().enumerate() {
            self.ring.add_assign(&mut self.coefficients[i], c);
        }

        self.truncate();

        self
    }
}

impl<'a, F: Ring> Add<&'a UnivariatePolynomial<F>> for &UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn add(self, other: &'a UnivariatePolynomial<F>) -> Self::Output {
        (self.clone()).add(other.clone())
    }
}

impl<F: Ring> Sub for UnivariatePolynomial<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.add(other.neg())
    }
}

impl<'a, F: Ring> Sub<&'a UnivariatePolynomial<F>> for &UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn sub(self, other: &'a UnivariatePolynomial<F>) -> Self::Output {
        (self.clone()).add(other.clone().neg())
    }
}

impl<F: Ring> Neg for UnivariatePolynomial<F> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        // Negate coefficients of all terms.
        for c in &mut self.coefficients {
            *c = self.ring.neg(&*c);
        }
        self
    }
}

impl<'a, F: Ring> Mul<&'a UnivariatePolynomial<F>> for &UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    #[inline]
    fn mul(self, rhs: &'a UnivariatePolynomial<F>) -> Self::Output {
        assert_eq!(self.ring, rhs.ring);

        if self.is_zero() || rhs.is_zero() {
            return self.zero();
        }

        if self.variable != rhs.variable {
            panic!("Cannot multiply polynomials with different variables");
        }

        let n = self.degree();
        let m = rhs.degree();

        if n == 0 {
            let mut r = rhs.clone();
            for c in &mut r.coefficients {
                self.ring.mul_assign(c, &self.coefficients[0]);
            }
            return r;
        }

        if m == 0 {
            let mut r = self.clone();
            for c in &mut r.coefficients {
                self.ring.mul_assign(c, &rhs.coefficients[0]);
            }
            return r;
        }

        let mut res = self.zero();
        res.coefficients = vec![self.ring.zero(); n + m + 1];

        for (e1, c1) in self.coefficients.iter().enumerate() {
            if self.ring.is_zero(c1) {
                continue;
            }

            for (e2, c2) in rhs.coefficients.iter().enumerate() {
                if !self.ring.is_zero(c2) {
                    self.ring
                        .add_mul_assign(&mut res.coefficients[e1 + e2], c1, c2);
                }
            }
        }

        res.truncate();
        res
    }
}

impl<'a, F: Ring> Mul<&'a UnivariatePolynomial<F>> for UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    #[inline]
    fn mul(self, rhs: &'a UnivariatePolynomial<F>) -> Self::Output {
        (&self) * rhs
    }
}

impl<'a, F: EuclideanDomain> Div<&'a UnivariatePolynomial<F>> for &UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn div(self, other: &'a UnivariatePolynomial<F>) -> Self::Output {
        self.try_div(other)
            .unwrap_or_else(|| panic!("No exact division of {self} by {other}"))
    }
}

impl<'a, F: EuclideanDomain> Div<&'a UnivariatePolynomial<F>> for UnivariatePolynomial<F> {
    type Output = UnivariatePolynomial<F>;

    fn div(self: UnivariatePolynomial<F>, other: &'a UnivariatePolynomial<F>) -> Self::Output {
        (&self).div(other)
    }
}

impl<F: EuclideanDomain> UnivariatePolynomial<F> {
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

    /// Compute the remainder `self % div`.
    pub fn rem(&self, div: &UnivariatePolynomial<F>) -> Self {
        self.quot_rem(div).1
    }

    pub fn quot_rem(&self, div: &Self) -> (Self, Self) {
        self.quot_rem_impl(div, false)
    }

    /// Compute the p-adic expansion of the polynomial.
    /// It returns `[a0, a1, a2, ...]` such that `a0 + a1 * p^1 + a2 * p^2 + ... = self`.
    pub fn p_adic_expansion(&self, p: &Self) -> Vec<Self> {
        if self.variable != p.variable {
            panic!("Cannot apply p-adic expansion with different variables");
        }

        let mut res = vec![];
        let mut r = self.clone();
        while !r.is_zero() {
            let (q, rem) = r.quot_rem(p);
            res.push(rem);
            r = q;
        }
        res
    }

    /// Integrate the polynomial w.r.t the variable `var`,
    /// producing the antiderivative with zero constant.
    pub fn integrate(&self) -> Self {
        if self.is_zero() {
            return self.zero();
        }

        let mut res = self.zero();
        res.coefficients
            .resize(self.coefficients.len() + 1, self.ring.zero());

        for (p, (nc, oc)) in res
            .coefficients
            .iter_mut()
            .skip(1)
            .zip(&self.coefficients)
            .enumerate()
        {
            if !self.ring.is_zero(oc) {
                let (q, r) = self.ring.quot_rem(oc, &self.ring.nth(Integer::from(p) + 1));
                if !self.ring.is_zero(&r) {
                    panic!(
                        "Could not compute integral since there is a remainder in the division of the exponent number."
                    );
                }
                *nc = q;
            }
        }

        res
    }
}

impl<F: Field> UnivariatePolynomial<F> {
    /// Make the polynomial monic, i.e., make the leading coefficient `1` by
    /// multiplying all monomials with `1/lcoeff`.
    pub fn make_monic(self) -> Self {
        if self.lcoeff() != self.ring.one() {
            let ci = self.ring.inv(&self.lcoeff());
            self.mul_coeff(&ci)
        } else {
            self
        }
    }

    /// Compute self^n % m where m is a polynomial
    pub fn exp_mod(&self, mut n: Integer, m: &mut Self) -> Self {
        if n.is_zero() {
            return self.one();
        }

        // use binary exponentiation and mod at every stage
        let mut x = self.rem(m);
        let mut y = self.one();
        while !n.is_one() {
            if (&n % &Integer::Single(2)).is_one() {
                y = (&y * &x).quot_rem(m).1;
                n -= &Integer::one();
            }

            x = (&x * &x).rem(m);
            n /= 2;
        }

        (x * &y).rem(m)
    }

    /// Compute `(g, s, t)` where `self * s + other * t = g`
    /// by means of the extended Euclidean algorithm.
    pub fn eea(&self, other: &Self) -> (Self, Self, Self) {
        if self.variable != other.variable {
            panic!("Cannot apply EEA with different variables");
        }

        let mut r0 = self.clone().make_monic();
        let mut r1 = other.clone().make_monic();
        let mut s0 = self.constant(self.ring.inv(&self.lcoeff()));
        let mut s1 = self.zero();
        let mut t0 = self.zero();
        let mut t1 = self.constant(self.ring.inv(&other.lcoeff()));

        while !r1.is_zero() {
            let (q, r) = r0.quot_rem(&r1);
            if self.ring.is_zero(&r.lcoeff()) {
                return (r1, s1, t1);
            }

            let a = self.ring.inv(&r.lcoeff());
            (r1, r0) = (r.mul_coeff(&a), r1);
            (s1, s0) = ((s0 - &q * &s1).mul_coeff(&a), s1);
            (t1, t0) = ((t0 - q * &t1).mul_coeff(&a), t1);
        }

        (r0, s0, t0)
    }

    /// Compute `(s1,...,n2)` where `A0 * s0 + ... + An * sn = g`
    /// where `Ai = prod(polys[j], j != i)`
    /// by means of the extended Euclidean algorithm.
    ///
    /// The `polys` must be pairwise co-prime.
    pub fn diophantine(polys: &mut [Self], b: &Self) -> Vec<Self> {
        if polys.len() < 2 {
            panic!("Need at least two polynomials for the diophantine equation");
        }

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
            let (g, s, t) = p.eea(aa);
            debug_assert!(g.is_one());
            let new_s = (t * &cur_s).rem(p);
            ss.push(new_s);
            cur_s = (s * &cur_s).rem(aa);
        }

        ss.push(cur_s);
        ss
    }

    /// Compute the univariate GCD using Euclid's algorithm. The result is normalized to 1.
    pub fn gcd(&self, b: &Self) -> Self {
        if self.is_zero() {
            return b.clone();
        }
        if b.is_zero() {
            return self.clone();
        }

        if self.variable != b.variable {
            panic!("Cannot compute GCD of polynomials with different variables");
        }

        let mut c = self.clone();
        let mut d = b.clone();
        if self.degree() < b.degree() {
            std::mem::swap(&mut c, &mut d);
        }

        // TODO: there exists an efficient algorithm for univariate poly
        // division in a finite field using FFT
        let mut r = c.quot_rem(&d).1;
        while !r.is_zero() {
            c = d;
            d = r;
            r = c.quot_rem(&d).1;
        }

        // normalize the gcd
        let l = d.coefficients.last().unwrap().clone();
        for x in &mut d.coefficients {
            self.ring.div_assign(x, &l);
        }

        d
    }

    /// Optimized division routine for univariate polynomials over a field, which
    /// makes the divisor monic first.
    pub fn quot_rem_field(
        &self,
        div: &mut UnivariatePolynomial<F>,
    ) -> (UnivariatePolynomial<F>, UnivariatePolynomial<F>) {
        if self.is_zero() {
            return (self.clone(), self.clone());
        }

        if self.variable != div.variable {
            panic!("Cannot divide polynomials with different variables");
        }

        let mut n = self.degree();
        let m = div.degree();

        let u = self.ring.inv(&div.coefficients[m]);

        let mut q = self.zero();
        q.coefficients = vec![self.ring.zero(); n - m + 1];

        let mut r = self.clone();

        while n >= m {
            let qq = self.ring.mul(&r.coefficients[n], &u);
            r = r - div.mul_exp(n - m).mul_coeff(&qq);
            q.coefficients[n - m] = qq;
            n = r.degree();
        }

        q.truncate();

        (q, r)
    }
}

impl<R: Ring, E: PositiveExponent> UnivariatePolynomial<PolynomialRing<R, E>> {
    /// Convert a univariate polynomial of multivariate polynomials to a multivariate polynomial.
    pub fn flatten(self) -> MultivariatePolynomial<R, E> {
        if self.is_zero() {
            return self.ring.zero();
        }

        let Some(pos) = self.coefficients[0]
            .variables
            .iter()
            .position(|x| x == self.variable.as_ref())
        else {
            panic!("Variable not found in the field");
        };

        let n_vars = self.coefficients[0].get_vars().len();
        let mut res = MultivariatePolynomial::new(
            &self.ring.ring,
            self.degree().into(),
            self.coefficients[0].get_vars().clone(),
        );

        for (p, mut c) in self.coefficients.into_iter().enumerate() {
            for (e, nc) in c.exponents.chunks_mut(n_vars).zip(c.coefficients) {
                e[pos] = E::from_u32(p as u32);
                res.append_monomial(nc, e);
            }
        }

        res
    }
}

#[cfg(test)]
mod test {
    use super::{
        PolynomialRootLayout, ROOT_CACHE, RootCache, RootLocation, RootPolynomialKey, RootState,
        UnivariatePolynomial,
    };
    use rug::Integer;
    use std::{
        cmp::Ordering,
        sync::{
            Arc, Barrier, RwLock,
            atomic::{AtomicUsize, Ordering as AtomicOrdering},
        },
        time::Duration,
    };

    use crate::{
        atom::AtomCore,
        domains::{
            float::{Complex, F64, Float, FloatField},
            integer::Z,
            rational::{Q, Rational},
        },
        parse,
        poly::{PolyVariable, univariate::IsolatedRoot},
    };

    #[test]
    fn derivative_integrate() {
        let a = parse!("x^2+5x+x^7+3")
            .to_polynomial::<_, u8>(&Q, None)
            .to_univariate_from_univariate(0);

        let r = a.integrate().derivative();

        assert_eq!(a, r);
    }

    #[test]
    fn test_uni() {
        let a = parse!("x^2+5x+x^7+3")
            .to_polynomial::<_, u8>(&Z, None)
            .to_univariate_from_univariate(0);
        let b = parse!("x^2 + 6")
            .to_polynomial::<_, u8>(&Z, None)
            .to_univariate_from_univariate(0);

        let a_plus_b = parse!("9+5*x+2*x^2+x^7")
            .to_polynomial::<_, u8>(&Z, None)
            .to_univariate_from_univariate(0);

        let a_mul_b = parse!("18+30*x+9*x^2+5*x^3+x^4+6*x^7+x^9")
            .to_polynomial::<_, u8>(&Z, None)
            .to_univariate_from_univariate(0);

        let a_quot_b = parse!("1+36*x+-6*x^3+x^5")
            .to_polynomial::<_, u8>(&Z, None)
            .to_univariate_from_univariate(0);

        let a_rem_b = parse!("-3+-211*x")
            .to_polynomial::<_, u8>(&Z, None)
            .to_univariate_from_univariate(0);

        assert_eq!(&a + &b, a_plus_b);
        assert_eq!(&a * &b, a_mul_b);
        assert_eq!(a.quot_rem(&b), (a_quot_b, a_rem_b));

        let c = a.evaluate(&5.into());
        assert_eq!(c, 78178);
    }

    #[test]
    fn isolate() {
        let p =
        parse!("-13559717115*x^6+624134407779*x^7+-13046815434285*x^8+163110612017313*x^9+-1347733455544188*x^10+7635969738026784*x^11+-29444295941654904*x^12+71604709665043392*x^13+-77045857071990336*x^14+-99619711608972096*x^15+375578692434494208*x^16+66256662107418624*x^17+-1548072112541055488*x^18+800263217632600064*x^19+4816054475648851968*x^20+-4271696436901249024*x^21+-12066471810013724672*x^22+10894783995791278080*x^23+28270081588804452352*x^24+-17402041731641245696*x^25+-56047633173904883712*x^26+8535267319469834240*x^27+82086860869945262080*x^28+30788799964221800448*x^29+-66898313364436418560*x^30+-66318040948916879360*x^31+44159548067414016*x^32+31084367995645984768*x^33+20957883496015069184*x^34+6860635897973440512*x^35+1254041389990150144*x^36+123004564822556672*x^37+5066549580791808*x^38")
        .to_polynomial::<_, u32>(&Q, None)
        .to_univariate_from_univariate(0);

        let roots = p.isolate_real_root_intervals();

        assert_eq!(
            roots,
            vec![
                ((-7, 1).into(), (-7, 2).into(), 6),
                ((-1, 1).into(), (-1, 1).into(), 3),
                ((0, 1).into(), (0, 1).into(), 6),
                ((1, 8).into(), (3, 16).into(), 3),
                ((15, 64).into(), (9, 32).into(), 1),
                ((3, 4).into(), (1, 1).into(), 1),
            ],
        );

        let ref_roots: Vec<_> = roots
            .into_iter()
            .map(|x| {
                let r = p.refine_root_interval((x.0, x.1), &(1, 1000).into());
                (r.0, r.1, x.2)
            })
            .collect();

        assert_eq!(
            ref_roots,
            vec![
                ((-3955, 1024).into(), (-987, 256).into(), 6),
                ((-1, 1).into(), (-1, 1).into(), 3),
                ((0, 1).into(), (0, 1).into(), 6),
                ((723, 4096).into(), (181, 1024).into(), 3),
                ((1023, 4096).into(), (2049, 8192).into(), 1),
                ((995, 1024).into(), (249, 256).into(), 1),
            ],
        );
    }

    #[test]
    fn complex_roots() {
        let p = parse!("x^10+9x^7+4x^3+2x+1")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let pc = p.approximate_roots::<F64>(10000, &1e-8.into()).unwrap();
        assert!(pc[0].0.re < 2f64.into());
        assert!(pc[9].0.re > 1f64.into());
    }

    fn assert_pairwise_isolated(roots: &[IsolatedRoot]) {
        for i in 0..roots.len() {
            for j in i + 1..roots.len() {
                assert!(roots[i].enclosure().is_disjoint(roots[j].enclosure()));
            }
        }
    }

    fn is_real(root: &IsolatedRoot) -> bool {
        matches!(root.location, Some(RootLocation::Real | RootLocation::Zero))
    }

    fn is_imaginary(root: &IsolatedRoot) -> bool {
        matches!(
            root.location,
            Some(RootLocation::Imaginary | RootLocation::Zero)
        )
    }

    fn assert_complex_roots_canonical(roots: &[IsolatedRoot]) {
        for pair in roots.windows(2) {
            assert!(
                UnivariatePolynomial::<Q>::cmp_complex_roots_canonical(&pair[0], &pair[1])
                    != Ordering::Greater
            );
        }
    }

    fn without_multiplicities(roots: Vec<(IsolatedRoot, usize)>) -> Vec<IsolatedRoot> {
        roots.into_iter().map(|(root, _)| root).collect()
    }

    fn refine_roots(
        roots: Vec<(IsolatedRoot, usize)>,
        tolerance: &Rational,
    ) -> Vec<(IsolatedRoot, usize)> {
        roots
            .into_iter()
            .map(|(root, multiplicity)| (root.refine(tolerance), multiplicity))
            .collect()
    }

    fn resolve_locations(roots: Vec<IsolatedRoot>) -> Vec<IsolatedRoot> {
        roots
            .into_iter()
            .map(|mut root| {
                root.location();
                root
            })
            .collect()
    }

    #[test]
    fn aberth_handles_very_close_roots_without_non_finite_iterates() {
        let p = parse!("(x^2+1)*((x-1/1000000)^2+1)*((x+1/1000000)^2+1)")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);

        let prec = 256;
        let tolerance = Float::with_val(prec, 1e-40);
        let field = FloatField::from_rep(Complex::from(tolerance.clone()));
        let p_float = p.map_coeff(|c| c.to_multi_prec_float(prec).into(), field);

        let roots = match p_float.roots(10000, &tolerance) {
            Ok(roots) | Err(roots) => roots,
        };

        assert_eq!(roots.len(), 6);
        assert!(roots.iter().all(|r| r.re.is_finite() && r.im.is_finite()));
    }

    #[test]
    fn complex_root_isolation_refines_across_factors() {
        let p = parse!("(x^2+1)*((x-1/100)^2+1)")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let roots = without_multiplicities(p.isolate_roots());

        assert_eq!(roots.len(), 4);
        assert_pairwise_isolated(&roots);
        assert_complex_roots_canonical(&roots);
    }

    #[test]
    fn complex_root_isolation_marks_real_roots() {
        let p = parse!("(x+1)*(x-2)*(x^2+1)")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let roots = resolve_locations(without_multiplicities(p.isolate_roots()));

        assert_eq!(roots.len(), 4);
        assert_eq!(roots.iter().filter(|root| is_real(root)).count(), 2);
        assert_eq!(roots.iter().filter(|root| is_imaginary(root)).count(), 2);
        assert!(roots.iter().all(|root| root.location.is_some()));
        assert_pairwise_isolated(&roots);
        assert_complex_roots_canonical(&roots);
    }

    #[test]
    fn real_root_isolation_returns_root_objects_with_multiplicity() {
        let p = parse!("(x+1)^2*(x-2)^3*(x^2+1)^4")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let mut roots = p.isolate_real_roots();

        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0].1, 2);
        assert_eq!(roots[1].1, 3);
        assert!(!roots[0].0.is_positive());
        assert!(roots[1].0.is_positive());
        assert!(roots.iter().all(|(root, _)| is_real(root)));
    }

    #[test]
    fn isolated_root_to_atom_uses_the_canonical_root_variable() {
        let x_polynomial = parse!("x^5+x+1")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let y_polynomial = parse!("y^5+y+1")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);

        assert_eq!(
            x_polynomial.root(0).unwrap().to_atom(),
            y_polynomial.root(0).unwrap().to_atom()
        );
    }

    #[test]
    fn exact_complex_isolated_root_to_atom_is_exact() {
        let field = FloatField::from_rep(Complex::from(Rational::one()));
        let mut polynomial =
            UnivariatePolynomial::new(&field, None, Arc::new(PolyVariable::Temporary(0)));
        polynomial.coefficients = vec![
            Complex::new(Rational::from(-1), Rational::from(-1)),
            Complex::new(Rational::one(), Rational::zero()),
        ];

        assert_eq!(polynomial.root(0).unwrap().to_atom(), parse!("1+1𝑖"));
    }

    #[test]
    fn complex_root_isolation_marks_imaginary_roots_from_common_axis_part() {
        let p = parse!("x^3+x")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let roots = resolve_locations(without_multiplicities(p.isolate_roots()));

        assert_eq!(roots.len(), 3);
        assert_eq!(roots.iter().filter(|root| is_imaginary(root)).count(), 3);
        assert_eq!(roots[1].location, Some(RootLocation::Zero));
        assert_pairwise_isolated(&roots);
        assert_complex_roots_canonical(&roots);
    }

    #[test]
    fn complex_root_isolation_handles_axis_binomial() {
        let p = parse!("x^4-2")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let tolerance = Rational::from((1, 1 << 16));
        let roots = resolve_locations(without_multiplicities(refine_roots(
            p.isolate_roots(),
            &tolerance,
        )));

        assert_eq!(roots.len(), 4);
        assert_eq!(roots.iter().filter(|root| is_real(root)).count(), 2);
        assert_eq!(roots.iter().filter(|root| is_imaginary(root)).count(), 2);
        assert_pairwise_isolated(&roots);
        assert!(is_real(&roots[0]));
        assert!(is_imaginary(&roots[1]) && roots[1].enclosure().center().im.is_negative());
        assert!(is_imaginary(&roots[2]) && roots[2].enclosure().center().im > Rational::zero());
        assert!(is_real(&roots[3]));
    }

    #[test]
    fn complex_root_reisolation_honors_target_radius_for_axis_roots() {
        let p = parse!("x^4-2")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let target_radius = Rational::from((Integer::from(1), Integer::from(1) << 32));
        let roots = p.isolate_roots_impl(Some(&target_radius));

        assert!(
            roots
                .iter()
                .all(|root| root.enclosure().radius() <= &target_radius)
        );
    }

    #[test]
    fn complex_root_isolation_handles_non_axis_binomial() {
        let p = parse!("x^8-2")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let tolerance = Rational::from((1, 1 << 16));
        let roots = resolve_locations(without_multiplicities(refine_roots(
            p.isolate_roots(),
            &tolerance,
        )));

        assert_eq!(roots.len(), 8);
        assert_eq!(roots.iter().filter(|root| is_real(root)).count(), 2);
        assert_eq!(roots.iter().filter(|root| is_imaginary(root)).count(), 2);
        assert_pairwise_isolated(&roots);
        assert!(is_real(&roots[0]));
        assert!(roots[1].enclosure().center().im.is_negative());
        assert!(roots[2].enclosure().center().im > Rational::zero());
        assert!(is_imaginary(&roots[3]) && roots[3].enclosure().center().im.is_negative());
        assert!(is_imaginary(&roots[4]) && roots[4].enclosure().center().im > Rational::zero());
        assert!(roots[5].enclosure().center().im.is_negative());
        assert!(roots[6].enclosure().center().im > Rational::zero());
        assert!(is_real(&roots[7]));
        assert!(
            roots
                .iter()
                .enumerate()
                .all(|(index, root)| root.index() == index)
        );
    }

    fn complex_root_contains(root: &IsolatedRoot, re: Rational, im: Rational) -> bool {
        (&root.enclosure.center.re - &re).abs() <= root.enclosure.radius
            && (&root.enclosure.center.im - &im).abs() <= root.enclosure.radius
    }

    #[test]
    fn complex_root_isolation_handles_exact_complex_coefficients() {
        let field = FloatField::from_rep(Complex::from(Rational::one()));
        let mut p = UnivariatePolynomial::new(&field, None, Arc::new(PolyVariable::Temporary(0)));
        p.coefficients = vec![
            Complex::new(Rational::from(3), Rational::one()),
            Complex::new(Rational::from(-3), Rational::zero()),
            Complex::new(Rational::one(), Rational::zero()),
        ];

        let roots = without_multiplicities(p.isolate_roots());

        assert_eq!(roots.len(), 2);
        assert!(complex_root_contains(
            &roots[0],
            Rational::one(),
            Rational::one()
        ));
        assert!(complex_root_contains(
            &roots[1],
            Rational::from(2),
            Rational::from(-1)
        ));
        assert_pairwise_isolated(&roots);
        assert_complex_roots_canonical(&roots);
    }

    #[test]
    fn exact_complex_root_location_is_resolved_and_cached() {
        let field = FloatField::from_rep(Complex::from(Rational::one()));
        let mut p = UnivariatePolynomial::new(&field, None, Arc::new(PolyVariable::Temporary(0)));
        p.coefficients = vec![
            Complex::new(Rational::one(), Rational::from(-1)),
            Complex::new(Rational::zero(), Rational::from(3)),
            Complex::new(Rational::from(-2), Rational::from(-2)),
            Complex::new(Rational::one(), Rational::zero()),
        ];

        let roots = without_multiplicities(p.isolate_roots());
        assert!(roots.iter().all(|root| root.location.is_none()));

        let mut first = roots[0].clone();
        let first_location = first.location();
        assert_eq!(first_location, RootLocation::Imaginary);
        assert!(is_imaginary(&first));

        let cached_real = p.root(1).unwrap();
        assert_eq!(cached_real.location, Some(RootLocation::Real));
        assert!(is_real(&cached_real));

        let cached_complex = p.root(2).unwrap();
        assert_eq!(cached_complex.location, Some(RootLocation::Complex));
        assert!(!is_real(&cached_complex));
        assert!(!is_imaginary(&cached_complex));
    }

    #[test]
    fn complex_root_canonical_sort_handles_equal_real_parts() {
        let field = FloatField::from_rep(Complex::from(Rational::one()));
        let mut p = UnivariatePolynomial::new(&field, None, Arc::new(PolyVariable::Temporary(0)));
        p.coefficients = vec![
            Complex::new(Rational::from(-1), Rational::from(3)),
            Complex::new(Rational::from(-2), Rational::from(-3)),
            Complex::new(Rational::one(), Rational::zero()),
        ];

        let roots = without_multiplicities(p.isolate_roots());

        assert_eq!(roots.len(), 2);
        assert!(complex_root_contains(
            &roots[0],
            Rational::one(),
            Rational::one()
        ));
        assert!(complex_root_contains(
            &roots[1],
            Rational::one(),
            Rational::from(2)
        ));
        assert_pairwise_isolated(&roots);
        assert_complex_roots_canonical(&roots);
    }

    #[test]
    fn complex_root_isolation_maps_rational_complex_coefficients_to_q() {
        let p = parse!("x^2+1")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let field = FloatField::from_rep(Complex::from(Rational::one()));
        let p_complex = p.map_coeff(|c| Complex::from(c.clone()), field);

        let roots = resolve_locations(without_multiplicities(p_complex.isolate_roots()));

        assert_eq!(roots.len(), 2);
        assert_eq!(roots.iter().filter(|root| is_imaginary(root)).count(), 2);
        assert_complex_roots_canonical(&roots);
    }

    #[test]
    fn complex_root_isolation_handles_very_close_conjugate_pairs() {
        let p = parse!("(x^2+1)*((x-1/10000)^2+1)*((x+1/10000)^2+1)")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let roots = without_multiplicities(p.isolate_roots());

        assert_eq!(roots.len(), 6);
        assert_pairwise_isolated(&roots);
        assert_complex_roots_canonical(&roots);
    }

    #[test]
    fn complex_root_isolation_handles_mixed_close_clusters() {
        let p = parse!(
            "((x-2)^2+(1/10)^2)*\
             ((x-2-1/100)^2+(1/10)^2)*\
             ((x+3)^2+(1/5)^2)*\
             (x^2+2*x+2)"
        )
        .to_polynomial::<_, u16>(&Q, None)
        .to_univariate_from_univariate(0);
        let refine = Rational::from((1, 1000));
        let roots = without_multiplicities(refine_roots(p.isolate_roots(), &refine));

        assert_eq!(roots.len(), 8);
        assert_pairwise_isolated(&roots);
        assert_complex_roots_canonical(&roots);
        assert!(
            roots
                .iter()
                .all(|root| root.enclosure().radius() <= &refine)
        );
    }

    #[test]
    fn targeted_complex_root_refinement_is_retained_in_cache() {
        let p = parse!("x^3-7919")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let tolerance = Rational::from((Integer::from(1), Integer::from(1) << 80));

        let original = p.root(0).unwrap();
        let defining_key = original
            .defining_polynomial()
            .try_map_to_rational()
            .unwrap()
            .coefficients;
        ROOT_CACHE
            .roots
            .write()
            .unwrap()
            .remove(&RootPolynomialKey::Rational(defining_key));
        let refined = original.clone().refine(&tolerance);
        assert!(refined.enclosure().radius() <= &tolerance);

        let cached = p.root(0).unwrap();
        assert_eq!(cached.enclosure().center(), refined.enclosure().center());
        assert_eq!(cached.enclosure().radius(), refined.enclosure().radius());

        let refreshed = original.refine(&Rational::one());
        assert_eq!(refreshed.enclosure().center(), refined.enclosure().center());
        assert_eq!(refreshed.enclosure().radius(), refined.enclosure().radius());
    }

    #[test]
    fn isolated_root_determines_real_sign() {
        let p = parse!("x*(x-2)*(x+3)")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let mut roots = without_multiplicities(p.isolate_roots());

        assert!(!roots[0].is_positive());
        assert_eq!(roots[1].location(), RootLocation::Zero);
        assert!(!roots[1].is_positive());
        assert!(roots[2].is_positive());

        let mut non_real = parse!("x^2+1")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0)
            .root(0)
            .unwrap();
        assert!(!non_real.is_positive());
    }

    #[test]
    fn all_root_refinement_is_retained_in_cache() {
        let p = parse!("x^4+43*x+103")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let tolerance = Rational::from((Integer::from(1), Integer::from(1) << 72));

        p.isolate_roots();
        let refined = refine_roots(p.isolate_roots(), &tolerance);
        assert!(
            refined
                .iter()
                .all(|(root, _)| root.enclosure.radius <= tolerance)
        );

        let cached = p.isolate_roots();
        assert!(cached.iter().zip(refined).all(
            |((cached, cached_multiplicity), (refined, refined_multiplicity))| {
                cached.enclosure().center() == refined.enclosure().center()
                    && cached.enclosure().radius() == refined.enclosure().radius()
                    && cached_multiplicity == &refined_multiplicity
            }
        ));
    }

    #[test]
    fn complex_root_multiplicity_expands_canonical_indices() {
        let p = parse!("(x+2)^2*(x-1)^3")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let roots = p.isolate_roots();

        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0].1, 2);
        assert_eq!(roots[1].1, 3);
        assert_eq!(roots[0].0.defining_polynomial().degree(), 1);
        assert_eq!(roots[0].0.index(), 0);
        assert_eq!(roots[1].0.index(), 0);
        assert!(
            ROOT_CACHE
                .layouts
                .read()
                .unwrap()
                .contains_key(&RootPolynomialKey::Rational(p.coefficients.clone()))
        );
        assert!(
            !ROOT_CACHE
                .roots
                .read()
                .unwrap()
                .contains_key(&RootPolynomialKey::Rational(p.coefficients.clone()))
        );
        assert_eq!(
            p.root(0).unwrap().enclosure().center(),
            p.root(1).unwrap().enclosure().center()
        );
        assert_eq!(
            p.root(2).unwrap().enclosure().center(),
            p.root(4).unwrap().enclosure().center()
        );
    }

    #[test]
    fn complex_root_cache_ignores_variable_name() {
        let p_x = parse!("x^2+1")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let p_y = parse!("y^2+1")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);

        let cache = RootCache::new();
        let mut calls = 0;
        let entry = cache.layout_entry(RootPolynomialKey::Rational(p_x.coefficients.clone()));
        entry.layout.get_or_init(|| {
            calls += 1;
            PolynomialRootLayout { roots: vec![] }
        });
        let entry = cache.layout_entry(RootPolynomialKey::Rational(p_y.coefficients.clone()));
        entry.layout.get_or_init(|| {
            calls += 1;
            PolynomialRootLayout { roots: vec![] }
        });

        assert_eq!(calls, 1);
    }

    #[test]
    fn concurrent_root_cache_miss_computes_once() {
        let polynomial = Arc::new(
            parse!("x^3-2")
                .to_polynomial::<_, u16>(&Q, None)
                .to_univariate_from_univariate(0),
        );
        let cache = Arc::new(RootCache::new());
        let calls = Arc::new(AtomicUsize::new(0));
        let start = Arc::new(Barrier::new(8));

        let threads = (0..8)
            .map(|_| {
                let polynomial = polynomial.clone();
                let cache = cache.clone();
                let calls = calls.clone();
                let start = start.clone();
                std::thread::spawn(move || {
                    start.wait();
                    let entry = cache
                        .layout_entry(RootPolynomialKey::Rational(polynomial.coefficients.clone()));
                    entry.layout.get_or_init(|| {
                        calls.fetch_add(1, AtomicOrdering::Relaxed);
                        std::thread::sleep(Duration::from_millis(25));
                        PolynomialRootLayout { roots: vec![] }
                    });
                })
            })
            .collect::<Vec<_>>();

        for thread in threads {
            thread.join().unwrap();
        }
        assert_eq!(calls.load(AtomicOrdering::Relaxed), 1);
    }

    #[test]
    fn refining_one_polynomial_does_not_lock_other_entries() {
        let p_a = parse!("x^2-2")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let p_b = parse!("x^2-3")
            .to_polynomial::<_, u16>(&Q, None)
            .to_univariate_from_univariate(0);
        let cache = Arc::new(RootCache::new());
        let entry_a = cache.root_entry(RootPolynomialKey::Rational(p_a.coefficients.clone()));
        entry_a
            .roots
            .get_or_init(|| RwLock::new(Vec::<RootState>::new()));
        let entry_b = cache.root_entry(RootPolynomialKey::Rational(p_b.coefficients.clone()));
        entry_b
            .roots
            .get_or_init(|| RwLock::new(Vec::<RootState>::new()));

        let roots_a = entry_a.roots.get().unwrap();
        let _refining_a = roots_a.write().unwrap();

        let (finished_tx, finished_rx) = std::sync::mpsc::channel();
        let worker = {
            let cache = cache.clone();
            std::thread::spawn(move || {
                let entry_b =
                    cache.root_entry(RootPolynomialKey::Rational(p_b.coefficients.clone()));
                let _roots_b = entry_b.roots.get().unwrap().read().unwrap();
                finished_tx.send(()).unwrap();
            })
        };

        finished_rx
            .recv_timeout(Duration::from_millis(250))
            .expect("an unrelated cache entry was blocked");
        worker.join().unwrap();
    }

    #[test]
    fn complex_root_cache_ignores_variable_name_for_complex_coefficients() {
        let field = FloatField::from_rep(Complex::from(Rational::one()));
        let mut p_x = UnivariatePolynomial::new(&field, None, Arc::new(PolyVariable::Temporary(0)));
        p_x.coefficients = vec![
            Complex::new(Rational::from(3), Rational::one()),
            Complex::new(Rational::from(-3), Rational::zero()),
            Complex::new(Rational::one(), Rational::zero()),
        ];
        let mut p_y = UnivariatePolynomial::new(&field, None, Arc::new(PolyVariable::Temporary(1)));
        p_y.coefficients = p_x.coefficients.clone();

        let cache = RootCache::new();
        let mut calls = 0;
        let entry = cache.layout_entry(RootPolynomialKey::Complex(p_x.coefficients.clone()));
        entry.layout.get_or_init(|| {
            calls += 1;
            PolynomialRootLayout { roots: vec![] }
        });
        let entry = cache.layout_entry(RootPolynomialKey::Complex(p_y.coefficients.clone()));
        entry.layout.get_or_init(|| {
            calls += 1;
            PolynomialRootLayout { roots: vec![] }
        });

        assert_eq!(calls, 1);
    }
}
