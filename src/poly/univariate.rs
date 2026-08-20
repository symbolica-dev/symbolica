//! Univariate polynomials and their ring.

#![warn(missing_docs)]

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
        algebraic::{AlgebraicExtension, AlgebraicNumber},
        float::{Complex, F64, FloatField, FloatLike, Real, SingleFloat},
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
    gcd::PolynomialGCD,
    polynomial::{MultivariatePolynomial, PolynomialRing},
};

/// A univariate polynomial ring.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct UnivariatePolynomialRing<R: Ring> {
    ring: R,
    variable: Arc<PolyVariable>,
}

impl<R: Ring> UnivariatePolynomialRing<R> {
    /// Construct a polynomial ring over `coeff_ring` in the supplied variable.
    pub fn new(coeff_ring: R, var_map: Arc<PolyVariable>) -> UnivariatePolynomialRing<R> {
        UnivariatePolynomialRing {
            ring: coeff_ring,
            variable: var_map,
        }
    }

    /// Construct the ring containing `poly`.
    pub fn from_polynomial(poly: &UnivariatePolynomial<R>) -> UnivariatePolynomialRing<R> {
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
    /// Coefficients in ascending degree order. The empty vector represents zero.
    pub(crate) coefficients: Vec<F::Element>,
    /// Variable used by this polynomial.
    pub(crate) variable: Arc<PolyVariable>,
    /// Ring containing the coefficients.
    pub(crate) ring: F,
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
    /// Construct a polynomial from coefficients in ascending degree order.
    ///
    /// Trailing zero coefficients are removed to preserve the canonical dense
    /// representation.
    pub fn from_coefficients(
        field: &F,
        coefficients: Vec<F::Element>,
        variable: Arc<PolyVariable>,
    ) -> Self {
        let mut polynomial = Self {
            coefficients,
            ring: field.clone(),
            variable,
        };
        polynomial.truncate();
        polynomial
    }

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

    /// Return the coefficients in ascending degree order.
    pub fn coefficients(&self) -> &[F::Element] {
        &self.coefficients
    }

    /// Return the coefficient ring.
    pub fn coefficient_ring(&self) -> &F {
        &self.ring
    }

    /// Return whether this is the zero polynomial.
    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    /// Return whether this is the constant polynomial one.
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

    /// Get a shared pointer to the variable.
    pub fn get_vars(&self) -> Arc<PolyVariable> {
        self.variable.clone()
    }

    /// Get a reference to the variable.
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

    /// Remove the lowest `exp` coefficient slots, corresponding to exact
    /// division by the variable to that power when those coefficients are zero.
    fn remove_variable_power(&self, exp: usize) -> Self {
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

    /// Divide exactly by `div`, returning `None` when the division is not exact
    /// or the polynomials use different variables.
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

mod roots;
#[cfg(test)]
mod tests;

pub use roots::{ComplexDisk, ExactComplexPolynomial, IsolatedRoot, RootLocation};

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

            p = p.remove_variable_power(1);
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
            p = p.remove_variable_power(1);
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
                    p = p.remove_variable_power(1);
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

                n1.p = n1.p.remove_variable_power(1);
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
                    n2.p = n2.p.remove_variable_power(1);
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
            match self
                .remove_variable_power(1)
                .roots(max_iterations, tolerance)
            {
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
            match self
                .remove_variable_power(1)
                .roots(max_iterations, tolerance)
            {
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
                let p_at_i = self.evaluate(&n[i]);
                let df_at_i = df.evaluate(&n[i]);
                if !p_at_i.is_finite() || !df_at_i.is_finite() || df_at_i.is_zero() {
                    return finite_error(&n);
                }

                let e = p_at_i / df_at_i;
                if !e.is_finite() {
                    return finite_error(&n);
                }

                let mut rep = e.zero();
                for j in 0..n.len() {
                    if i != j && n[i] != n[j] {
                        let diff = n[i].clone() - &n[j];
                        if !diff.is_finite() || diff.is_zero() {
                            return finite_error(&n);
                        }

                        let diff_inv = diff.inv();
                        if !diff_inv.is_finite() {
                            return finite_error(&n);
                        }

                        rep += diff_inv;
                    }
                }
                if !rep.is_finite() {
                    return finite_error(&n);
                }

                let denom = rep.one() - &e * rep;
                if !denom.is_finite() || denom.is_zero() {
                    return finite_error(&n);
                }

                let correction = e / denom;
                if !correction.is_finite() {
                    return finite_error(&n);
                }

                let updated = n[i].clone() - correction;
                if !updated.is_finite() {
                    return finite_error(&n);
                }
                n[i] = updated;
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

    /// Return the quotient and remainder of division by `div`.
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
    pub fn extended_gcd(&self, other: &Self) -> (Self, Self, Self) {
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
            let (g, s, t) = p.extended_gcd(aa);
            debug_assert!(g.is_one());
            let new_s = (t * &cur_s).rem(p);
            ss.push(new_s);
            cur_s = (s * &cur_s).rem(aa);
        }

        ss.push(cur_s);
        ss
    }

    /// Compute the univariate GCD using Euclid's algorithm. The result is made monic.
    ///
    /// Prefer [`Self::gcd`] when the coefficient ring implements [`PolynomialGCD`]. Exact
    /// Euclidean division can cause severe coefficient swell over fields such as the rationals.
    pub fn gcd_euclidean(&self, b: &Self) -> Self {
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

impl<F: Field + PolynomialGCD<u16>> UnivariatePolynomial<F> {
    /// Compute the GCD using the coefficient ring's preferred polynomial GCD algorithm.
    pub fn gcd(&self, b: &Self) -> Self {
        if self.variable != b.variable {
            panic!("Cannot compute GCD of polynomials with different variables");
        }

        self.clone()
            .to_multivariate::<u16>()
            .gcd(&b.clone().to_multivariate::<u16>())
            .to_univariate_from_univariate(0)
    }
}

impl<R: Ring, E: PositiveExponent> UnivariatePolynomial<PolynomialRing<R, E>> {
    /// Convert a univariate polynomial of multivariate polynomials to a multivariate polynomial.
    pub fn flatten(self) -> MultivariatePolynomial<R, E> {
        if self.is_zero() {
            return self.ring.zero();
        }

        let Some(pos) = self.coefficients[0]
            .variables()
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
