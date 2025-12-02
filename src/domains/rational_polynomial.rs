//! Rational polynomial field.

use std::{
    borrow::Cow,
    cmp::Ordering,
    fmt::{Display, Error},
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::Arc,
};

use ahash::HashMap;

use crate::{
    domains::{
        RingOps, Set,
        algebraic_number::AlgebraicExtension,
        float::{FloatField, SingleFloat},
    },
    poly::{
        GrevLexOrder, LexOrder, PolyVariable, PositiveExponent,
        factor::Factorize,
        gcd::PolynomialGCD,
        groebner::{Echelonize, GroebnerBasis},
        polynomial::{MultivariatePolynomial, PolynomialRing},
        univariate::UnivariatePolynomial,
    },
    printer::{PrintOptions, PrintState},
};

use super::{
    Derivable, EuclideanDomain, Field, InternalOrdering, Ring, SelfRing, UpgradeToField,
    finite_field::{FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField},
    integer::{Integer, IntegerRing, Z},
    rational::RationalField,
};

/// A rational polynomial field.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RationalPolynomialField<R: Ring, E: PositiveExponent> {
    ring: R,
    _phantom_exp: PhantomData<E>,
}

impl<R: Ring, E: PositiveExponent> RationalPolynomialField<R, E> {
    pub fn new(coeff_ring: R) -> RationalPolynomialField<R, E> {
        RationalPolynomialField {
            ring: coeff_ring,
            _phantom_exp: PhantomData,
        }
    }

    pub fn from_poly(poly: &MultivariatePolynomial<R, E>) -> RationalPolynomialField<R, E> {
        RationalPolynomialField {
            ring: poly.ring.clone(),
            _phantom_exp: PhantomData,
        }
    }
}

pub trait FromNumeratorAndDenominator<R: Ring, OR: Ring, E: PositiveExponent> {
    fn from_num_den(
        num: MultivariatePolynomial<R, E>,
        den: MultivariatePolynomial<R, E>,
        field: &OR,
        do_gcd: bool,
    ) -> RationalPolynomial<OR, E>;
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> UpgradeToField
    for PolynomialRing<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    type Upgraded = RationalPolynomialField<R, E>;

    fn upgrade(self) -> RationalPolynomialField<R, E> {
        RationalPolynomialField {
            ring: self.ring,
            _phantom_exp: PhantomData,
        }
    }

    fn upgrade_element(&self, element: <Self as Set>::Element) -> <Self::Upgraded as Set>::Element {
        let one = element.one();
        RationalPolynomial::from_num_den(element, one, &self.ring, false)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct RationalPolynomial<R: Ring, E: PositiveExponent> {
    pub numerator: MultivariatePolynomial<R, E>,
    pub denominator: MultivariatePolynomial<R, E>,
}

impl<R: Ring, E: PositiveExponent> InternalOrdering for RationalPolynomial<R, E> {
    /// An ordering of rational polynomials that has no intuitive meaning.
    fn internal_cmp(&self, other: &Self) -> Ordering {
        self.numerator
            .exponents
            .cmp(&other.numerator.exponents)
            .then_with(|| self.denominator.exponents.cmp(&other.denominator.exponents))
            .then_with(|| {
                self.numerator
                    .coefficients
                    .internal_cmp(&other.numerator.coefficients)
            })
            .then_with(|| {
                self.denominator
                    .coefficients
                    .internal_cmp(&other.denominator.coefficients)
            })
    }
}

impl<R: Ring, E: PositiveExponent> From<MultivariatePolynomial<R, E>> for RationalPolynomial<R, E>
where
    Self: FromNumeratorAndDenominator<R, R, E>,
{
    fn from(poly: MultivariatePolynomial<R, E>) -> Self {
        let d = poly.one();
        let field = poly.ring.clone();
        Self::from_num_den(poly, d, &field, false)
    }
}

impl<R: Ring, E: PositiveExponent> RationalPolynomial<R, E>
where
    Self: FromNumeratorAndDenominator<R, R, E>,
{
    pub fn unify_variables(&mut self, other: &mut Self) {
        assert_eq!(self.numerator.variables, self.denominator.variables);
        assert_eq!(other.numerator.variables, other.denominator.variables);

        if self.numerator.variables == other.numerator.variables {
            return;
        }

        // this may require a new normalization of the denominator
        self.numerator.unify_variables(&mut other.numerator);
        self.denominator.unify_variables(&mut other.denominator);

        *self = Self::from_num_den(
            self.numerator.clone(),
            self.denominator.clone(),
            &self.numerator.ring,
            false,
        );

        *other = Self::from_num_den(
            other.numerator.clone(),
            other.denominator.clone(),
            &other.numerator.ring,
            false,
        );
    }
}

impl<R: Ring, E: PositiveExponent> RationalPolynomial<R, E> {
    pub fn new(field: &R, var_map: Arc<Vec<PolyVariable>>) -> RationalPolynomial<R, E> {
        let num = MultivariatePolynomial::new(field, None, var_map);
        let den = num.one();

        RationalPolynomial {
            numerator: num,
            denominator: den,
        }
    }

    pub fn get_variables(&self) -> &Arc<Vec<PolyVariable>> {
        &self.numerator.variables
    }

    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    pub fn is_constant(&self) -> bool {
        self.numerator.is_constant() && self.denominator.is_constant()
    }

    /// Convert the coefficient from the current field to a finite field.
    pub fn to_finite_field<UField: FiniteFieldWorkspace>(
        &self,
        field: &FiniteField<UField>,
    ) -> RationalPolynomial<FiniteField<UField>, E>
    where
        R::Element: ToFiniteField<UField>,
        FiniteField<UField>: FiniteFieldCore<UField>,
        <FiniteField<UField> as Set>::Element: Copy,
    {
        // check the gcd, since the rational polynomial may simplify
        RationalPolynomial::from_num_den(
            self.numerator
                .map_coeff(|c| c.to_finite_field(field), field.clone()),
            self.denominator
                .map_coeff(|c| c.to_finite_field(field), field.clone()),
            field,
            true,
        )
    }
}

impl<R: Ring, E: PositiveExponent> SelfRing for RationalPolynomial<R, E> {
    fn is_zero(&self) -> bool {
        self.is_zero()
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
        if opts.explicit_rational_polynomial {
            if state.in_sum {
                f.write_char('+')?;
            }

            if self.denominator.is_one() {
                if self.numerator.is_zero() {
                    f.write_char('0')?;
                } else {
                    f.write_char('[')?;
                    self.numerator.format(opts, PrintState::new(), f)?;
                    f.write_char(']')?;
                }
            } else {
                f.write_char('[')?;
                self.numerator.format(opts, PrintState::new(), f)?;
                f.write_char(',')?;
                self.denominator.format(opts, PrintState::new(), f)?;
                f.write_char(']')?;
            }

            return Ok(false);
        }

        if self.denominator.is_one() {
            self.numerator.format(opts, state, f)
        } else {
            let write_par = state.in_exp || state.in_exp_base;
            if write_par {
                if state.in_sum {
                    state.in_sum = false;
                    f.write_char('+')?;
                }

                f.write_char('(')?;
                state.in_exp = false;
                state.in_exp_base = false;
            }

            if opts.mode.is_latex() {
                if state.in_sum {
                    f.write_char('+')?;
                }
                f.write_str("\\frac{")?;
                self.numerator.format(opts, PrintState::new(), f)?;
                f.write_str("}{")?;
                self.denominator.format(opts, PrintState::new(), f)?;
                f.write_str("}")?;
            } else {
                state.suppress_one = false;
                self.numerator
                    .format(opts, state.step(state.in_sum, true, false, false), f)?;
                f.write_char('/')?;
                self.denominator
                    .format(opts, state.step(false, false, false, true), f)?;
            }

            if write_par {
                f.write_char(')')?;
            }
            Ok(false)
        }
    }
}

impl<F: Field, E: PositiveExponent>
    FromNumeratorAndDenominator<AlgebraicExtension<F>, AlgebraicExtension<F>, E>
    for RationalPolynomial<AlgebraicExtension<F>, E>
where
    AlgebraicExtension<F>: Field + PolynomialGCD<E>,
{
    fn from_num_den(
        mut num: MultivariatePolynomial<AlgebraicExtension<F>, E>,
        mut den: MultivariatePolynomial<AlgebraicExtension<F>, E>,
        field: &AlgebraicExtension<F>,
        do_gcd: bool,
    ) -> RationalPolynomial<AlgebraicExtension<F>, E> {
        num.unify_variables(&mut den);

        if den.is_one() {
            RationalPolynomial {
                numerator: num,
                denominator: den,
            }
        } else {
            if do_gcd {
                let gcd = num.gcd(&den);

                if !gcd.is_one() {
                    num = num / &gcd;
                    den = den / &gcd;
                }
            }

            // normalize denominator to have positive leading coefficient
            let d = den.lcoeff();
            if !field.is_one(&d) {
                let c = field.inv(&d);
                num = num.mul_coeff(c.clone());
                den = den.mul_coeff(c);
            }

            RationalPolynomial {
                numerator: num,
                denominator: den,
            }
        }
    }
}

impl<E: PositiveExponent> FromNumeratorAndDenominator<RationalField, IntegerRing, E>
    for RationalPolynomial<IntegerRing, E>
{
    fn from_num_den(
        num: MultivariatePolynomial<RationalField, E>,
        den: MultivariatePolynomial<RationalField, E>,
        field: &IntegerRing,
        do_gcd: bool,
    ) -> RationalPolynomial<IntegerRing, E> {
        let content = num.ring.gcd(&num.content(), &den.content());

        let mut num_int = MultivariatePolynomial::new(&Z, None, num.variables);
        num_int.exponents = num.exponents;

        let mut den_int = MultivariatePolynomial::new(&Z, Some(den.nterms()), den.variables);
        den_int.exponents = den.exponents;

        if num.ring.is_one(&content) {
            num_int.coefficients = num
                .coefficients
                .into_iter()
                .map(|c| c.numerator())
                .collect();
            den_int.coefficients = den
                .coefficients
                .into_iter()
                .map(|c| c.numerator())
                .collect();
        } else {
            num_int.coefficients = num
                .coefficients
                .into_iter()
                .map(|c| num.ring.div(&c, &content).numerator())
                .collect();
            den_int.coefficients = den
                .coefficients
                .into_iter()
                .map(|c| den.ring.div(&c, &content).numerator())
                .collect();
        }

        <RationalPolynomial<IntegerRing, E> as FromNumeratorAndDenominator<
            IntegerRing,
            IntegerRing,
            E,
        >>::from_num_den(num_int, den_int, field, do_gcd)
    }
}

impl<E: PositiveExponent> FromNumeratorAndDenominator<IntegerRing, IntegerRing, E>
    for RationalPolynomial<IntegerRing, E>
{
    fn from_num_den(
        mut num: MultivariatePolynomial<IntegerRing, E>,
        mut den: MultivariatePolynomial<IntegerRing, E>,
        _field: &IntegerRing,
        do_gcd: bool,
    ) -> Self {
        num.unify_variables(&mut den);

        if den.is_one() {
            RationalPolynomial {
                numerator: num,
                denominator: den,
            }
        } else {
            if do_gcd {
                let gcd = num.gcd(&den);

                if !gcd.is_one() {
                    num = num / &gcd;
                    den = den / &gcd;
                }
            }

            // normalize denominator to have positive leading coefficient
            if den.lcoeff().is_negative() {
                num = -num;
                den = -den;
            }

            RationalPolynomial {
                numerator: num,
                denominator: den,
            }
        }
    }
}

impl<UField: FiniteFieldWorkspace, E: PositiveExponent>
    FromNumeratorAndDenominator<FiniteField<UField>, FiniteField<UField>, E>
    for RationalPolynomial<FiniteField<UField>, E>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
    <FiniteField<UField> as Set>::Element: Copy,
{
    fn from_num_den(
        mut num: MultivariatePolynomial<FiniteField<UField>, E>,
        mut den: MultivariatePolynomial<FiniteField<UField>, E>,
        field: &FiniteField<UField>,
        do_gcd: bool,
    ) -> Self {
        num.unify_variables(&mut den);

        if den.is_one() {
            RationalPolynomial {
                numerator: num,
                denominator: den,
            }
        } else {
            if do_gcd {
                let gcd = num.gcd(&den);

                if !gcd.is_one() {
                    num = num / &gcd;
                    den = den / &gcd;
                }
            }

            // normalize denominator to have leading coefficient of one
            if !field.is_one(&den.lcoeff()) {
                let c = den.lcoeff();
                num = num.div_coeff(&c);
                den = den.div_coeff(&c);
            }

            RationalPolynomial {
                numerator: num,
                denominator: den,
            }
        }
    }
}

impl<T: SingleFloat + std::hash::Hash + Eq + InternalOrdering, E: PositiveExponent>
    FromNumeratorAndDenominator<FloatField<T>, FloatField<T>, E>
    for RationalPolynomial<FloatField<T>, E>
{
    fn from_num_den(
        mut num: MultivariatePolynomial<FloatField<T>, E>,
        mut den: MultivariatePolynomial<FloatField<T>, E>,
        field: &FloatField<T>,
        _do_gcd: bool,
    ) -> Self {
        num.unify_variables(&mut den);

        if den.is_one() {
            RationalPolynomial {
                numerator: num,
                denominator: den,
            }
        } else {
            // normalize denominator to have leading coefficient of one
            if !field.is_one(&den.lcoeff()) {
                let c = den.lcoeff();
                num = num.div_coeff(&c);
                den = den.div_coeff(&c);
            }

            RationalPolynomial {
                numerator: num,
                denominator: den,
            }
        }
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> RationalPolynomial<R, E>
where
    Self: FromNumeratorAndDenominator<R, R, E>,
{
    #[inline]
    pub fn inv(self) -> Self {
        if self.numerator.is_zero() {
            panic!("Cannot invert 0");
        }

        let field = self.numerator.ring.clone();
        Self::from_num_den(self.denominator, self.numerator, &field, false)
    }

    pub fn pow(&self, e: u64) -> Self {
        if e > u32::MAX as u64 {
            panic!("Power of exponentiation is larger than 2^32: {e}");
        }
        let e = e as u32;

        // TODO: do binary exponentiation
        let mut poly = RationalPolynomial {
            numerator: self.numerator.one(),
            denominator: self.denominator.one(),
        };

        for _ in 0..e {
            poly = &poly * self;
        }
        poly
    }

    pub fn gcd(&self, other: &Self) -> Self {
        let gcd_num = self.numerator.gcd(&other.numerator);
        let gcd_den = self.denominator.gcd(&other.denominator);

        RationalPolynomial {
            numerator: gcd_num,
            denominator: (&other.denominator / &gcd_den) * &self.denominator,
        }
    }

    /// Convert the rational polynomial to a polynomial in the specified
    /// variables, with rational polynomial coefficients.
    /// If the specified variables appear in the denominator, an `Err` is returned.
    ///
    /// If `ignore_denominator` is `True`, the denominator is considered to be 1,
    /// after the variable check.
    pub fn to_polynomial(
        &self,
        variables: &[PolyVariable],
        ignore_denominator: bool,
    ) -> Result<MultivariatePolynomial<RationalPolynomialField<R, E>, E>, &'static str> {
        let index_mask: Vec<_> = self
            .numerator
            .variables
            .iter()
            .map(|v| variables.iter().position(|vv| vv == v))
            .collect();

        let coefficient_vars = Arc::new(
            self.numerator
                .variables
                .iter()
                .zip(&index_mask)
                .filter_map(|(var, mask)| {
                    if mask.is_none() {
                        Some(var.clone())
                    } else {
                        None
                    }
                })
                .collect::<Vec<_>>(),
        );

        if self.denominator.nvars() > 0 {
            for e in self.denominator.exponents_iter() {
                for (c, p) in index_mask.iter().zip(e) {
                    if c.is_some() && *p > E::zero() {
                        return Err("Not a polynomial");
                    }
                }
            }
        }

        let mut hm: HashMap<Vec<E>, RationalPolynomial<R, E>> = HashMap::default();

        let mut e_list = vec![E::zero(); variables.len()];

        let mut e_list_coeff = vec![E::zero(); coefficient_vars.len()];
        for e in self.numerator.into_iter() {
            for ee in &mut e_list {
                *ee = E::zero();
            }

            let mut elc_index = 0;
            for (ee, m) in e.exponents.iter().zip(&index_mask) {
                if let Some(p) = m {
                    e_list[*p] = *ee;
                } else {
                    e_list_coeff[elc_index] = *ee;
                    elc_index += 1;
                }
            }

            if let Some(r) = hm.get_mut(e_list.as_slice()) {
                r.numerator
                    .append_monomial(e.coefficient.clone(), &e_list_coeff);
            } else {
                let mut r =
                    RationalPolynomial::new(&self.numerator.ring.clone(), coefficient_vars.clone());
                r.numerator
                    .append_monomial(e.coefficient.clone(), &e_list_coeff);
                hm.insert(e_list.clone(), r);
            }
        }

        let v = Arc::new(variables.to_vec());
        let field = RationalPolynomialField::new(self.numerator.ring.clone());
        let mut poly = MultivariatePolynomial::new(&field, Some(hm.len()), v);

        if !ignore_denominator {
            let denom = RationalPolynomial::from_num_den(
                self.denominator.one(),
                self.denominator.clone(),
                &self.denominator.ring,
                false,
            );

            for coeff in hm.values_mut() {
                // divide by the denominator
                *coeff = coeff.mul(&denom);
            }
        }

        for (exp, coeff) in hm {
            poly.append_monomial(coeff, &exp);
        }

        Ok(poly)
    }

    // Convert from a univariate polynomial with rational polynomial coefficients to a rational polynomial.
    pub fn from_univariate(
        mut f: UnivariatePolynomial<RationalPolynomialField<R, E>>,
    ) -> RationalPolynomial<R, E> {
        if f.is_zero() {
            return RationalPolynomial {
                numerator: MultivariatePolynomial::new_zero(&f.ring.ring),
                denominator: MultivariatePolynomial::new_one(&f.ring.ring),
            };
        }

        let pos = f.coefficients[0]
            .get_variables()
            .iter()
            .position(|x| x == f.variable.as_ref())
            .unwrap_or_else(|| {
                let mut r = RationalPolynomial::new(
                    &f.coefficients[0].numerator.ring,
                    Arc::new(vec![f.variable.as_ref().clone()]),
                );
                for c in &mut f.coefficients {
                    c.unify_variables(&mut r);
                }

                f.coefficients[0].get_variables().len() - 1
            });

        let mut res =
            RationalPolynomial::new(&f.ring.ring, f.coefficients[0].get_variables().clone());

        let mut exp = vec![E::zero(); f.coefficients[0].get_variables().len()];
        exp[pos] = E::one();
        let v: RationalPolynomial<R, E> =
            res.numerator.monomial(res.numerator.ring.one(), exp).into();

        for (p, c) in f.coefficients.into_iter().enumerate() {
            res = &res + &(&v.pow(p as u64) * &c);
        }

        res
    }
}

impl<R: Field, E: PositiveExponent> RationalPolynomial<R, E> {
    /// Evaluate the rational polynomial.
    pub fn evaluate(&self, x: &[R::Element]) -> R::Element {
        let num = self.numerator.replace_all(x);
        let den = self.denominator.replace_all(x);
        self.numerator.ring.div(&num, &den)
    }
}

impl<R: Ring, E: PositiveExponent> RationalPolynomial<R, E> {
    /// Evaluate the rational polynomial at the given point, mapping coefficients to the ring `U`.
    pub fn evaluate_with_coeff_map<U: Field, T: Fn(&R::Element) -> U::Element + Clone>(
        &self,
        map_coeff: T,
        point: &[U::Element],
        ring: &U,
    ) -> U::Element {
        let num_eval = self
            .numerator
            .evaluate_with_coeff_map(map_coeff.clone(), point, ring);
        let den_eval = self
            .denominator
            .evaluate_with_coeff_map(map_coeff, point, ring);
        ring.div(&num_eval, &den_eval)
    }
}

impl<R: Ring, E: PositiveExponent> Display for RationalPolynomial<R, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format(&PrintOptions::from_fmt(f), PrintState::from_fmt(f), f)
            .map(|_| ())
    }
}

impl<R: Ring, E: PositiveExponent> Display for RationalPolynomialField<R, E> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> Set
    for RationalPolynomialField<R, E>
{
    type Element = RationalPolynomial<R, E>;

    fn size(&self) -> Option<Integer> {
        None
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> RingOps<RationalPolynomial<R, E>>
    for RationalPolynomialField<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    fn add(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        &a + &b
    }

    fn sub(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        // TODO: optimize
        self.add(a, self.neg(b))
    }

    fn mul(&self, a: Self::Element, b: Self::Element) -> Self::Element {
        &a * &b
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
        self.add_assign(a, &(&b * &c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: Self::Element, c: Self::Element) {
        self.sub_assign(a, &(&b * &c));
    }

    fn neg(&self, a: Self::Element) -> Self::Element {
        a.neg()
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> RingOps<&RationalPolynomial<R, E>>
    for RationalPolynomialField<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a + b
    }

    fn sub(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        // TODO: optimize
        self.add(a, &self.neg(b))
    }

    fn mul(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a * b
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
        self.add_assign(a, &(b * c));
    }

    fn sub_mul_assign(&self, a: &mut Self::Element, b: &Self::Element, c: &Self::Element) {
        self.sub_assign(a, &(b * c));
    }

    fn neg(&self, a: &Self::Element) -> Self::Element {
        a.clone().neg()
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> Ring
    for RationalPolynomialField<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    fn zero(&self) -> Self::Element {
        let num = MultivariatePolynomial::new_zero(&self.ring);
        RationalPolynomial {
            denominator: num.one(),
            numerator: num,
        }
    }

    fn one(&self) -> Self::Element {
        let num = MultivariatePolynomial::new_zero(&self.ring).one();
        RationalPolynomial {
            numerator: num.clone(),
            denominator: num,
        }
    }

    fn nth(&self, n: Integer) -> Self::Element {
        let mut r = self.one();
        r.numerator = r.numerator.mul_coeff(self.ring.nth(n));
        r
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        if e > u32::MAX as u64 {
            panic!("Power of exponentiation is larger than 2^32: {e}");
        }
        let e = e as u32;

        // TODO: do binary exponentiation
        let mut poly = RationalPolynomial {
            numerator: b.numerator.zero(),
            denominator: b.denominator.zero(),
        };
        poly.numerator = poly.numerator.add_constant(self.ring.one());
        poly.denominator = poly.denominator.add_constant(self.ring.one());

        for _ in 0..e {
            poly = self.mul(&poly, b);
        }
        poly
    }

    fn is_zero(&self, a: &Self::Element) -> bool {
        a.numerator.is_zero()
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        a.numerator.is_one() && a.denominator.is_one()
    }

    fn one_is_gcd_unit() -> bool {
        false
    }

    fn characteristic(&self) -> Integer {
        self.ring.characteristic()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if a.is_zero() { None } else { Some(self.inv(a)) }
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if b.is_zero() {
            None
        } else {
            Some(self.div(a, b))
        }
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
    ) -> Result<bool, Error> {
        element.format(opts, state, f)
    }

    fn has_independent_elements(&self) -> bool {
        true
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> EuclideanDomain
    for RationalPolynomialField<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    fn rem(&self, a: &Self::Element, _: &Self::Element) -> Self::Element {
        RationalPolynomial {
            numerator: a.numerator.zero(),
            denominator: a.numerator.one(),
        }
    }

    fn quot_rem(&self, a: &Self::Element, b: &Self::Element) -> (Self::Element, Self::Element) {
        (self.div(a, b), self.zero())
    }

    fn gcd(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a.gcd(b)
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> Field
    for RationalPolynomialField<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    fn div(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        a / b
    }

    fn div_assign(&self, a: &mut Self::Element, b: &Self::Element) {
        *a = self.div(a, b);
    }

    fn inv(&self, a: &Self::Element) -> Self::Element {
        a.clone().inv()
    }
}

impl<'a, R: EuclideanDomain + PolynomialGCD<E> + PolynomialGCD<E>, E: PositiveExponent>
    Add<&'a RationalPolynomial<R, E>> for &RationalPolynomial<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    type Output = RationalPolynomial<R, E>;

    fn add(self, other: &'a RationalPolynomial<R, E>) -> Self::Output {
        if self.get_variables() != other.get_variables() {
            let mut a = self.clone();
            let mut b = other.clone();
            a.unify_variables(&mut b);
            return &a + &b;
        }

        let denom_gcd = self.denominator.gcd(&other.denominator);

        let mut a_denom_red = Cow::Borrowed(&self.denominator);
        let mut b_denom_red = Cow::Borrowed(&other.denominator);

        if !denom_gcd.is_one() {
            a_denom_red = Cow::Owned(&self.denominator / &denom_gcd);
            b_denom_red = Cow::Owned(&other.denominator / &denom_gcd);
        }

        let num1 = &self.numerator * &b_denom_red;
        let num2 = &other.numerator * &a_denom_red;
        let mut num = num1 + num2;

        // prefer small * large over medium * medium sized polynomials
        let mut den = if self.denominator.nterms() > other.denominator.nterms()
            && self.denominator.nterms() > a_denom_red.nterms()
        {
            b_denom_red.as_ref() * &self.denominator
        } else {
            a_denom_red.as_ref() * &other.denominator
        };

        let g = num.gcd(&denom_gcd);

        if !g.is_one() {
            num = num / &g;
            den = den / &g;
        }

        RationalPolynomial {
            numerator: num,
            denominator: den,
        }
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> Sub for RationalPolynomial<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.add(&other.neg())
    }
}

impl<'a, R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>
    Sub<&'a RationalPolynomial<R, E>> for &RationalPolynomial<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    type Output = RationalPolynomial<R, E>;

    fn sub(self, other: &'a RationalPolynomial<R, E>) -> Self::Output {
        self.add(&other.clone().neg())
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> Neg for RationalPolynomial<R, E> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        RationalPolynomial {
            numerator: self.numerator.neg(),
            denominator: self.denominator,
        }
    }
}

impl<'a, R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>
    Mul<&'a RationalPolynomial<R, E>> for &RationalPolynomial<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    type Output = RationalPolynomial<R, E>;

    fn mul(self, other: &'a RationalPolynomial<R, E>) -> Self::Output {
        if self.get_variables() != other.get_variables() {
            let mut a = self.clone();
            let mut b = other.clone();
            a.unify_variables(&mut b);
            return &a * &b;
        }

        let gcd1 = self.numerator.gcd(&other.denominator);
        let gcd2 = self.denominator.gcd(&other.numerator);

        if gcd1.is_one() {
            if gcd2.is_one() {
                RationalPolynomial {
                    numerator: &self.numerator * &other.numerator,
                    denominator: &self.denominator * &other.denominator,
                }
            } else {
                RationalPolynomial {
                    numerator: &self.numerator * &(&other.numerator / &gcd2),
                    denominator: (&self.denominator / &gcd2) * &other.denominator,
                }
            }
        } else if gcd2.is_one() {
            RationalPolynomial {
                numerator: (&self.numerator / &gcd1) * &other.numerator,
                denominator: &self.denominator * &(&other.denominator / &gcd1),
            }
        } else {
            RationalPolynomial {
                numerator: (&self.numerator / &gcd1) * &(&other.numerator / &gcd2),
                denominator: (&self.denominator / &gcd2) * &(&other.denominator / &gcd1),
            }
        }
    }
}

impl<'a, R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>
    Div<&'a RationalPolynomial<R, E>> for &RationalPolynomial<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    type Output = RationalPolynomial<R, E>;

    fn div(self, other: &'a RationalPolynomial<R, E>) -> Self::Output {
        // TODO: optimize
        self * &other.clone().inv()
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> RationalPolynomial<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    /// Compute the derivative of the rational polynomial in `var`.
    pub fn derivative(&self, var: usize) -> Self {
        if self.numerator.degree(var) == E::zero() && self.denominator.degree(var) == E::zero() {
            return RationalPolynomial {
                numerator: self.numerator.zero(),
                denominator: self.denominator.one(),
            };
        }

        let dn = self.numerator.derivative(var);
        let dd = self.denominator.derivative(var);

        let a = RationalPolynomial::from_num_den(
            dn,
            self.denominator.clone(),
            &self.numerator.ring,
            false,
        );
        let b = RationalPolynomial::from_num_den(
            &self.numerator * &dd,
            &self.denominator * &self.denominator,
            &self.numerator.ring,
            false,
        );

        &a - &b
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> Derivable
    for RationalPolynomialField<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
{
    type Variable = PolyVariable;

    fn derivative(
        &self,
        p: &RationalPolynomial<R, E>,
        x: &PolyVariable,
    ) -> RationalPolynomial<R, E> {
        if let Some(pos) = p.get_variables().iter().position(|v| v == x) {
            p.derivative(pos)
        } else {
            self.zero()
        }
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> RationalPolynomial<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
{
    /// Compute the partial fraction decomposition of the rational polynomial in `var`.
    pub fn apart(&self, var: usize) -> Vec<Self> {
        if self.denominator.degree(var) == E::zero() {
            return vec![self.clone()];
        }

        let rat_field = RationalPolynomialField::from_poly(&self.numerator);
        let n = self
            .numerator
            .to_univariate(var)
            .map_coeff(|c| c.clone().into(), rat_field.clone());

        let mut hs = vec![];

        let rem = if self.numerator.degree(var) >= self.denominator.degree(var) {
            let d = self
                .denominator
                .to_univariate(var)
                .map_coeff(|c| c.clone().into(), rat_field.clone());
            let (q, r) = n.quot_rem(&d);
            if !q.is_zero() {
                hs.push(Self::from_univariate(q));
            }
            r
        } else {
            n
        };

        // partial fraction the denominator
        let mut fs = self
            .denominator
            .factor()
            .into_iter()
            .map(|(x, p)| (x.to_univariate(var), p))
            .collect::<Vec<_>>();

        let mut constant = fs[0].0.one();
        fs.retain(|x| {
            if x.0.is_constant() {
                constant = &constant * &x.0.pow(x.1);
                false
            } else {
                true
            }
        });

        let constant =
            Self::from_univariate(constant.map_coeff(|c| c.clone().into(), rat_field.clone()))
                .inv();

        assert!(!fs.is_empty());

        let mut expanded = fs
            .iter()
            .map(|(f, p)| f.pow(*p).map_coeff(|c| c.clone().into(), rat_field.clone()))
            .collect::<Vec<_>>();

        // perform partial fractioning
        let deltas = if expanded.len() > 1 {
            UnivariatePolynomial::diophantine(&mut expanded, &rem)
        } else {
            vec![rem]
        };

        let fs = fs
            .into_iter()
            .map(|(x, e)| (x.map_coeff(|c| c.clone().into(), rat_field.clone()), e))
            .collect::<Vec<_>>();

        for (d, (p, p_pow)) in deltas.into_iter().zip(fs) {
            let exp = d.p_adic_expansion(&p);
            let p_rat = Self::from_univariate(p);
            for (pow, d_exp) in exp.into_iter().enumerate() {
                hs.push(
                    &(&Self::from_univariate(d_exp) / &p_rat.pow(p_pow as u64 - pow as u64))
                        * &constant,
                );
            }
        }

        hs
    }
}

impl<R: EuclideanDomain + UpgradeToField + PolynomialGCD<E>, E: PositiveExponent>
    RationalPolynomial<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<<R as UpgradeToField>::Upgraded, R, E>,
    <R as UpgradeToField>::Upgraded: Echelonize,
    MultivariatePolynomial<R, E>: Factorize,
{
    /// Compute the multivariate partial fraction decomposition.
    /// Based on [MultivariateApart](https://arxiv.org/abs/2101.08283v1) by Heller and von Manteuffel.
    pub fn apart_multivariate(&self) -> Vec<Self> {
        let mut fs = self.denominator.factor();

        // sort by number of variables
        // TODO: block sort
        fs.sort_by_key(|(f, _)| {
            std::cmp::Reverse((0..f.nvars()).filter(|v| f.contains(*v)).count())
        });

        let mut vars = (0..fs.len())
            .map(PolyVariable::Temporary)
            .collect::<Vec<_>>();
        for v in self.numerator.get_vars_ref() {
            vars.push(v.clone());
        }

        let mut r = self.numerator.rearrange_with_growth(&vars).unwrap();
        let mut exp = vec![E::zero(); vars.len()];
        let mut system = Vec::with_capacity(fs.len());
        for (var, (f, p)) in fs.iter().enumerate() {
            exp[var] = E::from_u32(*p as u32);
            r = r.mul_exp(&exp);

            exp[var] = E::one();
            let f = f
                .rearrange_with_growth(&vars)
                .unwrap()
                .mul_exp(&exp)
                .add_constant(f.ring.neg(&f.ring.one()));

            system.push(f.reorder::<GrevLexOrder>().map_coeff(
                |c| f.ring.upgrade_element(c.clone()),
                f.ring.clone().upgrade(),
            ));
            exp[var] = E::zero();
        }

        let g = GroebnerBasis::new(&system, false);
        let res = r
            .map_coeff(
                |c| r.ring.upgrade_element(c.clone()),
                r.ring.clone().upgrade(),
            )
            .reorder::<GrevLexOrder>()
            .reduce(&g.system)
            .reorder::<LexOrder>();

        let fc = fs
            .iter()
            .map(|(f, _)| {
                f.map_coeff(
                    |c| r.ring.upgrade_element(c.clone()),
                    r.ring.clone().upgrade(),
                )
            })
            .collect::<Vec<_>>();

        let mut parts = vec![];
        let mut exp = vec![E::zero(); self.numerator.nvars()];
        for term in &res {
            let mut num = self
                .numerator
                .map_coeff(
                    |c| r.ring.upgrade_element(c.clone()),
                    r.ring.clone().upgrade(),
                )
                .constant(term.coefficient.clone());
            let mut den = num.one();
            for (i, v) in term.exponents.iter().enumerate() {
                if *v != E::zero() {
                    if i < fc.len() {
                        if *v == E::one() {
                            den = den * &fc[i];
                        } else {
                            den = den * &fc[i].pow(v.to_u32() as usize);
                        }
                    } else {
                        exp[i - fc.len()] = *v;
                        num = num.mul_exp(&exp);
                        exp[i - fc.len()] = E::zero();
                    }
                }
            }
            parts.push(Self::from_num_den(num, den, &self.numerator.ring, false));
        }

        parts
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> RationalPolynomial<R, E>
where
    RationalPolynomial<R, E>: FromNumeratorAndDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
{
    /// Integrate the rational function in `var`. It returns a tuple
    /// `(ps, ls)` where `ps` should be interpreted as the sum of the rational parts
    /// and `ls` as a sum of logarithmic parts. Each logarithmic part is a tuple `(r, a)`
    /// that represents `sum_(r(z) = 0) z*log(a(var, z))` if `r` is dependent on `z`,
    /// else it is `r*log(a)`.
    pub fn integrate(&self, var: usize) -> (Vec<Self>, Vec<(Self, Self)>) {
        let rat_field = RationalPolynomialField::from_poly(&self.numerator);
        let n = self
            .numerator
            .to_univariate(var)
            .map_coeff(|c| c.clone().into(), rat_field.clone());

        let d = self
            .denominator
            .to_univariate(var)
            .map_coeff(|c| c.clone().into(), rat_field.clone());

        if d.is_constant() {
            let r = &Self::from_univariate(n.integrate()) / &Self::from_univariate(d);
            return (vec![r], vec![]);
        }

        let (q, r) = n.quot_rem(&d);

        let mut v = if q.is_zero() {
            vec![]
        } else {
            let n_conv = q.map_coeff(|c| c.clone(), rat_field.clone());
            vec![Self::from_univariate(n_conv.integrate())]
        };

        // partial fraction the denominator
        let mut fs = self
            .denominator
            .square_free_factorization()
            .into_iter()
            .map(|(x, p)| (x.to_univariate(var), p))
            .collect::<Vec<_>>();

        let mut constant = fs[0].0.one();
        fs.retain(|x| {
            if x.0.is_constant() {
                constant = &constant * &x.0.pow(x.1);
                false
            } else {
                true
            }
        });

        let mut constant =
            Self::from_univariate(constant.map_coeff(|c| c.clone().into(), rat_field.clone()))
                .inv();

        assert!(!fs.is_empty());

        let mut expanded = fs
            .iter()
            .map(|(f, p)| f.pow(*p).map_coeff(|c| c.clone().into(), rat_field.clone()))
            .collect::<Vec<_>>();

        let rem = r.map_coeff(|c| c.clone(), rat_field.clone());

        // perform partial fractioning
        let deltas = if expanded.len() > 1 {
            UnivariatePolynomial::diophantine(&mut expanded, &rem)
        } else {
            vec![rem]
        };

        let fs = fs
            .into_iter()
            .map(|(x, e)| (x.map_coeff(|c| c.clone().into(), rat_field.clone()), e))
            .collect::<Vec<_>>();

        let mut hs = vec![];
        for (d, (p, p_pow)) in deltas.into_iter().zip(&fs) {
            let p_diff = p.derivative();
            let (_, s, t) = p.eea(&p_diff);

            let p_full = Self::from_univariate(p.clone());

            let mut d_exp = d.p_adic_expansion(p);

            // grow to the same level as the pow
            if d_exp.len() < *p_pow {
                d_exp.resize(*p_pow, d.zero());
            }

            // highest degree in 1/p last
            d_exp.reverse();

            // perform Hermite reduction
            for i in (1..d_exp.len()).rev() {
                let dd = d_exp[i].clone();
                let s_cor = s.clone() * &dd;
                let t_cor = t.clone() * &dd;

                d_exp[i - 1] = d_exp[i - 1].clone()
                    + s_cor
                    + t_cor.derivative().div_coeff(&(t_cor.ring.nth(i.into())));

                let t_full = Self::from_univariate(t_cor);

                let r = -(&t_full / &(&rat_field.nth(i.into()) * &p_full.pow(i as u64)));
                if !r.is_zero() {
                    v.push(r);
                }
            }

            if !d_exp[0].is_zero() {
                hs.push(d_exp.swap_remove(0));
            }
        }

        // create new temporary variable
        let mut t = MultivariatePolynomial::new(
            &self.numerator.ring,
            None,
            Arc::new(vec![PolyVariable::Temporary(0)]),
        )
        .monomial(self.numerator.ring.one(), vec![E::one()])
        .into();

        let mut w = vec![];

        for (mut h, (mut p, _)) in hs.into_iter().zip(fs.into_iter()) {
            for c in &mut p.coefficients {
                c.unify_variables(&mut t);
            }
            for c in &mut h.coefficients {
                c.unify_variables(&mut t);
            }

            constant.unify_variables(&mut t);

            let new_var = p.coefficients[0].numerator.nvars() - 1;

            let b = h.clone() - p.derivative().mul_coeff(&t);

            // TODO: use resultant_prs instead?
            let r = p.resultant(&b);

            // drop the denominator as it is constant in x
            let mut sqf = r.numerator.square_free_factorization();
            sqf.retain(|(x, _)| !x.is_constant());

            let factors: Vec<(Vec<_>, _, _)> = sqf
                .into_iter()
                .map(|(s, p)| (s.factor().into_iter().map(|x| x.0).collect(), s, p))
                .collect();

            // TODO: if there is only one factor, we know there will be no merging and we can give the result
            // in terms of a root sum of the original denominator instead of the more complicated one
            // this requires returning 3 terms: the root to solve for, the residue and the log content

            // perform monic euclidean algorithm and collect the remainders of the powers that appear in the factorized resultant
            // TODO: this may be wrong, see
            // A Note on Subresultants and the Lazard/Rioboo/Trager Formula in Rational Function Integration
            let mut prs = vec![];
            prs.push(p.clone().make_monic());
            prs.push(b.clone().make_monic());
            while !prs.last().unwrap().is_zero() {
                let r = prs[prs.len() - 2].rem(&prs[prs.len() - 1]);
                if r.lcoeff().is_zero() {
                    break;
                }
                prs.push(r.make_monic());
            }

            for r in &prs {
                let Some((xs, sqf, _)) = factors.iter().find(|(_, _, pp)| r.degree() == *pp) else {
                    continue;
                };

                // write the polynomial as a polynomial in x and t only with rational polynomial
                // coefficients in all other variables
                // since we will make the polynomial monic, the denominator drops out
                let ll = RationalPolynomial::from_univariate(r.clone())
                    .numerator
                    .to_multivariate_polynomial_list(&[var, new_var], true);

                let mut bivar_poly = MultivariatePolynomial::new(
                    &p.ring,
                    Some(ll.len()),
                    p.coefficients[0].get_variables().clone(),
                );
                for (e, p) in ll {
                    bivar_poly.append_monomial(p.into(), &e);
                }

                // convert defining polynomial to a univariate polynomial in t with rational polynomial coefficients
                let def_uni = sqf
                    .to_univariate(new_var)
                    .map_coeff(|c| c.clone().into(), p.ring.clone());

                // write the polynomial in x and t as a polynomial in x with rational polynomial coefficients in t and
                // all other variables and solve a diophantine equation
                let lcoeff = bivar_poly.to_univariate(var).lcoeff();
                let aa = lcoeff.to_univariate_from_univariate(new_var);
                let (_, s, _) = aa.eea(&def_uni);

                // convert the s to a multivariate polynomial and multiply with the bivariate polynomial
                // and mod with the defining polynomial
                // this is equivalent to making the bivariate polynomial monic in Q[t][x]
                // TODO: write conversion routine
                let mut ss = bivar_poly.zero();
                let mut exp = vec![E::zero(); bivar_poly.nvars()];
                for (e, p) in s.coefficients.into_iter().enumerate() {
                    exp[new_var] = E::from_u32(e as u32);
                    ss.append_monomial(p, &exp);
                }

                let bivar_poly_scaled = bivar_poly * &ss;

                let mut def_biv = lcoeff.zero();
                for (e, p) in def_uni.coefficients.into_iter().enumerate() {
                    exp[new_var] = E::from_u32(e as u32);
                    def_biv.append_monomial(p, &exp);
                }

                let monic = bivar_poly_scaled.rem(&def_biv);

                // convert the result to a multivariate rational polynomial
                let mut res = p.ring.zero();
                for t in &monic {
                    let mut exp = vec![E::zero(); p.lcoeff().numerator.nvars()];
                    exp.copy_from_slice(t.exponents);
                    let mm = p.coefficients[0]
                        .numerator
                        .monomial(self.numerator.ring.one(), exp);
                    res = &res + &(t.coefficient * &mm.into());
                }

                for ff in xs {
                    // solve linear equation
                    if ff.degree(new_var) == E::one() {
                        let a = ff.to_univariate(new_var);
                        let sol = RationalPolynomial::from_num_den(
                            -a.coefficients[0].clone(),
                            a.coefficients[1].clone(),
                            &a.coefficients[0].ring,
                            true,
                        );

                        let eval = monic.replace(new_var, &sol);

                        let mut res = p.ring.zero();
                        for t in &eval {
                            let mut exp = vec![E::zero(); p.lcoeff().numerator.nvars()];
                            exp.copy_from_slice(t.exponents);
                            let mm = p.coefficients[0]
                                .numerator
                                .monomial(self.numerator.ring.one(), exp);
                            res = &res + &(t.coefficient * &mm.into());
                        }

                        w.push((&sol * &constant, res));
                    } else {
                        w.push((
                            &RationalPolynomial::from(ff.clone()) * &constant,
                            res.clone(),
                        ));
                    }
                }
            }
        }

        (v, w)
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::{
        atom::AtomCore,
        domains::{
            InternalOrdering, Ring,
            finite_field::{ToFiniteField, Zp},
            integer::Z,
            rational::Q,
            rational_polynomial::RationalPolynomial,
        },
        parse, symbol,
    };

    use super::RationalPolynomialField;

    #[test]
    fn eval_map() {
        let a = parse!(" (x^2 + 19) / (x + 44) ").to_rational_polynomial::<_, _, u8>(&Z, &Z, None);
        let f = Zp::new(17);
        let res = a.evaluate_with_coeff_map(|c| c.to_finite_field(&f), &[f.nth(3.into())], &f);
        assert_eq!(res, f.nth(10.into()));
    }

    #[test]
    fn field() {
        let field = RationalPolynomialField::<_, u8>::new(Z);
        let one = field.one();
        let t = format!("{}", field.printer(&one));
        assert_eq!(t, "1");
    }

    #[test]
    fn hermite_reduction() {
        let p: RationalPolynomial<_, _> =
            parse!("1/(v1 + 1)^5").to_rational_polynomial::<_, _, u8>(&Q, &Z, None);

        let (r, l) = p.integrate(0);

        assert_eq!(
            r,
            vec![
                parse!("-1/(4+16*v1+24*v1^2+16*v1^3+4*v1^4)").to_rational_polynomial::<_, _, u8>(
                    &Q,
                    &Z,
                    r[0].get_variables().clone()
                )
            ]
        );
        assert_eq!(l, vec![]);
    }

    #[test]
    fn constant() {
        let p: RationalPolynomial<_, _> = parse!("(v1^4+v2+v1*v2+2*v1)/(v2 + 1)")
            .to_rational_polynomial::<_, _, u8>(
                &Q,
                &Z,
                Some(Arc::new(vec![symbol!("v1").into(), symbol!("v2").into()])),
            );

        let (r, l) = p.integrate(0);
        assert_eq!(
            r,
            vec![
                parse!("(10*v1*v2+10*v1^2+5*v1^2*v2+2*v1^5)/(10+10*v2)")
                    .to_rational_polynomial::<_, _, u8>(&Q, &Z, r[0].get_variables().clone())
            ]
        );
        assert_eq!(l, vec![]);
    }

    #[test]
    fn mixed_denominator() {
        let p: RationalPolynomial<_, _> = parse!("(v1^4+v2+v1*v2+2*v1)/(v1)/(v2 + 1)")
            .to_rational_polynomial::<_, _, u8>(
                &Q,
                &Z,
                Some(Arc::new(vec![symbol!("v1").into(), symbol!("v2").into()])),
            );

        let (r, l) = p.integrate(0);

        let v = l[0].0.get_variables().clone();

        assert_eq!(
            r,
            vec![
                parse!("(8*v1+4*v1*v2+v1^4)/(4+4*v2)").to_rational_polynomial::<_, _, u8>(
                    &Q,
                    &Z,
                    r[0].get_variables().clone()
                )
            ]
        );
        assert_eq!(
            l,
            vec![(
                parse!("v2/(1+v2)").to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone()),
                parse!("v1").to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone()),
            ),]
        );
    }

    #[test]
    fn three_factors() {
        let p: RationalPolynomial<_, _> =
            parse!("(36v1^2+1167v1+3549/2)/(v1^3+23/30v1^2-2/15v1-2/15)")
                .to_rational_polynomial::<_, _, u8>(&Q, &Z, None);

        let (r, mut l) = p.integrate(0);
        l.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.internal_cmp(&b.1)));

        let v = l[0].0.get_variables().clone();

        assert!(r.is_empty());
        assert_eq!(
            l,
            vec![
                (
                    parse!("-8000").to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone()),
                    parse!("(1+2*v1)/2").to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone()),
                ),
                (
                    parse!("37451/16").to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone()),
                    parse!("(-2+5*v1)/5").to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone()),
                ),
                (
                    parse!("91125/16").to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone()),
                    parse!("(2+3*v1)/3").to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone()),
                ),
            ]
        );
    }

    #[test]
    fn multiple_residues() {
        let p: RationalPolynomial<_, _> = parse!(
            "(7v1^13+10v1^8+4v1^7-7v1^6-4v1^3-4v1^2+3v1+3)/(v1^14-2v1^8-2v1^7-2v1^4-4v1^3-v1^2+2v1+1)"
        )
        .to_rational_polynomial::<_, _, u8>(&Q, &Z, None);

        let (r, mut l) = p.integrate(0);
        let new_var = symbol!("v2");

        // root sum in the answer, rename the temporary variable
        // TODO: add rename function
        let mut v = l[0].0.get_variables().as_ref().clone();
        *v.last_mut().unwrap() = new_var.into();
        let new_map = Arc::new(v);

        l[0].0.numerator.variables = new_map.clone();
        l[0].0.denominator.variables = new_map.clone();
        l[0].1.numerator.variables = new_map.clone();
        l[0].1.denominator.variables = new_map.clone();

        assert!(r.is_empty());
        assert_eq!(
            l,
            vec![(
                parse!("-1-4*v2+4*v2^2").to_rational_polynomial::<_, _, u8>(
                    &Q,
                    &Z,
                    new_map.clone()
                ),
                parse!("-1-2*v1*v2+v1^2-2*v1^2*v2+v1^7").to_rational_polynomial::<_, _, u8>(
                    &Q,
                    &Z,
                    new_map.clone()
                ),
            )]
        );
    }

    #[test]
    fn multi_factor() {
        let p: RationalPolynomial<_, _> =
            parse!("1/(v1^3+v1)").to_rational_polynomial::<_, _, u8>(&Q, &Z, None);

        let (r, mut l) = p.integrate(0);
        l.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.internal_cmp(&b.1)));

        let v = l[0].0.get_variables().clone();

        assert!(r.is_empty());
        assert_eq!(
            l,
            vec![
                (
                    parse!("-1/2").to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone()),
                    parse!("1+v1^2").to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone()),
                ),
                (
                    parse!("1").to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone()),
                    parse!("v1").to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone()),
                )
            ]
        );
    }

    #[test]
    fn multiple_variables() {
        let p: RationalPolynomial<_, _> = parse!("(v1^4+v2+v1*v2+2*v1)/((v1-v2)(v1-2)(v1-4))")
            .to_rational_polynomial::<_, _, u8>(&Q, &Z, None);

        let (r, mut l) = p.integrate(0);
        l.sort_by(|a, b| a.0.internal_cmp(&b.0).then(a.1.internal_cmp(&b.1)));

        let v = l[0].0.get_variables().clone();

        assert_eq!(
            r,
            vec![
                parse!("(12*v1+2*v1*v2+v1^2)/2").to_rational_polynomial::<_, _, u8>(
                    &Q,
                    &Z,
                    r[0].get_variables().clone()
                )
            ]
        );
        assert_eq!(
            l,
            vec![
                (
                    parse!("(-264-5*v2)/(-8+2*v2)").to_rational_polynomial::<_, _, u8>(
                        &Q,
                        &Z,
                        v.clone()
                    ),
                    parse!("-4+v1").to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone()),
                ),
                (
                    parse!("(20+3*v2)/(-4+2*v2)").to_rational_polynomial::<_, _, u8>(
                        &Q,
                        &Z,
                        v.clone()
                    ),
                    parse!("-2+v1").to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone()),
                ),
                (
                    parse!("(3*v2+v2^2+v2^4)/(8-6*v2+v2^2)").to_rational_polynomial::<_, _, u8>(
                        &Q,
                        &Z,
                        v.clone()
                    ),
                    parse!("-v2+v1").to_rational_polynomial::<_, _, u8>(&Q, &Z, v.clone()),
                )
            ]
        );
    }
}
