//! Rational polynomial field with factorized denominator.

use std::{
    borrow::Cow,
    cmp::Ordering,
    fmt::{Display, Error},
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::Arc,
};

use crate::{
    domains::{RingOps, Set, algebraic_number::AlgebraicExtension},
    poly::{
        PositiveExponent, PolyVariable, factor::Factorize, gcd::PolynomialGCD,
        polynomial::MultivariatePolynomial,
    },
    printer::{PrintOptions, PrintState},
};

use super::{
    EuclideanDomain, Field, InternalOrdering, Ring, SelfRing,
    finite_field::{FiniteField, FiniteFieldCore, FiniteFieldWorkspace, ToFiniteField},
    integer::{Integer, IntegerRing, Z},
    rational::RationalField,
};

/// A rational polynomial field with a factorized denominator.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FactorizedRationalPolynomialField<R: Ring, E: PositiveExponent> {
    ring: R,
    var_map: Arc<Vec<PolyVariable>>,
    _phantom_exp: PhantomData<E>,
}

impl<R: Ring, E: PositiveExponent> FactorizedRationalPolynomialField<R, E> {
    pub fn new(
        coeff_ring: R,
        var_map: Arc<Vec<PolyVariable>>,
    ) -> FactorizedRationalPolynomialField<R, E> {
        FactorizedRationalPolynomialField {
            ring: coeff_ring,
            var_map,
            _phantom_exp: PhantomData,
        }
    }

    pub fn new_from_poly(
        poly: &MultivariatePolynomial<R, E>,
    ) -> FactorizedRationalPolynomialField<R, E> {
        FactorizedRationalPolynomialField {
            ring: poly.ring.clone(),
            var_map: poly.variables.clone(),
            _phantom_exp: PhantomData,
        }
    }
}

pub trait FromNumeratorAndFactorizedDenominator<R: Ring, OR: Ring, E: PositiveExponent> {
    /// Construct a rational polynomial from a numerator and a factorized denominator.
    /// An empty denominator means a denominator of 1.
    fn from_num_den(
        num: MultivariatePolynomial<R, E>,
        dens: Vec<(MultivariatePolynomial<R, E>, usize)>,
        field: &OR,
        do_factor: bool,
    ) -> FactorizedRationalPolynomial<OR, E>;
}

/// A rational polynomial with a factorized denominator.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct FactorizedRationalPolynomial<R: Ring, E: PositiveExponent> {
    pub numerator: MultivariatePolynomial<R, E>,
    pub numer_coeff: R::Element,
    pub denom_coeff: R::Element,
    pub denominators: Vec<(MultivariatePolynomial<R, E>, usize)>, // TODO: sort factors?
}

impl<R: Ring, E: PositiveExponent> InternalOrdering for FactorizedRationalPolynomial<R, E> {
    /// An ordering of rational polynomials that has no intuitive meaning.
    fn internal_cmp(&self, _other: &Self) -> Ordering {
        todo!()
    }
}

impl<R: Ring, E: PositiveExponent> FactorizedRationalPolynomial<R, E> {
    pub fn new(field: &R, var_map: Arc<Vec<PolyVariable>>) -> FactorizedRationalPolynomial<R, E> {
        let num = MultivariatePolynomial::new(field, None, var_map);

        FactorizedRationalPolynomial {
            numerator: num,
            numer_coeff: field.zero(),
            denom_coeff: field.one(),
            denominators: vec![],
        }
    }

    pub fn get_variables(&self) -> &[PolyVariable] {
        self.numerator.get_vars_ref()
    }

    pub fn unify_variables(&mut self, other: &mut Self) {
        self.numerator.unify_variables(&mut other.numerator);

        for d in &mut self.denominators {
            d.0.unify_variables(&mut other.numerator);
        }
    }

    pub fn is_constant(&self) -> bool {
        self.numerator.is_constant() && self.denominators.is_empty()
    }

    /// Convert the coefficient from the current field to a finite field.
    pub fn to_finite_field<UField: FiniteFieldWorkspace>(
        &self,
        field: &FiniteField<UField>,
    ) -> FactorizedRationalPolynomial<FiniteField<UField>, E>
    where
        R::Element: ToFiniteField<UField>,
        FiniteField<UField>: FiniteFieldCore<UField>,
        <FiniteField<UField> as Set>::Element: Copy,
    {
        let constant = field.div(
            &self.numer_coeff.to_finite_field(field),
            &self.denom_coeff.to_finite_field(field),
        );

        // check the gcd, since the rational polynomial may simplify
        FactorizedRationalPolynomial::from_num_den(
            self.numerator
                .map_coeff(|c| c.to_finite_field(field), field.clone())
                .mul_coeff(constant),
            self.denominators
                .iter()
                .map(|(f, p)| (f.map_coeff(|c| c.to_finite_field(field), field.clone()), *p))
                .collect(),
            field,
            true,
        )
    }

    pub fn is_zero(&self) -> bool {
        self.numerator.is_zero()
    }

    pub fn is_one(&self) -> bool {
        self.numerator.is_one()
            && self.denominators.is_empty()
            && self.numerator.ring.is_one(&self.numer_coeff)
            && self.numerator.ring.is_one(&self.denom_coeff)
    }
}

impl<R: Ring, E: PositiveExponent> SelfRing for FactorizedRationalPolynomial<R, E> {
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
    ) -> Result<bool, Error> {
        let ring = &self.numerator.ring;
        let has_numer_coeff = !ring.is_one(&self.numer_coeff);
        let has_denom_coeff = !ring.is_one(&self.denom_coeff);

        if opts.explicit_rational_polynomial {
            if state.in_sum {
                f.write_char('+')?;
            }

            if !ring.is_zero(&self.numer_coeff) && has_numer_coeff {
                f.write_char('[')?;
                self.numerator
                    .ring
                    .format(&self.numer_coeff, opts, PrintState::new(), f)?;
                f.write_str("]*")?;
            }

            if self.denominators.is_empty() && !has_denom_coeff {
                if self.numerator.is_zero() {
                    if state.in_sum {
                        f.write_str("+0")?;
                    } else {
                        f.write_char('0')?;
                    }
                } else {
                    f.write_char('[')?;
                    self.numerator.format(opts, PrintState::new(), f)?;
                    f.write_str("]")?;
                }
            } else {
                f.write_char('[')?;
                self.numerator.format(opts, PrintState::new(), f)?;

                if has_denom_coeff {
                    f.write_char(',')?;
                    self.numerator
                        .ring
                        .format(&self.denom_coeff, opts, PrintState::new(), f)?;
                    f.write_str(",1")?;
                }

                for (d, p) in &self.denominators {
                    f.write_char(',')?;
                    d.format(opts, PrintState::new(), f)?;
                    f.write_fmt(format_args!(",{p}"))?;
                }

                f.write_char(']')?;
            }

            return Ok(false);
        }

        if ring.is_zero(&self.numer_coeff) {
            if state.in_sum {
                f.write_str("+0")?;
            } else {
                f.write_char('0')?;
            }

            return Ok(false);
        }

        if self.denominators.is_empty() && !has_denom_coeff {
            let write_par =
                has_numer_coeff && !self.numerator.is_one() && (state.in_exp || state.in_exp_base);
            if write_par {
                if state.in_sum {
                    state.in_sum = false;
                    f.write_char('+')?;
                }

                f.write_char('(')?;
                state.in_exp = false;
                state.in_exp_base = false;
            }

            if has_numer_coeff {
                state.in_product |= !self.numerator.is_one();
                self.numerator
                    .ring
                    .format(&self.numer_coeff, opts, state, f)?;
                state.in_sum = false;

                if self.numerator.is_one() {
                    return Ok(false);
                }
                f.write_char(opts.multiplication_operator)?;
            }

            if write_par {
                self.numerator.format(opts, state, f)?;
                f.write_char(')')?;
                return Ok(false);
            } else {
                return self.numerator.format(opts, state, f);
            }
        }

        state.suppress_one = false;

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
            if has_numer_coeff {
                state.suppress_one = true;
                state.in_product = true;
                self.numerator
                    .ring
                    .format(&self.numer_coeff, opts, state, f)?;
                state.suppress_one = false;
            }

            f.write_str("\\frac{")?;
            self.numerator.format(opts, PrintState::new(), f)?;
            f.write_str("}{")?;

            if has_denom_coeff {
                self.numerator.ring.format(
                    &self.denom_coeff,
                    opts,
                    state.step(false, !self.denominators.is_empty(), false, false),
                    f,
                )?;
            }

            for (d, p) in &self.denominators {
                d.format(opts, state.step(false, true, false, *p != 1), f)?;

                if *p != 1 {
                    f.write_fmt(format_args!("^{p}"))?;
                }
            }

            f.write_char('}')?;

            if write_par {
                f.write_char(')')?;
            }

            return Ok(false);
        }

        state.in_product = true;
        if has_numer_coeff || self.numerator.is_one() {
            self.numerator
                .ring
                .format(&self.numer_coeff, opts, state, f)?;
            state.in_sum = false;

            if !self.numerator.is_one() {
                f.write_char(opts.multiplication_operator)?;
            }
        }

        if !self.numerator.is_one() {
            self.numerator.format(opts, state, f)?;
        }

        f.write_char('/')?;

        if self.denominators.is_empty() {
            return self.numerator.ring.format(
                &self.denom_coeff,
                opts,
                state.step(false, true, false, false),
                f,
            );
        }

        state.in_product = has_denom_coeff || self.denominators.len() > 1;

        if state.in_product {
            f.write_char('(')?;
        }

        if has_denom_coeff {
            self.numerator.ring.format(
                &self.denom_coeff,
                opts,
                state.step(false, true, false, false),
                f,
            )?;
        }

        for (i, (d, p)) in self.denominators.iter().enumerate() {
            if has_denom_coeff || i > 0 {
                f.write_char(opts.multiplication_operator)?;
            }

            d.format(opts, state.step(false, true, false, *p != 1), f)?;

            if *p != 1 {
                f.write_fmt(format_args!(
                    "{}{}",
                    if opts.double_star_for_exponentiation {
                        "**"
                    } else {
                        "^"
                    },
                    p
                ))?;
            }
        }

        if state.in_product {
            f.write_char(')')?;
        }

        if write_par {
            f.write_char(')')?;
        }

        Ok(false)
    }
}

impl<E: PositiveExponent> FromNumeratorAndFactorizedDenominator<RationalField, IntegerRing, E>
    for FactorizedRationalPolynomial<IntegerRing, E>
{
    fn from_num_den(
        num: MultivariatePolynomial<RationalField, E>,
        dens: Vec<(MultivariatePolynomial<RationalField, E>, usize)>,
        field: &IntegerRing,
        do_factor: bool,
    ) -> FactorizedRationalPolynomial<IntegerRing, E> {
        let mut content = num.content();
        for (d, _) in &dens {
            content = d.ring.gcd(&content, &d.content());
        }

        let (num_int, dens_int) = if num.ring.is_one(&content) {
            (
                num.map_coeff(|c| c.numerator(), Z),
                dens.iter()
                    .map(|(d, p)| (d.map_coeff(|c| c.numerator(), Z), *p))
                    .collect(),
            )
        } else {
            (
                num.map_coeff(|c| num.ring.div(c, &content).numerator(), Z),
                dens.iter()
                    .map(|(d, p)| {
                        (
                            d.map_coeff(|c| num.ring.div(c, &content).numerator(), Z),
                            *p,
                        )
                    })
                    .collect(),
            )
        };

        <FactorizedRationalPolynomial<IntegerRing, E> as FromNumeratorAndFactorizedDenominator<
            IntegerRing,
            IntegerRing,
            E,
        >>::from_num_den(num_int, dens_int, field, do_factor)
    }
}

impl<E: PositiveExponent> FromNumeratorAndFactorizedDenominator<IntegerRing, IntegerRing, E>
    for FactorizedRationalPolynomial<IntegerRing, E>
{
    fn from_num_den(
        mut num: MultivariatePolynomial<IntegerRing, E>,
        mut dens: Vec<(MultivariatePolynomial<IntegerRing, E>, usize)>,
        _field: &IntegerRing,
        do_factor: bool,
    ) -> Self {
        for _ in 0..2 {
            for (d, _) in &mut dens {
                num.unify_variables(d);
            }
        }

        let mut num_const = num.ring.one();
        let mut den_const = num.ring.one();

        if dens.is_empty() {
            let g = num.content();
            if !g.is_one() {
                num = num.div_coeff(&g);
                num_const = g;
            }

            if num.lcoeff().is_negative() {
                num_const = num_const.neg();
                num = -num;
            }

            return FactorizedRationalPolynomial {
                numerator: num,
                numer_coeff: num_const,
                denom_coeff: den_const,
                denominators: dens,
            };
        }

        if do_factor {
            for (d, _) in &mut dens {
                let gcd = num.gcd(d);

                if !gcd.is_one() {
                    num = num / &gcd;
                    *d = &*d / &gcd;
                }
            }

            // factor all denominators, as they may be unfactored
            // TODO: add extra flag for this?
            let mut factored = vec![];
            for (d, p) in dens {
                for (f, p2) in d.factor() {
                    factored.push((f, p * p2));
                }
            }

            // TODO: fuse factors that are the same

            dens = factored;
        }

        dens.retain(|f| {
            if f.0.is_constant() {
                den_const = &den_const * &f.0.lcoeff().pow(f.1 as u64);
                false
            } else {
                true
            }
        });

        if den_const.is_negative() {
            num_const = num_const.neg();
            den_const = den_const.neg();
        }

        // normalize denominator to have positive leading coefficient
        for (d, _) in &mut dens {
            if d.lcoeff().is_negative() {
                num_const = num_const.neg();
                *d = -d.clone(); // TODO: prevent clone
            }
        }

        // TODO: add flag for this?
        let g = num.content();
        if !g.is_one() {
            num = num.div_coeff(&g);
            num_const = &num_const * &g;
        }

        // normalize numerator to have positive leading coefficient
        if num.lcoeff().is_negative() {
            num_const = num_const.neg();
            num = -num;
        }

        FactorizedRationalPolynomial {
            numerator: num,
            numer_coeff: num_const,
            denom_coeff: den_const,
            denominators: dens,
        }
    }
}

impl<UField: FiniteFieldWorkspace, E: PositiveExponent>
    FromNumeratorAndFactorizedDenominator<FiniteField<UField>, FiniteField<UField>, E>
    for FactorizedRationalPolynomial<FiniteField<UField>, E>
where
    FiniteField<UField>: FiniteFieldCore<UField>,
    <FiniteField<UField> as Set>::Element: Copy,
{
    fn from_num_den(
        mut num: MultivariatePolynomial<FiniteField<UField>, E>,
        mut dens: Vec<(MultivariatePolynomial<FiniteField<UField>, E>, usize)>,
        field: &FiniteField<UField>,
        do_factor: bool,
    ) -> Self {
        for _ in 0..2 {
            for (d, _) in &mut dens {
                num.unify_variables(d);
            }
        }

        let mut constant = num.ring.one();

        if dens.is_empty() {
            return FactorizedRationalPolynomial {
                numerator: num,
                numer_coeff: constant,
                denom_coeff: constant,
                denominators: dens,
            };
        }

        if do_factor {
            for (d, _) in &mut dens {
                let gcd = num.gcd(d);

                if !gcd.is_one() {
                    num = num / &gcd;
                    *d = &*d / &gcd;
                }
            }

            // factor all denominators, as they may be unfactored
            // TODO: add extra flag for this?
            let mut factored = vec![];
            for (d, p) in dens {
                for (f, p2) in d.factor() {
                    factored.push((f, p * p2));
                }
            }

            // TODO: fuse factors that are the same

            dens = factored;
        }

        dens.retain(|f| {
            if f.0.is_constant() {
                field.mul_assign(&mut constant, &field.pow(&f.0.coefficients[0], f.1 as u64));
                false
            } else {
                true
            }
        });

        num = num.mul_coeff(field.inv(&constant));
        constant = field.one();

        // normalize denominator to have leading coefficient of one
        for (d, _) in &mut dens {
            if !field.is_one(&d.lcoeff()) {
                let c = field.inv(&d.lcoeff());
                num = num.mul_coeff(c);
                *d = d.clone().mul_coeff(c); // TODO: prevent clone
            }
        }

        FactorizedRationalPolynomial {
            numerator: num,
            numer_coeff: constant,
            denom_coeff: constant,
            denominators: dens,
        }
    }
}

impl<F: Field, E: PositiveExponent>
    FromNumeratorAndFactorizedDenominator<AlgebraicExtension<F>, AlgebraicExtension<F>, E>
    for FactorizedRationalPolynomial<AlgebraicExtension<F>, E>
where
    AlgebraicExtension<F>: Field + PolynomialGCD<E>,
    MultivariatePolynomial<AlgebraicExtension<F>, E>: Factorize,
{
    fn from_num_den(
        mut num: MultivariatePolynomial<AlgebraicExtension<F>, E>,
        mut dens: Vec<(MultivariatePolynomial<AlgebraicExtension<F>, E>, usize)>,
        field: &AlgebraicExtension<F>,
        do_factor: bool,
    ) -> FactorizedRationalPolynomial<AlgebraicExtension<F>, E> {
        for _ in 0..2 {
            for (d, _) in &mut dens {
                num.unify_variables(d);
            }
        }

        let mut constant = num.ring.one();

        if dens.is_empty() {
            return FactorizedRationalPolynomial {
                numerator: num,
                numer_coeff: constant.clone(),
                denom_coeff: constant.clone(),
                denominators: dens,
            };
        }

        if do_factor {
            for (d, _) in &mut dens {
                let gcd = num.gcd(d);

                if !gcd.is_one() {
                    num = num / &gcd;
                    *d = &*d / &gcd;
                }
            }

            // factor all denominators, as they may be unfactored
            // TODO: add extra flag for this?
            let mut factored = vec![];
            for (d, p) in dens {
                for (f, p2) in d.factor() {
                    factored.push((f, p * p2));
                }
            }

            // TODO: fuse factors that are the same

            dens = factored;
        }

        dens.retain(|f| {
            if f.0.is_constant() {
                field.mul_assign(&mut constant, &field.pow(&f.0.coefficients[0], f.1 as u64));
                false
            } else {
                true
            }
        });

        num = num.mul_coeff(field.inv(&constant));
        constant = field.one();

        // normalize denominator to have leading coefficient of one
        for (d, _) in &mut dens {
            if !field.is_one(&d.lcoeff()) {
                let c = field.inv(&d.lcoeff());
                num = num.mul_coeff(c.clone());
                *d = d.clone().mul_coeff(c); // TODO: prevent clone
            }
        }

        FactorizedRationalPolynomial {
            numerator: num,
            numer_coeff: constant.clone(),
            denom_coeff: constant,
            denominators: dens,
        }
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> FactorizedRationalPolynomial<R, E>
where
    Self: FromNumeratorAndFactorizedDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
{
    /// Invert a factored rational polynomial. This is an expensive operation, as it requires
    /// factoring the numerator.
    #[inline]
    pub fn inv(self) -> Self {
        if self.numerator.is_zero() {
            panic!("Cannot invert 0");
        }

        let mut num = self.numerator.constant(self.denom_coeff);
        for (d, p) in self.denominators {
            num = num * &d.pow(p);
        }

        let mut dens = self.numerator.factor();
        dens.push((self.numerator.constant(self.numer_coeff), 1));

        let field = self.numerator.ring.clone();
        Self::from_num_den(num, dens, &field, false)
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> FactorizedRationalPolynomial<R, E>
where
    Self: FromNumeratorAndFactorizedDenominator<R, R, E>,
{
    pub fn pow(&self, e: u64) -> Self {
        if e > u32::MAX as u64 {
            panic!("Power of exponentiation is larger than 2^32: {e}");
        }
        let e = e as u32;

        // TODO: do binary exponentiation
        let mut poly = FactorizedRationalPolynomial {
            numerator: self.numerator.constant(self.numerator.ring.one()),
            numer_coeff: self.numerator.ring.one(),
            denom_coeff: self.numerator.ring.one(),
            denominators: vec![],
        };

        for _ in 0..e {
            poly = &poly * self;
        }
        poly
    }

    pub fn gcd(&self, other: &Self) -> Self {
        if self.get_variables() != other.get_variables() {
            let mut a = self.clone();
            let mut b = other.clone();
            a.unify_variables(&mut b);
            return a.gcd(&b);
        }

        let gcd_num = self.numerator.gcd(&other.numerator);

        let mut disjoint_factors = vec![];

        for (d, p) in &self.denominators {
            let mut found = false;
            for (d2, p2) in &other.denominators {
                if d == d2 {
                    disjoint_factors.push((d.clone(), *p.min(p2)));
                    found = true;
                    break;
                }
            }

            if !found {
                disjoint_factors.push((d.clone(), *p));
            }
        }

        for (d, p) in &other.denominators {
            let mut found = false;
            for (d2, _) in &self.denominators {
                if d == d2 {
                    found = true;
                    break;
                }
            }

            if !found {
                disjoint_factors.push((d.clone(), *p));
            }
        }

        let field = &self.numerator.ring;
        FactorizedRationalPolynomial {
            numerator: gcd_num,
            numer_coeff: field.gcd(&self.numer_coeff, &other.numer_coeff),
            denom_coeff: field.mul(
                &field
                    .quot_rem(
                        &self.denom_coeff,
                        &field.gcd(&self.denom_coeff, &other.denom_coeff),
                    )
                    .0,
                &other.denom_coeff,
            ),
            denominators: disjoint_factors,
        }
    }
}

impl<R: Field, E: PositiveExponent> FactorizedRationalPolynomial<R, E> {
    /// Evaluate the rational polynomial.
    pub fn evaluate(&self, x: &[R::Element]) -> R::Element {
        let ring = &self.numerator.ring;
        let mut num = self.numerator.replace_all(x);
        ring.mul_assign(&mut num, &self.numer_coeff);

        let mut den = self.denom_coeff.clone();
        for (d, p) in &self.denominators {
            let mut eval = d.replace_all(x);
            if *p != 1 {
                eval = ring.pow(&eval, *p as u64);
            }

            ring.mul_assign(&mut den, &eval);
        }

        ring.div(&num, &den)
    }
}

impl<R: Ring, E: PositiveExponent> FactorizedRationalPolynomial<R, E> {
    /// Evaluate the rational polynomial at the given point, mapping coefficients to the ring `U`.
    pub fn evaluate_with_coeff_map<U: Field, T: Fn(&R::Element) -> U::Element + Clone>(
        &self,
        map_coeff: T,
        point: &[U::Element],
        ring: &U,
    ) -> U::Element {
        let mut num = self
            .numerator
            .evaluate_with_coeff_map(map_coeff.clone(), point, ring);
        ring.mul_assign(&mut num, &map_coeff(&self.numer_coeff));

        let mut den = map_coeff(&self.denom_coeff);
        for (d, p) in &self.denominators {
            let mut eval = d.evaluate_with_coeff_map(map_coeff.clone(), point, ring);
            if *p != 1 {
                eval = ring.pow(&eval, *p as u64);
            }

            ring.mul_assign(&mut den, &eval);
        }

        ring.div(&num, &den)
    }
}

impl<R: Ring, E: PositiveExponent> Display for FactorizedRationalPolynomial<R, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format(&PrintOptions::from_fmt(f), PrintState::from_fmt(f), f)
            .map(|_| ())
    }
}

impl<R: Ring, E: PositiveExponent> Display for FactorizedRationalPolynomialField<R, E> {
    fn fmt(&self, _: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> Set
    for FactorizedRationalPolynomialField<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
{
    type Element = FactorizedRationalPolynomial<R, E>;

    fn size(&self) -> Option<Integer> {
        None
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>
    RingOps<FactorizedRationalPolynomial<R, E>> for FactorizedRationalPolynomialField<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
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

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>
    RingOps<&FactorizedRationalPolynomial<R, E>> for FactorizedRationalPolynomialField<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
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
    for FactorizedRationalPolynomialField<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
{
    fn zero(&self) -> Self::Element {
        FactorizedRationalPolynomial {
            numerator: MultivariatePolynomial::new(&self.ring, None, self.var_map.clone()),
            numer_coeff: self.ring.zero(),
            denom_coeff: self.ring.one(),
            denominators: vec![],
        }
    }

    fn one(&self) -> Self::Element {
        FactorizedRationalPolynomial {
            numerator: MultivariatePolynomial::new(&self.ring, None, self.var_map.clone()).one(),
            numer_coeff: self.ring.one(),
            denom_coeff: self.ring.one(),
            denominators: vec![],
        }
    }

    fn nth(&self, n: Integer) -> Self::Element {
        let mut r = self.one();
        r.numer_coeff = self.ring.nth(n);
        r
    }

    fn pow(&self, b: &Self::Element, e: u64) -> Self::Element {
        if e > u32::MAX as u64 {
            panic!("Power of exponentiation is larger than 2^32: {e}");
        }
        let e = e as u32;

        // TODO: do binary exponentiation
        let mut poly = FactorizedRationalPolynomial {
            numerator: b.numerator.constant(self.ring.one()),
            numer_coeff: self.ring.one(),
            denom_coeff: self.ring.one(),
            denominators: vec![],
        };

        for _ in 0..e {
            poly = self.mul(&poly, b);
        }
        poly
    }

    fn is_zero(&self, a: &Self::Element) -> bool {
        a.numerator.is_zero()
    }

    fn is_one(&self, a: &Self::Element) -> bool {
        a.numerator.is_one() && a.denominators.is_empty() && a.numerator.ring.is_one(&a.denom_coeff)
    }

    fn one_is_gcd_unit() -> bool {
        false
    }

    fn characteristic(&self) -> Integer {
        self.ring.characteristic()
    }

    fn try_inv(&self, a: &Self::Element) -> Option<Self::Element> {
        if a.is_zero() {
            None
        } else {
            Some(a.clone().inv())
        }
    }

    fn try_div(&self, a: &Self::Element, b: &Self::Element) -> Option<Self::Element> {
        if b.is_zero() {
            None
        } else {
            let (q, r) = self.quot_rem(a, b);
            if r.is_zero() { Some(q) } else { None }
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
    for FactorizedRationalPolynomialField<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
{
    fn rem(&self, a: &Self::Element, _: &Self::Element) -> Self::Element {
        FactorizedRationalPolynomial {
            numerator: a.numerator.zero(),
            numer_coeff: self.ring.one(),
            denom_coeff: self.ring.one(),
            denominators: vec![],
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
    for FactorizedRationalPolynomialField<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
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
    Add<&'a FactorizedRationalPolynomial<R, E>> for &FactorizedRationalPolynomial<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
{
    type Output = FactorizedRationalPolynomial<R, E>;

    fn add(self, other: &'a FactorizedRationalPolynomial<R, E>) -> Self::Output {
        if self.is_zero() {
            return other.clone();
        } else if other.is_zero() {
            return self.clone();
        }

        if self.get_variables() != other.get_variables() {
            let mut a = self.clone();
            let mut b = other.clone();
            a.unify_variables(&mut b);
            return &a + &b;
        }

        let mut den = Vec::with_capacity(self.denominators.len() + other.denominators.len());
        let mut num_1 = self.numerator.clone();
        let mut num_2 = other.numerator.clone();

        for (d, p) in &self.denominators {
            if let Some((_, p2)) = other.denominators.iter().find(|(d2, _)| d == d2) {
                if p > p2 {
                    num_2 = num_2 * &d.pow(*p - *p2);
                } else if p < p2 {
                    num_1 = num_1 * &d.pow(*p2 - *p);
                }
                den.push((d.clone(), *p.max(p2)));
                continue;
            }
            num_2 = num_2 * &d.pow(*p);
            den.push((d.clone(), *p));
        }
        for (d, p) in &other.denominators {
            if self.denominators.iter().any(|(d2, _)| d == d2) {
                continue;
            }
            num_1 = num_1 * &d.pow(*p);
            den.push((d.clone(), *p));
        }

        let ring = &self.numerator.ring;
        let mut coeff1 = self.numer_coeff.clone();
        let mut coeff2 = other.numer_coeff.clone();
        let mut new_denom = self.denom_coeff.clone();

        if self.denom_coeff != other.denom_coeff {
            let d_coeff = ring.gcd(&self.denom_coeff, &other.denom_coeff);
            ring.mul_assign(&mut coeff1, &ring.quot_rem(&other.denom_coeff, &d_coeff).0);
            ring.mul_assign(&mut coeff2, &ring.quot_rem(&self.denom_coeff, &d_coeff).0);
            ring.mul_assign(
                &mut new_denom,
                &ring.quot_rem(&other.denom_coeff, &d_coeff).0,
            );
        }

        let num_gcd = ring.gcd(&coeff1, &coeff2);
        let mut num = num_1.mul_coeff(ring.quot_rem(&coeff1, &num_gcd).0)
            + num_2.mul_coeff(ring.quot_rem(&coeff2, &num_gcd).0);

        if num.is_zero() {
            return FactorizedRationalPolynomial {
                numerator: num,
                numer_coeff: self.numerator.ring.zero(),
                denom_coeff: self.numerator.ring.one(),
                denominators: vec![],
            };
        }

        // TODO: are there some factors we can skip the division check for?
        for (d, p) in &mut den {
            while *p > 0 {
                if let Some(q) = num.try_div(d) {
                    num = q;
                    *p -= 1;
                } else {
                    break;
                }
            }
        }

        den.retain(|(_, p)| *p > 0);

        // make sure the numerator is properly normalized
        let mut r =
            FactorizedRationalPolynomial::from_num_den(num, vec![], &self.numerator.ring, false);

        let field = &r.numerator.ring;
        field.mul_assign(&mut r.numer_coeff, &num_gcd);
        let g = r.numerator.ring.gcd(&r.numer_coeff, &new_denom);
        if !field.is_one(&g) {
            r.numer_coeff = field.quot_rem(&r.numer_coeff, &g).0;
            new_denom = field.quot_rem(&new_denom, &g).0;
        }

        r.denom_coeff = new_denom;
        r.denominators = den;

        r
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> Sub
    for FactorizedRationalPolynomial<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
{
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.add(&other.neg())
    }
}

impl<'a, R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>
    Sub<&'a FactorizedRationalPolynomial<R, E>> for &FactorizedRationalPolynomial<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
{
    type Output = FactorizedRationalPolynomial<R, E>;

    fn sub(self, other: &'a FactorizedRationalPolynomial<R, E>) -> Self::Output {
        (self.clone()).sub(other.clone())
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> Neg
    for FactorizedRationalPolynomial<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
{
    type Output = Self;
    fn neg(self) -> Self::Output {
        FactorizedRationalPolynomial {
            numer_coeff: self.numerator.ring.neg(&self.numer_coeff),
            numerator: self.numerator,
            denom_coeff: self.denom_coeff,
            denominators: self.denominators,
        }
    }
}

impl<'a, R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>
    Mul<&'a FactorizedRationalPolynomial<R, E>> for &FactorizedRationalPolynomial<R, E>
{
    type Output = FactorizedRationalPolynomial<R, E>;

    fn mul(self, other: &'a FactorizedRationalPolynomial<R, E>) -> Self::Output {
        if self.is_one() {
            return other.clone();
        } else if other.is_one() {
            return self.clone();
        }

        if self.get_variables() != other.get_variables() {
            let mut a = self.clone();
            let mut b = other.clone();
            a.unify_variables(&mut b);
            return &a * &b;
        }

        let mut reduced_numerator_1 = Cow::Borrowed(&self.numerator);
        let mut reduced_numerator_2 = Cow::Borrowed(&other.numerator);

        let mut den = Vec::with_capacity(self.denominators.len() + other.denominators.len());

        for (d, p) in &other.denominators {
            if let Some((_, p2)) = self.denominators.iter().find(|(d2, _)| d == d2) {
                den.push((d.clone(), p + p2));
                continue;
            }

            let mut p = *p;
            while p > 0 {
                if let Some(q) = reduced_numerator_1.try_div(d) {
                    reduced_numerator_1 = Cow::Owned(q);
                    p -= 1;
                } else {
                    break;
                }
            }

            if p > 0 {
                den.push((d.clone(), p));
            }
        }

        for (d, p) in &self.denominators {
            if other.denominators.iter().any(|(d2, _)| d == d2) {
                continue;
            }

            let mut p = *p;
            while p > 0 {
                if let Some(q) = reduced_numerator_2.try_div(d) {
                    reduced_numerator_2 = Cow::Owned(q);
                    p -= 1;
                } else {
                    break;
                }
            }

            if p > 0 {
                den.push((d.clone(), p));
            }
        }

        let field = &self.numerator.ring;
        let mut constant = field.one();

        let mut numer_coeff;
        if !field.is_one(&self.denom_coeff) {
            let g = field.gcd(&other.numer_coeff, &self.denom_coeff);
            numer_coeff = field.quot_rem(&other.numer_coeff, &g).0;
            constant = field.quot_rem(&self.denom_coeff, &g).0;
        } else {
            numer_coeff = other.numer_coeff.clone();
        }
        if !field.is_one(&other.denom_coeff) {
            let g = field.gcd(&self.numer_coeff, &other.denom_coeff);
            field.mul_assign(&mut numer_coeff, &field.quot_rem(&self.numer_coeff, &g).0);
            field.mul_assign(&mut constant, &field.quot_rem(&other.denom_coeff, &g).0);
        } else {
            field.mul_assign(&mut numer_coeff, &self.numer_coeff);
        }

        FactorizedRationalPolynomial {
            numerator: reduced_numerator_1.as_ref() * reduced_numerator_2.as_ref(),
            numer_coeff,
            denom_coeff: constant,
            denominators: den,
        }
    }
}

impl<'a, R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent>
    Div<&'a FactorizedRationalPolynomial<R, E>> for &FactorizedRationalPolynomial<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
{
    type Output = FactorizedRationalPolynomial<R, E>;

    fn div(self, other: &'a FactorizedRationalPolynomial<R, E>) -> Self::Output {
        if other.is_one() {
            return self.clone();
        }
        if other.is_zero() {
            panic!("Cannot invert 0");
        }

        if self.get_variables() != other.get_variables() {
            let mut a = self.clone();
            let mut b = other.clone();
            a.unify_variables(&mut b);
            return &a / &b;
        }

        let mut reduced_numerator_1 = Cow::Borrowed(&self.numerator);

        let mut den = Vec::with_capacity(self.denominators.len() + 1);

        let mut reduced_numerator_2 = self.numerator.one();

        for (d, p) in &other.denominators {
            if let Some((_, p2)) = self.denominators.iter().find(|(d2, _)| d == d2) {
                if p > p2 {
                    reduced_numerator_2 = reduced_numerator_2 * &d.pow(p - p2);
                } else if p < p2 {
                    den.push((d.clone(), p2 - p));
                }
            } else {
                reduced_numerator_2 = reduced_numerator_2 * &d.pow(*p);
            }
        }
        for (d, p) in &self.denominators {
            if !other.denominators.iter().any(|(d2, _)| d == d2) {
                den.push((d.clone(), *p));
            }
        }

        // factorize the numerator and normalize
        let r = FactorizedRationalPolynomial::from_num_den(
            self.numerator.one(),
            other.numerator.factor(),
            &self.numerator.ring,
            false,
        );

        for (d, mut p) in r.denominators {
            if let Some((_, p2)) = den.iter_mut().find(|(d2, _)| &d == d2) {
                *p2 += p;
                continue;
            }

            while p > 0 {
                if let Some(q) = reduced_numerator_1.try_div(&d) {
                    reduced_numerator_1 = Cow::Owned(q);
                    p -= 1;
                } else {
                    break;
                }
            }

            if p > 0 {
                den.push((d, p));
            }
        }

        let field = &self.numerator.ring;

        let denom_coeff = field.mul(&r.denom_coeff, &other.numer_coeff);
        let mut numer_coeff = other.denom_coeff.clone();

        // multiply the numerator factor that appears during normalization of the new denominators
        if !field.is_one(&r.numerator.lcoeff()) {
            field.mul_assign(&mut numer_coeff, &r.numerator.lcoeff());
        }

        let mut constant = field.one();

        if !field.is_one(&self.denom_coeff) {
            let g = field.gcd(&numer_coeff, &self.denom_coeff);
            numer_coeff = field.quot_rem(&numer_coeff, &g).0;
            constant = field.quot_rem(&self.denom_coeff, &g).0;
        }
        if !field.is_one(&denom_coeff) {
            let g = field.gcd(&self.numer_coeff, &denom_coeff);
            field.mul_assign(&mut numer_coeff, &field.quot_rem(&self.numer_coeff, &g).0);
            field.mul_assign(&mut constant, &field.quot_rem(&denom_coeff, &g).0);
        } else {
            field.mul_assign(&mut numer_coeff, &self.numer_coeff);
        }
        let num = reduced_numerator_2 * &reduced_numerator_1;
        if !num.ring.is_one(&constant) {
            den.push((num.constant(constant), 1));
        }

        // properly normalize the rational polynomial
        let mut r =
            FactorizedRationalPolynomial::from_num_den(num, den, &self.numerator.ring, false);
        field.mul_assign(&mut r.numer_coeff, &numer_coeff);
        r
    }
}

impl<R: EuclideanDomain + PolynomialGCD<E>, E: PositiveExponent> FactorizedRationalPolynomial<R, E>
where
    FactorizedRationalPolynomial<R, E>: FromNumeratorAndFactorizedDenominator<R, R, E>,
    MultivariatePolynomial<R, E>: Factorize,
{
    /// Compute the partial fraction decomposition of the rational polynomial in `var`.
    pub fn apart(&self, var: usize) -> Vec<Self> {
        if self.denominators.len() == 1 {
            return vec![self.clone()];
        }

        let rat_field = FactorizedRationalPolynomialField::new_from_poly(&self.numerator);

        let mut poly_univ = vec![];
        for (f, p) in &self.denominators {
            let f = f.clone().pow(*p);

            let l = f.to_univariate_polynomial_list(var);
            let mut res: MultivariatePolynomial<_, E> = MultivariatePolynomial::new(
                &rat_field,
                Some(l.len()),
                self.numerator.variables.clone(),
            );

            let mut exp = vec![E::zero(); self.numerator.nvars()];
            for (p, e) in l {
                exp[var] = e;
                res.append_monomial(
                    FactorizedRationalPolynomial::from_num_den(
                        p,
                        vec![],
                        &self.numerator.ring,
                        false,
                    ),
                    &exp,
                );
            }
            poly_univ.push(res);
        }

        let rhs = poly_univ[0].one();
        let deltas = MultivariatePolynomial::diophantine_univariate(&mut poly_univ, &rhs);

        let mut factors = Vec::with_capacity(deltas.len());
        for (d, (p, pe)) in deltas.into_iter().zip(&self.denominators) {
            let mut unfold = rat_field.zero();
            for (c, e) in d
                .coefficients
                .into_iter()
                .zip(d.exponents.chunks(self.numerator.nvars()))
            {
                unfold = &unfold
                    + &(&c
                        * &FactorizedRationalPolynomial::from_num_den(
                            unfold
                                .numerator
                                .monomial(self.numerator.ring.one(), e.to_vec()),
                            vec![],
                            &self.numerator.ring,
                            true,
                        ));
            }

            unfold = &unfold
                * &(&FactorizedRationalPolynomial::from_num_den(
                    self.numerator.clone(),
                    vec![
                        (p.clone(), *pe),
                        (self.numerator.constant(self.denom_coeff.clone()), 1),
                    ],
                    &self.numerator.ring,
                    false,
                ) * &FactorizedRationalPolynomial {
                    numerator: self.numerator.one(),
                    numer_coeff: self.numer_coeff.clone(),
                    denom_coeff: self.numerator.ring.one(),
                    denominators: vec![],
                });

            factors.push(unfold.clone());
        }

        factors
    }
}

#[cfg(test)]
mod test {
    use crate::{
        atom::AtomCore,
        domains::{
            Ring,
            finite_field::{ToFiniteField, Zp},
            integer::Z,
            rational::Q,
        },
        parse,
    };

    #[test]
    fn eval_map() {
        let a = parse!("3 (x^2 + 19) / (4 (x + 44)(x-4)(x+4)) ")
            .to_factorized_rational_polynomial::<_, _, u8>(&Q, &Z, None);
        let f = Zp::new(17);
        let res = a.evaluate_with_coeff_map(|c| c.to_finite_field(&f), &[f.nth(3.into())], &f);
        assert_eq!(res, f.nth(5.into()));
    }
}
