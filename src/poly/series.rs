//! Methods related to series expansion.
//!
//! # Examples
//! ```
//! use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
//!
//! let x = symbol!("x");
//! let a = parse!("(1-cos(x))/sin(x)");
//!
//! let out = a
//!     .series(x, Atom::num(0), 4.into(), true)
//!     .unwrap();
//! println!("{}", out);
//! assert_eq!(out.to_atom(), parse!("1/2*x+1/24*x^3"));
//! ```
use core::panic;
use std::{
    cmp::Ordering,
    ops::{Add, Div, Mul, Neg, Sub},
    sync::Arc,
};

use crate::{
    atom::{Atom, AtomCore, AtomView, FunctionBuilder, Symbol},
    coefficient::CoefficientView,
    domains::{
        EuclideanDomain, InternalOrdering, Ring, SelfRing,
        atom::AtomField,
        integer::Integer,
        rational::{Q, Rational},
    },
    printer::{PrintOptions, PrintState},
};

use super::PolyVariable;

/// A Puiseux series. The truncation order is
/// relative to the lowest degree: i.e., a series in `x` with depth `d` is viewed as
/// `a*x^n*(1+bx+.. + O(x^(d+1)))`.
///
/// # Examples
///
/// A series can be constructed from an [Atom]:
/// ```
/// use symbolica::{atom::{Atom, AtomCore}, parse, symbol};
///
/// let x = symbol!("x");
/// let a = parse!("(1-cos(x))/sin(x)");
///
/// let out = a
///     .series(x, Atom::num(0), 4.into(), true)
///     .unwrap();
/// println!("{}", out);
/// assert_eq!(out.to_atom(), parse!("1/2*x+1/24*x^3"));
/// ```
#[derive(Clone)]
pub struct Series<F: Ring> {
    coefficients: Vec<F::Element>,
    variable: Arc<PolyVariable>,
    expansion_point: F::Element,
    field: F,
    shift: isize, // a shift in units of the ramification that indicates the starting power of the series
    order: usize, // the order of truncation in units of the ramification
    ramification: usize, // the factor that makes the exponents integer
}

impl<F: Ring + std::fmt::Debug> std::fmt::Debug for Series<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.order == 0 {
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

impl<F: Ring + std::fmt::Display> std::fmt::Display for Series<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.format(&PrintOptions::from_fmt(f), PrintState::from_fmt(f), f)
            .map(|_| ())
    }
}

impl<F: Ring> Series<F> {
    /// Constructs a zero series with fixed order. Instead of using this constructor,
    /// prefer to create new series from existing ones, so that the
    /// variable map and field are inherited.
    #[inline]
    pub fn new(
        field: &F,
        cap: Option<usize>,
        variable: Arc<PolyVariable>,
        expansion_point: F::Element,
        order: Rational,
    ) -> Self {
        if order.is_negative() {
            panic!("The order of the series must be positive.");
        }

        Self {
            coefficients: Vec::with_capacity(cap.unwrap_or(0)),
            expansion_point,
            field: field.clone(),
            variable,
            shift: 0,
            order: order.numerator().to_i64().unwrap() as usize
                * order.denominator().to_i64().unwrap() as usize,
            ramification: order.denominator().to_i64().unwrap() as usize,
        }
    }

    /// Constructs a finite-precision zero series, inheriting the field and variable from `self`.
    #[inline]
    pub fn zero(&self) -> Self {
        Self {
            coefficients: vec![],
            field: self.field.clone(),
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            shift: 0,
            order: self.order,
            ramification: 1,
        }
    }

    /// Constructs a finite-precision constant series,
    /// inheriting the field and variable from `self`.
    #[inline]
    pub fn constant(&self, coeff: F::Element) -> Self {
        if self.field.is_zero(&coeff) {
            return self.zero();
        }

        Self {
            coefficients: vec![coeff],
            field: self.field.clone(),
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            shift: 0,
            order: self.order,
            ramification: 1,
        }
    }

    /// Constructs a finite-precision series that is one, inheriting the field and variable from `self`.
    #[inline]
    pub fn one(&self) -> Self {
        Self {
            coefficients: vec![self.field.one()],
            field: self.field.clone(),
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            shift: 0,
            order: self.order,
            ramification: 1,
        }
    }

    /// Constructs a series that is one with infinite digits.
    /// Must be added or multiplied with another finite series.
    #[inline]
    fn one_inf_prec(&self) -> Self {
        Self {
            coefficients: vec![self.field.one()],
            field: self.field.clone(),
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            shift: 0,
            order: i64::MAX as usize, // considered infinity
            ramification: 1,
        }
    }

    /// Constructs a series with a single term.
    #[inline]
    pub fn monomial(&self, coeff: F::Element, exponent: Rational) -> Self {
        if self.field.is_zero(&coeff) {
            return self.zero();
        }

        let n = exponent.numerator().to_i64().unwrap() as isize;
        let d = exponent.denominator().to_i64().unwrap() as usize;

        let ram = Integer::from(self.ramification as i64)
            .lcm(exponent.denominator_ref())
            .to_i64()
            .unwrap() as usize;

        Self {
            coefficients: vec![coeff],
            field: self.field.clone(),
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            shift: n * d as isize,
            order: self.order * ram / d,
            ramification: ram,
        }
    }

    /// Constructs a series that is `x+a`, where `a` is a constant.
    #[inline]
    pub fn shifted_variable(&self, coeff: F::Element) -> Self {
        if self.field.is_zero(&coeff) {
            return self.monomial(self.field.one(), (1, 1).into());
        }

        Self {
            coefficients: vec![coeff, self.field.one()],
            field: self.field.clone(),
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            shift: 0,
            order: self.order,
            ramification: 1,
        }
    }

    /// Get the field of the series coefficients.
    #[inline]
    pub fn get_field(&self) -> &F {
        &self.field
    }

    // Map an index in the coefficient array to its power.
    #[inline]
    fn get_exponent(&self, index: usize) -> Rational {
        Self::get_exponent_from_shift_ram(index, self.shift, self.ramification)
    }

    /// Map an index in the coefficient array to its power, given a shift and ramification.
    fn get_exponent_from_shift_ram(index: usize, shift: isize, ram: usize) -> Rational {
        (Rational::from(index as i64) + Rational::from(shift as i64)) / Rational::from(ram as i64)
    }

    // Map a power to an index in the coefficient array.
    #[inline]
    fn get_index(&self, p: Rational) -> usize {
        let r = p * &Rational::from(self.ramification as i64);
        debug_assert!(r.is_integer());

        let i = r.numerator().to_i64().unwrap() - self.shift as i64;
        debug_assert!(i >= 0);
        i as usize
    }

    /// Get the relative order of the series expansion.
    #[inline]
    pub fn relative_order(&self) -> Rational {
        assert!(self.order < i64::MAX as usize);
        (self.order as i64, self.ramification as i64).into()
    }

    /// Get the absolute order of the series expansion.
    #[inline]
    pub fn absolute_order(&self) -> Rational {
        assert!(self.order < i64::MAX as usize);
        (
            self.order as i64 + self.shift as i64,
            self.ramification as i64,
        )
            .into()
    }

    /// Get the expansion point.
    #[inline]
    pub fn get_expansion_point(&self) -> F::Element {
        self.expansion_point.clone()
    }

    fn change_ramification(&mut self, ram: usize) {
        let ram = Integer::from(self.ramification as i64)
            .lcm(&Integer::from(ram as i64))
            .to_i64()
            .unwrap() as usize;

        if ram == self.ramification {
            return;
        }

        let mut s = Self {
            coefficients: vec![
                self.field.zero();
                self.coefficients.len() * ram / self.ramification
            ],
            field: self.field.clone(),
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            shift: self.shift * (ram / self.ramification) as isize,
            order: self.order * (ram / self.ramification),
            ramification: ram,
        };

        for (i, c) in std::mem::take(&mut self.coefficients)
            .into_iter()
            .enumerate()
        {
            let index = s.get_index(self.get_exponent(i));
            s.coefficients[index] = c;
        }

        *self = s;
    }

    /// Truncate the series to the desired relative order.
    /// If the new order is larger, nothing happens.
    #[inline]
    pub fn truncate_relative_order(&mut self, order: Rational) {
        if self.relative_order() < order {
            return;
        }

        if order.is_negative() {
            panic!("Cannot series expand to negative depth");
        }

        let ram = order.denominator().to_i64().unwrap() as usize;
        self.change_ramification(ram);
        let new_order = (order.numerator().to_i64().unwrap() as usize) * self.ramification / ram;

        self.coefficients.truncate(new_order);
        self.order = new_order;
        self.truncate();
    }

    /// Truncate the series to the desired absolute order.
    /// If the new order is larger, nothing happens.
    #[inline]
    pub fn truncate_absolute_order(&mut self, order: Rational) {
        if self.absolute_order() < order {
            return;
        }

        self.change_ramification(order.denominator().to_i64().unwrap() as usize);

        for i in (0..self.coefficients.len()).rev() {
            if self.get_exponent(i) >= order {
                self.coefficients.pop();
            } else {
                break;
            }
        }

        if self.coefficients.is_empty() {
            self.order = 0;
        } else {
            self.order = self.get_index(order);
        }
    }

    #[inline]
    fn joint_ramification(&self, other: &Self) -> usize {
        Integer::from(self.ramification as i64)
            .lcm(&Integer::from(other.ramification as i64))
            .to_i64()
            .unwrap() as usize
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.coefficients.is_empty()
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        self.coefficients.len() == 1 && self.field.is_one(&self.coefficients[0]) && self.shift == 0
    }

    /// Returns true if the series is constant.
    #[inline]
    pub fn is_constant(&self) -> bool {
        self.coefficients.len() <= 1 && self.shift == 0
    }

    #[inline]
    pub fn get_trailing_coefficient(&self) -> F::Element {
        if self.coefficients.is_empty() {
            self.field.zero()
        } else {
            self.coefficients[0].clone()
        }
    }

    #[inline]
    pub fn get_trailing_exponent(&self) -> Rational {
        self.get_exponent(0)
    }

    #[inline]
    pub fn get_ramification(&self) -> usize {
        self.ramification
    }

    /// Get a copy of the variable/
    pub fn get_variable(&self) -> Arc<PolyVariable> {
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
            .unwrap_or(&self.field.zero())
            .clone()
    }

    pub fn degree(&self) -> Rational {
        if self.order == 0 {
            return 0.into();
        }

        self.get_exponent(self.coefficients.len() - 1)
    }

    /// Compute `self^pow`. `pow` must be a positive integer.
    pub fn npow(&self, mut pow: usize) -> Self {
        if pow == 0 {
            panic!("Cannot create one with infinite precision");
        }

        let mut x = self.clone();
        let mut y = self.one_inf_prec();
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

    /// Multiply the exponents `exp` units of the ramification.
    pub fn mul_exp_units(mut self, exp: isize) -> Self {
        self.shift += exp;
        self
    }

    pub fn mul_coeff(mut self, coeff: &F::Element) -> Self {
        for c in &mut self.coefficients {
            if !self.field.is_zero(c) {
                self.field.mul_assign(c, coeff);
            }
        }

        self.truncate(); // zeros may have occurred

        self
    }

    /// Truncate the leading and trailing non-zeroes.
    fn truncate(&mut self) {
        if self.coefficients.is_empty() {
            return;
        }

        let d = self
            .coefficients
            .iter_mut()
            .rev()
            .position(|c| !self.field.is_zero(c))
            .unwrap_or(self.coefficients.len());

        self.coefficients.truncate(self.coefficients.len() - d);

        if self.coefficients.is_empty() {
            self.shift += self.order as isize;
            self.order = 0;
            return;
        }

        let d = self
            .coefficients
            .iter_mut()
            .position(|c| !self.field.is_zero(c))
            .unwrap_or(self.coefficients.len());

        self.shift += d as isize;
        self.order -= d;
        self.coefficients.drain(0..d);
    }

    /// Remove the constant term, if it is first and it exists.
    fn remove_constant(mut self) -> Self {
        if self.order > 0 && self.shift == 0 {
            self.coefficients[0] = self.field.zero();
            self.truncate();
        }

        self
    }

    /// Get the coefficient of the series at a given exponent.
    pub fn coefficient(&self, exponent: Rational) -> F::Element {
        let r = exponent * &Rational::from(self.ramification as i64);
        if !r.is_integer() {
            return self.field.zero();
        }

        let i = r.numerator().to_i64().unwrap() - self.shift as i64;

        if i >= 0 && i < self.coefficients.len() as i64 {
            self.coefficients[i as usize].clone()
        } else {
            self.field.zero()
        }
    }

    /// Iterate over the terms of the series, yielding the exponent and its coefficient.
    pub fn terms(&self) -> impl Iterator<Item = (Rational, &F::Element)> {
        self.coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| (self.get_exponent(i), c))
    }

    /// Map the coefficients of the series using a function `f`.
    pub fn map_coeff<M: Fn(&F::Element) -> F::Element>(&self, f: M) -> Series<F> {
        let mut s = Series {
            coefficients: self.coefficients.iter().map(f).collect(),
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            shift: self.shift,
            ramification: self.ramification,
            field: self.field.clone(),
            order: self.order,
        };

        s.truncate();
        s
    }
}

impl<'a, R: Ring> IntoIterator for Series<R> {
    type Item = (Rational, R::Element);
    type IntoIter = std::vec::IntoIter<(Rational, R::Element)>;

    /// Create an iterator over the terms of the series, yielding the exponent and its coefficient.
    fn into_iter(self) -> Self::IntoIter {
        let shift = self.shift;
        let ram = self.ramification;
        let v: Vec<_> = self
            .coefficients
            .into_iter()
            .enumerate()
            .map(|(i, c)| (Self::get_exponent_from_shift_ram(i, shift, ram), c))
            .collect();

        v.into_iter()
    }
}

impl<'a, R: Ring> IntoIterator for &'a Series<R> {
    type Item = (Rational, &'a R::Element);
    type IntoIter = std::vec::IntoIter<(Rational, &'a R::Element)>;

    /// Create an iterator over the terms of the series, yielding the exponent and its coefficient.
    fn into_iter(self) -> Self::IntoIter {
        let v: Vec<_> = self
            .coefficients
            .iter()
            .enumerate()
            .map(|(i, c)| (self.get_exponent(i), c))
            .collect();

        v.into_iter()
    }
}

impl<F: Ring> SelfRing for Series<F> {
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
        let v = self.variable.format_string(
            opts,
            PrintState {
                in_exp: true,
                ..state
            },
        );

        if self.coefficients.is_empty() {
            let o = self.absolute_order();
            if opts.mode.is_latex() {
                write!(f, "\\mathcal{{O}}\\left({v}^{{{o}}}\\right)")?;
            } else {
                write!(f, "𝒪({v}^")?;
                Q.format(&o, opts, state.step(false, false, true, false), f)?;
                f.write_char(')')?;
            }
            return Ok(false);
        }

        let add_paren = state.in_product || state.in_exp || state.in_exp_base;
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

        for (e, c) in self.coefficients.iter().enumerate() {
            if self.field.is_zero(c) {
                continue;
            }

            let e = self.get_exponent(e);

            state.in_product = in_product || !e.is_zero();
            state.suppress_one = !e.is_zero();
            let suppressed_one = self.field.format(
                c,
                opts,
                state.step(state.in_sum, state.in_product, false, false),
                f,
            )?;

            if !suppressed_one && !e.is_zero() {
                f.write_char(opts.multiplication_operator)?;
            }

            if e.is_one() {
                write!(f, "{v}")?;
            } else if !e.is_zero() {
                write!(f, "{v}^")?;
                state.suppress_one = false;

                if opts.mode.is_latex() {
                    f.write_char('{')?;
                }

                Q.format(&e, opts, state.step(false, false, true, false), f)?;

                if opts.mode.is_latex() {
                    f.write_char('}')?;
                }
            }

            state.in_sum = true;
        }

        let o = self.absolute_order();

        if opts.mode.is_latex() {
            write!(f, "+\\mathcal{{O}}\\left({v}^{{{o}}}\\right)")?;
        } else {
            write!(f, "+𝒪({v}^")?;
            Q.format(&o, opts, state.step(false, false, true, false), f)?;
            f.write_char(')')?;
        }

        if add_paren {
            f.write_str(")")?;
        }

        Ok(false)
    }
}

impl<F: Ring> InternalOrdering for Series<F> {
    fn internal_cmp(&self, other: &Self) -> Ordering {
        if self.variable != other.variable {
            return self.variable.cmp(&other.variable);
        }

        if self.shift != other.shift {
            return self.shift.cmp(&other.shift);
        }

        if self.ramification != other.ramification {
            return self.ramification.cmp(&other.ramification);
        }

        if self.order != other.order {
            return self.order.cmp(&other.order);
        }

        self.coefficients.internal_cmp(&other.coefficients)
    }
}

impl<F: Ring> PartialEq for Series<F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.variable != other.variable {
            if self.is_constant() != other.is_constant() {
                return false;
            }

            if self.order != other.order {
                // TODO: compare the common orders?
                return false;
            }

            if self.order == 0 {
                return true;
            }

            if self.is_constant() {
                return self.coefficients[0] == other.coefficients[0];
            }

            // TODO: what is expected here?
            unimplemented!("Cannot compare non-constant series with different variable maps yet");
        }

        if self.degree() != other.degree() {
            return false;
        }
        self.coefficients.eq(&other.coefficients)
    }
}

impl<F: Ring> std::hash::Hash for Series<F> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.coefficients.hash(state);
        self.variable.hash(state);
    }
}

impl<F: Ring> Eq for Series<F> {}

impl<R: Ring> PartialOrd for Series<R> {
    /// An ordering of series that has no intuitive meaning.
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.coefficients.internal_cmp(&other.coefficients))
    }
}

impl<F: Ring> Add for Series<F> {
    type Output = Self;

    fn add(mut self, mut other: Self) -> Self::Output {
        assert_eq!(self.field, other.field);
        assert_eq!(self.variable, other.variable);

        if self.shift == other.shift && self.ramification == other.ramification {
            if self.coefficients.len() < other.coefficients.len() {
                std::mem::swap(&mut self, &mut other);
            }

            self.order = self.order.min(other.order);
            self.coefficients.truncate(self.order);

            for (i, c) in other.coefficients.iter().enumerate() {
                if i < self.coefficients.len() {
                    self.field.add_assign(&mut self.coefficients[i], c);
                }
            }

            self.truncate();
            self
        } else {
            let r = self.joint_ramification(&other);
            let r_d_s = r / self.ramification;
            let r_d_o = r / other.ramification;

            let shift = (self.shift * r_d_s as isize).min(other.shift * r_d_o as isize);
            let order_s = (self.order * r_d_s) as isize + self.shift * r_d_s as isize;
            let order_o = (other.order * r_d_o) as isize + other.shift * r_d_o as isize;

            let mut s = Self {
                coefficients: vec![],
                variable: self.variable.clone(),
                expansion_point: self.expansion_point.clone(),
                field: self.field.clone(),
                shift,
                order: (order_s.min(order_o) - shift) as usize,
                ramification: r,
            };

            s.coefficients =
                vec![self.field.zero(); self.coefficients.len().max(other.coefficients.len())];
            for (i, c) in self.coefficients.iter().enumerate() {
                let index = s.get_index(self.get_exponent(i));
                if index < s.order {
                    if index >= s.coefficients.len() {
                        s.coefficients.resize(index * 2, self.field.zero());
                    }
                    s.coefficients[index] = c.clone(); // TODO: prevent copy
                }
            }
            for (i, c) in other.coefficients.iter().enumerate() {
                let index = s.get_index(other.get_exponent(i));
                if index < s.order {
                    if index >= s.coefficients.len() {
                        s.coefficients.resize(index * 2, self.field.zero());
                    }

                    s.field.add_assign(&mut s.coefficients[index], c);
                }
            }

            s.truncate();
            s
        }
    }
}

impl<'a, F: Ring> Add<&'a Series<F>> for &Series<F> {
    type Output = Series<F>;

    fn add(self, other: &'a Series<F>) -> Self::Output {
        (self.clone()).add(other.clone())
    }
}

impl<F: Ring> Sub for Series<F> {
    type Output = Self;

    fn sub(self, other: Self) -> Self::Output {
        self.add(other.neg())
    }
}

impl<'a, F: Ring> Sub<&'a Series<F>> for &Series<F> {
    type Output = Series<F>;

    fn sub(self, other: &'a Series<F>) -> Self::Output {
        (self.clone()).add(other.clone().neg())
    }
}

impl<F: Ring> Neg for Series<F> {
    type Output = Self;
    fn neg(mut self) -> Self::Output {
        // Negate coefficients of all terms.
        for c in &mut self.coefficients {
            *c = self.field.neg(&*c);
        }
        self
    }
}

impl<'a, F: Ring> Mul<&'a Series<F>> for &Series<F> {
    type Output = Series<F>;

    #[inline]
    fn mul(self, rhs: &'a Series<F>) -> Self::Output {
        assert_eq!(self.field, rhs.field);
        assert_eq!(self.variable, rhs.variable);

        let r = self.joint_ramification(rhs);
        let r_d_s = r / self.ramification;
        let r_d_o = r / rhs.ramification;
        let mut res = Series {
            coefficients: vec![],
            field: self.field.clone(),
            variable: self.variable.clone(),
            expansion_point: self.expansion_point.clone(),
            shift: self.shift * r_d_s as isize + rhs.shift * r_d_o as isize,
            order: (self.order * r_d_s).min(rhs.order * r_d_o),
            ramification: r,
        };

        res.coefficients =
            vec![self.field.zero(); self.coefficients.len().max(rhs.coefficients.len())];

        for (e1, c1) in self.coefficients.iter().enumerate() {
            if self.field.is_zero(c1) {
                continue;
            }

            let p1 = self.get_exponent(e1);

            for (e2, c2) in rhs.coefficients.iter().enumerate() {
                let p = &p1 + &rhs.get_exponent(e2);

                if !self.field.is_zero(c2) {
                    let index = res.get_index(p);
                    if index < res.order {
                        if index >= res.coefficients.len() {
                            res.coefficients.resize(index * 2, self.field.zero());
                        }

                        self.field
                            .add_mul_assign(&mut res.coefficients[index], c1, c2);
                    }
                }
            }
        }

        res.truncate();
        res
    }
}

impl<'a, F: Ring> Mul<&'a Series<F>> for Series<F> {
    type Output = Series<F>;

    #[inline]
    fn mul(self, rhs: &'a Series<F>) -> Self::Output {
        (&self) * rhs
    }
}

impl<'a> Div<&'a Series<AtomField>> for &Series<AtomField> {
    type Output = Series<AtomField>;

    fn div(self, other: &'a Series<AtomField>) -> Self::Output {
        other.rpow((-1, 1).into()).unwrap() * self
    }
}

impl<'a> Div<&'a Series<AtomField>> for Series<AtomField> {
    type Output = Series<AtomField>;

    fn div(self, other: &'a Series<AtomField>) -> Self::Output {
        (&self).div(other)
    }
}

impl<F: EuclideanDomain> Series<F> {
    /// Get the content from the coefficients.
    pub fn content(&self) -> F::Element {
        if self.coefficients.is_empty() {
            return self.field.zero();
        }
        let mut c = self.coefficients.first().unwrap().clone();
        for cc in self.coefficients.iter().skip(1) {
            // early return if possible (not possible for rationals)
            if F::one_is_gcd_unit() && self.field.is_one(&c) {
                break;
            }

            c = self.field.gcd(&c, cc);
        }
        c
    }

    /// Divide every coefficient with `other`.
    pub fn div_coeff(mut self, other: &F::Element) -> Self {
        for c in &mut self.coefficients {
            let (quot, rem) = self.field.quot_rem(c, other);
            debug_assert!(self.field.is_zero(&rem));
            *c = quot;
        }
        self
    }

    /// Make the series primitive by removing the content.
    pub fn make_primitive(self) -> Self {
        let c = self.content();
        self.div_coeff(&c)
    }
}

impl Series<AtomField> {
    /// Extract powers of `x` from an expression that comes from simplifying an exponential with logs
    /// i.e.: `exp(c + 3 log(x^5)) = exp(c)*x^15`.
    fn extract_exp_log(&self, e: AtomView, s: AtomView) -> Result<Self, String> {
        if !e.contains(s) {
            return Ok(self.constant(e.to_owned()));
        }

        match e {
            AtomView::Pow(p) => {
                let (b, exp) = p.get_base_exp();

                if b == s {
                    if let AtomView::Num(n) = exp {
                        if let CoefficientView::Natural(n, d, ni, _di) = n.get_coeff_view() {
                            if ni == 0 {
                                Ok(self.monomial(self.field.one(), (n, d).into()))
                            } else {
                                Err("Cannot series expand with complex exponent".to_owned())
                            }
                        } else {
                            Err("Cannot series expand with large exponents yet".to_owned())
                        }
                    } else {
                        Err("Power of variable must be rational".to_owned())
                    }
                } else {
                    Err("Unexpected term in exp-log simplification".to_owned())
                }
            }
            AtomView::Var(_) => Ok(self.monomial(self.field.one(), (1, 1).into())),
            AtomView::Mul(m) => {
                let mut shift_series = self.one_inf_prec();
                for a in m {
                    shift_series = &shift_series * &self.extract_exp_log(a, s)?;
                }

                Ok(shift_series)
            }
            _ => Err("Unexpected term in exp-log simplification".to_owned()),
        }
    }

    pub fn exp(&self) -> Result<Self, String> {
        if self.shift < 0 {
            return Err("Cannot compute the exponential of a series with poles".to_owned());
        }

        if self.order == 0 {
            return Ok(self.one_inf_prec()
                + Series::new(
                    &self.field,
                    Some(1),
                    self.variable.clone(),
                    self.expansion_point.clone(),
                    (self.shift as i64, 1).into(),
                ));
        }

        let c = if self.shift == 0 && self.order > 0 {
            self.coefficients[0].clone()
        } else {
            Atom::new()
        };

        // construct the constant term, log(x) in the argument will be turned into x
        let e = FunctionBuilder::new(Symbol::EXP).add_arg(&c).finish();

        // split the true constant part and the x-dependent part
        let var = self.variable.to_atom() - &self.expansion_point;
        let shift_series = self.extract_exp_log(e.as_view(), var.as_view())?; // FIXME: needs to be inf prec!

        let p = self.clone().remove_constant();

        let mut r = self.one_inf_prec();
        let mut sp = p.clone();
        for i in 1..=self.order {
            let s = sp
                .clone()
                .div_coeff(&Atom::num(Integer::factorial(i as u32)));

            sp = sp * &p;

            r = r + s;
        }

        Ok(r * &shift_series)
    }

    pub fn log(&self) -> Result<Self, String> {
        if self.order == 0 {
            return Err("Log argument needs to have a coefficient".to_owned());
        }

        // construct the log argument, which may contain x
        let c = (self.variable.to_atom() - &self.expansion_point).pow(self.get_exponent(0))
            * &self.coefficients[0];
        // normalize the series to 1 + ..
        let p = self
            .clone()
            .div_coeff(&self.coefficients[0])
            .mul_exp_units(-self.shift)
            - self.one();

        let mut e = self.constant(FunctionBuilder::new(Symbol::LOG).add_arg(&c).finish());
        let mut sp = p.clone();
        for i in 1..=self.order {
            let s = sp.clone().div_coeff(&Atom::num(i as i64));

            sp = sp * &p;

            if i % 2 == 0 {
                e = e - s;
            } else {
                e = e + s;
            }
        }

        Ok(e)
    }

    pub fn sin(&self) -> Result<Self, String> {
        if self.shift < 0 {
            return Err("Cannot compute the sine of a series with poles".to_owned());
        }

        if self.order == 0 {
            return Ok(Series::new(
                &self.field,
                Some(1),
                self.variable.clone(),
                self.expansion_point.clone(),
                (self.shift as i64, 1).into(),
            ));
        }

        let c = if self.shift == 0 {
            self.coefficients[0].clone()
        } else {
            Atom::new()
        };

        if c.contains(self.variable.to_atom()) {
            return Err(
                "Cannot compute the sine of a series with a constant term that depends on x"
                    .to_owned(),
            );
        }

        let p = self.clone().remove_constant();

        let mut e = self.constant(FunctionBuilder::new(Symbol::SIN).add_arg(&c).finish());
        let mut sp = p.clone();
        for i in 1..=self.order {
            let mut b = if i % 2 == 1 {
                FunctionBuilder::new(Symbol::COS).add_arg(&c).finish()
            } else {
                FunctionBuilder::new(Symbol::SIN).add_arg(&c).finish()
            };

            if i % 4 >= 2 {
                b = b.neg();
            }

            let s = sp
                .clone()
                .mul_coeff(&b)
                .div_coeff(&Atom::num(Integer::factorial(i as u32)));

            sp = sp * &p;

            e = e + s;
        }

        Ok(e)
    }

    pub fn cos(&self) -> Result<Self, String> {
        if self.shift < 0 {
            return Err("Cannot compute the sine of a series with poles".to_owned());
        }

        if self.order == 0 {
            return Ok(self.one_inf_prec()
                + Series::new(
                    &self.field,
                    Some(1),
                    self.variable.clone(),
                    self.expansion_point.clone(),
                    (self.shift as i64 * 2, 1).into(),
                ));
        }

        let c = if self.shift == 0 {
            self.coefficients[0].clone()
        } else {
            Atom::new()
        };

        if c.contains(self.variable.to_atom()) {
            return Err(
                "Cannot compute the cosine of a series with a constant term that depends on x"
                    .to_owned(),
            );
        }

        let p = self.clone().remove_constant();

        let mut e = self.constant(FunctionBuilder::new(Symbol::COS).add_arg(&c).finish());
        let mut sp = p.clone();
        for i in 1..=self.order {
            let mut b = if i % 2 == 1 {
                FunctionBuilder::new(Symbol::SIN).add_arg(&c).finish()
            } else {
                -FunctionBuilder::new(Symbol::COS).add_arg(&c).finish()
            };

            if i % 4 < 2 {
                b = b.neg();
            }

            let s = sp
                .clone()
                .mul_coeff(&b)
                .div_coeff(&Atom::num(Integer::factorial(i as u32)));

            sp = sp * &p;

            e = e + s;
        }

        Ok(e)
    }

    /// Take the series to the power of another series.
    pub fn pow(&self, pow: &Self) -> Result<Self, String> {
        (self.log()? * pow).exp()
    }

    /// Take the series to the power of a rational number.
    pub fn rpow(&self, pow: Rational) -> Result<Self, String> {
        if pow.is_zero() {
            Err(
                "Cannot raise series to the power of zero, as this generates infinite precision 1",
            )?;
        }

        if pow.is_integer() && !pow.is_negative() {
            return Ok(self.npow(pow.numerator().to_i64().unwrap() as usize));
        }

        let c = self.get_trailing_coefficient();

        if pow.is_negative() && c.is_zero() {
            Err("Cannot invert series with a zero constant term")?;
        }

        let c = self.coefficients[0].clone();

        // normalize the series to 1 + ...
        let cc = self.clone().div_coeff(&c).mul_exp_units(-self.shift) - self.one();

        let mut r = self.one();
        let mut x = self.one();

        let mut num = Rational::one();

        for i in 1..=self.order {
            num = &num * &(&pow - &(i as i64 - 1).into());

            x = x * &cc;

            let p = x
                .clone()
                .mul_coeff(&Atom::num(num.clone()))
                .div_coeff(&Atom::num(Integer::factorial(i as u32)));

            r = r + p;
        }

        let pow_ram = pow.denominator().to_i64().unwrap() as usize;
        r.change_ramification(pow_ram);

        let p = Rational::from((self.shift as i64, self.ramification as i64)) * &pow;

        let shift = p.numerator().to_i64().unwrap() as isize
            * (r.ramification / p.denominator().to_i64().unwrap() as usize) as isize;

        Ok(r.mul_coeff(&c.pow(pow)).mul_exp_units(shift))
    }

    pub fn to_atom(&self) -> Atom {
        let mut a = Atom::new();
        self.to_atom_into(&mut a);
        a
    }

    pub fn to_atom_into(&self, out: &mut Atom) {
        out.to_num(0.into());

        if self.order == 0 {
            return;
        }

        let v = self.variable.to_atom() - &self.expansion_point;
        for (e, c) in self.coefficients.iter().enumerate() {
            let p = self.get_exponent(e);
            if !c.is_zero() {
                *out = &*out + &(v.pow(p) * c);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        atom::{Atom, AtomCore},
        parse, symbol,
    };

    #[test]
    fn map_coeff() {
        let a = parse!("((v1+1)^2 - (v1^2 + 2v1 + 1))/v2")
            .series(symbol!("v2"), Atom::Zero, 0.into(), true)
            .unwrap();
        assert_eq!(a.get_trailing_exponent(), -1);

        let b = a.map_coeff(|x| x.expand());
        assert_eq!(b.get_trailing_exponent(), 1);
    }
}
