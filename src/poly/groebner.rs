//! Compute Groebner bases for polynomial ideals.
//!
//! # Examples
//! ```
//! use symbolica::prelude::*;
//!
//! let polys = [
//!     "v1 v2 v3 v4 - 1",
//!     "v1 v2 v3 + v1 v2 v4 + v1 v3 v4 + v2 v3 v4",
//!     "v1 v2 + v2 v3 + v1 v4 + v3 v4",
//!     "v1 + v2 + v3 + v4",
//! ];
//!
//! let ideal: Vec<MultivariatePolynomial<_, u16>> = polys
//! .iter()
//! .map(|x| {
//!     let a = parse!(x);
//!     a.to_polynomial(&Zp::new(13), None)
//! })
//! .collect();
//!
//! // compute the Groebner basis with lex ordering
//! let gb = GroebnerBasis::new(&ideal, false);
//!
//! // verify the result is correct
//! let res = [
//!     "v4+v3+v2+v1",
//!     "v4^2+2*v2*v4+v2^2",
//!     "11*v4^2+v3*v4+v3^2*v4^4-v2*v4+v2*v3",
//!     "-v4+v4^5-v2+v2*v4^4",
//!     "-v4-v3+v3^2*v4^3+v3^3*v4^2",
//!     "1-v4^4-v3^2*v4^2+v3^2*v4^6",
//! ];
//!
//! let res: Vec<MultivariatePolynomial<_, u16>> = res
//! .iter()
//! .map(|x| {
//!     let a = parse!(x);
//!     a.to_polynomial(&Zp::new(13), ideal[0].variables().clone())
//! })
//! .collect();
//!
//! assert_eq!(gb.system, res);
//! ```

use std::{
    any::TypeId, cmp::Ordering, collections::BTreeSet, marker::PhantomData, rc::Rc, sync::Arc,
};

use ahash::{HashMap, HashSet};

use crate::{
    atom::{Atom, AtomCore},
    domains::{
        Field, InternalOrdering, Ring, RingOps, Set,
        algebraic_number::{AlgebraicExtension, AlgebraicNumber, AlgebraicQuotient},
        finite_field::{FiniteFieldCore, Zp},
        integer::IntegerRing,
        rational::{Q, Rational, RationalField},
        rational_polynomial::{RationalPolynomial, RationalPolynomialField},
    },
    state::Workspace,
    tensors::matrix::{Matrix, MatrixError},
    transcendental::{TranscendentalFunctions, root_var},
};

use super::{
    Exponent, LexOrder, MonomialOrder, PolyVariable, PositiveExponent, factor::Factorize,
    polynomial::MultivariatePolynomial,
};

#[derive(Debug)]
pub struct CriticalPair<R: Field, E: Exponent, O: MonomialOrder> {
    lcm_diff_first: Vec<E>,
    poly_first: Rc<MultivariatePolynomial<R, E, O>>,
    index_first: usize,
    lcm_diff_sec: Vec<E>,
    poly_sec: Rc<MultivariatePolynomial<R, E, O>>,
    index_sec: usize,
    lcm: Vec<E>,
    degree: E,
    disjoint: bool,
}

impl<R: Field, E: Exponent, O: MonomialOrder> CriticalPair<R, E, O> {
    fn new(
        f1: Rc<MultivariatePolynomial<R, E, O>>,
        f2: Rc<MultivariatePolynomial<R, E, O>>,
        index1: usize,
        index2: usize,
    ) -> CriticalPair<R, E, O> {
        // determine the lcm of leading monomials
        let lcm: Vec<E> = f1
            .max_exp()
            .iter()
            .zip(f2.max_exp())
            .map(|(e1, e2)| *e1.max(e2))
            .collect();

        let lcm_diff_first: Vec<E> = lcm
            .iter()
            .zip(f1.max_exp())
            .map(|(e1, e2)| *e1 - *e2)
            .collect();

        let lcm_diff_sec: Vec<E> = lcm
            .iter()
            .zip(f2.max_exp())
            .map(|(e1, e2)| *e1 - *e2)
            .collect();

        CriticalPair {
            disjoint: lcm_diff_first == f2.max_exp(),
            degree: lcm.iter().cloned().sum::<E>(),
            lcm_diff_first,
            poly_first: f1,
            index_first: index1,
            lcm_diff_sec,
            poly_sec: f2,
            index_sec: index2,
            lcm,
        }
    }
}

/// A position of a monomial in the reduction matrix.
pub struct MonomialData {
    present: bool,
    column: usize,
}

/// A Groebner basis for a polynomial ideal.
pub struct GroebnerBasis<R: Field, E: Exponent, O: MonomialOrder> {
    pub system: Vec<MultivariatePolynomial<R, E, O>>,
    pub print_stats: bool,
}

/// One solution of a zero-dimensional polynomial system.
///
/// All values live in the same simple algebraic extension. A degree-one
/// extension represents the base field itself, so callers do not need a
/// separate value variant for rational and algebraic solutions.
#[derive(Clone, Debug)]
pub struct PolynomialSolution<R: Ring> {
    field: AlgebraicExtension<R>,
    values: HashMap<PolyVariable, AlgebraicNumber<R>>,
    // Preserve the minimal expression found when a value is introduced,
    // before later adjunctions embed it into a potentially much larger field.
    atom_values: HashMap<PolyVariable, Atom>,
}

impl<R: Ring> PolynomialSolution<R> {
    pub fn field(&self) -> &AlgebraicExtension<R> {
        &self.field
    }

    pub fn values(&self) -> &HashMap<PolyVariable, AlgebraicNumber<R>> {
        &self.values
    }

    pub fn get(&self, variable: &PolyVariable) -> Option<&AlgebraicNumber<R>> {
        self.values.get(variable)
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn into_parts(
        self,
    ) -> (
        AlgebraicExtension<R>,
        HashMap<PolyVariable, AlgebraicNumber<R>>,
    ) {
        (self.field, self.values)
    }
}

impl PolynomialSolution<RationalField> {
    /// Convert a polynomial-native solution to expression atoms.
    pub fn to_atom_map(&self) -> Result<HashMap<PolyVariable, Atom>, String> {
        self.values
            .iter()
            .map(|(variable, value)| {
                Ok((
                    variable.clone(),
                    self.atom_values
                        .get(variable)
                        .cloned()
                        .unwrap_or_else(|| self.field.element_to_atom_simplified(value)),
                ))
            })
            .collect()
    }
}

/// The rational-function field `Q(parameters)`.
pub type ParameterField<E = u16> = RationalPolynomialField<IntegerRing, E>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum ParametricExtensionConstruction<E: PositiveExponent> {
    Primitive,
    Adjoined {
        base: Arc<ParametricExtension<E>>,
        polynomial: MultivariatePolynomial<AlgebraicQuotient<ParameterField<E>>, u16>,
        shift: usize,
    },
}

/// A formal algebraic extension of `Q(parameters)`.
///
/// This type deliberately contains an [`AlgebraicQuotient`] rather than an
/// [`AlgebraicExtension`]: no analytic embedding exists until all parameters
/// have been specialized.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParametricExtension<E: PositiveExponent = u16> {
    quotient: AlgebraicQuotient<ParameterField<E>>,
    parameters: Arc<Vec<PolyVariable>>,
    generic_conditions: Vec<MultivariatePolynomial<IntegerRing, E>>,
    construction: ParametricExtensionConstruction<E>,
}

impl<E: PositiveExponent> ParametricExtension<E> {
    pub fn quotient(&self) -> &AlgebraicQuotient<ParameterField<E>> {
        &self.quotient
    }

    pub fn polynomial(&self) -> &MultivariatePolynomial<ParameterField<E>, u16> {
        self.quotient.poly()
    }

    pub fn parameters(&self) -> &[PolyVariable] {
        &self.parameters
    }

    /// Polynomials that must remain nonzero for the generic solution to apply.
    pub fn generic_conditions(&self) -> &[MultivariatePolynomial<IntegerRing, E>] {
        &self.generic_conditions
    }

    fn coefficient_to_atom(coefficient: &RationalPolynomial<IntegerRing, E>) -> Atom {
        let zero = Atom::Zero;
        let mut temporary_values = HashMap::default();
        for (index, variable) in coefficient.get_variables().iter().enumerate() {
            if matches!(variable, PolyVariable::Temporary(_)) {
                assert!(
                    coefficient.numerator.degree(index) == E::zero()
                        && coefficient.denominator.degree(index) == E::zero(),
                    "An active internal variable leaked into a parametric coefficient"
                );
                temporary_values.insert(variable.clone(), zero.as_view());
            }
        }

        let mut result = Atom::default();
        Workspace::get_local().with(|workspace| {
            coefficient.to_expression_with_map(workspace, &temporary_values, &mut result)
        });
        result
    }

    fn parameter_point(
        variables: &[PolyVariable],
        values: &HashMap<PolyVariable, Rational>,
    ) -> Result<Vec<Rational>, String> {
        variables
            .iter()
            .map(|variable| {
                values
                    .get(variable)
                    .cloned()
                    .ok_or_else(|| format!("Missing value for parameter {variable}"))
            })
            .collect()
    }

    fn evaluate_coefficient(
        coefficient: &RationalPolynomial<IntegerRing, E>,
        values: &HashMap<PolyVariable, Rational>,
    ) -> Result<Rational, String> {
        let point = coefficient
            .get_variables()
            .iter()
            .enumerate()
            .map(|(index, variable)| {
                if let Some(value) = values.get(variable) {
                    Ok(value.clone())
                } else if coefficient.numerator.degree(index) == E::zero()
                    && coefficient.denominator.degree(index) == E::zero()
                {
                    Ok(Rational::zero())
                } else {
                    Err(format!("Missing value for parameter {variable}"))
                }
            })
            .collect::<Result<Vec<_>, String>>()?;
        let denominator = coefficient.denominator.evaluate_with_coeff_map(
            |integer| integer.to_rational(),
            &point,
            &Q,
        );
        if denominator.is_zero() {
            return Err(format!(
                "A denominator vanishes while specializing {}",
                coefficient
            ));
        }
        Ok(Q.div(
            &coefficient.numerator.evaluate_with_coeff_map(
                |integer| integer.to_rational(),
                &point,
                &Q,
            ),
            &denominator,
        ))
    }

    fn specialize_polynomial(
        polynomial: &MultivariatePolynomial<ParameterField<E>, u16>,
        values: &HashMap<PolyVariable, Rational>,
    ) -> Result<MultivariatePolynomial<RationalField, u16>, String> {
        let mut result = MultivariatePolynomial::new(
            &Q,
            Some(polynomial.nterms()),
            polynomial.variables().clone(),
        );
        for term in polynomial {
            let coefficient = Self::evaluate_coefficient(term.coefficient, values)?;
            result.append_monomial(coefficient, term.exponents);
        }
        Ok(result)
    }

    fn check_generic_conditions(
        &self,
        values: &HashMap<PolyVariable, Rational>,
    ) -> Result<(), String> {
        for condition in &self.generic_conditions {
            let point = Self::parameter_point(condition.get_vars_ref(), values)?;
            let value =
                condition.evaluate_with_coeff_map(|integer| integer.to_rational(), &point, &Q);
            if value.is_zero() {
                return Err(format!(
                    "The specialization lies on the exceptional locus {} = 0",
                    condition
                ));
            }
        }
        Ok(())
    }

    fn specialize_extension_polynomial(
        polynomial: &MultivariatePolynomial<AlgebraicQuotient<ParameterField<E>>, u16>,
        values: &HashMap<PolyVariable, Rational>,
        field: &AlgebraicExtension<RationalField>,
    ) -> Result<MultivariatePolynomial<AlgebraicExtension<RationalField>, u16>, String> {
        let mut result = MultivariatePolynomial::new(
            field,
            Some(polynomial.nterms()),
            polynomial.variables().clone(),
        );
        for term in polynomial {
            let coefficient_polynomial =
                Self::specialize_polynomial(term.coefficient.poly(), values)?;
            let coefficient = field.try_to_element(coefficient_polynomial)?;
            result.append_monomial(coefficient, term.exponents);
        }
        Ok(result)
    }

    fn specialize_field(
        &self,
        values: &HashMap<PolyVariable, Rational>,
        conjugates: &[usize],
    ) -> Result<AlgebraicExtension<RationalField>, String> {
        self.check_generic_conditions(values)?;

        match &self.construction {
            ParametricExtensionConstruction::Primitive => {
                let [conjugate] = conjugates else {
                    return Err(format!(
                        "Expected one conjugate for a primitive parametric extension, got {}",
                        conjugates.len()
                    ));
                };
                let polynomial = Self::specialize_polynomial(self.quotient.poly(), values)?;
                let degree = polynomial.degree(0) as usize;
                if *conjugate >= degree {
                    return Err(format!(
                        "Conjugate {conjugate} is out of bounds after specialization to degree {degree}"
                    ));
                }

                // This goes through ROOT_CACHE and gives the formal conjugate
                // its analytic meaning only after parameter specialization.
                polynomial
                    .to_univariate_from_univariate(0)
                    .root(*conjugate)
                    .ok_or_else(|| {
                        format!("Could not isolate conjugate {conjugate} of {polynomial}")
                    })?;

                Ok(AlgebraicExtension::new_with_embedding(
                    polynomial, *conjugate,
                ))
            }
            ParametricExtensionConstruction::Adjoined {
                base, polynomial, ..
            } => {
                let Some((&conjugate, base_conjugates)) = conjugates.split_last() else {
                    return Err("Missing conjugates for an adjoined parametric extension".into());
                };
                let base_field = base.specialize_field(values, base_conjugates)?;
                let generic_degree = polynomial.degree(0) as usize;
                let polynomial =
                    Self::specialize_extension_polynomial(polynomial, values, &base_field)?;
                let degree = polynomial.degree(0) as usize;
                if degree != generic_degree {
                    return Err(format!(
                        "The degree of an adjoined polynomial changed from {generic_degree} to {degree}"
                    ));
                }
                if conjugate >= degree {
                    return Err(format!(
                        "Conjugate {conjugate} is out of bounds after specialization to degree {degree}"
                    ));
                }

                let nonconstant_factors = polynomial
                    .factor()
                    .into_iter()
                    .filter(|(factor, _)| !factor.is_constant())
                    .count();
                if nonconstant_factors != 1 {
                    return Err(format!(
                        "An adjoined polynomial becomes reducible under this specialization: {polynomial}"
                    ));
                }

                let extension = AlgebraicExtension::new_with_embedding(polynomial, conjugate);
                let variable = self.quotient.poly().get_vars_ref()[0].clone();
                Ok(base_field
                    .adjoin_with_embedding(&extension, Some(variable))
                    .0)
            }
        }
    }

    fn specialize_value(
        &self,
        value: &AlgebraicNumber<ParameterField<E>>,
        values: &HashMap<PolyVariable, Rational>,
        field: &AlgebraicExtension<RationalField>,
    ) -> Result<AlgebraicNumber<RationalField>, String> {
        let polynomial = Self::specialize_polynomial(value.poly(), values)?;
        field.try_to_element(polynomial)
    }

    fn primitive_atom(&self, conjugates: &[usize]) -> Atom {
        match &self.construction {
            ParametricExtensionConstruction::Primitive => {
                let root_variable: Atom = root_var().into();
                let mut defining_polynomial = Atom::Zero;
                for term in self.polynomial() {
                    let coefficient = Self::coefficient_to_atom(term.coefficient);
                    if term.exponents[0] == 0 {
                        defining_polynomial += coefficient;
                    } else {
                        defining_polynomial +=
                            coefficient * root_variable.pow(term.exponents[0] as u64);
                    }
                }
                defining_polynomial.root(conjugates[0])
            }
            ParametricExtensionConstruction::Adjoined {
                base,
                polynomial,
                shift,
            } => {
                let (&conjugate, base_conjugates) = conjugates
                    .split_last()
                    .expect("An adjoined extension must have a conjugate path");
                let base_generator = base.primitive_atom(base_conjugates);
                let root_variable: Atom = root_var().into();
                let mut defining_polynomial = Atom::Zero;
                for term in polynomial {
                    let coefficient =
                        base.value_to_atom(term.coefficient, base_conjugates, &base_generator);
                    if term.exponents[0] == 0 {
                        defining_polynomial += coefficient;
                    } else {
                        defining_polynomial +=
                            coefficient * root_variable.pow(term.exponents[0] as u64);
                    }
                }
                defining_polynomial.root(conjugate) + base_generator * Atom::num(*shift as i64)
            }
        }
    }

    fn value_to_atom(
        &self,
        value: &AlgebraicNumber<ParameterField<E>>,
        _conjugates: &[usize],
        primitive: &Atom,
    ) -> Atom {
        let mut result = Atom::Zero;
        for term in value.poly() {
            let coefficient = Self::coefficient_to_atom(term.coefficient);
            if term.exponents[0] == 0 {
                result += coefficient;
            } else {
                result += coefficient * primitive.clone().pow(term.exponents[0] as u64);
            }
        }
        result
    }
}

/// One algebraic value in a formal parametric extension.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ParametricRoot<E: PositiveExponent = u16> {
    field: Arc<ParametricExtension<E>>,
    value: AlgebraicNumber<ParameterField<E>>,
    conjugates: Arc<Vec<usize>>,
}

impl<E: PositiveExponent> ParametricRoot<E> {
    pub fn field(&self) -> &ParametricExtension<E> {
        &self.field
    }

    pub fn polynomial(&self) -> &MultivariatePolynomial<ParameterField<E>, u16> {
        self.field.polynomial()
    }

    pub fn conjugate(&self) -> usize {
        *self
            .conjugates
            .last()
            .expect("A parametric root must have a conjugate")
    }

    /// The root index selected at every level of the adjoining tower.
    pub fn conjugates(&self) -> &[usize] {
        &self.conjugates
    }

    pub fn value(&self) -> &AlgebraicNumber<ParameterField<E>> {
        &self.value
    }

    pub fn into_parts(
        self,
    ) -> (
        Arc<ParametricExtension<E>>,
        AlgebraicNumber<ParameterField<E>>,
        Arc<Vec<usize>>,
    ) {
        (self.field, self.value, self.conjugates)
    }

    pub fn specialize(
        &self,
        values: &HashMap<PolyVariable, Rational>,
    ) -> Result<
        (
            AlgebraicExtension<RationalField>,
            AlgebraicNumber<RationalField>,
        ),
        String,
    > {
        let field = self.field.specialize_field(values, &self.conjugates)?;
        let value = self.field.specialize_value(&self.value, values, &field)?;
        Ok((field, value))
    }

    /// Convert the formal parametric root to an expression-level `root`.
    pub fn to_atom(&self) -> Atom {
        let primitive = self.field.primitive_atom(&self.conjugates);
        self.field
            .value_to_atom(&self.value, &self.conjugates, &primitive)
    }
}

impl<E: PositiveExponent> std::fmt::Display for ParametricRoot<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_atom())
    }
}

/// One solution branch over `Q(parameters)`.
///
/// All values share one collapsed formal quotient field and one path of root
/// indices through its adjoining history.
#[derive(Clone, Debug)]
pub struct ParametricSolution<E: PositiveExponent = u16> {
    field: Arc<ParametricExtension<E>>,
    conjugates: Arc<Vec<usize>>,
    values: HashMap<PolyVariable, AlgebraicNumber<ParameterField<E>>>,
}

impl<E: PositiveExponent> ParametricSolution<E> {
    pub fn field(&self) -> &ParametricExtension<E> {
        &self.field
    }

    pub fn conjugate(&self) -> usize {
        *self
            .conjugates
            .last()
            .expect("A parametric solution must have a conjugate")
    }

    /// The root index selected at every level of the adjoining tower.
    pub fn conjugates(&self) -> &[usize] {
        &self.conjugates
    }

    pub fn values(&self) -> &HashMap<PolyVariable, AlgebraicNumber<ParameterField<E>>> {
        &self.values
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn get(&self, variable: &PolyVariable) -> Option<ParametricRoot<E>> {
        self.values
            .get(variable)
            .cloned()
            .map(|value| ParametricRoot {
                field: self.field.clone(),
                value,
                conjugates: self.conjugates.clone(),
            })
    }

    pub fn to_atom_map(&self) -> HashMap<PolyVariable, Atom> {
        self.values
            .iter()
            .map(|(variable, value)| {
                let root = ParametricRoot {
                    field: self.field.clone(),
                    value: value.clone(),
                    conjugates: self.conjugates.clone(),
                };
                (variable.clone(), root.to_atom())
            })
            .collect()
    }

    pub fn specialize(
        &self,
        values: &HashMap<PolyVariable, Rational>,
    ) -> Result<PolynomialSolution<RationalField>, String> {
        let field = self.field.specialize_field(values, &self.conjugates)?;
        let solution_values = self
            .values
            .iter()
            .map(|(variable, value)| {
                Ok((
                    variable.clone(),
                    self.field.specialize_value(value, values, &field)?,
                ))
            })
            .collect::<Result<HashMap<_, _>, String>>()?;
        Ok(PolynomialSolution {
            field,
            values: solution_values,
            atom_values: HashMap::default(),
        })
    }

    pub fn into_parts(
        self,
    ) -> (
        Arc<ParametricExtension<E>>,
        Arc<Vec<usize>>,
        HashMap<PolyVariable, AlgebraicNumber<ParameterField<E>>>,
    ) {
        (self.field, self.conjugates, self.values)
    }
}

struct OrderedMonomial<E, O> {
    exponents: Vec<E>,
    order: PhantomData<O>,
}

impl<E, O> OrderedMonomial<E, O> {
    fn new(exponents: Vec<E>) -> Self {
        Self {
            exponents,
            order: PhantomData,
        }
    }
}

impl<E: Exponent, O: MonomialOrder> PartialEq for OrderedMonomial<E, O> {
    fn eq(&self, other: &Self) -> bool {
        self.exponents == other.exponents
    }
}

impl<E: Exponent, O: MonomialOrder> Eq for OrderedMonomial<E, O> {}

impl<E: Exponent, O: MonomialOrder> PartialOrd for OrderedMonomial<E, O> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<E: Exponent, O: MonomialOrder> Ord for OrderedMonomial<E, O> {
    fn cmp(&self, other: &Self) -> Ordering {
        O::cmp(&self.exponents, &other.exponents)
    }
}

impl<R: Field + Echelonize, E: Exponent, O: MonomialOrder> GroebnerBasis<R, E, O> {
    /// Construct a Groebner basis for a polynomial ideal.
    ///
    /// Progress can be monitored with `print_stats`.
    pub fn new(
        ideal: &[MultivariatePolynomial<R, E, O>],
        print_stats: bool,
    ) -> GroebnerBasis<R, E, O> {
        if ideal.is_empty() {
            return GroebnerBasis {
                system: Vec::new(),
                print_stats,
            };
        }

        let mut ideal = ideal.to_vec();
        MultivariatePolynomial::unify_variables_list(&mut ideal);

        let mut b = GroebnerBasis {
            system: ideal,
            print_stats,
        };

        b.f4();
        b.reduce_basis()
    }

    #[inline]
    fn simplify(
        tab: &mut Vec<(Vec<E>, Rc<MultivariatePolynomial<R, E, O>>)>,
        lcm: &[E],
    ) -> Rc<MultivariatePolynomial<R, E, O>> {
        for (m, f) in tab.iter().rev() {
            if m == lcm {
                return f.clone();
            }

            if lcm.iter().zip(m).all(|(el, em)| *el >= *em) {
                let diff: Vec<_> = lcm.iter().zip(m).map(|(el, em)| *el - *em).collect();
                let a = Rc::new((**f).clone().mul_exp(&diff));
                tab.push((lcm.to_vec(), a.clone()));
                return a;
            }
        }
        panic!("Unknown polynomial associated with exponent map {lcm:?}");
    }

    /// The F4 algorithm for computing a Groebner basis.
    ///
    /// Adapted from [A new efficient algorithm for computing Gröbner bases (F4)](https://doi.org/10.1016/S0022-4049(99)00005-5) by Jean-Charles Faugére.
    ///
    fn f4(&mut self) {
        let nvars = self.system[0].nvars();
        let field = self.system[0].ring().clone();

        let mut simplifications = vec![];
        let mut basis = vec![];
        let mut critical_pairs = vec![];

        for (i, f) in self.system.drain(..).enumerate() {
            let poly = Rc::new(f.clone().make_monic());
            simplifications.push(vec![(vec![E::zero(); nvars], poly.clone())]);
            Self::update(&mut basis, &mut critical_pairs, poly, i);
        }

        let mut matrix = vec![];
        let mut zp_matrix = vec![];

        let mut all_monomials: HashMap<Vec<E>, MonomialData> = HashMap::default();
        let mut current_monomials = vec![];
        let mut sorted_monomial_indices = vec![];
        let mut exp = vec![E::zero(); nvars];
        let mut new_polys = vec![];
        let mut selected_polys = vec![];

        let mut buffer = vec![];
        let mut zp_buffer = vec![];
        let mut pivots: Vec<Option<usize>> = vec![];

        let mut iter_count = 1;
        while !critical_pairs.is_empty() {
            // select the critical pairs with the lowest lcm degree
            let lowest_lcm_deg = critical_pairs.iter().map(|x| x.degree).min().unwrap();

            if self.print_stats {
                println!(
                    "Iteration {}:\n\tDegree={}, Basis length={}, Critical pairs={}",
                    iter_count,
                    lowest_lcm_deg,
                    basis.len(),
                    critical_pairs.len(),
                );
            }
            iter_count += 1;

            selected_polys.clear();
            let mut i = critical_pairs.len() - 1;

            let mut l_tmp = vec![];
            loop {
                if critical_pairs[i].degree == lowest_lcm_deg {
                    let pair = critical_pairs.swap_remove(i);

                    let e = [
                        (pair.index_first, pair.lcm_diff_first),
                        (pair.index_sec, pair.lcm_diff_sec),
                    ];
                    for poly_info in e {
                        if !l_tmp.contains(&poly_info) {
                            let new_f1 =
                                Self::simplify(&mut simplifications[poly_info.0], &poly_info.1);
                            selected_polys.push(new_f1);
                            l_tmp.push(poly_info);
                        }
                    }
                }

                if i == 0 {
                    break;
                }

                i -= 1;
            }

            // symbolic preprocessing

            for x in all_monomials.values_mut() {
                x.present = false;
            }

            // flag all head monomials as done
            for p in &selected_polys {
                if let Some(m) = all_monomials.get_mut(p.max_exp()) {
                    m.present = true;
                } else {
                    all_monomials.insert(
                        p.max_exp().to_vec(),
                        MonomialData {
                            present: true,
                            column: 0,
                        },
                    );
                }
            }

            new_polys.clear();
            let mut i = 0;
            while i < selected_polys.len() {
                for monom in selected_polys[i].exponents_iter() {
                    if let Some(m) = all_monomials.get_mut(monom) {
                        if m.present {
                            continue;
                        }
                        m.present = true;
                    } else {
                        all_monomials.insert(
                            monom.to_vec(),
                            MonomialData {
                                present: true,
                                column: 0,
                            },
                        );
                    }

                    // search for a reducer and select the smallest for better performance
                    if let Some((index, g)) = basis
                        .iter()
                        .filter(|g| monom.iter().zip(g.1.max_exp()).all(|(pe, ge)| *pe >= *ge))
                        .min_by_key(|g| g.1.nterms())
                    {
                        for ((e, pe), ge) in exp.iter_mut().zip(monom).zip(g.max_exp()) {
                            *e = *pe - *ge;
                        }

                        let pp = Self::simplify(&mut simplifications[*index], &exp);
                        new_polys.push(pp);
                    }
                }

                i += 1;

                selected_polys.append(&mut new_polys);
            }

            // construct a matrix that is sparse in the columns

            current_monomials.clear();
            sorted_monomial_indices.clear();

            for (k, v) in &all_monomials {
                if v.present {
                    current_monomials.extend_from_slice(k);
                }
            }

            for i in 0..(current_monomials.len() / nvars) {
                sorted_monomial_indices.push(i);
            }

            if self.print_stats {
                println!(
                    "\tMonomials in use={}/{}",
                    sorted_monomial_indices.len(),
                    all_monomials.len()
                );
                println!(
                    "\tMatrix shape={}x{}, density={:.2}%",
                    selected_polys.len(),
                    sorted_monomial_indices.len(),
                    selected_polys.iter().map(|i| i.nterms()).sum::<usize>() as f64
                        / (sorted_monomial_indices.len() as f64 * selected_polys.len() as f64)
                        * 100.
                );
            }

            // sort monomials in descending order
            sorted_monomial_indices.sort_unstable_by(|e1, e2| {
                O::cmp(
                    &current_monomials[*e2 * nvars..(*e2 + 1) * nvars],
                    &current_monomials[*e1 * nvars..(*e1 + 1) * nvars],
                )
            });

            for (column, index) in sorted_monomial_indices.iter().enumerate() {
                all_monomials
                    .get_mut(&current_monomials[index * nvars..(index + 1) * nvars])
                    .unwrap()
                    .column = column;
            }

            echelonize(
                &mut matrix,
                &mut zp_matrix,
                &mut selected_polys,
                &all_monomials,
                &sorted_monomial_indices,
                &field,
                &mut buffer,
                &mut zp_buffer,
                &mut pivots,
                self.print_stats,
            );

            // construct new polynomials
            for m in &matrix {
                let lmi = sorted_monomial_indices[m[0].1];
                let lm = &current_monomials[lmi * nvars..(lmi + 1) * nvars];

                // create the new polynomial in the proper order
                let mut poly = selected_polys[0].zero_with_capacity(m.len());
                for (coeff, col) in m.iter().rev() {
                    let index = sorted_monomial_indices[*col];
                    let exp = &current_monomials[index * nvars..(index + 1) * nvars];
                    poly.append_monomial(coeff.clone(), exp);
                }

                let poly = Rc::new(poly);

                if selected_polys.iter().all(|p| p.max_exp() != lm) {
                    let new_index = simplifications.len();
                    simplifications.push(vec![(vec![E::zero(); nvars], poly.clone())]);

                    Self::update(&mut basis, &mut critical_pairs, poly, new_index);
                } else {
                    // update entries in the tab with simpler polynomials
                    let mut diff = vec![E::zero(); nvars];
                    'bf: for (g_ind, g) in &basis {
                        if poly
                            .last_exponents()
                            .iter()
                            .zip(g.last_exponents())
                            .all(|(pi, gi)| *pi >= *gi)
                        {
                            for ((d, pi), gi) in diff
                                .iter_mut()
                                .zip(poly.last_exponents())
                                .zip(g.last_exponents())
                            {
                                *d = *pi - *gi;
                            }

                            for (diff_e, p) in &mut simplifications[*g_ind] {
                                if diff == *diff_e {
                                    *p = poly.clone();
                                    continue 'bf;
                                }
                            }

                            // new polynomial
                            simplifications[*g_ind].push((diff.clone(), poly.clone()));
                        }
                    }
                }
            }
        }

        self.system = basis.into_iter().map(|x| (*x.1).clone()).collect();
    }
}

impl<R: Field, E: Exponent, O: MonomialOrder> MultivariatePolynomial<R, E, O> {
    /// Completely reduce the polynomial w.r.t the polynomials `gs`.
    /// For example reducing `f=y^2+x` by `g=[x]` yields `y^2`.
    pub fn reduce(
        &self,
        gs: &[MultivariatePolynomial<R, E, O>],
    ) -> MultivariatePolynomial<R, E, O> {
        if gs.iter().any(|x| self.variables() != x.variables()) {
            let mut sys: Vec<_> = gs.to_vec();
            sys.push(self.clone());
            Self::unify_variables_list(&mut sys);
            return sys.last().unwrap().reduce(&sys[..gs.len()]);
        }

        let mut q = self.zero_with_capacity(self.nterms());
        let mut r = self.clone();

        let mut rest_coeff = vec![];
        let mut rest_exponents = vec![];

        let mut monom = vec![E::zero(); self.nvars()];

        'term: while !r.is_zero() {
            // find a divisor that has the least amount of terms
            while let Some(g) = gs
                .iter()
                .filter(|g| {
                    r.max_exp()
                        .iter()
                        .zip(g.max_exp())
                        .all(|(h1, h2)| *h1 >= *h2)
                })
                .min_by_key(|g| g.nterms())
            {
                for ((e, e1), e2) in monom.iter_mut().zip(r.max_exp()).zip(g.max_exp()) {
                    *e = *e1 - *e2;
                }

                let ratio = g.ring().div(r.max_coeff(), g.max_coeff());
                r = r - g.clone().mul_exp(&monom).mul_coeff(ratio);

                if r.is_zero() {
                    break 'term;
                }
            }

            // strip leading monomial that is not reducible
            rest_exponents.extend_from_slice(r.exponents(r.nterms() - 1));
            rest_coeff.push(r.coefficients.pop().unwrap());
        }

        // append in sorted order
        while let Some(c) = rest_coeff.pop() {
            let l = rest_coeff.len();
            q.append_monomial(c, &rest_exponents[l * self.nvars()..(l + 1) * self.nvars()]);
        }

        q
    }
}

impl<R: Field, E: Exponent, O: MonomialOrder> GroebnerBasis<R, E, O> {
    /// Add a new polynomial to the basis, updating and filtering the existing
    /// basis and critical pairs, based on Gebauer and Moeller's redundant pair criteria.
    ///
    /// Adapted from "A Computational Approach to Commutative Algebra" by Thomas Becker Volker Weispfenning.
    fn update(
        basis: &mut Vec<(usize, Rc<MultivariatePolynomial<R, E, O>>)>,
        critical_pairs: &mut Vec<CriticalPair<R, E, O>>,
        f: Rc<MultivariatePolynomial<R, E, O>>,
        index: usize,
    ) {
        let mut new_pairs: Vec<_> = basis
            .iter()
            .map(|b| (CriticalPair::new(b.1.clone(), f.clone(), b.0, index), true))
            .collect();

        for i in 0..new_pairs.len() {
            new_pairs[i].1 = false;
            new_pairs[i].1 = new_pairs[i].0.disjoint
                || new_pairs.iter().all(|p2| {
                    !p2.1
                        || new_pairs[i]
                            .0
                            .lcm
                            .iter()
                            .zip(&p2.0.lcm)
                            .any(|(e1, e2)| *e1 < *e2)
                });
        }

        new_pairs.retain(|p| p.1 && !p.0.disjoint);

        critical_pairs.retain(|p| {
            p.lcm.iter().zip(f.max_exp()).any(|(e1, e2)| *e1 < *e2)
                || p.poly_first
                    .max_exp()
                    .iter()
                    .zip(f.max_exp())
                    .zip(&p.lcm)
                    .all(|((e1, e2), ecm)| e1.max(e2) == ecm)
                || p.poly_sec
                    .max_exp()
                    .iter()
                    .zip(f.max_exp())
                    .zip(&p.lcm)
                    .all(|((e1, e2), ecm)| e1.max(e2) == ecm)
        });

        critical_pairs.extend(new_pairs.into_iter().map(|np| np.0));

        basis.retain(|b| {
            b.1.max_exp()
                .iter()
                .zip(f.max_exp())
                .any(|(e1, e2)| *e1 < *e2)
        });

        basis.push((index, f));
    }

    pub fn reduce_basis(mut self) -> Self {
        // filter lead-reducible polynomials
        let mut res = vec![true; self.system.len()];
        'l1: for (i, p1) in self.system.iter().enumerate() {
            for (j, p2) in self.system.iter().enumerate() {
                if i != j
                    && res[j]
                    && p1
                        .max_exp()
                        .iter()
                        .zip(p2.max_exp())
                        .all(|(h1, h2)| *h1 >= *h2)
                {
                    res[i] = false;
                    continue 'l1;
                }
            }
        }

        let mut lead_reduced = vec![];
        for (i, p) in self.system.drain(..).enumerate() {
            if res[i] {
                lead_reduced.push(p);
            }
        }

        let mut basis = vec![];
        for i in 0..lead_reduced.len() {
            lead_reduced.swap(0, i);
            let h = lead_reduced[0].reduce(&lead_reduced[1..]);
            if !h.is_zero() {
                let i = h.ring().inv(h.max_coeff());
                basis.push(h.mul_coeff(i));
            }
        }

        basis.sort_by(|p1, p2| p2.max_exp().cmp(p1.max_exp()));

        GroebnerBasis {
            system: basis,
            print_stats: self.print_stats,
        }
    }

    pub fn is_groebner_basis(system: &[MultivariatePolynomial<R, E, O>]) -> bool {
        for (i, p1) in system.iter().enumerate() {
            for p2 in &system[i + 1..] {
                let lcm: Vec<E> = p1
                    .max_exp()
                    .iter()
                    .zip(p2.max_exp())
                    .map(|(e1, e2)| *e1.max(e2))
                    .collect();

                // construct s-polynomial
                let extra_factor_f1: Vec<E> = lcm
                    .iter()
                    .zip(p1.max_exp())
                    .map(|(e1, e2)| *e1 - *e2)
                    .collect();

                let extra_factor_f2: Vec<E> = lcm
                    .iter()
                    .zip(p2.max_exp())
                    .map(|(e1, e2)| *e1 - *e2)
                    .collect();
                let new_f1 = p1
                    .clone()
                    .mul_exp(&extra_factor_f1)
                    .mul_coeff(p1.ring().div(p2.max_coeff(), p1.max_coeff()));
                let new_f2 = p2
                    .clone()
                    .mul_exp(&extra_factor_f2)
                    .mul_coeff(p1.ring().div(p1.max_coeff(), p2.max_coeff()));

                let s = new_f1 - new_f2;

                if !s.reduce(system).is_zero() {
                    return false;
                }
            }
        }
        true
    }
}

impl<R: Field, E: PositiveExponent, O: MonomialOrder> GroebnerBasis<R, E, O> {
    fn fglm_linear_combination(
        field: &R,
        columns: &[Vec<R::Element>],
        target: &[R::Element],
    ) -> Result<Option<Vec<R::Element>>, String> {
        if columns.is_empty() {
            return Ok(target
                .iter()
                .all(|entry| field.is_zero(entry))
                .then(Vec::new));
        }

        let rows = target.len();
        let mut data = Vec::with_capacity(rows * columns.len());
        for row in 0..rows {
            for column in columns {
                data.push(column[row].clone());
            }
        }

        let matrix = Matrix::from_linear(data, rows as u32, columns.len() as u32, field.clone())?;
        let right_hand_side = Matrix::new_vec(target.to_vec(), field.clone());
        match matrix.solve(&right_hand_side) {
            Ok(solution) => Ok(Some(solution.into_vec())),
            Err(MatrixError::Inconsistent) => Ok(None),
            Err(MatrixError::Underdetermined { .. }) => {
                Err("FGLM encountered dependent quotient-basis vectors".to_string())
            }
            Err(error) => Err(format!("FGLM linear-algebra failure: {error}")),
        }
    }

    /// Change the monomial order of this zero-dimensional Gröbner basis using
    /// the FGLM algorithm.
    ///
    /// The input must be a Gröbner basis over a field. An error is returned
    /// when its leading ideal is not zero-dimensional.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::prelude::*;
    ///
    /// let ideal = ["x*y-1", "y^2-x"]
    ///     .iter()
    ///     .map(|polynomial| {
    ///         parse!(polynomial)
    ///             .to_polynomial::<_, u16>(&Q, None)
    ///             .reorder::<GrevLexOrder>()
    ///     })
    ///     .collect::<Vec<_>>();
    /// let grevlex = GroebnerBasis::new(&ideal, false);
    /// let lex = grevlex.change_order::<LexOrder>().unwrap();
    ///
    /// assert!(GroebnerBasis::is_groebner_basis(&lex.system));
    /// ```
    pub fn change_order<O2: MonomialOrder>(&self) -> Result<GroebnerBasis<R, E, O2>, String> {
        if self.system.is_empty() {
            return Err("FGLM requires a non-empty zero-dimensional basis".to_string());
        }
        if self
            .system
            .iter()
            .any(|polynomial| polynomial.variables() != self.system[0].variables())
        {
            return Err("FGLM requires a unified variable map".to_string());
        }

        let field = self.system[0].ring().clone();
        let variables = self.system[0].variables().clone();
        let nvars = variables.len();

        if self
            .system
            .iter()
            .any(|polynomial| !polynomial.is_zero() && polynomial.is_constant())
        {
            let mut one = MultivariatePolynomial::<R, E, O2>::new(&field, Some(1), variables);
            one.append_monomial(field.one(), &vec![E::zero(); nvars]);
            return Ok(GroebnerBasis {
                system: vec![one],
                print_stats: self.print_stats,
            });
        }
        if nvars == 0 {
            return Err("FGLM requires at least one polynomial variable".to_string());
        }

        let leading_monomials = self
            .system
            .iter()
            .filter(|polynomial| !polynomial.is_zero())
            .map(|polynomial| polynomial.max_exp().to_vec())
            .collect::<Vec<_>>();

        let mut pure_power_bounds = vec![None; nvars];
        for leading in &leading_monomials {
            for variable in 0..nvars {
                if leading[variable].is_zero()
                    || leading
                        .iter()
                        .enumerate()
                        .any(|(index, exponent)| index != variable && !exponent.is_zero())
                {
                    continue;
                }

                let bound = &mut pure_power_bounds[variable];
                if bound
                    .as_ref()
                    .is_none_or(|current| leading[variable] < *current)
                {
                    *bound = Some(leading[variable]);
                }
            }
        }
        let pure_power_bounds = pure_power_bounds
            .into_iter()
            .enumerate()
            .map(|(variable, bound)| {
                bound.ok_or_else(|| {
                    format!(
                        "The leading ideal is not zero-dimensional: no pure power of {}",
                        variables[variable]
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let is_standard = |monomial: &[E]| {
            !leading_monomials.iter().any(|leading| {
                monomial
                    .iter()
                    .zip(leading)
                    .all(|(exponent, divisor)| exponent >= divisor)
            })
        };

        let zero_monomial = vec![E::zero(); nvars];
        let mut pending_source = BTreeSet::from([zero_monomial.clone()]);
        let mut seen_source = HashSet::default();
        seen_source.insert(zero_monomial);
        let mut source_standard_monomials = Vec::new();

        while let Some(monomial) = pending_source.pop_first() {
            if !is_standard(&monomial) {
                continue;
            }
            source_standard_monomials.push(monomial.clone());

            for variable in 0..nvars {
                let Some(next_exponent) = monomial[variable].checked_add(&E::one()) else {
                    return Err(
                        "Exponent overflow while constructing the FGLM staircase".to_string()
                    );
                };
                if next_exponent >= pure_power_bounds[variable] {
                    continue;
                }

                let mut child = monomial.clone();
                child[variable] = next_exponent;
                if seen_source.insert(child.clone()) {
                    pending_source.insert(child);
                }
            }
        }

        let quotient_dimension = source_standard_monomials.len();
        if quotient_dimension == 0 {
            return Err("FGLM constructed an empty quotient basis".to_string());
        }
        let source_indices = source_standard_monomials
            .iter()
            .cloned()
            .enumerate()
            .map(|(index, monomial)| (monomial, index))
            .collect::<HashMap<_, _>>();

        let prototype = &self.system[0];
        let mut normal_form_cache: HashMap<Vec<E>, Vec<R::Element>> = HashMap::default();
        let mut normal_form_vector = |monomial: &[E]| -> Result<Vec<R::Element>, String> {
            if let Some(vector) = normal_form_cache.get(monomial) {
                return Ok(vector.clone());
            }

            let remainder = prototype
                .monomial(field.one(), monomial.to_vec())
                .reduce(&self.system);
            let mut vector = vec![field.zero(); quotient_dimension];
            for term in &remainder {
                let index = source_indices.get(term.exponents).ok_or_else(|| {
                    format!(
                        "Normal form contains a monomial outside the FGLM staircase: {:?}",
                        term.exponents
                    )
                })?;
                vector[*index] = term.coefficient.clone();
            }
            normal_form_cache.insert(monomial.to_vec(), vector.clone());
            Ok(vector)
        };

        let zero_monomial = vec![E::zero(); nvars];
        let mut pending_target =
            BTreeSet::from([OrderedMonomial::<E, O2>::new(zero_monomial.clone())]);
        let mut seen_target = HashSet::default();
        seen_target.insert(zero_monomial);
        let mut target_standard_monomials: Vec<Vec<E>> = Vec::with_capacity(quotient_dimension);
        let mut target_vectors: Vec<Vec<R::Element>> = Vec::with_capacity(quotient_dimension);
        let mut target_leading_monomials: Vec<Vec<E>> = Vec::new();
        let mut target_basis = Vec::new();

        while let Some(ordered_monomial) = pending_target.pop_first() {
            let monomial = ordered_monomial.exponents;
            if target_leading_monomials.iter().any(|leading| {
                monomial
                    .iter()
                    .zip(leading)
                    .all(|(exponent, divisor)| exponent >= divisor)
            }) {
                continue;
            }

            let vector = normal_form_vector(&monomial)?;
            if let Some(coefficients) =
                Self::fglm_linear_combination(&field, &target_vectors, &vector)?
            {
                let mut relation = MultivariatePolynomial::<R, E, O2>::new(
                    &field,
                    Some(coefficients.len() + 1),
                    variables.clone(),
                );
                relation.append_monomial(field.one(), &monomial);
                for (coefficient, standard) in
                    coefficients.into_iter().zip(&target_standard_monomials)
                {
                    relation.append_monomial(field.neg(&coefficient), standard);
                }
                target_leading_monomials.push(monomial);
                target_basis.push(relation);
                continue;
            }

            if target_standard_monomials.len() == quotient_dimension {
                return Err(
                    "FGLM found too many linearly independent quotient elements".to_string()
                );
            }

            target_standard_monomials.push(monomial.clone());
            target_vectors.push(vector);
            for variable in 0..nvars {
                let Some(next_exponent) = monomial[variable].checked_add(&E::one()) else {
                    return Err(
                        "Exponent overflow while constructing the lex staircase".to_string()
                    );
                };
                let mut child = monomial.clone();
                child[variable] = next_exponent;
                if seen_target.insert(child.clone()) {
                    pending_target.insert(OrderedMonomial::new(child));
                }
            }
        }

        if target_standard_monomials.len() != quotient_dimension {
            return Err(format!(
                "FGLM found {} lex standard monomials, expected {quotient_dimension}",
                target_standard_monomials.len()
            ));
        }

        Ok(GroebnerBasis {
            system: target_basis,
            print_stats: self.print_stats,
        }
        .reduce_basis())
    }
}

impl<E: PositiveExponent> GroebnerBasis<RationalField, E, LexOrder> {
    fn rational_univariate(
        polynomial: &MultivariatePolynomial<RationalField, E, LexOrder>,
        variable: usize,
    ) -> Result<MultivariatePolynomial<RationalField, u16>, String> {
        let target = PolyVariable::Temporary(0);
        let mut result =
            MultivariatePolynomial::new(&Q, Some(polynomial.nterms()), Arc::new(vec![target]));
        for term in polynomial {
            for (index, exponent) in term.exponents.iter().enumerate() {
                if index != variable && !exponent.is_zero() {
                    return Err("Expected a univariate polynomial".to_string());
                }
            }
            let exponent = u16::try_from(term.exponents[variable].to_u32())
                .map_err(|_| "Root exponent does not fit in u16".to_string())?;
            result.append_monomial(term.coefficient.clone(), &[exponent]);
        }
        Ok(result)
    }

    fn specialize_polynomial(
        polynomial: &MultivariatePolynomial<RationalField, E, LexOrder>,
        target: usize,
        variables: &[PolyVariable],
        solution: &PolynomialSolution<RationalField>,
    ) -> Result<Option<MultivariatePolynomial<AlgebraicExtension<Q>, u16>>, String> {
        let mut coefficients: HashMap<u16, AlgebraicNumber<Q>> = HashMap::default();
        for term in polynomial {
            let mut coefficient = solution.field.constant(term.coefficient.clone());
            for (index, exponent) in term.exponents.iter().enumerate() {
                if exponent.is_zero() || index == target {
                    continue;
                }
                let Some(value) = solution.values.get(&variables[index]) else {
                    return Ok(None);
                };
                coefficient = solution.field.mul(
                    &coefficient,
                    &solution.field.pow(value, u64::from(exponent.to_u32())),
                );
            }
            let exponent = u16::try_from(term.exponents[target].to_u32())
                .map_err(|_| "Specialized exponent does not fit in u16".to_string())?;
            coefficients
                .entry(exponent)
                .and_modify(|current| solution.field.add_assign(current, &coefficient))
                .or_insert(coefficient);
        }

        let variable = solution.field.get_new_var();
        let mut result = MultivariatePolynomial::new(
            &solution.field,
            Some(coefficients.len()),
            Arc::new(vec![variable]),
        );
        let mut coefficients = coefficients.into_iter().collect::<Vec<_>>();
        coefficients.sort_by_key(|(exponent, _)| *exponent);
        for (exponent, coefficient) in coefficients {
            if !solution.field.is_zero(&coefficient) {
                result.append_monomial(coefficient, &[exponent]);
            }
        }
        Ok(Some(result))
    }

    fn transport_element(
        element: &AlgebraicNumber<Q>,
        field: &AlgebraicExtension<Q>,
        old_generator: &AlgebraicNumber<Q>,
    ) -> AlgebraicNumber<Q> {
        let mut coefficients = vec![Q.zero(); element.poly().degree(0) as usize + 1];
        for term in element.poly() {
            coefficients[term.exponents[0] as usize] = term.coefficient.clone();
        }

        let mut result = field.zero();
        for coefficient in coefficients.into_iter().rev() {
            result = field.add(
                &field.mul(&result, old_generator),
                &field.constant(coefficient),
            );
        }
        result
    }

    fn with_adjoined_root(
        mut solution: PolynomialSolution<RationalField>,
        polynomial: MultivariatePolynomial<AlgebraicExtension<Q>, u16>,
        embedding: usize,
        target: PolyVariable,
    ) -> PolynomialSolution<RationalField> {
        let extension = AlgebraicExtension::new_with_embedding(polynomial, embedding);
        let new_variable = solution.field.get_new_var();
        let (field, old_generator, new_generator, new_generator_field) = solution
            .field
            .adjoin_with_embedding_and_generator_field(&extension, Some(new_variable));
        let atom = new_generator_field.element_to_atom(&new_generator_field.generator());
        for value in solution.values.values_mut() {
            *value = Self::transport_element(value, &field, &old_generator);
        }
        solution.field = field;
        solution.atom_values.insert(target.clone(), atom);
        solution.values.insert(target, new_generator);
        solution
    }

    /// Solve a zero-dimensional lexicographic Gröbner basis inside the
    /// polynomial domain.
    ///
    /// Every returned branch stores its values in one common algebraic
    /// extension. Conversion to expression atoms is deliberately left to
    /// [`PolynomialSolution::to_atom_map`].
    ///
    /// ```
    /// use symbolica::prelude::*;
    ///
    /// let ideal: Vec<MultivariatePolynomial<_, u16>> = ["x+y", "y^2-2"]
    ///     .iter()
    ///     .map(|polynomial| parse!(polynomial).to_polynomial(&Q, None))
    ///     .collect();
    /// let solutions = GroebnerBasis::new(&ideal, false).solve().unwrap();
    ///
    /// assert_eq!(solutions.len(), 2);
    /// assert!(solutions.iter().all(|solution| solution.len() == 2));
    /// assert!(solutions
    ///     .iter()
    ///     .all(|solution| solution.field().poly().degree(0) == 2));
    /// ```
    pub fn solve(&self) -> Result<Vec<PolynomialSolution<RationalField>>, String> {
        self.solve_in_base_field(false)
    }

    /// Solve the basis, optionally retaining only roots arising from linear
    /// factors over the rational base field.
    pub(crate) fn solve_in_base_field(
        &self,
        rational_roots_only: bool,
    ) -> Result<Vec<PolynomialSolution<RationalField>>, String> {
        if self.system.is_empty() {
            return Err(
                "Cannot enumerate the solutions of an empty, positive-dimensional basis"
                    .to_string(),
            );
        }

        if self
            .system
            .iter()
            .any(|polynomial| !polynomial.is_zero() && polynomial.is_constant())
        {
            return Ok(Vec::new());
        }

        let variables = self.system[0].variables().clone();
        if variables.is_empty() {
            return Err("The Gröbner basis has no variables".to_string());
        }
        if self
            .system
            .iter()
            .any(|polynomial| polynomial.variables() != &variables)
        {
            return Err("The Gröbner basis does not have a unified variable map".to_string());
        }

        let last = variables.len() - 1;
        let elimination_polynomial = self
            .system
            .iter()
            .filter(|polynomial| {
                polynomial.degree(last) > E::zero()
                    && (0..last).all(|index| polynomial.degree(index) == E::zero())
            })
            .min_by_key(|polynomial| polynomial.degree(last).to_u32())
            .ok_or_else(|| {
                format!(
                    "The lexicographic basis has no univariate elimination polynomial for {}",
                    variables[last]
                )
            })?;

        let mut branches = Vec::new();
        for (factor, _) in elimination_polynomial.factor() {
            let degree = factor.degree(last).to_u32() as usize;
            if degree == 0 {
                continue;
            }

            if degree == 1 {
                let field = AlgebraicExtension::trivial(Q);
                let root = Q.neg(&Q.div(&factor.get_constant(), &factor.lcoeff()));
                let mut values = HashMap::default();
                values.insert(variables[last].clone(), field.constant(root.clone()));
                let mut atom_values = HashMap::default();
                atom_values.insert(variables[last].clone(), Atom::num(root));
                branches.push(PolynomialSolution {
                    field,
                    values,
                    atom_values,
                });
                continue;
            }

            if rational_roots_only {
                continue;
            }

            let factor = Self::rational_univariate(&factor, last)?.make_monic();
            for embedding in 0..degree {
                let field = AlgebraicExtension::new_with_embedding(factor.clone(), embedding);
                let root = field.generator();
                let mut values = HashMap::default();
                values.insert(variables[last].clone(), root);
                let mut atom_values = HashMap::default();
                atom_values.insert(
                    variables[last].clone(),
                    field.element_to_atom(values.get(&variables[last]).unwrap()),
                );
                branches.push(PolynomialSolution {
                    field,
                    values,
                    atom_values,
                });
            }
        }

        for target in (0..last).rev() {
            let candidates = self
                .system
                .iter()
                .filter(|polynomial| {
                    polynomial.degree(target) > E::zero()
                        && (0..target).all(|index| polynomial.degree(index) == E::zero())
                })
                .collect::<Vec<_>>();

            if candidates.is_empty() {
                return Err(format!(
                    "The lexicographic basis has no triangular equation for {}",
                    variables[target]
                ));
            }

            let mut next_branches = Vec::new();
            for branch in branches {
                let mut branch_children = None;
                for polynomial in &candidates {
                    let Some(specialized) =
                        Self::specialize_polynomial(polynomial, target, &variables, &branch)?
                    else {
                        continue;
                    };
                    if specialized.is_zero() {
                        continue;
                    }
                    if specialized.is_constant() {
                        branch_children = Some(Vec::new());
                        break;
                    }

                    let mut children = Vec::new();
                    for (factor, _) in specialized.factor() {
                        let degree = factor.degree(0) as usize;
                        if degree == 0 {
                            continue;
                        }
                        if degree == 1 {
                            let root = branch
                                .field
                                .neg(&branch.field.div(&factor.get_constant(), &factor.lcoeff()));
                            let atom = branch.field.element_to_atom_simplified(&root);
                            let mut child = branch.clone();
                            child.atom_values.insert(variables[target].clone(), atom);
                            child.values.insert(variables[target].clone(), root);
                            children.push(child);
                            continue;
                        }

                        if rational_roots_only {
                            continue;
                        }

                        for embedding in 0..degree {
                            children.push(Self::with_adjoined_root(
                                branch.clone(),
                                factor.clone(),
                                embedding,
                                variables[target].clone(),
                            ));
                        }
                    }
                    branch_children = Some(children);
                    break;
                }

                let children = branch_children.ok_or_else(|| {
                    format!(
                        "Could not find a nonzero triangular equation for {} on this branch",
                        variables[target]
                    )
                })?;
                next_branches.extend(children);
            }
            branches = next_branches;
        }

        Ok(branches)
    }
}

#[derive(Clone)]
struct ParametricSolveBranch<E: PositiveExponent> {
    extension: Arc<ParametricExtension<E>>,
    conjugates: Vec<usize>,
    values: HashMap<PolyVariable, AlgebraicNumber<ParameterField<E>>>,
}

impl<E: PositiveExponent> GroebnerBasis<RationalPolynomialField<IntegerRing, E>, E, LexOrder> {
    fn parametric_univariate_factors<X: PositiveExponent>(
        polynomial: &MultivariatePolynomial<ParameterField<E>, X, LexOrder>,
        variable: usize,
    ) -> Result<
        Vec<MultivariatePolynomial<RationalPolynomialField<IntegerRing, E>, u16, LexOrder>>,
        String,
    > {
        if (0..polynomial.nvars())
            .any(|index| index != variable && polynomial.degree(index) != X::zero())
        {
            return Err("Expected a univariate polynomial over Q(parameters)".to_string());
        }

        let root_variable = polynomial.get_vars_ref()[variable].clone();
        let rational_polynomial =
            RationalPolynomial::from_univariate(polynomial.to_univariate_from_univariate(variable));
        let mut result = Vec::new();

        for (factor, _) in rational_polynomial.numerator.factor() {
            let Some(root_position) = factor
                .get_vars_ref()
                .iter()
                .position(|candidate| candidate == &root_variable)
            else {
                continue;
            };
            if factor.degree(root_position) == E::zero() {
                continue;
            }

            let factor: RationalPolynomial<IntegerRing, E> = factor.into();
            let factor = factor
                .to_polynomial(std::slice::from_ref(&root_variable), false)
                .map_err(str::to_owned)?;
            let mut converted = MultivariatePolynomial::new(
                factor.ring(),
                Some(factor.nterms()),
                Arc::new(vec![root_variable.clone()]),
            );
            for term in &factor {
                let exponent = u16::try_from(term.exponents[0].to_u32())
                    .map_err(|_| "Parametric root exponent does not fit in u16".to_string())?;
                converted.append_monomial(term.coefficient.clone(), &[exponent]);
            }
            result.push(converted.make_monic());
        }

        Ok(result)
    }

    fn quotient_square_free_factors(
        polynomial: &MultivariatePolynomial<AlgebraicQuotient<ParameterField<E>>, u16>,
    ) -> Vec<(
        MultivariatePolynomial<AlgebraicQuotient<ParameterField<E>>, u16>,
        usize,
    )> {
        if polynomial.is_zero() || polynomial.is_constant() {
            return Vec::new();
        }

        let mut repeated = polynomial
            .univariate_gcd(&polynomial.derivative(0))
            .make_monic();
        let mut current = (polynomial.clone().make_monic() / &repeated).make_monic();
        let mut multiplicity = 1;
        let mut factors = Vec::new();
        while !current.is_one() {
            let common = current.univariate_gcd(&repeated).make_monic();
            let factor = (current / &common).make_monic();
            if !factor.is_one() {
                factors.push((factor, multiplicity));
            }
            current = common;
            repeated = (repeated / &current).make_monic();
            multiplicity += 1;
        }
        factors
    }

    /// Factor a univariate polynomial over an existing formal extension of
    /// `Q(parameters)` using Trager's norm algorithm.
    fn parametric_quotient_factors(
        polynomial: &MultivariatePolynomial<AlgebraicQuotient<ParameterField<E>>, u16>,
    ) -> Result<Vec<MultivariatePolynomial<AlgebraicQuotient<ParameterField<E>>, u16>>, String>
    {
        if polynomial.nvars() != 1 {
            return Err("Expected a univariate polynomial over a parametric quotient".into());
        }

        let quotient = polynomial.ring().clone();
        if quotient.poly().degree(0) == 1 {
            let base_field = quotient.poly().ring().clone();
            let mut base_polynomial = MultivariatePolynomial::new(
                &base_field,
                Some(polynomial.nterms()),
                polynomial.variables().clone(),
            );
            for term in polynomial {
                if !term.coefficient.poly().is_constant() {
                    return Err(
                        "A degree-one parametric quotient contained its formal generator"
                            .to_string(),
                    );
                }
                base_polynomial
                    .append_monomial(term.coefficient.poly().get_constant(), term.exponents);
            }

            return Ok(Self::parametric_univariate_factors(&base_polynomial, 0)?
                .into_iter()
                .map(|factor| {
                    factor.map_coeff(
                        |coefficient| quotient.constant(coefficient.clone()),
                        quotient.clone(),
                    )
                })
                .collect());
        }

        let extension = quotient.as_extension();
        let mut result = Vec::new();
        for (square_free, _) in Self::quotient_square_free_factors(polynomial) {
            let square_free =
                square_free.map_coeff(|coefficient| coefficient.clone(), extension.clone());
            let (variable, shift, shifted, norm) = square_free.norm_impl();
            let norm_factors = Self::parametric_univariate_factors(&norm, variable)?;

            if norm_factors.len() == 1 {
                result.push(
                    square_free.map_coeff(|coefficient| coefficient.clone(), quotient.clone()),
                );
                continue;
            }

            let mut remaining = shifted.to_number_field(&extension);
            let alpha_polynomial = shifted
                .variable(&square_free.get_vars_ref()[variable])
                .unwrap()
                + shifted
                    .variable(&extension.poly().get_vars_ref()[0])
                    .unwrap()
                    * &shifted.constant(shifted.ring().nth((shift as u64).into()));

            for norm_factor in norm_factors {
                let mut norm_factor = norm_factor.to_number_field(&extension);
                norm_factor.unify_variables(&mut remaining);
                let gcd = norm_factor.univariate_gcd(&remaining).make_monic();
                if gcd.is_one() {
                    continue;
                }
                remaining = (remaining / &gcd).make_monic();

                let mut factor = MultivariatePolynomial::from_number_field(&gcd);
                let mut alpha_polynomial = alpha_polynomial.clone();
                factor.unify_variables(&mut alpha_polynomial);
                let factor_variable = factor
                    .get_vars_ref()
                    .iter()
                    .position(|candidate| candidate == &square_free.get_vars_ref()[variable])
                    .ok_or_else(|| "The shifted factor lost its polynomial variable".to_string())?;
                let factor = factor
                    .replace_with_poly(factor_variable, &alpha_polynomial)
                    .to_number_field(&extension)
                    .make_monic()
                    .map_coeff(|coefficient| coefficient.clone(), quotient.clone());
                result.push(factor);
            }
        }
        Ok(result)
    }

    fn transport_parametric_element(
        element: &AlgebraicNumber<ParameterField<E>>,
        field: &AlgebraicQuotient<ParameterField<E>>,
        old_generator: &AlgebraicNumber<ParameterField<E>>,
    ) -> AlgebraicNumber<ParameterField<E>> {
        let mut coefficients =
            vec![element.poly().ring().zero(); element.poly().degree(0) as usize + 1];
        for term in element.poly() {
            coefficients[term.exponents[0] as usize] = term.coefficient.clone();
        }

        let mut result = field.zero();
        for coefficient in coefficients.into_iter().rev() {
            result = field.add(
                &field.mul(&result, old_generator),
                &field.constant(coefficient),
            );
        }
        result
    }

    fn specialize_parametric_polynomial(
        polynomial: &MultivariatePolynomial<RationalPolynomialField<IntegerRing, E>, E, LexOrder>,
        target: usize,
        variables: &[PolyVariable],
        solution: &ParametricSolveBranch<E>,
    ) -> Result<
        Option<
            MultivariatePolynomial<AlgebraicQuotient<RationalPolynomialField<IntegerRing, E>>, u16>,
        >,
        String,
    > {
        let field = solution.extension.quotient();
        let mut coefficients = HashMap::default();
        for term in polynomial {
            let mut coefficient = field.constant(term.coefficient.clone());
            for (index, exponent) in term.exponents.iter().enumerate() {
                if exponent.is_zero() || index == target {
                    continue;
                }
                let Some(value) = solution.values.get(&variables[index]) else {
                    return Ok(None);
                };
                coefficient = field.mul(
                    &coefficient,
                    &field.pow(value, u64::from(exponent.to_u32())),
                );
            }

            let exponent = u16::try_from(term.exponents[target].to_u32())
                .map_err(|_| "Specialized exponent does not fit in u16".to_string())?;
            coefficients
                .entry(exponent)
                .and_modify(|current| field.add_assign(current, &coefficient))
                .or_insert(coefficient);
        }

        let variable = field.get_new_var();
        let mut result =
            MultivariatePolynomial::new(field, Some(coefficients.len()), Arc::new(vec![variable]));
        let mut coefficients = coefficients.into_iter().collect::<Vec<_>>();
        coefficients.sort_by_key(|(exponent, _)| *exponent);
        for (exponent, coefficient) in coefficients {
            if !field.is_zero(&coefficient) {
                result.append_monomial(coefficient, &[exponent]);
            }
        }
        Ok(Some(result))
    }

    fn parametric_extension(
        quotient: AlgebraicQuotient<ParameterField<E>>,
        system: &[MultivariatePolynomial<ParameterField<E>, E, LexOrder>],
        elimination_polynomial: &MultivariatePolynomial<ParameterField<E>, E, LexOrder>,
        elimination_variable: usize,
    ) -> ParametricExtension<E> {
        fn compact_condition<E: PositiveExponent>(
            polynomial: &MultivariatePolynomial<IntegerRing, E>,
            solve_variables: &HashSet<PolyVariable>,
        ) -> MultivariatePolynomial<IntegerRing, E> {
            let active = (0..polynomial.nvars())
                .filter(|index| {
                    polynomial.degree(*index) != E::zero()
                        && !solve_variables.contains(&polynomial.get_vars_ref()[*index])
                })
                .collect::<Vec<_>>();
            let variables = Arc::new(
                active
                    .iter()
                    .map(|index| polynomial.get_vars_ref()[*index].clone())
                    .collect(),
            );
            let mut result = MultivariatePolynomial::new(
                polynomial.ring(),
                Some(polynomial.nterms()),
                variables,
            );
            for term in polynomial {
                let exponents = active
                    .iter()
                    .map(|index| term.exponents[*index])
                    .collect::<Vec<_>>();
                result.append_monomial(term.coefficient.clone(), &exponents);
            }
            result
        }

        fn collect_coefficient<E: PositiveExponent>(
            coefficient: &RationalPolynomial<IntegerRing, E>,
            solve_variables: &HashSet<PolyVariable>,
            parameters: &mut HashSet<PolyVariable>,
            conditions: &mut HashSet<MultivariatePolynomial<IntegerRing, E>>,
        ) {
            for (index, variable) in coefficient.get_variables().iter().enumerate() {
                if !solve_variables.contains(variable)
                    && (coefficient.numerator.degree(index) != E::zero()
                        || coefficient.denominator.degree(index) != E::zero())
                {
                    parameters.insert(variable.clone());
                }
            }
            if !coefficient.denominator.is_one() {
                let condition = compact_condition(&coefficient.denominator, solve_variables);
                if !condition.is_constant() {
                    conditions.insert(condition);
                }
            }
        }

        let solve_variables: HashSet<PolyVariable> = system
            .first()
            .map(|polynomial| polynomial.variables().iter().cloned().collect())
            .unwrap_or_default();
        let mut parameters = HashSet::default();
        let mut conditions = HashSet::default();

        for polynomial in system {
            for coefficient in &polynomial.coefficients {
                collect_coefficient(
                    coefficient,
                    &solve_variables,
                    &mut parameters,
                    &mut conditions,
                );
            }
        }
        for coefficient in &quotient.poly().coefficients {
            collect_coefficient(
                coefficient,
                &solve_variables,
                &mut parameters,
                &mut conditions,
            );
        }

        let defining_polynomial = quotient.poly().to_univariate_from_univariate(0);
        let derivative = defining_polynomial.derivative();
        if !derivative.is_zero() {
            let discriminant = defining_polynomial.resultant_prs(&derivative);
            collect_coefficient(
                &discriminant,
                &solve_variables,
                &mut parameters,
                &mut conditions,
            );
            if !discriminant.numerator.is_constant() {
                conditions.insert(compact_condition(&discriminant.numerator, &solve_variables));
            }
        }

        let elimination_polynomial =
            elimination_polynomial.to_univariate_from_univariate(elimination_variable);
        let derivative = elimination_polynomial.derivative();
        if !derivative.is_zero() {
            let discriminant = elimination_polynomial.resultant_prs(&derivative);
            collect_coefficient(
                &discriminant,
                &solve_variables,
                &mut parameters,
                &mut conditions,
            );
            if !discriminant.numerator.is_constant() {
                conditions.insert(compact_condition(&discriminant.numerator, &solve_variables));
            }
        }

        let mut parameters = parameters.into_iter().collect::<Vec<_>>();
        parameters.sort();
        let mut generic_conditions = conditions.into_iter().collect::<Vec<_>>();
        generic_conditions.sort_by(|a, b| a.internal_cmp(b));

        ParametricExtension {
            quotient,
            parameters: Arc::new(parameters),
            generic_conditions,
            construction: ParametricExtensionConstruction::Primitive,
        }
    }

    /// Solve a zero-dimensional lexicographic Gröbner basis over
    /// `Q(parameters)`.
    ///
    /// Parametric root indices form a path through the adjoining tower. They
    /// acquire the usual complex-root ordering only after the parameters are
    /// specialized.
    /// The result describes the generic parameter locus: specializations at
    /// which a denominator vanishes, the degree drops, or the ideal ceases to
    /// be zero-dimensional must be considered separately.
    ///
    /// Each [`ParametricSolution`] stores one shared formal quotient field.
    /// Call [`ParametricSolution::specialize`] to check its generic conditions,
    /// select a cached analytic embedding, and obtain a [`PolynomialSolution`]
    /// over `Q`.
    pub fn solve_parametric(&self) -> Result<Vec<ParametricSolution<E>>, String> {
        self.solve_parametric_in_base_field(false)
    }

    /// Solve the basis, optionally retaining only roots arising from linear
    /// factors over `Q(parameters)` and its current rational quotient field.
    pub(crate) fn solve_parametric_in_base_field(
        &self,
        rational_roots_only: bool,
    ) -> Result<Vec<ParametricSolution<E>>, String> {
        if self.system.is_empty() {
            return Err(
                "Cannot enumerate the solutions of an empty, positive-dimensional basis"
                    .to_string(),
            );
        }
        if self
            .system
            .iter()
            .any(|polynomial| !polynomial.is_zero() && polynomial.is_constant())
        {
            return Ok(Vec::new());
        }

        let variables = self.system[0].variables().clone();
        if variables.is_empty() {
            return Err("The Gröbner basis has no variables".to_string());
        }
        if self
            .system
            .iter()
            .any(|polynomial| polynomial.variables() != &variables)
        {
            return Err("The Gröbner basis does not have a unified variable map".to_string());
        }

        let last = variables.len() - 1;
        let elimination_polynomial = self
            .system
            .iter()
            .filter(|polynomial| {
                polynomial.degree(last) > E::zero()
                    && (0..last).all(|index| polynomial.degree(index) == E::zero())
            })
            .min_by_key(|polynomial| polynomial.degree(last).to_u32())
            .ok_or_else(|| {
                format!(
                    "The lexicographic basis has no univariate elimination polynomial for {}",
                    variables[last]
                )
            })?;

        let factors = Self::parametric_univariate_factors(elimination_polynomial, last)?;
        let mut branches = Vec::new();
        for factor in factors {
            let degree = factor.degree(0) as usize;
            if degree == 0 {
                continue;
            }
            if degree == 1 {
                let field = AlgebraicQuotient::trivial(factor.ring().clone());
                let root = field.constant(
                    factor
                        .ring()
                        .neg(&factor.ring().div(&factor.get_constant(), &factor.lcoeff())),
                );
                let extension = Arc::new(Self::parametric_extension(
                    field,
                    &self.system,
                    elimination_polynomial,
                    last,
                ));
                let mut values = HashMap::default();
                values.insert(variables[last].clone(), root);
                branches.push(ParametricSolveBranch {
                    extension,
                    conjugates: vec![0],
                    values,
                });
                continue;
            }

            if rational_roots_only {
                continue;
            }

            let field = AlgebraicQuotient::new(factor);
            let extension = Arc::new(Self::parametric_extension(
                field,
                &self.system,
                elimination_polynomial,
                last,
            ));
            for conjugate in 0..degree {
                let root = extension.quotient().generator();
                let mut values = HashMap::default();
                values.insert(variables[last].clone(), root);
                branches.push(ParametricSolveBranch {
                    extension: extension.clone(),
                    conjugates: vec![conjugate],
                    values,
                });
            }
        }

        for target in (0..last).rev() {
            let candidates = self
                .system
                .iter()
                .filter(|polynomial| {
                    polynomial.degree(target) > E::zero()
                        && (0..target).all(|index| polynomial.degree(index) == E::zero())
                })
                .collect::<Vec<_>>();
            if candidates.is_empty() {
                return Err(format!(
                    "The lexicographic basis has no triangular equation for {}",
                    variables[target]
                ));
            }

            let mut next_branches = Vec::new();
            for branch in branches {
                let mut child = None;
                for polynomial in &candidates {
                    let Some(specialized) = Self::specialize_parametric_polynomial(
                        polynomial, target, &variables, &branch,
                    )?
                    else {
                        continue;
                    };
                    if specialized.is_zero() {
                        continue;
                    }
                    if specialized.is_constant() {
                        child = Some(None);
                        break;
                    }

                    let factors = Self::parametric_quotient_factors(&specialized)?;
                    let mut children = Vec::new();
                    for factor in factors {
                        let degree = factor.degree(0) as usize;
                        if degree == 0 {
                            continue;
                        }
                        if degree == 1 {
                            let field = branch.extension.quotient();
                            let root =
                                field.neg(&field.div(&factor.get_constant(), &factor.lcoeff()));
                            let mut solved = branch.clone();
                            solved.values.insert(variables[target].clone(), root);
                            children.push(solved);
                            continue;
                        }

                        if rational_roots_only {
                            continue;
                        }

                        let old_extension = branch.extension.clone();
                        let new_variable = old_extension.quotient().get_new_var();
                        let (field, old_generator, new_generator, shift) = old_extension
                            .quotient()
                            .adjoin_formal_with_shift(&factor, Some(new_variable));
                        let extension = Arc::new(ParametricExtension {
                            quotient: field,
                            parameters: old_extension.parameters.clone(),
                            generic_conditions: old_extension.generic_conditions.clone(),
                            construction: ParametricExtensionConstruction::Adjoined {
                                base: old_extension,
                                polynomial: factor,
                                shift,
                            },
                        });

                        for conjugate in 0..degree {
                            let mut solved = branch.clone();
                            solved.extension = extension.clone();
                            solved.conjugates.push(conjugate);
                            for value in solved.values.values_mut() {
                                *value = Self::transport_parametric_element(
                                    value,
                                    extension.quotient(),
                                    &old_generator,
                                );
                            }
                            solved
                                .values
                                .insert(variables[target].clone(), new_generator.clone());
                            children.push(solved);
                        }
                    }
                    child = Some((!children.is_empty()).then_some(children));
                    break;
                }

                match child {
                    Some(Some(solutions)) => next_branches.extend(solutions),
                    Some(None) => {}
                    None => {
                        return Err(format!(
                            "Could not find a nonzero triangular equation for {} on this branch",
                            variables[target]
                        ));
                    }
                }
            }
            branches = next_branches;
        }

        Ok(branches
            .into_iter()
            .map(|solution| ParametricSolution {
                field: solution.extension,
                conjugates: Arc::new(solution.conjugates),
                values: solution.values,
            })
            .collect())
    }
}

/// Marker for fields that can be echelonized by the F4 implementation.
///
/// All `'static` fields use the default sparse row reduction. `Zp` is detected
/// at the call site and routed through a specialized 32-bit finite-field path.
pub trait Echelonize: Field + 'static {}

impl<R: Field + 'static> Echelonize for R {}

fn echelonize<R: Echelonize, E: Exponent, O: MonomialOrder>(
    matrix: &mut Vec<Vec<(R::Element, usize)>>,
    zp_matrix: &mut Vec<Vec<(i64, usize)>>,
    selected_polys: &mut Vec<Rc<MultivariatePolynomial<R, E, O>>>,
    all_monomials: &HashMap<Vec<E>, MonomialData>,
    sorted_monomial_indices: &[usize],
    field: &R,
    buffer: &mut Vec<R::Element>,
    zp_buffer: &mut Vec<i64>,
    pivots: &mut Vec<Option<usize>>,
    print_stats: bool,
) {
    if TypeId::of::<R>() == TypeId::of::<Zp>() {
        // SAFETY: the TypeId check above proves that `R` is exactly the `Zp`
        // type alias, so the associated element and polynomial layouts match.
        unsafe {
            echelonize_zp(
                &mut *(matrix as *mut Vec<Vec<(R::Element, usize)>>
                    as *mut Vec<Vec<(<Zp as Set>::Element, usize)>>),
                zp_matrix,
                &mut *(selected_polys as *mut Vec<Rc<MultivariatePolynomial<R, E, O>>>
                    as *mut Vec<Rc<MultivariatePolynomial<Zp, E, O>>>),
                all_monomials,
                sorted_monomial_indices,
                &*(field as *const R as *const Zp),
                zp_buffer,
                pivots,
                print_stats,
            );
        }
    } else {
        echelonize_default(
            matrix,
            selected_polys,
            all_monomials,
            sorted_monomial_indices,
            field,
            buffer,
            pivots,
            print_stats,
        );
    }
}

#[inline(never)]
fn echelonize_default<R: Field, E: Exponent, O: MonomialOrder>(
    matrix: &mut Vec<Vec<(R::Element, usize)>>,
    selected_polys: &mut Vec<Rc<MultivariatePolynomial<R, E, O>>>,
    all_monomials: &HashMap<Vec<E>, MonomialData>,
    sorted_monomial_indices: &[usize],
    field: &R,
    buffer: &mut Vec<R::Element>,
    pivots: &mut Vec<Option<usize>>,
    print_stats: bool,
) {
    matrix.resize(selected_polys.len(), vec![]);
    for (row, p) in matrix.iter_mut().zip(selected_polys) {
        row.clear();

        for (coeff, exp) in p.coefficients.iter().zip(p.exponents_iter()).rev() {
            row.push((coeff.clone(), all_monomials.get(exp).unwrap().column));
        }
    }

    // Sort the matrix rows to put the shortest and most reduced pivots on top.
    sort_matrix_rows(matrix);

    for p in &mut *pivots {
        *p = None;
    }

    buffer.resize(sorted_monomial_indices.len(), field.zero());
    pivots.resize(sorted_monomial_indices.len(), None);

    let mut pc = 0;
    for r in 0..matrix.len() {
        // identify all pivots
        if let Some((coeff, col)) = matrix[r].first_mut()
            && pivots[*col].is_none()
        {
            pivots[*col] = Some(r);
            pc += 1;

            if !field.is_one(coeff) {
                let inv_pivot = field.inv(coeff);

                for (coeff, _) in &mut matrix[r] {
                    field.mul_assign(coeff, &inv_pivot);
                }
            }
        }
    }

    if print_stats {
        println!("\tPivots={}, rows to reduce={}", pc, matrix.len() - pc);
    }

    for r in 0..matrix.len() {
        if matrix[r].is_empty() {
            continue;
        }

        // do not reduce pivots
        if pivots.contains(&Some(r)) {
            continue;
        }

        // copy row into the buffer
        for (coeff, col) in &*matrix[r] {
            buffer[*col] = coeff.clone();
        }

        for i in 0..buffer.len() {
            if field.is_zero(&buffer[i]) {
                continue;
            }

            let Some(pivot_index) = pivots[i] else {
                continue;
            };

            let pivot: &Vec<(R::Element, usize)> = &matrix[pivot_index];
            let c = buffer[i].clone();

            buffer[i] = field.zero();

            for (coeff, col) in pivot.iter().skip(1) {
                field.sub_mul_assign(&mut buffer[*col], coeff, &c);
            }
        }

        matrix[r].clear();

        for (col, coeff) in buffer.iter_mut().enumerate() {
            if !field.is_zero(coeff) {
                matrix[r].push((coeff.clone(), col));
                *coeff = field.zero();
            }
        }

        if let Some((coeff, col)) = matrix[r].first() {
            pivots[*col] = Some(r);
            let inv_pivot = field.inv(coeff);

            for (coeff, _) in &mut matrix[r] {
                field.mul_assign(coeff, &inv_pivot);
            }
        }
    }

    matrix.retain(|r| !r.is_empty());
}

/// Specialized 32-bit finite field echelonization based on
/// "A Compact Parallel Implementation of F4" by Monagan and Pearce.
fn echelonize_zp<E: Exponent, O: MonomialOrder>(
    matrix: &mut Vec<Vec<(<Zp as Set>::Element, usize)>>,
    integer_matrix: &mut Vec<Vec<(i64, usize)>>,
    selected_polys: &mut Vec<Rc<MultivariatePolynomial<Zp, E, O>>>,
    all_monomials: &HashMap<Vec<E>, MonomialData>,
    sorted_monomial_indices: &[usize],
    field: &Zp,
    buffer: &mut Vec<i64>,
    pivots: &mut Vec<Option<usize>>,
    print_stats: bool,
) {
    integer_matrix.resize(selected_polys.len(), vec![]);
    for (row, p) in integer_matrix.iter_mut().zip(selected_polys) {
        row.clear();

        for (coeff, exp) in p.coefficients.iter().zip(p.exponents_iter()).rev() {
            row.push((
                field.from_element(coeff) as i64,
                all_monomials.get(exp).unwrap().column,
            ));
        }
    }

    // Sort the matrix rows to put the shortest and most reduced pivots on top.
    sort_matrix_rows(integer_matrix);

    // row-reduce the sparse matrix
    for p in &mut *pivots {
        *p = None;
    }

    buffer.resize(sorted_monomial_indices.len(), 0);
    pivots.resize(sorted_monomial_indices.len(), None);

    let p = field.get_prime() as i64;
    let p2 = p * p;

    let mut pc = 0;
    for r in 0..integer_matrix.len() {
        // identify all pivots
        if let Some((coeff, col)) = integer_matrix[r].first_mut()
            && pivots[*col].is_none()
        {
            pivots[*col] = Some(r);
            pc += 1;

            if *coeff != 1 {
                let inv_pivot = u32_inv(*coeff as u32, field.get_prime());

                for (coeff, _) in &mut integer_matrix[r] {
                    *coeff *= inv_pivot as i64;
                    *coeff %= field.get_prime() as i64;
                }
            }
        }
    }

    if print_stats {
        println!(
            "\tPivots={}, rows to reduce={}",
            pc,
            integer_matrix.len() - pc
        );
    }

    for r in 0..integer_matrix.len() {
        if integer_matrix[r].is_empty() {
            continue;
        }

        if let Some((coeff, col)) = integer_matrix[r].first_mut()
            && pivots[*col].is_none()
        {
            pivots[*col] = Some(r);

            if *coeff != 1 {
                let inv_pivot = u32_inv(*coeff as u32, field.get_prime());

                for (coeff, _) in &mut integer_matrix[r] {
                    *coeff *= inv_pivot as i64;
                    *coeff %= field.get_prime() as i64;
                }
            }
        }

        // do not reduce pivots
        if pivots.contains(&Some(r)) {
            continue;
        }

        // copy row into the buffer
        for (coeff, col) in &*integer_matrix[r] {
            buffer[*col] = *coeff;
        }

        for i in 0..buffer.len() {
            if buffer[i] != 0 {
                buffer[i] %= p;
            }

            if buffer[i] == 0 {
                continue;
            }

            let Some(pivot_index) = pivots[i] else {
                // keep on reducing this new pivot
                continue;
            };

            let pivot = &integer_matrix[pivot_index];
            let c = buffer[i];

            buffer[i] = 0;

            let mut t;
            let mut m;
            for (coeff, col) in pivot.iter().skip(1) {
                t = buffer[*col];
                m = *coeff * c;

                if t >= m {
                    t -= m;
                } else {
                    t += p2 - m;
                }

                buffer[*col] = t;
            }
        }

        integer_matrix[r].clear();

        for (col, coeff) in buffer.iter_mut().enumerate() {
            if *coeff != 0 {
                integer_matrix[r].push((*coeff, col));
                *coeff = 0;
            }
        }

        if let Some((coeff, col)) = integer_matrix[r].first() {
            pivots[*col] = Some(r);

            if *coeff != 1 {
                let inv_pivot = u32_inv(*coeff as u32, field.get_prime());

                for (coeff, _) in &mut integer_matrix[r] {
                    *coeff *= inv_pivot as i64;
                    *coeff %= field.get_prime() as i64;
                }
            }
        }
    }

    // TODO: do back substitution
    integer_matrix.retain(|r| !r.is_empty());

    matrix.clear();
    matrix.reserve(integer_matrix.len());
    for row in integer_matrix {
        matrix.push(
            row.iter()
                .map(|(coeff, col)| (field.to_element(*coeff as u32), *col))
                .collect(),
        );
    }
}

fn sort_matrix_rows<C>(matrix: &mut [Vec<(C, usize)>]) {
    matrix.sort_unstable_by(|r1, r2| {
        r1[0]
            .1
            .cmp(&r2[0].1)
            .then(r1.len().cmp(&r2.len()))
            .then_with(|| {
                for ((_, i1), (_, i2)) in r1.iter().zip(r2) {
                    match i1.cmp(i2) {
                        Ordering::Equal => {}
                        x => {
                            return x.reverse();
                        }
                    }
                }

                Ordering::Equal
            })
    });
}

fn u32_inv(coeff: u32, prime: u32) -> u32 {
    // Extended Euclidean algorithm: a x + b p = gcd(x, p) = 1 or a x = 1 (mod p).
    let mut u1: u32 = 1;
    let mut u3 = coeff;
    let mut v1: u32 = 0;
    let mut v3 = prime;
    let mut even_iter: bool = true;

    while v3 != 0 {
        let q = u3 / v3;
        let t3 = u3 % v3;
        let t1 = u1 + q * v1;
        u1 = v1;
        v1 = t1;
        u3 = v3;
        v3 = t3;
        even_iter = !even_iter;
    }

    if even_iter { u1 } else { prime - u1 }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::{
        atom::{Atom, AtomCore},
        domains::{
            Ring, RingOps, algebraic_number::AlgebraicContext, finite_field::Zp, rational::Q,
        },
        parse,
        poly::{
            GrevLexOrder, LexOrder, PolyVariable, groebner::GroebnerBasis,
            polynomial::MultivariatePolynomial,
        },
        symbol,
    };

    fn atom_solutions(
        solutions: Vec<super::PolynomialSolution<crate::domains::rational::RationalField>>,
    ) -> Vec<ahash::HashMap<PolyVariable, Atom>> {
        solutions
            .iter()
            .map(|solution| solution.to_atom_map().unwrap())
            .collect()
    }

    #[test]
    fn solve_test() {
        let x = PolyVariable::from(symbol!("x"));
        let y = PolyVariable::from(symbol!("y"));
        let z = PolyVariable::from(symbol!("z"));
        let variables = Arc::new(vec![x.clone(), y.clone(), z.clone()]);
        let ideal: Vec<MultivariatePolynomial<_, u16>> = ["x-y-z", "y^2-z", "z^2-2"]
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&Q, Some(variables.clone())))
            .collect();
        let solutions = atom_solutions(GroebnerBasis::new(&ideal, false).solve().unwrap());

        assert_eq!(solutions.len(), 4);
        let sqrt_two = parse!("2^(1/2)");
        let negative_sqrt_two = parse!("-2^(1/2)");
        for solution in solutions {
            let x_value = solution.get(&x).unwrap();
            let y_value = solution.get(&y).unwrap();
            let z_value = solution.get(&z).unwrap();
            println!(
                "x = {x_value} ({}), y = {y_value} ({}), z = {z_value} ({}))",
                x_value.to_float(16),
                y_value.to_float(16),
                z_value.to_float(16)
            );
            assert_algebraic_zero(x_value.clone() - y_value.clone() - z_value.clone());
            assert_algebraic_zero(y_value.clone().pow(Atom::num(2)) - z_value.clone());
            assert_algebraic_zero(z_value.clone().pow(Atom::num(2)) - Atom::num(2));
            assert!(
                z_value == &sqrt_two || z_value == &negative_sqrt_two,
                "solve should simplify algebraic-number output, got {z_value}"
            );
        }
    }

    fn assert_algebraic_zero(expression: Atom) {
        if expression.is_zero() {
            return;
        }

        let mut context = AlgebraicContext::from_atom(expression.as_view()).unwrap();
        assert!(
            !context.is_trivial(),
            "expression should contain an algebraic number"
        );
        let value = context.convert_atom(expression.as_view()).unwrap();
        assert!(
            context.field().is_zero(&value),
            "expected {expression} to be zero"
        );
    }

    #[test]
    fn cyclic4() {
        let polys = [
            "v1 v2 v3 v4 - 1",
            "v1 v2 v3 + v1 v2 v4 + v1 v3 v4 + v2 v3 v4",
            "v1 v2 + v2 v3 + v1 v4 + v3 v4",
            "v1 + v2 + v3 + v4",
        ];

        let ideal: Vec<MultivariatePolynomial<_, u16>> = polys
            .iter()
            .map(|x| {
                let a = parse!(x).expand();
                a.to_polynomial(&Zp::new(13), None)
            })
            .collect();

        // compute the Groebner basis with lex ordering
        let gb = GroebnerBasis::new(&ideal, false);

        let res = [
            "v4+v3+v2+v1",
            "v4^2+2*v2*v4+v2^2",
            "11*v4^2+v3*v4+v3^2*v4^4-v2*v4+v2*v3",
            "-v4+v4^5-v2+v2*v4^4",
            "-v4-v3+v3^2*v4^3+v3^3*v4^2",
            "1-v4^4-v3^2*v4^2+v3^2*v4^6",
        ];

        let res: Vec<MultivariatePolynomial<_, u16>> = res
            .iter()
            .map(|x| {
                let a = parse!(x).expand();
                a.to_polynomial(&Zp::new(13), ideal[0].variables().clone())
            })
            .collect();

        assert_eq!(gb.system, res);

        // compute the Groebner basis with grevlex ordering by converting the polynomials
        let grevlex_ideal: Vec<_> = ideal.iter().map(|p| p.reorder::<GrevLexOrder>()).collect();
        let gb = GroebnerBasis::new(&grevlex_ideal, false);

        let res = [
            "v4+v3+v2+v1",
            "v4^2+2*v2*v4+v2^2",
            "-v4^3-v2*v4^2+v3^2*v4+v2*v3^2",
            "-1-v4^4+v3*v4^3-v2*v4^3+v3^2*v4^2+v2*v3*v4^2",
            "-v4-v2+v4^5+v2*v4^4",
            "-v4-v3+v3^2*v4^3+v3^3*v4^2",
            "11*v4^2+v3*v4-v2*v4+v2*v3+v3^2*v4^4",
        ];

        let res: Vec<MultivariatePolynomial<_, u16, _>> = res
            .iter()
            .map(|x| {
                let a = parse!(x).expand();
                a.to_polynomial(&Zp::new(13), ideal[0].variables().clone())
                    .reorder::<GrevLexOrder>()
            })
            .collect();

        assert_eq!(gb.system, res);
    }

    #[test]
    fn rational_field_uses_default_echelonization() {
        let ideal: Vec<MultivariatePolynomial<_, u16>> = ["x+y", "x-y"]
            .iter()
            .map(|x| parse!(x).expand().to_polynomial(&Q, None))
            .collect();

        let gb = GroebnerBasis::new(&ideal, false);

        assert!(ideal.iter().all(|p| p.reduce(&gb.system).is_zero()));
    }

    #[test]
    fn change_order_converts_grevlex_basis_to_lex() {
        let x = PolyVariable::from(symbol!("x"));
        let y = PolyVariable::from(symbol!("y"));
        let variables = Arc::new(vec![x, y]);
        let ideal_lex: Vec<MultivariatePolynomial<_, u16>> = ["x*y-1", "y^2-x"]
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&Q, Some(variables.clone())))
            .collect();
        let ideal_grevlex = ideal_lex
            .iter()
            .map(|polynomial| polynomial.reorder::<GrevLexOrder>())
            .collect::<Vec<_>>();

        let grevlex_basis = GroebnerBasis::new(&ideal_grevlex, false);
        let converted = grevlex_basis.change_order::<LexOrder>().unwrap();
        let direct_lex = GroebnerBasis::new(&ideal_lex, false);

        assert_eq!(converted.system, direct_lex.system);
        assert!(GroebnerBasis::is_groebner_basis(&converted.system));
        assert!(
            ideal_lex
                .iter()
                .all(|polynomial| polynomial.reduce(&converted.system).is_zero())
        );
    }

    #[test]
    fn change_order_accepts_a_nonlex_target_order() {
        let ideal_lex: Vec<MultivariatePolynomial<_, u16>> = ["x*y-1", "y^2-x"]
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&Q, None))
            .collect();
        let lex_basis = GroebnerBasis::new(&ideal_lex, false);

        let converted = lex_basis.change_order::<GrevLexOrder>().unwrap();
        let ideal_grevlex = ideal_lex
            .iter()
            .map(|polynomial| polynomial.reorder::<GrevLexOrder>())
            .collect::<Vec<_>>();
        let direct_grevlex = GroebnerBasis::new(&ideal_grevlex, false);

        assert_eq!(converted.system, direct_grevlex.system);
        assert!(GroebnerBasis::is_groebner_basis(&converted.system));
    }

    #[test]
    fn change_order_works_over_finite_fields() {
        let field = Zp::new(7);
        let ideal_lex: Vec<MultivariatePolynomial<_, u16>> = ["x*y-1", "y^2-x"]
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&field, None))
            .collect();
        let ideal_grevlex = ideal_lex
            .iter()
            .map(|polynomial| polynomial.reorder::<GrevLexOrder>())
            .collect::<Vec<_>>();

        let converted = GroebnerBasis::new(&ideal_grevlex, false)
            .change_order::<LexOrder>()
            .unwrap();
        let direct_lex = GroebnerBasis::new(&ideal_lex, false);
        assert_eq!(converted.system, direct_lex.system);
    }

    #[test]
    fn change_order_preserves_nonradical_quotient_structure() {
        let ideal_lex: Vec<MultivariatePolynomial<_, u16>> = ["x^2", "x*y", "y^2"]
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&Q, None))
            .collect();
        let ideal_grevlex = ideal_lex
            .iter()
            .map(|polynomial| polynomial.reorder::<GrevLexOrder>())
            .collect::<Vec<_>>();

        let converted = GroebnerBasis::new(&ideal_grevlex, false)
            .change_order::<LexOrder>()
            .unwrap();
        let direct_lex = GroebnerBasis::new(&ideal_lex, false);
        assert_eq!(converted.system, direct_lex.system);
    }

    #[test]
    fn change_order_rejects_positive_dimensional_bases() {
        let ideal = vec![
            parse!("x*y")
                .to_polynomial::<_, u16>(&Q, None)
                .reorder::<GrevLexOrder>(),
        ];
        let error = GroebnerBasis::new(&ideal, false)
            .change_order::<LexOrder>()
            .err()
            .unwrap();
        assert!(error.contains("not zero-dimensional"));
    }

    #[test]
    fn change_order_preserves_the_inconsistent_ideal() {
        let ideal = ["x", "x-1"]
            .iter()
            .map(|polynomial| {
                parse!(polynomial)
                    .to_polynomial::<_, u16>(&Q, None)
                    .reorder::<GrevLexOrder>()
            })
            .collect::<Vec<_>>();
        let converted = GroebnerBasis::new(&ideal, false)
            .change_order::<LexOrder>()
            .unwrap();
        assert_eq!(converted.system.len(), 1);
        assert!(converted.system[0].is_one());
    }

    #[test]
    fn solve_lexicographic_shape_basis() {
        let ideal: Vec<MultivariatePolynomial<_, u16>> = ["x+y", "y^2-2"]
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&Q, None))
            .collect();
        let solutions = atom_solutions(GroebnerBasis::new(&ideal, false).solve().unwrap());

        let x = PolyVariable::from(symbol!("x"));
        let y = PolyVariable::from(symbol!("y"));
        assert_eq!(solutions.len(), 2);
        for solution in solutions {
            let y_value = solution.get(&y).unwrap();
            assert_eq!(solution.get(&x), Some(&(-y_value.clone())));
            assert_eq!(
                (y_value.clone().pow(Atom::num(2)) - Atom::num(2)).expand(),
                Atom::Zero
            );
        }
    }

    #[test]
    fn solve_ellipse_intersection_with_auxiliary_radii() {
        let kx = PolyVariable::from(symbol!("kx"));
        let ky = PolyVariable::from(symbol!("ky"));
        let r0 = PolyVariable::from(symbol!("r0"));
        let r1 = PolyVariable::from(symbol!("r1"));
        let r2 = PolyVariable::from(symbol!("r2"));
        let variables = Arc::new(vec![
            kx.clone(),
            ky.clone(),
            r0.clone(),
            r1.clone(),
            r2.clone(),
        ]);
        let equations = [
            "r0+r1-10",
            "r0+r2-10",
            "r0^2-kx^2-ky^2",
            "r1^2-(kx+1)^2-(ky+2)^2",
            "r2^2-(kx+6)^2-(ky+7)^2",
        ];
        let ideal: Vec<MultivariatePolynomial<_, u16>> = equations
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&Q, Some(variables.clone())))
            .collect();
        let solutions = atom_solutions(GroebnerBasis::new(&ideal, false).solve().unwrap());

        assert_eq!(solutions.len(), 2);
        for solution in solutions {
            let kx_value = solution.get(&kx).unwrap();
            let ky_value = solution.get(&ky).unwrap();
            let r0_value = solution.get(&r0).unwrap();
            let r1_value = solution.get(&r1).unwrap();
            let r2_value = solution.get(&r2).unwrap();

            assert_algebraic_zero(r0_value.clone() + r1_value.clone() - Atom::num(10));
            assert_algebraic_zero(r0_value.clone() + r2_value.clone() - Atom::num(10));
            assert_algebraic_zero(
                r0_value.clone().pow(Atom::num(2))
                    - kx_value.clone().pow(Atom::num(2))
                    - ky_value.clone().pow(Atom::num(2)),
            );
            assert_algebraic_zero(
                r1_value.clone().pow(Atom::num(2))
                    - (kx_value.clone() + Atom::num(1)).pow(Atom::num(2))
                    - (ky_value.clone() + Atom::num(2)).pow(Atom::num(2)),
            );
            assert_algebraic_zero(
                r2_value.clone().pow(Atom::num(2))
                    - (kx_value.clone() + Atom::num(6)).pow(Atom::num(2))
                    - (ky_value.clone() + Atom::num(7)).pow(Atom::num(2)),
            );

            // Positive auxiliary radii select the principal square-root
            // branches of the original, non-polynomial equations.
            assert!(f64::try_from(r0_value.to_float(16)).unwrap() > 0.0);
            assert!(f64::try_from(r1_value.to_float(16)).unwrap() > 0.0);
            assert!(f64::try_from(r2_value.to_float(16)).unwrap() > 0.0);
        }
    }

    #[test]
    fn polynomial_solutions_stay_in_one_algebraic_field() {
        let ideal: Vec<MultivariatePolynomial<_, u16>> = ["x+y", "y^2-2"]
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&Q, None))
            .collect();
        let solutions = GroebnerBasis::new(&ideal, false).solve().unwrap();
        let x = PolyVariable::from(symbol!("x"));
        let y = PolyVariable::from(symbol!("y"));

        assert_eq!(solutions.len(), 2);
        for solution in solutions {
            assert_eq!(solution.field().poly().degree(0), 2);
            let x_value = solution.get(&x).unwrap();
            let y_value = solution.get(&y).unwrap();
            assert!(
                solution
                    .field()
                    .is_zero(&solution.field().add(x_value, y_value))
            );
            assert!(solution.field().is_zero(&solution.field().sub(
                &solution.field().pow(y_value, 2),
                &solution.field().constant(2.into()),
            )));
        }
    }

    #[test]
    fn solve_parametric_shape_basis() {
        let variables = Arc::new(vec![
            PolyVariable::from(symbol!("x")),
            PolyVariable::from(symbol!("y")),
        ]);
        let ideal: Vec<MultivariatePolynomial<_, u16>> = ["x+y", "y^2-a"]
            .iter()
            .map(|expression| {
                parse!(expression)
                    .to_rational_polynomial::<_, _, u16>(&Q, &crate::domains::integer::Z, None)
                    .to_polynomial(&variables, true)
                    .unwrap()
            })
            .collect();

        let solutions = GroebnerBasis::new(&ideal, false)
            .solve_parametric()
            .unwrap();
        assert_eq!(solutions.len(), 2);

        let x = PolyVariable::from(symbol!("x"));
        let y = PolyVariable::from(symbol!("y"));
        let a = PolyVariable::from(symbol!("a"));
        let mut indices = Vec::new();
        for solution in solutions {
            let x_value = solution.get(&x).unwrap();
            let y_value = solution.get(&y).unwrap();
            assert!(std::ptr::eq(x_value.field(), y_value.field()));
            assert_eq!(y_value.polynomial().degree(0), 2);
            let field = y_value.field().quotient();
            assert!(field.is_zero(&field.add(x_value.value(), y_value.value())));
            indices.push(y_value.conjugate());

            assert_eq!(solution.field().parameters(), std::slice::from_ref(&a));
            assert!(!solution.field().generic_conditions().is_empty());

            let mut point = ahash::HashMap::default();
            point.insert(a.clone(), 2.into());
            let specialized = solution.specialize(&point).unwrap();
            let x_value = specialized.get(&x).unwrap();
            let y_value = specialized.get(&y).unwrap();
            assert!(
                specialized
                    .field()
                    .is_zero(&specialized.field().add(x_value, y_value))
            );

            point.insert(a.clone(), 0.into());
            assert!(solution.specialize(&point).is_err());
        }
        indices.sort_unstable();
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn solve_parametric_tracks_factor_collision_locus() {
        let variables = Arc::new(vec![PolyVariable::from(symbol!("x"))]);
        let ideal: Vec<MultivariatePolynomial<_, u16>> = ["x^2-a^2"]
            .iter()
            .map(|expression| {
                parse!(expression)
                    .to_rational_polynomial::<_, _, u16>(&Q, &crate::domains::integer::Z, None)
                    .to_polynomial(&variables, true)
                    .unwrap()
            })
            .collect();
        let solutions = GroebnerBasis::new(&ideal, false)
            .solve_parametric()
            .unwrap();
        assert_eq!(solutions.len(), 2);

        let a = PolyVariable::from(symbol!("a"));
        let mut point = ahash::HashMap::default();
        point.insert(a.clone(), 0.into());
        assert!(
            solutions
                .iter()
                .all(|solution| solution.specialize(&point).is_err())
        );

        point.insert(a, 2.into());
        assert!(
            solutions
                .iter()
                .all(|solution| solution.specialize(&point).is_ok())
        );
    }

    #[test]
    fn solve_parametric_adjoins_nonlinear_root_over_extension() {
        let variables = Arc::new(vec![
            PolyVariable::from(symbol!("x")),
            PolyVariable::from(symbol!("y")),
        ]);
        let ideal: Vec<MultivariatePolynomial<_, u16>> = ["x^2-y", "y^2-a"]
            .iter()
            .map(|expression| {
                parse!(expression)
                    .to_rational_polynomial::<_, _, u16>(&Q, &crate::domains::integer::Z, None)
                    .to_polynomial(&variables, true)
                    .unwrap()
            })
            .collect();

        let solutions = GroebnerBasis::new(&ideal, false)
            .solve_parametric()
            .unwrap();
        assert_eq!(solutions.len(), 4);

        let x = PolyVariable::from(symbol!("x"));
        let y = PolyVariable::from(symbol!("y"));
        let a = PolyVariable::from(symbol!("a"));
        let mut paths = solutions
            .iter()
            .map(|solution| solution.conjugates().to_vec())
            .collect::<Vec<_>>();
        paths.sort();
        assert_eq!(paths, vec![vec![0, 0], vec![0, 1], vec![1, 0], vec![1, 1]]);

        let mut point = ahash::HashMap::default();
        point.insert(a, 2.into());
        for solution in solutions {
            assert_eq!(solution.to_atom_map().len(), 2);
            let specialized = solution.specialize(&point).unwrap();
            let field = specialized.field();
            let x_value = specialized.get(&x).unwrap();
            let y_value = specialized.get(&y).unwrap();
            assert!(field.is_zero(&field.sub(&field.pow(x_value, 2), y_value,)));
            assert!(field.is_zero(&field.sub(&field.pow(y_value, 2), &field.constant(2.into()),)));
        }
    }

    #[test]
    fn solve_parametric_factors_nonlinear_polynomial_over_extension() {
        let variables = Arc::new(vec![
            PolyVariable::from(symbol!("x")),
            PolyVariable::from(symbol!("y")),
        ]);
        let ideal: Vec<MultivariatePolynomial<_, u16>> = ["x^2-a", "y^2-a"]
            .iter()
            .map(|expression| {
                parse!(expression)
                    .to_rational_polynomial::<_, _, u16>(&Q, &crate::domains::integer::Z, None)
                    .to_polynomial(&variables, true)
                    .unwrap()
            })
            .collect();

        let solutions = GroebnerBasis::new(&ideal, false)
            .solve_parametric()
            .unwrap();
        assert_eq!(solutions.len(), 4);

        let x = PolyVariable::from(symbol!("x"));
        let y = PolyVariable::from(symbol!("y"));
        for solution in solutions {
            assert_eq!(solution.conjugates().len(), 1);
            let field = solution.field().quotient();
            let x_value = solution.get(&x).unwrap();
            let y_value = solution.get(&y).unwrap();
            assert!(field.is_zero(&field.sub(
                &field.pow(x_value.value(), 2),
                &field.pow(y_value.value(), 2),
            )));
        }
    }

    #[test]
    fn solve_lexicographic_shape_basis_with_explicit_roots() {
        let ideal: Vec<MultivariatePolynomial<_, u16>> = ["x-y^2", "y^3-2"]
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&Q, None))
            .collect();
        let solutions = atom_solutions(GroebnerBasis::new(&ideal, false).solve().unwrap());

        let x = PolyVariable::from(symbol!("x"));
        let y = PolyVariable::from(symbol!("y"));
        assert_eq!(solutions.len(), 3);
        for solution in solutions {
            let x_value = solution.get(&x).unwrap();
            let y_value = solution.get(&y).unwrap();
            assert_algebraic_zero(x_value.clone() - y_value.clone().pow(Atom::num(2)));
        }
    }

    #[test]
    fn solve_lexicographic_basis_adjoins_nonlinear_roots() {
        let x = PolyVariable::from(symbol!("x"));
        let y = PolyVariable::from(symbol!("y"));
        let variables = Arc::new(vec![x.clone(), y.clone()]);
        let ideal: Vec<MultivariatePolynomial<_, u16>> = ["x^2-y", "y^2-2"]
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&Q, Some(variables.clone())))
            .collect();
        let solutions = atom_solutions(GroebnerBasis::new(&ideal, false).solve().unwrap());

        assert_eq!(solutions.len(), 4);
        for solution in solutions {
            let x_value = solution.get(&x).unwrap();
            let y_value = solution.get(&y).unwrap();
            assert_algebraic_zero(x_value.clone().pow(Atom::num(2)) - y_value.clone());
            assert_algebraic_zero(y_value.clone().pow(Atom::num(2)) - Atom::num(2));
        }
    }

    #[test]
    fn solve_lexicographic_basis_continues_after_adjoining() {
        let x = PolyVariable::from(symbol!("x"));
        let y = PolyVariable::from(symbol!("y"));
        let z = PolyVariable::from(symbol!("z"));
        let variables = Arc::new(vec![x.clone(), y.clone(), z.clone()]);
        let ideal: Vec<MultivariatePolynomial<_, u16>> = ["x-y-z", "y^2-z", "z^2-2"]
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&Q, Some(variables.clone())))
            .collect();
        let solutions = atom_solutions(GroebnerBasis::new(&ideal, false).solve().unwrap());

        assert_eq!(solutions.len(), 4);
        for solution in solutions {
            let x_value = solution.get(&x).unwrap();
            let y_value = solution.get(&y).unwrap();
            let z_value = solution.get(&z).unwrap();
            assert_algebraic_zero(x_value.clone() - y_value.clone() - z_value.clone());
            assert_algebraic_zero(y_value.clone().pow(Atom::num(2)) - z_value.clone());
            assert_algebraic_zero(z_value.clone().pow(Atom::num(2)) - Atom::num(2));
        }
    }

    #[test]
    fn solve_lexicographic_shape_basis_back_substitutes_in_order() {
        let ideal: Vec<MultivariatePolynomial<_, u16>> = ["x-y-1", "y-z-1", "z^2-1"]
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&Q, None))
            .collect();
        let solutions = atom_solutions(GroebnerBasis::new(&ideal, false).solve().unwrap());

        let x = PolyVariable::from(symbol!("x"));
        let y = PolyVariable::from(symbol!("y"));
        let z = PolyVariable::from(symbol!("z"));
        assert_eq!(solutions.len(), 2);
        assert!(solutions.iter().any(|solution| {
            solution.get(&x) == Some(&Atom::num(1))
                && solution.get(&y) == Some(&Atom::num(0))
                && solution.get(&z) == Some(&Atom::num(-1))
        }));
        assert!(solutions.iter().any(|solution| {
            solution.get(&x) == Some(&Atom::num(3))
                && solution.get(&y) == Some(&Atom::num(2))
                && solution.get(&z) == Some(&Atom::num(1))
        }));
    }

    #[test]
    fn solve_lexicographic_basis_deduplicates_roots() {
        let ideal: Vec<MultivariatePolynomial<_, u16>> = ["x-y", "(y-1)^2"]
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&Q, None))
            .collect();
        let solutions = atom_solutions(GroebnerBasis::new(&ideal, false).solve().unwrap());

        let x = PolyVariable::from(symbol!("x"));
        let y = PolyVariable::from(symbol!("y"));
        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].get(&x), Some(&Atom::num(1)));
        assert_eq!(solutions[0].get(&y), Some(&Atom::num(1)));
    }

    #[test]
    fn solve_lexicographic_basis_reports_unsupported_systems() {
        let ideal: Vec<MultivariatePolynomial<_, u16>> = ["x^2-2", "y^2-3"]
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&Q, None))
            .collect();
        let solutions = GroebnerBasis::new(&ideal, false).solve().unwrap();
        assert_eq!(solutions.len(), 4);

        let positive_dimensional: Vec<MultivariatePolynomial<_, u16>> = ["x-y"]
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&Q, None))
            .collect();
        assert!(
            GroebnerBasis::new(&positive_dimensional, false)
                .solve()
                .is_err()
        );

        let inconsistent: Vec<MultivariatePolynomial<_, u16>> = ["x", "x-1"]
            .iter()
            .map(|polynomial| parse!(polynomial).to_polynomial(&Q, None))
            .collect();
        assert!(
            GroebnerBasis::new(&inconsistent, false)
                .solve()
                .unwrap()
                .is_empty()
        );

        let empty: Vec<MultivariatePolynomial<_, u16>> = Vec::new();
        assert!(GroebnerBasis::new(&empty, false).solve().is_err());
    }
}
