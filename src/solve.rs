//! Solve systems of equations.
//!
//! See [AtomCore::solve_linear_system] and [AtomCore::nsolve_system].

use std::{ops::Neg, sync::Arc};

use ahash::{HashMap, HashSet};
use numerica::domains::{
    Field, RealEmbedding, Ring, RingOps,
    float::{Complex, Float, RealLike},
    rational::Rational,
};

use crate::{
    atom::{Atom, AtomCore, AtomView, Indeterminate},
    coefficient::{Coefficient, ConvertToRing},
    domains::{
        InternalOrdering, SelfRing,
        algebraic_number::AlgebraicContext,
        float::{FloatField, Real, SingleFloat},
        integer::Z,
        rational::Q,
        rational_polynomial::{RationalPolynomial, RationalPolynomialField},
    },
    evaluate::{EvaluationDomain, FunctionMap, OptimizationSettings},
    poly::{
        GrevLexOrder, LexOrder, PolyVariable, PositiveExponent, groebner::GroebnerBasis,
        polynomial::MultivariatePolynomial,
    },
    tensors::matrix::{Matrix, MatrixError},
};

#[derive(Clone)]
struct AuxiliaryPower {
    variable: PolyVariable,
    base: Atom,
    exponent: Atom,
    denominator: usize,
}

enum ParametricSolveResult {
    Solved(Vec<SolveBranch>),
    PositiveDimensional(usize),
    Inconsistent,
}

#[derive(Clone)]
pub(crate) struct SolveBranch {
    values: HashMap<PolyVariable, Atom>,
    nonzero_conditions: Vec<Atom>,
}

impl SolveBranch {
    fn unconditional(values: HashMap<PolyVariable, Atom>) -> Self {
        Self {
            values,
            nonzero_conditions: Vec::new(),
        }
    }
}

/// Errors that can occur when solving a system.
/// Underdetermined systems return a partial solution.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum SolveError {
    /// The system contains complex coefficients, but the solver works over the reals.
    ComplexCoefficients,
    /// The number of equations differs from the number of unknowns.
    NonSquareSystem,
    /// Initial values were not provided for all unknowns.
    IncompleteInitialValues,
    /// Newton's method encountered a zero derivative.
    ZeroDerivative,
    /// Newton's method could not invert the Jacobian.
    SingularJacobian,
    /// The solver did not converge within the iteration limit.
    NoConvergence,
    /// The input system is empty.
    EmptySystem,
    /// The input system is not linear in the requested variables.
    NonLinearSystem,
    /// The system was underdetermined. The partial solution is returned.
    Underdetermined {
        /// Rank of the system.
        rank: u32,
        /// Partial solution found, that may contain free variables.
        partial_solution: Vec<Atom>,
    },
    Other(String),
}

/// A domain supported by the exact solver.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum SolveDomain {
    /// Keep solutions whose requested values are integers.
    Integers,
    /// Keep solutions whose requested values are rational numbers.
    Rationals,
    /// Keep solutions whose requested values are real numbers.
    Reals,
    /// Keep all complex solutions.
    #[default]
    Complexes,
}

pub use SolveDomain::{Complexes, Integers, Rationals, Reals};

#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DomainMembership {
    Yes,
    No,
    Indeterminate,
}

fn algebraic_domain_membership(value: &Atom, domain: SolveDomain) -> DomainMembership {
    let Ok(mut context) = AlgebraicContext::from_atom(value.as_view()) else {
        return DomainMembership::Indeterminate;
    };
    let Ok(element) = context.convert_atom(value.as_view()) else {
        return DomainMembership::Indeterminate;
    };

    match domain {
        Integers => {
            if context
                .field()
                .element_to_atom_simplified(&element)
                .is_integer()
            {
                DomainMembership::Yes
            } else {
                DomainMembership::No
            }
        }
        Rationals => {
            let simplified = context.field().element_to_atom_simplified(&element);
            if Rational::try_from(simplified.as_view()).is_ok() {
                DomainMembership::Yes
            } else {
                DomainMembership::No
            }
        }
        Reals if context.field().try_sign(&element).is_ok() => DomainMembership::Yes,
        Reals => DomainMembership::No,
        Complexes => DomainMembership::Yes,
    }
}

pub(crate) fn value_in_domain(value: &Atom, domain: SolveDomain) -> DomainMembership {
    match domain {
        Complexes => DomainMembership::Yes,
        Integers if value.is_integer() => DomainMembership::Yes,
        Rationals if Rational::try_from(value.as_view()).is_ok() => DomainMembership::Yes,
        Reals if value.is_real() => DomainMembership::Yes,
        _ => algebraic_domain_membership(value, domain),
    }
}

fn rational_denominator<E: PositiveExponent + 'static>(expression: AtomView<'_>) -> Option<Atom> {
    let rational: RationalPolynomial<_, E> =
        expression.try_to_rational_polynomial(&Q, &Z, None).ok()?;
    (!rational.denominator.is_one()).then(|| rational.denominator.to_expression())
}

/// One branch of an exact solution to a system of equations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solution {
    values: HashMap<PolyVariable, Atom>,
    free_variables: Vec<PolyVariable>,
    conditions: Vec<SolutionCondition>,
    domain: SolveDomain,
}

impl Solution {
    fn new(
        values: HashMap<PolyVariable, Atom>,
        free_variables: Vec<PolyVariable>,
        conditions: Vec<SolutionCondition>,
        domain: SolveDomain,
    ) -> Self {
        Self {
            values,
            free_variables,
            conditions,
            domain,
        }
    }

    /// Return whether this branch contains a free variable or a domain
    /// membership that could not be decided exactly.
    pub fn is_indeterminate(&self) -> bool {
        !self.free_variables.is_empty()
            || self
                .conditions
                .iter()
                .any(|condition| matches!(condition, SolutionCondition::DomainMembership { .. }))
    }

    /// Return whether this branch has conditions that must be satisfied.
    pub fn is_conditional(&self) -> bool {
        !self.conditions.is_empty()
    }

    /// Return whether this branch describes a family with free variables.
    pub fn is_parametric(&self) -> bool {
        !self.free_variables.is_empty()
    }

    /// Return whether this branch leaves any requested variables free.
    pub fn is_underdetermined(&self) -> bool {
        !self.free_variables.is_empty()
    }

    /// Number of requested variables determined in terms of the free inputs.
    pub fn rank(&self) -> usize {
        self.values.len() - self.free_variables.len()
    }

    /// Dimension of this solution branch, measured by its free inputs.
    pub fn dimension(&self) -> usize {
        self.free_variables.len()
    }

    /// Variables treated as free inputs on this branch.
    pub fn free_variables(&self) -> &[PolyVariable] {
        &self.free_variables
    }

    /// Conditions under which this branch is valid.
    pub fn conditions(&self) -> &[SolutionCondition] {
        &self.conditions
    }

    /// Domain requested for this solve operation.
    pub fn domain(&self) -> SolveDomain {
        self.domain
    }

    /// Borrow the variable-value map for this branch.
    pub fn as_map(&self) -> &HashMap<PolyVariable, Atom> {
        &self.values
    }

    /// Consume the solution and return its variable-value map.
    pub fn into_values(self) -> HashMap<PolyVariable, Atom> {
        self.values
    }
}

/// A condition attached to an exact solution branch.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SolutionCondition {
    /// This expression must not vanish. Such conditions arise from
    /// denominators and exceptional loci of generic parametric solutions.
    NonZero(Atom),
    /// Membership of this value in the requested domain could not be decided
    /// exactly. The branch is retained and marked indeterminate.
    DomainMembership {
        variable: PolyVariable,
        value: Atom,
        domain: SolveDomain,
    },
}

impl std::fmt::Display for SolutionCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonZero(expression) => write!(f, "{expression} != 0"),
            Self::DomainMembership { value, domain, .. } => write!(f, "{value} in {domain:?}"),
        }
    }
}

impl std::fmt::Display for Solution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut values = self.values.iter().collect::<Vec<_>>();
        values.sort_by(|(left, _), (right, _)| left.cmp(right));

        f.write_str("{")?;
        for (index, (variable, value)) in values.into_iter().enumerate() {
            if index > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{variable} = {value}")?;
        }
        f.write_str("}")?;

        if !self.conditions.is_empty() {
            let mut conditions = self
                .conditions
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            conditions.sort();
            write!(f, " where {}", conditions.join(", "))?;
        }

        Ok(())
    }
}

impl std::ops::Deref for Solution {
    type Target = HashMap<PolyVariable, Atom>;

    fn deref(&self) -> &Self::Target {
        &self.values
    }
}

impl IntoIterator for Solution {
    type Item = (PolyVariable, Atom);
    type IntoIter = <HashMap<PolyVariable, Atom> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.values.into_iter()
    }
}

impl<'a> IntoIterator for &'a Solution {
    type Item = (&'a PolyVariable, &'a Atom);
    type IntoIter = <&'a HashMap<PolyVariable, Atom> as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

/// A pending exact solve operation.
pub struct SolveBuilder<'a, T: AtomCore> {
    system: &'a [T],
    domain: SolveDomain,
}

impl<'a, T: AtomCore> SolveBuilder<'a, T> {
    pub(crate) fn new(system: &'a [T]) -> Self {
        Self {
            system,
            domain: Complexes,
        }
    }

    /// Select the domain in which solutions must lie.
    pub fn over(mut self, domain: SolveDomain) -> Self {
        self.domain = domain;
        self
    }

    /// Solve the system with respect to `variables`, using `u16` exponents.
    pub fn wrt<V: AtomCore>(&self, variables: &[V]) -> Result<Vec<Solution>, SolveError> {
        self.wrt_with_exponent::<u16, V>(variables)
    }

    /// Solve the system with respect to `variables` using exponent type `E`.
    pub fn wrt_with_exponent<E: PositiveExponent + 'static, V: AtomCore>(
        &self,
        variables: &[V],
    ) -> Result<Vec<Solution>, SolveError> {
        let polynomial_variables = variables
            .iter()
            .map(|variable| variable.as_atom_view().to_owned().try_into())
            .collect::<Result<Vec<PolyVariable>, String>>()
            .map_err(SolveError::Other)?;

        let system_denominators = self
            .system
            .iter()
            .filter_map(|expression| rational_denominator::<E>(expression.as_atom_view()))
            .collect::<Vec<_>>();

        let raw_solutions =
            match AtomView::solve_impl::<E, _, _>(self.system, variables, self.domain) {
                Ok(solutions) => solutions,
                Err(SolveError::Underdetermined {
                    partial_solution, ..
                }) => vec![SolveBranch::unconditional(
                    polynomial_variables
                        .iter()
                        .cloned()
                        .zip(partial_solution)
                        .collect(),
                )],
                Err(error) => return Err(error),
            };

        Ok(raw_solutions
            .into_iter()
            .filter_map(|branch| {
                let values = branch.values;
                let mut free_variables = Vec::new();
                let mut conditions = branch
                    .nonzero_conditions
                    .into_iter()
                    .map(SolutionCondition::NonZero)
                    .collect::<Vec<_>>();

                let mut nonzero_conditions = conditions
                    .iter()
                    .filter_map(|condition| match condition {
                        SolutionCondition::NonZero(expression) => Some(expression.clone()),
                        SolutionCondition::DomainMembership { .. } => None,
                    })
                    .collect::<HashSet<_>>();
                let denominator_conditions = system_denominators
                    .iter()
                    .map(|denominator| {
                        AtomView::substitute_algebraic_solution(denominator.as_view(), &values)
                    })
                    .chain(
                        values
                            .values()
                            .filter_map(|value| rational_denominator::<E>(value.as_view())),
                    );
                for denominator in denominator_conditions {
                    let denominator = denominator.expand();
                    match AtomView::algebraically_zero(&denominator) {
                        Some(true) => return None,
                        Some(false) => continue,
                        None => {}
                    }
                    if nonzero_conditions.insert(denominator.clone()) {
                        conditions.push(SolutionCondition::NonZero(denominator));
                    }
                }

                for variable in &polynomial_variables {
                    let value = values.get(variable)?;
                    if value == &variable.to_atom() {
                        free_variables.push(variable.clone());
                        continue;
                    }
                    match value_in_domain(value, self.domain) {
                        DomainMembership::Yes => {}
                        DomainMembership::No => return None,
                        DomainMembership::Indeterminate => {
                            conditions.push(SolutionCondition::DomainMembership {
                                variable: variable.clone(),
                                value: value.clone(),
                                domain: self.domain,
                            });
                        }
                    }
                }
                Some(Solution::new(
                    values,
                    free_variables,
                    conditions,
                    self.domain,
                ))
            })
            .collect())
    }
}

impl std::error::Error for SolveError {}

impl From<String> for SolveError {
    fn from(value: String) -> Self {
        SolveError::Other(value)
    }
}

impl From<&str> for SolveError {
    fn from(value: &str) -> Self {
        SolveError::Other(value.to_owned())
    }
}

impl std::fmt::Display for SolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SolveError::ComplexCoefficients => {
                f.write_str("Complex coefficients are not supported")
            }
            SolveError::NonSquareSystem => {
                f.write_str("System must have same number of equations as there are unknowns")
            }
            SolveError::IncompleteInitialValues => {
                f.write_str("Initial values must be provided for all unknowns")
            }
            SolveError::ZeroDerivative => f.write_str("Derivative is zero"),
            SolveError::SingularJacobian => f.write_str("Could not invert Jacobian"),
            SolveError::NoConvergence => f.write_str("Did not converge"),
            SolveError::EmptySystem => f.write_str("Empty system"),
            SolveError::NonLinearSystem => f.write_str("Not a linear system"),
            SolveError::Underdetermined {
                rank,
                partial_solution,
            } => write!(
                f,
                "Underdetermined system of rank {}/{}. Partial solution: {:?}",
                rank,
                partial_solution.len(),
                partial_solution
            ),
            SolveError::Other(e) => f.write_str(e),
        }
    }
}

impl AtomView<'_> {
    fn leading_ideal_dimension<E: PositiveExponent>(
        leading_monomials: &[Vec<E>],
        nvars: usize,
    ) -> usize {
        fn subsets_of_size(
            nvars: usize,
            size: usize,
            start: usize,
            current: &mut Vec<usize>,
            subsets: &mut Vec<Vec<usize>>,
        ) {
            if current.len() == size {
                subsets.push(current.clone());
                return;
            }

            let remaining = size - current.len();
            for variable in start..=nvars - remaining {
                current.push(variable);
                subsets_of_size(nvars, size, variable + 1, current, subsets);
                current.pop();
            }
        }

        for size in (0..=nvars).rev() {
            let mut subsets = Vec::new();
            subsets_of_size(nvars, size, 0, &mut Vec::new(), &mut subsets);
            if subsets.into_iter().any(|subset| {
                !leading_monomials.iter().any(|leading| {
                    leading.iter().enumerate().all(|(variable, exponent)| {
                        exponent.is_zero() || subset.contains(&variable)
                    })
                })
            }) {
                return size;
            }
        }

        0
    }

    fn preferred_input_sets(nvars: usize, dimension: usize) -> Vec<Vec<usize>> {
        fn collect(
            nvars: usize,
            dimension: usize,
            start: usize,
            current: &mut Vec<usize>,
            result: &mut Vec<Vec<usize>>,
        ) {
            if current.len() == dimension {
                result.push(current.clone());
                return;
            }

            let remaining = dimension - current.len();
            for variable in start..=nvars - remaining {
                current.push(variable);
                collect(nvars, dimension, variable + 1, current, result);
                current.pop();
            }
        }

        let mut result = Vec::new();
        collect(nvars, dimension, 0, &mut Vec::new(), &mut result);
        result.sort_by(|left, right| right.cmp(left));
        result
    }

    fn solve_parametric_polynomial_system<E: PositiveExponent + 'static, T: AtomCore>(
        system: &[T],
        variables: &[PolyVariable],
        input_variables: &HashSet<PolyVariable>,
        domain: SolveDomain,
    ) -> Result<ParametricSolveResult, SolveError> {
        let rationals = system
            .iter()
            .map(|expression| {
                expression
                    .as_atom_view()
                    .try_to_rational_polynomial(&Q, &Z, None)
                    .map_err(|error| SolveError::Other(error.to_string()))
            })
            .collect::<Result<Vec<RationalPolynomial<_, E>>, SolveError>>()?;

        let has_denominators = rationals
            .iter()
            .any(|rational| !rational.denominator.is_one());
        let mut polynomial_variables = variables
            .iter()
            .filter(|variable| !input_variables.contains(*variable))
            .cloned()
            .collect::<Vec<_>>();
        let saturation_variable = if has_denominators {
            let mut index = 0;
            let variable = loop {
                let candidate = PolyVariable::Temporary(index);
                if !variables.contains(&candidate) {
                    break candidate;
                }
                index += 1;
            };
            polynomial_variables.push(variable.clone());
            Some(variable)
        } else {
            None
        };
        let polynomial_variables = Arc::new(polynomial_variables);

        let mut denominators = Vec::new();
        let mut polynomials = Vec::new();
        for rational in rationals {
            let numerator_one = rational.numerator.one();
            let numerator = RationalPolynomial {
                numerator: rational.numerator,
                denominator: numerator_one,
            }
            .to_polynomial(polynomial_variables.as_ref(), true)
            .map_err(|error| SolveError::Other(error.to_string()))?;
            if !numerator.is_zero() {
                polynomials.push(numerator.reorder::<GrevLexOrder>());
            }

            if !rational.denominator.is_one() {
                let denominator_one = rational.denominator.one();
                let denominator = RationalPolynomial {
                    numerator: rational.denominator,
                    denominator: denominator_one,
                }
                .to_polynomial(polynomial_variables.as_ref(), true)
                .map_err(|error| SolveError::Other(error.to_string()))?;
                denominators.push(denominator);
            }
        }

        if let Some(saturation_variable) = saturation_variable {
            let mut denominator_product = denominators
                .first()
                .expect("A saturation variable requires a denominator")
                .one();
            for denominator in denominators {
                denominator_product = &denominator_product * &denominator;
            }
            let helper = denominator_product
                .variable(&saturation_variable)
                .map_err(SolveError::Other)?;
            let saturation = &helper * &denominator_product - denominator_product.one();
            polynomials.push(saturation.reorder::<GrevLexOrder>());
        }

        if polynomials.is_empty() {
            if polynomial_variables.is_empty() {
                return Ok(ParametricSolveResult::Solved(vec![
                    SolveBranch::unconditional(
                        variables
                            .iter()
                            .map(|variable| (variable.clone(), variable.to_atom()))
                            .collect(),
                    ),
                ]));
            }
            return Ok(ParametricSolveResult::PositiveDimensional(
                polynomial_variables.len(),
            ));
        }

        let basis = GroebnerBasis::new(&polynomials, false);
        if basis
            .system
            .iter()
            .any(|polynomial| !polynomial.is_zero() && polynomial.is_constant())
        {
            return Ok(ParametricSolveResult::Inconsistent);
        }

        let leading_monomials = basis
            .system
            .iter()
            .filter(|polynomial| !polynomial.is_zero())
            .map(|polynomial| polynomial.max_exp().to_vec())
            .collect::<Vec<_>>();
        let dimension =
            Self::leading_ideal_dimension(&leading_monomials, polynomial_variables.len());
        if dimension > 0 {
            return Ok(ParametricSolveResult::PositiveDimensional(dimension));
        }

        let basis = basis
            .change_order::<LexOrder>()
            .map_err(SolveError::Other)?;
        let solutions = basis
            .solve_parametric_in_base_field(matches!(domain, Integers | Rationals))
            .map_err(SolveError::Other)?
            .into_iter()
            .map(|solution| {
                let nonzero_conditions = solution
                    .field()
                    .generic_conditions()
                    .iter()
                    .map(|condition| condition.to_expression())
                    .collect();
                let values = variables
                    .iter()
                    .filter_map(|variable| {
                        if input_variables.contains(variable) {
                            Some((variable.clone(), variable.to_atom()))
                        } else {
                            solution
                                .get(variable)
                                .map(|value| (variable.clone(), value.to_atom()))
                        }
                    })
                    .collect();
                SolveBranch {
                    values,
                    nonzero_conditions,
                }
            })
            .collect();
        Ok(ParametricSolveResult::Solved(solutions))
    }

    fn solve_positive_dimensional_polynomial_system<E: PositiveExponent + 'static, T: AtomCore>(
        system: &[T],
        variables: &[PolyVariable],
        dimension: usize,
        domain: SolveDomain,
    ) -> Result<Vec<SolveBranch>, SolveError> {
        if dimension > variables.len() {
            return Err(SolveError::Other(format!(
                "The polynomial system has dimension {dimension}, but only {} solve variables",
                variables.len()
            )));
        }

        let maximize_base_field_branches = matches!(domain, Integers | Rationals);
        let mut best_solutions = None;
        for indices in Self::preferred_input_sets(variables.len(), dimension) {
            let input_variables = indices
                .into_iter()
                .map(|index| variables[index].clone())
                .collect::<HashSet<_>>();
            match Self::solve_parametric_polynomial_system::<E, _>(
                system,
                variables,
                &input_variables,
                domain,
            )? {
                ParametricSolveResult::Solved(solutions) => {
                    if !maximize_base_field_branches {
                        return Ok(solutions);
                    }
                    if best_solutions
                        .as_ref()
                        .is_none_or(|best: &Vec<_>| solutions.len() > best.len())
                    {
                        best_solutions = Some(solutions);
                    }
                }
                ParametricSolveResult::PositiveDimensional(_)
                | ParametricSolveResult::Inconsistent => {}
            }
        }

        if let Some(solutions) = best_solutions {
            return Ok(solutions);
        }

        Err(SolveError::Other(
            "Could not select input variables for the positive-dimensional polynomial system"
                .to_string(),
        ))
    }

    fn rational_exponent_parts(exponent: AtomView<'_>) -> Option<(i64, usize)> {
        let exponent = Rational::try_from(exponent).ok()?;
        let numerator = exponent.numerator().to_i64()?;
        let denominator = usize::try_from(exponent.denominator().to_i64()?).ok()?;
        Some((numerator, denominator))
    }

    fn collect_auxiliary_powers<T: AtomCore>(system: &[T]) -> Vec<AuxiliaryPower> {
        let mut seen = HashSet::default();
        let mut powers = Vec::new();

        for expression in system {
            expression.visitor(&mut |atom| {
                let AtomView::Pow(power) = atom else {
                    return true;
                };
                let (base, exponent) = power.get_base_exp();
                let Some((_, denominator)) = Self::rational_exponent_parts(exponent) else {
                    return true;
                };
                if denominator <= 1 {
                    return true;
                }

                // Use the primitive principal power base^(1/denominator) as
                // the auxiliary. Other powers, including negative ones, are
                // represented as integer powers of this generator by the
                // rational-polynomial converter.
                let exponent = Atom::num(Rational::from((1, denominator as i64)));
                let power = base.pow(exponent.clone());
                if seen.insert(power.clone()) {
                    powers.push(AuxiliaryPower {
                        variable: PolyVariable::Power(power),
                        base: base.to_owned(),
                        exponent,
                        denominator,
                    });
                }
                true
            });
        }

        powers
    }

    fn substitute_algebraic_solution(
        expression: AtomView<'_>,
        solution: &HashMap<PolyVariable, Atom>,
    ) -> Atom {
        let replacements = solution
            .iter()
            .filter_map(|(variable, value)| {
                (!matches!(variable, PolyVariable::Temporary(_)))
                    .then(|| (variable.to_atom(), value))
            })
            .collect::<Vec<_>>();

        expression.replace_map_bottom_up(
            |atom, _, out| {
                if let Some((_, value)) = replacements
                    .iter()
                    .find(|(variable, _)| variable.as_view() == atom)
                {
                    **out = (*value).clone();
                }
            },
            true,
        )
    }

    fn numerically_zero(expression: &Atom) -> Result<bool, String> {
        let norm = |decimal_precision| -> Result<f64, String> {
            let approximation = expression.to_float(decimal_precision);
            let value = Complex::<Float>::try_from(approximation.as_view())
                .map_err(|error| error.to_string())?;
            Ok(value.norm().re.to_f64().abs())
        };

        let low_precision = norm(48)?;
        let high_precision = norm(96)?;
        if high_precision == 0.0 {
            return Ok(low_precision == 0.0);
        }

        Ok(high_precision < 1e-40 && low_precision > 0.0 && high_precision < low_precision * 1e-12)
    }

    fn algebraically_zero(expression: &Atom) -> Option<bool> {
        let expression = expression.expand();
        if expression.is_zero() {
            return Some(true);
        }

        if let Ok(value) = Rational::try_from(expression.as_view()) {
            return Some(value.is_zero());
        }

        if let Ok(mut context) = AlgebraicContext::from_atom(expression.as_view())
            && let Ok(value) = context.convert_atom(expression.as_view())
        {
            return Some(context.field().is_zero(&value));
        }

        None
    }

    fn auxiliary_branch_matches(
        auxiliary: &AuxiliaryPower,
        solution: &HashMap<PolyVariable, Atom>,
        context: &mut AlgebraicContext,
    ) -> Result<bool, String> {
        let candidate = solution.get(&auxiliary.variable).ok_or_else(|| {
            format!(
                "The Gröbner solution is missing auxiliary variable {}",
                auxiliary.variable
            )
        })?;

        // The polynomial solver has already put the auxiliary and all solved
        // variables in one algebraic field. For a positive real base, select
        // the principal branch there: it is the unique positive real d-th
        // root. This avoids rebuilding a compositum from the printed root
        // expressions merely to rediscover an element already in the field.
        let auxiliary_atom = auxiliary.variable.to_atom();
        if let (Ok(candidate_value), Ok(base_value)) = (
            context.convert_atom(auxiliary_atom.as_view()),
            context.convert_atom(auxiliary.base.as_view()),
        ) {
            let field = context.field();
            let relation = field.sub(
                &field.pow(&candidate_value, auxiliary.denominator as u64),
                &base_value,
            );
            if !field.is_zero(&relation) {
                return Ok(false);
            }
            if field.is_zero(&base_value) {
                return Ok(field.is_zero(&candidate_value));
            }
            if field.is_positive_real(&base_value).unwrap_or(false) {
                // For d <= 4 the principal positive root is the only d-th
                // root with positive real part. Checking that part directly
                // avoids computing a second minimal polynomial when the
                // primitive generator of the solution field is complex.
                if auxiliary.denominator <= 4 {
                    return field.has_positive_real_part(&candidate_value);
                }
                return field.is_positive_real(&candidate_value);
            }
        }

        let base = Self::substitute_algebraic_solution(auxiliary.base.as_view(), solution);
        let expected = base.pow(auxiliary.exponent.clone());
        let difference = (candidate.clone() - expected.clone()).expand();
        if let Some(is_zero) = Self::algebraically_zero(&difference) {
            return Ok(is_zero);
        }

        Self::numerically_zero(&difference)
    }

    /// Solve a system exactly for `vars`.
    ///
    /// Linear systems are delegated to [`Self::solve_linear_system`]. Polynomial
    /// nonlinear systems over `Q` or `Q(parameters)` are first converted to a
    /// grevlex Gröbner basis, changed to lex order with FGLM, and solved by
    /// triangular back-substitution. For a positive-dimensional nonlinear
    /// system, a maximal set of requested variables is treated as input
    /// parameters and mapped to itself. As in the linear solver, viable input
    /// sets containing variables later in `vars` are preferred. Rational powers
    /// such as `sqrt(x+3)` are replaced by auxiliary polynomial variables and
    /// defining equations; solutions on non-principal power branches are
    /// removed afterwards.
    /// Denominators involving solve variables are cleared before constructing
    /// the basis, and solutions on their zero loci are rejected.
    /// Rational-power auxiliaries combined with parameters are not yet
    /// supported because selecting their analytic branch requires assumptions
    /// on the parameters. Parametric results describe the generic parameter
    /// locus. This also applies to automatically selected input variables in a
    /// positive-dimensional result: exceptional input values where
    /// denominators vanish or the Gröbner basis changes must be solved
    /// separately.
    ///
    /// Every expression in `system` is understood to equal zero. Each internal
    /// branch retains its exact values and generic nonvanishing conditions.
    pub(crate) fn solve_impl<E: PositiveExponent + 'static, T1: AtomCore, T2: AtomCore>(
        system: &[T1],
        vars: &[T2],
        domain: SolveDomain,
    ) -> Result<Vec<SolveBranch>, SolveError> {
        let variables = vars
            .iter()
            .map(|variable| variable.as_atom_view().to_owned().try_into())
            .collect::<Result<Vec<PolyVariable>, String>>()
            .map_err(SolveError::Other)?;

        let auxiliaries = Self::collect_auxiliary_powers(system);
        if auxiliaries.is_empty() {
            match Self::solve_linear_system::<E, _, _>(system, vars) {
                Ok(values) => {
                    return Ok(vec![SolveBranch::unconditional(
                        variables.into_iter().zip(values).collect(),
                    )]);
                }
                Err(SolveError::NonLinearSystem) => {}
                Err(SolveError::Other(error)) if error == "Not a polynomial" => {}
                Err(error) => return Err(error),
            }
        }

        let mut augmented_variables = variables.clone();
        for auxiliary in &auxiliaries {
            if !augmented_variables.contains(&auxiliary.variable) {
                augmented_variables.push(auxiliary.variable.clone());
            }
        }
        let variable_map = Arc::new(augmented_variables);

        let system_views = system
            .iter()
            .map(|expression| expression.as_atom_view())
            .collect::<Vec<_>>();
        let parameters = Self::get_parameters(&system_views, &variables);
        if !parameters.is_empty() {
            if !auxiliaries.is_empty() {
                return Err(SolveError::Other(
                    "Parametric solving with rational-power auxiliary variables is not supported"
                        .to_string(),
                ));
            }

            return match Self::solve_parametric_polynomial_system::<E, _>(
                system,
                &variables,
                &HashSet::default(),
                domain,
            )? {
                ParametricSolveResult::Solved(solutions) => Ok(solutions),
                ParametricSolveResult::PositiveDimensional(dimension) => {
                    Self::solve_positive_dimensional_polynomial_system::<E, _>(
                        system, &variables, dimension, domain,
                    )
                }
                ParametricSolveResult::Inconsistent => Ok(Vec::new()),
            };
        }

        let mut denominators = Vec::new();
        let mut polynomials = system
            .iter()
            .map(|expression| {
                let rational: RationalPolynomial<_, E> = expression
                    .as_atom_view()
                    .try_to_rational_polynomial_preserve_power_variables(
                        &Q,
                        &Z,
                        Some(variable_map.clone()),
                    )
                    .map_err(|error| SolveError::Other(error.to_string()))?;
                let numerator = rational
                    .numerator
                    .map_coeff(|coefficient| coefficient.into(), Q);
                let denominator = rational
                    .denominator
                    .map_coeff(|coefficient| coefficient.into(), Q);
                if !denominator.is_one() {
                    denominators.push(denominator);
                }
                Ok(numerator.reorder::<GrevLexOrder>())
            })
            .collect::<Result<Vec<MultivariatePolynomial<_, E, GrevLexOrder>>, SolveError>>()?
            .into_iter()
            .filter(|polynomial| !polynomial.is_zero())
            .collect::<Vec<_>>();
        if !auxiliaries.is_empty() {
            let prototype = MultivariatePolynomial::<_, E>::new(&Q, None, variable_map.clone());
            for auxiliary in &auxiliaries {
                let helper = prototype
                    .variable(&auxiliary.variable)
                    .map_err(SolveError::Other)?;
                let base: RationalPolynomial<_, E> = auxiliary
                    .base
                    .as_view()
                    .try_to_rational_polynomial_preserve_power_variables(
                        &Q,
                        &Z,
                        Some(variable_map.clone()),
                    )
                    .map_err(|error| SolveError::Other(error.to_string()))?;
                let base_numerator = base
                    .numerator
                    .map_coeff(|coefficient| coefficient.into(), Q);
                let base_denominator = base
                    .denominator
                    .map_coeff(|coefficient| coefficient.into(), Q);
                let relation =
                    helper.pow(auxiliary.denominator) * &base_denominator - base_numerator;
                if !base_denominator.is_one() {
                    denominators.push(base_denominator);
                }
                polynomials.push(relation.reorder::<GrevLexOrder>());
            }
        }
        let basis = GroebnerBasis::new(&polynomials, false);
        let basis = match basis.change_order::<LexOrder>() {
            Ok(basis) => basis,
            Err(_) if auxiliaries.is_empty() => {
                return match Self::solve_parametric_polynomial_system::<E, _>(
                    system,
                    &variables,
                    &HashSet::default(),
                    domain,
                )? {
                    ParametricSolveResult::Solved(solutions) => Ok(solutions),
                    ParametricSolveResult::PositiveDimensional(dimension) => {
                        Self::solve_positive_dimensional_polynomial_system::<E, _>(
                            system, &variables, dimension, domain,
                        )
                    }
                    ParametricSolveResult::Inconsistent => Ok(Vec::new()),
                };
            }
            Err(error) => return Err(SolveError::Other(error)),
        };
        let polynomial_solutions = basis
            .solve_in_base_field(auxiliaries.is_empty() && matches!(domain, Integers | Rationals))
            .map_err(SolveError::Other)?;

        if auxiliaries.is_empty() && denominators.is_empty() {
            return polynomial_solutions
                .iter()
                .map(|solution| {
                    solution
                        .to_atom_map()
                        .map(SolveBranch::unconditional)
                        .map_err(SolveError::Other)
                })
                .collect();
        }

        let mut filtered = Vec::new();
        'solutions: for polynomial_solution in polynomial_solutions {
            let solution = polynomial_solution
                .to_atom_map()
                .map_err(SolveError::Other)?;
            let mut context = AlgebraicContext::new(polynomial_solution.field().clone());
            for (variable, value) in polynomial_solution.values() {
                context.insert_image(variable.to_atom(), value.clone());
                if let Some(atom) = solution.get(variable) {
                    context.insert_image(atom.clone(), value.clone());
                }
            }

            for denominator in &denominators {
                let denominator = denominator.to_expression();
                let is_zero = context
                    .convert_atom(denominator.as_view())
                    .ok()
                    .map(|value| context.field().is_zero(&value))
                    .or_else(|| {
                        let denominator =
                            Self::substitute_algebraic_solution(denominator.as_view(), &solution);
                        Self::algebraically_zero(&denominator)
                    });
                match is_zero {
                    Some(true) => continue 'solutions,
                    Some(false) => {}
                    None => {
                        return Err(SolveError::Other(format!(
                            "Could not determine whether denominator {denominator} is zero"
                        )));
                    }
                }
            }

            for auxiliary in &auxiliaries {
                if !Self::auxiliary_branch_matches(auxiliary, &solution, &mut context)
                    .map_err(SolveError::Other)?
                {
                    continue 'solutions;
                }
            }

            filtered.push(SolveBranch::unconditional(
                variables
                    .iter()
                    .filter_map(|variable| {
                        solution
                            .get(variable)
                            .cloned()
                            .map(|value| (variable.clone(), value))
                    })
                    .collect(),
            ));
        }

        Ok(filtered)
    }

    /// Find the root of a function in `x` numerically over the reals using Newton's method.
    pub(crate) fn nsolve<N: SingleFloat + Real + EvaluationDomain + PartialOrd>(
        &self,
        x: &Indeterminate,
        init: N,
        prec: N,
        max_iterations: usize,
    ) -> Result<N, SolveError> {
        if self.has_complex_coefficients() {
            return Err(SolveError::ComplexCoefficients);
        }

        let v: Atom = x.clone().into();
        let f = self
            .evaluator(std::slice::from_ref(&v))
            .build()
            .map_err(|e| SolveError::Other(e.to_string()))?;
        let df = self
            .derivative(x)
            .evaluator(std::slice::from_ref(&v))
            .build()
            .map_err(|e| SolveError::Other(e.to_string()))?;

        let mut f_e = f.map_coeff(&|x| init.from_rational(x.to_real().unwrap()));
        let mut df_e = df.map_coeff(&|x| init.from_rational(x.to_real().unwrap()));

        let mut cur = init.clone();

        for _ in 0..max_iterations {
            let df_val = df_e.evaluate_single(std::slice::from_ref(&cur));
            let f_val = f_e.evaluate_single(std::slice::from_ref(&cur));

            if !df_val.is_finite() || df_val.is_zero() {
                return Err(SolveError::ZeroDerivative);
            }

            cur -= f_val.clone() / df_val;
            if f_val.norm() < prec {
                return Ok(cur);
            }
        }

        Err(SolveError::NoConvergence)
    }

    /// Solve a non-linear system numerically over the reals using Newton's method.
    pub(crate) fn nsolve_system<
        N: SingleFloat
            + Real
            + EvaluationDomain
            + PartialOrd
            + InternalOrdering
            + Eq
            + std::hash::Hash,
        T: AtomCore,
    >(
        system: &[T],
        vars: &[Indeterminate],
        init: &[N],
        prec: N,
        max_iterations: usize,
    ) -> Result<Vec<N>, SolveError> {
        let system = system.iter().map(|v| v.as_atom_view()).collect::<Vec<_>>();
        AtomView::nsolve_system_impl(&system, vars, init, prec, max_iterations)
    }

    fn nsolve_system_impl<
        N: SingleFloat
            + Real
            + EvaluationDomain
            + PartialOrd
            + InternalOrdering
            + Eq
            + std::hash::Hash,
    >(
        system: &[AtomView],
        vars: &[Indeterminate],
        init: &[N],
        prec: N,
        max_iterations: usize,
    ) -> Result<Vec<N>, SolveError> {
        if system.len() != vars.len() {
            Err(SolveError::NonSquareSystem)?;
        }

        if vars.len() != init.len() {
            Err(SolveError::IncompleteInitialValues)?;
        }

        if system.is_empty() {
            return Ok(vec![]);
        }

        if system.iter().any(|a| a.has_complex_coefficients()) {
            return Err(SolveError::ComplexCoefficients);
        }

        if system.len() == 1 {
            return Ok(vec![system[0].nsolve(
                &vars[0],
                init[0].clone(),
                prec,
                max_iterations,
            )?]);
        }

        let avars = vars.iter().map(|v| v.clone().into()).collect::<Vec<_>>();

        let mut fs = system
            .iter()
            .map(|a| {
                Ok(a.to_evaluation_tree(&FunctionMap::new(), &avars)
                    .map_err(|e| SolveError::Other(e.to_string()))?
                    .optimize(&OptimizationSettings {
                        horner_iterations: 1,
                        n_cores: 0,
                        cpe_iterations: None,
                        hot_start: None,
                        abort_check: None,
                        verbose: false,
                        ..Default::default()
                    })
                    .map_coeff(&|x| init[0].from_rational(x.to_real().unwrap())))
            })
            .collect::<Result<Vec<_>, SolveError>>()?;

        let mut jacobian = Vec::with_capacity(vars.len() * system.len());
        for a in system {
            let mut row = Vec::with_capacity(vars.len());
            for v in vars {
                let deriv = a.derivative(v);

                let a = deriv
                    .evaluator(&avars)
                    .build()
                    .map_err(|e| SolveError::Other(e.to_string()))?
                    .map_coeff(&|x| init[0].from_rational(x.to_real().unwrap()));

                row.push(a);
            }
            jacobian.extend_from_slice(&row);
        }

        let field = FloatField::from_rep(init[0].clone());
        let mut cur = init.to_vec();

        for _ in 0..max_iterations {
            let f = fs
                .iter_mut()
                .map(|a| a.evaluate_single(&cur))
                .collect::<Vec<_>>();
            let f = Matrix::new_vec(f, field.clone());

            let df = jacobian
                .iter_mut()
                .map(|a| a.evaluate_single(&cur))
                .collect::<Vec<_>>();

            let df = Matrix::from_linear(df, system.len() as u32, vars.len() as u32, field.clone())
                .unwrap();

            let Ok(i) = df.inv() else {
                return Err(SolveError::SingularJacobian);
            };

            let mut ci = Matrix::new_vec(cur.to_vec(), field.clone());

            ci -= &(&i * &f);

            cur = ci.into_vec();

            if f.into_iter().all(|x| x.norm() < prec) {
                return Ok(cur);
            }
        }

        Err(SolveError::NoConvergence)
    }

    /// Solve a system that is linear in `vars`, if possible.
    /// Each expression in `system` is understood to yield 0.
    pub(crate) fn solve_linear_system<E: PositiveExponent, T1: AtomCore, T2: AtomCore>(
        system: &[T1],
        vars: &[T2],
    ) -> Result<Vec<Atom>, SolveError> {
        let system: Vec<_> = system.iter().map(|v| v.as_atom_view()).collect();

        let vars: Vec<_> = vars
            .iter()
            .map(|v| v.as_atom_view().to_owned().try_into())
            .collect::<Result<Vec<_>, _>>()
            .map_err(SolveError::Other)?;

        AtomView::solve_linear_system_impl::<E>(&system, &vars)
    }

    /// Convert a system of linear equations to a matrix representation, returning the matrix
    /// and the right-hand side.
    pub(crate) fn system_to_matrix<E: PositiveExponent, T1: AtomCore, T2: AtomCore>(
        system: &[T1],
        vars: &[T2],
    ) -> Result<
        (
            Matrix<RationalPolynomialField<Z, E>>,
            Matrix<RationalPolynomialField<Z, E>>,
        ),
        SolveError,
    > {
        let system: Vec<_> = system.iter().map(|v| v.as_atom_view()).collect();

        let vars: Vec<_> = vars
            .iter()
            .map(|v| v.as_atom_view().to_owned().try_into())
            .collect::<Result<Vec<_>, _>>()?;
        let params = Self::get_parameters(&system, &vars);

        AtomView::system_to_matrix_impl::<E>(&system, &vars, params)
    }

    fn system_to_matrix_impl<E: PositiveExponent>(
        system: &[AtomView],
        vars: &[PolyVariable],
        params: HashSet<AtomView>,
    ) -> Result<
        (
            Matrix<RationalPolynomialField<Z, E>>,
            Matrix<RationalPolynomialField<Z, E>>,
        ),
        SolveError,
    > {
        let mut mat = Vec::with_capacity(system.len() * vars.len());
        let mut row = vec![RationalPolynomial::<_, E>::new(&Z, Arc::new(vec![])); vars.len()];
        let mut rhs = vec![RationalPolynomial::<_, E>::new(&Z, Arc::new(vec![])); system.len()];

        let params = Arc::new(
            params
                .iter()
                .map(|x| x.to_owned().try_into())
                .collect::<Result<Vec<_>, String>>()
                .map_err(SolveError::Other)?,
        );

        for (si, a) in system.iter().enumerate() {
            let rat: RationalPolynomial<Z, E> = a
                .try_to_rational_polynomial(&Q, &Z, None)
                .map_err(|e| SolveError::Other(e.to_string()))?;

            let poly = rat
                .to_polynomial(vars, true)
                .map_err(|e| SolveError::Other(e.to_owned()))?;

            for e in &mut row {
                *e = RationalPolynomial::<_, E>::new(&Z, params.clone());
            }

            // get linear coefficients
            'next_monomial: for e in poly.into_iter() {
                if e.exponents.iter().cloned().sum::<E>() > E::one() {
                    Err(SolveError::NonLinearSystem)?;
                }

                for (rv, p) in row.iter_mut().zip(e.exponents) {
                    if !p.is_zero() {
                        *rv = e.coefficient.clone();
                        continue 'next_monomial;
                    }
                }

                // constant term
                rhs[si] = e.coefficient.clone().neg();
            }

            mat.extend_from_slice(&row);
        }

        let Some((first, rest)) = mat.split_first_mut() else {
            return Err(SolveError::EmptySystem);
        };

        for _ in 0..2 {
            for x in &mut *rest {
                first.unify_variables(x);
            }
            for x in &mut rhs {
                first.unify_variables(x);
            }
        }

        let field = RationalPolynomialField::new(Z);

        let m = Matrix::from_linear(mat, system.len() as u32, vars.len() as u32, field.clone())
            .unwrap();
        let b = Matrix::new_vec(rhs, field);

        Ok((m, b))
    }

    /// Get all parameters in the system that are not free variables.
    fn get_parameters<'a>(system: &[AtomView<'a>], vars: &[PolyVariable]) -> HashSet<AtomView<'a>> {
        let mut all_params = HashSet::default();
        for s in system {
            all_params.extend(s.get_all_indeterminates(false));
        }

        let v: Vec<_> = vars.iter().map(|x| x.to_atom()).collect();
        let mut all_vars = HashSet::default();
        for x in &v {
            all_vars.insert(x.as_view());
        }

        all_params
            .into_iter()
            .filter(|x| !all_vars.contains(x))
            .collect()
    }

    fn solve_linear_system_without_parameters<T: Field + ConvertToRing>(
        system: &[AtomView],
        vars: &[PolyVariable],
        field: T,
    ) -> Result<Vec<Atom>, SolveError>
    where
        T::Element: Into<Coefficient>,
    {
        let mut mat = vec![field.zero(); system.len() * vars.len()];
        let mut rhs = vec![field.zero(); system.len()];

        let vars = Arc::new(vars.to_vec());
        for (row, s) in system.iter().enumerate() {
            let poly = s
                .try_to_polynomial::<_, u8>(&field, Some(vars.clone()))
                .map_err(|e| SolveError::Other(e.to_string()))?;

            for e in &poly {
                if e.exponents.iter().copied().sum::<u8>() > 1 {
                    return Err(SolveError::NonLinearSystem);
                }

                let mut found = false;
                for j in 0..vars.len() {
                    if e.exponents[j] != 0 {
                        if found {
                            return Err(SolveError::Other("Not a linear system".to_owned()));
                        }
                        mat[row * vars.len() + j] = e.coefficient.clone();
                        found = true;
                    }
                }

                if !found {
                    rhs[row] = field.neg(e.coefficient);
                }
            }
        }

        let m = Matrix::from_linear(mat, system.len() as u32, vars.len() as u32, field.clone())
            .map_err(SolveError::Other)?;
        let rhs = Matrix::new_vec(rhs, field.clone());

        match m.solve(&rhs) {
            Ok(sol) => Ok(sol.into_vec().into_iter().map(Atom::num).collect()),
            Err(MatrixError::Underdetermined {
                rank,
                row_reduced_augmented_matrix,
            }) => {
                let mut sols = Vec::with_capacity(vars.len());

                let mut var_index = 0;
                for r in row_reduced_augmented_matrix.row_iter() {
                    while var_index < vars.len() as u32 && field.is_zero(&r[var_index as usize]) {
                        sols.push(vars[var_index as usize].to_atom());
                        var_index += 1;
                    }

                    if var_index >= vars.len() as u32 {
                        break;
                    }

                    if field.is_one(&r[var_index as usize]) {
                        let mut sol = Atom::num(r.last().unwrap().clone());

                        for (var, coeff) in vars.iter().zip(r).skip((var_index + 1) as usize) {
                            if !field.is_zero(coeff) {
                                sol -= Atom::num(coeff.clone()) * var.to_atom();
                            }
                        }

                        sols.push(sol);
                        var_index += 1;
                    }
                }

                for i in var_index as usize..vars.len() {
                    sols.push(vars[i].to_atom());
                }

                Err(SolveError::Underdetermined {
                    rank,
                    partial_solution: sols,
                })
            }
            Err(e) => Err(SolveError::Other(format!("Could not solve {e:?}"))),
        }
    }

    fn solve_linear_system_impl<E: PositiveExponent>(
        system: &[AtomView],
        vars: &[PolyVariable],
    ) -> Result<Vec<Atom>, SolveError> {
        let params = Self::get_parameters(system, vars);
        if params.is_empty() {
            if system.iter().any(|a| a.has_complex_coefficients()) {
                let f: FloatField<Complex<Rational>> = FloatField::from_rep(Complex::new_zero());
                return Self::solve_linear_system_without_parameters(system, vars, f);
            } else {
                return Self::solve_linear_system_without_parameters::<Q>(system, vars, Q);
            }
        }

        let (m, b) = Self::system_to_matrix_impl::<E>(system, vars, params)?;

        match m.solve(&b) {
            Ok(sol) => Ok(sol
                .into_vec()
                .into_iter()
                .map(|s| s.to_expression())
                .collect()),
            Err(MatrixError::Underdetermined {
                rank,
                row_reduced_augmented_matrix,
            }) => {
                let mut sols = Vec::with_capacity(vars.len());

                let mut var_index = 0;
                for r in row_reduced_augmented_matrix.row_iter() {
                    while var_index < vars.len() as u32 && r[var_index as usize].is_zero() {
                        sols.push(vars[var_index as usize].to_atom());
                        var_index += 1;
                    }

                    if var_index >= vars.len() as u32 {
                        break;
                    }

                    if r[var_index as usize].is_one() {
                        let mut sol = r.last().unwrap().to_expression();

                        for (var, coeff) in vars.iter().zip(r).skip((var_index + 1) as usize) {
                            if !coeff.is_zero() {
                                sol -= coeff.to_expression() * var.to_atom();
                            }
                        }

                        sols.push(sol);
                        var_index += 1;
                    }
                }

                for i in var_index as usize..vars.len() {
                    sols.push(vars[i].to_atom());
                }

                Err(SolveError::Underdetermined {
                    rank,
                    partial_solution: sols,
                })
            }
            Err(e) => Err(SolveError::Other(format!("Could not solve {e:?}"))),
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::{
        atom::{Atom, AtomCore, AtomView, representation::InlineVar},
        domains::{
            Ring,
            algebraic_number::AlgebraicContext,
            float::{Complex, F64, Real},
            integer::Z,
            rational::Q,
            rational_polynomial::{RationalPolynomial, RationalPolynomialField},
        },
        parse,
        poly::PolyVariable,
        solve::{Complexes, Integers, Rationals, Reals, Solution, SolutionCondition, SolveError},
        symbol,
        tensors::matrix::Matrix,
        transcendental::root,
    };

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
    fn exact_solve_dispatches_linear_systems() {
        let x = symbol!("x");
        let y = symbol!("y");
        let system = [parse!("x+y-3"), parse!("x-y-1")];
        let variables = [Atom::var(x), Atom::var(y)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();

        assert_eq!(solutions.len(), 1);
        assert_eq!(
            solutions[0].get(&PolyVariable::from(x)),
            Some(&Atom::num(2))
        );
        assert_eq!(
            solutions[0].get(&PolyVariable::from(y)),
            Some(&Atom::num(1))
        );
        assert!(!solutions[0].is_underdetermined());
        assert_eq!(solutions[0].rank(), 2);
        assert_eq!(solutions[0].dimension(), 0);
    }

    #[test]
    fn exact_solve_dispatches_nonlinear_polynomial_systems() {
        let x = symbol!("x");
        let y = symbol!("y");
        let system = [parse!("x+y"), parse!("y^2-2")];
        let variables = [Atom::var(x), Atom::var(y)];

        assert_eq!(
            AtomView::solve_linear_system::<u16, _, Atom>(&system, &variables),
            Err(SolveError::NonLinearSystem)
        );

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();
        assert_eq!(solutions.len(), 2);
        for solution in solutions {
            let x_value = solution.get(&PolyVariable::from(x)).unwrap();
            let y_value = solution.get(&PolyVariable::from(y)).unwrap();
            assert_eq!(x_value, &-y_value.clone());
            assert_eq!(
                (y_value.clone().pow(Atom::num(2)) - Atom::num(2)).expand(),
                Atom::Zero
            );
        }
    }

    #[test]
    fn solve_builder_filters_by_domain() {
        let x = symbol!("x");
        let variable = Atom::var(x);

        let real_solutions = Atom::solve(&[parse!("x^2-2")])
            .over(Reals)
            .wrt(std::slice::from_ref(&variable))
            .unwrap();
        assert_eq!(real_solutions.len(), 2);
        assert!(
            real_solutions
                .iter()
                .all(|solution| !solution.is_indeterminate())
        );

        let integer_solutions = Atom::solve(&[parse!("x^2-2")])
            .over(Integers)
            .wrt(std::slice::from_ref(&variable))
            .unwrap();
        assert!(integer_solutions.is_empty());

        let rational_solutions = Atom::solve(&[parse!("(x-1/2)*(x^2-2)")])
            .over(Rationals)
            .wrt(std::slice::from_ref(&variable))
            .unwrap();
        assert_eq!(rational_solutions.len(), 1);
        assert_eq!(
            rational_solutions[0].get(&PolyVariable::from(x)),
            Some(&parse!("1/2"))
        );

        let integer_solutions = Atom::solve(&[parse!("(x-1/2)*(x^2-2)")])
            .over(Integers)
            .wrt(std::slice::from_ref(&variable))
            .unwrap();
        assert!(integer_solutions.is_empty());

        let rational_radical_solution = Atom::solve(&[parse!("sqrt(x)-sqrt(2)")])
            .over(Rationals)
            .wrt(std::slice::from_ref(&variable))
            .unwrap();
        assert_eq!(rational_radical_solution.len(), 1);
        assert_eq!(
            rational_radical_solution[0].get(&PolyVariable::from(x)),
            Some(&Atom::num(2))
        );

        let nonreal_solutions = Atom::solve(&[parse!("x^2+1")])
            .over(Reals)
            .wrt(std::slice::from_ref(&variable))
            .unwrap();
        assert!(nonreal_solutions.is_empty());

        let complex_solutions = Atom::solve(&[parse!("x^2+1")])
            .over(Complexes)
            .wrt(std::slice::from_ref(&variable))
            .unwrap();
        assert_eq!(complex_solutions.len(), 2);
    }

    #[test]
    fn solution_reports_indeterminate_branches() {
        let x = symbol!("x");
        let y = symbol!("y");
        let variables = [Atom::var(x), Atom::var(y)];

        let solutions = Atom::solve(&[parse!("x+y-1")])
            .over(Complexes)
            .wrt(&variables)
            .unwrap();

        assert_eq!(solutions.len(), 1);
        assert!(solutions[0].is_indeterminate());
        assert!(solutions[0].is_parametric());
        assert!(solutions[0].is_underdetermined());
        assert_eq!(solutions[0].rank(), 1);
        assert_eq!(solutions[0].dimension(), 1);
        assert_eq!(solutions[0].free_variables().len(), 1);
        assert!(solutions[0].conditions().is_empty());
        assert_eq!(solutions[0].domain(), Complexes);
    }

    #[test]
    fn solution_reports_unresolved_domain_membership() {
        let x = symbol!("x");
        let variable = Atom::var(x);
        let solutions = Atom::solve(&[parse!("x-a")])
            .over(Reals)
            .wrt(std::slice::from_ref(&variable))
            .unwrap();

        assert_eq!(solutions.len(), 1);
        assert!(solutions[0].is_indeterminate());
        assert!(!solutions[0].is_parametric());
        assert!(solutions[0].conditions().iter().any(|condition| {
            matches!(
                condition,
                SolutionCondition::DomainMembership {
                    variable,
                    value,
                    domain: Reals,
                } if variable == &PolyVariable::from(x) && value == &parse!("a")
            )
        }));
    }

    #[test]
    fn solution_reports_nonzero_denominator_conditions() {
        let x = symbol!("x");
        let variable = Atom::var(x);

        for equation in [parse!("a*x-1"), parse!("(x-1)/a")] {
            let solutions = Atom::solve(&[equation])
                .wrt(std::slice::from_ref(&variable))
                .unwrap();
            assert_eq!(solutions.len(), 1);
            assert!(solutions[0].is_conditional());
            assert!(solutions[0].conditions().iter().any(|condition| {
                matches!(condition, SolutionCondition::NonZero(value) if value == &parse!("a"))
            }));
        }
    }

    #[test]
    fn rational_positive_dimensional_system_keeps_all_variables_mapped() {
        let x = symbol!("x");
        let y = symbol!("y");
        let z = symbol!("z");
        let variables = [Atom::var(x), Atom::var(y), Atom::var(z)];

        let solutions = Atom::solve(&[parse!("(x-z)*(x^2-3*x+y)"), parse!("z+y-3")])
            .over(Rationals)
            .wrt(&variables)
            .unwrap();

        assert_eq!(solutions.len(), 2);
        assert!(solutions.iter().all(Solution::is_indeterminate));
        assert!(solutions.iter().all(|solution| solution.len() == 3));
        assert!(
            solutions
                .iter()
                .all(|solution| { solution.get(&PolyVariable::from(x)) == Some(&Atom::var(x)) })
        );
        assert!(solutions.iter().any(|solution| {
            solution.get(&PolyVariable::from(y)) == Some(&parse!("3-x"))
                && solution.get(&PolyVariable::from(z)) == Some(&Atom::var(x))
        }));
        assert!(solutions.iter().any(|solution| {
            solution.get(&PolyVariable::from(y)) == Some(&parse!("-x^2+3*x"))
                && solution.get(&PolyVariable::from(z)) == Some(&parse!("x^2-3*x+3"))
        }));
    }

    #[test]
    fn exact_solve_returns_a_parametrization_for_positive_dimensional_systems() {
        let x = symbol!("x");
        let y = symbol!("y");
        let z = symbol!("z");
        let system = [parse!("x+y^2"), parse!("z-y")];
        let variables = [Atom::var(x), Atom::var(y), Atom::var(z)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();

        assert_eq!(solutions.len(), 1);
        let solution = &solutions[0];
        assert_eq!(solution.get(&PolyVariable::from(x)), Some(&parse!("-z^2")));
        assert_eq!(solution.get(&PolyVariable::from(y)), Some(&Atom::var(z)));
        assert_eq!(solution.get(&PolyVariable::from(z)), Some(&Atom::var(z)));
    }

    #[test]
    fn exact_solve_does_not_choose_a_constrained_variable_as_input() {
        let x = symbol!("x");
        let y = symbol!("y");
        let system = [parse!("y^2-1")];
        let variables = [Atom::var(x), Atom::var(y)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();

        assert_eq!(solutions.len(), 2);
        for solution in solutions {
            assert_eq!(solution.get(&PolyVariable::from(x)), Some(&Atom::var(x)));
            let y_value = solution.get(&PolyVariable::from(y)).unwrap();
            assert_eq!(
                (y_value.clone().pow(Atom::num(2)) - Atom::num(1)).expand(),
                Atom::Zero
            );
        }
    }

    #[test]
    fn exact_solve_branches_over_nonlinear_input_parameters() {
        let x = symbol!("x");
        let y = symbol!("y");
        let system = [parse!("x^2+y^2-1")];
        let variables = [Atom::var(x), Atom::var(y)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();

        assert_eq!(solutions.len(), 2);
        for solution in solutions {
            assert_eq!(solution.get(&PolyVariable::from(y)), Some(&Atom::var(y)));
            let x_value = solution
                .get(&PolyVariable::from(x))
                .unwrap()
                .replace(y)
                .with(Atom::num(0));
            assert_algebraic_zero(x_value.pow(Atom::num(2)) - Atom::num(1));
        }
    }

    #[test]
    fn exact_solve_combines_explicit_and_selected_parameters() {
        let a = symbol!("a");
        let x = symbol!("x");
        let y = symbol!("y");
        let system = [parse!("x+y^2-a")];
        let variables = [Atom::var(x), Atom::var(y)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();

        assert_eq!(solutions.len(), 1);
        let solution = &solutions[0];
        assert_eq!(solution.get(&PolyVariable::from(x)), Some(&parse!("a-y^2")));
        assert_eq!(solution.get(&PolyVariable::from(y)), Some(&Atom::var(y)));
        assert!(!solution.contains_key(&PolyVariable::from(a)));
    }

    #[test]
    fn exact_solve_cubic_over_quadratic_extension() {
        let x = symbol!("x");
        let y = symbol!("y");
        let system = [parse!("x^3+y+2"), parse!("y^2-3")];
        let variables = [Atom::var(x), Atom::var(y)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();
        assert_eq!(solutions.len(), 6);

        for solution in solutions {
            let x_value = solution.get(&PolyVariable::from(x)).unwrap();
            let y_value = solution.get(&PolyVariable::from(y)).unwrap();
            assert_algebraic_zero(
                x_value.clone().pow(Atom::num(3)) + y_value.clone() + Atom::num(2),
            );
            assert_algebraic_zero(y_value.clone().pow(Atom::num(2)) - Atom::num(3));
        }
    }

    #[test]
    fn exact_solve_cubic_with_algebraic_constant() {
        let x = symbol!("x");
        let y = symbol!("y");
        let system = [parse!("x^3+y+sqrt(2)"), parse!("y^2-3")];
        let variables = [Atom::var(x), Atom::var(y)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();
        assert_eq!(solutions.len(), 6);
        for solution in solutions {
            assert_eq!(solution.len(), 2);
            let x_value = solution.get(&PolyVariable::from(x)).unwrap();
            let y_value = solution.get(&PolyVariable::from(y)).unwrap();
            let first_residual = Complex::<f64>::try_from(
                (x_value.clone().pow(Atom::num(3)) + y_value.clone() + parse!("sqrt(2)"))
                    .to_float(16),
            )
            .unwrap();
            let second_residual = Complex::<f64>::try_from(
                (y_value.clone().pow(Atom::num(2)) - Atom::num(3)).to_float(16),
            )
            .unwrap();
            assert!(first_residual.re.hypot(first_residual.im) < 1e-12);
            assert!(second_residual.re.hypot(second_residual.im) < 1e-12);
        }
    }

    #[test]
    fn exact_solve_supports_polynomial_parameters() {
        let x = symbol!("x");
        let y = symbol!("y");
        let a = symbol!("a");
        let system = [parse!("x+y"), parse!("y^2-a")];
        let variables = [Atom::var(x), Atom::var(y)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();
        assert_eq!(solutions.len(), 2);

        for solution in solutions {
            let x_value = solution.get(&PolyVariable::from(x)).unwrap();
            let y_value = solution.get(&PolyVariable::from(y)).unwrap();
            assert_eq!((x_value.clone() + y_value).expand(), Atom::Zero);
            assert!(!solution.is_indeterminate());
            assert!(!solution.is_parametric());
            assert!(solution.is_conditional());
            assert!(
                solution
                    .conditions()
                    .iter()
                    .any(|condition| matches!(condition, SolutionCondition::NonZero(_)))
            );

            let specialized = y_value.replace(a).with(Atom::num(2));
            assert_algebraic_zero(specialized.pow(Atom::num(2)) - Atom::num(2));
        }
    }

    #[test]
    fn exact_solve_expands_parametric_binomial_cubic_roots() {
        let x = symbol!("x");
        let y = symbol!("y");
        let a = symbol!("a");
        let system = [parse!("x^3+y+1"), parse!("y^2-a")];
        let variables = [Atom::var(x), Atom::var(y)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();
        assert_eq!(solutions.len(), 6);

        for solution in solutions {
            let x_value = solution.get(&PolyVariable::from(x)).unwrap();
            let y_value = solution.get(&PolyVariable::from(y)).unwrap();
            assert!(!x_value.contains_symbol(root()));
            assert!(!y_value.contains_symbol(root()));

            let x_value = x_value.replace(a).with(Atom::num(2));
            let y_value = y_value.replace(a).with(Atom::num(2));
            assert_algebraic_zero(
                (x_value.pow(Atom::num(3)) + y_value.clone() + Atom::num(1)).expand(),
            );
            assert_algebraic_zero(y_value.pow(Atom::num(2)) - Atom::num(2));
        }
    }

    #[test]
    fn exact_solve_factors_over_rational_function_parameters() {
        let x = symbol!("x");
        let system = [parse!("x^2-a^2")];
        let variables = [Atom::var(x)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();
        assert_eq!(solutions.len(), 2);

        let values = solutions
            .iter()
            .map(|solution| solution.get(&PolyVariable::from(x)).unwrap())
            .collect::<Vec<_>>();
        assert!(values.contains(&&parse!("a")));
        assert!(values.contains(&&parse!("-a")));
    }

    #[test]
    fn exact_solve_supports_rational_function_parameters() {
        let x = symbol!("x");
        let a = symbol!("a");
        let b = symbol!("b");
        let system = [parse!("x^2-a/b")];
        let variables = [Atom::var(x)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();
        assert_eq!(solutions.len(), 2);
        for solution in solutions {
            let value = solution
                .get(&PolyVariable::from(x))
                .unwrap()
                .replace(a)
                .with(Atom::num(2))
                .replace(b)
                .with(Atom::num(1));
            assert_algebraic_zero(value.pow(Atom::num(2)) - Atom::num(2));
        }
    }

    #[test]
    fn exact_solve_clears_parametric_denominators_in_solve_variables() {
        let x = symbol!("x");
        let system = [parse!("x/(x-a)")];
        let variables = [Atom::var(x)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();

        assert_eq!(solutions.len(), 1);
        assert_eq!(
            solutions[0].get(&PolyVariable::from(x)),
            Some(&Atom::num(0))
        );
        assert_eq!(solutions[0].len(), 1);
    }

    #[test]
    fn exact_solve_polynomializes_radicals_and_filters_the_sign_branch() {
        let x = symbol!("x");
        let system = [parse!("sqrt(x+3)+x")];
        let variables = [Atom::var(x)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();

        assert_eq!(solutions.len(), 1);
        assert_eq!(solutions[0].len(), 1);
        let x_value = solutions[0].get(&PolyVariable::from(x)).unwrap();
        assert_algebraic_zero(x_value.clone().pow(Atom::num(2)) - x_value.clone() - Atom::num(3));
        assert_algebraic_zero((x_value.clone() + Atom::num(3)).sqrt() + x_value.clone());
    }

    #[test]
    fn exact_solve_rejects_a_nonprincipal_square_root_branch() {
        let x = symbol!("x");
        let system = [parse!("sqrt(x)+1")];
        let variables = [Atom::var(x)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();
        assert!(solutions.is_empty());
    }

    #[test]
    fn exact_solve_supports_rational_radical_equations() {
        let x = symbol!("x");
        let system = [parse!("1/x+1/sqrt(x)-1")];
        let variables = [Atom::var(x)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();

        assert_eq!(solutions.len(), 1);
        let x_value = solutions[0].get(&PolyVariable::from(x)).unwrap();
        assert_algebraic_zero(x_value.clone() - parse!("(3+sqrt(5))/2"));
        assert_algebraic_zero(
            x_value.clone().pow(Atom::num(-1)) + x_value.clone().pow(parse!("-1/2")) - Atom::num(1),
        );
    }

    #[test]
    fn exact_solve_rejects_a_zero_of_a_cleared_denominator() {
        let x = symbol!("x");
        let system = [parse!("(sqrt(x)-1)/(x-1)")];
        let variables = [Atom::var(x)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();

        assert!(solutions.is_empty());
    }

    #[test]
    fn exact_solve_polynomializes_nested_radicals() {
        let x = symbol!("x");
        let system = [parse!("sqrt(sqrt(x)+1)-2")];
        let variables = [Atom::var(x)];

        let solutions = Atom::solve(&system).wrt(&variables).unwrap();

        assert_eq!(solutions.len(), 1);
        assert_eq!(
            solutions[0].get(&PolyVariable::from(x)),
            Some(&Atom::num(9))
        );
    }

    #[test]
    fn underdetermined() {
        let v0 = symbol!("v0").into();
        let v1 = symbol!("v1").into();
        let v2 = symbol!("v2").into();
        let v3 = symbol!("v3").into();
        let v4 = symbol!("v4").into();
        let eqs = ["v1 + v2 - 3", "2*v1 + 2*v2 - 6", "v1 + v3 - 5"];

        let system: Vec<_> = eqs.iter().map(|e| parse!(e)).collect();
        let vars = [v0, v1, v2, v3, v4];

        let sol = AtomView::solve_linear_system::<u8, _, InlineVar>(&system, &vars);

        assert_eq!(
            sol,
            Err(SolveError::Underdetermined {
                rank: 2,
                partial_solution: vec![
                    parse!("v0"),
                    parse!("-v3+5"),
                    parse!("v3-2"),
                    parse!("v3"),
                    parse!("v4"),
                ],
            })
        );
    }

    #[test]
    fn solve() {
        let x = symbol!("v1").into();
        let y = symbol!("v2").into();
        let z = symbol!("v3").into();
        let eqs = [
            "v4*v1 + f1(v4)*v2 + v3 - 1",
            "v1 + v4*v2 + v3/v4 - 2",
            "(v4-1)v1 + v4*v3",
        ];

        let system: Vec<_> = eqs.iter().map(|e| parse!(e)).collect();

        let sol = AtomView::solve_linear_system::<u8, _, InlineVar>(&system, &[x, y, z]).unwrap();

        let res = [
            "(v4^3-2*v4^2*f1(v4))*(v4^2-f1(v4)-v4^3+v4^4+v4*f1(v4)-v4^2*f1(v4))^-1",
            "(-1+2*v4)*(v4^2-f1(v4))^-1",
            "(v4^2-v4^3-2*v4*f1(v4)+2*v4^2*f1(v4))*(v4^2-f1(v4)-v4^3+v4^4+v4*f1(v4)-v4^2*f1(v4))^-1",
        ];
        let res = res.iter().map(|x| parse!(x)).collect::<Vec<_>>();

        assert_eq!(sol, res);
    }

    #[test]
    fn solve_from_matrix() {
        let system = [
            ["v4", "v4+1", "v4^2+5"],
            ["1", "v4", "v4+1"],
            ["v4-1", "-1", "v4"],
        ];
        let rhs = ["1", "2", "-1"];

        let var_map = Arc::new(vec![PolyVariable::Symbol(symbol!("v4"))]);

        let system_rat: Vec<RationalPolynomial<_, u8>> = system
            .iter()
            .flatten()
            .map(|s| parse!(s).to_rational_polynomial(&Q, &Z, Some(var_map.clone())))
            .collect();

        let rhs_rat: Vec<RationalPolynomial<_, u8>> = rhs
            .iter()
            .map(|s| parse!(s).to_rational_polynomial(&Q, &Z, Some(var_map.clone())))
            .collect();

        let field = RationalPolynomialField::from_poly(&rhs_rat[0].numerator);
        let m = Matrix::from_linear(
            system_rat,
            system.len() as u32,
            system.len() as u32,
            field.clone(),
        )
        .unwrap();
        let b = Matrix::new_vec(rhs_rat, field);

        let sol = m.solve(&b).unwrap();

        let res = [
            "(10-2*v4+4*v4^2-v4^3)/(6-4*v4+5*v4^2-3*v4^3+v4^4)",
            "(-4+10*v4-5*v4^2+2*v4^3)/(6-4*v4+5*v4^2-3*v4^3+v4^4)",
            "(2-4*v4)/(6-4*v4+5*v4^2-3*v4^3+v4^4)",
        ];

        let res = res
            .iter()
            .map(|x| parse!(x).to_rational_polynomial(&Z, &Z, m[(0, 0)].get_variables().clone()))
            .collect::<Vec<_>>();

        assert_eq!(sol.into_vec(), res);
    }

    #[test]
    fn find_root() {
        let x = symbol!("x");
        let a = parse!("x^2 - 2");
        let a = a.as_view();

        let root = a.nsolve(&x.into(), 1.0, 1e-10, 1000).unwrap();
        assert!((root - 2f64.sqrt()).abs() < 1e-10);
    }

    #[test]
    fn solve_system_newton() {
        let a = parse!("5x^2+x*y^2+sin(2y)^2 - 2");
        let b = parse!("exp(2x-y)+4y - 3");

        let r = AtomView::nsolve_system(
            &[a.as_view(), b.as_view()],
            &[symbol!("x").into(), symbol!("y").into()],
            &[F64::from(1.), F64::from(1.)],
            F64::from(1e-10),
            100,
        )
        .unwrap();

        assert!((r[0] - F64::from(5.672_973_499_396_123e-1)).norm() < 1e-10.into());
        assert!((r[1] - F64::from(-3.0944227920271083e-1)).norm() < 1e-10.into());
    }
}
