//! Exact equation solutions and their validity conditions.
use super::*;

/// A restriction under which a solution branch is valid.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolutionCondition {
    /// The expression must equal zero.
    Zero(Atom),
    /// The expression must be nonzero.
    NonZero(Atom),
    /// The expression must be strictly positive.
    Positive(Atom),
    /// The assigned expression must belong to the specified domain.
    DomainMembership {
        /// The unknown whose assignment is restricted.
        variable: PolyVariable,
        /// The assigned expression, or the variable itself when it is free.
        value: Atom,
        /// The domain that must contain the value.
        domain: SolveDomain,
    },
}
impl SolutionCondition {
    pub(crate) fn format_with(&self, atom: &impl Fn(&Atom) -> String) -> String {
        match self {
            Self::Zero(a) => format!("{} = 0", atom(a)),
            Self::NonZero(a) => format!("{} != 0", atom(a)),
            Self::Positive(a) => format!("{} > 0", atom(a)),
            Self::DomainMembership { value, domain, .. } => {
                format!("{} in {domain:?}", atom(value))
            }
        }
    }
}
impl std::fmt::Display for SolutionCondition {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.format_with(&ToString::to_string))
    }
}

/// Assignments in the requested variable order, with free coordinates and validity conditions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Solution {
    coordinates: Vec<(PolyVariable, Atom)>,
    variables: Vec<PolyVariable>,
    free_variables: Vec<PolyVariable>,
    conditions: Vec<SolutionCondition>,
    domain: SolveDomain,
    dimension: Option<usize>,
}
impl Solution {
    /// Apply the unknown symbols' domain and positivity restrictions to a solved
    /// branch. Reject proven violations and retain undecidable restrictions as
    /// branch conditions.
    fn restrict_symbol_domains(&mut self) -> bool {
        for coordinate in &self.variables {
            let variable = coordinate.to_atom();
            let assigned = self.get(coordinate);
            let free = assigned.is_none();
            let value = assigned.unwrap_or(&variable).clone();
            let restriction = if variable.is_integer().is_true() && self.domain != Integers {
                Some(Integers)
            } else if variable.is_real().is_true() && self.domain == Complexes {
                Some(Reals)
            } else {
                None
            };
            if let Some(domain) = restriction {
                match if free {
                    DomainMembership::Indeterminate
                } else {
                    value_in_domain(&value, domain)
                } {
                    DomainMembership::No => return false,
                    DomainMembership::Yes => {}
                    DomainMembership::Indeterminate => {
                        self.conditions.push(SolutionCondition::DomainMembership {
                            variable: coordinate.clone(),
                            value: value.clone(),
                            domain,
                        });
                        if !self.free_variables.is_empty() {
                            self.dimension = None;
                        }
                    }
                }
            }
            if variable.is_positive().is_true() {
                let positive = if free {
                    None
                } else {
                    Option::<bool>::from(value.is_positive()).or_else(|| {
                        AlgebraicContext::from_atom(value.as_view())
                            .ok()
                            .and_then(|mut context| {
                                let element = context.convert_atom(value.as_view()).ok()?;
                                Some(
                                    context.field().try_sign(&element).ok()?
                                        == std::cmp::Ordering::Greater,
                                )
                            })
                    })
                };
                match positive {
                    Some(false) => return false,
                    Some(true) => {}
                    None => {
                        self.conditions
                            .push(SolutionCondition::Positive(value.clone()));
                        if !self.free_variables.is_empty() {
                            self.dimension = None;
                        }
                    }
                }
            }
        }
        true
    }

    fn new(
        values: HashMap<PolyVariable, Atom>,
        variables: &[PolyVariable],
        free_variables: Vec<PolyVariable>,
        conditions: Vec<SolutionCondition>,
        domain: SolveDomain,
        certified: bool,
    ) -> Result<Self, SolveError> {
        let mut pending = variables.to_vec();
        let mut coordinates = Vec::new();
        while !pending.is_empty() {
            let index = pending
                .iter()
                .position(|v| {
                    free_variables.contains(v)
                        || values.get(v).is_some_and(|a| {
                            !pending
                                .iter()
                                .any(|dependency| a.contains(dependency.to_atom().as_view()))
                        })
                })
                .ok_or_else(|| {
                    SolveError::UnsupportedProblem("Cyclic coordinate dependencies".into())
                })?;
            let variable = pending.remove(index);
            if !free_variables.contains(&variable) {
                let value = values[&variable].clone();
                validate_exact_expression(&value)?;
                if !value.is_finite() {
                    return Err(SolveError::InvalidInput(
                        "Solution values must be finite".into(),
                    ));
                }
                coordinates.push((variable, value));
            }
        }
        coordinates.sort_by_key(|(v, _)| variables.iter().position(|a| a == v).unwrap());
        let dimension = certified.then_some(free_variables.len());
        Ok(Self {
            coordinates,
            variables: variables.to_vec(),
            free_variables,
            conditions,
            domain,
            dimension,
        })
    }
    /// Assigned variables and their expressions, in the requested order.
    pub fn coordinates(&self) -> &[(PolyVariable, Atom)] {
        &self.coordinates
    }
    /// All requested variables, including free variables.
    pub fn variables(&self) -> &[PolyVariable] {
        &self.variables
    }
    /// Iterate over all requested variables in their original order, including free ones.
    pub fn coordinate_order(&self) -> impl Iterator<Item = &PolyVariable> {
        self.variables.iter()
    }
    /// Get an assignment; free and absent variables return None.
    pub fn get(&self, variable: &PolyVariable) -> Option<&Atom> {
        self.coordinates
            .iter()
            .find(|(v, _)| v == variable)
            .map(|(_, value)| value)
    }
    /// Return whether this branch assigns an expression to `variable`.
    /// Free variables are not assignment keys.
    pub fn contains_key(&self, variable: &PolyVariable) -> bool {
        self.get(variable).is_some()
    }
    /// Number of assignments, excluding free variables.
    pub fn len(&self) -> usize {
        self.coordinates.len()
    }
    /// Return whether the assignment mapping is empty.
    /// An empty mapping can describe an unrestricted family of solutions.
    pub fn is_empty(&self) -> bool {
        self.coordinates.is_empty()
    }
    /// Iterate over assigned variables and expressions in the requested variable order.
    pub fn iter(&self) -> std::slice::Iter<'_, (PolyVariable, Atom)> {
        self.coordinates.iter()
    }
    /// Return requested variables left unassigned by this branch.
    /// Their values must still respect the domain and branch conditions.
    pub fn free_variables(&self) -> &[PolyVariable] {
        &self.free_variables
    }
    /// Return the restrictions under which all assignments in this branch hold.
    pub fn conditions(&self) -> &[SolutionCondition] {
        &self.conditions
    }
    /// Return the domain selected for the unknowns when solving the system.
    pub fn domain(&self) -> SolveDomain {
        self.domain
    }
    /// Return whether the branch retains any validity conditions.
    pub fn is_conditional(&self) -> bool {
        !self.conditions.is_empty()
    }
    /// Fiber dimension over fixed external parameters: real dimension over
    /// Reals, complex dimension over Complexes; unavailable for uncertified guards.
    pub fn dimension(&self) -> Option<usize> {
        self.dimension
    }
    /// Return the number of requested variables minus the branch dimension.
    /// Returns `None` when the dimension cannot be established.
    pub fn codimension(&self) -> Option<usize> {
        self.dimension.map(|d| self.variables.len() - d)
    }
    /// Return whether all requested variables are assigned without branch conditions.
    pub fn is_point(&self) -> bool {
        !self.is_conditional() && self.free_variables.is_empty()
    }
    /// Copy the assignments of an unconditional point branch into a map.
    ///
    /// Returns [`SolveError::NotPoint`] if free variables or validity conditions remain.
    pub fn as_point_dict(&self) -> Result<HashMap<PolyVariable, Atom>, SolveError> {
        if !self.is_point() {
            return Err(SolveError::NotPoint);
        }
        Ok(self.coordinates.iter().cloned().collect())
    }
}
impl IntoIterator for Solution {
    type Item = (PolyVariable, Atom);
    type IntoIter = std::vec::IntoIter<(PolyVariable, Atom)>;
    fn into_iter(self) -> Self::IntoIter {
        self.coordinates.into_iter()
    }
}
impl<'a> IntoIterator for &'a Solution {
    type Item = &'a (PolyVariable, Atom);
    type IntoIter = std::slice::Iter<'a, (PolyVariable, Atom)>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl std::fmt::Display for Solution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{{}}}",
            self.coordinates
                .iter()
                .map(|(v, a)| format!("{v} = {a}"))
                .collect::<Vec<_>>()
                .join(", ")
        )?;
        if self.is_conditional() {
            write!(
                f,
                " where {}",
                self.conditions
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join(" & ")
            )?;
        }
        Ok(())
    }
}

/// The parameter values for which the represented branches cover the solutions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum SolveCoverage {
    /// All solutions in the requested domain are represented.
    #[default]
    Complete,
    /// Coverage applies where the set's coverage guard holds.
    Generic,
}
/// A union of solution branches, each describing a point or a family of points.
/// Length counts branches. Branches may overlap and appear in any order.
#[derive(Clone, Debug)]
pub struct SolutionSet {
    branches: Vec<Solution>,
    variables: Vec<PolyVariable>,
    parameters: Vec<PolyVariable>,
    domain: SolveDomain,
    coverage: SolveCoverage,
    coverage_guard: Vec<Atom>,
}
impl SolutionSet {
    /// Return the requested unknowns in the order supplied to the solver.
    pub fn variables(&self) -> &[PolyVariable] {
        &self.variables
    }
    /// Return symbols inferred from the equations that were not selected as unknowns.
    pub fn parameters(&self) -> &[PolyVariable] {
        &self.parameters
    }
    /// Return the domain selected for the unknowns.
    pub fn domain(&self) -> SolveDomain {
        self.domain
    }
    /// Return whether the branches cover the full system or only the generic parameter case.
    /// For generic coverage, consult [`Self::coverage_guard`].
    pub fn coverage(&self) -> SolveCoverage {
        self.coverage
    }
    /// Expressions that must all be nonzero for generic coverage to apply.
    pub fn coverage_guard(&self) -> &[Atom] {
        &self.coverage_guard
    }
    /// Extract the assignments of the sole point branch.
    ///
    /// For generic coverage, the caller must ensure every expression in
    /// [`Self::coverage_guard`] is nonzero at the chosen parameter values.
    pub fn as_point_dict(&self) -> Result<HashMap<PolyVariable, Atom>, SolveError> {
        if self.branches.len() != 1 {
            return Err(SolveError::NotPoint);
        }
        self.branches[0].as_point_dict()
    }
    /// Return the number of branches, not the number of individual solution points.
    pub fn len(&self) -> usize {
        self.branches.len()
    }
    /// Determine whether the solution set is empty.
    ///
    /// Returns [`SolveError::IncompleteCoverage`] for generic coverage or when
    /// every returned branch has unresolved validity conditions.
    pub fn is_empty(&self) -> Result<bool, SolveError> {
        if self.coverage != SolveCoverage::Complete {
            Err(SolveError::IncompleteCoverage(
                "Emptiness is undecidable".into(),
            ))
        } else if !self.branches.is_empty() && self.branches.iter().all(Solution::is_conditional) {
            Err(SolveError::IncompleteCoverage(
                "Emptiness depends on unresolved solution conditions".into(),
            ))
        } else {
            Ok(self.branches.is_empty())
        }
    }
    /// Maximum fiber dimension; -1 for the complete empty set, None if unknown.
    pub fn dimension(&self) -> Option<isize> {
        if self.coverage != SolveCoverage::Complete
            || self.branches.iter().any(Solution::is_conditional)
        {
            return None;
        }
        self.branches
            .iter()
            .try_fold(-1, |d, b| Some(d.max(b.dimension()? as isize)))
    }
    /// Iterate over the solution branches. Their order has no mathematical significance.
    pub fn iter(&self) -> std::slice::Iter<'_, Solution> {
        self.branches.iter()
    }
    /// Borrow a branch by its zero-based index, returning `None` when out of range.
    pub fn get(&self, index: usize) -> Option<&Solution> {
        self.branches.get(index)
    }
}
impl std::ops::Index<usize> for SolutionSet {
    type Output = Solution;
    fn index(&self, index: usize) -> &Solution {
        &self.branches[index]
    }
}
impl IntoIterator for SolutionSet {
    type Item = Solution;
    type IntoIter = std::vec::IntoIter<Solution>;
    fn into_iter(self) -> Self::IntoIter {
        self.branches.into_iter()
    }
}
impl<'a> IntoIterator for &'a SolutionSet {
    type Item = &'a Solution;
    type IntoIter = std::slice::Iter<'a, Solution>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Select the solve domain before specifying the unknowns with `wrt`.
pub struct SolveBuilder {
    equations: Vec<Atom>,
    pub(crate) denominators: Vec<Atom>,
    domain: SolveDomain,
}
impl SolveBuilder {
    pub(crate) fn new<T: AtomCore>(system: &[T]) -> Self {
        Self {
            equations: system.iter().map(|a| a.as_atom_view().to_owned()).collect(),
            denominators: Vec::new(),
            domain: Complexes,
        }
    }
    /// Select the domain of the unknowns. The default is [`Complexes`].
    pub fn over(mut self, domain: SolveDomain) -> Self {
        self.domain = domain;
        self
    }
    /// Solve for earlier unknowns first, preferentially leaving later ones free.
    /// For `x+y=1`, `[x,y]` assigns `x=1-y` and leaves `y` unrestricted.
    pub fn wrt<V: AtomCore>(&self, variables: &[V]) -> Result<SolutionSet, SolveError> {
        self.wrt_with_exponent::<u16, V>(variables)
    }
    /// Solve for `variables` using `E` to store polynomial exponents.
    ///
    /// This has the same variable-order and domain behavior as [`Self::wrt`],
    /// which uses `u16`. Choose a wider exponent type for computations that
    /// require larger polynomial exponents.
    pub fn wrt_with_exponent<E: PositiveExponent + 'static, V: AtomCore>(
        &self,
        variables: &[V],
    ) -> Result<SolutionSet, SolveError> {
        let mut set = self.solve_unrestricted::<E, V>(variables)?;
        let mut denominators = self.denominators.clone();
        let equations = &self.equations;
        denominators.extend(
            equations
                .iter()
                .filter_map(|e| rational_denominator::<E>(e.as_view())),
        );
        let guard_formulas = set.coverage_guard.clone();
        set.branches.retain_mut(|branch| {
            let values = assigned_values(branch);
            for denominator in &denominators {
                let value = AtomView::substitute_algebraic_solution(denominator.as_view(), &values);
                if !value.is_finite() {
                    return false;
                }
                let restrictions = value
                    .as_view()
                    .try_to_rational_polynomial::<_, _, E>(&Q, &Z, None)
                    .map(|r| vec![r.numerator.to_expression(), r.denominator.to_expression()])
                    .unwrap_or_else(|_| vec![value]);
                for value in restrictions {
                    match AtomView::algebraically_zero(&value) {
                        Some(true) => return false,
                        Some(false) => {}
                        None if provably_nonzero(&value) => {}
                        None => {
                            let formula = normalized_nonzero::<E>(value);
                            let condition = SolutionCondition::NonZero(formula.clone());
                            if !guard_formulas.contains(&formula)
                                && !branch.conditions.contains(&condition)
                            {
                                branch.conditions.push(condition);
                            }
                        }
                    }
                }
            }
            let mut unique = Vec::new();
            for condition in std::mem::take(&mut branch.conditions) {
                let condition = match condition {
                    SolutionCondition::NonZero(value) => {
                        SolutionCondition::NonZero(normalized_nonzero::<E>(value))
                    }
                    condition => condition,
                };
                if let SolutionCondition::NonZero(formula) = &condition
                    && guard_formulas.contains(formula)
                {
                    continue;
                }
                if !unique.contains(&condition) {
                    unique.push(condition);
                }
            }
            branch.conditions = unique;
            true
        });
        set.branches.retain_mut(Solution::restrict_symbol_domains);
        Ok(set)
    }

    fn solve_unrestricted<E: PositiveExponent + 'static, V: AtomCore>(
        &self,
        variables: &[V],
    ) -> Result<SolutionSet, SolveError> {
        let requested: Vec<Atom> = variables
            .iter()
            .map(|v| v.as_atom_view().to_owned())
            .collect();
        let convert = |vs: &[Atom]| -> Result<Vec<PolyVariable>, SolveError> {
            let mut seen = HashSet::default();
            vs.iter()
                .map(|v| {
                    if !seen.insert(v) {
                        return Err(SolveError::InvalidInput("Duplicate variable".into()));
                    }
                    v.clone().try_into().map_err(SolveError::InvalidInput)
                })
                .collect()
        };
        let polynomial_variables = convert(&requested)?;
        let system = self.equations.clone();
        let mut unique = Vec::new();
        for equation in system {
            let equation = equation.expand();
            if !equation.is_zero() && !unique.contains(&equation) {
                unique.push(equation);
            }
        }
        let system = unique;
        let equation_numerators: Vec<_> = system
            .iter()
            .map(|equation| {
                equation
                    .as_view()
                    .try_to_rational_polynomial::<_, _, E>(&Q, &Z, None)
                    .map(|r| r.numerator.to_expression())
                    .unwrap_or_else(|_| equation.clone())
            })
            .collect();
        // An unused unknown adds a free coordinate to every solution. Keep it
        // out of elimination so a finite system is not mistaken for a nonlinear
        // family that requires a more general backend.
        let unused_variables: Vec<_> = polynomial_variables
            .iter()
            .zip(&requested)
            .filter(|(_, variable)| {
                !equation_numerators
                    .iter()
                    .any(|e| e.contains(variable.as_view()))
            })
            .map(|(variable, _)| variable.clone())
            .collect();
        // Eliminate earlier requested variables first, leaving later ones free.
        let execution_variables = requested
            .iter()
            .zip(&polynomial_variables)
            .filter(|(_, variable)| !unused_variables.contains(variable))
            .map(|(variable, _)| variable.clone())
            .collect::<Vec<_>>();
        for equation in self.equations.iter().chain(&self.denominators) {
            validate_exact_expression(equation)?;
        }
        let expressions: Vec<_> = self
            .equations
            .iter()
            .chain(&self.denominators)
            .map(Atom::as_view)
            .collect();
        let mut inferred: Vec<Atom> = AtomView::get_parameters(&expressions, &polynomial_variables)
            .into_iter()
            .map(|a| a.to_owned())
            .collect();
        inferred.sort_by(|a, b| a.as_view().cmp(&b.as_view()));
        let parameters = convert(&inferred)?;
        let mut set = SolutionSet {
            branches: Vec::new(),
            variables: polynomial_variables.clone(),
            parameters,
            domain: self.domain,
            coverage: SolveCoverage::Complete,
            coverage_guard: Vec::new(),
        };
        if system
            .iter()
            .any(|e| AtomView::algebraically_zero(e) == Some(false))
        {
            return Ok(set);
        }
        if !system.is_empty()
            && let Some(result) = linear_solution_set::<E>(&system, &set)?
        {
            return Ok(result);
        }
        if let Some(result) = separated_solution_set::<E>(&system, &set)? {
            return Ok(result);
        }
        if let Some(result) = triangular_solution_set::<E>(&system, &set)? {
            return Ok(result);
        }
        if let Some(result) = univariate_coordinate_solutions::<E>(&system, &set)? {
            return Ok(result);
        }
        if (!inferred.is_empty() || execution_variables.len() > 1)
            && let Some(result) = parameterized_radical_solutions::<E>(&system, &set)?
        {
            return Ok(result);
        }
        if let Some(result) = eliminate_linear_coordinate::<E>(&system, &set)? {
            return Ok(result);
        }
        if !system.is_empty() && !inferred.is_empty() {
            if execution_variables.len() == 1
                && let Some((branches, guard)) =
                    parameterized_univariate_solutions::<E>(&system, &execution_variables[0], &set)?
            {
                set.coverage = if guard.is_empty() {
                    SolveCoverage::Complete
                } else {
                    SolveCoverage::Generic
                };
                set.coverage_guard = guard;
                set.branches = branches;
                return Ok(set);
            }
        }
        if system.is_empty() {
            set.branches.push(Solution::new(
                HashMap::default(),
                &polynomial_variables,
                polynomial_variables.clone(),
                Vec::new(),
                self.domain,
                matches!(self.domain, Reals | Complexes),
            )?);
            return Ok(set);
        }
        let system_denominators = system
            .iter()
            .filter_map(|e| rational_denominator::<E>(e.as_view()))
            .collect::<Vec<_>>();
        let mut linear = false;
        // Parameter elimination is valid away from exceptional parameter values;
        // the pivot conditions collected below delimit that coverage.
        if !inferred.is_empty() {
            set.coverage = SolveCoverage::Generic;
        }
        let require_principal_certification =
            !AtomView::collect_auxiliary_powers(&equation_numerators).is_empty();
        let raw_solutions = match AtomView::solve_impl_with_coverage::<E, _, _>(
            &equation_numerators,
            &execution_variables,
            self.domain,
            require_principal_certification,
        ) {
            Ok(solutions) => solutions,
            Err(SolveError::Underdetermined {
                partial_solution, ..
            }) => {
                linear = true;
                vec![SolveBranch::unconditional(
                    convert(&execution_variables)?
                        .into_iter()
                        .zip(partial_solution)
                        .collect(),
                )]
            }
            Err(SolveError::Inconsistent) => return Ok(set),
            Err(error) => return Err(error),
        };
        let mut guards = Vec::new();
        for branch in &raw_solutions {
            for expression in &branch.nonzero_conditions {
                if !provably_nonzero(expression) {
                    let guard = normalized_nonzero::<E>(expression.clone());
                    if !guards.contains(&guard) {
                        guards.push(guard);
                    }
                }
            }
        }
        if !guards.is_empty() {
            set.coverage = SolveCoverage::Generic;
        }
        set.coverage_guard = guards;
        for branch in raw_solutions {
            let mut values = branch.values;
            for variable in &unused_variables {
                values.insert(variable.clone(), variable.to_atom());
            }
            let free_variables: Vec<_> = polynomial_variables
                .iter()
                .filter(|v| values.get(*v) == Some(&v.to_atom()))
                .cloned()
                .collect();
            if free_variables.iter().any(|v| !unused_variables.contains(v))
                && (!linear || matches!(self.domain, Integers | Rationals))
            {
                set.coverage = SolveCoverage::Generic;
            }
            // Definedness and domain conditions filter the returned branches.
            let mut conditions = Vec::new();
            let mut excluded = false;
            for denominator in system_denominators
                .iter()
                .map(|d| AtomView::substitute_algebraic_solution(d.as_view(), &values))
                .chain(
                    values
                        .values()
                        .filter_map(|v| rational_denominator::<E>(v.as_view())),
                )
            {
                if !denominator.is_finite() {
                    excluded = true;
                    break;
                }
                match AtomView::algebraically_zero(&denominator) {
                    Some(true) => {
                        excluded = true;
                        break;
                    }
                    Some(false) => {}
                    None => conditions.push(SolutionCondition::NonZero(denominator)),
                }
            }
            if excluded {
                continue;
            }
            for variable in &polynomial_variables {
                let value = values.get(variable).ok_or_else(|| {
                    SolveError::IncompleteCoverage("Backend omitted a coordinate".into())
                })?;
                if free_variables.contains(variable) {
                    continue;
                }
                match value_in_domain(value, self.domain) {
                    DomainMembership::Yes => {}
                    DomainMembership::No => {
                        excluded = true;
                        break;
                    }
                    DomainMembership::Indeterminate
                        if linear
                            && self.domain == Reals
                            && !system
                                .iter()
                                .any(|a| a.as_view().has_complex_coefficients()) => {}
                    DomainMembership::Indeterminate => {
                        conditions.push(SolutionCondition::DomainMembership {
                            variable: variable.clone(),
                            value: value.clone(),
                            domain: self.domain,
                        })
                    }
                }
            }
            if excluded {
                continue;
            }
            let solution = Solution::new(
                values,
                &polynomial_variables,
                free_variables,
                conditions,
                self.domain,
                true,
            )?;
            set.branches.push(solution);
        }
        Ok(set)
    }
}

fn validate_exact_expression(expression: &Atom) -> Result<(), SolveError> {
    use crate::coefficient::CoefficientView;
    let mut invalid = false;
    expression.visitor(&mut |atom| {
        if let AtomView::Num(n) = atom {
            invalid |= matches!(
                n.get_coeff_view(),
                CoefficientView::Float(..)
                    | CoefficientView::FiniteField(..)
                    | CoefficientView::RationalPolynomial(..)
            );
        }
        !invalid
    });
    if invalid {
        return Err(SolveError::UnsupportedProblem("Solve requires exact characteristic-zero coefficients; convert inexact input explicitly".into()));
    }
    if expression
        .get_all_symbols(true)
        .iter()
        .any(|s| s.get_wildcard_level() > 0)
    {
        return Err(SolveError::InvalidInput(
            "Solve expressions cannot contain wildcards".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod contract_tests {
    use super::*;
    use crate::parse;
    #[test]
    fn nonradical_solutions_are_ordinary_owned_expressions() {
        let variable = parse!("x");
        let set = Atom::solve(&[parse!("x^5-x-1")])
            .over(Reals)
            .wrt(std::slice::from_ref(&variable))
            .unwrap();
        assert_eq!(set.len(), 1);
        let value = set[0]
            .get(&variable.clone().try_into().unwrap())
            .unwrap()
            .clone();
        drop(set);
        assert!(value.contains_symbol(crate::transcendental::root()));
        assert_eq!(
            AtomView::algebraically_zero(&(value.clone().pow(5) - &value - Atom::num(1))),
            Some(true)
        );
        assert!(matches!(
            Atom::solve(&[parse!("0.1*x")]).wrt(&[variable]),
            Err(SolveError::UnsupportedProblem(_))
        ));
    }
    #[test]
    fn zero_equations_distinguish_empty_set_from_empty_assignment() {
        for variables in [vec![], vec![parse!("x")]] {
            for system in [vec![], vec![Atom::Zero]] {
                let set = Atom::solve(&system).wrt(&variables).unwrap();
                assert_eq!(set.len(), 1);
                assert!(!set.is_empty().unwrap());
                assert_eq!(set[0].len(), 0);
                assert_eq!(set[0].variables().len(), variables.len());
                assert_eq!(set.dimension(), Some(variables.len() as isize));
            }
            let empty = Atom::solve(&[Atom::num(1)]).wrt(&variables).unwrap();
            assert_eq!(empty.len(), 0);
            assert!(empty.is_empty().unwrap());
            assert_eq!(empty.dimension(), Some(-1));
        }
        let truth = Atom::solve::<Atom>(&[]).wrt::<Atom>(&[]).unwrap();
        assert!(truth[0].as_point_dict().unwrap().is_empty());
        assert!(
            Atom::solve(&[parse!("1")])
                .wrt::<Atom>(&[])
                .unwrap()
                .is_empty()
                .unwrap()
        );
    }
    #[test]
    fn points_and_free_coordinates_use_the_same_lookup_protocol() {
        let xs = [parse!("x"), parse!("y")];
        let points = Atom::solve(&[parse!("x-2"), parse!("y-3")])
            .wrt(&xs)
            .unwrap();
        assert_eq!(
            points[0].as_point_dict().unwrap()[&xs[0].clone().try_into().unwrap()],
            parse!("2")
        );
        let family = Atom::solve(&[parse!("x+y-1")]).wrt(&xs).unwrap();
        assert_eq!(
            family.variables(),
            &xs.iter()
                .cloned()
                .map(|v| PolyVariable::try_from(v).unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            family[0]
                .coordinate_order()
                .map(PolyVariable::to_atom)
                .collect::<Vec<_>>(),
            xs
        );
        assert!(matches!(
            family[0].get(&xs[1].clone().try_into().unwrap()),
            None
        ));
        assert_eq!(family[0].as_point_dict(), Err(SolveError::NotPoint));
        let ordered = Atom::solve(&[parse!("x+y-1")])
            .over(Reals)
            .wrt(&[xs[1].clone(), xs[0].clone()])
            .unwrap();
        assert_eq!(
            ordered[0]
                .coordinate_order()
                .map(PolyVariable::to_atom)
                .collect::<Vec<_>>(),
            vec![xs[1].clone(), xs[0].clone()]
        );
        assert_eq!(ordered[0].dimension(), Some(1));
        assert!(matches!(
            Atom::solve(&[parse!("x+y")]).wrt(&[xs[0].clone(), xs[0].clone()]),
            Err(SolveError::InvalidInput(_))
        ));
    }
    #[test]
    fn mixed_symbolic_radicals_and_polynomials_preserve_principal_conditions() {
        let variables = [parse!("x"), parse!("y")];
        let set = Atom::solve(&[parse!("x^2-x-1"), parse!("sqrt(y)-z")])
            .wrt(&variables)
            .unwrap();
        assert_eq!(set.coverage(), SolveCoverage::Complete);
        assert_eq!(set.len(), 2);
        for branch in &set {
            let x = branch
                .get(&variables[0].clone().try_into().unwrap())
                .unwrap();
            assert_eq!((x * x - x - Atom::num(1)).expand(), Atom::Zero);
            assert_eq!(
                branch.get(&variables[1].clone().try_into().unwrap()),
                Some(&parse!("z^2"))
            );
            assert!(
                branch
                    .conditions()
                    .contains(&SolutionCondition::Zero(parse!("sqrt(z^2)-z")))
            );
        }
    }

    #[test]
    fn general_parameter_systems_return_guarded_backend_branches() {
        let variables = [parse!("x"), parse!("y")];
        let set = Atom::solve(&[parse!("x^2+y^2-a"), parse!("x*y-b")])
            .wrt(&variables)
            .unwrap();
        assert_eq!(set.coverage(), SolveCoverage::Generic);
        assert_eq!(set.len(), 4);
        let parameters = HashMap::from_iter([
            (PolyVariable::try_from(parse!("a")).unwrap(), Atom::num(5)),
            (PolyVariable::try_from(parse!("b")).unwrap(), Atom::num(2)),
        ]);
        for branch in &set {
            let values = variables
                .iter()
                .map(|v| {
                    let value = branch.get(&v.clone().try_into().unwrap()).unwrap();
                    AtomView::substitute_algebraic_solution(value.as_view(), &parameters)
                })
                .collect::<Vec<_>>();
            assert_eq!(
                AtomView::algebraically_zero(
                    &(&values[0] * &values[0] + &values[1] * &values[1] - Atom::num(5))
                ),
                Some(true)
            );
            assert_eq!(
                AtomView::algebraically_zero(&(&values[0] * &values[1] - Atom::num(2))),
                Some(true)
            );
        }
    }

    #[test]
    fn nonlinear_families_prefer_later_free_variables() {
        let variables = [parse!("x"), parse!("y")];
        let set = Atom::solve(&[parse!("x^2-y")]).wrt(&variables).unwrap();
        assert_eq!(set.len(), 2);
        assert_eq!(set.coverage(), SolveCoverage::Complete);
        for branch in &set {
            assert_eq!(
                branch.free_variables(),
                &[PolyVariable::try_from(variables[1].clone()).unwrap()]
            );
            let x = branch
                .get(&variables[0].clone().try_into().unwrap())
                .unwrap();
            assert_eq!((x * x - &variables[1]).expand(), Atom::Zero);
        }
    }

    #[test]
    fn symbolic_triangular_substitution_preserves_redundant_equations() {
        let variables = [parse!("x"), parse!("y")];
        let set = Atom::solve(&[parse!("x^2-z"), parse!("x-y"), parse!("y^2-z")])
            .wrt(&variables)
            .unwrap();
        assert_eq!(set.len(), 2);
        assert_eq!(set.coverage(), SolveCoverage::Complete);
        for branch in &set {
            let values = assigned_values(branch);
            for equation in [parse!("x^2-z"), parse!("x-y"), parse!("y^2-z")] {
                assert_eq!(
                    AtomView::substitute_algebraic_solution(equation.as_view(), &values).expand(),
                    Atom::Zero
                );
            }
        }
        assert!(
            Atom::solve(&[parse!("(sqrt(x)-z)/(x-z^2)")])
                .wrt(&variables[..1])
                .unwrap()
                .is_empty()
                .unwrap()
        );
    }

    #[test]
    fn reciprocal_equations_do_not_lose_their_nonpolynomial_terms() {
        let x = parse!("x");
        for (equation, expected) in [("1/x-2", "1/2"), ("2/x-1", "2"), ("1/x+1", "-1")] {
            let set = Atom::solve(&[parse!(equation)])
                .wrt(std::slice::from_ref(&x))
                .unwrap();
            assert_eq!(
                set.as_point_dict().unwrap()[&x.clone().try_into().unwrap()],
                parse!(expected)
            );
        }
    }

    #[test]
    fn rational_equalities_retain_poles_after_subtraction() {
        let x = parse!("x");
        let mut builder = Atom::solve(&[parse!("-x")]);
        builder.denominators.push(x.clone());
        assert!(
            builder
                .wrt(std::slice::from_ref(&x))
                .unwrap()
                .is_empty()
                .unwrap()
        );
        let mut builder = Atom::solve(&[Atom::Zero]);
        builder.denominators.push(x.clone());
        let set = builder.wrt(std::slice::from_ref(&x)).unwrap();
        assert_eq!(set.len(), 1);
        assert_eq!(
            set[0].free_variables(),
            &[PolyVariable::try_from(x.clone()).unwrap()]
        );
        assert_eq!(set[0].conditions(), &[SolutionCondition::NonZero(x)]);
    }

    #[test]
    fn redundant_linear_equations_preserve_the_generic_solution() {
        let x = parse!("x");
        for (system, expected, coverage) in [
            (
                [parse!("a*x-1"), parse!("2*a*x-2")],
                parse!("1/a"),
                SolveCoverage::Generic,
            ),
            (
                [parse!("x-a"), parse!("2*x-2*a")],
                parse!("a"),
                SolveCoverage::Complete,
            ),
        ] {
            let set = Atom::solve(&system).wrt(std::slice::from_ref(&x)).unwrap();
            assert_eq!(
                set.as_point_dict().unwrap()[&x.clone().try_into().unwrap()],
                expected
            );
            assert_eq!(set.coverage(), coverage);
        }
    }

    #[test]
    fn quadratic_collisions_remain_in_complete_solution_sets() {
        let variables = [parse!("x"), parse!("y")];
        let set = Atom::solve(&[parse!("x^2-x+z")]).wrt(&variables).unwrap();
        assert_eq!(set.coverage(), SolveCoverage::Complete);
        assert!(set.coverage_guard().is_empty());
        assert_eq!(set.len(), 2);
        let point = [(PolyVariable::try_from(parse!("z")).unwrap(), parse!("1/4"))]
            .into_iter()
            .collect();
        for branch in &set {
            let value = branch
                .get(&variables[0].clone().try_into().unwrap())
                .unwrap();
            assert_eq!(
                AtomView::substitute_algebraic_solution(value.as_view(), &point),
                parse!("1/2")
            );
        }
    }

    #[test]
    fn unused_unknowns_extend_finite_roots_without_changing_coverage() {
        let x = parse!("x");
        let y = parse!("y");
        let py: PolyVariable = y.clone().try_into().unwrap();
        for (domain, count) in [(Complexes, 3), (Reals, 1)] {
            let roots = Atom::solve(&[parse!("x^3-1")])
                .over(domain)
                .wrt(std::slice::from_ref(&x))
                .unwrap();
            for variables in [[x.clone(), y.clone()], [y.clone(), x.clone()]] {
                let set = Atom::solve(&[parse!("x^3-1")])
                    .over(domain)
                    .wrt(&variables)
                    .unwrap();
                assert_eq!(set.len(), count);
                assert_eq!(set.coverage(), SolveCoverage::Complete);
                assert_eq!(set.dimension(), Some(1));
                assert!(set.parameters().is_empty());
                for (branch, root) in set.iter().zip(roots.iter()) {
                    assert_eq!(
                        branch.get(&x.clone().try_into().unwrap()),
                        root.get(&x.clone().try_into().unwrap())
                    );
                    assert_eq!(branch.free_variables(), std::slice::from_ref(&py));
                    assert_eq!(branch.get(&py), None);
                    assert_eq!(
                        branch
                            .coordinate_order()
                            .map(PolyVariable::to_atom)
                            .collect::<Vec<_>>(),
                        variables
                    );
                }
            }
        }
    }
    #[test]
    fn constant_matrix_parameter_families_have_certified_complete_coverage() {
        let x = parse!("x");
        let y = parse!("y");
        let a = Atom::var(crate::symbol!("solve_real_a"; Real));
        for domain in [Reals, Complexes] {
            let set = Atom::solve(&[&x - &a * &a])
                .over(domain)
                .wrt(std::slice::from_ref(&x))
                .unwrap();
            assert_eq!(set.coverage(), SolveCoverage::Complete);
            assert_eq!(
                set.parameters(),
                &[PolyVariable::try_from(a.clone()).unwrap()]
            );
            assert_eq!(
                set[0].as_point_dict().unwrap()[&PolyVariable::try_from(x.clone()).unwrap()],
                &a * &a
            );
            let family = Atom::solve(&[&x + &y - &a])
                .over(domain)
                .wrt(&[x.clone(), y.clone()])
                .unwrap();
            assert_eq!(family.dimension(), Some(1));
        }
        assert_eq!(
            Atom::solve(&[parse!("a*x")])
                .wrt(std::slice::from_ref(&x))
                .unwrap()
                .coverage(),
            SolveCoverage::Generic
        );
        let unrestricted = parse!("a");
        assert!(unrestricted.is_real().is_inconclusive());
        let result = Atom::solve(&[&x - &unrestricted])
            .over(Reals)
            .wrt(std::slice::from_ref(&x))
            .unwrap();
        assert_eq!(
            result.as_point_dict().unwrap()[&PolyVariable::try_from(x).unwrap()],
            unrestricted
        );
        assert!(unrestricted.is_real().is_inconclusive());
    }
}

/// Nonnegativity alone cannot justify dropping a nonzero guard or a strict inequality.
fn provably_nonzero(expression: &Atom) -> bool {
    if let Some(zero) = AtomView::algebraically_zero(expression) {
        return !zero;
    }
    match expression.as_view() {
        AtomView::Var(v) => v.get_symbol().is_positive(),
        AtomView::Mul(m) => m.iter().all(|a| provably_nonzero(&a.to_owned())),
        AtomView::Pow(p) => {
            let (base, exponent) = p.get_base_exp();
            Rational::try_from(exponent).is_ok() && provably_nonzero(&base.to_owned())
        }
        AtomView::Add(a) => {
            a.iter().all(|t| t.is_nonnegative().is_true())
                && a.iter().any(|t| provably_nonzero(&t.to_owned()))
        }
        _ => false,
    }
}

fn normalized_nonzero<E: PositiveExponent>(expression: Atom) -> Atom {
    let normalized = expression
        .as_view()
        .try_to_polynomial::<_, E>(&Q, None)
        .ok()
        .filter(|p| !p.is_zero())
        .map(|p| p.make_monic().to_expression())
        .unwrap_or(expression);
    if let AtomView::Pow(power) = normalized.as_view() {
        let (base, exponent) = power.get_base_exp();
        if let Some((numerator, 1)) = AtomView::rational_exponent_parts(exponent)
            && numerator > 0
        {
            return normalized_nonzero::<E>(base.to_owned());
        }
    }
    normalized
}

/// Check a rational expression using the local domains of unknowns and parameters.
fn value_in_solve_context<E: PositiveExponent + 'static>(
    value: &Atom,
    set: &SolutionSet,
) -> DomainMembership {
    let membership = value_in_domain(value, set.domain);
    if membership != DomainMembership::Indeterminate {
        return membership;
    }
    let Ok(rational) = value
        .as_view()
        .try_to_rational_polynomial::<_, _, E>(&Q, &Z, None)
    else {
        return membership;
    };
    if rational
        .get_variables()
        .iter()
        .any(|v| !set.variables.contains(v) && !set.parameters.contains(v))
    {
        return membership;
    }
    if set.domain == Integers && !rational.denominator.is_one() {
        return membership;
    }
    DomainMembership::Yes
}

/// Solve rational equations whose numerators are linear. Row reduction selects
/// dependent coordinates in input order. A nonzero maximal minor proves
/// the generic rank, without exposing avoidable intermediate-pivot exclusions.
fn linear_solution_set<E: PositiveExponent + 'static>(
    system: &[Atom],
    template: &SolutionSet,
) -> Result<Option<SolutionSet>, SolveError> {
    let mut numerators = Vec::new();
    let mut denominators = Vec::new();
    for equation in system {
        let Ok(rational) = equation
            .as_view()
            .try_to_rational_polynomial::<_, _, E>(&Q, &Z, None)
        else {
            return Ok(None);
        };
        let denominator = rational.denominator.to_expression();
        if template.variables.is_empty()
            || template
                .variables
                .iter()
                .any(|v| denominator.contains(v.to_atom().as_view()))
        {
            numerators.push(rational.numerator.to_expression());
        } else {
            // Parameter-only denominators are coefficient poles, not a loss of
            // rank. Clearing them would create an artificial leading-coefficient guard.
            numerators.push(equation.clone());
        }
        if let Some(denominator) = rational_denominator::<E>(equation.as_view()) {
            denominators.push(denominator);
        }
    }
    let unknowns: Vec<_> = template
        .variables
        .iter()
        .map(PolyVariable::to_atom)
        .collect();
    let mut set = template.clone();
    let mut conditions = Vec::new();
    let mut values = HashMap::default();
    let mut assigned = HashSet::default();
    let mut consistency = Vec::new();
    let mut guards = Vec::new();
    if unknowns.is_empty() {
        consistency = numerators;
    } else {
        let field = RationalPolynomialField::new(Z);
        let polynomial_unknowns: Vec<_> = template.variables.clone();
        let mut coefficients = Vec::new();
        let mut right_hand_side = Vec::new();
        for equation in &numerators {
            let rational = equation
                .as_view()
                .try_to_rational_polynomial::<_, _, E>(&Q, &Z, None)
                .map_err(|e| SolveError::Other(e.to_string()))?;
            let Ok(polynomial) = rational.to_polynomial(&polynomial_unknowns, false) else {
                return Ok(None);
            };
            let mut row = vec![field.zero(); unknowns.len()];
            let mut rhs = field.zero();
            for term in &polynomial {
                // Conversion can introduce coefficients such as sqrt(x). They
                // still depend on x and belong to the algebraic backend.
                let coefficient = term.coefficient.to_expression();
                if unknowns.iter().any(|v| coefficient.contains(v.as_view()))
                    || term.exponents.iter().any(|e| e.to_u32() > 1)
                    || term.exponents.iter().filter(|e| !e.is_zero()).count() > 1
                {
                    return Ok(None);
                }
                if let Some(column) = term.exponents.iter().position(|e| !e.is_zero()) {
                    row[column] = field.add(&row[column], term.coefficient);
                } else {
                    rhs = field.sub(&rhs, term.coefficient);
                }
            }
            coefficients.extend(row);
            right_hand_side.push(rhs);
        }
        if let Some((first, rest)) = coefficients.split_first_mut() {
            for _ in 0..2 {
                for value in &mut *rest {
                    first.unify_variables(value);
                }
                for value in &mut right_hand_side {
                    first.unify_variables(value);
                }
            }
        }
        let matrix = Matrix::from_linear(
            coefficients,
            numerators.len() as u32,
            unknowns.len() as u32,
            field.clone(),
        )
        .map_err(SolveError::Other)?;
        let rhs = Matrix::new_vec(right_hand_side, field);
        let field = matrix.field();
        let original: Vec<_> = matrix.row_iter().map(|r| r.to_vec()).collect();
        let mut rows: Vec<_> = matrix
            .row_iter()
            .zip(rhs.row_iter())
            .map(|(r, b)| {
                let mut row = r.to_vec();
                row.push(b[0].clone());
                row
            })
            .collect();
        let mut row_ids: Vec<_> = (0..rows.len()).collect();
        let mut pivots = Vec::new();
        let mut selected_rows = Vec::new();
        let n = unknowns.len();
        for column in 0..n {
            let rank = pivots.len();
            let candidates = || (rank..rows.len()).filter(|&r| !rows[r][column].is_zero());
            let Some(pivot_row) = candidates()
                .find(|&r| provably_nonzero(&rows[r][column].numerator.to_expression()))
                .or_else(|| candidates().next())
            else {
                continue;
            };
            rows.swap(rank, pivot_row);
            row_ids.swap(rank, pivot_row);
            let pivot = rows[rank][column].clone();
            for value in &mut rows[rank][column..] {
                *value = field.div(value, &pivot);
            }
            let pivot_row = rows[rank].clone();
            for (index, row) in rows.iter_mut().enumerate() {
                if index == rank || row[column].is_zero() {
                    continue;
                }
                let factor = row[column].clone();
                for c in column..=n {
                    row[c] = field.sub(&row[c], &field.mul(&factor, &pivot_row[c]));
                }
            }
            pivots.push(column);
            selected_rows.push(row_ids[rank]);
        }
        if !pivots.is_empty() {
            let minor = Matrix::from_nested_vec(
                selected_rows
                    .iter()
                    .map(|&r| pivots.iter().map(|&c| original[r][c].clone()).collect())
                    .collect(),
                field.clone(),
            )
            .map_err(SolveError::Other)?;
            let determinant = minor.det().map_err(|e| SolveError::Other(e.to_string()))?;
            let expression = determinant.numerator.to_expression();
            if !provably_nonzero(&expression) {
                guards.push(normalized_nonzero::<E>(expression));
            }
        }
        for (r, &column) in pivots.iter().enumerate() {
            let mut value = rows[r][n].to_expression();
            for c in 0..n {
                if !pivots.contains(&c) && !rows[r][c].is_zero() {
                    value -= rows[r][c].to_expression() * &unknowns[c];
                }
            }
            let variable: PolyVariable = unknowns[column]
                .clone()
                .try_into()
                .map_err(SolveError::InvalidInput)?;
            assigned.insert(variable.clone());
            values.insert(variable, value.cancel());
        }
        consistency.extend(
            rows.iter()
                .skip(pivots.len())
                .map(|r| r[n].numerator.to_expression()),
        );
    }
    set.coverage_guard = guards.clone();
    set.coverage = if set.coverage_guard.is_empty() {
        SolveCoverage::Complete
    } else {
        SolveCoverage::Generic
    };
    for expression in consistency {
        match AtomView::algebraically_zero(&expression) {
            Some(true) => {}
            Some(false) => return Ok(Some(set)),
            None if provably_nonzero(&expression) => return Ok(Some(set)),
            None => conditions.push(SolutionCondition::Zero(expression)),
        }
    }
    for denominator in denominators {
        let substituted = AtomView::substitute_algebraic_solution(denominator.as_view(), &values);
        if !substituted.is_finite() {
            return Ok(Some(set));
        }
        let rational = substituted
            .as_view()
            .try_to_rational_polynomial::<_, _, E>(&Q, &Z, None)
            .map_err(|e| SolveError::Other(e.to_string()))?;
        for expression in [
            rational.numerator.to_expression(),
            rational.denominator.to_expression(),
        ] {
            if AtomView::algebraically_zero(&expression) == Some(true) {
                return Ok(Some(set));
            }
            if !provably_nonzero(&expression) {
                let formula = normalized_nonzero::<E>(expression);
                let condition = SolutionCondition::NonZero(formula.clone());
                if !guards.contains(&formula) && !conditions.contains(&condition) {
                    conditions.push(condition);
                }
            }
        }
    }
    for (variable, value) in &values {
        match value_in_solve_context::<E>(value, &set) {
            DomainMembership::No => return Ok(Some(set)),
            DomainMembership::Yes => {}
            DomainMembership::Indeterminate => {
                conditions.push(SolutionCondition::DomainMembership {
                    variable: variable.clone(),
                    value: value.clone(),
                    domain: set.domain,
                })
            }
        }
    }
    let free_variables: Vec<_> = set
        .variables
        .iter()
        .filter(|v| !assigned.contains(*v))
        .cloned()
        .collect();
    let certified_dimension = matches!(set.domain, Reals | Complexes) || free_variables.is_empty();
    set.branches.push(Solution::new(
        values,
        &set.variables,
        free_variables,
        conditions,
        set.domain,
        certified_dimension,
    )?);
    Ok(Some(set))
}

/// Reuse the parametric polynomial backend for a single active unknown. The
/// leading coefficient and the backend's specialization guards certify generic
/// coverage; poles of the original rational equation remain branch conditions.
fn parameterized_univariate_solutions<E: PositiveExponent + 'static>(
    system: &[Atom],
    variable: &Atom,
    set: &SolutionSet,
) -> Result<Option<(Vec<Solution>, Vec<Atom>)>, SolveError> {
    if system.len() != 1 {
        return Ok(None);
    }
    let Ok(rational) = system[0]
        .as_view()
        .try_to_rational_polynomial::<_, _, E>(&Q, &Z, None)
    else {
        return Ok(None);
    };
    let numerator = rational.numerator.to_expression();
    let denominator =
        rational_denominator::<E>(system[0].as_view()).unwrap_or_else(|| Atom::num(1));
    let variable: PolyVariable = variable
        .clone()
        .try_into()
        .map_err(SolveError::InvalidInput)?;
    let numerator_one = rational.numerator.one();
    let polynomial = RationalPolynomial {
        numerator: rational.numerator,
        denominator: numerator_one,
    }
    .to_polynomial(std::slice::from_ref(&variable), true)
    .map_err(|e| SolveError::Other(e.to_string()))?;
    if polynomial.is_constant() {
        return Ok(None);
    }
    let real_linear = set.domain == Reals
        && polynomial.degree(0) == E::one()
        && !system[0].as_view().has_complex_coefficients();
    let nonzero_condition = normalized_nonzero::<E>;
    let mut guards = Vec::new();
    let mut add_guard = |expression: Atom| {
        if !provably_nonzero(&expression) {
            let guard = nonzero_condition(expression);
            if !guards.contains(&guard) {
                guards.push(guard);
            }
        }
    };
    add_guard(polynomial.lcoeff().to_expression());
    let ParametricSolveResult::Solved(raw) = AtomView::solve_parametric_polynomial_system::<E, _>(
        &[numerator],
        std::slice::from_ref(&variable),
        &HashSet::default(),
        Complexes,
    )?
    else {
        return Ok(None);
    };
    // Linear and quadratic radical formulas remain finite at root collisions
    // whenever the leading coefficient stays nonzero. The quotient field's
    // square-free specialization guards are unnecessary for these expressions:
    // coincident branches still describe the repeated root. The same holds for
    // binomials when the backend supplies explicit radical expressions without
    // coefficient poles. Keep specialization guards for opaque root descriptors
    // and general higher-degree formulas, which can have extra singularities.
    let explicit_binomial = (&polynomial)
        .into_iter()
        .all(|t| t.exponents[0].is_zero() || t.exponents[0] == polynomial.degree(0))
        && raw.iter().all(|branch| {
            branch.values.values().all(|value| {
                !value.contains_symbol(crate::transcendental::root())
                    && rational_denominator::<E>(value.as_view())
                        .is_none_or(|d| provably_nonzero(&d))
            })
        });
    if polynomial.degree(0).to_u32() > 2 && !explicit_binomial {
        for branch in &raw {
            for guard in &branch.nonzero_conditions {
                add_guard(guard.clone());
            }
        }
    }
    let free_variables: Vec<_> = set
        .variables
        .iter()
        .filter(|v| **v != variable)
        .cloned()
        .collect();
    let mut branches = Vec::new();
    for branch in raw {
        let value = branch.values.get(&variable).ok_or_else(|| {
            SolveError::IncompleteCoverage("Backend omitted the univariate root".into())
        })?;
        let mut conditions = Vec::new();
        match value_in_domain(value, set.domain) {
            DomainMembership::No => continue,
            DomainMembership::Yes => {}
            DomainMembership::Indeterminate if real_linear => {}
            DomainMembership::Indeterminate => {
                conditions.push(SolutionCondition::DomainMembership {
                    variable: variable.clone(),
                    value: value.clone(),
                    domain: set.domain,
                })
            }
        }
        let pole = AtomView::substitute_algebraic_solution(denominator.as_view(), &branch.values);
        if !pole.is_finite() {
            continue;
        }
        let restrictions = if let Ok(rational) = pole
            .as_view()
            .try_to_rational_polynomial::<_, _, E>(&Q, &Z, None)
        {
            vec![
                rational.numerator.to_expression(),
                rational.denominator.to_expression(),
            ]
        } else {
            vec![pole]
        };
        let mut excluded = false;
        for restriction in restrictions {
            if AtomView::algebraically_zero(&restriction) == Some(true) {
                excluded = true;
                break;
            }
            if !provably_nonzero(&restriction) {
                let condition = nonzero_condition(restriction);
                if !guards.contains(&condition) {
                    conditions.push(SolutionCondition::NonZero(condition));
                }
            }
        }
        if !excluded {
            branches.push(Solution::new(
                branch.values,
                &set.variables,
                free_variables.clone(),
                conditions,
                set.domain,
                true,
            )?);
        }
    }
    Ok(Some((branches, guards)))
}

fn assigned_values(solution: &Solution) -> HashMap<PolyVariable, Atom> {
    solution.iter().cloned().collect()
}

fn solve_subsystem<E: PositiveExponent + 'static>(
    system: &[Atom],
    variables: &[PolyVariable],
    domain: SolveDomain,
) -> Result<SolutionSet, SolveError> {
    SolveBuilder::new(system)
        .over(domain)
        .wrt_with_exponent::<E, _>(
            &variables
                .iter()
                .map(PolyVariable::to_atom)
                .collect::<Vec<_>>(),
        )
}

/// Independent equation components can share external parameters without sharing
/// unknowns. Their Cartesian product is complete on the intersection of their guards.
fn separated_solution_set<E: PositiveExponent + 'static>(
    system: &[Atom],
    template: &SolutionSet,
) -> Result<Option<SolutionSet>, SolveError> {
    if system.len() < 2 {
        return Ok(None);
    }
    let mut groups: Vec<(Vec<Atom>, Vec<PolyVariable>)> = Vec::new();
    for equation in system {
        let mut variables: Vec<_> = template
            .variables
            .iter()
            .filter(|v| equation.contains(v.to_atom().as_view()))
            .cloned()
            .collect();
        let mut equations = vec![equation.clone()];
        let mut index = 0;
        while index < groups.len() {
            if groups[index].1.iter().any(|v| variables.contains(v)) {
                let (more_equations, more_variables) = groups.remove(index);
                equations.extend(more_equations);
                for v in more_variables {
                    if !variables.contains(&v) {
                        variables.push(v);
                    }
                }
                index = 0;
            } else {
                index += 1;
            }
        }
        groups.push((equations, variables));
    }
    if groups.len() < 2 {
        return Ok(None);
    }
    let used: Vec<_> = groups.iter().flat_map(|g| g.1.clone()).collect();
    let mut products = vec![(
        HashMap::default(),
        template
            .variables
            .iter()
            .filter(|v| !used.contains(v))
            .cloned()
            .collect::<Vec<_>>(),
        Vec::new(),
    )];
    let mut set = template.clone();
    let mut guards = Vec::new();
    for (equations, variables) in groups {
        let variables = template
            .variables
            .iter()
            .filter(|v| variables.contains(v))
            .cloned()
            .collect::<Vec<_>>();
        let part = solve_subsystem::<E>(&equations, &variables, template.domain)?;
        if part.coverage == SolveCoverage::Complete && part.branches.is_empty() {
            return Ok(Some(template.clone()));
        }
        if !part.coverage_guard.is_empty() {
            guards.extend(part.coverage_guard);
        }
        if part.coverage != SolveCoverage::Complete {
            set.coverage = SolveCoverage::Generic;
        }
        let mut next = Vec::new();
        for (values, free, conditions) in products {
            for branch in &part.branches {
                let mut values = values.clone();
                values.extend(assigned_values(branch));
                let mut free = free.clone();
                free.extend(branch.free_variables.clone());
                let mut conditions = conditions.clone();
                for c in &branch.conditions {
                    if !conditions.contains(c) {
                        conditions.push(c.clone());
                    }
                }
                next.push((values, free, conditions));
            }
        }
        products = next;
    }
    set.coverage_guard = guards;
    for (values, free, conditions) in products {
        let free = template
            .variables
            .iter()
            .filter(|v| free.contains(v))
            .cloned()
            .collect();
        set.branches.push(Solution::new(
            values,
            &template.variables,
            free,
            conditions,
            template.domain,
            matches!(template.domain, Reals | Complexes),
        )?);
    }
    Ok(Some(set))
}

/// Eliminate an equation affine in one unknown when division is unconditionally
/// safe. Its other coefficients may be nonlinear in the remaining unknowns.
fn eliminate_linear_coordinate<E: PositiveExponent + 'static>(
    system: &[Atom],
    template: &SolutionSet,
) -> Result<Option<SolutionSet>, SolveError> {
    if template.variables.len() < 2 {
        return Ok(None);
    }
    let active = template
        .variables
        .iter()
        .filter(|v| system.iter().any(|e| e.contains(v.to_atom().as_view())))
        .collect::<Vec<_>>();
    for (position, variable) in active.iter().copied().enumerate() {
        // In a family, eliminating a later variable too early can force an
        // earlier one to be free. Let the general backend choose a preferred
        // parameterization when the first active coordinate is not affine.
        if system.len() < active.len() && position > 0 {
            break;
        }
        for (index, equation) in system.iter().enumerate() {
            let Ok(rational) = equation
                .as_view()
                .try_to_rational_polynomial::<_, _, E>(&Q, &Z, None)
            else {
                continue;
            };
            let Ok(polynomial) = rational.to_polynomial(std::slice::from_ref(variable), false)
            else {
                continue;
            };
            if polynomial.degree(0) != E::one() {
                continue;
            }
            let mut constant = Atom::num(0);
            let mut coefficient = Atom::num(0);
            let mut dependent = false;
            for term in &polynomial {
                let value = term.coefficient.to_expression();
                if value.contains(variable.to_atom().as_view()) {
                    dependent = true;
                    break;
                }
                if term.exponents[0].is_zero() {
                    constant = value;
                } else {
                    coefficient = value;
                }
            }
            if dependent || !provably_nonzero(&coefficient) {
                continue;
            }
            let assignment = -constant / coefficient;
            let replacements = HashMap::from_iter([(variable.clone(), assignment.clone())]);
            let rest = system
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != index)
                .map(|(_, e)| {
                    AtomView::substitute_algebraic_solution(e.as_view(), &replacements).expand()
                })
                .collect::<Vec<_>>();
            let remaining = template
                .variables
                .iter()
                .filter(|v| *v != variable)
                .cloned()
                .collect::<Vec<_>>();
            let reduced = solve_subsystem::<E>(&rest, &remaining, template.domain)?;
            let mut set = template.clone();
            set.coverage = reduced.coverage;
            set.coverage_guard = reduced.coverage_guard;
            for branch in reduced.branches {
                let mut values = assigned_values(&branch);
                let value =
                    AtomView::substitute_algebraic_solution(assignment.as_view(), &values).expand();
                let mut conditions = branch.conditions;
                match value_in_solve_context::<E>(&value, template) {
                    DomainMembership::No => continue,
                    DomainMembership::Yes => {}
                    DomainMembership::Indeterminate => {
                        conditions.push(SolutionCondition::DomainMembership {
                            variable: variable.clone(),
                            value: value.clone(),
                            domain: template.domain,
                        })
                    }
                }
                values.insert(variable.clone(), value);
                set.branches.push(Solution::new(
                    values,
                    &template.variables,
                    branch.free_variables,
                    conditions,
                    template.domain,
                    matches!(template.domain, Reals | Complexes),
                )?);
            }
            return Ok(Some(set));
        }
    }
    Ok(None)
}

/// Polynomialize rational powers and solve for algebraic candidates. Check each
/// candidate against the original equations to enforce principal branches,
/// retaining undecidable checks as solution conditions.
fn parameterized_radical_solutions<E: PositiveExponent + 'static>(
    system: &[Atom],
    template: &SolutionSet,
) -> Result<Option<SolutionSet>, SolveError> {
    let auxiliaries = AtomView::collect_auxiliary_powers(system)
        .into_iter()
        .filter(|a| {
            template
                .variables
                .iter()
                .any(|v| a.base.contains(v.to_atom().as_view()))
        })
        .collect::<Vec<_>>();
    if auxiliaries.is_empty() {
        return Ok(None);
    }
    let mut helpers = Vec::new();
    let mut index = 0;
    while helpers.len() < auxiliaries.len() {
        let atom = Atom::var(crate::symbol!(format!("symbolica::solve_aux_{index}")));
        index += 1;
        if system.iter().any(|e| e.contains(atom.as_view()))
            || template.variables.iter().any(|v| v.to_atom() == atom)
        {
            continue;
        }
        helpers.push(PolyVariable::try_from(atom).map_err(SolveError::InvalidInput)?);
    }
    let polynomialize = |expression: &Atom| -> Result<Atom, SolveError> {
        let mut rational = expression
            .as_view()
            .try_to_rational_polynomial_preserve_power_variables::<_, _, E>(&Q, &Z, None)
            .map_err(|e| SolveError::Other(e.to_string()))?;
        for (auxiliary, helper) in auxiliaries.iter().zip(&helpers) {
            rational
                .numerator
                .rename_variable(&auxiliary.variable, helper);
            rational
                .denominator
                .rename_variable(&auxiliary.variable, helper);
        }
        Ok(rational.to_expression())
    };
    let mut augmented = system
        .iter()
        .map(&polynomialize)
        .collect::<Result<Vec<_>, _>>()?;
    for (auxiliary, helper) in auxiliaries.iter().zip(&helpers) {
        augmented.push(
            helper
                .to_atom()
                .pow(Atom::num(auxiliary.denominator as i64))
                - polynomialize(&auxiliary.base)?,
        );
    }
    let mut variables = helpers.clone();
    variables.extend(template.variables.clone());
    let raw = solve_subsystem::<E>(&augmented, &variables, Complexes)?;
    let mut set = template.clone();
    set.coverage = raw.coverage;
    set.coverage_guard = raw.coverage_guard;
    for branch in raw.branches {
        if branch.free_variables.iter().any(|v| helpers.contains(v)) {
            return Err(SolveError::UnsupportedProblem(
                "Cannot project a free radical auxiliary".into(),
            ));
        }
        let mut values = assigned_values(&branch);
        for helper in &helpers {
            values.remove(helper);
        }
        let mut conditions = branch.conditions;
        let mut excluded = false;
        for equation in system {
            let residual =
                AtomView::substitute_algebraic_solution(equation.as_view(), &values).expand();
            if !residual.is_finite() {
                excluded = true;
                break;
            }
            match AtomView::algebraically_zero(&residual) {
                Some(true) => {}
                Some(false) => {
                    excluded = true;
                    break;
                }
                None => {
                    let condition = SolutionCondition::Zero(residual);
                    if !conditions.contains(&condition) {
                        conditions.push(condition);
                    }
                }
            }
        }
        if excluded {
            continue;
        }
        for (variable, value) in &values {
            match value_in_solve_context::<E>(value, template) {
                DomainMembership::No => {
                    excluded = true;
                    break;
                }
                DomainMembership::Yes => {}
                DomainMembership::Indeterminate => {
                    conditions.push(SolutionCondition::DomainMembership {
                        variable: variable.clone(),
                        value: value.clone(),
                        domain: template.domain,
                    })
                }
            }
        }
        if excluded {
            continue;
        }
        let candidate = Solution::new(
            values,
            &template.variables,
            branch.free_variables,
            conditions,
            template.domain,
            matches!(template.domain, Reals | Complexes),
        )?;
        if !set.branches.contains(&candidate) {
            set.branches.push(candidate);
        }
    }
    Ok(Some(set))
}

/// A single equation defines branches over the later requested coordinates.
/// Reuse the one-unknown solver, treating those free coordinates as parameters.
fn univariate_coordinate_solutions<E: PositiveExponent + 'static>(
    system: &[Atom],
    template: &SolutionSet,
) -> Result<Option<SolutionSet>, SolveError> {
    if system.len() != 1 {
        return Ok(None);
    }
    let active = template
        .variables
        .iter()
        .filter(|v| system[0].contains(v.to_atom().as_view()))
        .collect::<Vec<_>>();
    if active.len() < 2 {
        return Ok(None);
    }
    let variable = active[0];
    let part = match solve_subsystem::<E>(system, std::slice::from_ref(variable), template.domain) {
        Ok(part) => part,
        Err(
            SolveError::UnsupportedProblem(_)
            | SolveError::IncompleteCoverage(_)
            | SolveError::Other(_),
        ) => return Ok(None),
        Err(error) => return Err(error),
    };
    let mut set = template.clone();
    set.coverage = part.coverage;
    set.coverage_guard = part.coverage_guard;
    for branch in part.branches {
        let free = template
            .variables
            .iter()
            .filter(|v| *v != variable || branch.free_variables.contains(v))
            .cloned()
            .collect();
        set.branches.push(Solution::new(
            assigned_values(&branch),
            &template.variables,
            free,
            branch.conditions,
            template.domain,
            true,
        )?);
    }
    Ok(Some(set))
}

/// Solve an equation involving one remaining unknown before substitution. This
/// preserves low-degree triangular structure instead of creating a higher-degree
/// eliminant, and lets extra equations become exact consistency conditions.
fn triangular_solution_set<E: PositiveExponent + 'static>(
    system: &[Atom],
    template: &SolutionSet,
) -> Result<Option<SolutionSet>, SolveError> {
    if system.len() < 2 {
        return Ok(None);
    }
    for (index, equation) in system.iter().enumerate() {
        let active = template
            .variables
            .iter()
            .filter(|v| equation.contains(v.to_atom().as_view()))
            .collect::<Vec<_>>();
        if active.len() != 1 {
            continue;
        }
        let variable = active[0];
        let roots = match solve_subsystem::<E>(
            std::slice::from_ref(equation),
            std::slice::from_ref(variable),
            template.domain,
        ) {
            Ok(roots) => roots,
            Err(
                SolveError::UnsupportedProblem(_)
                | SolveError::IncompleteCoverage(_)
                | SolveError::Other(_),
            ) => continue,
            Err(error) => return Err(error),
        };
        if roots.branches.iter().any(|b| !b.free_variables.is_empty()) {
            continue;
        }
        let mut set = template.clone();
        let mut guards = Vec::new();
        if !roots.coverage_guard.is_empty() {
            guards.extend(roots.coverage_guard);
        }
        set.coverage = roots.coverage;
        let remaining = template
            .variables
            .iter()
            .filter(|v| *v != variable)
            .cloned()
            .collect::<Vec<_>>();
        for root in roots.branches {
            let replacements = assigned_values(&root);
            let rest = system
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != index)
                .map(|(_, e)| {
                    AtomView::substitute_algebraic_solution(e.as_view(), &replacements).expand()
                })
                .collect::<Vec<_>>();
            let reduced = solve_subsystem::<E>(&rest, &remaining, template.domain)?;
            if reduced.coverage != SolveCoverage::Complete {
                set.coverage = SolveCoverage::Generic;
            }
            if !reduced.coverage_guard.is_empty()
                && !reduced.coverage_guard.iter().all(|a| guards.contains(a))
            {
                guards.extend(reduced.coverage_guard);
            }
            for branch in reduced.branches {
                let mut values = assigned_values(&branch);
                values.extend(replacements.clone());
                let mut conditions = branch.conditions;
                for condition in &root.conditions {
                    if !conditions.contains(condition) {
                        conditions.push(condition.clone());
                    }
                }
                set.branches.push(Solution::new(
                    values,
                    &template.variables,
                    branch.free_variables,
                    conditions,
                    template.domain,
                    matches!(template.domain, Reals | Complexes),
                )?);
            }
        }
        set.coverage_guard = guards;
        return Ok(Some(set));
    }
    Ok(None)
}
