//! Python protocols for eager exact solution sets.
use super::*;
use crate::solve::{SolutionSet, SolveCoverage, SolveError};

/// Show assignments and conditions; free variables remain in the metadata.
fn solution_text(solution: &Solution, atom: &impl Fn(&Atom) -> String) -> String {
    let assignments = solution
        .iter()
        .map(|(variable, value)| format!("{} = {}", atom(&variable.to_atom()), atom(value)))
        .collect::<Vec<_>>()
        .join(", ");
    let mut text = format!("{{{assignments}}}");
    if solution.is_conditional() {
        text.push_str(" where ");
        text.push_str(
            &solution
                .conditions()
                .iter()
                .map(|c| condition_text(c, atom))
                .collect::<Vec<_>>()
                .join(" & "),
        );
    }
    text
}

fn formatted_atom_with_color(atom: &Atom, color: ColorMode) -> String {
    atom.format_string(
        &PrintOptions::new()
            .max_line_length(Some(80))
            .multiplication_operator('·')
            .num_exp_as_superscript(true)
            .max_terms(Some(100))
            .color_mode(color),
        PrintState::new(),
    )
}

pub(super) fn formatted_atom(atom: &Atom) -> String {
    formatted_atom_with_color(atom, ColorMode::Always)
}

fn pretty_atom(atom: &Atom) -> String {
    formatted_atom_with_color(atom, ColorMode::Never)
}

fn condition_text(condition: &SolutionCondition, atom: &impl Fn(&Atom) -> String) -> String {
    condition.format_with(atom)
}

pub(super) fn inline_html(formatted: &str) -> String {
    let html = crate::printer::AnsiHtmlFormatter::new(formatted).to_string();
    html.strip_prefix("<div style=\"white-space: pre-wrap; margin: 0;\">")
        .and_then(|s| s.strip_suffix("</div>"))
        .unwrap_or(&html)
        .to_owned()
}

fn guard_text(guard: &[Atom], atom: &impl Fn(&Atom) -> String) -> String {
    guard
        .iter()
        .map(|a| format!("{} != 0", atom(a)))
        .collect::<Vec<_>>()
        .join(" & ")
}
fn guard_html(guard: &[Atom]) -> String {
    inline_html(&guard_text(guard, &formatted_atom))
}
fn guard_condition(guard: &[Atom]) -> crate::id::Condition<crate::id::Relation> {
    guard
        .iter()
        .map(|a| {
            crate::id::Condition::Yield(crate::id::Relation::Ne(
                a.clone().into(),
                Atom::num(0).into(),
            ))
        })
        .reduce(|a, b| a & b)
        .unwrap_or(crate::id::Condition::True)
}
fn relation_condition(
    condition: &SolutionCondition,
) -> Option<crate::id::Condition<crate::id::Relation>> {
    use crate::id::{Condition, Relation};
    let right = Atom::num(0).into();
    Some(Condition::Yield(match condition {
        SolutionCondition::Zero(a) => Relation::Eq(a.clone().into(), right),
        SolutionCondition::NonZero(a) => Relation::Ne(a.clone().into(), right),
        SolutionCondition::Positive(a) => Relation::Gt(a.clone().into(), right),
        SolutionCondition::DomainMembership { .. } => return None,
    }))
}

fn condition_html(condition: &SolutionCondition) -> String {
    inline_html(&condition_text(condition, &formatted_atom))
}

/// Keep equations aligned without relying on notebook table styles.
fn solution_html(
    solution: &Solution,
    conditions: &[SolutionCondition],
    index: Option<usize>,
) -> PyResult<String> {
    let assignments = solution.coordinates();
    let mut html = String::from(
        "<div style=\"display: flex; align-items: baseline; gap: 0.75em; line-height: 1.6\">",
    );
    if let Some(index) = index {
        html.push_str(&format!(
            "<span style=\"opacity: 0.65; font-size: 0.85em; flex-shrink: 0\">({index}) </span>"
        ));
    }
    html.push_str(
        "<div style=\"display: grid; grid-template-columns: max-content max-content minmax(0, 1fr); column-gap: 0.4em; align-items: baseline; min-width: 0\">",
    );
    if assignments.is_empty() {
        html.push_str("<div style=\"grid-column: 1 / -1\">{}</div>");
    }
    for coordinate in assignments {
        let variable: PythonExpression = coordinate.0.to_atom().into();
        let value: PythonExpression = coordinate.1.clone().into();
        html.push_str(&format!(
            "{}<span> = </span>{}",
            variable._repr_html_()?,
            value._repr_html_()?,
        ));
    }
    if !conditions.is_empty() {
        html.push_str(&format!(
            "<div style=\"grid-column: 1 / -1; margin-top: 0.25em\"><small>where {}</small></div>",
            conditions
                .iter()
                .map(condition_html)
                .collect::<Vec<_>>()
                .join(" &amp; "),
        ));
    }
    html.push_str("</div></div>");
    Ok(html)
}

/// Python exceptions raised by the equation solver.
pub mod errors {
    use pyo3::create_exception;
    create_exception!(
        symbolica,
        SolveError,
        pyo3::exceptions::PyValueError,
        "Base exception for invalid solve requests or failures to obtain the requested result."
    );
    create_exception!(
        symbolica,
        UnsupportedProblem,
        SolveError,
        "The equations require a solving method that is not supported."
    );
    create_exception!(
        symbolica,
        IncompleteCoverage,
        SolveError,
        "The requested conclusion cannot be established for all relevant cases."
    );
}
pub(super) fn solve_error(error: SolveError) -> PyErr {
    let message = error.to_string();
    match error {
        SolveError::UnsupportedProblem(_) => errors::UnsupportedProblem::new_err(message),
        SolveError::IncompleteCoverage(_) => errors::IncompleteCoverage::new_err(message),
        _ => errors::SolveError::new_err(message),
    }
}

/// A formula or domain restriction under which a solution branch is valid.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    frozen,
    skip_from_py_object,
    name = "SolutionCondition",
    module = "symbolica.core"
)]
#[derive(Clone)]
pub struct PythonSolutionCondition {
    condition: SolutionCondition,
}
#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl PythonSolutionCondition {
    /// The restriction type: ``"formula"`` or ``"domain_membership"``.
    #[getter]
    fn kind(&self) -> &'static str {
        match self.condition {
            SolutionCondition::DomainMembership { .. } => "domain_membership",
            _ => "formula",
        }
    }
    /// The symbolic condition, or None for a domain-membership restriction.
    #[getter]
    fn formula(&self) -> Option<PythonCondition> {
        relation_condition(&self.condition).map(Into::into)
    }
    /// The variable restricted by a domain-membership condition, otherwise None.
    #[getter]
    fn variable(&self) -> Option<PythonExpression> {
        match &self.condition {
            SolutionCondition::DomainMembership { variable, .. } => Some(variable.to_atom().into()),
            _ => None,
        }
    }
    /// The expression restricted by a domain-membership condition, otherwise None.
    #[getter]
    fn value(&self) -> Option<PythonExpression> {
        match &self.condition {
            SolutionCondition::DomainMembership { value, .. } => Some(value.clone().into()),
            _ => None,
        }
    }
    /// The required domain for a domain-membership condition, otherwise None.
    #[getter]
    fn domain(&self) -> Option<PythonSolveDomain> {
        match &self.condition {
            SolutionCondition::DomainMembership { domain, .. } => Some((*domain).into()),
            _ => None,
        }
    }
    fn __repr__(&self) -> String {
        self.condition.to_string()
    }
    fn _repr_html_(&self) -> String {
        condition_html(&self.condition)
    }
}
/// One equality-solution branch: Expression assignments and validity conditions.
///
/// Index or iterate over the mapping to read assigned variables and their values.
/// Use ``free_variables()`` for the family's free coordinates and ``variables``
/// for all requested unknowns. All assignments hold simultaneously, subject to
/// ``conditions()``, which includes the parent set's coverage guard.
///
/// Examples
/// --------
/// >>> from symbolica import Expression, S
/// >>> x, y = S("x", "y")
/// >>> branch = Expression.solve(x + y - 1, [x, y])[0]
/// >>> branch[x] == 1 - y
/// True
/// >>> dict(branch) == {x: 1-y}
/// True
/// >>> branch.free_variables() == [y]
/// True
/// >>> y in branch
/// False
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    frozen,
    skip_from_py_object,
    name = "Solution",
    module = "symbolica.core"
)]
#[derive(Clone)]
pub struct PythonSolution {
    solution: Solution,
    coverage_guard: Vec<Atom>,
}
impl From<Solution> for PythonSolution {
    fn from(solution: Solution) -> Self {
        Self {
            solution,
            coverage_guard: Vec::new(),
        }
    }
}
impl PythonSolution {
    fn text_with(&self, atom: &impl Fn(&Atom) -> String) -> String {
        let mut text = solution_text(&self.solution, atom);
        let mut conditional = self.solution.is_conditional();
        for condition in self
            .coverage_guard
            .iter()
            .map(|a| SolutionCondition::NonZero(a.clone()))
        {
            if !self.solution.conditions().contains(&condition) {
                text.push_str(if conditional { " & " } else { " where " });
                text.push_str(&condition_text(&condition, atom));
                conditional = true;
            }
        }
        text
    }
    fn from_set(solution: Solution, set: &SolutionSet) -> Self {
        Self {
            solution,
            coverage_guard: set.coverage_guard().to_vec(),
        }
    }
}
#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonSolution {
    /// Return assigned variables mapped directly to Expressions.
    ///
    /// Use ``free_variables()`` for free coordinates and ``conditions()`` for the
    /// restrictions under which these assignments are valid.
    #[gen_stub(override_return_type(type_repr = "dict[Expression, Expression]"))]
    fn as_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        for (k, v) in self.items() {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }
    /// All requested unknowns in coordinate order, including free variables.
    #[getter]
    fn variables(&self) -> Vec<PythonExpression> {
        self.solution
            .coordinate_order()
            .map(|v| v.to_atom().into())
            .collect()
    }
    /// Return the unknowns that remain free within this branch.
    fn free_variables(&self) -> Vec<PythonExpression> {
        self.solution
            .free_variables()
            .iter()
            .map(|v| v.to_atom().into())
            .collect()
    }
    /// Return additional conditions that must hold for this branch to be valid.
    fn conditions(&self) -> Vec<PythonSolutionCondition> {
        let mut conditions = self.solution.conditions().to_vec();
        for guard in self
            .coverage_guard
            .iter()
            .map(|a| SolutionCondition::NonZero(a.clone()))
        {
            if !conditions.contains(&guard) {
                conditions.push(guard);
            }
        }
        conditions
            .into_iter()
            .map(|condition| PythonSolutionCondition { condition })
            .collect()
    }
    /// The allowed domain of this branch's unknowns.
    #[getter]
    fn domain(&self) -> PythonSolveDomain {
        self.solution.domain().into()
    }
    /// Return whether this branch has additional validity conditions.
    fn is_conditional(&self) -> bool {
        self.solution.is_conditional() || !self.coverage_guard.is_empty()
    }
    /// Return whether all unknowns have assigned expressions, possibly subject to conditions.
    fn is_point(&self) -> bool {
        self.solution.free_variables().is_empty()
    }
    /// Return the number of independent coordinates in this branch, or None if unknown.
    fn dimension(&self) -> Option<usize> {
        self.solution.dimension()
    }
    /// Return the number of unknowns minus the dimension, or None if unknown.
    fn codimension(&self) -> Option<usize> {
        self.solution.codimension()
    }
    /// Return the assigned variables in coordinate order; free variables are omitted.
    fn keys(&self) -> Vec<PythonExpression> {
        self.solution
            .iter()
            .map(|(v, _)| v.to_atom().into())
            .collect()
    }
    /// Return the assigned Expressions in the same order as ``keys()``.
    fn values(&self) -> Vec<PythonExpression> {
        self.items().into_iter().map(|(_, v)| v).collect()
    }
    /// Return (variable, Expression) pairs, omitting free variables.
    fn items(&self) -> Vec<(PythonExpression, PythonExpression)> {
        self.solution
            .iter()
            .map(|(v, a)| (v.to_atom().into(), a.clone().into()))
            .collect()
    }
    /// Return an assigned Expression, or None for a free or absent variable.
    ///
    /// Parameters
    /// ----------
    /// variable: Expression
    ///     Variable whose assignment to retrieve. ``branch[variable]`` raises
    ///     KeyError when the variable is free or absent.
    fn get(&self, variable: &PythonExpression) -> PyResult<Option<PythonExpression>> {
        let variable = PolyVariable::try_from(variable.expr.clone())
            .map_err(exceptions::PyTypeError::new_err)?;
        Ok(self.solution.get(&variable).cloned().map(Into::into))
    }
    /// Retrieve an assigned Expression; raise KeyError for a free or absent variable.
    ///
    /// Parameters
    /// ----------
    /// variable: Expression
    ///     Variable whose assignment to retrieve.
    fn __getitem__(&self, variable: &PythonExpression) -> PyResult<PythonExpression> {
        self.get(variable)?
            .ok_or_else(|| exceptions::PyKeyError::new_err(variable.expr.to_string()))
    }
    /// Check whether a variable has an assignment in this branch.
    ///
    /// Parameters
    /// ----------
    /// variable: Expression
    ///     Variable to look up; free variables are not mapping keys.
    fn __contains__(&self, variable: &PythonExpression) -> PyResult<bool> {
        Ok(self.get(variable)?.is_some())
    }
    /// Count assignments, excluding free variables.
    fn __len__(&self) -> usize {
        self.solution.len()
    }
    #[gen_stub(override_return_type(type_repr = "typing.Iterator[Expression]"))]
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        self.keys().into_pyobject(py)?.try_iter()
    }
    fn __repr__(&self) -> String {
        format!("Solution({})", self.__str__())
    }
    fn __str__(&self) -> String {
        self.text_with(&ToString::to_string)
    }
    fn _repr_html_(&self) -> PyResult<String> {
        let conditions = self
            .conditions()
            .into_iter()
            .map(|c| c.condition)
            .collect::<Vec<_>>();
        solution_html(&self.solution, &conditions, None)
    }
    /// Display the assignments in IPython.
    ///
    /// Parameters
    /// ----------
    /// pretty: object
    ///     IPython's pretty printer.
    /// cycle: bool
    ///     Whether this object is already being displayed.
    fn _repr_pretty_(&self, pretty: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        pretty.call_method1(
            "text",
            (if cycle {
                "Solution(...)".into()
            } else {
                self.text_with(&pretty_atom)
            },),
        )?;
        Ok(())
    }
}
/// Exact solutions returned by ``Expression.solve``.
///
/// Iterate or index this set to get Solution branches. Each branch describes
/// an alternative: a point, or a family of points with free variables.
/// ``len(result)`` counts branches, so a set of length one can still contain
/// infinitely many points. Branch order is not guaranteed.
///
/// For a generic result, the assignments apply where ``coverage_guard`` is true.
/// Solve again after substituting parameter values outside that guard.
/// Use ``dict(result[i])`` or ``result[i].as_dict()`` to extract assignments.
/// Use ``branch.free_variables()`` and ``branch.conditions()`` to inspect a
/// family's free coordinates and validity restrictions.
///
/// Printing a set shows each branch's assignments and restrictions; notebooks
/// display them as aligned equations. A branch with an empty mapping describes
/// a family in which all requested variables are free.
///
/// Examples
/// --------
/// >>> from symbolica import Expression, S, Reals
/// >>> x = S("x")
/// >>> result = Expression.solve(x.eq(2), [x], domain=Reals)
/// >>> print(result)
/// [0] {x = 2}
/// >>> dict(result[0]) == {x: 2}
/// True
/// >>> bool(Expression.solve(x**2 + 1, [x], domain=Reals))
/// False
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    frozen,
    skip_from_py_object,
    name = "SolutionSet",
    module = "symbolica.core"
)]
#[derive(Clone)]
pub struct PythonSolutionSet {
    pub(super) set: SolutionSet,
}
impl PythonSolutionSet {
    fn text_with(&self, atom: &impl Fn(&Atom) -> String) -> String {
        let mut lines = Vec::new();
        if self.set.len() == 0 {
            lines.push(if self.set.coverage() == SolveCoverage::Complete {
                "No solutions.".into()
            } else {
                "No branches represented; emptiness is not established.".into()
            });
        } else {
            lines.extend(
                self.set.iter().enumerate().map(|(index, solution)| {
                    format!("[{}] {}", index, solution_text(solution, atom))
                }),
            );
        }
        lines.extend(self.display_context(&|guard| guard_text(guard, atom)));
        lines.join("\n")
    }
    fn display_context(&self, format_formula: &impl Fn(&[Atom]) -> String) -> Vec<String> {
        let mut context = Vec::new();
        if !self.set.coverage_guard().is_empty() {
            context.push(format!(
                "For {}; other parameter values are not covered.",
                format_formula(self.set.coverage_guard())
            ));
        }
        context
    }
}
#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonSolutionSet {
    /// The unknowns requested in the solve, in the original input order.
    #[getter]
    fn variables(&self) -> Vec<PythonExpression> {
        self.set
            .variables()
            .iter()
            .map(|v| v.to_atom().into())
            .collect()
    }
    /// Symbols treated as fixed parameters during the solve.
    #[getter]
    fn parameters(&self) -> Vec<PythonExpression> {
        self.set
            .parameters()
            .iter()
            .map(|v| v.to_atom().into())
            .collect()
    }
    /// The solve domain, also used as the default domain of external parameters.
    #[getter]
    fn domain(&self) -> PythonSolveDomain {
        self.set.domain().into()
    }
    /// Whether the represented solutions are complete or generic.
    #[getter]
    fn coverage(&self) -> &'static str {
        match self.set.coverage() {
            SolveCoverage::Complete => "complete",
            SolveCoverage::Generic => "generic",
        }
    }
    /// The condition under which the coverage claim applies.
    #[getter]
    fn coverage_guard(&self) -> PythonCondition {
        guard_condition(self.set.coverage_guard()).into()
    }
    /// Return the largest branch dimension for fixed external parameter values.
    ///
    /// A finite nonempty collection of points has dimension zero; a free line
    /// has dimension one. Returns -1 for a complete empty set, or None if unknown.
    fn dimension(&self) -> Option<isize> {
        self.set.dimension()
    }
    /// Return whether there are no solutions.
    ///
    /// Raises IncompleteCoverage if completeness is not established or emptiness
    /// depends on unresolved branch conditions. ``bool(result)`` is the opposite
    /// of this check; use ``len(result)`` to count represented branches.
    fn is_empty(&self) -> PyResult<bool> {
        self.set.is_empty().map_err(solve_error)
    }
    fn __len__(&self) -> usize {
        self.set.len()
    }
    fn __bool__(&self) -> PyResult<bool> {
        self.is_empty().map(|v| !v)
    }
    fn __getitem__(&self, index: isize) -> PyResult<PythonSolution> {
        let index = if index < 0 {
            self.set.len() as isize + index
        } else {
            index
        };
        usize::try_from(index)
            .ok()
            .and_then(|i| self.set.get(i))
            .cloned()
            .map(|solution| PythonSolution::from_set(solution, &self.set))
            .ok_or_else(|| PyIndexError::new_err("Solution branch index out of range"))
    }
    #[gen_stub(override_return_type(type_repr = "typing.Iterator[Solution]"))]
    fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        self.set
            .iter()
            .cloned()
            .map(|solution| PythonSolution::from_set(solution, &self.set))
            .collect::<Vec<_>>()
            .into_pyobject(py)?
            .try_iter()
    }
    fn __repr__(&self) -> String {
        self.__str__()
    }
    fn __str__(&self) -> String {
        self.text_with(&ToString::to_string)
    }
    /// Display all solution branches as aligned equations.
    fn _repr_html_(&self) -> PyResult<String> {
        let mut html = String::from("<div>");
        if self.set.len() == 0 {
            html.push_str(if self.set.coverage() == SolveCoverage::Complete {
                "<div>No solutions.</div>"
            } else {
                "<div>No branches represented; emptiness is not established.</div>"
            });
        } else {
            html.push_str("<div style=\"display: flex; flex-direction: column; gap: 0.65em; margin: 0.25em 0\">");
            for (index, solution) in self.set.iter().enumerate() {
                html.push_str(&solution_html(
                    solution,
                    solution.conditions(),
                    Some(index),
                )?);
            }
            html.push_str("</div>");
        }
        for context in self.display_context(&guard_html) {
            html.push_str(&format!("<div><small>{}</small></div>", context));
        }
        html.push_str("</div>");
        Ok(html)
    }
    /// Display all solution branches in IPython.
    ///
    /// Parameters
    /// ----------
    /// pretty: object
    ///     IPython's pretty printer.
    /// cycle: bool
    ///     Whether this object is already being displayed.
    fn _repr_pretty_(&self, pretty: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        pretty.call_method1(
            "text",
            (if cycle {
                "SolutionSet(...)".into()
            } else {
                self.text_with(&pretty_atom)
            },),
        )?;
        Ok(())
    }
}
