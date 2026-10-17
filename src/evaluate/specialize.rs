//! Exact parameter binding before evaluator optimization.
//!
//! Function bodies keep their lexical scope. Only definitions capturing bound
//! inputs need variants; ordinary functions taking those inputs as arguments
//! remain shared. Aliases are specialized in their caller's scope.

use std::{borrow::Cow, collections::BTreeSet};

use super::boolean::{BooleanEvaluation, NativeBoolean};
use super::*;

/// Exact values assigned to designated evaluator parameters, in row order.
///
/// The parameter representation is generic so source archives can store names
/// or encoded symbols while preserving the exact coefficient table.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParameterRows<A = Atom> {
    pub(super) parameters: Vec<A>,
    pub(super) rows: Vec<Vec<Rational>>,
}

impl<A> ParameterRows<A> {
    /// Create an exact table. Every row must have one value per parameter.
    /// Parameter identities are validated against evaluator inputs at build time.
    pub fn new(parameters: Vec<A>, rows: Vec<Vec<Rational>>) -> Result<Self, EvaluationError> {
        for row in &rows {
            if row.len() != parameters.len() {
                return Err(EvaluationError::InvalidParameterCount {
                    expected: parameters.len(),
                    actual: row.len(),
                });
            }
        }
        Ok(Self { parameters, rows })
    }

    /// The parameters corresponding to the ordered columns.
    pub fn parameters(&self) -> &[A] {
        &self.parameters
    }

    /// The exact rows, including repeated rows.
    pub fn rows(&self) -> &[Vec<Rational>] {
        &self.rows
    }

    /// Take the parameter representation and exact values.
    pub fn into_parts(self) -> (Vec<A>, Vec<Vec<Rational>>) {
        (self.parameters, self.rows)
    }

    /// Change only the parameter representation, retaining the exact rows.
    pub fn map_parameters<B>(self, map: impl FnMut(A) -> B) -> ParameterRows<B> {
        ParameterRows {
            parameters: self.parameters.into_iter().map(map).collect(),
            rows: self.rows,
        }
    }

    /// Fallibly change only the parameter representation.
    pub fn try_map_parameters<B, E>(
        self,
        map: impl FnMut(A) -> Result<B, E>,
    ) -> Result<ParameterRows<B>, E> {
        Ok(ParameterRows {
            parameters: self
                .parameters
                .into_iter()
                .map(map)
                .collect::<Result<_, _>>()?,
            rows: self.rows,
        })
    }
}

impl ParameterRows<Atom> {
    /// Validate symbolic input identities and return their bound column indices.
    ///
    /// This also validates deserialized row widths. Bound parameters must be
    /// indeterminates occurring exactly once in both the table and input list.
    pub fn input_positions<A: AtomCore>(
        &self,
        inputs: &[A],
    ) -> Result<Vec<usize>, EvaluationError> {
        let mut seen = HashSet::default();
        let mut positions = Vec::with_capacity(self.parameters.len());
        for parameter in &self.parameters {
            if Indeterminate::try_from(parameter.clone()).is_err() {
                return Err(EvaluationError::NotIndeterminate {
                    atom: parameter.clone(),
                });
            }
            let mut matches = inputs
                .iter()
                .enumerate()
                .filter(|(_, input)| input.as_atom_view() == parameter.as_view());
            let position = matches.next().map(|(position, _)| position);
            if !seen.insert(parameter) || position.is_none() || matches.next().is_some() {
                return Err(EvaluationError::EvaluatorConstructionFailed {
                    expression: parameter.clone(),
                    reason: "a bound parameter must occur exactly once in the input list and binding columns".into(),
                });
            }
            positions.push(position.unwrap());
        }
        for row in &self.rows {
            if row.len() != self.parameters.len() {
                return Err(EvaluationError::InvalidParameterCount {
                    expected: self.parameters.len(),
                    actual: row.len(),
                });
            }
        }
        Ok(positions)
    }

    pub(super) fn for_inputs<A: AtomCore>(
        inputs: &[Atom],
        parameters: &[A],
        rows: &[Vec<Rational>],
    ) -> Result<Self, EvaluationError> {
        let parameters = parameters
            .iter()
            .map(|parameter| parameter.as_atom_view().to_owned())
            .collect::<Vec<_>>();
        let table = Self::new(parameters, rows.to_vec())?;
        table.input_positions(inputs)?;
        Ok(table)
    }

    pub(super) fn specialize(
        &self,
        expressions: &[AtomView<'_>],
        functions: FunctionMap,
        inputs: &[Atom],
    ) -> Result<(Vec<Atom>, FunctionMap, Vec<Atom>), EvaluationError> {
        let symbol = crate::atom::AliasedAtom::fresh_alias_symbol();
        let mut specializer = Specializer {
            original: &functions,
            functions: Cow::Owned(functions.clone()),
            expand_aliases: false,
            inputs: inputs.iter().cloned().collect(),
            columns: self
                .parameters
                .iter()
                .enumerate()
                .map(|(index, parameter)| (parameter.clone(), index))
                .collect(),
            dependencies: HashMap::default(),
            variants: HashMap::default(),
            conditions: HashMap::default(),
            active: HashSet::default(),
            scopes: vec![HashSet::default()],
            scope_ids: HashMap::from_iter([(Vec::new(), 0)]),
            function_scopes: HashMap::default(),
            symbol,
        };
        let mut outputs = Vec::with_capacity(expressions.len());
        for expression in expressions {
            let mut terms = Vec::with_capacity(self.rows.len());
            for row in &self.rows {
                terms.push(specializer.atom(*expression, 0, row)?);
            }
            outputs.push(Atom::add_many(terms));
        }
        let parameters = inputs
            .iter()
            .filter(|parameter| !specializer.columns.contains_key(*parameter))
            .cloned()
            .collect::<Vec<_>>();
        let functions = specializer
            .functions
            .into_owned()
            .retain_reachable(&outputs, &parameters);
        Ok((outputs, functions, parameters))
    }
}

impl FunctionMap {
    /// Resolve caller-scoped aliases without inlining ordinary functions.
    ///
    /// `inputs` are opaque and shadow matching alias definitions. When preparing
    /// a function body for symbolic substitution, include its formal arguments
    /// here: resolve aliases before substituting the actual arguments, so local
    /// bindings cannot escape through an alias. Ordinary function bodies and
    /// their inlining policies remain unchanged, and fixed tags stay opaque.
    pub fn resolve_aliases<E: AtomCore, P: AtomCore>(
        &self,
        expression: &E,
        inputs: &[P],
    ) -> Result<Atom, EvaluationError> {
        let mut resolver = Specializer {
            original: self,
            functions: Cow::Borrowed(self),
            expand_aliases: true,
            inputs: inputs
                .iter()
                .map(|input| input.as_atom_view().to_owned())
                .collect(),
            columns: HashMap::default(),
            dependencies: HashMap::default(),
            variants: HashMap::default(),
            conditions: HashMap::default(),
            active: HashSet::default(),
            scopes: vec![HashSet::default()],
            scope_ids: HashMap::from_iter([(Vec::new(), 0)]),
            function_scopes: HashMap::default(),
            symbol: Symbol::IF, // Resolving aliases does not create definitions.
        };
        resolver.atom(expression.as_atom_view(), 0, &[])
    }

    /// Keep definitions reached by the given expressions and their callees.
    ///
    /// Exact input keys and formal arguments shadow definitions. Aliases inherit
    /// their caller's scope; ordinary functions resolve captures in global scope.
    /// Function identities and registration options are unchanged.
    pub fn retain_reachable<E: AtomCore, P: AtomCore>(
        mut self,
        expressions: &[E],
        inputs: &[P],
    ) -> Self {
        let retained = {
            let mut analysis = Specializer {
                original: &self,
                functions: Cow::Owned(FunctionMap::new()),
                expand_aliases: false,
                inputs: inputs
                    .iter()
                    .map(|input| input.as_atom_view().to_owned())
                    .collect(),
                columns: HashMap::default(),
                dependencies: HashMap::default(),
                variants: HashMap::default(),
                conditions: HashMap::default(),
                active: HashSet::default(),
                scopes: vec![HashSet::default()],
                scope_ids: HashMap::from_iter([(Vec::new(), 0)]),
                function_scopes: HashMap::default(),
                symbol: Symbol::IF, // No definitions are generated by dependency analysis.
            };
            let mut retained = HashSet::default();
            for expression in expressions {
                retained.extend(
                    analysis
                        .atom_dependencies(expression.as_atom_view(), 0, &mut HashSet::default())
                        .definitions,
                );
            }
            retained
        };
        self.retain_definition_ids(&retained);
        self
    }

    fn retain_definition_ids(&mut self, retained: &HashSet<usize>) {
        self.tagged_fn_map
            .retain(|_, function| retained.contains(&function.id));
        // Registration assigns the next ordinal from the table length. Keep
        // that invariant when callers append definitions after pruning.
        let mut functions = self.tagged_fn_map.values_mut().collect::<Vec<_>>();
        functions.sort_by_key(|function| function.id);
        for (id, function) in functions.into_iter().enumerate() {
            function.id = id;
        }
    }

    /// Whether the call's symbol and fixed tags select a registered definition.
    /// This includes caller-scoped aliases; argument counts are checked at build time.
    pub fn contains_function<A: AtomCore>(&self, call: &A) -> bool {
        self.get(call.as_atom_view()).is_some()
    }
}

#[derive(Clone, Default)]
struct Dependencies {
    columns: BTreeSet<usize>,
    definitions: HashSet<usize>,
}

impl Dependencies {
    fn extend(&mut self, other: Self) {
        self.columns.extend(other.columns);
        self.definitions.extend(other.definitions);
    }
}

type Scope = usize;
type DefinitionScope = (usize, Scope);
type Variant = (usize, Scope, Vec<Rational>);

struct Specializer<'a> {
    original: &'a FunctionMap,
    functions: Cow<'a, FunctionMap>,
    expand_aliases: bool,
    inputs: HashSet<Atom>,
    columns: HashMap<Atom, usize>,
    dependencies: HashMap<DefinitionScope, Dependencies>,
    variants: HashMap<Variant, Atom>,
    conditions: HashMap<Atom, Option<NativeBoolean>>,
    active: HashSet<Variant>,
    scopes: Vec<HashSet<Atom>>,
    scope_ids: HashMap<Vec<Atom>, Scope>,
    function_scopes: HashMap<usize, Scope>,
    symbol: Symbol,
}

impl Specializer<'_> {
    fn scope(&mut self, function: &Expr, caller: Scope) -> Scope {
        if function.is_alias {
            return caller;
        }
        if let Some(scope) = self.function_scopes.get(&function.id) {
            return *scope;
        }
        // Intern each formal set once. Alias definitions with the same lexical
        // bindings keep sharing one variant even when reached through different
        // functions, while per-atom membership and per-row keys stay constant-size.
        let mut parameters = function
            .args
            .iter()
            .cloned()
            .map(Atom::from)
            .collect::<Vec<_>>();
        parameters.sort();
        parameters.dedup();
        let scope = if let Some(scope) = self.scope_ids.get(&parameters) {
            *scope
        } else {
            let scope = self.scopes.len();
            self.scopes.push(parameters.iter().cloned().collect());
            self.scope_ids.insert(parameters, scope);
            scope
        };
        self.function_scopes.insert(function.id, scope);
        scope
    }

    fn definition_dependencies(
        &mut self,
        function: &Expr,
        scope: Scope,
        pending: &mut HashSet<DefinitionScope>,
    ) -> Dependencies {
        let key = (function.id, scope);
        if let Some(dependencies) = self.dependencies.get(&key) {
            return dependencies.clone();
        }
        if !pending.insert(key) {
            // A cycle may be in an inactive branch. Conservatively mark all
            // columns here; specialization rejects only a reached cycle.
            return Dependencies {
                columns: (0..self.columns.len()).collect(),
                definitions: [function.id].into_iter().collect(),
            };
        }
        let mut dependencies = self.atom_dependencies(function.body.as_view(), scope, pending);
        dependencies.definitions.insert(function.id);
        pending.remove(&key);
        self.dependencies.insert(key, dependencies.clone());
        dependencies
    }

    fn atom_dependencies(
        &mut self,
        atom: AtomView<'_>,
        scope: Scope,
        pending: &mut HashSet<DefinitionScope>,
    ) -> Dependencies {
        let mut dependencies = Dependencies::default();
        let original = self.original;
        atom.visitor(&mut |view| {
            if self.scopes[scope].contains(view.get_data()) {
                return false;
            }
            if let Some(column) = self.columns.get(view.get_data()) {
                dependencies.columns.insert(*column);
                return false;
            }
            if self.inputs.contains(view.get_data()) {
                return false;
            }
            if let Some(function) = original.get(view) {
                let function_scope = self.scope(function, scope);
                dependencies.extend(self.definition_dependencies(
                    function,
                    function_scope,
                    pending,
                ));
                if let AtomView::Fun(call) = view {
                    for argument in call.iter().skip(function.tag_len) {
                        dependencies.extend(self.atom_dependencies(argument, scope, pending));
                    }
                }
                return false;
            }
            if let AtomView::Fun(call) = view {
                let tags = call
                    .get_symbol()
                    .get_evaluation_info()
                    .map_or(0, |info| info.get_tag_count());
                for argument in call.iter().skip(tags) {
                    dependencies.extend(self.atom_dependencies(argument, scope, pending));
                }
                return false;
            }
            true
        });
        dependencies
    }

    fn atom(
        &mut self,
        atom: AtomView<'_>,
        scope: Scope,
        row: &[Rational],
    ) -> Result<Atom, EvaluationError> {
        let mut failure = None;
        let result = atom.replace_map(|view, _, output| {
            if failure.is_some() {
                output.set_from_view(&view);
                return;
            }
            if self.scopes[scope].contains(view.get_data()) {
                output.set_from_view(&view);
            } else if let Some(column) = self.columns.get(view.get_data()) {
                **output = Atom::num(row[*column].clone());
            } else if self.inputs.contains(view.get_data()) {
                output.set_from_view(&view);
            } else if matches!(view, AtomView::Fun(_) | AtomView::Var(_)) {
                match self.call(view, scope, row) {
                    Ok(value) => **output = value,
                    Err(error) => {
                        failure = Some(error);
                        output.set_from_view(&view);
                    }
                }
            }
        });
        failure.map_or(Ok(result), Err)
    }

    fn call(
        &mut self,
        atom: AtomView<'_>,
        scope: Scope,
        row: &[Rational],
    ) -> Result<Atom, EvaluationError> {
        if let AtomView::Fun(call) = atom
            && call.get_symbol() == Symbol::IF
            && call.get_nargs() == 3
        {
            let condition = self.atom(call.get(0), scope, row)?;
            if let Some(condition) = self.constant_condition(&condition) {
                return self.atom(
                    call.get(if condition.is_zero() { 2 } else { 1 }),
                    scope,
                    row,
                );
            }
            return Ok(Symbol::IF.call_args([
                condition,
                self.atom(call.get(1), scope, row)?,
                self.atom(call.get(2), scope, row)?,
            ]));
        }
        let original = self.original;
        if let Some(function) = original.get(atom) {
            if self.expand_aliases && function.is_alias {
                let key = (function.id, scope, Vec::new());
                if let Some(body) = self.variants.get(&key) {
                    return Ok(body.clone());
                }
                if !self.active.insert(key.clone()) {
                    return Err(EvaluationError::EvaluatorConstructionFailed {
                        expression: atom.to_owned(),
                        reason: "recursive alias dependency".into(),
                    });
                }
                let body = self.atom(function.body.as_view(), scope, row)?;
                self.active.remove(&key);
                self.variants.insert(key, body.clone());
                return Ok(body);
            }
            let arguments = match atom {
                AtomView::Fun(call) => call
                    .iter()
                    .skip(function.tag_len)
                    .map(|argument| self.atom(argument, scope, row))
                    .collect::<Result<Vec<_>, _>>()?,
                _ => Vec::new(),
            };
            if arguments.len() != function.args.len() {
                return Err(EvaluationError::WrongNumberOfArguments {
                    function: match atom {
                        AtomView::Fun(call) => call.get_symbol(),
                        AtomView::Var(variable) => variable.get_symbol(),
                        _ => unreachable!(),
                    },
                    expected: function.args.len() + function.tag_len,
                    actual: arguments.len() + function.tag_len,
                });
            }
            let scope = self.scope(function, scope);
            let dependencies = if self.columns.is_empty() {
                Dependencies::default()
            } else {
                self.definition_dependencies(function, scope, &mut HashSet::default())
            };
            if dependencies.columns.is_empty() {
                return Ok(match atom {
                    AtomView::Fun(call) => call.get_symbol().call_args(
                        call.iter()
                            .take(function.tag_len)
                            .map(|tag| tag.to_owned())
                            .chain(arguments),
                    ),
                    _ => atom.to_owned(),
                });
            }
            let values = dependencies
                .columns
                .iter()
                .map(|index| row[*index].clone())
                .collect();
            let key = (function.id, scope, values);
            let name = if let Some(name) = self.variants.get(&key) {
                name.clone()
            } else {
                if !self.active.insert(key.clone()) {
                    return Err(EvaluationError::EvaluatorConstructionFailed {
                        expression: atom.to_owned(),
                        reason: "recursive function or alias dependency".into(),
                    });
                }
                let body = self.atom(function.body.as_view(), scope, row)?;
                self.active.remove(&key);
                if function.is_alias && matches!(body.as_view(), AtomView::Num(_)) {
                    self.variants.insert(key, body.clone());
                    return Ok(body);
                }
                let name = self.symbol.call((function.id, self.variants.len()));
                self.functions.to_mut().add_function_with_options(
                    Indeterminate::try_from(name.clone()).unwrap(),
                    function.args.clone(),
                    body,
                    function.options.clone(),
                )?;
                if function.is_alias {
                    self.functions
                        .to_mut()
                        .tagged_fn_map
                        .get_mut(&(
                            self.symbol,
                            name.as_fun_view()
                                .unwrap()
                                .iter()
                                .map(|tag| tag.to_owned())
                                .collect(),
                        ))
                        .unwrap()
                        .is_alias = true;
                }
                self.variants.insert(key, name.clone());
                name
            };
            return Ok(match name.as_view() {
                AtomView::Fun(call) => call
                    .get_symbol()
                    .call_args(call.iter().map(|tag| tag.to_owned()).chain(arguments)),
                _ => name,
            });
        }
        if let AtomView::Fun(call) = atom {
            let tags = call
                .get_symbol()
                .get_evaluation_info()
                .map_or(0, |info| info.get_tag_count());
            let mut arguments = call
                .iter()
                .take(tags)
                .map(|tag| tag.to_owned())
                .collect::<Vec<_>>();
            arguments.extend(
                call.iter()
                    .skip(tags)
                    .map(|argument| self.atom(argument, scope, row))
                    .collect::<Result<Vec<_>, _>>()?,
            );
            return Ok(call.get_symbol().call_args(arguments));
        }
        Ok(atom.to_owned())
    }

    fn constant_condition(&mut self, condition: &Atom) -> Option<NativeBoolean> {
        if let Ok(value) = Complex::<Rational>::try_from(condition.as_view()) {
            // Literal conditions are already exact symbolic bindings. Function
            // boundaries below retain their native finite-precision semantics.
            return Some(NativeBoolean::from_truth(!value.is_zero()));
        }
        if let Some(value) = self.conditions.get(condition) {
            return *value;
        }
        let value = if self.inputs.contains(condition) {
            None
        } else {
            // Keep native function boundaries: a retained argument is rounded
            // before its body executes. Only precision-independent Boolean
            // operations may be used to certify a predicate across that boundary.
            let mut inputs = self.inputs.iter().cloned().collect::<Vec<_>>();
            inputs.sort();
            AtomView::linearize_multiple(
                &[condition],
                &self.functions,
                &inputs,
                OptimizationSettings::default(),
            )
            .ok()
            .and_then(|program| BooleanEvaluation::new(&program.external_fns).program(&program))
        };
        self.conditions.insert(condition.clone(), value);
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse, symbol};

    #[test]
    fn resolving_aliases_preserves_formals_tags_and_function_boundaries() {
        let q = parse!("resolve_aliases::coefficient(2,0)");
        let x = parse!("resolve_aliases::x");
        let alias = parse!("resolve_aliases::alias");
        let forwarded = parse!("resolve_aliases::forwarded");
        let retained = symbol!("resolve_aliases::retained");
        let mut functions = FunctionMap::new();
        functions
            .add_aliases([
                (q.clone(), Atom::num(999)),
                (alias.clone(), &q * &x),
                (forwarded.clone(), &alias + q.pow(2)),
            ])
            .unwrap();
        functions
            .add_tagged_function_with_options(
                retained,
                vec![alias.clone()],
                vec![Indeterminate::try_from(q.clone()).unwrap()],
                forwarded.clone(),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
        let inputs = [q.clone(), x.clone()];
        let body = functions.resolve_aliases(&forwarded, &inputs).unwrap();
        assert_eq!(body, &q * &x + q.pow(2));
        let call = retained.call_args([alias.clone(), forwarded]);
        assert_eq!(
            functions.resolve_aliases(&call, &inputs).unwrap(),
            retained.call_args([alias.clone(), body.clone()]),
            "fixed tags and the retained body must not be expanded"
        );
        let bound = body.replace(q.to_pattern()).with(Atom::num(3));
        let mut expected = bound
            .evaluator(std::slice::from_ref(&x))
            .build()
            .unwrap()
            .map_coeff(&|value| value.re.to_f64());
        let mut native = retained
            .call_args([alias, Atom::num(3)])
            .evaluator(std::slice::from_ref(&x))
            .function_map(functions)
            .build()
            .unwrap()
            .map_coeff(&|value| value.re.to_f64());
        assert_eq!(expected.evaluate_single(&[5.]), 24.);
        assert_eq!(native.evaluate_single(&[5.]), 24.);
    }

    #[test]
    fn resolving_aliases_keeps_opaque_inputs_and_skips_inactive_cycles() {
        let x = parse!("resolve_aliases_live::x");
        let opaque = &x + 1;
        let first = parse!("resolve_aliases_live::first");
        let second = parse!("resolve_aliases_live::second");
        let mut functions = FunctionMap::new();
        functions
            .add_aliases([
                (x.clone(), Atom::num(7)),
                (first.clone(), second.clone()),
                (second, first.clone()),
            ])
            .unwrap();
        assert_eq!(
            functions
                .resolve_aliases(&opaque, std::slice::from_ref(&opaque))
                .unwrap(),
            opaque
        );
        let selected = Symbol::IF.call_args([Atom::Zero, first.clone(), x.clone()]);
        assert_eq!(
            functions
                .resolve_aliases(&selected, std::slice::from_ref(&x))
                .unwrap(),
            x
        );
        assert!(functions.resolve_aliases(&first, &[] as &[Atom]).is_err());
    }

    #[test]
    fn retained_physical_predicates_preserve_native_zero_division() {
        let x = symbol!("row_predicate_domain::x");
        let c = symbol!("row_predicate_domain::c");
        let a = symbol!("row_predicate_domain::a");
        let b = symbol!("row_predicate_domain::b");
        let f = symbol!("row_predicate_domain::f");
        let mut functions = FunctionMap::new();
        functions
            .add_function_with_options(
                f,
                vec![a, b],
                Atom::var(a) / Atom::var(b),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
        let root = Symbol::IF.call_args([
            f.call_args([Atom::var(c), Atom::var(x)]),
            Atom::var(x) + 1,
            Atom::var(x) + 2,
        ]);
        let inputs = [Atom::var(x), Atom::var(c)];
        let mut reference = root
            .evaluator(&inputs)
            .function_map(functions.clone())
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        let mut summed = root
            .evaluator(&inputs)
            .function_map(functions)
            .sum_over_parameters(&inputs[1..], &[vec![Rational::zero()]])
            .unwrap()
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        for (x, expected) in [(0., 1.), (2., 4.)] {
            assert_eq!(reference.evaluate_single(&[x, 0.]), expected);
            assert_eq!(summed.evaluate_single(&[x]), expected);
        }
    }

    #[test]
    fn retained_nested_predicates_are_not_inlined_for_exact_probing() {
        let x = symbol!("row_nested_predicate_domain::x");
        let c = symbol!("row_nested_predicate_domain::c");
        let d = symbol!("row_nested_predicate_domain::d");
        let a = symbol!("row_nested_predicate_domain::a");
        let b = symbol!("row_nested_predicate_domain::b");
        let f = symbol!("row_nested_predicate_domain::f");
        let mut functions = FunctionMap::new();
        functions
            .add_function_with_options(
                f,
                vec![a, b],
                Symbol::IF.call_args([Atom::var(a) / Atom::var(b), Atom::one(), Atom::Zero]),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
        let root = Symbol::IF.call_args([
            f.call_args([Atom::var(c), Atom::var(d)]),
            Atom::var(x) + 1,
            Atom::var(x) + 2,
        ]);
        let inputs = [Atom::var(x), Atom::var(c), Atom::var(d)];
        let mut reference = root
            .evaluator(&inputs)
            .function_map(functions.clone())
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        assert_eq!(reference.evaluate_single(&[2., 0., 0.]), 3.);
        let mut summed = root
            .evaluator(&inputs)
            .function_map(functions)
            .sum_over_parameters(&inputs[1..], &[vec![Rational::zero(), Rational::zero()]])
            .unwrap()
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        assert_eq!(summed.evaluate_single(&[2.]), 3.);
    }

    #[test]
    fn exact_row_binding_preserves_retained_predicate_rounding() {
        let x = symbol!("row_retained_predicate_rounding::x");
        let c = symbol!("row_retained_predicate_rounding::c");
        let a = symbol!("row_retained_predicate_rounding::a");
        let f = symbol!("row_retained_predicate_rounding::f");
        let value = Rational::new(18_014_398_509_481_985i64, 18_014_398_509_481_984i64);
        let mut functions = FunctionMap::new();
        functions
            .add_function_with_options(
                f,
                vec![a],
                Symbol::IF.call_args([Atom::var(a) - 1, Atom::one(), Atom::Zero]),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
        let root = Symbol::IF.call_args([
            f.call_args([Atom::var(c)]),
            Atom::var(x) + 1,
            Atom::var(x) + 2,
        ]);
        let explicit = root.replace(Atom::var(c)).with(Atom::num(value.clone()));
        let mut reference = explicit
            .evaluator(&[Atom::var(x)])
            .function_map(functions.clone())
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        assert_eq!(reference.evaluate_single(&[2.]), 4.);
        let mut summed = root
            .evaluator(&[Atom::var(x), Atom::var(c)])
            .function_map(functions)
            .sum_over_parameters(&[Atom::var(c)], &[vec![value]])
            .unwrap()
            .build()
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());
        assert_eq!(
            summed.evaluate_single(&[2.]),
            4.,
            "constant probing must not override a retained function's native branch"
        );
    }

    #[test]
    fn cancelled_rows_remove_unreachable_function_bodies_before_optimization() {
        let inputs = [parse!("rows_cancel::x"), parse!("rows_cancel::c")];
        let root = parse!("rows_cancel::c*rows_cancel::n(rows_cancel::x)");
        let mut functions = FunctionMap::new();
        functions
            .add_function_with_options(
                symbol!("rows_cancel::n"),
                vec![symbol!("rows_cancel::t")],
                parse!("rows_cancel::t^2+1"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
        let rows = ParameterRows::new(
            vec![inputs[1].clone()],
            vec![vec![1.into()], vec![(-1).into()]],
        )
        .unwrap();
        let (roots, functions, remaining) = rows
            .specialize(&[root.as_view()], functions, &inputs)
            .unwrap();
        assert_eq!(roots, vec![Atom::Zero]);
        assert_eq!(remaining, inputs[..1]);
        assert!(!functions.contains_function(&parse!("rows_cancel::n(rows_cancel::x)")));
    }

    #[test]
    fn reachable_definitions_respect_formals_and_allow_later_registration() {
        let mut functions = FunctionMap::new();
        functions
            .add_function(
                symbol!("rows_reach::unused"),
                vec![symbol!("rows_reach::t")],
                parse!("rows_reach::t+99"),
            )
            .unwrap();
        functions
            .add_aliases([(parse!("rows_reach::a"), parse!("rows_reach::p^2"))])
            .unwrap();
        functions
            .add_function_with_options(
                symbol!("rows_reach::f"),
                vec![symbol!("rows_reach::p")],
                parse!("rows_reach::a"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
        functions
            .add_aliases([(parse!("rows_reach::p"), parse!("rows_reach::undefined"))])
            .unwrap();
        let expression = parse!("rows_reach::f(rows_reach::x)");
        let inputs = [parse!("rows_reach::x")];
        let mut functions = functions.retain_reachable(std::slice::from_ref(&expression), &inputs);
        assert!(functions.contains_function(&parse!("rows_reach::a")));
        assert!(functions.contains_function(&parse!("rows_reach::f(rows_reach::x)")));
        assert!(!functions.contains_function(&parse!("rows_reach::p")));
        assert!(!functions.contains_function(&parse!("rows_reach::unused(rows_reach::x)")));
        functions
            .add_function_with_options(
                symbol!("rows_reach::h"),
                vec![symbol!("rows_reach::t")],
                parse!("2*rows_reach::t"),
                FunctionRegistrationOptions::new().inlining(InliningPolicy::Never),
            )
            .unwrap();
        let mut evaluator = (expression + parse!("rows_reach::h(rows_reach::x)"))
            .evaluator(&inputs)
            .function_map(functions)
            .build()
            .unwrap()
            .map_coeff(&|value| value.re.to_f64());
        assert_eq!(evaluator.evaluate_single(&[3.]), 15.);
    }

    #[cfg(all(feature = "serde", feature = "bincode"))]
    #[test]
    fn exact_table_archives_keep_parameter_order_and_rational_values() {
        let original = ParameterRows::new(
            vec!["second".to_owned(), "first".to_owned()],
            vec![vec![Rational::from((1, 3)), Rational::from((-2, 7))]],
        )
        .unwrap();
        let json = serde_json::to_vec(&original).unwrap();
        let decoded: ParameterRows<String> = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded, original);
        let bytes = bincode::encode_to_vec(&original, bincode::config::standard()).unwrap();
        let (decoded, size): (ParameterRows<String>, _) =
            bincode::decode_from_slice(&bytes, bincode::config::standard()).unwrap();
        assert_eq!(size, bytes.len());
        assert_eq!(decoded, original);
        let symbols = decoded
            .try_map_parameters(|name| Symbol::parse(name, "rows_archive").map(Atom::var))
            .unwrap();
        assert_eq!(symbols.rows(), original.rows());
        assert_eq!(symbols.parameters()[0], parse!("rows_archive::second"));
    }

    #[test]
    fn exact_rows_sum_outputs_and_preserve_multiplicity() {
        let ring = crate::domains::float::FloatField::from_rep(Complex::<Rational>::default());
        let parameters = [parse!("rows::x"), parse!("rows::c")];
        let roots = [parse!("rows::x+rows::c"), parse!("rows::x*rows::c")];
        let rows = vec![
            vec![Rational::from((-1, 3))],
            vec![Rational::from((-1, 3))],
            vec![Rational::from((2, 3))],
        ];
        for direct in [false, true] {
            let mut evaluator = Atom::evaluator_multiple(&roots, &parameters)
                .sum_over_parameters(&parameters[1..], &rows)
                .unwrap()
                .direct_translation(direct)
                .build()
                .unwrap();
            assert_eq!(evaluator.get_input_len(), 1);
            assert_eq!(evaluator.get_output_len(), 2);
            let mut output = vec![Complex::default(); 2];
            evaluator.evaluate_in_ring(
                &[Complex::new(Rational::from(5), Rational::zero())],
                &mut output,
                &ring,
            );
            assert_eq!(
                output,
                vec![
                    Complex::new(Rational::from(15), Rational::zero()),
                    Complex::default()
                ]
            );
            let mut empty = Atom::evaluator_multiple(&roots, &parameters)
                .sum_over_parameters(&parameters[1..], &[])
                .unwrap()
                .direct_translation(direct)
                .build()
                .unwrap();
            empty.evaluate_in_ring(&[Complex::default()], &mut output, &ring);
            assert_eq!(output, vec![Complex::default(); 2]);
        }
    }

    #[test]
    fn formal_arguments_and_tags_keep_one_numerator_body() {
        let parameters = [parse!("rows_tag::x"), parse!("rows_tag::c")];
        for (policy, repetitions) in [InliningPolicy::Always, InliningPolicy::Never]
            .into_iter()
            .flat_map(|policy| [1, 32].map(|repetitions| (policy, repetitions)))
        {
            let rows = (0..repetitions)
                .flat_map(|_| [vec![1.into()], vec![2.into()], vec![1.into()]])
                .collect::<Vec<_>>();
            let mut evaluator = parse!("rows_tag::n(rows_tag::c,rows_tag::x,rows_tag::c)")
                .evaluator(&parameters)
                .add_aliases([(parse!("rows_tag::a"), parse!("rows_tag::c+rows_tag::x"))])
                .unwrap()
                .add_tagged_function_with_options(
                    symbol!("rows_tag::n"),
                    vec![parameters[1].clone()],
                    vec![symbol!("rows_tag::t"), symbol!("rows_tag::c")],
                    parse!("rows_tag::t*rows_tag::c+rows_tag::a"),
                    FunctionRegistrationOptions::new().inlining(policy),
                )
                .unwrap()
                .sum_over_parameters(&parameters[1..], &rows)
                .unwrap()
                .build()
                .unwrap()
                .map_coeff(&|value| value.re.to_f64());
            assert_eq!(evaluator.evaluate_single(&[3.]), 25. * repetitions as f64);
            let exported = evaluator.export_instructions();
            assert_eq!(
                exported.sub_evaluators.len(),
                usize::from(policy == InliningPolicy::Never)
            );
            if policy == InliningPolicy::Never {
                assert_eq!(exported.sub_evaluators[0].symbol, symbol!("rows_tag::n"));
                assert_eq!(
                    exported.sub_evaluators[0].tags,
                    vec![parameters[1].to_canonical_string()]
                );
            }
        }
    }

    #[test]
    fn equal_wide_formal_scopes_share_captured_alias_variants() {
        let namespace = "rows_shared_scope";
        let parameters = [
            parse!("rows_shared_scope::x"),
            parse!("rows_shared_scope::c"),
        ];
        let formals = (0..64)
            .map(|index| Symbol::parse(format!("p{index}"), namespace).unwrap())
            .collect::<Vec<_>>();
        let alias = parse!("rows_shared_scope::alias");
        let mut functions = FunctionMap::new();
        functions
            .add_aliases([(alias.clone(), Atom::var(formals[0]) + &parameters[1])])
            .unwrap();
        let mut roots = Vec::new();
        for (name, mut args) in [
            (symbol!("rows_shared_scope::f"), formals.clone()),
            (symbol!("rows_shared_scope::g"), formals.clone()),
        ] {
            if name == symbol!("rows_shared_scope::g") {
                args.reverse();
            }
            functions
                .add_function_with_options(
                    name,
                    args,
                    alias.clone(),
                    FunctionRegistrationOptions::new()
                        .inlining(InliningPolicy::Never)
                        .prune_unused_arguments(true),
                )
                .unwrap();
            roots.push(name.call_args(std::iter::repeat_n(parameters[0].clone(), formals.len())));
        }
        let root = Atom::add_many(roots);
        let rows = ParameterRows::new(
            vec![parameters[1].clone()],
            vec![vec![1.into()], vec![2.into()], vec![1.into()]],
        )
        .unwrap();
        let (roots, functions, inputs) = rows
            .specialize(&[root.as_view()], functions, &parameters)
            .unwrap();
        assert_eq!(
            functions
                .tagged_fn_map
                .values()
                .filter(|function| function.is_alias)
                .count(),
            2,
            "identical formal sets share an alias variant for each distinct captured value"
        );
        let mut evaluator = Atom::evaluator_multiple(&roots, &inputs)
            .function_map(functions)
            .build()
            .unwrap()
            .map_coeff(&|value| value.re.to_f64());
        assert_eq!(evaluator.evaluate_single(&[3.]), 26.);
    }

    #[test]
    fn captures_and_caller_scoped_aliases_have_distinct_bindings() {
        let parameters = [
            parse!("rows_scope::x"),
            parse!("rows_scope::c"),
            parse!("rows_scope::unused"),
        ];
        for policy in [InliningPolicy::Always, InliningPolicy::Never] {
            let mut evaluator = parse!("rows_scope::f(rows_scope::x)")
                .evaluator(&parameters)
                .add_aliases([(
                    parse!("rows_scope::a"),
                    parse!("rows_scope::c+rows_scope::x"),
                )])
                .unwrap()
                .add_function_with_options(
                    symbol!("rows_scope::g"),
                    vec![symbol!("rows_scope::u")],
                    parse!("rows_scope::c+rows_scope::u"),
                    FunctionRegistrationOptions::new().inlining(policy),
                )
                .unwrap()
                .add_function_with_options(
                    symbol!("rows_scope::f"),
                    vec![symbol!("rows_scope::c")],
                    parse!("rows_scope::a+rows_scope::g(rows_scope::c)"),
                    FunctionRegistrationOptions::new().inlining(policy),
                )
                .unwrap()
                .sum_over_parameters(
                    &parameters[1..],
                    &[
                        vec![1.into(), 10.into()],
                        vec![2.into(), 20.into()],
                        vec![1.into(), 30.into()],
                    ],
                )
                .unwrap()
                .build()
                .unwrap()
                .map_coeff(&|value| value.re.to_f64());
            assert_eq!(evaluator.evaluate_single(&[3.]), 31.);
            // The third row changes an irrelevant column and reuses both
            // specialized definitions from the first row.
            let exported = evaluator.export_instructions();
            assert_eq!(
                exported.sub_evaluators.len(),
                if policy == InliningPolicy::Never {
                    2
                } else {
                    0
                }
            );
            for outer in &exported.sub_evaluators {
                // The exporter nests callees under their caller, rather than
                // flattening all four retained bodies into the root table.
                assert_eq!(outer.instructions.sub_evaluators.len(), 1);
                assert!(
                    outer.instructions.sub_evaluators[0]
                        .instructions
                        .sub_evaluators
                        .is_empty()
                );
            }
        }
    }

    #[test]
    fn static_guards_discard_singular_and_recursive_aliases() {
        let parameters = [parse!("rows_lazy::x"), parse!("rows_lazy::c")];
        let root = Symbol::IF.call_args([
            parse!("rows_lazy::guard"),
            parse!("rows_lazy::pole"),
            parse!("rows_lazy::x"),
        ]);
        for direct in [false, true] {
            let mut evaluator = root
                .evaluator(&parameters)
                .add_aliases([
                    (parse!("rows_lazy::guard"), parse!("rows_lazy::c")),
                    (parse!("rows_lazy::pole"), parse!("1/rows_lazy::c")),
                ])
                .unwrap()
                .sum_over_parameters(&parameters[1..], &[vec![0.into()], vec![2.into()]])
                .unwrap()
                .direct_translation(direct)
                .build()
                .unwrap()
                .map_coeff(&|value| value.re.to_f64());
            assert_eq!(evaluator.evaluate_single(&[3.]), 3.5);
        }
        let root = Symbol::IF.call_args([
            parameters[1].clone(),
            parse!("rows_lazy::cycle"),
            parameters[0].clone(),
        ]);
        let mut evaluator = root
            .evaluator(&parameters)
            .add_aliases([(parse!("rows_lazy::cycle"), parse!("rows_lazy::cycle"))])
            .unwrap()
            .sum_over_parameters(&parameters[1..], &[vec![0.into()]])
            .unwrap()
            .build()
            .unwrap()
            .map_coeff(&|value| value.re.to_f64());
        assert_eq!(evaluator.evaluate_single(&[3.]), 3.);
    }

    #[test]
    fn registered_constant_predicates_select_before_pole_binding() {
        let parameters = [parse!("rows_predicate::x"), parse!("rows_predicate::c")];
        let root = Symbol::IF.call_args([
            parse!("rows_predicate::flag(rows_predicate::c)"),
            parse!("1/rows_predicate::c"),
            parameters[0].clone(),
        ]);
        for policy in [InliningPolicy::Always, InliningPolicy::Never] {
            for (body, at_zero) in [
                (parse!("rows_predicate::t"), 0.5),
                (
                    Symbol::IF.call_args([
                        parse!("rows_predicate::t"),
                        parameters[0].clone(),
                        Atom::Zero,
                    ]),
                    0.,
                ),
            ] {
                let mut evaluator = root
                    .evaluator(&parameters)
                    .add_aliases([(parameters[0].clone(), Atom::num(7))])
                    .unwrap()
                    .add_function_with_options(
                        symbol!("rows_predicate::flag"),
                        vec![symbol!("rows_predicate::t")],
                        body,
                        FunctionRegistrationOptions::new().inlining(policy),
                    )
                    .unwrap()
                    .sum_over_parameters(&parameters[1..], &[vec![0.into()], vec![2.into()]])
                    .unwrap()
                    .build()
                    .unwrap()
                    .map_coeff(&|value| value.re.to_f64());
                assert_eq!(evaluator.evaluate_single(&[3.]), 3.5);
                assert_eq!(evaluator.evaluate_single(&[0.]), at_zero);
            }
        }
    }

    #[test]
    fn opaque_inputs_shadow_functions_and_invalid_tables_fail() {
        let parameters = [
            parse!("rows_input::x"),
            parse!("rows_input::p(rows_input::x)"),
        ];
        let mut evaluator = parse!("rows_input::p(rows_input::x)+rows_input::x")
            .evaluator(&parameters)
            .add_aliases([(parameters[1].clone(), parse!("1/rows_input::x"))])
            .unwrap()
            .sum_over_parameters(&parameters[1..], &[vec![2.into()], vec![3.into()]])
            .unwrap()
            .build()
            .unwrap()
            .map_coeff(&|value| value.re.to_f64());
        assert_eq!(evaluator.evaluate_single(&[0.]), 5.);
        let root = parameters[0].clone();
        assert!(
            root.evaluator(&parameters)
                .sum_over_parameters(&parameters[1..], &[vec![]])
                .is_err()
        );
        assert!(
            root.evaluator(&parameters)
                .sum_over_parameters(&[parse!("rows_input::missing")], &[])
                .is_err()
        );
        assert!(
            root.evaluator(&parameters)
                .sum_over_parameters(&[parameters[0].clone(), parameters[0].clone()], &[])
                .is_err()
        );
    }
}
