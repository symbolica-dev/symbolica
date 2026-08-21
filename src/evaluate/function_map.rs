use crate::atom::InlineVar;

use super::*;

pub type EvalFnType<A, T> = Box<
    dyn Fn(
        &[T],
        &HashMap<A, T>,
        &HashMap<Symbol, EvaluationFn<A, T>>,
        &mut HashMap<AtomView<'_>, T>,
    ) -> T,
>;

/// A closure that can be called to evaluate a function called with arguments of type `T`.
pub struct EvaluationFn<A, T>(EvalFnType<A, T>);

impl<A, T> EvaluationFn<A, T> {
    pub fn new(f: EvalFnType<A, T>) -> EvaluationFn<A, T> {
        EvaluationFn(f)
    }

    /// Get a reference to the function that can be called to evaluate it.
    pub fn get(&self) -> &EvalFnType<A, T> {
        &self.0
    }
}

/// A map of functions and constants used for evaluating expressions.
///
/// Examples
/// --------
/// ```rust
/// use symbolica::prelude::*;
/// let params = vec![parse!("x")];
/// let mut evaluator = parse!("f(x)")
///     .evaluator(&params)
///     .add_function(symbol!("f"), vec![symbol!("x")], parse!("x^2 + 1")).unwrap()
///     .build()
///     .unwrap()
///     .map_coeff(&|x| x.re.to_f64());
/// assert_eq!(evaluator.evaluate_single(&[2.0]), 5.0);
/// ```
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::Encode),
    derive(bincode_trait_derive::Decode),
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
    trait_decode(trait = crate::state::HasStateMap)
)]
#[derive(Clone, Debug)]
pub struct FunctionMap {
    pub(super) tagged_fn_map: HashMap<(Symbol, Vec<Atom>), Expr>,
    pub(super) tag: HashMap<Symbol, usize>,
}

impl Default for FunctionMap {
    fn default() -> Self {
        Self::new()
    }
}

impl FunctionMap {
    /// Create a new, empty function map.
    pub fn new() -> Self {
        FunctionMap {
            tagged_fn_map: HashMap::default(),
            tag: HashMap::default(),
        }
    }

    /// Register a function. If `name` is a symbol, it will be treated as a regular function; if it is a function, its arguments will be treated as tags.
    /// Only provide explicit arguments in `args` if they are meant to shadow function arguments from a higher scope.
    pub fn add_function<S: Into<Indeterminate>, A: Into<Indeterminate>>(
        &mut self,
        name: S,
        args: Vec<A>,
        body: Atom,
    ) -> Result<(), EvaluationError> {
        self.add_function_impl(name, args, body, true)
    }

    /// Register a function that is evaluated by a separate sub-evaluator instead of being
    /// inlined at every call site.
    ///
    /// This can reduce the instruction-cache footprint when a large function is called one or
    /// more times. It adds function-call overhead, so small functions should normally remain
    /// inlined through [`Self::add_function`].
    pub fn add_function_no_inline<S: Into<Indeterminate>, A: Into<Indeterminate>>(
        &mut self,
        name: S,
        args: Vec<A>,
        body: Atom,
    ) -> Result<(), EvaluationError> {
        self.add_function_impl(name, args, body, false)
    }

    fn add_function_impl<S: Into<Indeterminate>, A: Into<Indeterminate>>(
        &mut self,
        name: S,
        args: Vec<A>,
        body: Atom,
        inline: bool,
    ) -> Result<(), EvaluationError> {
        let name = name.into();

        let (name, tags) = match name {
            Indeterminate::Symbol(name, _) => (name, vec![]),
            Indeterminate::Function(name, f) => {
                let tags = f
                    .as_fun_view()
                    .unwrap()
                    .iter()
                    .map(|x| x.to_owned())
                    .collect();
                (name, tags)
            }
        };

        self.add_tagged_function_impl(name, tags, args, body, inline)
    }

    /// Register a set of aliases (e.g. `s1 -> x^2+y`).
    pub fn add_aliases(
        &mut self,
        aliases: impl IntoIterator<Item = (Atom, Atom)>,
    ) -> Result<(), EvaluationError> {
        for (from, to) in aliases {
            self.add_function::<_, Indeterminate>(
                match &from {
                    Atom::Var(v) => {
                        Indeterminate::Symbol(v.get_symbol(), InlineVar::from(v.get_symbol()))
                    }
                    Atom::Fun(f) => Indeterminate::Function(f.get_symbol(), from),
                    _ => return Err(EvaluationError::NotIndeterminate { atom: from }),
                },
                vec![],
                to,
            )?;
        }
        Ok(())
    }

    /// Register a function, where the first arguments are `tags` instead of arguments.
    pub fn add_tagged_function<A: Into<Indeterminate>>(
        &mut self,
        name: Symbol,
        tags: Vec<Atom>,
        args: Vec<A>,
        body: Atom,
    ) -> Result<(), EvaluationError> {
        self.add_tagged_function_impl(name, tags, args, body, true)
    }

    /// Register a tagged function that is evaluated by a separate sub-evaluator instead of being
    /// inlined at every call site.
    pub fn add_tagged_function_no_inline<A: Into<Indeterminate>>(
        &mut self,
        name: Symbol,
        tags: Vec<Atom>,
        args: Vec<A>,
        body: Atom,
    ) -> Result<(), EvaluationError> {
        self.add_tagged_function_impl(name, tags, args, body, false)
    }

    fn add_tagged_function_impl<A: Into<Indeterminate>>(
        &mut self,
        name: Symbol,
        tags: Vec<Atom>,
        args: Vec<A>,
        body: Atom,
        inline: bool,
    ) -> Result<(), EvaluationError> {
        match self.tag.get(&name) {
            Some(&t) if t != tags.len() => {
                return Err(EvaluationError::InconsistentFunctionTagCount {
                    function: name,
                    expected: t,
                    actual: tags.len(),
                });
            }
            None => {
                self.tag.insert(name, tags.len());
            }
            _ => {}
        }

        let id = self.tagged_fn_map.len();
        let tag_len = tags.len();
        self.tagged_fn_map
            .entry((name, tags.clone()))
            .or_insert_with(|| Expr {
                id,
                tag_len,
                args: args.into_iter().map(|x| x.into()).collect(),
                body,
                inline,
            });

        Ok(())
    }

    pub(super) fn get_tag_len(&self, symbol: &Symbol) -> usize {
        self.tag.get(symbol).cloned().unwrap_or(0)
    }

    pub(super) fn get(&self, a: AtomView) -> Option<&Expr> {
        if let AtomView::Var(s) = a {
            return self.tagged_fn_map.get(&(s.get_symbol(), vec![]));
        } else if let AtomView::Fun(aa) = a {
            let s = aa.get_symbol();
            let tag_len = self.get_tag_len(&s);

            if aa.get_nargs() >= tag_len {
                let tag = aa.iter().take(tag_len).map(|x| x.to_owned()).collect();
                return self.tagged_fn_map.get(&(s, tag));
            }
        }

        None
    }

    pub(super) fn has_non_inlined_functions(&self) -> bool {
        self.tagged_fn_map.values().any(|expr| !expr.inline)
    }
}

/// Builder for creating an [`ExpressionEvaluator`]. Constructed using
/// [`AtomCore::evaluator_multiple`] or [`AtomCore::evaluator`].
///
/// Nested functions can be registerd using [`Self::add_function`] or [`Self::add_tagged_function`].
/// The [`Self::build`] method finalizes the builder and returns an [`ExpressionEvaluator`].
///
/// # Examples
///
/// ```
/// use symbolica::prelude::*;
/// let params = vec![parse!("x")];
/// let mut evaluator = parse!("f(x)")
///     .evaluator(&params)
///     .add_function(symbol!("f"), vec![symbol!("x")], parse!("x^2 + 1"))?
///     .build()?
///     .map_coeff(&|x| x.re.to_f64());
/// assert_eq!(evaluator.evaluate_single(&[2.0]), 5.0);
/// # Ok::<(), EvaluationError>(())
/// ```
#[derive(Clone, Debug)]
pub struct EvaluatorBuilder<'a> {
    exprs: Vec<AtomView<'a>>,
    fn_map: FunctionMap,
    params: Vec<Atom>,
    optimization_settings: OptimizationSettings,
}

impl<'a> EvaluatorBuilder<'a> {
    pub(crate) fn new<A: AtomCore>(expr: AtomView<'a>, params: &[A]) -> Self {
        Self {
            exprs: vec![expr],
            fn_map: FunctionMap::new(),
            params: params.iter().map(|p| p.as_atom_view().to_owned()).collect(),
            optimization_settings: OptimizationSettings::default(),
        }
    }

    pub(crate) fn new_multiple_views<A: AtomCore>(exprs: &[AtomView<'a>], params: &[A]) -> Self {
        Self {
            exprs: exprs.to_vec(),
            fn_map: FunctionMap::new(),
            params: params.iter().map(|p| p.as_atom_view().to_owned()).collect(),
            optimization_settings: OptimizationSettings::default(),
        }
    }

    pub(crate) fn new_multiple<E: AtomCore, A: AtomCore>(exprs: &'a [E], params: &[A]) -> Self {
        Self {
            exprs: exprs.iter().map(|e| e.as_atom_view()).collect(),
            fn_map: FunctionMap::new(),
            params: params.iter().map(|p| p.as_atom_view().to_owned()).collect(),
            optimization_settings: OptimizationSettings::default(),
        }
    }

    /// Set the function map.
    pub fn function_map(mut self, fn_map: FunctionMap) -> Self {
        self.fn_map = fn_map;
        self
    }

    /// Register a function. If `name` is a symbol, it will be treated as a regular function; if it is a function, its arguments will be treated as tags.
    pub fn add_function<S: Into<Indeterminate>, A: Into<Indeterminate>>(
        mut self,
        name: S,
        args: Vec<A>,
        body: Atom,
    ) -> Result<Self, EvaluationError> {
        self.fn_map.add_function(name, args, body)?;
        Ok(self)
    }

    /// Register a function that is evaluated by a separate sub-evaluator instead of being
    /// inlined at every call site.
    pub fn add_function_no_inline<S: Into<Indeterminate>, A: Into<Indeterminate>>(
        mut self,
        name: S,
        args: Vec<A>,
        body: Atom,
    ) -> Result<Self, EvaluationError> {
        self.fn_map.add_function_no_inline(name, args, body)?;
        Ok(self)
    }

    /// Register a function, where the first arguments are `tags` instead of arguments.
    pub fn add_tagged_function<A: Into<Indeterminate>>(
        mut self,
        name: Symbol,
        tags: Vec<Atom>,
        args: Vec<A>,
        body: Atom,
    ) -> Result<Self, EvaluationError> {
        self.fn_map.add_tagged_function(name, tags, args, body)?;
        Ok(self)
    }

    /// Register a tagged function that is evaluated by a separate sub-evaluator instead of being
    /// inlined at every call site.
    pub fn add_tagged_function_no_inline<A: Into<Indeterminate>>(
        mut self,
        name: Symbol,
        tags: Vec<Atom>,
        args: Vec<A>,
        body: Atom,
    ) -> Result<Self, EvaluationError> {
        self.fn_map
            .add_tagged_function_no_inline(name, tags, args, body)?;
        Ok(self)
    }

    /// Register a set of aliases (e.g. `s1 -> x^2+y`).
    pub fn add_aliases(
        mut self,
        aliases: impl IntoIterator<Item = (Atom, Atom)>,
    ) -> Result<Self, EvaluationError> {
        self.fn_map.add_aliases(aliases)?;
        Ok(self)
    }

    /// Set all optimization settings at once.
    pub fn optimization_settings(mut self, optimization_settings: OptimizationSettings) -> Self {
        self.optimization_settings = optimization_settings;
        self
    }

    /// Set the number of Horner scheme optimization iterations.
    pub fn horner_iterations(mut self, horner_iterations: usize) -> Self {
        self.optimization_settings.horner_iterations = horner_iterations;
        self
    }

    /// Set the number of CPU cores to use during optimization.
    pub fn cores(mut self, n_cores: usize) -> Self {
        self.optimization_settings.n_cores = n_cores;
        self
    }

    /// Set the number of common pair elimination iterations.
    pub fn cpe_iterations(mut self, cpe_iterations: Option<usize>) -> Self {
        self.optimization_settings.cpe_iterations = cpe_iterations;
        self
    }

    /// Set a hot-start expression list for optimization.
    pub fn hot_start(mut self, hot_start: Option<Vec<Expression<Complex<Rational>>>>) -> Self {
        self.optimization_settings.hot_start = hot_start;
        self
    }

    /// Set the abort check used during optimization.
    pub fn abort_check(mut self, abort_check: Option<Box<dyn AbortCheck>>) -> Self {
        self.optimization_settings.abort_check = abort_check;
        self
    }

    /// Set the abort polling level.
    pub fn abort_level(mut self, abort_level: usize) -> Self {
        self.optimization_settings.abort_level = abort_level;
        self
    }

    /// Set the maximum number of variables considered for a Horner scheme.
    pub fn max_horner_scheme_variables(mut self, max_horner_scheme_variables: usize) -> Self {
        self.optimization_settings.max_horner_scheme_variables = max_horner_scheme_variables;
        self
    }

    /// Set the maximum number of common-pair cache entries.
    pub fn max_common_pair_cache_entries(mut self, max_common_pair_cache_entries: usize) -> Self {
        self.optimization_settings.max_common_pair_cache_entries = max_common_pair_cache_entries;
        self
    }

    /// Set the maximum distance considered for common-pair elimination.
    pub fn max_common_pair_distance(mut self, max_common_pair_distance: usize) -> Self {
        self.optimization_settings.max_common_pair_distance = max_common_pair_distance;
        self
    }

    /// Enable or disable verbose optimization output.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.optimization_settings.verbose = verbose;
        self
    }

    /// Enable or disable direct translation.
    pub fn direct_translation(mut self, direct_translation: bool) -> Self {
        self.optimization_settings.direct_translation = direct_translation;
        self
    }

    /// Build the evaluator.
    pub fn build(self) -> Result<ExpressionEvaluator<Complex<Rational>>, EvaluationError> {
        // EvalTree::linearize expands every function call, so preserving an explicit call boundary
        // requires the direct translator even when the legacy tree optimizer was requested.
        if self.optimization_settings.direct_translation || self.fn_map.has_non_inlined_functions()
        {
            AtomView::to_evaluator(
                &self.exprs,
                &self.fn_map,
                &self.params,
                self.optimization_settings,
            )
        } else {
            let tree = AtomView::to_eval_tree_multiple(&self.exprs, &self.fn_map, &self.params)?;
            Ok(tree.optimize(&self.optimization_settings))
        }
    }
}

#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::Encode),
    derive(bincode_trait_derive::Decode),
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
    trait_decode(trait = crate::state::HasStateMap)
)]
#[derive(Clone, Debug)]
pub(super) struct Expr {
    pub(super) id: usize,
    pub(super) tag_len: usize,
    pub(super) args: Vec<Indeterminate>,
    pub(super) body: Atom,
    pub(super) inline: bool,
}

/// Settings for optimizing the evaluation of expressions.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone)]
pub struct OptimizationSettings {
    pub(crate) horner_iterations: usize,
    pub(crate) n_cores: usize,
    pub(crate) cpe_iterations: Option<usize>,
    pub(crate) hot_start: Option<Vec<Expression<Complex<Rational>>>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub(crate) abort_check: Option<Box<dyn AbortCheck>>,
    pub(crate) abort_level: usize,
    pub(crate) max_horner_scheme_variables: usize,
    pub(crate) max_common_pair_cache_entries: usize,
    pub(crate) max_common_pair_distance: usize,
    pub(crate) verbose: bool,
    pub(crate) direct_translation: bool,
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(OptimizationSettings);

#[cfg(feature = "bincode")]
impl bincode::Encode for OptimizationSettings {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.horner_iterations, encoder)?;
        bincode::Encode::encode(&self.n_cores, encoder)?;
        bincode::Encode::encode(&self.cpe_iterations, encoder)?;
        bincode::Encode::encode(&self.hot_start, encoder)?;
        bincode::Encode::encode(&self.max_horner_scheme_variables, encoder)?;
        bincode::Encode::encode(&self.max_common_pair_cache_entries, encoder)?;
        bincode::Encode::encode(&self.max_common_pair_distance, encoder)?;
        bincode::Encode::encode(&self.verbose, encoder)?;
        bincode::Encode::encode(&self.direct_translation, encoder)?;
        Ok(())
    }
}

#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for OptimizationSettings {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> core::result::Result<Self, bincode::error::DecodeError> {
        Ok(Self {
            horner_iterations: bincode::Decode::decode(decoder)?,
            n_cores: bincode::Decode::decode(decoder)?,
            cpe_iterations: bincode::Decode::decode(decoder)?,
            hot_start: bincode::Decode::decode(decoder)?,
            abort_check: None,
            abort_level: 0,
            max_horner_scheme_variables: bincode::Decode::decode(decoder)?,
            max_common_pair_cache_entries: bincode::Decode::decode(decoder)?,
            max_common_pair_distance: bincode::Decode::decode(decoder)?,
            verbose: bincode::Decode::decode(decoder)?,
            direct_translation: bincode::Decode::decode(decoder)?,
        })
    }
}

impl std::fmt::Debug for OptimizationSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("OptimizationSettings")
            .field("horner_iterations", &self.horner_iterations)
            .field("n_cores", &self.n_cores)
            .field("cpe_iterations", &self.cpe_iterations)
            .field("hot_start", &self.hot_start)
            .field("abort_check", &self.abort_check.is_some())
            .field("abort_level", &self.abort_level)
            .field("verbose", &self.verbose)
            .finish()
    }
}

impl Default for OptimizationSettings {
    fn default() -> Self {
        OptimizationSettings {
            horner_iterations: 10,
            n_cores: 1,
            cpe_iterations: None,
            hot_start: None,
            abort_check: None,
            abort_level: 0,
            max_horner_scheme_variables: 500,
            max_common_pair_cache_entries: 1_000_000,
            max_common_pair_distance: 1000,
            verbose: false,
            direct_translation: true,
        }
    }
}

impl OptimizationSettings {
    /// Create optimization settings with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of Horner scheme optimization iterations.
    pub fn horner_iterations(mut self, horner_iterations: usize) -> Self {
        self.horner_iterations = horner_iterations;
        self
    }

    /// Set the number of CPU cores to use during optimization.
    pub fn cores(mut self, n_cores: usize) -> Self {
        self.n_cores = n_cores;
        self
    }

    /// Set the number of common pair elimination iterations.
    pub fn cpe_iterations(mut self, cpe_iterations: Option<usize>) -> Self {
        self.cpe_iterations = cpe_iterations;
        self
    }

    /// Set a hot-start expression list for optimization.
    pub fn hot_start(mut self, hot_start: Option<Vec<Expression<Complex<Rational>>>>) -> Self {
        self.hot_start = hot_start;
        self
    }

    /// Set the abort check used during optimization.
    pub fn abort_check(mut self, abort_check: Option<Box<dyn AbortCheck>>) -> Self {
        self.abort_check = abort_check;
        self
    }

    /// Set the abort polling level.
    pub fn abort_level(mut self, abort_level: usize) -> Self {
        self.abort_level = abort_level;
        self
    }

    /// Set the maximum number of variables considered for a Horner scheme.
    pub fn max_horner_scheme_variables(mut self, max_horner_scheme_variables: usize) -> Self {
        self.max_horner_scheme_variables = max_horner_scheme_variables;
        self
    }

    /// Set the maximum number of common-pair cache entries.
    pub fn max_common_pair_cache_entries(mut self, max_common_pair_cache_entries: usize) -> Self {
        self.max_common_pair_cache_entries = max_common_pair_cache_entries;
        self
    }

    /// Set the maximum distance considered for common-pair elimination.
    pub fn max_common_pair_distance(mut self, max_common_pair_distance: usize) -> Self {
        self.max_common_pair_distance = max_common_pair_distance;
        self
    }

    /// Enable or disable verbose optimization output.
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Enable or disable direct translation.
    pub fn direct_translation(mut self, direct_translation: bool) -> Self {
        self.direct_translation = direct_translation;
        self
    }
}
