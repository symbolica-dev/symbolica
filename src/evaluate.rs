//! Evaluation of expressions.
//!
//! The main entry point is through [AtomCore::evaluator].
use ahash::{AHasher, HashMap, HashMapExt, HashSet};
use dyn_clone::DynClone;
use rand::Rng;
use self_cell::self_cell;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, hash_map::Entry},
    hash::{Hash, Hasher},
    os::raw::{c_ulong, c_void},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};
use symjit::{Applet, Config, Defuns, Translator};

use crate::{
    LicenseManager,
    atom::{Atom, AtomCore, AtomView, Indeterminate, KeyLookup, Symbol},
    coefficient::CoefficientView,
    combinatorics::unique_permutations,
    domains::{
        InternalOrdering,
        dual::DualNumberStructure,
        float::{
            Complex, Constructible, ErrorPropagatingFloat, F64, Float, FloatLike, Real, RealLike,
            SingleFloat,
        },
        integer::Integer,
        rational::Rational,
    },
    error,
    id::ConditionResult,
    info,
    numerical_integration::MonteCarloRng,
    state::State,
    utils::AbortCheck,
};

type EvalFnType<A, T> = Box<
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
/// use symbolica::{atom::AtomCore, parse, symbol};
/// use symbolica::evaluate::{FunctionMap, OptimizationSettings};
/// let mut fn_map = FunctionMap::new();
/// fn_map.add_function(symbol!("f"), "f".to_string(), vec![symbol!("x")], parse!("x^2 + 1")).unwrap();
///
/// let optimization_settings = OptimizationSettings::default();
/// let mut evaluator = parse!("f(x)")
///     .evaluator(&fn_map, &vec![parse!("x")], optimization_settings)
///     .unwrap().map_coeff(&|x| x.re.to_f64());
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
pub struct FunctionMap<T = Complex<Rational>> {
    map: HashMap<Atom, ConstOrExpr<T>>,
    tagged_fn_map: HashMap<(Symbol, Vec<Atom>), ConstOrExpr<T>>,
    external_fn: HashMap<Symbol, ConstOrExpr<T>>,
    tag: HashMap<Symbol, usize>,
}

impl<T> Default for FunctionMap<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> FunctionMap<T> {
    /// Create a new, empty function map.
    pub fn new() -> Self {
        FunctionMap {
            map: HashMap::default(),
            tagged_fn_map: HashMap::default(),
            tag: HashMap::default(),
            external_fn: HashMap::default(),
        }
    }

    /// Register a constant.
    pub fn add_constant(&mut self, key: Atom, value: T) {
        self.map.insert(key, ConstOrExpr::Const(value));
    }

    /// Register a function without tags. `rename` is the name used in exported code.
    pub fn add_function<A: Into<Indeterminate>>(
        &mut self,
        name: Symbol,
        rename: String,
        args: Vec<A>,
        body: Atom,
    ) -> Result<(), String> {
        if self.external_fn.contains_key(&name) {
            return Err(format!(
                "Cannot add function {}, as it is also an external function",
                name.get_name()
            ));
        }

        if let Some(t) = self.tag.insert(name, 0)
            && t != 0
        {
            return Err(format!(
                "Cannot add the same function {} with a different number of parameters",
                name.get_name()
            ));
        }

        let id = self.tagged_fn_map.len();
        self.tagged_fn_map.entry((name, vec![])).or_insert_with(|| {
            ConstOrExpr::Expr(Expr {
                id,
                name: rename,
                tag_len: 0,
                args: args.into_iter().map(|x| x.into()).collect(),
                body,
            })
        });

        Ok(())
    }

    /// Register a function, where the first arguments are `tags` instead of arguments. `rename` is the name used in exported code.
    pub fn add_tagged_function<A: Into<Indeterminate>>(
        &mut self,
        name: Symbol,
        tags: Vec<Atom>,
        rename: String,
        args: Vec<A>,
        body: Atom,
    ) -> Result<(), String> {
        if self.external_fn.contains_key(&name) {
            return Err(format!(
                "Cannot add function {}, as it is also an external function",
                name.get_name()
            ));
        }

        if let Some(t) = self.tag.insert(name, tags.len())
            && t != tags.len()
        {
            return Err(format!(
                "Cannot add the same function {} with a different number of parameters",
                name.get_name()
            ));
        }

        let id = self.tagged_fn_map.len();
        let tag_len = tags.len();
        self.tagged_fn_map
            .entry((name, tags.clone()))
            .or_insert_with(|| {
                ConstOrExpr::Expr(Expr {
                    id,
                    name: rename,
                    tag_len,
                    args: args.into_iter().map(|x| x.into()).collect(),
                    body,
                })
            });

        Ok(())
    }

    /// Register an external function that can later be linked with [ExpressionEvaluator::with_external_functions].
    pub fn add_external_function(&mut self, name: Symbol, rename: String) -> Result<(), String> {
        if self.tag.contains_key(&name) || self.external_fn.contains_key(&name) {
            return Err(format!(
                "Cannot add external function {}, as it is also a tagged function",
                name.get_name()
            ));
        }

        self.external_fn
            .insert(name, ConstOrExpr::External(self.external_fn.len(), rename));

        Ok(())
    }

    /// Register a conditional function that consists of three arguments:
    /// - the condition that should be non-zero
    /// - the true branch
    /// - the false branch
    pub fn add_conditional(&mut self, name: Symbol) -> Result<(), String> {
        if self.external_fn.contains_key(&name) {
            return Err(format!(
                "Cannot add function {}, as it is also an external function",
                name.get_name()
            ));
        }

        if let Some(t) = self.tag.insert(name, 0)
            && t != 0
        {
            return Err(format!(
                "Cannot add the same function {} with a different number of parameters",
                name.get_name()
            ));
        }

        self.tagged_fn_map
            .insert((name, vec![]), ConstOrExpr::Condition);

        Ok(())
    }

    fn get_tag_len(&self, symbol: &Symbol) -> usize {
        self.tag.get(symbol).cloned().unwrap_or(0)
    }

    fn get_constant(&self, a: AtomView) -> Option<&T> {
        match self.map.get(a.get_data()) {
            Some(ConstOrExpr::Const(c)) => Some(c),
            _ => None,
        }
    }

    fn get(&self, a: AtomView) -> Option<&ConstOrExpr<T>> {
        if let Some(c) = self.map.get(a.get_data()) {
            return Some(c);
        }

        if let AtomView::Fun(aa) = a {
            let s = aa.get_symbol();
            let tag_len = self.get_tag_len(&s);

            if let Some(s) = self.external_fn.get(&s) {
                return Some(s);
            }

            if aa.get_nargs() >= tag_len {
                let tag = aa.iter().take(tag_len).map(|x| x.to_owned()).collect();
                return self.tagged_fn_map.get(&(s, tag));
            }
        }

        None
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
enum ConstOrExpr<T> {
    Const(T),
    Expr(Expr),
    External(usize, String),
    Condition,
}

#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::Encode),
    derive(bincode_trait_derive::Decode),
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
    trait_decode(trait = crate::state::HasStateMap)
)]
#[derive(Clone, Debug)]
struct Expr {
    id: usize,
    name: String,
    tag_len: usize,
    args: Vec<Indeterminate>,
    body: Atom,
}

/// Settings for optimizing the evaluation of expressions.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone)]
pub struct OptimizationSettings {
    pub horner_iterations: usize,
    pub n_cores: usize,
    pub cpe_iterations: Option<usize>,
    pub hot_start: Option<Vec<Expression<Complex<Rational>>>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    pub abort_check: Option<Box<dyn AbortCheck>>,
    pub abort_level: usize,
    pub max_horner_scheme_variables: usize,
    pub max_common_pair_cache_entries: usize,
    pub max_common_pair_distance: usize,
    pub verbose: bool,
    pub direct_translation: bool,
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

#[derive(Debug, Clone)]
struct SplitExpression<T> {
    tree: Vec<Expression<T>>,
    subexpressions: Vec<Expression<T>>,
}

/// A tree representation of multiple expressions, including function definitions.
#[derive(Debug, Clone)]
pub struct EvalTree<T> {
    functions: Vec<(String, Vec<Indeterminate>, SplitExpression<T>)>,
    external_functions: Vec<String>,
    expressions: SplitExpression<T>,
    param_count: usize,
}

/// A built-in symbol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BuiltinSymbol(Symbol);

#[cfg(feature = "serde")]
impl serde::Serialize for BuiltinSymbol {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.get_id().serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for BuiltinSymbol {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let id: u32 = u32::deserialize(deserializer)?;
        Ok(BuiltinSymbol(unsafe { State::symbol_from_id(id) }))
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for BuiltinSymbol {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        u32::encode(&self.0.get_id(), encoder)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(BuiltinSymbol);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for BuiltinSymbol {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let id: u32 = u32::decode(decoder)?;
        Ok(BuiltinSymbol(unsafe { State::symbol_from_id(id) }))
    }
}

impl BuiltinSymbol {
    pub fn get_symbol(&self) -> Symbol {
        self.0
    }
}

impl<'a> AtomView<'a> {
    pub(crate) fn to_evaluator(
        expressions: &[Self],
        fn_map: &FunctionMap<Complex<Rational>>,
        params: &[Atom],
        settings: OptimizationSettings,
    ) -> Result<ExpressionEvaluator<Complex<Rational>>, String> {
        if settings.verbose {
            let mut cse = HashSet::default();
            let (mut n_add, mut n_mul) = (0, 0);
            for e in expressions {
                let (add, mul) = e.count_operations_with_subexpressions(&mut cse);
                n_add += add;
                n_mul += mul;
            }
            info!(
                "Initial ops: {} additions and {} multiplications",
                n_add, n_mul
            );
        }

        if settings.horner_iterations == 0 {
            return Self::linearize_multiple(expressions, fn_map, params, settings);
        }

        let v = match &settings.hot_start {
            Some(_) => {
                return Err(
                    "Hot start not supported before the deprecation of Expression".to_owned(),
                );
            }
            None => {
                // start with an occurence order Horner scheme
                let mut v = HashMap::default();

                for t in expressions {
                    t.count_indeterminates(true, &mut v);
                }

                let mut v: Vec<_> = v.into_iter().collect();
                v.retain(|(_, vv)| *vv > 1);
                v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                v.truncate(settings.max_horner_scheme_variables);
                v.into_iter()
                    .map(|(k, _)| Indeterminate::try_from(k.to_owned()).unwrap())
                    .collect::<Vec<_>>()
            }
        };

        let scheme = if settings.horner_iterations > 1 {
            Self::optimize_horner_scheme_multiple(expressions, &v, &settings)
        } else {
            v
        };

        let hornered_expressions = expressions
            .iter()
            .map(|x| x.horner_scheme(Some(&scheme), true))
            .collect::<Vec<_>>();

        if settings.horner_iterations == 1 && settings.verbose {
            let mut cse = HashSet::default();
            let (mut n_add, mut n_mul) = (0, 0);
            for e in expressions {
                let (add, mul) = e.count_operations_with_subexpressions(&mut cse);
                n_add += add;
                n_mul += mul;
            }
            info!(
                "Horner scheme ops: {} additions and {} multiplications",
                n_add, n_mul
            );
        }

        let mut f = fn_map.clone();
        for expr in f.tagged_fn_map.values_mut() {
            if let ConstOrExpr::Expr(Expr { body, .. }) = expr {
                *body = body.as_view().horner_scheme(Some(&scheme), true);
            }
        }

        let mut e = Self::linearize_multiple(&hornered_expressions, fn_map, params, settings)?;

        drop(f);
        drop(hornered_expressions);

        loop {
            let r = e.remove_common_instructions();

            if r == 0 || e.settings.abort_level > 0 {
                e.settings.abort_level = 0;
                break;
            }

            if e.settings.verbose {
                let (add_count, mul_count) = e.count_operations();
                info!(
                    "Removed {} common instructions: {} + and {} ×",
                    r, add_count, mul_count
                );
            }
        }

        for _ in 0..e.settings.cpe_iterations.unwrap_or(usize::MAX) {
            let r = e.remove_common_pairs();
            if r == 0 || e.settings.abort_level > 0 {
                e.settings.abort_level = 0;
                break;
            }

            if e.settings.verbose {
                let (add_count, mul_count) = e.count_operations();
                info!(
                    "Removed {} common pairs: {} + and {} ×",
                    r, add_count, mul_count
                );
            }
        }

        e.optimize_stack();
        Ok(e)
    }

    pub fn optimize_horner_scheme_multiple(
        expressions: &[Self],
        vars: &[Indeterminate],
        settings: &OptimizationSettings,
    ) -> Vec<Indeterminate> {
        if vars.is_empty() {
            return vars.to_vec();
        }

        let horner: Vec<_> = expressions
            .iter()
            .map(|x| x.horner_scheme(Some(&vars), true))
            .collect();
        let mut subexpr = HashSet::default();
        let mut best_ops = (0, 0);
        for h in &horner {
            let ops = h
                .as_view()
                .count_operations_with_subexpressions(&mut subexpr);
            best_ops = (best_ops.0 + ops.0, best_ops.1 + ops.1);
        }

        if settings.verbose {
            info!(
                "Initial Horner scheme ops: {} additions and {} multiplications",
                best_ops.0, best_ops.1
            );
        }

        let best_mul = Arc::new(AtomicUsize::new(best_ops.1));
        let best_add = Arc::new(AtomicUsize::new(best_ops.0));
        let best_scheme = Arc::new(Mutex::new(vars.to_vec()));

        let n_iterations = settings.horner_iterations.max(1) - 1;

        let permutations = if vars.len() < 10
            && Integer::factorial(vars.len() as u32) <= settings.horner_iterations.max(1)
        {
            let v: Vec<_> = (0..vars.len()).collect();
            Some(unique_permutations(&v).1)
        } else {
            None
        };
        let p_ref = &permutations;

        let n_cores = if LicenseManager::is_licensed() {
            settings.n_cores
        } else {
            1
        }
        .min(n_iterations);

        std::thread::scope(|s| {
            let abort = Arc::new(AtomicBool::new(false));

            for i in 0..n_cores {
                let mut rng = MonteCarloRng::new(0, i);

                let mut cvars = vars.to_vec();
                let best_scheme = best_scheme.clone();
                let best_mul = best_mul.clone();
                let best_add = best_add.clone();
                let mut last_mul = usize::MAX;
                let mut last_add = usize::MAX;
                let abort = abort.clone();

                let mut op = move || {
                    for j in 0..n_iterations / n_cores {
                        if abort.load(Ordering::Relaxed) {
                            return;
                        }

                        if i == n_cores - 1
                            && let Some(a) = &settings.abort_check
                            && a()
                        {
                            abort.store(true, Ordering::Relaxed);

                            if settings.verbose {
                                info!(
                                    "Aborting Horner optimization at step {}/{}.",
                                    j,
                                    settings.horner_iterations / n_cores
                                );
                            }

                            return;
                        }

                        // try a random swap
                        let mut t1 = 0;
                        let mut t2 = 0;

                        if let Some(p) = p_ref {
                            if j >= p.len() / n_cores {
                                break;
                            }

                            let perm = &p[i * (p.len() / n_cores) + j];
                            cvars = perm.iter().map(|x| vars[*x].clone()).collect();
                        } else {
                            t1 = rng.random_range(0..cvars.len());
                            t2 = rng.random_range(0..cvars.len() - 1);

                            cvars.swap(t1, t2);
                        }

                        let horner: Vec<_> = expressions
                            .iter()
                            .map(|x| x.horner_scheme(Some(&cvars), true))
                            .collect();
                        let mut subexpr = HashSet::default();
                        let mut cur_ops = (0, 0);

                        for h in &horner {
                            let ops = h
                                .as_view()
                                .count_operations_with_subexpressions(&mut subexpr);
                            cur_ops = (cur_ops.0 + ops.0, cur_ops.1 + ops.1);
                        }

                        // prefer fewer multiplications
                        if cur_ops.1 <= last_mul || cur_ops.1 == last_mul && cur_ops.0 <= last_add {
                            if settings.verbose {
                                info!(
                                    "Accept move at step {}/{}: {} + and {} ×",
                                    j,
                                    settings.horner_iterations / n_cores,
                                    cur_ops.0,
                                    cur_ops.1
                                );
                            }

                            last_add = cur_ops.0;
                            last_mul = cur_ops.1;

                            if cur_ops.1 <= best_mul.load(Ordering::Relaxed)
                                || cur_ops.1 == best_mul.load(Ordering::Relaxed)
                                    && cur_ops.0 <= best_add.load(Ordering::Relaxed)
                            {
                                let mut best_scheme = best_scheme.lock().unwrap();

                                // check again if it is the best now that we have locked
                                let best_mul_l = best_mul.load(Ordering::Relaxed);
                                let best_add_l = best_add.load(Ordering::Relaxed);
                                if cur_ops.1 <= best_mul_l
                                    || cur_ops.1 == best_mul_l && cur_ops.0 <= best_add_l
                                {
                                    if cur_ops.0 == best_add_l && cur_ops.1 == best_mul_l {
                                        if *best_scheme < cvars {
                                            // on a draw, accept the lexicographical minimum
                                            // to get a deterministic scheme
                                            *best_scheme = cvars.clone();
                                        }
                                    } else {
                                        best_mul.store(cur_ops.1, Ordering::Relaxed);
                                        best_add.store(cur_ops.0, Ordering::Relaxed);
                                        *best_scheme = cvars.clone();
                                    }
                                }
                            }
                        } else {
                            cvars.swap(t1, t2);
                        }
                    }
                };

                if i + 1 < n_cores {
                    s.spawn(op);
                } else {
                    // execute in the main thread and do the abort check on the main thread
                    // this helps with catching ctrl-c
                    op()
                }
            }
        });

        if settings.verbose {
            info!(
                "Final scheme: {} + and {} ×",
                best_add.load(Ordering::Relaxed),
                best_mul.load(Ordering::Relaxed)
            );
        }

        Arc::try_unwrap(best_scheme).unwrap().into_inner().unwrap()
    }

    pub(crate) fn linearize_multiple<T: AtomCore>(
        expressions: &[T],
        fn_map: &FunctionMap<Complex<Rational>>,
        params: &[Atom],
        settings: OptimizationSettings,
    ) -> Result<ExpressionEvaluator<Complex<Rational>>, String> {
        let mut constants = Vec::new();
        let mut constant_map = HashMap::new();
        let mut instr = Vec::new();

        // we can only safely remove entries that don't depend on any of the function arguments
        let mut subexpression: HashMap<AtomView, Slot> = HashMap::default();

        let mut external_functions = fn_map
            .external_fn
            .values()
            .map(|v| {
                if let ConstOrExpr::External(id, name) = v {
                    (*id, name.clone())
                } else {
                    unreachable!()
                }
            })
            .collect::<Vec<_>>();
        external_functions.sort_by_key(|(id, _)| *id);
        let external_functions = external_functions
            .into_iter()
            .map(|(_, name)| name)
            .collect::<Vec<_>>();

        let mut result_indices = vec![];
        let mut arg_stack = vec![];
        for expr in expressions {
            let res = expr.as_atom_view().linearize_impl(
                fn_map,
                params,
                &mut constants,
                &mut constant_map,
                &mut instr,
                &mut subexpression,
                &mut arg_stack,
                0,
            )?;
            result_indices.push(res);
        }

        let reserved_indices = params.len() + constants.len();

        let mut stack = vec![Complex::default(); params.len() + constants.len() + instr.len()];
        for (s, c) in stack.iter_mut().skip(params.len()).zip(constants) {
            *s = c;
        }

        macro_rules! slot_map {
            ($s: expr) => {
                match $s {
                    Slot::Param(i) => i,
                    Slot::Const(i) => params.len() + i,
                    Slot::Temp(i) => reserved_indices + i,
                    Slot::Out(_) => unreachable!(),
                }
            };
        }

        let mut instructions = vec![];
        for i in instr {
            match i {
                Instruction::Add(o, args, _) => {
                    instructions.push((
                        Instr::Add(
                            slot_map!(o),
                            args.clone().into_iter().map(|x| slot_map!(x)).collect(),
                        ),
                        ComplexPhase::default(),
                    ));
                }
                Instruction::Mul(o, args, _) => {
                    instructions.push((
                        Instr::Mul(
                            slot_map!(o),
                            args.clone().into_iter().map(|x| slot_map!(x)).collect(),
                        ),
                        ComplexPhase::default(),
                    ));
                }
                Instruction::Pow(o, base, exp, _) => {
                    instructions.push((
                        Instr::Pow(slot_map!(o), slot_map!(base), exp),
                        ComplexPhase::default(),
                    ));
                }
                Instruction::Powf(o, base, exp, _) => {
                    instructions.push((
                        Instr::Powf(slot_map!(o), slot_map!(base), slot_map!(exp)),
                        ComplexPhase::default(),
                    ));
                }
                Instruction::Fun(o, sym, arg, _) => {
                    instructions.push((
                        Instr::BuiltinFun(slot_map!(o), sym.clone(), slot_map!(arg)),
                        ComplexPhase::default(),
                    ));
                }
                Instruction::ExternalFun(o, name, args) => {
                    instructions.push((
                        Instr::ExternalFun(
                            slot_map!(o),
                            external_functions.iter().position(|x| x == &name).unwrap(),
                            args.clone().into_iter().map(|x| slot_map!(x)).collect(),
                        ),
                        ComplexPhase::default(),
                    ));
                }
                Instruction::Assign(_, _) => {
                    unimplemented!("Assign should not occur in input")
                }
                Instruction::IfElse(cond, l) => {
                    instructions.push((
                        Instr::IfElse(slot_map!(cond), Label(l)),
                        ComplexPhase::default(),
                    ));
                }
                Instruction::Goto(label) => {
                    instructions.push((Instr::Goto(Label(label)), ComplexPhase::default()));
                }
                Instruction::Label(label) => {
                    instructions.push((Instr::Label(Label(label)), ComplexPhase::default()));
                }
                Instruction::Join(o, cond, t, f) => {
                    instructions.push((
                        Instr::Join(slot_map!(o), slot_map!(cond), slot_map!(t), slot_map!(f)),
                        ComplexPhase::default(),
                    ));
                }
            }
        }

        Ok(ExpressionEvaluator {
            stack,
            param_count: params.len(),
            reserved_indices,
            instructions: instructions,
            result_indices: result_indices.iter().map(|s| slot_map!(*s)).collect(),
            external_fns: external_functions,
            settings: settings.clone(),
        })
    }

    // Yields the stack index that contains the output.
    fn linearize_impl(
        &self,
        fn_map: &'a FunctionMap<Complex<Rational>>,
        params: &[Atom],
        constants: &mut Vec<Complex<Rational>>,
        constant_map: &mut HashMap<Complex<Rational>, usize>,
        instr: &mut Vec<Instruction>,
        subexpressions: &mut HashMap<AtomView<'a>, Slot>,
        args: &mut Vec<(AtomView<'a>, Slot)>,
        arg_start: usize,
    ) -> Result<Slot, String> {
        if matches!(*self, AtomView::Var(_) | AtomView::Fun(_)) {
            if let Some(p) = args.iter().skip(arg_start).find(|s| *self == s.0) {
                return Ok(p.1);
            }

            if let Some(p) = params.iter().position(|a| a.as_view() == *self) {
                return Ok(Slot::Param(p));
            }
        }

        if let Some(c) = fn_map.get_constant(*self) {
            if let Some(&i) = constant_map.get(&c) {
                return Ok(Slot::Const(i));
            }

            let i = constants.len();
            constants.push(c.clone());
            constant_map.insert(c.clone(), i);
            return Ok(Slot::Const(i));
        }

        if let Some(s) = subexpressions.get(self) {
            return Ok(*s);
        }

        let res = match self {
            AtomView::Num(n) => {
                let c = match n.get_coeff_view() {
                    CoefficientView::Natural(n, d, ni, di) => {
                        Complex::new(Rational::from((n, d)), Rational::from((ni, di)))
                    }
                    CoefficientView::Large(l, i) => Complex::new(l.to_rat(), i.to_rat()),
                    CoefficientView::Float(r, i) => {
                        // TODO: converting back to rational is slow
                        Complex::new(r.to_float().to_rational(), i.to_float().to_rational())
                    }
                    CoefficientView::Indeterminate => Err("Cannot convert indeterminate")?,
                    CoefficientView::Infinity(_) => Err("Cannot convert infinity")?,
                    CoefficientView::FiniteField(_, _) => {
                        Err("Finite field not yet supported for evaluation".to_string())?
                    }
                    CoefficientView::RationalPolynomial(_) => Err(
                        "Rational polynomial coefficient not yet supported for evaluation"
                            .to_string(),
                    )?,
                };

                if let Some(&i) = constant_map.get(&c) {
                    return Ok(Slot::Const(i));
                }

                let i = constants.len();
                constants.push(c.clone());
                constant_map.insert(c, i);
                Slot::Const(i)
            }
            AtomView::Var(v) => Err(format!(
                "Variable {} not in constant map",
                v.get_symbol().get_name()
            ))?,
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [
                    Symbol::EXP_ID,
                    Symbol::LOG_ID,
                    Symbol::SIN_ID,
                    Symbol::COS_ID,
                    Symbol::SQRT_ID,
                    Symbol::ABS_ID,
                    Symbol::CONJ_ID,
                ]
                .contains(&name.get_id())
                {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.linearize_impl(
                        fn_map,
                        params,
                        constants,
                        constant_map,
                        instr,
                        subexpressions,
                        args,
                        arg_start,
                    )?;

                    let temp = Slot::Temp(instr.len());
                    let c = Instruction::Fun(temp, BuiltinSymbol(name), arg_eval, false);
                    instr.push(c);

                    subexpressions.insert(*self, temp);
                    return Ok(temp);
                }

                let fun = if name == Symbol::IF {
                    &ConstOrExpr::Condition
                } else if let Some(fun) = fn_map.get(*self) {
                    fun
                } else {
                    return Err(format!("Undefined function {}", self.to_plain_string()));
                };

                match fun {
                    ConstOrExpr::Const(c) => {
                        if let Some(&i) = constant_map.get(&c) {
                            return Ok(Slot::Const(i));
                        }

                        let i = constants.len();
                        constants.push(c.clone());
                        constant_map.insert(c.clone(), i);
                        Slot::Const(i)
                    }
                    ConstOrExpr::External(_e, name) => {
                        let eval_args = f
                            .iter()
                            .map(|arg| {
                                arg.linearize_impl(
                                    fn_map,
                                    params,
                                    constants,
                                    constant_map,
                                    instr,
                                    subexpressions,
                                    args,
                                    arg_start,
                                )
                            })
                            .collect::<Result<_, _>>()?;

                        let temp = Slot::Temp(instr.len());
                        instr.push(Instruction::ExternalFun(temp, name.clone(), eval_args));
                        temp
                    }
                    ConstOrExpr::Expr(Expr {
                        tag_len,
                        args: arg_spec,
                        body: e,
                        ..
                    }) => {
                        if f.get_nargs() != arg_spec.len() + tag_len {
                            return Err(format!(
                                "Function {} called with wrong number of arguments: {} vs {}",
                                f.get_symbol().get_name(),
                                f.get_nargs(),
                                arg_spec.len() + tag_len
                            ));
                        }

                        let old_arg_stack_len = args.len();

                        let mut arg_shadowed = false;
                        for (eval_arg, arg_spec) in f.iter().skip(*tag_len).zip(arg_spec) {
                            let slot = eval_arg.linearize_impl(
                                fn_map,
                                params,
                                constants,
                                constant_map,
                                instr,
                                subexpressions,
                                args,
                                arg_start,
                            )?;

                            if args.iter().any(|(a, _)| *a == arg_spec.as_view()) {
                                arg_shadowed = true;
                            }

                            args.push((arg_spec.as_view(), slot));
                        }

                        // inline function call
                        // we have to use a new subexpression list as the function has arguments that may be different per call
                        // this means that not all subexpressions will be shared across calls
                        let mut sub_expr_pos_child = HashMap::default();
                        let r = e.as_view().linearize_impl(
                            fn_map,
                            params,
                            constants,
                            constant_map,
                            instr,
                            if old_arg_stack_len == args.len() {
                                subexpressions
                            } else {
                                // we can only inherit the subexpressions if the new function argument symbols
                                // have not been used earlier
                                if !arg_shadowed {
                                    sub_expr_pos_child.clone_from(subexpressions);
                                }

                                &mut sub_expr_pos_child
                            },
                            args,
                            old_arg_stack_len,
                        )?;

                        args.truncate(old_arg_stack_len);

                        r
                    }
                    ConstOrExpr::Condition => {
                        if f.get_nargs() != 3 {
                            return Err(format!(
                                "Condition function called with wrong number of arguments: {} vs 3",
                                f.get_nargs(),
                            ));
                        }

                        let mut arg_iter = f.iter();
                        let cond = arg_iter.next().unwrap();
                        let then_branch = arg_iter.next().unwrap();
                        let else_branch = arg_iter.next().unwrap();

                        let instr_len = instr.len();
                        let subexpression_len = subexpressions.len();
                        let cond = cond.linearize_impl(
                            fn_map,
                            params,
                            constants,
                            constant_map,
                            instr,
                            subexpressions,
                            args,
                            arg_start,
                        )?;

                        // try to resolve the condition if it is fully numeric
                        fn resolve(
                            cond: Slot,
                            instr: &[Instruction],
                            constants: &[Complex<Rational>],
                        ) -> Option<Complex<Rational>> {
                            let i = match cond {
                                Slot::Param(_) => {
                                    return None;
                                }
                                Slot::Const(i) => {
                                    return Some(constants[i].clone());
                                }
                                Slot::Temp(t) => t,
                                Slot::Out(_) => {
                                    unreachable!()
                                }
                            };

                            match &instr[i] {
                                Instruction::Add(_, args, _) => {
                                    let mut res = Complex::default();
                                    for x in args {
                                        match resolve(*x, instr, constants) {
                                            Some(v) => res += v,
                                            None => return None,
                                        }
                                    }

                                    Some(res)
                                }
                                Instruction::Mul(_, args, _) => {
                                    let mut res = Complex::new(Rational::one(), Rational::zero());
                                    for x in args {
                                        match resolve(*x, instr, constants) {
                                            Some(v) => res *= v,
                                            None => return None,
                                        }
                                    }

                                    Some(res)
                                }
                                Instruction::Pow(_, base, exp, _) => {
                                    if let Some(base_val) = resolve(*base, instr, constants) {
                                        if *exp < 0 {
                                            Some(base_val.pow(exp.unsigned_abs()).inv())
                                        } else {
                                            Some(base_val.pow(exp.unsigned_abs()))
                                        }
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            }
                        }

                        if let Some(cond_res) = resolve(cond, instr, constants) {
                            // remove dead code
                            instr.truncate(instr_len);
                            if subexpression_len != subexpressions.len() {
                                // remove subexpressions that are created as part of the conditions
                                subexpressions.retain(|_, &mut v| {
                                    if let Slot::Temp(v) = v {
                                        v < instr_len
                                    } else {
                                        true
                                    }
                                });
                            }

                            let res = if !cond_res.is_zero() {
                                then_branch.linearize_impl(
                                    fn_map,
                                    params,
                                    constants,
                                    constant_map,
                                    instr,
                                    subexpressions,
                                    args,
                                    arg_start,
                                )?
                            } else {
                                else_branch.linearize_impl(
                                    fn_map,
                                    params,
                                    constants,
                                    constant_map,
                                    instr,
                                    subexpressions,
                                    args,
                                    arg_start,
                                )?
                            };

                            subexpressions.insert(*self, res);
                            return Ok(res);
                        }

                        let if_instr_pos = instr.len();
                        instr.push(Instruction::IfElse(cond, 0));

                        let mut sub_expr_pos_child = subexpressions.clone(); // TODO: prevent clone?
                        let then_branch = then_branch.linearize_impl(
                            fn_map,
                            params,
                            constants,
                            constant_map,
                            instr,
                            &mut sub_expr_pos_child,
                            args,
                            arg_start,
                        )?;

                        let label_end_pos = instr.len();
                        instr.push(Instruction::Goto(0));
                        instr[if_instr_pos] = Instruction::IfElse(cond, instr.len());
                        instr.push(Instruction::Label(instr.len()));

                        sub_expr_pos_child.clone_from(&subexpressions);
                        let else_branch = else_branch.linearize_impl(
                            fn_map,
                            params,
                            constants,
                            constant_map,
                            instr,
                            &mut sub_expr_pos_child,
                            args,
                            arg_start,
                        )?;

                        instr[label_end_pos] = Instruction::Goto(instr.len());
                        instr.push(Instruction::Label(instr.len()));

                        let temp = Slot::Temp(instr.len());
                        instr.push(Instruction::Join(temp, cond, then_branch, else_branch));
                        temp
                    }
                }
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let b_eval = b.linearize_impl(
                    fn_map,
                    params,
                    constants,
                    constant_map,
                    instr,
                    subexpressions,
                    args,
                    arg_start,
                )?;

                if let AtomView::Num(n) = e
                    && let CoefficientView::Natural(num, den, num_i, _den_i) = n.get_coeff_view()
                    && den == 1
                    && num_i == 0
                {
                    let new_base = if num.unsigned_abs() > 1 {
                        let temp = Slot::Temp(instr.len());
                        instr.push(Instruction::Mul(
                            temp,
                            vec![b_eval; num.unsigned_abs() as usize],
                            0,
                        ));
                        temp
                    } else {
                        b_eval
                    };

                    let res = if num > 0 {
                        new_base
                    } else {
                        let temp = Slot::Temp(instr.len());
                        instr.push(Instruction::Pow(temp, new_base, -1, false));
                        temp
                    };

                    subexpressions.insert(*self, res);
                    return Ok(res);
                }

                let e_eval = e.linearize_impl(
                    fn_map,
                    params,
                    constants,
                    constant_map,
                    instr,
                    subexpressions,
                    args,
                    arg_start,
                )?;

                let temp = Slot::Temp(instr.len());
                instr.push(Instruction::Powf(temp, b_eval, e_eval, false));
                temp
            }
            AtomView::Mul(m) => {
                let mut muls = vec![];
                for arg in m.iter() {
                    let a = arg.linearize_impl(
                        fn_map,
                        params,
                        constants,
                        constant_map,
                        instr,
                        subexpressions,
                        args,
                        arg_start,
                    )?;
                    muls.push(a);
                }

                muls.sort();

                let temp = Slot::Temp(instr.len());
                instr.push(Instruction::Mul(temp, muls, 0));
                temp
            }
            AtomView::Add(a) => {
                let mut adds = vec![];
                for arg in a.iter() {
                    adds.push(arg.linearize_impl(
                        fn_map,
                        params,
                        constants,
                        constant_map,
                        instr,
                        subexpressions,
                        args,
                        arg_start,
                    )?);
                }

                adds.sort();

                let temp = Slot::Temp(instr.len());
                instr.push(Instruction::Add(temp, adds, 0));
                temp
            }
        };

        subexpressions.insert(*self, res);
        Ok(res)
    }
}

/// A hash of an expression, used for common subexpression elimination.
pub type ExpressionHash = u64;

/// A tree representation of an expression.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expression<T> {
    Const(ExpressionHash, Box<T>),
    Parameter(ExpressionHash, usize),
    Eval(ExpressionHash, u32, Vec<Expression<T>>),
    Add(ExpressionHash, Vec<Expression<T>>),
    Mul(ExpressionHash, Vec<Expression<T>>),
    Pow(ExpressionHash, Box<(Expression<T>, i64)>),
    Powf(ExpressionHash, Box<(Expression<T>, Expression<T>)>),
    ReadArg(ExpressionHash, usize), // read nth function argument
    BuiltinFun(ExpressionHash, BuiltinSymbol, Box<Expression<T>>),
    ExternalFun(ExpressionHash, u32, Vec<Expression<T>>),
    IfElse(
        ExpressionHash,
        Box<(Expression<T>, Expression<T>, Expression<T>)>,
    ),
    SubExpression(ExpressionHash, usize),
}

impl<T> Expression<T> {
    fn get_hash(&self) -> ExpressionHash {
        match self {
            Expression::Const(h, _) => *h,
            Expression::Parameter(h, _) => *h,
            Expression::Eval(h, _, _) => *h,
            Expression::Add(h, _) => *h,
            Expression::Mul(h, _) => *h,
            Expression::Pow(h, _) => *h,
            Expression::Powf(h, _) => *h,
            Expression::ReadArg(h, _) => *h,
            Expression::BuiltinFun(h, _, _) => *h,
            Expression::SubExpression(h, _) => *h,
            Expression::ExternalFun(h, _, _) => *h,
            Expression::IfElse(h, _) => *h,
        }
    }
}

impl<T: Eq + InternalOrdering> PartialOrd for Expression<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq + InternalOrdering> Ord for Expression<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Expression::Const(_, a), Expression::Const(_, b)) => a.internal_cmp(b),
            (Expression::Parameter(_, a), Expression::Parameter(_, b)) => a.cmp(b),
            (Expression::Eval(_, a, b), Expression::Eval(_, c, d)) => {
                a.cmp(c).then_with(|| b.cmp(d))
            }
            (Expression::Add(_, a), Expression::Add(_, b)) => a.cmp(b),
            (Expression::Mul(_, a), Expression::Mul(_, b)) => a.cmp(b),
            (Expression::Pow(_, p1), Expression::Pow(_, p2)) => p1.cmp(p2),
            (Expression::Powf(_, p1), Expression::Powf(_, p2)) => p1.cmp(p2),
            (Expression::ReadArg(_, r1), Expression::ReadArg(_, r2)) => r1.cmp(r2),
            (Expression::BuiltinFun(_, a, b), Expression::BuiltinFun(_, c, d)) => {
                a.cmp(c).then_with(|| b.cmp(d))
            }
            (Expression::SubExpression(_, s1), Expression::SubExpression(_, s2)) => s1.cmp(s2),
            (Expression::ExternalFun(_, a, b), Expression::ExternalFun(_, c, d)) => {
                a.cmp(c).then_with(|| b.cmp(d))
            }
            (Expression::IfElse(_, a), Expression::IfElse(_, b)) => a.cmp(b),
            (Expression::Const(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Const(_, _)) => std::cmp::Ordering::Greater,
            (Expression::Parameter(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Parameter(_, _)) => std::cmp::Ordering::Greater,
            (Expression::Eval(_, _, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Eval(_, _, _)) => std::cmp::Ordering::Greater,
            (Expression::Add(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Add(_, _)) => std::cmp::Ordering::Greater,
            (Expression::Mul(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Mul(_, _)) => std::cmp::Ordering::Greater,
            (Expression::Pow(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Pow(_, _)) => std::cmp::Ordering::Greater,
            (Expression::Powf(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Powf(_, _)) => std::cmp::Ordering::Greater,
            (Expression::ReadArg(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::ReadArg(_, _)) => std::cmp::Ordering::Greater,
            (Expression::BuiltinFun(_, _, _), _) => std::cmp::Ordering::Less,
            (_, Expression::BuiltinFun(_, _, _)) => std::cmp::Ordering::Greater,
            (Expression::ExternalFun(_, _, _), _) => std::cmp::Ordering::Less,
            (_, Expression::ExternalFun(_, _, _)) => std::cmp::Ordering::Greater,
            (Expression::IfElse(_, _), _) => std::cmp::Ordering::Greater,
            (_, Expression::IfElse(_, _)) => std::cmp::Ordering::Less, // sort last so that parent common subexpressions can be used inside both branches
        }
    }
}

impl<T: Eq + Hash> Hash for Expression<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.get_hash())
    }
}

impl<T: Eq + Hash + Clone + InternalOrdering> Expression<T> {
    fn find_subexpression<'a>(&'a self, subexp: &mut HashMap<&'a Expression<T>, usize>) -> bool {
        if matches!(
            self,
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _)
        ) {
            return true;
        }

        if let Some(i) = subexp.get_mut(self) {
            *i += 1;
            return true;
        }

        subexp.insert(self, 1);

        match self {
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _) => {}
            Expression::Eval(_, _, ae) => {
                for arg in ae {
                    arg.find_subexpression(subexp);
                }
            }
            Expression::Add(_, a) | Expression::Mul(_, a) | Expression::ExternalFun(_, _, a) => {
                for arg in a {
                    arg.find_subexpression(subexp);
                }
            }
            Expression::Pow(_, p) => {
                p.0.find_subexpression(subexp);
            }
            Expression::Powf(_, p) => {
                p.0.find_subexpression(subexp);
                p.1.find_subexpression(subexp);
            }
            Expression::BuiltinFun(_, _, a) => {
                a.find_subexpression(subexp);
            }
            Expression::SubExpression(_, _) => {}
            Expression::IfElse(_, b) => {
                b.0.find_subexpression(subexp);
                b.1.find_subexpression(subexp);
                b.2.find_subexpression(subexp);
            }
        }

        false
    }

    fn replace_subexpression(&mut self, subexp: &HashMap<&Expression<T>, usize>, skip_root: bool) {
        if !skip_root && let Some(i) = subexp.get(self) {
            *self = Expression::SubExpression(self.get_hash(), *i); // TODO: do not recyle hash?
            return;
        }

        match self {
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _) => {}
            Expression::Eval(_, _, ae) => {
                for arg in &mut *ae {
                    arg.replace_subexpression(subexp, false);
                }
            }
            Expression::Add(_, a) | Expression::Mul(_, a) | Expression::ExternalFun(_, _, a) => {
                for arg in a {
                    arg.replace_subexpression(subexp, false);
                }
            }
            Expression::Pow(_, p) => {
                p.0.replace_subexpression(subexp, false);
            }
            Expression::Powf(_, p) => {
                p.0.replace_subexpression(subexp, false);
                p.1.replace_subexpression(subexp, false);
            }
            Expression::BuiltinFun(_, _, a) => {
                a.replace_subexpression(subexp, false);
            }
            Expression::SubExpression(_, _) => {}
            Expression::IfElse(_, b) => {
                b.0.replace_subexpression(subexp, false);
                b.1.replace_subexpression(subexp, false);
                b.2.replace_subexpression(subexp, false);
            }
        }
    }

    // Count the number of additions and multiplications in the expression, counting
    // subexpressions only once.
    pub fn count_operations_with_subexpression<'a>(
        &'a self,
        sub_expr: &mut HashMap<&'a Self, usize>,
    ) -> (usize, usize) {
        if matches!(
            self,
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _)
        ) {
            return (0, 0);
        }

        if sub_expr.contains_key(self) {
            return (0, 0);
        }

        sub_expr.insert(self, 1);

        match self {
            Expression::Const(_, _) => (0, 0),
            Expression::Parameter(_, _) => (0, 0),
            Expression::Eval(_, _, args) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in args {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add, mul)
            }
            Expression::Add(_, a) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in a {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add + a.len() - 1, mul)
            }
            Expression::Mul(_, m) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in m {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add, mul + m.len() - 1)
            }
            Expression::Pow(_, p) => {
                let (a, m) = p.0.count_operations_with_subexpression(sub_expr);
                (a, m + p.1.unsigned_abs() as usize - 1)
            }
            Expression::Powf(_, p) => {
                let (a, m) = p.0.count_operations_with_subexpression(sub_expr);
                let (a2, m2) = p.1.count_operations_with_subexpression(sub_expr);
                (a + a2, m + m2 + 1) // not clear how to count this
            }
            Expression::ReadArg(_, _) => (0, 0),
            Expression::BuiltinFun(_, _, b) => b.count_operations_with_subexpression(sub_expr), // not clear how to count this, third arg?
            Expression::SubExpression(_, _) => (0, 0),
            Expression::ExternalFun(_, _, a) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in a {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add + a.len() - 1, mul)
            }
            Expression::IfElse(_, b) => {
                let (a1, m1) = b.0.count_operations_with_subexpression(sub_expr);
                let (a2, m2) = b.1.count_operations_with_subexpression(sub_expr);
                let (a3, m3) = b.2.count_operations_with_subexpression(sub_expr);
                (a1 + a2 + a3, m1 + m2 + m3)
            }
        }
    }
}

impl<T: std::hash::Hash + Clone> Expression<T> {
    fn rehashed(mut self, partial: bool) -> Self {
        self.rehash(partial);
        self
    }

    fn rehash(&mut self, partial: bool) -> ExpressionHash {
        match self {
            Expression::Const(h, c) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(0);
                c.hash(&mut hasher);
                *h = hasher.finish();
                *h
            }
            Expression::Parameter(h, p) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(1);
                hasher.write_usize(*p);
                *h = hasher.finish();
                *h
            }
            Expression::Eval(h, i, v) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(2);
                hasher.write_u32(*i);
                for x in v {
                    hasher.write_u64(x.rehash(partial));
                }
                *h = hasher.finish();
                *h
            }
            Expression::Add(h, v) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(3);

                // do an additive hash
                let mut arg_sum = 0u64;
                for x in v {
                    arg_sum = arg_sum.wrapping_add(x.rehash(partial));
                }
                hasher.write_u64(arg_sum);
                *h = hasher.finish();
                *h
            }
            Expression::Mul(h, v) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(4);

                // do an additive hash
                let mut arg_sum = 0u64;
                for x in v {
                    arg_sum = arg_sum.wrapping_add(x.rehash(partial));
                }
                hasher.write_u64(arg_sum);
                *h = hasher.finish();
                *h
            }
            Expression::Pow(h, p) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(5);
                let hb = p.0.rehash(partial);
                hasher.write_u64(hb);
                hasher.write_i64(p.1);
                *h = hasher.finish();
                *h
            }
            Expression::Powf(h, p) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(6);
                let hb = p.0.rehash(partial);
                let he = p.1.rehash(partial);
                hasher.write_u64(hb);
                hasher.write_u64(he);
                *h = hasher.finish();
                *h
            }
            Expression::ReadArg(h, i) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(7);
                hasher.write_usize(*i);
                *h = hasher.finish();
                *h
            }
            Expression::BuiltinFun(h, s, a) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(8);
                s.hash(&mut hasher);
                let ha = a.rehash(partial);
                hasher.write_u64(ha);
                *h = hasher.finish();
                *h
            }
            Expression::SubExpression(h, i) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(9);
                hasher.write_usize(*i);
                *h = hasher.finish();
                *h
            }
            Expression::ExternalFun(h, s, a) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(10);
                s.hash(&mut hasher);
                for x in a {
                    hasher.write_u64(x.rehash(partial));
                }
                *h = hasher.finish();
                *h
            }
            Expression::IfElse(h, b) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(11);
                hasher.write_u64(b.0.rehash(partial));
                hasher.write_u64(b.1.rehash(partial));
                hasher.write_u64(b.2.rehash(partial));
                *h = hasher.finish();
                *h
            }
        }
    }
}

/// An optimized evaluator for expressions that can evaluate expressions with parameters.
/// The evaluator can be called directly using [Self::evaluate] or it can be exported
/// to high-performance C++ code using [Self::export_cpp].
///
/// To call the evaluator with external functions, use [Self::with_external_functions] to
/// register implementation for them.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Clone, Debug)]

pub struct ExpressionEvaluator<T> {
    stack: Vec<T>,
    param_count: usize,
    reserved_indices: usize,
    instructions: Vec<(Instr, ComplexPhase)>,
    result_indices: Vec<usize>,
    external_fns: Vec<String>,
    settings: OptimizationSettings,
}

impl<T: Clone> ExpressionEvaluator<T> {
    /// Register external functions for the evaluator.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// use ahash::HashMap;
    /// use symbolica::{atom::AtomCore, evaluate::{FunctionMap, OptimizationSettings, ExternalFunction}, parse, symbol};
    ///
    /// let mut ext: HashMap<String, Box<dyn ExternalFunction<f64>>> = HashMap::default();
    /// ext.insert("f".to_string(), Box::new(|a| a[0] * a[0] + a[1]));
    ///
    ///
    /// let mut f = FunctionMap::new();
    /// f.add_external_function(symbol!("f"), "f".to_string())
    ///     .unwrap();
    ///
    /// let params = vec![parse!("x"), parse!("y")];
    /// let optimization_settings = OptimizationSettings::default();
    /// let evaluator = parse!("f(x,y)").evaluator(&f, &params, optimization_settings).unwrap().map_coeff(&|x| x.re.to_f64());
    ///
    /// let mut ev = evaluator.with_external_functions(ext).unwrap();
    /// assert_eq!(ev.evaluate_single(&[2.0, 3.0]), 7.0);
    /// ```
    pub fn with_external_functions(
        &self,
        mut external_fns: HashMap<String, Box<dyn ExternalFunction<T>>>,
    ) -> Result<ExpressionEvaluatorWithExternalFunctions<T>, String> {
        let mut external = vec![];
        for e in &self.external_fns {
            if let Some(f) = external_fns.remove(e) {
                external.push((vec![], e.clone(), f));
            } else {
                return Err(format!("External function '{e}' not found"));
            }
        }

        Ok(ExpressionEvaluatorWithExternalFunctions {
            eval: self.clone(),
            external_fns: external,
        })
    }
}

impl<T: SingleFloat> ExpressionEvaluator<Complex<T>> {
    /// Check if the expression evaluator is real, i.e., all coefficients are real.
    pub fn is_real(&self) -> bool {
        self.stack.iter().all(|x| x.is_real())
    }
}

impl<T: Real> ExpressionEvaluator<T> {
    /// Evaluate the expression evaluator which yields a single result.
    pub fn evaluate_single(&mut self, params: &[T]) -> T {
        if self.result_indices.len() != 1 {
            panic!(
                "Evaluator does not return a single result but {} results",
                self.result_indices.len()
            );
        }

        let mut res = T::new_zero();
        self.evaluate(params, std::slice::from_mut(&mut res));
        res
    }

    /// Evaluate the expression evaluator and write the results in `out`.
    #[inline]
    pub fn evaluate(&mut self, params: &[T], out: &mut [T]) {
        self.evaluate_impl(params, &mut [], out);
    }

    #[cold]
    #[inline(never)]
    fn evaluate_impl_no_ops(
        stack: &mut [T],
        instr: &Instr,
        external_fns: &mut [(Vec<T>, String, Box<dyn ExternalFunction<T>>)],
    ) -> Option<usize> {
        match instr {
            Instr::Powf(r, b, e) => {
                stack[*r] = stack[*b].powf(&stack[*e]);
            }
            Instr::BuiltinFun(r, s, arg) => match s.0.get_id() {
                Symbol::EXP_ID => stack[*r] = stack[*arg].exp(),
                Symbol::LOG_ID => stack[*r] = stack[*arg].log(),
                Symbol::SIN_ID => stack[*r] = stack[*arg].sin(),
                Symbol::COS_ID => stack[*r] = stack[*arg].cos(),
                Symbol::SQRT_ID => stack[*r] = stack[*arg].sqrt(),
                Symbol::ABS_ID => stack[*r] = stack[*arg].norm(),
                Symbol::CONJ_ID => stack[*r] = stack[*arg].conj(),
                _ => unreachable!(),
            },
            Instr::ExternalFun(r, s, args) => {
                let (cache, _, f) = &mut external_fns[*s];

                if cache.len() < args.len() {
                    cache.resize(args.len(), T::new_zero());
                }

                for (i, v) in cache.iter_mut().zip(args) {
                    i.set_from(&stack[*v]);
                }

                stack[*r] = (f)(&cache[..args.len()]);
            }
            Instr::IfElse(n, label) => {
                // jump to else block
                if stack[*n].is_fully_zero() {
                    return Some(label.0);
                }
            }
            Instr::Goto(label) => {
                return Some(label.0);
            }
            Instr::Label(_) => {}
            Instr::Join(r, c, a, b) => {
                if !stack[*c].is_fully_zero() {
                    stack[*r] = stack[*a].clone();
                } else {
                    stack[*r] = stack[*b].clone();
                }
            }
            Instr::Add(..) | Instr::Mul(..) | Instr::Pow(..) => {
                unreachable!()
            }
        }

        None
    }

    /// Evaluate the expression evaluator and write the results in `out`.
    fn evaluate_impl(
        &mut self,
        params: &[T],
        external_fns: &mut [(Vec<T>, String, Box<dyn ExternalFunction<T>>)],
        out: &mut [T],
    ) {
        if self.param_count != params.len() {
            panic!(
                "Parameter count mismatch: expected {}, got {}",
                self.param_count,
                params.len()
            );
        }

        for (t, p) in self.stack.iter_mut().zip(params) {
            t.set_from(p);
        }

        let mut tmp = T::new_zero();
        let mut i = 0;
        while i < self.instructions.len() {
            let (instr, _) = unsafe { &self.instructions.get_unchecked(i) };
            match instr {
                Instr::Add(r, v) => unsafe {
                    match v.len() {
                        2 => {
                            tmp.set_from(self.stack.get_unchecked(*v.get_unchecked(0)));
                            tmp += self.stack.get_unchecked(*v.get_unchecked(1));
                        }
                        3 => {
                            tmp.set_from(self.stack.get_unchecked(*v.get_unchecked(0)));
                            tmp += self.stack.get_unchecked(*v.get_unchecked(1));
                            tmp += self.stack.get_unchecked(*v.get_unchecked(2));
                        }
                        _ => {
                            tmp.set_from(self.stack.get_unchecked(*v.get_unchecked(0)));
                            for x in v.get_unchecked(1..) {
                                tmp += self.stack.get_unchecked(*x);
                            }
                        }
                    }

                    std::mem::swap(self.stack.get_unchecked_mut(*r), &mut tmp);
                },
                Instr::Mul(r, v) => unsafe {
                    match v.len() {
                        2 => {
                            tmp.set_from(self.stack.get_unchecked(*v.get_unchecked(0)));
                            tmp *= self.stack.get_unchecked(*v.get_unchecked(1));
                        }
                        3 => {
                            tmp.set_from(self.stack.get_unchecked(*v.get_unchecked(0)));
                            tmp *= self.stack.get_unchecked(*v.get_unchecked(1));
                            tmp *= self.stack.get_unchecked(*v.get_unchecked(2));
                        }
                        _ => {
                            tmp.set_from(self.stack.get_unchecked(*v.get_unchecked(0)));
                            for x in v.get_unchecked(1..) {
                                tmp *= self.stack.get_unchecked(*x);
                            }
                        }
                    }

                    std::mem::swap(self.stack.get_unchecked_mut(*r), &mut tmp);
                },
                Instr::Pow(r, b, e) => {
                    if *e >= 0 {
                        self.stack[*r] = self.stack[*b].pow(*e as u64);
                    } else {
                        self.stack[*r] = self.stack[*b].pow(e.unsigned_abs()).inv();
                    }
                }
                _ => {
                    if let Some(idx) =
                        Self::evaluate_impl_no_ops(&mut self.stack, instr, external_fns)
                    {
                        i = idx;
                        continue;
                    }
                }
            }

            i += 1;
        }

        for (o, i) in out.iter_mut().zip(&self.result_indices) {
            o.set_from(&self.stack[*i]);
        }
    }
}

impl<T: Default> ExpressionEvaluator<T> {
    /// Map the coefficients to a different type.
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(self, f: &F) -> ExpressionEvaluator<T2> {
        ExpressionEvaluator {
            stack: self.stack.iter().map(f).collect(),
            param_count: self.param_count,
            reserved_indices: self.reserved_indices,
            instructions: self.instructions,
            result_indices: self.result_indices,
            external_fns: self.external_fns.clone(),
            settings: self.settings.clone(),
        }
    }

    #[allow(dead_code)]
    pub(crate) fn set_coeff<T2: Default + Clone>(self, coeffs: &[T2]) -> ExpressionEvaluator<T2> {
        if coeffs.len() != self.reserved_indices - self.param_count {
            panic!(
                "Wrong number of coefficients: {} vs {}",
                coeffs.len(),
                self.reserved_indices - self.param_count
            )
        }

        let mut stack = vec![T2::default(); self.stack.len()];
        for (s, coeff) in stack.iter_mut().skip(self.param_count).zip(coeffs) {
            *s = coeff.clone();
        }

        ExpressionEvaluator {
            stack,
            param_count: self.param_count,
            reserved_indices: self.reserved_indices,
            instructions: self.instructions,
            result_indices: self.result_indices,
            external_fns: self.external_fns,
            settings: self.settings,
        }
    }

    pub fn get_input_len(&self) -> usize {
        self.param_count
    }

    pub fn get_output_len(&self) -> usize {
        self.result_indices.len()
    }

    pub fn get_constants(&self) -> &[T] {
        &self.stack[self.param_count..self.reserved_indices]
    }

    /// Return the total number of additions and multiplications.
    pub fn count_operations(&self) -> (usize, usize) {
        let mut add_count = 0;
        let mut mul_count = 0;

        for (instr, _) in &self.instructions {
            match instr {
                Instr::Add(_, s) => add_count += s.len() - 1,
                Instr::Mul(_, s) => mul_count += s.len() - 1,
                _ => {}
            }
        }

        (add_count, mul_count)
    }

    /// Remove common instructions and return the number of removed instructions.
    fn remove_common_instructions(&mut self) -> usize {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
        enum CSE<'a> {
            Add(&'a [usize]),
            Mul(&'a [usize]),
            Pow(usize, i64),
            Powf(usize, usize),
            BuiltinFun(u32, usize),
            ExternalFun(u32, &'a [usize]),
        }

        let mut common_instr = HashMap::with_capacity(self.instructions.len());
        let mut new_instr = Vec::with_capacity(self.instructions.len());
        let mut i = 0;

        let mut rename_map: Vec<_> = (0..self.reserved_indices).collect();
        let mut removed = 0;

        let mut dag_nodes = vec![0]; // store index to parent node
        let mut current_node = 0;

        for (instr, phase) in &self.instructions {
            let new_pos = new_instr.len() + self.reserved_indices;

            let key = match &self.instructions[i].0 {
                Instr::Add(_, a) => Some(CSE::Add(a)),
                Instr::Mul(_, a) => Some(CSE::Mul(a)),
                Instr::Pow(_, b, e) => Some(CSE::Pow(*b, *e)),
                Instr::Powf(_, b, e) => Some(CSE::Powf(*b, *e)),
                Instr::BuiltinFun(_, s, a) => Some(CSE::BuiltinFun(s.0.get_id(), *a)),
                Instr::ExternalFun(_, s, a) => Some(CSE::ExternalFun(*s as u32, a)),
                _ => None,
            };

            if let Some(key) = key {
                match common_instr.entry(key) {
                    Entry::Occupied(mut o) => {
                        let (old_pos, branch) = o.get_mut();

                        let mut cur = current_node;
                        while cur > *branch {
                            cur = dag_nodes[cur];
                        }

                        if cur == *branch {
                            removed += 1;
                            rename_map.push(*old_pos);
                            i += 1;
                            continue;
                        } else {
                            // the previous occurrence was in a non-parent branch
                            // that cannot be reused, so treat this occurrence
                            // as the first
                            *old_pos = new_pos;
                            *branch = current_node;
                        }
                    }
                    Entry::Vacant(v) => {
                        v.insert((new_pos, current_node));
                    }
                }
            }

            let mut s = instr.clone();

            match &mut s {
                Instr::Add(p, a) | Instr::Mul(p, a) => {
                    let mut last = 0;
                    let mut sort = false;
                    for x in &mut *a {
                        *x = rename_map[*x];
                        if *x < last {
                            sort = true;
                        } else {
                            last = *x;
                        }
                    }

                    if sort {
                        a.sort_unstable();
                    }
                    *p = new_pos;
                }
                Instr::Pow(p, b, _) | Instr::BuiltinFun(p, _, b) => {
                    *b = rename_map[*b];
                    *p = new_pos;
                }
                Instr::Powf(p, a, b) => {
                    *a = rename_map[*a];
                    *b = rename_map[*b];
                    *p = new_pos;
                }
                Instr::ExternalFun(p, _, a) => {
                    *p = new_pos;
                    for x in a {
                        *x = rename_map[*x];
                    }
                }
                Instr::Join(p, a, b, c) => {
                    current_node = dag_nodes[current_node];
                    *a = rename_map[*a];
                    *b = rename_map[*b];
                    *c = rename_map[*c];
                    *p = new_pos;
                }
                Instr::IfElse(c, _) => {
                    dag_nodes.push(current_node); // enter if block
                    current_node = dag_nodes.len() - 1;
                    *c = rename_map[*c];
                }
                Instr::Goto(_) => {
                    let parent = dag_nodes[current_node];
                    dag_nodes.push(parent); // enter else block (included goto and labels)
                    current_node = dag_nodes.len() - 1;
                }
                _ => {}
            }

            new_instr.push((s, *phase));
            rename_map.push(new_pos);
            i += 1;
        }

        for x in &mut self.result_indices {
            *x = rename_map[*x];
        }

        self.instructions = new_instr;

        self.fix_labels();

        removed
    }

    /// Set the labels to the their instruction position.
    fn fix_labels(&mut self) {
        let mut label_map: HashMap<usize, usize> = HashMap::default();
        for (i, (x, _)) in self.instructions.iter_mut().enumerate().rev() {
            match x {
                Instr::Label(l) => {
                    label_map.insert(l.0, i);
                    l.0 = i;
                }
                Instr::Goto(l) => {
                    l.0 = label_map[&l.0];
                }
                Instr::IfElse(_, l) => {
                    l.0 = label_map[&l.0];
                }
                _ => {}
            }
        }
    }

    /// Remove common pairs of instructions. Assumes that the arguments
    /// of the instructions are sorted.
    fn remove_common_pairs(&mut self) -> usize {
        let mut affected_lines = vec![false; self.instructions.len()];

        // store the global branch a line belongs to
        let mut branch_id = vec![0; self.instructions.len()];
        let mut dag_nodes = vec![0]; // store index to parent node
        let mut current_node = 0;

        let mut common_ops_simple: HashMap<_, u32> = HashMap::default();

        if self.instructions.len() > u32::MAX as usize / 2 {
            // the extension is easy, but it will cost more memory.
            // will only be added when a user runs into it.
            error!(
                "Too many instructions to find common pairs. Reach out to Symbolica devs to extend the limit."
            );
            return 0;
        }

        for (p, (i, _)) in self.instructions.iter().enumerate() {
            if common_ops_simple.len() > self.settings.max_common_pair_cache_entries {
                break;
            }

            if p % 10000 == 0 {
                if let Some(abort_check) = &self.settings.abort_check {
                    if abort_check() {
                        self.settings.abort_level = 1;
                        break;
                    }
                }
            }

            match i {
                Instr::Add(_, a) | Instr::Mul(_, a) => {
                    let is_add = matches!(i, Instr::Add(_, _));
                    'add_loop: for (li, l) in a.iter().enumerate() {
                        for r in &a[li + 1..] {
                            let mut key = (*l as u64) << 32 | (*r as u64) << 1;
                            if !is_add {
                                key |= 1;
                            }

                            if common_ops_simple.len() > self.settings.max_common_pair_cache_entries
                            {
                                break 'add_loop;
                            }

                            common_ops_simple
                                .entry(key)
                                .and_modify(|x| *x += 1)
                                .or_insert(1);
                        }
                    }
                }
                Instr::IfElse(_, _) => {
                    branch_id[p] = current_node;
                    dag_nodes.push(current_node); // enter if block
                    current_node = dag_nodes.len() - 1;
                    continue;
                }
                Instr::Goto(_) => {
                    let parent = dag_nodes[current_node];
                    dag_nodes.push(parent); // enter else block (included goto and labels)
                    current_node = dag_nodes.len() - 1;
                }
                Instr::Join(..) => {
                    current_node = dag_nodes[current_node];
                }
                _ => {}
            }
            branch_id[p] = current_node;
        }

        common_ops_simple.retain(|_, v| *v > 1);

        let mut common_ops_2: HashMap<_, Vec<usize>> = HashMap::default();

        for (p, (i, _)) in self.instructions.iter().enumerate() {
            match i {
                Instr::Add(_, a) | Instr::Mul(_, a) => {
                    let is_add = matches!(i, Instr::Add(_, _));
                    for (li, l) in a.iter().enumerate() {
                        for r in &a[li + 1..] {
                            let mut key = (*l as u64) << 32 | (*r as u64) << 1;
                            if !is_add {
                                key |= 1;
                            }

                            if *common_ops_simple.get(&key).unwrap_or(&0) > 1 {
                                common_ops_2.entry(key).or_default().push(p);
                            }
                        }
                    }
                }
                _ => {}
            }
        }

        drop(common_ops_simple); // clear the memory

        if common_ops_2.is_empty() {
            return 0;
        }

        let mut to_remove: Vec<_> = common_ops_2.into_iter().collect();
        to_remove.retain_mut(|(_, v)| {
            let keep = v.len() > 1;
            v.dedup();
            keep
        });

        // sort in other direction since we pop
        to_remove.sort_by(|a, b| a.1.len().cmp(&b.1.len()).then_with(|| a.cmp(b)));

        let total_remove = to_remove.len();

        let old_len = self.instructions.len();

        let mut new_symb_branch = vec![];

        while let Some((key, lines)) = to_remove.pop() {
            let l = (key >> 32) as usize;
            let r = ((key >> 1) & 0x7FFFFFFF) as usize;
            let is_add = key & 1 == 0;

            if lines.iter().any(|x| affected_lines[*x]) {
                continue;
            }

            let new_idx = self.stack.len();
            let new_op = if is_add {
                Instr::Add(new_idx, vec![l, r])
            } else {
                Instr::Mul(new_idx, vec![l, r])
            };

            self.stack.push(T::default());
            self.instructions.push((new_op, ComplexPhase::Any));

            let mut branch = branch_id[lines[0]];
            for &line in &lines {
                affected_lines[line] = true;

                let mut new_branch = branch_id[line];
                // find common root
                while branch != new_branch {
                    if branch > new_branch {
                        branch = dag_nodes[branch];
                    } else {
                        new_branch = dag_nodes[new_branch];
                    }
                }

                if let Instr::Add(_, a) | Instr::Mul(_, a) = &mut self.instructions[line].0 {
                    if l == r {
                        let count = a.iter().filter(|x| **x == l).count();
                        let pairs = count / 2;
                        if pairs > 0 {
                            a.retain(|x| *x != l);

                            if count % 2 == 1 {
                                a.push(l);
                            }

                            a.extend(std::iter::repeat_n(new_idx, pairs));
                            a.sort_unstable();
                        }
                    } else {
                        let mut idx1_count = 0;
                        let mut idx2_count = 0;
                        for v in &*a {
                            if *v == l {
                                idx1_count += 1;
                            }
                            if *v == r {
                                idx2_count += 1;
                            }
                        }

                        let pair_count = idx1_count.min(idx2_count);

                        if pair_count > 0 {
                            a.retain(|x| *x != l && *x != r);

                            // add back removed indices in cases such as idx1*idx2*idx2
                            if idx1_count > pair_count {
                                a.extend(std::iter::repeat_n(l, idx1_count - pair_count));
                            }
                            if idx2_count > pair_count {
                                a.extend(std::iter::repeat_n(r, idx2_count - pair_count));
                            }

                            a.extend(std::iter::repeat_n(new_idx, pair_count));
                            a.sort_unstable();
                        }
                    }
                }
            }

            new_symb_branch.push((lines[0], branch));
        }

        // detect the earliest point and latest point for an instruction placement
        // earliest point: after last dependency
        // latest point: before first usage in the correct usage zone
        let mut placement_bounds = vec![];
        for ((i, _), (first_usage, branch)) in
            self.instructions.drain(old_len..).zip(new_symb_branch)
        {
            let deps = match &i {
                Instr::BuiltinFun(_, _, a) => std::slice::from_ref(a),
                Instr::Add(_, a) | Instr::Mul(_, a) | Instr::ExternalFun(_, _, a) => a.as_slice(),
                _ => unreachable!(),
            };

            let mut last_dep = deps[0];
            for v in deps {
                last_dep = last_dep.max(*v);
            }

            let ins = if last_dep < self.reserved_indices {
                0
            } else {
                last_dep + 1 - self.reserved_indices
            };

            let mut latest_pos = ins;
            for j in (ins..first_usage + 1).rev() {
                if branch_id[j] == branch {
                    latest_pos = j;
                    break;
                }
            }

            placement_bounds.push((ins, latest_pos, i));
        }

        placement_bounds.sort_by_key(|x| x.1);

        let mut new_instr = vec![];
        let mut i = 0;
        let mut j = 0;

        let mut sub_rename = HashMap::default();
        let mut rename_map: Vec<_> = (0..self.reserved_indices).collect();

        macro_rules! rename {
            ($i:expr) => {
                if $i >= self.reserved_indices + self.instructions.len() {
                    sub_rename[&$i]
                } else {
                    rename_map[$i]
                }
            };
        }

        while i < self.instructions.len() {
            let new_pos = new_instr.len() + self.reserved_indices;

            if j < placement_bounds.len() && i == placement_bounds[j].1 {
                let (o, a) = match &placement_bounds[j].2 {
                    Instr::Add(o, a) => (*o, a.as_slice()),
                    Instr::Mul(o, a) => (*o, a.as_slice()),
                    _ => unreachable!(),
                };

                let mut new_a = a.iter().map(|x| rename!(*x)).collect::<Vec<_>>();
                new_a.sort();

                match placement_bounds[j].2 {
                    Instr::Add(_, _) => {
                        new_instr.push((Instr::Add(new_pos, new_a), ComplexPhase::Any));
                    }
                    Instr::Mul(_, _) => {
                        new_instr.push((Instr::Mul(new_pos, new_a), ComplexPhase::Any));
                    }
                    _ => unreachable!(),
                }

                sub_rename.insert(o, new_pos);

                j += 1;
            } else {
                let (mut s, sc) = self.instructions[i].clone();

                match &mut s {
                    Instr::Add(p, a) | Instr::Mul(p, a) => {
                        for x in &mut *a {
                            *x = rename!(*x);
                        }
                        a.sort();

                        // remove assignments
                        if a.len() == 1 {
                            rename_map.push(a[0]);
                            i += 1;
                            continue;
                        }

                        *p = new_pos;
                    }
                    Instr::Pow(p, b, _) | Instr::BuiltinFun(p, _, b) => {
                        *b = rename!(*b);
                        *p = new_pos;
                    }
                    Instr::Powf(p, a, b) => {
                        *a = rename!(*a);
                        *b = rename!(*b);
                        *p = new_pos;
                    }
                    Instr::ExternalFun(p, _, a) => {
                        *p = new_pos;
                        for x in a {
                            *x = rename!(*x);
                        }
                    }
                    Instr::Join(p, a, b, c) => {
                        *a = rename!(*a);
                        *b = rename!(*b);
                        *c = rename!(*c);
                        *p = new_pos;
                    }
                    Instr::IfElse(c, _) => {
                        *c = rename!(*c);
                    }
                    Instr::Goto(_) | Instr::Label(_) => {}
                }

                new_instr.push((s, sc));
                rename_map.push(new_pos);
                i += 1;
            }
        }

        for x in &mut self.result_indices {
            *x = rename!(*x);
        }

        assert!(j == placement_bounds.len());

        self.instructions = new_instr;
        self.fix_labels();

        total_remove
    }
}

/// Settings for operation realness of complex evaluators, used in [ExpressionEvaluator::set_real_params].
#[derive(Clone, Debug)]
pub struct ComplexEvaluatorSettings {
    /// Whether sqrt with real arguments yields real results.
    pub sqrt_real: bool,
    /// Whether log with real arguments yields real results.
    pub log_real: bool,
    /// Whether powf with real arguments yields real results.
    pub powf_real: bool,
    /// Report on the number of converted operations.
    pub verbose: bool,
}

impl ComplexEvaluatorSettings {
    /// Create complex evaluator settings, used for [ExpressionEvaluator::set_real_params].
    pub fn new(sqrt_real: bool, log_real: bool, powf_real: bool, verbose: bool) -> Self {
        ComplexEvaluatorSettings {
            sqrt_real,
            log_real,
            powf_real,
            verbose,
        }
    }

    /// Set that all square roots with real arguments yield real results.
    pub fn sqrt_real(mut self) -> Self {
        self.sqrt_real = true;
        self
    }

    /// Set that all logarithms with real arguments yield real results.
    pub fn log_real(mut self) -> Self {
        self.log_real = true;
        self
    }

    /// Set that all powf with real arguments yield real results.
    pub fn powf_real(mut self) -> Self {
        self.powf_real = true;
        self
    }

    /// Set verbose reporting.
    pub fn verbose(mut self) -> Self {
        self.verbose = true;
        self
    }
}

impl Default for ComplexEvaluatorSettings {
    /// Create default complex evaluator settings.
    fn default() -> Self {
        ComplexEvaluatorSettings {
            sqrt_real: false,
            log_real: false,
            powf_real: false,
            verbose: false,
        }
    }
}

impl<T: Default + PartialEq> ExpressionEvaluator<Complex<T>> {
    /// Set which parameters are fully real. This allows for more optimal
    /// assembly output that uses real arithmetic instead of complex arithmetic
    /// where possible.
    ///
    /// You can also set if all encountered sqrt, log, and powf operations with real
    /// arguments are expected to yield real results.
    ///
    /// Must be called after all optimization functions and merging are performed
    /// on the evaluator, or the registration will be lost.
    pub fn set_real_params(
        &mut self,
        real_params: &[usize],
        settings: ComplexEvaluatorSettings,
    ) -> Result<(), String> {
        let mut subcomponents = vec![ComplexPhase::Any; self.stack.len()];

        for i in real_params {
            if *i >= self.param_count {
                return Err(format!(
                    "Real parameter index {} out of bounds (parameter count {})",
                    i, self.param_count
                ));
            }

            subcomponents[*i] = ComplexPhase::Real;
        }

        for (s, c) in subcomponents
            .iter_mut()
            .zip(self.stack.iter())
            .skip(self.param_count)
            .take(self.reserved_indices - self.param_count)
        {
            if c.im == T::default() {
                *s = ComplexPhase::Real;
            } else if c.re == T::default() {
                *s = ComplexPhase::Imag;
            }
        }

        let mut div_components = 0;
        let mut mul_components = 0;

        for (instr, sc) in &mut self.instructions {
            let is_add = matches!(instr, Instr::Add(_, _));
            match instr {
                Instr::Add(r, args) | Instr::Mul(r, args) => {
                    let real_parts = args
                        .iter()
                        .filter(|x| subcomponents[**x] == ComplexPhase::Real)
                        .count();

                    if real_parts > 0 && real_parts != args.len() {
                        args.sort_by_key(|x| !matches!(subcomponents[*x], ComplexPhase::Real)); // sort real components first
                    }

                    if !is_add && real_parts > 1 {
                        mul_components += real_parts - 1;
                    }

                    if real_parts == args.len() {
                        *sc = ComplexPhase::Real;
                    } else if args.iter().all(|x| subcomponents[*x] == ComplexPhase::Imag) {
                        *sc = ComplexPhase::Imag;
                    } else if real_parts > 0 {
                        *sc = ComplexPhase::PartialReal(real_parts);
                    } else {
                        *sc = ComplexPhase::Any;
                    }

                    subcomponents[*r] = *sc;
                }
                Instr::Pow(r, b, _) => {
                    if subcomponents[*b] == ComplexPhase::Real {
                        *sc = ComplexPhase::Real;
                        div_components += 1;
                    } else {
                        *sc = ComplexPhase::Any;
                    }
                    subcomponents[*r] = *sc;
                }
                Instr::BuiltinFun(r, s, a) => {
                    if s.0.is_real() {
                        *sc = ComplexPhase::Real;
                        subcomponents[*r] = *sc;
                        continue;
                    }

                    if subcomponents[*a] != ComplexPhase::Real {
                        subcomponents[*r] = ComplexPhase::Any;
                        *sc = ComplexPhase::Any;
                        continue;
                    }

                    match s.0.get_id() {
                        Symbol::EXP_ID | Symbol::CONJ_ID | Symbol::SIN_ID | Symbol::COS_ID => {
                            *sc = ComplexPhase::Real;
                        }
                        Symbol::SQRT_ID if settings.sqrt_real => {
                            *sc = ComplexPhase::Real;
                        }
                        Symbol::LOG_ID if settings.log_real => {
                            *sc = ComplexPhase::Real;
                        }
                        _ => {
                            *sc = ComplexPhase::Any;
                        }
                    }

                    subcomponents[*r] = *sc;
                }
                Instr::Join(r, _, t, f) => {
                    if subcomponents[*t] == subcomponents[*f] {
                        *sc = subcomponents[*t];
                    } else {
                        *sc = ComplexPhase::Any;
                    }
                    subcomponents[*r] = *sc;
                }
                Instr::Powf(r, b, e) => {
                    if settings.powf_real
                        && subcomponents[*b] == ComplexPhase::Real
                        && subcomponents[*e] == ComplexPhase::Real
                    {
                        *sc = ComplexPhase::Real;
                    } else {
                        *sc = ComplexPhase::Any;
                    }
                    subcomponents[*r] = *sc;
                }
                Instr::ExternalFun(r, ..) => {
                    *sc = ComplexPhase::Any;
                    subcomponents[*r] = *sc;
                }
                Instr::IfElse(..) | Instr::Goto(..) | Instr::Label(..) => {
                    *sc = ComplexPhase::Any;
                }
            }
        }

        if settings.verbose {
            info!(
                "Changed {} mul ops and {} div ops from complex to double",
                mul_components, div_components
            );
        }

        Ok(())
    }
}

impl<T: Default + Clone + Eq + Hash> ExpressionEvaluator<T> {
    /// Merge evaluator `other` into `self`. The parameters must be the same, and
    /// the outputs will be concatenated.
    ///
    /// The optional `cpe_rounds` parameter can be used to limit the number of common
    /// pair elimination rounds after the merge.
    pub fn merge(&mut self, mut other: Self, cpe_rounds: Option<usize>) -> Result<(), String> {
        if self.param_count != other.param_count {
            return Err(format!(
                "Parameter count is different: {} vs {}",
                self.param_count, other.param_count
            ));
        }
        if self.external_fns != other.external_fns {
            return Err(format!(
                "External functions do not match: {:?} vs {:?}",
                self.external_fns, other.external_fns
            ));
        }

        let mut constants = HashMap::default();

        for (i, c) in self.stack[self.param_count..self.reserved_indices]
            .iter()
            .enumerate()
        {
            constants.insert(c.clone(), i);
        }

        let old_len = self.stack.len() - self.reserved_indices;

        self.stack.truncate(self.reserved_indices);

        for c in &other.stack[self.param_count..other.reserved_indices] {
            if constants.get(c).is_none() {
                let i = constants.len();
                constants.insert(c.clone(), i);
                self.stack.push(c.clone());
            }
        }

        let new_reserved_indices = self.stack.len();
        let mut delta = new_reserved_indices - self.reserved_indices;

        // shift stack indices
        if delta > 0 {
            for (i, _) in &mut self.instructions {
                match i {
                    Instr::Add(r, a) | Instr::Mul(r, a) | Instr::ExternalFun(r, _, a) => {
                        *r += delta;
                        for aa in a {
                            if *aa >= self.reserved_indices {
                                *aa += delta;
                            }
                        }
                    }
                    Instr::Pow(r, b, _) | Instr::BuiltinFun(r, _, b) => {
                        *r += delta;
                        if *b >= self.reserved_indices {
                            *b += delta;
                        }
                    }
                    Instr::Powf(r, b, e) => {
                        *r += delta;
                        if *b >= self.reserved_indices {
                            *b += delta;
                        }
                        if *e >= self.reserved_indices {
                            *e += delta;
                        }
                    }
                    Instr::IfElse(c, _) => {
                        if *c >= self.reserved_indices {
                            *c += delta;
                        }
                    }
                    Instr::Join(r, c, t, f) => {
                        *r += delta;
                        if *c >= self.reserved_indices {
                            *c += delta;
                        }
                        if *t >= self.reserved_indices {
                            *t += delta;
                        }
                        if *f >= self.reserved_indices {
                            *f += delta;
                        }
                    }
                    Instr::Goto(..) | Instr::Label(..) => {}
                }
            }

            for x in &mut self.result_indices {
                if *x >= self.reserved_indices {
                    *x += delta;
                }
            }
        }

        delta = old_len + new_reserved_indices - other.reserved_indices;
        for (i, _) in &mut other.instructions {
            match i {
                Instr::Add(r, a) | Instr::Mul(r, a) | Instr::ExternalFun(r, _, a) => {
                    *r += delta;
                    for aa in a {
                        if *aa >= other.reserved_indices {
                            *aa += delta;
                        } else if *aa >= other.param_count {
                            *aa = self.param_count + constants[&other.stack[*aa]];
                        }
                    }
                }
                Instr::Pow(r, b, _) | Instr::BuiltinFun(r, _, b) => {
                    *r += delta;
                    if *b >= other.reserved_indices {
                        *b += delta;
                    } else if *b >= other.param_count {
                        *b = self.param_count + constants[&other.stack[*b]];
                    }
                }
                Instr::Powf(r, b, e) => {
                    *r += delta;
                    if *b >= other.reserved_indices {
                        *b += delta;
                    } else if *b >= other.param_count {
                        *b = self.param_count + constants[&other.stack[*b]];
                    }
                    if *e >= other.reserved_indices {
                        *e += delta;
                    } else if *e >= other.param_count {
                        *e = self.param_count + constants[&other.stack[*e]];
                    }
                }
                Instr::IfElse(c, l) => {
                    if *c >= other.reserved_indices {
                        *c += delta;
                    } else if *c >= other.param_count {
                        *c = self.param_count + constants[&other.stack[*c]];
                    }

                    l.0 += self.instructions.len();
                }
                Instr::Join(r, c, t, f) => {
                    *r += delta;
                    if *c >= other.reserved_indices {
                        *c += delta;
                    } else if *c >= other.param_count {
                        *c = self.param_count + constants[&other.stack[*c]];
                    }
                    if *t >= other.reserved_indices {
                        *t += delta;
                    } else if *t >= other.param_count {
                        *t = self.param_count + constants[&other.stack[*t]];
                    }
                    if *f >= other.reserved_indices {
                        *f += delta;
                    } else if *f >= other.param_count {
                        *f = self.param_count + constants[&other.stack[*f]];
                    }
                }
                Instr::Goto(l) | Instr::Label(l) => {
                    l.0 += self.instructions.len();
                }
            }
        }

        for x in &mut other.result_indices {
            if *x >= other.reserved_indices {
                *x += delta;
            } else if *x >= other.param_count {
                *x = self.param_count + constants[&other.stack[*x]];
            }
        }

        self.instructions.append(&mut other.instructions);
        self.result_indices.append(&mut other.result_indices);
        self.reserved_indices = new_reserved_indices;

        self.undo_stack_optimization();

        loop {
            if self.settings.abort_level > 0 || self.remove_common_instructions() == 0 {
                self.settings.abort_level = 0;
                break;
            }
        }

        for _ in 0..cpe_rounds.unwrap_or(usize::MAX) {
            if self.settings.abort_level > 0 || self.remove_common_pairs() == 0 {
                self.settings.abort_level = 0;
                break;
            }
        }

        self.optimize_stack();

        Ok(())
    }
}

impl<T> ExpressionEvaluator<T> {
    pub fn optimize_stack(&mut self) {
        let mut last_use: Vec<usize> = vec![0; self.stack.len()];

        for (i, (x, _)) in self.instructions.iter().enumerate() {
            match x {
                Instr::Add(_, a) | Instr::Mul(_, a) | Instr::ExternalFun(_, _, a) => {
                    for v in a {
                        last_use[*v] = i;
                    }
                }
                Instr::Pow(_, b, _) | Instr::BuiltinFun(_, _, b) => {
                    last_use[*b] = i;
                }
                Instr::Powf(_, a, b) => {
                    last_use[*a] = i;
                    last_use[*b] = i;
                }
                Instr::Join(_, c, a, b) => {
                    last_use[*c] = i;
                    last_use[*a] = i;
                    last_use[*b] = i;
                }
                Instr::IfElse(c, _) => {
                    last_use[*c] = i;
                }
                Instr::Goto(..) | Instr::Label(..) => {}
            };
        }

        // prevent init slots from being overwritten
        for i in 0..self.reserved_indices {
            last_use[i] = self.instructions.len();
        }

        // prevent the output slots from being overwritten
        for i in &self.result_indices {
            last_use[*i] = self.instructions.len();
        }

        let mut rename_map: Vec<_> = (0..self.stack.len()).collect(); // identity map

        let mut free_indices = BinaryHeap::<Reverse<(usize, usize)>>::new();

        let mut max_reg = self.reserved_indices;
        for (i, (x, _)) in self.instructions.iter_mut().enumerate() {
            let cur_reg = match x {
                Instr::Add(r, _)
                | Instr::Mul(r, _)
                | Instr::Pow(r, _, _)
                | Instr::Powf(r, _, _)
                | Instr::BuiltinFun(r, _, _)
                | Instr::ExternalFun(r, _, _)
                | Instr::Join(r, _, _, _) => *r,
                Instr::IfElse(c, _) => {
                    *c = rename_map[*c];
                    continue;
                }
                Instr::Goto(..) | Instr::Label(..) => continue,
            };

            let new_reg = if let Some(Reverse((last_pos, _))) = free_indices.peek()
                 // <= is ok because we store intermediate results in temp values
                && *last_pos <= i
            {
                free_indices.pop().unwrap().0.1
            } else {
                max_reg += 1;
                max_reg - 1
            };

            free_indices.push(Reverse((last_use[cur_reg], new_reg)));
            rename_map[cur_reg] = new_reg;

            match x {
                Instr::Add(r, a) | Instr::Mul(r, a) | Instr::ExternalFun(r, _, a) => {
                    *r = new_reg;
                    for v in a {
                        *v = rename_map[*v];
                    }
                }
                Instr::Pow(r, b, _) | Instr::BuiltinFun(r, _, b) => {
                    *r = new_reg;
                    *b = rename_map[*b];
                }
                Instr::Powf(r, a, b) => {
                    *r = new_reg;
                    *a = rename_map[*a];
                    *b = rename_map[*b];
                }
                Instr::Join(r, c, a, b) => {
                    *r = new_reg;
                    *c = rename_map[*c];
                    *a = rename_map[*a];
                    *b = rename_map[*b];
                }
                Instr::IfElse(_, _) | Instr::Goto(..) | Instr::Label(..) => {
                    unreachable!()
                }
            };
        }

        self.stack.truncate(max_reg + 1);

        for i in &mut self.result_indices {
            *i = rename_map[*i];
        }
    }
}

impl<T: Default> ExpressionEvaluator<T> {
    fn undo_stack_optimization(&mut self) {
        // undo the stack optimization
        let mut unfold = HashMap::default();
        for (index, (i, _c)) in &mut self.instructions.iter_mut().enumerate() {
            match i {
                Instr::Add(r, a) | Instr::Mul(r, a) | Instr::ExternalFun(r, _, a) => {
                    for aa in a {
                        if *aa >= self.reserved_indices {
                            *aa = unfold[aa];
                        }
                    }

                    unfold.insert(*r, index + self.reserved_indices);
                    *r = index + self.reserved_indices;
                }
                Instr::Pow(r, b, _) | Instr::BuiltinFun(r, _, b) => {
                    if *b >= self.reserved_indices {
                        *b = unfold[b];
                    }
                    unfold.insert(*r, index + self.reserved_indices);
                    *r = index + self.reserved_indices;
                }
                Instr::Powf(r, b, e) => {
                    if *b >= self.reserved_indices {
                        *b = unfold[b];
                    }
                    if *e >= self.reserved_indices {
                        *e = unfold[e];
                    }
                    unfold.insert(*r, index + self.reserved_indices);
                    *r = index + self.reserved_indices;
                }
                Instr::IfElse(r, _) => {
                    if *r >= self.reserved_indices {
                        *r = unfold[r];
                    }
                }
                Instr::Join(r, c, t, f) => {
                    if *c >= self.reserved_indices {
                        *c = unfold[c];
                    }
                    if *t >= self.reserved_indices {
                        *t = unfold[t];
                    }
                    if *f >= self.reserved_indices {
                        *f = unfold[f];
                    }
                    unfold.insert(*r, index + self.reserved_indices);
                    *r = index + self.reserved_indices;
                }
                Instr::Goto(..) | Instr::Label(..) => {}
            }
        }

        for i in &mut self.result_indices {
            if *i >= self.reserved_indices {
                *i = unfold[i];
            }
        }

        for _ in 0..self.instructions.len() {
            self.stack.push(T::default());
        }
    }
}

/// A number that can be exported to C++ code.
pub trait ExportNumber {
    /// Export the number as a string.
    fn export(&self) -> String;
    /// Export the number wrapped in a C++ type `T`.
    fn export_wrapped(&self) -> String {
        format!("T({})", self.export())
    }
    /// Export the number wrapped in a C++ type `wrapper`.
    fn export_wrapped_with(&self, wrapper: &str) -> String {
        format!("{wrapper}({})", self.export())
    }
    /// Check if the number is real.
    fn is_real(&self) -> bool;
}

impl ExportNumber for f64 {
    fn export(&self) -> String {
        format!("{:e}", self)
    }

    fn is_real(&self) -> bool {
        true
    }
}

impl ExportNumber for F64 {
    fn export(&self) -> String {
        format!("{:e}", self)
    }

    fn is_real(&self) -> bool {
        true
    }
}

impl ExportNumber for Float {
    fn export(&self) -> String {
        format!("{:e}", self)
    }

    fn is_real(&self) -> bool {
        true
    }
}

impl ExportNumber for Rational {
    fn export(&self) -> String {
        self.to_string()
    }

    fn is_real(&self) -> bool {
        true
    }
}

impl<T: ExportNumber + SingleFloat> ExportNumber for Complex<T> {
    fn export(&self) -> String {
        if self.im.is_zero() {
            self.re.export()
        } else {
            format!("{}, {}", self.re.export(), self.im.export())
        }
    }

    fn is_real(&self) -> bool {
        self.im.is_zero()
    }
}

/// The number class used for exporting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumberClass {
    RealF64,
    ComplexF64,
}

/// Settings for exporting the evaluation tree to C++ code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExportSettings {
    /// Include required `#include` statements in the generated code.
    pub include_header: bool,
    /// Set the inline assembly mode.
    /// With `inline_asm` set to any value other than `None`,
    /// high-performance inline ASM code will be generated for most
    /// evaluation instructions. This often gives better performance than
    /// the `O3` optimization level and results in very fast compilation.
    pub inline_asm: InlineASM,
    /// Custom header to include in the generated code.
    /// This can be used to include additional libraries or custom functions.
    pub custom_header: Option<String>,
}

impl Default for ExportSettings {
    fn default() -> Self {
        ExportSettings {
            include_header: true,
            inline_asm: InlineASM::default(),
            custom_header: None,
        }
    }
}

impl<T: ExportNumber + SingleFloat> ExpressionEvaluator<T> {
    /// Create a C++ code representation of the evaluation tree.
    /// The resulting source code can be compiled and loaded.
    ///
    /// You can also call `export_cpp` with types [f64], [wide::f64x4] for SIMD, [Complex] over [f64] and [wide::f64x4] for Complex SIMD, and [CudaRealf64] or
    /// [CudaComplexf64] for CUDA output.
    ///
    /// # Examples
    ///
    /// Create a C++ library that evaluates the function `x + y` for `f64` inputs:
    /// ```rust
    /// use symbolica::{atom::AtomCore, parse};
    /// use symbolica::evaluate::{CompiledNumber, FunctionMap, OptimizationSettings};
    /// let fn_map = FunctionMap::new();
    /// let params = vec![parse!("x"), parse!("y")];
    /// let optimization_settings = OptimizationSettings::default();
    /// let evaluator = parse!("x + y")
    ///     .evaluator(&fn_map, &params, optimization_settings)
    ///     .unwrap()
    ///     .map_coeff(&|x| x.to_real().unwrap().to_f64());
    ///
    /// let code = evaluator.export_cpp::<f64>("output.cpp", "my_function", Default::default()).unwrap();
    /// let lib = code.compile("out.so", f64::get_default_compile_options()).unwrap();
    /// let mut compiled_eval = lib.load().unwrap();
    ///
    /// let mut res = [0.];
    /// compiled_eval.evaluate(&[1., 2.], &mut res);
    /// assert_eq!(res, [3.]);
    /// ```
    pub fn export_cpp<F: CompiledNumber>(
        &self,
        path: impl AsRef<Path>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<ExportedCode<F>, std::io::Error> {
        let mut filename = path.as_ref().to_path_buf();
        if filename.extension().map(|x| x != ".cpp").unwrap_or(false) {
            filename.set_extension("cpp");
        }

        let mut source_code = format!(
            "// Auto-generated with Symbolica {}\n// Default build instructions: {} {}\n\n",
            env!("CARGO_PKG_VERSION"),
            F::get_default_compile_options().to_string(),
            filename.to_string_lossy(),
        );

        source_code += &self
            .export_cpp_str::<F>(function_name, settings)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidInput, e))?;

        std::fs::write(&filename, source_code)?;
        Ok(ExportedCode::<F> {
            path: filename,
            function_name: function_name.to_string(),
            _phantom: std::marker::PhantomData,
        })
    }

    /// Write the evaluation tree to a C++ source string.
    pub fn export_cpp_str<F: CompiledNumber>(
        &self,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String> {
        let function_name = F::construct_function_name(function_name);
        F::export_cpp(self, &function_name, settings)
    }

    pub fn export_simd_str(
        &self,
        function_name: &str,
        settings: ExportSettings,
        complex: bool,
        asm: InlineASM,
    ) -> String {
        let mut res = String::new();
        if settings.include_header {
            res += "#include \"xsimd/xsimd.hpp\"\n";
        }

        if complex {
            res += "#include <complex>\n";
            res += "using simd = xsimd::batch<std::complex<double>, xsimd::best_arch>;\n";
        } else {
            res += "using simd = xsimd::batch<double, xsimd::best_arch>;\n";
        }

        match asm {
            InlineASM::AVX2 => {
                res += &format!(
                    "extern \"C\" unsigned long {}_get_buffer_len()\n{{\n\treturn {};\n}}\n\n",
                    function_name,
                    self.stack.len()
                );

                if complex {
                    res += &format!(
                        "static const simd {}_CONSTANTS_complex[{}] = {{{}}};\n\n",
                        function_name,
                        self.reserved_indices - self.param_count + 2,
                        {
                            let mut nums = (self.param_count..self.reserved_indices)
                                .map(|i| format!("simd({})", self.stack[i].export()))
                                .collect::<Vec<_>>();
                            nums.push("-0.".to_string()); // used for inversion
                            nums.push("1".to_string()); // used for real inversion
                            nums.join(",")
                        }
                    );
                } else {
                    res += &format!(
                        "static const simd {}_CONSTANTS_double[{}] = {{{}}};\n\n",
                        function_name,
                        self.reserved_indices - self.param_count + 1,
                        {
                            let mut nums = (self.param_count..self.reserved_indices)
                                .map(|i| format!("simd({})", self.stack[i].export()))
                                .collect::<Vec<_>>();
                            nums.push("1".to_string()); // used for inversion
                            nums.join(",")
                        }
                    );
                }

                res += &format!(
                    "\nextern \"C\" void {function_name}(simd *params, simd *Z, simd *out) {{\n"
                );

                if complex {
                    self.export_asm_complex_impl(&self.instructions, function_name, asm, &mut res);
                } else {
                    self.export_asm_double_impl(&self.instructions, function_name, asm, &mut res);
                }

                res += "\treturn;\n}\n";
            }
            InlineASM::None => {
                res += &self.export_generic_cpp_str(function_name, &settings, NumberClass::RealF64);

                res += &format!(
                    "\nextern \"C\" {{\n\tvoid {function_name}(simd *params, simd *buffer, simd *out) {{\n\t\t{function_name}_gen(params, buffer, out);\n\t\treturn;\n\t}}\n}}\n"
                );
            }
            _ => panic!("Bad inline ASM option: {:?}", asm),
        }

        res
    }

    pub fn export_cuda_str(
        &self,
        function_name: &str,
        settings: ExportSettings,
        number_class: NumberClass,
    ) -> String {
        let mut res = String::new();
        if settings.include_header {
            res += "#include <cuda_runtime.h>\n";
            res += "#include <iostream>\n";
            res += "#include <stdio.h>\n";
            if number_class == NumberClass::ComplexF64 {
                res += "#include <cuda/std/complex>\n";
            } else {
                res += "template<typename T> T conj(T a) { return a; }\n";
            }
        };

        res += &format!("#define ERRMSG_LEN {}\n", CUDA_ERRMSG_LEN);

        if let Some(header) = &settings.custom_header {
            res += header;
            res += "\n\n";
        }
        if number_class == NumberClass::ComplexF64 {
            res += "typedef cuda::std::complex<double> CudaNumber;\n";
            res += "typedef std::complex<double> Number;\n";
        } else if number_class == NumberClass::RealF64 {
            res += "typedef double CudaNumber;\n";
            res += "typedef double Number;\n";
        }

        res += &format!(
            "\n__device__ void {}(CudaNumber* params, CudaNumber* out, size_t index) {{\n",
            function_name
        );

        res += &format!(
            "\tCudaNumber {};\n",
            (0..self.stack.len())
                .map(|x| format!("Z{}", x))
                .collect::<Vec<_>>()
                .join(", ")
        );

        res += &format!("\tint params_offset = index * {};\n", self.param_count);
        res += &format!(
            "\tint out_offset = index * {};\n",
            self.result_indices.len()
        );

        self.export_cpp_impl("params_offset + ", "CudaNumber", false, &mut res);

        for (i, r) in &mut self.result_indices.iter().enumerate() {
            res += &format!("\tout[out_offset + {i}] = ");
            res += &if *r < self.param_count {
                format!("params[params_offset + {r}]")
            } else if *r < self.reserved_indices {
                self.stack[*r].export_wrapped_with("CudaNumber")
            } else {
                format!("Z{r}")
            };

            res += ";\n";
        }

        res += "\treturn;\n}\n";

        res += &format!(
            r#"
struct {name}_EvaluationData {{
    CudaNumber *params;
    CudaNumber *out;
    size_t n; // Number of evaluations
    size_t block_size; // Number of threads per block
    size_t in_dimension = {in_dimension}; // Number of input parameters
    size_t out_dimension = {out_dimension}; // Number of output parameters
    int last_error = 0; // Last error code
    char last_error_msg[ERRMSG_LEN]; // error string buffer
}};

#define gpuErrchk(ans, data, context) gpuAssert((ans), data, __FILE__, __LINE__, context)
inline int gpuAssert(cudaError_t code, {name}_EvaluationData* data, const char *file, int line, const char *context)
{{
   if (code != cudaSuccess)
   {{
       const char* msg = cudaGetErrorString(code);
       if (msg) {{
           snprintf(
               data->last_error_msg,
               ERRMSG_LEN,
               "%s:%d:%s: CUDA error: %s",
                file,
                line,
                context,
                msg
            );
        }} else {{
            snprintf(
                data->last_error_msg,
                ERRMSG_LEN,
                "%s:%d:%s: CUDA error: unkown",
                file,
                line,
                context
            );
        }}
    }}
    // should always be 0
    if (data->last_error != 0) {{
        fprintf(stderr,
                "%s:%d:%s: CUDA fatal: previous error was not resolved",
                file,
                line,
                context
        );
        // flush output
        fflush(stderr);
        // we crash the evaluation since previous failure was not sanitized
        exit(-1);
    }}
    data->last_error = (int)code;
    return data->last_error;
}}



extern "C" {{

{name}_EvaluationData* {name}_init_data(size_t n, size_t block_size) {{
    {name}_EvaluationData* data = ({name}_EvaluationData*)malloc(sizeof({name}_EvaluationData));
    size_t in_dimension = {in_dimension};
    size_t out_dimension = {out_dimension};
    data->n = n;
    data->in_dimension = in_dimension;
    data->out_dimension = out_dimension;
    data->block_size = block_size;
    data->last_error = 0;
    // return data early since second failure => abort/crash code
    if(gpuErrchk(cudaMalloc((void**)&data->params, n*in_dimension * sizeof(CudaNumber)),data, "init_data_params")) return data;
    if(gpuErrchk(cudaMalloc((void**)&data->out, n*out_dimension*sizeof(CudaNumber)),data, "init_data_out")) return data;
    return data;
}}

int {name}_destroy_data({name}_EvaluationData* data) {{
    // since we free the evaluationData no error can be returned through it
    // neither a Result<(),String> return would make sense in rust drop
    cudaError_t error;
    error = cudaFree(data->params);
    if (error != cudaSuccess) return (int)error;
    error = cudaFree(data->out);
    if (error != cudaSuccess) return (int)error;
    free(data);
    return 0;
}}
}}
       "#,
            name = function_name,
            in_dimension = self.param_count,
            out_dimension = self.result_indices.len()
        );

        res += &format!(
            r#"
extern "C" {{
    __global__ void {name}_cuda(CudaNumber *params, CudaNumber *out, size_t n) {{
        int index = blockIdx.x * blockDim.x + threadIdx.x;
        if(index < n) {name}(params, out, index);
        return;
    }}
}}
"#,
            name = function_name
        );

        res += &format!(
            r#"
extern "C" {{
    void {name}_vec(Number *params, Number *out, {name}_EvaluationData* data) {{
        size_t n = data->n;
        size_t in_dimension = {in_dimension};
        size_t out_dimension = {out_dimension};

        if(gpuErrchk(cudaMemcpy(data->params, params, n*in_dimension * sizeof(CudaNumber), cudaMemcpyHostToDevice),data, "copy_data_params")) return;

        int blockSize = data->block_size; // Number of threads per block
        int gridSize = (n + blockSize - 1) / blockSize; // Number of blocks
        {name}_cuda<<<gridSize,blockSize>>>(data->params, data->out,n);
        // Collect launch errors
        if(gpuErrchk(cudaPeekAtLastError(), data, "launch")) return;
        // Collect runtime errors
        if(gpuErrchk(cudaDeviceSynchronize(), data, "runtime")) return;

        if(gpuErrchk(cudaMemcpy(out, data->out, n*out_dimension*sizeof(CudaNumber), cudaMemcpyDeviceToHost),data, "copy_data_out")) return;
        return;
    }}
}}
"#,
            name = function_name,
            in_dimension = self.param_count,
            out_dimension = self.result_indices.len()
        );

        res
    }

    fn export_generic_cpp_str(
        &self,
        function_name: &str,
        settings: &ExportSettings,
        number_class: NumberClass,
    ) -> String {
        let mut res = String::new();
        if settings.include_header {
            res += "#include <iostream>\n#include <cmath>\n\n";
            if number_class == NumberClass::ComplexF64 {
                res += "#include <complex>\n";
            } else {
                res += "template<typename T> T conj(T a) { return a; }\n";
            }
        };

        if number_class == NumberClass::ComplexF64 {
            res += "typedef std::complex<double> Number;\n";
        } else if number_class == NumberClass::RealF64 {
            res += "typedef double Number;\n";
        }

        if let Some(header) = &settings.custom_header {
            res += header;
            res += "\n";
        }

        res += &format!(
            "extern \"C\" unsigned long {}_get_buffer_len()\n{{\n\treturn {};\n}}\n\n",
            function_name,
            self.stack.len()
        );

        res += &format!(
            "\ntemplate<typename T>\nvoid {function_name}_gen(T* params, T* Z, T* out) {{\n"
        );

        self.export_cpp_impl("", "T", true, &mut res);

        for (i, r) in &mut self.result_indices.iter().enumerate() {
            res += &format!("\tout[{i}] = ");
            res += &if *r < self.param_count {
                format!("params[{r}]")
            } else if *r < self.reserved_indices {
                self.stack[*r].export_wrapped_with("T")
            } else {
                format!("Z[{r}]")
            };

            res += ";\n";
        }

        res += "\treturn;\n}\n";

        // if there are non-reals we can not use double evaluation
        assert!(
            !(!self.stack.iter().all(|x| x.is_real()) && number_class == NumberClass::RealF64),
            "Cannot export complex function with real numbers"
        );

        res
    }

    fn export_cpp_impl(
        &self,
        param_offset: &str,
        number_wrapper: &str,
        tmp_array: bool,
        out: &mut String,
    ) {
        macro_rules! get_input {
            ($i:expr) => {
                if $i < self.param_count {
                    format!("params[{}{}]", param_offset, $i)
                } else if $i < self.reserved_indices {
                    self.stack[$i].export_wrapped_with(number_wrapper)
                } else {
                    // TODO: subtract reserved indices
                    if tmp_array {
                        format!("Z[{}]", $i)
                    } else {
                        format!("Z{}", $i)
                    }
                }
            };
        }

        macro_rules! get_output {
            ($i:expr) => {
                if tmp_array {
                    format!("Z[{}]", $i)
                } else {
                    format!("Z{}", $i)
                }
            };
        }

        let mut close_else_branch = 0;
        for (ins, _c) in &self.instructions {
            match ins {
                Instr::Add(o, a) => {
                    let args = a
                        .iter()
                        .map(|x| get_input!(*x))
                        .collect::<Vec<_>>()
                        .join("+");

                    *out += format!("\t{} = {args};\n", get_output!(o)).as_str();
                }
                Instr::Mul(o, a) => {
                    let args = a
                        .iter()
                        .map(|x| get_input!(*x))
                        .collect::<Vec<_>>()
                        .join("*");

                    *out += format!("\t{} = {args};\n", get_output!(o)).as_str();
                }
                Instr::Pow(o, b, e) => {
                    let base = get_input!(*b);
                    if *e == -1 {
                        *out += format!("\t{} = {number_wrapper}(1) / {base};\n", get_output!(o))
                            .as_str();
                    } else {
                        *out += format!("\t{} = pow({base}, {e});\n", get_output!(o)).as_str();
                    }
                }
                Instr::Powf(o, b, e) => {
                    let base = get_input!(*b);
                    let exp = get_input!(*e);
                    *out += format!("\t{} = pow({base}, {exp});\n", get_output!(o)).as_str();
                }
                Instr::BuiltinFun(o, s, a) => match s.0.get_id() {
                    Symbol::EXP_ID => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = exp({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::LOG_ID => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = log({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::SIN_ID => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = sin({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::COS_ID => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = cos({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::SQRT_ID => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = sqrt({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::ABS_ID => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = std::abs({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::CONJ_ID => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = conj({arg});\n", get_output!(o)).as_str();
                    }
                    _ => unreachable!(),
                },
                Instr::ExternalFun(o, s, a) => {
                    let name = &self.external_fns[*s];
                    let args = a.iter().map(|x| get_input!(*x)).collect::<Vec<_>>();

                    *out +=
                        format!("\t{} = {}({});\n", get_output!(o), name, args.join(", ")).as_str();
                }
                Instr::IfElse(cond, _label) => {
                    *out += &format!("\tif ({} != 0.) {{\n", get_input!(*cond));
                }
                Instr::Goto(..) => {
                    *out += "\t} else {\n";
                    close_else_branch += 1;
                }
                Instr::Label(..) => {}
                Instr::Join(o, cond, a, b) => {
                    if close_else_branch > 0 {
                        close_else_branch -= 1;
                        *out += "\t}\n";
                    }
                    let arg_a = get_input!(*a);
                    let arg_b = get_input!(*b);
                    *out += format!(
                        "\t{} = ({} != 0.) ? {} : {};\n",
                        get_output!(o),
                        get_input!(*cond),
                        arg_a,
                        arg_b
                    )
                    .as_str();
                }
            }
        }
    }

    fn export_asm_real_str(&self, function_name: &str, settings: &ExportSettings) -> String {
        let mut res = String::new();
        if settings.include_header {
            res += "#include <iostream>\n#include <cmath>\n\n#include <complex>\n";
        };

        if let Some(header) = &settings.custom_header {
            res += header;
            res += "\n";
        }

        res += &format!(
            "extern \"C\" unsigned long {}_get_buffer_len()\n{{\n\treturn {};\n}}\n\n",
            function_name,
            self.stack.len()
        );

        if self.stack.iter().all(|x| x.is_real()) {
            res += &format!(
                "static const double {}_CONSTANTS_double[{}] = {{{}}};\n\n",
                function_name,
                self.reserved_indices - self.param_count + 1,
                {
                    let mut nums = (self.param_count..self.reserved_indices)
                        .map(|i| format!("double({})", self.stack[i].export()))
                        .collect::<Vec<_>>();
                    nums.push("1".to_string()); // used for inversion
                    nums.join(",")
                }
            );

            res += &format!(
                "extern \"C\" void {function_name}(const double *params, double* Z, double *out)\n{{\n"
            );

            self.export_asm_double_impl(
                &self.instructions,
                function_name,
                settings.inline_asm,
                &mut res,
            );

            res += "\treturn;\n}\n";
        } else {
            res += &format!(
                "extern \"C\" void {function_name}(const double *params, double* Z, double *out)\n{{\n\tstd::cout << \"Cannot evaluate complex function with doubles\" << std::endl;\n\treturn; \n}}",
            );
        }
        res
    }

    fn export_asm_complex_str(&self, function_name: &str, settings: &ExportSettings) -> String {
        let mut res = String::new();
        if settings.include_header {
            res += "#include <iostream>\n#include <complex>\n#include <cmath>\n\n";
        };

        if let Some(header) = &settings.custom_header {
            res += header;
            res += "\n";
        }

        res += &format!(
            "extern \"C\" unsigned long {}_get_buffer_len()\n{{\n\treturn {};\n}}\n\n",
            function_name,
            self.stack.len()
        );

        res += &format!(
            "static const std::complex<double> {}_CONSTANTS_complex[{}] = {{{}}};\n\n",
            function_name,
            self.reserved_indices - self.param_count + 2,
            {
                let mut nums = (self.param_count..self.reserved_indices)
                    .map(|i| format!("std::complex<double>({})", self.stack[i].export()))
                    .collect::<Vec<_>>();
                nums.push("std::complex<double>(0, -0.)".to_string()); // used for complex inversion
                nums.push("1".to_string()); // used for real inversion
                nums.join(",")
            }
        );

        res += &format!(
            "extern \"C\" void {function_name}(const std::complex<double> *params, std::complex<double> *Z, std::complex<double> *out)\n{{\n"
        );

        self.export_asm_complex_impl(
            &self.instructions,
            function_name,
            settings.inline_asm,
            &mut res,
        );

        res + "\treturn;\n}\n\n"
    }

    fn export_asm_double_impl(
        &self,
        instr: &[(Instr, ComplexPhase)],
        function_name: &str,
        asm_flavour: InlineASM,
        out: &mut String,
    ) -> bool {
        let mut second_index = 0;

        macro_rules! get_input {
            ($i:expr) => {
                if $i < self.param_count {
                    format!("params[{}]", $i)
                } else if $i < self.reserved_indices {
                    format!(
                        "{}_CONSTANTS_double[{}]",
                        function_name,
                        $i - self.param_count
                    )
                } else {
                    // TODO: subtract reserved indices
                    format!("Z[{}]", $i)
                }
            };
        }

        macro_rules! asm_load {
            ($i:expr) => {
                match asm_flavour {
                    InlineASM::X64 => {
                        if $i < self.param_count {
                            format!("{}(%2)", $i * 8)
                        } else if $i < self.reserved_indices {
                            format!("{}(%1)", ($i - self.param_count) * 8)
                        } else {
                            // TODO: subtract reserved indices
                            format!("{}(%0)", $i * 8)
                        }
                    }
                    InlineASM::AVX2 => {
                        if $i < self.param_count {
                            format!("{}(%2)", $i * 32)
                        } else if $i < self.reserved_indices {
                            format!("{}(%1)", ($i - self.param_count) * 32)
                        } else {
                            // TODO: subtract reserved indices
                            format!("{}(%0)", $i * 32)
                        }
                    }
                    InlineASM::AArch64 => {
                        if $i < self.param_count {
                            let dest = $i * 8;

                            if dest > 32760 {
                                // maximum allowed shift is 12 bits
                                let d = dest.ilog2();
                                let shift = d.min(12);
                                let coeff = dest / (1 << shift);
                                let rest = dest - (coeff << shift);
                                second_index = 0;
                                *out += &format!(
                                    "\t\t\"add x8, %2, {}, lsl {}\\n\\t\"\n",
                                    coeff, shift
                                );
                                format!("[x8, {}]", rest)
                            } else {
                                format!("[%2, {}]", dest)
                            }
                        } else if $i < self.reserved_indices {
                            let dest = ($i - self.param_count) * 8;
                            if dest > 32760 {
                                let d = dest.ilog2();
                                let shift = d.min(12);
                                let coeff = dest / (1 << shift);
                                let rest = dest - (coeff << shift);
                                second_index = 0;
                                *out += &format!(
                                    "\t\t\"add x8, %1, {}, lsl {}\\n\\t\"\n",
                                    coeff, shift
                                );
                                format!("[x8, {}]", rest)
                            } else {
                                format!("[%1, {}]", dest)
                            }
                        } else {
                            // TODO: subtract reserved indices
                            let dest = $i * 8;
                            if dest > 32760 && (dest < second_index || dest > 32760 + second_index)
                            {
                                let d = dest.ilog2();
                                let shift = d.min(12);
                                let coeff = dest / (1 << shift);
                                second_index = coeff << shift;
                                let rest = dest - second_index;
                                *out += &format!(
                                    "\t\t\"add x8, %0, {}, lsl {}\\n\\t\"\n",
                                    coeff, shift
                                );
                                format!("[x8, {}]", rest)
                            } else if dest <= 32760 {
                                format!("[%0, {}]", dest)
                            } else {
                                let offset = dest - second_index;
                                format!("[x8, {}]", offset)
                            }
                        }
                    }
                    InlineASM::None => unreachable!(),
                }
            };
        }

        macro_rules! end_asm_block {
            ($in_block: expr) => {
                if $in_block {
                    match asm_flavour {
                        InlineASM::X64 => {
                            *out += &format!("\t\t:\n\t\t: \"r\"(Z), \"r\"({}_CONSTANTS_double), \"r\"(params)\n\t\t: \"memory\", \"xmm0\", \"xmm1\", \"xmm2\", \"xmm3\", \"xmm4\", \"xmm5\", \"xmm6\", \"xmm7\", \"xmm8\", \"xmm9\", \"xmm10\", \"xmm11\", \"xmm12\", \"xmm13\", \"xmm14\", \"xmm15\");\n",  function_name);
                        }
                        InlineASM::AVX2 => {
                            *out += &format!("\t\t:\n\t\t: \"r\"(Z), \"r\"({}_CONSTANTS_double), \"r\"(params)\n\t\t: \"memory\", \"ymm0\", \"ymm1\", \"ymm2\", \"ymm3\", \"ymm4\", \"ymm5\", \"ymm6\", \"ymm7\", \"ymm8\", \"ymm9\", \"ymm10\", \"ymm11\", \"ymm12\", \"ymm13\", \"ymm14\", \"ymm15\");\n",  function_name);
                        }
                        InlineASM::AArch64 => {
                            *out += &format!("\t\t:\n\t\t: \"r\"(Z), \"r\"({}_CONSTANTS_double), \"r\"(params)\n\t\t: \"memory\", \"x8\", \"d0\", \"d1\", \"d2\", \"d3\", \"d4\", \"d5\", \"d6\", \"d7\", \"d8\", \"d9\", \"d10\", \"d11\", \"d12\", \"d13\", \"d14\", \"d15\", \"d16\", \"d17\", \"d18\", \"d19\", \"d20\", \"d21\", \"d22\", \"d23\", \"d24\", \"d25\", \"d26\", \"d27\", \"d28\", \"d29\", \"d30\", \"d31\");\n",  function_name);
                            #[allow(unused_assignments)] { second_index = 0;}; // the second index in x8 will be lost after the block, so reset it
                        }
                        InlineASM::None => unreachable!(),
                    }
                    $in_block = false;
                }
            };
        }

        let mut reg_last_use = vec![self.instructions.len(); self.instructions.len()];
        let mut stack_to_reg = HashMap::default();

        for (i, (ins, _)) in instr.iter().enumerate() {
            match ins {
                Instr::Add(r, a) | Instr::Mul(r, a) | Instr::ExternalFun(r, _, a) => {
                    for x in a {
                        if x >= &self.reserved_indices {
                            reg_last_use[stack_to_reg[x]] = i;
                        }
                    }

                    stack_to_reg.insert(r, i);
                }
                Instr::Pow(r, b, _) => {
                    if b >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[b]] = i;
                    }
                    stack_to_reg.insert(r, i);
                }
                Instr::Powf(r, b, e) => {
                    if b >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[b]] = i;
                    }
                    if e >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[e]] = i;
                    }
                    stack_to_reg.insert(r, i);
                }
                Instr::BuiltinFun(r, _, b) => {
                    if b >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[b]] = i;
                    }
                    stack_to_reg.insert(r, i);
                }
                Instr::IfElse(c, _) => {
                    if c >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[c]] = i;
                    }
                }
                Instr::Join(r, c, t, f) => {
                    if c >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[c]] = i;
                    }
                    if t >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[t]] = i;
                    }
                    if f >= &self.reserved_indices {
                        reg_last_use[stack_to_reg[f]] = i;
                    }
                    stack_to_reg.insert(r, i);
                }
                Instr::Goto(..) | Instr::Label(..) => {}
            }
        }

        for x in &self.result_indices {
            if x >= &self.reserved_indices {
                reg_last_use[stack_to_reg[x]] = self.instructions.len();
            }
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        enum MemOrReg {
            Mem(usize),
            Reg(usize),
        }

        #[derive(Debug, Clone)]
        #[allow(dead_code)]
        enum RegInstr {
            Add(MemOrReg, u16, Vec<MemOrReg>),
            Mul(MemOrReg, u16, Vec<MemOrReg>),
            Pow(MemOrReg, u16, MemOrReg, i64),
            Sqrt(MemOrReg, u16, MemOrReg),
            Powf(usize, usize, usize),
            BuiltinFun(usize, BuiltinSymbol, usize),
            ExternalFun(usize, usize, Vec<usize>),
            IfElse(usize),
            Goto,
            Label(Label),
            Join(usize, usize, usize, usize),
        }

        let mut new_instr: Vec<RegInstr> = instr
            .iter()
            .map(|(i, _)| match i {
                Instr::Add(r, a) => RegInstr::Add(
                    MemOrReg::Mem(*r),
                    u16::MAX,
                    a.iter().map(|x| MemOrReg::Mem(*x)).collect(),
                ),
                Instr::Mul(r, a) => RegInstr::Mul(
                    MemOrReg::Mem(*r),
                    u16::MAX,
                    a.iter().map(|x| MemOrReg::Mem(*x)).collect(),
                ),
                Instr::Pow(r, b, e) => {
                    RegInstr::Pow(MemOrReg::Mem(*r), u16::MAX, MemOrReg::Mem(*b), *e)
                }
                Instr::Powf(r, b, e) => RegInstr::Powf(*r, *b, *e),
                Instr::BuiltinFun(r, s, a) => {
                    if s.0 == Symbol::SQRT {
                        RegInstr::Sqrt(MemOrReg::Mem(*r), u16::MAX, MemOrReg::Mem(*a))
                    } else {
                        RegInstr::BuiltinFun(*r, *s, *a)
                    }
                }
                Instr::ExternalFun(r, s, a) => RegInstr::ExternalFun(*r, *s, a.clone()),
                Instr::IfElse(c, _) => RegInstr::IfElse(*c),
                Instr::Goto(_) => RegInstr::Goto,
                Instr::Label(l) => RegInstr::Label(*l),
                Instr::Join(r, c, a, b) => RegInstr::Join(*r, *c, *a, *b),
            })
            .collect();

        // sort the list of instructions based on the distance
        let mut reg_list = reg_last_use.iter().enumerate().collect::<Vec<_>>();
        reg_list.sort_by_key(|x| (*x.1 - x.0, x.0));

        'next: for (j, last_use) in reg_list {
            if *last_use == self.instructions.len() {
                continue;
            }

            let old_reg = if let RegInstr::Add(r, _, _)
            | RegInstr::Mul(r, _, _)
            | RegInstr::Pow(r, _, _, -1) = &new_instr[j]
            {
                if let MemOrReg::Mem(r) = r {
                    *r
                } else {
                    continue;
                }
            } else {
                continue;
            };

            // find free registers in the range
            // start at j+1 as we can recycle registers that are last used in iteration j
            let mut free_regs = u16::MAX & !(1 << 15); // leave xmmm15 open

            for k in &new_instr[j + 1..=*last_use] {
                match k {
                    RegInstr::Add(_, f, _)
                    | RegInstr::Mul(_, f, _)
                    | RegInstr::Pow(_, f, _, -1) => {
                        free_regs &= f;
                    }

                    _ => {
                        free_regs = 0; // the current instruction is not allowed to be used outside of ASM blocks
                    }
                }

                if free_regs == 0 {
                    continue 'next;
                }
            }

            if let Some(k) = (0..16).position(|k| free_regs & (1 << k) != 0) {
                if let RegInstr::Add(r, _, _) | RegInstr::Mul(r, _, _) | RegInstr::Pow(r, _, _, _) =
                    &mut new_instr[j]
                {
                    *r = MemOrReg::Reg(k);
                }

                for l in &mut new_instr[j + 1..=*last_use] {
                    match l {
                        RegInstr::Add(_, f, a) | RegInstr::Mul(_, f, a) => {
                            *f &= !(1 << k); // FIXME: do not set on last use?
                            for x in a {
                                if *x == MemOrReg::Mem(old_reg) {
                                    *x = MemOrReg::Reg(k);
                                }
                            }
                        }
                        RegInstr::Pow(_, f, a, -1) | RegInstr::Sqrt(_, f, a) => {
                            *f &= !(1 << k); // FIXME: do not set on last use?
                            if *a == MemOrReg::Mem(old_reg) {
                                *a = MemOrReg::Reg(k);
                            }
                        }
                        RegInstr::Pow(_, _, _, _) => {
                            panic!("use outside of ASM block");
                        }
                        RegInstr::Powf(_, a, b) => {
                            if *a == old_reg {
                                panic!("use outside of ASM block");
                            }
                            if *b == old_reg {
                                panic!("use outside of ASM block");
                            }
                        }
                        RegInstr::BuiltinFun(_, _, a) => {
                            if *a == old_reg {
                                panic!("use outside of ASM block");
                            }
                        }
                        RegInstr::ExternalFun(_, _, a) => {
                            if a.contains(&old_reg) {
                                panic!("use outside of ASM block");
                            }
                        }
                        RegInstr::IfElse(c) => {
                            if *c == old_reg {
                                panic!("use outside of ASM block");
                            }
                        }
                        RegInstr::Label(_) => {}
                        RegInstr::Goto => {}
                        RegInstr::Join(_, c, a, b) => {
                            if *c == old_reg || *a == old_reg || *b == old_reg {
                                panic!("use outside of ASM block");
                            }
                        }
                    }
                }

                // TODO: if last use is not already set to a register, we can set it to the current one
                // this prevents a copy
            }
        }

        let mut label_stack = vec![];
        let mut label_join_info = HashMap::default();
        let mut in_join_section = false;
        for (ins, _) in instr {
            if in_join_section && !matches!(ins, Instr::Join(..)) {
                in_join_section = false;
                label_stack.pop().unwrap();
            }

            match ins {
                Instr::IfElse(_, label) => {
                    label_stack.push((*label, None));
                }
                Instr::Goto(l) => {
                    if let Some(last) = label_stack.last_mut() {
                        last.1 = Some(*l);
                    }
                }
                Instr::Join(o, _, a, b) => {
                    in_join_section = true; // could be more than one join if vectorized

                    if let Some((label, label_2)) = label_stack.last() {
                        label_join_info
                            .entry(*label)
                            .or_insert(vec![])
                            .push((*o, *a));
                        label_join_info
                            .entry(label_2.unwrap())
                            .or_insert(vec![])
                            .push((*o, *b));
                    } else {
                        unreachable!("Goto without matching IfElse");
                    }
                }
                _ => {
                    in_join_section = false;
                }
            }
        }

        let mut in_asm_block = false;
        let mut next_label_is_true_branch_end = false;
        for ins in &new_instr {
            match ins {
                RegInstr::Add(o, free, a) | RegInstr::Mul(o, free, a) => {
                    if !in_asm_block {
                        *out += "\t__asm__(\n";
                        in_asm_block = true;
                    }

                    let oper = if matches!(ins, RegInstr::Add(_, _, _)) {
                        "add"
                    } else {
                        "mul"
                    };

                    match o {
                        MemOrReg::Reg(out_reg) => {
                            if let Some(j) = a.iter().find(|x| **x == MemOrReg::Reg(*out_reg)) {
                                // we can recycle the register completely
                                let mut first_skipped = false;
                                for i in a {
                                    if first_skipped || i != j {
                                        match i {
                                            MemOrReg::Reg(k) => match asm_flavour {
                                                InlineASM::X64 => {
                                                    *out += &format!(
                                                        "\t\t\"{oper}sd %%xmm{k}, %%xmm{out_reg}\\n\\t\"\n",
                                                    );
                                                }
                                                InlineASM::AVX2 => {
                                                    *out += &format!(
                                                        "\t\t\"v{oper}pd %%ymm{k}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n",
                                                    );
                                                }
                                                InlineASM::AArch64 => {
                                                    *out += &format!(
                                                        "\t\t\"f{oper} d{out_reg}, d{k}, d{out_reg}\\n\\t\"\n",
                                                    );
                                                }
                                                InlineASM::None => unreachable!(),
                                            },
                                            MemOrReg::Mem(k) => match asm_flavour {
                                                InlineASM::X64 => {
                                                    let addr = asm_load!(*k);
                                                    *out += &format!(
                                                        "\t\t\"{oper}sd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::AVX2 => {
                                                    let addr = asm_load!(*k);
                                                    *out += &format!(
                                                        "\t\t\"v{oper}pd {addr}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::AArch64 => {
                                                    let addr = asm_load!(*k);
                                                    *out += &format!(
                                                        "\t\t\"ldr d31, {addr}\\n\\t\"\n",
                                                    );

                                                    *out += &format!(
                                                        "\t\t\"f{oper} d{out_reg}, d31, d{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::None => unreachable!(),
                                            },
                                        }
                                    }
                                    first_skipped |= i == j;
                                }
                            } else if let Some(MemOrReg::Reg(j)) =
                                a.iter().find(|x| matches!(x, MemOrReg::Reg(_)))
                            {
                                match asm_flavour {
                                    InlineASM::X64 => {
                                        *out += &format!(
                                            "\t\t\"movapd %%xmm{j}, %%xmm{out_reg}\\n\\t\"\n"
                                        );
                                    }
                                    InlineASM::AVX2 => {
                                        *out += &format!(
                                            "\t\t\"vmovapd %%ymm{j}, %%ymm{out_reg}\\n\\t\"\n"
                                        );
                                    }
                                    InlineASM::AArch64 => {
                                        *out += &format!("\t\t\"fmov d{out_reg}, d{j}\\n\\t\"\n");
                                    }
                                    InlineASM::None => unreachable!(),
                                }

                                let mut first_skipped = false;
                                for i in a {
                                    if first_skipped || *i != MemOrReg::Reg(*j) {
                                        match i {
                                            MemOrReg::Reg(k) => match asm_flavour {
                                                InlineASM::X64 => {
                                                    *out += &format!(
                                                        "\t\t\"{oper}sd %%xmm{k}, %%xmm{out_reg}\\n\\t\"\n",
                                                    );
                                                }
                                                InlineASM::AVX2 => {
                                                    *out += &format!(
                                                        "\t\t\"v{oper}pd %%ymm{k}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n",
                                                    );
                                                }
                                                InlineASM::AArch64 => {
                                                    *out += &format!(
                                                        "\t\t\"f{oper} d{out_reg}, d{k}, d{out_reg}\\n\\t\"\n",
                                                    );
                                                }
                                                InlineASM::None => unreachable!(),
                                            },
                                            MemOrReg::Mem(k) => match asm_flavour {
                                                InlineASM::X64 => {
                                                    let addr = asm_load!(*k);
                                                    *out += &format!(
                                                        "\t\t\"{oper}sd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::AVX2 => {
                                                    let addr = asm_load!(*k);
                                                    *out += &format!(
                                                        "\t\t\"v{oper}pd {addr}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::AArch64 => {
                                                    let addr = asm_load!(*k);
                                                    *out += &format!(
                                                        "\t\t\"ldr d31, {addr}\\n\\t\"\n",
                                                    );

                                                    *out += &format!(
                                                        "\t\t\"f{oper} d{out_reg}, d31, d{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::None => unreachable!(),
                                            },
                                        }
                                    }
                                    first_skipped |= *i == MemOrReg::Reg(*j);
                                }
                            } else {
                                if let MemOrReg::Mem(k) = &a[0] {
                                    match asm_flavour {
                                        InlineASM::X64 => {
                                            let addr = asm_load!(*k);
                                            *out += &format!(
                                                "\t\t\"movsd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                            );
                                        }
                                        InlineASM::AVX2 => {
                                            let addr = asm_load!(*k);
                                            *out += &format!(
                                                "\t\t\"vmovapd {addr}, %%ymm{out_reg}\\n\\t\"\n"
                                            );
                                        }
                                        InlineASM::AArch64 => {
                                            let addr = asm_load!(*k);
                                            *out +=
                                                &format!("\t\t\"ldr d{out_reg}, {addr}\\n\\t\"\n",);
                                        }
                                        InlineASM::None => unreachable!(),
                                    }
                                } else {
                                    unreachable!();
                                }

                                for i in &a[1..] {
                                    if let MemOrReg::Mem(k) = i {
                                        match asm_flavour {
                                            InlineASM::X64 => {
                                                let addr = asm_load!(*k);
                                                *out += &format!(
                                                    "\t\t\"{oper}sd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AVX2 => {
                                                let addr = asm_load!(*k);
                                                *out += &format!(
                                                    "\t\t\"v{oper}pd {addr}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AArch64 => {
                                                let addr = asm_load!(*k);
                                                *out +=
                                                    &format!("\t\t\"ldr d31, {addr}\\n\\t\"\n",);

                                                *out += &format!(
                                                    "\t\t\"f{oper} d{out_reg}, d31, d{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::None => unreachable!(),
                                        }
                                    }
                                }
                            }
                        }
                        MemOrReg::Mem(out_mem) => {
                            // TODO: we would like a last-use check of the free here. Now we need to move
                            if let Some(out_reg) = (0..16).position(|k| free & (1 << k) != 0) {
                                if let Some(MemOrReg::Reg(j)) =
                                    a.iter().find(|x| matches!(x, MemOrReg::Reg(_)))
                                {
                                    match asm_flavour {
                                        InlineASM::X64 => {
                                            *out += &format!(
                                                "\t\t\"movapd %%xmm{j}, %%xmm{out_reg}\\n\\t\"\n"
                                            );
                                        }
                                        InlineASM::AVX2 => {
                                            *out += &format!(
                                                "\t\t\"vmovapd %%ymm{j}, %%ymm{out_reg}\\n\\t\"\n"
                                            );
                                        }
                                        InlineASM::AArch64 => {
                                            *out +=
                                                &format!("\t\t\"fmov d{out_reg}, d{j}\\n\\t\"\n");
                                        }
                                        InlineASM::None => unreachable!(),
                                    }

                                    let mut first_skipped = false;
                                    for i in a {
                                        if first_skipped || *i != MemOrReg::Reg(*j) {
                                            match i {
                                                MemOrReg::Reg(k) => match asm_flavour {
                                                    InlineASM::X64 => {
                                                        *out += &format!(
                                                            "\t\t\"{oper}sd %%xmm{k}, %%xmm{out_reg}\\n\\t\"\n"
                                                        );
                                                    }
                                                    InlineASM::AVX2 => {
                                                        *out += &format!(
                                                            "\t\t\"v{oper}pd %%ymm{k}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                        );
                                                    }
                                                    InlineASM::AArch64 => {
                                                        *out += &format!(
                                                            "\t\t\"f{oper} d{out_reg}, d{k}, d{out_reg}\\n\\t\"\n"
                                                        );
                                                    }
                                                    InlineASM::None => unreachable!(),
                                                },
                                                MemOrReg::Mem(k) => match asm_flavour {
                                                    InlineASM::X64 => {
                                                        let addr = asm_load!(*k);
                                                        *out += &format!(
                                                            "\t\t\"{oper}sd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                                        );
                                                    }
                                                    InlineASM::AVX2 => {
                                                        let addr = asm_load!(*k);
                                                        *out += &format!(
                                                            "\t\t\"v{oper}pd {addr}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                        );
                                                    }
                                                    InlineASM::AArch64 => {
                                                        let addr = asm_load!(*k);
                                                        *out += &format!(
                                                            "\t\t\"ldr d31, {addr}\\n\\t\"\n",
                                                        );

                                                        *out += &format!(
                                                            "\t\t\"f{oper} d{out_reg}, d31, d{out_reg}\\n\\t\"\n"
                                                        );
                                                    }
                                                    InlineASM::None => unreachable!(),
                                                },
                                            }
                                        }

                                        first_skipped |= *i == MemOrReg::Reg(*j);
                                    }
                                } else {
                                    if let MemOrReg::Mem(k) = &a[0] {
                                        let addr = asm_load!(*k);
                                        match asm_flavour {
                                            InlineASM::X64 => {
                                                *out += &format!(
                                                    "\t\t\"movsd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AVX2 => {
                                                *out += &format!(
                                                    "\t\t\"vmovapd {addr}, %%ymm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AArch64 => {
                                                *out += &format!(
                                                    "\t\t\"ldr d{out_reg}, {addr}\\n\\t\"\n",
                                                );
                                            }
                                            InlineASM::None => unreachable!(),
                                        }
                                    } else {
                                        unreachable!();
                                    }

                                    for i in &a[1..] {
                                        if let MemOrReg::Mem(k) = i {
                                            let addr = asm_load!(*k);
                                            match asm_flavour {
                                                InlineASM::X64 => {
                                                    *out += &format!(
                                                        "\t\t\"{oper}sd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::AVX2 => {
                                                    *out += &format!(
                                                        "\t\t\"v{oper}pd {addr}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                    );
                                                }
                                                InlineASM::AArch64 => {
                                                    *out += &format!(
                                                        "\t\t\"ldr d31, {addr}\\n\\t\"\n",
                                                    );

                                                    *out += &format!(
                                                        "\t\t\"f{oper} d{out_reg}, d31, d{out_reg}\\n\\t\"\n",
                                                    );
                                                }
                                                InlineASM::None => unreachable!(),
                                            }
                                        }
                                    }
                                }

                                let addr = asm_load!(*out_mem);
                                match asm_flavour {
                                    InlineASM::X64 => {
                                        *out += &format!(
                                            "\t\t\"movsd %%xmm{out_reg}, {addr}\\n\\t\"\n"
                                        );
                                    }
                                    InlineASM::AVX2 => {
                                        *out += &format!(
                                            "\t\t\"vmovupd %%ymm{out_reg}, {addr}\\n\\t\"\n"
                                        );
                                    }
                                    InlineASM::AArch64 => {
                                        *out += &format!("\t\t\"str d{out_reg}, {addr}\\n\\t\"\n",);
                                    }
                                    InlineASM::None => unreachable!(),
                                }
                            } else {
                                unreachable!("No free registers");
                                // move the value of xmm0 into the memory location of the output register
                                // and then swap later?
                            }
                        }
                    }
                }
                RegInstr::Pow(o, free, b, e) => {
                    if *e == -1 {
                        if !in_asm_block {
                            *out += "\t__asm__(\n";
                            in_asm_block = true;
                        }

                        match o {
                            MemOrReg::Reg(out_reg) => {
                                if *b == MemOrReg::Reg(*out_reg) {
                                    match asm_flavour {
                                        InlineASM::X64 => {
                                            if let Some(tmp_reg) =
                                                (0..16).position(|k| free & (1 << k) != 0)
                                            {
                                                *out += &format!(
                                                    "\t\t\"movapd %%xmm{out_reg}, %%xmm{tmp_reg}\\n\\t\"\n"
                                                );

                                                *out += &format!(
                                                    "\t\t\"movsd {}(%1), %%xmm{}\\n\\t\"\n",
                                                    (self.reserved_indices - self.param_count) * 8,
                                                    out_reg
                                                );

                                                *out += &format!(
                                                    "\t\t\"divsd %%xmm{tmp_reg}, %%xmm{out_reg}\\n\\t\"\n"
                                                );
                                            } else {
                                                panic!("No free registers for division")
                                            }
                                        }
                                        InlineASM::AVX2 => {
                                            if let Some(tmp_reg) =
                                                (0..16).position(|k| free & (1 << k) != 0)
                                            {
                                                *out += &format!(
                                                    "\t\t\"vmovapd %%ymm{out_reg}, %%ymm{tmp_reg}\\n\\t\"\n"
                                                );

                                                *out += &format!(
                                                    "\t\t\"vmovupd {}(%1), %%ymm{}\\n\\t\"\n",
                                                    (self.reserved_indices - self.param_count) * 32,
                                                    out_reg
                                                );

                                                *out += &format!(
                                                    "\t\t\"vdivsd %%ymm{tmp_reg}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                );
                                            } else {
                                                panic!("No free registers for division")
                                            }
                                        }
                                        InlineASM::AArch64 => {
                                            *out += &format!(
                                                "\t\t\"ldr d31, [%1, {}]\\n\\t\"\n",
                                                (self.reserved_indices - self.param_count) * 8
                                            );
                                            *out += &format!(
                                                "\t\t\"fdiv d{out_reg}, d31, d{out_reg}\\n\\t\"\n"
                                            );
                                        }
                                        InlineASM::None => unreachable!(),
                                    }
                                } else {
                                    // load 1 into out_reg
                                    match asm_flavour {
                                        InlineASM::X64 => {
                                            *out += &format!(
                                                "\t\t\"movsd {}(%1), %%xmm{}\\n\\t\"\n",
                                                (self.reserved_indices - self.param_count) * 8,
                                                out_reg,
                                            );
                                        }
                                        InlineASM::AVX2 => {
                                            *out += &format!(
                                                "\t\t\"vmovupd {}(%1), %%ymm{}\\n\\t\"\n",
                                                (self.reserved_indices - self.param_count) * 32,
                                                out_reg,
                                            );
                                        }
                                        InlineASM::AArch64 => {
                                            *out += &format!(
                                                "\t\t\"ldr d{}, [%1, {}]\\n\\t\"\n",
                                                out_reg,
                                                (self.reserved_indices - self.param_count) * 8
                                            );
                                        }
                                        InlineASM::None => unreachable!(),
                                    }

                                    match b {
                                        MemOrReg::Reg(j) => match asm_flavour {
                                            InlineASM::X64 => {
                                                *out += &format!(
                                                    "\t\t\"divsd %%xmm{j}, %%xmm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AVX2 => {
                                                *out += &format!(
                                                    "\t\t\"vdivpd %%ymm{j}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AArch64 => {
                                                *out += &format!(
                                                    "\t\t\"fdiv d{out_reg}, d{out_reg}, d{j}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::None => unreachable!(),
                                        },
                                        MemOrReg::Mem(k) => match asm_flavour {
                                            InlineASM::X64 => {
                                                let addr = asm_load!(*k);
                                                *out += &format!(
                                                    "\t\t\"divsd {addr}, %%xmm{out_reg}\\n\\t\"\n",
                                                );
                                            }
                                            InlineASM::AVX2 => {
                                                let addr = asm_load!(*k);
                                                *out += &format!(
                                                    "\t\t\"vdivpd {addr}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n",
                                                );
                                            }
                                            InlineASM::AArch64 => {
                                                let addr = asm_load!(*k);
                                                *out += &format!("\t\t\"ldr d31, {addr}\\n\\t\"\n");

                                                *out += &format!(
                                                    "\t\t\"fdiv d{out_reg}, d{out_reg}, d31\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::None => unreachable!(),
                                        },
                                    }
                                }
                            }
                            MemOrReg::Mem(out_mem) => {
                                if let Some(out_reg) = (0..16).position(|k| free & (1 << k) != 0) {
                                    match asm_flavour {
                                        InlineASM::X64 => {
                                            *out += &format!(
                                                "\t\t\"movsd {}(%1), %%xmm{}\\n\\t\"\n",
                                                (self.reserved_indices - self.param_count) * 8,
                                                out_reg
                                            );
                                        }
                                        InlineASM::AVX2 => {
                                            *out += &format!(
                                                "\t\t\"vmovupd {}(%1), %%ymm{}\\n\\t\"\n",
                                                (self.reserved_indices - self.param_count) * 32,
                                                out_reg
                                            );
                                        }
                                        InlineASM::AArch64 => {
                                            *out += &format!(
                                                "\t\t\"ldr d{}, [%1, {}]\\n\\t\"\n",
                                                out_reg,
                                                (self.reserved_indices - self.param_count) * 8
                                            );
                                        }
                                        InlineASM::None => unreachable!(),
                                    }

                                    match b {
                                        MemOrReg::Reg(j) => match asm_flavour {
                                            InlineASM::X64 => {
                                                *out += &format!(
                                                    "\t\t\"divsd %%xmm{j}, %%xmm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AVX2 => {
                                                *out += &format!(
                                                    "\t\t\"vdivpd %%ymm{j}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AArch64 => {
                                                *out += &format!(
                                                    "\t\t\"fdiv d{out_reg}, d{out_reg}, d{j}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::None => unreachable!(),
                                        },
                                        MemOrReg::Mem(k) => match asm_flavour {
                                            InlineASM::X64 => {
                                                let addr = asm_load!(*k);
                                                *out += &format!(
                                                    "\t\t\"divsd {addr}, %%xmm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AVX2 => {
                                                let addr = asm_load!(*k);
                                                *out += &format!(
                                                    "\t\t\"vdivpd {addr}, %%ymm{out_reg}, %%ymm{out_reg}\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::AArch64 => {
                                                let addr = asm_load!(*k);
                                                *out +=
                                                    &format!("\t\t\"ldr d31, {addr}\\n\\t\"\n",);

                                                *out += &format!(
                                                    "\t\t\"fdiv d{out_reg}, d{out_reg}, d31\\n\\t\"\n"
                                                );
                                            }
                                            InlineASM::None => unreachable!(),
                                        },
                                    }

                                    let addr = asm_load!(*out_mem);
                                    match asm_flavour {
                                        InlineASM::X64 => {
                                            *out += &format!(
                                                "\t\t\"movsd %%xmm{out_reg}, {addr}\\n\\t\"\n"
                                            );
                                        }
                                        InlineASM::AVX2 => {
                                            *out += &format!(
                                                "\t\t\"vmovupd %%ymm{out_reg}, {addr}\\n\\t\"\n"
                                            );
                                        }
                                        InlineASM::AArch64 => {
                                            *out +=
                                                &format!("\t\t\"str d{out_reg}, {addr}\\n\\t\"\n",);
                                        }
                                        InlineASM::None => unreachable!(),
                                    }
                                } else {
                                    unreachable!("No free registers");
                                    // move the value of xmm0 into the memory location of the output register
                                    // and then swap later?
                                }
                            }
                        }
                    } else {
                        unreachable!(
                            "Powers other than -1 should have been removed at an earlier stage"
                        );
                    }
                }
                RegInstr::Sqrt(o, free, b) => {
                    if !in_asm_block {
                        *out += "\t__asm__(\n";
                        in_asm_block = true;
                    }

                    let out_reg = match o {
                        MemOrReg::Reg(out_reg) => *out_reg,
                        MemOrReg::Mem(_) => {
                            let Some(out_reg) = (0..16).position(|k| free & (1 << k) != 0) else {
                                unreachable!("No free registers for sqrt output")
                            };
                            out_reg
                        }
                    };

                    let b_reg = match b {
                        MemOrReg::Reg(b_reg) => *b_reg,
                        MemOrReg::Mem(k) => {
                            match asm_flavour {
                                InlineASM::X64 => {
                                    let addr = asm_load!(*k);
                                    *out +=
                                        &format!("\t\t\"movsd {addr}, %%xmm{out_reg}\\n\\t\"\n",);
                                }
                                InlineASM::AVX2 => {
                                    let addr = asm_load!(*k);
                                    *out +=
                                        &format!("\t\t\"vmovupd {addr}, %%ymm{out_reg}\\n\\t\"\n",);
                                }
                                InlineASM::AArch64 => {
                                    let addr = asm_load!(*k);
                                    *out += &format!("\t\t\"ldr d{out_reg}, {addr}\\n\\t\"\n",);
                                }
                                InlineASM::None => unreachable!(),
                            }

                            out_reg
                        }
                    };

                    match asm_flavour {
                        InlineASM::X64 => {
                            *out += &format!("\t\t\"sqrtsd %%xmm{b_reg}, %%xmm{out_reg}\\n\\t\"\n");
                        }
                        InlineASM::AVX2 => {
                            *out +=
                                &format!("\t\t\"vsqrtpd %%ymm{b_reg}, %%ymm{out_reg}\\n\\t\"\n");
                        }
                        InlineASM::AArch64 => {
                            *out += &format!("\t\t\"fsqrt d{out_reg}, d{b_reg}\\n\\t\"\n");
                        }
                        InlineASM::None => unreachable!(),
                    }

                    if let MemOrReg::Mem(out_mem) = o {
                        let out_mem = asm_load!(*out_mem);
                        match asm_flavour {
                            InlineASM::X64 => {
                                *out += &format!("\t\t\"movsd %%xmm{out_reg}, {out_mem}\\n\\t\"\n");
                            }
                            InlineASM::AVX2 => {
                                *out +=
                                    &format!("\t\t\"vmovupd %%ymm{out_reg}, {out_mem}\\n\\t\"\n");
                            }
                            InlineASM::AArch64 => {
                                *out += &format!("\t\t\"str d{out_reg}, {out_mem}\\n\\t\"\n",);
                            }
                            InlineASM::None => unreachable!(),
                        }
                    }
                }
                RegInstr::Powf(o, b, e) => {
                    end_asm_block!(in_asm_block);

                    let base = get_input!(*b);
                    let exp = get_input!(*e);
                    *out += format!("\tZ[{o}] = pow({base}, {exp});\n").as_str();
                }
                RegInstr::BuiltinFun(o, s, a) => {
                    end_asm_block!(in_asm_block);

                    let arg = get_input!(*a);

                    match s.0.get_id() {
                        Symbol::EXP_ID => {
                            *out += format!("\tZ[{o}] = exp({arg});\n").as_str();
                        }
                        Symbol::LOG_ID => {
                            *out += format!("\tZ[{o}] = log({arg});\n").as_str();
                        }
                        Symbol::SIN_ID => {
                            *out += format!("\tZ[{o}] = sin({arg});\n").as_str();
                        }
                        Symbol::COS_ID => {
                            *out += format!("\tZ[{o}] = cos({arg});\n").as_str();
                        }
                        Symbol::ABS_ID => {
                            *out += format!("\tZ[{o}] = std::abs({arg});\n").as_str();
                        }
                        Symbol::CONJ_ID => {
                            *out += format!("\tZ[{o}] = {arg};\n").as_str();
                        }
                        _ => unreachable!(),
                    }
                }
                RegInstr::ExternalFun(o, s, a) => {
                    end_asm_block!(in_asm_block);

                    let name = &self.external_fns[*s];
                    let args = a.iter().map(|x| get_input!(*x)).collect::<Vec<_>>();

                    *out += format!("\tZ[{}] = {}({});\n", o, name, args.join(", ")).as_str();
                }
                RegInstr::IfElse(cond) => {
                    end_asm_block!(in_asm_block);

                    if asm_flavour == InlineASM::AVX2 {
                        *out += &format!("\tif (all({} != 0.)) {{\n", get_input!(*cond));
                    } else {
                        *out += &format!("\tif ({} != 0.) {{\n", get_input!(*cond));
                    }
                }
                RegInstr::Goto => {
                    next_label_is_true_branch_end = true;
                }
                RegInstr::Label(l) => {
                    end_asm_block!(in_asm_block);

                    for (o, b) in label_join_info.get(l).unwrap() {
                        let arg_a = get_input!(*o);
                        let arg_b = get_input!(*b);
                        *out += &format!("\t{} = {};\n", arg_a, arg_b);
                    }

                    if next_label_is_true_branch_end {
                        *out += "\t} else {\n";
                        next_label_is_true_branch_end = false;
                    } else {
                        *out += "\t}\n";
                    }
                }
                RegInstr::Join(_, _, _, _) => {}
            }
        }

        end_asm_block!(in_asm_block);

        let mut regcount = 0;
        *out += "\t__asm__(\n";
        for (i, r) in self.result_indices.iter().enumerate() {
            if *r < self.param_count {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!("\t\t\"movsd {}(%2), %%xmm{}\\n\\t\"\n", r * 8, regcount);
                    }
                    InlineASM::AVX2 => {
                        *out +=
                            &format!("\t\t\"vmovupd {}(%2), %%ymm{}\\n\\t\"\n", r * 32, regcount);
                    }
                    InlineASM::AArch64 => {
                        let addr = asm_load!(*r);
                        *out += &format!("\t\t\"ldr d{}, {}\\n\\t\"\n", regcount, addr);
                    }
                    InlineASM::None => unreachable!(),
                }
            } else if *r < self.reserved_indices {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!(
                            "\t\t\"movsd {}(%1), %%xmm{}\\n\\t\"\n",
                            (r - self.param_count) * 8,
                            regcount
                        );
                    }
                    InlineASM::AVX2 => {
                        *out += &format!(
                            "\t\t\"vmovupd {}(%1), %%ymm{}\\n\\t\"\n",
                            (r - self.param_count) * 32,
                            regcount
                        );
                    }
                    InlineASM::AArch64 => {
                        let addr = asm_load!(*r);
                        *out += &format!("\t\t\"ldr d{}, {}\\n\\t\"\n", regcount, addr);
                    }
                    InlineASM::None => unreachable!(),
                }
            } else {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!("\t\t\"movsd {}(%0), %%xmm{}\\n\\t\"\n", r * 8, regcount);
                    }
                    InlineASM::AVX2 => {
                        *out +=
                            &format!("\t\t\"vmovupd {}(%0), %%ymm{}\\n\\t\"\n", r * 32, regcount);
                    }
                    InlineASM::AArch64 => {
                        let addr = asm_load!(*r);
                        *out += &format!("\t\t\"ldr d{}, {}\\n\\t\"\n", regcount, addr);
                    }
                    InlineASM::None => unreachable!(),
                }
            }

            match asm_flavour {
                InlineASM::X64 => {
                    *out += &format!("\t\t\"movsd %%xmm{}, {}(%3)\\n\\t\"\n", regcount, i * 8);
                }
                InlineASM::AVX2 => {
                    *out += &format!("\t\t\"vmovupd %%ymm{}, {}(%3)\\n\\t\"\n", regcount, i * 32);
                }
                InlineASM::AArch64 => {
                    let dest = i * 8;
                    if dest > 32760 {
                        let d = dest.ilog2();
                        let shift = d.min(12);
                        let coeff = dest / (1 << shift);
                        let rest = dest - (coeff << shift);
                        second_index = 0;
                        *out += &format!("\t\t\"add x8, %3, {}, lsl {}\\n\\t\"\n", coeff, shift);
                        *out += &format!("\t\t\"str d{}, [x8, {}]\\n\\t\"\n", regcount, rest);
                    } else {
                        *out += &format!("\t\t\"str d{}, [%3, {}]\\n\\t\"\n", regcount, i * 8);
                    }
                }
                InlineASM::None => unreachable!(),
            }
            regcount = (regcount + 1) % 16;
        }

        match asm_flavour {
            InlineASM::X64 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(Z), \"r\"({function_name}_CONSTANTS_double), \"r\"(params), \"r\"(out)\n\t\t: \"memory\", \"xmm0\", \"xmm1\", \"xmm2\", \"xmm3\", \"xmm4\", \"xmm5\", \"xmm6\", \"xmm7\", \"xmm8\", \"xmm9\", \"xmm10\", \"xmm11\", \"xmm12\", \"xmm13\", \"xmm14\", \"xmm15\");\n"
                );
            }
            InlineASM::AVX2 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(Z), \"r\"({function_name}_CONSTANTS_double), \"r\"(params), \"r\"(out)\n\t\t: \"memory\", \"ymm0\", \"ymm1\", \"ymm2\", \"ymm3\", \"ymm4\", \"ymm5\", \"ymm6\", \"ymm7\", \"ymm8\", \"ymm9\", \"ymm10\", \"ymm11\", \"ymm12\", \"ymm13\", \"ymm14\", \"ymm15\");\n"
                );
            }
            InlineASM::AArch64 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(Z), \"r\"({function_name}_CONSTANTS_double), \"r\"(params), \"r\"(out)\n\t\t: \"memory\", \"x8\", \"d0\", \"d1\", \"d2\", \"d3\", \"d4\", \"d5\", \"d6\", \"d7\", \"d8\", \"d9\", \"d10\", \"d11\", \"d12\", \"d13\", \"d14\", \"d15\", \"d16\", \"d17\", \"d18\", \"d19\", \"d20\", \"d21\", \"d22\", \"d23\", \"d24\", \"d25\", \"d26\", \"d27\", \"d28\", \"d29\", \"d30\", \"d31\");\n"
                );
            }
            InlineASM::None => unreachable!(),
        }
        in_asm_block
    }

    fn export_asm_complex_impl(
        &self,
        instr: &[(Instr, ComplexPhase)],
        function_name: &str,
        asm_flavour: InlineASM,
        out: &mut String,
    ) -> bool {
        let mut second_index = 0;

        macro_rules! get_input {
            ($i:expr) => {
                if $i < self.param_count {
                    format!("params[{}]", $i)
                } else if $i < self.reserved_indices {
                    format!(
                        "{}_CONSTANTS_complex[{}]",
                        function_name,
                        $i - self.param_count
                    )
                } else {
                    // TODO: subtract reserved indices
                    format!("Z[{}]", $i)
                }
            };
        }

        macro_rules! asm_load {
            ($i:expr) => {
                match asm_flavour {
                    InlineASM::X64 => {
                        if $i < self.param_count {
                            (format!("{}(%2)", $i * 16), String::new())
                        } else if $i < self.reserved_indices {
                            (
                                format!("{}(%1)", ($i - self.param_count) * 16),
                                "NA".to_owned(),
                            )
                        } else {
                            // TODO: subtract reserved indices
                            (format!("{}(%0)", $i * 16), String::new())
                        }
                    }
                    InlineASM::AVX2 => {
                        if $i < self.param_count {
                            (format!("{}(%2)", $i * 64), format!("{}(%2)", $i * 64 + 32))
                        } else if $i < self.reserved_indices {
                            (
                                format!("{}(%1)", ($i - self.param_count) * 64),
                                format!("{}(%1)", ($i - self.param_count) * 64 + 32),
                            )
                        } else {
                            // TODO: subtract reserved indices
                            (format!("{}(%0)", $i * 64), format!("{}(%0)", $i * 64 + 32))
                        }
                    }
                    InlineASM::AArch64 => {
                        if $i < self.param_count {
                            let dest = $i * 16;

                            if dest > 32760 {
                                // maximum allowed shift is 12 bits
                                let d = dest.ilog2();
                                let shift = d.min(12);
                                let coeff = dest / (1 << shift);
                                let rest = dest - (coeff << shift);
                                second_index = 0;
                                *out += &format!(
                                    "\t\t\"add x8, %2, {}, lsl {}\\n\\t\"\n",
                                    coeff, shift
                                );
                                (format!("[x8, {}]", rest), format!("[x8, {}]", rest + 8))
                            } else {
                                (format!("[%2, {}]", dest), format!("[%2, {}]", dest + 8))
                            }
                        } else if $i < self.reserved_indices {
                            let dest = ($i - self.param_count) * 16;
                            if dest > 32760 {
                                let d = dest.ilog2();
                                let shift = d.min(12);
                                let coeff = dest / (1 << shift);
                                let rest = dest - (coeff << shift);
                                second_index = 0;
                                *out += &format!(
                                    "\t\t\"add x8, %1, {}, lsl {}\\n\\t\"\n",
                                    coeff, shift
                                );
                                (format!("[x8, {}]", rest), format!("[x8, {}]", rest + 8))
                            } else {
                                (format!("[%1, {}]", dest), format!("[%1, {}]", dest + 8))
                            }
                        } else {
                            // TODO: subtract reserved indices
                            let dest = $i * 16;
                            if dest > 32760 && (dest < second_index || dest > 32760 + second_index)
                            {
                                let d = dest.ilog2();
                                let shift = d.min(12);
                                let coeff = dest / (1 << shift);
                                second_index = coeff << shift;
                                let rest = dest - second_index;
                                *out += &format!(
                                    "\t\t\"add x8, %0, {}, lsl {}\\n\\t\"\n",
                                    coeff, shift
                                );
                                (format!("[x8, {}]", rest), format!("[x8, {}]", rest + 8))
                            } else if dest <= 32760 {
                                (format!("[%0, {}]", dest), format!("[%0, {}]", dest + 8))
                            } else {
                                let offset = dest - second_index;
                                (format!("[x8, {}]", offset), format!("[x8, {}]", offset + 8))
                            }
                        }
                    }
                    InlineASM::None => unreachable!(),
                }
            };
        }

        macro_rules! end_asm_block {
            ($in_block: expr) => {
                if $in_block {
                    match asm_flavour {
                        InlineASM::X64 => {
                            *out += &format!("\t\t:\n\t\t: \"r\"(Z), \"r\"({}_CONSTANTS_complex), \"r\"(params)\n\t\t: \"memory\", \"xmm0\", \"xmm1\", \"xmm2\", \"xmm3\", \"xmm4\", \"xmm5\", \"xmm6\", \"xmm7\", \"xmm8\", \"xmm9\", \"xmm10\", \"xmm11\", \"xmm12\", \"xmm13\", \"xmm14\", \"xmm15\");\n",  function_name);
                        }
                        InlineASM::AVX2 => {
                            *out += &format!("\t\t:\n\t\t: \"r\"(Z), \"r\"({}_CONSTANTS_complex), \"r\"(params)\n\t\t: \"memory\", \"ymm0\", \"ymm1\", \"ymm2\", \"ymm3\", \"ymm4\", \"ymm5\", \"ymm6\", \"ymm7\", \"ymm8\", \"ymm9\", \"ymm10\", \"ymm11\", \"ymm12\", \"ymm13\", \"ymm14\", \"ymm15\");\n",  function_name);
                        }
                        InlineASM::AArch64 => {
                            *out += &format!("\t\t:\n\t\t: \"r\"(Z), \"r\"({}_CONSTANTS_complex), \"r\"(params)\n\t\t: \"memory\", \"x8\", \"d0\", \"d1\", \"d2\", \"d3\", \"d4\", \"d5\", \"d6\", \"d7\", \"d8\", \"d9\", \"d10\", \"d11\", \"d12\", \"d13\", \"d14\", \"d15\", \"d16\", \"d17\", \"d18\", \"d19\", \"d20\", \"d21\", \"d22\", \"d23\", \"d24\", \"d25\", \"d26\", \"d27\", \"d28\", \"d29\", \"d30\", \"d31\");\n",  function_name);
                            #[allow(unused_assignments)] { second_index = 0;} // the second index in x8 will be lost after the block, so reset it
                        }
                        InlineASM::None => unreachable!(),
                    }
                    $in_block = false;
                }
            };
        }

        let mut label_stack = vec![];
        let mut label_join_info = HashMap::default();
        let mut in_join_section = false;
        for (ins, _) in instr {
            if in_join_section && !matches!(ins, Instr::Join(..)) {
                in_join_section = false;
                label_stack.pop().unwrap();
            }

            match ins {
                Instr::IfElse(_, label) => {
                    label_stack.push((*label, None));
                }
                Instr::Goto(l) => {
                    if let Some(last) = label_stack.last_mut() {
                        last.1 = Some(*l);
                    }
                }
                Instr::Join(o, _, a, b) => {
                    in_join_section = true; // could be more than one join if vectorized

                    if let Some((label, label_2)) = label_stack.last() {
                        label_join_info
                            .entry(*label)
                            .or_insert(vec![])
                            .push((*o, *a));
                        label_join_info
                            .entry(label_2.unwrap())
                            .or_insert(vec![])
                            .push((*o, *b));
                    } else {
                        unreachable!("Goto without matching IfElse");
                    }
                }
                _ => {
                    in_join_section = false;
                }
            }
        }

        let mut in_asm_block = false;
        let mut next_label_is_true_branch_end = false;
        for (ins, c) in instr {
            match ins {
                Instr::Add(o, a) => {
                    if !in_asm_block {
                        *out += "\t__asm__(\n";
                        in_asm_block = true;
                    }

                    match asm_flavour {
                        InlineASM::X64 => {
                            let (addr, _) = asm_load!(a[0]);
                            *out += &format!("\t\t\"movupd {addr}, %%xmm0\\n\\t\"\n");

                            for i in &a[1..] {
                                let (addr, _) = asm_load!(*i);
                                *out += &format!("\t\t\"movupd {addr}, %%xmm1\\n\\t\"\n");
                                *out += &format!("\t\t\"addpd %%xmm1, %%xmm0\\n\\t\"\n");
                            }
                            let (addr, _) = asm_load!(*o);
                            *out += &format!("\t\t\"movupd %%xmm0, {addr}\\n\\t\"\n");
                        }
                        InlineASM::AVX2 => {
                            let (addr, comp_addr) = asm_load!(a[0]);
                            *out += &format!("\t\t\"vmovupd {addr}, %%ymm0\\n\\t\"\n");
                            *out += &format!("\t\t\"vmovupd {comp_addr}, %%ymm1\\n\\t\"\n");

                            for i in &a[1..] {
                                let (addr, imag_addr) = asm_load!(*i);
                                *out += &format!("\t\t\"vaddpd {addr}, %%ymm0, %%ymm0\\n\\t\"\n");
                                *out +=
                                    &format!("\t\t\"vaddpd {imag_addr}, %%ymm1, %%ymm1\\n\\t\"\n");
                            }
                            let (addr, imag_addr) = asm_load!(*o);
                            *out += &format!("\t\t\"vmovupd %%ymm0, {addr}\\n\\t\"\n");
                            *out += &format!("\t\t\"vmovupd %%ymm1, {imag_addr}\\n\\t\"\n");
                        }
                        InlineASM::AArch64 => {
                            let (addr, _) = asm_load!(a[0]);
                            *out += &format!("\t\t\"ldr q0, {addr}\\n\\t\"\n");

                            for i in &a[1..] {
                                let (addr, _) = asm_load!(*i);
                                *out += &format!("\t\t\"ldr q1, {addr}\\n\\t\"\n");
                                *out += "\t\t\"fadd v0.2d, v1.2d, v0.2d\\n\\t\"\n";
                            }

                            let (addr, _) = asm_load!(*o);
                            *out += &format!("\t\t\"str q0, {addr}\\n\\t\"\n");
                        }
                        InlineASM::None => unreachable!(),
                    }
                }
                Instr::Mul(o, a) => {
                    if !in_asm_block {
                        *out += "\t__asm__(\n";
                        in_asm_block = true;
                    }

                    macro_rules! load_complex {
                        ($i: expr, $r: expr) => {
                            let (addr_re, addr_im) = asm_load!($r);
                            match asm_flavour {
                                InlineASM::X64 => {
                                    *out += &format!(
                                        "\t\t\"movupd {}, %%xmm{}\\n\\t\"\n",
                                        addr_re,
                                        $i + 1,
                                    );
                                }
                                InlineASM::AVX2 => {
                                    *out += &format!(
                                        "\t\t\"vmovupd {}, %%ymm{}\\n\\t\"\n",
                                        addr_re,
                                        2 * $i,
                                    );
                                    *out += &format!(
                                        "\t\t\"vmovupd {}, %%ymm{}\\n\\t\"\n",
                                        addr_im,
                                        2 * $i + 1,
                                    );
                                }
                                InlineASM::AArch64 => {
                                    if $r * 16 < 450 {
                                        *out += &format!(
                                            "\t\t\"ldp d{}, d{}, {}\\n\\t\"\n",
                                            2 * ($i + 1),
                                            2 * ($i + 1) + 1,
                                            addr_re,
                                        );
                                    } else {
                                        *out += &format!(
                                            "\t\t\"ldr d{}, {}\\n\\t\"\n",
                                            2 * ($i + 1),
                                            addr_re,
                                        );
                                        *out += &format!(
                                            "\t\t\"ldr d{}, {}\\n\\t\"\n",
                                            2 * ($i + 1) + 1,
                                            addr_im,
                                        );
                                    }
                                }
                                InlineASM::None => unreachable!(),
                            }
                        };
                    }

                    macro_rules! mul_complex {
                        ($i: expr, $real: expr) => {
                            match asm_flavour {
                                InlineASM::X64 => {
                                    if $real {
                                        *out += &format!(
                                            "\t\t\"mulpd %%xmm{0}, %%xmm1\\n\\t\"\n",
                                            $i + 1
                                        );
                                    } else {
                                        *out += &format!(
                                            "\t\t\"movapd %%xmm1, %%xmm0\\n\\t\"
\t\t\"unpckhpd %%xmm0, %%xmm0\\n\\t\"
\t\t\"unpcklpd %%xmm1, %%xmm1\\n\\t\"
\t\t\"mulpd %%xmm{0}, %%xmm0\\n\\t\"
\t\t\"mulpd %%xmm{0}, %%xmm1\\n\\t\"
\t\t\"shufpd $1, %%xmm0, %%xmm0\\n\\t\"
\t\t\"addsubpd %%xmm0, %%xmm1\\n\\t\"\n",
                                            $i + 1
                                        );
                                    }
                                }
                                InlineASM::AVX2 => {
                                    if $real {
                                        *out += &format!(
                                            "\t\t\"vmulpd %%ymm{0}, %%ymm0\\n\\t\"\n",
                                            $i + 1
                                        );
                                        *out +=
                                            &format!("\t\t\"vxorpd %%ymm1, %%ymm1, %%ymm1\\n\\t\""); // im = 0
                                    } else {
                                        *out += &format!(
                                            "\t\t\"vmulpd %%ymm0, %%ymm{0}, %%ymm14\\n\\t\"
\t\t\"vmulpd %%ymm0, %%ymm{1}, %%ymm15\\n\\t\"
\t\t\"vmulpd %%ymm1, %%ymm{1}, %%ymm0\\n\\t\"
\t\t\"vmulpd %%ymm1, %%ymm{0}, %%ymm{1}\\n\\t\"
\t\t\"vsubpd %%ymm0, %%ymm14, %%ymm0\\n\\t\"
\t\t\"vaddpd %%ymm15, %%ymm{1}, %%ymm1\\n\\t\"\n",
                                            2 * $i,
                                            2 * $i + 1,
                                        );
                                    }
                                }
                                InlineASM::AArch64 => {
                                    if $real {
                                        *out += &format!(
                                            "\t\t\"fmul d2, d{}, d2\\n\\t\"\n",
                                            2 * ($i + 1)
                                        );
                                        *out += &format!("\t\t\"fmov d3, xzr\\n\\t\""); // im = 0
                                    } else {
                                        *out += &format!(
                                            "
\t\t\"fmul    d0, d{0}, d3\\n\\t\"
\t\t\"fmul    d1, d{1}, d3\\n\\t\"
\t\t\"fmadd   d3, d{0}, d2, d1\\n\\t\"
\t\t\"fnmsub  d2, d{1}, d2, d0\\n\\t\"\n",
                                            2 * ($i + 1) + 1,
                                            2 * ($i + 1),
                                        )
                                    }
                                }
                                InlineASM::None => unreachable!(),
                            }
                        };
                    }

                    let num_real_args = match c {
                        ComplexPhase::Real => a.len(),
                        ComplexPhase::PartialReal(n) => *n,
                        ComplexPhase::Imag | ComplexPhase::Any => 0,
                    };

                    if !matches!(asm_flavour, InlineASM::AVX2) && a.len() < 15 || a.len() < 8 {
                        for (i, r) in a.iter().enumerate() {
                            load_complex!(i, *r);
                        }

                        for i in 1..a.len() {
                            // optimized complex multiplication
                            mul_complex!(i, i < num_real_args);
                        }
                    } else {
                        load_complex!(0, a[0]);

                        // load multiplications one after the other
                        for (i, r) in a.iter().enumerate().skip(1) {
                            load_complex!(1, *r);
                            mul_complex!(1, i < num_real_args);
                        }
                    }

                    let (addr_re, addr_im) = asm_load!(*o);
                    match asm_flavour {
                        InlineASM::X64 => {
                            *out += &format!("\t\t\"movupd %%xmm1, {addr_re}\\n\\t\"\n");
                        }
                        InlineASM::AVX2 => {
                            *out += &format!("\t\t\"vmovupd %%ymm0, {addr_re}\\n\\t\"\n");
                            *out += &format!("\t\t\"vmovupd %%ymm1, {addr_im}\\n\\t\"\n");
                        }
                        InlineASM::AArch64 => {
                            if *o * 16 < 450 {
                                *out += &format!("\t\t\"stp d2, d3, {addr_re}\\n\\t\"\n");
                            } else {
                                *out += &format!("\t\t\"str d2, {addr_re}\\n\\t\"\n");
                                *out += &format!("\t\t\"str d3, {addr_im}\\n\\t\"\n");
                            }
                        }
                        InlineASM::None => unreachable!(),
                    };
                }
                Instr::Pow(o, b, e) => {
                    if *e == -1 {
                        if !in_asm_block {
                            *out += "\t__asm__(\n";
                            in_asm_block = true;
                        }

                        let addr_b = asm_load!(*b);
                        let addr_o = asm_load!(*o);
                        match asm_flavour {
                            InlineASM::X64 => {
                                if let ComplexPhase::Real = *c {
                                    *out += &format!(
                                        "\t\t\"movupd {}, %%xmm0\\n\\t\"
\t\t\"movupd {}(%1), %%xmm1\\n\\t\"
\t\t\"divsd %%xmm0, %%xmm1\\n\\t\"
\t\t\"movupd %%xmm1, {}\\n\\t\"\n",
                                        addr_b.0,
                                        (self.reserved_indices - self.param_count + 1) * 16,
                                        addr_o.0
                                    );
                                } else {
                                    *out += &format!(
                                        "\t\t\"movupd {}, %%xmm0\\n\\t\"
\t\t\"movupd {}(%1), %%xmm1\\n\\t\"
\t\t\"movapd %%xmm0, %%xmm2\\n\\t\"
\t\t\"xorpd %%xmm1, %%xmm0\\n\\t\"
\t\t\"mulpd %%xmm2, %%xmm2\\n\\t\"
\t\t\"haddpd %%xmm2, %%xmm2\\n\\t\"
\t\t\"divpd %%xmm2, %%xmm0\\n\\t\"
\t\t\"movupd %%xmm0, {}\\n\\t\"\n",
                                        addr_b.0,
                                        (self.reserved_indices - self.param_count) * 16,
                                        addr_o.0
                                    );
                                }
                            }
                            InlineASM::AVX2 => {
                                if let ComplexPhase::Real = *c {
                                    *out += &format!(
                                        "\t\t\"vmovupd {0}, %%ymm0\\n\\t\"
\t\t\"vmovupd {1}(%1), %%ymm1\\n\\t\"
\t\t\"vdivpd  %%ymm0, %%ymm1, %%ymm0\\n\\t\"
\t\t\"vmovupd %%ymm0, {2}\\n\\t\"
\t\t\"vxorpd %%ymm1, %%ymm1, %%ymm1\\n\\t\"
\t\t\"vmovupd %%ymm1, {3}\\n\\t\"\n",
                                        addr_b.0,
                                        (self.reserved_indices - self.param_count + 1) * 64,
                                        addr_o.0,
                                        addr_o.1
                                    );
                                } else {
                                    // TODO: do FMA on top?
                                    *out += &format!(
                                        "\t\t\"vmovupd {0}, %%ymm0\\n\\t\"
\t\t\"vmovupd {1}, %%ymm1\\n\\t\"
\t\t\"vmulpd %%ymm0, %%ymm0, %%ymm3\\n\\t\"
\t\t\"vmulpd %%ymm1, %%ymm1, %%ymm4\\n\\t\"
\t\t\"vaddpd %%ymm3, %%ymm4, %%ymm3\\n\\t\"
\t\t\"vdivpd %%ymm3, %%ymm0, %%ymm0\\n\\t\"
\t\t\"vbroadcastsd {2}(%1), %%ymm4\\n\\t\"
\t\t\"vxorpd %%ymm4, %%ymm1, %%ymm1\\n\\t\"
\t\t\"vdivpd %%ymm3, %%ymm1, %%ymm1\\n\\t\"
\t\t\"vmovupd %%ymm0, {3}\\n\\t\"
\t\t\"vmovupd %%ymm1, {4}\\n\\t\"\n",
                                        addr_b.0,
                                        addr_b.1,
                                        (self.reserved_indices - self.param_count) * 64,
                                        addr_o.0,
                                        addr_o.1
                                    );
                                }
                            }
                            InlineASM::AArch64 => {
                                if *b * 16 < 450 {
                                    *out += &format!("\t\t\"ldp d0, d1, {}\\n\\t\"\n", addr_b.0);
                                } else {
                                    *out += &format!("\t\t\"ldr d0, {}\\n\\t\"\n", addr_b.0);
                                    *out += &format!("\t\t\"ldr d1, {}\\n\\t\"\n", addr_b.1);
                                }

                                if let ComplexPhase::Real = *c {
                                    *out += &format!(
                                        "\t\t\"ldr    d2, [%1, {}]\\n\\t\"
\t\t\"fdiv    d0, d2, d0\\n\\t\"\n",
                                        (self.reserved_indices - self.param_count + 1) * 16
                                    );
                                } else {
                                    *out += "
\t\t\"fmul    d2, d0, d0\\n\\t\"
\t\t\"fmadd   d2, d1, d1, d2\\n\\t\"
\t\t\"fneg    d1, d1\\n\\t\"
\t\t\"fdiv    d0, d0, d2\\n\\t\"
\t\t\"fdiv    d1, d1, d2\\n\\t\"\n";
                                }

                                if *o * 16 < 450 {
                                    *out += &format!("\t\t\"stp d0, d1, {}\\n\\t\"\n", addr_o.0);
                                } else {
                                    *out += &format!("\t\t\"str d0, {}\\n\\t\"\n", addr_o.0);
                                    *out += &format!("\t\t\"str d1, {}\\n\\t\"\n", addr_o.1);
                                }
                            }
                            InlineASM::None => unreachable!(),
                        }
                    } else {
                        end_asm_block!(in_asm_block);

                        let base = get_input!(*b);
                        *out += format!("\tZ[{o}] = pow({base}, {e});\n").as_str();
                    }
                }
                Instr::Powf(o, b, e) => {
                    end_asm_block!(in_asm_block);
                    let base = get_input!(*b);
                    let exp = get_input!(*e);

                    let suffix = if let ComplexPhase::Real = *c {
                        ".real()"
                    } else {
                        ""
                    };

                    *out += format!("\tZ[{o}] = pow({base}{suffix}, {exp}{suffix});\n").as_str();
                }
                Instr::BuiltinFun(o, s, a) => {
                    if in_asm_block
                        && s.0.get_id() == Symbol::SQRT_ID
                        && let ComplexPhase::Real = *c
                    {
                        let addr_a = asm_load!(*a);
                        let addr_o = asm_load!(*o);

                        match asm_flavour {
                            InlineASM::X64 => {
                                *out += &format!(
                                    "\t\t\"movupd {}, %%xmm0\\n\\t\"
\t\t\"sqrtsd %%xmm0, %%xmm0\\n\\t\"
\t\t\"movupd %%xmm0, {}\\n\\t\"\n",
                                    addr_a.0, addr_o.0
                                );
                            }
                            InlineASM::AVX2 => {
                                *out += &format!(
                                    "\t\t\"vmovupd {}, %%ymm0\\n\\t\"
\t\t\"vsqrtpd %%ymm0, %%ymm0\\n\\t\"
\t\t\"vxorpd %%ymm1, %%ymm1, %%ymm1\\n\\t\"
\t\t\"vmovupd %%ymm0, {}\\n\\t\"
\t\t\"vmovupd %%ymm1, {}\\n\\t\"\n",
                                    addr_a.0, addr_o.0, addr_o.1
                                );
                            }
                            InlineASM::AArch64 => {
                                *out += &format!(
                                    "\t\t\"ldr d0, {}\\n\\t\"
\t\t\"fsqrt d0, d0\\n\\t\"
\t\t\"str d0, {}\\n\\t\"\n",
                                    addr_a.0, addr_o.0
                                );
                            }
                            InlineASM::None => unreachable!(),
                        }

                        continue;
                    }

                    end_asm_block!(in_asm_block);

                    let arg = if let ComplexPhase::Real = *c {
                        get_input!(*a) + ".real()"
                    } else {
                        get_input!(*a)
                    };

                    match s.0.get_id() {
                        Symbol::EXP_ID => {
                            *out += format!("\tZ[{o}] = exp({arg});\n").as_str();
                        }
                        Symbol::LOG_ID => {
                            *out += format!("\tZ[{o}] = log({arg});\n").as_str();
                        }
                        Symbol::SIN_ID => {
                            *out += format!("\tZ[{o}] = sin({arg});\n").as_str();
                        }
                        Symbol::COS_ID => {
                            *out += format!("\tZ[{o}] = cos({arg});\n").as_str();
                        }
                        Symbol::SQRT_ID => {
                            *out += format!("\tZ[{o}] = sqrt({arg});\n").as_str();
                        }
                        Symbol::ABS_ID => {
                            *out += format!("\tZ[{o}] = std::abs({arg});\n").as_str();
                        }
                        Symbol::CONJ_ID => {
                            if let ComplexPhase::Real = *c {
                                *out += format!("\tZ[{o}] = {arg};\n").as_str();
                            } else {
                                *out += format!("\tZ[{o}] = conj({arg});\n").as_str();
                            }
                        }
                        _ => unreachable!(),
                    }
                }
                Instr::ExternalFun(o, s, a) => {
                    end_asm_block!(in_asm_block);

                    let name = &self.external_fns[*s];
                    let args = a.iter().map(|x| get_input!(*x)).collect::<Vec<_>>();

                    *out += format!("\tZ[{}] = {}({});\n", o, name, args.join(", ")).as_str();
                }
                Instr::IfElse(cond, _) => {
                    end_asm_block!(in_asm_block);

                    if asm_flavour == InlineASM::AVX2 {
                        *out += &format!("\tif (all({} != 0.)) {{\n", get_input!(*cond));
                    } else {
                        *out += &format!("\tif ({} != 0.) {{\n", get_input!(*cond));
                    }
                }
                Instr::Goto(_) => {
                    next_label_is_true_branch_end = true;
                }
                Instr::Label(l) => {
                    end_asm_block!(in_asm_block);

                    for (o, b) in label_join_info.get(l).unwrap() {
                        let arg_a = get_input!(*o);
                        let arg_b = get_input!(*b);
                        *out += &format!("\t{} = {};\n", arg_a, arg_b);
                    }

                    if next_label_is_true_branch_end {
                        *out += "\t} else {\n";
                        next_label_is_true_branch_end = false;
                    } else {
                        *out += "\t}\n";
                    }
                }
                Instr::Join(_, _, _, _) => {}
            }
        }

        end_asm_block!(in_asm_block);

        *out += "\t__asm__(\n";
        for (i, r) in &mut self.result_indices.iter().enumerate() {
            if *r < self.param_count {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!("\t\t\"movupd {}(%2), %%xmm0\\n\\t\"\n", r * 16);
                    }
                    InlineASM::AVX2 => {
                        *out += &format!("\t\t\"vmovupd {}(%2), %%ymm0\\n\\t\"\n", r * 64);
                        *out += &format!("\t\t\"vmovupd {}(%2), %%ymm1\\n\\t\"\n", r * 64 + 32);
                    }
                    InlineASM::AArch64 => {
                        let (addr_re, _) = asm_load!(*r);
                        *out += &format!("\t\t\"ldr q0, {}\\n\\t\"\n", addr_re);
                    }
                    InlineASM::None => unreachable!(),
                }
            } else if *r < self.reserved_indices {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!(
                            "\t\t\"movupd {}(%1), %%xmm0\\n\\t\"\n",
                            (r - self.param_count) * 16
                        );
                    }
                    InlineASM::AVX2 => {
                        *out += &format!(
                            "\t\t\"vmovupd {}(%1), %%ymm0\\n\\t\"\n",
                            (r - self.param_count) * 64
                        );
                        *out += &format!(
                            "\t\t\"vmovupd {}(%1), %%ymm1\\n\\t\"\n",
                            (r - self.param_count) * 64 + 32
                        );
                    }
                    InlineASM::AArch64 => {
                        let (addr_re, _) = asm_load!(*r);
                        *out += &format!("\t\t\"ldr q0, {}\\n\\t\"\n", addr_re);
                    }

                    InlineASM::None => unreachable!(),
                }
            } else {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!("\t\t\"movupd {}(%0), %%xmm0\\n\\t\"\n", r * 16);
                    }
                    InlineASM::AVX2 => {
                        *out += &format!("\t\t\"vmovupd {}(%0), %%ymm0\\n\\t\"\n", r * 64);
                        *out += &format!("\t\t\"vmovupd {}(%0), %%ymm1\\n\\t\"\n", r * 64 + 32);
                    }
                    InlineASM::AArch64 => {
                        let (addr_re, _) = asm_load!(*r);
                        *out += &format!("\t\t\"ldr q0, {}\\n\\t\"\n", addr_re);
                    }
                    InlineASM::None => unreachable!(),
                }
            }

            match asm_flavour {
                InlineASM::X64 => {
                    *out += &format!("\t\t\"movupd %%xmm0, {}(%3)\\n\\t\"\n", i * 16);
                }
                InlineASM::AVX2 => {
                    *out += &format!("\t\t\"vmovupd %%ymm0, {}(%3)\\n\\t\"\n", i * 64);
                    *out += &format!("\t\t\"vmovupd %%ymm1, {}(%3)\\n\\t\"\n", i * 64 + 32);
                }
                InlineASM::AArch64 => {
                    let dest = i * 16;
                    if dest > 32760 {
                        let d = dest.ilog2();
                        let shift = d.min(12);
                        let coeff = dest / (1 << shift);
                        let rest = dest - (coeff << shift);
                        second_index = 0;
                        *out += &format!("\t\t\"add x8, %3, {}, lsl {}\\n\\t\"\n", coeff, shift);
                        *out += &format!("\t\t\"str q0, [x8, {}]\\n\\t\"\n", rest);
                    } else {
                        *out += &format!("\t\t\"str q0, [%3, {}]\\n\\t\"\n", dest);
                    }
                }
                InlineASM::None => unreachable!(),
            }
        }

        match asm_flavour {
            InlineASM::X64 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(Z), \"r\"({function_name}_CONSTANTS_complex), \"r\"(params), \"r\"(out)\n\t\t: \"memory\", \"xmm0\", \"xmm1\", \"xmm2\", \"xmm3\", \"xmm4\", \"xmm5\", \"xmm6\", \"xmm7\", \"xmm8\", \"xmm9\", \"xmm10\", \"xmm11\", \"xmm12\", \"xmm13\", \"xmm14\", \"xmm15\");\n"
                );
            }
            InlineASM::AVX2 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(Z), \"r\"({function_name}_CONSTANTS_complex), \"r\"(params), \"r\"(out)\n\t\t: \"memory\", \"ymm0\", \"ymm1\", \"ymm2\", \"ymm3\", \"ymm4\", \"ymm5\", \"ymm6\", \"ymm7\", \"ymm8\", \"ymm9\", \"ymm10\", \"ymm11\", \"ymm12\", \"ymm13\", \"ymm14\", \"ymm15\");\n"
                );
            }
            InlineASM::AArch64 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(Z), \"r\"({function_name}_CONSTANTS_complex), \"r\"(params), \"r\"(out)\n\t\t: \"memory\", \"x8\", \"d0\", \"d1\", \"d2\", \"d3\", \"d4\", \"d5\", \"d6\", \"d7\", \"d8\", \"d9\", \"d10\", \"d11\", \"d12\", \"d13\", \"d14\", \"d15\", \"d16\", \"d17\", \"d18\", \"d19\", \"d20\", \"d21\", \"d22\", \"d23\", \"d24\", \"d25\", \"d26\", \"d27\", \"d28\", \"d29\", \"d30\", \"d31\");\n"
                );
            }
            InlineASM::None => unreachable!(),
        }

        in_asm_block
    }
}

/// An external function that can be called by [ExpressionEvaluatorWithExternalFunctions].
pub trait ExternalFunction<T>: Fn(&[T]) -> T + Send + Sync + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(<T> ExternalFunction<T>);
impl<T, F: Clone + Send + Sync + Fn(&[T]) -> T + Send + Sync> ExternalFunction<T> for F {}

/// An optimized evaluator for expressions that can evaluate expressions with parameters
/// and some registered external functions.
// TODO: deprecate
#[derive(Clone)]
pub struct ExpressionEvaluatorWithExternalFunctions<T> {
    eval: ExpressionEvaluator<T>,
    external_fns: Vec<(Vec<T>, String, Box<dyn ExternalFunction<T>>)>,
}

impl<T: Real> ExpressionEvaluatorWithExternalFunctions<T> {
    #[allow(dead_code)]
    pub(crate) fn update_stack(&mut self, e: ExpressionEvaluator<T>) {
        self.eval.stack = e.stack;
        self.eval.param_count = e.param_count;
        self.eval.instructions = e.instructions;
        self.eval.result_indices = e.result_indices;
    }

    pub fn get_evaluator(&self) -> &ExpressionEvaluator<T> {
        &self.eval
    }

    pub fn get_evaluator_mut(&mut self) -> &mut ExpressionEvaluator<T> {
        &mut self.eval
    }

    pub fn evaluate_single(&mut self, params: &[T]) -> T {
        if self.eval.result_indices.len() != 1 {
            panic!(
                "Evaluator does not return a single result but {} results",
                self.eval.result_indices.len()
            );
        }

        let mut res = T::new_zero();
        self.evaluate(params, std::slice::from_mut(&mut res));
        res
    }

    pub fn evaluate(&mut self, params: &[T], out: &mut [T]) {
        self.eval.evaluate_impl(params, &mut self.external_fns, out);
    }
}

/// A slot in a list that contains a numerical value.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Slot {
    /// An entry in the list of parameters.
    Param(usize),
    /// An entry in the list of constants.
    Const(usize),
    /// An entry in the list of temporary storage.
    Temp(usize),
    /// An entry in the list of results.
    Out(usize),
}

impl std::fmt::Display for Slot {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Slot::Param(i) => write!(f, "p{i}"),
            Slot::Const(i) => write!(f, "c{i}"),
            Slot::Temp(i) => write!(f, "t{i}"),
            Slot::Out(i) => write!(f, "o{i}"),
        }
    }
}

impl Slot {
    pub fn index(&self, index: usize) -> Slot {
        match self {
            Slot::Param(i) => Slot::Param(*i + index),
            Slot::Const(i) => Slot::Const(*i + index),
            Slot::Temp(i) => Slot::Temp(*i + index),
            Slot::Out(i) => Slot::Out(*i + index),
        }
    }
}

/// An evaluation instruction.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Clone)]
pub enum Instruction {
    /// `Add(o, [i0,...,i_n])` means `o = i0 + ... + i_n`, where the first
    /// `n_real` arguments are real.
    Add(Slot, Vec<Slot>, usize),
    /// `Mul(o, [i0,...,i_n], n_real)` means `o = i0 * ... * i_n`, where the first
    /// `n_real` arguments are real.
    Mul(Slot, Vec<Slot>, usize),
    /// `Pow(o, b, e, is_real)` means `o = b^e`. The `is_real` flag indicates
    /// whether the exponentiation is expected to yield a real number.
    Pow(Slot, Slot, i64, bool),
    /// `Powf(o, b, e, is_real)` means `o = b^e`. The `is_real` flag indicates
    /// whether the exponentiation is expected to yield a real number.
    Powf(Slot, Slot, Slot, bool),
    /// `Fun(o, s, a, is_real)` means `o = s(a)`, where `s` is assumed to
    /// be a built-in function such as `sin`. The `is_real` flag indicates
    /// whether the function is expected to yield a real number.
    Fun(Slot, BuiltinSymbol, Slot, bool),
    /// `ExternalFun(o, s, a,...)` means `o = s(a, ...)`, where `s` is an external function.
    ExternalFun(Slot, String, Vec<Slot>),
    /// `Assign(o, v)` means `o = v`.
    Assign(Slot, Slot),
    /// `IfElse(cond, label)` means jump to `label` if `cond` is zero.
    IfElse(Slot, usize),
    /// Unconditional jump to `label`.
    Goto(usize),
    /// A position in the instruction list to jump to.
    Label(usize),
    /// `Join(o, cond, t, f)` means `o = cond ? t : f`.
    Join(Slot, Slot, Slot, Slot),
}

impl std::fmt::Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Instruction::Add(o, a, _) => {
                write!(
                    f,
                    "{} = {}",
                    o,
                    a.iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("+")
                )
            }
            Instruction::Mul(o, a, _) => {
                write!(
                    f,
                    "{} = {}",
                    o,
                    a.iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join("*")
                )
            }
            Instruction::Pow(o, b, e, _) => {
                write!(f, "{o} = {b}^{e}")
            }
            Instruction::Powf(o, b, e, _) => {
                write!(f, "{o} = {b}^{e}")
            }
            Instruction::Fun(o, s, a, _) => {
                write!(f, "{} = {}({})", o, s.0, a)
            }
            Instruction::ExternalFun(o, s, a) => {
                write!(
                    f,
                    "{} = {}({})",
                    o,
                    s,
                    a.iter()
                        .map(|x| x.to_string())
                        .collect::<Vec<_>>()
                        .join(", ")
                )
            }
            Instruction::Assign(o, v) => {
                write!(f, "{} = {}", o, v)
            }
            Instruction::IfElse(cond, label) => {
                write!(f, "if {} == 0 goto L{}", cond, label)
            }
            Instruction::Goto(label) => {
                write!(f, "goto L{}", label)
            }
            Instruction::Label(label) => {
                write!(f, "L{}:", label)
            }
            Instruction::Join(o, cond, a, b) => {
                write!(f, "{} = {} ? {} : {}", o, cond, a, b)
            }
        }
    }
}

impl<T: Clone> ExpressionEvaluator<T> {
    /// Export the instructions, the size of the temporary storage, and the list of constants.
    /// This function can be used to create an evaluator in a different language.
    pub fn export_instructions(&self) -> (Vec<Instruction>, usize, Vec<T>) {
        let mut instr = vec![];
        let constants: Vec<_> = self.stack[self.param_count..self.reserved_indices].to_vec();

        macro_rules! get_slot {
            ($i:expr) => {
                if $i < self.param_count {
                    Slot::Param($i)
                } else if $i < self.reserved_indices {
                    Slot::Const($i - self.param_count)
                } else {
                    if self.result_indices.contains(&$i) {
                        Slot::Out(self.result_indices.iter().position(|x| *x == $i).unwrap())
                    } else {
                        Slot::Temp($i - self.reserved_indices)
                    }
                }
            };
        }

        for (i, sc) in &self.instructions {
            match i {
                Instr::Add(o, a) => {
                    let n_real_args = match sc {
                        ComplexPhase::Real => a.len(),
                        ComplexPhase::PartialReal(n) => *n,
                        _ => 0,
                    };

                    instr.push(Instruction::Add(
                        get_slot!(*o),
                        a.iter().map(|x| get_slot!(*x)).collect(),
                        n_real_args,
                    ));
                }
                Instr::Mul(o, a) => {
                    let n_real_args = match sc {
                        ComplexPhase::Real => a.len(),
                        ComplexPhase::PartialReal(n) => *n,
                        _ => 0,
                    };

                    instr.push(Instruction::Mul(
                        get_slot!(*o),
                        a.iter().map(|x| get_slot!(*x)).collect(),
                        n_real_args,
                    ));
                }
                Instr::Pow(o, b, e) => {
                    instr.push(Instruction::Pow(
                        get_slot!(*o),
                        get_slot!(*b),
                        *e,
                        *sc == ComplexPhase::Real,
                    ));
                }
                Instr::Powf(o, b, e) => {
                    instr.push(Instruction::Powf(
                        get_slot!(*o),
                        get_slot!(*b),
                        get_slot!(*e),
                        *sc == ComplexPhase::Real,
                    ));
                }
                Instr::BuiltinFun(o, s, a) => {
                    instr.push(Instruction::Fun(
                        get_slot!(*o),
                        *s,
                        get_slot!(*a),
                        *sc == ComplexPhase::Real,
                    ));
                }
                Instr::ExternalFun(o, f, a) => {
                    instr.push(Instruction::ExternalFun(
                        get_slot!(*o),
                        self.external_fns[*f].clone(),
                        a.iter().map(|x| get_slot!(*x)).collect(),
                    ));
                }
                Instr::IfElse(cond, label) => {
                    instr.push(Instruction::IfElse(get_slot!(*cond), label.0));
                }
                Instr::Goto(label) => {
                    instr.push(Instruction::Goto(label.0));
                }
                Instr::Label(label) => {
                    instr.push(Instruction::Label(label.0));
                }
                Instr::Join(o, cond, a, b) => {
                    instr.push(Instruction::Join(
                        get_slot!(*o),
                        get_slot!(*cond),
                        get_slot!(*a),
                        get_slot!(*b),
                    ));
                }
            }
        }

        for (out, i) in self.result_indices.iter().enumerate() {
            if get_slot!(*i) != Slot::Out(out) {
                instr.push(Instruction::Assign(Slot::Out(out), get_slot!(*i)));
            }
        }

        (instr, self.stack.len() - self.reserved_indices, constants)
    }
}

impl<T: Default + Clone> ExpressionEvaluator<T> {
    /// Redefine every operation to take `n` components in and
    /// yield `n` components. This can be used to define efficient
    /// evaluation over dual numbers.
    ///
    /// External functions must be mapped to `n` different functions
    /// that compute a single component each. The input to the functions
    /// is the flattened vector of all components of all parameters,
    /// followed by all previously computed output components.
    ///
    /// # Example
    ///
    /// Create a dual number and evaluate an expression over it:
    /// ```
    /// use ahash::HashMap;
    /// use symbolica::{
    ///     atom::{Atom, AtomCore},
    ///     create_hyperdual_single_derivative,
    ///     domains::{
    ///         float::{Complex, Float, FloatLike},
    ///         rational::Rational,
    ///     },
    ///     evaluate::{FunctionMap, OptimizationSettings, Dualizer},
    ///     parse,
    /// };
    ///
    /// create_hyperdual_single_derivative!(Dual2, 2);
    ///
    /// let ev = parse!("sin(x+y)^2+cos(x+y)^2 - 1")
    ///     .evaluator(
    ///         &FunctionMap::new(),
    ///         &[parse!("x"), parse!("y")],
    ///         OptimizationSettings::default(),
    ///    )
    ///    .unwrap();
    ///
    /// let dualizer = Dualizer::new(Dual2::<Complex<Rational>>::new_zero(), vec![]);
    /// let vec_ev = ev.vectorize(&dualizer, HashMap::default()).unwrap();
    ///
    /// let mut vec_f = vec_ev.map_coeff(&|x| x.re.to_f64());
    /// let mut dest = vec![0.; 3];
    /// vec_f.evaluate(&[2.0, 1.0, 0., 3.0, 1.0, 1.], &mut dest);
    ///
    /// assert!(dest.iter().all(|x| x.abs() < 1e-10));
    /// ```
    pub fn vectorize<V: Vectorize<T>>(
        mut self,
        v: &V,
        mut external_fn_map: HashMap<(String, usize), String>,
    ) -> Result<ExpressionEvaluator<T>, String> {
        let mut new_external_fns = vec![];
        let mut external_fn_index_map = HashMap::default();
        for external_fn in &self.external_fns {
            for i in 0..v.get_dimension() {
                if let Some(index) = external_fn_map.remove(&(external_fn.clone(), i)) {
                    new_external_fns.push(index);
                    external_fn_index_map
                        .insert((external_fn.clone(), i), new_external_fns.len() - 1);
                } else {
                    return Err(format!(
                        "No external function mapping found for function '{}' with index {}",
                        external_fn, i
                    ));
                }
            }
        }

        self.undo_stack_optimization();

        // unfold every instruction to a single operation
        let mut new_instr = vec![];
        for (x, c) in &mut self.instructions {
            match x {
                Instr::Add(o, a) => {
                    new_instr.push((Instr::Add(*o, vec![a[0], a[1]]), *c));
                    for x in a.iter().skip(2) {
                        new_instr.push((Instr::Add(*o, vec![*o, *x]), *c));
                    }
                }
                Instr::Mul(o, a) => {
                    new_instr.push((Instr::Mul(*o, vec![a[0], a[1]]), *c));
                    for x in a.iter().skip(2) {
                        new_instr.push((Instr::Mul(*o, vec![*o, *x]), *c));
                    }
                }
                _ => new_instr.push((x.clone(), *c)),
            }
        }

        self.instructions = new_instr;

        let mut constants = vec![];
        for c in &self.stack[self.param_count..self.reserved_indices] {
            constants.extend(v.map_coeff(c.clone()));
        }
        let old_constants_num = constants.len();

        let mut slot_map = HashMap::default();
        for x in 0..self.reserved_indices {
            slot_map.insert(x, x * v.get_dimension()); // set the start of the vector
        }

        self.param_count *= v.get_dimension();
        self.reserved_indices *= v.get_dimension();
        macro_rules! get_slot {
            ($i:expr) => {
                if $i < self.param_count {
                    Slot::Param($i)
                } else if $i < self.reserved_indices {
                    Slot::Const($i - self.param_count)
                } else {
                    Slot::Temp($i - self.reserved_indices)
                }
            };
        }

        macro_rules! from_slot {
            ($i:expr) => {
                match $i {
                    Slot::Param(x) => x,
                    Slot::Const(x) => x + self.param_count,
                    Slot::Temp(x) => x + self.reserved_indices,
                    Slot::Out(_) => unreachable!(),
                }
            };
        }

        let mut ins = InstructionList {
            instructions: vec![],
            constants,
            dim: v.get_dimension(),
        };

        for (i, _sc) in self.instructions.drain(..) {
            let (o, instr) = match i {
                Instr::Add(o, a) => (
                    o,
                    VectorInstruction::Add(get_slot!(slot_map[&a[0]]), get_slot!(slot_map[&a[1]])),
                ),
                Instr::Mul(o, a) => (
                    o,
                    VectorInstruction::Mul(get_slot!(slot_map[&a[0]]), get_slot!(slot_map[&a[1]])),
                ),
                Instr::Pow(o, a, e) => (o, VectorInstruction::Pow(get_slot!(slot_map[&a]), e)),
                Instr::Powf(o, b, e) => (
                    o,
                    VectorInstruction::Powf(get_slot!(slot_map[&b]), get_slot!(slot_map[&e])),
                ),
                Instr::BuiltinFun(o, f, a) => {
                    (o, VectorInstruction::BuiltinFun(f, get_slot!(slot_map[&a])))
                }
                Instr::ExternalFun(o, f, a) => {
                    let mut results = vec![];
                    for j in 0..v.get_dimension() {
                        let Some(index) =
                            external_fn_index_map.get(&(self.external_fns[f].clone(), j))
                        else {
                            return Err(format!(
                                "No external function mapping found for function '{}' with index {}",
                                self.external_fns[f], j
                            ));
                        };

                        // call with flattened arguments and pass all previously computed components
                        let r = ins.add(VectorInstruction::ExternalFun(
                            *index,
                            a.iter()
                                .map(|x| get_slot!(slot_map[&x]))
                                .map(|x| (0..v.get_dimension()).map(move |k| x.index(k)))
                                .flatten()
                                .chain(results.iter().cloned())
                                .collect(),
                        ));
                        results.push(r);
                    }

                    slot_map.insert(
                        o,
                        ins.instructions.len() + self.reserved_indices - v.get_dimension(),
                    );
                    continue;
                }
                Instr::Goto(l) => {
                    ins.instructions.push(VectorInstruction::Goto(l));
                    continue;
                }
                Instr::Label(l) => {
                    ins.instructions.push(VectorInstruction::Label(l));
                    continue;
                }
                Instr::IfElse(c, l) => {
                    ins.instructions
                        .push(VectorInstruction::IfElse(get_slot!(slot_map[&c]), l));
                    continue;
                }
                Instr::Join(o, c, t, f) => (
                    o,
                    VectorInstruction::Join(
                        get_slot!(slot_map[&c]),
                        get_slot!(slot_map[&t]),
                        get_slot!(slot_map[&f]),
                    ),
                ),
            };

            let r = v.map_instruction(&instr, &mut ins);
            assert_eq!(r.len(), v.get_dimension());
            for ii in r {
                ins.add(ii);
            }

            slot_map.insert(
                o,
                ins.instructions.len() + self.reserved_indices - v.get_dimension(),
            );
        }

        self.stack.clear();
        self.stack.resize(self.param_count, T::default());
        self.stack.extend(ins.constants);

        let stack_shift = self.stack.len() - old_constants_num - self.param_count;

        let mut new_result_indices = vec![];
        for x in 0..self.result_indices.len() {
            let mut p = slot_map[&self.result_indices[x]];
            if p >= self.reserved_indices {
                p += stack_shift;
            }

            for i in 0..v.get_dimension() {
                new_result_indices.push(p + i);
            }
        }

        self.reserved_indices += stack_shift;
        self.result_indices = new_result_indices;

        for i in ins.instructions {
            let out = self.instructions.len() + self.reserved_indices;
            match i {
                VectorInstruction::Add(slot, slot1) => {
                    let mut s1 = from_slot!(slot);
                    let mut s2 = from_slot!(slot1);
                    if s1 > s2 {
                        (s1, s2) = (s2, s1);
                    }

                    self.instructions
                        .push((Instr::Add(out, vec![s1, s2]), ComplexPhase::Any));
                }
                VectorInstruction::Assign(slot) => {
                    self.instructions
                        .push((Instr::Add(out, vec![from_slot!(slot)]), ComplexPhase::Any));
                }
                VectorInstruction::Mul(slot, slot1) => {
                    let mut s1 = from_slot!(slot);
                    let mut s2 = from_slot!(slot1);
                    if s1 > s2 {
                        (s1, s2) = (s2, s1);
                    }

                    self.instructions
                        .push((Instr::Mul(out, vec![s1, s2]), ComplexPhase::Any));
                }
                VectorInstruction::Pow(slot, e) => {
                    self.instructions
                        .push((Instr::Pow(out, from_slot!(slot), e), ComplexPhase::Any));
                }
                VectorInstruction::Powf(slot, slot1) => {
                    self.instructions.push((
                        Instr::Powf(out, from_slot!(slot), from_slot!(slot1)),
                        ComplexPhase::Any,
                    ));
                }
                VectorInstruction::BuiltinFun(builtin_symbol, slot) => {
                    self.instructions.push((
                        Instr::BuiltinFun(out, builtin_symbol, from_slot!(slot)),
                        ComplexPhase::Any,
                    ));
                }
                VectorInstruction::ExternalFun(f, args) => {
                    self.instructions.push((
                        Instr::ExternalFun(out, f, args.iter().map(|x| from_slot!(*x)).collect()),
                        ComplexPhase::Any,
                    ));
                }
                VectorInstruction::IfElse(cond, label) => {
                    self.instructions
                        .push((Instr::IfElse(from_slot!(cond), label), ComplexPhase::Any));
                }
                VectorInstruction::Goto(label) => {
                    self.instructions
                        .push((Instr::Goto(label), ComplexPhase::Any));
                }
                VectorInstruction::Label(label) => {
                    self.instructions
                        .push((Instr::Label(label), ComplexPhase::Any));
                }
                VectorInstruction::Join(cond, t, f) => {
                    self.instructions.push((
                        Instr::Join(out, from_slot!(cond), from_slot!(t), from_slot!(f)),
                        ComplexPhase::Any,
                    ));
                }
            }
        }

        self.stack.resize(
            self.reserved_indices + self.instructions.len(),
            T::default(),
        );
        self.external_fns = new_external_fns;

        self.remove_common_pairs();
        self.optimize_stack();
        self.fix_labels();

        Ok(self)
    }
}

/// A trait to define how to vectorize coefficients and instructions.
/// Every slot is mapped to `n` slots and every instruction is mapped to `n` instructions, where `n` is the dimension.
pub trait Vectorize<T> {
    /// Map a coefficient to a vector of coefficients of [Vectorize::get_dimension] length.
    fn map_coeff(&self, coeff: T) -> Vec<T>;

    /// Map an instruction applied to a vector of slots (components accessible with [Slot::index])
    /// to a vector of instructions of [Vectorize::get_dimension] length.
    fn map_instruction(
        &self,
        instr: &VectorInstruction,
        instr_addr: &mut InstructionList<T>,
    ) -> Vec<VectorInstruction>;

    /// Get the dimension of the vectorization.
    fn get_dimension(&self) -> usize;
}

/// A dualizer that maps coefficients and instructions to dual number components.
///
/// You can specify which components of the dual numbers are always zero
/// by providing a list of `(component index, dual index)` pairs in the constructor.
pub struct Dualizer<T: DualNumberStructure> {
    dual: T,
    zero_components: HashSet<(usize, usize)>, // component, index
}

impl<T: DualNumberStructure> Dualizer<T> {
    /// Create a new dualizer for the given dual number structure.
    /// You can specify which components are always zero
    /// by providing a list of `(component index, dual index)` pairs.
    pub fn new(dual: T, zero_components_per_parameter: Vec<(usize, usize)>) -> Self {
        Self {
            dual,
            zero_components: zero_components_per_parameter.into_iter().collect(),
        }
    }
}

impl<T: DualNumberStructure> Vectorize<Complex<Rational>> for Dualizer<T> {
    fn map_coeff(&self, coeff: Complex<Rational>) -> Vec<Complex<Rational>> {
        let mut r = vec![coeff.clone()];
        for _ in 1..self.dual.get_len() {
            r.push(Complex::new_zero());
        }
        r
    }

    fn map_instruction(
        &self,
        i: &VectorInstruction,
        instrs: &mut InstructionList<Complex<Rational>>,
    ) -> Vec<VectorInstruction> {
        fn is_zero<T: DualNumberStructure>(
            a: &Slot,
            dualizer: &Dualizer<T>,
            instrs: &InstructionList<Complex<Rational>>,
        ) -> bool {
            if instrs.is_zero(a) {
                return true;
            }

            if let Slot::Param(x) = a {
                dualizer
                    .zero_components
                    .contains(&(*x / dualizer.get_dimension(), *x % dualizer.get_dimension()))
            } else {
                false
            }
        }

        fn scalar_add<T: DualNumberStructure>(
            a: &Slot,
            b: &Slot,
            dualizer: &Dualizer<T>,
            instrs: &mut InstructionList<Complex<Rational>>,
        ) -> Slot {
            if is_zero(a, dualizer, instrs) {
                *b
            } else if instrs.is_zero(b) {
                *a
            } else {
                instrs.add(VectorInstruction::Add(*a, *b))
            }
        }

        fn scalar_yield_add<T: DualNumberStructure>(
            a: &Slot,
            b: &Slot,
            dualizer: &Dualizer<T>,
            instrs: &mut InstructionList<Complex<Rational>>,
        ) -> VectorInstruction {
            if is_zero(a, dualizer, instrs) {
                VectorInstruction::Assign(*b)
            } else if is_zero(b, dualizer, instrs) {
                VectorInstruction::Assign(*a)
            } else {
                VectorInstruction::Add(*a, *b)
            }
        }

        fn scalar_mul<T: DualNumberStructure>(
            a: &Slot,
            b: &Slot,
            dualizer: &Dualizer<T>,
            instrs: &mut InstructionList<Complex<Rational>>,
        ) -> Slot {
            if is_zero(a, dualizer, instrs) || instrs.is_one(b) {
                *a
            } else if is_zero(b, dualizer, instrs) || instrs.is_one(a) {
                *b
            } else {
                instrs.add(VectorInstruction::Mul(*a, *b))
            }
        }

        fn scalar_yield_mul<T: DualNumberStructure>(
            a: &Slot,
            b: &Slot,
            dualizer: &Dualizer<T>,
            instrs: &mut InstructionList<Complex<Rational>>,
        ) -> VectorInstruction {
            if is_zero(a, dualizer, instrs) || instrs.is_one(b) {
                VectorInstruction::Assign(*a)
            } else if is_zero(b, dualizer, instrs) || instrs.is_one(a) {
                VectorInstruction::Assign(*b)
            } else {
                VectorInstruction::Mul(*a, *b)
            }
        }

        fn rescale<T: DualNumberStructure>(
            a: &[Slot],
            c: &Slot,
            dualizer: &Dualizer<T>,
            instrs: &mut InstructionList<Complex<Rational>>,
        ) -> Vec<Slot> {
            a.iter()
                .map(|x| scalar_mul(x, c, dualizer, instrs))
                .collect()
        }

        fn add<T: DualNumberStructure>(
            a: &[Slot],
            b: &[Slot],
            dualizer: &Dualizer<T>,
            instrs: &mut InstructionList<Complex<Rational>>,
        ) -> Vec<Slot> {
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| scalar_add(x, y, dualizer, instrs))
                .collect()
        }

        fn mul<T: DualNumberStructure>(
            a: &[Slot],
            b: &[Slot],
            table: &[(usize, usize, usize)],
            dualizer: &Dualizer<T>,
            instrs: &mut InstructionList<Complex<Rational>>,
        ) -> Vec<Slot> {
            let mut current_index = vec![];
            for j in 0..a.len() {
                current_index.push(scalar_mul(&a[j], &b[0], dualizer, instrs));
            }

            for (si, oi, index) in table.iter() {
                let tmp = scalar_mul(&a[*si], &b[*oi], dualizer, instrs);
                current_index[*index] = scalar_add(&current_index[*index], &tmp, dualizer, instrs);
            }
            current_index
        }

        let mult_table = self.dual.get_multiplication_table();

        match i {
            VectorInstruction::Add(a, b) => (0..self.dual.get_len())
                .map(|j| scalar_yield_add(&a.index(j), &b.index(j), self, instrs))
                .collect(),
            VectorInstruction::Mul(a, b) => {
                let mut current_index = vec![];
                for j in 0..self.dual.get_len() {
                    current_index.push(scalar_mul(&a.index(j), b, self, instrs));
                }

                for (si, oi, index) in self.dual.get_multiplication_table().iter() {
                    let tmp = scalar_mul(&a.index(*si), &b.index(*oi), self, instrs);
                    current_index[*index] = scalar_add(&current_index[*index], &tmp, self, instrs);
                }

                current_index
                    .iter()
                    .map(|x| VectorInstruction::Assign(*x))
                    .collect()
            }
            VectorInstruction::BuiltinFun(f, a) => match f.get_symbol().get_id() {
                Symbol::SQRT_ID => {
                    let e = instrs.add(VectorInstruction::BuiltinFun(*f, *a));
                    let norm = instrs.add(VectorInstruction::Pow(*a, -1)); // TODO: check 0?

                    let zero = instrs.add_repeated_constant(Complex::new_zero());
                    let mut r = vec![zero];
                    r.extend(
                        (1..self.dual.get_len())
                            .map(|j| scalar_mul(&a.index(j), &norm, self, instrs)),
                    );

                    let one =
                        instrs.add_constant_in_first_component(Complex::from(Rational::one()));

                    let mut accum = (0..self.dual.get_len())
                        .map(|j| one.index(j))
                        .collect::<Vec<_>>();
                    let mut res = (0..self.dual.get_len())
                        .map(|j| one.index(j))
                        .collect::<Vec<_>>();
                    let mut num = Complex::from(Rational::one());

                    let mut scale = 1;
                    for p in 1..self.dual.get_max_depth() + 1 {
                        scale *= p;
                        num = num.clone()
                            * (num.from_usize(2).inv() - &num.from_usize(p as usize - 1));
                        accum = mul(&accum, &r, mult_table, self, instrs);

                        let c = instrs
                            .add_constant_in_first_component(&num * &num.from_usize(scale).inv());

                        res = add(&res, &rescale(&accum, &c, self, instrs), self, instrs);
                    }

                    res.iter()
                        .map(|x| scalar_yield_mul(x, &e, self, instrs))
                        .collect()
                }
                Symbol::EXP_ID => {
                    let e = instrs.add(VectorInstruction::BuiltinFun(*f, *a));

                    let one =
                        instrs.add_constant_in_first_component(Complex::from(Rational::one()));

                    let mut accum = (0..self.dual.get_len())
                        .map(|j| one.index(j))
                        .collect::<Vec<_>>();
                    let mut res = (0..self.dual.get_len())
                        .map(|j| one.index(j))
                        .collect::<Vec<_>>();

                    let zero = instrs.add_repeated_constant(Complex::new_zero());
                    let mut r = vec![zero];
                    r.extend((1..self.dual.get_len()).map(|j| a.index(j)));
                    let mut scale = Complex::from(Rational::one());
                    for p in 0..self.dual.get_max_depth() {
                        scale *= Rational::from(p + 1);
                        accum = mul(&accum, &r, mult_table, self, instrs);

                        let c = instrs.add_constant_in_first_component(scale.inv());

                        res = add(&res, &rescale(&accum, &c, self, instrs), self, instrs);
                    }

                    res.iter()
                        .map(|x| scalar_yield_mul(x, &e, self, instrs))
                        .collect()
                }
                Symbol::LOG_ID => {
                    let e = instrs.add(VectorInstruction::BuiltinFun(*f, *a));

                    let norm = instrs.add(VectorInstruction::Pow(*a, -1)); // TODO: check 0?

                    let zero = instrs.add_repeated_constant(Complex::new_zero());
                    let mut r = vec![zero];
                    r.extend(
                        (1..self.dual.get_len())
                            .map(|j| scalar_mul(&a.index(j), &norm, self, instrs)),
                    );

                    let mut accum = r.clone();

                    let mut res = (0..self.dual.get_len()).map(|_| zero).collect::<Vec<_>>();
                    res[0] = e;

                    let mut scale = Complex::from(Rational::from(-1));
                    for p in 1..self.dual.get_max_depth() + 1 {
                        scale *= Rational::from(-1);

                        let c = instrs
                            .add_constant_in_first_component((&scale * Rational::from(p)).inv());

                        res = add(&res, &rescale(&accum, &c, self, instrs), self, instrs);
                        accum = mul(&accum, &r, mult_table, self, instrs);
                    }

                    res.iter().map(|x| VectorInstruction::Assign(*x)).collect()
                }
                Symbol::SIN_ID => {
                    let s = instrs.add(VectorInstruction::BuiltinFun(*f, *a));
                    let c = instrs.add(VectorInstruction::BuiltinFun(
                        BuiltinSymbol(Symbol::COS),
                        *a,
                    ));

                    let zero = instrs.add_repeated_constant(Complex::new_zero());
                    let mut p = vec![zero];
                    p.extend((1..self.dual.get_len()).map(|j| a.index(j)));

                    let mut e = (0..self.dual.get_len()).map(|_| zero).collect::<Vec<_>>();
                    e[0] = s;

                    let mut sp = p.clone();
                    let mut scale = Complex::from(Rational::one());
                    for i in 1..self.dual.get_max_depth() + 1 {
                        scale *= Rational::from(i);
                        let b = if i % 2 == 1 { c.clone() } else { s.clone() };

                        let sc = instrs.add_constant_in_first_component(if i % 4 >= 2 {
                            -scale.inv()
                        } else {
                            scale.inv()
                        });

                        let s = rescale(&sp, &scalar_mul(&b, &sc, self, instrs), self, instrs);

                        sp = mul(&sp, &p, mult_table, self, instrs);

                        e = add(&e, &s, self, instrs);
                    }

                    e.iter().map(|x| VectorInstruction::Assign(*x)).collect()
                }
                Symbol::COS_ID => {
                    let s = instrs.add(VectorInstruction::BuiltinFun(
                        BuiltinSymbol(Symbol::SIN),
                        *a,
                    ));
                    let c = instrs.add(VectorInstruction::BuiltinFun(*f, *a));

                    let zero = instrs.add_repeated_constant(Complex::new_zero());
                    let mut p = vec![zero];
                    p.extend((1..self.dual.get_len()).map(|j| a.index(j)));

                    let mut e = (0..self.dual.get_len()).map(|_| zero).collect::<Vec<_>>();
                    e[0] = c;

                    let mut sp = p.clone();
                    let mut scale = Complex::from(Rational::one());
                    for i in 1..self.dual.get_max_depth() + 1 {
                        scale *= Rational::from(i);
                        let b = if i % 2 == 1 { s.clone() } else { c.clone() };

                        let sc =
                            instrs.add_constant_in_first_component(if (i % 2 == 0) ^ (i % 4 < 2) {
                                -scale.inv()
                            } else {
                                scale.inv()
                            });

                        let s = rescale(&sp, &scalar_mul(&b, &sc, self, instrs), self, instrs);

                        sp = mul(&sp, &p, mult_table, self, instrs);

                        e = add(&e, &s, self, instrs);
                    }

                    e.iter().map(|x| VectorInstruction::Assign(*x)).collect()
                }
                Symbol::ABS_ID => {
                    let n = instrs.add(VectorInstruction::BuiltinFun(
                        BuiltinSymbol(Symbol::ABS),
                        *a,
                    ));

                    let inv_val = instrs.add(VectorInstruction::Pow(*a, -1));
                    let scale = instrs.add(VectorInstruction::Mul(n, inv_val));

                    (0..self.dual.get_len())
                        .map(|j| scalar_yield_mul(&a.index(j), &scale, self, instrs))
                        .collect()
                }
                Symbol::CONJ_ID => {
                    // assume variables are real
                    (0..self.dual.get_len())
                        .map(|j| VectorInstruction::BuiltinFun(*f, a.index(j)))
                        .collect()
                }
                _ => unimplemented!(
                    "Vectorization not implemented for built-in function {}",
                    f.get_symbol().get_name()
                ),
            },
            VectorInstruction::Pow(a, b) => {
                assert_eq!(*b, -1); // only b = -1 is used in practice
                let a_inv = instrs.add(VectorInstruction::Pow(*a, -1));

                let zero = instrs.add_repeated_constant(Complex::new_zero());
                let mut r = vec![zero];
                r.extend(
                    (1..self.dual.get_len()).map(|j| scalar_mul(&a.index(j), &a_inv, self, instrs)),
                );

                let one = instrs.add_constant_in_first_component(Complex::from(Rational::one()));
                let neg_one =
                    instrs.add_constant_in_first_component(Complex::from(Rational::from(-1)));

                let mut accum = (0..self.dual.get_len())
                    .map(|j| one.index(j))
                    .collect::<Vec<_>>();
                let mut res = (0..self.dual.get_len())
                    .map(|j| one.index(j))
                    .collect::<Vec<_>>();

                for i in 1..self.dual.get_max_depth() + 1 {
                    accum = mul(&accum, &r, mult_table, self, instrs);
                    if i % 2 == 0 {
                        res = add(&res, &accum, self, instrs);
                    } else {
                        res = add(&res, &rescale(&accum, &neg_one, self, instrs), self, instrs);
                    }
                }

                res.iter()
                    .map(|x| scalar_yield_mul(x, &a_inv, self, instrs))
                    .collect()
            }
            VectorInstruction::Powf(b, e) => {
                let input = VectorInstruction::BuiltinFun(BuiltinSymbol(Symbol::LOG), *b);
                let log: Vec<_> = self
                    .map_instruction(&input, instrs)
                    .into_iter()
                    .map(|x| instrs.add(x))
                    .collect();
                let e = (0..self.dual.get_len())
                    .map(|j| e.index(j))
                    .collect::<Vec<_>>();
                let r = mul(&log, &e, mult_table, self, instrs);

                // exp needs adjacent slots
                let adjacent: Vec<_> = r
                    .into_iter()
                    .map(|x| instrs.add(VectorInstruction::Assign(x)))
                    .collect();
                let exp_in = VectorInstruction::BuiltinFun(BuiltinSymbol(Symbol::EXP), adjacent[0]);
                self.map_instruction(&exp_in, instrs)
            }
            VectorInstruction::Join(c, a, b) => (0..self.dual.get_len())
                .map(|j| VectorInstruction::Join(c.index(0), a.index(j), b.index(j)))
                .collect(),
            VectorInstruction::Assign(a) => (0..self.dual.get_len())
                .map(|j| VectorInstruction::Assign(a.index(j)))
                .collect(),
            VectorInstruction::ExternalFun(_, _)
            | VectorInstruction::Goto(_)
            | VectorInstruction::Label(_)
            | VectorInstruction::IfElse(..) => {
                unreachable!(
                    "Instruction {:?} should not appear inside vectorized instructions",
                    i
                )
            }
        }
    }

    fn get_dimension(&self) -> usize {
        self.dual.get_len()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VectorInstruction {
    Add(Slot, Slot),
    Assign(Slot),
    Mul(Slot, Slot),
    Pow(Slot, i64),
    Powf(Slot, Slot),
    BuiltinFun(BuiltinSymbol, Slot),
    ExternalFun(usize, Vec<Slot>),
    IfElse(Slot, Label),
    Goto(Label),
    Label(Label),
    Join(Slot, Slot, Slot),
}

pub struct InstructionList<T> {
    instructions: Vec<VectorInstruction>,
    constants: Vec<T>,
    dim: usize,
}

impl<T> InstructionList<T> {
    pub fn add(&mut self, instr: VectorInstruction) -> Slot {
        self.instructions.push(instr);
        Slot::Temp(self.instructions.len() - 1)
    }
}

impl<T: PartialEq + Clone + std::fmt::Debug> InstructionList<T> {
    pub fn add_constant(&mut self, value: Vec<T>) -> Slot {
        assert_eq!(value.len(), self.dim);
        if let Some(c) = self.constants.chunks(self.dim).position(|x| x == &value) {
            Slot::Const(c * self.dim)
        } else {
            self.constants.extend(value);
            Slot::Const(self.constants.len() - self.dim)
        }
    }

    pub fn add_repeated_constant(&mut self, value: T) -> Slot {
        if let Some(c) = self
            .constants
            .chunks(self.dim)
            .position(|x| x.iter().all(|x| *x == value))
        {
            Slot::Const(c * self.dim)
        } else {
            for _ in 0..self.dim {
                self.constants.push(value.clone());
            }
            Slot::Const(self.constants.len() - self.dim)
        }
    }
}

impl<T: SingleFloat> InstructionList<T> {
    pub fn is_zero(&self, slot: &Slot) -> bool {
        match slot {
            Slot::Const(c) => self.constants[*c].is_zero(),
            _ => false,
        }
    }

    pub fn is_one(&self, slot: &Slot) -> bool {
        match slot {
            Slot::Const(c) => self.constants[*c].is_one(),
            _ => false,
        }
    }

    pub fn add_constant_in_first_component(&mut self, value: T) -> Slot {
        let mut v = vec![value.clone()];
        v.extend((1..self.dim).map(|_| value.zero()));
        self.add_constant(v)
    }
}

/// A label in the instruction list.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Label(usize);

/// An evaluation instruction.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Clone, PartialEq)]
enum Instr {
    Add(usize, Vec<usize>),
    Mul(usize, Vec<usize>),
    Pow(usize, usize, i64),
    Powf(usize, usize, usize),
    BuiltinFun(usize, BuiltinSymbol, usize),
    ExternalFun(usize, usize, Vec<usize>),
    IfElse(usize, Label),
    Goto(Label),
    Label(Label),
    Join(usize, usize, usize, usize),
}

/// The phase of an operation in a complex evaluator.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Copy, Clone, PartialEq, Default, Hash)]
pub enum ComplexPhase {
    Real,
    Imag,
    PartialReal(usize),
    #[default]
    Any,
}

impl<T: Clone + PartialEq> SplitExpression<T> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> SplitExpression<T2> {
        SplitExpression {
            tree: self.tree.iter().map(|x| x.map_coeff(f)).collect(),
            subexpressions: self.subexpressions.iter().map(|x| x.map_coeff(f)).collect(),
        }
    }
}

impl<T: Clone + PartialEq> Expression<T> {
    /// Map the coefficients.
    ///
    /// Note that no rehashing is performed.
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> Expression<T2> {
        match self {
            Expression::Const(h, c) => Expression::Const(*h, Box::new(f(c))),
            Expression::Parameter(h, p) => Expression::Parameter(*h, *p),
            Expression::Eval(h, id, e_args) => {
                Expression::Eval(*h, *id, e_args.iter().map(|x| x.map_coeff(f)).collect())
            }
            Expression::Add(h, a) => {
                let new_args = a.iter().map(|x| x.map_coeff(f)).collect();
                Expression::Add(*h, new_args)
            }
            Expression::Mul(h, m) => {
                let new_args = m.iter().map(|x| x.map_coeff(f)).collect();
                Expression::Mul(*h, new_args)
            }
            Expression::Pow(h, p) => {
                let (b, e) = &**p;
                Expression::Pow(*h, Box::new((b.map_coeff(f), *e)))
            }
            Expression::Powf(h, p) => {
                let (b, e) = &**p;
                Expression::Powf(*h, Box::new((b.map_coeff(f), e.map_coeff(f))))
            }
            Expression::ReadArg(h, s) => Expression::ReadArg(*h, *s),
            Expression::BuiltinFun(h, s, a) => {
                Expression::BuiltinFun(*h, *s, Box::new(a.map_coeff(f)))
            }
            Expression::SubExpression(h, i) => Expression::SubExpression(*h, *i),
            Expression::ExternalFun(h, s, a) => {
                let new_args = a.iter().map(|x| x.map_coeff(f)).collect();
                Expression::ExternalFun(*h, *s, new_args)
            }
            Expression::IfElse(h, b) => {
                let (cond, then_expr, else_expr) = &**b;
                Expression::IfElse(
                    *h,
                    Box::new((
                        cond.map_coeff(f),
                        then_expr.map_coeff(f),
                        else_expr.map_coeff(f),
                    )),
                )
            }
        }
    }

    fn strip_constants(&mut self, stack: &mut Vec<T>, param_len: usize) {
        match self {
            Expression::Const(_, t) => {
                if let Some(p) = stack.iter().skip(param_len).position(|x| x == &**t) {
                    *self = Expression::Parameter(0, param_len + p);
                } else {
                    stack.push(t.as_ref().clone());
                    *self = Expression::Parameter(0, stack.len() - 1);
                }
            }
            Expression::Parameter(_, _) => {}
            Expression::Eval(_, _, e_args) => {
                for a in e_args {
                    a.strip_constants(stack, param_len);
                }
            }
            Expression::Add(_, a) | Expression::Mul(_, a) => {
                for arg in a {
                    arg.strip_constants(stack, param_len);
                }
            }
            Expression::Pow(_, p) => {
                p.0.strip_constants(stack, param_len);
            }
            Expression::Powf(_, p) => {
                p.0.strip_constants(stack, param_len);
                p.1.strip_constants(stack, param_len);
            }
            Expression::ReadArg(_, _) => {}
            Expression::BuiltinFun(_, _, a) => {
                a.strip_constants(stack, param_len);
            }
            Expression::SubExpression(_, _) => {}
            Expression::ExternalFun(_, _, a) => {
                for arg in a {
                    arg.strip_constants(stack, param_len);
                }
            }
            Expression::IfElse(_, b) => {
                b.0.strip_constants(stack, param_len);
                b.1.strip_constants(stack, param_len);
                b.2.strip_constants(stack, param_len);
            }
        }
    }
}

impl<T: Clone + PartialEq> EvalTree<T> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> EvalTree<T2> {
        EvalTree {
            expressions: SplitExpression {
                tree: self
                    .expressions
                    .tree
                    .iter()
                    .map(|x| x.map_coeff(f))
                    .collect(),
                subexpressions: self
                    .expressions
                    .subexpressions
                    .iter()
                    .map(|x| x.map_coeff(f))
                    .collect(),
            },
            functions: self
                .functions
                .iter()
                .map(|(s, a, e)| (s.clone(), a.clone(), e.map_coeff(f)))
                .collect(),
            external_functions: self.external_functions.clone(),
            param_count: self.param_count,
        }
    }
}

impl EvalTree<Complex<Rational>> {
    /// Create a linear version of the tree that can be evaluated more efficiently.
    pub fn linearize(
        &mut self,
        settings: &OptimizationSettings,
    ) -> ExpressionEvaluator<Complex<Rational>> {
        let mut stack = vec![Complex::<Rational>::default(); self.param_count];

        // strip every constant and move them into the stack after the params
        self.strip_constants(&mut stack); // FIXME
        let reserved_indices = stack.len();

        let mut sub_expr_pos = HashMap::default();
        let mut instructions = vec![];

        let mut result_indices = vec![];

        for t in &self.expressions.tree {
            let result_index = self.linearize_impl(
                t,
                &self.expressions.subexpressions,
                &mut stack,
                &mut instructions,
                &mut sub_expr_pos,
                &[],
                false,
                reserved_indices,
            );
            result_indices.push(result_index);
        }

        let mut e = ExpressionEvaluator {
            stack,
            param_count: self.param_count,
            reserved_indices,
            instructions,
            result_indices,
            external_fns: self.external_functions.clone(),
            settings: settings.clone(),
        };

        loop {
            let r = e.remove_common_instructions();

            if r == 0 || e.settings.abort_level > 0 {
                e.settings.abort_level = 0;
                break;
            }

            if settings.verbose {
                let (add_count, mul_count) = e.count_operations();
                info!(
                    "Removed {} common instructions: {} + and {} ×",
                    r, add_count, mul_count
                );
            }
        }

        for _ in 0..settings.cpe_iterations.unwrap_or(usize::MAX) {
            let r = e.remove_common_pairs();
            if r == 0 || e.settings.abort_level > 0 {
                e.settings.abort_level = 0;
                break;
            }

            if settings.verbose {
                let (add_count, mul_count) = e.count_operations();
                info!(
                    "Removed {} common pairs: {} + and {} ×",
                    r, add_count, mul_count
                );
            }
        }

        e.optimize_stack();
        e
    }

    fn strip_constants(&mut self, stack: &mut Vec<Complex<Rational>>) {
        for t in &mut self.expressions.tree {
            t.strip_constants(stack, self.param_count);
        }

        for e in &mut self.expressions.subexpressions {
            e.strip_constants(stack, self.param_count);
        }

        for (_, _, e) in &mut self.functions {
            for t in &mut e.tree {
                t.strip_constants(stack, self.param_count);
            }

            for e in &mut e.subexpressions {
                e.strip_constants(stack, self.param_count);
            }
        }
    }

    // Yields the stack index that contains the output.
    fn linearize_impl(
        &self,
        tree: &Expression<Complex<Rational>>,
        subexpressions: &[Expression<Complex<Rational>>],
        stack: &mut Vec<Complex<Rational>>,
        instr: &mut Vec<(Instr, ComplexPhase)>,
        sub_expr_pos: &mut HashMap<usize, usize>,
        args: &[usize],
        in_branch: bool,
        reserved_indices: usize,
    ) -> usize {
        match tree {
            Expression::Const(_, t) => {
                unreachable!(
                    "Constants should have been stripped from the expression tree. Found constant {}",
                    t
                );
            }
            Expression::Parameter(_, i) => *i,
            Expression::Eval(_, id, e_args) => {
                // inline the function
                let new_args: Vec<_> = e_args
                    .iter()
                    .map(|x| {
                        self.linearize_impl(
                            x,
                            subexpressions,
                            stack,
                            instr,
                            sub_expr_pos,
                            args,
                            in_branch,
                            reserved_indices,
                        )
                    })
                    .collect();

                let mut sub_expr_pos = HashMap::default();
                let func = &self.functions[*id as usize].2;
                self.linearize_impl(
                    &func.tree[0],
                    &func.subexpressions,
                    stack,
                    instr,
                    &mut sub_expr_pos,
                    &new_args,
                    in_branch,
                    reserved_indices,
                )
            }
            Expression::Add(_, a) => {
                let mut args: Vec<_> = a
                    .iter()
                    .map(|x| {
                        self.linearize_impl(
                            x,
                            subexpressions,
                            stack,
                            instr,
                            sub_expr_pos,
                            args,
                            in_branch,
                            reserved_indices,
                        )
                    })
                    .collect();
                args.sort();

                stack.push(Complex::default());
                let res = stack.len() - 1;

                let add = Instr::Add(res, args);
                instr.push((add, ComplexPhase::Any));

                res
            }
            Expression::Mul(_, m) => {
                let mut args: Vec<_> = m
                    .iter()
                    .map(|x| {
                        self.linearize_impl(
                            x,
                            subexpressions,
                            stack,
                            instr,
                            sub_expr_pos,
                            args,
                            in_branch,
                            reserved_indices,
                        )
                    })
                    .collect();
                args.sort();

                stack.push(Complex::default());
                let res = stack.len() - 1;

                let mul = Instr::Mul(res, args);
                instr.push((mul, ComplexPhase::Any));

                res
            }
            Expression::Pow(_, p) => {
                let b = self.linearize_impl(
                    &p.0,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    in_branch,
                    reserved_indices,
                );
                stack.push(Complex::default());
                let mut res = stack.len() - 1;

                if p.1 > 1 {
                    instr.push((Instr::Mul(res, vec![b; p.1 as usize]), ComplexPhase::Any));
                } else if p.1 < -1 {
                    instr.push((Instr::Mul(res, vec![b; -p.1 as usize]), ComplexPhase::Any));
                    stack.push(Complex::default());
                    res += 1;
                    instr.push((Instr::Pow(res, res - 1, -1), ComplexPhase::Any));
                } else {
                    instr.push((Instr::Pow(res, b, p.1), ComplexPhase::Any));
                }
                res
            }
            Expression::Powf(_, p) => {
                let b = self.linearize_impl(
                    &p.0,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    in_branch,
                    reserved_indices,
                );
                let e = self.linearize_impl(
                    &p.1,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    in_branch,
                    reserved_indices,
                );
                stack.push(Complex::default());
                let res = stack.len() - 1;

                instr.push((Instr::Powf(res, b, e), ComplexPhase::Any));
                res
            }
            Expression::ReadArg(_, a) => args[*a],
            Expression::BuiltinFun(_, s, v) => {
                let arg = self.linearize_impl(
                    v,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    in_branch,
                    reserved_indices,
                );
                stack.push(Complex::default());
                let c = Instr::BuiltinFun(stack.len() - 1, *s, arg);
                instr.push((c, ComplexPhase::Any));
                stack.len() - 1
            }
            Expression::SubExpression(_, id) => {
                if sub_expr_pos.contains_key(id) {
                    *sub_expr_pos.get(id).unwrap()
                } else {
                    let res = self.linearize_impl(
                        &subexpressions[*id],
                        subexpressions,
                        stack,
                        instr,
                        sub_expr_pos,
                        args,
                        in_branch,
                        reserved_indices,
                    );

                    // only register the subexpression as computed when it is not
                    // computed in a branch, as the sub expression may not be computed
                    // in the other branch
                    if !in_branch {
                        sub_expr_pos.insert(*id, res);
                    }

                    res
                }
            }
            Expression::ExternalFun(_, s, v) => {
                let args: Vec<_> = v
                    .iter()
                    .map(|x| {
                        self.linearize_impl(
                            x,
                            subexpressions,
                            stack,
                            instr,
                            sub_expr_pos,
                            args,
                            in_branch,
                            reserved_indices,
                        )
                    })
                    .collect();

                stack.push(Complex::default());
                let res = stack.len() - 1;

                let f = Instr::ExternalFun(res, *s as usize, args);
                instr.push((f, ComplexPhase::Any));

                res
            }
            Expression::IfElse(_, b) => {
                let instr_len = instr.len();
                let stack_len = stack.len();
                let subexpression_len = sub_expr_pos.len();
                let cond = self.linearize_impl(
                    &b.0,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    in_branch,
                    reserved_indices,
                );

                // try to resolve the condition if it is fully numeric
                fn resolve(
                    instr: &[(Instr, ComplexPhase)],
                    stack: &[Complex<Rational>],
                    cond: usize,
                    param_count: usize,
                    reserved_indices: usize,
                ) -> Option<Complex<Rational>> {
                    if cond < param_count {
                        return None;
                    }
                    if cond < reserved_indices {
                        return Some(stack[cond].clone());
                    }

                    match &instr[cond - reserved_indices].0 {
                        Instr::Add(_, args) => {
                            let mut res = Complex::default();
                            for x in args {
                                match resolve(instr, stack, *x, param_count, reserved_indices) {
                                    Some(v) => res += v,
                                    None => return None,
                                }
                            }

                            Some(res)
                        }
                        Instr::Mul(_, args) => {
                            let mut res = Complex::new(Rational::one(), Rational::zero());
                            for x in args {
                                match resolve(instr, stack, *x, param_count, reserved_indices) {
                                    Some(v) => res *= v,
                                    None => return None,
                                }
                            }

                            Some(res)
                        }
                        Instr::Pow(_, base, exp) => {
                            if let Some(base_val) =
                                resolve(instr, stack, *base, param_count, reserved_indices)
                            {
                                if *exp < 0 {
                                    Some(base_val.pow(exp.unsigned_abs()).inv())
                                } else {
                                    Some(base_val.pow(exp.unsigned_abs()))
                                }
                            } else {
                                None
                            }
                        }
                        _ => None,
                    }
                }

                if let Some(cond_res) =
                    resolve(instr, stack, cond, self.param_count, reserved_indices)
                {
                    // remove dead code
                    instr.truncate(instr_len);
                    stack.truncate(stack_len);
                    if subexpression_len != sub_expr_pos.len() {
                        // remove subexpressions that are created as part of the conditions
                        sub_expr_pos.retain(|_, &mut v| v < reserved_indices + instr_len);
                    }

                    return if !cond_res.is_zero() {
                        self.linearize_impl(
                            &b.1,
                            subexpressions,
                            stack,
                            instr,
                            sub_expr_pos,
                            args,
                            in_branch,
                            reserved_indices,
                        )
                    } else {
                        self.linearize_impl(
                            &b.2,
                            subexpressions,
                            stack,
                            instr,
                            sub_expr_pos,
                            args,
                            in_branch,
                            reserved_indices,
                        )
                    };
                }

                let label_else = Label(instr.len());
                stack.push(Complex::default());
                instr.push((Instr::IfElse(cond, label_else), ComplexPhase::Any));

                let then_branch = self.linearize_impl(
                    &b.1,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    true,
                    reserved_indices,
                );

                let label_end = Label(instr.len());
                stack.push(Complex::default());
                instr.push((Instr::Goto(label_end), ComplexPhase::Any));
                stack.push(Complex::default());
                instr.push((Instr::Label(label_else), ComplexPhase::Any));

                let else_branch = self.linearize_impl(
                    &b.2,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    true,
                    reserved_indices,
                );

                stack.push(Complex::default());
                instr.push((Instr::Label(label_end), ComplexPhase::Any));

                stack.push(Complex::default());
                let res = stack.len() - 1;

                instr.push((
                    Instr::Join(res, cond, then_branch, else_branch),
                    ComplexPhase::Any,
                ));

                res
            }
        }
    }

    /// Find a near-optimal Horner scheme that minimizes the number of multiplications
    /// and additions, using `iterations` iterations of the optimization algorithm
    /// and `n_cores` cores. Optionally, a starting scheme can be provided.
    pub fn optimize(
        mut self,
        settings: &OptimizationSettings,
    ) -> ExpressionEvaluator<Complex<Rational>> {
        if settings.verbose {
            let (n_add, n_mul) = self.count_operations();
            info!(
                "Initial ops: {} additions and {} multiplications",
                n_add, n_mul
            );
        }

        if settings.horner_iterations > 0 {
            let _ = self.optimize_horner_scheme(settings);
        }

        self.common_subexpression_elimination();
        self.linearize(settings)
    }

    /// Write the expressions in a Horner scheme where the variables
    /// are sorted by their occurrence count.
    pub fn horner_scheme(&mut self) {
        for t in &mut self.expressions.tree {
            t.occurrence_order_horner_scheme();
        }

        for e in &mut self.expressions.subexpressions {
            e.occurrence_order_horner_scheme();
        }

        for (_, _, e) in &mut self.functions {
            for t in &mut e.tree {
                t.occurrence_order_horner_scheme();
            }

            for e in &mut e.subexpressions {
                e.occurrence_order_horner_scheme();
            }
        }
    }

    /// Find a near-optimal Horner scheme that minimizes the number of multiplications
    /// and additions, using `iterations` iterations of the optimization algorithm
    /// and `n_cores` cores. Optionally, a starting scheme can be provided.
    pub fn optimize_horner_scheme(
        &mut self,
        settings: &OptimizationSettings,
    ) -> Vec<Expression<Complex<Rational>>> {
        let v = match &settings.hot_start {
            Some(a) => a.clone(),
            None => {
                let mut v = HashMap::default();

                for t in &mut self.expressions.tree {
                    t.find_all_variables(&mut v);
                }

                for e in &mut self.expressions.subexpressions {
                    e.find_all_variables(&mut v);
                }

                let mut v: Vec<_> = v.into_iter().collect();
                v.retain(|(_, vv)| *vv > 1);
                v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                v.truncate(settings.max_horner_scheme_variables);
                v.into_iter().map(|(k, _)| k).collect::<Vec<_>>()
            }
        };

        let scheme =
            Expression::optimize_horner_scheme_multiple(&self.expressions.tree, &v, settings);
        for e in &mut self.expressions.tree {
            e.apply_horner_scheme(&scheme);
        }

        for e in &mut self.expressions.subexpressions {
            e.apply_horner_scheme(&scheme);
        }

        for (name, _, e) in &mut self.functions {
            let mut v = HashMap::default();

            for t in &mut e.tree {
                t.find_all_variables(&mut v);
            }

            for e in &mut e.subexpressions {
                e.find_all_variables(&mut v);
            }

            let mut v: Vec<_> = v.into_iter().collect();
            v.retain(|(_, vv)| *vv > 1);
            v.sort_by_key(|k| Reverse(k.1));
            v.truncate(settings.max_horner_scheme_variables);
            let v = v.into_iter().map(|(k, _)| k).collect::<Vec<_>>();

            if settings.verbose {
                info!(
                    "Optimizing Horner scheme for function {} with {} variables",
                    name,
                    v.len()
                );
            }

            let scheme = Expression::optimize_horner_scheme_multiple(&e.tree, &v, settings);

            for t in &mut e.tree {
                t.apply_horner_scheme(&scheme);
            }

            for e in &mut e.subexpressions {
                e.apply_horner_scheme(&scheme);
            }
        }

        scheme
    }
}

impl Expression<Complex<Rational>> {
    pub fn apply_horner_scheme(&mut self, scheme: &[Expression<Complex<Rational>>]) {
        if scheme.is_empty() {
            return;
        }

        let a = match self {
            Expression::Add(_, a) => a,
            Expression::Eval(_, _, a) => {
                for arg in a {
                    arg.apply_horner_scheme(scheme);
                }
                return;
            }
            Expression::Mul(_, m) => {
                for a in m {
                    a.apply_horner_scheme(scheme);
                }
                return;
            }
            Expression::Pow(_, b) => {
                b.0.apply_horner_scheme(scheme);
                return;
            }
            Expression::Powf(_, b) => {
                b.0.apply_horner_scheme(scheme);
                b.1.apply_horner_scheme(scheme);
                return;
            }
            Expression::BuiltinFun(_, _, b) => {
                b.apply_horner_scheme(scheme);
                return;
            }
            _ => {
                return;
            }
        };

        a.sort();

        let mut max_pow: Option<i64> = None;
        for x in &*a {
            if let Expression::Mul(_, m) = x {
                let mut pow_counter = 0;
                for y in m {
                    if let Expression::Pow(_, p) = y {
                        if p.0 == scheme[0] && p.1 > 0 {
                            pow_counter += p.1;
                        }
                    } else if y == &scheme[0] {
                        pow_counter += 1; // support x*x*x^3 in term
                    }
                }

                if pow_counter > 0 && (max_pow.is_none() || pow_counter < max_pow.unwrap()) {
                    max_pow = Some(pow_counter);
                }
            } else if let Expression::Pow(_, p) = x {
                if p.0 == scheme[0] && p.1 > 0 && (max_pow.is_none() || p.1 < max_pow.unwrap()) {
                    max_pow = Some(p.1);
                }
            } else if x == &scheme[0] {
                max_pow = Some(1);
            }
        }

        // TODO: jump to next variable if the current variable only appears in one factor?
        // this will improve the scheme but may hide common subexpressions?

        let Some(max_pow) = max_pow else {
            return self.apply_horner_scheme(&scheme[1..]);
        };

        // extract GCD and phase of integer coefficients
        // keep rational coefficients untouched to avoid numerical precision issues
        let mut gcd = Complex::new(Rational::zero(), Rational::zero());
        for x in &*a {
            let mut num = None;

            if let Expression::Mul(_, m) = x {
                for y in m {
                    if let Expression::Const(_, c) = y
                        && c.re.is_integer()
                        && c.im.is_integer()
                    {
                        num = Some(c);
                    }
                }
            } else if let Expression::Const(_, c) = x
                && c.re.is_integer()
                && c.im.is_integer()
            {
                num = Some(c);
            }

            if let Some(n) = num {
                if n.im.is_zero() && gcd.im.is_zero() {
                    gcd = Complex::new(gcd.re.gcd(&n.re), Rational::zero());
                } else if n.re.is_zero() && gcd.re.is_zero() {
                    gcd = Complex::new(Rational::zero(), gcd.im.gcd(&n.im));
                } else {
                    gcd = Complex::new_one();
                }
            } else {
                gcd = Complex::new_one();
            }
        }

        if !gcd.is_zero() && !gcd.is_one() {
            for x in &mut *a {
                match x {
                    Expression::Mul(_, m) => {
                        for y in m {
                            if let Expression::Const(_, c) = y {
                                **c /= &gcd;
                                break;
                            }
                        }
                    }
                    Expression::Const(_, c) => {
                        **c /= &gcd;
                    }
                    _ => {
                        unreachable!()
                    }
                }
            }
        }

        let mut contains = vec![];
        let mut rest = vec![];

        for mut x in a.drain(..) {
            let mut found = false;
            if let Expression::Mul(_, m) = &mut x {
                let mut pow_counter = 0;

                m.retain(|y| {
                    if let Expression::Pow(_, p) = y {
                        if p.0 == scheme[0] && p.1 > 0 {
                            pow_counter += p.1;
                            false
                        } else {
                            true
                        }
                    } else if y == &scheme[0] {
                        pow_counter += 1;
                        false
                    } else {
                        true
                    }
                });

                if pow_counter > max_pow {
                    if pow_counter > max_pow + 1 {
                        m.push(
                            Expression::Pow(
                                0,
                                Box::new((scheme[0].clone(), pow_counter - max_pow)),
                            )
                            .rehashed(true),
                        );
                    } else {
                        m.push(scheme[0].clone());
                    }

                    m.sort();
                }

                if m.is_empty() {
                    x = Expression::Const(0, Box::new(Complex::new_one())).rehashed(true);
                } else if m.len() == 1 {
                    x = m.pop().unwrap();
                }

                found = pow_counter > 0;
            } else if let Expression::Pow(_, p) = &mut x {
                if p.0 == scheme[0] && p.1 > 0 {
                    if p.1 > max_pow + 1 {
                        p.1 -= max_pow;
                    } else if p.1 - max_pow == 1 {
                        x = scheme[0].clone();
                    } else {
                        x = Expression::Const(0, Box::new(Complex::new_one())).rehashed(true);
                    }
                    found = true;
                }
            } else if x == scheme[0] {
                found = true;
                x = Expression::Const(0, Box::new(Complex::new_one())).rehashed(true);
            }

            if found {
                contains.push(x);
            } else {
                rest.push(x);
            }
        }

        let extracted = if max_pow == 1 {
            scheme[0].clone()
        } else {
            Expression::Pow(0, Box::new((scheme[0].clone(), max_pow))).rehashed(true)
        };

        let mut contains = if contains.len() == 1 {
            contains.pop().unwrap()
        } else {
            Expression::Add(0, contains).rehashed(true)
        };

        contains.apply_horner_scheme(scheme); // keep trying with same variable

        let mut v = vec![];
        if let Expression::Mul(_, a) = contains {
            v.extend(a);
        } else {
            v.push(contains);
        }

        v.push(extracted);
        v.retain(|x| {
            if let Expression::Const(_, y) = x
                && y.is_one()
            {
                false
            } else {
                true
            }
        });
        v.sort();

        let mut c = if v.len() == 1 {
            v.pop().unwrap()
        } else {
            Expression::Mul(0, v).rehashed(true)
        };

        if rest.is_empty() {
            if !gcd.is_zero() && !gcd.is_one() {
                if let Expression::Mul(_, v) = &mut c {
                    v.push(Expression::Const(0, Box::new(gcd)));
                    v.sort();
                    *self = c.rehashed(true);
                } else {
                    *self = Expression::Mul(0, vec![Expression::Const(0, Box::new(gcd)), c])
                        .rehashed(true);
                }
            } else {
                *self = c.rehashed(true);
            }
        } else {
            let mut r = if rest.len() == 1 {
                rest.pop().unwrap()
            } else {
                Expression::Add(0, rest).rehashed(true)
            };

            r.apply_horner_scheme(&scheme[1..]);

            a.clear();
            a.push(c);

            if let Expression::Add(_, aa) = r {
                a.extend(aa);
            } else {
                a.push(r);
            }

            a.sort();

            if !gcd.is_zero() && !gcd.is_one() {
                *self = Expression::Mul(
                    0,
                    vec![
                        Expression::Const(0, Box::new(gcd)),
                        Expression::Add(0, std::mem::take(a)),
                    ],
                )
                .rehashed(true);
            }
        }
    }

    /// Apply a simple occurrence-order Horner scheme to every addition.
    pub fn occurrence_order_horner_scheme(&mut self) {
        match self {
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _) => {}
            Expression::Eval(_, _, ae) => {
                for arg in ae {
                    arg.occurrence_order_horner_scheme();
                }
            }
            Expression::Add(_, a) => {
                for arg in &mut *a {
                    arg.occurrence_order_horner_scheme();
                }

                let mut occurrence = HashMap::default();

                for arg in &*a {
                    match arg {
                        Expression::Mul(_, m) => {
                            for aa in m {
                                if let Expression::Pow(_, p) = aa
                                    && !matches!(p.0, Expression::Const(_, _))
                                {
                                    occurrence
                                        .entry(p.0.clone())
                                        .and_modify(|x| *x += 1)
                                        .or_insert(1);
                                } else if !matches!(aa, Expression::Const(_, _)) {
                                    occurrence
                                        .entry(aa.clone())
                                        .and_modify(|x| *x += 1)
                                        .or_insert(1);
                                }
                            }
                        }
                        x => {
                            if let Expression::Pow(_, p) = x
                                && !matches!(p.0, Expression::Const(_, _))
                            {
                                occurrence
                                    .entry(p.0.clone())
                                    .and_modify(|x| *x += 1)
                                    .or_insert(1);
                            } else if !matches!(x, Expression::Const(_, _)) {
                                occurrence
                                    .entry(x.clone())
                                    .and_modify(|x| *x += 1)
                                    .or_insert(1);
                            }
                        }
                    }
                }

                occurrence.retain(|_, v| *v > 1);
                let mut order: Vec<_> = occurrence.into_iter().collect();
                order.sort_by_key(|k| Reverse(k.1)); // occurrence order
                let scheme = order.into_iter().map(|(k, _)| k).collect::<Vec<_>>();

                self.apply_horner_scheme(&scheme);
            }
            Expression::Mul(_, a) => {
                for arg in a {
                    arg.occurrence_order_horner_scheme();
                }
            }
            Expression::Pow(_, p) => {
                p.0.occurrence_order_horner_scheme();
            }
            Expression::Powf(_, p) => {
                p.0.occurrence_order_horner_scheme();
                p.1.occurrence_order_horner_scheme();
            }
            Expression::BuiltinFun(_, _, a) => {
                a.occurrence_order_horner_scheme();
            }
            Expression::SubExpression(_, _) => {}
            Expression::ExternalFun(_, _, a) => {
                for arg in a {
                    arg.occurrence_order_horner_scheme();
                }
            }
            Expression::IfElse(_, b) => {
                b.0.occurrence_order_horner_scheme();
                b.1.occurrence_order_horner_scheme();
                b.2.occurrence_order_horner_scheme();
            }
        }
    }

    pub fn optimize_horner_scheme(
        &self,
        vars: &[Self],
        settings: &OptimizationSettings,
    ) -> Vec<Self> {
        Self::optimize_horner_scheme_multiple(std::slice::from_ref(self), vars, settings)
    }

    pub fn optimize_horner_scheme_multiple(
        expressions: &[Self],
        vars: &[Self],
        settings: &OptimizationSettings,
    ) -> Vec<Self> {
        if vars.is_empty() {
            return vars.to_vec();
        }

        let horner: Vec<_> = expressions
            .iter()
            .map(|x| {
                let mut h = x.clone();
                h.apply_horner_scheme(vars);
                h.rehashed(true)
            })
            .collect();
        let mut subexpr = HashMap::default();
        let mut best_ops = (0, 0);
        for h in &horner {
            let ops = h.count_operations_with_subexpression(&mut subexpr);
            best_ops = (best_ops.0 + ops.0, best_ops.1 + ops.1);
        }

        if settings.verbose {
            info!(
                "Initial Horner scheme ops: {} additions and {} multiplications",
                best_ops.0, best_ops.1
            );
        }

        let best_mul = Arc::new(AtomicUsize::new(best_ops.1));
        let best_add = Arc::new(AtomicUsize::new(best_ops.0));
        let best_scheme = Arc::new(Mutex::new(vars.to_vec()));

        let n_iterations = settings.horner_iterations.max(1) - 1;

        let permutations = if vars.len() < 10
            && Integer::factorial(vars.len() as u32) <= settings.horner_iterations.max(1)
        {
            let v: Vec<_> = (0..vars.len()).collect();
            Some(unique_permutations(&v).1)
        } else {
            None
        };
        let p_ref = &permutations;

        let n_cores = if LicenseManager::is_licensed() {
            settings.n_cores
        } else {
            1
        }
        .min(n_iterations);

        std::thread::scope(|s| {
            let abort = Arc::new(AtomicBool::new(false));

            for i in 0..n_cores {
                let mut rng = MonteCarloRng::new(0, i);

                let mut cvars = vars.to_vec();
                let best_scheme = best_scheme.clone();
                let best_mul = best_mul.clone();
                let best_add = best_add.clone();
                let mut last_mul = usize::MAX;
                let mut last_add = usize::MAX;
                let abort = abort.clone();

                let mut op = move || {
                    for j in 0..n_iterations / n_cores {
                        if abort.load(Ordering::Relaxed) {
                            return;
                        }

                        if i == n_cores - 1
                            && let Some(a) = &settings.abort_check
                            && a()
                        {
                            abort.store(true, Ordering::Relaxed);

                            if settings.verbose {
                                info!(
                                    "Aborting Horner optimization at step {}/{}.",
                                    j,
                                    settings.horner_iterations / n_cores
                                );
                            }

                            return;
                        }

                        // try a random swap
                        let mut t1 = 0;
                        let mut t2 = 0;

                        if let Some(p) = p_ref {
                            if j >= p.len() / n_cores {
                                break;
                            }

                            let perm = &p[i * (p.len() / n_cores) + j];
                            cvars = perm.iter().map(|x| vars[*x].clone()).collect();
                        } else {
                            t1 = rng.random_range(0..cvars.len());
                            t2 = rng.random_range(0..cvars.len() - 1);

                            cvars.swap(t1, t2);
                        }

                        let horner: Vec<_> = expressions
                            .iter()
                            .map(|x| {
                                let mut h = x.clone();
                                h.apply_horner_scheme(&cvars);
                                h.rehash(true);
                                h
                            })
                            .collect();
                        let mut subexpr = HashMap::default();
                        let mut cur_ops = (0, 0);

                        for h in &horner {
                            let ops = h.count_operations_with_subexpression(&mut subexpr);
                            cur_ops = (cur_ops.0 + ops.0, cur_ops.1 + ops.1);
                        }

                        // prefer fewer multiplications
                        if cur_ops.1 <= last_mul || cur_ops.1 == last_mul && cur_ops.0 <= last_add {
                            if settings.verbose {
                                info!(
                                    "Accept move at step {}/{}: {} + and {} ×",
                                    j,
                                    settings.horner_iterations / n_cores,
                                    cur_ops.0,
                                    cur_ops.1
                                );
                            }

                            last_add = cur_ops.0;
                            last_mul = cur_ops.1;

                            if cur_ops.1 <= best_mul.load(Ordering::Relaxed)
                                || cur_ops.1 == best_mul.load(Ordering::Relaxed)
                                    && cur_ops.0 <= best_add.load(Ordering::Relaxed)
                            {
                                let mut best_scheme = best_scheme.lock().unwrap();

                                // check again if it is the best now that we have locked
                                let best_mul_l = best_mul.load(Ordering::Relaxed);
                                let best_add_l = best_add.load(Ordering::Relaxed);
                                if cur_ops.1 <= best_mul_l
                                    || cur_ops.1 == best_mul_l && cur_ops.0 <= best_add_l
                                {
                                    if cur_ops.0 == best_add_l && cur_ops.1 == best_mul_l {
                                        if *best_scheme < cvars {
                                            // on a draw, accept the lexicographical minimum
                                            // to get a deterministic scheme
                                            *best_scheme = cvars.clone();
                                        }
                                    } else {
                                        best_mul.store(cur_ops.1, Ordering::Relaxed);
                                        best_add.store(cur_ops.0, Ordering::Relaxed);
                                        *best_scheme = cvars.clone();
                                    }
                                }
                            }
                        } else {
                            cvars.swap(t1, t2);
                        }
                    }
                };

                if i + 1 < n_cores {
                    s.spawn(op);
                } else {
                    // execute in the main thread and do the abort check on the main thread
                    // this helps with catching ctrl-c
                    op()
                }
            }
        });

        if settings.verbose {
            info!(
                "Final scheme: {} + and {} ×",
                best_add.load(Ordering::Relaxed),
                best_mul.load(Ordering::Relaxed)
            );
        }

        Arc::try_unwrap(best_scheme).unwrap().into_inner().unwrap()
    }

    fn find_all_variables(&self, vars: &mut HashMap<Expression<Complex<Rational>>, usize>) {
        match self {
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _) => {}
            Expression::Eval(_, _, ae) => {
                for arg in ae {
                    arg.find_all_variables(vars);
                }
            }
            Expression::Add(_, a) => {
                for arg in a {
                    arg.find_all_variables(vars);
                }

                for arg in a {
                    match arg {
                        Expression::Mul(_, m) => {
                            for aa in m {
                                if let Expression::Pow(_, p) = aa
                                    && !matches!(p.0, Expression::Const(_, _))
                                {
                                    vars.entry(p.0.clone()).and_modify(|x| *x += 1).or_insert(1);
                                } else if !matches!(aa, Expression::Const(_, _)) {
                                    vars.entry(aa.clone()).and_modify(|x| *x += 1).or_insert(1);
                                }
                            }
                        }
                        x => {
                            if let Expression::Pow(_, p) = x
                                && !matches!(p.0, Expression::Const(_, _))
                            {
                                vars.entry(p.0.clone()).and_modify(|x| *x += 1).or_insert(1);
                            } else if !matches!(x, Expression::Const(_, _)) {
                                vars.entry(x.clone()).and_modify(|x| *x += 1).or_insert(1);
                            }
                        }
                    }
                }
            }
            Expression::Mul(_, a) => {
                for arg in a {
                    arg.find_all_variables(vars);
                }
            }
            Expression::Pow(_, p) => {
                p.0.find_all_variables(vars);
            }
            Expression::Powf(_, p) => {
                p.0.find_all_variables(vars);
                p.1.find_all_variables(vars);
            }
            Expression::BuiltinFun(_, _, a) => {
                a.find_all_variables(vars);
            }
            Expression::SubExpression(_, _) => {}
            Expression::ExternalFun(_, _, a) => {
                for arg in a {
                    arg.find_all_variables(vars);
                }
            }
            Expression::IfElse(_, b) => {
                b.0.find_all_variables(vars);
                b.1.find_all_variables(vars);
                b.2.find_all_variables(vars);
            }
        }
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + InternalOrdering> EvalTree<T> {
    pub fn common_subexpression_elimination(&mut self) {
        self.expressions.common_subexpression_elimination();

        for (_, _, e) in &mut self.functions {
            e.common_subexpression_elimination();
        }
    }

    pub fn count_operations(&self) -> (usize, usize) {
        let mut add = 0;
        let mut mul = 0;
        for e in &self.functions {
            let (ea, em) = e.2.count_operations();
            add += ea;
            mul += em;
        }

        let (ea, em) = self.expressions.count_operations();
        (add + ea, mul + em)
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + InternalOrdering>
    SplitExpression<T>
{
    /// Eliminate common subexpressions in the expression, also checking for subexpressions
    /// up to length `max_subexpr_len`.
    pub fn common_subexpression_elimination(&mut self) {
        let mut subexpression_count = HashMap::default();

        for t in &mut self.tree {
            t.rehash(true);
            t.find_subexpression(&mut subexpression_count);
        }

        subexpression_count.retain(|_, v| *v > 1);

        let mut v: Vec<_> = subexpression_count
            .iter()
            .map(|(k, v)| (*v, (*k).clone()))
            .collect();
        v.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        // assign a unique index to every subexpression
        let mut h = HashMap::default();
        for (index, (_, e)) in v.iter().enumerate() {
            h.insert(&*e, self.subexpressions.len() + index);
        }

        for t in &mut self.tree {
            t.replace_subexpression(&h, false);
        }

        // replace subexpressions in subexpressions and
        // sort them based on their dependencies
        for (_, x) in &v {
            let mut he = x.clone();
            he.replace_subexpression(&h, true);
            self.subexpressions.push(he);
        }

        let mut dep_tree = vec![];
        for (i, s) in self.subexpressions.iter().enumerate() {
            let mut deps = vec![];
            s.get_dependent_subexpressions(&mut deps);
            dep_tree.push((i, deps.clone()));
        }

        let mut rename = HashMap::default();
        let mut new_subs = vec![];
        let mut i = 0;
        while !dep_tree.is_empty() {
            if dep_tree[i].1.iter().all(|x| rename.contains_key(x)) {
                rename.insert(dep_tree[i].0, new_subs.len());
                new_subs.push(self.subexpressions[dep_tree[i].0].clone());
                dep_tree.swap_remove(i);
                if i == dep_tree.len() {
                    i = 0;
                }
            } else {
                i = (i + 1) % dep_tree.len();
            }
        }

        for x in &mut new_subs {
            x.rename_subexpression(&rename);
        }
        for t in &mut self.tree {
            t.rename_subexpression(&rename);
        }

        self.subexpressions = new_subs;
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + InternalOrdering> Expression<T> {
    fn rename_subexpression(&mut self, subexp: &HashMap<usize, usize>) {
        match self {
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _) => {}
            Expression::Eval(_, _, ae) => {
                for arg in &mut *ae {
                    arg.rename_subexpression(subexp);
                }
            }
            Expression::Add(_, a) | Expression::Mul(_, a) => {
                for arg in &mut *a {
                    arg.rename_subexpression(subexp);
                }

                a.sort();
            }
            Expression::Pow(_, p) => {
                p.0.rename_subexpression(subexp);
            }
            Expression::Powf(_, p) => {
                p.0.rename_subexpression(subexp);
                p.1.rename_subexpression(subexp);
            }
            Expression::BuiltinFun(_, _, a) => {
                a.rename_subexpression(subexp);
            }
            Expression::SubExpression(h, i) => {
                *self = Expression::SubExpression(*h, *subexp.get(i).unwrap());
            }
            Expression::ExternalFun(_, _, a) => {
                for arg in a {
                    arg.rename_subexpression(subexp);
                }
            }
            Expression::IfElse(_, b) => {
                b.0.rename_subexpression(subexp);
                b.1.rename_subexpression(subexp);
                b.2.rename_subexpression(subexp);
            }
        }
    }

    fn get_dependent_subexpressions(&self, dep: &mut Vec<usize>) {
        match self {
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _) => {}
            Expression::Eval(_, _, ae) => {
                for arg in ae {
                    arg.get_dependent_subexpressions(dep);
                }
            }
            Expression::Add(_, a) | Expression::Mul(_, a) => {
                for arg in a {
                    arg.get_dependent_subexpressions(dep);
                }
            }
            Expression::Pow(_, p) => {
                p.0.get_dependent_subexpressions(dep);
            }
            Expression::Powf(_, p) => {
                p.0.get_dependent_subexpressions(dep);
                p.1.get_dependent_subexpressions(dep);
            }
            Expression::BuiltinFun(_, _, a) => {
                a.get_dependent_subexpressions(dep);
            }
            Expression::SubExpression(_, i) => {
                dep.push(*i);
            }
            Expression::ExternalFun(_, _, a) => {
                for arg in a {
                    arg.get_dependent_subexpressions(dep);
                }
            }
            Expression::IfElse(_, b) => {
                b.0.get_dependent_subexpressions(dep);
                b.1.get_dependent_subexpressions(dep);
                b.2.get_dependent_subexpressions(dep);
            }
        }
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + InternalOrdering>
    SplitExpression<T>
{
    pub fn count_operations(&self) -> (usize, usize) {
        let mut add = 0;
        let mut mul = 0;
        for e in &self.subexpressions {
            let (ea, em) = e.count_operations();
            add += ea;
            mul += em;
        }

        for e in &self.tree {
            let (ea, em) = e.count_operations();
            add += ea;
            mul += em;
        }

        (add, mul)
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + InternalOrdering> Expression<T> {
    // Count the number of additions and multiplications in the expression.
    pub fn count_operations(&self) -> (usize, usize) {
        match self {
            Expression::Const(_, _) => (0, 0),
            Expression::Parameter(_, _) => (0, 0),
            Expression::Eval(_, _, args) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in args {
                    let (a, m) = arg.count_operations();
                    add += a;
                    mul += m;
                }
                (add, mul)
            }
            Expression::Add(_, a) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in a {
                    let (a, m) = arg.count_operations();
                    add += a;
                    mul += m;
                }
                (add + a.len() - 1, mul)
            }
            Expression::Mul(_, m) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in m {
                    let (a, m) = arg.count_operations();
                    add += a;
                    mul += m;
                }
                (add, mul + m.len() - 1)
            }
            Expression::Pow(_, p) => {
                let (a, m) = p.0.count_operations();
                (a, m + p.1.unsigned_abs() as usize - 1)
            }
            Expression::Powf(_, p) => {
                let (a, m) = p.0.count_operations();
                let (a2, m2) = p.1.count_operations();
                (a + a2, m + m2 + 1) // not clear how to count this
            }
            Expression::ReadArg(_, _) => (0, 0),
            Expression::BuiltinFun(_, _, b) => b.count_operations(), // not clear how to count this, third arg?
            Expression::SubExpression(_, _) => (0, 0),
            Expression::ExternalFun(_, _, args) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in args {
                    let (a, m) = arg.count_operations();
                    add += a;
                    mul += m;
                }
                (add, mul)
            }
            Expression::IfElse(_, b) => {
                let (a1, m1) = b.0.count_operations();
                let (a2, m2) = b.1.count_operations();
                let (a3, m3) = b.2.count_operations();
                (a1 + a2 + a3, m1 + m2 + m3)
            }
        }
    }
}

impl<T: Real> EvalTree<T> {
    /// Evaluate the evaluation tree. Consider converting to a linear form for repeated evaluation.
    pub fn evaluate(&mut self, params: &[T], out: &mut [T]) {
        for (o, e) in out.iter_mut().zip(&self.expressions.tree) {
            *o = self.evaluate_impl(e, &self.expressions.subexpressions, params, &[])
        }
    }

    fn evaluate_impl(
        &self,
        expr: &Expression<T>,
        subexpressions: &[Expression<T>],
        params: &[T],
        args: &[T],
    ) -> T {
        match expr {
            Expression::Const(_, c) => c.as_ref().clone(),
            Expression::Parameter(_, p) => params[*p].clone(),
            Expression::Eval(_, f, e_args) => {
                let mut arg_buf = vec![T::new_zero(); e_args.len()];
                for (b, a) in arg_buf.iter_mut().zip(e_args.iter()) {
                    *b = self.evaluate_impl(a, subexpressions, params, args);
                }

                let func = &self.functions[*f as usize].2;
                self.evaluate_impl(&func.tree[0], &func.subexpressions, params, &arg_buf)
            }
            Expression::Add(_, a) => {
                let mut r = self.evaluate_impl(&a[0], subexpressions, params, args);
                for arg in &a[1..] {
                    r += self.evaluate_impl(arg, subexpressions, params, args);
                }
                r
            }
            Expression::Mul(_, m) => {
                let mut r = self.evaluate_impl(&m[0], subexpressions, params, args);
                for arg in &m[1..] {
                    r *= self.evaluate_impl(arg, subexpressions, params, args);
                }
                r
            }
            Expression::Pow(_, p) => {
                let (b, e) = &**p;
                let b_eval = self.evaluate_impl(b, subexpressions, params, args);

                if *e >= 0 {
                    b_eval.pow(*e as u64)
                } else {
                    b_eval.pow(e.unsigned_abs()).inv()
                }
            }
            Expression::Powf(_, p) => {
                let (b, e) = &**p;
                let b_eval = self.evaluate_impl(b, subexpressions, params, args);
                let e_eval = self.evaluate_impl(e, subexpressions, params, args);
                b_eval.powf(&e_eval)
            }
            Expression::ReadArg(_, i) => args[*i].clone(),
            Expression::BuiltinFun(_, s, a) => {
                let arg = self.evaluate_impl(a, subexpressions, params, args);
                match s.0.get_id() {
                    Symbol::EXP_ID => arg.exp(),
                    Symbol::LOG_ID => arg.log(),
                    Symbol::SIN_ID => arg.sin(),
                    Symbol::COS_ID => arg.cos(),
                    Symbol::SQRT_ID => arg.sqrt(),
                    Symbol::ABS_ID => arg.norm(),
                    Symbol::CONJ_ID => arg.conj(),
                    _ => unreachable!(),
                }
            }
            Expression::SubExpression(_, s) => {
                // TODO: cache
                self.evaluate_impl(&subexpressions[*s], subexpressions, params, args)
            }
            Expression::ExternalFun(_, name, _args) => {
                unimplemented!(
                    "External function calls not implemented for EvalTree: {}",
                    name
                );
            }
            Expression::IfElse(_, b) => {
                let cond = self.evaluate_impl(&b.0, subexpressions, params, args);
                if !cond.is_fully_zero() {
                    self.evaluate_impl(&b.1, subexpressions, params, args)
                } else {
                    self.evaluate_impl(&b.2, subexpressions, params, args)
                }
            }
        }
    }
}

/// Represents exported code that can be compiled with [Self::compile].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Clone)]
pub struct ExportedCode<T: CompiledNumber> {
    path: PathBuf,
    function_name: String,
    _phantom: std::marker::PhantomData<T>,
}

/// Represents a library that can be loaded with [Self::load].
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Clone)]
pub struct CompiledCode<T: CompiledNumber> {
    path: PathBuf,
    function_name: String,
    _phantom: std::marker::PhantomData<T>,
}

/// Maximum length stored in the error message buffer
const CUDA_ERRMSG_LEN: usize = 256;
/// Struct representing the data created for the CUDA evaluation.
#[repr(C)]
pub struct CudaEvaluationData {
    pub params: *mut c_void,
    pub out: *mut c_void,
    pub n: usize,             // Number of evaluations
    pub block_size: usize,    // Number of threads per block
    pub in_dimension: usize,  // Number of input parameters
    pub out_dimension: usize, // Number of output parameters
    pub last_error: i32,
    pub errmsg: [std::os::raw::c_char; CUDA_ERRMSG_LEN],
}

impl CudaEvaluationData {
    pub fn check_for_error(&self) -> Result<(), String> {
        unsafe {
            if self.last_error != 0 {
                let err_msg = std::ffi::CStr::from_ptr(self.errmsg.as_ptr())
                    .to_string_lossy()
                    .into_owned();
                return Err(format!("CUDA error: {}", err_msg));
            }
        }
        Ok(())
    }
}

/// Settings for CUDA.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Clone)]
pub struct CudaLoadSettings {
    pub number_of_evaluations: usize,
    /// The number of threads per block for CUDA evaluation.
    pub block_size: usize,
}

impl Default for CudaLoadSettings {
    fn default() -> Self {
        CudaLoadSettings {
            number_of_evaluations: 1,
            block_size: 256, // default CUDA block size
        }
    }
}

impl<T: CompiledNumber> CompiledCode<T> {
    /// Load the evaluator from the compiled shared library.
    pub fn load(&self) -> Result<T::Evaluator, String> {
        T::Evaluator::load(&self.path, &self.function_name)
    }

    /// Load the evaluator from the compiled shared library.
    pub fn load_with_settings(&self, settings: T::Settings) -> Result<T::Evaluator, String> {
        T::Evaluator::load_with_settings(&self.path, &self.function_name, settings)
    }
}

type EvalTypeWithBuffer<'a, T> =
    libloading::Symbol<'a, unsafe extern "C" fn(params: *const T, buffer: *mut T, out: *mut T)>;
type CudaEvalType<'a, T> = libloading::Symbol<
    'a,
    unsafe extern "C" fn(params: *const T, out: *mut T, data: *const CudaEvaluationData),
>;
type CudaInitDataType<'a> = libloading::Symbol<
    'a,
    unsafe extern "C" fn(n: usize, block_size: usize) -> *const CudaEvaluationData,
>;
type CudaDestroyDataType<'a> =
    libloading::Symbol<'a, unsafe extern "C" fn(data: *const CudaEvaluationData) -> i32>;
type GetBufferLenType<'a> = libloading::Symbol<'a, unsafe extern "C" fn() -> c_ulong>;

struct EvaluatorFunctionsRealf64<'lib> {
    eval: EvalTypeWithBuffer<'lib, f64>,
    get_buffer_len: GetBufferLenType<'lib>,
}

impl<'lib> EvaluatorFunctionsRealf64<'lib> {
    fn new(lib: &'lib libloading::Library, function_name: &str) -> Result<Self, String> {
        let function_name = f64::construct_function_name(function_name);
        unsafe {
            let eval: EvalTypeWithBuffer<'lib, f64> = lib
                .get(function_name.to_string().as_bytes())
                .map_err(|e| e.to_string())?;
            let get_buffer_len: GetBufferLenType<'lib> = lib
                .get(format!("{}_get_buffer_len", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            Ok(EvaluatorFunctionsRealf64 {
                eval,
                get_buffer_len,
            })
        }
    }
}

type L = std::sync::Arc<libloading::Library>;

self_cell!(
    struct LibraryRealf64 {
        owner: L,

        #[covariant]
        dependent: EvaluatorFunctionsRealf64,
    }
);

struct EvaluatorFunctionsSimdRealf64<'lib> {
    eval: EvalTypeWithBuffer<'lib, wide::f64x4>,
    get_buffer_len: GetBufferLenType<'lib>,
}

impl<'lib> EvaluatorFunctionsSimdRealf64<'lib> {
    fn new(lib: &'lib libloading::Library, function_name: &str) -> Result<Self, String> {
        let function_name = wide::f64x4::construct_function_name(function_name);
        unsafe {
            let eval: EvalTypeWithBuffer<'lib, wide::f64x4> = lib
                .get(function_name.to_string().as_bytes())
                .map_err(|e| e.to_string())?;
            let get_buffer_len: GetBufferLenType<'lib> = lib
                .get(format!("{}_get_buffer_len", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            Ok(EvaluatorFunctionsSimdRealf64 {
                eval,
                get_buffer_len,
            })
        }
    }
}

self_cell!(
    struct LibrarySimdComplexf64 {
        owner: L,

        #[covariant]
        dependent: EvaluatorFunctionsSimdComplexf64,
    }
);

struct EvaluatorFunctionsSimdComplexf64<'lib> {
    eval: EvalTypeWithBuffer<'lib, Complex<wide::f64x4>>,
    get_buffer_len: GetBufferLenType<'lib>,
}

impl<'lib> EvaluatorFunctionsSimdComplexf64<'lib> {
    fn new(lib: &'lib libloading::Library, function_name: &str) -> Result<Self, String> {
        let function_name = Complex::<wide::f64x4>::construct_function_name(function_name);
        unsafe {
            let eval: EvalTypeWithBuffer<'lib, Complex<wide::f64x4>> = lib
                .get(function_name.to_string().as_bytes())
                .map_err(|e| e.to_string())?;
            let get_buffer_len: GetBufferLenType<'lib> = lib
                .get(format!("{}_get_buffer_len", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            Ok(EvaluatorFunctionsSimdComplexf64 {
                eval,
                get_buffer_len,
            })
        }
    }
}

self_cell!(
    struct LibrarySimdRealf64 {
        owner: L,

        #[covariant]
        dependent: EvaluatorFunctionsSimdRealf64,
    }
);

struct EvaluatorFunctionsComplexf64<'lib> {
    eval: EvalTypeWithBuffer<'lib, Complex<f64>>,
    get_buffer_len: GetBufferLenType<'lib>,
}

impl<'lib> EvaluatorFunctionsComplexf64<'lib> {
    fn new(lib: &'lib libloading::Library, function_name: &str) -> Result<Self, String> {
        let function_name = Complex::<f64>::construct_function_name(function_name);
        unsafe {
            let eval: EvalTypeWithBuffer<'lib, Complex<f64>> = lib
                .get(function_name.to_string().as_bytes())
                .map_err(|e| e.to_string())?;
            let get_buffer_len: GetBufferLenType<'lib> = lib
                .get(format!("{}_get_buffer_len", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            Ok(EvaluatorFunctionsComplexf64 {
                eval,
                get_buffer_len,
            })
        }
    }
}

self_cell!(
    struct LibraryComplexf64 {
        owner: L,

        #[covariant]
        dependent: EvaluatorFunctionsComplexf64,
    }
);

struct EvaluatorFunctionsCudaRealf64<'lib> {
    eval: CudaEvalType<'lib, f64>,
    init_data: CudaInitDataType<'lib>,
    destroy_data: CudaDestroyDataType<'lib>,
}

impl<'lib> EvaluatorFunctionsCudaRealf64<'lib> {
    fn new(lib: &'lib libloading::Library, function_name: &str) -> Result<Self, String> {
        let function_name = CudaRealf64::construct_function_name(function_name);
        unsafe {
            let eval: CudaEvalType<'lib, f64> = lib
                .get(format!("{}_vec", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            let init_data: CudaInitDataType<'lib> = lib
                .get(format!("{}_init_data", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            let destroy_data: CudaDestroyDataType<'lib> = lib
                .get(format!("{}_destroy_data", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            Ok(EvaluatorFunctionsCudaRealf64 {
                eval,
                init_data,
                destroy_data,
            })
        }
    }
}

self_cell!(
    struct LibraryCudaRealf64 {
        owner: L,

        #[covariant]
        dependent: EvaluatorFunctionsCudaRealf64,
    }
);

struct EvaluatorFunctionsCudaComplexf64<'lib> {
    eval: CudaEvalType<'lib, Complex<f64>>,
    init_data: CudaInitDataType<'lib>,
    destroy_data: CudaDestroyDataType<'lib>,
}

impl<'lib> EvaluatorFunctionsCudaComplexf64<'lib> {
    fn new(lib: &'lib libloading::Library, function_name: &str) -> Result<Self, String> {
        let function_name = CudaComplexf64::construct_function_name(function_name);
        unsafe {
            let eval: CudaEvalType<'lib, Complex<f64>> = lib
                .get(format!("{}_vec", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            let init_data: CudaInitDataType<'lib> = lib
                .get(format!("{}_init_data", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            let destroy_data: CudaDestroyDataType<'lib> = lib
                .get(format!("{}_destroy_data", function_name).as_bytes())
                .map_err(|e| e.to_string())?;
            Ok(EvaluatorFunctionsCudaComplexf64 {
                eval,
                init_data,
                destroy_data,
            })
        }
    }
}

self_cell!(
    struct LibraryCudaComplexf64 {
        owner: L,

        #[covariant]
        dependent: EvaluatorFunctionsCudaComplexf64,
    }
);

impl ExpressionEvaluator<Complex<Rational>> {
    /// JIT-compiles the evaluator using SymJIT.
    ///
    /// You can supply the types [f64], [wide::f64x4] for SIMD, [Complex] over [f64] and [wide::f64x4] for Complex SIMD.
    ///
    /// # Examples
    ///
    /// Compile and evaluate the function `x + y` for `f64` inputs:
    /// ```rust
    /// # use symbolica::{atom::AtomCore, parse};
    /// # use symbolica::evaluate::{FunctionMap, OptimizationSettings};
    /// let params = vec![parse!("x"), parse!("y")];
    /// let mut evaluator = parse!("x + y")
    ///     .evaluator(&FunctionMap::new(), &params, OptimizationSettings::default())
    ///     .unwrap()
    ///     .jit_compile::<f64>()
    ///     .unwrap();
    ///
    /// let mut res = [0.];
    /// evaluator.evaluate(&[1., 2.], &mut res);
    /// assert_eq!(res, [3.]);
    pub fn jit_compile<T: JITCompiledNumber>(&self) -> Result<JITCompiledEvaluator<T>, String> {
        let (instructions, _, constants) = self.export_instructions();
        let constants = constants
            .into_iter()
            .map(|c| symjit::Complex::new(c.re.to_f64(), c.im.to_f64()))
            .collect::<Vec<_>>();
        T::jit_compile(instructions, constants, HashMap::default())
    }
}

impl<T: JITCompiledNumber + Clone> ExpressionEvaluator<T> {
    /// JIT-compiles the evaluator using SymJIT.
    ///
    /// # Examples
    ///
    /// Compile and evaluate the function `x + y` for `f64` inputs:
    /// ```rust
    /// # use symbolica::{atom::AtomCore, parse};
    /// # use symbolica::evaluate::{FunctionMap, OptimizationSettings};
    /// let params = vec![parse!("x"), parse!("y")];
    /// let mut evaluator = parse!("x + y")
    ///     .evaluator(&FunctionMap::new(), &params, OptimizationSettings::default())
    ///     .unwrap()
    ///     .jit_compile::<f64>()
    ///     .unwrap();
    ///
    /// let mut res = [0.];
    /// evaluator.evaluate(&[1., 2.], &mut res);
    /// assert_eq!(res, [3.]);
    pub fn jit_compile(&self) -> Result<JITCompiledEvaluator<T>, String> {
        let (instructions, _, constants) = self.export_instructions();
        let constants = constants
            .into_iter()
            .map(|c| c.to_complex_f64())
            .collect::<Result<Vec<_>, _>>()?;
        T::jit_compile(instructions, constants, HashMap::default())
    }
}

impl<T: JITCompiledNumber + Clone> ExpressionEvaluatorWithExternalFunctions<T> {
    /// JIT-compiles the evaluator using SymJIT.
    ///
    /// # Examples
    ///
    /// Compile and evaluate the function `x + y` for `f64` inputs:
    /// ```rust
    /// # use symbolica::{atom::AtomCore, parse};
    /// # use symbolica::evaluate::{FunctionMap, OptimizationSettings};
    /// let params = vec![parse!("x"), parse!("y")];
    /// let mut evaluator = parse!("x + y")
    ///     .evaluator(&FunctionMap::new(), &params, OptimizationSettings::default())
    ///     .unwrap()
    ///     .jit_compile::<f64>()
    ///     .unwrap();
    ///
    /// let mut res = [0.];
    /// evaluator.evaluate(&[1., 2.], &mut res);
    /// assert_eq!(res, [3.]);
    pub fn jit_compile(&self) -> Result<JITCompiledEvaluator<T>, String> {
        let (instructions, _, constants) = self.eval.export_instructions();
        let constants = constants
            .into_iter()
            .map(|c| c.to_complex_f64())
            .collect::<Result<Vec<_>, _>>()?;

        let external_fns = self
            .external_fns
            .iter()
            .map(|(_, name, f)| (name.clone(), f.clone()))
            .collect::<HashMap<_, _>>();
        T::jit_compile(instructions, constants, external_fns)
    }
}

fn translate_to_symjit(
    instructions: Vec<Instruction>,
    constants: Vec<symjit::Complex<f64>>,
    config: Config,
    defuns: Defuns,
) -> Result<Translator, String> {
    let mut translator = Translator::new(config, defuns);

    for z in constants {
        translator.append_constant(z).unwrap();
    }

    fn slot(s: Slot) -> symjit::compiler::Slot {
        match s {
            Slot::Param(id) => symjit::compiler::Slot::Param(id),
            Slot::Out(id) => symjit::compiler::Slot::Out(id),
            Slot::Const(id) => symjit::compiler::Slot::Const(id),
            Slot::Temp(id) => symjit::compiler::Slot::Temp(id),
        }
    }

    fn slot_list(v: &[Slot]) -> Vec<symjit::compiler::Slot> {
        v.iter()
            .map(|s| slot(*s))
            .collect::<Vec<symjit::compiler::Slot>>()
    }

    fn builtin_symbol(s: BuiltinSymbol) -> symjit::compiler::BuiltinSymbol {
        symjit::compiler::BuiltinSymbol(s.get_symbol().get_id())
    }

    for q in instructions {
        match q {
            Instruction::Add(lhs, args, num_reals) => translator
                .append_add(&slot(lhs), &slot_list(&args), num_reals)
                .unwrap(),
            Instruction::Mul(lhs, args, num_reals) => translator
                .append_mul(&slot(lhs), &slot_list(&args), num_reals)
                .unwrap(),
            Instruction::Pow(lhs, arg, p, is_real) => translator
                .append_pow(&slot(lhs), &slot(arg), p, is_real)
                .unwrap(),
            Instruction::Powf(lhs, arg, p, is_real) => translator
                .append_powf(&slot(lhs), &slot(arg), &slot(p), is_real)
                .unwrap(),
            Instruction::Assign(lhs, rhs) => {
                translator.append_assign(&slot(lhs), &slot(rhs)).unwrap()
            }
            Instruction::Fun(lhs, fun, arg, is_real) => translator
                .append_fun(&slot(lhs), &builtin_symbol(fun), &slot(arg), is_real)
                .unwrap(),
            Instruction::Join(lhs, cond, true_val, false_val) => translator
                .append_join(&slot(lhs), &slot(cond), &slot(true_val), &slot(false_val))
                .unwrap(),
            Instruction::Label(id) => translator.append_label(id).unwrap(),
            Instruction::IfElse(cond, id) => translator.append_if_else(&slot(cond), id).unwrap(),
            Instruction::Goto(id) => translator.append_goto(id).unwrap(),
            Instruction::ExternalFun(lhs, op, args) => translator
                .append_external_fun(&slot(lhs), &op, &slot_list(&args))
                .unwrap(),
        }
    }

    Ok(translator)
}

pub trait JITCompiledNumber: Sized {
    fn to_complex_f64(&self) -> Result<symjit::Complex<f64>, String>;

    /// Create a JIT-compiled evaluator for this number type.
    fn jit_compile(
        instructions: Vec<Instruction>,
        constants: Vec<symjit::Complex<f64>>,
        external_functions: HashMap<String, Box<dyn ExternalFunction<Self>>>,
    ) -> Result<JITCompiledEvaluator<Self>, String>;

    fn evaluate(eval: &mut JITCompiledEvaluator<Self>, args: &[Self], out: &mut [Self]);
}

impl JITCompiledNumber for f64 {
    fn to_complex_f64(&self) -> Result<symjit::Complex<f64>, String> {
        Ok(symjit::Complex::new(*self, 0.))
    }

    fn jit_compile(
        instructions: Vec<Instruction>,
        constants: Vec<symjit::Complex<f64>>,
        external_functions: HashMap<String, Box<dyn ExternalFunction<f64>>>,
    ) -> Result<JITCompiledEvaluator<Self>, String> {
        if constants.iter().any(|x| x.im != 0.) {
            return Err("complex constants are not supported for f64 JIT export".to_string());
        }

        let mut config = Config::default();
        config.set_complex(false);
        config.set_simd(false);

        let mut defuns = Defuns::new();

        for (name, f) in external_functions {
            let r: Box<Box<dyn Fn(&[Self]) -> Self + Send + Sync>> = Box::new(f);

            defuns
                .add_sliced_func(&name, r)
                .map_err(|e| e.to_string())?;
        }

        let mut translator = translate_to_symjit(instructions, constants, config, defuns)?;

        Ok(JITCompiledEvaluator {
            code: translator
                .compile()
                .map_err(|e| e.to_string())?
                .seal()
                .map_err(|e| e.to_string())?,
            batch_input_buffer: Vec::new(),
            batch_output_buffer: Vec::new(),
        })
    }

    #[inline(always)]
    fn evaluate(eval: &mut JITCompiledEvaluator<Self>, args: &[Self], out: &mut [Self]) {
        eval.code.evaluate(args, out);
    }
}

/// A JIT-compiled evaluator for expressions, using the SymJIT compiler.
#[derive(Clone)]
pub struct JITCompiledEvaluator<T> {
    code: Applet,
    batch_input_buffer: Vec<T>,
    batch_output_buffer: Vec<T>,
}

impl<T: JITCompiledNumber> JITCompiledEvaluator<T> {
    /// Evaluate the JIT compiled code.
    #[inline(always)]
    pub fn evaluate(&mut self, args: &[T], out: &mut [T]) {
        T::evaluate(self, args, out);
    }
}

impl BatchEvaluator<f64> for JITCompiledEvaluator<f64> {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;
        for (o, i) in out.chunks_mut(n_out).zip(params.chunks(n_params)) {
            self.evaluate(i, o);
        }

        Ok(())
    }
}

impl JITCompiledNumber for wide::f64x4 {
    fn to_complex_f64(&self) -> Result<symjit::Complex<f64>, String> {
        let a = self.as_array();
        if !a.iter().all(|x| *x == a[0]) {
            return Err(format!("SIMD value {:?} is not a scalar", self));
        }

        Ok(symjit::Complex::new(a[0], 0.))
    }

    fn jit_compile(
        instructions: Vec<Instruction>,
        constants: Vec<symjit::Complex<f64>>,
        external_functions: HashMap<String, Box<dyn ExternalFunction<Self>>>,
    ) -> Result<JITCompiledEvaluator<Self>, String> {
        let mut config = Config::default();
        config.set_complex(false);
        config.set_simd(true);

        let mut defuns = Defuns::new();

        for (name, f) in external_functions {
            let r: Box<Box<dyn Fn(&[Self]) -> Self + Send + Sync>> = Box::new(f.clone());

            defuns
                .add_sliced_func(&name, r)
                .map_err(|e| e.to_string())?;
        }

        let mut translator = translate_to_symjit(instructions, constants, config, defuns)?;

        Ok(JITCompiledEvaluator {
            code: translator
                .compile()
                .map_err(|e| e.to_string())?
                .seal()
                .map_err(|e| e.to_string())?,
            batch_input_buffer: Vec::new(),
            batch_output_buffer: Vec::new(),
        })
    }

    #[inline(always)]
    fn evaluate(
        eval: &mut JITCompiledEvaluator<wide::f64x4>,
        args: &[wide::f64x4],
        out: &mut [wide::f64x4],
    ) {
        eval.code.evaluate(args, out);
    }
}

impl BatchEvaluator<f64> for JITCompiledEvaluator<wide::f64x4> {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;

        self.batch_input_buffer
            .resize(batch_size.div_ceil(4) * n_params, wide::f64x4::ZERO);

        for (dest, i) in self
            .batch_input_buffer
            .chunks_mut(n_params)
            .zip(params.chunks(4 * n_params))
        {
            if i.len() / n_params == 4 {
                for (j, d) in dest.iter_mut().enumerate() {
                    *d = wide::f64x4::from([
                        i[j],
                        i[j + n_params],
                        i[j + 2 * n_params],
                        i[j + 3 * n_params],
                    ]);
                }
            } else {
                for (j, d) in dest.iter_mut().enumerate() {
                    *d = wide::f64x4::from([
                        i[j],
                        if j + n_params < i.len() {
                            i[j + n_params]
                        } else {
                            0.0
                        },
                        if j + 2 * n_params < i.len() {
                            i[j + 2 * n_params]
                        } else {
                            0.0
                        },
                        if j + 3 * n_params < i.len() {
                            i[j + 3 * n_params]
                        } else {
                            0.0
                        },
                    ]);
                }
            }
        }

        self.batch_output_buffer
            .resize(batch_size.div_ceil(4) * n_out, wide::f64x4::ZERO);

        let param_buffer = std::mem::take(&mut self.batch_input_buffer);
        let mut output_buffer = std::mem::take(&mut self.batch_output_buffer);

        for (o, i) in output_buffer
            .chunks_mut(n_out)
            .zip(param_buffer.chunks(n_params))
        {
            self.evaluate(i, o);
        }

        for (o, i) in out.chunks_mut(4 * n_out).zip(&output_buffer) {
            o.copy_from_slice(&i.as_array()[..o.len()]);
        }

        self.batch_input_buffer = param_buffer;
        self.batch_output_buffer = output_buffer;

        Ok(())
    }
}

impl JITCompiledNumber for Complex<f64> {
    fn to_complex_f64(&self) -> Result<symjit::Complex<f64>, String> {
        Ok(symjit::Complex::new(self.re, self.im))
    }

    fn jit_compile(
        instructions: Vec<Instruction>,
        constants: Vec<symjit::Complex<f64>>,
        external_functions: HashMap<String, Box<dyn ExternalFunction<Self>>>,
    ) -> Result<JITCompiledEvaluator<Complex<f64>>, String> {
        let mut config = Config::default();
        config.set_complex(true);
        config.set_simd(false);

        let mut defuns = Defuns::new();

        for (name, f) in external_functions {
            // TODO: implement symjit::Element on numeric::Complex
            let k = Box::new(move |x: &[symjit::Complex<f64>]| {
                let ars = unsafe { std::mem::transmute(x) };
                let res = f(ars);
                symjit::Complex::new(res.re, res.im)
            });

            defuns
                .add_sliced_func(&name, k)
                .map_err(|e| e.to_string())?;
        }

        let mut translator = translate_to_symjit(instructions, constants, config, defuns)?;

        Ok(JITCompiledEvaluator {
            code: translator
                .compile()
                .map_err(|e| e.to_string())?
                .seal()
                .map_err(|e| e.to_string())?,
            batch_input_buffer: Vec::new(),
            batch_output_buffer: Vec::new(),
        })
    }

    /// Evaluate the compiled code with double-precision floating point numbers.
    #[inline(always)]
    fn evaluate(
        eval: &mut JITCompiledEvaluator<Complex<f64>>,
        args: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) {
        let args: &[symjit::Complex<f64>] = unsafe { std::mem::transmute(args) };
        let out: &mut [symjit::Complex<f64>] = unsafe { std::mem::transmute(out) };
        eval.code.evaluate(args, out);
    }
}

impl BatchEvaluator<Complex<f64>> for JITCompiledEvaluator<Complex<f64>> {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;
        for (o, i) in out.chunks_mut(n_out).zip(params.chunks(n_params)) {
            self.evaluate(i, o);
        }

        Ok(())
    }
}

impl JITCompiledNumber for Complex<wide::f64x4> {
    fn to_complex_f64(&self) -> Result<symjit::Complex<f64>, String> {
        let re = self.re.as_array();
        if !re.iter().all(|x| *x == re[0]) {
            return Err(format!("SIMD value {:?} is not a scalar", self));
        }

        let im = self.im.as_array();
        if !im.iter().all(|x| *x == im[0]) {
            return Err(format!("SIMD value {:?} is not a scalar", self));
        }

        Ok(symjit::Complex::new(re[0], im[0]))
    }

    /// JIT-compiles the evaluator using SymJIT.
    fn jit_compile(
        instructions: Vec<Instruction>,
        constants: Vec<symjit::Complex<f64>>,
        external_functions: HashMap<String, Box<dyn ExternalFunction<Self>>>,
    ) -> Result<JITCompiledEvaluator<Self>, String> {
        let mut config = Config::default();
        config.set_complex(true);
        config.set_simd(true);

        let mut defuns = Defuns::new();

        for (name, f) in external_functions {
            // TODO: implement symjit::Element on numeric::Complex
            let k = Box::new(move |x: &[symjit::Complex<wide::f64x4>]| {
                let ars = unsafe { std::mem::transmute(x) };
                let res = f(ars);
                symjit::Complex::new(res.re, res.im)
            });

            defuns
                .add_sliced_func(&name, k)
                .map_err(|e| e.to_string())?;
        }

        let mut translator = translate_to_symjit(instructions, constants, config, defuns)?;

        Ok(JITCompiledEvaluator {
            code: translator
                .compile()
                .map_err(|e| e.to_string())?
                .seal()
                .map_err(|e| e.to_string())?,
            batch_input_buffer: Vec::new(),
            batch_output_buffer: Vec::new(),
        })
    }

    #[inline(always)]
    fn evaluate(
        eval: &mut JITCompiledEvaluator<Self>,
        args: &[Complex<wide::f64x4>],
        out: &mut [Complex<wide::f64x4>],
    ) {
        let args: &[symjit::Complex<wide::f64x4>] = unsafe { std::mem::transmute(args) };
        let out: &mut [symjit::Complex<wide::f64x4>] = unsafe { std::mem::transmute(out) };

        eval.code.evaluate(args, out);
    }
}

impl BatchEvaluator<Complex<f64>> for JITCompiledEvaluator<Complex<wide::f64x4>> {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;

        self.batch_input_buffer.resize(
            batch_size.div_ceil(4) * n_params,
            Complex::new(wide::f64x4::ZERO, wide::f64x4::ZERO),
        );

        for (dest, i) in self
            .batch_input_buffer
            .chunks_mut(n_params)
            .zip(params.chunks(4 * n_params))
        {
            if i.len() / n_params == 4 {
                for (j, d) in dest.iter_mut().enumerate() {
                    d.re = wide::f64x4::from([
                        i[j].re,
                        i[j + n_params].re,
                        i[j + 2 * n_params].re,
                        i[j + 3 * n_params].re,
                    ]);
                    d.im = wide::f64x4::from([
                        i[j].im,
                        i[j + n_params].im,
                        i[j + 2 * n_params].im,
                        i[j + 3 * n_params].im,
                    ]);
                }
            } else {
                for (j, d) in dest.iter_mut().enumerate() {
                    d.re = wide::f64x4::from([
                        i[j].re,
                        if j + n_params < i.len() {
                            i[j + n_params].re
                        } else {
                            0.0
                        },
                        if j + 2 * n_params < i.len() {
                            i[j + 2 * n_params].re
                        } else {
                            0.0
                        },
                        if j + 3 * n_params < i.len() {
                            i[j + 3 * n_params].re
                        } else {
                            0.0
                        },
                    ]);
                    d.im = wide::f64x4::from([
                        i[j].im,
                        if j + n_params < i.len() {
                            i[j + n_params].im
                        } else {
                            0.0
                        },
                        if j + 2 * n_params < i.len() {
                            i[j + 2 * n_params].im
                        } else {
                            0.0
                        },
                        if j + 3 * n_params < i.len() {
                            i[j + 3 * n_params].im
                        } else {
                            0.0
                        },
                    ]);
                }
            }
        }

        self.batch_output_buffer.resize(
            batch_size.div_ceil(4) * n_out,
            Complex::new(wide::f64x4::ZERO, wide::f64x4::ZERO),
        );

        let param_buffer = std::mem::take(&mut self.batch_input_buffer);
        let mut output_buffer = std::mem::take(&mut self.batch_output_buffer);

        for (o, i) in output_buffer
            .chunks_mut(n_out)
            .zip(param_buffer.chunks(n_params))
        {
            self.evaluate(i, o);
        }

        for (o, i) in out.chunks_mut(4 * n_out).zip(&output_buffer) {
            for (j, d) in o.iter_mut().enumerate() {
                d.re = i.re.as_array()[j];
                d.im = i.im.as_array()[j];
            }
        }

        self.batch_input_buffer = param_buffer;
        self.batch_output_buffer = output_buffer;

        Ok(())
    }
}

/// A number type that can be used to call a compiled evaluator.
pub trait CompiledNumber: Sized {
    type Evaluator: EvaluatorLoader<Self>;
    type Settings: Default;
    /// A unique suffix for the evaluation function for this particular number type.
    // NOTE: a rename of any suffix will prevent loading older libraries.
    const SUFFIX: &'static str;

    /// Export an evaluator to C++ code for this number type.
    fn export_cpp<T: ExportNumber + SingleFloat>(
        eval: &ExpressionEvaluator<T>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String>;

    fn construct_function_name(function_name: &str) -> String {
        format!("{}_{}", function_name, Self::SUFFIX)
    }

    /// Get the default compilation options for C++ code generated
    /// for this number type.
    fn get_default_compile_options() -> CompileOptions;
}

/// Load a compiled evaluator from a shared library, optionally with settings.
pub trait EvaluatorLoader<T: CompiledNumber>: Sized {
    /// Load a compiled evaluator from a shared library.
    fn load(file: impl AsRef<Path>, function_name: &str) -> Result<Self, String> {
        Self::load_with_settings(file, function_name, T::Settings::default())
    }
    fn load_with_settings(
        file: impl AsRef<Path>,
        function_name: &str,
        settings: T::Settings,
    ) -> Result<Self, String>;
}

/// Batch-evaluate the compiled code with basic types such as [f64] or [`Complex<f64>`],
/// automatically reorganizing the batches if necessary.
pub trait BatchEvaluator<T: CompiledNumber> {
    /// Evaluate the compiled code with batched input with the given input parameters, writing the results to `out`.
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[T],
        out: &mut [T],
    ) -> Result<(), String>;
}

impl CompiledNumber for f64 {
    type Evaluator = CompiledRealEvaluator;
    type Settings = ();
    const SUFFIX: &'static str = "realf64";

    fn export_cpp<T: ExportNumber + SingleFloat>(
        eval: &ExpressionEvaluator<T>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String> {
        if !eval.stack.iter().all(|x| x.is_real()) {
            return Err(
                "Cannot create real evaluator with complex coefficients. Use Complex<f64>".into(),
            );
        }

        Ok(match settings.inline_asm {
            InlineASM::X64 => eval.export_asm_real_str(function_name, &settings),
            InlineASM::AArch64 => eval.export_asm_real_str(function_name, &settings),
            InlineASM::AVX2 => {
                Err("AVX2 not supported for complexf64: use Complex<f64x6> instead".to_owned())?
            }
            InlineASM::None => {
                let r = eval.export_generic_cpp_str(function_name, &settings, NumberClass::RealF64);
                r + format!("\nextern \"C\" {{\n\tvoid {function_name}(double *params, double *buffer, double *out) {{\n\t\t{function_name}_gen(params, buffer, out);\n\t\treturn;\n\t}}\n}}\n").as_str()
            }
        })
    }

    fn get_default_compile_options() -> CompileOptions {
        CompileOptions::default()
    }
}

impl BatchEvaluator<f64> for CompiledRealEvaluator {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;
        for (o, i) in out.chunks_mut(n_out).zip(params.chunks(n_params)) {
            self.evaluate(i, o);
        }

        Ok(())
    }
}

impl CompiledNumber for Complex<f64> {
    type Evaluator = CompiledComplexEvaluator;
    type Settings = ();
    const SUFFIX: &'static str = "complexf64";

    fn export_cpp<T: ExportNumber + SingleFloat>(
        eval: &ExpressionEvaluator<T>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String> {
        Ok(match settings.inline_asm {
            InlineASM::X64 => eval.export_asm_complex_str(function_name, &settings),
            InlineASM::AArch64 => eval.export_asm_complex_str(function_name, &settings),
            InlineASM::AVX2 => {
                Err("AVX2 not supported for complexf64: use Complex<f64x6> instead".to_owned())?
            }
            InlineASM::None => {
                let r =
                    eval.export_generic_cpp_str(function_name, &settings, NumberClass::ComplexF64);
                r + format!("\nextern \"C\" {{\n\tvoid {function_name}(std::complex<double> *params, std::complex<double> *buffer, std::complex<double> *out) {{\n\t\t{function_name}_gen(params, buffer, out);\n\t\treturn;\n\t}}\n}}\n").as_str()
            }
        })
    }

    fn get_default_compile_options() -> CompileOptions {
        CompileOptions::default()
    }
}

impl BatchEvaluator<Complex<f64>> for CompiledComplexEvaluator {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;
        for (o, i) in out.chunks_mut(n_out).zip(params.chunks(n_params)) {
            self.evaluate(i, o);
        }

        Ok(())
    }
}

/// Efficient evaluator for compiled real-valued functions.
pub struct CompiledRealEvaluator {
    library: LibraryRealf64,
    path: PathBuf,
    fn_name: String,
    buffer_double: Vec<f64>,
}

impl EvaluatorLoader<f64> for CompiledRealEvaluator {
    fn load_with_settings(
        path: impl AsRef<Path>,
        function_name: &str,
        _settings: (),
    ) -> Result<Self, String> {
        CompiledRealEvaluator::load(path, function_name)
    }
}

impl CompiledRealEvaluator {
    pub fn load_new_function(&self, function_name: &str) -> Result<CompiledRealEvaluator, String> {
        let library = LibraryRealf64::try_new(self.library.borrow_owner().clone(), |lib| {
            EvaluatorFunctionsRealf64::new(lib, function_name)
        })?;

        let len = unsafe { (library.borrow_dependent().get_buffer_len)() } as usize;

        Ok(CompiledRealEvaluator {
            path: self.path.clone(),
            fn_name: function_name.to_string(),
            buffer_double: vec![0.; len],
            library,
        })
    }
    pub fn load(
        path: impl AsRef<Path>,
        function_name: &str,
    ) -> Result<CompiledRealEvaluator, String> {
        unsafe {
            let lib = match libloading::Library::new(path.as_ref()) {
                Ok(lib) => lib,
                Err(_) => libloading::Library::new(PathBuf::new().join("./").join(&path))
                    .map_err(|e| e.to_string())?,
            };
            let library = LibraryRealf64::try_new(std::sync::Arc::new(lib), |lib| {
                EvaluatorFunctionsRealf64::new(lib, function_name)
            })?;

            let len = (library.borrow_dependent().get_buffer_len)() as usize;

            Ok(CompiledRealEvaluator {
                fn_name: function_name.to_string(),
                path: path.as_ref().to_path_buf(),
                buffer_double: vec![0.; len],
                library,
            })
        }
    }
    /// Evaluate the compiled code with double-precision floating point numbers.
    #[inline(always)]
    pub fn evaluate(&mut self, args: &[f64], out: &mut [f64]) {
        unsafe {
            (self.library.borrow_dependent().eval)(
                args.as_ptr(),
                self.buffer_double.as_mut_ptr(),
                out.as_mut_ptr(),
            )
        }
    }
}

unsafe impl Send for CompiledRealEvaluator {}

impl std::fmt::Debug for CompiledRealEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompiledRealEvaluator({})", self.fn_name)
    }
}

impl Clone for CompiledRealEvaluator {
    fn clone(&self) -> Self {
        self.load_new_function(&self.fn_name).unwrap()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for CompiledRealEvaluator {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.path, &self.fn_name).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CompiledRealEvaluator {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (file, fn_name) = <(PathBuf, String)>::deserialize(deserializer)?;
        CompiledRealEvaluator::load(&file, &fn_name).map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for CompiledRealEvaluator {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.path, encoder)?;
        bincode::Encode::encode(&self.fn_name, encoder)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(CompiledRealEvaluator);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for CompiledRealEvaluator {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let file: PathBuf = bincode::Decode::decode(decoder)?;
        let fn_name: String = bincode::Decode::decode(decoder)?;
        CompiledRealEvaluator::load(&file, &fn_name)
            .map_err(|e| bincode::error::DecodeError::OtherString(e))
    }
}

/// Efficient evaluator for compiled complex-valued functions.
pub struct CompiledComplexEvaluator {
    path: PathBuf,
    fn_name: String,
    library: LibraryComplexf64,
    buffer_complex: Vec<Complex<f64>>,
}

impl EvaluatorLoader<Complex<f64>> for CompiledComplexEvaluator {
    fn load_with_settings(
        path: impl AsRef<Path>,
        function_name: &str,
        _settings: (),
    ) -> Result<Self, String> {
        CompiledComplexEvaluator::load(path, function_name)
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for CompiledComplexEvaluator {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.path, &self.fn_name).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CompiledComplexEvaluator {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (file, fn_name) = <(PathBuf, String)>::deserialize(deserializer)?;
        CompiledComplexEvaluator::load(&file, &fn_name).map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for CompiledComplexEvaluator {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.path, encoder)?;
        bincode::Encode::encode(&self.fn_name, encoder)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(CompiledComplexEvaluator);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for CompiledComplexEvaluator {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let file: PathBuf = bincode::Decode::decode(decoder)?;
        let fn_name: String = bincode::Decode::decode(decoder)?;
        CompiledComplexEvaluator::load(&file, &fn_name)
            .map_err(|e| bincode::error::DecodeError::OtherString(e))
    }
}

impl CompiledComplexEvaluator {
    /// Load a new function from the same library.
    pub fn load_new_function(
        &self,
        function_name: &str,
    ) -> Result<CompiledComplexEvaluator, String> {
        let library = LibraryComplexf64::try_new(self.library.borrow_owner().clone(), |lib| {
            EvaluatorFunctionsComplexf64::new(lib, function_name)
        })?;

        let len = unsafe { (library.borrow_dependent().get_buffer_len)() } as usize;

        Ok(CompiledComplexEvaluator {
            path: self.path.clone(),
            fn_name: function_name.to_string(),
            buffer_complex: vec![Complex::new_zero(); len],
            library,
        })
    }

    /// Load a compiled evaluator from a shared library.
    pub fn load(
        path: impl AsRef<Path>,
        function_name: &str,
    ) -> Result<CompiledComplexEvaluator, String> {
        unsafe {
            let lib = match libloading::Library::new(path.as_ref()) {
                Ok(lib) => lib,
                Err(_) => libloading::Library::new(PathBuf::new().join("./").join(&path))
                    .map_err(|e| e.to_string())?,
            };

            let library = LibraryComplexf64::try_new(std::sync::Arc::new(lib), |lib| {
                EvaluatorFunctionsComplexf64::new(lib, function_name)
            })?;

            let len = (library.borrow_dependent().get_buffer_len)() as usize;

            Ok(CompiledComplexEvaluator {
                path: path.as_ref().to_path_buf(),
                fn_name: function_name.to_string(),
                buffer_complex: vec![Complex::default(); len],
                library,
            })
        }
    }
    /// Evaluate the compiled code.
    #[inline(always)]
    pub fn evaluate(&mut self, args: &[Complex<f64>], out: &mut [Complex<f64>]) {
        unsafe {
            (self.library.borrow_dependent().eval)(
                args.as_ptr(),
                self.buffer_complex.as_mut_ptr(),
                out.as_mut_ptr(),
            )
        }
    }
}

unsafe impl Send for CompiledComplexEvaluator {}

impl std::fmt::Debug for CompiledComplexEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompiledComplexEvaluator({})", self.fn_name)
    }
}

impl Clone for CompiledComplexEvaluator {
    fn clone(&self) -> Self {
        self.load_new_function(&self.fn_name).unwrap()
    }
}

/// Evaluate 4 double-precision floating point numbers in parallel using SIMD instructions.
/// Make sure you add arguments such as `-march=native` to enable full SIMD support for your platform.
///
/// Failure to add this, may result in only two double-precision numbers being evaluated in parallel.
///
/// The compilation requires the `xsimd` C++ library to be installed.
impl CompiledNumber for wide::f64x4 {
    type Evaluator = CompiledSimdRealEvaluator;
    type Settings = ();
    const SUFFIX: &'static str = "simd_realf64";

    fn export_cpp<T: ExportNumber + SingleFloat>(
        eval: &ExpressionEvaluator<T>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String> {
        if !eval.stack.iter().all(|x| x.is_real()) {
            return Err(
                "Cannot create real evaluator with complex coefficients. Use Complex<f64>".into(),
            );
        }

        Ok(match settings.inline_asm {
            // assume AVX2 for X64
            InlineASM::X64 => eval.export_simd_str(function_name, settings, false, InlineASM::AVX2),
            InlineASM::AArch64 => {
                Err("Inline assembly not supported yet for SIMD f64x4".to_owned())?
            }
            asm @ InlineASM::AVX2 | asm @ InlineASM::None => {
                eval.export_simd_str(function_name, settings, false, asm)
            }
        })
    }

    fn get_default_compile_options() -> CompileOptions {
        CompileOptions::default()
    }
}

impl BatchEvaluator<f64> for CompiledSimdRealEvaluator {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;

        self.batch_input_buffer
            .resize(batch_size.div_ceil(4) * n_params, wide::f64x4::ZERO);

        for (dest, i) in self
            .batch_input_buffer
            .chunks_mut(n_params)
            .zip(params.chunks(4 * n_params))
        {
            if i.len() / n_params == 4 {
                for (j, d) in dest.iter_mut().enumerate() {
                    *d = wide::f64x4::from([
                        i[j],
                        i[j + n_params],
                        i[j + 2 * n_params],
                        i[j + 3 * n_params],
                    ]);
                }
            } else {
                for (j, d) in dest.iter_mut().enumerate() {
                    *d = wide::f64x4::from([
                        i[j],
                        if j + n_params < i.len() {
                            i[j + n_params]
                        } else {
                            0.0
                        },
                        if j + 2 * n_params < i.len() {
                            i[j + 2 * n_params]
                        } else {
                            0.0
                        },
                        if j + 3 * n_params < i.len() {
                            i[j + 3 * n_params]
                        } else {
                            0.0
                        },
                    ]);
                }
            }
        }

        self.batch_output_buffer
            .resize(batch_size.div_ceil(4) * n_out, wide::f64x4::ZERO);

        let param_buffer = std::mem::take(&mut self.batch_input_buffer);
        let mut output_buffer = std::mem::take(&mut self.batch_output_buffer);

        for (o, i) in output_buffer
            .chunks_mut(n_out)
            .zip(param_buffer.chunks(n_params))
        {
            self.evaluate(i, o);
        }

        for (o, i) in out.chunks_mut(4 * n_out).zip(&output_buffer) {
            o.copy_from_slice(&i.as_array()[..o.len()]);
        }

        self.batch_input_buffer = param_buffer;
        self.batch_output_buffer = output_buffer;

        Ok(())
    }
}

/// Efficient evaluator using simd for compiled real-valued functions.
pub struct CompiledSimdRealEvaluator {
    path: PathBuf,
    fn_name: String,
    library: LibrarySimdRealf64,
    buffer: Vec<wide::f64x4>,
    batch_input_buffer: Vec<wide::f64x4>,
    batch_output_buffer: Vec<wide::f64x4>,
}

impl EvaluatorLoader<wide::f64x4> for CompiledSimdRealEvaluator {
    fn load(path: impl AsRef<Path>, function_name: &str) -> Result<Self, String> {
        CompiledSimdRealEvaluator::load_with_settings(path, function_name, ())
    }

    fn load_with_settings(
        path: impl AsRef<Path>,
        function_name: &str,
        _settings: (),
    ) -> Result<Self, String> {
        CompiledSimdRealEvaluator::load(path, function_name)
    }
}

impl CompiledSimdRealEvaluator {
    pub fn load_new_function(
        &self,
        function_name: &str,
    ) -> Result<CompiledSimdRealEvaluator, String> {
        let library = LibrarySimdRealf64::try_new(self.library.borrow_owner().clone(), |lib| {
            EvaluatorFunctionsSimdRealf64::new(lib, function_name)
        })?;

        Ok(CompiledSimdRealEvaluator {
            path: self.path.clone(),
            fn_name: function_name.to_string(),
            buffer: vec![
                wide::f64x4::ZERO;
                unsafe { (library.borrow_dependent().get_buffer_len)() } as usize
            ],
            batch_input_buffer: Vec::new(),
            batch_output_buffer: Vec::new(),
            library,
        })
    }

    pub fn load(
        path: impl AsRef<Path>,
        function_name: &str,
    ) -> Result<CompiledSimdRealEvaluator, String> {
        unsafe {
            let lib = match libloading::Library::new(path.as_ref()) {
                Ok(lib) => lib,
                Err(_) => libloading::Library::new(PathBuf::new().join("./").join(&path))
                    .map_err(|e| e.to_string())?,
            };
            let library = LibrarySimdRealf64::try_new(std::sync::Arc::new(lib), |lib| {
                EvaluatorFunctionsSimdRealf64::new(lib, function_name)
            })?;

            Ok(CompiledSimdRealEvaluator {
                path: path.as_ref().to_path_buf(),
                fn_name: function_name.to_string(),
                buffer: vec![
                    wide::f64x4::ZERO;
                    (library.borrow_dependent().get_buffer_len)() as usize
                ],
                batch_input_buffer: Vec::new(),
                batch_output_buffer: Vec::new(),
                library,
            })
        }
    }

    /// Evaluate the compiled code with 4 double-precision floating point numbers.
    /// The `args` must be of length `number_of_evaluations * input`, where `input` is the number of inputs to the function.
    /// The `out` must be of length `number_of_evaluations * output`,
    /// where `output` is the number of outputs of the function.
    #[inline(always)]
    pub fn evaluate(&mut self, args: &[wide::f64x4], out: &mut [wide::f64x4]) {
        unsafe {
            (self.library.borrow_dependent().eval)(
                args.as_ptr(),
                self.buffer.as_mut_ptr(),
                out.as_mut_ptr(),
            )
        }
    }
}

unsafe impl Send for CompiledSimdRealEvaluator {}

impl std::fmt::Debug for CompiledSimdRealEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompiledSimdRealEvaluator({})", self.fn_name)
    }
}

impl Clone for CompiledSimdRealEvaluator {
    fn clone(&self) -> Self {
        self.load_new_function(&self.fn_name).unwrap()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for CompiledSimdRealEvaluator {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.path, &self.fn_name).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CompiledSimdRealEvaluator {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (file, fn_name) = <(PathBuf, String)>::deserialize(deserializer)?;
        CompiledSimdRealEvaluator::load(&file, &fn_name).map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for CompiledSimdRealEvaluator {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.path, encoder)?;
        bincode::Encode::encode(&self.fn_name, encoder)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(CompiledSimdRealEvaluator);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for CompiledSimdRealEvaluator {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let file: PathBuf = bincode::Decode::decode(decoder)?;
        let fn_name: String = bincode::Decode::decode(decoder)?;
        CompiledSimdRealEvaluator::load(&file, &fn_name)
            .map_err(|e| bincode::error::DecodeError::OtherString(e))
    }
}

/// Evaluate 4 double-precision floating point numbers in parallel using SIMD instructions.
/// Make sure you add arguments such as `-march=native` to enable full SIMD support for your platform.
///
/// Failure to add this, may result in only two double-precision numbers being evaluated in parallel.
///
/// The compilation requires the `xsimd` C++ library to be installed.
impl CompiledNumber for Complex<wide::f64x4> {
    type Evaluator = CompiledSimdComplexEvaluator;
    type Settings = ();
    const SUFFIX: &'static str = "simd_complexf64";

    fn export_cpp<T: ExportNumber + SingleFloat>(
        eval: &ExpressionEvaluator<T>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String> {
        if !eval.stack.iter().all(|x| x.is_real()) {
            return Err(
                "Cannot create real evaluator with complex coefficients. Use Complex<f64>".into(),
            );
        }

        Ok(match settings.inline_asm {
            // assume AVX2 for X64
            InlineASM::X64 => eval.export_simd_str(function_name, settings, true, InlineASM::AVX2),
            InlineASM::AArch64 => {
                Err("X64 inline assembly not supported for SIMD f64x4: use AVX2".to_owned())?
            }
            asm @ InlineASM::AVX2 | asm @ InlineASM::None => {
                eval.export_simd_str(function_name, settings, true, asm)
            }
        })
    }

    fn get_default_compile_options() -> CompileOptions {
        CompileOptions::default()
    }
}

impl BatchEvaluator<Complex<f64>> for CompiledSimdComplexEvaluator {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> Result<(), String> {
        if !params.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Parameter length {} not divisible by batch size {}",
                params.len(),
                batch_size
            ));
        }
        if !out.len().is_multiple_of(batch_size) {
            return Err(format!(
                "Output length {} not divisible by batch size {}",
                out.len(),
                batch_size
            ));
        }

        let n_params = params.len() / batch_size;
        let n_out = out.len() / batch_size;

        self.batch_input_buffer.resize(
            batch_size.div_ceil(4) * n_params,
            Complex::new(wide::f64x4::ZERO, wide::f64x4::ZERO),
        );

        for (dest, i) in self
            .batch_input_buffer
            .chunks_mut(n_params)
            .zip(params.chunks(4 * n_params))
        {
            if i.len() / n_params == 4 {
                for (j, d) in dest.iter_mut().enumerate() {
                    d.re = wide::f64x4::from([
                        i[j].re,
                        i[j + n_params].re,
                        i[j + 2 * n_params].re,
                        i[j + 3 * n_params].re,
                    ]);
                    d.im = wide::f64x4::from([
                        i[j].im,
                        i[j + n_params].im,
                        i[j + 2 * n_params].im,
                        i[j + 3 * n_params].im,
                    ]);
                }
            } else {
                for (j, d) in dest.iter_mut().enumerate() {
                    d.re = wide::f64x4::from([
                        i[j].re,
                        if j + n_params < i.len() {
                            i[j + n_params].re
                        } else {
                            0.0
                        },
                        if j + 2 * n_params < i.len() {
                            i[j + 2 * n_params].re
                        } else {
                            0.0
                        },
                        if j + 3 * n_params < i.len() {
                            i[j + 3 * n_params].re
                        } else {
                            0.0
                        },
                    ]);
                    d.im = wide::f64x4::from([
                        i[j].im,
                        if j + n_params < i.len() {
                            i[j + n_params].im
                        } else {
                            0.0
                        },
                        if j + 2 * n_params < i.len() {
                            i[j + 2 * n_params].im
                        } else {
                            0.0
                        },
                        if j + 3 * n_params < i.len() {
                            i[j + 3 * n_params].im
                        } else {
                            0.0
                        },
                    ]);
                }
            }
        }

        self.batch_output_buffer.resize(
            batch_size.div_ceil(4) * n_out,
            Complex::new(wide::f64x4::ZERO, wide::f64x4::ZERO),
        );

        let param_buffer = std::mem::take(&mut self.batch_input_buffer);
        let mut output_buffer = std::mem::take(&mut self.batch_output_buffer);

        for (o, i) in output_buffer
            .chunks_mut(n_out)
            .zip(param_buffer.chunks(n_params))
        {
            self.evaluate(i, o);
        }

        for (o, i) in out.chunks_mut(4 * n_out).zip(&output_buffer) {
            for (j, d) in o.iter_mut().enumerate() {
                d.re = i.re.as_array()[j];
                d.im = i.im.as_array()[j];
            }
        }

        self.batch_input_buffer = param_buffer;
        self.batch_output_buffer = output_buffer;

        Ok(())
    }
}

/// Efficient evaluator using simd for compiled complex-valued functions.
pub struct CompiledSimdComplexEvaluator {
    path: PathBuf,
    fn_name: String,
    library: LibrarySimdComplexf64,
    buffer: Vec<Complex<wide::f64x4>>,
    batch_input_buffer: Vec<Complex<wide::f64x4>>,
    batch_output_buffer: Vec<Complex<wide::f64x4>>,
}

impl EvaluatorLoader<Complex<wide::f64x4>> for CompiledSimdComplexEvaluator {
    fn load(path: impl AsRef<Path>, function_name: &str) -> Result<Self, String> {
        CompiledSimdComplexEvaluator::load_with_settings(path, function_name, ())
    }

    fn load_with_settings(
        path: impl AsRef<Path>,
        function_name: &str,
        _settings: (),
    ) -> Result<Self, String> {
        CompiledSimdComplexEvaluator::load(path, function_name)
    }
}

impl CompiledSimdComplexEvaluator {
    pub fn load_new_function(
        &self,
        function_name: &str,
    ) -> Result<CompiledSimdComplexEvaluator, String> {
        let library = LibrarySimdComplexf64::try_new(self.library.borrow_owner().clone(), |lib| {
            EvaluatorFunctionsSimdComplexf64::new(lib, function_name)
        })?;

        Ok(CompiledSimdComplexEvaluator {
            path: self.path.clone(),
            fn_name: function_name.to_string(),
            buffer: vec![
                Complex::new(wide::f64x4::ZERO, wide::f64x4::ZERO);
                unsafe { (library.borrow_dependent().get_buffer_len)() } as usize
            ],
            batch_input_buffer: Vec::new(),
            batch_output_buffer: Vec::new(),
            library,
        })
    }

    pub fn load(
        path: impl AsRef<Path>,
        function_name: &str,
    ) -> Result<CompiledSimdComplexEvaluator, String> {
        unsafe {
            let lib = match libloading::Library::new(path.as_ref()) {
                Ok(lib) => lib,
                Err(_) => libloading::Library::new(PathBuf::new().join("./").join(&path))
                    .map_err(|e| e.to_string())?,
            };
            let library = LibrarySimdComplexf64::try_new(std::sync::Arc::new(lib), |lib| {
                EvaluatorFunctionsSimdComplexf64::new(lib, function_name)
            })?;

            Ok(CompiledSimdComplexEvaluator {
                path: path.as_ref().to_path_buf(),
                fn_name: function_name.to_string(),
                buffer: vec![
                    Complex::new(wide::f64x4::ZERO, wide::f64x4::ZERO);
                    (library.borrow_dependent().get_buffer_len)() as usize
                ],
                batch_input_buffer: Vec::new(),
                batch_output_buffer: Vec::new(),
                library,
            })
        }
    }

    /// Evaluate the compiled code with 4 double-precision floating point numbers.
    /// The `args` must be of length `number_of_evaluations * input`, where `input` is the number of inputs to the function.
    /// The `out` must be of length `number_of_evaluations * output`,
    /// where `output` is the number of outputs of the function.
    #[inline(always)]
    pub fn evaluate(&mut self, args: &[Complex<wide::f64x4>], out: &mut [Complex<wide::f64x4>]) {
        unsafe {
            (self.library.borrow_dependent().eval)(
                args.as_ptr(),
                self.buffer.as_mut_ptr(),
                out.as_mut_ptr(),
            )
        }
    }
}

unsafe impl Send for CompiledSimdComplexEvaluator {}

impl std::fmt::Debug for CompiledSimdComplexEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompiledSimdComplexEvaluator({})", self.fn_name)
    }
}

impl Clone for CompiledSimdComplexEvaluator {
    fn clone(&self) -> Self {
        self.load_new_function(&self.fn_name).unwrap()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for CompiledSimdComplexEvaluator {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.path, &self.fn_name).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CompiledSimdComplexEvaluator {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (file, fn_name) = <(PathBuf, String)>::deserialize(deserializer)?;
        CompiledSimdComplexEvaluator::load(&file, &fn_name).map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for CompiledSimdComplexEvaluator {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.path, encoder)?;
        bincode::Encode::encode(&self.fn_name, encoder)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(CompiledSimdComplexEvaluator);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for CompiledSimdComplexEvaluator {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let file: PathBuf = bincode::Decode::decode(decoder)?;
        let fn_name: String = bincode::Decode::decode(decoder)?;
        CompiledSimdComplexEvaluator::load(&file, &fn_name)
            .map_err(|e| bincode::error::DecodeError::OtherString(e))
    }
}

/// CUDA real number type.
pub struct CudaRealf64 {}

impl CompiledNumber for CudaRealf64 {
    type Evaluator = CompiledCudaRealEvaluator;
    type Settings = CudaLoadSettings;
    const SUFFIX: &'static str = "cuda_realf64";

    fn export_cpp<T: ExportNumber + SingleFloat>(
        eval: &ExpressionEvaluator<T>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String> {
        if !eval.stack.iter().all(|x| x.is_real()) {
            return Err(
                "Cannot create real evaluator with complex coefficients. Use Complex<f64>".into(),
            );
        }

        Ok(eval.export_cuda_str(function_name, settings, NumberClass::RealF64))
    }

    fn get_default_compile_options() -> CompileOptions {
        CompileOptions::cuda()
    }
}

/// CUDA complex number type.
pub struct CudaComplexf64 {}

impl CompiledNumber for CudaComplexf64 {
    type Evaluator = CompiledCudaComplexEvaluator;
    type Settings = CudaLoadSettings;
    const SUFFIX: &'static str = "cuda_complexf64";

    fn export_cpp<T: ExportNumber + SingleFloat>(
        eval: &ExpressionEvaluator<T>,
        function_name: &str,
        settings: ExportSettings,
    ) -> Result<String, String> {
        Ok(eval.export_cuda_str(function_name, settings, NumberClass::ComplexF64))
    }

    fn get_default_compile_options() -> CompileOptions {
        CompileOptions::cuda()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for CompiledCudaRealEvaluator {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.path, &self.fn_name, &self.settings).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CompiledCudaRealEvaluator {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (file, fn_name, settings) =
            <(PathBuf, String, CudaLoadSettings)>::deserialize(deserializer)?;
        CompiledCudaRealEvaluator::load_with_settings(&file, &fn_name, settings)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for CompiledCudaRealEvaluator {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.path, encoder)?;
        bincode::Encode::encode(&self.fn_name, encoder)?;
        bincode::Encode::encode(&self.settings, encoder)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(CompiledCudaRealEvaluator);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for CompiledCudaRealEvaluator {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let file: PathBuf = bincode::Decode::decode(decoder)?;
        let fn_name: String = bincode::Decode::decode(decoder)?;
        let settings: CudaLoadSettings = bincode::Decode::decode(decoder)?;
        CompiledCudaRealEvaluator::load(&file, &fn_name, settings)
            .map_err(|e| bincode::error::DecodeError::OtherString(e))
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for CompiledCudaComplexEvaluator {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        (&self.path, &self.fn_name).serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for CompiledCudaComplexEvaluator {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let (file, fn_name, settings) =
            <(PathBuf, String, CudaLoadSettings)>::deserialize(deserializer)?;
        CompiledCudaComplexEvaluator::load(&file, &fn_name, settings)
            .map_err(serde::de::Error::custom)
    }
}

#[cfg(feature = "bincode")]
impl bincode::Encode for CompiledCudaComplexEvaluator {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> core::result::Result<(), bincode::error::EncodeError> {
        bincode::Encode::encode(&self.path, encoder)?;
        bincode::Encode::encode(&self.fn_name, encoder)?;
        bincode::Encode::encode(&self.settings, encoder)
    }
}

#[cfg(feature = "bincode")]
bincode::impl_borrow_decode!(CompiledCudaComplexEvaluator);
#[cfg(feature = "bincode")]
impl<Context> bincode::Decode<Context> for CompiledCudaComplexEvaluator {
    fn decode<D: bincode::de::Decoder<Context = Context>>(
        decoder: &mut D,
    ) -> Result<Self, bincode::error::DecodeError> {
        let file: PathBuf = bincode::Decode::decode(decoder)?;
        let fn_name: String = bincode::Decode::decode(decoder)?;
        let settings: CudaLoadSettings = bincode::Decode::decode(decoder)?;
        CompiledCudaComplexEvaluator::load(&file, &fn_name, settings)
            .map_err(|e| bincode::error::DecodeError::OtherString(e))
    }
}

/// Efficient evaluator using CUDA for compiled real-valued functions.
pub struct CompiledCudaRealEvaluator {
    path: PathBuf,
    fn_name: String,
    library: LibraryCudaRealf64,
    settings: CudaLoadSettings,
    data: *const CudaEvaluationData,
}

impl EvaluatorLoader<CudaRealf64> for CompiledCudaRealEvaluator {
    fn load(path: impl AsRef<Path>, function_name: &str) -> Result<Self, String> {
        CompiledCudaRealEvaluator::load_with_settings(
            path,
            function_name,
            CudaLoadSettings::default(),
        )
    }

    fn load_with_settings(
        path: impl AsRef<Path>,
        function_name: &str,
        settings: CudaLoadSettings,
    ) -> Result<Self, String> {
        CompiledCudaRealEvaluator::load(path, function_name, settings)
    }
}

impl BatchEvaluator<f64> for CompiledCudaRealEvaluator {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[f64],
        out: &mut [f64],
    ) -> Result<(), String> {
        if self.settings.number_of_evaluations != batch_size {
            return Err(format!(
                "Number of CUDA evaluations {} does not equal batch size {}",
                self.settings.number_of_evaluations, batch_size
            ));
        }

        self.evaluate(params, out)
    }
}

impl CompiledCudaRealEvaluator {
    pub fn load_new_function(
        &self,
        function_name: &str,
    ) -> Result<CompiledCudaRealEvaluator, String> {
        let library = LibraryCudaRealf64::try_new(self.library.borrow_owner().clone(), |lib| {
            EvaluatorFunctionsCudaRealf64::new(lib, function_name)
        })?;
        let data = unsafe {
            let data = (library.borrow_dependent().init_data)(
                self.settings.number_of_evaluations,
                self.settings.block_size,
            );
            (*data).check_for_error()?;
            data
        };

        Ok(CompiledCudaRealEvaluator {
            path: self.path.clone(),
            fn_name: function_name.to_string(),
            library,
            settings: self.settings.clone(),
            data,
        })
    }

    pub fn load(
        path: impl AsRef<Path>,
        function_name: &str,
        settings: CudaLoadSettings,
    ) -> Result<CompiledCudaRealEvaluator, String> {
        unsafe {
            let lib = match libloading::Library::new(path.as_ref()) {
                Ok(lib) => lib,
                Err(_) => libloading::Library::new(PathBuf::new().join("./").join(&path))
                    .map_err(|e| e.to_string())?,
            };
            let library = LibraryCudaRealf64::try_new(std::sync::Arc::new(lib), |lib| {
                EvaluatorFunctionsCudaRealf64::new(lib, function_name)
            })?;

            let data = (library.borrow_dependent().init_data)(
                settings.number_of_evaluations,
                settings.block_size,
            );
            (*data).check_for_error()?;

            Ok(CompiledCudaRealEvaluator {
                path: path.as_ref().to_path_buf(),
                fn_name: function_name.to_string(),
                library,
                settings,
                data,
            })
        }
    }

    /// Evaluate the compiled code with double-precision floating point numbers.
    /// The `args` must be of length `number_of_evaluations * input`, where `input` is the number of inputs to the function.
    /// The `out` must be of length `number_of_evaluations * output`,
    /// where `output` is the number of outputs of the function.
    #[inline(always)]
    pub fn evaluate(&mut self, args: &[f64], out: &mut [f64]) -> Result<(), String> {
        unsafe {
            if args.len() != (*self.data).in_dimension * (*self.data).n {
                return Err(format!(
                    "CUDA args length (={}) does not match the expected input dimension (={}*{}).",
                    args.len(),
                    (*self.data).in_dimension,
                    (*self.data).n
                ));
            }
            if out.len() != (*self.data).out_dimension * (*self.data).n {
                return Err(format!(
                    "CUDA out length (={}) does not match the expected output dimension (={}*{}).",
                    out.len(),
                    (*self.data).out_dimension,
                    (*self.data).n
                ));
            }
            (self.library.borrow_dependent().eval)(args.as_ptr(), out.as_mut_ptr(), self.data);
            (*self.data).check_for_error()?;
        }
        Ok(())
    }
}

/// Efficient evaluator using CUDA for compiled complex-valued functions.
pub struct CompiledCudaComplexEvaluator {
    path: PathBuf,
    fn_name: String,
    library: LibraryCudaComplexf64,
    settings: CudaLoadSettings,
    data: *const CudaEvaluationData,
}

impl EvaluatorLoader<CudaComplexf64> for CompiledCudaComplexEvaluator {
    fn load(path: impl AsRef<Path>, function_name: &str) -> Result<Self, String> {
        CompiledCudaComplexEvaluator::load_with_settings(
            path,
            function_name,
            CudaLoadSettings::default(),
        )
    }

    fn load_with_settings(
        path: impl AsRef<Path>,
        function_name: &str,
        settings: CudaLoadSettings,
    ) -> Result<Self, String> {
        CompiledCudaComplexEvaluator::load(path, function_name, settings)
    }
}

impl BatchEvaluator<Complex<f64>> for CompiledCudaComplexEvaluator {
    fn evaluate_batch(
        &mut self,
        batch_size: usize,
        params: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> Result<(), String> {
        if self.settings.number_of_evaluations != batch_size {
            return Err(format!(
                "Number of CUDA evaluations {} does not equal batch size {}",
                self.settings.number_of_evaluations, batch_size
            ));
        }

        self.evaluate(params, out)
    }
}

impl CompiledCudaComplexEvaluator {
    pub fn load_new_function(
        &self,
        function_name: &str,
    ) -> Result<CompiledCudaComplexEvaluator, String> {
        let library = LibraryCudaComplexf64::try_new(self.library.borrow_owner().clone(), |lib| {
            EvaluatorFunctionsCudaComplexf64::new(lib, function_name)
        })?;

        let data = unsafe {
            let data = (library.borrow_dependent().init_data)(
                self.settings.number_of_evaluations,
                self.settings.block_size,
            );
            (*data).check_for_error()?;
            data
        };
        Ok(CompiledCudaComplexEvaluator {
            path: self.path.clone(),
            fn_name: function_name.to_string(),
            library,
            settings: self.settings.clone(),
            data,
        })
    }

    pub fn load(
        path: impl AsRef<Path>,
        function_name: &str,
        settings: CudaLoadSettings,
    ) -> Result<CompiledCudaComplexEvaluator, String> {
        unsafe {
            let lib = match libloading::Library::new(path.as_ref()) {
                Ok(lib) => lib,
                Err(_) => libloading::Library::new(PathBuf::new().join("./").join(&path))
                    .map_err(|e| e.to_string())?,
            };
            let library = LibraryCudaComplexf64::try_new(std::sync::Arc::new(lib), |lib| {
                EvaluatorFunctionsCudaComplexf64::new(lib, function_name)
            })?;

            let data = (library.borrow_dependent().init_data)(
                settings.number_of_evaluations,
                settings.block_size,
            );
            (*data).check_for_error()?;

            Ok(CompiledCudaComplexEvaluator {
                path: path.as_ref().to_path_buf(),
                fn_name: function_name.to_string(),
                library,
                settings,
                data,
            })
        }
    }

    /// Evaluate the compiled code with complex numbers.
    /// The `args` must be of length `number_of_evaluations * input`, where `input` is the number of inputs to the function.
    /// The `out` must be of length `number_of_evaluations * output`,
    /// where `output` is the number of outputs of the function.
    #[inline(always)]
    pub fn evaluate(
        &mut self,
        args: &[Complex<f64>],
        out: &mut [Complex<f64>],
    ) -> Result<(), String> {
        unsafe {
            if args.len() != (*self.data).in_dimension * (*self.data).n {
                return Err(format!(
                    "CUDA args length (={}) does not match the expected input dimension (={}*{}).",
                    args.len(),
                    (*self.data).in_dimension,
                    (*self.data).n
                ));
            }
            if out.len() != (*self.data).out_dimension * (*self.data).n {
                return Err(format!(
                    "CUDA out length (={}) does not match the expected output dimension (={}*{}).",
                    out.len(),
                    (*self.data).out_dimension,
                    (*self.data).n
                ));
            }
            (self.library.borrow_dependent().eval)(args.as_ptr(), out.as_mut_ptr(), self.data);
            (*self.data).check_for_error()?;
        }
        Ok(())
    }
}

unsafe impl Send for CompiledCudaRealEvaluator {}
unsafe impl Send for CompiledCudaComplexEvaluator {}
unsafe impl Sync for CompiledCudaRealEvaluator {}
unsafe impl Sync for CompiledCudaComplexEvaluator {}

impl std::fmt::Debug for CompiledCudaRealEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompiledCudaRealEvaluator({})", self.fn_name)
    }
}

impl Drop for CompiledCudaRealEvaluator {
    fn drop(&mut self) {
        unsafe {
            let result = (self.library.borrow_dependent().destroy_data)(self.data);
            if result != 0 {
                error!("Warning: failed to free CUDA memory: {}", result);
            }
        }
    }
}

impl Clone for CompiledCudaRealEvaluator {
    fn clone(&self) -> Self {
        self.load_new_function(&self.fn_name).unwrap()
    }
}

impl std::fmt::Debug for CompiledCudaComplexEvaluator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CompiledCudaComplexEvaluator({})", self.fn_name)
    }
}

impl Drop for CompiledCudaComplexEvaluator {
    fn drop(&mut self) {
        unsafe {
            let result = (self.library.borrow_dependent().destroy_data)(self.data);
            if result != 0 {
                error!("Warning: failed to free CUDA memory: {}", result);
            }
        }
    }
}

impl Clone for CompiledCudaComplexEvaluator {
    fn clone(&self) -> Self {
        self.load_new_function(&self.fn_name).unwrap()
    }
}

/// Options for compiling exported code.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Clone)]
pub struct CompileOptions {
    pub optimization_level: usize,
    pub fast_math: bool,
    pub unsafe_math: bool,
    /// Compile for the native architecture.
    pub native: bool,
    pub compiler: String,
    /// Arguments for the compiler call. Arguments with spaces
    /// must be split into a separate strings.
    ///
    /// For CUDA, the argument `-x cu` is required.
    pub args: Vec<String>,
}

impl Default for CompileOptions {
    /// Default compile options.
    fn default() -> Self {
        CompileOptions {
            optimization_level: 3,
            fast_math: true,
            unsafe_math: true,
            native: true,
            compiler: "g++".to_string(),
            args: vec![],
        }
    }
}

impl CompileOptions {
    /// Set the compiler to `nvcc`.
    pub fn cuda() -> Self {
        CompileOptions {
            optimization_level: 3,
            fast_math: false,
            unsafe_math: false,
            native: false,
            compiler: "nvcc".to_string(),
            args: vec![],
        }
    }
}

impl ToString for CompileOptions {
    /// Convert the compilation options to the string that would be used
    /// in the compiler call.
    fn to_string(&self) -> String {
        let mut s = self.compiler.clone();

        s += &format!(" -shared -O{}", self.optimization_level);

        let nvcc = self.compiler.contains("nvcc");

        if !nvcc {
            s += " -fPIC";
        } else {
            // order is important here for nvcc
            s += " -Xcompiler -fPIC -x cu";
        }

        if self.fast_math && !nvcc {
            s += " -ffast-math";
        }
        if self.unsafe_math && !nvcc {
            s += " -funsafe-math-optimizations";
        }
        if self.native && !nvcc {
            s += " -march=native";
        }
        for arg in &self.args {
            s += " ";
            s += arg;
        }
        s
    }
}

impl<T: CompiledNumber> ExportedCode<T> {
    /// Create a new exported code object from a source file and function name.
    pub fn new(source_path: impl AsRef<Path>, function_name: String) -> Self {
        ExportedCode {
            path: source_path.as_ref().to_path_buf(),
            function_name,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Compile the code to a shared library.
    ///
    /// For CUDA, you may have to specify `-code=sm_XY` for your architecture `XY` in the compiler flags to prevent a potentially long
    /// JIT compilation upon the first evaluation.
    pub fn compile(
        &self,
        out: impl AsRef<Path>,
        options: CompileOptions,
    ) -> Result<CompiledCode<T>, std::io::Error> {
        let mut builder = std::process::Command::new(&options.compiler);
        builder
            .arg("-shared")
            .arg(format!("-O{}", options.optimization_level));

        if !options.compiler.contains("nvcc") {
            builder.arg("-fPIC");
        } else {
            // order is important here for nvcc
            builder.arg("-Xcompiler");
            builder.arg("-fPIC");
            builder.arg("-x");
            builder.arg("cu");
        }
        if options.fast_math && !options.compiler.contains("nvcc") {
            builder.arg("-ffast-math");
        }
        if options.unsafe_math && !options.compiler.contains("nvcc") {
            builder.arg("-funsafe-math-optimizations");
        }

        if options.native && !options.compiler.contains("nvcc") {
            builder.arg("-march=native");
        }

        for c in &options.args {
            builder.arg(c);
        }

        let r = builder
            .arg("-o")
            .arg(out.as_ref())
            .arg(&self.path)
            .output()?;

        if !r.status.success() {
            return Err(std::io::Error::other(format!(
                "Could not compile code: {} {}\n{}",
                builder.get_program().to_string_lossy(),
                builder
                    .get_args()
                    .map(|arg| arg.to_string_lossy().to_string())
                    .collect::<Vec<_>>()
                    .join(" "),
                String::from_utf8_lossy(&r.stderr)
            )));
        }

        Ok(CompiledCode {
            path: out.as_ref().to_path_buf(),
            function_name: self.function_name.clone(),
            _phantom: std::marker::PhantomData,
        })
    }
}

/// The inline assembly mode used to generate fast
/// assembly instructions for mathematical operations.
/// Set to `None` to disable inline assembly.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum InlineASM {
    /// Use instructions suitable for x86_64 machines.
    X64,
    /// Use instructions suitable for x86_64 machines with AVX2 support.
    AVX2,
    /// Use instructions suitable for ARM64 machines.
    AArch64,
    /// Do not generate inline assembly.
    None,
}

impl Default for InlineASM {
    /// Set the assembly mode suitable for the current
    /// architecture.
    fn default() -> Self {
        if cfg!(target_arch = "x86_64") {
            InlineASM::X64
        } else if cfg!(target_arch = "aarch64") {
            InlineASM::AArch64
        } else {
            InlineASM::None
        }
    }
}

impl<'a> AtomView<'a> {
    /// Convert nested expressions to a tree.
    pub fn to_evaluation_tree(
        &self,
        fn_map: &FunctionMap<Complex<Rational>>,
        params: &[Atom],
    ) -> Result<EvalTree<Complex<Rational>>, String> {
        Self::to_eval_tree_multiple(std::slice::from_ref(self), fn_map, params)
    }

    /// Convert nested expressions to a tree.
    pub fn to_eval_tree_multiple<A: AtomCore>(
        exprs: &[A],
        fn_map: &FunctionMap<Complex<Rational>>,
        params: &[Atom],
    ) -> Result<EvalTree<Complex<Rational>>, String> {
        let mut funcs = vec![];
        let mut func_id_to_index = HashMap::default();

        let tree = exprs
            .iter()
            .map(|t| {
                t.as_atom_view()
                    .to_eval_tree_impl(fn_map, params, &[], &mut func_id_to_index, &mut funcs)
                    .map(|x| x.rehashed(true))
            })
            .collect::<Result<_, _>>()?;

        let mut external_fns: Vec<_> = fn_map
            .external_fn
            .values()
            .map(|x| {
                let ConstOrExpr::External(e, name) = x else {
                    panic!("Expected external function");
                };

                (*e, name.clone())
            })
            .collect();
        external_fns.sort_by_key(|x| x.0);

        Ok(EvalTree {
            expressions: SplitExpression {
                tree,
                subexpressions: vec![],
            },
            functions: funcs,
            external_functions: external_fns.into_iter().map(|x| x.1).collect(),
            param_count: params.len(),
        })
    }

    fn to_eval_tree_impl(
        &self,
        fn_map: &FunctionMap<Complex<Rational>>,
        params: &[Atom],
        args: &[Indeterminate],
        fn_id_map: &mut HashMap<usize, usize>,
        funcs: &mut Vec<(
            String,
            Vec<Indeterminate>,
            SplitExpression<Complex<Rational>>,
        )>,
    ) -> Result<Expression<Complex<Rational>>, String> {
        if matches!(self, AtomView::Var(_) | AtomView::Fun(_)) {
            if let Some(p) = args.iter().position(|s| *self == s.as_view()) {
                return Ok(Expression::ReadArg(0, p));
            }

            if let Some(p) = params.iter().position(|a| a.as_view() == *self) {
                return Ok(Expression::Parameter(0, p));
            }
        }

        if let Some(c) = fn_map.get_constant(*self) {
            return Ok(Expression::Const(0, Box::new(c.clone())));
        }

        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, d, ni, di) => Ok(Expression::Const(
                    0,
                    Box::new(Complex::new(
                        Rational::from((n, d)),
                        Rational::from((ni, di)),
                    )),
                )),
                CoefficientView::Large(l, i) => Ok(Expression::Const(
                    0,
                    Box::new(Complex::new(l.to_rat(), i.to_rat())),
                )),
                CoefficientView::Float(r, i) => {
                    // TODO: converting back to rational is slow
                    Ok(Expression::Const(
                        0,
                        Box::new(Complex::new(
                            r.to_float().to_rational(),
                            i.to_float().to_rational(),
                        )),
                    ))
                }
                CoefficientView::Indeterminate => {
                    panic!("Cannot convert indeterminate")
                }
                CoefficientView::Infinity(_) => {
                    panic!("Cannot convert infinity")
                }
                CoefficientView::FiniteField(_, _) => {
                    Err("Finite field not yet supported for evaluation".to_string())
                }
                CoefficientView::RationalPolynomial(_) => Err(
                    "Rational polynomial coefficient not yet supported for evaluation".to_string(),
                ),
            },
            AtomView::Var(v) => Err(format!(
                "Variable {} not in constant map",
                v.get_symbol().get_name()
            )),
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [
                    Symbol::EXP_ID,
                    Symbol::LOG_ID,
                    Symbol::SIN_ID,
                    Symbol::COS_ID,
                    Symbol::SQRT_ID,
                    Symbol::ABS_ID,
                    Symbol::CONJ_ID,
                ]
                .contains(&name.get_id())
                {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.to_eval_tree_impl(fn_map, params, args, fn_id_map, funcs)?;

                    return Ok(Expression::BuiltinFun(
                        0,
                        BuiltinSymbol(f.get_symbol()),
                        Box::new(arg_eval),
                    ));
                }

                let fun = if name == Symbol::IF {
                    &ConstOrExpr::Condition
                } else if let Some(fun) = fn_map.get(*self) {
                    fun
                } else {
                    return Err(format!("Undefined function {}", self.to_plain_string()));
                };

                match fun {
                    ConstOrExpr::Const(t) => Ok(Expression::Const(0, Box::new(t.clone()))),
                    ConstOrExpr::External(e, _name) => {
                        let eval_args = f
                            .iter()
                            .map(|arg| {
                                arg.to_eval_tree_impl(fn_map, params, args, fn_id_map, funcs)
                            })
                            .collect::<Result<_, _>>()?;
                        Ok(Expression::ExternalFun(0, *e as u32, eval_args))
                    }
                    ConstOrExpr::Expr(Expr {
                        id,
                        name,
                        tag_len,
                        args: arg_spec,
                        body: e,
                    }) => {
                        if f.get_nargs() != arg_spec.len() + tag_len {
                            return Err(format!(
                                "Function {} called with wrong number of arguments: {} vs {}",
                                f.get_symbol().get_name(),
                                f.get_nargs(),
                                arg_spec.len() + tag_len
                            ));
                        }

                        let eval_args = f
                            .iter()
                            .skip(*tag_len)
                            .map(|arg| {
                                arg.to_eval_tree_impl(fn_map, params, args, fn_id_map, funcs)
                            })
                            .collect::<Result<_, _>>()?;

                        if let Some(pos) = fn_id_map.get(id) {
                            Ok(Expression::Eval(0, *pos as u32, eval_args))
                        } else {
                            let r = e
                                .as_view()
                                .to_eval_tree_impl(fn_map, params, arg_spec, fn_id_map, funcs)?;
                            funcs.push((
                                name.clone(),
                                arg_spec.clone(),
                                SplitExpression {
                                    tree: vec![r.clone()],
                                    subexpressions: vec![],
                                },
                            ));
                            fn_id_map.insert(*id, funcs.len() - 1);
                            Ok(Expression::Eval(0, funcs.len() as u32 - 1, eval_args))
                        }
                    }
                    ConstOrExpr::Condition => {
                        if f.get_nargs() != 3 {
                            return Err(format!(
                                "Condition function called with wrong number of arguments: {} vs 3",
                                f.get_nargs(),
                            ));
                        }

                        let mut arg_iter = f.iter();

                        let cond_eval = arg_iter
                            .next()
                            .unwrap()
                            .to_eval_tree_impl(fn_map, params, args, fn_id_map, funcs)?;

                        if let Expression::Const(0, c) = &cond_eval {
                            if !c.is_zero() {
                                let t_eval = arg_iter
                                    .next()
                                    .unwrap()
                                    .to_eval_tree_impl(fn_map, params, args, fn_id_map, funcs)?;
                                return Ok(t_eval);
                            } else {
                                let _ = arg_iter.next().unwrap();
                                let f_eval = arg_iter
                                    .next()
                                    .unwrap()
                                    .to_eval_tree_impl(fn_map, params, args, fn_id_map, funcs)?;
                                return Ok(f_eval);
                            }
                        }

                        let t_eval = arg_iter
                            .next()
                            .unwrap()
                            .to_eval_tree_impl(fn_map, params, args, fn_id_map, funcs)?;
                        let f_eval = arg_iter
                            .next()
                            .unwrap()
                            .to_eval_tree_impl(fn_map, params, args, fn_id_map, funcs)?;

                        Ok(Expression::IfElse(0, Box::new((cond_eval, t_eval, f_eval))))
                    }
                }
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let b_eval = b.to_eval_tree_impl(fn_map, params, args, fn_id_map, funcs)?;

                if let AtomView::Num(n) = e
                    && let CoefficientView::Natural(num, den, num_i, _den_i) = n.get_coeff_view()
                    && den == 1
                    && num_i == 0
                {
                    return Ok(Expression::Pow(0, Box::new((b_eval.clone(), num))));
                }

                let e_eval = e.to_eval_tree_impl(fn_map, params, args, fn_id_map, funcs)?;
                Ok(Expression::Powf(0, Box::new((b_eval, e_eval))))
            }
            AtomView::Mul(m) => {
                let mut muls = vec![];
                for arg in m.iter() {
                    let a = arg.to_eval_tree_impl(fn_map, params, args, fn_id_map, funcs)?;
                    if let Expression::Mul(0, m) = a {
                        muls.extend(m);
                    } else {
                        muls.push(a);
                    }
                }

                muls.sort();

                Ok(Expression::Mul(0, muls))
            }
            AtomView::Add(a) => {
                let mut adds = vec![];
                for arg in a.iter() {
                    adds.push(arg.to_eval_tree_impl(fn_map, params, args, fn_id_map, funcs)?);
                }

                adds.sort();

                Ok(Expression::Add(0, adds))
            }
        }
    }

    /// Evaluate an expression using a constant map and a function map.
    /// The constant map can map any literal expression to a value, for example
    /// a variable or a function with fixed arguments.
    ///
    /// All variables and all user functions in the expression must occur in the map.
    pub(crate) fn evaluate<A: AtomCore + KeyLookup, T: Real, F: Fn(&Rational) -> T + Copy>(
        &self,
        coeff_map: F,
        const_map: &HashMap<A, T>,
        function_map: &HashMap<Symbol, EvaluationFn<A, T>>,
    ) -> Result<T, String> {
        let mut cache = HashMap::default();
        self.evaluate_impl(coeff_map, const_map, function_map, &mut cache)
    }

    fn evaluate_impl<A: AtomCore + KeyLookup, T: Real, F: Fn(&Rational) -> T + Copy>(
        &self,
        coeff_map: F,
        const_map: &HashMap<A, T>,
        function_map: &HashMap<Symbol, EvaluationFn<A, T>>,
        cache: &mut HashMap<AtomView<'a>, T>,
    ) -> Result<T, String> {
        if let Some(c) = const_map.get(self.get_data()) {
            return Ok(c.clone());
        }

        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, d, ni, di) => {
                    if ni == 0 {
                        Ok(coeff_map(&Rational::from_int_unchecked(n, d)))
                    } else {
                        let num = coeff_map(&Rational::from_int_unchecked(n, d));
                        Ok(coeff_map(&Rational::from_int_unchecked(ni, di))
                            * num.i().ok_or_else(|| {
                                "Numerical type does not support imaginary unit".to_string()
                            })?
                            + num)
                    }
                }
                CoefficientView::Large(l, i) => {
                    if i.is_zero() {
                        Ok(coeff_map(&l.to_rat()))
                    } else {
                        let num = coeff_map(&l.to_rat());
                        Ok(coeff_map(&i.to_rat())
                            * num.i().ok_or_else(|| {
                                "Numerical type does not support imaginary unit".to_string()
                            })?
                            + num)
                    }
                }
                CoefficientView::Float(r, i) => {
                    // TODO: converting back to rational is slow
                    let rm = coeff_map(&r.to_float().to_rational());
                    if i.is_zero() {
                        Ok(rm)
                    } else {
                        Ok(coeff_map(&i.to_float().to_rational())
                            * rm.i().ok_or_else(|| {
                                "Numerical type does not support imaginary unit".to_string()
                            })?
                            + rm)
                    }
                }
                CoefficientView::Indeterminate => Err("Cannot evaluate indeterminate".to_string()),
                CoefficientView::Infinity(_) => Err("Cannot evaluate infinity".to_string()),
                CoefficientView::FiniteField(_, _) => {
                    Err("Finite field not yet supported for evaluation".to_string())
                }
                CoefficientView::RationalPolynomial(_) => Err(
                    "Rational polynomial coefficient not yet supported for evaluation".to_string(),
                ),
            },
            AtomView::Var(v) => {
                let s = v.get_symbol();
                match s.get_id() {
                    Symbol::E_ID => Ok(coeff_map(&1.into()).e()),
                    Symbol::PI_ID => Ok(coeff_map(&1.into()).pi()),
                    _ => {
                        if let Some(fun) = function_map.get(&s) {
                            if let Some(eval) = cache.get(self) {
                                return Ok(eval.clone());
                            }

                            let eval = fun.get()(&[], const_map, function_map, cache);
                            cache.insert(*self, eval.clone());
                            Ok(eval)
                        } else {
                            Err(format!(
                                "Variable {} not in constant map or function map",
                                v.get_symbol().get_name()
                            ))
                        }
                    }
                }
            }
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [
                    Symbol::EXP_ID,
                    Symbol::LOG_ID,
                    Symbol::SIN_ID,
                    Symbol::COS_ID,
                    Symbol::SQRT_ID,
                    Symbol::ABS_ID,
                    Symbol::CONJ_ID,
                ]
                .contains(&name.get_id())
                {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.evaluate_impl(coeff_map, const_map, function_map, cache)?;

                    return Ok(match f.get_symbol_id() {
                        Symbol::EXP_ID => arg_eval.exp(),
                        Symbol::LOG_ID => arg_eval.log(),
                        Symbol::SIN_ID => arg_eval.sin(),
                        Symbol::COS_ID => arg_eval.cos(),
                        Symbol::SQRT_ID => arg_eval.sqrt(),
                        Symbol::ABS_ID => arg_eval.norm(),
                        Symbol::CONJ_ID => arg_eval.conj(),
                        _ => unreachable!(),
                    });
                }

                if name == Symbol::IF {
                    if f.get_nargs() != 3 {
                        return Err(format!(
                            "Condition function called with wrong number of arguments: {} vs 3",
                            f.get_nargs(),
                        ));
                    }

                    let mut arg_iter = f.iter();

                    let cond_eval = arg_iter.next().unwrap().evaluate_impl(
                        coeff_map,
                        const_map,
                        function_map,
                        cache,
                    )?;

                    if !cond_eval.is_fully_zero() {
                        let t_eval = arg_iter.next().unwrap().evaluate_impl(
                            coeff_map,
                            const_map,
                            function_map,
                            cache,
                        )?;
                        return Ok(t_eval);
                    } else {
                        let _ = arg_iter.next().unwrap();
                        let f_eval = arg_iter.next().unwrap().evaluate_impl(
                            coeff_map,
                            const_map,
                            function_map,
                            cache,
                        )?;
                        return Ok(f_eval);
                    }
                }

                if let Some(eval) = cache.get(self) {
                    return Ok(eval.clone());
                }

                let mut args = Vec::with_capacity(f.get_nargs());
                for arg in f {
                    args.push(arg.evaluate_impl(coeff_map, const_map, function_map, cache)?);
                }

                let Some(fun) = function_map.get(&f.get_symbol()) else {
                    Err(format!("Missing function {}", f.get_symbol().get_name()))?
                };
                let eval = fun.get()(&args, const_map, function_map, cache);

                cache.insert(*self, eval.clone());
                Ok(eval)
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let b_eval = b.evaluate_impl(coeff_map, const_map, function_map, cache)?;

                if let AtomView::Num(n) = e
                    && let CoefficientView::Natural(num, den, ni, _di) = n.get_coeff_view()
                    && den == 1
                    && ni == 0
                {
                    if num >= 0 {
                        return Ok(b_eval.pow(num as u64));
                    } else {
                        return Ok(b_eval.pow(num.unsigned_abs()).inv());
                    }
                }

                let e_eval = e.evaluate_impl(coeff_map, const_map, function_map, cache)?;
                Ok(b_eval.powf(&e_eval))
            }
            AtomView::Mul(m) => {
                let mut it = m.iter();
                let mut r =
                    it.next()
                        .unwrap()
                        .evaluate_impl(coeff_map, const_map, function_map, cache)?;
                for arg in it {
                    r *= arg.evaluate_impl(coeff_map, const_map, function_map, cache)?;
                }
                Ok(r)
            }
            AtomView::Add(a) => {
                let mut it = a.iter();
                let mut r =
                    it.next()
                        .unwrap()
                        .evaluate_impl(coeff_map, const_map, function_map, cache)?;
                for arg in it {
                    r += arg.evaluate_impl(coeff_map, const_map, function_map, cache)?;
                }
                Ok(r)
            }
        }
    }

    /// Check if the expression could be 0, using (potentially) numerical sampling with
    /// a given tolerance and number of iterations.
    pub fn zero_test(&self, iterations: usize, tolerance: f64) -> ConditionResult {
        match self {
            AtomView::Num(num_view) => {
                if num_view.is_zero() {
                    ConditionResult::True
                } else {
                    ConditionResult::False
                }
            }
            AtomView::Var(_) => ConditionResult::False,
            AtomView::Fun(_) => ConditionResult::False,
            AtomView::Pow(p) => p.get_base().zero_test(iterations, tolerance),
            AtomView::Mul(mul_view) => {
                let mut is_zero = ConditionResult::False;
                for arg in mul_view {
                    match arg.zero_test(iterations, tolerance) {
                        ConditionResult::True => return ConditionResult::True,
                        ConditionResult::False => {}
                        ConditionResult::Inconclusive => {
                            is_zero = ConditionResult::Inconclusive;
                        }
                    }
                }

                is_zero
            }
            AtomView::Add(_) => self.zero_test_impl(iterations, tolerance),
        }
    }

    fn zero_test_impl(&self, iterations: usize, tolerance: f64) -> ConditionResult {
        // collect all variables and functions and fill in random variables

        let mut rng = MonteCarloRng::new(0, 0);

        if !self.is_real() {
            let mut vars: HashMap<_, _> = self
                .get_all_indeterminates(true)
                .into_iter()
                .filter_map(|x| {
                    let s = x.get_symbol().unwrap();
                    if !State::is_builtin(s) || s == Symbol::DERIVATIVE {
                        Some((x, Complex::new(0f64.into(), 0f64.into())))
                    } else {
                        None
                    }
                })
                .collect();

            for _ in 0..iterations {
                for x in vars.values_mut() {
                    *x = x.sample_unit(&mut rng);
                }

                let r = self
                    .evaluate(
                        |x| {
                            Complex::new(
                                ErrorPropagatingFloat::new(
                                    0f64.from_rational(x),
                                    -0f64.get_epsilon().log10(),
                                ),
                                ErrorPropagatingFloat::new(
                                    0f64.zero(),
                                    -0f64.get_epsilon().log10(),
                                ),
                            )
                        },
                        &vars,
                        &HashMap::default(),
                    )
                    .unwrap();

                let res_re = r.re.get_num().to_f64();
                let res_im = r.im.get_num().to_f64();
                if res_re.is_finite()
                    && (res_re - r.re.get_absolute_error() > 0.
                        || res_re + r.re.get_absolute_error() < 0.)
                    || res_im.is_finite()
                        && (res_im - r.im.get_absolute_error() > 0.
                            || res_im + r.im.get_absolute_error() < 0.)
                {
                    return ConditionResult::False;
                }

                if vars.is_empty() && r.re.get_absolute_error() < tolerance {
                    return ConditionResult::True;
                }
            }

            ConditionResult::Inconclusive
        } else {
            let mut vars: HashMap<_, ErrorPropagatingFloat<f64>> = self
                .get_all_indeterminates(true)
                .into_iter()
                .filter_map(|x| {
                    let s = x.get_symbol().unwrap();
                    if !State::is_builtin(s) || s == Symbol::DERIVATIVE {
                        Some((x, 0f64.into()))
                    } else {
                        None
                    }
                })
                .collect();

            for _ in 0..iterations {
                for x in vars.values_mut() {
                    *x = x.sample_unit(&mut rng);
                }

                let r = self
                    .evaluate(
                        |x| {
                            ErrorPropagatingFloat::new(
                                0f64.from_rational(x),
                                -0f64.get_epsilon().log10(),
                            )
                        },
                        &vars,
                        &HashMap::default(),
                    )
                    .unwrap();

                let res = r.get_num().to_f64();

                // trust the error when the relative error is less than 20%
                if res != 0.
                    && res.is_finite()
                    && r.get_absolute_error() / res.abs() < 0.2
                    && (res - r.get_absolute_error() > 0. || res + r.get_absolute_error() < 0.)
                {
                    return ConditionResult::False;
                }

                if vars.is_empty() && r.get_absolute_error() < tolerance {
                    return ConditionResult::True;
                }
            }

            ConditionResult::Inconclusive
        }
    }
}

#[cfg(test)]
mod test {
    use ahash::HashMap;
    use numerica::domains::dual::HyperDual;

    use crate::{
        atom::{Atom, AtomCore},
        create_hyperdual_from_components,
        domains::{
            float::{Complex, Float, FloatLike},
            rational::Rational,
        },
        evaluate::{Dualizer, EvaluationFn, ExternalFunction, FunctionMap, OptimizationSettings},
        id::ConditionResult,
        parse, symbol,
    };

    #[test]
    fn evaluate() {
        let x = symbol!("v1");
        let f = symbol!("f1");
        let g = symbol!("f2");
        let p0 = parse!("v2(0)");
        let a = parse!("v1*cos(v1) + f1(v1, 1)^2 + f2(f2(v1)) + v2(0)");

        let v = Atom::var(x);

        let mut const_map = HashMap::default();
        let mut fn_map: HashMap<_, EvaluationFn<_, _>> = HashMap::default();

        // x = 6 and p(0) = 7

        const_map.insert(v.as_view(), 6.); // .as_view()
        const_map.insert(p0.as_view(), 7.);

        // f(x, y) = x^2 + y
        fn_map.insert(
            f,
            EvaluationFn::new(Box::new(|args: &[f64], _, _, _| {
                args[0] * args[0] + args[1]
            })),
        );

        // g(x) = f(x, 3)
        fn_map.insert(
            g,
            EvaluationFn::new(Box::new(move |args: &[f64], var_map, fn_map, cache| {
                fn_map.get(&f).unwrap().get()(&[args[0], 3.], var_map, fn_map, cache)
            })),
        );

        let r = a.evaluate(|x| x.into(), &const_map, &fn_map).unwrap();
        assert_eq!(r, 2905.761021719902);
    }

    #[test]
    fn arb_prec() {
        let x = symbol!("v1");
        let a = parse!("128731/12893721893721 + v1");

        let mut const_map = HashMap::default();

        let v = Atom::var(x);
        const_map.insert(v.as_view(), Float::with_val(200, 6));

        let r = a
            .evaluate(
                |r| r.to_multi_prec_float(200),
                &const_map,
                &HashMap::default(),
            )
            .unwrap();

        assert_eq!(
            format!("{r}"),
            "6.00000000998400625211945786243908951675582851493871969158108"
        );
    }

    #[test]
    fn nested() {
        let e1 = parse!("x + pi + cos(x) + f(g(x+1),h(x*2)) + p(1,x)");
        let e2 = parse!("x + h(x*2) + cos(x)");
        let f = parse!("y^2 + z^2*y^2");
        let g = parse!("i(y+7)+x*i(y+7)*(y-1)");
        let h = parse!("y*(1+x*(1+x^2)) + y^2*(1+x*(1+x^2))^2 + 3*(1+x^2)");
        let i = parse!("y - 1");
        let p1 = parse!("3*z^3 + 4*z^2 + 6*z +8");

        let mut fn_map = FunctionMap::new();

        fn_map.add_constant(symbol!("pi").into(), Complex::from(Rational::from((22, 7))));
        fn_map
            .add_tagged_function(
                symbol!("p"),
                vec![Atom::num(1)],
                "p1".to_string(),
                vec![symbol!("z")],
                p1,
            )
            .unwrap();
        fn_map
            .add_function(
                symbol!("f"),
                "f".to_string(),
                vec![symbol!("y"), symbol!("z")],
                f,
            )
            .unwrap();
        fn_map
            .add_function(symbol!("g"), "g".to_string(), vec![symbol!("y")], g)
            .unwrap();
        fn_map
            .add_function(symbol!("h"), "h".to_string(), vec![symbol!("y")], h)
            .unwrap();
        fn_map
            .add_function(symbol!("i"), "i".to_string(), vec![symbol!("y")], i)
            .unwrap();

        let params = vec![parse!("x")];

        let evaluator =
            Atom::evaluator_multiple(&[e1, e2], &fn_map, &params, OptimizationSettings::default())
                .unwrap();

        let mut e_f64 = evaluator.map_coeff(&|x| x.clone().to_real().unwrap().into());
        let mut res = [0., 0.];
        e_f64.evaluate(&[1.1], &mut res);
        assert!((res[0] - 1622709.2254269677).abs() / 1622709.2254269677 < 1e-10);
    }

    #[test]
    fn zero_test() {
        let e = parse!(
            "(sin(v1)^2-sin(v1))(sin(v1)^2+sin(v1))^2 - (1/4 sin(2v1)^2-1/2 sin(2v1)cos(v1)-2 cos(v1)^2+1/2 sin(2v1)cos(v1)^3+3 cos(v1)^4-cos(v1)^6)"
        );
        assert_eq!(e.zero_test(10, f64::EPSILON), ConditionResult::Inconclusive);

        let e = parse!("x + (1+x)^2 + (x+2)*5");
        assert_eq!(e.zero_test(10, f64::EPSILON), ConditionResult::False);
    }

    #[test]
    fn branching() {
        let mut f = FunctionMap::new();
        f.add_conditional(symbol!("if")).unwrap();

        let tests = vec![
            ("if(y, x*x + z*z + x*z*z, x * x + 3)", 25., 12.),
            ("if(y+1, x*x + z*z + x*z*z, x * x + 3)", 12., 25.),
            ("if(y, x*x + z*z + x*z*z, 3)", 25., 3.),
            ("if(x + z, if(y, 1 + x, 1+x+y), 0)", 4., 4.),
            ("if(y, x * z, 0) + x * z", 12., 6.),
            ("if(y, x + 1, 2)*if(y+1, x + 1, 3)", 12., 8.),
            ("if(y, if(z, x + 1, 3)*if(z-2, x + 1, 4), 2)", 16., 2.),
        ];

        for (input, true_res, false_res) in tests {
            let mut eval = parse!(input)
                .evaluator(
                    &f,
                    &vec![crate::parse!("x"), crate::parse!("y"), crate::parse!("z")],
                    Default::default(),
                )
                .unwrap()
                .map_coeff(&|x| x.re.to_f64());

            let res = eval.evaluate_single(&[3., -1., 2.]);
            assert_eq!(res, true_res);
            let res = eval.evaluate_single(&[3., 0., 2.]);
            assert_eq!(res, false_res);
        }
    }

    #[test]
    fn vectorize_dual() {
        create_hyperdual_from_components!(
            Dual,
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [1, 1, 0],
                [1, 0, 1],
                [0, 1, 1],
                [1, 1, 1],
                [2, 0, 0]
            ]
        );

        let ev = parse!("sin(x+y)^2+cos(x+y)^2 - exp(sqrt(x)/sqrt(z)-1)")
            .evaluator(
                &FunctionMap::new(),
                &[parse!("x"), parse!("y"), parse!("z")],
                OptimizationSettings::default(),
            )
            .unwrap();

        let dual = Dualizer::new(Dual::<Complex<Rational>>::new_zero(), vec![]);
        let vec_ev = ev.vectorize(&dual, HashMap::default()).unwrap();

        let mut vec_f = vec_ev.map_coeff(&|x| x.re.to_f64());
        let mut dest = vec![0.; 9];
        vec_f.evaluate(
            &[
                2.0, 1.0, 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17.,
                2.0, 1.0, 2., 3., 4., 5., 6., 7., 8.,
            ],
            &mut dest,
        );

        assert!(dest.iter().all(|x| x.abs() < 1e-10));
    }

    #[test]
    fn vectorize_dual_with_external() {
        let dual = Dualizer::new(
            HyperDual::from_values(
                vec![vec![0], vec![1]],
                vec![Complex::<Rational>::new_zero(); 2],
            ),
            vec![],
        );

        let mut f = FunctionMap::new();
        f.add_external_function(symbol!("f"), "f".to_owned())
            .unwrap();
        let ev = parse!("f(x + 1)")
            .evaluator(&f, &[parse!("x")], OptimizationSettings::default())
            .unwrap();

        let mut vec_ext = HashMap::default();
        vec_ext.insert(("f".to_owned(), 0), "f0".to_owned());
        vec_ext.insert(("f".to_owned(), 1), "f1".to_owned());

        let vec_ev = ev
            .vectorize(&dual, vec_ext)
            .unwrap()
            .map_coeff(&|c| c.re.to_f64());

        let mut fns: HashMap<String, Box<dyn ExternalFunction<f64>>> = HashMap::default();
        fns.insert("f0".to_owned(), Box::new(|a: &[f64]| a[0]));
        fns.insert("f1".to_owned(), Box::new(|a: &[f64]| a[1]));

        let mut evr = vec_ev.with_external_functions(fns).unwrap();
        let mut out = vec![0.; 2];
        evr.evaluate(&[1., 2.], &mut out);
        assert_eq!(out, vec![2., 2.]);
    }

    #[test]
    fn jit_compile() {
        use crate::parse;
        let eval = parse!("x^2 * cos(x)")
            .evaluator(
                &FunctionMap::new(),
                &[parse!("x")],
                OptimizationSettings::default(),
            )
            .unwrap();

        let mut res = [0.; 1];
        let mut eval_re = eval.clone().map_coeff(&|x| x.re.to_f64());
        eval_re.evaluate(&[0.5], &mut res);

        let mut jit_eval_re = eval_re.jit_compile().unwrap();

        let mut jit_res = [0.; 1];
        jit_eval_re.evaluate(&[0.5], &mut jit_res);
        assert_eq!(res[0], jit_res[0]);

        let mut res = [Complex::new(0., 0.); 1];
        let mut eval_c = eval
            .clone()
            .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));
        eval_c.evaluate(&[Complex::new(0.5, 1.2)], &mut res);

        let mut jit_eval_c = eval.jit_compile::<Complex<f64>>().unwrap();
        let mut jit_res = [Complex::new(0., 0.); 1];
        jit_eval_c.evaluate(&[Complex::new(0.5, 1.2)], &mut jit_res);
        assert_eq!(res[0], jit_res[0]);
    }
}
