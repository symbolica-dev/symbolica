use std::{collections::HashMap as StdHashMap, hash::BuildHasher};

use crate::{
    atom::{AtomOrView, InlineVar},
    coefficient::ConvertToRing,
    state::Workspace,
};

use super::*;

/// Keep multiplication count as the primary objective, then minimize additions.
fn horner_cost(ops: OperationCount) -> (usize, usize) {
    (ops.multiplications, ops.additions)
}

/// Once search cancellation is requested, also stop optimization of the final
/// candidate even when the caller's callback only returns true once.
fn latched_horner_settings(
    settings: &OptimizationSettings,
) -> (OptimizationSettings, Arc<AtomicBool>) {
    let cancelled = Arc::new(AtomicBool::new(false));
    let mut search_settings = settings.clone();
    let requested_abort = settings.abort_check.clone();
    let abort_latch = cancelled.clone();
    search_settings.abort_check = Some(Box::new(move || {
        if abort_latch.load(Ordering::Relaxed) {
            return true;
        }
        if requested_abort.as_ref().is_some_and(|check| check()) {
            abort_latch.store(true, Ordering::Relaxed);
            true
        } else {
            false
        }
    }));
    (search_settings, cancelled)
}

/// Mix changes of different sizes: moving a variable through a block can expose
/// a factorization that no improving single transposition reaches.
fn mutate_horner_order<T>(order: &mut [T], rng: &mut impl Rng, mutation: usize) {
    let size = order.len();
    if mutation % 4 == 0 {
        let first = rng.random_range(0..size - 1);
        order.swap(first, first + 1);
        return;
    }
    let first = rng.random_range(0..size);
    let mut second = rng.random_range(0..size - 1);
    if second >= first {
        second += 1;
    }
    match mutation % 4 {
        1 => order.swap(first, second),
        2 if first < second => order[first..=second].rotate_left(1),
        2 => order[second..=first].rotate_right(1),
        _ => order[first.min(second)..=first.max(second)].reverse(),
    }
}

/// Score the instructions that will be executed, including binary powering.
/// Keep this local to the search: the public operation-count convention is
/// unchanged.
fn evaluator_horner_cost<T>(evaluator: &ExpressionEvaluator<T>) -> OperationCount {
    let mut count = OperationCount::default();
    for (instruction, _) in &evaluator.instructions {
        match instruction {
            Instr::Add(_, args) => count.additions += args.len() - 1,
            Instr::Mul(_, args) => count.multiplications += args.len() - 1,
            Instr::Pow(_, _, exponent) => {
                count.inversions += usize::from(*exponent < 0);
                let power = exponent.unsigned_abs();
                if power > 1 {
                    count.multiplications += (power.ilog2() + power.count_ones() - 1) as usize;
                }
            }
            Instr::Powf(..) | Instr::BuiltinFun(..) | Instr::ExternalFun(..) => {
                count.add_function_call();
            }
            _ => {}
        }
    }
    count
}

fn search_horner_order<T: Clone + Ord + Send + Sync>(
    vars: &[T],
    settings: &OptimizationSettings,
    score: impl Fn(&[T]) -> OperationCount + Sync,
) -> Vec<T> {
    search_horner_order_with_seed(vars, None, settings, score)
}

fn search_horner_order_with_seed<T: Clone + Ord + Send + Sync>(
    vars: &[T],
    alternative: Option<&[T]>,
    settings: &OptimizationSettings,
    score: impl Fn(&[T]) -> OperationCount + Sync,
) -> Vec<T> {
    let alternatives = alternative.into_iter().collect::<Vec<_>>();
    search_horner_order_with_seeds(vars, &alternatives, settings, score)
}

fn search_horner_order_with_seeds<T: Clone + Ord + Send + Sync>(
    vars: &[T],
    alternatives: &[&[T]],
    settings: &OptimizationSettings,
    score: impl Fn(&[T]) -> OperationCount + Sync,
) -> Vec<T> {
    if settings.horner_iterations == 0 {
        return vars.to_vec();
    }
    if vars.len() < 2 || settings.horner_iterations == 1 {
        return vars.to_vec();
    }

    // Exhaustive searches may use fewer candidates than the requested
    // iteration budget.
    let mut scored_orders = vec![vars.to_vec()];
    for &alternative in alternatives {
        if scored_orders.len() >= settings.horner_iterations {
            break;
        }
        if scored_orders.iter().any(|order| order == alternative) {
            continue;
        }
        scored_orders.push(alternative.to_vec());
    }
    let permutations =
        if vars.len() < 10 && Integer::factorial(vars.len() as u32) <= settings.horner_iterations {
            let indices: Vec<_> = (0..vars.len()).collect();
            // Seed orders were already scored. Visit every other permutation once.
            Some(
                unique_permutations(&indices)
                    .1
                    .into_iter()
                    .filter(|p| {
                        !scored_orders
                            .iter()
                            .any(|order| p.iter().zip(order).all(|(&i, value)| &vars[i] == value))
                    })
                    .collect::<Vec<_>>(),
            )
        } else {
            None
        };
    let candidates = permutations
        .as_ref()
        .map_or(settings.horner_iterations - scored_orders.len(), Vec::len);
    let mut initial = score(vars);

    let mut initial_order = vars.to_vec();
    for alternative in scored_orders.iter().skip(1) {
        let alternative_cost = score(alternative);

        if horner_cost(alternative_cost) < horner_cost(initial)
            || horner_cost(alternative_cost) == horner_cost(initial) && alternative < &initial_order
        {
            initial = alternative_cost;
            initial_order = alternative.clone();
        }
    }
    let best = Mutex::new((initial, initial_order.clone()));
    if candidates == 0 {
        return initial_order;
    }
    let workers = LicenseManager::max_threads(settings.n_cores)
        .max(1)
        .min(candidates);
    let abort = AtomicBool::new(false);
    let inherited_unlock = crate::license::InheritedLibraryUnlock::capture();

    std::thread::scope(|scope| {
        for worker in 0..workers {
            let best = &best;
            let abort = &abort;
            let permutations = &permutations;
            let score = &score;
            let initial_order = &initial_order;
            let run = move || {
                let _unlock = inherited_unlock.activate();
                let mut rng = MonteCarloRng::new(0, worker);
                let mut order = initial_order.clone();
                let mut last = initial;
                let worker_candidates = (candidates - 1 - worker) / workers + 1;
                // Keep short searches in one basin. Longer searches also try
                // reversed, shuffled, and perturbed-best starts, all charged to
                // the same candidate budget.
                let chain_length = if worker_candidates >= 128 {
                    worker_candidates.div_ceil(4)
                } else {
                    worker_candidates
                };
                let mut mutation = 0;
                // Striding includes the remainder when the budget is not divisible
                // by the number of workers.
                for (step, candidate) in (worker..candidates).step_by(workers).enumerate() {
                    if abort.load(Ordering::Relaxed) {
                        break;
                    }
                    if worker + 1 == workers
                        && let Some(check) = &settings.abort_check
                        && check()
                    {
                        abort.store(true, Ordering::Relaxed);
                        break;
                    }
                    let previous = order.clone();
                    let restart = permutations.is_none() && step > 0 && step % chain_length == 0;
                    if let Some(permutations) = &permutations {
                        order = permutations[candidate]
                            .iter()
                            .map(|&i| vars[i].clone())
                            .collect();
                    } else if restart {
                        match step / chain_length {
                            1 => order = vars.iter().rev().cloned().collect(),
                            2 => {
                                order = vars.to_vec();
                                for i in (1..order.len()).rev() {
                                    order.swap(i, rng.random_range(0..=i));
                                }
                            }
                            _ => {
                                order = best.lock().unwrap().1.clone();
                                for _ in 0..3 {
                                    mutate_horner_order(&mut order, &mut rng, mutation);
                                    mutation += 1;
                                }
                            }
                        }
                    } else {
                        mutate_horner_order(&mut order, &mut rng, mutation);
                        mutation += 1;
                    }
                    let current = score(&order);

                    if permutations.is_some()
                        || restart
                        || horner_cost(current) <= horner_cost(last)
                    {
                        last = current;
                        // Keep the score and order under one lock: independent
                        // atomics can expose a score assembled from different winners.
                        let mut winner = best.lock().unwrap();
                        if horner_cost(current) < horner_cost(winner.0)
                            || horner_cost(current) == horner_cost(winner.0) && order < winner.1
                        {
                            *winner = (current, order.clone());
                        }
                    } else {
                        order = previous;
                    }
                }
            };
            if worker + 1 == workers {
                run();
            } else {
                scope.spawn(run);
            }
        }
    });
    let (ops, order) = best.into_inner().unwrap();
    if settings.verbose {
        info!("Final Horner scheme ops: {}", ops);
    }

    order
}

#[cfg(test)]
mod horner_search_tests {
    use super::*;

    #[test]
    fn complete_permutations_with_worker_remainder_and_deterministic_ties() {
        let visited = Mutex::new(Vec::new());
        let settings = OptimizationSettings {
            horner_iterations: 24,
            n_cores: 3,
            ..Default::default()
        };
        let result = search_horner_order(&[3, 2, 1, 0], &settings, |order| {
            visited.lock().unwrap().push(order.to_vec());
            OperationCount::default()
        });
        let mut visited = visited.into_inner().unwrap();
        assert_eq!(visited.len(), 24);
        visited.sort();
        visited.dedup();
        assert_eq!(visited.len(), 24);
        assert_eq!(result, [0, 1, 2, 3]);
    }

    #[test]
    fn multiplication_priority_then_addition_and_initial_score() {
        let settings = OptimizationSettings {
            horner_iterations: 6,
            n_cores: 1,
            ..Default::default()
        };
        let result = search_horner_order(&[0, 1, 2], &settings, |order| {
            if order == [1, 0, 2] {
                OperationCount::new(50, 1, 0, 0)
            } else if order == [2, 1, 0] {
                OperationCount::new(20, 1, 0, 0)
            } else {
                OperationCount::new(0, 2, 0, 0)
            }
        });
        assert_eq!(result, [2, 1, 0]);

        let result = search_horner_order(
            &[0, 1, 2],
            &OptimizationSettings {
                horner_iterations: 2,
                ..settings
            },
            |order| OperationCount::new(0, usize::from(order != [0, 1, 2]), 0, 0),
        );
        assert_eq!(result, [0, 1, 2]);
    }

    #[test]
    fn random_search_spends_entire_budget_without_no_op_swaps() {
        let visited = Mutex::new(Vec::new());
        search_horner_order(
            &[0, 1, 2, 3],
            &OptimizationSettings {
                horner_iterations: 11,
                n_cores: 1,
                ..Default::default()
            },
            |order| {
                visited.lock().unwrap().push(order.to_vec());
                OperationCount::default()
            },
        );
        let visited = visited.into_inner().unwrap();
        assert_eq!(visited.len(), 11);
        assert!(visited.windows(2).all(|orders| orders[0] != orders[1]));
    }

    #[test]
    fn one_variable_needs_no_search() {
        assert_eq!(
            search_horner_order(&[0], &OptimizationSettings::default(), |_| panic!()),
            [0]
        );
    }
}

/// Errors that can occur while building or performing numerical expression evaluation.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum EvaluationError {
    /// An expression supplied as a parameter is not a supported indeterminate.
    NotIndeterminate {
        /// The invalid parameter expression.
        atom: Atom,
    },
    /// The number of supplied parameter values does not match the evaluator.
    InvalidParameterCount {
        /// The evaluator's parameter count.
        expected: usize,
        /// The number of supplied values.
        actual: usize,
    },
    /// The output buffer has the wrong length.
    InvalidOutputCount {
        /// The evaluator's output count.
        expected: usize,
        /// The supplied output length.
        actual: usize,
    },
    /// The requested hot-start optimization is unavailable for this evaluation path.
    UnsupportedHotStart,
    /// A coefficient cannot be converted to the evaluation domain.
    UnsupportedCoefficient {
        /// A printable description of the unsupported coefficient.
        coefficient: String,
    },
    /// A variable has neither a parameter value nor a registered evaluator.
    UndefinedVariable {
        /// The unresolved variable.
        symbol: Symbol,
    },
    /// A function call has neither a parameter value nor a registered evaluator.
    UndefinedFunction {
        /// The unresolved function call.
        expression: Atom,
    },
    /// A function call supplies the wrong number of arguments.
    WrongNumberOfArguments {
        /// The called function.
        function: Symbol,
        /// The registered argument count.
        expected: usize,
        /// The number of supplied arguments.
        actual: usize,
    },
    /// A built-in function is called with an unsupported number of arguments.
    UnsupportedBuiltinArity {
        /// The built-in function.
        function: Symbol,
        /// The supported argument count.
        expected: usize,
        /// The number of supplied arguments.
        actual: usize,
    },
    /// A function registration disagrees with the existing number of tag arguments.
    InconsistentFunctionTagCount {
        /// The function being registered.
        function: Symbol,
        /// The previously registered tag count.
        expected: usize,
        /// The new registration's tag count.
        actual: usize,
    },
    /// A function definition conflicts with an existing definition.
    FunctionRedefined {
        /// The function being redefined.
        function: Atom,
    },
    /// The evaluation domain cannot represent the imaginary unit required by the expression.
    NumericalTypeDoesNotSupportImaginaryUnit,
    /// An expression could not be evaluated.
    EvaluationFailed {
        /// The expression being evaluated.
        expression: Atom,
        /// The underlying failure message.
        reason: String,
    },
    /// An evaluation tree could not be constructed.
    EvaluationTreeConstructionFailed {
        /// The expression being converted.
        expression: Atom,
        /// The underlying failure message.
        reason: String,
    },
    /// An evaluator could not be constructed.
    EvaluatorConstructionFailed {
        /// The expression being converted.
        expression: Atom,
        /// The underlying failure message.
        reason: String,
    },
    /// A joint evaluator for several expressions could not be constructed.
    MultiEvaluatorConstructionFailed {
        /// The number of requested expressions.
        expression_count: usize,
        /// The underlying failure message.
        reason: String,
    },
    /// No evaluator is registered for an expression in the requested numerical domain.
    MissingEvaluator {
        /// The expression requiring an evaluator.
        expression: Atom,
        /// The requested evaluator type.
        eval_type: String,
    },
}

impl std::error::Error for EvaluationError {}

impl std::fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EvaluationError::NotIndeterminate { atom } => {
                write!(f, "atom is not indeterminate: {atom:?}")
            }
            EvaluationError::InvalidParameterCount { expected, actual } => {
                write!(
                    f,
                    "invalid parameter count: expected {expected}, got {actual}"
                )
            }
            EvaluationError::InvalidOutputCount { expected, actual } => {
                write!(f, "invalid output count: expected {expected}, got {actual}")
            }
            EvaluationError::UnsupportedHotStart => {
                f.write_str("hot start is not supported before the deprecation of Expression")
            }
            EvaluationError::UnsupportedCoefficient { coefficient } => {
                write!(
                    f,
                    "{coefficient} coefficients are not supported for evaluation"
                )
            }
            EvaluationError::UndefinedVariable { symbol } => {
                write!(
                    f,
                    "variable {symbol} is not in a parameter map and does not have an evaluator"
                )
            }
            EvaluationError::UndefinedFunction { expression } => {
                write!(
                    f,
                    "function {expression} is not in a parameter map and does not have an evaluator"
                )
            }
            EvaluationError::WrongNumberOfArguments {
                function,
                expected,
                actual,
            } => write!(
                f,
                "function {} called with wrong number of arguments: {actual} vs {expected}",
                function.get_name()
            ),
            EvaluationError::UnsupportedBuiltinArity {
                function,
                expected,
                actual,
            } => write!(
                f,
                "builtin function {} must have exactly {expected} argument(s), got {actual}",
                function.get_name()
            ),
            EvaluationError::InconsistentFunctionTagCount {
                function,
                expected,
                actual,
            } => write!(
                f,
                "function {} cannot be registered with {actual} tag(s); it was already registered with {expected} tag(s)",
                function.get_name()
            ),
            EvaluationError::FunctionRedefined { function } => {
                write!(f, "function {} is redefined", function.to_plain_string())
            }

            EvaluationError::NumericalTypeDoesNotSupportImaginaryUnit => {
                f.write_str("numerical type does not support imaginary unit")
            }
            EvaluationError::EvaluationFailed { expression, reason } => {
                write!(f, "could not evaluate {expression}: {reason}")
            }
            EvaluationError::EvaluationTreeConstructionFailed { expression, reason } => write!(
                f,
                "could not build an evaluation tree for {expression}: {reason}"
            ),
            EvaluationError::EvaluatorConstructionFailed { expression, reason } => {
                write!(f, "could not build an evaluator for {expression}: {reason}")
            }
            EvaluationError::MultiEvaluatorConstructionFailed {
                expression_count,
                reason,
            } => write!(
                f,
                "could not build an evaluator for {expression_count} expressions: {reason}"
            ),
            EvaluationError::MissingEvaluator {
                expression,
                eval_type,
            } => {
                write!(
                    f,
                    "could not find an evaluator of type {eval_type} for {expression}"
                )
            }
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
    external_functions: Vec<ExternalFunctionContainer<T>>,
    expressions: SplitExpression<T>,
    param_count: usize,
}

fn register_constant_external_container<T>(
    external_functions: &mut Vec<ExternalFunctionContainer<T>>,
    symbol: Symbol,
    tags: Vec<Atom>,
    fixed_args: Vec<Complex<Rational>>,
    constants: &mut Vec<Complex<Rational>>,
) -> usize {
    let index = if let Some(index) = external_functions
        .iter()
        .position(|x| x.symbol == symbol && x.tags == tags && x.fixed_args == fixed_args)
    {
        index
    } else {
        external_functions.push(ExternalFunctionContainer::new(symbol, tags, fixed_args));
        external_functions.len() - 1
    };

    if let Some(constant_index) = external_functions[index].constant_index {
        return constant_index;
    }

    let constant_index = constants.len();
    constants.push(Complex::default());
    external_functions[index].constant_index = Some(constant_index);
    constant_index
}

// Shared by the root and every function compiled in this build.
struct LinearizationContext<'a> {
    external_functions: Vec<ExternalFunctionContainer<Complex<Rational>>>,
    // Hidden global slots for each body; a pending entry detects recursive definitions.
    functions: HashMap<usize, Option<Vec<Slot>>>,
    // None denotes a registered constant whose value is resolved in the numerical domain.
    constants: Vec<Option<Complex<Rational>>>,
    constant_indices: HashMap<Complex<Rational>, usize>,
    // None denotes a callable; Some contains the fixed arguments of a constant.
    external_indices: HashMap<(Symbol, Vec<Atom>, Option<Vec<Complex<Rational>>>), usize>,
    inputs: Vec<AtomView<'a>>,
    expanding_functions: HashSet<usize>,
}

/// Optimize nested algebra regions once, bottom-up, while keeping function calls
/// and power contents terminal to each Horner search. The cache belongs to one
/// binding scope; exact evaluator inputs are never traversed or renamed.
struct HornerRegions<'a, 's> {
    tags: &'s HashMap<Symbol, usize>,
    settings: &'s OptimizationSettings,
    cancelled: Arc<AtomicBool>,
    parameters: HashSet<AtomView<'a>>,
    repeated: HashSet<AtomView<'a>>,
    optimized: HashMap<AtomView<'a>, AtomOrView<'a>>,
}

impl<'a, 's> HornerRegions<'a, 's> {
    fn new(
        tags: &'s HashMap<Symbol, usize>,
        settings: &'s OptimizationSettings,
        cancelled: Arc<AtomicBool>,
        parameters: impl IntoIterator<Item = AtomView<'a>>,
        roots: &[AtomView<'a>],
    ) -> Self {
        let mut regions = Self {
            tags,
            settings,
            cancelled,
            parameters: parameters.into_iter().collect(),
            repeated: HashSet::default(),
            optimized: HashMap::default(),
        };
        let mut counts = HashMap::default();
        for &root in roots {
            regions.count_children(root, &mut counts);
        }
        regions.repeated = counts
            .into_iter()
            .filter_map(|(region, count)| (count > 1).then_some(region))
            .collect();
        regions
    }

    fn tag_count(&self, symbol: Symbol) -> usize {
        self.tags.get(&symbol).copied().unwrap_or_else(|| {
            symbol
                .get_evaluation_info()
                .map_or(0, |info| info.get_tag_count())
        })
    }

    /// Count calls and power boundaries reached while preparing an algebra
    /// region. Ordinary Add/Mul subtrees are not separate cached regions.
    fn count_children(&self, expression: AtomView<'a>, counts: &mut HashMap<AtomView<'a>, usize>) {
        if self.parameters.contains(&expression) {
            return;
        }
        match expression {
            AtomView::Fun(_) | AtomView::Pow(_) => self.count_region(expression, counts),
            AtomView::Add(sum) => {
                for term in sum {
                    self.count_children(term, counts);
                }
            }
            AtomView::Mul(product) => {
                for factor in product {
                    self.count_children(factor, counts);
                }
            }
            _ => {}
        }
    }

    /// A repeated whole region will reuse its first result, so its descendants
    /// are only prepared once. Borrow all keys from the original scope roots.
    fn count_region(&self, expression: AtomView<'a>, counts: &mut HashMap<AtomView<'a>, usize>) {
        if matches!(expression, AtomView::Num(_) | AtomView::Var(_))
            || self.parameters.contains(&expression)
        {
            return;
        }
        let count = counts.entry(expression).or_insert(0);
        *count += 1;
        if *count > 1 {
            return;
        }
        match expression {
            AtomView::Fun(function) => {
                for argument in function.iter().skip(self.tag_count(function.get_symbol())) {
                    self.count_region(argument, counts);
                }
            }
            AtomView::Pow(power) => {
                let (base, exponent) = power.get_base_exp();
                self.count_region(base, counts);
                self.count_region(exponent, counts);
            }
            _ => self.count_children(expression, counts),
        }
    }

    fn cache_result(&mut self, original: AtomView<'a>, result: &AtomOrView<'a>) {
        if self.repeated.contains(&original) {
            self.optimized.insert(original, result.clone());
        }
    }

    /// Borrow unchanged regions. Rebuild an Add/Mul only after its first
    /// changed child, so large unaffected siblings are copied just once.
    fn children(&mut self, expression: AtomView<'a>) -> AtomOrView<'a> {
        if self.cancelled.load(Ordering::Relaxed) || self.parameters.contains(&expression) {
            return expression.into();
        }
        let (is_sum, children) = match expression {
            AtomView::Num(_) | AtomView::Var(_) => return expression.into(),
            AtomView::Fun(function) => {
                if let Some(cached) = self.optimized.get(&expression) {
                    return cached.clone();
                }
                let tag_count = self.tag_count(function.get_symbol());
                let arguments = function
                    .iter()
                    .enumerate()
                    .map(|(i, argument)| {
                        if i < tag_count {
                            argument.into()
                        } else {
                            self.region(argument)
                        }
                    })
                    .collect::<Vec<_>>();
                let result = if function
                    .iter()
                    .zip(&arguments)
                    .all(|(before, after)| before == after.as_view())
                {
                    expression.into()
                } else {
                    let views = arguments
                        .iter()
                        .map(AtomOrView::as_view)
                        .collect::<Vec<_>>();
                    function.get_symbol().call(&views).into()
                };
                self.cache_result(expression, &result);
                return result;
            }
            AtomView::Pow(power) => {
                if let Some(cached) = self.optimized.get(&expression) {
                    return cached.clone();
                }
                let (base, exponent) = power.get_base_exp();
                let new_base = self.region(base);
                let new_exponent = self.region(exponent);
                let result = if base == new_base.as_view() && exponent == new_exponent.as_view() {
                    expression.into()
                } else {
                    new_base.as_view().pow(new_exponent.as_view()).into()
                };
                self.cache_result(expression, &result);
                return result;
            }
            AtomView::Add(sum) => (true, sum.iter()),
            AtomView::Mul(product) => (false, product.iter()),
        };
        let mut rebuilt: Option<Atom> = None;
        for (i, child) in children.enumerate() {
            let prepared = self.children(child);
            if rebuilt.is_none() && prepared.as_view() != child {
                let mut result = Atom::new();
                if is_sum {
                    let sum = result.to_add();
                    for previous in children.take(i) {
                        sum.extend(previous);
                    }
                } else {
                    let product = result.to_mul();
                    for previous in children.take(i) {
                        product.extend(previous);
                    }
                }
                rebuilt = Some(result);
            }
            if let Some(result) = &mut rebuilt {
                match result {
                    Atom::Add(sum) => sum.extend(prepared.as_view()),
                    Atom::Mul(product) => product.extend(prepared.as_view()),
                    _ => unreachable!(),
                }
            }
        }
        if let Some(rebuilt) = rebuilt {
            Workspace::get_local().with(|ws| {
                let mut normalized = Atom::new();
                rebuilt.as_view().normalize(ws, &mut normalized);
                normalized.into()
            })
        } else {
            expression.into()
        }
    }

    fn region(&mut self, expression: AtomView<'a>) -> AtomOrView<'a> {
        if matches!(expression, AtomView::Num(_) | AtomView::Var(_))
            || self.parameters.contains(&expression)
            || self.cancelled.load(Ordering::Relaxed)
        {
            return expression.into();
        }
        if let Some(result) = self.optimized.get::<[u8]>(expression.get_data()) {
            return result.clone();
        }
        let prepared = self.children(expression);
        if self.cancelled.load(Ordering::Relaxed) {
            return prepared;
        }
        let result = if matches!(prepared.as_view(), AtomView::Add(_) | AtomView::Mul(_)) {
            let orders =
                AtomView::horner_orders(&[prepared.as_view()], &self.parameters, self.settings);
            let mut score_params = HashSet::default();
            if self.settings.horner_use_cpe {
                // Functions are already optimized, fixed values for this region.
                // Use them directly as numerical inputs when scoring its CSE/CPE.
                prepared.as_view().visitor(&mut |part| match part {
                    AtomView::Fun(_) => {
                        score_params.insert(part);
                        false
                    }
                    AtomView::Var(variable) => {
                        if self.parameters.contains(&part)
                            || !matches!(
                                variable.get_symbol().get_id(),
                                Symbol::E_ID | Symbol::PI_ID
                            )
                        {
                            score_params.insert(part);
                        }
                        false
                    }
                    _ => true,
                });
            }
            let mut score_params: Vec<_> = score_params.into_iter().collect();
            score_params.sort();
            let order = search_horner_order_with_seeds(
                &orders[0],
                &[&orders[1], &orders[2]],
                self.settings,
                |order| {
                    let candidate = prepared.as_view().horner_scheme_local(Some(order), true);
                    if self.settings.horner_use_cpe {
                        if let Ok(evaluator) = AtomView::linearize_multiple(
                            &[candidate.as_view()],
                            &FunctionMap::new(),
                            &score_params,
                            self.settings.clone(),
                        ) {
                            return evaluator_horner_cost(
                                &AtomView::optimize_evaluator_instructions(evaluator),
                            );
                        }
                    }
                    // Construction errors belong to the final evaluator build,
                    // including errors in branches which may be eliminated.
                    candidate
                        .as_view()
                        .count_operations_with_subexpressions(&mut HashSet::default())
                },
            );
            if order.is_empty() && !orders[0].is_empty() || self.cancelled.load(Ordering::Relaxed) {
                prepared
            } else {
                prepared
                    .as_view()
                    .horner_scheme_local(Some(&order), true)
                    .into()
            }
        } else {
            prepared
        };
        let result = if result.as_view() == expression {
            expression.into()
        } else {
            result
        };
        self.cache_result(expression, &result);
        result
    }
}

impl LinearizationContext<'_> {
    fn constant_slot(&mut self, constant: Complex<Rational>) -> Slot {
        let index = *self
            .constant_indices
            .entry(constant.clone())
            .or_insert_with(|| {
                self.constants.push(Some(constant));
                self.constants.len() - 1
            });
        Slot::Const(index)
    }

    fn external_constant_slot(
        &mut self,
        symbol: Symbol,
        tags: Vec<Atom>,
        fixed_args: Vec<Complex<Rational>>,
    ) -> Slot {
        let index = *self
            .external_indices
            .entry((symbol, tags.clone(), Some(fixed_args.clone())))
            .or_insert_with(|| {
                let mut external = ExternalFunctionContainer::new(symbol, tags, fixed_args);
                external.constant_index = Some(self.constants.len());
                self.constants.push(None);
                self.external_functions.push(external);
                self.external_functions.len() - 1
            });
        Slot::Const(self.external_functions[index].constant_index.unwrap())
    }
}

impl<'a> AtomView<'a> {
    /// Candidate orders for one algebra region, without looking inside calls
    /// or the children of powers. Each nested region gets its own search.
    fn horner_orders<'e>(
        expressions: &[AtomView<'e>],
        parameters: &HashSet<AtomView<'_>>,
        settings: &OptimizationSettings,
    ) -> [Vec<AtomView<'e>>; 3] {
        let mut counts = HashMap::default();
        for expression in expressions {
            expression.count_horner_candidates(parameters, &mut counts);
        }
        let mut counts: Vec<_> = counts.into_iter().filter(|(_, count)| *count > 1).collect();
        counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        counts.truncate(settings.max_horner_scheme_variables);
        let occurrence: Vec<_> = counts.into_iter().map(|(atom, _)| atom).collect();
        let mut computed_first = occurrence.clone();
        computed_first.sort_by_key(|atom| matches!(atom, AtomView::Var(_)));
        let mut lexical = occurrence.clone();
        lexical.sort_by_cached_key(|atom| atom.to_canonical_string());
        [computed_first, occurrence, lexical]
    }

    /// Evaluation treats a non-integer power as an atomic computed value, just
    /// like a function call. In particular exp(x) is stored as E^x, so looking
    /// only for variables and function atoms misses a useful common factor.
    fn count_horner_candidates(
        self,
        parameters: &HashSet<AtomView<'_>>,
        counts: &mut HashMap<AtomView<'a>, usize>,
    ) {
        if parameters.contains(&self) {
            *counts.entry(self).or_insert(0) += 1;
            return;
        }
        match self {
            AtomView::Num(_) => {}
            AtomView::Var(variable) => {
                if !matches!(variable.get_symbol().get_id(), Symbol::E_ID | Symbol::PI_ID) {
                    *counts.entry(self).or_insert(0) += 1;
                }
            }
            AtomView::Fun(_) => {
                *counts.entry(self).or_insert(0) += 1;
            }
            AtomView::Pow(power) => {
                let (base, exponent) = power.get_base_exp();
                // The base is one indeterminate at this level. Retain positive
                // integer degree without descending into the base's algebra.
                if i64::try_from(exponent).is_ok_and(|power| power > 0) {
                    if parameters.contains(&base)
                        || base != InlineVar::new(Symbol::E).as_view()
                            && base != InlineVar::new(Symbol::PI).as_view()
                    {
                        *counts.entry(base).or_insert(0) += 1;
                    }
                } else {
                    *counts.entry(self).or_insert(0) += 1;
                }
            }
            AtomView::Mul(product) => {
                for factor in product {
                    factor.count_horner_candidates(parameters, counts);
                }
            }
            AtomView::Add(sum) => {
                for term in sum {
                    term.count_horner_candidates(parameters, counts);
                }
            }
        }
    }

    pub(crate) fn to_evaluator(
        expressions: &[Self],
        mut fn_map: FunctionMap,
        params: &[Atom],
        settings: OptimizationSettings,
    ) -> Result<ExpressionEvaluator<Complex<Rational>>, EvaluationError> {
        if settings.verbose {
            let mut cse = HashSet::default();
            let mut count = OperationCount::default();
            for e in expressions {
                count += e.count_operations_with_subexpressions(&mut cse);
            }
            info!("Initial ops: {}", count);
        }

        if settings.horner_iterations == 0 {
            return Self::linearize_multiple(expressions, &fn_map, params, settings);
        }

        if settings.hot_start.is_some() {
            return Err(EvaluationError::UnsupportedHotStart);
        }

        // All argument, function-body, and root searches share a one-shot abort
        // latch, so cancellation of an inner region also stops later searches.
        let (search_settings, cancelled) = latched_horner_settings(&settings);

        // Function bodies have independent bindings and independent orders.
        // Alias bodies inherit their callers, so retain every possible formal
        // input identity when preparing their shared definition.
        // Split each owned definition into its immutable bindings and mutable
        // body. Alias bindings can then borrow arguments from all definitions
        // while the bodies are updated in place.
        let mut definitions: Vec<_> = fn_map
            .tagged_fn_map
            .values_mut()
            .map(|function| {
                (
                    function.args.as_slice(),
                    &mut function.body,
                    function.is_alias,
                )
            })
            .collect();
        let alias_parameters: Vec<_> = params
            .iter()
            .map(Atom::as_view)
            .chain(
                definitions
                    .iter()
                    .flat_map(|(args, _, _)| (*args).iter().map(Indeterminate::as_view)),
            )
            .collect();
        for (args, body, is_alias) in &mut definitions {
            let parameters = if *is_alias {
                alias_parameters.clone()
            } else {
                params
                    .iter()
                    .map(Atom::as_view)
                    .chain(args.iter().map(Indeterminate::as_view))
                    .collect()
            };
            let mut regions = HornerRegions::new(
                &fn_map.tag,
                &search_settings,
                cancelled.clone(),
                parameters,
                &[body.as_view()],
            );
            let prepared = regions.region(body.as_view());
            drop(regions);
            if let AtomOrView::Atom(prepared) = prepared {
                **body = prepared;
            }
        }
        drop(definitions);
        let fn_map = &fn_map;
        let mut regions = HornerRegions::new(
            &fn_map.tag,
            &search_settings,
            cancelled.clone(),
            params.iter().map(Atom::as_view),
            expressions,
        );
        let prepared = expressions
            .iter()
            .map(|expression| regions.children(*expression))
            .collect::<Vec<_>>();
        drop(regions);
        let expression_views = prepared.iter().map(AtomOrView::as_view).collect::<Vec<_>>();
        let expressions = expression_views.as_slice();
        if cancelled.load(Ordering::Relaxed) {
            let mut evaluator =
                Self::linearize_multiple(expressions, fn_map, params, search_settings)?;
            evaluator.fix_labels();
            evaluator.optimize_stack();
            evaluator.settings = settings;
            evaluator.settings.abort_level = 0;
            return Ok(evaluator);
        }
        let parameters = params.iter().map(Atom::as_view).collect();
        let [computed_first, v, lexical] = Self::horner_orders(expressions, &parameters, &settings);
        if settings.horner_iterations == 1 || computed_first.len() < 2 {
            let result = Self::build_horner_evaluator(
                expressions,
                fn_map,
                params,
                &computed_first,
                search_settings,
            );

            let mut evaluator = result?;
            evaluator.optimize_stack();
            evaluator.settings = settings;
            evaluator.settings.abort_level = 0;
            return Ok(evaluator);
        }

        if !settings.horner_use_cpe {
            let scheme = search_horner_order_with_seeds(
                &computed_first,
                &[&v, &lexical],
                &search_settings,
                |order| Self::horner_scheme_cost(expressions, order),
            );
            // The inexpensive search still receives the complete CSE/CPE
            // pipeline once, after selecting its winning Horner scheme.
            let mut evaluator = if scheme.is_empty() {
                Self::build_unhornered_evaluator(expressions, fn_map, params, search_settings)?
            } else {
                Self::build_horner_evaluator(expressions, fn_map, params, &scheme, search_settings)?
            };
            evaluator.settings = settings;
            evaluator.settings.abort_level = 0;
            evaluator.optimize_stack();

            return Ok(evaluator);
        }

        // Build and score the final CSE/CPE program for every candidate. A cheap
        // tree score can reject a scheme whose new common pairs make the final
        // evaluator smaller. Retain only the winning program, avoiding a final
        // rebuild and keeping memory bounded by one candidate per worker.
        let best_evaluator = Mutex::new(
            None::<(
                OperationCount,
                Vec<AtomView<'_>>,
                ExpressionEvaluator<Complex<Rational>>,
            )>,
        );
        let first_error = Mutex::new(None);
        let mut candidate_settings = search_settings.clone();
        candidate_settings.verbose = false;
        search_horner_order_with_seeds(
            &computed_first,
            &[&v, &lexical],
            &search_settings,
            |order| {
                let unavailable = OperationCount::new(0, usize::MAX, 0, 0);
                // The first candidate is built even if cancellation was requested,
                // so an interrupted optimization still returns a usable evaluator.
                if first_error.lock().unwrap().is_some()
                    || cancelled.load(Ordering::Relaxed) && best_evaluator.lock().unwrap().is_some()
                {
                    return unavailable;
                }
                let evaluator = match Self::build_horner_evaluator(
                    expressions,
                    fn_map,
                    params,
                    order,
                    candidate_settings.clone(),
                ) {
                    Ok(evaluator) => evaluator,
                    Err(error) => {
                        first_error.lock().unwrap().get_or_insert(error);
                        cancelled.store(true, Ordering::Relaxed);
                        return unavailable;
                    }
                };
                let cost = evaluator_horner_cost(&evaluator);
                let mut winner = best_evaluator.lock().unwrap();
                if winner.as_ref().is_none_or(|(best_cost, best_order, _)| {
                    horner_cost(cost) < horner_cost(*best_cost)
                        || horner_cost(cost) == horner_cost(*best_cost)
                            && order < best_order.as_slice()
                }) {
                    *winner = Some((cost, order.to_vec(), evaluator));
                }
                cost
            },
        );
        if let Some(error) = first_error.into_inner().unwrap() {
            return Err(error);
        }
        let mut evaluator = match best_evaluator.into_inner().unwrap() {
            Some((_, _, evaluator)) => evaluator,
            None => Self::build_unhornered_evaluator(expressions, fn_map, params, search_settings)?,
        };
        evaluator.settings = settings;
        evaluator.settings.abort_level = 0;
        evaluator.optimize_stack();

        Ok(evaluator)
    }

    /// Skipping the Horner stage still permits later optimization stages.
    fn build_unhornered_evaluator(
        expressions: &[AtomView<'_>],
        fn_map: &FunctionMap,
        params: &[Atom],
        settings: OptimizationSettings,
    ) -> Result<ExpressionEvaluator<Complex<Rational>>, EvaluationError> {
        let evaluator = Self::linearize_multiple(expressions, fn_map, params, settings)?;
        Ok(Self::optimize_evaluator_instructions(evaluator))
    }

    /// Construct one candidate without changing its variable order. Stack
    /// allocation is delayed until the winning candidate is selected.
    fn build_horner_evaluator<T: AtomCore>(
        expressions: &[AtomView<'_>],
        fn_map: &FunctionMap,
        params: &[Atom],
        scheme: &[T],
        settings: OptimizationSettings,
    ) -> Result<ExpressionEvaluator<Complex<Rational>>, EvaluationError> {
        let hornered_expressions = expressions
            .iter()
            .map(|x| x.horner_scheme_local(Some(scheme), true))
            .collect::<Vec<_>>();

        if settings.horner_iterations == 1 && settings.verbose {
            let mut cse = HashSet::default();
            let mut count = OperationCount::default();
            for e in &hornered_expressions {
                count += e.as_view().count_operations_with_subexpressions(&mut cse);
            }
            info!("Horner scheme ops: {}", count);
        }

        let e = Self::linearize_multiple(&hornered_expressions, fn_map, params, settings)?;

        drop(hornered_expressions);

        Ok(Self::optimize_evaluator_instructions(e))
    }

    fn optimize_evaluator_instructions(
        mut e: ExpressionEvaluator<Complex<Rational>>,
    ) -> ExpressionEvaluator<Complex<Rational>> {
        loop {
            let r = e.remove_common_instructions();

            if r == 0 || e.settings.abort_level > 0 {
                e.settings.abort_level = 0;
                break;
            }

            if e.settings.verbose {
                let count = e.count_operations();
                info!("Removed {} common instructions: {}", r, count);
            }
        }

        for _ in 0..e.settings.cpe_iterations.unwrap_or(usize::MAX) {
            let r = e.remove_common_pairs();
            if r == 0 || e.settings.abort_level > 0 {
                e.settings.abort_level = 0;
                break;
            }

            // Pair extraction can make previously different instruction DAGs
            // identical, including functions and powers that CPE cannot merge.
            e.remove_common_instructions();

            if e.settings.verbose {
                let count = e.count_operations();
                info!("Removed {} common pairs: {}", r, count);
            }
        }

        e
    }

    /// Search for a Horner variable order that reduces the combined operation
    /// count of `expressions`. `vars` supplies the initial order; `settings`
    /// controls the search budget and parallelism. Returns the best order found.
    pub fn optimize_horner_scheme_multiple(
        expressions: &[Self],
        vars: &[Indeterminate],
        settings: &OptimizationSettings,
    ) -> Vec<Indeterminate> {
        search_horner_order(vars, settings, |order| {
            Self::horner_scheme_cost(expressions, order)
        })
    }

    fn horner_scheme_cost<T: AtomCore>(
        expressions: &[AtomView<'_>],
        order: &[T],
    ) -> OperationCount {
        let hornered: Vec<_> = expressions
            .iter()
            .map(|expression| expression.horner_scheme_local(Some(order), true))
            .collect();
        let mut subexpr = HashSet::default();
        let mut operations = OperationCount::default();
        for expression in &hornered {
            operations += expression
                .as_view()
                .count_operations_with_subexpressions(&mut subexpr);
        }
        operations
    }

    pub(crate) fn linearize_multiple<T: AtomCore, P: AtomCore>(
        expressions: &[T],
        fn_map: &FunctionMap,
        params: &[P],
        settings: OptimizationSettings,
    ) -> Result<ExpressionEvaluator<Complex<Rational>>, EvaluationError> {
        let expressions = expressions
            .iter()
            .map(|e| e.as_atom_view())
            .collect::<Vec<_>>();
        let params = params.iter().map(|p| p.as_atom_view()).collect::<Vec<_>>();
        let mut context = LinearizationContext {
            external_functions: Vec::new(),
            functions: HashMap::default(),
            constants: Vec::new(),
            constant_indices: HashMap::default(),
            external_indices: HashMap::default(),
            inputs: params.clone(),
            expanding_functions: HashSet::default(),
        };
        let (instr, results) = AtomView::linearize_multiple_impl(
            &expressions,
            fn_map,
            &params,
            &settings,
            &mut context,
        )?;
        let param_count = params.len();
        let reserved_indices = param_count + context.constants.len();
        let mut stack = vec![Complex::default(); reserved_indices + instr.len()];
        let (instructions, result_indices) = AtomView::lower_instructions(
            instr,
            results,
            param_count,
            reserved_indices,
            &mut context,
        )?;
        for (slot, constant) in stack[param_count..reserved_indices]
            .iter_mut()
            .zip(context.constants)
        {
            if let Some(value) = constant {
                *slot = value;
            }
        }
        Ok(ExpressionEvaluator {
            stack,
            param_count,
            reserved_indices,
            instructions,
            result_indices,
            external_fns: context.external_functions,
            settings,
        })
    }

    fn linearize_multiple_impl(
        expressions: &[AtomView<'a>],
        fn_map: &'a FunctionMap,
        params: &[AtomView<'a>],
        settings: &OptimizationSettings,
        context: &mut LinearizationContext<'a>,
    ) -> Result<(Vec<Instruction>, Vec<Slot>), EvaluationError> {
        let mut instr = Vec::new();

        // we can only safely remove entries that don't depend on any of the function arguments
        let mut subexpression: HashMap<AtomView, Slot> = HashMap::default();

        let mut result_indices = vec![];
        let args = params
            .iter()
            .enumerate()
            .map(|(i, p)| (*p, Slot::Param(i)))
            .collect();
        for expr in expressions {
            let res = expr.linearize_impl(
                fn_map,
                context,
                &args,
                settings,
                &mut instr,
                &mut subexpression,
            )?;
            result_indices.push(res);
        }

        Ok((instr, result_indices))
    }

    fn lower_instructions(
        instr: Vec<Instruction>,
        results: Vec<Slot>,
        param_count: usize,
        reserved_indices: usize,
        context: &mut LinearizationContext<'a>,
    ) -> Result<(Vec<(Instr, ComplexPhase)>, Vec<usize>), EvaluationError> {
        macro_rules! slot_map {
            ($s: expr) => {
                match $s {
                    Slot::Param(i) => i,
                    Slot::Const(i) => param_count + i,
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
                            args.into_iter().map(|x| slot_map!(x)).collect(),
                        ),
                        ComplexPhase::default(),
                    ));
                }
                Instruction::Mul(o, args, _) => {
                    instructions.push((
                        Instr::Mul(
                            slot_map!(o),
                            args.into_iter().map(|x| slot_map!(x)).collect(),
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
                Instruction::Fun(o, b, _) => {
                    let (sym, tags, args) = *b;

                    if sym.is_fixed_builtin() {
                        if args.len() != 1 {
                            return Err(EvaluationError::UnsupportedBuiltinArity {
                                function: sym,
                                expected: 1,
                                actual: args.len(),
                            });
                        }
                        instructions.push((
                            Instr::BuiltinFun(slot_map!(o), sym, slot_map!(args[0])),
                            ComplexPhase::default(),
                        ));
                        continue;
                    }

                    let tags = tags.iter().map(|x| crate::parse!(x)).collect::<Vec<_>>();
                    let index = *context
                        .external_indices
                        .entry((sym, tags.clone(), None))
                        .or_insert_with(|| {
                            context
                                .external_functions
                                .push(ExternalFunctionContainer::new(sym, tags, vec![]));
                            context.external_functions.len() - 1
                        });

                    instructions.push((
                        Instr::ExternalFun(
                            slot_map!(o),
                            index,
                            args.into_iter().map(|x| slot_map!(x)).collect(),
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

        Ok((
            instructions,
            results.into_iter().map(|s| slot_map!(s)).collect(),
        ))
    }

    #[allow(clippy::too_many_arguments)]
    fn linearize_function_call(
        fn_map: &'a FunctionMap,
        context: &mut LinearizationContext<'a>,
        settings: &OptimizationSettings,
        symbol: Symbol,
        tags: Vec<Atom>,
        function: &'a Expr,
        arg_slots: Vec<Slot>,
        instr: &mut Vec<Instruction>,
    ) -> Result<Slot, EvaluationError> {
        if arg_slots.len() != function.args.len() {
            return Err(EvaluationError::WrongNumberOfArguments {
                function: symbol,
                expected: function.args.len() + function.tag_len,
                actual: arg_slots.len() + tags.len(),
            });
        }
        let recursive_call = || EvaluationError::EvaluatorConstructionFailed {
            expression: symbol.call(&tags),
            reason: "recursive function dependency".into(),
        };
        if !context.functions.contains_key(&function.id) {
            // A pending entry detects recursion. No caller bindings enter this compilation:
            // root inputs precede declared arguments, which shadow matching input names.
            context.functions.insert(function.id, None);
            let body_params = context
                .inputs
                .iter()
                .copied()
                .chain(function.args.iter().map(|arg| arg.as_view()))
                .collect::<Vec<_>>();
            let (mut instr, mut result_indices) = Self::linearize_multiple_impl(
                &[function.body.as_view()],
                fn_map,
                &body_params,
                settings,
                context,
            )?;
            // Root inputs and constants stay symbolic during construction. Compact only
            // the surviving references, including globals forwarded to callees, once the
            // body is complete. Declared arguments retain their original positions.
            let mut globals = Vec::new();
            let root_param_count = context.inputs.len();
            let mut param_count = function.args.len();
            let mut indices = HashMap::default();
            let mut lower_global = |slot: &mut Slot| match *slot {
                Slot::Param(index) if index >= root_param_count => {
                    *slot = Slot::Param(index - root_param_count);
                }
                Slot::Param(_) | Slot::Const(_) => {
                    let index = *indices.entry(*slot).or_insert_with(|| {
                        let index = param_count + globals.len();
                        globals.push(*slot);
                        index
                    });
                    *slot = Slot::Param(index);
                }
                Slot::Temp(_) | Slot::Out(_) => {}
            };
            for instruction in &mut instr {
                match instruction {
                    Instruction::Add(_, args, _) | Instruction::Mul(_, args, _) => {
                        args.iter_mut().for_each(&mut lower_global);
                    }
                    Instruction::Pow(_, base, _, _) | Instruction::Assign(_, base) => {
                        lower_global(base);
                    }
                    Instruction::Powf(_, base, exp, _) => {
                        lower_global(base);
                        lower_global(exp);
                    }
                    Instruction::Fun(_, function, _) => {
                        function.2.iter_mut().for_each(&mut lower_global);
                    }
                    Instruction::IfElse(cond, _) => lower_global(cond),
                    Instruction::Join(_, cond, then_value, else_value) => {
                        lower_global(cond);
                        lower_global(then_value);
                        lower_global(else_value);
                    }
                    Instruction::Goto(_) | Instruction::Label(_) => {}
                }
            }
            result_indices.iter_mut().for_each(lower_global);
            param_count += globals.len();

            let stack_size = param_count + instr.len();
            let (instructions, result_indices) =
                Self::lower_instructions(instr, result_indices, param_count, param_count, context)?;
            let body = FunctionBody {
                instructions,
                param_count,
                result_index: result_indices[0],
                stack_size,
            };
            // Recursive compilation has already registered every callee.
            context.external_indices.insert(
                (symbol, tags.clone(), None),
                context.external_functions.len(),
            );
            context
                .external_functions
                .push(ExternalFunctionContainer::with_body(
                    symbol,
                    tags.clone(),
                    Arc::new(body),
                ));
            context.functions.insert(function.id, Some(globals));
        }
        let globals = context.functions[&function.id]
            .as_ref()
            .ok_or_else(recursive_call)?;

        // Global slots bypass local name bindings, including shadowing arguments.
        // The caller's final lowering pass will compact them in its own signature.
        let mut call_args = arg_slots;
        call_args.extend_from_slice(globals);

        let result = Slot::Temp(instr.len());
        instr.push(Instruction::Fun(
            result,
            Box::new((
                symbol,
                tags.iter().map(|tag| tag.to_canonical_string()).collect(),
                call_args,
            )),
            false,
        ));
        Ok(result)
    }

    #[allow(clippy::too_many_arguments)]
    fn linearize_inline_function(
        &self,
        function: &'a Expr,
        arg_slots: Vec<Slot>,
        fn_map: &'a FunctionMap,
        context: &mut LinearizationContext<'a>,
        params: &HashMap<AtomView<'a>, Slot>,
        settings: &OptimizationSettings,
        instr: &mut Vec<Instruction>,
        subexpressions: &mut HashMap<AtomView<'a>, Slot>,
    ) -> Result<Slot, EvaluationError> {
        if !context.expanding_functions.insert(function.id) {
            return Err(EvaluationError::EvaluatorConstructionFailed {
                expression: self.to_owned(),
                reason: "recursive function or alias dependency".into(),
            });
        }
        let mut body_params = HashMap::default();
        if !function.is_alias {
            body_params.extend(
                context
                    .inputs
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (*p, Slot::Param(i))),
            );
            body_params.extend(
                function
                    .args
                    .iter()
                    .zip(arg_slots)
                    .map(|(arg, slot)| (arg.as_view(), slot)),
            );
        }
        // Aliases inherit the caller's bindings and cache. Ordinary functions, including
        // zero-argument functions, get their own scope and expression cache.
        let mut body_subexpressions = HashMap::default();
        let result = function.body.as_view().linearize_impl(
            fn_map,
            context,
            if function.is_alias {
                params
            } else {
                &body_params
            },
            settings,
            instr,
            if function.is_alias {
                subexpressions
            } else {
                &mut body_subexpressions
            },
        );
        context.expanding_functions.remove(&function.id);
        result
    }

    // Yields the stack index that contains the output.
    fn linearize_impl(
        &self,
        fn_map: &'a FunctionMap,
        context: &mut LinearizationContext<'a>,
        params: &HashMap<AtomView<'a>, Slot>,
        settings: &OptimizationSettings,
        instr: &mut Vec<Instruction>,
        subexpressions: &mut HashMap<AtomView<'a>, Slot>,
    ) -> Result<Slot, EvaluationError> {
        if matches!(*self, AtomView::Var(_) | AtomView::Fun(_)) {
            if let Some(slot) = params.get(self) {
                return Ok(*slot);
            }
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
                    CoefficientView::Indeterminate => {
                        return Err(EvaluationError::UnsupportedCoefficient {
                            coefficient: "indeterminate".to_string(),
                        });
                    }
                    CoefficientView::Infinity(_) => {
                        return Err(EvaluationError::UnsupportedCoefficient {
                            coefficient: "infinity".to_string(),
                        });
                    }
                    CoefficientView::FiniteField(_, _) => {
                        return Err(EvaluationError::UnsupportedCoefficient {
                            coefficient: "finite field".to_string(),
                        });
                    }
                    CoefficientView::RationalPolynomial(_) => {
                        return Err(EvaluationError::UnsupportedCoefficient {
                            coefficient: "rational polynomial".to_string(),
                        });
                    }
                };

                context.constant_slot(c)
            }
            AtomView::Var(v) => {
                let s = v.get_symbol();

                if let Some(expr) = fn_map.get(*self) {
                    if !expr.args.is_empty() {
                        return Err(EvaluationError::WrongNumberOfArguments {
                            function: s,
                            expected: expr.args.len(),
                            actual: 0,
                        });
                    }
                    if expr.is_alias || expr.options.should_inline() {
                        return self.linearize_inline_function(
                            expr,
                            vec![],
                            fn_map,
                            context,
                            params,
                            settings,
                            instr,
                            subexpressions,
                        );
                    }

                    return Self::linearize_function_call(
                        fn_map,
                        context,
                        settings,
                        s,
                        vec![],
                        expr,
                        vec![],
                        instr,
                    );
                }

                if s.get_evaluation_info().is_some() {
                    return Ok(context.external_constant_slot(s, vec![], vec![]));
                }

                return Err(EvaluationError::UndefinedVariable { symbol: s });
            }
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [
                    Symbol::LOG_ID,
                    Symbol::SIN_ID,
                    Symbol::COS_ID,
                    Symbol::ABS_ID,
                    Symbol::CONJ_ID,
                ]
                .contains(&name.get_id())
                {
                    if f.get_nargs() != 1 {
                        return Err(EvaluationError::WrongNumberOfArguments {
                            function: name,
                            expected: 1,
                            actual: f.get_nargs(),
                        });
                    }

                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.linearize_impl(
                        fn_map,
                        context,
                        params,
                        settings,
                        instr,
                        subexpressions,
                    )?;

                    let temp = Slot::Temp(instr.len());
                    let c = Instruction::Fun(temp, Box::new((name, vec![], vec![arg_eval])), false);
                    instr.push(c);

                    subexpressions.insert(*self, temp);
                    return Ok(temp);
                }

                let fun = if name == Symbol::IF {
                    if f.get_nargs() != 3 {
                        return Err(EvaluationError::WrongNumberOfArguments {
                            function: name,
                            expected: 3,
                            actual: f.get_nargs(),
                        });
                    }

                    let mut arg_iter = f.iter();
                    let cond = arg_iter.next().unwrap();
                    let then_branch = arg_iter.next().unwrap();
                    let else_branch = arg_iter.next().unwrap();

                    let instr_len = instr.len();
                    let subexpression_len = subexpressions.len();
                    let cond = cond.linearize_impl(
                        fn_map,
                        context,
                        params,
                        settings,
                        instr,
                        subexpressions,
                    )?;

                    // try to resolve the condition if it is fully numeric
                    fn resolve(
                        cond: Slot,
                        instr: &[Instruction],
                        constants: &[Option<Complex<Rational>>],
                    ) -> Option<Complex<Rational>> {
                        let i = match cond {
                            Slot::Param(_) => {
                                return None;
                            }
                            Slot::Const(i) => {
                                return constants[i].clone();
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
                                    if *exp == -1 {
                                        Some(base_val.inv())
                                    } else if *exp < 0 {
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

                    if let Some(cond_res) = resolve(cond, instr, &context.constants) {
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
                                context,
                                params,
                                settings,
                                instr,
                                subexpressions,
                            )?
                        } else {
                            else_branch.linearize_impl(
                                fn_map,
                                context,
                                params,
                                settings,
                                instr,
                                subexpressions,
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
                        context,
                        params,
                        settings,
                        instr,
                        &mut sub_expr_pos_child,
                    )?;

                    let label_end_pos = instr.len();
                    instr.push(Instruction::Goto(0));
                    instr[if_instr_pos] = Instruction::IfElse(cond, instr.len());
                    instr.push(Instruction::Label(instr.len()));

                    sub_expr_pos_child.clone_from(subexpressions);
                    let else_branch = else_branch.linearize_impl(
                        fn_map,
                        context,
                        params,
                        settings,
                        instr,
                        &mut sub_expr_pos_child,
                    )?;

                    instr[label_end_pos] = Instruction::Goto(instr.len());
                    instr.push(Instruction::Label(instr.len()));

                    let temp = Slot::Temp(instr.len());
                    instr.push(Instruction::Join(temp, cond, then_branch, else_branch));
                    return Ok(temp);
                } else if let Some(fun) = fn_map.get(*self) {
                    fun
                } else if let Some(eval_info) = name.get_evaluation_info() {
                    let tags = f
                        .iter()
                        .take(eval_info.get_tag_count())
                        .map(|x| x.to_owned())
                        .collect::<Vec<_>>();

                    let mut fixed_args = vec![];
                    for x in f.iter().skip(eval_info.get_tag_count()) {
                        if let Ok(n) = Complex::<Rational>::try_from(x) {
                            fixed_args.push(n);
                        }
                    }

                    // check if it a constant external function
                    if f.get_nargs() == eval_info.get_tag_count() + fixed_args.len() {
                        return Ok(context.external_constant_slot(name, tags, fixed_args));
                    };

                    let eval_args = f
                        .iter()
                        .skip(eval_info.get_tag_count())
                        .map(|arg| {
                            arg.linearize_impl(
                                fn_map,
                                context,
                                params,
                                settings,
                                instr,
                                subexpressions,
                            )
                        })
                        .collect::<Result<_, _>>()?;

                    let temp = Slot::Temp(instr.len());
                    instr.push(Instruction::Fun(
                        temp,
                        Box::new((
                            name,
                            tags.iter().map(|x| x.to_canonical_string()).collect(),
                            eval_args,
                        )),
                        false,
                    ));
                    // Use the same scope-local cache as arithmetic expressions.
                    // This branch returns before the common insertion below.
                    subexpressions.insert(*self, temp);
                    return Ok(temp);
                } else {
                    return Err(EvaluationError::UndefinedFunction {
                        expression: self.to_owned(),
                    });
                };

                {
                    let Expr {
                        tag_len,
                        args: arg_spec,
                        options,
                        ..
                    } = fun;

                    if f.get_nargs() != arg_spec.len() + tag_len {
                        return Err(EvaluationError::WrongNumberOfArguments {
                            function: f.get_symbol(),
                            expected: arg_spec.len() + tag_len,
                            actual: f.get_nargs(),
                        });
                    }

                    let mut arg_slots = Vec::with_capacity(arg_spec.len());
                    for eval_arg in f.iter().skip(*tag_len) {
                        let slot = eval_arg.linearize_impl(
                            fn_map,
                            context,
                            params,
                            settings,
                            instr,
                            subexpressions,
                        )?;

                        arg_slots.push(slot);
                    }

                    if !fun.is_alias && !options.should_inline() {
                        let tags = f.iter().take(*tag_len).map(|tag| tag.to_owned()).collect();
                        return Self::linearize_function_call(
                            fn_map, context, settings, name, tags, fun, arg_slots, instr,
                        );
                    }

                    self.linearize_inline_function(
                        fun,
                        arg_slots,
                        fn_map,
                        context,
                        params,
                        settings,
                        instr,
                        subexpressions,
                    )?
                }
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();

                if b == InlineVar::new(Symbol::E).as_view() {
                    let e_eval =
                        e.linearize_impl(fn_map, context, params, settings, instr, subexpressions)?;

                    let temp = Slot::Temp(instr.len());
                    let c = Instruction::Fun(
                        temp,
                        Box::new((Symbol::EXP, vec![], vec![e_eval])),
                        false,
                    );
                    instr.push(c);

                    subexpressions.insert(*self, temp);
                    return Ok(temp);
                }

                let b_eval =
                    b.linearize_impl(fn_map, context, params, settings, instr, subexpressions)?;

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

                if let Ok(r) = Rational::try_from(e) {
                    // FIXME: support 3,2 etc as well
                    if r == (1, 2) {
                        let temp = Slot::Temp(instr.len());
                        let c = Instruction::Fun(
                            temp,
                            Box::new((Symbol::SQRT, vec![], vec![b_eval])),
                            false,
                        );
                        instr.push(c);

                        subexpressions.insert(*self, temp);
                        return Ok(temp);
                    }
                }

                let e_eval =
                    e.linearize_impl(fn_map, context, params, settings, instr, subexpressions)?;

                let temp = Slot::Temp(instr.len());
                instr.push(Instruction::Powf(temp, b_eval, e_eval, false));
                temp
            }
            AtomView::Mul(m) => {
                let mut muls = vec![];
                for arg in m.iter() {
                    let a = arg.linearize_impl(
                        fn_map,
                        context,
                        params,
                        settings,
                        instr,
                        subexpressions,
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
                        context,
                        params,
                        settings,
                        instr,
                        subexpressions,
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
    /// A cached hash and a numerical constant.
    Const(ExpressionHash, Box<T>),
    /// A cached hash and an index into the evaluator's parameter storage.
    Parameter(ExpressionHash, usize),
    /// A cached hash, an index of a defined function, and its argument trees.
    Eval(ExpressionHash, u32, Vec<Expression<T>>),
    /// A cached hash and the terms of a sum.
    Add(ExpressionHash, Vec<Expression<T>>),
    /// A cached hash and the factors of a product.
    Mul(ExpressionHash, Vec<Expression<T>>),
    /// A cached hash and a base raised to an integer exponent.
    Pow(ExpressionHash, Box<(Expression<T>, i64)>),
    /// A cached hash and a pair of base and exponent expressions.
    Powf(ExpressionHash, Box<(Expression<T>, Expression<T>)>),
    /// A cached hash and a zero-based index of an argument of the current function.
    ReadArg(ExpressionHash, usize), // read nth function argument
    /// A cached hash, a built-in function symbol, and its argument.
    BuiltinFun(ExpressionHash, Symbol, Box<Expression<T>>),
    /// A cached hash, an external function symbol, its symbolic tags, and argument trees.
    Fun(ExpressionHash, Symbol, Vec<String>, Vec<Expression<T>>),
    /// A cached hash and a condition, true branch, and false branch.
    /// A zero condition selects the false branch.
    IfElse(
        ExpressionHash,
        Box<(Expression<T>, Expression<T>, Expression<T>)>,
    ),
    /// A cached hash and an index into the extracted common subexpressions.
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
            Expression::Fun(h, _, _, _) => *h,
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
            (Expression::Fun(_, a, b, c), Expression::Fun(_, d, e, f)) => {
                a.cmp(d).then_with(|| b.cmp(e)).then_with(|| c.cmp(f))
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
            (Expression::Fun(_, _, _, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Fun(_, _, _, _)) => std::cmp::Ordering::Greater,
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
            Expression::Add(_, a) | Expression::Mul(_, a) | Expression::Fun(_, _, _, a) => {
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
            Expression::Add(_, a) | Expression::Mul(_, a) | Expression::Fun(_, _, _, a) => {
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

    // Count the number of operations in the expression, counting
    // subexpressions only once.
    /// Count arithmetic operations, counting each distinct subexpression once.
    /// `sub_expr` records visited expressions and can be shared across calls to
    /// measure the combined cost of several trees.
    pub fn count_operations_with_subexpression<'a>(
        &'a self,
        sub_expr: &mut HashMap<&'a Self, usize>,
    ) -> OperationCount {
        if matches!(
            self,
            Expression::Const(_, _) | Expression::Parameter(_, _) | Expression::ReadArg(_, _)
        ) {
            return OperationCount::default();
        }

        if sub_expr.contains_key(self) {
            return OperationCount::default();
        }

        sub_expr.insert(self, 1);

        match self {
            Expression::Const(_, _) => OperationCount::default(),
            Expression::Parameter(_, _) => OperationCount::default(),
            Expression::Eval(_, _, args) => {
                let mut count = OperationCount::default();
                for arg in args {
                    count += arg.count_operations_with_subexpression(sub_expr);
                }
                count
            }
            Expression::Add(_, a) => {
                let mut count = OperationCount::default();
                for arg in a {
                    count += arg.count_operations_with_subexpression(sub_expr);
                }
                count.additions += a.len() - 1;
                count
            }
            Expression::Mul(_, m) => {
                let mut count = OperationCount::default();
                for arg in m {
                    count += arg.count_operations_with_subexpression(sub_expr);
                }
                count.multiplications += m.len() - 1;
                count
            }
            Expression::Pow(_, p) => {
                let mut count = p.0.count_operations_with_subexpression(sub_expr);
                count.add_integer_power(p.1);
                count
            }
            Expression::Powf(_, p) => {
                let mut count = p.0.count_operations_with_subexpression(sub_expr)
                    + p.1.count_operations_with_subexpression(sub_expr);
                count.add_function_call();
                count
            }
            Expression::ReadArg(_, _) => OperationCount::default(),
            Expression::BuiltinFun(_, _, b) => {
                let mut count = b.count_operations_with_subexpression(sub_expr);
                count.add_function_call();
                count
            }
            Expression::SubExpression(_, _) => OperationCount::default(),
            Expression::Fun(_, _, _, a) => {
                let mut count = OperationCount::default();
                for arg in a {
                    count += arg.count_operations_with_subexpression(sub_expr);
                }
                count.add_function_call();
                count
            }
            Expression::IfElse(_, b) => {
                b.0.count_operations_with_subexpression(sub_expr)
                    + b.1.count_operations_with_subexpression(sub_expr)
                    + b.2.count_operations_with_subexpression(sub_expr)
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
            Expression::Fun(h, s, tags, a) => {
                if partial && *h != 0 {
                    return *h;
                }

                let mut hasher = AHasher::default();
                hasher.write_u8(10);
                s.hash(&mut hasher);
                tags.hash(&mut hasher);
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
            Expression::Fun(h, s, tags, a) => {
                let new_args = a.iter().map(|x| x.map_coeff(f)).collect();
                Expression::Fun(*h, *s, tags.clone(), new_args)
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
}

impl Expression<Complex<Rational>> {
    fn strip_constants(
        &mut self,
        constants: &mut Vec<Complex<Rational>>,
        param_count: usize,
        external_functions: &mut Vec<ExternalFunctionContainer<Complex<Rational>>>,
    ) {
        match self {
            Expression::Const(_, t) => {
                if let Some(p) = constants.iter().position(|x| x == &**t) {
                    *self = Expression::Parameter(0, param_count + p);
                } else {
                    constants.push(t.as_ref().clone());
                    *self = Expression::Parameter(0, param_count + constants.len() - 1);
                }
            }
            Expression::Parameter(_, _) => {}
            Expression::Eval(_, _, e_args) => {
                for a in e_args {
                    a.strip_constants(constants, param_count, external_functions);
                }
            }
            Expression::Add(_, a) | Expression::Mul(_, a) => {
                for arg in a {
                    arg.strip_constants(constants, param_count, external_functions);
                }
            }
            Expression::Pow(_, p) => {
                p.0.strip_constants(constants, param_count, external_functions);
            }
            Expression::Powf(_, p) => {
                p.0.strip_constants(constants, param_count, external_functions);
                p.1.strip_constants(constants, param_count, external_functions);
            }
            Expression::ReadArg(_, _) => {}
            Expression::BuiltinFun(_, _, a) => {
                a.strip_constants(constants, param_count, external_functions);
            }
            Expression::SubExpression(_, _) => {}
            Expression::Fun(_, symbol, tags, a) => {
                let fixed_args = a
                    .iter()
                    .map(|arg| match arg {
                        Expression::Const(_, c) => Some(c.as_ref().clone()),
                        _ => None,
                    })
                    .collect::<Option<Vec<_>>>();

                if let Some(fixed_args) = fixed_args {
                    let tags = tags.iter().map(|x| crate::parse!(x)).collect::<Vec<_>>();
                    let i = register_constant_external_container(
                        external_functions,
                        *symbol,
                        tags,
                        fixed_args,
                        constants,
                    );
                    *self = Expression::Parameter(0, param_count + i);
                    return;
                }

                for arg in a {
                    arg.strip_constants(constants, param_count, external_functions);
                }
            }
            Expression::IfElse(_, b) => {
                b.0.strip_constants(constants, param_count, external_functions);
                b.1.strip_constants(constants, param_count, external_functions);
                b.2.strip_constants(constants, param_count, external_functions);
            }
        }
    }
}

impl<T: Clone + PartialEq> EvalTree<T> {
    /// Convert all numerical coefficients with `f`, preserving the expression
    /// structure, parameter order, and function definitions.
    pub fn map_coeff<T2: EvaluationDomain, F: Fn(&T) -> T2>(&self, f: &F) -> EvalTree<T2> {
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
            external_functions: self.external_functions.iter().map(|x| x.map()).collect(),
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
        let mut evaluator = self.linearize_without_stack_optimization(&settings);
        evaluator.optimize_stack();

        evaluator
    }

    fn linearize_without_stack_optimization(
        &mut self,
        settings: &OptimizationSettings,
    ) -> ExpressionEvaluator<Complex<Rational>> {
        // strip every constant and move them into the stack after the params
        let mut constants = vec![];
        self.strip_constants(&mut constants, self.param_count);

        let mut stack = vec![Complex::<Rational>::default(); self.param_count];
        stack.extend(constants);
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

        // Resolving branch destinations is required lowering, even when all
        // optional CSE/CPE stages are interrupted by cancellation.
        e.fix_labels();
        AtomView::optimize_evaluator_instructions(e)
    }

    fn strip_constants(&mut self, constants: &mut Vec<Complex<Rational>>, param_count: usize) {
        for t in &mut self.expressions.tree {
            t.strip_constants(constants, param_count, &mut self.external_functions);
        }

        for e in &mut self.expressions.subexpressions {
            e.strip_constants(constants, param_count, &mut self.external_functions);
        }

        for (_, _, e) in &mut self.functions {
            for t in &mut e.tree {
                t.strip_constants(constants, param_count, &mut self.external_functions);
            }

            for e in &mut e.subexpressions {
                e.strip_constants(constants, param_count, &mut self.external_functions);
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
            Expression::Fun(_, s, tags, v) => {
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

                let tag_atoms = tags.iter().map(|x| crate::parse!(x)).collect::<Vec<_>>();
                let index = self
                    .external_functions
                    .iter()
                    .position(|x| {
                        x.symbol == *s
                            && x.tags == tag_atoms
                            && x.fixed_args.is_empty()
                            && x.constant_index.is_none()
                    })
                    .expect("missing external function container");

                let f = Instr::ExternalFun(res, index, args);
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
                                if *exp == -1 {
                                    Some(base_val.inv())
                                } else if *exp < 0 {
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
            let count = self.count_operations();
            info!("Initial ops: {}", count);
        }

        if settings.horner_use_cpe && settings.horner_iterations > 1 {
            let (_, mut evaluator) = self.optimize_horner_scheme_with_cpe(settings);
            evaluator.settings = settings.clone();
            evaluator.settings.abort_level = 0;
            evaluator.optimize_stack();

            return evaluator;
        }

        let (search_settings, _) = latched_horner_settings(settings);
        if settings.horner_iterations > 0 {
            let _ = self.optimize_horner_scheme(&search_settings);
        }

        self.common_subexpression_elimination();
        let mut evaluator = self.linearize(&search_settings);
        evaluator.settings = settings.clone();
        evaluator.settings.abort_level = 0;

        evaluator
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
        if settings.horner_use_cpe && settings.horner_iterations > 1 {
            return self.optimize_horner_scheme_with_cpe(settings).0;
        }

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

    /// Preserve the legacy path's independent root/function searches, but score
    /// each candidate in the complete program where its function calls occur.
    fn optimize_horner_scheme_with_cpe(
        &mut self,
        settings: &OptimizationSettings,
    ) -> (
        Vec<Expression<Complex<Rational>>>,
        ExpressionEvaluator<Complex<Rational>>,
    ) {
        let (search_settings, cancelled) = latched_horner_settings(settings);
        let mut candidate_settings = search_settings.clone();
        candidate_settings.verbose = false;
        let mut root_scheme = Vec::new();
        let mut final_evaluator = None;

        for scope in 0..=self.functions.len() {
            if cancelled.load(Ordering::Relaxed) && final_evaluator.is_some() {
                break;
            }
            let expressions = if scope == 0 {
                &self.expressions
            } else {
                &self.functions[scope - 1].2
            };
            let vars = if scope == 0 && settings.hot_start.is_some() {
                settings.hot_start.clone().unwrap()
            } else {
                let mut counts = HashMap::default();
                for expression in expressions.tree.iter().chain(&expressions.subexpressions) {
                    expression.find_all_variables(&mut counts);
                }
                let mut counts: Vec<_> = counts.into_iter().collect();
                counts.retain(|(_, count)| *count > 1);
                counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
                counts.truncate(settings.max_horner_scheme_variables);
                counts
                    .into_iter()
                    .map(|(variable, _)| variable)
                    .collect::<Vec<_>>()
            };
            // An empty scheme leaves this scope unchanged, so a previously
            // compiled winner is still valid.
            if vars.is_empty() && final_evaluator.is_some() {
                continue;
            }
            let best = Mutex::new(
                None::<(
                    OperationCount,
                    Vec<Expression<Complex<Rational>>>,
                    ExpressionEvaluator<Complex<Rational>>,
                )>,
            );
            let score = |order: &[Expression<Complex<Rational>>]| {
                if cancelled.load(Ordering::Relaxed) && best.lock().unwrap().is_some() {
                    return OperationCount::new(0, usize::MAX, 0, 0);
                }
                let mut candidate = self.clone();
                candidate.apply_horner_scheme_to_scope(scope, order);
                candidate.common_subexpression_elimination();
                let evaluator = candidate.linearize_without_stack_optimization(&candidate_settings);
                let cost = evaluator_horner_cost(&evaluator);
                let mut winner = best.lock().unwrap();
                if winner.as_ref().is_none_or(|(best_cost, best_order, _)| {
                    horner_cost(cost) < horner_cost(*best_cost)
                        || horner_cost(cost) == horner_cost(*best_cost)
                            && order < best_order.as_slice()
                }) {
                    *winner = Some((cost, order.to_vec(), evaluator));
                }
                cost
            };
            let scheme = if vars.len() < 2 {
                score(&vars);
                vars
            } else {
                search_horner_order(&vars, &search_settings, score)
            };
            if let Some((_, _, evaluator)) = best.into_inner().unwrap() {
                self.apply_horner_scheme_to_scope(scope, &scheme);
                if scope == 0 {
                    root_scheme = scheme;
                }
                final_evaluator = Some(evaluator);
            }
        }
        (
            root_scheme,
            final_evaluator.unwrap_or_else(|| {
                let mut candidate = self.clone();
                candidate.common_subexpression_elimination();
                candidate.linearize_without_stack_optimization(&search_settings)
            }),
        )
    }

    fn apply_horner_scheme_to_scope(
        &mut self,
        scope: usize,
        scheme: &[Expression<Complex<Rational>>],
    ) {
        let expressions = if scope == 0 {
            &mut self.expressions
        } else {
            &mut self.functions[scope - 1].2
        };
        for expression in expressions
            .tree
            .iter_mut()
            .chain(&mut expressions.subexpressions)
        {
            expression.apply_horner_scheme(scheme);
        }
    }
}

impl Expression<Complex<Rational>> {
    /// Rewrite sums using a Horner scheme in the supplied variable order.
    /// The entries in `scheme` identify the subexpressions to factor out.
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
            Expression::Fun(_, _, _, a) => {
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

    /// Search for a Horner variable order that reduces this expression's operation
    /// count. `vars` supplies the initial order and `settings` bounds the search.
    pub fn optimize_horner_scheme(
        &self,
        vars: &[Self],
        settings: &OptimizationSettings,
    ) -> Vec<Self> {
        Self::optimize_horner_scheme_multiple(std::slice::from_ref(self), vars, settings)
    }

    /// Search for a shared Horner variable order that reduces the combined
    /// operation count of `expressions`, accounting for common subexpressions.
    /// `vars` supplies the initial order and `settings` bounds the search.
    pub fn optimize_horner_scheme_multiple(
        expressions: &[Self],
        vars: &[Self],
        settings: &OptimizationSettings,
    ) -> Vec<Self> {
        search_horner_order(vars, settings, |order| {
            let hornered: Vec<_> = expressions
                .iter()
                .map(|expression| {
                    let mut hornered = expression.clone();
                    hornered.apply_horner_scheme(order);
                    hornered.rehash(true);
                    hornered
                })
                .collect();
            let mut subexpr = HashMap::default();
            let mut operations = OperationCount::default();
            for expression in &hornered {
                operations += expression.count_operations_with_subexpression(&mut subexpr);
            }
            operations
        })
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
            Expression::Fun(_, _, _, a) => {
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
    /// Extract repeated subexpressions from the outputs and each function body
    /// so that subsequent evaluation can reuse their values.
    pub fn common_subexpression_elimination(&mut self) {
        self.expressions.common_subexpression_elimination();

        for (_, _, e) in &mut self.functions {
            e.common_subexpression_elimination();
        }
    }

    /// Count arithmetic operations in the outputs, extracted subexpressions,
    /// and function definitions, counting each stored tree once.
    pub fn count_operations(&self) -> OperationCount {
        let mut count = OperationCount::default();
        for e in &self.functions {
            count += e.2.count_operations();
        }

        count + self.expressions.count_operations()
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
            h.insert(e, self.subexpressions.len() + index);
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
            Expression::Fun(_, _, _, a) => {
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
            Expression::Fun(_, _, _, a) => {
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
    pub fn count_operations(&self) -> OperationCount {
        let mut count = OperationCount::default();
        for e in &self.subexpressions {
            count += e.count_operations();
        }

        for e in &self.tree {
            count += e.count_operations();
        }

        count
    }
}

impl<T: Clone + Default + std::fmt::Debug + Eq + std::hash::Hash + InternalOrdering> Expression<T> {
    // Count the number of operations in the expression.
    /// Count arithmetic operations recursively, including repeated occurrences
    /// of the same subtree. Use [`Self::count_operations_with_subexpression`]
    /// to count shared subexpressions only once.
    pub fn count_operations(&self) -> OperationCount {
        match self {
            Expression::Const(_, _) => OperationCount::default(),
            Expression::Parameter(_, _) => OperationCount::default(),
            Expression::Eval(_, _, args) => {
                let mut count = OperationCount::default();
                for arg in args {
                    count += arg.count_operations();
                }
                count
            }
            Expression::Add(_, a) => {
                let mut count = OperationCount::default();
                for arg in a {
                    count += arg.count_operations();
                }
                count.additions += a.len() - 1;
                count
            }
            Expression::Mul(_, m) => {
                let mut count = OperationCount::default();
                for arg in m {
                    count += arg.count_operations();
                }
                count.multiplications += m.len() - 1;
                count
            }
            Expression::Pow(_, p) => {
                let mut count = p.0.count_operations();
                count.add_integer_power(p.1);
                count
            }
            Expression::Powf(_, p) => {
                let mut count = p.0.count_operations() + p.1.count_operations();
                count.add_function_call();
                count
            }
            Expression::ReadArg(_, _) => OperationCount::default(),
            Expression::BuiltinFun(_, _, b) => {
                let mut count = b.count_operations();
                count.add_function_call();
                count
            }
            Expression::SubExpression(_, _) => OperationCount::default(),
            Expression::Fun(_, _, _, args) => {
                let mut count = OperationCount::default();
                for arg in args {
                    count += arg.count_operations();
                }
                count.add_function_call();
                count
            }
            Expression::IfElse(_, b) => {
                b.0.count_operations() + b.1.count_operations() + b.2.count_operations()
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

                if *e == -1 {
                    b_eval.inv()
                } else if *e >= 0 {
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
                match s.get_id() {
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
            Expression::Fun(_, name, _, _args) => {
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

impl<'a> AtomView<'a> {
    /// Convert nested expressions to a tree.
    pub fn to_evaluation_tree(
        &self,
        fn_map: &FunctionMap,
        params: &[Atom],
    ) -> Result<EvalTree<Complex<Rational>>, EvaluationError> {
        Self::to_eval_tree_multiple(std::slice::from_ref(self), fn_map, params)
    }

    /// Convert nested expressions to a tree.
    pub fn to_eval_tree_multiple<A: AtomCore>(
        exprs: &[A],
        fn_map: &FunctionMap,
        params: &[Atom],
    ) -> Result<EvalTree<Complex<Rational>>, EvaluationError> {
        let mut funcs = vec![];
        let mut func_id_to_index = HashMap::default();
        let mut external_functions = vec![];

        let tree = exprs
            .iter()
            .map(|t| {
                t.as_atom_view()
                    .to_eval_tree_impl(
                        fn_map,
                        params,
                        &[],
                        &mut func_id_to_index,
                        &mut funcs,
                        &mut external_functions,
                    )
                    .map(|x| x.rehashed(true))
            })
            .collect::<Result<_, _>>()?;

        Ok(EvalTree {
            expressions: SplitExpression {
                tree,
                subexpressions: vec![],
            },
            functions: funcs,
            external_functions,
            param_count: params.len(),
        })
    }

    fn to_eval_tree_impl(
        &self,
        fn_map: &FunctionMap,
        params: &[Atom],
        args: &[Indeterminate],
        fn_id_map: &mut HashMap<usize, Option<usize>>,
        funcs: &mut Vec<(
            String,
            Vec<Indeterminate>,
            SplitExpression<Complex<Rational>>,
        )>,
        external_functions: &mut Vec<ExternalFunctionContainer<Complex<Rational>>>,
    ) -> Result<Expression<Complex<Rational>>, EvaluationError> {
        if matches!(self, AtomView::Var(_) | AtomView::Fun(_)) {
            if let Some(p) = args.iter().position(|s| *self == s.as_view()) {
                return Ok(Expression::ReadArg(0, p));
            }

            if let Some(p) = params.iter().position(|a| a.as_view() == *self) {
                return Ok(Expression::Parameter(0, p));
            }
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
                CoefficientView::Indeterminate => Err(EvaluationError::UnsupportedCoefficient {
                    coefficient: "indeterminate".to_owned(),
                }),
                CoefficientView::Infinity(_) => Err(EvaluationError::UnsupportedCoefficient {
                    coefficient: "infinity".to_owned(),
                }),
                CoefficientView::FiniteField(_, _) => {
                    Err(EvaluationError::UnsupportedCoefficient {
                        coefficient: "finite field".to_owned(),
                    })
                }
                CoefficientView::RationalPolynomial(_) => {
                    Err(EvaluationError::UnsupportedCoefficient {
                        coefficient: "rational polynomial".to_owned(),
                    })
                }
            },
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if let Some(expr) = fn_map.get(*self) {
                    if !expr.args.is_empty() {
                        return Err(EvaluationError::WrongNumberOfArguments {
                            function: name,
                            expected: expr.args.len(),
                            actual: 0,
                        });
                    }
                    if fn_id_map.get(&expr.id) == Some(&None) {
                        return Err(EvaluationError::EvaluatorConstructionFailed {
                            expression: self.to_owned(),
                            reason: "recursive function or alias dependency".into(),
                        });
                    }
                    let previous = fn_id_map.insert(expr.id, None);
                    let result = expr.body.as_view().to_eval_tree_impl(
                        fn_map,
                        params,
                        if expr.is_alias { args } else { &[] },
                        fn_id_map,
                        funcs,
                        external_functions,
                    );
                    if let Some(index) = previous {
                        fn_id_map.insert(expr.id, index);
                    } else {
                        fn_id_map.remove(&expr.id);
                    }
                    return result;
                }

                if name.get_evaluation_info().is_some() {
                    if external_functions
                        .iter()
                        .all(|x| x.symbol != name || !x.tags.is_empty())
                    {
                        external_functions.push(ExternalFunctionContainer::new(
                            name,
                            vec![],
                            vec![],
                        ));
                    }

                    return Ok(Expression::Fun(0, name, vec![], vec![]));
                }

                Err(EvaluationError::UndefinedVariable {
                    symbol: v.get_symbol(),
                })
            }
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [
                    Symbol::LOG_ID,
                    Symbol::SIN_ID,
                    Symbol::COS_ID,
                    Symbol::ABS_ID,
                    Symbol::CONJ_ID,
                ]
                .contains(&name.get_id())
                {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.to_eval_tree_impl(
                        fn_map,
                        params,
                        args,
                        fn_id_map,
                        funcs,
                        external_functions,
                    )?;

                    return Ok(Expression::BuiltinFun(
                        0,
                        f.get_symbol(),
                        Box::new(arg_eval),
                    ));
                }

                if name == Symbol::IF {
                    if f.get_nargs() != 3 {
                        return Err(EvaluationError::WrongNumberOfArguments {
                            function: name,
                            expected: 3,
                            actual: f.get_nargs(),
                        });
                    }

                    let mut arg_iter = f.iter();
                    let cond_eval = arg_iter.next().unwrap().to_eval_tree_impl(
                        fn_map,
                        params,
                        args,
                        fn_id_map,
                        funcs,
                        external_functions,
                    )?;

                    if let Expression::Const(0, c) = &cond_eval {
                        if !c.is_zero() {
                            return arg_iter.next().unwrap().to_eval_tree_impl(
                                fn_map,
                                params,
                                args,
                                fn_id_map,
                                funcs,
                                external_functions,
                            );
                        }

                        let _ = arg_iter.next().unwrap();
                        return arg_iter.next().unwrap().to_eval_tree_impl(
                            fn_map,
                            params,
                            args,
                            fn_id_map,
                            funcs,
                            external_functions,
                        );
                    }

                    let t_eval = arg_iter.next().unwrap().to_eval_tree_impl(
                        fn_map,
                        params,
                        args,
                        fn_id_map,
                        funcs,
                        external_functions,
                    )?;
                    let f_eval = arg_iter.next().unwrap().to_eval_tree_impl(
                        fn_map,
                        params,
                        args,
                        fn_id_map,
                        funcs,
                        external_functions,
                    )?;

                    return Ok(Expression::IfElse(0, Box::new((cond_eval, t_eval, f_eval))));
                }

                if let Some(Expr {
                    id,
                    tag_len,
                    is_alias,
                    args: arg_spec,
                    body: e,
                    ..
                }) = fn_map.get(*self)
                {
                    return {
                        if f.get_nargs() != arg_spec.len() + tag_len {
                            return Err(EvaluationError::WrongNumberOfArguments {
                                function: f.get_symbol(),
                                expected: arg_spec.len() + tag_len,
                                actual: f.get_nargs(),
                            });
                        }

                        // Pending entries detect cycles before expanding aliases or bodies.
                        if fn_id_map.get(id) == Some(&None) {
                            return Err(EvaluationError::EvaluatorConstructionFailed {
                                expression: self.to_owned(),
                                reason: "recursive function or alias dependency".into(),
                            });
                        }
                        if *is_alias {
                            fn_id_map.insert(*id, None);
                            let result = e.as_view().to_eval_tree_impl(
                                fn_map,
                                params,
                                args,
                                fn_id_map,
                                funcs,
                                external_functions,
                            );
                            fn_id_map.remove(id);
                            return result;
                        }

                        let eval_args = f
                            .iter()
                            .skip(*tag_len)
                            .map(|arg| {
                                arg.to_eval_tree_impl(
                                    fn_map,
                                    params,
                                    args,
                                    fn_id_map,
                                    funcs,
                                    external_functions,
                                )
                            })
                            .collect::<Result<_, _>>()?;

                        if let Some(Some(pos)) = fn_id_map.get(id) {
                            Ok(Expression::Eval(0, *pos as u32, eval_args))
                        } else {
                            fn_id_map.insert(*id, None);
                            let r = e.as_view().to_eval_tree_impl(
                                fn_map,
                                params,
                                arg_spec,
                                fn_id_map,
                                funcs,
                                external_functions,
                            )?;
                            funcs.push((
                                name.get_name().to_owned(),
                                arg_spec.clone(),
                                SplitExpression {
                                    tree: vec![r.clone()],
                                    subexpressions: vec![],
                                },
                            ));
                            fn_id_map.insert(*id, Some(funcs.len() - 1));
                            Ok(Expression::Eval(0, funcs.len() as u32 - 1, eval_args))
                        }
                    };
                }

                if let Some(eval_info) = name.get_evaluation_info() {
                    let tags = f
                        .iter()
                        .take(eval_info.get_tag_count())
                        .map(|x| x.to_canonical_string())
                        .collect::<Vec<_>>();
                    let tag_atoms = tags.iter().map(|x| crate::parse!(x)).collect::<Vec<_>>();

                    if external_functions
                        .iter()
                        .all(|x| x.symbol != name || x.tags != tag_atoms)
                    {
                        external_functions.push(ExternalFunctionContainer::new(
                            name,
                            tag_atoms,
                            vec![],
                        ));
                    }

                    let eval_args = f
                        .iter()
                        .skip(eval_info.get_tag_count())
                        .map(|arg| {
                            arg.to_eval_tree_impl(
                                fn_map,
                                params,
                                args,
                                fn_id_map,
                                funcs,
                                external_functions,
                            )
                        })
                        .collect::<Result<_, _>>()?;
                    return Ok(Expression::Fun(0, name, tags, eval_args));
                }

                Err(EvaluationError::UndefinedFunction {
                    expression: self.to_owned(),
                })
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();

                if b == InlineVar::new(Symbol::E).as_view() {
                    let e_eval = e.to_eval_tree_impl(
                        fn_map,
                        params,
                        args,
                        fn_id_map,
                        funcs,
                        external_functions,
                    )?;
                    return Ok(Expression::BuiltinFun(0, Symbol::EXP, Box::new(e_eval)));
                }

                let b_eval = b.to_eval_tree_impl(
                    fn_map,
                    params,
                    args,
                    fn_id_map,
                    funcs,
                    external_functions,
                )?;

                if let AtomView::Num(n) = e
                    && let CoefficientView::Natural(num, den, num_i, _den_i) = n.get_coeff_view()
                    && den == 1
                    && num_i == 0
                {
                    return Ok(Expression::Pow(0, Box::new((b_eval.clone(), num))));
                }

                let e_eval = e.to_eval_tree_impl(
                    fn_map,
                    params,
                    args,
                    fn_id_map,
                    funcs,
                    external_functions,
                )?;
                Ok(Expression::Powf(0, Box::new((b_eval, e_eval))))
            }
            AtomView::Mul(m) => {
                let mut muls = vec![];
                for arg in m.iter() {
                    let a = arg.to_eval_tree_impl(
                        fn_map,
                        params,
                        args,
                        fn_id_map,
                        funcs,
                        external_functions,
                    )?;
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
                    adds.push(arg.to_eval_tree_impl(
                        fn_map,
                        params,
                        args,
                        fn_id_map,
                        funcs,
                        external_functions,
                    )?);
                }

                adds.sort();

                Ok(Expression::Add(0, adds))
            }
        }
    }

    pub(crate) fn evaluate<A: AtomCore + KeyLookup, T: Real + EvaluationDomain>(
        &self,
        map: &StdHashMap<A, T, impl BuildHasher>,
        binary_prec: u32,
    ) -> Result<T, EvaluationError> {
        let mut cache = HashMap::default();
        self.evaluate_impl(map, &mut cache, binary_prec)
    }

    fn evaluate_impl<A: AtomCore + KeyLookup, T: Real + EvaluationDomain>(
        &self,
        map: &StdHashMap<A, T, impl BuildHasher>,
        cache: &mut HashMap<AtomView<'a>, T>,
        binary_prec: u32,
    ) -> Result<T, EvaluationError> {
        match self {
            AtomView::Num(n) => {
                let f = n
                    .get_coeff_view()
                    .to_float(binary_prec)
                    .map_err(|e| EvaluationError::UnsupportedCoefficient { coefficient: e })?;
                T::try_from_complex_float(f).map_err(|e| EvaluationError::EvaluationFailed {
                    expression: self.to_owned(),
                    reason: e,
                })
            }
            AtomView::Var(v) => {
                let s = v.get_symbol();

                if let Some(val) = map.get::<[u8]>(self.get_data()) {
                    return Ok(val.clone());
                }

                if let Some(val) = cache.get(self) {
                    return Ok(val.clone());
                }

                if let Some(eval) = s.get_evaluation_info() {
                    if let Ok(val) = T::try_from_complex_float(
                        eval.evaluate_constant(&[], binary_prec).map_err(|e| {
                            EvaluationError::UnsupportedCoefficient { coefficient: e }
                        })?,
                    ) {
                        cache.insert(*self, val.clone());
                        return Ok(val);
                    } else {
                        return Err(EvaluationError::MissingEvaluator {
                            expression: self.to_owned(),
                            eval_type: std::any::type_name::<T>().to_string(),
                        });
                    }
                }

                Err(EvaluationError::UndefinedVariable { symbol: s })
            }
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [
                    Symbol::LOG_ID,
                    Symbol::SIN_ID,
                    Symbol::COS_ID,
                    Symbol::ABS_ID,
                    Symbol::CONJ_ID,
                ]
                .contains(&name.get_id())
                {
                    if f.get_nargs() != 1 {
                        return Err(EvaluationError::WrongNumberOfArguments {
                            function: name,
                            expected: 1,
                            actual: f.get_nargs(),
                        });
                    }

                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.evaluate_impl(map, cache, binary_prec)?;

                    return Ok(match f.get_symbol_id() {
                        Symbol::LOG_ID => arg_eval.log(),
                        Symbol::SIN_ID => arg_eval.sin(),
                        Symbol::COS_ID => arg_eval.cos(),
                        Symbol::ABS_ID => arg_eval.norm(),
                        Symbol::CONJ_ID => arg_eval.conj(),
                        _ => unreachable!(),
                    });
                }

                if name == Symbol::IF {
                    if f.get_nargs() != 3 {
                        return Err(EvaluationError::WrongNumberOfArguments {
                            function: name,
                            expected: 3,
                            actual: f.get_nargs(),
                        });
                    }

                    let mut arg_iter = f.iter();

                    let cond_eval =
                        arg_iter
                            .next()
                            .unwrap()
                            .evaluate_impl(map, cache, binary_prec)?;

                    if !cond_eval.is_fully_zero() {
                        let t_eval =
                            arg_iter
                                .next()
                                .unwrap()
                                .evaluate_impl(map, cache, binary_prec)?;
                        return Ok(t_eval);
                    } else {
                        let _ = arg_iter.next().unwrap();
                        let f_eval =
                            arg_iter
                                .next()
                                .unwrap()
                                .evaluate_impl(map, cache, binary_prec)?;
                        return Ok(f_eval);
                    }
                }

                if let Some(val) = map.get::<[u8]>(self.get_data()) {
                    return Ok(val.clone());
                }

                if let Some(eval) = cache.get(self) {
                    return Ok(eval.clone());
                }

                if let Some(eval) = name.get_evaluation_info() {
                    let tag_count = eval.get_tag_count();
                    let mut iter = f.iter();
                    let mut tags = Vec::with_capacity(tag_count);

                    for i in 0..tag_count {
                        if let Some(tag) = iter.next() {
                            tags.push(tag);
                        } else {
                            return Err(EvaluationError::WrongNumberOfArguments {
                                function: name,
                                expected: tag_count,
                                actual: i,
                            });
                        }
                    }

                    let mut args = Vec::with_capacity(f.get_nargs());
                    for arg in iter {
                        args.push(arg.evaluate_impl(map, cache, binary_prec)?);
                    }

                    if args.is_empty() && eval.has_constant_evaluator() {
                        let val = T::try_from_complex_float(
                            eval.evaluate_constant(&tags, binary_prec).map_err(|e| {
                                EvaluationError::EvaluationFailed {
                                    expression: self.to_owned(),
                                    reason: e,
                                }
                            })?,
                        )
                        .map_err(|_| {
                            EvaluationError::MissingEvaluator {
                                expression: self.to_owned(),
                                eval_type: std::any::type_name::<T>().to_string(),
                            }
                        })?;
                        cache.insert(*self, val.clone());
                        return Ok(val);
                    }

                    if let Some(eval_fun) = T::resolve_function(&tags, eval) {
                        let val = (eval_fun)(&args);
                        cache.insert(*self, val.clone());
                        return Ok(val);
                    } else {
                        return Err(EvaluationError::MissingEvaluator {
                            expression: self.to_owned(),
                            eval_type: std::any::type_name::<T>().to_string(),
                        });
                    }
                }

                Err(EvaluationError::UndefinedFunction {
                    expression: self.to_owned(),
                })
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();

                if b == InlineVar::new(Symbol::E).as_view() {
                    let e_eval = e.evaluate_impl(map, cache, binary_prec)?;
                    return Ok(e_eval.exp());
                }

                let b_eval = b.evaluate_impl(map, cache, binary_prec)?;

                if let AtomView::Num(n) = e
                    && let CoefficientView::Natural(num, den, ni, _di) = n.get_coeff_view()
                    && ni == 0
                {
                    if den == 1 {
                        if num == -1 {
                            return Ok(b_eval.inv());
                        } else if num >= 0 {
                            return Ok(b_eval.pow(num as u64));
                        } else {
                            return Ok(b_eval.pow(num.unsigned_abs()).inv());
                        }
                    } else if den == 2 {
                        if num == -1 {
                            return Ok(b_eval.sqrt().inv());
                        } else if num >= 0 {
                            return Ok(b_eval.sqrt().pow(num as u64));
                        } else {
                            return Ok(b_eval.sqrt().pow(num.unsigned_abs()).inv());
                        }
                    }
                }

                let e_eval = e.evaluate_impl(map, cache, binary_prec)?;
                Ok(b_eval.powf(&e_eval))
            }
            AtomView::Mul(m) => {
                let mut it = m.iter();
                let mut r = it.next().unwrap().evaluate_impl(map, cache, binary_prec)?;
                for arg in it {
                    r *= arg.evaluate_impl(map, cache, binary_prec)?;
                }
                Ok(r)
            }
            AtomView::Add(a) => {
                let mut it = a.iter();
                let mut r = it.next().unwrap().evaluate_impl(map, cache, binary_prec)?;
                for arg in it {
                    r += arg.evaluate_impl(map, cache, binary_prec)?;
                }
                Ok(r)
            }
        }
    }

    pub(crate) fn evaluate_in_ring<A: AtomCore + KeyLookup, R: ConvertToRing>(
        &self,
        map: &StdHashMap<A, R::Element, impl BuildHasher>,
        ring: &R,
    ) -> Result<R::Element, EvaluationError> {
        self.evaluate_in_ring_impl(map, &mut HashMap::new(), ring)
    }

    fn evaluate_in_ring_impl<A: AtomCore + KeyLookup, R: ConvertToRing>(
        &self,
        map: &StdHashMap<A, R::Element, impl BuildHasher>,
        cache: &mut HashMap<AtomView<'a>, R::Element>,
        ring: &R,
    ) -> Result<R::Element, EvaluationError> {
        match self {
            AtomView::Num(n) => ring
                .try_element_from_coefficient_view(n.get_coeff_view())
                .map_err(|e| EvaluationError::UnsupportedCoefficient { coefficient: e }),
            AtomView::Var(v) => {
                let s = v.get_symbol();

                if let Some(val) = map.get::<[u8]>(self.get_data()) {
                    return Ok(val.clone());
                }

                if let Some(val) = cache.get(self) {
                    return Ok(val.clone());
                }

                Err(EvaluationError::UndefinedVariable { symbol: s })
            }
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [
                    Symbol::LOG_ID,
                    Symbol::SIN_ID,
                    Symbol::COS_ID,
                    Symbol::ABS_ID,
                    Symbol::CONJ_ID,
                ]
                .contains(&name.get_id())
                {
                    return Err(EvaluationError::UnsupportedBuiltinArity {
                        function: name,
                        expected: 1,
                        actual: f.get_nargs(),
                    });
                }

                if name == Symbol::IF {
                    if f.get_nargs() != 3 {
                        return Err(EvaluationError::WrongNumberOfArguments {
                            function: name,
                            expected: 3,
                            actual: f.get_nargs(),
                        });
                    }

                    let mut arg_iter = f.iter();

                    let cond_eval = arg_iter
                        .next()
                        .unwrap()
                        .evaluate_in_ring_impl(map, cache, ring)?;

                    if !ring.is_zero(&cond_eval) {
                        let t_eval = arg_iter
                            .next()
                            .unwrap()
                            .evaluate_in_ring_impl(map, cache, ring)?;
                        return Ok(t_eval);
                    } else {
                        let _ = arg_iter.next().unwrap();
                        let f_eval = arg_iter
                            .next()
                            .unwrap()
                            .evaluate_in_ring_impl(map, cache, ring)?;
                        return Ok(f_eval);
                    }
                }

                if let Some(val) = map.get::<[u8]>(self.get_data()) {
                    return Ok(val.clone());
                }

                if let Some(eval) = cache.get(self) {
                    return Ok(eval.clone());
                }

                Err(EvaluationError::UndefinedFunction {
                    expression: self.to_owned(),
                })
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let mut b_eval = b.evaluate_in_ring_impl(map, cache, ring)?;

                if let AtomView::Num(n) = e
                    && let CoefficientView::Natural(num, den, ni, _di) = n.get_coeff_view()
                    && den == 1
                    && ni == 0
                {
                    if num < 0 {
                        b_eval = ring.try_inv(&b_eval).ok_or_else(|| {
                            EvaluationError::EvaluationFailed {
                                expression: self.to_owned(),
                                reason: format!("Could not invert {}", ring.printer(&b_eval)),
                            }
                        })?;
                    }

                    let num = num.unsigned_abs();

                    if num != 1 {
                        Ok(ring.pow(&b_eval, num as u64))
                    } else {
                        Ok(b_eval)
                    }
                } else {
                    Err(EvaluationError::EvaluationFailed {
                        expression: self.to_owned(),
                        reason: format!("Cannot evaluate {}", self),
                    })
                }
            }
            AtomView::Mul(m) => {
                let mut it = m.iter();
                let mut r = it.next().unwrap().evaluate_in_ring_impl(map, cache, ring)?;
                for arg in it {
                    ring.mul_assign(&mut r, arg.evaluate_in_ring_impl(map, cache, ring)?);
                }
                Ok(r)
            }
            AtomView::Add(a) => {
                let mut it = a.iter();
                let mut r = it.next().unwrap().evaluate_in_ring_impl(map, cache, ring)?;
                for arg in it {
                    ring.add_assign(&mut r, arg.evaluate_in_ring_impl(map, cache, ring)?);
                }
                Ok(r)
            }
        }
    }
}

#[cfg(test)]
mod post_cpe_cse_tests {
    use super::*;

    type Expr = Expression<Complex<Rational>>;

    fn parameter(index: usize) -> Expr {
        Expression::Parameter(0, index)
    }

    fn sum(terms: Vec<Expr>) -> Expr {
        Expression::Add(0, terms)
    }

    fn sine(argument: Expr) -> Expr {
        Expression::BuiltinFun(0, Symbol::SIN, Box::new(argument))
    }

    fn equivalent_sums() -> [Expr; 2] {
        [
            sum(vec![parameter(0), parameter(1), parameter(2)]),
            sum(vec![sum(vec![parameter(0), parameter(1)]), parameter(2)]),
        ]
    }

    fn build(
        expressions: Vec<Expr>,
        param_count: usize,
        cpe_iterations: Option<usize>,
    ) -> ExpressionEvaluator<Complex<Rational>> {
        EvalTree {
            functions: vec![],
            external_functions: vec![],
            expressions: SplitExpression {
                tree: expressions,
                subexpressions: vec![],
            },
            param_count,
        }
        .linearize(&OptimizationSettings {
            cpe_iterations,
            ..Default::default()
        })
    }

    #[test]
    fn pair_extraction_exposes_duplicate_functions_in_tree() {
        // Different addition trees initially hide the identical sine calls.
        // Extracting (a+b) makes both argument DAGs identical, even with one
        // CPE round. CSE must then propagate that identity through the sine.
        for rounds in [Some(1), None] {
            let mut evaluator = build(equivalent_sums().map(sine).to_vec(), 3, rounds)
                .map_coeff(&|coefficient| coefficient.re.to_f64());
            assert_eq!(evaluator.count_operations().function_calls, 1);
            assert_eq!(evaluator.count_operations().additions, 2);
            for point in [[0.2, 0.3, 0.4], [-0.7, 0.1, -0.2], [0., 0., 0.]] {
                let mut outputs = [0.; 2];
                evaluator.evaluate(&point, &mut outputs);
                let expected = (point[0] + point[1] + point[2]).sin();
                assert_eq!(outputs[0], outputs[1]);
                assert!((outputs[0] - expected).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn pair_extraction_exposes_duplicate_functions_after_merge() {
        let [flat, nested] = equivalent_sums();
        let mut evaluator = build(vec![sine(flat)], 3, Some(0));
        let other = build(vec![sine(nested)], 3, Some(0));
        evaluator.merge(other, Some(1)).unwrap();
        assert_eq!(evaluator.count_operations().function_calls, 1);
        let mut evaluator = evaluator.map_coeff(&|coefficient| coefficient.re.to_f64());
        let mut outputs = [0.; 2];
        evaluator.evaluate(&[0.2, 0.3, 0.4], &mut outputs);
        assert_eq!(outputs[0], outputs[1]);
        assert!((outputs[0] - 0.9_f64.sin()).abs() < 1e-14);
    }

    #[test]
    fn pair_extraction_merges_functions_only_in_dominating_branches() {
        let branch = || sum(equivalent_sums().map(sine).to_vec());
        let conditional = Expression::IfElse(
            0,
            Box::new((parameter(3), branch(), sum(vec![branch(), parameter(4)]))),
        );
        let mut evaluator =
            build(vec![conditional], 5, None).map_coeff(&|coefficient| coefficient.re.to_f64());
        // Each branch can share its two sine calls, but a sine evaluated only
        // in the true branch cannot supply the false branch (or vice versa).
        assert_eq!(evaluator.count_operations().function_calls, 2);
        for point in [
            [0.2, 0.3, 0.4, 0., 1.25],
            [-0.7, 0.1, -0.2, 1., 1.25],
            [0.6, -0.2, 0.1, 0., -0.5],
            [0., 0., 0., 1., 7.],
        ] {
            let extra = if point[3] == 0. { point[4] } else { 0. };
            let expected = 2. * (point[0] + point[1] + point[2]).sin() + extra;
            assert!((evaluator.evaluate_single(&point) - expected).abs() < 1e-14);
        }
    }
}

#[cfg(test)]
mod external_cache_tests {
    use crate::{
        atom::{Atom, AtomCore, EvaluationInfo, Symbol},
        domains::float::{Complex, Real},
        evaluate::OptimizationSettings,
        function, symbol,
    };
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };

    fn settings() -> OptimizationSettings {
        OptimizationSettings::new()
            .direct_translation(true)
            .horner_iterations(0)
            .cpe_iterations(Some(0))
    }

    fn counted<T: Real>(calls: Arc<AtomicUsize>) -> impl Fn(&[T]) -> T + Clone {
        move |arguments: &[T]| {
            calls.fetch_add(1, Ordering::SeqCst);
            arguments[0].clone() * &arguments[0]
        }
    }

    fn square(name: &str) -> (Symbol, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let f = symbol!(
            name,
            eval = EvaluationInfo::new()
                .register(counted::<f64>(calls.clone()))
                .register(counted::<Complex<f64>>(calls.clone()))
        );
        (f, calls)
    }

    #[test]
    fn direct_external_calls_share_across_vector_outputs_without_optimizer() {
        let (f, calls) = square("direct_external_cache::vector_f");
        let x = Atom::var(symbol!("direct_external_cache::vector_x"));
        let value = function!(f, &x);
        let exact = Atom::evaluator_multiple(
            &[value.clone(), &value + 1, 3 * &value],
            std::slice::from_ref(&x),
        )
        .optimization_settings(settings())
        .build()
        .unwrap();
        let mut real = exact.clone().map_coeff(&|c| c.re.to_f64());
        let mut output = [0.; 3];
        real.evaluate(&[2.], &mut output);
        assert_eq!(output, [4., 5., 12.]);
        assert_eq!(calls.swap(0, Ordering::SeqCst), 1);
        real.evaluate(&[3.], &mut output);
        assert_eq!(output, [9., 10., 27.]);
        assert_eq!(calls.swap(0, Ordering::SeqCst), 1);
        let mut complex = exact.map_coeff(&|c| Complex::new(c.re.to_f64(), c.im.to_f64()));
        let mut output = [Complex::new(0., 0.); 3];
        complex.evaluate(&[Complex::new(2., 1.)], &mut output);
        assert_eq!(output[0], Complex::new(3., 4.));
        assert_eq!(output[1], Complex::new(4., 4.));
        assert_eq!(output[2], Complex::new(9., 12.));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }
}

#[cfg(test)]
mod evaluator_horner_candidate_tests {
    use super::*;

    #[test]
    fn horner_orders_borrow_the_input_storage() {
        let source = crate::parse!("a*f(x^4+x^3+x^2+x+1)+b*f(x^4+x^3+x^2+x+1)+c*sin(x)+d*sin(x)");
        let view = source.as_view();
        let storage = view.get_data();
        let start = storage.as_ptr() as usize;
        let end = start + storage.len();
        let orders = AtomView::horner_orders(
            &[view],
            &HashSet::default(),
            &OptimizationSettings::default(),
        );
        assert_eq!(orders[0].len(), 2);
        for order in orders {
            for atom in order {
                let data = atom.get_data();
                let address = data.as_ptr() as usize;
                assert!(address >= start && address + data.len() <= end);
            }
        }
    }

    #[test]
    fn unchanged_regions_and_repeated_arguments_remain_borrowed() {
        let source = crate::parse!("sin(x+1)+cos(x+1)+x^2*y+z^5");
        let tags = HashMap::default();
        let settings = OptimizationSettings::default();
        let mut regions = HornerRegions::new(
            &tags,
            &settings,
            Arc::new(AtomicBool::new(false)),
            [],
            &[source.as_view()],
        );
        assert!(matches!(
            regions.children(source.as_view()),
            AtomOrView::View(view) if view.get_data().as_ptr() == source.as_view().get_data().as_ptr()
        ));
        let argument = crate::parse!("x+1");
        assert!(matches!(
            regions.optimized.get(&argument.as_view()),
            Some(AtomOrView::View(_))
        ));
    }

    #[test]
    fn two_candidate_budget_compares_seeds_with_multiplication_priority() {
        let visited = Mutex::new(Vec::new());
        let original = [0, 1, 2];
        let alternative = [2, 1, 0];
        let result = search_horner_order_with_seed(
            &original,
            Some(&alternative),
            &OptimizationSettings {
                horner_iterations: 2,
                ..Default::default()
            },
            |order| {
                visited.lock().unwrap().push(order.to_vec());
                if order == alternative {
                    OperationCount::new(100, 1, 0, 0)
                } else {
                    OperationCount::new(0, 2, 0, 0)
                }
            },
        );
        assert_eq!(result, alternative);
        assert_eq!(
            visited.into_inner().unwrap(),
            [original.to_vec(), alternative.to_vec()]
        );
    }

    #[test]
    fn duplicate_seed_does_not_consume_an_extra_candidate() {
        let visited = Mutex::new(Vec::new());
        let original = [0, 1, 2];
        search_horner_order_with_seed(
            &original,
            Some(&original),
            &OptimizationSettings {
                horner_iterations: 2,
                n_cores: 1,
                ..Default::default()
            },
            |order| {
                visited.lock().unwrap().push(order.to_vec());
                OperationCount::default()
            },
        );
        let visited = visited.into_inner().unwrap();
        assert_eq!(visited.len(), 2);
        assert_eq!(visited[0], original);
        assert_ne!(visited[1], original);
    }

    #[test]
    fn exhaustive_search_with_two_seeds_visits_each_order_once() {
        let visited = Mutex::new(Vec::new());
        let original = [3, 2, 1, 0];
        let alternative = [0, 2, 3, 1];
        let result = search_horner_order_with_seed(
            &original,
            Some(&alternative),
            &OptimizationSettings {
                horner_iterations: 24,
                n_cores: 3,
                ..Default::default()
            },
            |order| {
                visited.lock().unwrap().push(order.to_vec());
                OperationCount::default()
            },
        );
        let mut visited = visited.into_inner().unwrap();
        assert_eq!(visited.len(), 24);
        visited.sort();
        visited.dedup();
        assert_eq!(visited.len(), 24);
        assert!(visited.contains(&original.to_vec()));
        assert!(visited.contains(&alternative.to_vec()));
        assert_eq!(result, [0, 1, 2, 3]);
    }

    #[test]
    fn equal_cost_seeds_choose_same_lexicographic_winner() {
        let first = [2, 1, 0];
        let second = [0, 2, 1];
        for (original, alternative) in [(&first, &second), (&second, &first)] {
            let result = search_horner_order_with_seed(
                original,
                Some(alternative),
                &OptimizationSettings {
                    horner_iterations: 2,
                    ..Default::default()
                },
                |_| OperationCount::new(3, 5, 0, 0),
            );
            assert_eq!(result, second);
        }
    }

    #[test]
    fn discovery_counts_computed_values_but_not_builtin_constants() {
        let source = crate::parse!("a*exp(x+y)+b*exp(x+y)+c*sin(x)+d*sin(x)+x^4+pi");
        let exponential = crate::parse!("exp(x+y)");
        let sine = crate::parse!("sin(x)");
        let integer_power = crate::parse!("x^4");
        let x = crate::parse!("x");
        let y = crate::parse!("y");
        let e = Atom::var(Symbol::E);
        let pi = Atom::var(Symbol::PI);
        let mut candidates = HashMap::default();
        source
            .as_view()
            .count_horner_candidates(&HashSet::default(), &mut candidates);
        assert_eq!(candidates.get(&exponential.as_view()), Some(&2));
        assert_eq!(candidates.get(&sine.as_view()), Some(&2));
        assert_eq!(candidates.get(&x.as_view()), Some(&1));
        assert!(!candidates.contains_key(&y.as_view()));
        assert!(!candidates.contains_key(&integer_power.as_view()));
        assert!(!candidates.contains_key(&e.as_view()));
        assert!(!candidates.contains_key(&pi.as_view()));
    }

    #[test]
    fn discovery_keeps_explicit_parameters_opaque_even_for_builtin_constants() {
        let parameter = crate::parse!("f(x+x*y)");
        let source = crate::parse!("a*f(x+x*y)+b*f(x+x*y)");
        let parameters = [parameter.as_view()].into_iter().collect();
        let mut candidates = HashMap::default();
        source
            .as_view()
            .count_horner_candidates(&parameters, &mut candidates);
        assert_eq!(candidates.get(&parameter.as_view()), Some(&2));
        assert!(!candidates.contains_key(&crate::parse!("x").as_view()));
        assert!(!candidates.contains_key(&crate::parse!("y").as_view()));

        let e = Atom::var(Symbol::E);
        let pi = Atom::var(Symbol::PI);
        let source = &e + &pi;
        let parameters = [e.as_view(), pi.as_view()].into_iter().collect();
        let mut candidates = HashMap::default();
        source
            .as_view()
            .count_horner_candidates(&parameters, &mut candidates);
        assert_eq!(candidates.get(&e.as_view()), Some(&1));
        assert_eq!(candidates.get(&pi.as_view()), Some(&1));
    }

    #[test]
    fn direct_builder_preserves_single_occurrence_function_parameter() {
        let parameter = crate::parse!("f(x+x*y)");
        let source = crate::parse!("(x+x*y)*f(x+x*y)");
        let parameters = [parameter, crate::parse!("x"), crate::parse!("y")];
        for iterations in [1, 10] {
            let mut evaluator = source
                .evaluator(&parameters)
                .optimization_settings(OptimizationSettings {
                    direct_translation: true,
                    horner_iterations: iterations,
                    ..Default::default()
                })
                .build()
                .unwrap()
                .map_coeff(&|coefficient| coefficient.re.to_f64());
            for point in [[2., 0.5, -0.25], [3., -2., 0.25], [0., 7., 2.]] {
                let expected = point[0] * (point[1] + point[1] * point[2]);
                assert!((evaluator.evaluate_single(&point) - expected).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn direct_builder_factors_exponential_and_preserves_numerical_values() {
        let source = crate::parse!("a*exp(x)+b*exp(x)+c");
        let parameters = [
            crate::parse!("x"),
            crate::parse!("a"),
            crate::parse!("b"),
            crate::parse!("c"),
        ];
        let original = source
            .evaluator(&parameters)
            .optimization_settings(OptimizationSettings {
                direct_translation: true,
                horner_iterations: 0,
                ..Default::default()
            })
            .build()
            .unwrap();
        assert_eq!(original.count_operations().multiplications, 2);
        for iterations in [1, 2, 10] {
            let mut evaluator = source
                .evaluator(&parameters)
                .optimization_settings(OptimizationSettings {
                    direct_translation: true,
                    horner_iterations: iterations,
                    ..Default::default()
                })
                .build()
                .unwrap()
                .map_coeff(&|coefficient| coefficient.re.to_f64());
            let operations = evaluator.count_operations();
            assert_eq!(operations.multiplications, 1);
            assert_eq!(operations.function_calls, 1);
            for point in [
                [-2., 2., -3., 0.7],
                [0., -1., 4., 2.],
                [0.5, 3., 5., -2.],
                [2., 0.25, -0.5, -1.],
            ] {
                let expected = (point[1] + point[2]) * point[0].exp() + point[3];
                assert!((evaluator.evaluate_single(&point) - expected).abs() < 1e-13);
            }
        }
    }
}

#[cfg(test)]
mod local_function_parameter_scope_tests {
    use super::*;

    fn check_local_function_parameter(inherited_alias: bool) {
        let source = crate::parse!("g(u,x,y)");
        let root_parameters = [crate::parse!("u"), crate::parse!("x"), crate::parse!("y")];
        let declared_parameters: Vec<Indeterminate> = [
            crate::parse!("h(2*x+2*y)"),
            crate::parse!("x"),
            crate::parse!("y"),
        ]
        .into_iter()
        .map(|argument| argument.try_into().unwrap())
        .collect();
        let body = crate::parse!("(x+x*y)*h(2*x+2*y)");
        let alias = crate::parse!("local_parameter_scope_alias");
        for policy in [InliningPolicy::Always, InliningPolicy::Never] {
            for iterations in [0, 1, 10] {
                let function_body = if inherited_alias { &alias } else { &body };
                let mut builder = source
                    .evaluator(&root_parameters)
                    .add_function_with_options(
                        crate::symbol!("g"),
                        declared_parameters.clone(),
                        function_body.clone(),
                        FunctionRegistrationOptions::new().inlining(policy),
                    )
                    .unwrap()
                    .optimization_settings(OptimizationSettings {
                        direct_translation: true,
                        horner_iterations: iterations,
                        ..Default::default()
                    });
                if inherited_alias {
                    builder = builder
                        .add_aliases([(alias.clone(), body.clone())])
                        .unwrap();
                }
                let mut evaluator = builder
                    .build()
                    .unwrap()
                    .map_coeff(&|coefficient| coefficient.re.to_f64());
                for point in [[2., 3., 4.], [-3., 0.5, -0.25], [0., 7., 2.]] {
                    let expected = point[0] * (point[1] + point[1] * point[2]);
                    assert!((evaluator.evaluate_single(&point) - expected).abs() < 1e-14);
                }
            }
        }
    }

    #[test]
    fn inline_and_deferred_function_arguments_keep_their_atom_identity() {
        check_local_function_parameter(false);
    }

    #[test]
    fn inherited_aliases_preserve_inline_and_deferred_function_arguments() {
        check_local_function_parameter(true);
    }
}

#[cfg(test)]
mod optimized_horner_search_tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn mixed_mutations_preserve_permutations_and_include_larger_moves() {
        let original: Vec<_> = (0..12).collect();
        let mut rng = MonteCarloRng::new(0, 0);
        let mut large_insert = false;
        let mut large_reversal = false;
        for mutation in 0..128 {
            let mut order = original.clone();
            mutate_horner_order(&mut order, &mut rng, mutation);
            assert_ne!(order, original);
            let changed = order.iter().zip(&original).filter(|(a, b)| a != b).count();
            large_insert |= mutation % 4 == 2 && changed > 2;
            large_reversal |= mutation % 4 == 3 && changed > 2;
            order.sort_unstable();
            assert_eq!(order, original);
        }
        assert!(large_insert && large_reversal);
    }

    #[test]
    fn multiple_seeds_are_unique_and_charged_against_the_budget() {
        for budget in [2, 3] {
            let visited = Mutex::new(Vec::new());
            let initial = [0, 1, 2, 3];
            let second = [1, 0, 2, 3];
            let third = [3, 2, 1, 0];
            let selected = search_horner_order_with_seeds(
                &initial,
                &[&initial, &second, &second, &third],
                &OptimizationSettings {
                    horner_iterations: budget,
                    ..Default::default()
                },
                |order| {
                    visited.lock().unwrap().push(order.to_vec());
                    OperationCount::new(0, if order == third { 1 } else { 2 }, 0, 0)
                },
            );
            let visited = visited.into_inner().unwrap();
            assert_eq!(visited.len(), budget);
            assert_eq!(visited[0], initial);
            assert_eq!(visited[1], second);
            if budget == 3 {
                assert_eq!(visited[2], third);
                assert_eq!(selected, third);
            }
        }
    }

    #[test]
    fn long_search_restarts_without_exceeding_budget_or_losing_the_winner() {
        let initial: Vec<_> = (0..10).collect();
        let visited = Mutex::new(Vec::new());
        let selected = search_horner_order(
            &initial,
            &OptimizationSettings {
                horner_iterations: 130,
                n_cores: 1,
                ..Default::default()
            },
            |order| {
                visited.lock().unwrap().push(order.to_vec());
                OperationCount::new(0, usize::from(order != initial), 0, 0)
            },
        );
        let visited = visited.into_inner().unwrap();
        assert_eq!(visited.len(), 130);
        assert_eq!(
            visited[34],
            initial.iter().rev().copied().collect::<Vec<_>>()
        );
        assert_eq!(selected, initial);
    }

    #[test]
    fn finished_program_score_counts_binary_powering() {
        // A lowered instruction fixture keeps powers explicit: the Atom
        // linearizer expands positive integer powers into multiplication lists.
        let mut evaluator = ExpressionEvaluator {
            stack: vec![0_f64; 5],
            param_count: 2,
            reserved_indices: 2,
            instructions: vec![
                (Instr::Pow(2, 0, 7), ComplexPhase::default()),
                (Instr::Pow(3, 1, -4), ComplexPhase::default()),
                (Instr::Add(4, vec![2, 3]), ComplexPhase::default()),
            ],
            result_indices: vec![4],
            external_fns: vec![],
            settings: OptimizationSettings::default(),
        };
        let cost = evaluator_horner_cost(&evaluator);
        assert_eq!(cost.multiplications, 6);
        assert_eq!(cost.additions, 1);
        assert_eq!(cost.inversions, 1);
        assert_eq!(evaluator.evaluate_single(&[2., 2.]), 128.0625);
    }

    #[test]
    fn search_selects_fewer_multiplications_after_pair_elimination() {
        let parameters = [
            crate::parse!("horner_cost_a"),
            crate::parse!("horner_cost_b"),
            crate::parse!("horner_cost_c"),
        ];
        let permutations = unique_permutations(&[0_usize, 1, 2]).1;
        let settings = OptimizationSettings {
            horner_iterations: 6,
            horner_use_cpe: true,
            ..Default::default()
        };
        let mut strict_misranking = false;
        // Coefficient sign normalization depends on the canonical variable
        // order. Check every relabeling of the same small polynomial instead
        // of relying on symbols registered by another parallel test.
        for relabeling in &permutations {
            let [a, b, c] =
                std::array::from_fn::<_, 3, _>(|i| parameters[relabeling[i]].to_canonical_string());
            let source = crate::parse!(&format!(
                "{b}*{c}^3-3*{b}^2-{a}*{c}^2+2*{a}^2*{c}-2*{a}^2*{b}^2-3*{a}^3*{c}+2*{a}^3*{c}^2+3*{a}^3*{b}^2*{c}-2*{a}^3*{b}^2*{c}^2"
            ));
            let expressions = [source.as_view()];
            let mut outcomes = Vec::new();
            for permutation in &permutations {
                let order = permutation
                    .iter()
                    .map(|&i| parameters[i].clone())
                    .collect::<Vec<_>>();
                let before = horner_cost(AtomView::horner_scheme_cost(&expressions, &order));
                let evaluator = AtomView::build_horner_evaluator(
                    &expressions,
                    &FunctionMap::new(),
                    &parameters,
                    &order,
                    settings.clone(),
                )
                .unwrap();
                outcomes.push((
                    before,
                    horner_cost(evaluator_horner_cost(&evaluator)),
                    order,
                ));
            }
            let cheap_winner = outcomes
                .iter()
                .min_by(|left, right| left.0.cmp(&right.0).then_with(|| left.2.cmp(&right.2)))
                .unwrap();
            let finished_minimum = outcomes.iter().map(|outcome| outcome.1).min().unwrap();
            strict_misranking |= cheap_winner.1.0 > finished_minimum.0;
            for horner_use_cpe in [false, true] {
                let evaluator = source
                    .evaluator(&parameters)
                    .optimization_settings(settings.clone().horner_use_cpe(horner_use_cpe))
                    .build()
                    .unwrap();
                assert_eq!(
                    horner_cost(evaluator_horner_cost(&evaluator)),
                    if horner_use_cpe {
                        finished_minimum
                    } else {
                        cheap_winner.1
                    }
                );
                let mut evaluator = evaluator.map_coeff(&|coefficient| coefficient.re.to_f64());
                for point in [[1.2_f64, -0.3, 0.7], [-2., 1.5, -0.2], [0., 3., 2.]] {
                    let [a, b, c] = std::array::from_fn::<_, 3, _>(|i| point[relabeling[i]]);
                    let expected = b * c.powi(3) - 3. * b.powi(2) - a * c.powi(2)
                        + 2. * a.powi(2) * c
                        - 2. * a.powi(2) * b.powi(2)
                        - 3. * a.powi(3) * c
                        + 2. * a.powi(3) * c.powi(2)
                        + 3. * a.powi(3) * b.powi(2) * c
                        - 2. * a.powi(3) * b.powi(2) * c.powi(2);
                    assert!((evaluator.evaluate_single(&point) - expected).abs() < 1e-12);
                }
            }
        }
        assert!(
            strict_misranking,
            "the fixture must expose a strictly worse pre-CPE winner"
        );
    }

    #[test]
    fn legacy_function_search_scores_all_calls_and_reuses_the_winner() {
        let parameters = [
            crate::parse!("legacy_score_x"),
            crate::parse!("legacy_score_y"),
            crate::parse!("legacy_score_z"),
        ];
        let body = crate::parse!(
            "legacy_score_b*legacy_score_c^3-3*legacy_score_b^2\
            -legacy_score_a*legacy_score_c^2+2*legacy_score_a^2*legacy_score_c\
            -2*legacy_score_a^2*legacy_score_b^2-3*legacy_score_a^3*legacy_score_c\
            +2*legacy_score_a^3*legacy_score_c^2\
            +3*legacy_score_a^3*legacy_score_b^2*legacy_score_c\
            -2*legacy_score_a^3*legacy_score_b^2*legacy_score_c^2"
        );
        let source = crate::parse!(
            "legacy_score_f(legacy_score_x,legacy_score_y,legacy_score_z)\
            +legacy_score_f(legacy_score_y,legacy_score_x,legacy_score_z)"
        );
        let mut functions = FunctionMap::new();
        functions
            .add_function(
                crate::symbol!("legacy_score_f"),
                vec![
                    crate::symbol!("legacy_score_a"),
                    crate::symbol!("legacy_score_b"),
                    crate::symbol!("legacy_score_c"),
                ],
                body,
            )
            .unwrap();
        let tree = source
            .as_view()
            .to_evaluation_tree(&functions, &parameters)
            .unwrap();
        assert_eq!(tree.functions.len(), 1);
        let mut variables = HashMap::default();
        for expression in &tree.functions[0].2.tree {
            expression.find_all_variables(&mut variables);
        }
        let mut variables = variables
            .into_iter()
            .filter(|(_, count)| *count > 1)
            .map(|(variable, _)| variable)
            .collect::<Vec<_>>();
        variables.sort();
        assert_eq!(variables.len(), 3);
        let settings = OptimizationSettings {
            direct_translation: false,
            horner_use_cpe: true,
            horner_iterations: 6,
            ..Default::default()
        };
        let minimum = unique_permutations(&variables)
            .1
            .into_iter()
            .map(|order| {
                let mut candidate = tree.clone();
                candidate.apply_horner_scheme_to_scope(1, &order);
                candidate.common_subexpression_elimination();
                horner_cost(evaluator_horner_cost(&candidate.linearize(&settings)))
            })
            .min()
            .unwrap();
        let mut selected_tree = tree.clone();
        let (_, cached) = selected_tree.optimize_horner_scheme_with_cpe(&settings);
        assert_eq!(horner_cost(evaluator_horner_cost(&cached)), minimum);
        // The tree returned by the public scheme API must agree with the cached
        // full program even though only the latter has undergone CSE and CPE.
        selected_tree.common_subexpression_elimination();
        assert_eq!(
            horner_cost(evaluator_horner_cost(&selected_tree.linearize(&settings))),
            minimum
        );
        let evaluator = tree.optimize(&settings);
        assert_eq!(horner_cost(evaluator_horner_cost(&evaluator)), minimum);
        let mut evaluator = evaluator.map_coeff(&|coefficient| coefficient.re.to_f64());
        let expected_body = |a: f64, b: f64, c: f64| {
            b * c.powi(3) - 3. * b.powi(2) - a * c.powi(2) + 2. * a.powi(2) * c
                - 2. * a.powi(2) * b.powi(2)
                - 3. * a.powi(3) * c
                + 2. * a.powi(3) * c.powi(2)
                + 3. * a.powi(3) * b.powi(2) * c
                - 2. * a.powi(3) * b.powi(2) * c.powi(2)
        };
        for point in [[1.2, -0.3, 0.7], [-2., 1.5, -0.2], [0., 3., 2.]] {
            let [a, b, c] = point;
            let expected = expected_body(a, b, c) + expected_body(b, a, c);
            assert!((evaluator.evaluate_single(&point) - expected).abs() < 1e-11);
        }
    }

    #[test]
    fn candidate_build_errors_are_returned() {
        let source = crate::parse!("unregistered_horner_function(x+y)+x*y");
        for direct_translation in [false, true] {
            for horner_use_cpe in [false, true] {
                let result = source
                    .evaluator(&[crate::parse!("x"), crate::parse!("y")])
                    .optimization_settings(OptimizationSettings {
                        horner_iterations: 1000,
                        horner_use_cpe,
                        direct_translation,
                        n_cores: 3,
                        ..Default::default()
                    })
                    .build();
                assert!(matches!(
                    result,
                    Err(EvaluationError::UndefinedFunction { .. })
                ));
            }
        }
    }

    #[test]
    fn cancellation_is_latched_and_returns_a_working_candidate() {
        let source = crate::parse!("x*y+x*z+y*z+x*y*z");
        for direct_translation in [false, true] {
            for horner_use_cpe in [false, true] {
                let polls = Arc::new(AtomicUsize::new(0));
                let callback_polls = polls.clone();
                let mut evaluator = source
                    .evaluator(&[crate::parse!("x"), crate::parse!("y"), crate::parse!("z")])
                    .optimization_settings(OptimizationSettings {
                        horner_iterations: 1000,
                        horner_use_cpe,
                        direct_translation,
                        n_cores: 3,
                        // A one-shot callback tests the latch rather than relying on
                        // a caller that keeps returning true after cancellation.
                        abort_check: Some(Box::new(move || {
                            callback_polls.fetch_add(1, Ordering::Relaxed) == 0
                        })),
                        ..Default::default()
                    })
                    .build()
                    .unwrap()
                    .map_coeff(&|coefficient| coefficient.re.to_f64());
                assert_eq!(polls.load(Ordering::Relaxed), 1);
                assert_eq!(evaluator.evaluate_single(&[2., 3., 4.]), 50.);
            }
        }
    }
}
