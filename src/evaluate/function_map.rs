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
/// use symbolica::{atom::AtomCore, parse, symbol};
/// use symbolica::evaluate::{FunctionMap, OptimizationSettings};
/// let mut fn_map = FunctionMap::new();
/// fn_map.add_function(symbol!("f"), vec![symbol!("x")], parse!("x^2 + 1")).unwrap();
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
pub struct FunctionMap {
    pub(super) map: HashMap<Atom, Expr>,
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
            map: HashMap::default(),
            tagged_fn_map: HashMap::default(),
            tag: HashMap::default(),
        }
    }

    /// Register a function. If `name` is a symbol, it will be treated as a regular function; if it is a function, its arguments will be treated as tags.
    pub fn add_function<S: Into<Indeterminate>, A: Into<Indeterminate>>(
        &mut self,
        name: S,
        args: Vec<A>,
        body: Atom,
    ) -> Result<(), String> {
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

        self.add_tagged_function(name, tags, args, body)
    }

    /// Register a function, where the first arguments are `tags` instead of arguments.
    pub fn add_tagged_function<A: Into<Indeterminate>>(
        &mut self,
        name: Symbol,
        tags: Vec<Atom>,
        args: Vec<A>,
        body: Atom,
    ) -> Result<(), String> {
        if let Some(t) = self.tag.insert(name, tags.len())
            && t != tags.len()
        {
            return Err(format!(
                "Cannot add the same function {} with a different number of tags",
                name.get_name()
            ));
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
            });

        Ok(())
    }

    pub(super) fn get_tag_len(&self, symbol: &Symbol) -> usize {
        self.tag.get(symbol).cloned().unwrap_or(0)
    }

    pub(super) fn get(&self, a: AtomView) -> Option<&Expr> {
        if let Some(c) = self.map.get(a.get_data()) {
            return Some(c);
        }

        if let AtomView::Fun(aa) = a {
            let s = aa.get_symbol();
            let tag_len = self.get_tag_len(&s);

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
pub(super) struct Expr {
    pub(super) id: usize,
    pub(super) tag_len: usize,
    pub(super) args: Vec<Indeterminate>,
    pub(super) body: Atom,
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
