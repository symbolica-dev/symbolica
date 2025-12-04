//! Evaluation of expressions.
//!
//! The main entry point is through [AtomCore::evaluator].
use ahash::{AHasher, HashMap};
use rand::Rng;
use self_cell::self_cell;
use std::{
    hash::{Hash, Hasher},
    os::raw::{c_ulong, c_void},
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use crate::{
    LicenseManager,
    atom::{Atom, AtomCore, AtomView, KeyLookup, Symbol},
    coefficient::CoefficientView,
    combinatorics::unique_permutations,
    domains::{
        InternalOrdering,
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
    pub fn add_function(
        &mut self,
        name: Symbol,
        rename: String,
        args: Vec<Symbol>,
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

        self.tagged_fn_map.insert(
            (name, vec![]),
            ConstOrExpr::Expr(Expr {
                name: rename,
                tag_len: 0,
                args,
                body,
            }),
        );

        Ok(())
    }

    /// Register a function, where the first arguments are `tags` instead of arguments. `rename` is the name used in exported code.
    pub fn add_tagged_function(
        &mut self,
        name: Symbol,
        tags: Vec<Atom>,
        rename: String,
        args: Vec<Symbol>,
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

        let tag_len = tags.len();
        self.tagged_fn_map.insert(
            (name, tags),
            ConstOrExpr::Expr(Expr {
                name: rename,
                tag_len,
                args,
                body,
            }),
        );

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
    name: String,
    tag_len: usize,
    args: Vec<Symbol>,
    body: Atom,
}

/// Settings for optimizing the evaluation of expressions.
#[derive(Clone)]
pub struct OptimizationSettings {
    pub horner_iterations: usize,
    pub n_cores: usize,
    pub cpe_iterations: Option<usize>,
    pub hot_start: Option<Vec<Expression<Complex<Rational>>>>,
    pub abort_check: Option<Box<dyn AbortCheck>>,
    pub verbose: bool,
}

impl std::fmt::Debug for OptimizationSettings {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("OptimizationSettings")
            .field("horner_iterations", &self.horner_iterations)
            .field("n_cores", &self.n_cores)
            .field("cpe_iterations", &self.cpe_iterations)
            .field("hot_start", &self.hot_start)
            .field("abort_check", &self.abort_check.is_some())
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
            verbose: false,
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
    functions: Vec<(String, Vec<Symbol>, SplitExpression<T>)>,
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

/// A tree representation of an expression.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Expression<T> {
    Const(T),
    Parameter(usize),
    Eval(usize, Vec<Expression<T>>),
    Add(Vec<Expression<T>>),
    Mul(Vec<Expression<T>>),
    Pow(Box<(Expression<T>, i64)>),
    Powf(Box<(Expression<T>, Expression<T>)>),
    ReadArg(usize), // read nth function argument
    BuiltinFun(BuiltinSymbol, Box<Expression<T>>),
    ExternalFun(usize, Vec<Expression<T>>),
    IfElse(Box<(Expression<T>, Expression<T>, Expression<T>)>),
    SubExpression(usize),
}

impl<T: InternalOrdering + Eq> PartialOrd for Expression<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: InternalOrdering + Eq> Ord for Expression<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (Expression::Const(a), Expression::Const(b)) => a.internal_cmp(b),
            (Expression::Parameter(a), Expression::Parameter(b)) => a.cmp(b),
            (Expression::Eval(a, arg1), Expression::Eval(b, arg2)) => {
                a.cmp(b).then_with(|| arg1.cmp(arg2))
            }
            (Expression::Add(a), Expression::Add(b)) => a.cmp(b),
            (Expression::Mul(a), Expression::Mul(b)) => a.cmp(b),
            (Expression::Pow(a), Expression::Pow(b)) => a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)),
            (Expression::Powf(a), Expression::Powf(b)) => a.cmp(b),
            (Expression::ReadArg(a), Expression::ReadArg(b)) => a.cmp(b),
            (Expression::BuiltinFun(a, arg1), Expression::BuiltinFun(b, arg2)) => {
                a.cmp(b).then_with(|| arg1.cmp(arg2))
            }
            (Expression::ExternalFun(a, arg1), Expression::ExternalFun(b, arg2)) => {
                a.cmp(b).then_with(|| arg1.cmp(arg2))
            }
            (Expression::IfElse(a), Expression::IfElse(b)) => a.cmp(b),
            (Expression::SubExpression(a), Expression::SubExpression(b)) => a.cmp(b),
            (Expression::Const(_), _) => std::cmp::Ordering::Less,
            (_, Expression::Const(_)) => std::cmp::Ordering::Greater,
            (Expression::Parameter(_), _) => std::cmp::Ordering::Less,
            (_, Expression::Parameter(_)) => std::cmp::Ordering::Greater,
            (Expression::Eval(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::Eval(_, _)) => std::cmp::Ordering::Greater,
            (Expression::Add(_), _) => std::cmp::Ordering::Less,
            (_, Expression::Add(_)) => std::cmp::Ordering::Greater,
            (Expression::Mul(_), _) => std::cmp::Ordering::Less,
            (_, Expression::Mul(_)) => std::cmp::Ordering::Greater,
            (Expression::Pow(_), _) => std::cmp::Ordering::Less,
            (_, Expression::Pow(_)) => std::cmp::Ordering::Greater,
            (Expression::Powf(_), _) => std::cmp::Ordering::Less,
            (_, Expression::Powf(_)) => std::cmp::Ordering::Greater,
            (Expression::ReadArg(_), _) => std::cmp::Ordering::Less,
            (_, Expression::ReadArg(_)) => std::cmp::Ordering::Greater,
            (Expression::BuiltinFun(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::BuiltinFun(_, _)) => std::cmp::Ordering::Greater,
            (Expression::ExternalFun(_, _), _) => std::cmp::Ordering::Less,
            (_, Expression::ExternalFun(_, _)) => std::cmp::Ordering::Greater,
            (Expression::IfElse(_), _) => std::cmp::Ordering::Less,
            (_, Expression::IfElse(_)) => std::cmp::Ordering::Greater,
        }
    }
}

type ExpressionHash = u64;

#[derive(Debug, PartialEq, Eq, Clone)]
enum HashedExpression<T> {
    Const(ExpressionHash, T),
    Parameter(ExpressionHash, usize),
    Eval(ExpressionHash, usize, Vec<HashedExpression<T>>),
    Add(ExpressionHash, Vec<HashedExpression<T>>),
    Mul(ExpressionHash, Vec<HashedExpression<T>>),
    Pow(ExpressionHash, Box<(HashedExpression<T>, i64)>),
    Powf(
        ExpressionHash,
        Box<(HashedExpression<T>, HashedExpression<T>)>,
    ),
    ReadArg(ExpressionHash, usize), // read nth function argument
    BuiltinFun(ExpressionHash, BuiltinSymbol, Box<HashedExpression<T>>),
    ExternalFun(ExpressionHash, usize, Vec<HashedExpression<T>>),
    IfElse(
        ExpressionHash,
        Box<(
            HashedExpression<T>,
            HashedExpression<T>,
            HashedExpression<T>,
        )>,
    ),
    SubExpression(ExpressionHash, usize),
}

impl<T> HashedExpression<T> {
    fn get_hash(&self) -> ExpressionHash {
        match self {
            HashedExpression::Const(h, _) => *h,
            HashedExpression::Parameter(h, _) => *h,
            HashedExpression::Eval(h, _, _) => *h,
            HashedExpression::Add(h, _) => *h,
            HashedExpression::Mul(h, _) => *h,
            HashedExpression::Pow(h, _) => *h,
            HashedExpression::Powf(h, _) => *h,
            HashedExpression::ReadArg(h, _) => *h,
            HashedExpression::BuiltinFun(h, _, _) => *h,
            HashedExpression::SubExpression(h, _) => *h,
            HashedExpression::ExternalFun(h, _, _) => *h,
            HashedExpression::IfElse(h, _) => *h,
        }
    }
}

impl<T: Clone> HashedExpression<T> {
    fn to_expression(&self) -> Expression<T> {
        match self {
            HashedExpression::Const(_, c) => Expression::Const(c.clone()),
            HashedExpression::Parameter(_, p) => Expression::Parameter(*p),
            HashedExpression::Eval(_, i, v) => {
                Expression::Eval(*i, v.iter().map(|x| x.to_expression()).collect())
            }
            HashedExpression::Add(_, a) => {
                Expression::Add(a.iter().map(|x| x.to_expression()).collect())
            }
            HashedExpression::Mul(_, a) => {
                Expression::Mul(a.iter().map(|x| x.to_expression()).collect())
            }
            HashedExpression::Pow(_, p) => Expression::Pow(Box::new((p.0.to_expression(), p.1))),
            HashedExpression::Powf(_, p) => {
                Expression::Powf(Box::new((p.0.to_expression(), p.1.to_expression())))
            }
            HashedExpression::ReadArg(_, r) => Expression::ReadArg(*r),
            HashedExpression::BuiltinFun(_, s, a) => {
                Expression::BuiltinFun(*s, Box::new(a.to_expression()))
            }
            HashedExpression::SubExpression(_, s) => Expression::SubExpression(*s),
            HashedExpression::ExternalFun(_, s, a) => {
                Expression::ExternalFun(*s, a.iter().map(|x| x.to_expression()).collect())
            }
            HashedExpression::IfElse(_, b) => Expression::IfElse(Box::new((
                b.0.to_expression(),
                b.1.to_expression(),
                b.2.to_expression(),
            ))),
        }
    }
}

impl<T: Eq + InternalOrdering> PartialOrd for HashedExpression<T> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<T: Eq + InternalOrdering> Ord for HashedExpression<T> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        match (self, other) {
            (HashedExpression::Const(_, a), HashedExpression::Const(_, b)) => a.internal_cmp(b),
            (HashedExpression::Parameter(_, a), HashedExpression::Parameter(_, b)) => a.cmp(b),
            (HashedExpression::Eval(_, a, b), HashedExpression::Eval(_, c, d)) => {
                a.cmp(c).then_with(|| b.cmp(d))
            }
            (HashedExpression::Add(_, a), HashedExpression::Add(_, b)) => a.cmp(b),
            (HashedExpression::Mul(_, a), HashedExpression::Mul(_, b)) => a.cmp(b),
            (HashedExpression::Pow(_, p1), HashedExpression::Pow(_, p2)) => p1.cmp(p2),
            (HashedExpression::Powf(_, p1), HashedExpression::Powf(_, p2)) => p1.cmp(p2),
            (HashedExpression::ReadArg(_, r1), HashedExpression::ReadArg(_, r2)) => r1.cmp(r2),
            (HashedExpression::BuiltinFun(_, a, b), HashedExpression::BuiltinFun(_, c, d)) => {
                a.cmp(c).then_with(|| b.cmp(d))
            }
            (HashedExpression::SubExpression(_, s1), HashedExpression::SubExpression(_, s2)) => {
                s1.cmp(s2)
            }
            (HashedExpression::ExternalFun(_, a, b), HashedExpression::ExternalFun(_, c, d)) => {
                a.cmp(c).then_with(|| b.cmp(d))
            }
            (HashedExpression::IfElse(_, a), HashedExpression::IfElse(_, b)) => a.cmp(b),
            (HashedExpression::Const(_, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::Const(_, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::Parameter(_, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::Parameter(_, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::Eval(_, _, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::Eval(_, _, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::Add(_, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::Add(_, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::Mul(_, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::Mul(_, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::Pow(_, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::Pow(_, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::Powf(_, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::Powf(_, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::ReadArg(_, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::ReadArg(_, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::BuiltinFun(_, _, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::BuiltinFun(_, _, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::ExternalFun(_, _, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::ExternalFun(_, _, _)) => std::cmp::Ordering::Greater,
            (HashedExpression::IfElse(_, _), _) => std::cmp::Ordering::Less,
            (_, HashedExpression::IfElse(_, _)) => std::cmp::Ordering::Greater,
        }
    }
}

impl<T: Eq + Hash> Hash for HashedExpression<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(self.get_hash())
    }
}

impl<T: Eq + Hash + Clone + InternalOrdering> HashedExpression<T> {
    fn find_subexpression<'a>(
        &'a self,
        subexp: &mut HashMap<&'a HashedExpression<T>, usize>,
    ) -> bool {
        if matches!(
            self,
            HashedExpression::Const(_, _)
                | HashedExpression::Parameter(_, _)
                | HashedExpression::ReadArg(_, _)
        ) {
            return true;
        }

        if let Some(i) = subexp.get_mut(self) {
            *i += 1;
            return true;
        }

        subexp.insert(self, 1);

        match self {
            HashedExpression::Const(_, _)
            | HashedExpression::Parameter(_, _)
            | HashedExpression::ReadArg(_, _) => {}
            HashedExpression::Eval(_, _, ae) => {
                for arg in ae {
                    arg.find_subexpression(subexp);
                }
            }
            HashedExpression::Add(_, a)
            | HashedExpression::Mul(_, a)
            | HashedExpression::ExternalFun(_, _, a) => {
                for arg in a {
                    arg.find_subexpression(subexp);
                }
            }
            HashedExpression::Pow(_, p) => {
                p.0.find_subexpression(subexp);
            }
            HashedExpression::Powf(_, p) => {
                p.0.find_subexpression(subexp);
                p.1.find_subexpression(subexp);
            }
            HashedExpression::BuiltinFun(_, _, a) => {
                a.find_subexpression(subexp);
            }
            HashedExpression::SubExpression(_, _) => {}
            HashedExpression::IfElse(_, b) => {
                b.0.find_subexpression(subexp);
                b.1.find_subexpression(subexp);
                b.2.find_subexpression(subexp);
            }
        }

        false
    }

    fn replace_subexpression(
        &mut self,
        subexp: &HashMap<&HashedExpression<T>, usize>,
        skip_root: bool,
    ) {
        if !skip_root && let Some(i) = subexp.get(self) {
            *self = HashedExpression::SubExpression(self.get_hash(), *i); // TODO: do not recyle hash?
            return;
        }

        match self {
            HashedExpression::Const(_, _)
            | HashedExpression::Parameter(_, _)
            | HashedExpression::ReadArg(_, _) => {}
            HashedExpression::Eval(_, _, ae) => {
                for arg in &mut *ae {
                    arg.replace_subexpression(subexp, false);
                }
            }
            HashedExpression::Add(_, a)
            | HashedExpression::Mul(_, a)
            | HashedExpression::ExternalFun(_, _, a) => {
                for arg in a {
                    arg.replace_subexpression(subexp, false);
                }
            }
            HashedExpression::Pow(_, p) => {
                p.0.replace_subexpression(subexp, false);
            }
            HashedExpression::Powf(_, p) => {
                p.0.replace_subexpression(subexp, false);
                p.1.replace_subexpression(subexp, false);
            }
            HashedExpression::BuiltinFun(_, _, a) => {
                a.replace_subexpression(subexp, false);
            }
            HashedExpression::SubExpression(_, _) => {}
            HashedExpression::IfElse(_, b) => {
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
            HashedExpression::Const(_, _)
                | HashedExpression::Parameter(_, _)
                | HashedExpression::ReadArg(_, _)
        ) {
            return (0, 0);
        }

        if sub_expr.contains_key(self) {
            return (0, 0);
        }

        sub_expr.insert(self, 1);

        match self {
            HashedExpression::Const(_, _) => (0, 0),
            HashedExpression::Parameter(_, _) => (0, 0),
            HashedExpression::Eval(_, _, args) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in args {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add, mul)
            }
            HashedExpression::Add(_, a) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in a {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add + a.len() - 1, mul)
            }
            HashedExpression::Mul(_, m) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in m {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add, mul + m.len() - 1)
            }
            HashedExpression::Pow(_, p) => {
                let (a, m) = p.0.count_operations_with_subexpression(sub_expr);
                (a, m + p.1.unsigned_abs() as usize - 1)
            }
            HashedExpression::Powf(_, p) => {
                let (a, m) = p.0.count_operations_with_subexpression(sub_expr);
                let (a2, m2) = p.1.count_operations_with_subexpression(sub_expr);
                (a + a2, m + m2 + 1) // not clear how to count this
            }
            HashedExpression::ReadArg(_, _) => (0, 0),
            HashedExpression::BuiltinFun(_, _, b) => {
                b.count_operations_with_subexpression(sub_expr)
            } // not clear how to count this, third arg?
            HashedExpression::SubExpression(_, _) => (0, 0),
            HashedExpression::ExternalFun(_, _, a) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in a {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add + a.len() - 1, mul)
            }
            HashedExpression::IfElse(_, b) => {
                let (a1, m1) = b.0.count_operations_with_subexpression(sub_expr);
                let (a2, m2) = b.1.count_operations_with_subexpression(sub_expr);
                let (a3, m3) = b.2.count_operations_with_subexpression(sub_expr);
                (a1 + a2 + a3, m1 + m2 + m3)
            }
        }
    }
}

impl<T: std::hash::Hash + Clone> Expression<T> {
    fn to_hashed_expression(&self) -> (ExpressionHash, HashedExpression<T>) {
        match self {
            Expression::Const(c) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(0);
                c.hash(&mut hasher);
                let h = hasher.finish();
                (h, HashedExpression::Const(h, c.clone()))
            }
            Expression::Parameter(p) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(1);
                hasher.write_usize(*p);
                let h = hasher.finish();
                (h, HashedExpression::Parameter(h, *p))
            }
            Expression::Eval(i, v) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(2);
                hasher.write_usize(*i);
                let mut new_v = vec![];
                for x in v {
                    let (h, v) = x.to_hashed_expression();
                    new_v.push(v);
                    hasher.write_u64(h);
                }
                let h = hasher.finish();
                (h, HashedExpression::Eval(h, *i, new_v))
            }
            Expression::Add(v) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(3);
                let mut new_v = vec![];

                // do an additive hash
                let mut arg_sum = 0u64;
                for x in v {
                    let (h, v) = x.to_hashed_expression();
                    new_v.push(v);
                    arg_sum = arg_sum.wrapping_add(h);
                }
                hasher.write_u64(arg_sum);
                let h = hasher.finish();
                (h, HashedExpression::Add(h, new_v))
            }
            Expression::Mul(v) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(4);
                let mut new_v = vec![];

                // do an additive hash
                let mut arg_sum = 0u64;
                for x in v {
                    let (h, v) = x.to_hashed_expression();
                    new_v.push(v);
                    arg_sum = arg_sum.wrapping_add(h);
                }
                hasher.write_u64(arg_sum);
                let h = hasher.finish();
                (h, HashedExpression::Mul(h, new_v))
            }
            Expression::Pow(p) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(5);
                let (hb, vb) = p.0.to_hashed_expression();
                hasher.write_u64(hb);
                hasher.write_i64(p.1);
                let h = hasher.finish();
                (h, HashedExpression::Pow(h, Box::new((vb, p.1))))
            }
            Expression::Powf(p) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(6);
                let (hb, vb) = p.0.to_hashed_expression();
                let (he, ve) = p.1.to_hashed_expression();
                hasher.write_u64(hb);
                hasher.write_u64(he);
                let h = hasher.finish();
                (h, HashedExpression::Powf(h, Box::new((vb, ve))))
            }
            Expression::ReadArg(i) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(7);
                hasher.write_usize(*i);
                let h = hasher.finish();
                (h, HashedExpression::ReadArg(h, *i))
            }
            Expression::BuiltinFun(s, a) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(8);
                s.hash(&mut hasher);
                let (ha, va) = a.to_hashed_expression();
                hasher.write_u64(ha);
                let h = hasher.finish();
                (h, HashedExpression::BuiltinFun(h, *s, Box::new(va)))
            }
            Expression::SubExpression(i) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(9);
                hasher.write_usize(*i);
                let h = hasher.finish();
                (h, HashedExpression::SubExpression(h, *i))
            }
            Expression::ExternalFun(s, a) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(10);
                s.hash(&mut hasher);
                let mut args = vec![];
                for x in a {
                    let (h, v) = x.to_hashed_expression();
                    hasher.write_u64(h);
                    args.push(v);
                }
                let h = hasher.finish();
                (h, HashedExpression::ExternalFun(h, *s, args))
            }
            Expression::IfElse(b) => {
                let mut hasher = AHasher::default();
                hasher.write_u8(11);
                let (h1, v1) = b.0.to_hashed_expression();
                let (h2, v2) = b.1.to_hashed_expression();
                let (h3, v3) = b.2.to_hashed_expression();
                hasher.write_u64(h1);
                hasher.write_u64(h2);
                hasher.write_u64(h3);
                let h = hasher.finish();
                (h, HashedExpression::IfElse(h, Box::new((v1, v2, v3))))
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
#[derive(Clone, PartialEq, Debug)]

pub struct ExpressionEvaluator<T> {
    stack: Vec<T>,
    param_count: usize,
    reserved_indices: usize,
    instructions: Vec<Instr>,
    result_indices: Vec<usize>,
    external_fns: Vec<String>,
}

impl<T: Clone> ExpressionEvaluator<T> {
    /// Register external functions for the evaluator.
    ///
    /// Examples
    /// --------
    /// ```rust
    /// use ahash::HashMap;
    /// use symbolica::{atom::AtomCore, evaluate::{FunctionMap, OptimizationSettings}, parse, symbol};
    ///
    /// let mut ext: HashMap<String, Box<dyn Fn(&[f64]) -> f64 + Send + Sync>> = HashMap::default();
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
        mut external_fns: HashMap<String, Box<dyn Fn(&[T]) -> T + Send + Sync>>,
    ) -> Result<ExpressionEvaluatorWithExternalFunctions<T>, String> {
        let mut external = vec![];
        for e in &self.external_fns {
            if let Some(f) = external_fns.remove(e) {
                external.push((vec![], f));
            } else {
                return Err(format!("External function '{e}' not found"));
            }
        }

        Ok(ExpressionEvaluatorWithExternalFunctions {
            stack: self.stack.clone(),
            param_count: self.param_count,
            instructions: self.instructions.clone(),
            result_indices: self.result_indices.clone(),
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
    pub fn evaluate(&mut self, params: &[T], out: &mut [T]) {
        if self.param_count != params.len() {
            panic!(
                "Parameter count mismatch: expected {}, got {}",
                self.param_count,
                params.len()
            );
        }

        for (t, p) in self.stack.iter_mut().zip(params) {
            *t = p.clone();
        }

        let mut tmp;
        let mut i = 0;
        while i < self.instructions.len() {
            let instr = unsafe { &self.instructions.get_unchecked(i) };
            match instr {
                Instr::Add(r, v) => {
                    tmp = self.stack[v[0]].clone();
                    for x in &v[1..] {
                        let e = self.stack[*x].clone();
                        tmp += e;
                    }
                    std::mem::swap(&mut self.stack[*r], &mut tmp);
                }
                Instr::Mul(r, v) => {
                    tmp = self.stack[v[0]].clone();
                    for x in &v[1..] {
                        let e = self.stack[*x].clone();
                        tmp *= e;
                    }
                    std::mem::swap(&mut self.stack[*r], &mut tmp);
                }
                Instr::Pow(r, b, e) => {
                    if *e >= 0 {
                        self.stack[*r] = self.stack[*b].pow(*e as u64);
                    } else {
                        self.stack[*r] = self.stack[*b].pow(e.unsigned_abs()).inv();
                    }
                }
                Instr::Powf(r, b, e) => {
                    self.stack[*r] = self.stack[*b].powf(&self.stack[*e]);
                }
                Instr::BuiltinFun(r, s, arg) => match s.0 {
                    Symbol::EXP => self.stack[*r] = self.stack[*arg].exp(),
                    Symbol::LOG => self.stack[*r] = self.stack[*arg].log(),
                    Symbol::SIN => self.stack[*r] = self.stack[*arg].sin(),
                    Symbol::COS => self.stack[*r] = self.stack[*arg].cos(),
                    Symbol::SQRT => self.stack[*r] = self.stack[*arg].sqrt(),
                    Symbol::CONJ => self.stack[*r] = self.stack[*arg].conj(),
                    _ => unreachable!(),
                },
                Instr::ExternalFun(_, s, _) => {
                    panic!(
                        "External function {} is not set. Call `with_external_functions` first.",
                        self.external_fns[*s]
                    );
                }
                Instr::IfElse(n, label) => {
                    // jump to else block
                    if self.stack[*n].is_fully_zero() {
                        i = label.0;
                        continue;
                    }
                }
                Instr::Goto(label) => {
                    i = label.0;
                    continue;
                }
                Instr::Label(_) => {}
                Instr::Join(r, c, a, b) => {
                    if !self.stack[*c].is_fully_zero() {
                        self.stack[*r] = self.stack[*a].clone();
                    } else {
                        self.stack[*r] = self.stack[*b].clone();
                    }
                }
            }

            i += 1;
        }

        for (o, i) in out.iter_mut().zip(&self.result_indices) {
            *o = self.stack[*i].clone();
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
        }
    }

    pub fn get_input_len(&self) -> usize {
        self.param_count
    }

    pub fn get_output_len(&self) -> usize {
        self.result_indices.len()
    }

    /// Return the total number of additions and multiplications.
    pub fn count_operations(&self) -> (usize, usize) {
        let mut add_count = 0;
        let mut mul_count = 0;

        for instr in &self.instructions {
            match instr {
                Instr::Add(_, s) => add_count += s.len() - 1,
                Instr::Mul(_, s) => mul_count += s.len() - 1,
                _ => {}
            }
        }

        (add_count, mul_count)
    }

    /// Remove common pairs of instructions. Assumes that the arguments
    /// of the instructions are sorted.
    fn remove_common_pairs(&mut self) -> usize {
        #[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
        enum CommonInstruction {
            Add(usize, usize),
            Mul(usize, usize),
            BuiltinFun(Symbol, usize),
            ExternalFun(usize, Vec<usize>),
        }

        let mut common_ops: HashMap<_, Vec<usize>> = HashMap::default();

        let mut affected_lines = vec![false; self.instructions.len()];

        let mut line_usage_zone = vec![0; self.instructions.len()];

        let mut current_zone = 0;
        let mut current_zone_depth = 0;
        for (p, i) in self.instructions.iter().enumerate() {
            match i {
                Instr::Add(_, a) | Instr::Mul(_, a) => {
                    let is_add = matches!(i, Instr::Add(_, _));
                    for (li, l) in a.iter().enumerate() {
                        for r in &a[li + 1..] {
                            if is_add {
                                common_ops
                                    .entry(CommonInstruction::Add(*l, *r))
                                    .or_default()
                                    .push(p);
                            } else {
                                common_ops
                                    .entry(CommonInstruction::Mul(*l, *r))
                                    .or_default()
                                    .push(p);
                            }
                        }
                    }
                }
                Instr::BuiltinFun(_, f, a) => {
                    common_ops
                        .entry(CommonInstruction::BuiltinFun(f.0, *a))
                        .or_default()
                        .push(p);
                }
                Instr::ExternalFun(_, f, args) => {
                    common_ops
                        .entry(CommonInstruction::ExternalFun(*f, args.clone()))
                        .or_default()
                        .push(p);
                }
                Instr::IfElse(_, _) => {
                    line_usage_zone[p] = current_zone;
                    current_zone_depth += 1;
                    current_zone += 3_usize.pow(current_zone_depth);
                    continue;
                }
                Instr::Goto(_) => {
                    current_zone += 3_usize.pow(current_zone_depth);
                }
                Instr::Join(..) => {
                    current_zone %= 3_usize.pow(current_zone_depth);
                    current_zone_depth -= 1;
                }
                _ => {}
            }
            line_usage_zone[p] = current_zone;
        }

        let mut to_remove: Vec<_> = common_ops.clone().into_iter().collect();
        to_remove.retain_mut(|(_, v)| {
            let keep = v.len() > 1;
            v.dedup();
            keep
        });

        // sort in other direction since we pop
        to_remove.sort_by(|a, b| a.1.len().cmp(&b.1.len()).then_with(|| a.cmp(b)));

        let total_remove = to_remove.len();

        let old_len = self.instructions.len();

        let mut new_symb_usage_zone = vec![];

        while let Some((common, lines)) = to_remove.pop() {
            match common {
                CommonInstruction::Add(l, r) | CommonInstruction::Mul(l, r) => {
                    let is_add = matches!(common, CommonInstruction::Add(_, _));

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
                    self.instructions.push(new_op);

                    let mut usage_zone = line_usage_zone[lines[0]];
                    for &line in &lines {
                        affected_lines[line] = true;

                        let lu = line_usage_zone[line];
                        for pow in 1..32 {
                            if lu % 3usize.pow(pow) != usage_zone % 3usize.pow(pow) {
                                usage_zone %= 3usize.pow(pow - 1);
                                break;
                            }
                        }

                        let is_add = matches!(self.instructions[line], Instr::Add(_, _));

                        if let Instr::Add(_, a) | Instr::Mul(_, a) = &mut self.instructions[line] {
                            for (li, l) in a.iter().enumerate() {
                                for r in &a[li + 1..] {
                                    let pp = common_ops
                                        .entry(if is_add {
                                            CommonInstruction::Add(*l, *r)
                                        } else {
                                            CommonInstruction::Mul(*l, *r)
                                        })
                                        .or_default();
                                    pp.retain(|x| *x != line);
                                }
                            }

                            if l == r {
                                let count = a.iter().filter(|x| **x == l).count();
                                let pairs = count / 2;
                                if pairs > 0 {
                                    a.retain(|x| *x != l);

                                    if count % 2 == 1 {
                                        a.push(l);
                                    }

                                    a.extend(std::iter::repeat_n(new_idx, pairs));
                                    a.sort();
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
                                    a.sort();
                                }
                            }

                            // update the pairs for this line
                            for (li, l) in a.iter().enumerate() {
                                for r in &a[li + 1..] {
                                    common_ops
                                        .entry(if is_add {
                                            CommonInstruction::Add(*l, *r)
                                        } else {
                                            CommonInstruction::Mul(*l, *r)
                                        })
                                        .or_default()
                                        .push(line);
                                }
                            }
                        }
                    }

                    new_symb_usage_zone.push((lines[0], usage_zone));
                }
                CommonInstruction::BuiltinFun(_, _) | CommonInstruction::ExternalFun(_, _) => {
                    if lines.iter().any(|x| affected_lines[*x]) {
                        continue;
                    }

                    let new_idx = self.stack.len();
                    let new_op = match common {
                        CommonInstruction::BuiltinFun(s, a) => {
                            Instr::BuiltinFun(new_idx, BuiltinSymbol(s), a)
                        }
                        CommonInstruction::ExternalFun(s, a) => Instr::ExternalFun(new_idx, s, a),
                        _ => unreachable!(),
                    };

                    self.stack.push(T::default());
                    self.instructions.push(new_op);

                    let mut usage_zone = line_usage_zone[lines[0]];
                    for &line in &lines {
                        affected_lines[line] = true;

                        let lu = line_usage_zone[line];
                        for pow in 1..32 {
                            if lu % 3usize.pow(pow) != usage_zone % 3usize.pow(pow) {
                                usage_zone %= 3usize.pow(pow - 1);
                                break;
                            }
                        }

                        match &self.instructions[line] {
                            Instr::BuiltinFun(r, _, _) => {
                                self.instructions[line] = Instr::Add(*r, vec![new_idx]);
                            }
                            Instr::ExternalFun(r, _, _) => {
                                self.instructions[line] = Instr::Add(*r, vec![new_idx]);
                            }
                            _ => panic!("Expected BuiltinFun or ExternalFun instruction"),
                        }
                    }

                    new_symb_usage_zone.push((lines[0], usage_zone));
                }
            }
        }

        // detect the earliest point and latest point for an instruction placement
        // earliest point: after last dependency
        // latest point: before first usage in the correct usage zone
        let mut placement_bounds = vec![];
        for (i, (first_usage, zone)) in self.instructions.drain(old_len..).zip(new_symb_usage_zone)
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
                if line_usage_zone[j] == zone {
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
                    Instr::BuiltinFun(o, _, a) => (*o, std::slice::from_ref(a)),
                    Instr::ExternalFun(o, _, a) => (*o, a.as_slice()),
                    _ => unreachable!(),
                };

                let mut new_a = a.iter().map(|x| rename!(*x)).collect::<Vec<_>>();
                new_a.sort();

                match placement_bounds[j].2 {
                    Instr::Add(_, _) => {
                        new_instr.push(Instr::Add(new_pos, new_a));
                    }
                    Instr::Mul(_, _) => {
                        new_instr.push(Instr::Mul(new_pos, new_a));
                    }
                    Instr::BuiltinFun(_, b, _) => {
                        new_instr.push(Instr::BuiltinFun(new_pos, b, new_a[0]));
                    }
                    Instr::ExternalFun(_, fi, _) => {
                        new_instr.push(Instr::ExternalFun(new_pos, fi, new_a));
                    }
                    _ => unreachable!(),
                }

                sub_rename.insert(o, new_pos);

                j += 1;
            } else {
                let mut s = self.instructions[i].clone();

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

                new_instr.push(s);
                rename_map.push(new_pos);
                i += 1;
            }
        }

        // fix labels
        let mut label_map: HashMap<usize, usize> = HashMap::default();
        for (i, x) in new_instr.iter_mut().enumerate().rev() {
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

        for x in &mut self.result_indices {
            *x = rename!(*x);
        }

        assert!(j == placement_bounds.len());

        self.instructions = new_instr;

        total_remove
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
            for i in &mut self.instructions {
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
        for i in &mut other.instructions {
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

        for _ in 0..cpe_rounds.unwrap_or(usize::MAX) {
            if self.remove_common_pairs() == 0 {
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

        for (i, x) in self.instructions.iter().enumerate() {
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

        let mut max_reg = self.reserved_indices;
        for (i, x) in self.instructions.iter_mut().enumerate() {
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

            let cur_last_use = last_use[cur_reg];

            let new_reg = if let Some((new_v, lu)) = last_use[..cur_reg]
                .iter_mut()
                .enumerate()
                .find(|(_, r)| **r <= i)
            // <= is ok because we store intermediate results in temp values
            {
                *lu = cur_last_use; // set the last use to the current variable last use
                last_use[cur_reg] = 0; // make the current index available
                rename_map[cur_reg] = new_v; // set the rename map so that every occurrence on the rhs is replaced
                new_v
            } else {
                cur_reg
            };

            max_reg = max_reg.max(new_reg);

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
        for (index, i) in &mut self.instructions.iter_mut().enumerate() {
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
        self.to_string()
    }

    fn is_real(&self) -> bool {
        true
    }
}

impl ExportNumber for F64 {
    fn export(&self) -> String {
        self.to_string()
    }

    fn is_real(&self) -> bool {
        true
    }
}

impl ExportNumber for Float {
    fn export(&self) -> String {
        self.to_string()
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
                        self.reserved_indices - self.param_count + 1,
                        {
                            let mut nums = (self.param_count..self.reserved_indices)
                                .map(|i| format!("simd({})", self.stack[i].export()))
                                .collect::<Vec<_>>();
                            nums.push("-0.".to_string()); // used for inversion
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

        for ins in &self.instructions {
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
                        *out += format!("\t{} = T(1) / {base};\n", get_output!(o)).as_str();
                    } else {
                        *out += format!("\t{} = pow({base}, {e});\n", get_output!(o)).as_str();
                    }
                }
                Instr::Powf(o, b, e) => {
                    let base = get_input!(*b);
                    let exp = get_input!(*e);
                    *out += format!("\t{} = pow({base}, {exp});\n", get_output!(o)).as_str();
                }
                Instr::BuiltinFun(o, s, a) => match s.0 {
                    Symbol::EXP => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = exp({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::LOG => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = log({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::SIN => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = sin({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::COS => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = cos({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::SQRT => {
                        let arg = get_input!(*a);
                        *out += format!("\t{} = sqrt({arg});\n", get_output!(o)).as_str();
                    }
                    Symbol::CONJ => {
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
                }
                Instr::Label(..) => {}
                Instr::Join(o, cond, a, b) => {
                    *out += "\t}\n";
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
            res += "#include <iostream>\n#include <cmath>\n\n";
        };

        if let Some(header) = &settings.custom_header {
            res += header;
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
        }

        res += &format!(
            "extern \"C\" unsigned long {}_get_buffer_len()\n{{\n\treturn {};\n}}\n\n",
            function_name,
            self.stack.len()
        );

        res += &format!(
            "static const std::complex<double> {}_CONSTANTS_complex[{}] = {{{}}};\n\n",
            function_name,
            self.reserved_indices - self.param_count + 1,
            {
                let mut nums = (self.param_count..self.reserved_indices)
                    .map(|i| format!("std::complex<double>({})", self.stack[i].export()))
                    .collect::<Vec<_>>();
                nums.push("std::complex<double>(0, -0.)".to_string()); // used for inversion
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
        instr: &[Instr],
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
                        }
                        InlineASM::None => unreachable!(),
                    }
                    $in_block = false;
                }
            };
        }

        let mut reg_last_use = vec![self.instructions.len(); self.instructions.len()];
        let mut stack_to_reg = HashMap::default();

        for (i, ins) in instr.iter().enumerate() {
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
        enum RegInstr {
            Add(MemOrReg, u16, Vec<MemOrReg>),
            Mul(MemOrReg, u16, Vec<MemOrReg>),
            Pow(MemOrReg, u16, MemOrReg, i64),
            Powf(usize, usize, usize),
            BuiltinFun(usize, BuiltinSymbol, usize),
            ExternalFun(usize, usize, Vec<usize>),
            IfElse(usize),
            Goto,
            Label,
            Join(usize, usize, usize, usize),
        }

        let mut new_instr: Vec<RegInstr> = instr
            .iter()
            .map(|i| match i {
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
                Instr::BuiltinFun(r, s, a) => RegInstr::BuiltinFun(*r, *s, *a),
                Instr::ExternalFun(r, s, a) => RegInstr::ExternalFun(*r, *s, a.clone()),
                Instr::IfElse(c, _) => RegInstr::IfElse(*c),
                Instr::Goto(_) => RegInstr::Goto,
                Instr::Label(_) => RegInstr::Label,
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
                        RegInstr::Pow(_, f, a, -1) => {
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
                        RegInstr::Goto | RegInstr::Label => {}
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

        let mut in_asm_block = false;
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
                RegInstr::Powf(o, b, e) => {
                    end_asm_block!(in_asm_block);

                    let base = get_input!(*b);
                    let exp = get_input!(*e);
                    *out += format!("\tZ[{o}] = pow({base}, {exp});\n").as_str();
                }
                RegInstr::BuiltinFun(o, s, a) => {
                    end_asm_block!(in_asm_block);

                    let arg = get_input!(*a);

                    match s.0 {
                        Symbol::EXP => {
                            *out += format!("\tZ[{o}] = exp({arg});\n").as_str();
                        }
                        Symbol::LOG => {
                            *out += format!("\tZ[{o}] = log({arg});\n").as_str();
                        }
                        Symbol::SIN => {
                            *out += format!("\tZ[{o}] = sin({arg});\n").as_str();
                        }
                        Symbol::COS => {
                            *out += format!("\tZ[{o}] = cos({arg});\n").as_str();
                        }
                        Symbol::SQRT => {
                            *out += format!("\tZ[{o}] = sqrt({arg});\n").as_str();
                        }
                        Symbol::CONJ => {
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
                    end_asm_block!(in_asm_block);
                    *out += "\t} else {\n";
                }
                RegInstr::Label => {}
                RegInstr::Join(o, cond, a, b) => {
                    end_asm_block!(in_asm_block);
                    *out += "\t}\n";
                    let arg_a = get_input!(*a);
                    let arg_b = get_input!(*b);

                    if asm_flavour == InlineASM::AVX2 {
                        *out += &format!(
                            "\tZ[{}] = (all({} != 0.)) ? {} : {};\n",
                            o,
                            get_input!(*cond),
                            arg_a,
                            arg_b
                        );
                    } else {
                        *out += &format!(
                            "\tZ[{}] = ({} != 0.) ? {} : {};\n",
                            o,
                            get_input!(*cond),
                            arg_a,
                            arg_b
                        );
                    }
                }
            }
        }

        end_asm_block!(in_asm_block);

        let mut regcount = 0;
        *out += "\t__asm__(\n";
        for (i, r) in self.result_indices.iter().enumerate() {
            if *r < self.param_count {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!("\t\t\"movsd {}(%3), %%xmm{}\\n\\t\"\n", r * 8, regcount);
                    }
                    InlineASM::AVX2 => {
                        *out +=
                            &format!("\t\t\"vmovupd {}(%3), %%ymm{}\\n\\t\"\n", r * 32, regcount);
                    }
                    InlineASM::AArch64 => {
                        *out += &format!("\t\t\"ldr d{}, [%3, {}]\\n\\t\"\n", regcount, r * 8);
                    }
                    InlineASM::None => unreachable!(),
                }
            } else if *r < self.reserved_indices {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!(
                            "\t\t\"movsd {}(%2), %%xmm{}\\n\\t\"\n",
                            (r - self.param_count) * 8,
                            regcount
                        );
                    }
                    InlineASM::AVX2 => {
                        *out += &format!(
                            "\t\t\"vmovupd {}(%2), %%ymm{}\\n\\t\"\n",
                            (r - self.param_count) * 32,
                            regcount
                        );
                    }
                    InlineASM::AArch64 => {
                        *out += &format!(
                            "\t\t\"ldr d{}, [%2, {}]\\n\\t\"\n",
                            regcount,
                            (r - self.param_count) * 8
                        );
                    }
                    InlineASM::None => unreachable!(),
                }
            } else {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!("\t\t\"movsd {}(%1), %%xmm{}\\n\\t\"\n", r * 8, regcount);
                    }
                    InlineASM::AVX2 => {
                        *out +=
                            &format!("\t\t\"vmovupd {}(%1), %%ymm{}\\n\\t\"\n", r * 32, regcount);
                    }
                    InlineASM::AArch64 => {
                        *out += &format!("\t\t\"ldr d{}, [%1, {}]\\n\\t\"\n", regcount, r * 8);
                    }
                    InlineASM::None => unreachable!(),
                }
            }

            match asm_flavour {
                InlineASM::X64 => {
                    *out += &format!("\t\t\"movsd %%xmm{}, {}(%0)\\n\\t\"\n", regcount, i * 8);
                }
                InlineASM::AVX2 => {
                    *out += &format!("\t\t\"vmovupd %%ymm{}, {}(%0)\\n\\t\"\n", regcount, i * 32);
                }
                InlineASM::AArch64 => {
                    *out += &format!("\t\t\"str d{}, [%0, {}]\\n\\t\"\n", regcount, i * 8);
                }
                InlineASM::None => unreachable!(),
            }
            regcount = (regcount + 1) % 16;
        }

        match asm_flavour {
            InlineASM::X64 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(out), \"r\"(Z), \"r\"({function_name}_CONSTANTS_double), \"r\"(params)\n\t\t: \"memory\", \"xmm0\", \"xmm1\", \"xmm2\", \"xmm3\", \"xmm4\", \"xmm5\", \"xmm6\", \"xmm7\", \"xmm8\", \"xmm9\", \"xmm10\", \"xmm11\", \"xmm12\", \"xmm13\", \"xmm14\", \"xmm15\");\n"
                );
            }
            InlineASM::AVX2 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(out), \"r\"(Z), \"r\"({function_name}_CONSTANTS_double), \"r\"(params)\n\t\t: \"memory\", \"ymm0\", \"ymm1\", \"ymm2\", \"ymm3\", \"ymm4\", \"ymm5\", \"ymm6\", \"ymm7\", \"ymm8\", \"ymm9\", \"ymm10\", \"ymm11\", \"ymm12\", \"ymm13\", \"ymm14\", \"ymm15\");\n"
                );
            }
            InlineASM::AArch64 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(out), \"r\"(Z), \"r\"({function_name}_CONSTANTS_double), \"r\"(params)\n\t\t: \"memory\", \"d0\", \"d1\", \"d2\", \"d3\", \"d4\", \"d5\", \"d6\", \"d7\", \"d8\", \"d9\", \"d10\", \"d11\", \"d12\", \"d13\", \"d14\", \"d15\", \"d16\", \"d17\", \"d18\", \"d19\", \"d20\", \"d21\", \"d22\", \"d23\", \"d24\", \"d25\", \"d26\", \"d27\", \"d28\", \"d29\", \"d30\", \"d31\");\n"
                );
            }
            InlineASM::None => unreachable!(),
        }
        in_asm_block
    }

    fn export_asm_complex_impl(
        &self,
        instr: &[Instr],
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
                        }
                        InlineASM::None => unreachable!(),
                    }
                    $in_block = false;
                }
            };
        }

        let mut in_asm_block = false;
        for ins in instr {
            match ins {
                Instr::Add(o, a) => {
                    if !in_asm_block {
                        *out += "\t__asm__(\n";
                        in_asm_block = true;
                    }

                    match asm_flavour {
                        InlineASM::X64 => {
                            *out += "\t\t\"xorpd %%xmm0, %%xmm0\\n\\t\"\n";

                            // TODO: try loading in multiple registers for better instruction-level parallelism?
                            for i in a {
                                let (addr, _) = asm_load!(*i);
                                *out += &format!("\t\t\"addpd {addr}, %%xmm0\\n\\t\"\n");
                            }
                            let (addr, _) = asm_load!(*o);
                            *out += &format!("\t\t\"movupd %%xmm0, {addr}\\n\\t\"\n");
                        }
                        InlineASM::AVX2 => {
                            let (addr, comp_addr) = asm_load!(a[0]);
                            *out += &format!("\t\t\"vmovupd {addr}, %%ymm0\\n\\t\"\n");
                            *out += &format!("\t\t\"vmovupd {comp_addr}, %%ymm1\\n\\t\"\n");

                            // TODO: try loading in multiple registers for better instruction-level parallelism?
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
                    if !matches!(asm_flavour, InlineASM::AVX2) && a.len() < 15 || a.len() < 8 {
                        if !in_asm_block {
                            *out += "\t__asm__(\n";
                            in_asm_block = true;
                        }

                        // optimized complex multiplication
                        for (i, r) in a.iter().enumerate() {
                            let (addr_re, addr_im) = asm_load!(*r);
                            match asm_flavour {
                                InlineASM::X64 => {
                                    *out += &format!(
                                        "\t\t\"movupd {}, %%xmm{}\\n\\t\"\n",
                                        addr_re,
                                        i + 1,
                                    );
                                }
                                InlineASM::AVX2 => {
                                    *out += &format!(
                                        "\t\t\"vmovupd {}, %%ymm{}\\n\\t\"\n",
                                        addr_re,
                                        2 * i,
                                    );
                                    *out += &format!(
                                        "\t\t\"vmovupd {}, %%ymm{}\\n\\t\"\n",
                                        addr_im,
                                        2 * i + 1,
                                    );
                                }
                                InlineASM::AArch64 => {
                                    if *r * 16 < 450 {
                                        *out += &format!(
                                            "\t\t\"ldp d{}, d{}, {}\\n\\t\"\n",
                                            2 * (i + 1),
                                            2 * (i + 1) + 1,
                                            addr_re,
                                        );
                                    } else {
                                        *out += &format!(
                                            "\t\t\"ldr d{}, {}\\n\\t\"\n",
                                            2 * (i + 1),
                                            addr_re,
                                        );
                                        *out += &format!(
                                            "\t\t\"ldr d{}, {}\\n\\t\"\n",
                                            2 * (i + 1) + 1,
                                            addr_im,
                                        );
                                    }
                                }
                                InlineASM::None => unreachable!(),
                            }
                        }

                        for i in 1..a.len() {
                            match asm_flavour {
                                InlineASM::X64 => {
                                    *out += &format!(
                                        "\t\t\"movapd %%xmm1, %%xmm0\\n\\t\"
\t\t\"unpckhpd %%xmm0, %%xmm0\\n\\t\"
\t\t\"unpcklpd %%xmm1, %%xmm1\\n\\t\"
\t\t\"mulpd %%xmm{0}, %%xmm0\\n\\t\"
\t\t\"mulpd %%xmm{0}, %%xmm1\\n\\t\"
\t\t\"shufpd $1, %%xmm0, %%xmm0\\n\\t\"
\t\t\"addsubpd %%xmm0, %%xmm1\\n\\t\"\n",
                                        i + 1
                                    );
                                }
                                InlineASM::AVX2 => {
                                    *out += &format!(
                                        "\t\t\"vmulpd %%ymm0, %%ymm{0}, %%ymm14\\n\\t\"
\t\t\"vmulpd %%ymm0, %%ymm{1}, %%ymm15\\n\\t\"
\t\t\"vmulpd %%ymm1, %%ymm{1}, %%ymm0\\n\\t\"
\t\t\"vmulpd %%ymm1, %%ymm{0}, %%ymm{1}\\n\\t\"
\t\t\"vsubpd %%ymm0, %%ymm14, %%ymm0\\n\\t\"
\t\t\"vaddpd %%ymm15, %%ymm{1}, %%ymm1\\n\\t\"\n",
                                        2 * i,
                                        2 * i + 1,
                                    );
                                }
                                InlineASM::AArch64 => {
                                    *out += &format!(
                                        "
\t\t\"fmul    d0, d{0}, d3\\n\\t\"
\t\t\"fmul    d1, d{1}, d3\\n\\t\"
\t\t\"fmadd   d3, d{0}, d2, d1\\n\\t\"
\t\t\"fnmsub  d2, d{1}, d2, d0\\n\\t\"\n",
                                        2 * (i + 1) + 1,
                                        2 * (i + 1),
                                    )
                                }
                                InlineASM::None => unreachable!(),
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
                        }
                    } else {
                        // TODO: reuse registers

                        end_asm_block!(in_asm_block);

                        let args = a
                            .iter()
                            .map(|x| get_input!(*x))
                            .collect::<Vec<_>>()
                            .join("*");

                        *out += format!("\tZ[{o}] = {args};\n").as_str();
                    }
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
                            InlineASM::AVX2 => {
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
                            InlineASM::AArch64 => {
                                if *b * 16 < 450 {
                                    *out += &format!("\t\t\"ldp d0, d1, {}\\n\\t\"", addr_b.0);
                                } else {
                                    *out += &format!("\t\t\"ldr d0, {}\\n\\t\"", addr_b.0);
                                    *out += &format!("\t\t\"ldr d1, {}\\n\\t\"", addr_b.1);
                                }

                                *out += "
\t\t\"fmul    d2, d0, d0\\n\\t\"
\t\t\"fmadd   d2, d1, d1, d2\\n\\t\"
\t\t\"fneg    d1, d1\\n\\t\"
\t\t\"fdiv    d0, d0, d2\\n\\t\"
\t\t\"fdiv    d1, d1, d2\\n\\t\"\n";

                                if *o * 16 < 450 {
                                    *out += &format!("\t\t\"stp d0, d1, {}\\n\\t\"\n", addr_o.0);
                                } else {
                                    *out += &format!("\t\t\"str d0, {}\\n\\t\"", addr_o.0);
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
                    *out += format!("\tZ[{o}] = pow({base}, {exp});\n").as_str();
                }
                Instr::BuiltinFun(o, s, a) => {
                    end_asm_block!(in_asm_block);

                    let arg = get_input!(*a);

                    match s.0 {
                        Symbol::EXP => {
                            *out += format!("\tZ[{o}] = exp({arg});\n").as_str();
                        }
                        Symbol::LOG => {
                            *out += format!("\tZ[{o}] = log({arg});\n").as_str();
                        }
                        Symbol::SIN => {
                            *out += format!("\tZ[{o}] = sin({arg});\n").as_str();
                        }
                        Symbol::COS => {
                            *out += format!("\tZ[{o}] = cos({arg});\n").as_str();
                        }
                        Symbol::SQRT => {
                            *out += format!("\tZ[{o}] = sqrt({arg});\n").as_str();
                        }
                        Symbol::CONJ => {
                            *out += format!("\tZ[{o}] = conj({arg});\n").as_str();
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
                    end_asm_block!(in_asm_block);
                    *out += "\t} else {\n";
                }
                Instr::Label(_) => {}
                Instr::Join(o, cond, a, b) => {
                    end_asm_block!(in_asm_block);
                    *out += "\t}\n";
                    let arg_a = get_input!(*a);
                    let arg_b = get_input!(*b);

                    if asm_flavour == InlineASM::AVX2 {
                        *out += &format!(
                            "\tZ[{}] = (all({} != 0.)) ? {} : {};\n",
                            o,
                            get_input!(*cond),
                            arg_a,
                            arg_b
                        );
                    } else {
                        *out += &format!(
                            "\tZ[{}] = ({} != 0.) ? {} : {};\n",
                            o,
                            get_input!(*cond),
                            arg_a,
                            arg_b
                        );
                    }
                }
            }
        }

        end_asm_block!(in_asm_block);

        *out += "\t__asm__(\n";
        for (i, r) in &mut self.result_indices.iter().enumerate() {
            if *r < self.param_count {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!("\t\t\"movupd {}(%3), %%xmm0\\n\\t\"\n", r * 16);
                    }
                    InlineASM::AVX2 => {
                        *out += &format!("\t\t\"vmovupd {}(%3), %%ymm0\\n\\t\"\n", r * 64);
                        *out += &format!("\t\t\"vmovupd {}(%3), %%ymm1\\n\\t\"\n", r * 64 + 32);
                    }
                    InlineASM::AArch64 => {
                        *out += &format!("\t\t\"ldr q0, [%3, {}]\\n\\t\"\n", r * 16);
                    }
                    InlineASM::None => unreachable!(),
                }
            } else if *r < self.reserved_indices {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!(
                            "\t\t\"movupd {}(%2), %%xmm0\\n\\t\"\n",
                            (r - self.param_count) * 16
                        );
                    }
                    InlineASM::AVX2 => {
                        *out += &format!(
                            "\t\t\"vmovupd {}(%2), %%ymm0\\n\\t\"\n",
                            (r - self.param_count) * 64
                        );
                        *out += &format!(
                            "\t\t\"vmovupd {}(%2), %%ymm1\\n\\t\"\n",
                            (r - self.param_count) * 64 + 32
                        );
                    }
                    InlineASM::AArch64 => {
                        *out += &format!(
                            "\t\t\"ldr q0, [%2, {}]\\n\\t\"\n",
                            (r - self.param_count) * 16
                        );
                    }

                    InlineASM::None => unreachable!(),
                }
            } else {
                match asm_flavour {
                    InlineASM::X64 => {
                        *out += &format!("\t\t\"movupd {}(%1), %%xmm0\\n\\t\"\n", r * 16);
                    }
                    InlineASM::AVX2 => {
                        *out += &format!("\t\t\"vmovupd {}(%1), %%ymm0\\n\\t\"\n", r * 64);
                        *out += &format!("\t\t\"vmovupd {}(%1), %%ymm1\\n\\t\"\n", r * 64 + 32);
                    }
                    InlineASM::AArch64 => {
                        *out += &format!("\t\t\"ldr q0, [%1, {}]\\n\\t\"\n", r * 16);
                    }
                    InlineASM::None => unreachable!(),
                }
            }

            match asm_flavour {
                InlineASM::X64 => {
                    *out += &format!("\t\t\"movupd %%xmm0, {}(%0)\\n\\t\"\n", i * 16);
                }
                InlineASM::AVX2 => {
                    *out += &format!("\t\t\"vmovupd %%ymm0, {}(%0)\\n\\t\"\n", i * 64);
                    *out += &format!("\t\t\"vmovupd %%ymm1, {}(%0)\\n\\t\"\n", i * 64 + 32);
                }
                InlineASM::AArch64 => {
                    *out += &format!("\t\t\"str q0, [%0, {}]\\n\\t\"\n", i * 16);
                }
                InlineASM::None => unreachable!(),
            }
        }

        match asm_flavour {
            InlineASM::X64 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(out), \"r\"(Z), \"r\"({function_name}_CONSTANTS_complex), \"r\"(params)\n\t\t: \"memory\", \"xmm0\", \"xmm1\", \"xmm2\", \"xmm3\", \"xmm4\", \"xmm5\", \"xmm6\", \"xmm7\", \"xmm8\", \"xmm9\", \"xmm10\", \"xmm11\", \"xmm12\", \"xmm13\", \"xmm14\", \"xmm15\");\n"
                );
            }
            InlineASM::AVX2 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(out), \"r\"(Z), \"r\"({function_name}_CONSTANTS_complex), \"r\"(params)\n\t\t: \"memory\", \"ymm0\", \"ymm1\", \"ymm2\", \"ymm3\", \"ymm4\", \"ymm5\", \"ymm6\", \"ymm7\", \"ymm8\", \"ymm9\", \"ymm10\", \"ymm11\", \"ymm12\", \"ymm13\", \"ymm14\", \"ymm15\");\n"
                );
            }
            InlineASM::AArch64 => {
                *out += &format!(
                    "\t\t:\n\t\t: \"r\"(out), \"r\"(Z), \"r\"({function_name}_CONSTANTS_complex), \"r\"(params)\n\t\t: \"memory\", \"d0\", \"d1\", \"d2\", \"d3\", \"d4\", \"d5\", \"d6\", \"d7\", \"d8\", \"d9\", \"d10\", \"d11\", \"d12\", \"d13\", \"d14\", \"d15\", \"d16\", \"d17\", \"d18\", \"d19\", \"d20\", \"d21\", \"d22\", \"d23\", \"d24\", \"d25\", \"d26\", \"d27\", \"d28\", \"d29\", \"d30\", \"d31\");\n"
                );
            }
            InlineASM::None => unreachable!(),
        }

        in_asm_block
    }
}

/// An optimized evaluator for expressions that can evaluate expressions with parameters
/// and some registered external functions.
pub struct ExpressionEvaluatorWithExternalFunctions<T> {
    stack: Vec<T>,
    param_count: usize,
    instructions: Vec<Instr>,
    result_indices: Vec<usize>,
    external_fns: Vec<(Vec<T>, Box<dyn Fn(&[T]) -> T + Send + Sync>)>,
}

impl<T: Real> ExpressionEvaluatorWithExternalFunctions<T> {
    #[allow(dead_code)]
    pub(crate) fn update_stack(&mut self, e: ExpressionEvaluator<T>) {
        self.stack = e.stack;
        self.param_count = e.param_count;
        self.instructions = e.instructions;
        self.result_indices = e.result_indices;
    }

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

    pub fn evaluate(&mut self, params: &[T], out: &mut [T]) {
        if self.param_count != params.len() {
            panic!(
                "Parameter count mismatch: expected {}, got {}",
                self.param_count,
                params.len()
            );
        }

        for (t, p) in self.stack.iter_mut().zip(params) {
            *t = p.clone();
        }

        let mut tmp;
        let mut i = 0;
        while i < self.instructions.len() {
            let instr = unsafe { self.instructions.get_unchecked(i) };
            match instr {
                Instr::Add(r, v) => {
                    tmp = self.stack[v[0]].clone();
                    for x in &v[1..] {
                        let e = self.stack[*x].clone();
                        tmp += e;
                    }
                    std::mem::swap(&mut self.stack[*r], &mut tmp);
                }
                Instr::Mul(r, v) => {
                    tmp = self.stack[v[0]].clone();
                    for x in &v[1..] {
                        let e = self.stack[*x].clone();
                        tmp *= e;
                    }
                    std::mem::swap(&mut self.stack[*r], &mut tmp);
                }
                Instr::Pow(r, b, e) => {
                    if *e >= 0 {
                        self.stack[*r] = self.stack[*b].pow(*e as u64);
                    } else {
                        self.stack[*r] = self.stack[*b].pow(e.unsigned_abs()).inv();
                    }
                }
                Instr::Powf(r, b, e) => {
                    self.stack[*r] = self.stack[*b].powf(&self.stack[*e]);
                }
                Instr::BuiltinFun(r, s, arg) => match s.0 {
                    Symbol::EXP => self.stack[*r] = self.stack[*arg].exp(),
                    Symbol::LOG => self.stack[*r] = self.stack[*arg].log(),
                    Symbol::SIN => self.stack[*r] = self.stack[*arg].sin(),
                    Symbol::COS => self.stack[*r] = self.stack[*arg].cos(),
                    Symbol::SQRT => self.stack[*r] = self.stack[*arg].sqrt(),
                    Symbol::CONJ => self.stack[*r] = self.stack[*arg].conj(),
                    _ => unreachable!(),
                },
                Instr::ExternalFun(r, s, args) => {
                    let (cache, f) = &mut self.external_fns[*s];

                    if cache.len() < args.len() {
                        cache.resize(args.len(), self.stack[0].clone());
                    }

                    for (i, v) in cache.iter_mut().zip(args) {
                        *i = self.stack[*v].clone();
                    }

                    self.stack[*r] = (f)(&cache[..args.len()]);
                }
                Instr::IfElse(n, label) => {
                    // jump to else block
                    if self.stack[*n].is_fully_zero() {
                        i = label.0;
                        continue;
                    }
                }
                Instr::Goto(label) => {
                    i = label.0;
                    continue;
                }
                Instr::Label(_) => {}
                Instr::Join(r, c, a, b) => {
                    if !self.stack[*c].is_fully_zero() {
                        self.stack[*r] = self.stack[*a].clone();
                    } else {
                        self.stack[*r] = self.stack[*b].clone();
                    }
                }
            }

            i += 1;
        }

        for (o, i) in out.iter_mut().zip(&self.result_indices) {
            *o = self.stack[*i].clone();
        }
    }
}

/// A slot in a list that contains a numerical value.
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "bincode", derive(bincode::Encode, bincode::Decode))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// `Add(o, [i0,...,i_n])` means `o = i0 + ... + i_n`.
    Add(Slot, Vec<Slot>),
    /// `Mul(o, [i0,...,i_n])` means `o = i0 * ... * i_n`.
    Mul(Slot, Vec<Slot>),
    /// `Pow(o, b, e)` means `o = b^e`.
    Pow(Slot, Slot, i64),
    /// `Powf(o, b, e)` means `o = b^e`.
    Powf(Slot, Slot, Slot),
    /// `Fun(o, s, a)` means `o = s(a)`, where `s` is assumed to
    /// be a built-in function such as `sin`.
    Fun(Slot, BuiltinSymbol, Slot),
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
            Instruction::Add(o, a) => {
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
            Instruction::Mul(o, a) => {
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
            Instruction::Pow(o, b, e) => {
                write!(f, "{o} = {b}^{e}")
            }
            Instruction::Powf(o, b, e) => {
                write!(f, "{o} = {b}^{e}")
            }
            Instruction::Fun(o, s, a) => {
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

        for i in &self.instructions {
            match i {
                Instr::Add(o, a) => {
                    instr.push(Instruction::Add(
                        get_slot!(*o),
                        a.iter().map(|x| get_slot!(*x)).collect(),
                    ));
                }
                Instr::Mul(o, a) => {
                    instr.push(Instruction::Mul(
                        get_slot!(*o),
                        a.iter().map(|x| get_slot!(*x)).collect(),
                    ));
                }
                Instr::Pow(o, b, e) => {
                    instr.push(Instruction::Pow(get_slot!(*o), get_slot!(*b), *e));
                }
                Instr::Powf(o, b, e) => {
                    instr.push(Instruction::Powf(
                        get_slot!(*o),
                        get_slot!(*b),
                        get_slot!(*e),
                    ));
                }
                Instr::BuiltinFun(o, s, a) => {
                    instr.push(Instruction::Fun(get_slot!(*o), *s, get_slot!(*a)));
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

impl<T: Default + Clone + std::fmt::Debug> ExpressionEvaluator<T> {
    /// Redefine every operator to take `n` components in and
    /// yield `n` components. This can be used for example to define dual numbers.
    pub fn vectorize<
        C: Fn(T) -> Vec<T>,
        F: Fn(&VectorInstruction, &mut InstructionAdder<T>) -> Vec<VectorInstruction>,
    >(
        mut self,
        coeff_map: C,
        map: F,
        dim: usize,
    ) -> ExpressionEvaluator<T> {
        // unfold every instruction to a single operation
        let mut new_instr = vec![];
        for x in &mut self.instructions {
            match x {
                Instr::Add(o, a) => {
                    new_instr.push(Instr::Add(*o, vec![a[0], a[1]]));
                    for x in a.iter().skip(2) {
                        new_instr.push(Instr::Add(*o, vec![*o, *x]));
                    }
                }
                Instr::Mul(o, a) => {
                    new_instr.push(Instr::Mul(*o, vec![a[0], a[1]]));
                    for x in a.iter().skip(2) {
                        new_instr.push(Instr::Mul(*o, vec![*o, *x]));
                    }
                }
                _ => new_instr.push(x.clone()),
            }
        }

        self.instructions = new_instr;

        self.undo_stack_optimization();

        let mut constants = vec![];
        for c in &self.stack[self.param_count..self.reserved_indices] {
            constants.extend(coeff_map(c.clone()));
        }
        let old_constants_num = constants.len();

        let mut slot_map = HashMap::default();
        for x in 0..self.reserved_indices {
            slot_map.insert(x, x * dim); // set the start of the vector
        }

        self.param_count *= dim;
        self.reserved_indices *= dim;

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

        let mut ins = InstructionAdder {
            instructions: vec![],
            constants,
            dim,
        };

        for i in self.instructions.drain(..) {
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
                Instr::ExternalFun(o, f, a) => (
                    o,
                    VectorInstruction::ExternalFun(
                        f,
                        a.iter().map(|x| get_slot!(slot_map[x])).collect(),
                    ),
                ),
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

            let r = map(&instr, &mut ins);
            assert_eq!(r.len(), dim);
            for ii in r {
                ins.add(ii);
            }

            slot_map.insert(o, ins.instructions.len() + self.reserved_indices - dim);
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

            for i in 0..dim {
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

                    self.instructions.push(Instr::Add(out, vec![s1, s2]));
                }
                VectorInstruction::Assign(slot) => {
                    self.instructions
                        .push(Instr::Add(out, vec![from_slot!(slot)]));
                }
                VectorInstruction::Mul(slot, slot1) => {
                    let mut s1 = from_slot!(slot);
                    let mut s2 = from_slot!(slot1);
                    if s1 > s2 {
                        (s1, s2) = (s2, s1);
                    }

                    self.instructions.push(Instr::Mul(out, vec![s1, s2]));
                }
                VectorInstruction::Pow(slot, e) => {
                    self.instructions.push(Instr::Pow(out, from_slot!(slot), e));
                }
                VectorInstruction::Powf(slot, slot1) => {
                    self.instructions
                        .push(Instr::Powf(out, from_slot!(slot), from_slot!(slot1)));
                }
                VectorInstruction::BuiltinFun(builtin_symbol, slot) => {
                    self.instructions.push(Instr::BuiltinFun(
                        out,
                        builtin_symbol,
                        from_slot!(slot),
                    ));
                }
                VectorInstruction::ExternalFun(f, args) => {
                    self.instructions.push(Instr::ExternalFun(
                        out,
                        f,
                        args.iter().map(|x| from_slot!(*x)).collect(),
                    ));
                }
                VectorInstruction::IfElse(cond, label) => {
                    self.instructions
                        .push(Instr::IfElse(from_slot!(cond), label));
                }
                VectorInstruction::Goto(label) => {
                    self.instructions.push(Instr::Goto(label));
                }
                VectorInstruction::Label(label) => {
                    self.instructions.push(Instr::Label(label));
                }
                VectorInstruction::Join(cond, t, f) => {
                    self.instructions.push(Instr::Join(
                        out,
                        from_slot!(cond),
                        from_slot!(t),
                        from_slot!(f),
                    ));
                }
            }
        }

        self.stack.resize(
            self.reserved_indices + self.instructions.len(),
            T::default(),
        );

        self
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

pub struct InstructionAdder<T> {
    instructions: Vec<VectorInstruction>,
    constants: Vec<T>,
    dim: usize,
}

impl<T> InstructionAdder<T> {
    pub fn add(&mut self, instr: VectorInstruction) -> Slot {
        self.instructions.push(instr);
        Slot::Temp(self.instructions.len() - 1)
    }
}

impl<T: PartialEq + Clone> InstructionAdder<T> {
    pub fn add_constant(&mut self, value: Vec<T>) -> Slot {
        assert_eq!(value.len(), self.dim);
        if let Some(c) = self.constants.chunks(self.dim).position(|x| x == &value) {
            Slot::Const(c)
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
            Slot::Const(c)
        } else {
            for _ in 0..self.dim {
                self.constants.push(value.clone());
            }
            Slot::Const(self.constants.len() - 1)
        }
    }
}

#[cfg(test)]
pub mod tests {
    use numerica::domains::rational::Rational;

    use crate::{
        atom::{AtomCore, Symbol},
        evaluate::{FunctionMap, OptimizationSettings, VectorInstruction},
        parse,
    };

    #[test]
    fn vectorize() {
        let ev = parse!("x^2 + x*y + sqrt(x^2) + 3/x")
            .evaluator(
                &FunctionMap::new(),
                &[parse!("x"), parse!("y")],
                OptimizationSettings::default(),
            )
            .unwrap();

        let mut vec_ev = ev.vectorize(
            |c| vec![c.clone(), c.zero()],
            |i, instrs| match i {
                VectorInstruction::Add(a, b) => {
                    vec![i.clone(), VectorInstruction::Add(a.index(1), b.index(1))]
                }
                VectorInstruction::Mul(a, b) => {
                    let ab_ = instrs.add(VectorInstruction::Mul(*a, b.index(1))); // a + b'
                    let a_b = instrs.add(VectorInstruction::Mul(a.index(1), *b)); // a + b'
                    vec![i.clone(), VectorInstruction::Add(ab_, a_b)] // a * b, // a * b' + a' * b
                }
                VectorInstruction::BuiltinFun(f, a) => match f.get_symbol() {
                    Symbol::SQRT => {
                        let sqrt_a = instrs.add(VectorInstruction::BuiltinFun(*f, *a)); // sqrt(a)
                        let two_sqrt_a = instrs.add(VectorInstruction::Add(sqrt_a, sqrt_a)); // 2*sqrt(a)
                        let one_over_two_sqrt_a =
                            instrs.add(VectorInstruction::Pow(two_sqrt_a, -1)); // 1/(2*sqrt(a))
                        vec![
                            VectorInstruction::Assign(sqrt_a),
                            VectorInstruction::Mul(a.index(1), one_over_two_sqrt_a),
                        ] // a' * 1/(2*sqrt(a))
                    }
                    _ => unimplemented!(),
                },
                VectorInstruction::Pow(a, b) => {
                    assert_eq!(*b, -1); // only b = -1 is used in practice

                    let min_one = instrs.add_repeated_constant(Rational::from(-1).into());
                    let inv_a = instrs.add(VectorInstruction::Pow(*a, -1));
                    let a_inv_sq = instrs.add(VectorInstruction::Mul(inv_a, inv_a));
                    let min_a = instrs.add(VectorInstruction::Mul(min_one, a_inv_sq));

                    vec![
                        VectorInstruction::Assign(inv_a),
                        VectorInstruction::Mul(min_a, a.index(1)),
                    ]
                }
                VectorInstruction::Powf(_, _) => {
                    todo!()
                }
                VectorInstruction::ExternalFun(_, _) => {
                    todo!("External functions not fully supported as it returns a single value")
                }
                VectorInstruction::Join(c, a, b) => {
                    vec![
                        i.clone(),
                        VectorInstruction::Join(c.index(1), a.index(1), b.index(1)),
                    ]
                }
                VectorInstruction::Assign(a) => {
                    vec![i.clone(), VectorInstruction::Assign(a.index(1))]
                }
                VectorInstruction::Goto(_)
                | VectorInstruction::Label(_)
                | VectorInstruction::IfElse(..) => {
                    unreachable!()
                }
            },
            2,
        );

        vec_ev.remove_common_pairs();
        vec_ev.optimize_stack();

        let (instr, _a, _constants) = vec_ev.export_instructions();
        //println!("constants: {:?}", constants);
        for x in instr {
            println!("{}", x);
        }

        let mut vec_f = vec_ev.map_coeff(&|x| x.re.to_f64());

        let mut dest = vec![0.; 2];
        vec_f.evaluate(&[2.0, 1.0, 3.0, 2.0], &mut dest);
        println!("{:?}", dest)
    }
}

/// A label in the instruction list [Instr].
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

impl<T: Clone + PartialEq> SplitExpression<T> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> SplitExpression<T2> {
        SplitExpression {
            tree: self.tree.iter().map(|x| x.map_coeff(f)).collect(),
            subexpressions: self.subexpressions.iter().map(|x| x.map_coeff(f)).collect(),
        }
    }
}

impl<T: Clone + PartialEq> Expression<T> {
    pub fn map_coeff<T2, F: Fn(&T) -> T2>(&self, f: &F) -> Expression<T2> {
        match self {
            Expression::Const(c) => Expression::Const(f(c)),
            Expression::Parameter(p) => Expression::Parameter(*p),
            Expression::Eval(id, e_args) => {
                Expression::Eval(*id, e_args.iter().map(|x| x.map_coeff(f)).collect())
            }
            Expression::Add(a) => {
                let new_args = a.iter().map(|x| x.map_coeff(f)).collect();
                Expression::Add(new_args)
            }
            Expression::Mul(m) => {
                let new_args = m.iter().map(|x| x.map_coeff(f)).collect();
                Expression::Mul(new_args)
            }
            Expression::Pow(p) => {
                let (b, e) = &**p;
                Expression::Pow(Box::new((b.map_coeff(f), *e)))
            }
            Expression::Powf(p) => {
                let (b, e) = &**p;
                Expression::Powf(Box::new((b.map_coeff(f), e.map_coeff(f))))
            }
            Expression::ReadArg(s) => Expression::ReadArg(*s),
            Expression::BuiltinFun(s, a) => Expression::BuiltinFun(*s, Box::new(a.map_coeff(f))),
            Expression::SubExpression(i) => Expression::SubExpression(*i),
            Expression::ExternalFun(s, a) => {
                let new_args = a.iter().map(|x| x.map_coeff(f)).collect();
                Expression::ExternalFun(*s, new_args)
            }
            Expression::IfElse(b) => {
                let (cond, then_expr, else_expr) = &**b;
                Expression::IfElse(Box::new((
                    cond.map_coeff(f),
                    then_expr.map_coeff(f),
                    else_expr.map_coeff(f),
                )))
            }
        }
    }

    fn strip_constants(&mut self, stack: &mut Vec<T>, param_len: usize) {
        match self {
            Expression::Const(t) => {
                if let Some(p) = stack.iter().skip(param_len).position(|x| x == t) {
                    *self = Expression::Parameter(param_len + p);
                } else {
                    stack.push(t.clone());
                    *self = Expression::Parameter(stack.len() - 1);
                }
            }
            Expression::Parameter(_) => {}
            Expression::Eval(_, e_args) => {
                for a in e_args {
                    a.strip_constants(stack, param_len);
                }
            }
            Expression::Add(a) | Expression::Mul(a) => {
                for arg in a {
                    arg.strip_constants(stack, param_len);
                }
            }
            Expression::Pow(p) => {
                p.0.strip_constants(stack, param_len);
            }
            Expression::Powf(p) => {
                p.0.strip_constants(stack, param_len);
                p.1.strip_constants(stack, param_len);
            }
            Expression::ReadArg(_) => {}
            Expression::BuiltinFun(_, a) => {
                a.strip_constants(stack, param_len);
            }
            Expression::SubExpression(_) => {}
            Expression::ExternalFun(_, a) => {
                for arg in a {
                    arg.strip_constants(stack, param_len);
                }
            }
            Expression::IfElse(b) => {
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

impl<T: Clone + Default + PartialEq> EvalTree<T> {
    /// Create a linear version of the tree that can be evaluated more efficiently.
    pub fn linearize(mut self, cpe_rounds: Option<usize>, verbose: bool) -> ExpressionEvaluator<T> {
        let mut stack = vec![T::default(); self.param_count];

        // strip every constant and move them into the stack after the params
        self.strip_constants(&mut stack);
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
        };

        for _ in 0..cpe_rounds.unwrap_or(usize::MAX) {
            let r = e.remove_common_pairs();
            if r == 0 {
                break;
            }
            if verbose {
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

    fn strip_constants(&mut self, stack: &mut Vec<T>) {
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
        tree: &Expression<T>,
        subexpressions: &[Expression<T>],
        stack: &mut Vec<T>,
        instr: &mut Vec<Instr>,
        sub_expr_pos: &mut HashMap<usize, usize>,
        args: &[usize],
        in_branch: bool,
    ) -> usize {
        match tree {
            Expression::Const(t) => {
                stack.push(t.clone()); // TODO: do once and recycle, this messes with the logic as there is no associated instruction
                stack.len() - 1
            }
            Expression::Parameter(i) => *i,
            Expression::Eval(id, e_args) => {
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
                        )
                    })
                    .collect();

                let mut sub_expr_pos = HashMap::default();
                let func = &self.functions[*id].2;
                self.linearize_impl(
                    &func.tree[0],
                    &func.subexpressions,
                    stack,
                    instr,
                    &mut sub_expr_pos,
                    &new_args,
                    in_branch,
                )
            }
            Expression::Add(a) => {
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
                        )
                    })
                    .collect();
                args.sort();

                stack.push(T::default());
                let res = stack.len() - 1;

                let add = Instr::Add(res, args);
                instr.push(add);

                res
            }
            Expression::Mul(m) => {
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
                        )
                    })
                    .collect();
                args.sort();

                stack.push(T::default());
                let res = stack.len() - 1;

                let mul = Instr::Mul(res, args);
                instr.push(mul);

                res
            }
            Expression::Pow(p) => {
                let b = self.linearize_impl(
                    &p.0,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    in_branch,
                );
                stack.push(T::default());
                let mut res = stack.len() - 1;

                if p.1 > 1 {
                    instr.push(Instr::Mul(res, vec![b; p.1 as usize]));
                } else if p.1 < -1 {
                    instr.push(Instr::Mul(res, vec![b; -p.1 as usize]));
                    stack.push(T::default());
                    res += 1;
                    instr.push(Instr::Pow(res, res - 1, -1));
                } else {
                    instr.push(Instr::Pow(res, b, p.1));
                }
                res
            }
            Expression::Powf(p) => {
                let b = self.linearize_impl(
                    &p.0,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    in_branch,
                );
                let e = self.linearize_impl(
                    &p.1,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    in_branch,
                );
                stack.push(T::default());
                let res = stack.len() - 1;

                instr.push(Instr::Powf(res, b, e));
                res
            }
            Expression::ReadArg(a) => args[*a],
            Expression::BuiltinFun(s, v) => {
                let arg = self.linearize_impl(
                    v,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    in_branch,
                );
                stack.push(T::default());
                let c = Instr::BuiltinFun(stack.len() - 1, *s, arg);
                instr.push(c);
                stack.len() - 1
            }
            Expression::SubExpression(id) => {
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
            Expression::ExternalFun(s, v) => {
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
                        )
                    })
                    .collect();

                stack.push(T::default());
                let res = stack.len() - 1;

                let f = Instr::ExternalFun(res, *s, args);
                instr.push(f);

                res
            }
            Expression::IfElse(b) => {
                let cond = self.linearize_impl(
                    &b.0,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    in_branch,
                );

                let label_else = Label(instr.len());
                stack.push(T::default());
                instr.push(Instr::IfElse(cond, label_else));

                let then_branch = self.linearize_impl(
                    &b.1,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    true,
                );

                let label_end = Label(instr.len());
                stack.push(T::default());
                instr.push(Instr::Goto(label_end));
                stack.push(T::default());
                instr.push(Instr::Label(label_else));

                let else_branch = self.linearize_impl(
                    &b.2,
                    subexpressions,
                    stack,
                    instr,
                    sub_expr_pos,
                    args,
                    true,
                );

                stack.push(T::default());
                instr.push(Instr::Label(label_end));

                stack.push(T::default());
                let res = stack.len() - 1;

                instr.push(Instr::Join(res, cond, then_branch, else_branch));

                res
            }
        }
    }
}

impl EvalTree<Complex<Rational>> {
    /// Find a near-optimal Horner scheme that minimizes the number of multiplications
    /// and additions, using `iterations` iterations of the optimization algorithm
    /// and `n_cores` cores. Optionally, a starting scheme can be provided.
    pub fn optimize(
        &mut self,
        settings: &OptimizationSettings,
    ) -> ExpressionEvaluator<Complex<Rational>> {
        let _ = self.optimize_horner_scheme(settings);
        self.common_subexpression_elimination();
        self.clone().linearize(None, settings.verbose)
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

                // for now, limit for parameters only
                v.retain(|(x, _)| matches!(x, Expression::Parameter(_)));
                v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
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

        for (_, _, e) in &mut self.functions {
            let mut v = HashMap::default();

            for e in &mut e.subexpressions {
                e.find_all_variables(&mut v);
            }

            let mut v: Vec<_> = v.into_iter().collect();
            v.retain(|(x, _)| matches!(x, Expression::Parameter(_)));
            v.sort_by_key(|k| std::cmp::Reverse(k.1));
            let v = v.into_iter().map(|(k, _)| k).collect::<Vec<_>>();

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
            Expression::Add(a) => a,
            Expression::Eval(_, a) => {
                for arg in a {
                    arg.apply_horner_scheme(scheme);
                }
                return;
            }
            Expression::Mul(m) => {
                for a in m {
                    a.apply_horner_scheme(scheme);
                }
                return;
            }
            Expression::Pow(b) => {
                b.0.apply_horner_scheme(scheme);
                return;
            }
            Expression::Powf(b) => {
                b.0.apply_horner_scheme(scheme);
                b.1.apply_horner_scheme(scheme);
                return;
            }
            Expression::BuiltinFun(_, b) => {
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
            if let Expression::Mul(m) = x {
                let mut pow_counter = 0;
                for y in m {
                    if let Expression::Pow(p) = y {
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
            } else if let Expression::Pow(p) = x {
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

        let mut contains = vec![];
        let mut rest = vec![];

        for mut x in a.drain(..) {
            let mut found = false;
            if let Expression::Mul(m) = &mut x {
                let mut pow_counter = 0;

                m.retain(|y| {
                    if let Expression::Pow(p) = y {
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
                        m.push(Expression::Pow(Box::new((
                            scheme[0].clone(),
                            pow_counter - max_pow,
                        ))));
                    } else {
                        m.push(scheme[0].clone());
                    }

                    m.sort();
                }

                if m.is_empty() {
                    x = Expression::Const(Complex::new_one());
                } else if m.len() == 1 {
                    x = m.pop().unwrap();
                }

                found = pow_counter > 0;
            } else if let Expression::Pow(p) = &mut x {
                if p.0 == scheme[0] && p.1 > 0 {
                    if p.1 > max_pow + 1 {
                        p.1 -= max_pow;
                    } else if p.1 - max_pow == 1 {
                        x = scheme[0].clone();
                    } else {
                        x = Expression::Const(Complex::new_one());
                    }
                    found = true;
                }
            } else if x == scheme[0] {
                found = true;
                x = Expression::Const(Complex::new_one());
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
            Expression::Pow(Box::new((scheme[0].clone(), max_pow)))
        };

        let mut contains = if contains.len() == 1 {
            contains.pop().unwrap()
        } else {
            Expression::Add(contains)
        };

        contains.apply_horner_scheme(scheme); // keep trying with same variable

        let mut v = vec![];
        if let Expression::Mul(a) = contains {
            v.extend(a);
        } else {
            v.push(contains);
        }

        v.push(extracted);
        v.retain(|x| *x != Expression::Const(Rational::one().into()));
        v.sort();

        let c = if v.len() == 1 {
            v.pop().unwrap()
        } else {
            Expression::Mul(v)
        };

        if rest.is_empty() {
            *self = c;
        } else {
            let mut r = if rest.len() == 1 {
                rest.pop().unwrap()
            } else {
                Expression::Add(rest)
            };

            r.apply_horner_scheme(&scheme[1..]);

            a.clear();
            a.push(c);

            if let Expression::Add(aa) = r {
                a.extend(aa);
            } else {
                a.push(r);
            }

            a.sort();
        }
    }

    /// Apply a simple occurrence-order Horner scheme to every addition.
    pub fn occurrence_order_horner_scheme(&mut self) {
        match self {
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => {}
            Expression::Eval(_, ae) => {
                for arg in ae {
                    arg.occurrence_order_horner_scheme();
                }
            }
            Expression::Add(a) => {
                for arg in &mut *a {
                    arg.occurrence_order_horner_scheme();
                }

                let mut occurrence = HashMap::default();

                for arg in &*a {
                    match arg {
                        Expression::Mul(m) => {
                            for aa in m {
                                if let Expression::Pow(p) = aa {
                                    occurrence
                                        .entry(p.0.clone())
                                        .and_modify(|x| *x += 1)
                                        .or_insert(1);
                                } else {
                                    occurrence
                                        .entry(aa.clone())
                                        .and_modify(|x| *x += 1)
                                        .or_insert(1);
                                }
                            }
                        }
                        x => {
                            if let Expression::Pow(p) = x {
                                occurrence
                                    .entry(p.0.clone())
                                    .and_modify(|x| *x += 1)
                                    .or_insert(1);
                            } else {
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
                order.sort_by_key(|k| std::cmp::Reverse(k.1)); // occurrence order
                let scheme = order.into_iter().map(|(k, _)| k).collect::<Vec<_>>();

                self.apply_horner_scheme(&scheme);
            }
            Expression::Mul(a) => {
                for arg in a {
                    arg.occurrence_order_horner_scheme();
                }
            }
            Expression::Pow(p) => {
                p.0.occurrence_order_horner_scheme();
            }
            Expression::Powf(p) => {
                p.0.occurrence_order_horner_scheme();
                p.1.occurrence_order_horner_scheme();
            }
            Expression::BuiltinFun(_, a) => {
                a.occurrence_order_horner_scheme();
            }
            Expression::SubExpression(_) => {}
            Expression::ExternalFun(_, a) => {
                for arg in a {
                    arg.occurrence_order_horner_scheme();
                }
            }
            Expression::IfElse(b) => {
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
                h
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
                "Initial ops: {} additions and {} multiplications",
                best_ops.0, best_ops.1
            );
        }

        let best_mul = Arc::new(AtomicUsize::new(best_ops.1));
        let best_add = Arc::new(AtomicUsize::new(best_ops.0));
        let best_scheme = Arc::new(Mutex::new(vars.to_vec()));

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
        };

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
                    for j in 0..settings.horner_iterations / n_cores {
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
                                h.to_hashed_expression().1
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
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => {}
            Expression::Eval(_, ae) => {
                for arg in ae {
                    arg.find_all_variables(vars);
                }
            }
            Expression::Add(a) => {
                for arg in a {
                    arg.find_all_variables(vars);
                }

                for arg in a {
                    match arg {
                        Expression::Mul(m) => {
                            for aa in m {
                                if let Expression::Pow(p) = aa {
                                    vars.entry(p.0.clone()).and_modify(|x| *x += 1).or_insert(1);
                                } else {
                                    vars.entry(aa.clone()).and_modify(|x| *x += 1).or_insert(1);
                                }
                            }
                        }
                        x => {
                            if let Expression::Pow(p) = x {
                                vars.entry(p.0.clone()).and_modify(|x| *x += 1).or_insert(1);
                            } else {
                                vars.entry(x.clone()).and_modify(|x| *x += 1).or_insert(1);
                            }
                        }
                    }
                }
            }
            Expression::Mul(a) => {
                for arg in a {
                    arg.find_all_variables(vars);
                }
            }
            Expression::Pow(p) => {
                p.0.find_all_variables(vars);
            }
            Expression::Powf(p) => {
                p.0.find_all_variables(vars);
                p.1.find_all_variables(vars);
            }
            Expression::BuiltinFun(_, a) => {
                a.find_all_variables(vars);
            }
            Expression::SubExpression(_) => {}
            Expression::ExternalFun(_, a) => {
                for arg in a {
                    arg.find_all_variables(vars);
                }
            }
            Expression::IfElse(b) => {
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
        let mut h = HashMap::default();

        let mut hashed_tree = vec![];
        for t in &self.tree {
            let (_, t) = t.to_hashed_expression();
            hashed_tree.push(t);
        }

        for t in &hashed_tree {
            t.find_subexpression(&mut h);
        }

        h.retain(|_, v| *v > 1);

        let mut v: Vec<_> = h.iter().map(|(k, v)| (*v, (*k).clone())).collect();
        v.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        // make the second argument a unique index of the subexpression
        for (i, (index, e)) in v.iter_mut().enumerate() {
            *index = self.subexpressions.len() + i;
            *h.get_mut(e).unwrap() = *index;
        }

        let mut n_hash_tree = hashed_tree.clone();
        for t in &mut n_hash_tree {
            t.replace_subexpression(&h, false);
        }

        self.tree = n_hash_tree.iter().map(|x| x.to_expression()).collect();

        // replace subexpressions in subexpressions and
        // sort them based on their dependencies
        for (_, x) in v {
            let mut he = x.clone();
            he.replace_subexpression(&h, true);
            self.subexpressions.push(he.to_expression());
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
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => {}
            Expression::Eval(_, ae) => {
                for arg in &mut *ae {
                    arg.rename_subexpression(subexp);
                }
            }
            Expression::Add(a) | Expression::Mul(a) => {
                for arg in &mut *a {
                    arg.rename_subexpression(subexp);
                }

                a.sort();
            }
            Expression::Pow(p) => {
                p.0.rename_subexpression(subexp);
            }
            Expression::Powf(p) => {
                p.0.rename_subexpression(subexp);
                p.1.rename_subexpression(subexp);
            }
            Expression::BuiltinFun(_, a) => {
                a.rename_subexpression(subexp);
            }
            Expression::SubExpression(i) => {
                *self = Expression::SubExpression(*subexp.get(i).unwrap());
            }
            Expression::ExternalFun(_, a) => {
                for arg in a {
                    arg.rename_subexpression(subexp);
                }
            }
            Expression::IfElse(b) => {
                b.0.rename_subexpression(subexp);
                b.1.rename_subexpression(subexp);
                b.2.rename_subexpression(subexp);
            }
        }
    }

    fn get_dependent_subexpressions(&self, dep: &mut Vec<usize>) {
        match self {
            Expression::Const(_) | Expression::Parameter(_) | Expression::ReadArg(_) => {}
            Expression::Eval(_, ae) => {
                for arg in ae {
                    arg.get_dependent_subexpressions(dep);
                }
            }
            Expression::Add(a) | Expression::Mul(a) => {
                for arg in a {
                    arg.get_dependent_subexpressions(dep);
                }
            }
            Expression::Pow(p) => {
                p.0.get_dependent_subexpressions(dep);
            }
            Expression::Powf(p) => {
                p.0.get_dependent_subexpressions(dep);
                p.1.get_dependent_subexpressions(dep);
            }
            Expression::BuiltinFun(_, a) => {
                a.get_dependent_subexpressions(dep);
            }
            Expression::SubExpression(i) => {
                dep.push(*i);
            }
            Expression::ExternalFun(_, a) => {
                for arg in a {
                    arg.get_dependent_subexpressions(dep);
                }
            }
            Expression::IfElse(b) => {
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
            Expression::Const(_) => (0, 0),
            Expression::Parameter(_) => (0, 0),
            Expression::Eval(_, args) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in args {
                    let (a, m) = arg.count_operations();
                    add += a;
                    mul += m;
                }
                (add, mul)
            }
            Expression::Add(a) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in a {
                    let (a, m) = arg.count_operations();
                    add += a;
                    mul += m;
                }
                (add + a.len() - 1, mul)
            }
            Expression::Mul(m) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in m {
                    let (a, m) = arg.count_operations();
                    add += a;
                    mul += m;
                }
                (add, mul + m.len() - 1)
            }
            Expression::Pow(p) => {
                let (a, m) = p.0.count_operations();
                (a, m + p.1.unsigned_abs() as usize - 1)
            }
            Expression::Powf(p) => {
                let (a, m) = p.0.count_operations();
                let (a2, m2) = p.1.count_operations();
                (a + a2, m + m2 + 1) // not clear how to count this
            }
            Expression::ReadArg(_) => (0, 0),
            Expression::BuiltinFun(_, b) => b.count_operations(), // not clear how to count this, third arg?
            Expression::SubExpression(_) => (0, 0),
            Expression::ExternalFun(_, args) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in args {
                    let (a, m) = arg.count_operations();
                    add += a;
                    mul += m;
                }
                (add, mul)
            }
            Expression::IfElse(b) => {
                let (a1, m1) = b.0.count_operations();
                let (a2, m2) = b.1.count_operations();
                let (a3, m3) = b.2.count_operations();
                (a1 + a2 + a3, m1 + m2 + m3)
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
            Expression::Const(_) | Expression::Parameter(_,) | Expression::ReadArg(_)
        ) {
            return (0, 0);
        }

        if sub_expr.contains_key(self) {
            return (0, 0);
        }

        sub_expr.insert(self, 1);

        match self {
            Expression::Const(_) => (0, 0),
            Expression::Parameter(_) => (0, 0),
            Expression::Eval(_, args) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in args {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add, mul)
            }
            Expression::Add(a) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in a {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add + a.len() - 1, mul)
            }
            Expression::Mul(m) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in m {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add, mul + m.len() - 1)
            }
            Expression::Pow(p) => {
                let (a, m) = p.0.count_operations_with_subexpression(sub_expr);
                (a, m + p.1.unsigned_abs() as usize - 1)
            }
            Expression::Powf(p) => {
                let (a, m) = p.0.count_operations_with_subexpression(sub_expr);
                let (a2, m2) = p.1.count_operations_with_subexpression(sub_expr);
                (a + a2, m + m2 + 1) // not clear how to count this
            }
            Expression::ReadArg(_) => (0, 0),
            Expression::BuiltinFun(_, b) => b.count_operations_with_subexpression(sub_expr), // not clear how to count this, third arg?
            Expression::SubExpression(_) => (0, 0),
            Expression::ExternalFun(_, args) => {
                let mut add = 0;
                let mut mul = 0;
                for arg in args {
                    let (a, m) = arg.count_operations_with_subexpression(sub_expr);
                    add += a;
                    mul += m;
                }
                (add, mul)
            }
            Expression::IfElse(b) => {
                let (a1, m1) = b.0.count_operations_with_subexpression(sub_expr);
                let (a2, m2) = b.1.count_operations_with_subexpression(sub_expr);
                let (a3, m3) = b.2.count_operations_with_subexpression(sub_expr);
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
            Expression::Const(c) => c.clone(),
            Expression::Parameter(p) => params[*p].clone(),
            Expression::Eval(f, e_args) => {
                let mut arg_buf = vec![T::new_zero(); e_args.len()];
                for (b, a) in arg_buf.iter_mut().zip(e_args.iter()) {
                    *b = self.evaluate_impl(a, subexpressions, params, args);
                }

                let func = &self.functions[*f].2;
                self.evaluate_impl(&func.tree[0], &func.subexpressions, params, &arg_buf)
            }
            Expression::Add(a) => {
                let mut r = self.evaluate_impl(&a[0], subexpressions, params, args);
                for arg in &a[1..] {
                    r += self.evaluate_impl(arg, subexpressions, params, args);
                }
                r
            }
            Expression::Mul(m) => {
                let mut r = self.evaluate_impl(&m[0], subexpressions, params, args);
                for arg in &m[1..] {
                    r *= self.evaluate_impl(arg, subexpressions, params, args);
                }
                r
            }
            Expression::Pow(p) => {
                let (b, e) = &**p;
                let b_eval = self.evaluate_impl(b, subexpressions, params, args);

                if *e >= 0 {
                    b_eval.pow(*e as u64)
                } else {
                    b_eval.pow(e.unsigned_abs()).inv()
                }
            }
            Expression::Powf(p) => {
                let (b, e) = &**p;
                let b_eval = self.evaluate_impl(b, subexpressions, params, args);
                let e_eval = self.evaluate_impl(e, subexpressions, params, args);
                b_eval.powf(&e_eval)
            }
            Expression::ReadArg(i) => args[*i].clone(),
            Expression::BuiltinFun(s, a) => {
                let arg = self.evaluate_impl(a, subexpressions, params, args);
                match s.0 {
                    Symbol::EXP => arg.exp(),
                    Symbol::LOG => arg.log(),
                    Symbol::SIN => arg.sin(),
                    Symbol::COS => arg.cos(),
                    Symbol::SQRT => arg.sqrt(),
                    Symbol::CONJ => arg.conj(),
                    _ => unreachable!(),
                }
            }
            Expression::SubExpression(s) => {
                // TODO: cache
                self.evaluate_impl(&subexpressions[*s], subexpressions, params, args)
            }
            Expression::ExternalFun(name, _args) => {
                unimplemented!(
                    "External function calls not implemented for EvalTree: {}",
                    name
                );
            }
            Expression::IfElse(b) => {
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

/// Batch-evaluate the compiled code with basic types such as [f64] or [Complex<f64>],
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

impl<T: FloatLike> EvalTree<T> {
    /// Export the evaluation tree to C++ code. For much improved performance,
    /// optimize the tree instead.
    pub fn export_cpp_str(&self, function_name: &str, include_header: bool) -> String {
        let mut res = if include_header {
            "#include <iostream>\n#include <cmath>\n#include <complex>\n\n".to_string()
        } else {
            String::new()
        };

        for (name, arg_names, body) in &self.functions {
            let mut args = arg_names
                .iter()
                .map(|x| " T ".to_string() + x.to_string().as_str())
                .collect::<Vec<_>>();
            args.insert(0, "T* params".to_string());

            res += &format!(
                "\ntemplate<typename T>\nT {}({}) {{\n",
                name,
                args.join(",")
            );

            for (i, s) in body.subexpressions.iter().enumerate() {
                res += &format!("\tT Z{}_ = {};\n", i, self.export_cpp_impl(s, arg_names));
            }

            if body.tree.len() > 1 {
                panic!("Tensor functions not supported yet");
            }

            let ret = self.export_cpp_impl(&body.tree[0], arg_names);
            res += &format!("\treturn {ret};\n}}\n");
        }

        res += &format!("\ntemplate<typename T>\nvoid {function_name}(T* params, T* out) {{\n");

        for (i, s) in self.expressions.subexpressions.iter().enumerate() {
            res += &format!("\tT Z{}_ = {};\n", i, self.export_cpp_impl(s, &[]));
        }

        for (i, e) in self.expressions.tree.iter().enumerate() {
            res += &format!("\tout[{}] = {};\n", i, self.export_cpp_impl(e, &[]));
        }

        res += "\treturn;\n}\n";

        res += &format!(
            "\nextern \"C\" {{\n\tvoid {function_name}_double(double* params, double* out) {{\n\t\t{function_name}(params, out);\n\t\treturn;\n\t}}\n}}\n"
        );
        res += &format!(
            "\nextern \"C\" {{\n\tvoid {function_name}_complex(std::complex<double>* params, std::complex<double>* out) {{\n\t\t{function_name}(params, out);\n\t\treturn;\n\t}}\n}}\n"
        );

        res
    }

    fn export_cpp_impl(&self, expr: &Expression<T>, args: &[Symbol]) -> String {
        match expr {
            Expression::Const(c) => {
                format!("T({c})")
            }
            Expression::Parameter(p) => {
                format!("params[{p}]")
            }
            Expression::Eval(id, e_args) => {
                let mut r = format!("{}(params", self.functions[*id].0);

                for a in e_args {
                    r.push_str(", ");
                    r += &self.export_cpp_impl(a, args);
                }
                r.push(')');
                r
            }
            Expression::Add(a) => {
                let mut r = "(".to_string();
                r += &self.export_cpp_impl(&a[0], args);
                for arg in &a[1..] {
                    r.push_str(" + ");
                    r += &self.export_cpp_impl(arg, args);
                }
                r.push(')');
                r
            }
            Expression::Mul(m) => {
                let mut r = "(".to_string();
                r += &self.export_cpp_impl(&m[0], args);
                for arg in &m[1..] {
                    r.push_str(" * ");
                    r += &self.export_cpp_impl(arg, args);
                }
                r.push(')');
                r
            }
            Expression::Pow(p) => {
                let mut r = "pow(".to_string();
                r += &self.export_cpp_impl(&p.0, args);
                r.push_str(", ");
                r.push_str(&p.1.to_string());
                r.push(')');
                r
            }
            Expression::Powf(p) => {
                let mut r = "powf(".to_string();
                r += &self.export_cpp_impl(&p.0, args);
                r.push_str(", ");
                r += &self.export_cpp_impl(&p.1, args);
                r.push(')');
                r
            }
            Expression::ReadArg(s) => args[*s].to_string(),
            Expression::BuiltinFun(s, a) => match s.0 {
                Symbol::EXP => {
                    let mut r = "exp(".to_string();
                    r += &self.export_cpp_impl(a, args);
                    r.push(')');
                    r
                }
                Symbol::LOG => {
                    let mut r = "log(".to_string();
                    r += &self.export_cpp_impl(a, args);
                    r.push(')');
                    r
                }
                Symbol::SIN => {
                    let mut r = "sin(".to_string();
                    r += &self.export_cpp_impl(a, args);
                    r.push(')');
                    r
                }
                Symbol::COS => {
                    let mut r = "cos(".to_string();
                    r += &self.export_cpp_impl(a, args);
                    r.push(')');
                    r
                }
                Symbol::SQRT => {
                    let mut r = "sqrt(".to_string();
                    r += &self.export_cpp_impl(a, args);
                    r.push(')');
                    r
                }
                Symbol::CONJ => {
                    let mut r = "conj(".to_string();
                    r += &self.export_cpp_impl(a, args);
                    r.push(')');
                    r
                }
                _ => unreachable!(),
            },
            Expression::SubExpression(id) => {
                format!("Z{id}_")
            }
            Expression::ExternalFun(name, a) => {
                let mut r = name.to_string();
                r.push('(');
                r += &a
                    .iter()
                    .map(|x| self.export_cpp_impl(x, args))
                    .collect::<Vec<_>>()
                    .join(", ");
                r.push(')');
                r
            }
            Expression::IfElse(b) => {
                let mut r = "(".to_string();
                r += &self.export_cpp_impl(&b.0, args);
                r.push_str(" ? (");
                r += &self.export_cpp_impl(&b.1, args);
                r.push_str(") : (");
                r += &self.export_cpp_impl(&b.2, args);
                r.push_str("))");
                r
            }
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
        let tree = exprs
            .iter()
            .map(|t| {
                t.as_atom_view()
                    .to_eval_tree_impl(fn_map, params, &[], &mut funcs)
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
        args: &[Symbol],
        funcs: &mut Vec<(String, Vec<Symbol>, SplitExpression<Complex<Rational>>)>,
    ) -> Result<Expression<Complex<Rational>>, String> {
        if let Some(p) = params.iter().position(|a| a.as_view() == *self) {
            return Ok(Expression::Parameter(p));
        }

        if let Some(c) = fn_map.get_constant(*self) {
            return Ok(Expression::Const(c.clone()));
        }

        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(n, d, ni, di) => Ok(Expression::Const(Complex::new(
                    Rational::from((n, d)),
                    Rational::from((ni, di)),
                ))),
                CoefficientView::Large(l, i) => {
                    Ok(Expression::Const(Complex::new(l.to_rat(), i.to_rat())))
                }
                CoefficientView::Float(r, i) => {
                    // TODO: converting back to rational is slow
                    Ok(Expression::Const(Complex::new(
                        r.to_float().to_rational(),
                        i.to_float().to_rational(),
                    )))
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
            AtomView::Var(v) => {
                let name = v.get_symbol();

                if let Some(p) = args.iter().position(|s| *s == name) {
                    return Ok(Expression::ReadArg(p));
                }

                Err(format!("Variable {} not in constant map", name.get_name()))
            }
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [
                    Symbol::EXP,
                    Symbol::LOG,
                    Symbol::SIN,
                    Symbol::COS,
                    Symbol::SQRT,
                    Symbol::CONJ,
                ]
                .contains(&name)
                {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.to_eval_tree_impl(fn_map, params, args, funcs)?;

                    return Ok(Expression::BuiltinFun(
                        BuiltinSymbol(f.get_symbol()),
                        Box::new(arg_eval),
                    ));
                }

                let Some(fun) = fn_map.get(*self) else {
                    return Err(format!("Undefined function {}", self.to_plain_string()));
                };

                match fun {
                    ConstOrExpr::Const(t) => Ok(Expression::Const(t.clone())),
                    ConstOrExpr::External(e, _name) => {
                        let eval_args = f
                            .iter()
                            .map(|arg| arg.to_eval_tree_impl(fn_map, params, args, funcs))
                            .collect::<Result<_, _>>()?;
                        Ok(Expression::ExternalFun(*e, eval_args))
                    }
                    ConstOrExpr::Expr(Expr {
                        name,
                        tag_len,
                        args: arg_spec,
                        body: e,
                    }) => {
                        if f.get_nargs() != arg_spec.len() + *tag_len {
                            return Err(format!(
                                "Function {} called with wrong number of arguments: {} vs {}",
                                f.get_symbol().get_name(),
                                f.get_nargs(),
                                arg_spec.len() + *tag_len
                            ));
                        }

                        let eval_args = f
                            .iter()
                            .skip(*tag_len)
                            .map(|arg| arg.to_eval_tree_impl(fn_map, params, args, funcs))
                            .collect::<Result<_, _>>()?;

                        if let Some(pos) = funcs.iter().position(|f| f.0 == *name) {
                            Ok(Expression::Eval(pos, eval_args))
                        } else {
                            let r = e
                                .as_view()
                                .to_eval_tree_impl(fn_map, params, arg_spec, funcs)?;
                            funcs.push((
                                name.clone(),
                                arg_spec.clone(),
                                SplitExpression {
                                    tree: vec![r.clone()],
                                    subexpressions: vec![],
                                },
                            ));
                            Ok(Expression::Eval(funcs.len() - 1, eval_args))
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
                            .to_eval_tree_impl(fn_map, params, args, funcs)?;
                        let t_eval = arg_iter
                            .next()
                            .unwrap()
                            .to_eval_tree_impl(fn_map, params, args, funcs)?;
                        let f_eval = arg_iter
                            .next()
                            .unwrap()
                            .to_eval_tree_impl(fn_map, params, args, funcs)?;

                        Ok(Expression::IfElse(Box::new((cond_eval, t_eval, f_eval))))
                    }
                }
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                let b_eval = b.to_eval_tree_impl(fn_map, params, args, funcs)?;

                if let AtomView::Num(n) = e
                    && let CoefficientView::Natural(num, den, num_i, _den_i) = n.get_coeff_view()
                    && den == 1
                    && num_i == 0
                {
                    return Ok(Expression::Pow(Box::new((b_eval.clone(), num))));
                }

                let e_eval = e.to_eval_tree_impl(fn_map, params, args, funcs)?;
                Ok(Expression::Powf(Box::new((b_eval, e_eval))))
            }
            AtomView::Mul(m) => {
                let mut muls = vec![];
                for arg in m.iter() {
                    let a = arg.to_eval_tree_impl(fn_map, params, args, funcs)?;
                    if let Expression::Mul(m) = a {
                        muls.extend(m);
                    } else {
                        muls.push(a);
                    }
                }

                muls.sort();

                Ok(Expression::Mul(muls))
            }
            AtomView::Add(a) => {
                let mut adds = vec![];
                for arg in a.iter() {
                    adds.push(arg.to_eval_tree_impl(fn_map, params, args, funcs)?);
                }

                adds.sort();

                Ok(Expression::Add(adds))
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
            AtomView::Var(v) => match v.get_symbol() {
                Symbol::E => Ok(coeff_map(&1.into()).e()),
                Symbol::PI => Ok(coeff_map(&1.into()).pi()),
                s => {
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
            },
            AtomView::Fun(f) => {
                let name = f.get_symbol();
                if [
                    Symbol::EXP,
                    Symbol::LOG,
                    Symbol::SIN,
                    Symbol::COS,
                    Symbol::SQRT,
                    Symbol::CONJ,
                ]
                .contains(&name)
                {
                    assert!(f.get_nargs() == 1);
                    let arg = f.iter().next().unwrap();
                    let arg_eval = arg.evaluate_impl(coeff_map, const_map, function_map, cache)?;

                    return Ok(match f.get_symbol() {
                        Symbol::EXP => arg_eval.exp(),
                        Symbol::LOG => arg_eval.log(),
                        Symbol::SIN => arg_eval.sin(),
                        Symbol::COS => arg_eval.cos(),
                        Symbol::SQRT => arg_eval.sqrt(),
                        Symbol::CONJ => arg_eval.conj(),
                        _ => unreachable!(),
                    });
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

        if self.has_complex_coefficients() || self.has_roots() {
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

    use crate::{
        atom::{Atom, AtomCore},
        domains::{
            float::{Complex, Float},
            rational::Rational,
        },
        evaluate::{EvaluationFn, FunctionMap, OptimizationSettings},
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
}
