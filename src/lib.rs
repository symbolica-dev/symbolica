//! Symbolica is a blazing fast computer algebra system.
//!
//! Its main features are:
//! - Easily create and manipulate expressions in Rust and Python
//! - Fast code generation (C++/ASM/SIMD/CUDA) for expression evaluation
//! - Fast multivariate polynomial arithmetic
//! - Pattern matching and expression transformation
//! - Mixed exact and numerical computations with error propagation
//! - Handling and compression of very large expressions
//!
//! For example:
//!
//! ```
//! use symbolica::prelude::*;
//!
//! fn main() {
//!     let input = parse!("x^2*log(2*x + y) + exp(3*x)");
//!     let a = input.derivative(symbol!("x"));
//!     println!("d/dx {} = {}:", input, a);
//! }
//! ```
//!
//! The main object to represent a general expressions is [Atom](atom::Atom). Most operations on [Atom](atom::Atom) are
//! implemented as methods on the [AtomCore](atom::AtomCore) trait. The [Symbol](atom::Symbol) struct is used to represent
//! variables or named functions, potentially with additional properties, such as symmetries (see [atom::SymbolAttribute]).
//!
//! Instead of using general expressions, you can use more restricted formats such as [MultivariatePolynomial](poly::polynomial::MultivariatePolynomial), [UnivariatePolynomial](poly::univariate::UnivariatePolynomial) and [RationalPolynomial](domains::rational_polynomial::RationalPolynomial)
//! which have optimized methods.
//!
//! To use Symbolica's exact numbers, see [Integer](domains::integer::Integer), [Rational](domains::rational::Rational), and [FiniteField](domains::finite_field::FiniteField).
//! For evaluations with floating point numbers, see [Float](domains::float::Float), [F64](domains::float::F64) and [ErrorPropagatingFloat](domains::float::ErrorPropagatingFloat).
//!
//! For linear algebra, use [Matrix](tensors::matrix::Matrix) or [Vector](tensors::matrix::Vector).
//!
//! Check out the [guide](https://symbolica.io/docs/get_started.html) for more information, examples,
//! and additional documentation.

#![cfg_attr(docsrs, feature(doc_cfg))]

use std::sync::atomic::AtomicBool;

#[cfg(feature = "python_export")]
pub mod api;
#[cfg(not(feature = "python_export"))]
mod api;
pub mod atom;
pub mod coefficient;
mod collect;
mod derivative;
pub mod domains;
pub mod evaluate;
mod expand;
pub mod id;
pub mod license;
mod normalize;
pub mod parser;
pub mod poly;
pub mod printer;
pub mod solve;
pub mod state;
pub mod streaming;
pub mod tensors;
/// Built-in transcendental functions, constants, and their symbolic and numerical behavior.
pub mod transcendental;
pub mod transformer;
pub mod utils;

/// Common imports for working with Symbolica.
///
/// The prelude is intended for examples, applications, and notebooks where a compact import is more
/// useful than listing every trait and constructor separately.
///
/// ```
/// use symbolica::prelude::*;
///
/// let x = symbol!("x");
/// let expr = parse!("x^2 + 2*x + 1");
/// assert_eq!(expr.derivative(x), parse!("2 + 2*x"));
/// ```
pub mod prelude {
    pub use crate::rand::{Rng, RngCore, SeedableRng};

    pub use crate::{
        create_hyperdual_from_components, create_hyperdual_single_derivative, function, get_symbol,
        hide_namespace, initialize, namespace, parse, parse_lit, symbol, symbol_group, tag,
        try_parse, try_parse_lit, try_symbol, try_symbol_group,
    };

    pub use crate::atom::{
        AliasedAtom, Atom, AtomCore, AtomIndex, AtomOrView, AtomType, AtomView, EvaluationError,
        EvaluationInfo, FunctionArgument, FunctionBuilder, Indeterminate, InlineNum, InlineVar,
        PolynomialConversionError, SeriesError, Symbol, TensorCanonicalizationError, UserData,
        UserDataKey,
    };

    pub use crate::coefficient::{Coefficient, CoefficientView, ConvertToRing};

    pub use crate::domains::{
        EuclideanDomain, Field, OrderedRing, RealEmbedding, Ring, RingOps, SampleableRing, Set,
        algebraic::{
            AlgebraicContext, AlgebraicEmbedding, AlgebraicExtension, AlgebraicNumber,
            AlgebraicQuotient, Root,
        },
        atom::AtomField,
        dual::HyperDual,
        factorized_rational_polynomial::FactorizedRationalPolynomial,
        finite_field::{FiniteField, FiniteFieldCore, FiniteFieldElement, Z2, Zp, Zp64},
        float::{
            Complex, Constructible, DoubleFloat, ErrorPropagatingFloat, F64, Float, FloatLike,
            Real, RealLike, SingleFloat,
        },
        integer::{Integer, IntegerRing, Z},
        rational::{Q, Rational},
        rational_polynomial::{
            LogarithmicIntegralTerm, RationalIntegral, RationalPolynomial, RationalPolynomialField,
        },
    };

    #[cfg(feature = "native_code_generation")]
    pub use crate::evaluate::{
        BatchEvaluator, CompileOptions, CompiledCode, CompiledComplexEvaluator, CompiledNumber,
        CompiledRealEvaluator, CompiledSimdComplexEvaluator, CompiledSimdRealEvaluator,
        EvaluatorLoader, ExportNumber, ExportSettings, ExportedCode, InlineASM,
        JITCompilationSettings,
    };
    pub use crate::evaluate::{
        Dualizer, EvaluationDomain, EvaluationFn, EvaluatorBuilder, ExportedInstructions,
        ExportedSubEvaluator, ExpressionEvaluator, ExternalFunction, FunctionMap, Instruction,
        OperationCount, OptimizationSettings, Vectorize,
    };

    pub use crate::id::{
        AtomTreeIterator, BorrowReplacement, Condition, ConditionResult, Match, MatchError,
        MatchSettings, MatchStack, Pattern, PatternAtomTreeIterator, PatternRestriction, Relation,
        ReplaceBuilder, ReplaceIterator, ReplaceSettings, ReplaceWith, Replacement,
        WildcardRestriction,
    };

    pub use crate::numerical_integration::{
        ContinuousGrid, DiscreteGrid, Grid, MonteCarloRng, Sample,
    };

    pub use crate::parser::{ParseMode, ParseSettings, Token};

    pub use crate::poly::{
        CoefficientToExpression, Exponent, GrevLexOrder, IntoVariableMap, LexOrder, MonomialOrder,
        PolyVariable, PolynomialResultant, PositiveExponent,
        factor::Factorize,
        gcd::PolynomialGCD,
        groebner::{
            GroebnerBasis, ParameterField, ParametricExtension, ParametricRoot, ParametricSolution,
            PolynomialSolution,
        },
        polynomial::{
            MultivariatePolynomial, PolynomialRing, PolynomialSamplingPolicy,
            PositiveRealRootCountError,
        },
        series::{Series, SeriesDepth},
        univariate::{
            UnivariatePolynomial, UnivariatePolynomialRing, UnivariatePolynomialSamplingPolicy,
        },
    };

    pub use crate::printer::{
        AtomPrinter, CanonicalOrderingSettings, PrintMode, PrintOptions, PrintState,
    };

    pub use crate::solve::{
        Complexes, Integers, Rationals, Reals, Solution, SolutionCondition, SolutionSet,
        SolveBuilder, SolveCoverage, SolveDomain, SolveError,
    };

    pub use crate::license::LicenseManager;

    pub use crate::state::State;

    pub use crate::streaming::{TermStreamer, TermStreamerConfig};

    pub use crate::tensors::{
        CanonicalTensor,
        matrix::{Matrix, Vector},
    };

    pub use crate::transcendental::TranscendentalFunctions;

    pub use crate::transformer::Transformer;
}

pub use evaluate::OperationCount;
pub use graphica as graph; // re-export graphica
#[doc(hidden)]
pub use inventory as _inventory;
pub use numerica::*; // re-export numerica

#[cfg(feature = "faster_alloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

/// Global settings for Symbolica.
pub struct GlobalSettings {
    /// Set whether a default tracing subscriber is initialized upon the first call to a logging macro.
    pub initialize_tracing: AtomicBool,
    /// Use an experimental implementation of the Hu-Monagan polynomial GCD algorithm.
    pub use_hu_monagan_poly_gcd: AtomicBool,
    /// Force the use of the Hu-Monagan polynomial GCD algorithm.
    pub force_hu_monagan_poly_gcd: AtomicBool,
    /// Enable the univariate-start path for multivariate integer factorization.
    pub use_univariate_factorization: AtomicBool,
    /// Enable the bivariate-start path for multivariate integer factorization.
    pub use_bivariate_factorization: AtomicBool,
}

/// Global settings for Symbolica.
pub static GLOBAL_SETTINGS: GlobalSettings = GlobalSettings {
    initialize_tracing: AtomicBool::new(true),
    use_hu_monagan_poly_gcd: AtomicBool::new(true),
    force_hu_monagan_poly_gcd: AtomicBool::new(false),
    use_univariate_factorization: AtomicBool::new(true),
    use_bivariate_factorization: AtomicBool::new(true),
};

/// Write an error messages using `tracing`. Initializes a default tracing subscriber on the first call if [GlobalSettings::initialize_tracing] is `true`.
#[macro_export]
macro_rules! error {
    ($($arg:tt)*) => {
        if $crate::GLOBAL_SETTINGS.initialize_tracing.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = tracing_subscriber::fmt()
                    .with_env_filter(
                        tracing_subscriber::EnvFilter::builder()
                            .with_default_directive(tracing_subscriber::filter::LevelFilter::INFO.into())
                            .from_env_lossy(),
                    )
                    .try_init();
            $crate::GLOBAL_SETTINGS.initialize_tracing.store(false, std::sync::atomic::Ordering::Relaxed);
        }

        tracing::error!($($arg)*);
   };
}

/// Write warning messages using `tracing`. Initializes a default tracing subscriber on the first call if [GlobalSettings::initialize_tracing] is `true`.
#[macro_export]
macro_rules! warn {
    ($($arg:tt)*) => {
        if $crate::GLOBAL_SETTINGS.initialize_tracing.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = tracing_subscriber::fmt()
                    .with_env_filter(
                        tracing_subscriber::EnvFilter::builder()
                            .with_default_directive(tracing_subscriber::filter::LevelFilter::INFO.into())
                            .from_env_lossy(),
                    )
                    .try_init();
            $crate::GLOBAL_SETTINGS.initialize_tracing.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        tracing::warn!($($arg)*);
    };
}

/// Write info messages using `tracing`. Initializes a default tracing subscriber on the first call if [GlobalSettings::initialize_tracing] is `true`.
#[macro_export]
macro_rules! info {
    ($($arg:tt)*) => {
        if $crate::GLOBAL_SETTINGS.initialize_tracing.load(std::sync::atomic::Ordering::Relaxed) {
            let _ = tracing_subscriber::fmt()
                    .with_env_filter(
                        tracing_subscriber::EnvFilter::builder()
                            .with_default_directive(tracing_subscriber::filter::LevelFilter::INFO.into())
                            .from_env_lossy(),
                    )
                    .try_init();
            $crate::GLOBAL_SETTINGS.initialize_tracing.store(false, std::sync::atomic::Ordering::Relaxed);
        }
        tracing::info!($($arg)*);
    };
}
