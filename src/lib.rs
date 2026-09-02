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

use std::{
    cell::{Cell, RefCell},
    collections::HashMap,
    env,
    sync::atomic::{AtomicBool, Ordering::Relaxed},
};
#[cfg(not(target_arch = "wasm32"))]
use std::{
    fs::File,
    io::{Read, Write},
    net::{TcpStream, ToSocketAddrs},
    process::abort,
    time::{Duration, SystemTime},
};

use once_cell::sync::OnceCell;
use tinyjson::JsonValue;

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
mod normalize;
pub mod parser;
pub mod poly;
pub mod printer;
pub mod solve;
pub mod state;
pub mod streaming;
pub mod tensors;
pub mod transcendental;
pub mod transformer;
pub mod unlock;
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
    pub use crate::{
        LicenseManager, OperationCount, create_hyperdual_from_components,
        create_hyperdual_single_derivative, function, get_symbol, hide_namespace, initialize,
        namespace, parse, parse_lit, symbol, symbol_group, tag, try_parse, try_parse_lit,
        try_symbol, try_symbol_group,
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
        OptimizationSettings, Vectorize,
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
        Complexes, Inequality, Integers, Rationals, Reals, Solution, SolutionCondition,
        SolutionValue, SolveBuilder, SolveDomain, SolveError, VariableSolution,
    };

    pub use crate::state::State;

    pub use crate::streaming::{TermStreamer, TermStreamerConfig};

    pub use crate::tensors::{
        CanonicalTensor,
        matrix::{Matrix, Vector},
    };

    pub use crate::transcendental::TranscendentalFunctions;

    pub use crate::transformer::Transformer;
}

pub use graphica as graph; // re-export graphica
#[doc(hidden)]
pub use inventory as _inventory;
pub use numerica::*; // re-export numerica

/// The number of operations needed by an evaluator or expression tree.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct OperationCount {
    /// The number of additions.
    pub additions: usize,
    /// The number of multiplications.
    pub multiplications: usize,
    /// The number of inversions.
    pub inversions: usize,
    /// The number of function calls.
    pub function_calls: usize,
}

impl OperationCount {
    /// Create a new operation count.
    pub fn new(
        additions: usize,
        multiplications: usize,
        inversions: usize,
        function_calls: usize,
    ) -> Self {
        Self {
            additions,
            multiplications,
            inversions,
            function_calls,
        }
    }

    /// Add the cost of raising a value to an integer power.
    ///
    /// Negative powers count as one inversion plus the multiplications required for the absolute
    /// power. For example, `x^-3` counts as one inversion and two multiplications.
    pub fn add_integer_power(&mut self, exponent: i64) {
        if exponent < 0 {
            self.inversions += 1;
        }

        self.multiplications += exponent.unsigned_abs().saturating_sub(1) as usize;
    }

    /// Add one function call.
    pub fn add_function_call(&mut self) {
        self.function_calls += 1;
    }
}

impl std::ops::Add for OperationCount {
    type Output = OperationCount;

    fn add(self, rhs: Self) -> Self::Output {
        OperationCount {
            additions: self.additions + rhs.additions,
            multiplications: self.multiplications + rhs.multiplications,
            inversions: self.inversions + rhs.inversions,
            function_calls: self.function_calls + rhs.function_calls,
        }
    }
}

impl std::ops::AddAssign for OperationCount {
    fn add_assign(&mut self, rhs: Self) {
        self.additions += rhs.additions;
        self.multiplications += rhs.multiplications;
        self.inversions += rhs.inversions;
        self.function_calls += rhs.function_calls;
    }
}

impl std::fmt::Display for OperationCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} +, {} ×, {} x⁻¹, {} f(·)",
            self.additions, self.multiplications, self.inversions, self.function_calls
        )
    }
}

#[cfg(not(target_arch = "wasm32"))]
use crate::printer::AnsiWrap;

#[cfg(feature = "faster_alloc")]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

static LICENSE_KEY: OnceCell<String> = OnceCell::new();
#[cfg(not(target_arch = "wasm32"))]
static LICENSE_MANAGER: OnceCell<LicenseManager> = OnceCell::new();
static LICENSED: AtomicBool = LicenseManager::init();

std::thread_local! {
    static INTERNAL_LICENSE_BYPASS_DEPTH: Cell<usize> = const { Cell::new(0) };
}

#[allow(dead_code)]
pub(crate) struct InternalLicenseBypassGuard;

impl InternalLicenseBypassGuard {
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        INTERNAL_LICENSE_BYPASS_DEPTH.with(|depth| {
            depth.set(
                depth
                    .get()
                    .checked_add(1)
                    .expect("license bypass scope nesting overflow"),
            );
        });
        Self
    }
}

impl Drop for InternalLicenseBypassGuard {
    fn drop(&mut self) {
        INTERNAL_LICENSE_BYPASS_DEPTH.with(|depth| {
            let current = depth.get();
            debug_assert!(current > 0, "unbalanced license bypass scope");
            depth.set(current.saturating_sub(1));
        });
    }
}

#[allow(dead_code)]
pub(crate) fn bypass_license_check_internal() -> InternalLicenseBypassGuard {
    InternalLicenseBypassGuard::new()
}

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

/// Manage the license of the Symbolica instance.
#[allow(dead_code)]
pub struct LicenseManager {
    has_license: bool,
}

#[cfg(not(target_arch = "wasm32"))]
struct RestrictedThreadPermit {
    pid: u32,
    _lock: File,
}

#[cfg(not(target_arch = "wasm32"))]
std::thread_local! {
    static RESTRICTED_THREAD_PERMIT: RefCell<Option<RestrictedThreadPermit>> = const { RefCell::new(None) };
}

/// Runtime capabilities that depend on the current target and enabled features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionCapabilities {
    /// Whether this build requires a Symbolica license before unrestricted use.
    pub license_required: bool,
    /// Whether this instance is currently considered licensed.
    pub is_licensed: bool,
    /// Maximum number of worker threads Symbolica should use for licensed-gated parallel paths.
    pub max_threads: usize,
    /// Whether native code generation and shared-library evaluator loading are available.
    pub native_code_generation: bool,
    /// Whether the built-in license server networking path is available.
    pub license_networking: bool,
}

#[cfg(not(target_arch = "wasm32"))]
const RESTRICTED_THREAD_WARNING: &str = "┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Cannot start another restricted Symbolica thread while this user's thread allowance is in use.   │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘"
;

#[cfg(not(target_arch = "wasm32"))]
const RESOLVE_ERROR: &str = "
┌───────────────────────────────────────────────────────────┐
│ Could not resolve the IP of the Symbolica license server. │
│                                                           │
│ Please check your DNS configuration.                      │
└───────────────────────────────────────────────────────────┘";

#[cfg(not(target_arch = "wasm32"))]
const CONNECTION_ERROR: &str = "
┌────────────────────────────────────────────────┐
│ Could not connect to Symbolica license server. │
│                                                │
│ Some networks block traffic to uncommon ports. │
│ Consider switching networks or using a VPN.    │
└────────────────────────────────────────────────┘";

#[cfg(not(target_arch = "wasm32"))]
const NETWORK_ERROR: &str = "
┌───────────────────────────────────────────────────┐
│ Connection to Symbolica license server timed out. │
│                                                   │
│ Please check your network configuration.          │
└───────────────────────────────────────────────────┘";

#[cfg(not(target_arch = "wasm32"))]
const ACTIVATION_ERROR: &str = "
┌──────────────────────────────────────────┐
│ Could not activate the Symbolica license │
└──────────────────────────────────────────┘";

#[cfg(not(target_arch = "wasm32"))]
const MISSING_LICENSE_ERROR: &str = "
┌───────────────────────────────┐
│ Symbolica license key missing │
└───────────────────────────────┘";

#[cfg(not(target_arch = "wasm32"))]
const EXPIRED_OFFLINE_LICENSE_WARNING: &str = "Warning: the offline Symbolica license has expired, but its underlying license is still valid. Request a new offline key with get_license_key.";

impl Default for LicenseManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(not(target_arch = "wasm32"))]
const OEM_LICENSE_KEY: Option<&str> = option_env!("SYMBOLICA_OEM_LICENSE");

/// Activate an OEM license key. Regular users should call [LicenseManager::set_license_key] instead.
#[macro_export]
macro_rules! activate_oem_license {
    ($key: literal) => {{
        const KEY2: [u32; 6] = {
            let mut h: u32 = 5381;
            let b = env!("CARGO_CRATE_NAME").as_bytes();
            let mut i = 0;
            while i < b.len() {
                h = h.wrapping_mul(33).wrapping_add(b[i] as u32);
                i += 1;
            }
            [124124564, 26352342, 63345, 3812471234, 23523, h]
        };

        symbolica::LicenseManager::set_oem_license_key($key, &KEY2).unwrap_or_else(|e| {
            panic!("{}", e);
        });
    }};
}

/// Verify and register a signed library unlock token for the calling Rust crate.
///
/// Store the returned [`unlock::LibraryUnlock`] and unlock each library operation with a guard.
///
/// A static `LibraryUnlock` must not be exported beyond the crate.
///
/// # Examples
///
/// ```
/// pub(crate) static UNLOCK: LazyLock<LibraryUnlock> = LazyLock::new(|| {
///     register_library_unlock!("YOUR_KEY").unwrap()
/// });
///
/// fn main() {
///     let _unlock = UNLOCK.unlock();
/// }
/// ```
#[macro_export]
macro_rules! register_library_unlock {
    ($token:expr) => {{ $crate::unlock::LibraryUnlock::for_crate($token, env!("CARGO_CRATE_NAME")) }};
}

impl LicenseManager {
    #[inline]
    fn is_library_unlocked() -> bool {
        if crate::unlock::current_thread_is_unlocked() {
            return true;
        }

        #[cfg(any(feature = "python_api", feature = "python_export"))]
        {
            if crate::api::python::has_library_unlock_frame() {
                return true;
            }
        }

        false
    }

    #[inline]
    fn is_check_bypassed() -> bool {
        INTERNAL_LICENSE_BYPASS_DEPTH.with(|depth| depth.get() != 0) || Self::is_library_unlocked()
    }

    /// Create a new license manager.
    #[cfg(target_arch = "wasm32")]
    pub(crate) fn new() -> LicenseManager {
        LICENSED.store(true, Relaxed);
        LicenseManager { has_license: true }
    }

    /// Create a new license manager.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn new() -> LicenseManager {
        match Self::check_license_key() {
            Ok(()) => {
                return LicenseManager { has_license: true };
            }
            Err(e) => {
                if !e.contains("missing") {
                    eprintln!("{e}");
                }
            }
        }

        if env::var("SYMBOLICA_HIDE_BANNER").is_err() {
            println!(
                "┌────────────────────────────────────────────────────────┐
│ You are running a restricted Symbolica instance.       │
│                                                        │
│ This mode is only permitted for non-commercial use and │
│ is limited to one Symbolica thread per user.           │
│                                                        │
│ {} can easily acquire a {} license key        │
│ that unlocks all cores and removes this banner:        │
│                                                        │
│   from symbolica import *                              │
│   request_hobbyist_license('YOUR_NAME', 'YOUR_EMAIL')  │
│                                                        │
│ All other users can obtain a free 30-day trial key:    │
│                                                        │
│   from symbolica import *                              │
│   request_trial_license('NAME', 'EMAIL', 'EMPLOYER')   │
│                                                        │
│ See https://symbolica.io/docs/get_started.html#license │
└────────────────────────────────────────────────────────┘",
                AnsiWrap::new("Hobbyists").bold(),
                AnsiWrap::new("free").bold(),
            );
        }

        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build_global();

        LicenseManager { has_license: false }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn acquire_restricted_thread_permit() {
        let pid = std::process::id();
        RESTRICTED_THREAD_PERMIT.with(|permit| {
            let mut permit = permit.borrow_mut();
            if permit.as_ref().is_some_and(|permit| permit.pid == pid) {
                return;
            }

            // A forked child must not share its parent's inherited lock description.
            *permit = None;
            match crate::unlock::try_acquire_lock("symbolica-restricted-thread-0.lock") {
                Ok(Some(lock)) => {
                    *permit = Some(RestrictedThreadPermit { pid, _lock: lock });
                }
                Ok(None) => {
                    println!("{RESTRICTED_THREAD_WARNING}");
                    abort();
                }
                Err(error) => {
                    eprintln!("Could not acquire Symbolica thread permit: {error}");
                    println!("{RESTRICTED_THREAD_WARNING}");
                    abort();
                }
            }
        });
    }

    const fn init() -> AtomicBool {
        AtomicBool::new(false)
    }

    #[cfg(target_arch = "wasm32")]
    fn check_license_key() -> Result<(), String> {
        LICENSED.store(true, Relaxed);
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn check_license_key() -> Result<(), String> {
        let key = LICENSE_KEY
            .get()
            .cloned()
            .or(env::var("SYMBOLICA_LICENSE").ok());

        let Some(mut key) = key else {
            std::thread::spawn(|| {
                let mut m: HashMap<String, JsonValue> = HashMap::default();
                m.insert(
                    "version".to_owned(),
                    env!("CARGO_PKG_VERSION").to_owned().into(),
                );
                let mut v = JsonValue::from(m).stringify().unwrap();
                v.push('\n');

                if let Ok(mut stream) = Self::connect() {
                    let _ = stream.write_all(v.as_bytes());
                };
            });

            return Err(MISSING_LICENSE_ERROR.to_owned());
        };

        if key.contains('#') {
            let mut a = key.split('#');
            let f1 = a.next().ok_or_else(|| ACTIVATION_ERROR.to_owned())?;
            let f2 = a.next().ok_or_else(|| ACTIVATION_ERROR.to_owned())?;
            let f3 = a.next().ok_or_else(|| ACTIVATION_ERROR.to_owned())?;

            let mut h: u32 = 5381;
            for b in f2.as_bytes() {
                h = h.wrapping_mul(33).wrapping_add(*b as u32);
            }
            for b in f3.as_bytes() {
                h = h.wrapping_mul(33).wrapping_add(*b as u32);
            }

            let h = format!("{h:x}");
            if f1 != h {
                Err(ACTIVATION_ERROR.to_owned())?;
            }

            let t = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let t2 = u64::from_str_radix(f2, 16)
                .map_err(|_| ACTIVATION_ERROR.to_owned())
                .unwrap();

            key = f3.to_owned();
            if t > t2 {
                Self::check_expired_offline_license(key, Self::check_registration)?;
            } else {
                std::thread::spawn(|| {
                    if let Err(e) = Self::check_registration(key)
                        && e.contains("expired")
                    {
                        println!("{e}");
                        abort();
                    }
                });
            }
        } else {
            Self::check_registration(key)?;
        }

        LICENSED.store(true, Relaxed);
        Ok(())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn connect() -> Result<TcpStream, String> {
        let mut ip = ("symbolica.io", 12012)
            .to_socket_addrs()
            .map_err(|e| format!("{RESOLVE_ERROR}\nError: {e}"))?;
        let Some(n) = ip.next() else {
            return Err(RESOLVE_ERROR.to_owned());
        };

        let stream = match TcpStream::connect_timeout(&n, Duration::from_secs(5)) {
            Ok(stream) => stream,
            Err(_) => {
                return Err(CONNECTION_ERROR.to_owned());
            }
        };

        stream
            .set_read_timeout(Some(Duration::from_secs(5)))
            .map_err(|e| e.to_string())?;
        stream
            .set_write_timeout(Some(Duration::from_secs(5)))
            .map_err(|e| e.to_string())?;

        Ok(stream)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn check_registration(key: String) -> Result<(), String> {
        let mut stream = Self::connect()?;

        let mut m: HashMap<String, JsonValue> = HashMap::default();
        m.insert(
            "version".to_owned(),
            env!("CARGO_PKG_VERSION").to_owned().into(),
        );
        m.insert("license".to_owned(), key.into());
        let mut v = JsonValue::from(m).stringify().unwrap();
        v.push('\n');

        stream
            .write_all(v.as_bytes())
            .map_err(|e| format!("{NETWORK_ERROR}\nError: {e}"))?;

        let mut buf = Vec::new();
        stream
            .read_to_end(&mut buf)
            .map_err(|e| format!("{NETWORK_ERROR}\nError: {e}"))?;
        let read_str =
            std::str::from_utf8(&buf).map_err(|e| format!("{NETWORK_ERROR}\nError: {e}"))?;

        if read_str == "{\"status\":\"ok\"}\n" {
            Ok(())
        } else if read_str.is_empty() {
            Err("┌──────────────────────────────────────────┐
│ Could not activate the Symbolica license │
└──────────────────────────────────────────┘"
                .to_owned())
        } else {
            let message: JsonValue = read_str[..read_str.len() - 1]
                .parse()
                .map_err(|e| format!("{NETWORK_ERROR}\nError: {e}"))?;
            let message_parsed: &HashMap<_, _> = message
                .get()
                .ok_or_else(|| format!("{NETWORK_ERROR}\nError: Empty response"))?;
            let status: &String = message_parsed
                .get("status")
                .unwrap()
                .get()
                .ok_or_else(|| format!("{NETWORK_ERROR}\nError: missing status"))?;
            Err(format!(
                "┌──────────────────────────────────────────┐
│ Could not activate the Symbolica license │
└──────────────────────────────────────────┘
Error: {status}",
            ))
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn check_expired_offline_license(
        key: String,
        check_registration: impl FnOnce(String) -> Result<(), String>,
    ) -> Result<(), String> {
        check_registration(key)?;
        eprintln!("{EXPIRED_OFFLINE_LICENSE_WARNING}");
        Ok(())
    }

    pub(crate) fn check_library_unlock_registration(key: String) {
        std::thread::spawn(|| {
            if let Err(error) = Self::check_registration(key)
                && error.contains("Unknown license")
            {
                println!("{error}");
                abort();
            }
        });
    }

    #[inline(always)]
    #[cfg(target_arch = "wasm32")]
    fn check() {
        LICENSED.store(true, Relaxed);
    }

    #[inline(always)]
    #[cfg(not(target_arch = "wasm32"))]
    fn check() {
        if LICENSED.load(Relaxed) || Self::is_check_bypassed() {
            return;
        }

        Self::check_impl();
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn check_impl() {
        let manager = LICENSE_MANAGER.get_or_init(LicenseManager::new);

        if manager.has_license {
            return;
        }

        Self::acquire_restricted_thread_permit();
    }

    /// Set the license key. Can only be called before calling any other Symbolica functions.
    pub fn set_license_key(key: &str) -> Result<(), String> {
        if LICENSE_KEY.get_or_init(|| key.to_owned()) != key {
            Err("Different license key cannot be set in same session")?;
        }

        Self::check_license_key()
    }

    /// Activate an OEM license key. Should not be called directly, use [activate_oem_license] instead.
    #[cfg(target_arch = "wasm32")]
    pub fn set_oem_license_key(
        _key1: &'static str,
        _key2: &'static [u32; 6],
    ) -> Result<(), &'static str> {
        LICENSED.store(true, Relaxed);
        Ok(())
    }

    /// Activate an OEM license key. Should not be called directly, use [activate_oem_license] instead.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn set_oem_license_key(
        key1: &'static str,
        key2: &'static [u32; 6],
    ) -> Result<(), &'static str> {
        let Some(oom_key) = OEM_LICENSE_KEY else {
            return Err("OEM license key not set");
        };

        if !oom_key.starts_with("SYMBOLICA_OEM_") {
            return Err("Invalid OEM license key");
        }

        if !key1.starts_with("SYMBOLICA_OEM_KEY_") {
            return Err("Invalid OEM license key part");
        }

        let mut h: u32 = 5381;
        for b in oom_key.as_bytes() {
            h = h.wrapping_mul(33).wrapping_add(*b as u32);
        }
        for b in key2 {
            h = h.wrapping_mul(33).wrapping_add(*b);
        }

        if key1 == format!("SYMBOLICA_OEM_KEY_{h:x}") {
            LICENSED.store(true, Relaxed);
            Self::check_library_unlock_registration(oom_key.to_owned());

            Ok(())
        } else {
            Err("Invalid OEM license key: key does not match")
        }
    }

    /// Returns `true` iff this instance has a valid license key set.
    #[cfg(target_arch = "wasm32")]
    pub fn is_licensed() -> bool {
        LICENSED.store(true, Relaxed);
        true
    }

    /// Returns `true` iff this instance has a valid license key or active library unlock.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn is_licensed() -> bool {
        LICENSED.load(Relaxed) || Self::is_library_unlocked() || Self::check_license_key().is_ok()
    }

    /// Clamp a requested worker-thread count to what the current target and license allow.
    pub fn max_threads(requested: usize) -> usize {
        #[cfg(target_arch = "wasm32")]
        {
            return requested.min(1);
        }

        #[cfg(not(target_arch = "wasm32"))]
        {
            if Self::is_library_unlocked() {
                return requested;
            }

            if Self::is_licensed() {
                requested
            } else {
                requested.min(1)
            }
        }
    }

    /// Return target and licensing capabilities without requiring callers to know cfg details.
    pub fn execution_capabilities() -> ExecutionCapabilities {
        ExecutionCapabilities {
            license_required: !cfg!(target_arch = "wasm32"),
            is_licensed: Self::is_licensed(),
            max_threads: Self::max_threads(usize::MAX),
            native_code_generation: cfg!(feature = "native_code_generation"),
            license_networking: !cfg!(target_arch = "wasm32"),
        }
    }

    /// Get the current Symbolica version.
    pub fn get_version() -> &'static str {
        env!("SYMBOLICA_VERSION")
    }

    #[cfg(target_arch = "wasm32")]
    fn request_license_email(_data: HashMap<String, JsonValue>) -> Result<(), String> {
        Err("No Symbolica license key is required for WASM builds.".to_owned())
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn request_license_email(data: HashMap<String, JsonValue>) -> Result<(), String> {
        let mut stream = Self::connect()?;
        let mut v = JsonValue::from(data).stringify().unwrap();
        v.push('\n');

        stream
            .write_all(v.as_bytes())
            .map_err(|e| format!("{NETWORK_ERROR}\nError: {e}"))?;

        let mut buf = Vec::new();
        stream
            .read_to_end(&mut buf)
            .map_err(|e| format!("{NETWORK_ERROR}\nError: {e}"))?;
        let read_str = std::str::from_utf8(&buf).map_err(|_| "Bad server response".to_string())?;

        if read_str == "{\"status\":\"email sent\"}\n" {
            Ok(())
        } else if read_str.is_empty() {
            Err("Empty response".to_owned())
        } else {
            let message: JsonValue = read_str[..read_str.len() - 1]
                .parse()
                .map_err(|_| "Bad server response".to_string())?;
            let message_parsed: &HashMap<_, _> = message
                .get()
                .ok_or_else(|| "Bad server response".to_string())?;
            let status: &String = message_parsed
                .get("status")
                .unwrap()
                .get()
                .ok_or_else(|| "Bad server response".to_string())?;
            Err(status.clone())
        }
    }

    /// Request a key for **non-professional** use for the user `name`, that will be sent to the e-mail address
    /// `email`.
    pub fn request_hobbyist_license(name: &str, email: &str) -> Result<(), String> {
        let mut m: HashMap<String, JsonValue> = HashMap::default();
        m.insert("name".to_owned(), name.to_owned().into());
        m.insert("email".to_owned(), email.to_owned().into());
        m.insert("type".to_owned(), "hobbyist".to_owned().into());
        Self::request_license_email(m)
    }

    /// Request a key for a trial license for the user `name` working at `company`, that will be sent to the e-mail address
    /// `email`.
    pub fn request_trial_license(name: &str, email: &str, company: &str) -> Result<(), String> {
        let mut m: HashMap<String, JsonValue> = HashMap::default();
        m.insert("name".to_owned(), name.to_owned().into());
        m.insert("email".to_owned(), email.to_owned().into());
        m.insert("company".to_owned(), company.to_owned().into());
        m.insert("type".to_owned(), "trial".to_owned().into());
        Self::request_license_email(m)
    }

    /// Request a sublicense key for the user `name` working at `company` that has the site-wide license `super_license`.
    /// The key will be sent to the e-mail address `email`.
    pub fn request_sublicense(
        name: &str,
        email: &str,
        company: &str,
        super_license: &str,
    ) -> Result<(), String> {
        let mut m: HashMap<String, JsonValue> = HashMap::default();
        m.insert("name".to_owned(), name.to_owned().into());
        m.insert("email".to_owned(), email.to_owned().into());
        m.insert("company".to_owned(), company.to_owned().into());
        m.insert("type".to_owned(), "sublicense".to_owned().into());
        m.insert("super_license".to_owned(), super_license.to_owned().into());
        Self::request_license_email(m)
    }

    /// Get the license key for the account registered with the provided email address.
    pub fn get_license_key(email: &str) -> Result<(), String> {
        let mut m: HashMap<String, JsonValue> = HashMap::default();
        m.insert("email".to_owned(), email.to_owned().into());
        Self::request_license_email(m)
    }
}

#[cfg(test)]
mod license_bypass_tests {
    use super::*;

    #[test]
    fn internal_license_bypass_guard_is_nested_and_thread_local() {
        assert!(!LicenseManager::is_check_bypassed());

        let outer = bypass_license_check_internal();
        assert!(LicenseManager::is_check_bypassed());
        assert!(
            crate::parser::Token::parse("x + 1", crate::parser::ParseSettings::default()).is_ok()
        );

        {
            let _inner = InternalLicenseBypassGuard::new();
            assert!(LicenseManager::is_check_bypassed());
        }

        assert!(LicenseManager::is_check_bypassed());
        assert!(
            std::thread::spawn(|| !LicenseManager::is_check_bypassed())
                .join()
                .unwrap()
        );

        drop(outer);
        assert!(!LicenseManager::is_check_bypassed());
    }
}

#[cfg(test)]
mod offline_license_tests {
    use super::*;

    #[test]
    fn expired_offline_license_checks_underlying_key() {
        let mut checked = false;
        LicenseManager::check_expired_offline_license("underlying-key".to_owned(), |key| {
            assert_eq!(key, "underlying-key");
            checked = true;
            Ok(())
        })
        .unwrap();
        assert!(checked);
    }

    #[test]
    fn expired_offline_license_propagates_registration_failure() {
        let error = LicenseManager::check_expired_offline_license("expired-key".to_owned(), |_| {
            Err("underlying license expired".to_owned())
        })
        .unwrap_err();
        assert_eq!(error, "underlying license expired");
    }
}
