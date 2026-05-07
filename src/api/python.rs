//! Python API bindings.
//!
//! All Symbolica community extensions must implement the [SymbolicaCommunityModule] trait.

use std::{
    fs::File,
    hash::{Hash, Hasher},
    io::{BufReader, BufWriter},
    ops::{Deref, Neg},
    sync::{Arc, Mutex},
};

use ahash::{HashMap, HashSet};
use brotli::CompressorWriter;
use numpy::{
    AllowTypeChange, Complex64, IntoPyArray, PyArrayDyn, PyArrayLike1, PyArrayLikeDyn,
    ndarray::{ArrayD, Axis, CowArray, IxDyn},
};
use pyo3::{
    Borrowed, Bound, FromPyObject, IntoPyObject, IntoPyObjectExt, Py, PyAny, PyErr, PyRef,
    PyResult, PyTypeInfo, Python,
    exceptions::{self, PyIndexError},
    pybacked::PyBackedStr,
    pyclass::CompareOp,
    pyfunction, pymethods,
    types::{
        PyAnyMethods, PyBytes, PyBytesMethods, PyComplex, PyDict, PyDictMethods, PyInt, PyIterator,
        PyModule, PyNone, PyTuple, PyTupleMethods, PyType, PyTypeMethods,
    },
    wrap_pyfunction,
};
use pyo3::{pyclass, types::PyModuleMethods};

#[cfg(feature = "python_stubgen")]
use pyo3::types::PyList;

#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::{
    PyStubType, TypeInfo,
    derive::{gen_stub_pyclass, gen_stub_pyclass_enum, gen_stub_pyfunction, gen_stub_pymethods},
    impl_stub_type,
    inventory::submit,
    type_info::{
        MethodInfo, MethodType, ParameterDefault, ParameterInfo, ParameterKind, PyFunctionInfo,
        PyMethodsInfo,
    },
};
#[cfg(not(feature = "python_stubgen"))]
use pyo3_stub_gen_derive::remove_gen_stub;

use rand::{Rng, RngCore};
use rug::Complete;
use self_cell::self_cell;
use smallvec::SmallVec;
use smartstring::{LazyCompact, SmartString};

#[cfg(not(feature = "python_export"))]
use pyo3::pymodule;

use crate::{
    LicenseManager,
    atom::{
        Atom, AtomCore, AtomType, AtomView, DefaultNamespace, EvaluationInfo, Indeterminate,
        ListIterator, Symbol, SymbolAttribute, UserData, UserDataKey,
    },
    coefficient::{Coefficient, CoefficientView, ConvertToRing},
    domains::{
        Ring, RingOps, SelfRing,
        algebraic_number::AlgebraicExtension,
        atom::AtomField,
        dual::HyperDual,
        finite_field::{FiniteFieldCore, PrimeIteratorU64, ToFiniteField, Z2, Zp64},
        float::{Complex, DoubleFloat, F64, Float, PythonMultiPrecisionFloat, RealLike},
        integer::{FromFiniteField, Integer, IntegerRelationError, IntegerRing, Z},
        rational::{Q, Rational, RationalField},
        rational_polynomial::{
            FromNumeratorAndDenominator, RationalPolynomial, RationalPolynomialField,
        },
    },
    error,
    evaluate::{
        BatchEvaluator, CompileOptions, CompiledComplexEvaluator, CompiledCudaComplexEvaluator,
        CompiledCudaRealEvaluator, CompiledNumber, CompiledRealEvaluator,
        CompiledSimdComplexEvaluator, CompiledSimdRealEvaluator, ComplexEvaluatorSettings,
        CudaComplexf64, CudaLoadSettings, CudaRealf64, Dualizer, EvaluationFn, EvaluatorLoader,
        ExportSettings, ExpressionEvaluator, FunctionMap, InlineASM, Instruction,
        JITCompiledEvaluator, OptimizationSettings, Slot,
    },
    graph::{GenerationSettings, Graph, HalfEdge},
    id::{
        Condition, ConditionResult, Evaluate, Match, MatchSettings, MatchStack, Pattern,
        PatternAtomTreeIterator, PatternRestriction, Relation, ReplaceIterator, ReplaceSettings,
        ReplaceWith, Replacement, WildcardRestriction,
    },
    numerical_integration::{ContinuousGrid, DiscreteGrid, Grid, MonteCarloRng, Probe, Sample},
    parser::{ParseMode, ParseSettings, Token},
    poly::{
        GrevLexOrder, INLINED_EXPONENTS, LexOrder, PolyVariable, factor::Factorize,
        gcd::PolynomialGCD, groebner::GroebnerBasis, polynomial::MultivariatePolynomial,
        series::Series,
    },
    printer::{AtomPrinter, PrintMode, PrintOptions, PrintState},
    solve::SolveError,
    state::{RecycledAtom, State, Workspace},
    streaming::{TermStreamer, TermStreamerConfig},
    tensors::matrix::Matrix,
    transformer::{StatsOptions, Transformer, TransformerError, TransformerState},
    try_parse, warn,
};

#[cfg(feature = "python_stubgen")]
static NONE_ARG: fn() -> String = || "None".into();

const DEFAULT_PRINT_OPTIONS: PrintOptions = PrintOptions {
    hide_namespace: Some("python"),
    ..PrintOptions::new()
};

const PLAIN_PRINT_OPTIONS: PrintOptions = PrintOptions {
    hide_namespace: Some("python"),
    ..PrintOptions::file()
};

const LATEX_PRINT_OPTIONS: PrintOptions = PrintOptions {
    hide_namespace: Some("python"),
    ..PrintOptions::latex()
};

mod atom;
mod evaluator;
mod expression;
mod graph;
mod integer;
mod integration;
mod matrix;
mod polynomial;
mod series;

pub use atom::*;
pub use evaluator::*;
pub use expression::*;
pub use graph::*;
pub use integer::*;
pub use integration::*;
pub use matrix::*;
pub use polynomial::*;
pub use series::*;

/// Trait for registering Python submodules for Symbolica, which enables
/// multiple crates to use the same Symbolica kernel.
///
/// You must create a global variable called `CommunityModule`:
/// ```rust
/// pub struct CommunityModule;
///
/// impl SymbolicaCommunityModule for CommunityModule {
///     fn get_name() -> String {
///         "NAME".to_string()
///     }
///
///     fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
///         // add your functions and classes
///         Ok(())
///     }
/// }
/// ```
///
/// And you must set the modules of your functions and classes to
/// `symbolica.community.NAME`, .i.e,
/// ```
/// #[pyclass(module = "symbolica.community.NAME")]
/// struct MyPythonStruct {}
/// ```
#[cfg(feature = "python_export")]
pub trait SymbolicaCommunityModule {
    /// The name of the submodule. Must be used in all defined Python structures, as such:
    /// ```
    /// #[pyclass(module = "symbolica.community.NAME")]
    /// struct MyPythonStruct {}
    /// ```
    fn get_name() -> String;

    /// Register all classes, functions and methods in the submodule `m`.
    /// This function must not register any Symbolica symbols. All initialization
    /// should be performed in the [SymbolicaCommunityModule::initialize] function.
    fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()>;

    /// Initialize the community module. Called when the submodule is imported.
    fn initialize(_py: Python) -> PyResult<()> {
        Ok(())
    }
}

/// Specifies the print mode.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass_enum)]
#[pyclass(
    from_py_object,
    name = "ParseMode",
    eq,
    eq_int,
    module = "symbolica.core"
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PythonParseMode {
    /// parse using Symbolica notation.
    Symbolica,
    /// Parse using Mathematica notation.
    Mathematica,
}

impl From<PythonParseMode> for ParseMode {
    fn from(mode: PythonParseMode) -> Self {
        match mode {
            PythonParseMode::Symbolica => ParseMode::Symbolica,
            PythonParseMode::Mathematica => ParseMode::Mathematica,
        }
    }
}

/// Specifies the print mode.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass_enum)]
#[pyclass(
    from_py_object,
    name = "PrintMode",
    eq,
    eq_int,
    module = "symbolica.core"
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum PythonPrintMode {
    /// Print using Symbolica notation.
    Symbolica,
    /// Print using LaTeX notation.
    Latex,
    /// Print using Mathematica notation.
    Mathematica,
    /// Print using Sympy notation.
    Sympy,
    /// Print using Typst notation.
    Typst,
}

impl From<PrintMode> for PythonPrintMode {
    fn from(mode: PrintMode) -> Self {
        match mode {
            PrintMode::Symbolica => PythonPrintMode::Symbolica,
            PrintMode::Latex => PythonPrintMode::Latex,
            PrintMode::Mathematica => PythonPrintMode::Mathematica,
            PrintMode::Sympy => PythonPrintMode::Sympy,
            PrintMode::Typst => PythonPrintMode::Typst,
            _ => {
                error!("Unsupported PrintMode: {:?}", mode);
                PythonPrintMode::Symbolica
            }
        }
    }
}

impl From<PythonPrintMode> for PrintMode {
    fn from(mode: PythonPrintMode) -> Self {
        match mode {
            PythonPrintMode::Symbolica => PrintMode::Symbolica,
            PythonPrintMode::Latex => PrintMode::Latex,
            PythonPrintMode::Mathematica => PrintMode::Mathematica,
            PythonPrintMode::Sympy => PrintMode::Sympy,
            PythonPrintMode::Typst => PrintMode::Typst,
        }
    }
}

/// Create a Symbolica Python module.
pub fn create_symbolica_module<'a, 'b>(
    m: &'b Bound<'a, PyModule>,
) -> PyResult<&'b Bound<'a, PyModule>> {
    m.add_class::<PythonExpression>()?;
    m.add_class::<PythonHeldExpression>()?;
    m.add_class::<PythonTransformer>()?;
    m.add_class::<PythonPolynomial>()?;
    m.add_class::<PythonFiniteFieldPolynomial>()?;
    m.add_class::<PythonNumberFieldPolynomial>()?;
    m.add_class::<PythonRationalPolynomial>()?;
    m.add_class::<PythonFiniteFieldRationalPolynomial>()?;
    m.add_class::<PythonMatrix>()?;
    m.add_class::<PythonNumericalIntegrator>()?;
    m.add_class::<PythonSample>()?;
    m.add_class::<PythonProbe>()?;
    m.add_class::<PythonAtomType>()?;
    m.add_class::<PythonAtomTree>()?;
    m.add_class::<PythonSymbolAttribute>()?;
    m.add_class::<PythonParseMode>()?;
    m.add_class::<PythonPrintMode>()?;
    m.add_class::<PythonCondition>()?;
    m.add_class::<PythonReplacement>()?;
    m.add_class::<PythonExpressionEvaluator>()?;
    m.add_class::<PythonCompiledRealExpressionEvaluator>()?;
    m.add_class::<PythonCompiledComplexExpressionEvaluator>()?;
    m.add_class::<PythonCompiledSimdRealExpressionEvaluator>()?;
    m.add_class::<PythonCompiledSimdComplexExpressionEvaluator>()?;
    m.add_class::<PythonCompiledCudaRealExpressionEvaluator>()?;
    m.add_class::<PythonCompiledCudaComplexExpressionEvaluator>()?;
    m.add_class::<PythonRandomNumberGenerator>()?;
    m.add_class::<PythonPatternRestriction>()?;
    m.add_class::<PythonTermStreamer>()?;
    m.add_class::<PythonSeries>()?;
    m.add_class::<PythonHalfEdge>()?;
    m.add_class::<PythonGraph>()?;
    m.add_class::<PythonInteger>()?;

    m.add_function(wrap_pyfunction!(symbol_shorthand, m)?)?;
    m.add_function(wrap_pyfunction!(number_shorthand, m)?)?;
    m.add_function(wrap_pyfunction!(expression_shorthand, m)?)?;
    m.add_function(wrap_pyfunction!(transformer_shorthand, m)?)?;
    m.add_function(wrap_pyfunction!(poly_shorthand, m)?)?;

    m.add_function(wrap_pyfunction!(get_version, m)?)?;
    m.add_function(wrap_pyfunction!(is_licensed, m)?)?;
    m.add_function(wrap_pyfunction!(set_license_key, m)?)?;
    m.add_function(wrap_pyfunction!(request_hobbyist_license, m)?)?;
    m.add_function(wrap_pyfunction!(request_trial_license, m)?)?;
    m.add_function(wrap_pyfunction!(request_sublicense, m)?)?;
    m.add_function(wrap_pyfunction!(get_license_key, m)?)?;
    m.add_function(wrap_pyfunction!(use_custom_logger, m)?)?;
    m.add_function(wrap_pyfunction!(get_namespace, m)?)?;
    m.add_function(wrap_pyfunction!(set_namespace, m)?)?;

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(m)
}

fn print_options_to_dict<'py>(
    options: &PrintOptions,
    state: &PrintState,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("mode", PythonPrintMode::from(options.mode))?;
    dict.set_item("max_line_length", options.max_line_length)?;
    dict.set_item("indentation", options.indentation)?;
    dict.set_item("fill_indented_lines", options.fill_indented_lines)?;
    dict.set_item("terms_on_new_line", options.terms_on_new_line)?;
    dict.set_item("color_top_level_sum", options.color_top_level_sum)?;
    dict.set_item("color_builtin_symbols", options.color_builtin_symbols)?;
    dict.set_item("bracket_level_colors", options.bracket_level_colors)?;
    dict.set_item("print_ring", options.print_ring)?;
    dict.set_item(
        "symmetric_representation_for_finite_field",
        options.symmetric_representation_for_finite_field,
    )?;
    dict.set_item(
        "explicit_rational_polynomial",
        options.explicit_rational_polynomial,
    )?;
    dict.set_item(
        "number_thousands_separator",
        options.number_thousands_separator,
    )?;
    dict.set_item("multiplication_operator", options.multiplication_operator)?;
    dict.set_item(
        "double_star_for_exponentiation",
        options.double_star_for_exponentiation,
    )?;
    #[allow(deprecated)]
    dict.set_item(
        "square_brackets_for_function",
        options.square_brackets_for_function,
    )?;
    dict.set_item("function_brackets", options.function_brackets)?;
    dict.set_item("num_exp_as_superscript", options.num_exp_as_superscript)?;
    dict.set_item("precision", options.precision)?;
    dict.set_item("pretty_matrix", options.pretty_matrix)?;
    dict.set_item("hide_namespace", options.hide_namespace)?;
    dict.set_item("hide_all_namespaces", options.hide_all_namespaces)?;
    dict.set_item("color_namespace", options.color_namespace)?;
    dict.set_item("max_terms", options.max_terms)?;
    dict.set_item("custom_print_mode", options.custom_print_mode.map(|x| x.1))?;

    dict.set_item("level", state.level)?;
    dict.set_item("bracket_level", state.bracket_level)?;
    dict.set_item("indentation_level", state.indentation_level)?;

    Ok(dict)
}

/// Set the Symbolica namespace for the calling module.
/// All subsequently created symbols in the calling module will be defined within this namespace.
///
/// This function sets the `SYMBOLICA_NAMESPACE` variable in the global scope of the calling module.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.core")
)]
#[pyfunction]
pub fn set_namespace(py: Python, namespace: String) -> PyResult<()> {
    let ptr = unsafe { pyo3::ffi::PyEval_GetGlobals() };

    if ptr.is_null() {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "No active Python frame found to inject globals into.",
        ));
    }

    let globals = unsafe { Bound::from_borrowed_ptr(py, ptr) };

    globals.set_item("SYMBOLICA_NAMESPACE", namespace)?;

    Ok(())
}

static INTERNED_STRINGS: std::sync::LazyLock<Mutex<HashSet<&'static str>>> =
    std::sync::LazyLock::new(|| Mutex::new(HashSet::default()));

fn intern_string(string: &str) -> &'static str {
    let mut ns = INTERNED_STRINGS.lock().unwrap();
    if let Some(s) = ns.get::<str>(&string) {
        s
    } else {
        let b = Box::leak(string.to_string().into_boxed_str()) as &'static str;
        ns.insert(b);
        b
    }
}

/// Get the Symbolica namespace for the calling module.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.core")
)]
#[pyfunction]
pub fn get_namespace(py: Python) -> PyResult<&'static str> {
    let ptr = unsafe { pyo3::ffi::PyEval_GetGlobals() };

    if ptr.is_null() {
        return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "No active Python frame found",
        ));
    }

    let globals = unsafe { Bound::from_borrowed_ptr(py, ptr) };
    Ok(
        match globals.cast::<PyDict>()?.get_item("SYMBOLICA_NAMESPACE") {
            Ok(Some(val)) => intern_string(&val.extract::<PyBackedStr>()?),
            Err(_) => "python",
            Ok(None) => "python",
        },
    )
}

/// Symbolica is a blazing fast computer algebra system.
///
/// It can be used to perform mathematical operations,
/// such as symbolic differentiation, integration, simplification,
/// pattern matching and solving equations.
///
/// Examples
/// --------
///
/// >>> from symbolica import *
/// >>> e = E('x^2*log(2*x + y) + exp(3*x)')
/// >>> a = e.derivative(S('x'))
/// >>> print("d/dx {} = {}".format(e, a))
#[cfg(feature = "python_api")]
#[pymodule]
fn symbolica(m: &Bound<'_, PyModule>) -> PyResult<()> {
    pyo3_log::init();
    create_symbolica_module(m).map(|_| ())
}

/// Enable logging using Python's logging module instead of using the default logging.
/// This is useful when using Symbolica in a Jupyter notebook or other environments
/// where stdout is not easily accessible.
///
/// This function must be called before any Symbolica logging events are emitted.
#[pyfunction]
fn use_custom_logger() {
    crate::GLOBAL_SETTINGS
        .initialize_tracing
        .store(false, std::sync::atomic::Ordering::Relaxed);
}

/// Get the current Symbolica version.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.core")
)]
#[pyfunction]
fn get_version() -> String {
    LicenseManager::get_version().to_string()
}

/// Check if the current Symbolica instance has a valid license key set.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.core")
)]
#[pyfunction]
fn is_licensed() -> bool {
    LicenseManager::is_licensed()
}

/// Set the Symbolica license key for this computer. Can only be called before calling any other Symbolica functions
/// and before importing any community modules.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.core")
)]
#[pyfunction]
fn set_license_key(key: String) -> PyResult<()> {
    LicenseManager::set_license_key(&key).map_err(exceptions::PyException::new_err)
}

/// Request a key for **non-professional** use for the user `name`, that will be sent to the e-mail address
/// `email`.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.core")
)]
#[pyfunction]
fn request_hobbyist_license(name: String, email: String) -> PyResult<()> {
    LicenseManager::request_hobbyist_license(&name, &email)
        .map(|_| println!("A license key was sent to your e-mail address."))
        .map_err(exceptions::PyConnectionError::new_err)
}

/// Request a key for a trial license for the user `name` working at `company`, that will be sent to the e-mail address
/// `email`.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.core")
)]
#[pyfunction]
fn request_trial_license(name: String, email: String, company: String) -> PyResult<()> {
    LicenseManager::request_trial_license(&name, &email, &company)
        .map(|_| println!("A license key was sent to your e-mail address."))
        .map_err(exceptions::PyConnectionError::new_err)
}

/// Request a sublicense key for the user `name` working at `company` that has the site-wide license `super_license`.
/// The key will be sent to the e-mail address `email`.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.core")
)]
#[pyfunction]
fn request_sublicense(
    name: String,
    email: String,
    company: String,
    super_license: String,
) -> PyResult<()> {
    LicenseManager::request_sublicense(&name, &email, &company, &super_license)
        .map(|_| println!("A license key was sent to your e-mail address."))
        .map_err(exceptions::PyConnectionError::new_err)
}

/// Get the license key for the account registered with the provided email address.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.core")
)]
#[pyfunction]
fn get_license_key(email: String) -> PyResult<()> {
    LicenseManager::get_license_key(&email)
        .map(|_| println!("A license key was sent to your e-mail address."))
        .map_err(exceptions::PyConnectionError::new_err)
}

#[pyfunction(name = "S", signature = (*names,is_symmetric=None,is_antisymmetric=None,is_cyclesymmetric=None,is_linear=None,is_scalar=None,is_real=None,is_integer=None,is_positive=None,tags=None,aliases=None,normalization=None,print=None,derivative=None,series=None,eval=None,data=None))]
/// Shorthand notation for :func:`Expression.symbol`.
fn symbol_shorthand(
    names: &Bound<'_, PyTuple>,
    is_symmetric: Option<bool>,
    is_antisymmetric: Option<bool>,
    is_cyclesymmetric: Option<bool>,
    is_linear: Option<bool>,
    is_scalar: Option<bool>,
    is_real: Option<bool>,
    is_integer: Option<bool>,
    is_positive: Option<bool>,
    tags: Option<Vec<String>>,
    aliases: Option<Vec<String>>,
    normalization: Option<PythonTransformer>,
    print: Option<Py<PyAny>>,
    derivative: Option<Py<PyAny>>,
    series: Option<Py<PyAny>>,
    eval: Option<Py<PyAny>>,
    data: Option<PythonUserData>,
    py: Python,
) -> PyResult<Py<PyAny>> {
    PythonExpression::symbol(
        &PythonExpression::type_object(py),
        py,
        names,
        is_symmetric,
        is_antisymmetric,
        is_cyclesymmetric,
        is_linear,
        is_scalar,
        is_real,
        is_integer,
        is_positive,
        tags,
        aliases,
        normalization,
        print,
        derivative,
        series,
        eval,
        data,
    )
}

#[derive(Clone)]
struct PythonEvalSpec {
    tag_count: usize,
    float: Option<Py<PyAny>>,
    complex: Option<Py<PyAny>>,
    decimal: Option<Py<PyAny>>,
    decimal_complex: Option<Py<PyAny>>,
    constant: Option<Py<PyAny>>,
    cpp: Option<String>,
}

impl PythonEvalSpec {
    const ALLOWED_KEYS: &[&str] = &[
        "tag_count",
        "float",
        "complex",
        "decimal",
        "decimal_complex",
        "constant",
        "cpp",
    ];

    fn from_py(py: Python, eval: Py<PyAny>) -> PyResult<Self> {
        let eval_bound = eval.bind(py);

        let dict = eval_bound
            .cast::<PyDict>()
            .map_err(|_| exceptions::PyTypeError::new_err("eval must be a dictionary"))?;
        Self::validate_eval_dict_keys(dict)?;

        let tag_count = match dict.get_item("tag_count") {
            Ok(Some(t)) => t.extract::<usize>()?,
            Ok(None) => 0,
            Err(_) => 0,
        };

        let spec = Self {
            tag_count,
            float: Self::get_eval_callable(dict, "float")?,
            complex: Self::get_eval_callable(dict, "complex")?,
            decimal: Self::get_eval_callable(dict, "decimal")?,
            decimal_complex: Self::get_eval_callable(dict, "decimal_complex")?,
            constant: Self::get_eval_callable(dict, "constant")?,
            cpp: Self::get_eval_string(dict, "cpp")?,
        };

        if spec.constant.is_some()
            && (spec.float.is_some()
                || spec.complex.is_some()
                || spec.decimal.is_some()
                || spec.decimal_complex.is_some())
        {
            return Err(exceptions::PyValueError::new_err(
                "eval['constant'] cannot be combined with other eval callbacks",
            ));
        }

        Ok(spec)
    }

    fn validate_eval_dict_keys(dict: &Bound<'_, PyDict>) -> PyResult<()> {
        for key in dict.keys() {
            let key = key.extract::<String>().map_err(|_| {
                exceptions::PyTypeError::new_err("eval dictionary keys must be strings")
            })?;

            if !Self::ALLOWED_KEYS.contains(&key.as_str()) {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Unknown eval dictionary entry '{key}'. Allowed entries are: {}",
                    Self::ALLOWED_KEYS.join(", ")
                )));
            }
        }

        Ok(())
    }

    fn into_evaluation_info(self) -> EvaluationInfo {
        let tag_count = self.tag_count;
        let mut info = if let Some(f) = self.constant {
            EvaluationInfo::constant(move |tags, prec| {
                if tags.len() != tag_count {
                    return Err(format!(
                        "Python eval expected {tag_count} tags, got {}",
                        tags.len()
                    ));
                }

                Python::attach(|py| {
                    let f = Self::python_eval_callable(py, &f, tags, tag_count)?;
                    let decimal_prec = Self::decimal_digits_from_binary_prec(prec);
                    let args = Vec::<(PythonMultiPrecisionFloat, PythonMultiPrecisionFloat)>::new();
                    let value = f.call1(py, (args.into_py_any(py)?, decimal_prec))?;
                    Self::extract_python_constant(py, value)
                })
                .map_err(|e| e.to_string())
            })
            .with_tags(tag_count)
        } else {
            EvaluationInfo::new().with_tags(tag_count)
        };

        if let Some(f) = self.float {
            if tag_count == 0 {
                info = info.register(move |args: &[f64]| {
                    Python::attach(|py| {
                        f.call1(py, (args.to_vec().into_py_any(py)?,))?
                            .extract::<f64>(py)
                    })
                    .expect("Python eval callback for f64 failed")
                });
            } else {
                info = info.register_tagged(move |tags| {
                    let f =
                        Python::attach(|py| Self::python_eval_callable(py, &f, tags, tag_count))
                            .expect("Python tagged eval callback for f64 failed");
                    Box::new(move |args: &[f64]| {
                        Python::attach(|py| {
                            f.call1(py, (args.to_vec().into_py_any(py)?,))?
                                .extract::<f64>(py)
                        })
                        .expect("Python eval callback for f64 failed")
                    })
                });
            }
        }

        if let Some(f) = self.complex {
            if tag_count == 0 {
                info = info.register(move |args: &[Complex<f64>]| {
                    Python::attach(|py| {
                        let args = args
                            .iter()
                            .map(|x| PyComplex::from_doubles(py, x.re, x.im))
                            .collect::<Vec<_>>();
                        f.call1(py, (args.into_py_any(py)?,))?
                            .extract::<Complex<f64>>(py)
                    })
                    .expect("Python eval callback for complex f64 failed")
                });
            } else {
                info = info.register_tagged(move |tags| {
                    let f =
                        Python::attach(|py| Self::python_eval_callable(py, &f, tags, tag_count))
                            .expect("Python tagged eval callback for complex f64 failed");
                    Box::new(move |args: &[Complex<f64>]| {
                        Python::attach(|py| {
                            let args = args
                                .iter()
                                .map(|x| PyComplex::from_doubles(py, x.re, x.im))
                                .collect::<Vec<_>>();
                            f.call1(py, (args.into_py_any(py)?,))?
                                .extract::<Complex<f64>>(py)
                        })
                        .expect("Python eval callback for complex f64 failed")
                    })
                });
            }
        }

        if let Some(f) = self.decimal {
            if tag_count == 0 {
                info = info.register(move |args: &[Float]| {
                    Python::attach(|py| {
                        let args = args
                            .iter()
                            .cloned()
                            .map(PythonMultiPrecisionFloat)
                            .collect::<Vec<_>>();
                        f.call1(py, (args.into_py_any(py)?,))?
                            .extract::<PythonMultiPrecisionFloat>(py)
                            .map(|x| x.0)
                    })
                    .expect("Python eval callback for decimal failed")
                });
            } else {
                info = info.register_tagged(move |tags| {
                    let f =
                        Python::attach(|py| Self::python_eval_callable(py, &f, tags, tag_count))
                            .expect("Python tagged eval callback for decimal failed");
                    Box::new(move |args: &[Float]| {
                        Python::attach(|py| {
                            let args = args
                                .iter()
                                .cloned()
                                .map(PythonMultiPrecisionFloat)
                                .collect::<Vec<_>>();
                            f.call1(py, (args.into_py_any(py)?,))?
                                .extract::<PythonMultiPrecisionFloat>(py)
                                .map(|x| x.0)
                        })
                        .expect("Python eval callback for decimal failed")
                    })
                });
            }
        }

        if let Some(f) = self.decimal_complex {
            if tag_count == 0 {
                info = info.register(move |args: &[Complex<Float>]| {
                    Python::attach(|py| {
                        let args = args
                            .iter()
                            .map(|x| (x.re.clone().into(), x.im.clone().into()))
                            .collect::<Vec<(PythonMultiPrecisionFloat, PythonMultiPrecisionFloat)>>(
                            );
                        let (re, im) = f.call1(py, (args.into_py_any(py)?,))?.extract::<(
                            PythonMultiPrecisionFloat,
                            PythonMultiPrecisionFloat,
                        )>(
                            py
                        )?;
                        Ok::<Complex<Float>, PyErr>(Complex::new(re.0, im.0))
                    })
                    .expect("Python eval callback for decimal complex failed")
                });
            } else {
                info = info.register_tagged(move |tags| {
                    let f =
                        Python::attach(|py| Self::python_eval_callable(py, &f, tags, tag_count))
                            .expect("Python tagged eval callback for decimal complex failed");
                    Box::new(move |args: &[Complex<Float>]| {
                        Python::attach(|py| {
                            let args = args
                                .iter()
                                .map(|x| (x.re.clone().into(), x.im.clone().into()))
                                .collect::<Vec<(
                                    PythonMultiPrecisionFloat,
                                    PythonMultiPrecisionFloat,
                                )>>();
                            let (re, im) = f.call1(py, (args.into_py_any(py)?,))?.extract::<(
                                PythonMultiPrecisionFloat,
                                PythonMultiPrecisionFloat,
                            )>(
                                py
                            )?;
                            Ok::<Complex<Float>, PyErr>(Complex::new(re.0, im.0))
                        })
                        .expect("Python eval callback for decimal complex failed")
                    })
                });
            }
        }

        if let Some(snippet) = self.cpp {
            info.with_cpp(snippet)
        } else {
            info
        }
    }

    fn get_eval_callable(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Py<PyAny>>> {
        if let Ok(Some(value)) = dict.get_item(key) {
            if !value.is_callable() {
                return Err(exceptions::PyTypeError::new_err(format!(
                    "eval['{key}'] must be callable"
                )));
            }

            return Ok(Some(value.unbind()));
        }

        Ok(None)
    }

    fn get_eval_string(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
        if let Ok(Some(value)) = dict.get_item(key) {
            return value.extract::<String>().map(Some).map_err(|_| {
                exceptions::PyTypeError::new_err(format!("eval['{key}'] must be a string"))
            });
        }

        Ok(None)
    }

    fn decimal_digits_from_binary_prec(prec: u32) -> u32 {
        ((prec as f64 / std::f64::consts::LOG2_10).ceil() as u32).max(1)
    }

    fn extract_python_constant(py: Python, value: Py<PyAny>) -> PyResult<Complex<Float>> {
        if let Ok((re, im)) = value.extract::<(PythonExpression, PythonExpression)>(py)
            && let Ok(re_f) = Float::try_from(&re.expr)
            && let Ok(im_f) = Float::try_from(&im.expr)
        {
            Ok(Complex::new(re_f, im_f))
        } else if let Ok(re) = value.extract::<PythonExpression>(py)
            && let Ok(re_f) = Float::try_from(&re.expr)
        {
            Ok(re_f.into())
        } else if let Ok((re, im)) =
            value.extract::<(PythonMultiPrecisionFloat, PythonMultiPrecisionFloat)>(py)
        {
            Ok(Complex::new(re.0, im.0))
        } else if let Ok(re) = value.extract::<PythonMultiPrecisionFloat>(py) {
            Ok(re.0.into())
        } else if let Ok(value) = value.extract::<Complex<f64>>(py) {
            Ok(Complex::new(value.re.into(), value.im.into()))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "eval['constant'] must return a number or a (real, imag) tuple",
            ))
        }
    }

    fn atom_view_tags_to_python(py: Python, tags: &[AtomView]) -> PyResult<Py<PyAny>> {
        tags.iter()
            .map(|x| PythonExpression::from(x.to_owned()))
            .collect::<Vec<_>>()
            .into_py_any(py)
    }

    fn python_eval_callable(
        py: Python,
        f: &Py<PyAny>,
        tags: &[AtomView],
        tag_count: usize,
    ) -> PyResult<Py<PyAny>> {
        if tags.len() != tag_count {
            return Err(exceptions::PyValueError::new_err(format!(
                "Python eval expected {tag_count} tags, got {}",
                tags.len()
            )));
        }

        if tag_count == 0 {
            Ok(f.clone_ref(py))
        } else {
            let tagged = f.call1(py, (Self::atom_view_tags_to_python(py, tags)?,))?;
            if tagged.bind(py).is_callable() {
                Ok(tagged)
            } else {
                Err(exceptions::PyTypeError::new_err(
                    "tagged Python eval callback must return a callable",
                ))
            }
        }
    }
}

#[cfg(feature = "python_stubgen")]
submit! {
PyFunctionInfo {
            name: "S",
            parameters: &[
                ParameterInfo {
                    name: "names",
                    kind: ParameterKind::VarPositional,
                    type_info: || <&str>::type_input(),
                    default: ParameterDefault::Expr(NONE_ARG),
                },
                ParameterInfo {
                    name: "is_symmetric",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_antisymmetric",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_cyclesymmetric",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_linear",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_scalar",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_real",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_integer",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_positive",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "tags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
            ],
            r#return: || Vec::<PythonExpression>::type_output(),
            doc:
r#"Create new symbols from `names`. Symbols can have attributes,
such as symmetries. If no attributes
are specified and the symbol was previously defined, the attributes are inherited.
Once attributes are defined on a symbol, they cannot be redefined later.

Examples
--------
Define a regular symbol and use it as a variable:
>>> x = S('x')
>>> e = x**2 + 5
>>> print(e)
x**2 + 5

Define a regular symbol and use it as a function:
>>> f = S('f')
>>> e = f(1,2)
>>> print(e)
f(1,2)


Define a symmetric function:
>>> f = S('f', is_symmetric=True)
>>> e = f(2,1)
>>> print(e)
f(1,2)


Define a linear and symmetric function:
>>> p1, p2, p3, p4 = S('p1', 'p2', 'p3', 'p4')
>>> dot = S('dot', is_symmetric=True, is_linear=True)
>>> e = dot(p2+2*p3,p1+3*p2-p3)
dot(p1,p2)+2*dot(p1,p3)+3*dot(p2,p2)-dot(p2,p3)+6*dot(p2,p3)-2*dot(p3,p3)

Parameters
----------
names : str
    The name(s) of the symbol(s)
is_symmetric : Optional[bool]
    Set to true if the symbol is symmetric.
is_antisymmetric : Optional[bool]
    Set to true if the symbol is antisymmetric.
is_cyclesymmetric : Optional[bool]
    Set to true if the symbol is cyclesymmetric.
is_linear : Optional[bool]
    Set to true if the symbol is linear.
is_scalar : Optional[bool]
    Set to true if the symbol is a scalar. It will be moved out of linear functions.
is_real : Optional[bool]
    Set to true if the symbol is a real number.
is_integer : Optional[bool]
    Set to true if the symbol is an integer.
is_positive : Optional[bool]
    Set to true if the symbol is a positive number.
tags: Optional[Sequence[str]]
    A list of tags to associate with the symbol."#,
            module: Some("symbolica.core"),
            is_async: false,
            deprecated: None,
            type_ignored: None,
            is_overload: true,
            file: "symbolica.rs",
            line: line!(),
            column: column!(),
            index: 0,
        }
}

#[cfg(feature = "python_stubgen")]
submit! {
PyFunctionInfo {
            name: "S",
            parameters: &[
                ParameterInfo {
                    name: "name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "is_symmetric",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_antisymmetric",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_cyclesymmetric",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_linear",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_scalar",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_real",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_integer",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_positive",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "tags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "aliases",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "normalization",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<PythonTransformer>::type_input(),
                },
                ParameterInfo {
                    name: "print",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || TypeInfo::unqualified("typing.Optional[typing.Callable[..., typing.Optional[str]]]"),
                },
                ParameterInfo {
                    name: "derivative",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || TypeInfo::unqualified("typing.Optional[typing.Callable[[Expression, int], Expression]]"),
                },
                ParameterInfo {
                    name: "series",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || TypeInfo::unqualified("typing.Optional[typing.Callable[[typing.Sequence[Series]], typing.Optional[tuple[Expression, Expression]]]]"),
                },
                ParameterInfo {
                    name: "eval",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || TypeInfo::unqualified("typing.Optional[dict[str, typing.Any]]"),
                },
                ParameterInfo {
                    name: "data",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || TypeInfo::unqualified("typing.Optional[str | int | Expression | bytes | list | dict]"),
                },
            ],
            r#return: || PythonExpression::type_output(),
            doc:
r#"Create new symbols from a `name`. Symbols can have attributes,
such as symmetries. If no attributes
are specified and the symbol was previously defined, the attributes are inherited.
Once attributes are defined on a symbol, they cannot be redefined later.

Examples
--------
Define a regular symbol and use it as a variable:
>>> x = S('x')
>>> e = x**2 + 5
>>> print(e)
x**2 + 5

Define a regular symbol and use it as a function:
>>> f = S('f')
>>> e = f(1,2)
>>> print(e)
f(1,2)


Define a symmetric function:
>>> f = S('f', is_symmetric=True)
>>> e = f(2,1)
>>> print(e)
f(1,2)


Define a linear and symmetric function:
>>> p1, p2, p3, p4 = S('p1', 'p2', 'p3', 'p4')
>>> dot = S('dot', is_symmetric=True, is_linear=True)
>>> e = dot(p2+2*p3,p1+3*p2-p3)
dot(p1,p2)+2*dot(p1,p3)+3*dot(p2,p2)-dot(p2,p3)+6*dot(p2,p3)-2*dot(p3,p3)

Define a custom normalization function:
>>> e = S('real_log', normalization=T().replace(E("x_(exp(x1_))"), E("x1_")))
>>> E("real_log(exp(x)) + real_log(5)")

Define a custom print function:
>>> def print_mu(mu: Expression, mode: PrintMode, **kwargs) -> str | None:
>>>     if mode == PrintMode.Latex:
>>>         if mu.get_type() == AtomType.Fn:
>>>             return "\\mu_{" + ",".join(a.format() for a in mu) + "}"
>>>         else:
>>>             return "\\mu"
>>> mu = S("mu", print=print_mu)
>>> expr = E("mu + mu(1,2)")
>>> print(expr.to_latex())

If the function returns `None`, the default print function is used.

Define a custom derivative function:
>>> tag = S('tag', derivative=lambda f, index: f)
>>> x = S('x')
>>> tag(3, x).derivative(x)

Define a custom series function:
>>> def expand_tag(args: Sequence[Series]) -> tuple[Expression, Expression] | None:
>>>     return E("1/x"), args[0].to_expression()
>>> tag = S("tag", series=expand_tag)

Define a numeric evaluation function:
>>> sq = S("sq", eval={"complex": lambda args: args[0] * args[0]})
>>> x = S("x")
>>> ev = sq(x).evaluator({}, {}, [x])
>>> ev.evaluate_complex([2+0j])

Parameters
----------
name : str
    The name of the symbol
is_symmetric : Optional[bool]
    Set to true if the symbol is symmetric.
is_antisymmetric : Optional[bool]
    Set to true if the symbol is antisymmetric.
is_cyclesymmetric : Optional[bool]
    Set to true if the symbol is cyclesymmetric.
is_linear : Optional[bool]
    Set to true if the symbol is linear.
is_scalar : Optional[bool]
    Set to true if the symbol is a scalar. It will be moved out of linear functions.
is_real : Optional[bool]
    Set to true if the symbol is a real number.
is_integer : Optional[bool]
    Set to true if the symbol is an integer.
is_positive : Optional[bool]
    Set to true if the symbol is a positive number.
tags: Optional[Sequence[str]]
    A list of tags to associate with the symbol.
aliases: Optional[Sequence[str]]
    A list of aliases for the symbol.
normalization : Optional[Transformer]
    A transformer that is called after every normalization. Note that the symbol
    name cannot be used in the transformer as this will lead to a definition of the
    symbol. Use a wildcard with the same attributes instead.
print : Optional[Callable[..., Optional[str]]]:
    A function that is called when printing the variable/function, which is provided as its first argument.
    This function should return a string, or `None` if the default print function should be used.
    The custom print function takes in keyword arguments that are the same as the arguments of the `format` function.
derivative: Optional[Callable[[Expression, int], Expression]]:
    A function that is called when computing the derivative of a function in a given argument.
series: Optional[Callable[[Sequence[Series]], Optional[tuple[Expression, Expression]]]]:
    A function that is called for custom series expansion. It receives the argument series and can return
    the singular factor and regularized expression, or `None` to use the default series expansion.
eval: dict[str, Any] | None:
    Numeric evaluation function(s). The dictionary may contain:
    - `tag_count: int`: the number of leading symbolic tag arguments.
    - `cpp: str`: a C++ function definition inserted into exported C++ code for this symbol.

    For arbitrary precision evaluation of constant functions, register a function that
    maps the tags and the requested decimal precision to a number:
    - `constant`: (Sequence[Expression], int) -> Decimal | float | complex | tuple[Decimal, Decimal]]

    Evaluators for non-constant functions when `tag_count = 0`:
    - `float`: Sequence[float] -> float
    - `complex`: Sequence[complex] -> complex
    - `decimal`: Sequence[Decimal] -> Decimal
    - `decimal_complex`: Sequence[tuple[Decimal, Decimal]] -> tuple[Decimal, Decimal]

    Evaluators for non-constant functions when `tag_count > 0` are generators:
    - `float`: Sequence[Expression] -> (Sequence[float] -> float)
    - `complex`: Sequence[Expression] -> (Sequence[complex] -> complex)
    - `decimal`: Sequence[Expression] -> (Sequence[Decimal] -> Decimal)
    - `decimal_complex`: Sequence[Expression] -> (Sequence[tuple[Decimal, Decimal]] -> tuple[Decimal, Decimal])
data: str | int | Expression | bytes | list | dict | None = None
    Custom user data to associate with the symbol."#,
            module: Some("symbolica.core"),
            is_async: false,
            deprecated: None,
            type_ignored: None,
            is_overload: true,
            file: "symbolica.rs",
            line: line!(),
            column: column!(),
            index: 1,
        }
}

/// Create a new Symbolica number from an int, a float, or a string.
/// A floating point number is kept as a float with the same precision as the input,
/// but it can also be converted to the smallest rational number given a `relative_error`.
///
/// Examples
/// --------
/// >>> e = N(1) / 2
/// >>> print(e)
/// 1/2
///
/// >>> print(N(1/3))
/// >>> print(N(0.33, 0.1))
/// >>> print(N('0.333`3'))
/// >>> print(N(Decimal('0.1234')))
/// 3.3333333333333331e-1
/// 1/3
/// 3.33e-1
/// 1.2340e-1
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.core")
)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pyfunction(name = "N", signature = (num,relative_error=None))]
fn number_shorthand(
    #[gen_stub(override_type(type_repr = "int | float | complex | str | decimal.Decimal", imports = ("decimal")))]
    num: Py<PyAny>,
    relative_error: Option<f64>,
    py: Python,
) -> PyResult<PythonExpression> {
    PythonExpression::num(&PythonExpression::type_object(py), py, num, relative_error)
}

/// Parse a Symbolica expression from a string.
///
/// Parameters
/// ----------
/// input: str
///     An input string. UTF-8 characters are allowed.
/// mode: ParseMode
///     The parsing mode to use. Use `ParseMode.Mathematica` to parse Mathematica expressions.
/// default_namespace: str
///     The default namespace to use when parsing symbols.
///
/// Examples
/// --------
/// >>> e = E('x^2+y+y*4')
/// >>> print(e)
/// x^2+5*y
///
/// >>> e = E('Cos[test`x] (2+ 3 I)', mode=ParseMode.Mathematica)
/// >>> print(e)
///
/// `cos(test::x)(2+3i)`
///
/// Raises
/// ------
/// ValueError
///     If the input is not a valid expression.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.core")
)]
#[pyfunction(name = "E", signature = (expr, mode=PythonParseMode::Symbolica, default_namespace=None))]
fn expression_shorthand(
    expr: &str,
    mode: PythonParseMode,
    default_namespace: Option<String>,
    py: Python,
) -> PyResult<PythonExpression> {
    PythonExpression::parse(
        &PythonExpression::type_object(py),
        py,
        expr,
        mode,
        default_namespace,
    )
}

/// Create a new transformer that maps an expression.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.core")
)]
#[pyfunction(name = "T")]
fn transformer_shorthand() -> PythonTransformer {
    PythonTransformer::new()
}

#[pyfunction(name = "P", signature = (expr, default_namespace=None, modulus = None, power = None, minimal_poly = None, vars = None))]
fn poly_shorthand(
    expr: &str,
    default_namespace: Option<String>,
    modulus: Option<u64>,
    power: Option<(u16, Symbol)>,
    minimal_poly: Option<PythonPolynomial>,
    vars: Option<Vec<PythonExpression>>,
    py: Python,
) -> PyResult<Py<PyAny>> {
    PythonExpression::parse(
        &PythonExpression::type_object(py),
        py,
        expr,
        PythonParseMode::Symbolica,
        default_namespace,
    )?
    .to_polynomial(modulus, power, minimal_poly, vars, py)
}

#[cfg(feature = "python_stubgen")]
submit! {
PyFunctionInfo {
        name: "P",
        parameters: &[
            ParameterInfo {
                name: "poly",
                kind: ParameterKind::PositionalOrKeyword,
                default: ParameterDefault::None,
                type_info: || <&str>::type_input(),
            },
            ParameterInfo {
                name: "default_namespace",
                kind: ParameterKind::PositionalOrKeyword,
                default: ParameterDefault::Expr(NONE_ARG),
                type_info: || <Option<&str>>::type_input(),
            },
            ParameterInfo {
                name: "vars",
                kind: ParameterKind::PositionalOrKeyword,
                default: ParameterDefault::Expr(NONE_ARG),
                type_info: || Option::<Vec<PythonExpression>>::type_input(),
            },
        ],
        r#return: || PythonPolynomial::type_output(),
        doc:"
Parse a string to a polynomial, optionally, with the variable ordering specified in `vars`.
All non-polynomial parts will be converted to new, independent variables.",
        module: Some("symbolica.core"),
        is_async: false,
        deprecated: None,
        type_ignored: None,
        is_overload: true,
        file: "symbolica.rs",
        line: line!(),
        column: column!(),
        index: 0,
        }
    }

#[cfg(feature = "python_stubgen")]
submit! {
    PyFunctionInfo {
        name: "P",
        parameters: &[
            ParameterInfo {
                name: "poly",
                kind: ParameterKind::PositionalOrKeyword,
                default: ParameterDefault::None,
                type_info: || <&str>::type_input(),
            },
            ParameterInfo {
                name: "minimal_poly",
                kind: ParameterKind::PositionalOrKeyword,
                default: ParameterDefault::None,
                type_info: || PythonPolynomial::type_input(),
            },
            ParameterInfo {
                name: "default_namespace",
                kind: ParameterKind::PositionalOrKeyword,
                default: ParameterDefault::Expr(NONE_ARG),
                type_info: || <Option<&str>>::type_input(),
            },
            ParameterInfo {
                name: "vars",
                kind: ParameterKind::PositionalOrKeyword,
                default: ParameterDefault::Expr(NONE_ARG),
                type_info: || Option::<Vec<PythonExpression>>::type_input(),
            },
        ],
        r#return: || PythonNumberFieldPolynomial::type_output(),
        doc: "
Parse a string to a polynomial, optionally, with the variables and the ordering specified in `vars`.
All non-polynomial elements will be converted to new independent variables.

The coefficients will be converted to a number field with the minimal polynomial `minimal_poly`.
The minimal polynomial must be a monic, irreducible univariate polynomial.",
        module: Some("symbolica.core"),
        is_async: false,
        deprecated: None,
        type_ignored: None,
        is_overload: true,
        file: "symbolica.rs",
        line: line!(),
        column: column!(),
        index: 1,
    }
}

#[cfg(feature = "python_stubgen")]
submit! {
    PyFunctionInfo {
        name: "P",
        parameters: &[
            ParameterInfo {
                name: "poly",
                kind: ParameterKind::PositionalOrKeyword,
                default: ParameterDefault::None,
                type_info: || <&str>::type_input(),
            },
            ParameterInfo {
                name: "modulus",
                kind: ParameterKind::PositionalOrKeyword,
                default: ParameterDefault::None,
                type_info: || usize::type_input(),
            },
            ParameterInfo {
                name: "power",
                kind: ParameterKind::PositionalOrKeyword,
                default: ParameterDefault::Expr(NONE_ARG),
                type_info: || Option::<(usize, PythonExpression)>::type_input(),
            },
            ParameterInfo {
                name: "default_namespace",
                kind: ParameterKind::PositionalOrKeyword,
                default: ParameterDefault::Expr(NONE_ARG),
                type_info: || <Option<&str>>::type_input(),
            },
            ParameterInfo {
                name: "minimal_poly",
                kind: ParameterKind::PositionalOrKeyword,
                default: ParameterDefault::Expr(NONE_ARG),
                type_info: || Option::<PythonPolynomial>::type_input(),
            },
            ParameterInfo {
                name: "vars",
                kind: ParameterKind::PositionalOrKeyword,
                default: ParameterDefault::Expr(NONE_ARG),
                type_info: || Option::<Vec<PythonExpression>>::type_input(),
            },
        ],
        r#return: || PythonFiniteFieldPolynomial::type_output(),
        doc: "
Parse a string to a polynomial, optionally, with the variables and the ordering specified in `vars`.
All non-polynomial elements will be converted to new independent variables.

The coefficients will be converted to finite field elements modulo `modulus`.
If on top a `power` is provided, for example `(2, a)`, the polynomial will be converted to the Galois field
`GF(modulus^2)` where `a` is the variable of the minimal polynomial of the field.

If a `minimal_poly` is provided, the Galois field will be created with `minimal_poly` as the minimal polynomial.",
        module: Some("symbolica.core"),
        is_async: false,
        deprecated: None,
        type_ignored: None,
        is_overload: true,
        file: "symbolica.rs",
        line: line!(),
        column: column!(),
        index: 2,
    }
}
