use std::{
    borrow::Borrow,
    fs::File,
    hash::{Hash, Hasher},
    io::{BufReader, BufWriter},
    ops::{Deref, Neg},
    sync::{Arc, Mutex},
};

use ahash::HashMap;
use brotli::CompressorWriter;
use numpy::{
    Complex64, IntoPyArray, PyArrayDyn, PyArrayLike1, PyArrayLike2, PyUntypedArrayMethods,
    TypeMustMatch,
    ndarray::{ArrayD, Axis},
};
use pyo3::{
    Borrowed, Bound, FromPyObject, IntoPyObject, IntoPyObjectExt, Py, PyAny, PyErr, PyRef,
    PyResult, PyTypeInfo, Python,
    exceptions::{self, PyIndexError},
    pybacked::PyBackedStr,
    pyclass::CompareOp,
    pyfunction, pymethods,
    types::{
        PyAnyMethods, PyBytes, PyComplex, PyDict, PyInt, PyModule, PyNone, PyTuple, PyTupleMethods,
        PyType, PyTypeMethods,
    },
    wrap_pyfunction,
};
use pyo3::{pyclass, types::PyModuleMethods};

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

use rug::Complete;
use self_cell::self_cell;
use smallvec::SmallVec;
use smartstring::{LazyCompact, SmartString};

#[cfg(not(feature = "python_export"))]
use pyo3::pymodule;

use crate::{
    LicenseManager,
    atom::{
        Atom, AtomCore, AtomType, AtomView, DefaultNamespace, ListIterator, Symbol,
        SymbolAttribute, UserData, UserDataKey,
    },
    coefficient::{Coefficient, CoefficientView, ConvertToRing},
    domains::{
        Ring, RingOps, SelfRing,
        algebraic_number::AlgebraicExtension,
        atom::AtomField,
        finite_field::{FiniteFieldCore, PrimeIteratorU64, ToFiniteField, Z2, Zp64, is_prime_u64},
        float::{Complex, F64, Float, PythonMultiPrecisionFloat, RealLike},
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
        CompiledSimdComplexEvaluator, CompiledSimdRealEvaluator, CudaComplexf64, CudaLoadSettings,
        CudaRealf64, EvaluationFn, EvaluatorLoader, ExportSettings, ExpressionEvaluator,
        ExpressionEvaluatorWithExternalFunctions, FunctionMap, InlineASM, Instruction,
        OptimizationSettings, Slot,
    },
    graph::{GenerationSettings, Graph, HalfEdge},
    id::{
        Condition, ConditionResult, Evaluate, Match, MatchSettings, MatchStack, Pattern,
        PatternAtomTreeIterator, PatternRestriction, Relation, ReplaceIterator, ReplaceWith,
        Replacement, WildcardRestriction,
    },
    numerical_integration::{ContinuousGrid, DiscreteGrid, Grid, MonteCarloRng, Sample},
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
    try_parse,
    utils::Settable,
    warn,
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
    fn register_module(m: &Bound<'_, PyModule>) -> PyResult<()>;
}

/// Specifies the print mode.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass_enum(module = "symbolica.core")
)]
#[pyclass(name = "ParseMode", eq, eq_int, module = "symbolica.core")]
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
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass_enum(module = "symbolica.core")
)]
#[pyclass(name = "PrintMode", eq, eq_int, module = "symbolica.core")]
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
}

impl From<PrintMode> for PythonPrintMode {
    fn from(mode: PrintMode) -> Self {
        match mode {
            PrintMode::Symbolica => PythonPrintMode::Symbolica,
            PrintMode::Latex => PythonPrintMode::Latex,
            PrintMode::Mathematica => PythonPrintMode::Mathematica,
            PrintMode::Sympy => PythonPrintMode::Sympy,
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

    m.add("__version__", env!("CARGO_PKG_VERSION"))?;

    Ok(m)
}

fn print_options_to_dict<'py>(
    options: &PrintOptions,
    py: Python<'py>,
) -> PyResult<Bound<'py, PyDict>> {
    let dict = PyDict::new(py);
    dict.set_item("mode", PythonPrintMode::from(options.mode))?;
    dict.set_item("terms_on_new_line", options.terms_on_new_line)?;
    dict.set_item("color_top_level_sum", options.color_top_level_sum)?;
    dict.set_item("color_builtin_symbols", options.color_builtin_symbols)?;
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
    dict.set_item(
        "square_brackets_for_function",
        options.square_brackets_for_function,
    )?;
    dict.set_item("num_exp_as_superscript", options.num_exp_as_superscript)?;
    dict.set_item("precision", options.precision)?;
    dict.set_item("pretty_matrix", options.pretty_matrix)?;
    dict.set_item("hide_namespace", options.hide_namespace)?;
    dict.set_item("hide_all_namespaces", options.hide_all_namespaces)?;
    dict.set_item("color_namespace", options.color_namespace)?;
    dict.set_item("max_terms", options.max_terms)?;
    dict.set_item("custom_print_mode", options.custom_print_mode.map(|x| x.1))?;
    Ok(dict)
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
#[pyfunction()]
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

/// Set the Symbolica license key for this computer. Can only be called before calling any other Symbolica functions.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyfunction(module = "symbolica.core")
)]
#[pyfunction]
fn set_license_key(key: String) -> PyResult<()> {
    Err(exceptions::PyDeprecationWarning::new_err(format!(
        "This function no longer sets the license key.
You can set the license key through the environment variable SYMBOLICA_LICENSE before importing Symbolica:

import os  # noqa: E402
os.environ['SYMBOLICA_LICENSE'] = '{}'  # noqa: E402

from symbolica import *
",
        key
    )))
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

#[pyfunction(name = "S", signature = (*names,is_symmetric=None,is_antisymmetric=None,is_cyclesymmetric=None,is_linear=None,is_scalar=None,is_real=None,is_integer=None,is_positive=None,tags=None,custom_normalization=None,custom_print=None,custom_derivative=None,data=None))]
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
    custom_normalization: Option<PythonTransformer>,
    custom_print: Option<Py<PyAny>>,
    custom_derivative: Option<Py<PyAny>>,
    data: Option<PythonUserData>,
    py: Python<'_>,
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
        custom_normalization,
        custom_print,
        custom_derivative,
        data,
    )
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
                    name: "custom_normalization",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<PythonTransformer>::type_input(),
                },
                ParameterInfo {
                    name: "custom_print",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || TypeInfo::unqualified("typing.Optional[typing.Callable[..., typing.Optional[str]]]"),
                },
                ParameterInfo {
                    name: "custom_derivative",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || TypeInfo::unqualified("typing.Optional[typing.Callable[[Expression, int], Expression]]"),
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
>>> e = S('real_log', custom_normalization=T().replace(E("x_(exp(x1_))"), E("x1_")))
>>> E("real_log(exp(x)) + real_log(5)")

Define a custom print function:
>>> def print_mu(mu: Expression, mode: PrintMode, **kwargs) -> str | None:
>>>     if mode == PrintMode.Latex:
>>>         if mu.get_type() == AtomType.Fn:
>>>             return "\\mu_{" + ",".join(a.format() for a in mu) + "}"
>>>         else:
>>>             return "\\mu"
>>> mu = S("mu", custom_print=print_mu)
>>> expr = E("mu + mu(1,2)")
>>> print(expr.to_latex())

If the function returns `None`, the default print function is used.

Define a custom derivative function:
>>> tag = S('tag', custom_derivative=lambda f, index: f)
>>> x = S('x')
>>> tag(3, x).derivative(x)

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
custom_normalization : Optional[Transformer]
    A transformer that is called after every normalization. Note that the symbol
    name cannot be used in the transformer as this will lead to a definition of the
    symbol. Use a wildcard with the same attributes instead.
custom_print : Optional[Callable[..., Optional[str]]]:
    A function that is called when printing the variable/function, which is provided as its first argument.
    This function should return a string, or `None` if the default print function should be used.
    The custom print function takes in keyword arguments that are the same as the arguments of the `format` function.
custom_derivative: Optional[Callable[[Expression, int], Expression]]:
    A function that is called when computing the derivative of a function in a given argument."#,
            module: Some("symbolica.core"),
            is_async: false,
            deprecated: None,
            type_ignored: None,
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
    py: Python<'_>,
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
#[pyfunction(name = "E", signature = (expr, mode=PythonParseMode::Symbolica, default_namespace="python"))]
fn expression_shorthand(
    expr: &str,
    mode: PythonParseMode,
    default_namespace: &str,
    py: Python,
) -> PyResult<PythonExpression> {
    PythonExpression::parse(
        &PythonExpression::type_object(py),
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

#[pyfunction(name = "P", signature = (expr, default_namespace="python", modulus = None, power = None, minimal_poly = None, vars = None))]
pub fn poly_shorthand(
    expr: &str,
    default_namespace: &str,
    modulus: Option<u64>,
    power: Option<(u16, Symbol)>,
    minimal_poly: Option<PythonPolynomial>,
    vars: Option<Vec<PythonExpression>>,
    py: Python,
) -> PyResult<Py<PyAny>> {
    PythonExpression::parse(
        &PythonExpression::type_object(py),
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
    }
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass_enum(module = "symbolica.core")
)]
#[pyclass(name = "AtomType", eq, eq_int, module = "symbolica.core")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
/// Specifies the type of the atom.
pub enum PythonAtomType {
    /// The expression is a number.
    Num,
    /// The expression is a variable.
    Var,
    /// The expression is a function.
    Fn,
    /// The expression is a sum.
    Add,
    /// The expression is a product.
    Mul,
    /// The expression is a power.
    Pow,
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass_enum(module = "symbolica.core")
)]
#[pyclass(name = "SymbolAttribute", eq, eq_int, module = "symbolica.core")]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
/// Specifies the attributes of a symbol.
pub enum PythonSymbolAttribute {
    /// The function is symmetric.
    Symmetric,
    /// The function is antisymmetric.
    Antisymmetric,
    /// The function is cyclesymmetric.
    Cyclesymmetric,
    /// The function is linear.
    Linear,
    /// The symbol represents a scalar. It will be moved out of linear functions.
    Scalar,
    /// The symbol represents a real number.
    Real,
    /// The symbol represents an integer.
    Integer,
    /// The symbol represents a positive number.
    Positive,
}

impl From<SymbolAttribute> for PythonSymbolAttribute {
    fn from(attr: SymbolAttribute) -> Self {
        match attr {
            SymbolAttribute::Symmetric => PythonSymbolAttribute::Symmetric,
            SymbolAttribute::Antisymmetric => PythonSymbolAttribute::Antisymmetric,
            SymbolAttribute::Cyclesymmetric => PythonSymbolAttribute::Cyclesymmetric,
            SymbolAttribute::Linear => PythonSymbolAttribute::Linear,
            SymbolAttribute::Scalar => PythonSymbolAttribute::Scalar,
            SymbolAttribute::Real => PythonSymbolAttribute::Real,
            SymbolAttribute::Integer => PythonSymbolAttribute::Integer,
            SymbolAttribute::Positive => PythonSymbolAttribute::Positive,
        }
    }
}

/// A Python representation of a Symbolica expression.
/// The type of the atom is provided in `atom_type`.
///
/// The `head` contains the string representation of:
/// - a number if the type is `Num`
/// - the variable if the type is `Var`
/// - the function name if the type is `Fn`
/// - otherwise it is `None`.
///
/// The tail contains the child atoms:
/// - the summand for type `Add`
/// - the factors for type `Mul`
/// - the base and exponent for type `Pow`
/// - the function arguments for type `Fn`
#[derive(Clone)]
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "AtomTree", module = "symbolica.core")]
pub struct PythonAtomTree {
    /// The type of this atom.
    #[pyo3(get)]
    pub atom_type: PythonAtomType,
    /// The string data of this atom.
    #[pyo3(get)]
    pub head: Option<String>,
    /// The list of child atoms of this atom.
    #[pyo3(get)]
    pub tail: Vec<PythonAtomTree>,
}

impl<'a> From<AtomView<'a>> for PyResult<PythonAtomTree> {
    fn from(atom: AtomView<'a>) -> Self {
        let tree = match atom {
            AtomView::Num(_) => PythonAtomTree {
                atom_type: PythonAtomType::Num,
                head: Some(format!("{}", AtomPrinter::new(atom))),
                tail: vec![],
            },
            AtomView::Var(v) => PythonAtomTree {
                atom_type: PythonAtomType::Var,
                head: Some(v.get_symbol().get_name().to_string()),
                tail: vec![],
            },
            AtomView::Fun(f) => PythonAtomTree {
                atom_type: PythonAtomType::Fn,
                head: Some(f.get_symbol().get_name().to_string()),
                tail: f.iter().map(|x| x.into()).collect::<Result<Vec<_>, _>>()?,
            },
            AtomView::Add(a) => PythonAtomTree {
                atom_type: PythonAtomType::Add,
                head: None,
                tail: a.iter().map(|x| x.into()).collect::<Result<Vec<_>, _>>()?,
            },
            AtomView::Mul(m) => PythonAtomTree {
                atom_type: PythonAtomType::Mul,
                head: None,
                tail: m.iter().map(|x| x.into()).collect::<Result<Vec<_>, _>>()?,
            },
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                PythonAtomTree {
                    atom_type: PythonAtomType::Pow,
                    head: None,
                    tail: vec![
                        <AtomView as Into<PyResult<PythonAtomTree>>>::into(b)?,
                        <AtomView as Into<PyResult<PythonAtomTree>>>::into(e)?,
                    ],
                }
            }
        };

        Ok(tree)
    }
}

#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PythonUserDataKey(UserDataKey);

impl<'py> IntoPyObject<'py> for &PythonUserDataKey {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match &self.0 {
            UserDataKey::Integer(i) => i.into_bound_py_any(py),
            UserDataKey::Atom(a) => {
                let expr: PythonExpression = a.clone().into();
                expr.into_bound_py_any(py)
            }
            UserDataKey::String(s) => s.into_bound_py_any(py),
        }
    }
}

impl<'py> FromPyObject<'_, 'py> for PythonUserDataKey {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        // TODO: allow list as key
        if let Ok(num) = ob.extract::<i64>() {
            Ok(PythonUserDataKey(UserDataKey::Integer(num)))
        } else if let Ok(a) = ob.extract::<ConvertibleToExpression>() {
            Ok(PythonUserDataKey(UserDataKey::Atom(a.to_expression().expr)))
        } else if let Ok(s) = ob.extract::<PyBackedStr>() {
            Ok(PythonUserDataKey(UserDataKey::String(s.to_string())))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to ExtendedUserDataKey",
            ))
        }
    }
}

pub struct PythonUserData(UserData);

#[cfg(feature = "python_stubgen")]
impl_stub_type!(PythonUserData = ConvertibleToExpression | PyBackedStr | PyDict | PyList | PyBytes);

impl<'py> FromPyObject<'_, 'py> for PythonUserData {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if ob.extract::<Py<PyNone>>().is_ok() {
            Ok(PythonUserData(UserData::None))
        } else if let Ok(num) = ob.extract::<i64>() {
            Ok(PythonUserData(UserData::Integer(num)))
        } else if let Ok(a) = ob.extract::<ConvertibleToExpression>() {
            Ok(PythonUserData(UserData::Atom(a.to_expression().expr)))
        } else if let Ok(s) = ob.extract::<PyBackedStr>() {
            Ok(PythonUserData(UserData::String(s.to_string())))
        } else if let Ok(list) = ob.extract::<Vec<PythonUserData>>() {
            Ok(PythonUserData(UserData::List(
                list.into_iter().map(|x| x.0).collect(),
            )))
        } else if let Ok(map) = ob.extract::<HashMap<PythonUserDataKey, PythonUserData>>() {
            Ok(PythonUserData(UserData::Map(
                map.into_iter().map(|(k, v)| (k.0, v.0)).collect(),
            )))
        } else if let Ok(bytes) = ob.extract::<&[u8]>() {
            Ok(PythonUserData(UserData::Serialized(bytes.to_vec())))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to ExtendedUserData",
            ))
        }
    }
}

// impl<'py> IntoPyObject<'py> for PythonUserData {
//     type Target = PyObject;
//     type Output = Bound<'py, Self::Target>;
//     type Error = std::convert::Infallible;

//     fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
//         match self {
//             PythonUserData(ExtendedUserData::None) => Ok(None.into_py(py).into()),
//             PythonUserData(ExtendedUserData::Integer(i)) => Ok(i.into_py(py).into()),
//             PythonUserData(ExtendedUserData::Atom(a)) => {
//                 let expr: PythonExpression = a.into();
//                 Ok(expr.into_py(py).into())
//             }
//             PythonUserData(ExtendedUserData::String(s)) => {
//                 let ps: PyBackedStr = PyBackedStr::new(s);
//                 Ok(ps.into_py(py).into())
//             }
//             PythonUserData(ExtendedUserData::List(l)) => {
//                 let pl: Vec<PythonUserData> =
//                     l.into_iter().map(|x| PythonUserData(x)).collect();
//                 Ok(pl.into_py(py).into())
//             }
//             PythonUserData(ExtendedUserData::Map(m)) => {
//                 let pm: HashMap<PythonExtendedUserDataKey, PythonUserData> = m
//                     .into_iter()
//                     .map(|(k, v)| (k, PythonUserData(v)))
//                     .collect();
//                 Ok(pm.into_py(py).into())
//             }
//             PythonUserData(ExtendedUserData::Serialized(b)) => Ok(b.into_py(py).into()),
//         }
//     }
// }

struct PythonBorrowedUserData<'a>(&'a UserData);

impl<'a, 'py> IntoPyObject<'py> for PythonBorrowedUserData<'a> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            PythonBorrowedUserData(UserData::None) => PyNone::get(py).into_bound_py_any(py),
            PythonBorrowedUserData(UserData::Integer(i)) => i.into_bound_py_any(py),
            PythonBorrowedUserData(UserData::Atom(a)) => {
                let expr: PythonExpression = a.clone().into();
                expr.into_bound_py_any(py)
            }
            PythonBorrowedUserData(UserData::String(s)) => s.into_bound_py_any(py),
            PythonBorrowedUserData(UserData::List(l)) => {
                let pl: Vec<PythonBorrowedUserData> =
                    l.into_iter().map(|x| PythonBorrowedUserData(x)).collect();
                pl.into_bound_py_any(py)
            }
            PythonBorrowedUserData(UserData::Map(m)) => {
                let pm: HashMap<_, _> = m
                    .into_iter()
                    .map(|(k, v)| {
                        Ok((
                            PythonUserDataKey(k.clone()),
                            PythonBorrowedUserData(v).into_pyobject(py)?,
                        ))
                    })
                    .collect::<Result<_, PyErr>>()?;

                pm.into_bound_py_any(py)
            }
            PythonBorrowedUserData(UserData::Serialized(b)) => b.into_bound_py_any(py),
        }
    }
}

#[derive(FromPyObject)]
pub enum ConvertibleToPattern {
    Literal(ConvertibleToExpression),
    Held(PythonHeldExpression),
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(ConvertibleToPattern = ConvertibleToExpression | PythonHeldExpression);

impl ConvertibleToPattern {
    pub fn to_pattern(self) -> PyResult<PythonHeldExpression> {
        match self {
            Self::Literal(l) => Ok(l.to_expression().expr.to_pattern().into()),
            Self::Held(e) => Ok(e),
        }
    }
}

/// A pattern that is allowed to have unbound transformers
#[derive(FromPyObject)]
pub enum ConvertibleToOpenPattern {
    Closed(ConvertibleToPattern),
    Open(PythonTransformer),
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(ConvertibleToOpenPattern = ConvertibleToPattern | PythonTransformer);

impl ConvertibleToOpenPattern {
    pub fn to_pattern(self) -> PyResult<PythonHeldExpression> {
        match self {
            Self::Closed(l) => l.to_pattern(),
            Self::Open(e) => Ok(Pattern::Transformer(Box::new((None, e.chain))).into()),
        }
    }
}

#[derive(FromPyObject)]
pub enum ConvertibleToReplaceWith {
    Pattern(ConvertibleToPattern),
    Map(Py<PyAny>),
}

#[cfg(feature = "python_stubgen")]
pub struct ReplaceFunction;

#[cfg(feature = "python_stubgen")]
impl PyStubType for ReplaceFunction {
    fn type_output() -> TypeInfo {
        TypeInfo {
            name: "typing.Callable[[dict[Expression, Expression]], Expression] | int | float | complex | decimal.Decimal".into(),
            import: {
                let mut h = std::collections::HashSet::default();
                h.insert("decimal".into());
                h
            },
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(ConvertibleToReplaceWith = ConvertibleToPattern | ReplaceFunction);

impl ConvertibleToReplaceWith {
    pub fn to_replace_with(self) -> PyResult<ReplaceWith<'static>> {
        match self {
            Self::Pattern(p) => Ok(ReplaceWith::Pattern(p.to_pattern()?.expr.into())),
            Self::Map(m) => Ok(ReplaceWith::Map(Box::new(move |match_stack| {
                let match_stack: HashMap<PythonExpression, PythonExpression> = match_stack
                    .get_matches()
                    .iter()
                    .map(|x| (Atom::var(x.0).into(), x.1.to_atom().into()))
                    .collect();

                Python::attach(|py| {
                    m.call(py, (match_stack,), None)
                        .expect("Bad callback function")
                        .extract::<PythonExpression>(py)
                        .expect("Match map does not return an expression")
                })
                .expr
            }))),
        }
    }
}

#[derive(FromPyObject)]
pub enum OneOrMultiple<T> {
    One(T),
    Multiple(Vec<T>),
}

impl<T> OneOrMultiple<T> {
    pub fn to_iter(&self) -> impl Iterator<Item = &T> {
        match self {
            OneOrMultiple::One(a) => std::slice::from_ref(a).iter(),
            OneOrMultiple::Multiple(m) => m.iter(),
        }
    }
}

/// Operations that transform an expression.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "HeldExpression", subclass, module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonHeldExpression {
    pub expr: Pattern,
}

impl From<Pattern> for PythonHeldExpression {
    fn from(expr: Pattern) -> Self {
        PythonHeldExpression { expr }
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonHeldExpression {
    /// Execute a bound transformer. If the transformer is unbound,
    /// you can call it with an expression as an argument.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x = S('x')
    /// >>> e = (x+1)**5
    /// >>> e = e.hold(T().expand())()
    /// >>> print(e)
    pub fn __call__(&self, py: Python) -> PyResult<PythonExpression> {
        let mut out = Atom::default();

        // TODO: pass a transformer state?
        py.detach(|| {
            Workspace::get_local()
                .with(|workspace| {
                    self.expr.replace_wildcards_with_matches_impl(
                        workspace,
                        &mut out,
                        &MatchStack::new(),
                        true,
                        None,
                    )
                })
                .map_err(|e| match e {
                    TransformerError::Interrupt => {
                        exceptions::PyKeyboardInterrupt::new_err("Interrupted by user")
                    }
                    TransformerError::ValueError(v) => exceptions::PyValueError::new_err(v),
                })
        })?;

        Ok(out.into())
    }

    /// Compare two expressions. If one of the expressions is not a number, an
    /// internal ordering will be used.
    fn __richcmp__(&self, other: ConvertibleToPattern, op: CompareOp) -> PyResult<PythonCondition> {
        Ok(match op {
            CompareOp::Eq => PythonCondition {
                condition: Relation::Eq(self.expr.clone(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Ne => PythonCondition {
                condition: Relation::Ne(self.expr.clone(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Ge => PythonCondition {
                condition: Relation::Ge(self.expr.clone(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Gt => PythonCondition {
                condition: Relation::Gt(self.expr.clone(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Le => PythonCondition {
                condition: Relation::Le(self.expr.clone(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Lt => PythonCondition {
                condition: Relation::Lt(self.expr.clone(), other.to_pattern()?.expr).into(),
            },
        })
    }

    /// Test if the expression is of a certain type.
    pub fn is_type(&self, atom_type: PythonAtomType) -> PythonCondition {
        PythonCondition {
            condition: Condition::Yield(Relation::IsType(
                self.expr.clone(),
                match atom_type {
                    PythonAtomType::Num => AtomType::Num,
                    PythonAtomType::Var => AtomType::Var,
                    PythonAtomType::Add => AtomType::Add,
                    PythonAtomType::Mul => AtomType::Mul,
                    PythonAtomType::Pow => AtomType::Pow,
                    PythonAtomType::Fn => AtomType::Fun,
                },
            )),
        }
    }

    /// Returns true iff `self` contains `a` literally.
    pub fn contains(&self, s: ConvertibleToOpenPattern) -> PyResult<PythonCondition> {
        Ok(PythonCondition {
            condition: Condition::Yield(Relation::Contains(
                self.expr.clone(),
                s.to_pattern()?.expr,
            )),
        })
    }

    /// Create a transformer that tests whether the pattern is found in the expression.
    /// Restrictions on the pattern can be supplied through `cond`.
    #[pyo3(signature = (lhs, cond = None, level_range = None, level_is_tree_depth = None, allow_new_wildcards_on_rhs = None))]
    pub fn matches(
        &self,
        lhs: ConvertibleToPattern,
        cond: Option<ConvertibleToPatternRestriction>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: Option<bool>,
        allow_new_wildcards_on_rhs: Option<bool>,
    ) -> PyResult<PythonCondition> {
        let conditions = cond.map(|r| r.0).unwrap_or_default();
        let settings = MatchSettings {
            level_range: level_range.unwrap_or((0, None)),
            level_is_tree_depth: level_is_tree_depth.unwrap_or(false),
            allow_new_wildcards_on_rhs: allow_new_wildcards_on_rhs.unwrap_or(false),
            ..MatchSettings::default()
        };

        Ok(PythonCondition {
            condition: Condition::Yield(Relation::Matches(
                self.expr.clone(),
                lhs.to_pattern()?.expr,
                conditions,
                settings,
            )),
        })
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __add__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        let res = Workspace::get_local().with(|workspace| {
            Ok::<Pattern, PyErr>(self.expr.add(&rhs.to_pattern()?.expr, workspace))
        })?;

        Ok(res.into())
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __radd__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        self.__add__(rhs)
    }

    ///  Subtract `other` from this transformer, returning the result.
    pub fn __sub__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        self.__add__(ConvertibleToPattern::Held(rhs.to_pattern()?.__neg__()?))
    }

    ///  Subtract this transformer from `other`, returning the result.
    pub fn __rsub__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        rhs.to_pattern()?
            .__add__(ConvertibleToPattern::Held(self.__neg__()?))
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __mul__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        let res = Workspace::get_local().with(|workspace| {
            Ok::<Pattern, PyErr>(self.expr.mul(&rhs.to_pattern()?.expr, workspace))
        });

        Ok(res?.into())
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __rmul__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        self.__mul__(rhs)
    }

    /// Divide this transformer by `other`, returning the result.
    pub fn __truediv__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        let res = Workspace::get_local().with(|workspace| {
            Ok::<Pattern, PyErr>(self.expr.div(&rhs.to_pattern()?.expr, workspace))
        });

        Ok(res?.into())
    }

    /// Divide `other` by this transformer, returning the result.
    pub fn __rtruediv__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        rhs.to_pattern()?
            .__truediv__(ConvertibleToPattern::Held(self.clone()))
    }

    /// Take `self` to power `exp`, returning the result.
    pub fn __pow__(
        &self,
        exponent: ConvertibleToPattern,
        modulo: Option<i64>,
    ) -> PyResult<PythonHeldExpression> {
        if modulo.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "Optional number argument not supported",
            ));
        }

        let res = Workspace::get_local().with(|workspace| {
            Ok::<_, PyErr>(self.expr.pow(&exponent.to_pattern()?.expr, workspace))
        });

        Ok(res?.into())
    }

    /// Take `base` to power `self`, returning the result.
    pub fn __rpow__(
        &self,
        base: ConvertibleToPattern,
        modulo: Option<i64>,
    ) -> PyResult<PythonHeldExpression> {
        base.to_pattern()?
            .__pow__(ConvertibleToPattern::Held(self.clone()), modulo)
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __xor__(&self, _rhs: Py<PyAny>) -> PyResult<PythonHeldExpression> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor an expression. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __rxor__(&self, _rhs: Py<PyAny>) -> PyResult<PythonHeldExpression> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor an expression. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Negate the current transformer, returning the result.
    pub fn __neg__(&self) -> PyResult<PythonHeldExpression> {
        let res =
            Workspace::get_local().with(|workspace| Ok::<Pattern, PyErr>(self.expr.neg(workspace)));

        Ok(res?.into())
    }
}

/// Operations that transform an expression.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "Transformer", subclass, module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonTransformer {
    pub chain: Vec<Transformer>,
}

impl PythonTransformer {
    fn append_transformer(&self, transformer: Transformer) -> PyResult<PythonTransformer> {
        let mut r = self.clone();
        r.chain.push(transformer);
        Ok(r)
    }

    fn to_pattern(&self) -> Pattern {
        Pattern::Transformer(Box::new((None, self.chain.clone())))
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonTransformer {
    /// Create a new transformer.
    #[new]
    pub fn new() -> PythonTransformer {
        PythonTransformer { chain: vec![] }
    }

    /// Execute an unbound transformer on the given expression. If the transformer
    /// is bound, use `execute()` instead.
    ///
    /// Examples
    /// --------
    /// >>> x = S('x')
    /// >>> e = T().expand()((1+x)**2)
    ///
    /// Parameters
    /// ----------
    /// expr: Expression
    ///     The expression to transform.
    /// stats_to_file: str, optional
    ///     If set, the output of the `stats` transformer will be written to a file in JSON format.
    #[pyo3(signature = (expr, stats_to_file = None))]
    pub fn __call__(
        &self,
        expr: ConvertibleToExpression,
        stats_to_file: Option<String>,
        py: Python,
    ) -> PyResult<PythonExpression> {
        let e = expr.to_expression();

        let mut out = Atom::new();

        let state = if let Some(stats_to_file) = stats_to_file {
            let file = File::create(stats_to_file).map_err(|e| {
                exceptions::PyIOError::new_err(format!(
                    "Could not create file for transformer statistics: {e}",
                ))
            })?;
            TransformerState {
                stats_export: Some(Arc::new(Mutex::new(BufWriter::new(file)))),
                ..Default::default()
            }
        } else {
            TransformerState::default()
        };

        let _ = py.detach(|| {
            Workspace::get_local()
                .with(|ws| {
                    Transformer::execute_chain(e.as_view(), &self.chain, ws, &state, &mut out)
                })
                .map_err(|e| match e {
                    TransformerError::Interrupt => {
                        exceptions::PyKeyboardInterrupt::new_err("Interrupted by user")
                    }
                    TransformerError::ValueError(v) => exceptions::PyValueError::new_err(v),
                })
        })?;

        Ok(out.into())
    }

    /// Compare two expressions. If one of the expressions is not a number, an
    /// internal ordering will be used.
    fn __richcmp__(
        &self,
        other: ConvertibleToOpenPattern,
        op: CompareOp,
    ) -> PyResult<PythonCondition> {
        Ok(match op {
            CompareOp::Eq => PythonCondition {
                condition: Relation::Eq(self.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Ne => PythonCondition {
                condition: Relation::Ne(self.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Ge => PythonCondition {
                condition: Relation::Ge(self.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Gt => PythonCondition {
                condition: Relation::Gt(self.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Le => PythonCondition {
                condition: Relation::Le(self.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Lt => PythonCondition {
                condition: Relation::Lt(self.to_pattern(), other.to_pattern()?.expr).into(),
            },
        })
    }

    /// Test if the expression is of a certain type.
    pub fn is_type(&self, atom_type: PythonAtomType) -> PythonCondition {
        PythonCondition {
            condition: Condition::Yield(Relation::IsType(
                self.to_pattern(),
                match atom_type {
                    PythonAtomType::Num => AtomType::Num,
                    PythonAtomType::Var => AtomType::Var,
                    PythonAtomType::Add => AtomType::Add,
                    PythonAtomType::Mul => AtomType::Mul,
                    PythonAtomType::Pow => AtomType::Pow,
                    PythonAtomType::Fn => AtomType::Fun,
                },
            )),
        }
    }

    /// Returns true iff `self` contains `a` literally.
    pub fn contains(&self, s: ConvertibleToOpenPattern) -> PyResult<PythonCondition> {
        Ok(PythonCondition {
            condition: Condition::Yield(Relation::Contains(
                self.to_pattern(),
                s.to_pattern()?.expr,
            )),
        })
    }

    /// Create a transformer that tests whether the pattern is found in the expression.
    /// Restrictions on the pattern can be supplied through `cond`.
    #[pyo3(signature = (lhs, cond = None, level_range = None, level_is_tree_depth = None, allow_new_wildcards_on_rhs = None))]
    pub fn matches(
        &self,
        lhs: ConvertibleToOpenPattern,
        cond: Option<ConvertibleToPatternRestriction>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: Option<bool>,
        allow_new_wildcards_on_rhs: Option<bool>,
    ) -> PyResult<PythonCondition> {
        let conditions = cond.map(|r| r.0).unwrap_or_default();
        let settings = MatchSettings {
            level_range: level_range.unwrap_or((0, None)),
            level_is_tree_depth: level_is_tree_depth.unwrap_or(false),
            allow_new_wildcards_on_rhs: allow_new_wildcards_on_rhs.unwrap_or(false),
            ..MatchSettings::default()
        };

        Ok(PythonCondition {
            condition: Condition::Yield(Relation::Matches(
                self.to_pattern(),
                lhs.to_pattern()?.expr,
                conditions,
                settings,
            )),
        })
    }

    /// Create a transformer that expands products and powers.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x, x_ = S('x', 'x_')
    /// >>> f = S('f')
    /// >>> e = f((x+1)**2).replace(f(x_), x_.hold(T().expand()))
    /// >>> print(e)
    #[pyo3(signature = (var = None, via_poly = None))]
    pub fn expand(
        &self,
        var: Option<ConvertibleToExpression>,
        via_poly: Option<bool>,
    ) -> PyResult<PythonTransformer> {
        if let Some(var) = var {
            let e = var.to_expression();
            if matches!(e.expr, Atom::Var(_) | Atom::Fun(_)) {
                self.append_transformer(Transformer::Expand(
                    Some(e.expr),
                    via_poly.unwrap_or(false),
                ))
            } else {
                Err(exceptions::PyValueError::new_err(
                    "Expansion must be done wrt an indeterminate",
                ))
            }
        } else {
            self.append_transformer(Transformer::Expand(None, via_poly.unwrap_or(false)))
        }
    }

    /// Create a transformer that distributes numbers in the expression, for example:
    /// `2*(x+y)` -> `2*x+2*y`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> x, y = S('x', 'y')
    /// >>> e = 3*(x+y)*(4*x+5*y)
    /// >>> print(Transformer().expand_num()(e))
    ///
    /// yields
    ///
    /// ```log
    /// (3*x+3*y)*(4*x+5*y)
    /// ```
    pub fn expand_num(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::ExpandNum)
    }

    /// Create a transformer that computes the product of a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x__ = S('x__')
    /// >>> f = S('f')
    /// >>> e = f(2,3).replace(f(x__), x__.hold(T().prod()))
    /// >>> print(e)
    pub fn prod(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Product)
    }

    /// Create a transformer that computes the sum of a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x__ = S('x__')
    /// >>> f = S('f')
    /// >>> e = f(2,3).replace(f(x__), x__.hold(T().sum()))
    /// >>> print(e)
    pub fn sum(&self) -> PyResult<PythonTransformer> {
        let mut r = self.clone();
        r.chain.push(Transformer::Sum);
        Ok(r)
    }

    /// Create a transformer that returns the number of arguments.
    /// If the argument is not a function, return 0.
    ///
    /// If `only_for_arg_fun` is `True`, only count the number of arguments
    /// in the `arg()` function and return 1 if the input is not `arg`.
    /// This is useful for obtaining the length of a range during pattern matching.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x__ = S('x__')
    /// >>> f = S('f')
    /// >>> e = f(2,3,4).replace(f(x__), x__.hold(T().nargs()))
    /// >>> print(e)
    #[pyo3(signature = (only_for_arg_fun = false))]
    pub fn nargs(&self, only_for_arg_fun: bool) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::ArgCount(only_for_arg_fun))
    }

    /// Create a transformer that linearizes a function, optionally extracting `symbols`
    /// as well.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x, y, z, w, f, x__ = S('x', 'y', 'z', 'w', 'f', 'x__')
    /// >>> e = f(x+y, 4*z*w+3).replace(f(x__), f(x__).hold(T().linearize([z])))
    /// >>> print(e)
    ///
    /// yields `f(x,3)+f(y,3)+4*z*f(x,w)+4*z*f(y,w)`.
    #[pyo3(signature = (symbols = None))]
    pub fn linearize(&self, symbols: Option<Vec<PythonExpression>>) -> PyResult<PythonTransformer> {
        let mut c_symbols = vec![];
        if let Some(symbols) = symbols {
            for s in symbols {
                if let AtomView::Var(v) = s.expr.as_view() {
                    c_symbols.push(v.get_symbol());
                } else {
                    return Err(exceptions::PyValueError::new_err(
                        "Can only linearize in variables",
                    ));
                }
            }
        }

        self.append_transformer(Transformer::Linearize(if c_symbols.is_empty() {
            None
        } else {
            Some(c_symbols)
        }))
    }

    /// Create a transformer that sorts a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x_ = S('x__')
    /// >>> f = S('f')
    /// >>> e = f(3,2,1).replace(f(x__), x__.hold(T().sort()))
    /// >>> print(e)
    pub fn sort(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Sort)
    }

    /// Create a transformer that cycle-symmetrizes a function.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x_ = S('x__')
    /// >>> f = S('f')
    /// >>> e = f(1,2,4,1,2,3).replace(f(x__), x_.hold(T().cycle_symmetrize()))
    /// >>> print(e)
    ///
    /// Yields `f(1,2,3,1,2,4)`.
    pub fn cycle_symmetrize(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::CycleSymmetrize)
    }

    /// Create a transformer that removes elements from a list if they occur
    /// earlier in the list as well.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x__ = S('x__')
    /// >>> f = S('f')
    /// >>> e = f(1,2,1,2).replace(f(x__), x__.hold(T().deduplicate()))
    /// >>> print(e)
    ///
    /// Yields `f(1,2)`.
    pub fn deduplicate(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Deduplicate)
    }

    /// Create a transformer that extracts a rational polynomial from a coefficient.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> e = Expression.COEFF((x^2+1)/y^2).hold(T().from_coeff())
    /// >>> print(e)
    pub fn from_coeff(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::FromNumber)
    }

    /// Create a transformer that split a sum or product into a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x, x__ = S('x', 'x__')
    /// >>> f = S('f')
    /// >>> e = (x + 1).replace(x__, f(x__.hold(T().split())))
    /// >>> print(e)
    pub fn split(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Split)
    }

    /// Create a transformer that partitions a list of arguments into named bins of a given length,
    /// returning all partitions and their multiplicity.
    ///
    /// If the unordered list `elements` is larger than the bins, setting the flag `fill_last`
    /// will add all remaining elements to the last bin.
    ///
    /// Setting the flag `repeat` means that the bins will be repeated to exactly fit all elements,
    /// if possible.
    ///
    /// Note that the functions names to be provided for the bin names must be generated through `Expression.var`.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x_, f_id, g_id = S('x__', 'f', 'g')
    /// >>> f = S('f')
    /// >>> e = f(1,2,1,3).replace(f(x_), x_.hold(T().partitions([(f_id, 2), (g_id, 1), (f_id, 1)])))
    /// >>> print(e)
    ///
    /// yields:
    /// `2*f(1)*f(1,2)*g(3)+2*f(1)*f(1,3)*g(2)+2*f(1)*f(2,3)*g(1)+f(2)*f(1,1)*g(3)+2*f(2)*f(1,3)*g(1)+f(3)*f(1,1)*g(2)+2*f(3)*f(1,2)*g(1)`
    #[pyo3(signature = (bins, fill_last = false, repeat = false))]
    pub fn partitions(
        &self,
        bins: Vec<(ConvertibleToPattern, usize)>,
        fill_last: bool,
        repeat: bool,
    ) -> PyResult<PythonTransformer> {
        let mut conv_bins = vec![];

        for (x, len) in bins {
            let id = match &x.to_pattern()?.expr {
                Pattern::Literal(x) => {
                    if let AtomView::Var(x) = x.as_view() {
                        x.get_symbol()
                    } else {
                        return Err(exceptions::PyValueError::new_err(
                            "Derivative must be taken wrt a variable",
                        ));
                    }
                }
                Pattern::Wildcard(x) => *x,
                _ => {
                    return Err(exceptions::PyValueError::new_err(
                        "Derivative must be taken wrt a variable",
                    ));
                }
            };

            conv_bins.push((id, len));
        }

        self.append_transformer(Transformer::Partition(conv_bins, fill_last, repeat))
    }

    /// Create a transformer that generates all permutations of a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x_, f_id = S('x__', 'f')
    /// >>> f = S('f')
    /// >>> e = f(1,2,1,2).replace(f(x_), x_.hold(T().permutations(f_id)))
    /// >>> print(e)
    ///
    /// yields:
    /// `4*f(1,1,2,2)+4*f(1,2,1,2)+4*f(1,2,2,1)+4*f(2,1,1,2)+4*f(2,1,2,1)+4*f(2,2,1,1)`
    pub fn permutations(&self, function_name: ConvertibleToPattern) -> PyResult<PythonTransformer> {
        let id = match &function_name.to_pattern()?.expr {
            Pattern::Literal(x) => {
                if let AtomView::Var(x) = x.as_view() {
                    x.get_symbol()
                } else {
                    return Err(exceptions::PyValueError::new_err(
                        "Derivative must be taken wrt a variable",
                    ));
                }
            }
            Pattern::Wildcard(x) => *x,
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Derivative must be taken wrt a variable",
                ));
            }
        };

        self.append_transformer(Transformer::Permutations(id))
    }

    /// Create a transformer that apply a function `f`.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x_ = S('x_')
    /// >>> f = S('f')
    /// >>> e = f(2).replace(f(x_), x_.hold(T().map(lambda r: r**2)))
    /// >>> print(e)
    pub fn map(
        &self,
        #[gen_stub(override_type(
            type_repr = "typing.Callable[[Expression], Expression | int | float | complex | decimal.Decimal]"
        ))]
        f: Py<PyAny>,
    ) -> PyResult<PythonTransformer> {
        let transformer = Transformer::Map(Box::new(move |expr, _state, out| {
            let expr = PythonExpression {
                expr: expr.to_owned(),
            };

            let res = Python::attach(|py| {
                f.call(py, (expr,), None)
                    .map_err(|e| {
                        TransformerError::ValueError(format!("Bad callback function: {e}"))
                    })?
                    .extract::<ConvertibleToExpression>(py)
                    .map_err(|e| {
                        TransformerError::ValueError(format!(
                            "Function does not return a pattern, but {e}",
                        ))
                    })
            });

            match res {
                Ok(res) => {
                    out.set_from_view(&res.to_expression().expr.as_view());
                    Ok(())
                }
                Err(e) => Err(e),
            }
        }));

        self.append_transformer(transformer)
    }

    /// Map a chain of transformers over the terms of the expression, optionally using multiple cores.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x, y = S('x', 'y')
    /// >>> t = T().map_terms(T().print(), n_cores=2)
    /// >>> e = t(x + y)
    #[pyo3(signature = (*transformers, n_cores=1))]
    pub fn map_terms(
        &self,
        transformers: &Bound<'_, PyTuple>,
        n_cores: usize,
    ) -> PyResult<PythonTransformer> {
        let mut rep_chain = vec![];
        // fuse all sub-transformers into one chain
        for r in transformers {
            let p = r.extract::<PythonTransformer>()?;
            rep_chain.extend(p.chain);
        }

        let pool = if n_cores < 2 || !LicenseManager::is_licensed() {
            None
        } else {
            Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n_cores)
                    .build()
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!(
                            "Could not create thread pool: {e}",
                        ))
                    })?,
            ))
        };

        self.append_transformer(Transformer::MapTerms(rep_chain, pool))
    }

    /// Create a transformer that applies a transformer chain to every argument of the `arg()` function.
    /// If the input is not `arg()`, the transformer is applied to the input.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x = S('x')
    /// >>> f = S('f')
    /// >>> e = (1+x).hold(T().split().for_each(T().map(f)))()
    #[pyo3(signature = (*transformers))]
    pub fn for_each(&self, transformers: &Bound<'_, PyTuple>) -> PyResult<PythonTransformer> {
        let mut rep_chain = vec![];
        // fuse all sub-transformers into one chain
        for r in transformers {
            let p = r.extract::<PythonTransformer>()?;
            rep_chain.extend(p.chain);
        }

        self.append_transformer(Transformer::ForEach(rep_chain))
    }

    /// Create a transformer that checks for a Python interrupt,
    /// such as ctrl-c and aborts the current transformer.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x_ = S('x_')
    /// >>> f = S('f')
    /// >>> f(10).hold(T().repeat(T().replace(
    /// >>> f(x_), f(x_+1)).check_interrupt()))()
    pub fn check_interrupt(&self) -> PyResult<PythonTransformer> {
        let transformer = Transformer::Map(Box::new(move |expr, _state, out| {
            out.set_from_view(&expr);
            Python::attach(|py| py.check_signals()).map_err(|_| TransformerError::Interrupt)
        }));

        self.append_transformer(transformer)
    }

    /// Create a transformer that keeps executing the transformer chain until the input equals the output.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S('f')
    /// >>> e = E("f(5)")
    /// >>> e = e.hold(T().repeat(
    /// >>>     T().expand(),
    /// >>>     T().replace(f(x_), f(x_ - 1) + f(x_ - 2), x_.req_gt(1))
    /// >>> ))()
    #[pyo3(signature = (*transformers))]
    pub fn repeat(&self, transformers: &Bound<'_, PyTuple>) -> PyResult<PythonTransformer> {
        let mut rep_chain = vec![];
        // fuse all sub-transformers into one chain
        for r in transformers {
            let p = r.extract::<PythonTransformer>()?;
            rep_chain.extend(p.chain);
        }

        self.append_transformer(Transformer::Repeat(rep_chain))
    }

    /// Evaluate the condition and apply the `if_block` if the condition is true, otherwise apply the `else_block`.
    /// The expression that is the input of the transformer is the input for the condition, the `if_block` and the `else_block`.
    ///
    /// Examples
    /// --------
    /// >>> t = T.map_terms(T.if_then(T.contains(x), T.print()))
    /// >>> t(x + y + 4)
    ///
    /// prints `x`.
    #[pyo3(signature = (condition, if_block, else_block = None))]
    pub fn if_then(
        &self,
        condition: PythonCondition,
        if_block: PythonTransformer,
        else_block: Option<PythonTransformer>,
    ) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::IfElse(
            condition.condition,
            if_block.chain,
            else_block.map(|x| x.chain).unwrap_or_default(),
        ))
    }

    /// Execute the `condition` transformer. If the result of the `condition` transformer is different from the input expression,
    /// apply the `if_block`, otherwise apply the `else_block`. The input expression of the `if_block` is the output
    /// of the `condition` transformer.
    ///
    /// Examples
    /// --------
    /// >>> t = T.map_terms(T.if_changed(T.replace(x, y), T.print()))
    /// >>> print(t(x + y + 4))
    ///
    /// prints
    /// ```log
    /// y
    /// 2*y+4
    /// ```
    #[pyo3(signature = (condition, if_block, else_block = None))]
    pub fn if_changed(
        &self,
        condition: PythonTransformer,
        if_block: PythonTransformer,
        else_block: Option<PythonTransformer>,
    ) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::IfChanged(
            condition.chain,
            if_block.chain,
            else_block.map(|x| x.chain).unwrap_or_default(),
        ))
    }

    /// Break the current chain and all higher-level chains containing `if` transformers.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> t = T.map_terms(T.repeat(
    /// >>>     T.replace(y, 4),
    /// >>>     T.if_changed(T.replace(x, y),
    /// >>>                 T.break_chain()),
    /// >>>     T.print()  # print of y is never reached
    /// >>> ))
    /// >>> print(t(x))
    pub fn break_chain(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::BreakChain)
    }

    /// Chain several transformers. `chain(A,B,C)` is the same as `A.B.C`,
    /// where `A`, `B`, `C` are transformers.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x_ = S('x_')
    /// >>> f = S('f')
    /// >>> e = E("f(5)")
    /// >>> e = e.hold(T().repeat(
    /// >>>     T().expand(),
    /// >>>     T().replace(f(x_), f(x_ - 1) + f(x_ - 2), x_.req_gt(1))
    /// >>> ))()
    #[pyo3(signature = (*transformers))]
    pub fn chain(&self, transformers: &Bound<'_, PyTuple>) -> PyResult<PythonTransformer> {
        let mut r = self.clone();
        for t in transformers {
            let p = t.extract::<PythonTransformer>()?;
            r.chain.extend(p.chain);
        }

        Ok(r)
    }

    /// Set the coefficient ring to contain the variables in the `vars` list.
    /// This will move all variables into a rational polynomial function.
    ///
    /// Parameters
    /// ----------
    /// vars: List[Expression]
    ///     A list of variables
    pub fn set_coefficient_ring(&self, vars: Vec<PythonExpression>) -> PyResult<PythonTransformer> {
        let mut var_map = vec![];
        for v in vars {
            var_map.push(
                v.expr
                    .try_into()
                    .map_err(|e| exceptions::PyValueError::new_err(e))?,
            );
        }

        let a = Arc::new(var_map);

        self.append_transformer(Transformer::Map(Box::new(move |i, _state, o| {
            *o = i.set_coefficient_ring(&a);
            Ok(())
        })))
    }

    /// Create a transformer that collects terms involving the same power of `x`,
    /// where `x` is an indeterminate.
    /// Return the list of key-coefficient pairs and the remainder that matched no key.
    ///
    /// Both the key (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` transformers respectively.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x, y = S('x', 'y')
    /// >>> e = 5*x + x * y + x**2 + 5
    /// >>>
    /// >>> print(e.hold(T().collect(x))())
    ///
    /// yields `x^2+x*(y+5)+5`.
    ///
    /// >>> from symbolica import Expression, T
    /// >>> x, y, x_, var, coeff = S('x', 'y', 'x_', 'var', 'coeff')
    /// >>> e = 5*x + x * y + x**2 + 5
    /// >>> print(e.collect(x, key_map=T().replace(x_, var(x_)),
    ///         coeff_map=T().replace(x_, coeff(x_))))
    ///
    /// yields `var(1)*coeff(5)+var(x)*coeff(y+5)+var(x^2)*coeff(1)`.
    ///
    /// Parameters
    /// ----------
    /// x: Expression
    ///     The variable to collect terms in
    /// key_map: Transformer
    ///     A transformer to be applied to the quantity collected in
    /// coeff_map: Transformer
    ///     A transformer to be applied to the coefficient
    #[pyo3(signature = (*x, key_map = None, coeff_map = None))]
    pub fn collect(
        &self,
        x: Bound<'_, PyTuple>,
        key_map: Option<PythonTransformer>,
        coeff_map: Option<PythonTransformer>,
    ) -> PyResult<PythonTransformer> {
        let mut xs = vec![];
        for a in x {
            if let Ok(r) = a.extract::<PythonExpression>() {
                if matches!(r.expr, Atom::Var(_) | Atom::Fun(_)) {
                    xs.push(r.expr);
                } else {
                    return Err(exceptions::PyValueError::new_err(
                        "Collect must be done wrt a variable or function",
                    ));
                }
            } else {
                return Err(exceptions::PyValueError::new_err(
                    "Collect must be done wrt a variable or function",
                ));
            }
        }

        let key_map = if let Some(key_map) = key_map {
            key_map.chain
        } else {
            vec![]
        };

        let coeff_map = if let Some(coeff_map) = coeff_map {
            coeff_map.chain
        } else {
            vec![]
        };

        self.append_transformer(Transformer::Collect(xs, key_map, coeff_map))
    }

    /// Create a transformer that collects terms involving the same power of variables or functions with the name `x`.
    ///
    /// Both the key (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` transformers respectively.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x, f = S('x', 'f')
    /// >>> e = f(1,2) + x*f(1,2)
    /// >>>
    /// >>> print(e.hold(T().collect_symbol(x))())
    ///
    /// yields `(1+x)*f(1,2)`.
    ///
    /// Parameters
    /// ----------
    /// x: Expression
    ///      The symbol to collect in
    /// key_map: Transformer
    ///     A transformer to be applied to the quantity collected in
    /// coeff_map: Transformer
    ///     A transformer to be applied to the coefficient
    #[pyo3(signature = (x, key_map = None, coeff_map = None))]
    pub fn collect_symbol(
        &self,
        x: PythonExpression,
        key_map: Option<PythonTransformer>,
        coeff_map: Option<PythonTransformer>,
    ) -> PyResult<PythonTransformer> {
        let Some(x) = x.expr.get_symbol() else {
            return Err(exceptions::PyValueError::new_err(
                "Collect must be done wrt a variable or function",
            ));
        };

        let key_map = if let Some(key_map) = key_map {
            key_map.chain
        } else {
            vec![]
        };

        let coeff_map = if let Some(coeff_map) = coeff_map {
            coeff_map.chain
        } else {
            vec![]
        };

        self.append_transformer(Transformer::CollectSymbol(x, key_map, coeff_map))
    }

    /// Create a transformer that collects common factors from (nested) sums.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> e = E('x*(x+y*x+x^2+y*(x+x^2))')
    /// >>> e.hold(T().collect_factors())()
    ///
    /// yields
    ///
    /// ```log
    /// v1^2*(1+v1+v2+v2*(1+v1))
    /// ```
    pub fn collect_factors(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::CollectFactors)
    }

    /// Create a transformer that collects numerical factors by removing the numerical content from additions.
    /// For example, `-2*x + 4*x^2 + 6*x^3` will be transformed into `-2*(x - 2*x^2 - 3*x^3)`.
    ///
    /// The first argument of the addition is normalized to a positive quantity.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>>
    /// >>> x, y = S('x', 'y')
    /// >>> e = (-3*x+6*y)(2*x+2*y)
    /// >>> print(Transformer().collect_num()(e))
    ///
    /// yields
    ///
    /// ```log
    /// -6*(x-2*y)*(x+y)
    /// ```
    pub fn collect_num(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::CollectNum)
    }

    /// Complex conjugate the expression.
    pub fn conjugate(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Conjugate)
    }

    /// Create a transformer that collects terms involving the literal occurrence of `x`.
    pub fn coefficient(&self, x: ConvertibleToExpression) -> PyResult<PythonTransformer> {
        let a = x.to_expression().expr;
        self.append_transformer(Transformer::Map(Box::new(move |i, _state, o| {
            *o = i.coefficient(a.as_view());
            Ok(())
        })))
    }

    /// Create a transformer that computes the partial fraction decomposition in `x`.
    pub fn apart(&self, x: PythonExpression) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Map(Box::new(move |i, _state, o| {
            let poly = i
                .try_to_rational_polynomial::<_, _, u32>(&Q, &Z, None)
                .map_err(|e| {
                    TransformerError::ValueError(format!(
                        "Could not convert expression to rational polynomial: {e}",
                    ))
                })?;

            let x = poly
                .get_variables()
                .iter()
                .position(|v| match (v, x.expr.as_view()) {
                    (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                    (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                    _ => false,
                })
                .ok_or(TransformerError::ValueError(format!(
                    "Variable {} not found in polynomial",
                    x.expr
                )))?;

            let fs = poly.apart(x);

            Workspace::get_local().with(|ws| {
                let mut res = ws.new_atom();
                let a = res.to_add();
                for f in fs {
                    a.extend(f.to_expression().as_view());
                }

                res.as_view().normalize(ws, o);
            });

            Ok(())
        })))
    }

    /// Create a transformer that writes the expression over a common denominator.
    pub fn together(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Map(Box::new(|i, _state, o| {
            let poly = i
                .try_to_rational_polynomial::<_, _, u32>(&Q, &Z, None)
                .map_err(|e| {
                    TransformerError::ValueError(format!(
                        "Could not convert expression to rational polynomial: {e}",
                    ))
                })?;
            *o = poly.to_expression();
            Ok(())
        })))
    }

    /// Create a transformer that cancels common factors between numerators and denominators.
    /// Any non-canceling parts of the expression will not be rewritten.
    pub fn cancel(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Map(Box::new(|i, _state, o| {
            *o = i.cancel();
            Ok(())
        })))
    }

    /// Create a transformer that factors the expression over the rationals.
    pub fn factor(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Map(Box::new(|i, _state, o| {
            *o = i.factor();
            Ok(())
        })))
    }

    /// Create a transformer that derives `self` w.r.t the variable `x`.
    pub fn derivative(&self, x: PythonExpression) -> PyResult<PythonTransformer> {
        let id = x.expr.try_into().map_err(|e| {
            exceptions::PyValueError::new_err(format!(
                "Derivative must be taken wrt a variable: {e}"
            ))
        })?;

        self.append_transformer(Transformer::Derivative(id))
    }

    /// Create a transformer that series expands in `x` around `expansion_point` to depth `depth`.
    ///
    /// Examples
    /// -------
    /// >>> from symbolica import *
    /// >>> x, y = S('x', 'y')
    /// >>> f = S('f')
    /// >>>
    /// >>> e = 2* x**2 * y + f(x)
    /// >>> e = e.series(x, 0, 2)
    /// >>>
    /// >>> print(e)
    ///
    /// yields `f(0)+x*der(1,f(0))+1/2*x^2*(der(2,f(0))+4*y)`.
    #[pyo3(signature = (x, expansion_point, depth, depth_denom = 1, depth_is_absolute = true))]
    pub fn series(
        &self,
        x: PythonExpression,
        expansion_point: ConvertibleToExpression,
        depth: i64,
        depth_denom: i64,
        depth_is_absolute: bool,
    ) -> PyResult<PythonTransformer> {
        let id = x.expr.try_into().map_err(|e| {
            exceptions::PyValueError::new_err(format!(
                "Derivative must be taken wrt a variable: {e}",
            ))
        })?;

        self.append_transformer(Transformer::Series(
            id,
            expansion_point.to_expression().expr.clone(),
            (depth, depth_denom).into(),
            depth_is_absolute,
        ))
    }

    /// Create a transformer that replaces all subexpressions matching the pattern `pat` by the right-hand side `rhs`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> x, w1_, w2_ = S('x','w1_','w2_')
    /// >>> f = S('f')
    /// >>> t = T().replace(f(w1_, w2_), f(w1_ - 1, w2_**2), w1_ >= 1)
    /// >>> r = t(f(3,x))
    /// >>> print(r)
    ///
    /// Parameters
    /// ----------
    /// pat:
    ///     The pattern to match.
    /// rhs:
    ///     The right-hand side to replace the matched subexpression with. Can be a transformer, expression or a function that maps a dictionary of wildcards to an expression.
    /// cond:
    ///     Conditions on the pattern.
    /// non_greedy_wildcards:
    ///     Wildcards that try to match as little as possible.
    /// level_range:
    ///     Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
    /// level_is_tree_depth:
    ///     If set to `True`, the level is increased when going one level deeper in the expression tree.
    /// allow_new_wildcards_on_rhs:
    ///     If set to `True`, allow wildcards that do not appear in the pattern on the right-hand side.
    /// rhs_cache_size: int, optional
    ///     Cache the first `rhs_cache_size` substituted patterns. If set to `None`, an internally determined cache size is used.
    ///     **Warning**: caching should be disabled (`rhs_cache_size=0`) if the right-hand side contains side effects, such as updating a global variable.
    #[pyo3(signature = (lhs, rhs, cond = None, non_greedy_wildcards = None, level_range = None, level_is_tree_depth = None, allow_new_wildcards_on_rhs = None, rhs_cache_size = None))]
    pub fn replace(
        &self,
        lhs: ConvertibleToExpression,
        rhs: ConvertibleToReplaceWith,
        cond: Option<ConvertibleToPatternRestriction>,
        non_greedy_wildcards: Option<Vec<PythonExpression>>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: Option<bool>,
        allow_new_wildcards_on_rhs: Option<bool>,
        rhs_cache_size: Option<usize>,
    ) -> PyResult<PythonTransformer> {
        let mut settings = MatchSettings::cached();

        if let Some(ngw) = non_greedy_wildcards {
            settings.non_greedy_wildcards = ngw
                .iter()
                .map(|x| match x.expr.as_view() {
                    AtomView::Var(v) => {
                        let name = v.get_symbol();
                        if v.get_wildcard_level() == 0 {
                            return Err(exceptions::PyTypeError::new_err(
                                "Only wildcards can be restricted.",
                            ));
                        }
                        Ok(name)
                    }
                    _ => Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    )),
                })
                .collect::<Result<_, _>>()?;
        }
        if let Some(level_range) = level_range {
            settings.level_range = level_range;
        }
        if let Some(level_is_tree_depth) = level_is_tree_depth {
            settings.level_is_tree_depth = level_is_tree_depth;
        }
        if let Some(allow_new_wildcards_on_rhs) = allow_new_wildcards_on_rhs {
            settings.allow_new_wildcards_on_rhs = allow_new_wildcards_on_rhs;
        }
        if let Some(rhs_cache_size) = rhs_cache_size {
            settings.rhs_cache_size = rhs_cache_size;
        }

        self.append_transformer(Transformer::ReplaceAll(
            lhs.to_expression().expr.to_pattern(),
            rhs.to_replace_with()?,
            cond.map(|r| r.0).unwrap_or_default(),
            settings,
        ))
    }

    /// Create a transformer that replaces all atoms matching the patterns. See `replace` for more information.
    ///
    /// Examples
    /// --------
    ///
    /// >>> x, y, f = S('x', 'y', 'f')
    /// >>> e = f(x,y)
    /// >>> r = e.hold(T().replace_multiple([Replacement(x, y), Replacement(y, x)]))
    pub fn replace_multiple(
        &self,
        replacements: Vec<PythonReplacement>,
    ) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::ReplaceAllMultiple(
            replacements.into_iter().map(|r| r.replacement).collect(),
        ))
    }

    /// Create a transformer that prints the expression.
    ///
    /// Examples
    /// --------
    /// >>> E('f(10)').hold(T().print(terms_on_new_line = True))()
    #[pyo3(signature =
        (mode = PythonPrintMode::Symbolica,
            terms_on_new_line = false,
            color_top_level_sum = true,
            color_builtin_symbols = true,
            print_ring = true,
            symmetric_representation_for_finite_field = false,
            explicit_rational_polynomial = false,
            number_thousands_separator = None,
            multiplication_operator = '*',
            double_star_for_exponentiation = false,
            square_brackets_for_function = false,
            num_exp_as_superscript = true,
            precision = None,
            show_namespaces = false,
            include_attributes = false,
            max_terms = None,
            custom_print_mode = None)
        )]
    pub fn print(
        &self,
        mode: PythonPrintMode,
        terms_on_new_line: bool,
        color_top_level_sum: bool,
        color_builtin_symbols: bool,
        print_ring: bool,
        symmetric_representation_for_finite_field: bool,
        explicit_rational_polynomial: bool,
        number_thousands_separator: Option<char>,
        multiplication_operator: char,
        double_star_for_exponentiation: bool,
        square_brackets_for_function: bool,
        num_exp_as_superscript: bool,
        precision: Option<usize>,
        show_namespaces: bool,
        include_attributes: bool,
        max_terms: Option<usize>,
        custom_print_mode: Option<usize>,
    ) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Print(PrintOptions {
            terms_on_new_line,
            color_top_level_sum,
            color_builtin_symbols,
            print_ring,
            symmetric_representation_for_finite_field,
            explicit_rational_polynomial,
            number_thousands_separator,
            multiplication_operator,
            double_star_for_exponentiation,
            square_brackets_for_function,
            num_exp_as_superscript,
            mode: mode.into(),
            precision,
            pretty_matrix: false,
            hide_all_namespaces: !show_namespaces,
            color_namespace: true,
            hide_namespace: Some("python"),
            include_attributes,
            max_terms,
            custom_print_mode: custom_print_mode.map(|x| ("default", x)),
        }))
    }

    /// Print statistics of a transformer, tagging it with `tag`.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S('f')
    /// >>> e = E("f(5)")
    /// >>> e = e.hold(T().stats('replace', T().replace(f(x_), 1)))()
    ///
    /// yields
    /// ```log
    /// Stats for replace:
    ///     In  │ 1 │  10.00 B │
    ///     Out │ 1 │   3.00 B │ ⧗ 40.15µs
    /// ```
    #[pyo3(signature =
        (tag,
            transformer,
            color_medium_change_threshold = Some(10.),
            color_large_change_threshold = Some(100.))
        )]
    pub fn stats(
        &self,
        tag: String,
        transformer: PythonTransformer,
        color_medium_change_threshold: Option<f64>,
        color_large_change_threshold: Option<f64>,
    ) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Stats(
            StatsOptions {
                tag,
                color_medium_change_threshold,
                color_large_change_threshold,
            },
            transformer.chain,
        ))
    }
}

/// A Symbolica expression.
///
/// Supports standard arithmetic operations, such
/// as addition and multiplication.
///
/// Examples
/// --------
/// >>> x = S('x')
/// >>> e = x**2 + 2 - x + 1 / x**4
/// >>> print(e)
///
/// Attributes
/// ----------
/// E: Expression
///     Euler's number `e`.
/// PI: Expression
///     The mathematical constant `π`.
/// I: Expression
///     The mathematical constant `i`, where `i^2 = -1`.
/// COEFF: Expression
///     The built-in function that converts a rational polynomial to a coefficient.
/// COS: Expression
///     The built-in cosine function.
/// SIN: Expression
///     The built-in sine function.
/// EXP: Expression
///     The built-in exponential function.
/// LOG: Expression
///     The built-in logarithm function.
/// SQRT: Expression
///     The built-in square root function.
/// CONJ: Expression
///     The built-in complex conjugate function.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "Expression", subclass, module = "symbolica.core")]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PythonExpression {
    pub expr: Atom,
}

impl From<Atom> for PythonExpression {
    fn from(expr: Atom) -> Self {
        PythonExpression { expr }
    }
}

impl Deref for PythonExpression {
    type Target = Atom;

    fn deref(&self) -> &Self::Target {
        &self.expr
    }
}

/// A restriction on wildcards.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "PatternRestriction", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonPatternRestriction {
    pub condition: Condition<PatternRestriction>,
}

impl From<Condition<PatternRestriction>> for PythonPatternRestriction {
    fn from(condition: Condition<PatternRestriction>) -> Self {
        PythonPatternRestriction { condition }
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonPatternRestriction {
    /// Create a new pattern restriction that is the logical 'and' operation between two restrictions (i.e., both should hold).
    pub fn __and__(&self, other: Self) -> PythonPatternRestriction {
        (self.condition.clone() & other.condition.clone()).into()
    }

    /// Create a new pattern restriction that is the logical 'or' operation between two restrictions (i.e., one of the two should hold).
    pub fn __or__(&self, other: Self) -> PythonPatternRestriction {
        (self.condition.clone() | other.condition.clone()).into()
    }

    /// Create a new pattern restriction that takes the logical 'not' of the current restriction.
    pub fn __invert__(&self) -> PythonPatternRestriction {
        (!self.condition.clone()).into()
    }

    /// Create a pattern restriction based on the current matched variables.
    /// `match_fn` is a Python function that takes a dictionary of wildcards and their matched values
    /// and should return an integer. If the integer is less than 0, the restriction is false.
    /// If the integer is 0, the restriction is inconclusive.
    /// If the integer is greater than 0, the restriction is true.
    ///
    /// If your pattern restriction cannot decide if it holds since not all the required variables
    /// have been matched, it should return inclusive (0).
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> f, x_, y_, z_ = S('f', 'x_', 'y_', 'z_')
    /// >>>
    /// >>> def filter(m: dict[Expression, Expression]) -> int:
    /// >>>    if x_ in m and y_ in m:
    /// >>>        if m[x_] > m[y_]:
    /// >>>            return -1  # no match
    /// >>>        if z_ in m:
    /// >>>            if m[y_] > m[z_]:
    /// >>>                return -1
    /// >>>            return 1  # match
    /// >>>
    /// >>>    return 0  # inconclusive
    /// >>>
    /// >>>
    /// >>> e = f(1, 2, 3).replace(f(x_, y_, z_), 1,
    /// >>>         PatternRestriction.req_matches(filter))
    #[classmethod]
    pub fn req_matches(
        _cls: &Bound<'_, PyType>,
        #[gen_stub(override_type(
            type_repr = "typing.Callable[[dict[Expression, Expression]], int]"
        ))]
        match_fn: Py<PyAny>,
    ) -> PyResult<PythonPatternRestriction> {
        Ok(PythonPatternRestriction {
            condition: PatternRestriction::MatchStack(Box::new(move |m| {
                let matches: HashMap<PythonExpression, PythonExpression> = m
                    .get_matches()
                    .iter()
                    .map(|(s, t)| (Atom::var(*s).into(), t.to_atom().into()))
                    .collect();

                let r = Python::attach(|py| {
                    match_fn
                        .call(py, (matches,), None)
                        .expect("Bad callback function")
                        .extract::<isize>(py)
                        .expect("Pattern comparison does not return an integer")
                });

                if r < 0 {
                    false.into()
                } else if r == 0 {
                    ConditionResult::Inconclusive
                } else {
                    true.into()
                }
            }))
            .into(),
        })
    }
}

/// A restriction on wildcards.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "Condition", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonCondition {
    pub condition: Condition<Relation>,
}

impl From<Condition<Relation>> for PythonCondition {
    fn from(condition: Condition<Relation>) -> Self {
        PythonCondition { condition }
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonCondition {
    /// Return a string representation of the condition.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.condition)
    }

    /// Return a string representation of the condition.
    pub fn __str__(&self) -> String {
        format!("{}", self.condition)
    }

    /// Evaluate the condition.
    pub fn eval(&self) -> PyResult<bool> {
        Ok(self
            .condition
            .evaluate(&None)
            .map_err(exceptions::PyValueError::new_err)?
            == ConditionResult::True)
    }

    /// Return the boolean value of the condition.
    pub fn __bool__(&self) -> PyResult<bool> {
        self.eval()
    }

    /// Create a new pattern restriction that is the logical 'and' operation between two restrictions (i.e., both should hold).
    pub fn __and__(&self, other: Self) -> PythonCondition {
        (self.condition.clone() & other.condition.clone()).into()
    }

    /// Create a new pattern restriction that is the logical 'or' operation between two restrictions (i.e., one of the two should hold).
    pub fn __or__(&self, other: Self) -> PythonCondition {
        (self.condition.clone() | other.condition.clone()).into()
    }

    /// Create a new pattern restriction that takes the logical 'not' of the current restriction.
    pub fn __invert__(&self) -> PythonCondition {
        (!self.condition.clone()).into()
    }

    /// Convert the condition to a pattern restriction.
    pub fn to_req(&self) -> PyResult<PythonPatternRestriction> {
        self.condition
            .clone()
            .try_into()
            .map(|e| PythonPatternRestriction { condition: e })
            .map_err(exceptions::PyValueError::new_err)
    }
}

macro_rules! req_cmp_rel {
    ($self:ident,$num:ident,$cmp_any_atom:ident,$c:ident) => {{
        let num = if !$cmp_any_atom {
            if let Pattern::Literal(a) = $num {
                if let AtomView::Num(_) = a.as_view() {
                    a
                } else {
                    return Err("Can only compare to number");
                }
            } else {
                return Err("Can only compare to number");
            }
        } else if let Pattern::Literal(a) = $num {
            a
        } else {
            return Err("Pattern must be literal");
        };

        if let Pattern::Wildcard(name) = $self {
            if name.get_wildcard_level() == 0 {
                return Err("Only wildcards can be restricted.");
            }

            Ok(PatternRestriction::Wildcard((
                name,
                WildcardRestriction::Filter(Box::new(move |v: &Match| {
                    if let Match::Single(m) = v {
                        if !$cmp_any_atom {
                            if let AtomView::Num(_) = m {
                                return m.cmp(&num.as_view()).$c();
                            }
                        } else {
                            return m.cmp(&num.as_view()).$c();
                        }
                    }

                    false
                })),
            )))
        } else {
            Err("Only wildcards can be restricted.")
        }
    }};
}

impl TryFrom<Relation> for PatternRestriction {
    type Error = &'static str;

    fn try_from(value: Relation) -> Result<Self, &'static str> {
        match value {
            Relation::Eq(atom, atom1) => {
                req_cmp_rel!(atom, atom1, true, is_eq)
            }
            Relation::Ne(atom, atom1) => {
                req_cmp_rel!(atom, atom1, true, is_ne)
            }
            Relation::Gt(atom, atom1) => {
                req_cmp_rel!(atom, atom1, true, is_gt)
            }
            Relation::Ge(atom, atom1) => {
                req_cmp_rel!(atom, atom1, true, is_ge)
            }
            Relation::Lt(atom, atom1) => {
                req_cmp_rel!(atom, atom1, true, is_lt)
            }
            Relation::Le(atom, atom1) => {
                req_cmp_rel!(atom, atom1, true, is_le)
            }
            Relation::Contains(atom, atom1) => {
                if let Pattern::Wildcard(name) = atom {
                    if name.get_wildcard_level() == 0 {
                        return Err("Only wildcards can be restricted.");
                    }

                    if !matches!(&atom1, &Pattern::Literal(_)) {
                        return Err("Pattern must be literal");
                    }

                    Ok(PatternRestriction::Wildcard((
                        name,
                        WildcardRestriction::Filter(Box::new(move |m| {
                            let val = if let Pattern::Literal(a) = &atom1 {
                                a.as_view()
                            } else {
                                unreachable!()
                            };
                            match m {
                                Match::Single(v) => v.contains(val),
                                Match::Multiple(_, v) => v.iter().any(|x| x.contains(val)),
                                Match::FunctionName(_) => false,
                            }
                        })),
                    )))
                } else {
                    Err("LHS must be wildcard")
                }
            }
            Relation::Matches(atom, pattern, cond, settings) => {
                if let Pattern::Wildcard(name) = atom {
                    if name.get_wildcard_level() == 0 {
                        return Err("Only wildcards can be restricted.");
                    }

                    Ok(PatternRestriction::Wildcard((
                        name,
                        WildcardRestriction::Filter(Box::new(move |m| {
                            m.to_atom()
                                .pattern_match(&pattern, Some(&cond), Some(&settings))
                                .next_detailed()
                                .is_some()
                        })),
                    )))
                } else {
                    Err("LHS must be wildcard")
                }
            }
            Relation::IsType(atom, atom_type) => {
                if let Pattern::Wildcard(name) = atom {
                    Ok(PatternRestriction::Wildcard((
                        name,
                        WildcardRestriction::IsAtomType(atom_type),
                    )))
                } else {
                    Err("LHS must be wildcard")
                }
            }
        }
    }
}

impl TryFrom<Condition<Relation>> for Condition<PatternRestriction> {
    type Error = &'static str;

    fn try_from(value: Condition<Relation>) -> Result<Self, &'static str> {
        Ok(match value {
            Condition::True => Condition::True,
            Condition::False => Condition::False,
            Condition::Yield(r) => Condition::Yield(r.try_into()?),
            Condition::And(a) => Condition::And(Box::new((a.0.try_into()?, a.1.try_into()?))),
            Condition::Or(a) => Condition::Or(Box::new((a.0.try_into()?, a.1.try_into()?))),
            Condition::Not(a) => Condition::Not(Box::new((*a).try_into()?)),
        })
    }
}

pub struct ConvertibleToPatternRestriction(Condition<PatternRestriction>);

impl<'py> FromPyObject<'_, 'py> for ConvertibleToPatternRestriction {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<PythonPatternRestriction>() {
            Ok(ConvertibleToPatternRestriction(a.condition))
        } else if let Ok(a) = ob.extract::<PythonCondition>() {
            Ok(ConvertibleToPatternRestriction(
                a.condition
                    .try_into()
                    .map_err(exceptions::PyValueError::new_err)?,
            ))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to pattern restriction",
            ))
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(ConvertibleToPatternRestriction = PythonPatternRestriction | PythonCondition);

impl<'py> FromPyObject<'_, 'py> for ConvertibleToExpression {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<PythonExpression>() {
            Ok(ConvertibleToExpression(a))
        } else if let Ok(num) = ob.extract::<i64>() {
            Ok(ConvertibleToExpression(Atom::num(num).into()))
        } else if let Ok(num) = ob.cast::<PyInt>() {
            let a = num.to_string();
            let i = Integer::from(rug::Integer::parse(&a).unwrap().complete());
            Ok(ConvertibleToExpression(Atom::num(i).into()))
        } else if ob.extract::<PyBackedStr>().is_ok() {
            // disallow direct string conversion
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to expression",
            ))
        } else if let Ok(f) = ob.extract::<PythonMultiPrecisionFloat>() {
            Ok(ConvertibleToExpression(Atom::num(f.0).into()))
        } else if let Ok(num) = ob.extract::<Complex<f64>>() {
            Ok(ConvertibleToExpression(
                Atom::num(Complex::<Float>::new(num.re.into(), num.im.into())).into(),
            ))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to expression",
            ))
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(
    ConvertibleToExpression =
        PythonExpression | PyInt | PyBackedStr | pyo3::types::PyFloat | Complex64
);

impl<'py> FromPyObject<'_, 'py> for Symbol {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<PythonExpression>() {
            match a.expr.as_view() {
                AtomView::Var(v) => Ok(v.get_symbol()),
                e => Err(exceptions::PyTypeError::new_err(format!(
                    "Expected variable instead of {e}",
                ))),
            }
        } else {
            Err(exceptions::PyTypeError::new_err("Not a valid variable"))
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(Symbol = PythonExpression);

impl<'py> FromPyObject<'_, 'py> for PolyVariable {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        Ok(PolyVariable::Symbol(Symbol::extract(ob)?))
    }
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(PolyVariable = PythonExpression);

pub struct ConvertibleToExpression(PythonExpression);

impl ConvertibleToExpression {
    pub fn to_expression(self) -> PythonExpression {
        self.0
    }
}

macro_rules! req_cmp {
    ($self:ident,$num:ident,$cmp_any_atom:ident,$c:ident) => {{
        let num = $num.to_expression();

        if !$cmp_any_atom && !matches!(num.expr.as_view(), AtomView::Num(_)) {
            return Err(exceptions::PyTypeError::new_err(
                "Can only compare to number",
            ));
        };

        match $self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    condition: (
                        name,
                        WildcardRestriction::Filter(Box::new(move |v: &Match| {
                            let k = num.expr.as_view();

                            if let Match::Single(m) = v {
                                if !$cmp_any_atom {
                                    if let AtomView::Num(_) = m {
                                        return m.cmp(&k).$c();
                                    }
                                } else {
                                    return m.cmp(&k).$c();
                                }
                            }

                            false
                        })),
                    )
                        .into(),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }};
}

macro_rules! req_wc_cmp {
    ($self:ident,$other:ident,$cmp_any_atom:ident,$c:ident) => {{
        let id = match $self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }
                name
            }
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only wildcards can be restricted.",
                ));
            }
        };

        let other_id = match $other.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }
                name
            }
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only wildcards can be restricted.",
                ));
            }
        };

        Ok(PythonPatternRestriction {
            condition: (
                id,
                WildcardRestriction::Cmp(
                    other_id,
                    Box::new(move |m1: &Match, m2: &Match| {
                        if let Match::Single(a1) = m1 {
                            if let Match::Single(a2) = m2 {
                                if !$cmp_any_atom {
                                    if let AtomView::Num(_) = a1 {
                                        if let AtomView::Num(_) = a2 {
                                            return a1.cmp(a2).$c();
                                        }
                                    }
                                } else {
                                    return a1.cmp(a2).$c();
                                }
                            }
                        }
                        false
                    }),
                ),
            )
                .into(),
        })
    }};
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonExpression {
    /// Create a new symbol from a `name`. Symbols carry information about their attributes.
    /// The symbol can signal that it is symmetric if it is used as a function
    /// using `is_symmetric=True`, antisymmetric using `is_antisymmetric=True`,
    /// cyclesymmetric using `is_cyclesymmetric=True` and
    /// multilinear using `is_linear=True`. If no attributes
    /// are specified, the attributes are inherited from the symbol if it was already defined,
    /// otherwise all attributes are set to `false`.  A transformer that is executed
    /// after normalization can be defined with `custom_normalization`.
    ///
    /// Once attributes are defined on a symbol, they cannot be redefined later.
    ///
    /// Examples
    /// --------
    /// Define a regular symbol and use it as a variable:
    /// >>> x = S('x')
    /// >>> e = x**2 + 5
    /// >>> print(e)
    /// x**2 + 5
    ///
    /// Define a regular symbol and use it as a function:
    /// >>> f = S('f')
    /// >>> e = f(1,2)
    /// >>> print(e)
    /// f(1,2)
    ///
    /// Define a symmetric function:
    /// >>> f = S('f', is_symmetric=True)
    /// >>> e = f(2,1)
    /// >>> print(e)
    /// f(1,2)
    ///
    /// Define a linear and symmetric function:
    /// >>> p1, p2, p3, p4 = S('p1', 'p2', 'p3', 'p4')
    /// >>> dot = S('dot', is_symmetric=True, is_linear=True)
    /// >>> e = dot(p2+2*p3,p1+3*p2-p3)
    /// dot(p1,p2)+2*dot(p1,p3)+3*dot(p2,p2)-dot(p2,p3)+6*dot(p2,p3)-2*dot(p3,p3)
    ///
    ///
    /// Define a custom normalization function:
    /// >>> e = S('real_log', custom_normalization=Transformer().replace(E("x_(exp(x1_))"), E("x1_")))
    /// >>> E("real_log(exp(x)) + real_log(5)")
    #[gen_stub(skip)]
    #[pyo3(signature = (*names,is_symmetric=None,is_antisymmetric=None,is_cyclesymmetric=None,is_linear=None,is_scalar=None,is_real=None,is_integer=None,is_positive=None,tags=None,custom_normalization=None, custom_print=None, custom_derivative=None, data=None))]
    #[classmethod]
    pub fn symbol(
        _cls: &Bound<'_, PyType>,
        py: Python,
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
        custom_normalization: Option<PythonTransformer>,
        custom_print: Option<Py<PyAny>>,
        custom_derivative: Option<Py<PyAny>>,
        data: Option<PythonUserData>,
    ) -> PyResult<Py<PyAny>> {
        if names.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "At least one name must be provided",
            ));
        }

        let namespace = DefaultNamespace {
            namespace: "python".into(),
            data: "",
            file: "".into(),
            line: 0,
        };

        if is_symmetric.is_none()
            && is_antisymmetric.is_none()
            && is_cyclesymmetric.is_none()
            && is_linear.is_none()
            && is_scalar.is_none()
            && is_real.is_none()
            && is_integer.is_none()
            && is_positive.is_none()
            && tags.is_none()
            && custom_normalization.is_none()
            && custom_print.is_none()
            && custom_derivative.is_none()
            && data.is_none()
        {
            if names.len() == 1 {
                let name = names.get_item(0).unwrap().extract::<PyBackedStr>()?;

                let id = Symbol::new(namespace.attach_namespace(&name))
                    .build()
                    .map_err(|e| exceptions::PyTypeError::new_err(e.to_string()))?;
                let r = PythonExpression::from(Atom::var(id));
                return r.into_py_any(py);
            } else {
                let mut result = vec![];
                for a in names {
                    let name = a.extract::<PyBackedStr>()?;
                    let id = Symbol::new(namespace.attach_namespace(&name))
                        .build()
                        .map_err(|e| exceptions::PyTypeError::new_err(e.to_string()))?;

                    let r = PythonExpression::from(Atom::var(id));
                    result.push(r);
                }

                return result.into_py_any(py);
            }
        }

        let count = (is_symmetric == Some(true)) as u8
            + (is_antisymmetric == Some(true)) as u8
            + (is_cyclesymmetric == Some(true)) as u8;

        if count > 1 {
            Err(exceptions::PyValueError::new_err(
                "Function cannot be both symmetric, antisymmetric or cyclesymmetric",
            ))?;
        }

        let mut opts = vec![];

        if let Some(true) = is_symmetric {
            opts.push(SymbolAttribute::Symmetric);
        }

        if let Some(true) = is_antisymmetric {
            opts.push(SymbolAttribute::Antisymmetric);
        }

        if let Some(true) = is_cyclesymmetric {
            opts.push(SymbolAttribute::Cyclesymmetric);
        }

        if let Some(true) = is_linear {
            opts.push(SymbolAttribute::Linear);
        }

        if let Some(true) = is_scalar {
            opts.push(SymbolAttribute::Scalar);
        }

        if let Some(true) = is_real {
            opts.push(SymbolAttribute::Real);
        }

        if let Some(true) = is_integer {
            opts.push(SymbolAttribute::Integer);
        }

        if let Some(true) = is_positive {
            opts.push(SymbolAttribute::Positive);
        }

        if names.len() == 1 {
            let name = names.get_item(0).unwrap().extract::<PyBackedStr>()?;
            let name = namespace.attach_namespace(&name);

            let mut symbol = Symbol::new(name).with_attributes(opts);

            if let Some(f) = custom_normalization {
                symbol = symbol.with_normalization_function(Box::new(
                    move |input: AtomView<'_>, out: &mut Settable<Atom>| {
                        let _ = Workspace::get_local()
                            .with(|ws| {
                                Transformer::execute_chain(
                                    input,
                                    &f.chain,
                                    ws,
                                    &TransformerState::default(),
                                    &mut *out,
                                )
                            })
                            .unwrap();
                    },
                ))
            }

            if let Some(f) = custom_print {
                symbol = symbol.with_print_function(Box::new(
                    move |input: AtomView<'_>, opts: &PrintOptions| {
                        Python::attach(|py| {
                            let kwargs = print_options_to_dict(opts, py).unwrap();
                            f.call(
                                py,
                                (PythonExpression::from(input.to_owned()),),
                                Some(&kwargs),
                            )
                            .unwrap()
                            .extract::<Option<String>>(py)
                            .unwrap()
                        })
                    },
                ))
            }

            if let Some(f) = custom_derivative {
                symbol = symbol.with_derivative_function(Box::new(
                    move |input: AtomView<'_>, arg: usize, out: &mut Settable<Atom>| {
                        **out = Python::attach(|py| {
                            f.call1(py, (PythonExpression::from(input.to_owned()), arg))
                                .unwrap()
                                .extract::<PythonExpression>(py)
                                .unwrap()
                        })
                        .expr;
                    },
                ))
            }

            if let Some(t) = tags {
                symbol = symbol.with_tags(
                    t.into_iter()
                        .map(|x| {
                            if x.contains("::") {
                                x
                            } else {
                                format!("python::{x}")
                            }
                        })
                        .collect::<Vec<_>>(),
                );
            }

            if let Some(t) = data {
                symbol = symbol.with_user_data(t.0);
            }

            let symbol = symbol
                .build()
                .map_err(|e| exceptions::PyTypeError::new_err(e.to_string()))?;

            let r = PythonExpression::from(Atom::var(symbol));
            r.into_py_any(py)
        } else {
            let mut result = vec![];
            for a in names {
                let name = a.extract::<PyBackedStr>()?;
                let name = namespace.attach_namespace(&name);
                let mut symbol = Symbol::new(name).with_attributes(opts.clone());

                if let Some(f) = &custom_normalization {
                    let t = f.chain.clone();
                    symbol = symbol.with_normalization_function(Box::new(
                        move |input: AtomView<'_>, out: &mut Settable<Atom>| {
                            let _ = Workspace::get_local()
                                .with(|ws| {
                                    Transformer::execute_chain(
                                        input,
                                        &t,
                                        ws,
                                        &TransformerState::default(),
                                        &mut *out,
                                    )
                                })
                                .unwrap();
                        },
                    ))
                }

                if let Some(t) = tags.as_ref() {
                    symbol = symbol.with_tags(
                        t.into_iter()
                            .map(|x| {
                                if x.contains("::") {
                                    x.clone()
                                } else {
                                    format!("python::{x}")
                                }
                            })
                            .collect::<Vec<_>>(),
                    );
                }

                let symbol = symbol
                    .build()
                    .map_err(|e| exceptions::PyTypeError::new_err(e.to_string()))?;
                let r = PythonExpression::from(Atom::var(symbol));
                result.push(r);
            }

            result.into_py_any(py)
        }
    }

    /// Create a new Symbolica number from an int, a float, a Decimal, or a string.
    /// A floating point number is kept as a float with the same precision as the input,
    /// but it can also be converted to the smallest rational number given a `relative_error`.
    ///
    /// Examples
    /// --------
    /// >>> e = Expression.num(1) / 2
    /// >>> print(e)
    /// 1/2
    ///
    /// >>> print(Expression.num(1/3))
    /// >>> print(Expression.num(0.33, 0.1))
    /// >>> print(Expression.num('0.333`3'))
    /// >>> print(Expression.num(Decimal('0.1234')))
    /// 3.3333333333333331e-1
    /// 1/3
    /// 3.33e-1
    /// 1.2340e-1
    #[pyo3(signature = (num, relative_error = None))]
    #[classmethod]
    pub fn num(
        _cls: &Bound<'_, PyType>,
        py: Python,
        #[gen_stub(override_type(
            type_repr = "int | float | complex | str | decimal.Decimal",
            imports = ("decimal")
        ))]
        num: Py<PyAny>,
        relative_error: Option<f64>,
    ) -> PyResult<PythonExpression> {
        if let Ok(num) = num.extract::<i64>(py) {
            Ok(Atom::num(num).into())
        } else if let Ok(num) = num.cast_bound::<PyInt>(py) {
            let a = format!("{num}");
            PythonExpression::parse(_cls, &a, PythonParseMode::Symbolica, "python")
        } else if let Ok(f) = num.extract::<PythonMultiPrecisionFloat>(py) {
            if let Some(relative_error) = relative_error {
                let err = relative_error
                    .try_into()
                    .map_err(exceptions::PyValueError::new_err)?;
                let mut r: Rational = f.0.try_into().map_err(exceptions::PyValueError::new_err)?;
                r = r.round(&err);
                Ok(Atom::num(r).into())
            } else {
                Ok(Atom::num(f.0).into())
            }
        } else if let Ok(f) = num.extract::<Complex<f64>>(py) {
            if let Some(relative_error) = relative_error {
                let err = relative_error
                    .try_into()
                    .map_err(exceptions::PyValueError::new_err)?;
                let r = Rational::try_from(f.re)
                    .map_err(exceptions::PyValueError::new_err)?
                    .round(&err);
                let i = Rational::try_from(f.im)
                    .map_err(exceptions::PyValueError::new_err)?
                    .round(&err);
                Ok(Atom::num(Complex::new(r, i)).into())
            } else {
                Ok(Atom::num(Complex::<Float>::new(f.re.into(), f.im.into())).into())
            }
        } else {
            Err(exceptions::PyValueError::new_err("Not a valid number"))
        }
    }

    /// Euler's number `e`.
    #[classattr]
    #[pyo3(name = "E")]
    pub fn e() -> PythonExpression {
        Atom::var(Symbol::E).into()
    }

    /// The mathematical constant `π`.
    #[classattr]
    #[pyo3(name = "PI")]
    pub fn pi() -> PythonExpression {
        Atom::var(Symbol::PI).into()
    }

    /// The mathematical constant `i`, where
    /// `i^2 = -1`.
    #[classattr]
    #[pyo3(name = "I")]
    pub fn i() -> PythonExpression {
        Atom::i().into()
    }

    /// The number that represents infinity: `∞`.
    #[classattr]
    #[pyo3(name = "INFINITY")]
    pub fn inf() -> PythonExpression {
        Atom::num(Coefficient::Infinity(Some(Rational::one().into()))).into()
    }

    /// The number that represents infinity with an unknown complex phase: `⧞`.
    #[classattr]
    #[pyo3(name = "COMPLEX_INFINITY")]
    pub fn cinf() -> PythonExpression {
        Atom::num(Coefficient::Infinity(None)).into()
    }

    /// The number that represents indeterminacy: `¿`.
    #[classattr]
    #[pyo3(name = "INDETERMINATE")]
    pub fn indeterminate() -> PythonExpression {
        Atom::num(Coefficient::Indeterminate).into()
    }

    /// The built-in function that converts a rational polynomial to a coefficient.
    #[classattr]
    #[pyo3(name = "COEFF")]
    pub fn coeff() -> PythonExpression {
        Atom::var(Symbol::COEFF).into()
    }

    /// The built-in cosine function.
    #[classattr]
    #[pyo3(name = "COS")]
    pub fn cos_attr() -> PythonExpression {
        Atom::var(Symbol::COS).into()
    }

    /// The built-in sine function.
    #[classattr]
    #[pyo3(name = "SIN")]
    pub fn sin_attr() -> PythonExpression {
        Atom::var(Symbol::SIN).into()
    }

    /// The built-in exponential function.
    #[classattr]
    #[pyo3(name = "EXP")]
    pub fn exp_attr() -> PythonExpression {
        Atom::var(Symbol::EXP).into()
    }

    /// The built-in logarithm function.
    #[classattr]
    #[pyo3(name = "LOG")]
    pub fn log_attr() -> PythonExpression {
        Atom::var(Symbol::LOG).into()
    }

    /// The built-in square root function.
    #[classattr]
    #[pyo3(name = "SQRT")]
    pub fn sqrt_attr() -> PythonExpression {
        Atom::var(Symbol::SQRT).into()
    }

    /// The built-in complex conjugate function.
    #[classattr]
    #[pyo3(name = "CONJ")]
    pub fn conj_attr() -> PythonExpression {
        Atom::var(Symbol::CONJ).into()
    }

    /// Return all defined symbol names (function names and variables).
    #[classmethod]
    pub fn get_all_symbol_names(_cls: &Bound<'_, PyType>) -> PyResult<Vec<String>> {
        Ok(State::symbol_iter().map(|(_, x)| x.to_string()).collect())
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
    #[pyo3(signature = (input, mode = PythonParseMode::Symbolica, default_namespace = "python"))]
    #[classmethod]
    pub fn parse(
        _cls: &Bound<'_, PyType>,
        input: &str,
        mode: PythonParseMode,
        default_namespace: &str,
    ) -> PyResult<PythonExpression> {
        let e = try_parse!(
            input,
            settings = ParseSettings {
                mode: mode.into(),
                ..ParseSettings::default()
            },
            default_namespace = default_namespace.to_string()
        )
        .map_err(exceptions::PyValueError::new_err)?;
        Ok(e.into())
    }

    /// Create a new expression that represents 0.
    #[new]
    pub fn __new__() -> PythonExpression {
        Atom::new().into()
    }

    /// Construct an expression from a serialized state.
    pub fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        unsafe {
            self.expr = Atom::from_raw(state);
        }
        Ok(())
    }

    /// Get a serialized version of the expression.
    pub fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(self.expr.clone().into_raw())
    }

    /// Get the default positional arguments for `__new__`.
    pub fn __getnewargs__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        Ok(PyTuple::empty(py))
    }

    /// Copy the expression.
    pub fn __copy__(&self) -> PythonExpression {
        self.expr.clone().into()
    }

    /// Convert the expression into a portable string.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(self
            .expr
            .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new()))
    }

    /// Convert the expression into a human-readable string.
    pub fn __str__(&self) -> PyResult<String> {
        Ok(self.expr.format_string(
            &PrintOptions {
                max_terms: Some(100),
                ..DEFAULT_PRINT_OPTIONS
            },
            PrintState::new(),
        ))
    }

    /// Convert the expression into a canonical string that
    /// is independent on the order of the variables and other
    /// implementation details.
    pub fn to_canonical_string(&self) -> PyResult<String> {
        Ok(self.expr.to_canonical_string())
    }

    pub fn __contains__(&self, expr: &PythonExpression) -> bool {
        self.expr.contains(&expr.expr)
    }

    /// Get the number of bytes that this expression takes up in memory.
    pub fn get_byte_size(&self) -> usize {
        self.expr.as_view().get_byte_size()
    }

    /// Convert the expression into a human-readable string, with tunable settings.
    ///
    /// Examples
    /// --------
    /// >>> a = E('128378127123 z^(2/3)*w^2/x/y + y^4 + z^34 + x^(x+2)+3/5+f(x,x^2)')
    /// >>> print(a.format(number_thousands_separator='_', multiplication_operator=' '))
    #[pyo3(signature =
        (mode = PythonPrintMode::Symbolica,
            terms_on_new_line = false,
            color_top_level_sum = true,
            color_builtin_symbols = true,
            print_ring = true,
            symmetric_representation_for_finite_field = false,
            explicit_rational_polynomial = false,
            number_thousands_separator = None,
            multiplication_operator = '*',
            double_star_for_exponentiation = false,
            square_brackets_for_function = false,
            num_exp_as_superscript = true,
            precision = None,
            show_namespaces = false,
            include_attributes = false,
            max_terms = Some(100),
            custom_print_mode = None)
        )]
    pub fn format(
        &self,
        mode: PythonPrintMode,
        terms_on_new_line: bool,
        color_top_level_sum: bool,
        color_builtin_symbols: bool,
        print_ring: bool,
        symmetric_representation_for_finite_field: bool,
        explicit_rational_polynomial: bool,
        number_thousands_separator: Option<char>,
        multiplication_operator: char,
        double_star_for_exponentiation: bool,
        square_brackets_for_function: bool,
        num_exp_as_superscript: bool,
        precision: Option<usize>,
        show_namespaces: bool,
        include_attributes: bool,
        max_terms: Option<usize>,
        custom_print_mode: Option<usize>,
    ) -> PyResult<String> {
        Ok(format!(
            "{}",
            AtomPrinter::new_with_options(
                self.expr.as_view(),
                PrintOptions {
                    terms_on_new_line,
                    color_top_level_sum,
                    color_builtin_symbols,
                    print_ring,
                    symmetric_representation_for_finite_field,
                    explicit_rational_polynomial,
                    number_thousands_separator,
                    multiplication_operator,
                    double_star_for_exponentiation,
                    square_brackets_for_function,
                    num_exp_as_superscript,
                    mode: mode.into(),
                    precision,
                    pretty_matrix: false,
                    hide_all_namespaces: !show_namespaces,
                    color_namespace: true,
                    hide_namespace: Some("python"),
                    include_attributes,
                    max_terms,
                    custom_print_mode: custom_print_mode.map(|x| ("default", x)),
                },
            )
        ))
    }

    /// Convert the expression into a plain string, useful for importing and exporting.
    ///
    /// Examples
    /// --------
    /// >>> a = E('5 + x^2')
    /// >>> print(a.to_plain())
    ///
    /// Yields `5 + x^2`, without any coloring.
    pub fn format_plain(&self) -> PyResult<String> {
        Ok(self
            .expr
            .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new()))
    }

    /// Convert the expression into a LaTeX string.
    ///
    /// Examples
    /// --------
    /// >>> a = E('128378127123 z^(2/3)*w^2/x/y + y^4 + z^34 + x^(x+2)+3/5+f(x,x^2)')
    /// >>> print(a.to_latex())
    ///
    /// Yields `$$z^{34}+x^{x+2}+y^{4}+f(x,x^{2})+128378127123 z^{\\frac{2}{3}} w^{2} \\frac{1}{x} \\frac{1}{y}+\\frac{3}{5}$$`.
    pub fn to_latex(&self) -> PyResult<String> {
        Ok(format!(
            "$${}$$",
            AtomPrinter::new_with_options(self.expr.as_view(), LATEX_PRINT_OPTIONS,)
        ))
    }

    /// Convert the expression into a Sympy-parsable string.
    ///
    /// Examples
    /// --------
    /// >>> from sympy import *
    /// >>> s = sympy.parse_expr(E('x^2+f((1+x)^y)').to_sympy())
    pub fn to_sympy(&self) -> PyResult<String> {
        Ok(format!("{}", self.expr.printer(PrintOptions::sympy())))
    }

    /// Convert the expression into a Mathematica-parsable string.
    ///
    /// Examples
    /// --------
    /// >>> a = E('cos(x+2i + 3)+sqrt(conj(x)) + test::y')
    /// >>> print(a.to_mathematica())
    ///
    /// Yields ```test`y+Cos[x+3+2I]+Sqrt[Conjugate[x]]```.
    #[pyo3(signature = (show_namespaces = true))]
    pub fn to_mathematica(&self, show_namespaces: bool) -> PyResult<String> {
        Ok(format!(
            "{}",
            self.expr.printer(PrintOptions {
                hide_all_namespaces: !show_namespaces,
                hide_namespace: Some("python"),
                ..PrintOptions::mathematica()
            })
        ))
    }

    /// Convert the expression into an integer if possible.
    /// Raises a `ValueError` if the expression cannot be converted to an integer.
    ///
    /// Examples
    /// --------
    /// >>> e = E('7').to_int()
    pub fn to_int(&self) -> PyResult<Integer> {
        self.expr
            .clone()
            .try_into()
            .map_err(|e| exceptions::PyValueError::new_err(format!("Cannot convert to int: {e}")))
    }

    /// Hash the expression.
    pub fn __hash__(&self) -> u64 {
        let mut hasher = ahash::AHasher::default();
        self.expr.hash(&mut hasher);
        hasher.finish()
    }

    /// Save the expression and its state to a binary file.
    /// The data is compressed and the compression level can be set between 0 and 11.
    ///
    /// The expression can be loaded using `Expression.load`.
    ///
    /// Examples
    /// --------
    /// >>> e = E("f(x)+f(y)").expand()
    /// >>> e.save('export.dat')
    #[pyo3(signature = (filename, compression_level=9))]
    pub fn save(&self, filename: &str, compression_level: u32) -> PyResult<()> {
        let f = File::create(filename)
            .map_err(|e| exceptions::PyIOError::new_err(format!("Could not create file: {e}")))?;
        let mut writer = CompressorWriter::new(BufWriter::new(f), 4096, compression_level, 22);

        self.expr
            .as_view()
            .export(&mut writer)
            .map_err(|e| exceptions::PyIOError::new_err(format!("Could not write file: {e}")))
    }

    /// Load an expression and its state from a file. The state will be merged
    /// with the current one. If a symbol has conflicting attributes, the conflict
    /// can be resolved using the renaming function `conflict_fn`.
    ///
    /// Expressions can be saved using `Expression.save`.
    ///
    /// Examples
    /// --------
    /// If `export.dat` contains a serialized expression: `f(x)+f(y)`:
    /// >>> e = Expression.load('export.dat')
    ///
    /// whill yield `f(x)+f(y)`.
    ///
    /// If we have defined symbols in a different order:
    /// >>> y, x = S('y', 'x')
    /// >>> e = Expression.load('export.dat')
    ///
    /// we get `f(y)+f(x)`.
    ///
    /// If we define a symbol with conflicting attributes, we can resolve the conflict
    /// using a renaming function:
    ///
    /// >>> x = S('x', is_symmetric=True)
    /// >>> e = Expression.load('export.dat', lambda x: x + '_new')
    /// print(e)
    ///
    /// will yield `f(x_new)+f(y)`.
    #[pyo3(signature = (filename, conflict_fn=None))]
    #[classmethod]
    pub fn load(
        _cls: &Bound<'_, PyType>,
        filename: &str,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Callable[[str], str]]"))]
        conflict_fn: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let f = File::open(filename)
            .map_err(|e| exceptions::PyIOError::new_err(format!("Could not read file: {e}")))?;
        let mut reader = brotli::Decompressor::new(BufReader::new(f), 4096);

        Atom::import(
            &mut reader,
            match conflict_fn {
                Some(f) => Some(Box::new(move |name: &str| -> SmartString<LazyCompact> {
                    Python::attach(|py| {
                        f.call1(py, (name,)).unwrap().extract::<String>(py).unwrap()
                    })
                    .into()
                })),
                None => None,
            },
        )
        .map(|a| a.into())
        .map_err(|e| exceptions::PyIOError::new_err(format!("Could not read file: {e}")))
    }

    /// Get the type of the atom.
    pub fn get_type(&self) -> PythonAtomType {
        match self.expr.as_ref() {
            Atom::Num(_) => PythonAtomType::Num,
            Atom::Var(_) => PythonAtomType::Var,
            Atom::Fun(_) => PythonAtomType::Fn,
            Atom::Add(_) => PythonAtomType::Add,
            Atom::Mul(_) => PythonAtomType::Mul,
            Atom::Pow(_) => PythonAtomType::Pow,
            Atom::Zero => PythonAtomType::Num,
        }
    }

    /// Convert the expression to a tree.
    pub fn to_atom_tree(&self) -> PyResult<PythonAtomTree> {
        self.expr.as_view().into()
    }

    /// Get the name of a variable or function if the current atom
    /// is a variable or function, otherwise throw an error.
    pub fn get_name(&self) -> PyResult<String> {
        match self.expr.as_ref() {
            Atom::Var(v) => Ok(v.get_symbol().get_name().to_string()),
            Atom::Fun(f) => Ok(f.get_symbol().get_name().to_string()),
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "The expression {} is not a variable or function",
                self.expr
            ))),
        }
    }

    /// Get the attributes of a variable or function if the current atom
    /// is a variable or function, otherwise throw an error.
    pub fn get_attributes(&self) -> PyResult<Vec<PythonSymbolAttribute>> {
        match self.expr.as_ref() {
            Atom::Var(v) => Ok(v
                .get_symbol()
                .get_attributes()
                .into_iter()
                .map(|a| a.into())
                .collect()),
            Atom::Fun(f) => Ok(f
                .get_symbol()
                .get_attributes()
                .into_iter()
                .map(|a| a.into())
                .collect()),
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "The expression {} is not a variable or function",
                self.expr
            ))),
        }
    }

    /// Get the data of a variable or function if the current atom
    /// is a variable or function, otherwise throw an error.
    /// Optionally, provide a key to access a specific entry in the data map, if
    /// the data is a map.
    #[gen_stub(override_return_type(
        type_repr = "Expression | int | float | complex | str | bytes | dict[Expression | int | float | complex | str, Any] | list[Any]"
    ))]
    #[pyo3(signature = (key=None))]
    pub fn get_symbol_data(
        &self,
        #[gen_stub(override_type(type_repr = "Expression | int | float | complex | str"))]
        key: Option<Py<PyAny>>,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let data = match self.expr.as_ref() {
            Atom::Var(v) => v.get_symbol().get_data(),
            Atom::Fun(f) => f.get_symbol().get_data(),
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "The expression {} is not a variable or function",
                self.expr
            )))?,
        };

        if let Some(key) = key
            && let UserData::Map(map) = data
        {
            let key = key.extract::<PythonUserDataKey>(py)?;
            if let Some(value) = map.get(&key.0) {
                return PythonBorrowedUserData(value).into_py_any(py);
            } else {
                return Err(exceptions::PyKeyError::new_err(format!(
                    "The symbol data does not contain the key '{:?}'",
                    key.0
                )));
            }
        } else {
            PythonBorrowedUserData(data).into_py_any(py)
        }
    }

    /// Get the tags of a variable or function if the current atom
    /// is a variable or function, otherwise throw an error.
    pub fn get_tags(&self) -> PyResult<Vec<String>> {
        match self.expr.as_ref() {
            Atom::Var(v) => Ok(v.get_symbol().get_tags().to_vec()),
            Atom::Fun(f) => Ok(f.get_symbol().get_tags().to_vec()),
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "The expression {} is not a variable or function",
                self.expr
            ))),
        }
    }

    /// Check if the expression is a scalar. Symbols must have the scalar attribute.
    ///
    /// Examples
    /// --------
    /// >>> x = S('x', is_scalar=True)
    /// >>> e = (x +1)**2 + 5
    /// >>> print(e.is_scalar())
    /// True
    pub fn is_scalar(&self) -> bool {
        self.expr.is_scalar()
    }

    /// Check if the expression is real. Symbols must have the real attribute.
    ///
    /// Examples
    /// --------
    /// >>> x = S('x', is_real=True)
    /// >>> e = (x + 1)**2 / 2 + 5
    /// >>> print(e.is_real())
    /// True
    pub fn is_real(&self) -> bool {
        self.expr.is_real()
    }

    /// Check if the expression is integer. Symbols must have the integer attribute.
    ///
    /// Examples
    /// --------
    /// >>> x = S('x', is_integer=True)
    /// >>> e = (x + 1)**2 + 5
    /// >>> print(e.is_integer())
    /// True
    pub fn is_integer(&self) -> bool {
        self.expr.is_integer()
    }

    /// Check if the expression is a positive scalar. Symbols must have the positive attribute.
    ///
    /// Examples
    /// --------
    /// >>> x = S('x', is_positive=True)
    /// >>> e = (x + 1)**2 + 5
    /// >>> print(e.is_positive())
    /// True
    pub fn is_positive(&self) -> bool {
        self.expr.is_positive()
    }

    /// Check if the expression has no infinities and is not indeterminate.
    ///
    /// Examples
    /// --------
    /// >>> e = E('x + x^2 + log(0)')
    /// >>> print(e.is_finite())
    /// False
    pub fn is_finite(&self) -> bool {
        self.expr.is_finite()
    }

    /// Check if the expression is constant, i.e. contains no user-defined symbols or functions.
    ///
    /// Examples
    /// --------
    /// >>> e = E('cos(2 + exp(3)) + 5')
    /// >>> print(e.is_constant())
    /// True
    pub fn is_constant(&self) -> bool {
        self.expr.is_constant()
    }

    /// Add this expression to `other`, returning the result.
    pub fn __add__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let rhs = rhs.to_expression();
        Ok((self.expr.as_ref() + rhs.expr.as_ref()).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __radd__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        self.__add__(rhs)
    }

    /// Subtract `other` from this expression, returning the result.
    pub fn __sub__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        self.__add__(ConvertibleToExpression(rhs.to_expression().__neg__()?))
    }

    /// Subtract this expression from `other`, returning the result.
    pub fn __rsub__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        rhs.to_expression()
            .__add__(ConvertibleToExpression(self.__neg__()?))
    }

    /// Add this expression to `other`, returning the result.
    pub fn __mul__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let rhs = rhs.to_expression();
        Ok((self.expr.as_ref() * rhs.expr.as_ref()).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __rmul__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        self.__mul__(rhs)
    }

    /// Divide this expression by `other`, returning the result.
    pub fn __truediv__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let rhs = rhs.to_expression();
        Ok((self.expr.as_ref() / rhs.expr.as_ref()).into())
    }

    /// Divide `other` by this expression, returning the result.
    pub fn __rtruediv__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        rhs.to_expression()
            .__truediv__(ConvertibleToExpression(self.clone()))
    }

    /// Take `self` to power `exp`, returning the result.
    pub fn __pow__(
        &self,
        exponent: ConvertibleToExpression,
        modulo: Option<i64>,
    ) -> PyResult<PythonExpression> {
        if modulo.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "Optional number argument not supported",
            ));
        }

        let exponent = exponent.to_expression();
        Ok(self.expr.pow(&exponent.expr).into())
    }

    /// Take `base` to power `self`, returning the result.
    pub fn __rpow__(
        &self,
        base: ConvertibleToExpression,
        modulo: Option<i64>,
    ) -> PyResult<PythonExpression> {
        base.to_expression()
            .__pow__(ConvertibleToExpression(self.clone()), modulo)
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __xor__(&self, _rhs: Py<PyAny>) -> PyResult<PythonExpression> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor an expression. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __rxor__(&self, _rhs: Py<PyAny>) -> PyResult<PythonExpression> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor an expression. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Negate the current expression, returning the result.
    pub fn __neg__(&self) -> PyResult<PythonExpression> {
        Ok((-self.expr.as_ref()).into())
    }

    /// Return the length of the atom.
    fn __len__(&self) -> usize {
        match self.expr.as_view() {
            AtomView::Add(a) => a.get_nargs(),
            AtomView::Mul(a) => a.get_nargs(),
            AtomView::Fun(a) => a.get_nargs(),
            _ => 1,
        }
    }

    /// Create a Symbolica expression or transformer by calling the function with appropriate arguments.
    ///
    /// Examples
    /// -------
    /// >>> x = S('x')
    /// >>> f = S('f')
    /// >>> e = f(3,x)
    /// >>> print(e)
    /// f(3,x)
    #[gen_stub(skip)]
    #[pyo3(signature = (*args,))]
    pub fn __call__(&self, args: &Bound<'_, PyTuple>, py: Python) -> PyResult<Py<PyAny>> {
        let id = match self.expr.as_view() {
            AtomView::Var(v) => v.get_symbol(),
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only symbols can be called as functions",
                ));
            }
        };

        pub enum ExpressionOrTransformer {
            Expression(PythonExpression),
            Transformer(ConvertibleToPattern),
        }

        let mut fn_args = Vec::with_capacity(args.len());

        for arg in args {
            if let Ok(a) = arg.extract::<ConvertibleToExpression>() {
                fn_args.push(ExpressionOrTransformer::Expression(a.to_expression()));
            } else if let Ok(a) = arg.extract::<ConvertibleToPattern>() {
                fn_args.push(ExpressionOrTransformer::Transformer(a));
            } else {
                let msg = format!("Unknown type: {}", arg.get_type().name().unwrap());
                return Err(exceptions::PyTypeError::new_err(msg));
            }
        }

        if fn_args
            .iter()
            .all(|x| matches!(x, ExpressionOrTransformer::Expression(_)))
        {
            // simplify to literal expression
            Workspace::get_local().with(|workspace| {
                let mut fun_b = workspace.new_atom();
                let fun = fun_b.to_fun(id);

                for x in fn_args {
                    if let ExpressionOrTransformer::Expression(a) = x {
                        fun.add_arg(a.expr.as_view());
                    }
                }

                let mut out = Atom::default();
                fun_b.as_view().normalize(workspace, &mut out);

                PythonExpression::from(out).into_py_any(py)
            })
        } else {
            // convert all wildcards back from literals
            let mut transformer_args = Vec::with_capacity(args.len());
            for arg in fn_args {
                match arg {
                    ExpressionOrTransformer::Transformer(t) => {
                        transformer_args.push(t.to_pattern()?.expr);
                    }
                    ExpressionOrTransformer::Expression(a) => {
                        transformer_args.push(a.expr.to_pattern());
                    }
                }
            }

            let p = Pattern::Fn(id, transformer_args);
            PythonHeldExpression::from(p).into_py_any(py)
        }
    }

    /// Compute the cosine of the expression.
    pub fn cos(&self) -> PythonExpression {
        self.expr.cos().into()
    }

    /// Compute the sine of the expression.
    pub fn sin(&self) -> PythonExpression {
        self.expr.sin().into()
    }

    /// Compute the exponential of the expression.
    pub fn exp(&self) -> PythonExpression {
        self.expr.exp().into()
    }

    /// Compute the natural logarithm of the expression.
    pub fn log(&self) -> PythonExpression {
        self.expr.log().into()
    }

    /// Compute the square root of the expression.
    pub fn sqrt(&self) -> PythonExpression {
        self.expr.sqrt().into()
    }

    /// Take the complex conjugate of this expression, returning the result.
    ///
    /// Examples
    /// --------
    /// >>> e = E('x+2 + 3^x + (5+2i) * (test::{real}::real) + (-2)^x')
    /// >>> print(e.conj())
    ///
    /// Yields `(5-2𝑖)*real+3^conj(x)+conj(x)+conj((-2)^x)+2`.
    pub fn conj(&self) -> PythonExpression {
        self.expr.conj().into()
    }

    /// Create a held expression that delays the execution of the transformer `t` until the
    /// resulting held expression is called. Held expressions can be composed like regular expressions
    /// and are useful for the right-hand side of pattern matching, to act a transformer
    /// on a wildcard *after* it has been substituted.
    ///
    /// Examples
    /// -------
    /// >>> f, x, x_ = S('f', 'x', 'x_')
    /// >>> e = f((x+1)**2)
    /// >>> e = e.replace(f(x_), f(x_.hold(T().expand())))
    pub fn hold(&self, t: PythonTransformer) -> PyResult<PythonHeldExpression> {
        Ok(Pattern::Transformer(Box::new((Some(self.expr.to_pattern()), t.chain))).into())
    }

    /// Get the `idx`th component of the expression.
    fn __getitem__(&self, idx: isize) -> PyResult<PythonExpression> {
        let slice = match self.expr.as_view() {
            AtomView::Add(a) => a.to_slice(),
            AtomView::Mul(m) => m.to_slice(),
            AtomView::Fun(f) => f.to_slice(),
            AtomView::Pow(p) => p.to_slice(),
            _ => Err(PyIndexError::new_err("Cannot access child of leaf node"))?,
        };

        if idx.unsigned_abs() < slice.len() {
            Ok(if idx < 0 {
                slice
                    .get(slice.len() - idx.unsigned_abs())
                    .to_owned()
                    .into()
            } else {
                slice.get(idx as usize).to_owned().into()
            })
        } else {
            Err(PyIndexError::new_err(format!(
                "Index {} out of bounds: the atom only has {} children.",
                idx,
                slice.len(),
            )))
        }
    }

    /// Returns true iff `self` contains `a` literally.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x, y, z = S('x', 'y', 'z')
    /// >>> e = x * y * z
    /// >>> e.contains(x) # True
    /// >>> e.contains(x*y*z) # True
    /// >>> e.contains(x*y) # False
    pub fn contains(&self, s: ConvertibleToOpenPattern) -> PyResult<PythonCondition> {
        Ok(PythonCondition {
            condition: Condition::Yield(Relation::Contains(
                self.expr.to_pattern(),
                s.to_pattern()?.expr,
            )),
        })
    }

    /// Get all symbols in the current expression, optionally including function symbols.
    /// The symbols are sorted in Symbolica's internal ordering.
    #[pyo3(signature = (include_function_symbols = true))]
    pub fn get_all_symbols(&self, include_function_symbols: bool) -> Vec<PythonExpression> {
        let mut s: Vec<PythonExpression> = self
            .expr
            .get_all_symbols(include_function_symbols)
            .into_iter()
            .map(|x| Atom::var(x).into())
            .collect();
        s.sort_by(|x, y| x.expr.cmp(&y.expr));
        s
    }

    /// Get all symbols and functions in the current expression, optionally considering function arguments as well.
    /// The symbols are sorted in Symbolica's internal ordering.
    #[pyo3(signature = (enter_functions = true))]
    pub fn get_all_indeterminates(&self, enter_functions: bool) -> Vec<PythonExpression> {
        let mut s: Vec<PythonExpression> = self
            .expr
            .get_all_indeterminates(enter_functions)
            .into_iter()
            .map(|x| x.to_owned().into())
            .collect();
        s.sort_by(|x, y| x.expr.cmp(&y.expr));
        s
    }

    /// Convert all coefficients to floats with a given precision `decimal_prec`.
    /// The precision of floating point coefficients in the input will be truncated to `decimal_prec`.
    #[pyo3(signature = (decimal_prec = 16))]
    pub fn to_float(&self, decimal_prec: u32) -> PythonExpression {
        self.expr.to_float(decimal_prec).into()
    }

    /// Map all floating point and rational coefficients to the best rational approximation
    /// in the interval `[self*(1-relative_error),self*(1+relative_error)]`.
    #[pyo3(signature = (relative_error = 0.01))]
    pub fn rationalize(&self, relative_error: f64) -> PyResult<PythonExpression> {
        if relative_error <= 0. || relative_error > 1. {
            return Err(exceptions::PyValueError::new_err(
                "Relative error must be between 0 and 1",
            ));
        }

        Ok(self
            .expr
            .rationalize(
                &relative_error
                    .try_into()
                    .map_err(exceptions::PyValueError::new_err)?,
            )
            .into())
    }

    /// Create a pattern restriction based on the wildcard length before downcasting.
    #[pyo3(signature = (min_length, max_length=None))]
    pub fn req_len(
        &self,
        min_length: usize,
        max_length: Option<usize>,
    ) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    condition: (name, WildcardRestriction::Length(min_length, max_length)).into(),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction that tests the type of the atom.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x, x_ = S('x', 'x_')
    /// >>> f = S("f")
    /// >>> e = f(x)*f(2)*f(f(3))
    /// >>> e = e.replace(f(x_), 1, x_.req_type(AtomType.Num))
    /// >>> print(e)
    ///
    /// Yields `f(x)*f(1)`.
    pub fn req_type(&self, atom_type: PythonAtomType) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    condition: (
                        name,
                        WildcardRestriction::IsAtomType(match atom_type {
                            PythonAtomType::Num => AtomType::Num,
                            PythonAtomType::Var => AtomType::Var,
                            PythonAtomType::Add => AtomType::Add,
                            PythonAtomType::Mul => AtomType::Mul,
                            PythonAtomType::Pow => AtomType::Pow,
                            PythonAtomType::Fn => AtomType::Fun,
                        }),
                    )
                        .into(),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction based on the tag of a matched variable or function.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x = S('x', tags=['a', 'b'])
    /// >>> x_ = S('x_')
    /// >>> e = x.replace(x_, 1, x_.req_tag('b'))
    /// >>> print(e)
    /// Yields `1`.
    pub fn req_tag(&self, tag: &str) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                if tag.contains("::") {
                    Ok(PythonPatternRestriction {
                        condition: (name.filter_tag(tag.to_string())).into(),
                    })
                } else {
                    Ok(PythonPatternRestriction {
                        condition: (name.filter_tag(format!("python::{tag}"))).into(),
                    })
                }
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction based on the attribute of a matched variable or function.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x = S('f', is_linear=True)
    /// >>> x_ = S('x_')
    /// >>> print(E('f(x)').replace(E('x_(x)'), 1, ~S('x_').req_attr(SymbolAttribute.Linear)))
    /// >>> print(e)
    ///
    /// Yields `f(x)`.
    pub fn req_attr(&self, attribute: PythonSymbolAttribute) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                let f = move |s: Symbol| match attribute {
                    PythonSymbolAttribute::Symmetric => s.is_symmetric(),
                    PythonSymbolAttribute::Antisymmetric => s.is_antisymmetric(),
                    PythonSymbolAttribute::Cyclesymmetric => s.is_cyclesymmetric(),
                    PythonSymbolAttribute::Linear => s.is_linear(),
                    PythonSymbolAttribute::Scalar => s.is_scalar(),
                    PythonSymbolAttribute::Real => s.is_real(),
                    PythonSymbolAttribute::Integer => s.is_integer(),
                    PythonSymbolAttribute::Positive => s.is_positive(),
                };

                Ok(PythonPatternRestriction {
                    condition: name
                        .filter(move |m| match m {
                            Match::Single(v) => v.get_symbol().map(|s| f(s)).unwrap_or(false),
                            Match::Multiple(_, _) => false,
                            Match::FunctionName(n) => f(*n),
                        })
                        .into(),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction that filters for expressions that contain `a`.
    pub fn req_contains(&self, a: PythonExpression) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    condition: (
                        name,
                        WildcardRestriction::Filter(Box::new(move |m| match m {
                            Match::Single(v) => v.contains(a.expr.as_view()),
                            Match::Multiple(_, v) => v.iter().any(|x| x.contains(a.expr.as_view())),
                            Match::FunctionName(_) => false,
                        })),
                    )
                        .into(),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction that treats the wildcard as a literal variable,
    /// so that it only matches to itself.
    pub fn req_lit(&self) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    condition: (name, WildcardRestriction::IsLiteralWildcard(name)).into(),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Test if the expression is of a certain type.
    pub fn is_type(&self, atom_type: PythonAtomType) -> PythonCondition {
        PythonCondition {
            condition: Condition::Yield(Relation::IsType(
                self.expr.to_pattern(),
                match atom_type {
                    PythonAtomType::Num => AtomType::Num,
                    PythonAtomType::Var => AtomType::Var,
                    PythonAtomType::Add => AtomType::Add,
                    PythonAtomType::Mul => AtomType::Mul,
                    PythonAtomType::Pow => AtomType::Pow,
                    PythonAtomType::Fn => AtomType::Fun,
                },
            )),
        }
    }

    /// Compare two expressions. If one of the expressions is not a number, an
    /// internal ordering will be used.
    fn __richcmp__(&self, o: Py<PyAny>, op: CompareOp, py: Python) -> PyResult<PythonCondition> {
        let Ok(other) = o.extract::<ConvertibleToPattern>(py) else {
            return Err(exceptions::PyTypeError::new_err(format!(
                "Cannot compare {} with {} due to incompatible types.",
                self.expr, o
            )));
        };

        Ok(match op {
            CompareOp::Eq => PythonCondition {
                condition: Relation::Eq(self.expr.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Ne => PythonCondition {
                condition: Relation::Ne(self.expr.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Ge => PythonCondition {
                condition: Relation::Ge(self.expr.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Gt => PythonCondition {
                condition: Relation::Gt(self.expr.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Le => PythonCondition {
                condition: Relation::Le(self.expr.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Lt => PythonCondition {
                condition: Relation::Lt(self.expr.to_pattern(), other.to_pattern()?.expr).into(),
            },
        })
    }

    /// Create a pattern restriction that passes when the wildcard is smaller than a number `num`.
    /// If the matched wildcard is not a number, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace(f(x_), 1, x_.req_lt(2))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_lt(
        &self,
        other: ConvertibleToExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_cmp!(self, other, cmp_any_atom, is_lt)
    }

    /// Create a pattern restriction that passes when the wildcard is greater than a number `num`.
    /// If the matched wildcard is not a number, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace(f(x_), 1, x_.req_gt(2))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_gt(
        &self,
        other: ConvertibleToExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_cmp!(self, other, cmp_any_atom, is_gt)
    }

    /// Create a pattern restriction that passes when the wildcard is smaller than or equal to a number `num`.
    /// If the matched wildcard is not a number, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace(f(x_), 1, x_.req_le(2))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_le(
        &self,
        other: ConvertibleToExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_cmp!(self, other, cmp_any_atom, is_le)
    }

    /// Create a pattern restriction that passes when the wildcard is greater than or equal to a number `num`.
    /// If the matched wildcard is not a number, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace(f(x_), 1, x_.req_ge(2))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_ge(
        &self,
        other: ConvertibleToExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_cmp!(self, other, cmp_any_atom, is_ge)
    }

    /// Create a new pattern restriction that calls the function `filter_fn` with the matched
    /// atom that should return a boolean. If true, the pattern matches.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace(f(x_), 1, x_.req(lambda m: m == 2 or m == 3))
    pub fn req(
        &self,
        #[gen_stub(override_type(type_repr = "typing.Callable[[Expression], bool | Condition]"))]
        filter_fn: Py<PyAny>,
    ) -> PyResult<PythonPatternRestriction> {
        let id = match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }
                name
            }
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only wildcards can be restricted.",
                ));
            }
        };

        Ok(PythonPatternRestriction {
            condition: (
                id,
                WildcardRestriction::Filter(Box::new(move |m| {
                    let data: PythonExpression = m.to_atom().into();

                    Python::attach(|py| {
                        filter_fn
                            .call(py, (data,), None)
                            .expect("Bad callback function")
                            .is_truthy(py)
                            .expect("Pattern filter does not return a boolean")
                    })
                })),
            )
                .into(),
        })
    }

    /// Create a pattern restriction that passes when the wildcard is smaller than another wildcard.
    /// If the matched wildcards are not a numbers, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_, y_ = S('x_', 'y_')
    /// >>> f = S("f")
    /// >>> e = f(1,2)
    /// >>> e = e.replace(f(x_,y_), 1, x_.req_cmp_lt(y_))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_cmp_lt(
        &self,
        other: PythonExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_wc_cmp!(self, other, cmp_any_atom, is_lt)
    }

    /// Create a pattern restriction that passes when the wildcard is greater than another wildcard.
    /// If the matched wildcards are not a numbers, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_, y_ = S('x_', 'y_')
    /// >>> f = S("f")
    /// >>> e = f(2,1)
    /// >>> e = e.replace(f(x_,y_), 1, x_.req_cmp_gt(y_))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_cmp_gt(
        &self,
        other: PythonExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_wc_cmp!(self, other, cmp_any_atom, is_gt)
    }

    /// Create a pattern restriction that passes when the wildcard is less than or equal to another wildcard.
    /// If the matched wildcards are not a numbers, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_, y_ = S('x_', 'y_')
    /// >>> f = S("f")
    /// >>> e = f(1,2)
    /// >>> e = e.replace(f(x_,y_), 1, x_.req_cmp_le(y_))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_cmp_le(
        &self,
        other: PythonExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_wc_cmp!(self, other, cmp_any_atom, is_le)
    }

    /// Create a pattern restriction that passes when the wildcard is greater than or equal to another wildcard.
    /// If the matched wildcards are not a numbers, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_, y_ = S('x_', 'y_')
    /// >>> f = S("f")
    /// >>> e = f(2,1)
    /// >>> e = e.replace(f(x_,y_), 1, x_.req_cmp_ge(y_))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_cmp_ge(
        &self,
        other: PythonExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_wc_cmp!(self, other, cmp_any_atom, is_ge)
    }

    /// Create a new pattern restriction that calls the function `cmp_fn` with another the matched
    /// atom and the match atom of the `other` wildcard that should return a boolean. If true, the pattern matches.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_, y_ = S('x_', 'y_')
    /// >>> f = S("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace(f(x_)*f(y_), 1, x_.req_cmp(y_, lambda m1, m2: m1 + m2 == 4))
    pub fn req_cmp(
        &self,
        other: PythonExpression,
        #[gen_stub(override_type(
            type_repr = "typing.Callable[[Expression, Expression], bool | Condition]"
        ))]
        cmp_fn: Py<PyAny>,
    ) -> PyResult<PythonPatternRestriction> {
        let id = match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }
                name
            }
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only wildcards can be restricted.",
                ));
            }
        };

        let other_id = match other.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }
                name
            }
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only wildcards can be restricted.",
                ));
            }
        };

        Ok(PythonPatternRestriction {
            condition: (
                id,
                WildcardRestriction::Cmp(
                    other_id,
                    Box::new(move |m1, m2| {
                        let data1: PythonExpression = m1.to_atom().into();
                        let data2: PythonExpression = m2.to_atom().into();

                        Python::attach(|py| {
                            cmp_fn
                                .call(py, (data1, data2), None)
                                .expect("Bad callback function")
                                .is_truthy(py)
                                .expect("Pattern comparison does not return a boolean")
                        })
                    }),
                ),
            )
                .into(),
        })
    }

    /// Create an iterator over all atoms in the expression.
    fn __iter__(&self) -> PyResult<PythonAtomIterator> {
        match self.expr.as_view() {
            AtomView::Add(_) | AtomView::Mul(_) | AtomView::Fun(_) | AtomView::Pow(_) => {}
            x => {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Non-iterable type: {}",
                    x
                )));
            }
        };

        Ok(PythonAtomIterator::from_expr(self.clone()))
    }

    /// Map the transformations to every term in the expression.
    /// The execution happens in parallel, using `n_cores`.
    ///
    /// Examples
    /// --------
    /// >>> x, x_ = S('x', 'x_')
    /// >>> e = (1+x)**2
    /// >>> r = e.map(Transformer().expand().replace(x, 6))
    /// >>> print(r)
    #[pyo3(signature = (op, n_cores = None, stats_to_file = None))]
    pub fn map(
        &self,
        op: PythonTransformer,
        py: Python,
        n_cores: Option<usize>,
        stats_to_file: Option<String>,
    ) -> PyResult<PythonExpression> {
        let state = if let Some(stats_to_file) = stats_to_file {
            let file = File::create(stats_to_file).map_err(|e| {
                exceptions::PyIOError::new_err(format!(
                    "Could not create file for transformer statistics: {e}",
                ))
            })?;
            TransformerState {
                stats_export: Some(Arc::new(Mutex::new(BufWriter::new(file)))),
                ..Default::default()
            }
        } else {
            TransformerState::default()
        };

        // release the GIL as Python functions may be called from
        // within the term mapper
        let r = py.detach(move || {
            self.expr.as_view().map_terms(
                |x| {
                    let mut out = Atom::default();
                    Workspace::get_local().with(|ws| {
                        let _ = Transformer::execute_chain(x, &op.chain, ws, &state, &mut out)
                            .unwrap_or_else(|e| {
                                // TODO: capture and abort the parallel run
                                panic!("Transformer failed during parallel execution: {e:?}")
                            });
                    });
                    out
                },
                n_cores.unwrap_or(1),
            )
        });

        Ok(r.into())
    }

    /// Set the coefficient ring to contain the variables in the `vars` list.
    /// This will move all variables into a rational polynomial function.
    ///
    /// Parameters
    /// ----------
    /// vars: List[Expression]
    ///     A list of variables
    pub fn set_coefficient_ring(&self, vars: Vec<PythonExpression>) -> PyResult<PythonExpression> {
        let mut var_map = vec![];
        for v in vars {
            var_map.push(
                v.expr
                    .try_into()
                    .map_err(|e| exceptions::PyValueError::new_err(e))?,
            );
        }

        let b = self.expr.as_view().set_coefficient_ring(&Arc::new(var_map));

        Ok(b.into())
    }

    /// Expand the expression. Optionally, expand in `var` only.
    #[pyo3(signature = (var = None, via_poly = None))]
    pub fn expand(
        &self,
        var: Option<ConvertibleToExpression>,
        via_poly: Option<bool>,
    ) -> PyResult<PythonExpression> {
        if let Some(var) = var {
            let e = var.to_expression();

            if matches!(e.expr, Atom::Var(_) | Atom::Fun(_)) {
                if via_poly.unwrap_or(false) {
                    let b = self
                        .expr
                        .as_view()
                        .expand_via_poly::<i16>(Some(e.expr.as_view()));
                    Ok(b.into())
                } else {
                    let b = self.expr.as_view().expand_in(e.expr.as_view());
                    Ok(b.into())
                }
            } else {
                Err(exceptions::PyValueError::new_err(
                    "Expansion must be done wrt an indeterminate",
                ))
            }
        } else if via_poly.unwrap_or(false) {
            let b = self.expr.as_view().expand_via_poly::<i16>(None);
            Ok(b.into())
        } else {
            let b = self.expr.as_view().expand();
            Ok(b.into())
        }
    }

    /// Distribute numbers in the expression, for example:
    /// `2*(x+y)` -> `2*x+2*y`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x, y = S('x', 'y')
    /// >>> e = 3*(x+y)*(4*x+5*y)
    /// >>> print(e.expand_num())
    ///
    /// yields
    ///
    /// ```log
    /// (3*x+3*y)*(4*x+5*y)
    /// ```
    pub fn expand_num(&self) -> PythonExpression {
        self.expr.expand_num().into()
    }

    /// Collect terms involving the same power of `x`, where `x` is an indeterminate.
    /// Return the list of key-coefficient pairs and the remainder that matched no key.
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x, y = S('x', 'y')
    /// >>> e = 5*x + x * y + x**2 + 5
    /// >>>
    /// >>> print(e.collect(x))
    ///
    /// yields `x^2+x*(y+5)+5`.
    ///
    /// >>> from symbolica import Expression
    /// >>> x, y = S('x', 'y')
    /// >>> exp, coeff = Expression.funs('var', 'coeff')
    /// >>> e = 5*x + x * y + x**2 + 5
    /// >>>
    /// >>> print(e.collect(x, key_map=lambda x: exp(x), coeff_map=lambda x: coeff(x)))
    ///
    /// yields `var(1)*coeff(5)+var(x)*coeff(y+5)+var(x^2)*coeff(1)`.
    #[pyo3(signature = (*x, key_map = None, coeff_map = None))]
    pub fn collect(
        &self,
        x: &Bound<'_, PyTuple>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Callable[[Expression], Expression]]"
        ))]
        key_map: Option<Py<PyAny>>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Callable[[Expression], Expression]]"
        ))]
        coeff_map: Option<Py<PyAny>>,
    ) -> PyResult<PythonExpression> {
        if x.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "No variable or function specified",
            ));
        }

        let mut xs = vec![];
        for a in x {
            if let Ok(r) = a.extract::<PythonExpression>() {
                if matches!(r.expr, Atom::Var(_) | Atom::Fun(_)) {
                    xs.push(r.expr);
                } else {
                    return Err(exceptions::PyValueError::new_err(
                        "Collect must be done wrt a variable or function",
                    ));
                }
            } else {
                return Err(exceptions::PyValueError::new_err(
                    "Collect must be done wrt a variable or function",
                ));
            }
        }

        let b = self.expr.collect_multiple::<i16>(
            &Arc::new(xs),
            if let Some(key_map) = key_map {
                Some(Box::new(move |key, out| {
                    Python::attach(|py| {
                        let key: PythonExpression = key.to_owned().into();

                        out.set_from_view(
                            &key_map
                                .call(py, (key,), None)
                                .expect("Bad callback function")
                                .extract::<PythonExpression>(py)
                                .expect("Key map should return an expression")
                                .expr
                                .as_view(),
                        )
                    });
                }))
            } else {
                None
            },
            if let Some(coeff_map) = coeff_map {
                Some(Box::new(move |coeff, out| {
                    Python::attach(|py| {
                        let coeff: PythonExpression = coeff.to_owned().into();

                        out.set_from_view(
                            &coeff_map
                                .call(py, (coeff,), None)
                                .expect("Bad callback function")
                                .extract::<PythonExpression>(py)
                                .expect("Coeff map should return an expression")
                                .expr
                                .as_view(),
                        )
                    });
                }))
            } else {
                None
            },
        );

        Ok(b.into())
    }

    /// Collect terms involving the same power of variables or functions with the name `x`, e.g.
    ///
    /// ```math
    /// collect_symbol(f(1,2) + x*f*(1,2), f) = (1+x)*f(1,2)
    /// ```
    ///
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x, f = S('x', 'f')
    /// >>> e = f(1,2) + x*f(1,2)
    /// >>>
    /// >>> print(e.collect_symbol(f))
    ///
    /// yields `(1+x)*f(1,2)`.
    #[pyo3(signature = (x, key_map = None, coeff_map = None))]
    pub fn collect_symbol(
        &self,
        x: PythonExpression,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Callable[[Expression], Expression]]"
        ))]
        key_map: Option<Py<PyAny>>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Callable[[Expression], Expression]]"
        ))]
        coeff_map: Option<Py<PyAny>>,
    ) -> PyResult<PythonExpression> {
        let Some(x) = x.expr.get_symbol() else {
            return Err(exceptions::PyValueError::new_err(
                "Collect must be done wrt a variable or function",
            ));
        };

        let b = self.expr.collect_symbol::<i16>(
            x,
            if let Some(key_map) = key_map {
                Some(Box::new(move |key, out| {
                    Python::attach(|py| {
                        let key: PythonExpression = key.to_owned().into();

                        out.set_from_view(
                            &key_map
                                .call(py, (key,), None)
                                .expect("Bad callback function")
                                .extract::<PythonExpression>(py)
                                .expect("Key map should return an expression")
                                .expr
                                .as_view(),
                        )
                    });
                }))
            } else {
                None
            },
            if let Some(coeff_map) = coeff_map {
                Some(Box::new(move |coeff, out| {
                    Python::attach(|py| {
                        let coeff: PythonExpression = coeff.to_owned().into();

                        out.set_from_view(
                            &coeff_map
                                .call(py, (coeff,), None)
                                .expect("Bad callback function")
                                .extract::<PythonExpression>(py)
                                .expect("Coeff map should return an expression")
                                .expr
                                .as_view(),
                        )
                    });
                }))
            } else {
                None
            },
        );

        Ok(b.into())
    }

    /// Collect common factors from (nested) sums.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> e = E('x*(x+y*x+x^2+y*(x+x^2))')
    /// >>> e.collect_factors()
    ///
    /// yields
    ///
    /// ```log
    /// v1^2*(1+v1+v2+v2*(1+v1))
    /// ```
    pub fn collect_factors(&self) -> PythonExpression {
        self.expr.collect_factors().into()
    }

    /// Collect numerical factors by removing the numerical content from additions.
    /// For example, `-2*x + 4*x^2 + 6*x^3` will be transformed into `-2*(x - 2*x^2 - 3*x^3)`.
    ///
    /// The first argument of the addition is normalized to a positive quantity.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>>
    /// >>> x, y = S('x', 'y')
    /// >>> e = (-3*x+6*y)(2*x+2*y)
    /// >>> print(e.collect_num())
    ///
    /// yields
    ///
    /// ```log
    /// -6*(x-2*y)*(x+y)
    /// ```
    pub fn collect_num(&self) -> PythonExpression {
        self.expr.collect_num().into()
    }

    /// Collect terms involving the literal occurrence of `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>>
    /// >>> x, y = S('x', 'y')
    /// >>> e = 5*x + x * y + x**2 + y*x**2
    /// >>> print(e.coefficient(x**2))
    ///
    /// yields
    ///
    /// ```log
    /// y + 1
    /// ```
    pub fn coefficient(&self, x: ConvertibleToExpression) -> PythonExpression {
        let r = self.expr.coefficient(x.to_expression().expr.as_view());
        r.into()
    }

    /// Collect terms involving the same power of `x`, where `x` is an indeterminate.
    /// Return the list of key-coefficient pairs and the remainder that matched no key.
    ///
    /// Examples
    /// --------
    ///
    /// from symbolica import Expression
    /// >>>
    /// >>> x, y = S('x', 'y')
    /// >>> e = 5*x + x * y + x**2 + 5
    /// >>>
    /// >>> for a in e.coefficient_list(x):
    /// >>>     print(a[0], a[1])
    ///
    /// yields
    ///
    /// ```log
    /// x y+5
    /// x^2 1
    /// 1 5
    /// ```
    #[pyo3(signature = (*x,))]
    pub fn coefficient_list(
        &self,
        x: Bound<'_, PyTuple>,
    ) -> PyResult<Vec<(PythonExpression, PythonExpression)>> {
        if x.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "No variable or function specified",
            ));
        }

        let mut xs = vec![];
        for a in x {
            if let Ok(r) = a.extract::<PythonExpression>() {
                if matches!(r.expr, Atom::Var(_) | Atom::Fun(_)) {
                    xs.push(r.expr);
                } else {
                    return Err(exceptions::PyValueError::new_err(
                        "Collect must be done wrt a variable or function",
                    ));
                }
            } else {
                return Err(exceptions::PyValueError::new_err(
                    "Collect must be done wrt a variable or function",
                ));
            }
        }

        let list = self.expr.coefficient_list::<i16>(&xs);

        let py_list: Vec<_> = list
            .into_iter()
            .map(|e| (e.0.to_owned().into(), e.1.into()))
            .collect();

        Ok(py_list)
    }

    /// Derive the expression w.r.t the variable `x`.
    pub fn derivative(&self, x: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let id = if let AtomView::Var(x) = x.to_expression().expr.as_view() {
            x.get_symbol()
        } else {
            return Err(exceptions::PyValueError::new_err(
                "Derivative must be taken wrt a variable",
            ));
        };

        let b = self.expr.derivative(id);

        Ok(b.into())
    }

    /// Series expand in `x` around `expansion_point` to depth `depth`.
    ///
    /// Examples
    /// -------
    /// >>> from symbolica import Expression
    /// >>> x, y = S('x', 'y')
    /// >>> f = S('f')
    /// >>>
    /// >>> e = 2* x**2 * y + f(x)
    /// >>> e = e.series(x, 0, 2)
    /// >>>
    /// >>> print(e)
    ///
    /// yields `f(0)+x*der(1,f(0))+1/2*x^2*(der(2,f(0))+4*y)`.
    #[pyo3(signature = (x, expansion_point, depth, depth_denom = 1, depth_is_absolute = true))]
    pub fn series(
        &self,
        x: PythonExpression,
        expansion_point: ConvertibleToExpression,
        depth: i64,
        depth_denom: i64,
        depth_is_absolute: bool,
    ) -> PyResult<PythonSeries> {
        let id: crate::atom::Indeterminate = x.expr.try_into().map_err(|_| {
            exceptions::PyValueError::new_err(format!(
                "Series expansion must be done wrt a variable"
            ))
        })?;

        match self.expr.series(
            id,
            expansion_point.to_expression().expr.as_view(),
            (depth, depth_denom).into(),
            depth_is_absolute,
        ) {
            Ok(s) => Ok(PythonSeries { series: s }),
            Err(e) => Err(exceptions::PyValueError::new_err(e.to_string())),
        }
    }

    /// Compute the partial fraction decomposition in `x`.
    ///
    /// If `None` is passed, the expression will be decomposed in all variables
    /// which involves a potentially expensive Groebner basis computation.
    ///
    /// Examples
    /// --------
    ///
    /// >>> p = E('1/((x+y)*(x^2+x*y+1)(x+1))')
    /// >>> print(p.apart(S('x')))
    ///
    /// Multivariate partial fractioning
    /// >>> p = E('(2y-x)/(y*(x+y)*(y-x))')
    /// >>> print(p.apart())
    #[pyo3(signature = (x = None))]
    pub fn apart(&self, x: Option<PythonExpression>) -> PyResult<PythonExpression> {
        if let Some(x) = x {
            if let Some(r) = x.expr.get_symbol() {
                Ok(self.expr.apart(r).into())
            } else {
                Err(exceptions::PyValueError::new_err(
                    "Partial fraction decomposition must be done wrt a symbol",
                ))
            }
        } else {
            Ok(self.apart_multivariate().into())
        }
    }

    /// Write the expression over a common denominator.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('v1^2/2+v1^3/v4*v2+v3/(1+v4)')
    /// >>> print(p.together())
    pub fn together(&self) -> PyResult<PythonExpression> {
        Ok(self.expr.together().into())
    }

    /// Cancel common factors between numerators and denominators.
    /// Any non-canceling parts of the expression will not be rewritten.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('1+(y+1)^10*(x+1)/(x^2+2x+1)')
    /// >>> print(p.cancel())
    /// 1+(y+1)**10/(x+1)
    pub fn cancel(&self) -> PyResult<PythonExpression> {
        Ok(self.expr.cancel().into())
    }

    /// Factor the expression over the rationals.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('(6 + x)/(7776 + 6480*x + 2160*x^2 + 360*x^3 + 30*x^4 + x^5)')
    /// >>> print(p.factor())
    /// (x+6)**-4
    pub fn factor(&self) -> PyResult<PythonExpression> {
        Ok(self.expr.factor().into())
    }

    /// Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`.
    /// All non-polynomial elements will be converted to new independent variables.
    ///
    /// If a `modulus` is provided, the coefficients will be converted to finite field elements mod `modulus`.
    /// If on top a `power` is provided, for example `(2, a)`, the polynomial will be converted to the Galois field
    /// `GF(modulus^2)` where `a` is the variable of the minimal polynomial of the field.
    ///
    /// If a `minimal_poly` is provided, the polynomial will be converted to a number field with the given minimal polynomial.
    /// The minimal polynomial must be a monic, irreducible univariate polynomial. If a `modulus` is provided as well,
    /// the Galois field will be created with `minimal_poly` as the minimal polynomial.
    #[gen_stub(skip)]
    #[pyo3(signature = (modulus = None, power = None, minimal_poly = None, vars = None))]
    pub fn to_polynomial(
        &self,
        modulus: Option<u64>,
        mut power: Option<(u16, Symbol)>,
        minimal_poly: Option<PythonPolynomial>,
        vars: Option<Vec<PythonExpression>>,
        py: Python,
    ) -> PyResult<Py<PyAny>> {
        let mut var_map = vec![];
        if let Some(vm) = vars {
            for v in vm {
                var_map.push(
                    v.expr
                        .try_into()
                        .map_err(|e| exceptions::PyValueError::new_err(e))?,
                );
            }
        }

        let var_map = if var_map.is_empty() {
            None
        } else {
            Some(Arc::new(var_map))
        };

        if power.is_some() && modulus.is_none() {
            return Err(exceptions::PyValueError::new_err(
                "Extension field requires a modulus to be set",
            ));
        }

        let poly = minimal_poly.map(|x| x.poly);
        if let Some(p) = &poly {
            if p.nvars() != 1 {
                return Err(exceptions::PyValueError::new_err(
                    "Minimal polynomial must be a univariate polynomial",
                ));
            }

            if power.is_none() {
                if let PolyVariable::Symbol(name) = p.get_vars_ref()[0] {
                    power = Some((p.degree(0) as u16, name));
                } else {
                    return Err(exceptions::PyValueError::new_err(format!(
                        "Extension field polynomial {p} must have a symbol as a variable"
                    )));
                }
            }
        }

        if let Some(m) = modulus {
            if let Some((e, name)) = power {
                if let Some(p) = &poly {
                    if e != p.degree(0) {
                        return Err(exceptions::PyValueError::new_err(
                            "Extension field degree must match the minimal polynomial degree",
                        ));
                    }

                    if PolyVariable::Symbol(name) != p.get_vars_ref()[0] {
                        return Err(exceptions::PyValueError::new_err(
                            "Extension variable must be the same as the variable in the minimal polynomial",
                        ));
                    }

                    if m == 2 {
                        let p = p.map_coeff(|c| c.to_finite_field(&Z2), Z2);
                        if !p.is_irreducible() || e != p.degree(0) {
                            return Err(exceptions::PyValueError::new_err(
                                "Minimal polynomial must be irreducible and monic",
                            ));
                        }

                        let g = AlgebraicExtension::new(p);
                        PythonGaloisFieldPrimeTwoPolynomial {
                            poly: self
                                .expr
                                .try_to_polynomial(&Z2, var_map)
                                .map_err(|e| exceptions::PyValueError::new_err(e))?
                                .to_number_field(&g),
                        }
                        .into_py_any(py)
                    } else {
                        let f = Zp64::new(m);
                        let p = p.map_coeff(|c| c.to_finite_field(&f), f.clone());
                        if !p.is_irreducible() || !f.is_one(&p.lcoeff()) || e != p.degree(0) {
                            return Err(exceptions::PyValueError::new_err(
                                "Minimal polynomial must be irreducible and monic",
                            ));
                        }

                        let g = AlgebraicExtension::new(p);
                        PythonGaloisFieldPolynomial {
                            poly: self
                                .expr
                                .try_to_polynomial(&f, var_map)
                                .map_err(|e| exceptions::PyValueError::new_err(e))?
                                .to_number_field(&g),
                        }
                        .into_py_any(py)
                    }
                } else if m == 2 {
                    let g = AlgebraicExtension::galois_field(Z2, e as usize, name.into());
                    PythonGaloisFieldPrimeTwoPolynomial {
                        poly: self
                            .expr
                            .try_to_polynomial(&Z2, var_map)
                            .map_err(|e| exceptions::PyValueError::new_err(e))?
                            .to_number_field(&g),
                    }
                    .into_py_any(py)
                } else {
                    let f = Zp64::new(m);
                    let g = AlgebraicExtension::galois_field(Zp64::new(m), e as usize, name.into());
                    PythonGaloisFieldPolynomial {
                        poly: self
                            .expr
                            .try_to_polynomial(&f, var_map)
                            .map_err(|e| exceptions::PyValueError::new_err(e))?
                            .to_number_field(&g),
                    }
                    .into_py_any(py)
                }
            } else if m == 2 {
                PythonPrimeTwoPolynomial {
                    poly: self
                        .expr
                        .try_to_polynomial(&Z2, var_map)
                        .map_err(|e| exceptions::PyValueError::new_err(e))?,
                }
                .into_py_any(py)
            } else {
                PythonFiniteFieldPolynomial {
                    poly: self
                        .expr
                        .try_to_polynomial(&Zp64::new(m), var_map)
                        .map_err(|e| exceptions::PyValueError::new_err(e))?,
                }
                .into_py_any(py)
            }
            // FIXME: ignoring minimal poly!
        } else if let Some(p) = poly {
            if !p.is_irreducible() || !p.lcoeff().is_one() {
                return Err(exceptions::PyValueError::new_err(
                    "Minimal polynomial must be irreducible and monic",
                ));
            }

            let f = AlgebraicExtension::new(p);
            if f.poly().exponents == [0, 2] && f.poly().get_constant() == Rational::one() {
                // convert complex coefficients
                PythonNumberFieldPolynomial {
                    poly: self
                        .expr
                        .try_to_polynomial(&f, var_map)
                        .map_err(|e| exceptions::PyValueError::new_err(e))?,
                }
                .into_py_any(py)
            } else {
                PythonNumberFieldPolynomial {
                    poly: self
                        .expr
                        .try_to_polynomial(&Q, var_map)
                        .map_err(|e| exceptions::PyValueError::new_err(e))?
                        .to_number_field(&f),
                }
                .into_py_any(py)
            }
        } else {
            PythonPolynomial {
                poly: self
                    .expr
                    .try_to_polynomial(&Q, var_map)
                    .map_err(|e| exceptions::PyValueError::new_err(e))?,
            }
            .into_py_any(py)
        }
    }

    /// Convert the expression to a rational polynomial, optionally, with the variable ordering specified in `vars`.
    /// The latter is useful if it is known in advance that more variables may be added in the future to the
    /// rational polynomial through composition with other rational polynomials.
    ///
    /// All non-rational polynomial parts will automatically be converted to new independent variables.
    ///
    /// Examples
    /// --------
    /// >>> a = E('(1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 + 15*x7)^2 - 1').to_rational_polynomial()
    /// >>> print(a)
    #[pyo3(signature = (vars = None))]
    pub fn to_rational_polynomial(
        &self,
        vars: Option<Vec<PythonExpression>>,
    ) -> PyResult<PythonRationalPolynomial> {
        let mut var_map = vec![];
        if let Some(vm) = vars {
            for v in vm {
                var_map.push(
                    v.expr
                        .try_into()
                        .map_err(|e| exceptions::PyValueError::new_err(e))?,
                );
            }
        }

        let var_map = if var_map.is_empty() {
            None
        } else {
            Some(Arc::new(var_map))
        };

        let poly = self
            .expr
            .try_to_rational_polynomial(&Q, &Z, var_map)
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!(
                    "Could not convert expression to rational polynomial: {e}",
                ))
            })?;

        Ok(PythonRationalPolynomial { poly })
    }

    /// Return an iterator over the pattern `self` matching to `lhs`.
    /// Restrictions on the pattern can be supplied through `cond`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> x, x_ = S('x','x_')
    /// >>> f = S('f')
    /// >>> e = f(x)*f(1)*f(2)*f(3)
    /// >>> for match in e.match(f(x_)):
    /// >>>    for map in match:
    /// >>>        print(map[0],'=', map[1])
    #[pyo3(name = "match", signature = (lhs, cond = None, level_range = None, level_is_tree_depth = None, allow_new_wildcards_on_rhs = None))]
    pub fn pattern_match(
        &self,
        lhs: ConvertibleToExpression,
        cond: Option<ConvertibleToPatternRestriction>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: Option<bool>,
        allow_new_wildcards_on_rhs: Option<bool>,
    ) -> PyResult<PythonMatchIterator> {
        let conditions = cond.map(|r| r.0).unwrap_or_default();
        let settings = MatchSettings {
            level_range: level_range.unwrap_or((0, None)),
            level_is_tree_depth: level_is_tree_depth.unwrap_or(false),
            allow_new_wildcards_on_rhs: allow_new_wildcards_on_rhs.unwrap_or(false),
            ..MatchSettings::default()
        };
        Ok(PythonMatchIterator::new(
            (
                lhs.to_expression().expr.to_pattern(),
                self.expr.clone(),
                conditions,
                settings,
            ),
            move |(lhs, target, res, settings)| {
                PatternAtomTreeIterator::new(lhs, target.as_view(), Some(res), Some(settings))
            },
        ))
    }

    /// Test whether the pattern is found in the expression.
    /// Restrictions on the pattern can be supplied through `cond`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> f = S('f')
    /// >>> if f(1).matches(f(2)):
    /// >>>    print('match')
    #[pyo3(signature = (lhs, cond = None, level_range = None, level_is_tree_depth = None, allow_new_wildcards_on_rhs = None))]
    pub fn matches(
        &self,
        lhs: ConvertibleToExpression,
        cond: Option<ConvertibleToPatternRestriction>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: Option<bool>,
        allow_new_wildcards_on_rhs: Option<bool>,
    ) -> PyResult<PythonCondition> {
        let conditions = cond.map(|r| r.0).unwrap_or_default();
        let settings = MatchSettings {
            level_range: level_range.unwrap_or((0, None)),
            level_is_tree_depth: level_is_tree_depth.unwrap_or(false),
            allow_new_wildcards_on_rhs: allow_new_wildcards_on_rhs.unwrap_or(false),
            ..MatchSettings::default()
        };

        Ok(PythonCondition {
            condition: Condition::Yield(Relation::Matches(
                self.expr.to_pattern(),
                lhs.to_expression().expr.to_pattern(),
                conditions,
                settings,
            )),
        })
    }

    /// Return an iterator over the replacement of the pattern `self` on `lhs` by `rhs`.
    /// Restrictions on pattern can be supplied through `cond`.
    ///
    /// The `level_range` specifies the `[min,max]` level at which the pattern is allowed to match.
    /// The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree,
    /// depending on `level_is_tree_depth`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S('f')
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> for r in e.replace(f(x_), f(x_ + 1)):
    /// >>>     print(r)
    ///
    /// Yields:
    /// ```log
    /// f(2)*f(2)*f(3)
    /// f(1)*f(3)*f(3)
    /// f(1)*f(2)*f(4)
    /// ```
    #[pyo3(signature = (lhs, rhs, cond = None, level_range = None, level_is_tree_depth = None, allow_new_wildcards_on_rhs = None))]
    pub fn replace_iter(
        &self,
        lhs: ConvertibleToExpression,
        rhs: ConvertibleToReplaceWith,
        cond: Option<ConvertibleToPatternRestriction>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: Option<bool>,
        allow_new_wildcards_on_rhs: Option<bool>,
    ) -> PyResult<PythonReplaceIterator> {
        let conditions = cond.map(|r| r.0.clone()).unwrap_or_default();
        let settings = MatchSettings {
            level_range: level_range.unwrap_or((0, None)),
            level_is_tree_depth: level_is_tree_depth.unwrap_or(false),
            allow_new_wildcards_on_rhs: allow_new_wildcards_on_rhs.unwrap_or(false),
            ..MatchSettings::default()
        };

        Ok(PythonReplaceIterator::new(
            (
                lhs.to_expression().expr.to_pattern(),
                self.expr.clone(),
                rhs.to_replace_with()?,
                conditions,
                settings,
            ),
            move |(lhs, target, rhs, res, settings)| {
                ReplaceIterator::new(
                    lhs,
                    target.as_view(),
                    rhs.clone(),
                    Some(res),
                    Some(settings),
                )
            },
        ))
    }

    /// Replace all atoms matching the pattern `pattern` by the right-hand side `rhs`.
    /// Restrictions on pattern can be supplied through `cond`.
    ///
    /// The `level_range` specifies the `[min,max]` level at which the pattern is allowed to match.
    /// The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree,
    /// depending on `level_is_tree_depth`.
    ///
    /// The entire operation can be repeated until there are no more matches using `repeat=True`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> x, w1_, w2_ = S('x','w1_','w2_')
    /// >>> f = S('f')
    /// >>> e = f(3,x)
    /// >>> r = e.replace(f(w1_,w2_), f(w1_ - 1, w2_**2), (w1_ >= 1) & w2_.is_var())
    /// >>> print(r)
    ///
    /// Parameters
    /// ----------
    /// pattern: Transformer | Expression | int
    ///     The pattern to match.
    /// rhs: Transformer | Expression | int
    ///     The right-hand side to replace the matched subexpression with. Can be a transformer, expression or a function that maps a dictionary of wildcards to an expression.
    /// cond: Optional[PatternRestriction]
    ///     Conditions on the pattern.
    /// level_range: (int, int), optional
    ///     Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
    /// level_is_tree_depth: bool, optional
    ///     If set to `True`, the level is increased when going one level deeper in the expression tree.
    /// allow_new_wildcards_on_rhs: bool, optional
    ///     If set to `True`, wildcards that do not appear ion the pattern are allowed on the right-hand side.
    /// rhs_cache_size: int, optional
    ///      Cache the first `rhs_cache_size` substituted patterns. If set to `None`, an internally determined cache size is used.
    ///      Warning: caching should be disabled (`rhs_cache_size=0`) if the right-hand side contains side effects, such as updating a global variable.
    /// repeat: bool, optional
    ///     If set to `True`, the entire operation will be repeated until there are no more matches.
    #[pyo3(signature = (pattern, rhs, cond = None, non_greedy_wildcards = None, level_range = None, level_is_tree_depth = None, allow_new_wildcards_on_rhs = None, rhs_cache_size = None, repeat = None))]
    pub fn replace(
        &self,
        pattern: ConvertibleToExpression,
        rhs: ConvertibleToReplaceWith,
        cond: Option<ConvertibleToPatternRestriction>,
        non_greedy_wildcards: Option<Vec<PythonExpression>>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: Option<bool>,
        allow_new_wildcards_on_rhs: Option<bool>,
        rhs_cache_size: Option<usize>,
        repeat: Option<bool>,
    ) -> PyResult<PythonExpression> {
        let pattern = pattern.to_expression().expr.to_pattern();
        let rhs = &rhs.to_replace_with()?;

        let mut settings = MatchSettings::cached();

        if let Some(ngw) = non_greedy_wildcards {
            settings.non_greedy_wildcards = ngw
                .iter()
                .map(|x| match x.expr.as_view() {
                    AtomView::Var(v) => {
                        let name = v.get_symbol();
                        if v.get_wildcard_level() == 0 {
                            return Err(exceptions::PyTypeError::new_err(
                                "Only wildcards can be restricted.",
                            ));
                        }
                        Ok(name)
                    }
                    _ => Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    )),
                })
                .collect::<Result<_, _>>()?;
        }
        if let Some(level_range) = level_range {
            settings.level_range = level_range;
        }
        if let Some(level_is_tree_depth) = level_is_tree_depth {
            settings.level_is_tree_depth = level_is_tree_depth;
        }
        if let Some(allow_new_wildcards_on_rhs) = allow_new_wildcards_on_rhs {
            settings.allow_new_wildcards_on_rhs = allow_new_wildcards_on_rhs;
        }
        if let Some(rhs_cache_size) = rhs_cache_size {
            settings.rhs_cache_size = rhs_cache_size;
        }

        let mut expr_ref = self.expr.as_view();

        let cond = cond.map(|r| r.0);

        let mut out = RecycledAtom::new();
        let mut out2 = RecycledAtom::new();
        while expr_ref.replace_into(&pattern, rhs, cond.as_ref(), Some(&settings), &mut out) {
            if !repeat.unwrap_or(false) {
                break;
            }

            std::mem::swap(&mut out, &mut out2);
            expr_ref = out2.as_view();
        }

        Ok(out.into_inner().into())
    }

    /// Replace all atoms matching the patterns. See `replace` for more information.
    ///
    /// The entire operation can be repeated until there are no more matches using `repeat=True`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> x, y, f = S('x', 'y', 'f')
    /// >>> e = f(x,y)
    /// >>> r = e.replace_multiple([Replacement(x, y), Replacement(y, x)])
    /// >>> print(r)
    /// f(y,x)
    ///
    /// Parameters
    /// ----------
    /// replacements: Sequence[Replacement]
    ///     The list of replacements to apply.
    /// repeat: bool, optional
    ///     If set to `True`, the entire operation will be repeated until there are no more matches.
    #[pyo3(signature = (replacements, repeat = None))]
    pub fn replace_multiple(
        &self,
        replacements: Vec<PythonReplacement>,
        repeat: Option<bool>,
    ) -> PyResult<PythonExpression> {
        let reps = replacements
            .iter()
            .map(|x| x.replacement.borrow())
            .collect::<Vec<_>>();

        let mut expr_ref = self.expr.as_view();

        let mut out = RecycledAtom::new();
        let mut out2 = RecycledAtom::new();
        while expr_ref.replace_multiple_into(&reps, &mut out) {
            if !repeat.unwrap_or(false) {
                break;
            }

            std::mem::swap(&mut out, &mut out2);
            expr_ref = out2.as_view();
        }

        Ok(out.into_inner().into())
    }

    /// Replace all wildcards in the expression with the given replacements.
    pub fn replace_wildcards(
        &self,
        replacements: HashMap<PythonExpression, PythonExpression>,
    ) -> PyResult<PythonExpression> {
        let mut reps = HashMap::default();
        for (k, v) in replacements {
            let k = k.expr.as_view();
            let s = if let AtomView::Var(v) = k {
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be replaced.",
                    ));
                }
                v.get_symbol()
            } else {
                return Err(exceptions::PyTypeError::new_err(
                    "Only wildcards can be replaced.",
                ));
            };
            reps.insert(s, v.expr);
        }

        let res = self.expr.to_pattern().replace_wildcards(&reps);

        Ok(res.into())
    }

    /// Solve a linear system in the variables `variables`, where each expression
    /// in the system is understood to yield 0.
    ///
    /// If the system is underdetermined, a partial solution is returned
    /// where each bound variable is a linear combination of the free
    /// variables. The free variables are chosen such that they have the highest index in the `vars` list.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x, y, c = S('x', 'y', 'c')
    /// >>> f = S('f')
    /// >>> x_r, y_r = Expression.solve_linear_system([f(c)*x + y/c - 1, y-c/2], [x, y])
    /// >>> print('x =', x_r, ', y =', y_r)
    #[pyo3(signature = (system, variables, warn_if_underdetermined = true))]
    #[classmethod]
    pub fn solve_linear_system(
        _cls: &Bound<'_, PyType>,
        system: Vec<ConvertibleToExpression>,
        variables: Vec<PythonExpression>,
        warn_if_underdetermined: bool,
    ) -> PyResult<Vec<PythonExpression>> {
        let system: Vec<_> = system.into_iter().map(|x| x.to_expression().expr).collect();
        let vars: Vec<_> = variables.into_iter().map(|v| v.expr).collect();

        match AtomView::solve_linear_system::<u16, _, Atom>(&system, &vars) {
            Ok(res) => Ok(res.into_iter().map(|x| x.into()).collect()),
            Err(SolveError::Underdetermined {
                rank,
                partial_solution,
            }) => {
                if warn_if_underdetermined {
                    warn!(
                        "The system is underdetermined (rank {rank} < size {})",
                        vars.len()
                    );
                }

                Ok(partial_solution.into_iter().map(|x| x.into()).collect())
            }
            Err(SolveError::Other(e)) => Err(exceptions::PyValueError::new_err(e)),
        }
    }

    /// Find the root of an expression in `x` numerically over the reals using Newton's method.
    /// Use `init` as the initial guess for the root.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x, y, c = S('x', 'y', 'c')
    /// >>> f = S('f')
    /// >>> x_r, y_r = Expression.solve_linear_system([f(c)*x + y/c - 1, y-c/2], [x, y])
    /// >>> print('x =', x_r, ', y =', y_r)
    #[gen_stub(override_return_type(type_repr = "decimal.Decimal", imports = ("decimal")))]
    #[pyo3(signature =
        (variable,
        init,
        prec = 1e-4,
        max_iterations = 1000),
        )]
    pub fn nsolve(
        &self,
        variable: PythonExpression,
        init: PythonMultiPrecisionFloat,
        prec: f64,
        max_iterations: usize,
        py: Python,
    ) -> PyResult<Py<PyAny>> {
        let id: crate::atom::Indeterminate = variable.expr.try_into().map_err(|_| {
            exceptions::PyValueError::new_err(format!("Solve must be done wrt a variable"))
        })?;

        if init.0.prec() == 53 {
            let r = self
                .expr
                .nsolve::<F64, _>(id, init.0.to_f64().into(), prec.into(), max_iterations)
                .map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Could not solve system: {e}"))
                })?;
            r.into_inner().into_py_any(py)
        } else {
            PythonMultiPrecisionFloat(
                self.expr
                    .nsolve(id, init.0, prec.into(), max_iterations)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Could not solve system: {e}"))
                    })?,
            )
            .into_py_any(py)
        }
    }

    /// Find a common root of multiple expressions in `variables` numerically over the reals using Newton's method.
    /// Use `init` as the initial guess for the root.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x, y, c = S('x', 'y', 'c')
    /// >>> f = S('f')
    /// >>> x_r, y_r = Expression.solve_linear_system([f(c)*x + y/c - 1, y-c/2], [x, y])
    /// >>> print('x =', x_r, ', y =', y_r)
    #[gen_stub(override_return_type(type_repr = "decimal.Decimal", imports = ("decimal")))]
    #[pyo3(signature =
        (system,
        variables,
        init,
        prec = 1e-4,
        max_iterations = 1000),
        )]
    #[classmethod]
    pub fn nsolve_system(
        _cls: &Bound<'_, PyType>,
        system: Vec<ConvertibleToExpression>,
        variables: Vec<PythonExpression>,
        init: Vec<PythonMultiPrecisionFloat>,
        prec: f64,
        max_iterations: usize,
        py: Python,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let system: Vec<_> = system.into_iter().map(|x| x.to_expression()).collect();
        let system_b: Vec<_> = system.iter().map(|x| x.expr.as_view()).collect();

        let mut vars = vec![];
        for v in variables {
            let id: crate::atom::Indeterminate = v.expr.try_into().map_err(|_| {
                exceptions::PyValueError::new_err(format!("Solve must be done wrt a variable"))
            })?;
            vars.push(id);
        }

        if init[0].0.prec() == 53 {
            let init: Vec<_> = init.into_iter().map(|x| x.0.to_f64().into()).collect();

            let res: Vec<F64> =
                AtomView::nsolve_system(&system_b, &vars, &init, prec.into(), max_iterations)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Could not solve system: {e}"))
                    })?;

            Ok(res
                .into_iter()
                .map(|x| x.into_inner().into_py_any(py))
                .collect::<Result<_, _>>()?)
        } else {
            let init: Vec<_> = init.into_iter().map(|x| x.0).collect();

            let res: Vec<Float> =
                AtomView::nsolve_system(&system_b, &vars, &init, prec.into(), max_iterations)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Could not solve system: {e}"))
                    })?;

            Ok(res
                .into_iter()
                .map(|x| PythonMultiPrecisionFloat(x).into_py_any(py))
                .collect::<Result<_, _>>()?)
        }
    }

    /// Evaluate the expression, using a map of all the constants and
    /// user functions to a float.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> f = S('f')
    /// >>> e = E('cos(x)')*3 + f(x,2)
    /// >>> print(e.evaluate({x: 1}, {f: lambda args: args[0]+args[1]}))
    pub fn evaluate(
        &self,
        constants: HashMap<PythonExpression, f64>,
        #[gen_stub(override_type(
            type_repr = "dict[Expression, typing.Callable[[typing.Sequence[float]], float]]"
        ))]
        functions: HashMap<PolyVariable, Py<PyAny>>,
    ) -> PyResult<f64> {
        let constants = constants
            .iter()
            .map(|(k, v)| (k.expr.as_view(), *v))
            .collect();

        let functions = functions
            .into_iter()
            .map(|(k, v)| {
                let id = if let PolyVariable::Symbol(v) = k {
                    v
                } else {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Expected function name instead of {k:?}",
                    )))?
                };

                Ok((
                    id,
                    EvaluationFn::new(Box::new(move |args, _, _, _| {
                        Python::attach(|py| {
                            v.call(py, (args.to_vec(),), None)
                                .expect("Bad callback function")
                                .extract::<f64>(py)
                                .expect("Function does not return a float")
                        })
                    })),
                ))
            })
            .collect::<PyResult<_>>()?;

        self.expr
            .evaluate(|x| x.into(), &constants, &functions)
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not evaluate expression: {e}"))
            })
    }

    /// Evaluate the expression, using a map of all the constants and
    /// user functions using arbitrary precision arithmetic.
    /// The user has to specify the number of decimal digits of precision
    /// and provide all input numbers as floats, strings or `decimal`.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> from decimal import Decimal, getcontext
    /// >>> x = S('x', 'f')
    /// >>> e = E('cos(x)')*3 + f(x, 2)
    /// >>> getcontext().prec = 100
    /// >>> a = e.evaluate_with_prec({x: Decimal('1.123456789')}, {
    /// >>>                         f: lambda args: args[0] + args[1]}, 100)
    #[gen_stub(override_return_type(type_repr = "decimal.Decimal", imports = ("decimal")))]
    pub fn evaluate_with_prec(
        &self,
        constants: HashMap<PythonExpression, PythonMultiPrecisionFloat>,
        #[gen_stub(override_type(
            type_repr = "dict[Expression, typing.Callable[[typing.Sequence[decimal.Decimal]], float | str | decimal.Decimal]]"
        ))]
        functions: HashMap<PolyVariable, Py<PyAny>>,
        decimal_digit_precision: u32,
        py: Python,
    ) -> PyResult<Py<PyAny>> {
        let prec = (decimal_digit_precision as f64 * std::f64::consts::LOG2_10).ceil() as u32;

        let constants: HashMap<AtomView, Float> = constants
            .iter()
            .map(|(k, v)| {
                Ok((k.expr.as_view(), {
                    let mut vv = v.0.clone();
                    vv.set_prec(prec);
                    vv
                }))
            })
            .collect::<PyResult<_>>()?;

        let functions = functions
            .into_iter()
            .map(|(k, v)| {
                let id = if let PolyVariable::Symbol(v) = k {
                    v
                } else {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Expected function name instead of {k}",
                    )))?
                };

                Ok((
                    id,
                    EvaluationFn::new(Box::new(move |args: &[Float], _, _, _| {
                        Python::attach(|py| {
                            let mut vv = v
                                .call(
                                    py,
                                    (args
                                        .iter()
                                        .map(|x| {
                                            PythonMultiPrecisionFloat(x.clone())
                                                .into_pyobject(py)
                                                .expect("Could not convert to Python object")
                                        })
                                        .collect::<Vec<_>>(),),
                                    None,
                                )
                                .expect("Bad callback function")
                                .extract::<PythonMultiPrecisionFloat>(py)
                                .expect("Function does not return a string")
                                .0;
                            vv.set_prec(prec);
                            vv
                        })
                    })),
                ))
            })
            .collect::<PyResult<_>>()?;

        let a: PythonMultiPrecisionFloat = self
            .expr
            .evaluate(|x| x.to_multi_prec_float(prec), &constants, &functions)
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not evaluate expression: {e}"))
            })?
            .into();

        a.into_py_any(py)
    }

    /// Evaluate the expression, using a map of all the variables and
    /// user functions to a complex number.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x, y = S('x', 'y')
    /// >>> e = E('sqrt(x)')*y
    /// >>> print(e.evaluate_complex({x: 1 + 2j, y: 4 + 3j}, {}))
    pub fn evaluate_complex<'py>(
        &self,
        py: Python<'py>,
        constants: HashMap<PythonExpression, Complex<f64>>,
        #[gen_stub(override_type(
            type_repr = "dict[Expression, typing.Callable[[typing.Sequence[float | complex]], float | complex]]"
        ))]
        functions: HashMap<PolyVariable, Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyComplex>> {
        let constants = constants
            .iter()
            .map(|(k, v)| (k.expr.as_view(), *v))
            .collect();

        let functions = functions
            .into_iter()
            .map(|(k, v)| {
                let id = if let PolyVariable::Symbol(v) = k {
                    v
                } else {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Expected function name instead of {k:?}",
                    )))?
                };

                Ok((
                    id,
                    EvaluationFn::new(Box::new(move |args: &[Complex<f64>], _, _, _| {
                        Python::attach(|py| {
                            v.call(
                                py,
                                (args
                                    .iter()
                                    .map(|x| PyComplex::from_doubles(py, x.re, x.im))
                                    .collect::<Vec<_>>(),),
                                None,
                            )
                            .expect("Bad callback function")
                            .extract::<Complex<f64>>(py)
                            .expect("Function does not return a complex number")
                        })
                    })),
                ))
            })
            .collect::<PyResult<_>>()?;

        let r = self
            .expr
            .evaluate(|x| x.into(), &constants, &functions)
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not evaluate expression: {e}"))
            })?;
        Ok(PyComplex::from_doubles(py, r.re, r.im))
    }

    /// Create an evaluator that can evaluate (nested) expressions in an optimized fashion.
    /// All constants and functions should be provided as dictionaries, where the function
    /// dictionary has a key `(name, printable name, arguments)` and the value is the function
    /// body. For example the function `f(x,y)=x^2+y` should be provided as
    /// `{(f, "f", (x, y)): x**2 + y}`. All free parameters should be provided in the `params` list.
    ///
    /// Additionally, external functions can be registered that will call a Python function.
    ///
    /// If `KeyboardInterrupt` is triggered during the optimization, the optimization will stop and will yield the
    /// current best result.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x, y, z, pi, f, g = S(
    /// >>>     'x', 'y', 'z', 'pi', 'f', 'g')
    /// >>>
    /// >>> e1 = E("x + pi + cos(x) + f(g(x+1),x*2)")
    /// >>> fd = E("y^2 + z^2*y^2")
    /// >>> gd = E("y + 5")
    /// >>>
    /// >>> ev = e1.evaluator({pi: Expression.num(22)/7},
    /// >>>              {(f, "f", (y, z)): fd, (g, "g", (y, )): gd}, [x])
    /// >>> res = ev.evaluate([[1.], [2.], [3.]])  # evaluate at x=1, x=2, x=3
    /// >>> print(res)
    ///
    ///
    /// Define an external function:
    ///
    /// >>> E("f(x)").evaluator({}, {}, [S("x")],
    ///             external_functions={(S("f"), "F"): lambda args: args[0]**2 + 1})
    ///
    /// Define an conditional function which yields `x+1` when `y != 0` and `x+2` when `y == 0`:
    ///
    /// >>> E("if(y, x + 1, x + 2)").evaluator({}, {}, [S("x"), S("y")], conditional=[S("if")])
    ///
    /// Parameters
    /// ----------
    /// constants: dict[Expression, Expression]
    ///     A map of expressions to constants. The constants should be numerical expressions.
    /// funs: dict[Tuple[Expression, str, Sequence[Expression]], Expression]
    ///     A dictionary of functions. The key is a tuple of the function name, printable name and the argument variables.
    ///     The value is the function body.
    /// params: Sequence[Expression]
    ///     A list of free parameters.
    /// iterations: int, optional
    ///     The number of optimization iterations to perform.
    /// n_cores: int, optional
    ///     The number of cores to use for the optimization.
    /// verbose: bool, optional
    ///     Print the progress of the optimization.
    /// external_functions: Optional[dict[Tuple[Expression, str], Callable[[Sequence[float | complex]], float | complex]]]
    ///     A dictionary of external functions that can be called during evaluation.
    ///     The key is the function name and the value is a callable that takes a list of arguments and returns a float.
    ///     This is useful for functions that are not defined in Symbolica but are available in Python.
    /// conditionals: Optional[Sequence[Expression]], optional
    ///     A list of conditional functions. These functions should take three argument: a condition that is tested for
    ///     inequality with 0, the true branch and the false branch.
    #[pyo3(signature =
        (constants,
        functions,
        params,
        iterations = 100,
        n_cores = 4,
        verbose = false,
        external_functions = None,
        conditionals = None),
        )]
    pub fn evaluator(
        &self,
        constants: HashMap<PythonExpression, PythonExpression>,
        functions: HashMap<(PolyVariable, String, Vec<PolyVariable>), PythonExpression>,
        params: Vec<PythonExpression>,
        iterations: usize,
        n_cores: usize,
        verbose: bool,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[dict[tuple[Expression, str], typing.Callable[[
            typing.Sequence[float | complex]], float | complex]]]"
        ))]
        external_functions: Option<HashMap<(PolyVariable, String), Py<PyAny>>>,
        conditionals: Option<Vec<PolyVariable>>,
        py: Python<'_>,
    ) -> PyResult<PythonExpressionEvaluator> {
        let mut fn_map = FunctionMap::new();

        for (k, v) in constants {
            if let Ok(r) = v.expr.clone().try_into() {
                fn_map.add_constant(k.expr, r);
            } else {
                Err(exceptions::PyValueError::new_err("Constants must be complex rationals. If this is not possible, pass the value as a parameter".to_string()))?
            }
        }

        for ((symbol, rename, args), body) in functions {
            let symbol = symbol
                .get_id()
                .ok_or(exceptions::PyValueError::new_err(format!(
                    "Bad function name {symbol}",
                )))?;
            let args: Vec<_> = args
                .iter()
                .map(|x| {
                    x.get_id().ok_or(exceptions::PyValueError::new_err(format!(
                        "Bad function name {symbol}",
                    )))
                })
                .collect::<Result<_, _>>()?;

            fn_map
                .add_function(symbol, rename.clone(), args, body.expr)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
        }

        if let Some(ef) = &external_functions {
            for (symbol, name) in ef.keys() {
                let symbol = symbol
                    .get_id()
                    .ok_or(exceptions::PyValueError::new_err(format!(
                        "Bad function name {symbol}",
                    )))?;

                fn_map
                    .add_external_function(symbol, name.clone())
                    .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            }
        }

        if let Some(ef) = &conditionals {
            for symbol in ef {
                let symbol = symbol
                    .get_id()
                    .ok_or(exceptions::PyValueError::new_err(format!(
                        "Bad function name {symbol}",
                    )))?;

                fn_map
                    .add_conditional(symbol)
                    .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            }
        }

        let abort_check = Box::new(move || {
            Python::attach(|py| py.check_signals())
                .map(|_| false)
                .unwrap_or(true)
        });

        let settings = OptimizationSettings {
            horner_iterations: iterations,
            n_cores,
            verbose: verbose.into(),
            abort_check: Some(abort_check),
            ..OptimizationSettings::default()
        };

        let params: Vec<_> = params.iter().map(|x| x.expr.clone()).collect();

        let eval = py
            .detach(move || self.expr.evaluator(&fn_map, &params, settings))
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not create evaluator: {e}"))
            })?;

        let eval_f64 = if eval.is_real() {
            let external_functions_f64 = if let Some(ef) = external_functions.as_ref() {
                ef.clone()
                    .into_iter()
                    .map(move |((_, name), f)| {
                        let ff: Box<dyn Fn(&[f64]) -> f64 + Send + Sync> = Box::new(move |args| {
                            Python::attach(|py| {
                                f.call1(py, (args,)).unwrap().extract::<f64>(py).unwrap()
                            })
                        });

                        (name.clone(), ff)
                    })
                    .collect()
            } else {
                HashMap::default()
            };

            Some(
                eval.clone()
                    .map_coeff(&|x| x.to_real().unwrap().to_f64())
                    .with_external_functions(external_functions_f64)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!(
                            "Could not create complex evaluator: {e}",
                        ))
                    })?,
            )
        } else {
            None
        };
        let eval_complex = eval
            .clone()
            .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));

        let external_functions_complex = if let Some(ef) = external_functions {
            ef.clone()
                .into_iter()
                .map(move |((_, name), f)| {
                    let ff: Box<dyn Fn(&[Complex<f64>]) -> Complex<f64> + Send + Sync> =
                        Box::new(move |args| {
                            Python::attach(|py| {
                                let arg_map: Vec<_> = args
                                    .iter()
                                    .map(|x| PyComplex::from_doubles(py, x.re, x.im))
                                    .collect();

                                f.call1(py, (arg_map,))
                                    .unwrap()
                                    .extract::<Complex<f64>>(py)
                                    .unwrap()
                            })
                        });

                    (name.clone(), ff)
                })
                .collect()
        } else {
            HashMap::default()
        };

        let eval_complex_ext = eval_complex
            .with_external_functions(external_functions_complex)
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!(
                    "Could not create complex evaluator: {e}",
                ))
            })?;

        Ok(PythonExpressionEvaluator {
            eval_rat: eval,
            eval: eval_f64,
            eval_complex,
            eval_complex_ext,
        })
    }

    /// Create an evaluator that can jointly evaluate (nested) expressions in an optimized fashion.
    /// See `Expression.evaluator()` for more information.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x = S('x')
    /// >>> e1 = E("x^2 + 1")
    /// >>> e2 = E("x^2 + 2)
    /// >>> ev = Expression.evaluator_multiple([e1, e2], {}, {}, [x])
    ///
    /// will recycle the `x^2`
    #[classmethod]
    #[pyo3(signature =
        (exprs,
        constants,
        functions,
        params,
        iterations = 100,
        n_cores = 4,
        verbose = false,
        external_functions = None),
        )]
    pub fn evaluator_multiple(
        _cls: &Bound<'_, PyType>,
        exprs: Vec<PythonExpression>,
        constants: HashMap<PythonExpression, PythonExpression>,
        functions: HashMap<(PolyVariable, String, Vec<PolyVariable>), PythonExpression>,
        params: Vec<PythonExpression>,
        iterations: usize,
        n_cores: usize,
        verbose: bool,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[dict[tuple[Expression, str], typing.Callable[[
            typing.Sequence[float | complex]], float | complex]]]"
        ))]
        external_functions: Option<HashMap<(PolyVariable, String), Py<PyAny>>>,
    ) -> PyResult<PythonExpressionEvaluator> {
        let mut fn_map = FunctionMap::new();

        for (k, v) in constants {
            if let Ok(r) = v.expr.clone().try_into() {
                fn_map.add_constant(k.expr, r);
            } else {
                Err(exceptions::PyValueError::new_err("Constants must be complex rationals. If this is not possible, pass the value as a parameter".to_string()))?
            }
        }

        for ((symbol, rename, args), body) in functions {
            let symbol = symbol
                .get_id()
                .ok_or(exceptions::PyValueError::new_err(format!(
                    "Bad function name {symbol}",
                )))?;
            let args: Vec<_> = args
                .iter()
                .map(|x| {
                    x.get_id().ok_or(exceptions::PyValueError::new_err(format!(
                        "Bad function name {symbol}",
                    )))
                })
                .collect::<Result<_, _>>()?;

            fn_map
                .add_function(symbol, rename.clone(), args, body.expr)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
        }

        if let Some(ef) = &external_functions {
            for (symbol, name) in ef.keys() {
                let symbol = symbol
                    .get_id()
                    .ok_or(exceptions::PyValueError::new_err(format!(
                        "Bad function name {symbol}",
                    )))?;

                fn_map
                    .add_external_function(symbol, name.clone())
                    .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            }
        }

        let settings = OptimizationSettings {
            horner_iterations: iterations,
            n_cores,
            verbose: verbose.into(),
            ..OptimizationSettings::default()
        };

        let params: Vec<_> = params.iter().map(|x| x.expr.clone()).collect();

        let exprs = exprs.iter().map(|x| x.expr.as_view()).collect::<Vec<_>>();

        let eval = Atom::evaluator_multiple(&exprs, &fn_map, &params, settings).map_err(|e| {
            exceptions::PyValueError::new_err(format!("Could not create evaluator: {e}"))
        })?;

        let eval_f64 = if eval.is_real() {
            let external_functions_f64 = if let Some(ef) = external_functions.as_ref() {
                ef.clone()
                    .into_iter()
                    .map(move |((_, name), f)| {
                        let ff: Box<dyn Fn(&[f64]) -> f64 + Send + Sync> = Box::new(move |args| {
                            Python::attach(|py| {
                                f.call1(py, (args,)).unwrap().extract::<f64>(py).unwrap()
                            })
                        });

                        (name.clone(), ff)
                    })
                    .collect()
            } else {
                HashMap::default()
            };

            Some(
                eval.clone()
                    .map_coeff(&|x| x.to_real().unwrap().to_f64())
                    .with_external_functions(external_functions_f64)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!(
                            "Could not create complex evaluator: {e}",
                        ))
                    })?,
            )
        } else {
            None
        };
        let eval_complex = eval
            .clone()
            .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));

        let external_functions_complex = if let Some(ef) = external_functions {
            ef.clone()
                .into_iter()
                .map(move |((_, name), f)| {
                    let ff: Box<dyn Fn(&[Complex<f64>]) -> Complex<f64> + Send + Sync> =
                        Box::new(move |args| {
                            Python::attach(|py| {
                                let arg_map: Vec<_> = args
                                    .into_iter()
                                    .map(|x| PyComplex::from_doubles(py, x.re, x.im))
                                    .collect();

                                f.call1(py, (arg_map,))
                                    .unwrap()
                                    .extract::<Complex<f64>>(py)
                                    .unwrap()
                            })
                        });

                    (name.clone(), ff)
                })
                .collect()
        } else {
            HashMap::default()
        };

        let eval_complex_ext = eval_complex
            .with_external_functions(external_functions_complex)
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!(
                    "Could not create complex evaluator: {e}",
                ))
            })?;

        Ok(PythonExpressionEvaluator {
            eval_rat: eval,
            eval: eval_f64,
            eval_complex,
            eval_complex_ext,
        })
    }

    /// Canonize (products of) tensors in the expression by relabeling repeated indices.
    /// The tensors must be written as functions, with its indices as the arguments.
    /// Subexpressions, constants and open indices are supported.
    ///
    /// If the contracted indices are distinguishable (for example in their dimension),
    /// you can provide a group marker as the second element in the tuple of the index
    /// specification.
    /// This makes sure that an index will not be renamed to an index from a different group.
    ///
    /// Returns the canonical expression, as well as the external indices and ordered dummy indices
    /// appearing in the canonical expression.
    ///
    /// Examples
    /// --------
    /// g = S('g', is_symmetric=True)
    /// >>> fc = S('fc', is_cyclesymmetric=True)
    /// >>> mu1, mu2, mu3, mu4, k1 = S('mu1', 'mu2', 'mu3', 'mu4', 'k1')
    /// >>>
    /// >>> e = g(mu2, mu3)*fc(mu4, mu2, k1, mu4, k1, mu3)
    /// >>>
    /// >>> (r, external, dummy) = e.canonize_tensors([(mu1, 0), (mu2, 0), (mu3, 0), (mu4, 0)])
    /// >>> print(r)
    /// yields `g(mu1,mu2)*fc(mu1,mu3,mu2,k1,mu3,k1)`.
    fn canonize_tensors(
        &self,
        contracted_indices: Vec<(ConvertibleToExpression, ConvertibleToExpression)>,
    ) -> PyResult<(
        PythonExpression,
        Vec<(PythonExpression, PythonExpression)>,
        Vec<(PythonExpression, PythonExpression)>,
    )> {
        let contracted_indices = contracted_indices
            .into_iter()
            .map(|x| (x.0.to_expression().expr, x.1.to_expression().expr))
            .collect::<Vec<_>>();

        let r = self
            .expr
            .canonize_tensors(contracted_indices)
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not canonize tensors: {e}"))
            })?;

        Ok((
            r.canonical_form.into(),
            r.external_indices
                .into_iter()
                .map(|(t, g)| (t.into(), g.into()))
                .collect(),
            r.dummy_indices
                .into_iter()
                .map(|(t, g)| (t.into(), g.into()))
                .collect(),
        ))
    }
}

/// A raplacement, which is a pattern and a right-hand side, with optional conditions and settings.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "Replacement", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonReplacement {
    replacement: Replacement,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonReplacement {
    #[pyo3(signature = (pattern, rhs, cond=None, non_greedy_wildcards=None, level_range=None, level_is_tree_depth=None, allow_new_wildcards_on_rhs=None, rhs_cache_size=None))]
    #[new]
    pub fn new(
        pattern: ConvertibleToExpression,
        rhs: ConvertibleToReplaceWith,
        cond: Option<ConvertibleToPatternRestriction>,
        non_greedy_wildcards: Option<Vec<PythonExpression>>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: Option<bool>,
        allow_new_wildcards_on_rhs: Option<bool>,
        rhs_cache_size: Option<usize>,
    ) -> PyResult<Self> {
        let pattern = pattern.to_expression().expr.to_pattern();
        let rhs = rhs.to_replace_with()?;

        let mut settings = MatchSettings::cached();

        if let Some(ngw) = non_greedy_wildcards {
            settings.non_greedy_wildcards = ngw
                .iter()
                .map(|x| match x.expr.as_view() {
                    AtomView::Var(v) => {
                        let name = v.get_symbol();
                        if v.get_wildcard_level() == 0 {
                            return Err(exceptions::PyTypeError::new_err(
                                "Only wildcards can be restricted.",
                            ));
                        }
                        Ok(name)
                    }
                    _ => Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    )),
                })
                .collect::<Result<_, _>>()?;
        }
        if let Some(level_range) = level_range {
            settings.level_range = level_range;
        }
        if let Some(level_is_tree_depth) = level_is_tree_depth {
            settings.level_is_tree_depth = level_is_tree_depth;
        }
        if let Some(allow_new_wildcards_on_rhs) = allow_new_wildcards_on_rhs {
            settings.allow_new_wildcards_on_rhs = allow_new_wildcards_on_rhs;
        }
        if let Some(rhs_cache_size) = rhs_cache_size {
            settings.rhs_cache_size = rhs_cache_size;
        }

        Ok(Self {
            replacement: Replacement::new(pattern, rhs)
                .with_conditions(cond.map(|r| r.0).unwrap_or_default())
                .with_settings(settings),
        })
    }

    #[getter]
    fn pattern(&self) -> PyResult<PythonExpression> {
        Ok(self
            .replacement
            .pat
            .to_atom()
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not convert pattern to atom: {e}"))
            })?
            .into())
    }
}

#[cfg(feature = "python_stubgen")]
submit! {
PyMethodsInfo {
        struct_id: std::any::TypeId::of::<PythonExpression>,
        attrs: &[],
        getters: &[],
        setters: &[],
        methods: &[
            MethodInfo {
            name: "symbol",
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
                    name: "custom_normalization",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<PythonTransformer>::type_input(),
                },
                ParameterInfo {
                    name: "custom_print",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || TypeInfo::unqualified("typing.Optional[typing.Callable[..., typing.Optional[str]]]"),
                },
                ParameterInfo {
                    name: "custom_derivative",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || TypeInfo::unqualified("typing.Optional[typing.Callable[[Expression, int], Expression]]"),
                },
                ParameterInfo {
                    name: "data",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || TypeInfo::unqualified("typing.Optional[str | int | Expression | bytes | list | dict]"),
                },
            ],
            r#type: MethodType::Class,
            r#return: || PythonExpression::type_output(),
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
>>> p1, p2, p3, p4 = ES('p1', 'p2', 'p3', 'p4')
>>> dot = S('dot', is_symmetric=True, is_linear=True)
>>> e = dot(p2+2*p3,p1+3*p2-p3)
dot(p1,p2)+2*dot(p1,p3)+3*dot(p2,p2)-dot(p2,p3)+6*dot(p2,p3)-2*dot(p3,p3)

Define a custom normalization function:
>>> e = S('real_log', custom_normalization=T().replace(E("x_(exp(x1_))"), E("x1_")))
>>> E("real_log(exp(x)) + real_log(5)")

Define a custom print function:
>>> def print_mu(mu: Expression, mode: PrintMode, **kwargs) -> str | None:
>>>     if mode == PrintMode.Latex:
>>>         if mu.get_type() == AtomType.Fn:
>>>             return "\\mu_{" + ",".join(a.format() for a in mu) + "}"
>>>         else:
>>>             return "\\mu"
>>> mu = S("mu", custom_print=print_mu)
>>> expr = E("mu + mu(1,2)")
>>> print(expr.to_latex())

If the function returns `None`, the default print function is used.

Define a custom derivative function:
>>> tag = S('tag', custom_derivative=lambda f, index: f)
>>> x = S('x')
>>> tag(3, x).derivative(x)

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
custom_normalization : Optional[Transformer]
    A transformer that is called after every normalization. Note that the symbol
    name cannot be used in the transformer as this will lead to a definition of the
    symbol. Use a wildcard with the same attributes instead.
custom_print : Optional[Callable[..., Optional[str]]]:
    A function that is called when printing the variable/function, which is provided as its first argument.
    This function should return a string, or `None` if the default print function should be used.
    The custom print function takes in keyword arguments that are the same as the arguments of the `format` function.
custom_derivative: Optional[Callable[[Expression, int], Expression]]:
    A function that is called when computing the derivative of a function in a given argument."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
        },
            MethodInfo {
            name: "symbol",
            parameters: &[
                ParameterInfo {
                    name: "names",
                    kind: ParameterKind::VarPositional,
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
            ],
            r#type: MethodType::Class,
            r#return: || TypeInfo::unqualified("typing.Sequence[Expression]"),
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
>>> p1, p2, p3, p4 = ES('p1', 'p2', 'p3', 'p4')
>>> dot = S('dot', is_symmetric=True, is_linear=True)
>>> e = dot(p2+2*p3,p1+3*p2-p3)
dot(p1,p2)+2*dot(p1,p3)+3*dot(p2,p2)-dot(p2,p3)+6*dot(p2,p3)-2*dot(p3,p3)

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
    A list of tags to associate with the symbol."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
        }

          ],
    }
}

#[cfg(feature = "python_stubgen")]
submit! {
    PyMethodsInfo {
        struct_id: std::any::TypeId::of::<PythonExpression>,
        attrs: &[],
        getters: &[],
        setters: &[],
        methods: &[
            MethodInfo {
                name: "to_polynomial",
                parameters: &[
                    ParameterInfo {
                        name: "vars",
                        kind: ParameterKind::PositionalOrKeyword,
                        default: ParameterDefault::Expr(NONE_ARG),
                        type_info: || Option::<Vec<PythonExpression>>::type_input(),
                    },
                ],
                r#type: MethodType::Instance,
                r#return: || PythonPolynomial::type_output(),
                doc:"
Convert the expression to a polynomial, optionally, with the variable ordering specified in `vars`.
All non-polynomial parts will be converted to new, independent variables.",
                is_async: false,
                deprecated: None,
                type_ignored: None,
            },
            MethodInfo {
                name: "to_polynomial",
                parameters: &[
                    ParameterInfo {
                        name: "minimal_poly",
                        kind: ParameterKind::PositionalOrKeyword,
                        default: ParameterDefault::None,
                        type_info: || PythonPolynomial::type_input(),
                    },
                    ParameterInfo {
                        name: "vars",
                        kind: ParameterKind::PositionalOrKeyword,
                        default: ParameterDefault::Expr(NONE_ARG),
                        type_info: || Option::<Vec<PythonExpression>>::type_input(),
                    },
                ],
                r#type: MethodType::Instance,
                r#return: || PythonNumberFieldPolynomial::type_output(),
                doc: "
Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`.
All non-polynomial elements will be converted to new independent variables.

The coefficients will be converted to a number field with the minimal polynomial `minimal_poly`.
The minimal polynomial must be a monic, irreducible univariate polynomial.",
                is_async: false,
                deprecated: None,
                type_ignored: None,
            },
             MethodInfo {
                name: "to_polynomial",
                parameters: &[
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
                r#type: MethodType::Instance,
                r#return: || PythonFiniteFieldPolynomial::type_output(),
                doc: "
Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`.
All non-polynomial elements will be converted to new independent variables.

The coefficients will be converted to finite field elements modulo `modulus`.
If on top a `power` is provided, for example `(2, a)`, the polynomial will be converted to the Galois field
`GF(modulus^2)` where `a` is the variable of the minimal polynomial of the field.

If a `minimal_poly` is provided, the Galois field will be created with `minimal_poly` as the minimal polynomial.",
                is_async: false,
                deprecated: None,
                type_ignored: None,
            }
        ],
    }
}

#[cfg(feature = "python_stubgen")]
submit! {
    PyMethodsInfo {
        struct_id: std::any::TypeId::of::<PythonExpression>,
        attrs: &[],
        getters: &[],
        setters: &[],
        methods: &[
            MethodInfo {
                name: "__call__",
                parameters: &[
                    ParameterInfo {
                        name: "args",
                        kind: ParameterKind::VarPositional,
                        default: ParameterDefault::None,
                        type_info: || ConvertibleToExpression::type_input(),
                    },
                ],
                r#type: MethodType::Instance,
                r#return: || PythonExpression::type_output(),
                doc:"
Create a Symbolica expression by calling the function with appropriate arguments.

Examples
-------
>>> x, f = S('x', 'f')
>>> e = f(3,x)
>>> print(e)
f(3,x)",
                is_async: false,
                deprecated: None,
                type_ignored: None,
            },
            MethodInfo {
                name: "__call__",
                parameters: &[
                    ParameterInfo {
                        name: "args",
                        kind: ParameterKind::VarPositional,
                        default: ParameterDefault::None,
                        type_info: || PythonHeldExpression::type_input() | ConvertibleToExpression::type_input(),
                    },
                ],
                r#type: MethodType::Instance,
                r#return: || PythonHeldExpression::type_output(),
                doc: "
Create a Symbolica held expression by calling the function with appropriate arguments.

Examples
-------
>>> x, f = S('x', 'f')
>>> e = f(3,x)
>>> print(e)
f(3,x)",
                is_async: false,
                deprecated: None,
                type_ignored: None,
            }
        ],
    }
}

#[derive(FromPyObject)]
pub enum SeriesOrExpression {
    Series(PythonSeries),
    Expression(PythonExpression),
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(SeriesOrExpression = PythonSeries | PythonExpression);

/// A series expansion class.
///
/// Supports standard arithmetic operations, such
/// as addition and multiplication.
///
/// Examples
/// --------
/// >>> x = S('x')
/// >>> s = E("(1-cos(x))/sin(x)").series(x, 0, 4)
/// >>> print(s)
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "Series", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonSeries {
    pub series: Series<AtomField>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonSeries {
    /// Add this series to `rhs`, returning the result.
    pub fn __add__(&self, rhs: SeriesOrExpression) -> PyResult<Self> {
        match rhs {
            SeriesOrExpression::Series(rhs) => Ok(Self {
                series: &self.series + &rhs.series,
            }),
            SeriesOrExpression::Expression(rhs) => Ok(Self {
                series: (&self.series + &rhs.expr).map_err(exceptions::PyValueError::new_err)?,
            }),
        }
    }

    /// Add this series to `rhs`, returning the result.
    pub fn __radd__(&self, rhs: &PythonExpression) -> PyResult<Self> {
        Ok(Self {
            series: (&self.series + &rhs.expr).map_err(exceptions::PyValueError::new_err)?,
        })
    }

    pub fn __sub__(&self, rhs: SeriesOrExpression) -> PyResult<Self> {
        match rhs {
            SeriesOrExpression::Series(rhs) => Ok(Self {
                series: &self.series - &rhs.series,
            }),
            SeriesOrExpression::Expression(rhs) => Ok(Self {
                series: (&self.series - &rhs.expr).map_err(exceptions::PyValueError::new_err)?,
            }),
        }
    }

    pub fn __rsub__(&self, lhs: &PythonExpression) -> PyResult<Self> {
        Ok(Self {
            series: (&lhs.expr - &self.series).map_err(exceptions::PyValueError::new_err)?,
        })
    }

    pub fn __mul__(&self, rhs: SeriesOrExpression) -> PyResult<Self> {
        match rhs {
            SeriesOrExpression::Series(rhs) => Ok(Self {
                series: &self.series * &rhs.series,
            }),
            SeriesOrExpression::Expression(rhs) => Ok(Self {
                series: (&self.series * &rhs.expr).map_err(exceptions::PyValueError::new_err)?,
            }),
        }
    }

    pub fn __rmul__(&self, lhs: &PythonExpression) -> PyResult<Self> {
        Ok(Self {
            series: (&self.series * &lhs.expr).map_err(exceptions::PyValueError::new_err)?,
        })
    }

    pub fn __truediv__(&self, rhs: SeriesOrExpression) -> PyResult<Self> {
        match rhs {
            SeriesOrExpression::Series(rhs) => Ok(Self {
                series: &self.series / &rhs.series,
            }),
            SeriesOrExpression::Expression(rhs) => Ok(Self {
                series: (&self.series / &rhs.expr).map_err(exceptions::PyValueError::new_err)?,
            }),
        }
    }

    pub fn __rtruediv__(&self, lhs: &PythonExpression) -> PyResult<Self> {
        Ok(Self {
            series: (&lhs.expr / &self.series).map_err(exceptions::PyValueError::new_err)?,
        })
    }

    pub fn __pow__(&self, exponent: i64, modulo: Option<i64>) -> PyResult<Self> {
        if modulo.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "Optional number argument not supported",
            ));
        }

        Ok(Self {
            series: self
                .series
                .rpow((exponent, 1).into())
                .map_err(exceptions::PyValueError::new_err)?,
        })
    }

    pub fn __neg__(&self) -> Self {
        Self {
            series: -self.series.clone(),
        }
    }

    /// Convert the series into a portable string.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(self
            .series
            .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new()))
    }

    pub fn __str__(&self) -> PyResult<String> {
        Ok(self
            .series
            .format_string(&DEFAULT_PRINT_OPTIONS, PrintState::new()))
    }

    /// Convert the series into a LaTeX string.
    pub fn to_latex(&self) -> PyResult<String> {
        Ok(format!(
            "$${}$$",
            self.series
                .format_string(&LATEX_PRINT_OPTIONS, PrintState::new())
        ))
    }

    /// Convert the expression into a human-readable string, with tunable settings.
    ///
    /// Examples
    /// --------
    /// >>> a = E('128378127123 z^(2/3)*w^2/x/y + y^4 + z^34 + x^(x+2)+3/5+f(x,x^2)')
    /// >>> print(a.format(number_thousands_separator='_', multiplication_operator=' '))
    #[pyo3(signature =
        (mode = PythonPrintMode::Symbolica,
            terms_on_new_line = false,
            color_top_level_sum = true,
            color_builtin_symbols = true,
            print_ring = true,
            symmetric_representation_for_finite_field = false,
            explicit_rational_polynomial = false,
            number_thousands_separator = None,
            multiplication_operator = '*',
            double_star_for_exponentiation = false,
            square_brackets_for_function = false,
            num_exp_as_superscript = true,
            precision = None,
            show_namespaces = false,
            include_attributes = false,
            max_terms = None,
            custom_print_mode = None)
        )]
    pub fn format(
        &self,
        mode: PythonPrintMode,
        terms_on_new_line: bool,
        color_top_level_sum: bool,
        color_builtin_symbols: bool,
        print_ring: bool,
        symmetric_representation_for_finite_field: bool,
        explicit_rational_polynomial: bool,
        number_thousands_separator: Option<char>,
        multiplication_operator: char,
        double_star_for_exponentiation: bool,
        square_brackets_for_function: bool,
        num_exp_as_superscript: bool,
        precision: Option<usize>,
        show_namespaces: bool,
        include_attributes: bool,
        max_terms: Option<usize>,
        custom_print_mode: Option<usize>,
    ) -> PyResult<String> {
        Ok(self
            .series
            .format_string(
                &PrintOptions {
                    terms_on_new_line,
                    color_top_level_sum,
                    color_builtin_symbols,
                    print_ring,
                    symmetric_representation_for_finite_field,
                    explicit_rational_polynomial,
                    number_thousands_separator,
                    multiplication_operator,
                    double_star_for_exponentiation,
                    square_brackets_for_function,
                    num_exp_as_superscript,
                    mode: mode.into(),
                    precision,
                    pretty_matrix: false,
                    hide_all_namespaces: !show_namespaces,
                    color_namespace: true,
                    hide_namespace: Some("python"),
                    include_attributes,
                    max_terms,
                    custom_print_mode: custom_print_mode.map(|x| ("default", x)),
                },
                PrintState::new(),
            )
            .to_string())
    }

    pub fn sin(&self) -> PyResult<Self> {
        Ok(Self {
            series: self
                .series
                .sin()
                .map_err(exceptions::PyValueError::new_err)?,
        })
    }

    pub fn cos(&self) -> PyResult<Self> {
        Ok(Self {
            series: self
                .series
                .cos()
                .map_err(exceptions::PyValueError::new_err)?,
        })
    }

    pub fn exp(&self) -> PyResult<Self> {
        Ok(Self {
            series: self
                .series
                .exp()
                .map_err(exceptions::PyValueError::new_err)?,
        })
    }

    pub fn log(&self) -> PyResult<Self> {
        Ok(Self {
            series: self
                .series
                .log()
                .map_err(exceptions::PyValueError::new_err)?,
        })
    }

    pub fn pow(&self, num: i64, den: i64) -> PyResult<Self> {
        Ok(Self {
            series: self
                .series
                .rpow((num, den).into())
                .map_err(exceptions::PyValueError::new_err)?,
        })
    }

    pub fn spow(&self, pow: &Self) -> PyResult<Self> {
        Ok(Self {
            series: self
                .series
                .pow(&pow.series)
                .map_err(exceptions::PyValueError::new_err)?,
        })
    }

    /// Shift the series by `e` units of the ramification.
    pub fn shift(&self, e: isize) -> Self {
        Self {
            series: self.series.clone().mul_exp_units(e),
        }
    }

    /// Get the ramification.
    pub fn get_ramification(&self) -> usize {
        self.series.get_ramification()
    }

    /// Get the trailing exponent; the exponent of the first non-zero term.
    pub fn get_trailing_exponent(&self) -> PyResult<(i64, i64)> {
        let r = self.series.get_trailing_exponent();
        if let Integer::Single(n) = r.numerator_ref() {
            if let Integer::Single(d) = r.denominator_ref() {
                return Ok((*n, *d));
            }
        }

        Err(exceptions::PyValueError::new_err("Order is too large"))
    }

    /// Get the relative order.
    pub fn get_relative_order(&self) -> PyResult<(i64, i64)> {
        let r = self.series.relative_order();
        if let Integer::Single(n) = r.numerator_ref() {
            if let Integer::Single(d) = r.denominator_ref() {
                return Ok((*n, *d));
            }
        }

        Err(exceptions::PyValueError::new_err("Order is too large"))
    }

    /// Get the absolute order.
    pub fn get_absolute_order(&self) -> PyResult<(i64, i64)> {
        let r = self.series.absolute_order();
        if let Integer::Single(n) = r.numerator_ref() {
            if let Integer::Single(d) = r.denominator_ref() {
                return Ok((*n, *d));
            }
        }

        Err(exceptions::PyValueError::new_err("Order is too large"))
    }

    /// Convert the series into an expression.
    pub fn to_expression(&self) -> PythonExpression {
        self.series.to_atom().into()
    }
}

/// A term streamer that can handle large expressions, by
/// streaming terms to and from disk.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "TermStreamer", subclass, module = "symbolica.core")]
pub struct PythonTermStreamer {
    pub stream: TermStreamer<CompressorWriter<BufWriter<File>>>,
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(&mut PythonTermStreamer = PythonTermStreamer);

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonTermStreamer {
    /// Create a new term streamer with a given path for its files,
    /// the maximum size of the memory buffer and the number of cores.
    #[pyo3(signature = (path = None, max_mem_bytes = None, n_cores = None))]
    #[new]
    pub fn __new__(
        path: Option<&str>,
        max_mem_bytes: Option<usize>,
        n_cores: Option<usize>,
    ) -> PyResult<Self> {
        let d = TermStreamerConfig::default();

        Ok(PythonTermStreamer {
            stream: TermStreamer::new(TermStreamerConfig {
                n_cores: n_cores.unwrap_or(d.n_cores),
                max_mem_bytes: max_mem_bytes.unwrap_or(d.max_mem_bytes),
                path: path.map(|x| x.into()).unwrap_or(d.path),
            }),
        })
    }

    /// Add this expression to `other`, returning the result.
    ///
    pub fn __add__(&mut self, rhs: &mut Self) -> PyResult<Self> {
        Ok(Self {
            stream: &mut self.stream + &mut rhs.stream,
        })
    }

    pub fn __iadd__(&mut self, rhs: &mut Self) {
        self.stream += &mut rhs.stream;
    }

    /// Clear all terms from the term streamer.
    pub fn clear(&mut self) {
        self.stream.clear()
    }

    /// Load terms and their state from a binary stream into the term streamer.
    /// The state will be merged with the current one. If a symbol has conflicting attributes, the conflict
    /// can be resolved using the renaming function `conflict_fn`.
    ///
    /// A term stream can be exported using `TermStreamer.save`.

    #[pyo3(signature = (filename, conflict_fn=None))]
    pub fn load(
        &mut self,
        filename: &str,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Callable[[str], str]]"))]
        conflict_fn: Option<Py<PyAny>>,
    ) -> PyResult<u64> {
        let f = File::open(filename)
            .map_err(|e| exceptions::PyIOError::new_err(format!("Could not read file: {e}")))?;
        let mut reader = brotli::Decompressor::new(BufReader::new(f), 4096);

        self.stream
            .import(
                &mut reader,
                match conflict_fn {
                    Some(f) => Some(Box::new(move |name: &str| -> SmartString<LazyCompact> {
                        Python::attach(|py| {
                            f.call1(py, (name,)).unwrap().extract::<String>(py).unwrap()
                        })
                        .into()
                    })),
                    None => None,
                },
            )
            .map_err(|e| exceptions::PyIOError::new_err(format!("Could not read file: {e}")))
    }

    /// Export terms and their state to a binary stream.
    /// The resulting file can be read back using `TermStreamer.load` or
    /// by using `Expression.load`. In the latter case, the whole term stream will be read into memory
    /// as a single expression.
    #[pyo3(signature = (filename, compression_level=9))]
    pub fn save(&mut self, filename: &str, compression_level: u32) -> PyResult<()> {
        let f = File::create(filename)
            .map_err(|e| exceptions::PyIOError::new_err(format!("Could not create file: {e}")))?;
        let mut writer = CompressorWriter::new(BufWriter::new(f), 4096, compression_level, 22);
        self.stream
            .export(&mut writer)
            .map_err(exceptions::PyIOError::new_err)
    }

    /// Get the total number of bytes of the stream.
    pub fn get_byte_size(&self) -> usize {
        self.stream.get_byte_size()
    }

    /// Return true iff the stream fits in memory.
    pub fn fits_in_memory(&self) -> bool {
        self.stream.fits_in_memory()
    }

    /// Get the number of terms in the stream.
    pub fn get_num_terms(&self) -> usize {
        self.stream.get_num_terms()
    }

    /// Add an expression to the term stream.
    pub fn push(&mut self, expr: PythonExpression) {
        self.stream.push(expr.expr.clone());
    }

    /// Sort and fuse all terms in the stream.
    pub fn normalize(&mut self) {
        self.stream.normalize();
    }

    /// Convert the term stream into an expression. This may exceed the available memory.
    pub fn to_expression(&mut self) -> PythonExpression {
        self.stream.to_expression().into()
    }

    /// Map the transformations to every term in the stream.
    #[pyo3(signature = (op, stats_to_file=None))]
    pub fn map(
        &mut self,
        op: PythonTransformer,
        stats_to_file: Option<String>,
        py: Python,
    ) -> PyResult<Self> {
        let state = if let Some(stats_to_file) = stats_to_file {
            let file = File::create(stats_to_file).map_err(|e| {
                exceptions::PyIOError::new_err(format!(
                    "Could not create file for transformer statistics: {e}",
                ))
            })?;
            TransformerState {
                stats_export: Some(Arc::new(Mutex::new(BufWriter::new(file)))),
                ..Default::default()
            }
        } else {
            TransformerState::default()
        };

        // release the GIL as Python functions may be called from
        // within the term mapper
        py.detach(move || {
            // map every term in the expression
            let m = self.stream.map(|x| {
                let mut out = Atom::default();
                let _ = Workspace::get_local().with(|ws| {
                    let _ =
                        Transformer::execute_chain(x.as_view(), &op.chain, ws, &state, &mut out)
                            .unwrap_or_else(|e| {
                                // TODO: capture and abort the parallel run
                                panic!("Transformer failed during parallel execution: {e:?}")
                            });
                });
                out
            });
            Ok::<_, PyErr>(m)
        })
        .map(|x| PythonTermStreamer { stream: x })
    }

    /// Apply a transformer to all terms in the stream using a single thread.
    ///
    /// Parameters
    /// ----------
    /// f: Transformer
    ///     The transformer to apply.
    /// stats_to_file: str, optional
    ///     If set, the output of the `stats` transformer will be written to a file in JSON format.
    #[pyo3(signature = (op, stats_to_file=None))]
    pub fn map_single_thread(
        &mut self,
        op: PythonTransformer,
        stats_to_file: Option<String>,
    ) -> PyResult<Self> {
        let state = if let Some(stats_to_file) = stats_to_file {
            let file = File::create(stats_to_file).map_err(|e| {
                exceptions::PyIOError::new_err(format!(
                    "Could not create file for transformer statistics: {e}",
                ))
            })?;
            TransformerState {
                stats_export: Some(Arc::new(Mutex::new(BufWriter::new(file)))),
                ..Default::default()
            }
        } else {
            TransformerState::default()
        };

        // map every term in the expression
        let s = self.stream.map_single_thread(|x| {
            let mut out = Atom::default();
            Workspace::get_local().with(|ws| {
                let _ = Transformer::execute_chain(x.as_view(), &op.chain, ws, &state, &mut out)
                    .unwrap_or_else(|e| panic!("Transformer failed during execution: {e:?}"));
            });
            out
        });

        Ok(PythonTermStreamer { stream: s })
    }
}

self_cell!(
    #[cfg_attr(
        feature = "python_stubgen",
        gen_stub_pyclass(module = "symbolica.core")
    )]
    #[pyclass(name = "AtomIterator", module = "symbolica.core")]
    pub struct PythonAtomIterator {
        owner: Atom,
        #[covariant]
        dependent: ListIterator,
    }
);

impl PythonAtomIterator {
    /// Create a self-referential structure for the iterator.
    pub fn from_expr(expr: PythonExpression) -> PythonAtomIterator {
        PythonAtomIterator::new(expr.expr.clone(), |expr| match expr.as_view() {
            AtomView::Add(a) => a.iter(),
            AtomView::Mul(m) => m.iter(),
            AtomView::Fun(f) => f.iter(),
            _ => unreachable!(),
        })
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonAtomIterator {
    /// Create the iterator.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[gen_stub(override_return_type(type_repr = "Expression"))]
    fn __next__(&mut self) -> Option<PythonExpression> {
        self.with_dependent_mut(|_, i| {
            i.next().map(|e| {
                let mut owned = Atom::default();
                owned.set_from_view(&e);
                owned.into()
            })
        })
    }
}

type OwnedMatch = (Pattern, Atom, Condition<PatternRestriction>, MatchSettings);
type MatchIterator<'a> = PatternAtomTreeIterator<'a, 'a>;

self_cell!(
    /// An iterator over matches.
    #[cfg_attr(
        feature = "python_stubgen",
        gen_stub_pyclass(module = "symbolica.core")
    )]
    #[pyclass(name = "MatchIterator", module = "symbolica.core")]
    pub struct PythonMatchIterator {
        owner: OwnedMatch,
        #[not_covariant]
        dependent: MatchIterator,
    }
);

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonMatchIterator {
    /// Create the iterator.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Return the next match.
    #[gen_stub(override_return_type(type_repr = "builtins.dict[Expression, Expression]"))]
    fn __next__(&mut self) -> Option<HashMap<PythonExpression, PythonExpression>> {
        self.with_dependent_mut(|_, i| {
            i.next().map(|m| {
                m.into_iter()
                    .map(|(k, v)| (Atom::var(k).into(), { v.into() }))
                    .collect()
            })
        })
    }
}

type OwnedReplace = (
    Pattern,
    Atom,
    ReplaceWith<'static>,
    Condition<PatternRestriction>,
    MatchSettings,
);
type ReplaceIteratorOne<'a> = ReplaceIterator<'a, 'a>;

self_cell!(
    /// An iterator over all single replacements.
    #[cfg_attr(
        feature = "python_stubgen",
        gen_stub_pyclass(module = "symbolica.core")
    )]
    #[pyclass(name = "ReplaceIterator", module = "symbolica.core")]
    pub struct PythonReplaceIterator {
        owner: OwnedReplace,
        #[not_covariant]
        dependent: ReplaceIteratorOne,
    }
);

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonReplaceIterator {
    /// Create the iterator.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Return the next replacement.
    #[gen_stub(override_return_type(type_repr = "Expression"))]
    fn __next__(&mut self) -> PyResult<Option<PythonExpression>> {
        self.with_dependent_mut(|_, i| Ok(i.next().map(|x| x.into())))
    }
}

#[derive(FromPyObject)]
pub enum PolynomialOrInteger<T> {
    Polynomial(T),
    Integer(Integer),
}

#[cfg(feature = "python_stubgen")]
impl<T: PyStubType> PyStubType for PolynomialOrInteger<T> {
    fn type_output() -> TypeInfo {
        T::type_output() | Integer::type_output()
    }
    fn type_input() -> TypeInfo {
        T::type_input() | Integer::type_input()
    }
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "Polynomial", subclass, module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonPolynomial {
    pub poly: MultivariatePolynomial<RationalField, u16>,
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(&mut PythonPolynomial = PythonPolynomial);

#[cfg(feature = "python_stubgen")]
impl_stub_type!(OneOrMultiple<PythonExpression> = PythonExpression | Vec<PythonExpression>);

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonPolynomial {
    /// Compare two polynomials.
    fn __richcmp__(&self, o: Py<PyAny>, op: CompareOp, py: Python) -> PyResult<bool> {
        let Ok(other) = o.extract::<PolynomialOrInteger<PythonPolynomial>>(py) else {
            return Err(exceptions::PyTypeError::new_err(
                "Can only compare Polynomial with Polynomial or integer.",
            ));
        };

        match other {
            PolynomialOrInteger::Polynomial(other) => match op {
                CompareOp::Eq => Ok(self.poly == other.poly),
                CompareOp::Ne => Ok(self.poly != other.poly),
                _ => {
                    if self.poly.is_constant() && other.poly.is_constant() {
                        return Ok(match op {
                            CompareOp::Ge => self.poly.lcoeff() >= other.poly.lcoeff(),
                            CompareOp::Gt => self.poly.lcoeff() > other.poly.lcoeff(),
                            CompareOp::Le => self.poly.lcoeff() <= other.poly.lcoeff(),
                            CompareOp::Lt => self.poly.lcoeff() < other.poly.lcoeff(),
                            CompareOp::Eq => self.poly == other.poly,
                            CompareOp::Ne => self.poly != other.poly,
                        });
                    }

                    Err(exceptions::PyTypeError::new_err(format!(
                        "Inequalities between polynomials that are not numbers are not allowed in {} {} {}",
                        self.__str__()?,
                        match op {
                            CompareOp::Eq => "==",
                            CompareOp::Ge => ">=",
                            CompareOp::Gt => ">",
                            CompareOp::Le => "<=",
                            CompareOp::Lt => "<",
                            CompareOp::Ne => "!=",
                        },
                        other.__str__()?,
                    )))
                }
            },
            PolynomialOrInteger::Integer(i) => {
                if !self.poly.is_constant() && !matches!(op, CompareOp::Eq | CompareOp::Ne) {
                    return Err(exceptions::PyTypeError::new_err(format!(
                        "Inequalities between polynomials that are not numbers are not allowed in {} {} {}",
                        self.__str__()?,
                        match op {
                            CompareOp::Eq => "==",
                            CompareOp::Ge => ">=",
                            CompareOp::Gt => ">",
                            CompareOp::Le => "<=",
                            CompareOp::Lt => "<",
                            CompareOp::Ne => "!=",
                        },
                        i,
                    )));
                }

                let r: Rational = i.into();
                return Ok(match op {
                    CompareOp::Eq => self.poly == r,
                    CompareOp::Ne => self.poly != r,
                    CompareOp::Ge => self.poly.lcoeff() >= r,
                    CompareOp::Gt => self.poly.lcoeff() > r,
                    CompareOp::Le => self.poly.lcoeff() <= r,
                    CompareOp::Lt => self.poly.lcoeff() < r,
                });
            }
        }
    }

    /// Copy the polynomial.
    pub fn __copy__(&self) -> Self {
        Self {
            poly: self.poly.clone(),
        }
    }

    /// Convert the polynomial into a human-readable string, with tunable settings.
    ///
    /// Examples
    /// --------
    /// >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
    /// >>> print(p.format(symmetric_representation_for_finite_field=True))
    #[pyo3(signature =
        (mode = PythonPrintMode::Symbolica,
            terms_on_new_line = false,
            color_top_level_sum = true,
            color_builtin_symbols = true,
            print_ring = true,
            symmetric_representation_for_finite_field = false,
            explicit_rational_polynomial = false,
            number_thousands_separator = None,
            multiplication_operator = '*',
            double_star_for_exponentiation = false,
            square_brackets_for_function = false,
            num_exp_as_superscript = true,
            precision = None,
            show_namespaces = false,
            include_attributes = false,
            max_terms = None,
            custom_print_mode = None)
        )]
    pub fn format(
        &self,
        mode: PythonPrintMode,
        terms_on_new_line: bool,
        color_top_level_sum: bool,
        color_builtin_symbols: bool,
        print_ring: bool,
        symmetric_representation_for_finite_field: bool,
        explicit_rational_polynomial: bool,
        number_thousands_separator: Option<char>,
        multiplication_operator: char,
        double_star_for_exponentiation: bool,
        square_brackets_for_function: bool,
        num_exp_as_superscript: bool,
        precision: Option<usize>,
        show_namespaces: bool,
        include_attributes: bool,
        max_terms: Option<usize>,
        custom_print_mode: Option<usize>,
    ) -> PyResult<String> {
        Ok(self.poly.format_string(
            &PrintOptions {
                terms_on_new_line,
                color_top_level_sum,
                color_builtin_symbols,
                print_ring,
                symmetric_representation_for_finite_field,
                explicit_rational_polynomial,
                number_thousands_separator,
                multiplication_operator,
                double_star_for_exponentiation,
                square_brackets_for_function,
                num_exp_as_superscript,
                mode: mode.into(),
                precision,
                pretty_matrix: false,
                hide_all_namespaces: !show_namespaces,
                color_namespace: true,
                hide_namespace: Some("python"),
                include_attributes,
                max_terms,
                custom_print_mode: custom_print_mode.map(|x| ("default", x)),
            },
            PrintState::new(),
        ))
    }

    /// Convert the polynomial into a portable string.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new()))
    }

    /// Print the polynomial in a human-readable format.
    pub fn __str__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&DEFAULT_PRINT_OPTIONS, PrintState::new()))
    }

    pub fn __pow__(&self, exponent: usize, modulo: Option<i64>) -> PyResult<PythonPolynomial> {
        if modulo.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "Optional number argument not supported",
            ));
        }

        Ok(Self {
            poly: self.poly.pow(exponent),
        })
    }

    /// Convert the polynomial into a LaTeX string.
    pub fn to_latex(&self) -> PyResult<String> {
        Ok(format!(
            "$${}$$",
            self.poly
                .format_string(&LATEX_PRINT_OPTIONS, PrintState::new())
        ))
    }

    /// Get the number of terms.
    pub fn nterms(&self) -> usize {
        self.poly.nterms()
    }

    /// Get the list of variables in the internal ordering of the polynomial.
    pub fn get_variables(&self) -> PyResult<Vec<PythonExpression>> {
        let mut var_list = vec![];

        for x in self.poly.get_vars_ref() {
            match x {
                PolyVariable::Symbol(x) => {
                    var_list.push(Atom::var(*x).into());
                }
                PolyVariable::Temporary(_) => {
                    Err(exceptions::PyValueError::new_err(
                        "Temporary variable in polynomial".to_string(),
                    ))?;
                }
                PolyVariable::Function(_, a) | PolyVariable::Power(a) => {
                    var_list.push(a.as_ref().clone().into());
                }
            }
        }

        Ok(var_list)
    }

    /// Add two polynomials `self and `rhs`, returning the result.
    pub fn __add__(&self, rhs: PolynomialOrInteger<PythonPolynomial>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly + &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self.poly.clone().add_constant(Rational::from(i)),
            }),
        }
    }

    /// Subtract polynomials `rhs` from `self`, returning the result.
    pub fn __sub__(&self, rhs: PolynomialOrInteger<PythonPolynomial>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly - &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self.poly.clone().add_constant(-Rational::from(i)),
            }),
        }
    }

    /// Multiply two polynomials `self and `rhs`, returning the result.
    pub fn __mul__(&self, rhs: PolynomialOrInteger<PythonPolynomial>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly * &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self.poly.clone().mul_coeff(Rational::from(i)),
            }),
        }
    }

    pub fn __radd__(&self, rhs: PolynomialOrInteger<PythonPolynomial>) -> PyResult<Self> {
        self.__add__(rhs)
    }

    pub fn __rsub__(&self, rhs: PolynomialOrInteger<PythonPolynomial>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &p.poly - &self.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self.poly.clone().neg().add_constant(Rational::from(i)),
            }),
        }
    }

    pub fn __rmul__(&self, rhs: PolynomialOrInteger<PythonPolynomial>) -> PyResult<Self> {
        self.__mul__(rhs)
    }

    pub fn __floordiv__(&self, rhs: Self) -> PyResult<Self> {
        if rhs.poly.is_zero() {
            return Err(exceptions::PyValueError::new_err("Division by zero"));
        }

        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        };

        let (q, _r) = self.poly.quot_rem(&rhs.poly, false);

        Ok(Self { poly: q })
    }

    /// Divide the polynomial `self` by `rhs` if possible, returning the result.
    pub fn __truediv__(&self, rhs: Self) -> PyResult<Self> {
        if rhs.poly.is_zero() {
            return Err(exceptions::PyValueError::new_err("Division by zero"));
        }

        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        };

        let (q, r) = self.poly.quot_rem(&rhs.poly, false);

        if r.is_zero() {
            Ok(Self { poly: q })
        } else {
            Err(exceptions::PyValueError::new_err(format!(
                "The division has a remainder: {r}",
            )))
        }
    }

    pub fn unify_variables(&mut self, other: &mut Self) {
        self.poly.unify_variables(&mut other.poly);
    }

    pub fn __contains__(&self, var: &PythonExpression) -> bool {
        self.contains(var)
    }

    pub fn contains(&self, var: &PythonExpression) -> bool {
        if let Some(p) =
            self.poly
                .get_vars_ref()
                .iter()
                .position(|v| match (v, var.expr.as_view()) {
                    (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                    (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                    _ => false,
                })
        {
            self.poly.contains(p)
        } else {
            false
        }
    }

    pub fn degree(&self, var: &PythonExpression) -> PyResult<isize> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, var.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                var.__str__()?
            )))?;

        Ok(self.poly.degree(x) as isize)
    }

    /// Set a new variable ordering for the polynomial.
    /// This can be used to introduce new variables as well.
    pub fn reorder(&mut self, order: Vec<PythonExpression>) -> PyResult<()> {
        let vars: Vec<_> = order
            .into_iter()
            .map(|x| x.expr.try_into())
            .collect::<Result<_, _>>()
            .map_err(|e| exceptions::PyValueError::new_err(e))?;
        self.poly = self
            .poly
            .rearrange_with_growth(&vars)
            .map_err(exceptions::PyValueError::new_err)?;
        Ok(())
    }

    /// Divide `self` by `rhs`, returning the quotient and remainder.
    pub fn quot_rem(&self, rhs: Self) -> PyResult<(PythonPolynomial, PythonPolynomial)> {
        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        };

        if rhs.poly.is_zero() {
            Err(exceptions::PyValueError::new_err("Division by zero"))
        } else {
            let (q, r) = self.poly.quot_rem(&rhs.poly, false);
            Ok((Self { poly: q }, Self { poly: r }))
        }
    }

    /// Negate the polynomial.
    pub fn __neg__(&self) -> Self {
        Self {
            poly: self.poly.clone().neg(),
        }
    }

    /// Compute the remainder `self % rhs.
    pub fn __mod__(&self, rhs: Self) -> PyResult<Self> {
        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        };

        if rhs.poly.is_zero() {
            Err(exceptions::PyValueError::new_err("Division by zero"))
        } else {
            Ok(Self {
                poly: self.poly.rem(&rhs.poly),
            })
        }
    }

    /// Compute the greatest common divisor (GCD) of two or more polynomials.
    #[pyo3(signature = (*rhs))]
    pub fn gcd(
        &self,
        #[gen_stub(override_type(type_repr = "Polynomial"))] rhs: &Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
        if rhs.len() == 1 {
            let rhs = rhs.get_item(0)?.extract::<Self>()?;
            if self.poly.ring != rhs.poly.ring {
                Err(exceptions::PyValueError::new_err(
                    "Polynomials have different rings".to_string(),
                ))
            } else {
                Ok(Self {
                    poly: self.poly.gcd(&rhs.poly),
                })
            }
        } else {
            let mut args = vec![self.poly.clone()];
            for r in rhs.iter() {
                let p = r.extract::<Self>()?;
                if args[0].ring != p.poly.ring {
                    return Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ));
                }
                args.push(p.poly);
            }

            Ok(Self {
                poly: PolynomialGCD::gcd_multiple(args),
            })
        }
    }

    /// Compute the extended GCD of two polynomials, yielding the GCD and the Bezout coefficients `s` and `t`
    /// such that `self * s + rhs * t = gcd(self, rhs)`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> E('(1+x)(20+x)').to_polynomial().extended_gcd(E('x^2+2').to_polynomial())
    ///
    /// yields `(1, 1/67-7/402*x, 47/134+7/402*x)`.
    pub fn extended_gcd(
        &self,
        rhs: Self,
    ) -> PyResult<(PythonPolynomial, PythonPolynomial, PythonPolynomial)> {
        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        }

        if self.poly.variables != rhs.poly.variables
            || (0..self.poly.nvars())
                .filter(|i| self.poly.degree(*i) > 0 || rhs.poly.degree(*i) > 0)
                .count()
                > 1
        {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials are not univariate in the same variable".to_string(),
            ));
        }

        let (g, s, t) = self.poly.eea_univariate(&rhs.poly);
        Ok((Self { poly: g }, Self { poly: s }, Self { poly: t }))
    }

    /// Compute the resultant of two polynomials with respect to the variable `var`.
    pub fn resultant(&self, rhs: Self, var: &PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, var.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                var.__str__()?
            )))?;

        if self.poly.get_vars_ref() == rhs.poly.get_vars_ref() {
            let self_uni = self.poly.to_univariate(x);
            let rhs_uni = rhs.poly.to_univariate(x);

            Ok(Self {
                poly: self_uni.resultant_prs(&rhs_uni),
            })
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);

            let self_uni = new_self.to_univariate(x);
            let rhs_uni = new_rhs.to_univariate(x);

            Ok(Self {
                poly: self_uni.resultant_prs(&rhs_uni),
            })
        }
    }

    /// Compute the square-free factorization of the polynomial.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial()
    /// >>> print('Square-free factorization of {}:'.format(p))
    /// >>> for f, exp in p.factor_square_free():
    /// >>>     print('\t({})^{}'.format(f, exp))
    pub fn factor_square_free(&self) -> Vec<(PythonPolynomial, usize)> {
        self.poly
            .square_free_factorization()
            .into_iter()
            .map(|(f, p)| (Self { poly: f }, p))
            .collect()
    }

    /// Factorize the polynomial.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial()
    /// >>> print('Factorization of {}:'.format(p))
    /// >>> for f, exp in p.factor():
    /// >>>     print('\t({})^{}'.format(f, exp))
    pub fn factor(&self) -> Vec<(PythonPolynomial, usize)> {
        self.poly
            .factor()
            .into_iter()
            .map(|(f, p)| (Self { poly: f }, p))
            .collect()
    }

    /// Take a derivative in `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x^2+2').to_polynomial()
    /// >>> print(p.derivative(x))
    pub fn derivative(&self, x: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, x.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        Ok(Self {
            poly: self.poly.derivative(x),
        })
    }

    /// Get the content, i.e., the GCD of the coefficients.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('3x^2+6x+9').to_polynomial()
    /// >>> print(p.content())
    pub fn content(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.constant(self.poly.content()),
        })
    }

    /// Get the primitive part of the polynomial, i.e., the polynomial divided
    /// by its content.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('6x^2+3x+9').to_polynomial().primitive()
    /// >>> print(p)
    ///
    /// Yields `2*x^2+x+3`.
    pub fn primitive(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.clone().make_primitive(),
        })
    }

    /// Make the polynomial monic, i.e., divide by the leading coefficient.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> p = E('6x^2+3x+9').to_polynomial().monic()
    /// >>> print(p)
    ///
    /// Yields `x^2+1/2*x+3/2`.
    pub fn monic(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.clone().make_monic(),
        })
    }

    /// Get the leading coefficient.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> p = E('3x^2+6x+9').to_polynomial().lcoeff()
    /// >>> print(p)
    ///
    /// Yields `3`.
    pub fn lcoeff(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.constant(self.poly.lcoeff().clone()),
        })
    }

    /// Get the coefficient list, optionally in the variables `vars`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial()
    /// >>> for n, pp in p.coefficient_list(x):
    /// >>>     print(n, pp)
    #[pyo3(signature = (vars = None))]
    pub fn coefficient_list(
        &self,
        vars: Option<OneOrMultiple<PythonExpression>>,
    ) -> PyResult<Vec<(Vec<usize>, PythonPolynomial)>> {
        if let Some(vv) = vars {
            let mut vars = vec![];

            for vvv in vv.to_iter() {
                let x = self
                    .poly
                    .get_vars_ref()
                    .iter()
                    .position(|v| match (v, vvv.expr.as_view()) {
                        (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                        (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => {
                            f.as_view() == a
                        }
                        _ => false,
                    })
                    .ok_or(exceptions::PyValueError::new_err(format!(
                        "Variable {} not found in polynomial",
                        vvv.__str__()?
                    )))?;

                vars.push(x);
            }

            if vars.is_empty() {
                return Ok(self
                    .poly
                    .into_iter()
                    .map(|t| {
                        (
                            t.exponents.iter().map(|x| *x as usize).collect(),
                            Self {
                                poly: self.poly.constant(t.coefficient.clone()),
                            },
                        )
                    })
                    .collect());
            }

            if vars.len() == 1 {
                return Ok(self
                    .poly
                    .to_univariate_polynomial_list(vars[0])
                    .into_iter()
                    .map(|(f, p)| (vec![p as usize], Self { poly: f }))
                    .collect());
            }

            // sort the exponents wrt the var map
            let mut r: Vec<(Vec<_>, _)> = self
                .poly
                .to_multivariate_polynomial_list(&vars, true)
                .into_iter()
                .map(|(f, p)| {
                    (
                        vars.iter().map(|v| f[*v] as usize).collect(),
                        Self { poly: p },
                    )
                })
                .collect();
            r.sort_by(|a, b| a.0.cmp(&b.0));

            Ok(r)
        } else {
            Ok(self
                .poly
                .into_iter()
                .map(|t| {
                    (
                        t.exponents.iter().map(|x| *x as usize).collect(),
                        Self {
                            poly: self.poly.constant(t.coefficient.clone()),
                        },
                    )
                })
                .collect())
        }
    }

    /// Evaluate the polynomial at point `input`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> P('x*y+2*x+x^2').evaluate([2., 3.])
    ///
    /// Yields `14.0`.
    fn evaluate<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLike1<'py, f64, TypeMustMatch>,
    ) -> PyResult<f64> {
        let input = inputs.as_slice().map_err(|e| {
            exceptions::PyValueError::new_err(format!("Could not convert input to slice: {}", e))
        })?;

        if input.len() != self.poly.get_vars_ref().len() {
            return Err(exceptions::PyValueError::new_err(format!(
                "Expected {} variables, got {}",
                self.poly.get_vars_ref().len(),
                input.len()
            )));
        }

        Ok(self.poly.evaluate(|c| c.to_f64(), input))
    }

    /// Evaluate the polynomial at point `input` with complex input.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> P('x*y+2*x+x^2').evaluate([2+1j, 3+2j])
    ///
    /// Yields `11+13j`.
    fn evaluate_complex<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLike1<'py, Complex64, TypeMustMatch>,
    ) -> PyResult<Complex64> {
        let input = inputs.as_slice().map_err(|e| {
            exceptions::PyValueError::new_err(format!("Could not convert input to slice: {}", e))
        })?;

        if input.len() != self.poly.get_vars_ref().len() {
            return Err(exceptions::PyValueError::new_err(format!(
                "Expected {} variables, got {}",
                self.poly.get_vars_ref().len(),
                input.len()
            )));
        }

        let input = unsafe { std::mem::transmute::<&[Complex64], &[Complex<f64>]>(input) };

        let r = self.poly.evaluate(|c| Complex::new(c.to_f64(), 0.), input);
        Ok(Complex64::new(r.re, r.im))
    }

    /// Replace the variable `x` with a polynomial `v`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial()
    /// >>> r = E('y+1').to_polynomial())
    /// >>> p.replace(x, r)
    pub fn replace(&self, x: PythonExpression, v: PolynomialOrInteger<Self>) -> PyResult<Self> {
        let var: PolyVariable = x
            .expr
            .try_into()
            .map_err(|e| exceptions::PyValueError::new_err(e))?;

        let v = match v {
            PolynomialOrInteger::Polynomial(p) => p,
            PolynomialOrInteger::Integer(i) => Self {
                poly: self.poly.constant(i.into()),
            },
        };

        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|x| x == &var)
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {var} not found in polynomial",
            )))?;

        if self.poly.get_vars_ref() == v.poly.get_vars_ref() {
            Ok(Self {
                poly: self.poly.replace_with_poly(x, &v.poly),
            })
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = v.poly.clone();
            new_self.unify_variables(&mut new_rhs);
            Ok(Self {
                poly: new_self.replace_with_poly(x, &new_rhs),
            })
        }
    }

    /// Parse a polynomial with rational coefficients from a string.
    /// The input must be written in an expanded format and a list of all
    /// the variables must be provided.
    ///
    /// If these requirements are too strict, use `Expression.to_polynomial()` or
    /// `RationalPolynomial.parse()` instead.
    ///
    /// Examples
    /// --------
    /// >>> e = Polynomial.parse('3/4*x^2+y+y*4', ['x', 'y'])
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the input is not a valid Symbolica polynomial.
    #[pyo3(signature = (arg, vars, default_namespace = "python"))]
    #[classmethod]
    pub fn parse(
        _cls: &Bound<'_, PyType>,
        arg: &str,
        vars: Vec<PyBackedStr>,
        default_namespace: &str,
    ) -> PyResult<Self> {
        let mut var_map = vec![];
        let mut var_name_map: SmallVec<[SmartString<LazyCompact>; INLINED_EXPONENTS]> =
            SmallVec::new();

        let namespace = DefaultNamespace {
            namespace: default_namespace.to_string().into(),
            data: "",
            file: "".into(),
            line: 0,
        };

        for v in vars {
            let id = Symbol::new(namespace.attach_namespace(&v)).build().unwrap();
            var_map.push(id.into());
            var_name_map.push((*v).into());
        }

        let e = Token::parse(arg, ParseSettings::polynomial())
            .map_err(exceptions::PyValueError::new_err)?
            .to_polynomial(&Q, &Arc::new(var_map), &var_name_map)
            .map_err(exceptions::PyValueError::new_err)?;

        Ok(Self { poly: e })
    }

    /// Isolate the real roots of the polynomial. The result is a list of intervals with rational bounds that contain exactly one root,
    /// and the multiplicity of that root. Optionally, the intervals can be refined to a given precision.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> p = E('2016+5808*x+5452*x^2+1178*x^3+-753*x^4+-232*x^5+41*x^6').to_polynomial()
    /// >>> for a, b, n in p.isolate_roots():
    /// >>>     print('({},{}): {}'.format(a, b, n))
    ///
    /// yields
    /// ```
    /// (-56/45,-77/62): 1
    /// (-98/79,-119/96): 1
    /// (-119/96,-21/17): 1
    /// (-7/6,0): 1
    /// (0,6): 1
    /// (6,12): 1
    /// ```
    #[pyo3(signature = (refine = None))]
    pub fn isolate_roots(
        &self,
        refine: Option<PythonMultiPrecisionFloat>,
    ) -> PyResult<Vec<(PythonExpression, PythonExpression, usize)>> {
        let refine = refine.map(|x| x.0.to_rational());

        let var = if self.poly.nvars() == 1 {
            0
        } else {
            let degs: Vec<_> = (0..self.poly.nvars())
                .filter(|x| self.poly.degree(*x) > 0)
                .collect();
            if degs.len() > 1 || degs.is_empty() {
                Err(exceptions::PyValueError::new_err(
                    "Polynomial is not univariate",
                ))?
            } else {
                degs[0]
            }
        };

        let uni = self.poly.to_univariate_from_univariate(var);

        Ok(uni
            .isolate_roots(refine)
            .into_iter()
            .map(|(l, r, m)| (Atom::num(l).into(), Atom::num(r).into(), m))
            .collect())
    }

    /// Approximate all complex roots of a univariate polynomial, given a maximal number of iterations
    /// and a given tolerance. Returns the roots and their multiplicity.
    ///
    /// Examples
    /// --------
    ///
    /// >>> p = E('x^10+9x^7+4x^3+2x+1').to_polynomial()
    /// >>> for (r, m) in p.approximate_roots(1000, 1e-10):
    /// >>>     print(r, m)
    pub fn approximate_roots<'py>(
        &self,
        max_iterations: usize,
        tolerance: f64,
        py: Python<'py>,
    ) -> PyResult<Vec<(Bound<'py, PyComplex>, usize)>> {
        let var = if self.poly.nvars() == 1 {
            0
        } else {
            let degs: Vec<_> = (0..self.poly.nvars())
                .filter(|x| self.poly.degree(*x) > 0)
                .collect();
            if degs.len() > 1 || degs.is_empty() {
                Err(exceptions::PyValueError::new_err(
                    "Polynomial is not univariate",
                ))?
            } else {
                degs[0]
            }
        };

        let uni = self.poly.to_univariate_from_univariate(var);

        Ok(uni
            .approximate_roots::<F64>(max_iterations, &tolerance.into())
            .unwrap_or_else(|e| e)
            .into_iter()
            .map(|(r, p)| (PyComplex::from_doubles(py, r.re.to_f64(), r.im.to_f64()), p))
            .collect())
    }

    /// Convert the coefficients of the polynomial to a finite field with prime `prime`.
    pub fn to_finite_field(&self, prime: u64) -> PythonFiniteFieldPolynomial {
        let f = Zp64::new(prime);
        PythonFiniteFieldPolynomial {
            poly: self.poly.map_coeff(|c| c.to_finite_field(&f), f.clone()),
        }
    }

    /// Compute the Groebner basis of a polynomial system.
    ///
    /// If `grevlex=True`, reverse graded lexicographical ordering is used,
    /// otherwise the ordering is lexicographical.
    ///
    /// If `print_stats=True` intermediate statistics will be printed.
    ///
    /// Examples
    /// --------
    /// >>> basis = Polynomial.groebner_basis(
    /// >>>     [E("a b c d - 1").to_polynomial(),
    /// >>>      E("a b c + a b d + a c d + b c d").to_polynomial(),
    /// >>>      E("a b + b c + a d + c d").to_polynomial(),
    /// >>>      E("a + b + c + d").to_polynomial()],
    /// >>>     grevlex=True,
    /// >>>     print_stats=True
    /// >>> )
    /// >>> for p in basis:
    /// >>>     print(p)
    #[pyo3(signature = (system, grevlex = true, print_stats = false))]
    #[classmethod]
    pub fn groebner_basis(
        _cls: &Bound<'_, PyType>,
        system: Vec<Self>,
        grevlex: bool,
        print_stats: bool,
    ) -> Vec<Self> {
        if grevlex {
            let grevlex_ideal: Vec<_> = system
                .iter()
                .map(|p| p.poly.reorder::<GrevLexOrder>())
                .collect();
            let gb = GroebnerBasis::new(&grevlex_ideal, print_stats);

            gb.system
                .into_iter()
                .map(|p| Self {
                    poly: p.reorder::<LexOrder>(),
                })
                .collect()
        } else {
            let ideal: Vec<_> = system.into_iter().map(|p| p.poly).collect();
            let gb = GroebnerBasis::new(&ideal, print_stats);
            gb.system.into_iter().map(|p| Self { poly: p }).collect()
        }
    }

    /// Completely reduce the polynomial w.r.t the polynomials `gs`.
    /// For example reducing `f=y^2+x` by `g=[x]` yields `y^2`.
    #[pyo3(signature = (system, grevlex = true))]
    pub fn reduce(&self, system: Vec<Self>, grevlex: bool) -> Self {
        if grevlex {
            let p = self.poly.reorder::<GrevLexOrder>();
            let grevlex_ideal: Vec<_> = system
                .iter()
                .map(|p| p.poly.reorder::<GrevLexOrder>())
                .collect();

            let r = p.reduce(&grevlex_ideal);
            Self {
                poly: r.reorder::<LexOrder>(),
            }
        } else {
            let ideal: Vec<_> = system.into_iter().map(|p| p.poly).collect();
            Self {
                poly: self.poly.reduce(&ideal),
            }
        }
    }

    /// Integrate the polynomial in `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x^2+2').to_polynomial()
    /// >>> print(p.integrate(x))
    pub fn integrate(&self, x: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, x.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        Ok(Self {
            poly: self.poly.integrate(x),
        })
    }

    /// Convert the polynomial to an expression.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> e = E('x*y+2*x+x^2')
    /// >>> p = e.to_polynomial()
    /// >>> print(e - p.to_expression())
    pub fn to_expression(&self) -> PyResult<PythonExpression> {
        Ok(self.poly.to_expression().into())
    }

    /// Perform Newton interpolation in the variable `x` given the sample points
    /// `sample_points` and the values `values`.
    ///
    /// Examples
    /// --------
    /// >>> x, y = S('x', 'y')
    /// >>> a = Polynomial.interpolate(
    /// >>>         x, [4, 5], [(y**2+5).to_polynomial(), (y**3).to_polynomial()])
    /// >>> print(a)
    ///
    /// yields `25-5*x+5*y^2-y^2*x-4*y^3+y^3*x`.
    #[classmethod]
    pub fn interpolate(
        _cls: &Bound<'_, PyType>,
        x: PythonExpression,
        sample_points: Vec<ConvertibleToExpression>,
        values: Vec<PythonPolynomial>,
    ) -> PyResult<Self> {
        if values.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "Values must be provided".to_string(),
            ));
        }

        if sample_points.len() != values.len() {
            return Err(exceptions::PyValueError::new_err(
                "Sample points and values must have the same length".to_string(),
            ));
        }

        let var = x
            .expr
            .try_into()
            .map_err(|e| exceptions::PyValueError::new_err(e))?;

        let sample_points: Vec<Rational> = sample_points
            .into_iter()
            .map(|x| {
                if let AtomView::Num(x) = x.to_expression().expr.as_view() {
                    match x.get_coeff_view() {
                        CoefficientView::Natural(r, d, 0, 1) => {
                            Ok(Rational::from_int_unchecked(r, d))
                        }
                        CoefficientView::Large(r, i) => {
                            if i.is_zero() {
                                Ok(r.to_rat())
                            } else {
                                Err(exceptions::PyValueError::new_err(
                                    "Sample points must be rational numbers".to_string(),
                                ))
                            }
                        }
                        _ => Err(exceptions::PyValueError::new_err(
                            "Sample points must be rational numbers".to_string(),
                        )),
                    }
                } else {
                    Err(exceptions::PyValueError::new_err(
                        "Sample points must be rational numbers".to_string(),
                    ))?
                }
            })
            .collect::<Result<_, _>>()?;

        let mut values: Vec<_> = values.into_iter().map(|x| x.poly).collect();

        // add the variable to all the polynomials
        for v in &mut values {
            v.add_variable(&var);
        }

        MultivariatePolynomial::unify_variables_list(&mut values);

        // find the index of the variable
        let index = values[0]
            .get_vars_ref()
            .iter()
            .position(|v| v == &var)
            .unwrap();

        Ok(Self {
            poly: MultivariatePolynomial::newton_interpolation(&sample_points, &values, index),
        })
    }

    /// Convert the polynomial to a number field defined by the minimal polynomial `minimal_poly`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> a = P('a').to_number_field(P('a^2-2'))
    /// >>> print(a * a)
    ///
    /// Yields `2`.
    pub fn to_number_field(&self, minimal_poly: Self) -> PyResult<PythonNumberFieldPolynomial> {
        let a = AlgebraicExtension::new(minimal_poly.poly.clone());
        let poly_nf = self.poly.to_number_field(&a);

        Ok(PythonNumberFieldPolynomial { poly: poly_nf })
    }

    /// Adjoin the coefficient ring of this polynomial `R[a]` with `b`, whose minimal polynomial
    /// is `R[a][b]` and form `R[b]`. Also return the new representation of `a` and `b`.
    ///
    /// `b`  must be irreducible over `R` and `R[a]`; this is not checked.
    ///
    /// If `new_symbol` is provided, the variable of the new extension will be renamed to it.
    /// Otherwise, the variable of the new extension will be the same as that of `b`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> sqrt2 = P('a^2-2')
    /// >>> sqrt23 = P('b^2-a-3')
    /// >>> (min_poly, rep2, rep23) = sqrt2.adjoin(sqrt23)
    /// >>>
    /// >>> # convert to number field
    /// >>> a = P('a^2+b').replace(S('a'), rep2).replace(S('b'), rep23).to_number_field(min_poly)
    #[pyo3(signature = (b, new_symbol = None))]
    pub fn adjoin(
        &self,
        b: Self,
        new_symbol: Option<PolyVariable>,
    ) -> PyResult<(PythonPolynomial, PythonPolynomial, PythonPolynomial)> {
        let a = AlgebraicExtension::new(self.poly.clone());
        let bb = b.poly.to_number_field(&a);

        let (new_field, map1, map2) =
            AlgebraicExtension::new(self.poly.clone()).adjoin(&bb, new_symbol);

        Ok((
            Self {
                poly: new_field.poly().clone(),
            },
            PythonPolynomial {
                poly: map1.poly().clone(),
            },
            PythonPolynomial {
                poly: map2.poly().clone(),
            },
        ))
    }

    /// Find the minimal polynomial for the algebraic number represented by this polynomial
    /// expressed in the number field defined by `minimal_poly`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> (min_poly, rep2, rep23) = P('a^2-2').adjoin(P('b^2-3'))
    /// >>> rep2.simplify_algebraic_number(min_poly)
    ///
    /// Yields `b^2-2`.
    pub fn simplify_algebraic_number(&self, minimal_poly: Self) -> PyResult<Self> {
        let a = AlgebraicExtension::new(minimal_poly.poly);
        let m = a.try_to_element(self.poly.clone()).map_err(|e| {
            exceptions::PyValueError::new_err(format!(
                "Could not convert polynomial to algebraic number: {}",
                e
            ))
        })?;
        let poly_nf = a.simplify(&m).poly().clone();

        Ok(Self { poly: poly_nf })
    }
}

/// A Symbolica polynomial over finite fields.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "FiniteFieldPolynomial", subclass, module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonFiniteFieldPolynomial {
    pub poly: MultivariatePolynomial<Zp64, u16>,
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(&mut PythonFiniteFieldPolynomial = PythonFiniteFieldPolynomial);

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonFiniteFieldPolynomial {
    /// Compare two polynomials.
    fn __richcmp__(&self, o: Py<PyAny>, op: CompareOp, py: Python) -> PyResult<bool> {
        let Ok(other) = o.extract::<PolynomialOrInteger<PythonFiniteFieldPolynomial>>(py) else {
            return Err(exceptions::PyTypeError::new_err(
                "Can only compare Polynomial with Polynomial or integer.",
            ));
        };
        match op {
            CompareOp::Eq => match other {
                PolynomialOrInteger::Integer(i) => Ok(self.poly.is_constant()
                    && self.poly.get_constant() == self.poly.ring.element_from_integer(i)),
                PolynomialOrInteger::Polynomial(p) => Ok(self.poly == p.poly),
            },
            CompareOp::Ne => match other {
                PolynomialOrInteger::Integer(i) => Ok(!self.poly.is_constant()
                    || self.poly.get_constant() != self.poly.ring.element_from_integer(i)),
                PolynomialOrInteger::Polynomial(p) => Ok(self.poly != p.poly),
            },
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "Inequalities between polynomials are not allowed in {} {} {}",
                self.__str__()?,
                match op {
                    CompareOp::Eq => "==",
                    CompareOp::Ge => ">=",
                    CompareOp::Gt => ">",
                    CompareOp::Le => "<=",
                    CompareOp::Lt => "<",
                    CompareOp::Ne => "!=",
                },
                match other {
                    PolynomialOrInteger::Integer(i) => i.to_string(),
                    PolynomialOrInteger::Polynomial(p) => p.__str__()?,
                }
            ))),
        }
    }

    /// Copy the polynomial.
    pub fn __copy__(&self) -> Self {
        Self {
            poly: self.poly.clone(),
        }
    }

    /// Convert the polynomial into a human-readable string, with tunable settings.
    ///
    /// Examples
    /// --------
    /// >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
    /// >>> print(p.format(symmetric_representation_for_finite_field=True))
    #[pyo3(signature =
        (mode = PythonPrintMode::Symbolica,
            terms_on_new_line = false,
            color_top_level_sum = true,
            color_builtin_symbols = true,
            print_ring = true,
            symmetric_representation_for_finite_field = false,
            explicit_rational_polynomial = false,
            number_thousands_separator = None,
            multiplication_operator = '*',
            double_star_for_exponentiation = false,
            square_brackets_for_function = false,
            num_exp_as_superscript = true,
            precision = None,
            show_namespaces = false,
            include_attributes = false,
            max_terms = None,
            custom_print_mode = None)
        )]
    pub fn format(
        &self,
        mode: PythonPrintMode,
        terms_on_new_line: bool,
        color_top_level_sum: bool,
        color_builtin_symbols: bool,
        print_ring: bool,
        symmetric_representation_for_finite_field: bool,
        explicit_rational_polynomial: bool,
        number_thousands_separator: Option<char>,
        multiplication_operator: char,
        double_star_for_exponentiation: bool,
        square_brackets_for_function: bool,
        num_exp_as_superscript: bool,
        precision: Option<usize>,
        show_namespaces: bool,
        include_attributes: bool,
        max_terms: Option<usize>,
        custom_print_mode: Option<usize>,
    ) -> PyResult<String> {
        Ok(self.poly.format_string(
            &PrintOptions {
                terms_on_new_line,
                color_top_level_sum,
                color_builtin_symbols,
                print_ring,
                symmetric_representation_for_finite_field,
                explicit_rational_polynomial,
                number_thousands_separator,
                multiplication_operator,
                double_star_for_exponentiation,
                square_brackets_for_function,
                num_exp_as_superscript,
                mode: mode.into(),
                precision,
                pretty_matrix: false,
                hide_all_namespaces: !show_namespaces,
                color_namespace: true,
                hide_namespace: Some("python"),
                include_attributes,
                max_terms,
                custom_print_mode: custom_print_mode.map(|x| ("default", x)),
            },
            PrintState::new(),
        ))
    }

    /// Convert the polynomial into a portable string.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new()))
    }

    /// Print the polynomial in a human-readable format.
    pub fn __str__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&DEFAULT_PRINT_OPTIONS, PrintState::new()))
    }

    pub fn __pow__(
        &self,
        exponent: usize,
        modulo: Option<i64>,
    ) -> PyResult<PythonFiniteFieldPolynomial> {
        if modulo.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "Optional number argument not supported",
            ));
        }

        Ok(Self {
            poly: self.poly.pow(exponent),
        })
    }

    /// Convert the polynomial into a LaTeX string.
    pub fn to_latex(&self) -> PyResult<String> {
        Ok(format!(
            "$${}$$",
            self.poly
                .format_string(&LATEX_PRINT_OPTIONS, PrintState::new())
        ))
    }

    /// Get the number of terms.
    pub fn nterms(&self) -> usize {
        self.poly.nterms()
    }

    /// Get the list of variables in the internal ordering of the polynomial.
    pub fn get_variables(&self) -> PyResult<Vec<PythonExpression>> {
        let mut var_list = vec![];

        for x in self.poly.get_vars_ref() {
            match x {
                PolyVariable::Symbol(x) => {
                    var_list.push(Atom::var(*x).into());
                }
                PolyVariable::Temporary(_) => {
                    Err(exceptions::PyValueError::new_err(
                        "Temporary variable in polynomial".to_string(),
                    ))?;
                }
                PolyVariable::Function(_, a) | PolyVariable::Power(a) => {
                    var_list.push(a.as_ref().clone().into());
                }
            }
        }

        Ok(var_list)
    }

    /// Add two polynomials `self and `rhs`, returning the result.
    pub fn __add__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly + &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .add_constant(self.poly.ring.element_from_integer(i)),
            }),
        }
    }

    /// Subtract polynomials `rhs` from `self`, returning the result.
    pub fn __sub__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly - &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .add_constant(self.poly.ring.neg(&self.poly.ring.element_from_integer(i))),
            }),
        }
    }

    /// Multiply two polynomials `self and `rhs`, returning the result.
    pub fn __mul__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly * &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .mul_coeff(self.poly.ring.element_from_integer(i)),
            }),
        }
    }

    pub fn __radd__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        self.__add__(rhs)
    }

    pub fn __rsub__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &p.poly - &self.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .neg()
                    .add_constant(self.poly.ring.element_from_integer(i)),
            }),
        }
    }

    pub fn __rmul__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        self.__mul__(rhs)
    }

    pub fn __floordiv__(&self, rhs: Self) -> PyResult<Self> {
        if rhs.poly.is_zero() {
            return Err(exceptions::PyValueError::new_err("Division by zero"));
        }

        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        };

        let (q, _r) = self.poly.quot_rem(&rhs.poly, false);

        Ok(Self { poly: q })
    }

    /// Divide the polynomial `self` by `rhs` if possible, returning the result.
    pub fn __truediv__(&self, rhs: Self) -> PyResult<Self> {
        if rhs.poly.is_zero() {
            return Err(exceptions::PyValueError::new_err("Division by zero"));
        }
        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        };

        let (q, r) = self.poly.quot_rem(&rhs.poly, false);

        if r.is_zero() {
            Ok(Self { poly: q })
        } else {
            Err(exceptions::PyValueError::new_err(format!(
                "The division has a remainder: {}",
                r
            )))
        }
    }

    pub fn unify_variables(&mut self, other: &mut Self) {
        self.poly.unify_variables(&mut other.poly);
    }

    pub fn __contains__(&self, var: &PythonExpression) -> bool {
        self.contains(var)
    }

    pub fn contains(&self, var: &PythonExpression) -> bool {
        if let Some(p) =
            self.poly
                .get_vars_ref()
                .iter()
                .position(|v| match (v, var.expr.as_view()) {
                    (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                    (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                    _ => false,
                })
        {
            self.poly.contains(p)
        } else {
            false
        }
    }

    pub fn degree(&self, var: &PythonExpression) -> PyResult<isize> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, var.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                var.__str__()?
            )))?;

        Ok(self.poly.degree(x) as isize)
    }

    /// Set a new variable ordering for the polynomial.
    /// This can be used to introduce new variables as well.
    pub fn reorder(&mut self, order: Vec<PythonExpression>) -> PyResult<()> {
        let vars: Vec<_> = order
            .into_iter()
            .map(|x| x.expr.try_into())
            .collect::<Result<_, _>>()
            .map_err(|e| exceptions::PyValueError::new_err(e))?;

        self.poly = self
            .poly
            .rearrange_with_growth(&vars)
            .map_err(exceptions::PyValueError::new_err)?;
        Ok(())
    }

    /// Divide `self` by `rhs`, returning the quotient and remainder.
    pub fn quot_rem(
        &self,
        rhs: Self,
    ) -> PyResult<(PythonFiniteFieldPolynomial, PythonFiniteFieldPolynomial)> {
        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        };

        if rhs.poly.is_zero() {
            Err(exceptions::PyValueError::new_err("Division by zero"))
        } else {
            let (q, r) = self.poly.quot_rem(&rhs.poly, false);
            Ok((Self { poly: q }, Self { poly: r }))
        }
    }

    /// Negate the polynomial.
    pub fn __neg__(&self) -> Self {
        Self {
            poly: self.poly.clone().neg(),
        }
    }

    /// Compute the remainder `self % rhs.
    pub fn __mod__(&self, rhs: Self) -> PyResult<Self> {
        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        };

        if rhs.poly.is_zero() {
            Err(exceptions::PyValueError::new_err("Division by zero"))
        } else {
            Ok(Self {
                poly: self.poly.rem(&rhs.poly),
            })
        }
    }

    /// Compute the greatest common divisor (GCD) of two or more polynomials.
    #[pyo3(signature = (*rhs))]
    pub fn gcd(
        &self,
        #[gen_stub(override_type(type_repr = "FiniteFieldPolynomial"))] rhs: &Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
        if rhs.len() == 1 {
            let rhs = rhs.get_item(0)?.extract::<Self>()?;
            if self.poly.ring != rhs.poly.ring {
                Err(exceptions::PyValueError::new_err(
                    "Polynomials have different rings".to_string(),
                ))
            } else {
                Ok(Self {
                    poly: self.poly.gcd(&rhs.poly),
                })
            }
        } else {
            let mut args = vec![self.poly.clone()];
            for r in rhs.iter() {
                let p = r.extract::<Self>()?;
                if args[0].ring != p.poly.ring {
                    return Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ));
                }
                args.push(p.poly);
            }

            Ok(Self {
                poly: PolynomialGCD::gcd_multiple(args),
            })
        }
    }

    /// Compute the extended GCD of two polynomials, yielding the GCD and the Bezout coefficients `s` and `t`
    /// such that `self * s + rhs * t = gcd(self, rhs)`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> E('(1+x)(20+x)').to_polynomial(modulus=5).extended_gcd(E('x^2+2').to_polynomial(modulus=5))
    ///
    /// yields `(1, 3+4*x, 3+x)`.
    pub fn extended_gcd(
        &self,
        rhs: Self,
    ) -> PyResult<(
        PythonFiniteFieldPolynomial,
        PythonFiniteFieldPolynomial,
        PythonFiniteFieldPolynomial,
    )> {
        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        }

        if self.poly.variables != rhs.poly.variables
            || (0..self.poly.nvars())
                .filter(|i| self.poly.degree(*i) > 0 || rhs.poly.degree(*i) > 0)
                .count()
                > 1
        {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials are not univariate in the same variable".to_string(),
            ));
        }

        let (g, s, t) = self.poly.eea_univariate(&rhs.poly);
        Ok((Self { poly: g }, Self { poly: s }, Self { poly: t }))
    }

    /// Convert the finite field polynomial to a polynomial with integer coefficients.
    #[pyo3(signature = (symmetric_representation = true))]
    pub fn to_integer_polynomial(&self, symmetric_representation: bool) -> PythonPolynomial {
        PythonPolynomial {
            poly: if symmetric_representation {
                self.poly
                    .map_coeff(|x| self.poly.ring.to_symmetric_integer(x).into(), Q)
            } else {
                self.poly
                    .map_coeff(|x| self.poly.ring.to_integer(x).into(), Q)
            },
        }
    }

    /// Compute the resultant of two polynomials with respect to the variable `var`.
    pub fn resultant(&self, rhs: Self, var: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, var.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                var.__str__()?
            )))?;

        if self.poly.get_vars_ref() == rhs.poly.get_vars_ref() {
            let self_uni = self.poly.to_univariate(x);
            let rhs_uni = rhs.poly.to_univariate(x);

            Ok(Self {
                poly: self_uni.resultant_prs(&rhs_uni),
            })
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);

            let self_uni = new_self.to_univariate(x);
            let rhs_uni = new_rhs.to_univariate(x);

            Ok(Self {
                poly: self_uni.resultant_prs(&rhs_uni),
            })
        }
    }

    /// Compute the square-free factorization of the polynomial.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial()
    /// >>> print('Square-free factorization of {}:'.format(p))
    /// >>> for f, exp in p.factor_square_free():
    /// >>>     print('\t({})^{}'.format(f, exp))
    pub fn factor_square_free(&self) -> Vec<(PythonFiniteFieldPolynomial, usize)> {
        self.poly
            .square_free_factorization()
            .into_iter()
            .map(|(f, p)| (Self { poly: f }, p))
            .collect()
    }

    /// Factorize the polynomial.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial()
    /// >>> print('Factorization of {}:'.format(p))
    /// >>> for f, exp in p.factor():
    /// >>>     print('\t({})^{}'.format(f, exp))
    pub fn factor(&self) -> Vec<(PythonFiniteFieldPolynomial, usize)> {
        self.poly
            .factor()
            .into_iter()
            .map(|(f, p)| (Self { poly: f }, p))
            .collect()
    }

    /// Take a derivative in `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x^2+2').to_polynomial()
    /// >>> print(p.derivative(x))
    pub fn derivative(&self, x: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, x.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        Ok(Self {
            poly: self.poly.derivative(x),
        })
    }

    pub fn get_modulus(&self) -> u64 {
        self.poly.ring.get_prime()
    }

    /// Make the polynomial monic, i.e., the polynomial
    /// with a leading coefficient of 1.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('3x^2+6x+9').to_polynomial().monic()
    /// >>> print(p)
    ///
    /// Yields `x^2+2*x+3`.
    pub fn monic(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.clone().make_monic(),
        })
    }

    /// Get the leading coefficient.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> p = E('3x^2+6x+9').to_polynomial().lcoeff()
    /// >>> print(p)
    ///
    /// Yields `3`.
    pub fn lcoeff(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.constant(self.poly.lcoeff().clone()),
        })
    }

    /// Get the coefficient list, optionally in the variables `vars`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial()
    /// >>> for n, pp in p.coefficient_list(x):
    /// >>>     print(n, pp)
    #[pyo3(signature = (vars = None))]
    pub fn coefficient_list(
        &self,
        vars: Option<OneOrMultiple<PythonExpression>>,
    ) -> PyResult<Vec<(Vec<usize>, PythonFiniteFieldPolynomial)>> {
        if let Some(vv) = vars {
            let mut vars = vec![];

            for vvv in vv.to_iter() {
                let x = self
                    .poly
                    .get_vars_ref()
                    .iter()
                    .position(|v| match (v, vvv.expr.as_view()) {
                        (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                        (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => {
                            f.as_view() == a
                        }
                        _ => false,
                    })
                    .ok_or(exceptions::PyValueError::new_err(format!(
                        "Variable {} not found in polynomial",
                        vvv.__str__()?
                    )))?;

                vars.push(x);
            }

            if vars.is_empty() {
                return Ok(self
                    .poly
                    .into_iter()
                    .map(|t| {
                        (
                            t.exponents.iter().map(|x| *x as usize).collect(),
                            Self {
                                poly: self.poly.constant(*t.coefficient),
                            },
                        )
                    })
                    .collect());
            }

            if vars.len() == 1 {
                return Ok(self
                    .poly
                    .to_univariate_polynomial_list(vars[0])
                    .into_iter()
                    .map(|(f, p)| (vec![p as usize], Self { poly: f }))
                    .collect());
            }

            // sort the exponents wrt the var map
            let mut r: Vec<(Vec<_>, _)> = self
                .poly
                .to_multivariate_polynomial_list(&vars, true)
                .into_iter()
                .map(|(f, p)| {
                    (
                        vars.iter().map(|v| f[*v] as usize).collect(),
                        Self { poly: p },
                    )
                })
                .collect();
            r.sort_by(|a, b| a.0.cmp(&b.0));

            Ok(r)
        } else {
            Ok(self
                .poly
                .into_iter()
                .map(|t| {
                    (
                        t.exponents.iter().map(|x| *x as usize).collect(),
                        Self {
                            poly: self.poly.constant(*t.coefficient),
                        },
                    )
                })
                .collect())
        }
    }

    /// Evaluate the polynomial at the given values.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> x, y = S('x', 'y')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial(modulus=5)
    /// >>> print(p.evaluate([2, 3]))
    /// 4
    pub fn evaluate(&self, values: Vec<Integer>) -> PyResult<Integer> {
        if values.len() != self.poly.get_vars_ref().len() {
            return Err(exceptions::PyValueError::new_err(format!(
                "Expected {} values, got {}",
                self.poly.get_vars_ref().len(),
                values.len()
            )));
        }

        let input = values
            .into_iter()
            .map(|x| self.poly.ring.element_from_integer(x))
            .collect::<Vec<_>>();

        let r = self.poly.replace_all(&input);

        Ok(self.poly.ring.to_integer(&r))
    }

    /// Replace the variable `x` with a polynomial `v`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial()
    /// >>> r = E('y+1').to_polynomial())
    /// >>> p.replace(x, r)
    pub fn replace(&self, x: PythonExpression, v: PolynomialOrInteger<Self>) -> PyResult<Self> {
        let id = match x.expr.as_view() {
            AtomView::Var(x) => x.get_symbol(),
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Derivative must be taken wrt a variable",
                ));
            }
        };

        let v = match v {
            PolynomialOrInteger::Polynomial(p) => p.poly,
            PolynomialOrInteger::Integer(i) => {
                self.poly.constant(self.poly.ring.element_from_integer(i))
            }
        };

        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|x| match x {
                PolyVariable::Symbol(y) => *y == id,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        if self.poly.get_vars_ref() == v.get_vars_ref() {
            Ok(Self {
                poly: self.poly.replace_with_poly(x, &v),
            })
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = v.clone();
            new_self.unify_variables(&mut new_rhs);
            Ok(Self {
                poly: new_self.replace_with_poly(x, &new_rhs),
            })
        }
    }

    /// Compute the Groebner basis of a polynomial system.
    ///
    /// If `grevlex=True`, reverse graded lexicographical ordering is used,
    /// otherwise the ordering is lexicographical.
    ///
    /// If `print_stats=True` intermediate statistics will be printed.
    #[pyo3(signature = (system, grevlex = true, print_stats = false))]
    #[classmethod]
    pub fn groebner_basis(
        _cls: &Bound<'_, PyType>,
        system: Vec<Self>,
        grevlex: bool,
        print_stats: bool,
    ) -> Vec<Self> {
        if grevlex {
            let grevlex_ideal: Vec<_> = system
                .iter()
                .map(|p| p.poly.reorder::<GrevLexOrder>())
                .collect();
            let gb = GroebnerBasis::new(&grevlex_ideal, print_stats);

            gb.system
                .into_iter()
                .map(|p| Self {
                    poly: p.reorder::<LexOrder>(),
                })
                .collect()
        } else {
            let ideal: Vec<_> = system.iter().map(|p| p.poly.clone()).collect();
            let gb = GroebnerBasis::new(&ideal, print_stats);
            gb.system.into_iter().map(|p| Self { poly: p }).collect()
        }
    }

    /// Completely reduce the polynomial w.r.t the polynomials `gs`.
    /// For example reducing `f=y^2+x` by `g=[x]` yields `y^2`.
    #[pyo3(signature = (system, grevlex = true))]
    pub fn reduce(&self, system: Vec<Self>, grevlex: bool) -> Self {
        if grevlex {
            let p = self.poly.reorder::<GrevLexOrder>();
            let grevlex_ideal: Vec<_> = system
                .iter()
                .map(|p| p.poly.reorder::<GrevLexOrder>())
                .collect();

            let r = p.reduce(&grevlex_ideal);
            Self {
                poly: r.reorder::<LexOrder>(),
            }
        } else {
            let ideal: Vec<_> = system.into_iter().map(|p| p.poly).collect();
            Self {
                poly: self.poly.reduce(&ideal),
            }
        }
    }

    /// Integrate the polynomial in `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x^2+2').to_polynomial()
    /// >>> print(p.integrate(x))
    pub fn integrate(&self, x: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, x.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        Ok(Self {
            poly: self.poly.integrate(x),
        })
    }

    /// Parse a polynomial with integer coefficients from a string.
    /// The input must be written in an expanded format and a list of all
    /// the variables must be provided.
    ///
    /// If these requirements are too strict, use `Expression.to_polynomial()` or
    /// `RationalPolynomial.parse()` instead.
    ///
    /// Examples
    /// --------
    /// >>> e = Polynomial.parse('3*x^2+y+y*4', ['x', 'y'], 5)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the input is not a valid Symbolica polynomial.
    #[pyo3(signature = (arg, vars, prime, default_namespace = "python"))]
    #[classmethod]
    pub fn parse(
        _cls: &Bound<'_, PyType>,
        arg: &str,
        vars: Vec<PyBackedStr>,
        prime: u64,
        default_namespace: &str,
    ) -> PyResult<Self> {
        let mut var_map = vec![];
        let mut var_name_map = vec![];

        let namespace = DefaultNamespace {
            namespace: default_namespace.to_string().into(),
            data: "",
            file: "".into(),
            line: 0,
        };

        for v in vars {
            let id = Symbol::new(namespace.attach_namespace(&v)).build().unwrap();
            var_map.push(id.into());
            var_name_map.push((*v).into());
        }

        let e = Token::parse(arg, ParseSettings::polynomial())
            .map_err(exceptions::PyValueError::new_err)?
            .to_polynomial(&Zp64::new(prime), &Arc::new(var_map), &var_name_map)
            .map_err(exceptions::PyValueError::new_err)?;

        Ok(Self { poly: e })
    }

    /// Convert the polynomial to an expression.
    pub fn to_expression(&self) -> PyResult<PythonExpression> {
        let p = self
            .poly
            .map_coeff(|x| self.poly.ring.to_symmetric_integer(x), Z);

        Ok(p.to_expression().into())
    }

    /// Convert the polynomial to a Galois field defined by the minimal polynomial `minimal_poly`.
    pub fn to_galois_field(&self, minimal_poly: Self) -> PyResult<PythonGaloisFieldPolynomial> {
        if self.poly.ring != minimal_poly.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different moduli".to_string(),
            ));
        }

        let a = AlgebraicExtension::new(minimal_poly.poly.clone());
        let poly_nf = self.poly.to_number_field(&a);

        Ok(PythonGaloisFieldPolynomial { poly: poly_nf })
    }

    /// Adjoin the coefficient ring of this polynomial `R[a]` with `b`, whose minimal polynomial
    /// is `R[a][b]` and form `R[b]`. Also return the new representation of `a` and `b`.
    ///
    /// `b`  must be irreducible over `R` and `R[a]`; this is not checked.
    ///
    /// If `new_symbol` is provided, the variable of the new extension will be renamed to it.
    /// Otherwise, the variable of the new extension will be the same as that of `b`.
    #[pyo3(signature = (b, new_symbol = None))]
    pub fn adjoin(
        &self,
        b: Self,
        new_symbol: Option<PolyVariable>,
    ) -> PyResult<(
        PythonFiniteFieldPolynomial,
        PythonFiniteFieldPolynomial,
        PythonFiniteFieldPolynomial,
    )> {
        if self.poly.ring != b.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different moduli".to_string(),
            ));
        }

        let a = AlgebraicExtension::new(self.poly.clone());
        let bb = b.poly.to_number_field(&a);

        let (new_field, map1, map2) =
            AlgebraicExtension::new(self.poly.clone()).adjoin(&bb, new_symbol);

        Ok((
            Self {
                poly: new_field.poly().clone(),
            },
            Self {
                poly: map1.poly().clone(),
            },
            Self {
                poly: map2.poly().clone(),
            },
        ))
    }

    /// Find the minimal polynomial for the algebraic number represented by this polynomial
    /// expressed in the number field defined by `minimal_poly`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> (min_poly, rep2, rep23) = P('a^2-2').adjoin(P('b^2-3'))
    /// >>> rep2.simplify_algebraic_number(min_poly)
    ///
    /// Yields `b^2-2`.
    pub fn simplify_algebraic_number(&self, minimal_poly: Self) -> PyResult<Self> {
        let a = AlgebraicExtension::new(minimal_poly.poly);
        let m = a.try_to_element(self.poly.clone()).map_err(|e| {
            exceptions::PyValueError::new_err(format!(
                "Could not convert polynomial to algebraic number: {}",
                e
            ))
        })?;
        let poly_nf = a.simplify(&m).poly().clone();

        Ok(Self { poly: poly_nf })
    }
}

/// A Symbolica polynomial over Galois fields.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "PrimeTwoPolynomial", subclass, module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonPrimeTwoPolynomial {
    pub poly: MultivariatePolynomial<Z2, u16>,
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(&mut PythonPrimeTwoPolynomial = PythonPrimeTwoPolynomial);

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonPrimeTwoPolynomial {
    /// Compare two polynomials.
    fn __richcmp__(&self, o: Py<PyAny>, op: CompareOp, py: Python) -> PyResult<bool> {
        let Ok(other) = o.extract::<PolynomialOrInteger<PythonPrimeTwoPolynomial>>(py) else {
            return Err(exceptions::PyTypeError::new_err(
                "Can only compare Polynomial with Polynomial or integer.",
            ));
        };
        match op {
            CompareOp::Eq => match other {
                PolynomialOrInteger::Integer(i) => Ok(self.poly.is_constant()
                    && self.poly.get_constant() == self.poly.ring.element_from_integer(i)),
                PolynomialOrInteger::Polynomial(p) => Ok(self.poly == p.poly),
            },
            CompareOp::Ne => match other {
                PolynomialOrInteger::Integer(i) => Ok(!self.poly.is_constant()
                    || self.poly.get_constant() != self.poly.ring.element_from_integer(i)),
                PolynomialOrInteger::Polynomial(p) => Ok(self.poly != p.poly),
            },
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "Inequalities between polynomials are not allowed in {} {} {}",
                self.__str__()?,
                match op {
                    CompareOp::Eq => "==",
                    CompareOp::Ge => ">=",
                    CompareOp::Gt => ">",
                    CompareOp::Le => "<=",
                    CompareOp::Lt => "<",
                    CompareOp::Ne => "!=",
                },
                match other {
                    PolynomialOrInteger::Integer(i) => i.to_string(),
                    PolynomialOrInteger::Polynomial(p) => p.__str__()?,
                }
            ))),
        }
    }

    /// Copy the polynomial.
    pub fn __copy__(&self) -> Self {
        Self {
            poly: self.poly.clone(),
        }
    }

    /// Convert the polynomial into a human-readable string, with tunable settings.
    ///
    /// Examples
    /// --------
    /// >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
    /// >>> print(p.format(symmetric_representation_for_finite_field=True))
    #[pyo3(signature =
        (mode = PythonPrintMode::Symbolica,
            terms_on_new_line = false,
            color_top_level_sum = true,
            color_builtin_symbols = true,
            print_ring = true,
            symmetric_representation_for_finite_field = false,
            explicit_rational_polynomial = false,
            number_thousands_separator = None,
            multiplication_operator = '*',
            double_star_for_exponentiation = false,
            square_brackets_for_function = false,
            num_exp_as_superscript = true,
            precision = None,
            show_namespaces = false,
            include_attributes = false,
            max_terms = None,
            custom_print_mode = None)
        )]
    pub fn format(
        &self,
        mode: PythonPrintMode,
        terms_on_new_line: bool,
        color_top_level_sum: bool,
        color_builtin_symbols: bool,
        print_ring: bool,
        symmetric_representation_for_finite_field: bool,
        explicit_rational_polynomial: bool,
        number_thousands_separator: Option<char>,
        multiplication_operator: char,
        double_star_for_exponentiation: bool,
        square_brackets_for_function: bool,
        num_exp_as_superscript: bool,
        precision: Option<usize>,
        show_namespaces: bool,
        include_attributes: bool,
        max_terms: Option<usize>,
        custom_print_mode: Option<usize>,
    ) -> PyResult<String> {
        Ok(self.poly.format_string(
            &PrintOptions {
                terms_on_new_line,
                color_top_level_sum,
                color_builtin_symbols,
                print_ring,
                symmetric_representation_for_finite_field,
                explicit_rational_polynomial,
                number_thousands_separator,
                multiplication_operator,
                double_star_for_exponentiation,
                square_brackets_for_function,
                num_exp_as_superscript,
                mode: mode.into(),
                precision,
                pretty_matrix: false,
                hide_all_namespaces: !show_namespaces,
                color_namespace: true,
                hide_namespace: Some("python"),
                include_attributes,
                max_terms,
                custom_print_mode: custom_print_mode.map(|x| ("default", x)),
            },
            PrintState::new(),
        ))
    }

    /// Convert the polynomial into a portable string.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new()))
    }

    /// Print the polynomial in a human-readable format.
    pub fn __str__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&DEFAULT_PRINT_OPTIONS, PrintState::new()))
    }

    pub fn __pow__(
        &self,
        exponent: usize,
        modulo: Option<i64>,
    ) -> PyResult<PythonPrimeTwoPolynomial> {
        if modulo.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "Optional number argument not supported",
            ));
        }

        Ok(Self {
            poly: self.poly.pow(exponent),
        })
    }

    /// Convert the polynomial into a LaTeX string.
    pub fn to_latex(&self) -> PyResult<String> {
        Ok(format!(
            "$${}$$",
            self.poly
                .format_string(&LATEX_PRINT_OPTIONS, PrintState::new())
        ))
    }

    /// Get the number of terms.
    pub fn nterms(&self) -> usize {
        self.poly.nterms()
    }

    /// Get the list of variables in the internal ordering of the polynomial.
    pub fn get_variables(&self) -> PyResult<Vec<PythonExpression>> {
        let mut var_list = vec![];

        for x in self.poly.get_vars_ref() {
            match x {
                PolyVariable::Symbol(x) => {
                    var_list.push(Atom::var(*x).into());
                }
                PolyVariable::Temporary(_) => {
                    Err(exceptions::PyValueError::new_err(
                        "Temporary variable in polynomial".to_string(),
                    ))?;
                }
                PolyVariable::Function(_, a) | PolyVariable::Power(a) => {
                    var_list.push(a.as_ref().clone().into());
                }
            }
        }

        Ok(var_list)
    }

    /// Add two polynomials `self and `rhs`, returning the result.
    pub fn __add__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly + &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .add_constant(self.poly.ring.element_from_integer(i)),
            }),
        }
    }

    /// Subtract polynomials `rhs` from `self`, returning the result.
    pub fn __sub__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly - &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .add_constant(self.poly.ring.neg(&self.poly.ring.element_from_integer(i))),
            }),
        }
    }

    /// Multiply two polynomials `self and `rhs`, returning the result.
    pub fn __mul__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly * &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .mul_coeff(self.poly.ring.element_from_integer(i)),
            }),
        }
    }

    pub fn __radd__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        self.__add__(rhs)
    }

    pub fn __rsub__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &p.poly - &self.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .neg()
                    .add_constant(self.poly.ring.element_from_integer(i)),
            }),
        }
    }

    pub fn __rmul__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        self.__mul__(rhs)
    }

    pub fn __floordiv__(&self, rhs: Self) -> PyResult<Self> {
        if rhs.poly.is_zero() {
            return Err(exceptions::PyValueError::new_err("Division by zero"));
        }

        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        };

        let (q, _r) = self.poly.quot_rem(&rhs.poly, false);

        Ok(Self { poly: q })
    }

    /// Divide the polynomial `self` by `rhs` if possible, returning the result.
    pub fn __truediv__(&self, rhs: Self) -> PyResult<Self> {
        if rhs.poly.is_zero() {
            return Err(exceptions::PyValueError::new_err("Division by zero"));
        }
        let (q, r) = self.poly.quot_rem(&rhs.poly, false);

        if r.is_zero() {
            Ok(Self { poly: q })
        } else {
            Err(exceptions::PyValueError::new_err(format!(
                "The division has a remainder: {}",
                r
            )))
        }
    }

    pub fn unify_variables(&mut self, other: &mut Self) {
        self.poly.unify_variables(&mut other.poly);
    }

    pub fn __contains__(&self, var: &PythonExpression) -> bool {
        self.contains(var)
    }

    pub fn contains(&self, var: &PythonExpression) -> bool {
        if let Some(p) =
            self.poly
                .get_vars_ref()
                .iter()
                .position(|v| match (v, var.expr.as_view()) {
                    (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                    (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                    _ => false,
                })
        {
            self.poly.contains(p)
        } else {
            false
        }
    }

    pub fn degree(&self, var: &PythonExpression) -> PyResult<isize> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, var.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                var.__str__()?
            )))?;

        Ok(self.poly.degree(x) as isize)
    }

    /// Set a new variable ordering for the polynomial.
    /// This can be used to introduce new variables as well.
    pub fn reorder(&mut self, order: Vec<PythonExpression>) -> PyResult<()> {
        let vars: Vec<_> = order
            .into_iter()
            .map(|x| x.expr.try_into())
            .collect::<Result<_, _>>()
            .map_err(|e| exceptions::PyValueError::new_err(e))?;
        self.poly = self
            .poly
            .rearrange_with_growth(&vars)
            .map_err(exceptions::PyValueError::new_err)?;
        Ok(())
    }

    /// Divide `self` by `rhs`, returning the quotient and remainder.
    pub fn quot_rem(
        &self,
        rhs: Self,
    ) -> PyResult<(PythonPrimeTwoPolynomial, PythonPrimeTwoPolynomial)> {
        if rhs.poly.is_zero() {
            Err(exceptions::PyValueError::new_err("Division by zero"))
        } else {
            let (q, r) = self.poly.quot_rem(&rhs.poly, false);
            Ok((Self { poly: q }, Self { poly: r }))
        }
    }

    /// Negate the polynomial.
    pub fn __neg__(&self) -> Self {
        Self {
            poly: self.poly.clone().neg(),
        }
    }

    /// Compute the remainder `self % rhs.
    pub fn __mod__(&self, rhs: Self) -> PyResult<Self> {
        if rhs.poly.is_zero() {
            Err(exceptions::PyValueError::new_err("Division by zero"))
        } else {
            Ok(Self {
                poly: self.poly.rem(&rhs.poly),
            })
        }
    }

    /// Compute the greatest common divisor (GCD) of two polynomials.
    /// Compute the greatest common divisor (GCD) of two or more polynomials.
    #[pyo3(signature = (*rhs))]
    pub fn gcd(
        &self,
        #[gen_stub(override_type(type_repr = "FiniteFieldPolynomial"))] rhs: &Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
        if rhs.len() == 1 {
            let rhs = rhs.get_item(0)?.extract::<Self>()?;

            Ok(Self {
                poly: self.poly.gcd(&rhs.poly),
            })
        } else {
            let mut args = vec![self.poly.clone()];
            for r in rhs.iter() {
                let p = r.extract::<Self>()?;

                args.push(p.poly);
            }

            Ok(Self {
                poly: PolynomialGCD::gcd_multiple(args),
            })
        }
    }

    /// Compute the resultant of two polynomials with respect to the variable `var`.
    pub fn resultant(&self, rhs: Self, var: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, var.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                var.__str__()?
            )))?;

        if self.poly.get_vars_ref() == rhs.poly.get_vars_ref() {
            let self_uni = self.poly.to_univariate(x);
            let rhs_uni = rhs.poly.to_univariate(x);

            Ok(Self {
                poly: self_uni.resultant_prs(&rhs_uni),
            })
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);

            let self_uni = new_self.to_univariate(x);
            let rhs_uni = new_rhs.to_univariate(x);

            Ok(Self {
                poly: self_uni.resultant_prs(&rhs_uni),
            })
        }
    }

    /// Compute the square-free factorization of the polynomial.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial()
    /// >>> print('Square-free factorization of {}:'.format(p))
    /// >>> for f, exp in p.factor_square_free():
    /// >>>     print('\t({})^{}'.format(f, exp))
    pub fn factor_square_free(&self) -> Vec<(PythonPrimeTwoPolynomial, usize)> {
        self.poly
            .square_free_factorization()
            .into_iter()
            .map(|(f, p)| (Self { poly: f }, p))
            .collect()
    }

    /// Factorize the polynomial.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial()
    /// >>> print('Factorization of {}:'.format(p))
    /// >>> for f, exp in p.factor():
    /// >>>     print('\t({})^{}'.format(f, exp))
    pub fn factor(&self) -> Vec<(PythonPrimeTwoPolynomial, usize)> {
        self.poly
            .factor()
            .into_iter()
            .map(|(f, p)| (Self { poly: f }, p))
            .collect()
    }

    /// Take a derivative in `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x^2+2').to_polynomial()
    /// >>> print(p.derivative(x))
    pub fn derivative(&self, x: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, x.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        Ok(Self {
            poly: self.poly.derivative(x),
        })
    }

    /// Make the polynomial monic, i.e., divide by the leading coefficient.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> p = E('6x^2+3x+9').to_polynomial().monic()
    /// >>> print(p)
    ///
    /// Yields `x^2+1/2*x+3/2`.
    pub fn monic(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.clone().make_monic(),
        })
    }

    /// Get the leading coefficient.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> p = E('3x^2+6x+9').to_polynomial().lcoeff()
    /// >>> print(p)
    ///
    /// Yields `3`.
    pub fn lcoeff(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.constant(self.poly.lcoeff().clone()),
        })
    }

    /// Get the coefficient list, optionally in the variables `vars`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial()
    /// >>> for n, pp in p.coefficient_list(x):
    /// >>>     print(n, pp)
    #[pyo3(signature = (vars = None))]
    pub fn coefficient_list(
        &self,
        vars: Option<OneOrMultiple<PythonExpression>>,
    ) -> PyResult<Vec<(Vec<usize>, PythonPrimeTwoPolynomial)>> {
        if let Some(vv) = vars {
            let mut vars = vec![];

            for vvv in vv.to_iter() {
                let x = self
                    .poly
                    .get_vars_ref()
                    .iter()
                    .position(|v| match (v, vvv.expr.as_view()) {
                        (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                        (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => {
                            f.as_view() == a
                        }
                        _ => false,
                    })
                    .ok_or(exceptions::PyValueError::new_err(format!(
                        "Variable {} not found in polynomial",
                        vvv.__str__()?
                    )))?;

                vars.push(x);
            }

            if vars.is_empty() {
                return Ok(self
                    .poly
                    .into_iter()
                    .map(|t| {
                        (
                            t.exponents.iter().map(|x| *x as usize).collect(),
                            Self {
                                poly: self.poly.constant(*t.coefficient),
                            },
                        )
                    })
                    .collect());
            }

            if vars.len() == 1 {
                return Ok(self
                    .poly
                    .to_univariate_polynomial_list(vars[0])
                    .into_iter()
                    .map(|(f, p)| (vec![p as usize], Self { poly: f }))
                    .collect());
            }

            // sort the exponents wrt the var map
            let mut r: Vec<(Vec<_>, _)> = self
                .poly
                .to_multivariate_polynomial_list(&vars, true)
                .into_iter()
                .map(|(f, p)| {
                    (
                        vars.iter().map(|v| f[*v] as usize).collect(),
                        Self { poly: p },
                    )
                })
                .collect();
            r.sort_by(|a, b| a.0.cmp(&b.0));

            Ok(r)
        } else {
            Ok(self
                .poly
                .into_iter()
                .map(|t| {
                    (
                        t.exponents.iter().map(|x| *x as usize).collect(),
                        Self {
                            poly: self.poly.constant(*t.coefficient),
                        },
                    )
                })
                .collect())
        }
    }

    /// Evaluate the polynomial at the given values.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> x, y = S('x', 'y')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial(modulus=5)
    /// >>> print(p.evaluate([2, 3]))
    /// 4
    pub fn evaluate(&self, values: Vec<Integer>) -> PyResult<Integer> {
        if values.len() != self.poly.get_vars_ref().len() {
            return Err(exceptions::PyValueError::new_err(format!(
                "Expected {} values, got {}",
                self.poly.get_vars_ref().len(),
                values.len()
            )));
        }

        let input = values
            .into_iter()
            .map(|x| self.poly.ring.element_from_integer(x))
            .collect::<Vec<_>>();

        let r = self.poly.replace_all(&input);

        Ok(self.poly.ring.to_integer(&r))
    }

    /// Replace the variable `x` with a polynomial `v`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial()
    /// >>> r = E('y+1').to_polynomial())
    /// >>> p.replace(x, r)
    pub fn replace(&self, x: PythonExpression, v: PolynomialOrInteger<Self>) -> PyResult<Self> {
        let id = match x.expr.as_view() {
            AtomView::Var(x) => x.get_symbol(),
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Derivative must be taken wrt a variable",
                ));
            }
        };

        let v = match v {
            PolynomialOrInteger::Polynomial(p) => p.poly,
            PolynomialOrInteger::Integer(i) => {
                self.poly.constant(self.poly.ring.element_from_integer(i))
            }
        };

        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|x| match x {
                PolyVariable::Symbol(y) => *y == id,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        if self.poly.get_vars_ref() == v.get_vars_ref() {
            Ok(Self {
                poly: self.poly.replace_with_poly(x, &v),
            })
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = v;
            new_self.unify_variables(&mut new_rhs);
            Ok(Self {
                poly: new_self.replace_with_poly(x, &new_rhs),
            })
        }
    }

    /// Compute the Groebner basis of a polynomial system.
    ///
    /// If `grevlex=True`, reverse graded lexicographical ordering is used,
    /// otherwise the ordering is lexicographical.
    ///
    /// If `print_stats=True` intermediate statistics will be printed.
    #[pyo3(signature = (system, grevlex = true, print_stats = false))]
    #[classmethod]
    pub fn groebner_basis(
        _cls: &Bound<'_, PyType>,
        system: Vec<Self>,
        grevlex: bool,
        print_stats: bool,
    ) -> Vec<Self> {
        if grevlex {
            let grevlex_ideal: Vec<_> = system
                .iter()
                .map(|p| p.poly.reorder::<GrevLexOrder>())
                .collect();
            let gb = GroebnerBasis::new(&grevlex_ideal, print_stats);

            gb.system
                .into_iter()
                .map(|p| Self {
                    poly: p.reorder::<LexOrder>(),
                })
                .collect()
        } else {
            let ideal: Vec<_> = system.iter().map(|p| p.poly.clone()).collect();
            let gb = GroebnerBasis::new(&ideal, print_stats);
            gb.system.into_iter().map(|p| Self { poly: p }).collect()
        }
    }

    /// Completely reduce the polynomial w.r.t the polynomials `gs`.
    /// For example reducing `f=y^2+x` by `g=[x]` yields `y^2`.
    #[pyo3(signature = (system, grevlex = true))]
    pub fn reduce(&self, system: Vec<Self>, grevlex: bool) -> Self {
        if grevlex {
            let p = self.poly.reorder::<GrevLexOrder>();
            let grevlex_ideal: Vec<_> = system
                .iter()
                .map(|p| p.poly.reorder::<GrevLexOrder>())
                .collect();

            let r = p.reduce(&grevlex_ideal);
            Self {
                poly: r.reorder::<LexOrder>(),
            }
        } else {
            let ideal: Vec<_> = system.into_iter().map(|p| p.poly).collect();
            Self {
                poly: self.poly.reduce(&ideal),
            }
        }
    }

    /// Integrate the polynomial in `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x^2+2').to_polynomial()
    /// >>> print(p.integrate(x))
    pub fn integrate(&self, x: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, x.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        Ok(Self {
            poly: self.poly.integrate(x),
        })
    }

    /// Convert the polynomial to an expression.
    pub fn to_expression(&self) -> PyResult<PythonExpression> {
        let p = self
            .poly
            .map_coeff(|c| (*c as i64).into(), IntegerRing::new());

        Ok(p.to_expression().into())
    }
}

/// A Symbolica polynomial over Z2 Galois fields.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(
    name = "GaloisFieldPrimeTwoPolynomial",
    subclass,
    module = "symbolica.core"
)]
#[derive(Clone)]
pub struct PythonGaloisFieldPrimeTwoPolynomial {
    pub poly: MultivariatePolynomial<AlgebraicExtension<Z2>, u16>,
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(&mut PythonGaloisFieldPrimeTwoPolynomial = PythonGaloisFieldPrimeTwoPolynomial);

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonGaloisFieldPrimeTwoPolynomial {
    /// Compare two polynomials.
    fn __richcmp__(&self, o: Py<PyAny>, op: CompareOp, py: Python) -> PyResult<bool> {
        let Ok(other) = o.extract::<PolynomialOrInteger<PythonGaloisFieldPrimeTwoPolynomial>>(py)
        else {
            return Err(exceptions::PyTypeError::new_err(
                "Can only compare Polynomial with Polynomial or integer.",
            ));
        };
        match op {
            CompareOp::Eq => match other {
                PolynomialOrInteger::Integer(i) => Ok(self.poly.is_constant()
                    && self.poly.get_constant() == self.poly.ring.element_from_integer(i)),
                PolynomialOrInteger::Polynomial(p) => Ok(self.poly == p.poly),
            },
            CompareOp::Ne => match other {
                PolynomialOrInteger::Integer(i) => Ok(!self.poly.is_constant()
                    || self.poly.get_constant() != self.poly.ring.element_from_integer(i)),
                PolynomialOrInteger::Polynomial(p) => Ok(self.poly != p.poly),
            },
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "Inequalities between polynomials are not allowed in {} {} {}",
                self.__str__()?,
                match op {
                    CompareOp::Eq => "==",
                    CompareOp::Ge => ">=",
                    CompareOp::Gt => ">",
                    CompareOp::Le => "<=",
                    CompareOp::Lt => "<",
                    CompareOp::Ne => "!=",
                },
                match other {
                    PolynomialOrInteger::Integer(i) => i.to_string(),
                    PolynomialOrInteger::Polynomial(p) => p.__str__()?,
                }
            ))),
        }
    }

    /// Copy the polynomial.
    pub fn __copy__(&self) -> Self {
        Self {
            poly: self.poly.clone(),
        }
    }

    /// Convert the polynomial into a human-readable string, with tunable settings.
    ///
    /// Examples
    /// --------
    /// >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
    /// >>> print(p.format(symmetric_representation_for_finite_field=True))
    #[pyo3(signature =
    (mode = PythonPrintMode::Symbolica,
        terms_on_new_line = false,
        color_top_level_sum = true,
        color_builtin_symbols = true,
        print_ring = true,
        symmetric_representation_for_finite_field = false,
        explicit_rational_polynomial = false,
        number_thousands_separator = None,
        multiplication_operator = '*',
        double_star_for_exponentiation = false,
        square_brackets_for_function = false,
        num_exp_as_superscript = true,
        precision = None,
        show_namespaces = false,
            include_attributes = false,
        max_terms = None,
        custom_print_mode = None)
    )]
    pub fn format(
        &self,
        mode: PythonPrintMode,
        terms_on_new_line: bool,
        color_top_level_sum: bool,
        color_builtin_symbols: bool,
        print_ring: bool,
        symmetric_representation_for_finite_field: bool,
        explicit_rational_polynomial: bool,
        number_thousands_separator: Option<char>,
        multiplication_operator: char,
        double_star_for_exponentiation: bool,
        square_brackets_for_function: bool,
        num_exp_as_superscript: bool,
        precision: Option<usize>,
        show_namespaces: bool,
        include_attributes: bool,
        max_terms: Option<usize>,
        custom_print_mode: Option<usize>,
    ) -> PyResult<String> {
        Ok(self.poly.format_string(
            &PrintOptions {
                terms_on_new_line,
                color_top_level_sum,
                color_builtin_symbols,
                print_ring,
                symmetric_representation_for_finite_field,
                explicit_rational_polynomial,
                number_thousands_separator,
                multiplication_operator,
                double_star_for_exponentiation,
                square_brackets_for_function,
                num_exp_as_superscript,
                mode: mode.into(),
                precision,
                pretty_matrix: false,
                hide_all_namespaces: !show_namespaces,
                color_namespace: true,
                hide_namespace: Some("python"),
                include_attributes,
                max_terms,
                custom_print_mode: custom_print_mode.map(|x| ("default", x)),
            },
            PrintState::new(),
        ))
    }

    /// Convert the polynomial into a portable string.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new()))
    }

    /// Print the polynomial in a human-readable format.
    pub fn __str__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&DEFAULT_PRINT_OPTIONS, PrintState::new()))
    }

    pub fn __pow__(
        &self,
        exponent: usize,
        modulo: Option<i64>,
    ) -> PyResult<PythonGaloisFieldPrimeTwoPolynomial> {
        if modulo.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "Optional number argument not supported",
            ));
        }

        Ok(Self {
            poly: self.poly.pow(exponent),
        })
    }

    /// Convert the polynomial into a LaTeX string.
    pub fn to_latex(&self) -> PyResult<String> {
        Ok(format!(
            "$${}$$",
            self.poly
                .format_string(&LATEX_PRINT_OPTIONS, PrintState::new())
        ))
    }

    /// Get the number of terms.
    pub fn nterms(&self) -> usize {
        self.poly.nterms()
    }

    /// Get the list of variables in the internal ordering of the polynomial.
    pub fn get_variables(&self) -> PyResult<Vec<PythonExpression>> {
        let mut var_list = vec![];

        for x in self.poly.get_vars_ref() {
            match x {
                PolyVariable::Symbol(x) => {
                    var_list.push(Atom::var(*x).into());
                }
                PolyVariable::Temporary(_) => {
                    Err(exceptions::PyValueError::new_err(
                        "Temporary variable in polynomial".to_string(),
                    ))?;
                }
                PolyVariable::Function(_, a) | PolyVariable::Power(a) => {
                    var_list.push(a.as_ref().clone().into());
                }
            }
        }

        Ok(var_list)
    }

    /// Add two polynomials `self and `rhs`, returning the result.
    pub fn __add__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly + &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .add_constant(self.poly.ring.element_from_integer(i)),
            }),
        }
    }

    /// Subtract polynomials `rhs` from `self`, returning the result.
    pub fn __sub__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly - &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .add_constant(self.poly.ring.neg(&self.poly.ring.element_from_integer(i))),
            }),
        }
    }

    /// Multiply two polynomials `self and `rhs`, returning the result.
    pub fn __mul__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly * &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .mul_coeff(self.poly.ring.element_from_integer(i)),
            }),
        }
    }

    pub fn __radd__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        self.__add__(rhs)
    }

    pub fn __rsub__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &p.poly - &self.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .neg()
                    .add_constant(self.poly.ring.element_from_integer(i)),
            }),
        }
    }

    pub fn __rmul__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        self.__mul__(rhs)
    }

    pub fn __floordiv__(&self, rhs: Self) -> PyResult<Self> {
        if rhs.poly.is_zero() {
            return Err(exceptions::PyValueError::new_err("Division by zero"));
        }

        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        };

        let (q, _r) = self.poly.quot_rem(&rhs.poly, false);

        Ok(Self { poly: q })
    }

    /// Divide the polynomial `self` by `rhs` if possible, returning the result.
    pub fn __truediv__(&self, rhs: Self) -> PyResult<Self> {
        if rhs.poly.is_zero() {
            return Err(exceptions::PyValueError::new_err("Division by zero"));
        }
        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        };

        let (q, r) = self.poly.quot_rem(&rhs.poly, false);

        if r.is_zero() {
            Ok(Self { poly: q })
        } else {
            Err(exceptions::PyValueError::new_err(format!(
                "The division has a remainder: {}",
                r
            )))
        }
    }

    pub fn unify_variables(&mut self, other: &mut Self) {
        self.poly.unify_variables(&mut other.poly);
    }

    pub fn __contains__(&self, var: &PythonExpression) -> bool {
        self.contains(var)
    }

    pub fn contains(&self, var: &PythonExpression) -> bool {
        if let Some(p) =
            self.poly
                .get_vars_ref()
                .iter()
                .position(|v| match (v, var.expr.as_view()) {
                    (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                    (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                    _ => false,
                })
        {
            self.poly.contains(p)
        } else {
            false
        }
    }

    pub fn degree(&self, var: &PythonExpression) -> PyResult<isize> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, var.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                var.__str__()?
            )))?;

        Ok(self.poly.degree(x) as isize)
    }

    /// Set a new variable ordering for the polynomial.
    /// This can be used to introduce new variables as well.
    pub fn reorder(&mut self, order: Vec<PythonExpression>) -> PyResult<()> {
        let vars: Vec<_> = order
            .into_iter()
            .map(|x| x.expr.try_into())
            .collect::<Result<_, _>>()
            .map_err(|e| exceptions::PyValueError::new_err(e))?;
        self.poly = self
            .poly
            .rearrange_with_growth(&vars)
            .map_err(exceptions::PyValueError::new_err)?;
        Ok(())
    }

    /// Divide `self` by `rhs`, returning the quotient and remainder.
    pub fn quot_rem(
        &self,
        rhs: Self,
    ) -> PyResult<(
        PythonGaloisFieldPrimeTwoPolynomial,
        PythonGaloisFieldPrimeTwoPolynomial,
    )> {
        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        };

        if rhs.poly.is_zero() {
            Err(exceptions::PyValueError::new_err("Division by zero"))
        } else {
            let (q, r) = self.poly.quot_rem(&rhs.poly, false);
            Ok((Self { poly: q }, Self { poly: r }))
        }
    }

    /// Negate the polynomial.
    pub fn __neg__(&self) -> Self {
        Self {
            poly: self.poly.clone().neg(),
        }
    }

    /// Compute the remainder `self % rhs.
    pub fn __mod__(&self, rhs: Self) -> PyResult<Self> {
        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        };

        if rhs.poly.is_zero() {
            Err(exceptions::PyValueError::new_err("Division by zero"))
        } else {
            Ok(Self {
                poly: self.poly.rem(&rhs.poly),
            })
        }
    }

    /// Compute the greatest common divisor (GCD) of two or more polynomials.
    #[pyo3(signature = (*rhs))]
    pub fn gcd(
        &self,
        #[gen_stub(override_type(type_repr = "FiniteFieldPolynomial"))] rhs: &Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
        if rhs.len() == 1 {
            let rhs = rhs.get_item(0)?.extract::<Self>()?;
            if self.poly.ring != rhs.poly.ring {
                Err(exceptions::PyValueError::new_err(
                    "Polynomials have different rings".to_string(),
                ))
            } else {
                Ok(Self {
                    poly: self.poly.gcd(&rhs.poly),
                })
            }
        } else {
            let mut args = vec![self.poly.clone()];
            for r in rhs.iter() {
                let p = r.extract::<Self>()?;
                if args[0].ring != p.poly.ring {
                    return Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ));
                }
                args.push(p.poly);
            }

            Ok(Self {
                poly: PolynomialGCD::gcd_multiple(args),
            })
        }
    }

    /// Compute the extended GCD of two polynomials, yielding the GCD and the Bezout coefficients `s` and `t`
    /// such that `self * s + rhs * t = gcd(self, rhs)`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> E('(1+x)(20+x)').to_polynomial(modulus=5).extended_gcd(E('x^2+2').to_polynomial(modulus=5))
    ///
    /// yields `(1, 3+4*x, 3+x)`.
    pub fn extended_gcd(
        &self,
        rhs: Self,
    ) -> PyResult<(
        PythonGaloisFieldPrimeTwoPolynomial,
        PythonGaloisFieldPrimeTwoPolynomial,
        PythonGaloisFieldPrimeTwoPolynomial,
    )> {
        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        }

        if self.poly.variables != rhs.poly.variables
            || (0..self.poly.nvars())
                .filter(|i| self.poly.degree(*i) > 0 || rhs.poly.degree(*i) > 0)
                .count()
                > 1
        {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials are not univariate in the same variable".to_string(),
            ));
        }

        let (g, s, t) = self.poly.eea_univariate(&rhs.poly);
        Ok((Self { poly: g }, Self { poly: s }, Self { poly: t }))
    }

    /// Compute the resultant of two polynomials with respect to the variable `var`.
    pub fn resultant(&self, rhs: Self, var: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, var.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                var.__str__()?
            )))?;

        if self.poly.get_vars_ref() == rhs.poly.get_vars_ref() {
            let self_uni = self.poly.to_univariate(x);
            let rhs_uni = rhs.poly.to_univariate(x);

            Ok(Self {
                poly: self_uni.resultant_prs(&rhs_uni),
            })
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);

            let self_uni = new_self.to_univariate(x);
            let rhs_uni = new_rhs.to_univariate(x);

            Ok(Self {
                poly: self_uni.resultant_prs(&rhs_uni),
            })
        }
    }

    /// Compute the square-free factorization of the polynomial.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial()
    /// >>> print('Square-free factorization of {}:'.format(p))
    /// >>> for f, exp in p.factor_square_free():
    /// >>>     print('\t({})^{}'.format(f, exp))
    pub fn factor_square_free(&self) -> Vec<(PythonGaloisFieldPrimeTwoPolynomial, usize)> {
        self.poly
            .square_free_factorization()
            .into_iter()
            .map(|(f, p)| (Self { poly: f }, p))
            .collect()
    }

    /// Factorize the polynomial.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial()
    /// >>> print('Factorization of {}:'.format(p))
    /// >>> for f, exp in p.factor():
    /// >>>     print('\t({})^{}'.format(f, exp))
    pub fn factor(&self) -> Vec<(PythonGaloisFieldPrimeTwoPolynomial, usize)> {
        self.poly
            .factor()
            .into_iter()
            .map(|(f, p)| (Self { poly: f }, p))
            .collect()
    }

    /// Take a derivative in `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x^2+2').to_polynomial()
    /// >>> print(p.derivative(x))
    pub fn derivative(&self, x: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, x.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        Ok(Self {
            poly: self.poly.derivative(x),
        })
    }

    /// Make the polynomial monic, i.e., divide by the leading coefficient.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> p = E('6x^2+3x+9').to_polynomial().monic()
    /// >>> print(p)
    ///
    /// Yields `x^2+1/2*x+3/2`.
    pub fn monic(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.clone().make_monic(),
        })
    }

    /// Get the leading coefficient.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> p = E('3x^2+6x+9').to_polynomial().lcoeff()
    /// >>> print(p)
    ///
    /// Yields `3`.
    pub fn lcoeff(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.constant(self.poly.lcoeff().clone()),
        })
    }

    /// Get the coefficient list, optionally in the variables `vars`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial()
    /// >>> for n, pp in p.coefficient_list(x):
    /// >>>     print(n, pp)
    #[pyo3(signature = (vars = None))]
    pub fn coefficient_list(
        &self,
        vars: Option<OneOrMultiple<PythonExpression>>,
    ) -> PyResult<Vec<(Vec<usize>, PythonGaloisFieldPrimeTwoPolynomial)>> {
        if let Some(vv) = vars {
            let mut vars = vec![];

            for vvv in vv.to_iter() {
                let x = self
                    .poly
                    .get_vars_ref()
                    .iter()
                    .position(|v| match (v, vvv.expr.as_view()) {
                        (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                        (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => {
                            f.as_view() == a
                        }
                        _ => false,
                    })
                    .ok_or(exceptions::PyValueError::new_err(format!(
                        "Variable {} not found in polynomial",
                        vvv.__str__()?
                    )))?;

                vars.push(x);
            }

            if vars.is_empty() {
                return Ok(self
                    .poly
                    .into_iter()
                    .map(|t| {
                        (
                            t.exponents.iter().map(|x| *x as usize).collect(),
                            Self {
                                poly: self.poly.constant(t.coefficient.clone()),
                            },
                        )
                    })
                    .collect());
            }

            if vars.len() == 1 {
                return Ok(self
                    .poly
                    .to_univariate_polynomial_list(vars[0])
                    .into_iter()
                    .map(|(f, p)| (vec![p as usize], Self { poly: f }))
                    .collect());
            }

            // sort the exponents wrt the var map
            let mut r: Vec<(Vec<_>, _)> = self
                .poly
                .to_multivariate_polynomial_list(&vars, true)
                .into_iter()
                .map(|(f, p)| {
                    (
                        vars.iter().map(|v| f[*v] as usize).collect(),
                        Self { poly: p },
                    )
                })
                .collect();
            r.sort_by(|a, b| a.0.cmp(&b.0));

            Ok(r)
        } else {
            Ok(self
                .poly
                .into_iter()
                .map(|t| {
                    (
                        t.exponents.iter().map(|x| *x as usize).collect(),
                        Self {
                            poly: self.poly.constant(t.coefficient.clone()),
                        },
                    )
                })
                .collect())
        }
    }

    /// Evaluate the polynomial at the given values.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> x, y = S('x', 'y')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial(modulus=5)
    /// >>> print(p.evaluate([2, 3]))
    /// 4
    pub fn evaluate(&self, values: Vec<Integer>) -> PyResult<Integer> {
        if values.len() != self.poly.get_vars_ref().len() {
            return Err(exceptions::PyValueError::new_err(format!(
                "Expected {} values, got {}",
                self.poly.get_vars_ref().len(),
                values.len()
            )));
        }

        let input = values
            .into_iter()
            .map(|x| self.poly.ring.element_from_integer(x))
            .collect::<Vec<_>>();

        let r = self.poly.replace_all(&input);

        Ok(self.poly.ring.to_integer(&r))
    }

    /// Replace the variable `x` with a polynomial `v`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial()
    /// >>> r = E('y+1').to_polynomial())
    /// >>> p.replace(x, r)
    pub fn replace(&self, x: PythonExpression, v: PolynomialOrInteger<Self>) -> PyResult<Self> {
        let id = match x.expr.as_view() {
            AtomView::Var(x) => x.get_symbol(),
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Derivative must be taken wrt a variable",
                ));
            }
        };

        let v = match v {
            PolynomialOrInteger::Polynomial(p) => p.poly,
            PolynomialOrInteger::Integer(i) => {
                self.poly.constant(self.poly.ring.element_from_integer(i))
            }
        };

        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|x| match x {
                PolyVariable::Symbol(y) => *y == id,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        if self.poly.get_vars_ref() == v.get_vars_ref() {
            Ok(Self {
                poly: self.poly.replace_with_poly(x, &v),
            })
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = v;
            new_self.unify_variables(&mut new_rhs);
            Ok(Self {
                poly: new_self.replace_with_poly(x, &new_rhs),
            })
        }
    }

    /// Compute the Groebner basis of a polynomial system.
    ///
    /// If `grevlex=True`, reverse graded lexicographical ordering is used,
    /// otherwise the ordering is lexicographical.
    ///
    /// If `print_stats=True` intermediate statistics will be printed.
    #[pyo3(signature = (system, grevlex = true, print_stats = false))]
    #[classmethod]
    pub fn groebner_basis(
        _cls: &Bound<'_, PyType>,
        system: Vec<Self>,
        grevlex: bool,
        print_stats: bool,
    ) -> Vec<Self> {
        if grevlex {
            let grevlex_ideal: Vec<_> = system
                .iter()
                .map(|p| p.poly.reorder::<GrevLexOrder>())
                .collect();
            let gb = GroebnerBasis::new(&grevlex_ideal, print_stats);

            gb.system
                .into_iter()
                .map(|p| Self {
                    poly: p.reorder::<LexOrder>(),
                })
                .collect()
        } else {
            let ideal: Vec<_> = system.iter().map(|p| p.poly.clone()).collect();
            let gb = GroebnerBasis::new(&ideal, print_stats);
            gb.system.into_iter().map(|p| Self { poly: p }).collect()
        }
    }

    /// Completely reduce the polynomial w.r.t the polynomials `gs`.
    /// For example reducing `f=y^2+x` by `g=[x]` yields `y^2`.
    #[pyo3(signature = (system, grevlex = true))]
    pub fn reduce(&self, system: Vec<Self>, grevlex: bool) -> Self {
        if grevlex {
            let p = self.poly.reorder::<GrevLexOrder>();
            let grevlex_ideal: Vec<_> = system
                .iter()
                .map(|p| p.poly.reorder::<GrevLexOrder>())
                .collect();

            let r = p.reduce(&grevlex_ideal);
            Self {
                poly: r.reorder::<LexOrder>(),
            }
        } else {
            let ideal: Vec<_> = system.into_iter().map(|p| p.poly).collect();
            Self {
                poly: self.poly.reduce(&ideal),
            }
        }
    }

    /// Integrate the polynomial in `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x^2+2').to_polynomial()
    /// >>> print(p.integrate(x))
    pub fn integrate(&self, x: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, x.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        Ok(Self {
            poly: self.poly.integrate(x),
        })
    }

    /// Convert the polynomial to an expression.
    pub fn to_expression(&self) -> PyResult<PythonExpression> {
        Ok(self
            .poly
            .to_expression_with_coeff_map(|_, element, out| {
                let p = element
                    .poly
                    .map_coeff(|c| (*c as i64).into(), IntegerRing::new());
                p.to_expression_into(out);
            })
            .into())
    }

    /// Convert the polynomial to a polynomial over simple finite fields.
    pub fn to_polynomial(&self) -> PyResult<PythonPrimeTwoPolynomial> {
        let mut c = self.poly.clone();
        let mut min_poly = MultivariatePolynomial::new(
            &c.ring,
            None,
            Arc::new(self.poly.ring.poly().get_vars_ref().to_vec()),
        );
        c.unify_variables(&mut min_poly);

        let mut poly = MultivariatePolynomial::new(
            &c.ring.poly().ring,
            None,
            Arc::new(c.get_vars_ref().to_vec()),
        );

        for term in c.into_iter() {
            let mut t = term.coefficient.poly.clone();
            poly.unify_variables(&mut t);
            poly = poly + t.mul_exp(&term.exponents);
        }
        Ok(PythonPrimeTwoPolynomial { poly })
    }

    /// Get the minimal polynomial of the algebraic extension.
    pub fn get_minimal_polynomial(&self) -> PythonPrimeTwoPolynomial {
        PythonPrimeTwoPolynomial {
            poly: self.poly.ring.poly().clone(),
        }
    }
}

/// A Symbolica polynomial over Galois fields.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "GaloisFieldPolynomial", subclass, module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonGaloisFieldPolynomial {
    pub poly: MultivariatePolynomial<AlgebraicExtension<Zp64>, u16>,
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(&mut PythonGaloisFieldPolynomial = PythonGaloisFieldPolynomial);

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonGaloisFieldPolynomial {
    /// Compare two polynomials.
    fn __richcmp__(&self, o: Py<PyAny>, op: CompareOp, py: Python) -> PyResult<bool> {
        let Ok(other) = o.extract::<PolynomialOrInteger<PythonGaloisFieldPolynomial>>(py) else {
            return Err(exceptions::PyTypeError::new_err(
                "Can only compare Polynomial with Polynomial or integer.",
            ));
        };
        match op {
            CompareOp::Eq => match other {
                PolynomialOrInteger::Integer(i) => Ok(self.poly.is_constant()
                    && self.poly.get_constant() == self.poly.ring.element_from_integer(i)),
                PolynomialOrInteger::Polynomial(p) => Ok(self.poly == p.poly),
            },
            CompareOp::Ne => match other {
                PolynomialOrInteger::Integer(i) => Ok(!self.poly.is_constant()
                    || self.poly.get_constant() != self.poly.ring.element_from_integer(i)),
                PolynomialOrInteger::Polynomial(p) => Ok(self.poly != p.poly),
            },
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "Inequalities between polynomials are not allowed in {} {} {}",
                self.__str__()?,
                match op {
                    CompareOp::Eq => "==",
                    CompareOp::Ge => ">=",
                    CompareOp::Gt => ">",
                    CompareOp::Le => "<=",
                    CompareOp::Lt => "<",
                    CompareOp::Ne => "!=",
                },
                match other {
                    PolynomialOrInteger::Integer(i) => i.to_string(),
                    PolynomialOrInteger::Polynomial(p) => p.__str__()?,
                }
            ))),
        }
    }

    /// Copy the polynomial.
    pub fn __copy__(&self) -> Self {
        Self {
            poly: self.poly.clone(),
        }
    }

    /// Convert the polynomial into a human-readable string, with tunable settings.
    ///
    /// Examples
    /// --------
    /// >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
    /// >>> print(p.format(symmetric_representation_for_finite_field=True))
    #[pyo3(signature =
        (mode = PythonPrintMode::Symbolica,
            terms_on_new_line = false,
            color_top_level_sum = true,
            color_builtin_symbols = true,
            print_ring = true,
            symmetric_representation_for_finite_field = false,
            explicit_rational_polynomial = false,
            number_thousands_separator = None,
            multiplication_operator = '*',
            double_star_for_exponentiation = false,
            square_brackets_for_function = false,
            num_exp_as_superscript = true,
            precision = None,
            show_namespaces = false,
            include_attributes = false,
            max_terms = None,
            custom_print_mode = None)
        )]
    pub fn format(
        &self,
        mode: PythonPrintMode,
        terms_on_new_line: bool,
        color_top_level_sum: bool,
        color_builtin_symbols: bool,
        print_ring: bool,
        symmetric_representation_for_finite_field: bool,
        explicit_rational_polynomial: bool,
        number_thousands_separator: Option<char>,
        multiplication_operator: char,
        double_star_for_exponentiation: bool,
        square_brackets_for_function: bool,
        num_exp_as_superscript: bool,
        precision: Option<usize>,
        show_namespaces: bool,
        include_attributes: bool,
        max_terms: Option<usize>,
        custom_print_mode: Option<usize>,
    ) -> PyResult<String> {
        Ok(self.poly.format_string(
            &PrintOptions {
                terms_on_new_line,
                color_top_level_sum,
                color_builtin_symbols,
                print_ring,
                symmetric_representation_for_finite_field,
                explicit_rational_polynomial,
                number_thousands_separator,
                multiplication_operator,
                double_star_for_exponentiation,
                square_brackets_for_function,
                num_exp_as_superscript,
                mode: mode.into(),
                precision,
                pretty_matrix: false,
                hide_all_namespaces: !show_namespaces,
                color_namespace: true,
                hide_namespace: Some("python"),
                include_attributes,
                max_terms,
                custom_print_mode: custom_print_mode.map(|x| ("default", x)),
            },
            PrintState::new(),
        ))
    }

    /// Convert the polynomial into a portable string.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new()))
    }

    /// Print the polynomial in a human-readable format.
    pub fn __str__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&DEFAULT_PRINT_OPTIONS, PrintState::new()))
    }

    pub fn __pow__(
        &self,
        exponent: usize,
        modulo: Option<i64>,
    ) -> PyResult<PythonGaloisFieldPolynomial> {
        if modulo.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "Optional number argument not supported",
            ));
        }

        Ok(Self {
            poly: self.poly.pow(exponent),
        })
    }

    /// Convert the polynomial into a LaTeX string.
    pub fn to_latex(&self) -> PyResult<String> {
        Ok(format!(
            "$${}$$",
            self.poly
                .format_string(&LATEX_PRINT_OPTIONS, PrintState::new())
        ))
    }

    /// Get the number of terms.
    pub fn nterms(&self) -> usize {
        self.poly.nterms()
    }

    /// Get the list of variables in the internal ordering of the polynomial.
    pub fn get_variables(&self) -> PyResult<Vec<PythonExpression>> {
        let mut var_list = vec![];

        for x in self.poly.get_vars_ref() {
            match x {
                PolyVariable::Symbol(x) => {
                    var_list.push(Atom::var(*x).into());
                }
                PolyVariable::Temporary(_) => {
                    Err(exceptions::PyValueError::new_err(
                        "Temporary variable in polynomial".to_string(),
                    ))?;
                }
                PolyVariable::Function(_, a) | PolyVariable::Power(a) => {
                    var_list.push(a.as_ref().clone().into());
                }
            }
        }

        Ok(var_list)
    }

    /// Add two polynomials `self and `rhs`, returning the result.
    pub fn __add__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly + &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .add_constant(self.poly.ring.element_from_integer(i)),
            }),
        }
    }

    /// Subtract polynomials `rhs` from `self`, returning the result.
    pub fn __sub__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly - &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .add_constant(self.poly.ring.neg(&self.poly.ring.element_from_integer(i))),
            }),
        }
    }

    /// Multiply two polynomials `self and `rhs`, returning the result.
    pub fn __mul__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly * &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .mul_coeff(self.poly.ring.element_from_integer(i)),
            }),
        }
    }

    pub fn __radd__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        self.__add__(rhs)
    }

    pub fn __rsub__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &p.poly - &self.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .neg()
                    .add_constant(self.poly.ring.element_from_integer(i)),
            }),
        }
    }

    pub fn __rmul__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        self.__mul__(rhs)
    }

    pub fn __floordiv__(&self, rhs: Self) -> PyResult<Self> {
        if rhs.poly.is_zero() {
            return Err(exceptions::PyValueError::new_err("Division by zero"));
        }

        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        };

        let (q, _r) = self.poly.quot_rem(&rhs.poly, false);

        Ok(Self { poly: q })
    }

    /// Divide the polynomial `self` by `rhs` if possible, returning the result.
    pub fn __truediv__(&self, rhs: Self) -> PyResult<Self> {
        if rhs.poly.is_zero() {
            return Err(exceptions::PyValueError::new_err("Division by zero"));
        }
        let (q, r) = self.poly.quot_rem(&rhs.poly, false);

        if r.is_zero() {
            Ok(Self { poly: q })
        } else {
            Err(exceptions::PyValueError::new_err(format!(
                "The division has a remainder: {}",
                r
            )))
        }
    }

    pub fn unify_variables(&mut self, other: &mut Self) {
        self.poly.unify_variables(&mut other.poly);
    }

    pub fn __contains__(&self, var: &PythonExpression) -> bool {
        self.contains(var)
    }

    pub fn contains(&self, var: &PythonExpression) -> bool {
        if let Some(p) =
            self.poly
                .get_vars_ref()
                .iter()
                .position(|v| match (v, var.expr.as_view()) {
                    (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                    (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                    _ => false,
                })
        {
            self.poly.contains(p)
        } else {
            false
        }
    }

    pub fn degree(&self, var: &PythonExpression) -> PyResult<isize> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, var.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                var.__str__()?
            )))?;

        Ok(self.poly.degree(x) as isize)
    }

    /// Set a new variable ordering for the polynomial.
    /// This can be used to introduce new variables as well.
    pub fn reorder(&mut self, order: Vec<PythonExpression>) -> PyResult<()> {
        let vars: Vec<_> = order
            .into_iter()
            .map(|x| x.expr.try_into())
            .collect::<Result<_, _>>()
            .map_err(|e| exceptions::PyValueError::new_err(e))?;
        self.poly = self
            .poly
            .rearrange_with_growth(&vars)
            .map_err(exceptions::PyValueError::new_err)?;
        Ok(())
    }

    /// Divide `self` by `rhs`, returning the quotient and remainder.
    pub fn quot_rem(
        &self,
        rhs: Self,
    ) -> PyResult<(PythonGaloisFieldPolynomial, PythonGaloisFieldPolynomial)> {
        if rhs.poly.is_zero() {
            Err(exceptions::PyValueError::new_err("Division by zero"))
        } else {
            let (q, r) = self.poly.quot_rem(&rhs.poly, false);
            Ok((Self { poly: q }, Self { poly: r }))
        }
    }

    /// Negate the polynomial.
    pub fn __neg__(&self) -> Self {
        Self {
            poly: self.poly.clone().neg(),
        }
    }

    /// Compute the remainder `self % rhs.
    pub fn __mod__(&self, rhs: Self) -> PyResult<Self> {
        if rhs.poly.is_zero() {
            Err(exceptions::PyValueError::new_err("Division by zero"))
        } else {
            Ok(Self {
                poly: self.poly.rem(&rhs.poly),
            })
        }
    }

    /// Compute the greatest common divisor (GCD) of two or more polynomials.
    #[pyo3(signature = (*rhs))]
    pub fn gcd(
        &self,
        #[gen_stub(override_type(type_repr = "GaloisFieldPolynomial"))] rhs: &Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
        if rhs.len() == 1 {
            let rhs = rhs.get_item(0)?.extract::<Self>()?;
            if self.poly.ring != rhs.poly.ring {
                Err(exceptions::PyValueError::new_err(
                    "Polynomials have different rings".to_string(),
                ))
            } else {
                Ok(Self {
                    poly: self.poly.gcd(&rhs.poly),
                })
            }
        } else {
            let mut args = vec![self.poly.clone()];
            for r in rhs.iter() {
                let p = r.extract::<Self>()?;
                if args[0].ring != p.poly.ring {
                    return Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ));
                }
                args.push(p.poly);
            }

            Ok(Self {
                poly: PolynomialGCD::gcd_multiple(args),
            })
        }
    }

    /// Compute the resultant of two polynomials with respect to the variable `var`.
    pub fn resultant(&self, rhs: Self, var: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, var.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                var.__str__()?
            )))?;

        if self.poly.get_vars_ref() == rhs.poly.get_vars_ref() {
            let self_uni = self.poly.to_univariate(x);
            let rhs_uni = rhs.poly.to_univariate(x);

            Ok(Self {
                poly: self_uni.resultant_prs(&rhs_uni),
            })
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);

            let self_uni = new_self.to_univariate(x);
            let rhs_uni = new_rhs.to_univariate(x);

            Ok(Self {
                poly: self_uni.resultant_prs(&rhs_uni),
            })
        }
    }

    /// Compute the square-free factorization of the polynomial.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial()
    /// >>> print('Square-free factorization of {}:'.format(p))
    /// >>> for f, exp in p.factor_square_free():
    /// >>>     print('\t({})^{}'.format(f, exp))
    pub fn factor_square_free(&self) -> Vec<(PythonGaloisFieldPolynomial, usize)> {
        self.poly
            .square_free_factorization()
            .into_iter()
            .map(|(f, p)| (Self { poly: f }, p))
            .collect()
    }

    /// Factorize the polynomial.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial()
    /// >>> print('Factorization of {}:'.format(p))
    /// >>> for f, exp in p.factor():
    /// >>>     print('\t({})^{}'.format(f, exp))
    pub fn factor(&self) -> Vec<(PythonGaloisFieldPolynomial, usize)> {
        self.poly
            .factor()
            .into_iter()
            .map(|(f, p)| (Self { poly: f }, p))
            .collect()
    }

    /// Take a derivative in `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x^2+2').to_polynomial()
    /// >>> print(p.derivative(x))
    pub fn derivative(&self, x: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, x.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        Ok(Self {
            poly: self.poly.derivative(x),
        })
    }

    /// Make the polynomial monic, i.e., divide by the leading coefficient.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> p = E('6x^2+3x+9').to_polynomial().monic()
    /// >>> print(p)
    ///
    /// Yields `x^2+1/2*x+3/2`.
    pub fn monic(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.clone().make_monic(),
        })
    }

    /// Get the leading coefficient.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> p = E('3x^2+6x+9').to_polynomial().lcoeff()
    /// >>> print(p)
    ///
    /// Yields `3`.
    pub fn lcoeff(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.constant(self.poly.lcoeff().clone()),
        })
    }

    /// Get the coefficient list, optionally in the variables `vars`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial()
    /// >>> for n, pp in p.coefficient_list(x):
    /// >>>     print(n, pp)
    #[pyo3(signature = (vars = None))]
    pub fn coefficient_list(
        &self,
        vars: Option<OneOrMultiple<PythonExpression>>,
    ) -> PyResult<Vec<(Vec<usize>, PythonGaloisFieldPolynomial)>> {
        if let Some(vv) = vars {
            let mut vars = vec![];

            for vvv in vv.to_iter() {
                let x = self
                    .poly
                    .get_vars_ref()
                    .iter()
                    .position(|v| match (v, vvv.expr.as_view()) {
                        (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                        (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => {
                            f.as_view() == a
                        }
                        _ => false,
                    })
                    .ok_or(exceptions::PyValueError::new_err(format!(
                        "Variable {} not found in polynomial",
                        vvv.__str__()?
                    )))?;

                vars.push(x);
            }

            if vars.is_empty() {
                return Ok(self
                    .poly
                    .into_iter()
                    .map(|t| {
                        (
                            t.exponents.iter().map(|x| *x as usize).collect(),
                            Self {
                                poly: self.poly.constant(t.coefficient.clone()),
                            },
                        )
                    })
                    .collect());
            }

            if vars.len() == 1 {
                return Ok(self
                    .poly
                    .to_univariate_polynomial_list(vars[0])
                    .into_iter()
                    .map(|(f, p)| (vec![p as usize], Self { poly: f }))
                    .collect());
            }

            // sort the exponents wrt the var map
            let mut r: Vec<(Vec<_>, _)> = self
                .poly
                .to_multivariate_polynomial_list(&vars, true)
                .into_iter()
                .map(|(f, p)| {
                    (
                        vars.iter().map(|v| f[*v] as usize).collect(),
                        Self { poly: p },
                    )
                })
                .collect();
            r.sort_by(|a, b| a.0.cmp(&b.0));

            Ok(r)
        } else {
            Ok(self
                .poly
                .into_iter()
                .map(|t| {
                    (
                        t.exponents.iter().map(|x| *x as usize).collect(),
                        Self {
                            poly: self.poly.constant(t.coefficient.clone()),
                        },
                    )
                })
                .collect())
        }
    }

    /// Evaluate the polynomial at the given values.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> x, y = S('x', 'y')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial(modulus=5)
    /// >>> print(p.evaluate([2, 3]))
    /// 4
    pub fn evaluate(&self, values: Vec<Integer>) -> PyResult<Integer> {
        if values.len() != self.poly.get_vars_ref().len() {
            return Err(exceptions::PyValueError::new_err(format!(
                "Expected {} values, got {}",
                self.poly.get_vars_ref().len(),
                values.len()
            )));
        }

        let input = values
            .into_iter()
            .map(|x| self.poly.ring.element_from_integer(x))
            .collect::<Vec<_>>();

        let r = self.poly.replace_all(&input);

        Ok(self.poly.ring.to_integer(&r))
    }

    /// Replace the variable `x` with a polynomial `v`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial()
    /// >>> r = E('y+1').to_polynomial())
    /// >>> p.replace(x, r)
    pub fn replace(&self, x: PythonExpression, v: PolynomialOrInteger<Self>) -> PyResult<Self> {
        let id = match x.expr.as_view() {
            AtomView::Var(x) => x.get_symbol(),
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Derivative must be taken wrt a variable",
                ));
            }
        };

        let v = match v {
            PolynomialOrInteger::Polynomial(p) => p.poly,
            PolynomialOrInteger::Integer(i) => {
                self.poly.constant(self.poly.ring.element_from_integer(i))
            }
        };

        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|x| match x {
                PolyVariable::Symbol(y) => *y == id,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        if self.poly.get_vars_ref() == v.get_vars_ref() {
            Ok(Self {
                poly: self.poly.replace_with_poly(x, &v),
            })
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = v;
            new_self.unify_variables(&mut new_rhs);
            Ok(Self {
                poly: new_self.replace_with_poly(x, &new_rhs),
            })
        }
    }

    /// Compute the Groebner basis of a polynomial system.
    ///
    /// If `grevlex=True`, reverse graded lexicographical ordering is used,
    /// otherwise the ordering is lexicographical.
    ///
    /// If `print_stats=True` intermediate statistics will be printed.
    #[pyo3(signature = (system, grevlex = true, print_stats = false))]
    #[classmethod]
    pub fn groebner_basis(
        _cls: &Bound<'_, PyType>,
        system: Vec<Self>,
        grevlex: bool,
        print_stats: bool,
    ) -> Vec<Self> {
        if grevlex {
            let grevlex_ideal: Vec<_> = system
                .iter()
                .map(|p| p.poly.reorder::<GrevLexOrder>())
                .collect();
            let gb = GroebnerBasis::new(&grevlex_ideal, print_stats);

            gb.system
                .into_iter()
                .map(|p| Self {
                    poly: p.reorder::<LexOrder>(),
                })
                .collect()
        } else {
            let ideal: Vec<_> = system.iter().map(|p| p.poly.clone()).collect();
            let gb = GroebnerBasis::new(&ideal, print_stats);
            gb.system.into_iter().map(|p| Self { poly: p }).collect()
        }
    }

    /// Completely reduce the polynomial w.r.t the polynomials `gs`.
    /// For example reducing `f=y^2+x` by `g=[x]` yields `y^2`.
    #[pyo3(signature = (system, grevlex = true))]
    pub fn reduce(&self, system: Vec<Self>, grevlex: bool) -> Self {
        if grevlex {
            let p = self.poly.reorder::<GrevLexOrder>();
            let grevlex_ideal: Vec<_> = system
                .iter()
                .map(|p| p.poly.reorder::<GrevLexOrder>())
                .collect();

            let r = p.reduce(&grevlex_ideal);
            Self {
                poly: r.reorder::<LexOrder>(),
            }
        } else {
            let ideal: Vec<_> = system.into_iter().map(|p| p.poly).collect();
            Self {
                poly: self.poly.reduce(&ideal),
            }
        }
    }

    /// Integrate the polynomial in `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x^2+2').to_polynomial()
    /// >>> print(p.integrate(x))
    pub fn integrate(&self, x: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, x.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        Ok(Self {
            poly: self.poly.integrate(x),
        })
    }

    /// Convert the polynomial to an expression.
    pub fn to_expression(&self) -> PyResult<PythonExpression> {
        Ok(self
            .poly
            .to_expression_with_coeff_map(|_, element, out| {
                let p = element.poly.map_coeff(
                    |c| Integer::from_finite_field(&element.poly.ring, *c),
                    IntegerRing::new(),
                );
                p.to_expression_into(out);
            })
            .into())
    }

    /// Convert the polynomial to a polynomial over simple finite fields.
    pub fn to_polynomial(&self) -> PyResult<PythonFiniteFieldPolynomial> {
        let mut c = self.poly.clone();
        let mut min_poly = MultivariatePolynomial::new(
            &c.ring,
            None,
            Arc::new(self.poly.ring.poly().get_vars_ref().to_vec()),
        );
        c.unify_variables(&mut min_poly);

        let mut poly = MultivariatePolynomial::new(
            &c.ring.poly().ring,
            None,
            Arc::new(c.get_vars_ref().to_vec()),
        );

        for term in c.into_iter() {
            let mut t = term.coefficient.poly.clone();
            poly.unify_variables(&mut t);
            poly = poly + t.mul_exp(&term.exponents);
        }
        Ok(PythonFiniteFieldPolynomial { poly })
    }

    /// Get the minimal polynomial of the algebraic extension.
    pub fn get_minimal_polynomial(&self) -> PythonFiniteFieldPolynomial {
        PythonFiniteFieldPolynomial {
            poly: self.poly.ring.poly().clone(),
        }
    }

    /// Get the modulus of the base finite field.
    pub fn get_modulus(&self) -> u64 {
        self.poly.ring.poly().ring.get_prime()
    }
}

/// A Symbolica polynomial over number fields.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "NumberFieldPolynomial", subclass, module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonNumberFieldPolynomial {
    pub poly: MultivariatePolynomial<AlgebraicExtension<Q>, u16>,
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(&mut PythonNumberFieldPolynomial = PythonNumberFieldPolynomial);

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonNumberFieldPolynomial {
    /// Compare two polynomials.
    fn __richcmp__(&self, o: Py<PyAny>, op: CompareOp, py: Python) -> PyResult<bool> {
        let Ok(other) = o.extract::<PolynomialOrInteger<PythonNumberFieldPolynomial>>(py) else {
            return Err(exceptions::PyTypeError::new_err(
                "Can only compare Polynomial with Polynomial or integer.",
            ));
        };
        match op {
            CompareOp::Eq => match other {
                PolynomialOrInteger::Integer(i) => Ok(self.poly.is_constant()
                    && self.poly.get_constant() == self.poly.ring.element_from_integer(i)),
                PolynomialOrInteger::Polynomial(p) => Ok(self.poly == p.poly),
            },
            CompareOp::Ne => match other {
                PolynomialOrInteger::Integer(i) => Ok(!self.poly.is_constant()
                    || self.poly.get_constant() != self.poly.ring.element_from_integer(i)),
                PolynomialOrInteger::Polynomial(p) => Ok(self.poly != p.poly),
            },
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "Inequalities between polynomials are not allowed in {} {} {}",
                self.__str__()?,
                match op {
                    CompareOp::Eq => "==",
                    CompareOp::Ge => ">=",
                    CompareOp::Gt => ">",
                    CompareOp::Le => "<=",
                    CompareOp::Lt => "<",
                    CompareOp::Ne => "!=",
                },
                match other {
                    PolynomialOrInteger::Integer(i) => i.to_string(),
                    PolynomialOrInteger::Polynomial(p) => p.__str__()?,
                }
            ))),
        }
    }

    /// Copy the polynomial.
    pub fn __copy__(&self) -> Self {
        Self {
            poly: self.poly.clone(),
        }
    }

    /// Convert the polynomial into a human-readable string, with tunable settings.
    ///
    /// Examples
    /// --------
    /// >>> p = FiniteFieldPolynomial.parse("3*x^2+2*x+7*x^3", ['x'], 11)
    /// >>> print(p.format(symmetric_representation_for_finite_field=True))
    #[pyo3(signature =
    (mode = PythonPrintMode::Symbolica,
        terms_on_new_line = false,
        color_top_level_sum = true,
        color_builtin_symbols = true,
        print_ring = true,
        symmetric_representation_for_finite_field = false,
        explicit_rational_polynomial = false,
        number_thousands_separator = None,
        multiplication_operator = '*',
        double_star_for_exponentiation = false,
        square_brackets_for_function = false,
        num_exp_as_superscript = true,
        precision = None,
        show_namespaces = false,
            include_attributes = false,
        max_terms = None,
        custom_print_mode = None)
    )]
    pub fn format(
        &self,
        mode: PythonPrintMode,
        terms_on_new_line: bool,
        color_top_level_sum: bool,
        color_builtin_symbols: bool,
        print_ring: bool,
        symmetric_representation_for_finite_field: bool,
        explicit_rational_polynomial: bool,
        number_thousands_separator: Option<char>,
        multiplication_operator: char,
        double_star_for_exponentiation: bool,
        square_brackets_for_function: bool,
        num_exp_as_superscript: bool,
        precision: Option<usize>,
        show_namespaces: bool,
        include_attributes: bool,
        max_terms: Option<usize>,
        custom_print_mode: Option<usize>,
    ) -> PyResult<String> {
        Ok(self.poly.format_string(
            &PrintOptions {
                terms_on_new_line,
                color_top_level_sum,
                color_builtin_symbols,
                print_ring,
                symmetric_representation_for_finite_field,
                explicit_rational_polynomial,
                number_thousands_separator,
                multiplication_operator,
                double_star_for_exponentiation,
                square_brackets_for_function,
                num_exp_as_superscript,
                mode: mode.into(),
                precision,
                pretty_matrix: false,
                hide_all_namespaces: !show_namespaces,
                color_namespace: true,
                hide_namespace: Some("python"),
                include_attributes,
                max_terms,
                custom_print_mode: custom_print_mode.map(|x| ("default", x)),
            },
            PrintState::new(),
        ))
    }

    /// Convert the polynomial into a portable string.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new()))
    }

    /// Print the polynomial in a human-readable format.
    pub fn __str__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&DEFAULT_PRINT_OPTIONS, PrintState::new()))
    }

    pub fn __pow__(
        &self,
        exponent: usize,
        modulo: Option<i64>,
    ) -> PyResult<PythonNumberFieldPolynomial> {
        if modulo.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "Optional number argument not supported",
            ));
        }

        Ok(Self {
            poly: self.poly.pow(exponent),
        })
    }

    /// Convert the polynomial into a LaTeX string.
    pub fn to_latex(&self) -> PyResult<String> {
        Ok(format!(
            "$${}$$",
            self.poly
                .format_string(&LATEX_PRINT_OPTIONS, PrintState::new())
        ))
    }

    /// Get the number of terms.
    pub fn nterms(&self) -> usize {
        self.poly.nterms()
    }

    /// Get the list of variables in the internal ordering of the polynomial.
    pub fn get_variables(&self) -> PyResult<Vec<PythonExpression>> {
        let mut var_list = vec![];

        for x in self.poly.get_vars_ref() {
            match x {
                PolyVariable::Symbol(x) => {
                    var_list.push(Atom::var(*x).into());
                }
                PolyVariable::Temporary(_) => {
                    Err(exceptions::PyValueError::new_err(
                        "Temporary variable in polynomial".to_string(),
                    ))?;
                }
                PolyVariable::Function(_, a) | PolyVariable::Power(a) => {
                    var_list.push(a.as_ref().clone().into());
                }
            }
        }

        Ok(var_list)
    }

    /// Add two polynomials `self and `rhs`, returning the result.
    pub fn __add__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly + &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .add_constant(self.poly.ring.element_from_integer(i)),
            }),
        }
    }

    /// Subtract polynomials `rhs` from `self`, returning the result.
    pub fn __sub__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly - &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .add_constant(self.poly.ring.neg(&self.poly.ring.element_from_integer(i))),
            }),
        }
    }

    /// Multiply two polynomials `self and `rhs`, returning the result.
    pub fn __mul__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &self.poly * &p.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .mul_coeff(self.poly.ring.element_from_integer(i)),
            }),
        }
    }

    pub fn __radd__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        self.__add__(rhs)
    }

    pub fn __rsub__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        match rhs {
            PolynomialOrInteger::Polynomial(p) => {
                if self.poly.ring != p.poly.ring {
                    Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ))
                } else {
                    Ok(Self {
                        poly: &p.poly - &self.poly,
                    })
                }
            }
            PolynomialOrInteger::Integer(i) => Ok(Self {
                poly: self
                    .poly
                    .clone()
                    .neg()
                    .add_constant(self.poly.ring.element_from_integer(i)),
            }),
        }
    }

    pub fn __rmul__(&self, rhs: PolynomialOrInteger<Self>) -> PyResult<Self> {
        self.__mul__(rhs)
    }

    pub fn __floordiv__(&self, rhs: Self) -> PyResult<Self> {
        if rhs.poly.is_zero() {
            return Err(exceptions::PyValueError::new_err("Division by zero"));
        }

        if self.poly.ring != rhs.poly.ring {
            return Err(exceptions::PyValueError::new_err(
                "Polynomials have different rings".to_string(),
            ));
        };

        let (q, _r) = self.poly.quot_rem(&rhs.poly, false);

        Ok(Self { poly: q })
    }

    /// Divide the polynomial `self` by `rhs` if possible, returning the result.
    pub fn __truediv__(&self, rhs: Self) -> PyResult<Self> {
        if rhs.poly.is_zero() {
            return Err(exceptions::PyValueError::new_err("Division by zero"));
        }
        let (q, r) = self.poly.quot_rem(&rhs.poly, false);

        if r.is_zero() {
            Ok(Self { poly: q })
        } else {
            Err(exceptions::PyValueError::new_err(format!(
                "The division has a remainder: {}",
                r
            )))
        }
    }

    pub fn unify_variables(&mut self, other: &mut Self) {
        self.poly.unify_variables(&mut other.poly);
    }

    pub fn __contains__(&self, var: &PythonExpression) -> bool {
        self.contains(var)
    }

    pub fn contains(&self, var: &PythonExpression) -> bool {
        if let Some(p) =
            self.poly
                .get_vars_ref()
                .iter()
                .position(|v| match (v, var.expr.as_view()) {
                    (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                    (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                    _ => false,
                })
        {
            self.poly.contains(p)
        } else {
            false
        }
    }

    pub fn degree(&self, var: &PythonExpression) -> PyResult<isize> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, var.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                var.__str__()?
            )))?;

        Ok(self.poly.degree(x) as isize)
    }

    /// Set a new variable ordering for the polynomial.
    /// This can be used to introduce new variables as well.
    pub fn reorder(&mut self, order: Vec<PythonExpression>) -> PyResult<()> {
        let vars: Vec<_> = order
            .into_iter()
            .map(|x| x.expr.try_into())
            .collect::<Result<_, _>>()
            .map_err(|e| exceptions::PyValueError::new_err(e))?;
        self.poly = self
            .poly
            .rearrange_with_growth(&vars)
            .map_err(exceptions::PyValueError::new_err)?;
        Ok(())
    }

    /// Divide `self` by `rhs`, returning the quotient and remainder.
    pub fn quot_rem(
        &self,
        rhs: Self,
    ) -> PyResult<(PythonNumberFieldPolynomial, PythonNumberFieldPolynomial)> {
        if rhs.poly.is_zero() {
            Err(exceptions::PyValueError::new_err("Division by zero"))
        } else {
            let (q, r) = self.poly.quot_rem(&rhs.poly, false);
            Ok((Self { poly: q }, Self { poly: r }))
        }
    }

    /// Negate the polynomial.
    pub fn __neg__(&self) -> Self {
        Self {
            poly: self.poly.clone().neg(),
        }
    }

    /// Compute the remainder `self % rhs.
    pub fn __mod__(&self, rhs: Self) -> PyResult<Self> {
        if rhs.poly.is_zero() {
            Err(exceptions::PyValueError::new_err("Division by zero"))
        } else {
            Ok(Self {
                poly: self.poly.rem(&rhs.poly),
            })
        }
    }

    /// Compute the greatest common divisor (GCD) of two or more polynomials.
    #[pyo3(signature = (*rhs))]
    pub fn gcd(
        &self,
        #[gen_stub(override_type(type_repr = "NumberFieldPolynomial"))] rhs: &Bound<'_, PyTuple>,
    ) -> PyResult<Self> {
        if rhs.len() == 1 {
            let rhs = rhs.get_item(0)?.extract::<Self>()?;
            if self.poly.ring != rhs.poly.ring {
                Err(exceptions::PyValueError::new_err(
                    "Polynomials have different rings".to_string(),
                ))
            } else {
                Ok(Self {
                    poly: self.poly.gcd(&rhs.poly),
                })
            }
        } else {
            let mut args = vec![self.poly.clone()];
            for r in rhs.iter() {
                let p = r.extract::<Self>()?;
                if args[0].ring != p.poly.ring {
                    return Err(exceptions::PyValueError::new_err(
                        "Polynomials have different rings".to_string(),
                    ));
                }
                args.push(p.poly);
            }

            Ok(Self {
                poly: PolynomialGCD::gcd_multiple(args),
            })
        }
    }

    /// Compute the resultant of two polynomials with respect to the variable `var`.
    pub fn resultant(&self, rhs: Self, var: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, var.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                var.__str__()?
            )))?;

        if self.poly.get_vars_ref() == rhs.poly.get_vars_ref() {
            let self_uni = self.poly.to_univariate(x);
            let rhs_uni = rhs.poly.to_univariate(x);

            Ok(Self {
                poly: self_uni.resultant_prs(&rhs_uni),
            })
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);

            let self_uni = new_self.to_univariate(x);
            let rhs_uni = new_rhs.to_univariate(x);

            Ok(Self {
                poly: self_uni.resultant_prs(&rhs_uni),
            })
        }
    }

    /// Compute the square-free factorization of the polynomial.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('3*(2*x^2+y)(x^3+y)^2(1+4*y)^2(1+x)').expand().to_polynomial()
    /// >>> print('Square-free factorization of {}:'.format(p))
    /// >>> for f, exp in p.factor_square_free():
    /// >>>     print('\t({})^{}'.format(f, exp))
    pub fn factor_square_free(&self) -> Vec<(PythonNumberFieldPolynomial, usize)> {
        self.poly
            .square_free_factorization()
            .into_iter()
            .map(|(f, p)| (Self { poly: f }, p))
            .collect()
    }

    /// Factorize the polynomial.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('(x+1)(x+2)(x+3)(x+4)(x+5)(x^2+6)(x^3+7)(x+8)(x^4+9)(x^5+x+10)').expand().to_polynomial()
    /// >>> print('Factorization of {}:'.format(p))
    /// >>> for f, exp in p.factor():
    /// >>>     print('\t({})^{}'.format(f, exp))
    pub fn factor(&self) -> Vec<(PythonNumberFieldPolynomial, usize)> {
        self.poly
            .factor()
            .into_iter()
            .map(|(f, p)| (Self { poly: f }, p))
            .collect()
    }

    /// Take a derivative in `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x^2+2').to_polynomial()
    /// >>> print(p.derivative(x))
    pub fn derivative(&self, x: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, x.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        Ok(Self {
            poly: self.poly.derivative(x),
        })
    }

    /// Get the content, i.e., the GCD of the coefficients.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('3x^2+6x+9').to_polynomial()
    /// >>> print(p.content())
    pub fn content(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.constant(self.poly.content()),
        })
    }

    /// Get the primitive part of the polynomial, i.e., the polynomial divided
    /// by its content.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('6x^2+3x+9').to_polynomial().primitive()
    /// >>> print(p)
    ///
    /// Yields `2*x^2+x+3`.
    pub fn primitive(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.clone().make_primitive(),
        })
    }

    /// Make the polynomial monic, i.e., divide by the leading coefficient.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> p = E('6x^2+3x+9').to_polynomial().monic()
    /// >>> print(p)
    ///
    /// Yields `x^2+1/2*x+3/2`.
    pub fn monic(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.clone().make_monic(),
        })
    }

    /// Get the leading coefficient.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> p = E('3x^2+6x+9').to_polynomial().lcoeff()
    /// >>> print(p)
    ///
    /// Yields `3`.
    pub fn lcoeff(&self) -> PyResult<Self> {
        Ok(Self {
            poly: self.poly.constant(self.poly.lcoeff().clone()),
        })
    }

    /// Get the coefficient list, optionally in the variables `vars`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial()
    /// >>> for n, pp in p.coefficient_list(x):
    /// >>>     print(n, pp)
    #[pyo3(signature = (vars = None))]
    pub fn coefficient_list(
        &self,
        vars: Option<OneOrMultiple<PythonExpression>>,
    ) -> PyResult<Vec<(Vec<usize>, PythonNumberFieldPolynomial)>> {
        if let Some(vv) = vars {
            let mut vars = vec![];

            for vvv in vv.to_iter() {
                let x = self
                    .poly
                    .get_vars_ref()
                    .iter()
                    .position(|v| match (v, vvv.expr.as_view()) {
                        (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                        (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => {
                            f.as_view() == a
                        }
                        _ => false,
                    })
                    .ok_or(exceptions::PyValueError::new_err(format!(
                        "Variable {} not found in polynomial",
                        vvv.__str__()?
                    )))?;

                vars.push(x);
            }

            if vars.is_empty() {
                return Ok(self
                    .poly
                    .into_iter()
                    .map(|t| {
                        (
                            t.exponents.iter().map(|x| *x as usize).collect(),
                            Self {
                                poly: self.poly.constant(t.coefficient.clone()),
                            },
                        )
                    })
                    .collect());
            }

            if vars.len() == 1 {
                return Ok(self
                    .poly
                    .to_univariate_polynomial_list(vars[0])
                    .into_iter()
                    .map(|(f, p)| (vec![p as usize], Self { poly: f }))
                    .collect());
            }

            // sort the exponents wrt the var map
            let mut r: Vec<(Vec<_>, _)> = self
                .poly
                .to_multivariate_polynomial_list(&vars, true)
                .into_iter()
                .map(|(f, p)| {
                    (
                        vars.iter().map(|v| f[*v] as usize).collect(),
                        Self { poly: p },
                    )
                })
                .collect();
            r.sort_by(|a, b| a.0.cmp(&b.0));

            Ok(r)
        } else {
            Ok(self
                .poly
                .into_iter()
                .map(|t| {
                    (
                        t.exponents.iter().map(|x| *x as usize).collect(),
                        Self {
                            poly: self.poly.constant(t.coefficient.clone()),
                        },
                    )
                })
                .collect())
        }
    }

    /// Replace the variable `x` with a polynomial `v`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x*y+2*x+x^2').to_polynomial()
    /// >>> r = E('y+1').to_polynomial())
    /// >>> p.replace(x, r)
    pub fn replace(&self, x: PythonExpression, v: PolynomialOrInteger<Self>) -> PyResult<Self> {
        let id = match x.expr.as_view() {
            AtomView::Var(x) => x.get_symbol(),
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Derivative must be taken wrt a variable",
                ));
            }
        };

        let v = match v {
            PolynomialOrInteger::Polynomial(p) => p.poly,
            PolynomialOrInteger::Integer(i) => {
                self.poly.constant(self.poly.ring.element_from_integer(i))
            }
        };

        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|x| match x {
                PolyVariable::Symbol(y) => *y == id,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        if self.poly.get_vars_ref() == v.get_vars_ref() {
            Ok(Self {
                poly: self.poly.replace_with_poly(x, &v),
            })
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = v;
            new_self.unify_variables(&mut new_rhs);
            Ok(Self {
                poly: new_self.replace_with_poly(x, &new_rhs),
            })
        }
    }

    /// Compute the Groebner basis of a polynomial system.
    ///
    /// If `grevlex=True`, reverse graded lexicographical ordering is used,
    /// otherwise the ordering is lexicographical.
    ///
    /// If `print_stats=True` intermediate statistics will be printed.
    #[pyo3(signature = (system, grevlex = true, print_stats = false))]
    #[classmethod]
    pub fn groebner_basis(
        _cls: &Bound<'_, PyType>,
        system: Vec<Self>,
        grevlex: bool,
        print_stats: bool,
    ) -> Vec<Self> {
        if grevlex {
            let grevlex_ideal: Vec<_> = system
                .iter()
                .map(|p| p.poly.reorder::<GrevLexOrder>())
                .collect();
            let gb = GroebnerBasis::new(&grevlex_ideal, print_stats);

            gb.system
                .into_iter()
                .map(|p| Self {
                    poly: p.reorder::<LexOrder>(),
                })
                .collect()
        } else {
            let ideal: Vec<_> = system.iter().map(|p| p.poly.clone()).collect();
            let gb = GroebnerBasis::new(&ideal, print_stats);
            gb.system.into_iter().map(|p| Self { poly: p }).collect()
        }
    }

    /// Completely reduce the polynomial w.r.t the polynomials `gs`.
    /// For example reducing `f=y^2+x` by `g=[x]` yields `y^2`.
    #[pyo3(signature = (system, grevlex = true))]
    pub fn reduce(&self, system: Vec<Self>, grevlex: bool) -> Self {
        if grevlex {
            let p = self.poly.reorder::<GrevLexOrder>();
            let grevlex_ideal: Vec<_> = system
                .iter()
                .map(|p| p.poly.reorder::<GrevLexOrder>())
                .collect();

            let r = p.reduce(&grevlex_ideal);
            Self {
                poly: r.reorder::<LexOrder>(),
            }
        } else {
            let ideal: Vec<_> = system.into_iter().map(|p| p.poly).collect();
            Self {
                poly: self.poly.reduce(&ideal),
            }
        }
    }

    /// Integrate the polynomial in `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('x^2+2').to_polynomial()
    /// >>> print(p.integrate(x))
    pub fn integrate(&self, x: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .get_vars_ref()
            .iter()
            .position(|v| match (v, x.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        Ok(Self {
            poly: self.poly.integrate(x),
        })
    }

    /// Convert the polynomial to an expression.
    pub fn to_expression(&self) -> PyResult<PythonExpression> {
        Ok(self
            .poly
            .to_expression_with_coeff_map(|_, element, out| {
                element.poly.to_expression_into(out);
            })
            .into())
    }

    /// Convert the polynomial to a polynomial over rationals.
    pub fn to_polynomial(&self) -> PyResult<PythonPolynomial> {
        let mut c = self.poly.clone();
        let mut min_poly = MultivariatePolynomial::new(
            &c.ring,
            None,
            Arc::new(self.poly.ring.poly().get_vars_ref().to_vec()),
        );
        c.unify_variables(&mut min_poly);

        let mut poly = MultivariatePolynomial::new(&Q, None, Arc::new(c.get_vars_ref().to_vec()));

        for term in c.into_iter() {
            let mut t = term.coefficient.poly.clone();
            poly.unify_variables(&mut t);
            poly = poly + t.mul_exp(&term.exponents);
        }
        Ok(PythonPolynomial { poly })
    }

    /// Get the minimal polynomial of the algebraic extension.
    pub fn get_minimal_polynomial(&self) -> PythonPolynomial {
        PythonPolynomial {
            poly: self.poly.ring.poly().clone(),
        }
    }
}

/// A Symbolica rational polynomial.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "RationalPolynomial", subclass, module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonRationalPolynomial {
    pub poly: RationalPolynomial<IntegerRing, u16>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonRationalPolynomial {
    /// Copy the rational polynomial.
    pub fn __copy__(&self) -> Self {
        Self {
            poly: self.poly.clone(),
        }
    }

    /// Compare two polynomials.
    fn __richcmp__(&self, o: Py<PyAny>, op: CompareOp, py: Python) -> PyResult<bool> {
        let Ok(other) = o.extract::<PolynomialOrInteger<PythonRationalPolynomial>>(py) else {
            return Err(exceptions::PyTypeError::new_err(
                "Can only compare Polynomial with Polynomial or integer.",
            ));
        };
        match op {
            CompareOp::Eq => match other {
                PolynomialOrInteger::Integer(i) => Ok(self.poly.denominator.is_one()
                    && self.poly.numerator.get_constant()
                        == self.poly.numerator.ring.element_from_integer(i)),
                PolynomialOrInteger::Polynomial(p) => Ok(self.poly == p.poly),
            },
            CompareOp::Ne => match other {
                PolynomialOrInteger::Integer(i) => Ok(!self.poly.denominator.is_one()
                    || self.poly.numerator.get_constant()
                        != self.poly.numerator.ring.element_from_integer(i)),
                PolynomialOrInteger::Polynomial(p) => Ok(self.poly != p.poly),
            },
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "Inequalities between polynomials are not allowed in {} {} {}",
                self.__str__()?,
                match op {
                    CompareOp::Eq => "==",
                    CompareOp::Ge => ">=",
                    CompareOp::Gt => ">",
                    CompareOp::Le => "<=",
                    CompareOp::Lt => "<",
                    CompareOp::Ne => "!=",
                },
                match other {
                    PolynomialOrInteger::Integer(i) => i.to_string(),
                    PolynomialOrInteger::Polynomial(p) => p.__str__()?,
                }
            ))),
        }
    }

    /// Get the list of variables in the internal ordering of the polynomial.
    pub fn get_variables(&self) -> PyResult<Vec<PythonExpression>> {
        let mut var_list = vec![];

        for x in self.poly.get_variables().iter() {
            match x {
                PolyVariable::Symbol(x) => {
                    var_list.push(Atom::var(*x).into());
                }
                PolyVariable::Temporary(_) => {
                    Err(exceptions::PyValueError::new_err(
                        "Temporary variable in polynomial".to_string(),
                    ))?;
                }
                PolyVariable::Function(_, a) | PolyVariable::Power(a) => {
                    var_list.push(a.as_ref().clone().into());
                }
            }
        }

        Ok(var_list)
    }

    /// Convert the rational polynomial into a portable string.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new()))
    }

    /// Print the rational polynomial in a human-readable format.
    pub fn __str__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&DEFAULT_PRINT_OPTIONS, PrintState::new()))
    }

    /// Convert the rational polynomial into a LaTeX string.
    pub fn to_latex(&self) -> PyResult<String> {
        Ok(format!(
            "$${}$$",
            self.poly
                .format_string(&LATEX_PRINT_OPTIONS, PrintState::new())
        ))
    }

    /// Add two rational polynomials `self and `rhs`, returning the result.
    pub fn __add__(&self, rhs: Self) -> Self {
        if self.poly.get_variables() == rhs.poly.get_variables() {
            Self {
                poly: &self.poly + &rhs.poly,
            }
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);
            Self {
                poly: &new_self + &new_rhs,
            }
        }
    }

    /// Subtract rational polynomials `rhs` from `self`, returning the result.
    pub fn __sub__(&self, rhs: Self) -> Self {
        if self.poly.get_variables() == rhs.poly.get_variables() {
            Self {
                poly: &self.poly - &rhs.poly,
            }
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);
            Self {
                poly: &new_self - &new_rhs,
            }
        }
    }

    /// Multiply two rational polynomials `self and `rhs`, returning the result.
    pub fn __mul__(&self, rhs: Self) -> Self {
        if self.poly.get_variables() == rhs.poly.get_variables() {
            Self {
                poly: &self.poly * &rhs.poly,
            }
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);
            Self {
                poly: &new_self * &new_rhs,
            }
        }
    }

    /// Divide the rational polynomial `self` by `rhs` if possible, returning the result.
    pub fn __truediv__(&self, rhs: Self) -> Self {
        if self.poly.get_variables() == rhs.poly.get_variables() {
            Self {
                poly: &self.poly / &rhs.poly,
            }
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);
            Self {
                poly: &new_self / &new_rhs,
            }
        }
    }

    /// Negate the rational polynomial.
    pub fn __neg__(&self) -> Self {
        Self {
            poly: self.poly.clone().neg(),
        }
    }

    /// Compute the greatest common divisor (GCD) of two rational polynomials.
    pub fn gcd(&self, rhs: Self) -> Self {
        if self.poly.get_variables() == rhs.poly.get_variables() {
            Self {
                poly: self.poly.gcd(&rhs.poly),
            }
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);
            Self {
                poly: new_self.gcd(&new_rhs),
            }
        }
    }

    /// Take a derivative in `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
    /// >>> print(p.derivative(x))
    pub fn derivative(&self, x: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .numerator
            .get_vars_ref()
            .iter()
            .position(|v| match (v, x.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        Ok(Self {
            poly: self.poly.derivative(x),
        })
    }

    /// Compute the partial fraction decomposition in `x`. If `x` is `None`,
    /// compute the multivariate partial fraction decomposition.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
    /// >>> for pp in p.apart(x):
    /// >>>     print(pp)
    #[pyo3(signature = (x = None))]
    pub fn apart(&self, x: Option<PythonExpression>) -> PyResult<Vec<Self>> {
        if let Some(x) = x {
            let id = match x.expr.as_view() {
                AtomView::Var(x) => x.get_symbol(),
                _ => {
                    return Err(exceptions::PyValueError::new_err(
                        "Invalid variable specified.",
                    ));
                }
            };

            let x = self
                .poly
                .get_variables()
                .iter()
                .position(|x| match x {
                    PolyVariable::Symbol(y) => *y == id,
                    _ => false,
                })
                .ok_or(exceptions::PyValueError::new_err(format!(
                    "Variable {} not found in polynomial",
                    x.__str__()?
                )))?;

            Ok(self
                .poly
                .apart(x)
                .into_iter()
                .map(|f| Self { poly: f })
                .collect())
        } else {
            Ok(self
                .poly
                .apart_multivariate()
                .into_iter()
                .map(|f| Self { poly: f })
                .collect())
        }
    }

    /// Create a new rational polynomial from a numerator and denominator polynomial.
    #[new]
    pub fn __new__(num: &PythonPolynomial, den: &PythonPolynomial) -> Self {
        Self {
            poly: RationalPolynomial::from_num_den(num.poly.clone(), den.poly.clone(), &Z, true),
        }
    }

    /// Convert the coefficients to finite fields with prime `prime`.
    pub fn to_finite_field(&self, prime: u64) -> PythonFiniteFieldRationalPolynomial {
        PythonFiniteFieldRationalPolynomial {
            poly: self.poly.to_finite_field(&Zp64::new(prime)),
        }
    }

    /// Get the numerator.
    pub fn numerator(&self) -> PythonPolynomial {
        PythonPolynomial {
            poly: (&self.poly.numerator).into(),
        }
    }

    /// Get the denominator.
    pub fn denominator(&self) -> PythonPolynomial {
        PythonPolynomial {
            poly: (&self.poly.denominator).into(),
        }
    }

    /// Parse a rational polynomial from a string.
    /// The list of all the variables must be provided.
    ///
    /// If this requirements is too strict, use `Expression.to_polynomial()` instead.
    ///
    ///
    /// Examples
    /// --------
    /// >>> e = Polynomial.parse('3/4*x^2+y+y*4', ['x', 'y'])
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the input is not a valid Symbolica rational polynomial.
    #[pyo3(signature = (arg, vars, default_namespace = "python"))]
    #[classmethod]
    pub fn parse(
        _cls: &Bound<'_, PyType>,
        arg: &str,
        vars: Vec<PyBackedStr>,
        default_namespace: &str,
    ) -> PyResult<Self> {
        let mut var_map = vec![];
        let mut var_name_map = vec![];

        let namespace = DefaultNamespace {
            namespace: default_namespace.to_string().into(),
            data: "",
            file: "".into(),
            line: 0,
        };

        for v in vars {
            let id = Symbol::new(namespace.attach_namespace(&v)).build().unwrap();
            var_map.push(id.into());
            var_name_map.push((*v).into());
        }

        let e = Token::parse(arg, ParseSettings::polynomial())
            .map_err(exceptions::PyValueError::new_err)?
            .to_rational_polynomial(&Q, &Z, &Arc::new(var_map), &var_name_map)
            .map_err(exceptions::PyValueError::new_err)?;

        Ok(Self { poly: e })
    }

    /// Convert the rational polynomial to an expression.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> e = E('(x*y+2*x+x^2)/(1+y^2+x^7)')
    /// >>> p = e.to_rational_polynomial()
    /// >>> print((e - p.to_expression()).expand())
    pub fn to_expression(&self) -> PyResult<PythonExpression> {
        Ok(self.poly.to_expression().into())
    }
}

/// A Symbolica rational polynomial over finite fields.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(
    name = "FiniteFieldRationalPolynomial",
    subclass,
    module = "symbolica.core"
)]
#[derive(Clone)]
pub struct PythonFiniteFieldRationalPolynomial {
    pub poly: RationalPolynomial<Zp64, u16>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonFiniteFieldRationalPolynomial {
    /// Copy the rational polynomial.
    pub fn __copy__(&self) -> Self {
        Self {
            poly: self.poly.clone(),
        }
    }

    /// Compare two polynomials.
    fn __richcmp__(&self, o: Py<PyAny>, op: CompareOp, py: Python) -> PyResult<bool> {
        let Ok(other) = o.extract::<PolynomialOrInteger<PythonFiniteFieldRationalPolynomial>>(py)
        else {
            return Err(exceptions::PyTypeError::new_err(
                "Can only compare Polynomial with Polynomial or integer.",
            ));
        };
        match op {
            CompareOp::Eq => match other {
                PolynomialOrInteger::Integer(i) => Ok(self.poly.denominator.is_one()
                    && self.poly.numerator.get_constant()
                        == self.poly.numerator.ring.element_from_integer(i)),
                PolynomialOrInteger::Polynomial(p) => Ok(self.poly == p.poly),
            },
            CompareOp::Ne => match other {
                PolynomialOrInteger::Integer(i) => Ok(!self.poly.denominator.is_one()
                    || self.poly.numerator.get_constant()
                        != self.poly.numerator.ring.element_from_integer(i)),
                PolynomialOrInteger::Polynomial(p) => Ok(self.poly != p.poly),
            },
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "Inequalities between polynomials are not allowed in {} {} {}",
                self.__str__()?,
                match op {
                    CompareOp::Eq => "==",
                    CompareOp::Ge => ">=",
                    CompareOp::Gt => ">",
                    CompareOp::Le => "<=",
                    CompareOp::Lt => "<",
                    CompareOp::Ne => "!=",
                },
                match other {
                    PolynomialOrInteger::Integer(i) => i.to_string(),
                    PolynomialOrInteger::Polynomial(p) => p.__str__()?,
                }
            ))),
        }
    }

    /// Get the list of variables in the internal ordering of the polynomial.
    pub fn get_variables(&self) -> PyResult<Vec<PythonExpression>> {
        let mut var_list = vec![];

        for x in self.poly.get_variables().iter() {
            match x {
                PolyVariable::Symbol(x) => {
                    var_list.push(Atom::var(*x).into());
                }
                PolyVariable::Temporary(_) => {
                    Err(exceptions::PyValueError::new_err(
                        "Temporary variable in polynomial".to_string(),
                    ))?;
                }
                PolyVariable::Function(_, a) | PolyVariable::Power(a) => {
                    var_list.push(a.as_ref().clone().into());
                }
            }
        }

        Ok(var_list)
    }

    /// Convert the rational polynomial into a portable string.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new()))
    }

    /// Print the rational polynomial in a human-readable format.
    pub fn __str__(&self) -> PyResult<String> {
        Ok(self
            .poly
            .format_string(&DEFAULT_PRINT_OPTIONS, PrintState::new()))
    }

    /// Convert the rational polynomial into a LaTeX string.
    pub fn to_latex(&self) -> PyResult<String> {
        Ok(format!(
            "$${}$$",
            self.poly
                .format_string(&LATEX_PRINT_OPTIONS, PrintState::new())
        ))
    }

    /// Add two rational polynomials `self and `rhs`, returning the result.
    pub fn __add__(&self, rhs: Self) -> Self {
        if self.poly.get_variables() == rhs.poly.get_variables() {
            Self {
                poly: &self.poly + &rhs.poly,
            }
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);
            Self {
                poly: &new_self + &new_rhs,
            }
        }
    }

    /// Subtract rational polynomials `rhs` from `self`, returning the result.
    pub fn __sub__(&self, rhs: Self) -> Self {
        if self.poly.get_variables() == rhs.poly.get_variables() {
            Self {
                poly: &self.poly - &rhs.poly,
            }
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);
            Self {
                poly: &new_self - &new_rhs,
            }
        }
    }

    /// Multiply two rational polynomials `self and `rhs`, returning the result.
    pub fn __mul__(&self, rhs: Self) -> Self {
        if self.poly.get_variables() == rhs.poly.get_variables() {
            Self {
                poly: &self.poly * &rhs.poly,
            }
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);
            Self {
                poly: &new_self * &new_rhs,
            }
        }
    }

    /// Divide the rational polynomial `self` by `rhs` if possible, returning the result.
    pub fn __truediv__(&self, rhs: Self) -> Self {
        if self.poly.get_variables() == rhs.poly.get_variables() {
            Self {
                poly: &self.poly / &rhs.poly,
            }
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);
            Self {
                poly: &new_self / &new_rhs,
            }
        }
    }

    /// Negate the rational polynomial.
    pub fn __neg__(&self) -> Self {
        Self {
            poly: self.poly.clone().neg(),
        }
    }

    /// Compute the greatest common divisor (GCD) of two rational polynomials.
    pub fn gcd(&self, rhs: Self) -> Self {
        if self.poly.get_variables() == rhs.poly.get_variables() {
            Self {
                poly: self.poly.gcd(&rhs.poly),
            }
        } else {
            let mut new_self = self.poly.clone();
            let mut new_rhs = rhs.poly.clone();
            new_self.unify_variables(&mut new_rhs);
            Self {
                poly: new_self.gcd(&new_rhs),
            }
        }
    }

    /// Get the modulus of the finite field.
    pub fn get_modulus(&self) -> u64 {
        self.poly.numerator.ring.get_prime()
    }

    /// Take a derivative in `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
    /// >>> print(p.derivative(x))
    pub fn derivative(&self, x: PythonExpression) -> PyResult<Self> {
        let x = self
            .poly
            .numerator
            .get_vars_ref()
            .iter()
            .position(|v| match (v, x.expr.as_view()) {
                (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        Ok(Self {
            poly: self.poly.derivative(x),
        })
    }

    /// Compute the partial fraction decomposition in `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> p = E('1/((x+y)*(x^2+x*y+1)(x+1))').to_rational_polynomial()
    /// >>> for pp in p.apart(x):
    /// >>>     print(pp)
    pub fn apart(&self, x: PythonExpression) -> PyResult<Vec<Self>> {
        let id = match x.expr.as_view() {
            AtomView::Var(x) => x.get_symbol(),
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Invalid variable specified.",
                ));
            }
        };

        let x = self
            .poly
            .get_variables()
            .iter()
            .position(|x| match x {
                PolyVariable::Symbol(y) => *y == id,
                _ => false,
            })
            .ok_or(exceptions::PyValueError::new_err(format!(
                "Variable {} not found in polynomial",
                x.__str__()?
            )))?;

        Ok(self
            .poly
            .apart(x)
            .into_iter()
            .map(|f| Self { poly: f })
            .collect())
    }

    /// Parse a rational polynomial from a string.
    /// The list of all the variables must be provided.
    ///
    /// If this requirements is too strict, use `Expression.to_polynomial()` instead.
    ///
    ///
    /// Examples
    /// --------
    /// >>> e = Polynomial.parse('3/4*x^2+y+y*4', ['x', 'y'])
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the input is not a valid Symbolica rational polynomial.
    #[pyo3(signature = (arg, vars, prime, default_namespace = "python"))]
    #[classmethod]
    pub fn parse(
        _cls: &Bound<'_, PyType>,
        arg: &str,
        vars: Vec<PyBackedStr>,
        prime: u64,
        default_namespace: &str,
    ) -> PyResult<Self> {
        let mut var_map = vec![];
        let mut var_name_map = vec![];

        let namespace = DefaultNamespace {
            namespace: default_namespace.to_string().into(),
            data: "",
            file: "".into(),
            line: 0,
        };

        for v in vars {
            let id = Symbol::new(namespace.attach_namespace(&v)).build().unwrap();
            var_map.push(id.into());
            var_name_map.push((*v).into());
        }

        let field = Zp64::new(prime);
        let e = Token::parse(arg, ParseSettings::polynomial())
            .map_err(exceptions::PyValueError::new_err)?
            .to_rational_polynomial(&field, &field, &Arc::new(var_map), &var_name_map)
            .map_err(exceptions::PyValueError::new_err)?;

        Ok(Self { poly: e })
    }
}

#[derive(FromPyObject)]
pub enum ConvertibleToRationalPolynomial {
    Literal(PythonRationalPolynomial),
    Expression(ConvertibleToExpression),
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(ConvertibleToRationalPolynomial = PythonRationalPolynomial | PythonExpression);

impl ConvertibleToRationalPolynomial {
    pub fn to_rational_polynomial(self) -> PyResult<PythonRationalPolynomial> {
        match self {
            Self::Literal(l) => Ok(l),
            Self::Expression(e) => {
                let expr = &e.to_expression().expr;

                let poly = expr.try_to_rational_polynomial(&Q, &Z, None).map_err(|_| {
                    exceptions::PyValueError::new_err(
                        "Expression cannot be converted to a rational polynomial.".to_string(),
                    )
                })?;

                Ok(PythonRationalPolynomial { poly })
            }
        }
    }
}

/// An optimized evaluator for expressions.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "Evaluator", module = "symbolica.core")]
pub struct PythonExpressionEvaluator {
    pub eval_rat: ExpressionEvaluator<Complex<Rational>>,
    pub eval: Option<ExpressionEvaluatorWithExternalFunctions<f64>>,
    pub eval_complex: ExpressionEvaluator<Complex<f64>>,
    pub eval_complex_ext: ExpressionEvaluatorWithExternalFunctions<Complex<f64>>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonExpressionEvaluator {
    /// Return the instructions for efficiently evaluating the expression, the length of the list
    /// of temporary variables, and the list of constants. This can be used to generate
    /// code for the expression evaluation in any programming language.
    ///
    /// There are four lists that are used in the evaluation instructions:
    /// - `param`: the list of input parameters.
    /// - `temp`: the list of temporary slots. The size of it is provided as the second return value.
    /// - `const`: the list of constants.
    /// - `out`: the list of outputs.
    ///
    /// The instructions are of the form:
    /// - `('add', ('out', 0), [('const', 1), ('param', 0)])` which means `out[0] = const[1] + param[0]`.
    /// - `('mul', ('out', 0), [('temp', 0), ('param', 0)])` which means `out[0] = temp[0] * param[0]`.
    /// - `('pow', ('out', 0), ('param', 0), -1)` which means `out[0] = param[0]^-1`.
    /// - `('powf', ('out', 0), ('param', 0), ('param', 1))` which means `out[0] = param[0]^param[1]`.
    /// - `('fun', ('temp', 1), cos, ('param', 0))` which means `temp[1] = cos(param[0])`.
    /// - `('external_fun', ('temp', 1), f, [('param', 0)])` which means `temp[1] = f(param[0])`.
    /// - `('assign', ('out', 1), ('const', 2))` which means `out[1] = const[2]`.
    /// - `('if_else', ('temp', 0), 5)` which means `if temp[0] != 0 goto label 5`.
    /// - `('goto', 10)` which means `goto label 10`.
    /// - `('label', 3)` which means `label 3`.
    /// - `('join', ('out', 0), ('temp', 0), 3, 7)` which means `out[0] = (temp[0] != 0) ? label 3 : label 7`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> (ins, m, c) = E('x^2+5/3+cos(x)').evaluator({}, {}, [S('x')]).get_instructions()
    /// >>>
    /// >>> for x in ins:
    /// >>>     print(x)
    /// >>> print('temp list length:', m)
    /// >>> print('constants:', c)
    ///
    /// yields
    ///
    /// ```log
    /// ('mul', ('out', 0), [('param', 0), ('param', 0)])
    /// ('fun', ('temp', 1), cos, ('param', 0))
    /// ('add', ('out', 0), [('const', 0), ('out', 0), ('temp', 1)])
    /// temp list length: 2
    /// constants: [5/3]
    /// ```
    fn get_instructions<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(Vec<Bound<'py, PyTuple>>, usize, Vec<PythonExpression>)> {
        let (instr, max, consts) = self.eval_rat.export_instructions();

        fn slot_to_object(slot: &Slot) -> (&str, usize) {
            match slot {
                Slot::Const(x) => ("const", *x),
                Slot::Param(x) => ("param", *x),
                Slot::Temp(x) => ("temp", *x),
                Slot::Out(x) => ("out", *x),
            }
        }

        let mut v = vec![];
        for i in &instr {
            match i {
                Instruction::Add(o, s) | Instruction::Mul(o, s) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            if matches!(i, Instruction::Add(_, _)) {
                                "add"
                            } else {
                                "mul"
                            }
                            .into_pyobject(py)?
                            .as_any(),
                            slot_to_object(o).into_pyobject(py)?.as_any(),
                            s.iter()
                                .map(slot_to_object)
                                .collect::<Vec<_>>()
                                .into_pyobject(py)?
                                .as_any(),
                        ],
                    )?);
                }
                Instruction::Pow(o, b, e) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "pow".into_pyobject(py)?.as_any(),
                            slot_to_object(o).into_pyobject(py)?.as_any(),
                            slot_to_object(b).into_pyobject(py)?.as_any(),
                            e.into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
                Instruction::Powf(o, b, e) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "powf".into_pyobject(py)?.as_any(),
                            slot_to_object(o).into_pyobject(py)?.as_any(),
                            slot_to_object(b).into_pyobject(py)?.as_any(),
                            slot_to_object(e).into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
                Instruction::Fun(o, f, s) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "fun".into_pyobject(py)?.as_any(),
                            slot_to_object(o).into_pyobject(py)?.as_any(),
                            PythonExpression::from(Atom::var(f.get_symbol()))
                                .into_pyobject(py)?
                                .as_any(),
                            slot_to_object(s).into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
                Instruction::ExternalFun(o, f, s) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "external_fun".into_pyobject(py)?.as_any(),
                            slot_to_object(o).into_pyobject(py)?.as_any(),
                            f.into_pyobject(py)?.as_any(),
                            s.iter()
                                .map(slot_to_object)
                                .collect::<Vec<_>>()
                                .into_pyobject(py)?
                                .as_any(),
                        ],
                    )?);
                }
                Instruction::Assign(o, r) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "assign".into_pyobject(py)?.as_any(),
                            slot_to_object(o).into_pyobject(py)?.as_any(),
                            slot_to_object(r).into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
                Instruction::IfElse(cond, label) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "if_else".into_pyobject(py)?.as_any(),
                            slot_to_object(cond).into_pyobject(py)?.as_any(),
                            label.into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
                Instruction::Join(o, cond, t, f) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "join".into_pyobject(py)?.as_any(),
                            slot_to_object(o).into_pyobject(py)?.as_any(),
                            slot_to_object(cond).into_pyobject(py)?.as_any(),
                            slot_to_object(t).into_pyobject(py)?.as_any(),
                            slot_to_object(f).into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
                Instruction::Goto(label) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "goto".into_pyobject(py)?.as_any(),
                            label.into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
                Instruction::Label(label) => {
                    v.push(PyTuple::new(
                        py,
                        [
                            "label".into_pyobject(py)?.as_any(),
                            label.into_pyobject(py)?.as_any(),
                        ],
                    )?);
                }
            }
        }
        Ok((
            v,
            max,
            consts.into_iter().map(|x| Atom::num(x).into()).collect(),
        ))
    }

    /// Merge evaluator `other` into `self`. The parameters must be the same, and
    /// the outputs will be concatenated.
    ///
    /// The optional `cpe_rounds` parameter can be used to limit the number of common
    /// pair elimination rounds after the merge.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> e1 = E('x').evaluator({}, {}, [S('x')])
    /// >>> e2 = E('x+1').evaluator({}, {}, [S('x')])
    /// >>> e = e1.merge(e2)
    /// >>> e.evaluate([2])
    ///
    /// yields `[2, 3]`.
    #[pyo3(signature = (other, cpe_iterations = None))]
    fn merge(
        &mut self,
        other: &PythonExpressionEvaluator,
        cpe_iterations: Option<usize>,
    ) -> PyResult<()> {
        self.eval_rat
            .merge(other.eval_rat.clone(), cpe_iterations)
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not merge evaluators: {e}",))
            })?;

        if self.eval_rat.is_real()
            && let Some(old_eval) = &mut self.eval
        {
            let new_eval_rat = self
                .eval_rat
                .clone()
                .map_coeff(&|x| x.to_real().unwrap().to_f64());

            old_eval.update_stack(new_eval_rat);
        } else {
            self.eval = None;
        };

        self.eval_complex = self
            .eval_rat
            .clone()
            .map_coeff(&|x| Complex::new(x.re.to_f64(), x.im.to_f64()));

        self.eval_complex_ext
            .update_stack(self.eval_complex.clone());

        Ok(())
    }

    /// Evaluate the expression for multiple inputs that are flattened and return the flattened result.
    /// This method has less overhead than `evaluate`.
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.float64]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLike2<'py, f64, TypeMustMatch>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let eval = self.eval.as_mut().ok_or(exceptions::PyValueError::new_err(
            "Evaluator contains complex coefficients. Use evaluate_complex_flat instead.",
        ))?;

        let arr = inputs.as_array();

        if inputs.shape()[1] != self.eval_rat.get_input_len() {
            return Err(exceptions::PyValueError::new_err(format!(
                "Input length mismatch: expected {}, got {}",
                self.eval_rat.get_input_len(),
                inputs.shape()[1]
            )));
        }
        let n_inputs = inputs.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.eval_rat.get_output_len()][..]);
        for (i, mut o) in arr.axis_iter(Axis(0)).zip(out.axis_iter_mut(Axis(0))) {
            eval.evaluate(
                i.as_slice().ok_or_else(|| {
                    exceptions::PyValueError::new_err("Failed to convert input to slice")
                })?,
                o.as_slice_mut().unwrap(),
            );
        }

        Ok(out.into_pyarray(py))
    }

    /// Evaluate the expression for multiple inputs that are flattened and return the flattened result.
    /// This method has less overhead than `evaluate_complex`.
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.complex128]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate_complex<'py>(
        &mut self,
        py: Python<'py>,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLike2<'py, Complex64, TypeMustMatch>,
    ) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
        let arr = inputs.as_array();

        if inputs.shape()[1] != self.eval_rat.get_input_len() {
            return Err(exceptions::PyValueError::new_err(format!(
                "Input length mismatch: expected {}, got {}",
                self.eval_rat.get_input_len(),
                inputs.shape()[1]
            )));
        }
        let n_inputs = inputs.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.eval_rat.get_output_len()][..]);
        for (i, mut o) in arr.axis_iter(Axis(0)).zip(out.axis_iter_mut(Axis(0))) {
            let sc = unsafe {
                std::mem::transmute::<&[Complex64], &[Complex<f64>]>(i.as_slice().unwrap())
            };
            let os = unsafe {
                std::mem::transmute::<&mut [Complex64], &mut [Complex<f64>]>(
                    o.as_slice_mut().unwrap(),
                )
            };

            self.eval_complex.evaluate(sc, os);
        }

        Ok(out.into_pyarray(py))
    }

    /// Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.
    #[gen_stub(skip)]
    #[pyo3(signature =
        (function_name,
        filename,
        library_name,
        number_type,
        inline_asm = "default",
        optimization_level = 3,
        native = true,
        compiler_path = None,
        compiler_flags = None,
        custom_header = None,
        cuda_number_of_evaluations = 1,
        cuda_block_size = 512
    ))]
    fn compile(
        &self,
        function_name: &str,
        filename: &str,
        library_name: &str,
        number_type: &str,
        inline_asm: &str,
        optimization_level: u8,
        native: bool,
        compiler_path: Option<&str>,
        compiler_flags: Option<Vec<String>>,
        custom_header: Option<String>,
        cuda_number_of_evaluations: usize,
        cuda_block_size: usize,
        py: Python<'_>,
    ) -> PyResult<Py<PyAny>> {
        let mut options = match number_type {
            "real" | "complex" => CompileOptions {
                optimization_level: optimization_level as usize,
                native,
                ..f64::get_default_compile_options()
            },
            "real_4x" | "complex_4x" => CompileOptions {
                optimization_level: optimization_level as usize,
                native,
                ..<wide::f64x4>::get_default_compile_options()
            },
            "cuda_real" | "cuda_complex" => CompileOptions {
                optimization_level: optimization_level as usize,
                ..CudaRealf64::get_default_compile_options()
            },
            _ => {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Invalid number type {} specified.",
                    number_type,
                )));
            }
        };

        if let Some(compiler_path) = compiler_path {
            options.compiler = compiler_path.to_string();
        }

        if let Some(compiler_flags) = compiler_flags {
            options.args = compiler_flags;
        }

        let inline_asm = match inline_asm.to_lowercase().as_str() {
            "default" => InlineASM::default(),
            "x64" => InlineASM::X64,
            "avx2" => InlineASM::AVX2,
            "aarch64" => InlineASM::AArch64,
            "none" => InlineASM::None,
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Invalid inline assembly type specified.",
                ));
            }
        };

        match number_type {
            "real" => PythonCompiledRealExpressionEvaluator {
                eval: self
                    .eval_complex
                    .export_cpp::<f64>(
                        filename,
                        function_name,
                        ExportSettings {
                            include_header: true,
                            inline_asm,
                            custom_header,
                            ..Default::default()
                        },
                    )
                    .map_err(|e| exceptions::PyValueError::new_err(format!("Export error: {}", e)))?
                    .compile(library_name, options)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Compilation error: {}", e))
                    })?
                    .load()
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Library loading error: {}", e))
                    })?,
                input_len: self.eval_rat.get_input_len(),
                output_len: self.eval_rat.get_output_len(),
            }
            .into_py_any(py),
            "complex" => PythonCompiledComplexExpressionEvaluator {
                eval: self
                    .eval_complex
                    .export_cpp::<Complex<f64>>(
                        filename,
                        function_name,
                        ExportSettings {
                            include_header: true,
                            inline_asm,
                            custom_header,
                            ..Default::default()
                        },
                    )
                    .map_err(|e| exceptions::PyValueError::new_err(format!("Export error: {}", e)))?
                    .compile(library_name, options)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Compilation error: {}", e))
                    })?
                    .load()
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Library loading error: {}", e))
                    })?,
                input_len: self.eval_rat.get_input_len(),
                output_len: self.eval_rat.get_output_len(),
            }
            .into_py_any(py),
            "real_4x" => PythonCompiledSimdRealExpressionEvaluator {
                eval: self
                    .eval_complex
                    .export_cpp::<wide::f64x4>(
                        filename,
                        function_name,
                        ExportSettings {
                            include_header: true,
                            inline_asm,
                            custom_header,
                            ..Default::default()
                        },
                    )
                    .map_err(|e| exceptions::PyValueError::new_err(format!("Export error: {}", e)))?
                    .compile(library_name, options)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Compilation error: {}", e))
                    })?
                    .load()
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Library loading error: {}", e))
                    })?,
                input_len: self.eval_rat.get_input_len(),
                output_len: self.eval_rat.get_output_len(),
            }
            .into_py_any(py),
            "complex_4x" => PythonCompiledSimdComplexExpressionEvaluator {
                eval: self
                    .eval_complex
                    .export_cpp::<Complex<wide::f64x4>>(
                        filename,
                        function_name,
                        ExportSettings {
                            include_header: true,
                            inline_asm,
                            custom_header,
                            ..Default::default()
                        },
                    )
                    .map_err(|e| exceptions::PyValueError::new_err(format!("Export error: {}", e)))?
                    .compile(library_name, options)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Compilation error: {}", e))
                    })?
                    .load()
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Library loading error: {}", e))
                    })?,
                input_len: self.eval_rat.get_input_len(),
                output_len: self.eval_rat.get_output_len(),
            }
            .into_py_any(py),
            "cuda_real" => PythonCompiledCudaRealExpressionEvaluator {
                eval: self
                    .eval_complex
                    .export_cpp::<CudaRealf64>(
                        filename,
                        function_name,
                        ExportSettings {
                            include_header: true,
                            inline_asm,
                            custom_header,
                            ..Default::default()
                        },
                    )
                    .map_err(|e| exceptions::PyValueError::new_err(format!("Export error: {}", e)))?
                    .compile(library_name, options)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Compilation error: {}", e))
                    })?
                    .load_with_settings(CudaLoadSettings {
                        number_of_evaluations: cuda_number_of_evaluations,
                        block_size: cuda_block_size,
                    })
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Library loading error: {}", e))
                    })?,
                input_len: self.eval_rat.get_input_len(),
                output_len: self.eval_rat.get_output_len(),
            }
            .into_py_any(py),
            "cuda_complex" => PythonCompiledCudaComplexExpressionEvaluator {
                eval: self
                    .eval_complex
                    .export_cpp::<CudaComplexf64>(
                        filename,
                        function_name,
                        ExportSettings {
                            include_header: true,
                            inline_asm,
                            custom_header,
                            ..Default::default()
                        },
                    )
                    .map_err(|e| exceptions::PyValueError::new_err(format!("Export error: {}", e)))?
                    .compile(library_name, options)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Compilation error: {}", e))
                    })?
                    .load_with_settings(CudaLoadSettings {
                        number_of_evaluations: cuda_number_of_evaluations,
                        block_size: cuda_block_size,
                    })
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Library loading error: {}", e))
                    })?,
                input_len: self.eval_rat.get_input_len(),
                output_len: self.eval_rat.get_output_len(),
            }
            .into_py_any(py),
            _ => Err(exceptions::PyValueError::new_err(format!(
                "Invalid number type {} specified.",
                number_type,
            ))),
        }
    }
}

#[cfg(feature = "python_stubgen")]
static ONE: fn() -> String = || "1".into();
#[cfg(feature = "python_stubgen")]
static THREE: fn() -> String = || "3".into();
#[cfg(feature = "python_stubgen")]
static CUDA_BLOCK_DEFAULT: fn() -> String = || "256".into();
#[cfg(feature = "python_stubgen")]
static DEFAULT: fn() -> String = || "\"default\"".into();

#[cfg(feature = "python_stubgen")]
submit! {
PyMethodsInfo {
        struct_id: std::any::TypeId::of::<PythonExpressionEvaluator>,
        attrs: &[],
        getters: &[],
        setters: &[],
        methods: &[
            MethodInfo {
            name: "compile",
            parameters: &[
                ParameterInfo {
                    name: "function_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "filename",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "library_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "number_type",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || TypeInfo::unqualified("typing.Literal['real']"),
                },
                ParameterInfo {
                    name: "inline_asm",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(DEFAULT),
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "optimization_level",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(THREE),
                    type_info: || Option::<u8>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_path",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_flags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "custom_header",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },

            ],
            r#type: MethodType::Class,
            r#return: || PythonCompiledRealExpressionEvaluator::type_output(),
            doc:
r#"Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

Parameters
----------
function_name : str
    The name of the function to generate and compile.
filename : str
    The name of the file to generate.
library_name : str
    The name of the shared library to generate.
number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
    The type of numbers to use. Can be 'real' for double or 'complex' for complex double.
    For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
    For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
inline_asm : str
    The inline ASM option can be set to 'default', 'x64', 'aarch64' or 'none'.
optimization_level : int
    The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
compiler_path : Optional[str]
    The custom path to the compiler executable.
compiler_flags : Optional[Sequence[str]]
    The custom flags to pass to the compiler.
custom_header : Optional[str]
    The custom header to include in the generated code."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
        },
        MethodInfo {
            name: "compile",
            parameters: &[
                ParameterInfo {
                    name: "function_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "filename",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "library_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "number_type",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || TypeInfo::unqualified("typing.Literal['complex']"),
                },
                ParameterInfo {
                    name: "inline_asm",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(DEFAULT),
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "optimization_level",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(THREE),
                    type_info: || Option::<u8>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_path",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_flags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "custom_header",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },

            ],
            r#type: MethodType::Class,
            r#return: || PythonCompiledComplexExpressionEvaluator::type_output(),
            doc:
r#"Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

Parameters
----------
function_name : str
    The name of the function to generate and compile.
filename : str
    The name of the file to generate.
library_name : str
    The name of the shared library to generate.
number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
    The type of numbers to use. Can be 'real' for double or 'complex' for complex double.
    For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
    For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
inline_asm : str
    The inline ASM option can be set to 'default', 'x64', 'aarch64' or 'none'.
optimization_level : int
    The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
compiler_path : Optional[str]
    The custom path to the compiler executable.
compiler_flags : Optional[Sequence[str]]
    The custom flags to pass to the compiler.
custom_header : Optional[str]
    The custom header to include in the generated code."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
        },
        MethodInfo {
            name: "compile",
            parameters: &[
                ParameterInfo {
                    name: "function_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "filename",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "library_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "number_type",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || TypeInfo::unqualified("typing.Literal['real_4x']"),
                },
                ParameterInfo {
                    name: "inline_asm",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(DEFAULT),
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "optimization_level",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(THREE),
                    type_info: || Option::<u8>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_path",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_flags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "custom_header",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },

            ],
            r#type: MethodType::Class,
            r#return: || PythonCompiledSimdRealExpressionEvaluator::type_output(),
            doc:
r#"Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

Parameters
----------
function_name : str
    The name of the function to generate and compile.
filename : str
    The name of the file to generate.
library_name : str
    The name of the shared library to generate.
number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
    The type of numbers to use. Can be 'real' for double or 'complex' for complex double.
    For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
    For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
inline_asm : str
    The inline ASM option can be set to 'default', 'x64', 'aarch64' or 'none'.
optimization_level : int
    The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
compiler_path : Optional[str]
    The custom path to the compiler executable.
compiler_flags : Optional[Sequence[str]]
    The custom flags to pass to the compiler.
custom_header : Optional[str]
    The custom header to include in the generated code."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
        },
        MethodInfo {
            name: "compile",
            parameters: &[
                ParameterInfo {
                    name: "function_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "filename",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "library_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "number_type",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || TypeInfo::unqualified("typing.Literal['complex_4x']"),
                },
                ParameterInfo {
                    name: "inline_asm",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(DEFAULT),
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "optimization_level",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(THREE),
                    type_info: || Option::<u8>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_path",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_flags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "custom_header",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },

            ],
            r#type: MethodType::Class,
            r#return: || PythonCompiledSimdComplexExpressionEvaluator::type_output(),
            doc:
r#"Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

Parameters
----------
function_name : str
    The name of the function to generate and compile.
filename : str
    The name of the file to generate.
library_name : str
    The name of the shared library to generate.
number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
    The type of numbers to use. Can be 'real' for double or 'complex' for complex double.
    For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
    For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
inline_asm : str
    The inline ASM option can be set to 'default', 'x64', 'aarch64' or 'none'.
optimization_level : int
    The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
compiler_path : Optional[str]
    The custom path to the compiler executable.
compiler_flags : Optional[Sequence[str]]
    The custom flags to pass to the compiler.
custom_header : Optional[str]
    The custom header to include in the generated code."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
        },
        MethodInfo {
            name: "compile",
            parameters: &[
                ParameterInfo {
                    name: "function_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "filename",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "library_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "number_type",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || TypeInfo::unqualified("typing.Literal['cuda_real']"),
                },
                ParameterInfo {
                    name: "inline_asm",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(DEFAULT),
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "optimization_level",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(THREE),
                    type_info: || Option::<u8>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_path",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_flags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "custom_header",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "cuda_number_of_evaluations",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(ONE),
                    type_info: || Option::<usize>::type_input(),
                },
                ParameterInfo {
                    name: "cuda_block_size",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(CUDA_BLOCK_DEFAULT),
                    type_info: || Option::<usize>::type_input(),
                },
            ],
            r#type: MethodType::Class,
            r#return: || PythonCompiledCudaRealExpressionEvaluator::type_output(),
            doc:
r#"Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

You may have to specify `-code=sm_XY` for your architecture `XY` in the compiler flags to prevent a potentially long
JIT compilation upon the first evaluation.

Parameters
----------
function_name : str
    The name of the function to generate and compile.
filename : str
    The name of the file to generate.
library_name : str
    The name of the shared library to generate.
number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
    The type of numbers to use. Can be 'real' for double or 'complex' for complex double.
    For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
    For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
inline_asm : str
    The inline ASM option can be set to 'default', 'x64', 'aarch64' or 'none'.
optimization_level : int
    The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
compiler_path : Optional[str]
    The custom path to the compiler executable.
compiler_flags : Optional[Sequence[str]]
    The custom flags to pass to the compiler.
custom_header : Optional[str]
    The custom header to include in the generated code.
cuda_number_of_evaluations: Optional[int]
    The number of parallel evaluations to perform on the CUDA device. The input to evaluate must 
    have the length `cuda_number_of_evaluations * arg_len`.
cuda_block_size: Optional[int]
    The block size to use for CUDA kernel launches."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
        },
        MethodInfo {
            name: "compile",
            parameters: &[
                ParameterInfo {
                    name: "function_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "filename",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "library_name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "number_type",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || TypeInfo::unqualified("typing.Literal['cuda_complex']"),
                },
                ParameterInfo {
                    name: "inline_asm",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(DEFAULT),
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "optimization_level",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(THREE),
                    type_info: || Option::<u8>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_path",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "compiler_flags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "custom_header",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<String>::type_input(),
                },
                ParameterInfo {
                    name: "cuda_number_of_evaluations",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(ONE),
                    type_info: || Option::<usize>::type_input(),
                },
                ParameterInfo {
                    name: "cuda_block_size",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(CUDA_BLOCK_DEFAULT),
                    type_info: || Option::<usize>::type_input(),
                },
            ],
            r#type: MethodType::Class,
            r#return: || PythonCompiledCudaComplexExpressionEvaluator::type_output(),
            doc:
r#"Compile the evaluator to a shared library using C++ and optionally inline assembly and load it.

You may have to specify `-code=sm_XY` for your architecture `XY` in the compiler flags to prevent a potentially long
JIT compilation upon the first evaluation.

Parameters
----------
function_name : str
    The name of the function to generate and compile.
filename : str
    The name of the file to generate.
library_name : str
    The name of the shared library to generate.
number_type : Literal['real'] | Literal['complex'] | Literal['real_4x'] | Literal['complex_4x'] | Literal['cuda_real'] | Literal['cuda_complex']
    The type of numbers to use. Can be 'real' for double or 'complex' for complex double.
    For 4x SIMD runs, use 'real_4x' or 'complex_4x'.
    For GPU runs with CUDA, use 'cuda_real' or 'cuda_complex'.
inline_asm : str
    The inline ASM option can be set to 'default', 'x64', 'aarch64' or 'none'.
optimization_level : int
    The optimization level to use for the compiler. This can be set to 0, 1, 2 or 3.
compiler_path : Optional[str]
    The custom path to the compiler executable.
compiler_flags : Optional[Sequence[str]]
    The custom flags to pass to the compiler.
custom_header : Optional[str]
    The custom header to include in the generated code.
cuda_number_of_evaluations: Optional[int]
    The number of parallel evaluations to perform on the CUDA device. The input to evaluate must 
    have the length `cuda_number_of_evaluations * arg_len`.
cuda_block_size: Optional[int]
    The block size to use for CUDA kernel launches."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
        }

          ],
    }
}

/// A compiled and optimized evaluator for expressions.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "CompiledRealEvaluator", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonCompiledRealExpressionEvaluator {
    pub eval: CompiledRealEvaluator,
    pub input_len: usize,
    pub output_len: usize,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonCompiledRealExpressionEvaluator {
    /// Load a compiled library, previously generated with `compile`.
    #[classmethod]
    fn load(
        _cls: &Bound<'_, PyType>,
        filename: &str,
        function_name: &str,
        input_len: usize,
        output_len: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            eval: CompiledRealEvaluator::load(filename, function_name)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Load error: {}", e)))?,
            input_len,
            output_len,
        })
    }

    /// Evaluate the expression for multiple inputs and return the results.
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.float64]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLike2<'py, f64, TypeMustMatch>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let arr = inputs.as_array();

        if inputs.shape()[1] != self.input_len {
            return Err(exceptions::PyValueError::new_err(format!(
                "Input length mismatch: expected {}, got {}",
                self.input_len,
                inputs.shape()[1]
            )));
        }
        let n_inputs = inputs.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.output_len][..]);
        for (i, mut o) in arr.axis_iter(Axis(0)).zip(out.axis_iter_mut(Axis(0))) {
            self.eval.evaluate(
                i.as_slice().ok_or_else(|| {
                    exceptions::PyValueError::new_err("Failed to convert input to slice")
                })?,
                o.as_slice_mut().unwrap(),
            );
        }

        Ok(out.into_pyarray(py))
    }
}

/// A compiled and optimized evaluator for expressions.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "CompiledSimdRealEvaluator", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonCompiledSimdRealExpressionEvaluator {
    pub eval: CompiledSimdRealEvaluator,
    pub input_len: usize,
    pub output_len: usize,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonCompiledSimdRealExpressionEvaluator {
    /// Load a compiled library, previously generated with `compile`.
    #[classmethod]
    fn load(
        _cls: &Bound<'_, PyType>,
        filename: &str,
        function_name: &str,
        input_len: usize,
        output_len: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            eval: CompiledSimdRealEvaluator::load(filename, function_name)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Load error: {}", e)))?,
            input_len,
            output_len,
        })
    }

    /// Evaluate the expression for multiple inputs and return the results.
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.float64]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLike2<'py, f64, TypeMustMatch>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let arr = inputs.as_array();

        if inputs.shape()[1] != self.input_len {
            return Err(exceptions::PyValueError::new_err(format!(
                "Input length mismatch: expected {}, got {}",
                self.input_len,
                inputs.shape()[1]
            )));
        }
        let n_inputs = inputs.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.output_len][..]);

        self.eval
            .evaluate_batch(
                n_inputs,
                arr.as_slice().ok_or_else(|| {
                    exceptions::PyValueError::new_err("Failed to convert input to slice")
                })?,
                out.as_slice_mut().unwrap(),
            )
            .map_err(|e| exceptions::PyValueError::new_err(format!("Batch error: {}", e)))?;

        Ok(out.into_pyarray(py))
    }
}

/// A compiled and optimized evaluator for expressions.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "CompiledCudaRealEvaluator", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonCompiledCudaRealExpressionEvaluator {
    pub eval: CompiledCudaRealEvaluator,
    pub input_len: usize,
    pub output_len: usize,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonCompiledCudaRealExpressionEvaluator {
    /// Load a compiled library, previously generated with `compile`.
    #[pyo3(signature =
        (filename, function_name, input_len, output_len, number_of_evaluations, block_size = 512))]
    #[classmethod]
    fn load(
        _cls: &Bound<'_, PyType>,
        filename: &str,
        function_name: &str,
        input_len: usize,
        output_len: usize,
        number_of_evaluations: usize,
        block_size: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            eval: CompiledCudaRealEvaluator::load_with_settings(
                filename,
                function_name,
                CudaLoadSettings {
                    number_of_evaluations,
                    block_size,
                },
            )
            .map_err(|e| exceptions::PyValueError::new_err(format!("Load error: {}", e)))?,
            input_len,
            output_len,
        })
    }

    /// Evaluate the expression for multiple inputs and return the results.
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.float64]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLike2<'py, f64, TypeMustMatch>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let arr = inputs.as_array();

        if inputs.shape()[1] != self.input_len {
            return Err(exceptions::PyValueError::new_err(format!(
                "Input length mismatch: expected {}, got {}",
                self.input_len,
                inputs.shape()[1]
            )));
        }
        let n_inputs = inputs.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.output_len][..]);

        self.eval
            .evaluate(
                arr.as_slice().ok_or_else(|| {
                    exceptions::PyValueError::new_err("Failed to convert input to slice")
                })?,
                out.as_slice_mut().unwrap(),
            )
            .map_err(|e| exceptions::PyValueError::new_err(format!("Evaluation error: {}", e)))?;

        Ok(out.into_pyarray(py))
    }
}

/// A compiled and optimized evaluator for expressions.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "CompiledCudaComplexEvaluator", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonCompiledCudaComplexExpressionEvaluator {
    pub eval: CompiledCudaComplexEvaluator,
    pub input_len: usize,
    pub output_len: usize,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonCompiledCudaComplexExpressionEvaluator {
    /// Load a compiled library, previously generated with `compile`.
    #[pyo3(signature =
        (filename, function_name, input_len, output_len, number_of_evaluations, block_size = 512))]
    #[classmethod]
    fn load(
        _cls: &Bound<'_, PyType>,
        filename: &str,
        function_name: &str,
        input_len: usize,
        output_len: usize,
        number_of_evaluations: usize,
        block_size: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            eval: CompiledCudaComplexEvaluator::load_with_settings(
                filename,
                function_name,
                CudaLoadSettings {
                    number_of_evaluations,
                    block_size,
                },
            )
            .map_err(|e| exceptions::PyValueError::new_err(format!("Load error: {}", e)))?,
            input_len,
            output_len,
        })
    }

    /// Evaluate the expression for multiple inputs and return the results.
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.complex128]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLike2<'py, Complex64, TypeMustMatch>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
        let arr = inputs.as_array();

        if inputs.shape()[1] != self.input_len {
            return Err(exceptions::PyValueError::new_err(format!(
                "Input length mismatch: expected {}, got {}",
                self.input_len,
                inputs.shape()[1]
            )));
        }
        let n_inputs = inputs.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.output_len][..]);

        let sc = unsafe {
            std::mem::transmute::<&[Complex64], &[Complex<f64>]>(arr.as_slice().ok_or_else(
                || exceptions::PyValueError::new_err("Failed to convert input to slice"),
            )?)
        };
        let os = unsafe {
            std::mem::transmute::<&mut [Complex64], &mut [Complex<f64>]>(
                out.as_slice_mut().unwrap(),
            )
        };

        self.eval
            .evaluate(sc, os)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Evaluation error: {}", e)))?;

        Ok(out.into_pyarray(py))
    }
}

/// A compiled and optimized evaluator for expressions.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "CompiledComplexEvaluator", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonCompiledComplexExpressionEvaluator {
    pub eval: CompiledComplexEvaluator,
    pub input_len: usize,
    pub output_len: usize,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonCompiledComplexExpressionEvaluator {
    /// Load a compiled library, previously generated with `compile`.
    #[classmethod]
    fn load(
        _cls: &Bound<'_, PyType>,
        filename: &str,
        function_name: &str,
        input_len: usize,
        output_len: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            eval: CompiledComplexEvaluator::load(filename, function_name)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Load error: {}", e)))?,
            input_len,
            output_len,
        })
    }

    /// Evaluate the expression for multiple inputs and return the results.
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.complex128]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLike2<'py, Complex64, TypeMustMatch>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
        let arr = inputs.as_array();

        if inputs.shape()[1] != self.input_len {
            return Err(exceptions::PyValueError::new_err(format!(
                "Input length mismatch: expected {}, got {}",
                self.input_len,
                inputs.shape()[1]
            )));
        }
        let n_inputs = inputs.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.output_len][..]);
        for (i, mut o) in arr.axis_iter(Axis(0)).zip(out.axis_iter_mut(Axis(0))) {
            let sc = unsafe {
                std::mem::transmute::<&[Complex64], &[Complex<f64>]>(i.as_slice().unwrap())
            };
            let os = unsafe {
                std::mem::transmute::<&mut [Complex64], &mut [Complex<f64>]>(
                    o.as_slice_mut().unwrap(),
                )
            };

            self.eval.evaluate(sc, os);
        }

        Ok(out.into_pyarray(py))
    }
}

/// A compiled and optimized evaluator for expressions.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "CompiledSimdComplexEvaluator", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonCompiledSimdComplexExpressionEvaluator {
    pub eval: CompiledSimdComplexEvaluator,
    pub input_len: usize,
    pub output_len: usize,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonCompiledSimdComplexExpressionEvaluator {
    /// Load a compiled library, previously generated with `compile`.
    #[classmethod]
    fn load(
        _cls: &Bound<'_, PyType>,
        filename: &str,
        function_name: &str,
        input_len: usize,
        output_len: usize,
    ) -> PyResult<Self> {
        Ok(Self {
            eval: CompiledSimdComplexEvaluator::load(filename, function_name)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Load error: {}", e)))?,
            input_len,
            output_len,
        })
    }

    /// Evaluate the expression for multiple inputs and return the results.
    #[gen_stub(override_return_type(
        type_repr = "numpy.typing.NDArray[numpy.complex128]",
        imports = ("numpy.typing", "numpy")
    ))]
    fn evaluate<'py>(
        &mut self,
        #[gen_stub(override_type(
            type_repr = "numpy.typing.ArrayLike",
            imports = ("numpy.typing",),
        ))]
        inputs: PyArrayLike2<'py, Complex64, TypeMustMatch>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyArrayDyn<Complex64>>> {
        let arr = inputs.as_array();

        if inputs.shape()[1] != self.input_len {
            return Err(exceptions::PyValueError::new_err(format!(
                "Input length mismatch: expected {}, got {}",
                self.input_len,
                inputs.shape()[1]
            )));
        }
        let n_inputs = inputs.shape()[0];
        let mut out = ArrayD::zeros(&[n_inputs, self.output_len][..]);

        let sc = unsafe {
            std::mem::transmute::<&[Complex64], &[Complex<f64>]>(arr.as_slice().ok_or_else(
                || exceptions::PyValueError::new_err("Failed to convert input to slice"),
            )?)
        };
        let os = unsafe {
            std::mem::transmute::<&mut [Complex64], &mut [Complex<f64>]>(
                out.as_slice_mut().unwrap(),
            )
        };

        self.eval
            .evaluate_batch(n_inputs, sc, os)
            .map_err(|e| exceptions::PyValueError::new_err(format!("Batch error: {}", e)))?;

        Ok(out.into_pyarray(py))
    }
}

#[derive(FromPyObject)]
pub enum ScalarOrMatrix {
    Scalar(ConvertibleToRationalPolynomial),
    Matrix(PythonMatrix),
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(ScalarOrMatrix = ConvertibleToRationalPolynomial | PythonMatrix);

/// A Symbolica matrix with rational polynomial coefficients.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "Matrix", subclass, module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonMatrix {
    pub matrix: Matrix<RationalPolynomialField<IntegerRing, u16>>,
}

impl PythonMatrix {
    fn unify(&self, rhs: &PythonMatrix) -> (PythonMatrix, PythonMatrix) {
        let mut zero = self.matrix.field().zero();

        let mut self_data = self.matrix.clone().into_vec();
        let mut new_rhs_data = rhs.matrix.clone().into_vec();

        for e in &mut self_data {
            zero.unify_variables(e);
        }
        for e in &mut new_rhs_data {
            zero.unify_variables(e);
        }

        (
            PythonMatrix {
                matrix: Matrix::from_linear(
                    self_data,
                    self.matrix.nrows() as u32,
                    self.matrix.ncols() as u32,
                    RationalPolynomialField::new(Z),
                )
                .unwrap(),
            },
            PythonMatrix {
                matrix: Matrix::from_linear(
                    new_rhs_data,
                    rhs.matrix.nrows() as u32,
                    rhs.matrix.ncols() as u32,
                    RationalPolynomialField::new(Z),
                )
                .unwrap(),
            },
        )
    }

    fn unify_scalar(
        &self,
        rhs: &PythonRationalPolynomial,
    ) -> (PythonMatrix, PythonRationalPolynomial) {
        let mut zero = self.matrix.field().zero();

        let mut self_data = self.matrix.clone().into_vec();

        for e in &mut self_data {
            zero.unify_variables(e);
        }

        let mut new_rhs = rhs.poly.clone();
        zero.unify_variables(&mut new_rhs);

        (
            PythonMatrix {
                matrix: Matrix::from_linear(
                    self_data,
                    self.matrix.nrows() as u32,
                    self.matrix.ncols() as u32,
                    RationalPolynomialField::new(Z),
                )
                .unwrap(),
            },
            PythonRationalPolynomial { poly: new_rhs },
        )
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonMatrix {
    /// Create a new zeroed matrix with `nrows` rows and `ncols` columns.
    #[new]
    pub fn new(nrows: u32, ncols: u32) -> PyResult<PythonMatrix> {
        if nrows == 0 || ncols == 0 {
            return Err(exceptions::PyValueError::new_err(
                "The matrix must have at least one row and one column",
            ));
        }

        Ok(PythonMatrix {
            matrix: Matrix::new(nrows, ncols, RationalPolynomialField::new(Z)),
        })
    }

    /// Create a new square matrix with `nrows` rows and ones on the main diagonal and zeroes elsewhere.
    #[classmethod]
    pub fn identity(_cls: &Bound<'_, PyType>, nrows: u32) -> PyResult<PythonMatrix> {
        if nrows == 0 {
            return Err(exceptions::PyValueError::new_err(
                "The matrix must have at least one row and one column",
            ));
        }

        Ok(PythonMatrix {
            matrix: Matrix::identity(nrows, RationalPolynomialField::new(Z)),
        })
    }

    /// Create a new matrix with the scalars `diag` on the main diagonal and zeroes elsewhere.
    #[classmethod]
    pub fn eye(
        _cls: &Bound<'_, PyType>,
        diag: Vec<ConvertibleToRationalPolynomial>,
    ) -> PyResult<PythonMatrix> {
        if diag.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "The diagonal must have at least one entry",
            ));
        }

        let mut diag: Vec<_> = diag
            .into_iter()
            .map(|x| Ok(x.to_rational_polynomial()?.poly.clone()))
            .collect::<PyResult<_>>()?;

        // unify the entries
        let (first, rest) = diag.split_first_mut().unwrap();
        for _ in 0..2 {
            for x in &mut *rest {
                first.unify_variables(x);
            }
        }

        let field = RationalPolynomialField::new(Z);

        Ok(PythonMatrix {
            matrix: Matrix::eye(&diag, field),
        })
    }

    /// Create a new column vector from a list of scalars.
    #[classmethod]
    pub fn vec(
        _cls: &Bound<'_, PyType>,
        entries: Vec<ConvertibleToRationalPolynomial>,
    ) -> PyResult<PythonMatrix> {
        if entries.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "The matrix must have at least one row and one column",
            ));
        }

        let mut entries: Vec<_> = entries
            .into_iter()
            .map(|x| Ok(x.to_rational_polynomial()?.poly.clone()))
            .collect::<PyResult<_>>()?;

        // unify the entries
        let (first, rest) = entries.split_first_mut().unwrap();
        for _ in 0..2 {
            for x in &mut *rest {
                first.unify_variables(x);
            }
        }

        let field = RationalPolynomialField::new(Z);

        Ok(PythonMatrix {
            matrix: Matrix::new_vec(entries, field),
        })
    }

    /// Create a new row vector from a list of scalars.
    #[classmethod]
    pub fn from_linear(
        _cls: &Bound<'_, PyType>,
        nrows: u32,
        ncols: u32,
        entries: Vec<ConvertibleToRationalPolynomial>,
    ) -> PyResult<PythonMatrix> {
        if entries.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "The matrix must have at least one row and one column",
            ));
        }

        let mut entries: Vec<_> = entries
            .into_iter()
            .map(|x| Ok(x.to_rational_polynomial()?.poly.clone()))
            .collect::<PyResult<_>>()?;

        // unify the entries
        let (first, rest) = entries.split_first_mut().unwrap();
        for _ in 0..2 {
            for x in &mut *rest {
                first.unify_variables(x);
            }
        }

        let field = RationalPolynomialField::new(Z);

        Ok(PythonMatrix {
            matrix: Matrix::from_linear(entries, nrows, ncols, field)
                .map_err(|e| exceptions::PyValueError::new_err(format!("Invalid matrix: {}", e)))?,
        })
    }

    /// Create a new matrix from a 2-dimensional vector of scalars.
    #[classmethod]
    pub fn from_nested(
        cls: &Bound<'_, PyType>,
        entries: Vec<Vec<ConvertibleToRationalPolynomial>>,
    ) -> PyResult<PythonMatrix> {
        if entries.is_empty() || entries.iter().any(|x| x.is_empty()) {
            return Err(exceptions::PyValueError::new_err(
                "The matrix must have at least one row and one column",
            ));
        }

        let nrows = entries.len() as u32;
        let ncols = entries[0].len() as u32;

        if entries.iter().any(|x| x.len() != ncols as usize) {
            return Err(exceptions::PyValueError::new_err(
                "The matrix is not rectangular",
            ));
        }

        let entries: Vec<_> = entries.into_iter().flatten().collect();

        Self::from_linear(cls, nrows, ncols, entries)
    }

    /// Return the number of rows.
    pub fn nrows(&self) -> usize {
        self.matrix.nrows()
    }

    /// Return the number of columns.
    pub fn ncols(&self) -> usize {
        self.matrix.ncols()
    }

    /// Return true iff every entry in the matrix is zero.
    pub fn is_zero(&self) -> bool {
        self.matrix.is_zero()
    }

    /// Return true iff every non- main diagonal entry in the matrix is zero.
    pub fn is_diagonal(&self) -> bool {
        self.matrix.is_diagonal()
    }

    /// Return the transpose of the matrix.
    pub fn transpose(&self) -> PythonMatrix {
        PythonMatrix {
            matrix: self.matrix.transpose(),
        }
    }

    #[pyo3(signature = (row1, row2, start=0))]
    pub fn swap_rows(&mut self, row1: u32, row2: u32, start: u32) -> PyResult<()> {
        if row1 >= self.matrix.nrows() as u32 || row2 >= self.matrix.nrows() as u32 {
            return Err(exceptions::PyIndexError::new_err("Row index out of bounds"));
        }
        if start >= self.matrix.ncols() as u32 {
            return Err(exceptions::PyIndexError::new_err(
                "Start index out of bounds",
            ));
        }

        self.matrix.swap_rows(row1, row2, start);
        Ok(())
    }

    pub fn swap_cols(&mut self, col1: u32, col2: u32) -> PyResult<()> {
        if col1 >= self.matrix.ncols() as u32 || col2 >= self.matrix.ncols() as u32 {
            return Err(exceptions::PyIndexError::new_err(
                "Column index out of bounds",
            ));
        }

        self.matrix.swap_cols(col1, col2);
        Ok(())
    }

    /// Return the inverse of the matrix, if it exists.
    pub fn inv(&self) -> PyResult<PythonMatrix> {
        Ok(PythonMatrix {
            matrix: self
                .matrix
                .inv()
                .map_err(|e| exceptions::PyValueError::new_err(format!("{}", e)))?,
        })
    }

    /// Return the determinant of the matrix.
    pub fn det(&self) -> PyResult<PythonRationalPolynomial> {
        Ok(PythonRationalPolynomial {
            poly: self
                .matrix
                .det()
                .map_err(|e| exceptions::PyValueError::new_err(format!("{}", e)))?,
        })
    }

    /// Solve `A * x = b` for `x`, where `A` is the current matrix.
    pub fn solve(&self, b: PythonMatrix) -> PyResult<PythonMatrix> {
        let (new_self, new_rhs) = self.unify(&b);
        Ok(PythonMatrix {
            matrix: new_self
                .matrix
                .solve(&new_rhs.matrix)
                .map_err(|e| exceptions::PyValueError::new_err(format!("{}", e)))?,
        })
    }

    /// Solve `A * x = b` for `x`, where `A` is the current matrix and return any solution if the
    /// system is underdetermined.
    pub fn solve_any(&self, b: PythonMatrix) -> PyResult<PythonMatrix> {
        let (new_self, new_rhs) = self.unify(&b);
        Ok(PythonMatrix {
            matrix: new_self
                .matrix
                .solve_any(&new_rhs.matrix)
                .map_err(|e| exceptions::PyValueError::new_err(format!("{}", e)))?,
        })
    }

    /// Augment the matrix with another matrix, e.g. create `[A B]` from matrix `A` and `B`.
    ///
    /// Returns an error when the matrices do not have the same number of rows.
    pub fn row_reduce(&mut self, max_col: u32) -> usize {
        self.matrix.row_reduce(max_col)
    }

    /// Solve `A * x = b` for `x`, where `A` is the current matrix.
    pub fn augment(&self, b: PythonMatrix) -> PyResult<PythonMatrix> {
        let (a, b) = self.unify(&b);

        Ok(PythonMatrix {
            matrix: a
                .matrix
                .augment(&b.matrix)
                .map_err(|e| exceptions::PyValueError::new_err(format!("{}", e)))?,
        })
    }

    /// Solve `A * x = b` for `x`, where `A` is the current matrix.
    pub fn split_col(&self, index: u32) -> PyResult<(PythonMatrix, PythonMatrix)> {
        let (a, b) = self
            .matrix
            .split_col(index)
            .map_err(|e| exceptions::PyValueError::new_err(format!("{}", e)))?;

        Ok((PythonMatrix { matrix: a }, PythonMatrix { matrix: b }))
    }

    /// Get the content of the matrix, i.e. the gcd of all entries.
    pub fn content(&self) -> PythonRationalPolynomial {
        PythonRationalPolynomial {
            poly: self.matrix.content(),
        }
    }

    /// Construct the same matrix, but with the content removed.
    pub fn primitive_part(&self) -> PythonMatrix {
        PythonMatrix {
            matrix: self.matrix.primitive_part(),
        }
    }

    /// Apply a function `f` to every entry of the matrix.
    pub fn map(
        &self,
        #[gen_stub(override_type(
            type_repr = "typing.Callable[[RationalPolynomial], RationalPolynomial]"
        ))]
        f: Py<PyAny>,
    ) -> PyResult<PythonMatrix> {
        let data = self
            .matrix
            .into_iter()
            .map(|x| {
                let expr = PythonRationalPolynomial { poly: x.clone() };

                Python::attach(|py| {
                    Ok(f.call1(py, (expr,))?
                        .extract::<ConvertibleToRationalPolynomial>(py)?
                        .to_rational_polynomial()?
                        .poly
                        .clone())
                })
            })
            .collect::<PyResult<_>>()?;

        Ok(PythonMatrix {
            matrix: Matrix::from_linear(
                data,
                self.matrix.nrows() as u32,
                self.matrix.ncols() as u32,
                self.matrix.field().clone(),
            )
            .unwrap(),
        })
    }

    fn __getitem__(&self, mut idx: (isize, isize)) -> PyResult<PythonRationalPolynomial> {
        if idx.0 < 0 {
            idx.0 += self.matrix.nrows() as isize;
        }
        if idx.1 < 0 {
            idx.1 += self.matrix.ncols() as isize;
        }

        if idx.0 as usize >= self.matrix.nrows() || idx.1 as usize >= self.matrix.ncols() {
            return Err(exceptions::PyIndexError::new_err("Index out of bounds"));
        }

        Ok(PythonRationalPolynomial {
            poly: self.matrix[(idx.0 as u32, idx.1 as u32)].clone(),
        })
    }

    /// Convert the matrix into a human-readable string, with tunable settings.
    #[pyo3(signature =
        (mode = PythonPrintMode::Symbolica,
            pretty_matrix = true,
            number_thousands_separator = None,
            multiplication_operator = '*',
            double_star_for_exponentiation = false,
            square_brackets_for_function = false,
            num_exp_as_superscript = true,
            precision = None,
            show_namespaces = false,
            include_attributes = false,
            max_terms = None,
            custom_print_mode = None)
        )]
    pub fn format(
        &self,
        mode: PythonPrintMode,
        pretty_matrix: bool,
        number_thousands_separator: Option<char>,
        multiplication_operator: char,
        double_star_for_exponentiation: bool,
        square_brackets_for_function: bool,
        num_exp_as_superscript: bool,
        precision: Option<usize>,
        show_namespaces: bool,
        include_attributes: bool,
        max_terms: Option<usize>,
        custom_print_mode: Option<usize>,
    ) -> String {
        self.matrix.format_string(
            &PrintOptions {
                terms_on_new_line: false,
                color_top_level_sum: false,
                color_builtin_symbols: false,
                print_ring: false,
                symmetric_representation_for_finite_field: false,
                explicit_rational_polynomial: false,
                number_thousands_separator,
                multiplication_operator,
                double_star_for_exponentiation,
                square_brackets_for_function,
                num_exp_as_superscript,
                mode: mode.into(),
                precision,
                pretty_matrix,
                hide_all_namespaces: !show_namespaces,
                color_namespace: true,
                hide_namespace: Some("python"),
                include_attributes,
                max_terms,
                custom_print_mode: custom_print_mode.map(|x| ("default", x)),
            },
            PrintState::default(),
        )
    }

    /// Convert the matrix into a LaTeX string.
    pub fn to_latex(&self) -> PyResult<String> {
        Ok(format!(
            "$${}$$",
            self.matrix
                .format_string(&LATEX_PRINT_OPTIONS, PrintState::new())
        ))
    }

    /// Compare two matrices.
    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.matrix == other.matrix),
            CompareOp::Ne => Ok(self.matrix != other.matrix),
            _ => Err(exceptions::PyTypeError::new_err(
                "Inequalities between matrices are not supported".to_string(),
            )),
        }
    }

    /// Copy the matrix.
    pub fn __copy__(&self) -> Self {
        Self {
            matrix: self.matrix.clone(),
        }
    }

    /// Convert the matrix into a portable string.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(self
            .matrix
            .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new()))
    }

    /// Convert the matrix into a human-readable string.
    pub fn __str__(&self) -> PyResult<String> {
        Ok(self
            .matrix
            .format_string(&DEFAULT_PRINT_OPTIONS, PrintState::new()))
    }

    /// Add this matrix to `rhs`, returning the result.
    pub fn __add__(&self, rhs: PythonMatrix) -> PythonMatrix {
        let (new_self, new_rhs) = self.unify(&rhs);
        PythonMatrix {
            matrix: &new_self.matrix + &new_rhs.matrix,
        }
    }

    ///  Subtract `rhs` from this matrix, returning the result.
    pub fn __sub__(&self, rhs: PythonMatrix) -> PythonMatrix {
        self.__add__(rhs.__neg__())
    }

    /// Matrix multiply `self` and `rhs`, returning the result.
    pub fn __mul__(&self, rhs: ScalarOrMatrix) -> PyResult<PythonMatrix> {
        match rhs {
            ScalarOrMatrix::Scalar(s) => {
                let (new_self, new_rhs) = self.unify_scalar(&s.to_rational_polynomial()?);

                Ok(Self {
                    matrix: new_self.matrix.mul_scalar(&new_rhs.poly),
                })
            }
            ScalarOrMatrix::Matrix(m) => {
                let (new_self, new_rhs) = self.unify(&m);
                Ok(PythonMatrix {
                    matrix: &new_self.matrix * &new_rhs.matrix,
                })
            }
        }
    }

    /// Matrix multiply `rhs` and `self` returning the result.
    pub fn __rmul__(&self, rhs: ConvertibleToRationalPolynomial) -> PyResult<PythonMatrix> {
        self.__mul__(ScalarOrMatrix::Scalar(rhs))
    }

    /// Matrix multiply this matrix and `self`, returning the result.
    pub fn __matmul__(&self, rhs: ScalarOrMatrix) -> PyResult<PythonMatrix> {
        self.__mul__(rhs)
    }

    /// Matrix multiply `rhs` and `self`, returning the result.
    pub fn __rmatmul__(&self, rhs: ConvertibleToRationalPolynomial) -> PyResult<PythonMatrix> {
        self.__mul__(ScalarOrMatrix::Scalar(rhs))
    }

    /// Divide the matrix by the scalar, returning the result.
    pub fn __truediv__(&self, rhs: ConvertibleToRationalPolynomial) -> PyResult<PythonMatrix> {
        Ok(PythonMatrix {
            matrix: self.matrix.div_scalar(&rhs.to_rational_polynomial()?.poly),
        })
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __xor__(&self, _rhs: Py<PyAny>) -> PyResult<PythonMatrix> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor a matrix. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __rxor__(&self, _rhs: Py<PyAny>) -> PyResult<PythonMatrix> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor a matrix. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Negate the matrix, returning the result.
    pub fn __neg__(&self) -> PythonMatrix {
        PythonMatrix {
            matrix: -self.matrix.clone(),
        }
    }
}

/// A sample from the Symbolica integrator. It could consist of discrete layers,
/// accessible with `d` (empty when there are no discrete layers), and the final continuous layer `c` if it is present.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "Sample", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonSample {
    #[pyo3(get)]
    /// The weights the integrator assigned to this sample point, given in descending order:
    /// first the discrete layer weights and then the continuous layer weight.
    weights: Vec<f64>,
    #[pyo3(get)]
    /// A sample point per (nested) discrete layer. Empty if not present.
    d: Vec<usize>,
    #[pyo3(get)]
    /// A sample in the continuous layer. Empty if not present.
    c: Vec<f64>,
    uniform: bool,
}

impl PythonSample {
    fn into_sample(self) -> Sample<f64> {
        if self.uniform {
            return Sample::Uniform(self.weights[0], self.d, self.c);
        }

        assert_eq!(
            self.weights.len(),
            self.d.len() + if self.c.is_empty() { 0 } else { 1 }
        );
        let mut weight_index = self.weights.len() - 1;

        let mut sample = if !self.c.is_empty() {
            Some(Sample::Continuous(self.weights[weight_index], self.c))
        } else {
            None
        };

        for dd in self.d.iter().rev() {
            weight_index -= 1;
            sample = Some(Sample::Discrete(
                self.weights[weight_index],
                *dd,
                sample.map(Box::new),
            ));
        }

        sample.unwrap()
    }

    fn from_sample(mut sample: &Sample<f64>) -> PythonSample {
        let mut weights = vec![];
        let mut d = vec![];
        let mut c = vec![];
        let mut uniform = false;

        loop {
            match sample {
                Sample::Continuous(w, cs) => {
                    weights.push(*w);
                    c.extend_from_slice(cs);
                    break;
                }
                Sample::Discrete(w, i, s) => {
                    weights.push(*w);
                    d.push(*i);
                    if let Some(ss) = s {
                        sample = ss;
                    } else {
                        break;
                    }
                }
                Sample::Uniform(w, i, cs) => {
                    weights.push(*w);
                    d.clone_from(i);
                    c.clone_from(cs);
                    uniform = true;
                    break;
                }
            }
        }

        PythonSample {
            weights,
            d,
            c,
            uniform,
        }
    }
}

/// A reproducible, fast, non-cryptographic random number generator suitable for parallel Monte Carlo simulations.
/// A `seed` has to be set, which can be any `u64` number (small numbers work just as well as large numbers).
///
/// Each thread or instance generating samples should use the same `seed` but a different `stream_id`,
/// which is an instance counter starting at 0.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "RandomNumberGenerator", module = "symbolica.core")]
pub struct PythonRandomNumberGenerator {
    state: MonteCarloRng,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonRandomNumberGenerator {
    /// Create a new random number generator with a given `seed` and `stream_id`. For parallel runs,
    /// each thread or instance generating samples should use the same `seed` but a different `stream_id`.
    #[new]
    fn new(seed: u64, stream_id: usize) -> Self {
        Self {
            state: MonteCarloRng::new(seed, stream_id),
        }
    }
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "NumericalIntegrator", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonNumericalIntegrator {
    grid: Grid<f64>,
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(&mut PythonRandomNumberGenerator = PythonRandomNumberGenerator);

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonNumericalIntegrator {
    /// Create a new continuous grid for the numerical integrator.
    #[classmethod]
    #[pyo3(signature =
        (n_dims, n_bins = 128,
        min_samples_for_update = 100,
        bin_number_evolution = None,
        train_on_avg = false)
    )]
    pub fn continuous(
        _cls: &Bound<'_, PyType>,
        n_dims: usize,
        n_bins: usize,
        min_samples_for_update: usize,
        bin_number_evolution: Option<Vec<usize>>,
        train_on_avg: bool,
    ) -> PythonNumericalIntegrator {
        PythonNumericalIntegrator {
            grid: Grid::Continuous(ContinuousGrid::new(
                n_dims,
                n_bins,
                min_samples_for_update,
                bin_number_evolution,
                train_on_avg,
            )),
        }
    }

    /// Create a new discrete grid for the numerical integrator.
    /// Each bin can have a sub-grid.
    ///
    /// Examples
    /// --------
    /// >>> def integrand(samples: typing.Sequence[Sample]) -> list[float]:
    /// >>>     res = []
    /// >>>     for sample in samples:
    /// >>>         if sample.d[0] == 0:
    /// >>>             res.append(sample.c[0]**2)
    /// >>>         else:
    /// >>>             res.append(sample.c[0]**3)
    /// >>>     return res
    /// >>>
    /// >>> integrator = NumericalIntegrator.discrete(
    /// >>>     [NumericalIntegrator.continuous(1), NumericalIntegrator.continuous(1)])
    /// >>> integrator.integrate(integrand, min_error=1e-3)
    #[classmethod]
    #[pyo3(signature =
        (bins,
        max_prob_ratio = 100.,
        train_on_avg = false)
    )]
    pub fn discrete(
        _cls: &Bound<'_, PyType>,
        bins: Vec<Option<PythonNumericalIntegrator>>,
        max_prob_ratio: f64,
        train_on_avg: bool,
    ) -> PythonNumericalIntegrator {
        let bins = bins.into_iter().map(|b| b.map(|bb| bb.grid)).collect();

        PythonNumericalIntegrator {
            grid: Grid::Discrete(DiscreteGrid::new(bins, max_prob_ratio, train_on_avg)),
        }
    }

    /// Create a new uniform layered grid for the numerical integrator.
    /// `len(bins)` specifies the number of discrete layers, and each entry in `bins` specifies the number of bins in that layer.
    /// Each discrete bin has equal probability.
    ///
    /// Examples
    /// --------
    /// >>> def integrand(samples: typing.Sequence[Sample]) -> list[float]:
    /// >>>     res = []
    /// >>>     for sample in samples:
    /// >>>         if sample.d[0] == 0:
    /// >>>             res.append(sample.c[0]**2)
    /// >>>         else:
    /// >>>             res.append(sample.c[0]**3)
    /// >>>     return res
    /// >>>
    /// >>>
    /// >>> integrator = NumericalIntegrator.uniform(
    /// >>>     [2], NumericalIntegrator.continuous(1))
    /// >>> integrator.integrate(integrand, min_error=1e-3)
    #[classmethod]
    pub fn uniform(
        _cls: &Bound<'_, PyType>,
        bins: Vec<usize>,
        continuous_grid: PythonNumericalIntegrator,
    ) -> PyResult<PythonNumericalIntegrator> {
        if let Grid::Continuous(g) = continuous_grid.grid {
            Ok(PythonNumericalIntegrator {
                grid: Grid::Uniform(bins, g),
            })
        } else {
            return PyResult::Err(pyo3::exceptions::PyAssertionError::new_err(
                "The specified grid is not a continuous grid",
            ));
        }
    }

    /// Create a new random number generator, suitable for use with the integrator.
    /// Each thread of instance of the integrator should have its own random number generator,
    /// that is initialized with the same seed but with a different stream id.
    #[classmethod]
    pub fn rng(
        _cls: &Bound<'_, PyType>,
        seed: u64,
        stream_id: usize,
    ) -> PythonRandomNumberGenerator {
        PythonRandomNumberGenerator::new(seed, stream_id)
    }

    /// Copy the grid without any unprocessed samples.
    pub fn __copy__(&self) -> Self {
        Self {
            grid: self.grid.clone_without_samples(),
        }
    }

    /// Sample `num_samples` points from the grid using the random number generator
    /// `rng`. See `rng()` for how to create a random number generator.
    pub fn sample(
        &mut self,
        num_samples: usize,
        rng: &mut PythonRandomNumberGenerator,
    ) -> Vec<PythonSample> {
        let mut sample = Sample::new();

        let mut samples = Vec::with_capacity(num_samples);
        for _ in 0..num_samples {
            self.grid.sample(&mut rng.state, &mut sample);
            samples.push(PythonSample::from_sample(&sample));
        }

        samples
    }

    /// Add the samples and their corresponding function evaluations to the grid.
    /// Call `update` after to update the grid and to obtain the new expected value for the integral.
    fn add_training_samples(
        &mut self,
        samples: Vec<PythonSample>,
        evals: Vec<f64>,
    ) -> PyResult<()> {
        if evals.len() != samples.len() {
            return PyResult::Err(pyo3::exceptions::PyAssertionError::new_err(
                "Number of returned values does not equal number of samples",
            ));
        }

        for (s, f) in samples.into_iter().zip(evals) {
            self.grid
                .add_training_sample(&s.into_sample(), f)
                .map_err(pyo3::exceptions::PyAssertionError::new_err)?;
        }

        Ok(())
    }

    /// Import an exported grid from another thread or machine.
    /// Use `export_grid` to export the grid.
    #[classmethod]
    fn import_grid(_cls: &Bound<'_, PyType>, grid: Bound<'_, PyBytes>) -> PyResult<Self> {
        let grid = bincode::decode_from_slice(grid.extract()?, bincode::config::standard())
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?
            .0;

        Ok(PythonNumericalIntegrator { grid })
    }

    /// Export the grid, so that it can be sent to another thread or machine.
    /// If you are exporting your main grid, make sure to set `export_samples` to `False` to avoid copying unprocessed samples.
    ///
    /// Use `import_grid` to load the grid.
    #[pyo3(signature = (export_samples = true))]
    fn export_grid<'p>(
        &self,
        export_samples: bool,
        py: Python<'p>,
    ) -> PyResult<Bound<'p, PyBytes>> {
        if export_samples {
            bincode::encode_to_vec(&self.grid, bincode::config::standard())
        } else {
            bincode::encode_to_vec(
                &self.grid.clone_without_samples(),
                bincode::config::standard(),
            )
        }
        .map(|a| PyBytes::new(py, &a))
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    /// Get the estamate of the average, error, chi-squared, maximum negative and positive evaluations, and the number of processed samples
    /// for the current iteration, including the points submitted in the current iteration.
    fn get_live_estimate(&self) -> PyResult<(f64, f64, f64, f64, f64, usize)> {
        match &self.grid {
            Grid::Continuous(cs) | Grid::Uniform(_, cs) => {
                let mut a = cs.accumulator.shallow_copy();
                a.update_iter(false);
                Ok((
                    a.avg,
                    a.err,
                    a.chi_sq,
                    a.max_eval_negative,
                    a.max_eval_positive,
                    a.processed_samples,
                ))
            }
            Grid::Discrete(ds) => {
                let mut a = ds.accumulator.shallow_copy();
                a.update_iter(false);
                Ok((
                    a.avg,
                    a.err,
                    a.chi_sq,
                    a.max_eval_negative,
                    a.max_eval_positive,
                    a.processed_samples,
                ))
            }
        }
    }

    /// Add the accumulated training samples from the grid `other` to the current grid.
    /// The grid structure of `self` and `other` must be equivalent.
    fn merge(&mut self, other: &PythonNumericalIntegrator) -> PyResult<()> {
        self.grid
            .merge(&other.grid)
            .map_err(pyo3::exceptions::PyAssertionError::new_err)
    }

    /// Update the grid using the `discrete_learning_rate` and `continuous_learning_rate`.
    /// Examples
    /// --------
    /// >>> from symbolica import NumericalIntegrator, Sample
    /// >>>
    /// >>> def integrand(samples: list[Sample]):
    /// >>>     res = []
    /// >>>     for sample in samples:
    /// >>>         res.append(sample.c[0]**2+sample.c[1]**2)
    /// >>>     return res
    /// >>>
    /// >>> integrator = NumericalIntegrator.continuous(2)
    /// >>> for i in range(10):
    /// >>>     samples = integrator.sample(10000 + i * 1000)
    /// >>>     res = integrand(samples)
    /// >>>     integrator.add_training_samples(samples, res)
    /// >>>     avg, err, chi_sq = integrator.update(1.5, 1.5)
    /// >>>     print('Iteration {}: {:.6} +- {:.6}, chi={:.6}'.format(i+1, avg, err, chi_sq))
    fn update(
        &mut self,
        discrete_learning_rate: f64,
        continuous_learning_rate: f64,
    ) -> PyResult<(f64, f64, f64)> {
        self.grid
            .update(discrete_learning_rate, continuous_learning_rate);

        let stats = self.grid.get_statistics();
        Ok((stats.avg, stats.err, stats.chi_sq / stats.cur_iter as f64))
    }

    /// Integrate the function `integrand` that maps a list of `Sample`s to a list of `float`s.
    /// The return value is the average, the statistical error, and chi-squared of the integral.
    ///
    /// With `show_stats=True`, intermediate statistics will be printed. `max_n_iter` determines the number
    /// of iterations and `n_samples_per_iter` determine the number of samples per iteration. This is
    /// the same amount of samples that the integrand function will be called with.
    ///
    /// For more flexibility, use `sample`, `add_training_samples` and `update`. See `update` for an example.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import NumericalIntegrator, Sample
    /// >>>
    /// >>> def integrand(samples: list[Sample]):
    /// >>>     res = []
    /// >>>     for sample in samples:
    /// >>>         res.append(sample.c[0]**2+sample.c[1]**2)
    /// >>>     return res
    /// >>>
    /// >>> avg, err = NumericalIntegrator.continuous(2).integrate(integrand, True, 10, 100000)
    /// >>> print('Result: {} +- {}'.format(avg, err))
    #[pyo3(signature =
        (integrand,
        max_n_iter = 10_000_000,
        min_error = 0.01,
        n_samples_per_iter = 10_000,
        seed = 0,
        show_stats = true)
    )]
    pub fn integrate(
        &mut self,
        py: Python,
        #[gen_stub(override_type(
            type_repr = "typing.Callable[[typing.Sequence[Sample]], list[float]]"
        ))]
        integrand: Py<PyAny>,
        max_n_iter: usize,
        min_error: f64,
        n_samples_per_iter: usize,
        seed: u64,
        show_stats: bool,
    ) -> PyResult<(f64, f64, f64)> {
        let mut rng = MonteCarloRng::new(seed, 0);

        let mut samples = vec![Sample::new(); n_samples_per_iter];
        for iteration in 1..=max_n_iter {
            for sample in &mut samples {
                self.grid.sample(&mut rng, sample);
            }

            let p_samples: Vec<_> = samples.iter().map(PythonSample::from_sample).collect();

            let res = integrand
                .call(py, (p_samples,), None)?
                .extract::<Vec<f64>>(py)?;

            if res.len() != n_samples_per_iter {
                return Err(exceptions::PyValueError::new_err(
                    "Wrong number of arguments returned for integration function.",
                ));
            }

            for (s, r) in samples.iter().zip(res) {
                self.grid.add_training_sample(s, r).unwrap();
            }

            self.grid.update(1.5, 1.5);

            let stats = self.grid.get_statistics();
            if show_stats {
                println!(
                    "Iteration {:2}: {}  {:.2} χ²",
                    iteration,
                    stats.format_uncertainty(),
                    stats.chi_sq / stats.cur_iter as f64
                );
            }

            if stats.avg != 0. && stats.err / stats.avg.abs() <= min_error {
                break;
            }
        }

        let stats = self.grid.get_statistics();
        Ok((stats.avg, stats.err, stats.chi_sq / stats.cur_iter as f64))
    }
}

/// Represents a part of an edge that connects to one vertex. It can be directed or undirected.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "HalfEdge", module = "symbolica.core")]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PythonHalfEdge {
    half_edge: HalfEdge<Atom>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonHalfEdge {
    /// Create a new half-edge. The `data` can be any expression, and the `direction` can be `True` (outgoing),
    /// `False` (incoming) or `None` (undirected).
    #[new]
    #[pyo3(signature = (data, direction = None))]
    fn new(data: ConvertibleToExpression, direction: Option<bool>) -> Self {
        Self {
            half_edge: match direction {
                None => HalfEdge::undirected(data.to_expression().expr),
                Some(false) => HalfEdge::incoming(data.to_expression().expr),
                Some(true) => HalfEdge::outgoing(data.to_expression().expr),
            },
        }
    }

    /// Return a new half-edge with the direction flipped. Undirected edges remain undirected.
    fn flip(&self) -> Self {
        Self {
            half_edge: self.half_edge.flip(),
        }
    }

    /// Get the direction of the half-edge. `True` means outgoing, `False` means incoming, and `None` means undirected.
    fn direction(&self) -> Option<bool> {
        self.half_edge.direction
    }

    /// Get the data associated with the half-edge.
    fn data(&self) -> PythonExpression {
        self.half_edge.data.clone().into()
    }
}

/// A graph that supported directional edges, parallel edges, self-edges and custom data on the nodes and edges.
///
/// Warning: modifying the graph if it is contained in a `dict` or `set` will invalidate the hash.
#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "Graph", module = "symbolica.core")]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PythonGraph {
    graph: Graph<Atom, Atom>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonGraph {
    /// Create an empty graph.
    #[new]
    fn new() -> Self {
        Self {
            graph: Graph::new(),
        }
    }

    /// Convert the graph into a portable string.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.graph))
    }

    /// Print the graph in a human-readable format.
    fn __str__(&self) -> String {
        format!("{}", self.graph)
    }

    /// Hash the graph.
    fn __hash__(&self) -> u64 {
        let mut hasher = ahash::AHasher::default();
        self.graph.hash(&mut hasher);
        hasher.finish()
    }

    /// Copy the graph.
    fn __copy__(&self) -> PythonGraph {
        Self {
            graph: self.graph.clone(),
        }
    }

    /// Get the number of nodes.
    fn __len__(&self) -> usize {
        self.graph.nodes().len()
    }

    /// Compare two graphs.
    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.graph == other.graph),
            CompareOp::Ne => Ok(self.graph != other.graph),
            _ => Err(exceptions::PyTypeError::new_err(
                "Inequalities between graphs are not allowed".to_string(),
            )),
        }
    }

    /// Generate all connected graphs with `external_edges` half-edges and the given allowed list
    /// of vertex connections. The vertex signatures are given in terms of an edge direction (or `None` if
    /// there is no direction) and edge data.
    ///
    /// Returns the canonical form of the graph and the size of its automorphism group (including edge permutations).
    /// If `KeyboardInterrupt` is triggered during the generation, the generation will stop and will yield the currently generated
    /// graphs.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> g, q, gh = HalfEdge(S("g")), HalfEdge(S("q"), True), HalfEdge(S("gh"), True)
    /// >>> graphs = Graph.generate(
    /// >>>     external_nodes=[(1, g), (2, g)],
    /// >>>     vertex_signatures=[[g, g, g], [g, g, g, g],
    /// >>>                        [q.flip(), q, g], [gh.flip(), gh, g]],
    /// >>>     max_loops=2,
    /// >>> )
    /// >>> for (g, sym) in graphs.items():
    /// >>>     print(f'Symmetry factor = 1/{sym}:')
    /// >>>     print(g.to_dot())
    ///
    /// generates all connected graphs up to 2 loops with the specified vertices.
    ///
    /// Parameters
    /// ----------
    /// external_nodes: Sequence[tuple[Expression | int, HalfEdge]]
    ///     The external edges, consisting of a tuple of the node data and a tuple of the edge direction and edge data.
    ///     If the node data is the same, flip symmetries will be recognized.
    /// vertex_signatures: Sequence[Sequence[HalfEdge]]
    ///     The allowed connections for each vertex.
    /// max_vertices: int, optional
    ///     The maximum number of vertices in the graph.
    /// max_loops: int, optional
    ///     The maximum number of loops in the graph.
    /// max_bridges: int, optional
    ///     The maximum number of bridges in the graph.
    /// allow_self_loops: bool, optional
    ///     Whether self-edges are allowed.
    /// allow_zero_flow_edges: bool, optional
    ///     Whether bridges that do not need to be crossed to connect external vertices are allowed.
    /// filter_fn: Optional[Callable[[Graph, int], bool]], optional
    ///     Set a filter function that is called during the graph generation.
    ///     The first argument is the graph `g` and the second argument the vertex count `n`
    ///     that specifies that the first `n` vertices are completed (no new edges will) be
    ///     assigned to them. The filter function should return `true` if the current
    ///     incomplete graph is allowed, else it should return `false` and the graph is discarded.
    /// progress_fn: Optional[Callable[[Graph, bool]], optional
    ///     Set a progress function that is called every time a new unique graph is created.
    ///     The argument is the newly created graph.
    ///     If the function returns `false`, the generation is aborted and the currently
    ///     generated graphs are returned.
    #[pyo3(signature = (external_edges, vertex_signatures, max_vertices = None, max_loops = None,
        max_bridges = None, allow_self_loops = None, allow_zero_flow_edges = None, filter_fn = None, progress_fn = None))]
    #[classmethod]
    fn generate(
        _cls: &Bound<'_, PyType>,
        external_edges: Vec<(ConvertibleToExpression, PythonHalfEdge)>,
        vertex_signatures: Vec<Vec<PythonHalfEdge>>,
        max_vertices: Option<usize>,
        max_loops: Option<usize>,
        max_bridges: Option<usize>,
        allow_self_loops: Option<bool>,
        allow_zero_flow_edges: Option<bool>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Callable[[Graph, int], bool]]"
        ))]
        filter_fn: Option<Py<PyAny>>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Callable[[Graph], bool]]"))]
        progress_fn: Option<Py<PyAny>>,
    ) -> PyResult<HashMap<PythonGraph, PythonExpression>> {
        if max_vertices.is_none() && max_loops.is_none() {
            return Err(exceptions::PyValueError::new_err(
                "At least one of max_vertices or max_loop must be set",
            ));
        }

        let external_edges: Vec<_> = external_edges
            .into_iter()
            .map(|(a, b)| (a.to_expression().expr, b.half_edge))
            .collect();
        let vertex_signatures: Vec<_> = vertex_signatures
            .into_iter()
            .map(|v| v.into_iter().map(|x| x.half_edge).collect())
            .collect();

        let mut settings = GenerationSettings::new();
        if let Some(max_vertices) = max_vertices {
            settings = settings.max_vertices(max_vertices);
        }

        if let Some(max_loops) = max_loops {
            settings = settings.max_loops(max_loops);
        }

        if let Some(max_bridges) = max_bridges {
            settings = settings.max_bridges(max_bridges);
        }

        if let Some(allow_self_loops) = allow_self_loops {
            settings = settings.allow_self_loops(allow_self_loops);
        }

        if let Some(allow_zero_flow_edge) = allow_zero_flow_edges {
            settings = settings.allow_zero_flow_edges(allow_zero_flow_edge);
        }

        let abort = Arc::new(std::sync::atomic::AtomicBool::new(false));

        if let Some(filter_fn) = filter_fn {
            let abort = abort.clone();
            settings = settings.filter_fn(Box::new(move |g, v| {
                Python::attach(|py| {
                    match filter_fn.call(py, (Self { graph: g.clone() }, v), None) {
                        Ok(r) => r
                            .is_truthy(py)
                            .expect("Match map does not return a boolean"),
                        Err(e) => {
                            if e.is_instance_of::<exceptions::PyKeyboardInterrupt>(py) {
                                abort.store(true, std::sync::atomic::Ordering::Relaxed);
                                false
                            } else {
                                panic!("Bad callback function: {}", e);
                            }
                        }
                    }
                })
            }));
        }

        if let Some(progress_fn) = progress_fn {
            settings = settings.progress_fn(Box::new(move |g| {
                Python::attach(|py| {
                    match progress_fn.call(py, (Self { graph: g.clone() },), None) {
                        Ok(r) => r.is_truthy(py).unwrap_or(true),
                        Err(e) => {
                            error!("Bad callback function: {}", e);
                            false
                        }
                    }
                })
            }));
        }

        settings = settings.abort_check(Box::new(move || {
            if abort.load(std::sync::atomic::Ordering::Relaxed) {
                true
            } else {
                Python::attach(|py| py.check_signals())
                    .map(|_| false)
                    .unwrap_or(true)
            }
        }));

        Ok(
            Graph::generate(&external_edges, &vertex_signatures, settings)
                .unwrap_or_else(|e| e)
                .into_iter()
                .map(|(k, v)| (Self { graph: k }, Atom::num(v).into()))
                .collect(),
        )
    }

    /// Convert the graph to a graphviz dot string.
    fn to_dot(&self) -> String {
        self.graph.to_dot()
    }

    /// Convert the graph to a mermaid string.
    fn to_mermaid(&self) -> String {
        self.graph.to_mermaid()
    }

    /// Add a node with data `data` to the graph, returning the index of the node.
    /// The default data is the number 0.
    #[pyo3(signature = (data = None))]
    fn add_node(&mut self, data: Option<ConvertibleToExpression>) -> usize {
        self.graph
            .add_node(data.map(|x| x.to_expression().expr).unwrap_or_default())
    }

    /// Add an edge between the `source` and `target` nodes, returning the index of the edge.
    /// Optionally, the edge can be set as directed. The default data is the number 0.
    #[pyo3(signature = (source, target, directed = false, data = None))]
    fn add_edge(
        &mut self,
        source: usize,
        target: usize,
        directed: bool,
        data: Option<ConvertibleToExpression>,
    ) -> PyResult<usize> {
        self.graph
            .add_edge(
                source,
                target,
                directed,
                data.map(|x| x.to_expression().expr).unwrap_or_default(),
            )
            .map_err(exceptions::PyValueError::new_err)
    }

    /// Set the data of the node at index `index`, returning the old data.
    pub fn set_node_data(
        &mut self,
        index: isize,
        data: PythonExpression,
    ) -> PyResult<PythonExpression> {
        if index.unsigned_abs() < self.graph.nodes().len() {
            let n = if index < 0 {
                self.graph.nodes().len() - index.unsigned_abs()
            } else {
                index as usize
            };
            Ok(self.graph.set_node_data(n, data.expr).into())
        } else {
            Err(PyIndexError::new_err(format!(
                "Index {} out of bounds: the graph only has {} nodes.",
                index,
                self.graph.nodes().len(),
            )))
        }
    }

    /// Set the data of the edge at index `index`, returning the old data.
    pub fn set_edge_data(
        &mut self,
        index: isize,
        data: PythonExpression,
    ) -> PyResult<PythonExpression> {
        if index.unsigned_abs() < self.graph.edges().len() {
            let e = if index < 0 {
                self.graph.edges().len() - index.unsigned_abs()
            } else {
                index as usize
            };
            Ok(self.graph.set_edge_data(e, data.expr).into())
        } else {
            Err(PyIndexError::new_err(format!(
                "Index {} out of bounds: the graph only has {} edges.",
                index,
                self.graph.edges().len(),
            )))
        }
    }

    /// Set the directed status of the edge at index `index`, returning the old value.
    pub fn set_directed(&mut self, index: isize, directed: bool) -> PyResult<bool> {
        if index.unsigned_abs() < self.graph.edges().len() {
            let e = if index < 0 {
                self.graph.edges().len() - index.unsigned_abs()
            } else {
                index as usize
            };
            Ok(self.graph.set_directed(e, directed))
        } else {
            Err(PyIndexError::new_err(format!(
                "Index {} out of bounds: the graph only has {} edges.",
                index,
                self.graph.edges().len(),
            )))
        }
    }

    /// Get the `idx`th node.
    fn __getitem__(&self, idx: isize) -> PyResult<(Vec<usize>, PythonExpression)> {
        self.node(idx)
    }

    /// Get the number of nodes.
    fn num_nodes(&self) -> usize {
        self.graph.nodes().len()
    }

    /// Get the number of edges.
    fn num_edges(&self) -> usize {
        self.graph.edges().len()
    }

    /// Get the number of loops.
    fn num_loops(&self) -> usize {
        self.graph.num_loops()
    }

    /// Get the `idx`th node, consisting of the edge indices and the data.
    fn node(&self, idx: isize) -> PyResult<(Vec<usize>, PythonExpression)> {
        if idx.unsigned_abs() < self.graph.nodes().len() {
            let n = if idx < 0 {
                self.graph
                    .node(self.graph.nodes().len() - idx.unsigned_abs())
            } else {
                self.graph.node(idx as usize)
            };
            Ok((n.edges.clone(), n.data.clone().into()))
        } else {
            Err(PyIndexError::new_err(format!(
                "Index {} out of bounds: the graph only has {} nodes.",
                idx,
                self.graph.nodes().len(),
            )))
        }
    }

    /// Get all nodes, consisting of the edge indices and the data.
    fn nodes(&self) -> Vec<(Vec<usize>, PythonExpression)> {
        self.graph
            .nodes()
            .iter()
            .map(|n| (n.edges.clone(), n.data.clone().into()))
            .collect()
    }

    /// Get the `idx`th edge, consisting of the the source vertex, target vertex, whether the edge is directed, and the data.
    fn edge(&self, idx: isize) -> PyResult<(usize, usize, bool, PythonExpression)> {
        if idx.unsigned_abs() < self.graph.edges().len() {
            let e = if idx < 0 {
                self.graph
                    .edge(self.graph.edges().len() - idx.unsigned_abs())
            } else {
                self.graph.edge(idx as usize)
            };
            Ok((
                e.vertices.0,
                e.vertices.1,
                e.directed,
                e.data.clone().into(),
            ))
        } else {
            Err(PyIndexError::new_err(format!(
                "Index {} out of bounds: the graph only has {} edges.",
                idx,
                self.graph.edges().len(),
            )))
        }
    }

    /// Get all edges, consisting of the the source vertex, target vertex, whether the edge is directed, and the data.
    fn edges(&self) -> Vec<(usize, usize, bool, PythonExpression)> {
        self.graph
            .edges()
            .iter()
            .map(|e| {
                (
                    e.vertices.0,
                    e.vertices.1,
                    e.directed,
                    e.data.clone().into(),
                )
            })
            .collect()
    }

    /// Write the graph in a canonical form.
    /// Returns the canonicalized graph, the vertex map, the automorphism group size, and the orbit.
    fn canonize(&self) -> (PythonGraph, Vec<usize>, PythonExpression, Vec<usize>) {
        let c = self.graph.canonize();
        (
            Self { graph: c.graph },
            c.vertex_map,
            Atom::num(c.automorphism_group_size).into(),
            c.orbit,
        )
    }

    /// Sort and relabel the edges of the graph, keeping the vertices fixed.
    pub fn canonize_edges(&mut self) {
        self.graph.canonize_edges();
    }

    /// Return true `iff` the graph is isomorphic to `other`.
    fn is_isomorphic(&self, other: &PythonGraph) -> bool {
        self.graph.is_isomorphic(&other.graph)
    }
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "Integer", module = "symbolica.core")]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PythonInteger {}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonInteger {
    /// Create an iterator over all 64-bit prime numbers starting from `start`.
    #[pyo3(signature = (start = 1))]
    #[classmethod]
    fn prime_iter(_cls: &Bound<'_, PyType>, start: u64) -> PyResult<PythonPrimeIterator> {
        Ok(PythonPrimeIterator {
            cur: PrimeIteratorU64::new(start),
        })
    }

    /// Check if the 64-bit number `n` is a prime number.
    #[classmethod]
    fn is_prime(_cls: &Bound<'_, PyType>, n: u64) -> bool {
        is_prime_u64(n)
    }

    /// Factor a 64-bit number `n` into primes.
    #[classmethod]
    fn factor(_cls: &Bound<'_, PyType>, n: u64) -> Vec<(u64, u64)> {
        let mut factors = Vec::new();
        crate::domains::finite_field::factor(n, &mut factors);
        factors.sort();
        let mut ff = vec![];
        ff.push((factors[0], 1));
        for i in 1..factors.len() {
            if factors[i] == factors[i - 1] {
                ff.last_mut().unwrap().1 += 1;
            } else {
                ff.push((factors[i], 1));
            }
        }

        ff
    }

    /// Compute the Euler totient function for the number `n`.
    #[classmethod]
    fn totient(_cls: &Bound<'_, PyType>, n: u64) -> u64 {
        crate::domains::finite_field::totient(n)
    }

    /// Compute the greatest common divisor of the numbers `a` and `b`.
    #[classmethod]
    fn gcd(_cls: &Bound<'_, PyType>, n1: Integer, n2: Integer) -> Integer {
        n1.gcd(&n2)
    }

    /// Compute the greatest common divisor of the numbers `a` and `b` and the Bézout coefficients.
    #[classmethod]
    fn extended_gcd(
        _cls: &Bound<'_, PyType>,
        n1: Integer,
        n2: Integer,
    ) -> (Integer, Integer, Integer) {
        n1.extended_gcd(&n2)
    }

    /// Solve the Chinese remainder theorem for the equations:
    /// `x = n1 mod m1` and `x = n2 mod m2`.
    #[classmethod]
    fn chinese_remainder(
        _cls: &Bound<'_, PyType>,
        n1: Integer,
        m1: Integer,
        n2: Integer,
        m2: Integer,
    ) -> Integer {
        Integer::chinese_remainder(n1, n2, m1, m2)
    }

    /// Compute the least common multiple of the numbers `a` and `b`.
    #[classmethod]
    fn lcm(_cls: &Bound<'_, PyType>, n1: Integer, n2: Integer) -> Integer {
        n1.lcm(&n2)
    }

    /// Use the PSLQ algorithm to find a vector of integers `a` that satisfies `a.x = 0`,
    /// where every element of `a` is less than `max_coeff`, using a specified tolerance and number
    /// of iterations. The parameter `gamma` must be more than or equal to `2/sqrt(3)`.
    ///
    /// Examples
    /// --------
    /// Solve a `32.0177=b*pi+c*e` where `b` and `c` are integers:
    ///
    /// >>> r = Integer.solve_integer_relation([-32.0177, 3.1416, 2.7183], 1e-5, 100)
    /// >>> print(r)
    ///
    /// yields `[1,5,6]`.
    #[pyo3(signature = (x, tolerance, max_iter = 1000, max_coeff = None, gamma = None))]
    #[classmethod]
    fn solve_integer_relation<'py>(
        _cls: &Bound<'_, PyType>,
        x: Vec<PythonMultiPrecisionFloat>,
        tolerance: PythonMultiPrecisionFloat,
        max_iter: usize,
        max_coeff: Option<Integer>,
        gamma: Option<PythonMultiPrecisionFloat>,
        py: Python<'py>,
    ) -> PyResult<Vec<Bound<'py, PyInt>>> {
        let x: Vec<_> = x.into_iter().map(|x| x.0).collect();

        let res = Integer::solve_integer_relation(
            &x,
            tolerance.0,
            max_iter,
            max_coeff,
            gamma.map(|x| x.0),
        )
        .map_err(|e| match e {
            IntegerRelationError::CoefficientLimit => {
                exceptions::PyValueError::new_err("Coefficient limit exceeded")
            }
            IntegerRelationError::IterationLimit(_) => {
                exceptions::PyValueError::new_err("Iteration limit exceeded")
            }
            IntegerRelationError::PrecisionLimit => {
                exceptions::PyValueError::new_err("Precision limit exceeded")
            }
        })?;

        Ok(res
            .into_iter()
            .map(|x| x.into_pyobject(py).unwrap())
            .collect())
    }
}

#[cfg_attr(
    feature = "python_stubgen",
    gen_stub_pyclass(module = "symbolica.core")
)]
#[pyclass(name = "PrimeIterator", module = "symbolica.core")]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PythonPrimeIterator {
    cur: PrimeIteratorU64,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonPrimeIterator {
    /// Create the iterator.
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Return the next prime.
    #[gen_stub(override_return_type(type_repr = "int"))]
    fn __next__(&mut self) -> Option<u64> {
        self.cur.next()
    }
}

#[cfg(feature = "python_stubgen")]
pyo3_stub_gen::define_stub_info_gatherer!(stub_info);
