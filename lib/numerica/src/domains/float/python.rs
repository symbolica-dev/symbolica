//! Python scalar types and conversion adapters for embedding applications.

use super::{Complex, Float, FloatLike, Real, RealLike, SingleFloat};
use crate::domains::{integer::Integer, rational::Rational};
use numpy::Complex64;
use pyo3::{
    Borrowed, Bound, FromPyObject, IntoPyObject, IntoPyObjectExt, Py, PyAny, PyErr, PyRef,
    PyResult, Python, exceptions, pyclass,
    pyclass::CompareOp,
    pymethods,
    sync::PyOnceLock,
    types::{
        PyAnyMethods, PyBool, PyComplex, PyComplexMethods, PyDict, PyDictMethods, PyFloat,
        PyFloatMethods, PyInt, PyList, PyModule, PyModuleMethods, PyString, PyStringMethods,
        PyTuple, PyTupleMethods, PyType,
    },
};
#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::{
    PyStubType, TypeInfo,
    derive::{gen_stub_pyclass, gen_stub_pymethods},
    impl_stub_type,
};

/// An immutable real number with arbitrary binary precision.
///
/// Notes
/// -----
/// Values use binary floating-point arithmetic, so decimal fractions need not
/// be exact. Arithmetic tracks accuracy and may change the result's precision.
/// Native numeric operands are converted at this Float's precision; existing
/// Float operands keep their own precision. Increasing precision does not
/// recover lost digits. Operations return new values; Float is unhashable.
///
/// Supports arithmetic and real comparisons with Float, int, float and Decimal.
/// Addition, subtraction, multiplication and division with complex operands
/// return ComplexFloat. Use float(x), int(x), to_decimal() or as_integer_ratio()
/// for explicit conversion. str(x) displays significant digits; repr(x)
/// preserves value and precision. Formatting accepts Decimal-style specifications.
///
/// Examples
/// --------
/// >>> from symbolica import Float
/// >>> x = Float("1.25", decimal_digits=80)
/// >>> x.precision
/// 266
/// >>> x.as_integer_ratio()
/// (5, 4)
/// >>> str(x + 2)
/// '3.25'
/// >>> Float.from_ratio(1, 3, precision=200).to_decimal(10)
/// Decimal('0.3333333333')
/// >>> Float.pi(decimal_digits=60).sin().is_finite()
/// True
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(frozen, from_py_object, name = "Float")]
#[derive(Clone)]
pub struct PythonFloat(pub Float);

/// An immutable complex number with arbitrary-precision Float components.
///
/// Notes
/// -----
/// Each component tracks its own precision. The precision property reports
/// the minimum; real.precision and imag.precision expose each component.
///
/// Arithmetic accepts Float, ComplexFloat, int, float, complex and Decimal
/// operands. Use the constructor to convert strings or (real, imag) pairs.
/// Accuracy tracking may change component precision. Values are immutable and
/// unhashable. Equality is supported; ordering comparisons raise TypeError.
/// abs(z) and norm() return a real Float. Elementary functions use the principal
/// complex branch; signed zero selects the side of a branch cut where applicable.
///
/// Use complex(z) for a native complex value, as_tuple() for Float components,
/// or to_decimal_tuple() for Decimal components. Decimal conversion is exact
/// by default and independent of the global decimal context.
///
/// Examples
/// --------
/// >>> from symbolica import ComplexFloat, Float
/// >>> z = ComplexFloat("3", "4", decimal_digits=60)
/// >>> z.as_tuple() == (Float(3), Float(4))
/// True
/// >>> abs(z) == Float(5)
/// True
/// >>> str(z.conjugate())
/// '(3-4j)'
/// >>> ComplexFloat("1.25-2.5j").to_decimal_tuple()
/// (Decimal('1.25'), Decimal('-2.5'))
/// >>> ComplexFloat.i(precision=200) ** 2 == -1
/// True
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(frozen, from_py_object, name = "ComplexFloat")]
#[derive(Clone)]
pub struct PythonComplexFloat(pub Complex<Float>);

#[cfg(feature = "python-module")]
#[pyo3::pymodule]
fn numerica(module: &Bound<'_, PyModule>) -> PyResult<()> {
    register_python_floats(module)
}

/// Register Float and ComplexFloat in an embedding application's Python module.
/// Their __module__ attribute is set to the supplied module's name.
pub fn register_python_floats(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PythonFloat>()?;
    module.add_class::<PythonComplexFloat>()?;
    let name = module.name()?;
    module
        .py()
        .get_type::<PythonFloat>()
        .setattr("__module__", &name)?;
    module
        .py()
        .get_type::<PythonComplexFloat>()
        .setattr("__module__", &name)?;
    Ok(())
}

// Give embedding stub generators the concrete Decimal return type.
pub struct PythonDecimal(Py<PyAny>);
impl<'py> IntoPyObject<'py> for PythonDecimal {
    type Target = PyAny;
    type Output = Bound<'py, PyAny>;
    type Error = std::convert::Infallible;
    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        Ok(self.0.into_bound(py))
    }
}
#[cfg(feature = "python_stubgen")]
impl PyStubType for PythonDecimal {
    fn type_output() -> TypeInfo {
        TypeInfo::with_module("decimal.Decimal", "decimal".into())
    }
}

static PYDECIMAL: PyOnceLock<Py<PyType>> = PyOnceLock::new();
fn decimal_type(py: Python<'_>) -> PyResult<&Py<PyType>> {
    PYDECIMAL.get_or_try_init(py, || {
        Ok(py.import("decimal")?.getattr("Decimal")?.extract()?)
    })
}
fn precision_bits(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Option<u32>> {
    if precision.is_some() && decimal_digits.is_some() {
        return Err(exceptions::PyValueError::new_err(
            "Specify precision (bits) or decimal_digits, not both",
        ));
    }
    let precision = match decimal_digits {
        Some(digits) => Some(
            Float::decimal_digits_to_bits(digits as f64)
                .map_err(exceptions::PyValueError::new_err)?,
        ),
        None => precision,
    };
    if let Some(p) = precision {
        Float::check_precision(p).map_err(exceptions::PyValueError::new_err)?;
    }
    Ok(precision)
}
fn real_input(value: &Bound<'_, PyAny>, precision: Option<u32>) -> PyResult<Float> {
    if let Ok(value) = value.extract::<PyRef<'_, PythonFloat>>() {
        let mut result = value.0.clone();
        if let Some(p) = precision {
            result.set_prec(p);
        }
        return Ok(result);
    }
    if value.is_instance_of::<PyBool>() {
        return Ok(Float::with_val(
            precision.unwrap_or(53),
            u32::from(value.extract::<bool>()?),
        ));
    }
    if let Ok(value) = value.cast::<PyString>() {
        return Float::parse(&value.to_cow()?, precision)
            .map_err(exceptions::PyValueError::new_err);
    }
    if value.is_instance_of::<PyInt>()
        || value.is_instance(decimal_type(value.py())?.bind(value.py()))?
    {
        return Float::parse(&value.str()?.to_cow()?, precision)
            .map_err(exceptions::PyValueError::new_err);
    }
    if let Ok(value) = value.cast::<PyFloat>() {
        return Ok(Float::with_val(precision.unwrap_or(53), value.value()));
    }
    Err(exceptions::PyTypeError::new_err(
        "Expected Float, int, float, Decimal, or a decimal string",
    ))
}
fn complex_input(value: &Bound<'_, PyAny>, precision: Option<u32>) -> PyResult<Complex<Float>> {
    if let Ok(value) = value.extract::<PyRef<'_, PythonComplexFloat>>() {
        let mut result = value.0.clone();
        if let Some(p) = precision {
            result.re.set_prec(p);
            result.im.set_prec(p);
        }
        return Ok(result);
    }
    if let Ok(value) = value.cast::<PyComplex>() {
        let p = precision.unwrap_or(53);
        return Ok(Complex::new(
            Float::with_val(p, value.real()),
            Float::with_val(p, value.imag()),
        ));
    }
    if let Ok(pair) = value.cast::<PyTuple>() {
        if pair.len() != 2 {
            return Err(exceptions::PyValueError::new_err(
                "A complex pair must contain exactly two components",
            ));
        }
        return Ok(Complex::new(
            real_input(&pair.get_item(0)?, precision)?,
            real_input(&pair.get_item(1)?, precision)?,
        ));
    }
    if let Ok(text) = value.cast::<PyString>() {
        let text = text.to_cow()?;
        let text = text.trim();
        let text = text
            .strip_prefix('(')
            .and_then(|s| s.strip_suffix(')'))
            .unwrap_or(text);
        if let Some(body) = text.strip_suffix(['j', 'i']) {
            let split = body
                .char_indices()
                .skip(1)
                .filter(|&(i, c)| {
                    matches!(c, '+' | '-') && !matches!(body.as_bytes()[i - 1], b'e' | b'E')
                })
                .map(|(i, _)| i)
                .last();
            let (re, im) = split.map_or(("0", body), |i| body.split_at(i));
            let im = match im {
                "" | "+" => "1",
                "-" => "-1",
                s => s,
            };
            let re =
                Float::parse(re.trim(), precision).map_err(exceptions::PyValueError::new_err)?;
            let im =
                Float::parse(im.trim(), precision).map_err(exceptions::PyValueError::new_err)?;
            return Ok(Complex::new(re, im));
        }
    }
    let re = real_input(value, precision)?;
    let im = re.zero();
    Ok(Complex::new(re, im))
}

fn scalar_constant(
    name: &str,
    precision: Option<u32>,
    decimal_digits: Option<u32>,
) -> PyResult<Float> {
    let p = precision_bits(precision, decimal_digits)?.unwrap_or(53);
    let work = p
        .checked_add(8)
        .ok_or_else(|| exceptions::PyValueError::new_err("Precision is too large"))?;
    Float::check_precision(work).map_err(exceptions::PyValueError::new_err)?;
    let template = Float::new(work);
    let mut value = match name {
        "pi" => template.pi(),
        "e" => template.e(),
        "euler" => template.euler(),
        "phi" => template.phi(),
        _ => unreachable!(),
    };
    value.set_prec(p);
    Ok(value)
}
fn sample_float(py: Python<'_>, p: u32, rng: Option<&Bound<'_, PyAny>>) -> PyResult<Float> {
    let default_rng;
    let rng = match rng {
        Some(rng) => rng,
        None => {
            default_rng = py.import("random")?.into_any();
            &default_rng
        }
    };
    let numerator = rng
        .call_method1("getrandbits", (p,))?
        .extract::<Integer>()?;
    let denominator = Integer::from(1) << p;
    if numerator.is_negative() || numerator >= denominator {
        return Err(exceptions::PyValueError::new_err(
            "getrandbits returned a value outside the requested bit range",
        ));
    }
    Ok(Float::new(p).from_rational(&Rational::from((numerator, denominator))))
}
fn complex_integer_power(value: &Complex<Float>, mut power: u64) -> Complex<Float> {
    let mut base = value.clone();
    let mut result = value.one();
    while power != 0 {
        if power & 1 != 0 {
            result *= &base;
        }
        power >>= 1;
        if power != 0 {
            base = base.clone() * base;
        }
    }
    result
}

/// Return a Decimal using a private context. With no digit limit the conversion
/// is exact: a dyadic rational n/2^k requires at most digits(n)+k decimal digits.
fn to_decimal<'py>(
    py: Python<'py>,
    value: &Float,
    digits: Option<u32>,
) -> PyResult<Bound<'py, PyAny>> {
    if digits == Some(0) {
        return Err(exceptions::PyValueError::new_err("digits must be positive"));
    }
    let decimal = decimal_type(py)?.bind(py);
    if !value.is_finite() {
        let v = value.to_f64();
        return decimal.call1((if v.is_nan() {
            "NaN"
        } else if v.is_sign_negative() {
            "-Infinity"
        } else {
            "Infinity"
        },));
    }
    if value.is_zero() {
        return decimal.call1((if value.is_sign_negative() { "-0" } else { "0" },));
    }
    let ratio = value
        .try_to_rational()
        .expect("finite floats have an exact rational representation");
    let numerator = ratio.numerator_ref().to_string();
    let denominator = ratio.denominator_ref().to_string();
    let exact_digits = numerator.len() as u64 + ratio.denominator_ref().significant_bits();
    let module = py.import("decimal")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("prec", digits.map(u64::from).unwrap_or(exact_digits))?;
    kwargs.set_item("rounding", module.getattr("ROUND_HALF_EVEN")?)?;
    kwargs.set_item("traps", PyList::empty(py))?;
    kwargs.set_item("Emax", module.getattr("MAX_EMAX")?)?;
    kwargs.set_item("Emin", module.getattr("MIN_EMIN")?)?;
    let context = module.getattr("Context")?.call((), Some(&kwargs))?;
    context.call_method1(
        "divide",
        (decimal.call1((numerator,))?, decimal.call1((denominator,))?),
    )
}
fn display_float(py: Python<'_>, value: &Float, roundtrip: bool) -> PyResult<String> {
    if !value.is_finite() || value.is_zero() {
        return Ok(to_decimal(py, value, None)?.str()?.to_cow()?.into_owned());
    }
    let digits = if roundtrip {
        (value.prec() as f64 * std::f64::consts::LOG10_2).ceil() as u32 + 1
    } else {
        ((value.prec() as f64 * std::f64::consts::LOG10_2).floor() as u32).max(1)
    };
    let text = to_decimal(py, value, Some(digits))?
        .str()?
        .to_cow()?
        .into_owned();
    // Remove insignificant fractional zeroes without rounding through Python's global context.
    let (mantissa, exponent) = text
        .split_once('E')
        .map_or((text.as_str(), ""), |(m, _)| (m, &text[m.len()..]));
    let mantissa = if mantissa.contains('.') {
        mantissa.trim_end_matches('0').trim_end_matches('.')
    } else {
        mantissa
    };
    Ok(format!("{mantissa}{exponent}"))
}
fn latex_float(py: Python<'_>, value: &Float) -> PyResult<String> {
    // Reuse the decimal display so rich output never rounds through binary64
    // or Python's global decimal context.
    let text = display_float(py, value, false)?;
    Ok(match text.as_str() {
        "Infinity" => r"\infty".to_owned(),
        "-Infinity" => r"-\infty".to_owned(),
        "NaN" => r"\mathrm{NaN}".to_owned(),
        _ => match text.split_once('E') {
            Some((mantissa, exponent)) => {
                let exponent = exponent.strip_prefix('+').unwrap_or(exponent);
                format!(r"{mantissa}\times 10^{{{exponent}}}")
            }
            None => text,
        },
    })
}
fn python_int<'py>(py: Python<'py>, value: &Integer) -> PyResult<Bound<'py, PyAny>> {
    py.get_type::<PyInt>().call1((value.to_string(),))
}
fn exact_comparison(
    py: Python<'_>,
    value: &Float,
    other: &Bound<'_, PyAny>,
    op: CompareOp,
) -> PyResult<Py<PyAny>> {
    if let Ok(other) = other.cast::<PyComplex>() {
        if !matches!(op, CompareOp::Eq | CompareOp::Ne) {
            return Ok(py.NotImplemented());
        }
        let equal = other.imag() == 0.0
            && exact_comparison(
                py,
                value,
                PyFloat::new(py, other.real()).as_any(),
                CompareOp::Eq,
            )?
            .extract::<bool>(py)?;
        return (if matches!(op, CompareOp::Eq) {
            equal
        } else {
            !equal
        })
        .into_py_any(py);
    }
    let decimal = decimal_type(py)?.bind(py);
    let rhs = if let Ok(f) = other.extract::<PyRef<'_, PythonFloat>>() {
        to_decimal(py, &f.0, None)?
    } else if other.is_instance_of::<PyInt>() {
        decimal.call1((other,))?
    } else if other.is_instance_of::<PyFloat>() {
        decimal.call_method1("from_float", (other,))?
    } else if other.is_instance(decimal)? {
        other.clone()
    } else {
        return Ok(py.NotImplemented());
    };
    let lhs = to_decimal(py, value, None)?;
    if lhs.call_method0("is_nan")?.extract::<bool>()?
        || rhs.call_method0("is_nan")?.extract::<bool>()?
    {
        return matches!(op, CompareOp::Ne).into_py_any(py);
    }
    lhs.rich_compare(&rhs, op)?.unbind().into_py_any(py)
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl PythonFloat {
    /// Construct an immutable real number with arbitrary binary precision.
    ///
    /// Parameters
    /// ----------
    /// value : Float, int, float, str, Decimal, optional
    ///     Initial value; omitted or None means zero. Strings and Decimal values
    ///     are rounded directly to the requested binary precision. For decimal
    ///     input, use a string such as "0.1".
    /// precision : int, optional
    ///     Working precision in bits. Mutually exclusive with decimal_digits.
    /// decimal_digits : int, optional
    ///     Decimal working precision, converted to ceil(decimal_digits * log2(10)) bits.
    ///
    /// Notes
    /// -----
    /// Without a precision option, Float inputs retain their precision, native
    /// floats use 53 bits, and strings, integers and Decimal inputs infer precision
    /// from their significant decimal digits, with a minimum of 53 bits.
    /// Unsupported input types raise TypeError; malformed strings, invalid precision,
    /// or supplying both precision options raise ValueError. Negative or out-of-range
    /// integer precision arguments raise OverflowError.
    #[new]
    #[pyo3(signature = (value=None, *, precision=None, decimal_digits=None))]
    fn new(
        value: Option<&Bound<'_, PyAny>>,
        precision: Option<u32>,
        decimal_digits: Option<u32>,
    ) -> PyResult<Self> {
        let p = precision_bits(precision, decimal_digits)?;
        Ok(Self(match value {
            Some(v) => real_input(v, p)?,
            None => Float::new(p.unwrap_or(53)),
        }))
    }
    /// Construct numerator / denominator from two Python integers.
    ///
    /// Specify precision in bits or decimal_digits, never both; the default is
    /// 53 bits. The rational is rounded directly without conversion through a
    /// native float. A zero denominator raises ZeroDivisionError.
    ///
    /// Parameters
    /// ----------
    /// numerator : int
    ///     Numerator of the rational value; accepts arbitrary-sized Python
    ///     integers.
    /// denominator : int
    ///     Nonzero denominator of the rational value; either sign is accepted.
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits.
    ///
    /// Examples
    /// --------
    /// >>> Float.from_ratio(1, 8, precision=100).to_decimal()
    /// Decimal('0.125')
    #[staticmethod]
    #[pyo3(signature = (numerator, denominator, *, precision=None, decimal_digits=None))]
    fn from_ratio(
        numerator: &Bound<'_, PyInt>,
        denominator: &Bound<'_, PyInt>,
        precision: Option<u32>,
        decimal_digits: Option<u32>,
    ) -> PyResult<Self> {
        let p = precision_bits(precision, decimal_digits)?.unwrap_or(53);
        let n = numerator
            .str()?
            .to_cow()?
            .parse::<Integer>()
            .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
        let d = denominator
            .str()?
            .to_cow()?
            .parse::<Integer>()
            .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
        if d.is_zero() {
            return Err(exceptions::PyZeroDivisionError::new_err(
                "denominator is zero",
            ));
        }
        let ratio = Rational::from((n, d));
        Ok(Self(Float::new(p).from_rational(&ratio)))
    }
    /// The working precision in bits. Read-only; use with_precision() to return a rounded copy.
    #[getter(precision)]
    fn precision_property(&self) -> u32 {
        self.0.prec()
    }
    /// The real part as a Float with the same value and precision.
    #[getter]
    fn real(&self) -> PythonFloat {
        self.clone()
    }
    /// Zero as a Float at this value's precision.
    #[getter]
    fn imag(&self) -> PythonFloat {
        Self(self.0.zero())
    }
    /// Return a copy rounded to a new working precision.
    ///
    /// Specify exactly one of precision (bits) or decimal_digits. Missing, invalid,
    /// or conflicting precision options raise ValueError; negative or out-of-range
    /// integer arguments raise OverflowError. Increasing precision
    /// cannot recover digits already lost. The original value is unchanged.
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. Exactly one precision option is required.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. Exactly one precision
    ///     option is required.
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn with_precision(
        &self,
        precision: Option<u32>,
        decimal_digits: Option<u32>,
    ) -> PyResult<Self> {
        let p = precision_bits(precision, decimal_digits)?.ok_or_else(|| {
            exceptions::PyValueError::new_err("Specify precision or decimal_digits")
        })?;
        let mut value = self.0.clone();
        value.set_prec(p);
        Ok(Self(value))
    }
    /// Convert the stored binary value to Decimal.
    ///
    /// With digits omitted, conversion is exact. A positive digits value rounds
    /// to that many significant decimal digits, using round-half-even. Neither
    /// mode depends on or modifies Python's global decimal context. Preserves
    /// signed zero, NaN and infinity. Zero digits raises ValueError; negative or
    /// out-of-range integer digits raise OverflowError.
    ///
    /// Parameters
    /// ----------
    /// digits : int, optional
    ///     Positive number of significant decimal digits per converted value.
    ///     Omitted or None converts the stored binary value exactly; otherwise
    ///     round half-even.
    ///
    /// Examples
    /// --------
    /// >>> Float.from_ratio(1, 3, precision=100).to_decimal(digits=5)
    /// Decimal('0.33333')
    #[pyo3(signature = (digits=None))]
    fn to_decimal(&self, py: Python<'_>, digits: Option<u32>) -> PyResult<PythonDecimal> {
        Ok(PythonDecimal(to_decimal(py, &self.0, digits)?.unbind()))
    }
    /// Return the exact (numerator, denominator) of the stored binary value.
    ///
    /// Both entries are Python integers and the denominator is positive. This
    /// describes the stored value, which may approximate the original decimal
    /// input. NaN and infinity raise ValueError.
    ///
    /// Examples
    /// --------
    /// >>> Float("1.25").as_integer_ratio()
    /// (5, 4)
    fn as_integer_ratio(&self, py: Python<'_>) -> PyResult<(Py<PyInt>, Py<PyInt>)> {
        let r = self.0.try_to_rational().ok_or_else(|| {
            exceptions::PyValueError::new_err("NaN and infinity have no integer ratio")
        })?;
        Ok((
            python_int(py, r.numerator_ref())?
                .cast_into::<PyInt>()?
                .unbind(),
            python_int(py, r.denominator_ref())?
                .cast_into::<PyInt>()?
                .unbind(),
        ))
    }
    /// Return True for finite values, including zero; False for NaN and infinities.
    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }
    /// Return whether the value is NaN (not a number).
    fn is_nan(&self) -> bool {
        self.0.to_f64().is_nan()
    }
    /// Return whether the value is positive or negative infinity; False for NaN.
    fn is_infinite(&self) -> bool {
        !self.0.is_finite() && !self.is_nan()
    }
    /// Return a decimal display using significant digits appropriate to the precision.
    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        display_float(py, &self.0, false)
    }
    /// Return HTML with the same significant digits as str(self).
    fn _repr_html_(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("<pre>{}</pre>", self.__str__(py)?))
    }
    /// Return LaTeX with the same significant digits as str(self), using powers of ten for scientific notation.
    fn _repr_latex_(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("$${}$$", latex_float(py, &self.0)?))
    }
    /// Write the same significant digits as str(self) to a notebook pretty printer.
    ///
    /// Parameters
    /// ----------
    /// pretty : object
    ///     Pretty printer providing a text(string) method.
    /// cycle : bool
    ///     Whether the printer detected a reference cycle; prints ... if True.
    fn _repr_pretty_(&self, pretty: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        let text = if cycle {
            "...".to_owned()
        } else {
            self.__str__(pretty.py())?
        };
        pretty.call_method1("text", (text,))?;
        Ok(())
    }
    /// Return a constructor expression that preserves the value and its precision.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "Float({:?}, precision={})",
            display_float(py, &self.0, true)?,
            self.precision_property()
        ))
    }
    /// Format with a Decimal-style specification. An empty specification uses str(self).
    ///
    /// Parameters
    /// ----------
    /// spec : str
    ///     Decimal-style format specification, such as ".12f". An empty string uses
    ///     str(self).
    fn __format__(&self, py: Python<'_>, spec: &str) -> PyResult<String> {
        if spec.is_empty() {
            return self.__str__(py);
        }
        to_decimal(py, &self.0, None)?
            .call_method1("__format__", (spec,))?
            .extract()
    }
    /// Convert to a native binary64 float, potentially losing precision or overflowing to infinity.
    fn __float__(&self) -> f64 {
        self.0.to_f64()
    }
    /// Convert to a Python integer by truncating toward zero. NaN and infinity cannot be converted.
    fn __int__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(to_decimal(py, &self.0, None)?
            .call_method0("__int__")?
            .unbind())
    }
    /// Return False for zero and True otherwise, including NaN.
    fn __bool__(&self) -> bool {
        !self.0.is_zero()
    }
    /// Return a copy preserving the value and precision.
    fn __copy__(&self) -> Self {
        self.clone()
    }
    /// Return a copy preserving the value and precision.
    ///
    /// Parameters
    /// ----------
    /// memo : dict
    ///     Memo dictionary supplied by copy.deepcopy.
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
    /// Compare numerically with compatible scalars; NaN is unequal to every value. Complex values support equality only.
    ///
    /// Parameters
    /// ----------
    /// other : object
    ///     Value to compare numerically. Compatible numeric types compare by value;
    ///     unsupported types are not equal.
    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        exact_comparison(py, &self.0, other, op)
    }
    /// Return the additive inverse.
    fn __neg__(&self) -> Self {
        Self(-self.0.clone())
    }
    /// Return a copy of this value.
    fn __pos__(&self) -> Self {
        self.clone()
    }
    /// Return the magnitude as a real Float.
    fn __abs__(&self) -> Self {
        Self(self.0.norm())
    }
    /// Return the nonnegative square root. Negative real inputs yield NaN; use ComplexFloat for complex roots.
    fn sqrt(&self) -> Self {
        Self(self.0.sqrt())
    }
    /// Return the exponential e**self with accuracy tracking.
    fn exp(&self) -> Self {
        Self(self.0.exp())
    }
    /// Return the natural logarithm. Zero yields negative infinity; negative inputs yield NaN.
    fn ln(&self) -> Self {
        Self(self.0.log())
    }
    /// Alias for ln(), the natural logarithm (base e).
    fn log(&self) -> Self {
        self.ln()
    }
    /// Return log(1+self), retaining small increments lost when adding one.
    fn log1p(&self) -> Self {
        Self(self.0.log1p())
    }
    /// Return the sine, with the argument in radians.
    fn sin(&self) -> Self {
        Self(self.0.sin())
    }
    /// Return the cosine, with the argument in radians.
    fn cos(&self) -> Self {
        Self(self.0.cos())
    }
    /// Return the tangent, with the argument in radians.
    fn tan(&self) -> Self {
        Self(self.0.tan())
    }
    /// Return the inverse sine in radians, in [-pi/2, pi/2]. Inputs outside [-1, 1] yield NaN.
    fn asin(&self) -> Self {
        Self(self.0.asin())
    }
    /// Return the inverse cosine in radians, in [0, pi]. Inputs outside [-1, 1] yield NaN.
    fn acos(&self) -> Self {
        Self(self.0.acos())
    }
    /// Return the hyperbolic sine with accuracy tracking.
    fn sinh(&self) -> Self {
        Self(self.0.sinh())
    }
    /// Return the hyperbolic cosine with accuracy tracking.
    fn cosh(&self) -> Self {
        Self(self.0.cosh())
    }
    /// Return the hyperbolic tangent with accuracy tracking.
    fn tanh(&self) -> Self {
        Self(self.0.tanh())
    }
    /// Return the reciprocal hyperbolic cosine without overflowing an intermediate cosh.
    fn sech(&self) -> Self {
        Self(self.0.sech())
    }
    /// Return the reciprocal hyperbolic sine, retaining accuracy near zero and at infinity.
    fn csch(&self) -> Self {
        Self(self.0.csch())
    }
    /// Return the inverse hyperbolic sine.
    fn asinh(&self) -> Self {
        Self(self.0.asinh())
    }
    /// Return the nonnegative inverse hyperbolic cosine. Inputs below one yield NaN.
    fn acosh(&self) -> Self {
        Self(self.0.acosh())
    }
    /// Return the inverse hyperbolic tangent. Inputs outside [-1, 1] yield NaN; +/-1 yield signed infinity.
    fn atanh(&self) -> Self {
        Self(self.0.atanh())
    }
    /// Return the inverse tangent in radians, in [-pi/2, pi/2].
    fn atan(&self) -> Self {
        Self(self.0.atan2(&self.0.one()))
    }
    /// Construct pi with precision in bits or decimal_digits (default: 53 bits).
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits.
    #[staticmethod]
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn pi(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Self> {
        Ok(Self(scalar_constant("pi", precision, decimal_digits)?))
    }
    /// Construct Euler's number e with precision in bits or decimal_digits (default: 53 bits).
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits.
    #[staticmethod]
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn e(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Self> {
        Ok(Self(scalar_constant("e", precision, decimal_digits)?))
    }
    /// Construct the Euler-Mascheroni constant with precision in bits or decimal_digits (default: 53 bits).
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits.
    #[staticmethod]
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn euler(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Self> {
        Ok(Self(scalar_constant("euler", precision, decimal_digits)?))
    }
    /// Construct the golden ratio (1+sqrt(5))/2 with precision in bits or decimal_digits (default: 53 bits).
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits.
    #[staticmethod]
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn phi(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Self> {
        Ok(Self(scalar_constant("phi", precision, decimal_digits)?))
    }
    /// Alias for euler(), the Euler-Mascheroni constant; precision defaults to 53 bits.
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits.
    #[staticmethod]
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn euler_gamma(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Self> {
        Self::euler(precision, decimal_digits)
    }
    /// Return None. Construct the imaginary unit with ComplexFloat.i().
    #[staticmethod]
    fn i() -> Option<PythonFloat> {
        None
    }
    /// Return a copy with the same value and precision.
    fn conjugate(&self) -> Self {
        self.clone()
    }
    /// Alias for conjugate().
    fn conj(&self) -> Self {
        Self(self.0.conj())
    }
    /// Return the additive inverse, equivalent to -self.
    fn neg(&self) -> Self {
        Self(-self.0.clone())
    }
    /// Return the magnitude as a real Float, equivalent to abs(self).
    fn norm(&self) -> Self {
        self.__abs__()
    }
    /// Return sqrt(self**2 + other**2), avoiding unnecessary overflow and underflow.
    ///
    /// Parameters
    /// ----------
    /// other : Float, int, float or Decimal
    ///     Second coordinate. Native numbers use this value's precision;
    ///     existing Float operands retain their precision.
    fn hypot(&self, other: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self(self.0.hypot(&self.required_operand(other)?)))
    }
    /// Return zero at this value's precision.
    fn zero(&self) -> Self {
        Self(self.0.zero())
    }
    /// Return one at this value's precision.
    fn one(&self) -> Self {
        Self(self.0.one())
    }
    /// Return NaN at this value's precision.
    fn nan(&self) -> Self {
        Self(self.0.nan().expect("floating-point values support NaN"))
    }
    /// Return whether the value is zero; signed zero also counts as zero.
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    /// Return whether the value equals one.
    fn is_one(&self) -> bool {
        self.0.is_one()
    }
    /// Return whether the value is exactly zero.
    fn is_fully_zero(&self) -> bool {
        self.0.is_fully_zero()
    }
    /// Return the working precision in bits; alias for the precision property.
    fn get_precision(&self) -> u32 {
        self.0.get_precision()
    }
    /// Return 2**(-precision) as a native float. Very high precision can underflow to zero.
    fn get_epsilon(&self) -> f64 {
        self.0.get_epsilon()
    }
    /// Return False: arithmetic dynamically tracks precision for these scalar types.
    fn fixed_precision(&self) -> bool {
        self.0.fixed_precision()
    }
    /// Convert a nonnegative platform-sized integer at this value's precision. Out-of-range inputs raise OverflowError.
    ///
    /// Parameters
    /// ----------
    /// value : int
    ///     Integer in [0, 2**pointer_bits-1] to convert.
    fn from_usize(&self, value: usize) -> Self {
        Self(self.0.from_usize(value))
    }
    /// Convert a signed 64-bit integer at this value's precision. Out-of-range inputs raise OverflowError.
    ///
    /// Parameters
    /// ----------
    /// value : int
    ///     Integer in [-2**63, 2**63-1] to convert.
    fn from_i64(&self, value: i64) -> Self {
        Self(self.0.from_i64(value))
    }
    /// Return a new value converted from other, with precision inferred as in the constructor.
    ///
    /// Parameters
    /// ----------
    /// other : Float, int, float, str or Decimal
    ///     Value to copy or convert using constructor precision inference.
    fn set_from(&self, other: PythonMultiPrecisionFloat) -> Self {
        Self(other.0)
    }
    /// Construct zero with precision in bits or decimal_digits (default: 53 bits). Use zero() to retain instance precision.
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits.
    #[staticmethod]
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn new_zero(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Self> {
        let p = precision_bits(precision, decimal_digits)?.unwrap_or(53);
        Ok(Self(Float::with_val(p, 0)))
    }
    /// Construct one with precision in bits or decimal_digits (default: 53 bits). Use one() to retain instance precision.
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits.
    #[staticmethod]
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn new_one(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Self> {
        let p = precision_bits(precision, decimal_digits)?.unwrap_or(53);
        Ok(Self(Float::with_val(p, 1)))
    }
    /// Sample uniformly from [0, 1) using the full working precision.
    ///
    /// rng must supply getrandbits(bits); omitted or None uses Python's random
    /// module. Pass random.Random(seed) for reproducibility.
    ///
    /// Parameters
    /// ----------
    /// rng : object, optional
    ///     Random generator with a getrandbits(bits) method returning an integer in
    ///     [0, 2**bits). Omitted or None uses Python's random module; use
    ///     random.Random(seed) for reproducible samples.
    #[pyo3(signature = (rng=None))]
    fn sample_unit(&self, py: Python<'_>, rng: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let value = sample_float(py, self.precision_property(), rng)?;
        Ok(Self(value))
    }
    /// Convert numerator / denominator at this value's precision. Both inputs must be Python integers; zero denominator raises ZeroDivisionError.
    ///
    /// Parameters
    /// ----------
    /// numerator : int
    ///     Numerator of the rational value; accepts arbitrary-sized Python
    ///     integers.
    /// denominator : int
    ///     Nonzero denominator of the rational value; either sign is accepted.
    fn from_rational(
        &self,
        numerator: &Bound<'_, PyInt>,
        denominator: &Bound<'_, PyInt>,
    ) -> PyResult<Self> {
        Self::from_ratio(
            numerator,
            denominator,
            Some(self.precision_property()),
            None,
        )
    }
    /// Return 1/self. Zero raises ZeroDivisionError.
    fn inv(&self) -> PyResult<Self> {
        if self.0.is_zero() {
            return Err(exceptions::PyZeroDivisionError::new_err("division by zero"));
        }
        Ok(Self(self.0.inv()))
    }
    /// Raise to an unsigned 64-bit integer exponent. Negative or out-of-range exponents raise OverflowError; use ** for signed integer powers.
    ///
    /// Parameters
    /// ----------
    /// exponent : int
    ///     Unsigned exponent in [0, 2**64-1]; zero returns one, including for a
    ///     zero base.
    fn pow(&self, exponent: u64) -> Self {
        Self(self.0.pow(exponent))
    }

    /// Raise to a real numeric exponent with accuracy tracking. Zero to a negative power raises ZeroDivisionError; non-real results yield NaN.
    ///
    /// Parameters
    /// ----------
    /// exponent : Float, int, float or Decimal
    ///     Numeric exponent. Use ** for signed integer powers.
    fn powf(&self, exponent: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self
            .operand(exponent)?
            .ok_or_else(|| exceptions::PyTypeError::new_err("Expected a real numeric operand"))?;
        if self.0.is_zero() && rhs.is_negative() {
            return Err(exceptions::PyZeroDivisionError::new_err(
                "zero to a negative power",
            ));
        }
        Ok(Self(self.0.powf(&rhs)))
    }

    /// Return the quadrant-aware angle atan2(self, x) in radians in [-pi, pi]. Accepts real numeric operands and preserves signed-zero quadrant conventions.
    ///
    /// Parameters
    /// ----------
    /// x : Float, int, float or Decimal
    ///     Horizontal coordinate; self is the vertical coordinate in atan2(self,
    ///     x).
    fn atan2(&self, x: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self
            .operand(x)?
            .ok_or_else(|| exceptions::PyTypeError::new_err("Expected a real numeric operand"))?;
        Ok(Self(self.0.atan2(&rhs)))
    }
    /// Return self*a+b with accuracy tracking, rounding the multiplication and addition separately.
    ///
    /// Parameters
    /// ----------
    /// a : Float, int, float or Decimal
    ///     Multiplier in self*a+b.
    /// b : Float, int, float or Decimal
    ///     Addend in self*a+b.
    fn mul_add(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self(self.0.mul_add(
            &self.required_operand(a)?,
            &self.required_operand(b)?,
        )))
    }
    /// Convert to a native binary64 float; alias for float(self). Precision can be lost and overflow yields infinity.
    fn to_f64(&self) -> f64 {
        self.0.to_f64()
    }
    /// Return the nearest Python integer, rounding ties to even. NaN and infinity raise ValueError.
    fn round_to_nearest_integer(&self, py: Python<'_>) -> PyResult<Py<PyInt>> {
        if !self.0.is_finite() {
            return Err(exceptions::PyValueError::new_err(
                "Cannot round NaN or infinity to an integer",
            ));
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item(
            "rounding",
            py.import("decimal")?.getattr("ROUND_HALF_EVEN")?,
        )?;
        let value =
            to_decimal(py, &self.0, None)?.call_method("to_integral_value", (), Some(&kwargs))?;
        Ok(value
            .call_method0("__int__")?
            .cast_into::<PyInt>()?
            .unbind())
    }
    /// Round ties to even and clamp to [0, 2**pointer_bits-1]. Negative values become zero, positive infinity becomes the maximum, and NaN raises ValueError.
    fn to_usize_clamped(&self, py: Python<'_>) -> PyResult<usize> {
        if self.is_nan() {
            return Err(exceptions::PyValueError::new_err(
                "Cannot convert NaN to an integer",
            ));
        }
        if self.0.is_negative() {
            return Ok(0);
        }
        if self.is_infinite() {
            return Ok(usize::MAX);
        }
        Ok(self
            .round_to_nearest_integer(py)?
            .extract::<usize>(py)
            .unwrap_or(usize::MAX))
    }
    /// Return self + other with accuracy tracking; complex operands produce ComplexFloat.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '+', false)
    }
    /// Return other + self with accuracy tracking; complex operands produce ComplexFloat.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '+', true)
    }
    /// Return self - other with accuracy tracking; complex operands produce ComplexFloat.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '-', false)
    }
    /// Return other - self with accuracy tracking; complex operands produce ComplexFloat.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '-', true)
    }
    /// Return self * other with accuracy tracking; complex operands produce ComplexFloat.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '*', false)
    }
    /// Return other * self with accuracy tracking; complex operands produce ComplexFloat.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '*', true)
    }
    /// Return self / other with accuracy tracking; complex operands produce ComplexFloat. A zero divisor raises ZeroDivisionError.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '/', false)
    }
    /// Return other / self with accuracy tracking; complex operands produce ComplexFloat. A zero divisor raises ZeroDivisionError.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '/', true)
    }
    /// Raise to an integer or real numeric exponent. Negative integer powers are supported; zero to a negative power raises ZeroDivisionError. Modular powers are unsupported.
    ///
    /// Parameters
    /// ----------
    /// exponent : Float, int, float or Decimal
    ///     Numeric exponent. Use ** for signed integer powers.
    /// modulo : None, optional
    ///     Must be None. Three-argument modular exponentiation is unsupported.
    fn __pow__(
        &self,
        py: Python<'_>,
        exponent: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if modulo.is_some() {
            return Ok(py.NotImplemented());
        }
        if let Ok(e) = exponent.extract::<i64>() {
            if exponent.is_instance_of::<PyInt>() {
                if e < 0 && self.0.is_zero() {
                    return Err(exceptions::PyZeroDivisionError::new_err(
                        "zero to a negative power",
                    ));
                }
                let value = self.0.pow(e.unsigned_abs());
                return Self(if e < 0 { value.inv() } else { value }).into_py_any(py);
            }
        }
        let Some(rhs) = self.operand(exponent)? else {
            return Ok(py.NotImplemented());
        };
        if self.0.is_zero() && rhs.is_negative() {
            return Err(exceptions::PyZeroDivisionError::new_err(
                "zero to a negative power",
            ));
        }
        Self(self.0.powf(&rhs)).into_py_any(py)
    }
}
impl PythonFloat {
    fn required_operand(&self, value: &Bound<'_, PyAny>) -> PyResult<Float> {
        self.operand(value)?
            .ok_or_else(|| exceptions::PyTypeError::new_err("Expected a real numeric operand"))
    }
    fn operand(&self, other: &Bound<'_, PyAny>) -> PyResult<Option<Float>> {
        if other.is_instance_of::<PyString>() {
            return Ok(None);
        }
        if let Ok(f) = other.extract::<PyRef<'_, Self>>() {
            return Ok(Some(f.0.clone()));
        }
        match real_input(other, Some(self.0.prec())) {
            Ok(v) => Ok(Some(v)),
            Err(e) if e.is_instance_of::<exceptions::PyTypeError>(other.py()) => Ok(None),
            Err(e) => Err(e),
        }
    }
    fn binary(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: char,
        reverse: bool,
    ) -> PyResult<Py<PyAny>> {
        if other.is_instance_of::<PyComplex>() || other.is_instance_of::<PythonComplexFloat>() {
            return PythonComplexFloat(Complex::new(self.0.clone(), self.0.zero()))
                .binary(py, other, op, reverse);
        }

        let Some(rhs) = self.operand(other)? else {
            return Ok(py.NotImplemented());
        };
        let (a, b) = if reverse {
            (rhs, self.0.clone())
        } else {
            (self.0.clone(), rhs)
        };
        if op == '/' && b.is_zero() {
            return Err(exceptions::PyZeroDivisionError::new_err("division by zero"));
        }
        Self(match op {
            '+' => a + b,
            '-' => a - b,
            '*' => a * b,
            '/' => a / b,
            _ => unreachable!(),
        })
        .into_py_any(py)
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl PythonComplexFloat {
    /// Construct an immutable complex number from scalars or a complex value.
    ///
    /// Parameters
    /// ----------
    /// real : ComplexFloat, Float, int, float, complex, str, Decimal, tuple, optional
    ///     Real component when imag is supplied. Otherwise, accepts a real scalar,
    ///     a complete complex value, a (real, imag) pair, or a string such as
    ///     "1.25-2.5j", "(1+2i)" or "-j". Omitted or None means zero.
    /// imag : Float, int, float, str, Decimal, optional
    ///     Imaginary component. When supplied, real must also be a real scalar.
    /// precision : int, optional
    ///     Working precision of both components in bits; exclusive with decimal_digits.
    /// decimal_digits : int, optional
    ///     Decimal working precision for both components, converted to
    ///     ceil(decimal_digits * log2(10)) bits.
    ///
    /// Notes
    /// -----
    /// Without a precision option, existing components retain their precision;
    /// native complex components use 53 bits. Other components follow Float's
    /// precision inference. precision reports the minimum component precision;
    /// real.precision and imag.precision expose the individual values.
    /// Invalid inputs raise TypeError, ValueError or OverflowError, as for Float.
    #[new]
    #[pyo3(signature = (real=None, imag=None, *, precision=None, decimal_digits=None))]
    fn new(
        real: Option<&Bound<'_, PyAny>>,
        imag: Option<&Bound<'_, PyAny>>,
        precision: Option<u32>,
        decimal_digits: Option<u32>,
    ) -> PyResult<Self> {
        let p = precision_bits(precision, decimal_digits)?;
        Ok(Self(if let Some(imag) = imag {
            let re = match real {
                Some(r) => real_input(r, p)?,
                None => Float::new(p.unwrap_or(53)),
            };
            Complex::new(re, real_input(imag, p)?)
        } else if let Some(real) = real {
            complex_input(real, p)?
        } else {
            Complex::new(Float::new(p.unwrap_or(53)), Float::new(p.unwrap_or(53)))
        }))
    }
    /// The real component as a Float, preserving its own precision.
    #[getter]
    fn real(&self) -> PythonFloat {
        PythonFloat(self.0.re.clone())
    }
    /// The imaginary component as a Float, preserving its own precision.
    #[getter]
    fn imag(&self) -> PythonFloat {
        PythonFloat(self.0.im.clone())
    }
    /// The minimum component precision in bits. Read-only; inspect real.precision and imag.precision individually.
    #[getter(precision)]
    fn precision_property(&self) -> u32 {
        self.0.get_precision()
    }
    /// Return a copy with both components rounded to the requested precision in bits or decimal_digits. Specify exactly one option; invalid options raise ValueError or OverflowError. Increasing precision cannot recover lost digits.
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. Exactly one precision option is required. Applies to
    ///     both components.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. Exactly one precision
    ///     option is required. Applies to both components.
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn with_precision(
        &self,
        precision: Option<u32>,
        decimal_digits: Option<u32>,
    ) -> PyResult<Self> {
        let re = self.real().with_precision(precision, decimal_digits)?;
        let im = self.imag().with_precision(precision, decimal_digits)?;
        Ok(Self(Complex::new(re.0, im.0)))
    }
    /// Return (real, imag) as Decimal values.
    ///
    /// Conversion of each stored component is exact when digits is omitted.
    /// A positive digits value rounds each component to that many significant
    /// decimal digits, using round-half-even. Conversion is independent of
    /// Python's global decimal context and preserves signed zero and special values.
    ///
    /// Parameters
    /// ----------
    /// digits : int, optional
    ///     Positive number of significant decimal digits per converted value.
    ///     Omitted or None converts the stored binary value exactly; otherwise
    ///     round half-even.
    ///
    /// Examples
    /// --------
    /// >>> ComplexFloat("1.25", "-2.5").to_decimal_tuple()
    /// (Decimal('1.25'), Decimal('-2.5'))
    #[pyo3(signature = (digits=None))]
    fn to_decimal_tuple(
        &self,
        py: Python<'_>,
        digits: Option<u32>,
    ) -> PyResult<(PythonDecimal, PythonDecimal)> {
        Ok((
            PythonDecimal(to_decimal(py, &self.0.re, digits)?.unbind()),
            PythonDecimal(to_decimal(py, &self.0.im, digits)?.unbind()),
        ))
    }
    /// Return (real, imag) as two Float values, preserving their individual precisions.
    fn as_tuple(&self) -> (PythonFloat, PythonFloat) {
        (self.real(), self.imag())
    }
    /// Return True only when both components are finite.
    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }
    /// Return True if either component is NaN.
    fn is_nan(&self) -> bool {
        self.real().is_nan() || self.imag().is_nan()
    }
    /// Return the complex conjugate, negating the imaginary component.
    fn conjugate(&self) -> Self {
        Self(self.0.conj())
    }
    /// Return a decimal display using significant digits appropriate to the precision.
    fn __str__(&self, py: Python<'_>) -> PyResult<String> {
        let re = self.real().__str__(py)?;
        let im = self.imag().__str__(py)?;
        Ok(format!(
            "({re}{}{im}j)",
            if im.starts_with('-') { "" } else { "+" }
        ))
    }
    /// Return HTML with the same significant digits as str(self).
    fn _repr_html_(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!("<pre>{}</pre>", self.__str__(py)?))
    }
    /// Return LaTeX preserving each component's displayed precision, using i for the imaginary unit.
    fn _repr_latex_(&self, py: Python<'_>) -> PyResult<String> {
        let re = latex_float(py, &self.0.re)?;
        let im = latex_float(py, &self.0.im)?;
        Ok(format!(
            "$${re}{}{im}\\,i$$",
            if im.starts_with('-') { "" } else { "+" }
        ))
    }
    /// Write the same significant digits as str(self) to a notebook pretty printer.
    ///
    /// Parameters
    /// ----------
    /// pretty : object
    ///     Pretty printer providing a text(string) method.
    /// cycle : bool
    ///     Whether the printer detected a reference cycle; prints ... if True.
    fn _repr_pretty_(&self, pretty: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        let text = if cycle {
            "...".to_owned()
        } else {
            self.__str__(pretty.py())?
        };
        pretty.call_method1("text", (text,))?;
        Ok(())
    }
    /// Return a constructor expression that preserves the value and its precision.
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        Ok(format!(
            "ComplexFloat({}, {})",
            self.real().__repr__(py)?,
            self.imag().__repr__(py)?
        ))
    }
    /// Format with a Decimal-style specification, applied separately to complex components. An empty specification uses str(self).
    ///
    /// Parameters
    /// ----------
    /// spec : str
    ///     Decimal-style format specification, such as ".12f". An empty string uses
    ///     str(self). Applied to both components.
    fn __format__(&self, py: Python<'_>, spec: &str) -> PyResult<String> {
        if spec.is_empty() {
            return self.__str__(py);
        }
        let re = self.real().__format__(py, spec)?;
        let im = self.imag().__format__(py, spec)?;
        Ok(format!(
            "({re}{}{im}j)",
            if im.starts_with(['-', '+']) { "" } else { "+" }
        ))
    }
    /// Convert both components to native binary64 floats, potentially losing precision or overflowing to infinity.
    fn __complex__(&self, py: Python<'_>) -> Py<PyComplex> {
        PyComplex::from_doubles(py, self.0.re.to_f64(), self.0.im.to_f64()).unbind()
    }
    /// Return False for zero and True otherwise, including NaN.
    fn __bool__(&self) -> bool {
        !self.0.is_zero()
    }
    /// Return a copy preserving the value and component precisions.
    fn __copy__(&self) -> Self {
        self.clone()
    }
    /// Return a copy preserving the value and component precisions.
    ///
    /// Parameters
    /// ----------
    /// memo : dict
    ///     Memo dictionary supplied by copy.deepcopy.
    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }
    /// Return the additive inverse.
    fn __neg__(&self) -> Self {
        Self(-self.0.clone())
    }
    /// Return a copy of this value.
    fn __pos__(&self) -> Self {
        self.clone()
    }
    /// Return the magnitude as a real Float.
    fn __abs__(&self) -> PythonFloat {
        PythonFloat(self.0.norm().re)
    }
    /// Compare numerically with compatible scalars; NaN is unequal to every value. Complex values support equality only.
    ///
    /// Parameters
    /// ----------
    /// other : object
    ///     Value to compare numerically. Compatible numeric types compare by value;
    ///     unsupported types are not equal.
    fn __richcmp__(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: CompareOp,
    ) -> PyResult<Py<PyAny>> {
        if !matches!(op, CompareOp::Eq | CompareOp::Ne) {
            return Ok(py.NotImplemented());
        }
        if other.is_instance_of::<PyTuple>() || other.is_instance_of::<PyString>() {
            return Ok(py.NotImplemented());
        }
        let rhs = if let Ok(value) = other.extract::<PyRef<'_, PythonComplexFloat>>() {
            value.0.clone()
        } else if let Ok(value) = other.cast::<PyComplex>() {
            Complex::new(
                Float::with_val(53, value.real()),
                Float::with_val(53, value.imag()),
            )
        } else {
            let real_equal = exact_comparison(py, &self.0.re, other, CompareOp::Eq)?;
            if real_equal.bind(py).is(py.NotImplemented().bind(py)) {
                return Ok(py.NotImplemented());
            }
            let equal = self.0.im.is_zero() && real_equal.extract::<bool>(py)?;
            return (if matches!(op, CompareOp::Eq) {
                equal
            } else {
                !equal
            })
            .into_py_any(py);
        };
        let equal = self.0.re.partial_cmp(&rhs.re) == Some(std::cmp::Ordering::Equal)
            && self.0.im.partial_cmp(&rhs.im) == Some(std::cmp::Ordering::Equal);
        (if matches!(op, CompareOp::Eq) {
            equal
        } else {
            !equal
        })
        .into_py_any(py)
    }
    /// Return the principal complex square root; signed zero distinguishes the sides of the negative-real branch cut.
    fn sqrt(&self) -> Self {
        Self(self.0.sqrt())
    }
    /// Return the exponential e**self with accuracy tracking.
    fn exp(&self) -> Self {
        Self(self.0.exp())
    }
    /// Return the principal complex natural logarithm. Its imaginary part is the argument in [-pi, pi].
    fn ln(&self) -> Self {
        Self(self.0.log())
    }
    /// Alias for ln(), the principal natural logarithm (base e).
    fn log(&self) -> Self {
        self.ln()
    }
    /// Return the principal log(1+self), retaining small increments and signed-zero branch cuts.
    fn log1p(&self) -> Self {
        if self.0.im.is_zero() {
            if self.0.re >= -self.0.re.one() {
                return Self(Complex::new(self.0.re.log1p(), self.0.im.clone()));
            }
            return Self(Complex::new(self.0.re.one() + &self.0.re, self.0.im.clone()).log());
        }
        Self(self.0.log1p())
    }
    /// Return the sine, with the argument in radians.
    fn sin(&self) -> Self {
        Self(self.0.sin())
    }
    /// Return the cosine, with the argument in radians.
    fn cos(&self) -> Self {
        Self(self.0.cos())
    }
    /// Return the principal complex inverse sine; signed zero selects the side of a real-axis branch cut.
    fn asin(&self) -> Self {
        Self(self.axis_value("asin").unwrap_or_else(|| self.0.asin()))
    }
    /// Return the principal complex inverse cosine; signed zero selects the side of a real-axis branch cut.
    fn acos(&self) -> Self {
        Self(self.axis_value("acos").unwrap_or_else(|| self.0.acos()))
    }
    /// Return the hyperbolic sine with accuracy tracking.
    fn sinh(&self) -> Self {
        Self(self.0.sinh())
    }
    /// Return the hyperbolic cosine with accuracy tracking.
    fn cosh(&self) -> Self {
        Self(self.0.cosh())
    }
    /// Return the hyperbolic tangent with accuracy tracking.
    fn tanh(&self) -> Self {
        Self(self.0.tanh())
    }
    /// Return the reciprocal hyperbolic cosine without overflowing an intermediate cosh.
    fn sech(&self) -> Self {
        Self(self.0.sech())
    }
    /// Return the reciprocal hyperbolic sine, retaining accuracy near zero and at infinity.
    fn csch(&self) -> Self {
        Self(self.0.csch())
    }
    /// Return the principal complex inverse hyperbolic sine.
    fn asinh(&self) -> Self {
        Self(self.axis_value("asinh").unwrap_or_else(|| self.0.asinh()))
    }
    /// Return the principal complex inverse hyperbolic cosine.
    fn acosh(&self) -> Self {
        Self(self.axis_value("acosh").unwrap_or_else(|| self.0.acosh()))
    }
    /// Return the principal complex inverse hyperbolic tangent.
    fn atanh(&self) -> Self {
        Self(self.axis_value("atanh").unwrap_or_else(|| self.0.atanh()))
    }
    /// Return the tangent, with the argument in radians.
    fn tan(&self) -> Self {
        Self(self.0.tan())
    }
    /// Return the principal complex inverse tangent.
    fn atan(&self) -> Self {
        Self(
            self.axis_value("atan")
                .unwrap_or_else(|| self.0.atan2(&self.0.one())),
        )
    }
    /// Construct pi with precision in bits or decimal_digits (default: 53 bits). Complex results have zero imaginary part.
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits. Applies to
    ///     both components.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits. Applies to both components.
    #[staticmethod]
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn pi(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Self> {
        Ok(Self(Complex::from(scalar_constant(
            "pi",
            precision,
            decimal_digits,
        )?)))
    }
    /// Construct Euler's number e with precision in bits or decimal_digits (default: 53 bits). Complex results have zero imaginary part.
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits. Applies to
    ///     both components.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits. Applies to both components.
    #[staticmethod]
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn e(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Self> {
        Ok(Self(Complex::from(scalar_constant(
            "e",
            precision,
            decimal_digits,
        )?)))
    }
    /// Construct the Euler-Mascheroni constant with precision in bits or decimal_digits (default: 53 bits). Complex results have zero imaginary part.
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits. Applies to
    ///     both components.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits. Applies to both components.
    #[staticmethod]
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn euler(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Self> {
        Ok(Self(Complex::from(scalar_constant(
            "euler",
            precision,
            decimal_digits,
        )?)))
    }
    /// Construct the golden ratio (1+sqrt(5))/2 with precision in bits or decimal_digits (default: 53 bits). Complex results have zero imaginary part.
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits. Applies to
    ///     both components.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits. Applies to both components.
    #[staticmethod]
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn phi(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Self> {
        Ok(Self(Complex::from(scalar_constant(
            "phi",
            precision,
            decimal_digits,
        )?)))
    }
    /// Alias for euler(), the Euler-Mascheroni constant; precision defaults to 53 bits.
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits. Applies to
    ///     both components.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits. Applies to both components.
    #[staticmethod]
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn euler_gamma(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Self> {
        Self::euler(precision, decimal_digits)
    }
    /// Construct the imaginary unit 0+1j with precision in bits or decimal_digits (default: 53 bits).
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits. Applies to
    ///     both components.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits. Applies to both components.
    #[staticmethod]
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn i(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Self> {
        let p = precision_bits(precision, decimal_digits)?.unwrap_or(53);
        Ok(Self(Complex::new(Float::new(p), Float::with_val(p, 1))))
    }
    /// Alias for conjugate().
    fn conj(&self) -> Self {
        Self(self.0.conj())
    }
    /// Return the additive inverse, equivalent to -self.
    fn neg(&self) -> Self {
        Self(-self.0.clone())
    }
    /// Return the magnitude as a real Float, equivalent to abs(self).
    fn norm(&self) -> PythonFloat {
        self.__abs__()
    }
    /// Return sqrt(abs(self)**2 + abs(other)**2) as a real Float, using scaled arithmetic.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Second coordinate. Native numbers use this value's precision;
    ///     existing arbitrary-precision scalars retain their precision.
    fn hypot(&self, other: &Bound<'_, PyAny>) -> PyResult<PythonFloat> {
        let rhs = self.method_operand(other)?;
        Ok(PythonFloat(self.0.norm().re.hypot(&rhs.norm().re)))
    }
    /// Return zero at this value's precision, preserving component precisions.
    fn zero(&self) -> Self {
        Self(self.0.zero())
    }
    /// Return one at this value's precision, preserving component precisions.
    fn one(&self) -> Self {
        Self(self.0.one())
    }
    /// Return NaN at this value's precision; both complex components become NaN.
    fn nan(&self) -> Self {
        Self(self.0.nan().expect("floating-point values support NaN"))
    }
    /// Return whether the value is zero; signed zero also counts as zero.
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    /// Return whether the value equals one (1+0j for ComplexFloat).
    fn is_one(&self) -> bool {
        self.0.is_one()
    }
    /// Return whether the value is exactly zero in every component.
    fn is_fully_zero(&self) -> bool {
        self.0.is_fully_zero()
    }
    /// Return the working precision in bits; alias for the precision property.
    fn get_precision(&self) -> u32 {
        self.0.get_precision()
    }
    /// Return 2**(-precision) as a native float. Very high precision can underflow to zero.
    fn get_epsilon(&self) -> f64 {
        self.0.get_epsilon()
    }
    /// Return False: arithmetic dynamically tracks precision for these scalar types.
    fn fixed_precision(&self) -> bool {
        self.0.fixed_precision()
    }
    /// Convert a nonnegative platform-sized integer at this value's precision. Out-of-range inputs raise OverflowError.
    ///
    /// Parameters
    /// ----------
    /// value : int
    ///     Integer in [0, 2**pointer_bits-1] to convert.
    fn from_usize(&self, value: usize) -> Self {
        Self(self.0.from_usize(value))
    }
    /// Convert a signed 64-bit integer at this value's precision. Out-of-range inputs raise OverflowError.
    ///
    /// Parameters
    /// ----------
    /// value : int
    ///     Integer in [-2**63, 2**63-1] to convert.
    fn from_i64(&self, value: i64) -> Self {
        Self(self.0.from_i64(value))
    }
    /// Return a new value converted from other, with precision inferred as in the constructor.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex, str, Decimal or tuple
    ///     Value to copy or convert using constructor precision inference.
    /// A tuple supplies (real, imag).
    fn set_from(&self, other: PythonMultiPrecisionComplex) -> Self {
        Self(other.0)
    }
    /// Construct zero with precision in bits or decimal_digits (default: 53 bits). Use zero() to retain instance precision.
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits. Applies to
    ///     both components.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits. Applies to both components.
    #[staticmethod]
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn new_zero(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Self> {
        let p = precision_bits(precision, decimal_digits)?.unwrap_or(53);
        Ok(Self(Complex::from(Float::with_val(p, 0))))
    }
    /// Construct one with precision in bits or decimal_digits (default: 53 bits). Use one() to retain instance precision.
    ///
    /// Parameters
    /// ----------
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits. Applies to
    ///     both components.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits. Applies to both components.
    #[staticmethod]
    #[pyo3(signature = (*, precision=None, decimal_digits=None))]
    fn new_one(precision: Option<u32>, decimal_digits: Option<u32>) -> PyResult<Self> {
        let p = precision_bits(precision, decimal_digits)?.unwrap_or(53);
        Ok(Self(Complex::from(Float::with_val(p, 1))))
    }
    /// Sample uniformly from [0, 1) using the full working precision.
    ///
    /// rng must supply getrandbits(bits); omitted or None uses Python's random
    /// module. Pass random.Random(seed) for reproducibility. Complex samples have
    /// zero imaginary part and preserve the receiver's component precisions.
    ///
    /// Parameters
    /// ----------
    /// rng : object, optional
    ///     Random generator with a getrandbits(bits) method returning an integer in
    ///     [0, 2**bits). Omitted or None uses Python's random module; use
    ///     random.Random(seed) for reproducible samples.
    #[pyo3(signature = (rng=None))]
    fn sample_unit(&self, py: Python<'_>, rng: Option<&Bound<'_, PyAny>>) -> PyResult<Self> {
        let value = sample_float(py, self.0.re.prec(), rng)?;
        Ok(Self(Complex::new(value, self.0.im.zero())))
    }
    /// Construct numerator / denominator from two Python integers.
    ///
    /// Specify precision in bits or decimal_digits, never both; the default is
    /// 53 bits. The rational is rounded directly without conversion through a
    /// native float. A zero denominator raises ZeroDivisionError.
    ///
    /// Parameters
    /// ----------
    /// numerator : int
    ///     Numerator of the rational value; accepts arbitrary-sized Python
    ///     integers.
    /// denominator : int
    ///     Nonzero denominator of the rational value; either sign is accepted.
    /// precision : int, optional
    ///     Positive working precision in bits. Mutually exclusive with
    ///     decimal_digits. If neither option is supplied, use 53 bits. Applies to
    ///     both components.
    /// decimal_digits : int, optional
    ///     Positive decimal working precision, converted to ceil(decimal_digits *
    ///     log2(10)) bits. Mutually exclusive with precision. If neither option is
    ///     supplied, use 53 bits. Applies to both components.
    ///
    /// Examples
    /// --------
    /// >>> Float.from_ratio(1, 8, precision=100).to_decimal()
    /// Decimal('0.125')
    #[staticmethod]
    #[pyo3(signature = (numerator, denominator, *, precision=None, decimal_digits=None))]
    fn from_ratio(
        numerator: &Bound<'_, PyInt>,
        denominator: &Bound<'_, PyInt>,
        precision: Option<u32>,
        decimal_digits: Option<u32>,
    ) -> PyResult<Self> {
        Ok(Self(Complex::from(
            PythonFloat::from_ratio(numerator, denominator, precision, decimal_digits)?.0,
        )))
    }
    /// Convert numerator / denominator at this value's precision. Both inputs must be Python integers; zero denominator raises ZeroDivisionError.
    ///
    /// Parameters
    /// ----------
    /// numerator : int
    ///     Numerator of the rational value; accepts arbitrary-sized Python
    ///     integers.
    /// denominator : int
    ///     Nonzero denominator of the rational value; either sign is accepted.
    fn from_rational(
        &self,
        numerator: &Bound<'_, PyInt>,
        denominator: &Bound<'_, PyInt>,
    ) -> PyResult<Self> {
        Self::from_ratio(
            numerator,
            denominator,
            Some(self.precision_property()),
            None,
        )
    }
    /// Return 1/self. Zero raises ZeroDivisionError.
    fn inv(&self) -> PyResult<Self> {
        if self.0.is_zero() {
            return Err(exceptions::PyZeroDivisionError::new_err("division by zero"));
        }
        Ok(Self(self.0.inv()))
    }
    /// Raise to an unsigned 64-bit integer exponent. Negative or out-of-range exponents raise OverflowError; use ** for signed integer powers.
    ///
    /// Parameters
    /// ----------
    /// exponent : int
    ///     Unsigned exponent in [0, 2**64-1]; zero returns one, including for a
    ///     zero base.
    fn pow(&self, exponent: u64) -> Self {
        Self(complex_integer_power(&self.0, exponent))
    }

    /// Raise to a real or complex numeric exponent on the principal branch. Zero to a negative-real or non-real exponent raises ZeroDivisionError.
    ///
    /// Parameters
    /// ----------
    /// exponent : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric exponent. Use ** for signed integer powers. Non-integer powers
    ///     use the principal complex branch.
    fn powf(&self, exponent: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.method_operand(exponent)?;
        if self.0.is_zero() && (rhs.re.is_negative() || !rhs.im.is_zero()) {
            return Err(exceptions::PyZeroDivisionError::new_err(
                "zero to a negative or complex power",
            ));
        }
        Ok(Self(self.0.powf(&rhs)))
    }
    /// Return atan(self/x) for complex arguments; two real arguments use the usual quadrant-aware atan2. A zero complex denominator raises ZeroDivisionError.
    ///
    /// Parameters
    /// ----------
    /// x : Float, ComplexFloat, int, float, complex or Decimal
    ///     Horizontal coordinate; self is the vertical coordinate in atan2(self,
    ///     x). For non-real arguments, this is the divisor in atan(self/x).
    fn atan2(&self, x: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rhs = self.method_operand(x)?;
        if self.0.im.is_zero() && rhs.im.is_zero() {
            return Ok(Self(Complex::new(
                self.0.re.atan2(&rhs.re),
                self.0.im.zero(),
            )));
        }
        if rhs.is_zero() {
            return Err(exceptions::PyZeroDivisionError::new_err(
                "complex atan2 denominator is zero",
            ));
        }
        Ok(Self(self.0.atan2(&rhs)))
    }
    /// Return self*a+b with accuracy tracking, rounding the multiplication and addition separately.
    ///
    /// Parameters
    /// ----------
    /// a : Float, ComplexFloat, int, float, complex or Decimal
    ///     Multiplier in self*a+b.
    /// b : Float, ComplexFloat, int, float, complex or Decimal
    ///     Addend in self*a+b.
    fn mul_add(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Self> {
        Ok(Self(self.0.mul_add(
            &self.method_operand(a)?,
            &self.method_operand(b)?,
        )))
    }
    /// Return self + other with accuracy tracking; complex operands produce ComplexFloat.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __add__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '+', false)
    }
    /// Return other + self with accuracy tracking; complex operands produce ComplexFloat.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __radd__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '+', true)
    }
    /// Return self - other with accuracy tracking; complex operands produce ComplexFloat.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __sub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '-', false)
    }
    /// Return other - self with accuracy tracking; complex operands produce ComplexFloat.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __rsub__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '-', true)
    }
    /// Return self * other with accuracy tracking; complex operands produce ComplexFloat.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __mul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '*', false)
    }
    /// Return other * self with accuracy tracking; complex operands produce ComplexFloat.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __rmul__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '*', true)
    }
    /// Return self / other with accuracy tracking; complex operands produce ComplexFloat. A zero divisor raises ZeroDivisionError.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __truediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '/', false)
    }
    /// Return other / self with accuracy tracking; complex operands produce ComplexFloat. A zero divisor raises ZeroDivisionError.
    ///
    /// Parameters
    /// ----------
    /// other : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric operand. Existing arbitrary-precision scalars retain their
    ///     precision; native numbers are converted at the receiver's precision.
    fn __rtruediv__(&self, py: Python<'_>, other: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
        self.binary(py, other, '/', true)
    }
    /// Raise to an integer, real or complex numeric exponent. Integer powers support negative exponents; other powers use the principal branch. Modular powers are unsupported.
    ///
    /// Parameters
    /// ----------
    /// exponent : Float, ComplexFloat, int, float, complex or Decimal
    ///     Numeric exponent. Use ** for signed integer powers. Non-integer powers
    ///     use the principal complex branch.
    /// modulo : None, optional
    ///     Must be None. Three-argument modular exponentiation is unsupported.
    fn __pow__(
        &self,
        py: Python<'_>,
        exponent: &Bound<'_, PyAny>,
        modulo: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        if modulo.is_some() {
            return Ok(py.NotImplemented());
        }
        if exponent.is_instance_of::<PyInt>() {
            if let Ok(e) = exponent.extract::<i64>() {
                if e < 0 && self.0.is_zero() {
                    return Err(exceptions::PyZeroDivisionError::new_err(
                        "zero to a negative power",
                    ));
                }
                let value = complex_integer_power(&self.0, e.unsigned_abs());
                return Self(if e < 0 { value.inv() } else { value }).into_py_any(py);
            }
        }
        match self.powf(exponent) {
            Ok(value) => value.into_py_any(py),
            Err(e) if e.is_instance_of::<exceptions::PyTypeError>(py) => Ok(py.NotImplemented()),
            Err(e) => Err(e),
        }
    }
}
impl PythonComplexFloat {
    // Signed-zero limits on the branch cuts. The generic Complex formulas can
    // cancel the sign of zero in intermediate expressions, so handle the axes
    // explicitly using real functions at the original component precision.
    fn axis_value(&self, name: &str) -> Option<Complex<Float>> {
        let z = &self.0;
        if !z.is_finite() {
            return None;
        }
        let signed = |v: Float, sign: &Float| if sign.is_sign_negative() { -v } else { v };
        let half_pi = || z.re.pi() / z.re.from_i64(2);
        let one = z.re.one();
        let a = z.re.norm();
        let b = z.im.norm();
        if z.im.is_zero() {
            match name {
                "asinh" => return Some(Complex::new(z.re.asinh(), z.im.clone())),
                "atan" => return Some(Complex::new(z.re.atan2(&one), z.im.clone())),
                "atanh" if a <= one => return Some(Complex::new(z.re.atanh(), z.im.clone())),
                "asin" if a <= one => return Some(Complex::new(z.re.asin(), z.im.clone())),
                "asin" => {
                    return Some(Complex::new(
                        signed(half_pi(), &z.re),
                        signed(a.acosh(), &z.im),
                    ));
                }
                "acos" if a <= one => return Some(Complex::new(z.re.acos(), -z.im.clone())),
                "acos" => {
                    return Some(Complex::new(
                        if z.re.is_negative() {
                            z.re.pi()
                        } else {
                            z.re.zero()
                        },
                        -signed(a.acosh(), &z.im),
                    ));
                }
                "acosh" if z.re >= one => return Some(Complex::new(z.re.acosh(), z.im.clone())),
                "acosh" if a <= one => {
                    return Some(Complex::new(z.re.zero(), signed(z.re.acos(), &z.im)));
                }
                "acosh" => return Some(Complex::new(a.acosh(), signed(z.re.pi(), &z.im))),
                "atanh" if a > one => {
                    return Some(Complex::new(
                        signed(a.inv().atanh(), &z.re),
                        signed(half_pi(), &z.im),
                    ));
                }
                _ => (),
            }
        }
        if z.re.is_zero() {
            match name {
                "asinh" if b <= one => return Some(Complex::new(z.re.clone(), z.im.asin())),
                "asinh" => {
                    return Some(Complex::new(
                        signed(b.acosh(), &z.re),
                        signed(half_pi(), &z.im),
                    ));
                }
                "atan" if b <= one => return Some(Complex::new(z.re.clone(), z.im.atanh())),
                "atan" if b > one => {
                    return Some(Complex::new(
                        signed(half_pi(), &z.re),
                        signed(b.inv().atanh(), &z.im),
                    ));
                }
                _ => (),
            }
        }
        None
    }
    fn method_operand(&self, value: &Bound<'_, PyAny>) -> PyResult<Complex<Float>> {
        if value.is_instance_of::<PyString>() || value.is_instance_of::<PyTuple>() {
            return Err(exceptions::PyTypeError::new_err(
                "Expected a numeric operand",
            ));
        }
        let p = if value.is_instance_of::<PythonFloat>() || value.is_instance_of::<Self>() {
            None
        } else {
            Some(self.precision_property())
        };
        complex_input(value, p)
    }
    fn binary(
        &self,
        py: Python<'_>,
        other: &Bound<'_, PyAny>,
        op: char,
        reverse: bool,
    ) -> PyResult<Py<PyAny>> {
        if other.is_instance_of::<PyString>() || other.is_instance_of::<PyTuple>() {
            return Ok(py.NotImplemented());
        }
        let p = if other.is_instance_of::<PythonFloat>() || other.is_instance_of::<Self>() {
            None
        } else {
            Some(self.precision_property())
        };
        let rhs = match complex_input(other, p) {
            Ok(v) => v,
            Err(e) if e.is_instance_of::<exceptions::PyTypeError>(py) => {
                return Ok(py.NotImplemented());
            }
            Err(e) => return Err(e),
        };
        let (a, b) = if reverse {
            (rhs, self.0.clone())
        } else {
            (self.0.clone(), rhs)
        };
        if op == '/' && b.is_zero() {
            return Err(exceptions::PyZeroDivisionError::new_err("division by zero"));
        }
        Self(match op {
            '+' => a + b,
            '-' => a - b,
            '*' => a * b,
            '/' => a / b,
            _ => unreachable!(),
        })
        .into_py_any(py)
    }
}

/// Input adapter accepting Python numeric values; output is always a Python Float.
pub struct PythonMultiPrecisionFloat(pub Float);
impl From<Float> for PythonMultiPrecisionFloat {
    fn from(value: Float) -> Self {
        Self(value)
    }
}
impl<'py> IntoPyObject<'py> for PythonMultiPrecisionFloat {
    type Target = PythonFloat;
    type Output = Bound<'py, PythonFloat>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Bound::new(py, PythonFloat(self.0))
    }
}
impl<'py> FromPyObject<'_, 'py> for PythonMultiPrecisionFloat {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        real_input(&ob, None).map(Self)
    }
}
/// Complex input adapter; accepts complex scalars or real/imaginary pairs.
/// Output is always a Python ComplexFloat.
pub struct PythonMultiPrecisionComplex(pub Complex<Float>);
impl From<Complex<Float>> for PythonMultiPrecisionComplex {
    fn from(value: Complex<Float>) -> Self {
        Self(value)
    }
}
impl<'py> IntoPyObject<'py> for PythonMultiPrecisionComplex {
    type Target = PythonComplexFloat;
    type Output = Bound<'py, PythonComplexFloat>;
    type Error = PyErr;
    fn into_pyobject(self, py: Python<'py>) -> PyResult<Self::Output> {
        Bound::new(py, PythonComplexFloat(self.0))
    }
}
impl<'py> FromPyObject<'_, 'py> for PythonMultiPrecisionComplex {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        complex_input(&ob, None).map(Self)
    }
}
impl<'py> FromPyObject<'_, 'py> for Complex<f64> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(value) = ob.extract::<PyRef<'_, PythonComplexFloat>>() {
            return Ok(Self::new(value.0.re.to_f64(), value.0.im.to_f64()));
        }
        ob.extract::<Complex64>().map(|x| Self::new(x.re, x.im))
    }
}
impl<'py> FromPyObject<'_, 'py> for Complex<Float> {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        complex_input(&ob, None)
    }
}
#[cfg(feature = "python_stubgen")]
impl PyStubType for PythonMultiPrecisionFloat {
    fn type_output() -> TypeInfo {
        PythonFloat::type_output()
    }
    fn type_input() -> TypeInfo {
        PythonFloat::type_output()
            | <i64>::type_output()
            | <f64>::type_output()
            | <String>::type_output()
            | TypeInfo::with_module("decimal.Decimal", "decimal".into())
    }
}
#[cfg(feature = "python_stubgen")]
impl PyStubType for PythonMultiPrecisionComplex {
    fn type_output() -> TypeInfo {
        PythonComplexFloat::type_output()
    }
    fn type_input() -> TypeInfo {
        PythonComplexFloat::type_output()
            | PythonMultiPrecisionFloat::type_input()
            | <Complex64>::type_output()
            | <(PythonMultiPrecisionFloat, PythonMultiPrecisionFloat)>::type_input()
    }
}
#[cfg(feature = "python_stubgen")]
impl_stub_type!(Complex<f64> = Complex64);
#[cfg(feature = "python_stubgen")]
impl_stub_type!(Complex<Float> = PythonMultiPrecisionComplex);
