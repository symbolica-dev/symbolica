//! Python access to wide error magnitudes and uncertainty-preserving evaluation.
use super::*;
use crate::domains::float::{ErrorPropagatingFloat, Magnitude};

pub(super) type TrackedComplex = Complex<ErrorPropagatingFloat<Float>>;

/// A nonnegative error magnitude with 32 significant bits and a wide binary exponent.
/// Arithmetic uncertainty is an estimate, not a certified interval enclosure.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(from_py_object, frozen, name = "Magnitude", module = "symbolica")]
#[derive(Clone, Copy)]
pub struct PythonMagnitude(pub Magnitude);

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonMagnitude {
    /// Construct abs(value) * 2**exponent, rounding upward.
    #[new]
    #[pyo3(signature = (value = 0.0, exponent = 0))]
    fn new(value: f64, exponent: i64) -> Self {
        Self(Magnitude::from_f64_exp(value, exponent))
    }

    /// Reconstruct canonical parts returned by significand and exponent.
    #[staticmethod]
    fn from_parts(significand: u32, exponent: i32) -> PyResult<Self> {
        let mut bytes = [0; 8];
        bytes[..4].copy_from_slice(&significand.to_le_bytes());
        bytes[4..].copy_from_slice(&exponent.to_le_bytes());
        Magnitude::from_le_bytes(bytes)
            .map(Self)
            .map_err(exceptions::PyValueError::new_err)
    }

    /// Finite nonzero magnitudes equal significand * 2**(exponent - 31).
    #[getter]
    fn significand(&self) -> u32 {
        self.0.significand()
    }
    #[getter]
    fn exponent(&self) -> i32 {
        self.0.exponent()
    }

    /// Convert upward to float; return None for finite values outside its range.
    /// Explicit infinity and NaN return their corresponding float values.
    fn to_float(&self) -> Option<f64> {
        self.0.to_f64()
    }
    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
    fn is_finite(&self) -> bool {
        self.0.is_finite()
    }
    fn is_nan(&self) -> bool {
        self.0.is_nan()
    }
    fn is_infinite(&self) -> bool {
        self.0.is_infinite()
    }
    fn log10(&self) -> f64 {
        self.0.log10()
    }

    fn __float__(&self) -> PyResult<f64> {
        self.to_float().ok_or_else(|| {
            exceptions::PyOverflowError::new_err(
                "Magnitude is outside the float range; use its significand and exponent",
            )
        })
    }
    fn __repr__(&self) -> String {
        format!(
            "Magnitude.from_parts({}, {})",
            self.significand(),
            self.exponent()
        )
    }
    fn __richcmp__(&self, other: &Self, op: CompareOp) -> bool {
        match op {
            CompareOp::Eq => self.0 == other.0,
            CompareOp::Ne => self.0 != other.0,
            CompareOp::Lt => self.0 < other.0,
            CompareOp::Le => self.0 <= other.0,
            CompareOp::Gt => self.0 > other.0,
            CompareOp::Ge => self.0 >= other.0,
        }
    }
    fn __reduce__(slf: PyRef<'_, Self>) -> PyResult<(Py<PyAny>, (u32, i32))> {
        let constructor = PythonMagnitude::type_object(slf.py())
            .getattr("from_parts")?
            .unbind();
        Ok((constructor, (slf.significand(), slf.exponent())))
    }
}

/// A complex center and separate absolute error estimates for its components.
/// Pass this object to evaluate_with_error to retain its uncertainty. The value
/// property explicitly extracts the ordinary, untracked ComplexFloat center.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    frozen,
    name = "TrackedComplexFloat",
    module = "symbolica"
)]
#[derive(Clone)]
pub struct PythonTrackedComplexFloat(pub(super) TrackedComplex);

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonTrackedComplexFloat {
    /// Supply explicit absolute errors, or infer roundoff from the input precision.
    /// A zero imaginary component defaults to exact zero, preserving real inputs.
    #[new]
    #[pyo3(signature = (value, real_error = None, imaginary_error = None))]
    fn new(
        value: PythonMultiPrecisionComplex,
        real_error: Option<PythonMagnitude>,
        imaginary_error: Option<PythonMagnitude>,
    ) -> Self {
        let re = match real_error {
            Some(e) => ErrorPropagatingFloat::new_with_error_magnitude(value.0.re, e.0),
            None => ErrorPropagatingFloat::from(value.0.re),
        };
        let im = match imaginary_error {
            Some(e) => ErrorPropagatingFloat::new_with_error_magnitude(value.0.im, e.0),
            None if value.0.im.is_zero() => ErrorPropagatingFloat::exact_zero(value.0.im.prec()),
            None => ErrorPropagatingFloat::from(value.0.im),
        };
        Self(Complex::new(re, im))
    }
    /// Read a floating numerical coefficient with all its stored metadata.
    #[staticmethod]
    fn from_expression(expression: PythonExpression) -> PyResult<Self> {
        let AtomView::Num(n) = expression.expr.as_view() else {
            return Err(exceptions::PyTypeError::new_err(
                "Expected a floating numerical coefficient",
            ));
        };
        match n.get_coeff_view() {
            CoefficientView::Float(re, im) => Ok(Self(Complex::new(re.to_float(), im.to_float()))),
            _ => Err(exceptions::PyTypeError::new_err(
                "Expected a floating numerical coefficient",
            )),
        }
    }
    #[getter]
    fn value(&self) -> PythonComplexFloat {
        PythonComplexFloat(Complex::new(
            self.0.re.get_num().clone(),
            self.0.im.get_num().clone(),
        ))
    }
    #[getter]
    fn real_error(&self) -> PythonMagnitude {
        PythonMagnitude(self.0.re.get_error_magnitude())
    }
    #[getter]
    fn imaginary_error(&self) -> PythonMagnitude {
        PythonMagnitude(self.0.im.get_error_magnitude())
    }
    /// Working precision in binary bits, separately for real and imaginary components.
    #[getter]
    fn precision(&self) -> (u32, u32) {
        (self.0.re.get_num().prec(), self.0.im.get_num().prec())
    }
    /// Estimated relative decimal digits; None for a zero center.
    #[getter]
    fn decimal_precision(&self) -> (Option<f64>, Option<f64>) {
        (self.0.re.get_precision(), self.0.im.get_precision())
    }
    /// Estimated absolute decimal accuracy, separately for both components.
    #[getter]
    fn accuracy(&self) -> (f64, f64) {
        (self.0.re.get_accuracy(), self.0.im.get_accuracy())
    }
    fn is_real(&self) -> bool {
        self.0.im.is_exact_zero()
    }
    fn is_exact(&self) -> bool {
        self.0.re.is_exact() && self.0.im.is_exact()
    }
    /// Create a numerical expression without losing precision or uncertainty.
    fn to_expression(&self) -> PythonExpression {
        Atom::num(self.0.clone()).into()
    }
    fn __complex__(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        PyComplex::from_doubles(
            py,
            self.0.re.get_num().to_f64(),
            self.0.im.get_num().to_f64(),
        )
        .into_py_any(py)
    }
    fn __repr__(&self) -> String {
        let expression = self.to_expression();
        let text = expression.expr.printer(PrintOptions::file()).to_string();
        format!("TrackedComplexFloat.from_expression(E({text:?}))")
    }
    fn __reduce__(slf: PyRef<'_, Self>) -> PyResult<(Py<PyAny>, (PythonExpression,))> {
        let constructor = PythonTrackedComplexFloat::type_object(slf.py())
            .getattr("from_expression")?
            .unbind();
        Ok((constructor, (slf.to_expression(),)))
    }
}

pub struct PythonTrackedInput(Atom);

impl<'py> FromPyObject<'_, 'py> for PythonTrackedInput {
    type Error = PyErr;
    fn extract(ob: Borrowed<'_, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(value) = ob.extract::<PythonTrackedComplexFloat>() {
            return Ok(Self(Atom::num(value.0)));
        }
        let expression = if let Ok(expression) = ob.extract::<PythonExpression>() {
            expression
        } else {
            PythonExpression::num(
                &PythonExpression::type_object(ob.py()),
                ob.py(),
                ob.to_owned().unbind(),
                None,
            )?
        };
        if !matches!(expression.expr.as_view(), AtomView::Num(_)) {
            return Err(exceptions::PyTypeError::new_err(
                "Expected a numerical expression or a numerical scalar",
            ));
        }
        Ok(Self(expression.expr))
    }
}

#[cfg(feature = "python_stubgen")]
impl PyStubType for PythonTrackedInput {
    fn type_output() -> TypeInfo {
        TypeInfo::unqualified(
            "TrackedComplexFloat | Expression | Float | ComplexFloat | int | float | complex | str",
        )
    }
}

impl PythonTrackedInput {
    pub(super) fn to_tracked(&self, prec: u32) -> PyResult<TrackedComplex> {
        let AtomView::Num(n) = self.0.as_view() else {
            unreachable!()
        };
        n.get_coeff_view()
            .to_tracked_float(prec)
            .map_err(exceptions::PyValueError::new_err)
    }
}

pub(super) fn evaluate_with_error(
    expression: &PythonExpression,
    constants: Option<HashMap<PythonExpression, PythonTrackedInput>>,
    decimal_digit_precision: Option<u32>,
) -> PyResult<PythonTrackedComplexFloat> {
    let prec = decimal_digit_precision
        .map(|digits| Float::decimal_digits_to_bits(digits as f64))
        .transpose()
        .map_err(exceptions::PyValueError::new_err)?
        .unwrap_or(53);
    let constants = constants.unwrap_or_default();
    let constants: HashMap<AtomView, TrackedComplex> = constants
        .iter()
        .map(|(k, v)| Ok((k.expr.as_view(), v.to_tracked(prec)?)))
        .collect::<PyResult<_>>()?;
    expression
        .expr
        .evaluate_with_prec(&constants, prec)
        .map(PythonTrackedComplexFloat)
        .map_err(|e| {
            exceptions::PyValueError::new_err(format!("Could not evaluate with uncertainty: {e}"))
        })
}

pub(super) fn coefficient_error(
    expression: &PythonExpression,
) -> PyResult<(PythonMagnitude, PythonMagnitude)> {
    let AtomView::Num(n) = expression.expr.as_view() else {
        return Err(exceptions::PyTypeError::new_err(
            "get_error requires a numerical coefficient; use evaluate_with_error for an expression",
        ));
    };
    match n.get_coeff_view() {
        CoefficientView::Float(re, im) => Ok((
            PythonMagnitude(re.to_float().get_error_magnitude()),
            PythonMagnitude(im.to_float().get_error_magnitude()),
        )),
        CoefficientView::Natural(..) | CoefficientView::Large(..) => Ok((
            PythonMagnitude(Magnitude::ZERO),
            PythonMagnitude(Magnitude::ZERO),
        )),
        _ => Err(exceptions::PyTypeError::new_err(
            "This coefficient has no floating error estimate",
        )),
    }
}
