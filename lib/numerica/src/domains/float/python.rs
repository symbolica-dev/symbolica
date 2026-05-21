use rug::Float as MultiPrecisionFloat;

use super::{Complex, Float};

#[cfg(feature = "python")]
use numpy::Complex64;
#[cfg(feature = "python")]
use pyo3::{
    Borrowed, Bound, FromPyObject, IntoPyObject, Py, PyErr, PyResult, Python, exceptions,
    pybacked::PyBackedStr,
    sync::PyOnceLock,
    types::{PyAny, PyAnyMethods, PyComplex, PyComplexMethods, PyType},
};
#[cfg(feature = "python_stubgen")]
use pyo3_stub_gen::{PyStubType, TypeInfo, impl_stub_type};

#[cfg(feature = "python")]
/// A multi-precision floating point number for Python.
pub struct PythonMultiPrecisionFloat(pub Float);

#[cfg(feature = "python_stubgen")]
impl_stub_type!(PythonMultiPrecisionFloat = f64 | Decimal);

#[cfg(feature = "python_stubgen")]
pub struct Decimal;

#[cfg(feature = "python_stubgen")]
impl PyStubType for Decimal {
    fn type_output() -> TypeInfo {
        TypeInfo::with_module("decimal.Decimal", "decimal".into())
    }
}

#[cfg(feature = "python")]
impl From<Float> for PythonMultiPrecisionFloat {
    fn from(f: Float) -> Self {
        PythonMultiPrecisionFloat(f)
    }
}

#[cfg(feature = "python")]
static PYDECIMAL: PyOnceLock<Py<PyType>> = PyOnceLock::new();

#[cfg(feature = "python")]
fn get_decimal(py: Python<'_>) -> &Py<PyType> {
    PYDECIMAL.get_or_init(py, || {
        py.import("decimal")
            .unwrap()
            .getattr("Decimal")
            .unwrap()
            .extract()
            .unwrap()
    })
}

#[cfg(feature = "python")]
impl<'py> IntoPyObject<'py> for PythonMultiPrecisionFloat {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = std::convert::Infallible;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        get_decimal(py)
            .call1(py, (self.0.to_string(),))
            .expect("failed to call decimal.Decimal(value)")
            .into_pyobject(py)
    }
}

#[cfg(feature = "python")]
impl<'py> FromPyObject<'_, 'py> for PythonMultiPrecisionFloat {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if ob.is_instance(get_decimal(ob.py()).as_any().bind(ob.py()))? {
            let a = ob
                .call_method0("__str__")
                .unwrap()
                .extract::<PyBackedStr>()?;

            if a == "NaN" {
                return Ok(Float::from(MultiPrecisionFloat::with_val(
                    53,
                    rug::float::Special::Nan,
                ))
                .into());
            } else if a == "Infinity" {
                return Ok(Float::from(MultiPrecisionFloat::with_val(
                    53,
                    rug::float::Special::Infinity,
                ))
                .into());
            } else if a == "-Infinity" {
                return Ok(Float::from(MultiPrecisionFloat::with_val(
                    53,
                    rug::float::Special::NegInfinity,
                ))
                .into());
            }

            // get the number of accurate digits
            let mut digits = a
                .chars()
                .skip_while(|x| *x == '.' || *x == '0' || *x == '-')
                .filter(|x| *x != '.')
                .take_while(|x| x.is_ascii_digit())
                .count();

            // the input is 0, determine accuracy
            if digits == 0 {
                if let Some((_pre, exp)) = a.split_once('E') {
                    if let Ok(exp) = exp.parse::<isize>() {
                        digits = exp.unsigned_abs();
                    }
                } else {
                    digits = a
                        .chars()
                        .filter(|x| *x != '.' && *x != '-')
                        .take_while(|x| x.is_ascii_digit())
                        .count()
                }

                if digits == 0 {
                    return Err(exceptions::PyValueError::new_err(format!(
                        "Could not parse {a}",
                    )));
                }
            }

            Ok(Float::parse(
                &a,
                Some((digits as f64 * std::f64::consts::LOG2_10).ceil() as u32),
            )
            .map_err(|_| {
                exceptions::PyValueError::new_err(format!("Not a floating point number: {a}"))
            })?
            .into())
        } else if let Ok(a) = ob.extract::<PyBackedStr>() {
            Ok(Float::parse(&a, None)
                .map_err(|_| exceptions::PyValueError::new_err("Not a floating point number"))?
                .into())
        } else if let Ok(a) = ob.extract::<f64>() {
            if a.is_finite() {
                Ok(Float::with_val(53, a).into())
            } else {
                Err(exceptions::PyValueError::new_err(
                    "Floating point number is not finite",
                ))
            }
        } else {
            Err(exceptions::PyValueError::new_err(
                "Not a valid multi-precision float",
            ))
        }
    }
}

#[cfg(feature = "python")]
impl<'py> FromPyObject<'_, 'py> for Complex<f64> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        ob.extract::<Complex64>().map(|x| Complex::new(x.re, x.im))
    }
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(Complex<f64> = Complex64);

#[cfg(feature = "python")]
impl<'py> FromPyObject<'_, 'py> for Complex<Float> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<PythonMultiPrecisionFloat>() {
            let zero = Float::new(a.0.prec());
            Ok(Complex::new(a.0, zero))
        } else if let Ok(a) = ob.cast::<PyComplex>() {
            Ok(Complex::new(
                Float::with_val(53, a.real()),
                Float::with_val(53, a.imag()),
            ))
        } else {
            Err(exceptions::PyValueError::new_err(
                "Not a valid complex number",
            ))
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(Complex<Float> = Complex64);
