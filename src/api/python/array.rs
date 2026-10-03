//! Optional runtime NumPy integration for evaluation inputs and outputs.

use numpy::{
    AllowTypeChange, Element, IntoPyArray, PyArrayLike,
    ndarray::{Array, ArrayD, ArrayView, Axis, Dimension, IxDyn},
};
use pyo3::{
    Borrowed, Bound, FromPyObject, IntoPyObject, IntoPyObjectExt, PyAny, PyErr, PyResult, Python,
    exceptions::{PyModuleNotFoundError, PyValueError},
    types::{PyAnyMethods, PyModule},
};

pub(super) enum EvaluationInput<'py, T: Element, D: Dimension> {
    Numpy(PyArrayLike<'py, T, D, AllowTypeChange>),
    Sequence(Array<T, D>),
}

impl<'a, 'py, T, D> FromPyObject<'a, 'py> for EvaluationInput<'py, T, D>
where
    T: Element + 'py,
    D: Dimension + 'py,
    for<'b> T: FromPyObject<'b, 'py>,
{
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        // rust-numpy initializes its C API on first use and panics if NumPy is
        // missing. Import it through Python before touching any NumPy types.
        match PyModule::import(ob.py(), "numpy") {
            Ok(_) => return ob.extract().map(Self::Numpy),
            Err(error)
                if error.is_instance_of::<PyModuleNotFoundError>(ob.py())
                    && error.value(ob.py()).getattr("name")?.extract::<String>()? == "numpy" => {}
            Err(error) => return Err(error),
        }

        let array = if let Ok(values) = ob.extract::<Vec<T>>() {
            Array::from_vec(values).into_dyn()
        } else {
            let rows = ob.extract::<Vec<Vec<T>>>()?;
            let columns = rows.first().map_or(0, Vec::len);
            if rows.iter().any(|row| row.len() != columns) {
                return Err(PyValueError::new_err("Input rows must have equal lengths"));
            }
            ArrayD::from_shape_vec(
                IxDyn(&[rows.len(), columns]),
                rows.into_iter().flatten().collect(),
            )
            .map_err(|error| PyValueError::new_err(error.to_string()))?
        };

        array
            .into_dimensionality()
            .map(Self::Sequence)
            .map_err(|error| PyValueError::new_err(error.to_string()))
    }
}

impl<'py, T: Element, D: Dimension> EvaluationInput<'py, T, D> {
    pub(super) fn as_array(&self) -> ArrayView<'_, T, D> {
        match self {
            Self::Numpy(array) => array.as_array(),
            Self::Sequence(array) => array.view(),
        }
    }

    pub(super) fn as_slice(&self) -> PyResult<&[T]> {
        match self {
            Self::Numpy(array) => array
                .as_slice()
                .map_err(|error| PyValueError::new_err(error.to_string())),
            Self::Sequence(array) => array
                .as_slice()
                .ok_or_else(|| PyValueError::new_err("Input is not contiguous")),
        }
    }

    /// Keep the existing ndarray result when NumPy is installed. Otherwise,
    /// return one Python list per batch entry, including empty output rows.
    pub(super) fn output(&self, array: ArrayD<T>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>>
    where
        T: IntoPyObject<'py> + Clone,
    {
        match self {
            Self::Numpy(_) => Ok(array.into_pyarray(py).into_any()),
            Self::Sequence(_) => array
                .axis_iter(Axis(0))
                .map(|row| row.iter().cloned().collect::<Vec<_>>())
                .collect::<Vec<_>>()
                .into_bound_py_any(py),
        }
    }
}
