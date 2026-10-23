//! Typed extension points shared by a host and its optional implementations.
//!
//! A component declares an [`ExtensionPoint`]; the containing native extension
//! registers an [`ExtensionImplementation`] during initialization. Neither side
//! needs to import the other's Python module. Keep participating components in
//! one native library so they share the slot and their Rust/PyO3 types.
//!

use std::{error::Error, fmt, sync::OnceLock};

use pyo3::{
    Bound, BoundObject, IntoPyObject, IntoPyObjectExt, Py, PyAny, PyErr, PyResult, Python,
    exceptions::{PyImportError, PyRuntimeError},
    types::{PyDict, PyTuple},
};

/// A named implementation supplied by a component linked into the native host.
///
/// Declare one `static` descriptor per implementation and pass its reference to
/// every registration call. `const` descriptors and promoted temporaries do not
/// provide stable identity. The provider name is diagnostic, not an identity key.
/// Do not store interpreter-owned Python objects in a global descriptor.
pub struct ExtensionImplementation<T> {
    provider: &'static str,
    implementation: T,
}

impl<T> ExtensionImplementation<T> {
    pub const fn new(provider: &'static str, implementation: T) -> Self {
        Self {
            provider,
            implementation,
        }
    }
}

/// A typed operation whose implementation is installed once by the native host.
///
/// Registration is thread-safe. The same static descriptor can be registered
/// repeatedly; a competing descriptor is rejected without replacing the first.
/// Provider names keep missing and conflicting registrations understandable.
///
/// Implementations must be declared as actual `static` items, not `const` items
/// or promoted temporary references. Registration compares descriptor identity,
/// not function addresses or provider names. Registering the same static again
/// succeeds; a different static conflicts even if its provider name is equal.
///
/// # Typed Rust functions
///
/// ```
/// use symbolica::api::python::{ExtensionImplementation, ExtensionPoint};
///
/// static REDUCE: ExtensionPoint<fn(i64) -> i64> = ExtensionPoint::new("reduce");
/// static IMPLEMENTATION: ExtensionImplementation<fn(i64) -> i64> =
///     ExtensionImplementation::new("example reducer", |value| value / 2);
///
/// REDUCE.register(&IMPLEMENTATION)?;
/// assert_eq!(REDUCE.get()?(8), 4);
/// # Ok::<(), symbolica::api::python::ExtensionError>(())
/// ```
///
/// A slot can also contain a struct of function pointers, installing the entire
/// interface atomically. The scientific implementation retains its typed Rust
/// inputs, return values and errors.
///
/// # Methods on an existing class
///
/// The class's owning crate defines the Python method, its public signature and
/// documentation. Its thin `#[pymethods]` forwarder resolves the implementation:
///
/// ```
/// use pyo3::prelude::*;
/// use symbolica::api::python::{ExtensionImplementation, ExtensionPoint};
///
/// #[pyclass]
/// struct Diagram;
/// #[pyclass]
/// struct Reduced;
///
/// type Reduce = fn(&Diagram) -> PyResult<Reduced>;
/// static REDUCE: ExtensionPoint<Reduce> = ExtensionPoint::new("Diagram.tensor_reduce");
///
/// #[pymethods]
/// impl Diagram {
///     /// Reduce this diagram's tensor expression using the registered backend.
///     fn tensor_reduce(&self) -> PyResult<Reduced> {
///         REDUCE.get()?(self)
///     }
/// }
///
/// // The provider depends on the owning crate and implements its typed interface.
/// static BACKEND: ExtensionImplementation<Reduce> =
///     ExtensionImplementation::new("example reducer", |_diagram| Ok(Reduced));
/// REDUCE.register(&BACKEND)?;
/// assert!(Diagram.tensor_reduce().is_ok());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
///
/// The owner also generates or maintains the method's `.pyi` entry, including
/// argument/return types and documentation. Registration only fills the callback
/// slot; it does not install methods on a class or generate Python stubs. With a
/// variadic binding, the owner can instead forward through [`ExtensionPoint::call`]
/// while documenting its full keyword interface in the stub.
///
/// # Forwarding to a PyO3 binding
///
/// Reuse the backend's `#[pyfunction]` parser instead of copying its argument
/// conversion or defaults. This adapter resolves the function at compile time,
/// creates its Python callable for the current interpreter, and imports no module:
///
/// ```
/// use pyo3::{prelude::*, types::{PyDict, PyTuple}};
/// use symbolica::api::python::{
///     ExtensionImplementation, ExtensionPoint, PythonExtensionFunction,
/// };
///
/// #[pyfunction]
/// #[pyo3(signature = (value, *, divisor = 2))]
/// fn reduce(value: i64, divisor: i64) -> PyResult<i64> {
///     value.checked_div(divisor).ok_or_else(||
///         pyo3::exceptions::PyValueError::new_err("invalid division"))
/// }
///
/// fn forward(
///     args: &Bound<'_, PyTuple>,
///     kwargs: Option<&Bound<'_, PyDict>>,
/// ) -> PyResult<Py<PyAny>> {
///     Ok(wrap_pyfunction!(reduce, args.py())?.call(args, kwargs)?.unbind())
/// }
///
/// static REDUCE: ExtensionPoint<PythonExtensionFunction> = ExtensionPoint::new("reduce");
/// static IMPLEMENTATION: ExtensionImplementation<PythonExtensionFunction> =
///     ExtensionImplementation::new("example reducer", forward);
///
/// REDUCE.register(&IMPLEMENTATION)?;
/// Python::initialize();
/// Python::attach(|py| -> PyResult<()> {
///     assert_eq!(REDUCE.call(py, (8,), None)?.extract::<i64>(py)?, 4);
///     Ok(())
/// })?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct ExtensionPoint<T: 'static> {
    operation: &'static str,
    implementation: OnceLock<&'static ExtensionImplementation<T>>,
}

impl<T> ExtensionPoint<T> {
    pub const fn new(operation: &'static str) -> Self {
        Self {
            operation,
            implementation: OnceLock::new(),
        }
    }

    /// Register an actual `static` implementation descriptor.
    pub fn register(
        &self,
        implementation: &'static ExtensionImplementation<T>,
    ) -> Result<(), ExtensionError> {
        let registered = self.implementation.get_or_init(|| implementation);
        if std::ptr::eq(*registered, implementation) {
            Ok(())
        } else {
            Err(ExtensionError::ConflictingImplementation {
                operation: self.operation,
                registered_provider: registered.provider,
                attempted_provider: implementation.provider,
            })
        }
    }

    /// Resolve the implementation without importing or initializing a module.
    pub fn get(&self) -> Result<&'static T, ExtensionError> {
        let implementation =
            self.implementation
                .get()
                .copied()
                .ok_or(ExtensionError::MissingImplementation {
                    operation: self.operation,
                })?;
        Ok(&implementation.implementation)
    }

    /// Return the registered provider's diagnostic name, if any.
    pub fn provider(&self) -> Option<&'static str> {
        self.implementation
            .get()
            .map(|implementation| implementation.provider)
    }
}

/// An extension point could not be resolved or a different provider was installed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtensionError {
    MissingImplementation {
        operation: &'static str,
    },
    ConflictingImplementation {
        operation: &'static str,
        registered_provider: &'static str,
        attempted_provider: &'static str,
    },
}

impl fmt::Display for ExtensionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingImplementation { operation } => write!(
                formatter,
                "No implementation is registered for '{operation}'; initialize a host that provides it"
            ),
            Self::ConflictingImplementation {
                operation,
                registered_provider,
                attempted_provider,
            } => write!(
                formatter,
                "Cannot register '{attempted_provider}' for '{operation}': '{registered_provider}' is already registered"
            ),
        }
    }
}

impl Error for ExtensionError {}

impl From<ExtensionError> for PyErr {
    fn from(error: ExtensionError) -> Self {
        match error {
            ExtensionError::MissingImplementation { .. } => {
                PyImportError::new_err(error.to_string())
            }
            ExtensionError::ConflictingImplementation { .. } => {
                PyRuntimeError::new_err(error.to_string())
            }
        }
    }
}

/// A Rust entry point that forwards Python positional and keyword arguments.
///
/// The implementation owns argument parsing. Forward to its existing PyO3
/// wrapper with `wrap_pyfunction!` to preserve defaults and Python exceptions.
pub type PythonExtensionFunction =
    for<'py> fn(&Bound<'py, PyTuple>, Option<&Bound<'py, PyDict>>) -> PyResult<Py<PyAny>>;

impl ExtensionPoint<PythonExtensionFunction> {
    /// Forward arguments to the registered binding without a Python module lookup.
    ///
    /// Accepts a Rust tuple or an existing Python tuple. Argument objects,
    /// keyword dictionaries and backend exceptions are passed through unchanged.
    pub fn call<'py, A>(
        &self,
        py: Python<'py>,
        args: A,
        kwargs: Option<&Bound<'py, PyDict>>,
    ) -> PyResult<Py<PyAny>>
    where
        A: IntoPyObject<'py, Target = PyTuple>,
    {
        let function = self.get()?;
        let args = args.into_pyobject_or_pyerr(py)?.into_bound();
        function(&args, kwargs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;

    use pyo3::{
        exceptions::{PyTypeError, PyValueError},
        prelude::*,
        types::{PyCFunction, PyDictMethods, PyList, PyTupleMethods},
    };

    type Transform = fn(i32) -> i32;
    static DOUBLE: ExtensionImplementation<Transform> =
        ExtensionImplementation::new("arithmetic", |value| value * 2);
    static OTHER_DOUBLE: ExtensionImplementation<Transform> =
        ExtensionImplementation::new("arithmetic", |value| value * 2);

    #[test]
    fn typed_registration_uses_descriptor_identity_and_independent_slots() {
        let point = ExtensionPoint::new("double");
        assert_eq!(point.provider(), None);
        assert_eq!(
            point.get().unwrap_err(),
            ExtensionError::MissingImplementation {
                operation: "double"
            }
        );
        point.register(&DOUBLE).unwrap();
        point.register(&DOUBLE).unwrap();
        assert_eq!(point.provider(), Some("arithmetic"));
        assert_eq!(point.get().unwrap()(3), 6);
        assert_eq!(
            point.register(&OTHER_DOUBLE),
            Err(ExtensionError::ConflictingImplementation {
                operation: "double",
                registered_provider: "arithmetic",
                attempted_provider: "arithmetic",
            })
        );
        assert!(std::ptr::eq(point.get().unwrap(), &DOUBLE.implementation));

        let independent = ExtensionPoint::new("double");
        independent.register(&OTHER_DOUBLE).unwrap();
        assert!(std::ptr::eq(
            independent.get().unwrap(),
            &OTHER_DOUBLE.implementation
        ));
    }

    #[test]
    fn concurrent_registration_installs_one_complete_function_table() {
        struct Functions {
            first: Transform,
            second: Transform,
        }
        static INCREASE: ExtensionImplementation<Functions> = ExtensionImplementation::new(
            "increase",
            Functions {
                first: |value| value + 1,
                second: |value| value + 2,
            },
        );
        static DECREASE: ExtensionImplementation<Functions> = ExtensionImplementation::new(
            "decrease",
            Functions {
                first: |value| value - 1,
                second: |value| value - 2,
            },
        );
        let point = ExtensionPoint::new("transform");
        let barrier = Barrier::new(16);
        let results = std::thread::scope(|scope| {
            let handles: Vec<_> = (0..16)
                .map(|index| {
                    let point = &point;
                    let barrier = &barrier;
                    scope.spawn(move || {
                        let implementation = if index % 2 == 0 { &INCREASE } else { &DECREASE };
                        barrier.wait();
                        (implementation.provider, point.register(implementation))
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|handle| handle.join().unwrap())
                .collect::<Vec<_>>()
        });
        let provider = point.provider().unwrap();
        assert_eq!(
            results.iter().filter(|(_, result)| result.is_ok()).count(),
            8
        );
        for (attempted_provider, result) in results {
            if attempted_provider == provider {
                assert_eq!(result, Ok(()));
            } else {
                assert_eq!(
                    result,
                    Err(ExtensionError::ConflictingImplementation {
                        operation: "transform",
                        registered_provider: provider,
                        attempted_provider,
                    })
                );
            }
        }
        let functions = point.get().unwrap();
        let values = ((functions.first)(0), (functions.second)(0));
        assert_eq!(
            values,
            if provider == "increase" {
                (1, 2)
            } else {
                (-1, -2)
            }
        );
    }

    fn capture(
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        Ok((args, kwargs).into_pyobject(args.py())?.into_any().unbind())
    }

    #[test]
    fn python_forwarding_preserves_tuple_keyword_and_object_identities() {
        static CAPTURE: ExtensionImplementation<PythonExtensionFunction> =
            ExtensionImplementation::new("capture", capture);
        let point = ExtensionPoint::new("capture arguments");
        point.register(&CAPTURE).unwrap();
        Python::initialize();
        Python::attach(|py| {
            let empty = point.call(py, (), None).unwrap();
            let empty = empty.cast_bound::<PyTuple>(py).unwrap();
            assert!(
                empty
                    .get_item(0)
                    .unwrap()
                    .cast::<PyTuple>()
                    .unwrap()
                    .is_empty()
            );
            assert!(empty.get_item(1).unwrap().is_none());

            let first = PyList::empty(py);
            let second = PyDict::new(py);
            let kwargs = PyDict::new(py);
            kwargs.set_item("value", &first).unwrap();
            let args = (&first, &second).into_pyobject(py).unwrap();
            let result = point.call(py, &args, Some(&kwargs)).unwrap();
            let result = result.cast_bound::<PyTuple>(py).unwrap();
            let forwarded = result.get_item(0).unwrap();
            assert!(forwarded.is(&args));
            let forwarded = forwarded.cast::<PyTuple>().unwrap();
            assert!(forwarded.get_item(0).unwrap().is(&first));
            assert!(forwarded.get_item(1).unwrap().is(&second));
            assert!(result.get_item(1).unwrap().is(&kwargs));
        });
    }

    #[pyfunction]
    #[pyo3(signature = (value, count=2, *, observer=None))]
    fn parsed(
        py: Python<'_>,
        value: Py<PyAny>,
        count: usize,
        observer: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        match observer {
            Some(observer) => observer.call1(py, (value, count)),
            None => Ok((value, count).into_pyobject(py)?.into_any().unbind()),
        }
    }

    fn forward_parsed(
        args: &Bound<'_, PyTuple>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<Py<PyAny>> {
        Ok(wrap_pyfunction!(parsed, args.py())?
            .call(args, kwargs)?
            .unbind())
    }

    #[test]
    fn python_forwarding_reuses_parser_defaults_and_preserves_exception_identity() {
        static PARSED: ExtensionImplementation<PythonExtensionFunction> =
            ExtensionImplementation::new("parsed", forward_parsed);
        let point = ExtensionPoint::new("parse arguments");
        point.register(&PARSED).unwrap();
        Python::initialize();
        Python::attach(|py| {
            let value = PyList::empty(py);
            let defaulted = point.call(py, (&value,), None).unwrap();
            let defaulted = defaulted.cast_bound::<PyTuple>(py).unwrap();
            assert!(defaulted.get_item(0).unwrap().is(&value));
            assert_eq!(
                defaulted.get_item(1).unwrap().extract::<usize>().unwrap(),
                2
            );
            let positional = point.call(py, (&value, 5), None).unwrap();
            assert_eq!(
                positional
                    .bind(py)
                    .get_item(1)
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                5
            );
            let kwargs = PyDict::new(py);
            kwargs.set_item("value", &value).unwrap();
            kwargs.set_item("count", 3).unwrap();
            let named = point.call(py, (), Some(&kwargs)).unwrap();
            assert!(named.bind(py).get_item(0).unwrap().is(&value));
            assert_eq!(
                named
                    .bind(py)
                    .get_item(1)
                    .unwrap()
                    .extract::<usize>()
                    .unwrap(),
                3
            );

            let direct = wrap_pyfunction!(parsed, py).unwrap();
            for keyword in ["unknown", "count"] {
                let kwargs = PyDict::new(py);
                kwargs.set_item(keyword, "invalid").unwrap();
                let forwarded_error = point.call(py, (&value,), Some(&kwargs)).unwrap_err();
                let direct_error = direct.call((&value,), Some(&kwargs)).unwrap_err();
                assert!(forwarded_error.is_instance_of::<PyTypeError>(py));
                assert_eq!(forwarded_error.to_string(), direct_error.to_string());
            }

            let failure = PyValueError::new_err("observer failure");
            let expected = failure.value(py).clone().into_any().unbind();
            let observer = PyCFunction::new_closure(py, None, None, move |args, _| {
                Err::<(), _>(failure.clone_ref(args.py()))
            })
            .unwrap();
            let kwargs = PyDict::new(py);
            kwargs.set_item("observer", observer).unwrap();
            let error = point.call(py, (&value,), Some(&kwargs)).unwrap_err();
            assert!(error.value(py).is(expected.bind(py)));
        });
    }

    #[test]
    fn registration_errors_become_named_python_errors() {
        Python::initialize();
        Python::attach(|py| {
            let missing = PyErr::from(ExtensionError::MissingImplementation {
                operation: "sector decomposition",
            });
            assert!(missing.is_instance_of::<PyImportError>(py));
            assert!(missing.to_string().contains("sector decomposition"));
            let conflicting = PyErr::from(ExtensionError::ConflictingImplementation {
                operation: "sector decomposition",
                registered_provider: "first backend",
                attempted_provider: "second backend",
            });
            assert!(conflicting.is_instance_of::<PyRuntimeError>(py));
            for name in ["sector decomposition", "first backend", "second backend"] {
                assert!(conflicting.to_string().contains(name));
            }
        });
    }
}
