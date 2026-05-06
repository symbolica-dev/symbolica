use super::*;

/// Operations on integers.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(from_py_object, name = "Integer", module = "symbolica.core")]
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

    /// Check if the number `n` is a prime number.
    #[classmethod]
    #[pyo3(signature = (n, k = 24))]
    fn is_prime(_cls: &Bound<'_, PyType>, n: Integer, k: usize) -> bool {
        n.is_prime(k)
    }

    /// Factor the number `n` into primes.
    #[classmethod]
    fn factor(_cls: &Bound<'_, PyType>, n: Integer) -> Vec<(Integer, Integer)> {
        n.factor()
    }

    /// Compute the Euler totient function for the number `n`.
    #[classmethod]
    fn totient(_cls: &Bound<'_, PyType>, n: Integer) -> Integer {
        n.totient()
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

/// An iterator over all 64-bit prime numbers.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(from_py_object, name = "PrimeIterator", module = "symbolica.core")]
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
