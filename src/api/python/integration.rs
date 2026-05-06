use super::*;

/// A sample from the Symbolica integrator. It could consist of discrete layers,
/// accessible with `d` (empty when there are no discrete layers), and the final continuous layer `c` if it is present.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(from_py_object, name = "Sample", module = "symbolica.core")]
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

/// A probe that is used to access the Jacobian weight of a point or region
/// of interest.
///
/// For continuous probes, `None` skips that dimension and includes the full
/// range of the dimension (Jacobian weight of 1).
///
/// For discrete probes, the first vector specifies a path through nested
/// discrete grids, and the second vector specifies the final continuous probe.
/// The path may stop before the full grid depth, in which case the remaining
/// sub-Jacobian weight is 1 and the continuous probe must be empty.
///
/// For uniform probes, `None` in the discrete indices skips that discrete
/// dimension and includes its full range (Jacobian weight of 1).
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(from_py_object, name = "Probe", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonProbe {
    #[pyo3(get)]
    /// A sample point per (nested) discrete layer. Empty if not present.
    d: Vec<usize>,
    #[pyo3(get)]
    /// A sample in the continuous layer. Empty if not present.
    c: Vec<Option<f64>>,
    /// A uniform sample. Empty if not present.
    u: Vec<Option<usize>>,
}

impl PythonProbe {
    pub fn into_probe(self) -> Probe<f64> {
        if self.u.is_empty() {
            if self.d.is_empty() {
                Probe::Continuous(self.c)
            } else {
                Probe::Discrete(self.d, self.c)
            }
        } else {
            Probe::Uniform(self.u, self.c)
        }
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonProbe {
    #[pyo3(signature = (disc, cont=None))]
    #[classmethod]
    pub fn discrete(
        _cls: &Bound<'_, PyType>,
        disc: Vec<usize>,
        cont: Option<Vec<Option<f64>>>,
    ) -> Self {
        Self {
            d: disc,
            c: cont.unwrap_or_default(),
            u: Vec::new(),
        }
    }

    #[classmethod]
    pub fn continuous(_cls: &Bound<'_, PyType>, cont: Vec<Option<f64>>) -> Self {
        Self {
            d: Vec::new(),
            c: cont,
            u: Vec::new(),
        }
    }

    #[pyo3(signature = (uni, cont=None))]
    #[classmethod]
    pub fn uniform(
        _cls: &Bound<'_, PyType>,
        uni: Vec<Option<usize>>,
        cont: Option<Vec<Option<f64>>>,
    ) -> Self {
        Self {
            d: Vec::new(),
            c: cont.unwrap_or_default(),
            u: uni,
        }
    }
}

/// A reproducible, fast, non-cryptographic random number generator suitable for parallel Monte Carlo simulations.
/// A `seed` has to be set, which can be any `u64` number (small numbers work just as well as large numbers).
///
/// Each thread or instance generating samples should use the same `seed` but a different `stream_id`,
/// which is an instance counter starting at 0.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "RandomNumberGenerator",
    module = "symbolica.core"
)]
#[derive(Clone)]
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

    /// Clone the random number generator, creating a new instance with the same state. The cloned instance will generate the same sequence of random numbers as the original instance.
    fn __copy__(&self) -> Self {
        self.clone()
    }

    /// Generate the next random unsigned 64-bit number in the sequence.
    fn next(&mut self) -> u64 {
        self.state.next_u64()
    }

    /// Generate the next random floating-point number in the sequence, uniformly distributed in the range [0, 1).
    fn next_float(&mut self) -> f64 {
        self.state.random()
    }

    /// Import a random number generator from a previously exported state. The state should be a bytes object of length 32.
    #[classmethod]
    fn load(_cls: &Bound<'_, PyType>, state: Bound<'_, PyBytes>) -> PyResult<Self> {
        let state: [u8; 32] = state.as_bytes().try_into().map_err(|_| {
            exceptions::PyValueError::new_err("Invalid state size: expected 32 bytes")
        })?;

        Ok(PythonRandomNumberGenerator {
            state: MonteCarloRng::import(state),
        })
    }

    /// Export the random number generator state as a bytes object of length 32, which can be imported again to restore the state.
    fn save<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let state = self.state.export();
        PyBytes::new(py, &state).into()
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "NumericalIntegrator",
    module = "symbolica.core"
)]
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

    /// Probe the weight of a region in the grid.
    pub fn probe(&self, probe: PythonProbe) -> PyResult<f64> {
        self.grid
            .probe(&probe.into_probe())
            .map_err(|e| exceptions::PyValueError::new_err(e))
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
