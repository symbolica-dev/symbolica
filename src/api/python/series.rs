use super::*;

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
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(from_py_object, name = "Series", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonSeries {
    pub series: Series<AtomField>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonSeries {
    /// Get the coefficient of the `exp`th power of the expansion variable.
    pub fn __getitem__(&self, exp: ConvertibleToExpression) -> PyResult<PythonExpression> {
        self.get_coefficient(exp)
    }

    /// Get the coefficient of the term with exponent `exp`. Alternatively, use `series[exp]`.
    pub fn get_coefficient(&self, exp: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let idx = exp.to_expression().expr;
        let r: Rational = idx
            .try_into()
            .map_err(|e| exceptions::PyTypeError::new_err(e))?;

        Ok(self.series.coefficient(r).into())
    }

    /// Iterate over the terms of the series, yielding pairs of exponent and coefficient.
    #[gen_stub(override_return_type(type_repr = "typing.Iterator[tuple[Expression, Expression]]"))]
    pub fn __iter__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyIterator>> {
        let v: Vec<(PythonExpression, PythonExpression)> = (&self.series)
            .into_iter()
            .map(|(c, a)| (Atom::num(c).into(), a.clone().into()))
            .collect();

        v.into_pyobject(py)?.try_iter()
    }

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
            max_line_length = Some(80),
            indentation = 4,
            fill_indented_lines = true,
            terms_on_new_line = false,
            color_top_level_sum = true,
            color_builtin_symbols = true,
            bracket_level_colors = Some([
                244, 25, 97, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60,
            ]),
            print_ring = true,
            symmetric_representation_for_finite_field = false,
            explicit_rational_polynomial = false,
            number_thousands_separator = None,
            multiplication_operator = '·',
            double_star_for_exponentiation = false,
            square_brackets_for_function = false,
            function_brackets = ('(',')'),
            num_exp_as_superscript = true,
            precision = None,
            show_namespaces = false,
            hide_namespace = None,
            include_attributes = false,
            max_terms = None,
            custom_print_mode = None)
        )]
    pub fn format(
        &self,
        mode: PythonPrintMode,
        max_line_length: Option<usize>,
        indentation: usize,
        fill_indented_lines: bool,
        terms_on_new_line: bool,
        color_top_level_sum: bool,
        color_builtin_symbols: bool,
        bracket_level_colors: Option<[u8; 16]>,
        print_ring: bool,
        symmetric_representation_for_finite_field: bool,
        explicit_rational_polynomial: bool,
        number_thousands_separator: Option<char>,
        multiplication_operator: char,
        double_star_for_exponentiation: bool,
        square_brackets_for_function: bool,
        function_brackets: (char, char),
        num_exp_as_superscript: bool,
        precision: Option<usize>,
        show_namespaces: bool,
        hide_namespace: Option<&str>,
        include_attributes: bool,
        max_terms: Option<usize>,
        custom_print_mode: Option<usize>,
    ) -> PyResult<String> {
        Ok(self
            .series
            .format_string(
                &PrintOptions {
                    max_line_length,
                    indentation,
                    fill_indented_lines,
                    terms_on_new_line,
                    color_top_level_sum,
                    color_builtin_symbols,
                    bracket_level_colors,
                    print_ring,
                    symmetric_representation_for_finite_field,
                    explicit_rational_polynomial,
                    number_thousands_separator,
                    multiplication_operator,
                    double_star_for_exponentiation,
                    #[allow(deprecated)]
                    square_brackets_for_function,
                    function_brackets,
                    num_exp_as_superscript,
                    mode: mode.into(),
                    precision,
                    pretty_matrix: false,
                    hide_all_namespaces: !show_namespaces,
                    color_namespace: true,
                    hide_namespace: if show_namespaces {
                        hide_namespace.map(intern_string)
                    } else {
                        None
                    },
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

    #[pyo3(signature=(num, den = 1))]
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
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
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
    #[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
    #[pyclass(skip_from_py_object, name = "AtomIterator", module = "symbolica.core")]
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
            AtomView::Pow(p) => p.iter(),
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
    #[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
    #[pyclass(skip_from_py_object, name = "MatchIterator", module = "symbolica.core")]
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
    #[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
    #[pyclass(
        skip_from_py_object,
        name = "ReplaceIterator",
        module = "symbolica.core"
    )]
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

/// A helper enum to extract either a polynomial or an integer.
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
