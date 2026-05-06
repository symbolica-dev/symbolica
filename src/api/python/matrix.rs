use super::*;

/// A scalar or a matrix.
#[derive(FromPyObject)]
pub enum ScalarOrMatrix {
    Scalar(ConvertibleToRationalPolynomial),
    Matrix(PythonMatrix),
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(ScalarOrMatrix = ConvertibleToRationalPolynomial | PythonMatrix);

/// A Symbolica matrix with rational polynomial coefficients.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(from_py_object, name = "Matrix", subclass, module = "symbolica.core")]
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
            max_line_length = Some(80),
            indentation = 4,
            fill_indented_lines = true,
            pretty_matrix = true,
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
        pretty_matrix: bool,
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
        Ok(self.matrix.format_string(
            &PrintOptions {
                max_line_length,
                indentation,
                fill_indented_lines,
                terms_on_new_line: false,
                color_top_level_sum: false,
                color_builtin_symbols: false,
                bracket_level_colors: None,
                print_ring: false,
                symmetric_representation_for_finite_field: false,
                explicit_rational_polynomial: false,
                number_thousands_separator,
                multiplication_operator,
                double_star_for_exponentiation,
                #[allow(deprecated)]
                square_brackets_for_function,
                function_brackets,
                num_exp_as_superscript,
                mode: mode.into(),
                precision,
                pretty_matrix,
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
            PrintState::default(),
        ))
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
