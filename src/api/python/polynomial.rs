use super::*;

/// A multivariate polynomial with rational coefficients.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "Polynomial",
    subclass,
    module = "symbolica.core"
)]
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
            max_line_length = Some(80),
            indentation = 4,
            fill_indented_lines = true,
            terms_on_new_line = false,
            color_top_level_sum = true,
            color_builtin_symbols = true,
            bracket_level_colors = None,
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
        Ok(self.poly.format_string(
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
        inputs: PyArrayLike1<'py, f64, AllowTypeChange>,
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
        inputs: PyArrayLike1<'py, Complex64, AllowTypeChange>,
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
    #[pyo3(signature = (arg, vars, default_namespace = None))]
    #[classmethod]
    pub fn parse(
        _cls: &Bound<'_, PyType>,
        py: Python,
        arg: &str,
        vars: Vec<PyBackedStr>,
        default_namespace: Option<String>,
    ) -> PyResult<Self> {
        let mut var_map = vec![];
        let mut var_name_map: SmallVec<[SmartString<LazyCompact>; INLINED_EXPONENTS]> =
            SmallVec::new();

        let namespace = DefaultNamespace {
            namespace: if let Some(ns) = default_namespace {
                ns.into()
            } else {
                get_namespace(py)?.into()
            },
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
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "FiniteFieldPolynomial",
    subclass,
    module = "symbolica.core"
)]
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
            max_line_length = Some(80),
            indentation = 4,
            fill_indented_lines = true,
            terms_on_new_line = false,
            color_top_level_sum = true,
            color_builtin_symbols = true,
            bracket_level_colors = None,
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
        Ok(self.poly.format_string(
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
    #[pyo3(signature = (arg, vars, prime, default_namespace = None))]
    #[classmethod]
    pub fn parse(
        _cls: &Bound<'_, PyType>,
        py: Python,
        arg: &str,
        vars: Vec<PyBackedStr>,
        prime: u64,
        default_namespace: Option<String>,
    ) -> PyResult<Self> {
        let mut var_map = vec![];
        let mut var_name_map = vec![];

        let namespace = DefaultNamespace {
            namespace: if let Some(ns) = default_namespace {
                ns.into()
            } else {
                get_namespace(py)?.into()
            },
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
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "PrimeTwoPolynomial",
    subclass,
    module = "symbolica.core"
)]
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
            max_line_length = Some(80),
            indentation = 4,
            fill_indented_lines = true,
            terms_on_new_line = false,
            color_top_level_sum = true,
            color_builtin_symbols = true,
            bracket_level_colors = None,
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
        Ok(self.poly.format_string(
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
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
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
            max_line_length = Some(80),
            indentation = 4,
            fill_indented_lines = true,
            terms_on_new_line = false,
            color_top_level_sum = true,
            color_builtin_symbols = true,
            bracket_level_colors = None,
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
        Ok(self.poly.format_string(
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
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "GaloisFieldPolynomial",
    subclass,
    module = "symbolica.core"
)]
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
            max_line_length = Some(80),
            indentation = 4,
            fill_indented_lines = true,
            terms_on_new_line = false,
            color_top_level_sum = true,
            color_builtin_symbols = true,
            bracket_level_colors = None,
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
        Ok(self.poly.format_string(
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
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "NumberFieldPolynomial",
    subclass,
    module = "symbolica.core"
)]
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
            max_line_length = Some(80),
            indentation = 4,
            fill_indented_lines = true,
            terms_on_new_line = false,
            color_top_level_sum = true,
            color_builtin_symbols = true,
            bracket_level_colors = None,
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
        Ok(self.poly.format_string(
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
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "RationalPolynomial",
    subclass,
    module = "symbolica.core"
)]
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
    #[pyo3(signature = (arg, vars, default_namespace = None))]
    #[classmethod]
    pub fn parse(
        _cls: &Bound<'_, PyType>,
        py: Python,
        arg: &str,
        vars: Vec<PyBackedStr>,
        default_namespace: Option<String>,
    ) -> PyResult<Self> {
        let mut var_map = vec![];
        let mut var_name_map = vec![];

        let namespace = DefaultNamespace {
            namespace: if let Some(ns) = default_namespace {
                intern_string(&ns).into()
            } else {
                get_namespace(py)?.into()
            },
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
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
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
    #[pyo3(signature = (arg, vars, prime, default_namespace = None))]
    #[classmethod]
    pub fn parse(
        _cls: &Bound<'_, PyType>,
        py: Python,
        arg: &str,
        vars: Vec<PyBackedStr>,
        prime: u64,
        default_namespace: Option<String>,
    ) -> PyResult<Self> {
        let mut var_map = vec![];
        let mut var_name_map = vec![];

        let namespace = DefaultNamespace {
            namespace: if let Some(ns) = default_namespace {
                intern_string(&ns).into()
            } else {
                get_namespace(py)?.into()
            },
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

/// A type that can be converted to a rational polynomial.
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
