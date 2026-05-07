use super::*;
use crate::utils::Settable;

/// Operations that transform an expression.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "HeldExpression",
    subclass,
    module = "symbolica.core"
)]
#[derive(Clone)]
pub struct PythonHeldExpression {
    pub expr: Pattern,
}

impl From<Pattern> for PythonHeldExpression {
    fn from(expr: Pattern) -> Self {
        PythonHeldExpression { expr }
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonHeldExpression {
    /// Execute a bound transformer. If the transformer is unbound,
    /// you can call it with an expression as an argument.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x = S('x')
    /// >>> e = (x+1)**5
    /// >>> e = e.hold(T().expand())()
    /// >>> print(e)
    pub fn __call__(&self, py: Python) -> PyResult<PythonExpression> {
        let mut out = Atom::default();

        // TODO: pass a transformer state?
        py.detach(|| {
            Workspace::get_local()
                .with(|workspace| {
                    self.expr.replace_wildcards_with_matches_impl(
                        workspace,
                        &mut out,
                        &MatchStack::new(),
                        true,
                        None,
                    )
                })
                .map_err(|e| match e {
                    TransformerError::Interrupt => {
                        exceptions::PyKeyboardInterrupt::new_err("Interrupted by user")
                    }
                    TransformerError::ValueError(v) => exceptions::PyValueError::new_err(v),
                })
        })?;

        Ok(out.into())
    }

    /// Compare two expressions. If one of the expressions is not a number, an
    /// internal ordering will be used.
    fn __richcmp__(&self, other: ConvertibleToPattern, op: CompareOp) -> PyResult<PythonCondition> {
        Ok(match op {
            CompareOp::Eq => PythonCondition {
                condition: Relation::Eq(self.expr.clone(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Ne => PythonCondition {
                condition: Relation::Ne(self.expr.clone(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Ge => PythonCondition {
                condition: Relation::Ge(self.expr.clone(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Gt => PythonCondition {
                condition: Relation::Gt(self.expr.clone(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Le => PythonCondition {
                condition: Relation::Le(self.expr.clone(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Lt => PythonCondition {
                condition: Relation::Lt(self.expr.clone(), other.to_pattern()?.expr).into(),
            },
        })
    }

    /// Test if the expression is of a certain type.
    pub fn is_type(&self, atom_type: PythonAtomType) -> PythonCondition {
        PythonCondition {
            condition: Condition::Yield(Relation::IsType(
                self.expr.clone(),
                match atom_type {
                    PythonAtomType::Num => AtomType::Num,
                    PythonAtomType::Var => AtomType::Var,
                    PythonAtomType::Add => AtomType::Add,
                    PythonAtomType::Mul => AtomType::Mul,
                    PythonAtomType::Pow => AtomType::Pow,
                    PythonAtomType::Fn => AtomType::Fun,
                },
            )),
        }
    }

    /// Returns true iff `self` contains `a` literally.
    pub fn contains(&self, s: ConvertibleToOpenPattern) -> PyResult<PythonCondition> {
        Ok(PythonCondition {
            condition: Condition::Yield(Relation::Contains(
                self.expr.clone(),
                s.to_pattern()?.expr,
            )),
        })
    }

    /// Create a transformer that tests whether the pattern is found in the expression.
    /// Restrictions on the pattern can be supplied through `cond`.
    #[pyo3(signature = (lhs, cond = None, min_level=0, max_level=None, level_range = None, level_is_tree_depth = false, partial=true, allow_new_wildcards_on_rhs = false))]
    pub fn matches(
        &self,
        lhs: ConvertibleToPattern,
        cond: Option<ConvertibleToPatternRestriction>,
        min_level: usize,
        max_level: Option<usize>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: bool,
        partial: bool,
        allow_new_wildcards_on_rhs: bool,
    ) -> PyResult<PythonCondition> {
        let conditions = cond.map(|r| r.0).unwrap_or_default();
        let settings = MatchSettings {
            level_range: level_range.unwrap_or((min_level, max_level)),
            level_is_tree_depth,
            allow_new_wildcards_on_rhs,
            partial,
            ..MatchSettings::default()
        };

        Ok(PythonCondition {
            condition: Condition::Yield(Relation::Matches(
                self.expr.clone(),
                lhs.to_pattern()?.expr,
                conditions,
                settings,
            )),
        })
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __add__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        let res = Workspace::get_local().with(|workspace| {
            Ok::<Pattern, PyErr>(self.expr.add(&rhs.to_pattern()?.expr, workspace))
        })?;

        Ok(res.into())
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __radd__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        self.__add__(rhs)
    }

    ///  Subtract `other` from this transformer, returning the result.
    pub fn __sub__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        self.__add__(ConvertibleToPattern::Held(rhs.to_pattern()?.__neg__()?))
    }

    ///  Subtract this transformer from `other`, returning the result.
    pub fn __rsub__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        rhs.to_pattern()?
            .__add__(ConvertibleToPattern::Held(self.__neg__()?))
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __mul__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        let res = Workspace::get_local().with(|workspace| {
            Ok::<Pattern, PyErr>(self.expr.mul(&rhs.to_pattern()?.expr, workspace))
        });

        Ok(res?.into())
    }

    /// Add this transformer to `other`, returning the result.
    pub fn __rmul__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        self.__mul__(rhs)
    }

    /// Divide this transformer by `other`, returning the result.
    pub fn __truediv__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        let res = Workspace::get_local().with(|workspace| {
            Ok::<Pattern, PyErr>(self.expr.div(&rhs.to_pattern()?.expr, workspace))
        });

        Ok(res?.into())
    }

    /// Divide `other` by this transformer, returning the result.
    pub fn __rtruediv__(&self, rhs: ConvertibleToPattern) -> PyResult<PythonHeldExpression> {
        rhs.to_pattern()?
            .__truediv__(ConvertibleToPattern::Held(self.clone()))
    }

    /// Take `self` to power `exp`, returning the result.
    pub fn __pow__(
        &self,
        exponent: ConvertibleToPattern,
        modulo: Option<i64>,
    ) -> PyResult<PythonHeldExpression> {
        if modulo.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "Optional number argument not supported",
            ));
        }

        let res = Workspace::get_local().with(|workspace| {
            Ok::<_, PyErr>(self.expr.pow(&exponent.to_pattern()?.expr, workspace))
        });

        Ok(res?.into())
    }

    /// Take `base` to power `self`, returning the result.
    pub fn __rpow__(
        &self,
        base: ConvertibleToPattern,
        modulo: Option<i64>,
    ) -> PyResult<PythonHeldExpression> {
        base.to_pattern()?
            .__pow__(ConvertibleToPattern::Held(self.clone()), modulo)
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __xor__(&self, _rhs: Py<PyAny>) -> PyResult<PythonHeldExpression> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor an expression. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __rxor__(&self, _rhs: Py<PyAny>) -> PyResult<PythonHeldExpression> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor an expression. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Negate the current transformer, returning the result.
    pub fn __neg__(&self) -> PyResult<PythonHeldExpression> {
        let res =
            Workspace::get_local().with(|workspace| Ok::<Pattern, PyErr>(self.expr.neg(workspace)));

        Ok(res?.into())
    }
}

/// Operations that transform an expression.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "Transformer",
    subclass,
    module = "symbolica.core"
)]
#[derive(Clone)]
pub struct PythonTransformer {
    pub chain: Vec<Transformer>,
}

impl PythonTransformer {
    fn append_transformer(&self, transformer: Transformer) -> PyResult<PythonTransformer> {
        let mut r = self.clone();
        r.chain.push(transformer);
        Ok(r)
    }

    fn to_pattern(&self) -> Pattern {
        Pattern::Transformer(Box::new((None, self.chain.clone())))
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonTransformer {
    /// Create a new transformer.
    #[new]
    pub fn new() -> PythonTransformer {
        PythonTransformer { chain: vec![] }
    }

    /// Execute an unbound transformer on the given expression. If the transformer
    /// is bound, use `execute()` instead.
    ///
    /// Examples
    /// --------
    /// >>> x = S('x')
    /// >>> e = T().expand()((1+x)**2)
    ///
    /// Parameters
    /// ----------
    /// expr: Expression
    ///     The expression to transform.
    /// stats_to_file: str, optional
    ///     If set, the output of the `stats` transformer will be written to a file in JSON format.
    #[pyo3(signature = (expr, stats_to_file = None))]
    pub fn __call__(
        &self,
        expr: ConvertibleToExpression,
        stats_to_file: Option<String>,
        py: Python,
    ) -> PyResult<PythonExpression> {
        let e = expr.to_expression();

        let mut out = Atom::new();

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

        let _ = py.detach(|| {
            Workspace::get_local()
                .with(|ws| {
                    Transformer::execute_chain(e.as_view(), &self.chain, ws, &state, &mut out)
                })
                .map_err(|e| match e {
                    TransformerError::Interrupt => {
                        exceptions::PyKeyboardInterrupt::new_err("Interrupted by user")
                    }
                    TransformerError::ValueError(v) => exceptions::PyValueError::new_err(v),
                })
        })?;

        Ok(out.into())
    }

    /// Compare two expressions. If one of the expressions is not a number, an
    /// internal ordering will be used.
    fn __richcmp__(
        &self,
        other: ConvertibleToOpenPattern,
        op: CompareOp,
    ) -> PyResult<PythonCondition> {
        Ok(match op {
            CompareOp::Eq => PythonCondition {
                condition: Relation::Eq(self.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Ne => PythonCondition {
                condition: Relation::Ne(self.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Ge => PythonCondition {
                condition: Relation::Ge(self.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Gt => PythonCondition {
                condition: Relation::Gt(self.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Le => PythonCondition {
                condition: Relation::Le(self.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Lt => PythonCondition {
                condition: Relation::Lt(self.to_pattern(), other.to_pattern()?.expr).into(),
            },
        })
    }

    /// Test if the expression is of a certain type.
    pub fn is_type(&self, atom_type: PythonAtomType) -> PythonCondition {
        PythonCondition {
            condition: Condition::Yield(Relation::IsType(
                self.to_pattern(),
                match atom_type {
                    PythonAtomType::Num => AtomType::Num,
                    PythonAtomType::Var => AtomType::Var,
                    PythonAtomType::Add => AtomType::Add,
                    PythonAtomType::Mul => AtomType::Mul,
                    PythonAtomType::Pow => AtomType::Pow,
                    PythonAtomType::Fn => AtomType::Fun,
                },
            )),
        }
    }

    /// Returns true iff `self` contains `a` literally.
    pub fn contains(&self, s: ConvertibleToOpenPattern) -> PyResult<PythonCondition> {
        Ok(PythonCondition {
            condition: Condition::Yield(Relation::Contains(
                self.to_pattern(),
                s.to_pattern()?.expr,
            )),
        })
    }

    /// Create a transformer that tests whether the pattern is found in the expression.
    /// Restrictions on the pattern can be supplied through `cond`.
    #[pyo3(signature = (lhs, cond = None, min_level=0, max_level=None, level_range = None, level_is_tree_depth = false, partial=true, allow_new_wildcards_on_rhs = false))]
    pub fn matches(
        &self,
        lhs: ConvertibleToOpenPattern,
        cond: Option<ConvertibleToPatternRestriction>,
        min_level: usize,
        max_level: Option<usize>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: bool,
        partial: bool,
        allow_new_wildcards_on_rhs: bool,
    ) -> PyResult<PythonCondition> {
        let conditions = cond.map(|r| r.0).unwrap_or_default();
        let settings = MatchSettings {
            level_range: level_range.unwrap_or((min_level, max_level)),
            level_is_tree_depth,
            allow_new_wildcards_on_rhs,
            partial,
            ..MatchSettings::default()
        };

        Ok(PythonCondition {
            condition: Condition::Yield(Relation::Matches(
                self.to_pattern(),
                lhs.to_pattern()?.expr,
                conditions,
                settings,
            )),
        })
    }

    /// Create a transformer that expands products and powers.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, Transformer
    /// >>> x, x_ = S('x', 'x_')
    /// >>> f = S('f')
    /// >>> e = f((x+1)**2).replace(f(x_), x_.hold(T().expand()))
    /// >>> print(e)
    #[pyo3(signature = (var = None, via_poly = None))]
    pub fn expand(
        &self,
        var: Option<ConvertibleToExpression>,
        via_poly: Option<bool>,
    ) -> PyResult<PythonTransformer> {
        if let Some(var) = var {
            let e = var.to_expression();
            if matches!(e.expr, Atom::Var(_) | Atom::Fun(_)) {
                self.append_transformer(Transformer::Expand(
                    Some(e.expr),
                    via_poly.unwrap_or(false),
                ))
            } else {
                Err(exceptions::PyValueError::new_err(
                    "Expansion must be done wrt an indeterminate",
                ))
            }
        } else {
            self.append_transformer(Transformer::Expand(None, via_poly.unwrap_or(false)))
        }
    }

    /// Create a transformer that distributes numbers in the expression, for example:
    /// `2*(x+y)` -> `2*x+2*y`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> x, y = S('x', 'y')
    /// >>> e = 3*(x+y)*(4*x+5*y)
    /// >>> print(Transformer().expand_num()(e))
    ///
    /// yields
    ///
    /// ```log
    /// (3*x+3*y)*(4*x+5*y)
    /// ```
    pub fn expand_num(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::ExpandNum)
    }

    /// Create a transformer that computes the product of a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x__ = S('x__')
    /// >>> f = S('f')
    /// >>> e = f(2,3).replace(f(x__), x__.hold(T().prod()))
    /// >>> print(e)
    pub fn prod(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Product)
    }

    /// Create a transformer that computes the sum of a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x__ = S('x__')
    /// >>> f = S('f')
    /// >>> e = f(2,3).replace(f(x__), x__.hold(T().sum()))
    /// >>> print(e)
    pub fn sum(&self) -> PyResult<PythonTransformer> {
        let mut r = self.clone();
        r.chain.push(Transformer::Sum);
        Ok(r)
    }

    /// Create a transformer that returns the number of arguments.
    /// If the argument is not a function, return 0.
    ///
    /// If `only_for_arg_fun` is `True`, only count the number of arguments
    /// in the `arg()` function and return 1 if the input is not `arg`.
    /// This is useful for obtaining the length of a range during pattern matching.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x__ = S('x__')
    /// >>> f = S('f')
    /// >>> e = f(2,3,4).replace(f(x__), x__.hold(T().nargs()))
    /// >>> print(e)
    #[pyo3(signature = (only_for_arg_fun = false))]
    pub fn nargs(&self, only_for_arg_fun: bool) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::ArgCount(only_for_arg_fun))
    }

    /// Create a transformer that linearizes a function, optionally extracting `symbols`
    /// as well.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x, y, z, w, f, x__ = S('x', 'y', 'z', 'w', 'f', 'x__')
    /// >>> e = f(x+y, 4*z*w+3).replace(f(x__), f(x__).hold(T().linearize([z])))
    /// >>> print(e)
    ///
    /// yields `f(x,3)+f(y,3)+4*z*f(x,w)+4*z*f(y,w)`.
    #[pyo3(signature = (symbols = None))]
    pub fn linearize(&self, symbols: Option<Vec<PythonExpression>>) -> PyResult<PythonTransformer> {
        let mut c_symbols = vec![];
        if let Some(symbols) = symbols {
            for s in symbols {
                if let AtomView::Var(v) = s.expr.as_view() {
                    c_symbols.push(v.get_symbol());
                } else {
                    return Err(exceptions::PyValueError::new_err(
                        "Can only linearize in variables",
                    ));
                }
            }
        }

        self.append_transformer(Transformer::Linearize(if c_symbols.is_empty() {
            None
        } else {
            Some(c_symbols)
        }))
    }

    /// Create a transformer that sorts a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x_ = S('x__')
    /// >>> f = S('f')
    /// >>> e = f(3,2,1).replace(f(x__), x__.hold(T().sort()))
    /// >>> print(e)
    pub fn sort(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Sort)
    }

    /// Create a transformer that cycle-symmetrizes a function.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x_ = S('x__')
    /// >>> f = S('f')
    /// >>> e = f(1,2,4,1,2,3).replace(f(x__), x_.hold(T().cycle_symmetrize()))
    /// >>> print(e)
    ///
    /// Yields `f(1,2,3,1,2,4)`.
    pub fn cycle_symmetrize(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::CycleSymmetrize)
    }

    /// Create a transformer that removes elements from a list if they occur
    /// earlier in the list as well.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x__ = S('x__')
    /// >>> f = S('f')
    /// >>> e = f(1,2,1,2).replace(f(x__), x__.hold(T().deduplicate()))
    /// >>> print(e)
    ///
    /// Yields `f(1,2)`.
    pub fn deduplicate(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Deduplicate)
    }

    /// Create a transformer that extracts a rational polynomial from a coefficient.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> e = Expression.COEFF((x^2+1)/y^2).hold(T().from_coeff())
    /// >>> print(e)
    pub fn from_coeff(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::FromNumber)
    }

    /// Create a transformer that split a sum or product into a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x, x__ = S('x', 'x__')
    /// >>> f = S('f')
    /// >>> e = (x + 1).replace(x__, f(x__.hold(T().split())))
    /// >>> print(e)
    pub fn split(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Split)
    }

    /// Create a transformer that partitions a list of arguments into named bins of a given length,
    /// returning all partitions and their multiplicity.
    ///
    /// If the unordered list `elements` is larger than the bins, setting the flag `fill_last`
    /// will add all remaining elements to the last bin.
    ///
    /// Setting the flag `repeat` means that the bins will be repeated to exactly fit all elements,
    /// if possible.
    ///
    /// Note that the functions names to be provided for the bin names must be generated through `Expression.var`.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x_, f_id, g_id = S('x__', 'f', 'g')
    /// >>> f = S('f')
    /// >>> e = f(1,2,1,3).replace(f(x_), x_.hold(T().partitions([(f_id, 2), (g_id, 1), (f_id, 1)])))
    /// >>> print(e)
    ///
    /// yields:
    /// `2*f(1)*f(1,2)*g(3)+2*f(1)*f(1,3)*g(2)+2*f(1)*f(2,3)*g(1)+f(2)*f(1,1)*g(3)+2*f(2)*f(1,3)*g(1)+f(3)*f(1,1)*g(2)+2*f(3)*f(1,2)*g(1)`
    #[pyo3(signature = (bins, fill_last = false, repeat = false))]
    pub fn partitions(
        &self,
        bins: Vec<(ConvertibleToPattern, usize)>,
        fill_last: bool,
        repeat: bool,
    ) -> PyResult<PythonTransformer> {
        let mut conv_bins = vec![];

        for (x, len) in bins {
            let id = match &x.to_pattern()?.expr {
                Pattern::Literal(x) => {
                    if let AtomView::Var(x) = x.as_view() {
                        x.get_symbol()
                    } else {
                        return Err(exceptions::PyValueError::new_err(
                            "Derivative must be taken wrt a variable",
                        ));
                    }
                }
                Pattern::Wildcard(x) => *x,
                _ => {
                    return Err(exceptions::PyValueError::new_err(
                        "Derivative must be taken wrt a variable",
                    ));
                }
            };

            conv_bins.push((id, len));
        }

        self.append_transformer(Transformer::Partition(conv_bins, fill_last, repeat))
    }

    /// Create a transformer that generates all permutations of a list of arguments.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x_, f_id = S('x__', 'f')
    /// >>> f = S('f')
    /// >>> e = f(1,2,1,2).replace(f(x_), x_.hold(T().permutations(f_id)))
    /// >>> print(e)
    ///
    /// yields:
    /// `4*f(1,1,2,2)+4*f(1,2,1,2)+4*f(1,2,2,1)+4*f(2,1,1,2)+4*f(2,1,2,1)+4*f(2,2,1,1)`
    pub fn permutations(&self, function_name: ConvertibleToPattern) -> PyResult<PythonTransformer> {
        let id = match &function_name.to_pattern()?.expr {
            Pattern::Literal(x) => {
                if let AtomView::Var(x) = x.as_view() {
                    x.get_symbol()
                } else {
                    return Err(exceptions::PyValueError::new_err(
                        "Derivative must be taken wrt a variable",
                    ));
                }
            }
            Pattern::Wildcard(x) => *x,
            _ => {
                return Err(exceptions::PyValueError::new_err(
                    "Derivative must be taken wrt a variable",
                ));
            }
        };

        self.append_transformer(Transformer::Permutations(id))
    }

    /// Create a transformer that apply a function `f`.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x_ = S('x_')
    /// >>> f = S('f')
    /// >>> e = f(2).replace(f(x_), x_.hold(T().map(lambda r: r**2)))
    /// >>> print(e)
    pub fn map(
        &self,
        #[gen_stub(override_type(
            type_repr = "typing.Callable[[Expression], Expression | int | float | complex | decimal.Decimal]"
        ))]
        f: Py<PyAny>,
    ) -> PyResult<PythonTransformer> {
        let transformer = Transformer::Map(Box::new(move |expr, _state, out| {
            let expr = PythonExpression {
                expr: expr.to_owned(),
            };

            let res = Python::attach(|py| {
                f.call(py, (expr,), None)
                    .map_err(|e| {
                        TransformerError::ValueError(format!("Bad callback function: {e}"))
                    })?
                    .extract::<ConvertibleToExpression>(py)
                    .map_err(|e| {
                        TransformerError::ValueError(format!(
                            "Function does not return a pattern, but {e}",
                        ))
                    })
            });

            match res {
                Ok(res) => {
                    out.set_from_view(&res.to_expression().expr.as_view());
                    Ok(())
                }
                Err(e) => Err(e),
            }
        }));

        self.append_transformer(transformer)
    }

    /// Map a chain of transformers over the terms of the expression, optionally using multiple cores.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x, y = S('x', 'y')
    /// >>> t = T().map_terms(T().print(), n_cores=2)
    /// >>> e = t(x + y)
    #[pyo3(signature = (*transformers, n_cores=1))]
    pub fn map_terms(
        &self,
        transformers: &Bound<'_, PyTuple>,
        n_cores: usize,
    ) -> PyResult<PythonTransformer> {
        let mut rep_chain = vec![];
        // fuse all sub-transformers into one chain
        for r in transformers {
            let p = r.extract::<PythonTransformer>()?;
            rep_chain.extend(p.chain);
        }

        let pool = if n_cores < 2 || !LicenseManager::is_licensed() {
            None
        } else {
            Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(n_cores)
                    .build()
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!(
                            "Could not create thread pool: {e}",
                        ))
                    })?,
            ))
        };

        self.append_transformer(Transformer::MapTerms(rep_chain, pool))
    }

    /// Create a transformer that applies a transformer chain to every argument of the `arg()` function.
    /// If the input is not `arg()`, the transformer is applied to the input.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x = S('x')
    /// >>> f = S('f')
    /// >>> e = (1+x).hold(T().split().for_each(T().map(f)))()
    #[pyo3(signature = (*transformers))]
    pub fn for_each(&self, transformers: &Bound<'_, PyTuple>) -> PyResult<PythonTransformer> {
        let mut rep_chain = vec![];
        // fuse all sub-transformers into one chain
        for r in transformers {
            let p = r.extract::<PythonTransformer>()?;
            rep_chain.extend(p.chain);
        }

        self.append_transformer(Transformer::ForEach(rep_chain))
    }

    /// Create a transformer that checks for a Python interrupt,
    /// such as ctrl-c and aborts the current transformer.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x_ = S('x_')
    /// >>> f = S('f')
    /// >>> f(10).hold(T().repeat(T().replace(
    /// >>> f(x_), f(x_+1)).check_interrupt()))()
    pub fn check_interrupt(&self) -> PyResult<PythonTransformer> {
        let transformer = Transformer::Map(Box::new(move |expr, _state, out| {
            out.set_from_view(&expr);
            Python::attach(|py| py.check_signals()).map_err(|_| TransformerError::Interrupt)
        }));

        self.append_transformer(transformer)
    }

    /// Create a transformer that keeps executing the transformer chain until the input equals the output.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S('f')
    /// >>> e = E("f(5)")
    /// >>> e = e.hold(T().repeat(
    /// >>>     T().expand(),
    /// >>>     T().replace(f(x_), f(x_ - 1) + f(x_ - 2), x_.req_gt(1))
    /// >>> ))()
    #[pyo3(signature = (*transformers))]
    pub fn repeat(&self, transformers: &Bound<'_, PyTuple>) -> PyResult<PythonTransformer> {
        let mut rep_chain = vec![];
        // fuse all sub-transformers into one chain
        for r in transformers {
            let p = r.extract::<PythonTransformer>()?;
            rep_chain.extend(p.chain);
        }

        self.append_transformer(Transformer::Repeat(rep_chain))
    }

    /// Evaluate the condition and apply the `if_block` if the condition is true, otherwise apply the `else_block`.
    /// The expression that is the input of the transformer is the input for the condition, the `if_block` and the `else_block`.
    ///
    /// Examples
    /// --------
    /// >>> t = T.map_terms(T.if_then(T.contains(x), T.print()))
    /// >>> t(x + y + 4)
    ///
    /// prints `x`.
    #[pyo3(signature = (condition, if_block, else_block = None))]
    pub fn if_then(
        &self,
        condition: PythonCondition,
        if_block: PythonTransformer,
        else_block: Option<PythonTransformer>,
    ) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::IfElse(
            condition.condition,
            if_block.chain,
            else_block.map(|x| x.chain).unwrap_or_default(),
        ))
    }

    /// Execute the `condition` transformer. If the result of the `condition` transformer is different from the input expression,
    /// apply the `if_block`, otherwise apply the `else_block`. The input expression of the `if_block` is the output
    /// of the `condition` transformer.
    ///
    /// Examples
    /// --------
    /// >>> t = T.map_terms(T.if_changed(T.replace(x, y), T.print()))
    /// >>> print(t(x + y + 4))
    ///
    /// prints
    /// ```log
    /// y
    /// 2*y+4
    /// ```
    #[pyo3(signature = (condition, if_block, else_block = None))]
    pub fn if_changed(
        &self,
        condition: PythonTransformer,
        if_block: PythonTransformer,
        else_block: Option<PythonTransformer>,
    ) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::IfChanged(
            condition.chain,
            if_block.chain,
            else_block.map(|x| x.chain).unwrap_or_default(),
        ))
    }

    /// Break the current chain and all higher-level chains containing `if` transformers.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> t = T.map_terms(T.repeat(
    /// >>>     T.replace(y, 4),
    /// >>>     T.if_changed(T.replace(x, y),
    /// >>>                 T.break_chain()),
    /// >>>     T.print()  # print of y is never reached
    /// >>> ))
    /// >>> print(t(x))
    pub fn break_chain(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::BreakChain)
    }

    /// Chain several transformers. `chain(A,B,C)` is the same as `A.B.C`,
    /// where `A`, `B`, `C` are transformers.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x_ = S('x_')
    /// >>> f = S('f')
    /// >>> e = E("f(5)")
    /// >>> e = e.hold(T().repeat(
    /// >>>     T().expand(),
    /// >>>     T().replace(f(x_), f(x_ - 1) + f(x_ - 2), x_.req_gt(1))
    /// >>> ))()
    #[pyo3(signature = (*transformers))]
    pub fn chain(&self, transformers: &Bound<'_, PyTuple>) -> PyResult<PythonTransformer> {
        let mut r = self.clone();
        for t in transformers {
            let p = t.extract::<PythonTransformer>()?;
            r.chain.extend(p.chain);
        }

        Ok(r)
    }

    /// Set the coefficient ring to contain the variables in the `vars` list.
    /// This will move all variables into a rational polynomial function.
    ///
    /// Parameters
    /// ----------
    /// vars: List[Expression]
    ///     A list of variables
    pub fn set_coefficient_ring(&self, vars: Vec<PythonExpression>) -> PyResult<PythonTransformer> {
        let mut var_map = vec![];
        for v in vars {
            var_map.push(
                v.expr
                    .try_into()
                    .map_err(|e| exceptions::PyValueError::new_err(e))?,
            );
        }

        let a = Arc::new(var_map);

        self.append_transformer(Transformer::Map(Box::new(move |i, _state, o| {
            *o = i.set_coefficient_ring(&a);
            Ok(())
        })))
    }

    /// Create a transformer that collects terms involving the same power of `x`,
    /// where `x` is an indeterminate.
    /// Return the list of key-coefficient pairs and the remainder that matched no key.
    ///
    /// Both the key (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` transformers respectively.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x, y = S('x', 'y')
    /// >>> e = 5*x + x * y + x**2 + 5
    /// >>>
    /// >>> print(e.hold(T().collect(x))())
    ///
    /// yields `x^2+x*(y+5)+5`.
    ///
    /// >>> from symbolica import Expression, T
    /// >>> x, y, x_, var, coeff = S('x', 'y', 'x_', 'var', 'coeff')
    /// >>> e = 5*x + x * y + x**2 + 5
    /// >>> print(e.collect(x, key_map=T().replace(x_, var(x_)),
    ///         coeff_map=T().replace(x_, coeff(x_))))
    ///
    /// yields `var(1)*coeff(5)+var(x)*coeff(y+5)+var(x^2)*coeff(1)`.
    ///
    /// Parameters
    /// ----------
    /// x: Expression
    ///     The variable to collect terms in
    /// key_map: Transformer
    ///     A transformer to be applied to the quantity collected in
    /// coeff_map: Transformer
    ///     A transformer to be applied to the coefficient
    #[pyo3(signature = (*x, key_map = None, coeff_map = None))]
    pub fn collect(
        &self,
        x: Bound<'_, PyTuple>,
        key_map: Option<PythonTransformer>,
        coeff_map: Option<PythonTransformer>,
    ) -> PyResult<PythonTransformer> {
        let mut xs = vec![];
        for a in x {
            if let Ok(r) = a.extract::<PythonExpression>() {
                if matches!(r.expr, Atom::Var(_) | Atom::Fun(_)) {
                    xs.push(r.expr);
                } else {
                    return Err(exceptions::PyValueError::new_err(
                        "Collect must be done wrt a variable or function",
                    ));
                }
            } else {
                return Err(exceptions::PyValueError::new_err(
                    "Collect must be done wrt a variable or function",
                ));
            }
        }

        let key_map = if let Some(key_map) = key_map {
            key_map.chain
        } else {
            vec![]
        };

        let coeff_map = if let Some(coeff_map) = coeff_map {
            coeff_map.chain
        } else {
            vec![]
        };

        self.append_transformer(Transformer::Collect(xs, key_map, coeff_map))
    }

    /// Create a transformer that collects terms involving the same power of variables or functions with the name `x`.
    ///
    /// Both the key (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` transformers respectively.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression, T
    /// >>> x, f = S('x', 'f')
    /// >>> e = f(1,2) + x*f(1,2)
    /// >>>
    /// >>> print(e.hold(T().collect_symbol(x))())
    ///
    /// yields `(1+x)*f(1,2)`.
    ///
    /// Parameters
    /// ----------
    /// x: Expression
    ///      The symbol to collect in
    /// key_map: Transformer
    ///     A transformer to be applied to the quantity collected in
    /// coeff_map: Transformer
    ///     A transformer to be applied to the coefficient
    #[pyo3(signature = (x, key_map = None, coeff_map = None))]
    pub fn collect_symbol(
        &self,
        x: PythonExpression,
        key_map: Option<PythonTransformer>,
        coeff_map: Option<PythonTransformer>,
    ) -> PyResult<PythonTransformer> {
        let Some(x) = x.expr.get_symbol() else {
            return Err(exceptions::PyValueError::new_err(
                "Collect must be done wrt a variable or function",
            ));
        };

        let key_map = if let Some(key_map) = key_map {
            key_map.chain
        } else {
            vec![]
        };

        let coeff_map = if let Some(coeff_map) = coeff_map {
            coeff_map.chain
        } else {
            vec![]
        };

        self.append_transformer(Transformer::CollectSymbol(x, key_map, coeff_map))
    }

    /// Create a transformer that collects common factors from (nested) sums.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> e = E('x*(x+y*x+x^2+y*(x+x^2))')
    /// >>> e.hold(T().collect_factors())()
    ///
    /// yields
    ///
    /// ```log
    /// v1^2*(1+v1+v2+v2*(1+v1))
    /// ```
    pub fn collect_factors(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::CollectFactors)
    }

    /// Iteratively extract the minimal common powers of an indeterminate `v` for every term that contains `v`
    /// and continue to the next indeterminate in `variables`.
    /// This is a generalization of Horner's method for polynomials.
    ///
    /// If no variables are provided, a heuristically determined variable ordering is used
    /// that minimizes the number of operations.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> expr = E('v1 + v1*v2 + 2 v1*v2*v3 + v1^2 + v1^3*y + v1^4*z')
    /// >>> collected = expr.hold(T().collect_horner([S('v1'), S('v2')]))()
    ///
    /// yields `v1*(1+v1*(1+v1*(v1*z+y))+v2*(1+2*v3))`.
    #[pyo3(signature = (vars=None))]
    pub fn collect_horner(
        &self,
        vars: Option<Vec<PythonExpression>>,
    ) -> PyResult<PythonTransformer> {
        let vars = if let Some(vars) = vars {
            let vars: Vec<_> = vars
                .into_iter()
                .map(|e| Indeterminate::try_from(e.expr))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            Some(vars)
        } else {
            None
        };

        self.append_transformer(Transformer::CollectHorner(vars))
    }

    /// Create a transformer that collects numerical factors by removing the numerical content from additions.
    /// For example, `-2*x + 4*x^2 + 6*x^3` will be transformed into `-2*(x - 2*x^2 - 3*x^3)`.
    ///
    /// The first argument of the addition is normalized to a positive quantity.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>>
    /// >>> x, y = S('x', 'y')
    /// >>> e = (-3*x+6*y)(2*x+2*y)
    /// >>> print(Transformer().collect_num()(e))
    ///
    /// yields
    ///
    /// ```log
    /// -6*(x-2*y)*(x+y)
    /// ```
    pub fn collect_num(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::CollectNum)
    }

    /// Complex conjugate the expression.
    pub fn conjugate(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Conjugate)
    }

    /// Create a transformer that collects terms involving the literal occurrence of `x`.
    pub fn coefficient(&self, x: ConvertibleToExpression) -> PyResult<PythonTransformer> {
        let a = x.to_expression().expr;
        self.append_transformer(Transformer::Map(Box::new(move |i, _state, o| {
            *o = i.coefficient(a.as_view());
            Ok(())
        })))
    }

    /// Create a transformer that computes the partial fraction decomposition in `x`.
    pub fn apart(&self, x: PythonExpression) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Map(Box::new(move |i, _state, o| {
            let poly = i
                .try_to_rational_polynomial::<_, _, u32>(&Q, &Z, None)
                .map_err(|e| {
                    TransformerError::ValueError(format!(
                        "Could not convert expression to rational polynomial: {e}",
                    ))
                })?;

            let x = poly
                .get_variables()
                .iter()
                .position(|v| match (v, x.expr.as_view()) {
                    (PolyVariable::Symbol(y), AtomView::Var(vv)) => *y == vv.get_symbol(),
                    (PolyVariable::Function(_, f) | PolyVariable::Power(f), a) => f.as_view() == a,
                    _ => false,
                })
                .ok_or(TransformerError::ValueError(format!(
                    "Variable {} not found in polynomial",
                    x.expr
                )))?;

            let fs = poly.apart(x);

            Workspace::get_local().with(|ws| {
                let mut res = ws.new_atom();
                let a = res.to_add();
                for f in fs {
                    a.extend(f.to_expression().as_view());
                }

                res.as_view().normalize(ws, o);
            });

            Ok(())
        })))
    }

    /// Create a transformer that writes the expression over a common denominator.
    pub fn together(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Map(Box::new(|i, _state, o| {
            let poly = i
                .try_to_rational_polynomial::<_, _, u32>(&Q, &Z, None)
                .map_err(|e| {
                    TransformerError::ValueError(format!(
                        "Could not convert expression to rational polynomial: {e}",
                    ))
                })?;
            *o = poly.to_expression();
            Ok(())
        })))
    }

    /// Create a transformer that cancels common factors between numerators and denominators.
    /// Any non-canceling parts of the expression will not be rewritten.
    pub fn cancel(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Map(Box::new(|i, _state, o| {
            *o = i.cancel();
            Ok(())
        })))
    }

    /// Create a transformer that factors the expression over the rationals.
    pub fn factor(&self) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Map(Box::new(|i, _state, o| {
            *o = i.factor();
            Ok(())
        })))
    }

    /// Create a transformer that derives `self` w.r.t the variable `x`.
    pub fn derivative(&self, x: PythonExpression) -> PyResult<PythonTransformer> {
        let id = x.expr.try_into().map_err(|e| {
            exceptions::PyValueError::new_err(format!(
                "Derivative must be taken wrt a variable: {e}"
            ))
        })?;

        self.append_transformer(Transformer::Derivative(id))
    }

    /// Create a transformer that series expands in `x` around `expansion_point` to depth `depth`.
    ///
    /// Examples
    /// -------
    /// >>> from symbolica import *
    /// >>> x, y = S('x', 'y')
    /// >>> f = S('f')
    /// >>>
    /// >>> e = 2* x**2 * y + f(x)
    /// >>> e = e.series(x, 0, 2)
    /// >>>
    /// >>> print(e)
    ///
    /// yields `f(0)+x*der(1,f,0)+1/2*x^2*(der(2,f,0)+4*y)`.
    #[pyo3(signature = (x, expansion_point, depth, depth_denom = 1, depth_is_absolute = true))]
    pub fn series(
        &self,
        x: PythonExpression,
        expansion_point: ConvertibleToExpression,
        depth: i64,
        depth_denom: i64,
        depth_is_absolute: bool,
    ) -> PyResult<PythonTransformer> {
        let id = x.expr.try_into().map_err(|e| {
            exceptions::PyValueError::new_err(format!(
                "Derivative must be taken wrt a variable: {e}",
            ))
        })?;

        self.append_transformer(Transformer::Series(
            id,
            expansion_point.to_expression().expr.clone(),
            if depth_is_absolute {
                crate::poly::series::SeriesDepth::absolute((depth, depth_denom))
            } else {
                crate::poly::series::SeriesDepth::relative((depth, depth_denom))
            },
        ))
    }

    /// Create a transformer that replaces all subexpressions matching the pattern `pat` by the right-hand side `rhs`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> x, w1_, w2_ = S('x','w1_','w2_')
    /// >>> f = S('f')
    /// >>> t = T().replace(f(w1_, w2_), f(w1_ - 1, w2_**2), w1_ >= 1)
    /// >>> r = t(f(3,x))
    /// >>> print(r)
    ///
    /// Parameters
    /// ----------
    /// pat:
    ///     The pattern to match.
    /// rhs:
    ///     The right-hand side to replace the matched subexpression with. Can be a transformer, expression or a function that maps a dictionary of wildcards to an expression.
    /// cond:
    ///     Conditions on the pattern.
    /// non_greedy_wildcards:
    ///     Wildcards that try to match as little as possible.
    /// level_range:
    ///     Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
    /// level_is_tree_depth:
    ///     If set to `True`, the level is increased when going one level deeper in the expression tree.
    /// allow_new_wildcards_on_rhs:
    ///     If set to `True`, allow wildcards that do not appear in the pattern on the right-hand side.
    /// rhs_cache_size: int, optional
    ///     Cache the first `rhs_cache_size` substituted patterns. If set to `None`, an internally determined cache size is used.
    ///     **Warning**: caching should be disabled (`rhs_cache_size=0`) if the right-hand side contains side effects, such as updating a global variable.
    #[pyo3(signature = (lhs, rhs, cond = None, non_greedy_wildcards = None, min_level=0, max_level=None, level_range = None, level_is_tree_depth = false, partial=true, allow_new_wildcards_on_rhs = false, rhs_cache_size = None, once = false, bottom_up = false, nested = false))]
    pub fn replace(
        &self,
        lhs: ConvertibleToExpression,
        rhs: ConvertibleToReplaceWith,
        cond: Option<ConvertibleToPatternRestriction>,
        non_greedy_wildcards: Option<Vec<PythonExpression>>,
        min_level: usize,
        max_level: Option<usize>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: bool,
        partial: bool,
        allow_new_wildcards_on_rhs: bool,
        rhs_cache_size: Option<usize>,
        once: bool,
        bottom_up: bool,
        nested: bool,
    ) -> PyResult<PythonTransformer> {
        let mut settings = MatchSettings::cached();

        if let Some(ngw) = non_greedy_wildcards {
            settings.non_greedy_wildcards = ngw
                .iter()
                .map(|x| match x.expr.as_view() {
                    AtomView::Var(v) => {
                        let name = v.get_symbol();
                        if v.get_wildcard_level() == 0 {
                            return Err(exceptions::PyTypeError::new_err(
                                "Only wildcards can be restricted.",
                            ));
                        }
                        Ok(name)
                    }
                    _ => Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    )),
                })
                .collect::<Result<_, _>>()?;
        }

        settings.level_range = level_range.unwrap_or((min_level, max_level));
        settings.partial = partial;
        settings.level_is_tree_depth = level_is_tree_depth;
        settings.allow_new_wildcards_on_rhs = allow_new_wildcards_on_rhs;

        if let Some(rhs_cache_size) = rhs_cache_size {
            settings.rhs_cache_size = rhs_cache_size;
        }

        self.append_transformer(Transformer::ReplaceAll(
            lhs.to_expression().expr.to_pattern(),
            rhs.to_replace_with()?,
            cond.map(|r| r.0).unwrap_or_default(),
            settings,
            ReplaceSettings {
                once,
                bottom_up,
                nested,
            },
        ))
    }

    /// Create a transformer that replaces all atoms matching the patterns. See `replace` for more information.
    ///
    /// Examples
    /// --------
    ///
    /// >>> x, y, f = S('x', 'y', 'f')
    /// >>> e = f(x,y)
    /// >>> r = e.hold(T().replace_multiple([Replacement(x, y), Replacement(y, x)]))
    #[pyo3(signature = (replacements, once = false, bottom_up = false, nested = false))]
    pub fn replace_multiple(
        &self,
        replacements: Vec<PythonReplacement>,
        once: bool,
        bottom_up: bool,
        nested: bool,
    ) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::ReplaceAllMultiple(
            replacements.into_iter().map(|r| r.replacement).collect(),
            ReplaceSettings {
                once,
                bottom_up,
                nested,
            },
        ))
    }

    /// Create a transformer that prints the expression.
    ///
    /// Examples
    /// --------
    /// >>> E('f(10)').hold(T().print(terms_on_new_line = True))()
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
    pub fn print(
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
    ) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Print(PrintOptions {
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
        }))
    }

    /// Print statistics of a transformer, tagging it with `tag`.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S('f')
    /// >>> e = E("f(5)")
    /// >>> e = e.hold(T().stats('replace', T().replace(f(x_), 1)))()
    ///
    /// yields
    /// ```log
    /// Stats for replace:
    ///     In  │ 1 │  10.00 B │
    ///     Out │ 1 │   3.00 B │ ⧗ 40.15µs
    /// ```
    #[pyo3(signature =
        (tag,
            transformer,
            color_medium_change_threshold = Some(10.),
            color_large_change_threshold = Some(100.))
        )]
    pub fn stats(
        &self,
        tag: String,
        transformer: PythonTransformer,
        color_medium_change_threshold: Option<f64>,
        color_large_change_threshold: Option<f64>,
    ) -> PyResult<PythonTransformer> {
        self.append_transformer(Transformer::Stats(
            StatsOptions {
                tag,
                color_medium_change_threshold,
                color_large_change_threshold,
            },
            transformer.chain,
        ))
    }
}

/// A Symbolica expression.
///
/// Supports standard arithmetic operations, such
/// as addition and multiplication.
///
/// Examples
/// --------
/// >>> x = S('x')
/// >>> e = x**2 + 2 - x + 1 / x**4
/// >>> print(e)
///
/// Attributes
/// ----------
/// E: Expression
///     Euler's number `e`, approximately `2.7182`.
/// PI: Expression
///     The mathematical constant `π`, approximately `3.1415`.
/// EULER_GAMMA: Expression
///     The Euler-Mascheroni constant `γ`, approximately `0.57721`.
/// I: Expression
///     The mathematical constant `i`, where `i^2 = -1`.
/// COEFF: Expression
///     The built-in function that converts a rational polynomial to a coefficient.
/// COS: Expression
///     The built-in cosine function.
/// SIN: Expression
///     The built-in sine function.
/// EXP: Expression
///     The built-in exponential function.
/// LOG: Expression
///     The built-in logarithm function.
/// SQRT: Expression
///     The built-in square root function.
/// ABS: Expression
///    The built-in absolute value function.
/// CONJ: Expression
///     The built-in complex conjugate function.
/// IF: Expression
///    The built-in function for piecewise-defined expressions. `IF(cond, true_expr, false_expr)` evaluates to `true_expr` if `cond` is non-zero and `false_expr` otherwise.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(
    from_py_object,
    name = "Expression",
    subclass,
    module = "symbolica.core"
)]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PythonExpression {
    pub expr: Atom,
}

impl From<Atom> for PythonExpression {
    fn from(expr: Atom) -> Self {
        PythonExpression { expr }
    }
}

impl Deref for PythonExpression {
    type Target = Atom;

    fn deref(&self) -> &Self::Target {
        &self.expr
    }
}

/// A restriction on wildcards.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(from_py_object, name = "PatternRestriction", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonPatternRestriction {
    pub condition: Condition<PatternRestriction>,
}

impl From<Condition<PatternRestriction>> for PythonPatternRestriction {
    fn from(condition: Condition<PatternRestriction>) -> Self {
        PythonPatternRestriction { condition }
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonPatternRestriction {
    /// Create a new pattern restriction that is the logical 'and' operation between two restrictions (i.e., both should hold).
    pub fn __and__(&self, other: Self) -> PythonPatternRestriction {
        (self.condition.clone() & other.condition.clone()).into()
    }

    /// Create a new pattern restriction that is the logical 'or' operation between two restrictions (i.e., one of the two should hold).
    pub fn __or__(&self, other: Self) -> PythonPatternRestriction {
        (self.condition.clone() | other.condition.clone()).into()
    }

    /// Create a new pattern restriction that takes the logical 'not' of the current restriction.
    pub fn __invert__(&self) -> PythonPatternRestriction {
        (!self.condition.clone()).into()
    }

    /// Create a pattern restriction based on the current matched variables.
    /// `match_fn` is a Python function that takes a dictionary of wildcards and their matched values
    /// and should return an integer. If the integer is less than 0, the restriction is false.
    /// If the integer is 0, the restriction is inconclusive.
    /// If the integer is greater than 0, the restriction is true.
    ///
    /// If your pattern restriction cannot decide if it holds since not all the required variables
    /// have been matched, it should return inclusive (0).
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> f, x_, y_, z_ = S('f', 'x_', 'y_', 'z_')
    /// >>>
    /// >>> def filter(m: dict[Expression, Expression]) -> int:
    /// >>>    if x_ in m and y_ in m:
    /// >>>        if m[x_] > m[y_]:
    /// >>>            return -1  # no match
    /// >>>        if z_ in m:
    /// >>>            if m[y_] > m[z_]:
    /// >>>                return -1
    /// >>>            return 1  # match
    /// >>>
    /// >>>    return 0  # inconclusive
    /// >>>
    /// >>>
    /// >>> e = f(1, 2, 3).replace(f(x_, y_, z_), 1,
    /// >>>         PatternRestriction.req_matches(filter))
    #[classmethod]
    pub fn req_matches(
        _cls: &Bound<'_, PyType>,
        #[gen_stub(override_type(
            type_repr = "typing.Callable[[dict[Expression, Expression]], int]"
        ))]
        match_fn: Py<PyAny>,
    ) -> PyResult<PythonPatternRestriction> {
        Ok(PythonPatternRestriction {
            condition: PatternRestriction::MatchStack(Box::new(move |m| {
                let matches: HashMap<PythonExpression, PythonExpression> = m
                    .get_matches()
                    .iter()
                    .map(|(s, t)| (Atom::var(*s).into(), t.to_atom().into()))
                    .collect();

                let r = Python::attach(|py| {
                    match_fn
                        .call(py, (matches,), None)
                        .expect("Bad callback function")
                        .extract::<isize>(py)
                        .expect("Pattern comparison does not return an integer")
                });

                if r < 0 {
                    false.into()
                } else if r == 0 {
                    ConditionResult::Inconclusive
                } else {
                    true.into()
                }
            }))
            .into(),
        })
    }
}

/// A restriction on wildcards.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(from_py_object, name = "Condition", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonCondition {
    pub condition: Condition<Relation>,
}

impl From<Condition<Relation>> for PythonCondition {
    fn from(condition: Condition<Relation>) -> Self {
        PythonCondition { condition }
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonCondition {
    /// Return a string representation of the condition.
    pub fn __repr__(&self) -> String {
        format!("{:?}", self.condition)
    }

    /// Return a string representation of the condition.
    pub fn __str__(&self) -> String {
        format!("{}", self.condition)
    }

    /// Evaluate the condition.
    pub fn eval(&self) -> PyResult<bool> {
        Ok(self
            .condition
            .evaluate(&None)
            .map_err(exceptions::PyValueError::new_err)?
            == ConditionResult::True)
    }

    /// Return the boolean value of the condition.
    pub fn __bool__(&self) -> PyResult<bool> {
        self.eval()
    }

    /// Create a new pattern restriction that is the logical 'and' operation between two restrictions (i.e., both should hold).
    pub fn __and__(&self, other: Self) -> PythonCondition {
        (self.condition.clone() & other.condition.clone()).into()
    }

    /// Create a new pattern restriction that is the logical 'or' operation between two restrictions (i.e., one of the two should hold).
    pub fn __or__(&self, other: Self) -> PythonCondition {
        (self.condition.clone() | other.condition.clone()).into()
    }

    /// Create a new pattern restriction that takes the logical 'not' of the current restriction.
    pub fn __invert__(&self) -> PythonCondition {
        (!self.condition.clone()).into()
    }

    /// Convert the condition to a pattern restriction.
    pub fn to_req(&self) -> PyResult<PythonPatternRestriction> {
        self.condition
            .clone()
            .try_into()
            .map(|e| PythonPatternRestriction { condition: e })
            .map_err(exceptions::PyValueError::new_err)
    }
}

macro_rules! req_cmp_rel {
    ($self:ident,$num:ident,$cmp_any_atom:ident,$c:ident) => {{
        let num = if !$cmp_any_atom {
            if let Pattern::Literal(a) = $num {
                if let AtomView::Num(_) = a.as_view() {
                    a
                } else {
                    return Err("Can only compare to number");
                }
            } else {
                return Err("Can only compare to number");
            }
        } else if let Pattern::Literal(a) = $num {
            a
        } else {
            return Err("Pattern must be literal");
        };

        if let Pattern::Wildcard(name) = $self {
            if name.get_wildcard_level() == 0 {
                return Err("Only wildcards can be restricted.");
            }

            Ok(PatternRestriction::Wildcard((
                name,
                WildcardRestriction::Filter(Box::new(move |v: &Match| {
                    if let Match::Single(m) = v {
                        if !$cmp_any_atom {
                            if let AtomView::Num(_) = m {
                                return m.cmp(&num.as_view()).$c();
                            }
                        } else {
                            return m.cmp(&num.as_view()).$c();
                        }
                    }

                    false
                })),
            )))
        } else {
            Err("Only wildcards can be restricted.")
        }
    }};
}

impl TryFrom<Relation> for PatternRestriction {
    type Error = &'static str;

    fn try_from(value: Relation) -> Result<Self, &'static str> {
        match value {
            Relation::Eq(atom, atom1) => {
                req_cmp_rel!(atom, atom1, true, is_eq)
            }
            Relation::Ne(atom, atom1) => {
                req_cmp_rel!(atom, atom1, true, is_ne)
            }
            Relation::Gt(atom, atom1) => {
                req_cmp_rel!(atom, atom1, true, is_gt)
            }
            Relation::Ge(atom, atom1) => {
                req_cmp_rel!(atom, atom1, true, is_ge)
            }
            Relation::Lt(atom, atom1) => {
                req_cmp_rel!(atom, atom1, true, is_lt)
            }
            Relation::Le(atom, atom1) => {
                req_cmp_rel!(atom, atom1, true, is_le)
            }
            Relation::Contains(atom, atom1) => {
                if let Pattern::Wildcard(name) = atom {
                    if name.get_wildcard_level() == 0 {
                        return Err("Only wildcards can be restricted.");
                    }

                    if !matches!(&atom1, &Pattern::Literal(_)) {
                        return Err("Pattern must be literal");
                    }

                    Ok(PatternRestriction::Wildcard((
                        name,
                        WildcardRestriction::Filter(Box::new(move |m| {
                            let val = if let Pattern::Literal(a) = &atom1 {
                                a.as_view()
                            } else {
                                unreachable!()
                            };
                            match m {
                                Match::Single(v) => v.contains(val),
                                Match::Multiple(_, v) => v.iter().any(|x| x.contains(val)),
                                Match::FunctionName(_) => false,
                            }
                        })),
                    )))
                } else {
                    Err("LHS must be wildcard")
                }
            }
            Relation::Matches(atom, pattern, cond, settings) => {
                if let Pattern::Wildcard(name) = atom {
                    if name.get_wildcard_level() == 0 {
                        return Err("Only wildcards can be restricted.");
                    }

                    Ok(PatternRestriction::Wildcard((
                        name,
                        WildcardRestriction::Filter(Box::new(move |m| {
                            m.to_atom()
                                .pattern_match(&pattern, Some(&cond), Some(&settings))
                                .next_detailed()
                                .is_some()
                        })),
                    )))
                } else {
                    Err("LHS must be wildcard")
                }
            }
            Relation::IsType(atom, atom_type) => {
                if let Pattern::Wildcard(name) = atom {
                    Ok(PatternRestriction::Wildcard((
                        name,
                        WildcardRestriction::IsAtomType(atom_type),
                    )))
                } else {
                    Err("LHS must be wildcard")
                }
            }
        }
    }
}

impl TryFrom<Condition<Relation>> for Condition<PatternRestriction> {
    type Error = &'static str;

    fn try_from(value: Condition<Relation>) -> Result<Self, &'static str> {
        Ok(match value {
            Condition::True => Condition::True,
            Condition::False => Condition::False,
            Condition::Yield(r) => Condition::Yield(r.try_into()?),
            Condition::And(a) => Condition::And(Box::new((a.0.try_into()?, a.1.try_into()?))),
            Condition::Or(a) => Condition::Or(Box::new((a.0.try_into()?, a.1.try_into()?))),
            Condition::Not(a) => Condition::Not(Box::new((*a).try_into()?)),
        })
    }
}

/// An object that can be converted to a pattern restriction.
pub struct ConvertibleToPatternRestriction(Condition<PatternRestriction>);

impl<'py> FromPyObject<'_, 'py> for ConvertibleToPatternRestriction {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<PythonPatternRestriction>() {
            Ok(ConvertibleToPatternRestriction(a.condition))
        } else if let Ok(a) = ob.extract::<PythonCondition>() {
            Ok(ConvertibleToPatternRestriction(
                a.condition
                    .try_into()
                    .map_err(exceptions::PyValueError::new_err)?,
            ))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to pattern restriction",
            ))
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(ConvertibleToPatternRestriction = PythonPatternRestriction | PythonCondition);

impl<'py> FromPyObject<'_, 'py> for ConvertibleToExpression {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<PythonExpression>() {
            Ok(ConvertibleToExpression(a))
        } else if let Ok(num) = ob.extract::<i64>() {
            Ok(ConvertibleToExpression(Atom::num(num).into()))
        } else if let Ok(num) = ob.cast::<PyInt>() {
            let a = num.to_string();
            let i = Integer::from(rug::Integer::parse(&a).unwrap().complete());
            Ok(ConvertibleToExpression(Atom::num(i).into()))
        } else if ob.extract::<PyBackedStr>().is_ok() {
            // disallow direct string conversion
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to expression",
            ))
        } else if let Ok(f) = ob.extract::<PythonMultiPrecisionFloat>() {
            Ok(ConvertibleToExpression(Atom::num(f.0).into()))
        } else if let Ok(num) = ob.extract::<Complex<f64>>() {
            Ok(ConvertibleToExpression(
                Atom::num(Complex::<Float>::new(num.re.into(), num.im.into())).into(),
            ))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to expression",
            ))
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(
    ConvertibleToExpression =
        PythonExpression | PyInt | PyBackedStr | pyo3::types::PyFloat | Complex64
);

impl<'py> FromPyObject<'_, 'py> for Symbol {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if let Ok(a) = ob.extract::<PythonExpression>() {
            match a.expr.as_view() {
                AtomView::Var(v) => Ok(v.get_symbol()),
                e => Err(exceptions::PyTypeError::new_err(format!(
                    "Expected variable instead of {e}",
                ))),
            }
        } else {
            Err(exceptions::PyTypeError::new_err("Not a valid variable"))
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(Symbol = PythonExpression);

impl<'py> FromPyObject<'_, 'py> for PolyVariable {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        PythonExpression::extract(ob)?
            .expr
            .try_into()
            .map_err(|_| exceptions::PyTypeError::new_err("Not a valid polynomial variable"))
    }
}

impl<'py> IntoPyObject<'py> for PolyVariable {
    type Target = PythonExpression;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        PythonExpression::from(self.to_atom()).into_pyobject(py)
    }
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(PolyVariable = PythonExpression);

/// An object that can be converted to an expression.
pub struct ConvertibleToExpression(PythonExpression);

impl ConvertibleToExpression {
    pub fn to_expression(self) -> PythonExpression {
        self.0
    }
}

macro_rules! req_cmp {
    ($self:ident,$num:ident,$cmp_any_atom:ident,$c:ident) => {{
        let num = $num.to_expression();

        if !$cmp_any_atom && !matches!(num.expr.as_view(), AtomView::Num(_)) {
            return Err(exceptions::PyTypeError::new_err(
                "Can only compare to number",
            ));
        };

        match $self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    condition: (
                        name,
                        WildcardRestriction::Filter(Box::new(move |v: &Match| {
                            let k = num.expr.as_view();

                            if let Match::Single(m) = v {
                                if !$cmp_any_atom {
                                    if let AtomView::Num(_) = m {
                                        return m.cmp(&k).$c();
                                    }
                                } else {
                                    return m.cmp(&k).$c();
                                }
                            }

                            false
                        })),
                    )
                        .into(),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }};
}

macro_rules! req_wc_cmp {
    ($self:ident,$other:ident,$cmp_any_atom:ident,$c:ident) => {{
        let id = match $self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }
                name
            }
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only wildcards can be restricted.",
                ));
            }
        };

        let other_id = match $other.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }
                name
            }
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only wildcards can be restricted.",
                ));
            }
        };

        Ok(PythonPatternRestriction {
            condition: (
                id,
                WildcardRestriction::Cmp(
                    other_id,
                    Box::new(move |m1: &Match, m2: &Match| {
                        if let Match::Single(a1) = m1 {
                            if let Match::Single(a2) = m2 {
                                if !$cmp_any_atom {
                                    if let AtomView::Num(_) = a1 {
                                        if let AtomView::Num(_) = a2 {
                                            return a1.cmp(a2).$c();
                                        }
                                    }
                                } else {
                                    return a1.cmp(a2).$c();
                                }
                            }
                        }
                        false
                    }),
                ),
            )
                .into(),
        })
    }};
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonExpression {
    /// Create a new symbol from a `name`. Symbols carry information about their attributes.
    /// The symbol can signal that it is symmetric if it is used as a function
    /// using `is_symmetric=True`, antisymmetric using `is_antisymmetric=True`,
    /// cyclesymmetric using `is_cyclesymmetric=True` and
    /// multilinear using `is_linear=True`. If no attributes
    /// are specified, the attributes are inherited from the symbol if it was already defined,
    /// otherwise all attributes are set to `false`.  A transformer that is executed
    /// after normalization can be defined with `normalization`.
    ///
    /// Once attributes are defined on a symbol, they cannot be redefined later.
    ///
    /// Examples
    /// --------
    /// Define a regular symbol and use it as a variable:
    /// >>> x = S('x')
    /// >>> e = x**2 + 5
    /// >>> print(e)
    /// x**2 + 5
    ///
    /// Define a regular symbol and use it as a function:
    /// >>> f = S('f')
    /// >>> e = f(1,2)
    /// >>> print(e)
    /// f(1,2)
    ///
    /// Define a symmetric function:
    /// >>> f = S('f', is_symmetric=True)
    /// >>> e = f(2,1)
    /// >>> print(e)
    /// f(1,2)
    ///
    /// Define a linear and symmetric function:
    /// >>> p1, p2, p3, p4 = S('p1', 'p2', 'p3', 'p4')
    /// >>> dot = S('dot', is_symmetric=True, is_linear=True)
    /// >>> e = dot(p2+2*p3,p1+3*p2-p3)
    /// dot(p1,p2)+2*dot(p1,p3)+3*dot(p2,p2)-dot(p2,p3)+6*dot(p2,p3)-2*dot(p3,p3)
    ///
    ///
    /// Define a custom normalization function:
    /// >>> e = S('real_log', normalization=Transformer().replace(E("x_(exp(x1_))"), E("x1_")))
    /// >>> E("real_log(exp(x)) + real_log(5)")
    #[gen_stub(skip)]
    #[pyo3(signature = (*names,is_symmetric=None,is_antisymmetric=None,is_cyclesymmetric=None,is_linear=None,is_scalar=None,is_real=None,is_integer=None,is_positive=None,tags=None,aliases=None,normalization=None, print=None, derivative=None, series=None, eval=None, data=None))]
    #[classmethod]
    pub fn symbol(
        _cls: &Bound<'_, PyType>,
        py: Python,
        names: &Bound<'_, PyTuple>,
        is_symmetric: Option<bool>,
        is_antisymmetric: Option<bool>,
        is_cyclesymmetric: Option<bool>,
        is_linear: Option<bool>,
        is_scalar: Option<bool>,
        is_real: Option<bool>,
        is_integer: Option<bool>,
        is_positive: Option<bool>,
        tags: Option<Vec<String>>,
        aliases: Option<Vec<String>>,
        normalization: Option<PythonTransformer>,
        print: Option<Py<PyAny>>,
        derivative: Option<Py<PyAny>>,
        series: Option<Py<PyAny>>,
        eval: Option<Py<PyAny>>,
        data: Option<PythonUserData>,
    ) -> PyResult<Py<PyAny>> {
        if names.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "At least one name must be provided",
            ));
        }

        let namespace = DefaultNamespace {
            namespace: get_namespace(py)?.into(),
            data: "",
            file: "".into(),
            line: 0,
        };

        if is_symmetric.is_none()
            && is_antisymmetric.is_none()
            && is_cyclesymmetric.is_none()
            && is_linear.is_none()
            && is_scalar.is_none()
            && is_real.is_none()
            && is_integer.is_none()
            && is_positive.is_none()
            && tags.is_none()
            && aliases.is_none()
            && normalization.is_none()
            && print.is_none()
            && derivative.is_none()
            && series.is_none()
            && eval.is_none()
            && data.is_none()
        {
            if names.len() == 1 {
                let name = names.get_item(0).unwrap().extract::<PyBackedStr>()?;

                let id = Symbol::new(namespace.attach_namespace(&name))
                    .build()
                    .map_err(|e| exceptions::PyTypeError::new_err(e.to_string()))?;
                let r = PythonExpression::from(Atom::var(id));
                return r.into_py_any(py);
            } else {
                let mut result = vec![];
                for a in names {
                    let name = a.extract::<PyBackedStr>()?;
                    let id = Symbol::new(namespace.attach_namespace(&name))
                        .build()
                        .map_err(|e| exceptions::PyTypeError::new_err(e.to_string()))?;

                    let r = PythonExpression::from(Atom::var(id));
                    result.push(r);
                }

                return result.into_py_any(py);
            }
        }

        let count = (is_symmetric == Some(true)) as u8
            + (is_antisymmetric == Some(true)) as u8
            + (is_cyclesymmetric == Some(true)) as u8;

        if count > 1 {
            Err(exceptions::PyValueError::new_err(
                "Function cannot be both symmetric, antisymmetric or cyclesymmetric",
            ))?;
        }

        let mut opts = vec![];

        if let Some(true) = is_symmetric {
            opts.push(SymbolAttribute::Symmetric);
        }

        if let Some(true) = is_antisymmetric {
            opts.push(SymbolAttribute::Antisymmetric);
        }

        if let Some(true) = is_cyclesymmetric {
            opts.push(SymbolAttribute::Cyclesymmetric);
        }

        if let Some(true) = is_linear {
            opts.push(SymbolAttribute::Linear);
        }

        if let Some(true) = is_scalar {
            opts.push(SymbolAttribute::Scalar);
        }

        if let Some(true) = is_real {
            opts.push(SymbolAttribute::Real);
        }

        if let Some(true) = is_integer {
            opts.push(SymbolAttribute::Integer);
        }

        if let Some(true) = is_positive {
            opts.push(SymbolAttribute::Positive);
        }

        if names.len() == 1 {
            let name = names.get_item(0).unwrap().extract::<PyBackedStr>()?;
            let name = namespace.attach_namespace(&name);

            let mut symbol = Symbol::new(name).with_attributes(opts);

            if let Some(f) = normalization {
                symbol = symbol.with_normalization_function(Box::new(
                    move |input: AtomView<'_>, out: &mut Settable<Atom>| {
                        let _ = Workspace::get_local()
                            .with(|ws| {
                                Transformer::execute_chain(
                                    input,
                                    &f.chain,
                                    ws,
                                    &TransformerState::default(),
                                    &mut *out,
                                )
                            })
                            .unwrap();
                    },
                ))
            }

            if let Some(f) = print {
                symbol = symbol.with_print_function(Box::new(
                    move |input: AtomView<'_>, opts: &PrintOptions, state: &PrintState| {
                        Python::attach(|py| {
                            let kwargs = print_options_to_dict(opts, state, py).unwrap();
                            f.call(
                                py,
                                (PythonExpression::from(input.to_owned()),),
                                Some(&kwargs),
                            )
                            .unwrap()
                            .extract::<Option<String>>(py)
                            .unwrap()
                        })
                    },
                ))
            }

            if let Some(f) = derivative {
                symbol = symbol.with_derivative_function(Box::new(
                    move |input: AtomView<'_>, arg: usize, out: &mut Settable<Atom>| {
                        **out = Python::attach(|py| {
                            f.call1(py, (PythonExpression::from(input.to_owned()), arg))
                                .unwrap()
                                .extract::<PythonExpression>(py)
                                .unwrap()
                        })
                        .expr;
                    },
                ))
            }

            if let Some(f) = series {
                symbol =
                    symbol.with_series_function(Box::new(move |args: &[Series<AtomField>]| {
                        Python::attach(|py| {
                            let args = args
                                .iter()
                                .cloned()
                                .map(|series| PythonSeries { series })
                                .collect::<Vec<_>>();
                            f.call1(py, (args,))
                                .unwrap()
                                .extract::<Option<(PythonExpression, PythonExpression)>>(py)
                                .unwrap()
                                .map(|(singular, regularized)| (singular.expr, regularized.expr))
                        })
                    }))
            }

            if let Some(eval) = eval {
                symbol = symbol.with_evaluation_info(
                    PythonEvalSpec::from_py(py, eval)?.into_evaluation_info(),
                );
            }

            if let Some(t) = tags {
                symbol = symbol.with_tags(
                    t.into_iter()
                        .map(|x| {
                            if x.contains("::") {
                                x
                            } else {
                                format!("python::{x}")
                            }
                        })
                        .collect::<Vec<_>>(),
                );
            }

            if let Some(a) = aliases {
                symbol = symbol.with_aliases(a);
            }

            if let Some(t) = data {
                symbol = symbol.with_user_data(t.0);
            }

            let symbol = symbol
                .build()
                .map_err(|e| exceptions::PyTypeError::new_err(e.to_string()))?;

            let r = PythonExpression::from(Atom::var(symbol));
            r.into_py_any(py)
        } else {
            let mut result = vec![];
            for a in names {
                let name = a.extract::<PyBackedStr>()?;
                let name = namespace.attach_namespace(&name);
                let mut symbol = Symbol::new(name).with_attributes(opts.clone());

                if let Some(f) = &normalization {
                    let t = f.chain.clone();
                    symbol = symbol.with_normalization_function(Box::new(
                        move |input: AtomView<'_>, out: &mut Settable<Atom>| {
                            let _ = Workspace::get_local()
                                .with(|ws| {
                                    Transformer::execute_chain(
                                        input,
                                        &t,
                                        ws,
                                        &TransformerState::default(),
                                        &mut *out,
                                    )
                                })
                                .unwrap();
                        },
                    ))
                }

                if let Some(t) = tags.as_ref() {
                    symbol = symbol.with_tags(
                        t.into_iter()
                            .map(|x| {
                                if x.contains("::") {
                                    x.clone()
                                } else {
                                    format!("python::{x}")
                                }
                            })
                            .collect::<Vec<_>>(),
                    );
                }

                let symbol = symbol
                    .build()
                    .map_err(|e| exceptions::PyTypeError::new_err(e.to_string()))?;
                let r = PythonExpression::from(Atom::var(symbol));
                result.push(r);
            }

            result.into_py_any(py)
        }
    }

    /// Create a new Symbolica number from an int, a float, a Decimal, or a string.
    /// A floating point number is kept as a float with the same precision as the input,
    /// but it can also be converted to the smallest rational number given a `relative_error`.
    ///
    /// Examples
    /// --------
    /// >>> e = Expression.num(1) / 2
    /// >>> print(e)
    /// 1/2
    ///
    /// >>> print(Expression.num(1/3))
    /// >>> print(Expression.num(0.33, 0.1))
    /// >>> print(Expression.num('0.333`3'))
    /// >>> print(Expression.num(Decimal('0.1234')))
    /// 3.3333333333333331e-1
    /// 1/3
    /// 3.33e-1
    /// 1.2340e-1
    #[pyo3(signature = (num, relative_error = None))]
    #[classmethod]
    pub fn num(
        _cls: &Bound<'_, PyType>,
        py: Python,
        #[gen_stub(override_type(
            type_repr = "int | float | complex | str | decimal.Decimal",
            imports = ("decimal")
        ))]
        num: Py<PyAny>,
        relative_error: Option<f64>,
    ) -> PyResult<PythonExpression> {
        if let Ok(num) = num.extract::<i64>(py) {
            Ok(Atom::num(num).into())
        } else if let Ok(num) = num.cast_bound::<PyInt>(py) {
            let a = format!("{num}");
            PythonExpression::parse(_cls, py, &a, PythonParseMode::Symbolica, None)
        } else if let Ok(f) = num.extract::<PythonMultiPrecisionFloat>(py) {
            if let Some(relative_error) = relative_error {
                let err = relative_error
                    .try_into()
                    .map_err(exceptions::PyValueError::new_err)?;
                let mut r: Rational = f.0.try_into().map_err(exceptions::PyValueError::new_err)?;
                r = r.round(&err);
                Ok(Atom::num(r).into())
            } else {
                Ok(Atom::num(f.0).into())
            }
        } else if let Ok(f) = num.extract::<Complex<f64>>(py) {
            if let Some(relative_error) = relative_error {
                let err = relative_error
                    .try_into()
                    .map_err(exceptions::PyValueError::new_err)?;
                let r = Rational::try_from(f.re)
                    .map_err(exceptions::PyValueError::new_err)?
                    .round(&err);
                let i = Rational::try_from(f.im)
                    .map_err(exceptions::PyValueError::new_err)?
                    .round(&err);
                Ok(Atom::num(Complex::new(r, i)).into())
            } else {
                Ok(Atom::num(Complex::<Float>::new(f.re.into(), f.im.into())).into())
            }
        } else {
            Err(exceptions::PyValueError::new_err("Not a valid number"))
        }
    }

    /// Euler's number `e`, approximately `2.7182`.
    #[classattr]
    #[pyo3(name = "E")]
    pub fn e() -> PythonExpression {
        Atom::var(Symbol::E).into()
    }

    /// The mathematical constant `π`, approximately `3.1415`.
    #[classattr]
    #[pyo3(name = "PI")]
    pub fn pi() -> PythonExpression {
        Atom::var(Symbol::PI).into()
    }

    /// The Euler-Mascheroni constant `γ`, approximately `0.57721`.
    #[classattr]
    #[pyo3(name = "EULER_GAMMA")]
    pub fn euler_gamma() -> PythonExpression {
        Atom::from(crate::transcendental::euler_gamma()).into()
    }

    /// The mathematical constant `i`, where
    /// `i^2 = -1`.
    #[classattr]
    #[pyo3(name = "I")]
    pub fn i() -> PythonExpression {
        Atom::i().into()
    }

    /// The number that represents infinity: `∞`.
    #[classattr]
    #[pyo3(name = "INFINITY")]
    pub fn inf() -> PythonExpression {
        Atom::num(Coefficient::Infinity(Some(Rational::one().into()))).into()
    }

    /// The number that represents infinity with an unknown complex phase: `⧞`.
    #[classattr]
    #[pyo3(name = "COMPLEX_INFINITY")]
    pub fn cinf() -> PythonExpression {
        Atom::num(Coefficient::Infinity(None)).into()
    }

    /// The number that represents indeterminacy: `¿`.
    #[classattr]
    #[pyo3(name = "INDETERMINATE")]
    pub fn indeterminate() -> PythonExpression {
        Atom::num(Coefficient::Indeterminate).into()
    }

    /// The built-in function that converts a rational polynomial to a coefficient.
    #[classattr]
    #[pyo3(name = "COEFF")]
    pub fn coeff() -> PythonExpression {
        Atom::var(Symbol::COEFF).into()
    }

    /// The built-in cosine function.
    #[classattr]
    #[pyo3(name = "COS")]
    pub fn cos_attr() -> PythonExpression {
        Atom::var(Symbol::COS).into()
    }

    /// The built-in sine function.
    #[classattr]
    #[pyo3(name = "SIN")]
    pub fn sin_attr() -> PythonExpression {
        Atom::var(Symbol::SIN).into()
    }

    /// The built-in exponential function.
    #[classattr]
    #[pyo3(name = "EXP")]
    pub fn exp_attr() -> PythonExpression {
        Atom::var(Symbol::EXP).into()
    }

    /// The built-in logarithm function.
    #[classattr]
    #[pyo3(name = "LOG")]
    pub fn log_attr() -> PythonExpression {
        Atom::var(Symbol::LOG).into()
    }

    /// The built-in square root function.
    #[classattr]
    #[pyo3(name = "SQRT")]
    pub fn sqrt_attr() -> PythonExpression {
        Atom::var(Symbol::SQRT).into()
    }

    /// The built-in absolute value function.
    #[classattr]
    #[pyo3(name = "ABS")]
    pub fn abs_attr() -> PythonExpression {
        Atom::var(Symbol::ABS).into()
    }

    /// The built-in complex conjugate function.
    #[classattr]
    #[pyo3(name = "CONJ")]
    pub fn conj_attr() -> PythonExpression {
        Atom::var(Symbol::CONJ).into()
    }

    /// The built-in if function.
    #[classattr]
    #[pyo3(name = "IF")]
    pub fn if_attr() -> PythonExpression {
        Atom::var(Symbol::IF).into()
    }

    /// Return all defined symbol names (function names and variables).
    #[classmethod]
    pub fn get_all_symbol_names(_cls: &Bound<'_, PyType>) -> PyResult<Vec<String>> {
        Ok(State::symbol_iter().map(|(_, x)| x.to_string()).collect())
    }

    /// Parse a Symbolica expression from a string.
    ///
    /// Parameters
    /// ----------
    /// input: str
    ///     An input string. UTF-8 characters are allowed.
    /// mode: ParseMode
    ///     The parsing mode to use. Use `ParseMode.Mathematica` to parse Mathematica expressions.
    /// default_namespace: str
    ///     The default namespace to use when parsing symbols.
    ///
    /// Examples
    /// --------
    /// >>> e = E('x^2+y+y*4')
    /// >>> print(e)
    /// x^2+5*y
    ///
    /// >>> e = E('Cos[test`x] (2+ 3 I)', mode=ParseMode.Mathematica)
    /// >>> print(e)
    ///
    /// `cos(test::x)(2+3i)`
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If the input is not a valid expression.
    #[pyo3(signature = (input, mode = PythonParseMode::Symbolica, default_namespace = None))]
    #[classmethod]
    pub fn parse(
        _cls: &Bound<'_, PyType>,
        py: Python,
        input: &str,
        mode: PythonParseMode,
        default_namespace: Option<String>,
    ) -> PyResult<PythonExpression> {
        let namespace = if let Some(ns) = default_namespace {
            intern_string(&ns)
        } else {
            get_namespace(py)?
        };

        let e = try_parse!(
            input,
            settings = ParseSettings {
                mode: mode.into(),
                ..ParseSettings::default()
            },
            default_namespace = namespace
        )
        .map_err(exceptions::PyValueError::new_err)?;
        Ok(e.into())
    }

    /// Create a new expression that represents 0.
    #[new]
    pub fn __new__() -> PythonExpression {
        Atom::new().into()
    }

    /// Construct an expression from a serialized state.
    pub fn __setstate__(&mut self, state: Vec<u8>) -> PyResult<()> {
        unsafe {
            self.expr = Atom::from_raw(state);
        }
        Ok(())
    }

    /// Get a serialized version of the expression.
    pub fn __getstate__(&self) -> PyResult<Vec<u8>> {
        Ok(self.expr.clone().into_raw())
    }

    /// Get the default positional arguments for `__new__`.
    pub fn __getnewargs__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        Ok(PyTuple::empty(py))
    }

    /// Copy the expression.
    pub fn __copy__(&self) -> PythonExpression {
        self.expr.clone().into()
    }

    /// Convert the expression into a portable string.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(self
            .expr
            .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new()))
    }

    /// Convert the expression into a human-readable string.
    pub fn __str__(&self) -> PyResult<String> {
        Ok(self.expr.format_string(
            &PrintOptions {
                max_terms: Some(100),
                ..DEFAULT_PRINT_OPTIONS
            },
            PrintState::new(),
        ))
    }

    /// Convert the expression into a canonical string that
    /// is independent on the order of the variables and other
    /// implementation details.
    pub fn to_canonical_string(&self) -> PyResult<String> {
        Ok(self.expr.to_canonical_string())
    }

    pub fn __contains__(&self, expr: &PythonExpression) -> bool {
        self.expr.contains(&expr.expr)
    }

    /// Get the number of bytes that this expression takes up in memory.
    pub fn get_byte_size(&self) -> usize {
        self.expr.as_view().get_byte_size()
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
            max_terms = Some(100),
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
        Ok(format!(
            "{}",
            AtomPrinter::new_with_options(
                self.expr.as_view(),
                PrintOptions {
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
            )
        ))
    }

    /// Convert the expression into a plain string, useful for importing and exporting.
    ///
    /// Examples
    /// --------
    /// >>> a = E('5 + x^2')
    /// >>> print(a.to_plain())
    ///
    /// Yields `5 + x^2`, without any coloring.
    pub fn format_plain(&self) -> PyResult<String> {
        Ok(self
            .expr
            .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new()))
    }

    /// Convert the expression into a LaTeX string.
    ///
    /// Examples
    /// --------
    /// >>> a = E('128378127123 z^(2/3)*w^2/x/y + y^4 + z^34 + x^(x+2)+3/5+f(x,x^2)')
    /// >>> print(a.to_latex())
    ///
    /// Yields `$$z^{34}+x^{x+2}+y^{4}+f(x,x^{2})+128378127123 z^{\\frac{2}{3}} w^{2} \\frac{1}{x} \\frac{1}{y}+\\frac{3}{5}$$`.
    pub fn to_latex(&self) -> PyResult<String> {
        Ok(format!(
            "$${}$$",
            AtomPrinter::new_with_options(self.expr.as_view(), LATEX_PRINT_OPTIONS,)
        ))
    }

    /// Convert the expression into a Typst string.
    ///
    /// Examples
    /// --------
    /// >>> a = E('f(x+2i + 3) * 2 / x')
    /// >>> print(a.to_typst())
    ///
    /// Yields ```(2 op("f")(3+2𝑖+"x"))/"x"```.
    #[pyo3(signature = (show_namespaces = false))]
    pub fn to_typst(&self, show_namespaces: bool) -> PyResult<String> {
        Ok(format!(
            "{}",
            self.expr.printer(PrintOptions {
                hide_all_namespaces: !show_namespaces,
                hide_namespace: None,
                ..PrintOptions::typst()
            })
        ))
    }

    /// Convert the expression into a Sympy-parsable string.
    ///
    /// Examples
    /// --------
    /// >>> from sympy import *
    /// >>> s = sympy.parse_expr(E('x^2+f((1+x)^y)').to_sympy())
    pub fn to_sympy(&self) -> PyResult<String> {
        Ok(format!("{}", self.expr.printer(PrintOptions::sympy())))
    }

    /// Convert the expression into a Mathematica-parsable string.
    ///
    /// Examples
    /// --------
    /// >>> a = E('cos(x+2i + 3)+sqrt(conj(x)) + test::y')
    /// >>> print(a.to_mathematica())
    ///
    /// Yields ```test`y+Cos[x+3+2I]+Sqrt[Conjugate[x]]```.
    #[pyo3(signature = (show_namespaces = true))]
    pub fn to_mathematica(&self, show_namespaces: bool) -> PyResult<String> {
        Ok(format!(
            "{}",
            self.expr.printer(PrintOptions {
                hide_all_namespaces: !show_namespaces,
                hide_namespace: None,
                ..PrintOptions::mathematica()
            })
        ))
    }

    /// Hash the expression.
    pub fn __hash__(&self) -> u64 {
        let mut hasher = ahash::AHasher::default();
        self.expr.hash(&mut hasher);
        hasher.finish()
    }

    /// Save the expression and its state to a binary file.
    /// The data is compressed and the compression level can be set between 0 and 11.
    ///
    /// The expression can be loaded using `Expression.load`.
    ///
    /// Examples
    /// --------
    /// >>> e = E("f(x)+f(y)").expand()
    /// >>> e.save('export.dat')
    #[pyo3(signature = (filename, compression_level=9))]
    pub fn save(&self, filename: &str, compression_level: u32) -> PyResult<()> {
        let f = File::create(filename)
            .map_err(|e| exceptions::PyIOError::new_err(format!("Could not create file: {e}")))?;
        let mut writer = CompressorWriter::new(BufWriter::new(f), 4096, compression_level, 22);

        self.expr
            .as_view()
            .export(&mut writer)
            .map_err(|e| exceptions::PyIOError::new_err(format!("Could not write file: {e}")))
    }

    /// Load an expression and its state from a file. The state will be merged
    /// with the current one. If a symbol has conflicting attributes, the conflict
    /// can be resolved using the renaming function `conflict_fn`.
    ///
    /// Expressions can be saved using `Expression.save`.
    ///
    /// Examples
    /// --------
    /// If `export.dat` contains a serialized expression: `f(x)+f(y)`:
    /// >>> e = Expression.load('export.dat')
    ///
    /// whill yield `f(x)+f(y)`.
    ///
    /// If we have defined symbols in a different order:
    /// >>> y, x = S('y', 'x')
    /// >>> e = Expression.load('export.dat')
    ///
    /// we get `f(y)+f(x)`.
    ///
    /// If we define a symbol with conflicting attributes, we can resolve the conflict
    /// using a renaming function:
    ///
    /// >>> x = S('x', is_symmetric=True)
    /// >>> e = Expression.load('export.dat', lambda x: x + '_new')
    /// print(e)
    ///
    /// will yield `f(x_new)+f(y)`.
    #[pyo3(signature = (filename, conflict_fn=None))]
    #[classmethod]
    pub fn load(
        _cls: &Bound<'_, PyType>,
        filename: &str,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Callable[[str], str]]"))]
        conflict_fn: Option<Py<PyAny>>,
    ) -> PyResult<Self> {
        let f = File::open(filename)
            .map_err(|e| exceptions::PyIOError::new_err(format!("Could not read file: {e}")))?;
        let mut reader = brotli::Decompressor::new(BufReader::new(f), 4096);

        Atom::import(
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
        .map(|a| a.into())
        .map_err(|e| exceptions::PyIOError::new_err(format!("Could not read file: {e}")))
    }

    /// Get the type of the atom.
    pub fn get_type(&self) -> PythonAtomType {
        match self.expr.as_ref() {
            Atom::Num(_) => PythonAtomType::Num,
            Atom::Var(_) => PythonAtomType::Var,
            Atom::Fun(_) => PythonAtomType::Fn,
            Atom::Add(_) => PythonAtomType::Add,
            Atom::Mul(_) => PythonAtomType::Mul,
            Atom::Pow(_) => PythonAtomType::Pow,
            Atom::Zero => PythonAtomType::Num,
        }
    }

    /// Convert the expression to a tree.
    pub fn to_atom_tree(&self) -> PyResult<PythonAtomTree> {
        self.expr.as_view().into()
    }

    /// Get the name of a variable or function if the current atom
    /// is a variable or function, otherwise throw an error.
    pub fn get_name(&self) -> PyResult<String> {
        match self.expr.as_ref() {
            Atom::Var(v) => Ok(v.get_symbol().get_name().to_string()),
            Atom::Fun(f) => Ok(f.get_symbol().get_name().to_string()),
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "The expression {} is not a variable or function",
                self.expr
            ))),
        }
    }

    /// Get the attributes of a variable or function if the current atom
    /// is a variable or function, otherwise throw an error.
    pub fn get_attributes(&self) -> PyResult<Vec<PythonSymbolAttribute>> {
        match self.expr.as_ref() {
            Atom::Var(v) => Ok(v
                .get_symbol()
                .get_attributes()
                .into_iter()
                .map(|a| a.into())
                .collect()),
            Atom::Fun(f) => Ok(f
                .get_symbol()
                .get_attributes()
                .into_iter()
                .map(|a| a.into())
                .collect()),
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "The expression {} is not a variable or function",
                self.expr
            ))),
        }
    }

    /// Get the data of a variable or function if the current atom
    /// is a variable or function, otherwise throw an error.
    /// Optionally, provide a key to access a specific entry in the data map, if
    /// the data is a map.
    #[gen_stub(override_return_type(
        type_repr = "Expression | int | float | complex | str | bytes | dict[Expression | int | float | complex | str, typing.Any] | list[typing.Any]"
    ))]
    #[pyo3(signature = (key=None))]
    pub fn get_symbol_data(
        &self,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[Expression | int | float | complex | str]"
        ))]
        key: Option<Py<PyAny>>,
        py: Python,
    ) -> PyResult<Py<PyAny>> {
        let data = match self.expr.as_ref() {
            Atom::Var(v) => v.get_symbol().get_data(),
            Atom::Fun(f) => f.get_symbol().get_data(),
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "The expression {} is not a variable or function",
                self.expr
            )))?,
        };

        if let Some(key) = key
            && let UserData::Map(map) = data
        {
            let key = key.extract::<PythonUserDataKey>(py)?;
            if let Some(value) = map.get(&key.0) {
                return PythonBorrowedUserData(value).into_py_any(py);
            } else {
                return Err(exceptions::PyKeyError::new_err(format!(
                    "The symbol data does not contain the key '{:?}'",
                    key.0
                )));
            }
        } else {
            PythonBorrowedUserData(data).into_py_any(py)
        }
    }

    /// Get the tags of a variable or function if the current atom
    /// is a variable or function, otherwise throw an error.
    pub fn get_tags(&self) -> PyResult<Vec<String>> {
        match self.expr.as_ref() {
            Atom::Var(v) => Ok(v.get_symbol().get_tags().to_vec()),
            Atom::Fun(f) => Ok(f.get_symbol().get_tags().to_vec()),
            _ => Err(exceptions::PyTypeError::new_err(format!(
                "The expression {} is not a variable or function",
                self.expr
            ))),
        }
    }

    /// Check if the expression is a scalar. Symbols must have the scalar attribute.
    ///
    /// Examples
    /// --------
    /// >>> x = S('x', is_scalar=True)
    /// >>> e = (x +1)**2 + 5
    /// >>> print(e.is_scalar())
    /// True
    pub fn is_scalar(&self) -> bool {
        self.expr.is_scalar()
    }

    /// Check if the expression is real. Symbols must have the real attribute.
    ///
    /// Examples
    /// --------
    /// >>> x = S('x', is_real=True)
    /// >>> e = (x + 1)**2 / 2 + 5
    /// >>> print(e.is_real())
    /// True
    pub fn is_real(&self) -> bool {
        self.expr.is_real()
    }

    /// Check if the expression is integer. Symbols must have the integer attribute.
    ///
    /// Examples
    /// --------
    /// >>> x = S('x', is_integer=True)
    /// >>> e = (x + 1)**2 + 5
    /// >>> print(e.is_integer())
    /// True
    pub fn is_integer(&self) -> bool {
        self.expr.is_integer()
    }

    /// Check if the expression is a positive scalar. Symbols must have the positive attribute.
    ///
    /// Examples
    /// --------
    /// >>> x = S('x', is_positive=True)
    /// >>> e = (x + 1)**2 + 5
    /// >>> print(e.is_positive())
    /// True
    pub fn is_positive(&self) -> bool {
        self.expr.is_positive()
    }

    /// Check if the expression has no infinities and is not indeterminate.
    ///
    /// Examples
    /// --------
    /// >>> e = E('x + x^2 + log(0)')
    /// >>> print(e.is_finite())
    /// False
    pub fn is_finite(&self) -> bool {
        self.expr.is_finite()
    }

    /// Check if the expression is constant, i.e. contains no user-defined symbols or functions.
    ///
    /// Examples
    /// --------
    /// >>> e = E('cos(2 + exp(3)) + 5')
    /// >>> print(e.is_constant())
    /// True
    pub fn is_constant(&self) -> bool {
        self.expr.is_constant()
    }

    /// Add this expression to `other`, returning the result.
    pub fn __add__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let rhs = rhs.to_expression();
        Ok((self.expr.as_ref() + rhs.expr.as_ref()).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __radd__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        self.__add__(rhs)
    }

    /// Subtract `other` from this expression, returning the result.
    pub fn __sub__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        self.__add__(ConvertibleToExpression(rhs.to_expression().__neg__()?))
    }

    /// Subtract this expression from `other`, returning the result.
    pub fn __rsub__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        rhs.to_expression()
            .__add__(ConvertibleToExpression(self.__neg__()?))
    }

    /// Add this expression to `other`, returning the result.
    pub fn __mul__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let rhs = rhs.to_expression();
        Ok((self.expr.as_ref() * rhs.expr.as_ref()).into())
    }

    /// Add this expression to `other`, returning the result.
    pub fn __rmul__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        self.__mul__(rhs)
    }

    /// Divide this expression by `other`, returning the result.
    pub fn __truediv__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let rhs = rhs.to_expression();
        Ok((self.expr.as_ref() / rhs.expr.as_ref()).into())
    }

    /// Divide `other` by this expression, returning the result.
    pub fn __rtruediv__(&self, rhs: ConvertibleToExpression) -> PyResult<PythonExpression> {
        rhs.to_expression()
            .__truediv__(ConvertibleToExpression(self.clone()))
    }

    /// Take `self` to power `exp`, returning the result.
    pub fn __pow__(
        &self,
        exponent: ConvertibleToExpression,
        modulo: Option<i64>,
    ) -> PyResult<PythonExpression> {
        if modulo.is_some() {
            return Err(exceptions::PyValueError::new_err(
                "Optional number argument not supported",
            ));
        }

        let exponent = exponent.to_expression();
        Ok(self.expr.pow(&exponent.expr).into())
    }

    /// Take `base` to power `self`, returning the result.
    pub fn __rpow__(
        &self,
        base: ConvertibleToExpression,
        modulo: Option<i64>,
    ) -> PyResult<PythonExpression> {
        base.to_expression()
            .__pow__(ConvertibleToExpression(self.clone()), modulo)
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __xor__(&self, _rhs: Py<PyAny>) -> PyResult<PythonExpression> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor an expression. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Returns a warning that `**` should be used instead of `^` for taking a power.
    pub fn __rxor__(&self, _rhs: Py<PyAny>) -> PyResult<PythonExpression> {
        Err(exceptions::PyTypeError::new_err(
            "Cannot xor an expression. Did you mean to write a power? Use ** instead, i.e. x**2",
        ))
    }

    /// Negate the current expression, returning the result.
    pub fn __neg__(&self) -> PyResult<PythonExpression> {
        Ok((-self.expr.as_ref()).into())
    }

    /// Return the length of the atom.
    fn __len__(&self) -> usize {
        match self.expr.as_view() {
            AtomView::Add(a) => a.get_nargs(),
            AtomView::Mul(a) => a.get_nargs(),
            AtomView::Fun(a) => a.get_nargs(),
            _ => 1,
        }
    }

    fn __int__(&self) -> PyResult<Integer> {
        if let Ok(f) = Integer::try_from(&self.expr) {
            return Ok(f);
        }

        Err(exceptions::PyTypeError::new_err(format!(
            "Cannot convert {} to float",
            self.expr
        )))
    }

    fn __float__(&self) -> PyResult<f64> {
        if let Ok(f) = f64::try_from(&self.expr) {
            return Ok(f);
        }

        Err(exceptions::PyTypeError::new_err(format!(
            "Cannot convert {} to float",
            self.expr
        )))
    }

    fn __complex__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyComplex>> {
        if let Ok(c) = Complex::<f64>::try_from(&self.expr) {
            return Ok(PyComplex::from_doubles(py, c.re, c.im));
        }

        Err(exceptions::PyTypeError::new_err(format!(
            "Cannot convert {} to complex",
            self.expr
        )))
    }

    /// Create a Symbolica expression or transformer by calling the function with appropriate arguments.
    ///
    /// Examples
    /// -------
    /// >>> x = S('x')
    /// >>> f = S('f')
    /// >>> e = f(3,x)
    /// >>> print(e)
    /// f(3,x)
    #[gen_stub(skip)]
    #[pyo3(signature = (*args,))]
    pub fn __call__(&self, args: &Bound<'_, PyTuple>, py: Python) -> PyResult<Py<PyAny>> {
        let id = match self.expr.as_view() {
            AtomView::Var(v) => v.get_symbol(),
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only symbols can be called as functions",
                ));
            }
        };

        pub enum ExpressionOrTransformer {
            Expression(PythonExpression),
            Transformer(ConvertibleToPattern),
        }

        let mut fn_args = Vec::with_capacity(args.len());

        for arg in args {
            if let Ok(a) = arg.extract::<ConvertibleToExpression>() {
                fn_args.push(ExpressionOrTransformer::Expression(a.to_expression()));
            } else if let Ok(a) = arg.extract::<ConvertibleToPattern>() {
                fn_args.push(ExpressionOrTransformer::Transformer(a));
            } else {
                let msg = format!("Unknown type: {}", arg.get_type().name().unwrap());
                return Err(exceptions::PyTypeError::new_err(msg));
            }
        }

        if fn_args
            .iter()
            .all(|x| matches!(x, ExpressionOrTransformer::Expression(_)))
        {
            // simplify to literal expression
            Workspace::get_local().with(|workspace| {
                let mut fun_b = workspace.new_atom();
                let fun = fun_b.to_fun(id);

                for x in fn_args {
                    if let ExpressionOrTransformer::Expression(a) = x {
                        fun.add_arg(a.expr.as_view());
                    }
                }

                let mut out = Atom::default();
                fun_b.as_view().normalize(workspace, &mut out);

                PythonExpression::from(out).into_py_any(py)
            })
        } else {
            // convert all wildcards back from literals
            let mut transformer_args = Vec::with_capacity(args.len());
            for arg in fn_args {
                match arg {
                    ExpressionOrTransformer::Transformer(t) => {
                        transformer_args.push(t.to_pattern()?.expr);
                    }
                    ExpressionOrTransformer::Expression(a) => {
                        transformer_args.push(a.expr.to_pattern());
                    }
                }
            }

            let p = Pattern::Fn(id, transformer_args);
            PythonHeldExpression::from(p).into_py_any(py)
        }
    }

    /// Compute the cosine of the expression.
    pub fn cos(&self) -> PythonExpression {
        self.expr.cos().into()
    }

    /// Compute the sine of the expression.
    pub fn sin(&self) -> PythonExpression {
        self.expr.sin().into()
    }

    /// Compute the tangent of the expression.
    /// `tan(z)` is meromorphic with simple poles at `pi/2 + k pi`.
    pub fn tan(&self) -> PythonExpression {
        crate::function!(crate::transcendental::tan(), self.expr.clone()).into()
    }

    /// Compute the cotangent of the expression.
    /// `cot(z)` is meromorphic with simple poles at `k pi`.
    pub fn cot(&self) -> PythonExpression {
        crate::function!(crate::transcendental::cot(), self.expr.clone()).into()
    }

    /// Compute the secant of the expression.
    /// `sec(z)` is meromorphic with simple poles at `pi/2 + k pi`.
    pub fn sec(&self) -> PythonExpression {
        crate::function!(crate::transcendental::sec(), self.expr.clone()).into()
    }

    /// Compute the cosecant of the expression.
    /// `csc(z)` is meromorphic with simple poles at `k pi`.
    pub fn csc(&self) -> PythonExpression {
        crate::function!(crate::transcendental::csc(), self.expr.clone()).into()
    }

    /// Compute the inverse sine of the expression.
    /// Uses the principal branch with cuts on `(-infinity, -1]` and `[1, +infinity)`.
    pub fn asin(&self) -> PythonExpression {
        crate::function!(crate::transcendental::asin(), self.expr.clone()).into()
    }

    /// Compute the inverse cosine of the expression.
    /// Uses the principal branch with cuts on `(-infinity, -1]` and `[1, +infinity)`.
    pub fn acos(&self) -> PythonExpression {
        crate::function!(crate::transcendental::acos(), self.expr.clone()).into()
    }

    /// Compute the inverse tangent of the expression.
    /// Uses the principal branch with cuts on `(-i infinity, -i]` and `[i, i infinity)`.
    pub fn atan(&self) -> PythonExpression {
        crate::function!(crate::transcendental::atan(), self.expr.clone()).into()
    }

    /// Compute the inverse cotangent of the expression.
    /// Uses the principal branch with cuts on `(-i infinity, -i]` and `[i, i infinity)`.
    pub fn acot(&self) -> PythonExpression {
        crate::function!(crate::transcendental::acot(), self.expr.clone()).into()
    }

    /// Compute the inverse secant of the expression.
    /// Uses the principal branch with branch cut on `[-1, 1]`.
    pub fn asec(&self) -> PythonExpression {
        crate::function!(crate::transcendental::asec(), self.expr.clone()).into()
    }

    /// Compute the inverse cosecant of the expression.
    /// Uses the principal branch with branch cut on `[-1, 1]`.
    pub fn acsc(&self) -> PythonExpression {
        crate::function!(crate::transcendental::acsc(), self.expr.clone()).into()
    }

    /// Compute the hyperbolic sine of the expression.
    /// `sinh(z)` is entire.
    pub fn sinh(&self) -> PythonExpression {
        crate::function!(crate::transcendental::sinh(), self.expr.clone()).into()
    }

    /// Compute the hyperbolic cosine of the expression.
    /// `cosh(z)` is entire.
    pub fn cosh(&self) -> PythonExpression {
        crate::function!(crate::transcendental::cosh(), self.expr.clone()).into()
    }

    /// Compute the hyperbolic tangent of the expression.
    /// `tanh(z)` is meromorphic with simple poles at `i (pi/2 + k pi)`.
    pub fn tanh(&self) -> PythonExpression {
        crate::function!(crate::transcendental::tanh(), self.expr.clone()).into()
    }

    /// Compute the hyperbolic cotangent of the expression.
    /// `coth(z)` is meromorphic with simple poles at `i k pi`.
    pub fn coth(&self) -> PythonExpression {
        crate::function!(crate::transcendental::coth(), self.expr.clone()).into()
    }

    /// Compute the hyperbolic secant of the expression.
    /// `sech(z)` is meromorphic with simple poles at `i (pi/2 + k pi)`.
    pub fn sech(&self) -> PythonExpression {
        crate::function!(crate::transcendental::sech(), self.expr.clone()).into()
    }

    /// Compute the hyperbolic cosecant of the expression.
    /// `csch(z)` is meromorphic with simple poles at `i k pi`.
    pub fn csch(&self) -> PythonExpression {
        crate::function!(crate::transcendental::csch(), self.expr.clone()).into()
    }

    /// Compute the inverse hyperbolic sine of the expression.
    /// Uses the principal branch with cuts on `(-i infinity, -i]` and `[i, i infinity)`.
    pub fn asinh(&self) -> PythonExpression {
        crate::function!(crate::transcendental::asinh(), self.expr.clone()).into()
    }

    /// Compute the inverse hyperbolic cosine of the expression.
    /// Uses the principal branch with branch cut on `(-infinity, 1]`.
    pub fn acosh(&self) -> PythonExpression {
        crate::function!(crate::transcendental::acosh(), self.expr.clone()).into()
    }

    /// Compute the inverse hyperbolic tangent of the expression.
    /// Uses the principal branch with cuts on `(-infinity, -1]` and `[1, +infinity)`.
    pub fn atanh(&self) -> PythonExpression {
        crate::function!(crate::transcendental::atanh(), self.expr.clone()).into()
    }

    /// Compute the inverse hyperbolic cotangent of the expression.
    /// Uses the principal branch with branch cut on `[-1, 1]`.
    pub fn acoth(&self) -> PythonExpression {
        crate::function!(crate::transcendental::acoth(), self.expr.clone()).into()
    }

    /// Compute the inverse hyperbolic secant of the expression.
    /// Uses the principal branch with cuts on `(-infinity, 0]` and `[1, +infinity)`.
    pub fn asech(&self) -> PythonExpression {
        crate::function!(crate::transcendental::asech(), self.expr.clone()).into()
    }

    /// Compute the inverse hyperbolic cosecant of the expression.
    /// Uses the principal branch with branch cut on the imaginary interval `[-i, i]`.
    pub fn acsch(&self) -> PythonExpression {
        crate::function!(crate::transcendental::acsch(), self.expr.clone()).into()
    }

    /// Compute the exponential of the expression.
    pub fn exp(&self) -> PythonExpression {
        self.expr.exp().into()
    }

    /// Compute the natural logarithm of the expression.
    pub fn log(&self) -> PythonExpression {
        self.expr.log().into()
    }

    /// Compute the square root of the expression.
    pub fn sqrt(&self) -> PythonExpression {
        self.expr.sqrt().into()
    }

    /// Compute the absolute value of the expression.
    pub fn abs(&self) -> PythonExpression {
        self.expr.abs().into()
    }

    /// Compute the Riemann zeta function symbol `zeta`.
    /// `zeta(s)` is meromorphic with a simple pole at `s = 1` and no branch cuts.
    pub fn zeta(&self) -> PythonExpression {
        crate::function!(crate::transcendental::zeta(), self.expr.clone()).into()
    }

    /// Compute the gamma function of the expression.
    /// `gamma(z)` is meromorphic with simple poles at the non-positive integers.
    pub fn gamma(&self) -> PythonExpression {
        crate::function!(crate::transcendental::gamma(), self.expr.clone()).into()
    }

    /// Compute the polygamma function of order `n` at the expression.
    /// For fixed non-negative integer `n`, this is meromorphic with poles at the non-positive integers.
    pub fn polygamma(&self, n: ConvertibleToExpression) -> PythonExpression {
        crate::function!(
            crate::transcendental::polygamma(),
            n.to_expression().expr,
            self.expr.clone()
        )
        .into()
    }

    /// Compute the polylogarithm of order `s` at the expression.
    /// Uses the principal branch in `z`, with the standard branch cut on `[1, +infinity)`.
    pub fn polylog(&self, s: ConvertibleToExpression) -> PythonExpression {
        crate::function!(
            crate::transcendental::polylog(),
            s.to_expression().expr,
            self.expr.clone()
        )
        .into()
    }

    /// Compute the cylindrical Bessel function of the first kind of order `nu` at the expression.
    /// For fixed `nu`, `bessel_j(nu, z)` is entire in `z`.
    pub fn bessel_j(&self, nu: ConvertibleToExpression) -> PythonExpression {
        let a = nu.to_expression().expr;
        crate::function!(crate::transcendental::bessel_j(), a, self.expr.clone()).into()
    }

    /// Compute the cylindrical Bessel function of the second kind of order `nu` at the expression.
    /// Uses the principal branch in `z`, with branch cut on `(-infinity, 0]`.
    pub fn bessel_y(&self, nu: ConvertibleToExpression) -> PythonExpression {
        crate::function!(
            crate::transcendental::bessel_y(),
            nu.to_expression().expr,
            self.expr.clone()
        )
        .into()
    }

    /// Compute the modified Bessel function of the first kind of order `nu` at the expression.
    /// For fixed `nu`, `bessel_i(nu, z)` is entire in `z`.
    pub fn bessel_i(&self, nu: ConvertibleToExpression) -> PythonExpression {
        crate::function!(
            crate::transcendental::bessel_i(),
            nu.to_expression().expr,
            self.expr.clone()
        )
        .into()
    }

    /// Compute the modified Bessel function of the second kind of order `nu` at the expression.
    /// Uses the principal branch in `z`, with branch cut on `(-infinity, 0]`.
    pub fn bessel_k(&self, nu: ConvertibleToExpression) -> PythonExpression {
        crate::function!(
            crate::transcendental::bessel_k(),
            nu.to_expression().expr,
            self.expr.clone()
        )
        .into()
    }

    /// Take the complex conjugate of this expression, returning the result.
    ///
    /// Examples
    /// --------
    /// >>> e = E('x+2 + 3^x + (5+2i) * (test::{real}::real) + (-2)^x')
    /// >>> print(e.conj())
    ///
    /// Yields `(5-2𝑖)*real+3^conj(x)+conj(x)+conj((-2)^x)+2`.
    pub fn conj(&self) -> PythonExpression {
        self.expr.conj().into()
    }

    /// Create a held expression that delays the execution of the transformer `t` until the
    /// resulting held expression is called. Held expressions can be composed like regular expressions
    /// and are useful for the right-hand side of pattern matching, to act a transformer
    /// on a wildcard *after* it has been substituted.
    ///
    /// Examples
    /// -------
    /// >>> f, x, x_ = S('f', 'x', 'x_')
    /// >>> e = f((x+1)**2)
    /// >>> e = e.replace(f(x_), f(x_.hold(T().expand())))
    pub fn hold(&self, t: PythonTransformer) -> PyResult<PythonHeldExpression> {
        Ok(Pattern::Transformer(Box::new((Some(self.expr.to_pattern()), t.chain))).into())
    }

    /// Get the `idx`th component of the expression.
    fn __getitem__(&self, idx: isize) -> PyResult<PythonExpression> {
        let slice = match self.expr.as_view() {
            AtomView::Add(a) => a.to_slice(),
            AtomView::Mul(m) => m.to_slice(),
            AtomView::Fun(f) => f.to_slice(),
            AtomView::Pow(p) => p.to_slice(),
            _ => Err(PyIndexError::new_err("Cannot access child of leaf node"))?,
        };

        if idx.unsigned_abs() < slice.len() {
            Ok(if idx < 0 {
                slice
                    .get(slice.len() - idx.unsigned_abs())
                    .to_owned()
                    .into()
            } else {
                slice.get(idx as usize).to_owned().into()
            })
        } else {
            Err(PyIndexError::new_err(format!(
                "Index {} out of bounds: the atom only has {} children.",
                idx,
                slice.len(),
            )))
        }
    }

    /// Returns true iff `self` contains `a` literally.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x, y, z = S('x', 'y', 'z')
    /// >>> e = x * y * z
    /// >>> e.contains(x) # True
    /// >>> e.contains(x*y*z) # True
    /// >>> e.contains(x*y) # False
    pub fn contains(&self, s: ConvertibleToOpenPattern) -> PyResult<PythonCondition> {
        Ok(PythonCondition {
            condition: Condition::Yield(Relation::Contains(
                self.expr.to_pattern(),
                s.to_pattern()?.expr,
            )),
        })
    }

    /// Get all symbols in the current expression, optionally including function symbols.
    /// The symbols are sorted in Symbolica's internal ordering.
    #[pyo3(signature = (include_function_symbols = true))]
    pub fn get_all_symbols(&self, include_function_symbols: bool) -> Vec<PythonExpression> {
        let mut s: Vec<PythonExpression> = self
            .expr
            .get_all_symbols(include_function_symbols)
            .into_iter()
            .map(|x| Atom::var(x).into())
            .collect();
        s.sort_by(|x, y| x.expr.cmp(&y.expr));
        s
    }

    /// Get all symbols and functions in the current expression, optionally considering function arguments as well.
    /// The symbols are sorted in Symbolica's internal ordering.
    #[pyo3(signature = (enter_functions = true))]
    pub fn get_all_indeterminates(&self, enter_functions: bool) -> Vec<PythonExpression> {
        let mut s: Vec<PythonExpression> = self
            .expr
            .get_all_indeterminates(enter_functions)
            .into_iter()
            .map(|x| x.to_owned().into())
            .collect();
        s.sort_by(|x, y| x.expr.cmp(&y.expr));
        s
    }

    /// Convert all coefficients to floats with a given precision `decimal_prec`.
    /// The precision of floating point coefficients in the input will be truncated to `decimal_prec`.
    #[pyo3(signature = (decimal_prec = 16))]
    pub fn to_float(&self, decimal_prec: u32) -> PythonExpression {
        self.expr.to_float(decimal_prec).into()
    }

    /// Map all floating point and rational coefficients to the best rational approximation
    /// in the interval `[self*(1-relative_error),self*(1+relative_error)]`.
    #[pyo3(signature = (relative_error = 0.01))]
    pub fn rationalize(&self, relative_error: f64) -> PyResult<PythonExpression> {
        if relative_error <= 0. || relative_error > 1. {
            return Err(exceptions::PyValueError::new_err(
                "Relative error must be between 0 and 1",
            ));
        }

        Ok(self
            .expr
            .rationalize(
                &relative_error
                    .try_into()
                    .map_err(exceptions::PyValueError::new_err)?,
            )
            .into())
    }

    /// Create a pattern restriction based on the wildcard length before downcasting.
    #[pyo3(signature = (min_length, max_length=None))]
    pub fn req_len(
        &self,
        min_length: usize,
        max_length: Option<usize>,
    ) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    condition: (name, WildcardRestriction::Length(min_length, max_length)).into(),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction that tests the type of the atom.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x, x_ = S('x', 'x_')
    /// >>> f = S("f")
    /// >>> e = f(x)*f(2)*f(f(3))
    /// >>> e = e.replace(f(x_), 1, x_.req_type(AtomType.Num))
    /// >>> print(e)
    ///
    /// Yields `f(x)*f(1)`.
    pub fn req_type(&self, atom_type: PythonAtomType) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    condition: (
                        name,
                        WildcardRestriction::IsAtomType(match atom_type {
                            PythonAtomType::Num => AtomType::Num,
                            PythonAtomType::Var => AtomType::Var,
                            PythonAtomType::Add => AtomType::Add,
                            PythonAtomType::Mul => AtomType::Mul,
                            PythonAtomType::Pow => AtomType::Pow,
                            PythonAtomType::Fn => AtomType::Fun,
                        }),
                    )
                        .into(),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction based on the tag of a matched variable or function.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x = S('x', tags=['a', 'b'])
    /// >>> x_ = S('x_')
    /// >>> e = x.replace(x_, 1, x_.req_tag('b'))
    /// >>> print(e)
    /// Yields `1`.
    pub fn req_tag(&self, tag: &str) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                if tag.contains("::") {
                    Ok(PythonPatternRestriction {
                        condition: (name.filter_tag(tag.to_string())).into(),
                    })
                } else {
                    Ok(PythonPatternRestriction {
                        condition: (name.filter_tag(format!("python::{tag}"))).into(),
                    })
                }
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction based on the attribute of a matched variable or function.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x = S('f', is_linear=True)
    /// >>> x_ = S('x_')
    /// >>> print(E('f(x)').replace(E('x_(x)'), 1, ~S('x_').req_attr(SymbolAttribute.Linear)))
    /// >>> print(e)
    ///
    /// Yields `f(x)`.
    pub fn req_attr(&self, attribute: PythonSymbolAttribute) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                let f = move |s: Symbol| match attribute {
                    PythonSymbolAttribute::Symmetric => s.is_symmetric(),
                    PythonSymbolAttribute::Antisymmetric => s.is_antisymmetric(),
                    PythonSymbolAttribute::Cyclesymmetric => s.is_cyclesymmetric(),
                    PythonSymbolAttribute::Linear => s.is_linear(),
                    PythonSymbolAttribute::Scalar => s.is_scalar(),
                    PythonSymbolAttribute::Real => s.is_real(),
                    PythonSymbolAttribute::Integer => s.is_integer(),
                    PythonSymbolAttribute::Positive => s.is_positive(),
                };

                Ok(PythonPatternRestriction {
                    condition: name
                        .filter_match(move |m| match m {
                            Match::Single(v) => v.get_symbol().map(|s| f(s)).unwrap_or(false),
                            Match::Multiple(_, _) => false,
                            Match::FunctionName(n) => f(*n),
                        })
                        .into(),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction that filters for expressions that contain `a`.
    pub fn req_contains(&self, a: PythonExpression) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    condition: (
                        name,
                        WildcardRestriction::Filter(Box::new(move |m| match m {
                            Match::Single(v) => v.contains(a.expr.as_view()),
                            Match::Multiple(_, v) => v.iter().any(|x| x.contains(a.expr.as_view())),
                            Match::FunctionName(_) => false,
                        })),
                    )
                        .into(),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Create a pattern restriction that treats the wildcard as a literal variable,
    /// so that it only matches to itself.
    pub fn req_lit(&self) -> PyResult<PythonPatternRestriction> {
        match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }

                Ok(PythonPatternRestriction {
                    condition: (name, WildcardRestriction::IsLiteralWildcard(name)).into(),
                })
            }
            _ => Err(exceptions::PyTypeError::new_err(
                "Only wildcards can be restricted.",
            )),
        }
    }

    /// Test if the expression is of a certain type.
    pub fn is_type(&self, atom_type: PythonAtomType) -> PythonCondition {
        PythonCondition {
            condition: Condition::Yield(Relation::IsType(
                self.expr.to_pattern(),
                match atom_type {
                    PythonAtomType::Num => AtomType::Num,
                    PythonAtomType::Var => AtomType::Var,
                    PythonAtomType::Add => AtomType::Add,
                    PythonAtomType::Mul => AtomType::Mul,
                    PythonAtomType::Pow => AtomType::Pow,
                    PythonAtomType::Fn => AtomType::Fun,
                },
            )),
        }
    }

    /// Compare two expressions. If one of the expressions is not a number, an
    /// internal ordering will be used.
    fn __richcmp__(&self, o: Py<PyAny>, op: CompareOp, py: Python) -> PyResult<PythonCondition> {
        let Ok(other) = o.extract::<ConvertibleToPattern>(py) else {
            return Err(exceptions::PyTypeError::new_err(format!(
                "Cannot compare {} with {} due to incompatible types.",
                self.expr, o
            )));
        };

        Ok(match op {
            CompareOp::Eq => PythonCondition {
                condition: Relation::Eq(self.expr.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Ne => PythonCondition {
                condition: Relation::Ne(self.expr.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Ge => PythonCondition {
                condition: Relation::Ge(self.expr.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Gt => PythonCondition {
                condition: Relation::Gt(self.expr.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Le => PythonCondition {
                condition: Relation::Le(self.expr.to_pattern(), other.to_pattern()?.expr).into(),
            },
            CompareOp::Lt => PythonCondition {
                condition: Relation::Lt(self.expr.to_pattern(), other.to_pattern()?.expr).into(),
            },
        })
    }

    /// Create a pattern restriction that passes when the wildcard is smaller than a number `num`.
    /// If the matched wildcard is not a number, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace(f(x_), 1, x_.req_lt(2))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_lt(
        &self,
        other: ConvertibleToExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_cmp!(self, other, cmp_any_atom, is_lt)
    }

    /// Create a pattern restriction that passes when the wildcard is greater than a number `num`.
    /// If the matched wildcard is not a number, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace(f(x_), 1, x_.req_gt(2))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_gt(
        &self,
        other: ConvertibleToExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_cmp!(self, other, cmp_any_atom, is_gt)
    }

    /// Create a pattern restriction that passes when the wildcard is smaller than or equal to a number `num`.
    /// If the matched wildcard is not a number, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace(f(x_), 1, x_.req_le(2))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_le(
        &self,
        other: ConvertibleToExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_cmp!(self, other, cmp_any_atom, is_le)
    }

    /// Create a pattern restriction that passes when the wildcard is greater than or equal to a number `num`.
    /// If the matched wildcard is not a number, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace(f(x_), 1, x_.req_ge(2))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_ge(
        &self,
        other: ConvertibleToExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_cmp!(self, other, cmp_any_atom, is_ge)
    }

    /// Create a new pattern restriction that calls the function `filter_fn` with the matched
    /// atom that should return a boolean. If true, the pattern matches.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace(f(x_), 1, x_.req(lambda m: m == 2 or m == 3))
    pub fn req(
        &self,
        #[gen_stub(override_type(type_repr = "typing.Callable[[Expression], bool | Condition]"))]
        filter_fn: Py<PyAny>,
    ) -> PyResult<PythonPatternRestriction> {
        let id = match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }
                name
            }
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only wildcards can be restricted.",
                ));
            }
        };

        Ok(PythonPatternRestriction {
            condition: (
                id,
                WildcardRestriction::Filter(Box::new(move |m| {
                    let data: PythonExpression = m.to_atom().into();

                    Python::attach(|py| {
                        filter_fn
                            .call(py, (data,), None)
                            .expect("Bad callback function")
                            .is_truthy(py)
                            .expect("Pattern filter does not return a boolean")
                    })
                })),
            )
                .into(),
        })
    }

    /// Create a pattern restriction that passes when the wildcard is smaller than another wildcard.
    /// If the matched wildcards are not a numbers, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_, y_ = S('x_', 'y_')
    /// >>> f = S("f")
    /// >>> e = f(1,2)
    /// >>> e = e.replace(f(x_,y_), 1, x_.req_cmp_lt(y_))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_cmp_lt(
        &self,
        other: PythonExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_wc_cmp!(self, other, cmp_any_atom, is_lt)
    }

    /// Create a pattern restriction that passes when the wildcard is greater than another wildcard.
    /// If the matched wildcards are not a numbers, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_, y_ = S('x_', 'y_')
    /// >>> f = S("f")
    /// >>> e = f(2,1)
    /// >>> e = e.replace(f(x_,y_), 1, x_.req_cmp_gt(y_))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_cmp_gt(
        &self,
        other: PythonExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_wc_cmp!(self, other, cmp_any_atom, is_gt)
    }

    /// Create a pattern restriction that passes when the wildcard is less than or equal to another wildcard.
    /// If the matched wildcards are not a numbers, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_, y_ = S('x_', 'y_')
    /// >>> f = S("f")
    /// >>> e = f(1,2)
    /// >>> e = e.replace(f(x_,y_), 1, x_.req_cmp_le(y_))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_cmp_le(
        &self,
        other: PythonExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_wc_cmp!(self, other, cmp_any_atom, is_le)
    }

    /// Create a pattern restriction that passes when the wildcard is greater than or equal to another wildcard.
    /// If the matched wildcards are not a numbers, the pattern fails.
    ///
    /// When the option `cmp_any_atom` is set to `True`, this function compares atoms
    /// of any type. The result depends on the internal ordering and may change between
    /// different Symbolica versions.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_, y_ = S('x_', 'y_')
    /// >>> f = S("f")
    /// >>> e = f(2,1)
    /// >>> e = e.replace(f(x_,y_), 1, x_.req_cmp_ge(y_))
    #[pyo3(signature =(other, cmp_any_atom = false))]
    pub fn req_cmp_ge(
        &self,
        other: PythonExpression,
        cmp_any_atom: bool,
    ) -> PyResult<PythonPatternRestriction> {
        req_wc_cmp!(self, other, cmp_any_atom, is_ge)
    }

    /// Create a new pattern restriction that calls the function `cmp_fn` with another the matched
    /// atom and the match atom of the `other` wildcard that should return a boolean. If true, the pattern matches.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x_, y_ = S('x_', 'y_')
    /// >>> f = S("f")
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> e = e.replace(f(x_)*f(y_), 1, x_.req_cmp(y_, lambda m1, m2: m1 + m2 == 4))
    pub fn req_cmp(
        &self,
        other: PythonExpression,
        #[gen_stub(override_type(
            type_repr = "typing.Callable[[Expression, Expression], bool | Condition]"
        ))]
        cmp_fn: Py<PyAny>,
    ) -> PyResult<PythonPatternRestriction> {
        let id = match self.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }
                name
            }
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only wildcards can be restricted.",
                ));
            }
        };

        let other_id = match other.expr.as_view() {
            AtomView::Var(v) => {
                let name = v.get_symbol();
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    ));
                }
                name
            }
            _ => {
                return Err(exceptions::PyTypeError::new_err(
                    "Only wildcards can be restricted.",
                ));
            }
        };

        Ok(PythonPatternRestriction {
            condition: (
                id,
                WildcardRestriction::Cmp(
                    other_id,
                    Box::new(move |m1, m2| {
                        let data1: PythonExpression = m1.to_atom().into();
                        let data2: PythonExpression = m2.to_atom().into();

                        Python::attach(|py| {
                            cmp_fn
                                .call(py, (data1, data2), None)
                                .expect("Bad callback function")
                                .is_truthy(py)
                                .expect("Pattern comparison does not return a boolean")
                        })
                    }),
                ),
            )
                .into(),
        })
    }

    /// Create an iterator over all sub-atoms in the expression.
    fn __iter__(&self) -> PyResult<PythonAtomIterator> {
        match self.expr.as_view() {
            AtomView::Add(_) | AtomView::Mul(_) | AtomView::Fun(_) | AtomView::Pow(_) => {}
            x => {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Non-iterable type: {}",
                    x
                )));
            }
        };

        Ok(PythonAtomIterator::from_expr(self.clone()))
    }

    /// Map the transformations to every term in the expression.
    /// The execution happens in parallel, using `n_cores`.
    ///
    /// Examples
    /// --------
    /// >>> x, x_ = S('x', 'x_')
    /// >>> e = (1+x)**2
    /// >>> r = e.map(Transformer().expand().replace(x, 6))
    /// >>> print(r)
    #[pyo3(signature = (op, n_cores = None, stats_to_file = None))]
    pub fn map(
        &self,
        op: PythonTransformer,
        py: Python,
        n_cores: Option<usize>,
        stats_to_file: Option<String>,
    ) -> PyResult<PythonExpression> {
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
        let r = py.detach(move || {
            self.expr.as_view().map_terms(
                |x| {
                    let mut out = Atom::default();
                    Workspace::get_local().with(|ws| {
                        let _ = Transformer::execute_chain(x, &op.chain, ws, &state, &mut out)
                            .unwrap_or_else(|e| {
                                // TODO: capture and abort the parallel run
                                panic!("Transformer failed during parallel execution: {e:?}")
                            });
                    });
                    out
                },
                n_cores.unwrap_or(1),
            )
        });

        Ok(r.into())
    }

    /// Set the coefficient ring to contain the variables in the `vars` list.
    /// This will move all variables into a rational polynomial function.
    ///
    /// Parameters
    /// ----------
    /// vars: List[Expression]
    ///     A list of variables
    pub fn set_coefficient_ring(&self, vars: Vec<PythonExpression>) -> PyResult<PythonExpression> {
        let mut var_map = vec![];
        for v in vars {
            var_map.push(
                v.expr
                    .try_into()
                    .map_err(|e| exceptions::PyValueError::new_err(e))?,
            );
        }

        let b = self.expr.as_view().set_coefficient_ring(&Arc::new(var_map));

        Ok(b.into())
    }

    /// Expand the expression. Optionally, expand in `var` only.
    #[pyo3(signature = (var = None, via_poly = None))]
    pub fn expand(
        &self,
        var: Option<ConvertibleToExpression>,
        via_poly: Option<bool>,
    ) -> PyResult<PythonExpression> {
        if let Some(var) = var {
            let e = var.to_expression();

            if matches!(e.expr, Atom::Var(_) | Atom::Fun(_)) {
                if via_poly.unwrap_or(false) {
                    let b = self
                        .expr
                        .as_view()
                        .expand_via_poly::<i16>(Some(e.expr.as_view()));
                    Ok(b.into())
                } else {
                    let b = self.expr.as_view().expand_in(e.expr.as_view());
                    Ok(b.into())
                }
            } else {
                Err(exceptions::PyValueError::new_err(
                    "Expansion must be done wrt an indeterminate",
                ))
            }
        } else if via_poly.unwrap_or(false) {
            let b = self.expr.as_view().expand_via_poly::<i16>(None);
            Ok(b.into())
        } else {
            let b = self.expr.as_view().expand();
            Ok(b.into())
        }
    }

    /// Distribute numbers in the expression, for example:
    /// `2*(x+y)` -> `2*x+2*y`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x, y = S('x', 'y')
    /// >>> e = 3*(x+y)*(4*x+5*y)
    /// >>> print(e.expand_num())
    ///
    /// yields
    ///
    /// ```log
    /// (3*x+3*y)*(4*x+5*y)
    /// ```
    pub fn expand_num(&self) -> PythonExpression {
        self.expr.expand_num().into()
    }

    /// Collect terms involving the same power of `x`, where `x` is an indeterminate.
    /// Return the list of key-coefficient pairs and the remainder that matched no key.
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> x, y = S('x', 'y')
    /// >>> e = 5*x + x * y + x**2 + 5
    /// >>>
    /// >>> print(e.collect(x))
    ///
    /// yields `x^2+x*(y+5)+5`.
    ///
    /// >>> from symbolica import *
    /// >>> x, y = S('x', 'y')
    /// >>> exp, coeff = S('var', 'coeff')
    /// >>> e = 5*x + x * y + x**2 + 5
    /// >>>
    /// >>> print(e.collect(x, key_map=lambda x: exp(x), coeff_map=lambda x: coeff(x)))
    ///
    /// yields `var(1)*coeff(5)+var(x)*coeff(y+5)+var(x^2)*coeff(1)`.
    #[pyo3(signature = (*x, key_map = None, coeff_map = None))]
    pub fn collect(
        &self,
        x: &Bound<'_, PyTuple>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Callable[[Expression], Expression]]"
        ))]
        key_map: Option<Py<PyAny>>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Callable[[Expression], Expression]]"
        ))]
        coeff_map: Option<Py<PyAny>>,
    ) -> PyResult<PythonExpression> {
        if x.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "No variable or function specified",
            ));
        }

        let mut xs = vec![];
        for a in x {
            if let Ok(r) = a.extract::<PythonExpression>() {
                if matches!(r.expr, Atom::Var(_) | Atom::Fun(_)) {
                    xs.push(r.expr);
                } else {
                    return Err(exceptions::PyValueError::new_err(
                        "Collect must be done wrt a variable or function",
                    ));
                }
            } else {
                return Err(exceptions::PyValueError::new_err(
                    "Collect must be done wrt a variable or function",
                ));
            }
        }

        let xs = Arc::new(xs);
        let b = if key_map.is_none() && coeff_map.is_none() {
            self.expr.collect_multiple::<i16>(&xs)
        } else {
            let key_map_fn: Box<dyn Fn(AtomView, &mut Settable<'_, Atom>)> =
                if let Some(key_map) = key_map {
                    Box::new(move |key, out| {
                        Python::attach(|py| {
                            let key: PythonExpression = key.to_owned().into();

                            **out = key_map
                                .call(py, (key,), None)
                                .expect("Bad callback function")
                                .extract::<PythonExpression>(py)
                                .expect("Key map should return an expression")
                                .expr;
                        });
                    })
                } else {
                    Box::new(|_, _| {})
                };

            let coeff_map_fn: Box<dyn Fn(AtomView, &mut Settable<'_, Atom>)> =
                if let Some(coeff_map) = coeff_map {
                    Box::new(move |coeff, out| {
                        Python::attach(|py| {
                            let coeff: PythonExpression = coeff.to_owned().into();

                            **out = coeff_map
                                .call(py, (coeff,), None)
                                .expect("Bad callback function")
                                .extract::<PythonExpression>(py)
                                .expect("Coeff map should return an expression")
                                .expr;
                        });
                    })
                } else {
                    Box::new(|_, _| {})
                };

            self.expr.collect_multiple_mapped::<i16>(
                &xs,
                key_map_fn.as_ref(),
                coeff_map_fn.as_ref(),
            )
        };

        Ok(b.into())
    }

    /// Collect terms involving the same power of variables or functions with the name `x`, e.g.
    ///
    /// ```math
    /// collect_symbol(f(1,2) + x*f*(1,2), f) = (1+x)*f(1,2)
    /// ```
    ///
    ///
    /// Both the *key* (the quantity collected in) and its coefficient can be mapped using
    /// `key_map` and `coeff_map` respectively.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x, f = S('x', 'f')
    /// >>> e = f(1,2) + x*f(1,2)
    /// >>>
    /// >>> print(e.collect_symbol(f))
    ///
    /// yields `(1+x)*f(1,2)`.
    #[pyo3(signature = (x, key_map = None, coeff_map = None))]
    pub fn collect_symbol(
        &self,
        x: PythonExpression,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Callable[[Expression], Expression]]"
        ))]
        key_map: Option<Py<PyAny>>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Callable[[Expression], Expression]]"
        ))]
        coeff_map: Option<Py<PyAny>>,
    ) -> PyResult<PythonExpression> {
        let Some(x) = x.expr.get_symbol() else {
            return Err(exceptions::PyValueError::new_err(
                "Collect must be done wrt a variable or function",
            ));
        };

        let b = if key_map.is_none() && coeff_map.is_none() {
            self.expr.collect_symbol::<i16>(x)
        } else {
            let key_map_fn: Box<dyn Fn(AtomView, &mut Settable<'_, Atom>)> =
                if let Some(key_map) = key_map {
                    Box::new(move |key, out| {
                        Python::attach(|py| {
                            let key: PythonExpression = key.to_owned().into();

                            **out = key_map
                                .call(py, (key,), None)
                                .expect("Bad callback function")
                                .extract::<PythonExpression>(py)
                                .expect("Key map should return an expression")
                                .expr;
                        });
                    })
                } else {
                    Box::new(|_, _| {})
                };

            let coeff_map_fn: Box<dyn Fn(AtomView, &mut Settable<'_, Atom>)> =
                if let Some(coeff_map) = coeff_map {
                    Box::new(move |coeff, out| {
                        Python::attach(|py| {
                            let coeff: PythonExpression = coeff.to_owned().into();

                            **out = coeff_map
                                .call(py, (coeff,), None)
                                .expect("Bad callback function")
                                .extract::<PythonExpression>(py)
                                .expect("Coeff map should return an expression")
                                .expr;
                        });
                    })
                } else {
                    Box::new(|_, _| {})
                };

            self.expr
                .collect_symbol_mapped::<i16>(x, key_map_fn.as_ref(), coeff_map_fn.as_ref())
        };

        Ok(b.into())
    }

    /// Collect common factors from (nested) sums.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> e = E('x*(x+y*x+x^2+y*(x+x^2))')
    /// >>> e.collect_factors()
    ///
    /// yields
    ///
    /// ```log
    /// v1^2*(1+v1+v2+v2*(1+v1))
    /// ```
    pub fn collect_factors(&self) -> PythonExpression {
        self.expr.collect_factors().into()
    }

    /// Iteratively extract the minimal common powers of an indeterminate `v` for every term that contains `v`
    /// and continue to the next indeterminate in `variables`.
    /// This is a generalization of Horner's method for polynomials.
    ///
    /// If no variables are provided, a heuristically determined variable ordering is used
    /// that minimizes the number of operations.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import *
    /// >>> expr = E('v1 + v1*v2 + 2 v1*v2*v3 + v1^2 + v1^3*y + v1^4*z')
    /// >>> collected = expr.collect_horner([S('v1'), S('v2')])
    ///
    /// yields `v1*(1+v1*(1+v1*(v1*z+y))+v2*(1+2*v3))`.
    #[pyo3(signature = (vars=None))]
    pub fn collect_horner(
        &self,
        vars: Option<Vec<PythonExpression>>,
    ) -> PyResult<PythonExpression> {
        if let Some(vars) = vars {
            let vars: Vec<_> = vars
                .into_iter()
                .map(|e| Indeterminate::try_from(e.expr))
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e))?;
            Ok(self.expr.collect_horner(Some(&vars)).into())
        } else {
            Ok(self.expr.collect_horner::<Indeterminate>(None).into())
        }
    }

    /// Collect numerical factors by removing the numerical content from additions.
    /// For example, `-2*x + 4*x^2 + 6*x^3` will be transformed into `-2*(x - 2*x^2 - 3*x^3)`.
    ///
    /// The first argument of the addition is normalized to a positive quantity.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>>
    /// >>> x, y = S('x', 'y')
    /// >>> e = (-3*x+6*y)(2*x+2*y)
    /// >>> print(e.collect_num())
    ///
    /// yields
    ///
    /// ```log
    /// -6*(x-2*y)*(x+y)
    /// ```
    pub fn collect_num(&self) -> PythonExpression {
        self.expr.collect_num().into()
    }

    /// Collect terms involving the literal occurrence of `x`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>>
    /// >>> x, y = S('x', 'y')
    /// >>> e = 5*x + x * y + x**2 + y*x**2
    /// >>> print(e.coefficient(x**2))
    ///
    /// yields
    ///
    /// ```log
    /// y + 1
    /// ```
    pub fn coefficient(&self, x: ConvertibleToExpression) -> PythonExpression {
        let r = self.expr.coefficient(x.to_expression().expr.as_view());
        r.into()
    }

    /// Collect terms involving the same power of `x`, where `x` is an indeterminate.
    /// Return the list of key-coefficient pairs and the remainder that matched no key.
    ///
    /// Examples
    /// --------
    ///
    /// from symbolica import Expression
    /// >>>
    /// >>> x, y = S('x', 'y')
    /// >>> e = 5*x + x * y + x**2 + 5
    /// >>>
    /// >>> for a in e.coefficient_list(x):
    /// >>>     print(a[0], a[1])
    ///
    /// yields
    ///
    /// ```log
    /// x y+5
    /// x^2 1
    /// 1 5
    /// ```
    #[pyo3(signature = (*x,))]
    pub fn coefficient_list(
        &self,
        x: Bound<'_, PyTuple>,
    ) -> PyResult<Vec<(PythonExpression, PythonExpression)>> {
        if x.is_empty() {
            return Err(exceptions::PyValueError::new_err(
                "No variable or function specified",
            ));
        }

        let mut xs = vec![];
        for a in x {
            if let Ok(r) = a.extract::<PythonExpression>() {
                if matches!(r.expr, Atom::Var(_) | Atom::Fun(_)) {
                    xs.push(r.expr);
                } else {
                    return Err(exceptions::PyValueError::new_err(
                        "Collect must be done wrt a variable or function",
                    ));
                }
            } else {
                return Err(exceptions::PyValueError::new_err(
                    "Collect must be done wrt a variable or function",
                ));
            }
        }

        let list = self.expr.coefficient_list::<i16>(&xs);

        let py_list: Vec<_> = list
            .into_iter()
            .map(|e| (e.0.to_owned().into(), e.1.into()))
            .collect();

        Ok(py_list)
    }

    /// Derive the expression w.r.t the variable `x`.
    pub fn derivative(&self, x: ConvertibleToExpression) -> PyResult<PythonExpression> {
        let id = if let AtomView::Var(x) = x.to_expression().expr.as_view() {
            x.get_symbol()
        } else {
            return Err(exceptions::PyValueError::new_err(
                "Derivative must be taken wrt a variable",
            ));
        };

        let b = self.expr.derivative(id);

        Ok(b.into())
    }

    /// Series expand in `x` around `expansion_point` to depth `depth`.
    ///
    /// Examples
    /// -------
    /// >>> from symbolica import Expression
    /// >>> x, y = S('x', 'y')
    /// >>> f = S('f')
    /// >>>
    /// >>> e = 2* x**2 * y + f(x)
    /// >>> e = e.series(x, 0, 2)
    /// >>>
    /// >>> print(e)
    ///
    /// yields `f(0)+x*der(1,f,0)+1/2*x^2*(der(2,f,0)+4*y)`.
    #[pyo3(signature = (x, expansion_point, depth, depth_denom = 1, depth_is_absolute = true))]
    pub fn series(
        &self,
        x: PythonExpression,
        expansion_point: ConvertibleToExpression,
        depth: i64,
        depth_denom: i64,
        depth_is_absolute: bool,
    ) -> PyResult<PythonSeries> {
        let id: crate::atom::Indeterminate = x.expr.try_into().map_err(|_| {
            exceptions::PyValueError::new_err(format!(
                "Series expansion must be done wrt a variable"
            ))
        })?;

        let depth = if depth_is_absolute {
            crate::poly::series::SeriesDepth::absolute((depth, depth_denom))
        } else {
            crate::poly::series::SeriesDepth::relative((depth, depth_denom))
        };

        match self
            .expr
            .series(id, expansion_point.to_expression().expr.as_view(), depth)
        {
            Ok(s) => Ok(PythonSeries { series: s }),
            Err(e) => Err(exceptions::PyValueError::new_err(e.to_string())),
        }
    }

    /// Compute the partial fraction decomposition in `x`.
    ///
    /// If `None` is passed, the expression will be decomposed in all variables
    /// which involves a potentially expensive Groebner basis computation.
    ///
    /// Examples
    /// --------
    ///
    /// >>> p = E('1/((x+y)*(x^2+x*y+1)(x+1))')
    /// >>> print(p.apart(S('x')))
    ///
    /// Multivariate partial fractioning
    /// >>> p = E('(2y-x)/(y*(x+y)*(y-x))')
    /// >>> print(p.apart())
    #[pyo3(signature = (x = None))]
    pub fn apart(&self, x: Option<PythonExpression>) -> PyResult<PythonExpression> {
        if let Some(x) = x {
            if let Some(r) = x.expr.get_symbol() {
                Ok(self.expr.apart(r).into())
            } else {
                Err(exceptions::PyValueError::new_err(
                    "Partial fraction decomposition must be done wrt a symbol",
                ))
            }
        } else {
            Ok(self.apart_multivariate().into())
        }
    }

    /// Write the expression over a common denominator.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('v1^2/2+v1^3/v4*v2+v3/(1+v4)')
    /// >>> print(p.together())
    pub fn together(&self) -> PyResult<PythonExpression> {
        Ok(self.expr.together().into())
    }

    /// Cancel common factors between numerators and denominators.
    /// Any non-canceling parts of the expression will not be rewritten.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('1+(y+1)^10*(x+1)/(x^2+2x+1)')
    /// >>> print(p.cancel())
    /// 1+(y+1)**10/(x+1)
    pub fn cancel(&self) -> PyResult<PythonExpression> {
        Ok(self.expr.cancel().into())
    }

    /// Factor the expression over the rationals.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> p = E('(6 + x)/(7776 + 6480*x + 2160*x^2 + 360*x^3 + 30*x^4 + x^5)')
    /// >>> print(p.factor())
    /// (x+6)**-4
    pub fn factor(&self) -> PyResult<PythonExpression> {
        Ok(self.expr.factor().into())
    }

    /// Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`.
    /// All non-polynomial elements will be converted to new independent variables.
    ///
    /// If a `modulus` is provided, the coefficients will be converted to finite field elements mod `modulus`.
    /// If on top a `power` is provided, for example `(2, a)`, the polynomial will be converted to the Galois field
    /// `GF(modulus^2)` where `a` is the variable of the minimal polynomial of the field.
    ///
    /// If a `minimal_poly` is provided, the polynomial will be converted to a number field with the given minimal polynomial.
    /// The minimal polynomial must be a monic, irreducible univariate polynomial. If a `modulus` is provided as well,
    /// the Galois field will be created with `minimal_poly` as the minimal polynomial.
    #[gen_stub(skip)]
    #[pyo3(signature = (modulus = None, power = None, minimal_poly = None, vars = None))]
    pub fn to_polynomial(
        &self,
        modulus: Option<u64>,
        mut power: Option<(u16, Symbol)>,
        minimal_poly: Option<PythonPolynomial>,
        vars: Option<Vec<PythonExpression>>,
        py: Python,
    ) -> PyResult<Py<PyAny>> {
        let mut var_map = vec![];
        if let Some(vm) = vars {
            for v in vm {
                var_map.push(
                    v.expr
                        .try_into()
                        .map_err(|e| exceptions::PyValueError::new_err(e))?,
                );
            }
        }

        let var_map = if var_map.is_empty() {
            None
        } else {
            Some(Arc::new(var_map))
        };

        if power.is_some() && modulus.is_none() {
            return Err(exceptions::PyValueError::new_err(
                "Extension field requires a modulus to be set",
            ));
        }

        let poly = minimal_poly.map(|x| x.poly);
        if let Some(p) = &poly {
            if p.nvars() != 1 {
                return Err(exceptions::PyValueError::new_err(
                    "Minimal polynomial must be a univariate polynomial",
                ));
            }

            if power.is_none() {
                if let PolyVariable::Symbol(name) = p.get_vars_ref()[0] {
                    power = Some((p.degree(0) as u16, name));
                } else {
                    return Err(exceptions::PyValueError::new_err(format!(
                        "Extension field polynomial {p} must have a symbol as a variable"
                    )));
                }
            }
        }

        if let Some(m) = modulus {
            if let Some((e, name)) = power {
                if let Some(p) = &poly {
                    if e != p.degree(0) {
                        return Err(exceptions::PyValueError::new_err(
                            "Extension field degree must match the minimal polynomial degree",
                        ));
                    }

                    if PolyVariable::Symbol(name) != p.get_vars_ref()[0] {
                        return Err(exceptions::PyValueError::new_err(
                            "Extension variable must be the same as the variable in the minimal polynomial",
                        ));
                    }

                    if m == 2 {
                        let p = p.map_coeff(|c| c.to_finite_field(&Z2), Z2);
                        if !p.is_irreducible() || e != p.degree(0) {
                            return Err(exceptions::PyValueError::new_err(
                                "Minimal polynomial must be irreducible and monic",
                            ));
                        }

                        let g = AlgebraicExtension::new(p);
                        PythonGaloisFieldPrimeTwoPolynomial {
                            poly: self
                                .expr
                                .try_to_polynomial(&Z2, var_map)
                                .map_err(|e| exceptions::PyValueError::new_err(e))?
                                .to_number_field(&g),
                        }
                        .into_py_any(py)
                    } else {
                        let f = Zp64::new(m);
                        let p = p.map_coeff(|c| c.to_finite_field(&f), f.clone());
                        if !p.is_irreducible() || !f.is_one(&p.lcoeff()) || e != p.degree(0) {
                            return Err(exceptions::PyValueError::new_err(
                                "Minimal polynomial must be irreducible and monic",
                            ));
                        }

                        let g = AlgebraicExtension::new(p);
                        PythonGaloisFieldPolynomial {
                            poly: self
                                .expr
                                .try_to_polynomial(&f, var_map)
                                .map_err(|e| exceptions::PyValueError::new_err(e))?
                                .to_number_field(&g),
                        }
                        .into_py_any(py)
                    }
                } else if m == 2 {
                    let g = AlgebraicExtension::galois_field(Z2, e as usize, name.into());
                    PythonGaloisFieldPrimeTwoPolynomial {
                        poly: self
                            .expr
                            .try_to_polynomial(&Z2, var_map)
                            .map_err(|e| exceptions::PyValueError::new_err(e))?
                            .to_number_field(&g),
                    }
                    .into_py_any(py)
                } else {
                    let f = Zp64::new(m);
                    let g = AlgebraicExtension::galois_field(Zp64::new(m), e as usize, name.into());
                    PythonGaloisFieldPolynomial {
                        poly: self
                            .expr
                            .try_to_polynomial(&f, var_map)
                            .map_err(|e| exceptions::PyValueError::new_err(e))?
                            .to_number_field(&g),
                    }
                    .into_py_any(py)
                }
            } else if m == 2 {
                PythonPrimeTwoPolynomial {
                    poly: self
                        .expr
                        .try_to_polynomial(&Z2, var_map)
                        .map_err(|e| exceptions::PyValueError::new_err(e))?,
                }
                .into_py_any(py)
            } else {
                PythonFiniteFieldPolynomial {
                    poly: self
                        .expr
                        .try_to_polynomial(&Zp64::new(m), var_map)
                        .map_err(|e| exceptions::PyValueError::new_err(e))?,
                }
                .into_py_any(py)
            }
            // FIXME: ignoring minimal poly!
        } else if let Some(p) = poly {
            if !p.is_irreducible() || !p.lcoeff().is_one() {
                return Err(exceptions::PyValueError::new_err(
                    "Minimal polynomial must be irreducible and monic",
                ));
            }

            let f = AlgebraicExtension::new(p);
            if f.poly().exponents == [0, 2] && f.poly().get_constant() == Rational::one() {
                // convert complex coefficients
                PythonNumberFieldPolynomial {
                    poly: self
                        .expr
                        .try_to_polynomial(&f, var_map)
                        .map_err(|e| exceptions::PyValueError::new_err(e))?,
                }
                .into_py_any(py)
            } else {
                PythonNumberFieldPolynomial {
                    poly: self
                        .expr
                        .try_to_polynomial(&Q, var_map)
                        .map_err(|e| exceptions::PyValueError::new_err(e))?
                        .to_number_field(&f),
                }
                .into_py_any(py)
            }
        } else {
            PythonPolynomial {
                poly: self
                    .expr
                    .try_to_polynomial(&Q, var_map)
                    .map_err(|e| exceptions::PyValueError::new_err(e))?,
            }
            .into_py_any(py)
        }
    }

    /// Convert the expression to a rational polynomial, optionally, with the variable ordering specified in `vars`.
    /// The latter is useful if it is known in advance that more variables may be added in the future to the
    /// rational polynomial through composition with other rational polynomials.
    ///
    /// All non-rational polynomial parts will automatically be converted to new independent variables.
    ///
    /// Examples
    /// --------
    /// >>> a = E('(1 + 3*x1 + 5*x2 + 7*x3 + 9*x4 + 11*x5 + 13*x6 + 15*x7)^2 - 1').to_rational_polynomial()
    /// >>> print(a)
    #[pyo3(signature = (vars = None))]
    pub fn to_rational_polynomial(
        &self,
        vars: Option<Vec<PythonExpression>>,
    ) -> PyResult<PythonRationalPolynomial> {
        let mut var_map = vec![];
        if let Some(vm) = vars {
            for v in vm {
                var_map.push(
                    v.expr
                        .try_into()
                        .map_err(|e| exceptions::PyValueError::new_err(e))?,
                );
            }
        }

        let var_map = if var_map.is_empty() {
            None
        } else {
            Some(Arc::new(var_map))
        };

        let poly = self
            .expr
            .try_to_rational_polynomial(&Q, &Z, var_map)
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!(
                    "Could not convert expression to rational polynomial: {e}",
                ))
            })?;

        Ok(PythonRationalPolynomial { poly })
    }

    /// Return an iterator over the pattern `self` matching to `lhs`.
    /// Restrictions on the pattern can be supplied through `cond`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> x, x_ = S('x','x_')
    /// >>> f = S('f')
    /// >>> e = f(x)*f(1)*f(2)*f(3)
    /// >>> for match in e.match(f(x_)):
    /// >>>    for map in match:
    /// >>>        print(map[0],'=', map[1])
    #[pyo3(name = "match", signature = (lhs, cond = None, min_level=0, max_level=None, level_range = None, level_is_tree_depth = false, partial=true, allow_new_wildcards_on_rhs = false))]
    pub fn pattern_match(
        &self,
        lhs: ConvertibleToExpression,
        cond: Option<ConvertibleToPatternRestriction>,
        min_level: usize,
        max_level: Option<usize>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: bool,
        partial: bool,
        allow_new_wildcards_on_rhs: bool,
    ) -> PyResult<PythonMatchIterator> {
        let conditions = cond.map(|r| r.0).unwrap_or_default();
        let settings = MatchSettings {
            level_range: level_range.unwrap_or((min_level, max_level)),
            level_is_tree_depth,
            allow_new_wildcards_on_rhs,
            partial,
            ..MatchSettings::default()
        };
        Ok(PythonMatchIterator::new(
            (
                lhs.to_expression().expr.to_pattern(),
                self.expr.clone(),
                conditions,
                settings,
            ),
            move |(lhs, target, res, settings)| {
                PatternAtomTreeIterator::new(lhs, target.as_view(), Some(res), Some(settings))
            },
        ))
    }

    /// Test whether the pattern is found in the expression.
    /// Restrictions on the pattern can be supplied through `cond`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> f = S('f')
    /// >>> if f(1).matches(f(2)):
    /// >>>    print('match')
    #[pyo3(signature = (lhs, cond = None, min_level=0, max_level=None, level_range = None, level_is_tree_depth = false, partial=true, allow_new_wildcards_on_rhs = false))]
    pub fn matches(
        &self,
        lhs: ConvertibleToExpression,
        cond: Option<ConvertibleToPatternRestriction>,
        min_level: usize,
        max_level: Option<usize>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: bool,
        partial: bool,
        allow_new_wildcards_on_rhs: bool,
    ) -> PyResult<PythonCondition> {
        let conditions = cond.map(|r| r.0).unwrap_or_default();
        let settings = MatchSettings {
            level_range: level_range.unwrap_or((min_level, max_level)),
            level_is_tree_depth,
            allow_new_wildcards_on_rhs,
            partial,
            ..MatchSettings::default()
        };

        Ok(PythonCondition {
            condition: Condition::Yield(Relation::Matches(
                self.expr.to_pattern(),
                lhs.to_expression().expr.to_pattern(),
                conditions,
                settings,
            )),
        })
    }

    /// Return an iterator over the replacement of the pattern `self` on `lhs` by `rhs`.
    /// Restrictions on pattern can be supplied through `cond`.
    ///
    /// The `level_range` specifies the `[min,max]` level at which the pattern is allowed to match.
    /// The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree,
    /// depending on `level_is_tree_depth`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> from symbolica import Expression
    /// >>> x_ = S('x_')
    /// >>> f = S('f')
    /// >>> e = f(1)*f(2)*f(3)
    /// >>> for r in e.replace(f(x_), f(x_ + 1)):
    /// >>>     print(r)
    ///
    /// Yields:
    /// ```log
    /// f(2)*f(2)*f(3)
    /// f(1)*f(3)*f(3)
    /// f(1)*f(2)*f(4)
    /// ```
    #[pyo3(signature = (lhs, rhs, cond = None, min_level=0, max_level=None, level_range = None, level_is_tree_depth = false, partial=true, allow_new_wildcards_on_rhs = false))]
    pub fn replace_iter(
        &self,
        lhs: ConvertibleToExpression,
        rhs: ConvertibleToReplaceWith,
        cond: Option<ConvertibleToPatternRestriction>,
        min_level: usize,
        max_level: Option<usize>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: bool,
        partial: bool,
        allow_new_wildcards_on_rhs: bool,
    ) -> PyResult<PythonReplaceIterator> {
        let conditions = cond.map(|r| r.0.clone()).unwrap_or_default();
        let settings = MatchSettings {
            level_range: level_range.unwrap_or((min_level, max_level)),
            level_is_tree_depth,
            allow_new_wildcards_on_rhs,
            partial,
            ..MatchSettings::default()
        };

        Ok(PythonReplaceIterator::new(
            (
                lhs.to_expression().expr.to_pattern(),
                self.expr.clone(),
                rhs.to_replace_with()?,
                conditions,
                settings,
            ),
            move |(lhs, target, rhs, res, settings)| {
                ReplaceIterator::new(
                    lhs,
                    target.as_view(),
                    rhs.clone(),
                    Some(res),
                    Some(settings),
                )
            },
        ))
    }

    /// Replace all atoms matching the pattern `pattern` by the right-hand side `rhs`.
    /// Restrictions on pattern can be supplied through `cond`.
    ///
    /// The `level_range` specifies the `[min,max]` level at which the pattern is allowed to match.
    /// The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree,
    /// depending on `level_is_tree_depth`.
    ///
    /// The entire operation can be repeated until there are no more matches using `repeat=True`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> x, w1_, w2_ = S('x','w1_','w2_')
    /// >>> f = S('f')
    /// >>> e = f(3,x)
    /// >>> r = e.replace(f(w1_,w2_), f(w1_ - 1, w2_**2), (w1_ >= 1) & w2_.is_var())
    /// >>> print(r)
    ///
    /// Parameters
    /// ----------
    /// pattern: Transformer | Expression | int
    ///     The pattern to match.
    /// rhs: Transformer | Expression | int
    ///     The right-hand side to replace the matched subexpression with. Can be a transformer, expression or a function that maps a dictionary of wildcards to an expression.
    /// cond: Optional[PatternRestriction]
    ///     Conditions on the pattern.
    /// min_level: int, optional
    ///     The minimum level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
    /// max_level: int, optional
    ///     The maximum level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
    /// level_range: (int, int), optional
    ///     Specifies the `[min,max]` level at which the pattern is allowed to match. The first level is 0 and the level is increased when going into a function or one level deeper in the expression tree, depending on `level_is_tree_depth`.
    /// level_is_tree_depth: bool, optional
    ///     If set to `True`, the level is increased when going one level deeper in the expression tree.
    /// allow_new_wildcards_on_rhs: bool, optional
    ///     If set to `True`, wildcards that do not appear ion the pattern are allowed on the right-hand side.
    /// rhs_cache_size: int, optional
    ///      Cache the first `rhs_cache_size` substituted patterns. If set to `None`, an internally determined cache size is used.
    ///      Warning: caching should be disabled (`rhs_cache_size=0`) if the right-hand side contains side effects, such as updating a global variable.
    /// repeat: bool, optional
    ///     If set to `True`, the entire operation will be repeated until there are no more matches.
    /// once: bool, optional
    ///     If set to `True`, only the first match will be replaced.
    /// bottom_up: bool, optional
    ///     If set to `True`, the replacement will be applied from the bottom of the expression tree upwards.
    /// nested: bool, optional
    ///     If set to `True`, nested replacements will be allowed.
    #[pyo3(signature = (pattern, rhs, cond = None, non_greedy_wildcards = None, min_level=0, max_level=None, level_range = None, level_is_tree_depth = false, partial=true, allow_new_wildcards_on_rhs = false, rhs_cache_size = None, repeat = false, once = false, bottom_up = false, nested = false))]
    pub fn replace(
        &self,
        pattern: ConvertibleToExpression,
        rhs: ConvertibleToReplaceWith,
        cond: Option<ConvertibleToPatternRestriction>,
        non_greedy_wildcards: Option<Vec<PythonExpression>>,
        min_level: usize,
        max_level: Option<usize>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: bool,
        partial: bool,
        allow_new_wildcards_on_rhs: bool,
        rhs_cache_size: Option<usize>,
        repeat: bool,
        once: bool,
        bottom_up: bool,
        nested: bool,
    ) -> PyResult<PythonExpression> {
        let pattern = pattern.to_expression().expr.to_pattern();
        let rhs = &rhs.to_replace_with()?;

        let mut settings = MatchSettings::cached();

        if let Some(ngw) = non_greedy_wildcards {
            settings.non_greedy_wildcards = ngw
                .iter()
                .map(|x| match x.expr.as_view() {
                    AtomView::Var(v) => {
                        let name = v.get_symbol();
                        if v.get_wildcard_level() == 0 {
                            return Err(exceptions::PyTypeError::new_err(
                                "Only wildcards can be restricted.",
                            ));
                        }
                        Ok(name)
                    }
                    _ => Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    )),
                })
                .collect::<Result<_, _>>()?;
        }

        settings.level_range = level_range.unwrap_or((min_level, max_level));
        settings.partial = partial;
        settings.level_is_tree_depth = level_is_tree_depth;
        settings.allow_new_wildcards_on_rhs = allow_new_wildcards_on_rhs;

        if let Some(rhs_cache_size) = rhs_cache_size {
            settings.rhs_cache_size = rhs_cache_size;
        }

        let mut expr_ref = self.expr.as_view();

        let cond = cond.map(|r| r.0);

        let mut out = RecycledAtom::new();
        let mut out2 = RecycledAtom::new();
        while expr_ref.replace_into(
            &pattern,
            rhs,
            cond.as_ref(),
            Some(&settings),
            ReplaceSettings {
                once,
                bottom_up,
                nested,
            },
            &mut out,
        ) {
            if !repeat {
                break;
            }

            std::mem::swap(&mut out, &mut out2);
            expr_ref = out2.as_view();
        }

        Ok(out.into_inner().into())
    }

    /// Replace all atoms matching the patterns. See `replace` for more information.
    ///
    /// The entire operation can be repeated until there are no more matches using `repeat=True`.
    ///
    /// Examples
    /// --------
    ///
    /// >>> x, y, f = S('x', 'y', 'f')
    /// >>> e = f(x,y)
    /// >>> r = e.replace_multiple([Replacement(x, y), Replacement(y, x)])
    /// >>> print(r)
    /// f(y,x)
    ///
    /// Parameters
    /// ----------
    /// replacements: Sequence[Replacement]
    ///     The list of replacements to apply.
    /// repeat: bool, optional
    ///     If set to `True`, the entire operation will be repeated until there are no more matches.
    #[pyo3(signature = (replacements, repeat = false, once = false, bottom_up = false, nested = false))]
    pub fn replace_multiple(
        &self,
        replacements: Vec<PythonReplacement>,
        repeat: bool,
        once: bool,
        bottom_up: bool,
        nested: bool,
    ) -> PyResult<PythonExpression> {
        let reps = replacements
            .into_iter()
            .map(|x| x.replacement)
            .collect::<Vec<_>>();

        let mut expr_ref = self.expr.as_view();

        let settings = ReplaceSettings {
            once,
            bottom_up,
            nested,
        };
        let mut out = RecycledAtom::new();
        let mut out2 = RecycledAtom::new();
        while expr_ref.replace_multiple_into(&reps, settings, &mut out) {
            if !repeat {
                break;
            }

            std::mem::swap(&mut out, &mut out2);
            expr_ref = out2.as_view();
        }

        Ok(out.into_inner().into())
    }

    /// Replace all wildcards in the expression with the given replacements.
    pub fn replace_wildcards(
        &self,
        replacements: HashMap<PythonExpression, PythonExpression>,
    ) -> PyResult<PythonExpression> {
        let mut reps = HashMap::default();
        for (k, v) in replacements {
            let k = k.expr.as_view();
            let s = if let AtomView::Var(v) = k {
                if v.get_wildcard_level() == 0 {
                    return Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be replaced.",
                    ));
                }
                v.get_symbol()
            } else {
                return Err(exceptions::PyTypeError::new_err(
                    "Only wildcards can be replaced.",
                ));
            };
            reps.insert(s, v.expr);
        }

        let res = self.expr.to_pattern().replace_wildcards(&reps);

        Ok(res.into())
    }

    /// Solve a linear system in the variables `variables`, where each expression
    /// in the system is understood to yield 0.
    ///
    /// If the system is underdetermined, a partial solution is returned
    /// where each bound variable is a linear combination of the free
    /// variables. The free variables are chosen such that they have the highest index in the `vars` list.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x, y, c = S('x', 'y', 'c')
    /// >>> f = S('f')
    /// >>> x_r, y_r = Expression.solve_linear_system([f(c)*x + y/c - 1, y-c/2], [x, y])
    /// >>> print('x =', x_r, ', y =', y_r)
    #[pyo3(signature = (system, variables, warn_if_underdetermined = true))]
    #[classmethod]
    pub fn solve_linear_system(
        _cls: &Bound<'_, PyType>,
        system: Vec<ConvertibleToExpression>,
        variables: Vec<PythonExpression>,
        warn_if_underdetermined: bool,
    ) -> PyResult<Vec<PythonExpression>> {
        let system: Vec<_> = system.into_iter().map(|x| x.to_expression().expr).collect();
        let vars: Vec<_> = variables.into_iter().map(|v| v.expr).collect();

        match AtomView::solve_linear_system::<u16, _, Atom>(&system, &vars) {
            Ok(res) => Ok(res.into_iter().map(|x| x.into()).collect()),
            Err(SolveError::Underdetermined {
                rank,
                partial_solution,
            }) => {
                if warn_if_underdetermined {
                    warn!(
                        "The system is underdetermined (rank {rank} < size {})",
                        vars.len()
                    );
                }

                Ok(partial_solution.into_iter().map(|x| x.into()).collect())
            }
            Err(SolveError::Other(e)) => Err(exceptions::PyValueError::new_err(e)),
        }
    }

    /// Find the root of an expression in `x` numerically over the reals using Newton's method.
    /// Use `init` as the initial guess for the root.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x, y, c = S('x', 'y', 'c')
    /// >>> f = S('f')
    /// >>> x_r, y_r = Expression.solve_linear_system([f(c)*x + y/c - 1, y-c/2], [x, y])
    /// >>> print('x =', x_r, ', y =', y_r)
    #[gen_stub(override_return_type(type_repr = "decimal.Decimal", imports = ("decimal")))]
    #[pyo3(signature =
        (variable,
        init,
        prec = 1e-4,
        max_iterations = 1000),
        )]
    pub fn nsolve(
        &self,
        variable: PythonExpression,
        init: PythonMultiPrecisionFloat,
        prec: f64,
        max_iterations: usize,
        py: Python,
    ) -> PyResult<Py<PyAny>> {
        let id: crate::atom::Indeterminate = variable.expr.try_into().map_err(|_| {
            exceptions::PyValueError::new_err(format!("Solve must be done wrt a variable"))
        })?;

        if init.0.prec() == 53 {
            let r = self
                .expr
                .nsolve::<F64, _>(id, init.0.to_f64().into(), prec.into(), max_iterations)
                .map_err(|e| {
                    exceptions::PyValueError::new_err(format!("Could not solve system: {e}"))
                })?;
            r.into_inner().into_py_any(py)
        } else {
            PythonMultiPrecisionFloat(
                self.expr
                    .nsolve(id, init.0, prec.into(), max_iterations)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Could not solve system: {e}"))
                    })?,
            )
            .into_py_any(py)
        }
    }

    /// Find a common root of multiple expressions in `variables` numerically over the reals using Newton's method.
    /// Use `init` as the initial guess for the root.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x, y, c = S('x', 'y', 'c')
    /// >>> f = S('f')
    /// >>> x_r, y_r = Expression.solve_linear_system([f(c)*x + y/c - 1, y-c/2], [x, y])
    /// >>> print('x =', x_r, ', y =', y_r)
    #[gen_stub(override_return_type(type_repr = "decimal.Decimal", imports = ("decimal")))]
    #[pyo3(signature =
        (system,
        variables,
        init,
        prec = 1e-4,
        max_iterations = 1000),
        )]
    #[classmethod]
    pub fn nsolve_system(
        _cls: &Bound<'_, PyType>,
        system: Vec<ConvertibleToExpression>,
        variables: Vec<PythonExpression>,
        init: Vec<PythonMultiPrecisionFloat>,
        prec: f64,
        max_iterations: usize,
        py: Python,
    ) -> PyResult<Vec<Py<PyAny>>> {
        let system: Vec<_> = system.into_iter().map(|x| x.to_expression()).collect();
        let system_b: Vec<_> = system.iter().map(|x| x.expr.as_view()).collect();

        let mut vars = vec![];
        for v in variables {
            let id: crate::atom::Indeterminate = v.expr.try_into().map_err(|_| {
                exceptions::PyValueError::new_err(format!("Solve must be done wrt a variable"))
            })?;
            vars.push(id);
        }

        if init[0].0.prec() == 53 {
            let init: Vec<_> = init.into_iter().map(|x| x.0.to_f64().into()).collect();

            let res: Vec<F64> =
                AtomView::nsolve_system(&system_b, &vars, &init, prec.into(), max_iterations)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Could not solve system: {e}"))
                    })?;

            Ok(res
                .into_iter()
                .map(|x| x.into_inner().into_py_any(py))
                .collect::<Result<_, _>>()?)
        } else {
            let init: Vec<_> = init.into_iter().map(|x| x.0).collect();

            let res: Vec<Float> =
                AtomView::nsolve_system(&system_b, &vars, &init, prec.into(), max_iterations)
                    .map_err(|e| {
                        exceptions::PyValueError::new_err(format!("Could not solve system: {e}"))
                    })?;

            Ok(res
                .into_iter()
                .map(|x| PythonMultiPrecisionFloat(x).into_py_any(py))
                .collect::<Result<_, _>>()?)
        }
    }

    /// Evaluate the expression, using a map of all the constants and
    /// user functions to a float.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x = S('x')
    /// >>> f = S('f')
    /// >>> e = E('cos(x)')*3 + f(x,2)
    /// >>> print(e.evaluate({x: 1}, {f: lambda args: args[0]+args[1]}))
    pub fn evaluate(
        &self,
        constants: HashMap<PythonExpression, f64>,
        #[gen_stub(override_type(
            type_repr = "dict[Expression, typing.Callable[[typing.Sequence[float]], float]]"
        ))]
        functions: HashMap<PolyVariable, Py<PyAny>>,
    ) -> PyResult<f64> {
        let constants = constants
            .iter()
            .map(|(k, v)| (k.expr.as_view(), *v))
            .collect();

        let functions = functions
            .into_iter()
            .map(|(k, v)| {
                let id = if let PolyVariable::Symbol(v) = k {
                    v
                } else {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Expected function name instead of {k:?}",
                    )))?
                };

                Ok((
                    id,
                    EvaluationFn::new(Box::new(move |args, _, _, _| {
                        Python::attach(|py| {
                            v.call(py, (args.to_vec(),), None)
                                .expect("Bad callback function")
                                .extract::<f64>(py)
                                .expect("Function does not return a float")
                        })
                    })),
                ))
            })
            .collect::<PyResult<_>>()?;

        self.expr
            .evaluate(|x| x.into(), &constants, &functions)
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not evaluate expression: {e}"))
            })
    }

    /// Evaluate the expression, using a map of all the constants and
    /// user functions using arbitrary precision arithmetic.
    /// The user has to specify the number of decimal digits of precision
    /// and provide all input numbers as floats, strings or `decimal`.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> from decimal import Decimal, getcontext
    /// >>> x = S('x', 'f')
    /// >>> e = E('cos(x)')*3 + f(x, 2)
    /// >>> getcontext().prec = 100
    /// >>> a = e.evaluate_with_prec({x: Decimal('1.123456789')}, {
    /// >>>                         f: lambda args: args[0] + args[1]}, 100)
    #[gen_stub(override_return_type(type_repr = "decimal.Decimal", imports = ("decimal")))]
    pub fn evaluate_with_prec(
        &self,
        constants: HashMap<PythonExpression, PythonMultiPrecisionFloat>,
        #[gen_stub(override_type(
            type_repr = "dict[Expression, typing.Callable[[typing.Sequence[decimal.Decimal]], float | str | decimal.Decimal]]"
        ))]
        functions: HashMap<PolyVariable, Py<PyAny>>,
        decimal_digit_precision: u32,
        py: Python,
    ) -> PyResult<Py<PyAny>> {
        let prec = (decimal_digit_precision as f64 * std::f64::consts::LOG2_10).ceil() as u32;

        let constants: HashMap<AtomView, Float> = constants
            .iter()
            .map(|(k, v)| {
                Ok((k.expr.as_view(), {
                    let mut vv = v.0.clone();
                    vv.set_prec(prec);
                    vv
                }))
            })
            .collect::<PyResult<_>>()?;

        let functions = functions
            .into_iter()
            .map(|(k, v)| {
                let id = if let PolyVariable::Symbol(v) = k {
                    v
                } else {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Expected function name instead of {k}",
                    )))?
                };

                Ok((
                    id,
                    EvaluationFn::new(Box::new(move |args: &[Float], _, _, _| {
                        Python::attach(|py| {
                            let mut vv = v
                                .call(
                                    py,
                                    (args
                                        .iter()
                                        .map(|x| {
                                            PythonMultiPrecisionFloat(x.clone())
                                                .into_pyobject(py)
                                                .expect("Could not convert to Python object")
                                        })
                                        .collect::<Vec<_>>(),),
                                    None,
                                )
                                .expect("Bad callback function")
                                .extract::<PythonMultiPrecisionFloat>(py)
                                .expect("Function does not return a string")
                                .0;
                            vv.set_prec(prec);
                            vv
                        })
                    })),
                ))
            })
            .collect::<PyResult<_>>()?;

        let a: PythonMultiPrecisionFloat = self
            .expr
            .evaluate(|x| x.to_multi_prec_float(prec), &constants, &functions)
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not evaluate expression: {e}"))
            })?
            .into();

        a.into_py_any(py)
    }

    /// Evaluate the expression, using a map of all the variables and
    /// user functions to a complex number.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import Expression
    /// >>> x, y = S('x', 'y')
    /// >>> e = E('sqrt(x)')*y
    /// >>> print(e.evaluate_complex({x: 1 + 2j, y: 4 + 3j}, {}))
    pub fn evaluate_complex<'py>(
        &self,
        py: Python<'py>,
        constants: HashMap<PythonExpression, Complex<f64>>,
        #[gen_stub(override_type(
            type_repr = "dict[Expression, typing.Callable[[typing.Sequence[float | complex]], float | complex]]"
        ))]
        functions: HashMap<PolyVariable, Py<PyAny>>,
    ) -> PyResult<Bound<'py, PyComplex>> {
        let constants = constants
            .iter()
            .map(|(k, v)| (k.expr.as_view(), *v))
            .collect();

        let functions = functions
            .into_iter()
            .map(|(k, v)| {
                let id = if let PolyVariable::Symbol(v) = k {
                    v
                } else {
                    Err(exceptions::PyValueError::new_err(format!(
                        "Expected function name instead of {k:?}",
                    )))?
                };

                Ok((
                    id,
                    EvaluationFn::new(Box::new(move |args: &[Complex<f64>], _, _, _| {
                        Python::attach(|py| {
                            v.call(
                                py,
                                (args
                                    .iter()
                                    .map(|x| PyComplex::from_doubles(py, x.re, x.im))
                                    .collect::<Vec<_>>(),),
                                None,
                            )
                            .expect("Bad callback function")
                            .extract::<Complex<f64>>(py)
                            .expect("Function does not return a complex number")
                        })
                    })),
                ))
            })
            .collect::<PyResult<_>>()?;

        let r = self
            .expr
            .evaluate(|x| x.into(), &constants, &functions)
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not evaluate expression: {e}"))
            })?;
        Ok(PyComplex::from_doubles(py, r.re, r.im))
    }

    /// Create an evaluator that can evaluate (nested) expressions in an optimized fashion.
    /// All constants and functions should be provided as dictionaries, where the function
    /// dictionary has a key `(name, printable name, arguments)` and the value is the function
    /// body. For example the function `f(x,y)=x^2+y` should be provided as
    /// `{(f, "f", (x, y)): x**2 + y}`. All free parameters should be provided in the `params` list.
    ///
    /// Additionally, external functions can be registered that will call a Python function.
    ///
    /// If `KeyboardInterrupt` is triggered during the optimization, the optimization will stop and will yield the
    /// current best result.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x, y, z, pi, f, g = S(
    /// >>>     'x', 'y', 'z', 'pi', 'f', 'g')
    /// >>>
    /// >>> e1 = E("x + pi + cos(x) + f(g(x+1),x*2)")
    /// >>> fd = E("y^2 + z^2*y^2")
    /// >>> gd = E("y + 5")
    /// >>>
    /// >>> ev = e1.evaluator({pi: Expression.num(22)/7},
    /// >>>              {(f, "f", (y, z)): fd, (g, "g", (y, )): gd}, [x])
    /// >>> res = ev.evaluate([[1.], [2.], [3.]])  # evaluate at x=1, x=2, x=3
    /// >>> print(res)
    ///
    ///
    /// Define an external function:
    ///
    /// >>> E("f(x)").evaluator({}, {}, [S("x")],
    ///             external_functions={(S("f"), "F"): lambda args: args[0]**2 + 1})
    ///
    /// The built-in `if` yields `x+1` when `y != 0` and `x+2` when `y == 0`:
    ///
    /// >>> E("if(y, x + 1, x + 2)").evaluator({}, {}, [S("x"), S("y")])
    ///
    /// Parameters
    /// ----------
    /// params: Sequence[Expression]
    ///     A list of free parameters.
    /// functions: dict[Tuple[Expression, str, Sequence[Expression]], Expression] = {}
    ///     A dictionary of functions. The key is a tuple of the function name, printable name and the argument variables.
    ///     The value is the function body. If the function name entry contains arguments, these are considered tags.
    /// iterations: int, optional
    ///     The number of optimization iterations to perform.
    /// cpe_iterations: Optional[int], optional
    ///     The number of common subexpression elimination iterations to perform.
    /// n_cores: int, optional
    ///     The number of cores to use for the optimization.
    /// verbose: bool, optional
    ///     Print the progress of the optimization.
    /// jit_compile: bool, optional
    ///    If set to `True`, the optimized expression will be compiled using JIT compilation
    /// direct_translation: bool, optional
    ///    If set to `True`, the optimized expression will be directly constructed from atom manipulations without building a tree.
    /// max_horner_scheme_variables: int, optional
    ///     The maximum number of variables in a Horner scheme.
    /// max_common_pair_cache_entries: int, optional
    ///     The maximum number of entries in the common pair cache.
    /// max_common_pair_distance: int, optional
    ///     The maximum distance between common pairs. Used when clearing cache entries.
    #[pyo3(signature =
        (
        params,
        functions = HashMap::default(),
        iterations = 1,
        cpe_iterations = None,
        n_cores = 4,
        verbose = false,
        jit_compile = true,
        direct_translation = true,
        max_horner_scheme_variables = 500,
        max_common_pair_cache_entries = 1_000_000,
        max_common_pair_distance = 100),
        )]
    pub fn evaluator(
        &self,
        params: Vec<PythonExpression>,
        functions: HashMap<(PolyVariable, Vec<PolyVariable>), PythonExpression>,
        iterations: usize,
        cpe_iterations: Option<usize>,
        n_cores: usize,
        verbose: bool,
        jit_compile: bool,
        direct_translation: bool,
        max_horner_scheme_variables: usize,
        max_common_pair_cache_entries: usize,
        max_common_pair_distance: usize,
        py: Python,
    ) -> PyResult<PythonExpressionEvaluator> {
        let mut fn_map = FunctionMap::new();

        for ((symbol, args), body) in functions {
            let args: Vec<_> = args
                .into_iter()
                .map(|x| match x {
                    PolyVariable::Symbol(s) => Ok(Indeterminate::Symbol(s, s.into())),
                    PolyVariable::Function(s, f) => Ok(Indeterminate::Function(s, f)),
                    _ => Err(exceptions::PyValueError::new_err(format!(
                        "Bad function argument {x} in function {symbol}",
                    ))),
                })
                .collect::<Result<_, _>>()?;

            match symbol {
                PolyVariable::Symbol(s) => {
                    fn_map
                        .add_function(s, args, body.expr)
                        .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
                }
                PolyVariable::Function(s, fa) => {
                    let tags = fa
                        .as_fun_view()
                        .unwrap()
                        .iter()
                        .map(|x| x.to_owned())
                        .collect();

                    fn_map
                        .add_tagged_function(s, tags, args, body.expr)
                        .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
                }
                _ => Err(exceptions::PyValueError::new_err(format!(
                    "Expected function name instead of {symbol:?}",
                )))?,
            }
        }

        let abort_check = Box::new(move || {
            Python::attach(|py| {
                py.check_signals().map_err(|e| {
                    if verbose {
                        if e.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) {
                            crate::info!("Ctrl-c detected. Continuing to next optimization step.");
                        } else {
                            crate::warn!(
                                "Signal received: {:?}.  Continuing to next optimization step.",
                                e
                            );
                        }
                    }
                    ()
                })
            })
            .map(|_| false)
            .unwrap_or(true)
        });

        let settings = OptimizationSettings {
            horner_iterations: iterations,
            cpe_iterations,
            n_cores,
            verbose: verbose.into(),
            abort_check: Some(abort_check),
            max_horner_scheme_variables,
            max_common_pair_cache_entries,
            max_common_pair_distance,
            direct_translation,
            ..OptimizationSettings::default()
        };

        let params: Vec<_> = params.iter().map(|x| x.expr.clone()).collect();

        let eval = py
            .detach(move || self.expr.evaluator(&fn_map, &params, settings))
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not create evaluator: {e}"))
            })?;

        Ok(PythonExpressionEvaluator {
            rational_constants: eval.get_constants().to_vec(),
            eval_complex: eval.map_coeff(&|c| Complex::new(c.re.to_f64(), c.im.to_f64())),
            eval_real: None,
            jit_real: None,
            jit_complex: None,
            eval_double_float: None,
            eval_double_float_complex: None,
            eval_arb_prec: None,
            eval_arb_prec_complex: None,
            jit_compile,
        })
    }

    /// Create an evaluator that can jointly evaluate (nested) expressions in an optimized fashion.
    /// See `Expression.evaluator()` for more information.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> x = S('x')
    /// >>> e1 = E("x^2 + 1")
    /// >>> e2 = E("x^2 + 2")
    /// >>> ev = Expression.evaluator_multiple([e1, e2], {}, {}, [x])
    ///
    /// will recycle the `x^2`
    #[classmethod]
    #[pyo3(signature =
        (exprs,
        params,
        functions = HashMap::default(),
        iterations = 1,
        cpe_iterations = None,
        n_cores = 4,
        verbose = false,
        jit_compile = true,
        direct_translation = true,
        max_horner_scheme_variables = 500,
        max_common_pair_cache_entries = 1_000_000,
        max_common_pair_distance = 100)
    )]
    pub fn evaluator_multiple(
        _cls: &Bound<'_, PyType>,
        exprs: Vec<PythonExpression>,
        params: Vec<PythonExpression>,
        functions: HashMap<(PolyVariable, Vec<PolyVariable>), PythonExpression>,
        iterations: usize,
        cpe_iterations: Option<usize>,
        n_cores: usize,
        verbose: bool,
        jit_compile: bool,
        direct_translation: bool,
        max_horner_scheme_variables: usize,
        max_common_pair_cache_entries: usize,
        max_common_pair_distance: usize,
    ) -> PyResult<PythonExpressionEvaluator> {
        let mut fn_map = FunctionMap::new();

        for ((symbol, args), body) in functions {
            let args: Vec<_> = args
                .into_iter()
                .map(|x| match x {
                    PolyVariable::Symbol(s) => Ok(Indeterminate::Symbol(s, s.into())),
                    PolyVariable::Function(s, f) => Ok(Indeterminate::Function(s, f)),
                    _ => Err(exceptions::PyValueError::new_err(format!(
                        "Bad function argument {x} in function {symbol}",
                    ))),
                })
                .collect::<Result<_, _>>()?;

            match symbol {
                PolyVariable::Symbol(s) => {
                    fn_map
                        .add_function(s, args, body.expr)
                        .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
                }
                PolyVariable::Function(s, fa) => {
                    let tags = fa
                        .as_fun_view()
                        .unwrap()
                        .iter()
                        .map(|x| x.to_owned())
                        .collect();

                    fn_map
                        .add_tagged_function(s, tags, args, body.expr)
                        .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
                }
                _ => Err(exceptions::PyValueError::new_err(format!(
                    "Expected function name instead of {symbol:?}",
                )))?,
            }
        }

        let abort_check = Box::new(move || {
            Python::attach(|py| {
                py.check_signals().map_err(|e| {
                    if verbose {
                        if e.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) {
                            crate::info!("Ctrl-c detected. Continuing to next optimization step.");
                        } else {
                            crate::warn!(
                                "Signal received: {:?}. Continuing to next optimization step.",
                                e
                            );
                        }
                    }
                    ()
                })
            })
            .map(|_| false)
            .unwrap_or(true)
        });

        let settings = OptimizationSettings {
            horner_iterations: iterations,
            cpe_iterations,
            n_cores,
            verbose: verbose.into(),
            abort_check: Some(abort_check),
            max_horner_scheme_variables,
            max_common_pair_cache_entries,
            max_common_pair_distance,
            direct_translation,
            ..OptimizationSettings::default()
        };

        let params: Vec<_> = params.iter().map(|x| x.expr.clone()).collect();

        let exprs = exprs.iter().map(|x| x.expr.as_view()).collect::<Vec<_>>();

        let eval = Atom::evaluator_multiple(&exprs, &fn_map, &params, settings).map_err(|e| {
            exceptions::PyValueError::new_err(format!("Could not create evaluator: {e}"))
        })?;

        Ok(PythonExpressionEvaluator {
            rational_constants: eval.get_constants().to_vec(),
            eval_complex: eval.map_coeff(&|c| Complex::new(c.re.to_f64(), c.im.to_f64())),
            eval_real: None,
            jit_real: None,
            jit_complex: None,
            eval_double_float: None,
            eval_double_float_complex: None,
            eval_arb_prec: None,
            eval_arb_prec_complex: None,
            jit_compile,
        })
    }

    /// Canonize (products of) tensors in the expression by relabeling repeated indices.
    /// The tensors must be written as functions, with its indices as the arguments.
    /// Subexpressions, constants and open indices are supported.
    ///
    /// If the contracted indices are distinguishable (for example in their dimension),
    /// you can provide a group marker as the second element in the tuple of the index
    /// specification.
    /// This makes sure that an index will not be renamed to an index from a different group.
    ///
    /// Returns the canonical expression, as well as the external indices and ordered dummy indices
    /// appearing in the canonical expression.
    ///
    /// Examples
    /// --------
    /// g = S('g', is_symmetric=True)
    /// >>> fc = S('fc', is_cyclesymmetric=True)
    /// >>> mu1, mu2, mu3, mu4, k1 = S('mu1', 'mu2', 'mu3', 'mu4', 'k1')
    /// >>>
    /// >>> e = g(mu2, mu3)*fc(mu4, mu2, k1, mu4, k1, mu3)
    /// >>>
    /// >>> (r, external, dummy) = e.canonize_tensors([(mu1, 0), (mu2, 0), (mu3, 0), (mu4, 0)])
    /// >>> print(r)
    /// yields `g(mu1,mu2)*fc(mu1,mu3,mu2,k1,mu3,k1)`.
    fn canonize_tensors(
        &self,
        contracted_indices: Vec<(ConvertibleToExpression, ConvertibleToExpression)>,
    ) -> PyResult<(
        PythonExpression,
        Vec<(PythonExpression, PythonExpression)>,
        Vec<(PythonExpression, PythonExpression)>,
    )> {
        let contracted_indices = contracted_indices
            .into_iter()
            .map(|x| (x.0.to_expression().expr, x.1.to_expression().expr))
            .collect::<Vec<_>>();

        let r = self
            .expr
            .canonize_tensors(contracted_indices)
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not canonize tensors: {e}"))
            })?;

        Ok((
            r.canonical_form.into(),
            r.external_indices
                .into_iter()
                .map(|(t, g)| (t.into(), g.into()))
                .collect(),
            r.dummy_indices
                .into_iter()
                .map(|(t, g)| (t.into(), g.into()))
                .collect(),
        ))
    }
}

/// A raplacement, which is a pattern and a right-hand side, with optional conditions and settings.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(from_py_object, name = "Replacement", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonReplacement {
    replacement: Replacement,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonReplacement {
    #[pyo3(signature = (pattern, rhs, cond=None, non_greedy_wildcards=None, min_level=0, max_level=None, level_range=None, level_is_tree_depth=false, partial=true, allow_new_wildcards_on_rhs=false, rhs_cache_size=None))]
    #[new]
    pub fn new(
        pattern: ConvertibleToExpression,
        rhs: ConvertibleToReplaceWith,
        cond: Option<ConvertibleToPatternRestriction>,
        non_greedy_wildcards: Option<Vec<PythonExpression>>,
        min_level: usize,
        max_level: Option<usize>,
        level_range: Option<(usize, Option<usize>)>,
        level_is_tree_depth: bool,
        partial: bool,
        allow_new_wildcards_on_rhs: bool,
        rhs_cache_size: Option<usize>,
    ) -> PyResult<Self> {
        let pattern = pattern.to_expression().expr.to_pattern();
        let rhs = rhs.to_replace_with()?;

        let mut settings = MatchSettings::cached();

        if let Some(ngw) = non_greedy_wildcards {
            settings.non_greedy_wildcards = ngw
                .iter()
                .map(|x| match x.expr.as_view() {
                    AtomView::Var(v) => {
                        let name = v.get_symbol();
                        if v.get_wildcard_level() == 0 {
                            return Err(exceptions::PyTypeError::new_err(
                                "Only wildcards can be restricted.",
                            ));
                        }
                        Ok(name)
                    }
                    _ => Err(exceptions::PyTypeError::new_err(
                        "Only wildcards can be restricted.",
                    )),
                })
                .collect::<Result<_, _>>()?;
        }

        settings.level_range = level_range.unwrap_or((min_level, max_level));
        settings.partial = partial;
        settings.level_is_tree_depth = level_is_tree_depth;
        settings.allow_new_wildcards_on_rhs = allow_new_wildcards_on_rhs;

        if let Some(rhs_cache_size) = rhs_cache_size {
            settings.rhs_cache_size = rhs_cache_size;
        }

        Ok(Self {
            replacement: Replacement::new(pattern, rhs)
                .with_conditions(cond.map(|r| r.0).unwrap_or_default())
                .with_settings(settings),
        })
    }

    #[getter]
    fn pattern(&self) -> PyResult<PythonExpression> {
        Ok(self
            .replacement
            .pat
            .to_atom()
            .map_err(|e| {
                exceptions::PyValueError::new_err(format!("Could not convert pattern to atom: {e}"))
            })?
            .into())
    }
}

#[cfg(feature = "python_stubgen")]
submit! {
PyMethodsInfo {
        struct_id: std::any::TypeId::of::<PythonExpression>,
        attrs: &[],
        getters: &[],
        setters: &[],
        file: "python.rs",
        line: line!(),
        column: column!(),
        methods: &[
            MethodInfo {
            name: "symbol",
            parameters: &[
                ParameterInfo {
                    name: "name",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "is_symmetric",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_antisymmetric",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_cyclesymmetric",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_linear",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_scalar",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_real",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_integer",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_positive",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "tags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "aliases",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
                ParameterInfo {
                    name: "normalization",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<PythonTransformer>::type_input(),
                },
                ParameterInfo {
                    name: "print",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || TypeInfo::unqualified("typing.Optional[typing.Callable[..., typing.Optional[str]]]"),
                },
                ParameterInfo {
                    name: "derivative",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || TypeInfo::unqualified("typing.Optional[typing.Callable[[Expression, int], Expression]]"),
                },
                ParameterInfo {
                    name: "series",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || TypeInfo::unqualified("typing.Optional[typing.Callable[[typing.Sequence[Series]], typing.Optional[tuple[Expression, Expression]]]]"),
                },
                ParameterInfo {
                    name: "eval",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || TypeInfo::unqualified("typing.Optional[typing.Callable[[typing.Sequence[complex]], complex] | dict[str, typing.Any]]"),
                },
                ParameterInfo {
                    name: "data",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || TypeInfo::unqualified("typing.Optional[str | int | Expression | bytes | list | dict]"),
                },
            ],
            r#type: MethodType::Class,
            r#return: || PythonExpression::type_output(),
            doc:
r#"Create new symbols from `names`. Symbols can have attributes,
such as symmetries. If no attributes
are specified and the symbol was previously defined, the attributes are inherited.
Once attributes are defined on a symbol, they cannot be redefined later.

Examples
--------
Define a regular symbol and use it as a variable:
>>> x = S('x')
>>> e = x**2 + 5
>>> print(e)
x**2 + 5

Define a regular symbol and use it as a function:
>>> f = S('f')
>>> e = f(1,2)
>>> print(e)
f(1,2)


Define a symmetric function:
>>> f = S('f', is_symmetric=True)
>>> e = f(2,1)
>>> print(e)
f(1,2)


Define a linear and symmetric function:
>>> p1, p2, p3, p4 = ES('p1', 'p2', 'p3', 'p4')
>>> dot = S('dot', is_symmetric=True, is_linear=True)
>>> e = dot(p2+2*p3,p1+3*p2-p3)
dot(p1,p2)+2*dot(p1,p3)+3*dot(p2,p2)-dot(p2,p3)+6*dot(p2,p3)-2*dot(p3,p3)

Define a custom normalization function:
>>> e = S('real_log', normalization=T().replace(E("x_(exp(x1_))"), E("x1_")))
>>> E("real_log(exp(x)) + real_log(5)")

Define a custom print function:
>>> def print_mu(mu: Expression, mode: PrintMode, **kwargs) -> str | None:
>>>     if mode == PrintMode.Latex:
>>>         if mu.get_type() == AtomType.Fn:
>>>             return "\\mu_{" + ",".join(a.format() for a in mu) + "}"
>>>         else:
>>>             return "\\mu"
>>> mu = S("mu", print=print_mu)
>>> expr = E("mu + mu(1,2)")
>>> print(expr.to_latex())

If the function returns `None`, the default print function is used.

Define a custom derivative function:
>>> tag = S('tag', derivative=lambda f, index: f)
>>> x = S('x')
>>> tag(3, x).derivative(x)

Define a custom series function:
>>> def expand_tag(args: Sequence[Series]) -> tuple[Expression, Expression] | None:
>>>     return E("1/x"), args[0].to_expression()
>>> tag = S("tag", series=expand_tag)

Define a numeric evaluation function:
>>> sq = S("sq", eval={"complex": lambda args: args[0] * args[0]})
>>> x = S("x")
>>> ev = sq(x).evaluator({}, {}, [x])
>>> ev.evaluate_complex([2+0j])

Parameters
----------
name : str
    The name of the symbol
is_symmetric : Optional[bool]
    Set to true if the symbol is symmetric.
is_antisymmetric : Optional[bool]
    Set to true if the symbol is antisymmetric.
is_cyclesymmetric : Optional[bool]
    Set to true if the symbol is cyclesymmetric.
is_linear : Optional[bool]
    Set to true if the symbol is linear.
is_scalar : Optional[bool]
    Set to true if the symbol is a scalar. It will be moved out of linear functions.
is_real : Optional[bool]
    Set to true if the symbol is a real number.
is_integer : Optional[bool]
    Set to true if the symbol is an integer.
is_positive : Optional[bool]
    Set to true if the symbol is a positive number.
tags: Optional[Sequence[str]]
    A list of tags to associate with the symbol.
aliases: Optional[Sequence[str]]
    A list of aliases for the symbol.
normalization : Optional[Transformer]
    A transformer that is called after every normalization. Note that the symbol
    name cannot be used in the transformer as this will lead to a definition of the
    symbol. Use a wildcard with the same attributes instead.
print : Optional[Callable[..., Optional[str]]]:
    A function that is called when printing the variable/function, which is provided as its first argument.
    This function should return a string, or `None` if the default print function should be used.
    The custom print function takes in keyword arguments that are the same as the arguments of the `format` function.
derivative: Optional[Callable[[Expression, int], Expression]]:
    A function that is called when computing the derivative of a function in a given argument.
series: Optional[Callable[[Sequence[Series]], Optional[tuple[Expression, Expression]]]]:
    A function that is called for custom series expansion. It receives the argument series and can return
    the singular factor and regularized expression, or `None` to use the default series expansion.
eval: dict[str, Any] | None:
    Numeric evaluation function(s). The dictionary may contain:
    - `tag_count: int`: the number of leading symbolic tag arguments.
    - `cpp: str`: a C++ function definition inserted into exported C++ code for this symbol.

    For arbitrary precision evaluation of constant functions, register a function that
    maps the tags and the requested decimal precision to a number:
    - `constant`: (Sequence[Expression], int) -> Decimal | float | complex | tuple[Decimal, Decimal]]

    Evaluators for non-constant functions when `tag_count = 0`:
    - `float`: Sequence[float] -> float
    - `complex`: Sequence[complex] -> complex
    - `decimal`: Sequence[Decimal] -> Decimal
    - `decimal_complex`: Sequence[tuple[Decimal, Decimal]] -> tuple[Decimal, Decimal]

    Evaluators for non-constant functions when `tag_count > 0` are generators:
    - `float`: Sequence[Expression] -> (Sequence[float] -> float)
    - `complex`: Sequence[Expression] -> (Sequence[complex] -> complex)
    - `decimal`: Sequence[Expression] -> (Sequence[Decimal] -> Decimal)
    - `decimal_complex`: Sequence[Expression] -> (Sequence[tuple[Decimal, Decimal]] -> tuple[Decimal, Decimal])
data: str | int | Expression | bytes | list | dict | None = None
    Custom user data to associate with the symbol."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
            is_overload: true,
        },
            MethodInfo {
            name: "symbol",
            parameters: &[
                ParameterInfo {
                    name: "names",
                    kind: ParameterKind::VarPositional,
                    default: ParameterDefault::None,
                    type_info: || <&str>::type_input(),
                },
                ParameterInfo {
                    name: "is_symmetric",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_antisymmetric",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_cyclesymmetric",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_linear",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_scalar",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_real",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_integer",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "is_positive",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<bool>::type_input(),
                },
                ParameterInfo {
                    name: "tags",
                    kind: ParameterKind::PositionalOrKeyword,
                    default: ParameterDefault::Expr(NONE_ARG),
                    type_info: || Option::<Vec<String>>::type_input(),
                },
            ],
            r#type: MethodType::Class,
            r#return: || TypeInfo::unqualified("typing.Sequence[Expression]"),
            doc:
r#"Create new symbols from `names`. Symbols can have attributes,
such as symmetries. If no attributes
are specified and the symbol was previously defined, the attributes are inherited.
Once attributes are defined on a symbol, they cannot be redefined later.

Examples
--------
Define a regular symbol and use it as a variable:
>>> x = S('x')
>>> e = x**2 + 5
>>> print(e)
x**2 + 5

Define a regular symbol and use it as a function:
>>> f = S('f')
>>> e = f(1,2)
>>> print(e)
f(1,2)


Define a symmetric function:
>>> f = S('f', is_symmetric=True)
>>> e = f(2,1)
>>> print(e)
f(1,2)


Define a linear and symmetric function:
>>> p1, p2, p3, p4 = ES('p1', 'p2', 'p3', 'p4')
>>> dot = S('dot', is_symmetric=True, is_linear=True)
>>> e = dot(p2+2*p3,p1+3*p2-p3)
dot(p1,p2)+2*dot(p1,p3)+3*dot(p2,p2)-dot(p2,p3)+6*dot(p2,p3)-2*dot(p3,p3)

Parameters
----------
name : str
    The name of the symbol
is_symmetric : Optional[bool]
    Set to true if the symbol is symmetric.
is_antisymmetric : Optional[bool]
    Set to true if the symbol is antisymmetric.
is_cyclesymmetric : Optional[bool]
    Set to true if the symbol is cyclesymmetric.
is_linear : Optional[bool]
    Set to true if the symbol is linear.
is_scalar : Optional[bool]
    Set to true if the symbol is a scalar. It will be moved out of linear functions.
is_real : Optional[bool]
    Set to true if the symbol is a real number.
is_integer : Optional[bool]
    Set to true if the symbol is an integer.
is_positive : Optional[bool]
    Set to true if the symbol is a positive number.
tags: Optional[Sequence[str]]
    A list of tags to associate with the symbol."#,
            is_async: false,
            deprecated: None,
            type_ignored: None,
            is_overload: true,
        }

          ],
    }
}

#[cfg(feature = "python_stubgen")]
submit! {
    PyMethodsInfo {
        struct_id: std::any::TypeId::of::<PythonExpression>,
        attrs: &[],
        getters: &[],
        setters: &[],
        file: "python.rs",
        line: line!(),
        column: column!(),
        methods: &[
            MethodInfo {
                name: "to_polynomial",
                parameters: &[
                    ParameterInfo {
                        name: "vars",
                        kind: ParameterKind::PositionalOrKeyword,
                        default: ParameterDefault::Expr(NONE_ARG),
                        type_info: || Option::<Vec<PythonExpression>>::type_input(),
                    },
                ],
                r#type: MethodType::Instance,
                r#return: || PythonPolynomial::type_output(),
                doc:"
Convert the expression to a polynomial, optionally, with the variable ordering specified in `vars`.
All non-polynomial parts will be converted to new, independent variables.",
                is_async: false,
                deprecated: None,
                type_ignored: None,
                is_overload: true,
            },
            MethodInfo {
                name: "to_polynomial",
                parameters: &[
                    ParameterInfo {
                        name: "minimal_poly",
                        kind: ParameterKind::PositionalOrKeyword,
                        default: ParameterDefault::None,
                        type_info: || PythonPolynomial::type_input(),
                    },
                    ParameterInfo {
                        name: "vars",
                        kind: ParameterKind::PositionalOrKeyword,
                        default: ParameterDefault::Expr(NONE_ARG),
                        type_info: || Option::<Vec<PythonExpression>>::type_input(),
                    },
                ],
                r#type: MethodType::Instance,
                r#return: || PythonNumberFieldPolynomial::type_output(),
                doc: "
Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`.
All non-polynomial elements will be converted to new independent variables.

The coefficients will be converted to a number field with the minimal polynomial `minimal_poly`.
The minimal polynomial must be a monic, irreducible univariate polynomial.",
                is_async: false,
                deprecated: None,
                type_ignored: None,
                is_overload: true,
            },
             MethodInfo {
                name: "to_polynomial",
                parameters: &[
                    ParameterInfo {
                        name: "modulus",
                        kind: ParameterKind::PositionalOrKeyword,
                        default: ParameterDefault::None,
                        type_info: || usize::type_input(),
                    },
                    ParameterInfo {
                        name: "power",
                        kind: ParameterKind::PositionalOrKeyword,
                        default: ParameterDefault::Expr(NONE_ARG),
                        type_info: || Option::<(usize, PythonExpression)>::type_input(),
                    },
                    ParameterInfo {
                        name: "minimal_poly",
                        kind: ParameterKind::PositionalOrKeyword,
                        default: ParameterDefault::Expr(NONE_ARG),
                        type_info: || Option::<PythonPolynomial>::type_input(),
                    },
                    ParameterInfo {
                        name: "vars",
                        kind: ParameterKind::PositionalOrKeyword,
                        default: ParameterDefault::Expr(NONE_ARG),
                        type_info: || Option::<Vec<PythonExpression>>::type_input(),
                    },
                ],
                r#type: MethodType::Instance,
                r#return: || PythonFiniteFieldPolynomial::type_output(),
                doc: "
Convert the expression to a polynomial, optionally, with the variables and the ordering specified in `vars`.
All non-polynomial elements will be converted to new independent variables.

The coefficients will be converted to finite field elements modulo `modulus`.
If on top a `power` is provided, for example `(2, a)`, the polynomial will be converted to the Galois field
`GF(modulus^2)` where `a` is the variable of the minimal polynomial of the field.

If a `minimal_poly` is provided, the Galois field will be created with `minimal_poly` as the minimal polynomial.",
                is_async: false,
                deprecated: None,
                type_ignored: None,
                is_overload: true,
            }
        ],
    }
}

#[cfg(feature = "python_stubgen")]
submit! {
    PyMethodsInfo {
        struct_id: std::any::TypeId::of::<PythonExpression>,
        attrs: &[],
        getters: &[],
        setters: &[],
        file: "python.rs",
        line: line!(),
        column: column!(),
        methods: &[
            MethodInfo {
                name: "__call__",
                parameters: &[
                    ParameterInfo {
                        name: "args",
                        kind: ParameterKind::VarPositional,
                        default: ParameterDefault::None,
                        type_info: || ConvertibleToExpression::type_input(),
                    },
                ],
                r#type: MethodType::Instance,
                r#return: || PythonExpression::type_output(),
                doc:"
Create a Symbolica expression by calling the function with appropriate arguments.

Examples
-------
>>> x, f = S('x', 'f')
>>> e = f(3,x)
>>> print(e)
f(3,x)",
                is_async: false,
                deprecated: None,
                type_ignored: None,
                is_overload: true,
            },
            MethodInfo {
                name: "__call__",
                parameters: &[
                    ParameterInfo {
                        name: "args",
                        kind: ParameterKind::VarPositional,
                        default: ParameterDefault::None,
                        type_info: || PythonHeldExpression::type_input() | ConvertibleToExpression::type_input(),
                    },
                ],
                r#type: MethodType::Instance,
                r#return: || PythonHeldExpression::type_output(),
                doc: "
Create a Symbolica held expression by calling the function with appropriate arguments.

Examples
-------
>>> x, f = S('x', 'f')
>>> e = f(3,x)
>>> print(e)
f(3,x)",
                is_async: false,
                deprecated: None,
                type_ignored: None,
                is_overload: true,
            }
        ],
    }
}

/// An enum that can be either a series or an expression.
#[derive(FromPyObject)]
pub enum SeriesOrExpression {
    Series(PythonSeries),
    Expression(PythonExpression),
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(SeriesOrExpression = PythonSeries | PythonExpression);
