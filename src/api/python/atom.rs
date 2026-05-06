use super::*;

#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass_enum)]
#[pyclass(
    from_py_object,
    name = "AtomType",
    eq,
    eq_int,
    module = "symbolica.core"
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
/// Specifies the type of the atom.
pub enum PythonAtomType {
    /// The expression is a number.
    Num,
    /// The expression is a variable.
    Var,
    /// The expression is a function.
    Fn,
    /// The expression is a sum.
    Add,
    /// The expression is a product.
    Mul,
    /// The expression is a power.
    Pow,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass_enum)]
#[pyclass(
    from_py_object,
    name = "SymbolAttribute",
    eq,
    eq_int,
    module = "symbolica.core"
)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
/// Specifies the attributes of a symbol.
pub enum PythonSymbolAttribute {
    /// The function is symmetric.
    Symmetric,
    /// The function is antisymmetric.
    Antisymmetric,
    /// The function is cyclesymmetric.
    Cyclesymmetric,
    /// The function is linear.
    Linear,
    /// The symbol represents a scalar. It will be moved out of linear functions.
    Scalar,
    /// The symbol represents a real number.
    Real,
    /// The symbol represents an integer.
    Integer,
    /// The symbol represents a positive number.
    Positive,
}

impl From<SymbolAttribute> for PythonSymbolAttribute {
    fn from(attr: SymbolAttribute) -> Self {
        match attr {
            SymbolAttribute::Symmetric => PythonSymbolAttribute::Symmetric,
            SymbolAttribute::Antisymmetric => PythonSymbolAttribute::Antisymmetric,
            SymbolAttribute::Cyclesymmetric => PythonSymbolAttribute::Cyclesymmetric,
            SymbolAttribute::Linear => PythonSymbolAttribute::Linear,
            SymbolAttribute::Scalar => PythonSymbolAttribute::Scalar,
            SymbolAttribute::Real => PythonSymbolAttribute::Real,
            SymbolAttribute::Integer => PythonSymbolAttribute::Integer,
            SymbolAttribute::Positive => PythonSymbolAttribute::Positive,
        }
    }
}

/// A Python representation of a Symbolica expression.
/// The type of the atom is provided in `atom_type`.
///
/// The `head` contains the string representation of:
/// - a number if the type is `Num`
/// - the variable if the type is `Var`
/// - the function name if the type is `Fn`
/// - otherwise it is `None`.
///
/// The tail contains the child atoms:
/// - the summand for type `Add`
/// - the factors for type `Mul`
/// - the base and exponent for type `Pow`
/// - the function arguments for type `Fn`
#[derive(Clone)]
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(from_py_object, name = "AtomTree", module = "symbolica.core")]
pub struct PythonAtomTree {
    /// The type of this atom.
    #[pyo3(get)]
    pub atom_type: PythonAtomType,
    /// The string data of this atom.
    #[pyo3(get)]
    pub head: Option<String>,
    /// The list of child atoms of this atom.
    #[pyo3(get)]
    pub tail: Vec<PythonAtomTree>,
}

impl<'a> From<AtomView<'a>> for PyResult<PythonAtomTree> {
    fn from(atom: AtomView<'a>) -> Self {
        let tree = match atom {
            AtomView::Num(_) => PythonAtomTree {
                atom_type: PythonAtomType::Num,
                head: Some(format!("{}", AtomPrinter::new(atom))),
                tail: vec![],
            },
            AtomView::Var(v) => PythonAtomTree {
                atom_type: PythonAtomType::Var,
                head: Some(v.get_symbol().get_name().to_string()),
                tail: vec![],
            },
            AtomView::Fun(f) => PythonAtomTree {
                atom_type: PythonAtomType::Fn,
                head: Some(f.get_symbol().get_name().to_string()),
                tail: f.iter().map(|x| x.into()).collect::<Result<Vec<_>, _>>()?,
            },
            AtomView::Add(a) => PythonAtomTree {
                atom_type: PythonAtomType::Add,
                head: None,
                tail: a.iter().map(|x| x.into()).collect::<Result<Vec<_>, _>>()?,
            },
            AtomView::Mul(m) => PythonAtomTree {
                atom_type: PythonAtomType::Mul,
                head: None,
                tail: m.iter().map(|x| x.into()).collect::<Result<Vec<_>, _>>()?,
            },
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                PythonAtomTree {
                    atom_type: PythonAtomType::Pow,
                    head: None,
                    tail: vec![
                        <AtomView as Into<PyResult<PythonAtomTree>>>::into(b)?,
                        <AtomView as Into<PyResult<PythonAtomTree>>>::into(e)?,
                    ],
                }
            }
        };

        Ok(tree)
    }
}

/// A Python representation of Symbolica user data that can be used as a key in a map.
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PythonUserDataKey(pub(super) UserDataKey);

impl<'py> IntoPyObject<'py> for &PythonUserDataKey {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match &self.0 {
            UserDataKey::Integer(i) => i.into_bound_py_any(py),
            UserDataKey::Atom(a) => {
                let expr: PythonExpression = a.clone().into();
                expr.into_bound_py_any(py)
            }
            UserDataKey::String(s) => s.into_bound_py_any(py),
        }
    }
}

impl<'py> FromPyObject<'_, 'py> for PythonUserDataKey {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        // TODO: allow list as key
        if let Ok(num) = ob.extract::<i64>() {
            Ok(PythonUserDataKey(UserDataKey::Integer(num)))
        } else if let Ok(a) = ob.extract::<ConvertibleToExpression>() {
            Ok(PythonUserDataKey(UserDataKey::Atom(a.to_expression().expr)))
        } else if let Ok(s) = ob.extract::<PyBackedStr>() {
            Ok(PythonUserDataKey(UserDataKey::String(s.to_string())))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to ExtendedUserDataKey",
            ))
        }
    }
}

/// A Python representation of Symbolica user data.
pub struct PythonUserData(pub(super) UserData);

#[cfg(feature = "python_stubgen")]
impl_stub_type!(PythonUserData = ConvertibleToExpression | PyBackedStr | PyDict | PyList | PyBytes);

impl<'py> FromPyObject<'_, 'py> for PythonUserData {
    type Error = PyErr;

    fn extract(ob: Borrowed<'_, 'py, pyo3::PyAny>) -> PyResult<Self> {
        if ob.extract::<Py<PyNone>>().is_ok() {
            Ok(PythonUserData(UserData::None))
        } else if let Ok(num) = ob.extract::<i64>() {
            Ok(PythonUserData(UserData::Integer(num)))
        } else if let Ok(a) = ob.extract::<ConvertibleToExpression>() {
            Ok(PythonUserData(UserData::Atom(a.to_expression().expr)))
        } else if let Ok(s) = ob.extract::<PyBackedStr>() {
            Ok(PythonUserData(UserData::String(s.to_string())))
        } else if let Ok(list) = ob.extract::<Vec<PythonUserData>>() {
            Ok(PythonUserData(UserData::List(
                list.into_iter().map(|x| x.0).collect(),
            )))
        } else if let Ok(map) = ob.extract::<HashMap<PythonUserDataKey, PythonUserData>>() {
            Ok(PythonUserData(UserData::Map(
                map.into_iter().map(|(k, v)| (k.0, v.0)).collect(),
            )))
        } else if let Ok(bytes) = ob.extract::<&[u8]>() {
            Ok(PythonUserData(UserData::Serialized(bytes.to_vec())))
        } else {
            Err(exceptions::PyTypeError::new_err(
                "Cannot convert to ExtendedUserData",
            ))
        }
    }
}

// impl<'py> IntoPyObject<'py> for PythonUserData {
//     type Target = PyObject;
//     type Output = Bound<'py, Self::Target>;
//     type Error = std::convert::Infallible;

//     fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
//         match self {
//             PythonUserData(ExtendedUserData::None) => Ok(None.into_py(py).into()),
//             PythonUserData(ExtendedUserData::Integer(i)) => Ok(i.into_py(py).into()),
//             PythonUserData(ExtendedUserData::Atom(a)) => {
//                 let expr: PythonExpression = a.into();
//                 Ok(expr.into_py(py).into())
//             }
//             PythonUserData(ExtendedUserData::String(s)) => {
//                 let ps: PyBackedStr = PyBackedStr::new(s);
//                 Ok(ps.into_py(py).into())
//             }
//             PythonUserData(ExtendedUserData::List(l)) => {
//                 let pl: Vec<PythonUserData> =
//                     l.into_iter().map(|x| PythonUserData(x)).collect();
//                 Ok(pl.into_py(py).into())
//             }
//             PythonUserData(ExtendedUserData::Map(m)) => {
//                 let pm: HashMap<PythonExtendedUserDataKey, PythonUserData> = m
//                     .into_iter()
//                     .map(|(k, v)| (k, PythonUserData(v)))
//                     .collect();
//                 Ok(pm.into_py(py).into())
//             }
//             PythonUserData(ExtendedUserData::Serialized(b)) => Ok(b.into_py(py).into()),
//         }
//     }
// }

pub(super) struct PythonBorrowedUserData<'a>(pub(super) &'a UserData);

impl<'a, 'py> IntoPyObject<'py> for PythonBorrowedUserData<'a> {
    type Target = PyAny;
    type Output = Bound<'py, Self::Target>;
    type Error = PyErr;

    fn into_pyobject(self, py: Python<'py>) -> Result<Self::Output, Self::Error> {
        match self {
            PythonBorrowedUserData(UserData::None) => PyNone::get(py).into_bound_py_any(py),
            PythonBorrowedUserData(UserData::Integer(i)) => i.into_bound_py_any(py),
            PythonBorrowedUserData(UserData::Atom(a)) => {
                let expr: PythonExpression = a.clone().into();
                expr.into_bound_py_any(py)
            }
            PythonBorrowedUserData(UserData::String(s)) => s.into_bound_py_any(py),
            PythonBorrowedUserData(UserData::List(l)) => {
                let pl: Vec<PythonBorrowedUserData> =
                    l.into_iter().map(|x| PythonBorrowedUserData(x)).collect();
                pl.into_bound_py_any(py)
            }
            PythonBorrowedUserData(UserData::Map(m)) => {
                let pm: HashMap<_, _> = m
                    .into_iter()
                    .map(|(k, v)| {
                        Ok((
                            PythonUserDataKey(k.clone()),
                            PythonBorrowedUserData(v).into_pyobject(py)?,
                        ))
                    })
                    .collect::<Result<_, PyErr>>()?;

                pm.into_bound_py_any(py)
            }
            PythonBorrowedUserData(UserData::Serialized(b)) => b.into_bound_py_any(py),
        }
    }
}

/// A pattern that is either a literal expression or a held expression.
#[derive(FromPyObject)]
pub enum ConvertibleToPattern {
    Literal(ConvertibleToExpression),
    Held(PythonHeldExpression),
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(ConvertibleToPattern = ConvertibleToExpression | PythonHeldExpression);

impl ConvertibleToPattern {
    pub fn to_pattern(self) -> PyResult<PythonHeldExpression> {
        match self {
            Self::Literal(l) => Ok(l.to_expression().expr.to_pattern().into()),
            Self::Held(e) => Ok(e),
        }
    }
}

/// A pattern that is allowed to have unbound transformers
#[derive(FromPyObject)]
pub enum ConvertibleToOpenPattern {
    Closed(ConvertibleToPattern),
    Open(PythonTransformer),
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(ConvertibleToOpenPattern = ConvertibleToPattern | PythonTransformer);

impl ConvertibleToOpenPattern {
    pub fn to_pattern(self) -> PyResult<PythonHeldExpression> {
        match self {
            Self::Closed(l) => l.to_pattern(),
            Self::Open(e) => Ok(Pattern::Transformer(Box::new((None, e.chain))).into()),
        }
    }
}

/// A replacement pattern or a mapping function.
#[derive(FromPyObject)]
pub enum ConvertibleToReplaceWith {
    Pattern(ConvertibleToPattern),
    Map(Py<PyAny>),
}

#[cfg(feature = "python_stubgen")]
pub struct ReplaceFunction;

#[cfg(feature = "python_stubgen")]
impl PyStubType for ReplaceFunction {
    fn type_output() -> TypeInfo {
        TypeInfo {
            name: "typing.Callable[[dict[Expression, Expression]], Expression] | int | float | complex | decimal.Decimal".into(),
            import: {
                let mut h = std::collections::HashSet::default();
                h.insert("decimal".into());
                h
            },
        }
    }
}

#[cfg(feature = "python_stubgen")]
impl_stub_type!(ConvertibleToReplaceWith = ConvertibleToPattern | ReplaceFunction);

impl ConvertibleToReplaceWith {
    pub fn to_replace_with(self) -> PyResult<ReplaceWith<'static>> {
        match self {
            Self::Pattern(p) => Ok(ReplaceWith::Pattern(p.to_pattern()?.expr.into())),
            Self::Map(m) => Ok(ReplaceWith::Map(Box::new(move |match_stack| {
                let match_stack: HashMap<PythonExpression, PythonExpression> = match_stack
                    .get_matches()
                    .iter()
                    .map(|x| (Atom::var(x.0).into(), x.1.to_atom().into()))
                    .collect();

                Python::attach(|py| {
                    m.call(py, (match_stack,), None)
                        .expect("Bad callback function")
                        .extract::<PythonExpression>(py)
                        .expect("Match map does not return an expression")
                })
                .expr
            }))),
        }
    }
}

/// A value that is either a single item or multiple items.
#[derive(FromPyObject)]
pub enum OneOrMultiple<T> {
    One(T),
    Multiple(Vec<T>),
}

impl<T> OneOrMultiple<T> {
    pub fn to_iter(&self) -> impl Iterator<Item = &T> {
        match self {
            OneOrMultiple::One(a) => std::slice::from_ref(a).iter(),
            OneOrMultiple::Multiple(m) => m.iter(),
        }
    }
}
