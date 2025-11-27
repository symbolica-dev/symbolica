//! Defines the core structures and functions for handling general mathematical expressions.
//!
//! This module provides the core functionality for representing and manipulating mathematical symbols and expressions.
//! It includes definitions for various types of atoms (numbers, variables, functions, etc.), as well as utilities for
//! parsing, printing, and transforming these expressions.
//!
//! # Examples
//!
//! Parse a new expression and expand it:
//!
//! ```
//! use symbolica::{atom::AtomCore, parse};
//!
//! let a = parse!("(x+1)^2");
//! let b = a.expand();
//! let r = parse!("x^2+2x+1");
//! assert_eq!(b, r);
//! ```
//!
//! Create a new symbol and use it in an expression:
//!
//! ```
//! use symbolica::{atom::Atom, parse, symbol};
//!
//! let x = symbol!("x");
//! let expr = Atom::var(x) + 1;
//! let p = parse!("x + 1");
//! assert_eq!(expr, p);
//! ```
//!
//! Define a function with attributes and use it in an expression:
//!
//! ```
//! use symbolica::{function, parse, symbol};
//! use symbolica::atom::{Symbol, SymbolAttribute, Atom, AtomCore};
//!
//! let f = symbol!("f"; Symmetric);
//! let expr = function!(f, 3, 2) + (1, 4);
//! let p = parse!("f(2,3) + 1/4");
//! assert_eq!(expr, p);
//! ```
mod coefficient;
mod core;
pub mod representation;

use ahash::HashMap;
use colored::Colorize;
use smartstring::{LazyCompact, SmartString};

use crate::{
    coefficient::Coefficient,
    domains::{float::Complex, rational::Rational},
    parser::{ParseSettings, Token},
    printer::{AtomPrinter, PrintFunction, PrintOptions},
    state::{RecycledAtom, State, SymbolData, Workspace},
    transformer::StatsOptions,
    utils::{BorrowedOrOwned, Settable},
};

use std::{borrow::Cow, cmp::Ordering, hash::Hash, ops::DerefMut};

pub use self::core::AtomCore;
pub use self::representation::{
    Add, AddView, Fun, InlineNum, InlineVar, KeyLookup, ListIterator, ListSlice, Mul, MulView, Num,
    NumView, Pow, PowView, Var, VarView,
};
use self::representation::{FunView, RawAtom};

/// A symbol with a namespace, and optional positional data (file and line) of its definition.
/// Can be created with the [wrap_symbol!](crate::wrap_symbol) macro or by converting from a string that is
/// written as `namespace::symbol`.
#[derive(Clone)]
pub struct NamespacedSymbol {
    pub namespace: Cow<'static, str>,
    pub symbol: Cow<'static, str>,
    pub file: Cow<'static, str>,
    pub line: usize,
}

impl NamespacedSymbol {
    /// Parse a string into a namespaced symbol.
    /// Panics if input does not contain a symbol in the format `namespace::symbol`.
    pub fn parse(s: &str) -> NamespacedSymbol {
        let (namespace, _partial_symbol) = s.rsplit_once("::").unwrap_or_else(|| {
            panic!("Input {s} does not contain a symbol in the format `namespace::symbol`.")
        });

        NamespacedSymbol {
            namespace: namespace.to_string().into(),
            symbol: s.to_string().into(),
            file: "".into(),
            line: 0,
        }
    }

    /// Parse a string into a namespaced symbol.
    /// Panics if input does not contain a symbol in the format `namespace::symbol`.
    pub fn try_parse<S: AsRef<str>>(s: S) -> Option<NamespacedSymbol> {
        let (namespace, _partial_symbol) = s.as_ref().rsplit_once("::")?;
        Some(NamespacedSymbol {
            namespace: namespace.to_string().into(),
            symbol: s.as_ref().to_string().into(),
            file: "".into(),
            line: 0,
        })
    }

    /// Parse a string into a namespaced symbol.
    /// Panics if input does not contain a symbol in the format `namespace::symbol`.
    pub fn try_parse_lit(s: &'static str) -> Option<NamespacedSymbol> {
        let (namespace, _partial_symbol) = s.rsplit_once("::")?;
        Some(NamespacedSymbol {
            namespace: namespace.into(),
            symbol: s.into(),
            file: "".into(),
            line: 0,
        })
    }
}

impl TryFrom<&str> for NamespacedSymbol {
    type Error = &'static str;

    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(NamespacedSymbol::parse(value))
    }
}

/// Wrap a symbol with the current namespace and positional data (file and line).
/// Use [symbol] or [parse] instead.
#[doc(hidden)]
#[macro_export]
macro_rules! wrap_symbol {
    ($e:literal) => {{
        if let Some(mut s) = $crate::atom::NamespacedSymbol::try_parse_lit($e) {
            s.file = file!().into();
            s.line = line!() as usize;
            s
        } else {
            let ns = if $crate::state::State::BUILTIN_SYMBOL_NAMES.contains(&$e) {
                "symbolica"
            } else {
                $crate::namespace!()
            };
            $crate::atom::NamespacedSymbol {
                symbol: format!("{}::{}", ns, $e).into(),
                namespace: ns.into(),
                file: file!().into(),
                line: line!() as usize,
            }
        }
    }};
    ($e:expr) => {{
        if let Some(mut s) = $crate::atom::NamespacedSymbol::try_parse($e) {
            s.file = file!().into();
            s.line = line!() as usize;
            s
        } else {
            let ns = if $crate::state::State::is_builtin_name(&$e) {
                "symbolica"
            } else {
                $crate::namespace!()
            };
            $crate::atom::NamespacedSymbol {
                symbol: format!("{}::{}", ns, $e).into(),
                namespace: ns.into(),
                file: file!().into(),
                line: line!() as usize,
            }
        }
    }};
}

/// A string representation of an expression with a namespace, and optional positional data (file and line).
/// Can be created with the [wrap_input!](crate::wrap_input) macro.
pub struct DefaultNamespace<T> {
    pub namespace: Cow<'static, str>,
    pub data: T,
    pub file: Cow<'static, str>,
    pub line: usize,
}

impl<T> DefaultNamespace<T> {
    /// Parse a string into a namespaced string.
    pub fn attach_namespace(&self, s: &str) -> NamespacedSymbol {
        if let Some(mut s) = NamespacedSymbol::try_parse(s) {
            s.file = self.file.clone();
            s.line = self.line;
            s
        } else if State::BUILTIN_SYMBOL_NAMES.contains(&s) {
            NamespacedSymbol {
                symbol: format!("symbolica::{s}").into(),
                namespace: "symbolica".into(),
                file: "".into(),
                line: 0,
            }
        } else {
            NamespacedSymbol {
                symbol: format!("{}::{}", self.namespace, s).into(),
                namespace: self.namespace.clone(),
                file: self.file.clone(),
                line: self.line,
            }
        }
    }
}

#[doc(hidden)]
#[macro_export]
macro_rules! wrap_input {
    ($e:expr) => {
        $crate::atom::DefaultNamespace {
            data: $e,
            namespace: $crate::namespace!().into(),
            file: file!().into(),
            line: line!() as usize,
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! with_default_namespace {
    ($e:expr, $namespace: expr) => {
        $crate::atom::DefaultNamespace {
            data: $e,
            namespace: $namespace.into(),
            file: file!().into(),
            line: line!() as usize,
        }
    };
}

/// Get the current namespace, based on the location of the macro invocation.
#[macro_export]
macro_rules! namespace {
    () => {{ env!("CARGO_CRATE_NAME") }};
}

/// Hide the current namespace when printing an atom.
#[macro_export]
macro_rules! hide_namespace {
    ($e:expr) => {
        $crate::atom::AtomCore::printer(
            &$e,
            $crate::printer::PrintOptions {
                hide_namespace: Some($crate::namespace!()),
                ..$crate::printer::PrintOptions::new()
            },
        )
    };
}

/// A function that is called after normalization of the arguments.
/// Additional modifications on `view` can be made and the result
/// can be written into `out`.
/// If no further normalization is needed, `out` must be left untouched.
///
/// # Examples
///
/// ```
/// use symbolica::atom::{Atom, AtomView, NormalizationFunction};
///
/// let normalize_fn: NormalizationFunction = Box::new(|view, out| {
///     // Example normalization logic
///     if view.is_zero() {
///         out.to_num(0.into());
///     }
/// });
/// ```
pub type NormalizationFunction = Box<dyn Fn(AtomView, &mut Settable<Atom>) + Send + Sync>;

/// A function that is called when a derivative of its arguments gets taken.
/// The argument index to derive is specified in `arg`.
/// If the default derivative should be taken, do not modify `out`.
/// Otherwise, set the `out` argument to the normalized value.
///
/// # Examples
///
/// ```
/// use symbolica::atom::{Atom, DerivativeFunction};
///
/// let derivative_fn: DerivativeFunction = Box::new(|view, arg, out| {
///     // Example derivative logic for a transparent function
///     out.to_num(1.into());
/// });
/// ```
pub type DerivativeFunction = Box<dyn Fn(AtomView, usize, &mut Settable<Atom>) + Send + Sync>;

/// Attributes that can be assigned to symbols.
#[derive(Debug, Clone, PartialEq)]
pub enum SymbolAttribute {
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

/// A symbol, for example the name of a variable or the name of a function,
/// together with its properties.
///
/// Every symbol has a namespace, which is either assigned explicitly
/// as `namespace::symbol` or is assigned by the [symbol!](crate::symbol) or
/// [parse!](crate::parse) macros based on the location of the macro invocation.
///
/// # Examples
///
/// ```
/// use symbolica::symbol;
///
/// let x = symbol!("x");
/// let (x, y) = symbol!("x", "y");
/// let f = symbol!("f"; Symmetric);
/// ```
#[derive(Copy, Clone, Hash, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
    trait_decode(trait = crate::state::HasStateMap),
)]
pub struct Symbol {
    id: u32,
    wildcard_level: u8,
    is_symmetric: bool,
    is_antisymmetric: bool,
    is_cyclesymmetric: bool,
    is_linear: bool,
    is_scalar: bool,
    is_real: bool,
    is_integer: bool,
    is_positive: bool,
}

impl std::fmt::Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if f.alternate() {
            let data = self.get_data();
            write!(
                f,
                "Symbol(name: {}, id: {}, attributes: {:?}, tags: {:?})",
                data.name,
                self.id,
                self.get_attributes(),
                data.tags
            )
        } else {
            self.format(&PrintOptions::file(), f)
        }
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.format(&PrintOptions::from_fmt(f), f)
    }
}

impl<T: AtomCore> PartialEq<T> for Symbol {
    fn eq(&self, other: &T) -> bool {
        match other.as_atom_view() {
            AtomView::Var(v) => *self == v.get_symbol(),
            _ => false,
        }
    }
}

/// A builder for creating symbols with optional attributes.
pub struct SymbolBuilder {
    symbol: NamespacedSymbol,
    attributes: Option<Cow<'static, [SymbolAttribute]>>,
    tags: Vec<String>,
    normalization_function: Option<NormalizationFunction>,
    print_function: Option<PrintFunction>,
    derivative_function: Option<DerivativeFunction>,
}

impl SymbolBuilder {
    /// Create a new symbol builder with the given name and namespace.
    /// Use the [symbol!](crate::symbol) macro instead to define symbols with the current namespace.
    pub fn new(symbol: NamespacedSymbol) -> Self {
        SymbolBuilder {
            symbol,
            attributes: None,
            tags: vec![],
            normalization_function: None,
            print_function: None,
            derivative_function: None,
        }
    }

    /// Set symbol attributes.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::{atom::{Symbol, SymbolAttribute}, wrap_symbol};
    ///
    /// let f = Symbol::new(wrap_symbol!("f")).with_attributes(&[SymbolAttribute::Symmetric]).build().unwrap();
    /// ```
    pub fn with_attributes(
        mut self,
        attributes: impl Into<Cow<'static, [SymbolAttribute]>>,
    ) -> Self {
        self.attributes = Some(attributes.into());
        self
    }

    /// Set symbol tag. The tag must contain a namespace.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::{atom::{Symbol, SymbolAttribute}, wrap_symbol};
    ///
    /// let f = Symbol::new(wrap_symbol!("f")).with_tags(["tag::real"]).build().unwrap();
    /// ```
    pub fn with_tags<T: AsRef<[U]>, U: AsRef<str>>(mut self, tags: T) -> Self {
        self.tags = tags.as_ref().iter().map(|x| x.as_ref().into()).collect();
        self
    }

    /// ```
    /// use symbolica::{atom::{AtomView, Symbol}, wrap_symbol};
    ///
    /// let f = Symbol::new(wrap_symbol!("f")).with_normalization_function(|view, out| {
    ///     // Example normalization logic that sets odd-length function to 0
    ///     if let AtomView::Fun(f) = view {
    ///         if f.get_nargs() % 2 == 1 {
    ///             out.to_num(0.into());
    ///         }
    ///     } else {
    ///         unreachable!()
    ///     }
    /// }).build().unwrap();
    /// ```
    pub fn with_normalization_function(
        mut self,
        normalization_function: impl Fn(AtomView, &mut Settable<Atom>) + Send + Sync + 'static,
    ) -> Self {
        self.normalization_function = Some(Box::new(normalization_function));
        self
    }

    /// ```
    /// use symbolica::{atom::Symbol, wrap_symbol};
    ///
    /// let f = Symbol::new(wrap_symbol!("mu")).with_print_function(|view, opt| {
    ///     if !opt.mode.is_latex() {
    ///       None
    ///     } else {
    ///        Some("\\mu".to_string())
    ///     }
    /// }).build().unwrap();
    /// ```
    pub fn with_print_function(
        mut self,
        print_function: impl Fn(AtomView, &PrintOptions) -> Option<String> + Send + Sync + 'static,
    ) -> Self {
        self.print_function = Some(Box::new(print_function));
        self
    }

    /// Add a derivative function.
    ///
    /// # Example
    ///
    /// ```
    /// use symbolica::{atom::{Atom, Symbol}, wrap_symbol};
    ///
    /// let f = Symbol::new(wrap_symbol!("tag")).with_derivative_function(|view, arg, out| {
    ///       if arg == 1 {
    ///          out.to_num(1.into());
    ///       } else {
    ///          out.to_num(0.into());
    ///       }
    /// }).build().unwrap();
    /// ```
    pub fn with_derivative_function(
        mut self,
        derivative_function: impl Fn(AtomView, usize, &mut Settable<Atom>) + Send + Sync + 'static,
    ) -> Self {
        self.derivative_function = Some(Box::new(derivative_function));
        self
    }

    /// Create a new symbol or return the existing symbol with the same name.
    ///
    /// This function will return an error when an existing symbol is redefined
    /// with different attributes.
    pub fn build(self) -> Result<Symbol, SmartString<LazyCompact>> {
        self.build_with_state(&mut State::get_state_mut())
    }

    pub(crate) fn build_with_state(
        self,
        state: &mut State,
    ) -> Result<Symbol, SmartString<LazyCompact>> {
        let (namespace, partial_symbol) =
            self.symbol.symbol.rsplit_once("::").ok_or_else(|| {
                SmartString::from(format!(
                    "Input {} does not contain a symbol in the format `namespace::symbol`.",
                    self.symbol.symbol,
                ))
            })?;

        for tag in &self.tags {
            if !tag.contains("::") {
                return Err(format!("Tag {} must contain a namespace", tag).into());
            }
        }

        Token::check_symbol_namespace(namespace)?;
        Token::check_symbol_name(partial_symbol)?;

        if self.attributes.is_none()
            && self.normalization_function.is_none()
            && self.print_function.is_none()
            && self.derivative_function.is_none()
            && self.tags.is_empty()
        {
            state.get_symbol(self.symbol)
        } else {
            state.get_symbol_with_attributes(
                self.symbol,
                self.attributes.as_ref().map(|x| x.as_ref()).unwrap_or(&[]),
                self.normalization_function,
                self.print_function,
                self.derivative_function,
                self.tags,
            )
        }
    }
}

impl Symbol {
    /// The built-in function represents a list of function arguments.
    pub const ARG: Symbol = State::ARG;
    /// The built-in function that converts a rational polynomial to a coefficient.
    pub const COEFF: Symbol = State::COEFF;
    /// The exponent function.
    pub const EXP: Symbol = State::EXP;
    /// The logarithm function.
    pub const LOG: Symbol = State::LOG;
    /// The sine function.
    pub const SIN: Symbol = State::SIN;
    /// The cosine function.
    pub const COS: Symbol = State::COS;
    /// The square root function.
    pub const SQRT: Symbol = State::SQRT;
    /// The complex conjugate function.
    pub const CONJ: Symbol = State::CONJ;
    /// The built-in function that represents a logical separator of function arguments.
    pub const SEP: Symbol = State::SEP;
    /// The built-in function that represents an abstract derivative.
    pub const DERIVATIVE: Symbol = State::DERIVATIVE;
    /// The constant `𝑒`, the base of the natural logarithm.
    pub const E: Symbol = State::E;
    /// The mathematical constant `π`.
    pub const PI: Symbol = State::PI;
    /// The string representation of the constant `π`.
    pub const PI_STR: &'static str = "𝜋";
    /// The string representation of the constant `e`.
    pub const E_STR: &'static str = "𝑒";
    /// The string representation of [Symbol::SEP].
    pub const SEP_STR: &'static str = "‖";

    /// Create a builder for a new symbol with the given name and namespace.
    ///
    /// Use the [symbol!](crate::symbol) macro instead to define symbols in the current namespace.
    pub fn new(name: NamespacedSymbol) -> SymbolBuilder {
        SymbolBuilder::new(name)
    }

    /// Parse a symbol from a string with optional namespace and attributes.
    ///
    /// Use the [symbol!](crate::symbol) macro instead to define symbols in the current namespace.
    pub fn parse<T: AsRef<str>>(name: DefaultNamespace<T>) -> Result<Self, String> {
        Token::parse_symbol(
            name.data.as_ref(),
            &name,
            &mut HashMap::default(),
            &mut State::get_state_mut(),
        )
    }

    /// Looks up a symbol by its namespaced name without creating it.
    /// Use the [get_symbol!](crate::get_symbol) macro instead to define symbols in the current namespace.
    pub fn get_symbol(name: NamespacedSymbol) -> Option<Symbol> {
        State::get_global_state()
            .read()
            .unwrap()
            .fetch_symbol(name.symbol.as_ref())
    }

    /// Create a new variable from the symbol.
    pub fn to_atom(self) -> Atom {
        Atom::var(self)
    }

    /// Get the name of the symbol, which includes its namespace.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let x = symbol!("test::x");
    /// assert_eq!(x.get_name(), "test::x");
    /// ```
    pub fn get_name(&self) -> &str {
        State::get_name(*self)
    }

    /// Get the name of the symbol without the namespace.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let x = symbol!("test::x");
    /// assert_eq!(x.get_stripped_name(), "x");
    /// ```
    pub fn get_stripped_name(&self) -> &str {
        let d = self.get_data();
        &d.name[d.namespace.len() + 2..]
    }

    /// Get the internal id of the symbol.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let x = symbol!("x");
    /// println!("id = {}", x.get_id());
    /// ```
    pub fn get_id(&self) -> u32 {
        self.id
    }

    /// Get the definition location of the symbol.
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let x = symbol!("test::x");
    /// assert_eq!(x.get_namespace(), "test");
    /// ```
    pub fn get_namespace(&self) -> &'static str {
        State::get_symbol_namespace(*self)
    }

    /// Get the wildcard level of the symbol. This property
    /// is used for pattern matching.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let x = symbol!("x");
    /// let x_ = symbol!("x_");
    /// let x__ = symbol!("x__");
    /// let x___ = symbol!("x___");
    /// assert_eq!(x.get_wildcard_level(), 0);
    /// assert_eq!(x_.get_wildcard_level(), 1);
    /// assert_eq!(x__.get_wildcard_level(), 2);
    /// assert_eq!(x___.get_wildcard_level(), 3);
    /// ```
    pub fn get_wildcard_level(&self) -> u8 {
        self.wildcard_level
    }

    /// Check if the symbol is symmetric.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let f = symbol!("f"; Symmetric);
    /// assert!(f.is_symmetric());
    /// ```
    pub fn is_symmetric(&self) -> bool {
        self.is_symmetric
    }

    /// Check if the symbol is antisymmetric.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let f = symbol!("f"; Antisymmetric);
    /// assert!(f.is_antisymmetric());
    /// ```
    pub fn is_antisymmetric(&self) -> bool {
        self.is_antisymmetric
    }

    /// Check if the symbol is cyclesymmetric.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let f = symbol!("f"; Cyclesymmetric);
    /// assert!(f.is_cyclesymmetric());
    /// ```
    pub fn is_cyclesymmetric(&self) -> bool {
        self.is_cyclesymmetric
    }

    /// Check if the symbol is linear.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let f = symbol!("f"; Linear);
    /// assert!(f.is_linear());
    /// ```
    pub fn is_linear(&self) -> bool {
        self.is_linear
    }

    /// Check if the symbol is scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let f = symbol!("f"; Scalar);
    /// assert!(f.is_scalar());
    /// ```
    pub fn is_scalar(&self) -> bool {
        self.is_scalar
    }

    /// Check if the symbol is real.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let f = symbol!("f"; Real);
    /// assert!(f.is_real());
    /// ```
    pub fn is_real(&self) -> bool {
        self.is_real
    }

    /// Check if the symbol is integer.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let f = symbol!("f"; Integer);
    /// assert!(f.is_integer());
    /// ```
    pub fn is_integer(&self) -> bool {
        self.is_integer
    }

    /// Check if the symbol is positive.
    ///
    /// # Examples
    ///
    /// ```
    /// use symbolica::symbol;
    ///
    /// let f = symbol!("f"; Positive);
    /// assert!(f.is_positive());
    /// ```
    pub fn is_positive(&self) -> bool {
        self.is_positive
    }

    /// Returns `true` iff this identifier is defined by Symbolica.
    pub fn is_builtin(self) -> bool {
        State::is_builtin(self)
    }

    /// Get all tags of the symbol.
    pub fn get_tags(&self) -> &[String] {
        &self.get_data().tags
    }

    /// Check if the symbol has the tag `tag`.
    pub fn has_tag(&self, tag: impl AsRef<str>) -> bool {
        let r = tag.as_ref();
        self.get_data().tags.iter().any(|x| x == r)
    }

    /// Get the custom normalization function of the symbol, if any.
    pub fn get_normalization_function(&self) -> Option<&NormalizationFunction> {
        self.get_data().custom_normalization.as_ref()
    }

    /// Get the custom derivative function of the symbol, if any.
    pub fn get_derivative_function(&self) -> Option<&DerivativeFunction> {
        self.get_data().custom_derivative.as_ref()
    }

    /// Get the custom print function of the symbol, if any.
    pub fn get_print_function(&self) -> Option<&PrintFunction> {
        self.get_data().custom_print.as_ref()
    }

    /// Get all tags of the symbol.
    pub fn get_attributes(&self) -> Vec<SymbolAttribute> {
        let mut attrs = vec![];
        if self.is_symmetric {
            attrs.push(SymbolAttribute::Symmetric);
        }
        if self.is_antisymmetric {
            attrs.push(SymbolAttribute::Antisymmetric);
        }
        if self.is_cyclesymmetric {
            attrs.push(SymbolAttribute::Cyclesymmetric);
        }
        if self.is_linear {
            attrs.push(SymbolAttribute::Linear);
        }
        if self.is_scalar {
            attrs.push(SymbolAttribute::Scalar);
        }
        if self.is_real {
            attrs.push(SymbolAttribute::Real);
        }
        if self.is_integer {
            attrs.push(SymbolAttribute::Integer);
        }
        if self.is_positive {
            attrs.push(SymbolAttribute::Positive);
        }
        attrs
    }

    /// Test `self` has all the attributes and tags of `s`.
    ///
    /// # Example
    /// ```rust
    /// use symbolica::{atom::AtomCore, symbol, tag};
    /// let a = symbol!("symbolica::attr::x"; Linear, Antisymmetric; tags = [tag!("mytag"), "python::test2".to_string()]);
    /// let b = symbol!("symbolica::attr::y"; Linear; tags = [tag!("mytag")]);
    /// assert!(a.has_attributes_of(b));
    /// ```
    pub fn has_attributes_of(&self, s: Symbol) -> bool {
        for t in s.get_tags() {
            if !self.has_tag(t) {
                return false;
            }
        }

        (!s.is_antisymmetric() || self.is_antisymmetric())
            && (!s.is_symmetric() || self.is_symmetric())
            && (!s.is_cyclesymmetric() || self.is_cyclesymmetric())
            && (!s.is_linear() || self.is_linear())
            && (!s.is_positive() || self.is_positive())
            && (!s.is_integer() || self.is_integer())
            && (!s.is_real() || self.is_real())
            && (!s.is_scalar() || self.is_scalar())
    }

    /// Expert use: create a new variable symbol. This constructor should be used with care as there are no checks
    /// about the validity of the identifier.
    pub const fn raw_var(id: u32, wildcard_level: u8) -> Self {
        Symbol {
            id,
            wildcard_level,
            is_symmetric: false,
            is_antisymmetric: false,
            is_cyclesymmetric: false,
            is_linear: false,
            is_scalar: false,
            is_real: false,
            is_integer: false,
            is_positive: false,
        }
    }

    /// Expert use: create a new function symbol. This constructor should be used with care as there are no checks
    /// about the validity of the identifier.
    ///
    /// Sets related attributes automatically, e.g., a symbol that is marked as `integer` is also marked as `real`.
    pub const fn raw_fn(
        id: u32,
        wildcard_level: u8,
        is_symmetric: bool,
        is_antisymmetric: bool,
        is_cyclesymmetric: bool,
        is_linear: bool,
        is_scalar: bool,
        is_real: bool,
        is_integer: bool,
        is_positive: bool,
    ) -> Self {
        Symbol {
            id,
            wildcard_level,
            is_symmetric,
            is_antisymmetric,
            is_cyclesymmetric,
            is_linear,
            is_scalar,
            is_real: is_real || is_integer || is_positive,
            is_integer,
            is_positive,
        }
    }

    fn get_attributes_tuple_str(&self) -> [(&'static str, bool); 8] {
        [
            ("symmetric", self.is_symmetric),
            ("antisymmetric", self.is_antisymmetric),
            ("cyclesymmetric", self.is_cyclesymmetric),
            ("linear", self.is_linear),
            ("scalar", self.is_scalar),
            ("real", self.is_real),
            ("integer", self.is_integer),
            ("positive", self.is_positive),
        ]
    }

    /// Get the attributes of the symbol as a tuple of (attribute, bool).
    pub fn get_attributes_tuple(&self) -> [(SymbolAttribute, bool); 8] {
        [
            (SymbolAttribute::Symmetric, self.is_symmetric),
            (SymbolAttribute::Antisymmetric, self.is_antisymmetric),
            (SymbolAttribute::Cyclesymmetric, self.is_cyclesymmetric),
            (SymbolAttribute::Linear, self.is_linear),
            (SymbolAttribute::Scalar, self.is_scalar),
            (SymbolAttribute::Real, self.is_real),
            (SymbolAttribute::Integer, self.is_integer),
            (SymbolAttribute::Positive, self.is_positive),
        ]
    }

    pub fn format<W: std::fmt::Write>(
        &self,
        opts: &PrintOptions,
        f: &mut W,
    ) -> Result<(), std::fmt::Error> {
        let data = self.get_data();
        let (namespace, name) = (&data.namespace, &data.name[data.namespace.len() + 2..]);

        if let Some(custom_print) = &data.custom_print
            && let Some(s) = custom_print(InlineVar::new(*self).as_view(), opts)
        {
            f.write_str(&s)?;
            return Ok(());
        }

        if opts.mode.is_latex() {
            match *self {
                Symbol::E => f.write_char('e'),
                Symbol::PI => f.write_str("\\pi"),
                Symbol::COS => f.write_str("\\cos"),
                Symbol::SIN => f.write_str("\\sin"),
                Symbol::EXP => f.write_str("\\exp"),
                Symbol::LOG => f.write_str("\\log"),
                _ => {
                    f.write_str(name)?;
                    if !opts.hide_all_namespaces {
                        f.write_fmt(format_args!("_{{\\tiny \text{{{namespace}}}}}"))
                    } else {
                        Ok(())
                    }
                }
            }
        } else {
            if (!opts.hide_all_namespaces || opts.include_attributes)
                && !State::is_builtin(*self)
                && (opts.hide_namespace != Some(namespace) || opts.include_attributes)
            {
                if opts.color_namespace && opts.mode.is_symbolica() {
                    f.write_fmt(format_args!("{}", namespace.dimmed().italic()))?;

                    if opts.include_attributes {
                        f.write_fmt(format_args!("{}", "::{".dimmed()))?;
                        let mut first = true;
                        for (x, t) in self.get_attributes_tuple_str() {
                            if t {
                                if !first {
                                    f.write_fmt(format_args!("{}", ",".dimmed()))?;
                                }
                                first = false;
                                f.write_fmt(format_args!("{}", x.dimmed()))?;
                            }
                        }

                        if !self.get_tags().is_empty() {
                            for tag in self.get_tags() {
                                if !first {
                                    f.write_fmt(format_args!("{}", ",".dimmed()))?;
                                }
                                first = false;
                                f.write_fmt(format_args!("{}", tag.dimmed()))?;
                            }
                        }

                        f.write_fmt(format_args!("{}", "}".dimmed()))?;
                    }

                    f.write_fmt(format_args!("{}", "::".dimmed()))?;
                } else {
                    if opts.mode.is_mathematica() {
                        for part in namespace.split("::") {
                            let mut inside_full_form_unicode = false;
                            for c in part.split(Symbol::SEP_STR) {
                                if inside_full_form_unicode {
                                    f.write_fmt(format_args!("\\[{}]", c))?;
                                } else {
                                    f.write_str(c)?;
                                }
                                inside_full_form_unicode = !inside_full_form_unicode;
                            }

                            f.write_char('`')?;
                        }
                    } else {
                        f.write_fmt(format_args!("{namespace}::"))?;
                    }

                    if opts.mode.is_symbolica() && opts.include_attributes {
                        f.write_str("{")?;
                        let mut first = true;
                        for (x, t) in self.get_attributes_tuple_str() {
                            if t {
                                if !first {
                                    f.write_char(',')?;
                                }
                                first = false;
                                f.write_str(x)?;
                            }
                        }

                        if !self.get_tags().is_empty() {
                            for tag in self.get_tags() {
                                if !first {
                                    f.write_char(',')?;
                                }
                                first = false;
                                f.write_str(tag)?;
                            }
                        }

                        f.write_str("}::")?;
                    }
                }
            }

            if opts.mode.is_symbolica() && opts.color_builtin_symbols && name.ends_with('_') {
                f.write_fmt(format_args!("{}", name.cyan().italic()))
            } else if opts.mode.is_symbolica()
                && opts.color_builtin_symbols
                && State::is_builtin(*self)
            {
                f.write_fmt(format_args!("{}", name.purple()))
            } else if opts.mode.is_mathematica() {
                if State::is_builtin(*self) {
                    match *self {
                        Symbol::E => f.write_str("E"),
                        Symbol::PI => f.write_str("Pi"),
                        Symbol::COS => f.write_str("Cos"),
                        Symbol::SIN => f.write_str("Sin"),
                        Symbol::EXP => f.write_str("Exp"),
                        Symbol::LOG => f.write_str("Log"),
                        Symbol::SQRT => f.write_str("Sqrt"),
                        Symbol::CONJ => f.write_str("Conjugate"),
                        Symbol::DERIVATIVE => f.write_str("Derivative"),
                        _ => f.write_str(name),
                    }
                } else {
                    let mut inside_full_form_unicode = false;
                    for c in name.split(Symbol::SEP_STR) {
                        if inside_full_form_unicode {
                            f.write_fmt(format_args!("\\[{}]", c))?;
                        } else {
                            f.write_str(c)?;
                        }
                        inside_full_form_unicode = !inside_full_form_unicode;
                    }
                    Ok(())
                }
            } else {
                f.write_str(name)
            }
        }
    }

    /// Get data related to the symbol.
    pub(crate) fn get_data(self) -> &'static SymbolData {
        State::get_symbol_data(self)
    }
}

/// A symbol or a function.
///
/// ```rust
/// use symbolica::{atom::Indeterminate, parse, symbol};
/// let x: Indeterminate = symbol!("x").into();
/// let f: Indeterminate = parse!("f(x)").try_into().unwrap();
/// ```
#[derive(Clone, Hash, Eq, PartialOrd, Ord, Debug)]
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::Encode),
    derive(bincode_trait_derive::Decode),
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
    trait_decode(trait = crate::state::HasStateMap)
)]
pub enum Indeterminate {
    /// A symbol, for example x, y, z, etc.
    Symbol(Symbol, InlineVar),
    /// A function, for example f(x), sin(x), etc.
    Function(Symbol, Atom),
}

impl std::fmt::Display for Indeterminate {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Indeterminate::Symbol(v, _) => f.write_str(v.get_stripped_name()),
            Indeterminate::Function(_, a) => std::fmt::Display::fmt(a, f),
        }
    }
}

impl From<Symbol> for Indeterminate {
    fn from(i: Symbol) -> Indeterminate {
        Indeterminate::Symbol(i, i.into())
    }
}

impl PartialEq<Symbol> for Indeterminate {
    fn eq(&self, other: &Symbol) -> bool {
        match self {
            Indeterminate::Symbol(s, _) => s == other,
            _ => false,
        }
    }
}

impl From<Symbol> for BorrowedOrOwned<'_, Indeterminate> {
    fn from(atom: Symbol) -> Self {
        BorrowedOrOwned::Owned(Indeterminate::from(atom))
    }
}

impl<T: AtomCore> PartialEq<T> for Indeterminate {
    fn eq(&self, other: &T) -> bool {
        self.as_view() == other.as_atom_view()
    }
}

macro_rules! from_int {
    ($($t:ty),*) => {
        $(
            impl From<$t> for Atom {
                /// Convert an integer type to an atom. This will allocate memory.
                fn from(n: $t) -> Atom {
                    Atom::num(n as u64)
                }
            }
        )*
    };
}

from_int!(
    i8, i16, i32, i64, i128, isize, u8, u16, u32, u64, u128, usize
);

impl TryFrom<Atom> for Indeterminate {
    type Error = String;

    fn try_from(a: Atom) -> Result<Indeterminate, Self::Error> {
        match a {
            Atom::Var(v) => {
                let s = v.get_symbol();
                Ok(Indeterminate::Symbol(s, InlineVar::new(s)))
            }
            Atom::Fun(f) => Ok(Indeterminate::Function(f.get_symbol(), Atom::Fun(f))),
            _ => Err(format!(
                "Cannot convert {a} to a variable as it can be decomposed into a polynomial part"
            )),
        }
    }
}

impl TryFrom<Atom> for BorrowedOrOwned<'_, Indeterminate> {
    type Error = String;
    fn try_from(atom: Atom) -> Result<Self, Self::Error> {
        Ok(BorrowedOrOwned::Owned(Indeterminate::try_from(atom)?))
    }
}

impl From<Indeterminate> for Atom {
    fn from(val: Indeterminate) -> Self {
        match val {
            Indeterminate::Symbol(s, _) => Atom::var(s),
            Indeterminate::Function(_, a) => a,
        }
    }
}

impl Indeterminate {
    /// Get the head symbol of the indeterminate.
    pub fn get_symbol(&self) -> Symbol {
        match self {
            Indeterminate::Symbol(s, _) => *s,
            Indeterminate::Function(s, _) => *s,
        }
    }

    pub fn as_view(&self) -> AtomView<'_> {
        match self {
            Indeterminate::Symbol(_, v) => v.as_view(),
            Indeterminate::Function(_, a) => a.as_view(),
        }
    }
}

/// The type (variant) of an atom.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AtomType {
    Num,
    Var,
    Add,
    Mul,
    Pow,
    Fun,
}

impl std::fmt::Display for AtomType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtomType::Num => write!(f, "Num"),
            AtomType::Var => write!(f, "Var"),
            AtomType::Add => write!(f, "Add"),
            AtomType::Mul => write!(f, "Mul"),
            AtomType::Pow => write!(f, "Pow"),
            AtomType::Fun => write!(f, "Fun"),
        }
    }
}

/// The type (variant) of a slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SliceType {
    Add,
    Mul,
    Arg,
    One,
    Pow,
    Empty,
}

/// A (immutable) view of an [Atom].
pub enum AtomView<'a> {
    Num(NumView<'a>),
    Var(VarView<'a>),
    Fun(FunView<'a>),
    Pow(PowView<'a>),
    Mul(MulView<'a>),
    Add(AddView<'a>),
}

impl Clone for AtomView<'_> {
    fn clone(&self) -> Self {
        *self
    }
}

impl Copy for AtomView<'_> {}

impl Eq for AtomView<'_> {}

impl PartialOrd for AtomView<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AtomView<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.cmp(other)
    }
}

impl Hash for AtomView<'_> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            AtomView::Num(a) => a.hash(state),
            AtomView::Var(a) => a.hash(state),
            AtomView::Fun(a) => a.hash(state),
            AtomView::Pow(a) => a.hash(state),
            AtomView::Mul(a) => a.hash(state),
            AtomView::Add(a) => a.hash(state),
        }
    }
}

impl std::fmt::Display for AtomView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        AtomPrinter::new(*self).fmt(f)
    }
}

impl From<Symbol> for Atom {
    /// Convert a symbol to an atom. This will allocate memory.
    fn from(symbol: Symbol) -> Atom {
        Atom::var(symbol)
    }
}

impl From<AtomView<'_>> for Atom {
    /// Convert an `AtomView` to an atom. This will allocate memory.
    fn from(view: AtomView) -> Atom {
        view.to_owned()
    }
}

impl<'a> From<NumView<'a>> for AtomView<'a> {
    fn from(n: NumView<'a>) -> AtomView<'a> {
        AtomView::Num(n)
    }
}

impl<'a> From<VarView<'a>> for AtomView<'a> {
    fn from(n: VarView<'a>) -> AtomView<'a> {
        AtomView::Var(n)
    }
}

impl<'a> From<FunView<'a>> for AtomView<'a> {
    fn from(n: FunView<'a>) -> AtomView<'a> {
        AtomView::Fun(n)
    }
}

impl<'a> From<MulView<'a>> for AtomView<'a> {
    fn from(n: MulView<'a>) -> AtomView<'a> {
        AtomView::Mul(n)
    }
}

impl<'a> From<AddView<'a>> for AtomView<'a> {
    fn from(n: AddView<'a>) -> AtomView<'a> {
        AtomView::Add(n)
    }
}

/// A copy-on-write structure for `Atom` and `AtomView`.
#[derive(Clone, Debug)]
pub enum AtomOrView<'a> {
    Atom(Atom),
    View(AtomView<'a>),
}

impl std::fmt::Display for AtomOrView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AtomOrView::Atom(a) => a.fmt(f),
            AtomOrView::View(a) => a.fmt(f),
        }
    }
}

impl PartialEq for AtomOrView<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (AtomOrView::Atom(a), AtomOrView::Atom(b)) => a == b,
            (AtomOrView::View(a), AtomOrView::View(b)) => a == b,
            _ => self.as_view() == other.as_view(),
        }
    }
}

impl Eq for AtomOrView<'_> {}

impl PartialOrd for AtomOrView<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for AtomOrView<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        match (self, other) {
            (AtomOrView::Atom(a1), AtomOrView::Atom(a2)) => a1.as_view().cmp(&a2.as_view()),
            (AtomOrView::Atom(a1), AtomOrView::View(a2)) => a1.as_view().cmp(a2),
            (AtomOrView::View(a1), AtomOrView::Atom(a2)) => a1.cmp(&a2.as_view()),
            (AtomOrView::View(a1), AtomOrView::View(a2)) => a1.cmp(a2),
        }
    }
}

impl Hash for AtomOrView<'_> {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            AtomOrView::Atom(a) => a.as_view().hash(state),
            AtomOrView::View(a) => a.hash(state),
        }
    }
}

impl<'a, T> From<T> for AtomOrView<'a>
where
    T: Into<Coefficient>,
{
    fn from(v: T) -> AtomOrView<'a> {
        AtomOrView::Atom(Atom::num(v.into()))
    }
}

impl<'a> From<Symbol> for AtomOrView<'a> {
    fn from(s: Symbol) -> AtomOrView<'a> {
        AtomOrView::Atom(Atom::var(s))
    }
}

impl<'a> From<&'a Symbol> for AtomOrView<'a> {
    fn from(s: &'a Symbol) -> AtomOrView<'a> {
        AtomOrView::Atom(Atom::var(*s))
    }
}

impl<'a> From<Atom> for AtomOrView<'a> {
    fn from(a: Atom) -> AtomOrView<'a> {
        AtomOrView::Atom(a)
    }
}

impl<'a> From<&'a Atom> for AtomOrView<'a> {
    fn from(a: &'a Atom) -> AtomOrView<'a> {
        AtomOrView::View(a.as_view())
    }
}

impl<'a> From<AtomView<'a>> for AtomOrView<'a> {
    fn from(a: AtomView<'a>) -> AtomOrView<'a> {
        AtomOrView::View(a)
    }
}

impl<'a> From<&AtomView<'a>> for AtomOrView<'a> {
    fn from(a: &AtomView<'a>) -> AtomOrView<'a> {
        AtomOrView::View(*a)
    }
}

impl<'a> AtomOrView<'a> {
    pub fn into_owned(self) -> Atom {
        match self {
            AtomOrView::Atom(a) => a,
            AtomOrView::View(a) => a.to_owned(),
        }
    }

    pub fn as_view(&'a self) -> AtomView<'a> {
        match self {
            AtomOrView::Atom(a) => a.as_view(),
            AtomOrView::View(a) => *a,
        }
    }

    pub fn as_mut(&mut self) -> &mut Atom {
        match self {
            AtomOrView::Atom(a) => a,
            AtomOrView::View(a) => {
                let mut oa = Atom::default();
                oa.set_from_view(a);
                *self = AtomOrView::Atom(oa);
                match self {
                    AtomOrView::Atom(a) => a,
                    _ => unreachable!(),
                }
            }
        }
    }
}

impl AtomView<'_> {
    pub fn to_owned(&self) -> Atom {
        let mut a = Atom::default();
        a.set_from_view(self);
        a
    }

    pub fn clone_into(&self, target: &mut Atom) {
        target.set_from_view(self);
    }

    /// Print the view using the portable [`PrintOptions::file()`] options.
    pub fn to_plain_string(&self) -> String {
        format!("{}", self.printer(PrintOptions::file()))
    }

    /// Get the number of terms.
    pub fn nterms(&self) -> usize {
        if let AtomView::Add(a) = self {
            a.get_nargs()
        } else {
            1
        }
    }

    /// Print statistics about the operation `op`, such as its duration and term growth.
    pub fn with_stats<F: Fn(AtomView) -> Atom>(&self, op: F, o: &StatsOptions) -> Atom {
        let start_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or(std::time::Duration::from_secs(0));

        let t = std::time::Instant::now();
        let out = op(*self);
        let dt = t.elapsed();
        o.print(*self, out.as_view(), start_time, dt);
        out
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        if let AtomView::Num(n) = self {
            n.is_zero()
        } else {
            false
        }
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        if let AtomView::Num(n) = self {
            n.is_one()
        } else {
            false
        }
    }

    /// Subtract two atoms and return the buffer that contains the unnormalized result.
    fn sub_no_norm(&self, workspace: &Workspace, rhs: AtomView<'_>) -> RecycledAtom {
        let mut e = workspace.new_atom();
        let a = e.to_add();

        // TODO: check if self or rhs is add
        a.extend(*self);
        a.extend(rhs.neg_no_norm(workspace).as_view());
        e
    }

    /// Multiply two atoms and return the buffer that contains the unnormalized result.
    fn mul_no_norm(&self, workspace: &Workspace, rhs: AtomView<'_>) -> RecycledAtom {
        let mut e = workspace.new_atom();
        let a = e.to_mul();

        // TODO: check if self or rhs is mul
        a.extend(*self);
        a.extend(rhs);
        e
    }

    /// Construct `self^exp` and return the buffer that contains the unnormalized result.
    fn pow_no_norm(&self, workspace: &Workspace, exp: AtomView<'_>) -> RecycledAtom {
        let mut e = workspace.new_atom();
        e.to_pow(*self, exp);
        e
    }

    /// Divide `self` by `div` and return the buffer that contains the unnormalized result.
    fn div_no_norm(&self, workspace: &Workspace, div: AtomView<'_>) -> RecycledAtom {
        self.mul_no_norm(
            workspace,
            div.pow_no_norm(workspace, workspace.new_num(-1).as_view())
                .as_view(),
        )
    }

    /// Negate `self` and return the buffer that contains the unnormalized result.
    fn neg_no_norm(&self, workspace: &Workspace) -> RecycledAtom {
        self.mul_no_norm(workspace, workspace.new_num(-1).as_view())
    }

    /// Add `self` and `rhs`, writing the result in `out`.
    pub fn add_with_ws_into(&self, workspace: &Workspace, rhs: AtomView<'_>, out: &mut Atom) {
        self.add_normalized(rhs, workspace, out);
    }

    /// Subtract `rhs` from `self, writing the result in `out`.
    pub fn sub_with_ws_into(&self, workspace: &Workspace, rhs: AtomView<'_>, out: &mut Atom) {
        self.sub_no_norm(workspace, rhs)
            .as_view()
            .normalize(workspace, out);
    }

    /// Multiply `self` and `rhs`, writing the result in `out`.
    pub fn mul_with_ws_into(&self, workspace: &Workspace, rhs: AtomView<'_>, out: &mut Atom) {
        self.mul_no_norm(workspace, rhs)
            .as_view()
            .normalize(workspace, out);
    }

    /// Construct `self^exp`, writing the result in `out`.
    pub fn pow_with_ws_into(&self, workspace: &Workspace, exp: AtomView<'_>, out: &mut Atom) {
        self.pow_no_norm(workspace, exp)
            .as_view()
            .normalize(workspace, out);
    }

    /// Divide `self` by `div`, writing the result in `out`.
    pub fn div_with_ws_into(&self, workspace: &Workspace, div: AtomView<'_>, out: &mut Atom) {
        self.div_no_norm(workspace, div)
            .as_view()
            .normalize(workspace, out);
    }

    /// Negate `self`, writing the result in `out`.
    pub fn neg_with_ws_into(&self, workspace: &Workspace, out: &mut Atom) {
        self.neg_no_norm(workspace)
            .as_view()
            .normalize(workspace, out);
    }

    pub fn get_byte_size(&self) -> usize {
        match self {
            AtomView::Num(n) => n.get_byte_size(),
            AtomView::Var(v) => v.get_byte_size(),
            AtomView::Fun(f) => f.get_byte_size(),
            AtomView::Pow(p) => p.get_byte_size(),
            AtomView::Mul(m) => m.get_byte_size(),
            AtomView::Add(a) => a.get_byte_size(),
        }
    }
}

/// A mathematical expression.
///
/// Most operations are implemented in the [AtomCore] trait.
///
/// # Examples
///
/// Parse a new expression and expand it:
///
/// ```
/// use symbolica::{atom::AtomCore, parse};
///
/// let a = parse!("(x+1)^2");
/// let b = a.expand();
/// let r = parse!("x^2+2x+1");
/// assert_eq!(b, r);
/// ```
///
/// Create a new symbol and use it in an expression:
///
/// ```
/// use symbolica::{atom::Atom, parse, symbol};
///
/// let x = symbol!("x");
/// let expr = Atom::var(x) + 1;
/// let p = parse!("x + 1");
/// assert_eq!(expr, p);
/// ```
///
/// Define a function with attributes and use it in an expression:
///
/// ```
/// use symbolica::{function, parse, symbol};
/// use symbolica::atom::{Symbol, Atom, AtomCore};
///
/// let f = symbol!("f"; Symmetric);
/// let expr = function!(f, 3, 2) + (1, 4);
/// let p = parse!("f(2,3) + 1/4");
/// assert_eq!(expr, p);
/// ```
///
/// # Output
///
/// The output can be controlled with
/// ```
/// use symbolica::{hide_namespace, parse};
/// let a = parse!("x^2+cos(x)");
/// println!("{:+}", a); // print with a leading sign
/// println!("{:#}", a); // print the namespace in front of every variable (e.g. `test::x`)
/// println!("{}", hide_namespace!(a)); // hide the current namespace
/// println!("{:>+}", a); // print with a leading sign and print every term on a new line
/// ```
///
/// Advanced output options can be set using [PrintOptions]. Use [PrintOptions::file()]
/// to print an expression in a format that can be parsed again.
///
/// ```
/// use symbolica::{atom::AtomCore, parse, printer::PrintOptions};
/// let a = parse!("x^2+cos(x)");
/// println!("{}", a.printer(PrintOptions::latex()));
/// println!("{}", a.printer(PrintOptions::mathematica()));
/// println!("{}", a.printer(PrintOptions::file()));
/// println!("{}", a.printer(PrintOptions {
///      color_builtin_symbols: true,
///     ..PrintOptions::new()
/// }));
/// ```
#[must_use]
#[derive(Clone)]
#[cfg_attr(
    feature = "bincode",
    derive(bincode_trait_derive::BorrowDecodeFromDecode),
    trait_decode(trait = crate::state::HasStateMap)
)]
pub enum Atom {
    Num(Num),
    Var(Var),
    Fun(Fun),
    Pow(Pow),
    Mul(Mul),
    Add(Add),
    Zero,
}

impl Atom {
    /// The number suffix that represents the imaginary unit.
    /// The suffix `i` can also be used for parsing (e.g. `2+3𝑖` or `2+3i`).
    pub const I_STR: &'static str = "𝑖";

    /// The imaginary unit.
    pub fn i() -> Atom {
        Atom::num(Complex::<Rational>::new_i())
    }
}

impl Default for Atom {
    /// Create an atom that represents the number 0.
    #[inline]
    fn default() -> Self {
        Atom::Zero
    }
}

impl std::fmt::Display for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        AtomPrinter::new(self.as_view()).fmt(f)
    }
}

impl std::fmt::Debug for Atom {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.as_view().fmt(fmt)
    }
}

impl From<Num> for Atom {
    fn from(n: Num) -> Atom {
        Atom::Num(n)
    }
}

impl From<Var> for Atom {
    fn from(n: Var) -> Atom {
        Atom::Var(n)
    }
}

impl From<Add> for Atom {
    fn from(n: Add) -> Atom {
        Atom::Add(n)
    }
}

impl From<Mul> for Atom {
    fn from(n: Mul) -> Atom {
        Atom::Mul(n)
    }
}

impl From<Fun> for Atom {
    fn from(n: Fun) -> Atom {
        Atom::Fun(n)
    }
}

impl PartialEq for Atom {
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.as_view() == other.as_view()
    }
}

impl Eq for Atom {}

impl Hash for Atom {
    #[inline(always)]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.as_view().hash(state)
    }
}

impl PartialOrd for Atom {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Atom {
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_view().cmp(&other.as_view())
    }
}

impl<T: Into<Coefficient> + Clone> PartialEq<T> for Atom {
    fn eq(&self, other: &T) -> bool {
        *self == Atom::num(other.clone())
    }
}

impl<T: Into<Coefficient> + Clone> PartialOrd<T> for Atom {
    fn partial_cmp(&self, other: &T) -> Option<Ordering> {
        Some(self.cmp(&Atom::num(other.clone().into())))
    }
}

impl Atom {
    /// Create an atom that represents the number 0.
    pub fn new() -> Atom {
        Atom::default()
    }

    /// Parse an atom from a namespaced string. Prefer to use [parse!](crate::parse) instead.
    ///
    /// # Examples
    /// ```rust
    /// use symbolica::{atom::Atom, wrap_input, with_default_namespace, parser::ParseSettings};
    /// let x = Atom::parse(wrap_input!("x"), ParseSettings::default()).unwrap();
    /// let x_2 = Atom::parse(with_default_namespace!("x_2", "b"), ParseSettings::default()).unwrap();
    /// assert!(x != x_2);
    /// ```
    pub fn parse<T: AsRef<str>>(
        input: DefaultNamespace<T>,
        settings: ParseSettings,
    ) -> Result<Atom, String> {
        let mut name_map = HashMap::default();
        Workspace::get_local().with(|ws| {
            let t = Token::parse_with_atom_info(
                input.data.as_ref(),
                settings,
                Some((&input, &mut name_map, ws)),
            )?;

            // drop the input data before parsing further
            let ns = DefaultNamespace {
                data: "",
                namespace: input.namespace,
                file: input.file,
                line: input.line,
            };
            drop(input.data);

            t.to_atom(&ns, &mut name_map, ws)
        })
    }

    /// Create a new atom that represents a variable.
    #[inline]
    pub fn var(id: Symbol) -> Atom {
        Var::new(id).into()
    }

    /// Create a new atom that represents a number.
    #[inline]
    pub fn num<T: Into<Coefficient>>(num: T) -> Atom {
        let c = num.into();
        if c.is_zero() {
            Atom::Zero
        } else {
            Num::new(c).into()
        }
    }

    #[inline]
    pub fn is_zero(&self) -> bool {
        self.as_view().is_zero()
    }

    #[inline]
    pub fn is_one(&self) -> bool {
        self.as_view().is_one()
    }

    /// Get the number of terms in the expression.
    pub fn nterms(&self) -> usize {
        self.as_view().nterms()
    }

    /// Print the atom using the portable [`PrintOptions::file()`] options.
    pub fn to_plain_string(&self) -> String {
        format!("{}", self.printer(PrintOptions::file()))
    }

    /// Print statistics about the operation `op`, such as its duration and term growth.
    pub fn with_stats<F: Fn(AtomView) -> Atom>(&self, op: F, o: &StatsOptions) -> Atom {
        self.as_view().with_stats(op, o)
    }

    /// Repeatedly apply an operation on the atom until the atom no longer changes.
    pub fn repeat_map<F: Fn(AtomView) -> Atom>(&mut self, op: F) {
        let mut res;
        loop {
            res = op(self.as_view());
            if res == *self {
                break;
            }
            std::mem::swap(self, &mut res);
        }
    }

    #[inline]
    pub fn to_num(&mut self, coeff: Coefficient) -> &mut Num {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Num(Num::new_into(coeff, buffer));
        if let Atom::Num(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_var(&mut self, id: Symbol) -> &mut Var {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Var(Var::new_into(id, buffer));
        if let Atom::Var(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_fun(&mut self, id: Symbol) -> &mut Fun {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Fun(Fun::new_into(id, buffer));
        if let Atom::Fun(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_pow(&mut self, base: AtomView, exp: AtomView) -> &mut Pow {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Pow(Pow::new_into(base, exp, buffer));
        if let Atom::Pow(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_mul(&mut self) -> &mut Mul {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Mul(Mul::new_into(buffer));
        if let Atom::Mul(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline]
    pub fn to_add(&mut self) -> &mut Add {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        *self = Atom::Add(Add::new_into(buffer));
        if let Atom::Add(n) = self {
            n
        } else {
            unreachable!()
        }
    }

    #[inline(always)]
    pub fn into_raw(self) -> RawAtom {
        match self {
            Atom::Num(n) => n.into_raw(),
            Atom::Var(v) => v.into_raw(),
            Atom::Fun(f) => f.into_raw(),
            Atom::Pow(p) => p.into_raw(),
            Atom::Mul(m) => m.into_raw(),
            Atom::Add(a) => a.into_raw(),
            Atom::Zero => RawAtom::new(),
        }
    }

    #[inline(always)]
    pub fn set_from_view(&mut self, view: &AtomView) {
        let buffer = std::mem::replace(self, Atom::Zero).into_raw();
        match view {
            AtomView::Num(n) => *self = Atom::Num(Num::from_view_into(n, buffer)),
            AtomView::Var(v) => *self = Atom::Var(Var::from_view_into(v, buffer)),
            AtomView::Fun(f) => *self = Atom::Fun(Fun::from_view_into(f, buffer)),
            AtomView::Pow(p) => *self = Atom::Pow(Pow::from_view_into(p, buffer)),
            AtomView::Mul(m) => *self = Atom::Mul(Mul::from_view_into(m, buffer)),
            AtomView::Add(a) => *self = Atom::Add(Add::from_view_into(a, buffer)),
        }
    }

    #[inline(always)]
    pub fn as_view(&self) -> AtomView<'_> {
        match self {
            Atom::Num(n) => AtomView::Num(n.to_num_view()),
            Atom::Var(v) => AtomView::Var(v.to_var_view()),
            Atom::Fun(f) => AtomView::Fun(f.to_fun_view()),
            Atom::Pow(p) => AtomView::Pow(p.to_pow_view()),
            Atom::Mul(m) => AtomView::Mul(m.to_mul_view()),
            Atom::Add(a) => AtomView::Add(a.to_add_view()),
            Atom::Zero => AtomView::ZERO,
        }
    }

    #[inline(always)]
    pub(crate) fn set_normalized(&mut self, normalized: bool) {
        match self {
            Atom::Num(_) => {}
            Atom::Var(_) => {}
            Atom::Fun(a) => a.set_normalized(normalized),
            Atom::Pow(a) => a.set_normalized(normalized),
            Atom::Mul(a) => a.set_normalized(normalized),
            Atom::Add(a) => a.set_normalized(normalized),
            Atom::Zero => {}
        }
    }
}

/// A constructor of a function. Consider using the [crate::fun!] macro instead.
///
/// For example:
/// ```
/// # use symbolica::symbol;
/// # use symbolica::atom::{Atom, AtomCore, FunctionBuilder};
/// # fn main() {
/// let f_id = symbol!("f"; Symmetric);
/// let fb = FunctionBuilder::new(f_id);
/// let a = fb
///     .add_arg(&Atom::num(3))
///     .add_arg(&Atom::num(2))
///     .add_arg(&Atom::num(1))
///     .finish();
///
/// println!("{}", a);
/// # }
/// ```
#[derive(Clone)]
pub struct FunctionBuilder {
    handle: RecycledAtom,
}

impl FunctionBuilder {
    /// Create a new `FunctionBuilder`.
    pub fn new(name: Symbol) -> FunctionBuilder {
        let mut a = RecycledAtom::new();
        a.to_fun(name);
        FunctionBuilder { handle: a }
    }

    /// Add an argument to the function.
    pub fn add_arg<'a, T: Into<AtomOrView<'a>>>(mut self, arg: T) -> FunctionBuilder {
        if let Atom::Fun(f) = self.handle.deref_mut() {
            f.add_arg(arg.into().as_view());
        }

        self
    }

    /// Add multiple arguments to the function.
    pub fn add_args<'a, T>(mut self, args: &'a [T]) -> FunctionBuilder
    where
        &'a T: Into<AtomOrView<'a>>,
    {
        if let Atom::Fun(f) = self.handle.deref_mut() {
            for a in args {
                f.add_arg(a.into().as_view());
            }
        }

        self
    }

    /// Finish the function construction and return an `Atom`.
    pub fn finish(self) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut f = ws.new_atom();
            self.handle.as_view().normalize(ws, &mut f);
            f.into_inner()
        })
    }
}

/// A trait that allows to add an argument to a function builder.
pub trait FunctionArgument {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder;
}

impl FunctionArgument for Atom {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(self.as_view())
    }
}

impl FunctionArgument for &Atom {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(self.as_view())
    }
}

impl FunctionArgument for &mut Atom {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(self.as_view())
    }
}

impl FunctionArgument for AtomView<'_> {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(*self)
    }
}

impl FunctionArgument for &AtomView<'_> {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(**self)
    }
}

impl FunctionArgument for Symbol {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        let t = InlineVar::new(*self);
        f.add_arg(t.as_view())
    }
}

impl<T: Into<Coefficient> + Clone> FunctionArgument for T {
    fn add_arg_to_function_builder(&self, f: FunctionBuilder) -> FunctionBuilder {
        f.add_arg(Atom::num(self.clone()))
    }
}

/// Create a new function by providing its name as the first argument,
/// followed by the list of arguments. This macro uses [`FunctionBuilder`].
///
/// # Examples
///
/// ```
/// use symbolica::{atom::Atom, atom::Symbol, function, symbol, parse};
/// let f_id = symbol!("f");
/// let f = function!(symbol!("f"), 3, parse!("x"));
/// ```
#[macro_export]
macro_rules! function {
    ($name: expr) => {
        {
            $crate::atom::FunctionBuilder::new($name).finish()
        }
    };
    ($name: expr, $($id: expr),*) => {
        {
            let mut f = $crate::atom::FunctionBuilder::new($name);
            $(
                f = $crate::atom::FunctionArgument::add_arg_to_function_builder(&$id, f);
            )+
            f.finish()
        }
    };
}

/// Create a tag in the current project namespace if no explicit namespace is set.
/// This macro can be used in the `symbol!` macro:
/// ```
/// use symbolica::{symbol, tag};
/// let x = symbol!("tagged_x", tag = tag!("nonzero"));
/// assert_eq!(x.has_tag("symbolica::nonzero"), true);
/// ```
#[macro_export]
macro_rules! tag {
    ($name: expr) => {
        if !$name.contains("::") {
            let mut s = String::from($crate::namespace!());
            s.push_str("::");
            s.push_str($name.as_ref());
            s
        } else {
            String::from($name)
        }
    };
}

/// Create a new symbol or fetch the existing one with the same name.
/// If no namespace is specified, the symbol is created in the
/// current namespace. Use [get_symbol!] to only fetch existing symbols.
///
/// For example:
/// ```no_run
/// use symbolica::symbol;
/// let x = symbol!("x");
/// let (x, y, z) = symbol!("x", "y", "z");
/// let x_remote = symbol!("remote::x");
/// ```
///
/// Since no attributes were specified in the example above, the symbols
/// will inherit the attributes if the symbol already exists or will be
/// created with the default attributes.
///
/// You can specify attributes for the symbol, using `;` as a separator
/// between symbol names and attributes. See [SymbolAttribute] for all options.
///
/// ```no_run
/// use symbolica::symbol;
/// let x = symbol!("x"; Symmetric, Linear);
/// let (x, y, z) = symbol!("x", "y", "z"; Symmetric); // define all as symmetric
/// ```
///
/// Explicitly specifying a symbol without attributes:
/// ```no_run
/// use symbolica::symbol;
/// let x = symbol!("x";);
/// ```
/// will panic if the symbol was previously defined with attributes. Use
/// [try_symbol!] for a fallible version.
///
/// Special settings can be defined for a single symbol by following
/// the symbol name with a `,` as shown next.
///
/// ### Tags
/// You can set tags using `tag` or `tags` flags:
/// ```no_run
/// use symbolica::{symbol, tag};
/// let x = symbol!("x", tag = tag!("nonzero"));
/// let y = symbol!("y", tags = ["test::a", "test::b"]);
/// let (w, z) = symbol!("w", "z"; tags = ["test::a", "test::b"]);
/// ```
/// Tags can be used to create logical groups and can be queried and filtered on.
///
/// ### Normalization
/// You can specify a normalization function for the symbol using `norm` flag:
///
/// ```no_run
/// use symbolica::symbol;
/// use symbolica::atom::AtomView;
/// let x = symbol!("f", norm = |f, out| {
///     if let AtomView::Fun(ff) = f {
///         if ff.get_nargs() % 2 == 1 {
///            out.to_num(0.into());
///         }
///     }
/// });
/// ```
/// See [NormalizationFunction] for more details.
///
/// ### Printing
/// You can define a custom printing function using the `print` flag:
/// ```no_run
/// use symbolica::symbol;
/// use symbolica::atom::{AtomCore, AtomView};
/// use symbolica::printer::PrintState;
/// let _ = symbol!("mu", print = |a, opt| {
///     if !opt.mode.is_latex() {
///         return None; // use default printer
///     }
///
///     let mut fmt = "\\mu".to_string();
///     if let AtomView::Fun(f) = a {
///         fmt.push_str("_{");
///         let n_args = f.get_nargs();
///         for (i, a) in f.iter().enumerate() {
///             a.format(&mut fmt, opt, PrintState::new()).unwrap();
///             if i < n_args - 1 {
///                 fmt.push_str(",");
///             }
///         }
///         fmt.push_str("}");
///     }
///
///     Some(fmt)
/// });
/// ```
/// which renders the symbol/function as `\mu_{...}` in LaTeX.
/// See [PrintFunction] for more details.
///
/// ### Derivatives
/// You can define a custom derivative function using the `der` flag:
/// ```no_run
/// use symbolica::{atom::Atom, symbol};
/// let _ = symbol!("tag", der = |a, arg, out| {
///     out.set_from_view(&a); // function behaves as a tag
/// });
/// ```
/// See [DerivativeFunction] for more details.
///
/// To set special settings together with attributes, separate the attributes with another
/// `;`:
///
/// ```no_run
/// use symbolica::symbol;
/// let _ = symbol!("gamma"; Symmetric, Linear; print = |_, _| { None });
/// ```
#[macro_export]
macro_rules! symbol {
    ($id: expr) => {
        $crate::atom::Symbol::new($crate::wrap_symbol!($id)).build().unwrap()
    };
    ($id: expr; $($attr: ident),*) => {
        $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(&[$($crate::atom::SymbolAttribute::$attr,)*]).build().unwrap()
    };
    ($id: expr, $($a: tt = $value: expr),*) => {
        {
            let mut b =  $crate::atom::Symbol::new($crate::wrap_symbol!($id));

            $(
                b = $crate::symbol_set_attr!(b, $a = $value);
            )+

            b.build().unwrap()
        }
    };
    ($id: expr; $($attr: ident),+; $($a: ident = $value: expr),*) => {
        {
            let mut b =  $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(&[$($crate::atom::SymbolAttribute::$attr,)*]);

            $(
                b = $crate::symbol_set_attr!(b, $a = $value);
            )+

            b.build().unwrap()
        }
    };
    ($($id: expr),*) => {
        {
            (
                $(
                    $crate::atom::Symbol::new($crate::wrap_symbol!($id)).build().unwrap(),
                )+
            )
        }
    };
    ($($id: expr),*; tag = $tag: expr) => {
        {
                (
                $(
                    $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_tags(std::slice::from_ref(&$tag)).build().unwrap(),
                )+
            )
        }
    };
    ($($id: expr),*; tags = $tags: expr) => {
        {
                (
                $(
                    $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_tags($tags).build().unwrap(),
                )+
            )
        }
    };
    ($($id: expr),*; $($attr: ident),*) => {
        {
            macro_rules! gen_attr {
                () => {
                    &[$($crate::atom::SymbolAttribute::$attr,)*]
                };
            }

            (
                $(
                    $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(gen_attr!()).build().unwrap(),
                )+
            )
        }
    };
    ($($id: expr),*; $($attr: ident),*; tag = $tag: expr) => {
        {
            macro_rules! gen_attr {
                () => {
                    &[$($crate::atom::SymbolAttribute::$attr,)*]
                };
            }

            (
                $(
                    $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(gen_attr!()).with_tags(std::slice::from_ref(&$tag)).build().unwrap(),
                )+
            )
        }
    };
    ($($id: expr),*; $($attr: ident),*; tags = $tags: expr) => {
        {
            macro_rules! gen_attr {
                () => {
                    &[$($crate::atom::SymbolAttribute::$attr,)*]
                };
            }

            (
                $(
                    $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(gen_attr!()).with_tags($tags).build().unwrap(),
                )+
            )
        }
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! symbol_set_attr {
    () => {{}};
    ($b: expr, norm = $norm: expr) => {
        $b.with_normalization_function($norm)
    };
    ($b: expr, print = $print: expr) => {
        $b.with_print_function($print)
    };
    ($b: expr, der = $der: expr) => {
        $b.with_derivative_function($der)
    };
    ($b: expr, tag = $tag: expr) => {
        $b.with_tags(std::slice::from_ref(&$tag))
    };
    ($b: expr, tags = $tags: expr) => {
        $b.with_tags($tags)
    };
}

/// Try to create a new symbol or fetch the existing one with the same name.
/// This is a fallible version of the [symbol!] macro.
#[macro_export]
macro_rules! try_symbol {
    ($id: expr) => {
        $crate::atom::Symbol::new($crate::wrap_symbol!($id)).build()
    };
    ($id: expr; $($attr: ident),*) => {
        $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(&[$($crate::atom::SymbolAttribute::$attr,)*]).build()
    };
    ($id: expr, $($a: tt = $value: expr),*) => {
        {
            let mut b =  $crate::atom::Symbol::new($crate::wrap_symbol!($id));

            $(
                b = $crate::symbol_set_attr!(b, $a = $value);
            )+

            b.build()
        }
    };
    ($id: expr; $($attr: ident),+; $($a: ident = $value: expr),*) => {
        {
            let mut b =  $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(&[$($crate::atom::SymbolAttribute::$attr,)*]);

            $(
                b = $crate::symbol_set_attr!(b, $a = $value);
            )+

            b.build()
        }
    };
    ($($id: expr),*) => {
        {
            (
                $(
                    $crate::atom::Symbol::new($crate::wrap_symbol!($id)).build(),
                )+
            )
        }
    };
    ($($id: expr),*; $($attr: ident),*) => {
        {
            macro_rules! gen_attr {
                () => {
                    &[$($crate::atom::SymbolAttribute::$attr,)*]
                };
            }

            (
                $(
                    $crate::atom::Symbol::new($crate::wrap_symbol!($id)).with_attributes(gen_attr!()).build(),
                )+
            )
        }
    };
}

/// Looks up a symbol by its name without creating it.
/// Use [symbol!] to define new symbols.
///
/// Returns `None` if the symbol has not been defined yet.
/// ```
/// use symbolica::get_symbol;
/// let x = symbolica::get_symbol!("newsymbol::x");
/// assert!(x.is_none());
/// let sin = symbolica::get_symbol!("sin");
/// assert!(sin.is_some());
/// ```
#[macro_export]
macro_rules! get_symbol {
    ($id: expr) => {
        $crate::atom::Symbol::get_symbol($crate::wrap_symbol!($id))
    };
    ($($id: expr),*) => {
        {
            (
                $(
                    $crate::atom::Symbol::get_symbol($crate::wrap_symbol!($id)),
                )+
            )
        }
    };
}

/// Parse an atom from a string.
/// Use [parse_lit!](crate::parse_lit) to parse from literal code and
/// [try_parse!](crate::try_parse) for fallible parsing.
///
/// # Examples
/// Parse from a literal string:
/// ```
/// use symbolica::parse;
/// let a = parse!("x^2 + 5 + f(x)");
/// println!("{}", a);
/// ```
///
/// Parse a constructed string:
/// ```
/// use symbolica::parse;
/// let s = format!("x^{}", 2);
/// let a = parse!(s);
/// println!("{}", a);
/// ```
///
/// Parse using another default namespace:
/// ```
/// use symbolica::parse;
/// let a = parse!("test::x + y", default_namespace = "custom");
/// assert_eq!(a, parse!("test::x + custom::y"));
/// ```
///
/// Parse a complex number:
/// ```
/// use symbolica::parse;
/// let a = parse!("(2+3i)*x");
/// println!("{}", a);
/// ```
///
/// Parse a floating-point number in exponential notation with
/// a custom precision of 5 decimal digits:
/// ```
/// use symbolica::parse;
/// let a = parse!("1.23456e-6`5");
/// println!("{}", a);
/// ```
///
/// Parse Mathematica code:
/// ```
/// use symbolica::parse;
/// let a = parse!("Cos[x] + Sqrt[Conjugate[x]] + Test`y + Exp[x] + Log[x]", Mathematica);
/// println!("{}", a);
/// ```
///
/// Conversion from a subset of `InputForm` is supported, excluding certain operators
/// such as `.` and `->`. For maximal compatibility use `NumberMarks->False`,
/// or `FullForm` for expressions involving non-supported structures.
///
///
/// Parse with custom settings:
/// ```
/// use symbolica::{parse, parser::ParseSettings};
/// let a = parse!("Cos[x]", settings = ParseSettings::mathematica());
/// println!("{}", a);
/// ```
#[macro_export]
macro_rules! parse {
    ($($all_args:tt)*) => {
        $crate::try_parse!($($all_args)*).unwrap()
    };
}

/// Try to parse an atom from a string.
/// This is a fallible version of the [parse!](crate::parse) macro.
#[macro_export]
macro_rules! try_parse {
    ($s: expr) => {
        $crate::atom::Atom::parse(
            $crate::wrap_input!($s),
            $crate::parser::ParseSettings::symbolica(),
        )
    };
    ($s: expr, Mathematica) => {{
        $crate::atom::Atom::parse(
            $crate::wrap_input!($s),
            $crate::parser::ParseSettings::mathematica(),
        )
    }};
    ($s: expr, settings = $settings: expr) => {{ $crate::atom::Atom::parse($crate::wrap_input!($s), $settings) }};

    ($s: expr, default_namespace = $ns: expr) => {
        $crate::atom::Atom::parse(
            $crate::with_default_namespace!($s, $ns),
            $crate::parser::ParseSettings::symbolica(),
        )
    };
    ($s: expr, Mathematica, default_namespace = $ns: expr) => {{
        $crate::atom::Atom::parse(
            $crate::with_default_namespace!($s, $ns),
            $crate::parser::ParseSettings::mathematica(),
        )
    }};
    ($s: expr, settings = $settings: expr, default_namespace = $ns: expr) => {{ $crate::atom::Atom::parse($crate::with_default_namespace!($s, $ns), $settings) }};
}

/// Parse an atom from literal code. Use [parse!](crate::parse) to parse from a string.
/// Any new symbols are defined in the current namespace. Use [try_parse_lit](crate::try_parse_lit) for fallible parsing.
///
/// # Examples
/// ```
/// use symbolica::parse_lit;
/// let a = parse_lit!(x ^ 2 + 5 + f(x));
/// println!("{}", a);
/// ```
///
/// Parse using another default namespace:
/// ```
/// use symbolica::{parse, parse_lit};
/// let a = parse_lit!(test::x + y, default_namespace = "custom");
/// assert_eq!(a, parse!("test::x + custom::y"));
/// ```
#[macro_export]
macro_rules! parse_lit {
    ($s: expr) => {{
        $crate::atom::Atom::parse(
            $crate::wrap_input!(stringify!($s)),
            $crate::parser::ParseSettings::symbolica(),
        )
        .unwrap()
    }};
    ($s: expr, default_namespace = $ns: expr) => {{
        $crate::atom::Atom::parse(
            $crate::with_default_namespace!(stringify!($s), $ns),
            $crate::parser::ParseSettings::symbolica(),
        )
        .unwrap()
    }};
}

/// Try to parse an atom from literal code. Use [parse_lit!](crate::parse_lit) for parsing that panics on an error.
#[macro_export]
macro_rules! try_parse_lit {
    ($s: expr) => {{
        $crate::atom::Atom::parse(
            $crate::wrap_input!(stringify!($s)),
            $crate::parser::ParseSettings::symbolica(),
        )
    }};
    ($s: expr, default_namespace = $ns: expr) => {{
        $crate::atom::Atom::parse(
            $crate::with_default_namespace!(stringify!($s), $ns),
            $crate::parser::ParseSettings::symbolica(),
        )
    }};
}

impl Atom {
    /// Take the `self` to a numerical power `exp`
    pub fn npow<T: Into<Coefficient>>(&self, exp: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let n = ws.new_num(exp);
            let mut t = ws.new_atom();
            self.as_view()
                .pow_no_norm(ws, n.as_view())
                .as_view()
                .normalize(ws, &mut t);
            t.into_inner()
        })
    }

    /// Take the `self` to the power `exp`. Use [`Atom::npow()`] for a numerical power and [`Atom::rpow()`] for the reverse operation.
    pub fn pow<T: AtomCore>(&self, exp: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            self.as_view()
                .pow_no_norm(ws, exp.as_atom_view())
                .as_view()
                .normalize(ws, &mut t);
            t.into_inner()
        })
    }

    /// Take `base` to the power `self`.
    pub fn rpow<T: AtomCore>(&self, base: T) -> Atom {
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            base.as_atom_view()
                .pow_no_norm(ws, self.as_view())
                .as_view()
                .normalize(ws, &mut t);
            t.into_inner()
        })
    }

    /// Add the atoms in `args`.
    pub fn add_many<T: AtomCore>(args: &[T]) -> Atom {
        let mut out = Atom::new();
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            let add = t.to_add();
            for a in args {
                add.extend(a.as_atom_view());
            }

            t.as_view().normalize(ws, &mut out);
        });
        out
    }

    /// Multiply the atoms in `args`.
    pub fn mul_many<T: AtomCore>(args: &[T]) -> Atom {
        let mut out = Atom::new();
        Workspace::get_local().with(|ws| {
            let mut t = ws.new_atom();
            let add = t.to_mul();
            for a in args {
                add.extend(a.as_atom_view());
            }

            t.as_view().normalize(ws, &mut out);
        });
        out
    }
}

mod ops;

impl AsRef<Atom> for Atom {
    fn as_ref(&self) -> &Atom {
        self
    }
}

#[cfg(test)]
mod test {
    use crate::{
        atom::{Atom, AtomCore},
        function,
    };

    use super::FunctionBuilder;

    #[test]
    fn parse_macro() {
        assert_eq!(parse_lit!(x ^ 2 + 5 + f(x)), parse!("x ^ 2 + 5 + f(x)"));
    }

    #[test]
    fn debug() {
        let x = parse!("v1+f1(v2)");
        assert_eq!(
            format!("{x:#?}"),
            "AddView { data: [5, 17, 2, 13, 2, 1, 13, 3, 5, 0, 0, 0, 1, 43, 2, 1, 14] }"
        );
        assert_eq!(
            x.get_all_symbols(true),
            [symbol!("v1"), symbol!("v2"), symbol!("f1")]
                .into_iter()
                .collect(),
        );
        assert_eq!(x.as_view().get_byte_size(), 17);
    }

    #[test]
    fn composition() {
        let v1 = parse!("v1");
        let v2 = parse!("v2");
        let f1_id = symbol!("f1");

        let f1 = function!(f1_id, v1, v2, Atom::num(2));

        let r = (-(&v2 + &v1 + 2) * &v2 * 6).npow(5) / &v2.pow(&v1) * &f1 / 4;

        let res = parse!("1/4*(v2^v1)^-1*(-6*v2*(v1+v2+2))^5*f1(v1,v2,2)");
        assert_eq!(res, r);
    }

    #[test]
    fn building() {
        let _ = FunctionBuilder::new(symbol!("a"))
            .add_arg(1)
            .add_args(&[1, 2])
            .add_arg(symbol!("a"))
            .add_args(&[symbol!("b")])
            .add_args(&[parse!("a")])
            .add_arg(parse!("a"))
            .add_arg(parse!("a"))
            .add_arg(parse!("a").as_view())
            .finish();
    }
}
