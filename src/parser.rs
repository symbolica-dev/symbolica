//! Parsing of general expressions.
//!
//! Most users will want to use [crate::parse] to parse an atom.
//!
//! Use [Token::parse] to tokenize an expression, and
//! [Token::to_polynomial], [Token::to_rational_polynomial] or [Token::to_factorized_rational_polynomial] for accelerated parsing of polynomials written
//! in Symbolica's fast format.

use std::{borrow::Cow, fmt::Write, string::String, sync::Arc};

use ahash::HashMap;
use bytes::Buf;
use numerica::domains::float::ErrorPropagatingFloat;
use rug::Integer as MultiPrecisionInteger;

use smallvec::SmallVec;
use smartstring::{LazyCompact, SmartString};

use crate::{
    LicenseManager,
    atom::{Atom, DefaultNamespace, Symbol, SymbolAttribute, SymbolBuilder},
    coefficient::{Coefficient, ConvertToRing},
    domains::{
        Ring,
        float::{Complex, Float, FloatLike},
        integer::Integer,
        rational::Rational,
    },
    poly::{PolyVariable, PositiveExponent, polynomial::MultivariatePolynomial},
    state::{RecycledAtom, State, Workspace},
};

const HEX_DIGIT_MASK: [bool; 255] = [
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, true, true, true, true, true,
    true, true, true, true, true, false, false, false, false, false, false, false, true, true,
    true, true, true, true, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false,
];

const DIGIT_MASK: [bool; 255] = [
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, true, true, true, true, true,
    true, true, true, true, true, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false, false, false, false, false, false, false, false,
    false, false, false, false, false, false,
];

const HEX_TO_DIGIT: [u8; 24] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0, 10, 11, 12, 13, 14, 15, 0,
];

/// The current parsing state.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum ParseState {
    Identifier,
    Number,
    RationalPolynomial,
    Any,
}

/// An operator in the expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Operator {
    Mul,
    Add,
    Pow,
    Argument, // comma
    Neg,      // left side should be tagged as 'finished'
    Inv,      // left side should be tagged as 'finished', for internal use
}

/// The mode in which to parse the expression.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub enum ParseMode {
    #[default]
    Symbolica,
    Mathematica,
}

/// Settings for parsing.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ParseSettings {
    pub mode: ParseMode,
    /// Convert completed terms to atoms during parsing, to save memory
    pub convert_mul_to_atom: bool,
    pub distribute_neg: bool,
}

impl ParseMode {
    pub fn is_mathematica(&self) -> bool {
        *self == ParseMode::Mathematica
    }

    pub fn is_symbolica(&self) -> bool {
        *self == ParseMode::Symbolica
    }
}

impl ParseSettings {
    pub const fn symbolica() -> Self {
        ParseSettings {
            mode: ParseMode::Symbolica,
            convert_mul_to_atom: true,
            distribute_neg: false,
        }
    }

    pub const fn mathematica() -> Self {
        ParseSettings {
            mode: ParseMode::Mathematica,
            convert_mul_to_atom: true,
            distribute_neg: false,
        }
    }

    pub const fn polynomial() -> Self {
        ParseSettings {
            mode: ParseMode::Symbolica,
            convert_mul_to_atom: false,
            distribute_neg: true,
        }
    }
}

impl Default for ParseSettings {
    fn default() -> Self {
        ParseSettings {
            mode: ParseMode::Symbolica,
            convert_mul_to_atom: true,
            distribute_neg: false,
        }
    }
}

impl std::fmt::Display for Operator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operator::Mul => f.write_char('*'),
            Operator::Add => f.write_char('+'),
            Operator::Pow => f.write_char('^'),
            Operator::Argument => f.write_char(','),
            Operator::Neg => f.write_char('-'),
            Operator::Inv => f.write_char('/'),
        }
    }
}

impl Operator {
    #[inline]
    pub const fn get_arity(&self) -> usize {
        match self {
            Operator::Neg | Operator::Inv => 1,
            _ => 2,
        }
    }

    #[inline]
    pub const fn get_precedence(&self) -> u8 {
        match self {
            Operator::Mul => 8,
            Operator::Add => 7,
            Operator::Pow => 11,
            Operator::Argument => 6,
            Operator::Neg => 10,
            Operator::Inv => 9,
        }
    }

    #[inline]
    pub const fn left_associative(&self) -> bool {
        match self {
            Operator::Mul => true,
            Operator::Add => true,
            Operator::Pow => false,
            Operator::Argument => true,
            Operator::Neg => true,
            Operator::Inv => true,
        }
    }

    #[inline]
    pub const fn right_associative(&self) -> bool {
        match self {
            Operator::Mul => true,
            Operator::Add => true,
            Operator::Pow => true,
            Operator::Argument => true,
            Operator::Neg => true,
            Operator::Inv => true,
        }
    }
}

/// The position in a string, used for error messages.
pub struct Position {
    pub line_number: usize,
    pub char_pos: usize,
}

/// A token in a string.
///
/// From tokens, fast methods are available to parse
/// an expression or a polynomial.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Token {
    Number(SmartString<LazyCompact>, bool),
    SpecialNumber(char),
    ID(SmartString<LazyCompact>),
    RationalPolynomial(SmartString<LazyCompact>),
    Op(bool, bool, Operator, Vec<Token>),
    Fn(bool, bool, Vec<Token>),
    ParsedMul(Box<RecycledAtom>), // a partially parsed and converted expression
    Start,
    OpenParenthesis,
    CloseParenthesis,
    CloseBracket,
    EOF,
}

impl Token {
    const OPS: [char; 11] = ['\0', '^', '+', '*', '-', '(', ')', '/', ',', '[', ']'];
    const WHITESPACE: [char; 5] = [' ', '\t', '\n', '\r', '\\'];
    const WHITESPACE_MATHEMATICA: [char; 4] = [' ', '\t', '\n', '\r'];
    const FORBIDDEN: [char; 12] = [';', '&', '!', '%', '.', '"', '¿', '⧞', '∞', '{', '}', '`'];
}

impl std::fmt::Display for Token {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Number(n, is_imag) => {
                if *is_imag {
                    f.write_str(n)?;
                    f.write_char('𝑖')
                } else {
                    f.write_str(n)
                }
            }
            Token::SpecialNumber(c) => f.write_char(*c),
            Token::ID(v) => f.write_str(v),
            Token::RationalPolynomial(v) => {
                f.write_char('[')?;
                f.write_str(v)?;
                f.write_char(']')
            }
            Token::Op(_, _, o, m) => {
                let mut first = true;
                f.write_char('(')?;

                for mm in m {
                    if !first {
                        match o {
                            Operator::Mul => f.write_char('*')?,
                            Operator::Add => f.write_char('+')?,
                            Operator::Pow => f.write_char('^')?,
                            Operator::Argument => f.write_char(',')?,
                            Operator::Neg => f.write_char('-')?,
                            Operator::Inv => f.write_str("1/")?,
                        }
                    } else if *o == Operator::Neg {
                        f.write_char('-')?;
                    } else if *o == Operator::Inv {
                        f.write_str("1/")?;
                    }
                    first = false;

                    mm.fmt(f)?;
                }
                f.write_char(')')
            }
            Token::Fn(_, _, args) => {
                let mut first = true;

                match &args[0] {
                    Token::ID(s) => f.write_str(s)?,
                    _ => unreachable!(),
                };

                f.write_char('(')?;
                for aa in args.iter().skip(1) {
                    if !first {
                        f.write_char(',')?;
                    }
                    first = false;

                    aa.fmt(f)?;
                }
                f.write_char(')')
            }
            Token::Start => f.write_str("START"),
            Token::OpenParenthesis => f.write_char('('),
            Token::CloseParenthesis => f.write_char(')'),
            Token::CloseBracket => f.write_char(']'),
            Token::EOF => f.write_str("EOF"),
            Token::ParsedMul(a) => write!(f, "Atom({})", a),
        }
    }
}

impl Token {
    /// Check the validity of a symbol name.
    pub fn check_symbol_name(symbol: &str) -> Result<(), String> {
        if !symbol.starts_with('"') && symbol.contains(':') {
            return Err(format!("Symbol name {} cannot contain ':'", symbol));
        }

        Token::check_symbol_namespace(symbol)
    }

    /// Check the validity of a symbol namespace.
    pub fn check_symbol_namespace(symbol: &str) -> Result<(), String> {
        if symbol.is_empty() {
            return Err("Empty symbol namespace".to_string());
        }

        if symbol.starts_with(|c: char| c.is_numeric()) {
            return Err(format!("Symbol namespace '{symbol}' starts with a number"));
        }

        if symbol.starts_with('"') && symbol.ends_with('"') {
            let mut escaped = false;
            for (i, c) in symbol.chars().enumerate().skip(1).take(symbol.len() - 2) {
                if escaped {
                    escaped = false;
                } else if c == '\\' {
                    escaped = true;
                } else if c == '"' {
                    return Err(format!(
                        "Invalid character '\"' at position {} in symbol namespace '{symbol}'",
                        i
                    ));
                }
            }

            return Ok(());
        }

        for c in symbol.chars() {
            if Token::OPS.contains(&c)
                || Token::WHITESPACE.contains(&c)
                || Token::FORBIDDEN.contains(&c)
            {
                return Err(format!(
                    "Invalid character '{c}' in symbol namespace '{symbol}'"
                ));
            }
        }

        Ok(())
    }

    /// Return if the token does not require any further arguments.
    fn is_normal(&self) -> bool {
        match self {
            Token::Number(_, _) => true,
            Token::SpecialNumber(_) => true,
            Token::ID(_) => true,
            Token::RationalPolynomial(_) => true,
            Token::Op(more_left, more_right, _, _) => !more_left && !more_right,
            Token::Fn(more_right, _, _) => !more_right,
            _ => false,
        }
    }

    /// Get the precedence of the token.
    #[inline]
    const fn get_precedence(&self) -> u8 {
        match self {
            Token::Number(_, _) => 11,
            Token::SpecialNumber(_) => 11,
            Token::ID(_) => 11,
            Token::ParsedMul(_) => Operator::Mul.get_precedence(),
            Token::RationalPolynomial(_) => 11,
            Token::Op(_, _, o, _) => o.get_precedence(),
            Token::Fn(_, _, _)
            | Token::OpenParenthesis
            | Token::CloseParenthesis
            | Token::CloseBracket => 5,
            Token::Start | Token::EOF => 4,
        }
    }

    /// Add `other` to the left side of `self`, where `self` is a binary operation.
    #[inline]
    fn add_left(&mut self, other: Token) -> Result<(), String> {
        if let Token::Op(ml, _, o1, args) = self {
            debug_assert!(*ml);
            *ml = false;

            if let Token::Op(ml, mr, o2, mut args2) = other {
                debug_assert!(!ml && !mr);
                if *o1 == o2 && o2.left_associative() {
                    // add from the left by swapping and then extending from the right
                    std::mem::swap(args, &mut args2);
                    args.append(&mut args2);
                } else {
                    args.insert(0, Token::Op(false, false, o2, args2));
                }
            } else {
                args.insert(0, other);
            }
            Ok(())
        } else {
            Err(format!(
                "operator expected, but found '{self}'. Are parentheses unbalanced?"
            ))
        }
    }

    fn distribute_neg(&mut self, distribute_neg: bool) {
        match self {
            Token::Op(_, _, Operator::Neg, args) => {
                debug_assert!(args.len() == 1);
                *self = args.pop().unwrap();
            }
            Token::Op(_, _, Operator::Mul, args) => {
                if distribute_neg {
                    args[0].distribute_neg(distribute_neg);
                } else if let Token::Number(n, _) = &mut args[0] {
                    if n.starts_with('-') {
                        n.remove(0);
                    } else {
                        n.insert(0, '-');
                    }
                } else {
                    args.push(Token::Number("-1".into(), false));
                }
            }
            Token::Op(_, _, Operator::Add, args) => {
                if distribute_neg {
                    for a in args {
                        a.distribute_neg(true);
                    }
                } else {
                    let t = std::mem::replace(self, Token::EOF);
                    *self = Token::Op(false, false, Operator::Neg, vec![t]);
                }
            }
            Token::Number(n, _) => {
                if n.starts_with('-') {
                    n.remove(0);
                } else {
                    n.insert(0, '-');
                }
            }
            _ => {
                let t = std::mem::replace(self, Token::EOF);
                *self = Token::Op(false, false, Operator::Neg, vec![t]);
            }
        }
    }

    /// Add `other` to right side of `self`, where `self` is a binary operation.
    #[inline]
    fn add_right(&mut self, mut other: Token, distribute_neg: bool) -> Result<(), String> {
        if let Token::Op(_, mr, o1, args) = self {
            debug_assert!(*mr);
            *mr = false;

            if *o1 == Operator::Neg {
                if let Token::Number(n, _) = &mut other {
                    if n.starts_with('-') {
                        n.remove(0);
                    } else {
                        n.insert(0, '-');
                    }
                } else {
                    other.distribute_neg(distribute_neg);
                }
                *self = other;
                return Ok(());
            }

            if let Token::Op(ml, mr, o2, mut args2) = other {
                debug_assert!(!ml && !mr);
                if *o1 == o2 && o2.right_associative() {
                    if o2 == Operator::Neg || o2 == Operator::Inv {
                        // twice unary minus or inv cancels out
                        debug_assert!(args2.len() == 1);
                        *self = args2.pop().unwrap();
                    } else {
                        args.append(&mut args2)
                    }
                } else {
                    args.push(Token::Op(false, false, o2, args2));
                }
            } else {
                args.push(other);
            }

            Ok(())
        } else if let Token::Number(n, _) = other {
            Err(format!("operator expected between '{self}' and '{n}'"))
        } else {
            Err(format!(
                "operator expected, but found '{self}'. Are parentheses unbalanced?"
            ))
        }
    }

    /// Parse the token into an atom.
    pub fn to_atom<T>(
        &self,
        namespace: &DefaultNamespace<T>,
        name_map: &mut HashMap<SmartString<LazyCompact>, Symbol>,
        workspace: &Workspace,
    ) -> Result<Atom, String> {
        let mut atom = workspace.new_atom();

        {
            let mut state = State::get_global_state().write().unwrap();
            // do not normalize to prevent potential deadlocks
            self.to_atom_with_output_no_norm(
                namespace, name_map, &mut state, workspace, &mut atom,
            )?;
        }

        let mut out = Atom::new();
        atom.as_view().normalize(workspace, &mut out);

        Ok(out)
    }

    /// Parse a symbol with potential attributes.
    pub(crate) fn parse_symbol<T>(
        x: &str,
        namespace: &DefaultNamespace<T>,
        name_map: &mut HashMap<SmartString<LazyCompact>, Symbol>,
        state: &mut State,
    ) -> Result<Symbol, String> {
        if let Some(s) = name_map.get(x) {
            return Ok(*s);
        }

        let s = if let Some((namespace, attr)) = x.split_once("::{") {
            if let Some((attrs, symbol)) = attr.split_once("}::") {
                let mut attributes = vec![];
                let mut tags = vec![];
                for x in attrs.split(',') {
                    match x {
                        "" => {}
                        "linear" => attributes.push(SymbolAttribute::Linear),
                        "symmetric" => attributes.push(SymbolAttribute::Symmetric),
                        "antisymmetric" => attributes.push(SymbolAttribute::Antisymmetric),
                        "cyclesymmetric" => attributes.push(SymbolAttribute::Cyclesymmetric),
                        "scalar" => attributes.push(SymbolAttribute::Scalar),
                        "real" => attributes.push(SymbolAttribute::Real),
                        "integer" => attributes.push(SymbolAttribute::Integer),
                        "positive" => attributes.push(SymbolAttribute::Positive),
                        x => {
                            if x.contains("::") {
                                tags.push(x.to_string());
                            } else {
                                Err(format!(
                                    "Unknown attribute or tag: {x}. Tags must contain a namespace"
                                ))?;
                            }
                        }
                    }
                }

                let symbol_whole = format!("{}::{}", namespace, symbol);

                SymbolBuilder::new(crate::atom::NamespacedSymbol::parse(&symbol_whole))
                    .with_attributes(attributes)
                    .with_tags(tags)
                    .build_with_state(state)
                    .map_err(|e| e.to_string())
            } else {
                Err(format!("Malformatted attribute section in {}", x))
            }
        } else {
            SymbolBuilder::new(namespace.attach_namespace(x))
                .build_with_state(state)
                .map_err(|e| e.to_string())
        }?;

        // store the name in a map to prevent string allocation when the default namespace
        // gets added to the symbol
        name_map.insert(x.into(), s);
        Ok(s)
    }

    /// Parse the token into the atom `out`.
    fn to_atom_with_output_no_norm<T>(
        &self,
        namespace: &DefaultNamespace<T>,
        name_map: &mut HashMap<SmartString<LazyCompact>, Symbol>,
        state: &mut State,
        workspace: &Workspace,
        out: &mut Atom,
    ) -> Result<(), String> {
        match self {
            Token::SpecialNumber(c) => match c {
                '¿' => {
                    out.to_num(Coefficient::Indeterminate);
                }
                '⧞' => {
                    out.to_num(Coefficient::Infinity(None));
                }
                '∞' => {
                    out.to_num(Coefficient::Infinity(Some(Rational::one().into())));
                }
                _ => unreachable!(),
            },
            Token::Number(n, is_imag) => match n.parse::<Integer>() {
                Ok(x) => {
                    if *is_imag {
                        out.to_num(Complex::new(Rational::zero(), x.into()).into());
                    } else {
                        out.to_num(x.into());
                    }
                }
                Err(_) => match ErrorPropagatingFloat::parse(n, None) {
                    Ok(f) => {
                        let zero = ErrorPropagatingFloat::exact_zero(f.get_num().prec());
                        // derive precision from string length, should be overestimate
                        if *is_imag {
                            out.to_num(Complex::new(zero, f).into());
                        } else {
                            out.to_num(Coefficient::Float(Complex::new(f, zero)));
                        }
                    }
                    Err(e) => Err(format!("Error parsing number: {e}"))?,
                },
            },
            Token::ID(x) => {
                out.to_var(Self::parse_symbol(x, namespace, name_map, state)?);
            }
            Token::Op(_, _, op, args) => match op {
                Operator::Mul => {
                    let mul = out.to_mul();

                    let mut atom = workspace.new_atom();
                    for a in args {
                        a.to_atom_with_output_no_norm(
                            namespace, name_map, state, workspace, &mut atom,
                        )?;
                        mul.extend(atom.as_view());
                    }
                }
                Operator::Add => {
                    let add = out.to_add();

                    let mut atom = workspace.new_atom();
                    for a in args {
                        a.to_atom_with_output_no_norm(
                            namespace, name_map, state, workspace, &mut atom,
                        )?;
                        add.extend(atom.as_view());
                    }
                }
                Operator::Pow => {
                    // pow is right associative
                    args.last()
                        .unwrap()
                        .to_atom_with_output_no_norm(namespace, name_map, state, workspace, out)?;
                    for a in args.iter().rev().skip(1) {
                        let mut cur_base = workspace.new_atom();
                        a.to_atom_with_output_no_norm(
                            namespace,
                            name_map,
                            state,
                            workspace,
                            &mut cur_base,
                        )?;

                        let mut pow_h = workspace.new_atom();
                        pow_h.to_pow(cur_base.as_view(), out.as_view());
                        out.set_from_view(&pow_h.as_view());
                    }
                }
                Operator::Argument => return Err("Unexpected argument operator".into()),
                Operator::Neg => {
                    debug_assert!(args.len() == 1);

                    let mut base = workspace.new_atom();
                    args[0].to_atom_with_output_no_norm(
                        namespace, name_map, state, workspace, &mut base,
                    )?;

                    let num = workspace.new_num(-1);

                    let mul = out.to_mul();
                    mul.extend(base.as_view());
                    mul.extend(num.as_view());
                }
                Operator::Inv => {
                    debug_assert!(args.len() == 1);

                    let mut base = workspace.new_atom();
                    args[0].to_atom_with_output_no_norm(
                        namespace, name_map, state, workspace, &mut base,
                    )?;

                    let num = workspace.new_num(-1);

                    out.to_pow(base.as_view(), num.as_view());
                }
            },
            Token::Fn(_, _, args) => {
                let name = match &args[0] {
                    Token::ID(s) => s,
                    _ => unreachable!(),
                };

                let fun = out.to_fun(Self::parse_symbol(name, namespace, name_map, state)?);
                let mut atom = workspace.new_atom();
                for a in args.iter().skip(1) {
                    a.to_atom_with_output_no_norm(
                        namespace, name_map, state, workspace, &mut atom,
                    )?;
                    fun.add_arg(atom.as_view());
                }
            }
            Token::RationalPolynomial(_) => Err(format!(
                "Optimized rational polynomial input cannot be parsed yet as atom: {self}"
            ))?,
            Token::ParsedMul(a) => {
                out.set_from_view(&a.as_view());
            }
            x => return Err(format!("Unexpected token {x}")),
        }

        Ok(())
    }

    /// Parse the token into the atom `out` with pre-defined variables
    pub fn to_atom_with_output_and_var_map(
        &self,
        workspace: &Workspace,
        var_map: &Arc<Vec<PolyVariable>>,
        var_name_map: &[SmartString<LazyCompact>],
        out: &mut Atom,
    ) -> Result<(), String> {
        match self {
            Token::Number(n, is_imag) => match n.parse::<Integer>() {
                Ok(x) => {
                    if *is_imag {
                        out.to_num(Complex::new(Rational::zero(), x.into()).into());
                    } else {
                        out.to_num(x.into());
                    }
                }
                Err(_) => match Float::parse(n, None) {
                    Ok(f) => {
                        // derive precision from string length, should be overestimate
                        if *is_imag {
                            out.to_num(Complex::new(f.zero(), f).into());
                        } else {
                            out.to_num(Coefficient::Float(ErrorPropagatingFloat::from(f).into()));
                        }
                    }
                    Err(e) => Err(format!("Error parsing number: {e}"))?,
                },
            },
            Token::ID(name) => {
                let index = var_name_map
                    .iter()
                    .position(|x| x == name)
                    .ok_or_else(|| format!("Undefined variable {name}"))?;
                if let PolyVariable::Symbol(id) = var_map[index] {
                    out.to_var(id);
                } else {
                    Err(format!("Undefined variable {name}"))?;
                }
            }
            Token::Op(_, _, op, args) => match op {
                Operator::Mul => {
                    let mut mul_h = workspace.new_atom();
                    let mul = mul_h.to_mul();

                    let mut atom = workspace.new_atom();
                    for a in args {
                        a.to_atom_with_output_and_var_map(
                            workspace,
                            var_map,
                            var_name_map,
                            &mut atom,
                        )?;
                        mul.extend(atom.as_view());
                    }

                    mul_h.as_view().normalize(workspace, out);
                }
                Operator::Add => {
                    let mut add_h = workspace.new_atom();
                    let add = add_h.to_add();

                    let mut atom = workspace.new_atom();
                    for a in args {
                        a.to_atom_with_output_and_var_map(
                            workspace,
                            var_map,
                            var_name_map,
                            &mut atom,
                        )?;
                        add.extend(atom.as_view());
                    }

                    add_h.as_view().normalize(workspace, out);
                }
                Operator::Pow => {
                    let mut base = workspace.new_atom();
                    args[0].to_atom_with_output_and_var_map(
                        workspace,
                        var_map,
                        var_name_map,
                        &mut base,
                    )?;

                    let mut exp = workspace.new_atom();
                    args[1].to_atom_with_output_and_var_map(
                        workspace,
                        var_map,
                        var_name_map,
                        &mut exp,
                    )?;

                    let mut pow_h = workspace.new_atom();
                    pow_h.to_pow(base.as_view(), exp.as_view());
                    pow_h.as_view().normalize(workspace, out);
                }
                Operator::Argument => return Err("Unexpected argument operator".into()),
                Operator::Neg => {
                    debug_assert!(args.len() == 1);

                    let mut base = workspace.new_atom();
                    args[0].to_atom_with_output_and_var_map(
                        workspace,
                        var_map,
                        var_name_map,
                        &mut base,
                    )?;

                    let num = workspace.new_num(-1);

                    let mut mul_h = workspace.new_atom();
                    let mul = mul_h.to_mul();
                    mul.extend(base.as_view());
                    mul.extend(num.as_view());
                    mul_h.as_view().normalize(workspace, out);
                }
                Operator::Inv => {
                    debug_assert!(args.len() == 1);

                    let mut base = workspace.new_atom();
                    args[0].to_atom_with_output_and_var_map(
                        workspace,
                        var_map,
                        var_name_map,
                        &mut base,
                    )?;

                    let num = workspace.new_num(-1);

                    let mut pow_h = workspace.new_atom();
                    pow_h.to_pow(base.as_view(), num.as_view());
                    pow_h.as_view().normalize(workspace, out);
                }
            },
            Token::Fn(_, _, args) => {
                let name = match &args[0] {
                    Token::ID(s) => s,
                    _ => unreachable!(),
                };

                let index = var_name_map
                    .iter()
                    .position(|x| x == name)
                    .ok_or_else(|| format!("Undefined variable {name}"))?;
                if let PolyVariable::Symbol(id) = var_map[index] {
                    let mut fun_h = workspace.new_atom();
                    let fun = fun_h.to_fun(id);
                    let mut atom = workspace.new_atom();
                    for a in args.iter().skip(1) {
                        a.to_atom_with_output_and_var_map(
                            workspace,
                            var_map,
                            var_name_map,
                            &mut atom,
                        )?;
                        fun.add_arg(atom.as_view());
                    }

                    fun_h.as_view().normalize(workspace, out);
                } else {
                    Err(format!("Undefined variable {name}"))?;
                }
            }
            x => return Err(format!("Unexpected token {x}")),
        }

        Ok(())
    }

    /// Map Mathematica `FullForm` symbols to internal functions and operators.
    /// The first argument is the function name.
    fn map_mathematica_full_form_symbols(args: &mut Vec<Token>) -> Result<Option<Self>, String> {
        if let Some(Token::ID(name)) = args.first() {
            match name.as_str() {
                "Power" => {
                    let mut args = std::mem::take(args);
                    args.remove(0);
                    Ok(Some(Token::Op(false, false, Operator::Pow, args)))
                }
                "Times" => {
                    let mut args = std::mem::take(args);
                    args.remove(0);
                    Ok(Some(Token::Op(false, false, Operator::Mul, args)))
                }
                "Plus" => {
                    let mut args = std::mem::take(args);
                    args.remove(0);
                    Ok(Some(Token::Op(false, false, Operator::Add, args)))
                }
                "Rational" => {
                    if args.len() != 3 {
                        return Err("Rational takes exactly two arguments".into());
                    }
                    let num = std::mem::replace(&mut args[1], Token::EOF);
                    let den = std::mem::replace(&mut args[2], Token::EOF);
                    Ok(Some(Token::Op(
                        false,
                        false,
                        Operator::Mul,
                        vec![num, Token::Op(false, false, Operator::Inv, vec![den])],
                    )))
                }
                "Complex" => {
                    if args.len() != 3 {
                        return Err("Complex takes exactly two arguments".into());
                    }
                    let real = std::mem::replace(&mut args[1], Token::EOF);
                    let imag = std::mem::replace(&mut args[2], Token::EOF);

                    Ok(Some(Token::Op(
                        false,
                        false,
                        Operator::Add,
                        vec![
                            real,
                            Token::Op(
                                false,
                                false,
                                Operator::Mul,
                                vec![imag, Token::Number("1".into(), true)],
                            ),
                        ],
                    )))
                }
                "DirectedInfinity" => {
                    if args.len() == 1 {
                        Ok(Some(Token::SpecialNumber('⧞')))
                    } else if args.len() == 2 {
                        let arg = std::mem::replace(&mut args[1], Token::EOF);
                        Ok(Some(Token::Op(
                            false,
                            false,
                            Operator::Mul,
                            vec![arg, Token::SpecialNumber('∞')],
                        )))
                    } else {
                        Err("DirectedInfinity takes at most one argument".into())
                    }
                }
                "Derivative" | "der" => {
                    // will be called for every curried argument
                    args[0] = Token::ID("der".into());

                    let sep = Token::ID(Symbol::SEP_STR.into());
                    args.retain(|x| *x != sep);

                    let mut function_index = 1;
                    for x in &args[1..] {
                        if let Token::Number(_, _) = x {
                            function_index += 1;
                        }
                    }

                    if function_index == args.len() {
                        return Ok(None);
                    }

                    // merge curried arguments into one function
                    // Derivative[2][f][x,y] => der(2,f(x,y))
                    if let Token::ID(_) = &args[function_index] {
                        // upgrade argument to function
                        let fn_arg = args.split_off(function_index);
                        args.push(Token::Fn(false, false, fn_arg));
                    } else {
                        let new_fn_arg = args.split_off(function_index + 1);

                        if let Token::Fn(false, false, fn_arg) = &mut args[function_index] {
                            fn_arg.extend(new_fn_arg);
                        } else {
                            return Err("Derivative function argument is not a function".into());
                        }
                    }

                    Ok(None)
                }
                "Pattern" => {
                    if args.len() == 3
                        && let Some(Token::ID(_)) = args.get(1)
                        && let Some(Token::Fn(_, _, wildcard_type)) = args.get(2)
                        && wildcard_type.len() == 1
                        && let Some(Token::ID(wt)) = wildcard_type.first()
                    {
                        let level;
                        match wt.as_str() {
                            "Blank" => {
                                level = 1;
                            }
                            "BlankSequence" => {
                                level = 2;
                            }
                            "BlankNullSequence" => {
                                level = 3;
                            }
                            _ => {
                                return Ok(None);
                            }
                        }

                        let mut args = std::mem::take(args);
                        let mut t = args.swap_remove(1);
                        if let Token::ID(symbol) = &mut t {
                            for _ in 0..level {
                                symbol.push('_');
                            }
                        } else {
                            unreachable!();
                        }

                        Ok(Some(t))
                    } else {
                        Ok(None)
                    }
                }
                "SetAccuracy" => {
                    if args.len() == 3
                        && let Some(Token::Number(s, false)) = args.get(1)
                        && s == "0"
                        && let Some(Token::Number(prec, false)) = args.get(2)
                    {
                        // precision on 0 is treated as accuracy
                        Ok(Some(Token::Number(SmartString::from("0`") + prec, false)))
                    } else {
                        Ok(None)
                    }
                }
                _ => Ok(None),
            }
        } else {
            Ok(None)
        }
    }

    /// Parse a Symbolica expression, generating a token tree. For most users,
    /// it is recommended to use [crate::parse] instead, which returns an atom.
    pub fn parse(input: &str, settings: ParseSettings) -> Result<Token, String> {
        Token::parse_with_atom_info::<&'static str>(input, settings, None)
    }

    /// Parse a Symbolica expression, generating a token tree that may contain
    /// parsed atom. For most users, it is recommended to use [crate::parse] instead, which returns an atom.
    pub fn parse_with_atom_info<T>(
        input: &str,
        settings: ParseSettings,
        mut atom_info: Option<(
            &DefaultNamespace<T>,
            &mut HashMap<SmartString<LazyCompact>, Symbol>,
            &Workspace,
        )>,
    ) -> Result<Token, String> {
        LicenseManager::check();

        // Remove ANSI color codes from the input string.
        let mut input = Cow::Borrowed(input);
        if input.contains('\x1b') {
            let mut s = String::with_capacity(input.len());
            let mut in_ansi = false;
            for c in input.chars() {
                if c == '\x1b' {
                    in_ansi = true;
                } else if in_ansi && c == 'm' {
                    in_ansi = false;
                } else if !in_ansi {
                    s.push(c);
                }
            }

            input = Cow::Owned(s);
        }

        let whitespaces = if settings.mode.is_mathematica() {
            Token::WHITESPACE_MATHEMATICA.as_slice()
        } else {
            Token::WHITESPACE.as_slice()
        };

        let mut stack: Vec<_> = Vec::with_capacity(20);
        stack.push(Token::Start);
        let mut state = ParseState::Any;

        let mut char_iter = input.chars();
        let mut c = char_iter.next().unwrap_or('\0'); // add EOF as a token
        let mut extra_ops: SmallVec<[char; 6]> = SmallVec::new();

        let mut id_buffer = String::with_capacity(30);

        let mut line_counter = 1;
        let mut column_counter = 1;

        let mut inside_mathematica_full_form = false;
        loop {
            match state {
                ParseState::Identifier => {
                    if inside_mathematica_full_form && c == '[' {
                        // convert \[Alpha] to ‖Alpha‖
                        if !id_buffer.ends_with(Symbol::SEP_STR) {
                            Err(format!(
                                "Unexpected '{c:?}' in input at line {line_counter} and column {column_counter}"
                            ))?;
                        }
                    } else if inside_mathematica_full_form && c == ']' {
                        inside_mathematica_full_form = false;
                        id_buffer.push_str(Symbol::SEP_STR);
                    } else if Token::OPS.contains(&c) || whitespaces.contains(&c) {
                        state = ParseState::Any;

                        if settings.mode.is_mathematica() {
                            match id_buffer.as_str() {
                                "Pi" => {
                                    stack.push(Token::ID(Symbol::PI_STR.into()));
                                }
                                "E" => {
                                    stack.push(Token::ID(Symbol::E_STR.into()));
                                }
                                "I" => {
                                    stack.push(Token::Number("1".into(), true));
                                }
                                "Sqrt" => {
                                    stack.push(Token::ID("sqrt".into()));
                                }
                                "Cos" => {
                                    stack.push(Token::ID("cos".into()));
                                }
                                "Sin" => {
                                    stack.push(Token::ID("sin".into()));
                                }
                                "Log" => {
                                    stack.push(Token::ID("log".into()));
                                }
                                "Exp" => {
                                    stack.push(Token::ID("exp".into()));
                                }
                                "Conjugate" => {
                                    stack.push(Token::ID("conj".into()));
                                }
                                "Indeterminate" => {
                                    stack.push(Token::SpecialNumber('¿'));
                                }
                                _ => {
                                    stack.push(Token::ID(id_buffer.as_str().into()));
                                }
                            }
                        } else {
                            stack.push(Token::ID(id_buffer.as_str().into()));
                        }

                        id_buffer.clear();
                    } else if c == '{' && settings.mode == ParseMode::Symbolica {
                        while c != '}' && c != '\0' {
                            id_buffer.push(c);
                            c = char_iter.next().unwrap_or('\0');
                            column_counter += 1;
                        }

                        if c == '\0' {
                            Err(format!(
                                "Missing }} of bracket started at line {line_counter} and column {column_counter}"
                            ))?;
                        }

                        if Token::OPS.contains(&c) || Token::WHITESPACE.contains(&c) {
                            state = ParseState::Any;

                            stack.push(Token::ID(id_buffer.as_str().into()));
                            id_buffer.clear();
                        }

                        id_buffer.push(c);
                    } else if settings.mode.is_mathematica() && c == '`' {
                        id_buffer.push_str("::");
                    } else if settings.mode.is_mathematica() && c == '\\' {
                        inside_mathematica_full_form = true;
                        id_buffer.push_str(Symbol::SEP_STR);
                    } else if !Token::FORBIDDEN.contains(&c) {
                        id_buffer.push(c);
                    } else {
                        // check for some symbols that could be the result of copy-paste errors
                        // when importing from other languages
                        Err(format!(
                            "Unexpected '{c}' in input at line {line_counter} and column {column_counter}"
                        ))?;
                    }
                }
                ParseState::Number => {
                    let mut last_digit_is_exp = false;
                    loop {
                        if c.is_ascii_digit() {
                            id_buffer.push(c);
                            c = char_iter.next().unwrap_or('\0');
                            column_counter += 1;
                            last_digit_is_exp = false;
                            continue;
                        }

                        if last_digit_is_exp && c != '-' && c != '+' {
                            // input cannot be a floating point number
                            // add a multiplication operator, e.g 2.2ex => 2.2*ex

                            let e = id_buffer.pop().unwrap();
                            state = ParseState::Any;
                            stack.push(Token::Number(id_buffer.as_str().into(), false));
                            id_buffer.clear();

                            extra_ops.push(e);
                            extra_ops.push(c);
                            c = '*';

                            break;
                        }

                        if c == 'i' || c == '𝑖' {
                            let old_c = c;
                            // complex number has trailing i and must be followed by whitespace or an operator
                            c = char_iter.next().unwrap_or('\0');
                            column_counter += 1;

                            let is_imag = Token::WHITESPACE.contains(&c) || Token::OPS.contains(&c);

                            state = ParseState::Any;
                            stack.push(Token::Number(id_buffer.as_str().into(), is_imag));
                            id_buffer.clear();

                            if !is_imag {
                                extra_ops.push(old_c);
                                extra_ops.push(c);
                                c = '*';
                            }

                            last_digit_is_exp = true; // prevent adding the number again
                            break;
                        }

                        if c == '\0' {
                            last_digit_is_exp = false;
                            break;
                        }

                        if settings.mode.is_mathematica() && c == '*' {
                            c = char_iter.next().unwrap_or('\0');
                            column_counter += 1;

                            if c == '^' {
                                c = 'e';
                            } else {
                                extra_ops.push(c);
                                c = '*';
                                break;
                            }
                        }

                        let digit_is_exp = c == 'e' || c == 'E';

                        if c != '_' && c != ' ' {
                            if !digit_is_exp && !last_digit_is_exp && c != '.' && c != '`' {
                                break;
                            }

                            id_buffer.push(c);
                        }

                        last_digit_is_exp = digit_is_exp;

                        c = char_iter.next().unwrap_or('\0');
                        column_counter += 1;
                    }

                    if !last_digit_is_exp {
                        state = ParseState::Any;
                        stack.push(Token::Number(id_buffer.as_str().into(), false));
                        id_buffer.clear();
                    }
                }
                ParseState::RationalPolynomial => {
                    let start = char_iter.clone();
                    let mut pos = 0;

                    let mut s = SmartString::new();
                    s.push(c);

                    while c != ']' && c != '\0' {
                        pos += 1;
                        c = char_iter.next().unwrap_or('\0');
                        column_counter += 1;
                    }

                    if c == '\0' {
                        Err(format!(
                            "Missing ] of bracket started at line {line_counter} and column {column_counter}"
                        ))?;
                    }

                    s.push_str(&start.as_str()[..pos - 1]);
                    stack.push(Token::RationalPolynomial(s));

                    state = ParseState::Any;

                    column_counter += pos + 1;
                    c = char_iter.next().unwrap_or('\0');
                    column_counter += 1;
                }
                ParseState::Any => {}
            }

            if state == ParseState::Any {
                if whitespaces.contains(&c) {
                    if c == '\n' {
                        column_counter = 1;
                        line_counter += 1;
                    } else {
                        column_counter += 1;
                    }

                    c = char_iter.next().unwrap_or('\0');
                    continue;
                }

                match c {
                    '+' => {
                        if matches!(
                            unsafe { stack.last().unwrap_unchecked() },
                            Token::Start
                                | Token::OpenParenthesis
                                | Token::Fn(true, _, _)
                                | Token::Op(_, true, _, _)
                        ) {
                            // unary + operator, can be ignored as plus is the default
                        } else {
                            stack.push(Token::Op(true, true, Operator::Add, vec![]))
                        }
                    }
                    '^' => stack.push(Token::Op(true, true, Operator::Pow, vec![])),
                    '*' => stack.push(Token::Op(true, true, Operator::Mul, vec![])),
                    '-' => {
                        if matches!(
                            unsafe { stack.last().unwrap_unchecked() },
                            Token::Start
                                | Token::OpenParenthesis
                                | Token::Fn(true, _, _)
                                | Token::Op(_, true, _, _)
                        ) {
                            // unary minus only requires an argument to the right
                            stack.push(Token::Op(false, true, Operator::Neg, vec![]));
                        } else {
                            stack.push(Token::Op(true, true, Operator::Add, vec![]));
                            extra_ops.push('-'); // push a unary minus
                        }
                    }
                    '(' => {
                        // check if the opening bracket belongs to a function
                        if settings.mode == ParseMode::Symbolica
                            && let Some(Token::ID(_)) = stack.last()
                        {
                            let name = unsafe { stack.pop().unwrap_unchecked() };
                            if let Token::ID(_) = name {
                                stack.push(Token::Fn(true, false, vec![name])); // serves as open paren
                            }
                        } else if unsafe { stack.last().unwrap_unchecked() }.is_normal() {
                            // insert multiplication: x(...) -> x*(...)
                            stack.push(Token::Op(true, true, Operator::Mul, vec![]));
                            extra_ops.push(c);
                        } else {
                            stack.push(Token::OpenParenthesis)
                        }
                    }
                    ')' => stack.push(Token::CloseParenthesis),
                    '/' => {
                        if matches!(
                            stack.last().unwrap(),
                            Token::Start
                                | Token::OpenParenthesis
                                | Token::Fn(true, _, _)
                                | Token::Op(_, true, _, _)
                        ) {
                            // unary inv only requires an argument to the right
                            stack.push(Token::Op(false, true, Operator::Inv, vec![]));
                        } else {
                            stack.push(Token::Op(true, true, Operator::Mul, vec![]));
                            extra_ops.push('/'); // push a (unary) inverse
                        }
                    }
                    ',' => stack.push(Token::Op(true, true, Operator::Argument, vec![])),
                    '\0' => stack.push(Token::EOF),
                    '[' => {
                        if unsafe { stack.last().unwrap_unchecked() }.is_normal() {
                            if let Token::ID(_) = unsafe { stack.last().unwrap_unchecked() } {
                                let name = unsafe { stack.pop().unwrap_unchecked() };
                                if let Token::ID(_) = name {
                                    stack.push(Token::Fn(true, true, vec![name]));
                                }
                            } else if settings.mode == ParseMode::Symbolica {
                                // insert multiplication: f(x)[3,4] -> f(x)*[3,4]
                                stack.push(Token::Op(true, true, Operator::Mul, vec![]));
                                extra_ops.push(c);
                            } else if settings.mode == ParseMode::Mathematica
                                && let Token::Fn(need_more, _, args) =
                                    unsafe { stack.last_mut().unwrap_unchecked() }
                            {
                                // parse curried functions, e.g. f[x][y]
                                *need_more = true;
                                args.push(Token::ID(Symbol::SEP_STR.into()));
                            } else {
                                Err(format!(
                                    "Unexpected '[' in input at line {line_counter} and column {column_counter}"
                                ))?;
                            }
                        } else if settings.mode == ParseMode::Symbolica {
                            state = ParseState::RationalPolynomial;
                        } else {
                            Err(format!(
                                "Unexpected '[' in input at line {line_counter} and column {column_counter}"
                            ))?;
                        }
                    }
                    ']' => stack.push(Token::CloseBracket),
                    '"' => {
                        if !settings.mode.is_mathematica() {
                            Err(format!(
                                "Unexpected '\"' in input at line {line_counter} and column {column_counter}"
                            ))?;
                        }

                        // parse a string as a literal id in the current namespace
                        id_buffer.clear();
                        let mut escape = true;
                        while (escape || c != '"') && c != '\0' {
                            id_buffer.push(c);
                            escape = c == '\\';
                            c = char_iter.next().unwrap_or('\0');
                            column_counter += 1;
                        }

                        if c == '\0' {
                            Err(format!(
                                "Unexpected 'EOF' in input at line {line_counter} and column {column_counter}"
                            ))?;
                        }

                        id_buffer.push(c);
                        stack.push(Token::ID(id_buffer.as_str().into()));
                        id_buffer.clear();
                    }
                    _ => {
                        if unsafe { stack.last().unwrap_unchecked() }.is_normal()
                            && (!c.is_ascii_digit()
                                || !matches!(
                                    unsafe { stack.last().unwrap_unchecked() },
                                    Token::Number(_, _)
                                ))
                        {
                            // insert implicit multiplication: x y -> x*y
                            // do not allow implicit multiplication between two numbers as the risk
                            // of a typo is too high
                            stack.push(Token::Op(true, true, Operator::Mul, vec![]));
                            extra_ops.push(c);
                        } else if c.is_ascii_digit() {
                            state = ParseState::Number;
                            id_buffer.push(c);
                        } else if c == '¿' || c == '⧞' || c == '∞' {
                            stack.push(Token::SpecialNumber(c));
                        } else if !Token::FORBIDDEN.contains(&c) {
                            state = ParseState::Identifier;

                            if settings.mode.is_mathematica() && c == '\\' {
                                inside_mathematica_full_form = true;
                                id_buffer.push_str(Symbol::SEP_STR);
                            } else {
                                id_buffer.push(c);
                            }
                        } else {
                            Err(format!(
                                "Unexpected '{c}' in input at line {line_counter} and column {column_counter}"
                            ))?;
                        }
                    }
                }
            }

            // match on triplets of type operator identifier operator
            while state == ParseState::Any && stack.len() > 2 {
                if !unsafe { stack.get_unchecked(stack.len() - 2) }.is_normal() {
                    // check for the empty function

                    // check if the left operator needs a right-hand side and the new operator still needs a left-hand side
                    match unsafe { stack.get_unchecked(stack.len() - 1) } {
                        Token::Op(true, _, op, _) => {
                            Err(format!(
                                "Error at line {line_counter} and position {column_counter}: operator '{op}' is missing left-hand side",
                            ))?;
                        }

                        x @ Token::CloseParenthesis | x @ Token::CloseBracket => {
                            let c = x.clone();
                            let pos = stack.len() - 2;
                            // check if we have an empty function
                            if let Token::Fn(f, bracket, args) =
                                unsafe { stack.get_unchecked_mut(pos) }
                            {
                                if c == Token::CloseParenthesis && !*bracket
                                    || c == Token::CloseBracket && *bracket
                                {
                                    *f = false;

                                    if settings.mode.is_mathematica()
                                        && let Some(rewrite) =
                                            Token::map_mathematica_full_form_symbols(args)?
                                    {
                                        *unsafe { stack.get_unchecked_mut(pos) } = rewrite;
                                    }

                                    stack.pop();
                                } else {
                                    Err(format!(
                                        "Error at line {} and position {}: unexpected '{}'",
                                        line_counter,
                                        column_counter,
                                        if c == Token::CloseParenthesis {
                                            ")"
                                        } else {
                                            "]"
                                        }
                                    ))?;
                                }
                            } else {
                                Err(format!(
                                    "Error at line {} and position {}: unexpected '{}'",
                                    line_counter,
                                    column_counter,
                                    if c == Token::CloseParenthesis {
                                        ")"
                                    } else {
                                        "]"
                                    }
                                ))?;
                            }
                        }
                        _ => {}
                    }

                    // no simplification, get new token
                    break;
                }

                let mut last = unsafe { stack.pop().unwrap_unchecked() };
                let middle = unsafe { stack.pop().unwrap_unchecked() };
                let mut first = unsafe { stack.last_mut().unwrap_unchecked() };

                match first.get_precedence().cmp(&last.get_precedence()) {
                    std::cmp::Ordering::Greater => {
                        first
                            .add_right(middle, settings.distribute_neg)
                            .map_err(|e| {
                                format!(
                                    "Error at line {line_counter} and position {column_counter}: "
                                ) + e.as_str()
                            })?;
                        stack.push(last);
                    }
                    std::cmp::Ordering::Less => {
                        last.add_left(middle).map_err(|e| {
                            format!("Error at line {line_counter} and position {column_counter}: ")
                                + e.as_str()
                        })?;

                        stack.push(last);
                    }
                    std::cmp::Ordering::Equal => {
                        // same degree, special merges!
                        match (&mut first, middle, last) {
                            (Token::Start, mid, Token::EOF) => {
                                *first = mid;
                            }
                            (
                                Token::Fn(mr, bracket, args),
                                mid,
                                x @ Token::CloseParenthesis | x @ Token::CloseBracket,
                            ) => {
                                debug_assert!(*mr);
                                *mr = false;

                                if x == Token::CloseParenthesis && *bracket
                                    || x == Token::CloseBracket && !*bracket
                                {
                                    Err(format!(
                                        "Error at line {} and position {}: unexpected '{}'",
                                        line_counter,
                                        column_counter,
                                        if x == Token::CloseParenthesis {
                                            ")"
                                        } else {
                                            "]"
                                        }
                                    ))?;
                                }

                                if let Token::Op(_, _, Operator::Argument, arg2) = mid {
                                    args.extend(arg2);
                                } else {
                                    args.push(mid);
                                }

                                if settings.mode.is_mathematica()
                                    && let Some(rewrite) =
                                        Token::map_mathematica_full_form_symbols(args)?
                                {
                                    *first = rewrite;
                                }
                            }
                            (Token::OpenParenthesis, mid, Token::CloseParenthesis) => {
                                *first = mid;
                            }
                            (
                                Token::Op(ml1, mr1, o1, m),
                                mid,
                                Token::Op(ml2, mr2, mut o2, mut mm),
                            ) => {
                                debug_assert!(!*ml1);
                                debug_assert!(*mr1 && ml2);
                                // same precedence, so left associate

                                // flatten if middle identifier is also a binary operator of the same type that
                                // is also right associative
                                if let Token::Op(_, _, o_mid, mut m_mid) = mid {
                                    if o_mid == *o1 && o_mid.right_associative() {
                                        m.append(&mut m_mid);
                                    } else {
                                        if settings.convert_mul_to_atom
                                            && o_mid == Operator::Mul
                                            && let Some((ns, name_map, ws)) = &mut atom_info
                                        {
                                            // convert a finished multiplication sandwiched between
                                            // two additions (a term) into an atom
                                            // this representation will be more memory efficient
                                            let t = Token::Op(false, false, o_mid, m_mid);

                                            let mut atom = ws.new_atom();

                                            {
                                                let mut state =
                                                    State::get_global_state().write().unwrap();
                                                // do not normalize to prevent potential deadlocks
                                                t.to_atom_with_output_no_norm(
                                                    ns, name_map, &mut state, ws, &mut atom,
                                                )?;
                                            }

                                            let mut norm = ws.new_atom();
                                            atom.as_view().normalize(ws, &mut norm);

                                            m.push(Token::ParsedMul(Box::new(norm)));
                                        } else {
                                            m.push(Token::Op(false, false, o_mid, m_mid));
                                        }
                                    }
                                } else {
                                    m.push(mid)
                                }

                                // may not be the same operator, in the case of * and /
                                if *o1 == o2 {
                                    m.append(&mut mm);
                                    *mr1 = mr2;
                                } else {
                                    // embed operator 1 in operator 2
                                    *mr1 = mr2;
                                    std::mem::swap(o1, &mut o2);
                                    std::mem::swap(m, &mut mm);
                                    m.insert(0, Token::Op(false, false, o2, mm));
                                }
                            }
                            _ => return Err("Cannot merge operator".to_string()),
                        }
                    }
                }
            }

            if c == '\0' {
                break;
            }

            // first drain the queue of extra operators
            if extra_ops.is_empty() {
                if c == '\n' {
                    column_counter = 1;
                    line_counter += 1;
                } else {
                    column_counter += 1;
                }

                c = char_iter.next().unwrap_or('\0');
            } else {
                c = extra_ops.remove(0);
            }
        }

        if stack.len() == 1 {
            Ok(stack.pop().unwrap())
        } else {
            match stack.get(stack.len() - 2) {
                Some(Token::Op(false, true, op, _)) => Err(format!(
                    "Unexpected end of input: missing right-hand side for operator '{op}'"
                )),
                Some(Token::OpenParenthesis) => {
                    Err("Unexpected end of input: open parenthesis is not closed".to_string())
                }

                Some(Token::Fn(true, _, args)) => Err(format!(
                    "Unexpected end of input: Missing closing parenthesis for function '{}'",
                    args[0]
                )),
                Some(Token::Start) => Err("Expression is empty".to_string()),
                _ => Err(format!("Unknown parsing error: {stack:?}")),
            }
        }
    }

    /// A special routine that can parse a polynomial written in expanded form,
    /// where the coefficient comes first.
    pub fn parse_polynomial<'a, R: Ring + ConvertToRing, E: PositiveExponent>(
        mut input: &'a [u8],
        var_map: &Arc<Vec<PolyVariable>>,
        var_name_map: &[SmartString<LazyCompact>],
        field: &R,
    ) -> (&'a [u8], MultivariatePolynomial<R, E>) {
        let mut exponents = vec![E::zero(); var_name_map.len()];
        let mut poly = MultivariatePolynomial::new(field, None, var_map.clone());

        let mut last_pos = input;
        let mut c = input.get_u8();

        let mut digit_buffer = vec![];
        loop {
            if c == b'(' || c == b')' || c == b'/' {
                break;
            }

            // read a term
            let mut coeff = field.one();
            for e in &mut exponents {
                *e = E::zero();
            }

            if c == b'+' {
                last_pos = input;
                c = input.get_u8();
            }

            // read number
            let num_start = last_pos;
            if c == b'-' {
                last_pos = input;
                c = input.get_u8();
            }

            let mut is_hex = false;
            let mask = if c == b'#' {
                is_hex = true;
                last_pos = input;
                c = input.get_u8();
                &HEX_DIGIT_MASK
            } else {
                &DIGIT_MASK
            };

            while mask[c as usize] && !input.is_empty() {
                last_pos = input;
                c = input.get_u8();
            }

            // construct number
            let mut len = unsafe { input.as_ptr().offset_from(num_start.as_ptr()) } as usize;
            let mut last_read_is_non_digit = false;
            if !mask[c as usize] && (len > 1 || c != b'-') {
                last_read_is_non_digit = true;
                len -= 1;
            }

            if len > 0 {
                coeff = 'read_coeff: {
                    if len == 1 && num_start[0] == b'-' {
                        break 'read_coeff field.neg(&field.one());
                    }

                    if !is_hex && len <= 40 {
                        let n = unsafe { std::str::from_utf8_unchecked(&num_start[..len]) };

                        if len <= 20
                            && let Ok(n) = n.parse::<i64>()
                        {
                            break 'read_coeff field.element_from_coefficient(n.into());
                        }

                        if let Ok(n) = n.parse::<i128>() {
                            break 'read_coeff field
                                .element_from_coefficient(Integer::Double(n).into());
                        }
                    }

                    let (is_negative, digits) = if num_start[0] == b'-' {
                        (true, &num_start[1..len])
                    } else {
                        (false, &num_start[..len])
                    };

                    digit_buffer.clear();

                    if is_hex {
                        digit_buffer.extend(
                            digits[1..]
                                .iter()
                                .map(|&x| HEX_TO_DIGIT[(x - b'0') as usize]),
                        );
                    } else {
                        digit_buffer.extend(digits.iter().map(|&x| x - b'0'));
                    }

                    let mut p = MultiPrecisionInteger::new();
                    unsafe {
                        p.assign_bytes_radix_unchecked(
                            &digit_buffer,
                            if is_hex { 16 } else { 10 },
                            is_negative,
                        )
                    };

                    field.element_from_coefficient(p.into())
                }
            }

            if input.is_empty() && !last_read_is_non_digit {
                poly.append_monomial(coeff, &exponents);
                break;
            }

            if c == b'-' {
                // done with the term
                poly.append_monomial(coeff, &exponents);
                continue;
            }

            // read var^pow
            loop {
                let before_star = last_pos;
                if c == b'*' {
                    if input.is_empty() {
                        break;
                    }

                    last_pos = input;
                    c = input.get_u8();
                }
                if !c.is_ascii_alphabetic() {
                    if before_star[0] == b'*' {
                        last_pos = before_star; // bring back the *
                    }
                    break;
                }

                let var_start = last_pos;

                // read var
                while c.is_ascii_alphanumeric() {
                    if input.is_empty() {
                        break;
                    }

                    last_pos = input;
                    c = input.get_u8();
                }

                let mut len = unsafe { input.as_ptr().offset_from(var_start.as_ptr()) } as usize;
                if !c.is_ascii_alphanumeric() {
                    len -= 1;
                }

                let name = unsafe { std::str::from_utf8_unchecked(&var_start[..len]) };
                let index = var_name_map
                    .iter()
                    .position(|x| x == name)
                    .expect("Undefined variable");

                // read pow
                if c == b'^' {
                    let pow_start = input;

                    // read pow
                    loop {
                        last_pos = input;
                        c = input.get_u8();

                        if !c.is_ascii_digit() || input.is_empty() {
                            break;
                        }
                    }

                    let mut len =
                        unsafe { input.as_ptr().offset_from(pow_start.as_ptr()) } as usize;
                    if !c.is_ascii_digit() {
                        len -= 1;
                    }
                    let n = unsafe { std::str::from_utf8_unchecked(&pow_start[..len]) };
                    exponents[index] = E::from_u32(n.parse::<u32>().unwrap());
                } else {
                    exponents[index] = E::one();
                }

                if input.is_empty() {
                    break;
                }
            }

            // construct a new term
            poly.append_monomial(coeff, &exponents);

            if input.is_empty() {
                break;
            }
        }
        if input.is_empty() {
            (input, poly)
        } else {
            (last_pos, poly)
        }
    }
}

#[cfg(test)]
mod test {
    use std::sync::Arc;

    use crate::{
        atom::{AtomCore, SymbolAttribute},
        domains::{SelfRing, integer::Z},
        parse,
        parser::Token,
        printer::{PrintOptions, PrintState},
        symbol,
    };

    #[test]
    fn mathematica() {
        let a = parse!("cos(x+2i + 3)+sqrt(conj(x)) + test::y + exp(test::test2::x) + log(α)");
        let s = a.format_string(&PrintOptions::mathematica(), PrintState::default());
        let b = parse!(s, Mathematica);
        assert_eq!(a, b);

        let input = "a\\[Alpha]b+\\[Beta]";
        let b = parse!(input, Mathematica);
        assert_eq!(b, parse!("a‖Alpha‖b+‖Beta‖"));
        let s = b.format_string(&PrintOptions::mathematica(), PrintState::default());
        assert_eq!(s, input);

        let input =
            "Pattern[f, Blank[]][Pattern[x, BlankSequence[]], Pattern[v, BlankNullSequence[]]]";
        let b = parse!(input, Mathematica);
        assert_eq!(b, parse!("f_(x__, v___)"));

        let a = parse!("2*x+2*^5y + SetAccuracy[0, 10.]", Mathematica);
        assert_eq!(a, parse!("2*x+2e5*y+0.`10"));

        let a = parse!("\"a b $ * \\\" / // c\"", Mathematica);
        assert!(a.get_symbol().is_some());
    }

    #[test]
    fn attributes() {
        let input = parse!("symbolica::b::{linear,symmetric,test::a}::dot_ls");
        let s = input.get_symbol().unwrap();
        assert_eq!(s.get_name(), "symbolica::b::dot_ls");
        assert_eq!(
            s.get_attributes(),
            vec![SymbolAttribute::Symmetric, SymbolAttribute::Linear,]
        );
        assert_eq!(s.get_tags(), vec!["test::a"]);

        let input = parse!("symbolica::{}::f1()").get_symbol().unwrap();
        assert_eq!(input.get_name(), "symbolica::f1");
    }

    #[test]
    fn infinity() {
        let input = parse!("∞ + 5 - ¿ + 3*⧞ - ∞");
        assert_eq!(
            format!("{}", input.printer(PrintOptions::file_no_namespace())),
            "¿"
        );
    }

    #[test]
    fn pow() {
        let input = parse!("v1^v2^v3^3");
        assert_eq!(
            format!("{}", input.printer(PrintOptions::file_no_namespace())),
            "v1^v2^v3^3"
        );

        let input = parse!("(v1^v2)^v3");
        assert_eq!(
            format!("{}", input.printer(PrintOptions::file_no_namespace())),
            "(v1^v2)^v3"
        );
    }

    #[test]
    fn unary() {
        let input = parse!("-x^z");
        assert_eq!(
            format!("{}", input.printer(PrintOptions::file_no_namespace())),
            "-x^z"
        );

        let input = parse!("(-x)^z");
        assert_eq!(
            format!("{}", input.printer(PrintOptions::file_no_namespace())),
            "(-x)^z"
        );
    }

    #[test]
    fn liberal() {
        let input = parse!(
            "89233_21837281 x   
            ^2 / y + 5 + 5x"
        );
        let res = parse!("8923321837281*x^2*y^-1+5+5x");
        assert_eq!(input, res);
    }

    #[test]
    fn float() {
        let input = parse!("1.2`20x+1e-5`20+1e+5 * 1.1234e23 +2exp(5)");

        let r = format!("{}", input.printer(PrintOptions::file_no_namespace()));
        assert_eq!(
            r,
            "1.2000000000000000000`20.00*x+2*exp(5)+1.123400000000000e28`15.65"
        );
    }

    #[test]
    fn square_bracket_function() {
        let input = parse!("v1  [v1, v2]+5 + v1[]");
        let res = parse!("v1(v1,v2)+5+v1()");
        assert_eq!(input, res);
    }

    #[test]
    fn poly() {
        let var_names = ["v1".into(), "v2".into()];
        let var_map = Arc::new(vec![symbol!("v1").into(), symbol!("v2").into()]);
        let (rest, input) =
            Token::parse_polynomial::<_, u8>("#ABC*v1^2*v2+5".as_bytes(), &var_map, &var_names, &Z);

        assert!(rest.is_empty());
        assert_eq!(
            input,
            parse!("5+2748*v1^2*v2").to_polynomial(&Z, var_map.clone())
        );
    }

    #[test]
    fn two_way_parse() {
        let input = parse!("a::{real}::b(c)");
        let s = input.to_canonical_string();
        assert_eq!(parse!(s), input);
    }
}
