//! Methods for printing atoms and polynomials.

use std::{
    fmt::{self, Error, Write},
    io::IsTerminal,
    sync::LazyLock,
};

use crate::{
    atom::{
        AddView, Atom, AtomCore, AtomView, FunctionBuilder, InlineVar, MulView, NumView, PowView,
        Symbol, VarView, representation::FunView,
    },
    coefficient::CoefficientView,
    domains::{SelfRing, finite_field::FiniteFieldCore, float::Complex, rational::Rational},
    state::State,
};

pub use numerica::printer::*;

/// Track if Symbolica should colorize output, based on the `SYMBOLICA_COLOR` environment variable or if stdout is a terminal.
static SHOULD_COLORIZE: LazyLock<bool> = LazyLock::new(|| {
    std::env::var("SYMBOLICA_COLOR")
        .map(|v| v != "0")
        .unwrap_or_else(|_| std::io::stdout().is_terminal())
});

/// Wrap a printable object with ANSI escape codes for coloring and styling in terminal output.
pub struct AnsiWrap<T> {
    pub value: T,
    pub mode: u8,
    pub color: u8,
}

impl<T: fmt::Display> From<T> for AnsiWrap<T> {
    fn from(value: T) -> Self {
        AnsiWrap::new(value)
    }
}

impl<T> AnsiWrap<T> {
    pub const fn new(value: T) -> Self {
        Self {
            value,
            mode: 0,
            color: 0,
        }
    }

    pub const fn red(value: T) -> Self {
        Self::new(value).color(1)
    }

    pub const fn yellow(value: T) -> Self {
        Self::new(value).color(3)
    }

    pub const fn purple(value: T) -> Self {
        Self::new(value).color(5)
    }

    pub const fn cyan(value: T) -> Self {
        Self::new(value).color(6)
    }

    pub const fn bright_magenta(value: T) -> Self {
        Self::new(value).color(13)
    }

    pub const fn color(mut self, color: u8) -> Self {
        self.color = color;
        self
    }

    pub const fn bold(mut self) -> Self {
        self.mode |= 1;
        self
    }

    pub const fn dimmed(mut self) -> Self {
        self.mode |= 2;
        self
    }

    pub const fn italic(mut self) -> Self {
        self.mode |= 4;
        self
    }

    pub fn should_colorize() -> bool {
        *SHOULD_COLORIZE
    }

    /// Calculate the character length after stripping ANSI escape codes, for use in alignment and formatting decisions.
    pub fn ansi_stripped_len(a: T) -> usize
    where
        T: AsRef<str>,
    {
        let s = a.as_ref();
        let mut in_ansi = false;
        let mut len = 0;
        for c in s.chars() {
            if c == '\x1b' {
                in_ansi = true;
            } else if in_ansi && c == 'm' {
                in_ansi = false;
            } else if !in_ansi {
                len += 1;
            }
        }
        len
    }
}

impl<T: fmt::Display> fmt::Display for AnsiWrap<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if *SHOULD_COLORIZE {
            write!(
                f,
                "\u{1b}[{};38;5;{}m{}\u{1b}[0m",
                self.mode, self.color, self.value
            )
        } else {
            write!(f, "{}", self.value)
        }
    }
}

/// A function that takes an atom and prints it in a custom way.
/// If the function returns `None`, the default printing is used.
pub type PrintFunction = Box<dyn Fn(AtomView, &PrintOptions) -> Option<String> + Send + Sync>;

macro_rules! define_formatters {
    ($($a:ident),*) => {
        $(
        trait $a {
            fn fmt_debug(
                &self,
                f: &mut fmt::Formatter,
            ) -> fmt::Result;

            fn fmt_output<W: std::fmt::Write>(
                &self,
                f: &mut W,
                print_opts: &PrintOptions,
                print_state: PrintState,
            ) -> Result<bool, Error>;
        })+
    };
}

define_formatters!(
    FormattedPrintVar,
    FormattedPrintNum,
    FormattedPrintFn,
    FormattedPrintPow,
    FormattedPrintMul,
    FormattedPrintAdd
);

/// A printer for atoms, useful in a [format!].
///
/// # Examples
///
/// ```
/// use symbolica::{atom::AtomCore, parse};
/// use symbolica::printer::PrintOptions;
/// let a = parse!("x + y");
/// println!("{}", a.printer(PrintOptions::latex()));
/// ```
pub struct AtomPrinter<'a> {
    pub atom: AtomView<'a>,
    pub print_opts: PrintOptions,
}

impl<'a> AtomPrinter<'a> {
    /// Create a new atom printer with default printing options.
    pub fn new(atom: AtomView<'a>) -> AtomPrinter<'a> {
        AtomPrinter {
            atom,
            print_opts: PrintOptions::default(),
        }
    }

    pub fn new_with_options(atom: AtomView<'a>, print_opts: PrintOptions) -> AtomPrinter<'a> {
        AtomPrinter { atom, print_opts }
    }

    fn format_bracket<W: std::fmt::Write>(
        bracket: char,
        f: &mut W,
        opts: &PrintOptions,
        print_state: PrintState,
    ) -> fmt::Result {
        if let Some(bracket_colors) = opts.bracket_level_colors {
            f.write_fmt(format_args!(
                "{}",
                AnsiWrap::new(bracket.encode_utf8(&mut [0; 4]))
                    .color(bracket_colors[print_state.bracket_level.min(15) as usize])
            ))?;
        } else {
            f.write_char(bracket)?;
        }
        Ok(())
    }

    /// Format an integer. Input must be digits only.
    fn format_digits<W: std::fmt::Write>(
        mut s: String,
        opts: &PrintOptions,
        print_state: &PrintState,
        f: &mut W,
    ) -> fmt::Result {
        if print_state.superscript && opts.mode.is_symbolica() {
            let map = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
            s = s
                .as_bytes()
                .iter()
                .map(|x| map[(x - b'0') as usize])
                .collect();

            return f.write_str(&s);
        }

        if let Some(c) = opts.number_thousands_separator {
            let mut first = true;
            for triplet in s.as_bytes().chunks(3) {
                if !first {
                    f.write_char(c)?;
                }
                f.write_str(std::str::from_utf8(triplet).unwrap())?;
                first = false;
            }

            Ok(())
        } else {
            f.write_str(&s)
        }
    }
}

impl fmt::Display for AtomPrinter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.atom
            .format(
                f,
                &self.print_opts.update_with_fmt(f),
                PrintState::from_fmt(f),
            )
            .map(|_| ())
    }
}

/// Settings for canonical ordering of atoms when printing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CanonicalOrderingSettings {
    pub include_namespace: bool,
    pub include_attributes: bool,
    pub hide_namespace: Option<&'static str>,
}

impl Default for CanonicalOrderingSettings {
    fn default() -> Self {
        Self {
            include_namespace: true,
            include_attributes: true,
            hide_namespace: None,
        }
    }
}

impl AtomView<'_> {
    fn fmt_debug(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AtomView::Num(n) => n.fmt_debug(fmt),
            AtomView::Var(v) => v.fmt_debug(fmt),
            AtomView::Fun(f) => f.fmt_debug(fmt),
            AtomView::Pow(p) => p.fmt_debug(fmt),
            AtomView::Mul(m) => m.fmt_debug(fmt),
            AtomView::Add(a) => a.fmt_debug(fmt),
        }
    }

    pub(crate) fn format<W: std::fmt::Write>(
        &self,
        fmt: &mut W,
        opts: &PrintOptions,
        print_state: PrintState,
    ) -> Result<bool, Error> {
        match self {
            AtomView::Num(n) => n.fmt_output(fmt, opts, print_state),
            AtomView::Var(v) => v.fmt_output(fmt, opts, print_state),
            AtomView::Fun(f) => f.fmt_output(fmt, opts, print_state),
            AtomView::Pow(p) => p.fmt_output(fmt, opts, print_state),
            AtomView::Mul(t) => t.fmt_output(fmt, opts, print_state),
            AtomView::Add(e) => e.fmt_output(fmt, opts, print_state),
        }
    }

    /// Construct a printer for the atom with special options.
    pub(crate) fn printer(&self, opts: PrintOptions) -> AtomPrinter<'_> {
        AtomPrinter::new_with_options(*self, opts)
    }

    pub(crate) fn to_canonically_ordered_string(
        &self,
        settings: CanonicalOrderingSettings,
    ) -> String {
        let fixed = self.canonical_string_sign_fix(&settings);
        fixed.as_view().to_canonical_string_symmetric(&settings)
    }

    /// Print the atom in a form that is unique and independent of any implementation details.
    pub(crate) fn to_canonical_string(&self) -> String {
        let settings = CanonicalOrderingSettings::default();
        let fixed = self.canonical_string_sign_fix(&settings);
        fixed.as_view().to_canonical_string_symmetric(&settings)
    }

    /// Print the atom in a form that is unique and independent of any implementation details,
    /// treating all antisymmetric functions as symmetric.
    fn to_canonical_string_symmetric(&self, settings: &CanonicalOrderingSettings) -> String {
        let mut s = String::new();
        self.to_canonical_view_impl(settings, &mut s);
        s
    }

    /// Fix the sign of antisymmetric functions so that they can be treated as symmetric under
    /// canonical string ordering.
    fn canonical_string_sign_fix(&self, settings: &CanonicalOrderingSettings) -> Atom {
        match self {
            AtomView::Num(_) | AtomView::Var(_) => self.to_owned(),
            AtomView::Fun(f) => {
                let mut fb = FunctionBuilder::new(f.get_symbol());
                for aa in f.iter() {
                    fb = fb.add_arg(aa.canonical_string_sign_fix(settings));
                }

                // sign changes in the arguments may cause a reordering, and linearity may
                // move minus sign out of the function
                let res = fb.finish();

                if f.is_antisymmetric() {
                    fn sort_args(ff: &FunView, settings: &CanonicalOrderingSettings) -> Atom {
                        let mut args = vec![];
                        for (i, aa) in ff.iter().enumerate() {
                            // all antisymmetric functions in the arguments can be treated as symmetric under canonical string ordering
                            args.push((aa.to_canonical_string_symmetric(settings), i));
                        }

                        args.sort();

                        if ff.is_antisymmetric() {
                            // find the number of swaps needed to sort the arguments
                            let mut order: Vec<_> = (0..args.len())
                                .map(|i| args.iter().position(|(_, j)| *j == i).unwrap())
                                .collect();
                            let mut swaps = 0;
                            for i in 0..order.len() {
                                let pos = order[i..].iter().position(|&x| x == i).unwrap();
                                order.copy_within(i..i + pos, i + 1);
                                swaps += pos;
                            }

                            if swaps % 2 == 1 {
                                return -ff.as_view();
                            }
                        }

                        ff.as_view().to_owned()
                    }

                    match res.as_view() {
                        AtomView::Fun(ff) => sort_args(&ff, settings),
                        AtomView::Mul(m) => {
                            // find the antisymmetric function in the product
                            let mut it = m.iter();
                            let first = it.next().unwrap_or_else(|| {
                                panic!("Expected at least two terms in product: {}", self)
                            });
                            let second = it.next().unwrap_or_else(|| {
                                panic!("Expected at least one term in product: {}", self)
                            });
                            if it.next().is_some() {
                                panic!("Expected at most two terms in product: {}", self);
                            }

                            if let AtomView::Fun(fff) = first {
                                sort_args(&fff, settings) * second
                            } else if let AtomView::Fun(fff) = second {
                                sort_args(&fff, settings) * first
                            } else {
                                panic!(
                                    "Expected one term in product to be antisymmetric function: {}",
                                    self
                                );
                            }
                        }
                        _ => panic!(
                            "Unexpected result from antisymmetric function sign fix of {}",
                            self
                        ),
                    }
                } else {
                    res
                }
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                b.canonical_string_sign_fix(settings)
                    .pow(e.canonical_string_sign_fix(settings))
            }
            AtomView::Mul(m) => {
                let mut terms = vec![];
                for x in m.iter() {
                    terms.push(x.canonical_string_sign_fix(settings));
                }

                Atom::mul_many(&terms)
            }
            AtomView::Add(a) => {
                let mut terms = vec![];
                for x in a.iter() {
                    terms.push(x.canonical_string_sign_fix(settings));
                }

                Atom::add_many(&terms)
            }
        }
    }

    fn to_canonical_view_impl(&self, settings: &CanonicalOrderingSettings, out: &mut String) {
        fn add_paren(cur: AtomView, s: AtomView) -> bool {
            if let AtomView::Pow(_) = cur {
                match s {
                    AtomView::Var(_) => false,
                    AtomView::Num(n) => match n.get_coeff_view() {
                        CoefficientView::Natural(c, d, ic, _) => c < 0 || ic != 0 || d != 1,
                        CoefficientView::Large(r, i) => {
                            r.is_negative() || !i.is_zero() || !r.to_rat().is_integer()
                        }
                        _ => true,
                    },
                    _ => true,
                }
            } else if let AtomView::Mul(_) = cur {
                matches!(s, AtomView::Add(_))
            } else {
                false
            }
        }

        match self {
            AtomView::Num(_) => write!(out, "{}", self.printer(PrintOptions::file())).unwrap(),
            AtomView::Var(v) => {
                if settings.include_attributes {
                    v.get_symbol().format(&PrintOptions::full(), out).unwrap();
                } else if settings.include_namespace {
                    v.get_symbol()
                        .format(
                            &PrintOptions {
                                hide_namespace: settings.hide_namespace,
                                ..PrintOptions::file()
                            },
                            out,
                        )
                        .unwrap();
                } else {
                    v.get_symbol()
                        .format(&PrintOptions::file_no_namespace(), out)
                        .unwrap();
                }
            }
            AtomView::Fun(f) => {
                if settings.include_attributes {
                    f.get_symbol().format(&PrintOptions::full(), out).unwrap();
                } else if settings.include_namespace {
                    f.get_symbol()
                        .format(
                            &PrintOptions {
                                hide_namespace: settings.hide_namespace,
                                ..PrintOptions::file()
                            },
                            out,
                        )
                        .unwrap();
                } else {
                    f.get_symbol()
                        .format(&PrintOptions::file_no_namespace(), out)
                        .unwrap();
                }
                out.push('(');

                let mut args = vec![];

                for x in f.iter() {
                    let mut arg = String::new();
                    x.to_canonical_view_impl(settings, &mut arg);
                    args.push(arg);
                }

                // The potential sign flip for antisymmetric functions should have been
                // taken into account in the input, by the canonical_string_sign_fix() function
                if f.is_symmetric() || f.is_antisymmetric() {
                    args.sort();
                }

                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(out, ",").unwrap();
                    }
                    write!(out, "{arg}").unwrap();
                }

                write!(out, ")").unwrap();
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();

                if add_paren(*self, b) {
                    write!(out, "(").unwrap();
                    b.to_canonical_view_impl(settings, out);
                    write!(out, ")").unwrap();
                } else {
                    b.to_canonical_view_impl(settings, out);
                }

                if add_paren(*self, e) {
                    write!(out, "^(").unwrap();
                    e.to_canonical_view_impl(settings, out);
                    write!(out, ")").unwrap();
                } else {
                    write!(out, "^").unwrap();
                    e.to_canonical_view_impl(settings, out);
                }
            }
            AtomView::Mul(m) => {
                let mut terms = vec![];

                for x in m.iter() {
                    let mut term = if add_paren(*self, x) {
                        "(".to_string()
                    } else {
                        String::new()
                    };

                    x.to_canonical_view_impl(settings, &mut term);

                    if add_paren(*self, x) {
                        term.push(')');
                    }

                    terms.push(term);
                }

                terms.sort();

                for (i, term) in terms.iter().enumerate() {
                    if i > 0 {
                        write!(out, "*").unwrap();
                    }

                    write!(out, "{term}").unwrap();
                }
            }
            AtomView::Add(a) => {
                let mut terms = vec![];

                for x in a.iter() {
                    let mut term = if add_paren(*self, x) {
                        "(".to_string()
                    } else {
                        String::new()
                    };

                    x.to_canonical_view_impl(settings, &mut term);

                    if add_paren(*self, x) {
                        term.push(')');
                    }

                    terms.push(term);
                }

                terms.sort();

                for (i, term) in terms.iter().enumerate() {
                    if i > 0 {
                        write!(out, "+").unwrap();
                    }
                    write!(out, "{term}").unwrap();
                }
            }
        }
    }

    /// Estimate the length of the string representation of the atom, for use in deciding when to split terms onto new lines.
    fn estimate_char_length(&self, opts: &PrintOptions) -> usize {
        match self {
            AtomView::Num(n) => match n.get_coeff_view() {
                CoefficientView::Natural(num, den, num_i, den_i) => {
                    let mut len = 0;
                    if num <= 0 {
                        len += 1;
                    }
                    if num_i < 0 {
                        len += 1;
                    }
                    if num != 0 && num_i != 1 {
                        len += 1;
                    }

                    if num != 0 {
                        len += num.unsigned_abs().ilog10();
                    }
                    if num_i != 0 {
                        len += num_i.unsigned_abs().ilog10();
                    }
                    if den != 1 {
                        len += 1 + den.unsigned_abs().ilog10();
                    }
                    if den_i != 1 {
                        len += 1 + den_i.unsigned_abs().ilog10();
                    }

                    len as usize
                }
                CoefficientView::Large(_, _) => {
                    (self.get_byte_size() as f64 * 1.6) as usize // stored in hex
                }
                CoefficientView::Indeterminate | CoefficientView::Infinity(None) => 1,
                _ => self.get_byte_size(),
            },
            AtomView::Var(v) => {
                if !opts.hide_all_namespaces {
                    v.get_symbol().get_name().len()
                } else {
                    v.get_symbol().get_stripped_name().len()
                }
            }
            AtomView::Fun(f) => {
                let mut len = 2;
                if !opts.hide_all_namespaces {
                    len += f.get_symbol().get_name().len();
                } else {
                    len += f.get_symbol().get_stripped_name().len();
                }

                let iter = f.iter();
                if iter.len() > 0 {
                    len += iter.len() - 1;
                }
                for x in iter {
                    len += x.estimate_char_length(opts);
                }
                len
            }
            AtomView::Pow(p) => {
                let (b, e) = p.get_base_exp();
                b.estimate_char_length(opts) + e.estimate_char_length(opts) + 1
            }
            AtomView::Mul(m) => {
                let mut len = 0;
                let iter = m.iter();
                if iter.len() > 0 {
                    len += iter.len() - 1;
                }
                for x in iter {
                    len += x.estimate_char_length(opts);
                }
                len
            }
            AtomView::Add(a) => {
                let mut len = 0;
                let iter = a.iter();
                if iter.len() > 0 {
                    len += iter.len() - 1;
                }
                for x in iter {
                    len += x.estimate_char_length(opts);
                }
                len
            }
        }
    }
}

impl fmt::Debug for AtomView<'_> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        if fmt.alternate() {
            self.fmt_debug(fmt)
        } else {
            std::fmt::Display::fmt(
                &AtomPrinter::new_with_options(*self, PrintOptions::file()),
                fmt,
            )
        }
    }
}

impl FormattedPrintVar for VarView<'_> {
    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        print_state: PrintState,
    ) -> Result<bool, Error> {
        if print_state.in_sum {
            if print_state.top_level_add_child
                && opts.mode.is_symbolica()
                && opts.color_top_level_sum
            {
                f.write_fmt(format_args!("{}", AnsiWrap::yellow("+")))?;
            } else {
                f.write_char('+')?;
            }
        }

        let id = self.get_symbol();
        id.format(opts, f)?;
        Ok(false)
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }
}

impl FormattedPrintNum for NumView<'_> {
    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }

    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> Result<bool, Error> {
        let d = self.get_coeff_view();

        let global_negative = match d {
            CoefficientView::Natural(n, _, ni, _) => n < 0 && ni == 0 || ni < 0 && n == 0,
            CoefficientView::Large(r, ri) => {
                r.is_negative() && ri.is_zero() || ri.is_negative() && r.is_zero()
            }
            _ => false,
        } && print_state.in_sum;

        print_state.superscript &= match d {
            CoefficientView::Natural(_, d, ni, _) => d == 1 && ni == 0,
            CoefficientView::Large(r, ri) => r.to_rat().is_integer() && ri.is_zero(),
            _ => false,
        };

        if global_negative {
            if print_state.top_level_add_child
                && opts.mode.is_symbolica()
                && opts.color_top_level_sum
            {
                f.write_fmt(format_args!("{}", AnsiWrap::yellow("-")))?;
            } else if print_state.superscript {
                f.write_char('⁻')?;
            } else {
                f.write_char('-')?;
            }

            print_state.in_sum = false;
        } else if print_state.in_sum {
            if print_state.top_level_add_child
                && opts.mode.is_symbolica()
                && opts.color_top_level_sum
            {
                f.write_fmt(format_args!("{}", AnsiWrap::yellow("+")))?;
            } else {
                f.write_char('+')?;
            }

            print_state.in_sum = false;
        }

        let i_str = if opts.mode.is_symbolica() && opts.color_builtin_symbols {
            "\u{1b}\u{5b}\u{33}\u{35}\u{6d}\u{1d456}\u{1b}\u{5b}\u{30}\u{6d}"
        } else if opts.mode.is_mathematica() {
            "I"
        } else {
            "𝑖"
        };

        fn print_complex_rational<W: std::fmt::Write>(
            real: Rational,
            imag: Rational,
            global_negative: bool,
            i_str: &str,
            print_state: PrintState,
            f: &mut W,
            opts: &PrintOptions,
        ) -> Result<bool, Error> {
            if imag.is_zero()
                && real.denominator_ref().is_one()
                && print_state.suppress_one
                && (real.numerator_ref().is_one() || *real.numerator_ref() == -1)
            {
                if *real.numerator_ref() == -1 && !global_negative {
                    f.write_char('-')?;
                }
                return Ok(true);
            }

            let need_paren =
                (print_state.in_product || print_state.in_exp || print_state.in_exp_base)
                    && (!real.is_zero() && !imag.is_zero())
                    || print_state.in_exp_base
                        && (real.is_negative() || !imag.is_zero() || !real.is_integer())
                    || print_state.in_exp && (!real.is_integer() || !imag.is_zero());

            if need_paren {
                AtomPrinter::format_bracket('(', f, opts, print_state)?;
            }

            if !opts.mode.is_latex()
                && (opts.number_thousands_separator.is_some() || print_state.superscript)
            {
                if !real.is_zero() {
                    if !global_negative && real.is_negative() {
                        if print_state.superscript {
                            f.write_char('⁻')?;
                        } else {
                            f.write_char('-')?;
                        }
                    }

                    AtomPrinter::format_digits(
                        real.numerator_ref().abs().to_string(),
                        opts,
                        &print_state,
                        f,
                    )?;
                    if !real.is_integer() {
                        f.write_char('/')?;
                        AtomPrinter::format_digits(
                            real.denominator_ref().to_string(),
                            opts,
                            &print_state,
                            f,
                        )?;
                    }
                }

                if !real.is_zero() && !imag.is_zero() && !imag.is_negative() {
                    f.write_char('+')?;
                }

                if !imag.is_zero() {
                    if !global_negative && imag.is_negative() {
                        f.write_char('-')?;
                    }
                    AtomPrinter::format_digits(
                        imag.numerator_ref().abs().to_string(),
                        opts,
                        &print_state,
                        f,
                    )?;
                    f.write_str(i_str)?;
                    if !imag.is_integer() {
                        f.write_char('/')?;
                        AtomPrinter::format_digits(
                            imag.denominator_ref().to_string(),
                            opts,
                            &print_state,
                            f,
                        )?;
                    }
                }
            } else {
                if !real.is_zero() || imag.is_zero() {
                    if !global_negative && real.is_negative() {
                        f.write_char('-')?;
                    }
                    if !real.is_integer() {
                        if opts.mode.is_latex() {
                            f.write_fmt(format_args!(
                                "\\frac{{{}}}{{{}}}",
                                real.numerator_ref().abs(),
                                real.denominator_ref()
                            ))?;
                        } else {
                            f.write_fmt(format_args!(
                                "{}/{}",
                                real.numerator_ref().abs(),
                                real.denominator_ref()
                            ))?;
                        }
                    } else {
                        f.write_fmt(format_args!("{}", real.numerator_ref().abs()))?;
                    }
                }

                if !real.is_zero() && !imag.is_zero() && !imag.is_negative() {
                    f.write_char('+')?;
                }

                if !imag.is_zero() {
                    if !global_negative && imag.is_negative() {
                        f.write_char('-')?;
                    }

                    if !imag.is_integer() {
                        if opts.mode.is_latex() {
                            f.write_fmt(format_args!(
                                "\\frac{{{}}}{{{}}}𝑖",
                                imag.numerator_ref().abs(),
                                imag.denominator_ref(),
                            ))?;
                        } else {
                            f.write_fmt(format_args!(
                                "{}{}/{}",
                                imag.numerator_ref().abs(),
                                i_str,
                                imag.denominator_ref()
                            ))?;
                        }
                    } else {
                        f.write_fmt(format_args!("{}{}", imag.numerator_ref().abs(), i_str))?;
                    }
                }
            }

            if need_paren {
                AtomPrinter::format_bracket(')', f, opts, print_state)?;
            }

            Ok(false)
        }

        match d {
            CoefficientView::Natural(num, den, num_i, den_i) => {
                let real = Rational::from_int_unchecked(num, den);
                let imag = Rational::from_int_unchecked(num_i, den_i);

                print_complex_rational(real, imag, global_negative, i_str, print_state, f, opts)
            }
            CoefficientView::Float(r, i) => {
                if i.is_zero() {
                    r.to_float().format(opts, print_state, f)?;
                } else {
                    Complex::new(r.to_float(), i.to_float()).format(opts, print_state, f)?;
                }

                Ok(false)
            }
            CoefficientView::Large(r, i) => {
                let real = r.to_rat();
                let imag = i.to_rat();

                print_complex_rational(real, imag, global_negative, i_str, print_state, f, opts)
            }
            CoefficientView::Indeterminate => {
                f.write_char('¿')?;
                Ok(false)
            }
            CoefficientView::Infinity(None) => {
                f.write_char('⧞')?;
                Ok(false)
            }
            CoefficientView::Infinity(Some((r, i))) => {
                let real = r.to_rat();
                let imag = i.to_rat();

                if imag.is_zero() {
                    if real.is_negative() {
                        if opts.mode.is_latex() {
                            f.write_str("-\\infty")?;
                        } else {
                            f.write_str("-∞")?;
                        }
                    } else if opts.mode.is_latex() {
                        f.write_str("\\infty")?;
                    } else {
                        f.write_char('∞')?;
                    }
                } else {
                    print_state.in_product = true;
                    print_complex_rational(
                        real,
                        imag,
                        global_negative,
                        i_str,
                        print_state,
                        f,
                        opts,
                    )?;

                    if opts.mode.is_latex() {
                        f.write_str(" \\infty")?;
                    } else {
                        f.write_char(opts.multiplication_operator)?;
                        f.write_char('∞')?;
                    }
                }
                Ok(false)
            }
            CoefficientView::FiniteField(num, fi) => {
                let ff = State::get_finite_field(fi);
                f.write_fmt(format_args!(
                    "[{}%{}]",
                    ff.from_element(&num),
                    ff.get_prime()
                ))?;
                Ok(false)
            }
            CoefficientView::RationalPolynomial(p) => {
                f.write_char('[')?;
                p.deserialize().format(opts, print_state, f)?;
                f.write_char(']').map(|_| false)
            }
        }
    }
}

impl FormattedPrintMul for MulView<'_> {
    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }

    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> Result<bool, Error> {
        let add_paren = print_state.in_exp || print_state.in_exp_base;
        if add_paren {
            if print_state.in_sum {
                print_state.in_sum = false;
                f.write_char('+')?;
            }

            AtomPrinter::format_bracket('(', f, opts, print_state)?;
            print_state.in_exp = false;
            print_state.in_exp_base = false;
            print_state.bracket_level += 1;
        }

        print_state.in_product = true;

        let mut num_count = 0;
        let mut den_count = 0;

        // write the coefficient first
        let mut first = true;
        let mut skip_num = false;
        if let Some(AtomView::Num(n)) = self.iter().next() {
            print_state.suppress_one = true;
            first = n.fmt_output(f, opts, print_state)?;
            print_state.suppress_one = false;
            skip_num = true;
            num_count += 1;
        } else if print_state.in_sum {
            if print_state.top_level_add_child
                && opts.mode.is_symbolica()
                && opts.color_top_level_sum
            {
                f.write_fmt(format_args!("{}", AnsiWrap::yellow("+")))?;
            } else {
                f.write_char('+')?;
            }
        }

        print_state.top_level_add_child = false;
        print_state.level += 1;
        print_state.in_sum = false;

        let global_split = if !opts.fill_indented_lines
            && let Some(max) = opts.max_line_length
        {
            self.get_byte_size() > max + max / 2
                || self.as_view().estimate_char_length(opts)
                    + print_state.indentation_level as usize * opts.indentation
                    > max
        } else {
            false
        };

        let mut last_factor_has_brackets = false;
        let old_ident_level = print_state.indentation_level;
        let mut char_count = print_state.indentation_level as usize * opts.indentation;
        let mut den_char_count = 0;
        let mut count = 0; //if skip_num { 1 } else { 0 };
        let mut was_split = false;
        for x in self.iter().skip(if skip_num { 1 } else { 0 }) {
            // count and skip denominators
            if let AtomView::Pow(p) = x {
                let (_, e) = p.get_base_exp();
                if let AtomView::Num(n) = e {
                    if let CoefficientView::Natural(num, _, 0, 1) = n.get_coeff_view() {
                        if num < 0 {
                            den_count += 1;

                            if opts.fill_indented_lines && opts.max_line_length.is_some() {
                                den_char_count += x.estimate_char_length(opts);
                            }
                            continue;
                        }
                    }
                }
            }

            num_count += 1;

            let (local_split, arg_char_len, arg_splits_with_brackets) = if opts.fill_indented_lines
                && let Some(max) = opts.max_line_length
            {
                let arg_char_len = x.estimate_char_length(opts);
                (
                    char_count + arg_char_len > max,
                    arg_char_len,
                    matches!(x, AtomView::Fun(_) | AtomView::Add(_))
                        && arg_char_len + print_state.indentation_level as usize * opts.indentation
                            > max,
                )
            } else {
                (false, 0, false)
            };

            if global_split || local_split {
                if count >= 1 {
                    print_state.indentation_level = old_ident_level + 1;
                }

                if count > 0 && !last_factor_has_brackets {
                    f.write_char('\n')?;

                    for _ in 0..print_state.indentation_level as usize * opts.indentation {
                        f.write_char(' ')?;
                    }

                    was_split = true;
                    char_count = print_state.indentation_level as usize * opts.indentation + 1;
                }
            }

            // if the current arg ends with a bracket on a new line, reset the char count of the current line
            last_factor_has_brackets = arg_splits_with_brackets;
            if last_factor_has_brackets {
                char_count = print_state.indentation_level as usize * opts.indentation + 1;
            } else {
                char_count += arg_char_len;
            }

            if !first {
                if opts.mode.is_latex() {
                    f.write_char(' ')?;
                } else {
                    f.write_char(opts.multiplication_operator)?;
                }
            }

            x.format(f, opts, print_state)?;

            first = false;
            count += 1;
        }

        if den_count > 0 {
            if num_count == 0 {
                f.write_char('1')?;
            }

            // always do a global check on the args to see if we need to put
            // the division on a new line
            if (global_split
                || opts.fill_indented_lines
                    && if let Some(max) = opts.max_line_length {
                        char_count + den_char_count > max
                    } else {
                        false
                    })
                && count >= 1
            {
                print_state.indentation_level = old_ident_level + 1;

                f.write_char('\n')?;
                for _ in 0..print_state.indentation_level as usize * opts.indentation {
                    f.write_char(' ')?;
                }

                char_count = print_state.indentation_level as usize * opts.indentation + 2;
            } else {
                char_count += 2;
            }

            f.write_char('/')?;

            if den_count > 1 {
                AtomPrinter::format_bracket('(', f, opts, print_state)?;
                print_state.bracket_level += 1;
            }

            count = 0;
            first = true;
            last_factor_has_brackets = false;
            for x in self.iter() {
                if let AtomView::Pow(p) = x {
                    let (b, e) = p.get_base_exp();
                    if let AtomView::Num(n) = e
                        && let CoefficientView::Natural(num, den, 0, 1) = n.get_coeff_view()
                        && num < 0
                    {
                        let (local_split, arg_char_len, arg_splits_with_brackets) = if opts
                            .fill_indented_lines
                            && let Some(max) = opts.max_line_length
                        {
                            let arg_char_len = x.estimate_char_length(opts);
                            (
                                char_count + arg_char_len > max,
                                arg_char_len,
                                matches!(x, AtomView::Fun(_) | AtomView::Add(_))
                                    && arg_char_len
                                        + print_state.indentation_level as usize * opts.indentation
                                        > max,
                            )
                        } else {
                            (false, 0, false)
                        };

                        if global_split || local_split {
                            if count >= 1 {
                                print_state.indentation_level = old_ident_level + 2;
                            }

                            if count > 0 && !last_factor_has_brackets {
                                f.write_char('\n')?;

                                for _ in
                                    0..print_state.indentation_level as usize * opts.indentation
                                {
                                    f.write_char(' ')?;
                                }

                                was_split = true;
                                char_count =
                                    print_state.indentation_level as usize * opts.indentation + 1;
                            }
                        }

                        // if the current arg ends with a bracket on a new line, reset the char count of the current line
                        last_factor_has_brackets = arg_splits_with_brackets;
                        if last_factor_has_brackets {
                            char_count =
                                print_state.indentation_level as usize * opts.indentation + 1;
                        } else {
                            char_count += arg_char_len;
                        }

                        if !first {
                            if opts.mode.is_latex() {
                                f.write_char(' ')?;
                            } else {
                                f.write_char(opts.multiplication_operator)?;
                            }
                        }

                        count += 1;
                        first = false;

                        let mut new_print_state = print_state;
                        new_print_state.in_exp_base = true;
                        b.format(f, opts, new_print_state)?;

                        let exp = Rational::new(num, den).neg();
                        if exp.is_one() {
                            continue;
                        }

                        new_print_state.in_exp = true;
                        new_print_state.in_exp_base = false;

                        let superscript_exponent = opts.num_exp_as_superscript && den == 1;

                        if !superscript_exponent {
                            if opts.mode.is_sympy()
                                || (!opts.mode.is_latex() && opts.double_star_for_exponentiation)
                            {
                                f.write_str("**")?;
                            } else {
                                f.write_char('^')?;
                            }
                        }

                        if opts.mode.is_latex() {
                            f.write_char('{')?;
                            new_print_state.in_exp = false;
                            exp.format(opts, new_print_state, f)?;
                            f.write_char('}')?;
                        } else {
                            if superscript_exponent {
                                new_print_state.in_exp = false;
                                new_print_state.superscript = true;
                                AtomPrinter::format_digits(
                                    num.unsigned_abs().to_string(),
                                    opts,
                                    &new_print_state,
                                    f,
                                )?;
                            } else {
                                exp.format(opts, new_print_state, f)?;
                            }
                        }
                    }
                }
            }

            if den_count > 1 {
                print_state.bracket_level -= 1;
                AtomPrinter::format_bracket(')', f, opts, print_state)?;
            }
        }

        if add_paren {
            if was_split {
                print_state.indentation_level -= 1;
                f.write_char('\n')?;
                for _ in 0..print_state.indentation_level as usize * opts.indentation {
                    f.write_char(' ')?;
                }
            }

            print_state.bracket_level -= 1;
            AtomPrinter::format_bracket(')', f, opts, print_state)?;
        }
        Ok(false)
    }
}

impl FormattedPrintFn for FunView<'_> {
    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> Result<bool, Error> {
        if print_state.in_sum {
            if print_state.top_level_add_child
                && opts.mode.is_symbolica()
                && opts.color_top_level_sum
            {
                f.write_fmt(format_args!("{}", AnsiWrap::yellow("+")))?;
            } else {
                f.write_char('+')?;
            }
        }

        let id = self.get_symbol();
        if let Some(custom_print) = &id.get_global_data().custom_print
            && let Some(s) = custom_print(self.as_view(), opts)
        {
            f.write_str(&s)?;
            return Ok(false);
        }

        id.format(opts, f)?;

        if print_state.bracket_level == 0 {
            // the first level is only for top level products
            print_state.bracket_level = 1;
        }

        #[allow(deprecated)]
        let brackets = if opts.square_brackets_for_function {
            ('[', ']')
        } else {
            opts.function_brackets
        };

        if opts.mode.is_latex() {
            f.write_str("\\!\\left(")?;
        } else if opts.mode.is_mathematica() {
            f.write_char('[')?;
        } else {
            AtomPrinter::format_bracket(brackets.0, f, opts, print_state)?;
        }

        print_state.bracket_level += 1;

        print_state.top_level_add_child = false;
        print_state.level += 1;
        print_state.in_sum = false;
        print_state.in_product = false;
        print_state.in_exp = false;
        print_state.in_exp_base = false;
        print_state.suppress_one = false;

        let iter = self.iter();

        let global_split = if !opts.fill_indented_lines
            && let Some(max) = opts.max_line_length
        {
            self.get_byte_size() > max + max / 2
                || self.as_view().estimate_char_length(opts)
                    + print_state.indentation_level as usize * opts.indentation
                    > max
        } else {
            false
        };

        let old_ident_level = print_state.indentation_level;

        let mut char_count = print_state.indentation_level as usize * opts.indentation
            + InlineVar::new(id).as_view().estimate_char_length(opts);

        if global_split {
            print_state.indentation_level += 1;
        }

        let mut was_split = false;
        let mut first = true;
        for x in iter {
            if opts.mode.is_mathematica() {
                if let AtomView::Var(s) = x
                    && s.get_symbol() == Symbol::SEP
                {
                    first = true;
                    f.write_str("][")?;
                    continue;
                }

                // curry the derivative function
                if id == Symbol::DERIVATIVE
                    && let AtomView::Fun(fun) = x
                {
                    f.write_str("][")?;
                    fun.get_symbol().format(opts, f)?;
                    f.write_str("][")?;

                    first = true;
                    for x2 in fun.iter() {
                        if !first {
                            f.write_char(',')?;
                        } else {
                            first = false;
                        }

                        x2.format(f, opts, print_state)?;
                    }
                    continue;
                }
            }

            if !first {
                f.write_char(',')?;
            }

            let (local_split, arg_char_len, arg_split) = if opts.fill_indented_lines
                && let Some(max) = opts.max_line_length
            {
                let arg_char_len = x.estimate_char_length(opts);
                (
                    char_count + arg_char_len > max,
                    arg_char_len,
                    print_state.indentation_level as usize * opts.indentation + arg_char_len > max,
                )
            } else {
                (false, 0, false)
            };

            if !opts.mode.is_mathematica() {
                if !global_split && local_split {
                    print_state.indentation_level = old_ident_level + 1;
                }

                if local_split && first {
                    // the first argument splits, so move the closing
                    // bracket of a function to a new line
                    was_split = true;
                }

                if global_split || (local_split && (!first || !arg_split)) {
                    f.write_char('\n')?;
                    for _ in 0..print_state.indentation_level {
                        for _ in 0..opts.indentation {
                            f.write_char(' ')?;
                        }
                    }

                    was_split = true;

                    char_count = print_state.indentation_level as usize * opts.indentation;
                }

                char_count += arg_char_len;
            }

            first = false;

            x.format(f, opts, print_state)?;
        }

        if was_split {
            print_state.indentation_level -= 1;
            f.write_char('\n')?;
            for _ in 0..print_state.indentation_level as usize * opts.indentation {
                f.write_char(' ')?;
            }
        }

        print_state.bracket_level -= 1;
        if opts.mode.is_latex() {
            f.write_str("\\right)")?;
        } else if opts.mode.is_mathematica() {
            f.write_char(']')?;
        } else {
            AtomPrinter::format_bracket(brackets.1, f, opts, print_state)?;
        }

        Ok(false)
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }
}

impl FormattedPrintPow for PowView<'_> {
    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> Result<bool, Error> {
        if print_state.in_sum {
            if opts.mode.is_symbolica()
                && print_state.top_level_add_child
                && opts.color_top_level_sum
            {
                f.write_fmt(format_args!("{}", AnsiWrap::yellow("+")))?;
            } else {
                f.write_char('+')?;
            }
        }

        let add_paren = print_state.in_exp_base; // right associative
        if add_paren {
            AtomPrinter::format_bracket('(', f, opts, print_state)?;
            print_state.in_exp = false;
            print_state.in_exp_base = false;
            print_state.bracket_level += 1;
        }

        let b = self.get_base();
        let e = self.get_exp();

        print_state.top_level_add_child = false;
        print_state.level += 1;
        print_state.in_sum = false;
        print_state.in_product = false;
        print_state.suppress_one = false;

        let mut superscript_exponent = false;
        if opts.mode.is_latex() {
            if let AtomView::Num(n) = e
                && n.get_coeff_view() == CoefficientView::Natural(-1, 1, 0, 1)
            {
                // TODO: construct the numerator
                f.write_str("\\frac{1}{")?;
                b.format(f, opts, print_state)?;
                f.write_char('}')?;
                return Ok(false);
            }
        } else if opts.mode.is_symbolica()
            && opts.num_exp_as_superscript
            && let AtomView::Num(n) = e
        {
            superscript_exponent = n.get_coeff_view().is_integer()
        }

        print_state.in_exp_base = true;

        // detect denominator
        if let AtomView::Num(n) = e
            && let CoefficientView::Natural(num, den, 0, 1) = n.get_coeff_view()
            && num < 0
        {
            f.write_str("1/")?;
            b.format(f, opts, print_state)?;

            print_state.in_exp_base = false;

            let exp = Rational::new(num, den).neg();
            if !exp.is_one() {
                print_state.in_exp = true;

                let superscript_exponent = opts.num_exp_as_superscript && den == 1;

                if !superscript_exponent {
                    if opts.mode.is_sympy()
                        || (!opts.mode.is_latex() && opts.double_star_for_exponentiation)
                    {
                        f.write_str("**")?;
                    } else {
                        f.write_char('^')?;
                    }
                }

                if opts.mode.is_latex() {
                    f.write_char('{')?;
                    print_state.in_exp = false;
                    exp.format(opts, print_state, f)?;
                    f.write_char('}')?;
                } else if superscript_exponent {
                    print_state.in_exp = false;
                    print_state.superscript = true;
                    AtomPrinter::format_digits(
                        num.unsigned_abs().to_string(),
                        opts,
                        &print_state,
                        f,
                    )?;
                } else {
                    exp.format(opts, print_state, f)?;
                }
            }
        } else {
            b.format(f, opts, print_state)?;

            print_state.in_exp_base = false;
            print_state.in_exp = true;

            if !superscript_exponent {
                if opts.mode.is_sympy()
                    || (!opts.mode.is_latex() && opts.double_star_for_exponentiation)
                {
                    f.write_str("**")?;
                } else {
                    f.write_char('^')?;
                }
            }

            if opts.mode.is_latex() {
                f.write_char('{')?;
                print_state.in_exp = false;
                e.format(f, opts, print_state)?;
                f.write_char('}')?;
            } else {
                if superscript_exponent {
                    print_state.in_exp = false;
                    print_state.superscript = true;
                }

                e.format(f, opts, print_state)?;
            }
        }

        if add_paren {
            print_state.bracket_level -= 1;
            AtomPrinter::format_bracket(')', f, opts, print_state)?;
        }

        Ok(false)
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }
}

impl FormattedPrintAdd for AddView<'_> {
    fn fmt_output<W: std::fmt::Write>(
        &self,
        f: &mut W,
        opts: &PrintOptions,
        mut print_state: PrintState,
    ) -> Result<bool, Error> {
        print_state.top_level_add_child = print_state.level == 0;
        print_state.level += 1;
        print_state.suppress_one = false;

        let add_paren = print_state.in_product || print_state.in_exp || print_state.in_exp_base;
        if add_paren {
            if print_state.in_sum {
                if opts.mode.is_symbolica()
                    && print_state.top_level_add_child
                    && opts.color_top_level_sum
                {
                    f.write_fmt(format_args!("{}", AnsiWrap::yellow("+")))?;
                } else {
                    f.write_char('+')?;
                }
            }

            print_state.in_sum = false;
            print_state.in_product = false;
            print_state.in_exp = false;
            print_state.in_exp_base = false;

            if opts.mode.is_latex() {
                f.write_str("\\left(")?;
            } else {
                AtomPrinter::format_bracket('(', f, opts, print_state)?;
            }

            print_state.bracket_level += 1;
        }

        let global_split = print_state.top_level_add_child && opts.terms_on_new_line
            || if !opts.fill_indented_lines
                && let Some(max) = opts.max_line_length
            {
                self.get_byte_size() > max + max / 2
                    || self.as_view().estimate_char_length(opts)
                        + print_state.indentation_level as usize * opts.indentation
                        > max
            } else {
                false
            };

        let old_ident_level = print_state.indentation_level;
        let mut last_arg_splits_with_brackets = false;
        let mut count = 0;
        let mut char_count = print_state.indentation_level as usize * opts.indentation;
        let mut was_split = false;
        for x in self.iter() {
            if let Some(max_terms) = opts.max_terms
                && opts.mode.is_symbolica()
                && count >= max_terms
            {
                break;
            }

            let (local_split, arg_char_len, arg_splits_with_brackets) = if opts.fill_indented_lines
                && let Some(max) = opts.max_line_length
            {
                let arg_char_len = x.estimate_char_length(opts);
                (
                    char_count + arg_char_len > max,
                    arg_char_len,
                    matches!(x, AtomView::Fun(_))
                        && arg_char_len + print_state.indentation_level as usize * opts.indentation
                            > max,
                )
            } else {
                (false, 0, false)
            };

            if global_split || local_split {
                if count >= 1 {
                    print_state.indentation_level = old_ident_level + 1;
                }

                if count > 0 && !last_arg_splits_with_brackets {
                    f.write_char('\n')?;

                    if print_state.top_level_add_child && opts.terms_on_new_line {
                        char_count = 0; // do not indent top-level sum
                    } else {
                        for _ in 0..print_state.indentation_level as usize * opts.indentation {
                            f.write_char(' ')?;
                        }

                        char_count = print_state.indentation_level as usize * opts.indentation;
                    }

                    was_split = true;
                }
            }

            x.format(f, opts, print_state)?;

            last_arg_splits_with_brackets = arg_splits_with_brackets;
            if last_arg_splits_with_brackets {
                char_count = print_state.indentation_level as usize * opts.indentation + 1;
            } else {
                char_count += arg_char_len;
            }

            print_state.in_sum = true;
            count += 1;
        }

        if opts.max_terms.is_some() && count < self.get_nargs() {
            if was_split || print_state.top_level_add_child && opts.terms_on_new_line {
                f.write_char('\n')?;
            }
            if print_state.top_level_add_child
                && opts.mode.is_symbolica()
                && opts.color_top_level_sum
            {
                f.write_fmt(format_args!("{0}...", AnsiWrap::yellow("+")))?;
            } else {
                f.write_str("+...")?;
            }
        }

        if add_paren {
            print_state.bracket_level -= 1;
            if was_split {
                print_state.indentation_level -= 1;
                f.write_char('\n')?;
                for _ in 0..print_state.indentation_level as usize * opts.indentation {
                    f.write_char(' ')?;
                }
            }

            if opts.mode.is_latex() {
                f.write_str("\\right)")?;
            } else {
                AtomPrinter::format_bracket(')', f, opts, print_state)?;
            }
        }
        Ok(false)
    }

    fn fmt_debug(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }
}

#[cfg(test)]
mod test {
    use crate::{
        atom::{AtomCore, AtomView},
        domains::{SelfRing, finite_field::Zp, integer::Z},
        function, parse, parse_lit,
        printer::{AnsiWrap, AtomPrinter, PrintOptions, PrintState},
        symbol,
    };

    #[test]
    fn nested() {
        let b = parse_lit!(
            3 + v1
                * v2
                * v3
                * f(
                    87238723, sda98as9d8, dsdjsdjsd, dskdjaskj, sdkjsdksd, djksdskdj
                )
                / (f(x, y, z, a, b, c, e, f, g, g(x), h(x, y, z))
                    * g(x, y, z, a, b, c, e, f, g, g(x), h(x, y, z))
                    * h(x, y, z, a, b, c, e, f, g, g(x), h(x, y, z)))
                + h(4)
        );

        let out = b.format_string(
            &PrintOptions {
                max_line_length: Some(40),
                hide_all_namespaces: true,
                ..PrintOptions::file()
            },
            PrintState::default(),
        );

        assert_eq!(
            out,
            "3
    +v1*v2*v3
        *f(87238723,sda98as9d8,dsdjsdjsd,
            dskdjaskj,sdkjsdksd,djksdskdj
        )
        /(f(x,y,z,a,b,c,e,f,g,g(x),h(x,y,z))
            *g(x,y,z,a,b,c,e,f,g,g(x),h(x,y,z))
            *h(x,y,z,a,b,c,e,f,g,g(x),h(x,y,z)))
    +h(4)"
        );
    }

    #[test]
    fn atoms() {
        let a = parse!("f(x,y^2)^(x+z)/5+3");

        if AnsiWrap::<&str>::should_colorize() {
            assert_eq!(
                format!("{}", a.printer(PrintOptions::short())),
                "3\u{1b}[0;38;5;3m+\u{1b}[0m1/5*f\u{1b}[0;38;5;25m(\u{1b}[0mx,y^2\u{1b}[0;38;5;25m)\u{1b}[0m^\u{1b}[0;38;5;244m(\u{1b}[0mx+z\u{1b}[0;38;5;244m)\u{1b}[0m"
            );
        } else {
            assert_eq!(
                format!("{}", a.printer(PrintOptions::short())),
                "3+1/5*f(x,y^2)^(x+z)"
            );
        }

        assert_eq!(
            format!(
                "{}",
                AtomPrinter::new_with_options(a.as_view(), PrintOptions::latex())
            ),
            "3+\\frac{1}{5} f\\!\\left(x,y^{2}\\right)^{x+z}"
        );

        assert_eq!(
            format!(
                "{}",
                AtomPrinter::new_with_options(a.as_view(), PrintOptions::mathematica())
            ),
            "3+1/5 f[x,y^2]^(x+z)"
        );

        let a = parse!("8127389217 x^2");
        assert_eq!(
            format!(
                "{}",
                AtomPrinter::new_with_options(
                    a.as_view(),
                    PrintOptions {
                        number_thousands_separator: Some('_'),
                        multiplication_operator: ' ',
                        num_exp_as_superscript: true,
                        ..PrintOptions::file()
                    }
                )
            ),
            "812_738_921_7 symbolica::x²"
        );
    }

    #[test]
    fn polynomials() {
        let a = parse!("15 x^2").to_polynomial::<_, u8>(&Zp::new(17), None);

        let mut s = String::new();
        a.format(
            &PrintOptions {
                print_ring: true,
                symmetric_representation_for_finite_field: true,
                ..PrintOptions::file()
            },
            PrintState::new(),
            &mut s,
        )
        .unwrap();

        assert_eq!(s, "-2*x^2 % 17");
    }

    #[test]
    fn rational_polynomials() {
        let a = parse!("15 x^2 / (1+x)").to_rational_polynomial::<_, _, u8>(&Z, &Z, None);
        assert_eq!(format!("{a}"), "15*x^2/(1+x)");

        let a = parse!("(15 x^2 + 6) / (1+x)").to_rational_polynomial::<_, _, u8>(&Z, &Z, None);
        assert_eq!(format!("{a}"), "(6+15*x^2)/(1+x)");
    }

    #[test]
    fn factorized_rational_polynomials() {
        let a = parse!("15 x^2 / ((1+x)(x+2))")
            .to_factorized_rational_polynomial::<_, _, u8>(&Z, &Z, None);
        assert!(
            format!("{a}") == "15*x^2/((1+x)*(2+x))" || format!("{a}") == "15*x^2/((2+x)*(1+x))"
        );

        let a = parse!("(15 x^2 + 6) / ((1+x)(x+2))")
            .to_factorized_rational_polynomial::<_, _, u8>(&Z, &Z, None);
        assert!(
            format!("{a}") == "3*(2+5*x^2)/((1+x)*(2+x))"
                || format!("{a}") == "3*(2+5*x^2)/((2+x)*(1+x))"
        );

        let a = parse!("1/(v1*v2)").to_factorized_rational_polynomial::<_, _, u8>(&Z, &Z, None);
        assert!(format!("{a}") == "1/(v1*v2)" || format!("{a}") == "1/(v2*v1)");

        let a = parse!("-1/(2+v1)").to_factorized_rational_polynomial::<_, _, u8>(&Z, &Z, None);
        assert!(format!("{a}") == "-1/(2+v1)");
    }

    #[test]
    fn base_parentheses() {
        let a = parse!("-(1/2)^x+(-1)^(1+x)");
        assert_eq!(
            format!(
                "{}",
                AtomPrinter::new_with_options(a.as_view(), PrintOptions::file_no_namespace())
            ),
            "-(1/2)^x+(-1)^(1+x)"
        )
    }

    #[test]
    fn canon() {
        let _ = symbol!("canon_f"; Symmetric);
        let _ = symbol!("canon_y");
        let _ = symbol!("canon_x");

        let a = parse!(
            "canon_x^2 + 2*canon_x*canon_y + canon_y^2*(canon_x+canon_y) + canon_f(canon_x,canon_y)"
        );
        assert_eq!(
            a.to_canonical_string(),
            "(symbolica::{}::canon_x+symbolica::{}::canon_y)*symbolica::{}::canon_y^2+2*symbolica::{}::canon_x*symbolica::{}::canon_y+symbolica::{symmetric}::canon_f(symbolica::{}::canon_x,symbolica::{}::canon_y)+symbolica::{}::canon_x^2"
        );
    }

    #[test]
    fn canon_antisymmetric() {
        let (y, x) = symbol!(
            "symbolica::canon_antisymmetric::y",
            "symbolica::canon_antisymmetric::x"
        );
        let f = symbol!("symbolica::canon_antisymmetric::f"; Antisymmetric);
        let f_l = symbol!("symbolica::canon_antisymmetric::f_l"; Antisymmetric, Linear);
        let r1 = (function!(f, y, x) + 2).to_canonical_string();
        assert_eq!(
            r1,
            "-1*symbolica::canon_antisymmetric::{antisymmetric}::f(symbolica::canon_antisymmetric::{}::x,symbolica::canon_antisymmetric::{}::y)+2"
        );

        let r2 = (function!(f_l, function!(f, y, x), x) * 3).to_canonical_string();
        assert_eq!(
            r2,
            "-3*symbolica::canon_antisymmetric::{antisymmetric,linear}::f_l(symbolica::canon_antisymmetric::{antisymmetric}::f(symbolica::canon_antisymmetric::{}::x,symbolica::canon_antisymmetric::{}::y),symbolica::canon_antisymmetric::{}::x)"
        );
    }

    #[test]
    fn custom_print() {
        let _ = symbol!(
            "mu",
            print = |a, opt| {
                if !opt.mode.is_latex() {
                    return None; // use default printer
                }

                let mut fmt = String::new();
                fmt.push_str("\\mu");
                if let AtomView::Fun(f) = a {
                    fmt.push_str("_{");
                    let n_args = f.get_nargs();

                    for (i, a) in f.iter().enumerate() {
                        a.format(&mut fmt, opt, PrintState::new()).unwrap();
                        if i < n_args - 1 {
                            fmt.push(',');
                        }
                    }

                    fmt.push('}');
                }

                Some(fmt)
            }
        );

        let e = crate::parse!("mu^2 + mu(1) + mu(1,2)");
        let s = format!("{}", e.printer(PrintOptions::latex()));
        assert_eq!(s, "\\mu_{1}+\\mu_{1,2}+\\mu^{2}");
    }
}
