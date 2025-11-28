//! Methods for printing atoms and polynomials.

use std::fmt::{self, Error, Write};

use colored::Colorize;

use crate::{
    atom::{
        AddView, Atom, AtomView, FunctionBuilder, MulView, NumView, PowView, Symbol, VarView,
        representation::FunView,
    },
    coefficient::CoefficientView,
    domains::{SelfRing, finite_field::FiniteFieldCore, float::Complex, rational::Rational},
    state::State,
};

pub use numerica::printer::*;

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
                f.write_fmt(format_args!("{}", "+".yellow()))?;
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
        /// Input must be digits only.
        fn format_num<W: std::fmt::Write>(
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
                f.write_fmt(format_args!("{}", "-".yellow()))?;
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
                f.write_fmt(format_args!("{}", "+".yellow()))?;
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
                f.write_char('(')?;
            }

            if !opts.mode.is_latex()
                && (opts.number_thousands_separator.is_some() || print_state.superscript)
            {
                if !real.is_zero() {
                    if !global_negative && real.is_negative() {
                        f.write_char('-')?;
                    }

                    format_num(
                        real.numerator_ref().abs().to_string(),
                        opts,
                        &print_state,
                        f,
                    )?;
                    if !real.is_integer() {
                        f.write_char('/')?;
                        format_num(real.denominator_ref().to_string(), opts, &print_state, f)?;
                    }
                }

                if !real.is_zero() && !imag.is_zero() && !imag.is_negative() {
                    f.write_char('+')?;
                }

                if !imag.is_zero() {
                    if !global_negative && imag.is_negative() {
                        f.write_char('-')?;
                    }
                    format_num(
                        imag.numerator_ref().abs().to_string(),
                        opts,
                        &print_state,
                        f,
                    )?;
                    f.write_str(i_str)?;
                    if !imag.is_integer() {
                        f.write_char('/')?;
                        format_num(imag.denominator_ref().to_string(), opts, &print_state, f)?;
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
                f.write_char(')')?;
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
                // if i.is_zero() {
                //     r.to_float().format(opts, print_state, f)?;
                // } else {
                Complex::new(r.to_float(), i.to_float()).format(opts, print_state, f)?;
                //   }

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

            f.write_char('(')?;
            print_state.in_exp = false;
            print_state.in_exp_base = false;
        }

        print_state.in_product = true;

        // write the coefficient first
        let mut first = true;
        let mut skip_num = false;
        if let Some(AtomView::Num(n)) = self.iter().last() {
            print_state.suppress_one = true;
            first = n.fmt_output(f, opts, print_state)?;
            print_state.suppress_one = false;
            skip_num = true;
        } else if print_state.in_sum {
            if print_state.top_level_add_child
                && opts.mode.is_symbolica()
                && opts.color_top_level_sum
            {
                f.write_fmt(format_args!("{}", "+".yellow()))?;
            } else {
                f.write_char('+')?;
            }
        }

        print_state.top_level_add_child = false;
        print_state.level += 1;
        print_state.in_sum = false;

        for x in self.iter().take(if skip_num {
            self.get_nargs() - 1
        } else {
            self.get_nargs()
        }) {
            if !first {
                if opts.mode.is_latex() {
                    f.write_char(' ')?;
                } else {
                    f.write_char(opts.multiplication_operator)?;
                }
            }
            first = false;

            x.format(f, opts, print_state)?;
        }

        if add_paren {
            f.write_char(')')?;
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
                f.write_fmt(format_args!("{}", "+".yellow()))?;
            } else {
                f.write_char('+')?;
            }
        }

        let id = self.get_symbol();
        if let Some(custom_print) = &id.get_data().custom_print
            && let Some(s) = custom_print(self.as_view(), opts)
        {
            f.write_str(&s)?;
            return Ok(false);
        }

        id.format(opts, f)?;

        if opts.mode.is_latex() {
            f.write_str("\\!\\left(")?;
        } else if opts.square_brackets_for_function || opts.mode.is_mathematica() {
            f.write_char('[')?;
        } else {
            f.write_char('(')?;
        }

        print_state.top_level_add_child = false;
        print_state.level += 1;
        print_state.in_sum = false;
        print_state.in_product = false;
        print_state.in_exp = false;
        print_state.in_exp_base = false;
        print_state.suppress_one = false;
        let mut first = true;
        for x in self.iter() {
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
            first = false;

            x.format(f, opts, print_state)?;
        }

        if opts.mode.is_latex() {
            f.write_str("\\right)")?;
        } else if opts.square_brackets_for_function || opts.mode.is_mathematica() {
            f.write_char(']')?;
        } else {
            f.write_char(')')?;
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
                f.write_fmt(format_args!("{}", "+".yellow()))?;
            } else {
                f.write_char('+')?;
            }
        }

        let add_paren = print_state.in_exp_base; // right associative
        if add_paren {
            f.write_char('(')?;
            print_state.in_exp = false;
            print_state.in_exp_base = false;
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

        if add_paren {
            f.write_char(')')?;
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
        let mut first = true;
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
                    f.write_fmt(format_args!("{}", "+".yellow()))?;
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
                f.write_char('(')?;
            }
        }

        let mut count = 0;
        for x in self.iter() {
            if let Some(max_terms) = opts.max_terms
                && opts.mode.is_symbolica()
                && count >= max_terms
            {
                break;
            }

            if !first && print_state.top_level_add_child && opts.terms_on_new_line {
                f.write_char('\n')?;
            }
            first = false;

            x.format(f, opts, print_state)?;
            print_state.in_sum = true;
            count += 1;
        }

        if opts.max_terms.is_some() && count < self.get_nargs() {
            if print_state.top_level_add_child && opts.terms_on_new_line {
                f.write_char('\n')?;
            }
            if print_state.top_level_add_child
                && opts.mode.is_symbolica()
                && opts.color_top_level_sum
            {
                f.write_fmt(format_args!("{0}...", "+".yellow()))?;
            } else {
                f.write_str("+...")?;
            }
        }

        if add_paren {
            if opts.mode.is_latex() {
                f.write_str("\\right)")?;
            } else {
                f.write_char(')')?;
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
    use colored::control::ShouldColorize;

    use crate::{
        atom::{AtomCore, AtomView},
        domains::{SelfRing, finite_field::Zp, integer::Z},
        function, parse,
        printer::{AtomPrinter, PrintOptions, PrintState},
        symbol,
    };

    #[test]
    fn atoms() {
        let a = parse!("f(x,y^2)^(x+z)/5+3");

        if ShouldColorize::from_env().should_colorize() {
            assert_eq!(
                format!("{}", a.printer(PrintOptions::short())),
                "1/5*f(x,y^2)^(x+z)\u{1b}[33m+\u{1b}[0m3"
            );
        } else {
            assert_eq!(
                format!("{}", a.printer(PrintOptions::short())),
                "1/5*f(x,y^2)^(x+z)+3"
            );
        }

        assert_eq!(
            format!(
                "{}",
                AtomPrinter::new_with_options(a.as_view(), PrintOptions::latex())
            ),
            "\\frac{1}{5} f\\!\\left(x,y^{2}\\right)^{x+z}+3"
        );

        assert_eq!(
            format!(
                "{}",
                AtomPrinter::new_with_options(a.as_view(), PrintOptions::mathematica())
            ),
            "1/5 f[x,y^2]^(x+z)+3"
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
        let a = parse!("(-1)^(x+1)-(1/2)^x");
        assert_eq!(
            format!(
                "{}",
                AtomPrinter::new_with_options(a.as_view(), PrintOptions::file_no_namespace())
            ),
            "(-1)^(x+1)-(1/2)^x"
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
        assert_eq!(s, "\\mu^{2}+\\mu_{1}+\\mu_{1,2}");
    }
}
