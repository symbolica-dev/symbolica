//! Methods for printing rings.

use std::borrow::Cow;

use ahash::HashMap;

/// The overall print mode.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
#[non_exhaustive]
#[derive(Default)]
pub enum PrintMode {
    #[default]
    Symbolica,
    Latex,
    Mathematica,
    Sympy,
    Typst,
}

impl PrintMode {
    pub fn is_symbolica(&self) -> bool {
        *self == PrintMode::Symbolica
    }

    pub fn is_latex(&self) -> bool {
        *self == PrintMode::Latex
    }

    pub fn is_mathematica(&self) -> bool {
        *self == PrintMode::Mathematica
    }

    pub fn is_sympy(&self) -> bool {
        *self == PrintMode::Sympy
    }

    pub fn is_typst(&self) -> bool {
        *self == PrintMode::Typst
    }
}

/// Controls when print output should contain color/style escape codes.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Default)]
pub enum ColorMode {
    /// Colorize only when the output target supports it.
    #[default]
    Auto,
    /// Always emit color/style escape codes.
    Always,
    /// Never emit color/style escape codes.
    Never,
}

/// Represents user-defined data that can be used as a key in [PrintUserData].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum PrintUserDataKey {
    /// A small integer value.
    Integer(i64),
    /// A string value.
    String(String),
}

/// Represents user-defined data that can be attached to [PrintOptions].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PrintUserData {
    /// A small integer value.
    Integer(i64),
    /// A string value.
    String(String),
    /// A list of extended symbol data.
    List(Vec<PrintUserData>),
    /// A map from extended symbol data to extended symbol data.
    Map(HashMap<PrintUserDataKey, PrintUserData>),
}

/// Various options for printing expressions.
#[derive(Debug, Clone)]
pub struct PrintOptions {
    /// The overall print mode.
    pub mode: PrintMode,
    /// The maximum line length before splitting into multiple lines. If None, no splitting is done.
    pub max_line_length: Option<usize>,
    /// The number of spaces to use for indentation.
    pub indentation: usize,
    /// Whether to fill indented lines with as many operator arguments as possible, or whether
    /// to put each operator argument on a new line.
    pub fill_indented_lines: bool,
    /// Whether to put each term of a top-level sum on a new line.
    pub terms_on_new_line: bool,
    /// Whether color/style escape codes should be emitted.
    pub color_mode: ColorMode,
    /// Whether to color the top-level `+` and `-`.
    pub color_top_level_sum: bool,
    /// Whether to color built-in symbols.
    pub color_builtin_symbols: bool,
    /// Colors for different bracket levels.
    pub bracket_level_colors: Option<[u8; 16]>,
    /// Whether to print the ring.
    pub print_ring: bool,
    /// Whether to use a symmetric representation for finite fields.
    pub symmetric_representation_for_finite_field: bool,
    /// Whether to print rational polynomials in an explicit way or in a Symbolica optimized way.
    pub explicit_rational_polynomial: bool,
    /// The character to use as a thousands separator in numbers. If None, no separator is used.
    pub number_thousands_separator: Option<char>,
    /// The character to use for multiplication.
    pub multiplication_operator: char,
    /// Whether to use `**` for exponentiation (e.g. for sympy) instead of `^`.
    pub double_star_for_exponentiation: bool,
    /// The open and close brackets to use for function application.
    pub function_brackets: (char, char),
    /// Whether to print the exponent of numbers as a superscript.
    pub num_exp_as_superscript: bool,
    /// The precision of floating point numbers to print. If None, the available precision is used.
    pub precision: Option<usize>,
    /// Whether to print matrices in a tabular format (e.g. with newlines and indentation).
    pub pretty_matrix: bool,
    /// The namespace to hide when printing. If None, no namespace is hidden.
    pub hide_namespace: Option<Cow<'static, str>>,
    /// Whether to hide all namespaces when printing.
    pub hide_all_namespaces: bool,
    /// Print attribute and tags
    pub include_attributes: bool,
    /// Whether to color namespaces.
    pub color_namespace: bool,
    /// The maximum number of terms to print for a top-level sum.
    pub max_terms: Option<usize>,
    /// Provides a handle to set the behavior of the custom print function.
    /// Symbolica does not use this option for its own printing.
    pub custom_print_mode: HashMap<String, PrintUserData>,
}

impl PrintOptions {
    pub fn new() -> Self {
        Self {
            max_line_length: None,
            indentation: 4,
            fill_indented_lines: true,
            terms_on_new_line: false,
            color_mode: ColorMode::Auto,
            color_top_level_sum: true,
            color_builtin_symbols: true,
            bracket_level_colors: Some([
                244, 25, 97, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60,
            ]),
            print_ring: true,
            symmetric_representation_for_finite_field: false,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: '*',
            double_star_for_exponentiation: false,
            function_brackets: ('(', ')'),
            num_exp_as_superscript: false,
            mode: PrintMode::Symbolica,
            precision: None,
            pretty_matrix: false,
            hide_namespace: None,
            hide_all_namespaces: true,
            include_attributes: false,
            color_namespace: true,
            max_terms: None,
            custom_print_mode: HashMap::default(),
        }
    }

    /// Print the output in a Mathematica-readable format.
    pub fn mathematica() -> PrintOptions {
        Self {
            max_line_length: None,
            indentation: 4,
            fill_indented_lines: true,
            terms_on_new_line: false,
            color_mode: ColorMode::Never,
            color_top_level_sum: false,
            color_builtin_symbols: false,
            print_ring: true,
            symmetric_representation_for_finite_field: false,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: ' ',
            double_star_for_exponentiation: false,
            function_brackets: ('[', ']'),
            num_exp_as_superscript: false,
            mode: PrintMode::Mathematica,
            precision: None,
            pretty_matrix: false,
            hide_namespace: Some(Cow::Borrowed("symbolica")),
            hide_all_namespaces: false,
            include_attributes: false,
            color_namespace: false,
            max_terms: None,
            bracket_level_colors: None,
            custom_print_mode: HashMap::default(),
        }
    }

    /// Print the output in a Latex input format.
    pub fn latex() -> PrintOptions {
        Self {
            max_line_length: None,
            indentation: 4,
            fill_indented_lines: true,
            terms_on_new_line: false,
            color_mode: ColorMode::Never,
            color_top_level_sum: false,
            color_builtin_symbols: false,
            print_ring: true,
            symmetric_representation_for_finite_field: false,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: ' ',
            double_star_for_exponentiation: false,
            function_brackets: ('(', ')'),
            num_exp_as_superscript: false,
            mode: PrintMode::Latex,
            precision: None,
            pretty_matrix: false,
            hide_namespace: None,
            hide_all_namespaces: true,
            include_attributes: false,
            color_namespace: false,
            max_terms: None,
            bracket_level_colors: None,
            custom_print_mode: HashMap::default(),
        }
    }

    /// Print the output in a Typst-readable format.
    pub fn typst() -> PrintOptions {
        Self {
            max_line_length: None,
            indentation: 4,
            fill_indented_lines: true,
            terms_on_new_line: false,
            color_mode: ColorMode::Never,
            color_top_level_sum: false,
            color_builtin_symbols: false,
            print_ring: true,
            symmetric_representation_for_finite_field: false,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: ' ',
            double_star_for_exponentiation: false,
            function_brackets: ('(', ')'),
            num_exp_as_superscript: false,
            mode: PrintMode::Typst,
            precision: None,
            pretty_matrix: false,
            hide_namespace: None,
            hide_all_namespaces: true,
            include_attributes: false,
            color_namespace: false,
            max_terms: None,
            bracket_level_colors: None,
            custom_print_mode: HashMap::default(),
        }
    }

    /// Print the output suitable for a file.
    pub fn file() -> PrintOptions {
        Self {
            max_line_length: None,
            indentation: 4,
            fill_indented_lines: true,
            terms_on_new_line: false,
            color_mode: ColorMode::Never,
            color_top_level_sum: false,
            color_builtin_symbols: false,
            print_ring: false,
            symmetric_representation_for_finite_field: false,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: '*',
            double_star_for_exponentiation: false,
            function_brackets: ('(', ')'),
            num_exp_as_superscript: false,
            mode: PrintMode::Symbolica,
            precision: None,
            pretty_matrix: false,
            hide_namespace: None,
            hide_all_namespaces: false,
            include_attributes: false,
            color_namespace: false,
            max_terms: None,
            bracket_level_colors: None,
            custom_print_mode: HashMap::default(),
        }
    }

    /// Print the output suitable for a file without namespaces.
    pub fn file_no_namespace() -> PrintOptions {
        Self {
            hide_all_namespaces: true,
            ..Self::file()
        }
    }

    /// Print the output suitable for a file with namespaces
    /// and attributes and tags.
    pub fn full() -> PrintOptions {
        Self {
            include_attributes: true,
            ..Self::file()
        }
    }

    /// Print the output with namespaces suppressed.
    pub fn short() -> PrintOptions {
        Self {
            hide_all_namespaces: true,
            ..Self::new()
        }
    }

    /// Print the output in a sympy input format.
    pub fn sympy() -> PrintOptions {
        Self {
            double_star_for_exponentiation: true,
            mode: PrintMode::Sympy,
            ..Self::file()
        }
    }

    pub fn from_fmt(f: &std::fmt::Formatter) -> PrintOptions {
        PrintOptions {
            precision: f.precision(),
            hide_all_namespaces: !f.alternate(),
            max_line_length: f.width(),
            terms_on_new_line: f.align() == Some(std::fmt::Alignment::Right),
            ..Default::default()
        }
    }

    pub fn update_with_fmt(mut self, f: &std::fmt::Formatter) -> Self {
        self.precision = f.precision();

        if f.alternate() {
            self.hide_all_namespaces = false;
        }

        if let Some(width) = f.width() {
            self.max_line_length = Some(width);
        }

        if let Some(a) = f.align() {
            self.terms_on_new_line = a == std::fmt::Alignment::Right;
        }
        self
    }

    /// Set the overall print mode.
    pub const fn mode(mut self, mode: PrintMode) -> Self {
        self.mode = mode;
        self
    }

    /// Set the maximum line length before splitting into multiple lines.
    pub const fn max_line_length(mut self, max_line_length: Option<usize>) -> Self {
        self.max_line_length = max_line_length;
        self
    }

    /// Set the number of spaces to use for indentation.
    pub const fn indentation(mut self, indentation: usize) -> Self {
        self.indentation = indentation;
        self
    }

    /// Set whether indented lines should be filled with as many operator arguments as possible.
    pub const fn fill_indented_lines(mut self, fill_indented_lines: bool) -> Self {
        self.fill_indented_lines = fill_indented_lines;
        self
    }

    /// Set whether each term of a top-level sum should be put on a new line.
    pub const fn terms_on_new_line(mut self, terms_on_new_line: bool) -> Self {
        self.terms_on_new_line = terms_on_new_line;
        self
    }

    /// Set when color/style escape codes should be emitted.
    pub const fn color_mode(mut self, color_mode: ColorMode) -> Self {
        self.color_mode = color_mode;
        self
    }

    /// Set whether the top-level `+` and `-` should be colorized.
    pub const fn color_top_level_sum(mut self, color_top_level_sum: bool) -> Self {
        self.color_top_level_sum = color_top_level_sum;
        self
    }

    /// Set whether built-in symbols should be colorized.
    pub const fn color_builtin_symbols(mut self, color_builtin_symbols: bool) -> Self {
        self.color_builtin_symbols = color_builtin_symbols;
        self
    }

    /// Set the colors for successive bracket levels.
    pub const fn bracket_level_colors(mut self, bracket_level_colors: Option<[u8; 16]>) -> Self {
        self.bracket_level_colors = bracket_level_colors;
        self
    }

    /// Set whether the ring should be printed.
    pub const fn print_ring(mut self, print_ring: bool) -> Self {
        self.print_ring = print_ring;
        self
    }

    /// Set whether finite fields should use a symmetric representation.
    pub const fn symmetric_representation_for_finite_field(
        mut self,
        symmetric_representation_for_finite_field: bool,
    ) -> Self {
        self.symmetric_representation_for_finite_field = symmetric_representation_for_finite_field;
        self
    }

    /// Set whether rational polynomials should be printed explicitly.
    pub const fn explicit_rational_polynomial(
        mut self,
        explicit_rational_polynomial: bool,
    ) -> Self {
        self.explicit_rational_polynomial = explicit_rational_polynomial;
        self
    }

    /// Set the thousands separator for numbers.
    pub const fn number_thousands_separator(
        mut self,
        number_thousands_separator: Option<char>,
    ) -> Self {
        self.number_thousands_separator = number_thousands_separator;
        self
    }

    /// Set the multiplication operator character.
    pub const fn multiplication_operator(mut self, multiplication_operator: char) -> Self {
        self.multiplication_operator = multiplication_operator;
        self
    }

    /// Set whether exponentiation should be printed as `**` instead of `^`.
    pub const fn double_star_for_exponentiation(
        mut self,
        double_star_for_exponentiation: bool,
    ) -> Self {
        self.double_star_for_exponentiation = double_star_for_exponentiation;
        self
    }

    /// Set the open and close brackets used for function application.
    pub const fn function_brackets(mut self, function_brackets: (char, char)) -> Self {
        self.function_brackets = function_brackets;
        self
    }

    /// Set whether numeric exponents should be printed as superscripts.
    pub const fn num_exp_as_superscript(mut self, num_exp_as_superscript: bool) -> Self {
        self.num_exp_as_superscript = num_exp_as_superscript;
        self
    }

    /// Set the precision used when printing floating point numbers.
    pub const fn precision(mut self, precision: Option<usize>) -> Self {
        self.precision = precision;
        self
    }

    /// Set whether matrices should be printed in a tabular format.
    pub const fn pretty_matrix(mut self, pretty_matrix: bool) -> Self {
        self.pretty_matrix = pretty_matrix;
        self
    }

    /// Set the namespace to hide.
    pub fn hide_namespace<T: Into<Cow<'static, str>>>(mut self, namespace: T) -> Self {
        self.hide_namespace = Some(namespace.into());
        self
    }

    /// Set whether all namespaces should be hidden.
    pub const fn hide_all_namespaces(mut self, hide_all_namespaces: bool) -> Self {
        self.hide_all_namespaces = hide_all_namespaces;
        self
    }

    /// Set whether attributes and tags should be printed.
    pub const fn include_attributes(mut self, include_attributes: bool) -> Self {
        self.include_attributes = include_attributes;
        self
    }

    /// Set whether namespaces should be colorized.
    pub const fn color_namespace(mut self, color_namespace: bool) -> Self {
        self.color_namespace = color_namespace;
        self
    }

    /// Set the maximum number of terms to print for a top-level sum.
    pub const fn max_terms(mut self, max_terms: Option<usize>) -> Self {
        self.max_terms = max_terms;
        self
    }
}

impl Default for PrintOptions {
    fn default() -> Self {
        Self::new()
    }
}

/// The current state useful for printing. These
/// settings will control, for example, if parentheses are needed
/// (e.g., a sum in a product),
/// and if 1 should be suppressed (e.g. in a product).
#[derive(Debug, Copy, Clone)]
pub struct PrintState {
    pub in_sum: bool,
    pub in_product: bool,
    pub suppress_one: bool,
    pub in_exp: bool,
    pub in_exp_base: bool,
    pub top_level_add_child: bool,
    pub superscript: bool,
    pub level: u16,
    pub bracket_level: u16,
    pub indentation_level: u16,
}

impl Default for PrintState {
    fn default() -> Self {
        Self::new()
    }
}

impl PrintState {
    pub const fn new() -> PrintState {
        Self {
            in_sum: false,
            in_product: false,
            in_exp: false,
            in_exp_base: false,
            suppress_one: false,
            top_level_add_child: true,
            superscript: false,
            level: 0,
            bracket_level: 0,
            indentation_level: 0,
        }
    }

    pub fn from_fmt(f: &std::fmt::Formatter) -> PrintState {
        PrintState {
            in_sum: f.sign_plus(),
            ..Default::default()
        }
    }

    pub fn update_with_fmt(mut self, f: &std::fmt::Formatter) -> Self {
        self.in_sum = f.sign_plus();
        self
    }

    pub fn step(self, in_sum: bool, in_product: bool, in_exp: bool, in_exp_base: bool) -> Self {
        Self {
            in_sum,
            in_product,
            in_exp,
            in_exp_base,
            level: self.level + 1,
            bracket_level: self.bracket_level,
            indentation_level: self.indentation_level,
            ..self
        }
    }
}
