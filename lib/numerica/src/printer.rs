//! Methods for printing rings.

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
}

/// Various options for printing expressions.
#[derive(Debug, Copy, Clone)]
pub struct PrintOptions {
    pub mode: PrintMode,
    pub terms_on_new_line: bool,
    pub color_top_level_sum: bool,
    pub color_builtin_symbols: bool,
    pub print_ring: bool,
    pub symmetric_representation_for_finite_field: bool,
    pub explicit_rational_polynomial: bool,
    pub number_thousands_separator: Option<char>,
    pub multiplication_operator: char,
    pub double_star_for_exponentiation: bool,
    pub square_brackets_for_function: bool,
    pub num_exp_as_superscript: bool,
    pub print_precision: bool,
    pub precision: Option<usize>,
    pub pretty_matrix: bool,
    pub hide_namespace: Option<&'static str>,
    pub hide_all_namespaces: bool,
    /// Print attribute and tags
    pub include_attributes: bool,
    pub color_namespace: bool,
    pub max_terms: Option<usize>,
    /// Provides a handle to set the behavior of the custom print function.
    /// Symbolica does not use this option for its own printing.
    pub custom_print_mode: Option<(&'static str, usize)>,
}

impl PrintOptions {
    pub const fn new() -> Self {
        Self {
            terms_on_new_line: false,
            color_top_level_sum: true,
            color_builtin_symbols: true,
            print_ring: true,
            symmetric_representation_for_finite_field: false,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: '*',
            double_star_for_exponentiation: false,
            square_brackets_for_function: false,
            num_exp_as_superscript: false,
            mode: PrintMode::Symbolica,
            print_precision: false,
            precision: None,
            pretty_matrix: false,
            hide_namespace: None,
            hide_all_namespaces: true,
            include_attributes: false,
            color_namespace: true,
            max_terms: None,
            custom_print_mode: None,
        }
    }

    /// Print the output in a Mathematica-readable format.
    pub const fn mathematica() -> PrintOptions {
        Self {
            terms_on_new_line: false,
            color_top_level_sum: false,
            color_builtin_symbols: false,
            print_ring: true,
            symmetric_representation_for_finite_field: false,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: ' ',
            double_star_for_exponentiation: false,
            square_brackets_for_function: true,
            num_exp_as_superscript: false,
            mode: PrintMode::Mathematica,
            print_precision: true,
            precision: None,
            pretty_matrix: false,
            hide_namespace: Some("symbolica"),
            hide_all_namespaces: false,
            include_attributes: false,
            color_namespace: false,
            max_terms: None,
            custom_print_mode: None,
        }
    }

    /// Print the output in a Latex input format.
    pub const fn latex() -> PrintOptions {
        Self {
            terms_on_new_line: false,
            color_top_level_sum: false,
            color_builtin_symbols: false,
            print_ring: true,
            symmetric_representation_for_finite_field: false,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: ' ',
            double_star_for_exponentiation: false,
            square_brackets_for_function: false,
            num_exp_as_superscript: false,
            mode: PrintMode::Latex,
            print_precision: false,
            precision: None,
            pretty_matrix: false,
            hide_namespace: None,
            hide_all_namespaces: true,
            include_attributes: false,
            color_namespace: false,
            max_terms: None,
            custom_print_mode: None,
        }
    }

    /// Print the output suitable for a file.
    pub const fn file() -> PrintOptions {
        Self {
            terms_on_new_line: false,
            color_top_level_sum: false,
            color_builtin_symbols: false,
            print_ring: false,
            symmetric_representation_for_finite_field: false,
            explicit_rational_polynomial: false,
            number_thousands_separator: None,
            multiplication_operator: '*',
            double_star_for_exponentiation: false,
            square_brackets_for_function: false,
            num_exp_as_superscript: false,
            mode: PrintMode::Symbolica,
            print_precision: true,
            precision: None,
            pretty_matrix: false,
            hide_namespace: None,
            hide_all_namespaces: false,
            include_attributes: false,
            color_namespace: false,
            max_terms: None,
            custom_print_mode: None,
        }
    }

    /// Print the output suitable for a file without namespaces.
    pub const fn file_no_namespace() -> PrintOptions {
        Self {
            hide_all_namespaces: true,
            ..Self::file()
        }
    }

    /// Print the output suitable for a file with namespaces
    /// and attributes and tags.
    pub const fn full() -> PrintOptions {
        Self {
            include_attributes: true,
            ..Self::file()
        }
    }

    /// Print the output with namespaces suppressed.
    pub const fn short() -> PrintOptions {
        Self {
            hide_all_namespaces: true,
            ..Self::new()
        }
    }

    /// Print the output in a sympy input format.
    pub const fn sympy() -> PrintOptions {
        Self {
            double_star_for_exponentiation: true,
            ..Self::file()
        }
    }

    pub fn from_fmt(f: &std::fmt::Formatter) -> PrintOptions {
        PrintOptions {
            precision: f.precision(),
            hide_all_namespaces: !f.alternate(),
            terms_on_new_line: f.align() == Some(std::fmt::Alignment::Right),
            ..Default::default()
        }
    }

    pub fn update_with_fmt(mut self, f: &std::fmt::Formatter) -> Self {
        self.precision = f.precision();

        if f.alternate() {
            self.hide_all_namespaces = false;
        }

        if let Some(a) = f.align() {
            self.terms_on_new_line = a == std::fmt::Alignment::Right;
        }
        self
    }

    pub const fn hide_namespace(mut self, namespace: &'static str) -> Self {
        self.hide_namespace = Some(namespace);
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
            ..self
        }
    }
}
