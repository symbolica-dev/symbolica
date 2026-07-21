use std::sync::OnceLock;

use super::*;

/// The Python-facing representation of one accepted integration transformation.
///
/// Symbolica owns this type so that optional integration implementations can expose
/// a common Python API without becoming a dependency of the Symbolica crate.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(from_py_object, name = "IntegrationStep", module = "symbolica.core")]
#[derive(Clone)]
pub struct PythonIntegrationStep {
    pub rule: Option<u16>,
    pub depth: usize,
    pub description: String,
    pub references: Vec<String>,
    pub source: String,
    pub input: Atom,
    pub output: Atom,
}

impl PythonIntegrationStep {
    /// Construct a Python integration step from an integration backend's data.
    pub fn new(
        rule: Option<u16>,
        depth: usize,
        description: String,
        references: Vec<String>,
        source: String,
        input: Atom,
        output: Atom,
    ) -> Self {
        Self {
            rule,
            depth,
            description,
            references,
            source,
            input,
            output,
        }
    }

    fn rule_label(&self) -> String {
        self.rule
            .map(|rule| format!("Rule {rule}"))
            .unwrap_or_else(|| "Transformation".to_owned())
    }

    fn format_equation(&self, options: &PrintOptions) -> String {
        format!(
            "{} = {}",
            self.input.format_string(options, PrintState::new()),
            self.output.format_string(options, PrintState::new())
        )
    }

    fn format_text(&self) -> String {
        let indent = "  ".repeat(self.depth);
        let nested_indent = format!("{indent}  ");
        let mut text = format!("{indent}{}", self.rule_label());

        if !self.description.is_empty() {
            text.push_str(": ");
            text.push_str(
                &self
                    .description
                    .replace('\n', &format!("\n{nested_indent}")),
            );
        }

        text.push('\n');
        text.push_str(&nested_indent);
        text.push_str(&self.format_equation(&DEFAULT_PRINT_OPTIONS));

        for reference in &self.references {
            text.push('\n');
            text.push_str(&nested_indent);
            text.push_str("Reference: ");
            text.push_str(reference);
        }

        text
    }
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonIntegrationStep {
    /// Rubi's downvalue number for the applied rule, if available.
    #[getter]
    pub fn rule(&self) -> Option<u16> {
        self.rule
    }

    /// Zero-based depth in the recursive integration tree.
    #[getter]
    pub fn depth(&self) -> usize {
        self.depth
    }

    /// A description of the transformation.
    #[getter]
    pub fn description(&self) -> &str {
        &self.description
    }

    /// Bibliographic references associated with the rule.
    #[getter]
    pub fn references(&self) -> Vec<String> {
        self.references.clone()
    }

    /// The original rule or pattern used by the integration backend.
    #[getter]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// The integrand to which the rule was applied.
    #[getter]
    pub fn input(&self) -> PythonExpression {
        self.input.clone().into()
    }

    /// The immediate result produced by the rule.
    #[getter]
    pub fn output(&self) -> PythonExpression {
        self.output.clone().into()
    }

    /// Return a concise, portable representation of the integration step.
    pub fn __repr__(&self) -> String {
        let rule = self
            .rule
            .map(|rule| rule.to_string())
            .unwrap_or_else(|| "None".to_owned());
        format!(
            "IntegrationStep(rule={rule}, depth={}, input={}, output={})",
            self.depth,
            self.input
                .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new()),
            self.output
                .format_string(&PLAIN_PRINT_OPTIONS, PrintState::new())
        )
    }

    /// Format the rule description, transformation and references.
    pub fn __str__(&self) -> String {
        self.format_text()
    }

    /// Render the integration step as HTML in notebook environments.
    pub fn _repr_html_(&self) -> String {
        let heading = if self.description.is_empty() {
            self.rule_label()
        } else {
            format!("{}: {}", self.rule_label(), self.description)
        };
        let heading = crate::printer::AnsiHtmlFormatter::escape_html(&heading);
        let equation = self.format_equation(
            &PrintOptions::new()
                .max_line_length(Some(80))
                .multiplication_operator('·')
                .num_exp_as_superscript(true)
                .color_mode(ColorMode::Always),
        );
        let equation = crate::printer::AnsiHtmlFormatter::new(&equation).to_string();

        let references = if self.references.is_empty() {
            String::new()
        } else {
            let references = self
                .references
                .iter()
                .map(|reference| crate::printer::AnsiHtmlFormatter::escape_html(reference))
                .collect::<Vec<_>>()
                .join("; ");
            format!("<div><small>References: {references}</small></div>")
        };

        format!(
            "<div class=\"symbolica-integration-step\" style=\"margin-left: {}em\">\
             <div><strong>{heading}</strong></div>{equation}{references}</div>",
            self.depth * 2
        )
    }

    /// Render the transformation as LaTeX in notebook environments.
    pub fn _repr_latex_(&self) -> String {
        format!(
            "$$\\begin{{aligned}} {} & = {} \\end{{aligned}}$$",
            self.input
                .format_string(&LATEX_PRINT_OPTIONS, PrintState::new()),
            self.output
                .format_string(&LATEX_PRINT_OPTIONS, PrintState::new())
        )
    }

    /// Render the integration step with IPython's pretty printer.
    pub fn _repr_pretty_(&self, pretty: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        let text = if cycle {
            "...".to_owned()
        } else {
            self.format_text()
        };
        pretty.call_method1("text", (text,))?;
        Ok(())
    }
}

pub type PythonIntegrateFn = fn(&Atom, Symbol) -> Result<Atom, Atom>;
pub type PythonIntegrateWithStepsFn =
    fn(&Atom, Symbol) -> (Result<Atom, Atom>, String, Vec<PythonIntegrationStep>);

/// Function pointers supplied by the integration implementation linked into the
/// final Python extension module.
#[derive(Clone, Copy)]
pub struct PythonIntegrationFunctions {
    pub integrate: PythonIntegrateFn,
    pub integrate_with_steps: PythonIntegrateWithStepsFn,
}

static PYTHON_INTEGRATION_FUNCTIONS: OnceLock<PythonIntegrationFunctions> = OnceLock::new();

/// Install the symbolic integration implementation used by the Python API.
#[allow(unused)]
pub fn set_python_integration_functions(
    functions: PythonIntegrationFunctions,
) -> Result<(), &'static str> {
    if PYTHON_INTEGRATION_FUNCTIONS.set(functions).is_ok() {
        return Ok(());
    }

    let installed = PYTHON_INTEGRATION_FUNCTIONS
        .get()
        .expect("the integration function cell was initialized by another caller");
    if std::ptr::fn_addr_eq(installed.integrate, functions.integrate)
        && std::ptr::fn_addr_eq(
            installed.integrate_with_steps,
            functions.integrate_with_steps,
        )
    {
        Ok(())
    } else {
        Err("different Python integration functions have already been installed")
    }
}

pub(crate) fn python_integration_functions() -> Option<&'static PythonIntegrationFunctions> {
    PYTHON_INTEGRATION_FUNCTIONS.get()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step() -> PythonIntegrationStep {
        PythonIntegrationStep::new(
            Some(42),
            1,
            "Rewrite <the integrand>".to_owned(),
            vec!["A&B".to_owned()],
            "Int[x, x]".to_owned(),
            Atom::num(1),
            Atom::num(2),
        )
    }

    #[test]
    fn integration_step_has_readable_text_representations() {
        let step = step();

        assert_eq!(
            step.__repr__(),
            "IntegrationStep(rule=42, depth=1, input=1, output=2)"
        );
        assert_eq!(
            step.__str__(),
            "  Rule 42: Rewrite <the integrand>\n    1 = 2\n    Reference: A&B"
        );
    }

    #[test]
    fn integration_step_has_safe_rich_representations() {
        let step = step();
        let html = step._repr_html_();

        assert!(html.contains("Rewrite &lt;the integrand&gt;"));
        assert!(html.contains("A&amp;B"));
        assert_eq!(
            step._repr_latex_(),
            "$$\\begin{aligned} 1 & = 2 \\end{aligned}$$"
        );
    }
}
