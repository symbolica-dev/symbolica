//! Citation display and bibliography export for Python.

use super::*;

/// A scientific reference with explanations of its relevance to a computation.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(get_all, frozen, skip_from_py_object, module = "symbolica.core")]
#[derive(Clone, Debug)]
pub struct Citation {
    /// A stable identity for the citation, preferably a DOI or arXiv ID.
    pub id: String,
    /// A human-readable bibliographic reference.
    pub reference: String,
    /// A ready-to-export BibTeX entry.
    pub bibtex: String,
    /// The reasons for including this citation, optionally formatted as Markdown.
    pub reasons: Vec<String>,
    /// A description of the citation, optionally formatted as Markdown.
    pub description: String,
    /// The relevance score of this citation, if known.
    pub relevance: Option<usize>,
}

// Bibliographic references and identifiers are plain text, even in Markdown reports.
fn escape_markdown(text: &str) -> String {
    let mut escaped = String::new();
    for c in text.chars() {
        if c.is_ascii_punctuation() {
            escaped.push('\\');
        }
        escaped.push(c);
    }
    escaped
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[pymethods]
impl Citation {
    /// Create a citation. Reasons and description may contain Markdown.
    /// Relevance is a library-defined score; None means unknown, not zero.
    #[new]
    #[pyo3(signature = (id, reference, bibtex, *, reasons = Vec::new(), description = String::new(), relevance = None))]
    pub fn new(
        id: String,
        reference: String,
        bibtex: String,
        reasons: Vec<String>,
        description: String,
        relevance: Option<usize>,
    ) -> Self {
        Self {
            id,
            reference,
            bibtex,
            reasons,
            description,
            relevance,
        }
    }

    /// Display the reference, identifier, description, reasons and known relevance.
    /// The BibTeX entry is available separately through to_bibtex().
    pub fn __str__(&self) -> String {
        let mut sections = vec![self.reference.clone(), format!("ID: {}", self.id)];
        if !self.description.is_empty() {
            sections.push(self.description.clone());
        }
        if !self.reasons.is_empty() {
            sections.push(format!(
                "Reasons:\n{}",
                self.reasons
                    .iter()
                    .map(|reason| format!("- {}", reason.replace('\n', "\n  ")))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
        if let Some(relevance) = self.relevance {
            sections.push(format!("Relevance: {relevance}"));
        }
        sections.join("\n\n")
    }

    /// Return a compact representation suitable for lists of citations.
    pub fn __repr__(&self) -> String {
        let relevance = self
            .relevance
            .map_or_else(|| "None".to_owned(), |value| value.to_string());
        format!("Citation(id={:?}, relevance={relevance})", self.id)
    }

    /// Return a Markdown report, optionally including a fenced BibTeX entry.
    /// Reasons and description are preserved as Markdown.
    #[pyo3(signature = (include_bibtex = false))]
    pub fn to_markdown(&self, include_bibtex: bool) -> String {
        let mut sections = vec![
            escape_markdown(&self.reference),
            format!("**ID:** {}", escape_markdown(&self.id)),
        ];
        if !self.description.is_empty() {
            sections.push(self.description.clone());
        }
        if !self.reasons.is_empty() {
            sections.push(format!(
                "**Reasons:**\n\n{}",
                self.reasons
                    .iter()
                    .map(|reason| format!("- {}", reason.replace('\n', "\n  ")))
                    .collect::<Vec<_>>()
                    .join("\n")
            ));
        }
        if let Some(relevance) = self.relevance {
            sections.push(format!("**Relevance:** {relevance}"));
        }
        if include_bibtex {
            // A longer fence keeps backticks inside an entry from closing its code block.
            let fence_len = self
                .bibtex
                .split(|c| c != '`')
                .map(str::len)
                .max()
                .unwrap_or(0)
                .saturating_add(1)
                .max(3);
            let fence = "`".repeat(fence_len);
            sections.push(format!("{fence}bibtex\n{}\n{fence}", self.bibtex));
        }
        sections.join("\n\n")
    }

    /// Return the original BibTeX entry unchanged, ready to write to a .bib file.
    pub fn to_bibtex(&self) -> String {
        self.bibtex.clone()
    }

    /// Display a Markdown report in notebooks.
    pub fn _repr_markdown_(&self) -> String {
        self.to_markdown(false)
    }

    /// Display a formatted HTML report in notebooks.
    /// Description and reasons are escaped text with line breaks preserved;
    /// use to_markdown() to export their Markdown formatting.
    pub fn _repr_html_(&self) -> String {
        let escape = crate::printer::AnsiHtmlFormatter::escape_html;
        let mut html = format!(
            "<div class=\"symbolica-citation\">\
             <p style=\"white-space: pre-wrap\"><strong>{}</strong></p>\
             <p><strong>ID:</strong> <code>{}</code></p>",
            escape(&self.reference),
            escape(&self.id),
        );
        if !self.description.is_empty() {
            html.push_str(&format!(
                "<p style=\"white-space: pre-wrap\">{}</p>",
                escape(&self.description),
            ));
        }
        if !self.reasons.is_empty() {
            html.push_str("<p><strong>Reasons:</strong></p><ul>");
            for reason in &self.reasons {
                html.push_str(&format!(
                    "<li style=\"white-space: pre-wrap\">{}</li>",
                    escape(reason),
                ));
            }
            html.push_str("</ul>");
        }
        if let Some(relevance) = self.relevance {
            html.push_str(&format!("<p><strong>Relevance:</strong> {relevance}</p>"));
        }
        html.push_str("</div>");
        html
    }

    /// Display a readable report in IPython's pretty printer.
    pub fn _repr_pretty_(&self, pretty: &Bound<'_, PyAny>, cycle: bool) -> PyResult<()> {
        let text = if cycle {
            "Citation(...)".to_owned()
        } else {
            self.__str__()
        };
        pretty.call_method1("text", (text,))?;
        Ok(())
    }
}
