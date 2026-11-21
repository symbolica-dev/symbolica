//! Native algebraic containers use the same frontend and bounded protocol as atoms.
use super::*;
use crate::printer::mathml::{AlgebraicMathMl, MathMlPageOptions};

struct Source<T> {
    value: T,
    options: PrintOptions,
}

impl<T: AlgebraicMathMl + Send> MathMLPageSource for Source<T> {
    fn page(&mut self, _: Python<'_>, request: MathMLPageRequest<'_>) -> PyResult<MathMLPage> {
        let expansions = native_expansions(request.expansions);
        let page = self
            .value
            .page(
                &self.options,
                request.path,
                &MathMlPageOptions::new()
                    .start(request.start)
                    .limit(request.size)
                    .expansions(&expansions),
            )
            .ok_or_else(|| exceptions::PyValueError::new_err("Unknown algebraic subexpression"))?;
        let mut result = MathMLPage::new(page.html(), page.start, page.end, page.total);
        result.kind = self.value.kind().into();
        result.unit = "terms".into();
        result.markers = page
            .markers
            .into_iter()
            .map(|m| {
                let mut path = request.path.to_vec();
                path.extend(m.path);
                MathMLMarker::new(path, m.label, m.start)
            })
            .collect();
        Ok(result)
    }
}

pub(in crate::api::python) fn viewer<T: AlgebraicMathMl + Send + 'static>(
    py: Python<'_>,
    value: T,
    options: PrintOptions,
    size: usize,
    font_size: Option<f64>,
) -> PyResult<Py<PyAny>> {
    MathMLViewer::new(Source { value, options })
        .page_size(size)
        .supports_expansion(true)
        .font_size(font_size)
        .into_python(py)
}

/// The `formatted` output of an algebraic value: native MathML when
/// [`super::uses_mathml`], else highlighted text in the requested print mode.
pub(in crate::api::python) fn formatted_with_options<T: AlgebraicMathMl + Send + Sync + 'static>(
    value: T,
    options: PrintOptions,
    mathml: bool,
    size: Option<usize>,
    font_size: Option<f64>,
) -> PyResult<PythonFormattedOutput> {
    let size = super::page_size(size)?;
    let font_size = super::font_size(font_size)?;
    if super::uses_mathml(&options, mathml) {
        let options = PrintOptions {
            mode: PrintMode::MathML,
            ..options
        };
        return formatted(value, options, size, font_size);
    }
    Ok(super::text_output(&options, |options| {
        value.format_string(options, PrintState::new())
    }))
}

/// `size = None` selects the default preview of 25 terms.
pub(in crate::api::python) fn formatted<T: AlgebraicMathMl + Send + Sync + 'static>(
    value: T,
    options: PrintOptions,
    size: Option<usize>,
    font_size: Option<f64>,
) -> PyResult<PythonFormattedOutput> {
    let size = super::page_size(size)?.unwrap_or(25);
    let font_size = super::font_size(font_size)?;
    let source = value.clone();
    let source_options = options.clone();
    Ok(PythonFormattedOutput::lazy(move |_, backend| {
        if backend == FormattedOutputBackend::Latex {
            return Ok(None);
        }
        let page = value
            .page(&options, &[], &MathMlPageOptions::new().limit(size))
            .unwrap();
        Ok(Some(match backend {
            FormattedOutputBackend::Html => algebraic_fragment_html(&page.fragment(), font_size),
            FormattedOutputBackend::Text if page.is_partial() => format!(
                "{} preview ({}–{} of {}; use the MathML viewer to navigate)",
                value.kind(),
                page.start + 1,
                page.end,
                page.total
            ),
            FormattedOutputBackend::Text => value.format_string(
                &PrintOptions {
                    mode: PrintMode::Symbolica,
                    color_mode: ColorMode::Never,
                    ..options.clone()
                },
                PrintState::new(),
            ),
            FormattedOutputBackend::Latex => unreachable!(),
        }))
    })
    .with_notebook(move |py| viewer(py, source.clone(), source_options.clone(), size, font_size)))
}
