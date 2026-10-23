//! Native algebraic containers use the same frontend and bounded protocol as atoms.
use super::*;
use crate::printer::mathml::AlgebraicMathMl;

struct Source<T> {
    value: T,
    options: PrintOptions,
}

impl<T: AlgebraicMathMl + Send> MathMLPageSource for Source<T> {
    fn page(&mut self, _: Python<'_>, request: MathMLPageRequest<'_>) -> PyResult<MathMLPage> {
        let page = self
            .value
            .page_expanded(
                &self.options,
                request.path,
                request.start,
                request.size,
                request.expansions,
            )
            .ok_or_else(|| exceptions::PyValueError::new_err("Unknown algebraic subexpression"))?;
        let mut result = MathMLPage::new(page.html, page.start, page.end, page.total);
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
) -> PyResult<Py<PyAny>> {
    MathMLViewer::new(Source { value, options })
        .page_size(size)
        .supports_expansion(true)
        .into_python(py)
}

pub(in crate::api::python) fn formatted<T: AlgebraicMathMl + Send + Sync + 'static>(
    value: T,
    options: PrintOptions,
    size: usize,
) -> PyResult<PythonFormattedOutput> {
    if !(1..=500).contains(&size) {
        return Err(exceptions::PyValueError::new_err(
            "page_size must be between 1 and 500",
        ));
    }
    let source = value.clone();
    let source_options = options.clone();
    Ok(PythonFormattedOutput::lazy(move |_, backend| {
        if backend == FormattedOutputBackend::Latex {
            return Ok(None);
        }
        let page = value.page(&options, &[], 0, size).unwrap();
        Ok(Some(match backend {
            FormattedOutputBackend::Html => algebraic_fragment_html(&page.fragment()),
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
    .with_notebook(move |py| viewer(py, source.clone(), source_options.clone(), size)))
}
