//! Keep matrix geometry in HTML and retain native, independently paged cells.
use super::*;
use crate::printer::mathml::AlgebraicMathMl;
use std::ops::Deref;

const HTML_BUDGET: usize = 32 * 1024;
const LAYOUT_COUNT: usize = 6;

enum Snapshot<'a> {
    Borrowed(&'a PythonMatrix),
    Shared(Arc<PythonMatrix>),
}

impl Deref for Snapshot<'_> {
    type Target = PythonMatrix;
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Borrowed(matrix) => matrix,
            Self::Shared(matrix) => matrix,
        }
    }
}

struct Source<'a> {
    matrix: Snapshot<'a>,
    cell_size: usize,
}

impl Source<'_> {
    fn entry(
        &self,
        index: usize,
    ) -> PyResult<&crate::domains::rational_polynomial::RationalPolynomial<IntegerRing, u16>> {
        let matrix = &self.matrix.matrix;
        if index >= matrix.nrows() * matrix.ncols() {
            return Err(exceptions::PyValueError::new_err("Unknown matrix entry"));
        }
        Ok(&matrix[(
            (index / matrix.ncols()) as u32,
            (index % matrix.ncols()) as u32,
        )])
    }

    fn matrix_page(
        &self,
        request: &MathMLPageRequest<'_>,
        static_output: bool,
    ) -> PyResult<MathMLPage> {
        let rows = self.matrix.matrix.nrows();
        let cols = self.matrix.matrix.ncols();
        let total = rows;
        if total == 0 || cols == 0 {
            let mut page =
                MathMLPage::new("<math><mrow><mo>[</mo><mo>]</mo></mrow></math>", 0, 0, 0);
            page.kind = format!("Matrix {rows}×{cols}");
            page.unit = "rows".into();
            return Ok(page);
        }
        let start = request.start.min(total - 1);
        let end = (start + request.size).min(total);
        let show_rows = start != 0 || end != rows;
        let mut markers = Vec::new();
        let mut body = String::new();
        let mut bytes = 0;
        for row in start..end {
            body.push_str("<tr>");
            if show_rows {
                body.push_str(&format!(
                    "<th scope=\"row\" style=\"font:12px system-ui;padding:.2em\">{}</th>",
                    row + 1
                ));
            }
            for col in 0..cols {
                let index = row * cols + col;
                body.push_str("<td style=\"padding:.4em .6em;vertical-align:middle;min-width:0\">");
                body.push_str(&format!("<div data-symbolica-cell=\"\" data-symbolica-entry=\"{index}\" style=\"min-width:0\">"));
                let expansions = request
                    .expansions
                    .iter()
                    .filter_map(|(path, count)| {
                        path.strip_prefix(&[index])
                            .map(|rest| (rest.to_vec(), *count))
                    })
                    .collect::<Vec<_>>();
                let compact_size = (self.cell_size >> request.layout).max(1);
                let size = expansions
                    .iter()
                    .filter(|(path, _)| path.is_empty())
                    .map(|(_, count)| *count)
                    .max()
                    .unwrap_or(compact_size)
                    .max(compact_size);
                let page = self
                    .entry(index)?
                    .page_expanded(&default_options(), &[], 0, size, &expansions)
                    .unwrap();
                // Small, complete values should not become omission buttons
                // just because earlier rows used the detailed-expression budget.
                // The shared provider still enforces the hard page-size limit.
                let detailed = !page.markers.is_empty() || page.html.len() > 1024;
                if (detailed && bytes + page.html.len() > HTML_BUDGET)
                    || markers.len() + page.markers.len() > 900
                {
                    let id = markers.len();
                    body.push_str(&format!("<math>{}</math>", MathMLMarker::marker(id)));
                    markers.push(MathMLMarker::new(
                        vec![index],
                        format!("Open entry ({}, {})", row + 1, col + 1),
                        0,
                    ));
                } else {
                    if detailed {
                        bytes += page.html.len();
                    }
                    let offset = markers.len();
                    let mut html = page.html;
                    // Descending replacement avoids renumbering an ID twice.
                    for id in (0..page.markers.len()).rev() {
                        html = html
                            .replace(
                                &format!("data-symbolica-marker=\"{id}\""),
                                &format!("data-symbolica-marker=\"{}\"", offset + id),
                            )
                            .replace(
                                &format!(">[{}]</mtext>", id + 1),
                                &format!(">[{}]</mtext>", offset + id + 1),
                            );
                    }
                    if static_output {
                        html = super::html::wrapping_mathml(&html).unwrap_or(html);
                    }
                    body.push_str(&html);
                    for marker in page.markers {
                        let mut path = vec![index];
                        path.extend(marker.path);
                        markers.push(MathMLMarker::new(
                            path,
                            format!("Entry ({}, {}): {}", row + 1, col + 1, marker.label),
                            marker.start,
                        ));
                    }
                }
                body.push_str("</div></td>");
            }
            body.push_str("</tr>");
        }
        if end < total {
            let id = markers.len();
            let span = cols + usize::from(show_rows);
            body.push_str(&format!(
                "<tr data-symbolica-matrix-omission=\"\"><td colspan=\"{span}\" style=\"padding:.5em .6em;text-align:center\"><math>{}</math></td></tr>",
                MathMLMarker::marker(id)
            ));
            markers.push(MathMLMarker::new(vec![], "Open remaining matrix rows", end));
        }
        // CSS draws square brackets around HTML cells. Each cell keeps a
        // MathML root, so the shared frontend can wrap its sums portably.
        let edge = "position:absolute;top:0;bottom:0;width:.35em;border-top:1.5px solid currentColor;border-bottom:1.5px solid currentColor";
        let html = format!(
            "<div aria-label=\"Matrix with {rows} rows and {cols} columns\" style=\"width:fit-content;max-width:var(--symbolica-wrap-width,100%);min-width:min-content\"><div style=\"position:relative;padding:0 .35em;min-width:0\"><span aria-hidden=\"true\" style=\"{edge};left:0;border-left:1.5px solid currentColor\"></span><table style=\"border-collapse:collapse;table-layout:auto;width:100%;color:inherit\">{body}</table><span aria-hidden=\"true\" style=\"{edge};right:0;border-right:1.5px solid currentColor\"></span></div></div>"
        );
        let mut result = MathMLPage::new(html, start, end, total);
        result.kind = format!("Matrix {rows}×{cols}");
        result.unit = "rows".into();
        result.markers = markers;
        Ok(result)
    }
}

impl MathMLPageSource for Source<'static> {
    fn page(&mut self, _: Python<'_>, request: MathMLPageRequest<'_>) -> PyResult<MathMLPage> {
        let Some((&entry, path)) = request.path.split_first() else {
            return self.matrix_page(&request, false);
        };
        let value = self.entry(entry)?;
        let size = if path.is_empty() && !value.denominator.is_one() {
            (request.size >> request.layout).max(1)
        } else {
            request.size
        };
        let page = value
            .page_expanded(
                &default_options(),
                path,
                request.start,
                size,
                request.expansions,
            )
            .ok_or_else(|| exceptions::PyValueError::new_err("Unknown matrix subexpression"))?;
        let mut result = MathMLPage::new(page.html, page.start, page.end, page.total);
        let cols = self.matrix.matrix.ncols();
        result.kind = format!("Entry ({}, {})", entry / cols + 1, entry % cols + 1);
        result.unit = "terms".into();
        result.markers = page
            .markers
            .into_iter()
            .map(|marker| {
                let mut path = request.path.to_vec();
                path.extend(marker.path);
                MathMLMarker::new(path, marker.label, marker.start)
            })
            .collect();
        Ok(result)
    }
}

pub(in crate::api::python) fn viewer(
    py: Python<'_>,
    matrix: PythonMatrix,
    size: usize,
) -> PyResult<Py<PyAny>> {
    shared_viewer(py, Arc::new(matrix), size)
}

fn shared_viewer(py: Python<'_>, matrix: Arc<PythonMatrix>, size: usize) -> PyResult<Py<PyAny>> {
    MathMLViewer::new(Source {
        matrix: Snapshot::Shared(matrix),
        cell_size: size,
    })
    .page_size(size)
    .layout_count(LAYOUT_COUNT)
    .supports_expansion(true)
    .into_python(py)
}

pub(in crate::api::python) fn static_html(matrix: &PythonMatrix) -> PyResult<String> {
    let source = Source {
        matrix: Snapshot::Borrowed(matrix),
        cell_size: 25,
    };
    let page = source.matrix_page(
        &MathMLPageRequest {
            path: &[],
            start: 0,
            size: 25,
            layout: 0,
            expansions: &[],
        },
        true,
    )?;
    let mut html = page.html;
    for id in 0..page.markers.len() {
        html = html.replace(&MathMLMarker::marker(id), "<mtext>⋯</mtext>");
    }
    Ok(html)
}

pub(in crate::api::python) fn formatted(matrix: PythonMatrix) -> PyResult<PythonFormattedOutput> {
    let matrix = Arc::new(matrix);
    let notebook = Arc::clone(&matrix);
    Ok(PythonFormattedOutput::lazy(move |_, backend| {
        Ok(match backend {
            FormattedOutputBackend::Html => Some(static_html(&matrix)?),
            FormattedOutputBackend::Text => Some(format!(
                "Matrix {}×{} (MathML preview)",
                matrix.matrix.nrows(),
                matrix.matrix.ncols()
            )),
            FormattedOutputBackend::Latex => None,
        })
    })
    .with_notebook(move |py| shared_viewer(py, Arc::clone(&notebook), 25)))
}
