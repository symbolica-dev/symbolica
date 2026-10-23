//! Typed Rust adapters for the shared notebook viewer.
use pyo3::prelude::*;
use {pyo3::exceptions::PyRuntimeError, std::sync::Mutex};

/// A numbered link to an omitted part of the source. Paths may contain native
/// focus handles; only providers enabling inline expansion need tree paths.
#[pyclass(frozen, get_all, from_py_object, eq, module = "symbolica")]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MathMLMarker {
    pub path: Vec<usize>,
    pub label: String,
    /// First omitted item offset, used for hover previews.
    pub start: usize,
}

impl MathMLMarker {
    pub fn new(path: Vec<usize>, label: impl Into<String>, start: usize) -> Self {
        Self {
            path,
            label: label.into(),
            start,
        }
    }

    /// Insert this at the marker's zero-based index in `MathMLPage::markers`.
    /// Operators and fences belong outside the returned MathML.
    pub fn marker(index: usize) -> String {
        format!(
            "<mtext data-symbolica-marker=\"{index}\">[{}]</mtext>",
            index as u128 + 1
        )
    }

    /// Construct the corresponding public Python object.
    pub fn into_python(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Py::new(py, self).map(Py::into_any)
    }
}

#[pymethods]
impl MathMLMarker {
    #[new]
    #[pyo3(signature = (path, label, start=0))]
    fn py_new(
        #[pyo3(from_py_with = super::paging::path)] path: Vec<usize>,
        label: String,
        #[pyo3(from_py_with = super::paging::index)] start: usize,
    ) -> Self {
        Self::new(path, label, start)
    }

    #[staticmethod]
    #[pyo3(name = "marker")]
    fn py_marker(index: &Bound<'_, PyAny>) -> PyResult<String> {
        super::paging::integer(index, "marker index must be a nonnegative integer")
            .map(Self::marker)
    }
}

/// A bounded MathML page. The range is half-open, and may be shorter than the
/// requested size when a rendering budget is reached. HTML must come from the
/// trusted renderer; no complete hidden expression should be serialized here.
#[pyclass(frozen, get_all, from_py_object, module = "symbolica")]
#[derive(Clone, Debug)]
pub struct MathMLPage {
    pub html: String,
    pub start: usize,
    pub end: usize,
    pub total: usize,
    pub kind: String,
    pub unit: String,
    pub markers: Vec<MathMLMarker>,
}

impl MathMLPage {
    pub fn new(html: impl Into<String>, start: usize, end: usize, total: usize) -> Self {
        Self {
            html: html.into(),
            start,
            end,
            total,
            kind: "Expression".into(),
            unit: "items".into(),
            markers: Vec::new(),
        }
    }

    /// Construct the public Python page. The shared viewer validates page ranges,
    /// marker metadata and transfer limits when it consumes this object.
    pub fn into_python(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        Py::new(py, self).map(Py::into_any)
    }
}

#[pymethods]
impl MathMLPage {
    #[new]
    #[pyo3(signature = (html, start, end, total, kind="Expression", unit="items", markers=None))]
    fn py_new(
        html: String,
        #[pyo3(from_py_with = super::paging::index)] start: usize,
        #[pyo3(from_py_with = super::paging::index)] end: usize,
        #[pyo3(from_py_with = super::paging::index)] total: usize,
        kind: &str,
        unit: &str,
        markers: Option<Vec<MathMLMarker>>,
    ) -> Self {
        Self {
            html,
            start,
            end,
            total,
            kind: kind.into(),
            unit: unit.into(),
            markers: markers.unwrap_or_default(),
        }
    }
}

/// One requested page. `path` is absolute; expansion paths are relative to it.
/// `start` is an item offset, not a native continuation token. Cursor-based
/// sources should retain their offset-to-cursor mapping in the source.
pub struct MathMLPageRequest<'a> {
    pub path: &'a [usize],
    pub start: usize,
    pub size: usize,
    pub layout: usize,
    pub expansions: &'a [(Vec<usize>, usize)],
}

/// A native source owns its expression, handles, compiler and cursor cache.
/// It renders only requested bounded pages. Python is available for renderers
/// that call a Python compiler; purely native renderers can ignore it.
pub trait MathMLPageSource: Send {
    fn page(&mut self, py: Python<'_>, request: MathMLPageRequest<'_>) -> PyResult<MathMLPage>;
}

#[pyclass]
struct NativePageSource {
    source: Mutex<Box<dyn MathMLPageSource>>,
}

#[pymethods]
impl NativePageSource {
    #[pyo3(signature = (path, start, size, layout=0, expansions=None))]
    fn page(
        &self,
        py: Python<'_>,
        path: Vec<usize>,
        start: usize,
        size: usize,
        layout: usize,
        expansions: Option<Vec<(Vec<usize>, usize)>>,
    ) -> PyResult<Py<PyAny>> {
        // Never block with the GIL held: a compiler may temporarily release it.
        let mut source = self
            .source
            .try_lock()
            .map_err(|_| PyRuntimeError::new_err("MathML page source is busy or unavailable"))?;
        source
            .page(
                py,
                MathMLPageRequest {
                    path: &path,
                    start,
                    size,
                    layout,
                    expansions: &expansions.unwrap_or_default(),
                },
            )?
            .into_python(py)
    }
}

/// Construct a notebook viewer directly from a Rust page source. Construction
/// is lazy: no pages or optional notebook dependencies are loaded until display.
/// The resulting Python viewer owns the source until `close()` or collection.
pub struct MathMLViewer {
    source: Box<dyn MathMLPageSource>,
    page_size: usize,
    layout_count: usize,
    supports_expansion: bool,
}

impl MathMLViewer {
    pub fn new(source: impl MathMLPageSource + 'static) -> Self {
        Self {
            source: Box::new(source),
            page_size: 25,
            layout_count: 1,
            supports_expansion: false,
        }
    }

    /// Set the requested page size (1..=500), checked by `into_python`.
    pub fn page_size(mut self, size: usize) -> Self {
        self.page_size = size;
        self
    }

    /// Supply 1..=8 bounded alternatives, from most detailed to most compact.
    pub fn layout_count(mut self, count: usize) -> Self {
        self.layout_count = count;
        self
    }

    /// Enable only if the source honors tree-relative expansion requests.
    pub fn supports_expansion(mut self, enabled: bool) -> Self {
        self.supports_expansion = enabled;
        self
    }

    pub fn into_python(self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let source = Py::new(
            py,
            NativePageSource {
                source: Mutex::new(self.source),
            },
        )?;
        super::mathml_viewer(
            py,
            source.into_any(),
            self.page_size,
            self.layout_count,
            self.supports_expansion,
        )
    }
}
