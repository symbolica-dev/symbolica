//! Kernel-side paging, bounded caches and validation for the shared MathML UI.
use super::{MathMLMarker, MathMLPage};
use pyo3::{
    PyTraverseError, PyVisit,
    exceptions::{PyImportError, PyRuntimeError, PyTypeError, PyValueError},
    prelude::*,
    types::{PyAnyMethods, PyBool, PyDict, PyDictMethods, PyInt, PyList, PyTuple},
};
use std::collections::{BTreeMap, HashSet, VecDeque};

fn invalid(message: &str) -> PyErr {
    PyValueError::new_err(message.to_owned())
}

fn page_size(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    integer(value, "page_size must be between 1 and 500")
}

fn layout_count(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    integer(value, "layout_count must be between 1 and 8")
}

pub(super) fn integer(value: &Bound<'_, PyAny>, message: &str) -> PyResult<usize> {
    if !value.is_exact_instance_of::<PyInt>() || value.is_instance_of::<PyBool>() {
        return Err(invalid(message));
    }
    value.extract().map_err(|_| invalid(message))
}

fn field<'py>(dict: &Bound<'py, PyDict>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    dict.get_item(key)?
        .ok_or_else(|| invalid(&format!("Missing page field: {key}")))
}

pub(super) fn path(value: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    if !value.is_instance_of::<PyList>() && !value.is_instance_of::<PyTuple>() {
        return Err(invalid("omission paths must contain nonnegative integers"));
    }
    value
        .try_iter()?
        .map(|item| integer(&item?, "omission paths must contain nonnegative integers"))
        .collect()
}

pub(super) fn index(value: &Bound<'_, PyAny>) -> PyResult<usize> {
    integer(value, "index must be a nonnegative integer")
}

fn size(message: &Bound<'_, PyDict>, key: &str, default: usize) -> PyResult<usize> {
    let value = message
        .get_item(key)?
        .map(|v| integer(&v, "page_size must be between 1 and 500"))
        .transpose()?
        .unwrap_or(default);
    if !(1..=500).contains(&value) {
        return Err(invalid("page_size must be between 1 and 500"));
    }
    Ok(value)
}

#[derive(Clone)]
struct Page {
    value: MathMLPage,
    ids: Vec<usize>,
    layout: usize,
}

impl Page {
    fn from_python(value: &Bound<'_, PyAny>, size: usize, layout: usize) -> PyResult<Self> {
        let (page, ids) = if let Ok(page) = value.extract::<MathMLPage>() {
            let ids = (0..page.markers.len()).collect();
            (page, ids)
        } else {
            let dict = value.cast::<PyDict>().map_err(|_| {
                PyTypeError::new_err("source.page must return MathMLPage or a page dictionary")
            })?;
            let markers = if let Some(items) = dict.get_item("markers")? {
                if items.len()? > 1000 {
                    return Err(invalid("invalid omission metadata"));
                }
                let mut markers = Vec::new();
                for item in items.try_iter()? {
                    let item = item?;
                    if item.len()? != 2 {
                        return Err(invalid("invalid omission metadata"));
                    }
                    markers.push(MathMLMarker::new(
                        path(&item.get_item(0)?)?,
                        item.get_item(1)?.extract::<String>()?,
                        0,
                    ));
                }
                markers
            } else {
                Vec::new()
            };
            let mut page = MathMLPage {
                html: field(dict, "html")?
                    .extract()
                    .map_err(|_| invalid("page MathML must be a string of at most 256 KiB"))?,
                start: integer(&field(dict, "start")?, "invalid page range")?,
                end: integer(&field(dict, "end")?, "invalid page range")?,
                total: integer(&field(dict, "total")?, "invalid page range")?,
                kind: dict
                    .get_item("kind")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_else(|| "Expression".into()),
                unit: dict
                    .get_item("unit")?
                    .map(|v| v.extract())
                    .transpose()?
                    .unwrap_or_else(|| "items".into()),
                markers,
            };
            let indices = |key, default: Vec<usize>| -> PyResult<Vec<usize>> {
                match dict.get_item(key)? {
                    Some(items) => items
                        .try_iter()?
                        .map(|item| {
                            integer(
                                &item?,
                                "omission ids and offsets must be nonnegative integers",
                            )
                        })
                        .collect(),
                    None => Ok(default),
                }
            };
            let ids = indices("marker_ids", (0..page.markers.len()).collect())?;
            let starts = indices("marker_starts", vec![0; page.markers.len()])?;
            if starts.len() != page.markers.len() {
                return Err(invalid("invalid omission metadata"));
            }
            for (marker, start) in page.markers.iter_mut().zip(starts) {
                marker.start = start;
            }
            (page, ids)
        };
        if page.html.len() > 256 * 1024 {
            return Err(invalid("page MathML must be a string of at most 256 KiB"));
        }
        if !((page.start == 0 && page.end == 0 && page.total == 0)
            || (page.start < page.end && page.end <= page.total))
        {
            return Err(invalid(
                "invalid page range: require 0 <= start < end <= total (or an empty page)",
            ));
        }
        if page.end - page.start > size {
            return Err(invalid("provider exceeded the requested page size"));
        }
        if page.markers.len() > 1000
            || ids.len() != page.markers.len()
            || ids.iter().collect::<HashSet<_>>().len() != ids.len()
        {
            return Err(invalid("invalid omission metadata"));
        }
        let result = Self {
            value: page,
            ids,
            layout,
        };
        // Serialize only whitelisted fields; private provider trees never enter widget state.
        let py = value.py();
        let encoded: String = py
            .import("json")?
            .call_method1("dumps", (result.to_dict(py)?,))?
            .extract()?;
        if encoded.len() > 512 * 1024 {
            return Err(invalid(
                "page and omission metadata exceed the 512 KiB transfer budget",
            ));
        }
        Ok(result)
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new(py);
        let page = &self.value;
        dict.set_item("html", &page.html)?;
        dict.set_item("start", page.start)?;
        dict.set_item("end", page.end)?;
        dict.set_item("total", page.total)?;
        dict.set_item("next", (page.end < page.total).then_some(page.end))?;
        dict.set_item("kind", &page.kind)?;
        dict.set_item("unit", &page.unit)?;
        dict.set_item("layout", self.layout)?;
        dict.set_item(
            "markers",
            page.markers
                .iter()
                .map(|m| (m.path.clone(), m.label.clone()))
                .collect::<Vec<_>>(),
        )?;
        dict.set_item("marker_ids", &self.ids)?;
        dict.set_item(
            "marker_starts",
            page.markers.iter().map(|m| m.start).collect::<Vec<_>>(),
        )?;
        Ok(dict)
    }

    fn capability(&self) -> Capability {
        Capability {
            markers: self.value.markers.clone(),
            layout: self.layout,
            next: (self.value.end < self.value.total).then_some(self.value.end),
        }
    }
}

#[derive(Clone)]
struct Pages {
    main: Page,
    layouts: Vec<Page>,
}
impl Pages {
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let dict = self.main.to_dict(py)?;
        dict.set_item(
            "layouts",
            self.layouts
                .iter()
                .map(|p| p.to_dict(py))
                .collect::<PyResult<Vec<_>>>()?,
        )?;
        Ok(dict)
    }
}

#[derive(Clone)]
struct Capability {
    markers: Vec<MathMLMarker>,
    layout: usize,
    next: Option<usize>,
}
struct Popup {
    main: Capability,
    layouts: Vec<Capability>,
    origin: Option<String>,
}

#[derive(Clone, Default, PartialEq, Eq)]
struct Position {
    path: Vec<usize>,
    start: usize,
    previous: Vec<usize>,
    size: usize,
    compact_size: usize,
}
#[derive(Clone)]
struct Navigation {
    position: Position,
    parents: Vec<Position>,
    expanded: BTreeMap<Vec<usize>, usize>,
}
#[derive(PartialEq, Eq)]
struct CacheKey {
    path: Vec<usize>,
    start: usize,
    size: usize,
    expanded: BTreeMap<Vec<usize>, usize>,
}

/// Paging and notebook integration share one native owner.
#[pyclass(subclass, name = "MathMLViewer", module = "symbolica")]
pub(super) struct PythonMathMLViewer {
    source: Option<Py<PyAny>>,
    layout_count: usize,
    supports_expansion: bool,
    navigation: Navigation,
    cache: VecDeque<(CacheKey, Pages)>,
    popups: VecDeque<(String, Popup)>,
    serial: u64,
    views: HashSet<String>,
    last_page: Option<Py<PyDict>>,
    widget: Option<Py<PyAny>>,
}

impl PythonMathMLViewer {
    fn source_page(
        &self,
        py: Python<'_>,
        path: &[usize],
        start: usize,
        size: usize,
        layout: usize,
        expansions: &[(Vec<usize>, usize)],
    ) -> PyResult<Page> {
        let source = self
            .source
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err("This expression viewer is closed"))?;
        let value = source.call_method1(
            py,
            "page",
            (path.to_vec(), start, size, layout, expansions.to_vec()),
        )?;
        Page::from_python(value.bind(py), size, layout)
    }

    fn pages(
        &self,
        py: Python<'_>,
        path: &[usize],
        start: usize,
        size: usize,
        expansions: &[(Vec<usize>, usize)],
    ) -> PyResult<Pages> {
        let main = self.source_page(py, path, start, size, 0, expansions)?;
        let mut layouts = Vec::new();
        let mut previous_html = main.value.html.clone();
        for layout in 1..self.layout_count {
            let candidate = self.source_page(py, path, start, size, layout, expansions)?;
            if candidate.value.html != previous_html {
                previous_html = candidate.value.html.clone();
                layouts.push(candidate);
            }
        }
        Ok(Pages { main, layouts })
    }

    fn current(&mut self, py: Python<'_>) -> PyResult<Pages> {
        if self.source.is_none() {
            return Err(PyRuntimeError::new_err("This expression viewer is closed"));
        }
        let p = &self.navigation.position;
        let key = CacheKey {
            path: p.path.clone(),
            start: p.start,
            size: p.size,
            expanded: self.navigation.expanded.clone(),
        };
        let pages = if let Some(index) = self.cache.iter().position(|(k, _)| k == &key) {
            self.cache.remove(index).unwrap().1
        } else {
            let expansions = self.expansions();
            self.pages(py, &p.path, p.start, p.size, &expansions)?
        };
        self.cache.push_back((key, pages.clone()));
        while self.cache.len() > 3 {
            self.cache.pop_front();
        }
        Ok(pages)
    }

    fn expansions(&self) -> Vec<(Vec<usize>, usize)> {
        let prefix = &self.navigation.position.path;
        self.navigation
            .expanded
            .iter()
            .filter_map(|(path, count)| {
                (path.len() > prefix.len() && path.starts_with(prefix))
                    .then(|| (path[prefix.len()..].to_vec(), *count))
            })
            .collect()
    }

    fn can_collapse(&self) -> bool {
        let p = &self.navigation.position;
        self.supports_expansion
            && (p.size > p.compact_size
                || self
                    .navigation
                    .expanded
                    .keys()
                    .any(|path| path.len() > p.path.len() && path.starts_with(&p.path)))
    }

    fn context(message: &Bound<'_, PyDict>) -> PyResult<Option<String>> {
        message
            .get_item("context")?
            .filter(|v| !v.is_none())
            .map(|v| v.extract().map_err(|_| invalid("Unknown or expired popup")))
            .transpose()
    }

    fn selected(&mut self, py: Python<'_>, message: &Bound<'_, PyDict>) -> PyResult<Capability> {
        let layout = message
            .get_item("layout")?
            .map(|v| integer(&v, "Unknown layout"))
            .transpose()?
            .unwrap_or(0);
        if let Some(context) = Self::context(message)? {
            let index = self
                .popups
                .iter()
                .position(|(name, _)| name == &context)
                .ok_or_else(|| invalid("Unknown or expired popup"))?;
            let popup = &self.popups[index].1;
            let selected = if layout == 0 {
                Some(&popup.main)
            } else {
                popup.layouts.iter().find(|p| p.layout == layout)
            }
            .cloned()
            .ok_or_else(|| invalid("Unknown layout"))?;
            // Keep every active ancestor live while paging inside a nested popup.
            let mut current = Some(context);
            let mut visited = HashSet::new();
            while let Some(context) = current {
                if !visited.insert(context.clone()) {
                    break;
                }
                let Some(index) = self.popups.iter().position(|(name, _)| name == &context) else {
                    break;
                };
                let entry = self.popups.remove(index).unwrap();
                current = entry.1.origin.clone();
                self.popups.push_back(entry);
            }
            Ok(selected)
        } else {
            let pages = self.current(py)?;
            let selected = if layout == 0 {
                Some(&pages.main)
            } else {
                pages.layouts.iter().find(|p| p.layout == layout)
            }
            .ok_or_else(|| invalid("Unknown layout"))?;
            Ok(selected.capability())
        }
    }

    fn target(page: &Capability, message: &Bound<'_, PyDict>) -> PyResult<MathMLMarker> {
        let requested = path(&field(message, "value")?)?;
        page.markers
            .iter()
            .find(|m| m.path == requested)
            .cloned()
            .ok_or_else(|| invalid("Unknown subexpression"))
    }

    fn action(&mut self, py: Python<'_>, message: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
        let page = self.selected(py, message)?;
        let action: Option<String> = message
            .get_item("action")?
            .map(|v| v.extract())
            .transpose()?;
        match action.as_deref() {
            Some("next") => {
                if let Some(next) = page.next {
                    let p = &mut self.navigation.position;
                    p.previous.push(p.start);
                    p.start = next;
                }
            }
            Some("previous") => {
                if let Some(previous) = self.navigation.position.previous.pop() {
                    self.navigation.position.start = previous;
                }
            }
            Some("size") => {
                let size = size(message, "value", self.navigation.position.size)?;
                let p = &mut self.navigation.position;
                p.size = size;
                p.compact_size = size;
                p.start = 0;
                p.previous.clear();
            }
            Some("expand") => {
                if !self.supports_expansion {
                    return Err(invalid("this source does not support in-place expansion"));
                }
                let target = Self::target(&page, message)?;
                let size = size(message, "size", self.navigation.position.size)?;
                if target.path == self.navigation.position.path {
                    self.navigation.position.size = 500.min(self.navigation.position.size + size);
                } else {
                    let count = self.navigation.expanded.entry(target.path).or_default();
                    *count = 500.min((*count).max(target.start).saturating_add(size));
                }
            }
            Some("collapse") => {
                if !self.supports_expansion {
                    return Err(invalid("this source does not support in-place expansion"));
                }
                let p = &mut self.navigation.position;
                p.size = p.compact_size;
                p.start = 0;
                p.previous.clear();
                self.navigation
                    .expanded
                    .retain(|path, _| path.len() <= p.path.len() || !path.starts_with(&p.path));
            }
            Some("open") => {
                let target = Self::target(&page, message)?;
                let size = size(message, "size", self.navigation.position.size)?;
                self.navigation
                    .parents
                    .push(self.navigation.position.clone());
                self.navigation.position = Position {
                    path: target.path,
                    size,
                    compact_size: size,
                    ..Position::default()
                };
            }
            Some("back") => {
                if let Some(parent) = self.navigation.parents.pop() {
                    self.navigation.position = parent;
                }
            }
            _ => {}
        }
        self._page(py)
    }

    fn preview(&mut self, py: Python<'_>, message: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
        let page = self.selected(py, message)?;
        let target = Self::target(&page, message)?;
        let popup = message
            .get_item("popup")?
            .map(|v| v.is_truthy())
            .transpose()?
            .unwrap_or(false);
        let (start, size) = if popup {
            let start = message
                .get_item("start")?
                .map(|v| integer(&v, "Invalid preview offset"))
                .transpose()?
                .unwrap_or(0);
            if start as u128 >= 1u128 << 63 {
                return Err(invalid("Invalid preview offset"));
            }
            (start, size(message, "size", self.navigation.position.size)?)
        } else {
            (target.start, self.navigation.position.size.min(5))
        };
        let pages = self.pages(py, &target.path, start, size, &[])?;
        let dict = pages.to_dict(py)?;
        let p = &self.navigation.position;
        let shown = if target.path == p.path {
            p.size
        } else {
            target
                .start
                .max(*self.navigation.expanded.get(&target.path).unwrap_or(&0))
        };
        dict.set_item(
            "can_expand",
            self.supports_expansion
                && shown < 500
                && target.path.len().saturating_sub(p.path.len()) < 64,
        )?;
        self.serial += 1;
        let context = format!("preview-{}", self.serial);
        dict.set_item("context", &context)?;
        // Only omission capabilities are retained, never popup HTML or hidden trees.
        self.popups.push_back((
            context,
            Popup {
                main: pages.main.capability(),
                layouts: pages.layouts.iter().map(Page::capability).collect(),
                origin: Self::context(message)?,
            },
        ));
        while self.popups.len() > 32 {
            self.popups.pop_front();
        }
        Ok(dict.unbind())
    }

    fn clear_pages(&mut self) {
        self.cache.clear();
        self.popups.clear();
        self.last_page = None;
    }

    fn widget_type(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        let base = py.import("anywidget")?.getattr("AnyWidget")?;
        let traits = py.import("traitlets")?;
        let module = super::paging_module(py)?;
        if let Ok(class) = module.getattr("_MathPages") {
            return Ok(class);
        }
        // Use the widget's Python metaclass through PyO3; no Python source is evaluated.
        let attributes = PyDict::new(py);
        attributes.set_item("__module__", "_symbolica_paging")?;
        attributes.set_item("_esm", include_str!("paging.js"))?;
        let metadata = PyDict::new(py);
        metadata.set_item("sync", true)?;
        for name in ["page", "preview"] {
            let field = traits
                .getattr("Dict")?
                .call0()?
                .call_method("tag", (), Some(&metadata))?;
            attributes.set_item(name, field)?;
        }
        let class =
            py.import("builtins")?
                .getattr("type")?
                .call1(("MathPages", (base,), attributes))?;
        module.setattr("_MathPages", &class)?;
        Ok(class)
    }

    fn close_widget(slf: &Py<Self>, py: Python<'_>) -> PyResult<()> {
        let widget = slf.borrow_mut(py).widget.take();
        if let Some(widget) = widget {
            let kwargs = PyDict::new(py);
            kwargs.set_item("remove", true)?;
            // Release the Rust borrow before calling framework code or callbacks.
            widget.call_method(
                py,
                "on_msg",
                (slf.getattr(py, "_on_message")?,),
                Some(&kwargs),
            )?;
            widget.call_method0(py, "close")?;
        }
        Ok(())
    }
}

#[pymethods]
impl PythonMathMLViewer {
    #[new]
    #[pyo3(signature = (source, page_size=25, *, layout_count=1, supports_expansion=false))]
    fn new(
        py: Python<'_>,
        source: Py<PyAny>,
        #[pyo3(from_py_with=page_size)] page_size: usize,
        #[pyo3(from_py_with=layout_count)] layout_count: usize,
        supports_expansion: bool,
    ) -> PyResult<Self> {
        if !source
            .bind(py)
            .getattr("page")
            .map(|p| p.is_callable())
            .unwrap_or(false)
        {
            return Err(PyTypeError::new_err(
                "source must implement page(path, start, size, layout, expansions)",
            ));
        }
        if !(1..=500).contains(&page_size) {
            return Err(invalid("page_size must be between 1 and 500"));
        }
        if !(1..=8).contains(&layout_count) {
            return Err(invalid("layout_count must be between 1 and 8"));
        }
        Ok(Self {
            source: Some(source),
            layout_count,
            supports_expansion,
            navigation: Navigation {
                position: Position {
                    size: page_size,
                    compact_size: page_size,
                    ..Position::default()
                },
                parents: Vec::new(),
                expanded: BTreeMap::new(),
            },
            cache: VecDeque::new(),
            popups: VecDeque::new(),
            serial: 0,
            views: HashSet::new(),
            last_page: None,
            widget: None,
        })
    }

    #[getter]
    fn _widget(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.widget.as_ref().map(|widget| widget.clone_ref(py))
    }

    fn _get_widget(slf: Py<Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        if let Some(widget) = &slf.borrow(py).widget {
            return Ok(widget.clone_ref(py));
        }
        let class = Self::widget_type(py)?;
        let page = slf.borrow_mut(py)._page(py)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("page", page)?;
        let widget = class.call((), Some(&kwargs))?.unbind();
        widget.call_method1(py, "on_msg", (slf.getattr(py, "_on_message")?,))?;
        slf.borrow_mut(py).widget = Some(widget.clone_ref(py));
        Ok(widget)
    }

    fn _on_message(
        slf: Py<Self>,
        py: Python<'_>,
        widget: &Bound<'_, PyAny>,
        content: &Bound<'_, PyDict>,
        _buffers: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let response = slf.borrow_mut(py)._message(py, content)?;
        if let Some((field, value)) = response {
            if field.is_empty() {
                Self::close_widget(&slf, py)?;
            } else {
                widget.setattr(field, value)?;
            }
        }
        Ok(())
    }

    /// Release the retained source, widget and bounded caches.
    fn close(slf: Py<Self>, py: Python<'_>) -> PyResult<()> {
        Self::close_widget(&slf, py)?;
        slf.borrow_mut(py)._close();
        Ok(())
    }

    /// Return the bounded initial page and responsive layout alternatives.
    /// This is ordinary Python data suitable for JSON transport to a frontend.
    fn page(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        self._page(py)
    }

    /// Process a frontend request without a widget connection.
    /// Returns (field, payload), or None when no update is needed. An empty
    /// field acknowledges disposal. Pass responses to the JavaScript mount.
    fn handle_message(
        &mut self,
        py: Python<'_>,
        message: &Bound<'_, PyDict>,
    ) -> PyResult<Option<(String, Py<PyDict>)>> {
        self._message(py, message)
    }

    /// Return the pager ES module, including its dependency-free mount() API.
    /// Existing notebook adapters use the same module's default render export.
    #[staticmethod]
    fn javascript() -> &'static str {
        include_str!("paging.js")
    }

    #[pyo3(signature = (include=None, exclude=None))]
    fn _repr_mimebundle_(
        slf: Py<Self>,
        py: Python<'_>,
        include: Option<Vec<String>>,
        exclude: Option<Vec<String>>,
    ) -> PyResult<Py<PyDict>> {
        let wanted = |mime: &str| {
            include.as_ref().is_none_or(|v| v.iter().any(|s| s == mime))
                && !exclude
                    .as_ref()
                    .is_some_and(|v| v.iter().any(|s| s == mime))
        };
        let result = PyDict::new(py);
        if wanted("text/plain") {
            result.set_item("text/plain", slf.borrow(py).__repr__())?;
        }
        if wanted("text/html") {
            result.set_item("text/html", slf.borrow_mut(py)._repr_html_(py)?)?;
        }
        if wanted("application/vnd.jupyter.widget-view+json") {
            let bundle = (|| -> PyResult<Py<PyAny>> {
                let widget = Self::_get_widget(slf.clone_ref(py), py)?;
                let kwargs = PyDict::new(py);
                kwargs.set_item("include", &include)?;
                kwargs.set_item("exclude", &exclude)?;
                widget.call_method(py, "_repr_mimebundle_", (), Some(&kwargs))
            })();
            match bundle {
                Ok(bundle) if !bundle.is_none(py) => {
                    let value = bundle.bind(py);
                    let data = if value.is_instance_of::<PyTuple>() {
                        value.get_item(0)?
                    } else {
                        value.clone()
                    };
                    for (key, value) in data.cast::<PyDict>()?.iter() {
                        if wanted(&key.extract::<String>()?) {
                            result.set_item(key, value)?;
                        }
                    }
                    if wanted("text/plain") {
                        result.set_item("text/plain", slf.borrow(py).__repr__())?;
                    }
                }
                Ok(_) => {}
                Err(error) if error.is_instance_of::<PyImportError>(py) => {}
                Err(error) => return Err(error),
            }
        }
        Ok(result.unbind())
    }

    fn _display_(slf: Py<Self>, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let marimo = py.import("marimo")?;
        let result = (|| -> PyResult<Py<PyAny>> {
            let widget = Self::_get_widget(slf.clone_ref(py), py)?;
            Ok(marimo
                .getattr("ui")?
                .getattr("anywidget")?
                .call1((widget,))?
                .unbind())
        })();
        match result {
            Err(error) if error.is_instance_of::<PyImportError>(py) => {
                let html = slf.borrow_mut(py)._repr_html_(py)?;
                Ok(marimo.getattr("Html")?.call1((html,))?.unbind())
            }
            other => other,
        }
    }

    fn _page(&mut self, py: Python<'_>) -> PyResult<Py<PyDict>> {
        let dict = self.current(py)?.to_dict(py)?;
        dict.set_item("previous", !self.navigation.position.previous.is_empty())?;
        dict.set_item("back", !self.navigation.parents.is_empty())?;
        dict.set_item("page_size", self.navigation.position.size)?;
        dict.set_item("supports_expansion", self.supports_expansion)?;
        dict.set_item("can_collapse", self.can_collapse())?;
        self.last_page = Some(dict.clone().unbind());
        Ok(dict.unbind())
    }

    fn _render_page(&self, py: Python<'_>, layout: usize) -> PyResult<Py<PyDict>> {
        let p = &self.navigation.position;
        self.source_page(py, &p.path, p.start, p.size, layout, &self.expansions())?
            .to_dict(py)
            .map(Bound::unbind)
    }

    fn _selected_page(
        &mut self,
        py: Python<'_>,
        message: &Bound<'_, PyDict>,
    ) -> PyResult<Py<PyDict>> {
        let layout = self.selected(py, message)?.layout;
        let pages = self.current(py)?;
        if layout == 0 {
            return self._page(py);
        }
        let page = pages
            .layouts
            .iter()
            .find(|p| p.layout == layout)
            .ok_or_else(|| invalid("Unknown layout"))?;
        page.to_dict(py).map(Bound::unbind)
    }

    fn _action(&mut self, py: Python<'_>, message: &Bound<'_, PyDict>) -> PyResult<Py<PyDict>> {
        let state = self.navigation.clone();
        let result = self.action(py, message);
        if result.is_err() {
            self.navigation = state;
        }
        result
    }

    /// Return (trait name, payload); an empty trait name requests widget disposal.
    fn _message(
        &mut self,
        py: Python<'_>,
        message: &Bound<'_, PyDict>,
    ) -> PyResult<Option<(String, Py<PyDict>)>> {
        let action: Option<String> = message
            .get_item("action")?
            .map(|v| v.extract())
            .transpose()?;
        if action.as_deref() == Some("preview") {
            let result = match self.preview(py, message) {
                Ok(page) => page,
                Err(error) => {
                    let page = PyDict::new(py);
                    page.set_item(
                        "error",
                        error
                            .value(py)
                            .to_string()
                            .chars()
                            .take(1024)
                            .collect::<String>(),
                    )?;
                    page.unbind()
                }
            };
            result.bind(py).set_item(
                "request",
                message
                    .get_item("request")?
                    .unwrap_or_else(|| py.None().into_bound(py)),
            )?;
            return Ok(Some(("preview".into(), result)));
        }
        if action.as_deref() == Some("dispose") {
            let view = message
                .get_item("view")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_default();
            self.views.remove(&view);
            if self.views.is_empty() {
                self.clear_pages();
                return Ok(Some((String::new(), PyDict::new(py).unbind())));
            }
            return Ok(None);
        }
        if action.as_deref() == Some("attach") {
            let view = message
                .get_item("view")?
                .map(|v| v.extract::<String>())
                .transpose()?
                .unwrap_or_default();
            self.views.insert(view);
        }
        match self._action(py, message) {
            Ok(page) => Ok(Some(("page".into(), page))),
            Err(error) => {
                // Keep the last valid page; never call a failing provider again to report its error.
                let page = match &self.last_page {
                    Some(page) => page.bind(py).copy()?,
                    None => PyDict::new(py),
                };
                page.set_item(
                    "error",
                    error
                        .value(py)
                        .to_string()
                        .chars()
                        .take(1024)
                        .collect::<String>(),
                )?;
                Ok(Some(("page".into(), page.unbind())))
            }
        }
    }

    fn _clear_cache(&mut self) {
        self.clear_pages();
    }

    fn _close(&mut self) {
        self.source = None;
        self.clear_pages();
        self.views.clear();
        self.navigation.parents.clear();
        self.navigation.expanded.clear();
        self.navigation.position.previous.clear();
    }

    fn _repr_html_(&mut self, py: Python<'_>) -> PyResult<String> {
        let pages = self.current(py)?;
        let page = &pages.main.value;
        let html = super::scrollable_html(&page.html);
        if page.start == 0 && page.end == page.total && page.markers.is_empty() {
            return Ok(html);
        }
        Ok(format!(
            "<div>{}<p>Expression preview · items {}–{} of {}. Numbered markers identify omitted subexpressions. Navigation requires a live notebook with symbolica[notebook].</p></div>",
            html,
            page.start + 1,
            page.end,
            page.total
        ))
    }

    fn __repr__(&self) -> &'static str {
        "MathML viewer (use a live notebook to navigate)"
    }

    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(source) = &self.source {
            visit.call(source)?;
        }
        if let Some(page) = &self.last_page {
            visit.call(page)?;
        }
        if let Some(widget) = &self.widget {
            visit.call(widget)?;
        }
        Ok(())
    }
    fn __clear__(&mut self) {
        self.widget = None;
        self._close();
    }
}
