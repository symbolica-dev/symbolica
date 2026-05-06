use super::*;

/// Represents a part of an edge that connects to one vertex. It can be directed or undirected.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(from_py_object, name = "HalfEdge", module = "symbolica.core")]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PythonHalfEdge {
    half_edge: HalfEdge<Atom>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonHalfEdge {
    /// Create a new half-edge. The `data` can be any expression, and the `direction` can be `True` (outgoing),
    /// `False` (incoming) or `None` (undirected).
    #[new]
    #[pyo3(signature = (data, direction = None))]
    fn new(data: ConvertibleToExpression, direction: Option<bool>) -> Self {
        Self {
            half_edge: match direction {
                None => HalfEdge::undirected(data.to_expression().expr),
                Some(false) => HalfEdge::incoming(data.to_expression().expr),
                Some(true) => HalfEdge::outgoing(data.to_expression().expr),
            },
        }
    }

    /// Return a new half-edge with the direction flipped. Undirected edges remain undirected.
    fn flip(&self) -> Self {
        Self {
            half_edge: self.half_edge.flip(),
        }
    }

    /// Get the direction of the half-edge. `True` means outgoing, `False` means incoming, and `None` means undirected.
    fn direction(&self) -> Option<bool> {
        self.half_edge.direction
    }

    /// Get the data associated with the half-edge.
    fn data(&self) -> PythonExpression {
        self.half_edge.data.clone().into()
    }
}

/// A graph that supported directional edges, parallel edges, self-edges and custom data on the nodes and edges.
///
/// Warning: modifying the graph if it is contained in a `dict` or `set` will invalidate the hash.
#[cfg_attr(feature = "python_stubgen", gen_stub_pyclass)]
#[pyclass(from_py_object, name = "Graph", module = "symbolica.core")]
#[derive(Clone, PartialEq, Eq, Hash)]
pub struct PythonGraph {
    graph: Graph<Atom, Atom>,
}

#[cfg_attr(feature = "python_stubgen", gen_stub_pymethods)]
#[cfg_attr(not(feature = "python_stubgen"), remove_gen_stub)]
#[pymethods]
impl PythonGraph {
    /// Create an empty graph.
    #[new]
    fn new() -> Self {
        Self {
            graph: Graph::new(),
        }
    }

    /// Convert the graph into a portable string.
    pub fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{}", self.graph))
    }

    /// Print the graph in a human-readable format.
    fn __str__(&self) -> String {
        format!("{}", self.graph)
    }

    /// Hash the graph.
    fn __hash__(&self) -> u64 {
        let mut hasher = ahash::AHasher::default();
        self.graph.hash(&mut hasher);
        hasher.finish()
    }

    /// Copy the graph.
    fn __copy__(&self) -> PythonGraph {
        Self {
            graph: self.graph.clone(),
        }
    }

    /// Get the number of nodes.
    fn __len__(&self) -> usize {
        self.graph.nodes().len()
    }

    /// Compare two graphs.
    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.graph == other.graph),
            CompareOp::Ne => Ok(self.graph != other.graph),
            _ => Err(exceptions::PyTypeError::new_err(
                "Inequalities between graphs are not allowed".to_string(),
            )),
        }
    }

    /// Generate all connected graphs with `external_edges` half-edges and the given allowed list
    /// of vertex connections. The vertex signatures are given in terms of an edge direction (or `None` if
    /// there is no direction) and edge data.
    ///
    /// Returns the canonical form of the graph and the size of its automorphism group (including edge permutations).
    /// If `KeyboardInterrupt` is triggered during the generation, the generation will stop and will yield the currently generated
    /// graphs.
    ///
    /// Examples
    /// --------
    /// >>> from symbolica import *
    /// >>> g, q, gh = HalfEdge(S("g")), HalfEdge(S("q"), True), HalfEdge(S("gh"), True)
    /// >>> graphs = Graph.generate(
    /// >>>     external_nodes=[(1, g), (2, g)],
    /// >>>     vertex_signatures=[[g, g, g], [g, g, g, g],
    /// >>>                        [q.flip(), q, g], [gh.flip(), gh, g]],
    /// >>>     max_loops=2,
    /// >>> )
    /// >>> for (g, sym) in graphs.items():
    /// >>>     print(f'Symmetry factor = 1/{sym}:')
    /// >>>     print(g.to_dot())
    ///
    /// generates all connected graphs up to 2 loops with the specified vertices.
    ///
    /// Parameters
    /// ----------
    /// external_nodes: Sequence[tuple[Expression | int, HalfEdge]]
    ///     The external edges, consisting of a tuple of the node data and a tuple of the edge direction and edge data.
    ///     If the node data is the same, flip symmetries will be recognized.
    /// vertex_signatures: Sequence[Sequence[HalfEdge]]
    ///     The allowed connections for each vertex.
    /// max_vertices: int, optional
    ///     The maximum number of vertices in the graph.
    /// max_loops: int, optional
    ///     The maximum number of loops in the graph.
    /// max_bridges: int, optional
    ///     The maximum number of bridges in the graph.
    /// allow_self_loops: bool, optional
    ///     Whether self-edges are allowed.
    /// allow_zero_flow_edges: bool, optional
    ///     Whether bridges that do not need to be crossed to connect external vertices are allowed.
    /// filter_fn: Optional[Callable[[Graph, int], bool]], optional
    ///     Set a filter function that is called during the graph generation.
    ///     The first argument is the graph `g` and the second argument the vertex count `n`
    ///     that specifies that the first `n` vertices are completed (no new edges will) be
    ///     assigned to them. The filter function should return `true` if the current
    ///     incomplete graph is allowed, else it should return `false` and the graph is discarded.
    /// progress_fn: Optional[Callable[[Graph, bool]], optional
    ///     Set a progress function that is called every time a new unique graph is created.
    ///     The argument is the newly created graph.
    ///     If the function returns `false`, the generation is aborted and the currently
    ///     generated graphs are returned.
    #[pyo3(signature = (external_edges, vertex_signatures, max_vertices = None, max_loops = None,
        max_bridges = None, allow_self_loops = None, allow_zero_flow_edges = None, filter_fn = None, progress_fn = None))]
    #[classmethod]
    fn generate(
        _cls: &Bound<'_, PyType>,
        external_edges: Vec<(ConvertibleToExpression, PythonHalfEdge)>,
        vertex_signatures: Vec<Vec<PythonHalfEdge>>,
        max_vertices: Option<usize>,
        max_loops: Option<usize>,
        max_bridges: Option<usize>,
        allow_self_loops: Option<bool>,
        allow_zero_flow_edges: Option<bool>,
        #[gen_stub(override_type(
            type_repr = "typing.Optional[typing.Callable[[Graph, int], bool]]"
        ))]
        filter_fn: Option<Py<PyAny>>,
        #[gen_stub(override_type(type_repr = "typing.Optional[typing.Callable[[Graph], bool]]"))]
        progress_fn: Option<Py<PyAny>>,
    ) -> PyResult<HashMap<PythonGraph, PythonExpression>> {
        if max_vertices.is_none() && max_loops.is_none() {
            return Err(exceptions::PyValueError::new_err(
                "At least one of max_vertices or max_loop must be set",
            ));
        }

        let external_edges: Vec<_> = external_edges
            .into_iter()
            .map(|(a, b)| (a.to_expression().expr, b.half_edge))
            .collect();
        let vertex_signatures: Vec<_> = vertex_signatures
            .into_iter()
            .map(|v| v.into_iter().map(|x| x.half_edge).collect())
            .collect();

        let mut settings = GenerationSettings::new();
        if let Some(max_vertices) = max_vertices {
            settings = settings.max_vertices(max_vertices);
        }

        if let Some(max_loops) = max_loops {
            settings = settings.max_loops(max_loops);
        }

        if let Some(max_bridges) = max_bridges {
            settings = settings.max_bridges(max_bridges);
        }

        if let Some(allow_self_loops) = allow_self_loops {
            settings = settings.allow_self_loops(allow_self_loops);
        }

        if let Some(allow_zero_flow_edge) = allow_zero_flow_edges {
            settings = settings.allow_zero_flow_edges(allow_zero_flow_edge);
        }

        let abort = Arc::new(std::sync::atomic::AtomicBool::new(false));

        if let Some(filter_fn) = filter_fn {
            let abort = abort.clone();
            settings = settings.filter_fn(Box::new(move |g, v| {
                Python::attach(|py| {
                    match filter_fn.call(py, (Self { graph: g.clone() }, v), None) {
                        Ok(r) => r
                            .is_truthy(py)
                            .expect("Match map does not return a boolean"),
                        Err(e) => {
                            if e.is_instance_of::<exceptions::PyKeyboardInterrupt>(py) {
                                abort.store(true, std::sync::atomic::Ordering::Relaxed);
                                false
                            } else {
                                panic!("Bad callback function: {}", e);
                            }
                        }
                    }
                })
            }));
        }

        if let Some(progress_fn) = progress_fn {
            settings = settings.progress_fn(Box::new(move |g| {
                Python::attach(|py| {
                    match progress_fn.call(py, (Self { graph: g.clone() },), None) {
                        Ok(r) => r.is_truthy(py).unwrap_or(true),
                        Err(e) => {
                            error!("Bad callback function: {}", e);
                            false
                        }
                    }
                })
            }));
        }

        settings = settings.abort_check(Box::new(move || {
            if abort.load(std::sync::atomic::Ordering::Relaxed) {
                true
            } else {
                Python::attach(|py| py.check_signals())
                    .map(|_| false)
                    .unwrap_or(true)
            }
        }));

        Ok(
            Graph::generate(&external_edges, &vertex_signatures, settings)
                .unwrap_or_else(|e| e)
                .into_iter()
                .map(|(k, v)| (Self { graph: k }, Atom::num(v).into()))
                .collect(),
        )
    }

    /// Convert the graph to a graphviz dot string.
    fn to_dot(&self) -> String {
        self.graph.to_dot()
    }

    /// Convert the graph to a mermaid string.
    fn to_mermaid(&self) -> String {
        self.graph.to_mermaid()
    }

    /// Add a node with data `data` to the graph, returning the index of the node.
    /// The default data is the number 0.
    #[pyo3(signature = (data = None))]
    fn add_node(&mut self, data: Option<ConvertibleToExpression>) -> usize {
        self.graph
            .add_node(data.map(|x| x.to_expression().expr).unwrap_or_default())
    }

    /// Add an edge between the `source` and `target` nodes, returning the index of the edge.
    /// Optionally, the edge can be set as directed. The default data is the number 0.
    #[pyo3(signature = (source, target, directed = false, data = None))]
    fn add_edge(
        &mut self,
        source: usize,
        target: usize,
        directed: bool,
        data: Option<ConvertibleToExpression>,
    ) -> PyResult<usize> {
        self.graph
            .add_edge(
                source,
                target,
                directed,
                data.map(|x| x.to_expression().expr).unwrap_or_default(),
            )
            .map_err(exceptions::PyValueError::new_err)
    }

    /// Set the data of the node at index `index`, returning the old data.
    pub fn set_node_data(
        &mut self,
        index: isize,
        data: PythonExpression,
    ) -> PyResult<PythonExpression> {
        if index.unsigned_abs() < self.graph.nodes().len() {
            let n = if index < 0 {
                self.graph.nodes().len() - index.unsigned_abs()
            } else {
                index as usize
            };
            Ok(self.graph.set_node_data(n, data.expr).into())
        } else {
            Err(PyIndexError::new_err(format!(
                "Index {} out of bounds: the graph only has {} nodes.",
                index,
                self.graph.nodes().len(),
            )))
        }
    }

    /// Set the data of the edge at index `index`, returning the old data.
    pub fn set_edge_data(
        &mut self,
        index: isize,
        data: PythonExpression,
    ) -> PyResult<PythonExpression> {
        if index.unsigned_abs() < self.graph.edges().len() {
            let e = if index < 0 {
                self.graph.edges().len() - index.unsigned_abs()
            } else {
                index as usize
            };
            Ok(self.graph.set_edge_data(e, data.expr).into())
        } else {
            Err(PyIndexError::new_err(format!(
                "Index {} out of bounds: the graph only has {} edges.",
                index,
                self.graph.edges().len(),
            )))
        }
    }

    /// Set the directed status of the edge at index `index`, returning the old value.
    pub fn set_directed(&mut self, index: isize, directed: bool) -> PyResult<bool> {
        if index.unsigned_abs() < self.graph.edges().len() {
            let e = if index < 0 {
                self.graph.edges().len() - index.unsigned_abs()
            } else {
                index as usize
            };
            Ok(self.graph.set_directed(e, directed))
        } else {
            Err(PyIndexError::new_err(format!(
                "Index {} out of bounds: the graph only has {} edges.",
                index,
                self.graph.edges().len(),
            )))
        }
    }

    /// Get the `idx`th node.
    fn __getitem__(&self, idx: isize) -> PyResult<(Vec<usize>, PythonExpression)> {
        self.node(idx)
    }

    /// Get the number of nodes.
    fn num_nodes(&self) -> usize {
        self.graph.nodes().len()
    }

    /// Get the number of edges.
    fn num_edges(&self) -> usize {
        self.graph.edges().len()
    }

    /// Get the number of loops.
    fn num_loops(&self) -> usize {
        self.graph.num_loops()
    }

    /// Get the `idx`th node, consisting of the edge indices and the data.
    fn node(&self, idx: isize) -> PyResult<(Vec<usize>, PythonExpression)> {
        if idx.unsigned_abs() < self.graph.nodes().len() {
            let n = if idx < 0 {
                self.graph
                    .node(self.graph.nodes().len() - idx.unsigned_abs())
            } else {
                self.graph.node(idx as usize)
            };
            Ok((n.edges.clone(), n.data.clone().into()))
        } else {
            Err(PyIndexError::new_err(format!(
                "Index {} out of bounds: the graph only has {} nodes.",
                idx,
                self.graph.nodes().len(),
            )))
        }
    }

    /// Get all nodes, consisting of the edge indices and the data.
    fn nodes(&self) -> Vec<(Vec<usize>, PythonExpression)> {
        self.graph
            .nodes()
            .iter()
            .map(|n| (n.edges.clone(), n.data.clone().into()))
            .collect()
    }

    /// Get the `idx`th edge, consisting of the the source vertex, target vertex, whether the edge is directed, and the data.
    fn edge(&self, idx: isize) -> PyResult<(usize, usize, bool, PythonExpression)> {
        if idx.unsigned_abs() < self.graph.edges().len() {
            let e = if idx < 0 {
                self.graph
                    .edge(self.graph.edges().len() - idx.unsigned_abs())
            } else {
                self.graph.edge(idx as usize)
            };
            Ok((
                e.vertices.0,
                e.vertices.1,
                e.directed,
                e.data.clone().into(),
            ))
        } else {
            Err(PyIndexError::new_err(format!(
                "Index {} out of bounds: the graph only has {} edges.",
                idx,
                self.graph.edges().len(),
            )))
        }
    }

    /// Get all edges, consisting of the the source vertex, target vertex, whether the edge is directed, and the data.
    fn edges(&self) -> Vec<(usize, usize, bool, PythonExpression)> {
        self.graph
            .edges()
            .iter()
            .map(|e| {
                (
                    e.vertices.0,
                    e.vertices.1,
                    e.directed,
                    e.data.clone().into(),
                )
            })
            .collect()
    }

    /// Write the graph in a canonical form.
    /// Returns the canonicalized graph, the vertex map, the automorphism group size, and the orbit.
    fn canonize(&self) -> (PythonGraph, Vec<usize>, PythonExpression, Vec<usize>) {
        let c = self.graph.canonize();
        (
            Self { graph: c.graph },
            c.vertex_map,
            Atom::num(c.automorphism_group_size).into(),
            c.orbit,
        )
    }

    /// Sort and relabel the edges of the graph, keeping the vertices fixed.
    pub fn canonize_edges(&mut self) {
        self.graph.canonize_edges();
    }

    /// Return true `iff` the graph is isomorphic to `other`.
    fn is_isomorphic(&self, other: &PythonGraph) -> bool {
        self.graph.is_isomorphic(&other.graph)
    }
}
