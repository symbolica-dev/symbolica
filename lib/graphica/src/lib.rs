//! Graphs and related algorithms.
//!
//! # Examples
//!
//! ```
//! # use graphica::Graph;
//! let mut g = Graph::new();
//! let n0 = g.add_node(0);
//! let n1 = g.add_node(1);
//! let n2 = g.add_node(2);
//!
//! g.add_edge(n2, n1, true, 0).unwrap();
//! g.add_edge(n0, n0, false, 0).unwrap();
//! g.add_edge(n0, n1, true, 0).unwrap();
//! g.add_edge(n1, n0, false, 2).unwrap();
//!
//! g.canonize_edges();
//!
//! assert_eq!(g.node(0).edges, [0, 1, 2]);
//! ```

use ahash::{HashMap, HashSet};
use dyn_clone::DynClone;
use std::{
    cmp::Ordering,
    fmt::{Debug, Display},
    hash::Hash,
};

use numerica::{combinatorics::partitions, domains::integer::Integer, utils::AbortCheck};

/// A node in a graph, with arbitrary data.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Node<NodeData = Empty> {
    /// Arbitrary data associated with the node.
    pub data: NodeData,
    /// Indices of the edges connected to the node.
    pub edges: Vec<usize>,
    /// The valence of the node, i.e., the number of edges connected to it.
    /// A self-loop counts twice.
    pub valence: usize,
}

/// An edge in a graph, with arbitrary data.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Edge<EdgeData = Empty> {
    /// Indices of the vertices connected by the edge.
    pub vertices: (usize, usize),
    /// If `true`, the edge is directed from `vertices.0` to `vertices.1`.
    pub directed: bool,
    /// Arbitrary data associated with the edge.
    pub data: EdgeData,
}

/// Represents a part of an edge that connects to one vertex. It can be directed or undirected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct HalfEdge<E> {
    /// The direction of the half-edge with respect to a vertex it would connect to. `None` if there is no direction, `Some(true)` for outgoing, `Some(false)` for incoming.
    pub direction: Option<bool>,
    /// The data associated with the half-edge.
    pub data: E,
}

impl<E> HalfEdge<E> {
    /// Create an undirected half-edge.
    pub fn undirected(data: E) -> Self {
        Self {
            direction: None,
            data,
        }
    }

    /// Create an incoming half-edge.
    pub fn incoming(data: E) -> Self {
        Self {
            direction: Some(false),
            data,
        }
    }

    /// Create an outgoing half-edge.
    pub fn outgoing(data: E) -> Self {
        Self {
            direction: Some(true),
            data,
        }
    }

    /// Create a copy of this half-edge with the direction flipped (if not undirected).
    pub fn flip(&self) -> Self
    where
        E: Clone,
    {
        Self {
            direction: self.direction.map(|d| !d),
            data: self.data.clone(),
        }
    }
}

impl<EdgeData> Edge<EdgeData> {
    #[inline]
    pub fn is_self_loop(&self) -> bool {
        self.vertices.0 == self.vertices.1
    }

    pub fn is_neighbor(&self, n: usize) -> bool {
        self.vertices.0 == n || self.vertices.1 == n
    }
}

/// Empty data type.
#[derive(Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Empty;

impl Display for Empty {
    fn fmt(&self, _f: &mut std::fmt::Formatter) -> std::fmt::Result {
        Ok(())
    }
}

/// Data that has a public part and a private part. The private
/// part is not used for equality or hashing.
#[derive(Clone)]
pub struct HiddenData<T, U> {
    pub data: T,
    pub hidden: U,
}

impl<T, U> HiddenData<T, U> {
    pub fn new(data: T, hidden: U) -> Self {
        HiddenData { data, hidden }
    }
}

impl<T: PartialEq, U> PartialEq for HiddenData<T, U> {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<T: Eq, U> Eq for HiddenData<T, U> {}

impl<T: Hash, U> Hash for HiddenData<T, U> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.data.hash(state);
    }
}

impl<T: Display, U: Display> Display for HiddenData<T, U> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{} ({})", self.data, self.hidden)
    }
}

impl<T: PartialOrd, U> PartialOrd for HiddenData<T, U> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.data.partial_cmp(&other.data)
    }
}

impl<T: Ord, U> Ord for HiddenData<T, U> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.data.cmp(&other.data)
    }
}

/// A multigraph with support for arbitrary node and edge data.
///
/// Use [HiddenData] to hide parts of the data from all equality and hashing.
///
/// # Example
///
/// ```
/// use graphica::Graph;
/// let mut g = Graph::new();
/// let n0 = g.add_node(0);
/// let n1 = g.add_node(1);
/// let n2 = g.add_node(2);
///
/// g.add_edge(n2, n1, true, 0).unwrap();
/// g.add_edge(n0, n0, false, 0).unwrap();
/// g.add_edge(n0, n1, true, 0).unwrap();
/// g.add_edge(n1, n0, false, 2).unwrap();
///
/// g.canonize_edges();
///
/// assert_eq!(g.node(0).edges, [0, 1, 2]);
/// ```
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Graph<NodeData = Empty, EdgeData = Empty> {
    nodes: Vec<Node<NodeData>>,
    edges: Vec<Edge<EdgeData>>,
}

impl<N: Display, E: Display> std::fmt::Display for Graph<N, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        for (i, x) in self.nodes.iter().enumerate() {
            writeln!(f, "N{}: {}", i, x.data)?;
        }

        for x in &self.edges {
            if x.directed {
                writeln!(f, "{} {}->{}", x.data, x.vertices.0, x.vertices.1)?;
            } else {
                writeln!(f, "{} {}--{}", x.data, x.vertices.0, x.vertices.1)?;
            }
        }
        Ok(())
    }
}

impl<N: Display, E: Display> Graph<N, E> {
    pub fn to_dot(&self) -> String {
        let mut out = String::new();
        out.push_str("digraph G {\n");

        for (i, x) in self.nodes.iter().enumerate() {
            out.push_str(&format!("  {} [label=\"{}\"];\n", i, x.data));
        }

        for x in &self.edges {
            if x.directed {
                out.push_str(&format!(
                    "  {} -> {} [label=\"{}\"];\n",
                    x.vertices.0, x.vertices.1, x.data
                ));
            } else {
                out.push_str(&format!(
                    "  {} -> {} [dir=none,label=\"{}\"];\n",
                    x.vertices.0, x.vertices.1, x.data
                ));
            }
        }

        out.push_str("}\n");
        out
    }

    pub fn to_mermaid(&self) -> String {
        let mut out = String::new();
        out.push_str("graph TD;\n");

        for (i, x) in self.nodes.iter().enumerate() {
            let d = format!("{}", x.data);
            if d.is_empty() {
                out.push_str(&format!("  {i};\n"));
            } else {
                out.push_str(&format!("  {i}[\"{d}\"];\n"));
            }
        }

        for x in &self.edges {
            let d = format!("{}", x.data);

            if d.is_empty() {
                if x.directed {
                    out.push_str(&format!("  {} --> {};\n", x.vertices.0, x.vertices.1,));
                } else {
                    out.push_str(&format!("  {} --- {};\n", x.vertices.0, x.vertices.1,));
                }
            } else if x.directed {
                out.push_str(&format!(
                    "  {} -->|\"{}\"| {};\n",
                    x.vertices.0, x.data, x.vertices.1,
                ));
            } else {
                out.push_str(&format!(
                    "  {} ---|\"{}\"| {};\n",
                    x.vertices.0, x.data, x.vertices.1,
                ));
            }
        }

        out
    }
}

/// An edge that is not part of the spanning tree, as it would form a loop.
#[derive(Clone, Copy, Debug)]
pub struct BackEdge {
    /// The target of the back-edge is the node
    /// that is furthest from the root of the spanning tree.
    pub target: usize,
    /// The id of the chain that is created by this back-edge.
    /// It is the global index of this back-edge.
    /// Make sure to call [SpanningTree::chain_decomposition] first.
    pub chain_id: Option<usize>,
}

/// Information about a node in a spanning tree.
#[derive(Clone, Debug)]
pub struct NodeInfo {
    /// The position of the node in the DFS order (`None` if not visited)
    pub position: Option<usize>,
    // The index of the edge from the node going to the parent (`None` for root)
    pub edge_id: Option<usize>,
    /// The parent node. For the root, this is the node itself.
    pub parent: usize,
    /// The chain id this node belongs to (`None` if it is a bridge).
    /// Make sure to call [SpanningTree::chain_decomposition] first.
    pub chain_id: Option<usize>,
    /// If `true`, this node is an external node (valence 1 in the original graph).
    pub external: bool,
    /// The back-edges starting from this node.
    /// The start of a back-edge is always the node that is closest
    /// to the root in the spanning tree.
    pub back_edges: Vec<BackEdge>,
}

/// A spanning tree representation of a graph.
/// Parts of the graph may not be in the tree.
#[derive(Clone, Debug)]
pub struct SpanningTree {
    /// Information about the nodes in the spanning tree.
    pub nodes: Vec<NodeInfo>,
    /// The order in which the nodes were visited in the DFS.
    pub order: Vec<usize>,
}

impl SpanningTree {
    pub fn is_connected(&self) -> bool {
        self.nodes.iter().all(|x| x.position.is_some())
    }

    /// Assign chain ids to the nodes and back-edges in the spanning tree.
    pub fn chain_decomposition(&mut self) {
        let mut chain_id = 0;
        for &n in &self.order {
            let mut back_edge_index = 0;
            while back_edge_index < self.nodes[n].back_edges.len() {
                let back_edge = &mut self.nodes[n].back_edges[back_edge_index];
                back_edge.chain_id = Some(chain_id);
                chain_id += 1;

                let back_edge = back_edge.clone();
                back_edge_index += 1;

                if back_edge.target == n {
                    // self-loop
                    continue;
                }

                // set temporary blocker
                if self.nodes[n].chain_id.is_none() {
                    self.nodes[n].chain_id = back_edge.chain_id;
                }

                let mut target = back_edge.target;
                while self.nodes[target].chain_id.is_none() {
                    let nn = &mut self.nodes[target];
                    nn.chain_id = back_edge.chain_id;
                    target = nn.parent;
                }

                // the start node is always excluded from the chain,
                // as we define the chain to contain the edges
                // that connect the node to its parent
                if self.nodes[n].chain_id == back_edge.chain_id {
                    self.nodes[n].chain_id = None;
                }
            }
        }
    }

    /// Count non-external bridge nodes. Make sure to call [Self::chain_decomposition] first.
    pub fn count_bridges(&self) -> usize {
        self.bridges().count()
    }

    fn bridge_nodes(&self) -> impl Iterator<Item = (usize, &NodeInfo)> {
        self.nodes.iter().enumerate().filter(|(n, x)| {
            x.chain_id.is_none()
                && !self.nodes[x.parent].external
                && !x.external
                && x.parent != *n // exclude the root
                && !self.nodes[x.parent].back_edges.iter().any(|end| *n == end.target)
        })
    }

    /// Get non-external bridge nodes. Make sure to call [Self::chain_decomposition] first.
    pub fn bridges(&self) -> impl Iterator<Item = usize> {
        self.bridge_nodes().map(|x| x.1.edge_id.unwrap())
    }

    /// Detect if the graph has zero-flow bridges. This only works when the root
    /// node is an external node (if there are any).
    pub fn has_zero_flow_bridges(&self) -> bool {
        if self.nodes.iter().all(|x| !x.external) {
            return self.bridge_nodes().next().is_some();
        }

        assert!(self.nodes[0].external);
        let mut visited = vec![false; self.nodes.len()];
        for (x, n) in self.nodes.iter().enumerate() {
            if n.external {
                let mut p = x;
                while !visited[p] && self.nodes[p].parent != p {
                    visited[p] = true;
                    p = self.nodes[p].parent;
                }
            }
        }

        visited.iter().any(|x| !*x) && self.bridge_nodes().any(|b| !visited[b.0])
    }
}

impl<N, E> Default for Graph<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N, E> Graph<N, E> {
    /// Create an empty graph.
    pub fn new() -> Self {
        Graph {
            nodes: Vec::new(),
            edges: Vec::new(),
        }
    }

    /// Add a node to the graph, with arbitrary data, and return its index.
    pub fn add_node(&mut self, data: N) -> usize {
        let index = self.nodes.len();
        self.nodes.push(Node {
            edges: Vec::new(),
            data,
            valence: 0,
        });
        index
    }

    /// Add an edge between vertex indices `source` and `target` to the graph, with arbitrary data.
    /// If `directed` is true, the edge is directed from `source` to `target`.
    pub fn add_edge(
        &mut self,
        source: usize,
        target: usize,
        directed: bool,
        data: E,
    ) -> Result<usize, &'static str> {
        if source >= self.nodes.len() || target >= self.nodes.len() {
            return Err("Invalid node index");
        }

        let index = self.edges.len();
        self.edges.push(Edge {
            vertices: if !directed && source > target {
                (target, source)
            } else {
                (source, target)
            },
            directed,
            data,
        });
        self.nodes[source].edges.push(index);
        self.nodes[source].valence += 1;
        self.nodes[target].valence += 1;

        if source != target {
            self.nodes[target].edges.push(index);
        }

        Ok(index)
    }

    /// Set the data of the node at index `index`, returning the old data.
    pub fn set_node_data(&mut self, index: usize, data: N) -> N {
        std::mem::replace(&mut self.nodes[index].data, data)
    }

    /// Set the data of the edge at index `index`, returning the old data.
    pub fn set_edge_data(&mut self, index: usize, data: E) -> E {
        std::mem::replace(&mut self.edges[index].data, data)
    }

    /// Set the directed status of the edge at index `index`, returning the old value.
    pub fn set_directed(&mut self, index: usize, directed: bool) -> bool {
        std::mem::replace(&mut self.edges[index].directed, directed)
    }

    /// Delete the last added edge. This operation is O(1).
    pub fn delete_last_edge(&mut self) -> Option<Edge<E>> {
        if let Some(edge) = self.edges.pop() {
            self.nodes[edge.vertices.0].edges.pop();
            self.nodes[edge.vertices.0].valence -= 1;
            self.nodes[edge.vertices.1].valence -= 1;
            if edge.vertices.0 != edge.vertices.1 {
                self.nodes[edge.vertices.1].edges.pop();
            }
            Some(edge)
        } else {
            None
        }
    }

    /// Remove the last added empty node. This operation is O(1).
    pub fn delete_last_empty_node(&mut self) -> Option<Node<N>> {
        if let Some(node) = self.nodes.last()
            && node.edges.is_empty()
        {
            return self.nodes.pop();
        }
        None
    }

    /// Get the node with index `index`.
    #[inline(always)]
    pub fn node(&self, index: usize) -> &Node<N> {
        &self.nodes[index]
    }

    /// Get the edge with index `index`.
    #[inline(always)]
    pub fn edge(&self, index: usize) -> &Edge<E> {
        &self.edges[index]
    }

    /// Get all nodes of the graph.
    #[inline(always)]
    pub fn nodes(&self) -> &[Node<N>] {
        &self.nodes
    }

    /// Get all edges of the graph.
    #[inline(always)]
    pub fn edges(&self) -> &[Edge<E>] {
        &self.edges
    }

    // Get the number of loop in the graph, using E - V + 1
    pub fn num_loops(&self) -> usize {
        self.edges.len() + 1 - self.nodes.len()
    }

    /// Generate a spanning tree of the graph, starting at `start_vertex`.
    pub fn get_spanning_tree(&self, start_vertex: usize) -> SpanningTree {
        let mut nodes_to_visit = vec![(start_vertex, start_vertex, None)];
        let mut tree_nodes: Vec<_> = self
            .nodes
            .iter()
            .map(|n| NodeInfo {
                position: None,
                edge_id: None,
                parent: 0,
                chain_id: None,
                external: n.valence == 1,
                back_edges: vec![],
            })
            .collect();
        let mut order = vec![0; self.nodes.len()];

        let mut index = 0;
        while let Some((n, parent, edge_id)) = nodes_to_visit.pop() {
            if let Some(p) = tree_nodes[n].position {
                let par = &mut tree_nodes[parent];
                if par.position.unwrap() < p {
                    par.back_edges.push(BackEdge {
                        target: n,
                        chain_id: None,
                    });
                }
                continue;
            }

            tree_nodes[n].position = Some(index);
            tree_nodes[n].edge_id = edge_id;
            tree_nodes[n].parent = parent;
            order[index] = n;
            index += 1;

            for e in &self.node(n).edges {
                let edge = self.edge(*e);
                let target = if edge.vertices.0 == n {
                    edge.vertices.1
                } else {
                    edge.vertices.0
                };

                if n == target {
                    tree_nodes[n].back_edges.push(BackEdge {
                        target,
                        chain_id: None,
                    });
                }

                if tree_nodes[target].position.is_none() {
                    nodes_to_visit.push((target, n, Some(*e)));
                }
            }
        }

        SpanningTree {
            nodes: tree_nodes,
            order,
        }
    }

    /// Check if the graph is connected.
    pub fn is_connected(&self) -> bool {
        if self.nodes.is_empty() {
            return true;
        }

        self.get_spanning_tree(0)
            .nodes
            .iter()
            .all(|x| x.position.is_some())
    }
}

impl<N, E: Eq + Ord + Hash> Graph<N, E> {
    /// Get the number of different ways to permute the multi-edges, leading
    /// to the same graph, while keeping the vertices fixed.
    ///
    /// Every self-loop yields an additional factor two.
    pub fn get_edge_automorphism_group_size(&self) -> Integer {
        let mut count = Integer::one();
        let mut h = HashMap::default();

        for e in &self.edges {
            if e.vertices.0 == e.vertices.1 {
                count *= 2;
            }

            h.entry(e.vertices)
                .or_insert(vec![])
                .push((e.directed, &e.data));
        }

        for (_, mut v) in h.into_iter() {
            v.sort();

            let mut counter = 1;
            let mut last = &v[0];
            for d in v.iter().skip(1) {
                if d == last {
                    counter += 1;
                } else {
                    count *= Integer::factorial(counter);
                    counter = 1;
                    last = d;
                }
            }
            count *= Integer::factorial(counter);
        }

        count
    }

    /// Sort and relabel the edges of the graph, keeping the vertices fixed.
    pub fn canonize_edges(&mut self) {
        for n in &mut self.nodes {
            n.edges.clear();
        }

        self.edges.sort();

        for (i, e) in self.edges.iter().enumerate() {
            self.nodes[e.vertices.0].edges.push(i);
            if e.vertices.0 != e.vertices.1 {
                self.nodes[e.vertices.1].edges.push(i);
            }
        }
    }
}

struct GenerationSettingsAndInput<'a, N, E> {
    vertex_signatures: &'a [Vec<HalfEdge<E>>],
    allowed_structures: &'a HashSet<Vec<HalfEdge<E>>>,
    min_degree: usize,
    max_degree: usize,
    current_structure: Vec<HalfEdge<E>>,
    settings: &'a mut GenerationSettings<N, E>,
}

impl<N, E: Clone + Ord + Hash> GenerationSettingsAndInput<'_, N, E> {
    pub fn is_structure_allowed(&mut self, vertex: &[HalfEdge<E>]) -> bool {
        self.current_structure.clear();
        self.current_structure.extend_from_slice(vertex);
        self.current_structure.sort_unstable();
        self.allowed_structures.contains(&self.current_structure)
    }
}

pub trait FilterFn<N, E>: Fn(&Graph<N, E>, usize) -> bool + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(<N, E> FilterFn<N,E>);
impl<N, E, T: Clone + Send + Sync + Fn(&Graph<N, E>, usize) -> bool> FilterFn<N, E> for T {}

pub trait ProgressFn<N, E>: FnMut(&Graph<N, E>) -> bool + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(<N, E> ProgressFn<N,E>);
impl<N, E, T: Clone + Send + Sync + FnMut(&Graph<N, E>) -> bool> ProgressFn<N, E> for T {}

#[derive(Clone)]
pub struct GenerationSettings<N, E> {
    filter_fn: Option<Box<dyn FilterFn<N, E>>>,
    abort_check: Option<Box<dyn AbortCheck>>,
    progress_fn: Option<Box<dyn ProgressFn<N, E>>>,
    max_vertices: Option<usize>,
    max_loops: Option<usize>,
    max_bridges: Option<usize>,
    allow_self_loops: bool,
    allow_zero_flow_edges: bool,
}

impl<N, E> Default for GenerationSettings<N, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<N, E> std::fmt::Debug for GenerationSettings<N, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.debug_struct("GenerationSettings")
            .field("filter_fn", &self.filter_fn.is_some())
            .field("abort_check", &self.abort_check.is_some())
            .field("progress_fn", &self.progress_fn.is_some())
            .field("max_vertices", &self.max_vertices)
            .field("max_loops", &self.max_loops)
            .field("max_bridges", &self.max_bridges)
            .field("allow_self_loops", &self.allow_self_loops)
            .field("allow_zero_flow_edges", &self.allow_zero_flow_edges)
            .finish()
    }
}

impl<N, E> GenerationSettings<N, E> {
    /// Create a new settings object with default values.
    /// These are:
    /// ```text
    /// filter_fn: None
    /// abort_check: None
    /// max_vertices: None
    /// max_loops: None
    /// max_bridges: None
    /// allow_self_loops: false,
    /// allow_zero_flow_edges: false,
    /// ```
    pub const fn new() -> Self {
        Self {
            filter_fn: None,
            abort_check: None,
            progress_fn: None,
            max_vertices: None,
            max_loops: None,
            max_bridges: None,
            allow_self_loops: false,
            allow_zero_flow_edges: false,
        }
    }

    /// Set the maximum number of vertices in the generated graphs.
    pub fn max_vertices(mut self, max_vertices: usize) -> Self {
        self.max_vertices = Some(max_vertices);
        self
    }

    /// Set the maximum number of loops in the generated graphs.
    pub fn max_loops(mut self, max_loops: usize) -> Self {
        self.max_loops = Some(max_loops);
        self
    }

    /// Set the maximum number of bridges in the generated graphs.
    pub fn max_bridges(mut self, max_bridges: usize) -> Self {
        self.max_bridges = Some(max_bridges);
        self
    }

    /// Allow self-loops in the generated graphs.
    pub fn allow_self_loops(mut self, allow_self_loops: bool) -> Self {
        self.allow_self_loops = allow_self_loops;
        self
    }

    /// Allow bridges that do not need to be crossed to connect external vertices.
    pub fn allow_zero_flow_edges(mut self, allow_zero_flow_edges: bool) -> Self {
        self.allow_zero_flow_edges = allow_zero_flow_edges;
        self
    }

    /// Set a filter function that is called during the graph generation.
    /// The first argument is the graph `g` and the second argument the vertex count `n`
    /// that specifies that the first `n` vertices are completed (no new edges will) be
    /// assigned to them. The filter function should return `true` if the current
    /// incomplete graph is allowed, else it should return `false` and the graph is discarded.
    pub fn filter_fn(mut self, filter_fn: Box<dyn FilterFn<N, E>>) -> Self {
        self.filter_fn = Some(filter_fn);
        self
    }

    /// A function that is called during graph generation to see if the graph
    /// generation should be aborted. The graphs that were generated at this stage
    /// are returned.
    pub fn abort_check(mut self, abort_check: Box<dyn AbortCheck>) -> Self {
        self.abort_check = Some(abort_check);
        self
    }

    /// A function that is called after a new unique graph is generated.
    /// The map of all generated graphs is passed as an argument. If the function
    /// returns `false`, the graph generation is aborted and the currently
    /// generated graphs are returned.
    ///
    /// This function can be used to report progress or to abort the generation.
    pub fn progress_fn(mut self, progress_fn: Box<dyn ProgressFn<N, E>>) -> Self {
        self.progress_fn = Some(progress_fn);
        self
    }
}

impl<N: Default + Clone + Eq + Hash + Ord, E: Clone + Ord + Eq + Hash> Graph<N, E> {
    /// Generate all connected graphs with `external_edges` half-edges and the given allowed list
    /// of vertex connections.
    ///
    /// The external edges are specified as a list of pairs `(node_data, half_edge)`, where `node_data`
    /// is the data associated to the external node to which the half-edge is connected and `half_edge`
    /// is a [HalfEdge] specifying the edge data and direction.
    ///
    /// The vertex signatures specify all allowed connections of directed half-edges. For example:
    /// ```
    /// # use graphica::{HalfEdge, GenerationSettings, Graph};
    /// let g = HalfEdge::undirected("g");
    /// let q = HalfEdge::incoming("q");
    /// let vertex_signatures = [vec![g, g, g], vec![q.flip(), q, g]];
    /// ```
    /// allows a vertex with three undirected edges with data `g` and a vertex that joins one undirected edge with data `g`,
    /// one incoming edge with data `q` and one outgoing edge with data `q`.
    ///
    /// Returns the canonical form of the graph and the size of its automorphism group (including edge permutations).
    /// If the graph generation was aborted, all currently generated graphs are returned in the `Err` argument.
    ///
    /// # Example
    ///
    /// ```
    /// # use graphica::{HalfEdge, GenerationSettings, Graph};
    /// let g = HalfEdge::undirected("g");
    /// let q = HalfEdge::incoming("q");
    ///
    /// let mut graphs = Graph::generate(
    ///     &[(1, g), (2, g)],
    ///     &[
    ///         vec![g, g, g],
    ///         vec![q.flip(), q, g],
    ///         vec![g, g, g, g],
    ///     ],
    ///     GenerationSettings::new()
    ///         .max_loops(2)
    ///         .max_bridges(0)
    ///         .allow_self_loops(true)
    /// ).unwrap();
    /// assert_eq!(graphs.len(), 19);
    /// ```
    pub fn generate(
        external_edges: &[(N, HalfEdge<E>)],
        vertex_signatures: &[Vec<HalfEdge<E>>],
        mut settings: GenerationSettings<N, E>,
    ) -> Result<HashMap<Graph<N, E>, Integer>, HashMap<Graph<N, E>, Integer>> {
        if settings.max_vertices.is_none()
            && settings.max_loops.is_none()
            && settings.progress_fn.is_none()
            && settings.abort_check.is_none()
        {
            panic!(
                "Infinite loop for graph generation: at least one of max_vertices, max_loops, progress_fn, abort_check must be set"
            );
        }

        let vertex_sorted: Vec<_> = vertex_signatures
            .iter()
            .map(|x| {
                let mut x = x.clone();
                x.sort();
                x
            })
            .collect();

        let mut edge_signatures = vec![];
        let mut g = Self::new();
        for (n, _) in external_edges {
            edge_signatures.push(vec![]);
            g.add_node(n.clone());
        }

        if external_edges.is_empty() {
            edge_signatures.push(vec![]);
            g.add_node(N::default());
        }

        let mut allowed_structures = HashSet::default();
        for e in &vertex_sorted {
            for i in 0..=e.len() / 2 {
                let r = partitions(e, &[(0, i), (1, 0)], true, false);
                for (_, pp) in r {
                    for (_, v) in pp {
                        allowed_structures.insert(v);
                    }
                }
            }
        }

        let mut settings = GenerationSettingsAndInput {
            vertex_signatures: &vertex_sorted,
            allowed_structures: &allowed_structures,
            min_degree: vertex_sorted.iter().map(|x| x.len()).min().unwrap_or(0),
            max_degree: vertex_sorted.iter().map(|x| x.len()).max().unwrap_or(0),
            current_structure: vec![],
            settings: &mut settings,
        };

        let mut out = HashMap::default();
        match g.generate_impl(
            external_edges,
            0,
            &mut settings,
            &mut edge_signatures,
            &mut out,
        ) {
            Ok(()) => Ok(out),
            Err(()) => Err(out),
        }
    }

    fn generate_impl(
        &mut self,
        external_edges: &[(N, HalfEdge<E>)],
        cur_vertex: usize,
        settings: &mut GenerationSettingsAndInput<N, E>,
        edge_signatures: &mut Vec<Vec<HalfEdge<E>>>,
        out: &mut HashMap<Graph<N, E>, Integer>,
    ) -> Result<(), ()> {
        if let Some(max_vertices) = settings.settings.max_vertices
            && self.nodes.len() > max_vertices
        {
            return Ok(());
        }

        if let Some(max_loops) = settings.settings.max_loops {
            // filter based on an underestimate of the loop count
            // determine the minimal number of additional edges
            // and assume that we create one connected component
            let mut extra_edges = self
                .nodes
                .iter()
                .enumerate()
                .skip(cur_vertex)
                .map(|(i, x)| {
                    if i < external_edges.len() && x.edges.is_empty() {
                        1
                    } else if i >= external_edges.len() && x.valence < settings.min_degree {
                        settings.min_degree - x.valence
                    } else {
                        0
                    }
                })
                .sum::<usize>();
            extra_edges = extra_edges.div_ceil(2);

            let e = self.edges.len() + extra_edges + 1;
            if e > max_loops + self.nodes.len() {
                return Ok(());
            }
        }

        if let Some(f) = &settings.settings.filter_fn
            && !f(self, cur_vertex)
        {
            return Ok(());
        }

        if cur_vertex == self.nodes.len() {
            let mut spanning_tree = self.get_spanning_tree(0);

            if !spanning_tree.is_connected() {
                return Ok(());
            }

            if let Some(abort_check) = &settings.settings.abort_check
                && abort_check()
            {
                return Err(());
            }

            if settings.settings.max_bridges.is_some() || !settings.settings.allow_zero_flow_edges {
                spanning_tree.chain_decomposition();
            }

            if let Some(max_bridges) = settings.settings.max_bridges
                && spanning_tree.count_bridges() > max_bridges
            {
                return Ok(());
            }

            if !settings.settings.allow_zero_flow_edges && spanning_tree.has_zero_flow_bridges() {
                return Ok(());
            }

            let c = self.canonize();

            let mut cancel = false;
            out.entry(c.graph).or_insert_with_key(|g| {
                if let Some(p) = &mut settings.settings.progress_fn
                    && !p(g)
                {
                    cancel = true;
                }
                c.automorphism_group_size
            });

            if cancel {
                return Err(());
            }

            return Ok(());
        }

        // find completions for the current vertex
        if cur_vertex < external_edges.len() {
            // generate a single connection with the external edge
            let n = self.node(cur_vertex).valence;
            if n == 0 {
                let mut edges_left: Vec<(_, usize)> =
                    vec![(external_edges[cur_vertex].1.clone(), 1)];

                self.distribute_edges(
                    cur_vertex,
                    external_edges.len(), // do not allow connections to other external edges
                    edge_signatures,
                    external_edges,
                    &mut edges_left,
                    0,
                    settings,
                    out,
                )?;
            } else if n == 1 {
                self.generate_impl(
                    external_edges,
                    cur_vertex + 1,
                    settings,
                    edge_signatures,
                    out,
                )?;
            }

            return Ok(());
        }

        let mut cur_edges: Vec<_> = self
            .node(cur_vertex)
            .edges
            .iter()
            .map(|e| HalfEdge {
                direction: if self.edges[*e].directed {
                    Some(self.edges[*e].vertices.0 == cur_vertex)
                } else {
                    None
                },
                data: self.edge(*e).data.clone(),
            })
            .collect();
        cur_edges.sort();

        let mut edges_left: Vec<(HalfEdge<E>, usize)> = vec![];
        'next_signature: for d in settings.vertex_signatures {
            // check if the current state is compatible
            if d.len() < cur_edges.len() {
                continue;
            }

            if *d == cur_edges {
                self.generate_impl(
                    external_edges,
                    cur_vertex + 1,
                    settings,
                    edge_signatures,
                    out,
                )?;
                continue;
            }

            edges_left.clear();
            let mut edge_pos = 0;
            for e in d {
                if edge_pos < cur_edges.len() {
                    if cur_edges[edge_pos] == *e {
                        edge_pos += 1;
                        continue;
                    } else if cur_edges[edge_pos] < *e {
                        // incompatible
                        continue 'next_signature;
                    }
                }

                if let Some(last) = edges_left.last_mut()
                    && last.0 == *e
                {
                    last.1 += 1;
                    continue;
                }

                edges_left.push((e.clone(), 1));
            }

            if edge_pos < cur_edges.len() {
                // incompatible
                continue;
            }

            self.distribute_edges(
                cur_vertex,
                cur_vertex,
                edge_signatures,
                external_edges,
                &mut edges_left,
                0,
                settings,
                out,
            )?;
        }

        Ok(())
    }

    fn distribute_edges(
        &mut self,
        source: usize,
        cur_target: usize,
        edge_signatures: &mut Vec<Vec<HalfEdge<E>>>,
        external_edges: &[(N, HalfEdge<E>)],
        edge_count: &mut [(HalfEdge<E>, usize)],
        cur_edge_count_group_index: usize,
        settings: &mut GenerationSettingsAndInput<N, E>,
        out: &mut HashMap<Graph<N, E>, Integer>,
    ) -> Result<(), ()> {
        if edge_count.iter().all(|x| x.1 == 0) {
            // check if the source is not a zero-flow bridge
            //TODO: also when allowing bridges, we can filter
            // for disconnectedness if the source connects to an external node only

            if settings.settings.allow_self_loops
                && (!settings.settings.allow_zero_flow_edges
                    || settings.settings.max_bridges == Some(0))
                && (source > external_edges.len()
                    || external_edges.len() > 1 && source == external_edges.len())
                && self.node(source).edges.len()
                    - self
                        .node(source)
                        .edges
                        .iter()
                        .filter(|e| {
                            let e = self.edge(**e);
                            e.vertices.0 == source && e.vertices.1 == source
                        })
                        .count()
                    == 1
            {
                return Ok(());
            }

            // check if we created a zero-flow bridge
            // this happens when there is only one open vertex
            // that connects with a single edge to the source
            if source + 2 == self.nodes.len()
                && (settings.settings.max_bridges == Some(0)
                    || !settings.settings.allow_zero_flow_edges)
            {
                let es = &self.node(source + 1).edges;
                if es.len() == 1 && self.edge(es[0]).is_neighbor(source) {
                    return Ok(());
                }
            }

            return self.generate_impl(external_edges, source + 1, settings, edge_signatures, out);
        }

        let mut grown = false;
        if cur_target == self.nodes.len() {
            grown = true;
            edge_signatures.push(vec![]);
            self.add_node(N::default());
        } else {
            self.distribute_edges(
                source,
                cur_target + 1,
                edge_signatures,
                external_edges,
                edge_count,
                0,
                settings,
                out,
            )?;
        }

        if source == cur_target {
            if !settings.settings.allow_self_loops {
                return Ok(());
            }

            for p1 in cur_edge_count_group_index..edge_count.len() {
                if let Some(dir) = edge_count[p1].0.direction {
                    if edge_count[p1].1 == 0 {
                        continue;
                    }

                    // find the edge signature going in the other direction
                    for p2 in cur_edge_count_group_index + 1..edge_count.len() {
                        if edge_count[p2].0.direction == Some(!dir)
                            && edge_count[p1].0.data == edge_count[p2].0.data
                        {
                            if edge_count[p2].1 == 0 {
                                break;
                            }

                            edge_count[p1].1 -= 1;
                            edge_count[p2].1 -= 1;

                            self.add_edge(source, source, true, edge_count[p1].0.data.clone())
                                .unwrap();

                            self.distribute_edges(
                                source,
                                source,
                                edge_signatures,
                                external_edges,
                                edge_count,
                                p1,
                                settings,
                                out,
                            )?;

                            self.delete_last_edge();

                            edge_count[p1].1 += 1;
                            edge_count[p2].1 += 1;
                            break;
                        }
                    }
                } else {
                    let (e, count) = &mut edge_count[p1];
                    if *count < 2 {
                        continue;
                    }

                    *count -= 2;
                    self.add_edge(source, source, false, e.data.clone())
                        .unwrap();

                    self.distribute_edges(
                        source,
                        source,
                        edge_signatures,
                        external_edges,
                        edge_count,
                        p1,
                        settings,
                        out,
                    )?;

                    self.delete_last_edge();

                    edge_count[p1].1 += 2;
                }
            }

            return Ok(());
        }

        let max_degree = if cur_target < external_edges.len() {
            1
        } else {
            settings.max_degree
        };

        if self.node(cur_target).valence + 1 > max_degree {
            return Ok(());
        }

        // check if there is a previous node with the same signature
        // if so, skip assigning to this node as this is symmetric w.r.t
        // assigning to the previous node
        'next_v: for v in source + 1..cur_target {
            if self.node(v).data != self.node(cur_target).data
                || self.node(v).edges.len() != self.node(cur_target).edges.len()
            {
                continue;
            }

            for (e1, e2) in self.node(v).edges.iter().zip(&self.node(cur_target).edges) {
                let e1 = self.edge(*e1);
                let e2 = self.edge(*e2);

                if e1.data != e2.data
                    || e1.directed != e2.directed
                    || (e1.vertices.0 == v && e1.vertices.1 != e2.vertices.1)
                    || (e1.vertices.1 == v && e1.vertices.0 != e2.vertices.0)
                {
                    continue 'next_v;
                }
            }

            return Ok(());
        }

        for p in cur_edge_count_group_index..edge_count.len() {
            let (e, count) = &mut edge_count[p];

            if *count == 0 {
                continue;
            }

            if cur_target < external_edges.len() && e != &external_edges[cur_target].1 {
                continue;
            }

            // check if the target edge signature is allowed
            edge_signatures[cur_target].push(e.flip());

            if !settings.is_structure_allowed(&edge_signatures[cur_target]) {
                edge_signatures[cur_target].pop();
                continue;
            }

            *count -= 1;

            if let Some(dir) = e.direction {
                if dir {
                    self.add_edge(source, cur_target, true, e.data.clone())
                        .unwrap();
                } else {
                    self.add_edge(cur_target, source, true, e.data.clone())
                        .unwrap();
                }
            } else {
                self.add_edge(source, cur_target, false, e.data.clone())
                    .unwrap();
            }

            self.distribute_edges(
                source,
                cur_target,
                edge_signatures,
                external_edges,
                edge_count,
                p,
                settings,
                out,
            )?;

            edge_signatures[cur_target].pop();
            self.delete_last_edge(); // TODO: cache edge data

            edge_count[p].1 += 1;
        }

        if grown {
            edge_signatures.pop();
            self.delete_last_empty_node().unwrap();
        }

        Ok(())
    }
}

/// Information about the graph and its canonical form.
pub struct CanonicalForm<N, E> {
    /// Mapping of the vertices from the input graph to the canonical graph.
    pub vertex_map: Vec<usize>,
    /// Generators of the automorphism group of the graph.
    pub orbit_generators: Vec<Vec<Vec<usize>>>,
    /// An orbit of the graph.
    pub orbit: Vec<usize>,
    /// The size of the automorphism group of the graph, including
    /// the permutations stemming from identical edges.
    pub automorphism_group_size: Integer,
    pub graph: Graph<N, E>,
}

impl<N: Clone + PartialOrd + Ord + Eq + Hash, E: Clone + PartialOrd + Ord + Eq + Hash> Graph<N, E> {
    /// Canonize the graph using McKay's canonical graph labeling algorithm,
    /// returning the vertex mapping and the canonical form.
    pub fn canonize(&self) -> CanonicalForm<N, E> {
        if self.nodes.is_empty() {
            return CanonicalForm {
                vertex_map: vec![],
                orbit_generators: vec![],
                orbit: vec![],
                automorphism_group_size: Integer::one(),
                graph: self.clone(),
            };
        }

        if self.nodes.len() <= u16::MAX as usize {
            self.canonize_impl::<u16>(false)
        } else if self.nodes.len() <= u32::MAX as usize {
            self.canonize_impl::<u32>(false)
        } else {
            self.canonize_impl::<usize>(false)
        }
    }

    fn canonize_impl<I: NodeIndex>(&self, verbose: bool) -> CanonicalForm<N, E> {
        let mut stack = vec![SearchTreeNode::<I>::new(self)];
        let mut automorphisms = vec![];
        let mut minimal_representatives_per_generator = vec![];
        let mut leaf_nodes: HashMap<_, (Vec<_>, Vec<_>)> = HashMap::default(); // TODO: limit growth
        let mut left_leaf_invariant: Option<Vec<Invariant<I>>> = None;
        let mut current_best: Option<(Graph<&N, &E>, Vec<I>, Vec<Invariant<I>>)> = None;

        let mut node_buffer = vec![];

        let mut automorphism_group_len = Integer::one();
        let mut orbit = (0..self.nodes.len()).collect::<Vec<_>>();

        while let Some(mut node) = stack.pop() {
            if node.selected_vertex.is_none() {
                node.refine(self);
            }

            // the canonical form is defined as the maximal isomorph, prepended with the node invariants of the path
            // at each tree level, the node invariant must therefore be at least as good as the best
            // to be a potential canonical form
            let mut on_best_path = true;
            if let Some((_, _, best_invariant)) = &current_best {
                // do not skip exploring the current node if its invariant is the same as the left leaf invariant
                // this ensures that the size of the automorphism group is not underestimated
                let left_leaf_invariant = left_leaf_invariant.as_ref().unwrap();
                let left_leaf_equivalent = stack.len() < left_leaf_invariant.len()
                    && node.invariant == left_leaf_invariant[stack.len()];

                // we are on the best path if the ancestral invariants are the same and if the current invariants
                // list is not larger, as we prefer shorter paths
                on_best_path = stack.len() < best_invariant.len()
                    && stack
                        .iter()
                        .zip(best_invariant)
                        .all(|s| &s.0.invariant == s.1);

                if on_best_path {
                    // we are on the current best path, so check if the current node invariant is better
                    match node.invariant.cmp(&best_invariant[stack.len()]) {
                        Ordering::Less => {
                            on_best_path = false;
                        }
                        Ordering::Greater => {
                            // we will find a better isomorph on this path, so disable any future
                            // invariant checks until we reach the next leaf
                            current_best = None;
                        }
                        Ordering::Equal => {}
                    }
                }

                if !on_best_path && !left_leaf_equivalent {
                    // we are not on a left leaf equivalent path and the path is not the best, so we can skip this node
                    node_buffer.push(node);
                    continue;
                }
            }

            if node.partition.iter().all(|x| x.len() == 1) {
                let partition: Vec<_> = node.partition.iter().map(|x| x[0]).collect();

                let mut g = Graph::new();
                for i in &partition {
                    g.add_node(&self.node(i.to_usize()).data);
                }
                let mut edges = vec![];
                for e in &self.edges {
                    let a = partition
                        .iter()
                        .position(|&x| x.to_usize() == e.vertices.0)
                        .unwrap();
                    let b = partition
                        .iter()
                        .position(|&x| x.to_usize() == e.vertices.1)
                        .unwrap();

                    if e.directed || a <= b {
                        edges.push((a, b, e.directed, &e.data));
                    } else {
                        edges.push((b, a, e.directed, &e.data));
                    }
                }
                edges.sort();
                for (v1, v2, dir, d) in edges {
                    g.add_edge(v1, v2, dir, d).unwrap();
                }

                // update the best isomorph
                if let Some((b, p, inv)) = &mut current_best {
                    if on_best_path && *b > g {
                        *b = g.clone();
                        *p = partition.clone();
                        inv.extend(stack.iter().map(|x| x.invariant.clone()));
                        inv.push(node.invariant.clone());
                    }
                } else {
                    let mut cur_invariant: Vec<_> =
                        stack.iter().map(|x| x.invariant.clone()).collect();
                    cur_invariant.push(node.invariant.clone());

                    if left_leaf_invariant.is_none() {
                        left_leaf_invariant = Some(cur_invariant.clone());
                    }

                    current_best = Some((g.clone(), partition.clone(), cur_invariant));
                }

                let path: Vec<_> = stack.iter().map(|x| x.selected_vertex.unwrap()).collect();
                if let Some((old_partition, old_path)) = leaf_nodes.get(&g) {
                    // construct the automorphism transformation
                    let mut seen = vec![false; partition.len()];

                    let mut fixed = vec![];
                    let mut minimal_representatives = vec![];
                    let mut orbits = vec![];

                    for x in old_partition {
                        let mut cur: I = *x;
                        if seen[cur.to_usize()] {
                            continue;
                        }

                        let mut orbit = vec![cur];
                        loop {
                            cur = partition[old_partition.iter().position(|y| y == &cur).unwrap()];
                            if cur == *x {
                                break;
                            } else {
                                seen[cur.to_usize()] = true;
                                orbit.push(cur);
                            }
                        }

                        if orbit.len() == 1 {
                            fixed.push(orbit[0]);
                        } else {
                            // only store the minimal representative per orbit
                            minimal_representatives.push(*orbit.iter().min().unwrap());
                            orbits.push(orbit);
                        }
                    }

                    orbits.sort();
                    minimal_representatives_per_generator.push((fixed, minimal_representatives));
                    automorphisms.push(orbits);

                    // fall back to common ancestor in the search tree
                    let mut i = 0;
                    for (p1, p2) in old_path.iter().zip(path.iter()) {
                        if p1 == p2 {
                            i += 1;
                        } else {
                            break;
                        }
                    }

                    // we will pop an extra node at the start of the next loop, hence + 1
                    node_buffer.extend(stack.drain(i + 1..));
                } else {
                    // extend the list of terminal nodes in the hopes of finding new automorphisms faster
                    leaf_nodes.insert(g, (partition, path));
                    node_buffer.push(node);
                }
            } else {
                let (x, p) = if let Some(p) = node.selected_part {
                    // upon a repeat visit, filter the list of possible children with
                    // the automorphism group, taking the smallest out of every orbit
                    // for non-left node children, we apply a simpler filter that does
                    // not combine orbits and only takes the smallest vertex per orbit

                    for (i, o) in orbit.iter_mut().enumerate() {
                        *o = i;
                    }

                    if node.left_node {
                        // TODO: we can update the orbit globally, as there is only one left path
                        for s_orbits in automorphisms.iter() {
                            // filter orbits that do not fix the vertices that are fixed by the selected path
                            if stack
                                .iter()
                                .map(|x| x.selected_vertex.unwrap())
                                .any(|x| s_orbits.iter().any(|o| o.contains(&x)))
                            {
                                continue;
                            }

                            for s_orbits in s_orbits {
                                // find minimal representatives for each orbit
                                let min =
                                    s_orbits.iter().map(|x| orbit[x.to_usize()]).min().unwrap();
                                for a in s_orbits {
                                    let old_val = orbit[a.to_usize()];

                                    if old_val == min {
                                        continue;
                                    }

                                    for o in &mut orbit {
                                        if *o == old_val {
                                            *o = min;
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        for (fixed, reps) in &minimal_representatives_per_generator {
                            if stack
                                .iter()
                                .all(|x| fixed.contains(&x.selected_vertex.unwrap()))
                            {
                                node.children_to_visit
                                    .retain(|x| reps.contains(x) || fixed.contains(x));
                            }
                        }
                    }

                    let orig = node.orig_selected_vertex.unwrap().to_usize();

                    // check if the last vertex we tried turned out to be in the same orbit as the first one
                    // due to newly found automorphisms
                    if node.left_node && orbit[node.selected_vertex.unwrap().to_usize()] == orig {
                        node.children_visited_equal_to_first += 1;
                    }

                    node.selected_vertex = None;
                    while !node.children_to_visit.is_empty() {
                        // individualize x
                        let x = node.children_to_visit.remove(0);
                        if node.left_node && orbit[x.to_usize()] == orig {
                            node.children_visited_equal_to_first += 1;
                        }

                        if orbit[x.to_usize()] == x.to_usize() {
                            node.selected_vertex = Some(x);
                            break;
                        }
                    }

                    if node.children_to_visit.is_empty() && node.selected_vertex.is_none() {
                        // use the left-most path to determine the automorphism group size,
                        // by applying the orbit-stabilizer theorem; we determined the number of child
                        // vertices that are in the same orbit as the first child vertex
                        // multiplying this number at every level gives the automorphism group size, since
                        // at every level we stabilize the first vertex
                        if node.left_node {
                            if verbose {
                                println!(
                                    "Level={}, group size={}, orbit num={}",
                                    stack.len(),
                                    node.children_visited_equal_to_first,
                                    orbit.iter().enumerate().filter(|(i, x)| *i == **x).count()
                                );
                            }

                            automorphism_group_len *= node.children_visited_equal_to_first as u64;
                        }

                        node_buffer.push(node);
                        continue;
                    }

                    (node.selected_vertex.unwrap(), p.to_usize())
                } else {
                    // find the first minimal length non-trivial part to individualize
                    let smallest = node
                        .partition
                        .iter()
                        .map(|x| x.len())
                        .filter(|x| *x > 1)
                        .min()
                        .unwrap();

                    let (p, part) = node
                        .partition
                        .iter()
                        .enumerate()
                        .find(|x| x.1.len() == smallest)
                        .unwrap();

                    node.children_to_visit = part.clone();
                    node.selected_part = Some(I::from_usize(p));

                    // individualize x
                    let x = node.children_to_visit.remove(0);
                    node.selected_vertex = Some(x);

                    (x, p)
                };

                let mut new_node = match node_buffer.pop() {
                    Some(mut n) => {
                        n.partition.clear();
                        n.children_to_visit.clear();
                        n.children_visited_equal_to_first = 0;
                        n.orig_selected_vertex = None;
                        n.selected_part = None;
                        n.selected_vertex = None;
                        n
                    }
                    None => SearchTreeNode::default(),
                };

                new_node.left_node = node.left_node && node.orig_selected_vertex.is_none();

                if node.orig_selected_vertex.is_none() {
                    node.orig_selected_vertex = Some(x);
                }

                new_node
                    .partition
                    .extend(node.partition.iter().take(p).cloned());
                let part = &node.partition[p.to_usize()];
                new_node.partition.push(vec![x]);
                new_node
                    .partition
                    .push(part.iter().filter(|&y| *y != x).cloned().collect());
                new_node
                    .partition
                    .extend(node.partition.iter().skip(p + 1).cloned());

                stack.push(node);
                stack.push(new_node);
            }
        }

        let (best, map, _) = current_best.unwrap();
        let mut g = Graph::new();
        for i in 0..best.nodes.len() {
            g.add_node(best.node(i).data.clone());
        }
        for e in &best.edges {
            g.add_edge(e.vertices.0, e.vertices.1, e.directed, e.data.clone())
                .unwrap();
        }

        let inv_map: Vec<_> = (0..self.nodes.len())
            .map(|x| {
                map.iter()
                    .position(|y| y.to_usize() == x)
                    .unwrap()
                    .to_usize()
            })
            .collect();

        // transform the automorphisms to the new vertex numbering
        let automorphisms: Vec<_> = automorphisms
            .into_iter()
            .map(|x| {
                x.into_iter()
                    .map(|y| {
                        y.into_iter()
                            .map(|z| inv_map[z.to_usize()])
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            })
            .collect();

        let orbit = (0..orbit.len())
            .map(|x| inv_map[orbit[map[x].to_usize()]])
            .collect();

        CanonicalForm {
            vertex_map: inv_map,
            orbit_generators: automorphisms,
            automorphism_group_size: automorphism_group_len * g.get_edge_automorphism_group_size(),
            orbit,
            graph: g,
        }
    }

    /// Returns `true` iff the graph is isomorphic to `other`.
    pub fn is_isomorphic(&self, other: &Self) -> bool {
        if self.nodes.len() != other.nodes.len() || self.edges.len() != other.edges.len() {
            return false;
        }

        if self == other {
            return true;
        }

        let mut node = SearchTreeNode::<usize>::new(self);
        let mut other_node = SearchTreeNode::new(other);

        if node.partition.len() != other_node.partition.len() {
            return false;
        }

        // check if the vertex colors are the same
        for (i, j) in node.partition.iter().zip(&other_node.partition) {
            if i.len() != j.len() || self.node(i[0]).data != other.node(j[0]).data {
                return false;
            }
        }

        // check if the first refinement has the same node invariant
        node.refine(self);
        other_node.refine(other);

        if node.invariant != other_node.invariant {
            return false;
        }

        let other_canon = other.canonize().graph;

        if *self == other_canon {
            return true;
        }

        self.canonize().graph == other_canon
    }
}

/// An index type that can be used to index nodes in a graph.
/// Used to save memory for small graphs.
trait NodeIndex: Default + Copy + PartialOrd + Ord + Eq + Hash + Display + Debug {
    fn to_usize(&self) -> usize;
    fn from_usize(x: usize) -> Self;
}

impl NodeIndex for usize {
    #[inline(always)]
    fn to_usize(&self) -> usize {
        *self
    }

    #[inline(always)]
    fn from_usize(x: usize) -> Self {
        x
    }
}

impl NodeIndex for u32 {
    #[inline(always)]
    fn to_usize(&self) -> usize {
        *self as usize
    }

    #[inline(always)]
    fn from_usize(x: usize) -> Self {
        debug_assert!(x <= u32::MAX as usize);
        x as u32
    }
}

impl NodeIndex for u16 {
    #[inline(always)]
    fn to_usize(&self) -> usize {
        *self as usize
    }

    #[inline(always)]
    fn from_usize(x: usize) -> Self {
        debug_assert!(x <= u16::MAX as usize);
        x as u16
    }
}

/// A node invariant.
#[derive(Default, Debug, Clone, PartialEq, Eq)]
struct Invariant<N: NodeIndex> {
    partition_lengths: Vec<N>,
}

impl<N: NodeIndex> PartialOrd for Invariant<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<N: NodeIndex> Ord for Invariant<N> {
    fn cmp(&self, other: &Self) -> Ordering {
        // compare length first and then lexicographically
        (self.partition_lengths.len(), &self.partition_lengths)
            .cmp(&(other.partition_lengths.len(), &other.partition_lengths))
    }
}

/// A search tree node used for canonization.
#[derive(Default)]
struct SearchTreeNode<N: NodeIndex> {
    partition: Vec<Vec<N>>,
    selected_part: Option<N>,
    selected_vertex: Option<N>,
    children_to_visit: Vec<N>,
    left_node: bool,
    orig_selected_vertex: Option<N>,
    children_visited_equal_to_first: usize,
    invariant: Invariant<N>,
}

impl<I: NodeIndex> SearchTreeNode<I> {
    /// Create a new search tree node with a partition refined on vertex color.
    fn new<N: PartialOrd + Ord + Hash, E>(g: &Graph<N, E>) -> Self {
        let mut h = HashMap::default();
        for (i, x) in g.nodes.iter().enumerate() {
            h.entry(&x.data)
                .or_insert_with(Vec::new)
                .push(I::from_usize(i));
        }
        let mut partition: Vec<_> = h.into_iter().collect();
        partition.sort_by_key(|x| x.0);

        SearchTreeNode {
            partition: partition.into_iter().map(|x| x.1).collect(),
            selected_part: None,
            selected_vertex: None,
            children_to_visit: vec![],
            orig_selected_vertex: None,
            left_node: true,
            children_visited_equal_to_first: 0,
            invariant: Invariant::default(),
        }
    }

    /// Compute a node invariant.
    fn update_invariant(&mut self) {
        self.invariant.partition_lengths.clear();
        // TODO: improve invariant
        self.invariant
            .partition_lengths
            .extend(self.partition.iter().map(|x| I::from_usize(x.len())));
    }

    /// Apply a label-invariant refinement to the partition.
    /// The current refinement is based on the signature of the neighbors of a vertex.
    /// The signature takes directional edges and edge colors into account.
    fn refine<N, E: PartialOrd + Ord>(&mut self, g: &Graph<N, E>) {
        let largest_partition = self.partition.iter().map(|x| x.len()).max().unwrap();
        let mut degrees = vec![(vec![], I::from_usize(0)); largest_partition];

        let mut last_stable_index = 0; // no splits happened before this index in the last round
        'next: loop {
            for (ii, i) in self.partition.iter().enumerate() {
                if i.len() == 1 {
                    continue;
                }

                for (jj, j) in self.partition.iter().enumerate() {
                    if ii < last_stable_index && jj < last_stable_index {
                        // this part is already tested and stable
                        continue;
                    }

                    // sorted edge colors of edges in i that connect to vertices in j
                    // the length of this vector is the degree of the vertex i in j
                    for ((edge_data, vert), v) in degrees.iter_mut().zip(i) {
                        edge_data.clear();

                        for e in &g.node(v.to_usize()).edges {
                            let e = g.edge(*e);
                            let (k, is_source) = if e.vertices.0 == v.to_usize() {
                                (NodeIndex::from_usize(e.vertices.1), true)
                            } else {
                                (NodeIndex::from_usize(e.vertices.0), false)
                            };
                            // TODO: add special flag for self-edge and tag multi-edges differently
                            if j.contains(&k) {
                                if e.directed {
                                    edge_data.push((&e.data, e.directed, is_source));
                                } else {
                                    edge_data.push((&e.data, false, false));
                                }
                            }
                        }
                        edge_data.sort_unstable();
                        *vert = *v;
                    }

                    if degrees[..i.len()].windows(2).all(|w| w[0].0 == w[1].0) {
                        continue;
                    }

                    degrees[..i.len()].sort_unstable();

                    let mut degs = vec![];
                    let mut cur = vec![degrees[0].1];
                    for x in degrees[..i.len()].windows(2) {
                        if x[0].0 != x[1].0 {
                            degs.push(cur);
                            cur = vec![x[1].1];
                        } else {
                            cur.push(x[1].1);
                        }
                    }
                    degs.push(cur);

                    self.partition.splice(ii..=ii, degs);
                    last_stable_index = ii;
                    continue 'next;
                }
            }

            break;
        }

        self.update_invariant();
    }
}

#[cfg(test)]
mod test {
    use super::{GenerationSettings, Graph, HalfEdge, SearchTreeNode};
    use numerica::combinatorics::unique_permutations;

    #[test]
    fn directed() {
        let mut g = Graph::new();
        let n0 = g.add_node(0);
        let n1 = g.add_node(0);
        g.add_edge(n0, n1, false, 0).unwrap();
        g.add_edge(n0, n1, false, 1).unwrap();

        let mut node = SearchTreeNode::<usize>::default();
        node.partition = vec![vec![0, 1]];
        node.refine(&g);
        assert_eq!(node.partition.len(), 1); // cannot refine

        let mut g = Graph::new();
        let n0 = g.add_node(0);
        let n1 = g.add_node(0);
        g.add_edge(n0, n1, true, 0).unwrap();
        g.add_edge(n0, n1, false, 1).unwrap();

        let mut node = SearchTreeNode::<usize>::default();
        node.partition = vec![vec![0, 1]];
        node.refine(&g);
        assert_eq!(node.partition.len(), 2); // distinguish two nodes based on directed edge
    }

    #[test]
    fn isomorphic() {
        let mut g = Graph::new();
        let n0 = g.add_node(0);
        g.add_edge(n0, n0, false, 0).unwrap();
        g.add_edge(n0, n0, false, 0).unwrap();

        let mut g1 = Graph::new();
        let n0 = g1.add_node(0);
        g1.add_edge(n0, n0, false, 0).unwrap();

        assert!(!g.is_isomorphic(&g1));

        g1.add_edge(n0, n0, true, 0).unwrap();
        assert!(!g.is_isomorphic(&g1));

        g.add_edge(n0, n0, true, 0).unwrap();
        g1.add_edge(n0, n0, false, 0).unwrap();
        assert!(g.is_isomorphic(&g1));

        let _ = g.add_node(1);
        let _ = g1.add_node(0);
        assert!(!g.is_isomorphic(&g1));

        let _ = g1.add_node(1);
        let _ = g.add_node(0);
        assert!(g.is_isomorphic(&g1));
    }

    #[test]
    fn canonize() {
        let mut g = Graph::new();
        let n0 = g.add_node(1);
        let n1 = g.add_node(0);
        let n2 = g.add_node(1);
        let n3 = g.add_node(0);
        let n4 = g.add_node(2);
        let n5 = g.add_node(0);
        let n6 = g.add_node(1);
        let n7 = g.add_node(0);
        let n8 = g.add_node(1);

        g.add_edge(n0, n1, false, 0).unwrap();
        g.add_edge(n0, n3, false, 0).unwrap();
        g.add_edge(n1, n2, false, 0).unwrap();
        g.add_edge(n1, n3, false, 0).unwrap();
        g.add_edge(n1, n4, false, 0).unwrap();
        g.add_edge(n1, n5, false, 0).unwrap();
        g.add_edge(n2, n5, false, 0).unwrap();
        g.add_edge(n3, n4, false, 0).unwrap();
        g.add_edge(n3, n6, false, 0).unwrap();
        g.add_edge(n3, n7, false, 0).unwrap();
        g.add_edge(n4, n5, false, 0).unwrap();
        g.add_edge(n4, n7, false, 0).unwrap();
        g.add_edge(n5, n7, false, 0).unwrap();
        g.add_edge(n5, n8, false, 0).unwrap();
        g.add_edge(n6, n7, false, 0).unwrap();
        g.add_edge(n7, n8, false, 0).unwrap();

        let c = g.canonize();

        assert_eq!(c.orbit_generators.len(), 2);
        assert_eq!(c.automorphism_group_size, 8);
        assert_eq!(c.graph.edge(0).vertices, (0, 2));
    }

    #[test]
    pub fn iso_check_permutations() {
        let mut gc = None;
        for p in unique_permutations(&[0, 1, 2, 3, 4]).1 {
            let mut g = Graph::new();
            for _ in 0..5 {
                g.add_node(0);
            }

            g.add_edge(p[0], p[0], false, 0).unwrap();
            g.add_edge(p[0], p[3], false, 0).unwrap();
            g.add_edge(p[0], p[4], false, 0).unwrap();
            g.add_edge(p[1], p[2], false, 0).unwrap();
            g.add_edge(p[1], p[3], false, 0).unwrap();
            g.add_edge(p[1], p[4], false, 0).unwrap();
            g.add_edge(p[2], p[3], false, 0).unwrap();
            g.add_edge(p[2], p[4], false, 0).unwrap();
            g.add_edge(p[3], p[4], false, 0).unwrap();
            let c = g.canonize();

            if let Some(gg) = &gc {
                if &c.graph != gg {
                    panic!("Inequivalent isomorph with permutation {p:?}");
                }
            } else {
                gc = Some(c.graph);
            }
        }
    }

    #[test]
    fn canonize_edges() {
        let mut g = Graph::new();
        let n0 = g.add_node(0);
        let n1 = g.add_node(1);
        let n2 = g.add_node(2);

        g.add_edge(n2, n1, true, 0).unwrap();
        g.add_edge(n0, n0, false, 0).unwrap();
        g.add_edge(n0, n1, true, 0).unwrap();
        g.add_edge(n1, n0, false, 2).unwrap();

        g.canonize_edges();

        assert_eq!(g.node(0).edges, [0, 1, 2]);
    }

    #[test]
    fn generate() {
        let g = HalfEdge::undirected("g");
        let q = HalfEdge::incoming("q");
        let gs = Graph::<_, &str>::generate(
            &[(1, g), (2, g)],
            &[vec![g, g, g], vec![q.flip(), q, g], vec![g, g, g, g]],
            GenerationSettings::new()
                .max_loops(3)
                .max_bridges(0)
                .allow_self_loops(true),
        )
        .unwrap();

        assert_eq!(gs.len(), 210);
    }

    #[test]
    fn generate_tree() {
        let g = HalfEdge::undirected("g");

        let graphs = Graph::generate(
            &[(1, g), (2, g), (3, g), (4, g)],
            &[vec![g, g, g], vec![g, g, g, g]],
            GenerationSettings::new()
                .max_loops(2)
                .max_bridges(0)
                .allow_self_loops(true),
        )
        .unwrap();

        assert_eq!(graphs.len(), 278);
    }

    #[test]
    fn generate_with_filter() {
        let g = HalfEdge::undirected("g");
        let q = HalfEdge::incoming("q");

        let graphs = Graph::<_, &str>::generate(
            &[(1, g), (2, g)],
            &[vec![g, g, g], vec![q.flip(), q, g], vec![g, g, g, g]],
            GenerationSettings::new()
                .max_loops(4)
                .max_bridges(0)
                .allow_self_loops(true)
                .filter_fn(Box::new(|g, c| {
                    // each graph must have at most one node of type g-g-g
                    let mut count = 0;
                    for n in 0..c {
                        // loop only through frozen nodes
                        let node = g.node(n);
                        if node.edges.len() == 3
                            && node.edges.iter().all(|e| g.edge(*e).data == "g")
                        {
                            count += 1;
                            if count == 2 {
                                return false;
                            }
                        }
                    }

                    true
                })),
        )
        .unwrap();

        assert_eq!(graphs.len(), 845);
    }
}
