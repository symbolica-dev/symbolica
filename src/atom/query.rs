//! Reusable structural queries with explicit opaque expression boundaries.

use super::{AtomCore, AtomType, AtomView, HashSet};

/// Require an exact atom and/or a syntactic kind without rewriting an expression.
///
/// Each configured set is an OR requirement; the atom and kind requirements
/// are combined with AND. Omitted requirements are satisfied automatically,
/// while an explicitly empty requirement never matches. Exact atoms are checked
/// before opaque boundaries, whose internal structure and kind are not inspected.
/// No function definitions are resolved, and no algebraic equivalence is inferred.
#[derive(Clone, Debug)]
pub struct AtomQuery<'a> {
    atoms: Option<HashSet<AtomView<'a>>>,
    kinds: Option<HashSet<AtomType>>,
    opaque: HashSet<AtomView<'a>>,
    max_atom_bytes: usize,
    max_opaque_bytes: usize,
    enter_functions: bool,
}

impl Default for AtomQuery<'_> {
    fn default() -> Self {
        Self {
            atoms: None,
            kinds: None,
            opaque: HashSet::default(),
            max_atom_bytes: 0,
            max_opaque_bytes: 0,
            enter_functions: true,
        }
    }
}

impl<'a> AtomQuery<'a> {
    /// Create a query with no requirements or opaque boundaries.
    pub fn new() -> Self {
        Self::default()
    }

    /// Require at least one of these exact atoms, including at opaque boundaries.
    pub fn require_any_atom<A: AtomCore + 'a>(
        mut self,
        atoms: impl IntoIterator<Item = &'a A>,
    ) -> Self {
        let atoms = atoms
            .into_iter()
            .map(AtomCore::as_atom_view)
            .collect::<HashSet<_>>();
        self.max_atom_bytes = atoms
            .iter()
            .map(|atom| atom.get_byte_size())
            .max()
            .unwrap_or(0);
        self.atoms = Some(atoms);
        self
    }

    /// Require at least one node of these kinds outside opaque boundaries.
    pub fn require_any_kind(mut self, kinds: impl IntoIterator<Item = AtomType>) -> Self {
        self.kinds = Some(kinds.into_iter().collect());
        self
    }

    /// Treat these exact expressions as atomic inputs, including compound keys.
    pub fn opaque_inputs<A: AtomCore + 'a>(
        mut self,
        inputs: impl IntoIterator<Item = &'a A>,
    ) -> Self {
        self.opaque = inputs.into_iter().map(AtomCore::as_atom_view).collect();
        self.max_opaque_bytes = self
            .opaque
            .iter()
            .map(|atom| atom.get_byte_size())
            .max()
            .unwrap_or(0);
        self
    }

    /// Visit function arguments. Function nodes themselves remain visible unless opaque.
    pub fn enter_functions(mut self, enter: bool) -> Self {
        self.enter_functions = enter;
        self
    }

    /// Test the configured requirements in one traversal, stopping when all hold.
    pub fn matches<A: AtomCore>(&self, expression: &A) -> bool {
        let mut atom_found = self.atoms.is_none();
        let mut kind_found = self.kinds.is_none();
        if atom_found && kind_found {
            return true;
        }
        if self.atoms.as_ref().is_some_and(HashSet::is_empty)
            || self.kinds.as_ref().is_some_and(HashSet::is_empty)
        {
            return false;
        }
        let mut pending = vec![expression.as_atom_view()];
        while let Some(view) = pending.pop() {
            let bytes = view.get_byte_size();
            atom_found |= bytes <= self.max_atom_bytes
                && self
                    .atoms
                    .as_ref()
                    .is_some_and(|atoms| atoms.contains(&view));
            if bytes <= self.max_opaque_bytes && self.opaque.contains(&view) {
                if atom_found && kind_found {
                    return true;
                }
                continue;
            }
            kind_found |= self
                .kinds
                .as_ref()
                .is_some_and(|kinds| kinds.contains(&view.get_atom_type()));
            if atom_found && kind_found {
                return true;
            }
            match view {
                AtomView::Add(sum) => pending.extend(sum.iter()),
                AtomView::Mul(product) => pending.extend(product.iter()),
                AtomView::Pow(power) => pending.extend(power.iter()),
                AtomView::Fun(function) if self.enter_functions => {
                    pending.extend(function.iter());
                }
                _ => {}
            }
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn exact_formals_are_visible_before_opaque_function_boundaries() {
        let q = parse!("atom_query::q(2,0)");
        let x = parse!("atom_query::x");
        let opaque = &q + &x;
        let inputs = [q.clone(), opaque.clone()];
        let query = AtomQuery::new()
            .require_any_atom([&q])
            .require_any_kind([AtomType::Add])
            .opaque_inputs(&inputs)
            .enter_functions(false);
        assert!(query.matches(&(&q + 1)));
        assert!(!query.matches(&opaque));
        assert!(!query.matches(&opaque.pow(2)));
        assert!(!query.matches(&parse!("atom_query::f(atom_query::q(2,0)+1)")));
        assert!(query.matches(&(&q + opaque.pow(2))));
        assert!(
            query
                .clone()
                .enter_functions(true)
                .matches(&parse!("atom_query::f(atom_query::q(2,0)+1)"))
        );
    }

    #[test]
    fn requirements_are_or_sets_combined_with_and() {
        let x = parse!("atom_query_sets::x");
        let y = parse!("atom_query_sets::y");
        let query = AtomQuery::new()
            .require_any_atom([&x, &y])
            .require_any_kind([AtomType::Add, AtomType::Pow]);
        assert!(query.matches(&x.pow(2)));
        assert!(query.matches(&(&y + 1)));
        assert!(!query.matches(&(&x * &y)));
        assert!(!query.matches(&parse!("atom_query_sets::z+1")));
        assert!(
            !AtomQuery::new()
                .require_any_atom::<crate::atom::Atom>([])
                .matches(&x)
        );
        assert!(!AtomQuery::new().require_any_kind([]).matches(&x));
        assert!(AtomQuery::new().matches(&x));
    }

    #[test]
    fn function_arguments_can_be_opaque_without_hiding_the_call() {
        let query =
            AtomQuery::new().require_any_kind([AtomType::Add, AtomType::Mul, AtomType::Pow]);
        for (expression, expected) in [
            ("x", false),
            ("f(x)", false),
            ("f(x+1)", true),
            ("(x+1)^2", true),
        ] {
            assert_eq!(query.matches(&parse!(expression)), expected);
        }
        let function = parse!("atom_query_function::f(atom_query_function::x+1)");
        assert!(!query.enter_functions(false).matches(&function));
        assert!(
            AtomQuery::new()
                .require_any_kind([AtomType::Fun])
                .enter_functions(false)
                .matches(&function)
        );
    }
}
