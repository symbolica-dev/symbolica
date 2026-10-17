//! Prepare independently owned alias DAGs without expanding their definitions.

use std::sync::atomic::{AtomicUsize, Ordering};

use super::*;
use crate::atom::{NamespacedSymbol, Symbol, SymbolBuilder};

impl AliasedAtom {
    pub(crate) fn fresh_alias_symbol() -> Symbol {
        static NEXT_SCOPE: AtomicUsize = AtomicUsize::new(0);
        loop {
            let scope = NEXT_SCOPE.fetch_add(1, Ordering::Relaxed);
            let name = NamespacedSymbol::parse(&format!("symbolica::alias_scope_{scope}"));
            if Symbol::get_symbol(name.clone()).is_none() {
                return SymbolBuilder::new(name).build().unwrap();
            }
        }
    }

    /// Give all owned alias handles a fresh private scope in one pass.
    ///
    /// Definitions remain shared and are not expanded. A forwarding definition
    /// `alias = alias` keeps its original right-hand side, which can refer to an
    /// external parameter after the local handle has been renamed.
    pub fn scope_aliases(self) -> Self {
        if self.aliases.is_empty() {
            return self;
        }
        let symbol = Self::fresh_alias_symbol();
        let mut handles = self.aliases.keys().collect::<Vec<_>>();
        handles.sort();
        let renames = handles
            .into_iter()
            .enumerate()
            .map(|(index, handle)| (handle.as_view(), symbol.call_args([Atom::num(index)])))
            .collect::<HashMap<_, _>>();
        let max_bytes = renames.keys().map(|key| key.get_byte_size()).max().unwrap();
        let rename = |atom: &Atom| {
            atom.replace_map(|view, _, out| {
                if view.get_byte_size() <= max_bytes
                    && let Some(handle) = renames.get(&view)
                {
                    out.set_from_view(&handle.as_view());
                }
            })
        };
        Self {
            root: rename(&self.root),
            aliases: self
                .aliases
                .iter()
                .map(|(handle, body)| {
                    (
                        renames[&handle.as_view()].clone(),
                        if handle == body {
                            body.clone()
                        } else {
                            rename(body)
                        },
                    )
                })
                .collect(),
        }
    }

    /// Transform the root and each stored definition without unfolding aliases.
    /// Alias handles themselves retain their identity.
    pub fn map_expressions(self, mut map: impl FnMut(Atom) -> Atom) -> Self {
        Self {
            root: map(self.root),
            aliases: self
                .aliases
                .into_iter()
                .map(|(handle, body)| (handle, map(body)))
                .collect(),
        }
    }

    /// Transform the root and only the alias definitions reached afterwards.
    ///
    /// Each body is transformed before discovering its dependencies. This lets
    /// static branch selection discard unused singular definitions without
    /// touching them. Definitions remain shared, handles remain opaque, and
    /// each reached body is visited once, including cyclic graphs. Forwarding
    /// definitions retain their external right-hand side for later scoping.
    pub fn map_reachable_expressions(mut self, mut map: impl FnMut(Atom) -> Atom) -> Self {
        let root = map(std::mem::take(&mut self.root));
        let max_bytes = self
            .aliases
            .keys()
            .map(|key| key.as_view().get_byte_size())
            .max()
            .unwrap_or(0);
        let mut pending = self
            .direct_alias_dependencies(&root, max_bytes)
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        let mut aliases = HashMap::default();
        while let Some(handle) = pending.pop() {
            let Some(body) = self.aliases.remove(&handle) else {
                continue;
            };
            let body = if body == handle { body } else { map(body) };
            pending.extend(
                self.direct_alias_dependencies(&body, max_bytes)
                    .into_iter()
                    .cloned(),
            );
            aliases.insert(handle, body);
        }
        Self { root, aliases }
    }

    pub(super) fn direct_alias_dependencies<'a>(
        &'a self,
        atom: &Atom,
        max_bytes: usize,
    ) -> Vec<&'a Atom> {
        let mut dependencies = Vec::new();
        atom.visitor(&mut |view| {
            if view.get_byte_size() <= max_bytes
                && let Some((handle, _)) = self.aliases.get_key_value(view.get_data())
            {
                dependencies.push(handle);
                false
            } else {
                true
            }
        });
        dependencies
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    #[test]
    fn independently_scoped_aliases_keep_shared_definitions() {
        let key = parse!("alias_scope_test::key");
        let outer = parse!("alias_scope_test::outer(0)");
        let x = parse!("alias_scope_test::x");
        let mut first = AliasedAtom::from(&outer + &key);
        first.register_alias(outer.clone(), key.pow(2));
        first.register_alias(key.clone(), &x + 1);
        let expected = first.clone().into_inner();
        let scoped = first.clone().scope_aliases();
        assert_eq!(scoped.get_aliases().len(), 2);
        assert_eq!(scoped.clone().into_inner(), expected);
        assert!(
            scoped
                .get_aliases()
                .keys()
                .all(|handle| !first.get_aliases().contains_key(handle))
        );
        let mut second = AliasedAtom::from(key.clone());
        second.register_alias(key, &x + 7);
        let combined = scoped.try_add(&second.scope_aliases()).unwrap();
        assert_eq!(combined.into_inner(), expected + &x + 7);
    }

    #[test]
    fn scoped_forwarding_alias_preserves_the_external_input() {
        let input = parse!("alias_scope_test::input(1)");
        let mut source = AliasedAtom::from(input.pow(2));
        source.register_alias(input.clone(), input.clone());
        let scoped = source.scope_aliases();
        assert_eq!(scoped.clone().into_inner(), input.pow(2));
        let mapped = scoped.map_expressions(|atom| atom.replace(input.clone()).with(3));
        assert_eq!(mapped.into_inner(), Atom::num(9));
    }

    #[test]
    fn reachable_rewrites_discard_dead_poles_and_cycles_before_mapping() {
        let bit = parse!("alias_reachable::bit");
        let live = parse!("alias_reachable::live");
        let pole = parse!("alias_reachable::pole");
        let cycle_a = parse!("alias_reachable::cycle_a");
        let cycle_b = parse!("alias_reachable::cycle_b");
        let mut source =
            AliasedAtom::from(Symbol::IF.call_args([bit.clone(), pole.clone(), live.clone()]));
        source.register_alias(live.clone(), parse!("alias_reachable::x^2"));
        source.register_alias(pole.clone(), bit.pow(-1));
        source.register_alias(cycle_a.clone(), cycle_b.clone());
        source.register_alias(cycle_b, cycle_a);
        let mut visited = 0;
        let selected = source.map_reachable_expressions(|body| {
            visited += 1;
            assert!(
                !body.contains(bit.pow(-1)),
                "inactive pole body must not be visited"
            );
            body.replace(bit.clone()).with(0)
        });
        assert_eq!(visited, 2);
        assert_eq!(selected.get_root(), &live);
        assert_eq!(selected.get_aliases().len(), 1);
        assert_eq!(selected.into_inner(), parse!("alias_reachable::x^2"));
    }

    #[test]
    fn reachable_forwarding_aliases_keep_external_bindings() {
        let external = parse!("alias_reachable_forward::x");
        let mut source = AliasedAtom::from(external.pow(2));
        source.register_alias(external.clone(), external.clone());
        let selected = source
            .map_reachable_expressions(|atom| atom)
            .scope_aliases();
        assert_eq!(selected.get_aliases().len(), 1);
        assert_eq!(selected.into_inner(), external.pow(2));
    }
}
