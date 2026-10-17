use std::{borrow::Cow, collections::hash_map::Entry, fmt};

use ahash::{HashMap, HashSet};

use crate::{
    atom::EvaluationError,
    evaluate::EvaluatorBuilder,
    printer::{AtomPrinter, PrintOptions},
};

use super::{Atom, AtomCore, AtomType, AtomView};

mod prepare;

/// An atom that contains opaque aliases, together with a map from the aliases to their original atoms.
/// An aliased atom may have a lower memory footprint than the original atom if it contains many repeated subexpressions.
///
/// Use [AtomCore::alias_subexpressions] to create an aliased atom, or register aliases manually.
///
///
/// # Examples
///
/// ```
/// use symbolica::prelude::*;
///
/// let a: AliasedAtom = parse!("(x+1)^2+(x+1)^3").into();
/// let b = a.add_alias(parse!("s(1)"), parse!("x+1"));
/// assert!(!b.contains_symbol(symbol!("x")));
/// assert_eq!(b.get_root(), &parse!("s(1)^2+s(1)^3"));
/// ```
#[derive(Clone)]
pub struct AliasedAtom {
    pub(crate) root: Atom,
    pub(crate) aliases: HashMap<Atom, Atom>, // TODO: Rc?
}

impl std::fmt::Display for AliasedAtom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        AtomPrinter::new(self.get_root().as_view()).fmt(f)
    }
}

impl fmt::Debug for AliasedAtom {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let aliases = self.sorted_aliases();

        if f.alternate() {
            writeln!(f, "{{")?;
            write!(f, "    {}", AtomPrinter::new(self.root.as_view()))?;
            for (alias, original) in aliases {
                writeln!(f, ",")?;
                write!(
                    f,
                    "    {}={}",
                    AtomPrinter::new(alias.as_view()),
                    AtomPrinter::new(original.as_view())
                )?;
            }
            write!(f, "\n}}")
        } else {
            write!(f, "{{{}", AtomPrinter::new(self.root.as_view()))?;
            for (alias, original) in aliases {
                write!(
                    f,
                    ", {}={}",
                    AtomPrinter::new(alias.as_view()),
                    AtomPrinter::new(original.as_view())
                )?;
            }
            write!(f, "}}")
        }
    }
}

impl Default for AliasedAtom {
    /// Create an aliased atom that represents the number 0.
    #[inline]
    fn default() -> Self {
        AliasedAtom {
            root: Atom::Zero,
            aliases: HashMap::default(),
        }
    }
}

impl AliasedAtom {
    /// Get the root atom, which may contain aliases.
    pub fn get_root(&self) -> &Atom {
        &self.root
    }

    /// Get the map from aliases to their original atoms. These atoms may contain aliases themselves.
    pub fn get_aliases(&self) -> &HashMap<Atom, Atom> {
        &self.aliases
    }

    /// Return the number of bytes stored in the root atom and alias definitions.
    pub fn get_byte_size(&self) -> usize {
        self.root.as_view().get_byte_size()
            + self
                .aliases
                .iter()
                .map(|(alias, original)| {
                    alias.as_view().get_byte_size() + original.as_view().get_byte_size()
                })
                .sum::<usize>()
    }

    /// Print the root atom and alias definitions with the given options.
    pub fn printer(&self, opts: PrintOptions) -> AliasedAtomPrinter<'_> {
        AliasedAtomPrinter {
            atom: self,
            print_opts: opts,
        }
    }

    /// Map the root atom using the given function.
    pub fn map_root(self, f: impl FnOnce(Atom) -> Atom) -> Self {
        Self {
            root: f(self.root),
            aliases: self.aliases,
        }
    }

    /// Return the root atom and the alias map.
    pub fn into_inner_with_aliases(self) -> (Atom, HashMap<Atom, Atom>) {
        (self.root, self.aliases)
    }

    /// Undo the common subexpression extraction and return the original atom.
    pub fn into_inner(mut self) -> Atom {
        // TODO: this can be a one-pass if unfolded in reverse insertion order
        loop {
            let out = self.root.replace_map(|a, _, out| {
                if let Some(replacement) = self.aliases.get::<[u8]>(a.get_data()) {
                    out.set_from_view(&replacement.as_view());
                }
            });

            if out != self.root {
                self.root = out;
            } else {
                break;
            }
        }

        self.root
    }

    /// Extract common subexpressions from the root atom and replace them with aliases, which are returned in the map.
    pub fn alias_subexpressions(
        self,
        f: impl FnMut(AtomView, usize, usize) -> Option<Atom>,
    ) -> Self {
        // FIXME: do in one pass
        self.into_inner().as_atom_view().alias_subexpressions(f)
    }

    /// Register an alias, but do not apply it to the root atom.
    pub fn register_alias(&mut self, alias: Atom, original: Atom) {
        match self.aliases.entry(alias) {
            Entry::Occupied(entry) => {
                if entry.get() != &original {
                    panic!(
                        "Redefined alias: {} -> {} vs {}",
                        entry.key(),
                        entry.get(),
                        original
                    );
                }
            }
            Entry::Vacant(entry) => {
                // TODO: check for cycles in alias definitions?
                entry.insert(original);
            }
        }
    }

    /// Create an evaluator for a multiple-alias expression.
    pub fn evaluator_multiple<'a, P: AtomCore>(
        exprs: &'a [Self],
        params: &[P],
    ) -> Result<EvaluatorBuilder<'a>, EvaluationError> {
        let roots: Vec<_> = exprs.iter().map(|x| x.root.as_atom_view()).collect();
        let mut b = EvaluatorBuilder::new_multiple_views(&roots, params);

        for aliases in exprs {
            b = b.add_aliases(aliases.aliases.iter().map(|x| (x.0.clone(), x.1.clone())))?;
        }

        Ok(b)
    }

    /// Rename an alias handle and rewrite all uses of the handle in the root and alias bodies.
    ///
    /// Returns `true` if `old` was registered as an alias and was successfully renamed.
    /// Returns an error if `new` is already registered and has a different value than `old`.
    pub fn rename_alias(&mut self, old: Atom, new: Atom) -> Result<bool, ()> {
        if old == new {
            return Ok(self.aliases.contains_key(&old));
        }

        let Some(original) = self.aliases.remove(&old) else {
            return Ok(false);
        };

        if let Some(existing) = self.aliases.get(&new) {
            if existing != &original {
                return Err(());
            }
        }

        self.root = Self::rename_alias_in_atom(&self.root, &old, &new);
        for body in self.aliases.values_mut() {
            *body = Self::rename_alias_in_atom(body, &old, &new);
        }

        self.aliases.entry(new).or_insert(original);
        Ok(true)
    }

    /// Register an alias for an atom and apply the substitution to the root.
    pub fn add_alias(mut self, alias: Atom, original: Atom) -> Self {
        let new_root = self.root.replace_map(|a, _, out| {
            if a == original.as_view() {
                out.set_from_view(&alias.as_view());
            }
        });

        self.register_alias(alias, original);

        Self {
            root: new_root,
            aliases: self.aliases,
        }
    }

    /// Apply currently registered aliases to the root atom.
    pub fn apply_aliases(&self) -> Self {
        let inv_map = self
            .aliases
            .iter()
            .map(|(a, b)| (b.as_view(), a.as_view()))
            .collect::<HashMap<_, _>>();

        let new_root = self.root.replace_map(|a, _, out| {
            if let Some(alias) = inv_map.get(&a) {
                out.set_from_view(alias);
            }
        });

        Self {
            root: new_root,
            aliases: self.aliases.clone(),
        }
    }

    /// Apply currently registered aliases to the root atom and inside the body of other aliases.
    pub fn apply_aliases_nested(&self) -> Self {
        let replacements = self.alias_replacements();
        let root = Self::apply_alias_replacements(&self.root, &replacements, None);
        let aliases = self
            .aliases
            .iter()
            .map(|(alias, body)| {
                (
                    alias.clone(),
                    Self::apply_alias_replacements(body, &replacements, Some(body)),
                )
            })
            .collect();

        Self { root, aliases }
    }

    /// Add two aliased atoms, fusing their alias maps. On conflicts, the `resolve` function is called to generate a new alias handle.
    pub fn add(mut self, rhs: &Self, resolve: impl Fn(AtomView) -> Atom) -> Self {
        let new_rhs_root = self.fuse_aliases(rhs, resolve);
        Self {
            aliases: self.aliases,
            root: &self.root + &new_rhs_root,
        }
    }

    /// Multiply two aliased atoms, fusing their alias maps.  On conflicts, the `resolve` function is called to generate a new alias handle.
    pub fn mul(mut self, rhs: &Self, resolve: impl Fn(AtomView) -> Atom) -> Self {
        let new_rhs_root = self.fuse_aliases(rhs, resolve);
        Self {
            aliases: self.aliases,
            root: &self.root * &new_rhs_root,
        }
    }

    /// Raise one aliased atom to another, fusing their alias maps. On conflicts, the `resolve` function is called to generate a new alias handle.
    pub fn pow(mut self, rhs: &Self, resolve: impl Fn(AtomView) -> Atom) -> Self {
        let new_rhs_root = self.fuse_aliases(rhs, resolve);
        Self {
            aliases: self.aliases,
            root: self.root.pow(&new_rhs_root),
        }
    }

    /// Try to add two aliased atoms, fusing their alias maps.
    ///
    /// If both atoms define the same alias handle with different bodies, the conflicting alias
    /// handle is returned.
    pub fn try_add(&self, rhs: &Self) -> Result<Self, Atom> {
        Ok(Self {
            aliases: self.try_fuse_aliases(rhs)?,
            root: &self.root + &rhs.root,
        })
    }

    /// Try to multiply two aliased atoms, fusing their alias maps.
    ///
    /// If both atoms define the same alias handle with different bodies, the conflicting alias
    /// handle is returned.
    pub fn try_mul(&self, rhs: &Self) -> Result<Self, Atom> {
        Ok(Self {
            aliases: self.try_fuse_aliases(rhs)?,
            root: &self.root * &rhs.root,
        })
    }

    /// Try to raise one aliased atom to another, fusing their alias maps.
    ///
    /// If both atoms define the same alias handle with different bodies, the conflicting alias
    /// handle is returned.
    pub fn try_pow(&self, rhs: &Self) -> Result<Self, Atom> {
        Ok(Self {
            aliases: self.try_fuse_aliases(rhs)?,
            root: self.root.pow(&rhs.root),
        })
    }

    /// Removes any aliases in the root whose body has the same symmetric operator as its use in the root.
    /// For example `x+y+s(1), s(1)=z+w` will yield `x+y+z+w`. This is important to maintain invariants in pattern matching, such
    /// as argument counts.
    pub fn flatten_operators(&mut self) {
        self.root = self.root.replace_map(|a, ctx, out| {
            if (ctx.parent_type == Some(AtomType::Add) || ctx.parent_type == Some(AtomType::Mul))
                && let Some(alias) = self.aliases.get::<[u8]>(a.get_data())
                && ctx.parent_type == Some(alias.get_atom_type())
            {
                out.set_from_view(&alias.as_atom_view());
            }
        });
    }

    /// Removes any aliases that are not used in the root or any of the aliases.
    pub fn prune(&mut self) {
        let mut to_remove = Vec::new();

        for alias in self.aliases.keys() {
            if !self.root.contains(alias) && self.aliases.values().all(|a| !a.contains(alias)) {
                to_remove.push(alias.clone());
            }
        }

        let prunable = !to_remove.is_empty();
        for alias in to_remove {
            self.aliases.remove(&alias);
        }

        if prunable {
            // recursively prune until no more aliases are removable
            self.prune();
        }
    }

    /// Removes any duplicate aliases that map to the same body, keeping the first one encountered.
    pub fn fuse_duplicate_aliases(&mut self) {
        let mut inv_map: HashMap<&Atom, &Atom> = HashMap::default();
        let mut remap: HashMap<Atom, Atom> = HashMap::default();
        for (alias, body) in &self.aliases {
            if inv_map.contains_key(body) {
                remap.insert(alias.clone(), inv_map[body].clone());
            } else {
                inv_map.insert(body, alias);
            }
        }

        for (alias, old_alias) in remap {
            self.aliases.remove(&alias);

            self.root = self.root.replace_map(|atom, _, out| {
                if atom == alias.as_view() {
                    out.set_from_view(&old_alias.as_view());
                }
            });

            for x in self.aliases.values_mut() {
                *x = x.replace_map(|atom, _, out| {
                    if atom == alias.as_view() {
                        out.set_from_view(&old_alias.as_view());
                    }
                });
            }
        }
    }

    fn sorted_aliases(&self) -> Vec<(&Atom, &Atom)> {
        let mut aliases: Vec<_> = self.aliases.iter().collect();
        aliases.sort_by_cached_key(|(alias, _)| format!("{}", AtomPrinter::new(alias.as_view())));
        aliases
    }

    /// Fuses the alias maps of two aliased atoms, resolving conflicts with the provided function, using the minimal
    /// amount of renames by analyzing the dependency graph of aliases.
    fn fuse_aliases<'a>(&mut self, rhs: &'a Self, resolve: impl Fn(AtomView) -> Atom) -> Atom {
        struct NodeInfo<'a> {
            dependencies: HashSet<AtomView<'a>>,
            rename: Option<Atom>,
            resolved: bool,
        }

        let mut dependency_graph: HashMap<&Atom, _> = HashMap::default();
        let max_bytes = rhs
            .aliases
            .keys()
            .map(|key| key.as_view().get_byte_size())
            .max()
            .unwrap_or(0);
        for (alias, body) in &rhs.aliases {
            let dependencies = rhs
                .direct_alias_dependencies(body, max_bytes)
                .into_iter()
                .map(Atom::as_view)
                .collect();

            dependency_graph.insert(
                alias,
                NodeInfo {
                    dependencies,
                    rename: None,
                    resolved: false,
                },
            );
        }

        fn resolve_dependencies<'a>(
            current: AtomView<'a>,
            aliases: &mut HashMap<Atom, Atom>,
            rhs_aliases: &HashMap<Atom, Atom>,
            dependency_graph: &mut HashMap<&'a Atom, NodeInfo<'a>>,
            resolve: &impl Fn(AtomView) -> Atom,
        ) {
            let node = dependency_graph.get_mut(current.get_data()).unwrap();

            if node.resolved {
                return;
            }

            let dependencies = std::mem::take(&mut node.dependencies);
            for dep in &dependencies {
                resolve_dependencies(*dep, aliases, rhs_aliases, dependency_graph, resolve);
            }

            let new_body = if dependencies
                .iter()
                .all(|d| dependency_graph[d.get_data()].rename.is_none())
            {
                Cow::Borrowed(&rhs_aliases[current.get_data()]) // original body remains unchanged
            } else {
                let body = rhs_aliases[current.get_data()].replace_map(|a, _, out| {
                    if dependencies.contains(a.get_data())
                        && let Some(new_alias) = &dependency_graph.get(a.get_data()).unwrap().rename
                    {
                        {
                            out.set_from_view(&new_alias.as_view());
                        }
                    }
                });

                Cow::Owned(body)
            };

            let node = dependency_graph.get_mut(current.get_data()).unwrap();
            node.resolved = true;

            match aliases.get(current.get_data()) {
                Some(existing) if existing == &*new_body => {}
                Some(_) => {
                    let mut new_alias = resolve(current);
                    while aliases.contains_key(&new_alias) {
                        new_alias = resolve(new_alias.as_atom_view());
                    }

                    node.rename = Some(new_alias.clone());
                    aliases.insert(new_alias, new_body.into_owned());
                }
                None => {
                    aliases.insert(current.to_owned(), new_body.into_owned());
                }
            };
        }

        for alias in rhs.aliases.keys() {
            resolve_dependencies(
                alias.as_view(),
                &mut self.aliases,
                &rhs.aliases,
                &mut dependency_graph,
                &resolve,
            );
        }

        rhs.root.replace_map(|a, _, out| {
            if let Some(new_alias) = dependency_graph
                .get(a.get_data())
                .and_then(|node| node.rename.as_ref())
            {
                out.set_from_view(&new_alias.as_atom_view());
            }
        })
    }

    fn try_fuse_aliases(&self, rhs: &Self) -> Result<HashMap<Atom, Atom>, Atom> {
        let mut aliases = self.aliases.clone();

        for (alias, original) in &rhs.aliases {
            match aliases.entry(alias.clone()) {
                Entry::Occupied(entry) => {
                    if entry.get() != original {
                        return Err(alias.clone());
                    }
                }
                Entry::Vacant(entry) => {
                    entry.insert(original.clone());
                }
            }
        }

        Ok(aliases)
    }

    fn alias_replacements(&self) -> Vec<(Atom, Atom)> {
        let mut replacements: Vec<_> = self
            .aliases
            .iter()
            .map(|(alias, body)| (alias.clone(), body.clone()))
            .collect();
        replacements.sort_by(|(_, b1), (_, b2)| {
            b2.as_view()
                .get_byte_size()
                .cmp(&b1.as_view().get_byte_size())
        });
        replacements
    }

    fn apply_alias_replacements(
        atom: &Atom,
        replacements: &[(Atom, Atom)],
        skip_whole_atom: Option<&Atom>,
    ) -> Atom {
        atom.replace_map(|a, _, out| {
            for (alias, body) in replacements {
                if skip_whole_atom.is_some_and(|skip| a == skip.as_view()) && a == body.as_view() {
                    continue;
                }

                if a == body.as_view() {
                    out.set_from_view(&alias.as_view());
                    break;
                }
            }
        })
    }

    fn rename_alias_in_atom(atom: &Atom, old: &Atom, new: &Atom) -> Atom {
        atom.replace_map(|a, _, out| {
            if a == old.as_view() {
                out.set_from_view(&new.as_view());
            }
        })
    }
}

/// A printer for aliased atoms, useful in a [format!].
///
/// # Examples
///
/// ```
/// use symbolica::prelude::*;
/// let a = parse!("(x+1)^2+(x+1)^3")
///     .alias_subexpressions(|_a, count, i| (count > 1).then(|| function!(symbol!("se"), i)));
/// println!("{}", a.printer(PrintOptions::file()));
/// ```
pub struct AliasedAtomPrinter<'a> {
    atom: &'a AliasedAtom,
    print_opts: PrintOptions,
}

impl fmt::Display for AliasedAtomPrinter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let aliases = self.atom.sorted_aliases();

        if f.alternate() {
            writeln!(f, "{{")?;
            write!(
                f,
                "    {}",
                AtomPrinter::new_with_options(self.atom.root.as_view(), self.print_opts.clone())
            )?;
            for (alias, original) in aliases {
                writeln!(f, ",")?;
                write!(
                    f,
                    "    {}={}",
                    AtomPrinter::new_with_options(alias.as_view(), self.print_opts.clone()),
                    AtomPrinter::new_with_options(original.as_view(), self.print_opts.clone())
                )?;
            }
            write!(f, "\n}}")
        } else {
            write!(
                f,
                "{{{}",
                AtomPrinter::new_with_options(self.atom.root.as_view(), self.print_opts.clone())
            )?;
            for (alias, original) in aliases {
                write!(
                    f,
                    ", {}={}",
                    AtomPrinter::new_with_options(alias.as_view(), self.print_opts.clone()),
                    AtomPrinter::new_with_options(original.as_view(), self.print_opts.clone())
                )?;
            }
            write!(f, "}}")
        }
    }
}

impl From<Atom> for AliasedAtom {
    fn from(atom: Atom) -> Self {
        AliasedAtom {
            root: atom,
            aliases: HashMap::default(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::{atom::AtomCore, function, id::AliasedAtom, parse, symbol};

    #[test]
    fn create_alias() {
        let a: AliasedAtom = parse!("(x+1)^2+(x+1)^3").into();
        let b = a.add_alias(parse!("s(1)"), parse!("x+1"));
        let c = b + 2 + parse!("(x+1)^4");
        assert!(c.contains_symbol(symbol!("x")));
        let d = c.apply_aliases();
        assert!(!d.contains_symbol(symbol!("x")));
    }

    #[test]
    fn flatten() {
        let mut aliased: AliasedAtom = parse!("x+y+s(1)").into();
        aliased.register_alias(parse!("s(1)"), parse!("z+w"));
        aliased.flatten_operators();
        assert_eq!(aliased.get_root(), &parse!("x+y+z+w"));
    }

    fn aliased_atom() -> super::AliasedAtom {
        parse!("(x+1)^2+(x+1)^3")
            .alias_subexpressions(|_a, count, i| (count > 1).then(|| function!(symbol!("se"), i)))
    }

    #[test]
    fn byte_size_counts_root_and_aliases() {
        let aliased = aliased_atom();
        let expected = aliased.get_root().as_view().get_byte_size()
            + aliased
                .get_aliases()
                .iter()
                .map(|(alias, original)| {
                    alias.as_view().get_byte_size() + original.as_view().get_byte_size()
                })
                .sum::<usize>();

        assert_eq!(aliased.get_byte_size(), expected);
    }

    #[test]
    fn binary_ops_with_atoms_and_symbols_preserve_aliases() {
        let aliased = aliased_atom();
        let expected = parse!("(x+1)^2+(x+1)^3+y");

        let with_symbol_rhs = &aliased + symbol!("y");
        assert_eq!(with_symbol_rhs.get_aliases(), aliased.get_aliases());
        assert_eq!(with_symbol_rhs.into_inner(), expected);

        let with_atom_lhs = parse!("y") + &aliased;
        assert_eq!(with_atom_lhs.get_aliases(), aliased.get_aliases());
        assert_eq!(with_atom_lhs.into_inner(), expected);

        let with_symbol_lhs = symbol!("y") + &aliased;
        assert_eq!(with_symbol_lhs.get_aliases(), aliased.get_aliases());
        assert_eq!(with_symbol_lhs.into_inner(), expected);
    }

    #[test]
    fn rename_alias_updates_root_and_alias_bodies() {
        let mut aliased: AliasedAtom = parse!("s(1)+f(s(1))").into();
        aliased.register_alias(parse!("s(1)"), parse!("x"));
        aliased.register_alias(parse!("t(1)"), parse!("f1(s(1))"));

        assert!(
            aliased
                .rename_alias(parse!("s(1)"), parse!("u(1)"))
                .unwrap()
        );
        assert!(
            !aliased
                .rename_alias(parse!("missing(1)"), parse!("v(1)"))
                .unwrap()
        );

        assert_eq!(aliased.get_root(), &parse!("u(1)+f(u(1))"));
        assert_eq!(
            aliased.get_aliases().get(&parse!("u(1)")),
            Some(&parse!("x"))
        );
        assert_eq!(
            aliased.get_aliases().get(&parse!("t(1)")),
            Some(&parse!("f1(u(1))"))
        );
        assert!(!aliased.get_aliases().contains_key(&parse!("s(1)")));
    }

    #[test]
    fn try_ops_fuse_aliases() {
        let mut lhs: AliasedAtom = parse!("s(1)").into();
        lhs.register_alias(parse!("s(1)"), parse!("x+1"));

        let mut rhs: AliasedAtom = parse!("t(1)").into();
        rhs.register_alias(parse!("t(1)"), parse!("y+1"));

        let sum = lhs.try_add(&rhs).unwrap();
        assert_eq!(sum.get_aliases().len(), 2);
        assert_eq!(sum.into_inner(), parse!("x+y+2"));

        let product = lhs.try_mul(&rhs).unwrap();
        assert_eq!(product.get_aliases().len(), 2);
        assert_eq!(product.into_inner(), parse!("(1+x)*(1+y)"));

        let pow = lhs.try_pow(&rhs).unwrap();
        assert_eq!(pow.get_aliases().len(), 2);
        assert_eq!(pow.into_inner(), parse!("(1+x)^(1+y)"));
    }

    #[test]
    fn ops_with_resolver_fuse_independent_rhs_aliases() {
        let mut lhs: AliasedAtom = parse!("s(1)").into();
        lhs.register_alias(parse!("s(1)"), parse!("x+1"));

        let mut rhs: AliasedAtom = parse!("t(1)+z").into();
        rhs.register_alias(parse!("t(1)"), parse!("y+1"));

        let sum = lhs.add(&rhs, |_| parse!("u(1)"));

        assert_eq!(sum.get_aliases().len(), 2);
        assert_eq!(sum.into_inner(), parse!("x+y+z+2"));
    }

    #[test]
    fn ops_with_resolver_rename_rhs_alias_dependencies() {
        let mut lhs: AliasedAtom = parse!("t(1)").into();
        lhs.register_alias(parse!("t(1)"), parse!("x"));

        let mut rhs: AliasedAtom = parse!("s(1)").into();
        rhs.register_alias(parse!("s(1)"), parse!("t(1)+1"));
        rhs.register_alias(parse!("t(1)"), parse!("y"));

        let sum = lhs.add(&rhs, |_| parse!("u(1)"));

        assert_eq!(sum.get_aliases().get(&parse!("u(1)")), Some(&parse!("y")));
        assert_eq!(
            sum.get_aliases().get(&parse!("s(1)")),
            Some(&parse!("1+u(1)"))
        );
        assert_eq!(sum.into_inner(), parse!("x+y+1"));
    }

    #[test]
    fn try_ops_report_conflicting_alias_handle() {
        let mut lhs: AliasedAtom = parse!("s(1)").into();
        lhs.register_alias(parse!("s(1)"), parse!("x"));

        let mut rhs: AliasedAtom = parse!("s(1)").into();
        rhs.register_alias(parse!("s(1)"), parse!("y"));

        assert_eq!(lhs.try_add(&rhs).unwrap_err(), parse!("s(1)"));
        assert_eq!(lhs.try_mul(&rhs).unwrap_err(), parse!("s(1)"));
        assert_eq!(lhs.try_pow(&rhs).unwrap_err(), parse!("s(1)"));
    }

    #[test]
    fn apply_aliases_nested_updates_alias_bodies() {
        let mut aliased: AliasedAtom = parse!("f(x+1)+f2(f(x+1))").into();
        aliased.register_alias(parse!("s(1)"), parse!("x+1"));
        aliased.register_alias(parse!("t(1)"), parse!("f(x+1)"));

        let nested = aliased.apply_aliases_nested();

        assert_eq!(nested.get_root(), &parse!("t(1)+f2(t(1))"));
        assert_eq!(
            nested.get_aliases().get(&parse!("t(1)")),
            Some(&parse!("f(s(1))"))
        );
        assert_eq!(
            nested.get_aliases().get(&parse!("s(1)")),
            Some(&parse!("x+1"))
        );
    }
}
