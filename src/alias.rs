#![allow(deprecated)]

use std::{
    borrow::Borrow,
    hash::Hash,
    sync::{
        Arc, LazyLock, RwLock, Weak,
        atomic::{AtomicUsize, Ordering},
    },
};

use ahash::{HashMap, HashMapExt, HashSet};
use append_only_vec::AppendOnlyVec;
use arc_swap::ArcSwapOption;

use crate::atom::{Atom, AtomCore, AtomOrView, AtomView};

#[derive(Debug)]
struct AliasEntry {
    atom: Arc<Atom>,
    handle: Weak<AliasHandle>,
    dependencies: Vec<Arc<AliasHandle>>,
    generation: usize,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct AliasHandle {
    id: usize,
    generation: usize,
    atom: Arc<Atom>,
}

impl AliasHandle {
    pub(crate) fn id(&self) -> usize {
        self.id
    }

    pub(crate) fn token(&self) -> usize {
        pack_alias_token(self.id, self.generation)
    }

    pub(crate) fn to_atom(self: &Arc<Self>) -> Atom {
        Atom::alias(self.clone())
    }

    pub fn to_opaque_atom(self: &Arc<Self>) -> Atom {
        Atom::opaque_alias(self.clone())
    }

    pub(crate) fn atom(&self) -> &Atom {
        &self.atom
    }
}

impl Drop for AliasHandle {
    fn drop(&mut self) {
        let dependencies = {
            let mut store = ALIAS_STORE.write().unwrap();
            store.release(self.id, self.generation)
        };

        drop(dependencies);
    }
}

#[derive(Clone, Debug)]
struct AliasKey(Arc<Atom>);

impl Borrow<[u8]> for AliasKey {
    fn borrow(&self) -> &[u8] {
        self.0.as_view().get_data()
    }
}

impl PartialEq for AliasKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.as_view().get_data() == other.0.as_view().get_data()
    }
}

impl Eq for AliasKey {}

impl Hash for AliasKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.as_view().get_data().hash(state);
    }
}

#[derive(Debug)]
struct AliasStore {
    alias_map: HashMap<AliasKey, usize>,
    free: Vec<usize>,
}

struct AliasSlot {
    entry: ArcSwapOption<AliasEntry>,
    generation: AtomicUsize,
}

impl AliasSlot {
    fn new() -> Self {
        Self {
            entry: ArcSwapOption::from(None::<Arc<AliasEntry>>),
            generation: AtomicUsize::new(0),
        }
    }

    fn next_generation(&self) -> usize {
        self.generation.fetch_add(1, Ordering::AcqRel) + 1
    }
}

const ALIAS_GENERATION_BITS: u32 = usize::BITS / 2;
const ALIAS_SLOT_MASK: usize = (1usize << ALIAS_GENERATION_BITS) - 1;

fn pack_alias_token(slot: usize, generation: usize) -> usize {
    assert!(slot <= ALIAS_SLOT_MASK, "Too many aliases allocated");
    assert!(
        generation <= ALIAS_SLOT_MASK,
        "Alias generation counter overflow"
    );
    slot | (generation << ALIAS_GENERATION_BITS)
}

fn unpack_alias_token(token: usize) -> (usize, usize) {
    (token & ALIAS_SLOT_MASK, token >> ALIAS_GENERATION_BITS)
}

fn get_alias_entry(token: usize) -> Option<Arc<AliasEntry>> {
    let (slot, generation) = unpack_alias_token(token);
    if slot >= ALIASES.len() {
        return None;
    }

    let entry = ALIASES[slot].entry.load_full()?;
    (entry.generation == generation).then_some(entry)
}

pub(crate) fn get_alias_handle(token: usize) -> Option<Arc<AliasHandle>> {
    get_alias_entry(token)?.handle.upgrade()
}

impl AliasStore {
    pub fn new() -> Self {
        Self {
            alias_map: HashMap::new(),
            free: Vec::new(),
        }
    }

    fn insert(&mut self, atom: Arc<Atom>) -> Arc<AliasHandle> {
        let index = self
            .free
            .pop()
            .unwrap_or_else(|| ALIASES.push(AliasSlot::new()));
        let generation = ALIASES[index].next_generation();
        let handle = Arc::new(AliasHandle {
            id: index,
            generation,
            atom: atom.clone(),
        });
        let dependencies = collect_alias_handles_in(atom.as_view());
        let entry = AliasEntry {
            atom: atom.clone(),
            handle: Arc::downgrade(&handle),
            dependencies,
            generation,
        };

        ALIASES[index].entry.store(Some(Arc::new(entry)));
        self.alias_map
            .insert(AliasKey(atom), pack_alias_token(index, generation));
        handle
    }

    fn get_existing<'a, T: Into<AtomOrView<'a>>>(&self, a: T) -> Option<Arc<AliasHandle>> {
        let token = *self.alias_map.get(a.into().as_view().get_data())?;
        get_alias_handle(token)
    }

    fn release(&mut self, id: usize, generation: usize) -> Vec<Arc<AliasHandle>> {
        if id >= ALIASES.len() {
            return Vec::new();
        }

        let slot = &ALIASES[id];
        let Some(entry) = slot.entry.load_full() else {
            return Vec::new();
        };

        if entry.generation != generation {
            return Vec::new();
        }

        self.alias_map
            .remove::<[u8]>(entry.atom.as_view().get_data());
        let dependencies = entry.dependencies.clone();
        slot.entry.swap(None);
        self.free.push(id);
        dependencies
    }
}

#[derive(Clone, Debug)]
#[deprecated(
    note = "atoms now retain alias handles directly; use AtomCore::alias_repeated_subexpressions or AtomCore::alias_subexpressions on Atom/AtomView instead"
)]
pub struct AliasedAtom {
    pub(crate) root: Atom,
    pub(crate) aliases: Vec<Arc<AliasHandle>>,
}

impl AtomView<'_> {
    pub(crate) fn get_alias_handles(&self) -> HashSet<Arc<AliasHandle>> {
        collect_alias_handles_in(*self).into_iter().collect()
    }
}

pub(crate) fn collect_alias_handles_in(root: AtomView<'_>) -> Vec<Arc<AliasHandle>> {
    let mut handles = HashSet::default();
    collect_alias_handles_in_impl(root, &mut handles);

    let mut handles: Vec<_> = handles.into_iter().collect();
    handles.sort_by_key(|handle| handle.id());
    handles
}

pub(crate) fn collect_alias_handles_with_dependencies(root: AtomView<'_>) -> Vec<Arc<AliasHandle>> {
    fn visit(handle: Arc<AliasHandle>, seen: &mut HashSet<usize>, out: &mut Vec<Arc<AliasHandle>>) {
        if !seen.insert(handle.token()) {
            return;
        }

        for dependency in collect_alias_handles_in(handle.atom().as_view()) {
            visit(dependency, seen, out);
        }

        out.push(handle);
    }

    let mut seen = HashSet::default();
    let mut out = Vec::new();
    for handle in collect_alias_handles_in(root) {
        visit(handle, &mut seen, &mut out);
    }

    out
}

fn collect_alias_handles_in_impl(root: AtomView<'_>, handles: &mut HashSet<Arc<AliasHandle>>) {
    match root {
        AtomView::Num(_) | AtomView::Var(_) => {}
        AtomView::Alias(a) => {
            handles.insert(a.get_handle());
        }
        AtomView::Fun(f) => {
            for arg in f {
                collect_alias_handles_in_impl(arg, handles);
            }
        }
        AtomView::Pow(p) => {
            let (base, exp) = p.get_base_exp();
            collect_alias_handles_in_impl(base, handles);
            collect_alias_handles_in_impl(exp, handles);
        }
        AtomView::Mul(m) => {
            for child in m {
                collect_alias_handles_in_impl(child, handles);
            }
        }
        AtomView::Add(a) => {
            for child in a {
                collect_alias_handles_in_impl(child, handles);
            }
        }
    }
}

fn get_alias_handles(root: AtomView<'_>) -> Vec<Arc<AliasHandle>> {
    let mut handles: Vec<_> = root.get_alias_handles().into_iter().collect();
    handles.sort_by_key(|handle| handle.id());
    handles
}

fn merge_alias_handles(
    mut lhs: Vec<Arc<AliasHandle>>,
    rhs: Vec<Arc<AliasHandle>>,
) -> Vec<Arc<AliasHandle>> {
    lhs.extend(rhs);
    lhs.sort_by_key(|handle| handle.token());
    lhs.dedup_by_key(|handle| handle.token());
    lhs
}

fn aliased_from_arithmetic_result(root: Atom, live_aliases: Vec<Arc<AliasHandle>>) -> AliasedAtom {
    let aliases = aliases_from_arithmetic_result(root.as_view(), &live_aliases);
    drop(live_aliases);
    AliasedAtom { root, aliases }
}

fn aliases_from_arithmetic_result(
    root: AtomView<'_>,
    live_aliases: &[Arc<AliasHandle>],
) -> Vec<Arc<AliasHandle>> {
    if live_aliases.is_empty() {
        Vec::new()
    } else {
        get_alias_handles(root)
    }
}

impl std::ops::Neg for AliasedAtom {
    type Output = AliasedAtom;

    fn neg(self) -> Self::Output {
        let AliasedAtom {
            root,
            aliases: live_aliases,
        } = self;
        aliased_from_arithmetic_result(-root, live_aliases)
    }
}

impl std::ops::Neg for &AliasedAtom {
    type Output = AliasedAtom;

    fn neg(self) -> Self::Output {
        aliased_from_arithmetic_result(-&self.root, self.aliases.clone())
    }
}

macro_rules! impl_aliased_binary_ops {
    ($op_trait:ident, $op_method:ident) => {
        impl std::ops::$op_trait<AliasedAtom> for AliasedAtom {
            type Output = AliasedAtom;

            fn $op_method(self, rhs: AliasedAtom) -> Self::Output {
                let AliasedAtom {
                    root: lhs_root,
                    aliases: lhs_aliases,
                } = self;
                let AliasedAtom {
                    root: rhs_root,
                    aliases: rhs_aliases,
                } = rhs;
                let live_aliases = merge_alias_handles(lhs_aliases, rhs_aliases);
                aliased_from_arithmetic_result(
                    std::ops::$op_trait::$op_method(lhs_root, rhs_root),
                    live_aliases,
                )
            }
        }

        impl std::ops::$op_trait<&AliasedAtom> for AliasedAtom {
            type Output = AliasedAtom;

            fn $op_method(self, rhs: &AliasedAtom) -> Self::Output {
                let AliasedAtom {
                    root: lhs_root,
                    aliases: lhs_aliases,
                } = self;
                let live_aliases = merge_alias_handles(lhs_aliases, rhs.aliases.clone());
                aliased_from_arithmetic_result(
                    std::ops::$op_trait::$op_method(lhs_root, &rhs.root),
                    live_aliases,
                )
            }
        }

        impl std::ops::$op_trait<AliasedAtom> for &AliasedAtom {
            type Output = AliasedAtom;

            fn $op_method(self, rhs: AliasedAtom) -> Self::Output {
                let AliasedAtom {
                    root: rhs_root,
                    aliases: rhs_aliases,
                } = rhs;
                let live_aliases = merge_alias_handles(self.aliases.clone(), rhs_aliases);
                aliased_from_arithmetic_result(
                    std::ops::$op_trait::$op_method(&self.root, rhs_root),
                    live_aliases,
                )
            }
        }

        impl std::ops::$op_trait<&AliasedAtom> for &AliasedAtom {
            type Output = AliasedAtom;

            fn $op_method(self, rhs: &AliasedAtom) -> Self::Output {
                let live_aliases = merge_alias_handles(self.aliases.clone(), rhs.aliases.clone());
                aliased_from_arithmetic_result(
                    std::ops::$op_trait::$op_method(&self.root, &rhs.root),
                    live_aliases,
                )
            }
        }

        impl std::ops::$op_trait<Atom> for AliasedAtom {
            type Output = AliasedAtom;

            fn $op_method(self, rhs: Atom) -> Self::Output {
                let AliasedAtom {
                    root,
                    aliases: live_aliases,
                } = self;
                aliased_from_arithmetic_result(
                    std::ops::$op_trait::$op_method(root, rhs),
                    live_aliases,
                )
            }
        }

        impl std::ops::$op_trait<&Atom> for AliasedAtom {
            type Output = AliasedAtom;

            fn $op_method(self, rhs: &Atom) -> Self::Output {
                let AliasedAtom {
                    root,
                    aliases: live_aliases,
                } = self;
                aliased_from_arithmetic_result(
                    std::ops::$op_trait::$op_method(root, rhs),
                    live_aliases,
                )
            }
        }

        impl std::ops::$op_trait<Atom> for &AliasedAtom {
            type Output = AliasedAtom;

            fn $op_method(self, rhs: Atom) -> Self::Output {
                aliased_from_arithmetic_result(
                    std::ops::$op_trait::$op_method(&self.root, rhs),
                    self.aliases.clone(),
                )
            }
        }

        impl std::ops::$op_trait<&Atom> for &AliasedAtom {
            type Output = AliasedAtom;

            fn $op_method(self, rhs: &Atom) -> Self::Output {
                aliased_from_arithmetic_result(
                    std::ops::$op_trait::$op_method(&self.root, rhs),
                    self.aliases.clone(),
                )
            }
        }

        impl std::ops::$op_trait<AliasedAtom> for Atom {
            type Output = AliasedAtom;

            fn $op_method(self, rhs: AliasedAtom) -> Self::Output {
                let AliasedAtom {
                    root,
                    aliases: live_aliases,
                } = rhs;
                aliased_from_arithmetic_result(
                    std::ops::$op_trait::$op_method(self, root),
                    live_aliases,
                )
            }
        }

        impl std::ops::$op_trait<&AliasedAtom> for Atom {
            type Output = AliasedAtom;

            fn $op_method(self, rhs: &AliasedAtom) -> Self::Output {
                aliased_from_arithmetic_result(
                    std::ops::$op_trait::$op_method(self, &rhs.root),
                    rhs.aliases.clone(),
                )
            }
        }

        impl std::ops::$op_trait<AliasedAtom> for &Atom {
            type Output = AliasedAtom;

            fn $op_method(self, rhs: AliasedAtom) -> Self::Output {
                let AliasedAtom {
                    root,
                    aliases: live_aliases,
                } = rhs;
                aliased_from_arithmetic_result(
                    std::ops::$op_trait::$op_method(self, root),
                    live_aliases,
                )
            }
        }

        impl std::ops::$op_trait<&AliasedAtom> for &Atom {
            type Output = AliasedAtom;

            fn $op_method(self, rhs: &AliasedAtom) -> Self::Output {
                aliased_from_arithmetic_result(
                    std::ops::$op_trait::$op_method(self, &rhs.root),
                    rhs.aliases.clone(),
                )
            }
        }
    };
}

macro_rules! impl_aliased_assign_ops {
    ($assign_trait:ident, $assign_method:ident, $op_trait:ident, $op_method:ident) => {
        impl std::ops::$assign_trait<AliasedAtom> for AliasedAtom {
            fn $assign_method(&mut self, rhs: AliasedAtom) {
                let AliasedAtom {
                    root: rhs_root,
                    aliases: rhs_aliases,
                } = rhs;
                let live_aliases =
                    merge_alias_handles(std::mem::take(&mut self.aliases), rhs_aliases);
                self.root =
                    std::ops::$op_trait::$op_method(std::mem::take(&mut self.root), rhs_root);
                self.aliases = aliases_from_arithmetic_result(self.root.as_view(), &live_aliases);
                drop(live_aliases);
            }
        }

        impl std::ops::$assign_trait<&AliasedAtom> for AliasedAtom {
            fn $assign_method(&mut self, rhs: &AliasedAtom) {
                let live_aliases =
                    merge_alias_handles(std::mem::take(&mut self.aliases), rhs.aliases.clone());
                self.root =
                    std::ops::$op_trait::$op_method(std::mem::take(&mut self.root), &rhs.root);
                self.aliases = aliases_from_arithmetic_result(self.root.as_view(), &live_aliases);
                drop(live_aliases);
            }
        }

        impl std::ops::$assign_trait<Atom> for AliasedAtom {
            fn $assign_method(&mut self, rhs: Atom) {
                let live_aliases = std::mem::take(&mut self.aliases);
                self.root = std::ops::$op_trait::$op_method(std::mem::take(&mut self.root), rhs);
                self.aliases = aliases_from_arithmetic_result(self.root.as_view(), &live_aliases);
                drop(live_aliases);
            }
        }

        impl std::ops::$assign_trait<&Atom> for AliasedAtom {
            fn $assign_method(&mut self, rhs: &Atom) {
                let live_aliases = std::mem::take(&mut self.aliases);
                self.root = std::ops::$op_trait::$op_method(std::mem::take(&mut self.root), rhs);
                self.aliases = aliases_from_arithmetic_result(self.root.as_view(), &live_aliases);
                drop(live_aliases);
            }
        }
    };
}

impl_aliased_binary_ops!(Add, add);
impl_aliased_binary_ops!(Sub, sub);
impl_aliased_binary_ops!(Mul, mul);
impl_aliased_binary_ops!(Div, div);

impl_aliased_assign_ops!(AddAssign, add_assign, Add, add);
impl_aliased_assign_ops!(SubAssign, sub_assign, Sub, sub);
impl_aliased_assign_ops!(MulAssign, mul_assign, Mul, mul);
impl_aliased_assign_ops!(DivAssign, div_assign, Div, div);

impl AliasedAtom {
    pub fn new<'a, T: Into<AtomOrView<'a>>>(root: T) -> Self {
        let root = root.into().as_atom_view().to_owned();
        let aliases = get_alias_handles(root.as_view());
        Self { root, aliases }
    }

    pub fn get_root(&self) -> &Atom {
        &self.root
    }

    pub fn get_aliases(&self) -> &[Arc<AliasHandle>] {
        &self.aliases
    }

    pub fn into_inner(self) -> (Atom, Vec<Arc<AliasHandle>>) {
        let AliasedAtom { root, aliases } = self;
        (root, aliases)
    }

    pub fn alias_subexpressions(
        self,
        f: impl FnMut(AtomView, usize, usize) -> Option<Atom>,
    ) -> Self {
        let root = alias_subexpressions(self.root.as_view(), f);
        let aliases = get_alias_handles(root.as_view());
        Self { root, aliases }
    }

    pub fn add_alias(self, original: Atom) -> Self {
        self.alias_literal(original)
    }

    pub fn count_operations(&self) -> (usize, usize) {
        let (mut add, mut mul) = (0, 0);

        let mut counter = |a: AtomView<'_>| match a {
            AtomView::Mul(m) => {
                mul += m.get_nargs() - 1;
                true
            }
            AtomView::Add(a) => {
                add += a.get_nargs() - 1;
                true
            }
            AtomView::Pow(p) => {
                if let Ok(i) = isize::try_from(p.get_exp()) {
                    mul += i.unsigned_abs() - 1;
                }
                true
            }
            _ => true,
        };

        self.root.visitor(&mut counter);

        for alias in &self.aliases {
            if let Some(atom) = get_alias_entry(alias.token()).map(|entry| entry.atom.clone()) {
                atom.visitor(&mut counter);
            }
        }

        (add, mul)
    }

    pub fn new_with_aliases(root: Atom) -> Self {
        let aliases = ALIAS_STORE.read().unwrap();
        let mut handles = Vec::new();
        let new_root = root.replace_map(|x, _, out| {
            if let Some(alias) = aliases.get_existing(x) {
                out.set_from_view(&alias.to_atom().as_view());
                handles.push(alias);
            }
        });
        drop(aliases);
        let aliases = merge_alias_handles(handles, Vec::new());
        Self {
            root: new_root,
            aliases,
        }
    }

    fn alias_literal(&self, atom: Atom) -> Self {
        let wrapped = Self::new(atom.clone());
        let alias = register_aliased_atom(wrapped);
        let alias_atom = alias.0.to_atom();
        let new_root = self.root.replace_map(|x, _, out| {
            if x == atom.as_view() {
                out.set_from_view(&alias_atom.as_view());
            }
        });

        let aliases = merge_alias_handles(self.aliases.clone(), vec![alias.0]);
        Self {
            root: new_root,
            aliases,
        }
    }

    #[allow(dead_code)]
    fn alias_literal_alias(&self, atom: Self) -> Self {
        let copy = atom.root.clone();
        let alias = register_aliased_atom(atom);
        let alias_atom = alias.0.to_atom();

        let new_root = self.root.replace_map(|x, _, out| {
            if x == copy.as_view() {
                out.set_from_view(&alias_atom.as_view());
            }
        });

        let aliases = merge_alias_handles(self.aliases.clone(), vec![alias.0]);
        Self {
            root: new_root,
            aliases,
        }
    }
}

impl From<Atom> for AliasedAtom {
    fn from(atom: Atom) -> Self {
        Self::new(atom)
    }
}

pub(crate) fn register_aliased_atom(alias: AliasedAtom) -> (Arc<AliasHandle>, bool) {
    let AliasedAtom { root, aliases: _ } = alias;
    let atom = Arc::new(root);

    if let Some(handle) = ALIAS_STORE.read().unwrap().get_existing(atom.as_view()) {
        return (handle, false);
    }

    let mut is_new = false;
    let mut stale_dependencies = Vec::new();
    let handle = {
        let mut store = ALIAS_STORE.write().unwrap();

        match store.alias_map.get(atom.as_view().get_data()).copied() {
            Some(token) => {
                if let Some(handle) = get_alias_handle(token) {
                    handle
                } else {
                    let (id, generation) = unpack_alias_token(token);
                    stale_dependencies = store.release(id, generation);
                    is_new = true;
                    store.insert(atom)
                }
            }
            None => {
                is_new = true;
                store.insert(atom)
            }
        }
    };
    drop(stale_dependencies);

    (handle, is_new)
}

pub(crate) fn register_alias_atom(atom: Atom) -> Arc<AliasHandle> {
    register_aliased_atom(AliasedAtom::new(atom)).0
}

pub(crate) fn alias_subexpressions(
    root: AtomView<'_>,
    mut f: impl FnMut(AtomView, usize, usize) -> Option<Atom>,
) -> Atom {
    let mut subexpressions = HashMap::default();
    root.count_subexpressions(&mut subexpressions);
    let mut subexpr_vec: Vec<_> = subexpressions.into_iter().collect();
    subexpr_vec.sort_by(|(k1, _), (k2, _)| k2.get_byte_size().cmp(&k1.get_byte_size()));

    let mut subexpr_corrections = HashMap::default();
    let mut subs = HashMap::default();
    let mut inv_subs = HashMap::default();

    for (subexpr, mut count) in subexpr_vec.drain(..) {
        count += subexpr_corrections.get(&subexpr).cloned().unwrap_or(0);

        if count == 1 {
            continue;
        }

        if f(subexpr, count, subs.len()).is_some() {
            let alias = register_aliased_atom(AliasedAtom::new(subexpr));
            let replacement = alias.0.to_atom();
            subs.insert(replacement.clone(), subexpr.to_owned());
            inv_subs.insert(subexpr, replacement);
        } else {
            let mut subexpr_correction = HashMap::default();
            subexpr.count_subexpressions(&mut subexpr_correction);
            for (k, v) in subexpr_correction {
                *subexpr_corrections.entry(k).or_insert(0) += v * (count - 1);
            }
        }
    }

    let replaced_atom = root.replace_map(|a, _, out| {
        if let Some(replacement) = inv_subs.get(&a) {
            out.set_from_view(&replacement.as_view());
        }
    });

    for x in subs.values_mut() {
        *x = x.replace_map(|a, _, out| {
            if a != x.as_view()
                && let Some(replacement) = inv_subs.get(&a)
            {
                out.set_from_view(&replacement.as_view());
            }
        });
    }

    replaced_atom
}

pub fn to_atom(root: &Atom) -> Atom {
    root.replace_map(|x, _, out| {
        if let AtomView::Alias(a) = x {
            let nested = to_atom(get_alias_entry(a.get_token()).unwrap().atom.as_ref());
            out.set_from_view(&nested.as_view());
        }
    })
}

pub(crate) fn get_alias(id: usize) -> Option<Atom> {
    get_alias_entry(id).map(|entry| entry.atom.as_ref().clone())
}

static ALIAS_STORE: LazyLock<RwLock<AliasStore>> = LazyLock::new(|| RwLock::new(AliasStore::new()));
static ALIASES: AppendOnlyVec<AliasSlot> = AppendOnlyVec::new();

#[test]
fn test_alias_cleanup() {
    let a = crate::parse!("x+f(1)");
    let sa = AliasedAtom::new(a).alias_literal(crate::parse!("f(1)"));

    let b = crate::parse!("y*(x+f(1))");
    let a2 = AliasedAtom::new_with_aliases(b);
    let sa2 = a2.alias_literal_alias(sa.clone());

    drop(sa2);

    println!("{}", to_atom(&sa.root));
}

#[test]
fn alias_view_to_owned_keeps_handle_alive() {
    let aliased = AliasedAtom::new(crate::parse!("x+1")).alias_literal(crate::parse!("x+1"));
    let owned = aliased.root.as_view().to_owned();
    let aliases = aliased.aliases.clone();
    drop(aliased);
    drop(aliases);

    assert_eq!(to_atom(&owned), crate::parse!("x+1"));
}

#[test]
fn alias_derivative_acts_on_body() {
    use crate::{atom::AtomCore, symbol};

    let aliased = AliasedAtom::new(crate::parse!("x+1")).alias_literal(crate::parse!("x+1"));
    let derivative = aliased.derivative(symbol!("x"));

    assert_eq!(derivative.root, crate::parse!("1"));
}

#[test]
fn alias_pattern_match_uses_body() {
    use crate::atom::AtomCore;

    let aliased = AliasedAtom::new(crate::parse!("x+1")).alias_literal(crate::parse!("x+1"));
    let pattern = crate::parse!("x_+1").to_pattern();

    assert!(
        aliased
            .root
            .pattern_match(&pattern, None, None)
            .next()
            .is_some()
    );
}

#[test]
fn alias_replacement_descends_into_body() {
    use crate::atom::AtomCore;

    let aliased = AliasedAtom::new(crate::parse!("f(x)")).alias_literal(crate::parse!("f(x)"));
    let replaced = aliased
        .root
        .replace(crate::parse!("x"))
        .with(crate::parse!("y"));

    assert_eq!(replaced, crate::parse!("f(y)"));
}

#[test]
fn alias_contains_descends_past_small_alias_token() {
    use crate::atom::AtomCore;

    let body = crate::parse!("f(1009,1013,1019,1021)");
    let aliased = AliasedAtom::new(body.clone()).alias_literal(body.clone());

    assert!(aliased.root.contains(body.as_view()));
}

#[test]
fn alias_contains_descends_past_small_parent_with_alias() {
    use crate::atom::AtomCore;

    let body = crate::parse!("f(1009,1013,1019,1021)");
    let root = crate::function!(crate::symbol!("alias_contains_parent::g"), body.clone());
    let aliased = AliasedAtom::new(root).alias_literal(body.clone());

    assert!(aliased.root.contains(body.as_view()));
}

#[test]
fn alias_replacement_descends_past_small_parent_with_alias() {
    use crate::atom::AtomCore;

    let body = crate::parse!("f(1009,1013,1019,1021)");
    let root = crate::function!(crate::symbol!("alias_replacement_parent::g"), body.clone());
    let aliased = AliasedAtom::new(root).alias_literal(body.clone());
    let replaced = aliased.root.replace(body).with(crate::parse!("y"));

    assert_eq!(
        replaced,
        crate::function!(
            crate::symbol!("alias_replacement_parent::g"),
            crate::parse!("y")
        )
    );
}

#[test]
fn alias_normalization_resolves_add_terms() {
    let y_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("y"))).0;
    let sum = crate::parse!("y") + y_alias.to_atom();

    assert_eq!(sum, crate::parse!("2*y"));
}

#[test]
fn alias_normalization_cleans_handles_when_alias_disappears() {
    let x = Atom::var(crate::symbol!(
        "alias_normalization_cleans_handles_when_alias_disappears::x"
    ));
    let x_alias = register_aliased_atom(AliasedAtom::new(x.clone())).0;
    let token = x_alias.token();
    let sum = x + x_alias.to_atom();

    assert!(!sum.as_view().has_alias());

    drop(x_alias);
    assert!(get_alias(token).is_none());
}

#[test]
fn alias_normalization_prunes_removed_alias_handles() {
    let x = Atom::var(crate::symbol!(
        "alias_normalization_prunes_removed_alias_handles::x"
    ));
    let z = Atom::var(crate::symbol!(
        "alias_normalization_prunes_removed_alias_handles::z"
    ));
    let x_alias = register_aliased_atom(AliasedAtom::new(x.clone())).0;
    let z_alias = register_aliased_atom(AliasedAtom::new(z.clone())).0;
    let z_token = z_alias.token();
    let sum = x_alias.to_atom() + z - z_alias.to_atom();

    assert!(sum.as_view().has_alias());

    drop(z_alias);
    assert!(get_alias(z_token).is_none());
}

#[test]
fn raw_atom_keeps_alias_handles_alive() {
    let x = Atom::var(crate::symbol!("raw_atom_keeps_alias_handles_alive::x"));
    let x_alias = register_aliased_atom(AliasedAtom::new(x)).0;
    let token = x_alias.token();
    let raw = x_alias.to_atom().into_raw();

    drop(x_alias);
    assert!(get_alias(token).is_some());

    let restored = unsafe { Atom::from_raw(raw) };
    assert!(restored.as_view().has_alias());

    drop(restored);
    assert!(get_alias(token).is_none());
}

#[test]
fn export_import_keeps_used_alias() {
    let x = Atom::var(crate::symbol!("export_import_keeps_used_alias::x"));
    let x_alias = register_aliased_atom(AliasedAtom::new(x.clone())).0;
    let old_token = x_alias.token();
    let expr = x_alias.to_atom();

    let mut data = Vec::new();
    expr.export(&mut data).unwrap();
    drop(expr);
    drop(x_alias);
    assert!(get_alias(old_token).is_none());

    let imported = Atom::import(&mut std::io::Cursor::new(data), None).unwrap();
    let AtomView::Alias(alias) = imported.as_view() else {
        panic!("Expected imported alias");
    };

    assert_ne!(alias.get_token(), old_token);
    assert_eq!(alias.get_body(), x.as_view());
}

#[test]
fn export_import_keeps_alias_dependencies() {
    let x = Atom::var(crate::symbol!("export_import_keeps_alias_dependencies::x"));
    let inner_alias = register_aliased_atom(AliasedAtom::new(x.clone())).0;
    let inner_token = inner_alias.token();
    let outer_alias = register_aliased_atom(AliasedAtom::new(inner_alias.to_atom())).0;
    let old_outer_token = outer_alias.token();
    let expr = outer_alias.to_atom();

    let mut data = Vec::new();
    expr.export(&mut data).unwrap();
    drop(expr);
    drop(outer_alias);
    drop(inner_alias);
    assert!(get_alias(old_outer_token).is_none());
    assert!(get_alias(inner_token).is_none());

    let imported = Atom::import(&mut std::io::Cursor::new(data), None).unwrap();
    let AtomView::Alias(outer) = imported.as_view() else {
        panic!("Expected imported outer alias");
    };
    let AtomView::Alias(inner) = outer.get_body() else {
        panic!("Expected imported inner alias");
    };

    assert_ne!(outer.get_token(), old_outer_token);
    assert_ne!(inner.get_token(), inner_token);
    assert_eq!(inner.get_body(), x.as_view());
}

#[test]
fn alias_normalization_resolves_add_terms_independent_of_order() {
    let y_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("y"))).0;
    let sum = y_alias.to_atom() + crate::parse!("y");

    assert_eq!(sum, crate::parse!("2*y"));
}

#[test]
fn opaque_alias_addition_does_not_merge_with_body() {
    let x_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("x"))).0;
    let sum = crate::parse!("x") + x_alias.to_opaque_atom();

    assert!(sum.as_view().has_alias());
    assert_ne!(sum, crate::parse!("2*x"));
}

#[test]
fn opaque_alias_literal_pattern_does_not_match_body() {
    use crate::atom::AtomCore;

    let x_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("x"))).0;
    let opaque = x_alias.to_opaque_atom();
    let pattern = crate::parse!("x").to_pattern();

    assert!(opaque.pattern_match(&pattern, None, None).next().is_none());
}

#[test]
fn opaque_alias_derivative_treats_alias_as_atom() {
    use crate::atom::AtomCore;

    let x_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("x"))).0;
    let expr = crate::parse!("x") + x_alias.to_opaque_atom();

    assert_eq!(expr.derivative(crate::symbol!("x")), crate::parse!("1"));
}

#[test]
fn alias_print_mode_renders_selected_aliases() {
    use crate::{
        atom::AtomCore,
        printer::{AliasPrintMode, PrintOptions},
    };

    let x_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("x"))).0;
    let transparent = x_alias.to_atom();
    let opaque = x_alias.to_opaque_atom();
    let transparent_alias = "⟨x⟩";
    let opaque_alias = "⟪x⟫";

    assert_eq!(
        format!("{}", transparent.printer(PrintOptions::file_no_namespace())),
        "x"
    );
    assert_eq!(
        format!(
            "{}",
            transparent.printer(PrintOptions {
                alias_print_mode: AliasPrintMode::All,
                ..PrintOptions::file_no_namespace()
            })
        ),
        transparent_alias
    );
    assert_eq!(
        format!(
            "{}",
            transparent.printer(PrintOptions {
                alias_print_mode: AliasPrintMode::OpaqueOnly,
                ..PrintOptions::file_no_namespace()
            })
        ),
        "x"
    );
    assert_eq!(
        format!(
            "{}",
            opaque.printer(PrintOptions {
                alias_print_mode: AliasPrintMode::OpaqueOnly,
                ..PrintOptions::file_no_namespace()
            })
        ),
        opaque_alias
    );
    assert_eq!(
        format!(
            "{}",
            (crate::parse!("x") + opaque).printer(PrintOptions {
                alias_print_mode: AliasPrintMode::OpaqueOnly,
                ..PrintOptions::file_no_namespace()
            })
        ),
        "⟪x⟫+x"
    );
}

#[test]
fn alias_normalization_resolves_mul_factors() {
    let x_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("x"))).0;
    let product = crate::parse!("x") * x_alias.to_atom();

    assert_eq!(product, crate::parse!("x^2"));
}

#[test]
fn alias_normalization_resolves_mul_factors_independent_of_order() {
    let x_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("x"))).0;
    let product = x_alias.to_atom() * crate::parse!("x");

    assert_eq!(product, crate::parse!("x^2"));
}

#[test]
fn alias_addition_flattens_nested_alias_chain_with_add_body() {
    let zw_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("z+w"))).0;
    let nested_alias = register_aliased_atom(AliasedAtom::new(zw_alias.to_atom())).0;
    let sum = crate::parse!("x") + nested_alias.to_atom();

    assert!(!sum.as_view().has_alias());
    assert_eq!(sum, crate::parse!("w+x+z"));
}

#[test]
fn alias_multiplication_flattens_nested_alias_chain_with_mul_body() {
    let yz_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("y*z"))).0;
    let nested_alias = register_aliased_atom(AliasedAtom::new(yz_alias.to_atom())).0;
    let product = crate::parse!("x") * nested_alias.to_atom();

    assert!(!product.as_view().has_alias());
    assert_eq!(product, crate::parse!("x*y*z"));
}

#[test]
fn alias_addition_preserves_non_add_alias() {
    let huge_alias =
        register_aliased_atom(AliasedAtom::new(crate::parse!("exp(x)*log(y)*sin(z)"))).0;
    let sum = crate::parse!("x") + huge_alias.to_atom();

    assert!(sum.as_view().has_alias());
    assert_eq!(to_atom(&sum), crate::parse!("x+exp(x)*log(y)*sin(z)"));
}

#[test]
fn alias_addition_merges_nested_alias_equivalent_terms() {
    let x_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("x"))).0;
    let x_alias_atom = x_alias.to_atom();

    let mut left = crate::atom::Atom::new();
    left.to_fun(crate::atom::Symbol::EXP)
        .add_arg(x_alias_atom.as_view());

    let exp_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("exp(x)"))).0;
    let sum = left + exp_alias.to_atom();

    assert!(sum.as_view().has_alias());
    assert_eq!(to_atom(&sum), crate::parse!("2*exp(x)"));
}

#[test]
fn alias_addition_does_not_merge_different_powers() {
    let x3_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("x^3"))).0;
    let sum = crate::parse!("x^2") + x3_alias.to_atom();

    assert!(sum.as_view().has_alias());
    assert_eq!(to_atom(&sum), crate::parse!("x^2+x^3"));
}

#[test]
fn alias_addition_merges_mul_body_without_same_type_alias_factor() {
    let xy_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("x*y"))).0;
    let sum = xy_alias.to_atom() + crate::parse!("x*y");

    assert!(!sum.as_view().has_alias());
    assert_eq!(sum, crate::parse!("2*x*y"));
}

#[test]
fn alias_multiplication_merges_nested_alias_equivalent_factors() {
    let x_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("x"))).0;
    let x_alias_atom = x_alias.to_atom();

    let mut left = crate::atom::Atom::new();
    left.to_fun(crate::atom::Symbol::EXP)
        .add_arg(x_alias_atom.as_view());

    let exp_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("exp(x)"))).0;
    let product = left * exp_alias.to_atom();

    assert!(product.as_view().has_alias());
    assert_eq!(to_atom(&product), crate::parse!("exp(x)^2"));
}

#[test]
fn alias_multiplication_merges_semantic_power_bases() {
    let x_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("x"))).0;
    let product = x_alias.to_atom() * crate::parse!("x^2");

    assert!(product.as_view().has_alias());
    assert_eq!(to_atom(&product), crate::parse!("x^3"));
}

#[test]
fn alias_antisymmetric_function_detects_semantic_duplicate_args() {
    let f = crate::symbol!("alias_antisymmetric_function_detects_semantic_duplicate_args::f"; Antisymmetric);
    let x_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("x"))).0;

    let value = crate::function!(f, x_alias.to_atom(), crate::parse!("x"));

    assert_eq!(value, crate::parse!("0"));
}

#[test]
fn alias_pattern_match_flattens_same_type_children() {
    use crate::atom::AtomCore;

    let zw_alias = register_aliased_atom(AliasedAtom::new(crate::parse!("z+w"))).0;
    let zw_alias_atom = zw_alias.to_atom();
    let mut target = crate::atom::Atom::new();
    let add = target.to_add();
    add.extend(crate::parse!("x").as_view());
    add.extend(crate::parse!("y").as_view());
    add.extend(zw_alias_atom.as_view());

    let pattern = crate::parse!("a_+b_+c_+d_").to_pattern();
    assert!(target.pattern_match(&pattern, None, None).next().is_some());
}

#[test]
fn atom_core_alias_repeated_subexpressions_skips_variables() {
    use crate::atom::AtomCore;

    let expr = crate::parse!("exp(x+1)+log(x+1)+(x+1)^2+x+x");
    let aliased = expr.alias_repeated_subexpressions();

    assert_eq!(to_atom(&aliased), expr);
    let aliases = collect_alias_handles_in(aliased.as_view());
    assert_eq!(aliases.len(), 1);
    assert_eq!(aliases[0].atom.as_ref(), &crate::parse!("1+x"));
}

#[test]
fn test_aliased_atom_arithmetic_merges_handles() {
    let left =
        AliasedAtom::new(crate::parse!("x+exp(f(1009))")).alias_literal(crate::parse!("f(1009)"));
    let right =
        AliasedAtom::new(crate::parse!("y+log(g(1013))")).alias_literal(crate::parse!("g(1013)"));
    let expected = merge_alias_handles(left.aliases.clone(), right.aliases.clone())
        .into_iter()
        .map(|handle| handle.token())
        .collect::<Vec<_>>();

    let sum = &left + &right;
    let mut sum_tokens = sum
        .aliases
        .iter()
        .map(|handle| handle.token())
        .collect::<Vec<_>>();
    sum_tokens.sort();
    assert_eq!(sum_tokens, expected);

    let mut product = left.clone();
    product *= &right;
    let mut product_tokens = product
        .aliases
        .iter()
        .map(|handle| handle.token())
        .collect::<Vec<_>>();
    product_tokens.sort();
    assert_eq!(product_tokens, expected);

    drop(left);
    drop(right);

    for token in &expected {
        assert!(get_alias(*token).is_some());
    }

    drop(sum);
    drop(product);

    for token in expected {
        assert!(get_alias(token).is_none());
    }
}

#[test]
fn test_aliased_atom_evaluator_reuses_alias_instruction() {
    use crate::evaluate::{FunctionMap, OptimizationSettings};

    let aliased =
        AliasedAtom::new(crate::parse!("exp(x+y)+log(x+y)")).alias_literal(crate::parse!("x+y"));
    let params = vec![crate::parse!("x"), crate::parse!("y")];
    let direct_evaluator = aliased
        .evaluator(
            &FunctionMap::new(),
            &params,
            OptimizationSettings {
                horner_iterations: 0,
                cpe_iterations: Some(0),
                ..OptimizationSettings::default()
            },
        )
        .unwrap();
    let tree_evaluator = aliased
        .evaluator(
            &FunctionMap::new(),
            &params,
            OptimizationSettings {
                horner_iterations: 0,
                cpe_iterations: Some(0),
                direct_translation: false,
                ..OptimizationSettings::default()
            },
        )
        .unwrap();

    assert_eq!(direct_evaluator.count_operations().0, 2);
    assert_eq!(tree_evaluator.count_operations().0, 2);

    drop(aliased);

    let mut evaluator = direct_evaluator.map_coeff(&|x| x.re.to_f64());
    let value = evaluator.evaluate_single(&[1., 2.]);
    let expected = 3f64.exp() + 3f64.ln();
    assert!((value - expected).abs() < 1e-12);

    let mut evaluator = tree_evaluator.map_coeff(&|x| x.re.to_f64());
    let value = evaluator.evaluate_single(&[1., 2.]);
    assert!((value - expected).abs() < 1e-12);
}

#[test]
fn test_aliased_atom_evaluator_horners_alias_body() {
    use crate::evaluate::{FunctionMap, OptimizationSettings};

    let poly = crate::parse!("x^3+x^2+x+1");
    let aliased =
        AliasedAtom::new(crate::parse!("exp(x^3+x^2+x+1)+log(x^3+x^2+x+1)")).alias_literal(poly);
    let params = vec![crate::parse!("x")];
    let without_horner = aliased
        .evaluator(
            &FunctionMap::new(),
            &params,
            OptimizationSettings {
                horner_iterations: 0,
                cpe_iterations: Some(0),
                ..OptimizationSettings::default()
            },
        )
        .unwrap();
    let with_horner = aliased
        .evaluator(
            &FunctionMap::new(),
            &params,
            OptimizationSettings {
                horner_iterations: 1,
                cpe_iterations: Some(0),
                ..OptimizationSettings::default()
            },
        )
        .unwrap();

    assert!(with_horner.count_operations().1 < without_horner.count_operations().1);

    drop(aliased);

    let mut evaluator = with_horner.map_coeff(&|x| x.re.to_f64());
    let value = evaluator.evaluate_single(&[2.]);
    let expected = 15f64.exp() + 15f64.ln();
    assert!((value - expected).abs() < 1e-10);
}

#[test]
fn test_alias_inside_function_map_uses_outer_arguments() {
    use crate::evaluate::{FunctionMap, OptimizationSettings};
    use crate::symbol;

    let aliased_body = AliasedAtom::new(crate::parse!("x^2+4")).alias_literal(crate::parse!("x^2"));
    let (body, aliases) = aliased_body.into_inner();
    let mut fn_map = FunctionMap::new();
    fn_map
        .add_function(symbol!("f"), vec![symbol!("x")], body)
        .unwrap();

    let aliased = AliasedAtom {
        root: crate::function!(symbol!("f"), crate::parse!("y")),
        aliases,
    };
    let params = vec![crate::parse!("y")];

    let direct_evaluator = aliased
        .evaluator(
            &fn_map,
            &params,
            OptimizationSettings {
                horner_iterations: 0,
                cpe_iterations: Some(0),
                ..OptimizationSettings::default()
            },
        )
        .unwrap();
    let tree_evaluator = aliased
        .evaluator(
            &fn_map,
            &params,
            OptimizationSettings {
                horner_iterations: 0,
                cpe_iterations: Some(0),
                direct_translation: false,
                ..OptimizationSettings::default()
            },
        )
        .unwrap();

    let mut evaluator = direct_evaluator.map_coeff(&|x| x.re.to_f64());
    assert_eq!(evaluator.evaluate_single(&[3.]), 13.);

    let mut evaluator = tree_evaluator.map_coeff(&|x| x.re.to_f64());
    assert_eq!(evaluator.evaluate_single(&[3.]), 13.);
}
