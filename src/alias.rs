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

use crate::atom::{Atom, AtomCore, AtomOrView, AtomView, Symbol};

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

    pub(crate) fn to_atom(&self) -> Atom {
        crate::function!(Symbol::ALIAS, self.token())
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

fn get_alias_handle(token: usize) -> Option<Arc<AliasHandle>> {
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
pub struct AliasedAtom {
    pub(crate) root: Atom,
    pub(crate) aliases: Vec<Arc<AliasHandle>>,
}

impl AtomView<'_> {
    pub(crate) fn get_alias_handles(&self) -> HashSet<Arc<AliasHandle>> {
        collect_alias_handles_in(*self).into_iter().collect()
    }
}

fn collect_alias_handles_in(root: AtomView<'_>) -> Vec<Arc<AliasHandle>> {
    let mut handles = HashSet::default();
    root.visitor(&mut |a| {
        if let AtomView::Fun(f) = a
            && f.get_symbol() == Symbol::ALIAS
        {
            let token: usize = f.iter().next().unwrap().try_into().unwrap();
            handles.insert(get_alias_handle(token).unwrap());
        }

        true
    });

    let mut handles: Vec<_> = handles.into_iter().collect();
    handles.sort_by_key(|handle| handle.id());
    handles
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
        Self {
            root,
            aliases: Vec::new(),
        }
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
        alias_subexpressions(self.root.as_view(), f)
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

fn register_aliased_atom(alias: AliasedAtom) -> (Arc<AliasHandle>, bool) {
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

pub(crate) fn alias_subexpressions(
    root: AtomView<'_>,
    mut f: impl FnMut(AtomView, usize, usize) -> Option<Atom>,
) -> AliasedAtom {
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

    let aliases = get_alias_handles(replaced_atom.as_view());
    AliasedAtom {
        root: replaced_atom,
        aliases,
    }
}

pub fn to_atom(root: &Atom) -> Atom {
    root.replace_map(|x, _, out| {
        if let AtomView::Fun(f) = x
            && f.get_symbol() == Symbol::ALIAS
        {
            let token: usize = f.iter().next().unwrap().try_into().unwrap();
            let nested = to_atom(&get_alias_entry(token).unwrap().atom);
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
fn test_aliased_atom_arithmetic_merges_handles() {
    let left = AliasedAtom::new(crate::parse!("x+f(1009)")).alias_literal(crate::parse!("f(1009)"));
    let right =
        AliasedAtom::new(crate::parse!("y+g(1013)")).alias_literal(crate::parse!("g(1013)"));
    let expected = merge_alias_handles(left.aliases.clone(), right.aliases.clone())
        .into_iter()
        .map(|handle| handle.token())
        .collect::<Vec<_>>();

    let sum = &left + &right;
    let sum_tokens = sum
        .aliases
        .iter()
        .map(|handle| handle.token())
        .collect::<Vec<_>>();
    assert_eq!(sum_tokens, expected);

    let mut product = left.clone();
    product *= &right;
    let product_tokens = product
        .aliases
        .iter()
        .map(|handle| handle.token())
        .collect::<Vec<_>>();
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
