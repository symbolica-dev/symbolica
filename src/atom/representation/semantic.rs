//! Expression key semantics. RawAtom remains a byte-oriented storage key.

use std::{borrow::Cow, cmp::Ordering, hash::Hash};

use super::{
    ADD_ID, ALIAS_ID, ALIAS_OPAQUE_FLAG, AtomView, FUN_ID, HAS_ALIAS_FLAG, MUL_ID, NO_ALIASES,
    NOT_NORMALIZED, POW_ID, PackedRationalNumberWriter, TYPE_MASK, function_metadata,
    get_alias_handle, raw_has_alias, read_alias_token, read_packed_pair, skip_raw_atom,
};

/// A borrowed expression key with transparent-alias equality and hashing.
///
/// Use this instead of encoded bytes to query a map keyed by [`super::Atom`],
/// [`AtomView`], or [`super::AtomOrView`]. Opaque aliases retain their identity.
/// Equality preserves stored structure and normalization flags; it does not
/// normalize expressions or run user callbacks.
#[repr(transparent)]
pub struct BorrowedAtom([u8]);

impl BorrowedAtom {
    /// Borrow the expression as a key without allocating or extending its lifetime.
    #[inline]
    pub fn new<'a>(view: AtomView<'a>) -> &'a Self {
        Self::from_data(view.get_data())
    }

    #[inline]
    pub(super) fn from_data(data: &[u8]) -> &Self {
        // SAFETY: BorrowedAtom is transparent over [u8], so the slice pointer and
        // length are unchanged. Callers supply a complete, valid expression and
        // keep its owner (including any alias handles) alive for this borrow.
        unsafe { &*(data as *const [u8] as *const Self) }
    }
}

impl PartialEq for BorrowedAtom {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        encoded_eq(&self.0, &other.0)
    }
}

impl Eq for BorrowedAtom {}

impl Hash for BorrowedAtom {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        encoded_hash(&self.0, state);
    }
}

impl PartialOrd for BorrowedAtom {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BorrowedAtom {
    fn cmp(&self, other: &Self) -> Ordering {
        encoded_cmp(&self.0, &other.0)
    }
}

#[cold]
#[inline(never)]
pub(crate) fn aliased_atom_eq<T: crate::atom::AtomCore>(left: &super::Atom, right: &T) -> bool {
    aliased_view_eq(&left.as_view(), &right.as_atom_view())
}

#[cold]
#[inline(never)]
pub(crate) fn aliased_view_eq_other<T: crate::atom::AtomCore>(
    left: &AtomView<'_>,
    right: &T,
) -> bool {
    aliased_view_eq(left, &right.as_atom_view())
}

#[cold]
pub(crate) fn aliased_atom_hash<H: std::hash::Hasher>(atom: &super::Atom, state: &mut H) {
    aliased_view_hash(&atom.as_view(), state);
}

#[inline]
pub(crate) fn view_eq(left: &AtomView<'_>, right: &AtomView<'_>) -> bool {
    let left_data = left.get_data();
    let right_data = right.get_data();
    left_data == right_data
        || ((raw_has_alias(left_data) || raw_has_alias(right_data)) && aliased_view_eq(left, right))
}

#[cold]
fn aliased_view_eq(left: &AtomView<'_>, right: &AtomView<'_>) -> bool {
    if let Some(body) = borrowed_alias_body(left) {
        return view_eq(&body, right);
    }
    if let Some(body) = borrowed_alias_body(right) {
        return view_eq(left, &body);
    }
    expanded_eq(left.get_data(), right.get_data())
}

#[inline]
pub(crate) fn view_hash<H: std::hash::Hasher>(view: &AtomView<'_>, state: &mut H) {
    let data = view.get_data();
    if raw_has_alias(data) {
        aliased_view_hash(view, state);
    } else {
        data.hash(state);
    }
}

#[cold]
fn aliased_view_hash<H: std::hash::Hasher>(view: &AtomView<'_>, state: &mut H) {
    if let Some(body) = borrowed_alias_body(view) {
        view_hash(&body, state);
    } else {
        expanded_hash(view.get_data(), state);
    }
}

#[cold]
pub(crate) fn aliased_view_cmp(left: &AtomView<'_>, right: &AtomView<'_>) -> Ordering {
    if left.get_data() == right.get_data() {
        return Ordering::Equal;
    }
    if let Some(body) = borrowed_alias_body(left) {
        return body.cmp_public(right);
    }
    if let Some(body) = borrowed_alias_body(right) {
        return left.cmp_public(&body);
    }
    expanded_cmp(left.get_data(), right.get_data())
}

fn borrowed_alias_body<'a>(view: &AtomView<'a>) -> Option<AtomView<'a>> {
    if let AtomView::Alias(alias) = view
        && !alias.is_opaque()
        && !alias.aliases.is_empty()
    {
        // A live view already borrows the owner that keeps this body alive.
        // Detached byte views have no ownership context and use registry guards.
        Some(alias.get_body())
    } else {
        None
    }
}

#[inline]
pub(crate) fn encoded_eq(left: &[u8], right: &[u8]) -> bool {
    left == right || ((raw_has_alias(left) || raw_has_alias(right)) && expanded_eq(left, right))
}

#[cold]
fn expanded_eq(left: &[u8], right: &[u8]) -> bool {
    if is_transparent_alias(left) {
        let handle = alias_handle(left);
        return encoded_eq(handle.atom().as_view().get_data(), right);
    }
    if is_transparent_alias(right) {
        let handle = alias_handle(right);
        return encoded_eq(left, handle.atom().as_view().get_data());
    }
    expanded_data(left) == expanded_data(right)
}

#[inline]
pub(crate) fn encoded_hash<H: std::hash::Hasher>(data: &[u8], state: &mut H) {
    if raw_has_alias(data) {
        expanded_hash(data, state);
    } else {
        data.hash(state);
    }
}

#[cold]
fn expanded_hash<H: std::hash::Hasher>(data: &[u8], state: &mut H) {
    if is_transparent_alias(data) {
        let handle = alias_handle(data);
        encoded_hash(handle.atom().as_view().get_data(), state);
        return;
    }
    // Hash the same contiguous byte slice as an unaliased expression. Arbitrary
    // hashers need not treat several writes like one concatenated write.
    expanded_data(data).as_ref().hash(state);
}

#[inline]
pub(crate) fn encoded_cmp(left: &[u8], right: &[u8]) -> Ordering {
    if left == right {
        return Ordering::Equal;
    }
    if raw_has_alias(left) || raw_has_alias(right) {
        expanded_cmp(left, right)
    } else {
        compare_expanded(left, right)
    }
}

#[cold]
fn expanded_cmp(left: &[u8], right: &[u8]) -> Ordering {
    if is_transparent_alias(left) {
        let handle = alias_handle(left);
        return encoded_cmp(handle.atom().as_view().get_data(), right);
    }
    if is_transparent_alias(right) {
        let handle = alias_handle(right);
        return encoded_cmp(left, handle.atom().as_view().get_data());
    }
    let left = expanded_data(left);
    let right = expanded_data(right);
    compare_expanded(&left, &right)
}

#[inline]
fn compare_expanded(left: &[u8], right: &[u8]) -> Ordering {
    // Transparent aliases have been expanded. Remaining opaque aliases compare
    // by token, so these views never need to borrow their bodies or ownership.
    AtomView::from_with_aliases(left, &NO_ALIASES)
        .cmp(&AtomView::from_with_aliases(right, &NO_ALIASES))
        .then_with(|| left.cmp(right))
}

/// Preserve ordinary encoded function ordering after transparent expansion.
#[cold]
pub(crate) fn compare_expanded_function_data(left: &[u8], right: &[u8]) -> Ordering {
    expanded_data(left)
        .as_ref()
        .cmp(expanded_data(right).as_ref())
}

/// Compare coefficient-free term encodings using ordinary sum ordering.
#[cold]
pub(crate) fn compare_expanded_term_keys(left: &[u8], right: &[u8]) -> Ordering {
    let left = expanded_data(left);
    let right = expanded_data(right);
    AtomView::from_with_aliases(&left, &NO_ALIASES)
        .get_term_cmp_slice()
        .cmp(AtomView::from_with_aliases(&right, &NO_ALIASES).get_term_cmp_slice())
}

/// Compare sums in expanded term order, including older normalized aliased sums
/// whose stored order predates the use of encoded monomial keys.
#[cold]
pub(crate) fn compare_expanded_sum_contents(left: &[u8], right: &[u8]) -> Ordering {
    if left == right {
        return Ordering::Equal;
    }
    let left = expanded_data(left);
    let right = expanded_data(right);
    let AtomView::Add(left) = AtomView::from_with_aliases(&left, &NO_ALIASES) else {
        unreachable!()
    };
    let AtomView::Add(right) = AtomView::from_with_aliases(&right, &NO_ALIASES) else {
        unreachable!()
    };
    let left = left.to_slice();
    let right = right.to_slice();
    left.len().cmp(&right.len()).then_with(|| {
        // Compare children rather than these roots: opaque-only sums still have
        // alias flags, and comparing their roots would re-enter this helper.
        left.iter()
            .zip(right.iter())
            .map(|(a, b)| a.cmp(&b))
            .find(|order| !order.is_eq())
            .unwrap_or(Ordering::Equal)
    })
}

fn expanded_data(data: &[u8]) -> Cow<'_, [u8]> {
    if !raw_has_alias(data) || data[0] & TYPE_MASK == ALIAS_ID && !is_transparent_alias(data) {
        return Cow::Borrowed(data);
    }
    let mut out = Vec::with_capacity(data.len());
    expand_atom(data, &mut out);
    Cow::Owned(out)
}

#[inline]
fn is_transparent_alias(data: &[u8]) -> bool {
    data[0] & TYPE_MASK == ALIAS_ID && data[0] & ALIAS_OPAQUE_FLAG == 0
}

fn alias_handle(data: &[u8]) -> std::sync::Arc<crate::alias::AliasHandle> {
    get_alias_handle(read_alias_token(&data[1..]))
        .expect("Alias handle was released before the expression key")
}

/// Expand one complete encoded atom, retaining all non-alias metadata. Return
/// whether its expansion still contains opaque aliases.
fn expand_atom(data: &[u8], out: &mut Vec<u8>) -> bool {
    if !raw_has_alias(data) {
        out.extend_from_slice(data);
        return false;
    }
    match data[0] & TYPE_MASK {
        ALIAS_ID if data[0] & ALIAS_OPAQUE_FLAG != 0 => {
            out.extend_from_slice(data);
            true
        }
        ALIAS_ID => {
            // BorrowedAtom stores only encoded bytes. Hold a strong registry
            // guard while reading the body and recursively expanding dependencies.
            let handle = alias_handle(data);
            expand_atom(handle.atom().as_view().get_data(), out)
        }
        FUN_ID | MUL_ID | ADD_ID => {
            let is_fun = data[0] & TYPE_MASK == FUN_ID;
            let (name, n_args, mut children) = if is_fun {
                read_packed_pair(function_metadata(data))
            } else {
                let (n_args, _, children) = read_packed_pair(&data[1..]);
                (0, n_args, children)
            };
            let mut expanded = Vec::new();
            if is_fun {
                (name, n_args).write_packed(&mut expanded);
            }
            let mut term_ranges =
                (data[0] & TYPE_MASK == ADD_ID && data[0] & NOT_NORMALIZED == 0).then(Vec::new);
            let mut has_alias = false;
            for _ in 0..n_args {
                let tail = skip_raw_atom(children);
                let child = &children[..children.len() - tail.len()];
                let start = expanded.len();
                has_alias |= expand_atom(child, &mut expanded);
                if let Some(ranges) = &mut term_ranges {
                    ranges.push(start..expanded.len());
                }
                children = tail;
            }
            if let Some(ranges) = &mut term_ranges {
                // Restore encoded monomial order after expansion, including
                // older aliased sums stored in semantic factor order. This does
                // not normalize any expression or invoke user callbacks.
                ranges.sort_by(|left, right| {
                    AtomView::from_with_aliases(&expanded[left.clone()], &NO_ALIASES)
                        .get_term_cmp_slice()
                        .cmp(
                            AtomView::from_with_aliases(&expanded[right.clone()], &NO_ALIASES)
                                .get_term_cmp_slice(),
                        )
                });
            }
            out.push((data[0] & !HAS_ALIAS_FLAG) | if has_alias { HAS_ALIAS_FLAG } else { 0 });
            if is_fun {
                (expanded.len() as u64, 1).write_packed(out);
            } else {
                (n_args, expanded.len() as u64).write_packed(out);
            }
            if let Some(ranges) = term_ranges {
                for range in ranges {
                    out.extend_from_slice(&expanded[range]);
                }
            } else {
                out.extend_from_slice(&expanded);
            }
            has_alias
        }
        POW_ID => {
            let start = out.len();
            out.push(data[0] & !HAS_ALIAS_FLAG);
            let children = &data[1..];
            let exp = skip_raw_atom(children);
            let base = &children[..children.len() - exp.len()];
            let base_has_alias = expand_atom(base, out);
            let exp_has_alias = expand_atom(exp, out);
            if base_has_alias || exp_has_alias {
                out[start] |= HAS_ALIAS_FLAG;
            }
            base_has_alias || exp_has_alias
        }
        _ => unreachable!("Alias flag on a leaf expression"),
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeSet, HashSet},
        hash::{Hash, Hasher},
    };

    use crate::{
        atom::{Atom, AtomCore, AtomOrView, AtomView, FunctionBuilder, SymbolBuilder},
        parse, symbol, wrap_symbol,
    };

    use super::BorrowedAtom;

    #[derive(Default)]
    struct WriteTrace(Vec<Vec<u8>>);

    impl Hasher for WriteTrace {
        fn finish(&self) -> u64 {
            0
        }

        fn write(&mut self, bytes: &[u8]) {
            self.0.push(bytes.to_vec());
        }
    }

    fn hash_trace(value: &(impl Hash + ?Sized)) -> Vec<Vec<u8>> {
        let mut trace = WriteTrace::default();
        value.hash(&mut trace);
        trace.0
    }

    fn assert_key_equivalent(left: &Atom, right: &Atom) {
        assert_eq!(left, right);
        assert_eq!(left.as_view(), right.as_view());
        assert_eq!(hash_trace(left), hash_trace(right));
        assert_eq!(hash_trace(left), hash_trace(&left.as_view()));
        assert_eq!(
            hash_trace(left),
            hash_trace(BorrowedAtom::new(left.as_view()))
        );
        assert_eq!(Ord::cmp(left, right), std::cmp::Ordering::Equal);
        assert_eq!(
            Ord::cmp(&left.as_view(), &right.as_view()),
            std::cmp::Ordering::Equal
        );
        assert_eq!(left.partial_cmp(right), Some(std::cmp::Ordering::Equal));

        let hash_set = HashSet::from([left.clone(), right.clone()]);
        assert_eq!(hash_set.len(), 1);
        assert!(hash_set.contains(BorrowedAtom::new(right.as_view())));
        let tree_set = BTreeSet::from([left.clone(), right.clone()]);
        assert_eq!(tree_set.len(), 1);
        assert!(tree_set.contains(BorrowedAtom::new(right.as_view())));
        let view_set = HashSet::from([left.as_view(), right.as_view()]);
        assert_eq!(view_set.len(), 1);
        assert!(view_set.contains(BorrowedAtom::new(right.as_view())));
        let view_tree = BTreeSet::from([left.as_view(), right.as_view()]);
        assert_eq!(view_tree.len(), 1);
        let mixed_set = HashSet::from([
            AtomOrView::Atom(left.clone()),
            AtomOrView::View(right.as_view()),
        ]);
        assert_eq!(mixed_set.len(), 1);
        assert!(mixed_set.contains(BorrowedAtom::new(right.as_view())));

        macro_rules! check_typed {
            ($variant:ident, $view:ident) => {
                if let (Atom::$variant(left), Atom::$variant(right)) = (left, right) {
                    assert!(left == right);
                    assert_eq!(hash_trace(left), hash_trace(right));
                    assert_eq!(left.$view(), right.$view());
                    assert_eq!(hash_trace(&left.$view()), hash_trace(&right.$view()));
                }
            };
        }
        check_typed!(Alias, to_alias_view);
        check_typed!(Fun, to_fun_view);
        check_typed!(Pow, to_pow_view);
        check_typed!(Mul, to_mul_view);
        check_typed!(Add, to_add_view);
    }

    #[test]
    fn transparent_aliases_are_expression_keys() {
        for body in [
            Atom::Zero,
            Atom::num((-3, 7)),
            parse!("key_alias_x"),
            parse!("key_alias_f(x,y)"),
            parse!("key_alias_x^3"),
            parse!("2*key_alias_x*key_alias_y"),
            parse!("key_alias_x+key_alias_y+3"),
        ] {
            let alias = body.alias(false);
            let chain = alias.alias(false);
            assert_key_equivalent(&alias, &body);
            assert_key_equivalent(&chain, &body);
            assert_key_equivalent(&alias, &chain);
            let detached = AtomView::from(alias.as_view().get_data());
            assert_eq!(detached, body.as_view());
            assert_eq!(hash_trace(&detached), hash_trace(&body));
            assert_eq!(
                Ord::cmp(&detached, &body.as_view()),
                std::cmp::Ordering::Equal
            );
            assert_eq!(hash_trace(&body), hash_trace(body.as_view().get_data()));
            assert_ne!(alias.as_view().get_data(), body.as_view().get_data());
        }

        let x = symbol!("key_alias_symbol");
        let alias = Atom::var(x).alias(false);
        assert_eq!(alias, x);
        assert_eq!(x, alias);
    }

    #[test]
    fn nested_alias_keys_rebuild_headers_without_normalizing() {
        let mut body = Atom::new();
        let function = body.to_fun(symbol!("key_alias_large_body"));
        let child = parse!("key_alias_child");
        for _ in 0..130 {
            function.add_arg(child.as_view());
        }
        let alias = body.alias(false);
        for kind in 0..4 {
            let parent = |child: AtomView<'_>| {
                let mut atom = Atom::new();
                match kind {
                    0 => {
                        let fun = atom.to_fun(symbol!("key_alias_parent"));
                        fun.add_arg(child);
                        fun.add_arg(child);
                    }
                    1 => {
                        atom.to_pow(child, Atom::num(2).as_view());
                    }
                    2 => {
                        let mul = atom.to_mul();
                        mul.extend(Atom::num(3).as_view());
                        mul.extend(child);
                        mul.set_has_coefficient(true);
                    }
                    _ => {
                        let add = atom.to_add();
                        add.extend(Atom::num(3).as_view());
                        add.extend(child);
                    }
                }
                atom
            };
            let plain = parent(body.as_view());
            let aliased = parent(alias.as_view());
            assert!(plain.as_view().needs_normalization());
            assert_key_equivalent(&aliased, &plain);
        }
    }

    #[test]
    fn normalized_alias_sums_use_the_same_expression_key() {
        let terms = [
            parse!("key_sum_f(x)"),
            parse!("key_sum_g(x,y,z,w)"),
            parse!("key_sum_h(x,y)"),
            parse!("key_sum_a(x,y,z,w,v,u)"),
        ];
        let aliases: Vec<_> = terms.iter().map(|term| term.alias(false)).collect();
        for start in 0..terms.len() {
            let aliased_sum = Atom::add_many(&aliases[start..]);
            let plain_sum = Atom::add_many(&terms[start..]);
            assert_key_equivalent(&aliased_sum, &plain_sum);
            let aliased_product = Atom::mul_many(&aliases[start..]);
            let plain_product = Atom::mul_many(&terms[start..]);
            assert_key_equivalent(&aliased_product, &plain_product);
            let f = symbol!("key_sum_outer");
            let aliased = FunctionBuilder::new(f).add_arg(&aliased_sum).finish();
            let plain = FunctionBuilder::new(f).add_arg(&plain_sum).finish();
            assert_key_equivalent(&aliased, &plain);
        }
    }

    #[test]
    fn opaque_keys_and_dirty_flags_remain_distinct() {
        let body = parse!("key_opaque_f(x)");
        let opaque = body.alias(true);
        let wrapped_opaque = opaque.alias(false);
        assert_key_equivalent(&wrapped_opaque, &opaque);
        assert_ne!(opaque, body);
        assert_ne!(Ord::cmp(&opaque, &body), std::cmp::Ordering::Equal);
        assert_eq!(HashSet::from([opaque.clone(), body.clone()]).len(), 2);
        assert_eq!(BTreeSet::from([opaque, body.clone()]).len(), 2);

        let mut dirty = body.clone();
        dirty.set_normalized(false);
        let dirty_alias = dirty.alias(false);
        assert_key_equivalent(&dirty_alias, &dirty);
        assert_ne!(dirty, body);
        assert_ne!(dirty_alias, body);
        assert_ne!(Ord::cmp(&dirty_alias, &body), std::cmp::Ordering::Equal);
        assert_eq!(
            HashSet::from([dirty_alias.clone(), dirty.clone(), body.clone()]).len(),
            2
        );
        assert_eq!(BTreeSet::from([dirty_alias, dirty, body]).len(), 2);
    }

    #[test]
    fn function_ordering_respects_the_comparison_feature() {
        let head = symbol!("key_function_order_feature");
        let number = Atom::num(1u64 << 60);
        let variable = parse!("key_function_order_feature_x");
        let numeric = FunctionBuilder::new(head).add_arg(&number).finish();
        let symbolic = FunctionBuilder::new(head).add_arg(&variable).finish();
        let numeric_alias = FunctionBuilder::new(head)
            .add_arg(number.alias(false))
            .finish();
        let symbolic_alias = FunctionBuilder::new(head)
            .add_arg(variable.alias(false))
            .finish();
        let encoded = numeric
            .as_view()
            .get_data()
            .cmp(symbolic.as_view().get_data());
        let recursive = number.as_view().cmp(&variable.as_view());
        // A large integer has a longer encoding but precedes a variable in the
        // recursive type ordering. Exercise both feature-dependent orderings.
        assert_ne!(encoded, recursive);
        let expected = if cfg!(feature = "full_fn_cmp") {
            recursive
        } else {
            encoded
        };
        for left in [&numeric, &numeric_alias] {
            for right in [&symbolic, &symbolic_alias] {
                assert_eq!(Ord::cmp(left, right), expected);
                assert_eq!(left.as_view().cmp(&right.as_view()), expected);
                assert_eq!(left.as_view().cmp_factors(&right.as_view()), expected);
            }
        }
        assert_key_equivalent(&numeric, &numeric_alias);
        assert_key_equivalent(&symbolic, &symbolic_alias);

        // Registration order, rather than lexical name order, orders heads.
        let first = symbol!("key_function_order_z_first");
        let second = symbol!("key_function_order_a_second");
        assert!(first.get_id() < second.get_id());
        let first = FunctionBuilder::new(first).add_arg(&variable).finish();
        let second = FunctionBuilder::new(second).add_arg(&variable).finish();
        assert!(Ord::cmp(&first, &second).is_lt());
        assert!(first.as_view().cmp(&second.as_view()).is_lt());
        assert!(first.as_view().cmp_factors(&second.as_view()).is_lt());
    }

    #[test]
    fn function_ordering_is_transitive_across_alias_forms() {
        let head = symbol!("key_function_order_outer");
        let bodies = [
            Atom::num(1u64 << 60),
            parse!("key_function_order_x"),
            parse!("key_function_order_x^3"),
            parse!("key_function_order_inner(x,y,z)"),
            parse!("key_function_order_x+key_function_order_y"),
        ];
        let groups: Vec<_> = bodies
            .iter()
            .map(|body| {
                let plain = FunctionBuilder::new(head).add_arg(body).finish();
                let nested = FunctionBuilder::new(head)
                    .add_arg(body.alias(false).alias(false))
                    .finish();
                assert!(nested.as_view().has_alias());
                assert_key_equivalent(&plain, &nested);
                let root = plain.alias(false);
                [plain, nested, root]
            })
            .collect();
        for left in &groups {
            for right in &groups {
                let expected = Ord::cmp(&left[0], &right[0]);
                for a in left {
                    for b in right {
                        assert_eq!(Ord::cmp(a, b), expected);
                        assert_eq!(a.as_view().cmp(&b.as_view()), expected);
                        assert_eq!(a.as_view().cmp_factors(&b.as_view()), expected);
                        assert_eq!(a == b, expected.is_eq());
                    }
                }
            }
        }
        let values: Vec<_> = groups.iter().flatten().collect();
        for a in &values {
            for b in &values {
                for c in &values {
                    if Ord::cmp(a, b).is_le() && Ord::cmp(b, c).is_le() {
                        assert!(Ord::cmp(a, c).is_le());
                    }
                }
            }
        }
        let set: BTreeSet<_> = values.iter().map(|value| (**value).clone()).collect();
        assert_eq!(set.len(), groups.len());
        for value in values {
            assert!(set.contains(BorrowedAtom::new(value.as_view())));
        }
    }

    #[test]
    fn expression_key_operations_do_not_run_normalization_callbacks() {
        let f = SymbolBuilder::new(wrap_symbol!("key_without_callback"))
            .with_normalization_function(|_, _| panic!("key operation normalized a function"))
            .build()
            .unwrap();
        let x = parse!("key_without_callback_x");
        let mut body = Atom::new();
        body.to_fun(f).add_arg(x.as_view());
        let alias = body.alias(false);
        assert_key_equivalent(&alias, &body);
        let mut parent = Atom::new();
        parent.to_fun(f).add_arg(alias.as_view());
        let mut plain_parent = Atom::new();
        plain_parent.to_fun(f).add_arg(body.as_view());
        assert_key_equivalent(&parent, &plain_parent);
    }
}
