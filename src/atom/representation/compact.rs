//! Compact borrowed storage for expression collections and byte keys.

use super::{Atom, AtomView};
use crate::atom::AtomCore;

/// A slice-sized expression view with no separate enum tag (16 bytes on x86-64).
///
/// Equality and hashing operate directly on the encoded bytes and match
/// [`AtomView`]'s `Eq` and `Hash` implementations. Both ordering traits use byte
/// order, matching `AtomView`'s `Ord` trait implementation. `AtomView`'s inherent
/// [`cmp`](AtomView::cmp) method and `PartialOrd` instead compare expression
/// structure; use `left.unpack().cmp(&right.unpack())` for that ordering.
///
/// This representation is useful for stored collections and byte keys. Call
/// [`Self::unpack`] for the typed, exhaustively matchable enum; prefer keeping an
/// `AtomView` when repeatedly matching the expression's type.
/// Collections can look up keys by their encoded byte slices through
/// [`Borrow`](std::borrow::Borrow).
///
/// The view borrows immutable expression storage and cannot outlive its owner.
///
/// ```
/// use symbolica::{atom::{AtomView, CompactAtomView}, parse};
/// let atom = parse!("x+y");
/// let view = CompactAtomView::from(atom.as_view());
/// match view.unpack() {
///     AtomView::Add(sum) => assert_eq!(sum.get_nargs(), 2),
///     _ => unreachable!(),
/// }
/// ```
///
/// ```compile_fail
/// use symbolica::{atom::CompactAtomView, parse};
/// let view = {
///     let atom = parse!("x+y");
///     CompactAtomView::from(atom.as_view())
/// };
/// std::hint::black_box(view.get_data());
/// ```
#[repr(transparent)]
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CompactAtomView<'a> {
    data: &'a [u8],
}

impl std::borrow::Borrow<[u8]> for CompactAtomView<'_> {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self.data
    }
}

impl<'a> From<AtomView<'a>> for CompactAtomView<'a> {
    #[inline(always)]
    fn from(view: AtomView<'a>) -> Self {
        Self {
            data: view.get_data(),
        }
    }
}

impl<'a> CompactAtomView<'a> {
    /// Borrow the bytes without decoding their type.
    #[inline(always)]
    pub fn get_data(self) -> &'a [u8] {
        self.data
    }

    /// Decode the type byte for ordinary Rust enum matching.
    #[inline(always)]
    pub fn unpack(self) -> AtomView<'a> {
        AtomView::from(self.data)
    }
}

impl<'a> AtomCore for CompactAtomView<'a> {
    type Output = Atom;

    #[inline(always)]
    fn as_atom_view(&self) -> AtomView<'a> {
        self.unpack()
    }

    fn atom_to_output(&self, atom: Atom) -> Atom {
        atom
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coefficient::Coefficient;
    use crate::parse;
    use std::hash::{Hash, Hasher};

    #[derive(Default)]
    struct Writes(Vec<Vec<u8>>);
    impl Hasher for Writes {
        fn finish(&self) -> u64 {
            0
        }
        fn write(&mut self, data: &[u8]) {
            self.0.push(data.to_vec());
        }
    }

    #[test]
    fn compact_views_preserve_encodings_and_key_semantics() {
        let mut cases = vec![
            Atom::new(),
            Atom::num(3),
            parse!("x"),
            parse!("f(x)"),
            parse!("x*y"),
            parse!("x+y"),
            parse!("x^3"),
            parse!("(x^y)^z"),
            parse!("123456789012345678901234567890/17"),
            parse!("1+2*𝑖"),
            Atom::num(Coefficient::Indeterminate),
            Atom::num(Coefficient::Infinity(None)),
            parse!("1.23"),
        ];
        for n in [1, 50, 1000] {
            cases.push(parse!(&format!(
                "stored_extent({})",
                (0..n)
                    .map(|i| format!("x{i}"))
                    .collect::<Vec<_>>()
                    .join(",")
            )));
        }
        // Dirty builders and their unnormalized flags also survive round trips.
        let mut dirty = Atom::new();
        dirty.to_mul().extend(parse!("x").as_view());
        cases.push(dirty);
        for atom in &cases {
            let view = atom.as_view();
            let compact = CompactAtomView::from(view);
            assert_eq!(compact.get_data(), view.get_data());
            assert_eq!(compact.get_data().as_ptr(), view.get_data().as_ptr());
            assert_eq!(compact.unpack(), view);
            assert_eq!(
                std::mem::discriminant(&compact.unpack()),
                std::mem::discriminant(&view)
            );
            let mut expected = Writes::default();
            view.hash(&mut expected);
            let mut actual = Writes::default();
            compact.hash(&mut actual);
            assert_eq!(expected.0, actual.0);
            let borrowed: &[u8] = std::borrow::Borrow::borrow(&compact);
            assert_eq!(borrowed, view.get_data());
            assert_eq!(borrowed.as_ptr(), view.get_data().as_ptr());
            let mut borrowed_writes = Writes::default();
            borrowed.hash(&mut borrowed_writes);
            assert_eq!(actual.0, borrowed_writes.0);

            let mut map = ahash::HashMap::default();
            map.insert(compact, 42);
            {
                // The query owner can be shorter-lived than the stored key.
                let clone = atom.clone();
                assert!(compact == CompactAtomView::from(clone.as_view()));
                assert_eq!(map.get(clone.as_view().get_data()), Some(&42));
            }
            assert_eq!(map.get(view.get_data()), Some(&42));
            assert_eq!(map.get(&[][..]), None);
            for other in &cases {
                let right = other.as_view();
                assert_eq!(compact == CompactAtomView::from(right), view == right);
                assert_eq!(
                    Ord::cmp(&compact, &CompactAtomView::from(right)),
                    Ord::cmp(&view, &right)
                );
            }
        }
        // Subviews borrow the parent allocation, not a separate copy.
        let AtomView::Fun(f) = cases[3].as_view() else {
            unreachable!()
        };
        let child = f.iter().next().unwrap();
        assert_eq!(
            CompactAtomView::from(child).get_data().as_ptr(),
            child.get_data().as_ptr()
        );
    }

    #[test]
    fn compact_order_is_byte_order_while_unpacked_cmp_is_structural() {
        use std::cmp::Ordering;

        let negative = Atom::num(-2);
        let positive = Atom::num(1);
        let left = negative.as_view();
        let right = positive.as_view();
        let compact_left = CompactAtomView::from(left);
        let compact_right = CompactAtomView::from(right);

        assert_eq!(Ord::cmp(&left, &right), Ordering::Greater);
        assert_eq!(left.cmp(&right), Ordering::Less);
        assert_eq!(PartialOrd::partial_cmp(&left, &right), Some(Ordering::Less));
        assert_eq!(Ord::cmp(&compact_left, &compact_right), Ordering::Greater);
        assert_eq!(
            PartialOrd::partial_cmp(&compact_left, &compact_right),
            Some(Ordering::Greater)
        );
        assert_eq!(
            compact_left.unpack().cmp(&compact_right.unpack()),
            Ordering::Less
        );
    }

    #[test]
    fn compact_view_layout_and_thread_safety() {
        assert_eq!(
            std::mem::size_of::<CompactAtomView<'_>>(),
            2 * std::mem::size_of::<usize>()
        );
        fn send_sync<T: Send + Sync>() {}
        send_sync::<CompactAtomView<'_>>();
    }
}
