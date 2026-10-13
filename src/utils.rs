//! Utility traits and structures.

use std::{
    cmp::Ordering,
    ops::{Deref, DerefMut},
};

use dyn_clone::DynClone;

use crate::atom::{Atom, AtomView};

thread_local! {
    static HASH_STATE: ahash::RandomState = ahash::RandomState::new();
}

/// Returns a clone of this thread's randomized hash state for internal maps and sets.
///
/// `RandomState::new()` updates a process-global atomic counter. Initializing once
/// per thread and copying the keys avoids contention when creating many maps.
/// Calls on the same thread reuse the same keys; this does not provide fresh keys
/// for each map. The returned state is owned and can be moved between threads.
#[inline]
pub(crate) fn thread_local_hash_state() -> ahash::RandomState {
    HASH_STATE.with(Clone::clone)
}

/// Find a lexicographically minimal cyclic rotation using a total-order comparator.
/// Returns zero for an empty slice. Uses O(n) comparisons and O(1) extra space.
pub(crate) fn minimum_rotation_by<T>(
    values: &[T],
    mut compare: impl FnMut(&T, &T) -> Ordering,
) -> usize {
    let n = values.len();
    let (mut left, mut right, mut matched) = (0, 1, 0);
    while left < n && right < n && matched < n {
        match compare(
            &values[(left + matched) % n],
            &values[(right + matched) % n],
        ) {
            Ordering::Equal => {
                matched += 1;
                continue;
            }
            // After a mismatch, none of the matched + 1 starts on the losing
            // side can be minimal: each has a strictly smaller counterpart.
            Ordering::Less => {
                right += matched + 1;
                if right == left {
                    right += 1;
                }
            }
            Ordering::Greater => {
                left += matched + 1;
                if left == right {
                    left += 1;
                }
            }
        }
        matched = 0;
    }
    left.min(right)
}

/// A wrapper around a mutable reference that tracks if the value
/// has been mutably accessed.
#[derive(Debug)]
pub struct Settable<'a, T> {
    value: &'a mut T,
    is_set: bool,
}

impl<T> Deref for Settable<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        if !self.is_set {
            panic!("Attempted to deref a Settable that was not set");
        }
        self.value
    }
}

impl<T> DerefMut for Settable<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.is_set = true;
        self.value
    }
}

impl<'a, T> From<&'a mut T> for Settable<'a, T> {
    fn from(value: &'a mut T) -> Self {
        Self {
            value,
            is_set: false,
        }
    }
}

impl<T> Settable<'_, T> {
    /// Return a reference to the value if it has been mutably accessed.
    /// This does not mark the value as set.
    #[inline]
    pub fn get(&self) -> Option<&T> {
        if self.is_set { Some(self.value) } else { None }
    }

    /// Check if the value has been set.
    pub fn is_set(&self) -> bool {
        self.is_set
    }
}

impl Settable<'_, Atom> {
    /// Return the output's view if set, or `fallback` otherwise.
    /// This does not mark the value as set.
    #[inline]
    pub fn as_view_or<'a>(&'a self, fallback: AtomView<'a>) -> AtomView<'a> {
        self.get().map(Atom::as_view).unwrap_or(fallback)
    }
}

/// An enum that contains either an owned value of type `T` or a reference to a value of type `T`.
///
/// Use `Into<BorrowedOrOwned<'b, T>>>` to a accept both owned and borrowed values as arguments.
#[derive(Debug, Clone)]
pub enum BorrowedOrOwned<'a, T> {
    /// An owned value.
    Owned(T),
    /// A value borrowed for the lifetime of this wrapper.
    Borrowed(&'a T),
}

impl<'a, T> From<&'a T> for BorrowedOrOwned<'a, T> {
    fn from(v: &'a T) -> Self {
        Self::Borrowed(v)
    }
}

impl<'a, T> From<&'a mut T> for BorrowedOrOwned<'a, T> {
    fn from(v: &'a mut T) -> Self {
        Self::Borrowed(v)
    }
}

impl<T> From<T> for BorrowedOrOwned<'_, T> {
    fn from(v: T) -> Self {
        Self::Owned(v)
    }
}

impl<T> BorrowedOrOwned<'_, T> {
    /// Get a reference to the value.
    pub fn borrow(&self) -> &T {
        match self {
            BorrowedOrOwned::Owned(t) => t,
            BorrowedOrOwned::Borrowed(t) => t,
        }
    }
}

impl<T: Clone> BorrowedOrOwned<'_, T> {
    /// Yield an owned value.
    pub fn yield_owned(self) -> T {
        match self {
            BorrowedOrOwned::Owned(t) => t,
            BorrowedOrOwned::Borrowed(t) => t.clone(),
        }
    }

    /// Create an owned version of `Self`.
    pub fn into_owned(self) -> BorrowedOrOwned<'static, T> {
        match self {
            BorrowedOrOwned::Owned(t) => BorrowedOrOwned::Owned(t),
            BorrowedOrOwned::Borrowed(t) => BorrowedOrOwned::Owned(t.clone()),
        }
    }
}

impl<T> Deref for BorrowedOrOwned<'_, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.borrow()
    }
}

/// A cloneable function that checks for abort.
pub trait AbortCheck: Fn() -> bool + DynClone + Send + Sync {}
dyn_clone::clone_trait_object!(AbortCheck);
impl<T: Clone + Send + Sync + Fn() -> bool> AbortCheck for T {}

#[cfg(test)]
mod tests {
    use super::minimum_rotation_by;

    #[test]
    fn minimum_rotation_matches_exhaustive_search() {
        // Includes empty, singleton, periodic and repeated-prefix inputs.
        for n in 0..=9 {
            for mut encoding in 0..3usize.pow(n as u32) {
                let values: Vec<_> = (0..n)
                    .map(|_| {
                        let value = (encoding % 3) as u8;
                        encoding /= 3;
                        value
                    })
                    .collect();
                for reverse in [false, true] {
                    let start = minimum_rotation_by(&values, |left, right| {
                        if reverse {
                            right.cmp(left)
                        } else {
                            left.cmp(right)
                        }
                    });
                    let mut actual = values.clone();
                    actual.rotate_left(start);
                    let expected = (0..n)
                        .map(|shift| {
                            let mut rotated = values.clone();
                            rotated.rotate_left(shift);
                            rotated
                        })
                        .min_by(|left, right| {
                            if reverse {
                                right.cmp(left)
                            } else {
                                left.cmp(right)
                            }
                        })
                        .unwrap_or_default();
                    assert_eq!(actual, expected, "input={values:?}, reverse={reverse}");
                }
            }
        }
    }
}
