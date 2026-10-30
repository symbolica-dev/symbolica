//! Sparse linear combinations of ordered values over an arbitrary coefficient ring.
//!
//! Terms are stored in ascending order, with distinct values and nonzero coefficients.
//! A combination owns one ring and one contiguous term buffer; it has no per-term
//! allocation beyond storage required by the value or coefficient itself.
//!
//! ```
//! use numerica::{domains::integer::Z, linear_combination::LinearCombination};
//!
//! let a = LinearCombination::from_terms(vec![("y", 2.into()), ("x", 3.into())], Z);
//! let b = LinearCombination::from_terms(vec![("x", (-3).into())], Z);
//! assert_eq!((&a + &b).terms(), &[("y", 2.into())]);
//! let replacement = LinearCombination::from_terms(vec![("z", 4.into())], Z);
//! assert_eq!(a.substitute(&"x", &replacement).terms(),
//!            &[("y", 2.into()), ("z", 12.into())]);
//! ```

use std::{
    cmp::Ordering,
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{
    domains::{Field, Ring},
    printer::{PrintOptions, PrintState},
};

/// A finite sum of coefficient-weighted values, sorted by the values' [`Ord`].
///
/// The coefficient ring is stored once. Terms are private so that all safe operations
/// preserve strict ordering and remove zero coefficients. Owned arithmetic moves values;
/// only operations that borrow values require `T: Clone`.
/// Binary addition and subtraction use `O(m + n)` comparisons. Assignment reuses
/// storage for equal supports and disjoint ranges where possible; a general merge
/// allocates one output buffer.
///
/// Binary arithmetic panics if the coefficient rings differ, even for zero operands.
/// Coefficients supplied directly must belong to the supplied ring (the [`Ring`] trait
/// does not provide a general membership check).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LinearCombination<R: Ring, T: Ord> {
    terms: Vec<(T, R::Element)>,
    ring: R,
}

impl<R: Ring, T: Ord> LinearCombination<R, T> {
    /// Construct zero over `ring`, without allocating a term buffer.
    pub fn new(ring: R) -> Self {
        Self {
            terms: Vec::new(),
            ring,
        }
    }

    /// Construct zero with room for at least `capacity` terms.
    pub fn with_capacity(ring: R, capacity: usize) -> Self {
        Self {
            terms: Vec::with_capacity(capacity),
            ring,
        }
    }

    /// Construct from arbitrary `(value, coefficient)` pairs.
    ///
    /// Sorts, combines repeated values, and removes zeros in the supplied buffer,
    /// without cloning values or allocating another term buffer. Already sorted
    /// inputs take linear time; otherwise sorting takes `O(n log n)` comparisons.
    pub fn from_terms(mut terms: Vec<(T, R::Element)>, ring: R) -> Self {
        if !terms.is_sorted_by(|a, b| a.0 <= b.0) {
            terms.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        }
        terms.dedup_by(|a, b| {
            if a.0 == b.0 {
                ring.add_assign(&mut b.1, &a.1);
                true
            } else {
                false
            }
        });
        terms.retain(|(_, c)| !ring.is_zero(c));
        Self { terms, ring }
    }

    /// Adopt a canonical term buffer, validating it in linear time.
    ///
    /// # Panics
    /// Panics if values are not strictly increasing or a coefficient is zero.
    pub fn from_sorted_terms(terms: Vec<(T, R::Element)>, ring: R) -> Self {
        assert!(
            terms.is_sorted_by(|a, b| a.0 < b.0),
            "linear combination values must be strictly increasing"
        );
        assert!(
            terms.iter().all(|(_, c)| !ring.is_zero(c)),
            "linear combination coefficients must be nonzero"
        );
        Self { terms, ring }
    }

    /// The coefficient ring.
    pub fn ring(&self) -> &R {
        &self.ring
    }

    /// The sorted `(value, coefficient)` pairs.
    pub fn terms(&self) -> &[(T, R::Element)] {
        &self.terms
    }

    /// Iterate over terms in ascending order.
    pub fn iter(&self) -> std::slice::Iter<'_, (T, R::Element)> {
        self.terms.iter()
    }

    /// Consume the combination, returning its term buffer without copying.
    pub fn into_terms(self) -> Vec<(T, R::Element)> {
        self.terms
    }

    /// The number of nonzero terms.
    pub fn len(&self) -> usize {
        self.terms.len()
    }

    /// Whether the combination is zero.
    pub fn is_zero(&self) -> bool {
        self.terms.is_empty()
    }

    /// Whether the combination has no terms.
    pub fn is_empty(&self) -> bool {
        self.is_zero()
    }

    /// The number of terms that fit in the current allocation.
    pub fn capacity(&self) -> usize {
        self.terms.capacity()
    }

    /// Set to zero, retaining the allocation for reuse.
    pub fn clear(&mut self) {
        self.terms.clear();
    }

    /// Release spare term capacity, for example after substantial cancellation.
    pub fn shrink_to_fit(&mut self) {
        self.terms.shrink_to_fit();
    }

    /// Look up a coefficient. Missing values have coefficient zero.
    /// Uses a sorted linear scan for at most eight terms and binary search otherwise.
    pub fn coefficient(&self, value: &T) -> Option<&R::Element> {
        if self.len() <= 8 {
            for (key, coefficient) in &self.terms {
                match key.cmp(value) {
                    Ordering::Less => {}
                    Ordering::Equal => return Some(coefficient),
                    Ordering::Greater => return None,
                }
            }
            return None;
        }
        self.terms
            .binary_search_by(|(v, _)| v.cmp(value))
            .ok()
            .map(|i| &self.terms[i].1)
    }

    /// Add one term, maintaining canonical order. Insertion may shift `O(n)` terms.
    /// Prefer [`Self::from_terms`] or [`Self::sum`] when accumulating many terms.
    pub fn add_term(&mut self, value: T, coefficient: R::Element) {
        if self.ring.is_zero(&coefficient) {
            return;
        }
        match self.terms.binary_search_by(|(v, _)| v.cmp(&value)) {
            Ok(i) => {
                self.ring.add_assign(&mut self.terms[i].1, coefficient);
                if self.ring.is_zero(&self.terms[i].1) {
                    self.terms.remove(i);
                }
            }
            Err(i) => self.terms.insert(i, (value, coefficient)),
        }
    }

    /// Multiply coefficients on the right by `scalar`, reusing the term buffer.
    /// Zero products are removed, including in rings with zero divisors.
    pub fn scale_assign(&mut self, scalar: &R::Element) {
        if self.ring.is_zero(scalar) {
            self.clear();
            return;
        }
        if self.ring.is_one(scalar) {
            return;
        }
        self.terms.retain_mut(|(_, c)| {
            self.ring.mul_assign(c, scalar);
            !self.ring.is_zero(c)
        });
    }

    /// Map owned values, combining collisions and restoring sorted order in the result.
    pub fn map_values<U: Ord>(self, mut map: impl FnMut(T) -> U) -> LinearCombination<R, U> {
        LinearCombination::from_terms(
            self.terms.into_iter().map(|(v, c)| (map(v), c)).collect(),
            self.ring,
        )
    }

    /// Add owned combinations using an n-way merge. An empty iterator returns zero over `ring`.
    ///
    /// Use this for a single merge across all operands: a chain such as `a + b + c`
    /// performs successive binary additions instead.
    ///
    /// For `N` input terms and `k` nonempty inputs, this uses at most `O(N log k)`
    /// comparisons and `O(k)` auxiliary storage besides the input and output buffers.
    /// Equal-value groups are processed together, taking `O(N)` comparison work
    /// for fully overlapping inputs. Coefficients at each value are combined with
    /// [`Ring::sum`] to use domain-specific bulk arithmetic.
    /// Values and coefficients are moved, not cloned. Output storage follows the
    /// merged support, rather than reserving the sum of all input lengths.
    ///
    /// # Panics
    /// Panics if any input has a different ring, including empty inputs.
    pub fn sum(combinations: impl IntoIterator<Item = Self>, ring: R) -> Self {
        let mut inputs = Vec::new();
        for combination in combinations {
            assert_eq!(
                ring, combination.ring,
                "linear combination coefficient rings differ"
            );
            if !combination.is_zero() {
                inputs.push(combination);
            }
        }
        if inputs.len() == 1 {
            return inputs.pop().unwrap();
        }
        if inputs
            .windows(2)
            .all(|pair| pair[0].terms.last().unwrap().0 < pair[1].terms.first().unwrap().0)
        {
            // Monotone, disjoint ranges can be concatenated without merge work
            // or intermediate binary sums. Reuse the first owned allocation.
            let capacity = inputs.iter().map(Self::len).sum::<usize>();
            let mut inputs = inputs.into_iter();
            let mut result = inputs.next().unwrap_or_else(|| Self::new(ring));
            result.terms.reserve(capacity - result.len());
            for mut input in inputs {
                result.terms.append(&mut input.terms);
            }
            return result;
        }
        let identical = inputs.iter().skip(1).all(|c| inputs[0].same_support(c));
        let streams = inputs.into_iter().map(|c| c.terms.into_iter());
        let terms = if identical {
            merge_identical(&ring, streams)
        } else {
            merge_many(&ring, streams)
        };
        Self { terms, ring }
    }

    fn assert_same_ring(&self, other: &Self) {
        assert_eq!(
            self.ring, other.ring,
            "linear combination coefficient rings differ"
        );
    }

    fn combine_owned(&mut self, mut other: Self, subtract: bool) {
        self.assert_same_ring(&other);
        if other.is_zero() {
            return;
        }
        if self.is_zero() {
            if subtract {
                other = -other;
            }
            self.terms = other.terms;
            return;
        }
        // Appending disjoint ranges and updating equal supports need no extra buffer.
        if self.terms.last().unwrap().0 < other.terms[0].0 {
            if subtract {
                other = -other;
            }
            self.terms.append(&mut other.terms);
            return;
        }
        if other.terms.last().unwrap().0 < self.terms[0].0 {
            if subtract {
                other = -other;
            }
            other.terms.append(&mut self.terms);
            self.terms = other.terms;
            return;
        }
        if self.same_support(&other) {
            for ((_, c), (_, d)) in self.terms.iter_mut().zip(other.terms) {
                if subtract {
                    self.ring.sub_assign(c, d);
                } else {
                    self.ring.add_assign(c, d);
                }
            }
            self.terms.retain(|(_, c)| !self.ring.is_zero(c));
            return;
        }
        self.terms = merge_two(
            &self.ring,
            std::mem::take(&mut self.terms),
            other.terms,
            subtract,
        );
    }

    fn same_support(&self, other: &Self) -> bool {
        self.len() == other.len() && self.terms.iter().zip(&other.terms).all(|(a, b)| a.0 == b.0)
    }
}

impl<R: Ring, T: Ord + Clone> LinearCombination<R, T> {
    /// Return a scaled copy; see [`Self::scale_assign`].
    pub fn scale(&self, scalar: &R::Element) -> Self {
        if self.ring.is_zero(scalar) {
            return Self::new(self.ring.clone());
        }
        let mut result = self.clone();
        result.scale_assign(scalar);
        result
    }

    /// Add borrowed combinations with the same n-way algorithm as [`Self::sum`].
    /// Only one value is cloned per group of equal input values.
    ///
    /// # Panics
    /// Panics if any input has a different ring, including empty inputs.
    pub fn sum_ref<'a>(combinations: impl IntoIterator<Item = &'a Self>, ring: R) -> Self
    where
        R: 'a,
        T: 'a,
    {
        let inputs: Vec<_> = combinations
            .into_iter()
            .filter(|c| {
                assert_eq!(ring, c.ring, "linear combination coefficient rings differ");
                !c.is_zero()
            })
            .collect();
        let terms = if inputs
            .windows(2)
            .all(|pair| pair[0].terms.last().unwrap().0 < pair[1].terms.first().unwrap().0)
        {
            let capacity = inputs.iter().map(|c| c.len()).sum();
            let mut terms = Vec::with_capacity(capacity);
            for input in inputs {
                terms.extend_from_slice(&input.terms);
            }
            terms
        } else {
            let identical = inputs.iter().skip(1).all(|c| inputs[0].same_support(c));
            let streams = inputs.into_iter().map(|c| c.terms.iter());
            if identical {
                merge_identical(&ring, streams)
            } else {
                merge_many(&ring, streams)
            }
        };
        Self { terms, ring }
    }

    /// Compute `self + scalar * other` with a single merge and fused coefficient operations.
    /// Scalars multiply `other`'s coefficients on the left.
    pub fn add_scaled(&self, other: &Self, scalar: &R::Element) -> Self {
        self.assert_same_ring(other);
        if self.ring.is_zero(scalar) {
            return self.clone();
        }
        Self {
            terms: merge_scaled(&self.ring, &self.terms, &other.terms, scalar),
            ring: self.ring.clone(),
        }
    }

    /// In-place version of [`Self::add_scaled`].
    pub fn add_scaled_assign(&mut self, other: &Self, scalar: &R::Element) {
        self.assert_same_ring(other);
        if other.is_zero() || self.ring.is_zero(scalar) {
            return;
        }
        if self.same_support(other) {
            for ((_, c), (_, d)) in self.terms.iter_mut().zip(&other.terms) {
                self.ring.add_mul_assign(c, scalar, d);
            }
            self.terms.retain(|(_, c)| !self.ring.is_zero(c));
        } else {
            self.terms = merge_scaled(
                &self.ring,
                std::mem::take(&mut self.terms),
                &other.terms,
                scalar,
            );
        }
    }

    /// Replace one value by a linear combination in a single pass.
    ///
    /// If `value` has coefficient `c`, replacement coefficients are multiplied by `c`
    /// on the left. Occurrences of `value` inside `replacement` are not replaced again.
    /// Uses one linear merge without materializing a scaled replacement.
    pub fn substitute(&self, value: &T, replacement: &Self) -> Self {
        self.assert_same_ring(replacement);
        let Ok(i) = self.terms.binary_search_by(|(v, _)| v.cmp(value)) else {
            return self.clone();
        };
        let remaining = self.terms[..i].iter().chain(&self.terms[i + 1..]);
        Self {
            terms: merge_scaled(&self.ring, remaining, &replacement.terms, &self.terms[i].1),
            ring: self.ring.clone(),
        }
    }

    /// Simultaneously replace original values using an n-way merge of lazy scaled terms.
    ///
    /// `replacement` is called once for every original value. `None` keeps that term;
    /// `Some(zero)` removes it. Newly introduced values are not revisited. Multiplication
    /// uses the original coefficient on the left. Auxiliary storage is linear in the
    /// number of original terms, without storing the fully expanded replacements.
    ///
    /// # Panics
    /// Panics if a returned replacement uses a different ring.
    pub fn substitute_all<'a>(
        &'a self,
        mut replacement: impl FnMut(&T) -> Option<&'a Self>,
    ) -> Self {
        let streams = self
            .terms
            .iter()
            .enumerate()
            .map(|(i, (value, coefficient))| {
                let (terms, scalar) = if let Some(r) = replacement(value) {
                    self.assert_same_ring(r);
                    (r.terms(), Some(coefficient))
                } else {
                    (&self.terms[i..i + 1], None)
                };
                terms.iter().map(move |term| ScaledTerm {
                    term,
                    scalar,
                    ring: &self.ring,
                })
            });
        Self {
            terms: merge_many(&self.ring, streams),
            ring: self.ring.clone(),
        }
    }

    fn combine_ref(&mut self, other: &Self, subtract: bool) {
        self.assert_same_ring(other);
        if other.is_zero() {
            return;
        }
        if self.same_support(other) {
            for ((_, c), (_, d)) in self.terms.iter_mut().zip(&other.terms) {
                if subtract {
                    self.ring.sub_assign(c, d);
                } else {
                    self.ring.add_assign(c, d);
                }
            }
            self.terms.retain(|(_, c)| !self.ring.is_zero(c));
            return;
        }
        if self.is_zero() || self.terms.last().unwrap().0 < other.terms[0].0 {
            self.terms.extend(other.terms.iter().map(|(v, c)| {
                (
                    v.clone(),
                    if subtract {
                        self.ring.neg(c)
                    } else {
                        c.clone()
                    },
                )
            }));
            return;
        }
        self.terms = merge_two(
            &self.ring,
            std::mem::take(&mut self.terms),
            &other.terms,
            subtract,
        );
    }
}

// The merge algorithms can move owned terms, copy borrowed terms, or evaluate a
// scaled term only when consumed. In particular, discarded duplicate values are
// never cloned just to obtain their coefficient.
trait Term {
    type Value: Ord;
    type Coefficient;
    fn value(&self) -> &Self::Value;
    fn into_parts(self) -> (Self::Value, Self::Coefficient);
    fn into_coefficient(self) -> Self::Coefficient;

    fn accumulate<R: Ring<Element = Self::Coefficient>>(
        self,
        ring: &R,
        accumulator: &mut Self::Coefficient,
        subtract: bool,
    ) where
        Self: Sized,
    {
        if subtract {
            ring.sub_assign(accumulator, self.into_coefficient());
        } else {
            ring.add_assign(accumulator, self.into_coefficient());
        }
    }
}

impl<T: Ord, C> Term for (T, C) {
    type Value = T;
    type Coefficient = C;
    fn value(&self) -> &T {
        &self.0
    }
    fn into_parts(self) -> (T, C) {
        self
    }
    fn into_coefficient(self) -> C {
        self.1
    }
}

impl<T: Ord + Clone, C: Clone> Term for &(T, C) {
    type Value = T;
    type Coefficient = C;
    fn value(&self) -> &T {
        &self.0
    }
    fn into_parts(self) -> (T, C) {
        self.clone()
    }
    fn into_coefficient(self) -> C {
        self.1.clone()
    }

    fn accumulate<R: Ring<Element = C>>(self, ring: &R, accumulator: &mut C, subtract: bool) {
        // Borrowed coefficient operations avoid copying a potentially large
        // polynomial or multiprecision number merely to add it to an owned value.
        if subtract {
            ring.sub_assign(accumulator, &self.1);
        } else {
            ring.add_assign(accumulator, &self.1);
        }
    }
}

struct ScaledTerm<'a, R: Ring, T> {
    term: &'a (T, R::Element),
    scalar: Option<&'a R::Element>,
    ring: &'a R,
}

impl<R: Ring, T: Ord + Clone> Term for ScaledTerm<'_, R, T> {
    type Value = T;
    type Coefficient = R::Element;
    fn value(&self) -> &T {
        &self.term.0
    }
    fn into_parts(self) -> (T, R::Element) {
        (self.term.0.clone(), self.into_coefficient())
    }
    fn into_coefficient(self) -> R::Element {
        match self.scalar {
            Some(s) => self.ring.mul(s, &self.term.1),
            None => self.term.1.clone(),
        }
    }
}

fn merge_two<R, T, L, H>(ring: &R, left: L, right: H, subtract: bool) -> Vec<(T, R::Element)>
where
    R: Ring,
    T: Ord,
    L: IntoIterator,
    L::Item: Term<Value = T, Coefficient = R::Element>,
    H: IntoIterator,
    H::Item: Term<Value = T, Coefficient = R::Element>,
{
    let mut left = left.into_iter().peekable();
    let mut right = right.into_iter().peekable();
    let mut result = Vec::with_capacity(left.size_hint().0.max(right.size_hint().0));
    while let (Some(a), Some(b)) = (left.peek(), right.peek()) {
        match a.value().cmp(b.value()) {
            Ordering::Less => result.push(left.next().unwrap().into_parts()),
            Ordering::Greater => {
                let (v, mut c) = right.next().unwrap().into_parts();
                if subtract {
                    c = ring.neg(c);
                }
                if !ring.is_zero(&c) {
                    result.push((v, c));
                }
            }
            Ordering::Equal => {
                let (v, mut c) = left.next().unwrap().into_parts();
                right.next().unwrap().accumulate(ring, &mut c, subtract);
                if !ring.is_zero(&c) {
                    result.push((v, c));
                }
            }
        }
    }
    result.extend(left.map(Term::into_parts));
    result.extend(right.filter_map(|term| {
        let (v, c) = term.into_parts();
        let c = if subtract { ring.neg(c) } else { c };
        (!ring.is_zero(&c)).then_some((v, c))
    }));
    result
}

fn merge_scaled<R, T, L>(
    ring: &R,
    left: L,
    right: &[(T, R::Element)],
    scalar: &R::Element,
) -> Vec<(T, R::Element)>
where
    R: Ring,
    T: Ord + Clone,
    L: IntoIterator,
    L::Item: Term<Value = T, Coefficient = R::Element>,
{
    let mut left = left.into_iter().peekable();
    let mut right = right.iter().peekable();
    let mut result = Vec::with_capacity(left.size_hint().0.max(right.len()));
    while let (Some(a), Some(b)) = (left.peek(), right.peek()) {
        match a.value().cmp(&b.0) {
            Ordering::Less => result.push(left.next().unwrap().into_parts()),
            Ordering::Greater => {
                let (v, c) = right.next().unwrap();
                let c = ring.mul(scalar, c);
                if !ring.is_zero(&c) {
                    result.push((v.clone(), c));
                }
            }
            Ordering::Equal => {
                let (v, mut c) = left.next().unwrap().into_parts();
                ring.add_mul_assign(&mut c, scalar, &right.next().unwrap().1);
                if !ring.is_zero(&c) {
                    result.push((v, c));
                }
            }
        }
    }
    result.extend(left.map(Term::into_parts));
    for (v, c) in right {
        let c = ring.mul(scalar, c);
        if !ring.is_zero(&c) {
            result.push((v.clone(), c));
        }
    }
    result
}

struct Cursor<I: Iterator> {
    head: Option<I::Item>,
    rest: I,
}

// A node stores the earliest input with the minimum value in its subtree.
// Updating all equal leaves together costs O(m * (1 + log(k / m))) for a
// group of m contributors, rather than m independent O(log k) heap updates.
const EMPTY_WINNER: usize = usize::MAX;

struct MergeTournament<I: Iterator> {
    cursors: Vec<Cursor<I>>,
    winners: Vec<usize>,
    leaves: usize,
}

impl<I: Iterator> MergeTournament<I>
where
    I::Item: Term,
{
    fn new(streams: impl IntoIterator<Item = I>) -> Self {
        let cursors: Vec<_> = streams
            .into_iter()
            .filter_map(|mut rest| {
                rest.next().map(|head| Cursor {
                    head: Some(head),
                    rest,
                })
            })
            .collect();
        let leaves = cursors.len().next_power_of_two();
        let mut tree = Self {
            cursors,
            winners: vec![EMPTY_WINNER; 2 * leaves],
            leaves,
        };
        tree.rebuild();
        tree
    }

    #[inline]
    fn update(&mut self, node: usize) {
        let left = self.winners[2 * node];
        let right = self.winners[2 * node + 1];
        self.winners[node] = if left == EMPTY_WINNER {
            right
        } else if right == EMPTY_WINNER
            || self.cursors[left].head.as_ref().unwrap().value()
                <= self.cursors[right].head.as_ref().unwrap().value()
        {
            left
        } else {
            right
        };
    }

    fn rebuild(&mut self) {
        for (index, cursor) in self.cursors.iter().enumerate() {
            self.winners[self.leaves + index] = if cursor.head.is_some() {
                index
            } else {
                EMPTY_WINNER
            };
        }
        for node in (1..self.leaves).rev() {
            self.update(node);
        }
    }

    fn advance_equal(
        &mut self,
        node: usize,
        first: usize,
        value: &<I::Item as Term>::Value,
        coefficients: &mut Vec<<I::Item as Term>::Coefficient>,
    ) {
        let winner = self.winners[node];
        // The representative head was taken to own the comparison value. Only
        // its ancestors can cache its index, and they must still be visited.
        if winner == EMPTY_WINNER
            || (winner != first && self.cursors[winner].head.as_ref().unwrap().value() != value)
        {
            return;
        }
        if node >= self.leaves {
            let cursor = &mut self.cursors[winner];
            if winner != first {
                coefficients.push(cursor.head.take().unwrap().into_coefficient());
            }
            cursor.head = cursor.rest.next();
            if cursor.head.is_none() {
                self.winners[node] = EMPTY_WINNER;
            }
        } else {
            // Left-first traversal preserves input coefficient order.
            self.advance_equal(2 * node, first, value, coefficients);
            self.advance_equal(2 * node + 1, first, value, coefficients);
            self.update(node);
        }
    }

    // For dense groups, consume matches and find the next minimum in one
    // sequential pass. Cursors stay in input order even after exhaustion.
    fn advance_dense(
        &mut self,
        first: usize,
        value: &<I::Item as Term>::Value,
        coefficients: &mut Vec<<I::Item as Term>::Coefficient>,
    ) -> usize {
        let mut next = EMPTY_WINNER;
        let mut next_value = None;
        for (index, cursor) in self.cursors.iter_mut().enumerate() {
            if index == first || cursor.head.as_ref().is_some_and(|h| h.value() == value) {
                if index != first {
                    coefficients.push(cursor.head.take().unwrap().into_coefficient());
                }
                cursor.head = cursor.rest.next();
            }
            if let Some(head) = &cursor.head {
                if next_value.is_none_or(|v| head.value() < v) {
                    next = index;
                    next_value = Some(head.value());
                }
            }
        }
        next
    }
}

// Identical supports can be zipped without per-term key comparisons. Stream
// coefficients directly into Ring::sum, avoiding a coefficient scratch buffer.
fn merge_identical<R, T, I>(ring: &R, streams: impl IntoIterator<Item = I>) -> Vec<(T, R::Element)>
where
    R: Ring,
    T: Ord,
    I: Iterator,
    I::Item: Term<Value = T, Coefficient = R::Element>,
{
    let mut streams = streams.into_iter();
    let first = streams.next().unwrap();
    let mut others: Vec<_> = streams.collect();
    let mut result = Vec::new();
    for term in first {
        let (value, coefficient) = term.into_parts();
        let mut coefficients = std::iter::once(coefficient).chain(
            others
                .iter_mut()
                .map(|i| i.next().unwrap().into_coefficient()),
        );
        let coefficient = ring.sum(&mut coefficients);
        // A domain may short-circuit its sum; every stream must still advance.
        coefficients.for_each(drop);
        if !ring.is_zero(&coefficient) {
            result.push((value, coefficient));
        }
    }
    result
}

fn merge_many<R, T, I>(ring: &R, streams: impl IntoIterator<Item = I>) -> Vec<(T, R::Element)>
where
    R: Ring,
    T: Ord,
    I: Iterator,
    I::Item: Term<Value = T, Coefficient = R::Element>,
{
    let mut tournament = MergeTournament::new(streams);
    let mut result = Vec::new();
    let mut coefficients = Vec::new();
    let mut first = tournament.winners[1];
    let mut dense = false;
    // A scan is O(k), so use it following a group with at least k/2 matches.
    // If overlap disappears, charge the unsuccessful scan and O(k) rebuild
    // to that preceding dense group. This preserves the sparse bound.
    let dense_threshold = tournament.cursors.len().div_ceil(2);
    while first != EMPTY_WINNER {
        let term = tournament.cursors[first].head.take().unwrap();
        let (value, coefficient) = term.into_parts();
        coefficients.push(coefficient);
        first = if dense {
            tournament.advance_dense(first, &value, &mut coefficients)
        } else {
            tournament.advance_equal(1, first, &value, &mut coefficients);
            tournament.winners[1]
        };
        let next_dense = coefficients.len() >= dense_threshold;
        if dense && !next_dense && first != EMPTY_WINNER {
            tournament.rebuild();
        }
        dense = next_dense;
        let coefficient = if coefficients.len() == 1 {
            coefficients.pop().unwrap()
        } else {
            ring.sum(coefficients.drain(..))
        };
        if !ring.is_zero(&coefficient) {
            result.push((value, coefficient));
        }
    }
    result
}

macro_rules! impl_binary {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign:ident, $subtract:literal) => {
        impl<R: Ring, T: Ord> $trait for LinearCombination<R, T> {
            type Output = Self;
            fn $method(mut self, rhs: Self) -> Self {
                self.combine_owned(rhs, $subtract);
                self
            }
        }
        impl<R: Ring, T: Ord + Clone> $trait<&Self> for LinearCombination<R, T> {
            type Output = Self;
            fn $method(mut self, rhs: &Self) -> Self {
                self.combine_ref(rhs, $subtract);
                self
            }
        }
        impl<R: Ring, T: Ord + Clone> $trait<LinearCombination<R, T>> for &LinearCombination<R, T> {
            type Output = LinearCombination<R, T>;
            fn $method(self, rhs: LinearCombination<R, T>) -> Self::Output {
                self.assert_same_ring(&rhs);
                LinearCombination {
                    terms: merge_two(&self.ring, &self.terms, rhs.terms, $subtract),
                    ring: rhs.ring,
                }
            }
        }
        impl<R: Ring, T: Ord + Clone> $trait for &LinearCombination<R, T> {
            type Output = LinearCombination<R, T>;
            fn $method(self, rhs: Self) -> Self::Output {
                self.assert_same_ring(rhs);
                LinearCombination {
                    terms: merge_two(&self.ring, &self.terms, &rhs.terms, $subtract),
                    ring: self.ring.clone(),
                }
            }
        }
        impl<R: Ring, T: Ord> $assign_trait for LinearCombination<R, T> {
            fn $assign(&mut self, rhs: Self) {
                self.combine_owned(rhs, $subtract);
            }
        }
        impl<R: Ring, T: Ord + Clone> $assign_trait<&Self> for LinearCombination<R, T> {
            fn $assign(&mut self, rhs: &Self) {
                self.combine_ref(rhs, $subtract);
            }
        }
    };
}
impl_binary!(Add, add, AddAssign, add_assign, false);
impl_binary!(Sub, sub, SubAssign, sub_assign, true);

impl<R: Ring, T: Ord> Neg for LinearCombination<R, T> {
    type Output = Self;
    fn neg(mut self) -> Self {
        for (_, c) in &mut self.terms {
            *c = self.ring.neg(std::mem::replace(c, self.ring.zero()));
        }
        self
    }
}
impl<R: Ring, T: Ord + Clone> Neg for &LinearCombination<R, T> {
    type Output = LinearCombination<R, T>;
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}
impl<R: Ring, T: Ord> MulAssign<&R::Element> for LinearCombination<R, T> {
    fn mul_assign(&mut self, scalar: &R::Element) {
        self.scale_assign(scalar);
    }
}
impl<R: Ring, T: Ord> Mul<&R::Element> for LinearCombination<R, T> {
    type Output = Self;
    fn mul(mut self, scalar: &R::Element) -> Self {
        self.scale_assign(scalar);
        self
    }
}
impl<R: Ring, T: Ord + Clone> Mul<&R::Element> for &LinearCombination<R, T> {
    type Output = LinearCombination<R, T>;
    fn mul(self, scalar: &R::Element) -> Self::Output {
        self.scale(scalar)
    }
}
impl<R: Field, T: Ord> DivAssign<&R::Element> for LinearCombination<R, T> {
    fn div_assign(&mut self, scalar: &R::Element) {
        assert!(
            !self.ring.is_zero(scalar),
            "cannot divide a linear combination by zero"
        );
        if self.is_zero() || self.ring.is_one(scalar) {
            return;
        }
        let inverse = self.ring.inv(scalar);
        self.scale_assign(&inverse);
    }
}
impl<R: Field, T: Ord> Div<&R::Element> for LinearCombination<R, T> {
    type Output = Self;
    fn div(mut self, scalar: &R::Element) -> Self {
        self /= scalar;
        self
    }
}
impl<R: Field, T: Ord + Clone> Div<&R::Element> for &LinearCombination<R, T> {
    type Output = LinearCombination<R, T>;
    fn div(self, scalar: &R::Element) -> Self::Output {
        self.clone() / scalar
    }
}

impl<R: Ring, T: Ord> IntoIterator for LinearCombination<R, T> {
    type Item = (T, R::Element);
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.terms.into_iter()
    }
}
impl<'a, R: Ring, T: Ord> IntoIterator for &'a LinearCombination<R, T> {
    type Item = &'a (T, R::Element);
    type IntoIter = std::slice::Iter<'a, (T, R::Element)>;
    fn into_iter(self) -> Self::IntoIter {
        self.terms.iter()
    }
}

impl<R: Ring, T: Ord + fmt::Display> fmt::Display for LinearCombination<R, T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_zero() {
            return f.write_str("0");
        }
        let options = PrintOptions::from_fmt(f);
        for (i, (value, coefficient)) in self.terms.iter().enumerate() {
            if i != 0 {
                f.write_str(" + ")?;
            }
            f.write_str("(")?;
            self.ring
                .format(coefficient, &options, PrintState::default(), f)?;
            write!(f, ")*({value})")?;
        }
        Ok(())
    }
}
