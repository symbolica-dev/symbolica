//! Provides combinatorial utilities for generating combinations, permutations, and partitions of sets.
//!
//! # Examples
//!
//! Combinations without replacements:
//!
//! ```rust
//! # use numerica::combinatorics::CombinationIterator;
//!
//! let mut c = CombinationIterator::new(4, 3);
//! let mut combinations = vec![];
//! while let Some(a) = c.next() {
//!     combinations.push(a.to_vec());
//! }
//!
//! let ans = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
//!
//! assert_eq!(combinations, ans);
//! ```
//!
//! Partitions:
//!
//! ```rust
//! # use numerica::combinatorics::partitions;
//!
//! let p = partitions(
//!     &[1, 1, 1, 2, 2],
//!     &[('f', 2), ('g', 2), ('f', 1)],
//!     false,
//!     false,
//! );
//!
//! let res = vec![
//!     (3.into(), vec![('f', vec![1]), ('f', vec![1, 1]), ('g', vec![2, 2])]),
//!     (12.into(), vec![('f', vec![1]), ('f', vec![1, 2]), ('g', vec![1, 2])]),
//!     (3.into(), vec![('f', vec![1]), ('f', vec![2, 2]), ('g', vec![1, 1])]),
//!     (6.into(), vec![('f', vec![2]), ('f', vec![1, 1]), ('g', vec![1, 2])]),
//!     (6.into(), vec![('f', vec![2]), ('f', vec![1, 2]), ('g', vec![1, 1])]),
//! ];
//!
//! assert_eq!(p, res);
//! ```
use ahash::HashMap;
use smallvec::SmallVec;
use std::{cmp::Ordering, hash::Hash};

use crate::domains::integer::Integer;

/// An iterator type for generating combinations of indices without replacement.
///
/// # Examples
///
/// Create an iterator to generate combinations of 3 elements from a total of 4:
/// ```rust
/// # use numerica::combinatorics::CombinationIterator;
/// let mut combos = CombinationIterator::new(4, 3);
///
/// while let Some(c) = combos.next() {
///     println!("{:?}", c);
/// }
///
/// // The combinations output is:
/// // [0, 1, 2]
/// // [0, 1, 3]
/// // [0, 2, 3]
/// // [1, 2, 3]
/// ```
pub struct CombinationIterator {
    n: usize,
    indices: Vec<usize>,
    init: bool,
}

impl CombinationIterator {
    /// Creates a new `CombinationIterator` for generating combinations of `k` elements from a set of `n` elements.
    pub fn new(n: usize, k: usize) -> CombinationIterator {
        CombinationIterator {
            indices: (0..k).collect(),
            n,
            init: false,
        }
    }

    /// Advances the iterator and returns the next combination.
    pub fn next(&mut self) -> Option<&[usize]> {
        if self.indices.is_empty() || self.indices.len() > self.n {
            return None;
        }

        if !self.init {
            self.init = true;

            return Some(&self.indices);
        }

        if self.indices.is_empty() {
            return None;
        }

        let mut done = true;
        for (i, v) in self.indices.iter().enumerate().rev() {
            if *v < self.n - self.indices.len() + i {
                let a = *v + 1;
                for (p, vv) in &mut self.indices[i..].iter_mut().enumerate() {
                    *vv = a + p;
                }

                done = false;
                break;
            }
        }

        if done { None } else { Some(&self.indices) }
    }
}

/// An iterator that generates combinations of size `k` from a sequence of items, allowing repeat selections.
///
/// The iterator will produce each combination in ascending order
/// so that only unique combinations are generated, even though
/// each pick is allowed to repeat items.
///
/// # Example
///
/// ```rust
/// # use numerica::combinatorics::CombinationWithReplacementIterator;
///
/// let mut comb_iter = CombinationWithReplacementIterator::new(3, 2);
/// while let Some(combination) = comb_iter.next() {
///      println!("{:?}", combination);
/// }
/// ```
/// This would print out combinations like `[0, 0], [0, 1], [0, 2], [1, 1], [1, 2]`, etc.
pub struct CombinationWithReplacementIterator {
    indices: SmallVec<[u32; 10]>,
    k: u32,
    init: bool,
}

impl CombinationWithReplacementIterator {
    /// Creates a new `CombinationWithReplacementIterator` for generating combinations of `k` elements from a set of `n` elements with replacement.
    pub fn new(n: usize, k: u32) -> CombinationWithReplacementIterator {
        CombinationWithReplacementIterator {
            indices: (0..n).map(|_| 0).collect(),
            k,
            init: false,
        }
    }

    /// Advances the iterator and returns the next combination with replacement.
    pub fn next(&mut self) -> Option<&[u32]> {
        if self.indices.is_empty() {
            return None;
        }

        if !self.init {
            self.init = true;
            self.indices[0] = self.k;
            return Some(&self.indices);
        }

        if self.k == 0 {
            return None;
        }

        // find the last non-zero index that is not at the end
        let mut i = self.indices.len() - 1;
        while self.indices[i] == 0 {
            i -= 1;
        }

        // cannot move to the right more
        // find the next index
        let mut last_val = 0;
        if i == self.indices.len() - 1 {
            last_val = self.indices[i];
            self.indices[i] = 0;

            if self.indices.len() == 1 {
                return None;
            }

            i = self.indices.len() - 2;
            while self.indices[i] == 0 {
                if i == 0 {
                    return None;
                }

                i -= 1;
            }
        }

        self.indices[i] -= 1;
        self.indices[i + 1] = last_val + 1;

        Some(&self.indices)
    }
}

/// Generate all unique permutations of the `list` entries.
///
/// The combinatorial prefactor of each element is `list.len()! / out.len()` where
/// `out` is the returned list.
pub fn unique_permutations<T: Clone + Hash + Ord>(list: &[T]) -> (Integer, Vec<Vec<T>>) {
    let mut unique: HashMap<&T, usize> = HashMap::default();
    for e in list {
        *unique.entry(e).or_insert(0) += 1;
    }
    let mut unique: Vec<_> = unique.into_iter().collect();
    unique.sort();

    // determine pre-factor
    let mut prefactor = Integer::one();
    for (_, count) in &unique {
        prefactor *= &Integer::factorial(*count as u32);
    }

    let mut out = vec![];
    unique_permutations_impl(
        &mut unique,
        &mut Vec::with_capacity(list.len()),
        list.len(),
        &mut out,
    );
    (prefactor, out)
}

fn unique_permutations_impl<T: Clone>(
    unique: &mut Vec<(&T, usize)>,
    accum: &mut Vec<T>,
    len: usize,
    out: &mut Vec<Vec<T>>,
) {
    if accum.len() == len {
        out.push(accum.to_vec());
    }

    for i in 0..unique.len() {
        let (entry, count) = &mut unique[i];
        if *count > 0 {
            *count -= 1;
            accum.push(entry.clone());
            unique_permutations_impl(unique, accum, len, out);
            accum.pop();
            unique[i].1 += 1;
        }
    }
}

/// Partition the unordered list `elements` into named bins of unordered lists with a given length,
/// returning all partitions and their multiplicity.
///
/// # Arguments
///
/// * `elements` - A slice of elements to partition.
/// * `bins` - A slice of tuples where each tuple contains a bin identifier and the number of elements in that bin.
/// * `fill_last` - A boolean flag indicating whether to add all remaining elements to the last bin if the elements are larger than the bins.
/// * `repeat` - A boolean flag indicating whether to repeat the bins to exactly fit all elements, if possible.
///
/// # Returns
///
/// A `Vec` of tuples where each tuple contains:
/// * An `Integer` representing the multiplicity of the partition.
/// * A `Vec` of tuples where each tuple contains a bin identifier and a `Vec` of elements in that bin.
///
/// # Example
///
/// ```
/// # use numerica::combinatorics::partitions;
/// let result = partitions(
///     &[1, 1, 1, 2, 2],
///     &[('f', 2), ('g', 2), ('f', 1)],
///     false,
///     false
/// );
/// ```
/// generates all possible ways to partition the elements of three sets
/// and yields:
/// ```plain
/// [(3, [('g', [1]), ('f', [1, 1]), ('f', [2, 2])]), (6, [('g', [1]), ('f', [1, 2]),
/// ('f', [1, 2])]), (6, [('g', [2]), ('f', [1, 1]), ('f', [1, 2])])]
/// ```
///
/// This generates all possible ways to partition the elements into the specified bins.
pub fn partitions<T: Ord + Hash + Clone, B: Ord + Hash + Clone>(
    elements: &[T],
    bins: &[(B, usize)],
    fill_last: bool,
    repeat: bool,
) -> Vec<(Integer, Vec<(B, Vec<T>)>)> {
    if bins.is_empty() {
        return vec![];
    }

    let bin_sum = bins.iter().map(|b| b.1).sum::<usize>();
    match elements.len().cmp(&bin_sum) {
        Ordering::Less => {
            return vec![];
        }
        Ordering::Equal => {}
        Ordering::Greater => {
            if !fill_last && (!repeat || !elements.len().is_multiple_of(bin_sum)) {
                return vec![];
            }
        }
    }

    // create groups of equal elements
    let mut element_groups: HashMap<T, usize> = HashMap::default();
    for e in elements {
        *element_groups.entry(e.clone()).or_insert(0) += 1;
    }

    let mut element_sorted: Vec<(T, usize)> = element_groups.into_iter().collect();
    element_sorted.sort();

    let mut sorted_bins = bins.to_vec();

    // extend the bins if needed
    if fill_last {
        let last_bin = sorted_bins.last_mut().unwrap();
        last_bin.1 += elements.len() - bin_sum;
    }

    if repeat {
        for _ in 1..elements.len() / bin_sum {
            sorted_bins.extend_from_slice(bins);
        }
    }

    // sort the bins from largest to smallest and based on the bin id
    sorted_bins.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));

    fn fill_bin<T: Clone>(
        len: usize,
        elems: &mut [(T, usize)],
        accum: &mut Vec<T>,
        result: &mut Vec<Vec<T>>,
    ) {
        if len == 0 {
            result.push(accum.clone());
            return;
        }

        for i in 0..elems.len() {
            let (name, count) = &mut elems[i];
            if *count > 0 {
                *count -= 1;
                accum.push(name.clone());
                fill_bin(len - 1, &mut elems[i..], accum, result);
                accum.pop();
                elems[i].1 += 1;
            }
        }
    }

    fn fill_rec<T: Ord + Clone, B: Clone + Eq>(
        bins: &[(B, usize)],
        elems: &mut [(T, usize)],
        single_bin_accum: &mut Vec<T>,
        single_bin_fill: &mut Vec<Vec<T>>,
        accum: &mut Vec<(B, Vec<T>)>,
        result: &mut Vec<(Integer, Vec<(B, Vec<T>)>)>,
    ) {
        if bins.is_empty() {
            if elems.iter().all(|x| x.1 == 0) {
                result.push((Integer::one(), accum.clone()));
            }
            return;
        }
        debug_assert!(elems.iter().any(|x| x.1 > 0));

        let (bin_id, bin_len) = &bins[0];

        // find all possible ways to fill fun_len
        fill_bin(*bin_len, elems, single_bin_accum, single_bin_fill);

        let mut new_bin_fill = vec![];
        for a in single_bin_fill.drain(..) {
            // make sure we generate a descending list
            if let Some(l) = accum.last()
                && l.0 == *bin_id
                && a.len() == l.1.len()
                && a < l.1
            {
                continue;
            }

            // remove uses from the counters
            for x in &a {
                elems.iter_mut().find(|e| e.0 == *x).unwrap().1 -= 1;
            }

            accum.push((bin_id.clone(), a.clone()));
            fill_rec(
                &bins[1..],
                elems,
                single_bin_accum,
                &mut new_bin_fill,
                accum,
                result,
            );
            accum.pop();

            for x in &a {
                elems.iter_mut().find(|e| e.0 == *x).unwrap().1 += 1;
            }
        }
    }

    let mut res = vec![];
    fill_rec(
        &mut sorted_bins,
        &mut element_sorted,
        &mut vec![],
        &mut vec![],
        &mut vec![],
        &mut res,
    );

    // compute the prefactor
    let mut counter = vec![];
    let mut bin_groups: HashMap<&(B, Vec<T>), usize> = HashMap::default();
    for (pref, sol) in &mut res {
        for (e, _) in &element_sorted {
            counter.clear();
            for (_, bin) in &*sol {
                let c = bin.iter().filter(|be| *be == e).count();
                if c > 0 {
                    counter.push(c as u32);
                }
            }
            *pref *= &Integer::multinom(&counter);
        }

        // count the number of unique bins
        for named_bin in &*sol {
            *bin_groups.entry(named_bin).or_insert(0) += 1;
        }

        for (_, p) in bin_groups.drain() {
            *pref /= &Integer::new(p as i64);
        }
    }

    res
}

#[cfg(test)]
mod test {
    use super::{CombinationIterator, partitions};

    #[test]
    fn combinations() {
        let mut c = CombinationIterator::new(4, 3);
        let mut combinations = vec![];
        while let Some(a) = c.next() {
            combinations.push(a.to_vec());
        }

        let ans = vec![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

        assert_eq!(combinations, ans);
    }

    #[test]
    fn partitions_no_fill() {
        let p = partitions(
            &[1, 1, 1, 2, 2],
            &[('f', 2), ('g', 2), ('f', 1)],
            false,
            false,
        );

        let res = vec![
            (
                3.into(),
                vec![('f', vec![1]), ('f', vec![1, 1]), ('g', vec![2, 2])],
            ),
            (
                12.into(),
                vec![('f', vec![1]), ('f', vec![1, 2]), ('g', vec![1, 2])],
            ),
            (
                3.into(),
                vec![('f', vec![1]), ('f', vec![2, 2]), ('g', vec![1, 1])],
            ),
            (
                6.into(),
                vec![('f', vec![2]), ('f', vec![1, 1]), ('g', vec![1, 2])],
            ),
            (
                6.into(),
                vec![('f', vec![2]), ('f', vec![1, 2]), ('g', vec![1, 1])],
            ),
        ];

        assert_eq!(p, res);
    }
}
