//! Deterministic sparse linear-combination benchmarks over Zp.
//!
//! Run with `cargo bench --features symbolica_benchmarks --bench linear_combination`.
//! Each input/replacement has 128 terms; cases vary the number of inputs and their
//! support. Fixture construction and result validation are outside timed regions.
//! Owned addition uses `with_inputs` to exclude the same operand-cloning setup
//! for all three algorithms; borrowed addition includes any required cloning.
//! The materialized substitution baseline deliberately retains all scaled
//! replacements before summing, to compare with the lazy replacement streams.

use std::{fmt, hint::black_box};

use symbolica::{
    domains::finite_field::{FiniteFieldCore, Zp},
    linear_combination::LinearCombination,
};

type Combination = LinearCombination<Zp, usize>;

const TERMS: usize = 128;
const MODULUS: u32 = 2_147_483_647;

#[derive(Clone, Copy, Debug)]
enum Support {
    Interleaved,
    Separated,
    Overlapping,
    Identical,
    Cancelling,
    DenseThenSparse,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    support: Support,
    count: usize,
}

impl Case {
    const fn new(support: Support, count: usize) -> Self {
        Self { support, count }
    }
}

impl fmt::Display for Case {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let support = match self.support {
            Support::Interleaved => "interleaved_disjoint",
            Support::Separated => "separated_disjoint",
            Support::Overlapping => "overlapping",
            Support::Identical => "identical_supports",
            Support::Cancelling => "cancelling_overlap",
            Support::DenseThenSparse => "dense_then_sparse",
        };
        write!(f, "{support}/{}x{TERMS}", self.count)
    }
}

const CASES: [Case; 18] = [
    Case::new(Support::Interleaved, 8),
    Case::new(Support::Interleaved, 64),
    Case::new(Support::Interleaved, 256),
    Case::new(Support::Separated, 8),
    Case::new(Support::Separated, 64),
    Case::new(Support::Separated, 256),
    Case::new(Support::Overlapping, 8),
    Case::new(Support::Overlapping, 64),
    Case::new(Support::Overlapping, 256),
    Case::new(Support::Identical, 8),
    Case::new(Support::Identical, 64),
    Case::new(Support::Identical, 256),
    Case::new(Support::Cancelling, 8),
    Case::new(Support::Cancelling, 64),
    Case::new(Support::Cancelling, 256),
    Case::new(Support::DenseThenSparse, 8),
    Case::new(Support::DenseThenSparse, 64),
    Case::new(Support::DenseThenSparse, 256),
];

fn inputs(case: Case, ring: &Zp) -> Vec<Combination> {
    (0..case.count)
        .map(|input| {
            let terms = (0..TERMS)
                .map(|term| {
                    let value = match case.support {
                        Support::Interleaved => term * case.count + input,
                        Support::Separated => input * TERMS + term,
                        // Shift by at most seven positions: nearly all values
                        // overlap, without identical supports in every input.
                        Support::Overlapping => term + input % 8,
                        Support::Identical => term,
                        Support::Cancelling => term + (input / 2) % 8,
                        Support::DenseThenSparse => {
                            if term < TERMS / 2 {
                                term
                            } else {
                                TERMS / 2 + (term - TERMS / 2) * case.count + input
                            }
                        }
                    };
                    let coefficient = if matches!(case.support, Support::Cancelling) {
                        // Adjacent operands share a support and cancel at every
                        // value not divisible by four, even where pairs overlap.
                        let magnitude = (1 + ((input / 2) * 31 + term * 17) % 1000) as u32;
                        if input % 2 == 1 && value % 4 != 0 {
                            MODULUS - magnitude
                        } else {
                            magnitude
                        }
                    } else {
                        (1 + (input * 31 + term * 17) % 1000) as u32
                    };
                    let coefficient = ring.to_element(coefficient);
                    (value, coefficient)
                })
                .collect();
            Combination::from_sorted_terms(terms, ring.clone())
        })
        .collect()
}

fn repeated_sum(inputs: &[Combination], ring: &Zp) -> Combination {
    let mut result = Combination::new(ring.clone());
    for input in inputs {
        result += input;
    }
    result
}

fn balanced_sum(inputs: &[Combination], ring: &Zp) -> Combination {
    match inputs.len() {
        0 => Combination::new(ring.clone()),
        1 => inputs[0].clone(),
        _ => {
            let (left, right) = inputs.split_at(inputs.len() / 2);
            balanced_sum(left, ring) + balanced_sum(right, ring)
        }
    }
}

fn repeated_sum_owned(inputs: Vec<Combination>, ring: &Zp) -> Combination {
    let mut result = Combination::new(ring.clone());
    for input in inputs {
        result += input;
    }
    result
}

fn balanced_sum_owned(inputs: Vec<Combination>, ring: &Zp) -> Combination {
    fn reduce(
        inputs: &mut impl Iterator<Item = Combination>,
        count: usize,
        ring: &Zp,
    ) -> Combination {
        match count {
            0 => Combination::new(ring.clone()),
            1 => inputs.next().unwrap(),
            _ => reduce(inputs, count / 2, ring) + reduce(inputs, count - count / 2, ring),
        }
    }
    let count = inputs.len();
    reduce(&mut inputs.into_iter(), count, ring)
}

fn sum_fixture(case: Case) -> (Zp, Vec<Combination>) {
    let ring = Zp::new(MODULUS);
    let inputs = inputs(case, &ring);
    let expected = repeated_sum(&inputs, &ring);
    assert_eq!(Combination::sum_ref(&inputs, ring.clone()), expected);
    assert_eq!(balanced_sum(&inputs, &ring), expected);
    assert_eq!(Combination::sum(inputs.clone(), ring.clone()), expected);
    assert_eq!(repeated_sum_owned(inputs.clone(), &ring), expected);
    assert_eq!(balanced_sum_owned(inputs.clone(), &ring), expected);
    if matches!(case.support, Support::Cancelling) {
        assert!(expected.iter().all(|(value, _)| value % 4 == 0));
        assert!((TERMS / 4..=TERMS / 4 + 2).contains(&expected.len()));
    }
    (ring, inputs)
}

fn materialized_substitution(
    source: &Combination,
    replacements: &[Combination],
    ring: &Zp,
) -> Combination {
    let scaled: Vec<_> = source
        .iter()
        .map(|(value, coefficient)| replacements[*value].scale(coefficient))
        .collect();
    Combination::sum(scaled, ring.clone())
}

fn substitution_fixture(case: Case) -> (Zp, Combination, Vec<Combination>) {
    let ring = Zp::new(MODULUS);
    let replacements = inputs(case, &ring);
    let source = Combination::from_sorted_terms(
        (0..case.count)
            .map(|value| {
                // Equal weights preserve the cancellation within adjacent pairs.
                let weight = if matches!(case.support, Support::Cancelling) {
                    value / 2 + 2
                } else {
                    value + 2
                };
                (value, ring.to_element(weight as u32))
            })
            .collect(),
        ring.clone(),
    );
    assert_eq!(
        source.substitute_all(|value| Some(&replacements[*value])),
        materialized_substitution(&source, &replacements, &ring)
    );
    (ring, source, replacements)
}

fn main() {
    divan::main();
}

#[divan::bench_group(sample_count = 20, sample_size = 1, skip_ext_time)]
mod addition {
    use super::*;

    #[divan::bench(args = CASES)]
    fn n_way_sum_ref(bencher: divan::Bencher, case: Case) {
        let (ring, inputs) = sum_fixture(case);
        bencher.bench_local(|| Combination::sum_ref(black_box(&inputs), ring.clone()));
    }

    #[divan::bench(args = CASES)]
    fn repeated_add_assign(bencher: divan::Bencher, case: Case) {
        let (ring, inputs) = sum_fixture(case);
        bencher.bench_local(|| repeated_sum(black_box(&inputs), &ring));
    }

    #[divan::bench(args = CASES)]
    fn balanced_pairwise(bencher: divan::Bencher, case: Case) {
        let (ring, inputs) = sum_fixture(case);
        bencher.bench_local(|| balanced_sum(black_box(&inputs), &ring));
    }
}

#[divan::bench_group(sample_count = 20, sample_size = 1, skip_ext_time)]
mod owned_addition {
    use super::*;

    #[divan::bench(args = CASES)]
    fn n_way_sum(bencher: divan::Bencher, case: Case) {
        let (ring, inputs) = sum_fixture(case);
        bencher
            .with_inputs(|| inputs.clone())
            .bench_local_values(|inputs| Combination::sum(black_box(inputs), ring.clone()));
    }

    #[divan::bench(args = CASES)]
    fn repeated_add_assign(bencher: divan::Bencher, case: Case) {
        let (ring, inputs) = sum_fixture(case);
        bencher
            .with_inputs(|| inputs.clone())
            .bench_local_values(|inputs| repeated_sum_owned(black_box(inputs), &ring));
    }

    #[divan::bench(args = CASES)]
    fn balanced_pairwise(bencher: divan::Bencher, case: Case) {
        let (ring, inputs) = sum_fixture(case);
        bencher
            .with_inputs(|| inputs.clone())
            .bench_local_values(|inputs| balanced_sum_owned(black_box(inputs), &ring));
    }
}

#[divan::bench_group(sample_count = 20, sample_size = 1, skip_ext_time)]
mod substitution {
    use super::*;

    #[divan::bench(args = CASES)]
    fn lazy(bencher: divan::Bencher, case: Case) {
        let (_, source, replacements) = substitution_fixture(case);
        bencher.bench_local(|| {
            let replacements = black_box(&replacements);
            black_box(&source).substitute_all(|value| Some(&replacements[*value]))
        });
    }

    #[divan::bench(args = CASES)]
    fn materialized(bencher: divan::Bencher, case: Case) {
        let (ring, source, replacements) = substitution_fixture(case);
        bencher.bench_local(|| {
            materialized_substitution(black_box(&source), black_box(&replacements), &ring)
        });
    }
}

/// Each timing covers 64 mixed queries: half hits spread over the support,
/// half misses before, between, and after values. String keys share a prefix.
#[divan::bench_group(sample_count = 40, sample_size = 1, skip_ext_time)]
mod coefficient_lookup {
    use super::*;
    use symbolica::domains::finite_field::FiniteFieldElement;

    const SIZES: [usize; 9] = [0, 1, 2, 4, 8, 16, 32, 64, 128];

    trait Key: Ord {
        fn from_index(index: usize) -> Self;
    }

    impl Key for usize {
        fn from_index(index: usize) -> Self {
            index
        }
    }

    impl Key for String {
        fn from_index(index: usize) -> Self {
            format!("symbolica::coefficient::basis::{index:08}")
        }
    }

    fn benchmark<K: Key>(
        bencher: divan::Bencher,
        count: usize,
        lookup: impl Fn(&LinearCombination<Zp, K>, &K) -> Option<FiniteFieldElement<u32>>,
    ) {
        let ring = Zp::new(MODULUS);
        let input = LinearCombination::from_sorted_terms(
            (0..count)
                .map(|i| (K::from_index(4 * (i + 1)), ring.to_element(i as u32 + 1)))
                .collect(),
            ring,
        );
        let queries: Vec<_> = (0..64)
            .map(|i| {
                let position = (i * 37 + 11) % count.max(1);
                K::from_index(match i % 8 {
                    0 => 4,
                    1 => 0,
                    2 => 4 * (count / 2 + 1),
                    3 => 4 * (position + 1) + 1,
                    4 => 4 * count,
                    5 => 4 * (count + 1),
                    6 => 4 * (position + 1),
                    _ => 4 * (count / 2 + 1) + 1,
                })
            })
            .collect();
        for query in &queries {
            assert_eq!(lookup(&input, query), input.coefficient(query).copied());
        }
        bencher.bench_local(|| {
            let input = black_box(&input);
            let mut checksum = 0u64;
            for query in black_box(&queries) {
                checksum += lookup(input, query).map_or(0, |c| u64::from(*c.inner()));
            }
            checksum
        });
    }

    #[divan::bench(types = [usize, String], args = SIZES)]
    fn current<K: Key>(bencher: divan::Bencher, count: usize) {
        benchmark(bencher, count, |input: &LinearCombination<Zp, K>, query| {
            input.coefficient(query).copied()
        });
    }

    #[divan::bench(types = [usize, String], args = SIZES)]
    fn binary<K: Key>(bencher: divan::Bencher, count: usize) {
        benchmark(bencher, count, |input: &LinearCombination<Zp, K>, query| {
            input
                .terms()
                .binary_search_by(|(value, _)| value.cmp(query))
                .ok()
                .map(|i| input.terms()[i].1)
        });
    }

    #[divan::bench(types = [usize, String], args = SIZES)]
    fn linear<K: Key>(bencher: divan::Bencher, count: usize) {
        benchmark(bencher, count, |input: &LinearCombination<Zp, K>, query| {
            input
                .iter()
                .find(|(value, _)| value == query)
                .map(|(_, coefficient)| *coefficient)
        });
    }

    #[divan::bench(types = [usize, String], args = SIZES)]
    fn sorted_scan<K: Key>(bencher: divan::Bencher, count: usize) {
        benchmark(bencher, count, |input: &LinearCombination<Zp, K>, query| {
            for (value, coefficient) in input.iter() {
                match value.cmp(query) {
                    std::cmp::Ordering::Less => {}
                    std::cmp::Ordering::Equal => return Some(*coefficient),
                    std::cmp::Ordering::Greater => return None,
                }
            }
            None
        });
    }
}
