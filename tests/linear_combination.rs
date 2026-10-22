use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fmt,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering as AtomicOrdering},
    },
};

use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        EuclideanDomain, Ring, RingOps, Set,
        atom::AtomField,
        finite_field::{FiniteFieldCore, Zp},
        integer::{Integer, Z},
        rational::Q,
        rational_polynomial::RationalPolynomialField,
    },
    linear_combination::LinearCombination,
    parse,
    printer::{PrintOptions, PrintState},
    symbol,
};

type Combination = LinearCombination<Z, i32>;

fn lc(terms: &[(i32, i64)]) -> Combination {
    LinearCombination::from_terms(
        terms
            .iter()
            .map(|&(value, coefficient)| (value, coefficient.into()))
            .collect(),
        Z,
    )
}

fn assert_terms(actual: &Combination, expected: &[(i32, i64)]) {
    assert_eq!(actual.terms(), lc(expected).terms());
    assert!(
        actual
            .terms()
            .windows(2)
            .all(|terms| terms[0].0 < terms[1].0)
    );
    assert!(
        actual
            .terms()
            .iter()
            .all(|(_, coefficient)| !Z.is_zero(coefficient))
    );
}

#[test]
fn constructors_canonicalize_and_support_storage_reuse() {
    let mut value = lc(&[(3, 2), (1, 4), (2, 0), (3, -2), (1, 5), (4, -1)]);
    assert_terms(&value, &[(1, 9), (4, -1)]);
    assert_eq!(value.ring(), &Z);
    assert_eq!(value.len(), 2);
    assert_eq!(value.coefficient(&1), Some(&Integer::from(9)));
    assert_eq!(value.coefficient(&2), None);
    let capacity = value.capacity();
    value.clear();
    assert!(value.is_zero());
    assert_eq!(value.capacity(), capacity);
    value.shrink_to_fit();
    assert_eq!(value.capacity(), 0);

    let reserved = Combination::with_capacity(Z, 128);
    assert!(reserved.is_zero());
    assert!(reserved.capacity() >= 128);
    assert_eq!(Combination::new(Z).capacity(), 0);
    let sorted =
        LinearCombination::from_sorted_terms(vec![(1, Integer::from(3)), (4, 7.into())], Z);
    assert_eq!(sorted.into_terms(), vec![(1, 3.into()), (4, 7.into())]);
}

#[test]
fn coefficient_lookup_finds_all_values_and_gaps_across_small_and_large_supports() {
    for count in 0..=128usize {
        let integers = LinearCombination::from_sorted_terms(
            (0..count)
                .map(|index| (3 * index + 2, Integer::from(index as i64 + 1)))
                .collect(),
            Z,
        );
        let strings = LinearCombination::from_sorted_terms(
            (0..count)
                .map(|index| {
                    (
                        format!("{:03}", 3 * index + 2),
                        Integer::from(index as i64 + 1),
                    )
                })
                .collect(),
            Z,
        );
        // Covers every hit, two gaps between adjacent values, and misses
        // before and after the support, including zero- and one-term cases.
        for key in 0..3 * count + 4 {
            let expected = if key >= 2 && (key - 2) % 3 == 0 && (key - 2) / 3 < count {
                Some(Integer::from(((key - 2) / 3) as i64 + 1))
            } else {
                None
            };
            assert_eq!(
                integers.coefficient(&key),
                expected.as_ref(),
                "count={count}, key={key}"
            );
            assert_eq!(
                strings.coefficient(&format!("{key:03}")),
                expected.as_ref(),
                "count={count}, key={key}"
            );
        }
    }
}

#[test]
fn sorted_constructor_rejects_invalid_terms() {
    for terms in [vec![(2, 1), (1, 1)], vec![(1, 1), (1, 2)], vec![(1, 0)]] {
        assert!(
            std::panic::catch_unwind(|| {
                LinearCombination::from_sorted_terms(
                    terms
                        .into_iter()
                        .map(|(v, c)| (v, Integer::from(c)))
                        .collect(),
                    Z,
                )
            })
            .is_err()
        );
    }
}

#[test]
fn binary_arithmetic_handles_every_ownership_combination_and_cancellation() {
    let left = lc(&[(1, 2), (3, 5), (5, -4)]);
    let right = lc(&[(2, 7), (3, -5), (5, 6), (8, -2)]);
    let sum = [(1, 2), (2, 7), (5, 2), (8, -2)];
    for actual in [
        &left + &right,
        left.clone() + &right,
        &left + right.clone(),
        left.clone() + right.clone(),
    ] {
        assert_terms(&actual, &sum);
    }
    let difference = [(1, 2), (2, -7), (3, 10), (5, -10), (8, 2)];
    for actual in [
        &left - &right,
        left.clone() - &right,
        &left - right.clone(),
        left.clone() - right.clone(),
    ] {
        assert_terms(&actual, &difference);
    }
    let mut actual = left.clone();
    actual += &right;
    assert_terms(&actual, &sum);
    actual -= right.clone();
    assert_eq!(actual, left);
    actual += right.clone();
    actual -= &right;
    assert_eq!(actual, left);
    assert!((&left - &left).is_zero());
    assert!((left.clone() + -&left).is_zero());
    assert_eq!(-left.clone(), -&left);

    let zero = Combination::new(Z);
    assert_eq!(&left + &zero, left);
    assert_eq!(&zero + &left, left);
    assert_eq!(&left - &zero, left);
    assert_eq!(&zero - &left, -&left);
}

#[test]
fn scaling_fused_addition_and_field_division() {
    let value = lc(&[(1, 3), (4, -2)]);
    let factor = Integer::from(-3);
    assert_terms(&value.scale(&factor), &[(1, -9), (4, 6)]);
    assert_eq!(&value * &factor, value.clone() * &factor);
    let mut scaled = value.clone();
    scaled *= &factor;
    assert_eq!(scaled, value.scale(&factor));
    scaled.scale_assign(&Z.zero());
    assert!(scaled.is_zero());
    assert_eq!(value.scale(&Z.one()), value);

    let other = lc(&[(1, 1), (2, 5)]);
    let expected = lc(&[(2, -15), (4, -2)]);
    assert_eq!(value.add_scaled(&other, &factor), expected);
    let mut assigned = value.clone();
    assigned.add_scaled_assign(&other, &factor);
    assert_eq!(assigned, expected);
    assert_eq!(value.add_scaled(&other, &Z.zero()), value);

    let rational =
        LinearCombination::from_terms(vec![(1, Q.nth(3.into())), (2, Q.nth((-4).into()))], Q);
    let divisor = Q.nth(2.into());
    let divided = &rational / &divisor;
    assert_eq!(divided, rational.clone() / &divisor);
    assert_eq!(&divided * &divisor, rational);
    let mut assigned = rational.clone();
    assigned /= &divisor;
    assert_eq!(assigned, divided);
}

#[test]
fn n_way_sum_handles_empty_inputs_cancellation_and_overlap() {
    assert!(Combination::sum(std::iter::empty(), Z).is_zero());
    assert!(Combination::sum_ref(std::iter::empty(), Z).is_zero());
    let inputs = vec![
        lc(&[(1, 2), (3, 9)]),
        Combination::new(Z),
        lc(&[(1, -2), (2, 5)]),
        lc(&[(2, -5), (3, 1)]),
    ];
    assert_terms(&Combination::sum_ref(inputs.iter(), Z), &[(3, 10)]);
    assert_terms(&Combination::sum(inputs, Z), &[(3, 10)]);

    // A fully overlapping sum should not reserve space for every input term.
    let inputs: Vec<_> = (0..128)
        .map(|_| lc(&[(1, 1), (2, 1), (3, 1), (4, 1)]))
        .collect();
    let sum = Combination::sum(inputs, Z);
    assert_terms(&sum, &[(1, 128), (2, 128), (3, 128), (4, 128)]);
    assert!(sum.capacity() < 512);

    // Identical supports that cancel need no output allocation, even when large.
    let positive =
        Combination::from_sorted_terms((0..4096).map(|value| (value, 1.into())).collect(), Z);
    let cancelling = [positive.clone(), -positive];
    let borrowed = Combination::sum_ref(&cancelling, Z);
    assert!(borrowed.is_zero());
    assert_eq!(borrowed.capacity(), 0);
    let owned = Combination::sum(cancelling, Z);
    assert!(owned.is_zero());
    assert_eq!(owned.capacity(), 0);

    let mut disjoint = vec![lc(&[(-3, 2)]), Combination::new(Z), lc(&[(1, 3), (2, 4)])];
    let expected = lc(&[(-3, 2), (1, 3), (2, 4)]);
    assert_eq!(Combination::sum_ref(&disjoint, Z), expected);
    assert_eq!(Combination::sum(disjoint.clone(), Z), expected);
    disjoint.reverse();
    assert_eq!(Combination::sum_ref(&disjoint, Z), expected);
    assert_eq!(Combination::sum(disjoint, Z), expected);
}

#[test]
fn substitutions_are_simultaneous_and_combine_collisions() {
    let original = lc(&[(1, 2), (2, 3), (3, 4)]);
    let first = lc(&[(1, 5), (3, -2)]);
    let second = lc(&[(1, -1), (4, 6)]);
    assert_terms(&original.substitute(&1, &first), &[(1, 10), (2, 3)]);
    assert_eq!(original.substitute(&99, &first), original);
    assert_terms(
        &original.substitute(&2, &Combination::new(Z)),
        &[(1, 2), (3, 4)],
    );

    let mut visited = Vec::new();
    let result = original.substitute_all(|value| {
        visited.push(*value);
        match value {
            1 => Some(&first),
            2 => Some(&second),
            _ => None,
        }
    });
    assert_eq!(visited, vec![1, 2, 3]);
    assert_terms(&result, &[(1, 7), (4, 18)]);
    assert_terms(&original, &[(1, 2), (2, 3), (3, 4)]);
}

#[test]
fn mapping_values_resorts_and_combines_cancelling_terms() {
    let result = lc(&[(1, 2), (2, 5), (3, -2), (4, -3)]).map_values(|value| value % 2);
    assert_terms(&result, &[(0, 2)]);
    let result = lc(&[(1, 2), (2, 5), (3, -2)]).map_values(|value| -value);
    assert_terms(&result, &[(-3, -2), (-2, 5), (-1, 2)]);
}

#[test]
fn different_runtime_fields_are_rejected_even_for_empty_operands() {
    let small = Zp::new(5);
    let large = Zp::new(7);
    let left = LinearCombination::from_terms(vec![(1, small.to_element(2))], small.clone());
    let right = LinearCombination::from_terms(vec![(1, large.to_element(3))], large.clone());
    let empty = LinearCombination::<Zp, i32>::new(large);
    assert!(std::panic::catch_unwind(|| &left + &right).is_err());
    assert!(std::panic::catch_unwind(|| &left - &empty).is_err());
    assert!(std::panic::catch_unwind(|| left.add_scaled(&right, &small.zero())).is_err());
    assert!(
        std::panic::catch_unwind(|| LinearCombination::sum(
            vec![left.clone(), empty.clone()],
            small.clone()
        ))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(|| LinearCombination::sum_ref([&left, &empty], small)).is_err()
    );
    assert!(std::panic::catch_unwind(|| left.substitute(&1, &right)).is_err());
    assert!(std::panic::catch_unwind(|| left.substitute_all(|_| Some(&right))).is_err());
}

#[test]
fn finite_field_arithmetic_reduces_and_drops_zero_coefficients() {
    let field = Zp::new(7);
    let a = LinearCombination::from_terms(
        vec![(1, field.to_element(6)), (2, field.to_element(2))],
        field.clone(),
    );
    let b = LinearCombination::from_terms(
        vec![(1, field.to_element(1)), (2, field.to_element(6))],
        field.clone(),
    );
    let sum = &a + &b;
    assert_eq!(sum.terms(), &[(2, field.to_element(1))]);
    assert_eq!(
        LinearCombination::sum(vec![a.clone(), b.clone()], field.clone()),
        sum
    );
    assert_eq!(
        a.substitute(&1, &b).terms(),
        &[(1, field.to_element(6)), (2, field.to_element(3))]
    );
}

#[test]
fn add_term_preserves_canonical_order_and_display_uses_ring_formatting() {
    let mut value = LinearCombination::new(Z);
    assert_eq!(value.to_string(), "0");
    value.add_term("z", (-2).into());
    value.add_term("a", 4.into());
    value.add_term("m", 1.into());
    value.add_term("m", 2.into());
    value.add_term("a", (-4).into());
    value.add_term("b", 0.into());
    assert_eq!(value.terms(), &[("m", 3.into()), ("z", (-2).into())]);
    assert_eq!(value.to_string(), "(3)*(m) + (-2)*(z)");
    let field = Zp::new(7);
    let finite = LinearCombination::from_terms(vec![("x", field.to_element(3))], field);
    assert_eq!(finite.to_string(), "(3)*(x)");
    let compound = LinearCombination::from_terms(vec![("x + y", Integer::from(2))], Z);
    assert_eq!(compound.to_string(), "(2)*(x + y)");
}

#[test]
fn division_by_zero_panics_for_nonzero_and_empty_combinations() {
    let value = LinearCombination::from_terms(vec![(1, Q.one())], Q);
    let empty = LinearCombination::<_, i32>::new(Q);
    for input in [value, empty] {
        assert!(std::panic::catch_unwind(|| &input / &Q.zero()).is_err());
        assert!(std::panic::catch_unwind(|| input.clone() / &Q.zero()).is_err());
        assert!(
            std::panic::catch_unwind(|| {
                let mut assigned = input.clone();
                assigned /= &Q.zero();
            })
            .is_err()
        );
    }
}

#[test]
fn rational_polynomial_coefficients_preserve_numerator_denominator_context() {
    let variables = Arc::new(vec![
        symbol!("linear_combination_test_x").into(),
        symbol!("linear_combination_test_y").into(),
    ]);
    let convert =
        |atom: Atom| atom.to_rational_polynomial::<_, _, u16>(&Q, &Z, Some(variables.clone()));
    let first = convert(parse!(
        "(linear_combination_test_x+1)/(linear_combination_test_y+1)"
    ));
    let second = convert(parse!(
        "(linear_combination_test_y-linear_combination_test_x)/(linear_combination_test_y+1)"
    ));
    let field = RationalPolynomialField::<_, u16>::new(Z);
    let left =
        LinearCombination::from_terms(vec![(1, first.clone()), (2, first.clone())], field.clone());
    let right = LinearCombination::from_terms(
        vec![(1, second.clone()), (2, field.neg(&first))],
        field.clone(),
    );
    let sum = &left + &right;
    assert_eq!(sum.len(), 1);
    let coefficient = sum.coefficient(&1).unwrap();
    assert!(field.is_one(coefficient));
    assert_eq!(coefficient.numerator.variables(), &variables);
    assert_eq!(coefficient.denominator.variables(), &variables);
    assert_eq!(
        LinearCombination::sum(vec![left.clone(), right], field.clone()),
        sum
    );

    let replacement = LinearCombination::from_terms(vec![(2, second)], field.clone());
    let substituted = left.substitute(&1, &replacement);
    let expected = convert(parse!(
        "(linear_combination_test_x+1)*(2*linear_combination_test_y-linear_combination_test_x+1)/(linear_combination_test_y+1)^2"
    ));
    assert_eq!(substituted.terms(), &[(2, expected)]);
    let coefficient = substituted.coefficient(&2).unwrap();
    assert_eq!(coefficient.numerator.variables(), &variables);
    assert_eq!(coefficient.denominator.variables(), &variables);
    assert_eq!((&left / &first).coefficient(&1), Some(&field.one()));
}

#[test]
fn symbolic_coefficients_use_one_bulk_sum_for_each_value() {
    let normalizations = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&normalizations);
    let field = AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: false,
        custom_normalization: Some(Box::new(move |_, _| {
            counter.fetch_add(1, AtomicOrdering::Relaxed);
            false
        })),
    };
    let inputs: Vec<_> = [
        parse!("linear_combination_atom_a"),
        parse!("linear_combination_atom_b"),
        parse!("-linear_combination_atom_a"),
        parse!("linear_combination_atom_c"),
    ]
    .into_iter()
    .map(|coefficient| LinearCombination::from_terms(vec![(1, coefficient)], field.clone()))
    .collect();
    let borrowed = LinearCombination::sum_ref(inputs.iter(), field.clone());
    assert_eq!(
        borrowed.terms(),
        &[(
            1,
            parse!("linear_combination_atom_b+linear_combination_atom_c")
        )]
    );
    assert_eq!(normalizations.load(AtomicOrdering::Relaxed), 1);
    let owned = LinearCombination::sum(inputs, field);
    assert_eq!(owned, borrowed);
    assert_eq!(normalizations.load(AtomicOrdering::Relaxed), 2);
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
struct NonClone(i32);

#[test]
fn owned_operations_accept_values_that_cannot_be_cloned() {
    let make = || {
        LinearCombination::from_terms(
            vec![(NonClone(2), Integer::from(1)), (NonClone(1), 2.into())],
            Z,
        )
    };
    let sum = make() + make();
    assert_eq!(
        sum.terms(),
        &[(NonClone(1), 4.into()), (NonClone(2), 2.into())]
    );
    assert!((make() - make()).is_zero());
    let mut sum = make();
    sum += make();
    sum -= make();
    assert_eq!(sum, make());
    assert_eq!(
        (-make()).terms(),
        &[(NonClone(1), (-2).into()), (NonClone(2), (-1).into())]
    );
    assert_eq!(
        LinearCombination::sum([make(), make()], Z).terms(),
        &[(NonClone(1), 4.into()), (NonClone(2), 2.into())]
    );
    assert_eq!(
        make().map_values(|value| value.0).terms(),
        &[(1, 2.into()), (2, 1.into())]
    );
}

#[derive(Debug)]
struct CountedKey {
    value: i32,
    clones: Arc<AtomicUsize>,
}

impl Clone for CountedKey {
    fn clone(&self) -> Self {
        self.clones.fetch_add(1, AtomicOrdering::Relaxed);
        Self {
            value: self.value,
            clones: Arc::clone(&self.clones),
        }
    }
}

impl PartialEq for CountedKey {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}
impl Eq for CountedKey {}
impl PartialOrd for CountedKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for CountedKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

#[test]
fn owned_merges_move_values_without_cloning() {
    let clones = Arc::new(AtomicUsize::new(0));
    let make = |offset| {
        LinearCombination::from_terms(
            (0..20)
                .map(|i| {
                    (
                        CountedKey {
                            value: i + offset,
                            clones: Arc::clone(&clones),
                        },
                        Integer::from(1),
                    )
                })
                .collect(),
            Z,
        )
    };
    let sum = make(0) + make(10);
    assert_eq!(sum.len(), 30);
    let sum = LinearCombination::sum((0..12).map(make), Z);
    assert_eq!(sum.len(), 31);
    assert_eq!(clones.load(AtomicOrdering::Relaxed), 0);
    let mut assigned = make(0);
    assigned.add_scaled_assign(&make(10), &Integer::from(2));
    assert_eq!(assigned.len(), 30);
    // Only new rhs values need copying; the owned lhs survives the fused merge.
    assert_eq!(clones.load(AtomicOrdering::Relaxed), 10);
}

fn next_random(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 16
}

fn oracle(terms: impl IntoIterator<Item = (i32, i64)>) -> BTreeMap<i32, i64> {
    let mut result = BTreeMap::new();
    for (key, coefficient) in terms {
        *result.entry(key).or_default() += coefficient;
    }
    result.retain(|_, coefficient| *coefficient != 0);
    result
}

#[test]
fn deterministic_random_arithmetic_matches_independent_map_oracle() {
    let mut state = 0x6c69_6e65_6172_7375;
    for _ in 0..32 {
        let raw: Vec<Vec<(i32, i64)>> = (0..12)
            .map(|_| {
                (0..48)
                    .map(|_| {
                        (
                            (next_random(&mut state) % 37) as i32 - 18,
                            (next_random(&mut state) % 13) as i64 - 6,
                        )
                    })
                    .collect()
            })
            .collect();
        let inputs: Vec<_> = raw.iter().map(|terms| lc(terms)).collect();
        let expected: Vec<_> = oracle(raw.iter().flatten().copied()).into_iter().collect();
        assert_terms(&Combination::sum_ref(inputs.iter(), Z), &expected);
        assert_terms(&Combination::sum(inputs.clone(), Z), &expected);

        let expected: Vec<_> = oracle(
            raw[0]
                .iter()
                .copied()
                .chain(raw[1].iter().map(|&(v, c)| (v, -3 * c))),
        )
        .into_iter()
        .collect();
        assert_terms(
            &inputs[0].add_scaled(&inputs[1], &Integer::from(-3)),
            &expected,
        );
        let expected: Vec<_> = oracle(
            raw[0]
                .iter()
                .copied()
                .chain(raw[1].iter().map(|&(v, c)| (v, -c))),
        )
        .into_iter()
        .collect();
        assert_terms(&(&inputs[0] - &inputs[1]), &expected);
        let expected: Vec<_> = oracle(raw[0].iter().map(|&(v, c)| (v.rem_euclid(5), c)))
            .into_iter()
            .collect();
        assert_terms(
            &inputs[0].clone().map_values(|v| v.rem_euclid(5)),
            &expected,
        );
    }
}

#[test]
fn non_power_of_two_sums_handle_identical_shifted_and_cancelling_supports() {
    for count in [3, 5, 9, 17] {
        for shifted in [false, true] {
            let raw: Vec<Vec<_>> = (0..count)
                .map(|input| {
                    // Coefficients add to zero wherever all supports coincide.
                    let coefficient = if input + 1 == count {
                        -(count * (count - 1) / 2) as i64
                    } else {
                        (input + 1) as i64
                    };
                    (0..12)
                        .map(|key| (key + if shifted { input % 3 } else { 0 }, coefficient))
                        .collect()
                })
                .collect();
            let expected: Vec<_> = oracle(raw.iter().flatten().copied()).into_iter().collect();
            let inputs: Vec<_> = raw.iter().map(|terms| lc(terms)).collect();
            assert_terms(&Combination::sum_ref(&inputs, Z), &expected);
            assert_terms(&Combination::sum(inputs, Z), &expected);
            if !shifted {
                assert!(expected.is_empty());
            }
        }
    }
}

#[test]
fn sums_follow_dense_sparse_and_exhausted_stream_phases() {
    let raw: Vec<Vec<_>> = (0..11)
        .map(|input| {
            if input == 0 || input == 5 {
                return Vec::new();
            }
            let mut terms: Vec<_> = (0..8).map(|key| (key, 1)).collect();
            // Some streams end after the shared prefix; the others diverge
            // into interleaved sparse tails of different lengths.
            if input % 3 != 1 {
                terms.extend((0..input % 4 + 1).map(|key| (100 + 11 * key + input, 2)));
            }
            if input % 3 == 0 {
                terms.push((1000, -1));
            }
            terms
        })
        .collect();
    let expected: Vec<_> = oracle(raw.iter().flatten().copied()).into_iter().collect();
    let inputs: Vec<_> = raw.iter().map(|terms| lc(terms)).collect();
    assert_terms(&Combination::sum_ref(&inputs, Z), &expected);
    assert_terms(&Combination::sum(inputs.clone(), Z), &expected);
    let source = lc(&(0..11).map(|key| (key, 1)).collect::<Vec<_>>());
    assert_terms(
        &source.substitute_all(|value| Some(&inputs[*value as usize])),
        &expected,
    );
}

/// Intentionally not Clone: tournament merging must move owned values.
#[derive(Debug)]
struct MeasuredKey {
    value: i32,
    comparisons: Arc<AtomicUsize>,
}

impl PartialEq for MeasuredKey {
    fn eq(&self, other: &Self) -> bool {
        self.comparisons.fetch_add(1, AtomicOrdering::Relaxed);
        self.value == other.value
    }
}
impl Eq for MeasuredKey {}
impl PartialOrd for MeasuredKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MeasuredKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.comparisons.fetch_add(1, AtomicOrdering::Relaxed);
        self.value.cmp(&other.value)
    }
}

#[test]
fn fully_overlapping_sums_use_linear_key_comparisons_and_move_owned_values() {
    const TERMS: i32 = 32;
    for count in [17, 65, 257] {
        let comparisons = Arc::new(AtomicUsize::new(0));
        let inputs: Vec<_> = (0..count)
            .map(|_| {
                LinearCombination::from_sorted_terms(
                    (0..TERMS)
                        .map(|value| {
                            (
                                MeasuredKey {
                                    value,
                                    comparisons: Arc::clone(&comparisons),
                                },
                                Integer::from(1),
                            )
                        })
                        .collect(),
                    Z,
                )
            })
            .collect();
        comparisons.store(0, AtomicOrdering::Relaxed);
        let sum = LinearCombination::sum(inputs, Z);
        let measured = comparisons.load(AtomicOrdering::Relaxed);
        // A per-term heap update grows by log(count) and exceeds this bound;
        // grouping equal tournament winners stays linear in consumed terms.
        assert!(
            measured <= 4 * count * TERMS as usize + 2 * count,
            "{count} streams needed {measured} value comparisons"
        );
        assert_eq!(sum.len(), TERMS as usize);
        assert!(sum.capacity() <= 2 * TERMS as usize);
        for (expected_key, (key, coefficient)) in sum.terms().iter().enumerate() {
            assert_eq!(key.value, expected_key as i32);
            assert_eq!(coefficient, &Integer::from(count as i64));
        }
    }
}

#[derive(Clone, Debug)]
struct RecordingMod6 {
    batches: Arc<Mutex<Vec<Vec<u64>>>>,
}

impl PartialEq for RecordingMod6 {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.batches, &other.batches)
    }
}
impl Eq for RecordingMod6 {}
impl std::hash::Hash for RecordingMod6 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::ptr::hash(Arc::as_ptr(&self.batches), state);
    }
}
impl fmt::Display for RecordingMod6 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("Z/6Z (recorded sums)")
    }
}
impl Set for RecordingMod6 {
    type Element = u64;
    fn size(&self) -> Option<Integer> {
        Mod6.size()
    }
}
impl<T> RingOps<T> for RecordingMod6
where
    Mod6: RingOps<T>,
{
    fn add(&self, a: T, b: T) -> u64 {
        Mod6.add(a, b)
    }
    fn sub(&self, a: T, b: T) -> u64 {
        Mod6.sub(a, b)
    }
    fn mul(&self, a: T, b: T) -> u64 {
        Mod6.mul(a, b)
    }
    fn neg(&self, a: T) -> u64 {
        Mod6.neg(a)
    }
    fn add_assign(&self, a: &mut u64, b: T) {
        Mod6.add_assign(a, b);
    }
    fn sub_assign(&self, a: &mut u64, b: T) {
        Mod6.sub_assign(a, b);
    }
    fn mul_assign(&self, a: &mut u64, b: T) {
        Mod6.mul_assign(a, b);
    }
    fn add_mul_assign(&self, a: &mut u64, b: T, c: T) {
        Mod6.add_mul_assign(a, b, c);
    }
    fn sub_mul_assign(&self, a: &mut u64, b: T, c: T) {
        Mod6.sub_mul_assign(a, b, c);
    }
}
impl Ring for RecordingMod6 {
    fn zero(&self) -> u64 {
        Mod6.zero()
    }
    fn one(&self) -> u64 {
        Mod6.one()
    }
    fn nth(&self, n: Integer) -> u64 {
        Mod6.nth(n)
    }
    fn pow(&self, b: &u64, e: u64) -> u64 {
        Mod6.pow(b, e)
    }
    fn is_zero(&self, a: &u64) -> bool {
        Mod6.is_zero(a)
    }
    fn is_one(&self, a: &u64) -> bool {
        Mod6.is_one(a)
    }
    fn one_is_gcd_unit() -> bool {
        Mod6::one_is_gcd_unit()
    }
    fn characteristic(&self) -> Integer {
        Mod6.characteristic()
    }
    fn try_inv(&self, a: &u64) -> Option<u64> {
        Mod6.try_inv(a)
    }
    fn try_div(&self, a: &u64, b: &u64) -> Option<u64> {
        Mod6.try_div(a, b)
    }
    fn sum<I: IntoIterator<Item = u64>>(&self, elements: I) -> u64 {
        let elements: Vec<_> = elements.into_iter().collect();
        self.batches.lock().unwrap().push(elements.clone());
        Mod6.sum(elements)
    }
    fn format<W: fmt::Write>(
        &self,
        element: &u64,
        options: &PrintOptions,
        state: PrintState,
        output: &mut W,
    ) -> Result<bool, fmt::Error> {
        Mod6.format(element, options, state, output)
    }
}

#[test]
fn n_way_coefficient_batches_preserve_input_order_after_streams_exhaust() {
    let ring = RecordingMod6 {
        batches: Arc::new(Mutex::new(Vec::new())),
    };
    let inputs: Vec<_> = [
        vec![(0, 1), (3, 4), (5, 2)],
        vec![],
        vec![(0, 2), (1, 2), (3, 5)],
        vec![(0, 3), (2, 3), (3, 1), (9, 4)],
        vec![(0, 4), (3, 2), (5, 3)],
        vec![],
        vec![(0, 5), (1, 5), (3, 3), (9, 5)],
    ]
    .into_iter()
    .map(|terms| LinearCombination::from_sorted_terms(terms, ring.clone()))
    .collect();
    let expected_batches = vec![
        vec![1, 2, 3, 4, 5],
        vec![2, 5],
        vec![4, 5, 1, 2, 3],
        vec![2, 3],
        vec![4, 5],
    ];
    let expected = &[(0, 3), (1, 1), (2, 3), (3, 3), (5, 5), (9, 3)];
    let borrowed = LinearCombination::sum_ref(&inputs, ring.clone());
    assert_eq!(borrowed.terms(), expected);
    assert_eq!(*ring.batches.lock().unwrap(), expected_batches);
    ring.batches.lock().unwrap().clear();
    let owned = LinearCombination::sum(inputs.clone(), ring.clone());
    assert_eq!(owned.terms(), expected);
    assert_eq!(*ring.batches.lock().unwrap(), expected_batches);
    ring.batches.lock().unwrap().clear();
    let source = LinearCombination::from_sorted_terms(
        (0..inputs.len()).map(|i| (i, 1)).collect(),
        ring.clone(),
    );
    assert_eq!(
        source.substitute_all(|value| Some(&inputs[*value])).terms(),
        expected
    );
    assert_eq!(*ring.batches.lock().unwrap(), expected_batches);
}

/// The ring Z/6Z supplies genuine nonzero zero divisors without pretending that
/// a composite modulus is a finite field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Mod6;

impl fmt::Display for Mod6 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Z/6Z")
    }
}

impl Set for Mod6 {
    type Element = u64;
    fn size(&self) -> Option<Integer> {
        Some(6.into())
    }
}

impl RingOps<u64> for Mod6 {
    fn add(&self, a: u64, b: u64) -> u64 {
        (a + b) % 6
    }
    fn sub(&self, a: u64, b: u64) -> u64 {
        (a + 6 - b) % 6
    }
    fn mul(&self, a: u64, b: u64) -> u64 {
        (a * b) % 6
    }
    fn neg(&self, a: u64) -> u64 {
        (6 - a) % 6
    }
    fn add_assign(&self, a: &mut u64, b: u64) {
        *a = self.add(*a, b);
    }
    fn sub_assign(&self, a: &mut u64, b: u64) {
        *a = self.sub(*a, b);
    }
    fn mul_assign(&self, a: &mut u64, b: u64) {
        *a = self.mul(*a, b);
    }
    fn add_mul_assign(&self, a: &mut u64, b: u64, c: u64) {
        *a = self.add(*a, self.mul(b, c));
    }
    fn sub_mul_assign(&self, a: &mut u64, b: u64, c: u64) {
        *a = self.sub(*a, self.mul(b, c));
    }
}

impl RingOps<&u64> for Mod6 {
    fn add(&self, a: &u64, b: &u64) -> u64 {
        self.add(*a, *b)
    }
    fn sub(&self, a: &u64, b: &u64) -> u64 {
        self.sub(*a, *b)
    }
    fn mul(&self, a: &u64, b: &u64) -> u64 {
        self.mul(*a, *b)
    }
    fn neg(&self, a: &u64) -> u64 {
        self.neg(*a)
    }
    fn add_assign(&self, a: &mut u64, b: &u64) {
        self.add_assign(a, *b);
    }
    fn sub_assign(&self, a: &mut u64, b: &u64) {
        self.sub_assign(a, *b);
    }
    fn mul_assign(&self, a: &mut u64, b: &u64) {
        self.mul_assign(a, *b);
    }
    fn add_mul_assign(&self, a: &mut u64, b: &u64, c: &u64) {
        self.add_mul_assign(a, *b, *c);
    }
    fn sub_mul_assign(&self, a: &mut u64, b: &u64, c: &u64) {
        self.sub_mul_assign(a, *b, *c);
    }
}

impl Ring for Mod6 {
    fn zero(&self) -> u64 {
        0
    }
    fn one(&self) -> u64 {
        1
    }
    fn nth(&self, n: Integer) -> u64 {
        let residue = Z.rem(&n, &Integer::from(6));
        residue.to_i64().unwrap().rem_euclid(6) as u64
    }
    fn pow(&self, b: &u64, mut e: u64) -> u64 {
        let (mut power, mut result) = (*b, 1);
        while e != 0 {
            if e & 1 != 0 {
                result = self.mul(result, power);
            }
            power = self.mul(power, power);
            e >>= 1;
        }
        result
    }
    fn is_zero(&self, a: &u64) -> bool {
        *a == 0
    }
    fn is_one(&self, a: &u64) -> bool {
        *a == 1
    }
    fn one_is_gcd_unit() -> bool {
        true
    }
    fn characteristic(&self) -> Integer {
        6.into()
    }
    fn try_inv(&self, a: &u64) -> Option<u64> {
        match a {
            1 | 5 => Some(*a),
            _ => None,
        }
    }
    fn try_div(&self, a: &u64, b: &u64) -> Option<u64> {
        self.try_inv(b).map(|inverse| self.mul(*a, inverse))
    }
    fn format<W: fmt::Write>(
        &self,
        element: &u64,
        _: &PrintOptions,
        _: PrintState,
        output: &mut W,
    ) -> Result<bool, fmt::Error> {
        write!(output, "{element}")?;
        Ok(false)
    }
}

#[test]
fn zero_divisor_products_are_removed_from_all_scaled_paths() {
    let value = LinearCombination::from_terms(vec![(1, 2), (2, 3), (3, 1)], Mod6);
    assert_eq!(value.scale(&3).terms(), &[(2, 3), (3, 3)]);
    assert_eq!((&value * &3).terms(), &[(2, 3), (3, 3)]);
    let mut assigned = value.clone();
    assigned *= &3;
    assert_eq!(assigned.terms(), &[(2, 3), (3, 3)]);

    let left = LinearCombination::from_terms(vec![(1, 1)], Mod6);
    let right = LinearCombination::from_terms(vec![(1, 2), (2, 2), (3, 3)], Mod6);
    assert_eq!(left.add_scaled(&right, &3).terms(), &[(1, 1), (3, 3)]);
    let original = LinearCombination::from_terms(vec![(1, 2), (2, 3)], Mod6);
    let replacement = LinearCombination::from_terms(vec![(2, 3), (3, 3)], Mod6);
    assert_eq!(original.substitute(&1, &replacement).terms(), &[(2, 3)]);
    assert_eq!(
        original
            .substitute_all(|value| (*value == 1).then_some(&replacement))
            .terms(),
        &[(2, 3)]
    );
}
