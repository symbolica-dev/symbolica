use super::*;
use crate::domains::finite_field::{Z2, Zp};

fn polynomial<F: Ring>(
    ring: &F,
    variables: &[usize],
    terms: &[(i64, &[u32])],
) -> MultivariatePolynomial<F, u32> {
    let variables = Arc::new(
        variables
            .iter()
            .copied()
            .map(PolyVariable::Temporary)
            .collect(),
    );
    let mut result = MultivariatePolynomial::new(ring, None, variables);
    for &(coefficient, exponents) in terms {
        result.append_monomial(ring.nth(coefficient.into()), exponents);
    }
    result.check_consistency();
    result
}

fn assert_polynomial<F: Ring, E: Exponent>(
    actual: &MultivariatePolynomial<F, E>,
    expected: &MultivariatePolynomial<F, E>,
) {
    actual.check_consistency();
    assert_eq!(actual, expected);
    // Polynomial equality alone ignores variable maps for constants and zero.
    assert_eq!(actual.variables(), expected.variables());
    assert_eq!(actual.ring(), expected.ring());
    assert_eq!(actual.exponents, expected.exponents);
    assert_eq!(actual.coefficients, expected.coefficients);
}

#[test]
fn replace_multiple_uses_original_indices_and_preserves_unused_variables() {
    let variables = [41, 7, 93, 20, 5];
    let p = polynomial(
        &Z,
        &variables,
        &[
            (3, &[2, 1, 0, 0, 0]),
            (2, &[1, 0, 1, 1, 0]),
            (-1, &[0, 1, 1, 0, 0]),
            (7, &[0, 0, 0, 1, 0]),
            (5, &[0, 0, 0, 0, 0]),
        ],
    );
    let two = Z.nth(2.into());
    let minus_one = Z.nth((-1).into());
    let points = [(3, &minus_one), (0, &two)];
    let kept = polynomial(
        &Z,
        &variables,
        &[
            (12, &[0, 1, 0, 0, 0]),
            (-1, &[0, 1, 1, 0, 0]),
            (-4, &[0, 0, 1, 0, 0]),
            (-2, &[0, 0, 0, 0, 0]),
        ],
    );
    let erased = polynomial(
        &Z,
        &[7, 93, 5],
        &[
            (12, &[1, 0, 0]),
            (-1, &[1, 1, 0]),
            (-4, &[0, 1, 0]),
            (-2, &[0, 0, 0]),
        ],
    );
    assert_polynomial(&p.replace(3, &minus_one).replace(0, &two), &kept);
    for (erase, expected) in [(false, kept), (true, erased)] {
        assert_polynomial(&p.replace_multiple(&points, erase), &expected);
        assert_polynomial(
            &p.replace_multiple(&[points[1], points[0]], erase),
            &expected,
        );
    }
}

#[test]
fn replace_multiple_handles_cancellation_zero_and_constants() {
    let p = polynomial(&Z, &[8, 3, 9], &[(1, &[1, 1, 0]), (-1, &[0, 1, 0])]);
    let one = Z.one();
    let zero = Z.zero();
    for erase in [false, true] {
        let variables: &[usize] = if erase { &[3, 9] } else { &[8, 3, 9] };
        let expected_zero = polynomial(&Z, variables, &[]);
        assert_polynomial(&p.replace_multiple(&[(0, &one)], erase), &expected_zero);
        let powers: &[u32] = if erase { &[1, 0] } else { &[0, 1, 0] };
        assert_polynomial(
            &p.replace_multiple(&[(0, &zero)], erase),
            &polynomial(&Z, variables, &[(-1, powers)]),
        );

        for value in [0, 5] {
            let input = p.constant(value.into());
            let expected = if erase {
                polynomial(&Z, &[3], &[(value, &[0])])
            } else {
                input.clone()
            };
            // Erase explicitly requested slots even when neither occurs in a term.
            assert_polynomial(
                &input.replace_multiple(&[(2, &zero), (0, &one)], erase),
                &expected,
            );
        }
    }
}

#[test]
fn empty_batch_operations_preserve_zero_constant_and_nonconstant_contexts() {
    let p = polynomial(&Q, &[8, 3], &[(2, &[2, 1]), (-3, &[0, 0])]);
    for input in [
        p.zero(),
        p.one(),
        p,
        polynomial(&Q, &[], &[]),
        polynomial(&Q, &[], &[(7, &[])]),
    ] {
        for erase in [false, true] {
            assert_polynomial(&input.replace_multiple(&[], erase), &input);
        }
        assert_polynomial(&input.shift_vars(&[]), &input);
    }
}

fn check_replace_all<F: Ring>(ring: &F) {
    let p = polynomial(
        ring,
        &[4, 1, 7],
        &[
            (2, &[1, 1, 0]),
            (-3, &[0, 2, 1]),
            (5, &[1, 0, 1]),
            (7, &[0, 0, 0]),
        ],
    );
    let cancelling = polynomial(ring, &[4, 1, 7], &[(1, &[1, 0, 0]), (-1, &[0, 1, 0])]);
    for input in [p.zero(), p.one(), p, cancelling] {
        for values in [[0, 0, 0], [2, -1, 3], [1, 1, 1]] {
            let values = values.map(|value| ring.nth(value.into()));
            let points = [(2, &values[2]), (0, &values[0]), (1, &values[1])];
            let value = input.replace_all(&values);
            assert_polynomial(
                &input.replace_multiple(&points, false),
                &input.constant(value.clone()),
            );
            assert_polynomial(
                &input.replace_multiple(&points, true),
                &polynomial(ring, &[], &[]).constant(value),
            );
        }
    }
}

#[test]
fn replace_multiple_all_variables_matches_replace_all_in_each_ring() {
    check_replace_all(&Z);
    check_replace_all(&Q);
    check_replace_all(&Zp::new(5));
}

#[test]
fn replace_multiple_supports_sparse_exponents_above_i32_max() {
    let degree = i32::MAX as u32 + 2;
    let p = polynomial(
        &Z,
        &[11, 2],
        &[(1, &[degree, 3]), (3, &[0, 3]), (-4, &[degree - 1, 3])],
    );
    for (value, coefficient) in [(1, 0), (-1, -2)] {
        let value = Z.nth(value.into());
        assert_polynomial(
            &p.replace_multiple(&[(0, &value)], true),
            &polynomial(&Z, &[2], &[(coefficient, &[3])]),
        );
    }
}

fn check_shifts<F: Ring>(ring: &F) {
    let p = polynomial(
        ring,
        &[17, 4, 2, 9],
        &[
            (-3, &[0, 0, 0, 0]),
            (2, &[4, 1, 2, 0]),
            (-1, &[2, 3, 0, 0]),
            (5, &[0, 0, 3, 0]),
            (1, &[0, 2, 1, 0]),
        ],
    );
    let zero = ring.zero();
    let one = ring.one();
    let minus_two = ring.nth((-2).into());
    for input in [p.zero(), p.one(), p] {
        let cases: &[&[(usize, &F::Element)]] = &[
            &[],
            &[(0, &zero)],
            &[(3, &one)],
            &[(2, &zero), (0, &one), (1, &minus_two)],
            &[(2, &one), (0, &minus_two), (3, &one)],
        ];
        for shifts in cases {
            let expected = shifts.iter().fold(input.clone(), |p, &(variable, value)| {
                p.shift_var(variable, value)
            });
            assert_polynomial(&input.shift_vars(shifts), &expected);
            let reversed: Vec<_> = shifts.iter().rev().copied().collect();
            assert_polynomial(&input.shift_vars(&reversed), &expected);
        }
    }
}

#[test]
fn shift_vars_matches_sequential_shifts_over_integers_rationals_and_finite_fields() {
    check_shifts(&Z);
    check_shifts(&Q);
    check_shifts(&Z2);
    check_shifts(&Zp::new(5));
}

#[test]
fn shift_vars_merges_cancelling_terms_and_preserves_variable_map() {
    // (x - 1)^2 y, shifted by x -> x + 1 and y -> y - 2.
    let p = polynomial(
        &Z,
        &[9, 2, 7],
        &[(1, &[2, 1, 0]), (-2, &[1, 1, 0]), (1, &[0, 1, 0])],
    );
    assert_polynomial(
        &p.shift_vars(&[(1, &(-2).into()), (0, &1.into())]),
        &polynomial(&Z, &[9, 2, 7], &[(1, &[2, 1, 0]), (-2, &[2, 0, 0])]),
    );
}

#[test]
fn shift_vars_handles_fractional_coefficients_and_shifts() {
    let half = Q.div(&Q.one(), &Q.nth(2.into()));
    let minus_third = Q.div(&Q.nth((-1).into()), &Q.nth(3.into()));
    let p = polynomial(
        &Q,
        &[3, 8, 1],
        &[(3, &[3, 2, 0]), (-1, &[1, 0, 1]), (5, &[0, 0, 0])],
    )
    .mul_coeff(half.clone());
    let expected = p.shift_var(0, &half).shift_var(1, &minus_third);
    assert_polynomial(&p.shift_vars(&[(1, &minus_third), (0, &half)]), &expected);
}

#[test]
fn shift_vars_expands_degrees_at_and_above_the_field_characteristic() {
    fn check<F: Ring>(field: &F, characteristic: u32) {
        let p = polynomial(
            field,
            &[7, 3],
            &[
                (1, &[characteristic + 1, 0]),
                (1, &[0, characteristic]),
                (1, &[1, 1]),
            ],
        );
        let one = field.one();
        assert_polynomial(&p.shift_vars(&[(0, &one)]), &p.shift_var(0, &one));
        // Frobenius gives (x + 1)^(p + 1) = x^(p + 1) + x^p + x + 1.
        let expected = polynomial(
            field,
            &[7, 3],
            &[
                (1, &[characteristic + 1, 0]),
                (1, &[characteristic, 0]),
                (1, &[0, characteristic]),
                (1, &[1, 1]),
                (2, &[1, 0]),
                (1, &[0, 1]),
                (3, &[0, 0]),
            ],
        );
        assert_polynomial(&p.shift_vars(&[(1, &one), (0, &one)]), &expected);
    }
    check(&Z2, 2);
    check(&Zp::new(3), 3);
    check(&Zp::new(5), 5);
}

fn assert_panics_with(action: impl FnOnce(), expected: &str) {
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(action)).unwrap_err();
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .expect("panic should contain a message");
    assert!(message.contains(expected), "unexpected panic: {message}");
}

#[test]
fn batch_operations_validate_indices_before_zero_and_constant_shortcuts() {
    let p = polynomial(&Z, &[8, 2], &[(1, &[1, 1])]);
    let zero = Z.zero();
    let one = Z.one();
    let cases: &[(&[(usize, &Integer)], &str)] = &[
        (&[(2, &one)], "Variable index out of bounds"),
        (&[(usize::MAX, &zero)], "Variable index out of bounds"),
        (&[(0, &zero), (2, &zero)], "Variable index out of bounds"),
        (
            &[(0, &zero), (0, &zero)],
            "Variable index appears more than once",
        ),
        (
            &[(1, &one), (1, &zero)],
            "Variable index appears more than once",
        ),
    ];
    for input in [p.zero(), p.one(), p] {
        for &(points, message) in cases {
            for erase in [false, true] {
                assert_panics_with(
                    || {
                        input.replace_multiple(points, erase);
                    },
                    message,
                );
            }
            assert_panics_with(
                || {
                    input.shift_vars(points);
                },
                message,
            );
        }
    }
    for input in [polynomial(&Z, &[], &[]), polynomial(&Z, &[], &[(1, &[])])] {
        for erase in [false, true] {
            assert_panics_with(
                || {
                    input.replace_multiple(&[(0, &zero)], erase);
                },
                "Variable index out of bounds",
            );
        }
        assert_panics_with(
            || {
                input.shift_vars(&[(0, &zero)]);
            },
            "Variable index out of bounds",
        );
    }
}
