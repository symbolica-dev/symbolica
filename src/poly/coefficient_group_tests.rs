//! Regression tests for the guarded literal-Atom Add collector.
use super::*;

fn atom(input: &str) -> Atom {
    Atom::parse(input, "coefficient_group_tests", Default::default()).unwrap()
}

fn variables(names: &[&str]) -> Arc<Vec<PolyVariable>> {
    Arc::new(
        names
            .iter()
            .map(|name| PolyVariable::try_from(atom(name)).unwrap())
            .collect(),
    )
}

fn literal_field() -> AtomField {
    AtomField {
        statistical_zero_test: false,
        ..AtomField::new()
    }
}

// Keep the previous Add collector as a reference. Test terms have no nested
// Add collectors that could change the oracle's duplicate-exponent reduction.
fn legacy_add(
    input: &Atom,
    vars: &Arc<Vec<PolyVariable>>,
    field: &AtomField,
) -> MultivariatePolynomial<AtomField, i16> {
    let AtomView::Add(add) = input.as_view() else {
        return input
            .as_view()
            .to_polynomial_in_vars_with_field(vars, field);
    };
    let mut coefficients = Vec::new();
    let mut exponents = Vec::new();
    for term in add {
        let child = term.to_polynomial_in_vars_with_field::<i16>(vars, field);
        coefficients.extend(child.coefficients);
        exponents.extend(child.exponents);
    }
    MultivariatePolynomial::from_coefficient_list(coefficients, exponents, vars.clone(), field)
}

fn check(
    input: &Atom,
    vars: &Arc<Vec<PolyVariable>>,
    field: &AtomField,
) -> MultivariatePolynomial<AtomField, i16> {
    let expected = legacy_add(input, vars, field);
    let actual = input
        .as_view()
        .to_polynomial_in_vars_with_field::<i16>(vars, field);
    let coefficient_to_atom = |_: &AtomField, coefficient: &Atom, output: &mut Atom| {
        output.set_from_view(&coefficient.as_view());
    };
    assert!(
        actual == expected
            || (actual.to_expression_with_coeff_map(coefficient_to_atom)
                - expected.to_expression_with_coeff_map(coefficient_to_atom))
            .expand()
            .together()
            .is_zero()
    );
    assert_eq!(actual.variables(), vars);
    actual.check_consistency();
    actual
}

#[test]
fn literal_atom_groups_collect_many_opaque_coefficients_and_negative_powers() {
    let vars = variables(&["k(1)", "x", "unused"]);
    let field = literal_field();
    let input = Atom::add_many((1..=128).flat_map(|n| {
        [
            atom(&format!("spectator({n})*k(1)^-2*x")),
            atom(&format!("spectator({n})*k(1)*x^3")),
        ]
    }));
    assert!(matches!(input.as_view(), AtomView::Add(_)));
    let actual = check(&input, &vars, &field);
    assert_eq!(actual.nterms(), 2);
    assert_eq!(actual.exponents, vec![-2, 1, 0, 1, 3, 0]);
    let coefficient = Atom::add_many((1..=128).map(|n| atom(&format!("spectator({n})"))));
    assert_eq!(actual.coefficients, vec![coefficient.clone(), coefficient]);
}

#[test]
fn literal_atom_groups_preserve_empty_maps_singletons_and_cancellation() {
    let field = literal_field();
    let empty = variables(&[]);
    for text in [
        "0",
        "spectator(1)",
        "spectator(1)+spectator(2)+spectator(3)",
    ] {
        let input = atom(text);
        let actual = check(&input, &empty, &field);
        assert_eq!(actual.nvars(), 0);
        assert_eq!(actual.nterms(), usize::from(!input.is_zero()));
        assert_eq!(actual.get_constant(), input);
    }

    let vars = variables(&["k(1)"]);
    let input = atom("(a+b)*k(1)-a*k(1)-b*k(1)+spectator(7)*k(1)^-3");
    let actual = check(&input, &vars, &field);
    assert_eq!(actual.nterms(), 1);
    assert_eq!(actual.exponents, vec![-3]);
    assert_eq!(actual.coefficients, vec![atom("spectator(7)")]);
}

#[test]
fn literal_atom_groups_keep_statistical_and_custom_normalization_fallbacks() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let vars = variables(&["k(1)"]);
    let statistical = AtomField::new();
    let input = atom("(a^2-1)*k(1)+(1-a)*(a+1)*k(1)+spectator(2)*k(1)^2");
    let actual = check(&input, &vars, &statistical);
    assert_eq!(actual.nterms(), 1);

    let calls = Arc::new(AtomicUsize::new(0));
    let counter = calls.clone();
    let custom = AtomField {
        statistical_zero_test: false,
        custom_normalization: Some(Box::new(move |_, _| {
            counter.fetch_add(1, Ordering::Relaxed);
            false
        })),
        ..AtomField::new()
    };
    let input = Atom::add_many((1..=16).map(|n| atom(&format!("spectator({n})*k(1)"))));
    let expected = legacy_add(&input, &vars, &custom);
    let expected_calls = calls.swap(0, Ordering::Relaxed);
    let actual = input
        .as_view()
        .to_polynomial_in_vars_with_field::<i16>(&vars, &custom);
    assert_eq!(actual, expected);
    assert!(expected_calls > 1);
    assert_eq!(calls.load(Ordering::Relaxed), expected_calls);
}

#[test]
fn literal_atom_groups_keep_float_reduction_order() {
    let field = literal_field();
    let vars = variables(&["k(1)"]);
    let input = atom(
        "1.234567890123456789*left(1)*k(1)+9.876543210987654321*right(1)*k(1)+3.141592653589793*middle(1)*k(1)",
    );
    check(&input, &vars, &field);
}

#[test]
fn literal_atom_groups_preserve_exact_gaussian_and_factored_coefficients() {
    let field = literal_field();
    let vars = variables(&["k(1)"]);
    for text in [
        "k(1)*(a+b)+k(1)*a+k(1)*b",
        "123456789012345678901234567890123456789*a*k(1)+(1+3*symbolica::𝑖)/7*b*k(1)+symbolica::𝑖*a*k(1)",
    ] {
        check(&atom(text), &vars, &field);
    }
}

#[test]
fn literal_atom_groups_keep_nonfinite_and_finite_field_fallbacks() {
    use crate::{
        coefficient::Coefficient,
        domains::finite_field::{FiniteFieldCore, Zp64},
    };
    let field = literal_field();
    let vars = variables(&["k(1)"]);
    let k = atom("k(1)");
    let a = atom("a");
    let prime = Zp64::new(17);
    for coefficient in [
        Coefficient::positive_infinity(),
        Coefficient::indeterminate(),
        Coefficient::from_finite_field(prime.clone(), prime.to_element(3)),
    ] {
        let factor = Atom::num(coefficient) * &a;
        let input = (&k + Atom::one()) * &factor + (&k - Atom::one()) * (-factor) + &k * &a;
        let expected = legacy_add(&input, &vars, &field);
        let actual = input
            .as_view()
            .to_polynomial_in_vars_with_field::<i16>(&vars, &field);
        assert_eq!(actual, expected);
    }
}

#[test]
fn nonzero_child_groups_preserve_generic_normalization_sequence() {
    use std::sync::Mutex;

    let vars = variables(&["k(1)", "unused"]);
    let calls = Arc::new(Mutex::new(Vec::new()));
    let log = calls.clone();
    let trigger = atom("u+v");
    let target = trigger.clone();
    let replacement = atom("7");
    let field = AtomField {
        custom_normalization: Some(Box::new(move |input, output| {
            log.lock().unwrap().push(input.to_owned());
            if input == target.as_view() {
                output.set_from_view(&replacement.as_view());
                true
            } else {
                false
            }
        })),
        ..AtomField::new()
    };
    let input = atom("(a^2-1)*k(1)+(1-a)*(a+1)*k(1)+u*k(1)^2+v*k(1)^2");
    let expected = legacy_add(&input, &vars, &field);
    let expected_calls = std::mem::take(&mut *calls.lock().unwrap());
    assert!(expected_calls.contains(&trigger));
    let actual = input
        .as_view()
        .to_polynomial_in_vars_with_field::<i16>(&vars, &field);
    assert_eq!(actual, expected);
    assert_eq!(*calls.lock().unwrap(), expected_calls);
    assert_eq!(actual.coefficients, vec![atom("7")]);
    assert_eq!(actual.exponents, vec![2, 0]);
    assert_eq!(actual.variables(), &vars);
    actual.check_consistency();
}

#[test]
fn nonzero_child_groups_keep_checked_constructor_and_empty_maps() {
    let field = AtomField::new();
    let zero = atom("(a+1)*(1-a)+a^2-1");
    assert!(!zero.is_zero());
    assert!(field.is_zero(&zero));
    let empty = variables(&[]);
    // Arbitrary coefficient lists must still reject literal and semantic zeros.
    let checked = MultivariatePolynomial::<_, i16>::from_coefficient_list(
        vec![Atom::Zero, zero, atom("7")],
        Vec::new(),
        empty.clone(),
        &field,
    );
    assert_eq!(checked.coefficients, vec![atom("7")]);

    for vars in [empty, variables(&["unused", "k(1)"])] {
        let template = MultivariatePolynomial::<_, i16>::new(&field, None, vars.clone());
        let children = [
            template.constant(atom("a^2-1")),
            template.constant(atom("(1-a)*(a+1)")),
        ];
        let mut coefficients = Vec::new();
        let mut exponents = Vec::new();
        for child in children {
            assert_eq!(child.nterms(), 1);
            coefficients.extend(child.coefficients);
            exponents.extend(child.exponents);
        }
        let expected = MultivariatePolynomial::<_, i16>::from_coefficient_list(
            coefficients.clone(),
            exponents.clone(),
            vars.clone(),
            &field,
        );
        let actual = MultivariatePolynomial::<_, i16>::from_nonzero_coefficient_list(
            coefficients,
            exponents,
            vars.clone(),
            &field,
        );
        assert_eq!(actual, expected);
        assert!(actual.is_zero());
        assert_eq!(actual.variables(), &vars);
        actual.check_consistency();
    }
}
