use std::sync::Arc;

use symbolica::prelude::*;

fn check_product_and_constants<R: Ring>(
    polynomial: &MultivariatePolynomial<R, u32>,
    factors: &[(MultivariatePolynomial<R, u32>, usize)],
) {
    if polynomial.is_zero() {
        assert!(factors.is_empty());
        return;
    }

    let mut product = polynomial.one();
    let mut constants = 0;
    for (index, (factor, multiplicity)) in factors.iter().enumerate() {
        assert!(*multiplicity > 0);
        if factor.is_constant() {
            constants += 1;
            assert_eq!(index, 0, "the scalar factor must come first");
            assert_eq!(*multiplicity, 1);
            assert!(polynomial.is_constant() || !factor.is_one());
        }
        product = &product * &factor.pow(*multiplicity);
    }
    assert!(constants <= 1);
    // Trager's algorithm may retain an unused coordinate for the extension generator.
    let mut expected = polynomial.clone();
    product.unify_variables(&mut expected);
    assert_eq!(product, expected);
}

#[test]
fn integer_factor_signs_and_order_are_repeatable_with_unused_variables() {
    let variables = Arc::new(
        (0..10)
            .map(|index| symbol!(format!("factor_normalization_v{index}")).into())
            .collect(),
    );
    let polynomial =
        parse!("factor_normalization_v5*(factor_normalization_v6^4-4*factor_normalization_v7^2)")
            .to_polynomial::<_, u32>(&Z, Some(variables));
    let reference = polynomial.factor();
    assert_eq!(reference.len(), 3);
    check_product_and_constants(&polynomial, &reference);
    for (factor, _) in &reference {
        assert_eq!(factor.variables(), polynomial.variables());
        assert!(!factor.is_constant());
        assert!(!factor.lcoeff().is_negative());
        assert!(factor.content().is_one());
    }
    // Compare the actual returned vectors: sorting here would conceal the bug.
    for _ in 0..64 {
        assert_eq!(polynomial.factor(), reference);
    }

    let negative = -polynomial.clone();
    let mut expected = vec![(polynomial.constant(Integer::from(-1)), 1)];
    expected.extend(reference);
    for _ in 0..16 {
        let factors = negative.factor();
        assert_eq!(factors, expected);
        check_product_and_constants(&negative, &factors);
    }
}

#[test]
fn integer_units_respect_odd_and_even_multiplicities() {
    let variables = Arc::new(vec![
        symbol!("x").into(),
        symbol!("y").into(),
        symbol!("unused").into(),
    ]);
    for (expression, constant) in [
        ("-6*(x-y)^3*(x+y)^2", -6),
        ("6*(y-x)^3*(x+y)^2", -6),
        ("-6*(y-x)^2*(x+y)^3", -6),
        ("-12*x^3*(2*x+1)^2", -12),
    ] {
        let polynomial = parse!(expression)
            .expand()
            .to_polynomial::<_, u32>(&Z, Some(variables.clone()));
        let reference = polynomial.factor();
        assert_eq!(reference.len(), 3, "{expression}");
        assert_eq!(
            reference[0],
            (polynomial.constant(Integer::from(constant)), 1),
            "{expression}",
        );
        for (factor, _) in &reference[1..] {
            assert!(!factor.lcoeff().is_negative(), "{expression}: {factor}");
            assert!(factor.content().is_one());
        }
        check_product_and_constants(&polynomial, &reference);
        for _ in 0..8 {
            assert_eq!(polynomial.factor(), reference, "{expression}");
        }
    }
}

#[test]
fn rational_content_combines_with_the_integer_unit() {
    let variables = Arc::new(vec![symbol!("x").into(), symbol!("unused").into()]);
    let polynomial = parse!("-2/3*(2*x+1)^3*(x-1)^2")
        .expand()
        .to_polynomial::<_, u32>(&Q, Some(variables));
    let reference = polynomial.factor();
    assert_eq!(reference.len(), 3);
    assert_eq!(
        reference[0],
        (
            parse!("-2/3").to_polynomial(&Q, polynomial.variables().clone()),
            1,
        )
    );
    for (factor, _) in &reference[1..] {
        assert!(!factor.lcoeff().is_negative());
        assert!(factor.content().is_one());
    }
    // Rational factors retain the primitive integer convention, rather than
    // moving the leading coefficient 2 of 2*x+1 into the scalar.
    assert!(reference.contains(&(
        parse!("2*x+1").to_polynomial(&Q, polynomial.variables().clone()),
        3,
    )));
    check_product_and_constants(&polynomial, &reference);
    for _ in 0..16 {
        assert_eq!(polynomial.factor(), reference);
    }
}

#[test]
fn constants_have_exactly_one_factor_except_zero() {
    for value in [0, 1, -1, 2, -6] {
        let polynomial = Atom::num(value).to_polynomial::<_, u32>(&Z, None);
        let factors = polynomial.factor();
        check_product_and_constants(&polynomial, &factors);
        if value != 0 {
            assert_eq!(factors, vec![(polynomial, 1)]);
        }
    }
    for expression in ["0", "1", "-1", "2", "-6", "-2/3", "5/7"] {
        let polynomial = parse!(expression).to_polynomial::<_, u32>(&Q, None);
        let factors = polynomial.factor();
        check_product_and_constants(&polynomial, &factors);
        if !polynomial.is_zero() {
            assert_eq!(factors, vec![(polynomial, 1)]);
        }
    }
}

#[test]
fn finite_field_factors_are_monic_and_ordered_repeatably() {
    let field = Zp::new(7);
    let variables = Arc::new(vec![symbol!("x").into(), symbol!("y").into()]);
    for expression in ["3*(2*x+1)^3*(x+2)^2", "3*(2*x+y+1)^3*(3*x+y+2)^2"] {
        let polynomial = parse!(expression)
            .expand()
            .to_polynomial::<_, u32>(&field, Some(variables.clone()));
        let reference = polynomial.factor();
        assert_eq!(reference.len(), 3);
        assert_eq!(reference[0], (polynomial.constant(polynomial.lcoeff()), 1));
        for (factor, _) in &reference[1..] {
            assert!(field.is_one(&factor.lcoeff()));
        }
        check_product_and_constants(&polynomial, &reference);
        for _ in 0..32 {
            assert_eq!(polynomial.factor(), reference, "{expression}");
        }
    }
}

#[test]
fn algebraic_factors_keep_the_scalar_separate() {
    let field = AlgebraicExtension::complex(Q);
    let polynomial = parse!("-3*(2*x+1)^2*(x^2+1)")
        .expand()
        .to_polynomial::<_, u32>(&field, None);
    let reference = polynomial.factor();
    assert_eq!(reference.len(), 4);
    assert_eq!(reference[0], (polynomial.constant(polynomial.lcoeff()), 1));
    for (factor, _) in &reference[1..] {
        assert!(field.is_one(&factor.lcoeff()));
    }
    check_product_and_constants(&polynomial, &reference);
    for _ in 0..8 {
        assert_eq!(polynomial.factor(), reference);
    }
}
