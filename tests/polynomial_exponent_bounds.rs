//! Compact powers exercise exponent validation without expanding large sums.
use symbolica::{
    atom::{Atom, AtomCore},
    domains::atom::AtomField,
    parse,
};

#[test]
fn powers_beyond_signed_conversion_range_remain_coefficients() {
    let variable = parse!("poly_exponent_bounds::q");
    let field = AtomField {
        statistical_zero_test: false,
        cancel_check_on_division: true,
        custom_normalization: None,
    };
    for exponent in [
        "2147483648",
        "4294967295",
        "4294967296",
        "18446744073709551616",
    ] {
        let expression = Atom::parse(
            &format!("poly_exponent_bounds::q^{exponent}"),
            "poly_exponent_bounds",
            Default::default(),
        )
        .unwrap();
        let polynomial =
            expression.to_polynomial_in_vars_with_field::<u32>(&[variable.clone()], &field);
        assert_eq!(polynomial.nterms(), 1);
        let term = (&polynomial).into_iter().next().unwrap();
        assert_eq!(term.exponents, &[0]);
        assert_eq!(term.coefficient, &expression);
    }
}

#[test]
fn ordinary_and_wide_valid_exponents_keep_exact_values() {
    let variable = parse!("poly_exponent_bounds::q");
    for exponent in [0u32, 1, 7, 65535, 65536, 100000] {
        let expression = variable.pow(i64::from(exponent));
        let polynomial = expression.to_polynomial_in_vars::<u32>(&[variable.clone()]);
        assert_eq!(polynomial.nterms(), 1);
        assert_eq!(
            (&polynomial).into_iter().next().unwrap().exponents,
            &[exponent]
        );
    }
}
