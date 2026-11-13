use numerica::domains::{
    float::{Complex, Float, Numeric},
    rational::Rational,
};

#[test]
fn algebraic_square_is_distinct_from_squared_norm() {
    let value = Complex::new(Rational::from(3), Rational::from(4));
    assert_eq!(
        value.square(),
        Complex::new(Rational::from(-7), Rational::from(24))
    );
    assert_eq!(value.norm_squared(), Rational::from(25));
}

#[test]
fn signed_powers_preserve_exact_arithmetic_and_support_full_exponent_range() {
    let value = Rational::from((2, 3));
    assert_eq!(value.powi(-3), Rational::from((27, 8)));
    assert_eq!(value.powi(3), Rational::from((8, 27)));
    assert_eq!(value.powi(0), Rational::one());

    let minus_one = Rational::from(-1);
    assert_eq!(minus_one.powi(i64::MIN), Rational::one());
    assert_eq!(minus_one.powi(i64::MAX), minus_one);
    assert_eq!(Numeric::powi(&-1.0_f64, i64::MIN), 1.0);
    assert_eq!(Numeric::powi(&-1.0_f64, i64::MAX), -1.0);

    let imaginary_unit = Complex::new(Rational::from(0), Rational::from(1));
    assert_eq!(imaginary_unit.powi(-1), imaginary_unit.conj());
    assert_eq!(imaginary_unit.powi(i64::MIN), imaginary_unit.one());
}

#[test]
fn negative_powers_avoid_overflowing_the_positive_power() {
    let result = Numeric::powi(&1e100_f64, -3);
    assert!((result / 1e-300 - 1.0).abs() < 1e-14);
    assert!(Numeric::powi(&-0.0_f64, -3).is_sign_negative());
    assert_eq!(Numeric::powi(&-0.0_f64, -3), f64::NEG_INFINITY);
}

#[test]
fn powers_preserve_the_precision_of_nonzero_complex_components() {
    let value = Complex::new(Float::new(2), Float::with_val(160, 1));
    let result = value.powi(0);
    assert_eq!(result.re, Float::with_val(160, 1));
    assert_eq!(result.get_precision(), 160);

    let value = Rational::from((1, 3)).to_multi_prec_float(160);
    assert_eq!(value.powi(1), value);
    assert_eq!(value.powi(1).get_precision(), 160);
}
