use std::cmp::Ordering;

use numerica::domains::{
    float::{Complex, ElementaryFunctions, ErrorPropagatingFloat, Float, Numeric, RealBall},
    rational::Rational,
};

#[test]
fn complex_self_conjugacy_and_sign_follow_the_conjugation_domain() {
    let real = Complex::new(Rational::from(-2), Rational::from(0));
    assert!(real.is_self_conjugate());
    assert!(Numeric::is_self_conjugate(&real));
    assert_eq!(real.real_sign(), Some(Ordering::Less));

    let imaginary = Complex::new(Rational::from(1), Rational::from(1));
    assert!(!imaginary.is_self_conjugate());
    assert_eq!(imaginary.real_sign(), None);
    let nested = Complex::new(imaginary, real.zero());
    // Outer conjugation leaves the real component unchanged, even when it is complex.
    assert!(nested.is_self_conjugate());
    assert_eq!(nested.conj(), nested);
    assert_eq!(nested.real_sign(), None);

    let invalid = Complex::new(f64::NAN, 0.0);
    assert_eq!(invalid.real_sign(), None);
}

#[test]
fn interval_sign_requires_a_definite_sign() {
    let ball = |lower, upper| {
        RealBall::from_bounds(Float::with_val(80, lower), Float::with_val(80, upper))
    };
    assert_eq!(ball(1, 2).real_sign(), Some(Ordering::Greater));
    assert_eq!(ball(-2, -1).real_sign(), Some(Ordering::Less));
    assert_eq!(ball(0, 0).real_sign(), Some(Ordering::Equal));
    for interval in [ball(-1, 1), ball(0, 1), ball(-1, 0)] {
        assert!(interval.is_self_conjugate());
        assert_eq!(interval.real_sign(), None);
    }
    // A conjugation-invariant set is not necessarily pointwise self-conjugate.
    let rectangle = Complex::new(ball(1, 2), ball(-1, 1));
    assert!(!rectangle.is_self_conjugate());
    assert_eq!(rectangle.real_sign(), None);
}

#[test]
fn interval_square_uses_the_same_input_twice() {
    let interval = RealBall::from_bounds(Float::with_val(80, -2), Float::with_val(80, 1));
    let squared = interval.square();
    assert_eq!(squared.lower_bound(), Float::with_val(80, 0));
    assert_eq!(squared.upper_bound(), Float::with_val(80, 4));
}

#[test]
fn tracked_sign_respects_uncertainty() {
    let tracked = |value, error| ErrorPropagatingFloat::new_with_absolute_error(value, error);
    assert_eq!(tracked(2.0_f64, 0.5).real_sign(), Some(Ordering::Greater));
    assert_eq!(tracked(-2.0_f64, 0.5).real_sign(), Some(Ordering::Less));
    assert_eq!(tracked(0.0_f64, 0.0).real_sign(), Some(Ordering::Equal));
    for value in [tracked(1.0, 2.0), tracked(1.0, 1.0), tracked(1.0, f64::NAN)] {
        assert!(value.is_self_conjugate());
        assert_eq!(value.real_sign(), None);
    }
}

#[test]
fn complex_expm1_retains_small_real_and_imaginary_components() {
    let value = Complex::new(1e-20_f64, 2e-20);
    let result = value.expm1();
    assert!((result.re / 1e-20 - 1.0).abs() < 4e-16);
    assert!((result.im / 2e-20 - 1.0).abs() < 4e-16);

    let imaginary = Complex::new(0.0_f64, 1e-100).expm1();
    assert!((imaginary.re / -5e-201 - 1.0).abs() < 4e-16);
    assert_eq!(imaginary.im, 1e-100);

    let negative = Complex::new(-1000.0_f64, 1.0).expm1();
    assert!((negative.re + 1.0).abs() < 4e-16);
    assert_eq!(negative.im, 0.0);
    let zero = Complex::new(-0.0_f64, -0.0).expm1();
    assert!(zero.re.is_sign_negative());
    assert!(zero.im.is_sign_negative());
}

#[test]
fn interval_and_tracked_expm1_keep_small_values() {
    let tiny = Float::with_val(100, 1e-100);
    let expected = tiny.expm1();
    let interval = RealBall::exact(tiny).expm1();
    assert!(interval.center > Float::with_val(100, 0));
    assert!(interval.contains(&expected));

    let x = 1e-20_f64;
    let error = 1e-22_f64;
    let tracked = ErrorPropagatingFloat::new_with_absolute_error(x, error).expm1();
    assert_eq!(tracked.value, x.exp_m1());
    for endpoint in [x - error, x + error] {
        assert!((endpoint.exp_m1() - tracked.value).abs() <= tracked.get_absolute_error());
    }
}

#[test]
fn combined_sine_cosine_preserves_complex_and_tracked_results() {
    for value in [Complex::new(0.3_f64, 0.0), Complex::new(0.3, -0.4)] {
        let (sin, cos) = value.sin_cos();
        let expected_sin = value.sin();
        let expected_cos = value.cos();
        assert!((sin.re - expected_sin.re).abs() < 4e-16);
        assert!((sin.im - expected_sin.im).abs() < 4e-16);
        assert!((cos.re - expected_cos.re).abs() < 4e-16);
        assert!((cos.im - expected_cos.im).abs() < 4e-16);
    }
    let value = ErrorPropagatingFloat::new_with_absolute_error(0.3_f64, 1e-5);
    let (sin, cos) = value.sin_cos();
    assert_eq!(sin.value, value.sin().value);
    assert_eq!(cos.value, value.cos().value);
    assert!(sin.get_absolute_error() > 0.0);
    assert!(cos.get_absolute_error() > 0.0);
}
