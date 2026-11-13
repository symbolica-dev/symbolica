use std::cmp::Ordering::{Equal, Greater, Less};

use numerica::domains::{
    float::{DoubleFloat, ElementaryFunctions, F64, Float, Numeric, RealScalar},
    rational::Rational,
};

#[test]
fn real_scalar_signs_preserve_zero_and_reject_nan() {
    for (value, expected) in [
        (f64::NEG_INFINITY, Some(Less)),
        (-2.0, Some(Less)),
        (-0.0, Some(Equal)),
        (0.0, Some(Equal)),
        (2.0, Some(Greater)),
        (f64::INFINITY, Some(Greater)),
        (f64::NAN, None),
    ] {
        let wrapped = F64(value);
        let double = DoubleFloat::from(value);
        let float = Float::with_val(96, value);
        assert!(Numeric::is_self_conjugate(&value));
        assert!(wrapped.is_self_conjugate());
        assert!(double.is_self_conjugate());
        assert!(float.is_self_conjugate());
        assert_eq!(value.real_sign(), expected);
        assert_eq!(wrapped.real_sign(), expected);
        assert_eq!(double.real_sign(), expected);
        assert_eq!(float.real_sign(), expected);
    }
    for (value, expected) in [
        (Rational::from((-1, 3)), Less),
        (Rational::from(0), Equal),
        (Rational::from((1, 3)), Greater),
    ] {
        assert!(value.is_self_conjugate());
        assert_eq!(value.real_sign(), Some(expected));
    }
}

#[test]
fn signed_powers_cover_full_exponent_range_and_tiny_results() {
    assert_eq!(Numeric::powi(&-1.0, i64::MIN), 1.0);
    assert_eq!(Numeric::powi(&-1.0, i64::MAX), -1.0);
    assert_eq!(Numeric::powi(&2.0, -1074), f64::from_bits(1));
    assert!(Numeric::powi(&1e160_f64, -2) > 0.0);
    assert_eq!(Numeric::powi(&-0.0_f64, -3), f64::NEG_INFINITY);
    assert_eq!(Numeric::powi(&f64::NAN, 0), 1.0);
    let exact = Rational::from((-3, 2));
    assert_eq!(exact.square(), Rational::from((9, 4)));
    assert_eq!(Numeric::powi(&exact, -3), Rational::from((-8, 27)));
    assert_eq!(
        Numeric::powi(&Rational::from(-1), i64::MIN),
        Rational::from(1)
    );
}

#[test]
fn expm1_preserves_tiny_values_precision_and_signed_zero() {
    for value in [-1e-200_f64, -1e-20, 1e-20, 1e-200] {
        assert_eq!(ElementaryFunctions::expm1(&value), value.exp_m1());
        assert_eq!(F64(value).expm1().0, value.exp_m1());

        let float = Float::with_val(160, value);
        let reference = float.expm1();
        assert_eq!(reference.get_precision(), 160);
        assert!((reference.clone() / &float - 1).norm().to_f64() < 1e-18);

        let double = DoubleFloat::from(value).expm1();
        let error = (Float::from(double) / reference - 1).norm().to_f64();
        assert!(error < 2e-31, "x={value}, relative error={error}");
    }
    for value in [-0.0_f64, 0.0] {
        assert_eq!(
            ElementaryFunctions::expm1(&value).to_bits(),
            value.to_bits()
        );
        assert_eq!(F64(value).expm1().0.to_bits(), value.to_bits());
        assert_eq!(
            DoubleFloat::from(value).expm1().into_inner().hi().to_bits(),
            value.to_bits()
        );
        assert_eq!(
            Float::with_val(96, value).expm1().is_sign_negative(),
            value.is_sign_negative()
        );
    }
    for value in [f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(DoubleFloat::from(value).expm1().to_f64(), value.exp_m1());
        assert_eq!(Float::with_val(96, value).expm1().to_f64(), value.exp_m1());
    }
    assert!(DoubleFloat::from(f64::NAN).expm1().to_f64().is_nan());
    assert!(Float::with_val(96, f64::NAN).expm1().to_f64().is_nan());
}

#[test]
fn paired_trigonometric_results_match_scalar_functions() {
    for value in [-1.25, -0.0, 0.0, 0.25, 10.0] {
        assert_eq!(ElementaryFunctions::sin_cos(&value), value.sin_cos());
        let wrapped = F64(value).sin_cos();
        assert_eq!((wrapped.0.0, wrapped.1.0), value.sin_cos());
        let float = Float::with_val(160, value);
        assert_eq!(float.sin_cos(), (float.sin(), float.cos()));
        let double = DoubleFloat::from(value);
        let (sine, cosine) = double.sin_cos();
        let sine_error = (Float::from(sine) - float.sin()).norm().to_f64();
        let cosine_error = (Float::from(cosine) - float.cos()).norm().to_f64();
        assert!(sine_error < 1e-30, "x={value}, sine error={sine_error}");
        assert!(
            cosine_error < 1e-30,
            "x={value}, cosine error={cosine_error}"
        );
    }
}

#[test]
fn simd_signs_require_agreement_and_expm1_preserves_lanes() {
    use wide::f64x4;
    assert_eq!(
        f64x4::from([-1.0, -2.0, -3.0, -4.0]).real_sign(),
        Some(Less)
    );
    assert_eq!(f64x4::from([1.0, 2.0, 3.0, 4.0]).real_sign(), Some(Greater));
    assert_eq!(f64x4::from([0.0, -0.0, 0.0, -0.0]).real_sign(), Some(Equal));
    assert_eq!(f64x4::from([-1.0, 2.0, 3.0, 4.0]).real_sign(), None);
    assert_eq!(f64x4::from([0.0, 2.0, 3.0, 4.0]).real_sign(), None);
    assert_eq!(f64x4::from([f64::NAN, 2.0, 3.0, 4.0]).real_sign(), None);
    let values = [-1e-200_f64, 1e-20, -0.0, -1000.0];
    let lanes = f64x4::from(values);
    assert!(lanes.is_self_conjugate());
    for (result, value) in lanes.expm1().to_array().into_iter().zip(values) {
        assert_eq!(result.to_bits(), value.exp_m1().to_bits());
    }
    let (sine, cosine) = ElementaryFunctions::sin_cos(&lanes);
    assert_eq!(sine.to_array(), lanes.sin().to_array());
    assert_eq!(cosine.to_array(), lanes.cos().to_array());
}
