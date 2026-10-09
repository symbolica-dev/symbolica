use numerica::domains::float::{
    Complex, DoubleFloat, ErrorPropagatingFloat, Float, FloatLike, Real, RealLike, SingleFloat,
};

fn tracked<T: FloatLike>(value: T, accuracy: f64) -> ErrorPropagatingFloat<T> {
    ErrorPropagatingFloat::new_with_accuracy(value, accuracy)
}

fn scaled_hypot<T: Real + RealLike>(one: T) {
    let big = one.from_usize(10).pow(200);
    let tiny = big.inv();
    for scale in [big, tiny] {
        for sign in [1, -1] {
            let x = scale.clone() * one.from_i64(sign);
            let y = scale.clone() * one.from_usize(2);
            let expected = x.hypot(&y);
            let a = tracked(x, 220.);
            let b = tracked(y, 220.);
            let value = a.hypot(&b);
            assert!(value.is_finite());
            assert!(((value.get_num().clone() / &expected).to_f64() - 1.).abs() < 1e-14);
            assert!(value.get_absolute_error() > 0.);
            let reversed = b.hypot(&a);
            assert!(((reversed.get_num().clone() / &expected).to_f64() - 1.).abs() < 1e-14);
            let norm = Complex::new(a, b).norm();
            assert!(norm.re.is_finite());
            assert!(((norm.re.get_num().clone() / &expected).to_f64() - 1.).abs() < 1e-14);
        }
    }
}

#[test]
fn tracked_hypot_uses_existing_scaled_native_arithmetic() {
    scaled_hypot(1_f64);
    scaled_hypot(DoubleFloat::from(1.));
    scaled_hypot(Float::with_val(192, 1));
}

#[test]
fn tracked_hypot_retains_local_uncertainty() {
    let value = tracked(3_f64, 6.).hypot(&tracked(4., 6.));
    assert_eq!(*value.get_num(), 5.);
    // Existing linear propagation can overestimate shared intermediates, but
    // must retain both first derivatives, 3/5 and 4/5.
    assert!(value.get_absolute_error() >= 1.4e-6 * (1. - 1e-14));
    let value = tracked(3_f64, 6.).hypot(&tracked(0., 6.));
    assert_eq!(*value.get_num(), 3.);
    assert!(value.get_absolute_error() >= 1e-6);
}

#[test]
fn uncertain_origin_never_becomes_a_fabricated_exact_zero() {
    for (a, b) in [
        (tracked(0_f64, f64::INFINITY), tracked(0., 6.)),
        (tracked(0_f64, 6.), tracked(0., f64::INFINITY)),
    ] {
        let value = a.hypot(&b);
        assert_eq!(*value.get_num(), 0.);
        // The inherited arithmetic cannot differentiate the norm at the
        // origin. Preserve its nonfinite uncertainty rather than inventing a
        // derivative or deleting the uncertain input in a zero shortcut.
        assert!(!value.get_absolute_error().is_finite());
    }
}

#[test]
fn complex_square_root_retains_imaginary_uncertainty() {
    let value = Complex::new(tracked(1e-6_f64, f64::INFINITY), tracked(0., 12.)).sqrt();
    assert!((*value.re.get_num() - 1e-3).abs() < 1e-18);
    assert_eq!(*value.im.get_num(), 0.);
    assert!(value.im.get_absolute_error() >= 5e-10 * (1. - 1e-14));
}

#[test]
fn scalar_comparison_and_zero_predicates_are_unchanged() {
    let uncertain = tracked(0_f64, 12.);
    assert!(uncertain.real_cmp(&tracked(1., 12.)).is_none());
    assert!(uncertain.real_classify().is_none());
    assert!(!uncertain.needs_rescaling());
    assert!(uncertain.is_fully_zero());
    assert!(uncertain.is_zero());
}

#[test]
fn exceptional_inputs_preserve_existing_uncertainty_arithmetic() {
    for a in [f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
        for b in [0., 1., f64::NEG_INFINITY, f64::INFINITY, f64::NAN] {
            let expected =
                (tracked(a, 6.) * tracked(a, 6.) + tracked(b, 6.) * tracked(b, 6.)).sqrt();
            let value = tracked(a, 6.).hypot(&tracked(b, 6.));
            if expected.get_num().is_nan() {
                assert!(value.get_num().is_nan());
            } else {
                assert_eq!(value.get_num(), expected.get_num());
            }
            if expected.get_absolute_error().is_nan() {
                assert!(value.get_absolute_error().is_nan());
            } else {
                assert_eq!(value.get_absolute_error(), expected.get_absolute_error());
            }
        }
    }
}
