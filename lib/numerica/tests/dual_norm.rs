use numerica::{
    create_hyperdual_from_components,
    domains::{
        dual::HyperDual,
        float::{Complex, ElementaryFunctions, Numeric},
    },
};

create_hyperdual_from_components!(FirstOrder, [[0], [1]]);
create_hyperdual_from_components!(SecondOrder, [[0], [1], [2]]);
create_hyperdual_from_components!(MixedOrder, [[0, 0], [1, 0], [0, 1], [2, 0], [1, 1]]);

fn make_dynamic<T: Numeric>(values: [T; 3]) -> HyperDual<T> {
    HyperDual::from_values(vec![vec![0], vec![1], vec![2]], values.to_vec())
}

fn assert_real_complex(actual: Complex<f64>, expected: f64) {
    assert!(
        (actual.re - expected).abs() < 2e-13,
        "{actual:?} != {expected}"
    );
    assert!(actual.im.abs() < 2e-13, "{actual:?} is not real");
}

#[test]
fn complex_norm_differentiates_real_parameters_through_mixed_orders() {
    // |3 + 4i + s + it|, with Taylor coefficients rather than raw derivatives.
    let values = [
        Complex::new(3.0, 4.0),
        Complex::new(1.0, 0.0),
        Complex::new(0.0, 1.0),
        Complex::new(0.0, 0.0),
        Complex::new(0.0, 0.0),
    ];
    let fixed = MixedOrder { values }.norm();
    let dynamic = HyperDual::from_values(
        vec![vec![0, 0], vec![1, 0], vec![0, 1], vec![2, 0], vec![1, 1]],
        values.to_vec(),
    )
    .norm();
    for result in [&fixed.values[..], &dynamic.values[..]] {
        for (&actual, expected) in
            result
                .iter()
                .zip([5.0, 3.0 / 5.0, 4.0 / 5.0, 8.0 / 125.0, -12.0 / 125.0])
        {
            assert_real_complex(actual, expected);
        }
    }
}

#[test]
fn real_primal_with_imaginary_seed_keeps_the_second_order_term() {
    for primal in [3.0, -3.0] {
        // |±3 + it| = 3 + t²/6 + O(t⁴).
        let values = [
            Complex::new(primal, 0.0),
            Complex::new(0.0, 1.0),
            Complex::new(0.0, 0.0),
        ];
        let fixed = SecondOrder { values }.norm();
        let dynamic = make_dynamic(values).norm();
        for result in [&fixed.values[..], &dynamic.values[..]] {
            for (&actual, expected) in result.iter().zip([3.0, 0.0, 1.0 / 6.0]) {
                assert_real_complex(actual, expected);
            }
        }
    }
}

#[test]
fn real_norm_preserves_higher_coefficients_without_scaling() {
    for primal in [
        3.0_f64,
        -3.0,
        1e200,
        -1e200,
        1e-200,
        -1e-200,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let values = [primal, 2.0, -3.0];
        let expected = [primal.abs(), 2.0 * primal.signum(), -3.0 * primal.signum()];
        let fixed = SecondOrder { values }.norm();
        let dynamic = make_dynamic(values).norm();
        assert_eq!(fixed.values, expected);
        assert_eq!(dynamic.values.as_slice(), &expected);

        let values = values.map(|value| Complex::new(value, 0.0));
        let expected = expected.map(|value| Complex::new(value, 0.0));
        let fixed = SecondOrder { values }.norm();
        let dynamic = make_dynamic(values).norm();
        assert_eq!(fixed.values, expected);
        assert_eq!(dynamic.values.as_slice(), &expected);
    }
}

#[test]
fn complex_norm_preserves_extreme_primal_scales() {
    for scale in [1e200_f64, -1e200, 1e-200, -1e-200] {
        let values = [
            Complex::new(3.0 * scale, 4.0 * scale),
            Complex::new(1.0, 2.0),
        ];
        let fixed = FirstOrder { values }.norm();
        let dynamic = HyperDual::from_values(vec![vec![0], vec![1]], values.to_vec()).norm();
        for result in [&fixed.values[..], &dynamic.values[..]] {
            assert!((result[0].re / (5.0 * scale.abs()) - 1.0).abs() < 2e-15);
            assert_eq!(result[0].im, 0.0);
            assert_real_complex(result[1], 11.0 / 5.0 * scale.signum());
        }
    }
}

#[test]
fn constant_zero_norm_keeps_zero_derivatives_and_normalizes_signed_zero() {
    for primal in [0.0_f64, -0.0] {
        let values = [primal, 0.0, 0.0];
        let fixed = SecondOrder { values }.norm();
        let dynamic = make_dynamic(values).norm();
        for result in [&fixed.values[..], &dynamic.values[..]] {
            assert_eq!(result[0].to_bits(), 0.0_f64.to_bits());
            assert_eq!(&result[1..], &[0.0, 0.0]);
        }

        let values = values.map(|value| Complex::new(value, 0.0));
        let fixed = SecondOrder { values }.norm();
        let dynamic = make_dynamic(values).norm();
        for result in [&fixed.values[..], &dynamic.values[..]] {
            assert_eq!(result[0].re.to_bits(), 0.0_f64.to_bits());
            assert_eq!(result, &[Complex::new(0.0, 0.0); 3]);
        }
    }
}

#[test]
fn nonconstant_zero_primal_has_a_defined_value_but_singular_derivative() {
    // |t| is not differentiable at zero; retaining its seed would invent a derivative.
    let values = [0.0_f64, 1.0, 0.0];
    let fixed = SecondOrder { values }.norm();
    let dynamic = make_dynamic(values).norm();
    for result in [&fixed.values[..], &dynamic.values[..]] {
        assert_eq!(result[0], 0.0);
        assert!(result[1].is_nan());
    }

    let values = values.map(|value| Complex::new(value, 0.0));
    let fixed = SecondOrder { values }.norm();
    let dynamic = make_dynamic(values).norm();
    for result in [&fixed.values[..], &dynamic.values[..]] {
        assert_eq!(result[0], Complex::new(0.0, 0.0));
        assert!(result[1].re.is_nan() || result[1].im.is_nan());
    }
}

#[test]
fn nested_duals_do_not_use_primal_equality_as_a_realness_test() {
    // |3 + i(s + t)| = 3 + (s + t)²/6 - (s + t)⁴/216 + ... .
    let zero = Complex::new(0.0, 0.0);
    let values = [
        [Complex::new(3.0, 0.0), Complex::new(0.0, 1.0), zero],
        [Complex::new(0.0, 1.0), zero, zero],
        [zero; 3],
    ];
    let fixed = SecondOrder {
        values: values.map(|values| SecondOrder { values }),
    }
    .norm();
    let dynamic = make_dynamic(values.map(make_dynamic)).norm();
    let expected = [
        [3.0, 0.0, 1.0 / 6.0],
        [0.0, 1.0 / 3.0, 0.0],
        [1.0 / 6.0, 0.0, -1.0 / 36.0],
    ];
    // Dual PartialEq compares only the primal, so inspect every nested coefficient.
    for (outer, expected) in expected.into_iter().enumerate() {
        for (inner, expected) in expected.into_iter().enumerate() {
            assert_real_complex(fixed.values[outer].values[inner], expected);
            assert_real_complex(dynamic.values[outer].values[inner], expected);
        }
    }
}
