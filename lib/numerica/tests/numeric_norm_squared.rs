use numerica::{
    create_hyperdual_from_components,
    domains::{
        dual::HyperDual,
        float::{Complex, ElementaryFunctions, Numeric},
        rational::Rational,
    },
};

create_hyperdual_from_components!(SecondOrder, [[0], [1], [2]]);

fn squared_norm<T: Numeric>(value: &T) -> T {
    value.norm_squared()
}

fn complex(re: i64, im: i64) -> Complex<Rational> {
    Complex::new(re.into(), im.into())
}

#[test]
fn exact_real_and_complex_values_need_no_elementary_functions() {
    let rational = Rational::from((-3, 2));
    assert_eq!(squared_norm(&rational), Rational::from((9, 4)));
    assert_eq!(rational.conj(), rational);

    let value = Complex::new(Rational::from((3, 2)), Rational::from((-2, 3)));
    let expected = Complex::new(Rational::from((97, 36)), Rational::from(0));
    let concrete: Rational = value.norm_squared();
    assert_eq!(concrete, expected.re);
    assert_eq!(squared_norm(&value), expected);
    assert_eq!(Numeric::norm_squared(&value), expected);
    assert_eq!(Numeric::conj(&value).im, Rational::from((2, 3)));
}

#[test]
fn static_and_dynamic_duals_conjugate_every_derivative_component() {
    for (values, expected) in [
        // |1 + i*t|^2 = 1 + t^2 despite the real primal component.
        (
            [complex(1, 0), complex(0, 1), complex(0, 0)],
            [complex(1, 0), complex(0, 0), complex(1, 0)],
        ),
        // |(3 + 4i) + (2 - i)*t + (-1 + 3i)*t^2|^2 through order two.
        (
            [complex(3, 4), complex(2, -1), complex(-1, 3)],
            [complex(25, 0), complex(4, 0), complex(23, 0)],
        ),
    ] {
        let fixed = SecondOrder {
            values: values.clone(),
        };
        let dynamic = HyperDual::from_values(vec![vec![0], vec![1], vec![2]], values.to_vec());

        // Dual equality only compares the primal, so inspect all coefficients.
        assert_eq!(squared_norm(&fixed).values, expected);
        assert_eq!(squared_norm(&dynamic).values.as_slice(), &expected);
    }
}

#[test]
fn nested_duals_preserve_mixed_terms() {
    let inner = |values| SecondOrder { values };
    // z(s, t) = 1 + i*s + i*t, represented as a dual of duals.
    let value = SecondOrder {
        values: [
            inner([complex(1, 0), complex(0, 1), complex(0, 0)]),
            inner([complex(0, 1), complex(0, 0), complex(0, 0)]),
            inner([complex(0, 0), complex(0, 0), complex(0, 0)]),
        ],
    };
    let result = squared_norm(&value);
    // |z(s, t)|^2 = 1 + s^2 + 2*s*t + t^2.
    for (actual, expected) in result.values.iter().zip([
        [complex(1, 0), complex(0, 0), complex(1, 0)],
        [complex(0, 0), complex(2, 0), complex(0, 0)],
        [complex(1, 0), complex(0, 0), complex(0, 0)],
    ]) {
        assert_eq!(actual.values, expected);
    }
}

#[test]
fn norm_retains_range_when_squared_norm_overflows_or_underflows() {
    for scale in [1e200_f64, 1e-200_f64] {
        let scalar = -3.0 * scale;
        assert_eq!(ElementaryFunctions::norm(&scalar), 3.0 * scale);

        let value = Complex::new(3.0 * scale, 4.0 * scale);
        let norm = value.norm();
        assert!((norm.re / (5.0 * scale) - 1.0).abs() < 4e-16);
        assert_eq!(norm.im, 0.0);

        let scalar_squared = squared_norm(&scalar);
        let complex_squared = squared_norm(&value);
        assert_eq!(complex_squared.im, 0.0);
        if scale > 1.0 {
            assert!(scalar_squared.is_infinite());
            assert!(complex_squared.re.is_infinite());
        } else {
            assert_eq!(scalar_squared, 0.0);
            assert_eq!(complex_squared.re, 0.0);
        }
    }
}

#[test]
fn infinite_complex_squared_norm_has_zero_imaginary_part() {
    for value in [
        Complex::new(f64::INFINITY, 0.0),
        Complex::new(0.0, f64::NEG_INFINITY),
        Complex::new(f64::INFINITY, f64::INFINITY),
    ] {
        let squared = squared_norm(&value);
        assert_eq!(squared.re, f64::INFINITY);
        assert_eq!(squared.im, 0.0);
    }
}

#[test]
fn simd_squared_norm_operates_lane_by_lane() {
    let re = wide::f64x4::from([-3.0, 0.0, 5.0, -8.0]);
    let im = wide::f64x4::from([4.0, -2.0, 12.0, 15.0]);
    assert_eq!(squared_norm(&re).to_array(), [9.0, 0.0, 25.0, 64.0]);
    let squared = squared_norm(&Complex::new(re, im));
    assert_eq!(squared.re.to_array(), [25.0, 4.0, 169.0, 289.0]);
    assert_eq!(squared.im.to_array(), [0.0; 4]);
}
