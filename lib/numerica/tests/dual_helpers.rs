use std::cmp::Ordering;

use numerica::{
    create_hyperdual_from_components,
    domains::{
        dual::HyperDual,
        float::{Complex, ElementaryFunctions, ErrorPropagatingFloat, Numeric},
        rational::Rational,
    },
};

create_hyperdual_from_components!(SecondOrder, [[0], [1], [2]]);

fn complex(re: i64, im: i64) -> Complex<Rational> {
    Complex::new(re.into(), im.into())
}

fn make_dynamic<T: Numeric>(values: [T; 3]) -> HyperDual<T> {
    HyperDual::from_values(vec![vec![0], vec![1], vec![2]], values.to_vec())
}

#[test]
fn realness_and_sign_include_every_dual_component() {
    for (values, real, sign) in [
        (
            [complex(1, 0), complex(2, 0), complex(-3, 0)],
            true,
            Some(Ordering::Greater),
        ),
        (
            [complex(-1, 0), complex(2, 0), complex(-3, 0)],
            true,
            Some(Ordering::Less),
        ),
        (
            [complex(0, 0), complex(0, 0), complex(0, 0)],
            true,
            Some(Ordering::Equal),
        ),
        ([complex(0, 0), complex(1, 0), complex(0, 0)], true, None),
        ([complex(1, 0), complex(0, 1), complex(0, 0)], false, None),
        ([complex(1, 0), complex(0, 0), complex(0, 1)], false, None),
        ([complex(0, 1), complex(0, 0), complex(0, 0)], false, None),
    ] {
        let fixed = SecondOrder {
            values: values.clone(),
        };
        let dynamic = make_dynamic(values);
        assert_eq!(fixed.is_self_conjugate(), real);
        assert_eq!(fixed.real_sign(), sign);
        assert_eq!(dynamic.is_self_conjugate(), real);
        assert_eq!(dynamic.real_sign(), sign);
    }
}

#[test]
fn nested_dual_realness_and_sign_are_recursive() {
    let zero = SecondOrder {
        values: [complex(0, 0), complex(0, 0), complex(0, 0)],
    };
    let mut value = SecondOrder {
        values: [zero.one(), zero.clone(), zero.clone()],
    };
    assert!(value.is_self_conjugate());
    assert_eq!(value.real_sign(), Some(Ordering::Greater));

    value.values[2].values[1] = complex(0, 1);
    assert!(!value.is_self_conjugate());
    assert_eq!(value.real_sign(), None);

    value = value.zero();
    assert_eq!(value.real_sign(), Some(Ordering::Equal));
    value.values[2].values[1] = complex(1, 0);
    assert!(value.is_self_conjugate());
    assert_eq!(value.real_sign(), None);
}

#[test]
fn uncertain_zero_derivatives_do_not_prove_self_conjugacy() {
    let zero = ErrorPropagatingFloat::new_with_absolute_error(0.0_f64, 0.0);
    let mut imaginary = SecondOrder {
        values: std::array::from_fn(|_| zero.clone()),
    };
    imaginary.values[1] = ErrorPropagatingFloat::new_with_absolute_error(0.0, 1e-6);
    let dynamic = make_dynamic(imaginary.values.clone());
    assert!(!imaginary.is_exact_zero());
    assert!(!dynamic.is_exact_zero());
    assert!(!Complex::new(imaginary.one(), imaginary).is_self_conjugate());
    assert!(!Complex::new(dynamic.one(), dynamic).is_self_conjugate());
}

#[test]
fn exact_dual_square_and_signed_powers_preserve_derivatives() {
    let values = [Rational::from(2), Rational::from(1), Rational::from(0)];
    let fixed = SecondOrder {
        values: values.clone(),
    };
    let dynamic = make_dynamic(values);
    let squared = [Rational::from(4), Rational::from(4), Rational::from(1)];
    let inverse_squared = [
        Rational::from((1, 4)),
        Rational::from((-1, 4)),
        Rational::from((3, 16)),
    ];
    assert_eq!(fixed.square().values, squared);
    assert_eq!(dynamic.square().values, squared);
    assert_eq!(fixed.powi(-2).values, inverse_squared);
    assert_eq!(dynamic.powi(-2).values, inverse_squared);

    for base in [-1, 1] {
        let fixed = SecondOrder {
            values: [Rational::from(base), Rational::from(0), Rational::from(0)],
        };
        let dynamic = make_dynamic(fixed.values.clone());
        for exponent in [i64::MIN, i64::MAX, 0] {
            let expected = if base == -1 && exponent % 2 != 0 {
                -1
            } else {
                1
            };
            let expected = [
                Rational::from(expected),
                Rational::from(0),
                Rational::from(0),
            ];
            assert_eq!(fixed.powi(exponent).values, expected);
            assert_eq!(dynamic.powi(exponent).values, expected);
        }
    }
}

#[test]
fn expm1_preserves_tiny_primals_and_higher_derivatives() {
    for primal in [1e-20_f64, -1e-20, 0.0, -0.0, f64::from_bits(1)] {
        let values = [primal, 2.0, 3.0];
        let fixed = SecondOrder { values }.expm1();
        let dynamic = make_dynamic(values).expm1();
        let expected = [primal.exp_m1(), 2.0 * primal.exp(), 5.0 * primal.exp()];
        for actual in [&fixed.values[..], &dynamic.values[..]] {
            assert_eq!(actual[0].to_bits(), expected[0].to_bits());
            assert_eq!(&actual[1..], &expected[1..]);
        }
    }
}

#[test]
fn expm1_preserves_nested_mixed_derivatives() {
    let zero = SecondOrder {
        values: [0.0, 0.0, 0.0],
    };
    let value = SecondOrder {
        values: [
            SecondOrder {
                values: [1e-20, 1.0, 0.0],
            },
            zero.one(),
            zero,
        ],
    };
    let result = value.expm1();
    assert_eq!(result.values[0].values[0], 1e-20);
    assert_eq!(result.values[1].values[1], 1.0);
    assert_eq!(result.values[2].values[2], 0.25);
}

#[test]
fn sin_cos_preserves_real_and_complex_derivatives() {
    let primal = 0.3_f64;
    let values = [primal, 2.0, 3.0];
    let fixed = SecondOrder { values }.sin_cos();
    let dynamic = make_dynamic(values).sin_cos();
    let (s, c) = primal.sin_cos();
    let expected_sine = [s, 2.0 * c, 3.0 * c - 2.0 * s];
    let expected_cosine = [c, -2.0 * s, -3.0 * s - 2.0 * c];
    for (actual, expected) in [
        (&fixed.0.values[..], expected_sine),
        (&fixed.1.values[..], expected_cosine),
        (&dynamic.0.values[..], expected_sine),
        (&dynamic.1.values[..], expected_cosine),
    ] {
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-14);
        }
    }

    let z = Complex::new(0.3, -0.2);
    let a = Complex::new(2.0, 1.0);
    let b = Complex::new(-1.0, 0.5);
    let (s, c) = (z.sin(), z.cos());
    let expected_sine = [s, c * a, c * b - s * a * a / 2.0];
    let expected_cosine = [c, -s * a, -s * b - c * a * a / 2.0];
    let fixed = SecondOrder { values: [z, a, b] }.sin_cos();
    let dynamic = make_dynamic([z, a, b]).sin_cos();
    for (actual, expected) in [
        (&fixed.0.values[..], expected_sine),
        (&fixed.1.values[..], expected_cosine),
        (&dynamic.0.values[..], expected_sine),
        (&dynamic.1.values[..], expected_cosine),
    ] {
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual.re - expected.re).abs() < 1e-14);
            assert!((actual.im - expected.im).abs() < 1e-14);
        }
    }
}
