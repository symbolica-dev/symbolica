use std::str::FromStr;

use crate::domains::integer::Integer;

use super::DoubleFloat;
use super::{Complex, ErrorPropagatingFloat, Float, FloatLike, Rational, Real, RealLike};

fn assert_equal_hash<T: Eq + std::hash::Hash + std::fmt::Debug>(left: T, right: T) {
    use std::hash::{DefaultHasher, Hasher};
    assert_eq!(left, right);
    let hash = |value: &T| {
        let mut state = DefaultHasher::new();
        value.hash(&mut state);
        state.finish()
    };
    assert_eq!(hash(&left), hash(&right), "{left:?} and {right:?}");
    assert!(std::collections::HashSet::from([left]).contains(&right));
}

#[test]
fn signed_zero_hashes() {
    assert_equal_hash(super::F64::from(0.), super::F64::from(-0.));
    assert_equal_hash(DoubleFloat::from(0.), DoubleFloat::from(-0.));
    assert_equal_hash(Float::with_val(128, 0.), Float::with_val(128, -0.));
}

#[test]
fn signed_zero_norm_is_positive_zero() {
    let zero = Float::with_val(128, -0.).norm();
    assert_eq!(zero, Float::with_val(128, 0.));
    assert!(!zero.is_sign_negative());
}

#[test]
fn float_sign_predicates() {
    for value in [
        f64::NEG_INFINITY,
        -1.,
        -f64::from_bits(1),
        -0.,
        0.,
        f64::from_bits(1),
        1.,
        f64::INFINITY,
    ] {
        let float = Float::with_val(128, value);
        assert_eq!(float.is_negative(), value < 0., "{value}");
        assert_eq!(
            float.is_sign_negative(),
            value.is_sign_negative(),
            "{value}"
        );
    }
    for value in [f64::NAN, -f64::NAN] {
        assert!(!Float::with_val(128, value).is_negative());
    }
}

#[test]
fn float_copy_sign_preserves_signed_zero() {
    for sign in [0f64, -0., 1., -1.] {
        for value in [0f64, -0., 2., -2.] {
            let result = Float::with_val(128, value).copy_sign(&Float::with_val(128, sign));
            assert_eq!(result.to_f64(), value.copysign(sign));
            assert_eq!(result.is_sign_negative(), sign.is_sign_negative());
        }
    }
}

#[test]
fn float_signed_zero_branch_cuts() {
    for sign in [0f64, -0.] {
        let zero = Float::with_val(128, sign);
        let one = Float::with_val(128, 1.);
        for x in [0., -0., 1., -1.] {
            let result = zero.atan2(&Float::with_val(128, x));
            assert_eq!(result.to_f64(), sign.atan2(x));
            assert_eq!(result.is_sign_negative(), sign.is_sign_negative());
        }
        let z = Complex::new(-one.clone(), zero);
        let square_root = z.sqrt();
        assert_eq!(square_root.re.to_f64(), 0.);
        assert_eq!(square_root.im.to_f64(), 1f64.copysign(sign));
        let logarithm = z.log();
        assert_eq!(logarithm.re.to_f64(), 0.);
        assert_eq!(logarithm.im.to_f64(), std::f64::consts::PI.copysign(sign));
    }
}

#[test]
fn double_float_zero_low_component_hashes() {
    assert_equal_hash(DoubleFloat::from(-1.), -DoubleFloat::from(1.));
}

#[test]
fn float_equality_across_precisions() {
    for value in [0., -0., 1., -1.5, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(Float::with_val(53, value), Float::with_val(200, value));
    }
}

#[test]
fn float_hashes_across_precisions() {
    for value in [
        0.,
        -0.,
        1.,
        -1.5,
        f64::MIN_POSITIVE,
        f64::from_bits(1),
        f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        -f64::NAN,
    ] {
        for precision in [53, 64, 65, 128, 200, 256] {
            assert_equal_hash(
                Float::with_val(53, value),
                Float::with_val(precision, value),
            );
        }
    }
}

#[test]
#[cfg(feature = "float-astro")]
fn astro_raw_signed_zero_hashes() {
    assert_equal_hash(
        Float::with_val(128, 0.).into_raw(),
        Float::with_val(128, -0.).into_raw(),
    );
    assert_equal_hash(
        Float::with_val(53, 1.).into_raw(),
        Float::with_val(200, 1.).into_raw(),
    );
    assert_equal_hash(
        Float::with_val(53, f64::NAN).into_raw(),
        Float::with_val(200, -f64::NAN).into_raw(),
    );
}

#[test]
fn float_internal_ordering_distinguishes_nan() {
    use crate::domains::InternalOrdering;
    fn check<T: Eq + InternalOrdering>(values: Vec<T>) {
        for left in &values {
            for right in &values {
                assert_eq!(left == right, left.internal_cmp(right).is_eq());
                assert_eq!(left.internal_cmp(right), right.internal_cmp(left).reverse());
                for third in &values {
                    if left.internal_cmp(right).is_le() && right.internal_cmp(third).is_le() {
                        assert!(left.internal_cmp(third).is_le());
                    }
                }
            }
        }
    }
    let values = [
        f64::NEG_INFINITY,
        -1.,
        -0.,
        0.,
        1.,
        f64::INFINITY,
        f64::NAN,
        -f64::NAN,
    ];
    check(values.map(super::F64::from).to_vec());
    check(values.map(DoubleFloat::from).to_vec());
    check(values.map(|v| Float::with_val(128, v)).to_vec());
}

fn eval_test<T: Real>(v: &[T]) -> T {
    v[0].sqrt() + v[1].log() + v[1].sin() - v[0].cos() + v[1].tan() - v[2].asin() + v[3].acos()
        - v[0].atan2(&v[1])
        + v[1].sinh()
        - v[0].cosh()
        + v[1].tanh()
        - v[4].asinh()
        + v[1].acosh() / v[5].atanh()
        + v[1].powf(&v[0])
}

#[test]
fn multi_precision_float_raw_roundtrip() {
    let value = Float::with_val(128, 1.25);

    let borrowed = Float::from_raw(value.as_raw().clone());
    assert_eq!(borrowed, value);

    let copied = Float::from_raw(value.to_raw());
    assert_eq!(copied, value);

    let consumed = Float::from_raw(value.clone().into_raw());
    assert_eq!(consumed, value);
}

#[test]
fn double() {
    let r = eval_test(&[5., 7., 0.3, 0.5, 0.7, 0.4]);
    assert_eq!(r, 17293.219725825093);
}

#[test]
fn double_float() {
    let r = eval_test(&[
        DoubleFloat::from(5.),
        DoubleFloat::from(7.),
        DoubleFloat::from(3.) / DoubleFloat::from(10.),
        DoubleFloat::from(1.) / DoubleFloat::from(2.),
        DoubleFloat::from(7.) / DoubleFloat::from(10.),
        DoubleFloat::from(4.) / DoubleFloat::from(10.),
    ]);

    const N: u32 = 106;
    let expected = eval_test(&[
        Float::with_val(N, 5.),
        Float::with_val(N, 7.),
        Float::with_val(N, 3.) / Float::with_val(N, 10.),
        Float::with_val(N, 1.) / Float::with_val(N, 2.),
        Float::with_val(N, 7.) / Float::with_val(N, 10.),
        Float::with_val(N, 4.) / Float::with_val(N, 10.),
    ])
    .to_double_float();

    assert!((r - expected).norm() < DoubleFloat::from(2e-27));
}

#[test]
fn error_propagation() {
    let a = ErrorPropagatingFloat::new(5., 16.);
    let b = ErrorPropagatingFloat::new(7., 16.);
    let c = ErrorPropagatingFloat::new(0.3, 16.);
    let d = ErrorPropagatingFloat::new(0.5, 16.);
    let e = ErrorPropagatingFloat::new(0.7, 16.);
    let f = ErrorPropagatingFloat::new(0.4, 16.);

    let r = a.sqrt() + b.log() + b.sin() - a.cos() + b.tan() - c.asin() + d.acos() - a.atan2(&b)
        + b.sinh()
        - a.cosh()
        + b.tanh()
        - e.asinh()
        + b.acosh() / f.atanh()
        + b.powf(&a);
    assert_eq!(*r.get_num(), 17293.219725825093);
    // error is 14.836811363436391 when the f64 could have theoretically grown in between
    assert_eq!(r.get_precision(), Some(14.836795991431746));
}

#[test]
fn error_truncation() {
    let a = ErrorPropagatingFloat::new(0.0000000123456789, 9.)
        .exp()
        .log();
    assert_eq!(a.get_precision(), Some(8.046104745509947));
}

#[test]
fn large_cancellation() {
    let a = ErrorPropagatingFloat::new(Float::with_val(200, 1e-50), 60.);
    let r = (a.exp() - a.one()) / a;
    assert!(format!("{r}").starts_with("1.00000000"));
    assert!(r.get_precision().unwrap() > 9.);
}

#[test]
fn complex() {
    let a = Complex::new(1., 2.);
    let b: Complex<f64> = Complex::new(3., 4.);

    let r = a.sqrt() + b.log() - a.exp() + b.sin() - a.cos() + b.tan() - a.asin() + b.acos()
        - a.atan2(&b)
        + b.sinh()
        - a.cosh()
        + b.tanh()
        - a.asinh()
        + b.acosh() / a.atanh()
        + b.powf(&a);
    assert!((r.re - 0.1924131450685842).abs() < 1e-14);
    assert!((r.im + 39.83285329561913).abs() < 1e-13);
}

#[test]
fn complex_tangents_large_arguments() {
    for x in [400.0, 1000.0, f64::MAX, f64::INFINITY] {
        for sign in [-1.0, 1.0] {
            for y in [0.0, 0.3, -2.0, f64::MAX] {
                assert_eq!(Complex::new(sign * x, y).tanh(), Complex::new(sign, 0.0));
                assert_eq!(Complex::new(y, sign * x).tan(), Complex::new(0.0, sign));
            }
        }
    }

    // Doubling the trigonometric argument must not overflow either.
    let t = Complex::new(f64::MAX, 0.0).tan();
    assert!((t.re - f64::MAX.tan()).abs() < 1e-15);
    assert_eq!(t.im, 0.0);
    let t = Complex::new(0.0, f64::MAX).tanh();
    assert_eq!(t.re, 0.0);
    assert!((t.im - f64::MAX.tan()).abs() < 1e-15);
}

#[test]
fn complex_tangents_against_high_precision() {
    // The original double-angle formula is a useful independent reference at
    // high precision, where these arguments neither overflow nor cancel.
    for x in [0.0, 1e-200, 1e-12, 0.1, 1.0, 20.0, 350.0, 370.0, 400.0] {
        for y in [0.0, 1e-200, 0.3, std::f64::consts::FRAC_PI_2, 2.0] {
            for sign in [-1.0, 1.0] {
                let x = sign * x;
                let re = Float::with_val(256, x);
                let im = Float::with_val(256, y);
                let two_re = re.clone() + &re;
                let two_im = im.clone() + &im;
                let denominator = two_re.cosh() + two_im.cos();
                let expected = Complex::new(
                    (two_re.sinh() / &denominator).to_f64(),
                    (two_im.sin() / denominator).to_f64(),
                );
                for (actual, expected) in [
                    (Complex::new(x, y).tanh(), expected),
                    (
                        Complex::new(y, x).tan(),
                        Complex::new(expected.im, expected.re),
                    ),
                ] {
                    for (actual, expected) in [(actual.re, expected.re), (actual.im, expected.im)] {
                        assert!(
                            (actual - expected).abs()
                                <= 2e-14 * expected.abs() + 8.0 * f64::from_bits(1),
                            "x={x}, y={y}: {actual} != {expected}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn complex_tangents_double_float() {
    for x in [400.0, 1000.0, f64::MAX] {
        let z = Complex::new(DoubleFloat::from(x), DoubleFloat::from(0.0));
        assert_eq!(z.tanh(), z.one());
        let z = Complex::new(DoubleFloat::from(0.0), DoubleFloat::from(x));
        assert_eq!(z.tan(), z.i());
    }
}

#[test]
fn complex_tangents_preserve_precision() {
    for x in [0.1, 20.0, 400.0] {
        let re = Float::with_val(256, x);
        let im = Float::with_val(256, 0.3);
        let two_re = re.clone() + &re;
        let two_im = im.clone() + &im;
        let denominator = two_re.cosh() + two_im.cos();
        let expected = Complex::new(two_re.sinh() / &denominator, two_im.sin() / denominator);
        let actual = Complex::new(re, im).tanh();
        for (actual, expected) in [(&actual.re, &expected.re), (&actual.im, &expected.im)] {
            assert!(((actual.clone() - expected) / expected).norm().to_f64() < 1e-70);
        }

        if x < 400.0 {
            let actual = Complex::new(DoubleFloat::from(x), DoubleFloat::from(0.3)).tanh();
            for (actual, expected) in [(actual.re, &expected.re), (actual.im, &expected.im)] {
                let expected = expected.to_double_float();
                assert!(((actual - expected) / expected).norm() < DoubleFloat::from(1e-29));
            }
        }
    }
}

#[test]
fn float_int() {
    let a = Float::with_val(53, 0.123456789123456);
    let b = a / 10i64 * 1300;
    assert_eq!(b.get_precision(), 53);

    let a = Float::with_val(53, 12345.6789);
    let b = a - 12345;
    assert_eq!(b.get_precision(), 40);
}

#[test]
fn large_float_to_integer() {
    let value = Float::parse("123456789123456789123456789123456789", None).unwrap();
    assert_eq!(
        value.round_to_nearest_integer().to_string(),
        "123456789123456789123456789123456789"
    );
}

#[test]
fn high_precision_euler_constant() {
    let value = Float::new(200).euler();
    let formatted = format!("{value:.40e}");
    assert!(
        formatted.starts_with("5.77215664901532860606512090082402431042"),
        "{formatted}"
    );
}

#[test]
fn lower_exp_preserves_precision() {
    let value = Float::new(200).pi();
    let formatted = format!("{value:.55e}");
    assert!(formatted.starts_with("3.14159265358979323846264338327950288419716939937510"));
}

#[test]
fn float_rational() {
    let a = Float::with_val(53, 1000);
    let b: Float = a * Rational::from((-3001, 30)) / Rational::from((1, 2));
    assert_eq!(b.get_precision(), 53);

    let a = Float::with_val(53, 1000);
    let b: Float = a + Rational::new(
        Integer::from_str("-3128903712893789123789213781279").unwrap(),
        Integer::from_str("30890231478123748912372").unwrap(),
    );
    assert_eq!(b.get_precision(), 71);
}

#[test]
fn float_cancellation() {
    let a = Float::with_val(10, 1000);
    let b = a + 10i64;
    assert_eq!(b.get_precision(), 11);

    let a = Float::with_val(53, -1001);
    let b = a + 1000i64;
    assert_eq!(b.get_precision(), 45); // tight bound is 44 digits

    let a = Float::with_val(53, 1000);
    let b = Float::with_val(100, -1001);
    let c = a + b;
    assert_eq!(c.get_precision(), 45); // tight bound is 44 digits

    let a = Float::with_val(20, 1000);
    let b = Float::with_val(40, 1001);
    let c = a + b;
    assert_eq!(c.get_precision(), 22);

    let a = Float::with_val(4, 18.0);
    let b = Float::with_val(24, -17.9199009);
    let c = a + b;
    assert_eq!(c.get_precision(), 1); // capped at 1

    let a = Float::with_val(24, 18.00000);
    let b = Float::with_val(24, -17.992);
    let c = a + b;
    assert_eq!(c.get_precision(), 14);
}

#[test]
fn float_growth() {
    let a = Float::with_val(53, 0.01);
    let b = a.exp();
    assert_eq!(b.get_precision(), 60);

    let a = Float::with_val(53, 0.8);
    let b = a.exp();
    assert_eq!(b.get_precision(), 54);

    let a = Float::with_val(53, 200);
    let b = a.exp();
    assert_eq!(b.get_precision(), 46);

    let a = Float::with_val(53, 0.8);
    let b = a.log();
    assert_eq!(b.get_precision(), 53);

    let a = Float::with_val(53, 300.0);
    let b = a.log();
    assert_eq!(b.get_precision(), 57);

    let a = Float::with_val(53, 1.5709);
    let b = a.sin();
    assert_eq!(b.get_precision(), 53);

    let a = Float::with_val(53, 14.);
    let b = a.tanh();
    assert_eq!(b.get_precision(), 66);

    let a = Float::with_val(53, 1.);
    let b = Float::with_val(53, 0.1);
    let b = a.powf(&b);
    assert_eq!(b.get_precision(), 57);

    let a = Float::with_val(53, 1.);
    let b = Float::with_val(200, 0.1);
    let b = a.powf(&b);
    assert_eq!(b.get_precision(), 57);
}

#[test]
fn powf_prec() {
    let a = Float::with_val(53, 10.);
    let b = Float::with_val(200, 0.1);
    let c = a.powf(&b);
    assert_eq!(c.get_precision(), 57);

    let a = Float::with_val(200, 2.);
    let b = Float::with_val(53, 0.1);
    let c = a.powf(&b);
    assert_eq!(c.get_precision(), 58);

    let a = Float::with_val(53, 3.);
    let b = Float::with_val(200, 20.);
    let c = a.powf(&b);
    assert_eq!(c.get_precision(), 49);

    let a = Float::with_val(200, 1.);
    let b = Float::with_val(53, 0.1);
    let c = a.powf(&b);
    assert_eq!(c.get_precision(), 57); // a=1 is anomalous

    let a = Float::with_val(200, 0.4);
    let b = Float::with_val(53, 0.1);
    let c = a.powf(&b);
    assert_eq!(c.get_precision(), 57);
}

#[cfg(feature = "bincode")]
#[test]
fn bincode_export() {
    let a = Float::with_val(15, 1.127361273);
    let encoded = bincode::encode_to_vec(&a, bincode::config::standard()).unwrap();
    let b: Float = bincode::decode_from_slice(&encoded, bincode::config::standard())
        .unwrap()
        .0;
    assert_eq!(a, b);
}

#[test]
fn complex_gcd() {
    let gcd = Complex::new(Rational::new(3, 2), Rational::new(1, 2))
        .gcd(&Complex::new(Rational::new(1, 1), Rational::new(-1, 1)));
    assert_eq!(gcd, Complex::new((1, 2).into(), (-1, 2).into()));
}
