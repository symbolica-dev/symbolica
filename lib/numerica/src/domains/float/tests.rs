use rug::Complete;

use super::{Complex, DoubleFloat, ErrorPropagatingFloat, Float, FloatLike, Rational, Real};

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
    assert_eq!(format!("{r}"), "1.000000000");
    assert_eq!(r.get_precision(), Some(10.205999132796238));
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
    assert_eq!(r, Complex::new(0.1924131450685842, -39.83285329561913));
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
fn float_rational() {
    let a = Float::with_val(53, 1000);
    let b: Float = a * Rational::from((-3001, 30)) / Rational::from((1, 2));
    assert_eq!(b.get_precision(), 53);

    let a = Float::with_val(53, 1000);
    let b: Float = a + Rational::from(
        rug::Rational::parse("-3128903712893789123789213781279/30890231478123748912372")
            .unwrap()
            .complete(),
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
