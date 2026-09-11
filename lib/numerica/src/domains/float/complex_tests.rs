use super::{Complex, DoubleFloat, Float, FloatLike, Real, RealLike};

fn close(actual: f64, expected: f64) {
    if actual == expected {
        return;
    }
    assert!(
        (actual - expected).abs() <= 3e-14 * expected.abs() + 4.0 * f64::from_bits(1),
        "{actual:e} != {expected:e}"
    );
}

#[test]
fn reciprocal_hyperbolic_functions_avoid_overflow() {
    for x in [-1e13, 1e13, f64::NEG_INFINITY, f64::INFINITY] {
        for y in [0.0, 0.75] {
            let z = Complex::new(x, y);
            for result in [z.sech(), z.csch()] {
                assert_eq!(result.re, 0.0, "{z}: {result}");
                assert_eq!(result.im, 0.0, "{z}: {result}");
            }
        }
    }

    for x in [0.0, 1e-200, 0.25, 20.0, 400.0, 711.0, 745.0, 745.5] {
        for sign in [-1.0, 1.0] {
            for y in [0.0, 1e-200, 0.75, std::f64::consts::FRAC_PI_2] {
                if x == 0.0 && y == 0.0 {
                    continue;
                }
                let z = Complex::new(sign * x, y);
                let reference =
                    Complex::new(Float::with_val(256, sign * x), Float::with_val(256, y));
                for (actual, expected) in [
                    (z.sech(), reference.cosh().inv().to_f64()),
                    (z.csch(), reference.sinh().inv().to_f64()),
                ] {
                    close(actual.re, expected.re);
                    close(actual.im, expected.im);
                    if y == 0.0 && expected.re != 0.0 {
                        assert_ne!(actual.re, 0.0, "lost representable tail at {z}");
                    }
                }
                if y == 0.0 {
                    close(z.re.sech(), reference.re.cosh().inv().to_f64());
                    close(z.re.csch(), reference.re.sinh().inv().to_f64());
                }
            }
        }
    }
}

#[test]
fn complex_functions_against_independent_reference() {
    // Generated with mpmath at 800 decimal digits from the exact f64 inputs.
    // Each row contains sqrt, log, asinh, acosh, asin, acos and atanh.
    let cases = [
        (
            (1e+200, 1e+200),
            [
                (1.09868411346781e+100, 4.550898605622274e+99),
                (460.8635921890891, 0.7853981633974483),
                (461.55673936964905, 0.7853981633974483),
                (461.55673936964905, 0.7853981633974483),
                (0.7853981633974483, 461.55673936964905),
                (0.7853981633974483, -461.55673936964905),
                (5e-201, 1.5707963267948966),
            ],
        ),
        (
            (-1e+200, 1e-200),
            [
                (5e-301, 1e+100),
                (460.51701859880916, 3.141592653589793),
                (-461.2101657793691, 0.0),
                (461.2101657793691, 3.141592653589793),
                (-1.5707963267948966, 461.2101657793691),
                (3.141592653589793, -461.2101657793691),
                (-1e-200, 1.5707963267948966),
            ],
        ),
        (
            (1e-200, -1e-200),
            [
                (1.09868411346781e-100, -4.550898605622273e-101),
                (-460.17044500852916, -0.7853981633974483),
                (1e-200, -1e-200),
                (1e-200, -1.5707963267948966),
                (1e-200, -1e-200),
                (1.5707963267948966, 1e-200),
                (1e-200, -1e-200),
            ],
        ),
        (
            (-10000000000.0, 0.25),
            [
                (1.25e-06, 100000.0),
                (23.025850929940457, 3.141592653564793),
                (-23.7189981105004, 2.5e-11),
                (23.7189981105004, 3.141592653564793),
                (-1.5707963267698966, 23.7189981105004),
                (3.141592653564793, -23.7189981105004),
                (-1e-10, 1.5707963267948966),
            ],
        ),
        (
            (1.0, 1e-200),
            [
                (1.0, 5e-201),
                (0.0, 1e-200),
                (0.881373587019543, 7.071067811865475e-201),
                (1e-100, 1e-100),
                (1.5707963267948966, 1e-100),
                (1e-100, -1e-100),
                (230.60508288968455, 0.7853981633974483),
            ],
        ),
        (
            (1.0000000000000002, -1e-20),
            [
                (1.0, -4.999999999999999e-21),
                (2.2204460492503128e-16, -9.999999999999998e-21),
                (0.8813735870195432, -7.071067811865474e-21),
                (2.1073424260789764e-08, -4.745313280009497e-13),
                (1.570796326794422, -2.1073424260789764e-08),
                (4.745313280009497e-13, 2.1073424260789764e-08),
                (18.36840028433149, -1.570773808796775),
            ],
        ),
        (
            (0.9999999999999999, 1e-20),
            [
                (0.9999999999999999, 5e-21),
                (-1.1102230246251565e-16, 1.0000000000000001e-20),
                (0.8813735870195429, 7.071067811865475e-21),
                (6.710886393194353e-13, 1.490116120895923e-08),
                (1.5707963118937354, 6.710886393194353e-13),
                (1.490116120895923e-08, -6.710886393194353e-13),
                (18.714973873090283, 4.503599615191316e-05),
            ],
        ),
        (
            (0.0, 10000000000.0),
            [
                (70710.67811865476, 70710.67811865476),
                (23.025850929940457, 1.5707963267948966),
                (23.7189981105004, 1.5707963267948966),
                (23.7189981105004, 1.5707963267948966),
                (0.0, 23.7189981105004),
                (1.5707963267948966, -23.7189981105004),
                (0.0, 1.5707963266948965),
            ],
        ),
        (
            (2.0, -3.0),
            [
                (1.6741492280355401, -0.8959774761298381),
                (1.2824746787307684, -0.982793723247329),
                (1.9686379257930964, -0.9646585044076028),
                (1.9833870299165355, -1.0001435424737972),
                (0.5706527843210994, -1.9833870299165355),
                (1.0001435424737972, 1.9833870299165355),
                (0.14694666622552977, -1.3389725222944935),
            ],
        ),
        (
            (1.7976931348623157e+308, 1.7976931348623157e+308),
            [
                (1.4730945569055652e+154, 6.1017574412827024e+153),
                (710.1292864836639, 0.7853981633974483),
                (710.8224336642239, 0.7853981633974483),
                (710.8224336642239, 0.7853981633974483),
                (0.7853981633974483, 710.8224336642239),
                (0.7853981633974483, -710.8224336642239),
                (2.781342323134e-309, 1.5707963267948966),
            ],
        ),
    ];
    for ((x, y), expected) in cases {
        let z = Complex::new(x, y);
        let actual = [
            z.sqrt(),
            z.log(),
            z.asinh(),
            z.acosh(),
            z.asin(),
            z.acos(),
            z.atanh(),
        ];
        for (a, (re, im)) in actual.into_iter().zip(expected) {
            close(a.re, re);
            close(a.im, im);
        }
    }
}

#[test]
fn complex_scaling_extremes() {
    for x in [1e-310, 1e-200, 1e-160, 1e160, 1e200, 1e308] {
        let z = Complex::new(x, x);
        close(z.norm().re, x.hypot(x));
        let inverse = z.inv();
        close(inverse.re, 0.5 / x);
        close(inverse.im, -0.5 / x);
        let quotient = z / z;
        close(quotient.re, 1.0);
        close(quotient.im, 0.0);
        let quotient = z / z.conj();
        close(quotient.re, 0.0);
        close(quotient.im, 1.0);
        let (r, _) = z.to_polar_coordinates();
        close(r, x.hypot(x));
    }
    let z = Complex::new(1e-310, 0.0).inv();
    assert_eq!(z.re, f64::INFINITY);
    assert_eq!(z.im, 0.0);
    let z = Complex::new(0.0, 1e-310).inv();
    assert_eq!(z.re, 0.0);
    assert_eq!(z.im, f64::NEG_INFINITY);
    let a = Complex::new(f64::MAX, f64::MAX);
    let q = a / Complex::new(2.0, 2.0);
    close(q.re, f64::MAX / 2.0);
    close(q.im, 0.0);
}

#[test]
fn complex_small_components_and_branch_cuts() {
    for y in [0.0, -0.0, 1e-200, -1e-200, f64::from_bits(1)] {
        let z = Complex::new(1.0, y).sqrt();
        close(z.re, 1.0);
        close(z.im, y / 2.0);
        assert_eq!(z.im.is_sign_negative(), y.is_sign_negative());
        let z = Complex::new(-1.0, y).sqrt();
        close(z.re, y.abs() / 2.0);
        close(z.im, 1.0f64.copysign(y));
        let z = Complex::new(0.0, y).atanh();
        close(z.re, 0.0);
        close(z.im, y.atan());
    }
    close(Complex::new(1.0, 1e-10).log().re, 0.5 * 1e-20);
    for y in [0.0, -0.0] {
        let z = Complex::new(2.0, y).asin();
        close(z.re, std::f64::consts::FRAC_PI_2);
        close(z.im, 2.0f64.acosh().copysign(y));
        let z = Complex::new(-2.0, y).acosh();
        close(z.re, 2.0f64.acosh());
        close(z.im, std::f64::consts::PI.copysign(y));
        let z = Complex::new(1.0, y).atanh();
        assert_eq!(z.re, f64::INFINITY);
        assert_eq!(z.im.is_sign_negative(), y.is_sign_negative());
    }
}

#[test]
fn complex_overflow_preserves_finite_components() {
    for x in [1000.0, -1000.0] {
        let z = Complex::new(x, 0.0);
        assert_eq!(z.sinh().re, f64::INFINITY.copysign(x));
        assert_eq!(z.sinh().im, 0.0);
        assert_eq!(z.cosh().re, f64::INFINITY);
        assert_eq!(z.cosh().im, 0.0);
        let z = Complex::new(0.0, x);
        assert_eq!(z.sin().re, 0.0);
        assert_eq!(z.sin().im, f64::INFINITY.copysign(x));
        assert_eq!(z.cos().re, f64::INFINITY);
        assert_eq!(z.cos().im, 0.0);
    }
    let z = Complex::new(1000.0, 0.0).exp();
    assert_eq!(z.re, f64::INFINITY);
    assert_eq!(z.im, 0.0);
    // exp(720) overflows, but multiplying by the small sine is representable.
    let z = Complex::new(720.0, 1e-300);
    close(z.exp().im, 4.920700930263816e12);
    close(z.sinh().im, 2.460350465131908e12);
    close(z.cosh().im, 2.460350465131908e12);
    // A subnormal sinh is small, not an overflow that needs asymptotics.
    close(Complex::new(1e-310, 0.0).sinh().re, 1e-310);
}

fn scalar_backend<T: Real + RealLike>(x: T) {
    let z = Complex::new(x.clone(), x.zero());
    let s = z.sqrt();
    close(s.re.to_f64(), x.sqrt().to_f64());
    close(s.im.to_f64(), 0.0);
    close(z.log().re.to_f64(), x.log().to_f64());
    close((-z.clone()).asinh().re.to_f64(), -x.asinh().to_f64());
    close(z.acosh().re.to_f64(), x.acosh().to_f64());
    close(z.inv().re.to_f64(), x.inv().to_f64());
    close((z.clone() / z).re.to_f64(), 1.0);
}

#[test]
fn complex_scaling_preserves_scalar_backends() {
    scalar_backend(DoubleFloat::from(1e200));
    scalar_backend(Float::with_val(256, 1e200));
    scalar_backend(super::F64::from(1e200));
}

#[test]
fn complex_real_powers_scale_before_exponentiation() {
    let z = Complex::new(f64::MAX, f64::MAX);
    let s = z.powf(&Complex::new(0.5, 0.0));
    let expected = z.sqrt();
    close(s.re, expected.re);
    close(s.im, expected.im);
    let z = Complex::new(1e200, 0.0).powf(&Complex::new(2.0, 0.0));
    assert_eq!(z.re, f64::INFINITY);
    assert_eq!(z.im, 0.0);
}

#[test]
fn complex_inverse_functions_keep_extended_precision() {
    // Independent 80-digit reference values, evaluated at the exact f64 inputs.
    let cases = [
        (
            (0.125, 0.25),
            [
                (
                    "0.1286736648967109027457639517884827681274605535076511265877616119084130739092286030496",
                    "0.2505579941523818434155073929326317976680702180248792271058283047163720541027058297549",
                ),
                (
                    "0.2492677327284542842264112525233259759707409451756519262516598140081944759657629962133",
                    "1.449282851953965583096233725524840383809147408643340871711083325123246695160674479491",
                ),
                (
                    "0.1215134748409310361350879661149110582894372910442120387763889710306615079824300198225",
                    "0.2492677327284542842264112525233259759707409451756519262516598140081944759657629962133",
                ),
                (
                    "1.449282851953965583096233725524840383809147408643340871711083325123246695160674479491",
                    "-0.2492677327284542842264112525233259759707409451756519262516598140081944759657629962133",
                ),
                (
                    "0.1180898357345486551764562030175641011884566534857612265404390398346167323912747967519",
                    "0.2484843024395266566852020102413041743138317857402436335612804398211206214871638241422",
                ),
            ],
        ),
        (
            (-1e+20, 0.1),
            [
                (
                    "-46.74484904044085898977706121514546072009755540693571477478729353999607707334530624485",
                    "0.000000000000000000001000000000000000055511151231257827021181533071207682291663835597953872517485504921852",
                ),
                (
                    "46.744849040440858989777061215145460720097505406935714774787293539996077073345306245",
                    "3.141592653589793238461643383279502884141658248143847993953762959236608723994539611915",
                ),
                (
                    "-1.570796326794896619230321691639751442043073548456295083466290663082700520851435112601",
                    "46.744849040440858989777061215145460720097505406935714774787293539996077073345306245",
                ),
                (
                    "3.141592653589793238461643383279502884141658248143847993953762959236608723994539611915",
                    "-46.744849040440858989777061215145460720097505406935714774787293539996077073345306245",
                ),
                (
                    "-0.00000000000000000001000000000000000000000000000000000000000032333333333333333222311030870817676209484057",
                    "1.57079632679489661923132169163975144209857469968755291048691718464159562487289268249",
                ),
            ],
        ),
        (
            (1.0, 1e-100),
            [
                (
                    "0.8813735870195430252326093249797923090281603282616354107532956086533771842220260878337",
                    "7.07106781186547538537252281327348913136889440870743051639392652586369168903552301039e-101",
                ),
                (
                    "1.000000000000000009995949901301441759864373179602928079872321077594288477752423706834e-50",
                    "1.000000000000000009995949901301441759864373179602928079872321077594288477752423706834e-50",
                ),
                (
                    "1.570796326794896619231321691639751442098584699687542910487472296153808243644091484896",
                    "1.000000000000000009995949901301441759864373179602928079872321077594288477752423706834e-50",
                ),
                (
                    "1.000000000000000009995949901301441759864373179602928079872321077594288477752423706834e-50",
                    "-1.000000000000000009995949901301441759864373179602928079872321077594288477752423706834e-50",
                ),
                (
                    "115.4758282399822568456122388936458569541879585336799996461449069265778546762066791181",
                    "0.785398163397448309615660845819875721049292349843776455243736148076954101571552249657",
                ),
            ],
        ),
    ];
    for ((x, y), expected) in cases {
        let z = Complex::new(Float::with_val(256, x), Float::with_val(256, y));
        let actual = [z.asinh(), z.acosh(), z.asin(), z.acos(), z.atanh()];
        for (index, (actual, (re, im))) in actual.into_iter().zip(expected).enumerate() {
            for (a, e) in [(actual.re, re), (actual.im, im)] {
                let e = Float::parse(e, Some(256)).unwrap();
                let err = ((a.clone() - &e) / &e).norm().to_f64();
                assert!(
                    err < 2e-65,
                    "function {index} ({x},{y}): {a:e} != {e:e}, precision {}, relative error {err}",
                    a.get_precision()
                );
            }
        }
        let z = Complex::new(DoubleFloat::from(x), DoubleFloat::from(y));
        let actual = [z.asinh(), z.acosh(), z.asin(), z.acos(), z.atanh()];
        for (index, (actual, (re, im))) in actual.into_iter().zip(expected).enumerate() {
            for (a, e) in [(actual.re, re), (actual.im, im)] {
                let e = Float::parse(e, Some(256)).unwrap().to_double_float();
                let err = ((a - e) / e).norm();
                assert!(
                    err < DoubleFloat::from(2e-28),
                    "function {index} ({x},{y}): {a:e} != {e:e}, precision {}, relative error {err}",
                    a.get_precision()
                );
            }
        }
    }
}

#[test]
fn complex_division_preserves_small_components() {
    let q = Complex::new(1.0, 0.0) / Complex::new(1e-310, 1e-320);
    assert_eq!(q.re, f64::INFINITY);
    close(q.im, -(1e-320 / 1e-310) / 1e-310);
    let q = Complex::new(f64::MAX, 0.0) / Complex::new(2.0, f64::from_bits(1));
    close(q.re, f64::MAX / 2.0);
    close(q.im, -(f64::MAX * f64::from_bits(1)) / 4.0);
}

#[test]
fn exact_zero_does_not_destroy_float_precision() {
    let x = Float::with_val(256, 1.0);
    let zero = x.clone() - &x;
    let tiny = Float::with_val(256, 1e-100);
    assert_eq!(zero.get_precision(), 256);
    let z = Complex::new(zero, tiny).sqrt();
    assert!(z.re.get_precision() >= 250);
    assert!(z.im.get_precision() >= 250);
}

#[test]
fn complex_division_against_wide_exponent_reference() {
    let values = [
        0.0,
        f64::from_bits(1),
        1e-310,
        1e-160,
        1.0,
        1e160,
        1e308,
        f64::MAX,
    ];
    for a in values {
        for b in values {
            for c in values {
                for d in values {
                    if c == 0.0 && d == 0.0 {
                        continue;
                    }
                    let z = Complex::new(a, b);
                    let w = Complex::new(c, -d);
                    let actual = z / w;
                    let numerator = Complex::new(Float::with_val(256, a), Float::with_val(256, b));
                    let denominator =
                        Complex::new(Float::with_val(256, c), Float::with_val(256, -d));
                    let reference = numerator.clone() / &denominator;
                    let expected = reference.to_f64();
                    // The real numerator is a difference of products. Bound
                    // rounding by the magnitudes before cancellation.
                    let real_error = ((numerator.re * &denominator.re
                        - numerator.im * &denominator.im)
                        / denominator.norm_squared()
                        * Float::with_val(256, 3e-14))
                    .to_f64();
                    for (actual, expected, tolerance) in [
                        (actual.re, expected.re, real_error),
                        (actual.im, expected.im, 3e-14 * expected.im.abs()),
                    ] {
                        assert!(
                            actual == expected
                                || (actual.is_finite()
                                    && (actual - expected).abs()
                                        <= tolerance + 4.0 * f64::from_bits(1)),
                            "{z} / {w}: {actual} != {expected}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn complex_simd_magnitude_keeps_lane_semantics() {
    use wide::f64x4;
    let z = Complex::new(f64x4::ZERO, f64x4::from([1.0, 2.0, 3.0, 4.0]));
    assert_eq!(z.norm().re.to_array(), [1.0, 2.0, 3.0, 4.0]);
    let z = Complex::new(
        f64x4::from([3.0, 0.0, 5.0, 8.0]),
        f64x4::from([4.0, 2.0, 12.0, 15.0]),
    );
    assert_eq!(z.norm().re.to_array(), [5.0, 2.0, 13.0, 17.0]);
}

#[test]
fn subnormal_f64_conversion_is_exact() {
    for x in [
        f64::from_bits(1),
        f64::from_bits(2),
        1e-310,
        f64::MIN_POSITIVE / 2.0,
    ] {
        for sign in [-1.0, 1.0] {
            assert_eq!(Float::with_val(256, sign * x).to_f64(), sign * x);
        }
    }
}
