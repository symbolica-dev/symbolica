use numerica::domains::float::{
    Complex, ErrorPropagatingFloat, Float, FloatLike, Magnitude, Real, RealLike,
};

type Tracked = ErrorPropagatingFloat<Float>;

fn tracked(precision: u32, center: f64, error: f64) -> Tracked {
    Tracked::new_with_absolute_error(Float::with_val(precision, center), error)
}

fn covers(result: &Tracked, reference: Float, context: &str) {
    assert!(reference.is_finite(), "invalid reference: {context}");
    let difference = (reference - result.get_num()).absolute_magnitude();
    assert!(
        difference <= result.get_error_magnitude(),
        "{context}: difference {difference:?} exceeds {:?}, center {}",
        result.get_error_magnitude(),
        result.get_num()
    );
}

#[test]
fn integer_powers_cover_finite_changes_and_storage_rounding() {
    for precision in [53, 128, 256] {
        for (center, error) in [(1., 0.125), (-2., 0.5), (0., 0.25), (0.25, 0.5)] {
            let input = tracked(precision, center, error);
            for exponent in [0, 1, 2, 3, 8, 31] {
                let result = input.pow(exponent);
                for step in [-1., -0.5, 0., 0.5, 1.] {
                    covers(
                        &result,
                        Float::with_val(512, center + step * error).pow(exponent),
                        &format!("pow({center} ± {error}, {exponent}), step {step}"),
                    );
                }
            }
        }
    }
    let exact_binary = ErrorPropagatingFloat::new_with_absolute_error(1. + 2f64.powi(-27), 0.);
    let squared = exact_binary.pow(2);
    assert!(!squared.is_exact());
    assert!(squared.get_error_magnitude() >= Magnitude::from_f64(2f64.powi(-54)));
    assert!(exact_binary.zero().pow(2).is_exact_zero());
    assert!(exact_binary.one().pow(u64::MAX).is_exact());
    assert_eq!(*(-exact_binary.one()).pow(u64::MAX).get_num(), -1.);
}

#[test]
fn real_transcendentals_cover_stationary_points_and_finite_neighborhoods() {
    type Function = (fn(&Tracked) -> Tracked, fn(&Float) -> Float);
    let functions: &[(&str, Function, &[(f64, f64)])] = &[
        (
            "sin",
            (Tracked::sin, Float::sin),
            &[(0., 0.125), (1.5, 0.125), (2., 2.)],
        ),
        (
            "cos",
            (Tracked::cos, Float::cos),
            &[(0., 0.125), (1., 0.5), (2., 2.)],
        ),
        (
            "exp",
            (Tracked::exp, Float::exp),
            &[(0., 1.), (-20., 0.5), (20., 2.)],
        ),
        (
            "sinh",
            (Tracked::sinh, Float::sinh),
            &[(0., 0.5), (-2., 0.25), (2., 2.)],
        ),
        (
            "cosh",
            (Tracked::cosh, Float::cosh),
            &[(0., 0.5), (-2., 0.25), (2., 2.)],
        ),
        (
            "tanh",
            (Tracked::tanh, Float::tanh),
            &[(0., 0.5), (2., 1.), (10., 2.)],
        ),
        (
            "asinh",
            (Tracked::asinh, Float::asinh),
            &[(0., 0.5), (2., 1.), (3., 4.)],
        ),
        (
            "acosh",
            (Tracked::acosh, Float::acosh),
            &[(1., 0.125), (2., 1.), (10., 2.)],
        ),
        (
            "asin",
            (Tracked::asin, Float::asin),
            &[(0., 0.5), (0.875, 0.125), (1., 0.125)],
        ),
        (
            "acos",
            (Tracked::acos, Float::acos),
            &[(0., 0.5), (-0.875, 0.125), (1., 0.125)],
        ),
        (
            "atanh",
            (Tracked::atanh, Float::atanh),
            &[(0., 0.5), (0.75, 0.125)],
        ),
        ("tan", (Tracked::tan, Float::tan), &[(0., 0.5), (1., 0.125)]),
        (
            "log1p",
            (Tracked::log1p, Float::log1p),
            &[(0., 0.5), (-0.75, 0.125)],
        ),
    ];
    for precision in [53, 128, 256] {
        for &(name, (evaluate, reference), cases) in functions {
            for &(center, error) in cases {
                let result = evaluate(&tracked(precision, center, error));
                for step in [-1., -0.5, 0., 0.5, 1.] {
                    let endpoint = center + step * error;
                    if matches!(name, "asin" | "acos") && endpoint.abs() > 1.
                        || name == "acosh" && endpoint < 1.
                    {
                        // Bounds concern defined real perturbations; crossing a
                        // domain boundary does not make undefined inputs valid.
                        continue;
                    }
                    covers(
                        &result,
                        reference(&Float::with_val(512, endpoint)),
                        &format!("{name}({center} ± {error}), step {step}, precision {precision}"),
                    );
                }
            }
        }
    }
}

#[test]
fn poles_and_angle_branch_cuts_are_not_locally_linearized() {
    for precision in [53, 128, 256] {
        for input in [
            tracked(precision, 0.875, 0.25),
            tracked(precision, -0.875, 0.25),
        ] {
            assert!(input.atanh().get_error_magnitude().is_infinite());
        }
        assert!(
            tracked(precision, 1.5, 0.125)
                .tan()
                .get_error_magnitude()
                .is_infinite()
        );
        assert!(
            tracked(precision, 0., 2.)
                .log1p()
                .get_error_magnitude()
                .is_infinite()
        );
        for (re, im, dr, di) in [
            (-1., 0., 0.125, 0.125),
            (-1., 0.0625, 0., 0.125),
            (0.125, 0., 0.25, 0.),
        ] {
            let z = Complex::new(tracked(precision, re, dr), tracked(precision, im, di));
            let angle = z.arg();
            let logarithm = z.log();
            for sr in [-1., 0., 1.] {
                for si in [-1., 0., 1.] {
                    let point = Complex::new(
                        Float::with_val(512, re + sr * dr),
                        Float::with_val(512, im + si * di),
                    );
                    covers(&angle, point.arg(), "atan2 cut or origin");
                    let reference = point.log();
                    covers(&logarithm.re, reference.re, "complex log real");
                    covers(&logarithm.im, reference.im, "complex log imaginary cut");
                }
            }
        }
        let positive = Complex::new(
            tracked(precision, 2., 0.125),
            Tracked::exact_zero(precision),
        );
        assert!(positive.log().im.is_exact_zero());
    }
}

#[test]
fn powers_with_uncertain_exponents_and_transverse_hypot_cover_endpoints() {
    for precision in [53, 128, 256] {
        for (base, db, exponent, de) in [
            (0.25, 0.125, 0.75, 0.125),
            (2., 0.5, 3., 0.25),
            (1., 0.125, 0., 0.25),
        ] {
            let result = tracked(precision, base, db).powf(&tracked(precision, exponent, de));
            for sb in [-1., 0., 1.] {
                for se in [-1., 0., 1.] {
                    covers(
                        &result,
                        Float::with_val(512, base + sb * db)
                            .powf(&Float::with_val(512, exponent + se * de)),
                        "uncertain powf",
                    );
                }
            }
        }
        let transverse = tracked(precision, 2., 0.).hypot(&tracked(precision, 0., 0.25));
        covers(
            &transverse,
            Float::with_val(512, 2).hypot(&Float::with_val(512, 0.25)),
            "transverse hypot",
        );
        assert!(!transverse.is_exact());
        let sign = tracked(precision, 0., 0.125);
        let signed = tracked(precision, 2., 0.125).copy_sign(&sign);
        covers(&signed, Float::with_val(512, -2.125), "uncertain copy_sign");
    }
}

#[test]
fn stationary_point_bounds_keep_errors_outside_binary64_range() {
    for text in ["1e-1000", "1e1000"] {
        let error = Float::parse(text, Some(256)).unwrap().absolute_magnitude();
        let input = Tracked::new_with_error_magnitude(Float::with_val(256, 0), error);
        for result in [input.cos(), input.cosh()] {
            assert!(!result.get_error_magnitude().is_zero());
            assert!(!result.get_error_magnitude().is_nan());
        }
        let square = input.pow(2);
        assert!(square.get_error_magnitude() >= error * error);
        assert!(!square.is_exact_zero());
    }
}
