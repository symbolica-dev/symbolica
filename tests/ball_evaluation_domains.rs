use symbolica::{
    atom::{AtomCore, EvaluationInfo},
    domains::{
        float::{Complex, ComplexBall, Float, RealBall},
        rational::Rational,
    },
    evaluate::EvaluationDomain,
    parse,
};

#[test]
fn polynomial_evaluators_enclose_exact_rational_values() {
    let expression = parse!("x^3/3+2*x/7+5/11");
    let exact = expression.evaluator(&[parse!("x")]).build().unwrap();
    let rational = Rational::from((8, 3)) + Rational::from((4, 7)) + Rational::from((5, 11));
    let reference = RealBall::from_rational_ball(&rational, &Rational::from(0), 160);
    let mut real = exact.clone().map_coeff_with_prec(
        &|c| RealBall::from_rational_ball(&c.re, &Rational::from(0), 80),
        80,
    );
    let value = real.evaluate_single(&[RealBall::exact(Float::with_val(80, 2))]);
    assert!(value.contains_ball(&reference));

    let mut complex = exact.map_coeff_with_prec(
        &|c| ComplexBall::from_rational_ball(c, &Rational::from(0), 80),
        80,
    );
    let value = complex.evaluate_single(&[ComplexBall::new(
        RealBall::exact(Float::with_val(80, 2)),
        RealBall::exact(Float::with_val(80, 0)),
    )]);
    assert!(value.re.contains_ball(&reference));
    assert!(value.im.contains(&Float::with_val(80, 0)));
}

#[test]
fn rounded_constants_and_float_callback_fallbacks_are_not_admitted() {
    let rounded = Complex::new(Float::with_val(80, 1), Float::with_val(80, 0));
    assert!(RealBall::try_from_complex_float(rounded.clone()).is_err());
    assert!(ComplexBall::try_from_complex_float(rounded).is_err());
    let info = EvaluationInfo::new().register(|x: &[Complex<Float>]| x[0].clone());
    assert!(RealBall::resolve_function(&[], &info).is_none());
    assert!(ComplexBall::resolve_function(&[], &info).is_none());
    assert!(RealBall::FIXED_PRECISION.is_none());
    assert!(ComplexBall::FIXED_PRECISION.is_none());
}

#[test]
fn complex_polynomial_encloses_rational_causal_ray_value() {
    let mut evaluator = parse!("1-5*x*(1-x)")
        .evaluator(&[parse!("x")])
        .build()
        .unwrap()
        .map_coeff_with_prec(
            &|c| ComplexBall::from_rational_ball(c, &Rational::from(0), 80),
            80,
        );
    let point = ComplexBall::from_rational_ball(
        &Complex::new(Rational::from((1, 3)), Rational::from((1, 7))),
        &Rational::from(0),
        80,
    );
    let value = evaluator.evaluate_single(&[point]);
    let reference = ComplexBall::from_rational_ball(
        &Complex::new(Rational::from((-94, 441)), Rational::from((-5, 21))),
        &Rational::from(0),
        160,
    );
    assert!(value.contains_ball(&reference));
    assert!(value.im.is_strictly_negative());
}
