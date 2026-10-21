use super::*;
use crate::{
    atom::{Atom, AtomCore, AtomView},
    domains::{
        float::{ErrorQuality, FloatLike, Real},
        rational::Rational,
    },
    parse,
    transcendental::{GplLetter, endpoint_one_gpl, polylog, sixth_root_hpl},
};
use std::collections::HashMap;

#[test]
fn projections_preserve_selected_component_precision_and_error() {
    let input = Complex::new(Float::with_val(240, 3), Float::with_val(72, -4));
    let tracked = TrackedComplex::from_ball(
        Complex::new(
            RealBall::from_rational_ball(&Rational::from(3), &Rational::from((1, 1000)), 240),
            RealBall::from_rational_ball(&Rational::from(-4), &Rational::from((1, 100)), 72),
        ),
        ErrorQuality::Estimated,
    );
    for (symbol, component, ball) in [
        (real_part(), &input.re, &tracked.ball.re),
        (imaginary_part(), &input.im, &tracked.ball.im),
    ] {
        assert!(symbol.is_real());
        let info = symbol.get_evaluation_info().unwrap();
        let ordinary = info.get_evaluator::<Complex<Float>>(&[]).unwrap()(&[input.clone()]);
        assert_eq!(ordinary.re, *component);
        assert_eq!(ordinary.re.prec(), component.prec());
        assert!(ordinary.im.is_fully_zero());
        assert_eq!(ordinary.im.prec(), component.prec());
        let result = info.evaluate_tracked(&[], &[tracked.clone()], 512).unwrap();
        assert_eq!(result.ball.re, *ball);
        assert!(result.precision().imaginary.exact_zero());
        assert_eq!(result.error_quality, ErrorQuality::Estimated);
    }
}

#[test]
fn real_inputs_and_machine_precision_projections() {
    let input = Float::with_val(300, 7);
    for (symbol, expected) in [(real_part(), 7), (imaginary_part(), 0)] {
        let info = symbol.get_evaluation_info().unwrap();
        let result = info.get_evaluator::<Float>(&[]).unwrap()(&[input.clone()]);
        assert_eq!(result, Float::with_val(300, expected));
        assert_eq!(result.prec(), 300);
        let result = info.get_evaluator::<f64>(&[]).unwrap()(&[7.0]);
        assert_eq!(result, f64::from(expected));
    }
    let input = Complex::new(2.0, -5.0);
    for (symbol, expected) in [(real_part(), 2.0), (imaginary_part(), -5.0)] {
        let result = symbol
            .get_evaluation_info()
            .unwrap()
            .get_evaluator::<Complex<f64>>(&[])
            .unwrap()(&[input]);
        assert_eq!(result, Complex::new(expected, 0.0));
    }
}

#[test]
fn projections_of_hpl_gpl_and_polylog_are_native_and_real() {
    let bits = 200;
    let expressions = [
        sixth_root_hpl(&[0, 1], 1).unwrap(),
        endpoint_one_gpl(&[
            GplLetter::Zero,
            GplLetter::Zero,
            GplLetter::R4,
            GplLetter::One,
        ])
        .unwrap(),
        polylog().call((4, Atom::num((1, 2)))),
    ];
    for expression in expressions {
        let ordinary = expression
            .evaluate_with_prec(&HashMap::<Atom, Complex<Float>>::new(), bits)
            .unwrap();
        let tracked = expression
            .evaluate_tracked(&HashMap::<Atom, TrackedComplex>::new(), bits)
            .unwrap();
        for (symbol, component, ball) in [
            (real_part(), &ordinary.re, &tracked.ball.re),
            (imaginary_part(), &ordinary.im, &tracked.ball.im),
        ] {
            let projected = symbol.call(expression.clone());
            assert!(matches!(projected.as_view(), AtomView::Fun(_)));
            let value = projected
                .evaluate_with_prec(&HashMap::<Atom, Complex<Float>>::new(), bits)
                .unwrap();
            assert_eq!(value.re, *component);
            assert_eq!(value.re.prec(), component.prec());
            assert!(value.im.is_fully_zero());
            let value = projected
                .evaluate_tracked(&HashMap::<Atom, TrackedComplex>::new(), bits)
                .unwrap();
            assert_eq!(value.ball.re, *ball);
            assert!(value.precision().imaginary.exact_zero());
            assert_eq!(value.error_quality, ErrorQuality::Bounded);
        }
    }
    // Re-parsing exported formulas resolves these builtin names in any caller namespace.
    let expression = parse!("imaginary_part(hpl_at_sixth_root(hpl_indices(0),1))");
    let value = expression
        .evaluate_tracked(&HashMap::<Atom, TrackedComplex>::new(), bits)
        .unwrap();
    assert!(
        value
            .ball
            .re
            .contains(&(Float::with_val(bits * 2, 1).pi() / 3))
    );
    assert!(value.precision().imaginary.exact_zero());
}

#[test]
fn malformed_projections_are_fallible_without_panics() {
    let one = TrackedComplex::from_exact_rational(
        &Complex::new(Rational::from(1), Rational::from(0)),
        128,
    );
    for symbol in [real_part(), imaginary_part()] {
        let info = symbol.get_evaluation_info().unwrap();
        assert!(info.evaluate_tracked(&[], &[], 128).is_err());
        assert!(
            info.evaluate_tracked(&[], &[one.clone(), one.clone()], 128)
                .is_err()
        );
        assert!(
            info.evaluate_tracked(&[Atom::num(1).as_view()], &[one.clone()], 128)
                .is_err()
        );
        assert!(!info.get_evaluator::<Float>(&[]).unwrap()(&[]).is_finite());
        assert!(
            !info.get_evaluator::<Complex<Float>>(&[]).unwrap()(&[])
                .re
                .is_finite()
        );
        assert!(info.get_evaluator::<f64>(&[]).unwrap()(&[]).is_nan());
        assert!(
            info.get_evaluator::<Complex<f64>>(&[]).unwrap()(&[])
                .re
                .is_nan()
        );
    }
}
