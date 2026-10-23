use super::*;
use crate::domains::float::{ErrorQuality, Real};
use crate::{
    atom::{Atom, EvaluationInfo},
    function, parse, symbol,
};

#[test]
fn exact_rationals_and_source_cancellation_preserve_different_error_models() {
    let expression = parse!("x-1");
    let input =
        TrackedComplex::from_source_decimal("1.0000000000000000000000000000000000000001", 50, 256)
            .unwrap();
    let mut map = HashMap::new();
    map.insert(parse!("x"), input);
    let result = expression.evaluate_tracked(&map, 256).unwrap();
    let precision = result.precision();
    assert_eq!(result.error_quality, ErrorQuality::Estimated);
    assert!(precision.real.significant_digits().unwrap() <= 10);
    assert!(precision.real.significant_digits().unwrap() >= 8);
    assert!(precision.imaginary.exact_zero());
    let exact = parse!("x-x").evaluate_tracked(&map, 256).unwrap();
    assert!(exact.precision().real.exact_zero());
    assert_eq!(exact.error_quality, ErrorQuality::Bounded);
    let rational = parse!("1/3")
        .evaluate_tracked(&HashMap::<Atom, TrackedComplex>::new(), 80)
        .unwrap();
    assert!(
        rational
            .ball
            .re
            .contains(&Rational::from((1, 3)).to_multi_prec_float(160))
    );
}

#[test]
fn float_atom_structural_zeros_differ_from_uncertain_input_zeros() {
    let inputs = HashMap::<Atom, TrackedComplex>::new();
    let value = parse!("2.5`20").evaluate_tracked(&inputs, 256).unwrap();
    assert!(value.precision().imaginary.exact_zero());
    assert!(value.to_float_with_precision().is_ok());
    assert!(value.precision().real.significant_bits().unwrap() < 80);
    let mut inputs = HashMap::new();
    inputs.insert(
        parse!("x"),
        TrackedComplex::from_source_decimal("0", 50, 256).unwrap(),
    );
    let value = parse!("x").evaluate_tracked(&inputs, 256).unwrap();
    assert!(!value.precision().real.exact_zero());
    assert!(value.to_float_with_precision().is_err());
}

#[test]
fn fallible_tracked_callbacks_receive_tags_and_working_precision_once() {
    let scale = symbol!(
        "tracked_tests::scale",
        eval = EvaluationInfo::new()
            .with_tags(1)
            .register_tracked(|tags, args, bits| {
                assert_eq!(tags.len(), 1);
                if args.len() != 1 {
                    return Err("one numerical argument required".into());
                }
                let factor = Complex::<Rational>::try_from(tags[0]).map_err(|e| e.to_string())?;
                Ok(args[0].mul(&TrackedComplex::from_exact_rational(&factor, bits)))
            })
    );
    let value = function!(scale, 3, parse!("x"));
    let mut inputs = HashMap::new();
    inputs.insert(
        parse!("x"),
        TrackedComplex::from_source_decimal("2.5", 20, 200).unwrap(),
    );
    let result = value.evaluate_tracked(&inputs, 200).unwrap();
    assert!(result.ball.re.contains(&Float::with_val(200, 7.5)));
    assert!(function!(scale, 3).evaluate_tracked(&inputs, 200).is_err());
    let ordinary_only = symbol!(
        "tracked_tests::ordinary",
        eval = EvaluationInfo::constant(|_, bits| {
            Ok(Complex::new(Float::with_val(bits, 1), Float::new(bits)))
        })
    );
    assert!(
        Atom::var(ordinary_only)
            .evaluate_tracked(&inputs, 200)
            .is_err()
    );
}

#[test]
fn tracked_hpl_retains_component_balls_and_tag_validation() {
    let input = crate::transcendental::sixth_root_hpl(&[0], 1).unwrap();
    let result = input
        .evaluate_tracked(&HashMap::<Atom, TrackedComplex>::new(), 160)
        .unwrap();
    assert!(result.precision().real.exact_zero());
    assert!(result.ball.im.contains(&(Float::with_val(200, 1).pi() / 3)));
    assert_eq!(result.error_quality, ErrorQuality::Bounded);
    let malformed = function!(crate::transcendental::hpl_at_sixth_root(), parse!("x"), 1);
    assert!(
        malformed
            .evaluate_tracked(&HashMap::<Atom, TrackedComplex>::new(), 160)
            .is_err()
    );
}

#[test]
fn zero_containing_divisors_are_reported_without_panicking() {
    let mut inputs = HashMap::new();
    inputs.insert(
        parse!("x"),
        TrackedComplex::from_source_decimal("0", 40, 160).unwrap(),
    );
    assert!(parse!("1/x").evaluate_tracked(&inputs, 160).is_err());
}
