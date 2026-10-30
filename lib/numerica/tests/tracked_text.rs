use numerica::domains::{
    SelfRing,
    float::{Complex, ErrorPropagatingFloat, Float, Magnitude},
};
use numerica::printer::{PrintOptions, PrintState};

#[test]
fn lossless_scalar_and_complex_text_preserves_metadata() {
    for center in [
        0.,
        -0.,
        1.25,
        -1.25,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ] {
        for error in [
            Magnitude::ZERO,
            Magnitude::from_f64(1.).scale_pow2(-5000),
            Magnitude::INFINITY,
            Magnitude::NAN,
        ] {
            let value = ErrorPropagatingFloat::new_with_error_magnitude(
                Float::with_val(137, center),
                error,
            );
            let text = value.to_lossless_string();
            let restored = ErrorPropagatingFloat::parse(&text, None).unwrap();
            assert_eq!(restored.get_num().serialize(), value.get_num().serialize());
            assert_eq!(restored.get_num().prec(), 137);
            assert_eq!(
                restored.get_error_magnitude().to_le_bytes(),
                error.to_le_bytes()
            );
            assert!(ErrorPropagatingFloat::parse(&text, Some(53)).is_err());
            let complex = Complex::new(value, ErrorPropagatingFloat::exact_zero(83));
            let mut text = String::new();
            complex
                .format(&PrintOptions::file(), PrintState::new(), &mut text)
                .unwrap();
            let restored = Complex::<ErrorPropagatingFloat<Float>>::parse_lossless(&text).unwrap();
            assert_eq!(
                restored.re.get_num().serialize(),
                complex.re.get_num().serialize()
            );
            assert_eq!(
                restored.re.get_error_magnitude().to_le_bytes(),
                error.to_le_bytes()
            );
            assert_eq!(restored.im.get_num().prec(), 83);
            assert!(restored.im.is_exact_zero());
        }
    }
}
