use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::{Complex, ErrorPropagatingFloat, Float, Magnitude},
    parse,
    printer::PrintOptions,
    try_parse,
};

fn assert_roundtrip(atom: &Atom) {
    let text = atom.printer(PrintOptions::file()).to_string();
    let restored = try_parse!(&text).unwrap_or_else(|error| panic!("{text}: {error}"));
    assert_eq!(restored, *atom, "{text}");
    assert_eq!(restored.printer(PrintOptions::file()).to_string(), text);
}

#[test]
fn file_text_preserves_both_float_components_exactly() {
    for precision in [2, 7, 53, 64, 65, 256, 4096] {
        for center in ["1.25", "-1.125", "1e4000", "1e-4000"] {
            for error in [
                Magnitude::ZERO,
                Magnitude::from_f64(0.125),
                Magnitude::from_f64(1.).scale_pow2(-50_000),
                Magnitude::from_f64(1.).scale_pow2(50_000),
                Magnitude::INFINITY,
                Magnitude::NAN,
            ] {
                let real = ErrorPropagatingFloat::new_with_error_magnitude(
                    Float::parse(center, Some(precision)).unwrap(),
                    error,
                );
                for imaginary in [
                    ErrorPropagatingFloat::exact_zero(precision),
                    ErrorPropagatingFloat::exact_zero(precision + 19),
                    ErrorPropagatingFloat::new_with_error_magnitude(
                        Float::new(precision + 19),
                        error,
                    ),
                    ErrorPropagatingFloat::new_with_error_magnitude(
                        Float::with_val(precision + 19, -1.5),
                        error,
                    ),
                ] {
                    assert_roundtrip(&Atom::num(Complex::new(real.clone(), imaginary)));
                }
            }
        }
    }
}

#[test]
fn file_text_preserves_special_values_and_expression_contexts() {
    for center in [0., -0., f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        let coefficient = Atom::num(Complex::new(
            ErrorPropagatingFloat::new_with_absolute_error(Float::with_val(137, center), 0.),
            ErrorPropagatingFloat::exact_zero(83),
        ));
        assert_roundtrip(&coefficient);
        assert_roundtrip(&(&coefficient * parse!("text_roundtrip_x^(-2)")));
        assert_roundtrip(&(parse!("text_roundtrip_x") + &coefficient));
        assert_roundtrip(
            &parse!("text_roundtrip_f(x)")
                .replace(parse!("x"))
                .with(coefficient.clone()),
        );
    }
    let exact_float_unit = Atom::num(Float::with_val(24, 1)).pow(0);
    assert_roundtrip(&exact_float_unit);
    let uncertain_zero = parse!("0.``-4000");
    assert_roundtrip(&uncertain_zero);
    let value = parse!("-1.25`20");
    let text = value.printer(PrintOptions::file()).to_string();
    assert_eq!(try_parse!(&format!("-({text})")).unwrap(), -value);
}

#[test]
fn malformed_lossless_float_literals_are_rejected() {
    for text in [
        "0f[53:0000000000000000:010280",
        "0f[0:0000000000000000:00]",
        "0f[53:0000000000000000:06]",
        "0f[53:0100000000000000:00]",
        "0f[53:0000:00]",
        "0f[53:0000000000000000:0]",
        "0f[53:0000000000000000:]",
        "0f[53:0000000000000000:00:00]",
        "0f[53:0000000000000000:00;53:0000000000000000:00;53:0000000000000000:00]",
        "0f[53:0000000000000000:é0]",
    ] {
        assert!(try_parse!(text).is_err(), "accepted {text}");
    }
}
