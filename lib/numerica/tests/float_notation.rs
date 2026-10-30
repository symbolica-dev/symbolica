use numerica::{
    domains::{SelfRing, float::Float},
    printer::{PrintOptions, PrintState},
};

#[test]
fn default_display_and_plain_output_follow_mpfr_notation() {
    for (input, expected) in [
        (-4.47213595499957, "-4.47213595499957"),
        (1.0, "1.00000000000000"),
        (12.5, "12.5000000000000"),
        (0.5, "5.00000000000000e-1"),
        (1e20, "1.00000000000000e20"),
        (0.0, "0"),
        (-0.0, "-0"),
    ] {
        let value = Float::from(input);
        assert_eq!(value.to_string(), expected, "{input}");
        let mut plain = String::new();
        value
            .format(&PrintOptions::file(), PrintState::new(), &mut plain)
            .unwrap();
        assert_eq!(plain, expected, "{input}");
    }
}

#[test]
fn display_selects_notation_after_rounding_to_significant_digits() {
    for (input, digits, expected) in [
        ("1.25", 4, "1.250"),
        ("1250", 4, "1250"),
        ("12500", 4, "1.250e4"),
        ("-1250", 4, "-1250"),
        ("-12500", 4, "-1.250e4"),
        ("0.125", 4, "1.250e-1"),
        ("9.96875", 2, "10"),
        ("99.96875", 2, "1.0e2"),
        ("0.999999", 2, "1.0"),
        ("1.25", 2, "1.2"),
        ("1.75", 2, "1.8"),
        ("1e100000", 4, "1.000e100000"),
        ("1e-100000", 4, "1.000e-100000"),
    ] {
        let value = Float::parse(input, Some(128)).unwrap();
        assert_eq!(format!("{value:.digits$}"), expected, "{input}");
    }
}

#[test]
fn explicit_scientific_notation_uses_mpfr_precision_and_exponents() {
    for (input, expected) in [
        ("1.25", "1.25e0"),
        ("12.5", "1.25e1"),
        ("0.125", "1.25e-1"),
        ("-1250", "-1.25e3"),
        ("9999", "1.00e4"),
        ("0", "0"),
        ("-0", "-0"),
        ("inf", "inf"),
        ("-inf", "-inf"),
        ("NaN", "NaN"),
    ] {
        let value = Float::parse(input, Some(128)).unwrap();
        assert_eq!(format!("{value:.3e}"), expected, "{input}");
    }
}

#[test]
fn decimal_display_preserves_signs_padding_and_special_values() {
    let value = Float::parse("12.5", Some(128)).unwrap();
    assert_eq!(format!("{value:+.4}"), "+12.50");
    assert_eq!(format!("{value:>10.4}"), "     12.50");
    assert_eq!(format!("{value:<10.4}"), "12.50     ");
    assert_eq!(format!("{value:+010.4}"), "+000012.50");
    assert_eq!(format!("{value:+.4e}"), "+1.250e1");

    for (input, expected) in [
        ("0", "0"),
        ("-0", "-0"),
        ("inf", "inf"),
        ("-inf", "-inf"),
        ("NaN", "NaN"),
    ] {
        let value = Float::parse(input, Some(128)).unwrap();
        assert_eq!(format!("{value:.4}"), expected, "{input}");
    }
}

#[test]
fn zero_precision_uses_mpfr_automatic_digit_count() {
    for (bits, expected) in [
        (2, "1.0"),
        (3, "1.2"),
        (10, "1.2500"),
        (53, "1.2500000000000000"),
        (128, "1.250000000000000000000000000000000000000"),
    ] {
        let value = Float::with_val(bits, 1.25);
        assert_eq!(format!("{value:.0}"), expected, "{bits} bits");
        assert_eq!(value.as_raw().to_string(), expected, "{bits} bits");
    }
}
