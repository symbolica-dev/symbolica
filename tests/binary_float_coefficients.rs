use std::{
    collections::HashMap,
    hash::{DefaultHasher, Hash, Hasher},
};

use symbolica::{
    atom::{Atom, AtomCore, AtomView},
    coefficient::{Coefficient, CoefficientView},
    domains::float::{Complex, ErrorPropagatingFloat, Float, FloatLike, Magnitude, SingleFloat},
    parse,
    state::StateMap,
};

fn float_coefficient(atom: &Atom) -> Complex<Float> {
    let AtomView::Num(number) = atom.as_view() else {
        panic!("expected a numeric atom");
    };
    let Coefficient::Float(value) = number.get_coeff_view().to_owned() else {
        panic!("expected a floating coefficient");
    };
    Complex::new(value.re.get_num().clone(), value.im.get_num().clone())
}

fn hash(atom: &Atom) -> u64 {
    let mut hasher = DefaultHasher::new();
    atom.hash(&mut hasher);
    hasher.finish()
}

fn roundtrips(atom: &Atom) -> Vec<Atom> {
    let mut stateless = Vec::new();
    atom.as_view().write(&mut stateless).unwrap();
    let mut values =
        vec![Atom::import_with_map(&mut stateless.as_slice(), &StateMap::default()).unwrap()];

    let mut exported = Vec::new();
    atom.export(&mut exported).unwrap();
    values.push(Atom::import(&mut exported.as_slice(), None).unwrap());

    #[cfg(feature = "bincode")]
    {
        let config = bincode::config::standard();
        let encoded = bincode::encode_to_vec(atom, config).unwrap();
        let (decoded, read): (Atom, usize) =
            bincode::decode_from_slice_with_context(&encoded, config, StateMap::default()).unwrap();
        assert_eq!(read, encoded.len());
        values.push(decoded);
    }

    for value in &values {
        assert_eq!(value.as_view().get_data(), atom.as_view().get_data());
        assert_eq!(value, atom);
        assert_eq!(hash(value), hash(atom));
    }
    values
}

#[test]
fn compact_real_coefficients_omit_zero_value_and_preserve_both_precisions() {
    for precision in [8, 53, 64, 65, 256, 4096] {
        let same_precision = Atom::num(Complex::new(
            Float::with_val(precision, 1.25),
            Float::with_val(precision, 0),
        ));
        let full_complex = Atom::num(Complex::new(
            Float::with_val(precision, 1.25),
            Float::with_val(precision, 1.25),
        ));
        // A nonzero imaginary component stores four bytes of precision, eight
        // bytes of error magnitude and three value bytes. An exact zero is omitted.
        assert_eq!(
            full_complex.as_view().get_data().len(),
            same_precision.as_view().get_data().len() + 15
        );

        for imaginary_precision in [precision, precision + 17] {
            let atom = Atom::num(Complex::new(
                Float::with_val(precision, 1.25),
                Float::with_val(imaginary_precision, -0.),
            ));
            assert_eq!(
                atom.as_view().get_data().len(),
                same_precision.as_view().get_data().len()
                    + if imaginary_precision == precision {
                        0
                    } else {
                        4
                    }
            );
            let AtomView::Num(number) = atom.as_view() else {
                unreachable!();
            };
            let CoefficientView::Float(real, imaginary) = number.get_coeff_view() else {
                unreachable!();
            };
            assert_eq!(real.0.len(), 15);
            assert_eq!(imaginary.0, imaginary_precision.to_le_bytes());
            assert!(imaginary.is_zero());
            assert!(imaginary.is_finite());

            for decoded in roundtrips(&atom) {
                let value = float_coefficient(&decoded);
                assert_eq!(value.re, Float::with_val(precision, 1.25));
                assert_eq!(value.re.prec(), precision);
                assert!(value.im.is_zero());
                assert!(!value.im.is_sign_negative());
                assert_eq!(value.im.prec(), imaginary_precision);
                assert_eq!(Atom::num(value), atom);
            }
        }
    }
}

#[test]
fn binary_coefficients_keep_nonzero_imaginary_values_outside_f64_range() {
    for precision in [53, 65, 256] {
        for invert in [false, true] {
            let mut imaginary = Float::with_val(precision + 17, 2).pow(4000);
            if invert {
                imaginary = imaginary.inv();
            }
            let atom = Atom::num(Complex::new(
                Float::with_val(precision, 1.25),
                imaginary.clone(),
            ));
            for decoded in roundtrips(&atom) {
                let value = float_coefficient(&decoded);
                assert_eq!(value.re.prec(), precision);
                assert_eq!(value.im, imaginary);
                assert_eq!(value.im.prec(), precision + 17);
                assert!(!value.im.is_zero());
                assert!(
                    decoded
                        .evaluate_with_prec(&HashMap::<Atom, Float>::new(), precision + 17)
                        .is_err()
                );
            }
        }
    }
}

#[test]
fn canonical_imports_keep_uncertain_imaginary_zero_payloads() {
    for imaginary_precision in [128, 257] {
        for error in [
            Magnitude::MIN_POSITIVE,
            Magnitude::ONE.scale_pow2(-4000),
            Magnitude::ONE.scale_pow2(4000),
            Magnitude::INFINITY,
            Magnitude::NAN,
        ] {
            let atom = Atom::num(Complex::new(
                ErrorPropagatingFloat::new_with_error_magnitude(
                    Float::with_val(128, 1.25),
                    Magnitude::ZERO,
                ),
                ErrorPropagatingFloat::new_with_error_magnitude(
                    Float::new(imaginary_precision),
                    error,
                ),
            ));
            for decoded in roundtrips(&atom) {
                let AtomView::Num(number) = decoded.as_view() else {
                    unreachable!()
                };
                let CoefficientView::Float(_, imaginary) = number.get_coeff_view() else {
                    unreachable!()
                };
                assert_eq!(imaginary.0.len(), 13);
                assert!(imaginary.is_zero() && imaginary.has_error());
                assert!(!imaginary.is_exact_zero());
                let imaginary = imaginary.to_float();
                assert_eq!(imaginary.get_num().prec(), imaginary_precision);
                assert_eq!(
                    imaginary.get_error_magnitude().to_le_bytes(),
                    error.to_le_bytes()
                );
                assert!(
                    decoded
                        .evaluate_with_prec(
                            &HashMap::<Atom, ErrorPropagatingFloat<Float>>::new(),
                            512,
                        )
                        .is_err()
                );
            }
        }
    }
}

#[test]
fn binary_coefficient_specials_preserve_canonical_class_and_precision() {
    for precision in [7, 53, 256] {
        for value in [0., -0., f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
            let original = Float::with_val(precision, value);
            // Keep another nonzero component so even a floating zero is stored
            // rather than being simplified to the exact zero atom.
            let atom = Atom::num(Complex::new(
                original.clone(),
                Float::with_val(precision + 17, 2),
            ));
            for decoded in roundtrips(&atom) {
                let value = float_coefficient(&decoded);
                assert_eq!(value.re.serialize(), original.serialize());
                assert_eq!(value.re.prec(), precision);
                assert_eq!(value.im, Float::with_val(precision + 17, 2));
                assert_eq!(value.im.prec(), precision + 17);
            }
            if !original.is_zero() {
                let real_atom = Atom::num(Complex::new(
                    original.clone(),
                    Float::with_val(precision + 17, 0),
                ));
                for decoded in roundtrips(&real_atom) {
                    let value = float_coefficient(&decoded);
                    assert_eq!(value.re.serialize(), original.serialize());
                    assert_eq!(value.re.prec(), precision);
                    assert!(value.im.is_zero());
                    assert_eq!(value.im.prec(), precision + 17);
                }
            }
        }
    }
}

#[cfg(feature = "float-mpfr")]
#[test]
fn negative_nan_coefficients_share_the_canonical_nan_encoding() {
    let positive = Float::from_raw(rug::Float::with_val(53, rug::float::Special::Nan));
    let negative = -positive.clone();
    let positive = Atom::num(positive);
    let negative = Atom::num(negative);
    assert_eq!(positive.as_view().get_data(), negative.as_view().get_data());
    assert_eq!(hash(&positive), hash(&negative));
    for decoded in roundtrips(&negative) {
        let value = float_coefficient(&decoded);
        assert_eq!(value.re.serialize(), [5]);
        assert!(!value.re.is_sign_negative());
    }
}

#[test]
fn borrowed_coefficient_ordering_matches_decoded_components() {
    let mut atoms = Vec::new();
    for precision in [53, 256] {
        for real in [f64::NEG_INFINITY, -1.25, 1.25, f64::INFINITY, f64::NAN] {
            for imaginary in [-0.5, 0., 0.5, f64::NAN] {
                atoms.push(Atom::num(Complex::new(
                    Float::with_val(precision, real),
                    Float::with_val(precision + 17, imaginary),
                )));
            }
        }
        for error in [Magnitude::ZERO, Magnitude::ONE, Magnitude::INFINITY] {
            atoms.push(Atom::num(Complex::new(
                ErrorPropagatingFloat::new_with_error_magnitude(
                    Float::with_val(precision, 1.25),
                    error,
                ),
                ErrorPropagatingFloat::new_with_error_magnitude(Float::new(precision + 17), error),
            )));
        }
    }
    for left in &atoms {
        for right in &atoms {
            let left_value = float_coefficient(left);
            let right_value = float_coefficient(right);
            let expected = left_value
                .re
                .partial_cmp(&right_value.re)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    left_value
                        .im
                        .partial_cmp(&right_value.im)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            let AtomView::Num(left) = left.as_view() else {
                unreachable!()
            };
            let AtomView::Num(right) = right.as_view() else {
                unreachable!()
            };
            assert_eq!(left.get_coeff_view().cmp(&right.get_coeff_view()), expected);
        }
    }
}

#[test]
fn imported_compact_real_coefficients_evaluate_as_real_values() {
    for precision in [53, 128, 256] {
        for imaginary_precision in [precision / 2, precision, precision * 2] {
            let atom = Atom::num(Complex::new(
                Float::with_val(precision, 1.25),
                Float::with_val(imaginary_precision, 0),
            ));
            for decoded in roundtrips(&atom) {
                let value = decoded
                    .evaluate_with_prec(&HashMap::<Atom, Complex<Float>>::new(), 2 * precision)
                    .unwrap();
                assert_eq!(value.re.prec(), precision);
                assert_eq!(value.im.prec(), imaginary_precision);
                assert!(value.im.is_zero());
                let real = decoded
                    .evaluate_with_prec(&HashMap::<Atom, Float>::new(), 2 * precision)
                    .unwrap();
                assert_eq!(real, value.re);
                assert_eq!(real.prec(), precision);

                let x = parse!("binary_float_storage_x");
                let expression = &decoded * &x + &decoded;
                let result = expression
                    .evaluate_with_prec(
                        &HashMap::from([(x, Float::with_val(2 * precision, 2))]),
                        2 * precision,
                    )
                    .unwrap();
                assert_eq!(result, Float::with_val(precision, 3.75));
            }
        }
    }
}
