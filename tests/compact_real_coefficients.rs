use std::collections::HashMap;

use symbolica::{
    atom::{Atom, AtomCore},
    domains::float::{Complex, ErrorPropagatingFloat, Float, Magnitude, RealLike},
};

type Tracked = ErrorPropagatingFloat<Float>;

fn imported(atom: &Atom) -> Atom {
    let mut bytes = Vec::new();
    atom.export(&mut bytes).unwrap();
    Atom::import(&mut bytes.as_slice(), None).unwrap()
}

#[test]
fn compact_real_evaluation_retains_both_component_precisions() {
    for precision in [128, 256, 512] {
        for imaginary_precision in [precision / 2, precision, 2 * precision] {
            let atom = Atom::num(Complex::new(
                Tracked::new_with_absolute_error(Float::with_val(precision, 2), 0.125),
                Tracked::exact_zero(imaginary_precision),
            ));
            for atom in [atom.clone(), imported(&atom)] {
                let value = atom
                    .evaluate_with_prec(&HashMap::<Atom, Complex<Tracked>>::new(), 2 * precision)
                    .unwrap();
                assert_eq!(value.re.get_num().prec(), precision);
                assert_eq!(value.im.get_num().prec(), imaginary_precision);
                assert_eq!(value.re.get_num().to_f64(), 2.);
                assert_eq!(value.re.get_error_magnitude(), Magnitude::from_f64(0.125));
                assert!(value.im.is_exact_zero());
                let real = atom
                    .evaluate_with_prec(&HashMap::<Atom, Tracked>::new(), 2 * precision)
                    .unwrap();
                assert_eq!(real.get_num().prec(), precision);
                assert_eq!(real.get_error_magnitude(), value.re.get_error_magnitude());
            }
        }
    }
}

#[test]
fn imported_real_coefficients_keep_real_results_and_tracked_errors() {
    for precision in [128, 256, 512] {
        for imaginary_precision in [precision / 2, precision, 2 * precision] {
            let atom = imported(&Atom::num(Complex::new(
                Tracked::new_with_absolute_error(Float::with_val(precision, 2), 0.001),
                Tracked::exact_zero(imaginary_precision),
            )));
            for result in [
                &atom + 1,
                &atom - 1,
                &atom * 3,
                &atom / 3,
                atom.pow(3),
                atom.sqrt(),
                atom.log(),
                atom.exp(),
                atom.sin(),
                atom.cos(),
            ] {
                let value = result
                    .evaluate_with_prec(&HashMap::<Atom, Complex<Tracked>>::new(), precision)
                    .unwrap();
                assert!(value.im.is_exact_zero(), "{result}");
                assert!(!value.re.is_exact(), "{result}");
                assert!(
                    result
                        .evaluate_with_prec(&HashMap::<Atom, Tracked>::new(), precision)
                        .is_ok()
                );
            }
        }
    }
}

#[test]
fn imaginary_zero_with_error_outside_f64_range_is_not_omitted() {
    for precision in [128, 256, 512] {
        for exponent in [-4000, 4000] {
            let error = Magnitude::ONE.scale_pow2(exponent);
            assert_eq!(error.to_f64(), None);
            let atom = Atom::num(Complex::new(
                Tracked::new_with_error_magnitude(Float::with_val(precision, 1), Magnitude::ZERO),
                Tracked::new_with_error_magnitude(Float::with_val(precision / 2, 0), error),
            ));
            for atom in [atom.clone(), imported(&atom)] {
                let value = atom
                    .evaluate_with_prec(&HashMap::<Atom, Complex<Tracked>>::new(), precision)
                    .unwrap();
                assert_eq!(value.im.get_num().to_f64(), 0.);
                assert_eq!(value.im.get_num().prec(), precision / 2);
                assert_eq!(value.im.get_error_magnitude(), error);
                assert!(!value.im.is_exact_zero());
                assert!(
                    atom.evaluate_with_prec(&HashMap::<Atom, Tracked>::new(), precision)
                        .is_err()
                );
                assert!(
                    atom.evaluate(&HashMap::<Atom, ErrorPropagatingFloat<f64>>::new())
                        .is_err()
                );
                let binary64 = atom
                    .evaluate(&HashMap::<Atom, Complex<ErrorPropagatingFloat<f64>>>::new())
                    .unwrap();
                assert_eq!(*binary64.im.get_num(), 0.);
                assert!(binary64.im.get_error_magnitude() >= error);
                assert!(!binary64.im.is_exact_zero());
            }
        }
    }
}
