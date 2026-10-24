use numerica::{
    domains::{
        Ring,
        float::{Complex, ErrorPropagatingFloat, Float, FloatField, Magnitude, RealLike},
    },
    linear_combination::LinearCombination,
};

type Tracked = ErrorPropagatingFloat<Float>;
type Combination = LinearCombination<FloatField<Tracked>, u32>;

fn tracked(precision: u32, value: i64, error: f64) -> Tracked {
    Tracked::new_with_absolute_error(Float::with_val(precision, value), error)
}

fn field(precision: u32) -> FloatField<Tracked> {
    FloatField::from_rep(Tracked::exact_zero(precision))
}

fn assert_uncertain_zero(value: &Tracked, minimum_error: f64) {
    assert_eq!(value.get_num().to_f64(), 0.);
    assert!(!value.is_exact_zero());
    assert!(value.get_error_magnitude() >= Magnitude::from_f64(minimum_error));
}

#[test]
fn bulk_ring_identities_are_exact_and_singletons_retain_metadata() {
    for precision in [128, 256, 512] {
        let ring = field(precision);
        assert!(ring.sum([]).is_exact_zero());
        let product = ring.product([]);
        assert_eq!(product.get_num().to_f64(), 1.);
        assert!(product.is_exact());
        for original in [tracked(precision, 0, 0.125), tracked(precision, 1, 0.125)] {
            for result in [
                ring.sum([original.clone()]),
                ring.product([original.clone()]),
            ] {
                assert_eq!(result.get_num(), original.get_num());
                assert_eq!(result.get_error_magnitude(), original.get_error_magnitude());
            }
        }
        assert!(Combination::sum([], ring.clone()).is_zero());
        assert!(Combination::sum_ref([], ring).is_zero());
    }
}

#[test]
fn construction_and_binary_merges_keep_uncertain_zero_coefficients() {
    for precision in [128, 256, 512] {
        let ring = field(precision);
        let positive = tracked(precision, 3, 0.125);
        let negative = -positive.clone();
        let mut values = Combination::from_terms(
            vec![
                (2, tracked(precision, 0, 0.125)),
                (1, positive.clone()),
                (1, negative.clone()),
                (3, ring.zero()),
            ],
            ring.clone(),
        );
        assert_eq!(values.len(), 2);
        assert_uncertain_zero(values.coefficient(&1).unwrap(), 0.25);
        assert_uncertain_zero(values.coefficient(&2).unwrap(), 0.125);
        values.add_term(4, tracked(precision, 0, 0.125));
        values.add_term(5, positive.clone());
        values.add_term(5, negative);
        assert_uncertain_zero(values.coefficient(&4).unwrap(), 0.125);
        assert_uncertain_zero(values.coefficient(&5).unwrap(), 0.25);

        let canonical = Combination::from_sorted_terms(values.into_terms(), ring.clone());
        assert_eq!(canonical.len(), 4);
        let left = Combination::from_terms(vec![(1, positive)], ring);
        for result in [&left - &left, left.clone() - &left, &left - left.clone()] {
            assert_eq!(result.len(), 1);
            assert_uncertain_zero(result.coefficient(&1).unwrap(), 0.25);
        }
        let mut assigned = left.clone();
        assigned -= left;
        assert_uncertain_zero(assigned.coefficient(&1).unwrap(), 0.25);
    }
}

#[test]
fn n_way_merges_preserve_error_for_identical_and_mixed_supports() {
    for precision in [128, 256, 512] {
        let ring = field(precision);
        let make = |keys: &[u32]| {
            Combination::from_terms(
                keys.iter()
                    .map(|&key| (key, tracked(precision, 0, 0.125)))
                    .collect(),
                ring.clone(),
            )
        };
        for inputs in [
            vec![make(&[0, 2]), make(&[0, 2]), make(&[0, 2])],
            vec![make(&[0, 2]), make(&[0, 1, 2]), make(&[1, 3])],
            vec![make(&[0]), make(&[1]), make(&[2])],
        ] {
            for result in [
                Combination::sum_ref(&inputs, ring.clone()),
                Combination::sum(inputs.clone(), ring.clone()),
            ] {
                for key in 0..4 {
                    let contributors = inputs
                        .iter()
                        .filter(|input| input.coefficient(&key).is_some())
                        .count();
                    if contributors != 0 {
                        assert_uncertain_zero(
                            result.coefficient(&key).unwrap(),
                            contributors as f64 * 0.125,
                        );
                    } else {
                        assert!(result.coefficient(&key).is_none());
                    }
                }
            }
        }
    }
}

#[test]
fn scaling_only_skips_exact_zero_and_exact_one() {
    for precision in [128, 256, 512] {
        let ring = field(precision);
        let original = tracked(precision, 3, 0.125);
        let values = Combination::from_terms(vec![(1, original.clone())], ring.clone());
        let uncertain_one = tracked(precision, 1, 0.125);
        let uncertain_zero = tracked(precision, 0, 0.125);
        for result in [values.scale(&uncertain_one), &values / &uncertain_one] {
            let coefficient = result.coefficient(&1).unwrap();
            assert_eq!(coefficient.get_num().to_f64(), 3.);
            assert!(coefficient.get_error_magnitude() > original.get_error_magnitude());
        }
        let zero_product = values.scale(&uncertain_zero);
        assert_eq!(zero_product.len(), 1);
        assert_uncertain_zero(zero_product.coefficient(&1).unwrap(), 0.375);

        let exact_unit_product = values.scale(&ring.one());
        assert_eq!(
            exact_unit_product
                .coefficient(&1)
                .unwrap()
                .get_error_magnitude(),
            original.get_error_magnitude()
        );
        assert!(values.scale(&ring.zero()).is_zero());
    }
}

#[test]
fn fused_updates_and_substitutions_preserve_uncertainty() {
    for precision in [128, 256, 512] {
        let ring = field(precision);
        let left = Combination::from_terms(vec![(1, tracked(precision, 3, 0.125))], ring.clone());
        let right = Combination::from_terms(vec![(1, tracked(precision, -3, 0.125))], ring.clone());
        let unit = tracked(precision, 1, 0.125);
        let sum = left.add_scaled(&right, &unit);
        assert_uncertain_zero(sum.coefficient(&1).unwrap(), 0.625);
        let mut assigned = left.clone();
        assigned.add_scaled_assign(&right, &unit);
        assert_uncertain_zero(assigned.coefficient(&1).unwrap(), 0.625);

        let replacement =
            Combination::from_terms(vec![(2, tracked(precision, 0, 0.125))], ring.clone());
        for result in [
            left.substitute(&1, &replacement),
            left.substitute_all(|_| Some(&replacement)),
        ] {
            assert_eq!(result.len(), 1);
            assert_uncertain_zero(result.coefficient(&2).unwrap(), 0.375);
        }
    }
}

#[test]
fn complex_coefficients_keep_uncertain_imaginary_zero() {
    for precision in [128, 256, 512] {
        let ring = FloatField::from_rep(Complex::new(
            Tracked::exact_zero(precision),
            Tracked::exact_zero(precision),
        ));
        let value = Complex::new(Tracked::exact_zero(precision), tracked(precision, 0, 0.125));
        let values = LinearCombination::from_terms(vec![(1, value)], ring.clone());
        assert_eq!(values.len(), 1);
        let scalar = Complex::new(tracked(precision, 1, 0.), tracked(precision, 0, 0.125));
        assert!(!ring.is_one(&scalar));
        let scaled = values.scale(&scalar);
        let coefficient = scaled.coefficient(&1).unwrap();
        assert_uncertain_zero(&coefficient.im, 0.125);
        assert!(!ring.is_zero(coefficient));
    }
}
