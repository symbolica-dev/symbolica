use super::Series;
use crate::{
    atom::Atom,
    domains::{
        Ring,
        atom::AtomField,
        float::{F64, FloatField},
        rational::Q,
    },
    parse, symbol,
};
use std::sync::{Arc, Mutex};

// Keep the old rational-exponent convolution as an independent indexing oracle.
// In particular, do not derive its indices from the integer strides used by Mul.
pub(super) fn multiply_reference<F: Ring>(left: &Series<F>, right: &Series<F>) -> Series<F> {
    let ramification = left.joint_ramification(right);
    let left_stride = ramification / left.ramification;
    let right_stride = ramification / right.ramification;
    let mut result = Series {
        coefficients: vec![
            left.field.zero();
            left.coefficients.len().max(right.coefficients.len())
        ],
        field: left.field.clone(),
        variable: left.variable.clone(),
        expansion_point: left.expansion_point.clone(),
        shift: left.shift * left_stride as isize + right.shift * right_stride as isize,
        order: left
            .order
            .saturating_mul(left_stride)
            .min(right.order.saturating_mul(right_stride)),
        ramification,
    };
    for (i, a) in left.coefficients.iter().enumerate() {
        if left.field.is_zero(a) {
            continue;
        }
        let left_power = left.get_exponent(i);
        for (j, b) in right.coefficients.iter().enumerate() {
            let power = &left_power + &right.get_exponent(j);
            if !left.field.is_zero(b) {
                let index = result.get_index(power);
                if index < result.order {
                    if index >= result.coefficients.len() {
                        result.coefficients.resize(index * 2, left.field.zero());
                    }
                    left.field
                        .add_mul_assign(&mut result.coefficients[index], a, b);
                }
            }
        }
    }
    result.truncate();
    result
}

fn assert_same_storage<F: Ring>(actual: &Series<F>, expected: &Series<F>) {
    // Series::eq does not check all precision metadata.
    assert_eq!(actual.coefficients, expected.coefficients);
    assert_eq!(actual.shift, expected.shift);
    assert_eq!(actual.order, expected.order);
    assert_eq!(actual.ramification, expected.ramification);
    assert_eq!(actual.expansion_point, expected.expansion_point);
    assert_eq!(actual.variable, expected.variable);
}

#[test]
fn multiplication_matches_rational_grid_oracle() {
    let variable = Arc::new(super::PolyVariable::from(symbol!(
        "series_mul_grid_oracle::x"
    )));
    for left_ramification in [1, 2, 3, 4, 6] {
        for right_ramification in [1, 2, 3, 4, 6] {
            for seed in 0i64..40 {
                let mut left = Series {
                    coefficients: (0..seed % 8)
                        .map(|i| ((seed + 3 * i) % 7 - 3).into())
                        .collect(),
                    variable: variable.clone(),
                    expansion_point: 2.into(),
                    field: Q,
                    shift: (seed % 9 - 5) as isize,
                    order: (seed % 8 + seed % 4) as usize,
                    ramification: left_ramification,
                };
                let mut right = Series {
                    coefficients: (0..(seed * 3) % 7)
                        .map(|i| ((2 * seed + i) % 5 - 2).into())
                        .collect(),
                    variable: variable.clone(),
                    expansion_point: 2.into(),
                    field: Q,
                    shift: (4 - seed % 8) as isize,
                    order: ((seed * 3) % 7 + (seed + 1) % 4) as usize,
                    ramification: right_ramification,
                };
                left.truncate();
                right.truncate();
                assert_same_storage(&(&left * &right), &multiply_reference(&left, &right));
            }
        }
    }
}

#[test]
fn multiplication_preserves_cancellation_zero_tails_and_infinite_identity() {
    let prototype = Series::new(
        &Q,
        None,
        Arc::new(symbol!("series_mul_zero_oracle::x").into()),
        0.into(),
        5.into(),
    );
    let left = prototype.one() + prototype.monomial(1.into(), (1, 2).into());
    let right = prototype.one() + prototype.monomial((-1).into(), (1, 2).into());
    let product = &left * &right;
    assert_eq!(product.coefficient(0.into()), Some(1.into()));
    assert_eq!(product.coefficient((1, 2).into()), Some(0.into()));
    assert_eq!(product.coefficient(1.into()), Some((-1).into()));
    let cancelled = &left - &left;
    for a in [prototype.zero(), left, right, cancelled] {
        for b in [prototype.zero(), prototype.one_inf_prec()] {
            assert_same_storage(&(&a * &b), &multiply_reference(&a, &b));
            assert_same_storage(&(&b * &a), &multiply_reference(&b, &a));
        }
    }
    for denominator in [2, 3, 5] {
        let fractional = prototype.monomial(7.into(), (-1, denominator).into());
        let identity = prototype.one_inf_prec();
        assert_same_storage(&(&fractional * &identity), &fractional);
        assert_same_storage(&(&identity * &fractional), &fractional);
    }
}

#[test]
fn multiplication_preserves_floating_accumulation_order() {
    let left = Series {
        coefficients: [1e16, 1.0, -1e16, 1.0000000000000002, 1.0]
            .into_iter()
            .map(F64)
            .collect(),
        field: FloatField::<F64>::new(),
        variable: Arc::new(symbol!("series_mul_float_oracle::x").into()),
        expansion_point: F64(0.0),
        shift: -3,
        order: 8,
        ramification: 2,
    };
    let mut right = left.clone();
    right.coefficients = [1.0, 1.0, 1.0, -0.9999999999999999, -1.0]
        .into_iter()
        .map(F64)
        .collect();
    for ramification in [1, 2, 3] {
        right.ramification = ramification;
        assert_same_storage(&(&left * &right), &multiply_reference(&left, &right));
    }
}

#[test]
fn multiplication_preserves_custom_normalization_calls() {
    let log = Arc::new(Mutex::new(Vec::new()));
    let callback_log = log.clone();
    let field = AtomField {
        statistical_zero_test: false,
        custom_normalization: Some(Box::new(move |input, output| {
            callback_log.lock().unwrap().push(input.to_owned());
            *output = input.expand();
            true
        })),
        ..AtomField::new()
    };
    let left = Series {
        coefficients: vec![parse!("a+1"), Atom::Zero, parse!("a-1"), parse!("2*a")],
        field,
        variable: Arc::new(symbol!("series_mul_normalization_oracle::x").into()),
        expansion_point: Atom::Zero,
        shift: -2,
        order: 4,
        ramification: 2,
    };
    let mut right = left.clone();
    right.coefficients = vec![parse!("a-1"), parse!("1-a"), Atom::num(3)];
    right.ramification = 3;
    right.order = 3;
    let expected = multiply_reference(&left, &right);
    let expected_calls = std::mem::take(&mut *log.lock().unwrap());
    let actual = &left * &right;
    assert_same_storage(&actual, &expected);
    assert_eq!(*log.lock().unwrap(), expected_calls);
}
