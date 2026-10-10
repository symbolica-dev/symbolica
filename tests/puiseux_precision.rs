use std::sync::Arc;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::atom::AtomField,
    parse,
    poly::series::{Series, SeriesDepth},
    symbol,
};

fn seed() -> Series<AtomField> {
    Series::new(
        &AtomField::new(),
        None,
        Arc::new(symbol!("puiseux_precision::x").into()),
        Atom::Zero,
        (7, 3).into(),
    )
}

#[test]
fn fractional_order_is_measured_in_exponents() {
    let s = seed();
    assert_eq!(s.get_ramification(), 3);
    assert_eq!(s.relative_order(), (7, 3));
    assert_eq!(s.absolute_order(), (7, 3));
    for c in [s.zero(), s.one(), s.constant(Atom::num(2))] {
        assert_eq!(c.relative_order(), (7, 3));
        assert_eq!(c.absolute_order(), (7, 3));
        assert_eq!(c.coefficient((7, 3).into()), None);
    }
}

#[test]
fn monomials_align_fractional_exponents_and_precision() {
    for exponent in [(1, 1), (1, 2), (-2, 3), (5, 6)] {
        let s = seed().monomial(Atom::num(3), exponent.into());
        assert_eq!(s.get_trailing_exponent(), exponent);
        assert_eq!(s.relative_order(), (7, 3));
        assert_eq!(s.coefficient(exponent.into()), Some(Atom::num(3)));
        assert_eq!(s.coefficient(s.absolute_order()), None);
    }
}

#[test]
fn shifted_variable_keeps_integer_spacing_on_fractional_grid() {
    let s = seed().shifted_variable(Atom::num(2));
    assert_eq!(s.absolute_order(), (7, 3));
    assert_eq!(s.coefficient(0.into()), Some(Atom::num(2)));
    assert_eq!(s.coefficient(1.into()), Some(Atom::num(1)));
    assert_eq!(s.coefficient((1, 3).into()), Some(Atom::Zero));
    assert_eq!(s.coefficient((7, 3).into()), None);
}

#[test]
fn rational_power_refines_the_valuation_denominator() {
    let s = seed().monomial(Atom::num(1), (1, 3).into());
    let root = s.rpow((1, 3).into()).unwrap();
    assert_eq!(root.get_trailing_exponent(), (1, 9));
    assert_eq!(root.coefficient((1, 9).into()), Some(Atom::num(1)));
    assert_eq!(root.coefficient(0.into()), Some(Atom::Zero));
    assert_eq!(root.coefficient(root.absolute_order()), None);
}

#[test]
fn rational_power_scales_a_pure_remainder() {
    let s = seed().rpow((2, 3).into()).unwrap();
    assert!(s.is_zero());
    assert_eq!(s.absolute_order(), (14, 9));
    assert_eq!(s.coefficient((13, 9).into()), Some(Atom::Zero));
    assert_eq!(s.coefficient((14, 9).into()), None);
}

#[test]
fn rational_power_of_a_fractional_unit_preserves_valuation_and_cutoff() {
    let s = seed();
    let unit = s.one() + s.monomial(Atom::num(1), (1, 2).into());
    let base = &s.monomial(Atom::num(1), (1, 3).into()) * &unit;
    for (power, expected) in [
        (
            (2, 3),
            [
                ((2, 9), (1, 1)),
                ((13, 18), (2, 3)),
                ((11, 9), (-1, 9)),
                ((31, 18), (4, 81)),
            ],
        ),
        (
            (-1, 2),
            [
                ((-1, 6), (1, 1)),
                ((1, 3), (-1, 2)),
                ((5, 6), (3, 8)),
                ((4, 3), (-5, 16)),
            ],
        ),
    ] {
        let result = base.rpow(power.into()).unwrap();
        assert_eq!(result.relative_order(), (7, 3));
        for (exponent, coefficient) in expected {
            assert_eq!(
                result.coefficient(exponent.into()),
                Some(Atom::num(coefficient))
            );
        }
        assert_eq!(result.coefficient(result.absolute_order()), None);
    }
}

#[test]
fn fractional_depth_does_not_rescale_the_expansion_variable() {
    let x = symbol!("puiseux_precision::x");
    let s = parse!("(1+puiseux_precision::x)^(1/2)")
        .series(x, 0, SeriesDepth::absolute((7, 3)))
        .unwrap();
    assert_eq!(s.coefficient(0.into()), Some(Atom::num(1)));
    assert_eq!(s.coefficient(1.into()), Some(Atom::num((1, 2))));
    assert_eq!(s.coefficient(2.into()), Some(Atom::num((-1, 8))));
    assert_eq!(s.coefficient((1, 3).into()), Some(Atom::Zero));
    assert_eq!(s.coefficient(s.absolute_order()), None);
}

#[test]
fn fractional_product_and_laurent_series_have_correct_coefficients() {
    let x = symbol!("puiseux_precision::x");
    let s = parse!("puiseux_precision::x^(1/3)*(1+puiseux_precision::x)^(1/2)")
        .series(x, 0, SeriesDepth::absolute((7, 3)))
        .unwrap();
    for (exponent, coefficient) in [((1, 3), (1, 1)), ((4, 3), (1, 2)), ((7, 3), (-1, 8))] {
        assert_eq!(s.coefficient(exponent.into()), Some(Atom::num(coefficient)));
    }
    let square = (&s * &s).map_coeff(|a| a.expand());
    assert_eq!(square.coefficient((2, 3).into()), Some(Atom::num(1)));
    assert_eq!(square.coefficient((5, 3).into()), Some(Atom::num(1)));
    assert_eq!(square.coefficient((8, 3).into()), Some(Atom::Zero));
    let pole = parse!("puiseux_precision::x^(-1/3)*(1+puiseux_precision::x)")
        .series(x, 0, SeriesDepth::absolute((2, 3)))
        .unwrap();
    assert_eq!(pole.coefficient((-1, 3).into()), Some(Atom::num(1)));
    assert_eq!(pole.coefficient((2, 3).into()), Some(Atom::num(1)));
}

#[test]
fn fractional_depth_at_nonzero_centre_and_integer_control() {
    let x = symbol!("puiseux_precision::x");
    for depth in [SeriesDepth::absolute((7, 3)), SeriesDepth::absolute(3)] {
        let s = parse!("puiseux_precision::x^2")
            .series(x, 2, depth)
            .unwrap();
        assert_eq!(s.coefficient(0.into()), Some(Atom::num(4)));
        assert_eq!(s.coefficient(1.into()), Some(Atom::num(4)));
        assert_eq!(s.coefficient(2.into()), Some(Atom::num(1)));
        assert_eq!(s.coefficient((1, 3).into()), Some(Atom::Zero));
    }
}

#[test]
fn integer_precision_control_is_unchanged() {
    let x = symbol!("puiseux_precision::x");
    let s = parse!("(1+puiseux_precision::x)^(1/2)")
        .series(x, 0, SeriesDepth::absolute(3))
        .unwrap();
    for (power, coefficient) in [(0, (1, 1)), (1, (1, 2)), (2, (-1, 8)), (3, (1, 16))] {
        assert_eq!(s.coefficient(power.into()), Some(Atom::num(coefficient)));
    }
    assert_eq!(s.coefficient(4.into()), None);
}
