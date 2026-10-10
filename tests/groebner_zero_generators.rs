use std::sync::Arc;
use symbolica::{
    atom::{Atom, AtomCore},
    domains::{finite_field::Zp, rational::Q},
    parse,
    poly::{PolyVariable, groebner::GroebnerBasis, polynomial::MultivariatePolynomial},
    symbol,
};

#[test]
fn rational_ideal_ignores_zero_generators_in_every_position() {
    let variables = Arc::new(vec![
        PolyVariable::from(symbol!("zero_gb::x")),
        PolyVariable::from(symbol!("zero_gb::y")),
    ]);
    let polynomials: Vec<MultivariatePolynomial<_, u16>> = [
        parse!("zero_gb::x^2-zero_gb::y"),
        parse!("zero_gb::x*zero_gb::y-1"),
    ]
    .iter()
    .map(|p| p.to_polynomial(&Q, Some(variables.clone())))
    .collect();
    let zero = Atom::Zero.to_polynomial(&Q, Some(variables));
    let expected = GroebnerBasis::new(&polynomials, false).system;
    for position in 0..=polynomials.len() {
        let mut with_zero = polynomials.clone();
        with_zero.insert(position, zero.clone());
        assert_eq!(GroebnerBasis::new(&with_zero, false).system, expected);
    }
}

#[test]
fn zero_ideal_has_an_empty_basis() {
    let zero: MultivariatePolynomial<_, u16> = Atom::Zero.to_polynomial(&Q, None);
    for ideal in [vec![], vec![zero.clone()], vec![zero.clone(), zero]] {
        assert!(GroebnerBasis::new(&ideal, false).system.is_empty());
    }
}

#[test]
fn zero_generators_preserve_the_unified_variable_map() {
    let x = PolyVariable::from(symbol!("zero_map::x"));
    let y = PolyVariable::from(symbol!("zero_map::y"));
    let zero: MultivariatePolynomial<_, u16> =
        Atom::Zero.to_polynomial(&Q, Some(Arc::new(vec![x.clone()])));
    let p = parse!("zero_map::y+1").to_polynomial(&Q, Some(Arc::new(vec![y.clone()])));
    let ideal = vec![zero, p];
    let mut unified = ideal.clone();
    MultivariatePolynomial::unify_variables_list(&mut unified);
    let expected = unified[1].clone();
    assert_eq!(expected.nvars(), 2);
    assert_eq!(GroebnerBasis::new(&ideal, false).system, vec![expected]);
}

#[test]
fn finite_field_unit_ideal_accepts_zero_generators() {
    let variables = Arc::new(vec![PolyVariable::from(symbol!("zero_ff::x"))]);
    let ideal: Vec<MultivariatePolynomial<_, u16>> =
        [Atom::Zero, parse!("zero_ff::x"), Atom::num(1), Atom::Zero]
            .iter()
            .map(|p| p.to_polynomial(&Zp::new(13), Some(variables.clone())))
            .collect();
    let basis = GroebnerBasis::new(&ideal, false);
    assert_eq!(basis.system.len(), 1);
    assert!(basis.system[0].is_one());
}
