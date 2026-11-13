//! Check composite-basis behavior and automatic extension of seeded rings.
use std::sync::Arc;
use symbolica::{
    domains::{
        Field, Ring, RingOps,
        integer::Z,
        partial_fraction::PartialFractionRing,
        rational::Q,
        rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial},
    },
    poly::{PolyVariable, univariate::UnivariatePolynomial},
    symbol,
};
fn main() {
    let x: Arc<PolyVariable> = Arc::new(symbol!("pr170_x").into());
    let poly = |c: &[i64]| {
        UnivariatePolynomial::from_coefficients(
            &Q,
            c.iter().map(|&c| c.into()).collect(),
            x.clone(),
        )
    };
    let ring = PartialFractionRing::with_basis(Q, x.clone(), &[poly(&[-1, 0, 1])]).unwrap();
    let a = ring
        .try_from_fraction(&poly(&[1, 1]), &poly(&[-1, 0, 1]))
        .unwrap();
    assert_eq!(ring.to_fraction(&a), (poly(&[1]), poly(&[-1, 1])));
    assert_eq!(ring.evaluate(&a, &(-1).into()), Some((-1, 2).into()));
    assert_eq!(
        ring.try_from_fraction(&poly(&[1]), &poly(&[-1, 1])),
        Ok(a.clone())
    );
    let actual: RationalPolynomial<_, u16> = ring.to_rational_polynomial(&a, &Z);
    let expected = RationalPolynomial::from_num_den(
        poly(&[1]).to_multivariate(),
        poly(&[-1, 1]).to_multivariate(),
        &Z,
        true,
    );
    assert_eq!(actual, expected);
    let empty = PartialFractionRing::with_basis(Q, x.clone(), &[]).unwrap();
    let variable = empty.element_from_polynomial(&poly(&[0, 1]));
    assert_eq!(empty.factor_count(), 0);
    let inverse = empty.inv(&variable);
    assert_eq!(empty.factor_count(), 1);
    assert_eq!(empty.mul(&variable, &inverse), empty.one());
    assert_eq!(empty.div(&variable, &variable), empty.one());
    println!("Composite-basis imports, exports, removable poles and automatic extension passed.");
}
