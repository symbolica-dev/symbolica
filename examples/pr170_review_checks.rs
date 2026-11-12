//! Check corrected composite-basis behavior and the fixed ring API.
use std::sync::Arc;
use symbolica::{
    domains::{
        Ring,
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
    let ring = PartialFractionRing::with_fixed_basis(Q, x.clone(), &[poly(&[-1, 0, 1])]).unwrap();
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
    let fixed = PartialFractionRing::with_fixed_basis(Q, x.clone(), &[]).unwrap();
    let variable = fixed.element_from_polynomial(&poly(&[0, 1]));
    assert_eq!(fixed.try_div(&variable, &variable), Some(fixed.one()));
    assert!(fixed.try_inv(&variable).is_none());
    println!("Composite-basis imports, exports, removable poles and fixed-ring division passed.");
    // Field/EuclideanDomain are deliberately unavailable on this fixed mode;
    // the module's compile-fail doctest checks the Field boundary.
}
