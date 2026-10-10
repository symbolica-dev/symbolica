use std::sync::Arc;

use symbolica::{
    atom::{Atom, AtomCore},
    domains::{
        Field, Ring, RingOps,
        finite_field::{Z2, Zp},
        rational::{Q, Rational},
    },
    poly::{PolyVariable, polynomial::PolynomialRing, univariate::UnivariatePolynomial},
    symbol,
    tensors::matrix::Matrix,
};

fn polynomial(
    coefficients: &[i64],
) -> UnivariatePolynomial<symbolica::domains::rational::RationalField> {
    UnivariatePolynomial::from_coefficients(
        &Q,
        coefficients.iter().map(|x| Rational::from(*x)).collect(),
        Arc::new(PolyVariable::from(symbol!("resultant_linear_sign::x"))),
    )
}

#[test]
fn linear_resultant_uses_the_signed_evaluation_formula() {
    let linear = polynomial(&[5, -3]);
    let root = Rational::from((5, 3));
    for degree in 1..=7 {
        let mut coefficients = (0..=degree).map(|i| i as i64 - 2).collect::<Vec<_>>();
        coefficients[degree] = 2;
        let f = polynomial(&coefficients);
        let forward = Q.pow(&Rational::from(3), degree as u64) * f.evaluate(&root);
        let reverse = Q.pow(&Rational::from(-3), degree as u64) * f.evaluate(&root);
        assert_eq!(f.resultant(&linear), forward, "degree {degree}");
        assert_eq!(f.resultant_brown(&linear), forward);
        assert_eq!(f.resultant_primitive(&linear), forward);
        assert_eq!(f.resultant_euclidean(&linear), forward);
        assert_eq!(linear.resultant(&f), reverse);
        assert_eq!(linear.resultant_brown(&f), reverse);
        assert_eq!(linear.resultant_primitive(&f), reverse);
        assert_eq!(linear.resultant_euclidean(&f), reverse);
    }
}

#[test]
fn resultant_is_multiplicative_across_linear_and_general_paths() {
    let x = polynomial(&[0, 1]);
    let x3 = polynomial(&[0, 0, 0, 1]);
    let g = polynomial(&[-1, 0, 0, 1]);
    let r = x.resultant(&g);
    assert_eq!(r, Rational::from(-1));
    assert_eq!(x3.resultant(&g), Q.pow(&r, 3));

    let f = polynomial(&[2, 1]);
    let h = polynomial(&[-3, 4, 1]);
    let product = &f * &h;
    assert_eq!(product.resultant(&g), f.resultant(&g) * h.resultant(&g));
    assert_eq!(g.resultant(&product), g.resultant(&f) * g.resultant(&h));
}

#[test]
fn linear_sign_is_consistent_in_odd_characteristic() {
    let x = symbol!("resultant_linear_sign::finite_x");
    let variables = Arc::new(vec![x.into()]);
    for prime in [5, 17] {
        let field = Zp::new(prime);
        let convert = |a: Atom| {
            a.try_to_polynomial::<_, u16>(&field, variables.clone())
                .unwrap()
                .to_univariate_from_univariate(0)
        };
        let g = convert(Atom::num(3) * Atom::var(x) + Atom::one());
        let root = field.div(&field.neg(&g.coefficients()[0]), &g.coefficients()[1]);
        for degree in 1..=5 {
            let f = convert(Atom::var(x).pow(degree as u64) + Atom::num(2));
            let reverse = field.mul(
                &field.pow(&g.coefficients()[1], degree as u64),
                &f.evaluate(&root),
            );
            let forward = if degree % 2 == 0 {
                reverse
            } else {
                field.neg(&reverse)
            };
            assert_eq!(f.resultant(&g), forward, "prime {prime}, degree {degree}");
            assert_eq!(f.resultant_brown(&g), forward);
            assert_eq!(f.resultant_primitive(&g), forward);
            assert_eq!(f.resultant_euclidean(&g), forward);
            assert_eq!(g.resultant(&f), reverse);
        }
    }
}

#[test]
fn linear_sign_preserves_characteristic_two_resultants() {
    let x = symbol!("resultant_linear_sign::binary_x");
    let variables = Arc::new(vec![x.into()]);
    let convert = |a: Atom| {
        a.try_to_polynomial::<_, u16>(&Z2, variables.clone())
            .unwrap()
            .to_univariate_from_univariate(0)
    };
    let g = convert(Atom::var(x) + Atom::one());
    for degree in 1..=5 {
        let f = convert(Atom::var(x).pow(degree as u64));
        let with_shared_root = convert(Atom::var(x).pow(degree as u64) + Atom::one());
        assert_eq!(f.resultant(&g), Z2.one());
        assert_eq!(f.resultant_brown(&g), Z2.one());
        assert_eq!(f.resultant_primitive(&g), Z2.one());
        assert_eq!(f.resultant_euclidean(&g), Z2.one());
        assert_eq!(g.resultant(&f), Z2.one());
        assert_eq!(with_shared_root.resultant(&g), Z2.zero());
    }
}

#[test]
fn symbolic_linear_resultants_match_native_sylvester_determinants() {
    let x = symbol!("resultant_linear_sign::x");
    let a = symbol!("resultant_linear_sign::a");
    let b = symbol!("resultant_linear_sign::b");
    let vars = Arc::new(vec![x.into(), a.into(), b.into()]);
    for degree in 1..=5 {
        let f = Atom::var(x).pow(degree as u64) * Atom::num(2)
            + Atom::var(a) * Atom::var(x)
            + Atom::one();
        let g = Atom::var(b) * Atom::var(x) + Atom::num(3);
        let f = f
            .to_polynomial::<_, u16>(&Q, Some(vars.clone()))
            .to_univariate(0);
        let g = g
            .to_polynomial::<_, u16>(&Q, Some(vars.clone()))
            .to_univariate(0);
        let coefficient_ring = PolynomialRing::<_, u16>::new(Q);
        let dimension = degree + 1;
        let mut entries = vec![coefficient_ring.zero(); dimension * dimension];
        // Descending coefficients, one shifted row for f and degree rows for g.
        for (j, coefficient) in f.coefficients().iter().rev().enumerate() {
            entries[j] = coefficient.clone();
        }
        for i in 0..degree {
            for (j, coefficient) in g.coefficients().iter().rev().enumerate() {
                entries[(i + 1) * dimension + i + j] = coefficient.clone();
            }
        }
        let determinant = Matrix::from_linear(
            entries,
            dimension as u32,
            dimension as u32,
            coefficient_ring.clone(),
        )
        .unwrap()
        .det()
        .unwrap();
        assert_eq!(f.resultant(&g), determinant, "degree {degree}");
        let swapped = if degree % 2 == 0 {
            determinant.clone()
        } else {
            -determinant.clone()
        };
        assert_eq!(g.resultant(&f), swapped);
    }
}
