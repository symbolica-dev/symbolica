//! Align locally factored modular answers to a known rational denominator
//! template, then reconstruct zero-padded coefficients over Q.
//!
//! This example assumes that the rational block x^2+1 and the required pole
//! powers are known. It does not infer rational factor grouping from unrelated
//! modular factors. The block splits modulo 13 but stays irreducible modulo 7;
//! both images nevertheless use the same seven coordinates after alignment.
//!
//! The answer is 7 + (3/2 - 5*x/3)/(x^2+1) + 13/(x^2+1)^2. Some coordinates
//! vanish at the sampling primes: the polynomial part at 7, and the entire
//! highest pole block at 13. Their prescribed slots must not disappear.
//!
//! Run with `cargo run --release --example partial_fraction_reconstruction`.

use std::sync::Arc;

use symbolica::{
    domains::{
        Field, Ring, Set,
        finite_field::{FiniteFieldCore, ToFiniteField, Zp64},
        integer::Integer,
        partial_fraction::{PartialFractionAlignment, PartialFractionLayout, PartialFractionRing},
        rational::{Q, Rational, RationalField},
    },
    poly::{PolyVariable, univariate::UnivariatePolynomial},
    symbol,
};

fn rational_mod(value: &Rational, field: &Zp64) -> Option<<Zp64 as Set>::Element> {
    let denominator = value.denominator_ref().to_finite_field(field);
    (!field.is_zero(&denominator))
        .then(|| field.div(&value.numerator_ref().to_finite_field(field), &denominator))
}

/// Stand in for a modular computation that is free to factor its denominator
/// locally. Only the denominator template, not the answer coordinates, is
/// mapped from Q. The modular input is evaluated directly from integer data.
fn modular_coordinates(
    prime: u64,
    reference: &PartialFractionLayout<RationalField>,
    variable: &Arc<PolyVariable>,
) -> Vec<Integer> {
    let field = Zp64::new(prime);
    let layout = reference
        .map_field(field.clone(), |c| rational_mod(c, &field))
        .expect("the reference denominator has a valid image");
    let polynomial = |coefficients: &[i64]| {
        UnivariatePolynomial::from_coefficients(
            &field,
            coefficients.iter().map(|&c| field.nth(c.into())).collect(),
            variable.clone(),
        )
    };
    let local = PartialFractionRing::new(field.clone(), variable.clone(), &[]);
    // (42*x^4 - 10*x^3 + 93*x^2 - 10*x + 129) / (6*(x^2+1)^2).
    let answer = local
        .try_from_fraction(
            &polynomial(&[129, -10, 93, -10, 42]),
            &polynomial(&[6, 0, 12, 0, 6]),
        )
        .unwrap();
    assert_eq!(local.factor_count(), if prime % 4 == 1 { 2 } else { 1 });

    let alignment = PartialFractionAlignment::new(&local, layout.ring()).unwrap();
    let aligned = alignment.convert(&answer).unwrap();
    let coordinates = layout.export(&aligned).unwrap();
    assert_eq!(coordinates.len(), reference.len());
    assert_eq!(layout.import(&coordinates).unwrap(), aligned);
    if prime == 7 {
        assert!(answer.polynomial_part().is_empty());
        assert!(field.is_zero(&coordinates[0]));
    }
    if prime == 13 {
        assert!(answer.denominator_exponents().all(|(_, e)| e == 1));
        assert!(coordinates[5..].iter().all(|c| field.is_zero(c)));
    }
    println!(
        "p={prime}: {} local factor(s) -> {} reference block, {} coordinates",
        local.factor_count(),
        layout.ring().factor_count(),
        coordinates.len()
    );
    coordinates.iter().map(|c| field.to_integer(c)).collect()
}

fn main() {
    let variable: Arc<PolyVariable> = Arc::new(symbol!("reconstruction_x").into());
    let polynomial = |coefficients: &[i64]| {
        UnivariatePolynomial::from_coefficients(
            &Q,
            coefficients.iter().map(|&c| c.into()).collect(),
            variable.clone(),
        )
    };
    let reference_ring =
        PartialFractionRing::with_fixed_basis(Q, variable.clone(), &[polynomial(&[1, 0, 1])])
            .unwrap();
    // Three polynomial slots, followed by two two-coefficient pole blocks.
    let reference = PartialFractionLayout::new(&reference_ring, 3, &[(0, 2)]).unwrap();
    assert_eq!(reference.len(), 7);

    // Deliberately mix splitting patterns. The accumulated modulus is large
    // enough for the default maximal-quotient acceptance criterion; do not
    // weaken that criterion just to accept an answer from too few primes.
    let primes = [13_u64, 7, 17, 11, 19, 29, 31, 37];
    let mut modulus = Integer::one();
    let mut residues = vec![Integer::zero(); reference.len()];
    for (sample, prime) in primes.into_iter().enumerate() {
        let coordinates = modular_coordinates(prime, &reference, &variable);
        let next_modulus = Integer::from(prime);
        if sample == 0 {
            residues = coordinates;
        } else {
            for (accumulated, coordinate) in residues.iter_mut().zip(coordinates) {
                *accumulated = Integer::chinese_remainder(
                    accumulated.clone(),
                    coordinate,
                    modulus.clone(),
                    next_modulus.clone(),
                );
            }
        }
        modulus *= &next_modulus;
    }
    let reconstructed: Vec<Rational> = residues
        .iter()
        .map(|v| {
            Rational::maximal_quotient_reconstruction(v, &modulus, None)
                .expect("enough samples for these small rational coefficients")
        })
        .collect();
    let lifted = reference.import(&reconstructed).unwrap();

    // Independent modular verification at a prime unused in the CRT.
    let holdout = 43_u64;
    let expected_modular = modular_coordinates(holdout, &reference, &variable);
    let field = Zp64::new(holdout);
    let reconstructed_modular: Vec<_> = reconstructed
        .iter()
        .map(|c| field.to_integer(&rational_mod(c, &field).unwrap()))
        .collect();
    assert_eq!(reconstructed_modular, expected_modular);

    // This synthetic example also permits an exact rational check. In a real
    // calculation the rational answer is the output, not an available oracle.
    let expected = reference_ring
        .try_from_fraction(
            &polynomial(&[129, -10, 93, -10, 42]),
            &polynomial(&[6, 0, 12, 0, 6]),
        )
        .unwrap();
    assert_eq!(lifted, expected);
    assert_eq!(
        reconstructed,
        vec![
            Rational::from(7),
            0.into(),
            0.into(),
            (3, 2).into(),
            (-5, 3).into(),
            13.into(),
            0.into()
        ]
    );
    println!("CRT modulus: {modulus}");
    println!("Reconstructed coordinates: {reconstructed:?}");
    println!("Independent prime {holdout} and exact rational verification passed.");
}
