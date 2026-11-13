//! Sums of many products of univariate rational functions over Zp whose
//! denominators come from a few fixed factors, in three representations:
//! [PartialFractionRing], [RationalPolynomial] and [FactorizedRationalPolynomial].
//!
//! Run with `cargo bench --features symbolica_benchmarks --bench partial_fraction_ring`.
//!
//! The workload mimics the coefficient arithmetic of a linear-system
//! elimination: `SLOTS` accumulators start as coefficients; every step takes one
//! accumulator as the pivot, adds `pivot * c` to the next `FAN_OUT`
//! accumulators for coefficients `c` from a fixed pool of `COEFFICIENTS`
//! rational functions, and reinitializes the pivot's slot. The coefficients
//! have numerators of degree <= 2 and denominators that are products of one or
//! two basis factors. Fixture construction and validation (all
//! representations give the same results) are outside the timed regions.
//!
//! `partial_fraction_ring_cold` times a ring without hints,
//! including the conversion of the coefficients from fractions (which
//! trial-divides known factors and factors only remaining cofactors) and the workload;
//! `partial_fraction_ring_warm` initializes the same ring and
//! coefficients before timing, then measures only the arithmetic workload;
//! `partial_fraction_ring_warm_reimport` also initializes the ring before
//! timing, but rebuilds and re-imports the coefficient fractions on each
//! iteration, using the populated denominator cache, before the workload.
//!
//! The original mixed fixture is preserved, but its deterministic coefficient
//! pool never selects the quadratic. The extra `irreducible_quad` case forces
//! a quadratic denominator and uses x^2+1, irreducible for this prime.

use std::{hint::black_box, sync::Arc};

use symbolica::{
    domains::{
        Ring,
        factorized_rational_polynomial::{
            FactorizedRationalPolynomial, FactorizedRationalPolynomialField,
            FromNumeratorAndFactorizedDenominator,
        },
        finite_field::{FiniteFieldCore, Zp64},
        partial_fraction::{PartialFraction, PartialFractionRing},
        rational_polynomial::{
            FromNumeratorAndDenominator, RationalPolynomial, RationalPolynomialField,
        },
    },
    poly::{PolyVariable, polynomial::MultivariatePolynomial, univariate::UnivariatePolynomial},
    symbol,
};

const PRIME: u64 = 9_223_372_036_854_775_783; // largest prime below 2^63
const SLOTS: usize = 12;
const FAN_OUT: usize = 3;
const COEFFICIENTS: usize = 24;

#[derive(Clone, Copy, Debug)]
struct Case {
    /// Linear factors `x - 1, ..., x - linear`.
    linear: usize,
    /// Whether the basis also holds the quadratic `x^2 + x + 1`.
    quadratic: bool,
    /// Guarantee a used irreducible quadratic x^2 + 1 (p = 3 mod 4).
    covered_quadratic: bool,
    steps: usize,
}

impl std::fmt::Display for Case {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}lin{}_{}steps",
            self.linear,
            if self.covered_quadratic {
                "+irreducible_quad"
            } else if self.quadratic {
                "+quad"
            } else {
                ""
            },
            self.steps
        )
    }
}

const CASES: &[Case] = &[
    Case {
        linear: 4,
        quadratic: false,
        covered_quadratic: false,
        steps: 60,
    },
    Case {
        linear: 8,
        quadratic: false,
        covered_quadratic: false,
        steps: 120,
    },
    Case {
        linear: 6,
        quadratic: true,
        covered_quadratic: false,
        steps: 60,
    },
    Case {
        linear: 6,
        quadratic: true,
        covered_quadratic: true,
        steps: 60,
    },
];

/// A coefficient: ascending numerator and `(factor index, exponent)` list.
type Coefficient = (Vec<u64>, Vec<(usize, usize)>);

struct Fixture {
    field: Zp64,
    variable: Arc<PolyVariable>,
    factors: Vec<Vec<u64>>,
    coefficients: Vec<Coefficient>,
    steps: usize,
}

fn fixture(case: Case) -> Fixture {
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    let mut next = |bound: u64| {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state % bound
    };
    let mut factors: Vec<Vec<u64>> = (1..=case.linear as u64)
        .map(|r| vec![PRIME - r, 1])
        .collect();
    if case.quadratic {
        factors.push(if case.covered_quadratic {
            vec![1, 0, 1]
        } else {
            vec![1, 1, 1]
        });
    }
    let mut coefficients: Vec<Coefficient> = (0..COEFFICIENTS)
        .map(|_| {
            let numerator = (0..=next(3)).map(|_| 1 + next(20)).collect();
            let first = next(factors.len() as u64) as usize;
            let mut denominator = vec![(first, 1)];
            if next(2) == 0 {
                let second = next(factors.len() as u64) as usize;
                if second == first {
                    denominator[0].1 += 1;
                } else {
                    denominator.push((second, 1));
                }
            }
            (numerator, denominator)
        })
        .collect();
    if case.covered_quadratic {
        // The original random mixed fixture never selects its quadratic.
        // Preserve that fixture, and add this explicit nonlinear coverage.
        coefficients[0].1 = vec![(case.linear, 1)];
    }
    Fixture {
        field: Zp64::new(PRIME),
        variable: Arc::new(symbol!("x").into()),
        factors,
        coefficients,
        steps: case.steps,
    }
}

/// Run the elimination workload with the ring operations of `ring`.
fn run<R: Ring>(ring: &R, coefficients: &[R::Element], steps: usize) -> Vec<R::Element> {
    let mut slots: Vec<R::Element> = coefficients[..SLOTS].to_vec();
    for t in 0..steps {
        let pivot = std::mem::replace(
            &mut slots[t % SLOTS],
            coefficients[(t * 5 + 3) % COEFFICIENTS].clone(),
        );
        for k in 1..=FAN_OUT {
            let c = &coefficients[(t * 7 + k) % COEFFICIENTS];
            ring.add_mul_assign(&mut slots[(t + k) % SLOTS], &pivot, c);
        }
    }
    slots
}

impl Fixture {
    fn univariate(&self, c: &[u64]) -> UnivariatePolynomial<Zp64> {
        UnivariatePolynomial::from_coefficients(
            &self.field,
            c.iter().map(|&x| self.field.to_element(x)).collect(),
            self.variable.clone(),
        )
    }

    fn multivariate(&self, c: &[u64]) -> MultivariatePolynomial<Zp64, u16> {
        self.univariate(c).to_multivariate()
    }

    fn seeded_partial_fractions(&self) -> (PartialFractionRing<Zp64>, Vec<PartialFraction<Zp64>>) {
        let factors: Vec<_> = self.factors.iter().map(|f| self.univariate(f)).collect();
        let ring =
            PartialFractionRing::with_basis(self.field.clone(), self.variable.clone(), &factors)
                .unwrap();
        let one = ring.one();
        let coefficients = self
            .coefficients
            .iter()
            .map(|(n, d)| ring.mul_fraction(&one, &self.univariate(n), d))
            .collect();
        (ring, coefficients)
    }

    /// A ring without hints: every coefficient enters as a
    /// fraction; known factors are trial-divided before remaining cofactors
    /// are factored. Decompositions are cached per denominator.
    fn cold_partial_fractions(&self) -> (PartialFractionRing<Zp64>, Vec<PartialFraction<Zp64>>) {
        let ring = PartialFractionRing::new(self.field.clone(), self.variable.clone(), &[]);
        let coefficients = self.import_partial_fractions(&ring);
        (ring, coefficients)
    }

    /// Build the input fractions and convert them through the ring's public
    /// entry point. With a warmed ring, all denominators are cache hits.
    fn import_partial_fractions(
        &self,
        ring: &PartialFractionRing<Zp64>,
    ) -> Vec<PartialFraction<Zp64>> {
        self.coefficients
            .iter()
            .map(|(n, d)| {
                let mut denominator = self.univariate(&[1]);
                for &(f, e) in d {
                    denominator = &denominator * &self.univariate(&self.factors[f]).pow(e);
                }
                ring.try_from_fraction(&self.univariate(n), &denominator)
                    .unwrap()
            })
            .collect()
    }

    fn rational(&self) -> Vec<RationalPolynomial<Zp64, u16>> {
        self.coefficients
            .iter()
            .map(|(n, d)| {
                let mut denominator = self.multivariate(&[1]);
                for &(f, e) in d {
                    denominator = &denominator * &self.multivariate(&self.factors[f]).pow(e);
                }
                RationalPolynomial::from_num_den(
                    self.multivariate(n),
                    denominator,
                    &self.field,
                    true,
                )
            })
            .collect()
    }

    fn factorized(&self) -> Vec<FactorizedRationalPolynomial<Zp64, u16>> {
        self.coefficients
            .iter()
            .map(|(n, d)| {
                let denominators = d
                    .iter()
                    .map(|&(f, e)| (self.multivariate(&self.factors[f]), e))
                    .collect();
                FactorizedRationalPolynomial::from_num_den(
                    self.multivariate(n),
                    denominators,
                    &self.field,
                    false,
                )
            })
            .collect()
    }
}

/// Check once per case that the three representations agree.
fn validate(case: Case) {
    let f = fixture(case);
    let (ring, coefficients) = f.seeded_partial_fractions();
    if case.covered_quadratic {
        assert!(coefficients.iter().any(|a| {
            a.denominator_exponents()
                .any(|(i, _)| ring.factor(i).degree() == 2)
        }));
    }
    let pf = run(&ring, &coefficients, f.steps);
    let (cold_ring, cold_coefficients) = f.cold_partial_fractions();
    eprintln!(
        "{case}: seeded basis {} factors, degrees {:?}; learned basis {} factors, degrees {:?}",
        ring.factor_count(),
        (0..ring.factor_count())
            .map(|i| ring.factor(i).degree())
            .collect::<Vec<_>>(),
        cold_ring.factor_count(),
        (0..cold_ring.factor_count())
            .map(|i| cold_ring.factor(i).degree())
            .collect::<Vec<_>>()
    );
    let cold = run(&cold_ring, &cold_coefficients, f.steps);
    let factor_count = cold_ring.factor_count();
    let warm_coefficients = f.import_partial_fractions(&cold_ring);
    assert_eq!(cold_ring.factor_count(), factor_count);
    assert_eq!(warm_coefficients, cold_coefficients);
    let warm_reimport = run(&cold_ring, &warm_coefficients, f.steps);
    assert_eq!(cold_ring.factor_count(), factor_count);
    let rational = f.rational();
    let field = RationalPolynomialField::new(f.field.clone());
    let rp = run(&field, &rational, f.steps);
    let factorized = f.factorized();
    let frp_field = FactorizedRationalPolynomialField::new(
        f.field.clone(),
        Arc::new(vec![f.variable.as_ref().clone()]),
    );
    let frp = run(&frp_field, &factorized, f.steps);
    for (((a, b), c), d) in pf.iter().zip(&rp).zip(&frp).zip(&cold) {
        let a: RationalPolynomial<Zp64, u16> = ring.to_rational_polynomial(a, &f.field);
        assert_eq!(&a, b);
        let d: RationalPolynomial<Zp64, u16> = cold_ring.to_rational_polynomial(d, &f.field);
        assert_eq!(&d, b);
        let numerator = c.numerator.clone().mul_coeff(c.numer_coeff);
        let denominator = c.denominators.iter().fold(
            f.multivariate(&[1]).mul_coeff(c.denom_coeff),
            |d, (p, e)| &d * &p.pow(*e),
        );
        let c = RationalPolynomial::from_num_den(numerator, denominator, &f.field, true);
        assert_eq!(&c, b);
    }
    for (a, b) in warm_reimport.iter().zip(&rp) {
        let a: RationalPolynomial<Zp64, u16> = cold_ring.to_rational_polynomial(a, &f.field);
        assert_eq!(&a, b);
    }
}

fn main() {
    for &case in CASES {
        validate(case);
    }
    divan::main();
}

#[divan::bench_group(sample_count = 10, sample_size = 1, skip_ext_time)]
mod elimination {
    use super::*;

    /// Ring creation, coefficient conversion (including factor discovery),
    /// and the workload.
    #[divan::bench(args = CASES)]
    fn partial_fraction_ring_cold(bencher: divan::Bencher, case: Case) {
        let f = fixture(case);
        bencher.bench_local(|| {
            let (ring, coefficients) = f.cold_partial_fractions();
            run(&ring, black_box(&coefficients), f.steps)
        });
    }

    /// Arithmetic only, after the ring has learned the factors
    /// and all coefficients have been converted outside the timed region.
    #[divan::bench(args = CASES)]
    fn partial_fraction_ring_warm(bencher: divan::Bencher, case: Case) {
        let f = fixture(case);
        let (ring, coefficients) = f.cold_partial_fractions();
        bencher.bench_local(|| run(&ring, black_box(&coefficients), f.steps));
    }

    /// Rebuild and re-import the coefficient fractions with a warm
    /// denominator cache, then run the arithmetic workload. Ring creation
    /// and the first import are outside the timed region.
    #[divan::bench(args = CASES)]
    fn partial_fraction_ring_warm_reimport(bencher: divan::Bencher, case: Case) {
        let f = fixture(case);
        let (ring, _) = f.cold_partial_fractions();
        bencher.bench_local(|| {
            let coefficients = f.import_partial_fractions(&ring);
            run(&ring, black_box(&coefficients), f.steps)
        });
    }

    #[divan::bench(args = CASES)]
    fn rational_polynomial(bencher: divan::Bencher, case: Case) {
        let f = fixture(case);
        let coefficients = f.rational();
        let field = RationalPolynomialField::new(f.field.clone());
        bencher.bench_local(|| run(&field, black_box(&coefficients), f.steps));
    }

    #[divan::bench(args = CASES)]
    fn factorized_rational_polynomial(bencher: divan::Bencher, case: Case) {
        let f = fixture(case);
        let coefficients = f.factorized();
        let field = FactorizedRationalPolynomialField::new(
            f.field.clone(),
            Arc::new(vec![f.variable.as_ref().clone()]),
        );
        bencher.bench_local(|| run(&field, black_box(&coefficients), f.steps));
    }
}
