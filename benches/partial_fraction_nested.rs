//! Elimination with partial fractions in x over Q(y), compared with ordinary
//! and factorized rational polynomials over Q in the two variables x and y.
//!
//! Four basis factors are x - i*y - i^2, i=1..4. Coefficient numerators have
//! degree at most one in x, with genuinely rational coefficients in y; their
//! denominators also contain one or two basis factors. Distinct roots have
//! nonconstant differences in y, exercising rational-polynomial arithmetic
//! in the coefficient field (including the precomputed inverse differences).
//!
//! All fixture construction and exact validation are outside timing. The
//! partial-fraction basis is fixed: factorization over rational-function
//! coefficient fields is not provided by the current self-extending API.
//!
//! cargo bench --features symbolica_benchmarks --bench partial_fraction_nested

use std::{hint::black_box, sync::Arc};
use symbolica::{
    domains::{
        Field, Ring, RingOps,
        factorized_rational_polynomial::{
            FactorizedRationalPolynomial, FactorizedRationalPolynomialField,
            FromNumeratorAndFactorizedDenominator,
        },
        integer::{IntegerRing, Z},
        partial_fraction::{PartialFraction, PartialFractionRing},
        rational_polynomial::{
            FromNumeratorAndDenominator, RationalPolynomial, RationalPolynomialField,
        },
    },
    poly::{PolyVariable, polynomial::MultivariatePolynomial, univariate::UnivariatePolynomial},
    symbol,
};

type Poly = MultivariatePolynomial<IntegerRing, u16>;
type Rat = RationalPolynomial<IntegerRing, u16>;
type RatField = RationalPolynomialField<IntegerRing, u16>;
type Factored = FactorizedRationalPolynomial<IntegerRing, u16>;
const SLOTS: usize = 6;
const COEFFICIENTS: usize = 12;
const FAN_OUT: usize = 2;
const CASES: &[usize] = &[8, 16];

struct Fixture {
    field: RatField,
    x: Arc<PolyVariable>,
    xy: Arc<Vec<PolyVariable>>,
    y: Arc<Vec<PolyVariable>>,
}

impl Fixture {
    fn new() -> Self {
        let x: PolyVariable = symbol!("nested_x").into();
        let y: PolyVariable = symbol!("nested_y").into();
        Self {
            field: RatField::new(Z),
            x: Arc::new(x.clone()),
            xy: Arc::new(vec![x, y.clone()]),
            y: Arc::new(vec![y]),
        }
    }

    fn linear_y(&self, slope: i64, constant: i64) -> Poly {
        let mut p = Poly::new(&Z, None, self.y.clone());
        p.append_monomial(constant.into(), &[0]);
        p.append_monomial(slope.into(), &[1]);
        p
    }

    fn linear_xy(&self, x: i64, y: i64, constant: i64) -> Poly {
        let mut p = Poly::new(&Z, None, self.xy.clone());
        p.append_monomial(constant.into(), &[0, 0]);
        p.append_monomial(y.into(), &[0, 1]);
        p.append_monomial(x.into(), &[1, 0]);
        p
    }

    fn univariate(&self, coefficients: Vec<Rat>) -> UnivariatePolynomial<RatField> {
        UnivariatePolynomial::from_coefficients(&self.field, coefficients, self.x.clone())
    }

    fn factor_powers(i: usize) -> Vec<(usize, usize)> {
        let first = i % 4;
        if i % 2 == 0 {
            vec![(first, 1), ((first + 1) % 4, 1)]
        } else {
            vec![(first, 1)]
        }
    }

    // (a*x + b*y + c) / ((y+s) * product of basis factors).
    fn numbers(i: usize) -> (i64, i64, i64, i64) {
        (
            1 + (i % 3) as i64,
            1 + (i % 2) as i64,
            2 + (i % 5) as i64,
            5 + (i % 3) as i64,
        )
    }

    fn partial_fractions(
        &self,
    ) -> (
        PartialFractionRing<RatField, false>,
        Vec<PartialFraction<RatField>>,
    ) {
        let factors: Vec<_> = (1..=4)
            .map(|i| {
                let root = self.linear_y(-i, -i * i);
                self.univariate(vec![
                    Rat::from_num_den(root.clone(), root.one(), &Z, false),
                    self.field.one(),
                ])
            })
            .collect();
        let ring =
            PartialFractionRing::with_fixed_basis(self.field.clone(), self.x.clone(), &factors)
                .unwrap();
        let coefficients = (0..COEFFICIENTS)
            .map(|i| {
                let (a, b, c, s) = Self::numbers(i);
                let den = self.linear_y(1, s);
                let n = self.univariate(vec![
                    Rat::from_num_den(self.linear_y(b, c), den.clone(), &Z, true),
                    Rat::from_num_den(self.linear_y(0, a), den, &Z, true),
                ]);
                ring.mul_fraction(&ring.one(), &n, &Self::factor_powers(i))
            })
            .collect();
        (ring, coefficients)
    }

    fn factorized(&self) -> Vec<Factored> {
        (0..COEFFICIENTS)
            .map(|i| {
                let (a, b, c, s) = Self::numbers(i);
                let mut factors = vec![(self.linear_xy(0, 1, s), 1)];
                for (f, e) in Self::factor_powers(i) {
                    let j = f as i64 + 1;
                    factors.push((self.linear_xy(1, -j, -j * j), e));
                }
                Factored::from_num_den(self.linear_xy(a, b, c), factors, &Z, false)
            })
            .collect()
    }

    fn flatten_factorized(&self, a: &Factored) -> Rat {
        let n = a.numerator.clone().mul_coeff(a.numer_coeff.clone());
        let d = a.denominators.iter().fold(
            self.linear_xy(0, 0, 1).mul_coeff(a.denom_coeff.clone()),
            |d, (p, e)| &d * &p.pow(*e),
        );
        Rat::from_num_den(n, d, &Z, true)
    }

    fn rational(&self) -> Vec<Rat> {
        self.factorized()
            .iter()
            .map(|a| self.flatten_factorized(a))
            .collect()
    }

    fn flatten_univariate(&self, p: &UnivariatePolynomial<RatField>) -> Rat {
        let x = self.linear_xy(1, 0, 0);
        let x = Rat::from_num_den(x.clone(), x.one(), &Z, false);
        let mut result = self.field.zero();
        for c in p.coefficients().iter().rev() {
            result = self.field.add(&self.field.mul(&result, &x), c);
        }
        result
    }

    fn flatten_partial_fraction(
        &self,
        ring: &PartialFractionRing<RatField, false>,
        a: &PartialFraction<RatField>,
    ) -> Rat {
        let (n, d) = ring.to_fraction(a);
        self.field
            .div(&self.flatten_univariate(&n), &self.flatten_univariate(&d))
    }
}

fn run<R: Ring>(ring: &R, coefficients: &[R::Element], steps: usize) -> Vec<R::Element> {
    let mut slots = coefficients[..SLOTS].to_vec();
    for t in 0..steps {
        let pivot = std::mem::replace(
            &mut slots[t % SLOTS],
            coefficients[(t * 5 + 3) % COEFFICIENTS].clone(),
        );
        for k in 1..=FAN_OUT {
            ring.add_mul_assign(
                &mut slots[(t + k) % SLOTS],
                &pivot,
                &coefficients[(t * 7 + k) % COEFFICIENTS],
            );
        }
    }
    slots
}

fn validate(steps: usize) {
    let f = Fixture::new();
    let (pf_ring, pf_coefficients) = f.partial_fractions();
    let mut used = [false; 4];
    for coefficient in &pf_coefficients {
        for (factor, exponent) in coefficient.denominator_exponents() {
            assert!(exponent > 0);
            used[factor] = true;
        }
    }
    assert!(
        used.iter().all(|used| *used),
        "every basis factor must occur"
    );
    let rational = f.rational();
    let factored = f.factorized();
    let frp_ring = FactorizedRationalPolynomialField::new(Z, f.xy.clone());
    for (a, b) in pf_coefficients.iter().zip(&rational) {
        assert_eq!(&f.flatten_partial_fraction(&pf_ring, a), b);
    }
    let pf = run(&pf_ring, &pf_coefficients, steps);
    let rp = run(&f.field, &rational, steps);
    let frp = run(&frp_ring, &factored, steps);
    for ((a, b), c) in pf.iter().zip(&rp).zip(&frp) {
        assert_eq!(&f.flatten_partial_fraction(&pf_ring, a), b);
        assert_eq!(&f.flatten_factorized(c), b);
    }
    eprintln!("Validated Q(y)[x] partial fractions against Q(x,y): 4 factors, {steps} steps");
}

fn main() {
    for &steps in CASES {
        validate(steps);
    }
    divan::main();
}

#[divan::bench_group(sample_count = 10, sample_size = 1, skip_ext_time)]
mod elimination {
    use super::*;

    #[divan::bench(args = CASES)]
    fn partial_fraction_over_rational_polynomial(bencher: divan::Bencher, steps: usize) {
        let f = Fixture::new();
        let (ring, coefficients) = f.partial_fractions();
        bencher.bench_local(|| run(&ring, black_box(&coefficients), steps));
    }

    #[divan::bench(args = CASES)]
    fn rational_polynomial_two_variables(bencher: divan::Bencher, steps: usize) {
        let f = Fixture::new();
        let coefficients = f.rational();
        bencher.bench_local(|| run(&f.field, black_box(&coefficients), steps));
    }

    #[divan::bench(args = CASES)]
    fn factorized_rational_polynomial_two_variables(bencher: divan::Bencher, steps: usize) {
        let f = Fixture::new();
        let coefficients = f.factorized();
        let ring = FactorizedRationalPolynomialField::new(Z, f.xy.clone());
        bencher.bench_local(|| run(&ring, black_box(&coefficients), steps));
    }
}
