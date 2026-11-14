//! Partial fractions only in the dimension d, with rational functions of y
//! as coefficients. Unlike `partial_fraction_nested`, every pole is d - i
//! (i=1..4), independent of y; inverse differences between poles are rational
//! numbers, so partial fractioning introduces no kinematic denominators.
//!
//! The four representations are partial fractions over ordinary Q(y), partial
//! fractions over factorized Q(y), ordinary Q(d,y), and factorized Q(d,y).
//! All use the same coefficients (a*d+b*y+c)/((y+s)*product(d-i)), and the
//! same six-slot elimination workload with twelve coefficients and fanout two.
//! All four poles are used. Construction and exact input/result validation
//! are outside the timed regions; every benchmark measures arithmetic only.
//!
//! cargo bench --features symbolica_benchmarks --bench partial_fraction_dimension

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
type FactoredField = FactorizedRationalPolynomialField<IntegerRing, u16>;
const SLOTS: usize = 6;
const COEFFICIENTS: usize = 12;
const FAN_OUT: usize = 2;
const CASES: &[usize] = &[8, 16];

struct Fixture {
    ordinary: RatField,
    factorized_y: FactoredField,
    d: Arc<PolyVariable>,
    dy: Arc<Vec<PolyVariable>>,
    y: Arc<Vec<PolyVariable>>,
}

impl Fixture {
    fn new() -> Self {
        let d: PolyVariable = symbol!("dimension_d").into();
        let y: PolyVariable = symbol!("dimension_y").into();
        let y_variables = Arc::new(vec![y.clone()]);
        Self {
            ordinary: RatField::new(Z),
            factorized_y: FactoredField::new(Z, y_variables.clone()),
            d: Arc::new(d.clone()),
            dy: Arc::new(vec![d, y]),
            y: y_variables,
        }
    }

    fn linear_y(&self, slope: i64, constant: i64) -> Poly {
        let mut p = Poly::new(&Z, None, self.y.clone());
        p.append_monomial(constant.into(), &[0]);
        p.append_monomial(slope.into(), &[1]);
        p
    }

    fn linear_dy(&self, d: i64, y: i64, constant: i64) -> Poly {
        let mut p = Poly::new(&Z, None, self.dy.clone());
        p.append_monomial(constant.into(), &[0, 0]);
        p.append_monomial(y.into(), &[0, 1]);
        p.append_monomial(d.into(), &[1, 0]);
        p
    }

    fn factor_powers(i: usize) -> Vec<(usize, usize)> {
        let first = i % 4;
        if i % 2 == 0 {
            vec![(first, 1), ((first + 1) % 4, 1)]
        } else {
            vec![(first, 1)]
        }
    }

    fn numbers(i: usize) -> (i64, i64, i64, i64) {
        (
            1 + (i % 3) as i64,
            1 + (i % 2) as i64,
            2 + (i % 5) as i64,
            5 + (i % 3) as i64,
        )
    }

    fn partial_fractions<F: Field>(
        &self,
        field: F,
        coefficient: impl Fn(Poly, Poly) -> F::Element,
    ) -> (PartialFractionRing<F>, Vec<PartialFraction<F>>) {
        let univariate = |c| UnivariatePolynomial::from_coefficients(&field, c, self.d.clone());
        let factors: Vec<_> = (1..=4)
            .map(|i: i64| univariate(vec![field.nth((-i).into()), field.one()]))
            .collect();
        let ring =
            PartialFractionRing::with_basis(field.clone(), self.d.clone(), &factors).unwrap();
        let coefficients = (0..COEFFICIENTS)
            .map(|i| {
                let (a, b, c, s) = Self::numbers(i);
                let denominator = self.linear_y(1, s);
                let numerator = univariate(vec![
                    coefficient(self.linear_y(b, c), denominator.clone()),
                    coefficient(self.linear_y(0, a), denominator),
                ]);
                ring.mul_fraction(&ring.one(), &numerator, &Self::factor_powers(i))
            })
            .collect();
        (ring, coefficients)
    }

    fn ordinary_coefficients(
        &self,
    ) -> (
        PartialFractionRing<RatField>,
        Vec<PartialFraction<RatField>>,
    ) {
        self.partial_fractions(self.ordinary.clone(), |n, d| {
            Rat::from_num_den(n, d, &Z, true)
        })
    }

    fn factorized_coefficients(
        &self,
    ) -> (
        PartialFractionRing<FactoredField>,
        Vec<PartialFraction<FactoredField>>,
    ) {
        self.partial_fractions(self.factorized_y.clone(), |n, d| {
            Factored::from_num_den(n, vec![(d, 1)], &Z, false)
        })
    }

    fn flattened_factorized(&self) -> Vec<Factored> {
        (0..COEFFICIENTS)
            .map(|i| {
                let (a, b, c, s) = Self::numbers(i);
                let mut factors = vec![(self.linear_dy(0, 1, s), 1)];
                for (f, e) in Self::factor_powers(i) {
                    factors.push((self.linear_dy(1, 0, -(f as i64 + 1)), e));
                }
                Factored::from_num_den(self.linear_dy(a, b, c), factors, &Z, false)
            })
            .collect()
    }

    /// Works for both Q(y) coefficients and flattened Q(d,y) values.
    fn flatten_factorized(a: &Factored) -> Rat {
        let n = a.numerator.clone().mul_coeff(a.numer_coeff.clone());
        let d = a.denominators.iter().fold(
            a.numerator.one().mul_coeff(a.denom_coeff.clone()),
            |d, (p, e)| &d * &p.pow(*e),
        );
        Rat::from_num_den(n, d, &Z, true)
    }

    fn flattened_ordinary(&self) -> Vec<Rat> {
        self.flattened_factorized()
            .iter()
            .map(Self::flatten_factorized)
            .collect()
    }

    fn flatten_univariate<F: Field>(
        &self,
        p: &UnivariatePolynomial<F>,
        coefficient: &impl Fn(&F::Element) -> Rat,
    ) -> Rat {
        let d = self.linear_dy(1, 0, 0);
        let d = Rat::from_num_den(d.clone(), d.one(), &Z, false);
        let mut result = self.ordinary.zero();
        for c in p.coefficients().iter().rev() {
            result = self
                .ordinary
                .add(&self.ordinary.mul(&result, &d), &coefficient(c));
        }
        result
    }

    fn flatten_partial_fraction<F: Field>(
        &self,
        ring: &PartialFractionRing<F>,
        a: &PartialFraction<F>,
        coefficient: impl Fn(&F::Element) -> Rat,
    ) -> Rat {
        let (n, d) = ring.to_fraction(a);
        self.ordinary.div(
            &self.flatten_univariate(&n, &coefficient),
            &self.flatten_univariate(&d, &coefficient),
        )
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

fn assert_pole_coverage<F: Field>(coefficients: &[PartialFraction<F>]) {
    let mut used = [false; 4];
    for coefficient in coefficients {
        for (factor, exponent) in coefficient.denominator_exponents() {
            assert!(exponent > 0);
            used[factor] = true;
        }
    }
    assert!(
        used.iter().all(|used| *used),
        "every dimension pole must occur"
    );
}

fn validate(steps: usize) {
    let f = Fixture::new();
    let (pf_ordinary, ordinary_coefficients) = f.ordinary_coefficients();
    let (pf_factorized, factorized_coefficients) = f.factorized_coefficients();
    assert_pole_coverage(&ordinary_coefficients);
    assert_pole_coverage(&factorized_coefficients);
    let ordinary = f.flattened_ordinary();
    let factorized = f.flattened_factorized();
    let flattened_field = FactoredField::new(Z, f.dy.clone());
    for i in 0..COEFFICIENTS {
        assert_eq!(
            f.flatten_partial_fraction(&pf_ordinary, &ordinary_coefficients[i], Clone::clone),
            ordinary[i]
        );
        assert_eq!(
            f.flatten_partial_fraction(
                &pf_factorized,
                &factorized_coefficients[i],
                Fixture::flatten_factorized,
            ),
            ordinary[i]
        );
        assert_eq!(Fixture::flatten_factorized(&factorized[i]), ordinary[i]);
    }
    let pf_o = run(&pf_ordinary, &ordinary_coefficients, steps);
    let pf_f = run(&pf_factorized, &factorized_coefficients, steps);
    let rp = run(&f.ordinary, &ordinary, steps);
    let frp = run(&flattened_field, &factorized, steps);
    for i in 0..SLOTS {
        assert_eq!(
            f.flatten_partial_fraction(&pf_ordinary, &pf_o[i], Clone::clone),
            rp[i]
        );
        assert_eq!(
            f.flatten_partial_fraction(&pf_factorized, &pf_f[i], Fixture::flatten_factorized),
            rp[i]
        );
        assert_eq!(Fixture::flatten_factorized(&frp[i]), rp[i]);
    }
    eprintln!(
        "Validated dimension-only poles d-1..d-4: ordinary/factorized Q(y) coefficients and Q(d,y), {steps} steps"
    );
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
    fn partial_fraction_rational_coefficients(bencher: divan::Bencher, steps: usize) {
        let f = Fixture::new();
        let (ring, coefficients) = f.ordinary_coefficients();
        bencher.bench_local(|| run(&ring, black_box(&coefficients), steps));
    }

    #[divan::bench(args = CASES)]
    fn partial_fraction_factorized_coefficients(bencher: divan::Bencher, steps: usize) {
        let f = Fixture::new();
        let (ring, coefficients) = f.factorized_coefficients();
        bencher.bench_local(|| run(&ring, black_box(&coefficients), steps));
    }

    #[divan::bench(args = CASES)]
    fn ordinary_two_variables(bencher: divan::Bencher, steps: usize) {
        let f = Fixture::new();
        let coefficients = f.flattened_ordinary();
        bencher.bench_local(|| run(&f.ordinary, black_box(&coefficients), steps));
    }

    #[divan::bench(args = CASES)]
    fn factorized_two_variables(bencher: divan::Bencher, steps: usize) {
        let f = Fixture::new();
        let coefficients = f.flattened_factorized();
        let ring = FactoredField::new(Z, f.dy.clone());
        bencher.bench_local(|| run(&ring, black_box(&coefficients), steps));
    }
}
