//! Arithmetic of univariate rational functions with a few recurring
//! denominator factors in [PartialFractionRing], compared with
//! [RationalPolynomial] and [FactorizedRationalPolynomial].
//!
//! Run with `cargo bench --features symbolica_benchmarks --bench partial_fraction`.
//!
//! The elimination workload mimics the coefficient arithmetic of a linear
//! system: `SLOTS` accumulators start as coefficients; every step takes one
//! accumulator as the pivot, adds `pivot * c` to the next `FAN_OUT`
//! accumulators for coefficients `c` from a fixed pool, and reinitializes the
//! pivot's slot. The sum workload adds all pairwise products of the pool with
//! one call of [Ring::sum].
//!
//! - `zp`: functions of `x` over Zp with linear (and one irreducible
//!   quadratic) factors. `partial_fraction_cold` includes the creation of a
//!   ring without hints and the import of the coefficients from fractions,
//!   which discovers the factors; the other benchmarks only time arithmetic.
//! - `dimension`: poles `d - i` in the dimension `d`, with coefficients in
//!   `Q(y)` (ordinary or factorized), compared with `Q(d, y)`.
//! - `kinematic`: as `dimension`, but with poles `d - i y - i^2` whose roots
//!   depend on `y`, so that partial fractioning creates denominators in `y`.
//!
//! - `basis`: rings with `k` linear factors `x - i` over Zp and Q: creating
//!   the basis, importing fresh fractions whose denominators are products of
//!   three basis factors (on a ring whose denominator cache is empty), and
//!   short sums and products of such elements.
//!
//! Fixture construction and validation (all representations agree) are
//! outside the timed regions. Validation runs before all benchmarks and takes
//! a few seconds; set `PARTIAL_FRACTION_BENCH_NO_VALIDATION` to skip it, for
//! example when profiling single benchmarks.

use std::{fmt, hint::black_box, sync::Arc};

use symbolica::{
    domains::{
        Field, Ring, RingOps,
        factorized_rational_polynomial::{
            FactorizedRationalPolynomial, FactorizedRationalPolynomialField,
            FromNumeratorAndFactorizedDenominator,
        },
        finite_field::{FiniteFieldCore, Zp64},
        integer::{IntegerRing, Z},
        partial_fraction::{PartialFraction, PartialFractionRing},
        rational::Q,
        rational_polynomial::{
            FromNumeratorAndDenominator, RationalPolynomial, RationalPolynomialField,
        },
    },
    poly::{
        LexOrder, PolyVariable, factor::Factorize, polynomial::MultivariatePolynomial,
        univariate::UnivariatePolynomial,
    },
    symbol,
};

const PRIME: u64 = 9_223_372_036_854_775_783; // largest prime below 2^63
const SLOTS: usize = 12;
const FAN_OUT: usize = 3;

/// Run the elimination workload.
fn eliminate<R: Ring>(ring: &R, coefficients: &[R::Element], steps: usize) -> Vec<R::Element> {
    let n = coefficients.len();
    let mut slots: Vec<R::Element> = coefficients[..SLOTS.min(n)].to_vec();
    let slot_count = slots.len();
    for t in 0..steps {
        let pivot = std::mem::replace(
            &mut slots[t % slot_count],
            coefficients[(t * 5 + 3) % n].clone(),
        );
        for k in 1..=FAN_OUT {
            let c = &coefficients[(t * 7 + k) % n];
            ring.add_mul_assign(&mut slots[(t + k) % slot_count], &pivot, c);
        }
    }
    slots
}

/// All pairwise products of the coefficients, the summands of the sum workload.
fn products<R: Ring>(ring: &R, coefficients: &[R::Element]) -> Vec<R::Element> {
    let mut products = vec![];
    for (i, a) in coefficients.iter().enumerate() {
        for b in &coefficients[i..] {
            products.push(ring.mul(a, b));
        }
    }
    products
}

/// Repeated addition, the baseline of [Ring::sum].
fn fold<R: Ring>(ring: &R, terms: Vec<R::Element>) -> R::Element {
    let mut sum = ring.zero();
    for t in terms {
        ring.add_assign(&mut sum, t);
    }
    sum
}

mod zp {
    use super::*;

    const COEFFICIENTS: usize = 24;

    #[derive(Clone, Copy, Debug)]
    pub struct Case {
        /// Linear factors `x - 1, ..., x - linear`.
        linear: usize,
        /// Whether the basis also holds the irreducible quadratic `x^2 + 1`
        /// (`p = 3 mod 4`), which some coefficients use.
        quadratic: bool,
        pub steps: usize,
    }

    impl fmt::Display for Case {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            let quadratic = if self.quadratic { "+quad" } else { "" };
            write!(f, "{}lin{quadratic}_{}steps", self.linear, self.steps)
        }
    }

    pub const CASES: &[Case] = &[
        Case {
            linear: 4,
            quadratic: false,
            steps: 60,
        },
        Case {
            linear: 8,
            quadratic: false,
            steps: 120,
        },
        Case {
            linear: 6,
            quadratic: true,
            steps: 60,
        },
    ];

    /// A coefficient: ascending numerator and `(factor index, exponent)` list.
    type Coefficient = (Vec<u64>, Vec<(usize, usize)>);

    pub struct Fixture {
        pub field: Zp64,
        variable: Arc<PolyVariable>,
        factors: Vec<Vec<u64>>,
        coefficients: Vec<Coefficient>,
    }

    impl Fixture {
        pub fn new(case: Case) -> Self {
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
                factors.push(vec![1, 0, 1]);
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
            if case.quadratic {
                coefficients[0].1 = vec![(case.linear, 1)];
                coefficients[1].1 = vec![(0, 1), (case.linear, 2)];
            }
            Self {
                field: Zp64::new(PRIME),
                variable: Arc::new(symbol!("pf_bench_x").into()),
                factors,
                coefficients,
            }
        }

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

        /// Create a ring without hints and import the coefficients as
        /// fractions: known factors are trial-divided before remaining
        /// cofactors are factored.
        pub fn partial_fractions(&self) -> (PartialFractionRing<Zp64>, Vec<PartialFraction<Zp64>>) {
            let ring = PartialFractionRing::new(self.field.clone(), self.variable.clone(), &[]);
            let coefficients = self
                .coefficients
                .iter()
                .map(|(n, d)| {
                    let mut denominator = self.univariate(&[1]);
                    for &(f, e) in d {
                        denominator = &denominator * &self.univariate(&self.factors[f]).pow(e);
                    }
                    ring.try_from_fraction(&self.univariate(n), &denominator)
                        .unwrap()
                })
                .collect();
            (ring, coefficients)
        }

        pub fn rational(&self) -> Vec<RationalPolynomial<Zp64, u16>> {
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

        pub fn factorized_field(&self) -> FactorizedRationalPolynomialField<Zp64, u16> {
            FactorizedRationalPolynomialField::new(
                self.field.clone(),
                Arc::new(vec![self.variable.as_ref().clone()]),
            )
        }

        pub fn factorized(&self) -> Vec<FactorizedRationalPolynomial<Zp64, u16>> {
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

        fn flatten(
            &self,
            c: &FactorizedRationalPolynomial<Zp64, u16>,
        ) -> RationalPolynomial<Zp64, u16> {
            let numerator = c.numerator.clone().mul_coeff(c.numer_coeff);
            let denominator = c.denominators.iter().fold(
                self.multivariate(&[1]).mul_coeff(c.denom_coeff),
                |d, (p, e)| &d * &p.pow(*e),
            );
            RationalPolynomial::from_num_den(numerator, denominator, &self.field, true)
        }
    }

    pub fn validate(case: Case) {
        let f = Fixture::new(case);
        let (ring, coefficients) = f.partial_fractions();
        let degrees: Vec<_> = (0..ring.factor_count())
            .map(|i| ring.factor(i).degree())
            .collect();
        assert_eq!(degrees.len(), f.factors.len());
        let field = RationalPolynomialField::new(f.field.clone());
        let rational = f.rational();
        let factorized_field = f.factorized_field();
        let factorized = f.factorized();
        let check = |a: &PartialFraction<Zp64>, b: &RationalPolynomial<Zp64, u16>| {
            let a: RationalPolynomial<Zp64, u16> = ring.to_rational_polynomial(a, &f.field);
            assert_eq!(&a, b);
        };

        let pf = eliminate(&ring, &coefficients, case.steps);
        let rp = eliminate(&field, &rational, case.steps);
        let frp = eliminate(&factorized_field, &factorized, case.steps);
        for ((a, b), c) in pf.iter().zip(&rp).zip(&frp) {
            check(a, b);
            assert_eq!(&f.flatten(c), b);
        }

        let terms = products(&ring, &coefficients);
        let sum = ring.sum(terms.clone());
        assert_eq!(sum, fold(&ring, terms));
        check(&sum, &field.sum(products(&field, &rational)));
        eprintln!("zp {case}: basis degrees {degrees:?}, validated");
    }
}

mod nested {
    use super::*;

    type Poly = MultivariatePolynomial<IntegerRing, u16>;
    type Rat = RationalPolynomial<IntegerRing, u16>;
    pub type RatField = RationalPolynomialField<IntegerRing, u16>;
    type Factored = FactorizedRationalPolynomial<IntegerRing, u16>;
    pub type FactoredField = FactorizedRationalPolynomialField<IntegerRing, u16>;

    const COEFFICIENTS: usize = 12;
    const FACTORS: i64 = 4;

    #[derive(Clone, Copy, Debug)]
    pub struct Case {
        /// Whether the roots of the poles depend on `y`.
        pub kinematic: bool,
        pub steps: usize,
    }

    impl fmt::Display for Case {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "{}steps", self.steps)
        }
    }

    pub const DIMENSION: &[Case] = &[
        Case {
            kinematic: false,
            steps: 8,
        },
        Case {
            kinematic: false,
            steps: 16,
        },
    ];

    pub const KINEMATIC: &[Case] = &[
        Case {
            kinematic: true,
            steps: 8,
        },
        Case {
            kinematic: true,
            steps: 16,
        },
    ];

    /// Coefficients `(a d + b y + c) / ((y + s) prod_i (d - r_i)^e_i)`, with
    /// the roots `r_i = i` or `r_i = i y + i^2`.
    pub struct Fixture {
        kinematic: bool,
        pub ordinary: RatField,
        pub factorized: FactoredField,
        /// The flattened field `Q(d, y)`.
        pub flat: FactoredField,
        d: Arc<PolyVariable>,
        dy: Arc<Vec<PolyVariable>>,
        y: Arc<Vec<PolyVariable>>,
    }

    impl Fixture {
        pub fn new(case: Case) -> Self {
            let d: PolyVariable = symbol!("pf_bench_d").into();
            let y: PolyVariable = symbol!("pf_bench_y").into();
            let y_variables = Arc::new(vec![y.clone()]);
            let dy = Arc::new(vec![d.clone(), y]);
            Self {
                kinematic: case.kinematic,
                ordinary: RatField::new(Z),
                factorized: FactoredField::new(Z, y_variables.clone()),
                flat: FactoredField::new(Z, dy.clone()),
                d: Arc::new(d),
                dy,
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

        /// `r_i` as the slope and constant of a linear polynomial in `y`.
        fn root(&self, i: i64) -> (i64, i64) {
            if self.kinematic { (i, i * i) } else { (0, i) }
        }

        fn factor_powers(i: usize) -> Vec<(usize, usize)> {
            let first = i % FACTORS as usize;
            if i % 2 == 0 {
                vec![(first, 1), ((first + 1) % FACTORS as usize, 1)]
            } else {
                vec![(first, 1)]
            }
        }

        /// `(a, b, c, s)` of coefficient `i`.
        fn numbers(i: usize) -> (i64, i64, i64, i64) {
            (
                1 + (i % 3) as i64,
                1 + (i % 2) as i64,
                2 + (i % 5) as i64,
                5 + (i % 3) as i64,
            )
        }

        /// The coefficients in partial-fraction form in `d` over `field`,
        /// whose elements are created from a numerator and denominator in `y`.
        fn partial_fractions<F: Field>(
            &self,
            field: F,
            coefficient: impl Fn(Poly, Poly) -> F::Element,
        ) -> (PartialFractionRing<F>, Vec<PartialFraction<F>>) {
            let univariate = |c| UnivariatePolynomial::from_coefficients(&field, c, self.d.clone());
            let one = self.linear_y(0, 1);
            let factors: Vec<_> = (1..=FACTORS)
                .map(|i| {
                    let (slope, constant) = self.root(i);
                    let root = coefficient(self.linear_y(-slope, -constant), one.clone());
                    univariate(vec![root, field.one()])
                })
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

        pub fn ordinary_partial_fractions(
            &self,
        ) -> (
            PartialFractionRing<RatField>,
            Vec<PartialFraction<RatField>>,
        ) {
            self.partial_fractions(self.ordinary.clone(), |n, d| {
                Rat::from_num_den(n, d, &Z, true)
            })
        }

        pub fn factorized_partial_fractions(
            &self,
        ) -> (
            PartialFractionRing<FactoredField>,
            Vec<PartialFraction<FactoredField>>,
        ) {
            self.partial_fractions(self.factorized.clone(), |n, d| {
                let denominators = if d.is_one() { vec![] } else { vec![(d, 1)] };
                Factored::from_num_den(n, denominators, &Z, false)
            })
        }

        /// The coefficients in `Q(d, y)`, with factorized denominators.
        pub fn flat_factorized(&self) -> Vec<Factored> {
            (0..COEFFICIENTS)
                .map(|i| {
                    let (a, b, c, s) = Self::numbers(i);
                    let mut factors = vec![(self.linear_dy(0, 1, s), 1)];
                    for (f, e) in Self::factor_powers(i) {
                        let (slope, constant) = self.root(f as i64 + 1);
                        factors.push((self.linear_dy(1, -slope, -constant), e));
                    }
                    Factored::from_num_den(self.linear_dy(a, b, c), factors, &Z, false)
                })
                .collect()
        }

        /// The coefficients in `Q(d, y)`.
        pub fn flat_ordinary(&self) -> Vec<Rat> {
            self.flat_factorized().iter().map(Self::flatten).collect()
        }

        fn flatten(a: &Factored) -> Rat {
            let n = a.numerator.clone().mul_coeff(a.numer_coeff.clone());
            let d = a.denominators.iter().fold(
                a.numerator.one().mul_coeff(a.denom_coeff.clone()),
                |d, (p, e)| &d * &p.pow(*e),
            );
            Rat::from_num_den(n, d, &Z, true)
        }

        /// An element of a partial-fraction ring over `Q(y)` as an element of
        /// `Q(d, y)`.
        fn flatten_partial_fraction<F: Field>(
            &self,
            ring: &PartialFractionRing<F>,
            a: &PartialFraction<F>,
            coefficient: impl Fn(&F::Element) -> Rat,
        ) -> Rat {
            let d = self.linear_dy(1, 0, 0);
            let d = Rat::from_num_den(d.clone(), d.one(), &Z, false);
            let flatten = |p: &UnivariatePolynomial<F>| {
                let mut result = self.ordinary.zero();
                for c in p.coefficients().iter().rev() {
                    result = self
                        .ordinary
                        .add(&self.ordinary.mul(&result, &d), &coefficient(c));
                }
                result
            };
            let (n, d) = ring.to_fraction(a);
            self.ordinary.div(&flatten(&n), &flatten(&d))
        }
    }

    pub fn validate(case: Case) {
        let f = Fixture::new(case);
        let (o_ring, o_coefficients) = f.ordinary_partial_fractions();
        let (f_ring, f_coefficients) = f.factorized_partial_fractions();
        let flat = f.flat_ordinary();
        let flat_factorized = f.flat_factorized();
        let check =
            |a: &PartialFraction<RatField>, b: &PartialFraction<FactoredField>, expected: &Rat| {
                assert_eq!(
                    &f.flatten_partial_fraction(&o_ring, a, Clone::clone),
                    expected
                );
                assert_eq!(
                    &f.flatten_partial_fraction(&f_ring, b, Fixture::flatten),
                    expected
                );
            };
        for i in 0..COEFFICIENTS {
            check(&o_coefficients[i], &f_coefficients[i], &flat[i]);
        }

        let pf_o = eliminate(&o_ring, &o_coefficients, case.steps);
        let pf_f = eliminate(&f_ring, &f_coefficients, case.steps);
        let rp = eliminate(&f.ordinary, &flat, case.steps);
        let frp = eliminate(&f.flat, &flat_factorized, case.steps);
        for i in 0..rp.len() {
            check(&pf_o[i], &pf_f[i], &rp[i]);
            assert_eq!(Fixture::flatten(&frp[i]), rp[i]);
        }

        let o_terms = products(&o_ring, &o_coefficients);
        let o_sum = o_ring.sum(o_terms.clone());
        assert_eq!(o_sum, fold(&o_ring, o_terms));
        let f_sum = f_ring.sum(products(&f_ring, &f_coefficients));
        check(
            &o_sum,
            &f_sum,
            &f.ordinary.sum(products(&f.ordinary, &flat)),
        );
        let kind = if case.kinematic {
            "kinematic"
        } else {
            "dimension"
        };
        eprintln!("{kind} {case}: validated");
    }
}

mod basis {
    use super::*;

    pub const SIZES: &[usize] = &[16, 256, 1024];
    const IMPORTS: usize = 256;

    type Poly<F> = UnivariatePolynomial<F>;

    pub struct Fixture<F: Field> {
        field: F,
        variable: Arc<PolyVariable>,
        factors: Vec<Poly<F>>,
        /// Fractions `(a + b x) / (f_i f_j f_k)` for random basis factors.
        fractions: Vec<(Poly<F>, Poly<F>)>,
    }

    impl<F: Field> Fixture<F>
    where
        MultivariatePolynomial<F, u16, LexOrder>: Factorize,
    {
        pub fn new(field: F, k: usize) -> Self {
            let variable: Arc<PolyVariable> = Arc::new(symbol!("pf_bench_basis_x").into());
            let poly = |c: &[i64]| {
                Poly::from_coefficients(
                    &field,
                    c.iter().map(|&c| field.nth(c.into())).collect(),
                    variable.clone(),
                )
            };
            let factors: Vec<_> = (1..=k as i64).map(|i| poly(&[-i, 1])).collect();
            let mut state = 0x9e37_79b9_7f4a_7c15_u64;
            let mut next = |bound: usize| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state % bound as u64) as usize
            };
            let fractions = (0..IMPORTS)
                .map(|_| {
                    let numerator = poly(&[1 + next(10) as i64, next(10) as i64]);
                    let mut denominator = poly(&[1]);
                    for _ in 0..3 {
                        denominator = &denominator * &factors[next(k)];
                    }
                    (numerator, denominator)
                })
                .collect();
            Self {
                field,
                variable,
                factors,
                fractions,
            }
        }

        /// A ring with the basis factors and an empty denominator cache.
        pub fn ring(&self) -> PartialFractionRing<F> {
            PartialFractionRing::with_basis(
                self.field.clone(),
                self.variable.clone(),
                &self.factors,
            )
            .unwrap()
        }

        /// A factorizing ring with the basis factors and an empty denominator cache.
        pub fn factorizing_ring(&self) -> PartialFractionRing<F> {
            let ring = PartialFractionRing::new(self.field.clone(), self.variable.clone(), &[]);
            for f in &self.factors {
                ring.inv(&ring.element_from_polynomial(f));
            }
            ring
        }

        pub fn import(&self, ring: &PartialFractionRing<F>) -> Vec<PartialFraction<F>> {
            self.fractions
                .iter()
                .map(|(n, d)| ring.try_from_fraction(n, d).unwrap())
                .collect()
        }

        /// The sums of consecutive elements.
        pub fn sums(
            ring: &PartialFractionRing<F>,
            elements: &[PartialFraction<F>],
        ) -> Vec<PartialFraction<F>> {
            elements
                .windows(2)
                .map(|w| ring.sum(w.iter().cloned()))
                .collect()
        }

        /// The products of consecutive elements.
        pub fn products(
            ring: &PartialFractionRing<F>,
            elements: &[PartialFraction<F>],
        ) -> Vec<PartialFraction<F>> {
            elements
                .windows(2)
                .map(|w| ring.mul(&w[0], &w[1]))
                .collect()
        }

        pub fn validate(&self) {
            let ring = self.ring();
            let elements = self.import(&ring);
            assert_eq!(self.import(&self.factorizing_ring()), elements);
            // Compare values at x = -3, which is not a pole.
            let x = self.field.nth((-3).into());
            let value = |a: &PartialFraction<F>| ring.evaluate(a, &x).unwrap();
            let fractions: Vec<_> = self
                .fractions
                .iter()
                .map(|(n, d)| {
                    let at = |p: &Poly<F>| value(&ring.element_from_polynomial(p));
                    self.field.div(&at(n), &at(d))
                })
                .collect();
            for (i, (s, p)) in Self::sums(&ring, &elements)
                .iter()
                .zip(Self::products(&ring, &elements))
                .enumerate()
            {
                let (a, b) = (&fractions[i], &fractions[i + 1]);
                assert_eq!(value(&elements[i]), *a);
                assert_eq!(value(s), self.field.add(a, b));
                assert_eq!(value(&p), self.field.mul(a, b));
            }
        }
    }
}

fn main() {
    if std::env::var_os("PARTIAL_FRACTION_BENCH_NO_VALIDATION").is_none() {
        for &case in zp::CASES {
            zp::validate(case);
        }
        for &case in nested::DIMENSION.iter().chain(nested::KINEMATIC) {
            nested::validate(case);
        }
        basis::Fixture::new(Zp64::new(PRIME), 16).validate();
        basis::Fixture::new(Q, 16).validate();
    }
    divan::main();
}

/// The benchmarks of a large basis over the field `$field`.
macro_rules! basis_benchmarks {
    ($group:ident, $field:expr) => {
        #[divan::bench_group(sample_count = 10, sample_size = 1, skip_ext_time)]
        mod $group {
            use super::{basis::*, *};

            #[divan::bench(args = SIZES)]
            fn create(bencher: divan::Bencher, k: usize) {
                let f = Fixture::new($field, k);
                bencher.bench_local(|| f.ring());
            }

            #[divan::bench(args = SIZES)]
            fn import(bencher: divan::Bencher, k: usize) {
                let f = Fixture::new($field, k);
                bencher
                    .with_inputs(|| f.ring())
                    .bench_local_refs(|ring| f.import(ring));
            }

            #[divan::bench(args = SIZES)]
            fn import_factorizing(bencher: divan::Bencher, k: usize) {
                let f = Fixture::new($field, k);
                bencher
                    .with_inputs(|| f.factorizing_ring())
                    .bench_local_refs(|ring| f.import(ring));
            }

            #[divan::bench(args = SIZES)]
            fn sum_pairs(bencher: divan::Bencher, k: usize) {
                let f = Fixture::new($field, k);
                let ring = f.ring();
                let elements = f.import(&ring);
                bencher.bench_local(|| Fixture::sums(&ring, black_box(&elements)));
            }

            #[divan::bench(args = SIZES)]
            fn mul_pairs(bencher: divan::Bencher, k: usize) {
                let f = Fixture::new($field, k);
                let ring = f.ring();
                let elements = f.import(&ring);
                bencher.bench_local(|| Fixture::products(&ring, black_box(&elements)));
            }
        }
    };
}

basis_benchmarks!(basis_zp, Zp64::new(PRIME));
basis_benchmarks!(basis_q, Q);

#[divan::bench_group(sample_count = 10, sample_size = 1, skip_ext_time)]
mod zp_elimination {
    use super::{zp::*, *};

    #[divan::bench(args = CASES)]
    fn partial_fraction_cold(bencher: divan::Bencher, case: Case) {
        let f = Fixture::new(case);
        bencher.bench_local(|| {
            let (ring, coefficients) = f.partial_fractions();
            eliminate(&ring, black_box(&coefficients), case.steps)
        });
    }

    #[divan::bench(args = CASES)]
    fn partial_fraction(bencher: divan::Bencher, case: Case) {
        let (ring, coefficients) = Fixture::new(case).partial_fractions();
        bencher.bench_local(|| eliminate(&ring, black_box(&coefficients), case.steps));
    }

    #[divan::bench(args = CASES)]
    fn rational_polynomial(bencher: divan::Bencher, case: Case) {
        let f = Fixture::new(case);
        let coefficients = f.rational();
        let field = RationalPolynomialField::new(f.field.clone());
        bencher.bench_local(|| eliminate(&field, black_box(&coefficients), case.steps));
    }

    #[divan::bench(args = CASES)]
    fn factorized_rational_polynomial(bencher: divan::Bencher, case: Case) {
        let f = Fixture::new(case);
        let coefficients = f.factorized();
        let field = f.factorized_field();
        bencher.bench_local(|| eliminate(&field, black_box(&coefficients), case.steps));
    }
}

#[divan::bench_group(sample_count = 10, sample_size = 1, skip_ext_time)]
mod zp_sum {
    use super::{zp::*, *};

    #[divan::bench(args = CASES)]
    fn partial_fraction(bencher: divan::Bencher, case: Case) {
        let (ring, coefficients) = Fixture::new(case).partial_fractions();
        let terms = products(&ring, &coefficients);
        bencher
            .with_inputs(|| terms.clone())
            .bench_local_values(|terms| ring.sum(black_box(terms)));
    }

    #[divan::bench(args = CASES)]
    fn partial_fraction_fold(bencher: divan::Bencher, case: Case) {
        let (ring, coefficients) = Fixture::new(case).partial_fractions();
        let terms = products(&ring, &coefficients);
        bencher
            .with_inputs(|| terms.clone())
            .bench_local_values(|terms| fold(&ring, black_box(terms)));
    }

    #[divan::bench(args = CASES)]
    fn rational_polynomial(bencher: divan::Bencher, case: Case) {
        let f = Fixture::new(case);
        let field = RationalPolynomialField::new(f.field.clone());
        let terms = products(&field, &f.rational());
        bencher
            .with_inputs(|| terms.clone())
            .bench_local_values(|terms| field.sum(black_box(terms)));
    }
}

/// The benchmarks of the elimination and sum workloads over `Q(y)`, for the
/// cases `$cases`.
macro_rules! nested_benchmarks {
    ($elimination:ident, $sum:ident, $cases:path) => {
        #[divan::bench_group(sample_count = 10, sample_size = 1, skip_ext_time)]
        mod $elimination {
            use super::{nested::*, *};

            #[divan::bench(args = $cases)]
            fn partial_fraction(bencher: divan::Bencher, case: Case) {
                let (ring, coefficients) = Fixture::new(case).ordinary_partial_fractions();
                bencher.bench_local(|| eliminate(&ring, black_box(&coefficients), case.steps));
            }

            #[divan::bench(args = $cases)]
            fn partial_fraction_factorized(bencher: divan::Bencher, case: Case) {
                let (ring, coefficients) = Fixture::new(case).factorized_partial_fractions();
                bencher.bench_local(|| eliminate(&ring, black_box(&coefficients), case.steps));
            }

            #[divan::bench(args = $cases)]
            fn rational_polynomial(bencher: divan::Bencher, case: Case) {
                let f = Fixture::new(case);
                let coefficients = f.flat_ordinary();
                bencher
                    .bench_local(|| eliminate(&f.ordinary, black_box(&coefficients), case.steps));
            }

            #[divan::bench(args = $cases)]
            fn factorized_rational_polynomial(bencher: divan::Bencher, case: Case) {
                let f = Fixture::new(case);
                let coefficients = f.flat_factorized();
                bencher.bench_local(|| eliminate(&f.flat, black_box(&coefficients), case.steps));
            }
        }

        #[divan::bench_group(sample_count = 10, sample_size = 1, skip_ext_time)]
        mod $sum {
            use super::{nested::*, *};

            #[divan::bench(args = &$cases[..1])]
            fn partial_fraction(bencher: divan::Bencher, case: Case) {
                let (ring, coefficients) = Fixture::new(case).ordinary_partial_fractions();
                let terms = products(&ring, &coefficients);
                bencher
                    .with_inputs(|| terms.clone())
                    .bench_local_values(|terms| ring.sum(black_box(terms)));
            }

            #[divan::bench(args = &$cases[..1])]
            fn partial_fraction_fold(bencher: divan::Bencher, case: Case) {
                let (ring, coefficients) = Fixture::new(case).ordinary_partial_fractions();
                let terms = products(&ring, &coefficients);
                bencher
                    .with_inputs(|| terms.clone())
                    .bench_local_values(|terms| fold(&ring, black_box(terms)));
            }

            #[divan::bench(args = &$cases[..1])]
            fn rational_polynomial(bencher: divan::Bencher, case: Case) {
                let f = Fixture::new(case);
                let terms = products(&f.ordinary, &f.flat_ordinary());
                bencher
                    .with_inputs(|| terms.clone())
                    .bench_local_values(|terms| f.ordinary.sum(black_box(terms)));
            }
        }
    };
}

nested_benchmarks!(dimension_elimination, dimension_sum, nested::DIMENSION);
nested_benchmarks!(kinematic_elimination, kinematic_sum, nested::KINEMATIC);
