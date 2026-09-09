//! Black-box rational function reconstruction over a 64-bit prime field.
//!
//! Two experimental, sequential implementations share Thiele interpolation and
//! Symbolica polynomial arithmetic: Cuyt–Lee with Zippel interpolation (the
//! algorithm used by Kira through FireFly, arXiv:1904.00009, section 2.2), and
//! balanced Zippel (Smirnov–Zeng, arXiv:2409.19099, section 2.5).
//! Results are probabilistic: fresh probes check the result and failed attempts
//! restart with new anchors. Degree and probe limits make failure bounded.

use crate::{
    domains::{
        Field, Ring, RingOps,
        finite_field::{FiniteFieldCore, FiniteFieldElement, Zp64},
        rational_polynomial::{FromNumeratorAndDenominator, RationalPolynomial},
    },
    poly::{PolyVariable, polynomial::MultivariatePolynomial, univariate::UnivariatePolynomial},
    tensors::matrix::Matrix,
};
use rand::{Rng, SeedableRng, rngs::StdRng};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

mod automatic;
mod rational;
mod sparse_row;
mod support;
pub use rational::{RationalReconstructionStats, reconstruct_rational_function_over_q};

type Element = FiniteFieldElement<u64>;
type Polynomial = MultivariatePolynomial<Zp64, u16>;
type Fraction = RationalPolynomial<Zp64, u16>;

/// Reconstruction strategy. These are implementations, not external-code bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReconstructionMethod {
    /// Choose from learned slice structure and estimated dense interpolation
    /// costs. Selection probes count toward the shared budget. Uses balanced
    /// Zippel directly for one or two variables.
    Automatic,
    /// Homogenization, Thiele degree discovery, linear solves and polynomial Zippel.
    CuytLee,
    /// Reconstruct inexpensive shifted homogeneous components first, removing
    /// completed components from later line solves. Removes learned monomial
    /// factors and tries sparse shifts first.
    CuytLeePruned,
    /// Thiele degree discovery, degree-bounded rows and balanced sparse lifting.
    BalancedZippel,
    /// Try a denominator separable in the last variable, validate the result,
    /// and fall back to ordinary balanced Zippel if that hypothesis fails.
    BalancedZippelSeparated,
}

/// Resource bounds and reproducible random sampling controls.
#[derive(Clone, Debug)]
pub struct ReconstructionOptions {
    /// Maximum degree of either univariate numerator or denominator.
    /// For Cuyt–Lee this bounds total degree, for balanced Zippel individual degree.
    pub max_degree: u16,
    /// Maximum number of distinct black-box calls, including poles and retries.
    pub max_probes: usize,
    /// Number of fresh successful checks for early termination and final validation.
    pub verification_points: usize,
    /// Maximum number of attempts with independently sampled anchors.
    pub max_attempts: usize,
    /// Seed for reproducibility; use different seeds for independent confirmations.
    pub seed: u64,
    /// Race Thiele against unbalanced candidates with a small numerator or
    /// denominator, using the same probes at additional arithmetic cost.
    pub degree_race: bool,
    /// Reuse learned support and coefficient hypotheses when lifting over Q.
    /// Reduces probes at additional interpolation cost; unused for one prime.
    pub reuse_coefficients: bool,
    /// Reuse univariate powers observed in an earlier balanced row when they
    /// are sparse enough to save probes, with fresh row checks and fallback.
    pub reuse_row_support: bool,
}

impl Default for ReconstructionOptions {
    fn default() -> Self {
        Self {
            max_degree: 128,
            max_probes: 1_000_000,
            verification_points: 3,
            max_attempts: 4,
            seed: 0x7265636f6e737472,
            degree_race: false,
            reuse_coefficients: true,
            reuse_row_support: true,
        }
    }
}

/// Costs include unsuccessful attempts and validation. Cached probes are free.
#[derive(Clone, Debug, Default)]
pub struct ReconstructionStats {
    /// Method used by the successful attempt (including automatic selection).
    pub selected_method: Option<ReconstructionMethod>,
    /// Oracle calls spent selecting a method, including rejected pilots.
    pub selection_probes: usize,
    /// Accepted sparse-row hypotheses, including rows in later-failed attempts.
    pub sparse_rows: usize,
    /// Rejected sparse rows followed by the existing degree-bounded solver.
    pub sparse_row_fallbacks: usize,
    pub probes: usize,
    pub poles: usize,
    pub cache_hits: usize,
    pub univariate_interpolations: usize,
    /// Univariate rows interpolated using degrees learned from an earlier row.
    pub degree_interpolations: usize,
    /// Rejected separable-denominator candidates followed by ordinary balancing.
    pub separation_fallbacks: usize,
    pub linear_solves: usize,
    pub attempts: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReconstructionError {
    InvalidOptions,
    ProbeLimit,
    /// Too few prime images to lift and independently validate the coefficients.
    PrimeLimit,
    /// Degree limit, exceptional specializations, or verification failure.
    AttemptsExhausted,
}

impl std::fmt::Display for ReconstructionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidOptions => "reconstruction requires variables, positive limits, and a sufficiently large prime field",
            Self::ProbeLimit => "rational reconstruction exhausted its black-box probe budget",
            Self::PrimeLimit => "rational reconstruction exhausted its prime budget",
            Self::AttemptsExhausted => "rational reconstruction failed: increase the degree/attempt limit or change the seed/prime",
        })
    }
}
impl std::error::Error for ReconstructionError {}

type Result<T> = std::result::Result<T, ReconstructionError>;

/// Reconstruct a rational function from evaluations only. `None` reports a pole
/// or an otherwise unusable evaluation. The oracle must return consistent values
/// in `field`. Variable order is significant for performance.
///
/// `field` must have an odd prime modulus (as required by `Zp64`); this function
/// does not prove primality. No degrees or monomial supports are supplied by the
/// caller. Returned numerator and denominator are coprime and denominator-monic.
/// Three random checks are the default, not a deterministic identity proof.
pub fn reconstruct_rational_function<F>(
    field: Zp64,
    variables: Arc<Vec<PolyVariable>>,
    mut black_box: F,
    method: ReconstructionMethod,
    options: &ReconstructionOptions,
) -> Result<(Fraction, ReconstructionStats)>
where
    F: FnMut(&Zp64, &[Element]) -> Option<Element>,
{
    if variables.is_empty()
        || options.max_degree == 0
        || options.max_degree > u16::MAX / 2
        || options.max_probes == 0
        || options.verification_points == 0
        || options.max_attempts == 0
        || field.get_prime()
            < 8 * (options.max_degree as u64 + options.verification_points as u64 + 1)
    {
        return Err(ReconstructionError::InvalidOptions);
    }
    let template = Polynomial::new(&field, None, variables);
    let mut ctx = Context {
        field,
        template,
        black_box: &mut black_box,
        options,
        rng: StdRng::seed_from_u64(options.seed),
        cache: HashMap::new(),
        stats: Default::default(),
        monomial_factors: None,
    };
    for attempt in 0..options.max_attempts {
        ctx.stats.attempts = attempt + 1;
        let (selected, mut profile) = if method == ReconstructionMethod::Automatic {
            let before = ctx.stats.probes;
            let choice = ctx.select_method();
            ctx.stats.selection_probes += ctx.stats.probes - before;
            match choice {
                Ok(choice) => choice,
                Err(ReconstructionError::ProbeLimit) => {
                    return Err(ReconstructionError::ProbeLimit);
                }
                // A failed degree forecast need not prevent balanced reconstruction.
                Err(_) => (ReconstructionMethod::BalancedZippel, None),
            }
        } else {
            (method, None)
        };
        ctx.stats.selected_method = Some(selected);
        let candidates: &[bool] = if selected == ReconstructionMethod::BalancedZippelSeparated {
            &[true, false]
        } else {
            &[false]
        };
        for &separate in candidates {
            let result = match selected {
                ReconstructionMethod::Automatic => unreachable!(),
                ReconstructionMethod::CuytLee => ctx.cuyt_lee(false),
                ReconstructionMethod::CuytLeePruned => {
                    if let Some(profile) = profile.take() {
                        ctx.cuyt_lee_profile(true, profile.bounds, profile.minimum)
                    } else {
                        ctx.cuyt_lee(true)
                    }
                }
                ReconstructionMethod::BalancedZippel
                | ReconstructionMethod::BalancedZippelSeparated => ctx.balanced(separate),
            };
            match result {
                Ok(r) if ctx.verify(&r)? => {
                    let r = Fraction::from_num_den(r.numerator, r.denominator, &ctx.field, true);
                    return Ok((r, ctx.stats));
                }
                Err(ReconstructionError::ProbeLimit) => {
                    return Err(ReconstructionError::ProbeLimit);
                }
                _ => {}
            }
            if separate {
                ctx.stats.separation_fallbacks += 1;
                ctx.stats.selected_method = Some(ReconstructionMethod::BalancedZippel);
            }
        }
    }
    Err(ReconstructionError::AttemptsExhausted)
}

struct Context<'a, F> {
    field: Zp64,
    template: Polynomial,
    black_box: &'a mut F,
    options: &'a ReconstructionOptions,
    rng: StdRng,
    cache: HashMap<Vec<Element>, Option<Element>>,
    stats: ReconstructionStats,
    // Only active inside the pruned homogeneous reconstruction. The cache and
    // validation always retain the original oracle values.
    monomial_factors: Option<[Vec<u16>; 2]>,
}

fn unlucky<T>() -> Result<T> {
    Err(ReconstructionError::AttemptsExhausted)
}

fn coefficient(p: &Polynomial, variable: usize, degree: u16) -> Element {
    p.into_iter()
        .find(|m| m.exponents[variable] == degree)
        .map_or_else(|| p.ring().zero(), |m| *m.coefficient)
}

fn value(r: &Fraction, point: &[Element]) -> Option<Element> {
    let f = r.numerator.ring();
    let d = r.denominator.replace_all(point);
    (!f.is_zero(&d)).then(|| f.div(&r.numerator.replace_all(point), &d))
}

impl<F: FnMut(&Zp64, &[Element]) -> Option<Element>> Context<'_, F> {
    fn random(&mut self) -> Element {
        self.field
            .to_element(self.rng.random_range(1..self.field.get_prime()))
    }
    fn point(&mut self) -> Vec<Element> {
        (0..self.template.nvars()).map(|_| self.random()).collect()
    }
    fn probe(&mut self, point: &[Element]) -> Result<Option<Element>> {
        let Some(mut value) = self.probe_raw(point)? else {
            return Ok(None);
        };
        if let Some(powers) = &self.monomial_factors {
            let f = &self.field;
            for ((x, n), d) in point.iter().zip(&powers[0]).zip(&powers[1]) {
                if n != d && f.is_zero(x) {
                    // A removable zero or pole needs a limit, which a raw
                    // black-box value does not supply. Use another point.
                    return Ok(None);
                }
                if n > d {
                    value = f.div(&value, &f.pow(x, (n - d) as u64));
                } else if d > n {
                    value = f.mul(&value, &f.pow(x, (d - n) as u64));
                }
            }
        }
        Ok(Some(value))
    }
    fn probe_raw(&mut self, point: &[Element]) -> Result<Option<Element>> {
        if let Some(v) = self.cache.get(point) {
            self.stats.cache_hits += 1;
            return Ok(*v);
        }
        if self.stats.probes >= self.options.max_probes {
            return Err(ReconstructionError::ProbeLimit);
        }
        self.stats.probes += 1;
        let v = (self.black_box)(&self.field, point);
        self.stats.poles += usize::from(v.is_none());
        self.cache.insert(point.to_vec(), v);
        Ok(v)
    }
    fn verify(&mut self, r: &Fraction) -> Result<bool> {
        let mut good = 0;
        for _ in 0..self.options.verification_points * 16 {
            let point = self.point();
            if self.cache.contains_key(&point) {
                continue;
            }
            if let Some(v) = self.probe_raw(&point)? {
                if value(r, &point) != Some(v) {
                    return Ok(false);
                }
                good += 1;
                if good == self.options.verification_points {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    // Eq. (22)-(25) of FireFly. Convergents are built with Symbolica polynomials.
    fn thiele(
        &mut self,
        variable: usize,
        map: impl Fn(Element) -> Vec<Element>,
    ) -> Result<Fraction> {
        self.thiele_seeded(variable, map, None)
    }

    fn thiele_seeded(
        &mut self,
        variable: usize,
        map: impl Fn(Element) -> Vec<Element>,
        mut known: Option<(Element, Element)>,
    ) -> Result<Fraction> {
        self.stats.univariate_interpolations += 1;
        let f = self.field.clone();
        let mut nodes = Vec::new();
        let mut differences = Vec::new();
        let mut seen = HashSet::new();
        let dense = UnivariatePolynomial::new(
            &f,
            None,
            Arc::new(self.template.variables()[variable].clone()),
        );
        let mut p_prev = dense.one();
        let mut q_prev = dense.zero();
        let mut p = dense.zero();
        let mut q = dense.one();
        let mut checks = 0;
        let mut race = self
            .options
            .degree_race
            .then(|| DegreeRace::new(&dense, self.options.max_degree as usize));
        let mut reciprocal_race = self
            .options
            .degree_race
            .then(|| DegreeRace::new(&dense, self.options.max_degree as usize));
        for _ in 0..8 * (self.options.max_degree as usize + self.options.verification_points + 8) {
            // An already reconstructed slice supplies its intersection with
            // this row. This is derived data, not another black-box call.
            let sample = known.take();
            let t = sample.map_or_else(|| self.random(), |(t, _)| t);
            if !seen.insert(t) {
                continue;
            }
            let Some(y) = (if let Some((_, y)) = sample {
                Some(y)
            } else {
                self.probe(&map(t))?
            }) else {
                continue;
            };
            if let Some((p, q)) = race
                .as_mut()
                .and_then(|race| race.add(t, y, self.options.verification_points))
            {
                return Ok(Fraction::from_num_den(
                    embed(&self.template, variable, p),
                    embed(&self.template, variable, q),
                    &f,
                    true,
                ));
            }
            // The reciprocal covers large denominator degree with a small
            // numerator. Its poles are zeros of the original oracle; skip them
            // without extra calls. Final full-dimensional checks are unchanged.
            if !f.is_zero(&y)
                && let Some((p, q)) = reciprocal_race
                    .as_mut()
                    .and_then(|race| race.add(t, f.inv(&y), self.options.verification_points))
            {
                return Ok(Fraction::from_num_den(
                    embed(&self.template, variable, q),
                    embed(&self.template, variable, p),
                    &f,
                    true,
                ));
            }
            let dv = q.evaluate(&t);
            if !nodes.is_empty() && !f.is_zero(&dv) && f.mul(&y, &dv) == p.evaluate(&t) {
                checks += 1;
                if checks == self.options.verification_points {
                    return Ok(Fraction::from_num_den(
                        embed(&self.template, variable, p),
                        embed(&self.template, variable, q),
                        &f,
                        true,
                    ));
                }
                continue;
            }
            checks = 0;
            let mut b = y;
            let mut valid = true;
            for (x, a) in nodes.iter().zip(&differences) {
                let den = f.sub(&b, a);
                if f.is_zero(&den) {
                    valid = false;
                    break;
                }
                b = f.div(&f.sub(&t, x), &den);
            }
            if !valid {
                continue;
            }
            if nodes.is_empty() {
                p = dense.constant(b);
            } else {
                let factor = &dense.monomial(f.one(), 1) - &dense.constant(*nodes.last().unwrap());
                let next_p = p.clone().mul_coeff(&b) + &factor * &p_prev;
                let next_q = q.clone().mul_coeff(&b) + &factor * &q_prev;
                p_prev = p;
                q_prev = q;
                p = next_p;
                q = next_q;
            }
            nodes.push(t);
            differences.push(b);
            if p.degree() > self.options.max_degree as usize
                || q.degree() > self.options.max_degree as usize
            {
                return unlucky();
            }
        }
        unlucky()
    }

    // Rational interpolation at learned degrees: interpolate the samples as a
    // polynomial, then use a partial extended Euclidean algorithm to recover
    // P/Q modulo the product of (x-x_i). This costs O((deg P + deg Q)^2)
    // field operations rather than a dense rational linear solve.
    fn degree_row(
        &mut self,
        variable: usize,
        point: &[Element],
        known: Element,
        degrees: (u16, u16),
        min_degrees: (u16, u16),
        denominator: Option<&Polynomial>,
        reciprocal: bool,
        powers: &[Vec<u16>; 2],
    ) -> Result<Fraction> {
        self.stats.degree_interpolations += 1;
        let ordered: [&[u16]; 2] = if reciprocal {
            [&powers[1], &powers[0]]
        } else {
            [&powers[0], &powers[1]]
        };
        let sparse_count = ordered[0].len()
            + if denominator.is_some() {
                0
            } else {
                ordered[1].len().saturating_sub(1)
            };
        let dense_count = usize::from(degrees.0 - min_degrees.0)
            + if denominator.is_some() {
                0
            } else {
                usize::from(degrees.1 - min_degrees.1)
            }
            + 1;
        if self.options.reuse_row_support
            && 2 * (sparse_count + self.options.verification_points) < dense_count
        {
            match self.sparse_row(variable, point, known, ordered, denominator, reciprocal) {
                Ok(row) => {
                    self.stats.sparse_rows += 1;
                    return Ok(row);
                }
                Err(ReconstructionError::ProbeLimit) => {
                    return Err(ReconstructionError::ProbeLimit);
                }
                Err(_) => self.stats.sparse_row_fallbacks += 1,
            }
        }
        let f = self.field.clone();
        let dense = UnivariatePolynomial::new(
            &f,
            None,
            Arc::new(self.template.variables()[variable].clone()),
        );
        let mut interpolant = dense.zero();
        let mut modulus = dense.one();
        // A monomial factor seen in the first generic row predicts zero low
        // coefficients in later rows. Remove it from the interpolation problem;
        // the unchanged final identity checks guard exceptional specializations.
        let degrees = (degrees.0 - min_degrees.0, degrees.1 - min_degrees.1);
        let known_denominator = denominator.map(|p| {
            let cs = (min_degrees.1..=p.degree(variable))
                .map(|d| coefficient(p, variable, d))
                .collect();
            UnivariatePolynomial::from_coefficients(&f, cs, dense.variable.clone())
        });
        let count = degrees.0 as usize
            + if denominator.is_some() {
                0
            } else {
                degrees.1 as usize
            }
            + 1;
        let mut samples = Vec::with_capacity(count);
        let mut seen = HashSet::new();
        for _ in 0..count * 16 + 16 {
            let (t, y) = if samples.is_empty() {
                seen.insert(point[variable]);
                if reciprocal && f.is_zero(&known) {
                    return unlucky();
                }
                (
                    point[variable],
                    if reciprocal { f.inv(&known) } else { known },
                )
            } else {
                let t = self.random();
                if !seen.insert(t) {
                    continue;
                }
                let mut p = point.to_vec();
                p[variable] = t;
                let Some(y) = self.probe(&p)? else {
                    continue;
                };
                if reciprocal && f.is_zero(&y) {
                    continue;
                }
                (t, if reciprocal { f.inv(&y) } else { y })
            };
            let y = if min_degrees.0 != 0 {
                f.div(&y, &f.pow(&t, min_degrees.0 as u64))
            } else {
                y
            };
            let y = if min_degrees.1 != 0 {
                f.mul(&y, &f.pow(&t, min_degrees.1 as u64))
            } else {
                y
            };
            let polynomial_value = known_denominator
                .as_ref()
                .map_or(y, |d| f.mul(&y, &d.evaluate(&t)));
            let c = f.div(
                &f.sub(&polynomial_value, &interpolant.evaluate(&t)),
                &modulus.evaluate(&t),
            );
            if !f.is_zero(&c) {
                interpolant = interpolant + modulus.clone().mul_coeff(&c);
            }
            modulus = modulus * &(dense.monomial(f.one(), 1) - dense.constant(t));
            samples.push((t, y));
            if samples.len() == count {
                break;
            }
        }
        if samples.len() != count {
            return unlucky();
        }
        if let Some(denominator) = denominator {
            return Ok(Fraction {
                numerator: embed_shifted(&self.template, variable, interpolant, min_degrees.0),
                denominator: denominator.clone(),
            });
        }
        let (mut r0, mut r1) = (modulus, interpolant);
        let (mut q0, mut q1) = (dense.zero(), dense.one());
        while !r1.is_zero() && r1.degree() > degrees.0 as usize {
            let (quotient, remainder) = r0.quot_rem(&r1);
            (r0, r1) = (r1, remainder);
            let next = q0 - &quotient * &q1;
            (q0, q1) = (q1, next);
        }
        if q1.is_zero() || q1.degree() > degrees.1 as usize {
            return unlucky();
        }
        // These checks use the existing samples. Fresh full-dimensional checks
        // remain mandatory before any reconstruction is returned to the caller.
        if samples.iter().any(|(t, y)| {
            let d = q1.evaluate(t);
            f.is_zero(&d) || r1.evaluate(t) != f.mul(y, &d)
        }) {
            return unlucky();
        }
        Ok(Fraction::from_num_den(
            embed_shifted(&self.template, variable, r1, min_degrees.0),
            embed_shifted(&self.template, variable, q1, min_degrees.1),
            &f,
            true,
        ))
    }

    fn balanced(&mut self, separate: bool) -> Result<Fraction> {
        let anchors = self.point();
        let mut result = self.thiele(0, |t| {
            let mut p = anchors.clone();
            p[0] = t;
            p
        })?;
        for variable in 1..self.template.nvars() {
            let base = self.point();
            let polys = [&result.numerator, &result.denominator];
            let z = polys.iter().map(|p| p.nterms()).max().unwrap();
            let mut nodes = Vec::new();
            for p in polys {
                let x = monomial_nodes(p, &base, variable);
                distinct_nonzero(&self.field, &x)?;
                nodes.push(x);
            }
            let mut rows: Vec<Fraction> = Vec::new();
            let mut degrees: Option<(u16, u16)> = None;
            let mut min_degrees = (0, 0);
            let mut row_powers = [Vec::new(), Vec::new()];
            let mut completed: [Option<Polynomial>; 2] = [None, None];
            for i in 1..=z {
                let mut point = anchors.clone();
                for j in 0..variable {
                    point[j] = self.field.pow(&base[j], i as u64);
                }
                let known = value(&result, &point).ok_or(ReconstructionError::AttemptsExhausted)?;
                let mut row = if let Some(degrees) = degrees {
                    // The first row predicts the last-variable denominator
                    // factor. Fresh final verification guards this hypothesis.
                    let fixed = completed[1].as_ref().map(|p| (false, p)).or_else(|| {
                        completed[0]
                            .as_ref()
                            .filter(|p| !p.is_zero())
                            .map(|p| (true, p))
                    });
                    let specialized = fixed.map(|(inverse, p)| {
                        let mut p = p.clone();
                        for (j, x) in point.iter().enumerate().take(variable) {
                            p = p.replace(j, x);
                        }
                        (inverse, p)
                    });
                    let reciprocal = specialized.as_ref().is_some_and(|(inverse, _)| *inverse);
                    let denominator = if let Some((_, p)) = &specialized {
                        Some(p)
                    } else if separate && variable + 1 == self.template.nvars() {
                        Some(&rows[0].denominator)
                    } else {
                        None
                    };
                    let degrees = if reciprocal {
                        (degrees.1, degrees.0)
                    } else {
                        degrees
                    };
                    let min_degrees = if reciprocal {
                        (min_degrees.1, min_degrees.0)
                    } else {
                        min_degrees
                    };
                    let mut row = self.degree_row(
                        variable,
                        &point,
                        known,
                        degrees,
                        min_degrees,
                        denominator,
                        reciprocal,
                        &row_powers,
                    )?;
                    if reciprocal {
                        std::mem::swap(&mut row.numerator, &mut row.denominator);
                    }
                    row
                } else {
                    self.thiele_seeded(
                        variable,
                        |t| {
                            let mut p = point.clone();
                            p[variable] = t;
                            p
                        },
                        Some((point[variable], known)),
                    )?
                };
                if degrees.is_none() {
                    for (side, p) in [&row.numerator, &row.denominator].into_iter().enumerate() {
                        row_powers[side] = p.into_iter().map(|m| m.exponents[variable]).collect();
                        row_powers[side].sort_unstable();
                    }
                    let min = |p: &Polynomial| {
                        p.into_iter()
                            .map(|m| m.exponents[variable])
                            .min()
                            .unwrap_or(0)
                    };
                    min_degrees = (min(&row.numerator), min(&row.denominator));
                }
                degrees.get_or_insert((
                    row.numerator.degree(variable),
                    row.denominator.degree(variable),
                ));
                let den = row.denominator.replace_all(&point);
                let num = row.numerator.replace_all(&point);
                let scale = if !self.field.is_zero(&den) {
                    self.field
                        .div(&result.denominator.replace_all(&point), &den)
                } else if !self.field.is_zero(&num) {
                    self.field.div(&result.numerator.replace_all(&point), &num)
                } else {
                    return unlucky();
                };
                if self.field.is_zero(&scale) {
                    return unlucky();
                }
                row.numerator = row.numerator.mul_coeff(scale);
                row.denominator = row.denominator.mul_coeff(scale);
                rows.push(row);
                for side in 0..2 {
                    if completed[side].is_none() && rows.len() >= polys[side].nterms() {
                        completed[side] = Some(lift_rows(
                            &self.template,
                            polys[side],
                            &rows,
                            &nodes[side],
                            variable,
                            side,
                        ));
                    }
                }
            }
            result = Fraction {
                numerator: completed[0].take().unwrap(),
                denominator: completed[1].take().unwrap(),
            };
        }
        Ok(result)
    }

    fn cuyt_lee(&mut self, prune: bool) -> Result<Fraction> {
        if self.template.nvars() == 1 {
            return self.thiele(0, |t| vec![t]);
        }
        let degree_anchor = self.point();
        let mut bounds = [Vec::new(), Vec::new()];
        let mut minimum = [Vec::new(), Vec::new()];
        for variable in 0..self.template.nvars() {
            let slice = self.thiele(variable, |t| {
                let mut p = degree_anchor.clone();
                p[variable] = t;
                p
            })?;
            bounds[0].push(slice.numerator.degree(variable));
            bounds[1].push(slice.denominator.degree(variable));
            for (side, p) in [&slice.numerator, &slice.denominator]
                .into_iter()
                .enumerate()
            {
                let lo = if prune {
                    p.into_iter()
                        .map(|m| m.exponents[variable])
                        .min()
                        .unwrap_or(0)
                } else {
                    0
                };
                minimum[side].push(lo);
                bounds[side][variable] -= lo;
            }
        }
        self.cuyt_lee_profile(prune, bounds, minimum)
    }

    fn cuyt_lee_profile(
        &mut self,
        prune: bool,
        bounds: [Vec<u16>; 2],
        minimum: [Vec<u16>; 2],
    ) -> Result<Fraction> {
        let f = self.field.clone();
        // Generic slices expose factors shared by every term. Reduce the
        // degrees before homogenization, without additional discovery probes.
        self.monomial_factors = minimum.iter().flatten().any(|d| *d != 0).then_some(minimum);
        let result = self.cuyt_lee_bounded(prune, &bounds);
        // Clear the transformation on errors too, before another attempt can
        // reuse the original cached probes for degree discovery.
        let factors = self.monomial_factors.take();
        result.map(|mut r| {
            if let Some([n, d]) = factors {
                r.numerator = &r.numerator * &self.template.monomial(f.one(), n);
                r.denominator = &r.denominator * &self.template.monomial(f.one(), d);
            }
            r
        })
    }

    fn cuyt_lee_bounded(&mut self, prune: bool, bounds: &[Vec<u16>; 2]) -> Result<Fraction> {
        let f = self.field.clone();
        let mut shift = vec![f.zero(); self.template.nvars()];
        if self.probe(&shift)?.is_none() {
            if prune {
                let trial = self.point();
                let mut found = false;
                for i in 0..shift.len() {
                    shift[i] = trial[i];
                    if self.probe(&shift)?.is_some() {
                        found = true;
                        break;
                    }
                    shift[i] = f.zero();
                }
                if !found {
                    for i in 0..shift.len() {
                        shift[i] = trial[i];
                        if self.probe(&shift)?.is_some() {
                            found = true;
                            break;
                        }
                    }
                }
                if !found {
                    return unlucky();
                }
            } else {
                shift = self.point();
            }
        }
        let mut anchors = self.point();
        anchors[0] = f.one();
        let first = self.thiele(0, |t| line_point(&f, &shift, &anchors, t))?;
        let nd = first.numerator.degree(0);
        let dd = first.denominator.degree(0);
        if let Some(factors) = &self.monomial_factors {
            for (degree, powers) in [nd, dd].into_iter().zip(factors) {
                if degree as u64 + powers.iter().map(|d| *d as u64).sum::<u64>()
                    > self.options.max_degree as u64
                {
                    return unlucky();
                }
            }
        }
        let first = normalize_constant(first)?;
        if prune {
            return self.cuyt_lee_components(&shift, &anchors, first, bounds);
        }
        let num_powers: Vec<_> = (&first.numerator)
            .into_iter()
            .map(|m| m.exponents[0])
            .collect();
        let den_powers: Vec<_> = (&first.denominator)
            .into_iter()
            .map(|m| m.exponents[0])
            .filter(|d| *d != 0)
            .collect();
        let mut lines = HashMap::from([(anchors.clone(), first)]);
        let mut reconstructed = Vec::new();
        for (side, degree) in [(0, nd), (1, dd)] {
            let mut output = self.template.zero();
            let mut corrections = self.template.zero();
            for d in (0..=degree).rev() {
                let template = self.template.clone();
                let correction = homogeneous_part(&corrections, d);
                let component = zippel(&template, &anchors, d, &bounds[side], 0, |direction| {
                    if !lines.contains_key(direction) {
                        let r =
                            self.line_solve(&shift, direction, &num_powers, &den_powers, None)?;
                        lines.insert(direction.to_vec(), r);
                    }
                    let r = &lines[direction];
                    let p = if side == 0 {
                        &r.numerator
                    } else {
                        &r.denominator
                    };
                    Ok(f.sub(&coefficient(p, 0, d), &correction.replace_all(direction)))
                })?;
                let mut homogeneous = self.template.zero();
                for m in &component {
                    let sum: u32 = m.exponents.iter().skip(1).map(|e| *e as u32).sum();
                    if sum > d as u32 {
                        return unlucky();
                    }
                    let mut ex = m.exponents.to_vec();
                    ex[0] = d - sum as u16;
                    homogeneous.append_monomial(*m.coefficient, &ex);
                }
                // Remove shifts degree by degree to preserve the original sparse support.
                let translated = translate(&homogeneous, &shift);
                corrections = corrections + (&translated - &homogeneous);
                output = output + homogeneous;
            }
            reconstructed.push(output);
        }
        Ok(Fraction {
            numerator: reconstructed.remove(0),
            denominator: reconstructed.remove(0),
        })
    }

    fn cuyt_lee_components(
        &mut self,
        shift: &[Element],
        anchors: &[Element],
        first: Fraction,
        bounds: &[Vec<u16>; 2],
    ) -> Result<Fraction> {
        let f = self.field.clone();
        let mut powers: [Vec<u16>; 2] = [
            (&first.numerator)
                .into_iter()
                .map(|m| m.exponents[0])
                .collect(),
            (&first.denominator)
                .into_iter()
                .map(|m| m.exponents[0])
                .filter(|d| *d != 0)
                .collect(),
        ];
        for p in &mut powers {
            p.sort_unstable();
        }
        let mut lines = HashMap::from([(anchors.to_vec(), first)]);
        let mut known = [self.template.zero(), self.template.one()];
        // Degree bounds predict an upper bound on each component's support.
        // Small components on either side can finish before large ones need
        // new directions. No support is supplied by the source function.
        let sizes = [
            homogeneous_sizes(&bounds[0], powers[0].iter().copied().max().unwrap_or(0)),
            homogeneous_sizes(&bounds[1], powers[1].iter().copied().max().unwrap_or(0)),
        ];
        let mut components: Vec<_> = powers
            .iter()
            .enumerate()
            .flat_map(|(s, ds)| ds.iter().map(move |d| (s, *d)))
            .collect();
        let denominator_work = powers[1]
            .iter()
            .fold(0u64, |n, &d| n.saturating_add(sizes[1][d as usize]));
        let largest_numerator = powers[0]
            .iter()
            .map(|&d| sizes[0][d as usize])
            .max()
            .unwrap_or(0);
        // Finishing an inexpensive denominator avoids carrying its unknowns
        // through a large numerator component. Otherwise interleave both sides.
        let denominator_first = denominator_work < largest_numerator;
        components.sort_by_key(|&(side, d)| {
            (
                usize::from(denominator_first && side == 0),
                sizes[side][d as usize],
                d,
                1 - side,
            )
        });
        for (side, d) in components {
            let template = self.template.clone();
            let component = zippel(
                &template,
                anchors,
                d,
                &bounds[side],
                self.options.verification_points,
                |direction| {
                    if !lines.contains_key(direction) {
                        let fixed = [
                            homogeneous_line(&known[0], direction),
                            homogeneous_line(&known[1], direction),
                        ];
                        let r = self.line_solve(
                            shift,
                            direction,
                            &powers[0],
                            &powers[1],
                            Some(&fixed),
                        )?;
                        lines.insert(direction.to_vec(), r);
                    }
                    let r = &lines[direction];
                    Ok(coefficient(
                        if side == 0 {
                            &r.numerator
                        } else {
                            &r.denominator
                        },
                        0,
                        d,
                    ))
                },
            )?;
            for m in &component {
                let sum: u32 = m.exponents.iter().skip(1).map(|e| *e as u32).sum();
                if sum > d as u32 {
                    return unlucky();
                }
                let mut ex = m.exponents.to_vec();
                ex[0] = d - sum as u16;
                known[side].append_monomial(*m.coefficient, &ex);
            }
            powers[side].retain(|power| *power != d);
        }
        let inverse_shift: Vec<_> = shift.iter().map(|x| f.neg(x)).collect();
        Ok(Fraction {
            numerator: translate(&known[0], &inverse_shift),
            denominator: translate(&known[1], &inverse_shift),
        })
    }

    // FireFly Eq. (27), after Thiele has discovered the degrees. The constant
    // denominator coefficient fixes the scale on every homogenized line.
    fn line_solve(
        &mut self,
        shift: &[Element],
        direction: &[Element],
        num_powers: &[u16],
        den_powers: &[u16],
        known: Option<&[Vec<Element>; 2]>,
    ) -> Result<Fraction> {
        let f = self.field.clone();
        let n = num_powers.len() + den_powers.len();
        let degree = num_powers
            .iter()
            .chain(den_powers)
            .copied()
            .max()
            .unwrap_or(0) as usize;
        let degree = degree.max(known.map_or(0, |v| {
            v.iter().map(|p| p.len().saturating_sub(1)).max().unwrap()
        }));
        let mut data = Vec::new();
        let mut rhs = Vec::new();
        let mut seen = HashSet::new();
        for _ in 0..n * 16 + 16 {
            let t = self.random();
            if !seen.insert(t) {
                continue;
            }
            let Some(y) = self.probe(&line_point(&f, shift, direction, t))? else {
                continue;
            };
            let mut powers = vec![f.one(); degree + 1];
            for j in 1..powers.len() {
                powers[j] = f.mul(&powers[j - 1], &t);
            }
            data.extend(num_powers.iter().map(|j| powers[*j as usize]));
            data.extend(
                den_powers
                    .iter()
                    .map(|j| f.neg(&f.mul(&y, &powers[*j as usize]))),
            );
            rhs.push(if let Some(known) = known {
                let eval = |p: &[Element]| {
                    p.iter()
                        .zip(&powers)
                        .fold(f.zero(), |v, (c, t)| f.add(&v, &f.mul(c, t)))
                };
                f.sub(&f.mul(&y, &eval(&known[1])), &eval(&known[0]))
            } else {
                y
            });
            if rhs.len() == n {
                break;
            }
        }
        if rhs.len() != n {
            return unlucky();
        }
        self.stats.linear_solves += 1;
        let matrix = Matrix::from_linear(data, n as u32, n as u32, f.clone()).unwrap();
        let solution = matrix
            .solve(&Matrix::new_vec(rhs, f))
            .map_err(|_| ReconstructionError::AttemptsExhausted)?;
        let mut num = self.template.zero();
        let mut den = self.template.zero();
        if let Some(known) = known {
            for (p, cs) in [&mut num, &mut den].into_iter().zip(known) {
                for (d, c) in cs.iter().enumerate() {
                    let mut ex = vec![0; self.template.nvars()];
                    ex[0] = d as u16;
                    p.append_monomial(*c, &ex);
                }
            }
        } else {
            den = self.template.one();
        }
        for i in 0..n {
            let mut ex = vec![0; self.template.nvars()];
            ex[0] = if i < num_powers.len() {
                num_powers[i]
            } else {
                den_powers[i - num_powers.len()]
            };
            if i < num_powers.len() {
                num.append_monomial(solution[(i as u32, 0)], &ex);
            } else {
                den.append_monomial(solution[(i as u32, 0)], &ex);
            }
        }
        Ok(Fraction {
            numerator: num,
            denominator: den,
        })
    }
}

// Race Thiele against unbalanced approximants with small denominator degree.
// These use exactly the same samples. All winning candidates must satisfy the
// stored samples and the usual number of subsequent independent checks.
struct DegreeRace {
    interpolant: UnivariatePolynomial<Zp64>,
    modulus: UnivariatePolynomial<Zp64>,
    samples: Vec<(Element, Element)>,
    candidates: Vec<(
        UnivariatePolynomial<Zp64>,
        UnivariatePolynomial<Zp64>,
        usize,
    )>,
    max_degree: usize,
}

impl DegreeRace {
    fn new(template: &UnivariatePolynomial<Zp64>, max_degree: usize) -> Self {
        Self {
            interpolant: template.zero(),
            modulus: template.one(),
            samples: Vec::new(),
            candidates: Vec::new(),
            max_degree,
        }
    }
    fn add(
        &mut self,
        x: Element,
        y: Element,
        checks: usize,
    ) -> Option<(UnivariatePolynomial<Zp64>, UnivariatePolynomial<Zp64>)> {
        let f = self.interpolant.coefficient_ring().clone();
        self.candidates.retain_mut(|(p, q, good)| {
            let d = q.evaluate(&x);
            if f.is_zero(&d) || p.evaluate(&x) != f.mul(&y, &d) {
                return false;
            }
            *good += 1;
            true
        });
        for (p, q, good) in &self.candidates {
            if *good >= checks
                && self.samples.iter().all(|(t, v)| {
                    let d = q.evaluate(t);
                    !f.is_zero(&d) && p.evaluate(t) == f.mul(v, &d)
                })
            {
                return Some((p.clone(), q.clone()));
            }
        }
        let c = f.div(
            &f.sub(&y, &self.interpolant.evaluate(&x)),
            &self.modulus.evaluate(&x),
        );
        // mul_coeff(0) need not canonicalize the dense representation. Avoid
        // feeding an all-zero coefficient vector to Euclidean division.
        if !f.is_zero(&c) {
            self.interpolant = &self.interpolant + &self.modulus.clone().mul_coeff(&c);
        }
        self.modulus =
            &self.modulus * &(self.modulus.monomial(f.one(), 1) - self.modulus.constant(x));
        self.samples.push((x, y));
        let (mut r0, mut r1) = (self.modulus.clone(), self.interpolant.clone());
        let (mut q0, mut q1) = (self.modulus.zero(), self.modulus.one());
        // Balanced degrees are already handled efficiently by Thiele. The
        // bounded race targets polynomial-like rows without changing probes.
        while !r1.is_zero() && q1.degree() <= 32.min(self.max_degree) {
            if r1.degree() > q1.degree() + 1
                && r1.degree() <= self.max_degree
                && r1.degree() + q1.degree() < self.samples.len()
                && !self
                    .candidates
                    .iter()
                    .any(|(p, q, _)| p.degree() == r1.degree() && q.degree() == q1.degree())
            {
                self.candidates.push((r1.clone(), q1.clone(), 0));
            }
            let (a, r) = r0.quot_rem(&r1);
            (r0, r1) = (r1, r);
            let q = q0 - &a * &q1;
            (q0, q1) = (q1, q);
        }
        None
    }
}

fn lift_rows(
    template: &Polynomial,
    old: &Polynomial,
    rows: &[Fraction],
    nodes: &[Element],
    variable: usize,
    side: usize,
) -> Polynomial {
    let parts: Vec<_> = rows
        .iter()
        .take(old.nterms())
        .map(|r| {
            if side == 0 {
                &r.numerator
            } else {
                &r.denominator
            }
        })
        .collect();
    let degree = parts.iter().map(|p| p.degree(variable)).max().unwrap_or(0);
    let mut p = template.zero();
    if old.is_zero() {
        return p;
    }
    for d in 0..=degree {
        let rhs: Vec<_> = parts.iter().map(|r| coefficient(r, variable, d)).collect();
        let cs = template.solve_shifted_transposed_vandermonde(nodes, &rhs);
        for (c, m) in cs.into_iter().zip(old) {
            let mut ex = m.exponents.to_vec();
            ex[variable] = d;
            p.append_monomial(c, &ex);
        }
    }
    p
}

fn embed(template: &Polynomial, variable: usize, dense: UnivariatePolynomial<Zp64>) -> Polynomial {
    embed_shifted(template, variable, dense, 0)
}

fn embed_shifted(
    template: &Polynomial,
    variable: usize,
    dense: UnivariatePolynomial<Zp64>,
    shift: u16,
) -> Polynomial {
    let mut p = template.zero();
    let mut ex = vec![0; p.nvars()];
    for (d, c) in dense.coefficients.into_iter().enumerate() {
        ex[variable] = d as u16 + shift;
        p.append_monomial(c, &ex);
    }
    p
}

fn normalize_constant(mut r: Fraction) -> Result<Fraction> {
    let c = r.denominator.get_constant();
    if r.denominator.ring().is_zero(&c) {
        return unlucky();
    }
    r.numerator = r.numerator.div_coeff(&c);
    r.denominator = r.denominator.div_coeff(&c);
    Ok(r)
}
fn line_point(f: &Zp64, shift: &[Element], direction: &[Element], t: Element) -> Vec<Element> {
    shift
        .iter()
        .zip(direction)
        .map(|(s, z)| f.add(s, &f.mul(&t, z)))
        .collect()
}
fn homogeneous_part(p: &Polynomial, degree: u16) -> Polynomial {
    let mut r = p.zero();
    for m in p {
        if m.exponents.iter().map(|e| *e as u32).sum::<u32>() == degree as u32 {
            r.append_monomial(*m.coefficient, m.exponents);
        }
    }
    r
}

// Number of possible monomials of each total degree at the learned bounds.
// Saturation affects scheduling only, never an interpolation bound.
fn homogeneous_sizes(bounds: &[u16], degree: u16) -> Vec<u64> {
    let mut counts = vec![0u64; degree as usize + 1];
    counts[0] = 1;
    for &bound in bounds {
        let mut next = vec![0u64; counts.len()];
        for (d, n) in next.iter_mut().enumerate() {
            for e in 0..=d.min(bound as usize) {
                *n = n.saturating_add(counts[d - e]);
            }
        }
        counts = next;
    }
    counts
}

// Coefficients of p(t * direction), using only reconstructed components.
fn homogeneous_line(p: &Polynomial, direction: &[Element]) -> Vec<Element> {
    let f = p.ring();
    let mut bounds = vec![0; p.nvars()];
    let mut degree = 0;
    for m in p {
        let mut total = 0;
        for (bound, &e) in bounds.iter_mut().zip(m.exponents) {
            *bound = (*bound).max(e as usize);
            total += e as usize;
        }
        degree = degree.max(total);
    }
    let powers: Vec<Vec<_>> = bounds
        .iter()
        .zip(direction)
        .map(|(&bound, x)| {
            let mut powers = vec![f.one(); bound + 1];
            for i in 1..powers.len() {
                powers[i] = f.mul(&powers[i - 1], x);
            }
            powers
        })
        .collect();
    let mut result = vec![f.zero(); degree + 1];
    for m in p {
        let mut c = *m.coefficient;
        let mut degree = 0;
        for (&e, powers) in m.exponents.iter().zip(&powers) {
            degree += e as usize;
            if e != 0 {
                f.mul_assign(&mut c, &powers[e as usize]);
            }
        }
        f.add_assign(&mut result[degree], &c);
    }
    result
}
fn translate(p: &Polynomial, shift: &[Element]) -> Polynomial {
    let mut r = p.clone();
    for (i, s) in shift.iter().enumerate() {
        if p.ring().is_zero(s) {
            continue;
        }
        r = r.shift_var_cached(i, s);
    }
    r
}
fn monomial_nodes(p: &Polynomial, anchors: &[Element], end: usize) -> Vec<Element> {
    let f = p.ring();
    p.into_iter()
        .map(|m| {
            (0..end).fold(f.one(), |r, j| {
                f.mul(&r, &f.pow(&anchors[j], m.exponents[j] as u64))
            })
        })
        .collect()
}
fn distinct_nonzero(f: &Zp64, x: &[Element]) -> Result<()> {
    let mut seen = HashSet::new();
    if x.iter().any(|v| f.is_zero(v) || !seen.insert(*v)) {
        return unlucky();
    }
    Ok(())
}

// Polynomial Zippel, with x_0 = 1 and a known total-degree bound. The previous
// slice supplies only its nonzero support. Symbolica solves the transposed
// Vandermonde systems and expands the Newton interpolant.
fn zippel(
    template: &Polynomial,
    anchors: &[Element],
    degree: u16,
    bounds: &[u16],
    early_checks: usize,
    mut oracle: impl FnMut(&[Element]) -> Result<Element>,
) -> Result<Polynomial> {
    let f = template.ring();
    let mut p = template.constant(oracle(anchors)?);
    for variable in 1..template.nvars() {
        if p.is_zero() {
            return Ok(p);
        }
        let nodes = monomial_nodes(&p, anchors, variable);
        distinct_nonzero(f, &nodes)?;
        let exponents: Vec<_> = (&p).into_iter().map(|m| m.exponents.to_vec()).collect();
        // Homogeneity supplies both a lower and an upper bound. Factoring out
        // x_k^lower before Newton interpolation is particularly effective for
        // a high-total-degree coefficient with small individual degrees.
        let remaining: u32 = bounds[0] as u32
            + bounds[variable + 1..]
                .iter()
                .map(|d| *d as u32)
                .sum::<u32>();
        let mut lower = Vec::new();
        let mut widths = Vec::new();
        for ex in &exponents {
            let used: u32 = ex.iter().map(|d| *d as u32).sum();
            if used > degree as u32 {
                return unlucky();
            }
            let hi = (bounds[variable] as u32).min(degree as u32 - used);
            let lo = (degree as u32).saturating_sub(used + remaining);
            if lo > hi {
                return unlucky();
            }
            lower.push(lo as u16);
            widths.push((hi - lo) as usize);
        }
        let mut active: Vec<_> = (0..p.nterms()).collect();
        let mut values: Vec<Vec<Polynomial>> = vec![Vec::new(); p.nterms()];
        let mut differences: Vec<Vec<Element>> = vec![Vec::new(); p.nterms()];
        let mut zero_runs = vec![0; p.nterms()];
        let mut finished = vec![false; p.nterms()];
        let mut solved = template.zero();
        let mut xs = Vec::new();
        for j in 0..=widths.iter().copied().max().unwrap() {
            let x = f.pow(&anchors[variable], j as u64 + 1);
            if xs.contains(&x) {
                return unlucky();
            }
            xs.push(x);
            let cs = if j == 0 {
                p.coefficients.clone()
            } else {
                let active_nodes: Vec<_> = active.iter().map(|i| nodes[*i]).collect();
                let mut rhs = Vec::new();
                for i in 1..=active.len() {
                    let mut point = anchors.to_vec();
                    for k in 1..variable {
                        point[k] = f.pow(&anchors[k], i as u64);
                    }
                    point[variable] = x;
                    rhs.push(f.sub(&oracle(&point)?, &solved.replace_all(&point)));
                }
                template.solve_shifted_transposed_vandermonde(&active_nodes, &rhs)
            };
            let inverse_differences: Vec<_> = if early_checks > 0 {
                xs[..j].iter().map(|t| f.inv(&f.sub(&x, t))).collect()
            } else {
                Vec::new()
            };
            for (index, c) in active.iter().copied().zip(cs) {
                let c = f.div(&c, &f.pow(&x, lower[index] as u64));
                values[index].push(template.constant(c));
                if early_checks > 0 {
                    let mut delta = c;
                    for (old, inverse) in differences[index].iter().zip(&inverse_differences) {
                        delta = f.mul(&f.sub(&delta, old), inverse);
                    }
                    zero_runs[index] = if f.is_zero(&delta) {
                        zero_runs[index] + 1
                    } else {
                        0
                    };
                    differences[index].push(delta);
                }
                if j == widths[index] || (early_checks > 0 && zero_runs[index] >= early_checks) {
                    let coefficient =
                        Polynomial::newton_interpolation(&xs, &values[index], variable);
                    let mut ex = exponents[index].clone();
                    ex[variable] = lower[index];
                    solved = solved + coefficient * &template.monomial(f.one(), ex);
                    finished[index] = true;
                }
            }
            active.retain(|i| !finished[*i]);
            if active.is_empty() {
                break;
            }
        }
        p = solved;
    }
    Ok(p)
}
