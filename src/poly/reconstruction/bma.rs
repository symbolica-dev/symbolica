//! Sparse exponential-sum interpolation using the Hu-Monagan GCD primitives.
use super::*;
use crate::{
    domains::finite_field::{SMOOTH_PRIME_BASE, SMOOTH_PRIMES, Zp64DiscreteLogContext},
    poly::{gcd::HuMonaganKroneckerMap, univariate::DenseFiniteFieldRootContext},
};

fn parameters(field: &Zp64) -> Result<(Element, Vec<(u64, u32)>)> {
    let (_, generator, factors) = SMOOTH_PRIMES
        .iter()
        .find(|(prime, _, _)| *prime == field.get_prime())
        .ok_or(ReconstructionError::UnsupportedBmaGeometry)?;
    Ok((
        field.to_element(u64::from(*generator)),
        SMOOTH_PRIME_BASE
            .iter()
            .zip(factors)
            .filter_map(|(&p, &e)| (e != 0).then_some((p, u32::from(e))))
            .collect(),
    ))
}

fn encoding(options: &ReconstructionOptions, active: usize) -> Result<HuMonaganKroneckerMap> {
    let radices: Vec<_> = (0..active)
        .map(|i| {
            u32::from(
                options
                    .bma_degree_bounds
                    .as_ref()
                    .map_or(options.max_degree, |b| b[i]),
            ) + 1
        })
        .collect();
    HuMonaganKroneckerMap::new(&radices, 0).ok_or(ReconstructionError::UnsupportedBmaGeometry)
}

pub(super) fn validate(
    field: &Zp64,
    variables: usize,
    method: ReconstructionMethod,
    options: &ReconstructionOptions,
) -> Result<()> {
    parameters(field)?;
    if field.get_prime() <= minimum_prime(variables, method, options)? {
        return Err(ReconstructionError::UnsupportedBmaGeometry);
    }
    Ok(())
}

pub(super) fn minimum_prime(
    variables: usize,
    method: ReconstructionMethod,
    options: &ReconstructionOptions,
) -> Result<u64> {
    if variables == 0 {
        return Err(ReconstructionError::InvalidOptions);
    }
    if options.bma_degree_bounds.as_ref().is_some_and(|bounds| {
        bounds.len() != variables || bounds.iter().any(|&b| b > options.max_degree)
    }) {
        return Err(ReconstructionError::InvalidOptions);
    }
    let active = if method == ReconstructionMethod::PolynomialBma {
        variables
    } else {
        variables - 1
    };
    encoding(options, active)?
        .range()
        .checked_add(1)
        .map(|range| {
            range.max(8 * (u64::from(options.max_degree) + options.verification_points as u64 + 1))
        })
        .ok_or(ReconstructionError::UnsupportedBmaGeometry)
}

struct Geometry {
    map: HuMonaganKroneckerMap,
    bases: Vec<Element>,
    shift: Vec<Element>,
}

// Small supports are checked near 2T samples. For larger supports, geometric
// checkpoints keep repeated batch-BMA work quadratic overall.
fn recovery_checkpoint(samples: usize) -> bool {
    samples >= 4 && (samples <= 32 || samples.is_power_of_two())
}

impl Geometry {
    fn new(
        field: &Zp64,
        options: &ReconstructionOptions,
        active: usize,
        shift: &[Element],
        alpha: Element,
    ) -> Result<Self> {
        let map = encoding(options, active)?;
        if map.range() >= field.get_prime() - 1 {
            return Err(ReconstructionError::UnsupportedBmaGeometry);
        }
        let bases = (0..active)
            .map(|i| {
                if i == 0 {
                    alpha
                } else {
                    field.pow(&alpha, map.powers()[i - 1])
                }
            })
            .collect();
        Ok(Self {
            map,
            bases,
            shift: shift[..active].to_vec(),
        })
    }

    fn advance(&self, field: &Zp64, point: &mut [Element]) {
        for (x, base) in point.iter_mut().zip(&self.bases) {
            field.mul_assign(x, base);
        }
    }
}

/// Recover a coefficient polynomial from consecutive samples starting at k=0.
/// The recurrence, roots, logarithms, and shifted Vandermonde solver are exactly
/// the primitives used by the GCD route; no source support is supplied.
fn recover(
    template: &Polynomial,
    samples: &[Element],
    geometry: &Geometry,
    logs: &Zp64DiscreteLogContext<'_>,
    checks: usize,
) -> Option<Polynomial> {
    let f = template.ring();
    if samples.iter().all(|x| f.is_zero(x)) {
        return Some(template.zero());
    }
    let (recurrence, stable) = f.find_linear_recurrence_relation(samples);
    let terms = recurrence.len();
    if terms == 0
        || samples.len() < 2 * terms + checks
        || stable < checks
        || f.is_zero(&recurrence[0])
    {
        return None;
    }
    let mut characteristic: Vec<_> = recurrence.iter().rev().map(|x| f.neg(x)).collect();
    characteristic.push(f.one());
    let roots = DenseFiniteFieldRootContext::new(f).find_distinct_nonzero_roots(&characteristic)?;
    if roots.len() != terms {
        return None;
    }
    let encoded: Vec<_> = roots.iter().map(|root| logs.discrete_log(root)).collect();
    if encoded.iter().any(|&e| e >= geometry.map.range()) {
        return None;
    }
    let mut coefficients = template.solve_shifted_transposed_vandermonde(&roots, &samples[..terms]);
    // That solver starts at power one; our first sample is at power zero.
    for (c, root) in coefficients.iter_mut().zip(&roots) {
        f.mul_assign(c, root);
    }
    let mut current = coefficients.clone();
    for expected in samples {
        if current.iter().fold(f.zero(), |s, c| f.add(&s, c)) != *expected {
            return None;
        }
        for (c, root) in current.iter_mut().zip(&roots) {
            f.mul_assign(c, root);
        }
    }
    let mut result = template.zero();
    for (mut coefficient, exponent) in coefficients.into_iter().zip(encoded) {
        let mut powers = vec![0u16; template.nvars()];
        geometry
            .map
            .decode(exponent, &mut powers[..geometry.bases.len()])?;
        for (shift, &power) in geometry.shift.iter().zip(&powers) {
            f.div_assign(&mut coefficient, &f.pow(shift, u64::from(power)));
        }
        result.append_monomial(coefficient, &powers);
    }
    Some(result)
}

impl<F: FnMut(&Zp64, &[Element]) -> Option<Element>> Context<'_, F> {
    pub(super) fn bma_polynomial(
        &mut self,
        active: usize,
        fixed: &[Element],
        limit: usize,
    ) -> Result<Option<Fraction>> {
        let field = self.field.clone();
        let (alpha, factors) = parameters(&field)?;
        let geometry = Geometry::new(&field, self.options, active, fixed, alpha)?;
        let logs = Zp64DiscreteLogContext::new(&field, &alpha, field.get_prime() - 1, &factors);
        let mut point = fixed.to_vec();
        let mut samples = Vec::new();
        for _ in 0..limit.min((field.get_prime() - 1) as usize) {
            // A missing value breaks the consecutive sequence; retry a new
            // shift instead of silently feeding a gapped sequence to BMA.
            samples.push(
                self.probe(&point)?
                    .ok_or(ReconstructionError::AttemptsExhausted)?,
            );
            if recovery_checkpoint(samples.len())
                && let Some(numerator) = recover(
                    &self.template,
                    &samples,
                    &geometry,
                    &logs,
                    self.options.verification_points,
                )
            {
                self.stats.bma_sequences += 1;
                return Ok(Some(Fraction {
                    numerator,
                    denominator: self.template.one(),
                }));
            }
            geometry.advance(&field, &mut point);
        }
        Ok(None)
    }

    fn verify_active(
        &mut self,
        candidate: &Fraction,
        active: usize,
        fixed: &[Element],
    ) -> Result<bool> {
        let mut checked = 0;
        for _ in 0..self.options.verification_points * 16 {
            let mut point = fixed.to_vec();
            for x in &mut point[..active] {
                *x = self.random();
            }
            if self.has_cached_probe(&point) {
                continue;
            }
            if let Some(y) = self.probe(&point)? {
                if value(candidate, &point) != Some(y) {
                    return Ok(false);
                }
                checked += 1;
                if checked == self.options.verification_points {
                    return Ok(true);
                }
            }
        }
        Ok(false)
    }

    pub(super) fn hu_monagan(&mut self, active: usize, fixed: &[Element]) -> Result<Fraction> {
        if self.options.bma_polynomial_probe_limit > 0 {
            match self.bma_polynomial(active, fixed, self.options.bma_polynomial_probe_limit) {
                Ok(Some(candidate)) if self.verify_active(&candidate, active, fixed)? => {
                    return Ok(candidate);
                }
                Err(ReconstructionError::UnsupportedBmaGeometry) | Ok(_) => {}
                Err(error) => return Err(error),
            }
        }
        if active == 1 {
            return self.thiele(0, |t| {
                let mut point = fixed.to_vec();
                point[0] = t;
                point
            });
        }
        let main = active - 1;
        // This recursively reconstructed slice fixes a common scale across
        // monic rational images. A specialization can introduce a common
        // factor; fresh full-dimensional checks guard that hypothesis.
        let slice = self.hu_monagan(main, fixed)?;
        let slice = Fraction::from_num_den(slice.numerator, slice.denominator, &self.field, true);
        if slice.numerator.is_zero() {
            // A zero specialization has lost the denominator normalization.
            // Distinguish an identically zero function from an unlucky anchor.
            if self.verify_active(&slice, active, fixed)? {
                return Ok(slice);
            }
            return unlucky();
        }
        let field = self.field.clone();
        let (alpha, factors) = parameters(&field)?;
        let geometry = Geometry::new(&field, self.options, main, fixed, alpha)?;
        let logs = Zp64DiscreteLogContext::new(&field, &alpha, field.get_prime() - 1, &factors);
        let mut point = fixed.to_vec();
        let mut rows = Vec::new();
        let mut degrees = None;
        let mut minimum = (0, 0);
        let mut powers = [Vec::new(), Vec::new()];
        // Every noninitial row requires at least one new sample except for
        // constant slices. The explicit row bound also covers that case.
        for _ in 0..self
            .options
            .max_probes
            .min((field.get_prime() - 1) as usize)
        {
            let known = value(&slice, &point).ok_or(ReconstructionError::AttemptsExhausted)?;
            let mut row = if let Some(degrees) = degrees {
                self.degree_row(main, &point, known, degrees, minimum, None, false, &powers)?
            } else {
                let row = self.thiele_seeded(
                    main,
                    |t| {
                        let mut p = point.clone();
                        p[main] = t;
                        p
                    },
                    Some((point[main], known)),
                )?;
                degrees = Some((row.numerator.degree(main), row.denominator.degree(main)));
                for (side, poly) in [&row.numerator, &row.denominator].into_iter().enumerate() {
                    powers[side] = poly.into_iter().map(|m| m.exponents[main]).collect();
                    powers[side].sort_unstable();
                }
                minimum = (
                    powers[0].first().copied().unwrap_or(0),
                    powers[1].first().copied().unwrap_or(0),
                );
                row
            };
            let denominator = row.denominator.replace_all(&point);
            if field.is_zero(&denominator) {
                return unlucky();
            }
            let scale = field.div(&slice.denominator.replace_all(&point), &denominator);
            if field.is_zero(&scale) {
                return unlucky();
            }
            row.numerator = row.numerator.mul_coeff(scale);
            row.denominator = row.denominator.mul_coeff(scale);
            rows.push(row);
            if recovery_checkpoint(rows.len()) {
                let mut result = [self.template.zero(), self.template.zero()];
                let mut complete = true;
                for side in 0..2 {
                    for &power in &powers[side] {
                        let samples: Vec<_> = rows
                            .iter()
                            .map(|row| {
                                coefficient(
                                    if side == 0 {
                                        &row.numerator
                                    } else {
                                        &row.denominator
                                    },
                                    main,
                                    power,
                                )
                            })
                            .collect();
                        let Some(poly) = recover(
                            &self.template,
                            &samples,
                            &geometry,
                            &logs,
                            self.options.verification_points,
                        ) else {
                            complete = false;
                            break;
                        };
                        for monomial in &poly {
                            let mut exponents = monomial.exponents.to_vec();
                            exponents[main] = power;
                            result[side].append_monomial(*monomial.coefficient, &exponents);
                        }
                    }
                    if !complete {
                        break;
                    }
                }
                if complete {
                    self.stats.bma_sequences += powers.iter().map(Vec::len).sum::<usize>();
                    let [numerator, denominator] = result;
                    let candidate = Fraction {
                        numerator,
                        denominator,
                    };
                    if self.verify_active(&candidate, active, fixed)? {
                        return Ok(candidate);
                    }
                    // A stable but invalid support/normalization hypothesis
                    // needs a new anchor, not further repetitions of this row.
                    return unlucky();
                }
            }
            geometry.advance(&field, &mut point);
        }
        unlucky()
    }
}
