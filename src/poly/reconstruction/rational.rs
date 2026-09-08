//! Coefficient lifting shared by both finite-field reconstruction methods.
use super::*;
use crate::domains::{
    finite_field::{PrimeIteratorU64, ToFiniteField},
    integer::{Integer, IntegerRing, Z},
    rational::{Q, Rational},
};

/// Total work across prime fields, including independent verification primes.
#[derive(Clone, Debug, Default)]
pub struct RationalReconstructionStats {
    pub primes: usize,
    pub probes: usize,
    pub successful_images: usize,
    /// Restarts caused by changing modular support (e.g. an unlucky prime).
    pub support_resets: usize,
}

/// Reconstruct over Q by CRT and maximal-quotient coefficient reconstruction.
///
/// The result uses Symbolica's integer numerator/denominator representation.
/// Primes start above 2^61; `max_primes` includes verification primes. The probe
/// bound in `options` applies separately to each reconstructed image. Failed
/// modular interpolations and changing supports are retried within this bound.
/// A candidate is accepted only after checking at a prime unused in its CRT.
/// This is a probabilistic identity check, not a coefficient-height proof.
pub fn reconstruct_rational_function_over_q<F>(
    variables: Arc<Vec<PolyVariable>>,
    mut black_box: F,
    method: ReconstructionMethod,
    options: &ReconstructionOptions,
    max_primes: usize,
) -> Result<(
    RationalPolynomial<IntegerRing, u16>,
    RationalReconstructionStats,
)>
where
    F: FnMut(&Zp64, &[Element]) -> Option<Element>,
{
    if max_primes < 2 {
        return Err(ReconstructionError::InvalidOptions);
    }
    let mut primes = PrimeIteratorU64::new(1 << 61);
    let mut stats = RationalReconstructionStats::default();
    let mut support = Vec::new();
    let mut residues: Vec<Integer> = Vec::new();
    let mut modulus = Integer::one();
    let mut rng = StdRng::seed_from_u64(options.seed ^ 0x637274);
    while stats.primes < max_primes {
        let prime = primes.next().ok_or(ReconstructionError::PrimeLimit)?;
        stats.primes += 1;
        let field = Zp64::new(prime);
        let mut image_options = options.clone();
        image_options.seed = rng.random();
        let image = reconstruct_rational_function(
            field.clone(),
            variables.clone(),
            |f, p| {
                stats.probes += 1;
                black_box(f, p)
            },
            method,
            &image_options,
        );
        let (image, _) = match image {
            Ok(r) => r,
            Err(ReconstructionError::InvalidOptions) => {
                return Err(ReconstructionError::InvalidOptions);
            }
            Err(_) => continue,
        };
        stats.successful_images += 1;
        let image_support: Vec<_> = [&image.numerator, &image.denominator]
            .into_iter()
            .enumerate()
            .flat_map(|(side, p)| p.into_iter().map(move |m| (side, m.exponents.to_vec())))
            .collect();
        let cs: Vec<_> = image
            .numerator
            .coefficients
            .iter()
            .chain(&image.denominator.coefficients)
            .map(|c| Integer::from(field.from_element(c)))
            .collect();
        if image_support != support {
            stats.support_resets += usize::from(!support.is_empty());
            support = image_support;
            residues = cs;
            modulus = prime.into();
        } else {
            let p = Integer::from(prime);
            for (residue, c) in residues.iter_mut().zip(cs) {
                *residue =
                    Integer::chinese_remainder(c, residue.clone(), p.clone(), modulus.clone());
            }
            modulus *= &p;
        }
        let candidate: std::result::Result<Vec<_>, _> = residues
            .iter()
            .map(|r| Rational::maximal_quotient_reconstruction(r, &modulus, None))
            .collect();
        let Ok(coefficients) = candidate else {
            continue;
        };
        if stats.primes == max_primes {
            break;
        }
        let mut numerator = MultivariatePolynomial::<_, u16>::new(&Q, None, variables.clone());
        let mut denominator = numerator.zero();
        for ((side, ex), c) in support.iter().zip(coefficients) {
            if *side == 0 {
                numerator.append_monomial(c, ex);
            } else {
                denominator.append_monomial(c, ex);
            }
        }
        let prime = primes.next().ok_or(ReconstructionError::PrimeLimit)?;
        stats.primes += 1;
        let field = Zp64::new(prime);
        // Avoid attempting to invert coefficient denominators divisible by p.
        if numerator
            .coefficients
            .iter()
            .chain(&denominator.coefficients)
            .any(|c| field.is_zero(&c.denominator_ref().to_finite_field(&field)))
        {
            continue;
        }
        let modular = Fraction {
            numerator: numerator.map_coeff(|c| c.to_finite_field(&field), field.clone()),
            denominator: denominator.map_coeff(|c| c.to_finite_field(&field), field.clone()),
        };
        let mut checked = 0;
        let mut seen = HashSet::new();
        for _ in 0..(options.verification_points * 16).min(options.max_probes) {
            let point: Vec<_> = (0..variables.len())
                .map(|_| field.to_element(rng.random_range(1..prime)))
                .collect();
            if !seen.insert(point.clone()) {
                continue;
            }
            stats.probes += 1;
            if let Some(v) = black_box(&field, &point) {
                if value(&modular, &point) != Some(v) {
                    break;
                }
                checked += 1;
                if checked == options.verification_points {
                    let result = RationalPolynomial::from_num_den(numerator, denominator, &Z, true);
                    return Ok((result, stats));
                }
            }
        }
    }
    Err(ReconstructionError::PrimeLimit)
}
