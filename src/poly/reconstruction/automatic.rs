//! A probe-count heuristic based only on reconstructed generic slices.
use super::*;

pub(super) struct DegreeProfile {
    pub bounds: [Vec<u16>; 2],
    pub minimum: [Vec<u16>; 2],
}

impl<F> Context<'_, F>
where
    F: FnMut(&Zp64, &[Element]) -> Option<Element>,
{
    pub(super) fn select_method(
        &mut self,
    ) -> Result<(ReconstructionMethod, Option<DegreeProfile>)> {
        use ReconstructionMethod::*;
        self.balanced_pilot = None;
        let nv = self.template.nvars();
        if nv <= 2 {
            return Ok((BalancedZippel, None));
        }
        let anchor = self.point();
        let last = nv - 1;
        let slice = self.thiele(last, |t| {
            let mut p = anchor.clone();
            p[last] = t;
            p
        })?;
        let other_anchor = self.point();
        let other = self.thiele(last, |t| {
            let mut p = other_anchor.clone();
            p[last] = t;
            p
        })?;
        // Monic denominators agree when the last-variable factor separates.
        // This is only a hypothesis; ordinary final validation and fallback
        // remain responsible for accepting the reconstructed function.
        if slice.denominator == other.denominator {
            let univariate = slice.numerator == other.numerator;
            // Proportional numerators predict separation of the whole
            // rational factor, including its numerator's variable dependence.
            let factorizes = !slice.numerator.is_zero()
                && !other.numerator.is_zero()
                && slice.numerator.clone().mul_coeff(coefficient(
                    &other.numerator,
                    last,
                    other.numerator.degree(last),
                )) == other.numerator.clone().mul_coeff(coefficient(
                    &slice.numerator,
                    last,
                    slice.numerator.degree(last),
                ));
            self.balanced_pilot = Some(BalancedPilot {
                numerator_factor: if !factorizes
                    && !slice.numerator.is_zero()
                    && !other.numerator.is_zero()
                {
                    let factor = slice.numerator.gcd(&other.numerator);
                    (factor.degree(last) > 0).then_some(factor)
                } else {
                    None
                },
                point: anchor,
                row: slice,
                factorizes,
                univariate,
            });
            return Ok((BalancedZippelSeparated, None));
        }
        let mut profile = DegreeProfile {
            bounds: [vec![0; nv], vec![0; nv]],
            minimum: [vec![0; nv], vec![0; nv]],
        };
        let mut term_counts = [vec![0u64; nv], vec![0u64; nv]];
        let intersection = value(&slice, &anchor);
        for variable in 0..nv {
            let row = if variable == last {
                slice.clone()
            } else {
                self.thiele_seeded(
                    variable,
                    |t| {
                        let mut p = anchor.clone();
                        p[variable] = t;
                        p
                    },
                    intersection.map(|v| (anchor[variable], v)),
                )?
            };
            for (side, p) in [&row.numerator, &row.denominator].into_iter().enumerate() {
                let lo = p
                    .into_iter()
                    .map(|m| m.exponents[variable])
                    .min()
                    .unwrap_or(0);
                profile.minimum[side][variable] = lo;
                profile.bounds[side][variable] = p.degree(variable) - lo;
                term_counts[side][variable] = p.nterms() as u64;
            }
            // Both sides must be sparse: a sparse denominator alone can still
            // be cheap to prune underneath a large dense numerator.
            let sparse = |side: usize| {
                let span = u64::from(profile.bounds[side][variable]) + 1;
                span == 1 || span >= 2 * term_counts[side][variable].max(1)
            };
            let has_gap = (0..2).any(|side| {
                u64::from(profile.bounds[side][variable]) + 1
                    > 2 * term_counts[side][variable].max(1)
            });
            if has_gap && sparse(0) && sparse(1) {
                return Ok((BalancedZippel, None));
            }
        }
        let mut prefix = [1u64; 2];
        let mut balanced = 0u64;
        for (variable, &num_degree) in profile.bounds[0].iter().enumerate() {
            let row = u64::from(num_degree) + u64::from(profile.bounds[1][variable]) + 1;
            balanced = balanced.saturating_add(prefix[0].max(prefix[1]).saturating_mul(row));
            for (side, count) in prefix.iter_mut().enumerate() {
                *count = count.saturating_mul(term_counts[side][variable]);
            }
        }
        // If even the full degree box is cheaper, refining it with an affine
        // line cannot change the cost-model decision. Avoid those oracle calls.
        let box_bound = profile.bounds.iter().fold(0u64, |sum, bounds| {
            sum.saturating_add(
                bounds
                    .iter()
                    .fold(1u64, |n, d| n.saturating_mul(u64::from(*d) + 1)),
            )
        });
        let degrees_fit = (0..2).all(|side| {
            profile.bounds[side]
                .iter()
                .zip(&profile.minimum[side])
                .map(|(hi, lo)| u64::from(*hi) + u64::from(*lo))
                .sum::<u64>()
                <= u64::from(self.options.max_degree)
        });
        if degrees_fit && box_bound < balanced {
            return Ok((CuytLeePruned, Some(profile)));
        }
        // A generic affine line avoids cancellation of a common power of t at
        // the origin. Remove learned monomials while estimating residual degree.
        let direction = self.point();
        let field = self.field.clone();
        self.monomial_factors = Some(profile.minimum.clone());
        let line = self.thiele(0, |t| line_point(&field, &anchor, &direction, t));
        self.monomial_factors = None;
        let line = line?;
        let totals = [line.numerator.degree(0), line.denominator.degree(0)];
        if (0..2).any(|side| {
            u64::from(totals[side])
                + profile.minimum[side]
                    .iter()
                    .map(|d| u64::from(*d))
                    .sum::<u64>()
                > u64::from(self.options.max_degree)
        }) {
            return Ok((BalancedZippel, None));
        }
        let homogeneous = (0..2).fold(0u64, |cost, side| {
            homogeneous_sizes(&profile.bounds[side], totals[side])
                .into_iter()
                .fold(cost, u64::saturating_add)
        });
        // These dense support bounds estimate work, not actual support. Their
        // ranking can be wrong on sparse inputs; explicit methods remain available.
        if homogeneous < balanced {
            Ok((CuytLeePruned, Some(profile)))
        } else {
            Ok((BalancedZippel, None))
        }
    }
}
