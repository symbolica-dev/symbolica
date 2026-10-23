//! Discover univariate factors from independent specializations.
use super::*;

pub(super) fn common_factors(a: &Fraction, b: &Fraction, variable: usize) -> Fraction {
    let common = |a: &Polynomial, b: &Polynomial| {
        let factor = a.gcd(b);
        // Monomials are already handled by the ordinary degree survey.
        let minimum = (&factor)
            .into_iter()
            .map(|m| m.exponents[variable])
            .min()
            .unwrap_or(0);
        let mut result = factor.zero();
        for m in &factor {
            let mut exponents = m.exponents.to_vec();
            exponents[variable] -= minimum;
            result.append_monomial(*m.coefficient, &exponents);
        }
        result
    };
    Fraction {
        numerator: common(&a.numerator, &b.numerator),
        denominator: common(&a.denominator, &b.denominator),
    }
}

impl<F> Context<'_, F>
where
    F: FnMut(&Zp64, &[Element]) -> Option<Element>,
{
    pub(super) fn reconstruct_profile(
        &mut self,
        mut profile: automatic::DegreeProfile,
    ) -> Result<Fraction> {
        let Some(last_factor) = profile.factor_hint.take() else {
            return self.cuyt_lee_profile(true, profile.bounds, profile.minimum);
        };
        // The existing last-variable pilot must first expose a nonmonomial
        // factor. Only then pay for a second survey in the other variables.
        // This is a hypothesis: fresh checks of the restored full function
        // still decide acceptance, and a failed attempt chooses new anchors.
        let last = self.template.nvars() - 1;
        let anchor = self.point();
        let known = self
            .probe(&anchor)?
            .ok_or(ReconstructionError::AttemptsExhausted)?;
        let mut factors = Vec::new();
        for variable in 0..=last {
            let factor = if variable == last {
                last_factor.clone()
            } else {
                let first = if variable == 0 {
                    self.balanced_initial.as_ref().map(|(_, r)| r.clone())
                } else {
                    self.balanced_survey.get(variable).cloned().flatten()
                }
                .ok_or(ReconstructionError::AttemptsExhausted)?;
                // The first survey already supplies degree bounds. Share the
                // new intersection and interpolate at those bounds instead of
                // repeating Thiele degree discovery for every second slice.
                let powers = [&first.numerator, &first.denominator]
                    .map(|p| p.into_iter().map(|m| m.exponents[variable]).collect());
                let other = self.degree_row(
                    variable,
                    &anchor,
                    known,
                    (
                        first.numerator.degree(variable),
                        first.denominator.degree(variable),
                    ),
                    (profile.minimum[0][variable], profile.minimum[1][variable]),
                    None,
                    false,
                    &powers,
                )?;
                common_factors(&first, &other, variable)
            };
            for (side, p) in [&factor.numerator, &factor.denominator]
                .into_iter()
                .enumerate()
            {
                profile.bounds[side][variable] = profile.bounds[side][variable]
                    .checked_sub(p.degree(variable))
                    .ok_or(ReconstructionError::AttemptsExhausted)?;
            }
            if factor.numerator.degree(variable) > 0 || factor.denominator.degree(variable) > 0 {
                factors.push((variable, factor));
            }
        }
        let (initial_anchor, initial_row) = self
            .balanced_initial
            .as_ref()
            .ok_or(ReconstructionError::AttemptsExhausted)?;
        let transform = |mut row: Fraction, variable: usize| -> Result<Fraction> {
            for (factor_variable, factor) in &factors {
                if *factor_variable == variable {
                    row.numerator = row
                        .numerator
                        .try_div(&factor.numerator)
                        .ok_or(ReconstructionError::AttemptsExhausted)?;
                    row.denominator = row
                        .denominator
                        .try_div(&factor.denominator)
                        .ok_or(ReconstructionError::AttemptsExhausted)?;
                } else {
                    let n = factor.numerator.replace_all(initial_anchor);
                    let d = factor.denominator.replace_all(initial_anchor);
                    if self.field.is_zero(&n) || self.field.is_zero(&d) {
                        return unlucky();
                    }
                    row.numerator = row.numerator.mul_coeff(d);
                    row.denominator = row.denominator.mul_coeff(n);
                }
            }
            Ok(Fraction::from_num_den(
                row.numerator,
                row.denominator,
                &self.field,
                true,
            ))
        };
        let initial = transform(initial_row.clone(), 0)?;
        let survey: Vec<_> = self
            .balanced_survey
            .iter()
            .enumerate()
            .map(|(variable, row)| row.clone().map(|r| transform(r, variable)).transpose())
            .collect::<Result<_>>()?;
        self.balanced_initial = Some((initial_anchor.clone(), initial));
        self.balanced_survey = survey;
        self.removed_factors = factors
            .iter()
            .map(|(variable, factor)| {
                (
                    *variable,
                    [
                        factor.numerator.to_univariate_from_univariate(*variable),
                        factor.denominator.to_univariate_from_univariate(*variable),
                    ],
                )
            })
            .collect();
        self.stats.selected_method = Some(ReconstructionMethod::BalancedZippel);
        let result = self.balanced(false);
        // Cached values remain raw. Clear the transformation on errors too.
        self.removed_factors.clear();
        result.and_then(|mut result| {
            for (_, factor) in factors {
                result.numerator = result.numerator * &factor.numerator;
                result.denominator = result.denominator * &factor.denominator;
            }
            if [&result.numerator, &result.denominator]
                .into_iter()
                .any(|p| {
                    p.into_iter().any(|m| {
                        m.exponents.iter().map(|e| u64::from(*e)).sum::<u64>()
                            > u64::from(self.options.max_degree)
                    })
                })
            {
                return unlucky();
            }
            Ok(result)
        })
    }
}
