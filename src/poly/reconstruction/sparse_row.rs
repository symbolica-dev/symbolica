//! Sparse rational rows using powers learned from an earlier specialization.
use super::*;

impl<F> Context<'_, F>
where
    F: FnMut(&Zp64, &[Element]) -> Option<Element>,
{
    pub(super) fn sparse_row(
        &mut self,
        variable: usize,
        point: &[Element],
        known: Element,
        powers: [&[u16]; 2],
        denominator: Option<&Polynomial>,
        reciprocal: bool,
    ) -> Result<Fraction> {
        let f = self.field.clone();
        let fixed_degree = powers[1]
            .last()
            .copied()
            .ok_or(ReconstructionError::AttemptsExhausted)?;
        let denominator_powers = if denominator.is_some() {
            &[][..]
        } else {
            &powers[1][..powers[1].len() - 1]
        };
        let n = powers[0].len() + denominator_powers.len();
        if n == 0 {
            return unlucky();
        }
        let mut data = Vec::with_capacity(n * n);
        let mut rhs = Vec::with_capacity(n);
        let mut seen = HashSet::new();
        for _ in 0..n * 16 + 16 {
            let (t, y) = if seen.is_empty() {
                (point[variable], known)
            } else {
                let t = self.random();
                if seen.contains(&t) {
                    continue;
                }
                let mut p = point.to_vec();
                p[variable] = t;
                let Some(y) = self.probe(&p)? else {
                    seen.insert(t);
                    continue;
                };
                (t, y)
            };
            seen.insert(t);
            if reciprocal && f.is_zero(&y) {
                continue;
            }
            let y = if reciprocal { f.inv(&y) } else { y };
            data.extend(powers[0].iter().map(|d| f.pow(&t, u64::from(*d))));
            data.extend(
                denominator_powers
                    .iter()
                    .map(|d| f.neg(&f.mul(&y, &f.pow(&t, u64::from(*d))))),
            );
            let fixed = if let Some(denominator) = denominator {
                let mut p = point.to_vec();
                p[variable] = t;
                denominator.replace_all(&p)
            } else {
                f.pow(&t, u64::from(fixed_degree))
            };
            rhs.push(f.mul(&y, &fixed));
            if rhs.len() == n {
                break;
            }
        }
        if rhs.len() != n {
            return unlucky();
        }
        self.stats.linear_solves += 1;
        let solution = Matrix::from_linear(data, n as u32, n as u32, f.clone())
            .unwrap()
            .solve(&Matrix::new_vec(rhs, f.clone()))
            .map_err(|_| ReconstructionError::AttemptsExhausted)?;
        let mut num = self.template.zero();
        let mut den = denominator.cloned().unwrap_or_else(|| {
            let mut ex = vec![0; self.template.nvars()];
            ex[variable] = fixed_degree;
            self.template.monomial(f.one(), ex)
        });
        for (i, &d) in powers[0].iter().chain(denominator_powers).enumerate() {
            let mut ex = vec![0; self.template.nvars()];
            ex[variable] = d;
            if i < powers[0].len() {
                num.append_monomial(solution[(i as u32, 0)], &ex);
            } else {
                den.append_monomial(solution[(i as u32, 0)], &ex);
            }
        }
        let result = Fraction {
            numerator: num,
            denominator: den,
        };
        // Check new parameters against the original oracle before accepting a
        // learned support. The outer full-dimensional checks remain unchanged.
        let mut checked = 0;
        for _ in 0..self.options.verification_points * 16 {
            let t = self.random();
            let mut p = point.to_vec();
            p[variable] = t;
            if seen.contains(&t) || self.has_cached_probe(&p) {
                continue;
            }
            seen.insert(t);
            let Some(y) = self.probe(&p)? else {
                continue;
            };
            if reciprocal && f.is_zero(&y) {
                continue;
            }
            let y = if reciprocal { f.inv(&y) } else { y };
            if value(&result, &p) != Some(y) {
                return unlucky();
            }
            checked += 1;
            if checked == self.options.verification_points {
                return Ok(result);
            }
        }
        unlucky()
    }
}
