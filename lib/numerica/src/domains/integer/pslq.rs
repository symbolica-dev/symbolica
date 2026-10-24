//! A two-level PSLQ accelerator. Only the matrix guiding the row operations is
//! approximate: the row transformations and relation coefficients stay exact.
//! See Bailey, "The two-level multipair PSLQ algorithm" (2025), sections 3–4.

use super::{Integer, InternalOrdering};
use crate::{
    domains::float::{F64, FloatField, FloatLike, Real, RealLike},
    tensors::matrix::Matrix,
};

pub(super) trait PslqFloat:
    FloatLike + RealLike + Real + std::hash::Hash + Eq + InternalOrdering + PartialOrd
{
}

impl<T> PslqFloat for T where
    T: FloatLike + RealLike + Real + std::hash::Hash + Eq + InternalOrdering + PartialOrd
{
}

pub(super) enum Outcome {
    Found(Vec<Integer>),
    Exhausted(Vec<Integer>),
    Resume(usize),
    PrecisionLimit,
}

/// Original inputs shared by the accelerated search and fallback verification.
pub(super) struct Pslq<'a, T: PslqFloat> {
    x: &'a [T],
    tolerance: &'a T,
}

/// Reduction state between updates of the high-precision matrix.
struct Block<'a, T: PslqFloat> {
    full: &'a mut Matrix<FloatField<T>>,
    h: Matrix<FloatField<F64>>,
    // Rows of the exact transformation applied since the last refresh.
    d: Vec<Vec<i64>>,
    largest_d: u64,
    y: &'a mut [T],
    b: &'a mut [Vec<Integer>],
    representative: &'a T,
}

impl<'a, T: PslqFloat> Block<'a, T> {
    fn new(
        full: &'a mut Matrix<FloatField<T>>,
        y: &'a mut [T],
        b: &'a mut [Vec<Integer>],
        representative: &'a T,
    ) -> Option<Self> {
        let h = Self::native_matrix(full)?;
        let n = y.len();
        let mut block = Self {
            full,
            h,
            d: vec![vec![0; n]; n],
            largest_d: 1,
            y,
            b,
            representative,
        };
        block.reset_transform();
        Some(block)
    }

    fn native_matrix(full: &Matrix<FloatField<T>>) -> Option<Matrix<FloatField<F64>>> {
        let mut data = Vec::with_capacity(full.data.len());
        for value in full.iter() {
            let v = value.to_f64();
            // A wide exponent range is handled by the full-precision solver.
            if !v.is_finite() || (v == 0.0 && !value.is_zero()) || v.is_subnormal() {
                return None;
            }
            data.push(F64(v));
        }
        Matrix::from_linear(data, full.nrows, full.ncols, FloatField::<F64>::new()).ok()
    }

    fn remove_corner(&mut self, row: u32, col: u32) -> bool {
        let norm = self.h[(row, row)].0.hypot(self.h[(row, col)].0);
        if !norm.is_normal() {
            return false;
        }
        let c = self.h[(row, row)].0 / norm;
        let s = self.h[(row, col)].0 / norm;
        for i in row + 1..self.h.nrows {
            let a = self.h[(i, row)].0;
            let b = self.h[(i, col)].0;
            let left = c * a + s * b;
            let right = -s * a + c * b;
            if !left.is_finite() || !right.is_finite() {
                return false;
            }
            self.h[(i, row)] = F64(left);
            self.h[(i, col)] = F64(right);
        }
        self.h[(row, row)] = F64(norm);
        self.h[(row, col)] = F64(0.0);
        true
    }

    fn triangularize_native(&mut self) -> bool {
        for row in 0..self.h.ncols {
            for col in row + 1..self.h.ncols {
                if self.h[(row, col)].0 != 0.0 && !self.remove_corner(row, col) {
                    return false;
                }
            }
        }
        true
    }

    fn reset_transform(&mut self) {
        for (i, row) in self.d.iter_mut().enumerate() {
            row.fill(0);
            row[i] = 1;
        }
        self.largest_d = 1;
    }

    fn reduce(&mut self, i: usize, j: usize) -> bool {
        let row = i as u32;
        let col = j as u32;
        let diagonal = self.h[(col, col)].0;
        if !diagonal.is_normal() {
            return false;
        }
        let quotient = (self.h[(row, col)].0 / diagonal).round();
        // Besides protecting the floating reduction, these limits ensure that
        // every exact D update fits i64: |D| <= 2^32 and |q| <= 2^20.
        if !quotient.is_finite() || quotient.abs() > (1u64 << 20) as f64 {
            return false;
        }
        let q = quotient as i64;
        if q == 0 {
            return true;
        }

        let (before, after) = self.d.split_at_mut(i);
        for (target, source) in after[0].iter_mut().zip(&before[j]) {
            *target -= q * source;
            self.largest_d = self.largest_d.max(target.unsigned_abs());
        }
        let coefficient = Integer::from(q);
        let scalar = self.representative.from_i64(q);
        let term = scalar.mul_integer_coefficient(&self.y[i], &coefficient);
        self.y[j] += term;
        let (before, after) = self.b.split_at_mut(i);
        for (target, source) in before[j].iter_mut().zip(&after[0]) {
            target.fused_mul_assign(&coefficient, source, false);
        }

        // D, y and B now describe the same exact row operation, even if the
        // native matrix becomes unusable and we have to resume at full precision.
        for k in 0..=col {
            let value = self.h[(row, k)].0 - quotient * self.h[(col, k)].0;
            if !value.is_finite() {
                return false;
            }
            self.h[(row, k)] = F64(value);
        }
        self.largest_d <= 1u64 << 32
    }

    fn refresh(&mut self) {
        let n = self.y.len();
        let precision = self.representative.get_precision();
        let mut next = Matrix::new(n as u32, n as u32 - 1, self.full.field.clone());
        for i in 0..n {
            for k in 0..n {
                let q = self.d[i][k];
                if q == 0 {
                    continue;
                }
                let coefficient = Integer::from(q);
                let scalar = self.representative.from_i64(q);
                for j in 0..n - 1 {
                    let term = scalar
                        .mul_integer_coefficient(&self.full[(k as u32, j as u32)], &coefficient);
                    let target = &mut next[(i as u32, j as u32)];
                    *target += term;
                    // Use fixed working precision for the block product; the
                    // final relation is separately checked against the inputs.
                    target.set_precision(precision);
                }
            }
        }
        *self.full = next;
        self.reset_transform();
    }

    fn triangularize_full(&mut self) -> bool {
        let h = &mut self.full;
        for row in 0..h.ncols {
            for col in row + 1..h.ncols {
                if h[(row, col)].is_zero() {
                    continue;
                }
                let norm = h[(row, row)].hypot(&h[(row, col)]);
                if norm.is_zero() || !norm.is_finite() {
                    return false;
                }
                let c = h[(row, row)].clone() / &norm;
                let s = h[(row, col)].clone() / &norm;
                for i in row + 1..h.nrows {
                    let a = h[(i, row)].clone();
                    let b = h[(i, col)].clone();
                    h[(i, row)] = c.clone() * &a + s.clone() * &b;
                    h[(i, col)] = -s.clone() * a + c.clone() * b;
                }
                h[(row, row)] = norm;
                h[(row, col)] = h[(row, row)].zero();
            }
        }
        true
    }
}

impl<'a, T: PslqFloat> Pslq<'a, T> {
    pub(super) fn new(x: &'a [T], tolerance: &'a T) -> Self {
        Self { x, tolerance }
    }

    pub(super) fn can_accelerate(&self) -> bool {
        self.x.len() >= 8
            && self.x.iter().all(|v| {
                !v.fixed_precision()
                    && v.get_precision() >= 512
                    && v.real_classify().is_some()
                    && v.is_finite()
            })
    }

    pub(super) fn candidate<'b>(
        &self,
        y: &[T],
        b: &'b [Vec<Integer>],
        max_coeff: Option<&Integer>,
    ) -> Option<&'b [Integer]> {
        let i = y.iter().position(|v| v.norm() < *self.tolerance)?;
        let candidate = b[i].as_slice();
        max_coeff
            .is_none_or(|cap| candidate.iter().all(|c| c.abs() <= *cap))
            .then_some(candidate)
    }

    pub(super) fn verified_relation(&self, coefficients: &[Integer]) -> bool {
        let Self { x, tolerance } = *self;
        let input_bits = x.iter().map(FloatLike::get_precision).max().unwrap();
        let coefficient_bits = coefficients
            .iter()
            .map(Integer::significant_bits)
            .max()
            .unwrap();
        let Some(bits) = u32::try_from(coefficient_bits)
            .ok()
            .and_then(|b| input_bits.checked_add(b))
            .and_then(|b| b.checked_add(64 + x.len().ilog2()))
        else {
            return false;
        };
        let mut representative = x[0].clone();
        representative.set_precision(bits);
        let mut residual = representative.zero();
        let mut magnitude = representative.zero();
        let mut norm_squared = representative.zero();
        for (value, coefficient) in x.iter().zip(coefficients) {
            let mut value = value.clone();
            value.set_precision(bits);
            let scalar = representative.from_rational(&coefficient.clone().into());
            let term = scalar.mul_integer_coefficient(&value, coefficient);
            magnitude += term.norm();
            magnitude.set_precision(bits);
            residual += term;
            residual.set_precision(bits);
            norm_squared += value.clone() * value;
            norm_squared.set_precision(bits);
        }
        // Include a conservative arithmetic roundoff allowance. Raising precision
        // retains the supplied input values; it does not invent additional digits.
        let roundoff = magnitude * representative.from_usize(4 * x.len())
            / representative.from_usize(2).pow((bits - 2) as u64);
        let mut tolerance = tolerance.clone();
        tolerance.set_precision(bits);
        residual.norm() + roundoff < tolerance * norm_squared.sqrt()
    }

    /// Consume at most max_iter pivots. On fallback, return the transformed full
    /// matrix, y and B so the ordinary solver resumes without restarting the budget.
    pub(super) fn accelerate(
        &self,
        full: &mut Matrix<FloatField<T>>,
        y: &mut [T],
        b: &mut [Vec<Integer>],
        gamma: &T,
        max_iter: usize,
        max_coeff: Option<&Integer>,
    ) -> Outcome {
        let x = self.x;
        let n = x.len();
        let Some(mut block) = Block::new(full, y, b, &x[0]) else {
            return Outcome::Resume(0);
        };
        let mut gamma_power = gamma.to_f64();
        let gamma_native = gamma_power;
        let mut powers = Vec::with_capacity(n - 1);
        for _ in 0..n - 1 {
            if !gamma_power.is_normal() {
                return Outcome::Resume(0);
            }
            powers.push(gamma_power);
            gamma_power *= gamma_native;
        }
        let coefficient_cap = max_coeff.map(|c| x[0].from_rational(&c.clone().into()).to_f64());
        let mut completed = 0;
        'native: {
            for i in 1..n {
                for j in (0..i).rev() {
                    if !block.reduce(i, j) {
                        break 'native;
                    }
                }
            }
            // Initial reduction can already find a relation, before any pivot.
            if let Some(candidate) = self.candidate(block.y, block.b, max_coeff) {
                if self.verified_relation(candidate) {
                    return Outcome::Found(candidate.to_vec());
                }
                break 'native;
            }
            let mut scores = vec![0.0; n - 1];
            let mut changed = 0..n - 1;
            for iteration in 0..max_iter {
                if block.largest_d > 1 << 20 {
                    block.refresh();
                    let Some(h) = Block::native_matrix(block.full) else {
                        break 'native;
                    };
                    block.h = h;
                    if !block.triangularize_native() {
                        break 'native;
                    }
                    changed = 0..n - 1;
                }
                for j in changed.clone() {
                    scores[j] = block.h[(j as u32, j as u32)].0.abs() * powers[j];
                    if !scores[j].is_normal() {
                        break 'native;
                    }
                }
                let m = scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.total_cmp(b))
                    .unwrap()
                    .0;
                completed += 1;
                block.y.swap(m, m + 1);
                block.b.swap(m, m + 1);
                block.d.swap(m, m + 1);
                block.h.swap_rows(m as u32, m as u32 + 1, 0);
                if m < n - 2 && !block.remove_corner(m as u32, m as u32 + 1) {
                    break 'native;
                }
                changed = m..(m + 2).min(n - 1);
                for i in m + 1..n {
                    for j in (0..i.min(m + 2)).rev() {
                        if !block.reduce(i, j) {
                            break 'native;
                        }
                    }
                }

                if let Some(candidate) = self.candidate(block.y, block.b, max_coeff) {
                    if self.verified_relation(candidate) {
                        return Outcome::Found(candidate.to_vec());
                    }
                    break 'native;
                }

                if iteration % 20 == 0
                    && let Some(cap) = coefficient_cap
                {
                    let largest_norm = block
                        .h
                        .row_iter()
                        .map(|row| row.iter().map(|v| v.0 * v.0).sum::<f64>())
                        .fold(0.0, f64::max)
                        .sqrt();
                    // A native estimate never produces CoefficientLimit. A suspect
                    // bound switches to the ordinary full-precision test instead.
                    if !largest_norm.is_normal() || largest_norm.recip() > cap * 0.5 {
                        break 'native;
                    }
                }
            }
            let best = block
                .y
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    a.norm()
                        .partial_cmp(&b.norm())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap()
                .0;
            return Outcome::Exhausted(block.b[best].clone());
        }

        block.refresh();
        if block.triangularize_full() {
            Outcome::Resume(completed)
        } else {
            Outcome::PrecisionLimit
        }
    }
}

#[cfg(all(test, feature = "float-mpfr"))]
mod tests {
    use super::*;
    use crate::domains::float::{Float, SingleFloat};

    #[test]
    fn transform_limit_preserves_the_committed_operation() {
        let representative = Float::with_val(512, 1);
        let previous_limit = 1i64 << 32;
        let quotient = 1i64 << 20;
        // A unimodular D and its inverse B, with B stored as columns.
        let mut b = vec![
            vec![Integer::one(), Integer::zero()],
            vec![Integer::from(-previous_limit), Integer::one()],
        ];
        let input = [2i64, 3];
        let mut y = vec![
            representative.from_i64(input[0]),
            representative.from_i64(input[1] - previous_limit * input[0]),
        ];
        let mut full = Matrix::new(2, 1, FloatField::from_rep(representative.clone()));
        full[(0, 0)] = representative.from_i64(1);
        full[(1, 0)] = representative.from_i64(quotient);
        let mut block = Block::new(&mut full, &mut y, &mut b, &representative).unwrap();
        block.d = vec![vec![1, previous_limit], vec![0, 1]];
        block.largest_d = previous_limit as u64;

        assert!(!block.reduce(1, 0));
        assert_eq!(block.d[1], [-quotient, 1 - quotient * previous_limit]);
        assert!(block.largest_d > 1u64 << 32);
        assert!(block.largest_d < 1u64 << 53);
        assert_eq!(block.h[(1, 0)].0, 0.0);

        // The operation that triggered fallback must have reached D, B and y.
        for row in 0..2 {
            for col in 0..2 {
                let product = (0..2).fold(Integer::zero(), |sum, k| {
                    sum + Integer::from(block.d[row][k]) * &block.b[col][k]
                });
                assert_eq!(product, Integer::from(i64::from(row == col)));
            }
        }
        for col in 0..2 {
            let expected = input
                .iter()
                .zip(&block.b[col])
                .fold(Integer::zero(), |sum, (value, coefficient)| {
                    sum + Integer::from(*value) * coefficient
                });
            assert_eq!(block.y[col], representative.from_rational(&expected.into()));
        }
    }

    #[test]
    fn excessive_quotient_is_rejected_before_mutating_the_block() {
        let representative = Float::with_val(512, 1);
        let mut y = vec![representative.from_i64(3), representative.from_i64(5)];
        let mut b = vec![
            vec![Integer::one(), Integer::zero()],
            vec![Integer::zero(), Integer::one()],
        ];
        let previous_y = y.clone();
        let previous_b = b.clone();
        let mut full = Matrix::new(2, 1, FloatField::from_rep(representative.clone()));
        full[(0, 0)] = representative.from_i64(1);
        full[(1, 0)] = representative.from_i64((1i64 << 20) + 1);
        let mut block = Block::new(&mut full, &mut y, &mut b, &representative).unwrap();
        let previous_h = block.h.data.clone();
        let previous_d = block.d.clone();

        assert!(!block.reduce(1, 0));
        assert_eq!(block.d, previous_d);
        assert_eq!(block.h.data, previous_h);
        assert_eq!(block.y, previous_y);
        assert_eq!(block.b, previous_b);
        assert_eq!(block.largest_d, 1);
    }

    #[test]
    fn native_conversion_rejects_unrepresentable_exponents() {
        let representative = Float::with_val(512, 1);
        let mut h = Matrix::new(2, 1, FloatField::from_rep(representative.clone()));
        h[(1, 0)] = representative;
        for value in ["1e1000", "1e-1000", "1e-320"] {
            h[(0, 0)] = Float::parse(value, Some(512)).unwrap();
            assert!(Block::native_matrix(&h).is_none());
        }
        h[(0, 0)] = Float::parse("1e-100", Some(512)).unwrap();
        assert!(Block::native_matrix(&h).is_some());
    }

    #[test]
    fn relation_verification_rejects_an_inaccurate_candidate() {
        let input = [Float::with_val(512, 3), Float::with_val(512, 4)];
        let tolerance = Float::parse("1e-100", Some(512)).unwrap();
        let pslq = Pslq::new(&input, &tolerance);
        assert!(pslq.verified_relation(&[Integer::from(4), Integer::from(-3)]));
        assert!(!pslq.verified_relation(&[Integer::from(4), Integer::from(-2)]));
    }
}
