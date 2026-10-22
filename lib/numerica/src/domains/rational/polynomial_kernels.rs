//! Lift fraction coefficients once per bulk operation, so their underlying
//! domain can reuse its polynomial kernels without normalizing every product.

use crate::{
    domains::EuclideanDomain,
    kernels::{
        ChunkedDensePolynomialMulRequest, DensePolynomialMulRequest, PolynomialKernels,
        TotalDegreePolynomialMulRequest,
    },
};

use super::{Fraction, FractionField, FractionNormalization};

impl<R: EuclideanDomain + FractionNormalization> FractionField<R> {
    /// Bound setup before computing any LCM. This limits denominator variety,
    /// not coefficient bit sizes: a single large common denominator is allowed.
    fn polynomial_lift_is_bounded(
        &self,
        left: &[Fraction<R>],
        right: &[Fraction<R>],
        output_len: usize,
    ) -> bool {
        const MIN_COEFFICIENT_PRODUCTS: usize = 64;
        const MAX_DISTINCT_DENOMINATORS: usize = 16;
        let product_count = left.len().saturating_mul(right.len());
        if product_count < MIN_COEFFICIENT_PRODUCTS {
            return false;
        }
        let mut denominators = Vec::new();
        for coefficient in left.iter().chain(right) {
            if self.ring().is_one(&coefficient.denominator)
                || denominators.contains(&&coefficient.denominator)
            {
                continue;
            }
            if denominators.len() == MAX_DISTINCT_DENOMINATORS {
                return false;
            }
            denominators.push(&coefficient.denominator);
        }
        // With few collisions, a common denominator enlarges every coefficient
        // even though individual output coefficients only need a few input
        // denominators. Keep those products in the ordinary fraction domain.
        denominators.is_empty() || product_count >= output_len.saturating_mul(4)
    }

    fn dense_fraction_layout_is_valid(
        &self,
        request: &DensePolynomialMulRequest<'_, Fraction<R>>,
    ) -> bool {
        if request.left_coefficients.len() != request.left_indices.len()
            || request.right_coefficients.len() != request.right_indices.len()
            || request
                .left_indices
                .windows(2)
                .any(|indices| indices[0] >= indices[1])
            || request
                .right_indices
                .windows(2)
                .any(|indices| indices[0] >= indices[1])
        {
            return false;
        }
        match (request.left_indices.last(), request.right_indices.last()) {
            (Some(&left), Some(&right)) => (left as usize)
                .checked_add(right as usize)
                .is_some_and(|max| max < request.output_len),
            _ => true,
        }
    }

    fn lift_polynomial_coefficients(
        &self,
        coefficients: &[Fraction<R>],
    ) -> (Vec<R::Element>, R::Element) {
        let ring = self.ring();
        let mut denominator = ring.one();
        for coefficient in coefficients {
            if ring.is_one(&coefficient.denominator) || coefficient.denominator == denominator {
                continue;
            }
            if ring.is_one(&denominator) {
                denominator = coefficient.denominator.clone();
            } else {
                let gcd = ring.gcd(&denominator, &coefficient.denominator);
                let factor = ring.quot_rem(&coefficient.denominator, &gcd).0;
                ring.mul_assign(&mut denominator, factor);
            }
        }
        let values = if ring.is_one(&denominator) {
            coefficients.iter().map(|c| c.numerator.clone()).collect()
        } else {
            coefficients
                .iter()
                .map(|c| {
                    if c.denominator == denominator {
                        c.numerator.clone()
                    } else {
                        ring.mul(&c.numerator, &ring.quot_rem(&denominator, &c.denominator).0)
                    }
                })
                .collect()
        };
        (values, denominator)
    }

    fn restore_polynomial_coefficients(
        &self,
        coefficients: Vec<(u32, R::Element)>,
        denominator: R::Element,
    ) -> Vec<(u32, Fraction<R>)> {
        if self.ring().is_one(&denominator) {
            coefficients
                .into_iter()
                .map(|(index, value)| (index, self.to_element_numerator(value)))
                .collect()
        } else {
            coefficients
                .into_iter()
                .map(|(index, value)| (index, self.to_element(value, denominator.clone(), true)))
                .collect()
        }
    }
}

impl<R: EuclideanDomain + FractionNormalization> PolynomialKernels<Fraction<R>>
    for FractionField<R>
{
    fn try_dense_mul(
        &self,
        request: DensePolynomialMulRequest<'_, Fraction<R>>,
    ) -> Option<Vec<(u32, Fraction<R>)>> {
        let underlying = self.ring().kernels();
        let kernel = underlying.polynomial()?;
        if !self.dense_fraction_layout_is_valid(&request) {
            return None;
        }
        if request.left_coefficients.is_empty() || request.right_coefficients.is_empty() {
            return Some(Vec::new());
        }
        if !self.polynomial_lift_is_bounded(
            request.left_coefficients,
            request.right_coefficients,
            request.output_len,
        ) {
            return None;
        }
        let (left, left_denominator) = self.lift_polynomial_coefficients(request.left_coefficients);
        let (right, right_denominator) =
            self.lift_polynomial_coefficients(request.right_coefficients);
        let output = kernel.try_dense_mul(DensePolynomialMulRequest {
            output_len: request.output_len,
            left_coefficients: &left,
            left_indices: request.left_indices,
            right_coefficients: &right,
            right_indices: request.right_indices,
        })?;
        Some(self.restore_polynomial_coefficients(
            output,
            self.ring().mul(left_denominator, right_denominator),
        ))
    }

    fn try_chunked_dense_mul(
        &self,
        request: ChunkedDensePolynomialMulRequest<'_, Fraction<R>>,
    ) -> Option<Vec<(u32, Fraction<R>)>> {
        let underlying = self.ring().kernels();
        let kernel = underlying.polynomial()?;
        if !self.dense_fraction_layout_is_valid(&request.dense)
            || request.inner_len == 0
            || !request.dense.output_len.is_multiple_of(request.inner_len)
        {
            return None;
        }
        if request.dense.left_coefficients.is_empty() || request.dense.right_coefficients.is_empty()
        {
            return Some(Vec::new());
        }
        if !self.polynomial_lift_is_bounded(
            request.dense.left_coefficients,
            request.dense.right_coefficients,
            request.dense.output_len,
        ) {
            return None;
        }
        let (left, left_denominator) =
            self.lift_polynomial_coefficients(request.dense.left_coefficients);
        let (right, right_denominator) =
            self.lift_polynomial_coefficients(request.dense.right_coefficients);
        let output = kernel.try_chunked_dense_mul(ChunkedDensePolynomialMulRequest {
            dense: DensePolynomialMulRequest {
                output_len: request.dense.output_len,
                left_coefficients: &left,
                left_indices: request.dense.left_indices,
                right_coefficients: &right,
                right_indices: request.dense.right_indices,
            },
            inner_len: request.inner_len,
        })?;
        Some(self.restore_polynomial_coefficients(
            output,
            self.ring().mul(left_denominator, right_denominator),
        ))
    }

    fn try_total_degree_mul(
        &self,
        request: TotalDegreePolynomialMulRequest<'_, Fraction<R>>,
    ) -> Option<Vec<(u32, Fraction<R>)>> {
        let underlying = self.ring().kernels();
        let kernel = underlying.polynomial()?;
        if request.left_coefficients.len() != request.left_codes.len()
            || request.right_coefficients.len() != request.right_codes.len()
            || request.prefix_rank.len() != request.prefix_remaining.len()
            || request.suffix_code_count == 0
            || !request
                .suffix_rank
                .len()
                .is_multiple_of(request.suffix_code_count)
            || !self.polynomial_lift_is_bounded(
                request.left_coefficients,
                request.right_coefficients,
                request.output_len,
            )
        {
            return None;
        }
        let (left, left_denominator) = self.lift_polynomial_coefficients(request.left_coefficients);
        let (right, right_denominator) =
            self.lift_polynomial_coefficients(request.right_coefficients);
        let output = kernel.try_total_degree_mul(TotalDegreePolynomialMulRequest {
            output_len: request.output_len,
            left_coefficients: &left,
            left_codes: request.left_codes,
            right_coefficients: &right,
            right_codes: request.right_codes,
            prefix_rank: request.prefix_rank,
            prefix_remaining: request.prefix_remaining,
            suffix_rank: request.suffix_rank,
            suffix_code_count: request.suffix_code_count,
        })?;
        Some(self.restore_polynomial_coefficients(
            output,
            self.ring().mul(left_denominator, right_denominator),
        ))
    }

    fn preferred_total_degree_mul_workspace_ratio(
        &self,
        left: &[Fraction<R>],
        right: &[Fraction<R>],
        output_len: usize,
    ) -> Option<usize> {
        let underlying = self.ring().kernels();
        let kernel = underlying.polynomial()?;
        if !self.polynomial_lift_is_bounded(left, right, output_len) {
            return None;
        }
        let (left, _) = self.lift_polynomial_coefficients(left);
        let (right, _) = self.lift_polynomial_coefficients(right);
        kernel.preferred_total_degree_mul_workspace_ratio(&left, &right, output_len)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domains::{
        Ring, RingOps,
        integer::Integer,
        rational::{Q, Rational},
    };

    fn coefficients(count: usize, shift: u32, rational: bool) -> Vec<Rational> {
        (0..count)
            .map(|i| {
                let value =
                    Integer::from((i % 9 + 1) as i64 * if i % 3 == 0 { -1 } else { 1 }) << shift;
                Q.to_element(
                    value,
                    Integer::from(if rational { [1, 2, 3, 5, 7][i % 5] } else { 1 }),
                    true,
                )
            })
            .collect()
    }

    fn oracle(
        left: &[Rational],
        left_indices: &[u32],
        right: &[Rational],
        right_indices: &[u32],
        len: usize,
    ) -> Vec<(u32, Rational)> {
        let mut output = vec![Rational::zero(); len];
        for (left, &i) in left.iter().zip(left_indices) {
            for (right, &j) in right.iter().zip(right_indices) {
                Q.add_mul_assign(&mut output[(i + j) as usize], left, right);
            }
        }
        output
            .into_iter()
            .enumerate()
            .filter(|(_, value)| !value.is_zero())
            .map(|(i, value)| (i as u32, value))
            .collect()
    }

    #[test]
    fn dense_fraction_multiplication_matches_coefficient_oracle() {
        let indices: Vec<u32> = (0..64).collect();
        for shift in [0, 60, 120, 200] {
            for rational in [false, true] {
                let left = coefficients(64, shift, rational);
                let mut right = left.clone();
                right.reverse();
                let expected = oracle(&left, &indices, &right, &indices, 127);
                let actual =
                    Q.kernels()
                        .polynomial()
                        .unwrap()
                        .try_dense_mul(DensePolynomialMulRequest {
                            output_len: 127,
                            left_coefficients: &left,
                            left_indices: &indices,
                            right_coefficients: &right,
                            right_indices: &indices,
                        });
                // The optional large-integer backend may decline a kernel.
                if cfg!(feature = "integer-gmp") || shift == 0 {
                    assert_eq!(actual, Some(expected));
                } else if let Some(actual) = actual {
                    assert_eq!(actual, expected);
                }
                let (lifted, denominator) = Q.lift_polynomial_coefficients(&left);
                for (lifted, original) in lifted.into_iter().zip(&left) {
                    assert_eq!(Q.to_element(lifted, denominator.clone(), true), *original);
                }
            }
        }
    }

    #[test]
    fn chunked_fraction_multiplication_preserves_cancellations() {
        let indices: Vec<u32> = (0..16).chain(32..48).collect();
        let left = coefficients(32, 0, true);
        let mut right = left.clone();
        right.reverse();
        let expected = oracle(&left, &indices, &right, &indices, 256);
        let kernel = Q.kernels();
        let kernel = kernel.polynomial().unwrap();
        assert_eq!(
            kernel.try_chunked_dense_mul(ChunkedDensePolynomialMulRequest {
                dense: DensePolynomialMulRequest {
                    output_len: 256,
                    left_coefficients: &left,
                    left_indices: &indices,
                    right_coefficients: &right,
                    right_indices: &indices
                },
                inner_len: 32,
            }),
            Some(expected)
        );
        assert!(
            kernel
                .try_chunked_dense_mul(ChunkedDensePolynomialMulRequest {
                    dense: DensePolynomialMulRequest {
                        output_len: 256,
                        left_coefficients: &left,
                        left_indices: &indices,
                        right_coefficients: &right,
                        right_indices: &indices
                    },
                    inner_len: 0,
                })
                .is_none()
        );
    }

    #[test]
    fn total_degree_fraction_multiplication_and_invalid_fallback() {
        let indices: Vec<u32> = (0..64).collect();
        let codes: Vec<_> = (0..64).map(|index| (index, 0)).collect();
        let rank: Vec<u32> = (0..127).collect();
        let remaining = vec![0; 127];
        let left = coefficients(64, 0, true);
        let expected = oracle(&left, &indices, &left, &indices, 127);
        let kernel = Q.kernels();
        let kernel = kernel.polynomial().unwrap();
        assert_eq!(
            kernel.try_total_degree_mul(TotalDegreePolynomialMulRequest {
                output_len: 127,
                left_coefficients: &left,
                left_codes: &codes,
                right_coefficients: &left,
                right_codes: &codes,
                prefix_rank: &rank,
                prefix_remaining: &remaining,
                suffix_rank: &[0],
                suffix_code_count: 1,
            }),
            Some(expected)
        );
        assert!(
            kernel
                .try_dense_mul(DensePolynomialMulRequest {
                    output_len: 127,
                    left_coefficients: &left,
                    left_indices: &indices[..63],
                    right_coefficients: &left,
                    right_indices: &indices,
                })
                .is_none()
        );
        assert_eq!(
            kernel.try_dense_mul(DensePolynomialMulRequest {
                output_len: 0,
                left_coefficients: &[],
                left_indices: &[],
                right_coefficients: &[],
                right_indices: &[],
            }),
            Some(vec![])
        );
    }

    #[test]
    fn fraction_kernel_declines_excessive_denominator_variety_before_lifting() {
        let indices: Vec<u32> = (0..32).collect();
        let many: Vec<Rational> = (0..32)
            .map(|i| Q.to_element(1.into(), Integer::from(101 + 2 * i), true))
            .collect();
        let original = many.clone();
        assert!(!Q.polynomial_lift_is_bounded(&many, &many, 63));
        let kernels = Q.kernels();
        let kernel = kernels.polynomial().unwrap();
        assert!(
            kernel
                .try_dense_mul(DensePolynomialMulRequest {
                    output_len: 63,
                    left_coefficients: &many,
                    left_indices: &indices,
                    right_coefficients: &many,
                    right_indices: &indices,
                })
                .is_none()
        );
        assert!(
            kernel
                .preferred_total_degree_mul_workspace_ratio(&many, &many, 63)
                .is_none()
        );
        assert_eq!(many, original);
        let few = coefficients(32, 0, true);
        assert!(Q.polynomial_lift_is_bounded(&few, &few, 63));
        assert!(!Q.polynomial_lift_is_bounded(&few[..2], &few[..3], 4));
        let huge_denominator = Integer::one() << 8192u32;
        let common = vec![Q.to_element(1.into(), huge_denominator, true); 32];
        assert!(Q.polynomial_lift_is_bounded(&common, &common, 63));
    }

    #[test]
    fn fraction_kernel_declines_low_collision_nonintegral_products() {
        let left_indices: Vec<u32> = (0..32).map(|i| i * 32).collect();
        let right_indices: Vec<u32> = (0..32).collect();
        let coefficients = coefficients(32, 59, true);
        assert!(!Q.polynomial_lift_is_bounded(&coefficients, &coefficients, 1024));
        let kernels = Q.kernels();
        assert!(
            kernels
                .polynomial()
                .unwrap()
                .try_dense_mul(DensePolynomialMulRequest {
                    output_len: 1024,
                    left_coefficients: &coefficients,
                    left_indices: &left_indices,
                    right_coefficients: &coefficients,
                    right_indices: &right_indices,
                })
                .is_none()
        );
        let integers: Vec<Rational> = coefficients
            .iter()
            .map(|c| Rational::from(c.numerator()))
            .collect();
        assert!(Q.polynomial_lift_is_bounded(&integers, &integers, 1024));
    }
}
