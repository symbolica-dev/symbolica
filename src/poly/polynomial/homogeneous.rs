//! Dense multiplication after eliminating a redundant homogeneous coordinate.

use super::{
    DensePolynomialMulRequest, Exponent, LexOrder, MAX_DENSE_MUL_BUFFER_SIZE,
    MultivariatePolynomial, Ring,
};

const MIN_PAIR_PRODUCTS: usize = 1 << 20;
const MIN_PRODUCTS_PER_CELL: usize = 32;
const MAX_VERIFIED_CANDIDATES: usize = 8;

struct HomogeneousShape {
    variable: usize,
    subset: u8,
    degree: u64,
    radixes: Vec<usize>,
    output_len: usize,
}

fn subset_degree<E: Exponent>(exponents: &[E], subset: u8) -> u64 {
    exponents
        .iter()
        .enumerate()
        .filter(|(v, _)| subset & (1 << v) != 0)
        .map(|(_, e)| e.to_i32() as u64)
        .sum()
}

/// Compute a carry-free product grid, rejecting Laurent and overflowing exponents.
fn product_radixes<F: Ring, E: Exponent>(
    left: &MultivariatePolynomial<F, E>,
    right: &MultivariatePolynomial<F, E>,
) -> Option<Vec<usize>> {
    let maxima = |p: &MultivariatePolynomial<F, E>| {
        let mut degrees = vec![0i32; p.nvars()];
        for exponents in p.exponents_iter() {
            for (maximum, e) in degrees.iter_mut().zip(exponents) {
                let e = e.to_i32();
                if e < 0 {
                    return None;
                }
                *maximum = (*maximum).max(e);
            }
        }
        Some(degrees)
    };
    maxima(left)?
        .into_iter()
        .zip(maxima(right)?)
        .map(|(a, b)| {
            let degree = a.checked_add(b)?;
            E::try_from(degree).ok()?;
            (degree as usize).checked_add(1)
        })
        .collect()
}

impl HomogeneousShape {
    /// Candidates are in order of decreasing grid savings. Sampling only rejects
    /// candidates; a relation is always proved over both complete inputs before use.
    fn find<F: Ring, E: Exponent>(
        left: &MultivariatePolynomial<F, E>,
        right: &MultivariatePolynomial<F, E>,
        variables: &[usize],
        mut radixes: Vec<usize>,
    ) -> Option<Self> {
        let active = radixes
            .iter()
            .enumerate()
            .fold(0u8, |mask, (v, &r)| mask | if r > 1 { 1 << v } else { 0 });
        let mut candidates = Vec::new();
        for subset in 1u16..(1 << left.nvars()) {
            let subset = subset as u8;
            if subset.count_ones() < 2 || subset & !active != 0 {
                continue;
            }
            let Some(priority) = variables.iter().position(|v| subset & (1 << v) != 0) else {
                continue;
            };
            let degrees = [
                subset_degree(left.exponents(0), subset),
                subset_degree(right.exponents(0), subset),
            ];
            let plausible = [left, right].into_iter().zip(degrees).all(|(p, degree)| {
                p.exponents_iter()
                    .step_by((p.nterms() / 16).max(1))
                    .chain(std::iter::once(p.exponents(p.nterms() - 1)))
                    .all(|e| subset_degree(e, subset) == degree)
            });
            if plausible {
                candidates.push((priority, subset, degrees));
            }
        }
        candidates.sort_unstable_by_key(|(priority, subset, _)| (*priority, subset.count_ones()));
        // Bound unsuccessful full scans even for inputs engineered to fool sampling.
        for (priority, subset, degrees) in candidates.into_iter().take(MAX_VERIFIED_CANDIDATES) {
            if ![left, right].into_iter().zip(degrees).all(|(p, degree)| {
                p.exponents_iter()
                    .all(|e| subset_degree(e, subset) == degree)
            }) {
                continue;
            }
            let variable = variables[priority];
            radixes[variable] = 1;
            let output_len = radixes.iter().try_fold(1usize, |n, r| n.checked_mul(*r))?;
            return Some(Self {
                variable,
                subset,
                degree: degrees[0] + degrees[1],
                radixes,
                output_len,
            });
        }
        None
    }
}

impl<F: Ring, E: Exponent> MultivariatePolynomial<F, E, LexOrder> {
    /// Only replace an oversized dense grid when eliminating one variable yields
    /// a bounded, well-populated grid. Small products and existing bounded dense
    /// routes keep their dispatch. Rings without a suitable dense kernel fall back.
    #[inline]
    pub(super) fn try_homogeneous_dense_mul(&self, other: &Self) -> Option<Self> {
        if !(2..=8).contains(&self.nvars())
            || self.nterms().min(other.nterms()) < 2
            || self.nterms().saturating_mul(other.nterms()) < MIN_PAIR_PRODUCTS
        {
            return None;
        }
        self.ring().kernels().polynomial()?;
        self.try_large_homogeneous_dense_mul(other)
    }

    #[inline(never)]
    fn try_large_homogeneous_dense_mul(&self, other: &Self) -> Option<Self> {
        let radixes = product_radixes(self, other)?;
        let original_len = radixes.iter().fold(1usize, |n, r| n.saturating_mul(*r));
        if original_len <= MAX_DENSE_MUL_BUFFER_SIZE {
            return None;
        }
        let pairs = self.nterms().saturating_mul(other.nterms());
        let mut variables: Vec<_> = (0..self.nvars())
            .filter(|&v| {
                let reduced = radixes
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| *i != v)
                    .try_fold(1usize, |n, (_, r)| n.checked_mul(*r));
                radixes[v] > 1
                    && reduced.is_some_and(|n| {
                        n <= MAX_DENSE_MUL_BUFFER_SIZE && n <= pairs / MIN_PRODUCTS_PER_CELL
                    })
            })
            .collect();
        variables.sort_unstable_by_key(|&v| std::cmp::Reverse(radixes[v]));
        if variables.is_empty() {
            return None;
        }
        let shape = HomogeneousShape::find(self, other, &variables, radixes)?;
        self.mul_homogeneous_dense(other, &shape)
    }

    fn mul_homogeneous_dense(&self, other: &Self, shape: &HomogeneousShape) -> Option<Self> {
        let kernels = self.ring().kernels();
        let kernel = kernels.polynomial()?;
        // Removing a coordinate changes lexicographic order. Sort additive indices
        // and coefficients together to satisfy the dense kernel's input contract.
        // Homogeneity proves that each projected input index remains unique.
        let project = |p: &Self| {
            let mut order: Vec<_> =
                p.exponents_iter()
                    .enumerate()
                    .map(|(i, e)| {
                        let index = e.iter().zip(&shape.radixes).enumerate().fold(
                            0usize,
                            |n, (v, (e, r))| {
                                n * r
                                    + if v == shape.variable {
                                        0
                                    } else {
                                        e.to_i32() as usize
                                    }
                            },
                        );
                        (index as u32, i)
                    })
                    .collect();
            order.sort_unstable_by_key(|&(index, _)| index);
            let indices: Vec<_> = order.iter().map(|&(index, _)| index).collect();
            debug_assert!(indices.windows(2).all(|w| w[0] < w[1]));
            let coefficients: Vec<_> = order
                .into_iter()
                .map(|(_, i)| p.coefficients[i].clone())
                .collect();
            (indices, coefficients)
        };
        let (left_indices, left_coefficients) = project(self);
        let (right_indices, right_coefficients) = project(other);
        let coefficients = kernel.try_dense_mul(DensePolynomialMulRequest {
            output_len: shape.output_len,
            left_indices: &left_indices,
            left_coefficients: &left_coefficients,
            right_indices: &right_indices,
            right_coefficients: &right_coefficients,
        })?;
        let mut projected = self.zero_with_capacity(coefficients.len());
        for (position, coefficient) in coefficients {
            let mut position = position as usize;
            let start = projected.exponents.len();
            projected.exponents.resize(start + self.nvars(), E::zero());
            let exponents = &mut projected.exponents[start..];
            for (e, &radix) in exponents.iter_mut().zip(&shape.radixes).rev() {
                *e = E::try_from(i32::try_from(position % radix).ok()?).ok()?;
                position /= radix;
            }
            let missing = shape
                .degree
                .checked_sub(subset_degree(exponents, shape.subset))?;
            exponents[shape.variable] = E::try_from(i32::try_from(missing).ok()?).ok()?;
            projected.coefficients.push(coefficient);
        }
        // Restoring the coordinate changes order again. Move coefficients rather
        // than cloning them, preserving canonical lex order and the shared context.
        let mut order: Vec<_> = (0..projected.nterms()).collect();
        order.sort_unstable_by_key(|&i| projected.exponents(i));
        let mut result = self.zero_with_capacity(order.len());
        for i in order {
            result.coefficients.push(std::mem::replace(
                &mut projected.coefficients[i],
                self.ring().zero(),
            ));
            result.exponents.extend_from_slice(projected.exponents(i));
        }
        Some(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use std::{collections::BTreeMap, sync::Arc};

    fn variables() -> Arc<Vec<PolyVariable>> {
        Arc::new(
            ["h_d", "h_m", "h_u", "h_v", "h_w"]
                .map(|s| symbol!(s).into())
                .to_vec(),
        )
    }

    fn reference<E: Exponent>(
        a: &MultivariatePolynomial<IntegerRing, E>,
        b: &MultivariatePolynomial<IntegerRing, E>,
    ) -> MultivariatePolynomial<IntegerRing, E> {
        let mut terms = BTreeMap::<Vec<E>, Integer>::new();
        for x in a {
            for y in b {
                let e = x
                    .exponents
                    .iter()
                    .zip(y.exponents)
                    .map(|(x, y)| *x + *y)
                    .collect();
                *terms.entry(e).or_insert_with(|| Z.zero()) += x.coefficient * y.coefficient;
            }
        }
        let mut result = a.zero();
        for (e, c) in terms {
            result.append_monomial(c, &e);
        }
        result
    }

    #[test]
    fn homogeneous_restoration_handles_subsets_permutations_and_large_coefficients() {
        let a = parse!("(h_d^2+3*h_d+1)*(h_m^2+2*h_m*h_u-h_u^2+3*h_v*h_w)")
            .to_polynomial::<_, u16>(&Z, Some(variables()));
        let b = parse!("(h_d+2)*(h_m^3-3*h_m*h_u^2+2*h_u^3+h_v^2*h_w)")
            .to_polynomial::<_, u16>(&Z, Some(variables()));
        for permutation in [[0, 1, 2, 3, 4], [2, 4, 0, 1, 3], [4, 3, 2, 1, 0]] {
            let a = a.rearrange(&permutation, false);
            let b = b.rearrange(&permutation, false);
            let expected = reference(&a, &b);
            // Each homogeneous variable can be eliminated, including first/last
            // coordinates and a subset interrupted by an independent variable.
            for variable in 0..5 {
                let shape =
                    HomogeneousShape::find(&a, &b, &[variable], product_radixes(&a, &b).unwrap());
                if permutation[variable] == 0 {
                    assert!(shape.is_none());
                    continue;
                }
                let shape = shape.unwrap();
                assert_eq!(shape.degree, 5);
                assert_eq!(a.mul_homogeneous_dense(&b, &shape).unwrap(), expected);
            }
        }
        // Large-coefficient dense kernels are currently supplied by the GMP backend.
        #[cfg(feature = "integer-gmp")]
        {
            let big = parse!("12345678901234567890123456789012345678901234567890")
                .to_polynomial::<_, u16>(&Z, Some(variables()))
                .get_constant();
            let a = a.mul_coeff(big);
            let shape =
                HomogeneousShape::find(&a, &b, &[1], product_radixes(&a, &b).unwrap()).unwrap();
            assert_eq!(
                a.mul_homogeneous_dense(&b, &shape).unwrap(),
                reference(&a, &b)
            );
        }
    }

    #[test]
    fn homogeneous_restoration_preserves_cancellation_in_integer_and_finite_fields() {
        let a = parse!("h_m+h_u").to_polynomial::<_, u16>(&Z, Some(variables()));
        let b = parse!("h_m-h_u").to_polynomial::<_, u16>(&Z, Some(variables()));
        let shape = HomogeneousShape::find(&a, &b, &[1], product_radixes(&a, &b).unwrap()).unwrap();
        let expected = reference(&a, &b);
        assert_eq!(expected.nterms(), 2);
        assert_eq!(a.mul_homogeneous_dense(&b, &shape).unwrap(), expected);
        let field = Zp::new(17);
        let map = |p: &MultivariatePolynomial<IntegerRing, u16>| {
            p.map_coeff(|c| field.nth(c.clone()), field.clone())
        };
        assert_eq!(
            map(&a).mul_homogeneous_dense(&map(&b), &shape).unwrap(),
            map(&expected)
        );
        assert!(
            a.try_homogeneous_dense_mul(&b).is_none(),
            "small products keep existing dispatch"
        );
    }

    fn large_homogeneous() -> MultivariatePolynomial<IntegerRing, u16> {
        let mut coefficients = Vec::new();
        let mut exponents = Vec::new();
        for i in 0..=1024u16 {
            coefficients.push(Integer::from(if i % 3 == 0 { -2 } else { 1 }));
            exponents.extend([0, 4096 - i, i, 0, 0]);
        }
        MultivariatePolynomial::from_coefficient_list(coefficients, exponents, variables(), &Z)
    }

    #[test]
    fn homogeneous_automatic_dispatch_matches_independent_convolution() {
        let a = large_homogeneous();
        let b = -a.clone();
        let expected = reference(&a, &b);
        assert_eq!(a.try_homogeneous_dense_mul(&b).unwrap(), expected);
        assert_eq!(&a * &b, expected);
        assert_eq!(&b * &a, expected);
    }

    #[test]
    fn homogeneous_detection_verifies_unsampled_terms_and_both_inputs() {
        let a = large_homogeneous();
        let mut exponents = a.exponents.clone();
        // This interior term is missed by the evenly spaced samples. Re-sort
        // after changing it so the rejection does not rely on malformed input.
        exponents[5 + 1] += 1;
        let bad = MultivariatePolynomial::from_coefficient_list(
            a.coefficients.clone(),
            exponents,
            variables(),
            &Z,
        );
        let candidates = [1, 2];
        for (a, b) in [(&a, &bad), (&bad, &a)] {
            assert!(
                HomogeneousShape::find(a, b, &candidates, product_radixes(a, b).unwrap()).is_none()
            );
            assert!(a.try_homogeneous_dense_mul(b).is_none());
        }
        let with_constant = parse!("1+h_m+h_u").to_polynomial::<_, u16>(&Z, Some(variables()));
        assert!(
            HomogeneousShape::find(
                &a,
                &with_constant,
                &candidates,
                product_radixes(&a, &with_constant).unwrap()
            )
            .is_none()
        );
    }

    #[test]
    fn homogeneous_detection_rejects_laurent_and_exponent_overflow() {
        let a = parse!("h_m^-1+h_u").to_polynomial::<_, i16>(&Z, Some(variables()));
        assert!(product_radixes(&a, &a).is_none());
        let a = parse!("h_m^128+h_u^128").to_polynomial::<_, u8>(&Z, Some(variables()));
        assert!(product_radixes(&a, &a).is_none());
    }
}
