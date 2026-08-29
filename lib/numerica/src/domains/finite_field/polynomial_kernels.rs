//! Representation-specific bulk kernels for finite-field and modular-ring polynomial arithmetic.
//!
//! Coefficients are stored in Montgomery form. These kernels accumulate exact integer products
//! for as long as their fixed-width bounds permit, then perform one modular/Montgomery reduction
//! per nonzero output coefficient. Large dense convolutions may instead use Kronecker
//! substitution and GMP. Each multiplication is represented by a short-lived operation context
//! that validates the request, selects an index layout, and dispatches to its strategy methods. A
//! strategy returns `None` when its representation or workspace conditions do not hold so that
//! the context can try another strategy or polynomial dispatch can fall back.

use super::montgomery::{montgomery_reduce_u32, montgomery_reduce_u64};
use super::{FiniteField, FiniteFieldElement, Zp, Zp64};
use crate::domains::RingOps;
use crate::domains::integer::{Integer, Z};
#[cfg(feature = "integer-gmp")]
use crate::domains::integer::{MultiPrecisionInteger, RawMultiPrecisionInteger};
use crate::domains::polynomial_layouts::{SimplexKroneckerLayout, try_simplex_kronecker_layout};
use crate::kernels::{DensePolynomialMulRequest, PolynomialKernels};
#[cfg(feature = "integer-gmp")]
use rug::integer::Order as RugIntegerOrder;

const MAX_DENSE_MODULAR_MUL_BUFFER_SIZE: usize = 1 << 20;
const MAX_SPARSE_INTEGER_MONTGOMERY_OUTPUT_LEN: usize = 1 << 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum U64AccumulationMode {
    DirectMontgomeryReduction,
    NativeRemainder,
}

#[inline]
/// Validate the common shape and bounded-workspace requirements of a dense multiplication.
///
/// In particular, coefficient and index slices must correspond, the output buffer is capped, and
/// the sum of the largest input indices must fit in the requested output.
fn validate_dense_polynomial_mul<UField>(
    output_len: usize,
    left_coefficients: &[FiniteFieldElement<UField>],
    left_indices: &[u32],
    right_coefficients: &[FiniteFieldElement<UField>],
    right_indices: &[u32],
) -> Option<()> {
    if output_len > MAX_DENSE_MODULAR_MUL_BUFFER_SIZE
        || left_coefficients.len() != left_indices.len()
        || right_coefficients.len() != right_indices.len()
    {
        return None;
    }

    if let (Some(&left_max), Some(&right_max)) = (left_indices.last(), right_indices.last()) {
        let maximum_index = left_max.checked_add(right_max)?;
        if maximum_index as usize >= output_len {
            return None;
        }
    }

    Some(())
}

/// One validated dense-indexed multiplication over an arbitrary-size odd modulus.
///
/// Coefficients are stored in Montgomery form. The context accumulates their exact products in
/// [`Integer`] output cells and performs one Montgomery reduction for each nonzero coefficient.
/// If the complete convolution coefficient can exceed the Montgomery reduction range, it is
/// first reduced modulo the field modulus.
pub(super) struct DenseIntegerMontgomeryMul<'a> {
    field: &'a FiniteField<Integer>,
    request: DensePolynomialMulRequest<'a, FiniteFieldElement<Integer>>,
    pub(super) direct_montgomery_reduction: bool,
}

impl<'a> DenseIntegerMontgomeryMul<'a> {
    /// Validate the dense multiplication and determine whether its exact sums fit one Montgomery
    /// reduction without a preceding modular remainder.
    #[inline]
    pub(super) fn new(
        field: &'a FiniteField<Integer>,
        request: DensePolynomialMulRequest<'a, FiniteFieldElement<Integer>>,
    ) -> Option<Self> {
        validate_dense_polynomial_mul(
            request.output_len,
            request.left_coefficients,
            request.left_indices,
            request.right_coefficients,
            request.right_indices,
        )?;

        let number_of_products = request
            .left_coefficients
            .len()
            .saturating_mul(request.right_coefficients.len());
        // Widely separated supports can have a large dense span but only a few actual products.
        // Decline the exact-accumulator kernel when its dense buffer would be disproportionate.
        if !request.left_coefficients.is_empty()
            && !request.right_coefficients.is_empty()
            && request.output_len > MAX_SPARSE_INTEGER_MONTGOMERY_OUTPUT_LEN
            && request.output_len > number_of_products.saturating_mul(4)
        {
            return None;
        }

        // Unique input indices bound the number of products in one output coefficient by the
        // shorter input. For raw Montgomery representatives below p, C * (p - 1) < R proves
        // C * (p - 1)^2 < pR, the input range accepted by one Montgomery reduction.
        let maximum_collision_count = request
            .left_coefficients
            .len()
            .min(request.right_coefficients.len());
        let mut maximum_scaled_input = &field.p - &Integer::one();
        maximum_scaled_input *= &Integer::from(maximum_collision_count as u64);
        let direct_montgomery_reduction = maximum_scaled_input <= field.r_mask;

        Some(Self {
            field,
            request,
            direct_montgomery_reduction,
        })
    }

    /// Multiply the request, returning nonzero coefficients in increasing dense-index order.
    fn run(self) -> Vec<(u32, FiniteFieldElement<Integer>)> {
        if self.request.left_coefficients.is_empty() || self.request.right_coefficients.is_empty() {
            return Vec::new();
        }

        let mut accumulators = vec![Integer::zero(); self.request.output_len];
        for (left, &left_index) in self
            .request
            .left_coefficients
            .iter()
            .zip(self.request.left_indices)
        {
            for (right, &right_index) in self
                .request
                .right_coefficients
                .iter()
                .zip(self.request.right_indices)
            {
                let output_index = left_index as usize + right_index as usize;
                Z.add_mul_assign(&mut accumulators[output_index], &left.0, &right.0);
            }
        }

        let mut result = Vec::new();
        for (position, accumulator) in accumulators.into_iter().enumerate() {
            if accumulator.is_zero() {
                continue;
            }
            let input = if self.direct_montgomery_reduction {
                accumulator
            } else {
                accumulator % &self.field.p
            };
            let coefficient = self.field.montgomery_reduce_integer(input);
            if !coefficient.is_zero() {
                result.push((position as u32, FiniteFieldElement(coefficient)));
            }
        }
        result
    }
}

/// One validated dense-indexed multiplication over a 32-bit prime field.
///
/// The context owns request validation, layout selection, strategy dispatch, and the quantities
/// shared by its direct and Kronecker strategies.
pub(super) struct DenseZpMul<'a> {
    field: &'a Zp,
    request: DensePolynomialMulRequest<'a, FiniteFieldElement<u32>>,
    number_of_products: usize,
    simplex_layout: Option<SimplexKroneckerLayout>,
}

impl<'a> DenseZpMul<'a> {
    /// Validate a request and select its additive index layout once.
    #[inline]
    pub(super) fn new(
        field: &'a Zp,
        request: DensePolynomialMulRequest<'a, FiniteFieldElement<u32>>,
    ) -> Option<Self> {
        validate_dense_polynomial_mul(
            request.output_len,
            request.left_coefficients,
            request.left_indices,
            request.right_coefficients,
            request.right_indices,
        )?;
        let number_of_products = request
            .left_coefficients
            .len()
            .checked_mul(request.right_coefficients.len())?;
        const MIN_SIMPLEX_OUTPUT_LEN: usize = 1 << 16;
        let simplex_layout = (!request.left_coefficients.is_empty()
            && !request.right_coefficients.is_empty()
            && number_of_products < request.output_len
            && request.output_len >= MIN_SIMPLEX_OUTPUT_LEN)
            .then(|| {
                try_simplex_kronecker_layout(
                    request.output_len,
                    request.left_indices,
                    request.right_indices,
                )
            })
            .flatten();

        Some(Self {
            field,
            request,
            number_of_products,
            simplex_layout,
        })
    }

    #[inline]
    fn output_len(&self) -> usize {
        self.simplex_layout
            .as_ref()
            .map_or(self.request.output_len, |layout| layout.output_len)
    }

    #[inline]
    fn indices(&self) -> (&[u32], &[u32]) {
        self.simplex_layout.as_ref().map_or(
            (self.request.left_indices, self.request.right_indices),
            |layout| {
                (
                    layout.left_indices.as_slice(),
                    layout.right_indices.as_slice(),
                )
            },
        )
    }

    /// Select the reduction used after accumulating exact output coefficients in `u64`.
    ///
    /// Returns `None` when the largest possible coefficient requires a `u128` accumulator.
    #[inline]
    pub(super) fn u64_accumulation_mode(&self) -> Option<U64AccumulationMode> {
        let maximum_product = (self.field.p - 1) as u128 * (self.field.p - 1) as u128;
        // Unique input indices imply that one output index receives at most one product for each
        // term of either input, hence at most the length of the shorter input.
        let maximum_collision_count = self
            .request
            .left_coefficients
            .len()
            .min(self.request.right_coefficients.len());
        let maximum_coefficient = maximum_product.checked_mul(maximum_collision_count as u128)?;
        if maximum_coefficient > u64::MAX as u128 {
            return None;
        }
        if maximum_coefficient < (self.field.p as u128) << u32::BITS {
            Some(U64AccumulationMode::DirectMontgomeryReduction)
        } else {
            Some(U64AccumulationMode::NativeRemainder)
        }
    }

    /// Decode a compact simplex layout back to the polynomial layer's standard dense indices.
    fn decode_output(
        &self,
        mut output: Vec<(u32, FiniteFieldElement<u32>)>,
    ) -> Option<Vec<(u32, FiniteFieldElement<u32>)>> {
        if let Some(layout) = self.simplex_layout.as_ref() {
            for (index, _) in &mut output {
                let decoded = *layout.decode_indices.get(*index as usize)?;
                if decoded == u32::MAX {
                    return None;
                }
                *index = decoded;
            }
            output.sort_unstable_by_key(|term| term.0);
        }
        Some(output)
    }

    #[cfg(feature = "integer-gmp")]
    /// Multiply this operation's `Zp` polynomials with two-evaluation Kronecker substitution.
    ///
    /// The inputs are packed at positive and negative powers of a binary radix. Two GMP products then
    /// separate even and odd convolution coefficients, which are extracted and converted back to the
    /// field's Montgomery representation. Returns `None` if the exact digits need more than 128 bits
    /// or either packed operand would exceed the memory bound.
    pub(super) fn try_ks2(&self) -> Option<Vec<(u32, FiniteFieldElement<u32>)>> {
        let field = self.field;
        let output_len = self.output_len();
        let left_coefficients = self.request.left_coefficients;
        let right_coefficients = self.request.right_coefficients;
        let (left_indices, right_indices) = self.indices();
        let left_bits = left_coefficients
            .iter()
            .map(|coefficient| u32::BITS - coefficient.0.leading_zeros())
            .max()? as usize;
        let right_bits = right_coefficients
            .iter()
            .map(|coefficient| u32::BITS - coefficient.0.leading_zeros())
            .max()? as usize;
        let collision_count = left_coefficients.len().min(right_coefficients.len());
        let collision_bits = usize::BITS as usize - (collision_count - 1).leading_zeros() as usize;
        let coefficient_bits = left_bits
            .checked_add(right_bits)?
            .checked_add(collision_bits)?;
        // A single Montgomery reduction is sufficient while the exact coefficient is below pR.
        // This avoids a generic integer remainder for the small-prime packed-convolution case.
        let direct_montgomery_reduction = coefficient_bits
            <= (u32::BITS - field.p.leading_zeros()) as usize + u32::BITS as usize - 1;
        let evaluation_bits = coefficient_bits.div_ceil(2).max(left_bits).max(right_bits);
        let digit_bits = evaluation_bits.checked_mul(2)?;
        if digit_bits > 128 {
            return None;
        }

        const MAX_PACKED_BITS: usize = 1 << 29;
        for maximum_index in [left_indices.iter().max()?, right_indices.iter().max()?] {
            if (*maximum_index as usize + 1).checked_mul(evaluation_bits)? > MAX_PACKED_BITS {
                return None;
            }
        }

        fn pack_plus_and_minus(
            coefficients: &[FiniteFieldElement<u32>],
            indices: &[u32],
            evaluation_bits: usize,
        ) -> Option<(MultiPrecisionInteger, MultiPrecisionInteger)> {
            let packed_bits =
                (indices.iter().copied().max()? as usize + 1).checked_mul(evaluation_bits)?;
            let limb_count = packed_bits.checked_add(63)? / 64;
            let mut plus_limbs = vec![0u64; limb_count];
            let mut odd_limbs = vec![0u64; limb_count];

            #[inline(always)]
            fn write_coefficient(limbs: &mut [u64], bit_index: usize, coefficient: u32) {
                let limb_index = bit_index / 64;
                let shift = bit_index % 64;
                unsafe {
                    *limbs.get_unchecked_mut(limb_index) |= (coefficient as u64) << shift;
                    if shift > 32 && limb_index + 1 < limbs.len() {
                        *limbs.get_unchecked_mut(limb_index + 1) |=
                            (coefficient as u64) >> (64 - shift);
                    }
                }
            }

            for (coefficient, &index) in coefficients.iter().zip(indices) {
                let bit_index = (index as usize).checked_mul(evaluation_bits)?;
                write_coefficient(&mut plus_limbs, bit_index, coefficient.0);
                if index % 2 == 1 {
                    write_coefficient(&mut odd_limbs, bit_index, coefficient.0);
                }
            }

            let plus = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from_digits(
                &plus_limbs,
                RugIntegerOrder::Lsf,
            ));
            let odd = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from_digits(
                &odd_limbs,
                RugIntegerOrder::Lsf,
            ));
            let minus = &plus - (odd << 1usize);
            Some((plus, minus))
        }

        let (left_plus, left_minus) =
            pack_plus_and_minus(left_coefficients, left_indices, evaluation_bits)?;
        let (right_plus, right_minus) =
            pack_plus_and_minus(right_coefficients, right_indices, evaluation_bits)?;
        let plus_product = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from(
            left_plus.as_raw() * right_plus.as_raw(),
        ));
        let minus_product = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from(
            left_minus.as_raw() * right_minus.as_raw(),
        ));
        let even_coefficients = (&plus_product + &minus_product) >> 1usize;
        let odd_coefficients = ((plus_product - minus_product) >> 1usize) >> evaluation_bits;
        debug_assert!(!even_coefficients.is_negative());
        debug_assert!(!odd_coefficients.is_negative());
        let even_limbs = even_coefficients.as_raw().as_limbs();
        let odd_limbs = odd_coefficients.as_raw().as_limbs();

        #[inline(always)]
        fn extract_limb(limbs: &[u64], limb_index: usize, shift: usize) -> u64 {
            let mut value = limbs.get(limb_index).copied().unwrap_or(0) >> shift;
            if shift != 0 {
                value |= limbs.get(limb_index + 1).copied().unwrap_or(0) << (64 - shift);
            }
            value
        }

        let mut output = Vec::new();
        for index in 0..output_len {
            let digit_index = index / 2;
            let bit_index = digit_index.checked_mul(digit_bits)?;
            let limb_index = bit_index / 64;
            let shift = bit_index % 64;
            let limbs = if index % 2 == 0 {
                even_limbs
            } else {
                odd_limbs
            };
            let mut low = extract_limb(limbs, limb_index, shift);
            let high = if digit_bits <= 64 {
                if digit_bits < 64 {
                    low &= (1u64 << digit_bits) - 1;
                }
                0
            } else {
                let high_bits = digit_bits - 64;
                let mut high = extract_limb(limbs, limb_index + 1, shift);
                if high_bits < 64 {
                    high &= (1u64 << high_bits) - 1;
                }
                high
            };
            if low == 0 && high == 0 {
                continue;
            }

            let coefficient = if direct_montgomery_reduction {
                debug_assert_eq!(high, 0);
                montgomery_reduce_u32(field, low)
            } else {
                let exact_coefficient = low as u128 | ((high as u128) << 64);
                let residue = (exact_coefficient % field.p as u128) as u64;
                montgomery_reduce_u32(field, residue)
            };
            if coefficient != 0 {
                output.push((index as u32, FiniteFieldElement(coefficient)));
            }
        }
        Some(output)
    }

    /// Select and run the best applicable strategy for this request.
    ///
    /// Large, dense products use two-evaluation Kronecker substitution when available. Other
    /// products use exact `u64` or `u128` accumulators, choosing a dense array or sparse
    /// touched-slot array from the product density.
    #[inline]
    pub(super) fn run(self) -> Option<Vec<(u32, FiniteFieldElement<u32>)>> {
        if self.request.left_coefficients.is_empty() || self.request.right_coefficients.is_empty() {
            return Some(Vec::new());
        }

        #[cfg(feature = "integer-gmp")]
        if self.number_of_products >= 1_000_000
            && self.output_len().saturating_mul(64) < self.number_of_products
            && let Some(output) = self.try_ks2()
        {
            return self.decode_output(output);
        }

        // Small moduli can accumulate the entire convolution in u64. Besides using a
        // smaller buffer, this avoids the compiler's generic 128-bit remainder helper.
        let output = match self.u64_accumulation_mode() {
            Some(U64AccumulationMode::DirectMontgomeryReduction) => self.multiply_direct_u64(true),
            Some(U64AccumulationMode::NativeRemainder) => self.multiply_direct_u64(false),
            None => self.multiply_direct_u128(),
        };
        self.decode_output(output)
    }

    /// Accumulate the convolution in exact `u64` coefficients.
    fn multiply_direct_u64(&self, reduce_directly: bool) -> Vec<(u32, FiniteFieldElement<u32>)> {
        let output_len = self.output_len();
        let left_coefficients = self.request.left_coefficients;
        let right_coefficients = self.request.right_coefficients;
        let (left_indices, right_indices) = self.indices();
        let modulus = self.field.p as u64;
        if self.number_of_products >= output_len {
            let mut accumulators = vec![0u64; output_len];
            for (left, &left_index) in left_coefficients.iter().zip(left_indices) {
                for (right, &right_index) in right_coefficients.iter().zip(right_indices) {
                    let position = left_index as usize + right_index as usize;
                    unsafe {
                        *accumulators.get_unchecked_mut(position) += left.0 as u64 * right.0 as u64;
                    }
                }
            }

            let mut result = Vec::new();
            for (position, accumulator) in accumulators.into_iter().enumerate() {
                if accumulator == 0 {
                    continue;
                }

                let coefficient = montgomery_reduce_u32(
                    self.field,
                    if reduce_directly {
                        accumulator
                    } else {
                        accumulator % modulus
                    },
                );
                if coefficient != 0 {
                    result.push((position as u32, FiniteFieldElement(coefficient)));
                }
            }
            return result;
        }

        let mut coefficient_indices = vec![0u32; output_len];
        let mut accumulators = Vec::<u64>::with_capacity(left_coefficients.len());
        for (left, &left_index) in left_coefficients.iter().zip(left_indices) {
            for (right, &right_index) in right_coefficients.iter().zip(right_indices) {
                let position = left_index as usize + right_index as usize;
                let accumulator_index = unsafe { coefficient_indices.get_unchecked_mut(position) };
                let product = left.0 as u64 * right.0 as u64;
                if *accumulator_index == 0 {
                    accumulators.push(product);
                    *accumulator_index = accumulators.len() as u32;
                } else {
                    unsafe {
                        *accumulators.get_unchecked_mut(*accumulator_index as usize - 1) += product;
                    }
                }
            }
        }

        let mut result = Vec::with_capacity(accumulators.len());
        for (position, accumulator_index) in coefficient_indices.into_iter().enumerate() {
            if accumulator_index == 0 {
                continue;
            }

            let accumulator =
                unsafe { *accumulators.get_unchecked(accumulator_index as usize - 1) };
            let coefficient = montgomery_reduce_u32(
                self.field,
                if reduce_directly {
                    accumulator
                } else {
                    accumulator % modulus
                },
            );
            if coefficient != 0 {
                result.push((position as u32, FiniteFieldElement(coefficient)));
            }
        }
        result
    }

    /// Accumulate the convolution in exact `u128` coefficients.
    fn multiply_direct_u128(&self) -> Vec<(u32, FiniteFieldElement<u32>)> {
        let output_len = self.output_len();
        let left_coefficients = self.request.left_coefficients;
        let right_coefficients = self.request.right_coefficients;
        let (left_indices, right_indices) = self.indices();
        let modulus = self.field.p as u128;

        // A coefficient receives at most min(left.len(), right.len()) products. With the
        // buffer bound above, the sum of products of two u32 residues always fits in u128.
        if self.number_of_products >= output_len {
            let mut accumulators = vec![0u128; output_len];
            for (left, &left_index) in left_coefficients.iter().zip(left_indices) {
                for (right, &right_index) in right_coefficients.iter().zip(right_indices) {
                    let position = left_index as usize + right_index as usize;
                    unsafe {
                        *accumulators.get_unchecked_mut(position) +=
                            left.0 as u128 * right.0 as u128;
                    }
                }
            }

            let mut result = Vec::new();
            for (position, accumulator) in accumulators.into_iter().enumerate() {
                if accumulator == 0 {
                    continue;
                }

                let residue = (accumulator % modulus) as u64;
                let coefficient = montgomery_reduce_u32(self.field, residue);
                if coefficient != 0 {
                    result.push((position as u32, FiniteFieldElement(coefficient)));
                }
            }
            return result;
        }

        let mut coefficient_indices = vec![0u32; output_len];
        let mut accumulators = Vec::<u128>::with_capacity(left_coefficients.len());
        for (left, &left_index) in left_coefficients.iter().zip(left_indices) {
            for (right, &right_index) in right_coefficients.iter().zip(right_indices) {
                let position = left_index as usize + right_index as usize;
                let accumulator_index = unsafe { coefficient_indices.get_unchecked_mut(position) };
                let product = left.0 as u128 * right.0 as u128;
                if *accumulator_index == 0 {
                    accumulators.push(product);
                    *accumulator_index = accumulators.len() as u32;
                } else {
                    unsafe {
                        *accumulators.get_unchecked_mut(*accumulator_index as usize - 1) += product;
                    }
                }
            }
        }

        let mut result = Vec::with_capacity(accumulators.len());
        for (position, accumulator_index) in coefficient_indices.into_iter().enumerate() {
            if accumulator_index == 0 {
                continue;
            }

            let accumulator =
                unsafe { *accumulators.get_unchecked(accumulator_index as usize - 1) };
            let residue = (accumulator % modulus) as u64;
            let coefficient = montgomery_reduce_u32(self.field, residue);
            if coefficient != 0 {
                result.push((position as u32, FiniteFieldElement(coefficient)));
            }
        }
        result
    }
}

#[inline(always)]
/// Reduce a three-limb exact sum of products to the `Zp64` Montgomery representation.
///
/// The specialized large-modulus branch avoids overflowing intermediate products; the general
/// branch evaluates the three base-2^64 limbs modulo the field prime.
fn montgomery_reduce_three_limbs(field: &Zp64, low: u64, middle: u64, high: u64) -> u64 {
    if field.p >= 1u64 << 63 {
        let normalized_middle = if middle >= field.p {
            middle - field.p
        } else {
            middle
        };
        let lower = montgomery_reduce_u64(field, low as u128 | ((normalized_middle as u128) << 64));

        let radix = field.p.wrapping_neg();
        let upper = if let Some(value) = high.checked_mul(radix) {
            if value >= field.p {
                value % field.p
            } else {
                value
            }
        } else {
            (high as u128 * radix as u128 % field.p as u128) as u64
        };
        return if lower >= field.p - upper {
            lower - (field.p - upper)
        } else {
            lower + upper
        };
    }

    let modulus = field.p as u128;
    let radix = (1u128 << 64) % modulus;
    let radix_squared = radix * radix % modulus;
    let mut residue = low as u128 % modulus;
    residue += middle as u128 * radix % modulus;
    if residue >= modulus {
        residue -= modulus;
    }
    residue += high as u128 * radix_squared % modulus;
    if residue >= modulus {
        residue -= modulus;
    }
    montgomery_reduce_u64(field, residue)
}

#[inline(always)]
/// Add a `u128` product to a little-endian three-limb accumulator.
///
/// # Safety
///
/// `accumulator` must point to three consecutive writable `u64` limbs, and their exact sum must
/// fit in 192 bits. The enclosing convolution establishes both conditions.
unsafe fn add_u128_to_three_limbs(accumulator: *mut u64, product: u128) {
    let (low, carry_low) = unsafe { *accumulator }.overflowing_add(product as u64);
    unsafe { *accumulator = low };
    let (high, carry_high) = unsafe { *accumulator.add(1) }.overflowing_add((product >> 64) as u64);
    let (high, carry_from_low) = high.overflowing_add(carry_low as u64);
    unsafe {
        *accumulator.add(1) = high;
        *accumulator.add(2) += carry_high as u64 + carry_from_low as u64;
    }
}

/// One validated dense-indexed multiplication over a 64-bit prime field.
///
/// The context owns request validation, optional simplex remapping, strategy selection, and output
/// decoding.
pub(super) struct DenseZp64Mul<'a> {
    field: &'a Zp64,
    request: DensePolynomialMulRequest<'a, FiniteFieldElement<u64>>,
    product_count: usize,
    simplex_layout: Option<SimplexKroneckerLayout>,
}

impl<'a> DenseZp64Mul<'a> {
    /// Validate a request and select its additive index layout once.
    #[inline]
    pub(super) fn new(
        field: &'a Zp64,
        request: DensePolynomialMulRequest<'a, FiniteFieldElement<u64>>,
    ) -> Option<Self> {
        validate_dense_polynomial_mul(
            request.output_len,
            request.left_coefficients,
            request.left_indices,
            request.right_coefficients,
            request.right_indices,
        )?;
        let product_count = request
            .left_coefficients
            .len()
            .checked_mul(request.right_coefficients.len())?;
        const MIN_SIMPLEX_OUTPUT_LEN: usize = 1 << 16;
        let simplex_layout = (!request.left_coefficients.is_empty()
            && !request.right_coefficients.is_empty()
            && (cfg!(test) || product_count >= 20_000_000)
            && request.output_len >= MIN_SIMPLEX_OUTPUT_LEN)
            .then(|| {
                try_simplex_kronecker_layout(
                    request.output_len,
                    request.left_indices,
                    request.right_indices,
                )
            })
            .flatten();

        Some(Self {
            field,
            request,
            product_count,
            simplex_layout,
        })
    }

    #[inline]
    fn output_len(&self) -> usize {
        self.simplex_layout
            .as_ref()
            .map_or(self.request.output_len, |layout| layout.output_len)
    }

    #[inline]
    fn indices(&self) -> (&[u32], &[u32]) {
        self.simplex_layout.as_ref().map_or(
            (self.request.left_indices, self.request.right_indices),
            |layout| {
                (
                    layout.left_indices.as_slice(),
                    layout.right_indices.as_slice(),
                )
            },
        )
    }

    /// Decode a compact simplex layout back to the polynomial layer's standard dense indices.
    fn decode_output(
        &self,
        mut output: Vec<(u32, FiniteFieldElement<u64>)>,
    ) -> Option<Vec<(u32, FiniteFieldElement<u64>)>> {
        if let Some(layout) = self.simplex_layout.as_ref() {
            for (index, _) in &mut output {
                let decoded = *layout.decode_indices.get(*index as usize)?;
                if decoded == u32::MAX {
                    return None;
                }
                *index = decoded;
            }
            output.sort_unstable_by_key(|term| term.0);
        }
        Some(output)
    }

    #[cfg(feature = "integer-gmp")]
    #[inline(never)]
    /// Multiply this operation's `Zp64` polynomials with four GMP evaluation products.
    ///
    /// Forward and reversed, parity-separated packings recover exact coefficients whose overlapping
    /// digits occupy between one and two limbs. This is the 65--127-bit digit case; smaller products
    /// are cheaper in fixed-width accumulators and wider digits are handled by another kernel.
    /// Returns `None` when those width constraints or the packed-memory bound do not hold.
    pub(super) fn try_ks4(&self) -> Option<Vec<(u32, FiniteFieldElement<u64>)>> {
        let field = self.field;
        let left_coefficients = self.request.left_coefficients;
        let right_coefficients = self.request.right_coefficients;
        let (left_indices, right_indices) = self.indices();
        let left_len = left_indices.iter().copied().max()? as usize + 1;
        let right_len = right_indices.iter().copied().max()? as usize + 1;
        let product_len = left_len.checked_add(right_len)?.checked_sub(1)?;
        let left_bits = left_coefficients
            .iter()
            .map(|coefficient| u64::BITS - coefficient.0.leading_zeros())
            .max()? as usize;
        let right_bits = right_coefficients
            .iter()
            .map(|coefficient| u64::BITS - coefficient.0.leading_zeros())
            .max()? as usize;
        let collision_count = left_coefficients.len().min(right_coefficients.len());
        let collision_bits = usize::BITS - (collision_count - 1).leading_zeros();
        let coefficient_bits = left_bits
            .checked_add(right_bits)?
            .checked_add(collision_bits as usize)?;
        let evaluation_bits = coefficient_bits.div_ceil(4);
        let digit_bits = evaluation_bits.checked_mul(2)?;

        // This is the two-limb overlapping-digit recovery case. Smaller moduli are
        // handled more efficiently by the fixed-width accumulator above.
        if digit_bits <= 64 || digit_bits >= 128 {
            return None;
        }

        const MAX_PACKED_BITS: usize = 1 << 29;
        for length in [left_len, right_len] {
            if length.checked_add(1)?.checked_mul(evaluation_bits)? > MAX_PACKED_BITS {
                return None;
            }
        }

        fn pack_evaluations(
            coefficients: &[FiniteFieldElement<u64>],
            indices: &[u32],
            length: usize,
            evaluation_bits: usize,
            reverse: bool,
        ) -> Option<(MultiPrecisionInteger, MultiPrecisionInteger)> {
            let packed_bits = length
                .checked_add(1)?
                .checked_mul(evaluation_bits)?
                .checked_add(1)?;
            let limb_count = packed_bits.checked_add(63)? / 64;
            let mut even_limbs = vec![0u64; limb_count];
            let mut odd_limbs = vec![0u64; limb_count];

            #[inline(always)]
            fn write_coefficient(limbs: &mut [u64], bit_index: usize, coefficient: u64) {
                let limb_index = bit_index / 64;
                let shift = bit_index % 64;
                unsafe {
                    *limbs.get_unchecked_mut(limb_index) |= coefficient << shift;
                    if shift != 0 {
                        *limbs.get_unchecked_mut(limb_index + 1) |= coefficient >> (64 - shift);
                    }
                }
            }

            for (coefficient, &index) in coefficients.iter().zip(indices) {
                let exponent = if reverse {
                    length.checked_sub(index as usize + 1)?
                } else {
                    index as usize
                };
                let bit_index = exponent.checked_mul(evaluation_bits)?;
                if exponent % 2 == 0 {
                    write_coefficient(&mut even_limbs, bit_index, coefficient.0);
                } else {
                    write_coefficient(&mut odd_limbs, bit_index, coefficient.0);
                }
            }

            let even = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from_digits(
                &even_limbs,
                RugIntegerOrder::Lsf,
            ));
            let odd = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from_digits(
                &odd_limbs,
                RugIntegerOrder::Lsf,
            ));
            let plus = &even + &odd;
            let minus = even - odd;
            Some((plus, minus))
        }

        fn multiply_evaluations(
            left: (MultiPrecisionInteger, MultiPrecisionInteger),
            right: (MultiPrecisionInteger, MultiPrecisionInteger),
        ) -> (MultiPrecisionInteger, MultiPrecisionInteger) {
            let plus_product = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from(
                left.0.as_raw() * right.0.as_raw(),
            ));
            let minus_product = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from(
                left.1.as_raw() * right.1.as_raw(),
            ));
            let even = &plus_product + &minus_product;
            let odd = plus_product - minus_product;
            debug_assert!(!even.is_negative());
            debug_assert!(!odd.is_negative());
            (even, odd)
        }

        let normal = multiply_evaluations(
            pack_evaluations(
                left_coefficients,
                left_indices,
                left_len,
                evaluation_bits,
                false,
            )?,
            pack_evaluations(
                right_coefficients,
                right_indices,
                right_len,
                evaluation_bits,
                false,
            )?,
        );
        let reverse = multiply_evaluations(
            pack_evaluations(
                left_coefficients,
                left_indices,
                left_len,
                evaluation_bits,
                true,
            )?,
            pack_evaluations(
                right_coefficients,
                right_indices,
                right_len,
                evaluation_bits,
                true,
            )?,
        );

        let mut coefficients = vec![0u64; product_len];

        fn recover_coefficients(
            field: &Zp64,
            output: &mut [u64],
            stride: usize,
            normal: &[u64],
            normal_offset: usize,
            reverse: &[u64],
            reverse_offset: usize,
            count: usize,
            digit_bits: usize,
        ) {
            #[inline(always)]
            fn pair_less(left_high: u64, left_low: u64, right_high: u64, right_low: u64) -> bool {
                left_high < right_high || (left_high == right_high && left_low < right_low)
            }

            #[inline(always)]
            fn pair_sub(
                left_high: u64,
                left_low: u64,
                right_high: u64,
                right_low: u64,
            ) -> (u64, u64) {
                let (low, borrow) = left_low.overflowing_sub(right_low);
                (
                    left_high
                        .wrapping_sub(right_high)
                        .wrapping_sub(borrow as u64),
                    low,
                )
            }

            #[inline(always)]
            fn extract_digit(
                limbs: &[u64],
                index: usize,
                digit_bits: usize,
                bit_offset: usize,
                high_mask: u64,
            ) -> (u64, u64) {
                let bit_index = bit_offset + index * digit_bits;
                let limb_index = bit_index / 64;
                let shift = bit_index % 64;
                let extract_limb = |limb_index: usize| {
                    let mut value = limbs.get(limb_index).copied().unwrap_or(0) >> shift;
                    if shift != 0 {
                        value |= limbs.get(limb_index + 1).copied().unwrap_or(0) << (64 - shift);
                    }
                    value
                };
                (
                    extract_limb(limb_index),
                    extract_limb(limb_index + 1) & high_mask,
                )
            }

            let high_bits = digit_bits - 64;
            let high_mask = (1u64 << high_bits) - 1;
            let complementary_bits = 128 - digit_bits;
            let (mut normal_low, mut normal_high) =
                extract_digit(normal, 0, digit_bits, normal_offset, high_mask);
            let (mut reverse_carry_low, mut reverse_carry_high) =
                extract_digit(reverse, count, digit_bits, reverse_offset, high_mask);
            let mut borrow = false;

            for coefficient_index in 0..count {
                let (reverse_digit_low, reverse_digit_high) = extract_digit(
                    reverse,
                    count - coefficient_index - 1,
                    digit_bits,
                    reverse_offset,
                    high_mask,
                );
                let (normal_next_low, normal_next_high) = extract_digit(
                    normal,
                    coefficient_index + 1,
                    digit_bits,
                    normal_offset,
                    high_mask,
                );

                if pair_less(
                    reverse_digit_high,
                    reverse_digit_low,
                    normal_high,
                    normal_low,
                ) {
                    let (low, underflow) = reverse_carry_low.overflowing_sub(1);
                    reverse_carry_low = low;
                    reverse_carry_high = reverse_carry_high.wrapping_sub(underflow as u64);
                }

                let low = normal_low;
                let middle = (reverse_carry_low << high_bits).wrapping_add(normal_high);
                let high = (reverse_carry_high << high_bits)
                    .wrapping_add(reverse_carry_low >> complementary_bits);
                output[coefficient_index * stride] =
                    montgomery_reduce_three_limbs(field, low, middle, high);

                if borrow {
                    let (low, overflow) = reverse_carry_low.overflowing_add(1);
                    reverse_carry_low = low;
                    reverse_carry_high = reverse_carry_high.wrapping_add(overflow as u64);
                }
                borrow = pair_less(
                    normal_next_high,
                    normal_next_low,
                    reverse_carry_high,
                    reverse_carry_low,
                );
                let (next_high, next_low) = pair_sub(
                    normal_next_high,
                    normal_next_low,
                    reverse_carry_high,
                    reverse_carry_low,
                );
                let (carry_high, carry_low) = pair_sub(
                    reverse_digit_high,
                    reverse_digit_low,
                    normal_high,
                    normal_low,
                );
                reverse_carry_high = carry_high & high_mask;
                reverse_carry_low = carry_low;
                normal_high = next_high & high_mask;
                normal_low = next_low;
            }
        }

        let even_count = product_len.div_ceil(2);
        let odd_count = product_len / 2;

        let (reverse_even_value, reverse_even_offset, reverse_odd_value, reverse_odd_offset) =
            if product_len % 2 == 1 {
                (&reverse.0, 1, &reverse.1, evaluation_bits + 1)
            } else {
                (&reverse.1, evaluation_bits + 1, &reverse.0, 1)
            };
        recover_coefficients(
            field,
            &mut coefficients,
            2,
            normal.0.as_raw().as_limbs(),
            1,
            reverse_even_value.as_raw().as_limbs(),
            reverse_even_offset,
            even_count,
            digit_bits,
        );
        recover_coefficients(
            field,
            &mut coefficients[1..],
            2,
            normal.1.as_raw().as_limbs(),
            evaluation_bits + 1,
            reverse_odd_value.as_raw().as_limbs(),
            reverse_odd_offset,
            odd_count,
            digit_bits,
        );

        Some(
            coefficients
                .into_iter()
                .enumerate()
                .filter_map(|(index, coefficient)| {
                    (coefficient != 0).then_some((index as u32, FiniteFieldElement(coefficient)))
                })
                .collect(),
        )
    }

    #[cfg(feature = "integer-gmp")]
    #[inline(never)]
    /// Multiply this operation's `Zp64` polynomials with positive/negative Kronecker evaluations.
    ///
    /// Two GMP products separate even and odd convolution coefficients. Each exact digit occupies up
    /// to three limbs and is reduced once into Montgomery form. Returns `None` when the packed operands
    /// exceed the memory bound or an arithmetic size calculation overflows.
    pub(super) fn try_ks2(&self) -> Option<Vec<(u32, FiniteFieldElement<u64>)>> {
        let field = self.field;
        let output_len = self.output_len();
        let left_coefficients = self.request.left_coefficients;
        let right_coefficients = self.request.right_coefficients;
        let (left_indices, right_indices) = self.indices();
        let left_bits = left_coefficients
            .iter()
            .map(|coefficient| u64::BITS - coefficient.0.leading_zeros())
            .max()? as usize;
        let right_bits = right_coefficients
            .iter()
            .map(|coefficient| u64::BITS - coefficient.0.leading_zeros())
            .max()? as usize;
        let collision_count = left_coefficients.len().min(right_coefficients.len());
        let collision_bits = usize::BITS - (collision_count - 1).leading_zeros();
        let coefficient_bits = left_bits
            .checked_add(right_bits)?
            .checked_add(collision_bits as usize)?;
        let evaluation_bits = coefficient_bits.div_ceil(2).max(left_bits).max(right_bits);
        let digit_bits = evaluation_bits.checked_mul(2)?;

        const MAX_PACKED_BITS: usize = 1 << 29;
        for maximum_index in [left_indices.iter().max()?, right_indices.iter().max()?] {
            if (*maximum_index as usize + 1).checked_mul(evaluation_bits)? > MAX_PACKED_BITS {
                return None;
            }
        }

        fn pack_plus_and_minus(
            coefficients: &[FiniteFieldElement<u64>],
            indices: &[u32],
            evaluation_bits: usize,
        ) -> Option<(MultiPrecisionInteger, MultiPrecisionInteger)> {
            let packed_bits =
                (indices.iter().copied().max()? as usize + 1).checked_mul(evaluation_bits)?;
            let limb_count = packed_bits.checked_add(63)? / 64;
            let mut plus_limbs = vec![0u64; limb_count];
            let mut odd_limbs = vec![0u64; limb_count];

            #[inline(always)]
            fn write_coefficient(limbs: &mut [u64], bit_index: usize, coefficient: u64) {
                let limb_index = bit_index / 64;
                let shift = bit_index % 64;
                unsafe {
                    *limbs.get_unchecked_mut(limb_index) |= coefficient << shift;
                    if shift != 0 && limb_index + 1 < limbs.len() {
                        *limbs.get_unchecked_mut(limb_index + 1) |= coefficient >> (64 - shift);
                    }
                }
            }

            for (coefficient, &index) in coefficients.iter().zip(indices) {
                let bit_index = (index as usize).checked_mul(evaluation_bits)?;
                write_coefficient(&mut plus_limbs, bit_index, coefficient.0);
                if index % 2 == 1 {
                    write_coefficient(&mut odd_limbs, bit_index, coefficient.0);
                }
            }

            let plus = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from_digits(
                &plus_limbs,
                RugIntegerOrder::Lsf,
            ));
            let odd = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from_digits(
                &odd_limbs,
                RugIntegerOrder::Lsf,
            ));
            let minus = &plus - (odd << 1usize);
            Some((plus, minus))
        }

        let (left_plus, left_minus) =
            pack_plus_and_minus(left_coefficients, left_indices, evaluation_bits)?;
        let (right_plus, right_minus) =
            pack_plus_and_minus(right_coefficients, right_indices, evaluation_bits)?;
        let plus_product = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from(
            left_plus.as_raw() * right_plus.as_raw(),
        ));
        let minus_product = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from(
            left_minus.as_raw() * right_minus.as_raw(),
        ));

        let mut even_coefficients = &plus_product + &minus_product;
        even_coefficients = even_coefficients >> 1usize;
        let mut odd_coefficients = plus_product - minus_product;
        odd_coefficients = odd_coefficients >> 1usize;
        debug_assert!(!even_coefficients.is_negative());
        debug_assert!(!odd_coefficients.is_negative());

        let even_limbs = even_coefficients.as_raw().as_limbs();
        let odd_limbs = odd_coefficients.as_raw().as_limbs();
        let top_bits = digit_bits.saturating_sub(128);
        debug_assert!(digit_bits <= 192);

        #[inline(always)]
        fn extract_limb(limbs: &[u64], limb_index: usize, shift: usize) -> u64 {
            let mut value = limbs.get(limb_index).copied().unwrap_or(0) >> shift;
            if shift != 0 {
                value |= limbs.get(limb_index + 1).copied().unwrap_or(0) << (64 - shift);
            }
            value
        }

        let mut output = Vec::new();
        for index in 0..output_len {
            let bit_index = index.checked_mul(evaluation_bits)?;
            let limb_index = bit_index / 64;
            let shift = bit_index % 64;
            let limbs = if index % 2 == 0 {
                even_limbs
            } else {
                odd_limbs
            };
            let mut low = extract_limb(limbs, limb_index, shift);
            let mut high = extract_limb(limbs, limb_index + 1, shift);
            let mut top = extract_limb(limbs, limb_index + 2, shift);
            if digit_bits < 64 {
                low &= (1u64 << digit_bits) - 1;
                high = 0;
                top = 0;
            } else if digit_bits == 64 {
                high = 0;
                top = 0;
            } else if digit_bits < 128 {
                high &= (1u64 << (digit_bits - 64)) - 1;
                top = 0;
            } else if digit_bits == 128 {
                top = 0;
            } else if top_bits < 64 {
                top &= (1u64 << top_bits) - 1;
            }
            if low == 0 && high == 0 && top == 0 {
                continue;
            }

            let coefficient = montgomery_reduce_three_limbs(field, low, high, top);
            if coefficient != 0 {
                output.push((index as u32, FiniteFieldElement(coefficient)));
            }
        }
        Some(output)
    }

    /// Select and run the best applicable strategy for this request.
    ///
    /// Very dense inputs first try the four- or two-product Kronecker kernels. The direct strategy
    /// stores each exact coefficient in a three-limb accumulator, optionally using a blocked pair
    /// loop, and reduces it only after the convolution is complete.
    #[inline]
    pub(super) fn run(self) -> Option<Vec<(u32, FiniteFieldElement<u64>)>> {
        if self.request.left_coefficients.is_empty() || self.request.right_coefficients.is_empty() {
            return Some(Vec::new());
        }

        #[cfg(feature = "integer-gmp")]
        if self.product_count >= 1_000_000
            && self.product_count >= self.output_len().saturating_mul(128)
        {
            let (left_indices, right_indices) = self.indices();
            let densely_encoded = left_indices
                .last()
                .is_some_and(|&index| index as usize + 1 == self.request.left_coefficients.len())
                && right_indices.last().is_some_and(|&index| {
                    index as usize + 1 == self.request.right_coefficients.len()
                });
            if densely_encoded {
                if let Some(output) = self.try_ks4() {
                    return self.decode_output(output);
                }
                if let Some(output) = self.try_ks2() {
                    return self.decode_output(output);
                }
            } else {
                if let Some(output) = self.try_ks2() {
                    return self.decode_output(output);
                }
                if let Some(output) = self.try_ks4() {
                    return self.decode_output(output);
                }
            }
        }

        let output = self.multiply_direct();
        self.decode_output(output)
    }

    /// Accumulate exact `u128` products in three-limb output coefficients.
    fn multiply_direct(&self) -> Vec<(u32, FiniteFieldElement<u64>)> {
        let output_len = self.output_len();
        let left_coefficients = self.request.left_coefficients;
        let right_coefficients = self.request.right_coefficients;
        let (left_indices, right_indices) = self.indices();

        // Accumulate exact 128-bit products in three limbs. This postpones Montgomery
        // reduction until an output coefficient is complete.
        let right_terms = right_coefficients
            .iter()
            .zip(right_indices)
            .map(|(coefficient, &index)| (coefficient.0, index as usize * 3))
            .collect::<Vec<_>>();
        let mut accumulators = vec![0u64; output_len * 3];
        if self.product_count >= 1_000_000 {
            const BLOCK_SIZE: usize = 64;
            for left_block in (0..left_coefficients.len()).step_by(BLOCK_SIZE) {
                for right_block in right_terms.chunks(BLOCK_SIZE) {
                    let left_end = (left_block + BLOCK_SIZE).min(left_coefficients.len());
                    for left_term in left_block..left_end {
                        let left = unsafe { left_coefficients.get_unchecked(left_term) };
                        let left_offset =
                            unsafe { *left_indices.get_unchecked(left_term) as usize } * 3;
                        for &(right, right_offset) in right_block {
                            let product = left.0 as u128 * right as u128;
                            let accumulator = unsafe {
                                accumulators.as_mut_ptr().add(left_offset + right_offset)
                            };
                            unsafe { add_u128_to_three_limbs(accumulator, product) };
                        }
                    }
                }
            }
        } else {
            for (left, &left_index) in left_coefficients.iter().zip(left_indices) {
                let left_offset = left_index as usize * 3;
                for &(right, right_offset) in &right_terms {
                    let product = left.0 as u128 * right as u128;
                    let accumulator =
                        unsafe { accumulators.as_mut_ptr().add(left_offset + right_offset) };
                    unsafe { add_u128_to_three_limbs(accumulator, product) };
                }
            }
        }

        let mut result = Vec::new();
        for (position, accumulator) in accumulators.chunks_exact(3).enumerate() {
            let [low, high, top] = *accumulator else {
                unreachable!()
            };
            if low == 0 && high == 0 && top == 0 {
                continue;
            }

            let coefficient = montgomery_reduce_three_limbs(self.field, low, high, top);
            if coefficient != 0 {
                result.push((position as u32, FiniteFieldElement(coefficient)));
            }
        }
        result
    }
}

impl PolynomialKernels<FiniteFieldElement<u32>> for Zp {
    #[inline]
    fn try_dense_mul(
        &self,
        request: DensePolynomialMulRequest<'_, FiniteFieldElement<u32>>,
    ) -> Option<Vec<(u32, FiniteFieldElement<u32>)>> {
        DenseZpMul::new(self, request)?.run()
    }
}

impl PolynomialKernels<FiniteFieldElement<u64>> for Zp64 {
    #[inline]
    fn try_dense_mul(
        &self,
        request: DensePolynomialMulRequest<'_, FiniteFieldElement<u64>>,
    ) -> Option<Vec<(u32, FiniteFieldElement<u64>)>> {
        DenseZp64Mul::new(self, request)?.run()
    }
}

impl PolynomialKernels<FiniteFieldElement<Integer>> for FiniteField<Integer> {
    #[inline]
    fn try_dense_mul(
        &self,
        request: DensePolynomialMulRequest<'_, FiniteFieldElement<Integer>>,
    ) -> Option<Vec<(u32, FiniteFieldElement<Integer>)>> {
        Some(DenseIntegerMontgomeryMul::new(self, request)?.run())
    }
}
