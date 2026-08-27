//! Representation-specific bulk kernels for integer polynomial arithmetic.
//!
//! The polynomial layer supplies additive indices or compact total-degree layout tables. This
//! module selects an accumulator representation from the coefficient sizes and density, performs
//! the complete inner loop in that representation, and converts the nonzero result coefficients
//! back to [`Integer`]. Each polynomial operation is represented by a short-lived context that
//! validates its request once and dispatches to representation-specific strategy methods. A
//! strategy returning `None` means that its representation, size, or memory assumptions do not
//! hold and that the context should try another strategy or the generic polynomial implementation.

use super::{Integer, IntegerRing};
#[cfg(feature = "gmp")]
use super::{MultiPrecisionInteger, RawMultiPrecisionInteger};
#[cfg(feature = "gmp")]
use crate::domains::polynomial_layouts::try_simplex_kronecker_layout;
use crate::kernels::{
    DensePolynomialExactDivisionRequest, DensePolynomialMulRequest, PolynomialKernels,
    TotalDegreePolynomialMulRequest,
};
#[cfg(feature = "gmp")]
use gmp_mpfr_sys::gmp;
#[cfg(feature = "gmp")]
use rug::integer::Order as RugIntegerOrder;
#[cfg(feature = "gmp")]
use smallvec::SmallVec;

/// One dense-indexed integer multiplication operation.
///
/// Construction checks the invariants shared by every multiplication strategy. The context then
/// owns the strategy order and gives each strategy direct access to the already-validated slices.
#[derive(Clone, Copy)]
pub(super) struct DenseIntegerMul<'a> {
    output_len: usize,
    left_coefficients: &'a [Integer],
    left_indices: &'a [u32],
    right_coefficients: &'a [Integer],
    right_indices: &'a [u32],
}

impl<'a> DenseIntegerMul<'a> {
    /// Validate and retain a dense multiplication request.
    fn new(request: DensePolynomialMulRequest<'a, Integer>) -> Option<Self> {
        let context = Self {
            output_len: request.output_len,
            left_coefficients: request.left_coefficients,
            left_indices: request.left_indices,
            right_coefficients: request.right_coefficients,
            right_indices: request.right_indices,
        };
        if context.left_coefficients.len() != context.left_indices.len()
            || context.right_coefficients.len() != context.right_indices.len()
        {
            return None;
        }
        if let (Some(left_max), Some(right_max)) = (
            context.left_indices.iter().copied().max(),
            context.right_indices.iter().copied().max(),
        ) {
            if left_max as usize + right_max as usize >= context.output_len {
                return None;
            }
        }
        Some(context)
    }

    /// Run the first applicable dense multiplication strategy.
    fn run(self) -> Option<Vec<(u32, Integer)>> {
        if self.left_coefficients.is_empty() || self.right_coefficients.is_empty() {
            return Some(Vec::new());
        }

        if let Some(output) = self.try_i64() {
            return Some(output);
        }
        if let Some(output) = self.try_i64_i128() {
            return Some(output);
        }
        if let Some(output) = self.try_i128() {
            return Some(output);
        }

        #[cfg(feature = "gmp")]
        {
            if let Some(output) = self.try_kronecker() {
                return Some(output);
            }
            self.try_large_array()
        }
        #[cfg(feature = "no_gmp")]
        {
            None
        }
    }

    /// Multiply using a dense `i64` accumulator array.
    ///
    /// This strategy accepts only [`Integer::Single`] coefficients. A conservative coefficient
    /// bound must prove that every accumulated result fits in `i64`; otherwise it returns `None`.
    /// On success it performs a cache-blocked convolution and returns the nonzero coefficients in
    /// increasing dense-index order.
    fn try_i64(&self) -> Option<Vec<(u32, Integer)>> {
        let Self {
            output_len,
            left_coefficients,
            left_indices,
            right_coefficients,
            right_indices,
        } = *self;
        let maximum_single = |coefficients: &[Integer]| {
            coefficients
                .iter()
                .try_fold(0u64, |maximum, coefficient| match coefficient {
                    Integer::Single(value) => Some(maximum.max(value.unsigned_abs())),
                    Integer::Double(_) | Integer::Large(_) => None,
                })
        };
        let max_left = maximum_single(left_coefficients)?;
        let max_right = maximum_single(right_coefficients)?;
        let coefficient_bound = u128::from(max_left)
            .checked_mul(u128::from(max_right))?
            .checked_mul(left_coefficients.len().min(right_coefficients.len()) as u128)?;
        if coefficient_bound > i64::MAX as u128 {
            return None;
        }

        let left = left_coefficients
            .iter()
            .map(|coefficient| match coefficient {
                Integer::Single(value) => *value,
                Integer::Double(_) | Integer::Large(_) => unreachable!(),
            })
            .collect::<Vec<_>>();
        let right = right_coefficients
            .iter()
            .map(|coefficient| match coefficient {
                Integer::Single(value) => *value,
                Integer::Double(_) | Integer::Large(_) => unreachable!(),
            })
            .collect::<Vec<_>>();

        const BLOCK_SIZE: usize = 32;
        let mut output = vec![0i64; output_len];
        for left_block in (0..left.len()).step_by(BLOCK_SIZE) {
            for right_block in (0..right.len()).step_by(BLOCK_SIZE) {
                for i in left_block..(left_block + BLOCK_SIZE).min(left.len()) {
                    // SAFETY: the capability dispatch checks the coefficient/index lengths and largest
                    // possible output index before dispatching to this kernel.
                    let left_index = unsafe { *left_indices.get_unchecked(i) as usize };
                    let left_coefficient = unsafe { *left.get_unchecked(i) };
                    for j in right_block..(right_block + BLOCK_SIZE).min(right.len()) {
                        let index =
                            left_index + unsafe { *right_indices.get_unchecked(j) as usize };
                        unsafe {
                            *output.get_unchecked_mut(index) +=
                                left_coefficient * *right.get_unchecked(j);
                        }
                    }
                }
            }
        }

        Some(
            output
                .into_iter()
                .enumerate()
                .filter_map(|(index, coefficient)| {
                    (coefficient != 0).then(|| (index as u32, Integer::Single(coefficient)))
                })
                .collect(),
        )
    }

    /// Multiply `i64` coefficients using a dense `i128` accumulator array.
    ///
    /// This is the intermediate strategy for [`Integer::Single`] inputs whose convolution may
    /// overflow `i64` but whose coefficients are conservatively proven to fit in `i128`. It
    /// returns `None` when an input has a wider representation or the bound exceeds `i128`.
    fn try_i64_i128(&self) -> Option<Vec<(u32, Integer)>> {
        let Self {
            output_len,
            left_coefficients,
            left_indices,
            right_coefficients,
            right_indices,
        } = *self;
        let maximum_single = |coefficients: &[Integer]| {
            coefficients
                .iter()
                .try_fold(0u64, |maximum, coefficient| match coefficient {
                    Integer::Single(value) => Some(maximum.max(value.unsigned_abs())),
                    Integer::Double(_) | Integer::Large(_) => None,
                })
        };
        let max_left = maximum_single(left_coefficients)?;
        let max_right = maximum_single(right_coefficients)?;
        let coefficient_bound = u128::from(max_left)
            .checked_mul(u128::from(max_right))?
            .checked_mul(left_coefficients.len().min(right_coefficients.len()) as u128)?;
        if coefficient_bound > i128::MAX as u128 {
            return None;
        }

        let left = left_coefficients
            .iter()
            .map(|coefficient| match coefficient {
                Integer::Single(value) => *value,
                Integer::Double(_) | Integer::Large(_) => unreachable!(),
            })
            .collect::<Vec<_>>();
        let right = right_coefficients
            .iter()
            .map(|coefficient| match coefficient {
                Integer::Single(value) => *value,
                Integer::Double(_) | Integer::Large(_) => unreachable!(),
            })
            .collect::<Vec<_>>();

        const BLOCK_SIZE: usize = 128;
        let mut output = vec![0i128; output_len];
        for left_block in (0..left.len()).step_by(BLOCK_SIZE) {
            for right_block in (0..right.len()).step_by(BLOCK_SIZE) {
                for i in left_block..(left_block + BLOCK_SIZE).min(left.len()) {
                    // SAFETY: the capability dispatch checks the coefficient/index lengths and largest
                    // possible output index before dispatching to this kernel.
                    let left_index = unsafe { *left_indices.get_unchecked(i) as usize };
                    let left_coefficient = unsafe { *left.get_unchecked(i) };
                    for j in right_block..(right_block + BLOCK_SIZE).min(right.len()) {
                        let index =
                            left_index + unsafe { *right_indices.get_unchecked(j) as usize };
                        unsafe {
                            *output.get_unchecked_mut(index) +=
                                i128::from(left_coefficient) * i128::from(*right.get_unchecked(j));
                        }
                    }
                }
            }
        }

        Some(
            output
                .into_iter()
                .enumerate()
                .filter_map(|(index, coefficient)| {
                    (coefficient != 0).then(|| (index as u32, Integer::from_double(coefficient)))
                })
                .collect(),
        )
    }

    /// Multiply single- or double-width integers using dense `i128` accumulators.
    ///
    /// The inputs may contain [`Integer::Single`] and [`Integer::Double`] values, but not
    /// GMP-backed values. The strategy is used only when a conservative bound proves that the
    /// complete sum for every output coefficient fits in `i128`.
    fn try_i128(&self) -> Option<Vec<(u32, Integer)>> {
        let Self {
            output_len,
            left_coefficients,
            left_indices,
            right_coefficients,
            right_indices,
        } = *self;
        let fixed_value = |coefficient: &Integer| match coefficient {
            Integer::Single(value) => Some(i128::from(*value)),
            Integer::Double(value) => Some(value.get()),
            Integer::Large(_) => None,
        };
        let maximum_fixed = |coefficients: &[Integer]| {
            coefficients.iter().try_fold(0u128, |maximum, coefficient| {
                Some(maximum.max(fixed_value(coefficient)?.unsigned_abs()))
            })
        };
        let max_left = maximum_fixed(left_coefficients)?;
        let max_right = maximum_fixed(right_coefficients)?;
        let coefficient_bound = max_left
            .checked_mul(max_right)?
            .checked_mul(left_coefficients.len().min(right_coefficients.len()) as u128)?;
        if coefficient_bound > i128::MAX as u128 {
            return None;
        }

        let left = left_coefficients
            .iter()
            .map(|coefficient| fixed_value(coefficient).unwrap())
            .collect::<Vec<_>>();
        let right = right_coefficients
            .iter()
            .map(|coefficient| fixed_value(coefficient).unwrap())
            .collect::<Vec<_>>();

        const BLOCK_SIZE: usize = 32;
        let mut output = vec![0i128; output_len];
        for left_block in (0..left.len()).step_by(BLOCK_SIZE) {
            for right_block in (0..right.len()).step_by(BLOCK_SIZE) {
                for i in left_block..(left_block + BLOCK_SIZE).min(left.len()) {
                    // SAFETY: the capability dispatch checks the coefficient/index lengths and largest
                    // possible output index before dispatching to this kernel.
                    let left_index = unsafe { *left_indices.get_unchecked(i) as usize };
                    let left_coefficient = unsafe { *left.get_unchecked(i) };
                    for j in right_block..(right_block + BLOCK_SIZE).min(right.len()) {
                        let index =
                            left_index + unsafe { *right_indices.get_unchecked(j) as usize };
                        unsafe {
                            *output.get_unchecked_mut(index) +=
                                left_coefficient * *right.get_unchecked(j);
                        }
                    }
                }
            }
        }

        Some(
            output
                .into_iter()
                .enumerate()
                .filter_map(|(index, coefficient)| {
                    (coefficient != 0).then(|| (index as u32, Integer::from_double(coefficient)))
                })
                .collect(),
        )
    }

    #[cfg(feature = "gmp")]
    /// Multiply a dense-indexed convolution by Kronecker substitution.
    ///
    /// Each coefficient is encoded as a sufficiently wide signed radix digit, the two complete
    /// polynomials are packed into GMP integers, and one GMP multiplication computes the
    /// convolution. A compact additive embedding is used when the supplied indices describe a
    /// sufficiently large total-degree simplex with large holes in its mixed-radix box. The
    /// result is unpacked exactly, including signed carries.
    ///
    /// Returns `None` for small or insufficiently dense products, when a safe digit width cannot
    /// be represented, or when packing would exceed the memory limit.
    fn try_kronecker(&self) -> Option<Vec<(u32, Integer)>> {
        let Self {
            output_len,
            left_coefficients,
            left_indices,
            right_coefficients,
            right_indices,
        } = *self;
        // Constructing and sorting the compact layout only pays off once the packed product is
        // large. Below this point GMP's multiplication is already cheap enough that layout setup
        // dominates the saved limbs.
        const MIN_SIMPLEX_KRONECKER_OUTPUT_LEN: usize = 1 << 18;
        let simplex_layout = (cfg!(test) || output_len >= MIN_SIMPLEX_KRONECKER_OUTPUT_LEN)
            .then(|| try_simplex_kronecker_layout(output_len, left_indices, right_indices))
            .flatten();
        let (packed_output_len, packed_left_indices, packed_right_indices) =
            if let Some(layout) = simplex_layout.as_ref() {
                (
                    layout.output_len,
                    layout.left_indices.as_slice(),
                    layout.right_indices.as_slice(),
                )
            } else {
                (output_len, left_indices, right_indices)
            };

        let product_count = left_coefficients
            .len()
            .checked_mul(right_coefficients.len())?;
        let high_collision_density = packed_output_len.saturating_mul(128) < product_count;

        const MIN_CONTIGUOUS_KRONECKER_TERMS: usize = 32;
        let consecutive_support = |indices: &[u32]| {
            indices
                .windows(2)
                .all(|pair| pair[0].checked_add(1) == Some(pair[1]))
        };
        let active_output_span = left_coefficients
            .len()
            .checked_add(right_coefficients.len())?
            .checked_sub(1)?;
        let large_contiguous_support = left_coefficients.len().min(right_coefficients.len())
            >= MIN_CONTIGUOUS_KRONECKER_TERMS
            && consecutive_support(packed_left_indices)
            && consecutive_support(packed_right_indices)
            && packed_output_len <= active_output_span.saturating_mul(2);

        if product_count < 64 || !(high_collision_density || large_contiguous_support) {
            return None;
        }

        fn absolute_statistics(
            coefficients: &[Integer],
        ) -> (MultiPrecisionInteger, MultiPrecisionInteger) {
            let mut fixed_sum = 0u128;
            let mut fixed_maximum = 0u128;
            let fixed_statistics = coefficients.iter().all(|coefficient| {
                let magnitude = match coefficient {
                    Integer::Single(value) => u128::from(value.unsigned_abs()),
                    Integer::Double(value) => value.get().unsigned_abs(),
                    Integer::Large(_) => return false,
                };
                let Some(sum) = fixed_sum.checked_add(magnitude) else {
                    return false;
                };
                fixed_sum = sum;
                fixed_maximum = fixed_maximum.max(magnitude);
                true
            });
            if fixed_statistics {
                return (
                    MultiPrecisionInteger::from(fixed_sum),
                    MultiPrecisionInteger::from(fixed_maximum),
                );
            }

            let mut sum = MultiPrecisionInteger::default();
            let mut maximum = Integer::zero();
            for coefficient in coefficients {
                match coefficient {
                    Integer::Single(value) => sum += value.unsigned_abs(),
                    Integer::Double(value) => sum += value.get().unsigned_abs(),
                    Integer::Large(value) if value.is_negative() => sum -= value,
                    Integer::Large(value) => sum += value,
                }
                if coefficient.abs_cmp(&maximum).is_gt() {
                    maximum = coefficient.abs();
                }
            }

            (sum, maximum.to_multi_prec())
        }

        let (left_sum, left_maximum) = absolute_statistics(left_coefficients);
        let (right_sum, right_maximum) = absolute_statistics(right_coefficients);
        let collision_count = left_coefficients.len().min(right_coefficients.len());
        let signed_coefficients = left_coefficients
            .iter()
            .chain(right_coefficients)
            .any(Integer::is_negative);

        // Bound every output coefficient in three ways and use the tightest exact bound:
        //   min(n) max(a) max(b), ||a||_1 max(b), and ||b||_1 max(a).
        // This is commonly several bits tighter than rounding every factor up to a power of two.
        let mut coefficient_bound = &left_maximum * &right_maximum;
        coefficient_bound *= u64::try_from(collision_count).ok()?;
        let left_l1_bound = &left_sum * &right_maximum;
        if left_l1_bound < coefficient_bound {
            coefficient_bound = left_l1_bound;
        }
        let right_l1_bound = &right_sum * &left_maximum;
        if right_l1_bound < coefficient_bound {
            coefficient_bound = right_l1_bound;
        }
        let digit_bits = coefficient_bound
            .significant_bits()
            .checked_add(u64::from(signed_coefficients))?;
        let digit_bits_u32 = u32::try_from(digit_bits).ok()?;
        let digit_bits_usize = usize::try_from(digit_bits).ok()?;

        debug_assert!(
            left_indices
                .windows(2)
                .all(|indices| indices[0] < indices[1])
        );
        debug_assert!(
            right_indices
                .windows(2)
                .all(|indices| indices[0] < indices[1])
        );

        const MAX_PACKED_BITS: usize = 1 << 29;
        let left_packed_bits = (packed_left_indices.iter().copied().max()? as usize + 1)
            .checked_mul(digit_bits_usize)?;
        let right_packed_bits = (packed_right_indices.iter().copied().max()? as usize + 1)
            .checked_mul(digit_bits_usize)?;
        if left_packed_bits > MAX_PACKED_BITS || right_packed_bits > MAX_PACKED_BITS {
            return None;
        }

        fn pack(
            coefficients: &[Integer],
            indices: &[u32],
            digit_bits: usize,
        ) -> Option<MultiPrecisionInteger> {
            debug_assert_eq!(coefficients.len(), indices.len());

            #[inline(always)]
            fn low_mask(bits: usize) -> u64 {
                if bits == 64 {
                    u64::MAX
                } else {
                    (1u64 << bits) - 1
                }
            }

            #[inline(always)]
            fn fill_ones(limbs: &mut [u64], start: usize, end: usize) {
                if start == end {
                    return;
                }

                let first = start / 64;
                let last = (end - 1) / 64;
                let first_bit = start % 64;
                let last_bits = (end - 1) % 64 + 1;
                if first == last {
                    limbs[first] |= low_mask(last_bits) & !low_mask(first_bit);
                    return;
                }

                limbs[first] |= u64::MAX << first_bit;
                limbs[first + 1..last].fill(u64::MAX);
                limbs[last] |= low_mask(last_bits);
            }

            #[inline(always)]
            fn write_digit(limbs: &mut [u64], bit_index: usize, digit: &[u64]) {
                let limb_index = bit_index / 64;
                let shift = bit_index % 64;
                if shift == 0 {
                    limbs[limb_index..limb_index + digit.len()].copy_from_slice(digit);
                    return;
                }

                for (offset, &value) in digit.iter().enumerate() {
                    limbs[limb_index + offset] |= value << shift;
                    if limb_index + offset + 1 < limbs.len() {
                        limbs[limb_index + offset + 1] |= value >> (64 - shift);
                    }
                }
            }

            /// Encode a tagged fixed-width coefficient after applying the polynomial sign and the
            /// borrow propagated by the preceding signed radix digit.
            #[inline(always)]
            fn encode_primitive_digit(
                digit: &mut [u64],
                coefficient: i128,
                negate: bool,
                borrow: bool,
            ) -> bool {
                digit.fill(0);
                debug_assert!(!digit.is_empty());

                let mut magnitude = coefficient.unsigned_abs();
                let mut negative = magnitude != 0 && ((coefficient < 0) != negate);
                if borrow {
                    if negative {
                        magnitude += 1;
                    } else if magnitude == 0 {
                        magnitude = 1;
                        negative = true;
                    } else {
                        magnitude -= 1;
                    }
                }

                debug_assert!(digit.len() > 1 || magnitude <= u128::from(u64::MAX));
                digit[0] = magnitude as u64;
                if digit.len() > 1 {
                    digit[1] = (magnitude >> 64) as u64;
                }
                negative
            }

            let mut reordered = None;
            if !indices.windows(2).all(|indices| indices[0] < indices[1]) {
                let mut order = (0..indices.len()).collect::<Vec<_>>();
                order.sort_unstable_by_key(|&index| indices[index]);
                reordered = Some(order);
            }
            let term_index = |position: usize| {
                reordered
                    .as_ref()
                    .map_or(position, |order| unsafe { *order.get_unchecked(position) })
            };

            let leading_negative = coefficients[term_index(coefficients.len() - 1)].is_negative();
            let digit_count = indices[term_index(indices.len() - 1)] as usize + 1;
            let packed_bits = digit_count.checked_mul(digit_bits)?;
            let mut limbs = vec![0u64; packed_bits.checked_add(63)? / 64];
            let mut next_index = 0usize;
            let mut borrow = false;
            let mut digit = SmallVec::<[u64; 4]>::new();
            digit.resize(digit_bits.div_ceil(64), 0);

            for position in 0..coefficients.len() {
                let term = term_index(position);
                let coefficient = unsafe { coefficients.get_unchecked(term) };
                let index = unsafe { *indices.get_unchecked(term) } as usize;
                debug_assert!(index >= next_index);
                if borrow {
                    fill_ones(&mut limbs, next_index * digit_bits, index * digit_bits);
                }

                let borrow_in = borrow;
                borrow = match coefficient {
                    Integer::Single(value) => encode_primitive_digit(
                        &mut digit,
                        i128::from(*value),
                        leading_negative,
                        borrow_in,
                    ),
                    Integer::Double(value) => {
                        encode_primitive_digit(&mut digit, value.get(), leading_negative, borrow_in)
                    }
                    Integer::Large(value) => {
                        let mut value = value.clone();
                        if leading_negative {
                            value = -value;
                        }
                        if borrow_in {
                            value -= 1i64;
                        }

                        let negative = value.is_negative();
                        if negative {
                            value = -value;
                        }
                        digit.fill(0);
                        let value_limbs = value.as_raw().as_limbs();
                        debug_assert!(value_limbs.len() <= digit.len());
                        digit[..value_limbs.len()].copy_from_slice(value_limbs);
                        negative
                    }
                };

                if borrow {
                    let mut carry = true;
                    for limb in &mut digit {
                        let (value, overflow) = (!*limb).overflowing_add(u64::from(carry));
                        *limb = value;
                        carry = overflow;
                    }
                    debug_assert!(!carry);
                }
                if digit_bits % 64 != 0 {
                    *digit.last_mut().unwrap() &= low_mask(digit_bits % 64);
                }
                write_digit(&mut limbs, index * digit_bits, &digit);
                next_index = index + 1;
            }
            debug_assert!(!borrow);

            let mut packed = MultiPrecisionInteger::from_raw(
                RawMultiPrecisionInteger::from_digits(&limbs, RugIntegerOrder::Lsf),
            );
            if leading_negative {
                packed = -packed;
            }
            Some(packed)
        }

        /// Decode one signed radix digit with native arithmetic when its final magnitude fits in
        /// `i128`. The returned boolean is the signed carry into the next digit.
        #[inline(always)]
        fn try_decode_i128_digit(
            digit_limbs: &[u64],
            digit_bits: usize,
            carry_in: bool,
            signed_coefficients: bool,
            product_negative: bool,
        ) -> Option<(i128, bool)> {
            if digit_bits == 0 || digit_limbs.len() > 4 {
                return None;
            }

            let mut words = [0u64; 4];
            words[..digit_limbs.len()].copy_from_slice(digit_limbs);
            let last = digit_limbs.len().checked_sub(1)?;
            let trailing_bits = digit_bits % 64;
            let mut overflow_radix = false;
            if carry_in {
                let mut carry = true;
                for word in &mut words[..digit_limbs.len()] {
                    let (value, overflow) = word.overflowing_add(u64::from(carry));
                    *word = value;
                    carry = overflow;
                    if !carry {
                        break;
                    }
                }

                if trailing_bits == 0 {
                    overflow_radix = carry;
                } else {
                    overflow_radix = words[last] >> trailing_bits != 0;
                    words[last] &= (1u64 << trailing_bits) - 1;
                }
            }
            if overflow_radix && !signed_coefficients {
                return None;
            }

            let sign_bit_index = digit_bits - 1;
            let negative = signed_coefficients
                && words[sign_bit_index / 64] & (1u64 << (sign_bit_index % 64)) != 0;
            let carry_out = signed_coefficients && (overflow_radix || negative);

            if negative {
                let mut carry = true;
                for word in &mut words[..digit_limbs.len()] {
                    let (value, overflow) = (!*word).overflowing_add(u64::from(carry));
                    *word = value;
                    carry = overflow;
                }
                debug_assert!(!carry);
                if trailing_bits != 0 {
                    words[last] &= (1u64 << trailing_bits) - 1;
                }
            }

            if digit_limbs.len() > 2 && words[2..digit_limbs.len()].iter().any(|&word| word != 0) {
                return None;
            }
            let magnitude = u128::from(words[0]) | (u128::from(words[1]) << 64);
            let final_negative = magnitude != 0 && (negative != product_negative);
            let value = if final_negative {
                const I128_MIN_MAGNITUDE: u128 = 1u128 << 127;
                if magnitude > I128_MIN_MAGNITUDE {
                    return None;
                }
                if magnitude == I128_MIN_MAGNITUDE {
                    i128::MIN
                } else {
                    -(magnitude as i128)
                }
            } else {
                if magnitude > i128::MAX as u128 {
                    return None;
                }
                magnitude as i128
            };
            Some((value, carry_out))
        }

        let left = pack(left_coefficients, packed_left_indices, digit_bits_usize)?;
        let right = pack(right_coefficients, packed_right_indices, digit_bits_usize)?;
        let product = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from(
            left.as_raw() * right.as_raw(),
        ));

        let product_negative = product.is_negative();
        let limbs = product.as_raw().as_limbs();
        let limbs_per_digit = digit_bits_usize.div_ceil(64);
        let mut carry = false;
        let mut radix = None;
        let mut output = Vec::with_capacity(packed_output_len);
        let mut digit_limbs = SmallVec::<[u64; 4]>::new();
        digit_limbs.resize(limbs_per_digit, 0);
        let output_index = |index: usize| {
            if let Some(layout) = simplex_layout.as_ref() {
                let decoded = *layout.decode_indices.get(index)?;
                debug_assert_ne!(decoded, u32::MAX);
                Some(decoded)
            } else {
                Some(index as u32)
            }
        };
        for index in 0..packed_output_len {
            let bit_index = index.checked_mul(digit_bits_usize)?;
            let limb_index = bit_index / 64;
            let shift = bit_index % 64;
            digit_limbs.fill(0);
            for (offset, digit_limb) in digit_limbs.iter_mut().enumerate() {
                *digit_limb = limbs.get(limb_index + offset).copied().unwrap_or(0) >> shift;
                if shift != 0 {
                    *digit_limb |=
                        limbs.get(limb_index + offset + 1).copied().unwrap_or(0) << (64 - shift);
                }
            }
            if digit_bits_usize % 64 != 0 {
                *digit_limbs.last_mut().unwrap() &= (1u64 << (digit_bits_usize % 64)) - 1;
            }

            if let Some((value, carry_out)) = try_decode_i128_digit(
                &digit_limbs,
                digit_bits_usize,
                carry,
                signed_coefficients,
                product_negative,
            ) {
                carry = carry_out;
                if value != 0 {
                    output.push((output_index(index)?, Integer::from_double(value)));
                }
                continue;
            }

            let raw_digit =
                RawMultiPrecisionInteger::from_digits(&digit_limbs, RugIntegerOrder::Lsf);
            let mut digit = MultiPrecisionInteger::from_raw(raw_digit);
            if carry {
                digit += 1i64;
            }

            carry = signed_coefficients
                && (digit.significant_bits() > digit_bits
                    || digit.as_raw().get_bit(digit_bits_u32 - 1));
            if carry {
                let radix = radix
                    .get_or_insert_with(|| MultiPrecisionInteger::from(1u32) << digit_bits_usize);
                digit -= &*radix;
            }
            if product_negative {
                digit = -digit;
            }
            if !digit.is_zero() {
                output.push((output_index(index)?, Integer::from(digit)));
            }
        }
        debug_assert!(!carry);

        if simplex_layout.is_some() {
            output.sort_unstable_by_key(|term| term.0);
        }

        Some(output)
    }

    #[cfg(feature = "gmp")]
    /// Multiply into a dense array of reusable GMP integer accumulators.
    ///
    /// This strategy requires at least one [`Integer::Large`] input coefficient. It dispatches
    /// each coefficient pair by its tagged representation and uses fused GMP add-products for
    /// large operands, avoiding a separately allocated temporary product. It returns `None` when
    /// the product is too small or the dense output array would be disproportionately large.
    fn try_large_array(&self) -> Option<Vec<(u32, Integer)>> {
        let Self {
            output_len,
            left_coefficients,
            left_indices,
            right_coefficients,
            right_indices,
        } = *self;
        if !left_coefficients
            .iter()
            .chain(right_coefficients)
            .any(|coefficient| matches!(coefficient, Integer::Large(_)))
        {
            return None;
        }

        let product_count = left_coefficients
            .len()
            .checked_mul(right_coefficients.len())?;
        if product_count < 64
            || output_len > 1 << 20
            || output_len > product_count.saturating_mul(10)
        {
            return None;
        }

        #[inline(always)]
        fn add_product(accumulator: &mut MultiPrecisionInteger, left: &Integer, right: &Integer) {
            match (left, right) {
                (Integer::Single(left), Integer::Single(right)) => {
                    *accumulator += i128::from(*left) * i128::from(*right);
                }
                (Integer::Single(left), Integer::Double(right))
                | (Integer::Double(right), Integer::Single(left)) => {
                    if let Some(product) = i128::from(*left).checked_mul(right.get()) {
                        *accumulator += product;
                    } else {
                        let right = MultiPrecisionInteger::from(right.get());
                        accumulator.add_i64_mul_assign(*left, &right);
                    }
                }
                (Integer::Double(left), Integer::Double(right)) => {
                    if let Some(product) = left.get().checked_mul(right.get()) {
                        *accumulator += product;
                    } else {
                        let right = MultiPrecisionInteger::from(right.get());
                        accumulator.add_i128_mul_assign(left.get(), &right);
                    }
                }
                (Integer::Single(left), Integer::Large(right))
                | (Integer::Large(right), Integer::Single(left)) => {
                    accumulator.add_i64_mul_assign(*left, right);
                }
                (Integer::Double(left), Integer::Large(right))
                | (Integer::Large(right), Integer::Double(left)) => {
                    accumulator.add_i128_mul_assign(left.get(), right);
                }
                (Integer::Large(left), Integer::Large(right)) => {
                    accumulator.add_mul_assign(left, right);
                }
            }
        }

        const BLOCK_SIZE: usize = 32;
        let mut output = (0..output_len)
            .map(|_| MultiPrecisionInteger::default())
            .collect::<Vec<_>>();
        for left_block in (0..left_coefficients.len()).step_by(BLOCK_SIZE) {
            for right_block in (0..right_coefficients.len()).step_by(BLOCK_SIZE) {
                for i in left_block..(left_block + BLOCK_SIZE).min(left_coefficients.len()) {
                    // SAFETY: the capability dispatch checks the coefficient/index lengths and largest
                    // possible output index before dispatching to this kernel.
                    let left_index = unsafe { *left_indices.get_unchecked(i) as usize };
                    let left_coefficient = unsafe { left_coefficients.get_unchecked(i) };
                    for j in right_block..(right_block + BLOCK_SIZE).min(right_coefficients.len()) {
                        let index =
                            left_index + unsafe { *right_indices.get_unchecked(j) as usize };
                        unsafe {
                            add_product(
                                output.get_unchecked_mut(index),
                                left_coefficient,
                                right_coefficients.get_unchecked(j),
                            );
                        }
                    }
                }
            }
        }

        Some(
            output
                .into_iter()
                .enumerate()
                .filter_map(|(index, coefficient)| {
                    (!coefficient.is_zero()).then(|| (index as u32, Integer::from(coefficient)))
                })
                .collect(),
        )
    }

    /// Run only the Kronecker strategy so its packing and decoding can be tested in isolation.
    #[cfg(all(feature = "gmp", test))]
    pub(super) fn try_kronecker_for_test(
        output_len: usize,
        left_coefficients: &'a [Integer],
        left_indices: &'a [u32],
        right_coefficients: &'a [Integer],
        right_indices: &'a [u32],
    ) -> Option<Vec<(u32, Integer)>> {
        Self::new(DensePolynomialMulRequest {
            output_len,
            left_coefficients,
            left_indices,
            right_coefficients,
            right_indices,
        })?
        .try_kronecker()
    }
}

/// One integer multiplication supported on a compact total-degree simplex.
struct TotalDegreeIntegerMul<'a> {
    request: TotalDegreePolynomialMulRequest<'a, Integer>,
}

impl<'a> TotalDegreeIntegerMul<'a> {
    /// Retain the layout and coefficient slices for one total-degree multiplication.
    fn new(request: TotalDegreePolynomialMulRequest<'a, Integer>) -> Self {
        Self { request }
    }

    /// Run the specialized total-degree strategy when GMP limb operations are available.
    fn run(self) -> Option<Vec<(u32, Integer)>> {
        #[cfg(feature = "gmp")]
        {
            self.try_limb()
        }
        #[cfg(feature = "no_gmp")]
        {
            let Self { request } = self;
            let _ = request;
            None
        }
    }

    #[cfg(feature = "gmp")]
    /// Multiply with fixed-width limb accumulators.
    ///
    /// Input coefficients are flattened once into sign-and-magnitude GMP limbs. The supplied
    /// prefix and suffix rank tables map every exponent sum directly to its compact output rank.
    /// Each output slot is a fixed-width two's-complement limb slice sized from a conservative
    /// coefficient bound; products are formed in one shared scratch buffer and added to or
    /// subtracted from the slot with GMP's low-level `mpn` operations.
    ///
    /// Returns `None` if the layout tables are inconsistent, coefficients exceed the supported
    /// input width, GMP does not use 64-bit limbs without nails, or the bounded workspace would be
    /// too large.
    fn try_limb(self) -> Option<Vec<(u32, Integer)>> {
        const MAX_INPUT_LIMBS: usize = 32;
        const MAX_OUTPUT_LIMBS: usize = 1 << 26;
        const BLOCK_SIZE: usize = 32;

        if gmp::NUMB_BITS != 64 || gmp::NAIL_BITS != 0 {
            return None;
        }

        let TotalDegreePolynomialMulRequest {
            output_len,
            left_coefficients,
            left_codes,
            right_coefficients,
            right_codes,
            prefix_rank,
            prefix_remaining,
            suffix_rank,
            suffix_code_count,
        } = self.request;
        if left_codes.len() != left_coefficients.len()
            || right_codes.len() != right_coefficients.len()
            || left_coefficients.is_empty()
            || right_coefficients.is_empty()
            || prefix_rank.len() != prefix_remaining.len()
            || suffix_code_count == 0
            || !suffix_rank.len().is_multiple_of(suffix_code_count)
        {
            return None;
        }
        let suffix_rows = suffix_rank.len() / suffix_code_count;
        if prefix_remaining
            .iter()
            .copied()
            .filter(|remaining| *remaining != u8::MAX)
            .any(|remaining| remaining as usize >= suffix_rows)
        {
            return None;
        }

        #[derive(Clone, Copy)]
        struct LimbRange {
            offset: u32,
            length: u8,
            negative: bool,
        }

        /// Copy tagged integers into one contiguous limb buffer and describe each value by a range.
        fn flatten_coefficients(
            coefficients: &[Integer],
        ) -> Option<(Vec<gmp::limb_t>, Vec<LimbRange>, usize, u64)> {
            let mut limbs = Vec::new();
            let mut ranges = Vec::with_capacity(coefficients.len());
            let mut maximum_length = 0usize;
            let mut maximum_bits = 0u64;
            for coefficient in coefficients {
                let offset = u32::try_from(limbs.len()).ok()?;
                let negative = coefficient.is_negative();
                match coefficient {
                    Integer::Single(value) => {
                        let absolute = value.unsigned_abs();
                        if absolute != 0 {
                            limbs.push(absolute);
                        }
                    }
                    Integer::Double(value) => {
                        let absolute = value.get().unsigned_abs();
                        if absolute != 0 {
                            limbs.push(absolute as u64);
                            let high = (absolute >> 64) as u64;
                            if high != 0 {
                                limbs.push(high);
                            }
                        }
                    }
                    Integer::Large(value) => limbs.extend_from_slice(value.as_raw().as_limbs()),
                }
                let length = limbs.len() - offset as usize;
                maximum_length = maximum_length.max(length);
                maximum_bits = maximum_bits.max(coefficient.significant_bits());
                ranges.push(LimbRange {
                    offset,
                    length: u8::try_from(length).ok()?,
                    negative,
                });
            }
            Some((limbs, ranges, maximum_length, maximum_bits))
        }

        let (left_limbs, left_ranges, maximum_left_limbs, maximum_left_bits) =
            flatten_coefficients(left_coefficients)?;
        let (right_limbs, right_ranges, maximum_right_limbs, maximum_right_bits) =
            flatten_coefficients(right_coefficients)?;
        if maximum_left_limbs > MAX_INPUT_LIMBS || maximum_right_limbs > MAX_INPUT_LIMBS {
            return None;
        }

        let products_per_coefficient = left_coefficients.len().min(right_coefficients.len());
        let accumulation_bits = usize::BITS - products_per_coefficient.leading_zeros();
        let output_bits = maximum_left_bits
            .checked_add(maximum_right_bits)?
            .checked_add(u64::from(accumulation_bits))?
            .checked_add(1)?;
        let limbs_per_coefficient = usize::try_from(output_bits.div_ceil(64)).ok()?.max(1);
        let output_limb_count = output_len.checked_mul(limbs_per_coefficient)?;
        if output_limb_count > MAX_OUTPUT_LIMBS {
            return None;
        }

        let maximum_left_prefix = left_codes.iter().map(|code| code.0).max()?;
        let maximum_right_prefix = right_codes.iter().map(|code| code.0).max()?;
        let maximum_left_suffix = left_codes.iter().map(|code| code.1).max()?;
        let maximum_right_suffix = right_codes.iter().map(|code| code.1).max()?;
        if maximum_left_prefix.checked_add(maximum_right_prefix)? >= prefix_rank.len()
            || maximum_left_suffix.checked_add(maximum_right_suffix)? >= suffix_code_count
        {
            return None;
        }

        /// Add or subtract one unsigned limb product from a fixed-width two's-complement slot.
        ///
        /// `accumulator` must be wide enough for the exact accumulated coefficient, and `product`
        /// must have room for `left.len() + right.len()` limbs. Those properties are established by
        /// the bounds and workspace checks in the enclosing kernel.
        #[inline(always)]
        unsafe fn accumulate_product(
            accumulator: &mut [gmp::limb_t],
            left: &[gmp::limb_t],
            right: &[gmp::limb_t],
            negative: bool,
            product: &mut [gmp::limb_t],
        ) {
            if left.is_empty() || right.is_empty() {
                return;
            }
            let (long, short) = if left.len() >= right.len() {
                (left, right)
            } else {
                (right, left)
            };
            let product_length = long.len() + short.len();
            debug_assert!(product_length <= product.len());
            unsafe {
                gmp::mpn_mul(
                    product.as_mut_ptr(),
                    long.as_ptr(),
                    long.len() as gmp::size_t,
                    short.as_ptr(),
                    short.len() as gmp::size_t,
                );
            }
            let product_length = product_length - usize::from(product[product_length - 1] == 0);
            if product_length == 0 {
                return;
            }
            debug_assert!(product_length <= accumulator.len());
            let carry = unsafe {
                if negative {
                    gmp::mpn_sub_n(
                        accumulator.as_mut_ptr(),
                        accumulator.as_ptr(),
                        product.as_ptr(),
                        product_length as gmp::size_t,
                    )
                } else {
                    gmp::mpn_add_n(
                        accumulator.as_mut_ptr(),
                        accumulator.as_ptr(),
                        product.as_ptr(),
                        product_length as gmp::size_t,
                    )
                }
            };
            if carry != 0 && product_length < accumulator.len() {
                let remaining = accumulator.len() - product_length;
                unsafe {
                    if negative {
                        gmp::mpn_sub_1(
                            accumulator.as_mut_ptr().add(product_length),
                            accumulator.as_ptr().add(product_length),
                            remaining as gmp::size_t,
                            carry,
                        );
                    } else {
                        gmp::mpn_add_1(
                            accumulator.as_mut_ptr().add(product_length),
                            accumulator.as_ptr().add(product_length),
                            remaining as gmp::size_t,
                            carry,
                        );
                    }
                }
            }
        }

        let mut coefficients = vec![0 as gmp::limb_t; output_limb_count];
        let mut product = [0 as gmp::limb_t; MAX_INPUT_LIMBS * 2];
        for left_block in (0..left_coefficients.len()).step_by(BLOCK_SIZE) {
            for right_block in (0..right_coefficients.len()).step_by(BLOCK_SIZE) {
                for left_index in left_block..(left_block + BLOCK_SIZE).min(left_coefficients.len())
                {
                    let left_range = unsafe { *left_ranges.get_unchecked(left_index) };
                    let left = unsafe {
                        left_limbs.get_unchecked(
                            left_range.offset as usize
                                ..left_range.offset as usize + left_range.length as usize,
                        )
                    };
                    let (left_prefix, left_suffix) =
                        unsafe { *left_codes.get_unchecked(left_index) };
                    for right_index in
                        right_block..(right_block + BLOCK_SIZE).min(right_coefficients.len())
                    {
                        let right_range = unsafe { *right_ranges.get_unchecked(right_index) };
                        let right = unsafe {
                            right_limbs.get_unchecked(
                                right_range.offset as usize
                                    ..right_range.offset as usize + right_range.length as usize,
                            )
                        };
                        let (right_prefix, right_suffix) =
                            unsafe { *right_codes.get_unchecked(right_index) };
                        let prefix = left_prefix + right_prefix;
                        let suffix = left_suffix + right_suffix;
                        let remaining_degree = unsafe { *prefix_remaining.get_unchecked(prefix) };
                        if remaining_degree == u8::MAX {
                            return None;
                        }
                        let suffix_rank = unsafe {
                            *suffix_rank.get_unchecked(
                                remaining_degree as usize * suffix_code_count + suffix,
                            )
                        };
                        if suffix_rank == u32::MAX {
                            return None;
                        }
                        let prefix_rank = unsafe { *prefix_rank.get_unchecked(prefix) };
                        if prefix_rank == u32::MAX {
                            return None;
                        }
                        let rank = prefix_rank as usize + suffix_rank as usize;
                        if rank >= output_len {
                            return None;
                        }
                        let accumulator = unsafe {
                            coefficients.get_unchecked_mut(
                                rank * limbs_per_coefficient..(rank + 1) * limbs_per_coefficient,
                            )
                        };
                        unsafe {
                            accumulate_product(
                                accumulator,
                                left,
                                right,
                                left_range.negative ^ right_range.negative,
                                &mut product,
                            );
                        }
                    }
                }
            }
        }

        let mut output = Vec::with_capacity(output_len);
        for (index, coefficient) in coefficients.chunks_exact(limbs_per_coefficient).enumerate() {
            let negative = coefficient.last().copied().unwrap() >> 63 != 0;
            let mut magnitude = SmallVec::<[gmp::limb_t; 128]>::from_slice(coefficient);
            if negative {
                let mut carry = true;
                for limb in &mut magnitude {
                    let (value, overflow) = (!*limb).overflowing_add(gmp::limb_t::from(carry));
                    *limb = value;
                    carry = overflow;
                }
                debug_assert!(!carry);
            }
            while magnitude.last() == Some(&0) {
                magnitude.pop();
            }
            if magnitude.is_empty() {
                continue;
            }
            let mut value = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from_digits(
                &magnitude,
                RugIntegerOrder::Lsf,
            ));
            if negative {
                value = -value;
            }
            output.push((index as u32, Integer::from(value)));
        }
        Some(output)
    }
}

/// One exact dense-indexed integer polynomial division operation.
struct DenseIntegerExactDivision<'a> {
    total: usize,
    dividend_coefficients: &'a mut [Integer],
    dividend_indices: &'a [u32],
    divisor_coefficients: &'a [Integer],
    divisor_indices: &'a [u32],
}

impl<'a> DenseIntegerExactDivision<'a> {
    /// Retain an exact-division request without consuming its dividend coefficients.
    fn new(request: DensePolynomialExactDivisionRequest<'a, Integer>) -> Self {
        Self {
            total: request.total,
            dividend_coefficients: request.dividend_coefficients,
            dividend_indices: request.dividend_indices,
            divisor_coefficients: request.divisor_coefficients,
            divisor_indices: request.divisor_indices,
        }
    }

    /// Run the dense GMP-workspace strategy when it is applicable.
    fn run(self) -> Option<Vec<(u32, Integer)>> {
        #[cfg(feature = "gmp")]
        {
            self.try_large_array()
        }
        #[cfg(feature = "no_gmp")]
        {
            let Self {
                total,
                dividend_coefficients,
                dividend_indices,
                divisor_coefficients,
                divisor_indices,
            } = self;
            let _ = (
                total,
                dividend_coefficients,
                dividend_indices,
                divisor_coefficients,
                divisor_indices,
            );
            None
        }
    }

    #[cfg(feature = "gmp")]
    /// Exactly divide in a dense array of GMP accumulators.
    ///
    /// The dividend is expanded into dense multiprecision storage and processed from its leading
    /// coefficient downwards. Every quotient coefficient is divided exactly by the divisor's
    /// leading coefficient, then fused multiply-subtracts update the remaining workspace. Once
    /// this strategy commits to the operation it consumes the corresponding owned dividend
    /// coefficients so their GMP allocations can be reused.
    ///
    /// Returns `None` without changing the dividend when no large coefficient is present, the
    /// problem is too small for this representation, or the dense indices do not fit the supplied
    /// workspace.
    fn try_large_array(self) -> Option<Vec<(u32, Integer)>> {
        let Self {
            total,
            dividend_coefficients,
            dividend_indices,
            divisor_coefficients,
            divisor_indices,
        } = self;
        if dividend_coefficients.len() != dividend_indices.len()
            || divisor_coefficients.len() != divisor_indices.len()
            || divisor_coefficients.is_empty()
            || !dividend_coefficients
                .iter()
                .chain(divisor_coefficients)
                .any(|coefficient| matches!(coefficient, Integer::Large(_)))
            || dividend_coefficients
                .len()
                .saturating_mul(divisor_coefficients.len())
                < 64
        {
            return None;
        }

        let divisor_leading_index = *divisor_indices.last()? as usize;
        if divisor_leading_index >= total
            || dividend_indices
                .last()
                .is_some_and(|&index| index as usize >= total)
        {
            return None;
        }

        enum Divisors<'a> {
            Borrowed(Vec<&'a MultiPrecisionInteger>),
            Owned(Vec<MultiPrecisionInteger>),
        }

        let borrowed_divisors = divisor_coefficients
            .iter()
            .map(|coefficient| match coefficient {
                Integer::Large(value) => Some(value),
                Integer::Single(_) | Integer::Double(_) => None,
            })
            .collect::<Option<Vec<_>>>();
        let divisors = if let Some(divisors) = borrowed_divisors {
            Divisors::Borrowed(divisors)
        } else {
            Divisors::Owned(
                divisor_coefficients
                    .iter()
                    .cloned()
                    .map(Integer::to_multi_prec)
                    .collect(),
            )
        };
        let mut workspace = (0..total)
            .map(|_| MultiPrecisionInteger::default())
            .collect::<Vec<_>>();
        for (coefficient, &index) in dividend_coefficients.iter_mut().zip(dividend_indices) {
            workspace[index as usize] =
                std::mem::replace(coefficient, Integer::zero()).to_multi_prec();
        }

        let mut quotient = Vec::with_capacity(dividend_coefficients.len());
        for position in (0..total).rev() {
            let coefficient = std::mem::take(unsafe { workspace.get_unchecked_mut(position) });
            if coefficient.is_zero() {
                continue;
            }

            debug_assert!(position >= divisor_leading_index);
            let quotient_position = position - divisor_leading_index;
            let quotient_coefficient = match &divisors {
                Divisors::Borrowed(divisors) => {
                    coefficient.div_exact_owned(divisors.last().unwrap())
                }
                Divisors::Owned(divisors) => coefficient.div_exact_owned(divisors.last().unwrap()),
            };
            match &divisors {
                Divisors::Borrowed(divisors) => {
                    for (&divisor_position, &divisor_coefficient) in divisor_indices
                        [..divisor_indices.len() - 1]
                        .iter()
                        .zip(&divisors[..divisors.len() - 1])
                    {
                        let target = quotient_position + divisor_position as usize;
                        debug_assert!(target < position);
                        unsafe {
                            workspace
                                .get_unchecked_mut(target)
                                .sub_mul_assign(&quotient_coefficient, divisor_coefficient);
                        }
                    }
                }
                Divisors::Owned(divisors) => {
                    for (&divisor_position, divisor_coefficient) in divisor_indices
                        [..divisor_indices.len() - 1]
                        .iter()
                        .zip(&divisors[..divisors.len() - 1])
                    {
                        let target = quotient_position + divisor_position as usize;
                        debug_assert!(target < position);
                        unsafe {
                            workspace
                                .get_unchecked_mut(target)
                                .sub_mul_assign(&quotient_coefficient, divisor_coefficient);
                        }
                    }
                }
            }
            quotient.push((
                quotient_position as u32,
                Integer::from(quotient_coefficient),
            ));
        }

        quotient.reverse();
        Some(quotient)
    }
}

impl PolynomialKernels<Integer> for IntegerRing {
    #[inline]
    fn try_total_degree_mul(
        &self,
        request: TotalDegreePolynomialMulRequest<'_, Integer>,
    ) -> Option<Vec<(u32, Integer)>> {
        TotalDegreeIntegerMul::new(request).run()
    }

    #[inline]
    fn try_dense_mul(
        &self,
        request: DensePolynomialMulRequest<'_, Integer>,
    ) -> Option<Vec<(u32, Integer)>> {
        DenseIntegerMul::new(request)?.run()
    }

    #[inline]
    fn try_dense_exact_division(
        &self,
        request: DensePolynomialExactDivisionRequest<'_, Integer>,
    ) -> Option<Vec<(u32, Integer)>> {
        DenseIntegerExactDivision::new(request).run()
    }
}
