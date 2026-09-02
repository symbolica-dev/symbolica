//! Representation-specific bulk kernels for integer polynomial arithmetic.
//!
//! The polynomial layer supplies additive indices or compact total-degree layout tables. This
//! module selects an accumulator representation from the coefficient sizes and density, performs
//! the complete inner loop in that representation, and converts the nonzero result coefficients
//! back to [`Integer`]. Each polynomial operation is represented by a short-lived context that
//! validates its request once and dispatches to representation-specific strategy methods. A
//! strategy returning `None` means that its representation, size, or memory assumptions do not
//! hold and that the context should try another strategy or the generic polynomial implementation.

#[cfg(feature = "integer-gmp")]
use super::MultiPrecisionInteger;
#[cfg(all(test, feature = "integer-gmp"))]
use super::RawMultiPrecisionInteger;
use super::{Integer, IntegerRing};
#[cfg(feature = "integer-gmp")]
use crate::domains::polynomial_layouts::try_simplex_kronecker_layout;
use crate::kernels::{
    ChunkedDensePolynomialMulRequest, DensePolynomialExactDivisionRequest,
    DensePolynomialMulRequest, PolynomialKernels, TotalDegreePolynomialMulRequest,
};
#[cfg(feature = "integer-gmp")]
use gmp_mpfr_sys::gmp;
#[cfg(all(test, feature = "integer-gmp"))]
use rug::integer::Order as RugIntegerOrder;
use smallvec::SmallVec;

#[cfg(feature = "integer-gmp")]
/// Return the exact number of significant bits in a `u128` product without allocating an integer.
pub(super) fn u128_product_significant_bits(left: u128, right: u128) -> u64 {
    if let Some(product) = left.checked_mul(right) {
        return u64::from(u128::BITS - product.leading_zeros());
    }

    const LOW_MASK: u128 = u64::MAX as u128;
    let left_low = left & LOW_MASK;
    let left_high = left >> u64::BITS;
    let right_low = right & LOW_MASK;
    let right_high = right >> u64::BITS;

    // Accumulate the four 64-by-64 products just far enough to recover the high 128 bits.
    let low_product = left_low * right_low;
    let lower_cross = left_high * right_low + (low_product >> u64::BITS);
    let upper_cross = left_low * right_high + (lower_cross & LOW_MASK);
    let high = left_high * right_high + (lower_cross >> u64::BITS) + (upper_cross >> u64::BITS);
    debug_assert_ne!(high, 0);
    u64::from(u128::BITS) + u64::from(u128::BITS - high.leading_zeros())
}

#[cfg(feature = "integer-gmp")]
#[derive(Clone, Copy)]
struct FixedLimbAbsoluteBitStatistics {
    l1_bits: u64,
    maximum_bits: u64,
    has_negative: bool,
}

#[cfg(feature = "integer-gmp")]
/// Compute L1-sum and maximum-magnitude bit lengths plus sign presence with a bounded
/// native-limb accumulator.
///
/// This supplies a safe Kronecker digit bound without allocating GMP sums when every coefficient
/// magnitude occupies at most eight limbs and their complete sum fits in nine limbs.
fn try_fixed_limb_absolute_bit_statistics(
    coefficients: &[Integer],
) -> Option<FixedLimbAbsoluteBitStatistics> {
    const MAX_COEFFICIENT_LIMBS: usize = 8;
    let mut sum = [0u64; MAX_COEFFICIENT_LIMBS + 1];
    let mut maximum_bits = 0u64;
    let mut has_negative = false;

    #[inline(always)]
    fn add_magnitude(sum: &mut [u64; 9], magnitude: &[u64]) -> Option<u64> {
        if magnitude.len() > 8 {
            return None;
        }
        let significant_len = magnitude
            .iter()
            .rposition(|&limb| limb != 0)
            .map_or(0, |position| position + 1);
        if significant_len == 0 {
            return Some(0);
        }

        let mut carry = false;
        for (position, &limb) in magnitude[..significant_len].iter().enumerate() {
            let (value, add_overflow) = sum[position].overflowing_add(limb);
            let (value, carry_overflow) = value.overflowing_add(u64::from(carry));
            sum[position] = value;
            carry = add_overflow || carry_overflow;
        }
        let mut position = significant_len;
        while carry {
            let slot = sum.get_mut(position)?;
            let (value, overflow) = slot.overflowing_add(1);
            *slot = value;
            carry = overflow;
            position += 1;
        }

        let high = magnitude[significant_len - 1];
        Some(
            (significant_len as u64 - 1) * u64::from(u64::BITS)
                + u64::from(u64::BITS - high.leading_zeros()),
        )
    }

    for coefficient in coefficients {
        let (coefficient_bits, negative) = match coefficient {
            Integer::Single(value) => {
                let magnitude = value.unsigned_abs();
                (
                    add_magnitude(&mut sum, std::slice::from_ref(&magnitude))?,
                    *value < 0,
                )
            }
            Integer::Double(value) => {
                let value = value.get();
                let magnitude = value.unsigned_abs();
                let words = [magnitude as u64, (magnitude >> 64) as u64];
                (add_magnitude(&mut sum, &words)?, value < 0)
            }
            Integer::Large(value) => (
                add_magnitude(&mut sum, value.as_raw().as_limbs())?,
                value.is_negative(),
            ),
        };
        maximum_bits = maximum_bits.max(coefficient_bits);
        has_negative |= negative;
    }

    let sum_len = sum
        .iter()
        .rposition(|&limb| limb != 0)
        .map_or(0, |position| position + 1);
    let l1_bits = if sum_len == 0 {
        0
    } else {
        (sum_len as u64 - 1) * u64::from(u64::BITS)
            + u64::from(u64::BITS - sum[sum_len - 1].leading_zeros())
    };
    Some(FixedLimbAbsoluteBitStatistics {
        l1_bits,
        maximum_bits,
        has_negative,
    })
}

#[cfg(feature = "integer-gmp")]
#[inline]
fn bit_bound_from_factor_bits(left_bits: u64, right_bits: u64) -> Option<u64> {
    if left_bits == 0 || right_bits == 0 {
        Some(0)
    } else {
        left_bits.checked_add(right_bits)
    }
}

#[cfg(feature = "integer-gmp")]
/// Encode one GMP-backed coefficient as a sign and native magnitude limbs for radix packing.
///
/// `negate` applies the sign chosen for the complete packed polynomial. `borrow` subtracts the
/// carry propagated from the preceding signed radix digit. The caller converts a negative
/// magnitude to its fixed-width two's-complement representation.
#[inline(always)]
fn try_encode_large_kronecker_digit(
    digit: &mut [u64],
    coefficient: &MultiPrecisionInteger,
    negate: bool,
    borrow: bool,
) -> Option<bool> {
    if digit.is_empty() {
        return None;
    }

    digit.fill(0);
    let magnitude_limbs = coefficient.as_raw().as_limbs();
    if magnitude_limbs.len() > digit.len() {
        return None;
    }
    digit[..magnitude_limbs.len()].copy_from_slice(magnitude_limbs);

    let nonzero = !magnitude_limbs.is_empty();
    let mut negative = nonzero && (coefficient.is_negative() != negate);
    if !borrow {
        return Some(negative);
    }

    if negative {
        let mut carry = true;
        for limb in digit.iter_mut() {
            let (value, overflow) = limb.overflowing_add(u64::from(carry));
            *limb = value;
            carry = overflow;
            if !carry {
                break;
            }
        }
        if carry {
            return None;
        }
    } else if nonzero {
        let mut borrow = true;
        for limb in digit.iter_mut() {
            let (value, overflow) = limb.overflowing_sub(u64::from(borrow));
            *limb = value;
            borrow = overflow;
            if !borrow {
                break;
            }
        }
        debug_assert!(!borrow);
    } else {
        digit[0] = 1;
        negative = true;
    }

    Some(negative)
}

#[cfg(feature = "integer-gmp")]
enum NormalizedKroneckerDigit {
    Fixed(i128),
    Large { limb_count: usize, negative: bool },
}

#[cfg(feature = "integer-gmp")]
enum DecodedKroneckerDigit {
    Fixed {
        value: i128,
        carry_out: bool,
    },
    Large {
        magnitude_limbs: [u64; 8],
        limb_count: usize,
        negative: bool,
        carry_out: bool,
    },
}

#[cfg(feature = "integer-gmp")]
#[inline(always)]
fn try_normalize_fixed_kronecker_digit<const LIMBS: usize>(
    words: &mut [u64; LIMBS],
    digit_bits: usize,
    carry_in: bool,
    signed_coefficients: bool,
    product_negative: bool,
) -> Option<(NormalizedKroneckerDigit, bool)> {
    if digit_bits == 0 || LIMBS == 0 || LIMBS > 8 || digit_bits.div_ceil(64) != LIMBS {
        return None;
    }

    let last = LIMBS - 1;
    let trailing_bits = digit_bits % 64;
    debug_assert!(
        trailing_bits == 0 || words[last] >> trailing_bits == 0,
        "the extracted radix digit must be masked to its partial final limb",
    );

    let mut overflow_radix = false;
    if carry_in {
        let mut carry = true;
        for word in words.iter_mut() {
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
    let radix_negative =
        signed_coefficients && words[sign_bit_index / 64] & (1u64 << (sign_bit_index % 64)) != 0;
    let carry_out = signed_coefficients && (overflow_radix || radix_negative);

    if radix_negative {
        let mut carry = true;
        for word in words.iter_mut() {
            let (value, overflow) = (!*word).overflowing_add(u64::from(carry));
            *word = value;
            carry = overflow;
        }
        debug_assert!(!carry);
        if trailing_bits != 0 {
            words[last] &= (1u64 << trailing_bits) - 1;
        }
    }

    let magnitude_fits_u128 = LIMBS <= 2 || words[2..].iter().all(|&word| word == 0);
    let magnitude = magnitude_fits_u128.then(|| {
        u128::from(words[0])
            | if LIMBS > 1 {
                u128::from(words[1]) << 64
            } else {
                0
            }
    });
    let final_negative = magnitude.map_or(radix_negative != product_negative, |magnitude| {
        magnitude != 0 && (radix_negative != product_negative)
    });
    let fixed = magnitude.and_then(|magnitude| {
        if final_negative {
            const I128_MIN_MAGNITUDE: u128 = 1u128 << 127;
            if magnitude > I128_MIN_MAGNITUDE {
                None
            } else if magnitude == I128_MIN_MAGNITUDE {
                Some(i128::MIN)
            } else {
                Some(-(magnitude as i128))
            }
        } else if magnitude <= i128::MAX as u128 {
            Some(magnitude as i128)
        } else {
            None
        }
    });
    if let Some(value) = fixed {
        return Some((NormalizedKroneckerDigit::Fixed(value), carry_out));
    }

    let significant_limb_count = words.iter().rposition(|&word| word != 0)? + 1;
    Some((
        NormalizedKroneckerDigit::Large {
            limb_count: significant_limb_count,
            negative: final_negative,
        },
        carry_out,
    ))
}

#[cfg(feature = "integer-gmp")]
#[inline(always)]
fn try_decode_fixed_kronecker_digit<const LIMBS: usize>(
    mut words: [u64; LIMBS],
    digit_bits: usize,
    carry_in: bool,
    signed_coefficients: bool,
    product_negative: bool,
) -> Option<DecodedKroneckerDigit> {
    let (normalized, carry_out) = try_normalize_fixed_kronecker_digit(
        &mut words,
        digit_bits,
        carry_in,
        signed_coefficients,
        product_negative,
    )?;
    match normalized {
        NormalizedKroneckerDigit::Fixed(value) => {
            Some(DecodedKroneckerDigit::Fixed { value, carry_out })
        }
        NormalizedKroneckerDigit::Large {
            limb_count,
            negative,
        } => {
            let mut magnitude_limbs = [0u64; 8];
            magnitude_limbs[..LIMBS].copy_from_slice(&words);
            Some(DecodedKroneckerDigit::Large {
                magnitude_limbs,
                limb_count,
                negative,
                carry_out,
            })
        }
    }
}

#[cfg(feature = "integer-gmp")]
/// Normalize one signed radix digit using native limbs.
///
/// The incoming carry is applied before the radix sign is read. Values outside `i128` retain
/// their normalized sign-magnitude limbs so the caller can construct the final GMP coefficient
/// without repeating carry propagation or subtracting the radix.
#[inline(always)]
fn try_decode_small_kronecker_digit(
    digit_limbs: &[u64],
    digit_bits: usize,
    carry_in: bool,
    signed_coefficients: bool,
    product_negative: bool,
) -> Option<DecodedKroneckerDigit> {
    if digit_bits == 0 || digit_limbs.len() != digit_bits.div_ceil(64) {
        return None;
    }

    match digit_limbs.len() {
        1 => try_decode_fixed_kronecker_digit(
            [digit_limbs[0]],
            digit_bits,
            carry_in,
            signed_coefficients,
            product_negative,
        ),
        2 => try_decode_fixed_kronecker_digit(
            [digit_limbs[0], digit_limbs[1]],
            digit_bits,
            carry_in,
            signed_coefficients,
            product_negative,
        ),
        3 => try_decode_fixed_kronecker_digit(
            [digit_limbs[0], digit_limbs[1], digit_limbs[2]],
            digit_bits,
            carry_in,
            signed_coefficients,
            product_negative,
        ),
        4 => try_decode_fixed_kronecker_digit(
            [
                digit_limbs[0],
                digit_limbs[1],
                digit_limbs[2],
                digit_limbs[3],
            ],
            digit_bits,
            carry_in,
            signed_coefficients,
            product_negative,
        ),
        5 => try_decode_fixed_kronecker_digit(
            [
                digit_limbs[0],
                digit_limbs[1],
                digit_limbs[2],
                digit_limbs[3],
                digit_limbs[4],
            ],
            digit_bits,
            carry_in,
            signed_coefficients,
            product_negative,
        ),
        6 => try_decode_fixed_kronecker_digit(
            [
                digit_limbs[0],
                digit_limbs[1],
                digit_limbs[2],
                digit_limbs[3],
                digit_limbs[4],
                digit_limbs[5],
            ],
            digit_bits,
            carry_in,
            signed_coefficients,
            product_negative,
        ),
        7 => try_decode_fixed_kronecker_digit(
            [
                digit_limbs[0],
                digit_limbs[1],
                digit_limbs[2],
                digit_limbs[3],
                digit_limbs[4],
                digit_limbs[5],
                digit_limbs[6],
            ],
            digit_bits,
            carry_in,
            signed_coefficients,
            product_negative,
        ),
        8 => try_decode_fixed_kronecker_digit(
            [
                digit_limbs[0],
                digit_limbs[1],
                digit_limbs[2],
                digit_limbs[3],
                digit_limbs[4],
                digit_limbs[5],
                digit_limbs[6],
                digit_limbs[7],
            ],
            digit_bits,
            carry_in,
            signed_coefficients,
            product_negative,
        ),
        _ => None,
    }
}

#[cfg(feature = "integer-gmp")]
/// Decode consecutive bounded-width radix digits directly from a product limb slice.
///
/// A fixed limb count lets the compiler unroll extraction and normalization. The bit cursor is
/// advanced from the preceding position so each output avoids recomputing a multiplication,
/// division, and dynamic digit-buffer size.
#[inline]
fn try_decode_fixed_kronecker_digits<const LIMBS: usize, F>(
    limbs: &[u64],
    digit_bits: usize,
    output_len: usize,
    signed_coefficients: bool,
    product_negative: bool,
    mut output_index: F,
) -> Option<Vec<(u32, Integer)>>
where
    F: FnMut(usize) -> Option<u32>,
{
    if digit_bits == 0 || LIMBS == 0 || LIMBS > 8 || digit_bits.div_ceil(64) != LIMBS {
        return None;
    }

    let trailing_bits = digit_bits % 64;
    let mut carry = false;
    let mut limb_index = 0usize;
    let mut shift = 0usize;
    let mut output = Vec::with_capacity(output_len);
    for index in 0..output_len {
        let mut words = [0u64; LIMBS];
        let required_end = limb_index.checked_add(LIMBS)?;
        let direct = required_end <= limbs.len() && (shift == 0 || required_end < limbs.len());
        if direct {
            for (offset, word) in words.iter_mut().enumerate() {
                // SAFETY: `direct` checks the last low limb and, for nonzero shifts, the last high
                // limb needed by every iteration of this fixed-size loop.
                let low = unsafe { *limbs.get_unchecked(limb_index + offset) };
                *word = low >> shift;
                if shift != 0 {
                    *word |=
                        unsafe { *limbs.get_unchecked(limb_index + offset + 1) } << (64 - shift);
                }
            }
        } else {
            for (offset, word) in words.iter_mut().enumerate() {
                *word = limbs.get(limb_index + offset).copied().unwrap_or(0) >> shift;
                if shift != 0 {
                    *word |=
                        limbs.get(limb_index + offset + 1).copied().unwrap_or(0) << (64 - shift);
                }
            }
        }
        if trailing_bits != 0 {
            words[LIMBS - 1] &= (1u64 << trailing_bits) - 1;
        }

        let (normalized, carry_out) = try_normalize_fixed_kronecker_digit(
            &mut words,
            digit_bits,
            carry,
            signed_coefficients,
            product_negative,
        )?;
        match normalized {
            NormalizedKroneckerDigit::Fixed(value) => {
                carry = carry_out;
                if value != 0 {
                    output.push((output_index(index)?, Integer::from_double(value)));
                }
            }
            NormalizedKroneckerDigit::Large {
                limb_count,
                negative,
            } => {
                carry = carry_out;
                let digit =
                    MultiPrecisionInteger::try_from_lsf_limbs(&words[..limb_count], negative)?;
                debug_assert!(digit.to_i128().is_none());
                output.push((output_index(index)?, Integer::Large(digit)));
            }
        }

        let advanced_shift = shift.checked_add(digit_bits)?;
        limb_index = limb_index.checked_add(advanced_shift / 64)?;
        shift = advanced_shift % 64;
    }
    debug_assert!(!carry);
    Some(output)
}

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

        // Fixed-width strategies allocate and scan the entire mixed-radix coefficient box. For
        // sparse boxes the generic polynomial fallback is cheaper: it keeps only a dense u32
        // index and materializes coefficients that are actually reached. Small boxes are cheap
        // unconditionally; larger boxes require at least one coefficient product per cell.
        let product_count = self
            .left_coefficients
            .len()
            .checked_mul(self.right_coefficients.len())?;
        if self.output_len <= product_count.max(1024) {
            if let Some(output) = self.try_i64() {
                return Some(output);
            }
            if let Some(output) = self.try_i64_i128() {
                return Some(output);
            }
            if let Some(output) = self.try_i128() {
                return Some(output);
            }
        }

        #[cfg(feature = "integer-gmp")]
        {
            if let Some(output) = self.try_kronecker() {
                return Some(output);
            }
            self.try_large_array()
        }
        #[cfg(feature = "integer-malachite")]
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

        // Most fixed-coefficient dense products are short enough to keep both converted inputs
        // on the stack. Larger inputs spill transparently while retaining the same convolution.
        let collect_single = |coefficients: &[Integer]| {
            coefficients
                .iter()
                .map(|coefficient| match coefficient {
                    Integer::Single(value) => *value,
                    Integer::Double(_) | Integer::Large(_) => unreachable!(),
                })
                .collect::<SmallVec<[i64; 128]>>()
        };
        let left = collect_single(left_coefficients);
        let right = collect_single(right_coefficients);

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

    #[cfg(feature = "integer-gmp")]
    /// Multiply a dense-indexed convolution by Kronecker substitution.
    ///
    /// Each coefficient is encoded as a sufficiently wide signed radix digit, the two complete
    /// polynomials are packed into native limb vectors, and one GMP limb multiplication computes
    /// the convolution. A compact additive embedding is used when the supplied indices describe
    /// a sufficiently large total-degree simplex with large holes in its mixed-radix box. The
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
        if gmp::NUMB_BITS != 64 || gmp::NAIL_BITS != 0 {
            return None;
        }
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

        const MIN_CONTIGUOUS_KRONECKER_TERMS: usize = 31;
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

        let normalize_contiguous_indices = simplex_layout.is_none()
            && consecutive_support(packed_left_indices)
            && consecutive_support(packed_right_indices);
        let left_index_offset = if normalize_contiguous_indices {
            *packed_left_indices.first()?
        } else {
            0
        };
        let right_index_offset = if normalize_contiguous_indices {
            *packed_right_indices.first()?
        } else {
            0
        };
        let output_index_offset = left_index_offset.checked_add(right_index_offset)?;
        let decoded_output_len = if normalize_contiguous_indices {
            active_output_span
        } else {
            packed_output_len
        };

        enum AbsoluteStatistics {
            Fixed {
                sum: u128,
                maximum: u128,
            },
            Large {
                sum: MultiPrecisionInteger,
                maximum: MultiPrecisionInteger,
            },
        }

        fn absolute_statistics(coefficients: &[Integer]) -> AbsoluteStatistics {
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
                return AbsoluteStatistics::Fixed {
                    sum: fixed_sum,
                    maximum: fixed_maximum,
                };
            }

            let mut sum = MultiPrecisionInteger::default();
            let mut maximum = None::<&Integer>;
            for coefficient in coefficients {
                match coefficient {
                    Integer::Single(value) => sum += value.unsigned_abs(),
                    Integer::Double(value) => sum += value.get().unsigned_abs(),
                    Integer::Large(value) if value.is_negative() => sum -= value,
                    Integer::Large(value) => sum += value,
                }
                if maximum.is_none_or(|current| coefficient.abs_cmp(current).is_gt()) {
                    maximum = Some(coefficient);
                }
            }

            let mut maximum = maximum.unwrap().clone().to_multi_prec();
            if maximum.is_negative() {
                maximum = -maximum;
            }

            AbsoluteStatistics::Large { sum, maximum }
        }

        // Fixed-limb statistics revisit every coefficient of a GMP-backed input and return the
        // complete sign flag, so this preliminary scan can stop at the first large coefficient.
        // If bounded statistics fail, the fallback below resumes after that coefficient.
        let mut signed_coefficients = false;
        let first_large_index =
            left_coefficients
                .iter()
                .chain(right_coefficients)
                .position(|coefficient| {
                    signed_coefficients |= coefficient.is_negative();
                    matches!(coefficient, Integer::Large(_))
                });
        let contains_large = first_large_index.is_some();

        let fixed_limb_bound = if contains_large {
            match (
                try_fixed_limb_absolute_bit_statistics(left_coefficients),
                try_fixed_limb_absolute_bit_statistics(right_coefficients),
            ) {
                (Some(left), Some(right)) => {
                    signed_coefficients = left.has_negative || right.has_negative;
                    Some(
                        bit_bound_from_factor_bits(left.l1_bits, right.maximum_bits)?.min(
                            bit_bound_from_factor_bits(right.l1_bits, left.maximum_bits)?,
                        ),
                    )
                }
                _ => {
                    signed_coefficients |= left_coefficients
                        .iter()
                        .chain(right_coefficients)
                        .skip(first_large_index.unwrap() + 1)
                        .any(Integer::is_negative);
                    None
                }
            }
        } else {
            None
        };

        // Wider values fall back to a max-bit/collision bound only when its logarithmic slack is
        // at most one thirty-second of the two coefficient widths. This caps the possible radix
        // inflation while avoiding a multiprecision L1 pass. Height-skewed inputs retain the
        // tighter exact calculation below.
        let bounded_collision_bound = if contains_large && fixed_limb_bound.is_none() {
            let maximum_bits = |coefficients: &[Integer]| {
                coefficients
                    .iter()
                    .map(Integer::significant_bits)
                    .max()
                    .unwrap_or(0)
            };
            let maximum_product_bits =
                maximum_bits(left_coefficients).checked_add(maximum_bits(right_coefficients))?;
            let collision_count = left_coefficients.len().min(right_coefficients.len());
            let collision_slack = if collision_count <= 1 {
                0
            } else {
                u64::from(usize::BITS - (collision_count - 1).leading_zeros())
            };
            collision_slack
                .checked_mul(32)
                .is_some_and(|scaled| scaled <= maximum_product_bits)
                .then(|| maximum_product_bits.checked_add(collision_slack))
                .flatten()
        } else {
            None
        };

        // Every convolution coefficient is bounded both by ||a||_1 max(b) and by
        // ||b||_1 max(a). Fixed coefficients use exact `u128` products; exceptional wide values
        // use the same exact multiprecision products as before.
        let coefficient_bound_bits = if let Some(bound) = fixed_limb_bound {
            bound
        } else if let Some(bound) = bounded_collision_bound {
            bound
        } else {
            let left_statistics = absolute_statistics(left_coefficients);
            let right_statistics = absolute_statistics(right_coefficients);
            match (left_statistics, right_statistics) {
                (
                    AbsoluteStatistics::Fixed {
                        sum: left_sum,
                        maximum: left_maximum,
                    },
                    AbsoluteStatistics::Fixed {
                        sum: right_sum,
                        maximum: right_maximum,
                    },
                ) => u128_product_significant_bits(left_sum, right_maximum)
                    .min(u128_product_significant_bits(right_sum, left_maximum)),
                (left_statistics, right_statistics) => {
                    let into_large = |statistics| match statistics {
                        AbsoluteStatistics::Fixed { sum, maximum } => (
                            MultiPrecisionInteger::from(sum),
                            MultiPrecisionInteger::from(maximum),
                        ),
                        AbsoluteStatistics::Large { sum, maximum } => (sum, maximum),
                    };
                    let (left_sum, left_maximum) = into_large(left_statistics);
                    let (right_sum, right_maximum) = into_large(right_statistics);
                    let left_l1_bound = &left_sum * &right_maximum;
                    let right_l1_bound = &right_sum * &left_maximum;
                    left_l1_bound
                        .significant_bits()
                        .min(right_l1_bound.significant_bits())
                }
            }
        };
        let digit_bits = coefficient_bound_bits.checked_add(u64::from(signed_coefficients))?;
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
        let left_packed_bits = (packed_left_indices
            .iter()
            .copied()
            .max()?
            .checked_sub(left_index_offset)? as usize
            + 1)
        .checked_mul(digit_bits_usize)?;
        let right_packed_bits = (packed_right_indices
            .iter()
            .copied()
            .max()?
            .checked_sub(right_index_offset)? as usize
            + 1)
        .checked_mul(digit_bits_usize)?;
        if left_packed_bits > MAX_PACKED_BITS || right_packed_bits > MAX_PACKED_BITS {
            return None;
        }

        struct PackedKronecker {
            limbs: Vec<u64>,
            negative: bool,
        }

        fn pack(
            coefficients: &[Integer],
            indices: &[u32],
            index_offset: u32,
            digit_bits: usize,
        ) -> Option<PackedKronecker> {
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

            /// Pack consecutive coefficients with a fixed stack digit.
            ///
            /// This path is used by dense univariate products whose signed radix digit occupies at
            /// most eight limbs. The general packer below handles sparse supports and wider
            /// coefficients.
            #[inline]
            fn pack_contiguous<const LIMBS: usize>(
                coefficients: &[Integer],
                indices: &[u32],
                index_offset: u32,
                digit_bits: usize,
            ) -> Option<PackedKronecker> {
                if coefficients.len() != indices.len()
                    || coefficients.is_empty()
                    || digit_bits.div_ceil(64) != LIMBS
                    || !indices
                        .windows(2)
                        .all(|pair| pair[0].checked_add(1) == Some(pair[1]))
                {
                    return None;
                }

                let leading_negative = coefficients.last()?.is_negative();
                let digit_count = indices.last()?.checked_sub(index_offset)? as usize + 1;
                let packed_bits = digit_count.checked_mul(digit_bits)?;
                let mut packed_limbs = vec![0u64; packed_bits.checked_add(63)? / 64];
                let mut borrow = false;
                let mut bit_index = (indices.first()?.checked_sub(index_offset)? as usize)
                    .checked_mul(digit_bits)?;

                for coefficient in coefficients {
                    let mut digit = [0u64; LIMBS];
                    let borrow_in = borrow;
                    borrow = match coefficient {
                        Integer::Single(value) => Some(encode_primitive_digit(
                            &mut digit,
                            i128::from(*value),
                            leading_negative,
                            borrow_in,
                        )),
                        Integer::Double(value) => Some(encode_primitive_digit(
                            &mut digit,
                            value.get(),
                            leading_negative,
                            borrow_in,
                        )),
                        Integer::Large(value) => try_encode_large_kronecker_digit(
                            &mut digit,
                            value,
                            leading_negative,
                            borrow_in,
                        ),
                    }?;
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
                        *digit.last_mut()? &= low_mask(digit_bits % 64);
                    }

                    let limb_index = bit_index / 64;
                    let shift = bit_index % 64;
                    if shift == 0 {
                        packed_limbs[limb_index..limb_index + LIMBS].copy_from_slice(&digit);
                    } else {
                        let preserved_low = packed_limbs[limb_index] & low_mask(shift);
                        packed_limbs[limb_index] = preserved_low | (digit[0] << shift);
                        for offset in 1..LIMBS {
                            packed_limbs[limb_index + offset] =
                                (digit[offset - 1] >> (64 - shift)) | (digit[offset] << shift);
                        }
                        if let Some(high) = packed_limbs.get_mut(limb_index + LIMBS) {
                            *high = digit[LIMBS - 1] >> (64 - shift);
                        }
                    }
                    bit_index = bit_index.checked_add(digit_bits)?;
                }
                if borrow {
                    return None;
                }

                while packed_limbs.last() == Some(&0) {
                    packed_limbs.pop();
                }
                (!packed_limbs.is_empty()).then_some(PackedKronecker {
                    limbs: packed_limbs,
                    negative: leading_negative,
                })
            }

            let fixed_contiguous = match digit_bits.div_ceil(64) {
                1 => pack_contiguous::<1>(coefficients, indices, index_offset, digit_bits),
                2 => pack_contiguous::<2>(coefficients, indices, index_offset, digit_bits),
                3 => pack_contiguous::<3>(coefficients, indices, index_offset, digit_bits),
                4 => pack_contiguous::<4>(coefficients, indices, index_offset, digit_bits),
                5 => pack_contiguous::<5>(coefficients, indices, index_offset, digit_bits),
                6 => pack_contiguous::<6>(coefficients, indices, index_offset, digit_bits),
                7 => pack_contiguous::<7>(coefficients, indices, index_offset, digit_bits),
                8 => pack_contiguous::<8>(coefficients, indices, index_offset, digit_bits),
                _ => None,
            };
            if fixed_contiguous.is_some() {
                return fixed_contiguous;
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
            let digit_count =
                indices[term_index(indices.len() - 1)].checked_sub(index_offset)? as usize + 1;
            let packed_bits = digit_count.checked_mul(digit_bits)?;
            let mut limbs = vec![0u64; packed_bits.checked_add(63)? / 64];
            let mut next_index = 0usize;
            let mut borrow = false;
            let mut digit = SmallVec::<[u64; 4]>::new();
            digit.resize(digit_bits.div_ceil(64), 0);

            for position in 0..coefficients.len() {
                let term = term_index(position);
                let coefficient = unsafe { coefficients.get_unchecked(term) };
                let index =
                    unsafe { *indices.get_unchecked(term) }.checked_sub(index_offset)? as usize;
                debug_assert!(index >= next_index);
                if borrow {
                    fill_ones(&mut limbs, next_index * digit_bits, index * digit_bits);
                }

                let borrow_in = borrow;
                borrow = match coefficient {
                    Integer::Single(value) => Some(encode_primitive_digit(
                        &mut digit,
                        i128::from(*value),
                        leading_negative,
                        borrow_in,
                    )),
                    Integer::Double(value) => Some(encode_primitive_digit(
                        &mut digit,
                        value.get(),
                        leading_negative,
                        borrow_in,
                    )),
                    Integer::Large(value) => try_encode_large_kronecker_digit(
                        &mut digit,
                        value,
                        leading_negative,
                        borrow_in,
                    ),
                }?;

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
            if borrow {
                return None;
            }

            while limbs.last() == Some(&0) {
                limbs.pop();
            }
            (!limbs.is_empty()).then_some(PackedKronecker {
                limbs,
                negative: leading_negative,
            })
        }

        let left = pack(
            left_coefficients,
            packed_left_indices,
            left_index_offset,
            digit_bits_usize,
        )?;
        let right = pack(
            right_coefficients,
            packed_right_indices,
            right_index_offset,
            digit_bits_usize,
        )?;
        let product_negative = left.negative != right.negative;
        let (long, short) = if left.limbs.len() >= right.limbs.len() {
            (left.limbs.as_slice(), right.limbs.as_slice())
        } else {
            (right.limbs.as_slice(), left.limbs.as_slice())
        };
        let product_limb_count = long.len().checked_add(short.len())?;
        let long_limb_count = gmp::size_t::try_from(long.len()).ok()?;
        let short_limb_count = gmp::size_t::try_from(short.len()).ok()?;
        let mut product_limbs = Vec::<u64>::with_capacity(product_limb_count);
        // Both packed operands are normalized nonzero magnitudes and the 2^29-bit workspace
        // bound makes their checked GMP sizes representable. On this 64-bit, nail-free path the
        // native limb pointers have the same representation as `u64`. GMP initializes exactly
        // the sum-length, nonoverlapping destination before its vector length is exposed.
        unsafe {
            gmp::mpn_mul(
                product_limbs.as_mut_ptr().cast(),
                long.as_ptr().cast(),
                long_limb_count,
                short.as_ptr().cast(),
                short_limb_count,
            );
            product_limbs.set_len(product_limb_count);
        }
        drop(left);
        drop(right);
        if product_limbs.last() == Some(&0) {
            product_limbs.pop();
        }
        debug_assert!(!product_limbs.is_empty());
        let limbs = product_limbs.as_slice();
        let limbs_per_digit = digit_bits_usize.div_ceil(64);

        macro_rules! decode_fixed_digits {
            ($limb_count:literal) => {{
                if let Some(layout) = simplex_layout.as_ref() {
                    try_decode_fixed_kronecker_digits::<$limb_count, _>(
                        limbs,
                        digit_bits_usize,
                        decoded_output_len,
                        signed_coefficients,
                        product_negative,
                        |index| {
                            let decoded = *layout.decode_indices.get(index)?;
                            debug_assert_ne!(decoded, u32::MAX);
                            Some(decoded)
                        },
                    )
                } else {
                    try_decode_fixed_kronecker_digits::<$limb_count, _>(
                        limbs,
                        digit_bits_usize,
                        decoded_output_len,
                        signed_coefficients,
                        product_negative,
                        |index| u32::try_from(index).ok()?.checked_add(output_index_offset),
                    )
                }
            }};
        }

        let fixed_output = match limbs_per_digit {
            1 => decode_fixed_digits!(1),
            2 => decode_fixed_digits!(2),
            3 => decode_fixed_digits!(3),
            4 => decode_fixed_digits!(4),
            5 => decode_fixed_digits!(5),
            6 => decode_fixed_digits!(6),
            7 => decode_fixed_digits!(7),
            8 => decode_fixed_digits!(8),
            _ => None,
        };
        if let Some(mut output) = fixed_output {
            if simplex_layout.is_some() {
                output.sort_unstable_by_key(|term| term.0);
            }
            return Some(output);
        }

        let mut carry = false;
        let mut radix = None;
        let mut output = Vec::with_capacity(decoded_output_len);
        let mut digit_limbs = SmallVec::<[u64; 4]>::new();
        digit_limbs.resize(limbs_per_digit, 0);
        let output_index = |index: usize| {
            if let Some(layout) = simplex_layout.as_ref() {
                let decoded = *layout.decode_indices.get(index)?;
                debug_assert_ne!(decoded, u32::MAX);
                Some(decoded)
            } else {
                u32::try_from(index).ok()?.checked_add(output_index_offset)
            }
        };
        for index in 0..decoded_output_len {
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

            if let Some(decoded) = try_decode_small_kronecker_digit(
                &digit_limbs,
                digit_bits_usize,
                carry,
                signed_coefficients,
                product_negative,
            ) {
                match decoded {
                    DecodedKroneckerDigit::Fixed { value, carry_out } => {
                        carry = carry_out;
                        if value != 0 {
                            output.push((output_index(index)?, Integer::from_double(value)));
                        }
                    }
                    DecodedKroneckerDigit::Large {
                        magnitude_limbs,
                        limb_count,
                        negative,
                        carry_out,
                    } => {
                        carry = carry_out;
                        let digit = MultiPrecisionInteger::try_from_lsf_limbs(
                            &magnitude_limbs[..limb_count],
                            negative,
                        )?;
                        debug_assert!(digit.to_i128().is_none());
                        output.push((output_index(index)?, Integer::Large(digit)));
                    }
                }
                continue;
            }

            let mut digit = MultiPrecisionInteger::try_from_lsf_limbs(&digit_limbs, false)?;
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

    #[cfg(feature = "integer-gmp")]
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
    #[cfg(all(feature = "integer-gmp", test))]
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

/// One integer multiplication evaluated through reusable mixed-radix coefficient chunks.
struct ChunkedDenseIntegerMul<'a> {
    dense: DenseIntegerMul<'a>,
    inner_len: usize,
}

impl<'a> ChunkedDenseIntegerMul<'a> {
    /// Validate the dense layout and its carry-free split into outer and inner indices.
    fn new(request: ChunkedDensePolynomialMulRequest<'a, Integer>) -> Option<Self> {
        const MAX_INNER_LEN: usize = 1 << 20;
        const MAX_OUTER_CHUNKS: usize = 256;

        let inner_len = request.inner_len;
        let dense = DenseIntegerMul::new(request.dense)?;
        if inner_len == 0
            || inner_len > MAX_INNER_LEN
            || !dense.output_len.is_multiple_of(inner_len)
            || dense.output_len / inner_len > MAX_OUTER_CHUNKS
            || dense
                .left_indices
                .windows(2)
                .any(|indices| indices[0] >= indices[1])
            || dense
                .right_indices
                .windows(2)
                .any(|indices| indices[0] >= indices[1])
        {
            return None;
        }

        Some(Self { dense, inner_len })
    }

    /// Multiply machine-size coefficients through a reusable inner accumulator.
    ///
    /// Terms are grouped into monotone outer rows without division or remainder operations. Each
    /// compatible row pair contributes to one output chunk, whose inner indices remain additive.
    /// The proven coefficient bound selects either `i64` or `i128` accumulator cells.
    fn run(self) -> Option<Vec<(u32, Integer)>> {
        const BLOCK_SIZE: usize = 128;

        let Self { dense, inner_len } = self;
        if dense.left_coefficients.is_empty() || dense.right_coefficients.is_empty() {
            return Some(Vec::new());
        }

        let outer_count = dense.output_len / inner_len;
        let prepare = |coefficients: &[Integer],
                       indices: &[u32],
                       values: &mut SmallVec<[i64; 128]>,
                       ranges: &mut SmallVec<[(usize, usize); 32]>| {
            let mut maximum = 0u64;
            let mut maximum_inner = 0usize;
            let mut maximum_row_len = 0usize;
            let mut position = 0usize;
            for outer in 0..outer_count {
                let row_start = position;
                let base = outer.checked_mul(inner_len)?;
                let end = base.checked_add(inner_len)?;
                while position < indices.len() && (indices[position] as usize) < end {
                    let index = indices[position] as usize;
                    if index < base {
                        return None;
                    }
                    let Integer::Single(coefficient) = coefficients[position] else {
                        return None;
                    };
                    let inner = index - base;
                    maximum = maximum.max(coefficient.unsigned_abs());
                    maximum_inner = maximum_inner.max(inner);
                    values.push(coefficient);
                    position += 1;
                }
                maximum_row_len = maximum_row_len.max(position - row_start);
                ranges.push((row_start, position));
            }
            (position == indices.len()).then_some((maximum, maximum_inner, maximum_row_len))
        };

        let mut left_values = SmallVec::<[i64; 128]>::new();
        left_values.reserve_exact(dense.left_coefficients.len());
        let mut left_ranges = SmallVec::<[(usize, usize); 32]>::new();
        left_ranges.reserve_exact(outer_count);
        let (maximum_left, maximum_left_inner, maximum_left_row_len) = prepare(
            dense.left_coefficients,
            dense.left_indices,
            &mut left_values,
            &mut left_ranges,
        )?;
        let mut right_values = SmallVec::<[i64; 128]>::new();
        right_values.reserve_exact(dense.right_coefficients.len());
        let mut right_ranges = SmallVec::<[(usize, usize); 32]>::new();
        right_ranges.reserve_exact(outer_count);
        let (maximum_right, maximum_right_inner, maximum_right_row_len) = prepare(
            dense.right_coefficients,
            dense.right_indices,
            &mut right_values,
            &mut right_ranges,
        )?;
        if maximum_left_inner.checked_add(maximum_right_inner)? >= inner_len {
            return None;
        }

        let coefficient_bound = u128::from(maximum_left)
            .checked_mul(u128::from(maximum_right))?
            .checked_mul(left_values.len().min(right_values.len()) as u128)?;
        if coefficient_bound > i128::MAX as u128 {
            return None;
        }

        let mut active_inner_lengths = SmallVec::<[usize; 32]>::new();
        active_inner_lengths.reserve_exact(outer_count);
        let mut active_scan = 0usize;
        for output_outer in 0..outer_count {
            let mut active_inner_len = 0usize;
            for left_outer in 0..=output_outer.min(outer_count - 1) {
                let right_outer = output_outer - left_outer;
                if right_outer >= outer_count {
                    continue;
                }
                let (left_start, left_end) = left_ranges[left_outer];
                let (right_start, right_end) = right_ranges[right_outer];
                if left_start == left_end || right_start == right_end {
                    continue;
                }
                let left_base = left_outer * inner_len;
                let right_base = right_outer * inner_len;
                active_inner_len = active_inner_len.max(
                    unsafe { *dense.left_indices.get_unchecked(left_end - 1) as usize } - left_base
                        + unsafe { *dense.right_indices.get_unchecked(right_end - 1) as usize }
                        - right_base
                        + 1,
                );
            }
            active_scan = active_scan.checked_add(active_inner_len)?;
            active_inner_lengths.push(active_inner_len);
        }
        if active_scan.checked_mul(2)? > dense.output_len {
            return None;
        }
        let outer_pair_probes = outer_count.checked_mul(outer_count.checked_add(1)?)? / 2;
        if outer_pair_probes.checked_mul(2)? > dense.output_len {
            return None;
        }

        fn multiply_fixed<T>(
            output_len: usize,
            inner_len: usize,
            outer_count: usize,
            active_scan: usize,
            active_inner_lengths: &[usize],
            accumulator: &mut [T],
            left_values: &[i64],
            left_indices: &[u32],
            left_ranges: &[(usize, usize)],
            right_values: &[i64],
            right_indices: &[u32],
            right_ranges: &[(usize, usize)],
            use_blocking: bool,
            into_integer: impl Fn(T) -> Integer,
        ) -> Vec<(u32, Integer)>
        where
            T: Copy
                + Default
                + PartialEq
                + From<i64>
                + std::ops::AddAssign
                + std::ops::Mul<Output = T>,
        {
            debug_assert_eq!(accumulator.len(), inner_len);
            let mut output = Vec::with_capacity(active_scan.min(output_len));

            for output_outer in 0..outer_count {
                for left_outer in 0..=output_outer.min(outer_count - 1) {
                    let right_outer = output_outer - left_outer;
                    if right_outer >= outer_count {
                        continue;
                    }
                    let (left_start, left_end) = left_ranges[left_outer];
                    let (right_start, right_end) = right_ranges[right_outer];
                    if left_start == left_end || right_start == right_end {
                        continue;
                    }

                    let output_base = output_outer * inner_len;
                    if use_blocking {
                        for left_block in (left_start..left_end).step_by(BLOCK_SIZE) {
                            for right_block in (right_start..right_end).step_by(BLOCK_SIZE) {
                                for left_index in
                                    left_block..(left_block + BLOCK_SIZE).min(left_end)
                                {
                                    let left_coefficient =
                                        T::from(unsafe { *left_values.get_unchecked(left_index) });
                                    let left_position =
                                        unsafe { *left_indices.get_unchecked(left_index) as usize };
                                    for right_index in
                                        right_block..(right_block + BLOCK_SIZE).min(right_end)
                                    {
                                        let position = left_position
                                            + unsafe {
                                                *right_indices.get_unchecked(right_index) as usize
                                            }
                                            - output_base;
                                        unsafe {
                                            *accumulator.get_unchecked_mut(position) +=
                                                left_coefficient
                                                    * T::from(
                                                        *right_values.get_unchecked(right_index),
                                                    );
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        for left_index in left_start..left_end {
                            let left_coefficient =
                                T::from(unsafe { *left_values.get_unchecked(left_index) });
                            let left_position =
                                unsafe { *left_indices.get_unchecked(left_index) as usize };
                            for right_index in right_start..right_end {
                                let position = left_position
                                    + unsafe { *right_indices.get_unchecked(right_index) as usize }
                                    - output_base;
                                unsafe {
                                    *accumulator.get_unchecked_mut(position) += left_coefficient
                                        * T::from(*right_values.get_unchecked(right_index));
                                }
                            }
                        }
                    }
                }

                let output_base = output_outer * inner_len;
                let active_inner_len = active_inner_lengths[output_outer];
                for (inner, coefficient) in accumulator[..active_inner_len].iter_mut().enumerate() {
                    let coefficient = std::mem::take(coefficient);
                    if coefficient != T::default() {
                        output.push(((output_base + inner) as u32, into_integer(coefficient)));
                    }
                }
            }
            output
        }

        let use_blocking = maximum_left_row_len > BLOCK_SIZE || maximum_right_row_len > BLOCK_SIZE;
        if coefficient_bound <= i64::MAX as u128 {
            if inner_len <= 256 {
                let mut accumulator = [0i64; 256];
                return Some(multiply_fixed::<i64>(
                    dense.output_len,
                    inner_len,
                    outer_count,
                    active_scan,
                    &active_inner_lengths,
                    &mut accumulator[..inner_len],
                    &left_values,
                    dense.left_indices,
                    &left_ranges,
                    &right_values,
                    dense.right_indices,
                    &right_ranges,
                    use_blocking,
                    Integer::from,
                ));
            }
            let mut accumulator = vec![0i64; inner_len];
            return Some(multiply_fixed::<i64>(
                dense.output_len,
                inner_len,
                outer_count,
                active_scan,
                &active_inner_lengths,
                &mut accumulator,
                &left_values,
                dense.left_indices,
                &left_ranges,
                &right_values,
                dense.right_indices,
                &right_ranges,
                use_blocking,
                Integer::from,
            ));
        }

        if inner_len <= 256 {
            let mut accumulator = [0i128; 256];
            return Some(multiply_fixed::<i128>(
                dense.output_len,
                inner_len,
                outer_count,
                active_scan,
                &active_inner_lengths,
                &mut accumulator[..inner_len],
                &left_values,
                dense.left_indices,
                &left_ranges,
                &right_values,
                dense.right_indices,
                &right_ranges,
                use_blocking,
                Integer::from_double,
            ));
        }
        let mut accumulator = vec![0i128; inner_len];
        Some(multiply_fixed::<i128>(
            dense.output_len,
            inner_len,
            outer_count,
            active_scan,
            &active_inner_lengths,
            &mut accumulator,
            &left_values,
            dense.left_indices,
            &left_ranges,
            &right_values,
            dense.right_indices,
            &right_ranges,
            use_blocking,
            Integer::from_double,
        ))
    }
}

/// One integer multiplication supported on a compact total-degree simplex.
struct TotalDegreeIntegerMul<'a> {
    request: TotalDegreePolynomialMulRequest<'a, Integer>,
}

impl<'a> TotalDegreeIntegerMul<'a> {
    const MAX_OUTPUT_CELLS: usize = 1 << 20;

    /// Retain the layout and coefficient slices for one total-degree multiplication.
    fn new(request: TotalDegreePolynomialMulRequest<'a, Integer>) -> Self {
        Self { request }
    }

    /// Run the first applicable compact total-degree multiplication strategy.
    fn run(self) -> Option<Vec<(u32, Integer)>> {
        if let Some(output) = self.try_single() {
            return Some(output);
        }

        #[cfg(feature = "integer-gmp")]
        {
            self.try_limb()
        }
        #[cfg(feature = "integer-malachite")]
        {
            let Self { request } = self;
            let _ = request;
            None
        }
    }

    /// Check the compact rank tables and every encoded input range used by the inner loops.
    fn layout_is_valid(&self) -> bool {
        let request = &self.request;
        if request.left_codes.len() != request.left_coefficients.len()
            || request.right_codes.len() != request.right_coefficients.len()
            || request.left_coefficients.is_empty()
            || request.right_coefficients.is_empty()
            || request.output_len == 0
            || request.output_len > Self::MAX_OUTPUT_CELLS
            || u32::try_from(request.output_len.saturating_sub(1)).is_err()
            || request.prefix_rank.len() != request.prefix_remaining.len()
            || request.suffix_code_count == 0
            || !request
                .suffix_rank
                .len()
                .is_multiple_of(request.suffix_code_count)
        {
            return false;
        }

        let suffix_rows = request.suffix_rank.len() / request.suffix_code_count;
        if request
            .prefix_remaining
            .iter()
            .copied()
            .filter(|remaining| *remaining != u8::MAX)
            .any(|remaining| remaining as usize >= suffix_rows)
        {
            return false;
        }

        let Some(maximum_prefix) =
            request
                .left_codes
                .iter()
                .map(|code| code.0)
                .max()
                .and_then(|left| {
                    request
                        .right_codes
                        .iter()
                        .map(|code| code.0)
                        .max()
                        .and_then(|right| left.checked_add(right))
                })
        else {
            return false;
        };
        let Some(maximum_suffix) =
            request
                .left_codes
                .iter()
                .map(|code| code.1)
                .max()
                .and_then(|left| {
                    request
                        .right_codes
                        .iter()
                        .map(|code| code.1)
                        .max()
                        .and_then(|right| left.checked_add(right))
                })
        else {
            return false;
        };

        maximum_prefix < request.prefix_rank.len() && maximum_suffix < request.suffix_code_count
    }

    /// Multiply machine-size input coefficients in native fixed-width output cells.
    ///
    /// This path serves total-degree products whose tagged coefficients are all
    /// [`Integer::Single`]. A bound on the largest product times the maximum collision count proves
    /// that every partial sum fits in either `i64` or `i128`; wider cases retain the limb kernel.
    fn try_single(&self) -> Option<Vec<(u32, Integer)>> {
        const BLOCK_SIZE: usize = 32;

        if !self.layout_is_valid() {
            return None;
        }
        let request = &self.request;
        let collect_single = |coefficients: &[Integer]| {
            let mut values = Vec::with_capacity(coefficients.len());
            let mut maximum = 0u64;
            for coefficient in coefficients {
                let Integer::Single(value) = coefficient else {
                    return None;
                };
                maximum = maximum.max(value.unsigned_abs());
                values.push(*value);
            }
            Some((values, maximum))
        };
        let (left_coefficients, maximum_left) = collect_single(request.left_coefficients)?;
        let (right_coefficients, maximum_right) = collect_single(request.right_coefficients)?;
        let coefficient_bound = u128::from(maximum_left)
            .checked_mul(u128::from(maximum_right))?
            .checked_mul(left_coefficients.len().min(right_coefficients.len()) as u128)?;
        if coefficient_bound > i128::MAX as u128 {
            return None;
        }

        /// Accumulate one compact-simplex product in the fixed type selected by the proven bound.
        fn multiply_fixed<T>(
            request: &TotalDegreePolynomialMulRequest<'_, Integer>,
            left_coefficients: &[T],
            right_coefficients: &[T],
            into_integer: impl Fn(T) -> Integer,
        ) -> Option<Vec<(u32, Integer)>>
        where
            T: Copy + Default + PartialEq + std::ops::AddAssign + std::ops::Mul<Output = T>,
        {
            let mut coefficients = vec![T::default(); request.output_len];
            for left_block in (0..left_coefficients.len()).step_by(BLOCK_SIZE) {
                for right_block in (0..right_coefficients.len()).step_by(BLOCK_SIZE) {
                    for left_index in
                        left_block..(left_block + BLOCK_SIZE).min(left_coefficients.len())
                    {
                        let left_coefficient =
                            unsafe { *left_coefficients.get_unchecked(left_index) };
                        let (left_prefix, left_suffix) =
                            unsafe { *request.left_codes.get_unchecked(left_index) };
                        for right_index in
                            right_block..(right_block + BLOCK_SIZE).min(right_coefficients.len())
                        {
                            let (right_prefix, right_suffix) =
                                unsafe { *request.right_codes.get_unchecked(right_index) };
                            let prefix = left_prefix + right_prefix;
                            let suffix = left_suffix + right_suffix;
                            let remaining_degree =
                                unsafe { *request.prefix_remaining.get_unchecked(prefix) };
                            if remaining_degree == u8::MAX {
                                return None;
                            }
                            let suffix_rank = unsafe {
                                *request.suffix_rank.get_unchecked(
                                    remaining_degree as usize * request.suffix_code_count + suffix,
                                )
                            };
                            if suffix_rank == u32::MAX {
                                return None;
                            }
                            let prefix_rank = unsafe { *request.prefix_rank.get_unchecked(prefix) };
                            if prefix_rank == u32::MAX {
                                return None;
                            }
                            let rank = usize::try_from(prefix_rank)
                                .ok()?
                                .checked_add(usize::try_from(suffix_rank).ok()?)?;
                            if rank >= request.output_len {
                                return None;
                            }
                            unsafe {
                                *coefficients.get_unchecked_mut(rank) += left_coefficient
                                    * *right_coefficients.get_unchecked(right_index);
                            }
                        }
                    }
                }
            }

            let mut output = Vec::with_capacity(request.output_len);
            for (rank, coefficient) in coefficients.into_iter().enumerate() {
                if coefficient != T::default() {
                    output.push((u32::try_from(rank).ok()?, into_integer(coefficient)));
                }
            }
            Some(output)
        }

        if coefficient_bound <= i64::MAX as u128 {
            return multiply_fixed(
                request,
                &left_coefficients,
                &right_coefficients,
                Integer::from,
            );
        }

        let left_coefficients = left_coefficients
            .into_iter()
            .map(i128::from)
            .collect::<Vec<_>>();
        let right_coefficients = right_coefficients
            .into_iter()
            .map(i128::from)
            .collect::<Vec<_>>();
        multiply_fixed(
            request,
            &left_coefficients,
            &right_coefficients,
            Integer::from_double,
        )
    }

    #[cfg(feature = "integer-gmp")]
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
        const MAX_INPUT_LIMBS: usize = 128;
        const MAX_OUTPUT_LIMBS: usize = 1 << 26;
        const BLOCK_SIZE: usize = 32;

        if gmp::NUMB_BITS != 64 || gmp::NAIL_BITS != 0 {
            return None;
        }

        if !self.layout_is_valid() {
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
                        let rank = usize::try_from(prefix_rank)
                            .ok()?
                            .checked_add(usize::try_from(suffix_rank).ok()?)?;
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
        for (index, coefficient) in coefficients
            .chunks_exact_mut(limbs_per_coefficient)
            .enumerate()
        {
            let negative = coefficient.last().copied().unwrap() >> 63 != 0;
            if negative {
                let mut carry = true;
                for limb in coefficient.iter_mut() {
                    let (value, overflow) = (!*limb).overflowing_add(gmp::limb_t::from(carry));
                    *limb = value;
                    carry = overflow;
                }
                debug_assert!(!carry);
            }
            let significant_len = coefficient
                .iter()
                .rposition(|&limb| limb != 0)
                .map_or(0, |position| position + 1);
            if significant_len == 0 {
                continue;
            }
            let value = MultiPrecisionInteger::try_from_lsf_limbs(
                &coefficient[..significant_len],
                negative,
            )?;
            output.push((u32::try_from(index).ok()?, Integer::from(value)));
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
        #[cfg(feature = "integer-gmp")]
        {
            self.try_large_array()
        }
        #[cfg(feature = "integer-malachite")]
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

    #[cfg(feature = "integer-gmp")]
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
    fn try_chunked_dense_mul(
        &self,
        request: ChunkedDensePolynomialMulRequest<'_, Integer>,
    ) -> Option<Vec<(u32, Integer)>> {
        ChunkedDenseIntegerMul::new(request)?.run()
    }

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

    #[inline]
    fn preferred_total_degree_mul_workspace_ratio(
        &self,
        left_coefficients: &[Integer],
        right_coefficients: &[Integer],
        output_len: usize,
    ) -> Option<usize> {
        const FIXED_I128_ACCUMULATOR_WORKSPACE_RATIO: usize = 16;
        #[cfg(feature = "integer-gmp")]
        const LIMB_ACCUMULATOR_WORKSPACE_RATIO: usize = 8;

        if preferred_fixed_i128_dense_accumulator_is_bounded(left_coefficients, right_coefficients)
        {
            return Some(FIXED_I128_ACCUMULATOR_WORKSPACE_RATIO);
        }

        #[cfg(feature = "integer-gmp")]
        {
            total_degree_limb_workspace_is_bounded(
                left_coefficients,
                right_coefficients,
                output_len,
            )
            .then_some(LIMB_ACCUMULATOR_WORKSPACE_RATIO)
        }
        #[cfg(feature = "integer-malachite")]
        {
            let _ = (left_coefficients, right_coefficients, output_len);
            None
        }
    }
}

/// Return whether mixed-radix multiplication could accumulate these coefficients in `i128` cells.
fn preferred_fixed_i128_dense_accumulator_is_bounded(
    left_coefficients: &[Integer],
    right_coefficients: &[Integer],
) -> bool {
    let maximum_fixed = |coefficients: &[Integer]| {
        coefficients.iter().try_fold(0u128, |maximum, coefficient| {
            let value = match coefficient {
                Integer::Single(value) => i128::from(*value),
                Integer::Double(value) => value.get(),
                Integer::Large(_) => return None,
            };
            Some(maximum.max(value.unsigned_abs()))
        })
    };
    let Some(maximum_left) = maximum_fixed(left_coefficients) else {
        return false;
    };
    let Some(maximum_right) = maximum_fixed(right_coefficients) else {
        return false;
    };
    maximum_left
        .checked_mul(maximum_right)
        .and_then(|bound| {
            bound.checked_mul(left_coefficients.len().min(right_coefficients.len()) as u128)
        })
        .is_some_and(|bound| bound <= i128::MAX as u128)
}

#[cfg(feature = "integer-gmp")]
/// Return whether the compact limb kernel can represent these inputs and its complete output.
fn total_degree_limb_workspace_is_bounded(
    left_coefficients: &[Integer],
    right_coefficients: &[Integer],
    output_len: usize,
) -> bool {
    const MAX_INPUT_LIMBS: u64 = 128;
    const MAX_OUTPUT_LIMBS: usize = 1 << 26;

    if gmp::NUMB_BITS != 64 || gmp::NAIL_BITS != 0 {
        return false;
    }
    let maximum_bits = |coefficients: &[Integer]| {
        coefficients
            .iter()
            .map(Integer::significant_bits)
            .max()
            .unwrap_or(0)
    };
    let maximum_left_bits = maximum_bits(left_coefficients);
    let maximum_right_bits = maximum_bits(right_coefficients);
    if maximum_left_bits.div_ceil(64) > MAX_INPUT_LIMBS
        || maximum_right_bits.div_ceil(64) > MAX_INPUT_LIMBS
    {
        return false;
    }

    let products_per_coefficient = left_coefficients.len().min(right_coefficients.len());
    let accumulation_bits = usize::BITS - products_per_coefficient.leading_zeros();
    maximum_left_bits
        .checked_add(maximum_right_bits)
        .and_then(|bits| bits.checked_add(u64::from(accumulation_bits)))
        .and_then(|bits| bits.checked_add(1))
        .and_then(|bits| usize::try_from(bits.div_ceil(64)).ok())
        .map(|limbs_per_coefficient| limbs_per_coefficient.max(1))
        .and_then(|limbs_per_coefficient| output_len.checked_mul(limbs_per_coefficient))
        .is_some_and(|output_limb_count| output_limb_count <= MAX_OUTPUT_LIMBS)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "integer-gmp")]
    #[test]
    fn fixed_limb_absolute_statistics_match_integer_arithmetic() {
        let large = Integer::from(
            (MultiPrecisionInteger::from(1u32) << 200u32) + MultiPrecisionInteger::from(37u32),
        );
        let coefficients = vec![
            Integer::zero(),
            Integer::from(-17),
            Integer::from_double(i128::MIN),
            large.clone(),
            -large,
        ];
        let statistics = try_fixed_limb_absolute_bit_statistics(&coefficients).unwrap();
        let mut sum = MultiPrecisionInteger::default();
        let mut maximum_bits = 0;
        for coefficient in &coefficients {
            let mut magnitude = coefficient.clone().to_multi_prec();
            if magnitude.is_negative() {
                magnitude = -magnitude;
            }
            maximum_bits = maximum_bits.max(magnitude.significant_bits());
            sum += magnitude;
        }
        assert_eq!(statistics.l1_bits, sum.significant_bits());
        assert_eq!(statistics.maximum_bits, maximum_bits);
        assert!(statistics.has_negative);

        let positive = Integer::from(MultiPrecisionInteger::from(1u32) << 200u32);
        assert!(
            !try_fixed_limb_absolute_bit_statistics(&[positive])
                .unwrap()
                .has_negative
        );

        let too_wide = Integer::from(MultiPrecisionInteger::from(1u32) << 512u32);
        assert!(try_fixed_limb_absolute_bit_statistics(&[too_wide]).is_none());
    }

    #[cfg(feature = "integer-gmp")]
    #[test]
    fn kronecker_sign_scan_remains_complete_after_first_large_coefficient() {
        fn assert_matches_direct(mut left: Vec<Integer>) {
            const LENGTH: usize = 32;
            left[LENGTH - 1] = Integer::from(-1);
            let right = vec![Integer::one(); LENGTH];
            let indices = (0..LENGTH as u32).collect::<Vec<_>>();
            let output_len = LENGTH * 2 - 1;

            let actual = DenseIntegerMul::try_kronecker_for_test(
                output_len, &left, &indices, &right, &indices,
            )
            .unwrap();
            let mut expected = vec![Integer::zero(); output_len];
            for (left_index, left_coefficient) in left.iter().enumerate() {
                for right_index in 0..LENGTH {
                    expected[left_index + right_index] += left_coefficient;
                }
            }
            let mut actual_dense = vec![Integer::zero(); output_len];
            for (index, coefficient) in actual {
                actual_dense[index as usize] = coefficient;
            }
            assert_eq!(actual_dense, expected);
        }

        assert_matches_direct(vec![Integer::one(); 32]);

        let bounded_large = Integer::from(MultiPrecisionInteger::from(1u32) << 200u32);
        let mut bounded = vec![Integer::one(); 32];
        bounded[0] = bounded_large;
        assert_matches_direct(bounded);

        let too_wide = Integer::from(MultiPrecisionInteger::from(1u32) << 512u32);
        let mut fallback = vec![Integer::one(); 32];
        fallback[0] = too_wide;
        assert_matches_direct(fallback);
    }

    #[cfg(feature = "integer-gmp")]
    #[test]
    fn large_kronecker_encoder_applies_sign_and_borrow_in_native_limbs() {
        let magnitude =
            (MultiPrecisionInteger::from(1u32) << 180u32) + MultiPrecisionInteger::from(37u32);
        let values = [
            MultiPrecisionInteger::from(0u32),
            MultiPrecisionInteger::from(1u32),
            MultiPrecisionInteger::from(-1i32),
            magnitude.clone(),
            -magnitude,
        ];

        for coefficient in values {
            for negate in [false, true] {
                for borrow in [false, true] {
                    let mut digit = [0u64; 4];
                    let negative =
                        try_encode_large_kronecker_digit(&mut digit, &coefficient, negate, borrow)
                            .unwrap();
                    let mut actual = MultiPrecisionInteger::from_raw(
                        RawMultiPrecisionInteger::from_digits(&digit, RugIntegerOrder::Lsf),
                    );
                    if negative {
                        actual = -actual;
                    }

                    let mut expected = coefficient.clone();
                    if negate {
                        expected = -expected;
                    }
                    if borrow {
                        expected -= 1u32;
                    }
                    assert_eq!(actual, expected);
                }
            }
        }

        let overflow = -MultiPrecisionInteger::from(u64::MAX);
        assert!(try_encode_large_kronecker_digit(&mut [0u64; 1], &overflow, false, true).is_none());
    }

    #[cfg(feature = "integer-gmp")]
    #[test]
    fn small_kronecker_decoder_applies_carry_before_the_radix_sign() {
        const DIGIT_BITS: usize = 130;

        let sign_threshold_minus_one = [u64::MAX, u64::MAX, 1];
        let decoded = try_decode_small_kronecker_digit(
            &sign_threshold_minus_one,
            DIGIT_BITS,
            true,
            true,
            false,
        );
        match decoded {
            Some(DecodedKroneckerDigit::Large {
                magnitude_limbs,
                limb_count,
                negative,
                carry_out,
            }) => {
                assert_eq!(limb_count, 3);
                assert_eq!(&magnitude_limbs[..limb_count], &[0, 0, 2]);
                assert!(negative);
                assert!(carry_out);
            }
            _ => panic!("carry must cross the sign threshold before the digit is classified"),
        }

        let radix_minus_one = [u64::MAX, u64::MAX, 3];
        assert!(matches!(
            try_decode_small_kronecker_digit(&radix_minus_one, DIGIT_BITS, true, true, false,),
            Some(DecodedKroneckerDigit::Fixed {
                value: 0,
                carry_out: true,
            })
        ));
    }

    #[cfg(feature = "integer-gmp")]
    #[test]
    fn small_kronecker_decoder_classifies_both_signs_of_two_to_127() {
        const DIGIT_BITS: usize = 130;
        const HIGH_I128_BIT: u64 = 1u64 << 63;

        let positive = [0, HIGH_I128_BIT, 0];
        match try_decode_small_kronecker_digit(&positive, DIGIT_BITS, false, true, false) {
            Some(DecodedKroneckerDigit::Large {
                magnitude_limbs,
                limb_count,
                negative,
                carry_out,
            }) => {
                assert_eq!(limb_count, 2);
                assert_eq!(&magnitude_limbs[..limb_count], &[0, HIGH_I128_BIT]);
                assert!(!negative);
                assert!(!carry_out);
            }
            _ => panic!("positive 2^127 must use the large representation"),
        }
        assert!(matches!(
            try_decode_small_kronecker_digit(&positive, DIGIT_BITS, false, true, true,),
            Some(DecodedKroneckerDigit::Fixed {
                value: i128::MIN,
                carry_out: false,
            })
        ));

        let negative = [0, HIGH_I128_BIT, 3];
        assert!(matches!(
            try_decode_small_kronecker_digit(&negative, DIGIT_BITS, false, true, false,),
            Some(DecodedKroneckerDigit::Fixed {
                value: i128::MIN,
                carry_out: true,
            })
        ));
        match try_decode_small_kronecker_digit(&negative, DIGIT_BITS, false, true, true) {
            Some(DecodedKroneckerDigit::Large {
                magnitude_limbs,
                limb_count,
                negative,
                carry_out,
            }) => {
                assert_eq!(limb_count, 2);
                assert_eq!(&magnitude_limbs[..limb_count], &[0, HIGH_I128_BIT]);
                assert!(!negative);
                assert!(carry_out);
            }
            _ => panic!("negating i128::MIN must produce positive 2^127"),
        }
    }

    #[cfg(feature = "integer-gmp")]
    #[test]
    fn small_kronecker_decoder_matches_integer_arithmetic_at_limb_boundaries() {
        fn decoded_value(decoded: DecodedKroneckerDigit) -> (Integer, bool) {
            match decoded {
                DecodedKroneckerDigit::Fixed { value, carry_out } => {
                    (Integer::from_double(value), carry_out)
                }
                DecodedKroneckerDigit::Large {
                    magnitude_limbs,
                    limb_count,
                    negative,
                    carry_out,
                } => {
                    let raw = RawMultiPrecisionInteger::from_digits(
                        &magnitude_limbs[..limb_count],
                        RugIntegerOrder::Lsf,
                    );
                    let mut value = MultiPrecisionInteger::from_raw(raw);
                    if negative {
                        value = -value;
                    }
                    (Integer::from(value), carry_out)
                }
            }
        }

        for digit_bits in [
            64usize, 65, 127, 128, 129, 191, 192, 193, 255, 256, 257, 319, 320, 321, 383, 384, 385,
            447, 448, 449, 511, 512,
        ] {
            let radix = MultiPrecisionInteger::from(1u32) << digit_bits;
            let sign_threshold = MultiPrecisionInteger::from(1u32) << (digit_bits - 1);
            let mut sign_threshold_minus_one = sign_threshold.clone();
            sign_threshold_minus_one -= 1u32;
            let mut radix_minus_one = radix.clone();
            radix_minus_one -= 1u32;
            let values = [
                MultiPrecisionInteger::from(0u32),
                sign_threshold_minus_one,
                sign_threshold.clone(),
                radix_minus_one,
            ];

            for raw_value in values {
                let mut digit_limbs = vec![0u64; digit_bits.div_ceil(64)];
                let raw_limbs = raw_value.as_raw().as_limbs();
                digit_limbs[..raw_limbs.len()].copy_from_slice(raw_limbs);

                for signed_coefficients in [false, true] {
                    for carry_in in [false, true] {
                        let mut adjusted = raw_value.clone();
                        adjusted += u32::from(carry_in);
                        let overflow_radix = adjusted >= radix;
                        if overflow_radix && !signed_coefficients {
                            assert!(
                                try_decode_small_kronecker_digit(
                                    &digit_limbs,
                                    digit_bits,
                                    carry_in,
                                    signed_coefficients,
                                    false,
                                )
                                .is_none()
                            );
                            continue;
                        }

                        let carry_out =
                            signed_coefficients && (overflow_radix || adjusted >= sign_threshold);
                        if carry_out {
                            adjusted -= &radix;
                        }
                        for product_negative in [false, true] {
                            let mut expected = adjusted.clone();
                            if product_negative {
                                expected = -expected;
                            }
                            let decoded = try_decode_small_kronecker_digit(
                                &digit_limbs,
                                digit_bits,
                                carry_in,
                                signed_coefficients,
                                product_negative,
                            )
                            .expect("at most eight limbs must use the bounded decoder");
                            let (actual, actual_carry_out) = decoded_value(decoded);
                            assert_eq!(actual, Integer::from(expected));
                            assert_eq!(actual_carry_out, carry_out);
                        }
                    }
                }
            }
        }

        assert!(try_decode_small_kronecker_digit(&[0; 9], 513, false, true, false).is_none());
    }

    #[test]
    fn preferred_fixed_i128_accumulator_checks_width_and_collisions() {
        assert!(preferred_fixed_i128_dense_accumulator_is_bounded(
            &[Integer::from(i128::MAX)],
            &[Integer::from(1)],
        ));
        assert!(!preferred_fixed_i128_dense_accumulator_is_bounded(
            &[Integer::from(i128::MAX)],
            &[Integer::from(2)],
        ));
        let collision_heavy = vec![Integer::from(i64::MAX); 3];
        assert!(!preferred_fixed_i128_dense_accumulator_is_bounded(
            &collision_heavy,
            &collision_heavy,
        ));
    }

    #[cfg(feature = "integer-gmp")]
    #[test]
    fn total_degree_limb_workspace_accepts_wide_bounded_coefficients() {
        let left = vec![
            Integer::from(MultiPrecisionInteger::from(1u32) << 4096u32),
            Integer::from(-(MultiPrecisionInteger::from(1u32) << 4095u32)),
        ];
        let right = vec![
            Integer::from((MultiPrecisionInteger::from(1u32) << 4094u32) + 3u32),
            Integer::from(-17),
        ];

        assert!(total_degree_limb_workspace_is_bounded(&left, &right, 1_287,));
        assert!(!total_degree_limb_workspace_is_bounded(
            &[Integer::from(MultiPrecisionInteger::from(1u32) << 8192u32,)],
            &right,
            1_287,
        ));
    }
}

#[cfg(test)]
pub(super) fn try_chunked_dense_mul_for_test(
    request: ChunkedDensePolynomialMulRequest<'_, Integer>,
) -> Option<Vec<(u32, Integer)>> {
    ChunkedDenseIntegerMul::new(request)?.run()
}
