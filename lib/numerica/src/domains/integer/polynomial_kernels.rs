//! Representation-specific bulk kernels for integer polynomial arithmetic.

use super::{Integer, IntegerRing};
#[cfg(feature = "gmp")]
use super::{MultiPrecisionInteger, RawMultiPrecisionInteger};
#[cfg(feature = "gmp")]
use crate::domains::polynomial_layouts::try_simplex_kronecker_layout;
use crate::domains::{
    DensePolynomialExactDivisionRequest, DensePolynomialMulRequest, PolynomialKernels,
    TotalDegreePolynomialMulRequest,
};
#[cfg(feature = "gmp")]
use gmp_mpfr_sys::gmp;
#[cfg(feature = "gmp")]
use rug::integer::Order as RugIntegerOrder;
#[cfg(feature = "gmp")]
use smallvec::SmallVec;

fn try_dense_i64_polynomial_mul(
    output_len: usize,
    left_coefficients: &[Integer],
    left_indices: &[u32],
    right_coefficients: &[Integer],
    right_indices: &[u32],
) -> Option<Vec<(u32, Integer)>> {
    let left = left_coefficients
        .iter()
        .map(|coefficient| match coefficient {
            Integer::Single(value) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let right = right_coefficients
        .iter()
        .map(|coefficient| match coefficient {
            Integer::Single(value) => Some(*value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;

    let max_left = left.iter().map(|value| value.unsigned_abs()).max()?;
    let max_right = right.iter().map(|value| value.unsigned_abs()).max()?;
    let coefficient_bound = u128::from(max_left)
        .checked_mul(u128::from(max_right))?
        .checked_mul(left.len().min(right.len()) as u128)?;
    if coefficient_bound > i64::MAX as u128 {
        return None;
    }

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
                    let index = left_index + unsafe { *right_indices.get_unchecked(j) as usize };
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

fn try_dense_i64_i128_polynomial_mul(
    output_len: usize,
    left_coefficients: &[Integer],
    left_indices: &[u32],
    right_coefficients: &[Integer],
    right_indices: &[u32],
) -> Option<Vec<(u32, Integer)>> {
    let left = left_coefficients
        .iter()
        .map(|coefficient| match coefficient {
            Integer::Single(value) => Some(*value),
            Integer::Double(_) | Integer::Large(_) => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let right = right_coefficients
        .iter()
        .map(|coefficient| match coefficient {
            Integer::Single(value) => Some(*value),
            Integer::Double(_) | Integer::Large(_) => None,
        })
        .collect::<Option<Vec<_>>>()?;

    let max_left = left.iter().map(|value| value.unsigned_abs()).max()?;
    let max_right = right.iter().map(|value| value.unsigned_abs()).max()?;
    let coefficient_bound = u128::from(max_left)
        .checked_mul(u128::from(max_right))?
        .checked_mul(left.len().min(right.len()) as u128)?;
    if coefficient_bound > i128::MAX as u128 {
        return None;
    }

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
                    let index = left_index + unsafe { *right_indices.get_unchecked(j) as usize };
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

fn try_dense_i128_polynomial_mul(
    output_len: usize,
    left_coefficients: &[Integer],
    left_indices: &[u32],
    right_coefficients: &[Integer],
    right_indices: &[u32],
) -> Option<Vec<(u32, Integer)>> {
    let left = left_coefficients
        .iter()
        .map(|coefficient| match coefficient {
            Integer::Single(value) => Some(i128::from(*value)),
            Integer::Double(value) => Some(value.get()),
            Integer::Large(_) => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let right = right_coefficients
        .iter()
        .map(|coefficient| match coefficient {
            Integer::Single(value) => Some(i128::from(*value)),
            Integer::Double(value) => Some(value.get()),
            Integer::Large(_) => None,
        })
        .collect::<Option<Vec<_>>>()?;

    let max_left = left.iter().map(|value| value.unsigned_abs()).max()?;
    let max_right = right.iter().map(|value| value.unsigned_abs()).max()?;
    let coefficient_bound = max_left
        .checked_mul(max_right)?
        .checked_mul(left.len().min(right.len()) as u128)?;
    if coefficient_bound > i128::MAX as u128 {
        return None;
    }

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
                    let index = left_index + unsafe { *right_indices.get_unchecked(j) as usize };
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
pub(super) fn try_kronecker_polynomial_mul(
    output_len: usize,
    left_coefficients: &[Integer],
    left_indices: &[u32],
    right_coefficients: &[Integer],
    right_indices: &[u32],
) -> Option<Vec<(u32, Integer)>> {
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
    if product_count < 64 || packed_output_len.saturating_mul(128) >= product_count {
        return None;
    }

    fn absolute_statistics(
        coefficients: &[Integer],
    ) -> (MultiPrecisionInteger, MultiPrecisionInteger) {
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
    let left_packed_bits =
        (packed_left_indices.iter().copied().max()? as usize + 1).checked_mul(digit_bits_usize)?;
    let right_packed_bits =
        (packed_right_indices.iter().copied().max()? as usize + 1).checked_mul(digit_bits_usize)?;
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

        for position in 0..coefficients.len() {
            let term = term_index(position);
            let coefficient = unsafe { coefficients.get_unchecked(term) };
            let index = unsafe { *indices.get_unchecked(term) } as usize;
            debug_assert!(index >= next_index);
            if borrow {
                fill_ones(&mut limbs, next_index * digit_bits, index * digit_bits);
            }

            let mut value = coefficient.clone().to_multi_prec();
            if leading_negative {
                value = -value;
            }
            if borrow {
                value -= 1i64;
            }

            borrow = value.is_negative();
            if borrow {
                value = -value;
            }

            let mut digit = SmallVec::<[u64; 4]>::new();
            digit.resize(digit_bits.div_ceil(64), 0);
            let value_limbs = value.as_raw().as_limbs();
            debug_assert!(value_limbs.len() <= digit.len());
            digit[..value_limbs.len()].copy_from_slice(value_limbs);

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

        let mut packed = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from_digits(
            &limbs,
            RugIntegerOrder::Lsf,
        ));
        if leading_negative {
            packed = -packed;
        }
        Some(packed)
    }

    let left = pack(left_coefficients, packed_left_indices, digit_bits_usize)?;
    let right = pack(right_coefficients, packed_right_indices, digit_bits_usize)?;
    let product = MultiPrecisionInteger::from_raw(RawMultiPrecisionInteger::from(
        left.as_raw() * right.as_raw(),
    ));

    let radix = MultiPrecisionInteger::from(1u32) << digit_bits_usize;
    let product_negative = product.is_negative();
    let limbs = product.as_raw().as_limbs();
    let limbs_per_digit = digit_bits_usize.div_ceil(64);
    let mut carry = false;
    let mut output = Vec::with_capacity(packed_output_len);
    for index in 0..packed_output_len {
        let bit_index = index.checked_mul(digit_bits_usize)?;
        let limb_index = bit_index / 64;
        let shift = bit_index % 64;
        let mut digit_limbs = SmallVec::<[u64; 4]>::new();
        digit_limbs.resize(limbs_per_digit, 0);
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
        let raw_digit = RawMultiPrecisionInteger::from_digits(&digit_limbs, RugIntegerOrder::Lsf);
        let mut digit = MultiPrecisionInteger::from_raw(raw_digit);
        if carry {
            digit += 1i64;
        }

        carry = signed_coefficients
            && (digit.significant_bits() > digit_bits
                || digit.as_raw().get_bit(digit_bits_u32 - 1));
        if carry {
            digit -= &radix;
        }
        if product_negative {
            digit = -digit;
        }
        if !digit.is_zero() {
            let output_index = if let Some(layout) = simplex_layout.as_ref() {
                let decoded = *layout.decode_indices.get(index)?;
                debug_assert_ne!(decoded, u32::MAX);
                decoded
            } else {
                index as u32
            };
            output.push((output_index, Integer::from(digit)));
        }
    }
    debug_assert!(!carry);

    if simplex_layout.is_some() {
        output.sort_unstable_by_key(|term| term.0);
    }

    Some(output)
}

#[cfg(feature = "gmp")]
fn try_large_array_polynomial_mul(
    output_len: usize,
    left_coefficients: &[Integer],
    left_indices: &[u32],
    right_coefficients: &[Integer],
    right_indices: &[u32],
) -> Option<Vec<(u32, Integer)>> {
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
    if product_count < 64 || output_len > 1 << 20 || output_len > product_count.saturating_mul(10) {
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
                    let index = left_index + unsafe { *right_indices.get_unchecked(j) as usize };
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

#[cfg(feature = "gmp")]
fn try_total_degree_limb_polynomial_mul(
    request: TotalDegreePolynomialMulRequest<'_, Integer>,
) -> Option<Vec<(u32, Integer)>> {
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
    } = request;
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
            for left_index in left_block..(left_block + BLOCK_SIZE).min(left_coefficients.len()) {
                let left_range = unsafe { *left_ranges.get_unchecked(left_index) };
                let left = unsafe {
                    left_limbs.get_unchecked(
                        left_range.offset as usize
                            ..left_range.offset as usize + left_range.length as usize,
                    )
                };
                let (left_prefix, left_suffix) = unsafe { *left_codes.get_unchecked(left_index) };
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
                        *suffix_rank
                            .get_unchecked(remaining_degree as usize * suffix_code_count + suffix)
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

#[cfg(feature = "gmp")]
fn try_large_array_polynomial_exact_division(
    total: usize,
    dividend_coefficients: &mut [Integer],
    dividend_indices: &[u32],
    divisor_coefficients: &[Integer],
    divisor_indices: &[u32],
) -> Option<Vec<(u32, Integer)>> {
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
        workspace[index as usize] = std::mem::replace(coefficient, Integer::zero()).to_multi_prec();
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
            Divisors::Borrowed(divisors) => coefficient.div_exact_owned(divisors.last().unwrap()),
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

impl PolynomialKernels<Integer> for IntegerRing {
    #[inline]
    fn try_total_degree_mul(
        &self,
        request: TotalDegreePolynomialMulRequest<'_, Integer>,
    ) -> Option<Vec<(u32, Integer)>> {
        #[cfg(feature = "gmp")]
        {
            try_total_degree_limb_polynomial_mul(request)
        }
        #[cfg(feature = "no_gmp")]
        {
            let _ = request;
            None
        }
    }

    #[inline]
    fn try_dense_mul(
        &self,
        request: DensePolynomialMulRequest<'_, Integer>,
    ) -> Option<Vec<(u32, Integer)>> {
        let DensePolynomialMulRequest {
            output_len,
            left_coefficients,
            left_indices,
            right_coefficients,
            right_indices,
        } = request;
        if left_coefficients.len() != left_indices.len()
            || right_coefficients.len() != right_indices.len()
        {
            return None;
        }
        let (Some(left_max), Some(right_max)) = (
            left_indices.iter().copied().max(),
            right_indices.iter().copied().max(),
        ) else {
            return Some(Vec::new());
        };
        if left_max as usize + right_max as usize >= output_len {
            return None;
        }

        if let Some(output) = try_dense_i64_polynomial_mul(
            output_len,
            left_coefficients,
            left_indices,
            right_coefficients,
            right_indices,
        ) {
            return Some(output);
        }

        if let Some(output) = try_dense_i64_i128_polynomial_mul(
            output_len,
            left_coefficients,
            left_indices,
            right_coefficients,
            right_indices,
        ) {
            return Some(output);
        }

        if let Some(output) = try_dense_i128_polynomial_mul(
            output_len,
            left_coefficients,
            left_indices,
            right_coefficients,
            right_indices,
        ) {
            return Some(output);
        }

        #[cfg(feature = "gmp")]
        {
            if let Some(output) = try_kronecker_polynomial_mul(
                output_len,
                left_coefficients,
                left_indices,
                right_coefficients,
                right_indices,
            ) {
                return Some(output);
            }

            try_large_array_polynomial_mul(
                output_len,
                left_coefficients,
                left_indices,
                right_coefficients,
                right_indices,
            )
        }
        #[cfg(feature = "no_gmp")]
        {
            None
        }
    }

    #[inline]
    fn try_dense_exact_division(
        &self,
        request: DensePolynomialExactDivisionRequest<'_, Integer>,
    ) -> Option<Vec<(u32, Integer)>> {
        let DensePolynomialExactDivisionRequest {
            total,
            dividend_coefficients,
            dividend_indices,
            divisor_coefficients,
            divisor_indices,
        } = request;
        #[cfg(feature = "gmp")]
        {
            try_large_array_polynomial_exact_division(
                total,
                dividend_coefficients,
                dividend_indices,
                divisor_coefficients,
                divisor_indices,
            )
        }
        #[cfg(feature = "no_gmp")]
        {
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
}
