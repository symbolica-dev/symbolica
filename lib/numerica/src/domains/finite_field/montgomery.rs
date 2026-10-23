//! Fixed-width Montgomery reductions shared by finite-field bulk kernels.

use super::{Zp, Zp64};

/// Reduce four products of canonical u32 Montgomery residues in u64 lanes.
/// The result contains canonical u32 residues, zero-extended to u64 lanes.
/// The caller must check AVX2 support.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
pub(super) unsafe fn montgomery_reduce_u32x4(
    field: &Zp,
    product: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;

    let prime = _mm256_set1_epi64x(field.p as i64);
    let multiplier = _mm256_set1_epi64x(field.m as i64);
    let sign = _mm256_set1_epi64x(i64::MIN);
    let factor = _mm256_mul_epu32(product, multiplier);
    let correction = _mm256_mul_epu32(factor, prime);
    let sum = _mm256_add_epi64(product, correction);
    // Flip the sign bit to compare unsigned 64-bit sums.
    let overflow = _mm256_cmpgt_epi64(_mm256_xor_si256(product, sign), _mm256_xor_si256(sum, sign));
    let quotient = _mm256_srli_epi64::<32>(sum);
    let subtract = _mm256_or_si256(
        overflow,
        _mm256_cmpgt_epi64(quotient, _mm256_set1_epi64x(field.p as i64 - 1)),
    );
    let reduced = _mm256_sub_epi64(quotient, _mm256_and_si256(subtract, prime));
    // Narrowing implements wrapping u32 subtraction when the sum overflowed.
    _mm256_and_si256(reduced, _mm256_set1_epi64x(u32::MAX as i64))
}

/// Apply one 32-bit Montgomery reduction to an exact `u64` value.
///
/// Bulk accumulators call this once after all products for a coefficient have been summed.
#[inline(always)]
pub(super) fn montgomery_reduce_u32(field: &Zp, value: u64) -> u32 {
    let m = (value as u32).wrapping_mul(field.m);
    let (sum, overflow) = value.overflowing_add(m as u64 * field.p as u64);
    let reduced = (sum >> 32) as u32;
    if overflow {
        reduced.wrapping_sub(field.p)
    } else if reduced >= field.p {
        reduced - field.p
    } else {
        reduced
    }
}

/// Apply one 64-bit Montgomery reduction to an exact `u128` value.
#[inline(always)]
pub(super) fn montgomery_reduce_u64(field: &Zp64, value: u128) -> u64 {
    let m = (value as u64).wrapping_mul(field.m);
    let (sum, overflow) = value.overflowing_add(m as u128 * field.p as u128);
    let reduced = (sum >> 64) as u64;
    if overflow {
        reduced.wrapping_sub(field.p)
    } else if reduced >= field.p {
        reduced - field.p
    } else {
        reduced
    }
}
