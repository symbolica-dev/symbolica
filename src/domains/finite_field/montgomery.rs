//! Fixed-width Montgomery reductions shared by finite-field bulk kernels.

use super::{Zp, Zp64};

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
