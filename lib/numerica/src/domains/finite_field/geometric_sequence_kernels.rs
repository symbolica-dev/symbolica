//! Bulk geometric-sequence evaluation over fixed-width prime fields.

use super::{FiniteFieldElement, Zp, Zp64};
use crate::kernels::{GeometricSequenceKernels, GeometricSequenceStepRequest};

use super::montgomery::{montgomery_reduce_u32, montgomery_reduce_u64};

impl GeometricSequenceKernels<FiniteFieldElement<u32>> for Zp {
    #[inline]
    fn try_sum_and_advance_geometric_sequences(
        &self,
        request: GeometricSequenceStepRequest<'_, FiniteFieldElement<u32>>,
    ) -> Option<FiniteFieldElement<u32>> {
        let GeometricSequenceStepRequest { current, ratios } = request;
        if current.len() != ratios.len()
            || (current.len() as u128) * (self.p.saturating_sub(1) as u128) > u64::MAX as u128
        {
            return None;
        }

        #[cfg(target_arch = "x86_64")]
        if current.len() >= 16 && std::is_x86_feature_detected!("avx2") {
            // The shape and sum bound were checked before any coefficients are mutated.
            return Some(unsafe { sum_and_advance_u32_avx2(self, current, ratios) });
        }

        // Montgomery residues are additive modulo p, so the raw values can be summed exactly and
        // reduced once after the loop. Each product needs one Montgomery reduction to advance its
        // sequence while preserving the representation.
        let mut sum = 0u64;
        for (current, ratio) in current.iter_mut().zip(ratios) {
            sum += current.0 as u64;
            current.0 = montgomery_reduce_u32(self, current.0 as u64 * ratio.0 as u64);
        }

        Some(FiniteFieldElement((sum % self.p as u64) as u32))
    }
}

/// Advance four Montgomery residues per vector while accumulating their old values.
///
/// The caller must check AVX2 support, equal slice lengths, and that the sum fits u64.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_and_advance_u32_avx2(
    field: &Zp,
    current: &mut [FiniteFieldElement<u32>],
    ratios: &[FiniteFieldElement<u32>],
) -> FiniteFieldElement<u32> {
    use std::arch::x86_64::*;

    let mut accumulated = _mm256_setzero_si256();
    let vector_len = current.len() / 4 * 4;
    // Keep the hot loop explicit: iterator adapters can remain out-of-line
    // across the target-feature boundary in builds without LTO.
    let mut index = 0;
    while index < vector_len {
        // Explicit arrays avoid assuming a layout for FiniteFieldElement.
        let values = [
            current[index].0,
            current[index + 1].0,
            current[index + 2].0,
            current[index + 3].0,
        ];
        let ratios = [
            ratios[index].0,
            ratios[index + 1].0,
            ratios[index + 2].0,
            ratios[index + 3].0,
        ];
        // Both unaligned loads read exactly four initialized u32 values.
        let values = _mm256_cvtepu32_epi64(unsafe { _mm_loadu_si128(values.as_ptr().cast()) });
        let ratios = _mm256_cvtepu32_epi64(unsafe { _mm_loadu_si128(ratios.as_ptr().cast()) });
        accumulated = _mm256_add_epi64(accumulated, values);

        let product = _mm256_mul_epu32(values, ratios);
        let reduced = unsafe { super::montgomery::montgomery_reduce_u32x4(field, product) };
        let shuffled = _mm256_shuffle_epi32::<0b10_00_10_00>(reduced);
        let packed = _mm_unpacklo_epi64(
            _mm256_castsi256_si128(shuffled),
            _mm256_extracti128_si256::<1>(shuffled),
        );
        let mut output = [0u32; 4];
        // The store writes exactly the four allocated u32 values.
        unsafe { _mm_storeu_si128(output.as_mut_ptr().cast(), packed) };
        current[index].0 = output[0];
        current[index + 1].0 = output[1];
        current[index + 2].0 = output[2];
        current[index + 3].0 = output[3];
        index += 4;
    }
    let mut partial_sums = [0u64; 4];
    // The store writes exactly the four allocated u64 values.
    unsafe { _mm256_storeu_si256(partial_sums.as_mut_ptr().cast(), accumulated) };
    let mut sum = partial_sums[0] + partial_sums[1] + partial_sums[2] + partial_sums[3];
    while index < current.len() {
        sum += current[index].0 as u64;
        current[index].0 =
            montgomery_reduce_u32(field, current[index].0 as u64 * ratios[index].0 as u64);
        index += 1;
    }
    FiniteFieldElement((sum % field.p as u64) as u32)
}

impl GeometricSequenceKernels<FiniteFieldElement<u64>> for Zp64 {
    #[inline]
    fn try_sum_and_advance_geometric_sequences(
        &self,
        request: GeometricSequenceStepRequest<'_, FiniteFieldElement<u64>>,
    ) -> Option<FiniteFieldElement<u64>> {
        let GeometricSequenceStepRequest { current, ratios } = request;
        if current.len() != ratios.len()
            || (current.len() as u128)
                .checked_mul(self.p.saturating_sub(1) as u128)
                .is_none()
        {
            return None;
        }

        // Sum before mutating so every term in the returned value belongs to the same step.
        let mut sum = 0u128;
        for current in current.iter() {
            sum += current.0 as u128;
        }
        for (current, ratio) in current.iter_mut().zip(ratios) {
            current.0 = montgomery_reduce_u64(self, current.0 as u128 * ratio.0 as u128);
        }

        Some(FiniteFieldElement((sum % self.p as u128) as u64))
    }
}
