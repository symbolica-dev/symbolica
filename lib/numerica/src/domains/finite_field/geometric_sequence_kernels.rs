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
