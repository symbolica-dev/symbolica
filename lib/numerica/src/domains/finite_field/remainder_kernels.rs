//! In-place dense univariate remainders over fixed-width prime fields.

use super::montgomery::{montgomery_reduce_u32, montgomery_reduce_u32x4};
use super::{FiniteFieldElement, Zp};
use crate::domains::Field;

/// The caller must check AVX2 support, a nonzero leading divisor coefficient,
/// and that the dividend is at least as long as the nonconstant divisor.
#[target_feature(enable = "avx2")]
pub(super) unsafe fn remainder_u32_avx2(
    field: &Zp,
    dividend: &mut Vec<FiniteFieldElement<u32>>,
    divisor: &[FiniteFieldElement<u32>],
) {
    use std::arch::x86_64::*;

    let divisor_degree = divisor.len() - 1;
    let leading_inverse = if divisor[divisor_degree] == field.one {
        field.one.0
    } else {
        field.inv(&divisor[divisor_degree]).0
    };
    let prime = _mm256_set1_epi64x(field.p as i64);
    let (divisor_chunks, divisor_tail) = divisor[..divisor_degree].as_chunks::<4>();
    for degree in (divisor_degree..dividend.len()).rev() {
        let leading = dividend[degree].0;
        if leading == 0 {
            continue;
        }
        let quotient = montgomery_reduce_u32(field, leading as u64 * leading_inverse as u64);
        let quotient_vector = _mm256_set1_epi64x(quotient as i64);
        let shift = degree - divisor_degree;
        let (current_chunks, current_tail) = dividend[shift..degree].as_chunks_mut::<4>();
        // Explicit indexing avoids out-of-line iterator adapters at the target-feature boundary.
        let mut index = 0;
        while index < current_chunks.len() {
            let current = &mut current_chunks[index];
            let divisor = &divisor_chunks[index];
            // Copy fields explicitly rather than assuming FiniteFieldElement's layout.
            let current_values = [current[0].0, current[1].0, current[2].0, current[3].0];
            let divisor_values = [divisor[0].0, divisor[1].0, divisor[2].0, divisor[3].0];
            // Each unaligned load reads four initialized u32 values.
            let values =
                _mm256_cvtepu32_epi64(unsafe { _mm_loadu_si128(current_values.as_ptr().cast()) });
            let coefficients =
                _mm256_cvtepu32_epi64(unsafe { _mm_loadu_si128(divisor_values.as_ptr().cast()) });
            let product = unsafe {
                montgomery_reduce_u32x4(field, _mm256_mul_epu32(coefficients, quotient_vector))
            };
            let difference = _mm256_sub_epi64(values, product);
            let borrow = _mm256_cmpgt_epi64(product, values);
            let result = _mm256_add_epi64(difference, _mm256_and_si256(borrow, prime));
            let shuffled = _mm256_shuffle_epi32::<0b10_00_10_00>(result);
            let packed = _mm_unpacklo_epi64(
                _mm256_castsi256_si128(shuffled),
                _mm256_extracti128_si256::<1>(shuffled),
            );
            let mut output = [0u32; 4];
            // The store writes exactly the four allocated u32 values.
            unsafe { _mm_storeu_si128(output.as_mut_ptr().cast(), packed) };
            current[0].0 = output[0];
            current[1].0 = output[1];
            current[2].0 = output[2];
            current[3].0 = output[3];
            index += 1;
        }
        let mut index = 0;
        while index < current_tail.len() {
            let product =
                montgomery_reduce_u32(field, divisor_tail[index].0 as u64 * quotient as u64);
            let current = current_tail[index].0;
            current_tail[index].0 = if current >= product {
                current - product
            } else {
                field.p - (product - current)
            };
            index += 1;
        }
    }
    // Leading slots need not be zeroed during division: later rows never read them.
    dividend.truncate(divisor_degree);
    while dividend
        .last()
        .is_some_and(|coefficient| coefficient.0 == 0)
    {
        dividend.pop();
    }
}
