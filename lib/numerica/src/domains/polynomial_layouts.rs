//! Coefficient-domain-independent dense polynomial layouts.

pub(super) struct SimplexKroneckerLayout {
    pub(super) left_indices: Vec<u32>,
    pub(super) right_indices: Vec<u32>,
    pub(super) output_len: usize,
    pub(super) decode_indices: Vec<u32>,
}

/// Find a shorter additive Kronecker map for a dense three-variable total-degree simplex.
///
/// A mixed-radix map for total degree `h` uses weights `1, h + 1, (h + 1)^2` and therefore
/// spans `(h + 1)^3` positions. For odd `h`, the weights
///
/// `1, h(h + 1)/2 + 1, (h + 1)(h + 2)/2 + 1`
///
/// are often collision-free on the degree-`h` simplex and have roughly half the span. We build
/// the inverse table and explicitly reject the map if a collision is found, so correctness does
/// not depend on the formula being collision-free for degrees outside the useful cases.
pub(super) fn try_simplex_kronecker_layout(
    output_len: usize,
    left_indices: &[u32],
    right_indices: &[u32],
) -> Option<SimplexKroneckerLayout> {
    let radix = (2..=256usize).find(|&radix| radix.checked_pow(3) == Some(output_len))?;
    let total_degree = radix - 1;
    if total_degree < 3 || total_degree % 2 == 0 {
        return None;
    }

    let middle_weight = total_degree
        .checked_mul(total_degree + 1)?
        .checked_div(2)?
        .checked_add(1)?;
    let high_weight = (total_degree + 1)
        .checked_mul(total_degree + 2)?
        .checked_div(2)?
        .checked_add(1)?;
    let maximum_code = total_degree.checked_mul(high_weight)?;
    if maximum_code + 1 >= output_len {
        return None;
    }

    let radix_squared = radix.checked_mul(radix)?;
    let decode = |index: u32| {
        let index = index as usize;
        let high = index / radix_squared;
        let remainder = index % radix_squared;
        let middle = remainder / radix;
        let low = remainder % radix;
        (high, middle, low)
    };
    let input_total_degree = |indices: &[u32]| {
        indices
            .iter()
            .map(|&index| {
                let (high, middle, low) = decode(index);
                high.checked_add(middle)?.checked_add(low)
            })
            .collect::<Option<Vec<_>>>()
            .and_then(|degrees| degrees.into_iter().max())
    };
    if input_total_degree(left_indices)?.checked_add(input_total_degree(right_indices)?)?
        != total_degree
    {
        return None;
    }

    let mut decode_indices = vec![u32::MAX; maximum_code + 1];
    for high in 0..=total_degree {
        for middle in 0..=total_degree - high {
            for low in 0..=total_degree - high - middle {
                let code = high
                    .checked_mul(high_weight)?
                    .checked_add(middle.checked_mul(middle_weight)?)?
                    .checked_add(low)?;
                let standard_index = high
                    .checked_mul(radix_squared)?
                    .checked_add(middle.checked_mul(radix)?)?
                    .checked_add(low)?;
                if decode_indices[code] != u32::MAX {
                    return None;
                }
                decode_indices[code] = u32::try_from(standard_index).ok()?;
            }
        }
    }

    let remap = |indices: &[u32]| {
        indices
            .iter()
            .map(|&index| {
                let (high, middle, low) = decode(index);
                high.checked_mul(high_weight)?
                    .checked_add(middle.checked_mul(middle_weight)?)?
                    .checked_add(low)
                    .and_then(|code| u32::try_from(code).ok())
            })
            .collect::<Option<Vec<_>>>()
    };

    Some(SimplexKroneckerLayout {
        left_indices: remap(left_indices)?,
        right_indices: remap(right_indices)?,
        output_len: maximum_code + 1,
        decode_indices,
    })
}
