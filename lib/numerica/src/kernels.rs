//! Optional coefficient-domain kernels for bulk polynomial and interpolation operations.

/// A dense-indexed polynomial multiplication request.
///
/// `left_indices[i]` and `right_indices[i]` are the additive dense indices of the corresponding
/// coefficients. Every possible sum of one left and one right index must be smaller than
/// `output_len`.
pub struct DensePolynomialMulRequest<'a, E> {
    pub output_len: usize,
    pub left_coefficients: &'a [E],
    pub left_indices: &'a [u32],
    pub right_coefficients: &'a [E],
    pub right_indices: &'a [u32],
}

/// A dense-indexed multiplication split into cache-sized inner coefficient chunks.
///
/// `dense` supplies the additive mixed-radix indices. `inner_len` is positive, divides
/// `dense.output_len`, and decomposes every index as `outer * inner_len + inner`. The sum of any
/// left and right inner index is smaller than `inner_len`, so a kernel can accumulate one outer
/// coefficient chunk at a time and reuse the same inner workspace.
pub struct ChunkedDensePolynomialMulRequest<'a, E> {
    pub dense: DensePolynomialMulRequest<'a, E>,
    pub inner_len: usize,
}

/// A multiplication request for polynomials supported on total-degree simplices.
///
/// Each exponent vector is split into additive prefix and suffix codes. `prefix_rank`,
/// `prefix_remaining`, and `suffix_rank` map sums of those codes to compact lexicographic simplex
/// ranks. Each coefficient slice has the same length as its code slice, every code describes one
/// distinct polynomial term, and every pair of left and right codes maps to a rank below
/// `output_len`.
pub struct TotalDegreePolynomialMulRequest<'a, E> {
    pub output_len: usize,
    pub left_coefficients: &'a [E],
    pub left_codes: &'a [(usize, usize)],
    pub right_coefficients: &'a [E],
    pub right_codes: &'a [(usize, usize)],
    pub prefix_rank: &'a [u32],
    pub prefix_remaining: &'a [u8],
    pub suffix_rank: &'a [u32],
    pub suffix_code_count: usize,
}

/// An exact dense-indexed polynomial division request.
///
/// Divisibility is guaranteed by the caller. A kernel may consume dividend coefficients after it
/// decides to handle the request, but must leave them unchanged when returning `None`.
pub struct DensePolynomialExactDivisionRequest<'a, E> {
    pub total: usize,
    pub dividend_coefficients: &'a mut [E],
    pub dividend_indices: &'a [u32],
    pub divisor_coefficients: &'a [E],
    pub divisor_indices: &'a [u32],
}

/// One step over a collection of coefficient-weighted geometric sequences.
///
/// Before the step, `current[i]` contains `c_i * ratios[i]^k` for some common step `k`. A
/// successful kernel returns the sum of those current values and replaces each one by
/// `current[i] * ratios[i]`, preparing the state for step `k + 1`.
pub struct GeometricSequenceStepRequest<'a, E> {
    pub current: &'a mut [E],
    pub ratios: &'a [E],
}

/// Optional coefficient-domain kernels for evaluating geometric sequences during sparse
/// evaluation and interpolation.
pub trait GeometricSequenceKernels<E> {
    /// Sum the current values of several geometric sequences and advance all sequences one step.
    ///
    /// Returns `None` when the request is invalid or the coefficient representation cannot perform
    /// this step with its specialized bounded accumulator. The caller must then use generic ring
    /// operations, starting from the unchanged `request.current` values.
    fn try_sum_and_advance_geometric_sequences(
        &self,
        _request: GeometricSequenceStepRequest<'_, E>,
    ) -> Option<E> {
        None
    }
}

/// Optional coefficient-domain kernels for polynomial multiplication and exact division.
///
/// A ring returns this capability through [`crate::domains::Ring::kernels`]. Returning `None` from
/// an operation requests the generic polynomial implementation.
pub trait PolynomialKernels<E> {
    /// Multiply coefficients whose exponents have already been mapped to additive dense indices.
    ///
    /// On success, the result contains only nonzero `(dense_index, coefficient)` pairs in strictly
    /// increasing index order. The input coefficient/index slices have equal lengths, their
    /// indices are strictly increasing, and every possible summed index fits in
    /// `request.output_len`.
    fn try_dense_mul(&self, _request: DensePolynomialMulRequest<'_, E>) -> Option<Vec<(u32, E)>> {
        None
    }

    /// Multiply dense-indexed coefficients one outer chunk at a time.
    ///
    /// On success, the result has the same sorted sparse representation as
    /// [`Self::try_dense_mul`].
    fn try_chunked_dense_mul(
        &self,
        _request: ChunkedDensePolynomialMulRequest<'_, E>,
    ) -> Option<Vec<(u32, E)>> {
        None
    }

    /// Multiply coefficients on total-degree simplices.
    ///
    /// Compact simplex ranks are not additive. The request's code and rank tables map each product
    /// directly to its compact output rank. On success, the result contains only nonzero
    /// `(rank, coefficient)` pairs in strictly increasing rank order.
    fn try_total_degree_mul(
        &self,
        _request: TotalDegreePolynomialMulRequest<'_, E>,
    ) -> Option<Vec<(u32, E)>> {
        None
    }

    /// Divide dense-indexed coefficients exactly by another polynomial.
    ///
    /// On success, the quotient follows the same sparse, strictly increasing representation as
    /// [`Self::try_dense_mul`].
    fn try_dense_exact_division(
        &self,
        _request: DensePolynomialExactDivisionRequest<'_, E>,
    ) -> Option<Vec<(u32, E)>> {
        None
    }

    /// Return a stricter mixed-radix-to-simplex workspace ratio for an early compact total-degree
    /// attempt below the unconditional product-density threshold.
    ///
    /// The coefficient slices and compact output length are supplied before the rank tables are
    /// built, allowing a kernel with several accumulator representations to select the workspace
    /// threshold for one that can handle the complete request. Returning `None` leaves mixed-radix
    /// multiplication first. The caller has already checked its standard workspace threshold, so
    /// implementations should return at least that ratio. Requests at or above the unconditional
    /// density threshold retain the standard workspace rule without calling this method.
    fn preferred_total_degree_mul_workspace_ratio(
        &self,
        _left_coefficients: &[E],
        _right_coefficients: &[E],
        _output_len: usize,
    ) -> Option<usize> {
        Some(8)
    }
}

/// Coefficient-domain kernels returned by [`crate::domains::Ring::kernels`].
pub struct RingKernels<'a, E> {
    polynomial: Option<&'a dyn PolynomialKernels<E>>,
    geometric_sequences: Option<&'a dyn GeometricSequenceKernels<E>>,
    preferred_total_degree_mul_density: Option<usize>,
}

impl<'a, E> RingKernels<'a, E> {
    /// Construct a bundle with no specialized kernels.
    #[inline]
    pub const fn empty() -> Self {
        Self {
            polynomial: None,
            geometric_sequences: None,
            preferred_total_degree_mul_density: None,
        }
    }

    /// Add polynomial multiplication and division kernels.
    #[inline]
    #[must_use]
    pub fn with_polynomial(mut self, kernels: &'a dyn PolynomialKernels<E>) -> Self {
        self.polynomial = Some(kernels);
        self
    }

    /// Add geometric-sequence evaluation kernels.
    #[inline]
    #[must_use]
    pub fn with_geometric_sequences(
        mut self,
        kernels: &'a dyn GeometricSequenceKernels<E>,
    ) -> Self {
        self.geometric_sequences = Some(kernels);
        self
    }

    /// Try the coefficient domain's total-degree multiplication kernel before mixed-radix dense
    /// multiplication once this many coefficient products contribute per compact output cell.
    #[inline]
    #[must_use]
    pub fn with_preferred_total_degree_mul_density(mut self, density: usize) -> Self {
        self.preferred_total_degree_mul_density = (density > 0).then_some(density);
        self
    }

    /// Return polynomial multiplication and division kernels, when available.
    #[inline]
    pub fn polynomial(&self) -> Option<&'a dyn PolynomialKernels<E>> {
        self.polynomial
    }

    /// Return geometric-sequence evaluation kernels, when available.
    #[inline]
    pub fn geometric_sequences(&self) -> Option<&'a dyn GeometricSequenceKernels<E>> {
        self.geometric_sequences
    }

    /// Return the product-density threshold for an early total-degree kernel attempt.
    #[inline]
    pub fn preferred_total_degree_mul_density(&self) -> Option<usize> {
        self.polynomial.and(self.preferred_total_degree_mul_density)
    }
}
