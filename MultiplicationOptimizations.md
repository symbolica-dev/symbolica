# Multivariate Integer Polynomial Multiplication Optimizations

This note documents the multiplication performance work introduced in commits `9f1bb90`
(`Reuse large integer allocations`) and `2cd9103` (`Optimize dense integer polynomial
arithmetic`), together with the follow-up refinements in the current worktree.

The implementation is single-threaded. It improves the representation and arithmetic used by
one multiplication; it does not split a product across threads.

## Previous path

`MultivariatePolynomial::mul_dense` already mapped a multivariate exponent vector to one mixed-
radix integer. If variable `i` has product degree bound `d_i`, its radix is `d_i + 1`. These
radices are large enough that adding the encoded positions of two input monomials cannot carry
between variables. Polynomial multiplication can therefore use

```text
output_index = left_index + right_index
```

for every coefficient product.

The old dense implementation still performed generic pairwise coefficient arithmetic. Every
pair went through the tagged `Integer` operations, and an arbitrary-precision product could be
created before it was added to an output coefficient. After accumulation, the implementation
scanned its dense coefficient or index workspace and decoded each surviving position.

The heap multiplier remains the fallback when the mixed-radix box is too large or the inputs are
not ordinary polynomials.

## Kernel dispatch

`Ring::kernels` returns a short-lived `RingKernels` capability bundle. The polynomial layer builds
a `DensePolynomialMulRequest` after computing the dense layout, then asks the optional
`PolynomialKernels` capability to handle it:

```rust
fn try_dense_mul(
    &self,
    request: DensePolynomialMulRequest<'_, E>,
) -> Option<Vec<(u32, E)>>;
```

The request types and capability traits live in `lib/numerica/src/kernels.rs`. Integer-specific
operation contexts live in `lib/numerica/src/domains/integer/polynomial_kernels.rs`. Returning
`None` preserves the generic polynomial fallback. A successful call returns only nonzero
`(dense_index, coefficient)` pairs in strictly increasing index order.

`IntegerRing` performs the following sequential dispatch:

```text
all coefficients fit i64 and the output bound fits i64
    -> blocked i64 convolution
else all coefficients fit i64 and the output bound fits i128
    -> blocked i128-accumulator convolution
else all coefficients fit i128 and the output bound fits i128
    -> blocked i128 convolution
else GMP build and the product has very high coefficient collision
    -> Kronecker substitution
else GMP build, at least one coefficient is large, and the dense box is reasonable
    -> blocked GMP accumulator array
else
    -> generic dense or heap multiplication
```

`DenseIntegerMul::new` validates coefficient/index lengths and the largest possible output index
once. Its `run` method owns the ordered strategy selection.

## Fixed-width dense kernels

The `DenseIntegerMul::{try_i64, try_i64_i128, try_i128}` strategies handle coefficients that can
be accumulated without overflow.

For each type, the kernel computes the conservative bound

```text
max_abs(left) * max_abs(right) * min(left_terms, right_terms)
```

before doing any unchecked arithmetic. Strictly increasing input indices imply that at most the
smaller number of input terms can collide at one output position. If the bound does not fit, the
kernel declines and dispatch continues to a wider representation.

After the check, coefficients are unboxed once into a contiguous fixed-width vector. The
convolution uses 32-by-32 term blocks. Loop-invariant left indices and coefficients are loaded
outside the inner loop, and bounds checks are omitted inside the verified kernel.

This avoids repeated `Integer::{Single, Double, Large}` matching and keeps the hot loop on native
integer add and multiply instructions.

## Kronecker substitution

`DenseIntegerMul::try_kronecker` packs a polynomial into one GMP integer, performs one large
integer multiplication, and unpacks the coefficients.

The digit width includes:

- the largest bit length in each input;
- enough carry bits for the maximum number of colliding products;
- one sign bit;
- rounding to a whole 64-bit limb.

Negative coefficients are handled by signed digits and carry correction during unpacking. The
packed operands are capped at `1 << 29` bits.

Packing is only selected when there are at least 64 pair products and

```text
output_len * 128 < left_terms * right_terms
```

so the cost of packing is reserved for products with very high collision density. Less dense
products continue to the direct array kernel or generic fallback.

## Direct GMP accumulator kernel

`DenseIntegerMul::try_large_array` handles products containing at least one
`Integer::Large`. It allocates one `MultiPrecisionInteger` accumulator per dense output position
and performs the same 32-by-32 blocked convolution as the fixed-width kernels.

The kernel is limited to cases with:

- at least 64 pair products;
- at most `1 << 20` dense output positions;
- an output box no more than ten times the number of pair products.

Its `add_product` helper specializes all combinations of `Single`, `Double`, and `Large`:

- fixed-width products remain in `i128` when possible;
- small-by-large products use GMP add-multiply with an `i64` or `i128` scalar;
- large-by-large products use GMP add-multiply directly into the destination.

In the GMP cases, the product is not materialized as a temporary `Integer`. This is the concrete
difference from generic pairwise coefficient multiplication: the generic expression conceptually
computes `product = left * right` and then `output += product`, whereas the specialized kernel
updates `output += left * right` in one accumulator operation.

## Sparse result construction

All specialized kernels now return only nonzero output terms. This keeps zero coefficients out of
the result and moves representation conversion to the point at which a coefficient is known to
survive.

In `MultivariatePolynomial::mul_dense`, `advance_uni_var` incrementally advances the exponent
odometer from the previous returned dense index. It replaces a complete mixed-radix decode for
every possible output slot. This is especially useful when the dense coefficient workspace
contains cancellations or holes.

The older generic dense fallback remains available. For large boxes it reuses the thread-local
`DENSE_MUL_BUFFER` index array rather than allocating and clearing an index map for every product.

## Large integer support

The polynomial kernels rely on lower-level changes to make repeated large coefficient operations
cheaper.

### Fused multiply-accumulate operations

`MultiPrecisionInteger` gained these in-place helpers in
`lib/numerica/src/domains/backend/integer.rs`:

- `add_mul_assign` and `sub_mul_assign`;
- `add_i64_mul_assign` and `sub_i64_mul_assign`;
- `add_i128_mul_assign` and `sub_i128_mul_assign`.

`Integer::fused_mul_assign` selects among native and GMP forms while promoting the accumulator at
most once. `IntegerRing::add_mul_assign` and `IntegerRing::sub_mul_assign` use this helper, so the
generic polynomial paths also avoid temporary GMP products.

### Bounded GMP allocation cache

`MultiPrecisionInteger` uses a thread-local cache of cleared GMP integers:

- at most 32 cached values per thread;
- only allocations with capacity at most `1 << 20` bits are retained;
- `Default` pops an available value;
- `Drop` clears and returns a cacheable value when space remains.

This preserves useful limb capacity for short-lived accumulators without unbounded retention or
cross-thread synchronization. Owned arithmetic also extracts the raw GMP value with `into_raw`
where possible, allowing GMP to reuse the consumed operand's storage.

The `faster_alloc` feature remains enabled by Symbolica's default feature set. The GMP cache is an
additional, more targeted reuse mechanism rather than a replacement for the process allocator.

## Added and changed functions

| Function or item | Location | Purpose |
|---|---|---|
| `Ring::kernels` and `RingKernels` | `lib/numerica/src/domains.rs`, `lib/numerica/src/kernels.rs` | Expose optional coefficient-domain operation kernels |
| `PolynomialKernels` request interface | `lib/numerica/src/kernels.rs` | Describe dense, total-degree, and exact-division operations |
| `DenseIntegerMul::{try_i64,try_i64_i128,try_i128}` | `lib/numerica/src/domains/integer/polynomial_kernels.rs` | Overflow-checked fixed-width convolutions |
| `DenseIntegerMul::try_kronecker` | `lib/numerica/src/domains/integer/polynomial_kernels.rs` | Pack, multiply once with GMP, and unpack |
| `DenseIntegerMul::try_large_array` | `lib/numerica/src/domains/integer/polynomial_kernels.rs` | Dense GMP accumulator convolution |
| `Integer::fused_mul_assign` | `lib/numerica/src/domains/integer.rs` | Select a temporary-free multiply-accumulate operation |
| `MultiPrecisionInteger::{add,sub}_mul_assign` | `lib/numerica/src/domains/backend/integer.rs` | Large-by-large fused accumulation |
| `MultiPrecisionInteger::{add,sub}_i{64,128}_mul_assign` | `lib/numerica/src/domains/backend/integer.rs` | Small-by-large fused accumulation |
| `LARGE_INTEGER_CACHE` | `lib/numerica/src/domains/backend/integer.rs` | Bounded thread-local reuse of GMP limb allocations |
| `advance_uni_var` | `src/poly/polynomial.rs` | Incrementally decode sparse dense-index results |
| `INTEGER_MULTIPLICATION_CASES` | `benches/support/cases.rs` | Shared Symbolica/FLINT integer multiplication fixtures |

## Benchmark coverage

Matching Symbolica and FLINT cases are defined once in `benches/support/cases.rs` and executed by
`benches/symbolica_polynomial.rs` and `benches/flint_comparison.rs`.

The multiplication cases are:

| Case | Left power | Right power | Actual term counts (left/right/product) | Purpose |
|---|---:|---:|---:|---|
| dense small | 12 | 11 | 455 / 364 / 2600 | Moderate native-coefficient baseline |
| dense high | 12 | 11 | 455 / 364 / 2600 | Same support with large coefficient growth |
| dense large | 24 | 23 | 2925 / 2600 / 19596 | Larger native-coefficient product |
| dense very large | 40 | 39 | 12341 / 11480 / 88557 | Very large dense support |
| dense high large | 20 | 19 | 1771 / 1540 / 11480 | Larger support and high coefficients |
| sparse separated | 7 | 7 | 792 / 792 / 11628 | Lacunary support with widely separated exponent vectors |
| sparse large | 7 | 7 | 3432 / 3432 / 116280 | Large lacunary heap-multiplication case |
| seven-variable power-minus-one | 7 | 7 | 3431 / 3431 / 116272 | Square of `(1+3*x1+...+15*x7)^7-1` |

Both Divan runners validate the expected term counts outside the timed region. FLINT calls
`flint_set_num_threads(1)`, and the runners configure Rayon for one thread. Paired mode warms both
implementations, alternates their execution order, and reports the median Symbolica/FLINT ratio:

```bash
SYMBOLICA_FLINT_BENCH_PAIRED=1 \
SYMBOLICA_FLINT_BENCH_FILTER='dense very large multiplication' \
SYMBOLICA_FLINT_BENCH_SAMPLES=8 \
taskset -c 0 cargo bench --features flint_benchmarks --bench flint_comparison
```

The Symbolica-only runner uses `--features symbolica_benchmarks --bench symbolica_polynomial` and
the same shared fixtures. Both runners require the normal license environment to be configured;
no license value is embedded in the benchmark or this document.

## Correctness and fallback behavior

The specialized interface is optional and conservative. A kernel returns `None` whenever a bound,
layout, size, feature, or coefficient-representation requirement is not satisfied. Multiplication
then uses the existing generic dense implementation, and finally the heap implementation if a
dense layout itself is unsuitable.

Unit coverage includes fixed-width dense multiplication, signed Kronecker unpacking, large GMP
array multiplication, fused large integer products, and bounded cache reuse. The benchmark also
constructs the product before timing and reports its term count, which makes unexpected support or
cancellation changes visible.
