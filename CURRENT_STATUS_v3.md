# Current Symbolica/FLINT polynomial performance

Version `v3` records commit `f39c09b` relative to `v2`. The initial unbounded bivariate sampler
now accepts the first admissible primitive image when the main-variable leading coefficient is
constant; any failed reconstruction retries with the existing conservative factor bound and
three-image policy. The former canonical worst case, dense three-variable degree-6/5
factorization, improves from `1.915064` to `0.672529` S/F; its exact-commit median is
`2.797780 ms` versus FLINT's `4.160090 ms`, and the source-matched Symbolica control was
`7.240756 ms`. The affected guard family was refreshed at the same final source: generated dense
two-variable factorization changes `0.575899 -> 0.565520`; PolyBench 5v uniform cases
`#32/#159/#163/#131` change `1.134035/0.996512/0.224814/1.740523` to
`1.124230/0.982263/0.221031/1.756528`; 8v sharp cases `#178/#92/#84/#176` change
`0.238964/0.839857/0.759183/1.009899 -> 0.239560/0.836402/0.688572/1.005858`; and 8v uniform
`#44/#105/#159` change `0.100709/1.092682/0.171694 -> 0.097756/1.106606/0.170000`.
These guard refreshes include ordinary timing movement and are not all attributed to the new
branch. An exact dense-convolution-window prototype was correct and often beat Symbolica's generic
full-product fallback, but its Newton/Hensel integration regressed degree 64
(`1.290968 -> 1.313523`), so both the integration and the otherwise-unused public kernel surface
were rejected.
The live latest view is [CURRENT_STATUS.md](CURRENT_STATUS.md).

`S/F` is Symbolica's median time divided by FLINT's median time. Lower is better: a value below
`1` means Symbolica is faster. Repeated-process rows use the median of the process-level paired
ratios. Benchmarks are sequential and single-threaded, use release/default features (including
`faster_alloc`), and compare against system FLINT 3.6.0 in the same process. Rows accumulated
over several accepted source snapshots, so comparisons between different rows are descriptive;
use source-matched alternating A/B runs to attribute an optimization.

Detailed optimization decisions, profiles, and rejected experiments are in
[PolynomialPerformanceHandoff.md](PolynomialPerformanceHandoff.md).

## Summary

| Measure | Current result |
|---|---:|
| All current comparisons | 134 |
| Canonical comparisons | 122 |
| Duplicate Brown/CRT comparisons | 12 |
| Symbolica faster than FLINT, all rows | 89 |
| Symbolica slower than FLINT, all rows | 45 |
| Worst S/F across all rows | **2.581685** — Brown high-height resultant |
| Best S/F across all rows | **0.028678** — high-gap eight-variable GCD |
| Worst canonical S/F | **1.905386** — sparse eight-variable degree-5 GCD product construction |
| Median canonical S/F | **0.690194** |
| Best canonical S/F | **0.028678** — high-gap eight-variable GCD |
| Median across all 134 rows | **0.720416** |
| PolyBench GCD product median (12 cases) | **0.604569** |
| PolyBench factor product median (11 cases) | **0.601621** |
| All PolyBench product median (23 cases) | **0.601621** |

The canonical inventory contains one current implementation per workload. For resultants that is
the main Ducos entry point. Brown and CRT measurements duplicate those six inputs and are marked
`alternative`; they appear in the globally sorted table but do not alter the 122-row canonical
summary.

## All current comparisons, worst to best

| Rank | S/F | Inventory | Category | Benchmark |
|---:|---:|---|---|---|
| 1 | 2.581685 | `alternative` | `brown_resultant_alternative` | resultant Brown: large high-height degrees 14/10 |
| 2 | 2.464208 | `alternative` | `brown_resultant_alternative` | resultant Brown: dense outer degrees 10/8 CRT crossover |
| 3 | 2.287301 | `alternative` | `crt_resultant_alternative` | resultant CRT: lacunary outer degrees 18/11 |
| 4 | 2.059289 | `alternative` | `crt_resultant_alternative` | resultant CRT: dense outer degrees 7/6 |
| 5 | 1.905386 | `canonical` | `generated_gcd_product` | generated GCD products: sparse 8 variables degree 5 |
| 6 | 1.901314 | `canonical` | `generated_gcd_product` | generated GCD products: sparse 5 variables degree 7 |
| 7 | 1.889336 | `canonical` | `generated_gcd_product` | generated GCD products: dense 5 variables degree 7 |
| 8 | 1.878681 | `canonical` | `generated_gcd_product` | generated GCD products: high-gap 8 variables degree 4 gap 256 |
| 9 | 1.862295 | `alternative` | `brown_resultant_alternative` | resultant Brown: outer-sparse degrees 12/9 CRT crossover |
| 10 | 1.756528 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #131 |
| 11 | 1.584134 | `canonical` | `default_gcd_product` | GCD products: dense 7 variables degree 7 |
| 12 | 1.549423 | `canonical` | `generated_gcd_product` | generated GCD products: dense 8 variables degree 5 |
| 13 | 1.499961 | `canonical` | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 128 |
| 14 | 1.438647 | `alternative` | `brown_resultant_alternative` | resultant Brown: lacunary outer degrees 18/11 |
| 15 | 1.348957 | `canonical` | `dedicated_multiplication` | seven-variable power-minus-one multiplication |
| 16 | 1.333630 | `alternative` | `brown_resultant_alternative` | resultant Brown: dense outer degrees 7/6 |
| 17 | 1.296474 | `canonical` | `generated_factor_operation` | generated factorization: dense 1-variable degrees 33/31 total 64 |
| 18 | 1.268240 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #140 |
| 19 | 1.249141 | `alternative` | `crt_resultant_alternative` | resultant CRT: large high-height degrees 14/10 |
| 20 | 1.234835 | `canonical` | `dense_gcd_operation` | configured dense univariate GCD degree 80 |
| 21 | 1.213809 | `canonical` | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 256 |
| 22 | 1.184942 | `canonical` | `generated_factor_product` | generated factor product: dense 1-variable degrees 33/32 total 65 |
| 23 | 1.181340 | `canonical` | `dense_gcd_operation` | configured dense univariate GCD degree 64 |
| 24 | 1.179446 | `canonical` | `generated_gcd_product` | generated GCD products: dense 1 variables degree 32 |
| 25 | 1.164159 | `canonical` | `generated_factor_operation` | generated factorization: dense 1-variable degrees 33/32 total 65 |
| 26 | 1.137177 | `canonical` | `dedicated_multiplication` | GF(18446744073709551557) dense large multiplication |
| 27 | 1.129343 | `alternative` | `crt_resultant_alternative` | resultant CRT: nonunit leading degrees 9/7 |
| 28 | 1.124230 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #32 |
| 29 | 1.124006 | `alternative` | `brown_resultant_alternative` | resultant Brown: nonunit leading degrees 9/7 |
| 30 | 1.112701 | `canonical` | `generated_factor_operation` | generated factorization: dense high-height 1-variable degrees 17/16 total 33 |
| 31 | 1.106606 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #105 |
| 32 | 1.097676 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 5v uniform nontrivial GCD #11 |
| 33 | 1.094628 | `canonical` | `generated_factor_product` | generated factor product: dense 1-variable degrees 33/31 total 64 |
| 34 | 1.072266 | `canonical` | `dense_gcd_operation` | configured dense univariate GCD degree 48 |
| 35 | 1.060731 | `canonical` | `generated_gcd_operation` | generated GCD auto: dense 2 variables degree 5 |
| 36 | 1.056738 | `canonical` | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 512 |
| 37 | 1.056183 | `canonical` | `dedicated_multiplication` | dense large multiplication |
| 38 | 1.055155 | `canonical` | `dedicated_multiplication` | GF(17) dense very large multiplication |
| 39 | 1.054611 | `canonical` | `generated_gcd_product` | generated GCD products: high-height 8 variables degree 3 coefficient bits 256 |
| 40 | 1.034406 | `canonical` | `generated_factor_product` | generated factor product: dense 1-variable degrees 32/31 |
| 41 | 1.029727 | `canonical` | `generated_factor_operation` | generated factorization: dense 1-variable degrees 32/31 |
| 42 | 1.027247 | `canonical` | `main_ducos_resultant` | resultant Ducos: outer-sparse degrees 12/9 CRT crossover |
| 43 | 1.012638 | `canonical` | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 1024 |
| 44 | 1.009108 | `canonical` | `generated_factor_product` | generated factor product: dense 3-variable degrees 6/5 |
| 45 | 1.005858 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #176 |
| 46 | 0.994841 | `canonical` | `generated_gcd_operation` | generated GCD auto: dense 1 variables degree 32 |
| 47 | 0.982263 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #159 |
| 48 | 0.968978 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform trivial GCD #11 |
| 49 | 0.966693 | `canonical` | `dedicated_multiplication` | dense high multiplication |
| 50 | 0.956438 | `canonical` | `dedicated_multiplication` | dense high large multiplication |
| 51 | 0.940972 | `canonical` | `dedicated_multiplication` | GF(500000003) dense univariate degrees 128/64 accumulator-bound multiplication |
| 52 | 0.932951 | `canonical` | `main_ducos_resultant` | resultant Ducos: dense outer degrees 10/8 CRT crossover |
| 53 | 0.913795 | `canonical` | `dedicated_multiplication` | dense small multiplication |
| 54 | 0.912259 | `canonical` | `generated_gcd_product` | generated GCD products: dense 3 variables degree 7 |
| 55 | 0.902871 | `canonical` | `default_gcd_operation` | GCD auto: dense 7 variables degree 7 |
| 56 | 0.870516 | `canonical` | `dedicated_multiplication` | GF(17) dense univariate degree-4912 multiplication |
| 57 | 0.861806 | `canonical` | `main_ducos_resultant` | resultant Ducos: large high-height degrees 14/10 |
| 58 | 0.858115 | `canonical` | `generated_factor_product` | generated factor product: dense 2-variable degrees 10/9 |
| 59 | 0.836402 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #92 |
| 60 | 0.828262 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #55 |
| 61 | 0.821527 | `canonical` | `dedicated_multiplication` | GF(17) dense large multiplication |
| 62 | 0.799572 | `canonical` | `generated_gcd_product` | generated GCD products: high-gap 5 variables degree 5 gap 64 |
| 63 | 0.798104 | `canonical` | `dedicated_multiplication` | GF(65000011) dense univariate degrees 128/64 accumulator-bound multiplication |
| 64 | 0.761777 | `canonical` | `exact_integer_division` | high-height exact division |
| 65 | 0.750982 | `canonical` | `generated_gcd_operation` | generated GCD auto: dense 5 variables degree 7 |
| 66 | 0.748107 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #11 |
| 67 | 0.730824 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #53 |
| 68 | 0.710007 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #84 |
| 69 | 0.705610 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #176 |
| 70 | 0.692630 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #11 |
| 71 | 0.691815 | `canonical` | `dedicated_multiplication` | sparse separated multiplication |
| 72 | 0.688572 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #84 |
| 73 | 0.680006 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #35 |
| 74 | 0.674333 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #11 |
| 75 | 0.672529 | `canonical` | `generated_factor_operation` | generated factorization: dense 3-variable degrees 6/5 |
| 76 | 0.661312 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #92 |
| 77 | 0.657180 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 5v sharp nontrivial GCD #11 |
| 78 | 0.642318 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #35 |
| 79 | 0.635520 | `canonical` | `generated_gcd_operation` | generated GCD auto: sparse 5 variables degree 7 |
| 80 | 0.627013 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #105 |
| 81 | 0.625893 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #178 |
| 82 | 0.621653 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #140 |
| 83 | 0.607815 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #56 |
| 84 | 0.607539 | `canonical` | `dedicated_multiplication` | GF(17) sparse large multiplication |
| 85 | 0.607374 | `alternative` | `crt_resultant_alternative` | resultant CRT: dense outer degrees 10/8 CRT crossover |
| 86 | 0.601621 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #44 |
| 87 | 0.601321 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #188 |
| 88 | 0.598450 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #32 |
| 89 | 0.595237 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #131 |
| 90 | 0.593728 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform trivial GCD #11 |
| 91 | 0.593060 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #55 |
| 92 | 0.591952 | `canonical` | `generated_gcd_operation` | generated GCD auto: dense 8 variables degree 5 |
| 93 | 0.591190 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #168 |
| 94 | 0.590298 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 5v uniform nontrivial GCD #11 |
| 95 | 0.586162 | `canonical` | `dedicated_multiplication` | sparse large multiplication |
| 96 | 0.585393 | `canonical` | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #11 |
| 97 | 0.577782 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #159 |
| 98 | 0.574786 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #159 |
| 99 | 0.574252 | `canonical` | `dedicated_multiplication` | GF(18446744073709551557) sparse large multiplication |
| 100 | 0.573810 | `canonical` | `generated_factor_product` | generated factor product: dense high-height 1-variable degrees 17/16 total 33 |
| 101 | 0.570573 | `canonical` | `main_ducos_resultant` | resultant Ducos: lacunary outer degrees 18/11 |
| 102 | 0.566137 | `canonical` | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #163 |
| 103 | 0.565520 | `canonical` | `generated_factor_operation` | generated factorization: dense 2-variable degrees 10/9 |
| 104 | 0.554473 | `canonical` | `main_ducos_resultant` | resultant Ducos: nonunit leading degrees 9/7 |
| 105 | 0.537410 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #168 |
| 106 | 0.519856 | `canonical` | `main_ducos_resultant` | resultant Ducos: dense outer degrees 7/6 |
| 107 | 0.514253 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 5v sharp nontrivial GCD #11 |
| 108 | 0.513866 | `canonical` | `dedicated_multiplication` | GF(18446744073709551557) dense very large multiplication |
| 109 | 0.472101 | `canonical` | `generated_gcd_operation` | generated GCD auto: high-height 8 variables degree 3 coefficient bits 256 |
| 110 | 0.470237 | `canonical` | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 1024 |
| 111 | 0.458120 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #188 |
| 112 | 0.449493 | `canonical` | `dedicated_multiplication` | dense very large multiplication |
| 113 | 0.447907 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #56 |
| 114 | 0.442621 | `canonical` | `exact_integer_division` | dense large exact division |
| 115 | 0.441603 | `canonical` | `exact_integer_division` | dense exact division |
| 116 | 0.437142 | `canonical` | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 512 |
| 117 | 0.436875 | `canonical` | `dedicated_multiplication` | GF(18446744073709551557) seven-variable power-minus-one multiplication |
| 118 | 0.434133 | `canonical` | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 256 |
| 119 | 0.432838 | `alternative` | `crt_resultant_alternative` | resultant CRT: outer-sparse degrees 12/9 CRT crossover |
| 120 | 0.429881 | `canonical` | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 128 |
| 121 | 0.413575 | `canonical` | `dedicated_multiplication` | GF(18446744073709551557) five-variable total-degree multiplication |
| 122 | 0.390026 | `canonical` | `dedicated_multiplication` | GF(17) seven-variable power-minus-one multiplication |
| 123 | 0.387980 | `canonical` | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #53 |
| 124 | 0.378042 | `canonical` | `dedicated_multiplication` | GF(17) five-variable total-degree multiplication |
| 125 | 0.356182 | `canonical` | `generated_gcd_operation` | generated GCD auto: sparse 8 variables degree 5 |
| 126 | 0.292980 | `canonical` | `generated_gcd_product` | generated GCD products: dense 2 variables degree 5 |
| 127 | 0.252283 | `canonical` | `dedicated_multiplication` | GF(18446744073709551557) dense univariate degree-4912 multiplication |
| 128 | 0.239560 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #178 |
| 129 | 0.221031 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #163 |
| 130 | 0.201668 | `canonical` | `generated_gcd_operation` | generated GCD auto: dense 3 variables degree 7 |
| 131 | 0.170000 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #159 |
| 132 | 0.097756 | `canonical` | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #44 |
| 133 | 0.073806 | `canonical` | `generated_gcd_operation` | generated GCD auto: high-gap 5 variables degree 5 gap 64 |
| 134 | 0.028678 | `canonical` | `generated_gcd_operation` | generated GCD auto: high-gap 8 variables degree 4 gap 256 |

## Current high-interest replacements

| Optimization | Previous source-matched S/F | Current S/F | Result |
|---|---:|---:|---|
| Initial primitive bivariate sample, dense three-variable degrees 6/5 | 1.747643 | 0.672529 | Symbolica median reduced 61.4%; now 1.49x faster than FLINT |
| Three-factor Hensel root, high-height degree 33 | 1.997930 | 1.112701 | Symbolica absolute time reduced about 40% |
| Sparse high-gap five-variable product heap advance | 1.040981 | 0.799572 | S/F reduced about 23.2% |
| PolyBench factorization #84 variable order | 8.690880 | 0.688572 | Current S/F is 12.62x below the pre-selection ratio |
| Dense degree-64 factorization | 1.748240 | 1.296474 | Dense EDF reduced absolute time 19.2%; Hensel image reuse adds 0.67% |

The high-height degree-33 product itself is `0.573810` S/F, so construction is faster than FLINT;
the remaining factorization gap is in the factorization pipeline, not that multiplication.

## Versioning and update protocol

`CURRENT_STATUS.md` is the live latest view. Every accepted refresh must also create the next
immutable `CURRENT_STATUS_v<i>.md` snapshot. Never edit an older numbered snapshot. The first
paragraph of each new snapshot must identify the preceding version and summarize every affected
case or family with its old and new S/F values.

After every accepted performance change:

1. Rerun every affected case sequentially with the current source-matched Symbolica/FLINT binary.
2. For repeated processes, replace the row with the median process-level S/F. Do not retain an old
   candidate or control as another current row.
3. Merge the 122 canonical rows and 12 alternative Brown/CRT rows, sort all 134 numerically by S/F
   in descending order, and renumber them.
4. Recompute the summary and verify 122 unique canonical workload/category pairs, 12 alternatives,
   and monotonic descending ratios.
5. Create the next numbered snapshot with its opening delta paragraph, then make
   `CURRENT_STATUS.md` mirror that latest snapshot.
6. Record benchmark binary hashes, raw timing artifacts, profiles, accepted changes, and rejected
   experiments in [PolynomialPerformanceHandoff.md](PolynomialPerformanceHandoff.md).

The `v3` inventory was programmatically checked for 122 unique canonical rows, 12 alternatives,
and monotonically descending S/F values after all 13 affected factorization rows were refreshed.
