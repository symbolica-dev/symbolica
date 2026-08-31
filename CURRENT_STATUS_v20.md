# Current Symbolica/FLINT polynomial performance

Version `v20` attacks the largest `v19` fixed-fixture gaps with generic support-, width-, and
workspace-based integer kernels. A lower-band compact-total-degree selector changes the final
combined S/F ratios of the ordinary asymmetric eight-variable 165/495/1,287-term product rows from
`1.327830/1.295551/0.746378` to `0.973857/0.871657/0.780811`; the 256-bit asymmetric guard changes
`1.174073 -> 1.051418`. The first two former headline losses are now Symbolica wins. Admission
depends on product density, mixed-radix/simplex workspace ratio, and a proven accumulator bound,
not a fixture identity.

Bounded Hensel corrections now compute `((value rem divisor) * multiplier) rem divisor` in native
`i128` storage when checked bounds cover both long divisions and the convolution. The final
dense degree-63/64/65/high-height factor rows change from
`1.047138/0.853381/1.093223/0.986543` to
`1.048729/0.778657/1.008840/0.958149`. The isolated degree-65 A/B reduces Symbolica time by 8.87%.

The contiguous Kronecker path now keeps one-to-eight-limb digits in native storage through direct
`mpn_mul`, streams the decoder, constructs large results through GMP's writable-limb interface,
and computes bounded L1 statistics without temporary GMP products. Dense degree-63/64/65 factor
products change from `0.995256/1.027605/1.102061` to `0.890401/0.884083/0.918407`; the newly added
dense degree-80 product is `1.006322`. The dense one-variable degree-32 GCD product changes
`1.029726 -> 0.814288`, while the refreshed three-variable factor-input product exposes a small,
setup-heavy regression, `1.009108 -> 1.109145`, and becomes the new worst primary row.

The compact GMP-limb kernel now accepts bounded inputs through 128 limbs. Degree-four expansions
of nominal 512- and 1,024-bit coefficients previously crossed the 32-limb cap and fell back from a
1,287-cell simplex to a 59,049-cell mixed-radix workspace. Their final S/F ratios change
`1.054182/1.017696 -> 0.970210/0.991526`; 128/256-bit guards remain wins at
`0.867228/0.949953`. Other refreshed guards include dense bivariate GCD
`1.060731 -> 1.017892`, dense-large integer multiplication `1.056183 -> 1.004833`, and dense-small
integer multiplication `0.913795 -> 0.906259`.

The `v20` primary inventory has 125 rows: 116 Symbolica wins, nine losses, median `0.636796`, and
worst `1.109145`. On the old v14-compatible population that worst remains below v14's `1.137177`.
All reported selectors are single-core and use algebraic or computational properties of the input.

Across the 3,200 paired PolyBench measurements, the median per-problem S/F ratio is `0.962614`,
Symbolica wins 1,832 problems, and eight of sixteen setup medians favor Symbolica. These
subprocess distributions use exact FLINT 3.5.0 and are reported separately from the fixed-fixture
primary scoreboard, which uses same-process FLINT 3.6.0.

Ducos, Brown, and CRT resultant measurements remain a historical appendix note and never
participate in the primary ranks or summary.

The live latest view is [CURRENT_STATUS.md](CURRENT_STATUS.md).

`S/F` in the primary scoreboard is Symbolica's median time divided by FLINT's median time. Lower
is better: a value below `1` means Symbolica is faster. Repeated-process rows use the median of
the process-level paired ratios. Primary benchmarks are sequential and single-threaded, use
release/default features (including `faster_alloc`), and compare against system FLINT 3.6.0 in
the same process. The separate PolyBench distribution table uses the median of 200 paired
per-problem ratios against subprocess FLINT 3.5.0. Rows accumulated over several accepted source
snapshots, so comparisons between different rows are descriptive; use source-matched alternating
A/B runs to attribute an optimization.

Detailed optimization decisions, profiles, and rejected experiments are in
[PolynomialPerformanceHandoff.md](PolynomialPerformanceHandoff.md).

## Summary

| Measure | Current result |
|---|---:|
| Primary comparisons | 125 |
| Symbolica faster than FLINT | 116 |
| Symbolica slower than FLINT | 9 |
| Worst primary S/F | **1.109145** — dense three-variable degrees-6/5 factor-input product, an 11 us setup-heavy multiplication |
| Worst operation S/F | **1.072572** — configured dense univariate GCD degree 80 |
| Worst on v14's 116-row population | **1.109145** — still below v14's **1.137177** |
| Median primary S/F | **0.636796** |
| Best primary S/F | **0.028678** — generated high-gap eight-variable GCD |
| Focused #131 S/F | **0.256988** — Symbolica is 3.89x faster than FLINT |
| PolyBench GCD product median (12 cases) | **0.613690** |
| PolyBench factor product median (11 cases) | **0.617710** |
| All PolyBench product median (23 cases) | **0.617324** |
| Full PolyBench distributions | **16 setups / 3,200 measured problems** |
| Full PolyBench paired median S/F | **0.962614** — Symbolica wins 1,832/3,200 problems |
| Worst full-distribution setup | **1.595357** — 5v sharp trivial factorization, `0007` |
| Best full-distribution setup | **0.266315** — 5v uniform nontrivial factorization, `0004` |

## Primary comparisons, worst to best

Resultant backends are excluded from this table and its statistics.

| Rank | S/F | Category | Benchmark |
|---:|---:|---|---|
| 1 | 1.109145 | `generated_factor_product` | generated factor product: dense 3-variable degrees 6/5 |
| 2 | 1.072572 | `dense_gcd_operation` | configured dense univariate GCD degree 80 |
| 3 | 1.051418 | `generated_gcd_product` | generated GCD products: high-height 8 variables asymmetric cofactor terms 8/45 common terms 165 degrees 1/2/3 coefficient bits 256 |
| 4 | 1.048729 | `generated_factor_operation` | generated factorization: dense 1-variable degrees 32/31 |
| 5 | 1.045860 | `dedicated_multiplication` | GF(17) dense very large multiplication |
| 6 | 1.017892 | `generated_gcd_operation` | generated GCD auto: dense 2 variables degree 5 |
| 7 | 1.008840 | `generated_factor_operation` | generated factorization: dense 1-variable degrees 33/32 total 65 |
| 8 | 1.006322 | `dense_gcd_product` | configured dense univariate GCD products degree 80 |
| 9 | 1.004833 | `dedicated_multiplication` | dense large multiplication |
| 10 | 0.997484 | `dedicated_multiplication` | GF(18446744073709551557) dense large multiplication |
| 11 | 0.991526 | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 1024 |
| 12 | 0.991346 | `generated_gcd_operation` | generated GCD auto: dense 1 variables degree 32 |
| 13 | 0.986146 | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #176 |
| 14 | 0.978321 | `polybench_gcd_operation` | polybench GCD: polybench 5v uniform nontrivial GCD #11 |
| 15 | 0.973857 | `generated_gcd_product` | generated GCD products: dense 8 variables asymmetric cofactor terms 8/45 common terms 165 degrees 1/2/3 |
| 16 | 0.970210 | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 512 |
| 17 | 0.968978 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform trivial GCD #11 |
| 18 | 0.966693 | `dedicated_multiplication` | dense high multiplication |
| 19 | 0.958149 | `generated_factor_operation` | generated factorization: dense high-height 1-variable degrees 17/16 total 33 |
| 20 | 0.956438 | `dedicated_multiplication` | dense high large multiplication |
| 21 | 0.949953 | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 256 |
| 22 | 0.942832 | `generated_gcd_operation` | generated GCD auto: high-height 8 variables asymmetric cofactor terms 8/45 common terms 165 degrees 1/2/3 coefficient bits 256 |
| 23 | 0.940972 | `dedicated_multiplication` | GF(500000003) dense univariate degrees 128/64 accumulator-bound multiplication |
| 24 | 0.918407 | `generated_factor_product` | generated factor product: dense 1-variable degrees 33/32 total 65 |
| 25 | 0.913425 | `generated_gcd_product` | generated GCD products: high-height 8 variables degree 3 coefficient bits 256 |
| 26 | 0.906259 | `dedicated_multiplication` | dense small multiplication |
| 27 | 0.902871 | `default_gcd_operation` | GCD auto: dense 7 variables degree 7 |
| 28 | 0.893922 | `dense_gcd_operation` | configured dense univariate GCD degree 48 |
| 29 | 0.890401 | `generated_factor_product` | generated factor product: dense 1-variable degrees 32/31 |
| 30 | 0.884083 | `generated_factor_product` | generated factor product: dense 1-variable degrees 33/31 total 64 |
| 31 | 0.878519 | `generated_gcd_product` | generated GCD products: dense 3 variables degree 7 |
| 32 | 0.871657 | `generated_gcd_product` | generated GCD products: dense 8 variables asymmetric cofactor terms 8/45 common terms 495 degrees 1/2/4 |
| 33 | 0.870516 | `dedicated_multiplication` | GF(17) dense univariate degree-4912 multiplication |
| 34 | 0.867228 | `generated_gcd_product` | generated GCD products: high-height 5 variables degree 4 coefficient bits 128 |
| 35 | 0.858115 | `generated_factor_product` | generated factor product: dense 2-variable degrees 10/9 |
| 36 | 0.853611 | `dense_gcd_operation` | configured dense univariate GCD degree 64 |
| 37 | 0.828262 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #55 |
| 38 | 0.824694 | `dedicated_multiplication` | GF(17) dense large multiplication |
| 39 | 0.822914 | `generated_gcd_product` | generated GCD products: high-gap 5 variables degree 5 gap 64 |
| 40 | 0.820838 | `generated_gcd_operation` | generated GCD auto: dense 5 variables degree 7 |
| 41 | 0.814288 | `generated_gcd_product` | generated GCD products: dense 1 variables degree 32 |
| 42 | 0.798104 | `dedicated_multiplication` | GF(65000011) dense univariate degrees 128/64 accumulator-bound multiplication |
| 43 | 0.791465 | `generated_gcd_product` | generated GCD products: sparse 5 variables degree 7 |
| 44 | 0.780811 | `generated_gcd_product` | generated GCD products: dense 8 variables asymmetric cofactor terms 8/45 common terms 1287 degrees 1/2/5 |
| 45 | 0.780017 | `generated_gcd_product` | generated GCD products: sparse 8 variables degree 5 |
| 46 | 0.778657 | `generated_factor_operation` | generated factorization: dense 1-variable degrees 33/31 total 64 |
| 47 | 0.761777 | `exact_integer_division` | high-height exact division |
| 48 | 0.750365 | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #53 |
| 49 | 0.748808 | `generated_gcd_product` | generated GCD products: dense 5 variables degree 7 |
| 50 | 0.723532 | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #176 |
| 51 | 0.711976 | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #84 |
| 52 | 0.704684 | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #11 |
| 53 | 0.691815 | `dedicated_multiplication` | sparse separated multiplication |
| 54 | 0.688808 | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #35 |
| 55 | 0.688010 | `generated_gcd_operation` | generated GCD auto: dense 8 variables degree 5 |
| 56 | 0.675222 | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #92 |
| 57 | 0.674700 | `polybench_gcd_product` | polybench GCD products: polybench 5v sharp nontrivial GCD #11 |
| 58 | 0.674333 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #11 |
| 59 | 0.652103 | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #92 |
| 60 | 0.642318 | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #35 |
| 61 | 0.638505 | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #105 |
| 62 | 0.637854 | `polybench_factor_product` | polybench factor product: polybench 8v sharp nontrivial factor #178 |
| 63 | 0.636796 | `polybench_gcd_product` | polybench GCD products: polybench 8v sharp nontrivial GCD #140 |
| 64 | 0.635520 | `generated_gcd_operation` | generated GCD auto: sparse 5 variables degree 7 |
| 65 | 0.617710 | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #44 |
| 66 | 0.617324 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #56 |
| 67 | 0.615198 | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #32 |
| 68 | 0.613128 | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #131 |
| 69 | 0.610055 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #188 |
| 70 | 0.607964 | `polybench_gcd_product` | polybench GCD products: polybench 5v uniform nontrivial GCD #11 |
| 71 | 0.607539 | `dedicated_multiplication` | GF(17) sparse large multiplication |
| 72 | 0.603600 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #55 |
| 73 | 0.602930 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform trivial GCD #11 |
| 74 | 0.601913 | `polybench_factor_product` | polybench factor product: polybench 8v uniform nontrivial factor #159 |
| 75 | 0.600192 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #168 |
| 76 | 0.599923 | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #159 |
| 77 | 0.599179 | `polybench_gcd_product` | polybench GCD products: polybench 8v uniform nontrivial GCD #11 |
| 78 | 0.586252 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #159 |
| 79 | 0.586162 | `dedicated_multiplication` | sparse large multiplication |
| 80 | 0.586125 | `polybench_factor_product` | polybench factor product: polybench 5v uniform nontrivial factor #163 |
| 81 | 0.580344 | `generated_factor_product` | generated factor product: dense high-height 1-variable degrees 17/16 total 33 |
| 82 | 0.576212 | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #105 |
| 83 | 0.574252 | `dedicated_multiplication` | GF(18446744073709551557) sparse large multiplication |
| 84 | 0.554623 | `generated_factor_operation` | generated factorization: dense 3-variable degrees 6/5 |
| 85 | 0.537410 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #168 |
| 86 | 0.524530 | `generated_gcd_product` | generated GCD products: high-gap 8 variables degree 4 gap 256 |
| 87 | 0.517077 | `dedicated_multiplication` | GF(18446744073709551557) dense very large multiplication |
| 88 | 0.514253 | `polybench_gcd_operation` | polybench GCD: polybench 5v sharp nontrivial GCD #11 |
| 89 | 0.479027 | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #140 |
| 90 | 0.470237 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 1024 |
| 91 | 0.467658 | `generated_gcd_operation` | generated GCD auto: dense 8 variables asymmetric cofactor terms 8/45 common terms 1287 degrees 1/2/5 |
| 92 | 0.460421 | `generated_gcd_operation` | generated GCD auto: dense 8 variables asymmetric cofactor terms 8/45 common terms 495 degrees 1/2/4 |
| 93 | 0.458120 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #188 |
| 94 | 0.450165 | `generated_gcd_operation` | generated GCD auto: high-height 8 variables degree 3 coefficient bits 256 |
| 95 | 0.448043 | `dedicated_multiplication` | seven-variable power-minus-one multiplication |
| 96 | 0.447907 | `polybench_gcd_operation` | polybench GCD: polybench 8v uniform nontrivial GCD #56 |
| 97 | 0.445428 | `generated_gcd_operation` | generated GCD auto: dense 8 variables asymmetric cofactor terms 8/45 common terms 165 degrees 1/2/3 |
| 98 | 0.442621 | `exact_integer_division` | dense large exact division |
| 99 | 0.441603 | `exact_integer_division` | dense exact division |
| 100 | 0.437142 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 512 |
| 101 | 0.436875 | `dedicated_multiplication` | GF(18446744073709551557) seven-variable power-minus-one multiplication |
| 102 | 0.434133 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 256 |
| 103 | 0.431152 | `dedicated_multiplication` | dense very large multiplication |
| 104 | 0.429881 | `generated_gcd_operation` | generated GCD auto: high-height 5 variables degree 4 coefficient bits 128 |
| 105 | 0.423081 | `default_gcd_product` | GCD products: dense 7 variables degree 7 |
| 106 | 0.421076 | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #84 |
| 107 | 0.413575 | `dedicated_multiplication` | GF(18446744073709551557) five-variable total-degree multiplication |
| 108 | 0.409963 | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #11 |
| 109 | 0.401382 | `generated_factor_operation` | generated factorization: dense 2-variable degrees 10/9 |
| 110 | 0.390026 | `dedicated_multiplication` | GF(17) seven-variable power-minus-one multiplication |
| 111 | 0.387980 | `polybench_gcd_operation` | polybench GCD: polybench 8v sharp nontrivial GCD #53 |
| 112 | 0.378042 | `dedicated_multiplication` | GF(17) five-variable total-degree multiplication |
| 113 | 0.356182 | `generated_gcd_operation` | generated GCD auto: sparse 8 variables degree 5 |
| 114 | 0.326626 | `generated_gcd_product` | generated GCD products: dense 8 variables degree 5 |
| 115 | 0.289808 | `generated_gcd_product` | generated GCD products: dense 2 variables degree 5 |
| 116 | 0.256988 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #131 |
| 117 | 0.252283 | `dedicated_multiplication` | GF(18446744073709551557) dense univariate degree-4912 multiplication |
| 118 | 0.224514 | `polybench_factor_operation` | polybench factorization: polybench 8v sharp nontrivial factor #178 |
| 119 | 0.201668 | `generated_gcd_operation` | generated GCD auto: dense 3 variables degree 7 |
| 120 | 0.164792 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #32 |
| 121 | 0.161334 | `polybench_factor_operation` | polybench factorization: polybench 5v uniform nontrivial factor #163 |
| 122 | 0.094995 | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #159 |
| 123 | 0.073806 | `generated_gcd_operation` | generated GCD auto: high-gap 5 variables degree 5 gap 64 |
| 124 | 0.063262 | `polybench_factor_operation` | polybench factorization: polybench 8v uniform nontrivial factor #44 |
| 125 | 0.028678 | `generated_gcd_operation` | generated GCD auto: high-gap 8 variables degree 4 gap 256 |

## PolyBench 0.4.3 full distributions, worst to best

These rows use the median of 200 paired per-problem `Symbolica_i / FLINT_i` ratios. They are
kept separate from the primary scoreboard because the workloads are heterogeneous distributions
and use subprocess FLINT 3.5.0. The complete plots, CSVs, logs, protocol, and checksums are in
[benchmark-results/polybench-0.4.3-current](benchmark-results/polybench-0.4.3-current/README.md).

| Rank | Paired S/F | Variables | Setup | Workload | Symbolica wins |
|---:|---:|---:|---:|---|---:|
| 1 | 1.595357 | 5 | 0007 | sharp trivial factorization | 56/200 |
| 2 | 1.425747 | 5 | 0003 | uniform trivial factorization | 26/200 |
| 3 | 1.364144 | 5 | 0008 | sharp nontrivial factorization | 89/200 |
| 4 | 1.323497 | 5 | 0002 | uniform nontrivial GCD | 11/200 |
| 5 | 1.234940 | 8 | 0008 | sharp nontrivial factorization | 84/200 |
| 6 | 1.213492 | 5 | 0006 | sharp nontrivial GCD | 84/200 |
| 7 | 1.018250 | 5 | 0001 | uniform trivial GCD | 31/200 |
| 8 | 1.001295 | 8 | 0007 | sharp trivial factorization | 100/200 |
| 9 | 0.978281 | 5 | 0005 | sharp trivial GCD | 131/200 |
| 10 | 0.969194 | 8 | 0001 | uniform trivial GCD | 174/200 |
| 11 | 0.950340 | 8 | 0005 | sharp trivial GCD | 132/200 |
| 12 | 0.523724 | 8 | 0006 | sharp nontrivial GCD | 188/200 |
| 13 | 0.515150 | 8 | 0003 | uniform trivial factorization | 183/200 |
| 14 | 0.484165 | 8 | 0002 | uniform nontrivial GCD | 184/200 |
| 15 | 0.472376 | 8 | 0004 | uniform nontrivial factorization | 186/200 |
| 16 | 0.266315 | 5 | 0004 | uniform nontrivial factorization | 173/200 |

## Current high-interest replacements

| Optimization | Earlier/reference S/F | Current S/F | Result |
|---|---:|---:|---|
| Lower-band compact-simplex routing, asymmetric 8v products with 165/495 common terms | 1.327830 / 1.295551 | 0.973857 / 0.871657 | Both former headline losses are now wins; admission uses density, workspace, and coefficient bounds |
| Native-limb Kronecker pipeline, dense degree-63/64/65 factor products | 0.995256 / 1.027605 / 1.102061 | 0.890401 / 0.884083 / 0.918407 | Packing, `mpn_mul`, bounded L1 sizing, streaming decode, and writable GMP output avoid transient multiprecision work |
| Bounded native Hensel multiply/remainder, dense degree-64/65 factorization | 0.853381 / 1.093223 | 0.778657 / 1.008840 | Checked `i128` correction buffers reduce modular remainder and convolution overhead |
| 128-limb compact total-degree admission, 512/1024-bit five-variable products | 1.054182 / 1.017696 | 0.970210 / 0.991526 | Degree-four coefficients no longer miss the compact route at the old 32-limb cliff |
| Stack-backed short fixed-coefficient inputs, dedicated dense-small multiplication | 0.913795 | 0.906259 | The isolated A/B reduces Symbolica time by about 5.1%; the 11 us dense-three product remains setup dominated |
| Deferred product-tree Bezout updates, dense degree-64 factorization | 1.055646 | 0.853381 | Exact reconstruction succeeds before the exponent-39 cofactor update; source-matched Symbolica time falls 18.87% |
| Native exact Kronecker-bound bit lengths, dense degrees 63/64/65 products | 1.026466 / 1.048842 / 1.119754 | 0.995256 / 1.027605 / 1.102061 | Allocation-free `u128` wide products reduce Symbolica time 4.4%, 2.7%, and 2.4%; the large-integer fallback is unchanged |
| Exact monomial coefficient-content certificate, PolyBench 8v #105 | 1.106606 | 0.576212 | Symbolica time falls 47.1% against the exact preceding source and is now 1.74x faster than FLINT |
| Coefficient-height-aware Hu prime sizing, asymmetric high-height 8v GCD | 1.631601 | 0.942832 | Targeting at most eight CRT images reduces Symbolica time by 42.06%; all four asymmetric GCD operations now favor Symbolica |
| Bounded automatic bivariate retries, PolyBench 5v `0004` / `0008` distributions | stack overflow / stack overflow | 0.266315 / 1.364144 | Iterative retries and a one-shot univariate crossover complete all 400 measured problems; the first setup is 3.76x faster than FLINT by paired median |
| Batched Vandermonde inversion and reusable last-variable images, PolyBench 5v GCD #11 | 1.121593 | 0.978321 | One inversion replaces linear-many finite-field inversions, and repeated images reuse row/power/output storage; Symbolica is now 1.02x faster than FLINT |
| Four-way direct `DenseZp64` accumulation, near-`2^64` dense-large multiplication | 1.137177 | 0.992282 | Source-matched Symbolica median falls 14.58%; the kernel now overlaps four independent carry chains |
| Balanced two-leaf Hensel exact reconstruction, dense degree 65 | 1.171321 | 1.093223 | Symbolica median falls 6.7%; the exact pair leaves `p^51` rather than `p^86` root precision |
| Pre-content Hu main-variable planning, PolyBench 8v sharp GCD #140 / #11 | 1.268240 / 0.748107 | 0.479027 / 0.409963 | Both schedules use 16 rather than 80 images; Symbolica is now 2.09x and 2.44x faster than FLINT |
| Fixed-width compact-simplex integer accumulation, dense 8v / dense 7v | 1.610087 / 1.584134 | 0.326626 / 0.423081 | Symbolica medians are now 3.06x and 2.36x faster than FLINT |
| Chunked mixed-radix dense-five degree-7 multiplication | 1.679650 | 0.748808 | Symbolica median reduced 56.27%; now 1.34x faster than FLINT |
| Cumulative bivariate reconstruction, Wang lifting, dense Montgomery products, and coefficient-content certificates, PolyBench 5v #131 | 1.745716 | 0.256988 | Symbolica median reduced 85.3%; now 3.89x faster than FLINT |
| Dense large-modulus Montgomery products, PolyBench 5v #131 | 0.345860 | 0.276987 | Source-matched Symbolica median reduced 19.90% in commit `f14d182` |
| Bivariate reconstruction, Wang lift fast paths, and coefficient-content certificate, PolyBench 5v #32 | 1.124230 | 0.164792 | Symbolica median reduced 85.3%; now 6.07x faster than FLINT |
| Primitive bivariate dispatch, lazy exact-product exit, and coefficient-content certificates, PolyBench 5v #159 | 0.991235 | 0.586252 | Symbolica is now 1.71x faster than FLINT |
| Wide packed `u16` row merge, high-gap eight-variable degree 4 | 1.654253 | 0.524530 | Symbolica median reduced 68.9%; now 1.91x faster than FLINT |
| Few-row packed merge, sparse eight-variable degree 5 | 1.340679 | 0.552281 | Symbolica median reduced about 58.9%; the consistent full-sweep inventory is 0.780017 S/F |
| Initial primitive bivariate sample, direct reconstruction, and coefficient-content certificate, dense three-variable degrees 6/5 | 1.747643 | 0.554623 | Symbolica median is 68.3% below the original ratio |
| Dense-state quadratic Hensel lifting, high-height degree 33 | 1.997930 | 0.986543 | Symbolica is now slightly faster than FLINT; the exact residual and corrections remain dense across all p-adic rounds |
| Fixed-width contiguous Kronecker packing and native bound sizing, dense degree-64 product | 1.094628 | 1.027605 | The original packing gain is retained, and native bound sizing cuts another source-matched 2.7% from Symbolica time |
| Sparse high-gap five-variable product heap advance | 1.040981 | 0.799572 | S/F reduced about 23.2% |
| PolyBench factorization #84 variable order and coefficient-content certificate | 8.690880 | 0.421076 | Current S/F is 20.64x below the original ratio |
| Guarded geometric prime sampling and deferred Bezout updates, dense degree-64 factorization | 1.748240 | 0.853381 | Current final artifact is 1.17x faster than FLINT |
| Constant-coordinate reconstruction, dense univariate GCD degree 64 | 1.161142 | 0.853611 | Source-matched Symbolica median reduced about 26.5%; now 1.17x faster than FLINT |

The high-height degree-33 product itself is `0.580344` S/F and final factorization is `0.958149`,
so both construction and the complete operation are now faster than FLINT.

## Appendix note: historical resultant backend measurements

No resultant measurement participates in the primary scoreboard. For historical algorithm
reference only, these 18 frozen `v5` measurements repeat the same six workloads for three
Symbolica backends.

| Workload | Ducos S/F | Brown S/F | CRT S/F |
|---|---:|---:|---:|
| dense outer degrees 7/6 | 0.519856 | 1.333630 | 2.059289 |
| nonunit leading degrees 9/7 | 0.554473 | 1.124006 | 1.129343 |
| dense outer degrees 10/8 CRT crossover | 0.932951 | 2.464208 | 0.607374 |
| outer-sparse degrees 12/9 CRT crossover | 1.027247 | 1.862295 | 0.432838 |
| large high-height degrees 14/10 | 0.861806 | 2.581685 | 1.249141 |
| lacunary outer degrees 18/11 | 0.570573 | 1.438647 | 2.287301 |

In this frozen `v5` sample, Ducos is faster than FLINT on five of six inputs, with median
`0.716190` S/F. Brown is an explicit reference alternative and is slower on all six; CRT wins two
of six. FLINT's resultant and Symbolica's default both use a Ducos-family recurrence, so the Brown
rows are not default-path performance gaps.

## Versioning and update protocol

`CURRENT_STATUS.md` is the live latest view. Every accepted refresh must also create the next
immutable `CURRENT_STATUS_v<i>.md` snapshot. Never edit an older numbered snapshot. The first
paragraph of each new snapshot must identify the preceding version and summarize every affected
case or family with its old and new S/F values.

After every accepted performance change:

1. Rerun every affected case sequentially with the current source-matched Symbolica/FLINT binary.
2. For repeated processes, replace the row with the median process-level S/F. Do not retain an old
   candidate or control as another current row.
3. Merge the 125 primary non-resultant rows, sort them numerically by S/F in descending order, and
   renumber them. Keep the 18 resultant-backend references in the appendix only.
4. Recompute the summary and verify 125 unique primary workload/category pairs, 18 appendix
   measurements arranged as six workloads by three algorithms, and monotonic descending primary
   ratios.
5. Create the next numbered snapshot with its opening delta paragraph, then make
   `CURRENT_STATUS.md` mirror that latest snapshot.
6. Record benchmark binary hashes, raw timing artifacts, profiles, accepted changes, and rejected
   experiments in [PolynomialPerformanceHandoff.md](PolynomialPerformanceHandoff.md).

The `v20` inventory was programmatically checked for 125 unique primary rows, 116 rows faster than
FLINT, nine rows slower than FLINT, median `0.636796`, and monotonically descending S/F values. The
separate full-distribution table contains 16 setups and 3,200 validated paired measurements. The
18 resultant measurements remain six appendix workloads by three explicit algorithms.
