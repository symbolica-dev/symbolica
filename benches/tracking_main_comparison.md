# Error tracking versus main and binary storage

Measured on 2026-10-23 with the same public-API workload in three checkouts:

- **Main:** `170ba932d0cf73508da34cc78fbe82361158147e`, existing hexadecimal float storage.
- **Binary:** `1e51900739f5e0f3271acfc9cba794f1833d8a18`, the main-based canonical binary-storage proposal without error tracking.
- **Tracked:** implementation commit `6efef83b472477bb51ae699fed774b7409d53690`. The executable was built from its candidate working tree based on `238be647928ec5b7fb14cb773c187b6e0ef11d42`, before final commit. The measured coefficient arithmetic and packing code was unchanged between that build and the commit. Executable hashes identify the measured snapshots in `tracking_main_comparison_provenance.json`.

This distinguishes the net effect of the proposal against current main from the incremental cost of uncertainty relative to binary storage alone. These are fixed coefficient microbenchmarks, not whole-application measurements or a test of nonlinear-function throughput.

## Method

The shared example is `examples/tracking_main_comparison.rs`. Both operands have real parts `1/7` and `3/7`; complex cases additionally have imaginary parts `2/11` and `4/11`. The divisions occur at the requested 53, 256 or 4096 bits before timing. Before each executable, a two-second busy loop warms CPU 2. Each operation then has 100 warmup iterations followed by seven samples of 20,000 iterations. The CSV records the median nanoseconds per operation. The clock is Linux `CLOCK_THREAD_CPUTIME_ID`; processes are pinned to CPU 2.

Construction includes cloning the source coefficient and packing an atom. Decoding materializes an owned coefficient from a borrowed packed view. Coefficient arithmetic includes operand cloning. Atom arithmetic includes its normal decoding, normalization and repacking. Cancellation differs semantically: tracking retains propagated uncertainty, whereas main returns an exact zero.

Builds use `cargo build --offline --profile bench --example tracking_main_comparison`, `CARGO_BUILD_JOBS=3`, the repository's bench profile (optimization 3, debug information, 16 codegen units, no LTO, incremental enabled), and identical Cargo.lock package versions/checksums. Default features select MPFR/GMP. Precision labels are requested bits; Astro internally pads to word precision, so absolute MPFR-versus-Astro timings are not a comparison at identical effective bit counts. Astro uses:

```text
--no-default-features --features tracing_max_level_info,faster_alloc,integer-malachite,float-astro,native_code_generation
```

The system Cargo configuration adds `-C target-cpu=native -C link-arg=-fuse-ld=mold`. The machine is an AMD Ryzen 9 5900X, with the powersave governor and boost enabled, running Rust 1.99.0 (`b940084d7`, LLVM 23.1.1). The allocator is the default enabled mimalloc feature. Tracked builds use a separate target directory. When reusing a target across baseline worktrees, workspace-crate artifacts were explicitly cleaned to prevent stale cross-worktree reuse; packed sizes were checked in each saved executable.

The host was **not isolated**: another worktree was compiling during the measurements. Thread CPU time excludes descheduling but cannot remove frequency, cache or memory-bandwidth effects. The final six executable runs were serial and began after this team's test builds finished. An earlier pass is retained separately in `tracking_main_comparison_first_pass.csv`; the final pass adds the CPU warmup after an early low-precision construction outlier. Other worktree builds continued. Treat small timing differences cautiously; no significance or application-level throughput claim is made.

For memory, each of the 36 backend/precision/kind/revision combinations uses seven fresh processes retaining 200,000 independently constructed atoms. Reported RSS is the median increase in `/proc/self/status` `VmRSS`. Vector capacity is reserved before the initial reading, but pages touched while filling it, allocation size classes/slack and transient allocator caches still contribute. RSS bytes per atom are **not** the serialized size or an allocator call count. Packed bytes are the exact length of one atom's raw encoding.

Reproduce timing for a saved executable with:

```sh
NUMERICA_BENCH_CLOCK=thread taskset -c 2 ./tracking_main_comparison label 20000
```

Reproduce one memory sample with:

```sh
taskset -c 2 ./tracking_main_comparison label memory 256 real 200000
```

Raw timing medians are in `tracking_main_comparison_{main,binary,tracked}_{mpfr,astro}.csv`. All 252 individual RSS observations are in `tracking_main_comparison_memory.csv`, with medians and ranges in `tracking_main_comparison_memory_summary.csv`. Full per-case timing ratios are in `tracking_main_comparison_ratios.csv`.

## Runtime results

These are unweighted geometric means of ratios across the six precision/type cases per backend and operation. A ratio below 1 means less time; this is a descriptive microbenchmark summary, not a weighted application speedup. Full nanosecond measurements and all individual ratios remain in the CSV files.

| Backend | Operation | Binary / main | Tracked / main | Tracked / binary |
|---|---|---:|---:|---:|
| MPFR | construct | 0.42× | 0.43× | 1.01× |
| MPFR | decode | 0.17× | 0.20× | 1.21× |
| MPFR | coefficient_add | 0.97× | 1.33× | 1.37× |
| MPFR | coefficient_mul | 1.01× | 1.53× | 1.51× |
| MPFR | atom_add | 0.22× | 0.26× | 1.19× |
| MPFR | atom_mul | 0.30× | 0.38× | 1.28× |
| MPFR | atom_div | 0.35× | 0.45× | 1.29× |
| MPFR | atom_cancel | 0.29× | 0.34× | 1.15× |
| ASTRO | construct | 0.29× | 0.23× | 0.81× |
| ASTRO | decode | 0.17× | 0.18× | 1.05× |
| ASTRO | coefficient_add | 1.02× | 1.11× | 1.09× |
| ASTRO | coefficient_mul | 0.93× | 0.98× | 1.05× |
| ASTRO | atom_add | 0.30× | 0.31× | 1.03× |
| ASTRO | atom_mul | 0.44× | 0.44× | 1.01× |
| ASTRO | atom_div | 0.55× | 0.64× | 1.17× |
| ASTRO | atom_cancel | 0.43× | 0.71× | 1.65× |

The aggregate gains do not apply to every case. The largest measured regressions against main and against binary storage are shown below; the two lists may overlap. Cancellation also retains different metadata, as explained above.

| Compared with | Backend | Bits | Kind | Operation | Baseline ns | Tracked ns | Tracked / baseline |
|---|---|---:|---|---|---:|---:|---:|
| main | MPFR | 53 | complex | coefficient_mul | 260.5 | 711.7 | 2.73× |
| main | MPFR | 256 | complex | coefficient_mul | 352.3 | 793.7 | 2.25× |
| main | MPFR | 53 | complex | coefficient_add | 159.7 | 236.6 | 1.48× |
| binary | ASTRO | 4096 | real | atom_cancel | 7201.7 | 21011.6 | 2.92× |
| binary | MPFR | 53 | complex | coefficient_mul | 262.7 | 711.7 | 2.71× |
| binary | ASTRO | 4096 | complex | atom_cancel | 9878.3 | 24525.1 | 2.48× |

Across all 288 timing medians, the final/first-pass ratio ranges from 0.48× to 1.94×; the median absolute change is 2.0% and the 90th percentile is 8.1%. This spread is direct evidence of shared-host noise. Interpret the small overhead estimates as approximate; both passes are retained.


## Storage and retained memory

Entries list **main / binary / tracked**. Packed bytes are exact. RSS values are observed medians divided by 200,000 atoms; the raw sample ranges are included in the summary CSV.

| Backend | Bits | Kind | Packed bytes: main / binary / tracked | RSS bytes per atom: main / binary / tracked |
|---|---:|---|---:|---:|
| MPFR | 53 | real | 45 / 18 / 26 | 94.4 / 62.9 / 62.9 |
| MPFR | 53 | complex | 62 / 31 / 47 | 94.4 / 62.9 / 83.9 |
| MPFR | 256 | real | 96 / 43 / 51 | 230.7 / 83.9 / 94.4 |
| MPFR | 256 | complex | 164 / 81 / 97 | 231.0 / 125.8 / 146.8 |
| MPFR | 4096 | real | 1056 / 524 / 532 | 2653.2 / 681.7 / 681.7 |
| MPFR | 4096 | complex | 2084 / 1043 / 1059 | 2653.2 / 1321.7 / 1321.7 |
| ASTRO | 53 | real | 49 / 19 / 27 | 115.3 / 62.9 / 62.9 |
| ASTRO | 53 | complex | 69 / 33 / 49 | 115.3 / 83.9 / 94.4 |
| ASTRO | 256 | real | 97 / 43 / 51 | 230.7 / 83.9 / 94.4 |
| ASTRO | 256 | complex | 166 / 81 / 97 | 230.7 / 125.8 / 146.8 |
| ASTRO | 4096 | real | 1057 / 524 / 532 | 2663.7 / 681.7 / 681.7 |
| ASTRO | 4096 | complex | 2086 / 1043 / 1059 | 2663.7 / 1321.3 / 1321.3 |

Tracking uses 24–50% fewer packed bytes than current main in these cases, despite adding an 8-byte error magnitude for each stored component. Relative to binary storage alone, tracking adds 8 bytes for a real coefficient or 16 for a complex coefficient. The exact zero imaginary component remains omitted. Measured retained RSS decreases by 11–74% versus main; allocation size classes explain why some 8- or 16-byte encoded-size changes do not change RSS.

The binary representation accounts for most of the packing, decoding and atom-level gains. Uncertainty propagation still costs arithmetic and metadata work, especially in small complex coefficients and cancellation. These measurements support the storage/representation improvements while leaving nonlinear evaluation and application-level performance as separate questions.
