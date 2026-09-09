# Reuse ordinary balanced surveys and test mixed-variable IBP coefficients

Automatic selection used to discard its degree-survey slices when it chose
ordinary balanced Zippel for more than two variables. The subsequent
reconstruction repeated those rows. It now reuses the first slice as the initial
interpolant and the other available slices as geometric row one at the same
anchor. This also works when a sparse slice ends the survey early.

The slices are consumed once and cleared before another attempt. They are
stored only after the first-variable slice succeeds, so a failed survey cannot
mix a saved row with an unrelated fallback anchor. Method selection, sparse-row
validation, full-dimensional verification, and fresh-prime checks keep their
existing settings. The selector receives only oracle values.

## Probe reductions

The baseline is `a7b9ee79`, saved before editing. All counts below include
selection, interpolation and verification calls. The modular runs use the same
63-bit prime and cached-power oracle as the preceding comparisons.

| Input | Baseline probes | Current probes | Reduction |
|---|---:|---:|---:|
| FireFly f1, 20 variables | 1,945 | 1,858 | 4.5% |
| FireFly f2, high degree | 1,478 | 1,264 | 14.5% |
| Three-variable early survey exit | 279 | 192 | 31.2% |
| Three-variable later survey exit | 246 | 154 | 37.4% |

Counts agree across seeds 1–3. The two small controls are respectively
`(x^20+y^20+z^20)/(x^20-y^20+2*z^20)` and
`(x+y^20+z^20)/(x-2*y^20+3*z^20)`; their inputs and hashes are archived.
The existing larger controls retain their probe counts. The full-Q four-loop
coefficient remains at 71,019, the amplitude variants at 43,957 and 43,955,
and modular f3/f4 at 26,702 and 53,531.

## Harder coordinate controls from public IBP coefficients

`shear_ibp_inputs.py` takes the first four source-length ranks in the pinned
complete `nb0` table and substitutes `d -> d+u`, retaining variable order
`u,v,w,d`. The inverse is `d -> d-u`. This deliberately mixes the previously
separated dimension variable with a kinematic variable. These are controlled
reparameterizations of existing public IBP coefficients, not four independent
IBP families. Selection is fixed before measuring any transformed result.

The source revision and hashes remain those in
[the full-table report](reconstruction-factor-lift.md). The exporter verifies
the original coefficient hashes and records the transformation and output hashes.
Independent FLINT checks confirm both the forward export and the inverse
identity for all four coefficients.

| Original source rank | Expanded numerator + denominator terms | Symbolica | FireFly default | FireFly scan | FIRE7 learned | rare |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 5,148 + 2,070 | 15,297 | 40,327 | 29,895 | 45,109 | 57,888 |
| 2 | 5,247 + 2,070 | 16,138 | 41,106 | 30,437 | 47,200 | 58,995 |
| 3 | 4,488 + 1,215 | 12,535 | 30,685 | 25,192 | 39,787 | 49,486 |
| 4 | 4,312 + 1,215 | 11,671 | 29,546 | 23,478 | 37,276 | 45,527 |

These are complete Q reconstruction counts. Every method passes exact identity
checking on every transformed case. Symbolica selects pruned Cuyt–Lee on all
four, so the survey-reuse change does not alter these counts. It uses about
half as many probes as the better FireFly setting and fewer than both other
implementations. Symbolica and scanned FireFly are additionally measured with
seeds 1–3. Native primes, coefficient lifting and confirmation counts remain
different between libraries, as documented in the
[scaling reference report](reconstruction-scaling-reference.md). No support,
degrees, factors, or transformation metadata are supplied to reconstruction.

## Validation and reproduction

All 41 reconstruction tests pass. New controls require exact recovery within
the reduced 192-probe budget, reject support hidden on a reused slice and
recover with a fresh attempt, and recover within one attempt when every probe
of the first-variable survey is a pole. They check actual oracle accounting.

The final release reruns all 3,319 `nb0` and 945 `graph5` coefficients, including
the smaller cases that favor FireFly. Every exact reconstruction passes and
retains its baseline probe count, per-prime distribution and selected method;
totals remain 505,874 and 20,311 probes. All 4,333 archived benchmark runs pass
exact identity checks. Clippy completes with the existing allowance and 249
warnings; changed Rust files pass formatting checks.

Use the pinned [external setup](external/README.md), including its loader
environment where necessary. Build the release example with LTO disabled and
16 codegen units, then run:

```sh
python3 benches/external/shear_ibp_inputs.py
export BENCH_INPUT_DIR="$PWD/target/reconstruction-external/fire-table-inputs/sheared"
mapfile -t cases < "$BENCH_INPUT_DIR/suite-cases.txt"
BENCH_CPU=24 BENCH_METHODS=Automatic,FireFly_default,FireFly_scan,FIRE7_Q_learned,Rare_scaling \
  BENCH_TIMEOUT=180 PROCESS_TIMEOUT=300 \
  python3 benches/external/run_q_stress.py \
  target/reconstruction-external/sheared-reference.csv "${cases[@]}"
```

The [result archive](results/reconstruction/survey-reuse/) contains the raw
CSVs, exact-check outputs, manifests, hashes, regression comparisons and
validation logs. Timings are recorded but shared-host load is uncontrolled;
probe counts are the primary comparison.

The search for additional independent IBP families also located
[`magv/ibp-benchmark`](https://github.com/magv/ibp-benchmark/tree/f518de1f4f89cc716a31d5d9a9a9ba0a3b72f458),
the configuration generator associated with the
[Ratracer paper](https://arxiv.org/abs/2211.03572). Its pinned checkout contains
solver configurations rather than precomputed coefficient tables. Its unmodified
generator has produced 84 configurations covering seven families, with hashes
recorded in the archive. Those families have not yet been reconstructed with
Symbolica; they are not included in the performance claims above.
