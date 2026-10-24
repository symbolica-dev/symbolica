# Scoped library-unlock overhead

The ignored unit benchmark in `src/license/unlock_benchmark.rs` compares the actual
scoped implementation with a benchmark-only copy of the previous implementation
from commit `1a1829b1f3d0a88b53b44085b3b9f34ef77945b3`. The baseline retains its
Boolean capture token, TLS vector of `Arc<AtomicBool>`, fresh allocation on worker
activation, and guard removal on drop. It uses separate TLS so it cannot authorize
the implementation being measured.

Run on an otherwise idle machine using the repository's optimized benchmark profile:

```sh
cargo test --profile bench --features python_export --lib scoped_library_unlock_overhead --no-run
taskset -c 2 cargo test --profile bench --features python_export --lib license::unlock_benchmark::scoped_library_unlock_overhead -- --ignored --exact --nocapture --test-threads=1
```

Omit `taskset -c 2` on platforms without Linux CPU affinity, or choose an available
CPU. Build before applying affinity so compilation can use all available cores.
The benchmark profile uses optimization level 3, 16 codegen units, LTO off, and
debug information. The `python_export` feature matches the recorded build; the timed paths do not
enter Python. The benchmark uses synthetic enabled state, requires no key, and
performs no license-server requests.

Each result is the median of 11 batches. Baseline and scoped batches alternate in
execution order, with 10,000 warm-up iterations before sampling. Hot checks use
10,000,000 iterations per batch; activation and capture use 500,000. The emitted
CSV also includes sample minima and maxima. `std::hint::black_box` consumes each
result; loop and black-box overhead remain in both measurements.

The cases separate costs that occur at different frequencies:

- `direct_hot_check`: TLS authorization checks under a direct library guard.
- `capture_and_exit`: capture existing caller authorization, then finish an empty
  operation. The old snapshot has no scope state to allocate or invalidate.
- `inherited_hot_check`: TLS authorization checks with only inherited state on a
  worker. Activation is outside the timed loop.
- `worker_activation_check_drop`: activate inherited state, check once, and remove
  it. This includes the old implementation's allocation on every activation.
- `nested_inherited_hot_check`: a second-generation worker's check, with source
  scope ancestry preserved by the scoped implementation. The old Boolean token
  discarded that ancestry.
- `two_source_inherited_hot_check`: a worker inheriting two independently enabled
  library registrations. Either source can authorize it.

Thread creation, joins, and dispatch are outside the timed regions. Separate
worker threads prevent a caller's direct guard from masking the inherited path.
This is a bookkeeping microbenchmark; it does not predict end-to-end Symbolica
workload speed.

## Concurrent worker activation

A separate ignored benchmark measures shared-reference-count contention:

```sh
taskset -c 2,3,4,5 cargo test --profile bench --features python_export --lib license::unlock_benchmark::scoped_library_unlock_contended_activation -- --ignored --exact --nocapture --test-threads=1
```

Choose four distinct physical cores for the affinity mask (`lscpu -e=CPU,CORE` on
Linux), or omit affinity and allow the scheduler to use the available cores.
Do not run this case with the single-CPU mask from the preceding benchmark.
Four workers synchronize before starting; spawning and joining are outside the
timed interval, while the fixed start/finish barrier cost is included. Results
are elapsed wall-clock nanoseconds divided by the total number of activations
across all workers: an aggregate-throughput measure, not per-worker latency.

The two cases perform one or 1,000 hot checks per activation, using 250,000 or
2,000 activations per worker per sample respectively. They use the same 11-sample
alternating-order protocol. The new implementation's workers share a scope Arc;
the old implementation allocates a separate Arc for every activation. Batching
therefore matters when amortizing the new shared reference-count updates.

## Results

Measured on 2026-10-24 with an AMD Ryzen 9 5900X (12 physical cores / 24 threads),
Linux 7.2.11-arch1-1, and `rustc 1.99.0 (b940084d7 2026-09-28)`. The build enabled
`python_export` and the default features, including mimalloc, with the repository's
benchmark profile and local Rust flags `-C target-cpu=native -C link-arg=-fuse-ld=mold`.
CPU frequency boost remained enabled. No Cargo or rustc processes ran during
measurement. The compiled test executable was invoked directly using the filters
shown above: CPU 2 for uncontended tests, CPUs 2–5 (four distinct physical cores)
for concurrent tests. Individual workers were scheduled within that four-core mask.

The table reports the first run's medians in nanoseconds per operation:

| Operation | Previous | Scoped | Change |
| --- | ---: | ---: | ---: |
| Direct library check | 2.608 | 3.276 | +0.668 ns |
| Inherited worker check | 2.611 | 3.717 | +1.106 ns |
| Worker activation + one check + drop | 16.478 | 14.631 | −1.847 ns |
| Capture + scope exit on caller | 2.820 | 45.287 | +42.467 ns |
| Second-generation inherited check | 2.644 | 4.164 | +1.520 ns |
| Inherited check with two library sources | 2.649 | 6.203 | +3.554 ns |

The ordinary inherited check adds about 1.1 ns in this configuration. A new scope
also incurs a one-time capture/cleanup cost on the caller. Uncontended worker
activation is slightly faster because it reuses shared scope state instead of
allocating an enabled flag on every activation. These are measurements of the
specific operations above, not estimates of overall application speed.

Shared state changes the concurrent activation tradeoff:

| Four-worker operation | Previous aggregate ns/activation | Scoped aggregate ns/activation | Ratio |
| --- | ---: | ---: | ---: |
| Activation + 1 check + drop | 6.239 | 52.322 | 8.387× |
| Activation + 1,000 checks + drop | 626.620 | 1205.406 | 1.924× |

The second row is equivalent to 0.627 versus 1.205 aggregate wall-clock ns/check.
These throughput figures must not be read as individual worker latencies.
Concurrent one-check jobs expose contention on the shared scope's atomic reference
count. Wrapping a whole batch in `run` amortizes activation; Symbolica's internal
Rayon paths retain activation per folder, rather than adding it to every term.
Hot authorization checks read atomic flags and do not update the reference count.

A second complete run confirmed the uncontended inherited-check delta
(2.617 → 3.727 ns). Concurrent medians were noisier: one-check activations measured
6.915 → 52.950 aggregate ns, and 1,000-check activations measured
756.097 → 1288.767 aggregate ns. Both runs are included below; the result tables
consistently use the first run rather than selecting favorable samples.

### Raw batch summaries

Each row contains the median, minimum, and maximum across 11 timed batches.
`iterations` is the per-worker count for four-worker cases, and the single-thread
count otherwise. The ratios and ranges were printed directly by the harness.

```csv
run,case,iterations,legacy_median_ns,scoped_median_ns,ratio,legacy_min_ns,legacy_max_ns,scoped_min_ns,scoped_max_ns
primary,direct_hot_check,10000000,2.608,3.276,1.256,2.605,3.056,3.269,3.679
primary,capture_and_exit,500000,2.820,45.287,16.057,2.803,2.842,44.959,45.675
primary,inherited_hot_check,10000000,2.611,3.717,1.423,2.606,3.279,3.712,3.911
primary,worker_activation_check_drop,500000,16.478,14.631,0.888,16.404,17.274,14.598,14.969
primary,nested_inherited_hot_check,10000000,2.644,4.164,1.575,2.604,2.661,4.140,4.200
primary,two_source_inherited_hot_check,10000000,2.649,6.203,2.341,2.616,2.656,6.164,6.216
primary,four_workers_activation_1_checks,250000,6.239,52.322,8.387,4.167,9.874,49.430,54.220
primary,four_workers_activation_1000_checks,2000,626.620,1205.406,1.924,609.763,967.871,1128.862,1576.503
repeat,direct_hot_check,10000000,2.617,3.279,1.253,2.610,2.719,3.257,3.428
repeat,capture_and_exit,500000,2.820,45.086,15.987,2.806,2.939,44.899,45.204
repeat,inherited_hot_check,10000000,2.617,3.727,1.424,2.606,2.647,3.711,3.746
repeat,worker_activation_check_drop,500000,16.466,14.635,0.889,16.450,16.613,14.619,27.650
repeat,nested_inherited_hot_check,10000000,2.612,4.125,1.579,2.598,2.971,4.098,7.062
repeat,two_source_inherited_hot_check,10000000,2.612,6.122,2.344,2.601,2.624,6.108,6.137
repeat,four_workers_activation_1_checks,250000,6.915,52.950,7.657,4.142,7.859,44.459,53.505
repeat,four_workers_activation_1000_checks,2000,756.097,1288.767,1.704,614.941,1266.967,1209.431,1749.377
```
