#!/usr/bin/env bash
# Reproduce this experiment without changing the working tree.
# Run from a shell with the repository's Rust and native dependencies available.
# HOTPATH_CPU optionally pins timings; HOTPATH_PROFILE defaults to release.
set -euo pipefail
hotpath_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
hotpath_root=$(mktemp -d /tmp/symbolica-hotpaths.XXXXXX)
hotpath_profile=${HOTPATH_PROFILE:-release}
hotpath_passes=${HOTPATH_PASSES:-3}
case $hotpath_profile in
    release|bench) ;;
    *) printf 'HOTPATH_PROFILE must be release or bench\n' >&2; exit 1 ;;
esac
mkdir -p "$hotpath_root/tree" "$hotpath_root/bin" "$hotpath_root/log"
for hotpath_item in Cargo.toml build.rs License.md Readme.md src lib examples benches; do
    cp -a --reflink=auto "$hotpath_repo/$hotpath_item" "$hotpath_root/tree/"
done
if [[ -f $hotpath_repo/Cargo.lock ]]; then
    cp --reflink=auto "$hotpath_repo/Cargo.lock" "$hotpath_root/tree/"
fi
cd "$hotpath_root/tree"
export CARGO_TARGET_DIR="$hotpath_root/target"
export CARGO_BUILD_BUILD_DIR="$hotpath_root/target"
if [[ $hotpath_profile == bench ]]; then
    export CARGO_PROFILE_BENCH_DEBUG=0
fi
git apply --unidiff-zero --reverse --check benches/performance_hotpaths.patch
git apply --unidiff-zero --reverse benches/performance_hotpaths.patch
for hotpath_variant in baseline optimized; do
    if [[ $hotpath_variant == optimized ]]; then
        git apply --unidiff-zero benches/performance_hotpaths.patch
    fi
    touch src/normalize.rs src/expand.rs src/id.rs src/state.rs
    cargo build --profile "$hotpath_profile" --example performance_hotpaths \
        -j "${HOTPATH_BUILD_JOBS:-4}" > "$hotpath_root/log/build-$hotpath_variant.log" 2>&1
    cp "$CARGO_TARGET_DIR/release/examples/performance_hotpaths" "$hotpath_root/bin/$hotpath_variant"
done
hotpath_runner=()
if [[ -n ${HOTPATH_CPU:-} ]]; then
    hotpath_runner=(taskset -c "$HOTPATH_CPU")
fi
mapfile -t hotpath_cases < <(
    HOTPATH_MS=1 HOTPATH_SAMPLES=1 "${hotpath_runner[@]}" "$hotpath_root/bin/baseline" \
        | awk -F, 'NR > 1 {print $1}'
)
for ((hotpath_pass=1; hotpath_pass<=hotpath_passes; hotpath_pass++)); do
    for hotpath_variant in baseline optimized; do
        printf 'case,bytes,median_ns,min_ns,max_ns\n' \
            > "$hotpath_root/log/$hotpath_variant-$hotpath_pass.csv"
    done
    for ((hotpath_index=0; hotpath_index<${#hotpath_cases[@]}; hotpath_index++)); do
        hotpath_order=(baseline optimized)
        if (((hotpath_pass + hotpath_index) % 2 == 0)); then hotpath_order=(optimized baseline); fi
        for hotpath_variant in "${hotpath_order[@]}"; do
            HOTPATH_CASE="${hotpath_cases[hotpath_index]}" \
                "${hotpath_runner[@]}" "$hotpath_root/bin/$hotpath_variant" \
                | tail -n 1 >> "$hotpath_root/log/$hotpath_variant-$hotpath_pass.csv"
        done
    done
done
printf 'Binaries, build logs, and raw measurements: %s\n' "$hotpath_root"
