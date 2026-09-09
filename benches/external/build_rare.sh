#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
external="$PWD/target/reconstruction-external"
mkdir -p "$external"
# The pinned GMP configure checks predate C23's strict empty parameter lists.
CFLAGS="${CFLAGS:--O3} -std=gnu17" \
cargo build --locked --release --manifest-path benches/external/rare/Cargo.toml \
    --target-dir "$external/rare-target"
cp "$external/rare-target/release/symbolica-rare-benchmark" "$external/rare-q-stress"
read -r -a includes <<< "$(pkg-config --cflags flint)"
read -r -a libraries <<< "$(pkg-config --libs flint)"
"${CXX:-g++}" -O3 -std=c++17 "${includes[@]}" benches/external/check_q_result.cpp \
    "${libraries[@]}" -lgmpxx -o "$external/check-q-result"
