#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
external="$PWD/target/reconstruction-external"
test "$(git -C "$external/ratracer" rev-parse HEAD)" = 88646ca7b65c24bfce3a8be6e1093a9b89731f23
read -r -a flags <<< "$(pkg-config --cflags --libs flint)"
"${CXX:-g++}" -O3 -std=c++17 -fPIC -shared -include gmp.h \
    -I "$external/ratracer" benches/external/trace_oracle.cpp \
    "${flags[@]}" -o "$external/libtrace-oracle.so"
