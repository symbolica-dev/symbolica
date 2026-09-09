#!/usr/bin/env bash
# Equation export and trace generation; no changes to Symbolica dependencies.
set -euo pipefail
cd "$(dirname "$0")/../.."
external="$PWD/target/reconstruction-external"
mkdir -p "$external"
clone_at() {
    local url=$1 directory=$2 revision=$3
    if [[ ! -d "$directory/.git" ]]; then git init "$directory"; fi
    if [[ $(git -C "$directory" rev-parse HEAD 2>/dev/null || true) != "$revision" ]]; then
        git -C "$directory" fetch --depth 1 "$url" "$revision"
        git -C "$directory" checkout --detach FETCH_HEAD
    fi
}
clone_at https://gitlab.com/kira-pyred/kira.git "$external/kira" aadf0671ed090b8427e14a3d908d6c7033d46b2c
clone_at https://github.com/magv/ratracer "$external/ratracer" 88646ca7b65c24bfce3a8be6e1093a9b89731f23
clone_at https://github.com/magv/firefly "$external/ratracer-firefly" 27d4bdec27436bbfb000ce47fd5dbc1ff08ae8a7
clone_at https://github.com/flintlib/flintxx "$external/ratracer-flintxx" 0be0a5f4da4dcf475eff00e6adbf5a728fb0153b
clone_at https://github.com/magv/ibp-benchmark "$external/ibp-benchmark" f518de1f4f89cc716a31d5d9a9a9ba0a3b72f458

meson_args=()
if [[ -f "$external/kira-build/build.ninja" ]]; then meson_args+=(--reconfigure); fi
meson setup "${meson_args[@]}" "$external/kira-build" "$external/kira" \
    -Dfirefly=false -Dflint=false --wrap-mode=nofallback
ninja -C "$external/kira-build" -j "${BUILD_JOBS:-6}"

# Match Ratracer's Makefile layout for the separately maintained FLINT C++ headers.
mkdir -p "$external/ratracer-include/flint"
for header in "$external/ratracer-flintxx"/src/{flintxx,flintxx_public}/*.h; do
    sed 's,"../flint.h",<flint/flint.h>,g' "$header" \
        > "$external/ratracer-include/flint/$(basename "$header")"
done
if [[ ! -e "$external/ratracer-include/flintxx" ]]; then
    ln -s flint "$external/ratracer-include/flintxx"
fi
flint_include=$(pkg-config --variable=includedir flint)
cmake -S "$external/ratracer-firefly" -B "$external/ratracer-firefly/build" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release \
    -DENABLE_SHARED=OFF -DENABLE_FF_INSERT=OFF -DENABLE_EXAMPLE=OFF \
    -DFLINT_INCLUDE_DIR="$flint_include" \
    -DFLINT_LIBRARY="$(pkg-config --variable=libdir flint)/libflint.so" \
    -DZLIB_INCLUDE_DIR="$(pkg-config --variable=includedir zlib)" \
    -DZLIB_LIBRARY="$(pkg-config --variable=libdir zlib)/libz.so" \
    -DCMAKE_CXX_FLAGS="${CXXFLAGS:-} $(pkg-config --cflags flint) -I$flint_include/flint -I$external/ratracer-include -include gmp.h"
cmake --build "$external/ratracer-firefly/build" --target FireFly_static -j "${BUILD_JOBS:-6}"
read -r -a rat_flags <<< "$(pkg-config --cflags --libs flint zlib)"
"${CXX:-g++}" -O3 -std=c++17 -fopenmp -include gmp.h \
    -I "$external/ratracer-firefly/source/include" \
    -I "$external/ratracer-firefly/build/include" \
    -I "$external/ratracer-include" -I "$flint_include/flint" \
    "$external/ratracer/ratracer.cpp" "$external/ratracer-firefly/build/libfirefly.a" \
    "${rat_flags[@]}" -lgmpxx -ldl -o "$external/ratracer-tool"
