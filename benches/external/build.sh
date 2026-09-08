#!/usr/bin/env bash
# Linux build, using system FLINT >=3, GMP, MPFR and zlib development packages.
set -euo pipefail
cd "$(dirname "$0")/../.."
root=$PWD
external="$root/target/reconstruction-external"
mkdir -p "$external"
clone_at() {
    local url=$1 directory=$2 revision=$3
    if [[ ! -d "$directory/.git" ]]; then git clone "$url" "$directory"; fi
    if [[ $(git -C "$directory" rev-parse HEAD) != "$revision" ]]; then
        git -C "$directory" fetch origin "$revision"
        git -C "$directory" checkout --detach "$revision"
    fi
}
clone_at https://github.com/jklappert/FireFly "$external/firefly" 4ce258e5ace6361513c4bdaac93a247cc0e3fdbb
clone_at https://gitlab.srcc.msu.ru/feynmanintegrals/fire "$external/fire" d132e5365dd2a13db9cd9dbaf5c200b53d489cfd
patch_file="$root/benches/external/firefly-portability-seed.patch"
if git -C "$external/firefly" apply --check "$patch_file"; then
    git -C "$external/firefly" apply "$patch_file"
else
    git -C "$external/firefly" apply --reverse --check "$patch_file"
fi
git -C "$external/fire" submodule update --init FIRE7/extra/fuel
fuel="$external/fire/FIRE7/extra/fuel"
test "$(git -C "$fuel" rev-parse HEAD)" = 8627acefec5be237080e94c1ea903a52c680aa32
cmake -S "$external/firefly" -B "$external/firefly/build" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_BUILD_TYPE=Release -DWITH_FLINT=true \
    -DGMP_INCLUDE_DIR="$(pkg-config --variable=includedir gmp)" \
    -DGMP_LIBRARY="$(pkg-config --variable=libdir gmp)" \
    -DFLINT_INCLUDE_DIR="$(pkg-config --variable=includedir flint)" \
    -DFLINT_LIBRARY="$(pkg-config --variable=libdir flint)/libflint.so" \
    -DZLIB_INCLUDE_DIR="$(pkg-config --variable=includedir zlib)" \
    -DZLIB_LIBRARY="$(pkg-config --variable=libdir zlib)/libz.so"
cmake --build "$external/firefly/build" --target FireFly_static -j "${BUILD_JOBS:-4}"
read -r -a includes <<< "$(pkg-config --cflags flint zlib)"
read -r -a libraries <<< "$(pkg-config --libs flint zlib)"
# Explicit fmpq.h compensates for FUEL's reliance on older transitive includes.
cflags=(-O3 -std=c++17 -DENABLE_FLINT -fopenmp -ffunction-sections -fdata-sections -include flint/fmpq.h)
make -C "$fuel/library" -j "${BUILD_JOBS:-4}" ENABLE_FLINT=1 \
    CXX="${CXX:-g++}" CFLAGS="${cflags[*]} ${includes[*]}"
mkdir -p "$external/fire-objects"
for unit in reconstruction tools primes tables; do
    "${CXX:-g++}" "${cflags[@]}" -DMPRIME=1 "${includes[@]}" \
        -c "$external/fire/FIRE7/sources/tools/$unit.cpp" -o "$external/fire-objects/$unit.o"
done
"${CXX:-g++}" "${cflags[@]}" "${includes[@]}" \
    -I "$external/fire/FIRE7/sources/tools" benches/external/fire7.cpp \
    "$external"/fire-objects/*.o "$fuel/library/libfuel.a" \
    -Wl,--gc-sections "${libraries[@]}" -lgmpxx -ldl -o "$external/fire7-bench"
"${CXX:-g++}" -O3 -std=c++17 -pthread "${includes[@]}" \
    -I "$external/firefly/source/include" -I "$external/firefly/build/include" \
    benches/external/firefly.cpp "$external/firefly/build/libfirefly.a" \
    "${libraries[@]}" -lgmpxx -o "$external/firefly-bench"
