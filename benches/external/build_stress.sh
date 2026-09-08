#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
external="$PWD/target/reconstruction-external"
if [[ ! -f "$external/firefly/build/libfirefly.a" || ! -f "$external/fire-objects/reconstruction.o" ]]; then
    bash benches/external/build.sh
fi
test "$(git -C "$external/firefly" rev-parse HEAD)" = 4ce258e5ace6361513c4bdaac93a247cc0e3fdbb
test "$(git -C "$external/fire" rev-parse HEAD)" = d132e5365dd2a13db9cd9dbaf5c200b53d489cfd
read -r -a includes <<< "$(pkg-config --cflags flint zlib)"
read -r -a libraries <<< "$(pkg-config --libs flint zlib)"
"${CXX:-g++}" -O3 -std=c++17 -pthread "${includes[@]}" \
    -I "$external/firefly/source/include" -I "$external/firefly/build/include" \
    benches/external/firefly_stress.cpp "$external/firefly/build/libfirefly.a" \
    "${libraries[@]}" -lgmpxx -o "$external/firefly-stress"
"${CXX:-g++}" -O3 -std=c++17 -pthread "${includes[@]}" \
    -I "$external/firefly/source/include" -I "$external/firefly/build/include" \
    benches/external/firefly_q_stress.cpp "$external/firefly/build/libfirefly.a" \
    "${libraries[@]}" -lgmpxx -o "$external/firefly-q-stress"
fuel="$external/fire/FIRE7/extra/fuel"
"${CXX:-g++}" -O3 -std=c++17 -DENABLE_FLINT -fopenmp \
    -ffunction-sections -fdata-sections -include flint/fmpq.h "${includes[@]}" \
    -I "$external/fire/FIRE7/sources/tools" benches/external/fire7_stress.cpp \
    "$external"/fire-objects/*.o "$fuel/library/libfuel.a" \
    -Wl,--gc-sections "${libraries[@]}" -lgmpxx -ldl -o "$external/fire7-stress"
