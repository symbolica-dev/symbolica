#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
external="$PWD/target/reconstruction-external"
test "$(git -C "$external/firefly" rev-parse HEAD)" = 4ce258e5ace6361513c4bdaac93a247cc0e3fdbb
read -r -a includes <<< "$(pkg-config --cflags flint zlib)"
read -r -a libraries <<< "$(pkg-config --libs flint zlib)"
"${CXX:-g++}" -O3 -std=c++17 -pthread "${includes[@]}" \
    -I "$external/firefly/source/include" -I "$external/firefly/build/include" \
    benches/external/firefly_joint_stress.cpp "$external/firefly/build/libfirefly.a" \
    "${libraries[@]}" -lgmpxx -ldl -o "$external/firefly-joint-stress"
