#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p target/reconstruction-external
fetch() {
    local url=$1 directory=$2 revision=$3
    if [[ ! -d "$directory/.git" ]]; then git clone "$url" "$directory"; fi
    if [[ $(git -C "$directory" rev-parse HEAD) != "$revision" ]]; then
        git -C "$directory" fetch origin "$revision"
        git -C "$directory" checkout --detach "$revision"
    fi
}
fetch https://github.com/a-maier/scaling-rec target/reconstruction-external/scaling-rec e79e8886f9a50a577029f7fe98dc538ab5b6e0f3
fetch https://github.com/jklappert/FireFly target/reconstruction-external/firefly 4ce258e5ace6361513c4bdaac93a247cc0e3fdbb
sha256sum -c benches/external/stress-inputs.sha256
