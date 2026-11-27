#!/usr/bin/env bash
# A/B for the divergence fix on one binary: alternate --no-sort and sorted sweeps so
# clock/thermal drift hits both arms equally. Prints sim_shared ms per run.
# Usage: scripts/ab-sort.sh [build-dir] [grid]     env: RUNS=5 DATA=data/nq_2019_2026.gbt
set -euo pipefail
BUILD=${1:-$HOME/build}
GRID=${2:-default}
SRC=$(cd "$(dirname "$0")/.." && pwd)
DATA=${DATA:-$SRC/data/nq_2019_2026.gbt}
DATA_ARG=()
[ -f "$DATA" ] && DATA_ARG=(--data "$DATA")
k() { "$BUILD/gbt" sweep "${DATA_ARG[@]}" --grid "$GRID" --top 1 "$@" | awk '$1=="sim_shared"||$1=="sim_grouped"{t+=$4} END{print t}'; }
"$BUILD/gbt" sweep "${DATA_ARG[@]}" --grid "$GRID" --top 1 >/dev/null  # warm-up
for i in $(seq "${RUNS:-5}"); do
  printf 'run %d  grid-order %8s ms   sorted %8s ms\n' "$i" "$(k --no-sort)" "$(k)"
done
# Same answers, not just faster: full-grid checksum in each order.
sum() { "$BUILD/gbt" bench "${DATA_ARG[@]}" --grid "$GRID" --variants 100000000 --backends gpu --json "$@" |
        sed -n 's/.*"seconds":\([0-9.]*\).*"checksum":"\([0-9a-f]*\)".*/\2  (\1 s end to end)/p'; }
echo "checksum grid-order $(sum --no-sort)"
echo "checksum sorted     $(sum)"
