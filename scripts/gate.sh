#!/usr/bin/env bash
# The gate every change must pass before it is committed.
#   1. build + all tests (scripts/build.sh)
#   2. bench: every backend on the same subset must agree to the bit with cpu1
#   3. full default sweep on real data, GPU timed (best of N runs)
#   4. one line appended to results/perf-log.tsv so speed is tracked commit to commit
# Usage: scripts/gate.sh [NVIDIA|AMD|OFF] [build-dir]      env: RUNS=5 DATA=data/nq_2019_2026.gbt
set -euo pipefail
GPU=${1:-NVIDIA}
BUILD=${2:-$HOME/build}
SRC=$(cd "$(dirname "$0")/.." && pwd)
RUNS=${RUNS:-5}
DATA=${DATA:-$SRC/data/nq_2019_2026.gbt}
LOG=$SRC/results/perf-log.tsv

echo "== 1/4 build + tests"
bash "$SRC/scripts/build.sh" "$GPU" "$BUILD"
GBT=$BUILD/gbt

DATA_ARG=()
[ -f "$DATA" ] && DATA_ARG=(--data "$DATA") || echo "   (no $DATA: synthetic sessions)"

echo "== 2/4 bench: backends must agree to the bit"
BACKENDS=cpu1,cpu
[ "$GPU" != "OFF" ] && BACKENDS=cpu1,cpu,gpu-naive,gpu
"$GBT" bench "${DATA_ARG[@]}" --variants 4096 --backends "$BACKENDS" | tee /tmp/gbt-bench.txt
grep -q "agree to the bit: yes" /tmp/gbt-bench.txt || { echo "GATE FAIL: checksum mismatch"; exit 1; }

SWEEP_S=-
KERNEL_MS=-
SWEEP_SUM=-
if [ "$GPU" != "OFF" ]; then
  echo "== 3/4 full default sweep, best of $RUNS"
  best=999
  kbest=999999
  for i in $(seq "$RUNS"); do
    "$GBT" sweep "${DATA_ARG[@]}" --grid default --top 1 > /tmp/gbt-sweep.txt 2>/dev/null
    s=$(sed -n 's/.* in \([0-9.]*\)s = .*/\1/p' /tmp/gbt-sweep.txt)
    k=$(awk '$1=="sim_shared"||$1=="sim_grouped"{t+=$4} END{print t}' /tmp/gbt-sweep.txt)
    best=$(awk -v a="$best" -v b="$s" 'BEGIN{print (b<a)?b:a}')
    kbest=$(awk -v a="$kbest" -v b="$k" 'BEGIN{print (b<a)?b:a}')
    echo "   run $i: ${s}s wall, sim kernel ${k} ms"
  done
  SWEEP_S=$best
  KERNEL_MS=$kbest
  SWEEP_SUM=$("$GBT" bench "${DATA_ARG[@]}" --grid default --variants 1000000 --backends gpu --json |
              sed -n 's/.*"checksum":"\([0-9a-f]*\)".*/\1/p')
  echo "   best ${SWEEP_S}s wall, sim kernel ${KERNEL_MS} ms, full-grid checksum $SWEEP_SUM"
  # The full-grid checksum is the correctness anchor for the default grid on this data:
  # a speed change that moves it changed the answers, not just the speed.
  if [ -n "${EXPECT_SUM:-}" ] && [ "$SWEEP_SUM" != "$EXPECT_SUM" ]; then
    echo "GATE FAIL: full-grid checksum $SWEEP_SUM != expected $EXPECT_SUM"; exit 1
  fi
fi

echo "== 4/4 log"
[ -f "$LOG" ] || printf 'date\tcommit\tdirty\tgpu\tsweep_default_s\tsim_shared_ms\tfull_grid_checksum\tnote\n' > "$LOG"
COMMIT=$(git -C "$SRC" rev-parse --short HEAD)
DIRTY=$([ -n "$(git -C "$SRC" status --porcelain -- src include apps)" ] && echo yes || echo no)
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%F)" "$COMMIT" "$DIRTY" "$GPU" "$SWEEP_S" \
  "$KERNEL_MS" "$SWEEP_SUM" "${NOTE:-}" >> "$LOG"
tail -3 "$LOG"
echo "GATE PASS"
