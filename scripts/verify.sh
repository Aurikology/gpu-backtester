#!/usr/bin/env bash
# The double check before every commit. Runs, in order, and stops at the first failure:
#   1. CPU-only build + tests (what GitHub CI builds)
#   2. the NVIDIA gate with the default-grid answers pinned (scripts/gate.sh)
#   3. the large grid's full checksum, pinned too
#   4. the diff has no stray files: only paths under src include apps tests scripts
#      results docs + the repo's markdown
# Usage: scripts/verify.sh [note]                 env: RUNS (gate), SKIP_LARGE=1
set -euo pipefail
SRC=$(cd "$(dirname "$0")/.." && pwd)
NOTE=${1:-verify}
DATA=$SRC/data/nq_2019_2026.gbt
EXPECT_DEFAULT=a6bb0fa189bff7ee   # gbt bench --grid default, all 152,064 variants, real data
EXPECT_LARGE=a72cfddb9ccc5ff1     # same, large grid, 855,360 variants

echo "#### 1/4 CPU-only build (CI shape)"
bash "$SRC/scripts/build.sh" OFF /root/build-off 2>&1 | grep -E '^\[  (PASSED|FAILED|SKIPPED)|tests from|error' || true
/root/build-off/gbt_tests --gtest_brief=1 >/dev/null || { echo "VERIFY FAIL: CPU-only tests"; exit 1; }

echo "#### 2/4 NVIDIA gate"
EXPECT_SUM=$EXPECT_DEFAULT NOTE=$NOTE bash "$SRC/scripts/gate.sh" NVIDIA /root/build 2>&1 |
  grep -v '^\[[0-9]' | grep -E 'tests|PASSED|FAILED|agree|best|FAIL|PASS'
[ "${PIPESTATUS[0]}" = 0 ] || { echo "VERIFY FAIL: gate"; exit 1; }

if [ "${SKIP_LARGE:-0}" != 1 ]; then
  echo "#### 3/4 large-grid checksum"
  got=$(/root/build/gbt bench --data "$DATA" --grid large --variants 100000000 --backends gpu --json |
        sed -n 's/.*"checksum":"\([0-9a-f]*\)".*/\1/p')
  [ "$got" = "$EXPECT_LARGE" ] || { echo "VERIFY FAIL: large checksum $got != $EXPECT_LARGE"; exit 1; }
  echo "large grid checksum $got ok"
fi

echo "#### 4/4 changed paths"
cd "$SRC"
bad=$(git status --porcelain | awk '{print $2}' |
      grep -vE '^(src|include|apps|tests|scripts|results|docs|\.github)/|^[A-Z]+\.md$|^CMakeLists\.txt$' || true)
git status --short
[ -z "$bad" ] || { echo "VERIFY FAIL: unexpected paths: $bad"; exit 1; }
echo "VERIFY PASS"
