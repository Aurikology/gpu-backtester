# gpu-backtester — how work happens here

Read `NOW.md` first; it holds the current state and the ranked candidates (Aurik ranks them).
`README.md` is the public write-up and `docs/TALKING-POINTS.md` is the interview script: any
number that changes in one changes in both, in the same commit.

## Toolchain

- Everything builds and runs in WSL **`Ubuntu-24.04`** (the default distro is `docker-desktop`,
  which has no bash). Call `wsl -d Ubuntu-24.04 -u root -- bash <file.sh>` from PowerShell. Put
  multi-step shell in a `.sh` file (in the scratchpad if it's throwaway): PowerShell eats `$`
  and quoting inside `wsl ... bash -c "..."`, and a mangled flag runs both A/B arms the same way.
- Build dir is root's `/root/build` (source `/mnt/c/dev/gpu-backtester`).
- Data: `data/nq_2019_2026.gbt` (gitignored, from nq-quant's Dukascopy CSV).

## The loop, for every engine or kernel change

1. **Baseline first.** `NOTE=<label> bash scripts/gate.sh NVIDIA /root/build` on the
   unchanged tree. It logs a row to `results/perf-log.tsv`.
2. **Change one thing.** Keep the old path reachable behind a flag (like `--no-sort`) so the
   before and after run on one binary.
3. **Gate with the answer pinned:** `EXPECT_SUM=<full-grid checksum> NOTE=<label> bash
   scripts/gate.sh ...`. Default grid on the real data is `a6bb0fa189bff7ee`; large is
   `a72cfddb9ccc5ff1`. A speed change that moves the checksum changed the answers: fail.
4. **A/B on one binary, arms alternating**, at least 5 runs default + 3 large (see
   `scripts/ab-sort.sh`). Commit the output under `results/`. Quote the kernel time from the
   profiler span *and* the end-to-end wall time; they differ by ~0.2 s of fixed start-up.
5. **Tell the story with the number:** README section + TALKING-POINTS answer + resume bullet,
   only from numbers in `results/`.
6. **Update `NOW.md`** (state, numbers, next candidates).

Never ship a number that was not measured this way. Rules for correctness are absolute:
every GPU configuration matches `cpu1` to the bit (FMA off on both compilers).

## Git

- **Commit continuously** (Aurik, 2026-11-27): every piece that passes the gate gets its own
  commit right away, unasked. Don't batch a session's work into one commit at the end.
- `main` is the branch; commit subjects are `[scope] subject` (`engine`, `stats`, `docs`, `repo`).
- Remotes: `origin` = public GitHub (CI runs CPU + sanitizers), `gitlab` = private mirror.
  Push both at session end once the gate passes; prove it with `git ls-remote` on each.
- Stage explicit paths. Commit message via file (`git commit -F`).
