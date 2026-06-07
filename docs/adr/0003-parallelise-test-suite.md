---
status: accepted
---

# Parallelise the test suite with ParallelTestRunner.jl

## Context

The serial test suite runs in ~43 minutes (CI, single process). Per-file timings:

| File | Time | | File | Time |
| --- | --- | --- | --- | --- |
| 18 Mean Risk | 18m24s | | 13 Phylogeny | 1m27s |
| 16 Risk Budgeting | 4m38s | | 20 Near Optimal Centering | 1m01s |
| 22 Nested/Stacking | 4m27s | | 24 Cross Validation | 58s |
| 08 Moments | 2m34s | | 09 Risk Measures | 39s |
| 17 Clustering | 2m08s | | 01 Structs | 31s |
| 25 Plotting | 1m42s | | (others) | < 30s each |
| 12 Prior | 1m59s | | | |

Total ≈ 2564 s. One file (18 Mean Risk, 1105 s) is 43 % of all work.

The current `test/runtests.jl` `walkdir`s `test/`, derives a title from each
`test_*.jl` filename, and `include`s the file inside a `@testset`. Each file opens
with `@safetestset "Title" begin`, a `using ...` block, and a locally-defined
`find_tol` helper (duplicated in 18 of 26 files, with minor textual drift), followed
by a per-file setup block (load returns, compute a prior, build asset sets) shared by
that file's `@testset`s.

Target environment: **GitHub Free** runners — 2 CPU cores, ~7 GB RAM.

## Decision

Adopt **ParallelTestRunner.jl** (the runner from the [JuMP testing
tutorial](https://jump.dev/tutorials/2026/02/24/testing/)). It auto-discovers
`test/*.jl`, runs each file in an **isolated worker module**, records per-file
durations, and on subsequent runs schedules longest-first (LPT) across workers.

### 1. Runner: ParallelTestRunner.jl, not ReTestItems.jl

File-level parallelism matches the suite's existing one-concern-per-file structure.
ReTestItems was considered; it needs every assertion wrapped in `@testitem` blocks —
a larger, more invasive rewrite for no extra benefit at this granularity.

### 2. Drop `@safetestset`, use bare `@testset`

ParallelTestRunner already runs each file in its own module on a separate worker
process, so `@safetestset`'s module isolation is redundant. Each file's outer
`@safetestset "Title" begin ... end` becomes `@testset "Title" begin ... end`.
`SafeTestsets` is removed from `test/Project.toml`.

### 3. Shared helpers via `init_code`

`find_tol` and the near-universal `using` core
(`Test, PortfolioOptimisers, CSV, TimeSeries, DataFrames, StableRNGs, StatsBase,
LinearAlgebra`) move into the `init_code` keyword of `runtests`, which is evaluated
in every test file's sandbox module. `find_tol` is standardised on a single variant
(the `$name1/$name2` form). Test files keep `using` only for their **rare/heavy**
dependencies (`StatsPlots`, `GraphRecipes`, `Pajarito`, `HiGHS`, `SCS`, `Clustering`,
`AverageShiftedHistograms`, `FLoops`) so memory-heavy deps stay out of sandboxes that
do not use them — this matters on a 2-core / 7 GB runner.

### 4. Split large files into ~90 s equal-duration files; shared per-group setup file

The makespan floor on N-way parallelism is `total / N`, achievable only while no
single file exceeds it. On **2 workers** the floor is ~21.4 min and file 18 (18.4 min)
already fits — LPT balances the 26 files to ~21 min with no splitting. But the suite
is decomposed into **roughly equal ~90 s files** anyway, so the same file set scales to
a future **GitHub Actions job matrix** (free tier: up to 20 concurrent jobs) without
re-splitting. Files 18, 16, 22, 08, 17, 12, 13, 25 are binned by their internal
`@testset` groups into ~90 s files; small files are left as-is.

**Setup lives once per split-group in a shared `testNN_setup.jl` that each split
`include`s (option A′), rather than being textually copied into every split (option A)
or hoisted to `init_worker_code` (option B).** Each `test_18x.jl` is a thin shim:
`include(joinpath(@__DIR__, "test18_setup.jl"))` plus a one-line call into a block
function (`mr_block1(1:8)`, `mr_block2(1:47)`, …) defined in that setup. The `include`
is **not** shared cross-file state: ParallelTestRunner runs every `test_*.jl` in its own
worker module, so the include defines fresh `rd` / `pr` / `mr_block*` bindings per file
every time. A′ therefore has the same strict per-file isolation and the same
recompute-per-split cost as true duplication (A), but without A's copy-paste drift — the
very failure mode §3 cites for the old per-file `find_tol`. B is rejected for the same
reason as before: its once-per-worker shared fixtures reintroduce exactly the hidden
cross-file global state file isolation exists to remove, and force every worker to
compute every fixture even for files that do not use it. The recompute cost — setup time
× number of splits — is bounded by keeping splits coarse: the ~90 s target keeps a
group's shared setup a small fraction (<~15 %) of its runtime. B is held in reserve only
for a specific file where measurement shows setup genuinely dominates.

The setup files are named `testNN_setup.jl` (no underscore after `test`) precisely so the
`startswith(basename, "test_")` discovery filter in `runtests.jl` skips them — they are
fixtures `include`d by split files, not independently discoverable test files.

## Measured split plan

Per-`@testset` timings were taken in a warm kaimon REPL (`--project=test`). Comparing
**warm execution** against the **serial CI** time per file exposes a decisive split:

| File | CI total | Warm exec | Class | Hotspots |
| --- | --- | --- | --- | --- |
| 18 Mean Risk | 18m25s | 868 s | **exec** | "Mean Risk" 696 s, Formulations 72 s |
| 22 NCO/Stacking | 4m27s | 333 s | **exec** | Mix optimisers 169 s, Risk measure views 79 s |
| 16 Risk Budgeting | 4m38s | 149 s | exec+compile | Factor RB 65 s, MIP RB 46 s, Asset RB 38 s |
| 08 Moments | 2m34s | ~37 s | **compile** | (mostly first-touch compilation) |
| 17 Clustering | 2m08s | 97 s | exec+compile | HierarchicalRiskParity 55 s |
| 12 Prior | 1m59s | 137 s | **exec** | ExpEntropyPooling 81 s, LogEP 21 s |
| 25 Plotting | 1m42s | 113 s | compile | plotting-compile setup 38 s; rest tiny |
| 13 Phylogeny | 1m27s | 0.8 s | **pure compile** | 87 s CI is ~100 % compilation |

**Principle: split exec-bound files, keep compile-bound files whole.** A file's CI time
is `first-touch compilation + execution`. Splitting an **exec-bound** file spreads real
compute across workers — a clean win. Splitting a **compile-bound** file only makes each
worker recompile the same code paths — negative value (more total work, no makespan gain).
Confirmed empirically: test_13 is 87 s cold but **0.8 s** warm — its entire CI cost is
one-time compilation, so it stays a single file.

The exec-bound files are split to a ~90 s target (matrix-ready); compile-bound files
(08, 13, 25) stay whole:

- **test_18** (696 s "Mean Risk" testset dominates): the testset's core is
  `for r in rs, obj in objs, ret in rets` (rs = 47 risk measures × 3 objs × 2 returns)
  validated row-by-row against `MeanRisk1.csv.gz`; `i` increments innermost, so any
  contiguous `rs[a:b]` slice maps to a contiguous reference-row block — slicing is clean.
  Split `rs` into ~7 files + 2 files for the DT/IT/BDV/VSK sub-blocks + 2 files for the
  remaining top-level testsets (Formulations; then Cardinality/Phylogeny/Tracking/Budget/…
  grouped). **≈ 11 files.**
- **test_22** → Mix optimisers | Risk measure views | (Fees + Advanced use + Efficient
  frontier + Prior views). 3.
- **test_16** → Asset RB | Factor RB | MIP RB. 3.
- **test_12** → ExpEntropyPooling | (LogEntropyPooling + rest). 2.
- **test_17** → HRP | (HERC + Schur + bounds). 2.
- **test_08, test_13, test_25** → keep whole.

Net: 8 files → ~21; suite ~26 → ~39 files. On 2 workers the makespan floor is total/2
≈ 21 min and "Mean Risk" (11.6 min) already fits under it, so the fine split buys nothing
*today* — its payoff is scaling to a job matrix without further test surgery. Fine-splitting
exec-bound files is cheap (no compile penalty; ~5 s duplicated setup per slice).

**Measurement caveat:** the kaimon env used `JULIA_CONDAPKG_BACKEND=Null`, so test_08's
`MutualInfoCovariance` (Knuth binning) hit a `ModuleNotFoundError: astropy` via PythonCall.
That is an environment artifact, not a test fault — astropy is present in CI.

## Verification

- Per-`@testset` timings for the large files are measured in a warm REPL (kaimon)
  before binning (above), so each emitted file lands near the ~90 s target.
- `julia --project=test test/runtests.jl --jobs=2` reproduces the CI environment
  locally; total wall-clock should land near ~21 min (down from ~43 min).
- The full suite must stay green and no test must be dropped or silently overwritten
  by the split.

**Measured (2026-06-07, dev-test branch, 2-worker local run):**
`julia --project=test test/runtests.jl --jobs=2` → **4074 passes, 0 failures, 23m03s
wall clock** (vs ~43 min serial). The assertion count is higher than the 3771 from the
original serial baseline because the branch includes new features added since that
measurement; no assertions were lost. `test_08_moments` passed with the real conda/pixi
backend (astropy resolved correctly).

## Consequences

- **`test/Project.toml`**: add `ParallelTestRunner`, remove `SafeTestsets`.
- **`test/runtests.jl`** shrinks to discovery + `runtests(PortfolioOptimisers, ARGS;
  init_code)`.
- **First CI run has no duration history**, so its scheduling is order-based and may be
  less balanced than steady state; subsequent runs self-tune via recorded durations.
- **More files** (~30 vs 26): per-file worker startup is paid more often, but the
  worker process is reused across files so package-load cost is amortised per worker,
  not per file.
- **Matrix-ready**: equal-duration files mean widening to a job matrix later is a CI
  config change, not another test rewrite.
- Test-file count and split boundaries become a maintenance surface — a new slow
  `@testset` should be added with the ~90 s target in mind.
