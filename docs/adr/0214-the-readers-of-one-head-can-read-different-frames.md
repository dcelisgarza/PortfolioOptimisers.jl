---
status: accepted
---

# The readers of one head can read different frames

## Context

ADR [0211](0211-each-reader-of-the-returns-states-its-own-source.md) puts an `x_src` field on each
reader of the returns and takes the source off every head. ADR
[0212](0212-a-returns-source-is-a-singleton-type.md) gives the field the values `nothing`,
`DataSource()`, `ExcessSource()` and `PriorSource()`. With no frame on the head, two readers of one
head can read different frames. This ADR decides if that is legal, and what refuses or reports a
mismatch (#1682).

The facts at `0ca32d9513`:

- Today one `brt` on the head gives one frame to every reader of the head. #1651 found the defect
  class of two frames at one head: the JuMP builders got the raw `rd` while the prior got `X - B`.
- `assert_benchmark_frame` (`src/21_Pipeline/02_StepExecution.jl`) refuses a `Pipeline` whose step
  frame differs from the frame of the optimiser that receives the value of the step (ADR 0204,
  decision 7). After ADR 0211 the optimiser holds no frame, so the check has no target.
- ADR 0212, decision 2, adds `DataSource()` beside `nothing` so that a check can tell a reader that
  the caller forgot from a raw reader that the caller stated.
- `nothing` does not mean "raw" on every reader. A prior and a graph reader read `rd.X`. A regression
  reader takes `pr.rr` when the prior holds loadings, so it reads the frame of the prior, and it
  refits on `rd.X` only when the prior holds none.
- `asset_panel(ape::RegressionPanel, pr, rd, X)` reads `pr.rr.L`, the loadings of the prior. So the
  features of a `FeatureDistance` with a `RegressionPanel` are in the frame of the prior, not in the
  frame of the graph reader that holds them. A `PhylogenyPanel` reads the `X` of its graph reader,
  and `ape = nothing` reads the Asset Panel, which has no returns frame.

## Decision

The maintainer decided on 2026-10-13.

1. **A free mix.** The readers of one head can read different frames. No head holds a frame, and no
   head refuses a mix.
2. **A stated mix is silent.** A reader that states `DataSource()` beside a reader that states
   `ExcessSource()` builds and runs with no log.
3. **A forgotten raw reader warns at construction.** A group warns when it holds a reader that states
   `ExcessSource()` and an outermost reader at `nothing` that reads the raw returns data. The warning
   names the head, the path of the reader at `nothing`, and tells the caller to state `DataSource()`
   or `ExcessSource()` on it. Then the head builds and runs as configured. The configuration alone
   decides the warning, so it fires in the constructor. An inner reader keeps `nothing` by rule (ADR
   0211, decision 3), so the walk skips it. A reader that states `PriorSource()` reads the frame of
   the prior and never warns by itself.
4. **A regression reader at `nothing` counts only when it refits.** It counts when the prior estimator
   of its head cannot give loadings, because then it refits on the raw `rd.X`. Under a prior that
   holds loadings it reads the frame of the prior, so it does not count. When the type of the prior
   does not show this at construction, for example a precomputed Prior Result, it does not count.
5. **The group is one head.** The group is a head and every reader in its slots, the readers inside
   its risk measures, constraints, uncertainty sets and losses included. The walk stops at an inner
   optimiser, a fallback and a leader, because each is a head with its own group (ADR 0204, decision
   4). In `OnlinePortfolioSelection`, the source of the step, the prior of a `RiskLoss` and the priors
   of a `ProgrammeAllocationSet` form one group.
6. **A `Pipeline` joins the group of its optimiser.** The steps whose values the last optimiser
   receives join the group of that optimiser, and the `Pipeline` constructor warns by decision 3.
   `assert_benchmark_frame` and its refusal are deleted.
7. **Each different mismatch logs one time.** The warning is a `@warn` with `maxlog = 1` and an `_id`
   made from its message. The message names the type of the head and the path of the reader, so a
   head that a cross-validation rebuilds logs one time, and a different mismatch still logs.

The scenario of #1682: a `HierarchicalOptimiser` whose `ClustersEstimator` states `ExcessSource()`
and holds `FeatureDistance(; ape = RegressionPanel())`, and whose prior stays at `nothing`. The prior
reads `rd.X` beside an excess reader, so the head warns by decision 3. If the caller states
`DataSource()` on the prior, the head is silent: the clustering is in the Excess Frame, and its
features are loadings of the raw returns.

## Consequences

- A tracking-error `MeanRisk` can fit its prior in the Excess Frame and cluster on the raw returns.
  Today a caller cannot build it.
- The build adds one walk over the reader slots of a head, shared by every head constructor and by
  the `Pipeline` constructor. ADR 0213 refused a walk of this kind for a deprecation, because it
  had to rewrite the readers. This walk only reads them.
- A head with no reader in the Excess Frame never warns, so a caller with no benchmark sees no
  change.
- The decision about the frame of the online step, the outer problem of a meta-optimiser and the
  covariance forecast evaluation stays with #1683.

## Alternatives considered

- **One frame per head.** Every reader of a head must read the same frame. It keeps the guarantee of
  today's `brt`, but it brings back a frame on the head, which ADR 0211 removed, and it refuses a
  legal model: a prior in the Excess Frame with a clustering on the raw returns.
- **Refuse a forgotten raw reader.** The caller must write `DataSource()` to mix. It is the strict
  form, but the maintainer chose a warning.
- **No check.** A mix of any kind is silent, and `DataSource()` only documents a choice. It brings
  back the defect class of #1651 through a default.
- **A regression reader at `nothing` never counts.** Its fallback refit on the raw returns under a
  prior with no loadings is then silent.
- **A regression reader at `nothing` always counts.** It warns under a factor prior in the Excess
  Frame, where the reader takes loadings in that frame, so it warns on a correct configuration.
- **The whole tree is one group.** A meta-optimiser then warns when one inner optimiser reads the
  Excess Frame and another reads the raw returns, which ADR 0204, decision 4, lets each one choose.
- **Keep `assert_benchmark_frame`.** It compares a step with the frame of the optimiser, and the
  optimiser has none.
- **The `Pipeline` steps are not checked.** A prior step in the Excess Frame with a clustering step
  at `nothing` is then silent, while the same readers in the slots of one head warn.
- **A warning at each construction.** A cross-validation that rebuilds the head fills the log.
- **`maxlog = 1` with no `_id`.** The first mismatch of a session hides every later one, even a
  different one.

## Related

- ADR 0204: the current rule for `brt`, decision 7 of which this ADR replaces.
- ADR 0211: the readers that hold `x_src`.
- ADR 0212: the values of `x_src`, and the reason for `DataSource()`.
- ADR 0213: `brt` is removed in one break.
- `assert_benchmark_frame`, `pipe_prior_brt`, `asset_panel`, `resolve_factor_regression`.
