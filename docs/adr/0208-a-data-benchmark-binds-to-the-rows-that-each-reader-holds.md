---
status: accepted
---

# A data benchmark binds to the rows that each reader holds

## Context

A `ReturnsTracking` series is fixed at construction, so it aligns with the rows of one fit only.
[#1656](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1656) made every reader check
its length and made the online step refuse it.
[#1669](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1669) records the case that the
length check cannot see: the folds of a batch cross-validation that share one training length.

The returns data already carries a benchmark, `B` of the `ReturnsResult`, and `B` travels row-aligned
with `X` through every route: the Fold Context of the online step, the rows window of an online
portfolio selection head, the folds of a cross-validation and the prediction. No tracking form read
it. [#1657](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1657) asked for one.

## Decision

The maintainer decided three questions on 2026-10-13.

1. **A new type.** `DataTracking()` is a new `AbstractTrackingAlgorithm`, not a sentinel in
   `ReturnsTracking`. It holds no series of its own. Its field `B` is `nothing` until a reader binds
   it.
2. **A matrix `B` gives the series `B * w`.** A vector `B` is the benchmark series. A matrix `B` holds
   one benchmark per asset, so the benchmark of the portfolio holds the same weights in the benchmark
   of each asset, and the tracking error is that of `(X - B) * w`. It is linear in the weights, so
   every JuMP form stays conic. It is the rule that `collapse_benchmark` applies to a prediction with
   no Held Weights. The path on a precomputed portfolio series refuses a matrix, because it has no
   weights.
3. **`brt = true` refuses a data benchmark.** A reader in the Excess Frame fits on `X - B`, and a
   tracking error against `B` would subtract the benchmark twice.

Each reader that holds the returns data binds the data benchmark to the `B` of its own rows, with
`bind_data_benchmark`, and rebuilds an object through its keyword constructor only when a part
changes:

- a JuMP head, after `investable_view`, so the benchmark has the rows and the investable columns of
  the model; every builder, corner solve and frontier risk then reads the bound benchmark;
- `HierarchicalRiskParity` and `HierarchicalEqualRiskContribution`, after `investable_reduction`;
- the value-level `expected_risk` on a `ReturnsResult`, in `resolve_risk_inputs`;
- an online portfolio selection head: the tracking errors and the risk ceilings of a programme set on
  the rows the head holds, and a `RiskLoss` on its window, after the view at the Investable Mask.

A binding replaces a benchmark bound earlier, so a benchmark never reads the rows of another reader.
A reader with no returns data leaves the benchmark unbound, and `tracking_benchmark` refuses it.

### A batch cross-validation refuses a returns series

**A batch cross-validation with more than one fold refuses a `ReturnsTracking` series** (#1669). The
maintainer chose this, option 1 of the issue, on 2026-10-13, after `DataTracking` was on `dev`. The
session that built it chose the four details below, and the maintainer accepted them on the same day.

1. **The batch arms refuse once, before the folds run.** The fold loop calls
   `assert_folds_track_no_series` next to `assert_batch_entry`, so the error comes before any solve
   and does not repeat on each worker. The error names the field and points at `DataTracking`. The
   online arm gets no new check, because its step already refuses a series.
2. **One fold passes.** One fold is one batch fit, so a series of its training rows aligns, and the
   length check catches a wrong length.
3. **The series of an online selection head is refused.** The head cuts the series to its first rows.
   Each fold of a batch walk-forward restarts the head at its first training row, so the cut aligns
   only with a fold that starts at the first row. A `DataTracking` set follows the rows of each fold.
   The walk, `returns_tracking_path`, enters an algorithm as well as an estimator and a Result,
   because the programme set is an algorithm.
4. **A `TimeDependent` schedule answers through its `default` only.** Entry `i` is the value of fold
   `i` alone, so a schedule can give each fold the series of its own rows. The `default` is one value,
   and more than one solve can read it: a solve with no fold, a direct online step, or an inner
   optimiser under `bind = :nearest`.

## Considered options

- **Refuse at every fold count.** Rejected. It refuses one fold, which is one batch fit and aligns.
- **Refuse in each fold, or in the online arm too.** Rejected. A check in each fold runs on the
  workers after the first solves, and the online step already refuses a series.
- **Let the series of the head through.** Rejected. Every fold after the first misaligns with no
  error, the defect of #1669. `DataTracking` in the set covers the case.
- **Refuse every part of a schedule.** Rejected. It refuses a schedule of one series for each fold,
  which aligns. test_24b holds that it equals the batch fit of each fold.
- **Let every part of a schedule through.** Rejected. The `default` is one series that more than one
  solve can read, the defect again.
- **A walk method for the allocation set only.** Rejected. The online step reads the same walk, so it
  has the same side effect, and it misses a series that another algorithm holds.
- **A second walk for the fold loop only.** Rejected. It keeps the online step as it was, but it is a
  second copy of one walk that can drift from the first.

## Consequences

- The online step, a rolling online window and every fold of a walk-forward read the benchmark of
  their own rows, so online equals batch with no more work. A rolling batch walk-forward, the case of
  #1669, tracks the rows of each fold, and the same walk-forward refuses a `ReturnsTracking` series.
- The online step and the batch fold loop both read a schedule through its `default` only, so a
  schedule of one series for each fold passes the online step too.
- The online step of a meta-optimiser that holds a head with a series in its set refuses it too,
  because the step reads the same walk.
- Two refusals are stricter than the defect needs. Each has `DataTracking` as the way around it, and
  each is the first place to look if a refusal is reported as wrong:
  - a batch loop refuses the `default` of a schedule under `bind = :outermost`, which the loop never
    reads;
  - the online step of a meta-optimiser refuses the series of a head when the window grows from the
    first row, where the cut to the first rows aligns. Under `max_history` the window rolls, and the
    refusal is correct.
- A result keeps its risk measures bound to the rows of the fit, as it keeps them resolved against
  the prior of the fit. A value-level call on a `ReturnsResult` binds them again.
- A data benchmark pays no fee. A `WeightsTracking` benchmark keeps its own `fees`.
- The online selection head has no `brt` of its own yet, so it binds under `brt = false`.
  [#1662](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1662) and
  [#1663](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1663) add it.
- Scoring a prediction against its benchmark is
  [#1664](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1664).
