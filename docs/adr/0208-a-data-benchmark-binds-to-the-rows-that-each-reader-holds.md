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

## Consequences

- The online step, a rolling online window and every fold of a walk-forward read the benchmark of
  their own rows, so online equals batch with no more work. A rolling batch walk-forward, the case of
  #1669, tracks the rows of each fold.
- A result keeps its risk measures bound to the rows of the fit, as it keeps them resolved against
  the prior of the fit. A value-level call on a `ReturnsResult` binds them again.
- A data benchmark pays no fee. A `WeightsTracking` benchmark keeps its own `fees`.
- The online selection head has no `brt` of its own yet, so it binds under `brt = false`.
  [#1662](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1662) and
  [#1663](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1663) add it.
- Scoring a prediction against its benchmark is
  [#1664](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1664).
