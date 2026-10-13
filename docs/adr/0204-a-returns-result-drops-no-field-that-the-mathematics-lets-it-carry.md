---
status: accepted
---

# A Returns Result drops no field that the mathematics lets it carry

## Context

The benchmark census of #1632 found two defects with one cause: code that builds a `ReturnsResult`
from another one and leaves a field at `nothing`.

- [#1645](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1645): the online step of
  every optimiser that holds a prior folded the raw `rd`, while the batch path first ran
  `returns_result_picker(rd, brt)`. With `brt = true` the prior folded `X` and the batch fit read
  `X - B`, so online weights differed from batch weights by up to `7e-2`, with no error.
- [#1646](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1646): the rows window of an
  online portfolio selection head kept `nx`, `X` and the active mask, and dropped every other field.
  A follow-the-leader leader with `brt = true` never saw a benchmark. `returns_result_picker` then
  ignored `brt` on returns with no `B`, so the leader fitted the absolute returns with no signal.

`returns_result_picker` rebuilt `rd` with `X - B` as the returns, and dropped `nb` and `B`, so that a
second pick could not subtract the benchmark twice. Every pick that met data with no `B` therefore
had to pass through in silence, and a silent pass-through hides a forgotten benchmark. A first fix
kept `nb` and carried `B - B` as the benchmark, a zero that a second pick subtracts. That fix lost
the values of `B`: after a pick, neither `B` nor the raw `X` could be recovered.
[#1651](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1651) replaced it.

The census of #1651 found a second defect of the rebuild. The JuMP heads give the raw `rd` to their
model builders, so under `brt = true` an uncertainty set with a prior of its own, the uncertainty set
of `UncertaintySetVariance` and `NearOptimalCentering`, and the factor loadings that
`RiskBudgeting`, `RelaxedRiskBudgeting` and `FactorRiskContribution` refit read the raw `X`, while
the prior read `X - B`.

The benchmark census for the online estimators found a third gap
([#1659](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1659)). A `Pipeline` prior
step had no `brt`, and `inject_context` put its Prior Result into the optimiser unchanged, because
`prior` returns a Prior Result as it is. Under the optimiser's `brt = true` that prior held the
moments of the raw `X`, while every other reader of the optimiser read `X - B`. The uncertainty-set,
clustering, phylogeny, centrality and exposure steps had the same gap
([#1670](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1670)). The same census found
that `covariance_forecast_evaluation` had no `brt`
([#1658](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1658)). A caller could not
evaluate a forecast of the covariance of `X - B`, which a tracking-error portfolio reads, unless
they built `ReturnsResult(; X = X .- B)` by hand and lost `B`.

The maintainer stated the rule on 2026-10-12: the quantities of a `ReturnsResult` are never dropped,
unless they truly cannot be carried for mathematical reasons. The maintainer decided #1651 on the
same day: `brt` stops rebuilding the `ReturnsResult`, and each reader subtracts the benchmark at the
point of use. This is how the library already treats a benchmark: `WeightsTracking`,
`ReturnsTracking` and `TrackingRiskMeasure` hold the benchmark in the estimator and subtract it from
whatever `X` is in scope.

## Decision

1. **A function that builds a `ReturnsResult` from another one carries every field.** A field is
   dropped only when it has no meaning on the new axis, and the code says why in a comment or its
   docstring. Examples of such a reason: an asset-indexed field after a collapse to synthetic
   assets with no collapse rule, or a window whose rows hold no such field upstream. A refusal by
   name is acceptable where a field cannot travel. A silent drop is not.
2. **`brt` never rebuilds the `ReturnsResult`.** The returns data `rd` reaches every consumer with
   its raw `X` and its `B`. Each reader of the returns takes `brt` as an argument and reads
   `benchmark_relative_returns(rd, brt)`, a matrix: `X`, or `X - B` under `brt`. The readers are the
   prior, the factor regression of an exposure constraint and of a factor risk contribution, an
   uncertainty set with a prior of its own, and a clustering, phylogeny or centrality estimator under
   `x_src = :data`. A reader subtracts once, from raw data, so no reader can subtract twice.
3. **`brt = true` on returns that hold `X` and no `B` is an `ArgumentError`.** No excess return
   exists without a benchmark. Returns data with no `X`, which the empty `ReturnsResult` of a
   prior result is, passes through, because it holds nothing to subtract from.
4. **Each optimiser applies its own `brt` and nothing else.** An inner optimiser of a
   meta-optimiser, a subset optimiser of `SubsetResampling`, a fallback, and the outer optimiser of
   a meta-optimiser read the raw `rd` and apply their own `brt`. The `brt` of a meta-optimiser
   changes its own prior and its clustering. The outer problem reads the raw returns of the
   sub-portfolios, `X * wi`, with `B` collapsed beside them, in the path without folds and in the
   cross-validated path alike. In the path without folds the returns of the prior are `X - B` under
   the meta's `brt`, so `outer_returns_source` adds the benchmark back. The corner solves of
   `NearOptimalCentering` take the `brt` of the head, because they are its own sub-problem.
5. **The online step of an optimiser folds `X - B` into its prior and keeps the raw rows**
   (`fold_returns`). `prior_step_returns` gives the prior the returns data with `X - B` in `X`, the
   one `ReturnsResult` that `brt` still builds. In an optimiser the prior alone reads it, and
   decision 8 gives it to the fold loop of a covariance forecast evaluation. The Fold Context keeps the
   raw rows under `brt`, so `optimise(opt)` with no returns rebuilds the raw `rd`, and it equals
   `optimise(opt, rd[1:t])` under `brt` too (ADR 0137).
6. **The rows buffer of an online portfolio selection head is a `ReturnsBufferState` that owns the
   returns**, as the Fold Context of a head with no prior is. The window that every rule, leader
   and programme set reads carries every field of the folded rows: `nf`, `F`, `nb`, `B`, `ne`, `E`,
   `ts`, `iv`, `ivpa` and the static panel. So the head no longer refuses `iv`. Every prior that
   the head fits on those rows reads the surface: the prior of a programme set, the prior of a
   Risk Loss, and the prior of a forecaster. The refit arm of a forecast rule gives `iv` and
   `ivpa` to every forecaster, as `prior` over a `ReturnsResult` gives them to every member
   ([#1644](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1644)). A
   `PriorExpectedReturns` gives them to its prior, a `WindowedExpectedReturns` cuts them to its
   window, and every other forecaster ignores them. The covariance step of
   `covariance_forecast_evaluation` gives them to a buffer, and an exact fold, which holds numbers
   alone, ignores them as its batch fit does
   ([#1643](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1643)).

7. **A step of a `Pipeline` that reads the returns states its frame in its `PipelineStep`, and
   the frame must equal the frame of the optimiser that receives the value of the step** (#1659,
   #1670). A step estimator holds no `brt`, so `PipelineStep(; est = pe, writes = :prior,
   brt = true)` fits the step on `X - B`: in batch through `prior(pe, rd; brt)`, and in the online
   step through `fold_prior` under `brt`, with the raw rows in the Fold Context of the Pipeline, as
   decision 5 states for an optimiser. A clustering, network, uncertainty-set, phylogeny,
   centrality or exposure step reads the returns as the optimiser reads them under its `brt` and
   `x_src = :data`. `assert_benchmark_frame` refuses a mismatch when the `Pipeline` is built,
   because the configuration alone decides it. A bare step fits on the raw returns, so it meets an
   optimiser with `brt = true` with that refusal. A clustering step is checked against the `brt`
   of the optimiser, which is also the frame of its prior. A network step writes a phylogeny matrix
   that no optimiser receives, so it takes a frame and is not checked. `brt = true` on a step that
   reads no returns is refused too: an uncertainty set that reads the prior, an exposure constraint
   that reads the loadings of the prior, and every other family, because the wrapper passes `brt`
   to none of them.
8. **`covariance_forecast_evaluation(est, rd, cv; brt = true)` evaluates a forecast of `X - B`**
   (#1658). Every reader of the returns in an evaluation reads one frame: the fit over a training
   window, the online step and the test rows. No reader needs the raw `X`, so the verb gives its
   fold loop `prior_step_returns(rd, brt)`, which keeps `B`. No reader subtracts twice, because
   the estimator receives no `brt`. A vector `B` is subtracted from every asset, which is the frame
   of a tracking error, and a matrix `B` asset by asset, as `benchmark_relative_returns` does. The
   Result records `brt`. So `covariance_forecast_portfolio` reads its test rows in the frame of the
   stored forecasts and takes no `brt` of its own, and `covariance_forecast_compare` refuses two
   evaluations in different frames. An `ImpliedVolatility` reads the surface of the assets beside
   `X - B`, as the prior of an optimiser does under `brt`. The returns data holds no surface of the
   excess returns, so the docstring states this and the verb does not refuse it.
9. **`OnlinePortfolioSelection(; brt = true)` selects in the Excess Frame** (#1662). The step
   reads the price relative of the excess return, `1 + r - b`, from
   `benchmark_relative_returns` over the block. Each reader of the rows inside a rule reads
   `X - B` through `head_frame_returns`: the refit and the fold of a forecaster, the scale of a
   `ForecastReversion`, the window of an `AntiCorrelation` and the row selector of a
   `FollowTheLeader`. These readers take the rows through the signature of `online_update!`, which
   has no `brt`, so the head puts its `brt` on the `ProjectionStep` of the row beside `rows` and
   `strict`. The rows stay raw, as decision 2 states. So a leader, the prior of a Risk Loss and the
   prior of a programme set apply their own `brt`, as decision 4 states for an inner optimiser.
   The state pins `nb` under `brt = true`, because the frame of every row is the excess over that
   benchmark. `predict` reports the raw wealth of the fund, because the fund holds the assets.
10. **A Risk Loss and a programme set take a `brt` of their own** (#1663). `RiskLoss` and
    `ProgrammeAllocationSet` take `brt::Bool = false`, and fit their prior with
    `prior(pe, rows; brt)`. So a `Variance` loss or ceiling under `brt = true` is a tracking
    variance against the `B` that the rows carry. Each slot of the set that reads the rows takes
    the set's `brt`, as the same slot of a `JuMPOptimiser` takes the optimiser's: the exposure
    rows, the centrality and phylogeny rows, the return floor, and the binding of a
    `DataTracking`. The built-in programmes of `ShortTermLossControlPortfolio` and
    `LowDimensionEnsemblePortfolio` take a `brt` keyword and give it to the `JuMPOptimiser` of
    their `MeanRisk`. The `brt` of the head does not reach them, as decision 4 states.

## Consequences

- A leader, a programme set or a fallback of the head reads the rows as a batch fit over the same
  window reads them. A leader with `brt = true` fits the excess returns.
- A caller that sets `brt = true` and forgets `B` gets an error at the first reader, in batch and
  online alike.
- `rd.B` keeps its values everywhere, so `prepare_outer_rd` collapses the real benchmark for the
  outer problem of a meta-optimiser.
- A reader that a later change adds must take `brt`, or it reads the raw `X` under `brt = true`.
  The cost of the decision is this obligation, one keyword at each reader.
- Under the online step with `brt = true`, the context and the buffer of the prior both hold one
  matrix of the rows: the raw rows and the excess rows.
- `NestedClustered(brt = true)` and `Stacking(brt = true)` no longer make their inner optimisers fit
  `X - B`. A caller who wants that sets `brt = true` on each inner optimiser.
- The window of the head is views into its buffers, as before, so the step takes no copy.
- A caller who evaluated a hand-built `ReturnsResult(; X = X .- B)` passes `rd` and `brt = true`, and
  gets the same diagnostics with `B` kept. A caller who builds a `CovarianceForecastEvaluationResult`
  by position passes `brt` after `target`.
- Under a vector `B`, the Excess Frame moves a log-wealth rule such as `ExponentiatedGradient`,
  because `1 + r - b` is not a scaling of `1 + r` by one number. A rule with an absolute threshold
  or a correlation moves too.
- A tracking variance can bound an online allocation in two ways: a `Variance` ceiling of a set
  with `brt = true`, or a `TrackingRiskMeasure` ceiling on a `DataTracking` of a set with
  `brt = false`. The set refuses a `DataTracking` under `brt = true`, as a `JuMPOptimiser` does.
- A pipeline that injected a raw prior into an optimiser with `brt = true` now fails when it is
  built. The caller wraps the prior step with `brt = true`. The same holds for a raw uncertainty
  set, clustering, phylogeny, centrality or exposure step.

## Alternatives considered

- **Carry `B - B` after the pick** (the first fix of #1646). A zero benchmark is its own fixed
  point, so a second pick changes no value, but the values of `B` and the raw `X` are lost.
- **Rebuild `X` and keep `B`, with a marker field that says the data is already relative.** It is
  lossless, but it keeps the rebuild and adds a field that every constructor, view, buffer and fold
  must carry.
- **Drop `nb` and `B` after the pick, as before.** This forces the silent pass-through of decision 3,
  and it breaks the rule.
- **Refuse `brt = true` on a follow-the-leader leader** (option 1 of #1646). It hides the defect for
  one caller, leaves the head without a benchmark-relative leader, and keeps the silent no-op for
  every other caller.
- **Refuse `brt = true` on an optimiser below a prior step** (option 1 of #1659). It makes the mix
  loud, but it leaves no way to fit a pipeline prior in the Excess Frame.
- **Pass the optimiser's `brt` up to the prior step when the pipeline is built** (option 3 of
  #1659). It hides the frame of the step from the caller who reads the step, and a step in a
  nested pipeline or under a schedule has no single optimiser to read it from.
- **Subtract at each reader of a covariance forecast evaluation, as decision 2 does.** The online
  arm of the fold loop calls `partial_fit!(est, rd)` with no `brt`, so the loop would have to
  thread a `brt` to every estimator that it steps, for no reader that needs the raw `X`.
- **A `brt` keyword on `covariance_forecast_portfolio`** (the proposal of #1658). A caller could
  score a forecast of `X - B` on the rows of `X`, with no error.
- **Let the `brt` of a meta-optimiser pass down to its inner optimisers.** It keeps the weights that
  the rebuild gave, but an inner optimiser that holds no `brt` field cannot take it, and the
  fallback and the corner solves already apply their own `brt`.

- **Select on the relative price relative `(1 + r) / (1 + b)` under the head's `brt`** (the second
  frame of #1662). It does not change a log-wealth rule under a vector `B`, because
  `x / <w, x>` does not change when every entry is scaled by one number. It is a second frame
  beside the `X - B` of every other reader of the library.
- **Thread `brt` through `online_update!`.** Each of its methods and every rule over other rules
  would take one more argument that most of them do not read.
- **Let a Risk Loss and a programme set read the head's `brt`** (#1663). It needs no new field,
  but a caller could not bound the tracking variance of a head that selects on raw returns, and
  the rule would differ from decision 4 for every other inner estimator.

## Related

- ADR 0137: the online step forwards to the prior and the read-out runs the batch path.
- ADR 0164: the Allocation Set Constraint through which a leader honours the head's set.
- `benchmark_relative_returns`, `prior_step_returns`, `outer_returns_source`, `fold_returns`,
  `buffer_returns_result`, `ReturnsBufferState`, `PipelineStep`, `assert_benchmark_frame`,
  `covariance_forecast_evaluation`, `CovarianceForecastEvaluationResult`.
