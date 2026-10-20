---
status: accepted
---

# The paths that read the frame of a head read the source of a reader

## Context

ADR [0211](0211-each-reader-of-the-returns-states-its-own-source.md) removes `brt` from every head
and puts `x_src` on each reader estimator of the returns. ADR
[0212](0212-a-returns-source-is-a-singleton-type.md) gives the field the values `nothing`,
`DataSource()`, `ExcessSource()` and `PriorSource()`. ADR
[0214](0214-the-readers-of-one-head-can-read-different-frames.md) lets the readers of one head mix
frames. Some paths read the frame of a head and are not a reader estimator. This ADR decides where
each one gets the frame (#1683).

The facts at `37613955bc`:

- **The online fold.** `fold_returns` (`src/17_Optimisation/09_OnlineOptimisation.jl`) reads
  `opt.brt` for two jobs. `fold_prior` folds `X - B` into the buffer of the prior through
  `prior_step_returns`. `fold_context` keeps the raw rows in the Fold Context only under `brt`,
  because the buffer of the prior then holds `X - B`. Only the prior folds. The other readers of the
  head refit from the raw `rd` that `batch_from_state` rebuilds, so each one reads its own field.
  The online `Pipeline` folds a prior owner through the same `fold_prior`, with the `brt` of its
  `PipelineStep`.
- **The drop of a Shared Series gap.** A vector `B` with a `NaN` row makes that row of `X - B` all
  `NaN`. `optimiser_shared_series_view(rd, opt)` drops such rows one time, before the solve, under
  `opt.brt`, so every reader of the head pairs the same observations (ADR 0117). The online step
  keeps the row and marks it under the same `brt`, and `batch_from_state` cuts the rebuilt returns
  data with the same function (#1675).
- **The outer problem of a meta-optimiser.** `outer_returns_source(pr, rd, brt)`
  (`src/17_Optimisation/06_Meta/01_Base_MetaOptimisation.jl`) serves the path with no folds of
  `NestedClustered` and `Stacking`. It reads `pr.X`, the returns of the meta's own Prior Result.
  Under `brt` these are `X - B`, so it adds `B` back. The CV path reads no frame: it predicts each
  inner head on the raw `rd`. `prior` returns a precomputed Prior Result unchanged, so today a meta
  with `brt = true` declares that a precomputed `pr.X` is in the Excess Frame. `SubsetResampling`
  has no outer problem.
- **The covariance forecast evaluation.** `brt` is a keyword of the verb
  `covariance_forecast_evaluation(est, rd, cv; ..., brt)`. The verb is the only part of the call
  that receives `rd`. The estimator receives a matrix, so ADR 0211 gives it no field.
  `CovarianceForecastEvaluationResult` records `brt` by position. `covariance_forecast_compare`
  refuses two evaluations with different `brt`, and `covariance_forecast_portfolio` reads its test
  rows under `cfer.brt`.
- **The Pipeline.** `pipe_prior_brt` reads the frame of the optimiser for `assert_benchmark_frame`
  alone. ADR 0214, decision 6, deletes both.

## Decision

The maintainer decided on 2026-10-13, and decided decision 3 again on 2026-10-14 (#1701).

1. **The online fold reads the source of the prior that it folds.** `fold_returns` reads
   `pe.x_src`. Under `ExcessSource()` the buffer of the prior gets `X - B`, and the Fold Context
   keeps the raw rows. Under `nothing` or `DataSource()` the buffer holds the raw rows, and the
   context keeps no copy. A prior owner of an online `Pipeline` reads the `x_src` of the prior
   estimator of its step in the same way.
2. **A head drops a gap of a vector benchmark when any reader of its group states
   `ExcessSource()`.** The group is the group of ADR 0214, decision 5. In batch,
   `optimiser_shared_series_view` drops the rows one time, before the solve. Online, the step marks
   the rows by the same rule, and `batch_from_state` cuts by it. A raw reader of the group loses the
   rows too, so every reader of the head pairs the same observations. The gap of the factor returns
   keeps its rule, which reads the prior. A fallback is a head with its own group, so it drops by its
   own rule, as today. A `Pipeline` follows the same rule, because its steps join the group of its
   optimiser (ADR 0214, decision 6). Each step whose value the optimiser receives, and the
   optimiser, reads the returns data with the gaps of the group dropped: a prior step, a clustering
   step, an uncertainty step and a constraint step alike. Online, a prior owner marks the gaps of
   the group, and an optimiser owner folds under the source of the group of the Pipeline. A step
   that the optimiser does not receive drops by the rule of its own reader. The maintainer decided
   this on 2026-10-20 (#1774), for one default in a head and in a `Pipeline`.
   **The gap rule decides which observations a reader reads, never its frame.** Each reader reads
   the frame that its own `x_src` states, so the readers of one group can still mix frames (ADR
   0214, decision 1). The one rule about the frame is the refusal of a double subtraction of the
   benchmark: a `DataBenchmarkTracking` beside a prior at `ExcessSource()`, in a head or through a
   prior step of a `Pipeline`, is refused at construction.
3. **The outer problem reads the source that the Prior Result records.** `LowOrderPrior` holds
   `x_src`, and a `HighOrderPrior` answers it from the `LowOrderPrior` that it wraps, as it answers
   `X`. `prior(pe, rd)` and the read-out `prior(pe)` of an online fold record the source of `pe`,
   and a returns-matrix method records `nothing`. `outer_returns_source` adds `B` back when the
   Prior Result records `ExcessSource()`, and otherwise reads `pr` as before. So the read-out of an
   online fold, a prior step of a `Pipeline` and a precomputed Prior Result each keep their frame.
   The frame walk of ADR 0214 counts a Prior Result in the place of a prior by the source that it
   records, with no raw reader at `nothing`. So a data benchmark refuses an excess Result (ADR
   [0208](0208-a-data-benchmark-binds-to-the-rows-that-each-reader-holds.md), decision 3). The CV
   path does not change. The heads piece (#1701) found the cause: `batch_from_state` puts the
   read-out in `opt.pe`, so the outer problem of a `NestedClustered` in the Excess Frame lost the
   `+ B`, and its weights differed from the batch fit by 0.0092.
4. **`covariance_forecast_evaluation` takes an `x_src` keyword.** This is the one exception to ADR
   0211, decision 1. The verb is the outermost reader of the evaluation, as the step of
   `OnlinePortfolioSelection` is for its rules: no estimator in the call receives `rd`. The keyword
   admits `nothing`, `DataSource()` and `ExcessSource()`, and the default is `nothing`. The shared
   assertion of ADR 0212, decision 6, refuses `PriorSource()`, because the verb reads no prior.
5. **The Result records the resolved frame.** `CovarianceForecastEvaluationResult` holds
   `x_src::AbstractReturnsSource` in the position of `brt`. The verb records `DataSource()` for
   `nothing`. The field type refuses a `Bool`, so a positional `Bool` builds no Result (ADR
   [0213](0213-brt-is-removed-in-one-break.md), decision 4). `covariance_forecast_compare` refuses
   two evaluations whose `x_src` differ, by equality. `covariance_forecast_portfolio` reads its test
   rows with `cfer.x_src`.
6. **Sites that need no frame.** `pipe_prior_brt` and `assert_benchmark_frame` go (ADR 0214,
   decision 6). The corner solves of `NearOptimalCentering` use the readers of the head, so each
   one reads its own field. The CV path of a meta-optimiser predicts on the raw `rd`. The step of
   `OnlinePortfolioSelection` keeps its own `x_src` (ADR 0211, decision 7).

## Consequences

- After the migration builds, no code reads a frame from a head. Decisions 4, 5, 7 and 8 of ADR
  [0204](0204-a-returns-result-drops-no-field-that-the-mathematics-lets-it-carry.md) describe the
  head-level `brt`, and the map supersedes them when the builds land.
- `fold_returns` reads two inputs: `pe.x_src` for the frame of the fold, and the group rule for the
  marks of a gap. The walk of ADR 0214 serves the warning and the gap rule, so one shared function
  answers both.
- A raw prior in a group with an excess reader loses the gap rows of `B`. So its Prior Result inside
  the head can differ from `prior(pe, rd)` on the same data, which keeps those rows. The same holds
  for a raw prior step of a `Pipeline` in such a group, so the Prior Result of the step equals the
  Prior Result that the same head fits.
- A precomputed Prior Result keeps the frame that its prior stated, because `prior(pe, rd)` records
  it. A caller who builds a Prior Result by hand states its frame with
  `forward_prior(pr; x_src = ExcessSource())`. A printed `LowOrderPrior` shows an `x_src` row.
- The pins of `test/test_78_brt_pinned_outputs.jl` stay valid when the helpers state
  `ExcessSource()` on the prior of each head and of each meta-optimiser. Then the group rule and
  the rule of the prior agree.
- #1691 found that `covariance_forecast_portfolio` indexes `cfer.test_idx` into rows where no gap
  is dropped. The build of decision 5 reads the rows through the same drop as the evaluation.

## Alternatives considered

- **The Fold Context always keeps the raw rows.** The context then reads no frame, but under a raw
  prior two buffers hold the same matrix.
- **The prior decides the gap drop.** It is simpler, but a graph or regression reader in the Excess
  Frame beside a raw prior reads `NaN` rows.
- **Each reader drops its own gap rows.** The readers of one head then read different observations,
  which the head drop of ADR 0117 exists to prevent.
- **A step of a `Pipeline` drops by its own reader** (the code before #1774). The prior step and the
  uncertainty step dropped under their own source, so a raw prior step in an excess group kept the
  gap rows that the optimiser dropped. The Pipeline then differed from the same head, and a head
  and a Pipeline had two defaults for one configuration.
- **The outer problem reads the source of the meta's prior estimator** (the decision of
  2026-10-13). Three paths put a Prior Result in the place of the estimator: the read-out of an
  online fold, a prior step of a `Pipeline`, and a precomputed Result. Each one read as raw, so the
  outer problem lost the `+ B`.
- **A keyword through the call.** `kwargs...` reach every attempt of the fallback chain and every
  inner optimiser, so the keyword must be stripped two times, and it repairs the read-out only.
- **A positional argument.** The fallback chain passes the same `args...` to each attempt, so a
  `MeanRisk` fallback gets a `MethodError`.
- **A frame field on `ReturnsResult`.** It is a larger change than one field.
- **A refit of the prior in the two meta-optimisers.** It loses the read of the folded state.
- **A field on `HighOrderPrior` too.** The first text of the decision gave both types a field.
  `HighOrderPrior` holds none of the returns fields and forwards them to the `LowOrderPrior` that
  it wraps, so a second copy of the source could disagree with the first.
- **The outer problem always reads the raw `rd.X` on the rows of `pr`.** It needs no frame and it
  agrees with the CV path. But under a factor prior `pr.X` is a reconstruction, so the outer problem
  moves with no benchmark too, which needs a re-pin build past this map.
- **The evaluated estimator holds `x_src`.** A covariance estimator is an inner reader everywhere
  else, and ADR 0211, decision 3, makes an inner reader keep `nothing`.
- **The scheme `cv` holds `x_src`.** The split reads no returns.
- **The Result records the value as stated.** The field is then `Option{<:AbstractReturnsSource}`,
  and the compare must treat `nothing` and `DataSource()` as one frame.
- **The keyword defaults to `DataSource()`.** It is the only `x_src` with a default other than
  `nothing`.

## Related

- ADR 0117: the drop of a Shared Series gap, and the marks of the online step (#1675).
- ADR 0204: the rule for `brt`, which this map supersedes.
- ADR 0211: the readers that hold `x_src`, and decision 1, to which decision 4 is the exception.
- ADR 0212: the values of `x_src`, and the shared assertion.
- ADR 0213: `brt` is removed in one break, and a positional `Bool` builds no Result.
- ADR 0214: the group of a head, and the warning of a forgotten raw reader.
- `fold_returns`, `fold_prior`, `fold_context`, `optimiser_shared_series_view`, `batch_from_state`,
  `outer_returns_source`, `covariance_forecast_evaluation`, `CovarianceForecastEvaluationResult`,
  `covariance_forecast_compare`, `covariance_forecast_portfolio`, `pipe_prior_brt`.
