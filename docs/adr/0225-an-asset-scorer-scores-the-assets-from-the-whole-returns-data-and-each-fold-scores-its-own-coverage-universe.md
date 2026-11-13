---
status: accepted
---

# An Asset Scorer scores the assets from the whole returns data, and each fold scores its own Coverage Universe

## Context

`ScoreSelector` and `RedundancySelector` score each asset with a risk measure on the training
window alone, through `asset_scores(r, X)`. Decision 4 of
[ADR 0029](0029-asset-selection-is-returns-preprocessing.md) refused a new score taxonomy, because
a wrapper of a risk measure carries no information. Issue
[#1931](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1931) found two scores that a
risk measure on one column cannot state:

1. **A fold-aggregated risk score.** Evaluate a risk measure on each test window of a
   cross-validation scheme, and aggregate the values. One window of bad luck then does not decide
   the rank of an asset.
2. **A weight-mass score.** Run an optimiser or a `Pipeline` through a cross-validation scheme,
   over one or more configurations, and score each asset by its mean weight or by how often it
   gets a weight. This score reads the whole cross-section of each fold, so it is not a function
   of one column.

The second score is the reason that the family cannot be a risk measure. A risk measure maps one
return series to one number, and the weight of an asset depends on every other asset of the fold.

## Decision

### The seam is `AbstractAssetScorer`, and a risk measure stays a valid score

`abstract type AbstractAssetScorer <: AbstractEstimator` is `public`, and its `# Interfaces`
section names two mandatory methods: `asset_scores(s, rd::AbstractReturnsResult) -> VecNum` and
`bigger_is_better(s) -> Bool`. The family has no default orientation. A risk measure defaults to
`false`, and that default stays. A scorer must state its orientation, because an ordinal rule cuts
the wrong end when the orientation is a guess.

A risk measure stays a valid `score` with no wrapper. `asset_scores(r, rd) = asset_scores(r, rd.X)`
is the in-sample case. The `score` field of both selectors widens to the named alias
`BaseRM_AsSc = Union{AbstractBaseRiskMeasure, AbstractAssetScorer}`. The two families share no
supertype, so the `Union` is the right bound. `assert_scoreable` on a scorer returns `nothing`,
and a scorer checks its inner measure in its own constructor. `ZeroVarianceFilter` does not
change.

### Configuration is in the fields

The scheme, the inner measure, the optimiser and the grid are fields of the scorer, and
`select_assets(sel, rd)` does not change. A selector is a `Pipeline` step, so a field has a
parameter path (`"select.score.cv"`, `"select.score.agg"`) that `search_cross_validation` can
tune. An argument of `select_assets` has no such path.

### `FoldRiskScorer` aggregates a Fold Statistic over the distinct test windows

`FoldRiskScorer(; r, sca = SumScalariser(), cv = KFold(), agg = MeanFoldScore(), min_folds = 1)`
evaluates `r` on each distinct test window of `split(cv, rd)`. `r` is one risk measure or a
vector of them, and `sca` reduces a vector as ADR 0231 and ADR 0232 state. A `CombinatorialCrossValidation` split holds each test
block in many splits, so the scorer counts each block once. A `MultipleRandomised` split draws
asset subsets, and the scorer ignores them, because the score of an asset reads its own column.

The aggregations are `MeanFoldScore`, `MedianFoldScore` and `WorstFoldScore`. Each holds a Fold
Statistic in its `alg` field: `Value()` keeps the raw value of the measure, and `Rank()` keeps the
rank of the asset in its window. The maintainer chose this shape over a separate `MeanFoldRank`:
every aggregation reads the raw value or the rank, so the statistic is a part that one swaps, not
a fourth aggregation. The family `AbstractFoldStatistic` is `public` and states two methods,
`fold_statistic` and `statistic_bigger_is_better`. An aggregation keeps the orientation of its
statistic. `Value` has the orientation of the measure, and `Rank` has `false`, because the best
asset takes the lowest rank. So `bigger_is_better(s) =
statistic_bigger_is_better(s.agg.alg, bigger_is_better(s.r))`, and the aggregation family needs no
orientation verb of its own. `WorstFoldScore` takes the maximum of a statistic whose orientation
is `false`, and the minimum otherwise.

`Rank` is the centred midrank percentile `(n_lt + n_le) / (2n)` of `CrossSectionalPercentileRank`,
computed with the same `cross_sectional_rank_counts`. A tie shares the average rank. The
percentile is free of the scale of the window, so a volatile window weighs no more than a calm
one. It is also free of the number of covered assets, which matters under the next decision.

`Value` is a singleton, and ADR 0222 spells "the raw statistic" with `nothing`. The type stays
because it names a rule with content of its own. It is one of two peers that a field swaps, and it
states the orientation of the statistic through its own method of `statistic_bigger_is_better`. A
`nothing` would move that method to `Nothing`, and a reader of `MeanFoldScore(; alg = nothing)`
would not see that the score keeps the units and the orientation of the measure.

### `WeightMassScorer` counts each split once, over the assets it offered

`WeightMassScorer(; opt, p = nothing, cv = KFold(), tol = nothing, min_folds = 1, ex)` runs
`cross_val_predict` on each candidate of the grid `p`. It expands the grid with `lens_val_grid` or
`pipeline_lens_val_grid` and `search_candidate`, the code of the search, so one grid syntax holds.
It checks each candidate at construction with `assert_search_candidates`. `tol = nothing` scores
the signed mean weight, and a number scores the selection frequency `|w| > tol` (ADR 0222).

- **Each split counts once.** The scorer pins the draw of `cv` with `pin_draw`, splits the data
  once, and finds the prediction of each split by the first row of its test window. Under
  `CombinatorialCrossValidation` it reads the first block of the split in the path that holds it.
  Under `MultipleRandomised` it reads the fold in the path of the split.
- **Offered is not weighted.** A split offers the Coverage Universe of its training window,
  restricted to the subset that a `MultipleRandomised` split drew, plus every asset that the fold
  made investable. An asset that a selector inside `opt` drops was offered, and counts with weight
  zero. An asset outside the subset was not offered, and does not count. The scorer maps the
  weights to the assets by the names of the investable assets of the prediction, because a
  selector or a subset makes the universe of the fold smaller than `rd`.
- **A failed split leaves both the sum and the count**, as a failed candidate takes no part in
  the search (ADR 0120).
- The cost is `C × K` optimisations, and inside a search it multiplies by the outer folds and
  candidates. The docstring states it, and the scorer keeps no cache.

### Missing data: each fold scores its own Coverage Universe

The issue first refused an `rd.X` with a non-finite entry. The maintainer overruled it: the rest
of the library is robust to missing data, so the scorers follow the precedent of ADR 0117 and
ADR 0120. Each test window of `FoldRiskScorer` scores only the assets of its Coverage Universe,
through `cv_live_assets`, the verb that draws the subsets of `MultipleRandomised`. A rank then
ranks only the covered assets. Each split of `WeightMassScorer` offers only the Coverage Universe
of its training window. So an asset with a gap in one fold keeps its score from the other folds.

Both scorers take `min_folds`. When fewer than `min_folds` folds scored an asset, `asset_scores`
writes `NaN` at that asset, through `min_folds_scores`. The default `1` marks an asset that no
fold scored. Every selection rule and every redundancy algorithm drops a non-finite score
(ADR 0029, amendment of 2026-11-02), so the asset leaves the selection and the run does not
stop, as the rest of the library is robust to missing data. A selector with no scored asset
keeps no asset, and `fit_preprocessing` throws `IsEmptyError`. Inside a selector the training
window is already reduced to its Coverage Universe, so the case occurs on a direct call, under
`MultipleRandomised`, and with a `min_folds` above one.

The drop has a cost that the docstring of `WeightMassScorer` states. A draw of asset subsets can
miss an asset completely, and a selector then drops it only because of the draw. The library
documents a degenerate value and does not warn on it (#1282).

## Considered options

| Question | Refused | Why |
| --- | --- | --- |
| The score | A wrapper `RiskMeasureScore(r)` for the in-sample case. | It carries no information, the reason of ADR 0029. A risk measure stays a valid `score`. |
| The orientation | A fallback `bigger_is_better(::AbstractAssetScorer) = false`. | The two scorers disagree on it, and a guess cuts the wrong end of an ordinal rule. |
| The aggregation | A mean over σ ("information ratio"). | For a risk measure the ratio has no orientation. For a reward that crosses zero, the sign of the ratio flips with the sign of the mean. |
| The aggregation | `MeanFoldRank` beside three value aggregations. | Every aggregation can read the rank, so the statistic is a field of each. |
| The rank | The plain rank `1, …, n` of each window. | A window that covers fewer assets has a smaller range of ranks, so a mean over windows weighs it less. The percentile is free of `n`. |
| Missing data | Refuse a non-finite `rd.X` with `IsNonFiniteError`. | The library reduces each fit to its Coverage Universe, and a scorer that refuses breaks that contract for a direct call. |
| Missing data | Throw an `ArgumentError` that names each asset with fewer than `min_folds` folds. | It stops a run on a gap in the data, and the rest of the library drops the asset. The rules exclude a `NaN` since #1933, so a `NaN` no longer takes a place in an order. |
| The splits | Read the paths of `cross_val_predict`. | One combinatorial split is in many paths, so a path count weighs some splits more than others. |
| The weights | Fit each split by hand with `optimise`. | The fold loop resolves `TimeDependent` schedules and threads the previous weights. A second loop drifts from it. |

## Consequences

- Decision 4 of ADR 0029 is amended: the family is new, and a risk measure stays the in-sample
  score.
- `GLOSSARY.md` gains **Asset Scorer**, **Fold Aggregation** and **Fold Statistic**, and the
  **Selection Rule** and **Coverage Universe** entries name the scorers.
- `public`: `AbstractAssetScorer`, `AbstractFoldStatistic`, `AbstractFoldAggregation`,
  `fold_statistic`, `statistic_bigger_is_better`, `fold_aggregate`. Exported: `FoldRiskScorer`,
  `WeightMassScorer`, `MeanFoldScore`, `MedianFoldScore`, `WorstFoldScore`, `Value`, `Rank`.
- The scorers live in `src/21_Pipeline/07_AssetScorers.jl`, after the pipeline, so `opt` can be
  bound to `Union{OptE_TD, Pipeline}`. The seam stays in `src/20_AssetSelection.jl`.
- `examples/1_foundations/03_Asset_Preselection.jl` shows a `FoldRiskScorer` under `Value` and
  under `Rank`, and a `WeightMassScorer` that cuts the best quantile.
