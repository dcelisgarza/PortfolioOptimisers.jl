---
status: accepted
---

# An observation that a fit drops takes its weight with it, and the kept weights keep their type

## Context

Three fixes drop observations from a sample that carries observation weights:

- [#1622](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1622) drops each `NaN`
  observation of a numeric risk measure and evaluates the measure on the rest.
- [#1956](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1956) fits each asset, and
  each pair of assets, over its own finite rows under a `CoveragePolicy`. With observation weights,
  each fit needs the weights of its own rows.
- [#1960](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1960) drops the rows of a
  Shared Series gap from the returns data, and the fixed weights of the estimators keep the full
  length.

Issue #1622 stated why a rule must come first. The bias correction of a weighted variance depends
on the type of its weights, through `StatsBase.varcorrection`:

| Type | Divisor of the corrected variance |
| ---- | --------------------------------- |
| `FrequencyWeights` | `sum(w) - 1` |
| `AnalyticWeights` | `sum(w) - sum(w .^ 2) / sum(w)` |
| `ProbabilityWeights` | `sum(w) * (n - 1) / n`, `n` the count of nonzero weights |
| `Weights` | no correction |

A drop moves each divisor in a different way. A frequency weight is a count, not a probability, so
"normalise the remaining weights" has no single meaning across the types. A
`DynamicAbstractWeights` computes its weights from the sample, so it has no weights to drop until
it resolves. #1622 asked four questions: is a `NaN` row dropped for every type or refused for
frequency weights, does the dropped vector keep its type, are the kept weights scaled, and is the
value of a sample with no positive weight left `NaN` or an error. Issue #1959 factored the rule
out of #1622, so that each of the three fixes calls one function.

## Decision

**An observation that a fit drops takes its observation weight with it. The other observations keep
their weights, with their type, and with no scale.**
`observed_sample(X, w, obs; dims = 1, kwargs...)` in `src/01_Base/09_ObservationWeights.jl` holds
the rule. It returns the observations of `X` at which `obs` is `true`, and their weights from
`observed_weights(w, X_obs, obs)`. With no `obs`, `observed_sample(X, w; dims = 1)` observes each
observation whose entries are all finite, which serves a return series and one pair of assets. A
caller gives `obs` when the reason for the drop is outside `X`: the non-finite portfolio return
`X * w` of #1622, the Shared Series gap of #1960, and the active mask of ADR 0117.

1. **Every weight type drops the observation, frequency weights too.** A frequency weight counts
   the copies of one observation. A missing observation is missing in every copy, so the drop
   removes every copy, and the kept counts are the counts of the observations that remain.
2. **The kept weights keep their type.** `observed_weights` indexes the weights with the mask,
   `w[obs]`. `StatsBase` returns weights of the type of `w` for every type, `UnitWeights` included,
   so `varcorrection` gives the correction of that type on the observed sample. Each row of the table
   above is then the correction of the reduced sample, with no rule of the library in between. A
   `view(w, obs)` is a `SubArray`, which loses the type, so the rule never takes a view.
3. **The kept weights are not scaled.** The corrections of analytic, probability and plain weights do
   not change when the weights scale, and every weighted kernel divides by `sum(w)`. So a scale
   changes nothing for these types. A frequency weight is a count, so a scale makes it wrong.
4. **A `DynamicAbstractWeights` resolves on the observed sample.** It is a rule that computes the
   weights from the sample, so it reads the sample that the fit reads. A decay then ages each
   observation by the observations that the fit reads. This is the clock of
   [ADR 0181](0181-an-exponentially-weighted-covariance-holds-a-correlation-on-a-holiday.md), where
   each asset ages its observations on the count of its own valid observations, and of item 3 of
   #1622, where an OWA weight that is a function of `T` is evaluated at the reduced length.
5. **A sample with no observation of positive weight left has no value.** The consumer answers
   `NaN` and documents it. It does not warn and does not raise. `has_positive_weight(w, T)` answers
   the test. The `NaN` is the marker of a missing estimate that the library reads elsewhere: the
   `NaN` frame outside the Coverage Universe (ADR 0117), and the `NaN` below the first folded level
   (ADR 0172).

When `obs` keeps every observation, `observed_sample` returns the sample and the weights
themselves, borrowed in the sense of `get_observation_weights`. A weight vector of the wrong length
raises a `DimensionMismatch` that names `w`, and a mask of the wrong length raises one that names
`obs`.

### The cut of an estimator before the fit (#1960)

A Shared Series gap drops observations from the returns data before the fit, and the weights sit
in the fields of the estimator tree, not beside a sample. So the fix of #1960 cuts the tree.
`observed_weights(w, obs)` is the rule for one weights field before the fit. It checks the length
of `w`, indexes it with the type kept, and keeps a `DynamicAbstractWeights` and `nothing`, by
items 2 and 4. `shared_series_estimator(est, rows, T)` builds the mask of the full sample and walks
the tree with `obs_weights_view`, which gives each weights field to `obs_weights_index` and so to
this rule. `shared_series_fit` cuts the estimator and the returns data together, and the batch
entries of a prior, a regression, an uncertainty set, an optimiser, a pipeline group and a
covariance forecast evaluation call it. ADR 0061 and ADR 0107 record the change of the walk: the
observation channel opens on `@fprop`, and `@oprop` marks a child that only the cut reads.

A type outside the tags that holds a child which reads the observations needs a hand-written
`obs_weights_view`, or the walk stops at it. The optimiser, the uncertainty sets, the phylogeny and
centrality estimators of a solve and `PriorExpectedReturns` have one. Most of them cut every field
through `obs_weights_fields`. A field whose weights weight the assets is not cut. The bound `B` of
a `CentralityConstraint` can be a `MeanValue` over the assets, so its method cuts `A` alone.

A buffer fit of an online estimator drops the observations that its buffer marks, through
`shared_series_buffer`. A member that refits on the buffer can hold a fixed weight vector with one
entry for each buffered observation, so `shared_series_buffer_estimator` cuts it at the kept
observations of the buffer. A member that folds reads its own state, which never folded a marked
observation, and it refuses fixed weights, so `buffer_refit_member` leaves it as it is.

A fixed series of one value for each observation that is not a weight follows the same rule
(#1964). The benchmark returns of a `ReturnsTracking` are the one such series. A benchmark return
at a dropped observation drops with it, as a weight does. `ReturnsTracking` has a hand-written
`obs_weights_view` that gives its series to `obs_weights_index`. `TrackingRiskMeasure` reaches it
through `@oprop tr`, because `factory` must not replace the series, and `TrackingError` reaches it
through `@fprop tr`.

The pin of a Choice Rule on a buffer (#1964) fits the buffer with the marked observations dropped,
so it pins a copy of the prior cut by `shared_series_buffer_fit`. The online step keeps its
estimator for every later step, and a later buffer marks other observations, so the cut copy must
not become that estimator. `adopt_pinned_choice` writes the choice fields of the pinned copy, the
`included` of a `StepwiseRegression` and the `proj` of a `DimensionReductionRegression`, into the
uncut prior. The pin of a Cross-Sectional Factor Prior returns the dropped members alone, so it
fits the cut copy and writes nothing back.

The fixed vector of an `OrderedWeightsArray` is a weight for each sorted observation, not for each
row, so the cut does not reach it. It has a rule of its own with no gap (#1963): `owa_weights`
refuses a vector whose length is not the number of observations, in the value and in every JuMP
formulation.

## Consequences

- The fixes of #1622 and #1956 call `observed_sample` and `has_positive_weight`. The fix of #1960
  calls `observed_weights(w, obs)` on each weights field of the estimator tree. They state no rule
  of their own for the weights. #1956 can then fit an available-case sample under
  observation weights, in place of a refusal of the pair `w` and `cvg`.
- The extension contract of `DynamicAbstractWeights` states that a method receives only the
  observations that the fit reads.
- A frequency weight that sums to less than one after a drop gives a negative divisor. The guard of
  the weighted moment that returns `NaN` on a divisor that is not positive covers that case. This
  ADR does not change it.

## Rejected

- **Refuse a drop under frequency weights.** A count of copies has a clear meaning after a drop,
  so a refusal removes a valid fit.
- **Scale the kept weights to the sum of the full sample.** It changes nothing for analytic,
  probability and plain weights, and it gives frequency weights counts that no observation has.
- **Resolve a dynamic weight on the full sample and index it.** A decay then counts the dropped
  observations as time that passed with no data. ADR 0181 does not age an asset on its holiday,
  and the oracle that it ports freezes the state of the asset there. A user who wants the calendar
  clock can give a fixed weight vector, which keeps the weight of each date.
- **Raise on a sample with no positive weight.** ADR 0092 raises on a degenerate shrinkage sample,
  because that sample is a wrong input. A missing observation is data, and the library marks a
  missing estimate with `NaN`, so the Investable Mask can read it.

## Related

- [ADR 0043](0043-nothing-observation-weights-means-unweighted-not-unavailable.md): `nothing` means
  unweighted.
- [ADR 0117](0117-a-prior-reduces-to-the-coverage-universe-and-a-plain-moment-estimator-refuses-a-non-finite-sample.md):
  the Coverage Universe and the Shared Series drop.
- [ADR 0181](0181-an-exponentially-weighted-covariance-holds-a-correlation-on-a-holiday.md): the
  clock of a decay over a gap.
- [ADR 0202](0202-a-tail-measure-reaches-its-worst-weighted-loss-at-level-zero-and-a-risk-measure-checks-its-weights-against-the-data.md):
  the checks of the weights of a risk measure.
