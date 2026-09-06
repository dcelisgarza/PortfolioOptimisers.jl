---
status: accepted
---

# A fold zeroes a Held Gap once, and a value-level verb reduces to the Investable Mask

## Context

[ADR 0115](0115-every-optimisation-estimator-reduces-once-at-its-entry-and-its-result-carries-the-investable-mask.md)
reduces every optimisation to the Investable Mask at its entry, so no optimiser meets a gap. The
mask comes from the training window. Three consumers still meet one.

- **A fold's test window.** A walk-forward over a panel with listings and delistings fits on one
  window and scores on the next. An asset that delists inside the test window carries a non-zero
  weight and then a `NaN` return. The fold forms its series at
  `src/20_Optimisation/02_CrossValidation/01_Base_CrossValidation.jl:1435` through
  `calc_net_returns`, which is the plain product `X * w`, and `0 * NaN` is `NaN`. The census of
  [#671](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/671) measured it: the library
  and the reference implementation agree on every observation until the delisting and disagree on
  every observation after it, and a weight of exactly zero on the dead asset poisons the date all
  the same.
- **A value-level verb against a Prior Result.** `expected_risk(r, w, pr)`, `expected_return`,
  `risk_contribution` and `factor_risk_contribution` take weights of the full length against a
  result on the full universe. A non-investable asset carries `NaN` in `mu`, on the diagonal of
  `sigma` and in its column of `pr.X`, so `dot(w, pr.sigma, w)` and `pr.X * w` are `NaN` at any
  weight. The optimiser's own result on a gapped panel cannot be scored by hand.
- **A young asset under a mask-aware estimator.** The whole-window rule of
  [ADR 0117](0117-a-prior-reduces-to-the-coverage-universe-and-a-plain-moment-estimator-refuses-a-non-finite-sample.md)
  puts an asset in the Coverage Universe only when every row of its window is finite, so a plain
  prior never carries a `NaN` row for an investable asset. The exponentially weighted family that
  the map ports is mask-aware: an asset that lists inside the window is investable once its warm-up
  is met, while its early rows of the prior's returns matrix are still `NaN`. The JuMP model reads
  those rows as constraint coefficients.

Two facts shape the answer. The tail measures answer a **finite wrong number** on a gapped series:
`partialsort` orders a `NaN` after every real, so a CVaR reads its order statistic off the finite
prefix and divides by the poisoned length. The census measured a gapped CVaR that reports the answer
for the ten observations it did not have. And the library has no `NaN` policy in the risk measures
at all: 29 files, zero finiteness checks.

The reference implementation makes its portfolio return series always finite at one line,
`portfolio/_portfolio.py:592`: it replaces every gap by zero and takes the plain dot product, with
no renormalisation and no diagnostic. That is why its measure layer can afford to be permissive. Its
empirical prior zero-fills the missing scenarios of an investable asset and warns above a 5% share,
and its warning names the cost: the fill understates that asset's risk in a scenario-based measure
and leaves `mu` and the covariance untouched. Its market return renormalises to the live set
instead, and the reference never reconciles the two conventions. A caller of the reference cannot
learn that a third of the portfolio sat in cash.

The library has its own precedent. The factor attribution, decided on
[#844](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/844), forms its realised series
per finite pair: a zero weight at a gap is silent, and a held gap is zeroed and named through
`strict_diagnostic`, a warning by default and an `ArgumentError` under `strict`. The maintainer
also refused, on this ticket, a finiteness check on a hot path when the caller can clean the value
upstream: a scan is paid only where the library itself makes the gapped value and the caller cannot
reach it first, and it is paid once there.

## Decision

### A fold reduces its test window to the Investable Mask, then zeroes a Held Gap once

A fold's test window holds two kinds of gap, and the fold takes them in order. The first is the
column of an asset the fit found non-investable, whose weight is `0` by ADR 0115. In `predict`,
before anything reads the window, the fold views the test window and the weights at `res.imsk`,
so that column is never read, and the fees are viewed with them. The Investable Mask is the one
record of which assets the fold traded, and this is the rule of ADR 0115 carried to the window the
weights are scored on. A result whose `imsk` is `nothing` views nothing.

The second is a **Held Gap**: an (observation, asset) pair at which the portfolio's weight is
non-zero and the asset's return is missing. It arises where the universe changes after the fit, an
asset that was investable on the training window and delists inside the test window, and the mask
is a per-fit fact that cannot see it. The fold filters the reduced window once, before it forms the
series and before the Weight Drift compounds: every non-finite entry becomes `0`, a zero weight at a
gap is silent, and the Held Gaps are named through `strict_diagnostic`. There is no renormalisation,
so the missing weight sits in cash on that observation, which is the one reading that invents no
trade the weights never stated. The filtered window feeds `calc_net_returns` and
`held_weights_result` alike, and the held weights after the last observation expand back to the
full length, because the next fold's turnover reads them. `strict` is a field of every
cross-validation estimator beside `wd` and `store_weight_path`, and a keyword of `predict`.

The identity a test pins, with the fee taken over the full weight vector:

```text
returns[t] == sum_i w_i * (isfinite(X[t, i]) ? X[t, i] : 0) - fee
```

`calc_net_returns` and `calc_net_asset_returns` stay the plain product. Their docstrings state
that a non-finite entry poisons its observation, and that a gapped panel is scored through
`predict(res, rd)`. The funnel has 70 call sites in 16 files, the hierarchical solves and the risk
contribution among them, and most hand it a finite matrix. A scan there is paid at every one of
them on every evaluation. A scan in the fold is paid once, over the investable columns alone, on a
window the fold already multiplies once, at the one place the library slices a window the caller
cannot reach first. Ticket
[#674](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/674) put the mask view before
the filter, so that both doors of this ADR reduce to the Investable Mask.

### A value-level verb against a Prior Result reduces at its entry

`expected_risk`, `expected_return`, `risk_contribution` and `factor_risk_contribution`, in their
prior-taking methods, derive the Investable Mask once, name a held non-investable asset through
`strict_diagnostic`, slice the weights and the prior through `port_opt_view`, and expand a per-asset
answer into a zero vector of the full length. That is the rule of ADR 0115 at the value-level door,
through a third method of `investable_reduction` that takes the prior, the weights and `strict`.
Every block of the prior is reduced together by the `port_opt_view` method the prior's owner
already writes, so a new block or a new measure cannot be forgotten, and the reduced `pr.X` has no
dead column, so nothing of a non-investable asset reaches the funnel. A bare returns matrix entry is
unchanged.

### A series the caller holds is documented, not checked

`expected_risk_from_returns`, `performance_summary(ret)`, `cumulative_returns` and `drawdowns` take
no finiteness check. Every one has internal callers that hand it a finite series: the three ratio
kernels call the first for each half at every evaluation, and the performance summary and nine plot
sites call the others on the funnel's output. Their docstrings state the Precomputed-returns
contract: the series must be finite, a tail measure on a gapped series answers a finite wrong
number, and `x[isfinite.(x)]` before the call reproduces the reference implementation's
drop-per-column answer exactly, for every kernel but its unbiased semi-variance, whose Bessel
correction reads the full length and is a defect the census recorded.

### A mask-aware prior zero-fills the missing rows of an investable young asset once per fit

The prior that reads the Asset Panel writes `0` at every (row, asset) pair where the asset is
investable and its return is missing. A non-investable asset keeps its `NaN` column, so the
Investable Mask is unchanged, and `mu` and `sigma` are untouched because the mask-aware estimator
computed them from the rows it saw. Every consumer, the JuMP model, the meta-optimisers and the
value-level door among them, then reads finite investable columns. The fill is paid once, on the
estimator's own pass.

Under `strict = false` the prior warns when the filled fraction exceeds `SCENARIO_FILL_LIMIT`, a
scoped config after the pattern of `STRING_DISTANCE`, with `set_scenario_fill_limit!`,
`with_scenario_fill_limit` and the preference key `"scenario_fill_limit"`, default `0.05`. The
fraction is the count of filled entries over the count of entries of the returns matrix, the
reference's denominator. The warning names the assets, the count and the consequence for a
scenario-based measure. Under `strict = true` any fill refuses, whatever the fraction.

### A drawn plot keeps the frame, and a computed plot reduces

A heatmap or a bar chart of a Prior Result, `plot_mu`, `plot_sigma`, `plot_correlation`,
`plot_prior`, the factor plots, `plot_coskewness` and `plot_cokurtosis`, draws the full universe and
the backend leaves a blank for a non-investable asset. A plot that computes on the matrix,
`plot_eigenspectrum`, `plot_network` and `plot_centrality`, reduces to the Investable Mask first.
A series plot reads the funnel, and its prediction methods are finite by the fold's filter. A
per-asset plot keeps the gap, which draws as a break. A weight plot draws the zero.

### The tripwire is a census test by reflection

One test walks the concrete subtypes of `AbstractBaseRiskMeasure`. For every measure with a
value-level method, on a Prior Result with one non-investable asset: a zero weight there gives the
same answer as the measure on the result fitted with that column removed by hand, the oracle of
`test/test_50_investable_reduction.jl`; a held weight there gives one warning and the zeroed
answer, and an `ArgumentError` under `strict`; a per-asset answer has the full length with exactly
`0` at the dead asset. A second testset drives one fold with a delisting inside its test window and
asserts a finite series and one warning.

### The refused options

| Option | Why it was refused |
| --- | --- |
| The fold filters the full window, with no view at the Investable Mask first | The same series, but the filter scans a dead column the fit already excluded, and the fold and the value-level door of this ADR then reduce by two different rules. |
| A held gap renormalises the live weights to the held budget, the reference's market-return convention | It rebalances into the survivors on that observation, which the caller's weights never said, and the series is no longer linear in the weights. |
| A held gap refuses with a named error, the census's recommendation | The first delisting inside a test window stops the walk-forward, so the map cannot close. |
| Any gap refuses | The optimiser's own result on a gapped panel cannot be scored, because the dead asset's zero weight still meets its gap. |
| The funnel scans the product and takes a slow path on a `NaN` | Paid at 70 call sites on every evaluation, on the hierarchical solves and the risk contribution among them. |
| The funnel scans the whole matrix, the attribution's pattern | The same, at a higher order. |
| The seam zeros the rows of every bound array | One rule per block, `mu`, `sigma`, `sk`, `kt`, the columns of `X` and the factor block; a forgotten block is a silent `NaN`, and the seam does not see the weights. |
| Each weights-only functor guards its own array | Six sites, and a new measure forgets. |
| A finiteness check at the doors that take a caller's series, or in every kernel | A scan on a large series is paid by every internal caller that already hands a finite one; the caller cleans a value they built with one line. |
| The doors compact the series, as the reference does | A denominator that changes with the gap count and that nothing records, an all-gap series that answers `NaN` in silence, and a hole to put back for the cumulative and the drawdown forms. |
| The optimiser's reduction zero-fills the young asset's rows | A second site, and the value-level door and the meta-optimisers each need their own. |
| A young asset is non-investable until its window is finite | It undoes the mask-aware family: the estimator that handles a listing is refused on every listing. |
| The fill is silent, or it warns on any fill | Silent leaves the caller with the docstring alone; any fill warns on most folds of a walk-forward over a panel with listings. |
| Every prior plot reduces, or every prior plot draws the frame | One rule loses the blank row that shows a dead asset, the other fails the plots that compute on the matrix. |
| A hand-written list of measures in the tripwire | A new measure is forgotten in the list, which is the failure the tripwire exists to catch. |

## Consequences

- Three build tickets on map
  [#667](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/667): the fold's filter, the
  value-level doors and the census test in one; the plots in a second, blocked by the first; the
  fill and its scoped config in a third, blocked by the two moment build tickets. All three block
  the closing verification.
- The map's closing walk-forward can score: a fold whose test window holds a delisting yields a
  finite series and one warning, and the search scores a number, not a `NaN`. What a search does
  with a `NaN` score from a failed fold is the cross-validation decision of the map.
- The invested fraction shrinks on a gapped observation, and the warning is what says so. A caller
  who wants the honest two-asset portfolio zeros the weights over the observations the asset is
  inactive, or passes a weight history.
- A scenario-based measure understates the risk of a young asset over its filled rows under a
  mask-aware estimator. The docstrings of the family state it, and `SCENARIO_FILL_LIMIT` is where a
  caller tightens the notice.
- `CONTEXT.md` gains the **Held Gap** entry, and the **Precomputed-returns contract** entry states
  the finiteness rule.
