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

Three measurements taken on this ticket, after available-case admission landed, redraw the fill's
half of the answer. **The matrix-wide denominator is blind to the case it was written for**: the
share is the count of filled entries over the count of entries of the whole returns matrix, the
reference's denominator, so one asset of a hundred whose column is seven-tenths invented is seven
thousandths of the matrix, beneath any limit a caller would set. **The limit and the coverage floor
are the same number**: `admits` reads coverage as an asset's own observation count over the number
of observations folded, and the fill counts that column's non-finite entries over the same
denominator, so the two are complements, and a limit that did not know it would fire on nearly
every fold of an available-case walk-forward. And **the fill reaches further than the tail**: a
census of the readers of `pr.X` found that 48 of the 54 concrete risk measures read the sample
rather than a moment alone, and that three consumers outside the measure layer read the invented
cells too — the dendrogram, which recomputes its correlation from the returns matrix and not from
`pr.sigma`; the entropy-pooling view resolvers, which read the matrix column by column; and the
meta-optimisers, which build the outer problem's returns from it.

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

Under `strict = false` the prior warns when the filled share exceeds `fill_limit`, a **field of
`EmpiricalPrior`** holding an `Option{<:Real}` in `(0, 1]`, default `nothing`. Under `strict = true`
any fill refuses, whatever the share.

**The share is per asset**: the worst investable column's own count of filled entries over the
number of observations. The matrix-wide share, the reference's denominator, is reported in the
message and trips nothing. It scales with the universe, so the column the notice exists to catch
disappears inside it — one asset of a hundred whose column is seven-tenths invented is seven
thousandths of the matrix, under any limit a caller would set. A per-column denominator means one
number with one meaning in a panel of ten assets and in a panel of a thousand.

The limit is a field rather than a global because the decision it governs is a property of **one
fit**. A caller who wants two priors in one program to fill at two different shares says so on the
two estimators; dynamic scope could not express that, because it colours whatever fits inside its
block rather than the estimator that fills. This is the shape `strict` already takes in this ADR,
and the shape the library gives every other piece of configuration: the value is inspectable on the
estimator and it prints under `show`.

`EmpiricalPrior` is the only estimator that carries the field, and it is the only one eligible. A
prior is eligible when it holds a mask-aware moment estimator **directly** and puts the caller's own
returns matrix into the result's `X`. Every other low-order prior wraps an inner prior estimator and
synthesises its `X` — a posterior matrix, an expanded matrix, or a reweighted one — so the fill is
paid once, at the `EmpiricalPrior` at the bottom of the chain, and the field rides there through
`factory`. `CrossSectionalFactorPrior` holds a `ce` directly, but it estimates the covariance of
standardised idiosyncratic residuals, carries its own investability guard, and synthesises its
scenarios, so it never reaches the verb.

### `fill_limit` and `min_coverage` are one number

`admits` reads an asset's coverage share as its own observation count over the number of
observations folded, and the fill counts that column's non-finite entries over the same denominator.
So `filled_share == 1 - coverage_share` identically, and the admission test `share >= min_coverage`
**is** the test `filled_share <= 1 - min_coverage`. Two spellings of one number. A `fill_limit` that
ignored that would fire on nearly every fold of an available-case walk-forward, naming the caller
for doing precisely what they configured — which is the failure the old global's 5% default had, in
a new spelling.

`nothing` therefore derives. Where any arm of the fitting estimator carries a
[`CoveragePolicy`](0117-a-prior-reduces-to-the-coverage-universe-and-a-plain-moment-estimator-refuses-a-non-finite-sample.md),
`nothing` means `1 - maximum(min_coverage)` over the arms that state a floor, and it never fires:
every admitted column satisfies it by construction. The **maximum** is the binding floor because
admission is the conjunction of the arms — the Investable Mask needs `mu` and the diagonal of
`sigma` finite — and `coverage_admission` reads the per-asset count off the diagonal, the same
number for both arms. A floor stated on **either** arm therefore bounds every investable column
whatever the other arm does, so a mixed configuration needs no rule of its own.

Where **no** arm states a floor, `nothing` keeps its original meaning and names every fill. The
exponentially weighted family is mask-aware without a policy: it gates on `min_obs`, a *count*
whose default is about six observations, which says nothing about the share of a thousand-row
window. A caller who set no floor has weighed no trade, and is told that the fill happened.

An explicit `fill_limit` overrides the derivation and must be **tighter** than admission. A value
above `1 - min_coverage` is refused, because it is dead by construction: nothing that reaches the
fill could trip it, and a knob that cannot fire is worse than no knob. What an explicit value buys
is the one configuration the derivation cannot express — admit broadly and be told anyway.
`min_coverage = 0.3` with `fill_limit = 0.5` admits a column seven-tenths invented and names it
past half.

`0` is not a value the field accepts — `nothing` already spells that answer, and a second spelling
of one answer is a defect waiting to be found. The upper end is closed: `1` accepts the whole
matrix.

`min_coverage` keeps its default of `0`, which derives a limit of `1`, so a policy taken bare
admits a one-observation column and fills it in silence. That is what `0` asks for. The notice fires
on a floor the caller stated and the fit violated, never on a floor the caller declined to state,
and `CoveragePolicy`'s docstring says so where the default is written.

### The notice names every consumer of a filled column, not the scenario measure alone

The fill is paid once so that every consumer reads a finite investable column, and every consumer
therefore reads the invented cells. The message names four consequences, one per consumer that a
census of `pr.X`'s readers found, because a caller who is told only about the tail will not look for
the other three:

- **A scenario-based measure understates the asset's risk.** The cost is exact rather than
  qualitative, and the invented zeros are not the reason a reader might expect. They do not enter
  the tail; they inflate the denominator. A measure at level `alpha` reads the worst
  `ceil(alpha * T)` values, and for an admitted column of coverage `c` those are all observed
  returns whenever the asset has that many losses, so the measure reads the observed sample's
  `alpha / c` level. At `c = 0.3` a 5% CVaR is a 16.7% CVaR.
- **A hierarchical optimiser may branch the asset alone and then overweight it.** `clusterise`
  recomputes the correlation from the returns matrix rather than from `pr.sigma`, so the fill
  reaches the dendrogram independently of the covariance the estimator answered. An invented column
  has depressed variance and attenuated correlation, which reads as idiosyncrasy, and an
  inverse-variance allocation over that branch then buys more of it. This is the one consequence
  that does not merely understate a risk; it moves capital.
- **An entropy-pooling view on the asset is calibrated on the filled column.** The view resolvers
  read `pr.X` column by column to turn a stated view into a target, so the target is computed from
  the invented cells.
- **A meta-optimiser carries the fill into the outer problem.** The outer returns matrix is built
  from the inner Sub-Portfolios' net returns over `pr.X`, so the invented cells reach a level the
  caller never handed a returns matrix to.

No consumer reduces, recomputes or scans for this. The rule of this ADR holds — a check is paid
where the library makes the gapped value and the caller cannot reach it first, and the fill is that
one place. The invented cells are recoverable exactly wherever the caller's own returns are in hand,
as the non-finite entries of that matrix intersected with the Investable Mask, because the fill
copies and never mutates what the caller passed; the clustering, optimiser and meta-optimiser doors
all hold it. Nothing is stored on the carrier, which would be a field with no reader.

### A drawn plot keeps the frame, and a computed plot reduces

A heatmap or a bar chart of a Prior Result, `plot_mu`, `plot_sigma`, `plot_correlation`,
`plot_prior`, the factor plots, `plot_coskewness` and `plot_cokurtosis` under `heatmap = true`,
draws the full universe and the backend leaves a blank for a non-investable asset. A plot that
computes on the matrix, `plot_eigenspectrum`, `plot_cokurtosis` under its default, `plot_network`
and `plot_centrality`, reduces to the Investable Mask first. `plot_cokurtosis` takes both sides
because its two arities are two figures: one draws the matrix and the other takes its eigenvalues.
A series plot reads the funnel, and its prediction methods are finite by the fold's filter. A
per-asset plot keeps the gap, which draws as a break, and an aggregate of several assets excludes
what it cannot value, because one `NaN` poisons a whole sum. A weight plot draws the zero.

The ranking and the colour limits of a drawn plot read the **finite** entries alone. `NaN` sorts
first under `rev = true`, so a blank bar would take the top slot from a live asset; `inv(dot(v, v))`
and `maximum(abs, A)` are both `NaN`, and `ceil(Int, NaN)` raises. A frame that keeps the gap must
therefore reduce over the finite entries wherever it ranks or scales.

### The tripwire is a census test by reflection

One test walks the concrete subtypes of `AbstractBaseRiskMeasure`. For every measure with a
value-level method, on a Prior Result with one non-investable asset: a zero weight there gives the
same answer as the measure on the result fitted with that column removed by hand, the oracle of
`test/test_50_investable_reduction.jl`; a held weight there gives one warning and the zeroed
answer, and an `ArgumentError` under `strict`; a per-asset answer has the full length with exactly
`0` at the dead asset. A second testset drives one fold with a delisting inside its test window and
asserts a finite series and one warning.

Two further testsets pin the fill. The first pins what a scenario measure reads: for an admitted
column of coverage `c` whose asset carries at least `ceil(alpha * T)` losses, the measure at level
`alpha` over the filled matrix equals the measure at level `alpha / c` over that column's observed
rows, exactly. The second pins the notice: no policy and `fill_limit = nothing` names every fill; a
policy and `fill_limit = nothing` names none, on every fold of a walk-forward with listings; an
explicit `fill_limit` above `1 - min_coverage` refuses at the fit; and `strict = true` refuses any
fill under either.

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
| A second, scenario-only Investable Mask | An asset moment-investable and scenario-non-investable widens ADR 0115's contract and makes every scenario consumer reduce by a second mask. One coverage floor already expresses the same admission. |
| Each scenario measure carries a per-asset valid-row denominator | The refusal this ADR already makes for a finiteness check in every kernel, now at 48 of 54 measures. |
| The fill writes the asset's own `mu` rather than zero | The column's mean would agree with the reported `mu`, at the cost of two conventions for one gap: the fit would invent a return on an observation where the fold invents cash. |
| The carrier records which cells the fill invented | A field with no reader. Every consumer was left unchanged, and the cells are recoverable exactly as the non-finite entries of the caller's own matrix intersected with the Investable Mask, which every door that could act on them holds. |
| `fill_limit` is deleted, leaving `min_coverage` as the only share | The exponentially weighted family — the family this fill was written for — carries no policy and gates on a count, so it would lose its only share. |
| The derived limit is capped below one, so a mostly-invented column always names | A second number, and a hidden constant rather than a field the caller can read off the estimator. |
| `min_coverage` takes a non-zero default | Any choice is arbitrary, because the honest floor depends on the window length and the measure, and it makes a bare `CoveragePolicy()` mean something the caller did not type. |

## Consequences

- Three build tickets on map
  [#667](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/667): the fold's filter, the
  value-level doors and the census test in one; the plots in a second, blocked by the first; the
  fill in a third, blocked by the two moment build tickets. All three block the closing
  verification.
- The map's closing walk-forward can score: a fold whose test window holds a delisting yields a
  finite series and one warning, and the search scores a number, not a `NaN`. What a search does
  with a `NaN` score from a failed fold is the cross-validation decision of the map.
- The invested fraction shrinks on a gapped observation, and the warning is what says so. A caller
  who wants the honest two-asset portfolio zeros the weights over the observations the asset is
  inactive, or passes a weight history.
- A scenario-based measure understates the risk of a young asset over its filled rows under a
  mask-aware estimator. The docstrings of the family state it, and `EmpiricalPrior`'s `fill_limit`
  is where a caller loosens the notice. It is tight by default: every fill is named until the caller
  says how much of the trade to accept.
- A scenario-based measure over an admitted column of coverage `c` reads the observed sample's
  `alpha / c` level, so at `c = 0.3` a 5% CVaR is a 16.7% CVaR. The docstrings of the family state
  the identity rather than the adjective.
- Turning on a `CoveragePolicy` does not turn on a warning. The floor the caller states is the share
  the fill accepts in silence, so an available-case walk-forward names nothing while it does what it
  was configured to do, and names a column the moment a caller's own coverage algorithm admits one
  thinner than the floor it was given.
- A caller who wants to admit broadly and be told anyway states `fill_limit` explicitly, tighter
  than `1 - min_coverage`. A looser value refuses, because it could never fire.
- The dendrogram, the entropy-pooling view resolvers and the meta-optimisers read the invented cells
  and are unchanged. The notice is what says so, and a repair to any of them is a later decision
  with the census in hand.
- `CONTEXT.md` gains the **Held Gap** entry, and the **Precomputed-returns contract** entry states
  the finiteness rule.
