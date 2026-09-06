---
status: accepted
---

# A fold scores on the Investable Mask, a prior-free head and pre-selection reduce to the Coverage Universe, and a failed candidate loses the search

## Context

[ADR 0115](0115-every-optimisation-estimator-reduces-once-at-its-entry-and-its-result-carries-the-investable-mask.md)
reduces every optimisation estimator that fits a prior to the Investable Mask at its entry, and its
result carries the mask.
[ADR 0117](0117-a-prior-reduces-to-the-coverage-universe-and-a-plain-moment-estimator-refuses-a-non-finite-sample.md)
gives the prior the Coverage Universe, the assets whose return is finite and whose active mask is
`true` at every row of the window, and the one verb that reduces to it.
[ADR 0118](0118-a-fold-zeroes-a-held-gap-once-and-a-value-level-verb-reduces-to-the-investable-mask.md)
zeroes a Held Gap once in a fold's `predict`. Five sites of the cross-validation and pre-selection
layers still met a changing universe with no rule, and ticket
[#674](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/674) on map
[#667](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/667) decided them.

- **The fold's test window.** The filter of ADR 0118 scanned the full window, so the column of an
  asset the fit had already excluded was still read, and the fold and the value-level door of that
  ADR reduced by two different rules.
- **The two prior-free heads.** `EqualWeighted` and `RandomWeighted` read the width of the returns
  matrix and weight every column. ADR 0115 left them mask-free, because a mask from the returns
  matrix needed a threshold the moment decision had not settled. ADR 0117 settled the rule with no
  threshold.
- **Pre-selection.** The census of
  [#671](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/671) measured three
  behaviours on a dead asset. `ScoreSelector` refuses through `asset_scores`.
  `CompleteAssetSelector` drops the column on finiteness alone and reads no active mask. The
  redundancy selector with `PairwiseCorrelation` keeps the dead asset in silence, because its
  correlations are all `NaN` and a `NaN` is never a pruning candidate, so it survives every prune
  while it looks maximally uncorrelated.
- **The search.** A failed solve writes a vector of `NaN` weights, so the fold's score is `NaN`.
  `HighestMeanScore` is `argmax` over the column means, and `argmax` orders through `isless`,
  under which a `NaN` is greater than every real. The census measured it: column means
  `[0.80, NaN, 0.50]` return the second column. A parameter set that failed a fold wins. The
  population sorts fail the same way: `sort_by_measure` with `rev = true` places a `NaN` first,
  and `quantile_by_measure` throws a bare `ArgumentError` from `Statistics`.
- **The random asset subsets.** `MultipleRandomised` draws each path's subset over the full asset
  count before it draws the path's window, with no notion of an asset being alive. A subset with
  dead assets reduces at the head, so `subset_size` becomes a ceiling, and a subset with no live
  asset throws `IsEmptyError` from `investable_mask` and stops the run.

The reference implementation zero-fills the whole test window at one line, at every weight, so a
caller cannot tell a dead asset at weight zero from a delisting at a live weight. Its naive family
reads the width of the returns matrix and derives no mask, so an equal-weighted benchmark over a
panel with a `NaN` column cannot run. Its default completeness selector reads the two endpoints of
each column and keeps an interior gap, four of its six selectors refuse a gap, and the order in a
pipeline is load-bearing and stated nowhere. Its batch validation raises on the fold after a failed
portfolio, and its online validation averages with a `nan`-aware mean, so one candidate can lose or
the run can die in the same package. Its multiple randomised scheme draws column indices uniformly
over the full width.

## Decision

### A fold reduces its test window to the Investable Mask, then zeroes a Held Gap

In `predict`, before anything reads the test window, the fold views the window and the weights at
`res.imsk`, so the column of a non-investable asset is never read, and the fees are viewed with
them. The Held Gap filter of ADR 0118 then runs over the investable columns alone, and the held
weights after the last observation expand back to the full length, because the next fold's
turnover reads them. ADR 0118 is rewritten to this order, so both of its doors reduce to the
Investable Mask. A result whose `imsk` is `nothing` views nothing. This is the rule of ADR 0115
carried to the window the weights are scored on.

### A prior-free head reduces to the Coverage Universe and carries the mask

`EqualWeighted` and `RandomWeighted` derive their mask from the Coverage Universe of their window,
through the verb of ADR 0117, at their entry. Each weights the reduced universe, views its weight
bounds and its sets at the mask, and carries the mask as `imsk`, so its keyword constructor expands
the weights as every other result's does. A stale finite price during an inactive spell weights
nothing. An all-dead window throws `IsEmptyError`. The last section of ADR 0115 is rewritten to
this rule, and the invariant becomes total: every optimisation result of the library carries
`imsk`, and a reader has one idiom.

### Pre-selection reduces to the Coverage Universe before every selector

The one funnel of the family, `fit_preprocessing` over an asset selector, reduces the carrier to
the Coverage Universe, runs `select_assets` on the reduced carrier, and records the surviving
names. Every selector ranks among live assets alone, a score or a redundancy is computed over live
columns only, and a new selector cannot forget the rule. `CompleteAssetSelector` becomes the
identity on the Coverage Universe, so it reads the panel's active mask through the funnel, and it
stays as the explicit step that drops dead assets and nothing else. No selector is a default
anywhere, because every head reduces itself. The reference's endpoint mode is not added: an
interior gap puts an asset outside the Coverage Universe at every head already, so a selector that
keeps it gains nothing.

### A candidate with a non-finite fold score loses the search

The search hands its scorer the columns of the score matrix whose every entry is finite, and maps
the index the scorer returns back to the grid through the list of those columns. The scorer never
sees a failed candidate, so it can compute anything on the matrix it receives, a mean, a spread, a
rank, and a failed candidate can never win. The raw matrix stays on the result, so its columns line
up with the grid and a reader sees which fold failed. When no column is finite the search throws
`IsNonFiniteError`. The population sorts place a member whose measure is non-finite last whatever
`rev` is, and a quantile is taken over the finite members.

### A random asset subset is drawn from the Coverage Universe of its window

`MultipleRandomised` draws each path's window first, then draws `subset_size` assets from the
Coverage Universe of that window, with the one random stream the seed governs. A subset is always
`subset_size` live assets, and a dead asset is never drawn. A window with fewer live assets than
`subset_size` throws `IsEmptyError`. The two draws swap order in one stream, so a seeded split gives
different indices from the released one.

### Two consequences that are not decisions

A walk-forward over a changing universe is one series. The strategy traded a different universe in
each fold because the universe changed, and that is its history, not an artefact. The per-fold
mask on each fold's result is the record, and nothing is normalised across folds. The reference's
calibration ratio belongs to its covariance forecast evaluation, which the library does not have.

Under the whole-window rule of ADR 0117 and the rolling window that `IndexWalkForward` defaults
to, a young asset joins the universe at the first fold whose whole training window it covers.
Under an expanding window it never joins, unless a mask-aware estimator carries it.

## Considered options

| Question | Refused | Why |
| --- | --- | --- |
| The test window | Full universe, and the fold zero-fills every gap with no mask view, the reference's shape and the first text of ADR 0118. | The filter scans a dead column the fit already excluded, and the fold and the value-level door reduce by two different rules. |
| The test window | Full universe, and the fold refuses any gap. | A dead asset always has a gap in the test window, so every point-in-time backtest refuses and the map's closing test cannot pass. |
| The prior-free heads | Nothing reduces; the heads refuse a non-finite column, and the caller precedes them with `CompleteAssetSelector` in a Pipeline. | Two results with no mask, so a reader has two idioms, and a bare equal-weighted benchmark over a point-in-time panel refuses. |
| The prior-free heads | The Fold Loop reduces every head's data to the Coverage Universe of the training window. | The Fold Loop gains a universe policy that ADR 0115 gave to the optimiser, a prior-fitting head reduces twice, and a bare call outside a fold is not covered. |
| Pre-selection | Each value-reading selector refuses a non-finite column, the precedent of `asset_scores`. | Three sites to guard, the CVaR hole of `asset_scores` stays, and the order in a pipeline becomes load-bearing, which is the reference's hazard. |
| Pre-selection | The reference's endpoint mode beside the strict one, and a default selector in the fold. | An interior gap is outside the Coverage Universe at every head already, and every head reduces itself. |
| The search | A non-finite column is handed to the scorer as `-Inf`. | Safe for the orientation the search fixes, and for a scorer that reads order alone, but a scorer that reads a spread computes `NaN` from a column that holds `-Inf`, and the `NaN` wins again. |
| The search | The search refuses any non-finite fold score. | A wide grid with one infeasible corner never finishes, and the caller must prune the grid by hand. |
| The subsets | Draw over the full universe, as released. | `subset_size` is a ceiling, and a dead-only subset refuses the whole run. |
| The subsets | Draw over the assets alive over the whole sample. | A survivorship filter on the sample, which is the bias the map removes. |

## Consequences

- Two build tickets on map [#667](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/667):
  [#859](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/859) builds the two
  prior-free heads, blocked by the Coverage Universe verb and by the mask on the naive result;
  [#860](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/860) builds the funnel, the
  draw and the search, blocked by the verb. Both block the closing verification. The mask view at
  the fold joins the fold ticket of ADR 0118,
  [#856](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/856).
- ADR 0115's last section and ADR 0118's first section are rewritten in place, because neither
  reached `main`.
- `CONTEXT.md` gains the three doors on the **Coverage Universe** entry, and the **Investable
  Mask** and **Held Gap** entries state the fold's view.
- A seeded `MultipleRandomised` split gives different indices from the released one.
- The two prior-free heads read the panel's active mask, so a stale finite price during an
  inactive spell no longer weights an asset.
