---
status: proposed
---

# An online selection head keeps a non-finite return and the active mask in its rows buffer, and every statistic over the rows reduces to its Coverage Universe

## Context

The rest of the library treats a non-finite return by reducing the fit to the Coverage Universe —
the assets whose return is finite and whose active mask is `true` at every row of the window —
and expanding the result back with `NaN` outside it
([ADR 0117](0117-a-prior-reduces-to-the-coverage-universe-and-a-plain-moment-estimator-refuses-a-non-finite-sample.md),
[ADR 0120](0120-a-fold-scores-on-the-investable-mask-a-prior-free-head-and-pre-selection-reduce-to-the-coverage-universe-and-a-failed-candidate-loses-the-search.md)).
A mask-aware moment estimator reads the panel's masks itself. The online optimisation head
carries the active mask of a time-varying panel into the prior's buffer beside the rows
([ADR 0137](0137-an-optimiser-forwards-to-its-prior-alone-and-a-read-out-reconstitutes-the-carrier-and-runs-the-batch-path.md)).

The online selection head did neither. Under
[ADR 0157](0157-an-online-selection-state-is-a-rule-state-beside-the-rows-held-once-and-a-block-of-rows-is-that-many-single-row-updates.md)
and
[ADR 0162](0162-an-online-selection-head-buffers-returns-and-starts-from-a-given-allocation-or-a-uniform-one-over-the-pinned-universe.md)
the head filled every non-finite return with `0` once, before the rows buffer and before the
rule, and the buffer held the filled rows and no mask. That is correct for the step itself — the
leg sat in cash, `x = 1` — and wrong for every reader of a statistic over the buffer, which read
a zero return where there was no return.
[#1237](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1237) measured two
failures on the fixture of `test/test_67f_family_verification.jl` (asset `D` unlisted over rows
`1:10`), and its census named every reader.

- **A second-moment fit over the unlisted span throws.** `FollowTheLeader` over
  `MeanRisk` and `LastRows(15)`, `MirrorDescent` over `RiskLoss`, and a
  `ProgrammeAllocationSet` with a fitted risk ceiling or a tracking error under **any** rule, all
  on `rows(rdg, 1:15)`: `ArgumentError: matrix contains Infs or NaNs`. The window carries a
  constant zero column, the default covariance's positive-definite repair divides it by its zero
  standard deviation, and `NearestCorrelationMatrix`'s `eigen` throws on the `NaN`. Only the
  `MirrorDescent` docstring named the failure and its escape.
- **A windowed statistic over a relisted asset is diluted, silently.**
  `MovingAverageReversion(; window = 15)` at row 15 read five real returns of `D` and ten filled
  zeros, so its forecast was the real signal scaled by one third. Nothing failed and nothing
  warned. Every forecaster on `me`, every leader over a window and every risk loss over a window
  read the same dilution for as long as the window straddled the listing.

The census found one more fact. `port_opt_view(::CustomJuMPConstraint, …)` answers `nothing`,
so a batch head's reduction to its Investable Mask drops a custom constraint. The
`AllocationSetConstraint` a leader appends to its held optimiser had no view, and the reduction
that keeping the `NaN` triggers would have dropped the head's Allocation Set from the leader's
programme in silence.

ADR 0157, ADR 0158, ADR 0162 and their consequence lines are on `dev` and not on `main`, so they
are drafts and are rewritten in place.

## Decision

### The buffer keeps the `NaN`, and the row's active mask beside it

The rows buffer `X` on `OnlinePortfolioSelectionState` holds the Returns Result's rows
**verbatim**, `NaN` where there was no return, and under a time-varying panel the row's active
mask on the buffer's own `A`, through the `active_mask` keyword `SampleBufferState`'s fold
already takes. `rows_needed` caps both. The Held Gap diagnostic stays where it was and writes
nothing: `report_row_gaps` reads `isfinite(r[i])`, `amsk[i]` and `w[i]`, warns or refuses under
`strict`, and the row reaches the buffer as it came.

### The step reads a gap as cash, and every statistic reads it as a gap

The gap is read in two ways, and each reader gets the one the rest of the library gives it.

The **Online Update** reads `x = price_relative.(r)`: `1 + r`, and one at a gap. That is the
number the library's Scenario Fill writes for the same reason — the asset contributed nothing on
that observation, so its slice sat in cash — and the recursion holding a leg over an unlisted
span at that flat return is the re-entry ADR 0157 and ADR 0162 ruled: no forced zero from the
head, re-entry at the recursion's own weight. A **kernel over price relatives** reads the same
number: the pattern-matching Sample Selectors compare windows in which an unlisted leg sat in
cash, and `AntiCorrelation`'s two windows take the log of `price_relative`. A window that
straddles a listing is therefore compared as such rather than dropped, which is the design
point that #1237 asked the build to state.

Every **statistic over the rows** reads the rows as the batch verb reads a carrier. The head
reads the buffer out at every row as a `ReturnsResult` — the rows verbatim under the pinned
names, the buffer's masks as a time-varying `AssetPanel` with `emsk = amsk`, which is what
`step_active_mask` admits — and hands that carrier to `online_update!` as its `rows` argument
and to the `ProjectionStep`. So:

- a forecaster's refit is `mean(me, rd.X, rd.pnl)`, and its fold folds the carrier's last row —
  the current row verbatim — under the row's active mask;
- a `RiskLoss` fits `prior(pe, rd)`, reads its gradient on the result's Investable Mask at the
  iterate sliced to it, and writes zero at every other leg;
- a `FollowTheLeader` re-solves `optimise(opt, port_opt_view(rd, idx, :))`, so a prior-free
  head reduces to the Coverage Universe of the selected rows and a prior-fitting one to the
  Investable Mask of its prior, and both answer a zero outside it;
- a `ProgrammeAllocationSet` fits `prior(pe, rd)` and the projection programme runs on the
  result's Investable Mask exactly as a batch head runs: the set, the raw step, the
  Price-Adjusted Allocation and the carrier are viewed at the mask, the reduced programme is
  solved, and the answer is expanded with a zero at every asset the prior could not price.

A plain estimator answers `NaN`, and so holds, at an asset with a gap anywhere in its window; a
mask-aware one answers it from the rows it has. A plain folding moment estimator's running
statistic is `NaN` from the first gap on, the Coverage Universe of the prefix. Either is honest.
The filled mean was neither.

One folding family took the cash reading when this ADR was written: a
`PriceLevelExpectedReturns` over a folding statistic. Its recursion runs over price levels — an
exponential average, a peak, a kernel trend with a memory of levels and an elastic-net polish —
so a level that goes undefined never recovers and a regression over a memory with a `NaN` throws.
The head therefore handed it the row with each gap read as a zero return, and a relisted asset
re-entered that recursion warm. That family now reads the mask itself:
[ADR 0172](0172-a-folding-price-level-statistic-resets-an-asset-the-active-mask-turns-off-and-answers-nan-below-its-first-folded-level.md)
resets an asset the mask turns off, folds the active assets of the row alone, and answers `NaN`
until the asset has folded a level, so the fold arm hands every forecaster the row verbatim and
the hook that filled it is gone.

### A folding forecaster reads the current row, so the head holds one row for it

`rows_needed` of an expected-returns estimator with an exact fold, and of a folding price-level
statistic, is `1` and no longer `0`. The fold needs the row verbatim — its gaps and its active
mask — and the step's `x` is finite by construction, so the head holds the current row in the
buffer and the fold reads it from the carrier. Outside the head, with no carrier, the fold arm
folds `x .- 1` as it did. ADR 0158's *the head holds no rows for it* is rewritten.

### The programme set is the one place the recursion takes a zero it did not step to

A programme set that fits on the rows writes a zero at a leg its prior cannot price, as a batch
head does, and re-admits the leg the step its prior prices it. Whether the raw step then gives
that leg mass is the rule's geometry — a multiplicative step re-enters from zero only through the
constraints — which is the same fact ADR 0162 states for a zero in the Start Allocation. ADR
0157's *the state's `w` never carries a forced zero* is rewritten to *never carries one from the
head*. The Price-Adjusted Allocation is sliced and not renormalised, as the batch reduction
slices a turnover reference: a delisted leg leaves the reference, and the budget of one over the
investable legs is what the programme re-allocates, a forced liquidation a turnover cap that
cannot absorb it holds.

A leader's programme trades its own prior's Investable Mask, and the set's prior is fitted on
the same rows: under a plain `pe` on both the masks agree. A set whose `pe` is plain beside a
leader whose prior is mask-aware can leave a young asset unpriced that the leader trades, and
its ceiling would carry `NaN` there, so `assert_set_prior_priced` refuses it by name rather than
at the solver.

### The Allocation Set Constraint views with the optimiser

`AllocationSetConstraint` gains a `port_opt_view`: the set against the carrier's unreduced
returns matrix, the Price-Adjusted Allocation sliced, the carrier viewed. The held estimator's
reduction to its Investable Mask therefore carries the head's set into the reduced programme
instead of dropping it.

## Considered options

1. **Keep the fill, document the failure on every reader.** Rejected: the dilution is silent
   and no docstring makes a wrong number right; the throw has an escape only through a
   covariance that skips its repair, which treats a symptom.
2. **Hand the rules a bare matrix with the `NaN` kept, and the mask on the `ProjectionStep`.**
   Rejected: every estimator door needs the mask beside the rows, the selected rows of a leader
   need the mask sliced by the same index, and the carrier that pairs the two already exists.
   A `ReturnsResult` costs one keyword constructor per row and no copy.
3. **A kernel over price relatives drops a window with a gap, or compares over the assets
   finite in both windows.** Rejected: a per-candidate universe is a new design the papers do
   not state, and the cash reading is the one the step already takes at the same cell.
4. **A programme set parks a leg its prior cannot price at the recursion's weight and
   re-allocates the rest.** Recorded as the runner-up. Rejected: a held leg outside the risk
   cone invents a semantic no batch head has, and the bounds and the cardinality of the set
   would have to say whether a parked leg counts.
5. **A folding forecaster keeps `rows_needed = 0` and reads the raw row off the
   `ProjectionStep`.** Rejected: an argument the update already takes should carry it, and one
   row costs nothing.
6. **Announce a non-investable departure at every step, as the batch reduction does.**
   Rejected: the read-out's Investable Mask states it at every step, and a warning per row is
   noise.

## Consequences

- ADR 0157 is rewritten in place at its state table row for `X`, its fill section and the
  consequence lines that name the fill; ADR 0162 at its buffer section; ADR 0158 at its sentence
  on the rows a folding forecaster needs.
- `CONTEXT.md`'s *Rule State* and *Online Update* entries say the buffer keeps the `NaN` and the
  mask, the step reads a gap as cash, and a statistic over the rows reduces to its Coverage
  Universe.
- `test/test_67f_family_verification.jl`'s time-varying block loses the `@test_throws` on the
  risk loss and gains a leader with a second-moment head and a programme set with a fitted
  ceiling; `test/test_67_online_portfolio_selection.jl` pins the buffer's `NaN` and mask, the
  cash reading of the step, and the undiluted moving average.
- The `MirrorDescent` docstring's constant-column paragraph and its escape go; the
  `FollowTheLeader` docstring states the point-in-time rule.
- `LowDimensionEnsemblePrior`'s `prior` ignored its panel and never reduced, against ADR 0117,
  which the roster's `LowDimensionEnsemblePortfolio` found the moment its leader met a `NaN`;
  it now reduces to the Coverage Universe and expands both moments, as every prior does.
- A mask-aware fold for the price-level statistics — a reset at an inactive row so a relisting
  starts cold, as `ExpWeightedExpectedReturns` does — was the missing piece for a caller who
  wants a folding price-level forecast to re-admit a late lister cold. It was a follow-up on the
  moment layer rather than on this head, and
  [ADR 0172](0172-a-folding-price-level-statistic-resets-an-asset-the-active-mask-turns-off-and-answers-nan-below-its-first-folded-level.md)
  settles it: [#1238](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1238).
- Issue #1237.
