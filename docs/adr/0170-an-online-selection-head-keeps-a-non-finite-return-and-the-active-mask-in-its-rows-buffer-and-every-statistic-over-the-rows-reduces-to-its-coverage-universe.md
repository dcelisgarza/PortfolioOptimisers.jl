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
- `GLOSSARY.md`'s *Rule State* and *Online Update* entries say the buffer keeps the `NaN` and the
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

## Amendment (2026-11-13): the set enters the leader as a Constraint Block, and the entry intersects the masks

[ADR 0237](0237-a-constraint-block-holds-the-feasible-set-of-a-programme-and-the-entry-of-a-head-takes-a-list-of-blocks.md)
changed two parts of the Decision. The rest of the Decision holds.

**The view of the set.** *The Allocation Set Constraint views with the optimiser* is superseded.
The leader re-solve passes the set as a second block of the entry of the leader, and
`AllocationSetConstraint` is gone (#2047, ADR 0164, amendment of 2026-11-13). The entry views the
set at the Investable Mask of the step before it resolves the block of the set
(`programme_investable_reduction`). So the reduction of the held estimator carries the set by
construction, and no adapter needs a view.

**A set prior narrower than the prior of the leader.** The Decision refused such a set prior by
name, through `assert_set_prior_priced`. Now the entry intersects the Investable Masks of every
block (#2043). So an asset that the prior of the set cannot price takes a zero weight with a
warning that names it, and the `strict` flag of the head refuses it with an `IsNonFiniteError`.
The projection programme reads the same rule, because the leader and the projection go through
one shared entry, `resolve_block_entry` (#2046). `assert_set_prior_priced` now runs only in a direct
call of `set_allocation_set_constraints!`, which resolves the step itself and cannot drop an
asset.

## Amendment (2026-11-25): a reader that cannot price a leg removes it from the step

Issue #2339. The Decision gave a `RiskLoss` a zero gradient at a leg that its window does not
cover, and it gave a forecast-reading rule a forecast of one. Both were wrong for the same reason:
the projection moves the book on the difference between the legs. A zero gradient is the best
gradient when every priced leg has a positive one, and a forecast of one is the best forecast when
every other one is below one. So a step could move the whole book to a leg with no data. Under a
time-varying panel the Recursion Result then had no mass on an active asset, and it was an
`OptimisationFailure` with `NaN` weights. With no panel it was a success with the whole book on
that leg. On 300 rows of eight assets, one of them inactive at every row, the default
`MovingAverageReversion()` failed at four rows.

Two readers of a statistic already removed such a leg and wrote a zero: a `FollowTheLeader` and a
`ProgrammeAllocationSet`, as every batch head does. Every reader does so now.

- **A `NaN` is the mark.** `loss_gradient` and `forecast_relative` answer `NaN` at an asset that
  the reader cannot price, as a prior does in `mu`, and `investable_mask` of the vector reads it.
  A custom objective or predictor uses the same mark. A vector with no finite entry states no
  universe: a forecast is then flat, and a gradient is refused with the raw step.
- **The step runs on the mask.** It starts from the allocation on the priced legs, renormalised
  (`investable_book`). The gradient reads that book too, where the Decision read the slice with no
  renormalisation. The projection takes the mask in a last slot, reduces the geometry and the set,
  and writes a zero outside it. A `ProgrammeAllocationSet` intersects the mask with the one of its
  prior. So on a panel with an asset that never lists, the book is the book of the same rule over
  the other assets, to rounding.
- **The iterate of a `MirrorDescent` keeps its weight at such a leg** (`parked_iterate`), and the
  step moves the rest of its mass alone. The book holds a zero, and the leg comes back at the
  weight of the iterate. A multiplicative step from a zero would hold the zero for ever, which is
  the fact that ADR 0157 and ADR 0162 rest on. An `AdaptiveSubgradient` keeps no iterate. Its step
  is additive, so it can buy the leg from the zero.
- **Bounds that do not reach the budget on the mask hold the step.**

*The programme set is the one place the recursion takes a zero it did not step to* is superseded:
every reader of a statistic is such a place. *The step reads a gap as cash* holds. A kernel over
price relatives, log wealth among them, prices a gap at a flat return, so it forces no zero, and a
relisted asset comes back at the weight of the recursion. Such a kernel can still move the whole
book to an inactive leg when its step is large. #2346 holds that question.

Rejected alternatives:

1. **Hold the mass of the unpriced legs, and project the priced legs onto the simplex of the
   rest.** The book then holds a leg outside the loss. Option 4 above rejected the same form for a
   programme set.
2. **Reduce the head to the active mask once per row.** Every Rule State then needs an inverse of
   its view, and the state of a folding forecaster has none. It also removes the cash reading of
   every kernel, which ADR 0162 rules.

## Amendment (2026-11-25): an inactive asset leaves the step of a kernel too

Issue #2346. *The step reads a gap as cash* held for a kernel over the price relatives, at an inactive
asset as at a Held Gap. A step could then buy cash: on the fixture of the amendment above, at row 242
every priced asset fell, so a `GradientProjection(; eta = 1e5)` on log wealth put the whole book on the
asset that the panel marks inactive at every row, and the Recursion Result was an
`OptimisationFailure` with `NaN` weights. The same reading made the fund of most rules differ from the
fund over the listed names, and it read a stale finite return of an inactive asset as a real one.

- **The price relative holds `NaN` at an inactive asset** (`row_relatives`), the mark that the amendment
  above gave to a gradient and to a forecast, so `investable_mask` of `x` is the mask of the active
  assets. A row with no active asset states no universe and holds no mark. A gap at an active asset
  stays a Held Gap and reads as one.
- **Every rule steps on the mask.** It reads its allocation on the active assets, renormalised, and the
  projection takes the mask in its last slot. A rule that reads a forecast or a Risk Loss intersects the
  two masks. A `FollowTheLeader` marks a departed asset inactive at every row of its sample, because a
  sample of past rows can hold every return of it. The two mixtures read each expert on the
  mask.
- **A state keeps its entry at an inactive asset.** The gradient sums of the Newton step, the wealth of
  `TopK` and of the weak aggregating algorithm, and the belief of the confidence-weighted reversion do
  not move there. A rule that multiplies its own allocation keeps the weight of the asset on its iterate
  (`parked_iterate`), as the `MirrorDescent` did. So `BuyAndHold`, `AggregatingAlgorithm` and
  `SwitchingWeighting` hold an `AllocationIterateState`, where they held no state, and the state of
  `ExpectationMaximisation` holds the iterate beside the start.
- **The Price-Adjusted Allocation reads the mark as one.** It is the book of the fund, and a leg that
  departs is sold at that value.
- **A programme set that reads no rows takes a mask.** A kernel keeps no rows, so the set reads the names
  outside the mask off its own universe.
- **A step from an allocation with no mass on an active asset starts from the uniform allocation over
  the active assets**, the reading of `renormalised_view` and of the amendment above. So a step that is
  not held gives mass to an active asset. The `OptimisationFailure` stays for a last row with no active
  asset, and for a hold on a book of departed assets, and the fallback answers both.

So on a panel with an asset that never lists, the book of every rule is the book of the same rule over
the other assets, to rounding. Rejected alternatives:

1. **Keep the cash reading, and project the raw step at the mask alone.** The whole book could not move
   to the asset, but the gross return, the mean and the variance of the step would still read it, and
   the book of a rule would depend on how many unlisted names are pinned.
2. **Keep the cash reading, and read out the uniform allocation when no active asset has mass.** The
   recursion then holds a book that the fund does not, and the next step starts from it.
3. **Keep the whole weight of the multiplicative rules in the allocation of the head, and slice it when
   the Result is read.** The fund does not hold that weight, so every turnover ceiling would measure the
   trade from a book that does not exist.
4. **Pass the mask in a slot of its own.** Every rule of a user then takes a new slot, and two sources
   of one fact can disagree.
