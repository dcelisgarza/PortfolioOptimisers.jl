---
status: proposed
---

# A folding price-level statistic resets an asset the active mask turns off, and answers `NaN` below its first folded level

## Context

[ADR 0170](0170-an-online-selection-head-keeps-a-non-finite-return-and-the-active-mask-in-its-rows-buffer-and-every-statistic-over-the-rows-reduces-to-its-coverage-universe.md)
made the rows buffer of an online selection head keep a non-finite return and the row's active
mask, and made every statistic over the rows reduce to its Coverage Universe. One family kept the
old reading. A `PriceLevelExpectedReturns` over a folding statistic —
`ExponentialMovingAverage`, `ReweightedPriceRelative`, `KernelTrendPattern` — carries a recursion
over price levels rather than a sample, and a level that goes undefined never recovers, so the
head filled each gap with a zero return before the fold and the fold ignored the `active_mask` it
was handed.

That reading is wrong under a point-in-time universe, and it is wrong in two directions.

- **A relisting starts warm.** An asset unlisted over a span entered the recursion at a flat
  level for the whole span, so it relisted carrying an exponential average of `1.0` levels, a peak
  of `1.0`, and a kernel memory of flat relatives. The forecast it then answered was a forecast of
  a price path that never traded.
- **The two arms of one estimator disagreed.** The refit siblings of the same estimator —
  `MovingAverage`, `LaggedPrice`, the windowed statistics — reduce to the Coverage Universe
  through `mean(me, X, pnl)` and answer `NaN` at an asset with a gap anywhere in the window. The
  fold answered a number there. One estimator read one gap two ways.

`ExpWeightedExpectedReturns` and a `SimpleExpectedReturns` under a `CoveragePolicy` already state
the reading a moment takes: reset the asset at an inactive row so a relisting starts cold, and
answer `NaN` below the warm-up. The price-level family is a moment of the same panel, so it owes
the same reading.

Two facts about the family shape the decision. Every folding statistic writes its own cold seed
inline at its first fold, and the seed is not the same for all three: the exponential average
seeds its carried forecast at one, the reweighted relative at the row's own relative, and the
kernel pattern divides its carried prediction by the row's relative, so its seed is that relative
too. And one of the three couples the assets: the kernel pattern's elastic-net regression takes
the assets as its observations, so a dead asset does not merely carry a wrong number of its own,
it enters every other asset's fit.

## Decision

### A row is folded on its valid assets, and an inactive asset is reset to its cold seed

An asset is **valid** at a row when its return is finite and the active mask admits it, which is
the condition `coverage_valid` already states for the moment families. `partial_fit!` reads the
row's valid assets and its active ones, and takes three readings.

- A **valid** asset is folded, and its count of folded levels rises by one.
- An **inactive** asset is reset: its carried statistic is the cold seed, its memory column is
  flattened to one, and its count is zero. It re-enters the recursion cold when it relists.
- An asset that is **active but not finite** is a holiday inside a listing. Its price relative is
  one — the level did not move, which is the reading the fold already took — and its count does
  not rise.

With no mask the finite assets are the active ones, so a gap is a delisting rather than a
holiday. Nothing else states which it is, and the reading that fabricates no level is the one that
is safe to take by default.

### The cold seed is a verb of the interface, because each statistic states a different one

`cold_statistic(alg, x)` answers the carried statistic before the first row, in the units the
statistic's own recursion reads it: the value that makes `fold_statistic` answer the seed the
statistic states. The exponential average answers one, the reweighted relative and the kernel
pattern answer `x`. Each `fold_statistic` now reads its seed through that verb rather than writing
it inline, so the reset and the first fold cannot drift apart.

### The caller reduces, so a coupled statistic needs no mask of its own

`fold_active` folds the active assets alone: it slices the carried statistic, the memory and the
relative to them, calls the recursion once, and writes the answer back. The active assets are the
row's Coverage Universe, so the kernel pattern's regression pools the live assets and no dead one
reaches an `lu`. A row on which no asset is active folds nothing.

The reduction lives in the caller rather than in each statistic, so `fold_statistic` keeps its
three-argument and four-argument shapes and a statistic added later is mask-aware for free.

### The read-out answers `NaN` below the first folded level

`PriceLevelForecastState` carries a per-asset count of the levels folded since the asset's last
reset, and `mean(me)` answers `NaN` for an asset at zero, as `ExpWeightedExpectedReturns` answers
below its `min_obs`. Every folding statistic truncates to the levels it holds, so one folded level
is the whole warm-up and the `NaN` names an asset the mask has turned off or one that has quoted
nothing yet. In the head, `flat_where_undefined` holds that leg, as it holds any undefined
forecast.

### The batch arm of a folding statistic is that fold

`mean(me, X, pnl)` for a folding statistic overrides the reduce-and-expand root of the verb and
runs the fold over the rows under the panel's active mask, from a cold state. The batch and the
fold are then the same code over the same rows, so the batch-online identity is exact by
construction rather than by measurement, and a gapped window reads the same on both arms. A
windowed statistic keeps the root: it is fitted on the Coverage Universe of its window and
expanded with `NaN` outside it, and it refuses an `active_mask`, because it carries no recursion
to reset.

### `fold_row` goes

The fold arm of the head hands every forecaster the row verbatim with its active mask. The
price-level family reads the gap as the moment families do, so the hook that filled it is gone
and `forecast_relative` calls `partial_fit!` directly.

## Considered options

- **Freeze an inactive asset rather than reset it.** The statistic would stand still over the dead
  span and the asset would relist carrying its pre-delisting forecast. That is the reading
  `DecayCoverage` takes for a holiday, and it is the wrong one for a delisting: a relisted ticker
  may be a different company, which is the case `ResetCoverage` exists for. Refused.
- **Give `fold_statistic` an `active_mask` keyword.** Each statistic would then decide what to do
  with a dead asset. That writes the same reduction three times, and a fourth time for every
  statistic added later. The caller's reduction gives every statistic the same answer and keeps
  the interface at its two arities. Refused.
- **Keep the flat-level filling and answer from the count alone.** The count would hold the leg
  until the asset had folded a level, but the memory of a coupled statistic would still carry the
  dead asset's flat column into every other asset's fit. Refused.
- **Leave the two-argument `mean(me, X)` mask-aware as well, through a keyword.** It takes the
  keyword for a folding statistic, which is how the panel arm reaches it; a windowed statistic
  refuses one. A plain batch verb over a gapped matrix with no panel keeps the family's rule: the
  Asset Panel is the seam where a gap is explained.

## Consequences

- `PriceLevelForecastState` gains `nu`, the per-asset count. `copy` and `port_opt_view` carry it.
- `cold_statistic` joins the `public` list and the interface of
  `AbstractPriceLevelStatistic`; `fold_statistic_row` and `fold_active` are private.
- `partial_fit!` and `mean(me, X)` on the family take an `active_mask`, and
  `mean(me, X, pnl)` is a method of the family rather than the root for a folding statistic.
- A caller that folded a gap-filled row by hand no longer needs to fill it, and one that relies on
  a fabricated forecast at an unlisted asset now reads `NaN`, which every rule of the online
  selection head holds on.
- `test/test_68_forecast_arm.jl` gains the reset, the count, the holiday, the dead row, the cold
  seed of each statistic, the refusals, and the agreement of the two arms on a gapped panel;
  `test/test_67_online_portfolio_selection.jl` reads the fold arm's row verbatim.
- ADR 0170's paragraph naming this as its follow-up is rewritten: the family now takes the
  Coverage Universe reading of that ADR rather than the cash reading.
- Issue #1238.
