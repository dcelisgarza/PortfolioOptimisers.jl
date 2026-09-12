---
status: accepted
---

# The ingestion layer seams at the Listing Span, and the clock draws the Pipeline boundary

## Context

[Map #955](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/955) designs the ingestion
layer from zero, so that a point-in-time asset panel is the natural case rather than a special one:
a caller hands raw prices spanning listings, delistings and suspensions, and the library returns
returns-level data whose non-investable cells carry `NaN`, together with an `AssetPanel` carrying
the two universe masks that describe them. Rewrite cost carries no weight on that map; every option
is argued on architecture, maintainability, ergonomics and performance alone.

Two decisions were settled while charting, and are inputs here rather than results. The **active
mask** is derived by the span rule — per asset column, a leading run of gaps is an asset not yet
listed, a trailing run is a delisting, and an interior gap is a suspension on an asset that is
still listed, so the gap's *position* carries the distinction and a caller holding only prices is
never blocked. The **estimation mask** is `amsk .& isfinite.(returns)`. What both leave open, and
what this ADR settles, is *where they live*.

[Issue #957](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/957) measured the
reference implementation and found no route from a price table to a masked panel at all: its
returns verb emits no mask, and its panel takes a mask and no prices. The destination's route is
one the reference does not have, so the decomposition is not a port.

[Issue #958](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/958) verified that
deriving the masks **panel-wide** is leak-free: identical Coverage Universe and identical weights
against a per-window derivation on every fold, differing only in a test window's Held Gap
reporting.

### The one hard ordering constraint

The mask derivation straddles the conversion. The span rule reads **price** gaps, so it runs before
the conversion; the masks are `observations × assets` on the **returns** clock, so they are
expressed after it; and both must be fixed before any policy drops a row or a column, or the policy
changes the answer the mask gives. A decomposition that makes "derive the masks" and "convert to
returns" freely-orderable steps has to say how it survives this.

### What the Pipeline already fixes

Three facts, measured on `dev`, constrain any answer:

- **A seam is only expressible in a `Pipeline` if its currency is a carrier.**
  `PIPELINE_DATA_SLOTS` is `(:prices, :returns)`, and `pipe_writes` infers a step's slot from the
  abstract taxonomy — `AbstractPricesPreprocessingEstimator` writes `:prices`,
  `AbstractReturnsPreprocessingEstimator` writes `:returns`. A third currency needs a new family, a
  new abstract type and a `PIPELINE_INVALIDATES` row that nothing downstream derives from.
- **A step only ever sees a window, never the panel.** `predict(res::PipelineResult, data,
  test_idx, cols)` slices first and replays the fitted steps on the slice, and
  `assert_split_position` requires a `TrainTestSplit` to be the *first* step, because "a stateful
  step fitted before the split would have seen the held-out test rows". So no step can be given the
  whole panel.
- **Folds are cut once, on the carrier's clock.** `search_cross_validation` computes
  `cv = split(gscv.cv, data)` before the grid loop, so every grid point is scored on identical test
  windows. A step that changed the observation count would desynchronise `train_idx` and
  `test_idx`; and a hyperparameter that renumbers the observations changes the test set itself, so
  it could not be scored against one that does not.

## Decision

### The layer is three pieces, seamed at a Listing Span

The path from a raw price table to a masked returns panel is three pieces, not one:

1. `listing_span(X)` reads the raw price gaps and returns a **Listing Span** on the price clock.
2. `PricesToReturns` converts prices to returns under the return and padding conventions.
3. `universe_masks(span, R)` projects the span onto the returns clock and intersects it with
   finiteness, giving `amsk` and `emsk`.

The straddle is discharged by **naming the object that crosses the clock boundary** rather than by
hiding it inside a monolith. The Listing Span is price-clock, the masks are returns-clock, and the
projection is the only code that knows both. A conversion with `padding = true` keeps the first
observation with a `NaN` return and the two clocks align row for row; without it the returns clock
is one shorter, and the projection drops the leading row. The projection is the one place that
knows which.

### A Listing Span is a lazy `AbstractMatrix{Bool}`

`ListingSpan <: AbstractMatrix{Bool}` stores `first` and `last` per asset plus the price-clock
length, answering `size` as `(observations, assets)` and `getindex(t, i)` as
`first[i] <= t <= last[i]`. It is Base-only and unexported, exactly as `RepeatedLeading` is: it
owns `size`, `getindex` and `show`, and nothing dispatches on it.

The compression is exact, not approximate. Under the span rule an asset's active set **is** the
interval `[first priced, last priced]`, because an interior gap stays active; a column that is
gaps throughout is the empty interval `first > last`. So the derived active mask costs two integers
per asset rather than `observations × assets` booleans.

Because the type bound is `AbstractMatrix{Bool}` rather than `ListingSpan`, a caller's declaration
enters at the **same** point: a listing calendar or an arbitrary constituency that leaves and
rejoins is any other `AbstractMatrix{Bool}`. There is one override entry point and one type bound,
and the derived case keeps its compression. No span-rule algorithm family is minted — one
derivation rule exists, and a declaration needs no wrapper — and a family may be added if a second
rule ever earns one.

### The span rides on the price carrier, and no Pipeline slot is added

The price carrier gains a `span` field bound to `Option{<:AbstractMatrix{Bool}}`.
`PIPELINE_DATA_SLOTS` is untouched.

This is a seam rather than an internal step because the two halves live on different clocks and the
carrier is the only thing that survives a fold slice: `port_opt_view(pr, i, j)` slices the span in
step with `X`, for free and by construction. A third data slot would instead need its own branch in
`pipeline_data_view` to be sliced in step, plus a family, an abstract type and a
`PIPELINE_INVALIDATES` row with no meaning — nothing derives from a span the way a prior derives
from returns.

### The span is carrier data, not fitted state

The span rule runs when the carrier is built, **outside** the `Pipeline`. It cannot be a step: a
step only ever sees a window, and a window-local span reads a delisting that straddles the window
end as dead rather than held, which is precisely the divergence #958 measured.

What licenses a panel-wide derivation, where `assert_split_position` otherwise forbids anything
before the split, is that **a listing calendar is a fact about the instruments, not an estimate
from returns**. The split rule guards against a *stateful* step learning from held-out rows; a span
learns nothing. This is why #958 measured it leak-free rather than merely convenient.

A conversion that meets a carrier holding gaps and no span **states no universe**: it carries the
gaps into the returns and emits no panel, so `pnl === nothing` keeps its one meaning. It does not
derive a span window-locally, because a window-local span reads a delisting that straddles the
window end as dead rather than held — the divergence above — and a silently wrong universe is worse
than none. The gaps are still handled downstream: with no panel the Coverage Universe reads
finiteness alone.
[ADR 0133](0133-the-conversion-computes-a-return-and-ingestion-is-the-only-door.md) rewrote this
paragraph, which previously derived the span window-locally, warned, and refused under `strict`.

### Moving the observation clock is what puts work outside the Pipeline

The assembly requirements divide by one checkable rule: **moves the clock → ingestion; touches
values only → step.**

Unification of the two absent-price conventions defines what counts as a gap, so it precedes the
span rule. The factor and benchmark join adds rows under an outer join. A frequency collapse
renumbers every observation. All three therefore run in `PriceIngestion`, an estimator that is not
a `Pipeline` step and that runs once on the whole panel, emitting the span-carrying price carrier.

The rule's other side is a transform that touches values only. It leaves the clock and the gap
pattern intact, so it is expressible as an ordinary `:prices → :prices` step, and the layer mints
**no such step**. The rule says where one would go if the library wanted one; it does not say the
library wants one. See *The value-only side of the rule mints nothing* below.

The rule is forced by the fold mechanics rather than chosen. It is also why nothing is lost by
placing the clock-movers outside: a clock-changing hyperparameter changes the test set, so it is
unsearchable in principle under fold-based CV, not merely by implementation. To compare
frequencies, a caller runs two ingestions and two searches and compares two studies, which is
honest about what changed.

The clock-movers are therefore **not** also admitted as steps. Doing so would be constructible —
they are stateless, so `assert_split_position` could be relaxed to "no *stateful* step before the
split", and a collapse can carry a span through its own bucket map — but it would add a relaxed
split rule, a span re-projection obligation on every clock-moving step, and a cross-validation
refusal, for a surface nothing can search. The cost accepted in exchange is that the join rule and
the collapse are recorded in the caller's script rather than in the `PipelineResult`.

### The Asset Panel is the universe statement; Panel Fields are optional

An `AssetPanel` today refuses an empty `pf` — "an Asset Panel needs at least one Panel Field" — and
with masks present requires its fields to carry an observation axis. The layer's common case
produces two masks and **no feature data at all**: a caller holding only prices has no market
capitalisation and no sector. As it stood, the layer could not emit a panel.

The two masks become the panel's defining content and the Panel Fields become optional payload. The
empty-`pf` refusal goes, and when `pf` is empty the shared axis comes from the masks instead of the
fields. Masks and fields keep one carrier and one axis statement, which is what every consumer
already assumes: the Descriptor and Exposure families validate their own `observations × assets`
shape against `pnl.amsk` while holding a panel and often no returns carrier at all.

Moving the masks onto the returns carrier instead was rejected: masks and fields share identical
axes, so separating them puts two things with one axis in two places, and it would strand every
reader that holds only a panel.

### Requirements 8 and 10 are served by no piece

**Implied volatilities are carried, not converted.** An implied volatility is a volatility, not a
price, so the conversion clock-aligns it and leaves `ivpa` untouched; both carriers hold the pair.
No step is minted for it.

**The train/test split belongs to the evaluation protocol, not to the layer.** A holdout is a
statement about rows, not about units or gaps, and `TrainTestSplit` is the one preprocessing
estimator not pinned to a data level. The layer serves the requirement by making its output
sliceable: the span, both masks and every field follow `port_opt_view` in step with `X`.

### Free verbs and estimators

`listing_span` and `universe_masks` are **free verbs over bare arrays**, public and unit-testable
with no carrier and no conversion in sight — the library's bare-arrays-first hierarchy, and the
surface [#961](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/961) builds against.

`PriceIngestion` is an **estimator that is not a step**. `PricesToReturns` is a **stateless step**:
it carries conventions rather than learned state, so `fit_preprocessing` returns the estimator
itself.

### The value-only side of the rule mints nothing

A price-level transform that touches values only is a step by the rule above, and the layer mints
none. The conversion's `map_func` keyword, which
[ADR 0133](0133-the-conversion-computes-a-return-and-ingestion-is-the-only-door.md) removes, is
therefore removed outright rather than rehomed.

Three measurements settle it, taken on
[#1001](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1001):

- **Nothing outside a test ever used it.** `map_func`'s only call sites in the repository were two
  assertions in `test/test_06_preprocessing.jl` — a scale and a shift. No example, no user-guide
  page and no `src/` caller.
- **A step buys nothing the fold mechanics could not already give.** Such a step is stateless, so
  applying it per window and mapping the table once before the door produce identical numbers. What
  a step would buy is provenance in the `PipelineResult` and searchability over the function — and
  this ADR already accepts the opposite trade for the join and the collapse, which the caller's
  script records rather than the result.
- **The one scalar price transform with a meaning is already served.** `PricesToReturns` takes
  `ret_method = :log`, so log returns need no logarithm applied to prices.

A transform that reads a date or a second series — a currency conversion, a deflation — is not
elementwise and would not be served by an elementwise step in any case. A caller expresses either
kind by mapping the table they hold before the door, in one line.

The rule stays whole: if a concrete value-only transform appears that the library should own, this
ADR already says it is a `:prices → :prices` step. Minting the family before there is a member
would cost an estimator, its export, its Capability Catalogue entry, its API-page entry, its sweep
row and five baselines, and buy nothing.

## Consequences

The ten requirements the layer must serve each have one named owner or a stated reason for having
none. Conversion, mask emission and the panel are `PricesToReturns`; unification, the join and the
collapse are `PriceIngestion`; the active mask is `listing_span` and the estimation mask
`universe_masks`; implied volatilities are carriage plus clock alignment; the split is
`TrainTestSplit`, which the layer does not own; and the elementwise map is owned by nothing, which
*The value-only side of the rule mints nothing* states and argues.

`CONTEXT.md` mints **Listing Span** and **Span Rule**, and amends **Asset Panel** — its
point-in-time shape is now stated by the two masks, with Panel Fields as optional payload rather
than as the panel's defining content.

`PIPELINE_DATA_SLOTS`, `PIPELINE_INVALIDATES`, `pipe_reads` and `pipe_writes` are unchanged, and
`assert_split_position` keeps its present rule. The decomposition adds no construction-time gate to
`Pipeline`.

A caller who builds a price carrier by hand, outside `PriceIngestion`, gets a span of `nothing`, and
the conversion emits no panel for it whether or not it holds gaps. A clean carrier is unaffected.

The filtering and imputation policy inherits its position rather than choosing it, and the seams
above split it by what it does. A policy that **drops** a row or a column runs at the returns level,
after both masks, because the masks are the truth it is judged against. A policy that **fills** runs
at the price level, after the span and before the masks, because a price convention cannot be stated
after the conversion — and the span is read first either way, so neither can move it.
[ADR 0130](0130-a-universe-policy-is-fitted-and-the-only-fill-is-a-span-bounded-price-convention.md)
owns that rule and the policy surface itself.

What the layer emits as a whole — one object carrying its own panel, or two a caller composes — and
how a fold replays it are not settled here. The seams this ADR fixes are what make those questions
sharp, and they are ticketed off map #955 rather than assumed.
