---
status: accepted
---

# The layer emits one carrier, and alignment splits into a fixed axis and a provenance check

## Context

[ADR 0129](0129-the-ingestion-layer-seams-at-the-listing-span-and-the-clock-draws-the-pipeline-boundary.md)
fixed the ingestion layer's seams — `listing_span` → `PricesToReturns` → `universe_masks`, with the
Listing Span riding on the price carrier — and closed by naming two things it did not settle: what
the layer emits *as a whole*, and how a fold replays it. This ADR settles both.

Rewrite cost carries no weight, as it does nowhere on
[map #955](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/955). Every option below is
argued on architecture, maintainability, ergonomics and performance alone.

### What the Pipeline fixes about the emission

A `Pipeline` step has one out-slot. `apply_fitted_step(fitted, data) -> data′` takes one carrier
and returns one carrier, and `PIPELINE_DATA_SLOTS` is `(:prices, :returns)`. ADR 0129 assigns
conversion, mask emission and the panel to `PricesToReturns`, which is a step. So a layer that
emitted two objects would have to move mask emission out of that step — and it cannot go to
`PriceIngestion`, which runs on the price clock, where the masks are returns-clock by construction.
The Pipeline path and a direct path would then emit different shapes for the same work.

### What the carried universe changes about alignment

`assert_universe_aligned` (`src/23_Pipeline/03_Pipeline.jl`) is one comparison, `rd.nx ==
train.nx`, called from both `predict` methods. It exists because a fold's universe was per-window
inference: a stateless conversion that deleted an all-gap column produced a different asset set in
train and test, and the terminal weights, indexed by the training universe, would misalign against
the test returns.

[Issue #958](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/958) measured that with a
panel the check never fires on the returns level, and that on the price level a row-deleting filter
makes it refuse a fold with a panel and without one alike. Under the layer this ADR completes,
neither cause survives: `PriceIngestion` fixes the asset axis before the split,
[ADR 0130](0130-a-universe-policy-is-fitted-and-the-only-fill-is-a-span-bounded-price-convention.md)
admits no row- or column-dropping policy of the layer's own, and an `AbstractAssetSelector` replays
by name onto an axis that no longer moves, so it always finds its names.

### What the check never covered

The failure the check names is *structural* — two windows with different asset sets. The failure a
gapped panel actually produces is *semantic*: an asset present in both windows and non-investable
in one of them. `CONTEXT.md` already separates these. A Held Gap "is per observation, and it arises
where the universe changes after the fit", and `filter_held_gaps` reads the weights and the returns
directly, needing no panel. An asset live in a test window but outside the training window's
Coverage Universe is not a Held Gap at all: reduce-and-expand gives it weight zero, and a zero
weight at a missing return is silent.

## Decision

### The layer emits one carrier, and the panel rides on it

The returns carrier holds the `AssetPanel`. `PricesToReturns`'s return type is unchanged, the step
contract is satisfied without amendment, and `port_opt_view` slices the universe in step with `X`
for free and by construction — the same argument ADR 0129 used to put the Listing Span on the price
carrier rather than in a third data slot.

The free verbs below the carrier are unaffected: `listing_span` and `universe_masks` remain public
verbs over bare arrays, returning a span and a mask pair with no carrier in sight. Two surfaces
returning different shapes is the library's bare-arrays-first hierarchy, not an inconsistency — the
composed surface returns one carrier because a carrier is what a `Pipeline` moves.

What a caller pays for the pairing is small in both directions. A caller who wants only returns
holds a panel they never read, whose active mask is `O(assets)`. A consumer who wants only the
universe reaches it through the carrier, which is where the Descriptor and Exposure families
already look.

### The layer always emits a panel

Every carrier the layer produces holds one. A price table with no gaps is not a special case that
skips the panel; it is a panel whose masks are all true.

This is what makes the sentinel say one thing. `pnl === nothing` means **this carrier was not built
by the layer** — the same crisp signal ADR 0129 gives a `nothing` span, and the condition under
which the conversion warns and refuses under `strict`. Were a gapless ingestion to emit `nothing`
instead, the value would carry two meanings at once and no fold could assume a carried universe
without asking which.

### The all-true mask is a distinguished type, internal and undispatched-upon by callers

The gapless case is an all-true `AbstractMatrix{Bool}` storing no cells, so "always emit" costs
`O(1)` rather than `observations × assets` bits, and a consumer's fast path is chosen by dispatch
rather than by scanning the mask.

It is **unexported and carries no `CONTEXT.md` entry**, exactly as `ListingSpan` does: it owns
`size`, `getindex` and `show`, and it is an internal compression. The public bound stays
`AbstractMatrix{Bool}`, which is what a caller writes against and what a caller's own declaration
enters as. A method specialised on the all-true type is a library-internal optimisation, not a
contract an extension author may rely on.

The library's usual spelling for "all of them" — the `nothing` of a Coverage Universe or an
Investable Mask — is unavailable here, because `nothing` masks on an `AssetPanel` already mean the
panel is **static**, and that is what makes staticness a type parameter rather than a runtime
branch. A third meaning on the same field would cost that.

### The active mask is the projected span; the estimation mask is a snapshot

`amsk` is the Listing Span projected onto the returns clock — two integers per asset, so ADR 0129's
exact compression survives the conversion rather than being expanded and discarded at it. ADR 0131
fixes the projection as `[first + 1, last]`.

`emsk` is materialised once, at emission, from the returns the conversion produced. It is a
**snapshot of what ingestion observed**, not a view over whatever `X` later holds. A returns-level
step that touches values — a `CrossSectionalWinsoriser`, say — therefore does not move it.

The alternative, a lazy `emsk` reading `amsk[t, i] & isfinite(X[t, i])` on every access, keeps the
defining equation true at every instant and costs `O(assets)`. It was rejected on two counts. It
couples the panel to the carrier's values, so `port_opt_view` must slice an internal reference in
step with the outer slice or the two silently disagree. And it lets a value-level step move the
estimation universe underneath a fold that was already scored against it, which is a result
changing without a decision.

This does not contradict ADR 0131's "the estimation mask follows the values, unaware of the
algorithm". The snapshot is taken **after** the conversion, so a Gap Return algorithm's cells are
already in the returns the mask reads. What the snapshot refuses is tracking values that change
after the layer has spoken.

### A caller owns the active mask; the estimation mask is never theirs to state

A caller who hands the layer an `AssetPanel` of their own keeps their Panel Fields and their
`amsk`. A listing calendar, or a constituency that leaves and rejoins, is a fact about the
instruments, and ADR 0129 already gave it the same entry point and the same `AbstractMatrix{Bool}`
bound as the derived span.

`emsk` is re-derived regardless, as `amsk .& isfinite.(X)`. It is a statement about the *data*, not
about the instruments: a declared `emsk` marking a cell true where the return is not finite would
put a `NaN` into a cross-sectional fit, and `assert_panel_masks` would not catch it — it checks
`emsk ⊆ amsk`, not finiteness. Re-deriving makes the subset invariant hold by construction rather
than by refusal, and removes the one way a declaration can produce an incoherent panel.

### The alignment guarantee splits in two

**The axis half becomes structural, and a fold assumes it for free.** The asset axis is carrier data
fixed by `PriceIngestion` before the split, and `port_opt_view` slices it, so every window of every
fold carries every asset. Reduce-and-expand therefore always expands onto a fixed axis, and a
window can no longer silently lose a column.

**The semantic half was never this check's, and needs nothing new.** An asset present in both
windows and non-investable in one is the Investable Mask's and the Held Gap's business, already
owned by `filter_held_gaps` under the strictness policy.

**What `assert_universe_aligned` becomes is a provenance check.** It keeps its call sites, and its
content changes from "did the transform drop columns?" to "did this window come from the same
ingestion?": `nx` equality, plus panel-presence parity between the fitted context and the replayed
window. `check_asset_panel` binds the panel's asset axis to the carrier's at construction, so `nx`
equality compares the panel's axis transitively and no separate assertion is owed.

Its remaining reach is exactly the population that can still break the invariant: a caller who
built a carrier outside the layer, and a third-party step that changes the asset set. Its message
is restated accordingly — it no longer instructs the caller to carry gaps or to pin the universe
with a filter and an imputer, because on the layer's path neither situation can arise.

The check is not deleted. Deleting it would report a bypassed layer as a dimension mismatch inside
the risk calculation, which is the symptom the check exists to pre-empt by name.

**What changes for the fold's contract** is that the panel now *states* the universe where the fold
previously inferred it. A test window holding an asset the training window's Coverage Universe
excluded was always handled correctly — it holds zero — but the fold reached that outcome by the
asset failing to be investable rather than by reading a universe. It now reads one.

## Consequences

The layer's emission is one returns carrier holding an `AssetPanel`, always. `PricesToReturns` is
unchanged in return type and in step contract; `PIPELINE_DATA_SLOTS`, `pipe_reads` and `pipe_writes`
stay as ADR 0129 left them; and no new data currency, family or abstract type is minted.

`listing_span` and `universe_masks` remain free verbs over bare arrays. `PriceIngestion` remains an
estimator that is not a step. `PricesToReturns` remains a stateless step. The answer moves none of
them.

`AssetPanel`'s masks gain a third representation alongside a dense mask and `nothing`: an
`O(assets)` projected span for `amsk`, and an `O(1)` all-true type for both masks of a gapless
ingestion. Both are unexported internals under an `AbstractMatrix{Bool}` bound, so no consumer's
signature changes and `CONTEXT.md` mints no term for them.

`assert_universe_aligned` keeps its call sites and narrows its content to `nx` equality plus
panel-presence parity, with a message naming the two situations that can still reach it. Its
docstring's two remedies go: on the layer's path there is nothing to remedy.

`CONTEXT.md`'s **Asset Panel** entry is amended. `nothing` masks read as *static, or hand-built*,
and never as *gapless* — a panel the ingestion layer emits always carries both.

A caller's declared `AssetPanel` keeps its Panel Fields and its `amsk`; its `emsk` is re-derived.

What remains unsettled, and is not settled here: how the `AssetPanel`'s Panel Fields are
subselected now that the masks are its defining content, and what a square tensor Panel Field's
label axis does under a layer that carries gaps rather than dropping the rows and columns holding
them. Map #955 keeps both in its *Not yet specified* section, where they hang on the build rather
than on a decision.
