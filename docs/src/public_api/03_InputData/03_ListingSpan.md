```@meta
Description = "The Listing Span, public API of PortfolioOptimisers.jl: listing_span, universe_masks."
```

# The Listing Span

## The Listing Span

A **Listing Span** is the interval of the price clock over which an asset is listed, one per asset
column. The **Span Rule** derives it from the position of a gap: a *leading* run of gaps is an asset
not yet listed, a *trailing* run is a delisting, and an *interior* gap is a suspension or a holiday
on an asset that is still listed and still held. So a caller holding only prices can state a
universe, because the position already carries the distinction a listing calendar would supply.

[`listing_span`](@ref) derives the span, and [`universe_masks`](@ref) projects it onto the returns
clock and intersects it with finiteness, giving the two universe masks an [`AssetPanel`](@ref)
carries. A return consumes the *earlier* price of its pair, so the projection is `[first + 1, last]`
under padding and `[first, last - 1]` without it — the active mask bounds exactly the run of a
column's finite returns, and an inception emits no **Held Gap**. The two masks then differ only
inside an interior gap, which is what the fold reports and zeroes.

Both verbs work on bare arrays and carry no estimator. The type bound is `AbstractMatrix{Bool}`, so
a caller's own declaration — a listing calendar, or a constituency that leaves and rejoins — enters
at the same point and replaces the derived active mask outright. The estimation mask is never the
caller's to state and is always re-derived, which is what makes `emsk ⊆ amsk` hold by construction.
See `docs/adr/0129-the-ingestion-layer-seams-at-the-listing-span-and-the-clock-draws-the-pipeline-boundary.md`
and `docs/adr/0131-a-return-needs-two-consecutive-prices-and-a-gap-return-writes-only-the-cells-that-lack-them.md`.

```@docs
listing_span
universe_masks
```
