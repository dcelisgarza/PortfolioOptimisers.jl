# Price gap fill

## The price gap fill

A **Held Price** is the last priced observation of an asset, carried forward across a gap. Stating
one is what [`PriceGapFill`](@ref) is for: it is the ingestion layer's only fill, it is off unless a
caller adds it, and it exists to state a *price convention* across a suspension rather than to
remove a gap — a gap is carried through the conversion, so nothing downstream needs it gone.

The fill is bounded by the **Listing Span**, so it touches Held Gaps alone and can never fabricate a
price where an asset was not yet listed or has been delisted. It is fitted on a training window and
replayed: [`CarriedPrice`](@ref) records the last observed training price, which seeds a
carry-forward on a window that opens inside a gap, and a [`Num_VecToScaM`](@ref) records the
reduction of that window's observed prices. A per-asset constant manufactures two moves the market
never printed, which is why the carried price is the convention a caller reaching for a fill
normally wants.

It runs at the price level, before [`PricesToReturns`](@ref), because a carried price cannot be
stated after the conversion: zeroing the returns a gap left non-finite discards the move across the
gap entirely. A filled cell is therefore finite in the returns and its estimation mask entry is
`true` — a caller who filled has said the asset traded. See
`docs/adr/0130-a-universe-policy-is-fitted-and-the-only-fill-is-a-span-bounded-price-convention.md`.

```@docs
CarriedPrice
PriceGapFill
PriceGapFillResult
PortfolioOptimisers.carrier_listing_span
PortfolioOptimisers.gap_fill_span
PortfolioOptimisers.gap_fill_seed
PortfolioOptimisers.gap_fill_column!
```
