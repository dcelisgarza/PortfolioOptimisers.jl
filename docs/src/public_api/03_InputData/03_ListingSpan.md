```@meta
Description = "The Listing Span, public API of PortfolioOptimisers.jl: listing_span, universe_masks."
```

# The Listing Span

## The listing span

The listing span of an asset is the range of observations over which the asset is listed.
[`listing_span`](@ref) finds one span per asset column from the position of each gap in the
prices. A run of gaps at the start of a column means that the asset is not listed yet. A run of
gaps at the end means that the asset was delisted. A gap with prices on both sides is a suspension
or a holiday, and the asset is still listed and still held. With this rule, you can derive the
universe from the prices alone, without a listing calendar.

[`universe_masks`](@ref) moves the spans onto the dates of the returns and makes the two universe
masks that an [`AssetPanel`](@ref) carries. The active mask marks the assets that are listed at
each observation. The estimation mask is the active mask where the return is also finite. A return
needs the earlier price of its pair. A span from `first` to `last` on the prices therefore becomes
`[first + 1, last]` on padded returns, and `[first, last - 1]` on returns without padding. The active
mask then covers exactly the finite returns of a column, so the listing of an asset never shows
as a missing return. The two masks differ only inside a gap with prices on both sides. There
the asset is listed but has no return, and a cross-validation fold that holds the asset reports
the observation and counts its return as zero.

Both functions take plain arrays, and neither needs an estimator. `universe_masks` also accepts
your own `AbstractMatrix{Bool}` in place of the spans, such as a listing calendar or the
membership of an index that an asset leaves and joins again. Your matrix takes the place of the
derived spans, and `universe_masks` moves it onto the dates of the returns in the same way. You
cannot give the estimation mask. `universe_masks` always derives it from
the active mask and the returns, so the estimation mask never marks a cell that the active mask
leaves out.

```@docs
listing_span
universe_masks
```
