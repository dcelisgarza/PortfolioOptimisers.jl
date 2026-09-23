```@meta
Description = "Price gap fill, public API of PortfolioOptimisers.jl: CarriedPrice, PriceGapFill, PriceGapFillResult."
```

# Price gap fill

## The price gap fill

[`PriceGapFill`](@ref) fills the price gaps inside the listing span of an asset. It is the only
fill of the price data, and it runs only if you add it. Use it to state what the price was during a
suspension or a holiday. You do not need it to remove gaps, because the conversion to returns and
every later step accept a `NaN`.

The fill writes only inside the listing span, so it never makes a price before an asset is listed
or after it is delisted. You fit it on a training window and apply it to later windows. With
[`CarriedPrice`](@ref), the fill carries the last observed price forward across the gap. The fitted
result stores the last price of the training window, and the fill starts from it when a later
window opens inside a gap. With a [`Num_VecToScaM`](@ref), a fixed number or a reduction such as the
median, the fill writes one constant per asset, computed from the observed prices of the training
window. A constant makes two price moves that the market never made, one into the gap and one out
of it. `CarriedPrice` makes no such moves, and it is usually the fill you want.

The fill runs on prices, before [`PricesToReturns`](@ref). After the conversion, the only repair is
to set the missing returns to zero, and that loses the price move across the gap. For the prices
`p₀, _, _, p₃`, the carried prices give the returns `0, 0, p₃/p₀ - 1`. A filled cell has a finite
return, and its estimation mask entry is `true`, because a fill states that the asset traded.

```@docs
CarriedPrice
PriceGapFill
PriceGapFillResult
```
