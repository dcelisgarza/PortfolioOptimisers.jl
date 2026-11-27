```@meta
Description = "Preprocessing partial fit, public API of PortfolioOptimisers.jl: fit_preprocessing."
```

# Preprocessing partial fit

## Incremental updates of the data steps

A [`Pipeline`](@ref) accepts `partial_fit!`, and so do the data steps before its prior step, or
before its optimiser step when it has no prior. A step that transforms each row on its own updates
with each block. It takes a block of observations and returns the transformed block, through
[`PortfolioOptimisers.partial_fit_transform`](@ref). To do so, [`PricesToReturns`](@ref) stores the
last row of prices it saw, [`PriceGapFill`](@ref) with a [`CarriedPrice`](@ref) stores the carried
prices, [`MissingDataFilter`](@ref) stores its counts of missing values, and
[`PriceJumpFilter`](@ref) stores the assets that jumped and their prices at each end. Call
[`fit_preprocessing`](@ref) on such a step with no data to get its fitted result. A step whose
configuration needs a whole window returns `false` from
[`PortfolioOptimisers.supports_partial_fit`](@ref), and the pipeline throws an error that names the
step before the first update.

```@docs
fit_preprocessing(ptr::PricesToReturns)
fit_preprocessing(est::PriceGapFill)
fit_preprocessing(mdf::MissingDataFilter)
fit_preprocessing(pjf::PriceJumpFilter)
```
