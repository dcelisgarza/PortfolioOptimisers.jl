```@meta
Description = "Price-level expected returns, public API of PortfolioOptimisers.jl: AbstractPriceLevelStatistic, MovingAverage, ExponentialMovingAverage, SpatialMedian, …"
```

# Price-level expected returns

`PriceLevelExpectedReturns` forecasts the next return of each asset from its recent price levels. Many online portfolio selection rules read this forecast, and any field that takes an expected returns estimator, `me`, can hold it. It sets the price level of the last observation to one and rebuilds the earlier levels from the returns. The expected return is then a statistic of those levels, such as a moving average, divided by the last level, minus one.

A window statistic reads the last `window - 1` returns, and it uses fewer rows at the start of the data. A recursive statistic updates exactly with each new observation. Add observations with `partial_fit!`, and read the forecast with `mean(me)`. It can also keep the last few price relatives next to its running value. A recursive statistic reads the active mask. It restarts an asset that the mask turns off, so a relisted asset starts from nothing, and it returns `NaN` for an asset that has no price level yet.

```@docs
PortfolioOptimisers.AbstractPriceLevelStatistic
MovingAverage
ExponentialMovingAverage
SpatialMedian
WindowPeak
LaggedPrice
ReweightedPriceRelative
PriceLevelExpectedReturns
PortfolioOptimisers.PriceLevelForecastState
PortfolioOptimisers.rows_needed(me::PriceLevelExpectedReturns)
PortfolioOptimisers.window_rows
PortfolioOptimisers.folds
PortfolioOptimisers.memory_rows
PortfolioOptimisers.price_level_statistic
PortfolioOptimisers.fold_statistic
PortfolioOptimisers.cold_statistic
mean(me::PriceLevelExpectedReturns, X::MatNum; dims::Int = 1, active_mask::Option{<:AbstractMatrix{<:Bool}} = nothing, kwargs...)
mean(me::PriceLevelExpectedReturns, X::MatNum, pnl::Option{<:AssetPanel}; dims::Int = 1, kwargs...)
mean(me::PriceLevelExpectedReturns; kwargs...)
partial_fit!(me::PriceLevelExpectedReturns, x::VecNum; active_mask::Option{<:AbstractVector{<:Bool}} = nothing, kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
