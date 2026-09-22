```@meta
Description = "Price-level expected returns, public API of PortfolioOptimisers.jl: AbstractPriceLevelStatistic, MovingAverage, ExponentialMovingAverage, SpatialMedian, …"
```

# Price-level expected returns

The Price Relative Forecast of the online portfolio selection family, as an expected-returns estimator any `me` slot may hold: the level of the last observation is set to one, the earlier levels are reconstructed from the returns, and the expected return is a statistic of those levels over the last one, less one. A windowed statistic reads its last `window - 1` returns and truncates the window over the first rows; a folding statistic is an exact recursion over the price relatives, folded through `partial_fit!` and read with `mean(me)`, and may carry a memory of the last relatives beside its vector. A folding statistic is mask-aware: it resets an asset the active mask turns off, so a relisting starts cold, and it answers `NaN` for an asset that has folded no level.

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
