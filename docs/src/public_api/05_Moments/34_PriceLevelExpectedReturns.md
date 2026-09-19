```@meta
Description = "Price-level expected returns, public API of PortfolioOptimisers.jl: AbstractPriceLevelStatistic, MovingAverage, ExponentialMovingAverage, SpatialMedian, …"
```

# Price-level expected returns

The Price Relative Forecast of the online portfolio selection family, as an expected-returns estimator any `me` slot may hold: the level of the last observation is set to one, the earlier levels are reconstructed from the returns, and the expected return is a statistic of those levels over the last one, less one. A windowed statistic reads its last `window - 1` returns and truncates the window over the first rows; a folding statistic is an exact recursion over the price relatives, folded through `partial_fit!` and read with `mean(me)`.

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
PortfolioOptimisers.price_level_statistic
PortfolioOptimisers.fold_statistic
mean(me::PriceLevelExpectedReturns, X::MatNum; dims::Int = 1, kwargs...)
mean(me::PriceLevelExpectedReturns; kwargs...)
partial_fit!(me::PriceLevelExpectedReturns, x::VecNum; kwargs...)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
