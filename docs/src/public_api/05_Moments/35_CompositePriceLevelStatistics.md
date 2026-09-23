```@meta
Description = "Composite price-level statistics, public API of PortfolioOptimisers.jl: TruncatedExponentialMovingAverage, GaussianWeightedDoubleEstimate, …"
```

# Composite price-level statistics

This page has more statistics for [`PriceLevelExpectedReturns`](@ref), from later papers on online portfolio selection. They are a truncated exponential moving average, a double estimate with Gaussian weights, and a statistic that switches per asset between three others on the sign of a trend test. There is also a radial-basis mix of several trend forecasts, centred on the trend with the best worst-case return over the last `window` periods. The last is the three-state price prediction of kernel trend pattern tracking, which keeps its previous prediction and fits its first state with an elastic-net regression.

```@docs
TruncatedExponentialMovingAverage
GaussianWeightedDoubleEstimate
PortfolioOptimisers.AbstractTrendTest
PairwiseSlopeSum
RegressionSlope
PortfolioOptimisers.trend_sign
TrendSwitch
CompositeTrend
PortfolioOptimisers.member_statistic
ElasticNetPath
KernelTrendPattern
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
