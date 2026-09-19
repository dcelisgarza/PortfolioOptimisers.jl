```@meta
Description = "Composite price-level statistics, public API of PortfolioOptimisers.jl: TruncatedExponentialMovingAverage, GaussianWeightedDoubleEstimate, …"
```

# Composite price-level statistics

The statistics the later papers of the forecast-reading arm add to [`PriceLevelExpectedReturns`](@ref): a truncated exponential average, a Gaussian-weighted double estimate, a statistic that switches per asset on a trend test, and a radial-basis composite of several trends centred on the one with the best worst back-tested return.

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
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
