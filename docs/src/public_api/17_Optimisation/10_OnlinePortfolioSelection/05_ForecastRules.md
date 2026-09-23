```@meta
Description = "Online selection rules: the forecast-reading arm, public API of PortfolioOptimisers.jl: ForecastReversion, MovingAverageReversion, …"
```

# Online selection rules: the forecast-reading arm

These rules read a forecast of the next price relatives, which the expected returns estimator in their `me` field makes. `ForecastReversion` is the passive-aggressive reversion step, with an optional diagonal scale. Six constructors build it with the forecast of their paper: `MovingAverageReversion`, `ExponentialMovingAverageReversion`, `RobustMedianReversion`, `ReweightedPriceRelativeTracking`, `GaussianWeightingReversion` and `LocalAdaptiveLearning`.

`ForecastTracking` is the tracking step of fixed length, and three constructors build it: `PeakPriceTracking`, `AdaptiveInputCompositeTrend` and `TrendPromotePriceTracking`. `KernelTrendTracking` scales the tracking step with a kernel, and `KernelTrendPatternTracking` builds it. `TransactionCostOptimisation` takes a step that accounts for transaction costs and leaves small trades at zero. `ShortTermSparsePortfolio` is the sparse portfolio, which uses one of three algorithms to find the point it projects.

A forecast with an exact update stores its state with the state of the rule. Any other forecast refits on the rows that `OnlinePortfolioSelection` stores.

```@docs
ForecastReversion
MovingAverageReversion
ExponentialMovingAverageReversion
RobustMedianReversion
ReweightedPriceRelativeTracking
GaussianWeightingReversion
LocalAdaptiveLearning
ForecastTracking
PeakPriceTracking
AdaptiveInputCompositeTrend
TrendPromotePriceTracking
KernelTrendTracking
KernelTrendPatternTracking
TransactionCostOptimisation
ShortTermSparsePortfolio
PortfolioOptimisers.ForecasterState
PortfolioOptimisers.forecast_relative
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
