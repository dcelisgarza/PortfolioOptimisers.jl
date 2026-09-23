```@meta
Description = "Online selection rules: the forecast-reading arm, public API of PortfolioOptimisers.jl: ForecastReversion, MovingAverageReversion, …"
```

# Online selection rules: the forecast-reading arm

The rules that consume a Price Relative Forecast: the passive-aggressive reversion step with its optional diagonal scale and the six paper constructors that fill its forecaster, the fixed-length tracking step with its three, the kernel-scaled tracking step with its one, the cost-aware soft-thresholded step and the sparse portfolio, whose iterate one of three algorithms finds. A forecaster with an exact fold is carried on the Rule State; one without is refit on the rows the head holds.

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
