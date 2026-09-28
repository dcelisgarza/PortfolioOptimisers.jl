```@meta
Description = "Observed Exposure, public API of PortfolioOptimisers.jl: AbstractObservedExposureEstimator, ObservedExposure, CurrencyExposure, factor_exposure, …"
```

# [Observed Exposure](@id api-observed-exposure)

## Types

```@docs
AbstractObservedExposureEstimator
ObservedExposure
CurrencyExposure
```

## Functions

```@docs
factor_exposure(xe::ObservedExposure, rd::ReturnsResult)
factor_exposure(xe::CurrencyExposure, rd::ReturnsResult)
observed_series
```
