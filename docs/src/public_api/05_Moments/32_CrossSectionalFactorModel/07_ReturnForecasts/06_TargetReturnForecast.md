```@meta
Description = "Target Return Forecast, public API of PortfolioOptimisers.jl: TargetReturnForecast, TargetReturnForecastResult, AbstractCalibrationWarmup, NaNWarmup, …"
```

# [Target Return Forecast](@id api-target-return-forecast)

## Types

```@docs
TargetReturnForecast
TargetReturnForecastResult
AbstractCalibrationWarmup
NaNWarmup
InSampleWarmup
```

## Functions

```@docs
return_forecast(rfe::TargetReturnForecast, rd::ReturnsResult, csfm::CrossSectionalFactorModel)
target_forecast_warmup
```
