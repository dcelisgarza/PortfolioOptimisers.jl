```@meta
Description = "Fixed Weighted Return Forecast, public API of PortfolioOptimisers.jl: FixedWeightedReturnForecast, FixedWeightedReturnForecastResult, return_forecast, …"
```

# [Fixed Weighted Return Forecast](@id api-fixed-weighted-return-forecast)

## Types

```@docs
FixedWeightedReturnForecast
FixedWeightedReturnForecastResult
```

## Functions

```@docs
return_forecast(rfe::FixedWeightedReturnForecast, rd::ReturnsResult, csfm::CrossSectionalFactorModel)
port_opt_view(rf::FixedWeightedReturnForecastResult, i, args...)
```
