```@meta
Description = "Fixed Weighted Return Forecast, private API of PortfolioOptimisers.jl: ForecastStepHistory, assert_signed_composite_weights, signed_composite_weights, …"
```

# Fixed Weighted Return Forecast: private API

## Types

```@docs
ForecastStepHistory
```

## Functions

```@docs
assert_signed_composite_weights
signed_composite_weights
signed_composite_accumulate!
return_forecast_step(rfe::FixedWeightedReturnForecast, P::PanelScores, csfm::CrossSectionalFactorModel, fs::Nothing, cre::Nothing = nothing)
return_forecast_result(rfe::FixedWeightedReturnForecast, hist::MatNum, fs::Nothing)
```
