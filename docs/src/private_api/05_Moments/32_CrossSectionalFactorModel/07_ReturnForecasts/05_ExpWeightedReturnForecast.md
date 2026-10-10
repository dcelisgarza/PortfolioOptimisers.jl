```@meta
Description = "Exp weighted Return Forecast, private API of PortfolioOptimisers.jl: ew_forecast_weights, ew_forecast_valid, ew_forecast_design, ew_forecast_accumulate!, …"
```

# Exp weighted Return Forecast: private API

## Functions

```@docs
ew_forecast_weights
ew_forecast_valid
ew_forecast_design
ew_forecast_accumulate!
ew_forecast_solve
ew_forecast_state
ew_forecast_history
return_forecast_step(rfe::ExpWeightedReturnForecast, P::NamedTuple, csfm::CrossSectionalFactorModel, fs::Option{<:NamedTuple})
return_forecast_result(rfe::ExpWeightedReturnForecast, hist::MatNum, fs::NamedTuple)
```
