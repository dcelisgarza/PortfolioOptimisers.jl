```@meta
Description = "Target Return Forecast Step, private API of PortfolioOptimisers.jl: target_forecast_folds, target_forecast_design, return_forecast_step, …"
```

# [Target Return Forecast Step: private API](@id private-api-target-return-forecast-step)

## Functions

```@docs
target_forecast_folds
target_forecast_design
target_forecast_calibration_state
target_forecast_calibration_step
target_forecast_calibrated
target_forecast_state
target_forecast_latest_row
target_forecast_matured!
return_forecast_step(rfe::TargetReturnForecast, P::NamedTuple, csfm::CrossSectionalFactorModel, fs::Option{<:NamedTuple}, cre::Option{<:AbstractCrossSectionalRegressionEstimator})
return_forecast_result(rfe::TargetReturnForecast, hist::MatNum, fs::NamedTuple)
```
