```@meta
Description = "Target Return Forecast Step, private API of PortfolioOptimisers.jl: TargetDesign, CalibrationSums, TargetRecursion, target_forecast_folds, …"
```

# [Target Return Forecast Step: private API](@id private-api-target-return-forecast-step)

## Types

```@docs
TargetDesign
CalibrationSums
TargetRecursion
```

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
return_forecast_step(rfe::TargetReturnForecast, P::PanelScores, csfm::CrossSectionalFactorModel, fs::Option{<:TargetRecursion}, cre::Option{<:AbstractCrossSectionalRegressionEstimator} = nothing)
return_forecast_result(rfe::TargetReturnForecast, hist::MatNum, fs::TargetRecursion)
```
