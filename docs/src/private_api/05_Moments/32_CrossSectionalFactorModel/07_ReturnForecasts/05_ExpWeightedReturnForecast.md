```@meta
Description = "Exp weighted Return Forecast, private API of PortfolioOptimisers.jl: EWNormalEquations, ew_forecast_weights, ew_forecast_valid, ew_forecast_design, …"
```

# Exp weighted Return Forecast: private API

## Types

```@docs
EWNormalEquations
```

## Functions

```@docs
ew_forecast_weights
ew_forecast_valid
ew_forecast_design
ew_forecast_accumulate!
ew_forecast_solve
ew_forecast_state
ew_forecast_history
return_forecast_step(rfe::ExpWeightedReturnForecast, P::PanelScores, csfm::CrossSectionalFactorModel, fs::Option{<:EWNormalEquations}, cre::Nothing = nothing)
return_forecast_result(rfe::ExpWeightedReturnForecast, hist::MatNum, fs::EWNormalEquations)
```
