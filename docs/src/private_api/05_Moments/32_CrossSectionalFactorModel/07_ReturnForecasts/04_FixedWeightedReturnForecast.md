```@meta
Description = "Fixed Weighted Return Forecast, private API of PortfolioOptimisers.jl: assert_signed_composite_weights, signed_composite_weights, …"
```

# Fixed Weighted Return Forecast: private API

## Functions

```@docs
assert_signed_composite_weights
signed_composite_weights
signed_composite_accumulate!
return_forecast_step(rfe::FixedWeightedReturnForecast, P::NamedTuple, csfm::CrossSectionalFactorModel, fs::Nothing)
return_forecast_result(rfe::FixedWeightedReturnForecast, hist::MatNum, fs::Nothing)
```
