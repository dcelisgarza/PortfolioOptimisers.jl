```@meta
Description = "Base descriptor, private API of PortfolioOptimisers.jl: EWMoments, EWReturnWindow, EWBetaState, DescriptorFold, CarriedEstimators, descriptor_field_values, …"
```

# Base descriptor: private API

## Types

```@docs
EWMoments
EWReturnWindow
EWBetaState
DescriptorFold
CarriedEstimators
```

## Functions

```@docs
descriptor_field_values
descriptor_asset_panel
assert_log_returns
market_return_series
ew_moments
Base.copy(x::PortfolioOptimisers.EWMoments)
ew_moments_reset!
ew_moments_update!
ew_moments_vary
ew_return_window
Base.copy(x::PortfolioOptimisers.EWReturnWindow)
ew_window_reset!
ew_window_add!
Base.copy(x::PortfolioOptimisers.EWBetaState)
ew_beta_state
ew_regression_series!
ew_regression_reset!
ew_regression_add!
ew_reference_load!
ew_regression_take!
ew_regression_close!
ew_regression_update!
ew_regression_refresh!
ew_model_reset!
ew_model_update!
ew_model_beta
ew_beta_shrink_row!
nan_fill_value
descriptor_active_fill!
positive_divide
lookback_max
```
