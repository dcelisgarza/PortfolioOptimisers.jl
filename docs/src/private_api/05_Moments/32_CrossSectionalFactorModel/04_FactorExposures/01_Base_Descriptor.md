```@meta
Description = "Base descriptor, private API of PortfolioOptimisers.jl: EWBetaState, DescriptorFold, CarriedEstimators, descriptor_field_values, descriptor_asset_panel, …"
```

# Base descriptor: private API

## Types

```@docs
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
ew_beta_series
ew_beta_series!
ew_beta_state
Base.copy(x::PortfolioOptimisers.EWBetaState)
ew_beta_reset!
nan_fill_value
descriptor_active_fill!
positive_divide
lookback_max
```
