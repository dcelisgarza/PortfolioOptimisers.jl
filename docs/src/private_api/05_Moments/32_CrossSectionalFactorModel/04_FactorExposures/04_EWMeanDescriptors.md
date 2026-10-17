```@meta
Description = "EW mean descriptors, private API of PortfolioOptimisers.jl: EWMeanState, half_life_decay, half_life_min_obs, assert_ew_decay, assert_ew_ratio_side, …"
```

# EW mean descriptors: private API

## Types

```@docs
EWMeanState
```

## Functions

```@docs
half_life_decay
half_life_min_obs
assert_ew_decay
assert_ew_ratio_side
ew_ratio_values
ew_mean_series
ew_mean_series!
Base.copy(x::PortfolioOptimisers.EWMeanState)
ew_state_buffer
ew_mean_state
ew_mean_delayed!
ew_mean_fold
show_fields(de::Union{PortfolioOptimisers.EWMean, PortfolioOptimisers.EWVolumeRatio, PortfolioOptimisers.DaysToCover})
```
