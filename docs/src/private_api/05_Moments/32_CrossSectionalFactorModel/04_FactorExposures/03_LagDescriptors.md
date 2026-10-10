```@meta
Description = "Lag Descriptors, private API of PortfolioOptimisers.jl: LagDescriptorState, assert_descriptor_lag, Base.copy, lag_descriptor_inputs, lag_descriptor_lagged, …"
```

# Lag Descriptors: private API

## Types

```@docs
LagDescriptorState
```

## Functions

```@docs
assert_descriptor_lag
Base.copy(x::PortfolioOptimisers.LagDescriptorState)
lag_descriptor_inputs
lag_descriptor_lagged
lag_descriptor_value
lag_state_seed
lag_descriptor_fold
show_fields(de::Union{PortfolioOptimisers.GrowthRate, PortfolioOptimisers.ChangeToScale, PortfolioOptimisers.ChangeInIntensity})
```
