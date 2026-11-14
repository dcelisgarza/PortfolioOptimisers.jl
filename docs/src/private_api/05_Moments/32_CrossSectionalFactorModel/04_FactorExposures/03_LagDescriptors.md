```@meta
Description = "Lag Descriptors, private API of PortfolioOptimisers.jl: LagDescriptorState, FieldScaleValues, assert_descriptor_lag, Base.copy, lag_descriptor_inputs, …"
```

# Lag Descriptors: private API

## Types

```@docs
LagDescriptorState
FieldScaleValues
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
