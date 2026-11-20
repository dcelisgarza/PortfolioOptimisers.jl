```@meta
Description = "EW volatility descriptors, private API of PortfolioOptimisers.jl: EWVolatilityState, ew_variance_estimator, ew_volatility_input, assert_mask_aware_variance, …"
```

# EW volatility descriptors: private API

## Types

```@docs
EWVolatilityState
```

## Functions

```@docs
ew_variance_estimator
ew_volatility_input
assert_mask_aware_variance
window_leaves_warmup
ew_residual_returns
ew_residual_unresolved!
Base.copy(x::PortfolioOptimisers.EWVolatilityState)
ew_volatility_parts
ew_volatility_state
ew_volatility_variance
ew_volatility_fold
assert_ew_volatility_finite
show_fields(de::Union{PortfolioOptimisers.EWVolatility, PortfolioOptimisers.EWResidualVolatility})
```
