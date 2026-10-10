```@meta
Description = "EW volatility descriptors, private API of PortfolioOptimisers.jl: EWVolatilityState, ew_variance_estimator, ew_volatility_input, ew_residual_returns, …"
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
ew_residual_returns
Base.copy(x::PortfolioOptimisers.EWVolatilityState)
ew_volatility_parts
ew_volatility_state
ew_volatility_variance
ew_volatility_fold
show_fields(de::Union{PortfolioOptimisers.EWVolatility, PortfolioOptimisers.EWResidualVolatility})
```
