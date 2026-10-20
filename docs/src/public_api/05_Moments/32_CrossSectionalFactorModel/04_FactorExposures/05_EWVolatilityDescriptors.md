```@meta
Description = "EW volatility descriptors, public API of PortfolioOptimisers.jl: EWVolatility, EWResidualVolatility, descriptor, partial_fit!, merge_states, …"
```

# [EW volatility descriptors](@id api-ew-volatility-descriptors)

## Types

```@docs
EWVolatility
EWResidualVolatility
```

## Functions

```@docs
descriptor(de::EWVolatility, rd::ReturnsResult)
descriptor(de::EWResidualVolatility, rd::ReturnsResult)
PortfolioOptimisers.partial_fit!(de::Union{EWVolatility{<:RegimeAdjustedExpWeightedVariance}, EWResidualVolatility{<:Any, <:RegimeAdjustedExpWeightedVariance}}, rd::ReturnsResult)
merge_states(::PortfolioOptimisers.EWVolatilityState, ::PortfolioOptimisers.EWVolatilityState)
EWDownsideVolatility
EWResidualDownsideVolatility
```
