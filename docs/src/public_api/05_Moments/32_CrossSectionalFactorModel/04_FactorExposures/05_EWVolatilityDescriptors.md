```@meta
Description = "Exponentially Weighted Volatility Descriptors, public API of PortfolioOptimisers.jl: EWVolatility, EWResidualVolatility, descriptor, EWDownsideVolatility, …"
```

# [Exponentially Weighted Volatility Descriptors](@id api-ew-volatility-descriptors)

## Types

```@docs
EWVolatility
EWResidualVolatility
```

## Functions

```@docs
descriptor(de::EWVolatility, rd::ReturnsResult)
descriptor(de::EWResidualVolatility, rd::ReturnsResult)
EWDownsideVolatility
EWResidualDownsideVolatility
```
