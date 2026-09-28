```@meta
Description = "EW volatility descriptors, public API of PortfolioOptimisers.jl: EWVolatility, EWResidualVolatility, descriptor, EWDownsideVolatility, …"
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
EWDownsideVolatility
EWResidualDownsideVolatility
```
