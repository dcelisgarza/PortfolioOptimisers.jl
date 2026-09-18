```@meta
Description = "Lag Descriptors, public API of PortfolioOptimisers.jl: GrowthRate, ChangeToScale, ChangeInIntensity, descriptor, AssetsGrowthRate, SalesGrowthRate, …"
```

# [Lag Descriptors](@id api-lag-descriptors)

## Types

```@docs
GrowthRate
ChangeToScale
ChangeInIntensity
```

## Functions

```@docs
descriptor(de::GrowthRate, rd::ReturnsResult)
AssetsGrowthRate
SalesGrowthRate
IssuanceGrowthRate
EarningsChangeToPrice
CapexToAssetsChangeInIntensity
```
