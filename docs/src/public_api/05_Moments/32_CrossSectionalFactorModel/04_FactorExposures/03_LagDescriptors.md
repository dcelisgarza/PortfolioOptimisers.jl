```@meta
Description = "Lag Descriptors, public API of PortfolioOptimisers.jl: GrowthRate, ChangeToScale, ChangeInIntensity, descriptor, partial_fit!, merge_states, …"
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
descriptor(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity}, rd::ReturnsResult)
PortfolioOptimisers.partial_fit!(de::Union{GrowthRate, ChangeToScale, ChangeInIntensity}, rd::ReturnsResult)
merge_states(a::PortfolioOptimisers.LagDescriptorState, b::PortfolioOptimisers.LagDescriptorState)
AssetsGrowthRate
SalesGrowthRate
IssuanceGrowthRate
EarningsChangeToPrice
CapexToAssetsChangeInIntensity
```
