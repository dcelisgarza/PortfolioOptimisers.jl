```@meta
Description = "EW mean descriptors, public API of PortfolioOptimisers.jl: EWMean, EWVolumeRatio, DaysToCover, descriptor, partial_fit!, merge_states, EWMomentum, …"
```

# [EW mean descriptors](@id api-ew-mean-descriptors)

## Types

```@docs
EWMean
EWVolumeRatio
DaysToCover
```

## Functions

```@docs
descriptor(de::EWMean, rd::ReturnsResult)
PortfolioOptimisers.partial_fit!(de::Union{EWMean, EWVolumeRatio, DaysToCover}, rd::ReturnsResult)
merge_states(::PortfolioOptimisers.EWMeanState, ::PortfolioOptimisers.EWMeanState)
EWMomentum
EWShareTurnover
EWAmihudIlliquidity
```
