```@meta
Description = "Rolling Descriptors, public API of PortfolioOptimisers.jl: RollingLogReturn, RollingMax, descriptor, partial_fit!, merge_states, descriptor_step, …"
```

# [Rolling Descriptors](@id api-rolling-descriptors)

## Types

```@docs
RollingLogReturn
RollingMax
```

## Functions

```@docs
descriptor(de::RollingLogReturn, rd::ReturnsResult)
PortfolioOptimisers.partial_fit!(de::RollingLogReturn{<:Any, <:Any, <:Any, <:Any, <:PortfolioOptimisers.Option{<:PortfolioOptimisers.RollingLogReturnState}}, rd::ReturnsResult)
merge_states(::PortfolioOptimisers.RollingLogReturnState, ::PortfolioOptimisers.RollingLogReturnState)
PortfolioOptimisers.descriptor_step
PortfolioOptimisers.carry_lookback
descriptor(de::PortfolioOptimisers.CarriedDescriptor, rd::ReturnsResult)
RollingMomentum
Reversal
MaxReturn
```
