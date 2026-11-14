```@meta
Description = "Base descriptor, public API of PortfolioOptimisers.jl: AbstractDescriptorEstimator, DescriptorStepResult, descriptor, lookback, merge_states."
```

# [Base descriptor](@id api-base-descriptor)

## Types

```@docs
AbstractDescriptorEstimator
DescriptorStepResult
```

## Functions

```@docs
descriptor
PortfolioOptimisers.lookback
merge_states(::PortfolioOptimisers.EWBetaState, ::PortfolioOptimisers.EWBetaState)
```
