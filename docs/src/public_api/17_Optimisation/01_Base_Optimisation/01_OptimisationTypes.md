```@meta
Description = "Optimisation types, public API of PortfolioOptimisers.jl: AbstractOptimisationEstimator, OptimisationEstimator, NonFiniteAllocationOptimisationEstimator, …"
```

# Optimisation types

Every optimiser has a full type name, and many also have a short alias, which the [aliases](@aliases) page lists.

```@docs
AbstractOptimisationEstimator
OptimisationEstimator
NonFiniteAllocationOptimisationEstimator
OptimisationSuccess
OptimisationFailure
factory(res::NonFiniteAllocationOptimisationResult, fb::Option{<:OptE_Opt_FbChain})
factory(opt::OptE_Opt, ::Any)
BaseOptimisationEstimator
OptimisationAlgorithm
OptimisationResult
NonFiniteAllocationOptimisationResult
OptimisationReturnCode
OptimisationModelResult
needs_previous_weights(::OptE_Opt)
```
