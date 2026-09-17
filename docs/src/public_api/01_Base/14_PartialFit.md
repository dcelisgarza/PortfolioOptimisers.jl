```@meta
Description = "Partial fit, public API of PortfolioOptimisers.jl: partial_fit!, partial_fit, obs_weights_view."
```

# Partial fit

An incremental fit folds one observation into an estimate without reading the sample again. [`partial_fit!`](@ref) is the verb each family writes, [`partial_fit`](@ref) is the value form that folds a copy of the state, its running quantities live in a [`AbstractPartialFitState`](@ref), and [`merge_states`](@ref) combines the states of two disjoint blocks of observations into the state of the concatenated block.

```@docs
partial_fit!
partial_fit
partial_fit(est::Union{<:PortfolioOptimisers.AbstractEstimator, <:StatsBase.CovarianceEstimator}, args...; kwargs...)
PortfolioOptimisers.obs_weights_view(::PortfolioOptimisers.AbstractPartialFitState, ::Any)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
