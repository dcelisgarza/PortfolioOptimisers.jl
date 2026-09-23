```@meta
Description = "Partial fit, public API of PortfolioOptimisers.jl: partial_fit!, partial_fit, merge_states, obs_weights_view."
```

# Partial fit

An incremental fit adds new observations to an estimate without reading the earlier observations again. [`partial_fit!`](@ref) adds a block of observations to an estimator and returns the updated estimator. [`partial_fit`](@ref) does the same to a copy of the state, so the estimator you pass in keeps its old state. The running quantities of the fit are stored in an [`AbstractPartialFitState`](@ref). [`merge_states`](@ref) combines the states of two separate blocks of observations into the state of the two blocks placed end to end.

```@docs
partial_fit!
partial_fit
partial_fit(est::Union{<:PortfolioOptimisers.AbstractEstimator, <:StatsBase.CovarianceEstimator}, args...; kwargs...)
merge_states
PortfolioOptimisers.obs_weights_view(::PortfolioOptimisers.AbstractPartialFitState, ::Any)
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
