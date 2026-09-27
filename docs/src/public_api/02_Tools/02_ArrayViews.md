```@meta
Description = "Array views, public API of PortfolioOptimisers.jl: port_opt_view, obs_weights_view."
```

# Array views

## View functions

[`NestedClustered`](@ref) runs one inner optimisation for each cluster of assets, so it needs each estimator, constraint and result restricted to the assets of that cluster. `port_opt_view` restricts an object to a subset of the assets. `obs_weights_view` restricts it to a subset of the observations.

```@docs
port_opt_view(x, i, args...)
port_opt_view(x::VecScalar, i, args...)
port_opt_view(x::AbstractVector{<:Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}}, i, args...; kwargs...)
obs_weights_view(x, ::Any)
obs_weights_view(x::AbstractVector{<:Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}}, i)
```
