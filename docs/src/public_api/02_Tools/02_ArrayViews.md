```@meta
Description = "Array views, public API of PortfolioOptimisers.jl: port_opt_view, InvestableIndex, selection_index, obs_weights_view."
```

# Array views

## View functions

[`NestedClustered`](@ref) runs one inner optimisation for each cluster of assets, so it needs each estimator, constraint and result restricted to the assets of that cluster. `port_opt_view` restricts an object to a subset of the assets. `obs_weights_view` restricts it to a subset of the observations.

The entry of a fit gives its views an [`InvestableIndex`](@ref), which also holds the mask of the assets that the fit can trade, so a method of `port_opt_view` that charges the sale of an asset that left can dispatch on it. A meta-optimiser gives its inner heads the plain index of [`selection_index`](@ref), so they read no departure of their parent.

```@docs
port_opt_view(x, i, args...)
port_opt_view(x::AbstractVector{<:Union{Nothing, AbstractEstimator, AbstractAlgorithm, AbstractResult}}, i, args...; kwargs...)
InvestableIndex
selection_index
obs_weights_view(x, ::Any)
obs_weights_view(x::AbstractVector{<:Union{Nothing, AbstractEstimator, AbstractAlgorithm, AbstractResult}}, i)
```
