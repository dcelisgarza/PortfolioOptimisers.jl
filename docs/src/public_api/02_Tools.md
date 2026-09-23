```@meta
Description = "Tools, public API of PortfolioOptimisers.jl: traverse_concrete_subtypes, concrete_typed_array, factory, @propagatable, factory_child, @fprop, @vprop, …"
```

# Tools

This page lists the general functions, macros and types that the rest of the library uses.

## Utility functions

`traverse_concrete_subtypes` lists the struct types under an abstract type. `concrete_typed_array` converts an array with an abstract element type into an array with a concrete element type. `factory` rebuilds an estimator with the values a fit supplies, such as the prior moments, the observation weights and the previous portfolio weights, and `factory_child` applies it to one field. `@propagatable` defines a struct and writes its `factory` and `port_opt_view` methods from the tags on its fields, which are `@fprop`, `@vprop`, `@pprop`, `@wprop` and `@cprop`. `@forward_properties` writes `getproperty` and `propertynames` for a type from a list of forwarding rules.

```@docs
traverse_concrete_subtypes
concrete_typed_array
factory(a::Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}, args...; kwargs...)
@propagatable
factory_child
@fprop
@vprop
@pprop
@wprop
@cprop
@forward_properties
```

## View functions

[`NestedClustered`](@ref) runs one inner optimisation for each cluster of assets, so it needs each estimator, constraint and result restricted to the assets of that cluster. `port_opt_view` restricts an object to a subset of the assets. `obs_weights_view` restricts it to a subset of the observations.

```@docs
port_opt_view(x, i, args...)
port_opt_view(x::VecScalar, i, args...)
port_opt_view(x::AbstractVector{<:Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}}, i, args...; kwargs...)
obs_weights_view(x, ::Any)
obs_weights_view(x::AbstractVector{<:Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}}, i)
```

## Summary statistics

Some estimators and constraints reduce a vector to one number, such as its minimum, mean, median or maximum. Each type below names one reduction, and some of them take observation weights. `vec_to_real_measure` applies the reduction to a vector.

```@docs
VectorToScalarMeasure
MinValue
MeanValue
MedianValue
MaxValue
StdValue
VarValue
SumValue
ProdValue
ModeValue
StandardisedValue
factory(mv::MeanValue, args...; kwargs...)
factory(mdv::MedianValue, args...; kwargs...)
factory(sv::StdValue, args...; kwargs...)
factory(vv::VarValue, args...; kwargs...)
factory(msv::StandardisedValue, args...; kwargs...)
vec_to_real_measure
```
