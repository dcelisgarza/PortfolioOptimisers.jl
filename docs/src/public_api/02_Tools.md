```@meta
Description = "Tools, public API of PortfolioOptimisers.jl: traverse_concrete_subtypes, concrete_typed_array, factory, @propagatable, factory_child, @fprop, @vprop, …"
```

# Tools

`PortfolioOptimisers.jl` is a complex codebase which uses a variety of general purpose tools including functions, constants and types.

## Utility functions

We strive to be as type-stable, inferrable, and immutable as possible in order to improve robustness, performance, and correctness. These functions help us achieve these goals.

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

[`NestedClustered`](@ref) optimisations need to index the asset universe in order to produce the inner optimisations. These indexing operations are implemented as views, indexing, and custom index generators.

```@docs
port_opt_view(x, i, args...)
port_opt_view(x::VecScalar, i, args...)
port_opt_view(x::AbstractVector{<:Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}}, i, args...; kwargs...)
obs_weights_view(x, ::Any)
obs_weights_view(x::AbstractVector{<:Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}}, i)
```

## Summary statistics

Some estimators and constraints are based on summary statistics of vectors. These types are used to dispatch the appropriate functions and encapsulate auxiliary data such as weights.

```@docs
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
```
