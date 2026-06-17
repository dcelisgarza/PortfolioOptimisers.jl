# Tools

`PorfolioOptimisers.jl` is a complex codebase which uses a variety of general purpose tools including functions, constants and types.

## Utility functions

We strive to be as type-stable, inferrable, and immutable as possible in order to improve robustness, performance, and correctness. These functions help us achieve these goals.

```@docs
traverse_concrete_subtypes
concrete_typed_array
factory(a::Union{Nothing, <:AbstractEstimator, <:AbstractAlgorithm, <:AbstractResult}, args...; kwargs...)
get_window
@propagatable
factory_child
@fprop
@vprop
@pprop
@wprop
@cprop
_is_prop_tag_call
_is_fprop_macro
_is_vprop_macro
_is_pprop_macro
_is_wprop_macro
_is_cprop_macro
_is_doc_macro
_ctx
_wprop
sel
_extract_field_name
_propagatable_find_struct
_propagatable_bare_name
_try_field_name
_peel_prop_tags
_propagatable_parse_body
@forward_properties
_forward_nonnothing
_forward_flatten_path
_forward_walk_expr
```

## Mathematical functions

`PortfolioOptimisers.jl` makes use of various mathematical operators, some of which are generic to support the variety of inputs supported by the library.

```@docs
:⊗
:⊙
:⊘
:⊕
:⊖
dot_scalar
```

## View functions

[`NestedClustered`](@ref) optimisations need to index the asset universe in order to produce the inner optimisations. These indexing operations are implemented as views, indexing, and custom index generators.

```@docs
port_opt_view(x, i, args...)
port_opt_view(x::VecScalar, i, args...)
nothing_scalar_array_view
nothing_scalar_array_view_odd_order
nothing_scalar_array_getindex
nothing_scalar_array_getindex_odd_order
fourth_moment_index_generator
```

## Summary statistics

Some estimators and constraints are based on summary statistics of vectors. These types are used to dispatch the appropriate functions and encapsulate auxiliary data such as weights.

```@docs
VectorToScalarMeasure
Num_VecToScaM
MinValue
MeanValue
factory(mv::MeanValue, args...; kwargs...)
MedianValue
factory(mdv::MedianValue, args...; kwargs...)
MaxValue
StdValue
factory(sv::StdValue, args...; kwargs...)
VarValue
factory(vv::VarValue, args...; kwargs...)
SumValue
ProdValue
ModeValue
StandardisedValue
factory(msv::StandardisedValue, args...; kwargs...)
vec_to_real_measure
```
