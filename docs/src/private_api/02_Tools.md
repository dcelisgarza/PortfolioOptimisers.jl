```@meta
Description = "Tools, private API of PortfolioOptimisers.jl: PROP_TAG_NAMES, PROP_TAG_MACRO_NAMES, PROP_TAG_CHANNELS, PROPAGATABLE_CONTRACTS, …"
```

# Tools: private API

## Utility functions

Most functions below serve the macros that declare the types of the library. A field tagged [`@pprop`](@ref) or `@cprop` takes its value from the prior or from the optimiser when the caller leaves it empty, and [`sel`](@ref) makes that choice. [`@propagatable`](@ref) records the tagged fields of each type it declares, and [`@forward_properties`](@ref) gives a type properties that read through to the fields it holds. The rest are helpers that many files share, such as [`get_window`](@ref), which gives the index range of an observation window.

```@docs
PROP_TAG_NAMES
PROP_TAG_MACRO_NAMES
PROP_TAG_CHANNELS
PROPAGATABLE_CONTRACTS
concrete_typed_array_if_abstract
get_window
prop_tag
is_prop_tag_call
prop_tag_expr
prop_channel_active
prop_channel_pairs
check_prop_tag_macros
is_doc_macro
_ctx
_wprop
resolve_deferred_quantities
sel
extract_field_name
propagatable_find_struct
propagatable_bare_name
try_field_name
peel_prop_tags
propagatable_parse_body
propagatable_register!
propagatable_keywords
propagatable_contract_violations
check_propagatable_contracts
forward_nonnothing
forward_flatten_path
forward_walk_expr
```

## Mathematical functions

`⊙`, `⊘`, `⊕` and `⊖` multiply, divide, add and subtract element by element, and each accepts a scalar or an array on either side. `⊗` gives the outer product of two arrays as a matrix. [`dot_scalar`](@ref) takes the dot product of two vectors. When one side is a scalar, it treats that scalar as a vector of equal entries, and the scalar can be a `JuMP` expression.

```@docs
:⊗
:⊙
:⊘
:⊕
:⊖
dot_scalar
```

## View functions

An optimisation over a subset of the assets, such as each inner optimisation of [`NestedClustered`](@ref), needs the part of every input that belongs to those assets. The functions below take that part of a vector, a matrix or a higher moment, and return `nothing` or a scalar unchanged. [`fourth_moment_index_generator`](@ref) gives the indices that select the entries of the subset from a fourth-moment matrix.

```@docs
nothing_scalar_array_view
nothing_scalar_array_view_odd_order
nothing_scalar_array_getindex
nothing_scalar_array_getindex_odd_order
fourth_moment_index_generator
```

## Summary statistics

Some estimators and constraints reduce a vector to one number. A field of type [`Num_VecToScaM`](@ref) can hold a fixed number, a `VectorToScalarMeasure`, or a function that does the reduction.

```@docs
Num_VecToScaM
```
