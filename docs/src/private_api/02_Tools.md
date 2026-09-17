```@meta
Description = "Tools, private API of PortfolioOptimisers.jl: PROP_TAG_NAMES, PROP_TAG_MACRO_NAMES, PROP_TAG_CHANNELS, PROPAGATABLE_CONTRACTS, …"
```

# Tools: private API

## Utility functions

We strive to be as type-stable, inferrable, and immutable as possible in order to improve robustness, performance, and correctness. These functions help us achieve these goals.

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
vec_to_real_measure
```
