# Tools

`PorfolioOptimisers.jl` is a complex codebase which uses a variety of general purpose functions, constants and types.

## Assertions

## Public

```@docs
VecScalar
brinson_attribution
traverse_concrete_subtypes
concrete_typed_array
factory(::Nothing, args...; kwargs...)
```

## Private

```@docs
Num_VecNum_VecScalar
Num_ArrNum_VecScalar
assert_nonempty_nonneg_finite_val
assert_nonempty_finite_val
assert_nonempty_geq0_finite_val
assert_matrix_issquare
:⊗
:⊙
:⊘
:⊕
:⊖
dot_scalar
nothing_scalar_array_view
nothing_scalar_array_view_odd_order
nothing_scalar_array_getindex
nothing_scalar_array_getindex_odd_order
nothing_asset_sets_view
fourth_moment_index_generator
```
