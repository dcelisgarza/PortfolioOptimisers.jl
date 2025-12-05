# Tools

`PorfolioOptimisers.jl` is a complex codebase which uses a variety of general purpose tools including functions, constants and types.

## Utility functions

We strive to be as type-stable, inferrable, immutable as possible in order to improve robustness, performance, and correctness. These functions help us achieve these goals.

```@docs
traverse_concrete_subtypes
concrete_typed_array
factory(::Nothing, args...; kwargs...)
```

## Assertions

In order to increase correctness, robustness, and safety, we make extensive use of [defensive programming](https://en.wikipedia.org/wiki/Defensive_programming). The following functions perform some of these validations and are usually called at variable instantiation.

```@docs
assert_nonempty_nonneg_finite_val
assert_nonempty_gt0_finite_val
assert_nonempty_finite_val
assert_matrix_issquare
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

[`NestedClustered`](@ref) optimisations need to index the asset universe in order to produce the inner optimisations. These indexing operations are implemented as views and custom index generators.

```@docs
nothing_scalar_array_view
nothing_scalar_array_view_odd_order
nothing_scalar_array_getindex
nothing_scalar_array_getindex_odd_order
fourth_moment_index_generator
```
