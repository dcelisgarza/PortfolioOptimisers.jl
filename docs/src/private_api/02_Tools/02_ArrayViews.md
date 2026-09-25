```@meta
Description = "Array views, private API of PortfolioOptimisers.jl: get_window, nothing_scalar_array_view, nothing_scalar_array_view_odd_order, …"
```

# Array views: private API

## Utility functions

Most functions below serve the macros that declare the types of the library. A field tagged [`@pprop`](@ref) or `@cprop` takes its value from the prior or from the optimiser when the caller leaves it empty, and [`sel`](@ref) makes that choice. [`@propagatable`](@ref) records the tagged fields of each type it declares, and [`@forward_properties`](@ref) gives a type properties that read through to the fields it holds. The rest are helpers that many files share, such as [`get_window`](@ref), which gives the index range of an observation window.

```@docs
get_window
```

## View functions

An optimisation over a subset of the assets, such as each inner optimisation of [`NestedClustered`](@ref), needs the part of every input that belongs to those assets. The functions below take that part of a vector, a matrix or a higher moment, and return `nothing` or a scalar unchanged. [`fourth_moment_index_generator`](@ref) gives the indices that select the entries of the subset from a fourth-moment matrix. [`view_child`](@ref) views one field of an estimator for [`port_opt_view`](@ref), and keeps a precomputed optimisation result, such as a fallback, as it is.

```@docs
nothing_scalar_array_view
nothing_scalar_array_view_odd_order
nothing_scalar_array_getindex
nothing_scalar_array_getindex_odd_order
fourth_moment_index_generator
view_child
```
