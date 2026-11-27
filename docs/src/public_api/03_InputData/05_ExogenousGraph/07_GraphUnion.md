```@meta
Description = "Graph union, public API of PortfolioOptimisers.jl: GraphUnion, graph_union, union_parts_in_force."
```

# Graph union

A union of exogenous graphs joins several relations into one graph before a consumer reads it, so
a path can change relation on its way. [`graph_union`](@ref) builds the union of graphs or series
that you hold, and you can store it under a name of its own in the returns data. A
[`GraphUnion`](@ref) in the `ng` field of a consumer states the same union by the names of its
parts, and the consumer builds it at the point of use. The rule for a part with no graph in force
is a member of [`AbstractUnknownEntryRule`](@ref), through [`union_parts_in_force`](@ref).

```@docs
GraphUnion
graph_union
PortfolioOptimisers.union_parts_in_force
```
