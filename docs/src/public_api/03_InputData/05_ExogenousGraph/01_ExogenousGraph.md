```@meta
Description = "Exogenous graph, public API of PortfolioOptimisers.jl: ExogenousGraph, adjacency, Base.isequal, Graphs.SimpleDiGraph."
```

# Exogenous graph

## Types

An [`ExogenousGraph`](@ref) holds one directed, weighted graph over named nodes, such as the
suppliers of each asset. The nodes can be assets, entities that are not assets, or labels. The
adjacency `At` holds the sources in its columns, and [`adjacency`](@ref) gives the matrix with the
sources in its rows, with no copy. The returns data holds the graphs by name in `ng` and `G`.

```@docs
ExogenousGraph
```

## Functions

```@docs
adjacency
Base.isequal(::ExogenousGraph, ::ExogenousGraph)
PortfolioOptimisers.Graphs.SimpleDiGraph(::ExogenousGraph)
```
