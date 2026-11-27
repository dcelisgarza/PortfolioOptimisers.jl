```@meta
Description = "Exogenous graph series, public API of PortfolioOptimisers.jl: ExogenousGraphSeries, Base.isequal."
```

# Exogenous graph series

## Types

An [`ExogenousGraphSeries`](@ref) is a list of change points over the observation axis. The
graph at a key is in force from that key to the next key, and a `nothing` key ends the graph
before it. A key is a 1-based row position or a timestamp.

```@docs
ExogenousGraphSeries
```

## Functions

```@docs
Base.isequal(::ExogenousGraphSeries, ::ExogenousGraphSeries)
```
