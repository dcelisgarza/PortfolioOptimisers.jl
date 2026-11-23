```@meta
Description = "Graph observations, public API of PortfolioOptimisers.jl: graph_in_force."
```

# Graph observations

## Read the graph of a window

A consumer of a graph of the returns data reads the graph in force at the last row of its window. A view of
the returns data on some of its rows, such as a cross-validation fold, keeps the graph in force at
its first row and the change points inside its window, so a training fold holds no graph of its
test window.

```@docs
graph_in_force
```
