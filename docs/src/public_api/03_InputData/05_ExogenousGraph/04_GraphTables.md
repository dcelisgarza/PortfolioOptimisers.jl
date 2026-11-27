```@meta
Description = "Graph tables, public API of PortfolioOptimisers.jl: graph_dataframe, graph_manifest, exogenous_graphs."
```

# Graph tables

## Write the graphs to tables and read them back

`graph_dataframe` writes one row for each edge of each snapshot of the named graphs of the returns data.
`graph_manifest` writes what the edges cannot show: the order of the nodes, a node with no edge, a
`nothing` key, a static graph and the storage of each adjacency. `exogenous_graphs` reads the
`ng` and the `G` of the returns data back from the two tables, or from the edge table alone.

```@docs
graph_dataframe
graph_manifest
exogenous_graphs
```
