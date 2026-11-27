```@meta
Description = "Edge rules, public API of PortfolioOptimisers.jl: AbstractEdgeDirection, OutEdges, InEdges, BothEdges, AnyEdges, walk_parts, exposure_parts, …"
```

# Edge rules

An edge rule says how a consumer of an exogenous graph reads its edges. A direction says whether
the consumer reads the out-edges of a node, its in-edges, both apart, or the undirected graph. A
normalisation says whether it reads the stored weights or the share of each edge in the total
weight of its node. The walk, the union of graphs and the dependency exposure read both rules.

```@docs
PortfolioOptimisers.AbstractEdgeDirection
OutEdges
InEdges
BothEdges
AnyEdges
PortfolioOptimisers.walk_parts
PortfolioOptimisers.exposure_parts
PortfolioOptimisers.AbstractEdgeNormalisation
WeightShare
PortfolioOptimisers.normalise_edge_weights
```
