```@meta
Description = "Edge rules, public API of PortfolioOptimisers.jl: AbstractEdgeDirection, OutEdges, InEdges, BothEdges, walk_parts, AbstractEdgeNormalisation, WeightShare, …"
```

# Edge rules

An edge rule says how a consumer of an exogenous graph reads its edges. A direction says whether
the consumer reads the out-edges of a node, its in-edges, or both apart. A normalisation says
whether it reads the stored weights or the share of each edge in the total weight of its node.
The walk and the union of graphs read both rules.

```@docs
PortfolioOptimisers.AbstractEdgeDirection
OutEdges
InEdges
BothEdges
PortfolioOptimisers.walk_parts
PortfolioOptimisers.AbstractEdgeNormalisation
WeightShare
PortfolioOptimisers.normalise_edge_weights
```
