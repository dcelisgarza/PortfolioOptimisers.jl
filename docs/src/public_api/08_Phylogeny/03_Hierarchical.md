```@meta
Description = "Hierarchical, public API of PortfolioOptimisers.jl: ClusterNode, AbstractPreorderBy, PreorderTreeByID, get_node_property, is_leaf, pre_order, to_tree, …"
```

# Hierarchical

```@docs
ClusterNode
AbstractPreorderBy
PreorderTreeByID
get_node_property
is_leaf
pre_order
to_tree
optimal_number_clusters
clusterise(cle::ClustersEstimator{<:Any, <:Any, <:HClustAlgorithm, <:Any}, X::MatNum; branchorder::Symbol = :optimal, dims::Int = 1, kwargs...)
assignments
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
