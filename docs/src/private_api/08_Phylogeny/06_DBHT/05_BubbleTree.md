```@meta
Description = "Bubble Tree, private API of PortfolioOptimisers.jl: BubbleHierarchy, DirectHb, BubbleCluster8s, BubbleMember."
```

# [Bubble Tree: private API](@id private-api-bubble-tree)

The functions on this page and on the [DBHT dendrogram](@ref private-api-dbht-dendrogram) page are two parts
of [`DBHTs`](@ref). The functions here build the bubble tree of the planar graph and assign each
asset to a cluster. The functions on the dendrogram page turn the clusters into a hierarchy. Each
function takes and returns matrices, so you can call each one on its own. The bubble hierarchy
`Hb` is a sparse matrix.

```@docs
BubbleHierarchy
DirectHb
BubbleCluster8s
BubbleMember
```
