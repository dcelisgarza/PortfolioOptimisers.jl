```@meta
Description = "DBHT Dendrogram, private API of PortfolioOptimisers.jl: DendroConstruct, LinkageFunction, build_link_and_dendro, HierarchyConstruct4s, …"
```

# [DBHT Dendrogram: private API](@id private-api-dbht-dendrogram)

The functions on this page turn the clusters of [`DBHTs`](@ref) into a hierarchy, after the
functions on the [bubble tree](@ref private-api-bubble-tree) page assign the assets to clusters.
[`HierarchyConstruct4s`](@ref) takes the clusters and the bubble membership, and returns a linkage
matrix in the Matlab format. [`turn_into_Hclust_merges`](@ref) converts that matrix into the format
of `Clustering.Hclust`, and it is the only function here that uses that format.

```@docs
DendroConstruct
LinkageFunction
build_link_and_dendro
HierarchyConstruct4s
turn_into_Hclust_merges
```
