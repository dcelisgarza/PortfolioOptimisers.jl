# DBHT Dendrogram

The dendrogram is the second half of the seam behind [`DBHTs`](@ref), after
[Bubble Tree](10_BubbleTree.md). [`HierarchyConstruct4s`](@ref) takes the clusters and the bubble
membership and returns a Matlab-style linkage matrix, and [`turn_into_Hclust_merges`](@ref) is the
one step that speaks the `Clustering.Hclust` convention. Each function is driven directly, from
inputs small enough to work by hand, in `test/test_13f_dbht_seam.jl`.

```@docs
DendroConstruct
LinkageFunction
build_link_and_dendro
HierarchyConstruct4s
turn_into_Hclust_merges
```
