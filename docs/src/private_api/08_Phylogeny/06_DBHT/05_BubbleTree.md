```@meta
Description = "Bubble Tree, private API of PortfolioOptimisers.jl: BubbleHierarchy, DirectHb, BubbleCluster8s, BubbleMember."
```

# Bubble Tree: private API

The functions on this page and on the [DBHT dendrogram](06_DBHTDendrogram.md) page are two parts
of [`DBHTs`](@ref). The functions here build the bubble tree of the planar graph and assign each
asset to a cluster. The functions on the dendrogram page turn the clusters into a hierarchy. Each
function takes and returns matrices, so `test/test_13f_dbht_seam.jl` tests each one on its own,
from a bubble structure built by hand. The functions store the bubble hierarchy `Hb` as a sparse
matrix, because [`DirectHb`](@ref) removes an edge from a copy of it.

```@docs
BubbleHierarchy
DirectHb
BubbleCluster8s
BubbleMember
```
