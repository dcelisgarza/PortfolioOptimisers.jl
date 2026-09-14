# Bubble Tree

The bubble tree is the first half of the seam behind [`DBHTs`](@ref): every function on this page
and on [DBHT Dendrogram](11_DBHTDendrogram.md) reads matrices alone, so each is driven directly
from a hand-built bubble structure in `test/test_13f_dbht_seam.jl`, and a wrong answer is
caught at its own step rather than diagnosed backwards from the final clustering. `Hb` is sparse
throughout, because [`DirectHb`](@ref) cuts an edge out of a copy of it.

```@docs
BubbleHierarchy
DirectHb
BubbleCluster8s
BubbleMember
```
