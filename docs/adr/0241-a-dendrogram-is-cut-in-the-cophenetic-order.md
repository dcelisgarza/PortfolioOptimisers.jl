---
status: accepted
---

# A dendrogram is cut in the cophenetic order, and the library owns the linkages that Clustering.jl lacks

## Context

`HClustAlgorithm` gave a `Symbol` to `Clustering.hclust`, which has the single, complete, average
and Ward linkages. SciPy and R also have WPGMA ("weighted"), centroid (UPGMC) and median (WPGMC).
Clustering.jl has no hook for a new linkage: `hclust` hard-codes its five methods, and its
nearest-neighbour chain fits only a reducible linkage and clamps away an inversion.

The centroid and median linkages are not reducible, so a merge can be lower than a merge below it,
an inversion (Section 2.4 of `mullner2011`). On correlation distances of random returns, the
centroid tree has an inversion in 97 % or more of the cases at 7 assets, and in all the cases at
10 assets or more. `Clustering.cutree` replays the merges in the order of `sortperm(heights)`, so
on an inversion it replays a parent before its child and throws a `BoundsError`. The library read
the same order in `validate_k_value`, `valid_k_clusters`, the Integer `optimal_number_clusters`,
HERC and the R branch order.

The oracle cuts with `fcluster(Z, k, "maxclust")`, which gives fewer than `k` clusters on an
inverted tree, and its number of clusters reads `cut_tree`, which orders the nodes by raw height
and garbles the partition. Issue #2170 measured both, and decided this record.

## Decision

1. **A linkage is a member of `AbstractLinkage`**, a singleton family with an `# Interfaces`
   section: `SingleLinkage`, `CompleteLinkage`, `AverageLinkage`, `WardLinkage(; presquared)`,
   `WeightedLinkage`, `CentroidLinkage` and `MedianLinkage`. A `Symbol` fails at construction.
2. **The four linkages of Clustering.jl stay in `Clustering.hclust`.** Each member converts to its
   `Symbol` at the call.
3. **The library owns a Lance-Williams kernel for the other three**, a port of SciPy:
   `nn_chain_merges` (the nearest-neighbour chain of `nn_chain`) for WPGMA, and
   `generic_linkage_merges` (the generic algorithm of `fast_linkage`, with its binary heap) for
   centroid and median. The merges, their order and the rule for a tie are those of SciPy.
   `res.heights` keeps the raw heights, at parity with `Z[:, 2]`.
4. **Every reader of a dendrogram that needs a monotone tree reads the cophenetic order.** The
   cophenetic height of a merge is the largest merge height in its subtree. The merges sorted by
   it, a tie to the earlier merge, put each parent after its children. `cut_dendrogram(res, k)`
   does the first `N - k` merges of that order, so it always gives `k` clusters, and on a tree with
   no inversion it gives the labels of `Clustering.cutree`. `assignments`, the number of clusters,
   HERC and the R branch order read it.
5. **The tie rule of `validate_k_value` stays.** `tall_first_nodes` orders the nodes by their
   cophenetic height, the earlier node first in a tie of equal heights, and a merge above an
   inversion before the merges of its tie. So a tie that spans a cut still rejects it, and an
   inversion alone rejects no cut.

## Consequences

- A tree with no inversion gives the same tree, the same cuts and the same weights as before.
- `Clustering.cutree` on a centroid or median tree throws. The migration guide names
  `cut_dendrogram` as its replacement.
- A parity test marks the oracle's `fcluster` and `cut_tree` on an inverted tree `@test_broken`.
- The number of clusters of the library has its own rule, which differs from the oracle's rule.
  Issue #2239 asks whether a member of ours gives the oracle's count.
