---
status: accepted
---

# Recursive bisection owns its branch order

## Context

Six clustering optimisers open with the same prologue: reset the time-dependent estimator, pick
the returns, fit the prior, clusterise. Four of them declare a `branchorder` keyword and forward
it to `clusterise`. Two do not:
`HierarchicalRiskParity` and `SchurComplementHierarchicalRiskParity` call `clusterise` with no
`branchorder`, so a caller's `branchorder = :barjoseph` lands in `kwargs...` and is dropped.

The maintainability review of 2026-08-16 recorded that drop as finding 1, a copy that had
drifted. **That reading is wrong.** The two families consume different parts of the same
clustering result, and the difference is load-bearing.

`branchorder` controls the **leaf permutation** of the dendrogram. It never changes the merge
tree, and it never changes the cluster memberships. `:optimal` runs a seriation that puts
similar leaves next to each other. `:barjoseph` and `:r` are cheaper and give a worse
permutation.

The two families read the clustering result in two different ways.

| Optimiser | What it reads | Sensitive to the leaf order |
| --- | --- | --- |
| `HierarchicalRiskParity` | `clr.res.order` | **Yes** |
| `SchurComplementHierarchicalRiskParity` | `clr.res.order` | **Yes** |
| `HierarchicalEqualRiskContribution` | `cutree` memberships, `to_tree(clr.res)` | No |
| `NestedClustered` | `cutree` memberships | No |

HRP and Schur run López de Prado's **recursive bisection**. The algorithm starts from
`items = [clr.res.order]` and splits that one permutation in half, again and again. The split
point is the midpoint of the vector. So the permutation *is* the algorithm's input: it decides
which assets fall on the same side of every cut. A bad permutation bisects between two similar
assets and allocates as if they were unrelated. The quasi-diagonalisation step is the whole
reason the method works, and `:optimal` is the ordering that gives it.

HERC and Nested Clustered never bisect a permutation. They cut the tree at `clr.k` and read the
resulting memberships, which are invariant under any branch order. The leaf permutation only
affects how a dendrogram is drawn, so a caller can trade seriation cost for speed with no change
to the weights.

## Decision

**A clustering optimiser declares `branchorder` only if the leaf permutation is optional for its
allocation. Recursive bisection needs the optimal ordering, so it fixes `branchorder` internally
and does not offer the choice.**

Two consequences follow.

1. `HierarchicalRiskParity` and `SchurComplementHierarchicalRiskParity` do not declare
   `branchorder` and must not gain it. They call `clusterise` without the keyword and take its
   `:optimal` default. The swallow by `kwargs...` is the deliberate result, not a leak.
2. `HierarchicalEqualRiskContribution`, `NestedClustered`, `Stacking` and `SubsetResampling`
   declare `branchorder` and forward it, because for them the ordering is a cost dial.

The prologue can still be consolidated. A shared helper must keep the split, not erase it:
the keyword belongs to the four optimisers that can honour it, and the bisection pair must
continue to pin `:optimal`.

## Consequences

- Finding 1 of the maintainability review of 2026-08-16 is **withdrawn**. Its "direction" — one
  `clustering_preamble(x, rd; dims, branchorder, kwargs...)` that declares the keyword once for
  all six — would let a caller degrade an HRP or a Schur allocation. Do not apply it as written.
- A caller who passes `branchorder` to HRP or Schur still gets no diagnostic. That is the
  ordinary Julia behaviour for an unknown keyword absorbed by `kwargs...`, and it is shared with
  every other keyword the clustering optimisers do not read. A general diagnostic for unread
  keywords is a separate question from this one.
- If a future variant of recursive bisection splits on the merge tree instead of the leaf
  permutation, this decision does not cover it. Amend this ADR in that case.

## References

- `src/20_Optimisation/05_HierarchicalRiskParity.jl` — `items = [clr.res.order]`.
- `src/20_Optimisation/06_SchurComplementHierarchicalRiskParity.jl` — the same two sites.
- `src/20_Optimisation/07_HierarchicalEqualRiskContribution.jl` — `cutree` memberships and
  `to_tree`.
- `src/11_Phylogeny/03_Hierarchical.jl` — `branchorder` reaches `Clustering.hclust`.
- `docs/reports/maintainability-review-20260816.html` — finding 1, withdrawn.

## Amendment (2026-09-09)

`:optimal` is not a single algorithm this library owns; it is
`Clustering.orderbranches_barjoseph!`, a routine the `Clustering.jl` dependency ships. #947
measured that this library's leaf order and a reference implementation's disagree on the same
dendrogram — same distance matrix, same linkage heights, same merge tree, different branch
order — because each side calls its own package's optimal-leaf-ordering routine, and the two
routines do not attain the same objective value on every fixture. Substituting the reference's
order into this library's HRP reproduces the reference's weights exactly, so nothing in this
library's HRP, its distance, or its linkage is implicated; the divergence is entirely in the
third-party ordering step.

This ADR's decision stands: recursive bisection still fixes `branchorder` internally rather than
exposing it, because the leaf permutation is still load-bearing for HRP and Schur. What this
amendment adds is the scope of that guarantee: **reproducibility of the leaf order, and therefore
of an HRP or Schur allocation, holds within this library's own dependency versions, not across
implementations.** Pinning the ordering to a routine this library owns and tests, so that the
allocation stops moving with `Clustering.jl`'s version or with a port to another language, is a
breaking numerical change to every hierarchical caller and is not taken here. A future ADR may
take it; until then, a cross-language HRP or Schur parity test should hold the leaf order equal
rather than assert on it.

### References (amendment)

- Issue #947 — the reproduction, the per-fold leaf orders, and the brute-force objective scores
  that show neither routine attains the minimum on every fold.
- `test/test_57b_reference_walk_forward_parity.jl` — the walk-forward parity that holds the leaf
  order equal and passes to `5.6e-17`.
