---
status: accepted
---

# An undefined standard deviation makes `StandardisedValue` the mean

## Context

[`StandardisedValue`](../../src/02_Tools.jl) reduces a vector to its mean divided by its standard
deviation. The denominator carried one guard: an **exact** zero became `sqrt(eps)`. An
**undefined** denominator carried none.

A corrected standard deviation needs two values. On a one-value vector `Statistics.std` answers
`NaN`, so the reduction answered `NaN` too:

```julia
julia> PortfolioOptimisers.vec_to_real_measure(PortfolioOptimisers.StandardisedValue(), [0.37])
NaN
```

`SecondOrderDifference` reaches that vector. It is the default `alg` of `OptimalNumberClusters`,
so it is the library-wide rule for the number of clusters, and it applies its measure to **one
cluster at a time**. A cluster of exactly two assets carries one pairwise distance. The
within-cluster dispersion ``W_{c}`` was then `NaN` for every cut in which such a cluster appeared,
the whole two-difference gap series was `NaN`, and `valid_k_clusters` took its
`all(!isfinite, arr)` branch and returned the **length of the series** — a number that maximises
nothing. A partly-`NaN` series was worse: `argmax` returns the first `NaN`, so the `NaN` won.

A universe with a tightly correlated pair reaches this. Over 20 assets with two such pairs, every
cut from `2` to `6` clusters carries a two-asset cluster.

Issue #392 set out three options: change the default `alg` to `MeanValue()`, define the reduction
on a short vector, or refuse a `NaN` score. The first changes the selected `k` on every universe,
not only a degenerate one. The third turns a silent wrong answer into an error and leaves the
statistic undefined where the source defines it.

## Decision

**An undefined standard deviation is a denominator of one.** The guard now has three branches:

```math
\tilde{\sigma} = \begin{cases} 1 & \hat{\sigma} \ \mathrm{undefined} \\
                               \sqrt{\varepsilon} & \hat{\sigma} = 0 \\
                               \hat{\sigma} & \mathrm{otherwise} \end{cases}
```

A one-value vector therefore reduces to its own mean, which is the value itself. The reduction is
defined on every non-empty vector, and `StandardisedValue` agrees with `MeanValue` exactly where
the standardisation has nothing to divide by.

The default `alg` of `SecondOrderDifference` stays `StandardisedValue()`.

## Consequences

- A two-asset cluster contributes its single pairwise distance to ``W_{c}``. The gap series stays
  finite, and the selected `c` is a real maximiser rather than the fallback length.
- The selected number of clusters changes on any universe whose candidate cuts carry a two-asset
  cluster. It changes on no other universe: the guard fires only where the old value was `NaN`.
- The zero guard is untouched. `[2.0, 2.0, 2.0]` still gives `2 / sqrt(eps(Float64))`.
- Every other caller of the reduction gains the same definition on a one-value input — a
  one-asset correlation column, a one-entry centrality vector, a one-observation imputation
  window. Each answered `NaN` before.
- The two measures still differ on every cluster of three assets or more, so `alg = MeanValue()`
  remains the way to ask for the source's own ``W_{c}``.
