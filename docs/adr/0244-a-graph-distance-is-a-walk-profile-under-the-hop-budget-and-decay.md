---
status: accepted
---

# A graph distance is a Walk Profile under the hop budget and decay

## Context

ADR 0242 put named Exogenous Graphs on the returns data. ADR 0243 kept `FeatureDistance` and
made a graph distance a producer in its `ape` slot. The oracle measures a graph distance from
the cosine of the dependency profiles of the assets: row `i` of `Σ_{h=1}^{H} α^{h-1} M^h`, with
`M` the adjacency after an optional normalisation of the rows, and `sqrt(1 - c)` as the
distance. It refuses after an overflow, loses an underflow with no error, and writes `NaN` for
an asset that is not a node or that has no out-edge, which makes HERC raise. Issue #2167
decided this record, and issue #2172 built it.

## Decision

1. **`WalkProfilePanel <: AbstractAssetPanelEstimator`** builds one tensor Panel Field for
   each graph and each single direction, `"<prefix>_<dir>_<graph>"` on the axis `"node"`, with
   `prefix = nothing` for `"walk"`. The profile of asset `i` is column `i` of
   `Σ_h w(h) M̃^h`, scaled to unit length, with `w(h) = separation_decay(decay, h, n)`. It reads
   the graph in force at the last row of its window.
2. **The budget and the decay are the types of the library.** `sep::HopCount` with
   `CountOf(2)` and `decay = ExponentialDecay(; rate = log(2))` give the oracle's defaults,
   because `α^h` is `α^{h-1}` times one factor. `decay = nothing` is `α = 1`, and `HopCount(1)`
   is `α = 0`. A `FractionOf` takes its quantile over the hop separations of the pairs of an
   asset and a node that the walk direction reaches, and a rule receives the producer and the
   graph of the steps. A `PathLength` fails at dispatch, because a matrix power counts edges.
3. **`dir`**: `OutEdges()` (the oracle), `InEdges()` (the transpose), `BothEdges()` (the two
   as two fields). **`self::Bool = false`** adds `w(0)` at the node of the asset.
   **`nrm = WeightShare()`** divides each weight by its node's total along the direction, also
   at one hop. **`wgt = UnitEdgeWeights()`** reads each edge as one, at the point of use.
4. **The walk on `nrm = nothing` is scaled**: each hop vector is kept divided by its largest
   entry with its log scale apart, so it cannot overflow and is exact to rounding. Under
   `WeightShare` the total of each distribution is at most one, and the plain walk is used.
5. **The element type is derived**: `promote_type(eltype(At), eltype(ws))`.
6. **`readable_assets(ape, rd)`** is a hook of the producer interface, with the fallback
   `nothing`. `feature_readable_mask` composes it for a `FeatureDistance` with a producer
   before `investable_reduction`. An asset that is not a node of a graph has a placeholder in
   the fields of that graph. An asset whose profile is zero in every field has a placeholder in
   every field. Either is unreadable: a fit drops it with the `@info` of the reduction, and a
   direct call refuses it by name.
7. **`OrthantChordDist`, `sqrt(1 - c)` with `1 - c` clamped to `[0, 1]`, and its exact inverse
   `OrthantChordSimilarity`, `1 - D²`**, sit beside `ChordDist`, by the precedent of
   `SimpleDistance` and `CorrelationDistance`. On the non-negative orthant the longest chord is
   `sqrt(2)`, so the metric uses `[0, 1]`. `OrthantChordSimilarity` is non-negative for
   `D <= 1`, so it joins the non-negative family, with that bound in `assert_similarity_domain`.
8. **`WalkDistance(; ng, sep, decay, dir, nrm, wgt, self, metric = OrthantChordDist())`** in
   `src/23_Aliases.jl` returns the `FeatureDistance`. It does not forward `prefix`.
9. **`graph_requirements(::WalkProfilePanel) = (NonNegativeWeights(),)`**, and
   `graph_names` is `ng`.

## Consequences

- The parity test `test/test_13i_walk_distance.jl` measures D to 2.9e-15 and S to 5.6e-16
  against the oracle on 56 cases, the unreadable set equal to the oracle's `NaN` set, Float32 at
  1.3e-7, and HERC to 5.6e-17 through a fixed cut.
- Where the oracle raises on an asset with no out-edge, HERC drops it and gives it a zero
  weight. Where the oracle refuses an overflow, the scaled walk agrees with `BigFloat` to
  2e-15.
- `readable_assets` reads every field of the producer, not only the fields that `sel` picks.
  A fit can then drop an asset that a `sel` of one graph could read.
- `RegressionPanel` and `PhylogenyPanel` answer `nothing`, so their fits do not change.
