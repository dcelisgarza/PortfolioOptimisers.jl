---
status: accepted
---

# The default metric of a feature distance is the chord, because it is Euclidean

## Context

`FeatureDistance` measured with `AngularDist`, `acos(cos) / π`, by default ([ADR
0045](0045-a-feature-matrix-is-data-not-estimator-configuration.md) § 5). The angle is a metric,
but it is not Euclidean. The four feature vectors `(1, 0)`, `(0, 1)`, `(-1, 0)` and `(0, -1)` are
at `0.5` from each neighbour and at `1` across. For four points of a Euclidean space, the sum of
the squares of the four sides less the sum of the squares of the two diagonals is
`‖a - b + c - d‖² >= 0`. Here it is `4 · 0.25 - 2 = -1`.

The default linkage of `HClustAlgorithm` is `WardLinkage()`. The Ward update of Figure 2 of
`mullner2011` holds for a Euclidean distance between points. So the default feature distance fed
the default linkage a distance outside the domain of its update.

Issue #2163 measured the out-of-sample effect of the feature distances on two data sets, under
HRP, HERC, Schur and NCO. No feature kind gave a lower risk or a higher Sharpe ratio than the
returns-only distances on both data sets, and `AngularDist` against `CosineDist` showed no
consistent sign. The measurement cannot pick the default, so the math picks it. Issue #2166
decided this record.

## Decision

1. **`ChordDist`, `sqrt((1 - cos) / 2)`, is the default metric of `FeatureDistance`.** It is
   `‖ẑᵢ - ẑⱼ‖ / 2` for the unit feature vectors, so the points `ẑᵢ / 2` realise it, and it is
   Euclidean. It has the form of `SimpleDistance`, `sqrt((1 - ρ) / 2)`, on the returns path.
2. **`ChordSimilarity`, `1 - 2D²`, is its exact inverse** and its `default_similarity`. It goes
   negative at a negative cosine, so it is not a member of the non-negative family, and the PMFG
   path refuses it.
3. **`ChordDist` keeps the contract of `AngularDist`.** It takes the zero-vector convention, two
   zero vectors at `0` and a zero vector at `1` from every non-zero vector. It has a `gemm` matrix
   path through the `CosineDist` kernel with an exact zero diagonal, a `result_type` that keeps a
   float type, and no rescale under `StackObservations`, as a ratio metric.
4. **`AngularDist` and `AngularSimilarity` stay as options.** Their docs make no claim of a merit
   out of sample over `CosineDist`.
5. **The docs of `FeatureDistance`, of the three panel estimators and of the two feature examples
   state the result of #2163.** A fixed taxonomy lowers the turnover, and no feature kind lowered
   the risk.

## Consequences

- Every distance of a `FeatureDistance()` with no `metric` moves. The order of the pairs does not
  move, because both metrics fall as the cosine rises. A single or a complete linkage gives the
  same tree with other heights. A Ward or an average linkage can give another tree, and so other
  weights of a hierarchical optimiser.
- `FeatureDistance(; metric = AngularDist())` gives the v0.34 distances. The migration guide
  names the change.
- The graph distance of the oracle is `sqrt(1 - c)`, which is `sqrt(2)` times the chord. It gives
  the same trees and the same leaf order. Issue #2167 decides how the factor enters a parity form.
