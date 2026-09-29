---
status: accepted
---

# The chi-squared radius of a covariance set reads the free entries of a symmetric matrix

## Context

`ChiSqKUncertaintyAlgorithm` sizes the radius of an ellipsoid or of a norm ball as the square root
of a chi-squared quantile at level `1 - q`. It read the degrees of freedom off the side of the
shape: `size(sigma_X, 1)` in `k_ucs`, and `size(L, 2)` of the Cholesky factor in `norm_ball_set`.
That is `N` on the mean axis and `N²` on the covariance axis.

On the covariance axis the estimation error is a vectorised symmetric `N × N` matrix. The entries
`(i, j)` and `(j, i)` are one number, so the error spans `N(N + 1) / 2` coordinates, not `N²`. The
Normal shape `(I + K)(Σ ⊗ Σ) / T` has exactly that rank. The positive definite repair lifts its
`N(N − 1) / 2` zero eigenvalues, the antisymmetric directions, to about `1e-14` of the largest, so
the shape has full rank on paper and its side reads `N²`. The squared Mahalanobis distance of a
normal error is chi-squared at the rank of its shape, with an expected value equal to that rank.
So the full sets were too large for their stated level. At `N = 5` and `q = 0.05` the radius was
`6.136` against `5.000`, and the set covered `0.999` of the errors. At `N ≥ 20` it covered all of
them, and `q` had no effect. The ratio of the two radii falls to `1 / √2` as `N` increases.

The parity measure of map #1375 (#1390) found the defect, and #1425 records it. The oracle reads
`N²` on both of its covariance estimators, so parity did not reveal it. Three builders had it:
the Normal ellipsoid, the bootstrap ellipsoid, whose shape is the sample covariance of the
resampled covariances, and the Normal norm ball. The bootstrap norm ball already read `rank(L)`,
because its map holds the deviations themselves. The two routes of one set thus gave two radii at
one `q`. The docstring of `ChiSqKUncertaintyAlgorithm` recorded the `N²` rule as a conservative
extension of the source, which states the closed form for the mean axis only.

## Decision

The degrees of freedom are the **dimension of the set**. `ucs_dimension(class, diagonal, m)`
states the rule: `m` on the mean axis and on a diagonal shape, and `N(N + 1) / 2` on a full
covariance shape. `ellipsoidal_set` and `norm_ball_set` pass it to `k_ucs` and `k_norm_ball`, and
the deviation route keeps `rank(L)`.

A diagonal shape keeps `N²`. Its statistic `Σ_ij e_ij² / ω_ij` sums `N²` terms of unit mean, so its
expected value is `N²`. The terms are correlated, so the statistic is not chi-squared at any
degrees of freedom, and the radius is an approximation there on every rule.

`ChiSqKUncertaintyAlgorithm` takes `ambient::Bool = false`. `ambient = true` reads the side of the
shape, or the row count of the map, as before: `N²` on every covariance shape, the oracle's rule.
Its output stays one keyword away (ADR 0186). Under `ambient = true` the bootstrap norm ball reads
`N²` too, which no keyword reached before.

`k_ucs` takes the dimension as an optional fifth argument `df`, as `k_norm_ball` does. Its default
`size(sigma_X, 1)` keeps the four-argument call of a released version. `NormalKUncertaintyAlgorithm`
absorbs `df`, because its sampled distances carry the dimension themselves.

**A bootstrap ellipsoid caps the dimension at `M − 1`.** Its full shape is the sample covariance of
`M` resampled errors, so its rank is at most `M − 1`, and the repair makes it flat in the other
directions. `ucs_dimension(class, diagonal, m, M)` gives `min(d, M − 1)` on a full shape and `d` on
a diagonal one, and `bootstrap_ellipsoidal_set` passes it to `ellipsoidal_set` as `df`. The norm-ball
deviation route measures the same rank with `rank(L)`, so the two routes of one bootstrap set give
one radius. The cap binds when `M − 1 < N(N + 1) / 2`, which at the default of 3 000 resamples
starts at `N = 77`. It assumes a plain sample covariance, the default `ce`: a shrinkage estimator
gives a shape of full rank, and the cap then understates its dimension. `ellipsoidal_set` takes the
set algorithm in place of its diagonal switch and its radius algorithm, as `norm_ball_set` does, so
the new keyword adds no argument to the file.

**`LinearAlgebra.rank` of the shape is not the rule.** The repair lifts the zero eigenvalues to
between `1e-14` and `1e-12` of the largest, above the default tolerance, so `rank` of the repaired
Normal shape returns `N²` (25, 100 and 400 at `N` = 5, 10 and 20): the defect this ADR removes.
Before the repair it returns `N(N + 1) / 2`, which the closed form gives without an `O(N⁶)`
decomposition and a tolerance. `rank` also needs a singular value decomposition, which a `Rational`
or a dual number of automatic differentiation does not have. It is the rule only for a map that is
not repaired, the deviation route.

## Consequences

- The default radius of every full covariance set on the chi-squared rule falls, by `0.815` at
  `N = 5` and towards `0.707` as `N` increases. The default `diagonal = true` sets do not move.
  A robust portfolio on a full covariance set is less conservative, at the level the caller
  stated.
- The chi-squared radius and the empirical radius of `NormalKUncertaintyAlgorithm` now agree on a
  full covariance set, as they agree on the mean axis.
- A radius algorithm that a caller defines receives `df` as a fifth argument from
  `ellipsoidal_set`, and must take it.
- A bootstrap ellipsoid with fewer than `N(N + 1) / 2 + 1` resamples reads `M − 1` degrees of
  freedom, as its norm ball does. Its set is flat where the true error is not, so no radius makes it
  a confidence region of the whole error. More resamples is the remedy.
- `test/test_10_uncertainty_set.jl` pins the rule, the keyword, the agreement of the two radii, the
  cap, and the bootstrap ellipsoid on both rules, with the cap binding at 8 resamples.
  `test/test_10d_parity_uncertainty_sets.jl` pins the default as Better and the oracle's radius
  under `ambient = true` on both the Normal and the bootstrap norm ball.
