---
status: accepted
---

# A regime statistic divides by the bias that its method reads

## Context

Every regime statistic of `RegimeAdjustedExpWeightedVariance` and
`RegimeAdjustedExpWeightedCovariance` divides a realised square by an **estimated** variance. The
inverse of an estimate is too large on average (Jensen's inequality), so under a correctly
calibrated model the statistic is biased up, and the multiplier reads the covariance too large.
The fix of #1415 corrected the Mahalanobis target, whose bias grows with `n / hl`. Issue #1428
found the same bias at `n = 1` in the scalar estimator, in `DiagonalTarget` and in
`PortfolioTarget`: 1.07 at a half-life of 10 and 1.017 at a half-life of 40, and 1.7 in the
warm-up.

At `n = 1` the estimate is `v̂ = σ² Q`, with `Q = Σ_j w_j z_j²` on the normalised exponential
weights. The three regime methods read three moments of `Q`, not one:

| Method | The squared multiplier reads | Factor |
| --- | --- | --- |
| `RootMeanSquaredAdjusted` | `E[z²]` | `E[1/Q]` |
| `FirstMomentRegimeAdjusted` | `E[abs(z)]²` | `E[Q^(-1/2)]²` |
| `LogRegimeAdjusted` | `exp(E[ln z²])` | `exp(-E[ln Q])` |

At a half-life of 10 the three are 1.071, 1.053 and 1.035. The fixed point `b(w, n)` of #1415 is
the first. So the division by `b(w, 1)` that #1428 proposed over-corrects the default method.
The Laplace transform `E[exp(-tQ)] = Π_j (1 + 2 w_j t)^(-1/2)` gives each moment as one integral,
and the simulated estimators match each moment within the noise.

## Decision

**`debias::Bool = true` is a field of both estimators**, not of a target. The bias is a property
of the estimate. `DiagonalTarget` is a field-less singleton that `GeodesicShrinkageCovariance` also
reads, and the scalar estimator has no target. The field that #1415 put on `MahalanobisTarget` moves
to the estimator. That commit had not reached `main`. `debias = false` is the raw statistic, bit
for bit, and the oracle's rule of map #1375, one keyword away (ADR 0186).

**A statistic of one direction divides by the exact moment that its method reads.**
`regime_bias_table(method, decay, K)` makes the factor for every count of observations in one pass:
`G_{K+1}(s) = G_K(s) (1 + λ^K s)^(-1/2)` on a fixed grid of `s = e^x`, and a trapezoid rule on it.
At equal weights the table agrees with `K / (K − 2)`, the ratio of gamma functions and the digamma
form. The state holds the table and grows it to twice the largest count it meets. The table stops
at the count where `λ^K` falls below the machine epsilon. A fit at a half-life of 40 thus makes at
most 2 100 entries, in about 0.05 s. The scalar estimator divides each `z²` by the factor at the
count of its asset. `PortfolioTarget` divides each direction by the factor at the smallest count of
the contributing assets.

**`DiagonalTarget` divides each term by `E[1/Q_i]` at its own count, whatever the method.** A sum
of `n` terms averages the errors of `n` estimates, so the root and the log see almost the mean's
factor. At 12 assets and a half-life of 10 the biases are 1.073, 1.068 and 1.063 at a correlation of
0.3. The factor of each method alone is exact only at one asset or a correlation of one. The mean's
factor keeps `E[S] = n` exactly at every correlation. That calibration is the only one that the
target's docstring states as exact.

**The gate is `K > n + 3`, where the variance of the statistic is finite.** At `n = 1` an estimate
of four observations or fewer is not scored. The gate reads the count of each asset, so a young
asset leaves the statistic as it does below `min_obs`.

## Consequences

- Every default fit of a regime estimator moves, and so does every default fit of the
  Cross-Sectional Factor Prior, whose `ve` and `pe` are regime estimators. A stored oracle case
  states `debias = false`.
- **Four limits remain, and four child issues of map #1375 hold them.**
  - The default `PortfolioTarget()` builds its inverse-volatility direction from the estimate
    itself. The direction and the error of the estimate are correlated, so a bias that no factor of
    `(λ, K)` removes remains: 1.058 at a half-life of 10 (#1430).
  - `MahalanobisTarget` keeps `b`, the moment of the mean, for every method, so it over-corrects
    FirstMoment and Log: 0.975 and 0.956 at 12 assets and a half-life of 10. The root and the log of
    `v' Ŵ⁻¹ v` for `n > 1` have no one-dimensional integral (#1431).
  - The FirstMoment and Log calibrations of `DiagonalTarget` assume a `χ²(n)` sum. Before this
    decision the Jensen bias hid part of that error; after it, correlated assets read 0.941 and 0.962
    (#1432).
  - The factor assumes the plain exponential weights, and a HAC estimate has others. At two lags it
    over-corrects the scalar estimator (0.898) and under-corrects the covariance targets (#1433).

## Alternatives rejected

- **`b(w, 1)` for every method**, as #1428 first proposed. It is the moment of
  `RootMeanSquaredAdjusted` alone.
- **A moment-matched chi-square on the Kish count.** It has closed forms, but it is 1.5 % wrong at
  `K = 5`. The exact table costs little.
- **A keyword on each target.** `DiagonalTarget` would carry a field that a geodesic shrinkage
  ignores, and the scalar estimator would need a keyword of its own anyway.
