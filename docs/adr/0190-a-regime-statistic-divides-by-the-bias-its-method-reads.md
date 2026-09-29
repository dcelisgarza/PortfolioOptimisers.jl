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

**The default inverse-volatility direction also divides by `1 + Δ`, its second-order excess
(#1430).** `PortfolioTarget()` builds its direction from the estimate that it divides by, so the
direction and the error of the estimate are correlated, and the factor of a fixed direction leaves
1.058 at 12 assets and a half-life of 10. In units of the true volatilities the statistic is
`q'Rq / 1'R̂1`, with `q_i = σ_i / σ̂_i`. A second-order expansion in the error of the estimate gives
`Δ = s_v (1 + Σ₃/A − 2B/A²) + s_vc (1 − Σ₃/A − 2T/A² + 2B/A²)`, with `c = R1`, `A = 1'c`,
`T = Σ c_i³`, `B = Σ c_i R_ij² c_j`, `Σ₃ = Σ R_ij³`, `s_v` the sum of the squared weights of the
variance and `s_vc` the sum of the products of the variance and the correlation weights. On one
decay it is `2 s₂ (1 − T/A²)`. The excess is the same for the three methods to first order, and
`inverse_volatility_bias` reads it on the estimated correlation of the block in `O(n²)`. On
simulated iid Normal returns the formula at the true `R` matches the excess to `5 × 10⁻⁴` on both
paths, and the plug-in leaves at most 0.4 % in the steady state at a half-life of 10 and 0.03 % at
40. The next order, about `9 s₂²`, is stated and not corrected: it is largest in the warm-up rows,
1.08 at `K = 10` and a half-life of 10, and 1.006 on the first row at the default half-life of 40.
A fixed `w` takes no change.

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
- The default inverse-volatility direction keeps the next order of its excess, about `9 s₂²`,
  which the docstring of its statistic states with the warm-up rows.
- **Three limits remain, and three child issues of map #1375 hold them.**
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
- **An inverse-volatility direction from an earlier or a slower estimate** (#1430). An exponential
  estimate shares almost all its observations with the estimate of the step before and with a
  slower one, so their errors stay correlated: at a half-life of 10 and `R = I` the residual is
  1.064 with the current direction, 1.059 with the direction of the step before and 1.025 at four
  times the half-life. Only a direction that reads no estimate is exact.
- **A fixed default direction** (#1430). The factor of #1428 is exact for it on every row, but the
  target would then measure another portfolio, which the most volatile assets dominate.
- **A keyword on each target.** `DiagonalTarget` would carry a field that a geodesic shrinkage
  ignores, and the scalar estimator would need a keyword of its own anyway.
