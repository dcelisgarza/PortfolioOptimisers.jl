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

**`DiagonalTarget` then divides its sum by the factor of its law, for FirstMoment and Log (#1432).**
The sum of `n` correlated squares is `S = Σ_k μ_k χ²_k(1)` on the eigenvalues `μ_k` of the
correlation, so its root and its log read the correlation, which the constants `√n` and
`ψ(x n) + ln y` do not. `diagonal_law_factor` divides `S` by `(E[√S] / √n)²` or by
`exp(E[ln S] − κ)`, each one integral of the Laplace transform `Π_k (1 + y μ_k t / n)^(-x)`. The Log
method takes each square as a `Gamma(x, y)` variate, which is the law that its constant assumes, so
its factor is one at `μ = 1` whatever its parameters. The spectrum is that of the estimated
correlation, shrunk towards the identity until `Σ_{i≠j} r_ij²` is its unbiased estimate: the
variance of a sample correlation is `(1 − r²)² / K_ij`, with `K_ij` from the weight of the pair.
The factor reads the correlation alone, so it keeps the scale invariance. `debias = false` takes no
factor, and `RootMeanSquaredAdjusted` needs none, so neither pays the eigen-decomposition.

**A HAC estimate reads the spectrum of its weight matrix (#1433).** Over `K` observations the
HAC estimate is `Q = z' A z`, with `A = c B` banded: `B_jj = λ^j` and `B_{j,j+i} = λ^j k_i`, with
the Bartlett weight `k_i = 1 − i/(L + 1)`. For the covariance, `Ĉ = Z' A Z`. Diagonalised, `U'Z`
has the law of `Z`, so a HAC estimate has **exactly** the law of a plain estimate whose weights are
the eigenvalues `μ` of `A`, and every factor above holds with `μ` in place of `w`. None needs an
eigen-decomposition. `regime_bias_table(method, decay, K, hac_lags)` reads
`G(s) = det(I + s B)^(-1/2)` from a banded LDLᵀ factorisation that adds one row per `K`, in one pass
as before. `mahalanobis_bias(decay, K, n, hac_lags)` solves its fixed point from `ℓ'(t)` and
`ℓ''(t)` of `ℓ = ln det(I + t A)`, carried through the same recursion. The inverse-volatility
excess reads `s_v = tr(A²)` and `s_vc = tr(A_v A_c)`, and the pair count of `DiagonalTarget` reads
`1 / tr(A²)`, all closed sums over the band. On a positive definite `A` the table agrees with the
eigenvalues to `10⁻¹²`, and at two lags and a half-life of 10 the steady-state factor is 1.148, twice
the excess of the plain weights. Without HAC, every path runs as before.

**`A` is indefinite, so the Laplace integral stops at the minimum of `G`.** The exponential
weights break the Fejér identity that makes the equal-weight Bartlett estimate a sum of squares.
The smallest eigenvalue of `B` is negative from `K` of 5 to 50, by the half-life: −0.0067 times
the largest at a half-life of 10 and two lags. So `Q` can fall below zero, and no moment of its
inverse is finite. Past the minimum of `G` the negative eigenvalues rule the transform, and it
rises to the first zero of the determinant; the table cuts it at the minimum, with the half weight
of a closed trapezoid. Where `P(Q ≤ 0)` is below the reach of a Monte Carlo of 200 000 draws, as
from a half-life of 5 at two lags, the cut matches the Monte Carlo to its noise.

**The per-term floor at zero is off by default.** The scalar estimator and the variance of the
separate correlation path floored each HAC square `x_t² + 2 Σ k_i x_t x_{t−i}` at zero, which the
documented formula does not state. On returns with no autocorrelation the floor makes the variance
6.7 %, 16 % and 41 % too large at one, two and five lags, and the scalar variance then differs from
the diagonal of the covariance of one decay, which has no floor. The floor also leaves the law
without a closed form. `hac_floor = false`, the default, keeps the recursion the quadratic form,
and the scalar estimator floors the variance that it returns at zero. `hac_floor = true` is the
oracle's rule of map #1375 (ADR 0186).

**The factor of the law also reads the noise of each estimate, in the variable of the method
(#1434).** Each term carries the noise `a_i = 1 / (f Q_i)` of its estimated variance, with
`E[a_i] = 1`, but the root and the log read other moments of it, and the correlation of the assets
correlates the noise: `Cov(Q_i, Q_j) = r_ij² Var(Q)`. The mean's factor thus left 0.986 and 0.971
at a correlation of 0.9 and a half-life of 10, and 0.967 and 0.935 at 0.99 and a half-life of 5.
The factor now expands the root in `u_i = a_i^(1/2)` and the log in `ℓ_i = ln a_i`, about their
means. The root is 1-homogeneous in `u` and the log is additive in `ℓ`, so each is linear along the
direction where every term carries the same noise, and the second-order term vanishes there. The
factor is the exact law of `D R̃ D`, with `D = diag(b_i^(1/2))` at the mean of the variable
(`b = E[Q^(-1/2)]² / f` or `exp(-E[ln Q]) / f`), plus that second-order term, whose covariance is
`(v_i v_j)^(1/2) r̃_ij²` (`v = f / E[Q^(-1/2)]² − 1` or `Var[ln Q]`). `RegimeTermMoments` tables
`(f, b, v)` for each count in the one pass of `regime_bias_table`, and `Var[ln Q]` is two single
integrals of the Laplace transform, which agree with the trigamma function at equal weights. A HAC
estimate is a banded quadratic form in the same returns, so its table reads the spectrum of its
weight matrix, and the covariance of two of its estimates is still `r_ij²` times their variance.
The factor is exact at one asset, where it is the scalar method's factor, and at a correlation of one.
On simulated iid Normal returns with the true correlation (12 assets, 8 × 4 000 000 rows, standard
error 1e-4), it is within 0.05 % from a correlation of 0 to 0.99 at a half-life of 5, and within
the noise at 10 and 40, at 1, 2 and 12 assets. The quadratic term reads
`Σ_ij C_ij M_ij(τ)²` on the tilted covariance `M(τ) = V diag(d(τ)) V'`: the τ-integral of
`d d'` is a positive semi-definite matrix whose eigenvalues fall fast, so its truncation at the
machine epsilon keeps 2 to 8 terms at 12 and 40 assets, and the cost is a few products of order
`n³` per scored row, not one of order `n⁴`.

**`MahalanobisTarget` divides by the moment that its method reads too (#1431).** The squared
distance of a correctly calibrated return is `χ²_n R`, with `R = 1/S` and `S` the Schur complement
of one direction in `W = Σ_j w_j z_j z_j'`, so each method reads a moment of `R`. For `n > 1` on
exponential weights no one-dimensional integral gives it. Given the other `n − 1` directions,
`S` is a weighted chi-square, and the matrix determinant lemma gives its Laplace transform as the
`n = 1` transform times `E[(det(Z₂'DZ₂) / det(Z₂'D_tZ₂))^{1/2}]`. The family `w/(1 + s w)` is
closed under that step, so one recursion over the directions carries the transform from level to
level, with one approximation: the mean log-determinant of each deflation goes inside the
exponential. The recursion is exact at `n = 1` and at equal weights for every `n`. Against a
Monte Carlo of a million draws at 12 assets and a half-life of 10, its three factors are 5e-4 to
7e-4 low, where the fixed point `b` of #1415 is 0.55 % high for the mean and 2.5 % and 4.5 % high
for the root and the log. The squared multipliers become 1.000 within 0.002 for all three
methods (they were 0.994, 0.975 and 0.956). The recursion runs in its differentiated form, which
is stable: an error at `s` flows only toward smaller `s`. It runs at 16 counts of observations,
and the ratio to the inverse-Wishart law of mean `b` interpolates the other counts to 1e-6. The
state keeps the nodes for each count of assets, at 0.3 s for 12 assets. The measure is on #1431.
Under HAC the target keeps the fixed point of the mean on the spectrum of `A` for every method:
the recursion reads `ln(1 + s μ)` over the spectrum, and `A` has negative eigenvalues (#1438).

**The gate is `K > n + 3`, where the variance of the statistic is finite.** At `n = 1` an estimate
of four observations or fewer is not scored. The gate reads the count of each asset, so a young
asset leaves the statistic as it does below `min_obs`. Under HAC the effective count
`1 / tr(A²)` must also pass `n + 1`, the count at which the mean of the inverse of a plain estimate
is finite. At 12 assets, a half-life of 10 and two lags, `K > n + 3` alone opens at `K = 16`,
where 11 % of draws are not positive definite, and the effective count opens at `K = 54`. Over
half-lives of 5, 10 and 40, one to five lags and one to 12 assets, the first scored row is not
positive definite in at most `2 × 10⁻⁴` of draws, and a steady state with too few effective
observations, such as 12 assets at a half-life of 5, is never scored.

## Consequences

- Every default fit of a regime estimator moves, and so does every default fit of the
  Cross-Sectional Factor Prior, whose `ve` and `pe` are regime estimators. A stored oracle case
  states `debias = false`.
- The default inverse-volatility direction keeps the next order of its excess, about `9 s₂²`,
  which the docstring of its statistic states with the warm-up rows.
- **Two limits remain, and child issues of map #1375 hold them.**
  - Under HAC `MahalanobisTarget` keeps the fixed point of the mean for every method, so it
    over-corrects FirstMoment and Log there: 0.952 and 0.920 at two lags (#1438). #1431 gave each
    method its own moment without HAC.
  - On the separate correlation path the Mahalanobis factor reads `cor_decay` alone, so the noise
    of the variance at `decay` is not in it: 1.031 without HAC and 1.090 at two lags (#1437).
- The FirstMoment and Log calibrations of `DiagonalTarget` assumed a `χ²(n)` sum. Before this
  decision the Jensen bias hid part of that error; after it, correlated assets read 0.941 and 0.962.
  #1432 divides the sum by the factor of its law, and #1434 makes that law read the noise of each
  estimate (see the decisions above). The rest is of the third order in the noise, and the estimate
  of the correlation that the law reads.
- The diagonal factor of the first moment and the log costs one eigen-decomposition of `D R̃ D`, one
  of the `n × n` integral of the tilted law, and a few products of order `n³` per scored row.
  `RootMeanSquaredAdjusted` and `debias = false` pay none of it.

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
- **The law of the estimated correlation, not shrunk**, as #1432 first proposed. The eigenvalues of
  a sample correlation are too dispersed, so it over-corrects: 1.008 and 1.016 at 12 assets and a
  half-life of 10, and 1.013 and 1.027 at 30 assets.
- **A two-moment Gamma law on the unbiased `Σ r_ij²`.** It is cheap, but a spike in the spectrum is
  not a Gamma variate: at a correlation of 0.9 the Log method reads 1.36.
- **The Bartlett inflation alone** (#1433): the plain table at the effective count
  `K / (1 + 2 Σ k_i²)`. The spectrum of `A` is not a rescaled geometric sequence, and it has
  negative eigenvalues, so the rescaling is not exact.
- **A HAC gate on `K > (n + 3)(1 + 2 Σ k_i²)`.** It guards the warm-up, but a steady state with too
  few effective observations is scored forever: at a half-life of 5, two lags and 12 assets, 2.5 %
  of draws are not positive definite at every `K`, and 95 % at five lags.
- **A HAC gate on `1 / tr(A²) > n + 3`.** It never opens at a half-life of 10, two lags and 12
  assets, where no draw is non-positive-definite.
- **The per-term floor as the default.** It is 16 % too large at two lags, and it has no closed
  law, so no exact factor.
- **A second-order expansion in `δ_i = a_i − 1`** (#1434, fix 2). It is second-order correct, but
  the root and the log are not linear in `a` along the common direction, and `a` is skewed, so it
  over-corrects where the noise is common: 1.0038 and 1.0073 at a correlation of 0.99 and a
  half-life of 5, and 1.0039 and 1.0078 at one asset, where the scalar factor is exact.
- **An interpolation on the effective count `n² / Σ μ²`** between the mean's factor and the
  method's factor at one asset (#1434, fix 3). It is exact at a correlation of one, but at
  independent assets the root and the log of a sum of 12 terms still see the noise: 0.992 and
  0.985 at a half-life of 5.
- **The exact law at `R = I` alone** (#1434, fix 1). It is a double integral for each count, and it
  is exact only where the assets do not correlate.
- **For the Mahalanobis target, a scaled inverse chi-square matched to `b`** (route 2 of #1431).
  It is exact at equal weights, but at 12 assets and a half-life of 10 it misses the mean by
  +0.5 %, the root by +0.07 % and the log by −0.3 %.
- **For the Mahalanobis target, a first-order (Marchenko–Pastur) log-determinant in the Laplace
  transform.** It is exact at both limits, but 0.34 % high at 12 assets. Its error is the
  finite-size term that the recursion over directions carries exactly.
- **A keyword on each target.** `DiagonalTarget` would carry a field that a geodesic shrinkage
  ignores, and the scalar estimator would need a keyword of its own anyway.
