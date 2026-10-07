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

**`debias::AbstractRegimeDebias = ExactDebias()` is a field of both estimators**, not of a target. The bias is a property
of the estimate. `DiagonalTarget` is a field-less singleton that `GeodesicShrinkageCovariance` also
reads, and the scalar estimator has no target. The field that #1415 put on `MahalanobisTarget` moves
to the estimator. That commit had not reached `main`. `debias = RawStatistic()` is the raw statistic, bit
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
The factor reads the correlation alone, so it keeps the scale invariance. `debias = RawStatistic()` takes no
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
without a closed form. `hac_floor = NoHacFloor()`, the default, keeps the recursion the quadratic form,
and the scalar estimator floors the variance that it returns at zero. `hac_floor = PerTermHacFloor()` is the
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

**Under HAC the recursion runs on the spectrum of the banded weight matrix (#1438).** A HAC
estimate is `Z'AZ`, so the recursion starts from `D₀(s) = ln det(I + sA)`, tabulated on its lattice
by the banded LDLᵀ of #1433 with the derivative carried through each pivot. `A` has negative
eigenvalues, so no moment of `R` is finite, and the transform is cut at the maximum `s*` of `D₀`,
as the table of one direction is; a bisection on the exact slope places it, because a shift of one
sixteenth in `ln s` moves a factor by 0.3 % near the gate. The recursion cannot read `D₀` past
`s*` as it stands. Two continuations failed: a hard cut, where the transform is zero past `s*`,
puts a pole `k / (s* − s)` into the level `k`, and a continuation from `s*` makes `ρ₀` rise, which
no Laplace transform of a positive law does. Both give factors that are not finite near the gate.
The recursion reads `D₀` continued past the peak of its saturated count `σ ρ₀(σ)` as the transform
of those saturated weights, `ρ₀ = N_m / σ`, which falls and joins with one derivative, and the last
transform reads the true `D₀` up to `s*` and stops there. The factor is exact at one asset on that
cut and where `A` is positive definite, and at no lag it is the plain factor to 2e-14. At 12
assets, a half-life of 10 and two lags, the three factors are 1.2e-3, 8e-4 and 3e-4 below a Monte
Carlo of four million draws of the HAC estimate itself, where the mean's fixed point was 1.8 %,
5.3 % and 8.9 % above; near the gate they are 0.2 % to 0.7 % below it, as the plain recursion is
at the same effective count. The squared multipliers become 1.0035, 1.0031 and 1.0023 within
0.004 over 8 seeds, from 0.984, 0.952 and 0.920. Near the gate a HAC factor is steep, so the first
32 counts after it take the recursion itself, and sixteen nodes past them interpolate the ratio to
the law of the plain fixed point, within 1e-9 at half-lives up to 40 and 5e-6 at 250.

**On the separate correlation path the Mahalanobis factor also carries the noise of the variance
at `decay`** (#1437). The block is `D R̂ D`, with `D` from the variance at `decay` and `R̂` from
the correlation at `cor_decay`, and the factor at `cor_decay` holds the noise of a diagonal at
`cor_decay`. At `R = I` and equal weights `R̂` is independent of the variances, so the diagonal
enters through `E[1/Q]` alone: the factor is multiplied by
`κ = E_decay[1/Q] / E_cor_decay[1/Q]`, the ratio of the two exact tables, plain or HAC
(`variance_noise_bias!`). The noise of the variances averages over the assets, so it moves the
statistic mainly through its mean, and the per-method scalar ratios (1.026 and 1.017) correct too
little at 12 assets. **It also spreads the statistic** (#1439), which lowers the root and the log.
At one asset the block is the variance alone and the exact factor is the method's own table at
`decay`, which is `κ` times the ratio `ρ_λ / ρ_λc` of the method's moment to the mean's at the two
decays. On a sphere of `n` independent terms the second-order change of the root and the log is
`3 / (n + 2)` of the change at one asset, so the factor is `κ (ρ_λ / ρ_λc)^(3 / (n + 2))`: exact
at one asset, one for the mean, and `κ` alone as `n` grows. At 12 assets, a half-life of 10 and a
correlation half-life of 20 the power is 0.9981 and 0.9963 (0.9960 and 0.9921 at two lags), and
the squared multipliers become 0.9975, 0.9970 and 0.9964 over 8 seeds, from 1.0321, 1.0297 and
1.0272 at `cor_decay` alone and 0.9975, 0.9952 and 0.9927 with one `κ`; at two HAC lags the three
methods read 1.001, 1.001 and 1.000, from 1.090 for RMS at `cor_decay` alone (1.026 where each row
divides by the volatility after its update). Against the true factor of the block, measured without
return noise over 64 seeds, the three methods read 1.0018, 1.0018 and 1.0017 at `R = I`, 0.9972,
0.9966 and 0.9958 on a random factor model, and 0.9969, 0.9943 and 0.9917 at an equicorrelation of
0.8. The power holds the noise of the variances with the correlation fixed:
on 4 assets and half-lives of 5 and 40 it leaves 0.02 % and 0.06 % of a gap of 1.5 % and 3.0 %.
In the full path the noise of `R̂` amplifies the spread, the data the two estimates share reduces
it, and the rule that divides each correlation row by a volatility moves it. Without HAC they
cancel at `R = I`, and at an equicorrelation of 0.8 the power carries 42 % of the spread. At two
lags, on the kernel of the damped rows below, FirstMoment and Log read 0.02 % and 0.01 % above RMS
at `R = I`, and 0.56 % and 1.18 % below it at an equicorrelation of 0.8. The maintainer ruled
on #1439 that the term lands at this strength, because the mean at a correlated `R` needs the noise
of `V̂_i / Q_ii` together with its coupling to the correlation (#1447): the bias of the standardised
correlation and a compensating rise in its noise are of one order there, and a fix of `R = I`
alone makes the correlated cases worse.

**Each row of the correlation state divides by the volatility after its update, and under HAC by
the volatility before it** (#1440, #1444). The step of `Q` divides the product of row `t` by the
root of the two variances. Without HAC the diagonal of the row is `x_t² ≥ 0`. On iid Normal
returns the two rules have the same bias, and where the volatility moves the rule after the update
is 0.5 points closer to the true correlation, so it stays (#1440). Under HAC the diagonal
`x_t² + 2 Σ_l w_l x_t x_{t−l}` can be negative. A variance that holds the row damps a positive row
and amplifies a negative one, so the diagonal of `Q` is skewed down and can fall to zero. At two
lags the rule after the update has 9 % to 19 % more mean squared error than the rule before it on
11 of 13 cases of returns, 20 % to 40 % more at four lags or at half-lives of 5 and 10, and 136 %
to 252 % more at eight lags. There 1.1 % to 3.0 % of its rows fall on the guards of the state,
which hold the covariance where a diagonal of `Q` is not above `min_val` and clamp the correlation
to `[−1, 1]`; the rule before the update trips no guard. So under HAC the row divides by the
variance before the step. It needs no new state. `hac_vol_before = VolatilityAfterUpdate()` keeps the rule after the
update, the rule of the oracle (ADR 0186).

**Under HAC the tables at `cor_decay` read the kernel of the damped rows** (#1445). Under the rule
before the update, row `p` divides the lagged return `x_{p−l}` by `σ_{p−1}`, and `V_{p−1}` already
holds `x_{p−l}²` and the HAC products of `x_{p−l}`. So each lagged term is damped (its variance
falls to 0.880 and 0.895 of the variance of the same return divided by its own volatility, at a
half-life of 10 and two lags) and correlated with the returns near it, while its mean product with
`x_p` stays zero. The correlation carries less noise than the Bartlett kernel of `cor_decay`
assumes, so that kernel over-corrected: at `R = I` the statistic read 0.8 % to 9.5 % low over 4 to
24 assets, half-lives of 5 to 20, correlation half-lives of 20 and 40 and one to four lags, and up
to 10.2 % low at a correlated `R`. The damping is also why this rule has the lowest error of the
correlation: a rule that divides each return by its own volatility before its own update makes
`Q = Y'AY` exactly and the factor hold to 1.5 %, but it has 9 % to 23 % more squared error on `R̂`.
Regressed on the returns divided by their own volatility within `2L` of its row, each damped term
is `Σ_d β_{l,d} y_{p−d} + e_{p,l}`, with a small residual (0.008 to 0.026 of `E[1/V]`). A row is then
a raw HAC product of the `y` with the kernel `ω_d = Σ_l k_l β_{l,d}` of width `2L`, the residual
variance in quadrature at `d = l`. A rebuild of the separate path with that stand-in gives
`E[(R̂⁻¹)_ii]` within 0.01 % of the rule at two lags and 0.54 % at four. `β` and the residual come
exactly from the law of the steady-state HAC variance: `1 / √(ab)` is an integral over `θ` of
`1 / (a sin²θ + b cos²θ)`, which turns the product of two volatilities into one quadratic form, and
each moment is a Laplace integral on the banded LDLᵀ factorisation (`hac_row_kernel`). The kernel
at a half-life of 10 and two lags is 0.613, 0.290, −0.022 and −0.003, against the Bartlett 0.667 and
0.333. With it the statistic reads 0.2 % to 2.2 % high at `R = I` and −2.6 % to +3.4 % over every
correlation of the same grid. What remains is the part that every row rule has, the time
modulation of the rows and their coupling with `1 / V̂`, which the `κ` of the variance does not
read; it grows with the count of assets and of lags. The shrink of the `DiagonalTarget` spectrum
subtracts the same noise from `Σ r̂²`, and it reads the same kernel (#1461): the Bartlett kernel put
the mean of `r̂²` at `R = I` 14 % too high, the kernel 4 %, and the FirstMoment and Log factors move
towards their truth by up to 0.3 % and 0.7 % at a correlated `R`, and by at most 0.26 %
away from it at `R = I`, where the Bartlett over-shrink stopped on the identity. The variance keeps
the Bartlett kernel, and `hac_vol_before = VolatilityAfterUpdate()` keeps the Bartlett kernel for the correlation too.

**Without HAC the factor does not read the correlation of the assets, and this is a documented
limit** (the maintainer ruled so on #1447). With `h_i = √(Q_ii / V̂_i)` the statistic splits
exactly as `tr(Ĉ⁻¹ R) = Σ_i h_i² (Q⁻¹ R)_ii − ½ Σ_ij (Q⁻¹)_ij R_ij (h_i − h_j)²`. On rows that are not
standardised the mean of the first sum does not read `R` (the part of `r̃_i = R^{−1/2} e_i` off
the direction of column `i` is odd under a sign flip of the other directions), so every effect of
`R` is a sum over pairs. The library standardises each row, which is not a congruence, and both
sums then move with `R`: at 12 assets, half-lives of 10 and 20 and an equicorrelation of 0.8, the
first by −1.66 % (the standardisation biases each correlation down by `0.012 ρ (1 − ρ²)`) and the
second by +1.17 %, net −0.49 %. The mean-field value, which keeps the exact means of `Q⁻¹`, `h²` and
`(h_i − h_j)²` and drops their covariance, moves by +0.89 % over the same step: the covariance of
the two noises moves 2.8 times the net effect, with the opposite sign. A fix therefore needs each
sum to about 2.5 %, from the joint law of two correlated volatility chains and the pair's own
Schur complement, which no expansion in the noise of the variances reaches. Over six correlations
the three methods read 0.9897 to 1.0018 of the truth, and a three-factor model with a `γ` of 1.95
reads lower on the mean (0.9947) than every equicorrelation, so no function of `γ` alone holds it.

**The gate is `K > n + 3`, where the variance of the statistic is finite.** At `n = 1` an estimate
of four observations or fewer is not scored. The gate reads the count of each asset, so a young
asset leaves the statistic as it does below `min_obs`. Under HAC the effective count
`1 / tr(A²)` must also pass `n + 1`, the count at which the mean of the inverse of a plain estimate
is finite. At 12 assets, a half-life of 10 and two lags, `K > n + 3` alone opens at `K = 16`,
where 11 % of draws are not positive definite, and the effective count opens at `K = 54`. Over
half-lives of 5, 10 and 40, one to five lags and one to 12 assets, the first scored row is not
positive definite in at most `2 × 10⁻⁴` of draws, and a steady state with too few effective
observations, such as 12 assets at a half-life of 5, is never scored.

**The estimated location is normalised, and each term is divided by its exact factor (#1507).**
The location of an asset was the recursion `m ← λ m + (1 − λ) x` from zero, so it estimated
`(1 − λ^k) μ`, and each deviation kept part of the mean: even with an exact factor the variance
was 2.4 % (half-life 10, 20 rows, `μ / σ = 0.3`) to 36 % (half-life 40, 60 rows, `μ / σ = 1`) too
large in the warm-up. The location is now the exponentially weighted mean divided by its weight,
and a return gives a term from the second valid return of its asset. For returns independent in
time the deviation from the location before it has the variance `σ² (1 + 1 / n_eff)`, with
`n_eff` the Kish count of the weights of the location, so each squared deviation is divided by
that factor. Each product of a pair is divided by `1 + c_ij`, with `c_ij` the overlap of the two
locations over their common returns (ADR 0181, amendment of 2026-10-06). Under HAC a lagged
product of two estimated deviations has a non-zero mean too, and the factor adds it, from the
counts, the valid assets and the overlap of each lagged observation. With equal weights the
factor is `k / (k − 1)`, the factor of the sample variance. A simulation checked the three
factors against Monte Carlo means before the build (z-scores with an RMS of 0.97 and 1.02), and
`test_08z` pins the estimate at the truth from the second row, with and without HAC.

**Centring is a field, and so are the three switches of this ADR.** The estimators hold
`centring = EstimatedCentring()`, `PreCentred()` or `ZeroStartCentring()`, `debias = ExactDebias()` or
`RawStatistic()`, `hac_floor = NoHacFloor()` or `PerTermHacFloor()`, and, on the covariance,
`hac_vol_before = VolatilityBeforeUpdate()` or `VolatilityAfterUpdate()`. Each abstract root is
public and each member is exported. The rule of the oracle is one keyword away for every switch:
`PreCentred()` is its default centring, `RawStatistic()`, `PerTermHacFloor()` and
`VolatilityAfterUpdate()` are its other rules, bit for bit. `ZeroStartCentring()` is the oracle's
estimated location, the recursion from zero that is not divided by its weight, with no factor. It
estimates `(1 − λ^k) μ` rather than the mean that it subtracts, so it is not the default, and it
stays one keyword away (ADR 0186) at parity with every stored oracle case of that rule.

**Under the estimated location the tables read the exact law of the shared deviations (#1548).**
Each term has the right mean, but the terms share their location, so the estimate is a quadratic
form `Q = c x'Mx` in the returns with a dense `M`, and the next deviation `u` reads the same
location. Two parts move each factor: the law of `Q`, and the dependence of `u` on `Q`, which the
root mean square (`E[u²/Q]`) and the first moment (`E[|u|/√Q]`) read and the log does not. Without
HAC they nearly cancel; with HAC they add, so the pre-centred tables read the risk low (#1527): at
a half-life of 10 the root-mean-squared factor was 0.26 %, 0.58 % and 1.39 % too large at one, two
and four lags, and the Mahalanobis factor 1.3 % to 3.1 % at two to 12 assets. The estimate after
`K` terms is `R_K = λ R_{K−1} + q_K`, so on a lattice of step `|ln λ|` in `ln s` the transform at
`s` reads the transform at `λ s` of the count before, and a Gaussian state of the location and the
last `L` deviations carries both parts exactly, one step per point and count. The scalar table
under `EstimatedCentring()` reads it with or without HAC, and so do the diagonal target and the
portfolio target on one decay. It agrees with the eigenvalues of `M` to `10⁻¹²` where the estimate
is positive definite. Past `λ^K = √ε` it holds its ratio to the pre-centred table, within
`4 × 10⁻⁹`. It costs 0.12 s to 0.16 s at a half-life of 40 and 3.4 s at 250, once per state. A
Monte Carlo of the estimator over half-lives 5 to 40, one to four lags and every target puts the
mean squared multiplier within its noise (z from −1.2 to 0.5), where the scalar estimator was at
z = −3.0 before.

The Mahalanobis target under HAC starts its level recursion from `ln det(I + σM_K)` of the same
lattice, on a grid of the lattice's step, with the peak placed on the exact slope of one chain.
The root mean square adds the dependence exactly: `E[u'W⁻¹u] = E[tr W⁻¹] + E[y'W⁻¹y]` with
`y = Z'g`, and the second term is the derivative in `t` of `E[ln det Z'(M + t gg')Z]`, which the
sum of the levels gives. The first moment and the log read the law alone. Against a Monte Carlo of
a million draws of the statistic the root mean square is within 0.3 % at the steady state and
0.6 % at the first scored count, and the first moment and the log are 0.4 % to 1.3 % high, where
the pre-centred factors were 1.3 % to 3.1 % high, and 7.5 % at the first scored count of a
half-life of 40. The nodes cost about twice the pre-centred nodes. Without HAC the Mahalanobis
target keeps the pre-centred recursion: there the two parts cancel within 0.16 % for every method,
and the law alone would read the first moment and the log 0.17 % to 0.65 % high. The exact
dependence of the first moment and the log has a route, `E ln(1 + t d²) = E ln det W_t − E ln det W`
for every `t`, at the cost of one run of the recursion per `t`; #1549 holds that question of
cost. The separate path keeps the pre-centred law at `cor_decay`, with the limits below.

## Consequences

- Every default fit of a regime estimator moves, and so does every default fit of the
  Cross-Sectional Factor Prior, whose `ve` and `pe` are regime estimators. A stored oracle case
  states `debias = RawStatistic()`.
- The default inverse-volatility direction keeps the next order of its excess, about `9 s₂²`,
  which the docstring of its statistic states with the warm-up rows.
- **Two limits remain.** On the separate correlation path the Mahalanobis factor does not read the
  correlation of the assets: without HAC the three methods read 0.9897 to 1.0018 of the truth, a
  documented limit (#1447). Under HAC, on the kernel of the damped rows, the statistic reads
  0.2 % to 2.2 % high at `R = I` and −2.6 % to +3.4 % at correlated `R` over 4 to 24 assets and
  one to four lags (#1445).
- Under HAC and the estimated location, the Mahalanobis first moment and log read the law of the
  shared deviations and not the dependence of the deviation: 0.4 % to 1.3 % high against a Monte
  Carlo of the statistic, a documented limit until #1549 decides its cost.
- The exact table of the estimated location costs 0.12 s to 0.16 s per state at a half-life of 40
  and 3.4 s at 250, where the pre-centred table costs 0.04 s at 40. The Mahalanobis nodes under
  HAC cost about twice the pre-centred nodes.
- A HAC fit on the separate path moves with the row rule of #1444; no stored oracle case runs that
  path, and `hac_vol_before = VolatilityAfterUpdate()` gives the rule of the oracle bit for bit.
- The Mahalanobis nodes under HAC cost 0.3 s to 1.5 s for each count of assets at half-lives up
  to 40, and up to 3.7 s at a half-life of 250 and five lags. On the separate path under HAC the
  kernel of the damped rows costs 0.1 s at a half-life of 10 and two lags, and 4 s at a half-life
  of 250, once for each state of a Mahalanobis target, or of a Diagonal target whose method is
  FirstMoment or Log.
- The FirstMoment and Log calibrations of `DiagonalTarget` assumed a `χ²(n)` sum. Before this
  decision the Jensen bias hid part of that error; after it, correlated assets read 0.941 and 0.962.
  #1432 divides the sum by the factor of its law, and #1434 makes that law read the noise of each
  estimate (see the decisions above). The rest is of the third order in the noise, and the estimate
  of the correlation that the law reads.
- The diagonal factor of the first moment and the log costs one eigen-decomposition of `D R̃ D`, one
  of the `n × n` integral of the tilted law, and a few products of order `n³` per scored row.
  `RootMeanSquaredAdjusted` and `debias = RawStatistic()` pay none of it.

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
- **For the Mahalanobis target under HAC, the mean's factor times the ratio of the method's
  moment to the mean from the plain recursion at the effective count `1 / tr(A²)`** (#1438,
  fix 2). It is cheap, but it is exact in no limit.
- **For the Mahalanobis target under HAC, a hard cut or a continuation from the cut in the
  recursion** (#1438). The first puts a pole into each level and the second is not the transform
  of a law, and both give factors that are not finite near the gate.
- **On the separate path, one `κ` for the three methods** (#1437). It holds the mean alone, and
  left FirstMoment and Log 0.19 % and 0.38 % below RMS at `R = I`, and 1.5 % and 3.0 % at 4
  assets and half-lives of 5 and 40.
- **On the separate path, the linear term `1 − 3Δσ² / (n + 2)` and `1 − 6Δσ² / (n + 2)`**, with
  `Δσ²` the difference of the relative variances of `1 / σ̂` at the two decays (#1439). It has the
  same second order as the power, but it is not exact at one asset, and it over-corrects at 4
  assets and half-lives of 5 and 40.
- **On the separate path, the plain mean at `R = I` from a one-asset deterministic equivalent**
  (#1439). It is within 0.005 % of the truth at `R = I`, but the correlated cases move from about
  −0.3 % to −0.5 % for RMS, because the part it removes offsets the part that the correlation
  adds (#1447).
- **On the separate path, a second-order expansion in the noise of `ln(V̂_i / Q_ii)` with its
  coupling to `Q`** (#1447). The coupling is not a correction to the expansion: without it the
  mean moves the wrong way with `R`, and on rows that are not standardised the pair term is 21 %
  below the product of its means at an equicorrelation of 0.8. A term in `γ` alone cannot hold a
  factor model and an equicorrelation at once.
- **Under HAC, each row divided by the volatility after its update** (#1444). It is the rule of
  the oracle and of the plain path, but a variance that holds a HAC row amplifies a negative
  diagonal: 9 % to 19 % more mean squared error at two lags, and the guards of the state bind on
  up to 3 % of rows at eight lags. It stays one keyword away.
- **Under HAC, each row divided by the volatility of a plain variance after its update** (#1444).
  It has the smallest bias of the three rules and the lowest mean squared error at a correlation of
  0.9 with a moving volatility, but a higher one at 0.3 and 0.6 on every setting, by 3 % to 8 % at
  two lags and up to 34 % at eight, and it needs one more vector in the state.
- **Under HAC, each Bartlett weight scaled by the root of the variance of its damped term** (#1445).
  It is one Laplace integral for each lag, and it holds the mean at `R = I` to 0.6 % at one and two
  lags, but in part by a cancellation: the modulation of the rows offsets the correlation of the
  damped terms, which it drops, and at four lags it reads 3.6 % low.
- **Under HAC, one scale for every Bartlett weight that matches the second moment of the
  correlation** (#1445). The second moment of `Q_ij` does not fix the law: the stand-in reads 1.2 %
  and 3.6 % below the rule at two and four lags.
- **Under HAC, each return divided by its own volatility before its own update** (#1445). It makes
  the row a congruence and the factor hold to 1.5 %, but it has 9 % to 23 % more squared error on
  `R̂` than the rule before the update.
- **Under HAC, a documented limit** (#1445), as without HAC. The mean read up to 9.5 % low at
  `R = I`, and the damping has an exact model.
- **A keyword on each target.** `DiagonalTarget` would carry a field that a geodesic shrinkage
  ignores, and the scalar estimator would need a keyword of its own anyway.
