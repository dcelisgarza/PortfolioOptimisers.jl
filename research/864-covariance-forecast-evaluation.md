# 864 — The mathematics of a covariance forecast evaluation, batch and online

Research ticket #864 of wayfinder map #861. Written 2026-09-06.

Sources are the papers the ticket names, read from the authors' own copies where the published
text is paywalled: Patton (2011) in its Journal of Econometrics form; Patton and Sheppard in the
Oxford working paper 2008fe22 that became the Handbook chapter, so its equation numbers are the
working paper's; Laurent, Rombouts and Violante in the CIRANO working paper 2009s-45 that became
the Journal of Econometrics article, so its numbers are the working paper's too; Engle and
Colacito (2006) in its journal form; Andersen, Bollerslev, Christoffersen and Diebold (2006),
already cited as `andersen2006` in `docs/src/References.bib`; Diebold, Hahn and Tay (1999); and
the Barra USE4 methodology notes of Menchero, Orr and Wang (2011), whose Appendix A defines the
bias statistic that the portfolio ratio is. Library facts are cited to files of `dev` at
`9a00bd06d5`. Every claim about a document carries its section or equation number, and where a
working paper was read the number is flagged as the working paper's.

Notation, fixed here and used throughout. The papers write the forecast as `H_t` and the proxy
as `Σ̂_t`; the ticket writes the forecast as `Σ̂ₜ` and the realised quantity as `S`. This report
follows the ticket: `Σ̂_t` is the covariance forecast at step `t`, `z_t` the realised return
vector the forecast is judged on, `S_t` the realised covariance proxy, `N` the number of assets,
`h` the horizon, `M` the number of steps of the walk-forward.

---

## Summary

- **The per-step protocol is one-step in the papers, and a fixed horizon `h` extends it by a
  sum.** At step `t`, `Σ̂_t` is formed from the training window and judged on the next `h`
  observations. For `h = 1` the realised quantity is the outer product `z_t z_t'`, which is a
  conditionally unbiased proxy of the true covariance (Patton and Sheppard §1.1 eq. (4)). For
  `h > 1` it is the realised covariance `S_t = Σ_s x_{t+s} x_{t+s}'`, whose target is `h Σ̂_t`
  under serially uncorrelated returns (Andersen et al. §3.6 eq. (3.27)). A fixed `h` keeps every
  step's diagnostic identically distributed under the null, so a mean has one target and one band.
- **The three ratios are one statistic read at three resolutions.** The Mahalanobis ratio
  `z' Σ̂⁻¹ z / N` reads the whole matrix; its target is one by a distribution-free moment identity,
  and under a Gaussian null `N` times it is `χ²_N`. The diagonal ratio `z_i² / Σ̂_ii` reads the
  variances alone and is the squared bias statistic of USE4 Appendix A. The portfolio ratio
  `w' S w / w' Σ̂ w` reads one direction and is Engle and Colacito's accuracy regressand, eq. (18).
  The Mahalanobis ratio is the mean of the portfolio ratio over the eigenportfolios of `Σ̂`, which
  is a consequence of the definitions and ties the three together. Above one is under-prediction,
  below one is over-prediction, for all three.
- **Two losses are robust to a noisy proxy and the library needs both.** The Frobenius (matrix
  MSE) loss `‖S − Σ̂‖_F²` and the Stein (matrix QLIKE) loss `tr(Σ̂⁻¹ S) − log|Σ̂⁻¹ S| − N`. Patton
  (2011) Proposition 1 eq. (23) gives the necessary and sufficient form in the univariate case,
  and Laurent, Rombouts and Violante Proposition 3 eq. (4) gives it in the multivariate case; each
  proof is in the paper's appendix. The Frobenius *norm*, the log and proportional variants, the
  standard-deviation MSE and every correlation-based loss are not robust. At `h = 1` the proxy has
  rank one, so the QLIKE must be written without its proxy-only term: `log|Σ̂| + z' Σ̂⁻¹ z`. That
  is exactly Patton's own eq. (6) convention, and it equals `log|Σ̂| + N m_t`.
- **Aggregation is a mean per diagnostic, and a closed-form band exists for the ratios under a
  Gaussian null.** The mean Mahalanobis ratio over `M` steps is `χ²_{NM} / (NM)`, so its band is
  `1 ± z_{α/2} √(2 / (N M))`; the diagonal and portfolio ratios have `χ²_M / M` and the band
  `1 ± z_{α/2} √(2 / M)`. Kurtosis widens the band to `√((κ − 1) / M)`, and the distribution-free
  band uses a heteroskedasticity-and-autocorrelation-consistent variance of the per-step series,
  as the Diebold–Mariano–West test does. The median's target is **not** one. A loss aggregates as
  a mean whose level means nothing on a proxy; only the difference between two forecasts does.
- **The online form changes the forecast and nothing else.** Every diagnostic is a function of
  `(Σ̂_t, z_t)` alone, so a threaded Partial Fit State that reads out `Σ̂_t` enters the same
  formulas. The one identity a test pins: with an exact seam and an expanding window, the online
  `Σ̂_t` equals the batch refit over the same observations, so every per-step diagnostic and every
  aggregate agrees to the last bit, because the same inputs reach the same functions. The map's
  own oracle statement scopes "last bit" to sum-of-blocks statistics and "a stated tolerance"
  otherwise, so the identity is exact in exact arithmetic and the test measures it in floating
  point.
- **Section 6 gives each definition in the `# Mathematical definition` form**, with existing
  `math_dict` keys interpolated where the definition already fits (`:N`, `:w_port`, `:x_t_obs`,
  `:Sigma_hat`) and new symbols marked for the build ticket to key or keep inline.

---

## 1. The per-step protocol

**Answer.** At step `t` the forecast `Σ̂_t` is a function of the observations in the training
window, and it is judged on the `h` observations of the test window that follows. For `h = 1` the
realised quantity is the outer product of the next return vector; for `h > 1` it is the realised
covariance, the sum of the `h` outer products, judged against `h Σ̂_t`. A fixed `h` is what makes
the steps comparable.

**In the library's words.** A walk-forward (`IndexWalkForward`, `DateWalkForward`; CONTEXT.md
§4.6) hands the Fold Loop one Fold per step: its training index vector, its test index vector,
its already-viewed data. The forecast is fitted on `train_idx` and judged on `test_idx`, so
`h = test_size`, and `expand_train` decides whether the training window expands or rolls
(`src/20_Optimisation/02_CrossValidation/04_WalkForward.jl:54`). The non-optimisation root a
covariance evaluation fits is `NonOptimisationSequentialCrossValidationEstimator`
(`01_Base_CrossValidation.jl:199`), and the map's ground truth #13 names it as the slot.

**`h = 1`.** Patton and Sheppard §1.1 set `E_{t−1}[r_t] = 0`, so `Σ_t = E_{t−1}[r_t r_t']`, and
name the outer product `r_t r_t'` and the realised covariance `RC_t^{(m)}` as the covariance
proxies in common use. Their eq. (4) states the one property the evaluation needs of the proxy:
conditional unbiasedness, `E[Σ̂_t | F_{t−1}] = Σ_t`. Andersen et al. §1.1 eq. (1.8) writes the
same fact as `ε_t² = σ²_{t|t−1} + σ²_{t|t−1}(z_t² − 1)`: the squared innovation is the forecast
plus a mean-zero error, and the error's variance is an order of magnitude above the variation of
the forecast itself, which is why a one-observation proxy is noisy. So at `h = 1` the realised
quantity is `S_t = z_t z_t'` with `z_t` the next return vector, and it is unbiased for `Σ_t`,
of rank one, and noisy.

**The mean.** The papers judge raw returns, on the zero-mean assumption (Patton §1.1; Patton and
Sheppard §1.1). Engle and Colacito Theorem 2 eq. (7) shows that subtracting the in-sample mean
changes nothing asymptotically when the conditional mean is constant, and Theorem 3 eq. (10)
that it changes nothing as the sampling interval shrinks. Patton and Sheppard §4.1 make the same
point in prose. So `z_t` is the raw return, or the return centred at the training window's mean,
and the two agree at the order of the mean squared, which is below the noise of the proxy at a
daily frequency. Which one the library takes is a decision, not a mathematical fact, and either
is consistent.

**`h > 1`.** Two realised quantities are in the literature, and they differ in their noise.

1. The realised covariance over the horizon, `S_t = Σ_{s=1}^{h} x_{t+s} x_{t+s}'`. It is the
   multivariate form of the realised variance that Andersen et al. §7.2 eq. (7.9) put on the
   left of the evaluation regression, with the `h` test observations playing the part of the
   intra-period sample. Its target: if the `h` returns are serially uncorrelated, the covariance
   of their sum is the sum of their covariances (Andersen et al. §3.6 eq. (3.27), stated for the
   variance and read matrix by matrix), and a one-step forecast held across the horizon gives
   `E[S_t | F_t] = h Σ̂_t` when the forecast is correct. So the per-observation quantity is
   `S_t / h`, and it targets `Σ̂_t`.
2. The horizon return `z_t = Σ_{s=1}^{h} x_{t+s}` with proxy `z_t z_t'` and target `h Σ̂_t`.
   Andersen et al. §3.6 name this the standardised residual of the multi-period return. It is
   what a holder over the horizon experiences, and it has rank one whatever `h` is.

Form 1 has `h` times the observations of form 2, so its proxy noise is smaller by the same factor
that Patton §2.2 measures for realised variance against the squared return, and under the
Gaussian null it gives a chi-squared with `hN` degrees of freedom where form 2 gives `N`. Form 1
is the default the literature points to, and form 2 is the alternative a caller may ask for. Both
are stated in section 6.

**Why a fixed `h`.** Three reasons, each from the sources.

- The null distribution of a ratio depends on `h` through its degrees of freedom (section 4), so
  a mean over steps of unequal `h` mixes chi-squared laws with different variances, and has no
  single band.
- The proxy's noise depends on `h` (Patton §2.2: the mean squared error of a realised variance
  falls as `1/m` in the number of intra-period returns), so the loss levels of two steps with
  different `h` are not on one scale.
- Andersen et al. §7.1 note that an `h`-day forecast from a daily model overlaps, so the first
  `h − 1` autocorrelations of the per-step series are non-zero and the variance estimator must
  allow for them. That bandwidth is one number only when `h` is one number.

The walk-forward's `reduce_test` lets the last fold be shorter. The evaluation should either drop
that fold or aggregate as a ratio of sums over all test observations (section 4), which weights a
short fold by its length and reduces to the mean of ratios at a fixed `h`.

## 2. The calibration ratios

**Answer.** Each ratio has target one; above one the forecast under-predicts the dispersion the
ratio reads, and below one it over-predicts. The Mahalanobis ratio reads the whole matrix, the
diagonal ratio the variances, the portfolio ratio one direction. The three are one statistic:
the Mahalanobis ratio is the mean of the portfolio ratio over the `N` eigenportfolios of `Σ̂_t`.

### 2.1 The Mahalanobis ratio

Definition, at `h = 1`:

```text
m_t = z_t' Σ̂_t⁻¹ z_t / N
```

and at `h > 1` with the realised covariance, `m_t = tr(Σ̂_t⁻¹ S_t) / (N h)`, which is the mean of
the per-observation quadratic forms over the test window.

*Target one, distribution-free.* When the forecast is the true covariance, `E[z z'] = Σ̂`, so
`E[z' Σ̂⁻¹ z] = tr(Σ̂⁻¹ E[z z']) = tr(I_N) = N`. This uses second moments only, and it is the
matrix form of the unbiasedness in Patton and Sheppard eq. (4).

*Chi-squared under a Gaussian null.* Patton and Sheppard §1.1 define the standardised vector
`ε_t = Σ_t^{−1/2} r_t` and recommend the spectral square root, since the Cholesky root depends on
the asset order. If `z ~ N(0, Σ̂)` then `ε = Σ̂^{−1/2} z ~ N(0, I_N)`, so `z' Σ̂⁻¹ z = ε' ε` is a
sum of `N` independent squared standard normals, which is `χ²_N` by the definition of the
chi-squared law. This is the standard result on the quadratic form of a Gaussian vector, stated
in Anderson's textbook (chapter 3); I did not verify its theorem number there, and the two-line
derivation above is the proof.

*Relation to the multivariate probability integral transform.* Diebold, Hahn and Tay §II.B
factor the joint forecast density into conditionals, `p(y_1) p(y_2 | y_1) ⋯ p(y_N | y_{<N})`,
transform each coordinate by its conditional distribution, and show that the `N` transforms are
i.i.d. uniform when the forecast is correct, in each of the `N!` orderings. Under a Gaussian
forecast the conditional of `y_i` given `y_{<i}` is Gaussian with a variance that is the `i`-th
Cholesky pivot, so the transforms are the standard normal distribution function applied to the
coordinates of the Cholesky-whitened return. The Mahalanobis statistic `ε' ε` is the sum of the
squares of those coordinates, the same in every ordering. It is therefore the one-number summary
of the multivariate transform, and the transform is the finer diagnostic that the statistic
collapses.

*Reading.* `m_t > 1` says the realised return was larger than the forecast in the metric of
`Σ̂_t`: along some direction the forecast under-predicts, or the correlations are wrong in a way
that makes `z` improbable under `Σ̂_t`. `m_t < 1` says the forecast over-predicts. A forecast with
correct variances and wrong correlations moves `m_t` (a Gaussian `z` with a wrong correlation is
whitened into correlated coordinates whose sum of squares still has mean `N` but the wrong
variance, so the mean of `m_t` stays near one and its dispersion does not) — this is why the
distribution of `m_t`, not only its mean, is read in section 4.

### 2.2 The diagonal ratio

Definition, per asset `i`:

```text
d_{t,i} = z_{t,i}² / Σ̂_{t,ii}
```

and at `h > 1`, `d_{t,i} = S_{t,ii} / (h Σ̂_{t,ii})`.

It is the square of the standardised return `ε_t = r_t / σ_t` of Patton §1.1, and the square of
the standardised return `b_{nt} = R_{nt} / σ_{nt}` of USE4 Appendix A eq. (A1), whose "expected
standard deviation is 1" under a perfect forecast. Its mean over a window is the squared bias
statistic of USE4 eq. (A2), which "represents the ratio of realized risk to forecast risk". Its
target is one by the same second-moment identity as above, and under a Gaussian null it is
`χ²_1` per step. It reads the variances alone: a forecast with correct variances and wrong
correlations passes it on every asset. Above one, asset `i`'s variance is under-predicted.

Its regression form is the elementwise Mincer–Zarnowitz regression of Patton and Sheppard §2.3
eqs. (13)–(16), `σ̂_{ij,t} = α_{ij} + β_{ij} h_{ij,t} + e_{ij,t}` with `α = 0, β = 1` under the
null, run per element or jointly on `vech`. The ratio is the constrained form of that regression,
one number per asset per step.

### 2.3 The portfolio ratio

Definition, for a given weight vector `w`:

```text
p_t(w) = w' S_t w / w' Σ̂_t w = (w' z_t)² / w' Σ̂_t w     at h = 1
```

and at `h > 1`, `p_t(w) = w' S_t w / (h w' Σ̂_t w)`.

*Source.* Engle and Colacito §3.2 eq. (18) test "whether the portfolio variance divided by the
predicted variance has a conditional mean of 1", by the regression
`(w_t' r_t)² / (w_t' H_t w_t) − 1 = X_t β + ε_t` with `H_0: β = 0`, where `X_t` may hold an
intercept, a lag, and dummies for extreme predicted variances. Patton and Sheppard §4.1 say the
same in prose: the squared portfolio return `w' r r' w`, or the realised volatility of the
portfolio, is the proxy, and `w' H w` the forecast, evaluated by the univariate methods of their
§2. USE4 Appendix A is the standard-deviation form of the same ratio with the mean subtracted.

*Target and reading.* One, by the second-moment identity; `χ²_1` per step under a Gaussian null
with `w` fixed. Above one, the variance of the portfolio `w` is under-predicted. It reads the
forecast along one direction only, so a forecast can pass on one `w` and fail on another.

*`w` must be given, not solved from `Σ̂_t`.* USE4 §4.2 reports that with `w` optimised on the
forecast itself, all one hundred optimised portfolios had bias statistics above the confidence
interval, "indicating underprediction of risk", because the optimiser exploits the estimation
error in the forecast. That is the reason the ratio is a calibration statement only for a `w`
chosen without reading `Σ̂_t`. When `w` is solved from the forecast the right object is Engle and
Colacito's comparison instead: Theorem 1 states that the portfolio built on the true covariance
has variance no larger than the portfolio built on any other forecast, for every required-return
vector, and §3.3 eqs. (19)–(27) build the test on the difference of squared portfolio returns of
two forecasts. That is a ranking of forecasts, not a calibration of one.

### 2.4 The identity that ties the three

With the eigendecomposition `Σ̂_t = Q Λ Q'` and `q_i` the `i`-th eigenvector,

```text
z' Σ̂⁻¹ z = Σ_i (q_i' z)² / λ_i = Σ_i (q_i' S q_i) / (q_i' Σ̂ q_i) = Σ_i p_t(q_i)
```

so `m_t = (1/N) Σ_i p_t(q_i)`: the Mahalanobis ratio is the mean of the portfolio ratio over the
eigenportfolios of the forecast. USE4 §4.2 reads exactly those "eigenfactor bias statistics", one
per eigenportfolio, and finds them rising with the eigenvalue rank under a sample covariance. The
diagonal ratio is the portfolio ratio at `w = e_i`. So the three ratios are one statistic read
over the eigenbasis, over the coordinate basis, and along one chosen vector.

## 3. The loss functions

**Answer.** The library needs the Frobenius loss (the matrix MSE) and the Stein loss (the matrix
QLIKE). Both rank two forecasts consistently under a conditionally unbiased proxy; the proof is
Patton (2011) Proposition 1 in the univariate case and Laurent, Rombouts and Violante
Proposition 3 in the multivariate case. The unsquared Frobenius norm, the losses on logs, square
roots, proportions, absolute values, and correlations are not robust.

### 3.1 The univariate definitions and the robustness theorem

Patton §2 eqs. (5) and (6), with `σ̂²` the proxy and `h` the forecast:

```text
MSE:    L(σ̂², h) = (σ̂² − h)²
QLIKE:  L(σ̂², h) = log h + σ̂² / h
```

Definition 1 eq. (3): a loss is *robust* if the ranking of any two forecasts by expected loss is
the same under the true conditional variance and under any conditionally unbiased proxy.
Proposition 1 eq. (23): under A1–A5, a loss is robust if and only if

```text
L(σ̂², h) = C̃(h) + B(σ̂²) + C(h)(σ̂² − h)
```

with `C` strictly decreasing and `C̃` its antiderivative. The proof is in the Appendix. Remark 2:
MSE is `C(z) = −z`, QLIKE is `C(z) = 1/z`, `C̃(z) = log z`, `B = 0`. Proposition 2: MSE is the
only robust loss that depends on the forecast error `σ̂² − h` alone, and QLIKE the only one that
depends on the standardised error `σ̂² / h` alone. Proposition 3: a homogeneous loss ranks
invariantly to a rescaling of the data, and a robust but non-homogeneous one may not.
Proposition 4 eq. (24) gives the whole robust homogeneous family, indexed by `b`, with `b = 0`
the MSE and `b = −2` the QLIKE:

```text
L(σ̂², h; b) = (σ̂^{2b+4} − h^{b+2}) / ((b+1)(b+2)) − h^{b+1}(σ̂² − h) / (b+1),  b ∉ {−1, −2}
L(σ̂², h; −2) = σ̂² / h − log(σ̂² / h) − 1
```

Note that eq. (6) drops the terms of eq. (24) at `b = −2` that depend on the proxy alone,
`−log σ̂² − 1`. Those terms are the same for every forecast, so they cancel in a comparison. That
convention matters in the matrix case below.

Patton §2 also shows why the other common losses fail: under MSE-SD (eq. (8)) the optimal
forecast with a squared-return proxy is `(E|r|)² = (2/π) σ²` (eq. (14)), biased down by a
third, and under MSE-prop (eq. (9)) it is biased up by the kurtosis. Table 1 of the paper lists
the optimal forecast under each loss. Only MSE and QLIKE have `h* = σ²` for every proxy, which
is the necessary condition (§3 opening).

### 3.2 The multivariate forms

*Patton and Sheppard §3.6 eq. (50)*, with `Σ̂_t` the proxy and `H_t` the forecast in their
notation, is the matrix analogue of Patton's family:

```text
L(Σ̂, H; b) = tr(Σ̂^{b+2} − H^{b+2}) / (b+2) − tr(H^{b+1}(Σ̂ − H)),   b ∉ {−1, −2}
L(Σ̂, H; −2) = tr(H⁻¹ Σ̂) − log|H⁻¹ Σ̂| − K
```

They call the `b = −2` member the multivariate QLIKE and the `b = 0` member the multivariate
MSE, and note that unlike the univariate case the family does not exhaust the robust homogeneous
losses, because "there are many functions `C` that can be used to weight the forecast errors".
At `b = 0`, in this report's notation (`S` proxy, `Σ̂` forecast), expanding the traces gives

```text
½ tr(S² − Σ̂²) − tr(Σ̂(S − Σ̂)) = ½ tr((S − Σ̂)²) = ½ ‖S − Σ̂‖_F²
```

so the multivariate MSE is half the Frobenius loss. (The traces commute because both matrices
are symmetric.)

*Laurent, Rombouts and Violante* (working paper numbering). Definition 2 eq. (2) is Patton's
robustness, called *consistency of the ranking*. Proposition 1 gives a sufficient condition: the
second derivative of the loss in the proxy does not depend on the forecast. Proposition 3 eq. (4)
gives the necessary and sufficient form:

```text
L(Σ̂, H) = C̃(H) − C̃(Σ̂) + C(H)' vech(Σ̂ − H)
```

with `C̃` scalar on the positive definite matrices, `C = ∇C̃`, and the Hessian of `C̃` negative
definite; Corollary 1 eq. (5) writes it as a trace, `C̃(H) − C̃(Σ̂) + tr[C̄(H)(Σ̂ − H)]`. The
proof is in Appendix A. This is the Bregman divergence of `−C̃`, as the ticket anticipated.
Proposition 4 eq. (7): every consistent loss that depends on the forecast error alone is a
quadratic form `vech(Σ̂ − H)' Λ̂ vech(Σ̂ − H)`, homogeneous of degree two, symmetric under
`Σ̂ − H ↦ H − Σ̂`. Example 4 eq. (12): the Frobenius loss

```text
L_F = tr[(Σ̂ − H)'(Σ̂ − H)] = Σ_{i,j} (σ̂_{ij} − h_{ij})²
```

is the member with `Λ̂` diagonal weighting the off-diagonal elements twice; Remark 4 eq. (13)
derives it as the least-squares loss of a matrix-normal proxy, with `C̃(H) = −tr(H'H)`. Example
5 eq. (14): the Stein loss

```text
L_S = tr(H⁻¹ Σ̂) − log|H⁻¹ Σ̂| − N
```

is the member with `C̃(H) = log|H|`, `C̄(H) = H⁻¹`, homogeneous of degree zero, and it is the
negative log-likelihood of a Wishart proxy with mean `H`; it "is asymmetric with respect to
over/under predictions, and, in particular, underpredictions are heavily penalized". It is Patton
and Sheppard's `b = −2` member.

*What is not robust.* Laurent, Rombouts and Violante Table 1 and the text after Proposition 1:
of the losses in common use, only the Frobenius distance, the Stein loss, the Euclidean distance
and the weighted Euclidean distance satisfy Proposition 1. The entrywise 1-norm, the proportional
Frobenius, both log-Frobenius forms, and the correlation-based loss do not. Remark 3 states the
trap that matters most: the Frobenius *norm*, the square root of eq. (12), "does not satisfy
Proposition 1", and neither does the Euclidean norm. So the library must average the squared
Frobenius distance, never the norm. Proposition 2 adds the scale of the damage: the distortion of
an inconsistent loss vanishes as the proxy's conditional variance goes to zero, so it is largest
at `h = 1`, where the proxy is a single outer product.

### 3.3 The rank-one proxy and the form the library computes

At `h = 1`, `S_t = z_t z_t'` has rank one, so `log|S_t| = −∞` and the Stein loss as written in
eq. (14) is undefined. The remedy is Patton's own convention between eqs. (24) and (6): drop the
terms that depend on the proxy alone, since they are common to every forecast. Expanding
`log|Σ̂⁻¹ S| = log|S| − log|Σ̂|`, the forecast-dependent part of the Stein loss is

```text
QLIKE_t = log|Σ̂_t| + z_t' Σ̂_t⁻¹ z_t = log|Σ̂_t| + N m_t
```

which is `−2` times the Gaussian log-likelihood of `z_t` under `N(0, Σ̂_t)` up to a constant, and
which links the QLIKE to the Mahalanobis ratio by one line. At `h > 1` the same form is
`h log|Σ̂_t| + tr(Σ̂_t⁻¹ S_t)`, the sum of the per-observation QLIKEs, and it equals `h` times
the Stein loss between `S_t / h` and `Σ̂_t` plus terms in `S_t` alone. The Frobenius loss needs
no such care: at `h = 1` it is `‖z_t z_t' − Σ̂_t‖_F²`, and at `h > 1` it is `‖S_t / h − Σ̂_t‖_F²`.

Two properties, both from Patton after Proposition 2. The QLIKE's standardised error has a
variance near two under Gaussianity "regardless of the level of volatility", so the average
QLIKE is less moved by extreme observations; the MSE's error has variance proportional to `σ⁴`,
so it is dominated by the high-volatility steps and its level carries the units of the returns
to the fourth power. Patton §3 and Patton and Sheppard §3.5 add that Diebold–Mariano–West tests
under QLIKE have more power and need weaker moment conditions than under MSE.

### 3.4 Comparing two forecasts on a proxy

Patton §2: with `u_{i,t} = L(σ̂²_t, h_{i,t})` and `d_t = u_{1,t} − u_{2,t}`, the
Diebold–Mariano–West test is a Wald test of `E[d_t] = 0`. Patton and Sheppard §3.1 eqs.
(30)–(32) write it for matrices: `d_t = L(Σ̂_t, H_t^A) − L(Σ̂_t, H_t^B)`, statistic
`√M d̄ / √(avar(√M d̄))`, with a Newey–West variance, asymptotically standard normal. Both papers
take the forecasts as primitive, so the Diebold–Mariano and West forms coincide (Patton footnote
5). Engle and Colacito §3.3 eq. (23) scale their difference series by the geometric mean of the
two forecast variances, `v_t = u_t / [2(μ' H₁⁻¹ μ)(μ' H₂⁻¹ μ)]^{1/2}`, which "does not change the
null or alternative" and improves the sampling properties under heteroskedasticity; the same
scaling applies to any loss difference whose variance scales with the level of volatility, which
is the MSE's case and not the QLIKE's.

## 4. The aggregation across steps

**Answer.** Each diagnostic aggregates as a mean over the `M` steps, written as a ratio of sums so
a short last fold weights by its length. A closed-form band exists for each ratio under a Gaussian
null, and a distribution-free band comes from the sample variance of the per-step series with an
autocorrelation-consistent estimator. The median has a target below one. A loss aggregates as a
mean whose level is not interpretable on a proxy; only the difference between two forecasts is,
and that difference is tested as in §3.4.

### 4.1 The mean and its band

Write `n_t` for the number of test observations at step `t` (all `h` at a fixed horizon). The
aggregate Mahalanobis ratio is

```text
m̄ = Σ_t tr(Σ̂_t⁻¹ S_t) / (N Σ_t n_t)
```

which is the mean of `m_t` at a fixed `h`. Under a Gaussian null with the forecast correct and
the whitened returns independent across observations, `N Σ_t n_t · m̄ ~ χ²_{N Σ n_t}`, so

```text
E[m̄] = 1,   Var[m̄] = 2 / (N Σ_t n_t),   band  1 ± z_{α/2} √(2 / (N Σ_t n_t))
```

or the exact chi-squared quantiles divided by the degrees of freedom. The diagonal ratio of asset
`i` and the portfolio ratio each have `Σ_t n_t` degrees of freedom, so their band is
`1 ± z_{α/2} √(2 / Σ_t n_t)`. USE4 eq. (A3) states the standard-deviation form of the same band,
`[1 − 2/√T, 1 + 2/√T]` for "roughly 95 percent" under normality; the variance band above, taken
to its square root, is `1 ± 0.98 √(2/T) ≈ 1 ± 1.39/√T`, so USE4's rule of thumb is wider than
the delta-method band, and I report both rather than reconcile them.

*Kurtosis.* If the whitened coordinates have fourth moment `κ` (three under Gaussianity), then
`Var(ε' ε) = N(κ − 1)`, so `Var[m̄] = (κ − 1) / (N Σ n_t)` and the band widens by
`√((κ − 1) / 2)`. USE4 Appendix A1 reports the effect by simulation: at kurtosis five, only 86
percent of bias statistics fall inside the Gaussian band at `T = 120`. This is the reason a
Gaussian band is a reference and not a test.

*Distribution-free band.* Treat the per-step series `m_t − 1` as the `d_t` of a
Diebold–Mariano–West test with a zero null: the statistic is `√M (m̄ − 1) / √(avar)`, with the
Newey–West variance and a bandwidth of `h − 1` for the overlap Andersen et al. §7.1 name. This
needs no distributional assumption beyond stationarity and finite fourth moments, and it is the
band the library should report by default, with the Gaussian band beside it as the reference.

### 4.2 The median and the distribution

The median of `χ²_N / N` is below one: `0.455` at `N = 1`, and rising toward one as `N` grows
(the Wilson–Hilferty approximation puts it near `1 − 2/(9N)`). So a median of the per-step
Mahalanobis ratio is compared with the chi-squared median at the right degrees of freedom, never
with one. The same holds for the diagonal and portfolio ratios at `N = 1`.

The distribution of `m_t` is the finer diagnostic. Under the Gaussian null the transforms
`u_t = F_{χ²_N}(N m_t)` are i.i.d. uniform, which is the Diebold, Hahn and Tay §II.B test applied
to the Mahalanobis statistic rather than to each coordinate: a histogram or a quantile plot of
`u_t` against the uniform, and the autocorrelation of `u_t`, read calibration and independence
together. A forecast with correct variances and wrong correlations passes the mean and fails
this. Andersen et al. §7.1 eq. (7.2) give the regression form of the independence test: the
generalised forecast error `∂L/∂ŷ` regressed on anything known at `t` should have zero
coefficients; for the QLIKE that error is `Σ̂⁻¹ − Σ̂⁻¹ S Σ̂⁻¹`, whose trace against `Σ̂` is
`N(m_t − 1)`, so the regression of `m_t − 1` on lagged values and on the level of the forecast
is the Mincer–Zarnowitz regression of Andersen et al. eq. (7.5) and Engle and Colacito eq. (18).

### 4.3 The losses

A loss aggregates as `L̄ = (1/M) Σ_t L_t`, the `d̄_T` of Patton and Sheppard eq. (32). Patton §2
states that "the actual level of expected loss obtained using a proxy will be larger than that
which would be obtained when using the true conditional variance", so the level is not a
calibration reading; the difference between two forecasts on the same proxy is the reading, and
§3.4 tests it. The library therefore reports each loss's mean per forecast, and a comparison verb
reports the Diebold–Mariano–West statistic of the difference. A ratio of sums handles a short
last fold for the losses as for the ratios: sum the per-observation losses and divide by the
number of observations.

## 5. The online form

**Answer.** The protocol, the realised quantity, the ratios, the losses and the aggregation are
unchanged. The forecast alone differs: in the batch form the estimator refits on the fold's
training window; in the online form the estimator's Partial Fit State has folded the observations
up to `t` through `partial_fit!` (CONTEXT.md §1, ADR 0107), and a read-out verb turns the state
into the ordinary Result that `Σ̂_t` is read from. Every diagnostic is a function of `(Σ̂_t, S_t)`
and nothing else, so the online form enters the same formulas.

**The identity a test pins.** Take an estimator with an exact seam over an expanding window
(`expand_train = true`) and the same warm-up length as the batch first fold's `train_size`. Then
the state after folding observations `1..t` equals the batch fit over `1..t`, so `Σ̂_t^online =
Σ̂_t^batch` at every step, and every per-step diagnostic and every aggregate agrees because the
same inputs reach the same functions. The map's oracle statement (#861, "The oracle") scopes the
equality: "to the last bit for a sum-of-blocks statistic and to a stated tolerance otherwise".
The identity is exact in exact arithmetic for every exact seam; whether it is bit-exact in
floating point depends on whether the state's accumulator and the batch two-pass covariance round
identically, and the map assigns "last bit" to the sum-of-blocks families. ADR 0106 names
`chan_merge` as the mathematics of the merge, so the build ticket's test must pin the identity on
one sum-of-blocks family first (`GeneralCovariance`, map ground truth #2), and measure the
tolerance on the others. A state whose family refuses the merge but folds sequentially exactly
(`RegimeAdjustedVarianceState`, ADR 0106) also gives the identity, since the expanding-window
walk-forward folds in order.

**What the identity does not cover.** A rolling window (`expand_train = false`) has no exact
online form until the capped buffer of the map's capability 3 exists, because the state cannot
forget an observation. A Held Gap (CONTEXT.md §4.6, ADR 0118) is zeroed once per fold in the
batch form; the online form must zero the same observation before folding it, and the test must
pin that too.

**What the online form buys.** The `h = 1` protocol at every observation costs the batch form one
refit per observation; the online form costs one fold per observation. So the online form is what
makes the finest protocol affordable, and the mathematics of the diagnostics is indifferent to
which form produced the forecast.

## 6. The docstring form

Each block below is in the form the docstring instructions state
(`.github/instructions/julia-docstrings.instructions.md`, "Mathematical Notation"): an `align`
environment, one equation per line, a `Where:` list after the last block, existing `math_dict`
keys interpolated, and no identifier from any body. The backslashes are doubled as they are
inside a `"""` docstring. Symbols the table does not carry are given in prose; the build ticket
keys those that two or more Units share (`\\hat{\\mathbf{\\Sigma}}_t`, `\\boldsymbol{z}_t`,
`\\mathbf{S}_t`, `h`, `M` are shared by every block below, so they become keys), and notes that a
new `math_dict` key moves the copy ratchet in `test/test_26_docs.jl`.

Existing keys used: `$(math_dict[:N])` (number of assets), `$(math_dict[:w_port])` (portfolio
weights vector), `$(math_dict[:x_t_obs])` (asset returns for observation `t`),
`$(math_dict[:Sigma_hat])` (estimated covariance matrix; used where the forecast is a plain
estimate rather than a per-step one). `$(math_dict[:T])` is "number of observations" and is not
reused for the number of steps, under the one-glyph-one-key rule; the number of steps is `M`.

### 6.1 The realised covariance of a step

````julia
# Mathematical definition

```math
\\begin{align}
\\mathbf{S}_t &= \\sum_{s=1}^{h} \\boldsymbol{x}_{t+s} \\boldsymbol{x}_{t+s}^\\intercal\\,, \\\\
\\mathbb{E}\\left[\\mathbf{S}_t \\mid \\mathcal{F}_t\\right] &= h\\, \\mathbf{\\Sigma}_t\\,.
\\end{align}
```

Where:

- ``\\mathbf{S}_t``: Realised covariance of step ``t``, the sum of the outer products of the
  ``h`` returns that follow the step.
- ``h``: Horizon of a step, the number of observations the forecast is judged on.
- ``\\mathbf{\\Sigma}_t``: True conditional covariance of the returns that follow step ``t``.
- ``\\mathcal{F}_t``: Information available at step ``t``.
- $(math_dict[:x_t_obs])

The expectation holds when the ``h`` returns are serially uncorrelated, and it makes
``\\mathbf{S}_t / h`` a conditionally unbiased proxy of ``\\mathbf{\\Sigma}_t`` whatever the
distribution of the returns. At ``h = 1`` the proxy has rank one.
````

The horizon-return alternative replaces the first line by
`\\boldsymbol{z}_t = \\sum_{s=1}^{h} \\boldsymbol{x}_{t+s}` and
`\\mathbf{S}_t = \\boldsymbol{z}_t \\boldsymbol{z}_t^\\intercal`, with the same expectation.

### 6.2 The Mahalanobis ratio

````julia
# Mathematical definition

```math
\\begin{align}
m_t &= \\frac{\\operatorname{tr}\\left(\\hat{\\mathbf{\\Sigma}}_t^{-1} \\mathbf{S}_t\\right)}{N h}\\,, \\\\
\\bar{m} &= \\frac{\\sum_{t=1}^{M} \\operatorname{tr}\\left(\\hat{\\mathbf{\\Sigma}}_t^{-1} \\mathbf{S}_t\\right)}{N \\sum_{t=1}^{M} h_t}\\,.
\\end{align}
```

Where:

- ``m_t``: Mahalanobis ratio of step ``t``.
- ``\\bar{m}``: Mahalanobis ratio over the walk-forward.
- ``\\hat{\\mathbf{\\Sigma}}_t``: Covariance forecast formed at step ``t``.
- ``\\mathbf{S}_t``: Realised covariance of step ``t``.
- ``h_t``: Horizon of step ``t``, equal to ``h`` at every step but a shortened last one.
- ``M``: Number of steps of the walk-forward.
- $(math_dict[:N])

When the forecast is the true covariance, ``\\mathbb{E}[m_t] = 1`` by
``\\operatorname{tr}(\\hat{\\mathbf{\\Sigma}}_t^{-1} \\mathbb{E}[\\mathbf{S}_t]) = N h``, whatever the
distribution of the returns. When the returns are further Gaussian,
``N h\\, m_t \\sim \\chi^2_{N h}`` and ``N \\sum_t h_t\\, \\bar{m} \\sim \\chi^2_{N \\sum_t h_t}``,
so ``\\bar{m}`` lies in ``1 \\pm z_{\\alpha/2} \\sqrt{2 / (N \\sum_t h_t)}`` with probability
``1 - \\alpha``. A ratio above one is an under-prediction of the dispersion in the metric of the
forecast, and a ratio below one an over-prediction. With ``\\hat{\\mathbf{\\Sigma}}_t = \\mathbf{Q}
\\mathbf{\\Lambda} \\mathbf{Q}^\\intercal``, ``m_t`` is the mean over the eigenvectors
``\\boldsymbol{q}_i`` of the portfolio ratio at ``\\boldsymbol{w} = \\boldsymbol{q}_i``.
````

### 6.3 The diagonal ratio

````julia
# Mathematical definition

```math
\\begin{align}
d_{t,i} &= \\frac{(\\mathbf{S}_t)_{ii}}{h\\, (\\hat{\\mathbf{\\Sigma}}_t)_{ii}}\\,, \\\\
\\bar{d}_i &= \\frac{\\sum_{t=1}^{M} (\\mathbf{S}_t)_{ii}}{\\sum_{t=1}^{M} h_t\\, (\\hat{\\mathbf{\\Sigma}}_t)_{ii}}\\,.
\\end{align}
```

Where:

- ``d_{t,i}``: Diagonal ratio of asset ``i`` at step ``t``.
- ``\\bar{d}_i``: Diagonal ratio of asset ``i`` over the walk-forward.
- ``\\hat{\\mathbf{\\Sigma}}_t``: Covariance forecast formed at step ``t``.
- ``\\mathbf{S}_t``: Realised covariance of step ``t``.
- ``h_t``: Horizon of step ``t``.
- ``M``: Number of steps of the walk-forward.

``d_{t,i}`` is the squared standardised return of asset ``i``, and it reads the forecast
variances alone: a forecast with the true variances and any correlations has
``\\mathbb{E}[d_{t,i}] = 1``. Under Gaussian returns ``h\\, d_{t,i} \\sim \\chi^2_{h}``, so
``\\bar{d}_i`` lies in ``1 \\pm z_{\\alpha/2} \\sqrt{2 / \\sum_t h_t}`` with probability
``1 - \\alpha``.
````

### 6.4 The portfolio ratio

````julia
# Mathematical definition

```math
\\begin{align}
p_t &= \\frac{\\boldsymbol{w}^\\intercal \\mathbf{S}_t \\boldsymbol{w}}{h\\, \\boldsymbol{w}^\\intercal \\hat{\\mathbf{\\Sigma}}_t \\boldsymbol{w}}\\,, \\\\
\\bar{p} &= \\frac{\\sum_{t=1}^{M} \\boldsymbol{w}^\\intercal \\mathbf{S}_t \\boldsymbol{w}}{\\sum_{t=1}^{M} h_t\\, \\boldsymbol{w}^\\intercal \\hat{\\mathbf{\\Sigma}}_t \\boldsymbol{w}}\\,.
\\end{align}
```

Where:

- ``p_t``: Portfolio ratio at step ``t``.
- ``\\bar{p}``: Portfolio ratio over the walk-forward.
- ``\\hat{\\mathbf{\\Sigma}}_t``: Covariance forecast formed at step ``t``.
- ``\\mathbf{S}_t``: Realised covariance of step ``t``.
- ``h_t``: Horizon of step ``t``.
- ``M``: Number of steps of the walk-forward.
- $(math_dict[:w_port])

``p_t`` is the ratio of the realised to the forecast variance of the portfolio
``\\boldsymbol{w}``, and it reads the forecast along that one direction. Its target is one when
``\\boldsymbol{w}`` does not depend on ``\\hat{\\mathbf{\\Sigma}}_t``; a ``\\boldsymbol{w}`` solved
from the forecast reads the estimation error the solver exploited, and sits above one. Under
Gaussian returns ``h\\, p_t \\sim \\chi^2_{h}``, so ``\\bar{p}`` lies in
``1 \\pm z_{\\alpha/2} \\sqrt{2 / \\sum_t h_t}`` with probability ``1 - \\alpha``.
````

### 6.5 The QLIKE loss

````julia
# Mathematical definition

```math
\\begin{align}
L^{\\mathrm{QLIKE}}_t &= h \\log\\left|\\hat{\\mathbf{\\Sigma}}_t\\right| + \\operatorname{tr}\\left(\\hat{\\mathbf{\\Sigma}}_t^{-1} \\mathbf{S}_t\\right)\\,, \\\\
\\bar{L}^{\\mathrm{QLIKE}} &= \\frac{1}{\\sum_{t=1}^{M} h_t} \\sum_{t=1}^{M} L^{\\mathrm{QLIKE}}_t\\,.
\\end{align}
```

Where:

- ``L^{\\mathrm{QLIKE}}_t``: QLIKE loss of step ``t``.
- ``\\bar{L}^{\\mathrm{QLIKE}}``: QLIKE loss per observation over the walk-forward.
- ``\\hat{\\mathbf{\\Sigma}}_t``: Covariance forecast formed at step ``t``.
- ``\\mathbf{S}_t``: Realised covariance of step ``t``.
- ``h_t``: Horizon of step ``t``.
- ``M``: Number of steps of the walk-forward.

``L^{\\mathrm{QLIKE}}_t`` is the Stein loss between ``\\mathbf{S}_t / h`` and
``\\hat{\\mathbf{\\Sigma}}_t``, scaled by ``h`` and stripped of the terms that depend on
``\\mathbf{S}_t`` alone, so it is finite at ``h < N`` where ``\\mathbf{S}_t`` is singular, and the
difference of the losses of two forecasts on one realised covariance is unchanged. It equals
``-2`` times the Gaussian log-likelihood of the ``h`` returns under the forecast, up to a
constant, and ``L^{\\mathrm{QLIKE}}_t = h \\log|\\hat{\\mathbf{\\Sigma}}_t| + N h\\, m_t``. Its
expected value is minimised by the true covariance under any conditionally unbiased proxy, so
it ranks two forecasts the same on the proxy as on the truth. It is homogeneous of degree zero,
so a rescaling of the returns shifts it by a constant.
````

### 6.6 The Frobenius loss

````julia
# Mathematical definition

```math
\\begin{align}
L^{\\mathrm{F}}_t &= \\left\\lVert \\frac{\\mathbf{S}_t}{h} - \\hat{\\mathbf{\\Sigma}}_t \\right\\rVert_F^2
    = \\sum_{i=1}^{N} \\sum_{j=1}^{N} \\left(\\frac{(\\mathbf{S}_t)_{ij}}{h} - (\\hat{\\mathbf{\\Sigma}}_t)_{ij}\\right)^2\\,, \\\\
\\bar{L}^{\\mathrm{F}} &= \\frac{1}{M} \\sum_{t=1}^{M} L^{\\mathrm{F}}_t\\,.
\\end{align}
```

Where:

- ``L^{\\mathrm{F}}_t``: Frobenius loss of step ``t``.
- ``\\bar{L}^{\\mathrm{F}}``: Frobenius loss over the walk-forward.
- ``\\hat{\\mathbf{\\Sigma}}_t``: Covariance forecast formed at step ``t``.
- ``\\mathbf{S}_t``: Realised covariance of step ``t``.
- ``h``: Horizon of a step.
- ``M``: Number of steps of the walk-forward.
- $(math_dict[:N])

``L^{\\mathrm{F}}_t`` is the matrix form of the mean squared error, and it weights each
off-diagonal error twice, once in each triangle. Its expected value is minimised by the true
covariance under any conditionally unbiased proxy, so it ranks two forecasts the same on the
proxy as on the truth; its square root, the Frobenius norm, does not. It is homogeneous of degree
two, so it carries the units of the returns to the fourth power and is dominated by the steps of
highest volatility.
````

### 6.7 The robust homogeneous family

The two losses are the `b = −2` and `b = 0` members of one family, and a docstring that offers
the family states it once:

````julia
# Mathematical definition

```math
\\begin{align}
L_t(b) &= \\frac{\\operatorname{tr}\\left(\\mathbf{P}_t^{b+2} - \\hat{\\mathbf{\\Sigma}}_t^{b+2}\\right)}{b + 2}
    - \\operatorname{tr}\\left(\\hat{\\mathbf{\\Sigma}}_t^{b+1} \\left(\\mathbf{P}_t - \\hat{\\mathbf{\\Sigma}}_t\\right)\\right)\\,,
    \\quad b \\notin \\{-1, -2\\}\\,, \\\\
L_t(-2) &= \\operatorname{tr}\\left(\\hat{\\mathbf{\\Sigma}}_t^{-1} \\mathbf{P}_t\\right)
    - \\log\\left|\\hat{\\mathbf{\\Sigma}}_t^{-1} \\mathbf{P}_t\\right| - N\\,, \\\\
\\mathbf{P}_t &= \\frac{\\mathbf{S}_t}{h}\\,.
\\end{align}
```

Where:

- ``L_t(b)``: Loss of step ``t`` in the robust homogeneous family with shape ``b``.
- ``\\mathbf{P}_t``: Realised covariance of step ``t`` per observation.
- ``\\hat{\\mathbf{\\Sigma}}_t``: Covariance forecast formed at step ``t``.
- ``\\mathbf{S}_t``: Realised covariance of step ``t``.
- ``h``: Horizon of a step.
- $(math_dict[:N])

``L_t(0) = \\tfrac{1}{2} L^{\\mathrm{F}}_t`` and ``L_t(-2)`` is the Stein loss, whose
``\\log|\\mathbf{P}_t|`` term is dropped in ``L^{\\mathrm{QLIKE}}_t``. Every member is homogeneous of
degree ``b + 2`` and ranks two forecasts the same on a conditionally unbiased proxy as on the
truth; ``b < 0`` penalises under-prediction more than over-prediction, and ``b > 0`` the reverse.
The family does not exhaust the robust homogeneous losses on matrices.
````

### 6.8 The comparison of two forecasts

````julia
# Mathematical definition

```math
\\begin{align}
\\delta_t &= L^{A}_t - L^{B}_t\\,, \\\\
\\bar{\\delta} &= \\frac{1}{M} \\sum_{t=1}^{M} \\delta_t\\,, \\\\
\\mathrm{DMW} &= \\frac{\\sqrt{M}\\, \\bar{\\delta}}{\\sqrt{\\hat{\\omega}^2}}\\,, \\\\
\\hat{\\omega}^2 &= \\hat{\\gamma}_0 + 2 \\sum_{k=1}^{h-1} \\left(1 - \\frac{k}{h}\\right) \\hat{\\gamma}_k\\,.
\\end{align}
```

Where:

- ``\\delta_t``: Difference of the losses of forecasts ``A`` and ``B`` at step ``t``.
- ``\\bar{\\delta}``: Mean loss difference over the walk-forward.
- ``\\mathrm{DMW}``: Diebold–Mariano–West statistic.
- ``\\hat{\\omega}^2``: Long-run variance of ``\\delta_t``, the Bartlett-kernel estimate with
  ``h - 1`` lags.
- ``\\hat{\\gamma}_k``: Sample autocovariance of ``\\delta_t`` at lag ``k``.
- ``h``: Horizon of a step.
- ``M``: Number of steps of the walk-forward.

Under equal expected loss, ``\\mathrm{DMW}`` is asymptotically standard normal. A positive value
says forecast ``A`` lost more than forecast ``B``. The ``h - 1`` lags absorb the overlap of
consecutive steps that share observations.
````

---

## Tables

### Table 1. The per-step diagnostics

| Diagnostic | Per step, `h = 1` | Per step, `h > 1` | Target | Above one | Source |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Mahalanobis ratio | `z' Σ̂⁻¹ z / N` | `tr(Σ̂⁻¹ S) / (N h)` | 1; `χ²_{Nh}/(Nh)` Gaussian | under-prediction in the `Σ̂` metric | second-moment identity; Patton and Sheppard §1.1 (standardised vector); Diebold, Hahn and Tay §II.B |
| Diagonal ratio | `z_i² / Σ̂_ii` | `S_ii / (h Σ̂_ii)` | 1; `χ²_h/h` Gaussian | variance of `i` under-predicted | Patton §1.1 (`ε_t`); USE4 App. A eqs. (A1)–(A2); Patton and Sheppard §2.3 eqs. (13)–(16) |
| Portfolio ratio | `(w'z)² / w'Σ̂w` | `w'Sw / (h w'Σ̂w)` | 1; `χ²_h/h` Gaussian, `w` fixed | portfolio variance under-predicted | Engle and Colacito §3.2 eq. (18); Patton and Sheppard §4.1; USE4 App. A |
| QLIKE loss | `log|Σ̂| + z'Σ̂⁻¹z` | `h log|Σ̂| + tr(Σ̂⁻¹S)` | none; difference only | — | Patton eq. (6), Prop. 4 eq. (24) at `b = −2`; Patton and Sheppard eq. (50); LRV Ex. 5 eq. (14) |
| Frobenius loss | `‖zz' − Σ̂‖_F²` | `‖S/h − Σ̂‖_F²` | none; difference only | — | Patton eq. (5); Patton and Sheppard eq. (50) at `b = 0`; LRV Ex. 4 eq. (12) |

### Table 2. Robustness of the losses to a conditionally unbiased proxy

| Loss | Robust | Where proved | Note |
| :--- | :--- | :--- | :--- |
| MSE `(σ̂² − h)²` | yes | Patton Prop. 1, Prop. 2(i) | only robust loss in the error alone |
| QLIKE `log h + σ̂²/h` | yes | Patton Prop. 1, Prop. 2(ii) | only robust loss in the ratio alone |
| Patton family `L(·; b)` | yes | Patton Prop. 4 | all robust homogeneous univariate losses |
| Frobenius distance `‖·‖_F²` | yes | LRV Prop. 4, Ex. 4, Rem. 4 | off-diagonals weighted twice |
| Euclidean, weighted Euclidean on `vech` | yes | LRV Prop. 4, Ex. 1–2 | |
| Stein `tr(H⁻¹Σ̂) − log|H⁻¹Σ̂| − N` | yes | LRV Prop. 3, Ex. 5 | Wishart likelihood; degree zero |
| Patton–Sheppard matrix family eq. (50) | yes | Patton and Sheppard §3.6 eqs. (47)–(49) | does not exhaust the class |
| Bregman form `C̃(H) − C̃(Σ̂) + C(H)'vech(Σ̂ − H)` | yes, iff | LRV Prop. 3 eq. (4), App. A | the whole class |
| Frobenius *norm* `‖·‖_F` | no | LRV Rem. 3 | the square root breaks it |
| MSE on standard deviations | no | Patton §2.1 eq. (14) | optimum `(2/π)σ²` |
| MSE-LOG, MAE, MAE-LOG, MAE-SD, proportional | no | Patton §2.1, Table 1 | |
| Log-Frobenius, proportional Frobenius, entrywise 1-norm | no | LRV Table 1, Rem. 3 | |
| Correlation-based losses | no | LRV Table 1 | |

### Table 3. Aggregates and bands

| Aggregate | Form | Gaussian null | Band at level `α` | Distribution-free |
| :--- | :--- | :--- | :--- | :--- |
| `m̄` | ratio of sums, §4.1 | `χ²_{NΣh}/(NΣh)` | `1 ± z_{α/2}√(2/(NΣh))` | DMW on `m_t − 1`, `h − 1` lags |
| `d̄_i` | ratio of sums | `χ²_{Σh}/Σh` | `1 ± z_{α/2}√(2/Σh)` | DMW on `d_{t,i} − 1` |
| `p̄` | ratio of sums | `χ²_{Σh}/Σh` | `1 ± z_{α/2}√(2/Σh)` | DMW on `p_t − 1` |
| median of `m_t` | order statistic | median of `χ²_{Nh}/(Nh)`, below one | quantile of that law | — |
| distribution of `m_t` | `u_t = F_{χ²}(Nh m_t)` | i.i.d. uniform | uniform quantile plot; autocorrelation | Diebold, Hahn and Tay §II.B |
| `L̄` per forecast | mean | level uninterpretable on a proxy | — | Patton §2 |
| `δ̄` between forecasts | mean of differences | `DMW ~ N(0, 1)` | `|DMW| > z_{α/2}` | Patton and Sheppard eqs. (30)–(32) |

### Table 4. What a later ticket must measure

| Question | Why the sources do not settle it |
| :--- | :--- |
| Whether the online `Σ̂_t` is bit-identical to the batch `Σ̂_t` for each seam family | the map scopes "last bit" to sum-of-blocks statistics; rounding order is an implementation fact |
| Whether `z_t` is centred at the training mean | Engle and Colacito Thms. 2–3 make both consistent; the choice is a decision |
| Which of realised covariance and horizon return is the default at `h > 1` | both are in the literature; §1 recommends the former for its lower noise |
| The kurtosis of the whitened returns on the library's example data | sets the width of the honest band, §4.1 |

## References

Entries in the shape `docs/src/References.bib` uses. `andersen2006` already exists there and is
not repeated.

```bibtex
@article{patton2011,
  title     = {Volatility forecast comparison using imperfect volatility proxies},
  author    = {Patton, Andrew J.},
  journal   = {Journal of Econometrics},
  volume    = {160},
  number    = {1},
  pages     = {246--256},
  year      = {2011},
  publisher = {Elsevier}
}
@incollection{pattonsheppard2009,
  title     = {Evaluating volatility and correlation forecasts},
  author    = {Patton, Andrew J. and Sheppard, Kevin},
  booktitle = {Handbook of Financial Time Series},
  editor    = {Andersen, Torben G. and Davis, Richard A. and Kreiss, Jens-Peter and Mikosch, Thomas},
  pages     = {801--838},
  year      = {2009},
  publisher = {Springer},
  address   = {Berlin, Heidelberg}
}
@article{laurent2013,
  title     = {On loss functions and ranking forecasting performances of multivariate volatility models},
  author    = {Laurent, S{\'e}bastien and Rombouts, Jeroen V. K. and Violante, Francesco},
  journal   = {Journal of Econometrics},
  volume    = {173},
  number    = {1},
  pages     = {1--10},
  year      = {2013},
  publisher = {Elsevier}
}
@article{englecolacito2006,
  title     = {Testing and valuing dynamic correlations for asset allocation},
  author    = {Engle, Robert F. and Colacito, Riccardo},
  journal   = {Journal of Business \& Economic Statistics},
  volume    = {24},
  number    = {2},
  pages     = {238--253},
  year      = {2006},
  publisher = {Taylor \& Francis}
}
@article{diebold1999,
  title     = {Multivariate density forecast evaluation and calibration in financial risk management: high-frequency returns on foreign exchange},
  author    = {Diebold, Francis X. and Hahn, Jinyong and Tay, Anthony S.},
  journal   = {The Review of Economics and Statistics},
  volume    = {81},
  number    = {4},
  pages     = {661--673},
  year      = {1999},
  publisher = {MIT Press}
}
@techreport{menchero2011,
  title       = {The Barra US Equity Model (USE4): Methodology Notes},
  author      = {Menchero, Jose and Orr, D. J. and Wang, Jun},
  year        = {2011},
  institution = {MSCI},
  type        = {Model Insight}
}
@book{anderson2003,
  title     = {An Introduction to Multivariate Statistical Analysis},
  author    = {Anderson, Theodore W.},
  edition   = {3},
  year      = {2003},
  publisher = {John Wiley \& Sons},
  address   = {Hoboken, NJ}
}
```

The page range of `pattonsheppard2009` is the published chapter's as the publisher lists it; the
equation numbers cited above are those of the Oxford working paper 2008fe22, and the build ticket
that copies a citation into a docstring should check them against the chapter. The same holds for
`laurent2013` against the CIRANO working paper 2009s-45.
