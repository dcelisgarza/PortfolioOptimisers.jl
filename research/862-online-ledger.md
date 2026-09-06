# 862 — The ledger of the seam: which estimators fold exactly, which merge, and which refit

Research ticket #862 of wayfinder map #861. Written 2026-09-06.

Sources are the code and docstrings of `src/` on branch `research/online-ledger`, cut from
`dev` at `a003ca910b`; ADR 0106 and ADR 0107 under `docs/adr/`; `CONTEXT.md`; the resolution
comments of #308 and #701; ticket #854; `research/prototypes/19_online_moments.jl` and section
3.13 of `research/PortfolioOptimisers_Improvements_9.md`; and the papers named in prose below.
No Julia was run. Every line number is a claim about that commit.

---

## Summary

- **Ground truth 2 of the map is stale.** Eight estimators carry the seam, not seven.
  `RegimeAdjustedExpWeightedCovariance` has `partial_fit!`
  (`src/08_Moments/37_RegimeAdjustedExpWeightedCovariance.jl:1725`, `:1788`), the one-argument
  `cov(ce)` and `cor(ce)` read-outs (`:1892`, `:1986`), a `merge_states` refusal (`:2020`) and
  `copy` (`:2048`). #711 is closed by the code, whatever its issue state says. The directory
  holds 47 files and two subdirectories (13 and 6 files), not 40.
- **Six members fold exactly and merge.** `SimpleExpectedReturns`, `GeneralCovariance`,
  `Covariance{FullMoment}`, `SimpleVariance`, `Coskewness{FullMoment}` and
  `Cokurtosis{FullMoment}`. All six are Welford's update at the observation and Chan's merge at
  the block; the higher two shift their accumulators to a common centre before adding, which is
  the Welford recursion carried to orders three and four. The formula is the stable one prototype
  19 measured at `7.37e-14`.
- **Two members fold exactly and refuse the merge**, the regime-adjusted variance and covariance.
  Their exponentially weighted accumulator folds as `decay^n_B v_A + v_B`, which #701 measured to
  `3.5e-18`, but the regime state is gated by the running count and reads a standardised
  innovation, so a cold block records less than the same block after another. The route is the
  sequential fold, which is exact.
- **The three exponentially weighted estimators of #854 are pending and will be exact.** Their
  recursion is the one `ew_mean_series` and `process_observation!` already run. The raw
  accumulator of a centred, zero-seeded recursion merges as `decay^n_B S_A + S_B` per asset, and
  a running location or a HAC buffer refuses, as #701 measured. The build ticket must pin which.
- **Every composed moment, prior, uncertainty set and matrix transform that reads a moment alone
  is exact by composition, and owes no state.** It owes a read-out method that applies its
  transform to the inner state's answer, which #308 named for the covariance transforms. The
  count reaches the layer above through `n` in the state, so `Denoise`'s `q = T / N`, the
  Black-Litterman `omega`, the normal set's `T` and the shrinkage intensities all fold.
- **Eighteen moment members refit**, for six reasons: a threshold or band that reads the
  whole-sample dispersion (Gerber, Smyth-Broby, Gerber IQ), a double centring or a rank over all
  pairs of observations (distance correlation, Kendall, Spearman), a quantile or an order
  statistic (lower tail dependence, median), a histogram (mutual information), a clamp about a
  batch centre (the three `SemiMoment` arms), an external series (implied volatility) and a
  removal (the five windowed wrappers). Kendall alone has a known exact `O(T)` update from the
  buffer; the median has the approximate P² recursion; nothing else has a recursion worth naming.
- **Nine of twelve priors refit**, because they run a time-series regression (`FactorPrior`, the
  three factor Black-Litterman priors, `HighOrderFactorPriorEstimator`), an optimisation over the
  observation weights (both entropy pooling priors, `OpinionPoolingPrior`), or a per-observation
  pipeline of descriptors, regressions and forecasts whose components take different routes
  (`CrossSectionalFactorPrior`). `EmpiricalPrior`, `HighOrderPriorEstimator` and
  `BlackLittermanPrior` are exact by composition.
- **Three of five uncertainty-set estimators are exact by composition** (`Delta`, `Normal`,
  `Characteristic`); the bootstrap set resamples the rows of `pr.X` and the orthogonal set reads
  a regression block, so both refit. Nine of eleven calibration rules read counts and moments;
  the two tail-decay rules read order statistics of a series and refit.
- **An optimiser reads the sample through exactly four doors**: a risk measure whose
  `risk_input_kind` is `NetReturnsInput` or `WeightsReturnsFeesInput`, the `LogarithmicReturn`
  term, the hierarchical family's `X = pr.X`, and a meta-optimiser's cross-validation. Six risk
  measures read a moment only (`Variance`, `StandardDeviation`, `UncertaintySetVariance`,
  `NegativeSkewness`, `TurnoverRiskMeasure`, `EqualRisk`), and `ArithmeticReturn` reads
  `pr.mu`. An optimiser built from those is exact by composition; every other configuration
  refits from the buffer `LowOrderPrior.X` already carries. The finite allocators read no
  sample and owe no method.
- **The cross-sectional regression is the one estimator whose "merge" is concatenation.** One
  observation is one solve, the histories `f`, `eps`, `n` and `b` grow by one row, and a block
  fitted alone equals the same block fitted after another. It is exact and it merges, and it is
  the natural spine of an online `CrossSectionalFactorPrior`.

---

## 1. The vocabulary and the seam as it stands

An **Estimator** is configuration and a **Result** is data (`CONTEXT.md`). A **Partial Fit
State** is the one Result an Estimator holds, in a `cache` field bound to
`Option{<:AbstractPartialFitState}`, and ADR 0106 records the exception. `partial_fit!` is the
method each family writes and its cheapest exact fold; `partial_fit` is one generic method with
value semantics (`src/01_Base/14_PartialFit.jl:102`); every state answers `merge_states` and
`copy` (ADR 0107). `chan_merge` at `14_PartialFit.jl` carries the merge once for every
second-order family, and `assert_mergeable_states` refuses a pair of different types or shapes.
`obs_weights_view` drops a state (`14_PartialFit.jl`, the root method returning `nothing`) and
`port_opt_view` slices it by index copy.

The membership rule of #308 is the rule this ledger applies: *the statistic must have an exact
update from the state plus one new observation, and no observation may ever leave the sample.*
A member whose fold is exact but whose state is not a sufficient statistic for its block takes the
middle route, **exact but no merge**, which #701 measured on the regime-adjusted variance.

Two refusals sit on every second-order and higher-order member and are worth stating once, so the
tables need not repeat them. `assert_partial_fittable`
(`src/08_Moments/40_HigherMomentPartialFit.jl`, the method before `merge_states` at `:323`)
refuses observation weights, because a weights vector describes a sample of fixed length and a
`DynamicAbstractWeights` derives every weight from that length, and refuses a centring estimator
other than an unweighted `SimpleExpectedReturns`, because the state carries the running sample
mean and no other centre. So `w = nothing` and `me = SimpleExpectedReturns()` are conditions of
every "exact" verdict below.

## 2. The routes, and what each one costs the layer above

| Route | Meaning | What the member owes |
| --- | --- | --- |
| **exact** | A recursion folds one observation into the state and reproduces the batch answer; two block states merge into the state of the concatenated block | `partial_fit!`, `merge_states`, `copy`, `port_opt_view`, a read-out; or, for a composed member, a read-out that applies its transform to the inner state's answer |
| **exact but no merge** | The fold is exact but the state of a cold block is not what that block contributes after another | The same, with a `merge_states` that refuses and names the reason (ADR 0106) |
| **refit** | No exact recursion exists; the member fits again from a buffer of the observations seen so far | Nothing on the seam; the buffer is `LowOrderPrior.X`, and the docstring says so |

"Exact by composition" is the **exact** route for a member that holds no state of its own. It
reads an inner estimator's state through the inner read-out and applies its transform. #308 named
this route for `DenoiseCovariance`, `DetoneCovariance`, `ProcessedCovariance`,
`CorrelationCovariance`, the shrunk means and `PortfolioOptimisersCovariance`. This ledger extends
it to every layer whose inputs are moments and counts.

## 3. Numerical stability, once for the whole ledger

Prototype 19 (`research/prototypes/19_online_moments.jl`, section 3.13 of
`research/PortfolioOptimisers_Improvements_9.md:640-668`) measured three formulas on data with
mean 1000 and unit spread:

- **Welford's update**, `d = x - mu_old; mu = mu_old + d / n; M = M + d (x - mu_new)'`, error
  `7.37e-14`. The first factor reads the old mean and the second the new one, and that asymmetry
  is what keeps `M` positive semi-definite. Welford (1962), Technometrics 4(3).
- **The textbook formula**, `M = sum(x x') - n mu mu'`, error `6.38e-10`, 8653 times worse, and
  Chan, Golub and LeVeque (1983) show it can return a negative variance on real data. That paper
  is `chan1983` in `docs/src/References.bib`.
- **Chan's merge**, `mu = mu_A + delta n_B / n; M = M_A + M_B + delta delta' n_A n_B / n`, equal
  to the sequential answer to `1.15e-13`.

Every "Welford" entry below is the first formula, and every "Chan" entry the third; the library
writes them at `src/08_Moments/03_Covariance.jl:914` and `src/01_Base/14_PartialFit.jl`
(`chan_merge`). The higher-moment shifts (`shift_comoment3`, `shift_comoment4`,
`40_HigherMomentPartialFit.jl` between `:130` and `:320`) are the same construction one and two
orders up: a block records its co-moments about its own mean, and the merge moves each block to
the common centre before adding, which is the one-pass parallel formula Pébay (2008, Sandia
report SAND2008-6212) gives for arbitrary order. The scalar cases the docstrings state,
`M3 <- M3 - 3 a M2 - m a^3` and `M4 <- M4 - 4 a M3 + 6 a^2 M2 + m a^4`, are the univariate
Welford recursion. An **exponentially weighted** recursion `S <- lambda S + (1 - lambda) x` is a
convex combination and cancels nothing; its one hazard is the bias correction `1 / (1 - lambda^n)`
at small `n`, which the library floors at `eps` (`36_…:761`). So no member of this ledger runs
the textbook formula, and the stability column reads "Welford", "Chan", "convex" or "n/a".

---

## 4. The moments, `src/08_Moments/`

The columns are the six items of the ticket: the statistic; the exact fold and its source; the
merge; the stability formula; the reason no fold exists and any known approximation; the route.

### 4.1 The seam members, and the three pending

| Member (file:line) | Statistic | Exact fold, source | Merge | Stability | No fold / approximation | Route |
| --- | --- | --- | --- | --- | --- | --- |
| `SimpleExpectedReturns` (`02_:68`, state `:255`) | Sample mean per asset | Welford mean, `partial_fit!` `02_:383` (`n += 1; mu += (x - mu) / n`) | Chan on `(n, mu)`, `02_:299`; accumulator argument `false` | Welford | — | **exact** |
| `GeneralCovariance` (`03_:77`, state `:693`) | Sample covariance, bias-corrected by the inner `StatsBase.SimpleCovariance` | Welford covariance, `03_:914`; block arm `:954`, vector arm `:973` | Chan with the outer-product method, `03_:741` | Welford | — | **exact** |
| `Covariance{FullMoment}` (`03_:338`) | Sample covariance about `me`'s mean | Same state and fold, arms `03_:1008`, `:1027` | Chan, `03_:741` | Welford | — | **exact** |
| `SimpleVariance` (`04_:87`, state `:621`) | Sample variance per asset | Welford per-asset, `04_:759`; arms `:799`, `:818` | Chan with the elementwise method, `04_:669` | Welford | — | **exact** |
| `Coskewness{FullMoment}` (`19_:205`, state `40_:22`) | Third central co-moment tensor `N × N²` | One-row block through `comoment_block` and `shift_comoment3`, `40_:543`, `:580` | Chan on `(n, mu, M2)`, then shift-and-add on `M3`, `40_:323`; asserted associative over three blocks (ADR 0107) | Welford at order 3 (Pébay 2008) | — | **exact** |
| `Cokurtosis{FullMoment}` (`20_:197`, state `40_:63`) | Fourth central co-moment matrix `N² × N²` | Same, with `shift_comoment4`, `40_:696`, `:733`; `partial_fit` overridden because the state is 800 MB at 100 assets | Shift-and-add on `M4`, `40_:418` | Welford at order 4 | — | **exact** |
| `RegimeAdjustedExpWeightedVariance` (`36_:421`, state `:527`) | EW variance per asset, bias-corrected, scaled by a squared regime multiplier read off EW standardised squared innovations | `process_observation!` `36_:717`; `partial_fit!` `:1126`, `:1187`; doctest at `:1110` pins `isequal(var(partial_fit!(ce, X)), var(ce, X))` | **Refused**, `36_:1498`. Variance folds as `decay^n_B v_A + v_B` to `3.5e-18` (#701); `n_regime_obs` loses exactly `min_obs`; uncentred misses `1.5e-3`, HAC `1.0e-3` | Convex; `z² = x² / var_corrected` divides by a variance floored at `min_val` | — | **exact but no merge** |
| `RegimeAdjustedExpWeightedCovariance` (`37_:287`, state `:418`) | EW covariance, optionally at a separate correlation decay with a pairwise count, scaled by a regime multiplier from a Mahalanobis, diagonal or portfolio statistic | `process_observation!` `37_:1224`, `update_var_cor!` `:986`; `partial_fit!` `:1725`, `:1788`; read-outs `cov(ce)` `:1892`, `cor(ce)` `:1986` | **Refused**, `37_:2020`: the regime statistic scores each observation against the state before it, gated by the running count; the separate correlation state carries its running variance | Convex; the separate path re-symmetrises and clips `rho` to `[-1, 1]` at every step | — | **exact but no merge** |
| EW expected returns (#854, pending) | EW mean per asset with per-asset count, freeze on holiday, reset on inactive, warm-up `NaN`, correction `1 / (1 - λⁿ)` | The recursion of `ew_mean_series`, `42_FactorExposures/04_:188`, and of `process_observation!`'s location line `36_:748` | The zero-seeded accumulator merges as `λ^{n_B} S_A + S_B` per asset, with `n_B` from the per-asset count; exact by the identity #701 measured on the variance. Confirm on the built code | Convex | — | **exact**, merge pending measurement |
| EW variance (#854, pending) | EW variance per asset, same contract | `process_observation!` `36_:767` with `regime_method = nothing`; `ew_variance_estimator` at `42_…/05_:36` already builds exactly this | Same identity when `centred = true` and `hac_lags = 0`; a running location or a HAC buffer refuses (#701) | Convex | — | **exact**, merge conditional |
| EW covariance (#854, pending) | EW covariance with correction `1 / sqrt((1 - λⁱ)(1 - λʲ))`, re-symmetrised on the active block | `37_:1224` with `regime_method = nothing` | Same identity on the matrix accumulator; the pairwise count is a sum of blocks; the separate-decay correlation state is normalised by a running variance and refuses | Convex | — | **exact**, merge conditional |

### 4.2 Composed members, exact by composition

Each of these reads a moment its inner estimator produces and a count, and holds no state. Its
online form is a read-out method that applies the transform to the inner state's answer. The
merge and stability columns are the inner member's.

| Member (file:line) | Statistic | Reads | Route |
| --- | --- | --- | --- |
| `PortfolioOptimisersCovariance` (`15_:63`, `cov` `:121`) | Inner covariance, then `matrix_processing!` (posdef, denoise, detone, algorithm) | `sigma`; `Denoise` reads `q = T / N`, which is `n` in the state | **exact** by composition |
| `DenoiseCovariance`, `DetoneCovariance`, `ProcessedCovariance` (`12_:79`, `13_:73`, `14_:71`) | Constructors of the row above with a fixed `order` | as above | **exact** by composition |
| `CorrelationCovariance` (`25_:54`, `cov` `:98`) | `cor(ce.ce, X)` returned as the covariance | `rho` | **exact** by composition |
| `var`, `std` of any covariance estimator (`26_:54`, `:112`); `cov`, `cor` of any variance estimator (`:142`, `:170`) | Diagonal of `cov`; diagonal matrix of `var` | `sigma` or `var` | **exact** by composition |
| `variance_series` (`26_:280`) | Expanding-window variance, one row per observation | The inner estimator on each prefix; one fold per row under a seam member | **exact** by composition |
| `ShrunkExpectedReturns` with `JamesStein`, `BayesStein`, `BodnarOkhrinParolya` (`16_:455`; means `:663`, `:738`, `:830`) | `(1 - α) mu + α b` with `α` from the eigenvalues or the inverse of `sigma`, `mu - b`, `N`, `T` | `mu`, `sigma`, `T = n`; the three targets read `mu`, `sigma \ I` and `tr(sigma) / T` (`:562`-`:580`) | **exact** by composition |
| `EquilibriumExpectedReturns` (`17_:83`, `:160`) | `l Σ w` | `sigma` | **exact** by composition |
| `ExcessExpectedReturns` (`18_:54`, `:127`) | `mean(me.me, X) .- rf` | `mu` | **exact** by composition |
| `StandardDeviationExpectedReturns`, `VarianceExpectedReturns` (`27_:64`, `:193`) | `std` or `var` of `me.ce` as a return proxy | `sigma` diagonal | **exact** by composition |
| `CustomValueExpectedReturns` (`34_:95`, `:171`, `:216`, `:228`) | A constant, a stored vector, or a callable of `X` | Nothing (`Number`, `VecNum`); the callable reads `X` | **exact**, stateless; the callable branch is the caller's own contract |

### 4.3 Members that refit

| Member (file:line) | Statistic | Exact fold | Merge | Stability | Why no fold, and the known approximation | Route |
| --- | --- | --- | --- | --- | --- | --- |
| `Covariance{SemiMoment}` (`03_:1038` refusal) | Covariance of `min(X - mu, 0)` | None | — | n/a | The clamp reads the batch centre `mu`; when `mu` moves every past clamp moves and a past observation's downside membership flips (#308). No approximation short of freezing the centre | **refit** |
| `Coskewness{SemiMoment}`, `Cokurtosis{SemiMoment}` (`40_:636`, `:789` refusals) | Higher co-moments of the clamped deviations | None | — | n/a | Same argument | **refit** |
| `GerberCovariance`, `Gerber0/1/2` (`05_:290`; `gerber_updown` `:376`; `cov` `:781`) | Concordance counts of returns beyond `±t σ_i`, ratio per pair, then `cor2cov` by `σ` | None | — | n/a | The band edge `t σ_i` reads the whole-sample `std(ce.ve, X)` (`:783`); a moving `σ` re-marks every past observation. A running count of crossings against a frozen threshold is an approximation, unnamed in the literature | **refit** |
| `SmythBrobyCovariance` and its nine algorithms (`06_:624`; `sb_delta` `:740`) | Gerber counts weighted by `κ / (1 + γⁿ)` over centred standardised returns | None | — | n/a | Reads standardised returns, so the same whole-sample `σ`; the weights compound the dependence | **refit** |
| `GerberIQCovariance` with `BasicGerberIQ`, `PartialGerberIQ`, `FullGerberIQ` (`35_:1958`; `regenerate_decay` `:750`; `cor` `:2428`) | Gerber statistic with a noise zone `c σ_i`, a decay `ExpGerberIQDecay(T, k)` and a scaler | None | — | n/a | Reads `σ` as above and resolves the decay parameters from `X` (`:753`); the decay is a function of `T` | **refit** |
| `DistanceCovariance` (`07_:64`; `cor_distance` `:295`; the definition at `:212`) | Distance correlation of Székely, Rizzo and Bakirov (2007), `szekely2007` | None | — | n/a | Double centring subtracts row, column and grand means of a `T × T` distance matrix, so one new observation changes every `A_ts`. Huo and Székely (2016, Technometrics) give an `O(T log T)` batch algorithm, not a fold | **refit** |
| `LowerTailDependenceCovariance` (`08_:68`; `lower_tail_dependence` `:148`) | `(1/k) Σ 1[x_ti ≤ q_i, x_tj ≤ q_j]` with `q_i` the `⌈Tα⌉`-th order statistic | None | — | n/a | An order statistic per asset; the count `k` and the quantile both move with `T`. Quantile sketches (Greenwald and Khanna 2001, SIGMOD) approximate `q_i`, and the joint count still needs the buffer | **refit** |
| `KendallCovariance` (`09_:71`; `cor` `:149`) | Tie-corrected `τ_b` from `StatsBase.corkendall` | None on the seam. **An exact `O(T)` update from the buffer exists**: a new observation adds `T` comparisons to `C - D` and to the tie counts, so the pair statistic is a sum over pairs and folds against the buffer | Concatenation of buffers | n/a | Reads every pair of observations; the update is exact but not sample-free, which the membership rule refuses. Knight (1966, JASA) is the `O(T log T)` batch form | **refit** (from the buffer, with a cheap update available) |
| `SpearmanCovariance` (`09_:209`; `cor` `:285`) | Pearson correlation of the ranks | None | — | n/a | Every rank moves when one observation arrives | **refit** |
| `MutualInfoCovariance` (`11_:85`; `cor` `:149`) | Mutual information over a joint histogram, bins from `Knuth`, `FreedmanDiaconis`, `Scott` or `HacineGharbiRavier` (`10_`) | None | — | n/a | The bin width reads the whole sample (`T`, the IQR, a Bayesian fit); a fixed binning would fold the counts. Streaming histograms (Ben-Haim and Tom-Tov 2010, JMLR) approximate | **refit** |
| `MedianExpectedReturns` (`33_:58`; `mean` `:141`, `:165`) | Per-asset median, weighted or not | None | — | n/a | An order statistic. The P² algorithm (Jain and Chlamtac 1985, Communications of the ACM) is the classic approximate recursion; it is not exact | **refit** |
| `ImpliedVolatility` with `ImpliedVolatilityPremium`, `ImpliedVolatilityRegression` (`24_:265`; `realised_vol` `:383`; `cov` `:743`) | Inner correlation rescaled by a predicted realised volatility, from an implied-volatility series `iv` per block of `ws` observations | None | — | n/a | Reads an external series and a rolling realised volatility over non-overlapping blocks; the regression branch fits `rv` on `iv` over all blocks | **refit** |
| `WindowedExpectedReturns`, `WindowedCovariance`, `WindowedVariance`, `WindowedCoskewness`, `WindowedCokurtosis` (`28_`-`32_`, one `@windowed_estimator` each; `windowed_preamble` `01_:1120`) | The inner moment over the last `window` observations, or an index vector | None | — | n/a | Needs a removal, and Welford's update has no numerically stable inverse: the reverse update is the textbook cancellation (#308, ADR 0107). The map's capability 3, a capped buffer of `window` rows, is the exact route; the exponentially weighted family is the classic approximation of a window (RiskMetrics Technical Document, 1996) | **refit** |

### 4.4 The regression estimators (files `21_`-`23_`)

| Member (file:line) | Statistic | Exact fold | Merge | Stability | Why no fold | Route |
| --- | --- | --- | --- | --- | --- | --- |
| `StepwiseRegression` with `PValue`, `ForwardSelection`, `BackwardElimination`, and the AIC/BIC/R² criteria (`22_:152`; `regression` `:645`) | Per-asset loadings after a stepwise factor search, fitted by `LinearModel` or `GeneralisedLinearModel` | None | — | n/a | The search reads p-values or information criteria over the whole sample, and a selected set can change with one observation. For a **fixed** factor set the normal equations `F'F`, `F'y` are sums of blocks and recursive least squares (Plackett 1950, Biometrika) folds them exactly, which the library does not do today: `fit(::LinearModel)` calls the batch solver (`21_:527`) | **refit** |
| `DimensionReductionRegression` with `PCA`, `PPCA` (`23_:289`; `regression` `:495`; `fit(::PCA)` `:124`) | Regression on the principal components of the standardised factors | None | — | n/a | `PCA` is an eigendecomposition of the factor covariance, which a covariance state gives; the standardisation reads the factor mean and variance, which a variance state gives; the second-stage regression is the row above. `PPCA` is an EM fit over the whole sample | **refit** today; an exact composition for `PCA` is a build option once the regression folds |
| `LinearModel`, `GeneralisedLinearModel` targets (`21_:461`, `:679`) | The fit of one asset on one design | See the row above | — | n/a | Targets, not estimators; they follow the estimator that calls them | — |

### 4.5 The cross-sectional family (files `38_`, `39_`, `41_`)

| Member (file:line) | Statistic | Exact fold | Merge | Stability | Why no fold | Route |
| --- | --- | --- | --- | --- | --- | --- |
| `CrossSectionalLinearRegression`, `CrossSectionalTargetRegression` (`38_:315`, `:382`; `cross_sectional_regression` after `:584`) | One weighted least-squares solve per observation on the lagged exposures, giving factor returns `f[t, :]`, idiosyncratic returns `eps[t, :]`, the eligible count `n[t]` and the intercept `b[t]` | **Yes, and it is a running count**: observation `t` reads only row `t` of `X`, `Z` and `W`, so folding one observation appends one row to each history | **Concatenation.** A block fitted alone equals the same block fitted after another, row for row | n/a per observation; the solve is `UncheckedSolve`, `MinimumNormSolve`, `PseudoInverseFallback` or `RankDeficiencyRefusal` (`:468`-`:477`) | — | **exact** |
| `MarketCapWeights` (`39_:163`) | `mcap^p` of the current cross-section | Pointwise per observation | Concatenation | n/a | — | **exact** |
| `BlendedInverseVarianceWeights` (`39_:246`; `cross_sectional_lagged_inverse_variance` `:452`; refine `:649`) | A blend of cap weights and a winsorised lagged inverse idiosyncratic variance, second pass | Pointwise per observation given the lagged variance history, which `variance_series` produces | Concatenation, given the history | n/a | The lagged inverse variance is an expanding statistic; exact under a seam member for `ve`, refit otherwise | **exact** by composition |
| `CrossSectionalFactorModel` (`41_:396`) | The Result: `M`, `L`, `b`, `csr`, the histories `Ms`, `vs`, `rw`, `bw`, the basis `fcb`, the forecast `rf` | A Result, not an estimator | — | — | — | — |

### 4.6 The descriptors, `42_FactorExposures/`

These fit a panel per observation. The stateless ones read the current row alone.

| Member (file:line) | Statistic | Exact fold | Merge | Stability | Why no fold | Route |
| --- | --- | --- | --- | --- | --- | --- |
| `EWMean` (`04_:369`; `ew_mean_series` `:188`), `EWVolumeRatio` (`:470`), `DaysToCover` (`:572`) | `S_t = λ S_{t-1} + (1 - λ) r_t` per asset, with a warm-up count | The EW recursion, `04_:188` | `λ^{n_B} S_A + S_B` per asset, from the per-asset count | Convex | — | **exact** |
| `EWVolatility`, `EWResidualVolatility` (`05_:99`, `:333`; `ew_variance_estimator` `:36`) | Square root of the `variance_series` of a `RegimeAdjustedExpWeightedVariance` with `regime_method = nothing`, on the returns or on the residuals of an EW beta | The variance member's fold, `36_:717` | The variance member's, which merges when centred and without HAC | Convex | — | **exact** |
| `EWBeta`, `EWMacroSensitivity`, `EWDownsideBeta` (`06_:567`, `:981`, `:1220`; the recursion at `:502`) | `β_t = C_t / (V_t + min_val)` with EW covariance `C_t` and market variance `V_t` about EW means; a cross-sectional shrinkage of `β` towards a group mean (`:349`) | The EW recursions on `C`, `V` and the two means | The raw accumulators merge; the shrinkage is pointwise on the current cross-section | Convex; divides by `V_t + min_val` | — | **exact** |
| `PanelFieldRatio`, `PanelFieldLog`, `Passthrough` (`02_:196`, `:273`, `:323`); `ConstantExposure` (`12_:49`); `OneHotExposure` (`11_:42`); `DerivedExposure` (`10_:55`); `CompositeExposure` (`09_:152`) | A function of the current row of one or several panel fields or descriptors | Pointwise | Concatenation | n/a | — | **exact**, stateless |
| `GrowthRate`, `ChangeToScale`, `ChangeInIntensity` (`03_:87`, `:160`, `:239`) | `z_t / z_{t-ℓ} - 1` and its two siblings | Pointwise given a buffer of `ℓ` rows of the panel field | Concatenation, given the buffer | n/a | Reads a lag `ℓ`; a capped buffer of depth `ℓ` makes it exact | **refit** from a buffer of depth `ℓ` |
| `RollingLogReturn` (`07_:177`), `RollingMax` (`:263`; algorithm `:284`) | A windowed sum of `log1p` returns with a skip; the windowed maximum | None on the seam. The sum is a difference of cumulative sums, exact against a buffer of `window + skip` rows; the maximum needs the window | — | The cumulative-sum difference cancels on long histories | A window needs a removal; a capped buffer is the exact route (capability 3) | **refit** from a capped buffer |

### 4.7 The return forecasts, `45_ReturnForecasts/`, and the family basis

| Member (file:line) | Statistic | Exact fold | Merge | Stability | Why no fold | Route |
| --- | --- | --- | --- | --- | --- | --- |
| `DescriptorScores` (`02_:22`; `descriptor_scores` `:295`; `neutralise_scores!` `:204`) | Descriptor scores per observation, neutralised across the cross-section | Pointwise per observation, given the descriptors' route | Concatenation | n/a | — | **exact** by composition |
| `ExpWeightedReturnForecast` (`05_:322`; the definition at `:239`) | `A_t = λ A_{t-1} + (1 - λ) S_t' W_t S_t`, `c_t` likewise, `β_t = (A_t + ρ_t I)⁻¹ c_t` | The EW recursion on the normal equations, `:138` | `λ^{n_B} A_A + A_B`, and the same on `c` | Convex; the ridge `ρ_t` reads the diagonal of `A_t` | — | **exact** |
| `FixedWeightedReturnForecast` (`04_:162`) | A fixed signed combination of scores, scaled | Pointwise | Concatenation | n/a | — | **exact**, stateless |
| `CustomValueReturnForecast` (`03_:22`) | A stored `mu` | Nothing to fold | — | n/a | — | **exact**, stateless |
| `TargetReturnForecast` (`06_:103`; `target_forecast_fit` `:350`; the cross-validated branch `:403`; calibration `:534`) | A model fitted on stacked `(scores, forward target)` samples, then a calibration regression, optionally cross-validated | None | — | n/a | A fit over the whole stacked sample, and a cross-validation over it | **refit** |
| `FactorFamilyBasis` (`43_:66`; `factor_family_basis` `:489`; `weighted_family_exposures` `:553`) | The re-basis of a Factor Family, from the exposure history `Ms` and the benchmark-weight history `bw` | Not settled here. The builder takes the two histories; whether it reads the last slice alone or every slice is a claim a later ticket measures | — | — | — | route follows that measurement |

---

## 5. The priors, `src/13_Prior/`

`LowOrderPrior` carries `X` (`01_Base_Prior.jl:1217`), so an online prior saves the arithmetic
and not the memory, as ground truth 9 says. Every row reads the moments of its inner estimators
and, where it says so, the sample.

| Member (file:line) | Statistic | Exact fold | Merge | Stability | Why no fold | Route |
| --- | --- | --- | --- | --- | --- | --- |
| `EmpiricalPrior`, plain arm (`02_:83`; `prior` `:162`) | `mu = mean(me, X)`, `sigma = cov(ce, X)`, carried with `X` | By composition of `me` and `ce`; the default `ce` is `PortfolioOptimisersCovariance`, so the composition route of §4.2 is on the path | The inner states' | The inner states' | — | **exact** by composition |
| `EmpiricalPrior`, horizon arm (`prior` `:240`; algorithm `:86`) | Log-return moments scaled by `horizon`, mapped back by the log-normal closed forms | By composition on `log1p.(X)`: the inner estimators fold the log returns, and the closed forms read `mu`, `sigma` alone | The inner states' | The inner states' | The `log1p` transform is per observation | **exact** by composition |
| `HighOrderPriorEstimator` (`04_:705`; `prior` `:803`) | The low-order block plus `kt` from `kte`, `sk` and `V` from `ske`, and the structure matrices at `N` | By composition of `pe`, `kte`, `ske`; exact under the `FullMoment` seam members, refit under `SemiMoment` | The inner states' | The inner states' | `negative_spectral_coskewness` is an eigendecomposition of the read-out, not of the sample | **exact** by composition |
| `BlackLittermanPrior` (`06_:151`; `prior` `:699`; algorithm `:80`) | The master equations on `prior_mu`, `prior_sigma`, views `P`, `Q`, `tau`, `omega` | By composition: `bl_preroll` reads `prior_sigma` and `size(X, 1) = n`; `vanilla_posteriors` reads moments | The inner states' | The inner states' | A view set is configuration, and #852 settled that an empty one answers the prior | **exact** by composition |
| `FactorPrior` (`03_:115`; `prior` `:417`) | Factor moments lifted through a time-series regression `rr` | None: `factor_reconstruction` runs `StepwiseRegression` or `DimensionReductionRegression` (§4.4) | — | — | The regression refits | **refit** |
| `BayesianBlackLittermanPrior` (`07_:197`; `prior` `:373`) | Factor views combined in precision form, lifted through `rr` | None, for the regression | — | — | Reads `pr.rr` | **refit** |
| `FactorBlackLittermanPrior` (`08_:167`; `prior` `:357`) | Master equations on the factor axis, lifted through `rr` | None, for the regression | — | — | Step 4 regresses `X` on `F` | **refit** |
| `AugmentedBlackLittermanPrior` (`09_:276`; `prior` `:447`) | Master equations on the stacked asset-and-factor space | None, for the regression | — | — | Step 5 regresses `X` on `F` | **refit** |
| `HighOrderFactorPriorEstimator` (`14_:308`; `prior` `:448`) | Factor higher moments projected through `kron(M, M)` plus residual corrections | None, for the regression and for the residual co-moments of `X - posterior_X` | — | — | Reads `pr.rr` and the reconstruction error over the whole sample | **refit** |
| `MeucciEntropyPoolingPrior` with `H0/H1/H2` (`11_:172`; `prior` `:680`) | Observation weights `w` of length `T` minimising relative entropy under moment views, `meucci2008` | None | — | n/a | The unknown is one weight per observation, so one new observation changes the dimension of the problem; every view constraint reads the sample | **refit** |
| `EntropyPoolingPrior` with `OptimEntropyPooling`, `JuMPEntropyPooling`, `ConditionalValueatRiskEntropyPooling`, and the sequential heuristics (`12_:3330`; `prior` `:3527`) | The same, with tail views (`EPTail`, `EPRLVaR`, `vorobets2021`) | None | — | n/a | Same; the CVaR and EVaR views read order statistics of `X w` | **refit** |
| `OpinionPoolingPrior` with `LinearOpinionPooling`, `LogarithmicOpinionPooling` (`13_:243`; `prior` `:502`) | A pool of entropy-pooling weight vectors, then `pe2` refitted under the pooled `w` | None | — | n/a | Every pooled estimator is the row above, and the refit under `w` is a weighted fit, which `assert_partial_fittable` refuses | **refit** |
| `CrossSectionalFactorPrior` (`17_:60`; `prior` `:280`; algorithm `:70`) | Descriptors → neutralisation → family basis → lagged per-observation regression → `variance_series` → `pe.pe` on the factor returns → return forecast → lift | None as one verb. Component by component: the regression appends (§4.5), the EW descriptors fold (§4.6), `variance_series` and `pe.pe` fold under seam members, the rolling and lag descriptors need a capped buffer, and `TargetReturnForecast` refits | Concatenation for the regression; the others as their rows | — | The verb runs the whole panel once, and warm-up drops leading observations (`cross_sectional_warmup`) | **refit** today; the decision ticket decides whether it becomes a per-observation fold over the components above |
| `RegressionPanel` (`15_:382`; `asset_panel` `:557`) | The loadings `pr.rr.L` as a panel field | Follows the regression | — | — | — | route of the regression |
| `PhylogenyPanel` with `Proximity` (`15_:491`; `asset_panel` `:568`) | A proximity matrix graded off a network over `X` | By composition of the network's covariance and distance (§7) | — | — | — | route of the network |

---

## 6. The uncertainty sets, `src/14_UncertaintySets/`

| Member (file:line) | Statistic | Exact fold | Merge | Stability | Why no fold | Route |
| --- | --- | --- | --- | --- | --- | --- |
| `DeltaUncertaintySet` (`02_:69`; `ucs` `:270`; builders `:142`, `:200`) | Box sets `dmu \|mu\|` and `sigma ± dsigma \|sigma\|` | By composition: reads `pr.mu`, `pr.sigma` alone | The prior's | The prior's | — | **exact** by composition |
| `NormalUncertaintySet` with `BoxUncertaintySetAlgorithm`, `EllipsoidalUncertaintySetAlgorithm`, `NormBallUncertaintySetAlgorithm` and the `K` algorithms (`03_:85`; `choose_scaling_parameter` `:257`; boxes `:433`, `:477`) | Asymptotic-normal boxes and ellipsoids from `sigma / T`, `(I + K) kron(sigma_mu, sigma_mu) T`; the covariance box from Wishart draws | By composition: reads `pr.mu`, `pr.sigma` and `T` (`ue.ens`, `pr.ens` or `size(pr.X, 1)`, which is `n`); the draws are seeded | The prior's | The prior's | The Wishart box is a Monte Carlo of the moment, not of the sample | **exact** by composition |
| `ARCHUncertaintySet` with `StationaryBootstrap`, `CircularBootstrap`, `MovingBootstrap` (`04_:334`; `bootstrap_generator` after `:417`; `ucs` after `:574`) | Quantiles of `me` and `ce` over block-bootstrap resamples of `pr.X` (`politis1994stationary`, `politis1992circular`, `kunsch1989`) | None | — | n/a | Resamples the rows of `pr.X`; the index stream and every resample read `T` | **refit** |
| `CharacteristicUncertaintySet` with `L1UncertaintySetAlgorithm`, `SignedL1UncertaintySetAlgorithm`, `ActiveAssetsUncertaintyAlgorithm` (`05_:604`; `mu_ucs` `:994`, `:1003`) | An `ℓ₁` ball whose radius is bisected on the activation ladder of the sorted `pr.mu`, scaled by `sqrt(diag(pr.sigma))` (`quintile`) | By composition: reads `pr.mu`, `pr.sigma` alone | The prior's | The prior's | The sort is over `N` assets, not `T` observations | **exact** by composition |
| `OrthogonalUncertaintySet` with `IdentityScaling`, `IdiosyncraticVarianceScaling` and the four metrics (`09_:346`; `ucs` `:563`; weight histories `:135`) | Mean and covariance sets on the orthogonal complement of the weighted factor span of `pr.rr` | None on the seam: reads the loadings block, its `esigma`, and the last row of `rw` or `bw` | — | — | Follows the regression that produced `pr.rr`; under a cross-sectional block the last row is pointwise | route of the regression: **refit** today |
| `BoxUncertaintySet`, `EllipsoidalUncertaintySet`, `L1UncertaintySet`, `SignedL1UncertaintySet`, `NormBallUncertaintySet`, `CompactCovarianceUncertaintySet` (`01_:758`, `:1319`, `05_:137`, `:291`, `08_:92`, `07_:75`) | Results | — | — | — | — | — |
| `ScenarioCount` (`06_:720`), `RateSignificance` (`:812`), `RateRadius` (`:1966`), `ConcentrationRadius` (`:1846`), `DimensionalRateRadius` (`:2079`), `EffectiveAssetFloor` (`:2579`) | A significance or a radius from `T` (or `effective_sample_size`, `:681`), `N` and a confidence | By composition: counts and moments | — | n/a | — | **exact** by composition |
| `EntropyBudget` (`:906`), `DualNormRadius` (`:2220`, scale `:2336`), `TailTermParity` (`:2407`) | A deformation, a dual-norm radius, a tail weight, from `pr.ens`, `pr.mu`, `pr.sigma` and the observation weights | By composition, on the reading above. **Assumption**: none of the three reads an order statistic of `pr.X`; the `# Algorithm` sections read say counts, `ens` and moments. A build ticket confirms | — | n/a | — | **exact** by composition, to confirm |
| `HillTailDecay` (`:1162`; `hill_tail_index` `:1045`), `RadialTailDecay` (`:1436`; `radial_tail_index` `:1368`) | A tail index from the `kmin` largest order statistics of a returns or drawdown series, or of the whitened radial norms of `X` | None | — | n/a | Order statistics of the sample; the drawdown series reads the whole path | **refit** |

---

## 7. The matrix processing, the distances, the phylogeny and pre-selection

| Member (file:line) | Statistic | Exact fold | Merge | Stability | Why no fold | Route |
| --- | --- | --- | --- | --- | --- | --- |
| `Posdef` (`04_PosdefMatrix.jl:107`; `posdef!` `:205`) | Nearest correlation matrix projection (`higham2002`) | A transform of a matrix; by composition | — | — | — | **exact** by composition |
| `Denoise` with `SpectralDenoise`, `FixedDenoise`, `ShrunkDenoise` (`05_Denoise.jl:429`; `denoise!` `:814`) | Marčenko-Pastur split of the spectrum at `q = T / N` (`mlp1`, `mpdist`) | By composition: reads the matrix and `q`, which is `n / N` | — | — | An eigendecomposition of the read-out, not of the sample | **exact** by composition |
| `Detone` (`06_Detone.jl:151`; `detone!` `:234`) | Removal of the top `n` principal components | By composition | — | — | — | **exact** by composition |
| `MatrixProcessing` (`07_MatrixProcessing.jl:266`; `matrix_processing!` `:407`) | The three above plus an algorithm, in `order` | By composition; the `:alg` step receives `X` and is the algorithm's own contract | — | — | — | **exact** by composition |
| `Distance` with `SimpleDistance`, `SimpleAbsoluteDistance`, `LogDistance`, `CorrelationDistance`, `CanonicalDistance` (`09_Distance/02_:102`; `distance` `:336`, `:468`) | `sqrt(0.5 (1 - ρ))` and its siblings, of a correlation | By composition of `cor(ce, X)` | — | — | — | **exact** by composition |
| `Distance` with `VariationInfoDistance` (`01_:279`; `distance` `:405`, `:410`) | Variation of information over joint histograms of `X` | None | — | n/a | The same histogram argument as `MutualInfoCovariance` | **refit** |
| `DistanceDistance` (`03_:79`; `distance` `:148`) | A metric applied pairwise to the columns of a base distance matrix | By composition of the base distance | — | — | — | route of the base distance |
| `FeatureDistance` with `LastObservation`, `StackObservations`, `AggregateFeatures`, `AggregateDistances`, `MeanCollapse`, `MedianCollapse` (`05_:534`; `feature_matrix` `:1110`) | A distance over a Feature Matrix from an Asset Panel producer | `LastObservation` reads one row; `StackObservations` and the aggregates read the history; `MedianCollapse` is an order statistic | — | — | Follows its producer (§5's panels) and its collapse | route of the producer; **refit** under a history-reading collapse |
| `ClustersEstimator` with `HClustAlgorithm`, `KMeansAlgorithm`, `DBHT` and `OptimalNumberClusters` with `SilhouetteScore` (`11_Phylogeny/02_:696`, `:599`, `05_:57`, `04_:102`, `02_:458`; `clusterise` `03_:357`, `05_:257`, `12_:144`) | A dendrogram or a partition of `(S, D)`, cut at an optimal `k` | By composition: `cor_and_dist(de, ce, X)` is the only sample read, and it is a moment | — | — | The tree is recomputed from the moment at every step; an incremental linkage is not needed for exactness | **exact** by composition |
| `NetworkEstimator` with `KruskalTree`, `BoruvkaTree`, `PrimTree`, the similarity algorithms and `PMFG` (`17_:150`); `NetworkClustersEstimator` (`:262`; `clusterise` `21_:115`, `:195`); `CentralityEstimator` with the eight centralities (`18_:101`; `centrality_vector` `23_:178`) | A graph over `D`, its clusters or its centralities | By composition of `(S, D)` | — | — | — | **exact** by composition |
| `LoGo` (`13_:218`; `logo!` `:362`) | A sparse inverse covariance from the PMFG cliques and separators (`J_LoGo`) | By composition: reads `sigma` and `distance(je.de, S, X)`; the second is a moment unless `de` is the variation of information | — | — | — | **exact** by composition |
| `ScoreSelector` with `ThresholdRule`, `RankRule`, `QuantileRule` (`22_Preselection.jl:620`; `select_assets` `:664`) | A keep-mask from a risk measure scored on each asset column of `rd.X` | Only when the score's `risk_input_kind` is `WeightsInput` (§8); a `NetReturnsInput` score is a functional of the column | — | — | A CVaR or a drawdown of a column is an order statistic or a path | route of the score: **exact** by composition for a second-moment score, **refit** otherwise |
| `CompleteAssetSelector` (`:696`) | The columns with no `NaN` | A running count of `NaN` per column | A sum of blocks | n/a | — | **exact** |
| `RedundancySelector` with `PairwiseCorrelation`, `CorrelationComponents`, `ClusterGroups` (`:1379`; `redundancy_keep` `:1027`, `:1152`, `:1296`) | Survivors of correlation groups over `cor(alg.ce, rd.X)` or a clustering of `rd.X`, ranked by an optional score | By composition of the correlation or the clustering, and of the score as the row above | — | — | — | **exact** by composition, or the score's route |

---

## 8. The optimisers, `src/20_Optimisation/`

An optimiser owns no statistic of its own. Its route is the composition of the routes of its
inputs, so the ledger names which input reads the sample and which reads a moment only.

### 8.1 The four doors through which an optimiser reads the sample

1. **A risk measure's `risk_input_kind`** (`src/19_RiskMeasures/01_Base_RiskMeasures.jl:200`
   and the declaration beside each type). `WeightsInput` is evaluated as `r(w)` and reads a
   moment the prior carries; `NetReturnsInput` is evaluated as `r(X w - fees)` and
   `WeightsReturnsFeesInput` as `r(w, X, fees)`, and both read `pr.X`.
2. **The return term of the JuMP family.** `ArithmeticReturn` reads `pr.mu`
   (`09_JuMPConstraints/02_Returns_and_ObjectiveFunctions.jl:1509`); `LogarithmicReturn` sets
   the portfolio returns from `pr.X` (`:1836`); `NoReturn` reads nothing.
3. **The hierarchical family's `X = pr.X`**, read at `05_:327`, `06_:1022` and `07_:744`, and
   passed to `expected_risk(r, w, X, fees)` for every cluster (`05_:353`, `07_:531`).
4. **A meta-optimiser's cross-validation** field `cv` (`17_:459`, `18_:286`), which splits the
   sample.

`Kurtosis` and `NegativeSkewness` are resolved by `factory` against the prior's `kt` and `sk`
(`04_Kurtosis.jl:382`, `05_NegativeSkewness.jl:230`), so their moment is the prior's; the input
kind governs only the functor.

### 8.2 The risk measures by input kind

| Input kind | Members | Route of an optimiser that holds one |
| --- | --- | --- |
| `WeightsInput` (moment only) | `Variance`, `StandardDeviation`, `UncertaintySetVariance` (`02_Variance.jl:1204`-`:1206`), `NegativeSkewness` (`05_:272`), `TurnoverRiskMeasure` (`17_:157`), `EqualRisk` (`23_:83`) | **exact** by composition |
| `WeightsReturnsFeesInput` | `Kurtosis` (`04_:443`), `LowOrderMoment` and `HighOrderMoment` (`03_:1548`), `Skewness`, `VarianceSkewKurtosis` (`20_:710`, `:711`), `TrackingRiskMeasure`, `RiskTrackingRiskMeasure` (`18_:605`, `:606`), `ThirdCentralMoment` (`25_:491`), `MedianAbsoluteDeviation` (`24_:328`) | **refit**. The moment measures are functions of second, third or fourth moments about a centre and would fold through the seam members, but the functor reads `X` today |
| `NetReturnsInput` | `ValueatRisk`, `ValueatRiskRange`, `DrawdownatRisk`, `RelativeDrawdownatRisk` (`06_:1257`-`:1279`); `ConditionalValueatRisk`, its distributionally robust twin, the two ranges, `ConditionalDrawdownatRisk` and its twins (`07_:1293`-`:1299`); `EntropicValueatRisk` and its three siblings (`08_:605`-`:608`); `RelativisticValueatRisk` and its three siblings (`09_:873`-`:876`); `PowerNormValueatRisk` and its three siblings (`19_:660`-`:663`); `AverageDrawdown`, `RelativeAverageDrawdown` (`11_:264`); `MaximumDrawdown`, `RelativeMaximumDrawdown` (`13_:200`); `UlcerIndex`, `RelativeUlcerIndex` (`12_:201`); `WorstRealisation` (`15_:84`); `Range` (`16_Range.jl:80`); `BrownianDistanceVariance` (`14_:198`); `OrderedWeightsArray`, `OrderedWeightsArrayRange` (`10_:2335`); `GenericValueatRiskRange` (`21_:165`); `MeanReturn` (`25_:490`); `NoRisk` (`16_NoRisk.jl:81`, declared and unused) | **refit**. Each is a quantile, an order statistic, a path statistic or a double centring of `X w`; the arguments of §4.3 apply. `MeanReturn` alone is a mean and would fold; `NoRisk` reads nothing |

### 8.3 The optimiser families

| Family and member (file:line) | Reads the sample through | Reads a moment only through | Route |
| --- | --- | --- | --- |
| Naive: `InverseVolatility` (`03_:319`; `optimise` `:443`) | — | `diag(pr.sigma)` | **exact** by composition |
| Naive: `EqualWeighted` (`:529`; `:612`) | `size(rd.X, 2)`, a count | — | **exact**, stateless |
| Naive: `RandomWeighted` (`:703`; `:816`) | — | An `rng` and `alpha` | **exact**, stateless |
| Hierarchical: `HierarchicalRiskParity` (`05_:186`), `HierarchicalEqualRiskContribution` (`07_:218`), `SchurComplementHierarchicalRiskParity` (`06_:550`), over `HierarchicalOptimiser` (`04_:427`) | Door 3, for every `r` whose kind is not `WeightsInput`; the Schur variant reads `r.sigma` for the complement and `X` for the risk | `prior(opt.pe, rd)`; `clusterise(opt.cle, pr)`, a composition of `ce` and `de` (§7); a `WeightsInput` risk measure | **exact** by composition when `pe` and every `r` are; **refit** otherwise |
| JuMP: `MeanRisk` (`11_:293`), `FactorRiskContribution` (`12_:235`), `NearOptimalCentering` (`13_:314`), `RiskBudgeting` with `AssetRiskBudgeting`, `FactorRiskBudgeting` (`14_:576`), `RelaxedRiskBudgeting` (`15_:238`), over `JuMPOptimiser` (`10_:546`; `pr = prior(opt.pe, rd)` `:1310`) | Doors 1 and 2; `FactorRiskContribution` and `FactorRiskBudgeting` also read a regression `re` | `ArithmeticReturn` on `pr.mu`; a `WeightsInput` risk measure on `pr.sigma`, `pr.kt`, `pr.sk`; the uncertainty sets of §6 on `pr` | **exact** by composition under `ArithmeticReturn` or `NoReturn` and `WeightsInput` measures, with an exact prior; **refit** otherwise. The model is rebuilt per step today; reuse is the map's capability 1 |
| Meta: `NestedClustered` (`17_:459`; `optimise` `:803`) | Door 4 through `cv`, and the outer `opto` over the inner portfolios' return series | The inner `opti` per cluster, as its own row | **refit**; the inner and outer optimisers take their own routes |
| Meta: `Stacking` (`18_:286`; `optimise` `:601`; `needs_previous_weights` `:461`) | Door 4 through `cv`, and `opto` over the inner returns | The inner `opti` | **refit** |
| Meta: `SubsetResampling` (`19_:251`; `optimise` `:620`) | Only what `opt` reads; the subsets are over assets, drawn from `rng` | `opt` on each subset | route of `opt` |
| Finite allocation: `DiscreteAllocation` (`22_:203`; `optimise` `:469`), `GreedyAllocation` (`23_:164`; `:370`), over `FiniteAllocationInput` (`21_:106`) | Nothing: reads `w`, `prices`, `cash`, `horizon`, `fees` | — | **no method owed**; it is a per-step discretisation of a given `w`, which is what the map's *Not yet specified* suspected |
| The fold loop and the walk-forward (`02_CrossValidation/04_WalkForward.jl:825`) | `fit_and_predict` calls `optimise` on the fold's training slice, so every step refits today | — | The consumer of this ledger, not a member |

---

## 9. Summary counts

A "member" is a concrete estimator type; an algorithm tag that changes the route is counted as a
member of its own (the three `SemiMoment` arms, `VariationInfoDistance`), and one that does not
is folded into its estimator's row. Composed members are counted under **exact**, because the
route they take is exact; the note says how many of them owe a state and how many owe a read-out
alone.

| Family | Exact | of which merge | of which by composition (no state) | Exact but no merge | Refit | Pending (#854) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Moments proper (§4.1-4.3) | 14 | 6 | 8 | 2 | 18 | 3 | The 6 that merge hold a state today; the 8 composed owe a read-out; the 3 pending owe a state and, on measurement, a merge |
| Regression (§4.4) | 0 | — | — | 0 | 2 | — | `LinearModel` and `GeneralisedLinearModel` are targets, not members; recursive least squares is the known exact fold for a fixed design |
| Cross-sectional (§4.5) | 4 | 4 | 1 | 0 | 0 | — | The regression's merge is concatenation |
| Descriptors (§4.6) | 15 | 8 EW + 7 stateless | 7 | 0 | 5 | — | The 5 refit from a capped buffer of depth `ℓ` or `window`, which capability 3 supplies |
| Return forecasts and basis (§4.7) | 4 | 1 EW + 2 stateless + 1 composed | 2 | 0 | 1 | — | `FactorFamilyBasis` is unresolved: a later ticket measures what it reads |
| Priors (§5) | 3 | — | 3 | 0 | 9 | — | Plus two panel producers that follow their inputs. `CrossSectionalFactorPrior` is the one refit member whose components are mostly exact |
| Uncertainty sets, estimators (§6) | 3 | — | 3 | 0 | 2 | — | `OrthogonalUncertaintySet` follows the regression; six Results carry no route |
| Uncertainty sets, calibration rules (§6) | 9 | — | 9 | 0 | 2 | — | Three of the nine are marked *to confirm* |
| Matrix processing, distance, phylogeny, pre-selection (§7) | 16 | 1 (`CompleteAssetSelector`) | 15 | 0 | 1 | — | `FeatureDistance` and `ScoreSelector` follow their inputs; `VariationInfoDistance` is the one refit |
| Optimisers (§8) | 3 naive; hierarchical and JuMP by configuration | — | all | 0 | 2 meta by construction; hierarchical and JuMP by configuration | — | 6 risk measures read a moment only, 9 read `(w, X, fees)`, 34 read `X w`; `LogarithmicReturn` reads `X`; finite allocation owes no method |

The three routes, in one line each:

- **exact, with a merge**: the six Welford members; the cross-sectional regression and its
  weights; the eight exponentially weighted descriptors and the exponentially weighted forecast,
  on the identity `λ^{n_B} S_A + S_B`; `CompleteAssetSelector`; and, pending #854 and a
  measurement, the three exponentially weighted moments.
- **exact but no merge**: the two regime-adjusted members, and any exponentially weighted member
  built with a running location or a HAC buffer.
- **refit**: every member of §4.3 and §4.4, nine priors, the bootstrap and orthogonal sets, the
  two tail-decay rules, the variation-of-information distance, the lag and rolling descriptors
  until a capped buffer exists, the target forecast, and any optimiser that holds a returns-based
  risk measure, a logarithmic return or a cross-validation.

## 10. What a later ticket must measure

- The merge identity `λ^{n_B} S_A + S_B` on the three estimators of #854 once they exist, with
  the per-asset count, a holiday freeze and an inactive reset inside the join. #701 measured it on
  the variance recursion alone.
- What `factor_family_basis` reads of `Ms` and `bw` (`43_:489`, `:553`).
- Whether `EntropyBudget`, `DualNormRadius` and `TailTermParity` read an order statistic of
  `pr.X` anywhere; this ledger read their `# Algorithm` sections and found counts and moments.
- Whether `DescriptorScores`' neutralisation is pointwise per observation, which this ledger
  assumed from `neutralise_scores!` acting on one slice.
- For the JuMP family, which constraint builders outside `20_RiskMeasureConstraints/` read
  `pr.X` (a grep found `set_portfolio_returns!` for the logarithmic return and the tracking
  constraints); the decision ticket that names the optimiser route needs the complete list.
