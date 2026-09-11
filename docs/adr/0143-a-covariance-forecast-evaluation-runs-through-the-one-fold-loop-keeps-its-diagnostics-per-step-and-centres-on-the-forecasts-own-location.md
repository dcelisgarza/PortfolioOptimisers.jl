---
status: accepted
---

# A covariance forecast evaluation runs through the one fold loop, keeps its diagnostics per step, and centres on the forecast's own location

## Context

[Map #861](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/861) names four
best-effort capabilities. The fourth is a covariance forecast evaluation: a per-step diagnostic
of a forecast `Σ̂_t` against the returns realised after it, in a batch form over a walk-forward
and in an online form over a threaded Partial Fit State. The library had no batch form.
[#864](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/864) researched the
mathematics — the per-step protocol, the three calibration ratios, the losses that are robust to
a noisy proxy, the aggregation and its bands — and
[#873](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/873) decided the design.

Six facts shaped the decision.

1. **The fold loop's online arm is generic over the estimator.** ADR 0140's `online_folds`
   reads nothing of an optimiser: it calls `assert_online_entry`, `update_online_estimator` and
   `partial_fit!(est, rd_view)`, then hands the callback the threaded estimator. A covariance
   estimator answers `cov(ce)` and a prior `prior(pe)` with no data, so either reads out where a
   refit would have run. The online form of an evaluation is therefore the walk-forward's Fold
   Fit, `ff = OnlineStep()`, and not a second verb — provided the evaluation runs through
   `fold_loop`. ADR 0067 made that loop the one seam, and ADR 0141 removed the last loop outside
   it as a defect.
2. **The `NonOptimisationCrossValidationEstimator` family holds schemes, not evaluations.** The
   ticket named it as "the slot a non-optimisation evaluation fits". Its one member is
   `MultipleRandomised`, a resampling *split*. A scheme says how folds are cut; an evaluation
   consumes a scheme, as `fit_and_predict(opt, rd, cv)` does.
3. **The reference has this feature, and #864 did not read it.** Its batch and online verbs each
   hand-roll a walk-forward and share one per-step kernel; its Result stores per-step scalars
   only — the squared Mahalanobis distance, its ratio, the diagonal ratio averaged over assets,
   the standardised portfolio return and the portfolio QLIKE per test portfolio, the active count
   — and never a forecast. Its realised quantity at `h > 1` is the horizon return `R = Σ_s z_s`,
   the noisier of the two forms #864 §1 stated. Its QLIKE is projected on the test portfolio only;
   it has no whole-matrix loss, no analytic band, and its comparison overlays plots without a test.
   It never centres the realised return. Its default test portfolio is inverse volatility
   recomputed every step from `Σ̂_t`, a matrix of portfolios is accepted, and a gap in the test
   window contributes zero to the aggregated return while the forecast is scaled by the pairwise
   observation count, `H ⊙ Σ̂`.
4. **The pairing here is `N² × M`, not `T × N`.** Map #931's ruling for the return-forecast
   evaluation — a lean Result holding the pairing, every statistic a verb over it — rests on its
   pairing costing `T × N`. Storing every `Σ̂_t` costs 80 MB at `N = 100` over 1,000 steps and
   640 MB at `N = 200` over 2,000, and the run the online form exists for is `h = 1` at every
   observation. The realised side costs nothing: `Z_t = rd.X[test_idx[i], :]`.
5. **The location the forecast is about is already in the estimator.** `Covariance` holds `me`
   and its folded `CovarianceState` holds `mu`; `GeneralCovariance` subtracts the weighted sample
   mean inside `StatsBase.cov` and its state holds `mu`; `ExpWeightedCovariance` and
   `RegimeAdjustedExpWeightedCovariance` hold `location` in their states and a `centred` flag that
   makes it zero; the composite and the transforms forward to the `ce` they hold; a prior publishes
   `mu`. So `E[(z − c)(z − c)'] = Σ̂` has a definite `c` for every family that fits a second moment,
   and the raw proxy `zz'` estimates `Σ + (μ − c)(μ − c)'` instead. The bias is `c'Σ̂⁻¹c / N`:
   0.16 % on daily data, 3.2 % on monthly data or for the horizon return at `h = 20`.
6. **Under a Coverage Policy a forecast is a `NaN` frame** (ADR 0117), which is the reference's
   own convention for an inactive asset: "NaN diagonal entries mark inactive assets".

## Decision

**One verb through the one fold loop.**
`covariance_forecast_evaluation(est, rd, cv; w = nothing, target = RealisedCovariance(),
store_forecasts = false)` takes an `AbstractCovarianceEstimator` or an `AbstractPriorEstimator` —
either wrapped in `Online` under `ff = OnlineStep()`, which is how the rolling online form is
written — a carrier, and an `IndexWalkForward` or `DateWalkForward`. Per fold the callback reads
`Σ̂` off the fold's estimator through the Asset Panel seam of the moment verbs —
`cov(est, X_train, pnl)` or `prior(pe, rd_train).sigma` under a refit, `cov(est)` or
`prior(pe).sigma` under `OnlineStep()` — and runs the per-step kernel on the test rows. The
batch expanding, batch rolling, online expanding and online rolling forms are the four
compositions the walk-forward and the `Online` wrapper already express, and the verb reads none
of them. An `Online` at the root under a scheme with no Fold Fit is refused by name, because a
batch fold would never seed its buffer. The verb and the kernel live in
`src/20_Optimisation/02_CrossValidation/13_CovarianceForecastEvaluation.jl`, after the loop it
runs through, and the three verbs above the Result in `14_CovarianceForecastSummary.jl`. It owes
the loop four small methods: `partial_fit!(pe::AbstractPriorEstimator, rd)`, which is
`fold_prior`; `partial_fit!(ce::AbstractCovarianceEstimator, rd)`; `is_time_dependent` and
`needs_previous_weights` for a non-optimiser and for an `Online` root, both `false`; and an
`advance_previous_fold` arm for a step record that carries no weights.

**The realised quantity is a typed family, `AbstractRealisedTarget`.** `RealisedCovariance()`,
the default, compares `S_t = Σ_s (z_s − c)(z_s − c)'` against `h Σ̂_t` — a per-row statistic whose
ratio has per-step variance `2 / (N h)` under a Gaussian null. `HorizonReturn()`, the reference's
member, compares `R R'` with `R = Σ_s (z_s − c)` against the same target — the `h`-day holder's
question, rank one, per-step variance `2 / N`. The two coincide at `h = 1`, so the reference is
an oracle at `h = 1` for both and at any `h` for `HorizonReturn()`. A gap in the test window
contributes to neither `S` nor its count, so the target of cell `ij` is `H_ij Σ̂_ij` with `H` the
pairwise count of finite rows — the reference's convention, and the per-cell denominator ADR 0117
made the moment layer's own.

**The realised return is centred on the location the forecast is about, always.** One verb,
`forecast_location(est, X)` for a refit and `forecast_location(est)` for a state, answers it per
family as fact 5 lists: `mean(ce.me, X)` or `state.mu`, and under a `CoveragePolicy` the fold's
`state.mu`, the diagonal of the per-pair centre; the weighted sample mean or `state.mu`;
`state.location`, or zero for every asset when `centred = true`, because the estimator then never
writes its location and the state keeps a `NaN` seed for an asset inactive at the first row; the
inner `ce`'s answer for a composite or a transform; `mu` for a prior; and the window's own sample
mean over the finite rows for an estimator that keeps no location. Under `Online`, whatever the
family, the location is the family's own data form over the buffer's rows and masks, because a
buffer means the batch verb over the buffer's rows for every read-out. The Asset Panel form of
the verb follows the estimator's own `cov` panel arm: reduce-and-expand under no policy, the
active mask under a policy and for the two mask-aware exponentially weighted families. There is
no `demean` flag: the location is the estimator's, not the caller's, and an estimator that assumed
a zero mean is evaluated at zero. An asset whose location is not finite is not active at the step,
beside one whose forecast variance is not: a return that cannot be centred cannot be scored.

**The diagnostics are the reference's five and three more.** Per step and free of `w`: the
Mahalanobis ratio `tr((H ⊙ Σ̂)⁻¹S) / N` — `tr(Σ̂⁻¹S) / (N h)` with no gap, and the reference's
`R'(H ⊙ Σ̂)⁻¹R / N` under its target — the diagonal ratio per asset `S_ii / (H_ii Σ̂_ii)` — kept
as `M × N`, where the reference keeps the mean over assets — the whole-matrix QLIKE, coded through
the identity `h log|Σ̂| + N h m_t` so that it is `h log|Σ̂| + tr(Σ̂⁻¹S)` with no gap and stays
finite under one, and the Frobenius loss `Σ_{H_ij > 0} (S_ij / H_ij − Σ̂_ij)²`, the per-cell form
of `‖S/h − Σ̂‖²_F`; the two losses #864 proved robust to the proxy. Per step and per test
portfolio: the standardised return `b_t = w'R / √(w'(H ⊙ Σ̂)w)` and the portfolio QLIKE, which
reads the per-row portfolio returns whichever target formed `S`. Over the run: the mean, median
and quantiles of each ratio, where the mean is the ratio of sums #864 §4.1 states — the per-step
ratios weighted by their degrees of freedom, `N_t h_t` under the realised covariance and `N_t`
under the horizon return, which is the plain mean at a fixed universe and horizon and is what the
reference reports — the Gaussian band `1 ± z_{α/2} √(2 / Σ_t dof_t)` on the mean ratio and
`√(2 / Σ_t dof'_t)` on one asset's or one portfolio's ratio, the bias statistic `B = std(b_t)` per
portfolio with its cross-portfolio quantiles at `(5, 25, 75, 95)`, the mean losses, and the
exceedance rate of `dof_t m_t` against `χ²_{dof_t}` at each of `levels`, `(0.95, 0.99)` by
default, kept as an `evaluations × levels` matrix beside the `levels` themselves rather than as
two named columns, because a caller who passes other levels would otherwise read a column named
for a level it does not hold. The kurtosis correction to the band is stated in the docstring and
not computed.

**The test portfolio.** `w = nothing` is inverse volatility recomputed every step from `Σ̂_t`;
a vector is one static portfolio; a vector of vectors is `P` portfolios. Each is renormalised
over the step's active assets, as the reference does. A portfolio *solved* per step by an optimiser
on the forecast — Engle and Colacito's minimum-variance test — is not built here; it needs the
optimiser's read-out under the online arm, and is fog on the map.

**The Result keeps its diagnostics per step and its forecasts on request.**
`CovarianceForecastEvaluationResult` holds `dates`, `test_idx`, `horizon`, `target`, `n_valid`,
`mahalanobis_ratio`, `diagonal_ratio`, `qlike`, `frobenius`, `standardised_return`,
`portfolio_qlike`, `w`, and `sigma` and `location`, both `Option{<:AbstractVector}` and `nothing`
unless `store_forecasts = true` keeps every `Σ̂_t` and the `c_t` it was centred on — a re-projection
needs both. `horizon` is a vector, one entry per step, and not the scalar `cv.test_size` the
decision named: a `DateWalkForward` over a calendar period cuts folds of 31, 30 and 31 rows by
nature, and a horizon per step is the `h_t` #864 §4.1 already wrote, so nothing is refused and the
summary weights each step by its own length. The step row type is a `NamedTuple`, because the
kernel's numeric types are known only after the first fit, as the optimiser entry's
`PredictionResult` is a `UnionAll`. This is the opposite of map #931's lazy ruling, for
the reason fact 4 gives, and the shape of the reference, so its numbers are an oracle. The
per-step kernel is exported as the level-1 verb `covariance_forecast_step(Σ̂, Z, c, w, target)`
on bare arrays, so a caller holding forecasts of their own builds the same Result by hand.

**Three verbs above it.** `covariance_forecast_summary(cfers...)` answers
`CovarianceForecastSummaryResult`, columnar, one entry per evaluation, so the length-2 case is the
side-by-side and the column names are one set (the shape [#941](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/941) built for the return forecasts).
`covariance_forecast_compare(a, b; lags = h − 1)` answers a Diebold–Mariano–West `z` and `p` on
the per-step loss difference for each loss, with a Newey–West variance at `h − 1` lags, and refuses
two evaluations whose dates or horizon differ. `covariance_forecast_portfolio(cfer, rd, w)`
re-projects stored forecasts and locations on a new `w` and refuses `sigma === nothing` by name.

**The identity.** Under `OnlineStep()` with an exact seam and the batch warm-up, the `M` rows of
every per-step column equal the batch expanding walk-forward's rows. Measured on 2026-09-11 by the
build ticket #1023 over eight assets with a listing and a delisting: **exact, to the bit**, for
`ExpWeightedCovariance`, `RegimeAdjustedExpWeightedCovariance` and a `Covariance` under a
`CoveragePolicy`, because each one's batch fit *is* the pass; and to `1e-15` on the ratios and
`1e-13` on the QLIKE for `GeneralCovariance`, `Covariance`, `PortfolioOptimisersCovariance` and
`EmpiricalPrior`, whose block fold is a row-by-row Welford recursion and not the sum-of-blocks
merge the decision assumed, so their identity is at the rounding of that recursion and not at the
bit. The rolling identity is `Online(est; max_history = w)` against `expand_train = false`, as ADR
0140 states it, and it is exact for every family measured, because the capped buffer's read-out
is the batch verb over the rolling window's rows. The reference is the oracle for
`HorizonReturn()` on a fixture whose location is zero: its per-step kernel agrees to `1e-15` on
all five numbers, with and without a missing cell, and its three summaries to the six digits it
prints. The centring term is asserted as the difference: with `R` the raw column sums,
`m_raw − m_c = (2 R'Σ̂⁻¹c − h c'Σ̂⁻¹c) / (N h)` at every step, whose expectation under `E[z] = c`
is the bias `c'Σ̂⁻¹c / N`.

## Considered options

On the loop:

1. **One verb through `fold_loop`, the scheme is the walk-forward** — chosen. One body, the
   identity by construction, `DateWalkForward`, purging and the rolling window for free, and
   the reference's two verbs collapse to one call.
2. **The reference's shape: two verbs, each with a private loop** — rejected. A ninth fold loop,
   the shape ADR 0141 removed; no date form; the identity by arithmetic coincidence.
3. **Map #931's shape: a fixed grid of refits on `1:t`, no scheme** — rejected. No rolling
   window, no purge, no online form without a second loop. #931 chose it because it had no
   scheme to reach; this evaluation does.

On the Result:

1. **Per-step diagnostics stored, forecasts on request** — chosen. `O(M (N + P))` by default,
   `O(N² M)` when the caller asks for it, and the re-projection verb is what the request buys.
2. **Lean: store every `Σ̂_t`, every diagnostic a verb** — rejected on fact 4. The rule that fits
   a `T × N` pairing does not fit an `N² × M` one.
3. **Per-step diagnostics only, no forecasts ever** — the reference. Rejected because a caller
   who can pay for the forecasts loses a re-projection they would otherwise rerun the loop for.

On the diagnostics: **the reference's five alone** was rejected because a loss projected on one
portfolio is blind to a correlation error the portfolio does not touch, and a summary without a
band or a test prints numbers a reader cannot judge; **the five plus the whole-matrix losses and
the band, without the comparison** was the recommended middle and was passed over for the test,
because two mean losses side by side invite exactly the reading the test exists to refuse.

On the realised quantity: **the horizon return as the default** was rejected because it is the
noisier statistic when the better one costs nothing; **the realised covariance alone** was
rejected because it makes the `h`-day holder's question inexpressible, a capability the reference
has.

On centring: **no centring** (the reference) and **a `demean` flag reading the prior's `mu`**
were both drafted and both rejected when the location turned out to be a field or a state
component of every family that fits a second moment. A flag would have let a caller evaluate a
forecast against a proxy for a moment the forecast does not estimate.

## Consequences

- `partial_fit!` gains a carrier arity on the prior and the covariance estimator, and
  `is_time_dependent`, `needs_previous_weights` and `advance_previous_fold` a method for a
  non-optimiser; `fold_loop` is otherwise untouched.
- `math_dict` gains four keys the two files share: `:Sigma_hat_t`, `:S_t_realised`, `:h_step`
  and `:M_steps`.
- `AbstractRealisedTarget`, `RealisedCovariance` and `HorizonReturn` are a new exported family,
  alongside `AbstractForecastTarget`'s three members; the abstract root is not exported.
- `forecast_location` is a new verb with a method per family that carries a location and a
  fallback that reads the window; a caller's `AbstractCovarianceEstimator` gets the fallback.
- `CONTEXT.md` gains **Covariance Forecast Evaluation** and **Realised Target**, and the *Avoid*
  line of **Forecast Calibration** names the covariance case.
- The reference's raw ratios differ from the library's by the centring term; a parity test
  states it.
- The build was [#1023](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1023) on
  map #861; the plots, the optimiser-solved test portfolio, a `Pipeline` in the estimator slot,
  and the covariance metrics as search scorers are fog there.
