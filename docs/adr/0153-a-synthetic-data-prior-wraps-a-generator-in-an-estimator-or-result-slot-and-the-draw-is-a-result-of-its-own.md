---
status: proposed
---

# A synthetic-data prior wraps a generator in an estimator-or-Result slot, and the draw is a Result of its own

## Context

Map [#1080](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1080) builds a
vine-copula synthetic-data prior. ADR 0152 fixed the dependency shape and which layer of the vine
fit this library owns. Decision ticket
[#1086](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1086) had to fix the
estimator, the fitted vine as a Result, the prior Result, and how "fit once, stress many" is
expressed, before the build tickets could name a type.

The facts the ruling rests on, read in the source on 2026-09-15:

- **The library already says "fit once, use many" one way.** An optimiser's `pe` slot is bounded
  `PrE_Pr = Union{<:AbstractPriorEstimator, <:AbstractPriorResult}`, and `prior` has a
  pass-through method on the Result; a hierarchical optimiser's `cle` slot is `ClE_Cl` on the
  same pattern; an uncertainty set with `pe = nothing` is calibrated on the result it is handed
  (ADR 0138). A prior estimator that wraps another prior (`FactorPrior`, `HighOrderPriorEstimator`,
  `EntropyPoolingPrior`) bounds its `pe` on estimators only.
- **`LowOrderPrior` cannot carry the caller's panel as `o_X` beside a synthetic `X`.** Its
  constructor demands `size(o_X) == size(X)` and `!isnothing(rr)`
  (`src/10_Prior/01_Base_Prior.jl`), and `FactorAttribution` computes `original_X - X` row by row.
  A synthetic panel has `n_sim` rows against the caller's `T`, so the field's contract — same
  observations, same assets — does not hold, whatever the guard says.
- **`ens` is read by one consumer.** The normal uncertainty set's `choose_scaling_parameter` reads
  `ue.ens`, else `pr.ens`, else `size(pr.X, 1)`. ADR 0138 binds `ens` to `w` as a diagnostic of
  the rows the moments were fitted over.
- **`LowOrderPrior` already carries estimator-specific fields.** `kld` is set by entropy pooling
  alone, `rr`/`fpr` by the factor block alone, `o_X` by the reconstructing priors alone; every
  other estimator leaves them `nothing`, `forward_prior` forwards them, `port_opt_view` slices or
  passes them.
- **Entropy pooling refits its nested estimator.** `EntropyPoolingPrior` fits `pe.pe` once, reads
  the prior probabilities `w0` off that Result's `w` (else uniform), solves the tilted `w1` on
  those rows, then calls `factory(pe, w1)` and `prior(pe.pe, X, …)` a second time so the weighted
  moment estimators see `w1`. A nested estimator that draws from a generator draws again on that
  second call: with a `seed`, `resolve_rng` reseeds a private copy and the panel is bit-identical;
  without one, `rng` is used as it stands and the second panel is a different draw, so `w1`
  weights rows it was never solved for, with no error.
- **A fixed asset breaks the default covariance path inside `prior`.** A Stress Statement that
  fixes an asset at one value makes that column constant across every scenario, so `sigma` has a
  zero row and column. `PortfolioOptimisersCovariance` defaults to `MatrixProcessing(pdm =
  Posdef())`, and `posdef!` (`src/04_MatrixProcessing/01_PosdefMatrix.jl`) fails `isposdef`, takes
  `s = sqrt.(diag(X))` and calls `cov2cor!(X, s)`, which divides the zero row and column by
  `s = 0`; the fit returns a `sigma` of `NaN` behind a `@warn`. Downstream, `cholesky(sigma)` in
  the variance constraint and the relaxed risk-budgeting constraint throws `PosDefException` on
  an exactly singular matrix. Risk measures that read only `X` are unaffected. The reference
  implementation repairs this with `1e-6 · N(0, 1)` noise from an unseeded global generator, one
  vector broadcast onto every constant column.
- **The reference implementation** takes a distribution estimator, `n_samples` and a
  `sample_args` dictionary; conditioning lives in `sample_args`; every re-stress is a refit; its
  `sample` returns a bare array with no read of how it was drawn; the prior is the plain sample
  mean and covariance of that array.
- **The maintainer** named two synthetic-data generators the prior should later admit without
  changing shape: a bootstrap-based one, and a time-series forecasting one on a Julia package that
  simulates paths.

## Decision

**The prior is generic over synthetic-data generators.** `SyntheticDataPrior <:
AbstractLowOrderPriorEstimator_A` holds:

- `est::Union{<:AbstractSyntheticDataEstimator, <:AbstractSyntheticDataResult}` — the generator
  as a recipe, or the generator already fitted. Both roots are the library's own and live in
  `src/`. `VineCopulaEstimator <: AbstractSyntheticDataEstimator` is the first member and carries
  **every** vine knob: marginal candidates and their criterion, pair-copula families, selection
  criterion, independence level, tau inversion or maximum likelihood, structure, central assets, a
  fixed structure, truncation, tree criterion, the log transform, the observation weights `w`
  (the moment-estimator idiom, `obs_weights_view` aware), and the Stress Statement. Its fitted
  Result subtypes `AbstractVineCopulaResult <: AbstractSyntheticDataResult` in `src/`, and the
  extension owns the one concrete `VineCopulaResult` (ADR 0152). A later bootstrap or
  path-simulating generator joins the family with its own configuration and no change to the
  prior.
- `n_sim` — the scenario count, default 1000, capped by `assert_resource_cap(n_sim,
  RESOURCE_LIMITS[].max_n_sim, :n_sim, :max_n_sim)` (ADR 0041).
- `rng` and `seed` — the library's pair, resolved once by `resolve_rng` at the top of `prior`.
- `jitter` — the constant-column repair, below.
- `pe::AbstractLowOrderPriorEstimator_A` — the inner moment estimator, default `EmpiricalPrior()`,
  fitted on the synthetic panel. This is the first improvement over the reference, whose moments
  are the plain sample moments.

**Two verbs.** `synthetic_data(est, X) -> AbstractSyntheticDataResult` fits the generator, with a
pass-through method on a Result so `prior` calls it once and never branches. `simulate(rng, sr, n)
-> (X, SimulationResult)` draws `n` scenarios from a fitted generator; the matrix is observations ×
assets, and `SimulationResult` is the read of how the draw ran: the route (exact, rejection or
importance), the acceptance rate, the draws taken, and the mask of columns the repair touched.
This pays the shape question of [#306](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/306):
the generator is an estimator, randomness is a field pair, the draw returns a Result beside its
matrix.

**The Stress Statement lives on `VineCopulaEstimator`, and the fit copies it onto the fitted
Result.** Conditioning is vine knowledge — the central-asset boost at fit time is what makes the
exact route admissible — so it is not a field of the generic prior. `simulate` reads the Result's
own `stress`. **Fit once, stress many** is therefore `vr = synthetic_data(VineCopulaEstimator(;
stress = s1, …), X)`, then `@set vr.stress = s2`, then `SyntheticDataPrior(; est = vr2)`: the fit
is never repeated. A statement that names assets the fit did not centre may make the exact route
inadmissible, and the sampler then declares rejection or importance in `SimulationResult` rather
than refusing. Ticket [#1087](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1087)
decides the statement's forms and the route rules.

**The importance route resamples by default.** A `Bool` knob on the route (its name is #1087's;
it is not `w`, because it carries no weights) switches to a weighted panel, where `w` holds the
normalised importance weights and `ens` their Kish count — the pair ADR 0138 already binds — and
`pe.pe` is rebound with those weights before the moment fit. The resampled default leaves
`w = nothing` so a consumer that ignores `w` sees a stressed panel, not a proposal panel.

**The prior Result is a `LowOrderPrior` with one new field, `sim::Option{<:SimulationResult}`.**
`X` is the synthetic panel; `o_X = nothing`, because the caller's panel is not over the same
observations and the fitted generator is what remembers the fit; `mu`, `sigma`, `chol`, `w`,
`ens`, `kld`, `ow` are whatever `prior(pe.pe, X_synth)` answered, forwarded whole through
`forward_prior` (ADR 0046) — so `ens` is the inner estimator's own count over the synthetic rows,
`nothing` for a plain `EmpiricalPrior`; `rr`/`fpr` are `nothing`. `sim` is the fourth
estimator-specific field of the carrier, on the `kld` precedent: default `nothing`, forwarded by
`forward_prior`, passed by `port_opt_view` through a `port_opt_view` method of its own that slices
the repaired-column mask on the asset axis and keeps the route and rates, which are not per asset.
`SimulationResult` is a struct, not loose values, so a reader dispatches on it.

**Entropy pooling's refit reuses the drawn panel, by dispatch on `sim`.** The refit step of
`EntropyPoolingPrior` (both algorithms) becomes a verb dispatched on the first fit's Result: on a
`LowOrderPrior` whose `sim` is `nothing` it does what it does today, `factory(pe, w1)` then
`prior(pe.pe, X, …)`; on one whose `sim` is a `SimulationResult` it refits the inner moment
estimator on `pr.X` under `w1` and rebuilds the carrier with the same `sim`. No seed is required,
no draw is repeated, and no estimator that does not draw changes behaviour.

**The constant-column repair is `jitter` on `SyntheticDataPrior`, per column, from the
estimator's own generator, nullable, on by default.** It runs after the draw and before `pe.pe`,
so every generator gets it. Each constant column takes its own `Normal(0, s)` draw from the
resolved `rng`, so two fixed assets are independent noise and every run is reproducible from the
estimator's fields. The default scale is the reference's `1e-6`; `nothing` disables it, for a
caller who regularises inside `pe.pe` or runs `X`-only risk measures, and the docstring names the
`NaN` outcome of the default covariance path. The first build ships a scalar and a per-asset
vector; three richer forms graduate to a ticket of their own: per-asset pairs `"asset" => value`
resolved through a required `sets::UniverseSets` (the estimator form), a scalable distribution in
place of the normal, and a calibrated scale read off the panel. The docstring carries the
modelling caveat in every form: the repair fixes the factorisation, not the semantics — a fixed
asset reads to a moment-based optimiser as a risk-free bet at the stated value.

**Composition is free and is asserted by tests, not by code.** `FactorPrior(; pe =
SyntheticDataPrior(…))` projects a factor stress onto the assets through the loadings because the
prior is in the `_A` family; `HighOrderPriorEstimator(; pe = SyntheticDataPrior(…))` takes the
coskewness and cokurtosis of the synthetic panel; `EntropyPoolingPrior(; pe = SyntheticDataPrior(…))`
tilts a stressed panel. Each has a test; none has a line of code beyond the refit dispatch above.
A fitted generator in `est` is pinned across cross-validation folds and subset-resampling
subproblems, exactly as `cle = ClusteringResult` is; the docstring says so, and a caller who wants
a refit per fold passes the estimator.

## Considered options

- **One estimator, the fitted vine as a data argument** (`prior(pe, vr::AbstractVineCopulaResult)`
  samples without refitting; no Result in any estimator field). Rejected: a pre-fitted vine could
  not be composed into `FactorPrior` or `HighOrderPriorEstimator`, which call `prior(pe, X)`; and
  the `PrE_Pr`/`ClE_Cl` slot is the library's own idiom for a handed-in fit, which the
  "estimators never hold Results internally" rule (ADR 0106) does not refuse.
- **Both** the slot and the data-argument method. Rejected: two ways to say one thing.
- **The Stress Statement on the prior, generic over generators**, with a rejection/importance
  fallback written once on the abstract Result and an exact route per generator. Considered
  seriously: rejection and importance need only an unconditional draw and are defined for a
  bootstrap or a path simulator. Rejected by the maintainer: a stress is vine knowledge and the
  prior stays a plain wrapper; a later generator that conditions carries its own statement.
- **The fitted Result without a stress field**, re-stressed by a new estimator on the fixed
  structure. Rejected: a refit of the pairs, not fit-once-stress-many.
- **The route read on the draw Result only**, with a `@warn` on fallback and no new carrier
  field. Rejected: the read should survive into the prior an optimiser holds, and the carrier has
  the `kld` precedent for an estimator-specific field.
- **A weighted importance panel by default.** Rejected: a consumer that ignores `w` would see the
  proposal panel; the knob keeps it available.
- **Closing the refit hazard with a predicate that makes entropy pooling refuse an unseeded
  synthetic nest**, or **by documentation alone**. Rejected for the dispatch: it closes the hazard
  without a seed and without touching any nest that does not draw.
- **`ens` as the fitted count `T`.** Rejected: it would override what the inner estimator stated
  and is not bound to the prior's `w`; the fitted count lives on the fitted generator.
- **No constant-column repair**, or **off by default**. Rejected: the default covariance path
  returns `NaN` inside `prior`, so a scalar stress would break every moment-based optimiser out of
  the box.

## Consequences

- `LowOrderPrior` gains `sim`; `forward_prior`, `port_opt_view`, `field_dict` and every docstring
  that lists the carrier's fields move with it. The entropy-pooling refit becomes a dispatched verb.
- `src/10_Prior/` gains `AbstractSyntheticDataEstimator`, `AbstractSyntheticDataResult`,
  `AbstractVineCopulaResult`, `VineCopulaEstimator`, `SyntheticDataPrior`, `SimulationResult`, the
  verbs `synthetic_data` and `simulate`; the extension gains `VineCopulaResult` and the methods.
- `CONTEXT.md` gains **Synthetic-Data Generator**, **Synthetic-Data Prior** and **Simulation
  Result** under §3.6; the Stress Statement term is #1087's.
- The build tickets [#1090](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1090),
  [#1091](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1091) and
  [#1092](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1092) are sharpened to these
  names; the jitter's richer forms are a new ticket behind #1091.
- #306 is paid in shape here and closes with the map.
