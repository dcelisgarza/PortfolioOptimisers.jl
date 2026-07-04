---
status: accepted
---

# Uncertainty-set estimators own their posdef and moment configuration

## Context

[NormalUncertaintySets.jl](../../src/14_UncertaintySets/03_NormalUncertaintySets.jl) reached
four levels into its prior estimator to borrow a posdef projector: all 21 posdef calls were
`posdef!(ue.pe.ce.mp.pdm, sigma)`, cleaning the uncertainty set's own simulated (Wishart)
matrices with whatever `Posdef` estimator happened to sit inside the prior's covariance's
matrix-processing. [BootstrapUncertaintySets.jl](../../src/14_UncertaintySets/04_BootstrapUncertaintySets.jl)
(`ARCHUncertaintySet`) reached the prior instance too — `Statistics.mean(ue.pe.me, Xi)` and
`Statistics.cov(ue.pe.ce, Xi)` to recompute per-bootstrap-sample moments, and
`Statistics.cov(ue.pe.ce, X_mu)` for the ellipsoid deviation covariances (where the posdef runs
transitively inside the covariance estimator's matrix processing).

[ADR 0009](0009-matrix-processing-order-as-step-symbols.md) explicitly *preserved* the
`mp.pdm` field layout **because** of these external readers, and named an accessor as the
deferred half of that decision. Reviewing that half surfaced two problems with an accessor
(`posdef_estimator(x)` / `@forward_properties` threading down the prior tree):

1. The reach is fragile in the ADR-0009 sense — renaming any intermediate field
   (`pe`/`ce`/`mp`/`pdm`) breaks 21 sites — and the accessor only relocates the coupling, it
   does not remove it.
2. `NormalUncertaintySet.pe` accepts any `AbstractLowOrderPriorEstimator`, and nine prior types
   forward a `.ce`, so a correct accessor would have to define the "which posdef" answer for all
   nine. That answer diverges (`FactorPrior` and the Black–Litterman variants clean their
   *posterior* covariance with their own `pe.mp`, not the inner asset prior's `ce.mp.pdm` the
   reach lands on), yet **only `EmpiricalPrior` is ever passed to an uncertainty set** across the
   whole test and docs corpus. The accessor would invest per-prior semantics and tests in paths
   nobody exercises.

The posdef reach in `NormalUncertaintySet` is *incidental* — it needs *a* projector for its own
matrices and grabbed the prior's. The `ARCHUncertaintySet` reach is *intentional* moment-matching
— it recomputes the moments it bounds — but is the same coupling class and the same fragility.

## Decision

**Uncertainty-set estimators carry their own posdef / moment configuration instead of reaching
into the prior instance.** No accessor; no changes to the prior estimators.

- `NormalUncertaintySet` gains `pdm::Option{<:AbstractPosdefEstimator} = Posdef()`. All 21
  `posdef!(ue.pe.ce.mp.pdm, …)` become `posdef!(ue.pdm, …)`. `Posdef()` is the default already
  reached (`EmpiricalPrior() → PortfolioOptimisersCovariance() → MatrixProcessing() → Posdef()`),
  so default usage is byte-identical; `nothing` opts out of cleaning via the existing
  `posdef!(::Nothing, …)` no-op.
- `ARCHUncertaintySet` gains `me::AbstractExpectedReturnsEstimator = SimpleExpectedReturns()` and
  `ce::AbstractCovarianceEstimator = PortfolioOptimisersCovariance()`. Every `ue.pe.me` / `ue.pe.ce`
  in the bootstrap generators and deviation covariances becomes `ue.me` / `ue.ce`. `pe` stays for
  the central prior the ellipsoid is built around. The concrete defaults reproduce
  `EmpiricalPrior()`'s `me`/`ce`, so default usage is byte-identical.
- `DeltaUncertaintySet` is unchanged — it only calls `prior(ue.pe, …)` and never reaches the
  prior's internals.

Alongside, the 12 near-duplicate `ucs`/`mu_ucs`/`sigma_ucs` methods in `NormalUncertaintySets.jl`
factor their shared assembly into two builders (`mu_asymptotic_cov(pdm, sigma, T)`,
`sigma_asymptotic_cov(pdm, sigma_mu, sigma, T)` — posdef-cleaned, pre-diagonal), leaving each
method only its geometry-specific `k`/simulation step. That is a pure internal dedup, not part of
the coupling decision.

## Considered options

- **Accessor `posdef_estimator(x)` / `@forward_properties` threading the prior tree.** Rejected:
  relocates rather than removes the coupling, and requires correct per-prior semantics across nine
  prior types for a path only `EmpiricalPrior` ever exercises. It also cannot express
  `ARCHUncertaintySet`'s need, which is a full covariance/mean *estimator*, not a bare `pdm`.
- **Own fields defaulting to `nothing` = fall back to the prior instance.** Rejected: guarantees
  zero divergence even under a customised prior, but the default path still reaches
  `ue.pe.ce.mp.pdm` / `ue.pe.ce`, leaving the fragility the change exists to remove.
- **Leave `ARCHUncertaintySet` coupled, decouple only `NormalUncertaintySet`.** Rejected: same
  fragility class; splitting the treatment leaves the review's concern half-addressed.

## Consequences

- The uncertainty-set subset of ADR 0009's "~24 external `mp.pdm` readers" (21 `NormalUncertaintySet`
  sites, plus the transitive `ARCHUncertaintySet` covariance reads) is **removed**; ADR 0009's
  rationale for preserving `mp.pdm` now rests only on the *internal* prior readers (Factor / BL /
  high-order priors cleaning their own posterior covariances), which are unchanged.
- Byte-identical for every current test and example — all pass the default prior.
- Behaviour diverges from a *customised* prior only when the user sets a non-default `pe.ce`/`pe.me`
  and does **not** mirror it on the uncertainty set. This is now an explicit, visible choice: the
  uncertainty set's projector / moment estimators are its own fields. A user wanting the old
  coupling passes the same estimator to both.
- New public fields on two estimators (`NormalUncertaintySet.pdm`; `ARCHUncertaintySet.me`, `.ce`).
  The docstring doctest trees for both types gain the corresponding lines; the doctest suite is the
  regression net.
