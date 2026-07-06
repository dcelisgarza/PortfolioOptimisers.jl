---
status: accepted
---

# NormError is a shared Base primitive, not a tracking detail (LxTracking → LxNorm)

## Context

The norm-based error formulations — L1, L2, squared-L2, general-Lp and L∞ — lived in
[18_Tracking.jl](../../src/18_Tracking.jl) as `L1Tracking`, `L2Tracking`, `SquaredL2Tracking`,
`LpTracking`, `LInfTracking`. Their names and location said "these are how you measure *tracking
error*." But a norm on a residual vector is not tracking-specific: it is the same primitive
whether the residual is `w − w_ref` (tracking), a risk-measure target, or the gap between a set of
`ConditionalValueatRisk` view estimates that entropy pooling must reconcile.

The concrete trigger was entropy pooling. `EntropyPoolingPrior` reconciles **multiple CVaR views**
by minimising the norm of their disagreement, and the old code hard-coded a bare
`LinearAlgebra.norm(...)` with a `#! Customise with different norms (see L2Tracking)` comment —
i.e. the author already knew the tracking norms were the right abstraction but they were trapped
under a tracking-flavoured name in a tracking file, unreachable from the prior layer without
implying "tracking" where none exists.

## Decision

**Promote the norm family to a first-class `Base` vocabulary primitive named `NormError`, and
rename the `Lx*` types from `*Tracking` to `*Norm`.**

- `abstract type NormError <: AbstractEstimator` moves to [01_Base.jl](../../src/01_Base.jl),
  alongside the other cross-cutting reducers (the Vector-to-Scalar Reducers already live here).
  Concrete members: `L1Norm`, `L2Norm`, `SquaredL2Norm`, `LpNorm(p, ddof)`, `LInfNorm`.
- The evaluation seam is `norm_error(f::NormError, a, b, T)` (and the one-argument
  `norm_error(f, a, T)`), so any layer can apply a caller-chosen norm to a residual.
- The three roles now share the one family: tracking risk measures/constraints, risk-measure
  targets, and `EntropyPoolingPrior` gain `err::Option{<:NormError} = nothing` (default = L2)
  used to reconcile multiple CVaR views via `norm_error(...)` instead of a hard-coded `norm`.
- `CONTEXT.md` is updated: the glossary gains an **"LxNorm error family"** entry and the
  **Tracking Error** entry is reworded to point return-tracking formulations at the LxNorm family
  rather than owning `L*Tracking` names.

## Considered options

- **Leave the norms in `18_Tracking.jl` and reach into them from the prior layer.** Rejected: it
  forces the prior/entropy-pooling code to depend on the tracking module and names a "tracking"
  concept in a context (view reconciliation) where tracking is meaningless — exactly the confusion
  the `#! see L2Tracking` comment betrayed.
- **Duplicate a norm helper in the prior layer.** Rejected: two implementations of the same Lp
  maths drift, and it denies entropy-pooling users the `LpNorm(p, ddof)` / `LInfNorm` variants the
  tracking side already had.
- **Keep the `*Tracking` names but move the file.** Rejected: the name is the lie. Once the family
  is used outside tracking, `L2Tracking` in an entropy-pooling call reads as a bug.

## Consequences

- **Breaking rename** `L{1,2,p,Inf}Tracking` / `SquaredL2Tracking` → `L{1,2,p,Inf}Norm` /
  `SquaredL2Norm`. This is the same class of decision as [ADR 0015](0015-disambiguation-suffix-naming.md)
  but the axis is *generalisation* (a concept escaped its origin module), not disambiguation of
  colliding prefixes.
- Entropy pooling of multiple CVaR views is now customisable per the same norm family used for
  tracking, closing the `#!` TODO. Single-view pooling is unaffected (the norm only matters when
  there is more than one view to reconcile).
- New shared surface: extension authors adding a norm write one `NormError` subtype +
  `norm_error` method and it is immediately usable in tracking, risk targets and priors.
