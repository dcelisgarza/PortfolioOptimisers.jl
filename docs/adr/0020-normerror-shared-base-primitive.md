---
status: accepted
---

# NormError is a shared Base primitive, not a tracking detail (LxTracking → LxNorm)

## Context

The norm-based error formulations — L1, L2, squared-L2, general-Lp and L∞ — lived in
[18_Tracking.jl](../../src/15_Tracking.jl) as `L1Tracking`, `L2Tracking`, `SquaredL2Tracking`,
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

- `abstract type NormError <: AbstractEstimator` moves to [01_Base/12_NormError.jl](../../src/01_Base/12_NormError.jl),
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

## Amendment (2026-09-23)

Issue #1271 found that the family had no rule for the factor that divides each norm, and that
`LInfNorm` broke the pattern the others follow. The family now has one rule: **the norm of order
`p` divides by `(T - d)^(1/p)`**, where `T` is the number of observations and `d` is `ddof`.
`L1Norm` and `L2Norm` are the templates.

| Norm            | Factor          | Default `ddof` |
| :-------------- | :-------------- | :------------- |
| `L1Norm`        | `T - d`         | `0`            |
| `L2Norm`        | `sqrt(T - d)`   | `1`            |
| `SquaredL2Norm` | `T - d`         | `1`            |
| `LpNorm`        | `(T - d)^(1/p)` | `1`            |
| `LInfNorm`      | `1`             | no field       |

- **`LInfNorm` divides by `1`**, the limit of `(T - d)^(1/p)` as `p` grows. Its old factor was
  `T - d`, which no source gave. With 252 daily returns, `err = 0.01` then allowed a worst day of
  2.52, so the constraint almost never bound. `err` is now the worst single-period difference.
  The `ddof` field had no effect after that change, so it was removed: `LInfNorm()` takes no
  arguments. This is breaking for a caller that passed `ddof`.
- **`L1Norm` gains `ddof`**, default `0`. The default keeps the denominator `T` of the source
  (Cajas 2025, Eq. 9.17), so no present `L1Norm` number moves.
- **`LpNorm` defaults to `ddof = 1`**, so `LpNorm(; p = 2)` equals `L2Norm()`. This is breaking for
  a caller that relied on the old default `0`. `L1Norm` keeps `0`, so `LpNorm(; p = 1)` and
  `L1Norm()` differ by `T / (T - 1)`. We kept that difference on purpose: the `T - 1` correction
  gives the sample standard deviation, and it has no such reason for a mean absolute deviation.
- **`SquaredL2Norm`** is the square of the `L2Norm` error, so its factor is the square of the
  `L2Norm` factor. It is the one member that does not follow the rule in `p`.
- **`norm_factor` is the one source of the factor.** `norm_error` read it before. Now the JuMP
  models of `TrackingError` and `TrackingRiskMeasure` read it too, at every norm, where each had
  written the factor by hand. The drift that #1271 found lived in those copies. The second-order
  cone of `SquaredL2Norm` reads the square root of its factor.
- Entropy pooling reads the same `norm_factor`, so an `LpNorm` or `LInfNorm` passed as its `err`
  takes the new factor.
