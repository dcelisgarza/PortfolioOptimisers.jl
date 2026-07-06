---
status: accepted
---

# One `scalarise` seam for reducing a vector of risk measures

## Context

A `Scalariser` (`SumScalariser`, `MaxScalariser`, `MinScalariser`, `LogSumExpScalariser`) collapses
a *vector* of risk measures into the single scalar an optimiser minimises. Three optimisers needed
this — `HierarchicalRiskParity`, `HierarchicalEqualRiskContribution` (inner and outer levels) and
`NearOptimalCentering` — and each carried its own copy: `hrp_scalarised_risk`,
`herc_scalarised_risk_i!`, `herc_scalarised_risk_o!`, plus NOC's variants. Each of those was
further split by scalariser type *and* by whether the risk came as a vector or a matrix, so the
four scalarisers × three sites × two shapes produced ~20 near-identical methods.

The duplication was not benign: the `LogSumExpScalariser` path was subtly wrong in HERC — the
per-optimiser copies applied the `logsumexp` reduction over the wrong risk slice, so HERC's
LogSumExp-scalarised weights were incorrect (the reference fixture changed once fixed). With the
logic copied four ways, the bug lived in some copies and not others, and there was no single place
to fix it.

## Decision

**Provide one generic `scalarise(f, ::Scalariser, itr; by)` seam in
[01_Base_RiskMeasures.jl](../../src/19_RiskMeasures/01_Base_RiskMeasures.jl) and delete the
per-optimiser copies.**

- `scalarise(f, ::SumScalariser, itr)`, `::MaxScalariser`, `::MinScalariser` reduce with the obvious
  operator; `scalarise(f, sca::LogSumExpScalariser, itr)` is the one correct implementation:
  `scalarise_map(x -> x / γ, scalarise_logsumexp([scalarise_map(x -> γ*x, f(el)) for el in itr]))`.
- Helpers `scalarise_map`, `scalarise_combine` and `scalarise_logsumexp` handle the vector-vs-tuple
  shape polymorphism once, so callers no longer branch on it. `scalarise_logsumexp` reduces each
  component across the collection (`logsumexp` per index), which is the reduction HERC's copies got
  wrong.
- HRP, HERC (both levels) and NOC call `scalarise` with their site-specific `f`; their bespoke
  `*_scalarised_risk*` methods are removed.

## Considered options

- **Fix the bug in place, leave the four copies.** Rejected: it re-fixes one copy of a bug whose
  root cause is that the reduction is written four times — the next scalariser or optimiser
  reintroduces the drift.
- **A macro generating the per-optimiser methods.** Rejected: it removes the edit duplication but
  keeps four generated bodies to audit and still lets the LogSumExp reduction be specified
  per-site; a single runtime seam makes "there is exactly one LogSumExp reduction" true by
  construction.

## Consequences

- Correctness: the `LogSumExpScalariser` reduction is defined once and shared, fixing HERC. The
  `HierarchicalEqualRiskContribution2` reference fixture was regenerated to the corrected values,
  and [test_09d_scalarise.jl](../../test/test_09d_scalarise.jl) pins the seam directly.
- ~350 lines removed across HRP / HERC / NOC; adding a new scalariser is now one `scalarise` method
  rather than a change at every optimiser site.
- **Internal only** — `Scalariser` types and their meaning are unchanged; this is a consolidation of
  how they are applied, with a behaviour fix confined to the previously-incorrect HERC LogSumExp
  path.
