# Risk-measure ↔ optimiser compatibility is a type-derived trait

Which risk measures an optimiser accepts was previously discoverable only from prose docs.
We expose it programmatically as `supports_risk_measure(::Type{<:Opt}, ::Type{<:RM}) -> Bool`
(with `supported_risk_measures(::Type{<:Opt})` returning the accepted abstract category), and
we generate the user-guide compatibility table from the same predicate so the table and
dispatch cannot drift.

The single primitive is `supported_risk_measures(::Type{<:Opt})`, keyed on the optimiser's
family supertype (the same abstract type its constraint builders dispatch on):
`RiskJuMPOptimisationEstimator → RiskMeasure`,
`ClusteringOptimisationEstimator → OptimisationRiskMeasure`, `Union{}` fallback. The predicate
is *derived* from it — `supports_risk_measure(O, R) === R <: supported_risk_measures(O)` — so
predicate, category, and dispatch share one source of truth and the trait is structurally
non-drifting. Keeping the primitive single-argument also sidesteps a two-argument dispatch
ambiguity: `NestedClustered <: ClusteringOptimisationEstimator`, so a meta override and the
clustering method would be ambiguous as a two-arg predicate, but are unambiguous as a
single-arg `supported_risk_measures` (the `NestedClustered` method is strictly more specific).

## Scope

Measure *acceptance* only — the clean two-level type bound. The orthogonal axes (a measure
needing a `HighOrderPrior`, per-optimiser `ub` support, the `MinScalariser`/NOC gap) keep
their existing runtime errors and belong to the `@risk_measure` macro (M5) and the
meta-optimiser legality seam (M11), not here — duplicating them would reintroduce the drift
this trait removes.

## Meta-optimisers delegate (intersection)

`NestedClustered`, `Stacking`, and `SubsetResampling` have no `r` of their own; acceptance is a
property of their constituent optimisers. Each meta accepts a measure only when *every* one of
its optimisers does, so `supported_risk_measures` returns the **`typeintersect`** of its
children's categories: `NestedClustered` intersects inner (`opti`) and outer (`opto`);
`Stacking` intersects outer (`opto`) with all inner (`opti`); `SubsetResampling` delegates to its
single inner optimiser (`opt`). Because the predicate is `R <: supported_risk_measures(O)`, this
means exactly "R is accepted by every child".

Methods are keyed on the **concrete** meta types (not the `Base*` abstract supertypes) and read
the child optimiser field types, which `@concrete` exposes as type parameters. This is exact for
a fully parametric meta type or a constructed instance; an under-specified bare meta `UnionAll`
(children unresolved) conservatively yields `Union{}`. A heterogeneous `Stacking` inner vector
spanning different optimiser *families* also collapses to `Union{}` via the vector `eltype`, the
one known imprecision. Meta-optimisers are excluded from the generated table (their acceptance is
instance-specific) with a pointer note.

## Consequences

- `InteractiveUtils.subtypes` (used to enumerate concrete measures for the table) stays a
  docs-only concern; the runtime predicate/enumeration never depend on it.
- Adding a new leaf optimiser family gets correct compatibility for free by subtyping the right
  abstract type; a new *meta*-optimiser must add an explicit "delegated" method.
