---
status: accepted
---

# One deep optimisation Result, mirroring the JuMP optimiser estimators

## Context

Architecture-review candidate 2 (`docs/architecture-review-20260613-135207.html`). The continuous
optimisation result types under `NonFiniteAllocationOptimisationResult` repeat the same shape ~10
times: a `:w`-virtual `getproperty` (extracting weights from `sol`, falling through to `pa`), a
`factory(res, fb)` that rebuilds the struct only to swap the fallback optimiser, and a near-identical
field block (`oe, pa, retcode, sol, model, fb`). The four JuMP results — `MeanRiskResult`,
`RiskBudgetingResult` (also produced by `RelaxedRiskBudgeting`), `NearOptimalCenteringResult`,
`FactorRiskContributionResult` — differ only in a handful of unique fields and the order of the
forward chain. The bodies are copy-pasted; a forgotten field is a silent wrong slice.

The estimator side already solved the analogous problem: shared JuMP configuration lives in one
`JuMPOptimiser <: BaseJuMPOptimisationEstimator` struct (a *separate, non-invokable* branch off
`BaseOptimisationEstimator`), embedded as field `.opt` in each concrete optimiser, while the
optimisers themselves subtype `RiskJuMPOptimisationEstimator <: JuMPOptimisationEstimator`. See
`JuMPOptimiser`, `RiskJuMPOptimisationEstimator` and `[[factory]]` in `CONTEXT.md`.

## Decision

Mirror the estimator structure on the result side.

**Type hierarchy** (additive — existing dispatch is on `NonFiniteAllocationOptimisationResult` /
`OptimisationResult`, which remain supertypes; no method dispatches on the four concrete JuMP result
types today):

```text
AbstractResult
├── BaseJuMPOptimisationResult                 (new abstract — off AbstractResult, not in the result hierarchy)
│    └── JuMPOptimisationResult                (new struct: oe, pa, retcode, sol, model)
└── OptimisationResult
     ├── NonFiniteAllocationOptimisationResult
     │    ├── RiskJuMPOptimisationResult       (new abstract)
     │    │    ├── MeanRiskResult                 (jr, fb)
     │    │    ├── RiskBudgetingResult            (jr, prb, fb)
     │    │    ├── NearOptimalCenteringResult     (jr, w_min/opt/max/noc_retcode, fb)
     │    │    └── FactorRiskContributionResult   (jr, rr, frc_plr, fb)
     │    └── NonJuMPOptimisationResult        (new abstract)
     │         ├── NaiveOptimisationResult
     │         ├── HierarchicalResult
     │         ├── SchurComplementHierarchicalRiskParityResult
     │         ├── NestedClusteredResult
     │         ├── StackingResult
     │         └── SubsetResamplingResult
     └── FiniteAllocationOptimisationResult   (DiscreteAllocationResult, GreedyAllocationResult — unchanged)
```

- **Embedded core.** `JuMPOptimisationResult` holds the five always-present fields
  (`oe, pa, retcode, sol, model`). It is the analogue of `JuMPOptimiser`: it sits on its own
  `BaseJuMPOptimisationResult` branch and **does not** enter the result hierarchy. Each concrete JuMP
  result embeds it as its **first** field, named `jr`.
- **`fb` stays the last field of each concrete result** (not in the core), because *every*
  optimisation result — JuMP, non-JuMP, and finite-allocation — already ends in `fb`.
- **One generic `factory(res, fb)`** on `OptimisationResult` rebuilds via the fb-last convention
  (`constructorof(typeof(res))(Base.front(fields)…, fb)`), collapsing ~11 hand-written `factory`
  methods. Two bodies (NonFinite `Option{<:OptE_Opt}` vs Finite `Option{<:FOptE_FOpt}`) preserve the
  fb-type distinction; overridable per concrete type if a result ever needs more than an fb swap.
- **`getproperty` forwarding kept, but deepened.** The common forwarding (`:w` from `sol`, the five
  core fields, fall-through to `pa`) lives **once** on `JuMPOptimisationResult`. A default thin
  `getproperty` on `RiskJuMPOptimisationResult` resolves unique fields by `getfield` and delegates the
  rest to `jr` — serving `MeanRiskResult` and `NearOptimalCenteringResult` directly. Only `FRC` and
  `RB` override it, to forward into `rr` / `prb` before delegating. The divergent `:w` variants (NOC
  used `getproperty`, RB had no vector handling) unify to the vector-aware form.
- **Construction** wraps explicitly at the ~6 `_optimise` sites:
  `MeanRiskResult(JuMPOptimisationResult(typeof(mr), attrs, retcode, sol, model), fb)`.

## Considered options

- **Literal estimator mirror — drop `getproperty` forwarding.** The estimator side does no
  forwarding (`mr.opt.pe` is explicit). Applying that to results would force `res.jr.sol.w`,
  `res.jr.pa.pr`, etc. **Rejected:** ~200+ access sites rely on `res.w` / `res.pr` / `res.retcode`;
  the `:w` virtual is a result-only concern with no estimator analogue.
- **Macro-driven forward chains** (extend the `@propagatable` / `@fprop` tag machinery of ADR 0002 /
  0010). **Rejected for now:** only two types (`FRC`, `RB`) need a non-default chain — a trait or two
  explicit overrides are lighter than new macro surface.
- **No `NonJuMPOptimisationResult` umbrella** (mirror the estimators exactly, where naive / clustering
  / meta hang directly off `NonFiniteAllocationOptimisationEstimator` with no non-JuMP umbrella).
  **Rejected:** we accept the asymmetry deliberately for a tidy JuMP-vs-non-JuMP dispatch handle, even
  though the umbrella carries no shared behaviour.
- **FullMoment `BaseOptimisationResult` mirror layer.** **Rejected (YAGNI):** nothing else needs it yet, so
  `BaseJuMPOptimisationResult <: AbstractResult` directly.

## Consequences

- The `:w` extraction and `pa` fall-through become the single test surface; the field-omission bug
  class disappears for JuMP results.
- The generic `factory` silently depends on the **fb-last field convention** holding for every
  optimisation result — a new result type that puts `fb` elsewhere, or carries more than `fb` as
  mutable-on-rebuild state, must override `factory`.
- **`NonJuMPOptimisationResult` is intentionally asymmetric** with the estimator hierarchy (no
  estimator twin); it exists purely as a dispatch marker.
- Naming proximity to watch: existing `JuMPResult <: AbstractJuMPResult` is the raw JuMP *model*
  solve stored in `sol`; new `JuMPOptimisationResult` is the optimisation *result core*. Doc both so
  they are not conflated.
- **Property introspection must become `:jr`-aware.** `propertynames(res)` for a JuMP result is now
  `(:jr, …, :fb)` — the forwarded names (`:pr`, `:pa`, `:fees`, `:retcode`, …) are reachable via
  `getproperty` but are no longer listed. Helpers that *introspect* a result (`hasproperty` /
  `:x in propertynames`) rather than just accessing it must check `:jr` for the JuMP case:
  `extract_pr` and `extract_fees` were updated to `hasproperty(res, :pr/:fees) || hasproperty(res, :jr)`.
  This surfaced (and fixed) a latent `extract_fees` bug — it had silently returned `nothing` for
  every JuMP result, because `:fees` was never in their flat `propertynames` even before the refactor.
- Verified: full suite green (4544 passing); the only failures pre-fix were 7 plotting errors, all
  routed through `extract_pr`, cleared by the introspection fix.

## Amendment (2026-08-16)

The value-level `expected_*` multiplicity work made a resolved risk measure **fitted state**, so a
result must be able to name the measure and the scalariser it optimised. Two branches of the tree
above changed as a consequence. The diagram in the Decision section is **wrong, not superseded** —
the embedded-core pattern and the `fb`-last rule both survive unchanged, and this amendment redraws
the tree rather than replacing the decision.

### The redrawn tree

```text
AbstractResult
├── BaseJuMPOptimisationResult                    (abstract — off AbstractResult, not in the result hierarchy)
│    └── JuMPOptimisationResult                   (core: pa, retcode, sol, model)
├── BaseHierarchicalOptimisationResult            (new abstract — off AbstractResult, not in the result hierarchy)
│    └── HierarchicalResult                       (now a core: pr, clr, wb, fees, retcode, w)
└── OptimisationResult
     ├── NonFiniteAllocationOptimisationResult
     │    ├── RiskJuMPOptimisationResult          (every subtype now carries a resolved r)
     │    │    ├── MeanRiskResult                    (jr, r, fb)
     │    │    ├── RiskBudgetingResult               (jr, r, prb, fb)
     │    │    ├── NearOptimalCenteringResult        (jr, r, w_min/opt/max/noc_retcode, fb)
     │    │    └── FactorRiskContributionResult      (jr, r, rr, frc_plr, fb)
     │    ├── NonRiskJuMPOptimisationResult       (new abstract — the JuMP results that carry no measure)
     │    │    └── RelaxedRiskBudgetingResult        (jr, prb, fb)
     │    └── NonJuMPOptimisationResult
     │         ├── NaiveOptimisationResult           (flat)
     │         ├── HierarchicalOptimisationResult (new abstract)
     │         │    ├── HierarchicalRiskParityResult              (hr, r, sca, fb)
     │         │    ├── HierarchicalEqualRiskContributionResult   (hr, ri, ro, scai, scao, fb)
     │         │    └── SchurComplementHierarchicalRiskParityResult (flat: pr, wb, clr, r, gamma, retcode, w, fb)
     │         ├── NestedClusteredResult             (flat)
     │         ├── StackingResult                    (flat)
     │         └── SubsetResamplingResult            (flat)
     └── FiniteAllocationOptimisationResult       (DiscreteAllocationResult, GreedyAllocationResult — unchanged)
```

One drift predates this amendment and is recorded here rather than silently corrected: the Decision
section calls `JuMPOptimisationResult` "the five always-present fields (`oe, pa, retcode, sol,
model`)". `oe` is no longer a field; the core carries four. The diagram above states the measured
tree.

### The hierarchical core

`HierarchicalResult` was a concrete leaf shared by `HierarchicalRiskParity` and
`HierarchicalEqualRiskContribution`. The two estimators carry a **different arity** of measures —
HRP holds one measure and one scalariser, HERC holds an intra-cluster pair and an inter-cluster
pair — so a shared leaf could only have grown `Option` slots or union-typed fields. It became a
**core** instead, on the same pattern the Decision section already fixed for `JuMPOptimisationResult`:

- `HierarchicalResult` keeps the six always-present fields (`pr, clr, wb, fees, retcode, w`) and is
  embedded as the **first** field, named `hr`, of each leaf.
- It sits on its own `BaseHierarchicalOptimisationResult` branch off `AbstractResult` and **does
  not** enter the result hierarchy, so the generic
  `factory(res::NonFiniteAllocationOptimisationResult, fb)` never reaches it.
- It **lost** its `fb`, exactly as the `fb`-last rule requires: `fb` belongs to the concrete result,
  not the core. Both leaves end in `fb`, so the one generic `factory` rebuilds them unchanged.
- Property forwarding is declared **per leaf** with `@forward_properties`, not on the abstract type,
  because the third family member has no `hr` — see below.

### The two new abstract types

- **`BaseHierarchicalOptimisationResult`** is the core branch, mirroring
  `BaseJuMPOptimisationResult`. Its one subtype is `HierarchicalResult`.
- **`HierarchicalOptimisationResult`** groups the results of the estimators that embed a
  `HierarchicalOptimiser`. The membership rule is exact and is the estimator's, not the field
  block's. The family is deliberately **not** named `ClusteringOptimisationResult`:
  `ClusteringOptimisationEstimator` has a fourth subtype, `NestedClustered`, which holds no
  `HierarchicalOptimiser`, so `NestedClusteredResult` stays flat under `NonJuMPOptimisationResult`.

Both are **unexported**, per the repository convention that abstract types stay unexported unless
asked for.

### Schur is a member by supertype only

`SchurComplementHierarchicalRiskParityResult` moved from `NonJuMPOptimisationResult` to
`HierarchicalOptimisationResult`, because its estimator embeds a `HierarchicalOptimiser`. It does
**not** embed `HierarchicalResult`: its field set genuinely differs — it carries `gamma`, and it has
no `fees` field. It gained a resolved `r` and **no** scalariser, because `SchurComplementParams.r`
is bounded `Sd_Var` and it holds one measure per bundle rather than a vector to combine. This
supertype-only membership is why the `hr` forwarding rule is declared on the two leaves rather than
on `HierarchicalOptimisationResult`.

### The other four `NonJuMPOptimisationResult` types stay flat

`NaiveOptimisationResult`, `NestedClusteredResult`, `StackingResult` and `SubsetResamplingResult`
are untouched. None embeds a `HierarchicalOptimiser`, none gained a core, and none gained an `r`.
The umbrella keeps the role the Decision section gave it: a dispatch marker carrying no shared
behaviour.

### The JuMP half split in two

`RiskJuMPOptimisationResult` was the only JuMP result branch. Making `r` fitted state made it
mandatory on that branch — but a `RelaxedRiskBudgeting` run builds its constraints straight from
`pr.sigma` and never resolves a measure. Rather than make `r` an `Option` slot on a shared type,
the branch split:

- **`RiskJuMPOptimisationResult`** now means "JuMP result that carries a risk measure". Every
  subtype carries a mandatory resolved `r` as its second field, straight after `jr`.
- **`NonRiskJuMPOptimisationResult`** is the sibling half, for JuMP results that carry none.
  `RelaxedRiskBudgetingResult` is its first member; it previously shared `RiskBudgetingResult`.
- **`RJR_NRJR`** is the union of the two halves. The default `getproperty` and `propertynames`
  moved onto the union, **not** onto either half. `MeanRiskResult` and `NearOptimalCenteringResult`
  declare no `@forward_properties` rule and depend on that default for `res.w`, so binding it to
  one half alone would silently cost the next measure-less leaf its property forwarding.

The scalariser rides the attributes rather than the result core on this branch:
`ProcessedJuMPOptimiserAttributes` gained `sca`, so a JuMP result reaches it as `res.sca` through
the existing `pa` fall-through and no JuMP result type needed a `sca` field.

### Consequences of this amendment

- The `fb`-last convention still holds for **every** optimisation result, so the generic `factory`
  is unchanged. Two new cores now sit off the result hierarchy instead of one.
- `propertynames` on a hierarchical result is now `(:hr, …, :fb)`, with the six core names reachable
  through `getproperty` but no longer listed — the same introspection caveat the Decision section
  recorded for `:jr`. `SchurComplementHierarchicalRiskParityResult` is unaffected; it is still flat.
- `HierarchicalResult` remains **exported** and remains constructible, but it is now a core rather
  than something `optimise` returns. The two names to reach for are `HierarchicalRiskParityResult`
  and `HierarchicalEqualRiskContributionResult`.
- Three abstract branches now exist where a caller might have written
  `res isa RiskJuMPOptimisationResult` to mean "any JuMP result". That predicate must become
  `res isa RJR_NRJR`.
