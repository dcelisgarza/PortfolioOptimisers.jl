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
- **Full `BaseOptimisationResult` mirror layer.** **Rejected (YAGNI):** nothing else needs it yet, so
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
