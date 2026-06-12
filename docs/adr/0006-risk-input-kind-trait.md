---
status: accepted — trait (`risk_input_kind`) over abstract subtypes
---

# Classify a risk measure's expected-risk input shape with a trait, not abstract subtypes

## Context

`expected_risk(r, w, X, fees)` computes the expected value of a risk measure for a
portfolio. *How* a measure is evaluated depends on what its functor consumes:

- **net returns** — `r(calc_net_returns(w, X, fees))` (most quantile/drawdown measures);
- **weights, returns, fees** — `r(w, X, fees)` (the moment families and tracking);
- **weights only** — `r(w)` (variance/standard-deviation family).

Until now this fact was recorded centrally in `19_RiskMeasures/26_ExpectedRisk.jl` as
four hand-curated `Union` aliases (`ERkNetRet`, `ERkwXFees`, `ERkw`, plus the unused
`ERkX`/`SlvRM`/`TnTrRM`), and `expected_risk` dispatched on them. Three problems:

1. **The fact lives far from the measure.** Whether `ConditionalValueatRisk` needs net
   returns is a fact *about CVaR*, but it was stated in a distant union. Adding a measure
   meant editing a file you might never open.
2. **Misclassification fails silently.** The net-returns and weights-only branches both
   call the functor with a *single vector* (`r(::VecNum)`). Putting a measure in the wrong
   one of those two unions passes the wrong vector and returns a wrong number with no
   error. (Outright omission already `MethodError`s — it is *mis*classification that is
   silent.)
3. **The interface enumerates every measure.** "Compute the risk" leaked the full census
   of measures into its dispatch surface.

This is the same shape of decision ADR 0001 faced for `factory`: a domain fact that
"cannot be recovered from types alone — it must be stated at the definition site."

## Decision

Replace the routing unions with a **trait** answered by each measure at its definition
site:

```julia
abstract type RiskInputKind end
struct NetReturnsInput        <: RiskInputKind end   # r(calc_net_returns(w, X, fees))
struct WeightsReturnsFeesInput <: RiskInputKind end  # r(w, X, fees)
struct WeightsInput           <: RiskInputKind end   # r(w)

risk_input_kind(r::AbstractBaseRiskMeasure) =        # erroring default
    throw(ArgumentError("`risk_input_kind` is not defined for `$(typeof(r))`. …"))

risk_input_kind(::ConditionalValueatRisk) = NetReturnsInput()   # at the definition site
```

`expected_risk` becomes a two-level Holy-trait dispatch — a generic entry that consults
the trait, and one leaf handler per kind:

```julia
expected_risk(r::AbstractBaseRiskMeasure, w::VecNum, args...; kwargs...) =
    expected_risk(risk_input_kind(r), r, w, args...; kwargs...)

expected_risk(::NetReturnsInput, r, w, X::MatNum, fees=nothing; kw...) = r(calc_net_returns(w, X, fees))
expected_risk(::WeightsReturnsFeesInput, r, w, X::MatNum, fees=nothing; kw...) = r(w, X, fees)
expected_risk(::WeightsInput, r, w, args...; kw...) = r(w)
```

**Scope.** Only the *leaf* input-shape routing moves to the trait. Composite measures
(`RkRatioRM`, `MeanReturnRiskRatio`) and the prior-extraction / vector-of-weights methods
keep their explicit `expected_risk` definitions — they orchestrate other measures rather
than declaring a leaf I/O shape, and dispatch specificity keeps them ahead of the generic
entry.

**Erroring default, not a fallback value.** `risk_input_kind` on the abstract supertype
throws, naming the type and the three valid returns. A measure that forgets to declare
fails loudly with an actionable message instead of silently routing to a wrong shape.

**Three dead unions deleted.** `ERkX`, `SlvRM`, `TnTrRM` were defined and documented but
never dispatched on anywhere (verified by grep + a green single-core
`test_09_risk_measures` run with them removed). They go too. `RkRatioRM` is kept (the
composite methods still use it).

**Not exported.** Like the sibling traits `needs_previous_weights` / `bigger_is_better`,
`risk_input_kind` and the kind singletons stay internal; custom-measure authors reach them
as `PortfolioOptimisers.risk_input_kind(::MyMeasure) = PortfolioOptimisers.NetReturnsInput()`,
the same qualified-access pattern already used for `<: PortfolioOptimisers.RiskMeasure`.

## Considered options

- **A. Keep the `Union` registry (status quo).** Rejected. It is the locality and
  silent-misclassification problem itself.
- **B. Abstract subtypes for the input kind** (`NetReturnsRiskMeasure`, … as supertypes
  each measure subtypes). **Rejected — collides with single inheritance.** The input-kind
  axis is *orthogonal* to the existing legal-usage axis (`RiskMeasure` /
  `HierarchicalRiskMeasure` / `NonOptimisationRiskMeasure`): net-returns measures span all
  three legal-usage types (`ConditionalValueatRisk` is a `RiskMeasure`,
  `RelativeMaximumDrawdown` is `Hierarchical`, `MeanReturn` is `NonOptimisation`), and
  weights-returns-fees measures span two (`Kurtosis`/`RiskMeasure`,
  `HighOrderMoment`/`Hierarchical`). A type gets one supertype chain, and that slot is
  already spent on the legal-usage axis, which the whole optimisation layer dispatches on.
  Encoding both as supertypes forces either a ~9-way cross-product of abstract types
  (`OptimisationNetReturnsRiskMeasure`, …) or demoting the load-bearing legal-usage axis to
  a trait — both strictly worse.
- **C. Trait (`risk_input_kind`).** **Adopted.** The idiomatic Julia answer for an
  orthogonal classification axis, and exactly what this hierarchy already does with
  `needs_previous_weights` and `bigger_is_better`. States the fact at the definition site
  (ADR 0001 lineage), turns silent misclassification into a loud error, and shrinks the
  dispatch surface.

## Consequences

- **The fact moves to the definition site.** ~51 one-line `risk_input_kind` declarations,
  each beside its `struct`, grouped with that type's other trait answers — matching the
  existing `bigger_is_better(::Skewness)` placement convention.
- **A bonus ambiguity hack disappears.** The old
  `expected_risk(r::ERkw, w, ::Pr_RR, args...)` method existed only to break a tie between
  the specific-on-`r` `ERkw` method and the generic prior-recursion method. The new entry
  is generic on `r`, so the prior-recursion method strictly dominates it — no tie. The
  workaround is deleted, not replaced.
- **The generic entry is broader than the old method set.** There was no
  `(AbstractBaseRiskMeasure, VecNum, X::MatNum)` catch-all before; such calls `MethodError`ed.
  After this, an undeclared measure reaching the entry throws the trait's `ArgumentError`
  (better message, same loudness). Explicit methods (composites, `ExpectedReturn`,
  prior-recursion, vector-of-weights) stay more specific and still win. Guarded by an Aqua
  `test_ambiguities` check.
- **Verification is differential + completeness.** An equivalence test asserts every member
  of the old routing unions resolves to the matching kind (so the new routing reproduces the
  old, and the three kinds are disjoint); a completeness test enumerates concrete
  `AbstractBaseRiskMeasure` subtypes and asserts each resolves a kind or is on a small
  explicit-handling allowlist — so a future measure added without a kind fails CI, not just
  at runtime.
- **Extends ADR 0001/0005.** Same "decide at the definition site" move `@prop` made for
  factory fields; sits alongside the typed/namespaced model-state interfaces.
