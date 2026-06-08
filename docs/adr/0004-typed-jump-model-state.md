---
status: accepted
---

# Give the JuMP model a typed state interface

## Context

JuMP-based optimisers build one `JuMP.Model` incrementally: dozens of constraint
and risk builders across [09_JuMPConstraints](../../src/20_Optimisation/09_JuMPConstraints/)
and [19_RiskMeasureConstraints](../../src/20_Optimisation/19_RiskMeasureConstraints/)
read and write shared entries by raw symbol key — `model[:sc]`, `model[:w]`,
`model[:k]`, … — with no accessor layer. There are ~650 `model[:sym]` references
over ~120 distinct keys, plus 126 `haskey(model, :sym)` guards and 112 raw
`model[:sym] = …` writes. The model *is* the seam between builders, but it has no
interface: existence and population-order are invariants every caller must just
know.

Inspecting the keys shows they are **two different things**, not one:

- **Category A — shared singleton infrastructure** (~25 keys): `:sc`, `:so`, `:w`,
  `:k`, `:X`, `:net_X`, `:Xap1`, `:ddap1`, `:G`, `:dd`, `:W`/`:M`/`:M_PSD`,
  `:Au`/`:Al`, `:E`/`:WpE`, `:fees`, `:variance_flag`, `:rc_variance`, `:risk_vec`,
  `:risk`, … Created once (guarded by `haskey`), written by one builder, read by
  many. `:sc` alone has 129 references. **This is the actual blackboard.**
- **Category B — dynamically-indexed scratch keys** (~100, 239 registrations):
  `Symbol(:variance_risk_, i)`, `Symbol(:sd_risk_, i)`, `Symbol(:sigma_W_, i)`, …
  Produced and consumed inside a single `set_risk!` call for risk-measure instance
  `i`, usually returned as an explicit `(expr, key)` tuple. Per-instance scratch,
  already locally scoped and self-documenting.

The cost of the missing interface is concrete, not cosmetic. The clearest example is
[18_TrackingRiskMeasureConstraints.jl](../../src/20_Optimisation/19_RiskMeasureConstraints/18_TrackingRiskMeasureConstraints.jl):
`RiskTrackingRiskMeasure` applies a risk measure to the portfolio-vs-benchmark
difference, which requires rebuilding the risk expressions against a different
returns vector. It does this with ~150 lines that manually rename every live
Category-A key to `:oldX`, `:oldW`, `:oldM`, `:oldAu`, `:oldE`, `:oldvariance_flag`,
… build fresh entries, then restore — key by key, in two places. Adding a new
Category-A key silently corrupts this measure unless the author remembers to extend
the swap block. This is the "every future risk measure must know which keys exist"
hazard made real.

A second, quieter hazard: `set_net_portfolio_returns!` treats `net_X = X` with **no
error** when `:fees` is absent, so a fees builder running *after* a net-returns
consumer silently drops fees. Today this is held together only by call order in the
orchestrator ([11_MeanRisk.jl](../../src/20_Optimisation/11_MeanRisk.jl), fees at
line 649 before risk at 650), not by anything structural.

## Decision

Give Category A — and only Category A — a named, checked accessor interface, and
route all access through it. Category B keeps its existing `(expr, key)` convention.

### 1. Scope: Category A only

The leak the review worries about is the shared blackboard. Category B is per-instance
scratch that already works; forcing it behind `i`-parameterised accessors would be a
large surface for no payoff, or would collapse 24 measures' private scratch into one
namespace (worse). Out of scope.

### 2. Mechanism: accessor functions over the raw `JuMP.Model`, not a wrapper type

The model stays a plain `JuMP.Model` passed everywhere. Reads/writes funnel through
named functions (`get_constraint_scale(model)`, `get_w(model)`, `has_*` predicates,
the existing `set_*!` writers). A wrapper struct was rejected: every
`JuMP.@variable`/`@expression`/`@constraint` needs a real `JuMP.Model`, so a wrapper
would be unwrapped and rewrapped at ~650 sites for negative ergonomics. Const-symbol
renaming buys nothing.

The trade-off: this is a **discipline seam, not a compiler-enforced one**.
`PortfolioOptimisers` is a single flat module with no nested namespaces, so nothing
*prevents* a future caller from writing `model[:sc]` raw. Enforcement is a test
(§6), not the type system. Accepted as the cost of not fighting JuMP.

### 3. Home: `08_Base_JuMPOptimisation.jl`

The interface lives in [08_Base_JuMPOptimisation.jl](../../src/20_Optimisation/08_Base_JuMPOptimisation.jl),
which already holds the embryonic setters (`set_w!`, `set_model_scales!`,
`set_portfolio_returns!`) and is included before both constraint directories, so
include-order — the only hard constraint in a flat module — is satisfied.

### 4. Guarantee: existence now, ordering later

v1 enforces **existence**: assert-on-read accessors (`@argcheck(haskey(model, …))`)
turn an absent/typo'd key into a clear error at the read site instead of an opaque
`KeyError`. The harder **true-ordering** hazard (fees-before-net-returns) is *not*
silently folded in: it is handled as a separate fast-follow (a "fees sealed" marker
so `set_net_portfolio_returns!` asserts finalisation and `add_to_fees!` errors if
called too late). This ADR does **not** claim the interface "enforces ordering"
broadly — it enforces existence and names one ordering hazard.

### 5. Snapshot/restore is part of the interface

Category-A keys are modelled as a **registered, enumerable set** with companion
links (`:W` travels with `:M`, `:M_PSD`), and the interface exposes a selective
scoped swap (`with_swapped_state` / `snapshot_state!` + `restore_state!`). The
150-line hand-rolled block in `RiskTrackingRiskMeasure` collapses to one scoped
call, and new keys auto-participate. This is the highest-leverage part of the work,
above the read accessors.

### 6. Migration: incremental, easy-first, behaviour-locked

The behaviour oracle is the existing characterisation suite (MeanRisk
[test_18a–18l](../../test/), formulations, tracking) plus kaimon-driven checks.
Order:

1. Land the interface additively (raw access still works) — non-breaking.
2. Sweep the easy reads (`:sc`, `:w`, `:so`, `:k`, …) file-by-file, tests green
   between each.
3. Sweep the `haskey` guards to `has_*` predicates.
4. **Last:** migrate the `RiskTrackingRiskMeasure` snapshot/restore — the one
   genuinely tricky change — in isolation, so a bug there is not buried among
   hundreds of mechanical diffs.
5. Lock the seam: a test that fails on raw `model[:` outside
   `08_Base_JuMPOptimisation.jl`, so the blackboard cannot silently re-leak.

(The review suggested doing the snapshot first to validate the registry early; we
deliberately chose easy-first for smaller, individually-safe steps on a dangerous
change.)

## Consequences

- The model-state vocabulary lives in one place; absent/typo'd-key bugs concentrate
  at named read sites.
- `RiskTrackingRiskMeasure`'s key-by-key save/restore is replaced by a registry the
  new keys join automatically — the main correctness win.
- The seam is convention-enforced, not type-enforced; the §6 test is what keeps it
  honest. If it is removed, the complexity reappears across ~650 sites (deletion
  test passes).
- "Model State" is added to the glossary as a one-line concept; it is implementation,
  so it stays out of the domain detail in CONTEXT.md.
