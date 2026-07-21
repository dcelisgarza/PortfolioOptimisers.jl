---
status: accepted
amends: 0028 §3 (extends the prior route to every optimiser that has a `pe`; makes the construction-time claim true, narrowly), 0028 §4 (scopes it to slots, not the optimiser-facing seam)
---

# Optimisers own where a Pipeline slot lands; the Pipeline owns only the fan-out

## Context

[ADR 0028](0028-pipeline-workflow-estimator.md) §3 decided that computed slots override the
optimisation step's internal configuration, and sold a specific benefit: "a shared prior is
computed once per fold instead of once per consumer". The implementation put the whole
mapping in the Pipeline, as two hand-written methods over
`Union{JuMPOptimiser, HierarchicalOptimiser}` plus a `hasproperty(opt, :opt)` probe to find
the configuration.

Centralising the mapping had three costs, measured on `dev` before this change. Only the
first was the one the 2026-07-19 architecture review named.

- **The Pipeline's interface to an optimiser was that optimiser's private field layout.**
  `Accessors.PropertyLens{:pe}`, `{:wb}`, `{:lcse}`, `{:ple}`, `{:cle}` were spelled in
  `23_Pipeline/03_Pipeline.jl`, so a field rename broke the Pipeline at run time, far from
  the rename.

- **The table stopped at the types its author enumerated, so §3 was under-delivered.**
  `Stacking`, `NestedClustered` and `SubsetResampling` each carry their own `pe`, and
  `NestedClustered` a `cle`; all three, plus all three naive optimisers, carry an
  asset-dimensioned `wb`. None of them has an `:opt` *configuration* field, so every one
  fell through to the fail-closed tail: the computed prior was silently dropped and
  recomputed — precisely the "once per consumer" cost §3 promised to remove, in exactly the
  case with the most consumers — while a computed `WeightBounds` threw, though `wb` was
  sitting right there. A `WeightBoundsEstimator → EqualWeighted` pipeline could not be
  built at all.

- **The probe was structurally unsound.** `SubsetResampling` has a field literally named
  `opt` holding an inner *estimator*, not a configuration. `hasproperty(opt, :opt)` walked
  straight into that collision and was saved only by the `isa` check behind it.

One claim in §3 was simply not implemented: "Injection is type-checked at pipeline
construction where possible." The constructor validated slot ordering, availability and
invalidation, and nothing about routability, so an unroutable uncertainty set surfaced at
injection — under `cross_val_predict`, after the first fold had already fitted every
earlier step.

## Decision

### 1. The seam inverts: `pipe_route(x, Val(target), v)`

The optimiser decides which of its fields a target lands in. The Pipeline calls blind. The
two halves live where their knowledge lives:
[`20_Optimisation/01_Base_Optimisation.jl`](../../src/20_Optimisation/01_Base_Optimisation.jl)
declares the generics; each optimiser file declares its own behaviour.

`inject_config` and `inject_sigma_ucs` are **deleted**. `inject_context` survives as the
Pipeline-side driver.

Nothing on the optimiser side names a Pipeline type. That is load-bearing, not incidental:
`23_Pipeline/` is included *after* all of `20_Optimisation/`, so the review's sketched
`apply_context_slots(opt, ctx::PipelineContext)` could not have been written in
`10_JuMPOptimiser.jl` at all — it would have forced either the methods back into the
Pipeline (no locality gained) or `PipelineContext` out of it.

### 2. Routing Targets are finer than slots, and are named after fields

`PIPELINE_ROUTING_TARGETS = (:pe, :cle, :wb, :lcse, :ple, :mu_ucs, :sigma_ucs)`.

The Pipeline owns slot → target fan-out, which is where slot-level heterogeneity is
resolved: splitting the `PipelineUncertaintySets` pair, grouping `constraints` elements by
result type, packing one-or-many. Optimisers own target → field.

Five targets are named after the field they land in. Naming them for the *domain* instead
was considered and rejected: `pe`, `cle`, `wb`, `lcse`, `ple` are entries in the shared
`field_dict` in [`01_Base.jl`](../../src/01_Base.jl), reused across every optimiser's
docstrings. They are already package-wide vocabulary, so "field name" and "domain name" are
the same string, and inventing a second set of names would add a synonym without adding
information.

### 3. Because targets are field names, the default method *is* the routing rule

`pipe_route`'s fallback sets the like-named field of whichever optimiser has one, and
`pipe_accepts` is exactly `hasfield`. There are **no per-optimiser route declarations**, and
therefore nothing that can drift out of sync with the fields — the anti-drift property is
structural rather than enforced.

The lookup is `hasfield`, not `hasproperty`: routing rebuilds the object through the field,
so a name reachable only as a forwarded property could be read but not set.

This reproduces the pre-inversion behaviour for the two configurations *derivatively* rather
than by restating it: `JuMPOptimiser` has no `cle` field, so phylogeny remains ignored;
`HierarchicalOptimiser` has no `lcse`/`ple`, so those still throw.

### 4. Two exceptions carry policy, and are declared per concrete type

`:mu_ucs` requires an `ArithmeticReturn` and lands in `ret.ucs`; `:sigma_ucs` lands in the
`UncertaintySetVariance` measures of `r`. Neither names a plain field.

`:sigma_ucs` is declared per **concrete** type by `@pipe_route_sigma_ucs`, not on a
supertype, for two reasons. It must out-specialise the delegating forwarder on the same
type; and carrying a configuration does not imply carrying risk measures —
`RelaxedRiskBudgeting` has an `opt` but no `r`, and correctly does not accept the target.

### 5. Delegation is declared, never probed

`@pipe_delegates T opt` emits `pipe_config_field` plus forwarding `pipe_route`/`pipe_accepts`.
Declaring rather than probing is what keeps `SubsetResampling`'s `opt` field — an inner
estimator — from being mistaken for a configuration.

Both macros escape their `quote` block **whole**. Unescaped, hygiene renames the generics
being extended into gensyms, so every method lands on a throwaway function while the package
still loads clean; the only symptom is `pipe_accepts` returning `false` everywhere.

### 6. Ignore-versus-fail-closed is a property of the target, declared once

`PIPELINE_OPTIONAL_TARGETS = (:pe, :cle)`. `unroutable_target` ignores those and throws for
the rest, replacing four separate `@argcheck`s that each restated the policy.

The asymmetry turns on whether dropping the value changes the *answer*. A prior is
recomputable — an optimiser with no `pe` either needs none or computes an equivalent one,
which is what §3 means by every stage being optional. A `JuMPOptimiser` has no `cle` because
phylogeny reaches it as constraint results generated from returns, so the structure is
surplus to it. A weight bound, linear constraint, phylogeny constraint or uncertainty set
that reaches no field would silently change the solved portfolio.

The cost this accepts should be stated, because it was mis-stated during design: **no step
reads the `prior` or `phylogeny` slots** — `inject_context` is their only consumer — so a
step writing a slot the terminal optimiser cannot receive is silently wasted computation.
That is a performance trap, not a correctness one, which is why it is tolerated; it is also
the reason the optional list stays short.

`:cle` and `:ple` come from *different slots* despite both concerning phylogeny, and are one
letter apart: `:cle` is a clustering structure from the `phylogeny` slot, `:ple` a phylogeny
*constraint result* from the `constraints` slot. They can never both land, because no
optimiser has both fields — `cle` exists only on `HierarchicalOptimiser` and
`NestedClustered`, `ple` only on `JuMPOptimiser`. A pipeline writing both therefore has one
ignored (`:cle`, into a JuMP-based optimiser) or errors (`:ple`, into a hierarchical one),
which is unchanged from before the inversion.

### 7. Routability is checked at construction — for uncertainty only

`assert_routable` asks `pipe_accepts` directly, so it stays honest as optimisers gain or
lose fields.

It covers the `uncertainty` slot and nothing else, and this is a real limit rather than an
implementation gap. An uncertainty step declares which parameters it bounds through its
`PipelineStep` wrapper, and that declaration is a *field of the step*. The other fail-closed
targets are chosen by the *result type* a constraint estimator produces at fit time, and
every constraint estimator writes the same `constraints` slot, so which of `:wb`, `:lcse`,
`:ple` a step will write cannot be read off its type.

The check is structural: it establishes the optimiser *family* can receive the target, not
that a particular configuration will accept the value. A `JuMPOptimiser` carrying a
non-`ArithmeticReturn` still constructs and still fails at injection — that condition
belongs to `pipe_route` and is not duplicated. It is skipped when the terminal step is a
`TimeDependent` schedule or a precomputed result, since the optimiser is not known until the
fold loop resolves it.

### 8. The seam stays internal

Unexported, matching `pipe_reads`/`pipe_writes`. Exporting it was considered and rejected:
ADR 0028 §4 rejected one-slot-per-optimiser-field as "a maintenance mirror of
`JuMPOptimiser` that turns the domain vocabulary into plumbing", and a published,
per-field Routing Target vocabulary is close enough to that to deserve the same answer.
§4 governs `PIPELINE_SLOTS` — pipeline-author vocabulary, unchanged and still coarse — and
targets address optimiser authors instead; but the vocabulary is young, and the deferred
work below will likely change it.

## Consequences

**Behaviour changes.** A computed prior now reaches `Stacking`/`NestedClustered`/
`SubsetResampling`/`InverseVolatility` instead of being recomputed; a computed
`WeightBounds` now reaches every naive and meta optimiser instead of throwing;
`NestedClustered` receives a computed clustering structure. One `test_32` assertion flipped
from throw to route — it had encoded an implementation limit as a rule.

**A dependency worth naming.** Routing a `PriorResult` into a meta-optimiser's `pe` is legal
only because the guards rejecting one live in `assert_external_optimiser`, which fires when
a meta-optimiser is *nested inside another*. A Pipeline's optimisation step is terminal and
therefore never external. This holds **because** ADR 0028 §10 forbids wrapping a Pipeline in
a meta-optimiser; if that is relaxed, this route must be re-examined.

**What is not claimed.** With targets named after fields, a field rename still touches the
seam. It touches only that optimiser's own declaration — or, for the five derived targets,
nothing at all, since the rule reads the field list directly. The *remote* breakage was the
defect; rename-immunity was never on offer.

**Verified.** test_26 2/2, test_28 14/14, test_30 71/71, test_31 50/50, test_32 74/74,
test_33 117/117, test_34 51/51, test_37 523/523, and a new solver-free
`test_38_pipeline_routing.jl` 76/76 whose expected-routing table locks the derived
consequences of the field layouts in both directions.

**Deferred.** Recursion into the inner optimisers a meta-optimiser wraps (`opti`/`opto`/
`opt`) — the genuinely useful case, where an asset-level `WeightBounds` reaches each inner
`MeanRisk` — is not attempted here; it interacts with the `port_opt_view` subsetting
`SubsetResampling` applies per subset, and with `NestedClustered`'s cluster-space `opto`.
Extending the construction-time check to constraints would need a per-estimator target
trait, reintroducing a declaration that can drift.
