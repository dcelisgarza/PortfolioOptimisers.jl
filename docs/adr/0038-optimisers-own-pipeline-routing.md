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

**Rejected: recursion into the optimisers a meta-optimiser wraps.** Pushing targets down
into `opti`/`opto`/`opt`, so that an asset-level `WeightBounds` reaches each inner
`MeanRisk`, was considered and is not wanted.

A routing target names a destination, and inside a meta-optimiser the destination is
genuinely ambiguous rather than merely unimplemented — every inner optimiser, some of them,
the outer one as well, subset-viewed or not. Resolving that needs a language for saying
where each item goes, which is a larger and more speculative interface than the one this
ADR removes.

It would also buy less than it appears. The Pipeline exists to keep the test window out of
stateful preparation; a meta-optimiser's inner optimisers are invoked with the *fold's
training returns*, so whatever they compute internally is already leakage-safe. Recursion
would deduplicate computation, not fix correctness — and that is not worth a new interface.
Hyperparameter search reaches inner configuration through tuning lenses, which covers the
fixed-configuration case; the per-fold computed case is covered by the inner optimiser
computing its own.

The residual gap, recorded rather than fixed: an inner optimiser computes *a* prior, not
necessarily *the* prior the pipeline computed. `Stacking` uses `prior(st.pe, rd)` for the
outer combination but calls `optimise(opt, rd)` on each inner, so a prior that cannot be
reproduced from returns alone — entropy pooling with views, say — now reaches `Stacking.pe`
and still does not reach the inner optimisers. Should that matter, the fix is `Stacking`
threading its own prior down, which is an optimiser-internal question and does not involve
this seam.

**Deferred.** Extending the construction-time check to constraints would need a per-estimator
target trait, reintroducing a declaration that can drift from the results the estimator
actually produces.

## Amendment (2026-08-16): a vector of return terms is refused at this seam

`JuMPOptimiser.ret` now takes one return term or several (ADR 0052). The `:mu_ucs` route is
unchanged and still requires a single `ArithmeticReturn`: an uncertainty set is a neighbourhood
of the one quantity it was calibrated on (ADR 0050), so routing one set into *k* terms would
broadcast it across quantities it never bounded. The guard already refused anything that is not
an `ArithmeticReturn`; it now says which of the two failures happened, and points the caller at
the `ucs` field of the term the set belongs to.

## Amendment (2026-08-18): a constraint family declares its Routing Target

§7 above scoped the construction-time routability check to the `uncertainty` slot, and the
*Deferred* note gave the reason: "Extending the construction-time check to constraints would
need a per-estimator target trait, reintroducing a declaration that can drift from the
results the estimator actually produces."

That reasoning left a hole, and the 2026-08-17 architecture review found it. Two constraint
families were **steppable but unroutable**. `ThresholdEstimator` and `RiskBudgetEstimator`
each had a `run_step` that computed a result and wrote it to the `constraints` slot, and
neither result shape was a case in `constraint_targets`. `pipe_required_targets` returned `()`
for both, so `assert_routable` passed. A pipeline carrying one therefore fitted every earlier
step and then **always** threw at injection — under `cross_val_predict`, after the first fold
had already run. The gap ran the other way too: `CentralityConstraint` produces a
`LinearConstraint`, which routes, but had no `run_step` and was refused as a bare step.

### 1. Every constraint family that computes a value is a step, and declares where it lands

`pipe_constraint_targets(ce)` gives the routing targets a family's step can write. The
tuple's length is the contract: one target means the step needs no annotation, several mean
the step must name one through its `PipelineStep` wrapper, none means the family computes
nothing for the slot and is not a step.

```text
WeightBoundsEstimator                  → (:wb,)
LinearConstraintEstimator              → (:lcse,)
ExposureConstraintEstimator            → (:lcse,)
AbstractCentralityConstraint           → (:cte,)
AbstractPhylogenyConstraintEstimator   → (:ple,)
RiskBudgetEstimator                    → (:rkb,)
ThresholdEstimator                     → (:lt, :st, :slt, :sst, :sglt, :sgst)
AssetSetsMatrixEstimator               → (:smtx, :sgmtx)
JuMPConstraintEstimator                → ()
```

`PIPELINE_ROUTING_TARGETS` grows to seventeen accordingly. All but `:mu_ucs`, `:sigma_ucs`
and `:rkb` remain derived — they are `JuMPOptimiser` field names, so §3's rule places them
with no per-optimiser declaration.

### 2. The declaration cannot drift, because it is the only encoding

The deferred note feared a trait that disagrees with what the estimator produces. The fix is
that there is nothing left for it to disagree with. `run_constraint_step` resolves the target
from `pipe_constraint_targets` and pairs it with the computed value as a `TargetedConstraint`,
which `constraint_targets` reads straight off. The construction check reads the same
function. One declaration, three readers, no second encoding of the same fact.

Routing by result type survives only for values that arrive **without** a declaration — a
callable step writing `:constraints`, or a precomputed result. That path gained a `RiskBudget`
case (one field names it) and keeps refusing a bare `Threshold`, now with an error that says
which six fields it could mean and how to declare one.

### 3. `:rkb` is the one target that names a field an optimiser does not carry

A risk budget belongs to the risk-budgeting *algorithm*, at `rba.rkb`, so the derived
`hasfield` rule cannot reach it. `@pipe_route_rkb` declares it on `RiskBudgeting` and
`RelaxedRiskBudgeting`, in the same per-concrete-type shape and for the same reason as
`@pipe_route_sigma_ucs` (§4). Acceptance asks the algorithm the optimiser is actually
carrying, so an optimiser whose `rba` holds a `TimeDependent` schedule declines the target.

### 4. Refusing Threshold and RiskBudget at the fan-out was never the answer

The review's own reading was that refusing them is *correct*, because a `Threshold` names no
unique field. That is true of the **result** and false of the **step**. Six homes is a choice
the user makes, not an ambiguity the library must resolve, and `PipelineStep`'s `target` field
already existed to record exactly that kind of choice for uncertainty sets. `PIPELINE_STEP_TARGETS`
is extended with the eight declarable constraint targets, and the check that a declared target
belongs to the family runs at construction.

### 5. Two behaviour changes worth naming

**A second write to a single-valued target is now an error.** `constraint_targets` accumulates
into `PIPELINE_ACCUMULATING_TARGETS` and refuses a second write anywhere else, where it would
have silently replaced the first — `:wb` previously did exactly that.

Membership is decided by what generation does with several estimators, and checking that rather
than reasoning about the type aliases corrected two claims made earlier in this amendment's
drafting. The set is `(:lcse, :cte, :ple, :slt, :sst, :sglt, :sgst, :smtx, :sgmtx)`. `:lcse` and `:ple` hold order-free
blocks. The other six hold a *positional* list, entry `i` belonging to scenario or group block
`i` and paired with `scard`/`sgcarde` — so write order is block order, and a count that does
not match is **not** a silent mis-pairing: `JuMPOptimiser` validates those lengths against each
other when the routed value is absorbed, so it fails at injection with a `DimensionMismatch`.
That check is what makes packing them safe.

`:cte` accumulates too, but by **folding** rather than packing, and that distinction is the
rule rather than an exception to it. What a target accepts is read off what constraint
generation does when handed several estimators at once: for the packed targets that is one
result per estimator, and for `cte` it is one result *total* — `centrality_constraints` over a
vector of `CentralityConstraint`s appends every row of every estimator into a single
`LinearConstraint`. Verified: three estimators passed together give one constraint whose rows
are exactly the row-concatenation of the three computed separately.

So `accumulate_constraint_values` carries the shape. The default packs; `Val{:cte}` merges via
`merge_linear_constraints`. That is what makes *n* centrality steps in a pipeline agree with one
`cte` field holding *n* estimators — without it, the two ways of writing the same mandate would
build different models, which is precisely the failure this ADR's seam exists to prevent.

`Lc_CC_VecCC` is left alone. It never needed widening: the fold produces the single
`LinearConstraint` the field already accepts.

**Refusal moves earlier for the families that already routed.** `Pipeline(steps = (LinearConstraintEstimator(…), EqualWeighted()))`
was constructible and threw at injection; it is now refused at construction. The one case this
newly rejects rather than merely rejecting sooner is a step whose generation returns `nothing`
— a non-`strict` run whose names were all unknown — in front of an optimiser with no such
field. Refusing it is consistent with the declaration: the step says it writes `:lcse`, and
nothing there can receive it.

**Verified.** test_30 87/87, test_31 50/50, test_32 94/94, test_33 117/117, test_37 66/66,
test_38 180/180 — 594 passes over the pipeline, plus test_02 175/175 and test_18k 127/127 for
the constraint generation the fold reuses. 896 passes, 0 failures, all cold-loaded. Two censuses in test_38 keep the
declarations honest: every concrete constraint estimator outside the `JuMPConstraintEstimator`
branch declares a target rather than inheriting the empty fallback, and every member of
`PIPELINE_ACCUMULATING_TARGETS` is checked against the field that receives it, and the `:cte`
fold is checked against the equivalence it exists for — three centrality steps in a pipeline
against one `cte` field holding three estimators — so the set is asserted rather than
restated.
