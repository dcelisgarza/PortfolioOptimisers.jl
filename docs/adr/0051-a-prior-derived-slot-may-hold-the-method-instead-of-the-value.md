---
status: accepted
---

# A prior-derived slot may hold the method instead of the value

## Context

A risk measure that carries a prior-derived field took a **value** or `nothing`.
[`Variance`](../../src/19_RiskMeasures/02_Variance.jl) took a covariance matrix,
[`Kurtosis`](../../src/19_RiskMeasures/04_Kurtosis.jl) a cokurtosis tensor,
[`ArithmeticReturn`](../../src/20_Optimisation/09_JuMPConstraints/02_Returns_and_ObjectiveFunctions.jl)
an expected-returns vector. `nothing` meant "take the optimisation's own prior"; a value meant
"use this one".

Those are the only two settings, and neither survives a refit. Cross-Validation refits the prior
per fold. A meta-optimiser — Subset Resampling, Nested Clustered — refits it per subproblem. And
`port_opt_view` runs **before** `factory` (`19_SubsetResampling.jl:536` takes the view,
`:516` computes the prior), so a stated matrix crosses that boundary as the whole universe's
answer while every other input is re-derived on the subset. The caller who wants
`LedoitWolf` covariance in the measure, rather than the optimiser's default, has to paste in a
matrix, and the paste is then pinned for the rest of the run.

There is a third setting the type system already permitted and nothing used: **the method**. An
Estimator is a description of how to compute for whatever input it is given, so an Estimator in
the slot survives a refit precisely because it holds no numbers.

The library already had two precedents.
[`MedianAbsoluteDeviation`](../../src/19_RiskMeasures/24_MedianAbsoluteDeviationRisk.jl)'s `mu`
admitted a value or a centring strategy, resolved by `calc_moment_target` at the point of use.
ADR 0048's `HopCount.n` and `PathLength.dmax` admit an integer or a **rule** called at the point
of use, with refusal methods in the kernel.

## Decision

**A prior-derived slot admits the value or the Estimator that computes it.** The domain noun is a
**Deferred Quantity** (`CONTEXT.md` §1). Four quantities defer — `mu`, `sigma`, `kt`, `sk` — and
each has a field-bound alias: `MuSlot`, `SigmaSlot`, `KtSlot`, `SkSlot`. `DeferredQuantity` names
the dynamic half alone: the four moment-estimator families, plus `AbstractPriorEstimator`, which
computes every quantity at once.

Fourteen risk-measure types and `ArithmeticReturn` take one.

### The count is of *deferrable* slots, not of prior-derived fields

**A measure with exactly one deferrable slot widens that slot.** A measure with two or more takes
a `pe` slot instead, which fans one fit out over every quantity the measure leaves unstated.

A *derived* slot does not count: it travels with its source rather than being fanned out
separately. `chol` travels with `sigma`, and `V` travels with `sk`. So `Variance`,
`StandardDeviation` and `NegativeSkewness` widen their source slot and take no `pe` —
`Variance(pe = P)` would say exactly what `Variance(sigma = P)` already says, and with `sigma`
stated it could not fill `chol` at all. Four types take a `pe`: `DistributionValueatRisk`,
`Kurtosis`, `Skewness` and `VarianceSkewKurtosis`.

`pe` is bounded `Option{<:AbstractPriorEstimator}` — **narrower than every other `pe` in the
library**, which is `PrE_Pr` and admits a precomputed Prior Result. A risk measure is an
Estimator, and an Estimator never holds a Result.

### Precedence: a stated field wins, `pe` fills the rest, nothing refuses

A Deferred Quantity on a slot beats the container's `pe`, one level down as well: a deferred slot
on a child of `VarianceSkewKurtosis` resolves and wins over the container's fan-out. `pe` then
fills whatever is still unstated, and the prior supplies what neither names.

**Nothing refuses a mixed configuration.** A caller may state `mu` by hand and defer `kt`, and the
two need not describe the same distribution. The design chose **flexibility plus a warning** over
refusal, because there is no way to tell a deliberate mix from an accident: a caller who states a
robust `mu` beside a sample `kt` is doing something legitimate, and a rule that refused it would
refuse the whole point of a per-slot bound. Every widened slot therefore carries the same
admonition: a caller who wants one consistent set names the prior-recipe slot alone and lets it
fill everything, and a caller who states fields by hand must make sure that they agree.

### The one refusal: a derived slot stated without its source

`chol` is a factorisation of `sigma`, and `V` is built from `sk`. Neither ever defers, because
neither is separately fittable — each arrives as one pair with whatever its source resolves to.

Stating the derived slot **without** its source is an `ArgumentError` at construction
(`assert_derived_slot_has_source`). Otherwise the caller's factor would pair with a covariance
matrix the caller never saw, because the prior supplies it, and the model would optimise one
quantity while the functor evaluates another. `NegativeSkewness` expresses the same rule with its
own shape: `V` is both-or-neither with a matrix `sk`, and forbidden beside a deferred one.

This is API-breaking. No caller in the repo did it.

**A stated `chol` is never rebuilt from `sigma`.** Under a factor prior the factorisation is
sparse and special, and a rebuild would throw that structure away.

### Three resolution points, not one

The charting assumed `factory(rm, pr, …)` was the single seam. It is not.

| Point | Why it exists |
| ----- | ------------- |
| `factory(x, pr, …)` | the frontier rebuild and `NearOptimalCentering` call it; so does the returns estimator's route |
| `set_risk_constraints!` | the `JuMP` model builders **never** call `factory` on a risk measure — each applies its own prior fallback |
| `resolve_risk_inputs` (the value-level) | `expected_risk(r, w, pr)` resolves; `expected_risk(r, w, X)` has no prior and refuses |

Each point resolves **once**, not once per evaluation. Inside `risk_contribution`'s `2N` loop a
per-evaluation resolution would refit a deferred covariance `2N` times.

The per-type entry point is `resolve_deferred_quantities(x, pr)`, which resolves the deferred
state and **nothing else**. The prior fallback for an unstated slot stays where it already was.

### Which slots defer is a per-type declaration, never a field walk

`deferred_slots(x)` returns the slots a type defers, as a `NamedTuple`, and
`assert_resolved_slots` recurses through it to refuse at the value-level seam.

It cannot be derived by walking fields and testing `isa(field, DeferredQuantity)`. The
falsification witness: `SimpleVariance isa DeferredQuantity`, and it stands legitimately in
`Skewness.ve`, `LowOrderMoment.alg.ve` and `UncertaintySetVariance.ucs` — slots that hold an
Estimator **by design**. A walk would refuse three correct configurations. Every type that
declares `resolve_deferred_quantities` declares `deferred_slots` beside it.

### The data the fit runs on

A Deferred Quantity fits on `pr.original_X`, with `pr.w` threaded so that `_wprop` replaces the
estimator's own observation weights. `original_X` and not `X`: under a factor prior `X` is the
reconstruction `F * M' .+ b'`, which has rank `size(F, 2)` and carries no residual, so fitting a
covariance estimator there returns a **singular** matrix whenever there are more assets than
factors. See the `o_X` amendment to ADR 0046.

**`F` reaches a slot only through a prior estimator.** No moment estimator takes factor returns;
`prior(pe, X, F)` and `regression(re, X, F)` are the only interfaces that do. A slot that must see
factors has to hold an `AbstractPriorEstimator`.

### A higher moment is a moment about a centre

A co-moment estimator supplies the tensor **and** the centre it was taken about, out of one
object: an unstated `mu` beside a deferred `kt` or `sk` is read off that estimator's own `me` and
threaded in as `mean =`. A stated `mu` wins and is threaded through instead.

A prior estimator centres itself and cannot take a centre. `fit_deferred_moment` says so by
dispatch rather than swallowing the keyword.

The same rule governs the `sk`/`V` pair's matrix processor. `HighOrderPrior.skmp` was the
precedent and is now the rule: whatever fit produced `V` also names the `mp` that built it, and
the consumer records **that** `mp` rather than its own, so a later windowed rebuild stays in step.

### The high-order gate asks for the quantity, not the prior type

`assert_high_order_quantity` passes when **either** the measure or the prior carries the tensor.
The two `pr::LowOrderPrior` refusal methods are deleted. `S2` and `L2` are rebuilt from `N` by
`dup_elim_sum_selector`, and the rebuild is `==` the prior's rather than an approximation, because
`dup_elim_sum_matrices` is a pure function of `n`. `NegativeSkewness` gates on `sk` and not on the
`V` its kernel reads, because the pair is both-or-neither and `sk` is the slot a caller can state.

A high-order measure that holds its own tensor now solves under any prior, and gives the same
weights it gave under a `HighOrderPrior`.

## Rejected alternatives

**Refuse a mixed configuration.** Rejected above: a legitimate mix and an accidental one are
indistinguishable, and refusing both removes the reason to have a per-slot bound at all. The
warning carries the cost instead.

**Recover deferral from the type.** Rejected on the `SimpleVariance` witness — a Deferred Quantity
is not recoverable from the type, so the declaration must be per type.

**Resolve inside the kernel, at the point of use.** This is the shape ADR 0048 uses for
`HopCount.n`. It does not fit here: a kernel sees the returns matrix but not `pr.w`, not the
factor returns, and not the prior's other fields, so it would resolve under a different rule than
the settled one. Resolution belongs to the consumer, and the kernel refuses — which is ADR 0048's
shape one level up.

**Take a Prior Result in `pe`.** Rejected by the standing rule that an Estimator never holds a
Result. It is also self-defeating: a Result is exactly the pinned thing the feature exists to
avoid.

## Consequences

- **Three API breaks.** `Variance`, `StandardDeviation` and `DistributionValueatRisk` refuse a
  `chol` stated without a `sigma`. `NegativeSkewness` refuses a `V` beside a deferred `sk`.
- **`chol` is now selected as a pair with `sigma`**, rather than field by field, which changes one
  existing case: a stated `sigma` with no factor no longer picks up the prior's factorisation.
- **`expected_risk(Variance(), w, pr)` stops being a `MethodError`.** The value-level seam runs
  `factory`, which fills an unstated slot as well as resolving a deferred one. With fees a bare
  `LowOrderMoment()` changes number, and now agrees with the optimiser.
- **`pe` is appended last on every struct that gains it.** `Kurtosis` alone has 28 partial
  parameterisations, so any other position would break them all.
- **Four latent defects were fixed in passing**, all of the same family — a slot resolved on one
  path and not on another. `Skewness.port_opt_view` dropped `settings`, silently re-enabling the
  risk expression of a `VarianceSkewKurtosis` child. `VarianceSkewKurtosis` had no prior-`factory`
  method at all, because `@propagatable` emits one only for a `@pprop`/`@cprop` field. A prior
  fitted with `kte = nothing` made a measure that stated its own `kt` multiply by `nothing`. And
  `nothing_scalar_array_view`'s identity union covered `<:AbstractEstimator` but not
  `StatsBase.CovarianceEstimator`, which is a `MethodError` on the first covariance estimator to
  cross a view.

## Amendment (2026-08-12) — from [#287](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/287)

Final verification read the shipped API against the map. Two records above are wrong, and one of
them was a real split in the behaviour.

### The derived-slot rule is one rule, and it now refuses two states

The decision above described a single refusal — a derived slot stated with no source. The shipped
code had a second, on `NegativeSkewness` alone: a `V` stated beside a **deferred** `sk`. The
`sigma`/`chol` carriers took the opposite route in that state and **discarded** the stated `chol`
silently.

Both states are the same mistake, so both are now refused, and by one helper.
`assert_derived_slot_has_source` takes the deferred case as well:

| Derived slot | Source slot | Result |
| :-- | :-- | :-- |
| stated | stated value | both kept. The derived value is never rebuilt |
| stated | **Deferred Quantity** | **`ArgumentError` at construction** |
| stated | not given | **`ArgumentError` at construction** |
| not given | stated value | the kernel derives the derived value |
| not given | Deferred Quantity | the fit supplies the pair |
| not given | not given | the prior supplies the pair |

A silent discard is the shape this ADR names as the thing to avoid: the caller states a factor,
the library keeps a different one, and nothing says so. Refusing costs the caller one keyword and
tells them the rule once.

**Neither refusal breaks an existing caller.** A source slot could not hold an Estimator before
this ADR, so `Variance(sigma = <estimator>, chol = C)` was a `MethodError` and not a working
configuration. The break list above is therefore **one** shape, not three: a derived slot stated
with no source at all.

### `pe` is not last on `DistributionValueatRisk`

`dist` follows it, so the positional constructor changed from
`DistributionValueatRisk(mu, sigma, chol, dist)` to
`DistributionValueatRisk(mu, sigma, chol, pe, dist)`. The claim above holds for `Kurtosis`,
`Skewness` and `VarianceSkewKurtosis` and not for this one. It breaks loudly rather than silently
— a `Distribution` is not an `Option{<:AbstractPriorEstimator}`, so the old call is a
`MethodError` — and no caller in the repo uses the positional form. It is still an API break, and
it belongs on the list.

### What verification confirmed

- The whole export set is unchanged across the map: 752 names before and after. No abstract type
  was exported.
- A Deferred Quantity refits per cross-validation fold. Under a three-fold `KFold` with a
  denoising covariance estimator, a pasted matrix gives the **same** weights in every fold
  (spread `0.0`) and the deferred estimator gives different ones (spread `3.1e-2`); the two
  differ by up to `5.1e-2`. The subset half of the argument is asserted in
  `test_09e_deferred_quantity.jl`.
- `ArithmeticReturn` declares `resolve_deferred_quantities` and **no** `deferred_slots`. The rule
  beside `deferred_slots` says to declare both. It is unreachable today, because `expected_return`
  has no bare-returns-matrix arm for `assert_resolved_slots` to guard, so this is a latent trap
  rather than a live defect. Adding such an arm must add the declaration with it.

## Amendment (2026-08-17) — from candidate B of the architecture review of 2026-08-16

The decision above pairs two declarations per type: `deferred_slots` names the slots, and
`resolve_deferred_quantities` resolves them. The review read the shipped code and found that the
pair is **not symmetric**. The naming half is one line per type. The resolving half is one line
per type for a leaf and a *forwarding method* for a container — and a container gets that
forwarding for free on the `factory` path, from `@fprop`, while on the `JuMP` path it has to be
written by hand. Four containers had not written it.

### Container recursion is derived, never written

**`resolve_deferred_quantities` now has a derived method that reads `deferred_slots`.** It
resolves each declared slot, rebuilds the type from its own field list when a slot moved, and
returns the argument itself when none did. `resolve_deferred_child` carries the per-slot rule: a
child resolves through its own method, a vector of children resolves element by element, and
anything else is returned unchanged.

**A type that resolves a quantity of its own still declares a method**, which is more specific
than the derived one and wins. That half cannot be derived, and the reason is the rule this ADR
already states: slots that travel together must be resolved together. A deferred `sigma` supplies
`chol` from the same fit, a deferred `sk` supplies `V` and the `mp` that built it, and a `pe`
fans one fit out over every slot the caller left unstated. No derivation can know any of that.

So the interface is now one declaration per container and two per leaf, and the JuMP path and the
factory path read the same declaration.

### The gap was six containers, not two

The review named `RiskTrackingRiskMeasure` and `GenericValueatRiskRange`. Reading every risk
measure and returns estimator that holds a child found four more: `RiskRatio`,
`NonOptimisationRiskRatio`, `ExpectedReturn` and `ExpectedReturnRiskRatio`. All six declare
`deferred_slots` now and need no forwarding method. `ValueatRisk` and `ValueatRiskRange` keep
their declarations and **lose** their hand-written forwarders.

`MeanReturnRiskRatio` declares its risk axis alone. Its `rt` is bounded to `MeanReturn`, which
defers nothing, so declaring it would name a slot that can never hold a Deferred Quantity.

**`ArithmeticReturn` now declares `deferred_slots`**, which the 2026-08-12 amendment recorded as a
latent trap. It is no longer latent: `ExpectedReturn.rt` and `ExpectedReturnRiskRatio.rt` reach it,
so the value-level check now refuses a deferred `mu` one level down instead of passing it on.

### A declaration without a resolver is refused at the call

`assert_declared_slot_resolver` runs inside the derived method. A **Deferred Quantity** that
survives the recursion means the type declared the slot and wrote no resolver, so the estimator
would have reached a model builder and been multiplied as though it were a matrix. It now raises
an `ArgumentError` naming the type, the slot and the method to declare. This is what makes the
derived half safe to rely on: forgetting the resolving half is loud rather than silent.

### The value-level check had the same vector gap

`assert_resolved_slots` recursed into a slot that held one child and **not** into a slot that held
a vector of them, so `NonOptimisationRiskRatio(r1 = [StandardDeviation(sigma = ce)])` passed the
bare-returns-matrix entry point and failed several frames down. It now takes the same
element-by-element arm as `resolve_deferred_child`. Both consumers of the declaration read it the
same way.

### Rejected alternatives

**Generate the resolver from a field tag, beside `@fprop`.** A sixth `@propagatable` tag would
put the declaration on the field and generate both halves from it, with no runtime reflection.
Rejected because `RiskTrackingRiskMeasure` is not `@propagatable`, so the tag would not reach the
container the review named first, and because the macro cannot tell a container from a leaf — it
would have to generate a resolver for `Variance` too, and that one must stay hand-written.

**Take a dependency for the rebuild.** `ConstructionBase.setproperties` does exactly what
`rebuild_with_slots` does. Rejected under the standing preference for deriving the field list and
recovering the constructor over adding a dependency for reflection-style work; the hand-written
version is three lines and re-runs the inner constructor's guards.

## Amendment (2026-08-17) — from candidate C of the architecture review of 2026-08-16

The clause "the prior fallback for an unstated slot stays where it already was" **stands**, and
this amendment records why, because the review proposed to reverse it. The proposal was to reach
both halves of the precedence rule through one per-type declaration, so that a consumer holding a
prior applies the whole rule by naming one function. Reading the shipped code found one real defect
behind that proposal, and two independent reasons the proposal cannot be carried out: the moment
slots would lose the model's shared factor, and the `w` slots have no declaration that covers them.

### The value-level point was bypassed by two of its own callers

`resolve_risk_inputs` is this decision's value-level resolution point, and the ratio family did
not use it. `expected_ratio` and `expected_risk_ret_ratio` handed the risk axis `pr.X` and handed
the return axis `pr`, two lines apart. So an unstated slot met the kernel as `nothing`, and a
Deferred Quantity was refused by an error whose own text reads "Pass the prior result itself —
`expected_risk(r, w, pr, fees)`", which is what the caller had done. The `VecVecNum` twin resolved
first, so `expected_risk(errr, [w], pr)` answered where `expected_risk(errr, w, pr)` threw.

**Both now hand the risk axis `pr`.** This is not a change to the decision. It makes two callers
use the point the decision already named. A third caller had the same defect and the review did
not cite it: `expected_risk(r, res::OptimisationResult, pr, fees)` took the prior out of the
result and then unwrapped `pr.X`, so `expected_risk(Variance(), res)` — the call its own docstring
asks for — reached the kernel with an unstated `sigma`. No test in the suite called that overload,
because every other site passes `res.w` by hand. It now forwards the carrier whole.

### The fallback has three spellings, not two

The review counted two, a declarative `@pprop` tag and an imperative selector call. There is a
third, and it is the one that matters:

| Spelling | Where | Count |
| -------- | ----- | ----- |
| `@pprop <field>` | on the struct body, read by the generated `factory` | 29 tags, 28 types |
| a hand-written `factory(r::T, pr, …)` method | when a tag cannot express the rule | `Variance`, `Kurtosis`, … |
| `nothing_scalar_array_selector(r.w, pr.w)` | inside a `JuMP` risk builder | 23 sites |

Of the 29 `@pprop` tags, 28 are `w` and one type adds `mu`. **`Variance.sigma` carries no tag at
all**, and this is deliberate: `@pprop` selects field by field, and `sigma`/`chol` must be
selected as a pair or the model optimises one quantity while the functor evaluates another. A
field tag cannot say "as a pair", so `Variance` writes its `factory` by hand. Candidate B's
rejected alternatives reach the same place from the other direction: a generated resolver "would
have to generate a resolver for `Variance` too, and that one must stay hand-written."

Pairing is not the only thing a tag cannot say. `Kurtosis` writes its `factory` by hand for a
different reason: it needs one method per prior type, because a `HighOrderPrior` carries `kt` and
a `LowOrderPrior` does not, and the second method has to fall the slot back to nothing rather
than to a field that is absent. A field tag names a field, not a prior type.

### Pre-resolving the measure would destroy the shared factor

Three of the four `JuMP` optimisers hand the builders the raw measure — `MeanRisk`,
`RiskBudgeting` and `FactorRiskContribution` pass `*.r` straight into `assemble_jump_model!`.
`NearOptimalCentering` passes a `factory`-resolved one, so for that optimiser alone the builders'
fallback calls are already no-ops. That is a third behaviour among four heads, and it is the part
of the review's reading that holds up.

It does **not** follow that the builders should resolve first. `chol_sigma_selector` reads
`isnothing(r.sigma) && isnothing(r.chol)` as the signal that the prior supplies the pair, and only
on that branch does it take the model-level `G` through `get_chol_or_sigma_pm`, which caches one
expression across every measure that needs a factor. Resolve the measure before the builder and
that signal is gone: each measure arrives holding a dense `sigma`, takes the
`LinearAlgebra.cholesky(r.sigma).U` branch, and the single shared `G` becomes one factorisation
per measure. The failure is silent and it is in the model, not in an error.

So the moment slots must keep their fallback in the builder. This is also why the review's
companion item, to collapse `sigma_chol_selector` and `chol_sigma_selector` into one, is
**rejected**: the first returns a pair with the fallback applied, the second returns one `JuMP`
expression and the cache decision, and once the pair is in hand the caller can no longer tell
whether the prior supplied it.

### The `w` half is not derivable from the declaration either

The `w` half looked like the consolidatable remainder, and it is not. Its fallback carries no
model-level cache, so nothing is lost by resolving it early: each builder reads
`wi = nothing_scalar_array_selector(r.w, pr.w)` and then branches on `isnothing(wi)` to choose
between a weighted and an unweighted expression. That branch is the *result* of the fallback and
not a signal about its source, so it survives a pre-filled `w` unchanged. On that much the review
is right.

**The declaration does not cover the sites.** Of the 23 builder sites, 19 belong to a type tagged
`@pprop w`. The other four belong to `LowOrderMoment`, which is a bare `@concrete struct` and not
`@propagatable` at all, so it cannot carry the tag. A pass derived from `@pprop` would fill 19 and
skip 4 in silence, and the symptom would be a risk measure that quietly ignores the prior's
observation weights.

Two ways to close that gap, both rejected:

**Make `LowOrderMoment` `@propagatable` and tag `w`.** Rejected. It declares a hand-written inner
constructor that guards `mu` and `w`, and the generic constructor `@concrete` generates for a
`@propagatable` type bypasses a narrow inner bound rather than raising a `MethodError`. Buying a
23-site consolidation with a silent validation bypass is the wrong trade. `HighOrderMoment` has the
same shape.

**Declare the prior slots per type, beside `deferred_slots`.** Rejected. It replaces 23 imperative
sites with about 20 declarations, which is not leverage, and it adds a sixth declaration mechanism
to a family that already has five. Candidate B rejected a field tag for the neighbouring problem on
the same ground, that `RiskTrackingRiskMeasure` is not `@propagatable`; the constraint here is the
same one, met from the other side.

So the fallback stays in the builders for the `w` slot too, and the clause this amendment opened
with is now confirmed for both halves rather than only for the moment slots. The drift the review
saw is real — 23 copies of one rule — but every mechanism that would remove the copies costs more
than the copies do.

## Amendment (2026-08-28) — from [#586](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/586)

Two clauses of this ADR moved when the calibration slot shipped. Both are recorded in full by
ADR 0095 and ADR 0096; this amendment says what changed here.

**The per-type entry point no longer resolves the deferred state and nothing else.** It resolves a
Calibration Rule as well, in the same method and the same rebuild, because a measure that carries
both kinds of slot must not be rebuilt twice and because the two resolutions have an order between
them. The declaration and the one-slot resolver stay parallel — `calibration_slots` beside
`deferred_slots`, `resolve_calibration_slot` beside `resolve_slot` — so only the entry point is
shared. Its name is now narrower than the method it names.

**Its signature is `resolve_deferred_quantities(x, pr::AbstractPriorResult, slv = nothing)`.** The
third argument is the effective solver, which a rule may need in order to call `ERM` or `RRM`. On
the `factory` route the selection has already settled it; on the `JuMP` route
`set_risk_constraints!` threads `opt.opt.slv` in, because no selection runs there.

**The `factory` resolution point resolves last.** The generated prior `factory` used to resolve
before it selected. It now selects first and hands the selected struct to the resolution, so a
deferred slot sees the solver, the observation weights and the children already settled. The three
resolution points of this ADR are unchanged; only the order inside the first one moved. ADR 0096
carries the decision.
