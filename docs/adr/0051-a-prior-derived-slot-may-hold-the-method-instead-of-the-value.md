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
