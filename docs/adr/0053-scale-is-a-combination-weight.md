---
status: accepted
---

# `scale` is a combination weight

## Context

Four places in the package multiply something by a `settings.scale` before adding it to a
running total:

1. the model's `:risk_vec`, through
   [`set_risk_expression!`](../../src/20_Optimisation/20_RiskMeasureConstraints/01_BaseRiskConstraints.jl),
2. the value-level `scalarise` closures in Near Optimal Centering, Hierarchical Risk Parity
   and Hierarchical Equal Risk Contribution,
3. the model's `:ret_vec`, through
   [`set_return_expression!`](../../src/20_Optimisation/09_JuMPConstraints/02_Returns_and_ObjectiveFunctions.jl),
4. Schur Complement Hierarchical Risk Parity's blend over its parameter bundles.

`scale` had no ADR and no `CONTEXT.md` entry. ADR 0024 owns the `scalarise` **seam** and never
mentions `scale`; ADR 0052 owns the return expression and states an invariance for one term
only at `scale = 1`. The meaning of the number itself lived in source comments and in
`field_dict`, and the two axes' field docs had drifted apart: `field_dict[:scale_rm]` carried
an inertness clause and `field_dict[:scale_rt]` carried none.

That drift was not cosmetic. The risk axis dropped a lone measure's weight before building
`:risk`; the return axis did not. So a lone return term at `scale = 2` built
`model[:ret] == 2 * model[:ret_1]`, and nothing said whether that was intended.

The question that forced the issue: does the rule bind only the value-level `expected_*`
reporting surface, or the whole package, the JuMP model included?

## Decision

**`scale` is the weight an element carries inside a combination of several elements. One
element is not a combination, so the weight is inert.**

The noun is deliberately wider than "a risk measure inside an aggregate of risk measures". It
has to be, because Schur is a site where `scale` weights **portfolios**, not risks: its
`_optimise(::…{<:Any, <:AbstractVector})` accumulates `w .+= ps.r.settings.scale * wi` and then
divides by `sum(w)`. Each `wi` sums to one, so the scales are a relative blend weight over
weight vectors — a genuine weighted average of portfolios.

The rule binds **every** site, the model included, and it binds both axes. A singular route
drops the weight before it can reach an expression:

- the risk axis, through `unit_scale_risk_measure` at the singular `set_risk_constraints!`,
- the return axis, through `unit_scale_returns_estimator` at the singular
  `set_return_constraints!`.

Both helpers return their argument unchanged when the scale is already one, reset **only**
`scale`, and preserve every other setting — `ub`/`lb`, `rke`/`rte`, `fee` and `mic`. None of
those is a weight.

## Consequences

**Inertness is now true rather than nearly true.** Across `MeanRisk` and Near Optimal Centering
under `MinimumRisk`, `MaximumUtility`, `MaximumRatio` and `MaximumReturn`, a lone element at
`scale = 50` gives weights identical to `scale = 1` to the last bit. At `scale = 1` every
result is bit-identical to the behaviour before this decision, so the change is a no-op for
every caller who never set the field.

**`MaximumUtility` is the objective that exposed it, on both axes.** It optimises
`ret - l * risk`, so a weight on either side changes the trade-off instead of cancelling. The
other three objectives are argmax-invariant under a positive rescale, which is why the defect
survived. Before this decision, a lone element at `scale = 50` moved `MaximumUtility`'s weights
by **0.014** on the risk axis and by **0.268** on the return axis.

**The two sides of a barrier must move together, and a half-applied drop is worse than none.**
Near Optimal Centering compares a value-level risk target against the model's `:risk`
expression inside an exponential cone, `[log_risk, 1, rk_opt - risk]`. Dropping the weight from
the barrier target alone, while the model kept scaling, made `rk_opt - 50·risk` negative at the
optimum and the sub-problem **provably infeasible** under all three objectives tested. The
return axis carries the same hazard in a different shape: `pret` is read twice in the singular
route, and the second read feeds `aggregate_return_characteristic`, which applies the weight to
`mu_i` in its own right. The drop therefore rebinds the local `pret` **before both uses** —
dropping it at the first call alone would leave `MaximumRatio`'s normalisation scaled while
`:ret` was not.

**Schur was always conformant, but by a different mechanism, and that is worth saying once.**
Its singular inertness holds by **renormalisation**, not by omission: with one bundle the weight
cancels in `w / sum(w)`. Deleting that multiplication would be a no-op rather than a fix, so
Schur is not a template for the other three sites.

**The proof is structural, not numerical.** The regression test asserts
`model[:ret] == model[:ret_1]` as affine functions for a lone term at any scale, and that
`:risk` carries unit coefficients for a lone measure at any scale. It also asserts the converse
— that several terms or several measures do **not** collapse — so the test fails if a future
change either reintroduces the multiplier or over-applies the drop to the vector path. Solver
tolerances cannot weaken it.

**One accidental pin already existed, and it moved.** `test_20`'s scalariser testset compares a
lone `scale = 50` Near Optimal Centering run against its vector counterparts. The singular run
now solves on numbers fifty times smaller, which shifted solver conditioning by about `1e-5`
and pushed one comparison from just inside `1e-4` to just outside it. The portfolios agree to
five significant figures; the tolerance moved to `5e-4`. This is exactly the kind of pin the
structural test replaces.

**A latent trap is recorded, not acted on.** `unit_scale_risk_measure(::HierarchicalRiskMeasure)`
is the identity. That is harmless today, because a hierarchical measure never reaches the model
and the hierarchical optimisers' singular paths never multiply in the first place. Under a
package-wide rule it is a trap for whoever next routes a hierarchical measure through a
scaling site.

**The value level is unchanged and already conformant.** `expected_return(::VecJRE, …)` applies
the weight per element and the singular method never reads it; the risk side behaves the same
way. `NoReturn` and `NoRisk` hold no per-asset quantity, so the weight was already inert on
them by an earlier decision and the new step is a no-op.

## Amendment (2026-08-18)

### A fifth site, and it had no application site at all

The decision above enumerates four sites and states that the rule binds every one. The census
missed a fifth: `Stacking.scale`. It is a length-`K` vector, one entry per inner optimiser, and
before this amendment the weight never reached the combination.

`_optimise` built `swi = wi .* transpose(st.scale)` and handed it to
`predict_outer_st_estimator_returns`, which fed `prepare_outer_rd`. Four consumers read the
matrix and three of them cancelled the weight:

- the outer returns matrix `X` never touched it — its columns come from
  `calc_net_returns(res, pr, fees)`, one per inner *result*,
- `iv` and `ivpa` collapse through `synthetic_asset_weights`, which divides each column by its
  sum,
- the feature matrix collapses through the same normaliser,
- `B = rd.B * swi` kept it.

So the one observable effect of the whole field was a tilt of the benchmark column. Both
cross-validated methods took the matrix as an argument and never referenced it, so under any `cv`
the field was entirely inert. The documented mathematics,
`w* = opto(Σ s_k W_k)`, described something the code did not do.

The three cancellations are **not** defects. `iv`, `ivpa` and a feature matrix are intensive, so a
sub-portfolio held at a different weight keeps the same rate and the same feature vector. Only the
returns column, the benchmark column and the recombination were candidates.

### The weight acts at the combination, not on the outer problem

`Stacking` differs from the four sites above in that an *optimiser* chooses the combination. The
weight is a fixed belief about the same quantity `opto` decides, so the two multiply and the
coefficients are rescaled:

    c = scale .* v,  w* = wi * (c * sum(v) / sum(c))

The alternative — scaling the synthetic return columns so `opto` sees the weight — was rejected on
two counts:

1. **A uniform weight would stop being neutral.** A common rescale of every synthetic return column
   moves `MaximumUtility`'s trade-off between return and risk. That is the same objective, and the
   same mechanism, that exposed the original defect on both axes.
2. **Every `predict_outer_st_estimator_returns` overload would have to re-apply it.** The two
   cross-validation methods already dropped it, and a custom overload would drop it silently. `cv`
   is execution control, and the surrounding code already commits to `cv` not changing what the
   outer problem measures.

### The rescale is what carries the rule

`combination_weights` rescales the tilted coefficients to `sum(v)`, the total `opto` itself chose.
That single factor delivers three properties, and they are the reason the field needs no normalised
form of its own:

- **A common factor cancels**, so the weight expresses a ratio between siblings and nothing else —
  which is what the decision above says it is.
- **A uniform weight gives back `v`**, whatever `v` sums to. The neutral setting is neutral.
- **One inner optimiser gives back `v`.** One element is not a combination, so the weight is inert
  on a lone element. `Stacking` therefore joins Schur in being inert by **rescale** rather than by
  omission, and like Schur it is not a template for the four sites that drop the weight instead.

Rescaling to `1` instead of to `sum(v)` was considered and rejected: it would silently overrule an
outer optimiser configured with a budget other than one.

### The degenerate case

The factor `sum(v) / sum(scale .* v)` does not always exist. A zero denominator is a tilt that
cancels a long-short outer allocation; a zero numerator is a dollar-neutral outer allocation, and
rescaling to it would collapse the portfolio. One test — the factor is zero or not finite — covers
both and the `0/0` of the two together, and the tilted coefficients then stand unrescaled. They are
finite and their ratios are intact, and no scalar rescale can hold a zero total while tilting.

A denominator that is merely near zero is not degenerate. The large factor is the answer, and if it
overflows, `finalise_weight_bounds` already reports an `OptimisationFailure`.

### What did not change

`scale` keeps its validation — finite, and matching `length(opti)` — and its `nothing` default. It
is **not** normalised at construction, because the rescale already makes a common factor
unobservable, so normalising would only change the stored value and the tests that read it.
`SubsetResampling` required `> 0` where `Stacking` requires only `isfinite`. That divergence is
resolved by amendment 2 below, which removes the field from `SubsetResampling` entirely.

At `scale === nothing` every result is bit-identical to the behaviour before this amendment.

## Amendment 2 (2026-08-18)

### `SubsetResampling.scale` is removed

A Combination Weight names an element. `SubsetResampling`'s elements are **randomly drawn asset
subsets**, so there is no element to name and the field is removed.

The randomness is not incidental to how the subsets are picked, it is both branches of
`sample_unique_assets`. When the combination count is within `max_comb`, the function samples
random *ranks* with `StatsBase.sample` and decodes each into a subset; otherwise it samples the
assets directly. Neither branch enumerates in order. Subset `i` is therefore "whatever the random
number generator drew `i`-th", and `scale[i]` weighted that.

Two further properties made the field unusable rather than merely useless:

- **Its own arity was not knowable at construction.** `n_subsets` may be a `NumberSubsetsE`
  resolved from the prior at solve time, which is why the length check lived in `_optimise` rather
  than in the constructor — the one field whose validation could not sit with the others.
- **The entries were not weights on comparable objects.** Each subset spans a different set of
  assets, so `scale[i]` and `scale[j]` did not price the same thing.

A caller who wants to weight specific sub-portfolios has `Stacking`, whose `opti` positions the
caller names and orders.

### What was removed

The field, its `> 0` validation, its `TimeDependent` slot, the `length(scale) == n_subsets` check
in `_optimise`, the `port_opt_view` forward, and the two `subset_resampling_finaliser` methods that
carried it. The two surviving methods lose their trailing dispatch argument. `fb` moves from type
parameter 14 to 13, so the `optimise` head that pins it to `Nothing` moved with it.

Behaviour is unchanged for every caller who never set the field: the surviving average
`w / n_subsets` is what a uniform `scale` already produced, and the four tests that asserted
`scale = ones(20)` equals no scale were the field's only coverage.
