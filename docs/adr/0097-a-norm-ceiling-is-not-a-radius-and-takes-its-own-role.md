---
status: accepted
---

# A norm ceiling is not an ambiguity radius, and it takes its own role

## Context

ADR 0070 widened twelve slots so that a radius could be computed rather than pasted, and ADR
0095 records the shape that shipped: a **Calibration Role** places a rule in the slot of one
quantity and **names the quantity**, and the slot's bound pairs `Number` with exactly one
role. A head role in a tail slot is refused at construction, and no guard method is written
for the mismatch.

Three slots of [`JuMPOptimiser`](../../src/20_Optimisation/10_JuMPOptimiser.jl) were left as
bare numbers by both decisions: `l2c`, `lpc` and `linfc`. They read as if they belonged to
the widened set, because they are spelled beside `l1` and `linf` and are read against the
same norms. They do not.

**A ceiling is not a radius.** `l1` and `linf` are coefficients: the model adds
`l1 * norm(w, 1)` to the objective, and ADR 0070's ground-metric table gives the identity
that makes such a coefficient the radius of a Wasserstein ball. `l2c` is a bound:
`set_weight_norm_2_constraints!` emits `norm(w, 2) <= l2c * k`. The reciprocal of that bound
is a floor on the effective number of assets, which the builder's own docstring states as
`l2c = 1 / sqrt(m)` for `m` effective assets. A diversification floor says nothing about the
set of measures the model prices, and neither shipped radius rule computes one:
`ConcentrationRadius` returns a scale times `sqrt(chi2_q / T)` and `RateRadius` returns
`c / sqrt(T)`.

**One slot already carried the confusion.** `LpRegularisation` is a penalty in the `lp`
field and a norm constraint in the `lpc` field, and ADR 0070 widened its `val` to
`Num_AmbRadCal` for the penalty reading alone. The type is shared, so `lpc` inherited a
bound it was never meant to have. Nothing resolved it: `set_weight_norm_p_constraints!` read
`lp.val` raw and handed it to a JuMP expression, so an `AmbiguityRadiusCalibration` in
`lpc` reached the solver layer and failed there with a message about JuMP. The `factory`
docstring on that type asserted the opposite — "That field keeps its own bound, so nothing
there widens and this method never reaches it" — which was never true.

## Decision

**A Norm Ceiling is its own quantity, and it takes its own family, role and bound.**

The domain noun is a **Norm Ceiling** (`CONTEXT.md` section 3.9), defined against the
Ambiguity Radius it is not. The mechanism is ADR 0095's, unchanged: a family of rules under
`AbstractCalibrationAlgorithm`, a `Func_` bound for the `alg` field that also admits a plain
`Function`, one role under `AbstractCalibrationEstimator`, and a `Num_` bound that pairs the
role with `Number`.

```julia
AbstractNormCeilingCalibrationAlgorithm <: AbstractCalibrationAlgorithm
const Func_NormCeilCal  = Union{<:Function, <:AbstractNormCeilingCalibrationAlgorithm}
NormCeilingCalibration  <: AbstractCalibrationEstimator
const Num_NormCeilCal   = Union{<:NormCeilingCalibration, <:Number}
```

A ceiling names no end of a distribution, so the family carries **one** role and
`mirror_role` needs no method for it. That is the shape the radius family already has, and
for the same reason.

### Three slots widen

| Slot | Type | Bound |
| :--- | :--- | :--- |
| `l2c` | `JuMPOptimiser` | `TD_Option{<:Num_NormCeilCal}` |
| `linfc` | `JuMPOptimiser` | `TD_Option{<:Num_NormCeilCal}` |
| `val` | `LpRegularisation`, reached through `JuMPOptimiser.lpc` | `Num_AmbRadNormCeilCal` |

### One rule ships, and it refits

`EffectiveAssetFloor(; fraction = 0.5, p = nothing)` returns the ceiling that holds
`fraction * N` assets effective: `m^(1/p - 1)` for a finite order, and `1/m` for the
infinite one.

The count the ceiling is read against is the order-`p` effective number of assets,
`(sum_i |w_i|^p)^(1/(1 - p))`. It is `number_effective_assets` taken to an arbitrary order:
the two are one number at `p = 2`, and at every order an equal-weight portfolio over `m`
assets reports exactly `m`. So the infinite arm is the limit of the finite one, because
`m^(1/p - 1)` tends to `1/m` as `p` grows, and the two arms meet.

The sweep of issue #617 corrected this. The first draft of the rule, and the
`set_weight_norm_p_constraints!` builder it mirrored, read the count as
`1 / ||w||_p^p`. That expression is the same number at `p = 2` and is not an effective
count anywhere else: an equal-weight portfolio over ten assets reports one hundred at
`p = 3`, which is a count above the size of the universe. Both statements now carry the
order-`p` reading.

The rule states a **fraction of the universe** rather than a count, and it reads `N` off the
Prior. So a cluster, a subset view and a cross-validation fold each get the floor their own
universe earns. A stated count would be pinned to the universe it was written for, which is
the trade ADR 0070 records for every other rule.

**The role's bound is where the radius family is refused.** `Num_NormCeilCal` names one role,
so an `AmbiguityRadiusCalibration` in `l2c` is a `TypeError` at construction, and a
`NormCeilingCalibration` in `l1` is the same. No guard method is written for either.

### The norm order travels through the rule

A ceiling is read against one norm order, and that order belongs to the **constraint** rather
than to the rule. One rule placed in `lpc` serves every term, and each term carries its own
`p`. `key` names the slot and not the order, so the order reaches the rule as `ctx.p`, the
norm-order field of the `CalibrationContext` each constraint site builds. This is ADR 0095's
travelling shape, and the sibling significance level is its precedent.

**No rule carries an order of its own.** A rule cannot know which of the three sites it
reached, so there is no field for the site's order to overwrite and no precedence between the
two to state. A caller who runs the rule outside those sites builds the context the site
would have built.

### The dual-use slot is settled by the field that holds the term, not by its bound

`LpRegularisation.val` is the one slot in the library that two readings share, so it is the
one exception to ADR 0095's rule that the bound is the whole of the role validation. One
field cannot carry two bounds. `Num_AmbRadNormCeilCal` therefore admits both roles, and two
`assert_` methods settle the reading where it becomes known:

- `assert_penalty_coefficient_role` refuses a ceiling role. It runs in `JuMPOptimiser`'s
    constructor against `lp`, and again in `factory` for a term that reaches the objective by
    another route.
- `assert_norm_ceiling_role` refuses a radius role. It runs in `JuMPOptimiser`'s constructor
    against `lpc`, and again in `norm_ceiling_factory`.

**The constructor is the important half.** It fires where the caller wrote the field, which
is the point ADR 0095 insists on, and it is as early as any check can be for this slot. The
factory halves are the backstop for a term assembled by another path.

`norm_ceiling_factory` is a second verb rather than a `factory` method because the two routes
read one field as two quantities. Each refuses the role that has no reading on its own route,
and each resolves the slot under its own key: `:lpreg_val` for the penalty and `:lpc` for the
constraint. Both bind the term's own norm order first, because a rule placed in either field
serves every term and each term carries its own `p`.

### The three ceilings resolve at the constraint site

`assemble_jump_model!` already resolves `l1`, `l2`, `lp` and `linf` there, because
`JuMPOptimiser` has no value-level entry point and the bundle carries no slot for them. The
three ceilings join them, each binding its own order first: `2` for `l2c`, `Inf` for `linfc`,
and the term's own `p` for each entry of `lpc`. `JuMPOptimiser` still declares no
`calibration_slots`, so `assert_calibrated_slots` still has nothing to say about it.

## Rejected alternatives

**Bounding `l2c` and `linfc` at `Num_AmbRadCal`.** About twenty lines, and mechanically
identical to `l1` and `linf`. Rejected because it reverses ADR 0095's central rule: the role
would no longer name the quantity, the bound's name would contradict its slot, and both ADRs
would need an amendment saying so. The saving is a type and a `const`; the cost is the one
rule that makes every other bound in the mechanism readable.

**One `NormCeilingCalibration` with a `p` field the caller states.** No order in the context,
and the rule carries its own. Rejected because `lpc` holds several terms with several orders
and one rule serves them all, so a caller-stated order would be wrong for every term but
one. It would also let a caller state `p = 3` in `l2c`, where the answer would be a
ceiling for the wrong norm and nothing would catch it.

**Leaving `lpc` alone and only fixing its resolution.** Smaller, and it closes the reported
failure. Rejected because the failure is a symptom: the field would still admit a radius role
that has no reading there, and the refusal would still arrive from the JuMP layer rather than
from the library.

**A guard method instead of a bound for `l2c` and `linfc`.** Rejected on ADR 0095's grounds.
The bound already refuses every mismatch it can see, at construction, with no message to keep
in step.

**Shipping the family with no rule, as the tail weight ships.** Rejected because a ceiling
rule is not a guess. The reciprocal reading is already written in all three constraint
builders' docstrings, and `EffectiveAssetFloor` is that reading with the count read off the
Prior instead of pasted.

## Consequences

- **A diversification floor survives a refit.** A ceiling stated as a fraction of the
    universe re-derives per fold, per cluster and per subset view, which is what a pasted
    `1 / sqrt(m)` never did.
- **A defect is closed.** An `AmbiguityRadiusCalibration` in `lpc` used to reach a JuMP
    expression unresolved. It is now refused in `JuMPOptimiser`'s constructor, and a
    `NormCeilingCalibration` in `lpc` resolves.
- **ADR 0095's rule gains one stated exception.** Every bound but `Num_AmbRadNormCeilCal`
    still pairs `Number` with one role. That one is dual-use because the type it belongs to
    is, and the exception is confined to it.
- **The order a rule reads is never on the rule.** A caller who wants a particular order
    states it in a `CalibrationContext` and runs the rule by hand; inside a constraint the
    site's order is the only one there is.
- **A `TD_` wrapper holding a rule is still unspecified.** `l2c` and `linfc` join `l1` and
    `linf` as fields with two deferral channels, and ADR 0030 still considered only one.
    `test_09i_norm_ceiling_calibration.jl` records what the code does rather than ratifying
    a design, on the same terms as the radius family's file.
- **No rule computes a ceiling from anything but the universe size.** A rule that read the
    covariance matrix — a floor on effective *independent* bets rather than on assets — has a
    reading and no member. The family admits a caller's `Function` for it.
