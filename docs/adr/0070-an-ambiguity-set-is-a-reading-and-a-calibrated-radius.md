---
status: accepted
---

# An ambiguity set is a reading, and what is missing is a calibrated radius

## Context

Report 9 asked for three ambiguity families — Wasserstein (the data moves), Gelbrich (the moments
are wrong), and divergence (the probabilities are wrong) — and shipped a prototype for each. Issue
[#311](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/311) asked whether they become a
type family, and where they live beside
[`src/14_UncertaintySets/`](../../src/14_UncertaintySets/), whose five files hold Base, Delta,
Normal, Bootstrap and L1.

**All three collapse onto machinery that already exists.**

`L2Regularisation(; val, alg = SOCRiskExpr())` penalises `val * norm(w, 2)`
([`12_RegularisationConstraints.jl`](../../src/20_Optimisation/09_JuMPConstraints/12_RegularisationConstraints.jl)),
and `SOCRiskExpr` is the default. Blanchet, Chen and Zhou (2022) give
`sup{W2 <= delta} sd(w'xi) = sqrt(w'Sigma w) + delta * norm(w, 2)`, so that estimator is the
Wasserstein robust counterpart today.

Gelbrich collapses onto the same term, and prototype 11 says so in its own docstrings: its
worst-case standard deviation is `sqrt(w'Sw) + rho * norm(w, 2)` and its worst-case mean is
`dot(mu, w) - rho * norm(w, 2)`, which it calls "numerically identical to the type-2 Wasserstein
result". The mean half is an l2 ball on `mu`, which `EllipsoidalUncertaintySet` with an identity
shape already expresses.

The divergence family ships **twice over**.
[`EntropicValueatRisk`](../../src/19_RiskMeasures/08_EntropicXatRisk.jl) is the Kullback-Leibler ball
at radius `-log(alpha)`: its functor is the scalar minimisation `ERM(x, slv, alpha, w)` and its
constraint form is the exponential cone. `RelativisticValueatRisk` is the Renyi counterpart.

What none of them has is a **radius a caller can choose**. Every radius in the library is a bare
number written by hand: `r` on the three distributionally robust measures, `val` on the two
Regularisation Estimators, `l1` and `linf` on `JuMPOptimiser`. A hand-written number is pinned to
the sample it was chosen for, and Cross-Validation refits per fold while a meta-optimiser refits per
subproblem. So the radius is stated for the whole universe while every other input is re-derived.

The mechanism for fixing that also already existed, confined to one family.
[`01_Base_UncertaintySets.jl`](../../src/14_UncertaintySets/01_Base_UncertaintySets.jl) carries
`AbstractUncertaintyKAlgorithm`, the alias `Num_UcSK = Union{<:AbstractUncertaintyKAlgorithm,
<:Number}`, a member that computes a radius from a chi-squared quantile and the data, and the
pass-through `k_ucs(type::Number, args...) = type`. `AbstractUncertaintyEpsAlgorithm` and
`Num_UcSEps` sit beside it.

## Decision

**No ambiguity set becomes a type. A radius slot admits the rule that computes it.**

The domain noun is an **Ambiguity Set** (`CONTEXT.md` section 3.9), and it names a *reading* of
existing machinery rather than an object. An Uncertainty Set stays what it was: a neighbourhood of a
mean vector or a covariance matrix, that a caller constructs and passes. Nothing is added to
`src/14_UncertaintySets/`.

### One calibration mechanism, generalised rather than duplicated

```julia
abstract type AbstractCalibrationAlgorithm <: AbstractAlgorithm end

AbstractUncertaintyKAlgorithm   <: AbstractCalibrationAlgorithm   # re-parented
AbstractUncertaintyEpsAlgorithm <: AbstractCalibrationAlgorithm   # re-parented
AmbiguityRadiusAlgorithm        <: AbstractCalibrationAlgorithm   # new
AmbiguityTailWeightAlgorithm    <: AbstractCalibrationAlgorithm   # new
```

`Num_UcSK` and `Num_UcSEps` keep their meaning and their spelling; only their supertype moves. The
shared root is what lets one resolver and one census test see the whole mechanism, and it is the
reason the library does not gain a second way to spell the same idea.

### One alias per quantity, following the two precedents

ADR 0051 gave `MuSlot`, `SigmaSlot`, `KtSlot` and `SkSlot`; ADR 0048 gave `HopCountValue` and
`PathLengthValue`. This decision gives two more:

```julia
const AmbiguityRadiusValue     = Union{<:Number, <:AmbiguityRadiusAlgorithm, <:Function}
const AmbiguityTailWeightValue = Union{<:Number, <:AmbiguityTailWeightAlgorithm, <:Function}
```

The noun is **Ambiguity Radius**, not Wasserstein radius. A Gelbrich ball and a type-2 Wasserstein
ball emit the same term, so naming the quantity after one of the two would be too narrow.

### Twelve slots widen, across six types

| Alias | Slots | Type |
| :--- | :--- | :--- |
| `AmbiguityRadiusValue` | `r` | `DistributionallyRobustConditionalValueatRisk` |
| | `r_a`, `r_b` | `DistributionallyRobustConditionalValueatRiskRange` |
| | `r` | `DistributionallyRobustConditionalDrawdownatRisk` |
| | `val` | `L2Regularisation` |
| | `val` | `LpRegularisation` |
| | `l1`, `linf` | `JuMPOptimiser` |
| `AmbiguityTailWeightValue` | `l` | `DistributionallyRobustConditionalValueatRisk` |
| | `l_a`, `l_b` | `DistributionallyRobustConditionalValueatRiskRange` |
| | `l` | `DistributionallyRobustConditionalDrawdownatRisk` |

**`r` is the radius and `l` is a tail weight**, which is the reverse of what the name suggests. ADR
0051's sibling work in issue #313 rewrote both field texts after checking the constraint code, and
the constraint body itself reads `b1 = r.l` and `radius = r.r`. The two quantities therefore take
two different aliases, and a reader who assumes otherwise binds them backwards.

Including `l1` and `linf` completes the ground-metric table. A type-1 ball with ground metric order
`q` yields a penalty in the dual norm `p`:

| ground metric `q` | dual norm `p` | slot |
| :--- | :--- | :--- |
| 1 | inf | `JuMPOptimiser.linf` |
| 2 | 2 | `L2Regularisation` with `SOCRiskExpr` |
| inf | 1 | `JuMPOptimiser.l1` |
| between | `1/p + 1/q = 1` | `LpRegularisation` |

`LpRegularisation` validates `p > 1` and `isfinite(p)` and so reaches neither end, but `l1` and
`linf` are separate fields with their own constraint builders. **No cone is missing.**

### The channel is Factory

A rule resolves during `factory`, against the optimisation's own Prior, and the struct is rebuilt
through its ordinary keyword constructor. **Every existing validation therefore re-runs on the
calibrated number with no new code.** This is ADR 0051's channel and not ADR 0048's: resolving at
the point of use rebuilds nothing, so ADR 0048 had to place refusal methods in every kernel that
read a rule.

The cost is that `factory(opt::JuMPOptimiser, ...)` forwards `l2` and `lp` untouched today, so each
Regularisation Estimator needs a `factory` method and the optimiser's factory needs to call it. The
three distributionally robust measures are already `@propagatable` and already own the channel.

### Two rules ship, and both refit

- `ConcentrationRadius(; confidence = 0.95, scale = nothing)` — the Blanchet, Kang and Murthy
    (2019) form `scale * sqrt(chi2_q(confidence, N) / T)`. **`scale = nothing` reads the Prior**
    rather than demanding a number in return units. `Distributions` is already a direct dependency,
    so the quantile is exact.
- `RateRadius(; c = 1.0)` — `c / sqrt(T)`. The parametric rate is the trustworthy part and the
    constant is the part to calibrate, so this is the form a `GridSearchCrossValidation` grid moves
    over.

Both read `T` and `N` from the Prior, so both re-derive per fold and per subproblem. That is the
whole reason a rule beats a number, and it is the same trade ADR 0048 records for
`PathLengthQuantile`: a stated value holds the radius still and lets the derived quantity move,
while a rule holds the derived quantity still and lets the radius move. Neither is stable in both
senses, because the fit moves either way.

**No rule ships for the tail weight.** The alias admits a bare `Function`, but nothing computes an
Esfahani-Kuhn tail weight, and inventing one here would be a guess.

### The guard is an overloadable assertion

The Blanchet, Chen and Zhou identity holds **only** for `SOCRiskExpr`, which penalises
`norm(w, 2)`. `SquaredSOCRiskExpr`, `QuadRiskExpr` and `RSOCRiskExpr` penalise `norm(w, 2)^2`, where
a radius is the wrong quantity. `L2Regularisation` stores `alg` beside `val`, so its inner
constructor refuses the pairing through an `assert_` method in the style of
[`src/01_Base.jl`](../../src/01_Base.jl)'s family: a permissive fallback plus a refusing method, so
**a new formulation adds a method rather than editing a hardcoded check**. A plain `Number` in `val`
stays legal with every formulation.

This departs from ADR 0051's ruling of flexibility plus a warning over refusal. ADR 0051 refused to
refuse because a caller who states a robust `mu` beside a sample `kt` is doing something legitimate
that the library cannot distinguish from an accident. A radius beside a squared penalty has no
legitimate reading, which puts it with `assert_derived_slot_has_source` instead.

## Rejected alternatives

**A new `src/15_AmbiguitySets/` family.** Three sets, an abstract supertype, a Result type and a
resolver, with the existing sites re-expressed through them. Rejected because every one of the three
already has a working expression in the library, so the family would add surface area and no
capability. `src/` is 109,665 lines with 229 abstract types, and the map's own rule is to prefer a
seam over a leaf.

**A calibration function returning a bare number.** `wasserstein_radius(T, N; confidence)` called by
the caller, whose answer is pasted into the field. Rejected for the reason ADR 0051 gives: a pasted
number does not survive a refit, and it crosses the view boundary as the whole universe's answer
while every other input is re-derived on the subset.

**A sibling calibration family, leaving `AbstractUncertaintyKAlgorithm` alone.** Nothing existing
would break. Rejected because the library would then hold two unrelated spellings of number-or-rule
and no census test could see them as one thing.

**Reusing `Num_UcSK` directly with no new supertype.** The smallest possible change. Rejected
because `k_ucs`'s signature is `(alg, q, X, sigma_X)`, shaped for a confidence level, and a radius
rule that needs the whole Prior does not fit it without a change.

## Consequences

- **A caller can say which ambiguity they mean, and the library computes the size.** The three
    families keep their distinct meanings in prose and in `CONTEXT.md` without three distinct types.
- **The abstract hierarchy breaks.** Two supertypes are re-parented. The package is pre-1.0 and
    the map's charter permits it; the aliases callers actually write are unchanged.
- **Validation stays where it is.** Because Factory rebuilds through the keyword constructor, no
    range check moves and no kernel gains a refusal method.
- **A rule must return its own field's quantity.** There is no unit conversion in the resolver, so
    a rule that wants to be configured in other units converts inside itself.
- **The tail weight and `kappa` have slots with nothing in them yet.** Both admit a `Function`
    today. Issue [#352](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/352) carries
    `kappa` and the 43 tail-probability slots.
- **The entropic docstrings understate what they implement.** `EntropicValueatRisk` and its seven
    relatives are divergence balls and say so nowhere, so a caller cannot find them. Issue #352
    carries the edits.
- **A `TD_` wrapper holding a calibration rule is unresolved.** `JuMPOptimiser.l1` and `.linf` are
    bounded `TD_Option`, so such a slot has two deferral channels and ADR 0030 considered only one.
    Nothing has checked what the code does today.

## Amendment (2026-08-28) — from [#586](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/586)

Map [#580](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/580) built the rest of this
mechanism, and three statements above are now out of date. ADR 0095 records the shape that shipped.

**The re-parenting has not shipped.** `AbstractUncertaintyKAlgorithm` and
`AbstractUncertaintyEpsAlgorithm` still subtype `AbstractAlgorithm` directly.
`AbstractCalibrationAlgorithm` lives in
[`src/19_RiskMeasures/01_Base_RiskMeasures.jl`](../../src/19_RiskMeasures/01_Base_RiskMeasures.jl),
beside `resolve_slot` and the mechanism that reads it, and re-parenting the two families needs the
root in [`src/01_Base.jl`](../../src/01_Base.jl) first. So "the abstract hierarchy breaks" describes
a break that has not happened, and `Num_UcSK` and `Num_UcSEps` are unrelated to the calibration
bounds today.

**A quantity takes a role, and the alias is named for it.** `AmbiguityRadiusValue` and
`AmbiguityTailWeightValue` are not the spellings that shipped. The rule lives in the `alg` field of
a **Calibration Role**, and a slot is bounded by `Num_AmbRadCal` or `Num_AmbTwtCal`, each pairing
`Number` with one concrete role. The two quantities keep the two separate bounds this ADR insisted
on, for the reason it gives: `r` is the radius and `l` is a tail weight.

**`kappa` has a rule.** `EntropyBudget` computes the Kaniadakis deformation parameter that spends a
stated entropy budget. The tail weight still has none, so that half of the observation stands.

## Amendment (2026-08-29) — from ADR 0097

**The `lpc` field was widened by accident, and nothing resolved it.** This decision widened
`LpRegularisation.val` for the penalty reading of the `lp` field. The same type also serves
as a norm *constraint* through `JuMPOptimiser.lpc`, so that field inherited a bound naming a
role that has no reading in it, and `set_weight_norm_p_constraints!` read `val` raw. An
`AmbiguityRadiusCalibration` placed in `lpc` therefore reached a JuMP expression unresolved.
ADR 0097 closes it: the field refuses a radius role at construction, and a ceiling role in it
resolves.

**The three norm ceilings are not radius slots.** `l2c`, `lpc` and `linfc` are spelled beside
`l1` and `linf` and are read against the same norms, so the twelve-slot table above reads as
though it had passed over them. It did not. A radius is the coefficient of a norm penalty and
a ceiling is a bound on that norm, and neither shipped rule of this decision computes a
ceiling. ADR 0097 gives the ceiling its own family, role and bound, on the shape ADR 0095
records.

## Amendment (2026-08-29) — from issue #613

**The tail weight has a rule.** `TailTermParity` computes the Esfahani-Kuhn tail weight that
prices the tail term of the loss at a stated multiple of its mean term. The observation above
that "no rule ships for the tail weight" and that "inventing one would be a guess" was right
about a preference and wrong about a unit: `l` is dimensionless and is not scale-free in the
sample, so one stated number is a different trade-off at every sampling frequency. The rule
carries the sample's own units, and the preference stays in the caller's `ratio`.

**A second pair travels through `bind_alpha`.** The rule reads the significance level of its
own slot, because its tail-term scale is a CVaR at that level. `alpha` and `l` therefore
travel together on the shape `alpha` and `kappa` already use, and the three sites that resolve
an `l` slot bind the level first.
