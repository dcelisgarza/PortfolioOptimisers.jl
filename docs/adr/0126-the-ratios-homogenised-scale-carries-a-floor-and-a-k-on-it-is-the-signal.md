---
status: accepted
---

# The ratio's homogenised scale carries a floor, and a `k` on it is the signal

## Context

`MaximumRatio` cannot optimise its quotient directly, so it homogenises: it introduces a scalar
`k`, optimises in `y = k w`, and de-homogenises with `w = y / k`. Every constraint the model
writes is then homogeneous in `(y, k)` — `A y - k B <= 0`, the budget `sum(y) = bgt k`, the
weight bounds `lb k <= y <= ub k` — so **all of them hold at the origin**. Only the normalisation
keeps the solver off that ray, and the two normalisations do not hold it equally well:

- The **return form** writes `mu' y - rf k = ohf` with `ohf > 0`. Under finite weight bounds
    `k = 0` forces `y = 0` and the equality then reads `0 = ohf`, so the ray is infeasible.
- The **risk form** writes `R(y) <= ohf` and maximises `ret - rf k`. That is an inequality, and
    the origin satisfies it.

The risk form is what a robust return term forces: a box, ellipsoidal or norm-ball **mean**
uncertainty set raises a cone the return form cannot carry, so `any(robust)` selects the risk form
outright. When the set's radius is large enough that no feasible portfolio's worst-case return
beats `rf`, the objective `ret - rf k` is non-positive along every ray of the feasible cone, its
supremum is zero, and it is attained at the origin. The solver walks down that ray and stops where
its own feasibility tolerance stops it.

Issue [#924](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/924) measured what that
costs. Measured again here on twenty assets, an `ub = 0.1` bound and a binding linear row, over
five risk measures:

| Risk measure | `k`, healthy | `k`, collapsed | `max w` collapsed, against `ub = 0.1` |
| --- | --- | --- | --- |
| `Variance` | 7.06 | 6.6e-9 | 0.1149 |
| `StandardDeviation` | 0.223 | 4.2e-10 | 0.1570 |
| `ConditionalValueatRisk` | 0.121 | 7.2e-10 | 0.0804 |
| `WorstRealisation` | 0.108 | 1.1e-10 | 0.3574 |
| `MaximumDrawdown` | 0.050 | **-2.1e-11** | 0.1396 |

Every row returns `OptimisationSuccess`. At a `k` of that size each homogeneous constraint holds
to about the solver's tolerance rather than on its own terms, so the recovered `w = y / k` breaks
the caller's mandate — the linear row it was written against, and the weight bound nobody wrote it
against — while nothing in the result says the scale collapsed. `MaximumDrawdown` comes back at a
**negative** `k`, so even the existing `k >= 0` on the variable holds only to tolerance. A denser
row carries more of the violation: a `FactorSpace` exposure row, dense by construction, misses a
0.9 exposure mandate outright.

The issue named three routes and declined to choose between them: bound the scale, refuse after
the solve, or refuse the combination outright. Refusing after the solve is smaller but leaves the
combination unusable, and refusing it outright removes a model that is perfectly well posed
whenever the set is narrow enough — the first three rows of the table above are exactly that model,
and they are correct today.

## Decision

**`MaximumRatio` gains a third field, `kmin`, and `k`'s own lower bound is tightened to it.**
`set_maximum_ratio_factor_variables!` already declares `k >= 0`; `set_maximum_ratio_scale_floor!`
raises that same bound, beside the normalisation, **on the risk form** — the branch whose
normalisation admits the ray. A `kmin` the caller names is raised on either branch, because a
caller who asks for a floor is asking for this bound.

It is a variable bound and not a constraint row deliberately. A row carries a dual and moves the
interior point the solver lands on even where it cannot bind, and a `MaximumRatio` answer feeds
discrete allocators and stacked optimisers that this library pins far tighter than that
perturbation.

`kmin` follows `ohf`'s rule exactly, which is why it lives beside it: a number is validated
(`kmin > 0`) and used as written, and `nothing` sizes it from the resolved aggregate
characteristic,

```text
kmin = 1e-4 * ohf / max(ohf, maximum(mu) - rf).
```

`ohf / (maximum(mu) - rf)` is the return form's own floor: no long-only fully invested portfolio
earns more than its best asset, so no such model can pin `k` below it. That quantity is **not** a
bound on the risk form, whose scale is `ohf^(1/d) / R(w)` for a risk measure homogeneous of degree
`d` — over the five measures above it lands between an eighth of that quantity and seventeen times
it, and using it undivided would silently bind on four of the five.

The `1e-4` is the margin, and three measurements fix it, not taste. The floor must sit **below**
the smallest scale a well-posed model pins — `MaximumDrawdown`'s, measured at `0.050`; **above**
the feasibility tolerance the collapsed ray answers on, or `w = y / k` carries a useless residual
anyway; and **below the point where a slack bound spoils the conditioning of the widest model that
reaches here**. That last one is not a guess: `ExactOrderedWeightsArray` under a `LogarithmicReturn`
solves cleanly at `k = 0.0742` with a floor of `1e-4` or smaller, and stops converging at `4e-4`
though the floor is three orders below its own scale. `1e-4` is the value that clears all three, and
it buys the middle constraint the least — a floored model meets a weight bound to about `1e-5`
rather than the `1e-7` a larger floor gave, against the outright breach the collapsed ray returned.

The denominator's `max` keeps a universe whose characteristic is smaller than `ohf` from lifting the
floor above `1e-4`.

The floor is deliberately **not** a refusal, and it does not need to be one, because it is
self-reporting. A feasible ray with a positive ratio is scaled by the normalisation alone, so the
floor can only bind on a model that has no tangency portfolio to find. A `k` that comes back
sitting on `kmin` is therefore the signal that the objective never rose above zero, and that the
weights beside it maximise the return expression at that scale rather than the ratio. The floor
being caller-visible is what makes that signal readable, and what lets a caller who wants the old
scale ask for it.

## Consequences

- The collapse is closed. Every row of the table above now pins at the floor and returns weights
    that meet both the linear row and the weight bound. `test_18q_degeneracy_guard.jl` holds the
    asset-axis case, the dense group row, the inertness of the floor where a tangency portfolio
    exists, the caller-set floor and the validation;
    `test_02b_factor_exposure_constraints.jl` holds the `FactorSpace` row, and reproduces the
    defect with `kmin = 1e-12` so the fix is pinned from both sides.
- The floor is inert where the ratio is well posed. Measured across the five risk measures, the
    healthy `k` is unchanged to seven significant figures, and a floor a million times smaller
    gives the same answer.
- An answer **can** move: a model whose optimal `k` falls below the floor now returns the floored
    point instead. Such a model had no tangency portfolio, so the point it used to return was the
    degenerate one, but the number does change and `kmin` is how a caller overrides it.
- **Inert is not free**, which is why the derived floor is neither registered uniformly nor
    written as a row. Both were tried first, and both moved answers they could not bind on. A
    floor written on the return form sits three orders below the scale that form produces and
    never binds there; written as a conic row it still moved `w` by about `1e-6`, and
    `test_23_finite_optimisation.jl` pins a discrete allocation downstream of one, which flipped
    onto a different integer answer. The branch the reader has to hold, and the bound rather than
    the row, are the price of not perturbing every return-form answer in the library.
- **The collapse is not confined to uncertainty sets, and the fix moves three stored weights.**
    `all(mu <= rf)` selects the risk form too, so a universe — or, under `NestedClustered`, an
    asset *cluster* on a cross-validation fold — whose sample means all sit below the rate
    collapses with no uncertainty set anywhere near it. Instrumenting one iteration of
    `test_22a_mix_optimisers.jl` found four such solves, every one returning `k` on the floor.
    Columns 19, 20 and 25 of `test/assets/NestedClustered.csv.gz` were therefore a record of the
    degenerate answer and are regenerated here; every other cell of that file is byte-identical,
    and the largest weight that moved moved by `8.8e-4`.
- `RiskBudgeting`'s `k` is untouched. It is the one head that does not come through
    `set_maximum_ratio_factor_variables!`, its log-barrier pins its own scale, and its `k` is free
    by design.

## Alternatives refused

- **Refuse after the solve** (#924's route 2). Strictly better than silence, but it leaves the
    combination unusable where the floor makes it answerable, and it still has to decide what a
    collapsed `k` is — which is the same threshold, spent on a worse outcome.
- **Refuse `MaximumRatio` with a positive-radius mean set outright** (#924's route 3). It removes
    a model that is correct whenever the radius is narrow, which is most of them.
- **`ohf / (maximum(mu) - rf)` undivided.** The exact return-form bound, and measurably too
    aggressive: it binds on `StandardDeviation`, `ConditionalValueatRisk`, `WorstRealisation` and
    `MaximumDrawdown`, silently changing four answers of five.
- **A bare constant.** `1e-4` clears the same measurements, but it does not move with the
    problem, and `ohf` beside it already sets a precedent for sizing a homogenisation number from
    the characteristic.
- **An equality on the risk cap**, `R(y) == ohf`, which also excludes the origin. A convex risk
    under an equality is a non-convex feasible set.
- **Sizing the floor from the risk side**, `ohf / max_i R(e_i)`, the exact bound for a long-only
    fully invested book under a convex homogeneous measure. It needs a value-level evaluation of
    the model's aggregate risk expression, which the seam that registers the floor does not hold,
    and it is not a bound at all once the risk measure's homogeneity degree is not one.
