---
status: accepted
---

# A tail view carries its own level and formulation

## Context

Every entropy pooling view before this one is a **linear function of the posterior
probabilities**. That is what lets `add_ep_constraint!` reduce a whole view family to rows of one
constraint matrix, and it is why `OptimEntropyPooling` can solve the dual: the caller adds no
variables.

A view on a quantile risk measure is not of that shape. [EPTail](@cite) adds views on the
posterior **CVaR** and **EVaR**, and each needs auxiliary variables of its own. They are a
different kind of statement about the distribution, so they are named a **Tail View** and they
reach the model through their own seam, `add_ep_tail_view!`.

Two settings had to be placed. Each tail view is read at a **significance level**, and each has
more than one **formulation**. The obvious home for both is the estimator, beside the view fields
it already carries — a `*_alpha` and a `*_alg` per measure. That home does not survive contact
with the domain. The CVaR at 1% and the CVaR at 10% are different statistics of the same series,
so an estimator-level level forces a caller who wants both to build two estimators and pool them.
A setting that can be written in two places is a setting that drifts from the view it belongs to.

## Decision

### The group is the unit

A tail view is stated as a **group**: the view equations, the level they are read under, and the
formulation that writes them.

```julia
ConditionalValueatRiskView(; views::LinearConstraintEstimator, alpha::Number = 0.05,
                           alg::Option{<:CVaRVF_VecCVaRVF} = nothing)
EntropicValueatRiskView(;   views::LinearConstraintEstimator, alpha::Number = 0.05,
                           alg::Option{<:EVaRVF_VecEVaRVF} = nothing)
```

`var_views`, `cvar_views` and `evar_views` each take one such group or a vector of them, and
nothing else. There is **no** estimator-level `*_alpha` or `*_alg`. `alg` takes one formulation
for every view in the group, or a vector for one per view.

A `prior(...)` reference inside a group resolves at **that group's** level, which is what makes a
view stated against the prior move with the level it is read under.

`ValueatRiskView` carries `views` and `alpha` and no `alg`. A VaR view is linear in the posterior
probabilities, so there is no formulation to choose and it reaches `OptimEntropyPooling` as
readily as `JuMPEntropyPooling`.

### `nothing` picks the cheapest formulation that is exact

Each measure has an **exact dual** formulation and a **general** one. The dual formulation
(`LinearConditionalValueatRiskView`, `ConicEntropicValueatRiskView`) is continuous and cheap, and
writes an equality as a lower bound, so it expresses `>=` exactly and `==` only where the target
is at or above the prior value. The general formulation
(`IntegerConditionalValueatRiskView`, `GridEntropicValueatRiskView`) expresses every operator, and
relative CVaR views, at the cost of binary variables and a solver that handles mixed-integer
exponential cone programs.

`alg = nothing` is the default and resolves per view:

```julia
function ep_cvar_formulation(::Nothing, single::Bool, op::Symbol, rhs::Number, pv::Number)
    return if single && (op == :geq || op == :eq && rhs >= pv)
        LinearConditionalValueatRiskView()
    else
        IntegerConditionalValueatRiskView()
    end
end
```

`ep_evar_formulation` is the same rule without the `single` term, because [EPTail](@cite) gives no
formulation for a relative EVaR view. A stated formulation is returned unchanged, so the caller
can always overrule the choice. The caller therefore pays for integers only where the view needs
them.

### The integer CVaR window is ascending, as the paper writes it

`IntegerConditionalValueatRiskView` sorts the loss series **ascending** and keeps the `sbar`
largest losses, so the window ends at the largest loss the sample holds. The CVaR tail is
therefore a **suffix** of the window, and the indicator monotonicity of eq. 3 of [EPTail](@cite)
is written in the paper's own direction:

```julia
[j = 1:(sb - 1)], sc1 * (y[j] - y[j + 1]) <= 0
```

that is `y[j+1] >= y[j]`. Once an indicator turns on it stays on, so the marked observations run
to the end of the window.

The order is a convention, not a degree of freedom. It fixes two other sites, and all three must
move together or the model constrains some statistic other than the CVaR:

1. `ep_add_cvar_view!` truncates the sorted order to its **last** `sbar` entries.
2. `ep_sbar` counts the positions at which the prior probabilities first reach `alpha` from the
   **end** of that order.
3. `add_ep_tail_view!` writes the monotonicity above.

## Consequences

`opt` must be a `JuMPEntropyPooling` whenever `cvar_views` is set, and whenever an `evar_views`
entry is anything other than a lower bound under `GridEntropicValueatRiskView`. That formulation's
lower-bound half is the one tail formulation linear in the posterior probabilities alone, so it is
the only half that reaches `OptimEntropyPooling`. The `@argcheck` says so in those words.

**A group name is a sum, not an average.** A group expands to its members, each carrying the
coefficient the group carried, so a view on a group constrains the sum of the members' CVaRs. A
group of more than one member is therefore a *relative* view, which sends CVaR to
`IntegerConditionalValueatRiskView` whatever the operator, and which EVaR refuses.

**`MeucciEntropyPoolingPrior` stays.** It is kept as its own estimator because its CVaR route is a
different *algorithm*, not a different formulation: it root-finds the Value at Risk level and
re-solves the whole entropy pooling problem at each candidate, where `EntropyPoolingPrior` runs
one solve. It reads the same groups, flattens them into that root-find, and refuses a group
carrying `alg`, since it writes no formulation to apply.

**Testing the two CVaR formulations.** The two constraint sets describe the same feasible
posterior, so on a shared `>=` view they must land on the same answer, and that agreement is what
pins the monotonicity direction. What separates them in practice is the outer-approximation gap
Pajarito stops on. Assert the **realised risk measure** on each posterior, and assert the
divergences against each other in the two directions that gap allows: the integer solve cannot
beat the continuous one (`pi_.kld >= pl.kld * (1 - 1.0e-6)`) and cannot stray far from it
(`isapprox(pi_.kld, pl.kld, rtol = 1.0e-2)`). Do **not** assert that the two weight vectors are
equal at that tolerance — the same tail mass is split visibly differently across the same
observations. `test/test_12h_entropy_pooling_tail_views.jl` holds the check.

**Naming.** A **Tail View** is not a **View**. The latter is the library's index-selection
mechanism; the collision is inherited from the entropy pooling literature, in which every
`*_views` field is a statement about the distribution. `CONTEXT.md` holds both terms and the
warning.

## Amendment (2026-09-01)

### The bracket is the third setting, and its name states its reading

This ADR placed the **level** and the **formulation**. A tail view whose measure is computed by a
scalar search carries a third setting, the **Search Bracket** the search runs over, and the ADR did
not place it. Two families placed it in two shapes, and both called the lower end `zlo`:

| Family | Field | Reading | Default |
| --- | --- | --- | --- |
| `EntropicValueatRiskView` | `zlo` | a fraction of the upper end, in `(0, 1)` | `nothing` |
| `RelativisticValueatRiskViewBracket` | `zlo` | an additive offset on the logarithm of the loss range | `-20` |

One name carried two parameterisations, and neither guard refuses the other's number: `zlo = 0.5`
on the relativistic bracket passes `zlo < zhi` and runs a search over a bracket four orders of
magnitude away from the one the caller meant. The two readings therefore take two names:

- `EntropicValueatRiskView.zlo_frac`, and the `zlo_frac` keyword of `ep_evar` and everything that
  forwards it.
- `RelativisticValueatRiskViewBracket.log_zlo` and `.log_zhi`.

`zlo` names nothing in the library.

### The shape follows the default

A search bracket takes one of two shapes, and the default decides which:

 1. **A field on the view estimator, defaulting to `nothing`**, where the default follows from the
    data. `EntropicValueatRiskView.zlo_frac` is one. Its default is `sqrt(eps(T))` for the element
    type `T` of the loss series, which a caller holding no data cannot write, so `nothing` resolves
    in `ep_evar`, where the data is.
 2. **Its own `AbstractAlgorithm` type**, where the defaults are plain numbers the caller can
    write. `RelativisticValueatRiskViewBracket` is one. Its three settings are data-independent,
    two searches read them, and a rule pairs two of them (`log_zlo < log_zhi`), so they earn a type
    whose constructor states that rule once.

A fourth tail-view family takes shape 1 for a knob whose default needs the data, and shape 2 for a
group of knobs whose defaults do not. `CONTEXT.md` holds the **Search Bracket** term and the
warning that its two readings are not interchangeable.

## Amendment (2026-09-03)

### A view over several assets is convex when its coefficients share one sign

This ADR said a group of more than one member is a relative view, which sends CVaR to the
integer formulation and which EVaR refuses. That reading was too coarse, and issue #350 asked
whether it could be lifted. Each of the three measures is a maximum of a linear function over a
set that is jointly convex in the dual weights and the posterior probabilities, so each is
**concave** in the probabilities. Two consequences replace the rule above.

- A view whose coefficients share one sign, `gA >= c`, is a positive combination of concave
  functions bounded from below: a convex set. The dual formulations write it exactly, one dual
  block per asset and one row over the coefficient-weighted sum. `ep_cvar_formulation` reads a
  `mixed` flag in place of `single`, and `ep_evar_formulation` and `ep_rlvar_formulation` take the
  same flag. No group view needs an integer variable, and no measure refuses one.
- A view whose coefficients carry both signs, `A - B >= c`, is a difference of concave functions,
  and its feasible set is not convex. No convex program describes it exactly. It takes
  `IntegerConditionalValueatRiskView` by default for CVaR, as [EPTail](@cite) writes it, and the
  sequential formulation below for EVaR and RLVaR, which have no integer formulation.

### The sequential formulation is the third formulation of every measure

`SequentialConditionalValueatRiskView`, `SequentialEntropicValueatRiskView` and
`SequentialRelativisticValueatRiskView` write every view the dual formulation cannot, with no
integer variable. The view is oriented as a lower bound. Each asset with a positive coefficient
keeps its exact dual block. Each asset with a negative coefficient takes a **Surrogate Row**, a
linear upper bound on its measure read from the primal representation at fixed multipliers: the
value at risk for CVaR, the primal pair for RLVaR, and the tangent of the fixed-dual-variable
primal for EVaR, whose fixed form is concave rather than linear. The row is sufficient for the
view, and tight at the posterior it was read at.

`entropy_pooling` re-reads the row at each posterior and solves again, up to `iters` times or
until the row is tight to `tol`. The last posterior stays feasible for the re-read row, so the
divergence never rises between solves, and at the fixed point the view holds exactly. The answer
is a local minimiser of the divergence, which is why the sequential formulation is the default
only where no exact one exists. An equality is written as the bound the prior violates, and the
entropy minimiser makes it tight, as the dual formulations already do.

A linear bound read at one point has a floor, and for the relativistic measure the floor sits
close to the prior, so a target well below it leaves the first solve with no feasible point.
`ep_sequential_start` walks the multipliers toward the target with a chain of exponential tilts
before the first solve, as the grid anchors do, so the first row can meet the view.

The carriers under `AbstractSequentialTailViewConstraint` hold the two sides and the row.
`ep_tail_dual_block!` writes one asset's dual block for the dual carrier and the sequential one of
a measure alike, so each block is written once. `ep_jump_entropy_pooling` is the one solve, and
`entropy_pooling` is the loop around it, so the three `ep_prior` stages refine within each stage.

## Amendment (2026-09-23)

Issue #1260 found that `IntegerConditionalValueatRiskView` solved a restriction of the view, not the
view. The formulation of [EPTail](@cite) pins `q_j = w_[j] y_j` with the row
`q_j >= w_[j] - (1 - y_j)` for every observation of the window, so every marked observation enters
the tail in full, and the tail must be whole observations of mass exactly `alpha`. The CVaR of a
discrete law lets its value at risk observation enter in part. The census of #1254 measured the
integer posterior at 8.4% above the least divergence on an upper bound, and at 107% above it on a
relative view.

### An observation enters in full only when the one below it is marked

The row now reads the indicator of the observation below:

```julia
[j = 2:sb], sc1 * (pw[ordi[j]] - (one(alpha) - y[j - 1]) - q[j]) <= 0
```

The lowest marked observation keeps `0 <= q_j <= w_[j]` alone, and `sum(q) == alpha` fixes it at
`alpha` less the mass above it. That is the value at risk observation, so the model reads the
posterior CVaR exactly. The change adds no variable and removes one row per asset. The feasible set
only grows, so no posterior that was feasible becomes infeasible, and a released posterior can only
move to a smaller divergence. This departs from the reference on purpose. The ascending order and
the monotonicity of the section above are unchanged.

### The window stays, and it warns where it binds

`sbar` keeps the rule of thumb of the reference. The window is the one restriction left: the model
admits the posteriors that put at least `alpha` on the `sbar` largest losses. An upper-bound view
moves mass down the order and meets that bound first. After the last solve `entropy_pooling` calls
`ep_check_tail_window`, which warns where the window holds no more than `alpha` plus `alpha` times
the cube root of the machine epsilon. On the census fixture the binding window held `alpha` to
`7e-12`, and the five windows that did not bind held `0.011` or more above it. `sbar = T` restricts
nothing. On that fixture it reached the least divergence of the census to `3e-8` on all four views,
in 2 to 4 seconds each.

## Amendment (2026-09-23) — from [#1264](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1264)

Issue #1264 found that `GridEntropicValueatRiskView` missed an upper bound by 60%, and that
nothing reported it. The row of a grid point was divided by its largest coefficient, so the
coefficients sat in `(0, 1]`, and the bound was `alpha / sc`. At a small dual variable `sc` is of
the order `1e10`, and the bound fell to `1.5e-11`. The whole row was then inside the feasibility
tolerance of the conic solver, about `1e-7`. The solver took that row as met by a posterior near
the prior, and the selector picked the grid point whose row it could ignore. The anchor was not
at fault: it converged to the dual variable of the posterior of least divergence.

### Every upper-bound row reads against one

The row of grid point `k` reaches the model divided by its bound `b_k`:

```julia
c = c ./ b
Mk = M * (maximum(c) - one(b))
sc1 * (dot(c, pw) - one(b) - Mk * (one(b) - y[k])) <= 0
```

The solver then meets the row to its tolerance relative to the bound, not relative to the largest
coefficient. The posterior sums to one, so the left hand side never exceeds `maximum(c)`, and
`maximum(c) - 1` is the smallest constant that releases the row. The data fix it, row by row.

On the fixture of the census of #1254, an upper bound at half the prior EVaR, the posterior met the
view to `1.2e-10` at the divergence of a scan over the dual variable of one-row tilts, `0.08822`.
The solve took 2.6 s. Two floors on the scaled bound were measured and refused. A floor at `1e-7`
still missed the view by 15%. A floor at `cbrt(eps)` met it, but it dropped the grid point of least
divergence and raised the divergence by 2.5%. The relativistic grid met its view on that fixture
before the change, with its smallest bound at `1.1e-7`. It takes the same rows, because its rows are
scaled the same way.

The largest constant on that fixture is `6.8e10`. Pajarito fixes the binary vector before its last
conic solve, so the tolerance on integrality does not reach the selected row there. A solver that
does not fix it can open a slack of the constant times that tolerance on the selected row.

### `M` is a multiplier

`M` stays on `GridEntropicValueatRiskView` and `GridRelativisticValueatRiskView`, and it now
multiplies the constant of each row. Its default moves from `10` to `1`, and its domain from
`M > 0` to `M >= 1`, because a multiplier below one cuts off a posterior the view admits. A
multiplier above one gives the released rows headroom for a posterior that sums to one only to the
solver's tolerance. It also widens the slack the tolerance on integrality can open on the selected
row. A caller who passed an `M` below one now meets a `DomainError`.

### An upper-bound grid point needs a positive bound

The upper-bound half keeps a point only when its row is finite and its bound is positive. The
coefficients are positive and the posterior sums to one, so an upper-bound row with a bound at or
below zero holds at no posterior. The bound of an entropic row underflows to zero where its
coefficients overflow, and the bound of a relativistic row is at or below zero where the target
lies below what the point can reach. The error of an upper-bound half that keeps no point names both
causes. Before this change the solver reported that case as infeasible, so the error comes earlier
and names the cause, and it is not a new refusal.

The lower-bound half keeps every finite point. A lower-bound row with a bound at or below zero holds
at every posterior, so it changes no answer. A filter there would only add a refusal where every
point is like that, and the grid is centred on the point the posterior attains, so such a grid
states a view the prior already meets.

### The lower-bound half keeps the norm scale

The rows of the lower-bound half go through `add_ep_constraint!`, which divides a row by its norm.
They are not divided by their bound, because they do not show the defect. A measurement over 13
settings covered the fixture of #1254, grids widened to `pct = 0.97` and `K = 41`, and the fixtures of
`test_12h`, with bounds down to `5.1e-11`. No row with a small bound was violated, and those rows were
slack at the posterior by factors up to `9.6e7`. Every negative residual sat at the one binding row,
whose bound was `6e-3` or more, and it was `1.7e-8` of that bound or less. A posterior that meets a
lower bound moves mass toward the largest loss, where every row takes its largest coefficient, so a
row that binds has a bound of at least the prior probability of that loss.

Division by the bound gave the same posterior where it solved, and it did harm elsewhere. The dual
of `OptimEntropyPooling` failed with `Inf` or `NaN` in 7 of the 13 settings, and took 8 to 25 times
more function evaluations in the others. Clarabel stopped at `SLOW_PROGRESS` where a bound was near
`1e-10`, which includes the fine grid of the existing test.

A check after the solve, that the posterior statistic meets the target, was considered and refused.
It would set a tolerance on a residual, which this library does not set (#573).

## Amendment (2026-09-23) — from [#1287](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1287)

Issue #1287 found that an upper bound on a group failed on the EVaR and the RLVaR when `alg` was
left at `nothing`. A group is a view over several assets whose coefficients all have one sign. The
selectors sent every upper bound whose coefficients had one sign to the grid, and so every equality
below the prior value. The grid holds the measure of one asset, and it refused the view with an
`ArgumentError`.

### The selector reads the number of assets

`ep_evar_formulation` and `ep_rlvar_formulation` take a new argument `single`, which says whether
the view names one asset. A lower bound, and an equality at or above the prior value, keep the conic
formulation at any number of assets. The grid keeps the upper bound and the low equality of one
asset. The same views over several assets take the sequential formulation, which already took every
relative view:

```julia
function ep_evar_formulation(::Nothing, mixed::Bool, single::Bool, op::Symbol, rhs::Number,
                             pv::Number)
    return if mixed
        SequentialEntropicValueatRiskView()
    elseif op == :geq || op == :eq && rhs >= pv
        ConicEntropicValueatRiskView()
    elseif single
        GridEntropicValueatRiskView()
    else
        SequentialEntropicValueatRiskView()
    end
end
```

`ep_sequential_sides` negates an upper bound, so every asset of the group goes to the primal side,
and the view is met with no integer variable. On the three technology assets of the S&P 500 slice
of the examples, an upper bound at 95% of the prior value and an equality at 97% of it met their
targets to a relative `8e-9` on both measures.

`ep_cvar_formulation` does not change. It sends the same views to
`IntegerConditionalValueatRiskView`, which keeps one window per asset and so accepts a group.

The error of the grid named the conic formulation for any positive combination, but the conic
formulation refuses an upper bound. It now names the conic formulation for a lower bound on a
positive combination, and the sequential formulation for an upper bound or a relative view. The
conic errors say that the grid holds a single asset.
