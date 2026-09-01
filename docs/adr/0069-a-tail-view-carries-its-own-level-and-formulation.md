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
