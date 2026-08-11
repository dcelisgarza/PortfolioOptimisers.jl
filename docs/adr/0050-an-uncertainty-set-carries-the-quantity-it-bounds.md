---
status: accepted
---

# An uncertainty set carries the quantity it bounds

## Context

An uncertainty set is a **neighbourhood of a specific quantity**.
[`L1UncertaintySet`](../../src/14_UncertaintySets/05_L1UncertaintySets.jl) says so in its own
docstring: the set is `{mu_hat + e : ‖e ⊘ sd‖₁ <= eps}`. A radius without its centre is not that
object. The same holds of every other shape in the family — a box is `mu_hat ± delta`, an
ellipsoid is `(mu - mu_hat)' Σ⁻¹ (mu - mu_hat) <= k²`.

The library calibrated the radius on one fit and applied it to a different one. **Both axes had
it.**

On the mean axis,
[`set_return_constraints!`](../../src/20_Optimisation/09_JuMPConstraints/02_Returns_and_ObjectiveFunctions.jl)
took the centre from the outer prior:

```julia
mu = ifelse(isnothing(pret.mu), pr.mu, pret.mu)          # the OUTER prior
set_ucs_return_constraints!(model, mu_ucs(ucs, rd; kwargs...), mu)
```

`mu_ucs` fits `ue.pe`, derives `eps` and `sd` from **that** prior, and then discards its `mu`. The
two fits are independent, and nothing refused a configuration in which they named different
quantities.

On the covariance axis,
[`set_risk_constraints!`](../../src/20_Optimisation/20_RiskMeasureConstraints/02_VarianceConstraints.jl)
had the identical shape: it bounded `pr.sigma` with an ellipsoid whose shape matrix came from
`ue.pe`.

The consequence is visible in the shipped documentation. Example 11
(`examples/2_moments_priors/11_L1_Uncertainty_Quintile_Portfolios.jl`) puts the characteristic in
the **outer** prior's `me`. That forces the two fits to agree by hand, and it re-centres every
other consumer of `pr.mu` as a side effect — the moment risk measures, the XatRisk measures,
kurtosis, skew, and the ratio objective's scale factor.

## Decision

**An uncertainty set carries the quantity it was calibrated on, and that quantity wins.**

Each Result gains one optional field holding the quantity it is a neighbourhood of.

| Result                      | Field | Holds                                                      |
| --------------------------- | ----- | ---------------------------------------------------------- |
| `L1UncertaintySet`          | `mu`  | the characteristic vector                                  |
| `SignedL1UncertaintySet`    | `mu`  | the characteristic vector                                  |
| `BoxUncertaintySet`         | `val` | a vector on the mean axis, a matrix on the covariance axis |
| `EllipsoidalUncertaintySet` | `val` | a vector on the mean axis, a matrix on the covariance axis |

The ℓ1 family is mean-only — its `ucs` and `sigma_ucs` throw — so `mu` is precise there. The other
two serve **both** axes, so `val` is the neutral noun.

**Precedence is the carried quantity, then the estimator's own field, then the prior.** The field
is optional, so a hand-built set that names nothing keeps the previous behaviour, and nothing
breaks at construction.

**The read happens inside the dispatching method.** `set_ucs_return_constraints!` already has one
method per set type, so each reads its own field name and no `isa` chain appears anywhere. The
method **returns** the resolved quantity, because the caller needs it: `MaximumRatio` reads `mu` as
a value through `set_max_ratio_return_constraints!`. Returning a value from a bang-function is this
path's own idiom — `set_ucs_variance_risk!` already returns `(ucs_variance_risk, key)`.

```julia
function set_ucs_return_constraints!(model, ucs::L1UncertaintySet, mu)
    mu = something(ucs.mu, mu)      # its own field, no branch
    ...
    return mu
end
```

### The box covariance route names no centre, and that is correct

The worst-case variance over a box is `tr(A_u Σ_u) - tr(A_l Σ_l)`. It is built from the bounds
alone, so `set_ucs_variance_risk!(::BoxUncertaintySet, …)` ignores both the carried value and the
fallback. The field is populated on that route all the same, because the set is still a
neighbourhood of the fit it came from and a later consumer may need to know of what.

### The scalar twin resolves the same way

`ucs_variance` is the value-level evaluation of the same worst case, and
`UncertaintySetVariance`'s functor dispatches to it. It resolves `val` identically, so a scalar
risk evaluation and the JuMP expression cannot disagree about which covariance they are centred
on.

### A view carries the field

`port_opt_view` slices the carried quantity with the asset index. On the covariance ellipsoid the
index mapping differs from the shape matrix's: `val` is the `N × N` covariance and takes the asset
index, whereas the `N² × N²` shape matrix takes the fourth-moment index. The ℓ1 sets gain their
first `port_opt_view` methods, which slice `sd` as well — until now they reached the generic
fallback and raised a `MethodError`.

## Rejected: fit once at the call site

The call site could fit `ue.pe` once and use that prior for both the radius and the centre. It
fixes the estimator route and **cannot survive a pre-built set** crossing the boundary: by then
the fit is gone and there is nothing left to read. `ArithmeticReturn.ucs` and
`UncertaintySetVariance.ucs` both accept a pre-built set, and the documented route in example 11
uses one. That is the whole argument for the field.

## Consequences

- **This changes results.** Any configuration whose `ue.pe` differs from `opt.pe` produces
  different numbers. Construction does not break; behaviour does.
- **Example 11 is untouched.** It never passes a `CharacteristicUncertaintySet` *estimator* to
  `ArithmeticReturn.ucs`. Every set it optimises with is hand-built with a bare `eps`, so all of
  them carry no `mu`, land on the fallback, and behave exactly as before. Its outer-prior hijack
  still works.
- **The glossary changes.** `CONTEXT.md` said the Prior gives the central estimate and the
  Uncertainty Set only bounds it. On a set-bearing route the set gives the central estimate and
  the Prior is the fallback. Rewritten in the same change.
- This is a fix to the **single**-characteristic case. It is a prerequisite for multiplicity
  ([#265](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/265)), not part of it.
