---
status: accepted
---

# A drawdown tail measure is its returns twin on another series

## Context

Six conic tail measures ship as a pair: a **returns** twin that reduces the net portfolio
returns, and a **drawdown** twin that reduces the drawdown path.

| family       | returns twin                                   | drawdown twin                                     |
| ------------ | ---------------------------------------------- | ------------------------------------------------- |
| big-M        | `ValueatRisk` under `MIPValueatRisk`           | `DrawdownatRisk`                                  |
| conditional  | `ConditionalValueatRisk`                       | `ConditionalDrawdownatRisk`                       |
| DR           | `DistributionallyRobustConditionalValueatRisk` | `DistributionallyRobustConditionalDrawdownatRisk` |
| entropic     | `EntropicValueatRisk`                          | `EntropicDrawdownatRisk`                          |
| relativistic | `RelativisticValueatRisk`                      | `RelativisticDrawdownatRisk`                      |
| power-norm   | `PowerNormValueatRisk`                         | `PowerNormDrawdownatRisk`                         |

Each pair is **one** conic programme. The only semantic difference is the series the
programme reduces, under the substitution `net_X -> -dd[2:T+1]`.

Every pair wrote that programme twice, and the substitution was spelled by hand in five
different ways, so each of the six cones lived at two sites:

```julia
# EVaR                                    # EDaR
[sc * (-net_X[i] - t_evar), …]            [sc * (dd[i + 1] - t_edar), …]
# CVaR                                    # CDaR
sc * ((z_cvar + net_X) .+ var) >= 0       sc * ((z_cdar - view(dd, 2:(T + 1))) .+ dar) >= 0
# RLVaR                                   # RLDaR
sc * ((eps + omega - net_X) .- t) <= 0    sc * ((eps + omega + view(dd, 2:(T + 1))) .- t) <= 0
```

Every sign was correct, by hand, at every site. Nothing in the code said the two sides were
one programme, so a fix to any cone landed at two places and a seventh measure would inherit
whichever copy its author read.

The copies had already drifted in ways that had nothing to do with the cone:

- **Observation weights.** The returns twins resolved `get_observation_weights(wi, net_X)`.
    `net_X` is a vector of JuMP expressions, which matches neither documented arity, so a
    `DynamicAbstractWeights` **always** raised there — while the drawdown twins passed
    `pr.X` and resolved. One substitution, two answers.
- **Dead stores.** `08:164` computed `at = r.alpha * T` and `07:325` computed
    `iat = inv(r.alpha * T)`, both immediately overwritten by the branch below them.
- **The `loss::Bool` seam.** Present on all six returns twins, absent on all six drawdown
    twins, with nothing stating why.

## Decision

A tail measure **asks for the series it reduces, and its builder is written once**.

`risk_series(model, alg, pr; …) -> (series, T)` is the one place the substitution is made.
`alg` is a marker: `NetReturnsRiskSeries()` returns the net portfolio returns,
`DrawdownRiskSeries()` returns `-dd[2:T+1]`.

The series is signed as a **return** on both markers: a loss is a negative entry. That is why
the drawdown branch negates — `dd` is a non-negative loss path — and it is the whole of what
lets one builder body serve both twins.

Each family has one kernel — `set_mip_quantile_risk_constraints!`,
`set_conditional_risk_constraints!`, `set_dr_conditional_risk_constraints!`,
`set_entropic_risk_constraints!`, `set_relativistic_risk_constraints!`,
`set_power_norm_risk_constraints!` — taking `(series, T)` and a `NamedTuple` of the bare Model
State names it registers. The twelve `set_risk_constraints!` methods remain as the dispatch
surface and are each a call to `risk_series` plus a call to the kernel.

The names are passed as **literal symbols at the two call sites** rather than composed inside
the kernel from a tag. A key stays greppable in the file that owns it, which is what a reader
querying Model State needs, and it keeps the kernel out of the business rule 1 of ADR 0037
polices.

`risk_series` takes `loss::Bool` on `NetReturnsRiskSeries` **only**. A drawdown has no gain
tail — a run-up is a different recurrence, not this one negated — so a caller that tries to
range-compose a drawdown measure fails at the call site instead of silently building the loss
tail twice.

## Consequences

- Twelve builder bodies become six kernels plus twelve delegations.
    `20_RiskMeasureConstraints/` loses 239 lines of code, and six hand-signed substitutions
    become one.
- The models are **unchanged**. Thirty-six assembled models — all eighteen affected measures,
    weighted and unweighted — are byte-identical to the ones the previous code built,
    including their Model State key sets.
- A `DynamicAbstractWeights` now resolves for the returns twins, against `pr.X`, the same as
    for the drawdown twins. This is the one behaviour change. A type that cannot resolve the
    matrix arity still raises rather than going quietly unweighted — now on both sides rather
    than only one.
- A fix to any of the six cones lands once. A new tail measure declares which series it
    reduces rather than restating a cone.
- The drawdown twins become **range-composable in principle**: a drawdown series is now
    obtainable through the same seam the range builder uses. Nothing composes one yet, and
    `risk_series` refuses `loss = false` on that marker until the gain-side recurrence is
    defined.

## Scope

Measures with **no** twin keep their own builder and are untouched: `ValueatRisk` under
`DistributionValueatRisk` (closed-form, no series), `AverageDrawdown`, `UlcerIndex`,
`MaximumDrawdown`, and the moment, OWA, turnover and tracking families.

The remaining `get_observation_weights(wi, net_X)` sites in
`03_MomentRiskMeasureConstraints.jl` carry the same latent defect and are **not** in this
change: they have no drawdown twin, so nothing there is a re-encoding, and correcting them is
a separate decision about the moment families.

## Notes

The MIP `ValueatRisk` gain tail's pre-existing degeneracy (ADR 0057 §Notes) is preserved
exactly: `set_mip_quantile_risk_constraints!` negates the binary indicators with the series
under `loss = false`, as the hand-written body did. Correcting it is still a separate change.

## Related

- [0037](0037-model-state-accessor-interface.md) — the Model State keys the kernels register.
- [0043](0043-nothing-observation-weights-means-unweighted-not-unavailable.md) — why an
    unresolvable `DynamicAbstractWeights` raises rather than resolving to `nothing`.
- [0057](0057-a-range-risk-measure-is-its-base-measure-applied-twice.md) — the same move on
    the **tail** axis; this one is the **series** axis.
