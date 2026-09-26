---
status: accepted
---

# A relaxed factor budget prices the whole portfolio

## Context

`RelaxedRiskBudgeting` is the relaxation of Gambeta and Kwon: it minimises `psi - gamma`, where
the rotated cones state `gamma^2 b_i ≤ x_i zeta_i` with `zeta` the marginal risks, and a cone
states `psi ≥ ‖G w‖`. The chain `gamma^2 ≤ sum_i x_i zeta_i = w' Σ w ≤ psi^2`, for a budget that
sums to one, holds the objective at or above zero, and the objective is zero exactly at the risk
budgeting portfolio.

With a `FactorRiskBudgeting` the budgeted weights are the factor weights `w1`. Under
`flag = false` the asset weights are `w = b1 w1`, and the relaxation runs on `w1` with the
covariance `b1' Σ b1`. Under `flag = true` they are `w = b1 w1 + b2 w2`, where the columns of `b2`
are an orthonormal basis of the null space of `B'` (ADR 0182). The builder kept the `flag = false`
model: `zeta = b1' Σ b1 w1` and `psi ≥ ‖chol(b1' Σ b1) w1‖`. So it priced and budgeted the factor
part `b1 w1` alone, and `w2` was limited only by the weight constraints (#1351). On the SP500
fixture, with an empirical prior and no weight bounds, the standard deviation of the returned weights was 1.88 while `psi`
was 0.219.

With the whole portfolio priced, the variance splits into the factor contributions and the
off-factor contribution `c2 = w2' b2' Σ w`, the `RC(ỹ_af)` of the Euler decomposition of risk
factors (Roncalli and Weisang). `c2` has no sign, and the chain becomes
`gamma^2 ≤ w' Σ w - c2`. No second-order cone restores it with `w2` free: a bound on `-c2` that is
tight where `b2' Σ w = 0` needs the product `‖K w1‖ ‖w2 - w2*‖`, which is not convex.

The log-barrier `RiskBudgeting` under `Variance` has the first-order condition `b2' Σ w = 0` in
`w2` when no other constraint binds. With these rows, `w = M w1` with
`M = b1 - b2 (b2' Σ b2)^{-1} b2' Σ b1`, `b1' Σ w = M' Σ M w1`, and
`M' Σ M = (B' Σ^{-1} B)^{-1}`. So the relaxation with the rows is the relaxation of Gambeta and
Kwon on that factor covariance, and it is exact.

## Decision

1. **The relaxed risk reads the whole portfolio under `flag = true`.** `zeta = b1' Σ w`, so
   `w1_j zeta_j` is the Euler contribution of factor `j` that `factor_risk_contribution` reports,
   and the cones of `alg` read `w` and `Σ`, so `psi` bounds the standard deviation of the returned
   weights. Under `flag = false` nothing changes.
2. **The penalty of `RegularisedPenalisedRelaxedRiskBudgeting` reads the decision vector.** It is
   `z = [w1; w2]` under `flag = true`, with the covariance `P' Σ P` of the basis `P = [b1 b2]`, as
   ADR 0182 lifts the decision vector. A penalty on `w1` alone moved 97 % of the variance into the
   unpenalised off-factor weights at `p = 50`.
3. **`FactorRiskBudgeting` takes a field `hedge::Bool = false`.** With `hedge = true` and
   `flag = true`, `set_factor_risk_contribution_constraints!` adds the rows `b2' Σ w = 0`. It serves
   `RiskBudgeting` and `RelaxedRiskBudgeting` alike, since it states how the off-factor weights are
   set, not how the risk is relaxed. Under `flag = false` it is ignored.
4. **`hedge = false` is the default.** The off-factor weights stay free, so every model that was
   feasible stays feasible.

## Why not one formulation

| | `hedge = false` | `hedge = true` |
| --- | --- | --- |
| Feasible under the bounds `flag = true` meets | yes | no: the weights lie in a subspace of dimension `N_f`, and the default long-only bounds are infeasible on the SP500 fixture |
| Distance from the `RiskBudgeting` optimum under `Variance`, no bounds | 24 % of the weights | 0 |
| Off-factor contribution | free, −0.9 % of the variance on the fixture | zero |
| Factor contributions relative to each other | exact when no bound binds | exact when no bound binds |

Neither dominates, so the maintainer chose both behind a switch. A field on
`RelaxedRiskBudgeting` alone was refused, because the rows are a property of the factor basis and
`RiskBudgeting` reads them the same way. Tag types were refused, because a two-state choice
beside the `Bool` `flag` does not need new exports.

## Consequences

- The stored weights of the two factor tests of `test/test_15_relaxed_risk_budgetting_optimisation.jl`
  change, and were regenerated. The loop of those tests now passes its variant `alg`, which it
  omitted, so each column had held the basic variant.
- `test/test_12i_cross_sectional_factor_carrier.jl` unpacked two values from
  `set_factor_risk_contribution_constraints!`, which returns three since ADR 0182. It unpacks three.

Issue #1351.
