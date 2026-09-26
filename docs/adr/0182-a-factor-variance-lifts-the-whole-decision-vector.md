---
status: accepted
---

# A factor variance lifts the whole decision vector

## Context

`FactorRiskContribution` solves for factor weights `w1` in place of asset weights. The asset
weights are `w = b1 w1` when `flag = false`, with `b1` the pseudoinverse of `B'`. When
`flag = true` they are `w = b1 w1 + b2 w2`, where the columns of `b2` are an orthonormal basis of the
null space of `B'` and `w2` holds the off-factor weights. This is the decomposition of the weights
into factor and additional-factor exposures of equation 13 of the semidefinite risk-parity paper
(`sdprp`), after Meucci (2007).

A `Variance` takes the semidefinite formulation over the factor weights: the lift `W_f` of `w1`,
the variance `tr(b1' Σ b1 W_f)`, and the factor rows on `diag(b1' Σ b1 W_f)`. That is formulation 16
of `sdprp`. Formulation 16 fixes `x = (B')^+ x_f`, which is the `flag = false` case. The paper does
not state the `flag = true` case.

Under `flag = true` the builder kept the lift of `w1` alone. The expression was then
`w1' b1' Σ b1 w1`, the variance of the factor part `b1 w1`. It omitted `(b2 w2)' Σ (b2 w2)` and the
cross term `2 (b1 w1)' Σ (b2 w2)` (#1350). On the SP500 fixture, the model value under
`MinimumRisk` was 43 % below the variance of the returned weights. The objective minimised the
wrong quantity, and a factor row stated a share of the variance of the factor part and not of the
portfolio. (`FactorRiskContribution` refuses `settings.ub` with a warning, so no bound was at
stake.)

The issue gave two repairs:

1. Lift the whole weight vector, and keep the factor rows on the factor lift, with a coupling
   between the two lifts.
2. Keep the factor lift for the rows, and state the objective and the bound on the exact variance
   through a second-order cone `‖G (b1 w1 + b2 w2)‖`.

## Decision

1. **The lift covers the whole decision vector.** The decision vector is `z = w1` when
   `flag = false`, and `z = [w1; w2]` when `flag = true`. `set_sdp_frc_constraints!` lifts `z` into
   `W_z ⪰ z z' / k`, and the builder of `FactorRiskContribution` passes the basis `P` of `z` to the
   risk builders: `P = b1`, or `P = [b1 b2]`. `set_factor_risk_contribution_constraints!` returns
   `b2`, `nothing` under `flag = false`.
2. **The variance is `tr(P' Σ P W_z)`.** At rank one it is `w' Σ w` under either value of `flag`.
   Under `flag = true`, `P` is square and invertible, so the problem is the asset formulation 9 of
   `sdprp` in the basis of the factors, with the same relaxation. Under `flag = false` nothing
   changes.
3. **A factor row reads `diag(b1' Σ P W_z)`, one entry for each factor.** At rank one entry `j` is
   `[w1]_j [b1' Σ w]_j`. That is the Euler contribution of factor `j` of equations 14 and 15 of
   `sdprp`, after Roncalli and Weisang (2012), and the value `factor_risk_contribution` reports. A
   row states it as a share of the whole variance. The off-factor weights take no row. Their
   contribution is the remainder.
4. **The factor phylogeny reads the leading `N_f × N_f` block of `W_z`.** A principal submatrix of
   a positive semidefinite matrix is positive semidefinite, so the block is a lift of `w1` by
   itself, and the rows mean what they meant before.

## Why not the other repairs

- **Two lifts with a coupling (option 1 of the issue).** A lift of `w` and a lift of `w1` are the
  same object under the invertible map `P`, since `W = P W_z P'`. Two lifts add a second PSD block
  and a linear coupling to state one fact. The lift of `z` states it once.
- **A second-order cone for the objective, with the factor lift for the rows (option 2).** The
  rows then state shares of the variance of the factor part `b1 w1`, which is not a quantity of the
  portfolio and not what `factor_risk_contribution` reports. The ratio of two traces of `W_f` is
  also no longer held down by the objective, since the objective no longer reads `W_f`. A row
  `diag(S W_f) ≤ b tr(S W_f)` can then be met by an inflated `W_f` far from rank one. For example,
  `W_f + α S^{-1}` moves every share towards `1 / N_f` as `α` grows, and no term of the model
  prices `α`. The rows then bind nothing.

## Consequences

- Under `flag = true` the semidefinite block is `(N + 1) × (N + 1)` in place of
  `(N_f + 1) × (N_f + 1)`. This is the size of the asset formulation of `MeanRisk`, and `flag = true`
  already has `N` free variables.
- Under `flag = true` and `MinimumRisk` without rows, the solution is the minimum variance of
  `MeanRisk`, since `P` reaches every weight vector.
- The stored weights of the two `FactorRiskContribution(; flag = true)` entries of
  `test/test_22a_mix_optimisers.jl` change, and were regenerated.
- `RelaxedRiskBudgeting` with a `FactorRiskBudgeting(; flag = true)` prices the factor part alone
  in the same way. Its relaxation needs its own derivation, so it is a separate issue, #1351.

Issue #1350.
