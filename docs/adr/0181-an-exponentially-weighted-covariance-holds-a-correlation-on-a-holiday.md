---
status: accepted
---

# An exponentially weighted covariance holds a correlation on a holiday

## Context

`ExpWeightedCovariance` ports the exponentially weighted covariance of a reference
implementation. An asset is on a holiday at an observation where the active mask admits it and its
return is not finite. The port and the reference updated only the block of the valid assets at each
observation:

```text
S[V, V] <- λ S[V, V] + (1 - λ) e e'
```

and left every other entry unchanged. On a holiday of asset `j`, the variance of a valid asset `i`
decays by `λ` and the covariance `S[i, j]` keeps its value. Both claimed that the state stays
positive semidefinite at every step. The claim is false (#1343).

The failure is not a rounding effect. The step is the Hadamard product `M ∘ S` plus a positive
semidefinite term, where `M[i, j]` is `λ` when both assets are valid and one otherwise. The mask `M`
holds the principal minor `[λ 1; 1 1]`, whose determinant `λ - 1` is negative, so the Schur
product theorem does not apply. With `S = 1 1'` and `k` holidays of `j` at which `i` has a zero
deviation, the determinant of the pair is `λ^k - 1 < 0`, and the implied correlation is
`λ^(-k/2)`, without bound. In `Rational{BigInt}` arithmetic, two equal series with five holidays
of one of them give the determinant `-9.18e-7` and the correlation `5.95`. Over 2000 random panels
with holidays, the smallest eigenvalue of the state reached `-1.15` times its largest entry.

This is the exponentially weighted form of pairwise deletion: each pair ages its observations on
the clock of its common observations. Pairwise deletion is not positive semidefinite in general.

## Decision

1. **Each asset ages its observations on its own clock, and a pair weights an observation by the
   geometric mean of the two weights.** At each observation the recursion is

   ```text
   S <- D S D + (1 - λ) e e',   D = diag(sqrt(λ) for a valid asset, 1 for any other)
   ```

   with `e` zero outside the valid assets. In closed form,
   `S[i, j] = (1 - λ) Σ_t λ^((c_i(t) + c_j(t)) / 2) e[t, i] e[t, j]`, where `c_i(t)` counts the
   valid observations of asset `i` after `t`. With `w[t, i] = λ^(c_i(t) / 2)`, `S` is the sum of
   the outer products of `w_t ∘ e_t`, so it is positive semidefinite for every pattern of
   holidays, and every implied correlation lies in `[-1, 1]` by the Cauchy-Schwarz inequality.
2. **A holiday holds the correlation of each pair that contains its asset.** Before the new
   product, the step scales `S[i, j]` by `sqrt(λ)`, `S[i, i]` by `λ` and `S[j, j]` by one, and
   the ratio `S[i, j] / sqrt(S[i, i] S[j, j])` does not move. A holiday carries no information
   about the co-movement of its asset. The old rule held the covariance instead, so it moved the
   correlation by `λ^(-1/2)` at each step with no data to justify the move.
3. **The departure from the reference is limited to holidays.** Without a holiday `D = sqrt(λ) I`
   and the step is `λ S + (1 - λ) e e'`, the old recursion. A diagonal entry, the location, the
   count, the reset on an inactive period and the read-out are unchanged. The parity constants of
   `test/test_08z_exp_weighted_moments.jl` have no holiday, and they pass unchanged.
4. **`RegimeAdjustedExpWeightedCovariance` takes the same step on both of its paths.** The path
   with one decay takes it on the covariance state (#1343). The path with a separate `cor_decay`
   takes it on the correlation state, with `sqrt(cor_decay)` in `D` (#1346). That path used to
   update the correlation state on the valid pairs alone, and to divide each pair by its own
   count of common observations at read-out. Neither operation is a congruence, and 200 random
   panels with holidays gave a smallest eigenvalue of `-0.1075` times the largest entry. The
   correction for the zero seed is now the congruence
   `sqrt((1 - cor_decay^n_i)(1 - cor_decay^n_j))`, and it cancels in the normalisation to a unit
   diagonal, so the read-out normalises the state directly. The pair count then has no reader,
   and the state `RegimeAdjustedCovarianceState` drops its field `pair_obs_count`.

## Considered options

- **Keep the recursion and document the defect.** This was the state after #891. It keeps parity
  with the reference on a holiday, but the reference is wrong there: its own docstring claims the
  property that it breaks. A consumer that factors the matrix, such as a JuMP risk measure through a
  Cholesky factor or an SOC row, gets an indefinite input from a bare estimator.
- **Repair at read-out, for example with `posdef!`.** A projection moves every entry, including the
  entries of pairs that had no holiday, and it repairs a state whose error grows without bound
  with the length of a holiday. The repair hides the cause rather than removing it.
- **Another mask.** A mask `M` keeps every positive semidefinite state positive semidefinite
  exactly when `M` is itself positive semidefinite. The arithmetic mean of the two decays gives
  the minor `[λ (1+λ)/2; (1+λ)/2 1]`, whose determinant is negative, so the defect stays. A mask
  that keeps `λ` on a pair of valid assets, as the recursion without a holiday does, must give a
  holiday asset the same entry `a` with every valid asset, and `|a| <= sqrt(λ)`. The value
  `a = sqrt(λ)` holds the correlation. A smaller value shrinks the correlation to zero at each step
  of a holiday, which is a move that the data does not support either.

## Consequences

- The estimate of `ExpWeightedCovariance` is positive semidefinite for every sample, so a bare
  estimator needs no repair wrapper.
- `cor` reads a ratio that lies in `[-1, 1]`, so its clamp removes round-off only.
- The value of a pair whose two assets have different holidays changes. The value of every other
  entry is the same as before.
- The estimate of `RegimeAdjustedExpWeightedCovariance` is positive semidefinite on a holiday on
  both paths, where `hac_lags` is `nothing`. On its separate path, a sample with a holiday gives
  a new answer, and a sample without one gives the old answer, because a common count then
  corrects every pair by the same scalar.

## Amendment (2026-09-29)

Decision 4 and the last consequence claimed that a sample without a holiday gives the old answer
on the separate path, "because a common count then corrects every pair by the same scalar". That
is true only when every asset has the same count. An asset that lists late, or lists again after
an inactive period, has fewer observations than an asset that did not. Its pairs then hold the
weight `1 - cor_decay^n_j` in the correlation state, and the congruence of decision 4 divides them
by `sqrt((1 - cor_decay^n_i)(1 - cor_decay^n_j))`. So the normalisation shrank each correlation of
the late asset by `sqrt((1 - cor_decay^n_j) / (1 - cor_decay^n_i))`. With a correlation half-life
of 20 and the default `min_obs` of 40, the shrink is 13% when the asset first enters the report
(#1383).

The shrink is not a property of the model. It comes from the division of a sum over the common
observations of a pair by weight totals over other observations. A simulation measured it: six
assets with a true correlation of 0.6, two of them listed late, 1000 draws, a variance half-life of
40 and a correlation half-life of 20.

| Observations of the late assets | RMSE, congruence | RMSE, weight of the pair | Bias of the late pairs, congruence | Bias, weight of the pair |
| --- | --- | --- | --- | --- |
| 40 | 0.145 | 0.118 | -0.148 | -0.078 |
| 60 | 0.111 | 0.099 | -0.096 | -0.061 |
| 120 | 0.083 | 0.083 | -0.029 | -0.024 |

The separate path now keeps a weight state `W`, which takes the step of decision 1 on a unit outer
product: `W <- D W D + (1 - cor_decay) v v'`, with `v` one on a valid asset and zero on any other.
The report divides each pair by its own weight:

```text
rho[i, j] = (Q[i, j] / W[i, j]) / sqrt((Q[i, i] / W[i, i]) (Q[j, j] / W[j, j]))
```

- Without a holiday, `W[i, j]` is `1 - cor_decay^c` with `c` the count of the common observations
  of the pair. The report then equals the correction by pair count of the oracle of map #1375.
- On a holiday, `W` scales exactly as `Q` does, so decision 2 still holds: the correlation of each
  pair that contains the asset does not move.
- Where every asset has the same history, `W` is one scalar and the report is the old one.

A matrix of pairs, each divided by its own weight, need not be positive semidefinite. When `W` is
not one scalar and the smallest eigenvalue of the report is below `-n eps` times the largest, the
report clips the negative eigenvalues to zero and restores the unit diagonal. The considered
option "Repair at read-out" stays rejected for the holiday rule, where the error grew without
bound. Here the division removes a bias, and the repair binds only on a short history: in the
simulation above with a correlation half-life of 10, 19% of the draws at 10 observations, and none
from 20 observations. The block that the regime statistic reads at each observation takes the
division and no repair, because the Cholesky factor of the Mahalanobis target refuses a block that
is not positive definite.

The last consequence now reads: the estimate of `RegimeAdjustedExpWeightedCovariance` is positive
semidefinite on both paths where `hac_lags` is `nothing`, by construction on the path with one
decay, and through the repair on the separate path. On the separate path, a sample whose assets
share one history gives the old answer, and a late listing or a holiday gives a new one.
