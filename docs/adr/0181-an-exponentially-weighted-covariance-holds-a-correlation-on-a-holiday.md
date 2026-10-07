---
status: accepted
---

# An exponentially weighted covariance holds a correlation on a holiday

## Context

`ExpWeightedCovariance` ports the exponentially weighted covariance of the
oracle. An asset is on a holiday at an observation where the active mask admits it and its
return is not finite. The port and the oracle updated only the block of the valid assets at each
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
3. **The departure from the oracle is limited to holidays.** Without a holiday `D = sqrt(λ) I`
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
  with the oracle on a holiday, but the oracle is wrong there: its own docstring claims the
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

Decisions 1, 2 and 4 are replaced. The congruence of decision 1 buys a positive semidefinite
estimate with a bias: every pair whose two assets do not share one history holds less weight than
the product of its two diagonal weights, by the Cauchy-Schwarz inequality, and the congruence
divides it by that product. So each correlation that meets a holiday or a late listing shrinks
towards zero. The last consequence claimed that a sample without a holiday keeps its numbers,
because "a common count then corrects every pair by the same scalar". That is false for a late
listing: its pairs hold `1 - λ^n_j`, and the congruence divided them by
`sqrt((1 - λ^n_i)(1 - λ^n_j))`. With a correlation half-life of 20 and the default `min_obs` of
40, a late asset entered the report with each correlation shrunk by 13% (#1383).

### The measure

A simulation compared five rules on ten assets with a true correlation of 0.5 unless stated, 400
observations, a half-life of 40 and 1000 draws (#1383, #1420). **O** steps only the block of the
valid assets and reads out through the per-asset congruence, then repairs a matrix that is not
positive definite. **A** is decision 1. **P** is A's step with each pair divided by the weight it
holds. **Q** is O's step with each pair divided by the weight it holds, then a repair where the
matrix is not positive semidefinite. The hand-written A equals `ExpWeightedCovariance` before this
amendment to 3.1e-15. The matrix error is `‖Σ̂ - Σ‖ / ‖Σ‖`; the portfolio error is the mean
`|w'Σ̂w / w'Σw - 1|` over 200 random portfolios; the bias is the mean error of the off-diagonal
correlations.

| Scenario | Matrix error Q / O / A / P | Bias Q / O / A / P | Portfolio error Q / O / A / P |
| --- | --- | --- | --- |
| iid holidays 10% | 0.177 / 0.177 / 0.190 / 0.185 | -0.004 / -0.004 / -0.054 / -0.004 | 0.109 / 0.109 / 0.116 / 0.113 |
| 30 assets, correlation 0.9, half-life 20, iid 10% | 0.172 / 0.172 / 0.196 / 0.178 | -0.013 / -0.013 / -0.092 / -0.016 | 0.193 / 0.193 / 0.324 / 0.195 |
| three assets listed 40 observations ago | 0.243 / 0.262 / 0.262 / 0.243 | -0.010 / -0.135 / -0.135 / -0.010 | 0.146 / 0.161 / 0.161 / 0.146 |
| the same, iid holidays 5% | 0.250 / 0.275 / 0.277 / 0.252 | -0.010 / -0.153 / -0.158 / -0.010 | 0.153 / 0.170 / 0.171 / 0.153 |

The shrink of A grows with the rate of holidays: about 0.5% of a correlation of 0.5 per 1% of
holidays. Where the matrix is near singular, O and Q without their repair were not positive
semidefinite in 81% to 100% of the draws, with a smallest correlation eigenvalue of -1.64, as
decision 1 warned. The repair removes that failure at no measurable cost, and after it Q is the
best rule or equal to the best on every measure. A had the smaller matrix error in one scenario,
a volatility shift that coincides with the holidays (0.484 against 0.534). That is an offset of
two errors, not a merit: A's correlation and portfolio errors are the larger ones in the same
runs.

### The decision

Each pair ages on the clock of its own common observations, and each entry is divided by the
weight that its pair holds. On all three paths, `ExpWeightedCovariance`,
`RegimeAdjustedExpWeightedCovariance` with one decay, and its separate path at `cor_decay`:

```text
S[V, V] <- λ S[V, V] + (1 - λ) e_V e_V'      every other entry holds
W[V, V] <- λ W[V, V] + (1 - λ)
Σ[i, j]  = S[i, j] / W[i, j]
```

- `W[i, j]` is `1 - λ^n_ij`, with `n_ij` the count of the common valid observations of the pair,
  so each entry is the exponentially weighted mean of the product over those observations. On the
  diagonal it is the per-asset correction of decision 1, so no variance moves.
- A holiday holds every entry of its asset and its weight: the observation carries no information
  about them. The covariance of the pair holds, and its correlation moves only with the variance
  of the other asset, which has a new observation.
- Where every pair shares one history, `W[i, j] = sqrt(W[i, i] W[j, j])`, the division is the
  congruence of decision 1, and the estimate is positive semidefinite by construction. Otherwise
  the report clips the negative eigenvalues of its correlation to zero, restores the unit
  diagonal and keeps the variances, where the smallest eigenvalue is below `-n eps` times the
  largest. The repair reads the block of the assets that the report returns, so an asset in its
  warm-up moves nothing.
- The block that the regime statistic reads takes the division and no repair, because the
  Cholesky factor of the Mahalanobis target refuses a block that is not positive definite. A
  `variance_series` reads the diagonal, which the repair keeps, so it skips the repair.

The considered option "Repair at read-out" was rejected because it hid an error that grew without
bound. Under this decision the division removes the bias, the error of each entry does not grow,
and the repair binds only where the entries read different observations. The oracle of map #1375
takes the same step, so the raw state equals its raw state; its read-out divides by the per-asset
congruence, so it shrinks a late listing as decision 1 did, and this rule is Better there.

## Amendment (2026-10-06)

The step of the amendment of 2026-09-29 takes `e` as the deviation of a return. #1507 replaced
the Bool `centred` with the `centring` field, and its default `EstimatedCentring()` takes each
deviation from the normalised location of the returns of its own asset before it (ADR 0190). That
location is an estimate, so for returns independent in time the product of a pair has the mean
`σ_ij (1 + c_ij)`, with `c_ij` the sum, over the common valid returns of the two locations, of the
products of their normalised weights. The step divides each product by that factor:

```text
S[D, D] <- λ S[D, D] + (1 - λ) (e_D e_D') ./ (1 + C[D, D])      every other entry holds
W[D, D] <- λ W[D, D] + (1 - λ)
Σ[i, j]  = S[i, j] / W[i, j]
```

- `D` is the set of the valid assets that give a deviation. Under `EstimatedCentring` an asset
  gives one from its second valid return, because the first has no location before it. Under
  `PreCentred()` `D` is the valid set, `C` is zero, and the step is the step above.
- `C[i, j] = P[i, j] / (S1_i S1_j)`, with `S1_i = 1 - λ^n_i` the sum of the weights of the
  location of asset `i`, and `P` the overlap that the state carries, `P <- λ^(v_i + v_j) P +
  v_i v_j (1 - λ)²` on the valid indicators `v`. It reads the common returns of the pair, so rule
  Q holds: each pair still ages on its own common observations and is divided by its own weight.
- Where every asset shares one history, `C` is one scalar on every entry, so the step is a scaled
  outer product and the estimate stays positive semidefinite by construction. Otherwise the repair
  of the amendment above binds as before.
- A simulation on the holiday fixture of `test_08z` confirmed the factor against the Monte Carlo
  mean of each product before the build: the z-scores had an RMS of 0.97 over 245 cells.
