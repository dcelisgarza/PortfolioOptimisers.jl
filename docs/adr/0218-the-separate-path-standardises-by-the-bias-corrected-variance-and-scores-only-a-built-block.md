---
status: accepted
---

# The separate path standardises by the bias-corrected variance and scores only a built block

## Context

Where `cor_decay` is not `nothing`, `RegimeAdjustedExpWeightedCovariance` runs a variance at
`decay` and a correlation state at `cor_decay`, and `update_var_cor!` divides each outer product by
the running volatilities before the correlation state takes it. Two defects of the warm-up of this
path came out of #1726.

1. **The raw variance weights the first products (#1729).** The variance state is seeded at zero,
   so after `K` steps it holds `1 - λ^K` times the variance. A product divided by the raw state thus
   carries an extra weight of about `1 / (1 - λ^K)`, which falls with `K`. The normalisation of
   `pair_weighted_correlation` cancels a scale that is the same for every product, but not this one.
   The first products of the correlation thus weigh more than the weights
   `(1 - λ_c) λ_c^l / (1 - λ_c^K)` that every bias table of the regime statistic assumes. The
   oracle reads the raw state. On a replica of the path, checked against the library to 1e-15,
   with 200 000 draws per row, iid Normal returns, half-lives of 10 and 20 and an equal-weight
   direction, the true factor of `PortfolioTarget` over the factor that the code reads was, for the
   root mean square:

   | assets, correlation | K | raw variance | bias-corrected variance |
   | --- | --- | --- | --- |
   | 4, `R = I` | 20 | 1.0353 | 0.9995 |
   | 4, `R = I` | 40 | 1.0068 | 1.0000 |
   | 12, equicorrelation 0.5 | 20 | 1.0802 | 1.0172 |
   | 12, equicorrelation 0.5 | 40 | 1.0448 | 1.0095 |
   | 4, equicorrelation 0.5 | 200 | 0.9996 | 0.9995 |

   The first moment and the log moved the same way. The rest at a correlated `R` is the second
   order of the estimated correlation in the warm-up, which the factor of #1726 does not model.

2. **The statistic scores a block that is not an estimate (#1730).** `update_var_cor!` writes the
   covariance block only when every diagonal entry of the correlation state of the active block is
   above `min_val`. Under HAC with `hac_vol_before = VolatilityBeforeUpdate()`, an asset contributes
   nothing on its first row, and a HAC row can have a negative diagonal, so the block can keep its
   zero seed for the first rows. `update_regime!` scored a row once the centring was ready and the
   bias was open, and neither reads whether the block is built. `PortfolioTarget` and
   `DiagonalTarget` floor a variance at `min_val`, so on a zero block the statistic is
   `r² / 1e-12`, and the regime state keeps it for about 40 regime half-lives. A built HAC block
   in the warm-up can also be indefinite. On 8 assets at an equicorrelation of 0.5 and two lags,
   one seed of 20 gave a built block with the smallest eigenvalue −0.68 after 5 rows, and the
   equal-weight direction had the variance −0.024. The floor scored it at about 1e11.
   `MahalanobisTarget` refuses the indefinite block, because its Cholesky factor fails. It does
   not refuse the zero block: `safe_regime_cholesky` adds a ridge of `min_val`, so the factor of
   a zero block succeeds. Under a debiasing rule its gate keeps the first rows out, and under
   `debias = RawStatistic()` it scored the zero block too.

## Decision

1. `RegimeAdjustedExpWeightedCovariance` holds the rule for the variance that standardises each
   product of the separate path in a field `cor_vol::AbstractCorrelationVolatility`.
   `BiasCorrectedVolatility()`, the default, divides the variance state by its weight
   `1 - λ^K` before the product divides by it, with `K` the count of the steps that the state
   holds: the count before the row under `VolatilityBeforeUpdate()` with HAC, and one more after
   it otherwise. `RawVolatility()` reads the raw state, the rule of the oracle. A new rule is a
   method of `correlation_variance`. The root is `public`, the two rules are exported.
2. On the separate path, `update_regime!` scores an asset only once `regime_block_built` holds
   for it: an asset whose variance state is positive and whose covariance diagonal is zero was
   never written, and it is left out. A written asset has a diagonal equal to its variance state.
   A constant asset has a variance of zero and a zero diagonal, so it stays in the statistic, and
   its zero row is the estimate.
3. `regime_statistic(::PortfolioTarget, …)` refuses the row, it returns `nothing`, where the
   variance of a direction is not above `min_val`, as `MahalanobisTarget` refuses a block that
   does not factorise. It no longer floors that variance at `min_val`.
4. `PortfolioTarget` holds the rule for a block that is not positive definite in a field
   `indefinite::AbstractIndefiniteBlockRule` (#1736). `RefuseIndefiniteWarmUp()`, the default,
   refuses a block with a negative eigenvalue while the block is young, and scores it once the
   block is old. A singular block, such as a block of fewer observations than assets, is kept,
   because each direction reads a true variance on it. The block is young while the gate that
   `MahalanobisTarget` reads under `ExactDebias()` is closed for the same assets:
   `regime_bias_open(ExactDebias(), …)` at the count of the youngest asset, on the decay of the
   correlation structure and the HAC lags of the estimator. The gate reads `ExactDebias()` under
   every `debias`, so a `RawStatistic()` estimate has the same young blocks. `ClipIndefiniteBlock()` clips the eigenvalues of
   the correlation of the block at zero with `restore_psd!`, keeps the variances, and scores the
   clipped block at every count. `ScoreIndefiniteBlock()` scores the block as it is, the rule
   before #1736. A new rule is a method of `indefinite_block`. The root is `public`, the three
   rules are exported. The maintainer asked for the three rules behind singleton types, and the
   strict rule is the default, because it does not move the steady state of an estimate whose
   gate opens.

## Consequences

- Decisions 1 and 2 do not move the steady state: the weight `1 - λ^K` goes to one, and a long
  fit writes every block. Decisions 3 and 4 act at every count. Decision 3 refuses a row where
  any direction has a variance at or below `min_val`, such as a direction that holds only a
  constant asset, where the floor scored that direction at zero. Decision 4 refuses an indefinite
  block at every count where the gate never opens, as the last consequence states.
- The warm-up of the separate path moves for every target, in the report and in the regime state.
  The parity cases of the oracle on the separate path, `test_12y` and the `separate corr` case of
  `test_08y`, state `cor_vol = RawVolatility()`, and the portfolio case of `test_12y` also states
  `indefinite = ScoreIndefiniteBlock()`.
- The first rows of the regime statistic under HAC on the separate path are now skipped until the
  block is built, where they scored 1e12. A row whose direction has a variance at or below
  `min_val` is skipped on both paths, for `PortfolioTarget`. On the case of #1730, 8 assets, an
  equicorrelation of 0.5, 120 000 rows, half-lives of 10 and 20 and two lags, the squared
  multiplier of `PortfolioTarget` was 1.8e6 and of `DiagonalTarget` 2.1e6. It is now 0.99467 and
  1.0079, beside 0.99522 and 1.0079 on one decay. Over 3 seeds and both centrings, 108 fits, no
  multiplier and no regime state after 6 rows reached 1e3, and the two rules of `cor_vol` gave the
  same multiplier to 4e-5.
- A built HAC block in the warm-up can be indefinite and still give a direction a small positive
  variance, and the row then scores high. A refusal of every indefinite block would also refuse
  the blocks of a holiday or a late listing, which ADR 0181 lets be indefinite, in the steady
  state. The gate of the Mahalanobis target separates the two: before it the block is young, and
  after it `RefuseIndefiniteWarmUp()` scores every block, so the steady state does not move. On 8
  assets at an equicorrelation of 0.5, half-lives of 10 and 20, two lags and 30 seeds of 8 rows,
  the largest and second largest regime state of the inverse-volatility direction were:

  | rule | one decay | separate path |
  | --- | --- | --- |
  | `ScoreIndefiniteBlock()` | 14.8, 4.95 | 51.4, 28.7 |
  | `ClipIndefiniteBlock()` | 14.8, 4.95 | 24.3, 15.1 |
  | `RefuseIndefiniteWarmUp()` | 14.8, 4.95 | no seed scored |

  On one decay the young blocks of those rows are singular, 8 assets over 6 rows, but they have
  no negative eigenvalue, so the rule scores them as before: each direction reads a true variance
  on a singular block. A first version that refused every block with no Cholesky factor refused
  those rows too, and it would have moved the warm-up of every plain panel with more assets than
  `min_obs`. On the separate path the first scored row is row 14 in the median and row 21 at
  most, where the gate opens at 20 rows. Before #1736 it was row 6, and row 8 at most. An estimate whose steady state never opens the gate, such
  as 12 assets at a half-life of 5 under HAC, refuses an indefinite block at every count.

## Alternatives considered

- **Keep the raw rule as the default.** It is the oracle's rule, but it puts a bias of up to 8 %
  in the factor of `PortfolioTarget` in the warm-up, which the bias tables do not model. The
  oracle's rule stays one keyword away.
- **Refuse a block with a diagonal entry at or below `min_val`.** It removes the explosion, but it
  also refuses every block that holds a constant asset, whose variance the separate path keeps at
  zero on purpose (#1383).
- **Read the weight state of the correlation.** `update_var_cor!` advances `cor_weight` before it
  decides not to write the block, so the weight is positive while the block is still zero.
- **Floor a direction's variance at a larger value.** Any floor scores the row against a number
  that is not an estimate.
- **Refuse an indefinite block only while `K < min_obs + hac_lags`** (#1736). The count has no
  tie to the law of the block: the HAC weight matrix has negative eigenvalues far past it, and
  the gate of the Mahalanobis target is the count at which the estimator already trusts a block
  of the same assets.

## Related

- ADR 0181, ADR 0190
- #1726, #1729, #1730, #1736

## Amendment (2026-10-27)

### Context

The sweep of `02_RegimeAdjustedExpWeightedCovariance_a.jl` (#631) found that the rule of Context
item 2 freezes the block of every asset. `update_var_cor!` wrote the block only when every active
asset had a diagonal of the correlation state above `min_val`, and else it wrote nothing. So one
asset with no diagonal, such as an asset on the row of its first deviation under
`VolatilityBeforeUpdate()`, kept the row of every other asset at the row before, and the next row
scored those assets against it. On 3 assets, half-lives of about 6.6 and 13.5, two lags, and a
third asset that lists at row 21, the block of the two old assets at row 22 was 16.9 % from the
block that the same two assets give alone. The oracle has the same rule.

### Decision

`update_var_cor!` writes the row of each active asset whose diagonal of the correlation state is
above `min_val`. It sets to zero the row and the column of every other active asset with a
positive variance. Decision 2 then holds for each asset: `regime_block_built` reads such an
asset as not built, and the statistic leaves it out. The other assets take their row at every
observation.

### Consequences

- Without HAC, or after the warm-up, every diagonal is positive, and nothing moves. The parity
  cases of `test_12y` and `test_08y` do not move.
- In the warm-up under HAC, a row can now score the built assets before every asset is built.
  On the panel of the table above, 30 seeds of 8 rows, the largest and second largest regime
  state of the inverse-volatility direction on the separate path are now 28.7 and 4.73 under
  `ScoreIndefiniteBlock()`, and 24.3 and 4.65 under `ClipIndefiniteBlock()`.
  `RefuseIndefiniteWarmUp()` still scores no seed. The path with one decay does not move. The
  statistic of row 7 of the seed with the largest score is the same: 51.4 under
  `ScoreIndefiniteBlock()`, and 15.1 under `ClipIndefiniteBlock()`. Its state is lower now,
  because the row before it is scored too.
- Under `RefuseIndefiniteWarmUp()`, over 30 seeds of 40 rows of the same panel, the first scored
  row is row 10 at least, 15.5 in the median, and 21 at most.
