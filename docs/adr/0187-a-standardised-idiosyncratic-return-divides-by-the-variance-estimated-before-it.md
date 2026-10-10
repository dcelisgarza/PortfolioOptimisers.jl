---
status: accepted
---

# A standardised idiosyncratic return divides by the variance estimated before it

## Context

The idiosyncratic group of the cross-sectional diagnostics (#800) scores the variance history of a
fitted block. `standardised_idio_returns` divides each idiosyncratic return by a predicted
volatility, and the calibration, the tail rate, the excess kurtosis, the skewness and their summary
read the answer.

Row `t` of the variance history `vs` is estimated from the observations up to `t`, so it has read
the return `ε_t`. The group divided `ε_t` by the volatility of the same row. The parity measure of
map #1375 (#1387) found that the oracle does the same, and that the rule is wrong for three
reasons:

1. **A forecast is scored against a return it has not seen.** The variance of row `t - 1` is the
   forecast of row `t`. The bias statistic of a risk model divides a return by the forecast made
   before it, and the oracle's own regime-adjusted variance standardises its regime signal in that
   way.
2. **The same-row ratio is bounded.** An exponentially weighted variance puts the weight `1 - λ`
   on the latest squared return, so `v_t ≥ (1 - λ) ε_t²` and `|z_t| ≤ 1 / sqrt(1 - λ)`, however
   large the shock. It pulls the deviation below `1` and the tails below the Gaussian reference for
   a forecast that is right. On the large parity panel, whose residuals are Gaussian, the 3-sigma
   tail rate is 0.12 % under the same-row rule and 0.55 % one step ahead, against a reference of
   0.27 %. The mean deviation is 0.988 against 1.009.
3. **The group contradicted itself.** `idio_vol_residual_dependence` already divided `|ε_{t+1}|`
   by the volatility of row `t`, and named it `|z_{t+1}|`. Under the same-row rule that name was
   wrong, and under the one-step-ahead rule it is exact.

## Decision

`standardised_idio_returns`, `idio_calibration`, `idio_tail_rate`, `idio_kurtosis`,
`idio_skewness`, `idio_calibration_summary` and their plots take `ahead::Bool = true`. Under the
default, row `t` of the standardised returns is `ε_t / σ̂_{t-1}`, and the first row is `NaN`,
because it has no forecast. `ahead = false` gives the same-row rule, which is the oracle's, so its
output stays one keyword away (ADR 0186). The two dependence series keep their definition, which
is the one-step-ahead rule already.

## Consequences

- A calibration series of a released version moves: its first entry is `NaN`, and every other
  entry reads the variance of the previous row. `test/test_08t_idio_diagnostics.jl` pins the
  oracle's literals under `ahead = false`, and `test/test_12r_parity_cs_diagnostics.jl` pins the
  default against the oracle's own kernels on the one-step-ahead standardisation.
- The verbs over `z` (`idio_calibration(z)` and its siblings) do not change: they read the
  standardised returns they are given.
