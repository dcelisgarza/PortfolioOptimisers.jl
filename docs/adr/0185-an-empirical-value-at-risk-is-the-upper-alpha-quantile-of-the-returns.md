---
status: accepted
---

# An empirical Value at Risk is the upper alpha-quantile of the returns

## Context

`cajas2025`, Section 7.2.2.3, states the Value at Risk twice. Equation 7.50 is the definition:

```text
VaR_α(X) = -inf{x : F_X(x) > α} = F_Y^{-1}(1 - α),   Y = -X.
```

Equation 7.51 is a mixed-integer programme for it: minimise `t` subject to
`t ≥ -R_t x - M z_t` and `1'z ≤ (α - π) T`, with `π` "a very small number".

On a sample of `T` returns sorted ascending, `F(x_(k)) = k / T`, so Equation 7.50 selects the
position `k = ⌊αT⌋ + 1`, the `(⌊αT⌋ + 1)`-th largest loss. The programme exempts the largest losses
while their count stays within `(α - π) T`, so its minimum is at the position `⌊(α - π) T⌋ + 1`.
That is `⌈αT⌉` while `πT < 1`. The two agree when `αT` is not an integer. When `αT` is an integer
they differ by one position: the slack `π` refuses the last exemption that Equation 7.50 allows.

The library followed the programme, with `π` the slack `s` of `MIPValueatRisk` and
`DrawdownatRisk`, default `1e-5`, in its model and in its functor `empirical_value_at_risk`. The
kernel feeds `ValueatRisk`, `ValueatRiskRange`, `DrawdownatRisk` and `RelativeDrawdownatRisk`.
Issue #1364 asked which convention the library keeps.

The fixed slack had a second effect. `s` is a fraction of `T`, so once `sT ≥ 1`, at `T ≥ 100 000`,
the position moved also at a non-integer `αT`.

## Decision

1. **The empirical Value at Risk selects the position of Equation 7.50,
   `k = min{k : W_k > α W_T}`,** with `W_k` the cumulative observation weight of the `k` smallest
   returns. Without weights that is `k = ⌊αT⌋ + 1`. The model and the functor select this one
   position, so they still report one number.
2. **The slack loosens the cardinality row instead of tightening it.** The row is
   `Σ w_t z_t ≤ (α + s) W_T`, and the kernel reads `(α + s)` in the same place. The slack only
   absorbs the rounding error of `α W_T`, for example `0.29 * 100 = 28.999999999999996`.
3. **The default slack is `1e-9`.** It exceeds the rounding error of `α W_T` and of a cumulative
   sum of up to about `10^6` weights, and `sT` stays below `10^{-3}` up to `T = 10^6`, so the slack
   moves no position on a sample of that size.
4. **Drawdown at Risk follows the same rule.** The reference calls it "the application of VaR to
   the drawdowns distribution", and its programme, Equation 7.91, is Equation 7.51 on the drawdowns.
5. **The Conditional Value at Risk reads the same order statistic.** Its closed form splits the
   tail at the boundary index `k* = min{k : W_k > α W_T}`, the index of the Value at Risk, in place
   of `min{k : W_k ≥ α W_T}`. Where the two differ, `W_{k*-1} = α W_T` and the boundary entry
   takes zero weight, so the value of every CVaR, CVaR range and CDaR is unchanged. The `var`
   that each kernel reads is then the Value at Risk of the same level, and
   `CVaR = VaR + Σ w_t (ℓ_t - VaR)^+ / (α W_T)`.

## Why Equation 7.50

- **It is the definition.** Both forms of Equation 7.50 give the same position, and the programme
  of Equation 7.51 is stated as a formulation of it. Without the slack, the programme
  `1'z ≤ αT` selects exactly the position of Equation 7.50. The slack of the reference moves it.
- **It is the loss that the portfolio exceeds with probability `α`.** Equation 7.50 is the
  smallest threshold `ℓ` with `P(L > ℓ) ≤ α`. At an integer `αT`, `αT` losses exceed it, which is
  a share of exactly `α`. The position `⌈αT⌉` is exceeded by `αT - 1` losses, so it is not the
  smallest such threshold.
- **It is the Value at Risk that the definition of the CVaR names.** The library defines the CVaR
  as the minimum of the function of Rockafellar and Uryasev, `t + Σ w_t (ℓ_t - t)^+ / (α W_T)`.
  At an integer `αT` its minimisers are the interval from the `(⌊αT⌋ + 1)`-th to the `⌈αT⌉`-th
  largest loss. Rockafellar and Uryasev define the Value at Risk as the left end of that interval,
  the lower `(1 - α)`-quantile of the losses, which is Equation 7.50. With it, the loss exceeds
  the Value at Risk with probability `α` exactly, so the CVaR is also `E[L | L > VaR]`. Both ends
  give the same CVaR, so only the left end is a choice that the definition of the CVaR makes.
- **It does not collapse onto the worst loss.** At `αT = 1`, for example `α = 0.01` and
  `T = 100`, the position `⌈αT⌉` is the largest loss. The Value at Risk then equals the
  Conditional Value at Risk and the worst realisation. Equation 7.50 gives the second largest
  loss, which stays below the tail mean as a quantile must.

## Consequences

- At an integer `αT` every empirical Value at Risk, Value at Risk range, Drawdown at Risk and
  Relative Drawdown at Risk is one position deeper into the sample, so it is equal to or smaller
  than before. At a non-integer `αT` below `T = 10^6` nothing changes.
- A caller that stated `s` reads a different row: `(α + s)` in place of `(α - s)`. A stated `s`
  of `1e-5` now moves the position once `T ≥ 100 000`, as the default did before, so a caller
  keeps `s` small.
- The value of the Conditional Value at Risk does not change. Its kernels move their boundary
  index to the index of the Value at Risk, which gives the same tail mean.
- A value at risk view of entropy pooling pins the posterior mass of the losses at or beyond its
  target to `α`. A mass of exactly `α` now reads the largest loss under the target, and a mass
  above `α` reads the smallest loss at or above it. Both bracket the target, as the note of
  `ep_var_views!` already stated, so the rows of the view do not change.
