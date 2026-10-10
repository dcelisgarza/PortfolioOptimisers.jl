---
status: accepted
---

# A tail measure reaches its worst weighted loss at level zero, and a risk measure checks its weights against the data

## Context

[#1621](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1621) audited the observation
weights of the risk measures. It found three gaps.

**The Ulcer Index took no weights.** Every other drawdown measure takes `w::Option{<:ObsWeights}`
and falls back to the prior's `pr.w`, so a weighted prior gave a weighted average drawdown beside
an unweighted Ulcer Index. Its JuMP formulation also kept one expression for each model, so two
Ulcer Index measures with different weights would have shared it.

**The weights were checked weakly.** No check compared the length of the weights with the number of
observations. The weighted quantile kernels read the weights through the sort permutation of the
returns, so a weight vector one entry too long passed in silence, and `ConditionalValueatRisk`
scaled its tail by the sum of every weight. No constructor refused weights that sum to zero.
`LowOrderMoment` checked only that its weights were not empty. A `DynamicAbstractWeights` has no
values until it resolves, so no check saw them.

**No weighted measure gave the worst loss.** `WorstRealisation` and `MaximumDrawdown` read every
observation, an observation of zero weight included. The oracle keeps both unweighted, and its tests
pin that a loss of zero weight still sets them. It reaches the worst loss among the observations of
positive weight through its four tail measures at confidence level one instead: CVaR, EVaR, CDaR and
EDaR. Our significance level `alpha` is one minus that confidence level, and every tail measure
refused `alpha = 0`.

The maintainer chose among four options: a weight field on `WorstRealisation` and `MaximumDrawdown`
that reads the measure's own weights only, the same field with the prior fallback, level zero of the
tail measures, or both. The choice was level zero.

## Decision

**`UlcerIndex` and `RelativeUlcerIndex` take `w::Option{<:ObsWeights}`** with the prior fallback of
`@propagatable`. The weighted index is the square root of the weighted mean of the squared
drawdowns, with the drawdowns taken on the full return path. The kernel `ulcer_index(dd, w)` serves
both measures, as `average_drawdown` serves the average drawdown. In JuMP each drawdown is scaled by
`sqrt(w_t)` inside the second-order cone, and the expression divides by `sqrt(sum(w))`. The model
keys `:uci_`, `:uci_risk_` and `:cuci_soc_` are indexed by the measure, as the keys of
`AverageDrawdown` are.

**One assertion validates the observation weights of a risk measure at construction.**
`assert_observation_weights` refuses empty, non-finite and negative weights, and weights that sum to
zero. Every risk measure constructor that holds observation weights calls it.

**One resolver checks the weights against the data at evaluation.** `checked_observation_weights(w, X)`
resolves the weights, validates the values of a resolved `DynamicAbstractWeights` or prior weights,
and raises a `DimensionMismatch` that names the weights when `length(w) != size(X, 1)`. Every
numeric and JuMP site of the risk measures resolves its weights through it, and a moment measure
with stored weights runs it through `resolve_observation_weights`. The moment estimators keep
`get_observation_weights`: some of them pass `dims = 2`, so a length check there is a change of its
own.

**`ConditionalValueatRisk`, `EntropicValueatRisk`, `ConditionalDrawdownatRisk`,
`EntropicDrawdownatRisk`, `RelativeConditionalDrawdownatRisk` and `RelativeEntropicDrawdownatRisk`
take `0 <= alpha < 1`.** At `alpha = 0` the measure is the largest loss among the observations of
positive weight, `worst_positive_weight_loss(x, w)`. Unweighted, it equals `WorstRealisation` on the
returns and `MaximumDrawdown` on the drawdowns. In JuMP, `set_worst_loss_constraints!` writes one
variable above the loss of each observation of positive weight, and both the conditional and the
entropic builder call it at level zero. A drawdown measure takes its drawdowns on the full path, so
a loss of zero weight still moves every later drawdown, as ADR 0059 states for the drawdown twins.

`WorstRealisation` and `MaximumDrawdown` stay unweighted. `ValueatRisk`, the range measures and the
distributionally robust measures keep `0 < alpha < 1`.

## Consequences

- A weight vector of the wrong length raises at the first evaluation or model build that reads it.
  A caller that relied on the silent truncation now meets an error.
- A `DynamicAbstractWeights` that resolves to invalid values raises at evaluation, not at
  construction, because it has no values before.
- `alpha = 0` is a valid input of six measures, and a Calibration Rule that returns zero for them
  now passes the rebuild check.
- `NaN` observations are not part of this decision.
  [#1622](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1622) decides them, because
  the rule depends on the type of the weights.

## Related

- [ADR 0043](0043-nothing-observation-weights-means-unweighted-not-unavailable.md)
- [ADR 0059](0059-a-drawdown-tail-measure-is-its-returns-twin-on-another-series.md)
