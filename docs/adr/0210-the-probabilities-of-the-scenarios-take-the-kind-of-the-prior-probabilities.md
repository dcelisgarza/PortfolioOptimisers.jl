---
status: accepted
---

# The probabilities of the scenarios take the kind of the prior probabilities

## Context

[#1687](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1687) found that the
posterior covariance of `EntropyPoolingPrior`, `MeucciEntropyPoolingPrior` and
`OpinionPoolingPrior` took the bias correction of `StatsBase.ProbabilityWeights`. That correction
is `T / (T - 1)` for every set of weights. The posterior weights are probabilities of the
scenarios, that is reliability weights. The unbiased weighted covariance under reliability weights
`p` with `sum(p) == 1` divides by `1 - sum(p .^ 2)`. This is `n_eff / (n_eff - 1)` with the Kish
count `n_eff = 1 / sum(p .^ 2)`, the count that ADR 0203 reads in a weighted regression. So the
posterior `sigma` was too small, and the gap grew as a view concentrated the weights. On the panel
of the issue, one mean view of 0.010 gave `n_eff = 49.70` of `T = 60`, and a gap of `3.5e-3` on
every entry of `sigma`.

ADR 0087 fixes the correction by the type of the weights, because `corrected = true` is the
default of the library. So the type that each site writes is the decision. Of the four kinds of
`StatsBase`, only `StatsBase.AnalyticWeights` gives the correction of reliability weights.
`StatsBase.FrequencyWeights` that sum to one divide by zero, and `StatsBase.Weights` refuses
`corrected = true`.

## Decision

**The probabilities of the scenarios take the kind of the prior probabilities, and the default
kind is `StatsBase.AnalyticWeights`.**

1. The private alias `ScenarioProbabilities` is `Union{<:StatsBase.AnalyticWeights,
   <:StatsBase.ProbabilityWeights}`. The field `w` of `EntropyPoolingPrior` and
   `MeucciEntropyPoolingPrior` is `Option{<:ScenarioProbabilities}`. The bound was
   `Option{<:StatsBase.ProbabilityWeights}`, so every value that was valid stays valid.
2. When a caller gives `w`, every posterior that a solver answers is made with the type
   constructor of `w`, through `scenario_probabilities(w, x)`. A caller who gives
   `StatsBase.pweights` keeps the old correction.
3. When `w` is `nothing`, `ep_prior_probabilities` reads the `w` of the nested result. It keeps
   that kind when it is a `ScenarioProbabilities`, and makes any other vector
   `StatsBase.AnalyticWeights`. Without a nested `w`, the uniform prior probabilities are
   `StatsBase.AnalyticWeights`.
4. `compute_pooling` answers `StatsBase.AnalyticWeights`, because an opinion pool has no
   scenario weights of a caller. `scenario_weights` keeps the kind of the probabilities that it
   pads. `cross_sectional_scenario_weights` turns a kind it cannot keep into
   `StatsBase.AnalyticWeights`.
5. The weights that the tail views give to `ConditionalValueatRisk`, `EntropicValueatRisk` and
   `RelativisticValueatRisk` stay `StatsBase.pweights`. These measures apply no bias correction,
   so the kind does not move them.

Under uniform weights the two corrections are equal, so a prior with no view answers the same
`sigma` as before. Under a view, every posterior covariance with the default `w` moves by the
factor `(n_eff / (n_eff - 1)) / (T / (T - 1))`. The weights and `mu` do not move, because a
weighted mean has no correction.

## Alternatives rejected

- **Keep `ProbabilityWeights` in `pr.w`, and give `aweights` of them to the moment estimators
  only.** The field bound holds, but a reader of `pr.w` that computes a variance still takes
  `T / (T - 1)`. The weights are reliability weights, and the type says so.
- **A field on the three priors whose singleton types name the correction.** The type of the
  weights already names the correction (ADR 0087). A second field could disagree with it.
- **Widen the bound to every `StatsBase.AbstractWeights`.** `FrequencyWeights` and `Weights` give
  no usable correction under the default `corrected = true`, so the bound refuses them at the
  constructor rather than at the refit.
