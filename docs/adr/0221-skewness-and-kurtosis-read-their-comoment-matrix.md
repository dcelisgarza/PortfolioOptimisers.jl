---
status: accepted
---

# Skewness and Kurtosis read their comoment matrix

## Context

[#1846](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1846) found that the
functors of `Skewness` and `Kurtosis` gave a value that their models do not bound.

- The functor of `Skewness` read the returns and gave the standardised skewness. The term of
  `VarianceSkewKurtosis` reads the coskewness matrix and bounds the raw third moment
  `w' M3 (w ⊗ w)`.
- The functor of `Kurtosis` read the returns and gave `m4`, or `√m4` under `SOCRiskExpr`. Every
  model of `Kurtosis` reads the cokurtosis matrix. The two values agreed only when the matrix was
  the sample cokurtosis about the Moment Target of the measure.

So `expected_risk` of either measure did not value the expression that a bound of the measure
holds. #1840 met this in the span of a sweep on the skewness floor, and it added local methods
of `frontier_risk` for the two measures.

The fields `w`, `mu`, `alg1` and the `ve` of `Skewness` acted on the functor alone. The matrix
carries its own centre, observation weights and tail, from the estimator that makes it.

## Decision

**`Skewness` and `Kurtosis` read their comoment matrix, as `Variance` reads `sigma`. The values
that their functors read from the returns move to the moment measures.**

1. The functor of `Skewness` gives `portfolio_third_moment(r.sk, w)`. The functor of `Kurtosis`
   gives `portfolio_fourth_moment(r.kt, w)`, square-rooted under `SOCRiskExpr`. Both are
   `WeightsInput` measures, and `factory` with a `HighOrderPrior` fills the matrix. A measure
   with no matrix is refused by name at a value-level entry point, through `functor_slots`.
2. `Skewness` keeps `settings` and `sk`. `Kurtosis` keeps `settings`, `kt`, `N`, `alg` and
   `mtx_sqrt`. The fields `w`, `mu` and `alg1` go, and so does the `ve` of `Skewness`. The
   centre, the weights and the semi-moment belong to the estimator in `sk` or `kt`, such as
   `Cokurtosis(; w = w, alg = SemiMoment())`.
3. `alg2` of `Kurtosis` becomes `alg`. A measure with one formulation names it `alg`, as
   `Variance` and `NegativeSkewness` do. `SecondMoment` and `BrownianDistanceVariance` keep
   `alg1` and `alg2`, because each holds two algorithms.
4. The `pe` of both measures goes. Each now carries one deferrable slot, and the rule of
   `NegativeSkewness` holds: such a slot takes the prior estimator itself, as
   `Kurtosis(; kt = HighOrderPriorEstimator())`. The `pe` of `VarianceSkewKurtosis` stays, and
   it fills `sigma`, `sk` and `kt`.
5. `ThirdCentralMoment` gains `ve = nothing`. With a variance estimator in `ve`, it divides the
   third moment by the cube of the standard deviation, as `StandardisedHighOrderMoment` does for
   the fourth moment. So `ThirdCentralMoment(; ve = SimpleVariance())` gives the old value of
   the functor of `Skewness`.
6. `frontier_risk` goes. With the two local methods removed it only forwarded to
   `expected_risk`, and `_rebuild_risk_frontier` calls `expected_risk` directly.
7. `deferred_centre`, `centring_target` and `fit_deferred_moment` go. They threaded a stated
   `mu` into the fit of a comoment matrix, and no measure states that `mu` any more.

Each old value of the returns has a name that states what it is:

| Old functor | New measure |
| --- | --- |
| `Skewness(; w, mu)` | `ThirdCentralMoment(; w, mu, ve = SimpleVariance())` |
| `Kurtosis(; w, mu, alg2 = QuadRiskExpr())` | `HighOrderMoment(; w, mu, alg = FourthMoment())` |
| `Kurtosis(; w, mu)` | `LowOrderMoment(; w, mu, alg = EvenMoment())` |
| `alg1 = SemiMoment()` | `FourthMoment(; alg = SemiMoment())`, `EvenMoment(; alg = SemiMoment())` |

## Considered options

- **A full `ThirdMoment` for `StandardisedHighOrderMoment`.** The issue proposed it. It is
  rejected because `HighOrderMoment` is a `HierarchicalRiskMeasure`. A signed third moment,
  standardised or not, can be negative and is better when larger. Every other measure with
  `bigger_is_better = true` is a `NonOptimisationRiskMeasure`, which the type bound of every
  optimiser refuses. A negated `ThirdMoment` keeps the ranking but does not give the old value.
- **A functor that falls back to the returns when the matrix is `nothing`.** Rejected by the
  maintainer on the issue. The two values differ whenever the matrix is not the sample comoment
  about the Moment Target, so one functor would give two quantities.

## Consequences

- `expected_risk` of `Skewness` and of `Kurtosis` is the value of the bounded expression, so a
  frontier owner needs no special method. A sweep on the skewness floor and a sweep on a
  kurtosis ceiling take their spans from the functor.
- `Skewness` and `Kurtosis` cannot score one asset in an asset selection or a parameter search,
  because a `WeightsInput` measure has no precomputed-returns form. The refusal of
  `assert_scoreable` names the measure of the returns that can.
- With an integer `N`, the model bounds the approximate moment of `cokurtosis_spectrum`, which
  lies at or below the moment of the matrix. The functor gives the full moment of the matrix. The
  docstring of `Kurtosis` states the gap.
- The change breaks code that states `w`, `mu`, `alg1`, `alg2` or `pe` on either measure, or `ve`
  on `Skewness`, and code that evaluates either measure on a bare returns matrix. The migration
  guide of the next release takes the table above.
