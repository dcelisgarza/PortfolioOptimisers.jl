---
status: accepted
---

# A score orients each risk measure before it reduces a vector

## Context

Map #1980 lets a search score with a vector of risk measures. The scorers of the library read
one number for each prediction, and "higher is better" for a search score (ADR 0226). The
issue #1982 asked four questions about the slot that reduces the vector: its bound, its seam,
what it reads, and the rule for a vector whose measures disagree on orientation.

Two facts decided the answer.

1. **A reader of a risk and a scorer need different things.** `expected_risk` and its siblings
   give the risk of a portfolio, and the derivative readers (`risk_gradient`,
   `risk_contribution`, `RiskLoss`) need the chain-rule weight and the degree of homogeneity of
   the reduction. A function, and most `VectorToScalarMeasure` types, state neither. A scorer
   needs only one number from numbers that are already computed.
2. **The raw reduction of a mixed vector has no orientation.** `quantile_by_measure` reduced the
   raw risks of the vector and then applied one `sign`. Its docstring said that the sign "lets a
   mixed vector through". But for `[ConditionalValueatRisk(), SharpeRatio()]` under the default
   `SumScalariser`, the value of a path was `sign * (CVaR + Sharpe)`. With `sign = 1` a higher
   Sharpe ratio made a path worse, and with `sign = -1` a higher CVaR made it better. No sign
   gave the correct order, so `NearestQuantilePrediction` selected a wrong path and gave no
   warning. `sort_by_measure` refused the same vector, because `bigger_is_better` throws on it.

## Decision

**A score orients each risk measure, then multiplies it by its scale, then reduces the vector.
The readers of a risk do not change.**

| Item | Rule |
| --- | --- |
| The readers of a risk | `expected_risk`, its value siblings, the derivative readers and the optimisers keep `sca::Scalariser`. `JuMPOptimiser.sca` keeps `NonHierarchicalScalariser`. `nothing` is not a value of their `sca`. |
| The score | `oriented_score(sca, r, p; kwargs...)`, internal, in `07_Scoring.jl`. The caller holds the reduction and passes it first. A keyword form that took `sca` out of `kwargs` is removed (ADR 0232). For member `k` it reads `o_k * expected_risk(r_k, p; kwargs...) * r_k.settings.scale`, with `o_k = 1` when `bigger_is_better(r_k)` holds and `-1` otherwise. |
| The bound | `const Sca_VecToScaM = Union{Scalariser, VectorToScalarMeasure, Function}`, after the precedent `Num_VecToScaM`. A slot that admits `nothing` binds `Option{<:Sca_VecToScaM}`. A `Number` is not a member, because it gives one score to every candidate. |
| The seams | Neither seam changes. A `Scalariser` reduces the values through `scalarise` (ADR 0024). A `VectorToScalarMeasure` or a function reduces the vector of values through `vec_to_real_measure`. |
| The orientation | Every value is "higher is better" before the reduction. So a vector of mixed orientation has one score, and no outer sign exists. `MaxScalariser` and `MaxValue` give the best member, and `MinScalariser` and `MinValue` the worst. The score reads every reduction as "higher is better", so `StdValue` gives the highest score to the largest spread. |
| A negative scale | Used as given. It is a negative weight, so that member counts against the score. |
| `nothing` | Returns the vector of oriented values, so each member keeps its own score. |
| One measure | The slot and the scale are inert, as in `expected_risk`. The score is `o * expected_risk(r, p)`. |

The scorers of paths take the score now:

- `sort_by_measure` sorts by the score in descending order, and admits a mixed vector.
- `quantile_by_measure` reads the negated score, so `q = 0` is the best path and `q = 1` the
  worst, whatever the orientation of `r`. Its `sign` keyword is removed, and so is the `sign`
  argument of the `NearestQuantilePrediction` functor. The orientation of each measure does
  their job. A caller who followed the old docstring, `sign = -1` for a measure of which a
  bigger value is better, would invert that measure a second time.

The two searches and `FoldRiskScorer` take the same score in the build tickets of map #1980.
ADR 0232 (#1983) decides where a search holds the reduction of its risk measures, its default,
and the shape of the scores under `nothing`. Issue #1984 decides the same for the search scorers.

## Considered options

- **Widen `sca` of `expected_risk` and the other value readers.** Rejected by the maintainer:
  the slot is for scoring, and a wrapper is called when the searches score. A widened reader
  of a risk also leaves the derivative readers with no rule for the new members.
- **Widen the derivative readers too, with a finite difference of the reduction.** Rejected:
  a score needs no derivative, and the Euler sum of the contributions holds only for a
  reduction of degree one.
- **Take the sign from the members when they agree, or let the caller state it.** Rejected by
  the maintainer. The members cannot give one sign to a mixed vector, and a stated sign keeps
  the defect of `quantile_by_measure` for a caller who omits it.
- **Read the raw scaled risks, as `expected_risk` does.** Rejected: the reduced number has no
  orientation when the members disagree.

## Consequences

- The defect of `quantile_by_measure` and `NearestQuantilePrediction` on a vector of mixed
  orientation is fixed, with a regression test in `test/test_24_cross_validation.jl`.
- One measure, and a vector under `SumScalariser`, select and sort as before, bit for bit:
  negation is exact, and a stable sort keeps the order of ties. A vector under `MaxScalariser`,
  `MinScalariser` or `LogSumExpScalariser` moves, because the reduction reads oriented values.
  `MinScalariser` gives the old `MaxScalariser` path. One measure of which a bigger value is
  better, under the old default `sign = 1`, now reads `q = 1` as its worst path. The migration
  guide names each break.
- `NearestQuantilePrediction`, `sort_by_measure` and `quantile_by_measure` admit a
  `VectorToScalarMeasure` or a function as the reduction of a vector, through `sca`.
- No released number of a search moves, because both searches refuse a vector of measures now.
- **A call that uses no new capability costs what it cost before.** The orientation is a trait
  of the type, so the compiler drops its branch, and the negation in `quantile_by_measure`
  cancels the flip of a risk measure. A vector method reads the returns of the prediction once
  with `prediction_returns`, and passes them to each member. The members of a mixed vector have
  an abstract type, so a call that passed the prediction itself would box it once for each
  member, 64 more bytes per path for two members. Measured against the old bodies, on one risk
  measure, one reward measure and a vector under `SumScalariser`, over 9 paths of 30
  observations and 50 paths of 250: the same bytes in every case, and times within the noise
  of about one percent.
