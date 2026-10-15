---
status: accepted
---

# A reader of the prior returns pairs the implied volatilities of the last rows

## Context

[#1655](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1655) found that an HRP over
a `CrossSectionalFactorPrior`, whose clustering reads `ImpliedVolatility`, raised a
`DimensionMismatch` in batch. At that time every graph reader read `pr.X` by default. Today a
reader reads `pr.X` when its `x_src` is `PriorSource()` (ADR 0211, ADR 0212). The clustering read
`pr.X`, which holds the fitted rows of the prior alone, and `rd.iv`, which holds every row of the
returns data. An `EmpiricalPrior` with a Scenario Cap raised the same error, so every prior whose
`X` holds fewer rows than the returns data was affected. `ImpliedVolatility` pairs row `t` of `iv`
with row `t` of `X`, and it refuses two matrices of different sizes.

The issue named two choices. Each prior that drops rows states which rows of the returns data its
`X` holds, or the clustering refuses `iv` under `x_src = PriorSource()` when the row counts differ.

## Decision

**A forwarder that reads `pr.X` gives the estimator the last `size(pr.X, 1)` rows of `iv`.** A
Scenario Cap keeps the last rows (ADR 0136), and a `CrossSectionalFactorPrior` drops the rows of
the Descriptor warm-up and of the exposure lag, so each of them drops leading rows. A prior also
drops the observations of a Shared Series gap, which are not leading rows (#1672). An optimiser
drops those observations from `rd` before its prior and its readers read it, with
`shared_series_view`. So inside an optimiser the rows of `pr.X` are the last rows of the
observations that the optimiser kept, and the last rows of `rd.iv` pair them. ADR 0206 reads the
probabilities `w` of a prior result on the same last rows, through `scenario_rows`.

1. `implied_volatility_picker` gives the last `size(X, 1)` rows of `iv`, where `X` is the matrix
   that `returns_matrix_picker` picked. Each forwarder of `Pr_RR` that calls
   `returns_matrix_picker` takes `iv` as a keyword and passes on the rows that it picks.
2. Under any other source, `nothing` (the default), `DataSource()` or `ExcessSource()`, the picked
   `X` is `rd.X` or its excess form, `iv` already has its rows, and nothing changes.
3. An `iv` with fewer rows than `X` pairs no row. It passes on unchanged, and the estimator refuses
   it with its own `DimensionMismatch`.

The rule states no row key, because an optimiser cuts `rd` to the rows of its prior first. Two
calls do not cut `rd`: a graph verb that a caller gives a prior result and the raw `rd`, and a head
whose `pe` is a fitted prior result, which reads no factor return and so cuts no factor gap. After
a Shared Series gap the last rows of `iv` then pair the wrong observations.
[#1744](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1744) records the case, and
proposes that a prior result records the rows of the data that its `X` holds. A prior that keeps
other rows must answer its own rows, and it must amend this ADR.

## Consequences

- An optimiser that holds a prior estimator and clusters the prior returns with `ImpliedVolatility`
  fits in batch over every prior that drops rows. Its online step gives the batch weights, measured
  on the carry of `CrossSectionalFactorPrior`.
- `cor` of `ImpliedVolatility` takes the correlation of its inner estimator on `X`, and it discards
  the prediction from `iv`, which runs only so that `cor` refuses what `cov` refuses. So for a graph
  reader the pairing decides the validation of `iv` and not the answer: a wrong pairing shows as a
  false refusal of a non-finite `iv` (#1744).
- The rows of `src/10_Prior/01_Base_Prior.jl` grew past the size ceiling, so the Scenario Fill and
  Scenario Cap helpers moved unchanged to `src/10_Prior/19_PriorScenarios.jl`.
