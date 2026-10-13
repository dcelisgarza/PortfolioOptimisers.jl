---
status: accepted
---

# A reader of the prior returns pairs the implied volatilities of the last rows

## Context

[#1655](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1655) found that an HRP over
a `CrossSectionalFactorPrior`, whose clustering reads `ImpliedVolatility`, raised a
`DimensionMismatch` in batch under the default `x_src = :prior`. The clustering read `pr.X`, which
holds the fitted rows of the prior alone, and `rd.iv`, which holds every row of the returns data.
An `EmpiricalPrior` with a Scenario Cap raised the same error, so every prior whose `X` holds fewer
rows than the returns data was affected. `ImpliedVolatility` pairs row `t` of `iv` with row `t` of
`X`, and it refuses two matrices of different sizes.

The issue named two choices. Each prior that drops rows states which rows of the returns data its
`X` holds, or the clustering refuses `iv` under `x_src = :prior` when the row counts differ.

## Decision

**A forwarder that reads `pr.X` gives the estimator the last `size(pr.X, 1)` rows of `iv`.** Each
prior that drops rows drops leading rows. A Scenario Cap keeps the last rows (ADR 0136), and a
`CrossSectionalFactorPrior` drops the rows of the Descriptor warm-up and of the exposure lag. So
the rows of `pr.X` are the last rows of the observations. ADR 0206 reads the probabilities `w` of a
prior result on the same last rows, through `scenario_rows`.

1. `implied_volatility_picker` gives the last `size(X, 1)` rows of `iv`, where `X` is the matrix
   that `returns_matrix_picker` picked. Each forwarder of `Pr_RR` that calls
   `returns_matrix_picker` takes `iv` as a keyword and passes on the rows that it picks.
2. Under `x_src = :data` the picked `X` is `rd.X`, and `iv` already has its rows, so nothing
   changes.
3. An `iv` with fewer rows than `X` pairs no row. It passes on unchanged, and the estimator refuses
   it with its own `DimensionMismatch`.

No prior states a row key for this. The rule needs none, because every prior keeps the last rows.
A prior that keeps other rows must answer its own rows, and it must amend this ADR.

## Consequences

- An optimiser that clusters the prior returns with `ImpliedVolatility` fits in batch over every
  prior that drops rows, and its online step gives the batch weights.
- The rows of `src/10_Prior/01_Base_Prior.jl` grew past the size ceiling, so the Scenario Fill and
  Scenario Cap helpers moved unchanged to `src/10_Prior/19_PriorScenarios.jl`.
