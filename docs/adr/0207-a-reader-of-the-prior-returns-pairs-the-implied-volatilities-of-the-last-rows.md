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

A prior also drops the observations of a Shared Series gap, which are not leading rows (#1672).
[#1744](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1744) found that after that
cut no object stated which observations of the returns data the rows of `pr.X` are. Three readers
paired the wrong rows. A graph verb that a caller gave a prior result and the raw `rd` paired the
last rows of `iv`. An online fallback read the `rd` that its head cut. The row key `rr.idx` of the
prior of an optimiser counted the positions of the data that the optimiser cut.

## Decision

**A prior records the observations that its Shared Series cut dropped, and every other row that a
fit drops is a leading row.** A Scenario Cap keeps the last rows (ADR 0136), and a
`CrossSectionalFactorPrior` drops the rows of the Descriptor warm-up and of the exposure lag. So
the rows of `pr.X` are the last `size(pr.X, 1)` observations that the record keeps.

1. `LowOrderPrior` gains `ss_gaps`, one mark for each observation of the returns data that the
   prior read, `true` at an observation that the prior dropped for a Shared Series gap. It is
   `nothing` when the prior dropped none. The constructor refuses marks with no gap, and marks
   that keep fewer observations than `X` has rows. `HighOrderPrior` answers it from the
   `LowOrderPrior` that it wraps.
2. `shared_series_record` writes the marks and maps the row key of the factor model block. It
   replaces `shared_series_row_key`. `prior(pe, rd)` calls it after it fits, and so does each
   read-out of an online fold: a buffer refit, the carry of `EmpiricalPrior` and the carry of
   `CrossSectionalFactorPrior`.
3. A reader that holds returns data with as many observations as the marks holds the data that
   the prior read, and it pairs the rows of `X` with the last observations that the marks keep. A
   reader that holds fewer observations holds the data after the same cut, or data with no record,
   and it pairs the last rows. The marks are `nothing` when nothing was cut, so the two cases never
   have the same length. `prior_data_rows` states the rule. `implied_volatility_picker` and
   `outer_prior_rows` read it.
4. An optimiser solves through `shared_series_solve`. It cuts `rd` once before the solve, so every
   reader inside the solve pairs the last rows, as before. A head whose prior is a result also
   drops the observations that the record of the result marks. After the solve, when the head
   fitted its prior on the cut data, the function maps that prior back to the observations of
   `rd` through the same rows. The prior of `optimise(opt, rd)` then states the rows that
   `prior(pe, rd)` states.
5. `batch_from_state` returns the whole rebuilt `rd`. The head drops the observations that its
   folded prior marks, and each fallback drops by its own rule, as in batch.

A prior that keeps rows other than the last ones must record them, and it must amend this ADR.

## Consequences

- An optimiser that holds a prior estimator and clusters the prior returns with `ImpliedVolatility`
  fits in batch over every prior that drops rows. Its online step gives the batch weights, measured
  on the carry of `CrossSectionalFactorPrior`.
- A graph verb that a caller gives a prior result and the raw returns data pairs the right rows
  after a Shared Series gap. A head that holds such a result gives the weights of the head that
  fits the same prior.
- A realised `factor_attribution` of the prior of an optimiser after a Shared Series gap matches
  each fold to the right row, also when the returns data carries no timestamps.
- An online fallback gives the batch weights after a Shared Series gap.
- `cor` of `ImpliedVolatility` takes the correlation of its inner estimator on `X`, and it discards
  the prediction from `iv`, which runs only so that `cor` refuses what `cov` refuses. So for a graph
  reader the pairing decides the validation of `iv` and not the answer: a wrong pairing showed as a
  false refusal of a non-finite `iv` (#1744).
- A wrapping prior that rebuilds a `LowOrderPrior` from named fields forwards `ss_gaps`, as it
  forwards `rr` and `fpr`.
- The rows of `src/10_Prior/01_Base_Prior.jl` grew past the size ceiling, so the Scenario Fill and
  Scenario Cap helpers moved unchanged to `src/10_Prior/19_PriorScenarios.jl`.
