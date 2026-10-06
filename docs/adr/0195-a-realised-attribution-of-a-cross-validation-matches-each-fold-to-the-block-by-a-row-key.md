---
status: accepted
---

# A realised attribution of a cross-validation matches each fold to the block by a row key

## Context

`factor_attribution(pred::MultiPeriodPredictionResult, pr)` decomposes the out-of-sample series
of a cross-validation against the factor model block of a prior result. The usual `pr` is one fit
of the prior over the whole sample. It is a Result that the attribution reads after the
cross-validation, not a prior that an estimator holds: inside the cross-validation, each fold fits
its own prior on its own training window. A fold's prior covers its training rows alone, so a
realised attribution of the test rows needs a block whose rows cover them.

Before this decision the method stacked the fold series and lined it up with the block at the
tail, which needs a series at least as long as the block that ends on the last block row.
[#1493](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1493) measured the normal
case on the large parity fixture, 250 × 40 with an exposure lag of one, under
`IndexWalkForward(120, 40)`. The series covers data rows 121 to 240, and the block covers data
rows 2 to 250. The method refused with a `DimensionMismatch`, and a tail alignment would be wrong
for a longer series too, because the series ends at row 240 and the block at row 250. Neither side
recorded which data rows it describes:

- a fold rebuilds its returns as a new net series, so no view back to the data survives;
  `MultiPeriodPredictionResult.id` names a fold, not a row; `rd.ts` exists only when the data
  carries timestamps;
- `CrossSectionalFactorModel` recorded no rows. Its fit drops the Descriptor warm-up and the lag.

[ADR 0045](0045-a-feature-matrix-is-data-not-estimator-configuration.md) had chosen to recover a
fold's rows from its timestamps rather than store them on `PredictionResult`, and to refuse a
missing `ts` only where it is load-bearing.

## Decision

**Both sides carry a row key, and the method matches the keys.**

1. `CrossSectionalFactorModel` gains `idx`, the position of each fit row in the returns data that
   the prior read, and `ts`, the timestamp of each fit row when that data carries timestamps. The
   batch fit records `rw[r]`, the rows after the warm-up and the lag. The carry fold records the
   positions among every observation it folded, and no timestamp, because its buffer keeps none.
   A time-cut of the block cuts the key with it. Both default to `nothing`, so a block built by
   hand keeps the old behaviour.
2. `PredictionResult` gains `idx`, the `test_idx` of the fold. The indexed `predict` records it.
   The whole-sample `predict` records `nothing`, because it cannot know where its returns data
   came from. The timestamps stay on `rd.ts`.
3. The method matches by timestamp when the block and the series both carry timestamps, and by
   position otherwise. A timestamp names an observation in every returns data that holds it. A
   position names it in one returns data alone, so the positions agree only when the prior read
   the returns data that the cross-validation split, and the method cannot check that.
4. **The lag is applied on the whole block first, and the block is cut to the matched rows
   second.** The first test row therefore keeps the exposures of the row before it.
5. Test rows before the first row that the block decomposes, or after its last row, are left out.
   A test row between two matched rows that the block does not hold is refused, and so is a series
   that repeats a block row or does not keep its order.
6. With no shared key, the method lines the series up at the tail, as before.

A block of another type, `Regression`, records no key. It keeps the tail rule.

## Consequences

- The case of #1493 answers in one call and keeps data row 121. With the exposure lag of one, row
  `t` splits exactly as `r_p,t = w_t' B_{t-1} f_t + w_t' ε_t`, and the full block holds `f_121`,
  `ε_121` and `B_120`. A rule that cuts the block to the test rows before it applies the lag
  removes `B_120` and loses the row, although the row is inside the factor-model window.
- On the rows that both rules keep, the two agree.
- The key reaches every scheme whose folds the indexed `predict` makes: `KFold`, the
  walk-forwards, `HindsightSplit`, each path of `CombinatorialCrossValidation` and of
  `MultipleRandomised`, an online run and its `Resume`, and the rows of a
  `BudgetedHindsightPath`. A population of paths takes one attribution per path. A path of
  `MultipleRandomised` holds a subset of the assets, so it is attributed against a prior fitted
  on the same subset, and weights over another universe are refused with a `DimensionMismatch`.
- `CrossSectionalFactorModel` and `PredictionResult` each carry one more type parameter, after the
  existing ones, so a dispatch that names a prefix of the parameters is unchanged.
- The cost is one integer vector, a range on the batch fit, on each block, the timestamps of the
  fit rows when the data carries them, and one range on each fold.

## Rejected alternatives

- **Timestamps only, as ADR 0045 decided for the Feature Matrix.** The attribution of a
  walk-forward over data with no timestamps would stay refused, and the oracle answers that case.
  [ADR 0186](0186-an-oracle-mode-is-built-when-a-caller-cannot-reach-its-output-and-five-differences-are-deliberate.md)
  builds a mode of the oracle that a caller cannot reach.
- **A `rows` keyword on `factor_attribution`.** The caller would split the cross-validation again
  to pass it, and a wrong range gives wrong numbers with no error.
- **Positions only.** A prior fitted on other returns data, for example a longer sample, would
  match the wrong rows with no error. Timestamps remove that risk where they exist.
- **A full label array on every Result, always.** It pays the storage on every fit. A range costs
  two integers.
