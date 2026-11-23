---
status: accepted
---

# A concentration limit is a slot of the block, with its own binary over the threshold

## Context

The UCITS rule of Article 52(2) of Directive 2009/65/EC caps each issuer at 10 % of a fund, and
limits the sum of the issuers over 5 % to 40 %. Every UCITS mandate states it. The weight bounds
give the cap. The limit of the sum depends on which positions are over the threshold, so no
linear constraint gives it. Issue #2018 asked for the rule, and the maintainer settled its design
in a grilling session on 2026-11-23 (#2147).

## Decision

1. **A new pair of a constraint kind.** `ConcentrationLimitEstimator` configures a rule, and
   `ConcentrationLimit` is its resolved form, as `ThresholdEstimator` and `Threshold` are. The
   numbers `t`, `c` and `s` are keywords with the UCITS defaults 0.05, 0.10 and 0.40. The
   constructor refuses a value that is not finite, `t <= 0`, `c <= t` and `s <= 0`.
2. **A slot of the Constraint Block.** `ConstraintBlock` gains `conc`, typed `TD_Option`, which
   takes one rule or a vector of rules. A vector holds a mandate of several rules, such as a
   5/10/40 rule and a 5/25/80 rule for covered bonds. `JuMPOptimiser` and
   `ProgrammeAllocationSet` take it as a flat keyword. The clustering and the naive optimisers get
   no field, as they have none for a cardinality or a threshold.
3. **Its own binary.** The rule has one binary for each position, "the position is over `t`". It
   is not the held bit of `HeldIndicators` or `LongShortIndicators`, because "held" and "over `t`"
   are two facts. The rows are `x_j <= t + (c - t) z_j`, `y_j >= x_j - c (1 - z_j)`, `y_j >= 0`
   and `sum(y) <= s`. The feasible set is the rule itself. Each number is multiplied by the budget
   scale of `get_k`, as the weight bounds are. `declare_mip_gate!` declares the binary and its
   gate `z k`. The held bit, the long and the short bits, and the binary of the exact cost budget
   come from the same function, so the rows that link a binary to a free budget exist once
   (ADR 0021). The rule is a member of the MIP space family, `ConcentrationMIPSpace`: its
   `mip_wx!` gives the positions, which are a reading of `w`, `lw` and `sw`, and its `mip_key`
   names rule `i` `conc_<name>_i` under the prefix of the block. The space has no `mip_bounds`,
   because the rule states its cap on the positions. `mip_wb` does not serve the rule, because it
   holds a position at zero when its bit is off, and the rule lets it rise to `t`.
4. **The positions.** The field `set` takes what `smtx` takes. `nothing` makes each asset a
   position. An `AssetSetsMatrixEstimator` or a matrix makes each row a position, such as an
   issuer, so two share classes of one issuer are one position and the cap is on the issuer. The
   rule resolves at stage 1 of the block, as `smtx` does, and the view at the Investable Mask
   cuts the columns of its matrix.
5. **The filter `on`.** `AssetSetsMatrixEstimator` gains the field `on = nothing`. It names
   assets or groups of `sets.dict`, and only the assets that it names give rows. It resolves each
   name through `name_to_val!`, so an unknown name follows `strict`, and a name on the
   Non-Investable Axis drops with no message. The `smtx` and `sgmtx` slots gain the filter too.
   The exceptions of Article 52(3) to 52(5) are then more rules, and a government issuer takes a
   group bound.
6. **The reading of a short position.** The field `pos` takes `GrossPosition()`, the default,
   `NetPosition()` or `LongPosition()`. The gross reading reads `lw + sw`, the net reading `w`
   and the long reading `lw`. A long-only book has no short part, and there the three read `w`.
   The abstract supertype `AbstractPositionReading` is not exported. The name avoids the word
   "exposure", which the factor exposures use.
7. **A safe form with no integer.** The field `alg` takes `ExactConcentration()`, the default,
   or `SumLargestConcentration()`. The linear form bounds the sum of the `k = ceil(s / t)` largest
   positions by `s`, with the cap rows. A book that meets it meets the rule. It is stricter than
   the rule: four positions at 10 % and six at 5 % meet the rule and fail the form. The form "the
   sum of the 4 largest positions is at most 40 %", which #2018 suggested, is not safe: eight
   positions at 9 % meet it with a sum of 72 % over 5 %.
8. **A check on any book.** `concentration_margins(cl, w; sets = nothing)` returns a
   `ConcentrationMargins` of the cap margin of each position and the sum margin. A margin `<= 0`
   passes. It serves the mandate check of a book from any optimiser.
9. **NCO.** `assert_internal_optimiser` refuses a rule whose `set` is a matrix in an inner
   optimiser, as it refuses a precomputed `lcse` and `gcarde`, because the columns of the matrix
   belong to the full universe.

## Consequences

- A finaliser that projects the weights of a clustering or a naive optimiser onto a block is a
  new feature that reaches every slot of the block. It is #2257, with its decision #2258.
- `asset_sets_matrix` on an estimator takes the keyword `strict`. The printout of a
  `ConstraintBlock` and of an `AssetSetsMatrixEstimator` gains one line.
- The gross reading can be stricter than `|w|` when the short budget `sbgt` pins the sum of the
  parts and `xbgt` is off. The docstring of `GrossPosition` states it.
