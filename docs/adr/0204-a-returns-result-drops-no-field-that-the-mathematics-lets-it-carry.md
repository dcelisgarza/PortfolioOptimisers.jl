---
status: accepted
---

# A Returns Result drops no field that the mathematics lets it carry

## Context

The benchmark census of #1632 found two defects with one cause: code that builds a `ReturnsResult`
from another one and leaves a field at `nothing`.

- [#1645](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1645): the online step of
  every optimiser that holds a prior folded the raw `rd`, while the batch path first ran
  `returns_result_picker(rd, brt)`. With `brt = true` the prior folded `X` and the batch fit read
  `X - B`, so online weights differed from batch weights by up to `7e-2`, with no error.
- [#1646](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1646): the rows window of an
  online portfolio selection head kept `nx`, `X` and the active mask, and dropped every other field.
  A follow-the-leader leader with `brt = true` never saw a benchmark. `returns_result_picker` then
  ignored `brt` on returns with no `B`, so the leader fitted the absolute returns with no signal.

`returns_result_picker` also dropped `nb` and `B` after the subtraction, so that a second pick could
not subtract the benchmark twice. Every pick that met data with no `B` therefore had to pass through
in silence, and a silent pass-through hides a forgotten benchmark.

The maintainer stated the rule on 2026-10-12: the quantities of a `ReturnsResult` are never dropped,
unless they truly cannot be carried for mathematical reasons.

## Decision

1. **A function that builds a `ReturnsResult` from another one carries every field.** A field is
   dropped only when it has no meaning on the new axis, and the code says why in a comment or its
   docstring. Examples of such a reason: an asset-indexed field after a collapse to synthetic
   assets with no collapse rule, or a window whose rows hold no such field upstream. A refusal by
   name is acceptable where a field cannot travel. A silent drop is not.
2. **The benchmark pick moves the data to the Excess Frame and drops nothing.**
   `returns_result_picker(rd, true)` returns `X - B` as the returns, keeps `nb`, and carries
   `B - B` as the benchmark. The benchmark's excess return over itself is zero where `B` is finite
   and `NaN` at a gap. So a second pick subtracts zero and changes no value, and an inner optimiser
   of a meta-optimiser or the read-out of an online step can pick again.
3. **`brt = true` on returns that hold `X` and no `B` is an `ArgumentError`.** No excess return
   exists without a benchmark. Returns data with no `X`, which the empty `ReturnsResult` of a
   prior result is, passes through, because it holds nothing to pick.
4. **The online step of an optimiser picks before it folds** (`fold_returns`). The prior and the
   Fold Context then hold the Excess Frame, and `optimise(opt)` with no returns equals
   `optimise(opt, rd[1:t])` under `brt` too (ADR 0137).
5. **The rows buffer of an online portfolio selection head is a `ReturnsBufferState` that owns the
   returns**, as the Fold Context of a head with no prior is. The window that every rule, leader
   and programme set reads carries every field of the folded rows: `nf`, `F`, `nb`, `B`, `ne`, `E`,
   `ts`, `iv`, `ivpa` and the static panel. So the head no longer refuses `iv`. Every prior that
   the head fits on those rows reads the surface: the prior of a programme set, the prior of a
   Risk Loss, and the prior of a forecaster. The refit arm of a forecast rule gives `iv` and
   `ivpa` to every forecaster, as `prior` over a `ReturnsResult` gives them to every member
   ([#1644](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1644)). A
   `PriorExpectedReturns` gives them to its prior, a `WindowedExpectedReturns` cuts them to its
   window, and every other forecaster ignores them. The covariance step of
   `covariance_forecast_evaluation` gives them to a buffer, and an exact fold, which holds numbers
   alone, ignores them as its batch fit does
   ([#1643](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/1643)).

## Consequences

- A leader, a programme set or a fallback of the head reads the rows as a batch fit over the same
  window reads them. A leader with `brt = true` fits the excess returns.
- A caller that sets `brt = true` and forgets `B` gets an error at the first pick, in batch and
  online alike.
- After a pick, `rd.B` is zero (or `NaN` at a gap of `B`) and `rd.nb` names the benchmark. No
  library code reads `B` after a pick except a second pick.
- The window of the head is views into its buffers, as before, so the step takes no copy.

## Alternatives considered

- **Keep `B` unchanged after the pick, and mark the data as picked.** A mark is a new field of
  `ReturnsResult` or a new type, and every constructor, view and fold would carry it. The Excess
  Frame needs no mark, because a zero benchmark is its own fixed point.
- **Drop `nb` and `B` after the pick, as before.** This forces the silent pass-through of option 3
  above, and it breaks the rule.
- **Refuse `brt = true` on a follow-the-leader leader** (option 1 of #1646). It hides the defect for
  one caller, leaves the head without a benchmark-relative leader, and keeps the silent no-op for
  every other caller.

## Related

- ADR 0137: the online step forwards to the prior and the read-out runs the batch path.
- ADR 0164: the Allocation Set Constraint through which a leader honours the head's set.
- `returns_result_picker`, `fold_returns`, `buffer_returns_result`, `ReturnsBufferState`.
