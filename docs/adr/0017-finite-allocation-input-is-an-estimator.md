# FiniteAllocationInput is an Estimator, not a Result

The finite-allocation optimisers (`DiscreteAllocation`, `GreedyAllocation`) previously took
their problem data as loose positionals: `optimise(da, w, prices, cash, T, fees)`. We bundle
that shared data into a single struct, `FiniteAllocationInput` (fields `w`, `prices`, `cash`,
`horizon`, `fees`), passed as `optimise(da, fai)`, and classify it as a subtype of
`AbstractEstimator` rather than of the `Result` tree.

This deviates from the codebase convention that user-supplied *data* structs live under the
Result tree (`WeightBounds` and `RiskBudget` both subtype `AbstractConstraintResult`; the
Estimator sibling is the thing that *computes* the data). The deviation is deliberate.

## Considered Options

- **`AbstractResult` (bare)** — matches the `WeightBounds`/`RiskBudget` precedent and grants
  the generic pretty-printer + `factory`. Rejected because it stretches the glossary's
  "Result = computed output" definition (this struct carries user inputs), and because being
  in the Result tree invites confusion with the *output* it is fed alongside.
- **`FiniteAllocationOptimisationResult`** — rejected outright: it chains to
  `OptimisationResult`, which is a broad dispatch magnet (~15 `plot_*` methods and
  `factory(res::FiniteAllocationOptimisationResult, …)` would silently capture an *input* and
  fail downstream on result fields it does not have).
- **`AbstractEstimator` (chosen)** — the struct is the primary positional input to
  `optimise`, which the glossary already frames as the role of an Estimator. It gets the same
  uniform printing/`factory` machinery (keyed on `Union{AbstractEstimator, AbstractAlgorithm,
  AbstractResult}`) with no harmful catch-all, and keeps the entire `Result` tree reserved for
  computed outputs — so allocation *inputs* can never collide with allocation *results*.

## Consequences

- `FiniteAllocationInput` is the one pure-data struct classified as an Estimator. Extension
  authors adding data structs should still follow the `WeightBounds`/`RiskBudget` precedent
  (data → Result tree) unless they have this same inputs-must-not-collide-with-outputs reason.
- The positional `optimise(da, w, prices, cash, …)` form is removed (breaking); this landed as
  part of the Phase-5 ergonomics release, so no deprecation shim was kept.
- No `factory`/`port_opt_view` methods: finite allocation is terminal post-processing, never
  sub-selected by meta-optimisers or Cross-Validation.

## Amendment (2026-11-25)

The allocators and the input have views, and no allocator records a cut
([#2348](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/2348)).

A caller who selects assets views the allocator and the input at one index, as the estimators of
the continuous optimisers and the returns data are viewed at one index.

- **The allocators** tag `unit`, `sets` and `fb` `@vprop`. The field `idx` of v0.34.7 goes. It held
  the positions that a view kept, so that a `LotUnitEstimator` resolved on the whole universe. No
  other estimator records a cut, and nothing compared the positions with the input. A name that a
  view left out is now a name outside the `sets` of the view, as in the view of every other
  constraint. The lots still resolve in `optimise`, in the number type of the input.
- **The input** lifts its fee onto the axis of `w`, views each term with its asset, and reduces it
  again at the view of `imsk`. So an exit charge is in the view of its asset alone, and the exit
  charges of the views of a partition add to those of the whole. The rule `upw` runs on the whole
  book before the cut. The cash and the collateral belong to the account, so a view keeps them. A
  view is the book in which only its assets trade, and not a part of the account. So the
  allocations of a partition do not add to the whole when a collateral cap binds: each view can
  tie up the whole cap.

Refused: a view that resolves the lots. It fixes the number type before the input is known, and it
reports a wrong name of a fallback that never runs. Refused: `lq` and `flq` whole in each view,
because two views of one book at disjoint assets then each charge all the exits.
