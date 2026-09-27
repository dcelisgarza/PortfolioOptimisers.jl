---
status: accepted
---

# The iterative weight finaliser keeps its answer and falls back to the exact Euclidean projection

## Context

`IterativeWeightFinaliser` is the default `wf` of every naive estimator, `HierarchicalOptimiser`
(HRP, HERC, Schur HRP), `NestedClustered`, `Stacking`, `SubsetResampling` and
`BestConstantRebalancedPortfolio`. Issue #1256 showed three defects.

- **The loop can exit outside the bounds.** It stalls when no weight lies strictly inside its
  bounds after the clip: `[0.6, 0.4, 0.0]` under `ub = 0.4` stays at `[0.5, 0.5, 0.0]` for any
  `iter`. On long-short bounds it can diverge, because it spreads mass over a sum of mixed signs:
  one census instance reached `[277, -515, 933, -694]`. In 18 461 random long-short instances,
  14 674 exited infeasible.
- **`finalise_weight_bounds` tested finiteness alone**, so every such exit was an
  `OptimisationSuccess`, and the fallback chain never ran.
- **A scalar bound tested only the first weight.** The bound test was `any(map(>, lb, w))`, and
  `map` over a scalar and a vector stops after one element. The estimators resolve a scalar bound
  to a vector through `weight_bounds_constraints`, so only a direct call of `opt_weight_bounds`
  or `finalise_weight_bounds` met it. `JuMPWeightFinaliser` had the same test.

The exact Euclidean projection onto `{lb ≤ w ≤ ub, Σw = Σw₀}` is `clip(w₀ − θ, lb, ub)` at the
scalar `θ` that restores the budget. The budget is piecewise linear and monotone in `θ`, so the
root is found exactly from its kinks. The library already carries that search,
`breakpoint_root`, for the online Constrained Update. We
measured the projection against the loop before we chose a role for it.

- **It holds any net budget.** At `s = 1, 0.5, 0, −0.3, 2`, no instance of about 20 300 random
  long-short instances broke a bound or the budget.
- **It changes the answer where the loop was correct.** In 19 416 long-only instances where the
  loop reached the bounds, the two answers differed by a median of 0.012 and a maximum of 0.316.
  In 3 526 long-short instances they differed by a median of 0.088 and a maximum of 0.985.
- **It is not strictly better.** It is the closest point in L2, but the loop keeps the ratios of
  the free weights, and the projection moves each free weight by the same amount. The loop's
  answer equalled the exact proportional root `clip(c·w₀, lb, ub)` to `1e-9` in 73 % of the
  long-only instances. Under `ub = 0.5`, `[0.9, 0.09, 0.01]` becomes `[0.5, 0.45, 0.05]` from the
  loop and `[0.5, 0.29, 0.21]` from the projection. The smallest weight grows twenty-one times,
  which is a large change to a risk-parity allocation.

The exact entropic projection is `clip(w₀ / Z, lb, ub)` at the scalar `Z` that restores the
budget. It scales every free weight by one factor, so it is the exact form of the loop's intent,
and the online code already carries it for `EntropicProjection`. It is defined only for
non-negative weights, a non-negative `lb` and a positive budget: with mixed signs the budget is
not monotone in `Z`. A zero weight cannot take mass under any `Z`, so the stall case above,
`[0.6, 0.4, 0.0]` under `ub = 0.4`, has no entropic point at all.

## Decision

1. **A new finaliser, `EuclideanWeightFinaliser`, returns the exact Euclidean projection.** It is
   not the default, because it changes answers the loop gets right. It is open to every estimator
   that takes a `wf`, because it holds any budget and long-short bounds.
2. **A new finaliser, `EntropicWeightFinaliser`, returns the exact entropic projection.** It keeps
   the ratios of the free weights, as the loop does, and needs no passes. Outside its domain, it
   returns the Euclidean projection, so it always gives a feasible vector when one exists. It is
   not the default either: it still moves the answer in the 27 % of the long-only instances where
   the loop converged to a different point.
3. **`IterativeWeightFinaliser` stays the default and keeps its answer.** If its last pass still
   breaks a bound or is not finite, it returns the Euclidean projection of its input instead. A
   loop that reaches the bounds returns its own vector, bit for bit.
4. **`finalise_weight_bounds` reports an `OptimisationFailure` when the finalised weights are not
   finite, break a bound, or leave the budget of the input**, each to `sqrt(eps(eltype(w)))`
   (`weights_meet_bounds`). After the fallback, the only case left is a bound set that cannot hold
   the budget, `Σ lb > Σw₀` or `Σ ub < Σw₀`. The failure lets the fallback chain run.
5. **Every finaliser tests the bounds by broadcast** (`weights_break_bounds`), so a scalar bound
   is tested against every weight.

## Considered options

- **Make the projection the default.** Refused: it is not strictly better. It changes the answer
  of every default estimator whose bounds bind, and it discards the ratios the loop keeps.
- **Keep the projection exclusive to the hierarchical optimisers.** Refused: the restriction had
  one reason, a projection that could not hold other budgets, and the measurement shows that it
  holds them.
- **Make the loop fall back to the entropic projection.** The loop fails in two cases: a stall,
  where a zero weight must take mass, and long-short bounds. The entropic projection is not
  defined in either case, so the fallback would be the Euclidean projection every time.
- **Refuse a long-short input in `EntropicWeightFinaliser`,** as the online `EntropicProjection`
  does with a `DomainError`. Refused: a finaliser runs after an estimator has done its work, and
  a throw there loses the answer. The Euclidean projection is the closest feasible point in the
  other geometry, and the docstring states the switch.
- **Throw on a bound set that cannot hold the budget,** as `assert_feasible_bounds` does for the
  online Constrained Update. Refused: a finaliser answers through its return code, and a failure
  code lets the fallback chain run where a throw stops the estimator.

## Consequences

- Every estimator whose default loop stalled or diverged now returns weights in their bounds. Its
  answer moves from a vector outside the bounds to the Euclidean projection.
- Every estimator whose bounds cannot hold its budget now reports an `OptimisationFailure`, where
  it reported an `OptimisationSuccess` over a vector outside the bounds.
- A `JuMPWeightFinaliser` whose solve fails falls back to the iterative finaliser, and so to the
  projection when the loop fails.
- The `opt_weight_bounds` method of `IterativeWeightFinaliser` moved to
  `src/17_Optimisation/11_ProjectionWeightFinalisers.jl`, beside the two projections.
