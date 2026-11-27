---
status: accepted
---

# A failed sub-portfolio is a non-investable synthetic asset

## Context

One failed sub-portfolio failed the whole meta-optimiser. A `Stacking` whose second candidate of
three was infeasible gave `NaN` weights. A `NestedClustered` with one infeasible cluster did the
same at each asset of that cluster, and a `SubsetResampling` at each asset of a failed subset. The
fallback of the meta-optimiser then replaced every sub-portfolio that solved. The one remedy was a
fallback on the inner optimiser, which substitutes a portfolio that the caller must name.

The outer solve already does the work. A synthetic asset whose returns are `NaN` is outside the
Investable Mask of the outer prior. So it leaves the bounds, the constraints and the sets of the
outer problem, and its weight is zero. Two lines alone made the meta-optimiser fail: the product
`wi * w`, because `NaN * 0` is `NaN`, and the test that turned a success into a failure when one
inner code was a failure.

## Decision

**A failed sub-portfolio is a synthetic asset with no returns, and `min_solved` states how many
sub-portfolios must solve.**

| Item | Rule |
| --- | --- |
| `min_solved` | A `CountOf` or a `FractionOf` of the sub-portfolios, on `NestedClustered`, `Stacking` and `SubsetResampling`. A count above their number asks for every one. A fraction compares the share that solved, and rounds no count. The field takes a `TimeDependent` schedule. |
| The default | `FractionOf(1)`, every sub-portfolio. |
| Solved | `any_solved` of the return code that the fallback chain of the sub-portfolio ended with. The inner fallback substitutes, then the meta-optimiser drops, then the fallback of the meta-optimiser replaces. |
| `NestedClustered`, `Stacking` | The column of a dropped sub-portfolio is zero in the inner weights and `NaN` in the outer returns. The second rule is necessary for a candidate whose folds solved and whose full-sample solve did not. |
| `SubsetResampling` | The average runs over the subsets that solved, so it keeps the budget. Each point of a frontier drops its own subsets and has its own code. |
| The return code | With enough sub-portfolios, an `OptimisationSuccess` whose `res` is the payload of a failure, and one warning. With fewer, the failure and the weights of a call with no drop. With none, a failure under every `min_solved`. |
| A failed fold | A gap in the series of the candidate. The coverage policy of the outer prior decides it, and no code states a second rule. A scheme that holds the previous book through a failed fold gives the return of that book, and it stays. |
| No synthetic asset with a return | The outer problem has no asset, so its result is a failure with no solve, and the fallback runs. |
| A combinatorial candidate with no solved path | Its scorer has no path to select, so its column is `NaN`. |

Two collapses met the weights of a failed solve with an error, and a call with `iv` in the returns
data raised before any fallback ran. `synthetic_asset_weights` gives a column of zeros for weights
that are not finite. An entry of a collapsed `ivpa` for a column of zeros is the mean of `ivpa`,
because a Returns Result refuses a zero there. No reader uses either value.

## Considered options

- **A policy type, `DropFailed()` or `ReplaceFailed(opt)`.** Rejected: the second is the fallback of
  the inner optimiser under a second name, and the first needs the minimum count, which is the field.
- **Drop the columns, and view the outer optimiser on the others.** Rejected: the names `_k`, the
  sets, the bounds, the previous weights and `scale` each need a view. The Investable Mask does it.
- **Equal weights for a failed cluster.** Rejected: it is a substitution that the caller did not ask
  for, and `fb = EqualWeighted()` on the inner optimiser states it.
- **`CountOf(1)` as the default.** Rejected: a frontier sweep keeps its points because each point has
  its own code. A meta-optimiser has one code for the whole, so a caller asks for a success on a part
  of the configured sub-portfolios. For `NestedClustered` it is a portfolio on a smaller universe.

## Consequences

- The default changes no released answer. Calls that raised now give a failed result, and their
  fallback runs: a failed inner solve with `iv` in the data, and a meta-optimiser whose every inner
  solve fails.
- A combinatorial candidate with no solved path raised too. Now the outer problem drops it, as it
  drops a candidate whose every fold failed under another scheme. The return code reads the
  full-sample solve of the candidate alone. When that solve succeeds, the result is a success with
  no payload and no warning, under every `min_solved`.
- An outer prior whose every synthetic asset has a gap in a part of its series still raises its own
  `IsEmptyError`, as it does for a universe of real assets. The coverage policy of that prior is the
  remedy.
- A `NestedClustered` gives zero to each asset of a dropped cluster. A lower bound above zero on
  such an asset acts in the weight finaliser, which moves weight to the asset or fails.
- A success can hold a payload in `res`, so a reader that tests `isnothing(retcode.res)` sees it.
- An inner frontier sweep stays refused in `NestedClustered` and `Stacking`. A sub-portfolio is one
  column.
