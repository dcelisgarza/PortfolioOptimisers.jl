---
status: accepted
---

# A group pair view is one row per spanned pair, and `prior(gA, gB)` resolves per pair

## Context

An entropy pooling correlation or covariance view names an asset pair: `"(AAPL, XOM) == 0.35"`.
It can also name a **pair of groups**, `"(gA, gB) == 0.35"`. `replace_group_by_assets` expands
that form into one variable holding two index lists, and it requires the two groups to have equal
length. The pair list is `zip(i, j)`, so a group pair view **spans** one asset pair per element.

The constraint side has always read the view that way. `ep_rho_views!` and `ep_cov_views!` build
`view(X, :, i) .* view(X, :, j)`, which is one column per spanned pair, and they hand
`add_ep_constraint!` one row per pair.

The prior side did not. A view can state its target relative to the prior, `"(gA, gB) ==
1.5*prior(gA, gB)"`, and `get_pr_value` resolved `prior(gA, gB)` to a single number:

```julia
LinearAlgebra.norm(StatsBase.cov2cor(pr.sigma)[i, j]) / length(i)
```

Three things are wrong with that number, and the third is why this is an ADR and not only a fix.

1. **It aggregates.** One value reached every row, so a multiplier applied to none of the pairs.
   On a two-pair view whose prior correlations were `0.1074` and `0.0377`, the multiplier `1.5`
   sent both rows to `0.1207` — `1.12` times the first prior and `3.20` times the second.
2. **It indexes as an outer product.** `rho[[1, 2], [3, 3]]` is a 2x2 matrix whose two columns are
   both column 3, not the two pairs `(1, 3)` and `(2, 3)`. A group that names an asset twice
   counted that asset twice inside the norm, and the norm over `length(i)` is not the mean of the
   entries either.
3. **It carried no dispatch tag.** The single-pair methods dispatch on `Val{:rho}` and `Val{:cov}`
   and read the correlation and the covariance respectively. The group method took neither, so a
   `cov_views` group view was answered with a **correlation**.

Issue #403 records the reproduction. The docstring sweep of the entropy pooling prior, #379, found
it. Nothing in `test/` exercised a group pair view, so no test pinned the old behaviour.

Three readings of a group pair view are defensible.

- **Per pair.** Each spanned pair takes its own `prior(...)` value, scaled by the view.
- **Aggregate.** The view constrains one summary of the block, and the **constraint** side changes
  to emit one row on that summary.
- **Refuse.** A `prior(...)` reference inside a group pair view raises, and the caller writes the
  target out.

## Decision

### A group pair view is one constraint row per spanned pair

This is what the constraint side already built, and it is now the stated meaning rather than an
accident of the loop. The aggregate reading is rejected because no aggregate of a correlation
block has a settled definition here — a mean, a norm and a weighted mean are all summaries, and
picking one silently is how the old number came about. The refusal reading is rejected because the
relative form is the useful one: a caller who already knows every pair's prior value does not need
the reference.

### `prior(gA, gB)` resolves to one value per spanned pair

`get_pr_value` returns a vector in the order of `zip(i, j)`, one entry per row:

```julia
function get_pr_value(pr::AbstractPriorResult, i::VecInt, j::VecInt, ::Val{:rho}, args...)
    rho = StatsBase.cov2cor(pr.sigma)
    return [rho[a, b] for (a, b) in zip(i, j)]
end
```

### The group methods carry the same dispatch tags as the single-pair methods

There is a `Val{:rho}` method and a `Val{:cov}` method, and the untagged method is gone. A `cov`
view is answered with a covariance, and a `rho` view with a correlation, at both arities.

### A `RhoParsingResult` may carry a vector right-hand side

One right-hand side per row. `replace_coprior_views` subtracts the resolved vector from the
scalar `rhs` the parser produced, so a view that references a prior widens to the vector form and
a view that states a plain number keeps the scalar. The constructor holds the invariant: a vector
`rhs` needs every entry of `ij` to be a group pair of the same length, and a mismatch raises a
`DimensionMismatch` at construction rather than a shape error deep inside the model build.

## Consequences

- A group pair view that references a prior now sends each spanned pair to its own target. This
  changes the posterior of any caller who wrote one. No test pinned the old value, and the old
  value was not the quantity the reference named.
- `[-1, 1]` validation of a correlation right-hand side runs over every entry, so one bad pair
  fails the whole view.
- The three group readings are settled here. A caller who wants an aggregate view states the
  aggregate as a view of its own; the constraint side grows no summary row.
- A group whose two sides have different lengths is still refused by `replace_group_by_assets`,
  which is where that check has always lived.
