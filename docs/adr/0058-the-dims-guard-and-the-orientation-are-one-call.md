---
status: accepted
---

# The `dims` guard and the orientation are one call

## Context

Almost every estimator entry point takes a `dims` keyword. `dims = 1` says that the
observations lie along the rows of `X`, which is the orientation the whole library computes in;
`dims = 2` says that they lie along the columns. Two things follow from that keyword, and they
were spelled by hand, together, at 42 sites:

```julia
assert_dims(dims)
if dims == 2
    X = transpose(X)
end
```

A site with an optional second matrix spelled a nested branch as well:

```julia
assert_dims(dims)
if dims == 2
    X = transpose(X)
    if !isnothing(F)
        F = transpose(F)
    end
end
```

Because the guard was a separate statement from the orientation it guards, each leaf re-decided
whether to write it at all. Five did not, and they answered `dims = 3` with the *input*:

```julia-repl
julia> X = reshape(1.0:12.0, 4, 3);   # a statistic must be (1, 3)

julia> size(mean(SimpleExpectedReturns(), X; dims = 3))
(4, 3)
```

`Statistics.mean(X; dims = 3)` reduces over a singleton trailing axis, so a `dims` a user got
wrong reached no guard, threw nothing, and came back as a `4 × 3` matrix where the caller
expected a `1 × 3` row. The same held for `SimpleVariance`, `ExcessExpectedReturns`,
`StandardDeviationExpectedReturns` and `VarianceExpectedReturns`.

One user error therefore had four different outcomes: the library's `DomainError` at the 42
guarded sites, silence at those five leaves, `StatsBase`'s `ArgumentError: Argument dims can
only be 1 or 2` where `robust_cov`/`robust_cor` handed `dims` to a foreign estimator, and
`ArgumentError: mapslices does not accept dimensions > ndims(A) = 2` from
`MedianExpectedReturns{Nothing}`. The guard itself had drifted too:
`MedianExpectedReturns{<:ObsWeights}` hand-rolled `@argcheck(dims ∈ (1, 2), DomainError(dims,
"dims must be 1 or 2"))`, which drops the `dims => 3` echo the canonical message carries.

## Decision

**A caller cannot orient a matrix without validating `dims`, because orienting *is* the call
that validates.**

`dims_oriented(dims, A)` asserts through `assert_dims` and returns `A` when `dims == 1` or
`transpose(A)` when `dims == 2`. `dims_oriented(dims, A, B, Cs...)` returns a tuple of them,
each oriented. A `nothing` passes through unchanged, so an optional matrix costs no branch of
its own. The 42 fused preambles become one line each:

```julia
X = dims_oriented(dims, X)
X, F = dims_oriented(dims, X, F)
```

A site that forwards `dims` downwards rather than orienting keeps the bare `assert_dims(dims)`
— it makes no orientation decision, so there is nothing to fuse. The five silent leaves and
`MedianExpectedReturns{Nothing}` gained that call, and `robust_cov`/`robust_cor` gained it at
the boundary where `dims` leaves the library for `StatsBase`.

## Consequences

- 42 preambles become 42 one-liners, and `src/` no longer contains the string `if dims == 2`.
- A `dims` outside `(1, 2)` is one `DomainError` with one message. Verified by running every
    concrete moment estimator: 41 of 41 estimator-and-verb pairings throw it.
- Five silent wrong answers are closed. `mean(SimpleExpectedReturns(), X; dims = 3)` now
    throws instead of returning `X`.
- `MedianExpectedReturns{<:ObsWeights}` reports `dims must be 1 or 2. Got dims => 3` rather
    than the drifted `dims must be 1 or 2`.
- A new estimator cannot re-open the hole by omitting the guard, because the orientation it
    does write carries the guard with it. `test_08d_dims_guard.jl` locks both halves: a
    behavioural census derived from `subtypes`, and a source lint that rejects a hand-written
    `if dims == 2` or a hand-rolled `@argcheck` on `dims`. Neither names a site, so an
    estimator added in future is covered the day it is written — the closed polarity of
    ADR 0037's rules.
- The jldoctest examples in `01_Base_Moments.jl` that show a reader how to write a covariance
    estimator now teach the kernel rather than the two-statement preamble.

## Scope

`dims_oriented` handles the two-dimensional case, which is what `dims in (1, 2)` means. The
Feature Matrix keeps its own rules: a carried `Z` is canonically assets-major and has no `dims`
at all (ADR 0045), and the raw-matrix `distance(de, Z; dims)` entry point uses `dims` to
*retarget* from `X` to `Z` rather than to orient, so it keeps a bare `assert_dims`.

## Notes

Two pre-existing defects surfaced while the census was built. Neither is caused or repaired by
this change, and both reproduce at `dims = 1`:

- `RegimeAdjustedExpWeightedCovariance` declares no `cov` or `cor` method at all, and
    `RegimeAdjustedExpWeightedVariance` declares only `var`. The generic
    `cov(::AbstractCovarianceEstimator, X)` falls through to `cor`, which falls back to `cov`,
    so `cov` on either overflows the stack at any `dims`. `RegimeAdjustedExpWeightedCovariance`
    has no test of its own.
- `cov`/`cor` on the two `AbstractVarianceEstimator` leaves recurses the same way, for the same
    reason: a variance estimator answers `var`/`std`, and the covariance fallback assumes a
    `cor` method that is not there.

The census skips those pairings and says why, so the skip disappears when the missing methods
are declared.

## Amendment (2026-09-01)

The first Note's skip is gone from this census, and the gap it named now has a gate of its own.

[ADR 0099](0099-choice-surface-membership-means-the-verbs-exist.md) rules that a leaf which
joins a moment Choice Surface answers that surface's verbs.
`test_08l_moment_verb_census.jl` measures it, and `moment_family_setup.jl` holds the family cut
and the ownership predicate that both censuses read. The two leaves of the first Note are that
census's one exemption, named with a reason and with
[issue #637](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/637), and the census
asserts that each still fails, so the exemption cannot outlive the repair.

`test_08d_dims_guard.jl` therefore names no leaf any more. It takes its pairings from the
ownership predicate: a leaf that answers no verb reaches no guard to test, because the call
overflows the stack first. The census count is unchanged at 41 pairings, because the predicate
admits exactly the leaves the hand-written skip admitted.

The second Note stands. `cov`/`cor` on the two `AbstractVarianceEstimator` leaves still
recurse, and giving that family a covariance reading is a separate decision.

## Related

- [0037](0037-model-state-accessor-interface.md) — the closed-rule polarity the locks copy: a
    rule that names no keys covers an entry added in future.
- [0045](0045-a-feature-matrix-is-data-not-estimator-configuration.md) — why the Feature Matrix has no
    `dims` to orient by.
