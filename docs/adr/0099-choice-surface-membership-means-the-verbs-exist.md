---
status: accepted
---

# Choice Surface membership means the verbs exist

## Context

A moment estimator joins a Choice Surface by subtyping it, and the library exports the leaf.
Those two acts are the whole contract a caller relies on: the surface's docstring says that a
covariance estimator answers `cov` and `cor`, that a variance estimator answers `var` and
`std`, and that an expected returns estimator answers `mean`.

Two exported leaves satisfy the type and not the behaviour.

```julia-repl
julia> X = reshape(collect(1.0:12.0), 4, 3);

julia> cov(RegimeAdjustedExpWeightedCovariance(), X)
ERROR: StackOverflowError:
```

`RegimeAdjustedExpWeightedCovariance` declares no verb at all. Its docstring states the whole
mathematics of an online exponentially weighted covariance with a regime adjustment, and no
code implements it. `RegimeAdjustedExpWeightedVariance` declares `Statistics.var` alone while
it subtypes `AbstractCovarianceEstimator`, so the surface it joined asks it for two verbs it
does not have. All four pairings over the two types were measured, and all four overflow the
stack.

The crash rather than a `MethodError` comes from the surface's own fallbacks.
`Statistics.cov(ce::AbstractCovarianceEstimator, X)` reads `Statistics.cor(ce, X)` and
rescales the result. A leaf that owns no `cor` reaches `StatsBase`'s generic
`cor(::CovarianceEstimator, ::AbstractMatrix)`, which reads `cov(ce, X)` straight back into
that fallback. The pair is a cycle with no base case, so the interface is the same width as
every sibling's and the implementation behind it is empty.

Nothing caught the gap. The export is public API, the subtype declaration is one line, and
neither is measured. [ADR 0058](0058-the-dims-guard-and-the-orientation-are-one-call.md) met
the two leaves when its `dims` census tried to drive them, and recorded them in its Notes as
pre-existing defects it neither caused nor repaired. `test_08d_dims_guard.jl` carried a
hand-written skip that named both types and asked a later reader to delete it. That skip was
the only place in the repository that said the promise was broken, and it said so inside a
census about something else.

## Decision

**A leaf that joins a moment Choice Surface answers that surface's verbs, and the answer is a
method `PortfolioOptimisers` declares below the surface.**

`test_08l_moment_verb_census.jl` gates it. The rule names no leaf: it walks `subtypes` from the
three surfaces, cuts the leaves into three families, and asks each leaf for its family's verbs.
A leaf added in future is covered the day it is written — the closed polarity of
[ADR 0037](0037-model-state-accessor-interface.md)'s rules, which ADR 0058's two censuses
already copy.

### What counts as an answer

A leaf **owns** a verb when the method the call dispatches to is declared by
`PortfolioOptimisers` for a type strictly below the surface. Two readings are therefore not an
answer: a method from `StatsBase`, which is the generic `CovarianceEstimator` one, and a method
declared **on** a surface, which is a fallback the surface offers rather than the leaf's own
result.

The leaf types arrive from `subtypes` as `UnionAll`s, and dispatch on a `UnionAll` answers the
wrong method — `which(cov, Tuple{Covariance, Matrix{Float64}})` reads the surface's fallback,
because no single method covers every `Covariance{T1, T2, T3}`. A caller holds an instance, so
the census asks about `typeof(S())`.

### The covariance surface offers one fallback, and it has preconditions

`cov(ce::AbstractCovarianceEstimator, X)` is the **Correlation Rescale**: it reads `cor(ce, X)`
and rescales it by `std(ce.ve, X)`. A leaf may take it in place of its own `cov`. It may not
take it in place of its own `cor`, because there is no `cor` on the surface to take. So the
rule for the covariance family is that `cor` is owned, and `cov` is owned or rescaled.

The fallback reads `ce.ve`, and no type bound states that field. The census holds the
precondition instead: every leaf that takes the Correlation Rescale owns `cor` and carries a
`ve` field. Four leaves take it today.

### One exemption, and it is the subject of the gate

`VERB_EXEMPT` in `moment_family_setup.jl` names the two leaves, and
[issue #637](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/637) carries the
missing mathematics. The census also asserts that each exempt name **still fails**, so an
exemption that has been paid reds the build rather than standing as a decision. This is the
shape [ADR 0082](0082-the-coverage-terminal-condition-is-a-per-file-ratchet-and-a-named-exemption.md)
gave the Coverage Exemption: a named exemption with a reason, not a silent skip.

The mathematics is not invented here. The docstrings state an update, a correlation and a
rescale; they do not state the mean estimate, the initialisation, the bias correction, the HAC
treatment, or how each of the three targets computes its statistic. Those are the maintainer's
decisions, and there is no reference in the repository to check the numbers against.

### The split lives once

`moment_family_setup.jl` holds the family cut and the ownership predicate.
`test_08d_dims_guard.jl` reads them too: a pairing this census rejects cannot be driven at
`dims = 3` at all, because the call recurses through the surface's fallback and overflows the
stack before it reaches a guard. Both files reading one predicate is what stops the two
censuses drifting apart.

## Consequences

- The promise is measured. 27 concrete leaves over three families, 25 of which answer their
  verbs, and the two that do not are named once, with a reason and an issue.
- The hand-written skip in `test_08d_dims_guard.jl` is deleted. That file now names no leaf,
  and it takes its pairings from the predicate. Its census count is unchanged at 41 pairings,
  because the predicate admits exactly the leaves the skip admitted.
- A new leaf that declares no verb reds the build on the day it lands, rather than on the day a
  caller runs it.
- A leaf that takes the Correlation Rescale without a `ve` field reds the build, rather than
  raising a `FieldError` from inside the library.
- The two crashes stay crashes until #637 lands. The census does not repair them; it makes them
  impossible to acquire again.

## Scope

The census reads the three moment surfaces. It says nothing about `cov`/`cor` on the two
`AbstractVarianceEstimator` leaves, which recurse the same way and for the same reason: a
variance estimator answers `var`/`std`, both leaves own both, and giving the variance family a
covariance reading is a separate decision. ADR 0058's Notes record that case, and it stands.

## Related

- [0037](0037-model-state-accessor-interface.md) — the closed-rule polarity this census copies:
    a rule that names no leaf covers one added in future.
- [0058](0058-the-dims-guard-and-the-orientation-are-one-call.md) — its Notes recorded the gap,
    and its census carried the skip this one replaces.
- [0082](0082-the-coverage-terminal-condition-is-a-per-file-ratchet-and-a-named-exemption.md) —
    the named-exemption shape, and the ADR whose 0.0 % coverage case is these same two files.
