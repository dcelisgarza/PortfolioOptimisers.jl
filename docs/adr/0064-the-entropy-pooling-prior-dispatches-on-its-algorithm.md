---
status: accepted
---

# The entropy pooling prior dispatches on its algorithm, not on a parameter position

## Context

`EntropyPoolingPrior` carries seventeen type parameters. The last one is `alg`, the entropy
pooling algorithm. Two `prior` methods split the estimator by that algorithm: `StagedEP`
(`H1_EntropyPooling` and `H2_EntropyPooling`) enforces the views in stages, and
`H0_EntropyPooling` enforces them all in one optimisation.

Each method reached `alg` by counting. It wrote `<:Any` sixteen times, then the algorithm
bound:

```julia
function prior(pe::EntropyPoolingPrior{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                       <:Any, <:Any, <:StagedEP}, X::MatNum, …)
```

The count was written at four sites: the two method heads, the two hand-written docstring
signatures, and the hand-written API page `@docs` block. Two of the four were already wrong.
Both docstrings wrote fifteen placeholders, not sixteen, so the signature they showed binds
the algorithm to parameter sixteen, `w`, the prior observation weights. The docstrings also
showed `F` as a keyword argument while both methods take it positionally.

The failure mode is silent. A partially applied `EntropyPoolingPrior{…}` is a legal type with
its remaining parameters free, so a field added anywhere before `alg` does not raise. The
constraint slides onto the neighbouring field, the method stops matching every estimator it
used to match, and the fallback is a `MethodError` at a call site far from the edit.

The count also bought nothing. The two methods partition the estimator exhaustively, and both
carry the same argument list.

Sibling estimators in the same directory already dispatch on the algorithm as a value:
`compute_pooling(pe.alg, ow, pw)` in `11_OpinionPoolingPrior.jl`, and
`phylogeny_features(ze.alg, ze.pl, X)` in `13_FeaturePrior.jl`.

Raised as "the `<:Any`×16 dispatch" in the architecture review of 2026-08-17.

## Decision

`prior` has one method on the bare `EntropyPoolingPrior`, matching the shape every other
prior estimator in `src/13_Prior/` uses. It orients the data and forwards the algorithm as a
value:

```julia
function prior(pe::EntropyPoolingPrior, X::MatNum, F::Option{<:MatNum} = nothing;
               dims::Int = 1, strict::Bool = false, kwargs...)
    X, F = dims_oriented(dims, X, F)
    return ep_prior(pe.alg, pe, X, F; strict = strict, kwargs...)
end
```

`ep_prior` takes the algorithm first and carries the two bodies unchanged:

- `ep_prior(alg::StagedEP, pe, X, F; strict, kwargs...)`
- `ep_prior(alg::H0_EntropyPooling, pe, X, F; strict, kwargs...)`

`ep_prior` is unexported and documented, like the `ep_*_views!` family it sits with.

Two consequences of moving the split follow from the seam, not from taste:

1. The orientation runs once, in `prior`. `ep_prior` receives oriented matrices and takes no
   `dims`, so ADR 0058's fused guard keeps its single call site per entry point.
2. The staged body reads its `alg` argument, not `pe.alg`. `pe` is rebound by `factory` inside
   that body, so the argument is the stable spelling.

## Consequences

A field added to `EntropyPoolingPrior` cannot unbind the split. Nothing counts parameters, so
there is no count to update and no drifted copy to find.

Behaviour is unchanged. The same two bodies run under the same conditions, on the same
arguments, and the estimator is exhaustively partitioned by `StagedEP` and
`H0_EntropyPooling` exactly as before.

The dispatch is one hop deeper. A `MethodError` for an algorithm outside the two names now
reports `ep_prior` rather than `prior`, which names the thing that failed to match.

`docs/src/api/13_Prior/10_EntropyPoolingPrior.md` lists three entries where it listed two: the
single `prior` method and the two `ep_prior` methods. The two drifted docstring signatures are
corrected as part of the move.

Nothing in ADR 0046 (`forward_prior`) or ADR 0058 (the fused `dims` guard) is contradicted.
