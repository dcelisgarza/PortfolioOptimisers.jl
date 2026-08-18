---
status: accepted
---

# A meta-optimiser's sub-portfolio enumeration is an adapter

## Context

A meta-optimiser that owns an outer optimiser runs the same module. It solves one inner
problem per sub-portfolio, predicts each sub-portfolio's returns, and hands the outer
optimiser a synthetic universe with one asset per sub-portfolio. Two ship: `NestedClustered`
and `Stacking`.

That module was written twice, under two names.
`predict_outer_nco_estimator_returns` and `predict_outer_st_estimator_returns` carried three
methods each — fold-less, non-combinatorial cross-validation, combinatorial cross-validation
— and the six were near-byte-identical. Each pair built the outer returns buffer from the
prior and the fees, ran the inner cross-validation in an `FLoops` loop, applied the scorer
default, and rebuilt the outer returns result.

The only variation is what enumerates a sub-portfolio:

- `NestedClustered` enumerates **cluster index sets**. One inner optimiser is viewed onto each
  cluster (`cross_val_predict(opti, …; cols = cl)`), and the prior and the fees are viewed onto
  it too.
- `Stacking` enumerates **inner optimisers**. Each sees the whole universe, so nothing is
  viewed.

`16_Base_MetaOptimisation.jl` had already extracted the parts where the two agree exactly:
`prepare_outer_rd`, `rebuild_returns_result`, `outer_optimisation_finaliser`. The extraction
stopped where one adapter parameter would have been needed. The difference was carried one
level down instead, as a `cls::Nothing` / `cls::VecVecInt` pair on `fold_weight_matrix` and an
`Option{<:VecVecInt}` sentinel threaded through `rebuild_returns_result` and
`rebuild_feature_matrix`. The seam existed; it was not pulled up.

The three cross-validation methods dispatched on the meta-optimiser's **type parameters** —
`Stacking{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:Any, <:OptimisationCrossValidation{…}}`
— which is a positional spelling of the `cv` field. Both docstrings told the caller to
"overload this using `nco.cv`", so the dispatch was on a proxy for the thing the contract
names. The positions are load-bearing and they move: ADR 0053 amendment 2 moved `Stacking`'s
`fb` to type parameter 13.

`cvi = !hasfield(typeof(cv), :rng) ? cv : copy(cv)` appeared at four sites — every occurrence
in `src/` — with no statement of why a copy is needed.

## Decision

The sub-portfolio enumeration is an adapter, `SubPortfolioUniverse`, and the module is written
once.

Two declarations:

- `FullUniverse` — sub-portfolios are the inner optimisers, each sees the whole universe.
  `Stacking` passes it.
- `ClusterUniverse(cls)` — sub-portfolios are cluster index sets, each sees its own cluster.
  `NestedClustered` passes it.

Four verbs read them back:

- `sub_portfolio_count(u, opti)` and `sub_portfolio_predict(u, opti, i, rd, cv, ex)` — the
  enumeration itself. The predict verb makes the whole `cross_val_predict` call, because the
  call *shape* is part of the difference: `ClusterUniverse` passes `cols = u.cls[i]`, and
  `FullUniverse` passes no `cols` at all. A colon is not a substitute — the arity that takes a
  precomputed optimisation result has no `cols` keyword, so passing one is a `MethodError`, and
  a `Stacking` whose `opti` holds a solved result is a shipped configuration (`test_22a`).
- `sub_portfolio_view(u, x, i)` — view a full-universe quantity (the prior, the fees) onto
  sub-portfolio `i`. `FullUniverse` returns it unchanged.
- `fold_weight_matrix(predictions, u, f, na)` — the padding at the outer collapse, which was
  already dispatching on this difference and now dispatches on the adapter that names it.

`predict_outer_returns` replaces both verbs, with the same three methods the two had six of.
**It dispatches on `cv`, first positional**, which is what the docstrings already told the
caller to overload; the meta-optimiser follows, so an overload may still narrow on it. The
sentinel is gone: `rebuild_returns_result` and `rebuild_feature_matrix` take a
`SubPortfolioUniverse` and no longer admit `nothing`.

`sub_portfolio_cv(cv)` is the RNG copy, stated once: the sub-portfolios run in parallel, and a
scheme that draws its splits from a generator holds mutable state that two of them advancing
at once would neither reproduce nor agree on.

## Consequences

`NestedClustered` loses 65 lines and `Stacking` 62. The base file gains 294, of which the
adapter family, the module and their docstrings are all new prose — the six methods deleted
carried two docstrings between them. A third meta-optimiser of this shape is a
`SubPortfolioUniverse` declaration and a call, not a third copy.

The rename is breaking, and deliberately loud. An overload of either old verb is a method of a
function that no longer exists, so it is a `MethodError` on a name, not a silently unreached
method. `docs/src/api/20_Optimisation/17_NestedClustered.md` and `18_Stacking.md` drop their
entry; `16_Base_MetaOptimisation.md` gains the family.

No assembled problem changes. The fold-less path computes the same net returns from the same
views, the two cross-validation paths make the same `cross_val_predict` calls with the same
`cols`, and `fold_weight_matrix` pads exactly as before — `FullUniverse` where `cls` was
`nothing`, `ClusterUniverse` where it was a vector.

ADR 0045 (the feature-matrix collapse) and ADR 0053 (the Combination Weight acts at the
combination, and the outer problem never sees it) are untouched: this moves where the
difference is stated, not what either path computes. ADR 0030's rule that `cv` is execution
control still holds, and `rebuild_returns_result` still enforces it.
