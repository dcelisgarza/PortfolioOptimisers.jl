---
status: accepted
---

# A view set that states nothing answers the prior

## Context

`strict = false` is the documented "warn and continue" path of every view family. A view row that
names no asset in the universe is reported by `strict_diagnostic` and dropped, and
[`get_linear_constraints`](../../src/12_ConstraintGeneration/02_LinearConstraintGeneration.jl)
answers `nothing` when **every** row of a group is dropped that way.

Five view verbs read a block off that answer without checking it: `ep_mu_views!`,
`ep_var_views!`, `ep_sigma_views!`, `ep_sk_views!` and `ep_kt_views!`, all in
[`10_Base_EntropyPoolingPrior.jl`](../../src/13_Prior/10_Base_EntropyPoolingPrior.jl), together with
`ep_cvar_views_setup` in
[`11_MeucciEntropyPoolingPrior.jl`](../../src/13_Prior/11_MeucciEntropyPoolingPrior.jl). The fit
therefore raised `FieldError: type Nothing has no field ineq`, one call after the warning that
already named the cause. Issue
[#852](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/852) holds the reproduction: a
view written against a universe of numeric asset names, where the parser reads `1` as the constant
rather than as the name of an asset.

Two families were already correct. `ep_cov_views!` and `ep_rho_views!` drop such a row inside their
own loop, under issue #538, and `ep_tail_views!` drops one through the `nothing` that
`ep_view_terms` answers. So the reading was settled for three families and unsettled for six.

The guard alone does not finish the question. When every family of a fit states no view, the
constraint dictionary is empty and the solve reweights nothing. Whether that answers the prior or
refuses is the decision issue #852 asked for, and it is the reason the issue was filed rather than
fixed.

## Decision

**A view group that states no view adds no row, and a fit whose every family states no view answers
the prior.**

The two halves are separate, and each is enforced where it belongs.

1. **A group that states no view adds no row.** Each of the six verbs returns as soon as the parse
   answers `nothing`: the two that answer `nothing` return `nothing`, the three that answer a fixing
   mask return a mask that names no asset, and `ep_cvar_views_setup` skips the group. A setup whose
   every group was skipped answers `nothing`, which is the answer it already gives when the caller
   states no conditional value at risk view at all, and which sends the stage down the plain solve.
   The warning `strict_diagnostic` already emitted is the whole diagnosis; the verbs add none.

2. **A view set that states nothing answers the prior.** `entropy_pooling` returns `w` when `epc`
   holds no row and `tvs` names no tail view. The posterior of an empty view set is the prior, and
   answering it directly rather than solving over the normalisation row alone keeps it exact: `kld`
   is zero rather than a rounding of zero.

3. **A stage solves on the rows it holds, not on the fields the caller set.** The four `ep_prior`
   bodies test `epc` and `tvs` — and, on the Meucci route, `cvv` — rather than testing whether a
   view field is `nothing`. A stage that states no view runs neither the solve nor the refit, so
   `factory(pe, w1)` does not push a uniform vector into an estimator whose weighted path disagrees
   with its unweighted one. [ADR 0116](0116-a-prior-that-reweights-observations-works-on-the-axis-its-nested-prior-answered.md)
   measured that disagreement: a `FactorPrior` over a `StepwiseRegression` selects a different
   factor set under a uniform `ProbabilityWeights`.

So the fit of issue #852 warns once and answers the wrapped prior unchanged: `w` is the prior
probabilities, `kld` is zero, `ens` is `T`, and `mu` and `sigma` are the nested fit's.

`strict = true` is untouched. There, `strict_diagnostic` raises on the first row that names no
asset, so the parse never reaches the guard.

## Alternatives considered

| Alternative | Shape | Why not |
| --- | --- | --- |
| **Refuse an empty view set** | Raise where every family states no view. | It converts a warning into a raise at a distance. The caller asked for `strict = false`, which is the documented "warn and continue" path, and they were already told which rows were dropped. A raise there states nothing the warning did not. |
| **Refuse an empty *group*** | Raise where one group states no view, and let a fit with another non-empty family proceed. | The same objection, and it splits the contract by family: `ep_cov_views!` and `ep_tail_views!` already drop such a group silently under #538, so this reading would need those two reversed as well. |
| **Solve the empty problem** | Guard the `FieldError` alone and let `entropy_pooling` run over the normalisation row. | It answers `w` to solver tolerance rather than exactly, so `kld` is a rounding of zero and `ens` a rounding of `T`. It also costs a solve and a refit of the nested estimator per stage, and that refit pushes a uniform vector into the estimator, which ADR 0116 showed is not a no-op. |
| **A `warn_empty` keyword** | A third setting between `strict = true` and `strict = false`. | `strict` already carries this axis. A third setting states the same choice twice, and every view verb and every fit body would have to thread it. |

## Consequences

+ The six verbs are now consistent with `ep_cov_views!`, `ep_rho_views!` and `ep_tail_views!`, so
  `strict = false` warns and drops in every view family without exception.
+ A fit whose every family states no view is indistinguishable from `prior(pe.pe, X, F, pnl)`, apart
  from the warnings. `kld` is exactly zero and `w` is exactly the prior probabilities, so a caller
  can read `iszero(pr.kld)` to find that no view survived.
+ A stage that states no view no longer refits the nested estimator. On the staged routes, a fit
  whose `sigma_views` all dropped but whose `mu_views` survived still re-solves the accumulated
  rows, because `epc` is then non-empty; that is the behaviour those routes always had.
+ `ep_cvar_views_setup` gains a second reason to answer `nothing`. Its `Returns` section names both.
