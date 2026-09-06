---
status: accepted
---

# A prior that reweights observations works on the axis its nested prior answered

## Context

Three prior estimators reweight observations: [`EntropyPoolingPrior`](../../src/13_Prior/12_EntropyPoolingPrior.jl),
[`MeucciEntropyPoolingPrior`](../../src/13_Prior/11_MeucciEntropyPoolingPrior.jl) and
[`OpinionPoolingPrior`](../../src/13_Prior/13_OpinionPoolingPrior.jl). Each one fits a nested prior
and pools a probability vector over the scenarios that fit produced.

Every one of them built its probability vector over the rows of the matrix it was **handed**, and
[ADR 0114](0114-the-asset-panel-travels-as-the-third-positional-argument-of-the-returns-matrix-prior-method.md)
left that as the second, untaken decision of the composition contract.
[`CrossSectionalFactorPrior`](../../src/13_Prior/17_CrossSectionalFactorPrior.jl) answers on fewer
rows than it is given: it drops the observations its Descriptors warm up over, and the observations
its exposure lag consumes. Its scenarios are the window the fit is defined on, and that is by
design.

So `ep_prior` sized `w0` at `size(X, 1)` and pushed it down the whole estimator tree with
[`factory`](../../src/02_Tools.jl) **before** it fitted the nested prior. The nested prior then fitted
on fewer rows than the weights it carried, and the call raised a bare
`DimensionMismatch: Inconsistent array dimension.` out of `StatsBase`, one call below any check that
could name it. Issue [#849](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/849) holds
the reproduction.

Four measurements settled the decision.

1. **The reduced axis is the correct one, with no exception.** `prior(factory(csfp, w60), rd)` on a
   60-observation panel raises. `prior(factory(csfp, w59), rd)` fits, and answers `59 × 20`. Every
   weight slot inside the estimator — its `ve`, its `ce` and its nested factor prior — consumes the
   reduced rows. Nothing in the tree wants the input axis.
2. **The library already implements the reading.** `OpinionPoolingPrior` fits `pe1` first and
   replaces `X` with that result, so its axis is already the nested result's.
   `OpinionPoolingPrior(; pe1 = csfp, pes = [ep])` fitted before this change, and gave 59 scenarios
   with 59 pooled weights.
3. **A uniform push does not state nothing, and that is an argument for removing it.** Five
   moment estimators answer the same under a uniform `ProbabilityWeights` as under no weights, to
   1e-16: `SimpleExpectedReturns`, `MedianExpectedReturns`, `SimpleVariance` corrected and
   uncorrected, and `Covariance`. Two do not.
     + `DistanceCovariance` moves by exactly `1/T`, because its weighted path multiplies the data
       by `w` before it takes the pairwise distances. That is a defect of that estimator, and
       [issue #851](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/851) holds it.
     + A `FactorPrior` over a `StepwiseRegression` selects a **different factor set for every one
       of the 20 assets** of `test/test12_setup.jl`, and its loadings move by up to `0.59`. A
       `ProbabilityWeights` that sums to one gives GLM an effective sample size of one, so the
       `:bic` penalty is computed against a different `n`. Before this decision, an
       `EntropyPoolingPrior` wrapping a `FactorPrior` therefore selected different factors from
       the same `FactorPrior` fitted on its own. After it, the two agree.
4. **A nested pooling result's `w` was discarded.** Two nested `EntropyPoolingPrior`s over one
   50 × 5 matrix: the inner answered a non-uniform `w`, and the outer started from uniform. The
   outer reported `kld = 0.0966` against uniform, where the divergence from the prior it actually
   wrapped is `0.2239`. That is the class [ADR 0046](0046-wrapping-priors-forward-by-default-and-document-every-drop.md)
   was written for, and `w` is the first of the three defects issue #181 named.

## Decision

**A prior that reweights observations works on the observation axis its nested prior answered. It
fits that prior first, and it reads its prior probabilities on the rows of the result.**

`ep_prior` fits before it weighs:

```julia
pr = prior(pe.pe, X, F, pnl; strict = strict, kwargs...)
w1 = w0 = ep_prior_probabilities(pe.w, pr, size(X, 1))
if !isnothing(pe.w)
    pe = factory(pe, w0)
    pr = prior(pe.pe, X, F, pnl; strict = strict, kwargs...)
end
```

`ep_prior_probabilities`, in [`10_Base_EntropyPoolingPrior.jl`](../../src/13_Prior/10_Base_EntropyPoolingPrior.jl),
holds the one rule for all four `ep_prior` bodies. It reads three sources in order:

1. **`pe.w`**, the caller's own tilt, which no fit can state. Its length must equal
   `size(pr.X, 1)`, and a length that does not match raises a `DimensionMismatch` naming that
   count, the count the estimator was handed, and the rule.
2. **`pr.w`**, where the nested fit answered one. A nested pooling prior already tilted the
   scenarios it produced, so uniform is not the prior it wrapped.
3. **The uniform `1/T`** over the rows of `pr.X`.

Only a caller's `pe.w` reaches the nested estimator before the first fit. A uniform vector states no
tilt, and `pr.w` is already carried by the fit that answered it, so neither is pushed. Every later
`factory(pe, w1)` and refit is unchanged.

**An opinion pool weights one scenario set, and it refuses an opinion that leaves it.** The pool's
axis is `pe1`'s result when `pe1` is set, and `X` when it is `nothing`. An opinion that answers on a
shorter axis carries probabilities over other scenarios, so it is refused by name. The message gives
the opinion's index, its type, both counts, and the fix: move the estimator that drops rows into
`pe1`.

## What the reference does

The reference fits its nested prior first and pushes no weights into it. It reads the observation
count off the result, it reads the prior sample weights off the result, and it falls back to uniform
over that count. It never refits the nested estimator; between stages it recomputes the mean and the
variance from the scenarios.

So the reference already answers this question the same way. This library keeps two capabilities the
reference does not have, and both survive the change: the nested estimator is refitted under the
posterior weights, so a factor model carries the views; and a caller states prior probabilities in
`pe.w`.

## Alternatives considered

| Alternative | Shape | Why not |
| --- | --- | --- |
| **Keep the uniform push** | Fit unweighted to learn the axis, then push the uniform vector and refit. | It reproduces every number to the last bit, and it buys that with one extra nested fit on **every** call, whose only purpose is to learn a row count. A uniform probability vector states no tilt, so the fit it pays for answers the same thing twice. |
| **A row-count verb** | A verb per estimator answers the row count from the estimator and the input. | The warm-up is **data**-dependent: `cross_sectional_warmup` scans the exposure history for the first row that carries a finite return and a finite exposure. A verb must rebuild that history, so it is a second authority that can drift from the fit, and it needs a recursion rule for every wrapper. Only the fit knows its own axis. |
| **A `pe1` field on the two pooling estimators** | The caller puts the row-reducing estimator in `pe1` and an ordinary prior in `pe`. | It moves no number and costs no fit, and the factor model is then never refitted under the posterior weights. `pe1` and `pe2` cannot both hold the estimator either: the Asset Panel keeps its 60 rows while `X` becomes 59, and the fit refuses by name. |
| **Refuse by name, no composition** | Compare `size(pr.X, 1)` against `size(X, 1)` and raise. | It needs the same unweighted fit that the decision needs, so it costs the same and delivers less. The composition is the one issue #782 needs. |

## Consequences

+ `EntropyPoolingPrior`, `MeucciEntropyPoolingPrior` and `OpinionPoolingPrior` all compose a
  `CrossSectionalFactorPrior`. `test/test_12k_cross_sectional_factor_prior.jl` fits the three.
+ `pe.w` is stated on an axis the caller cannot see until the fit has run. The refusal names both
  counts and the rule, which is what the caller needs to correct it.
+ A uniform `pe.w` of `nothing` no longer reaches the nested estimator before the first fit. Every
  estimator whose weighted path disagrees with its unweighted path at uniform weights moves. Five
  moment estimators move by 1e-16. `DistanceCovariance` moves by `1/T`, and that is a defect of its
  weighted path, held by issue #851. A `FactorPrior` over a `StepwiseRegression` moves its whole factor selection, and
  it moves it **onto** the selection the same estimator makes on its own; two tests in
  `test/test_12a_entropy_pooling.jl` pinned the old, inconsistent answer and were repaired in this
  change. The posterior weights of that composition move by `3.4e-3` in relative terms, and the
  enforced mean by `5.0e-4`.
+ A pooling prior nested inside another now starts from the inner posterior. `kld` and `ens` are
  reported against the prior that was actually wrapped.
+ An opinion whose wrapped estimator drops rows is refused by name rather than by a bare
  `DimensionMismatch` out of an array write.
