---
status: accepted
---

# Every optimisation estimator reduces once at its entry, and its result carries the Investable Mask

## Context

A Prior Estimator fits on the coverage universe and returns a result on the full asset universe,
in which an asset it could not estimate carries `NaN` in `mu` and on the diagonal of `sigma`. The
JuMP families already reduce to the assets that are investable: `investable_mask` derives the
Investable Mask from the prior result, `investable_reduction` takes a `port_opt_view` of the prior,
the optimiser and the returns data at it, and the keyword constructor of `JuMPOptimisationResult`
expands the solved weights back onto the full universe. The mask rides on the bundle as `imsk`.

Nine sites outside the JuMP families fit a prior and then solve: the three hierarchical
optimisers, the nested clustered optimiser, `InverseVolatility`, `Stacking` and
`SubsetResampling`. None of them derives a mask. Each allocates its weight vector at the full width
and fills it, so a `NaN` on the diagonal reaches the unitary risks of a bisection, the Schur
complement, the inverse volatility, and the distance a clustering is built from. The result is a
silently wrong number, not a refusal.

The reference implementation reduces in its convex family only. Its hierarchical, nested, naive
and ensemble families derive no mask, its hierarchical base omits the mask from its input cleaner,
and its nested optimiser gives a lone non-investable asset the whole weight of its cluster. It
gives no answer for these nine sites.

Map [#667](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/667) carries the contract
of [#647](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/647) to every layer, and
ticket [#669](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/669) settled where the
reduction and the expansion live for the families the JuMP prelude does not reach.

## Decision

### Every family reduces once at its entry

Every optimisation estimator that fits a prior reduces once, immediately after its prior fit and
before its clustering, its sampling and its inner solves. The whole algorithm below that line runs
on the investable universe. The site writes one line:

```julia
pr = prior(opt.pe, rd; dims = dims)
imsk, pr, opt, rd = investable_reduction(pr, opt, rd)
```

So the distance a hierarchical optimiser clusters on never sees a `NaN`, the cluster count is
chosen on the investable universe, a subset is drawn from the investable universe alone, and no
cluster of the nested optimiser holds a non-investable asset. A meta-optimiser composes two masks:
the one its own prior yields at its entry, and the one each inner head yields inside its own
solve. Both expand, and the cluster slice of the nested optimiser indexes the reduced axis.

### One shared verb, bound to the root

`investable_reduction` keeps its three methods and widens its bound from `JuMPOptimiser` to
`AbstractOptimisationEstimator`, which every head and the JuMP configuration sit under. It moves
beside the `port_opt_view` catch-all in the base file of the optimisation directory, so the files
that load before the JuMP prelude reach it without a back reference. The `nothing` path stays
dispatch, so a universe with nothing to exclude allocates nothing and the complexity gate does
not move.

### The result carries the mask, and its keyword constructor expands

The six result types of these families gain an `imsk` field, bound to `Option{BitVector}` and
`nothing` when every asset was investable: `HierarchicalResult`, which `HierarchicalRiskParity`
and `HierarchicalEqualRiskContribution` share, `SchurComplementHierarchicalRiskParityResult`,
`NaiveOptimisationResult`, `NestedClusteredResult`, `StackingResult` and `SubsetResamplingResult`.

The keyword constructor of each is the one door its `_optimise` exits through, and it expands `w`
through `expand_investable_weights`, which gains methods over a plain vector and a vector of them
beside its JuMP methods. The positional constructor never expands, because every retcode rebuild
goes through it and a second pass would expand twice. This is the JuMP door, transferred.

The result carries the objects of the reduced universe beside the mask: `pr`, `clr`, `wb`, `fees`
and the subset index are the ones the algorithm ran on, as the JuMP bundle carries the reduced
prior and constraints. The reduced prior can no longer yield the mask, which is why the result
must carry it. A reader reads `res.w` on the full universe and `res.imsk` on every result of the
library with one idiom, and the mask is what tells a zero from a refusal.

### A pre-fitted clustering result of the wrong width is refused

A caller can state a fitted clustering result instead of an estimator. `clusterise` returns it
unchanged and the hierarchical view does not slice it, so under the mask its leaf order would
index a reduced returns matrix. One shared check after `clusterise` compares the result's width to
the reduced universe and throws `DimensionMismatch`, the type the `Clusters` constructor and
`expand_investable_weights` already throw. The same check closes the hazard that a subset or a
nested view of a hierarchical head carried before the mask existed. A fixed clustering under a
changing universe is stated as an estimator.

### No asset investable throws `IsEmptyError`

`investable_mask` throws it where the mask is derived, so every family has the refusal for free.

### The two prior-free naive heads reduce to the Coverage Universe

`EqualWeighted` and `RandomWeighted` fit no prior, so no Prior Result yields a mask for them. Each
derives its mask from the Coverage Universe of its window instead, through the one verb the priors
use, weights the reduced universe, and carries the mask as `imsk`, so its keyword constructor
expands the weights as every other result's does. A stale finite price during an inactive spell
weights nothing. Ticket [#674](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/674)
decided it, in
[ADR 0120](0120-a-fold-scores-on-the-investable-mask-a-prior-free-head-and-pre-selection-reduce-to-the-coverage-universe-and-a-failed-candidate-loses-the-search.md),
once ADR 0117 had settled the rule with no threshold. Every result of the library carries `imsk`.

## Considered options

| Question | Refused | Why |
| --- | --- | --- |
| Where the reduction happens | The hierarchical and naive families reduce, and the meta families rely on the inner heads, which is the reference's shape. | The outer prior still feeds `NaN` into the fees and the outer bounds, the nested distance is poisoned, and a subset can draw only dead assets. |
| Where the reduction happens | Cluster the full universe, then drop the non-investable leaves. | The distance needs finite columns, so it hangs on the moment decision, and a pruned dendrogram has no defined branch order. |
| The verb | Three lines and a condition per site. | Nine copies of one idiom, nine conditions the complexity gate counts, and nine places that drift. |
| The exit | The `_optimise` body expands one line before the constructor. | Nine sites can each forget the line. |
| The exit | `finalise_weight_bounds` takes the mask and expands. | It fuses the bound check and the universe expansion in one verb, and the two meta finalisers must thread the mask through. |
| A pre-fitted clustering result | Slice it with a `port_opt_view`. | The matrices slice, but dropping a leaf from the dendrogram changes the merges, the branch order and `k`, so the result is no longer the caller's clustering. |
| The prior-free naive heads | No mask, and the caller reduces through pre-selection; or an optional prior slot for the mask alone. | Two results with no mask, so a reader has two idioms, and a bare equal-weighted benchmark over a point-in-time panel weights a dead asset; or a prior fit that no weight reads. |

## Consequences

- The build tickets [#675](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/675) and
  [#676](https://github.com/dcelisgarza/PortfolioOptimisers.jl/issues/676) follow this rule. The
  oracle of every test is the same optimisation with the non-investable asset removed by hand.
- The `field_dict` text for `imsk` stops naming the bundle, because six result types now carry
  the field.
- The base file of the optimisation directory grows by the verb, so its size baseline rises.
- The subset index on a `SubsetResamplingResult` is in reduced positions when a mask was applied,
  as its `pr` and `wb` are.
