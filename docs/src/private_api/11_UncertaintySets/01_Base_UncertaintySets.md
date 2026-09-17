```@meta
Description = "Base Uncertainty Sets, private API of PortfolioOptimisers.jl: AbstractUncertaintySetEstimator, AbstractPriorUncertaintySetEstimator, …"
```

# Base Uncertainty Sets: private API

```@docs
AbstractUncertaintySetEstimator
AbstractPriorUncertaintySetEstimator
AbstractUncertaintySetAlgorithm
AbstractUncertaintySetResult
AbstractUncertaintyKAlgorithm
UcSE_UcS
Num_UcSK
Num_UcSEps
AbstractCompactRadiusAlgorithm
Num_CptRad
AbstractUncertaintySetClass
reads_prior_result
ucs_prior
ucs_selector
k_ucs
investable_ucs_reduction
expand_investable_ucs(set::AbstractUncertaintySetResult, ::Nothing, ::AbstractPriorResult)
expand_investable_ucs(set::BoxUncertaintySet{<:VecNum, <:VecNum}, imsk::BitVector, pr::AbstractPriorResult)
expand_investable_ucs(set::BoxUncertaintySet{<:MatNum, <:MatNum}, imsk::BitVector, pr::AbstractPriorResult)
expand_investable_ucs(set::EllipsoidalUncertaintySet{<:MatNum, <:Any, <:SigmaUncertaintySetClass}, imsk::BitVector, pr::AbstractPriorResult)
expand_investable_ucs(set::EllipsoidalUncertaintySet{<:MatNum, <:Any, <:MuUncertaintySetClass}, imsk::BitVector, pr::AbstractPriorResult)
vec_quantile_bounds
ellipsoidal_set
box_quantile_bounds
```

## References

```@bibliography
Pages = [@__FILE__]
Canonical = false
```
