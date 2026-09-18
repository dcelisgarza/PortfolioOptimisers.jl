```@meta
Description = "Base Uncertainty Sets, private API of PortfolioOptimisers.jl: UcSE_UcS, Num_UcSK, Num_UcSEps, Num_CptRad, ucs_prior, ucs_selector, investable_ucs_reduction, …"
```

# Base Uncertainty Sets: private API

```@docs
UcSE_UcS
Num_UcSK
Num_UcSEps
Num_CptRad
ucs_prior
ucs_selector
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
