```@meta
Description = "Moment Risk Measures, private API of PortfolioOptimisers.jl: MomentMeasureAlgorithm, LowOrderMomentMeasureAlgorithm, …"
```

# Moment Risk Measures: private API

```@docs
MomentMeasureAlgorithm
LowOrderMomentMeasureAlgorithm
UnstandardisedLowOrderMomentMeasureAlgorithm
HighOrderMomentMeasureAlgorithm
UnstandardisedHighOrderMomentMeasureAlgorithm
LoHiOrderMoment
calc_moment_target(::LoHiOrderMoment{<:Any, Nothing, Nothing, <:Any}, ::Any, x::VecNum)
calc_moment_target(r::LoHiOrderMoment{<:Any, <:StatsBase.AbstractWeights, Nothing, <:Any}, ::Any, x::VecNum)
calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecNum, <:Any}, w::VecNum, ::Any)
calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecScalar, <:Any}, w::VecNum, ::Any)
calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:Number, <:Any}, ::Any, ::Any)
calc_deviations_vec(r::LoHiOrderMoment, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
calc_deviations_vec(r::LoHiOrderMoment, x::VecNum)
resolve_deferred_quantities(r::LowOrderMoment, pr::AbstractPriorResult)
resolve_deferred_quantities(r::HighOrderMoment, pr::AbstractPriorResult)
supports_precomputed_returns(r::LoHiOrderMoment)
moment_risk
```
