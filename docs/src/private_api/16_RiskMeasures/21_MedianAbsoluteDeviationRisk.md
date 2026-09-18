```@meta
Description = "Median Absolute Deviation Risk, private API of PortfolioOptimisers.jl: MedianCenteringFunction, MedAbsDevMu, resolve_deferred_quantities, …"
```

# Median Absolute Deviation Risk: private API

```@docs
MedianCenteringFunction
MedAbsDevMu
resolve_deferred_quantities(r::MedianAbsoluteDeviation, pr::AbstractPriorResult)
nothing_scalar_array_view(x::MedianCenteringFunction, ::Any)
calc_moment_target(::MedianAbsoluteDeviation{<:Any, Nothing, <:MeanCentering, <:Any}, ::Any, x::VecNum)
calc_deviations_vec(r::MedianAbsoluteDeviation, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
calc_deviations_vec(r::MedianAbsoluteDeviation, x::VecNum)
weight_independent_target(::MedianCenteringFunction)
supports_precomputed_returns(r::MedianAbsoluteDeviation)
```
