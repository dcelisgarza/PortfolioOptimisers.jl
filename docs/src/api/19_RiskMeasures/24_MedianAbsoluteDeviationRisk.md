# Median Absolute Deviation Risk

```@docs
MedianCenteringFunction
MedianCentering
MeanCentering
MedAbsDevMu
MedianAbsoluteDeviation
nothing_scalar_array_view(x::MedianCenteringFunction, ::Any)
port_opt_view(r::MedianAbsoluteDeviation, i, args...)
calc_moment_target(::MedianAbsoluteDeviation{<:Any, Nothing, <:MeanCentering, <:Any}, ::Any, x::VecNum)
calc_deviations_vec(r::MedianAbsoluteDeviation, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
calc_deviations_vec(r::MedianAbsoluteDeviation, x::VecNum)
weight_independent_target(::MedianCenteringFunction)
supports_precomputed_returns(r::MedianAbsoluteDeviation)
```
