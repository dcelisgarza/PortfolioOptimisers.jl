# Median Absolute Deviation Risk

```@docs
MedianCenteringFunction
MedianCentering
MeanCentering
MedAbsDevMu
MedianAbsoluteDeviation
factory(r::MedianAbsoluteDeviation, pr::AbstractPriorResult, args...; kwargs...)
risk_measure_view(r::MedianAbsoluteDeviation, i, args...)
calc_moment_target(::MedianAbsoluteDeviation{<:Any, Nothing, <:MeanCentering, <:Any}, ::Any, x::VecNum)
calc_deviations_vec(r::MedianAbsoluteDeviation, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
```
