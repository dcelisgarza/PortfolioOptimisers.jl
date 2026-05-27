# Non-Optimisation Risk Measures

```@docs
MeanReturn
factory(r::MeanReturn, pr::AbstractPriorResult, args...)
risk_measure_view(r::MeanReturn, ::Any, args...)
MeanReturnRiskRatio
needs_previous_weights(r::MeanReturnRiskRatio)
factory(r::MeanReturnRiskRatio, args...; kwargs...)
factory(r::MeanReturnRiskRatio, w::VecNum)
ThirdCentralMoment
factory(r::ThirdCentralMoment, pr::AbstractPriorResult, args...; kwargs...)
risk_measure_view(r::ThirdCentralMoment, i, args...)
Skewness
factory(r::Skewness, pr::AbstractPriorResult, args...; kwargs...)
risk_measure_view(r::Skewness, i, args...)
TCM_Sk
calc_moment_target(::TCM_Sk{Nothing, Nothing}, ::Any, x::VecNum)
calc_deviations_vec(r::TCM_Sk, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
```
