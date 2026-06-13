# Non-Optimisation Risk Measures

```@docs
MeanReturn
factory(r::MeanReturn, pr::AbstractPriorResult, args...)
port_opt_view(r::MeanReturn, ::Any, args...)
MeanReturnRiskRatio
needs_previous_weights(r::MeanReturnRiskRatio)
factory(r::MeanReturnRiskRatio, args...; kwargs...)
factory(r::MeanReturnRiskRatio, w::VecNum)
ThirdCentralMoment
factory(r::ThirdCentralMoment, pr::AbstractPriorResult, args...; kwargs...)
port_opt_view(r::ThirdCentralMoment, i, args...)
TCM_Sk
calc_moment_target(::TCM_Sk{Nothing, Nothing}, ::Any, x::VecNum)
calc_deviations_vec(r::TCM_Sk, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
calc_deviations_vec(r::TCM_Sk, x::VecNum)
supports_precomputed_returns(r::ThirdCentralMoment)
```
