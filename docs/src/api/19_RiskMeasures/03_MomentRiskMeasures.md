# Moment Risk Measures

```@docs
FirstLowerMoment
MeanAbsoluteDeviation
SecondMoment
factory(alg::SecondMoment, w::ObsWeights)
EvenMoment
LowOrderMoment
ThirdLowerMoment
FourthMoment
StandardisedHighOrderMoment
factory(alg::StandardisedHighOrderMoment, w::ObsWeights)
HighOrderMoment
MomentMeasureAlgorithm
factory(alg::MomentMeasureAlgorithm, args...; kwargs...)
LowOrderMomentMeasureAlgorithm
UnstandardisedLowOrderMomentMeasureAlgorithm
HighOrderMomentMeasureAlgorithm
UnstandardisedHighOrderMomentMeasureAlgorithm
LoHiOrderMoment
calc_moment_target(::LoHiOrderMoment{<:Any, Nothing, Nothing, <:Any},
                            ::Any, x::VecNum)
calc_moment_target(r::LoHiOrderMoment{<:Any, <:StatsBase.AbstractWeights, Nothing, <:Any},
                       ::Any, x::VecNum)
calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecNum,
                                                      <:Any}, w::VecNum, ::Any)
calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                            w::VecNum, ::Any)
calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:Number, <:Any}, ::Any, ::Any)
calc_deviations_vec(r::LoHiOrderMoment, w::VecNum, X::MatNum, fees::Option{<:Fees} = nothing)
```
