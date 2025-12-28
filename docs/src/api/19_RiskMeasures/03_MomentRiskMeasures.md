# Moment Risk Measures

```@docs
FirstLowerMoment
MeanAbsoluteDeviation
SecondMoment
LowOrderMoment
ThirdLowerMoment
FourthMoment
StandardisedHighOrderMoment
HighOrderMoment
MomentMeasureAlgorithm
LowOrderMomentMeasureAlgorithm
UnstandardisedLowOrderMomentMeasureAlgorithm
HighOrderMomentMeasureAlgorithm
UnstandardisedHighOrderMomentMeasureAlgorithm
calc_moment_target(::LoHiOrderMoment{<:Any, Nothing, Nothing, <:Any},
                            ::Any, x::VecNum)
calc_moment_target(r::LoHiOrderMoment{<:Any, <:StatsBase.AbstractWeights, Nothing, <:Any},
                       ::Any, x::VecNum)
calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecNum,
                                                      <:Any}, w::VecNum, ::Any)
calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                            w::VecNum, ::Any)
calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:Number, <:Any}, ::Any, ::Any)
calc_deviations_vec(r::LoHiOrderMoment, w::VecNum,
                         X::MatNum, fees::Option{<:Fees} = nothing)
```
