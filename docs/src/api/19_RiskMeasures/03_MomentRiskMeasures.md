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
PortfolioOptimisers.MomentMeasureAlgorithm
PortfolioOptimisers.LowOrderMomentMeasureAlgorithm
PortfolioOptimisers.UnstandardisedLowOrderMomentMeasureAlgorithm
PortfolioOptimisers.HighOrderMomentMeasureAlgorithm
PortfolioOptimisers.UnstandardisedHighOrderMomentMeasureAlgorithm
PortfolioOptimisers.calc_moment_target(::LoHiOrderMoment{<:Any, Nothing, Nothing, <:Any},
                            ::Any, x::VecNum)
PortfolioOptimisers.calc_moment_target(r::LoHiOrderMoment{<:Any, <:AbstractWeights, Nothing, <:Any},
                       ::Any, x::VecNum)
PortfolioOptimisers.calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecNum,
                                                      <:Any}, w::VecNum, ::Any)
PortfolioOptimisers.calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                            w::VecNum, ::Any)
PortfolioOptimisers.calc_moment_target(r::LoHiOrderMoment{<:Any, <:Any, <:Number, <:Any}, ::Any, ::Any)
PortfolioOptimisers.calc_deviations_vec(r::LoHiOrderMoment, w::VecNum,
                         X::MatNum, fees::Option{<:Fees} = nothing)
```
