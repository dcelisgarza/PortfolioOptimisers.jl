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
PortfolioOptimisers.calc_moment_target(::LHOrderMoment{<:Any, Nothing, Nothing, <:Any},
                            ::Any, x::VecNum)
PortfolioOptimisers.calc_moment_target(r::LHOrderMoment{<:Any, <:AbstractWeights, Nothing, <:Any},
                       ::Any, x::VecNum)
PortfolioOptimisers.calc_moment_target(r::LHOrderMoment{<:Any, <:Any, <:VecNum,
                                                      <:Any}, w::VecNum, ::Any)
PortfolioOptimisers.calc_moment_target(r::LHOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                            w::VecNum, ::Any)
PortfolioOptimisers.calc_moment_target(r::LHOrderMoment{<:Any, <:Any, <:Number, <:Any}, ::Any, ::Any)
PortfolioOptimisers.calc_deviations_vec(r::LHOrderMoment, w::VecNum,
                         X::MatNum, fees::Option{<:Fees} = nothing)
```
