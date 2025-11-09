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
PortfolioOptimisers.calc_moment_target(::Union{<:LowOrderMoment{<:Any, Nothing, Nothing, <:Any},
                                    <:HighOrderMoment{<:Any, Nothing, Nothing, <:Any}},
                            ::Any, x::VecNum)
PortfolioOptimisers.calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:AbstractWeights, Nothing, <:Any},
                                <:HighOrderMoment{<:Any, <:AbstractWeights, Nothing, <:Any}},
                       ::Any, x::VecNum)
PortfolioOptimisers.calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:VecNum,
                                                      <:Any},
                                     <:HighOrderMoment{<:Any, <:Any, <:VecNum,
                                                       <:Any}}, w::VecNum, ::Any)
PortfolioOptimisers.calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                                     <:HighOrderMoment{<:Any, <:Any, <:VecScalar, <:Any}},
                            w::VecNum, ::Any)
PortfolioOptimisers.calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:Number, <:Any},
                                <:HighOrderMoment{<:Any, <:Any, <:Number, <:Any}}, ::Any, ::Any)
PortfolioOptimisers.calc_deviations_vec(r::Union{<:LowOrderMoment, <:HighOrderMoment}, w::VecNum,
                         X::MatNum, fees::Option{<:Fees} = nothing)
```
