# Moment Risk Measures

```@docs
FirstLowerMoment
MeanAbsoluteDeviation
SecondLowerMoment
SecondCentralMoment
StandardisedLowOrderMoment
LowOrderMoment
ThirdLowerMoment
FourthLowerMoment
FourthCentralMoment
StandardisedHighOrderMoment
HighOrderMoment
PortfolioOptimisers.MomentMeasureAlgorithm
PortfolioOptimisers.LowOrderMomentMeasureAlgorithm
PortfolioOptimisers.UnstandardisedLowOrderMomentMeasureAlgorithm
PortfolioOptimisers.UnstandardisedSecondMomentAlgorithm
PortfolioOptimisers.HighOrderMomentMeasureAlgorithm
PortfolioOptimisers.UnstandardisedHighOrderMomentMeasureAlgorithm
PortfolioOptimisers.calc_moment_target(::Union{<:LowOrderMoment{<:Any, Nothing, Nothing, <:Any},
                                    <:HighOrderMoment{<:Any, Nothing, Nothing, <:Any}},
                            ::Any, x::AbstractVector)
PortfolioOptimisers.calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:AbstractWeights, Nothing, <:Any},
                                <:HighOrderMoment{<:Any, <:AbstractWeights, Nothing, <:Any}},
                       ::Any, x::AbstractVector)
PortfolioOptimisers.calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:AbstractVector,
                                                      <:Any},
                                     <:HighOrderMoment{<:Any, <:Any, <:AbstractVector,
                                                       <:Any}}, w::AbstractVector, ::Any)
PortfolioOptimisers.calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:VecScalar, <:Any},
                                     <:HighOrderMoment{<:Any, <:Any, <:VecScalar, <:Any}},
                            w::AbstractVector, ::Any)
PortfolioOptimisers.calc_moment_target(r::Union{<:LowOrderMoment{<:Any, <:Any, <:Real, <:Any},
                                <:HighOrderMoment{<:Any, <:Any, <:Real, <:Any}}, ::Any, ::Any)
PortfolioOptimisers.calc_deviations_vec(r::Union{<:LowOrderMoment, <:HighOrderMoment}, w::AbstractVector,
                         X::AbstractMatrix, fees::Union{Nothing, <:Fees} = nothing)
```
