const SquaredRiskMeasures = Union{<:Variance,
                                  <:LowOrderMoment{<:Any,
                                                   <:LowOrderDeviation{<:Any,
                                                                       <:SecondLowerMoment},
                                                   <:Any, <:Any}, <:SquareRootKurtosis,
                                  <:BrownianDistanceVariance,
                                  <:NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                     <:QuadraticNegativeSkewness},
                                  <:UncertaintySetVariance}
const CubedRiskMeasures = Union{<:ThirdCentralMoment, <:Skewness,
                                <:HighOrderMoment{<:Any, <:ThirdLowerMoment, <:Any, <:Any},
                                <:HighOrderMoment{<:Any,
                                                  <:HighOrderDeviation{<:Any,
                                                                       <:ThirdLowerMoment},
                                                  <:Any, <:Any}}
const FourthPowerRiskMeasures = Union{<:HighOrderMoment{<:Any, <:FourthLowerMoment, <:Any,
                                                        <:Any},
                                      <:HighOrderMoment{<:Any, <:FourthCentralMoment, <:Any,
                                                        <:Any},
                                      <:HighOrderMoment{<:Any,
                                                        <:HighOrderDeviation{<:Any,
                                                                             <:FourthLowerMoment},
                                                        <:Any, <:Any},
                                      <:HighOrderMoment{<:Any,
                                                        <:HighOrderDeviation{<:Any,
                                                                             <:FourthCentralMoment},
                                                        <:Any, <:Any}}
function adjust_risk_contribution(::Any, val::Real, args...)
    return val
end
function adjust_risk_contribution(::SquaredRiskMeasures, val::Real, args...)
    return val * 0.5
end
function adjust_risk_contribution(::CubedRiskMeasures, val::Real, args...)
    return val / 3
end
function adjust_risk_contribution(::FourthPowerRiskMeasures, val::Real, args...)
    return val * 0.25
end
function adjust_risk_contribution(::EqualRiskMeasure, val::Real, delta::Real = 0.0)
    return val + delta
end
