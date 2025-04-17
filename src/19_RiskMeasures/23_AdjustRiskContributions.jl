const SquaredRiskMeasures = Union{<:Variance,
                                  <:LowOrderMoment{<:Any, <:SemiVariance, <:Any, <:Any},
                                  <:SquareRootKurtosis, <:BrownianDistanceVariance,
                                  <:NegativeSkewness{<:Any, <:QuadraticNegativeSkewness,
                                                     <:Any, <:Any, <:Any},
                                  <:UncertaintySetVariance}
const CubedRiskMeasures = Union{<:Skewness,
                                <:HighOrderMoment{<:Any, <:ThirdLowerMoment, <:Any, <:Any},
                                <:HighOrderMoment{<:Any,
                                                  <:HighOrderDeviation{<:ThirdLowerMoment,
                                                                       <:Any}, <:Any,
                                                  <:Any}}
const FourthPowerRiskMeasures = Union{<:HighOrderMoment{<:Any, <:FourthLowerMoment, <:Any,
                                                        <:Any},
                                      <:HighOrderMoment{<:Any, <:FourthCentralMoment, <:Any,
                                                        <:Any},
                                      <:HighOrderMoment{<:Any,
                                                        <:HighOrderDeviation{<:FourthLowerMoment,
                                                                             <:Any}, <:Any,
                                                        <:Any},
                                      <:HighOrderMoment{<:Any,
                                                        <:HighOrderDeviation{<:FourthCentralMoment,
                                                                             <:Any}, <:Any,
                                                        <:Any}}
function adjust_risk_contribution(::Any, val, args...)
    return val
end
function adjust_risk_contribution(::SquaredRiskMeasures, val, args...)
    return val * 0.5
end
function adjust_risk_contribution(::Skewness, val, args...)
    return val / 3
end
function adjust_risk_contribution(::FourthPowerRiskMeasures, val, args...)
    return val * 0.25
end
function adjust_risk_contributions(::EqualRiskMeasure, val, delta::Real = 0.0)
    return val + delta
end
