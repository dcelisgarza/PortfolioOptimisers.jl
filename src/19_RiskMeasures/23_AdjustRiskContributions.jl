const SquaredRiskMeasures = Union{<:Variance, <:BrownianDistanceVariance,
                                  <:UncertaintySetVariance,
                                  <:LowOrderMoment{<:Any, <:Any, <:Any,
                                                   <:StandardisedLowOrderMoment{<:Any,
                                                                                <:SecondCentralMoment}},
                                  <:LowOrderMoment{<:Any, <:Any, <:Any,
                                                   <:StandardisedLowOrderMoment{<:Any,
                                                                                <:SecondLowerMoment}},
                                  <:Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:SOCRiskExpr},
                                  <:NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                     <:Union{<:SquaredSOCRiskExpr,
                                                             <:QuadRiskExpr}}}
const QuadExpressionRiskMeasures = Union{<:Variance, <:BrownianDistanceVariance,
                                         <:LowOrderMoment{<:Any, <:Any, <:Any,
                                                          <:StandardisedLowOrderMoment{<:Any,
                                                                                       <:SecondCentralMoment{<:Union{<:QuadRiskExpr,
                                                                                                                     <:RSOCRiskExpr,
                                                                                                                     <:SquaredSOCRiskExpr}}}},
                                         <:LowOrderMoment{<:Any, <:Any, <:Any,
                                                          <:StandardisedLowOrderMoment{<:Any,
                                                                                       <:SecondLowerMoment{<:Union{<:QuadRiskExpr,
                                                                                                                   <:RSOCRiskExpr,
                                                                                                                   <:SquaredSOCRiskExpr}}}},
                                         <:NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                            <:Union{<:SquaredSOCRiskExpr,
                                                                    <:QuadRiskExpr}}}
const CubedRiskMeasures = Union{<:ThirdCentralMoment, <:Skewness,
                                <:HighOrderMoment{<:Any, <:Any, <:Any, <:ThirdLowerMoment},
                                <:HighOrderMoment{<:Any, <:Any, <:Any,
                                                  <:StandardisedHighOrderMoment{<:Any,
                                                                                <:ThirdLowerMoment}}}
const FourthPowerRiskMeasures = Union{<:HighOrderMoment{<:Any, <:Any, <:Any,
                                                        <:FourthLowerMoment},
                                      <:HighOrderMoment{<:Any, <:Any, <:Any,
                                                        <:StandardisedHighOrderMoment{<:Any,
                                                                                      <:FourthLowerMoment}},
                                      <:HighOrderMoment{<:Any, <:Any, <:Any,
                                                        <:FourthCentralMoment},
                                      <:HighOrderMoment{<:Any, <:Any, <:Any,
                                                        <:StandardisedHighOrderMoment{<:Any,
                                                                                      <:FourthCentralMoment}},
                                      <:Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                 <:Union{<:SquaredSOCRiskExpr,
                                                         <:QuadRiskExpr}}}
const DrawdownRiskMeasures = Union{<:DrawdownatRisk, <:RelativeDrawdownatRisk,
                                   <:ConditionalDrawdownatRisk,
                                   <:RelativeConditionalDrawdownatRisk,
                                   <:EntropicDrawdownatRisk,
                                   <:RelativeEntropicDrawdownatRisk,
                                   <:RelativisticDrawdownatRisk,
                                   <:RelativeRelativisticDrawdownatRisk}
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
