"""
    const SquaredRiskMeasures

Union of risk measures whose expected risk is a squared quantity. When computing risk
contributions via finite differences, the library halves the raw gradient value to account
for the square.
"""
const SquaredRiskMeasures = Union{<:Variance, <:BrownianDistanceVariance,
                                  <:UncertaintySetVariance,
                                  <:LowOrderMoment{<:Any, <:Any, <:Any,
                                                   <:SecondMoment{<:Any, <:Any,
                                                                  <:QuadSecondMomentFormulations}},
                                  <:LowOrderMoment{<:Any, <:Any, <:Any, <:EvenMoment},
                                  <:Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                             <:SOCRiskExpr},
                                  <:NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                     <:NSkeQuadFormulations},
                                  <:TrackingRiskMeasure{<:Any, <:Any, <:SquaredL2Norm}}
"""
    const QuadExpressionRiskMeasures

Union of risk measures that use quadratic JuMP expressions in their constraint
formulations.
"""
const QuadExpressionRiskMeasures = Union{<:Variance, <:BrownianDistanceVariance,
                                         <:LowOrderMoment{<:Any, <:Any, <:Any,
                                                          <:SecondMoment{<:Any, <:Any,
                                                                         <:QuadSecondMomentFormulations}},
                                         <:NegativeSkewness{<:Any, <:Any, <:Any, <:Any,
                                                            <:NSkeQuadFormulations},
                                         <:Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any,
                                                    <:Any, <:QuadSecondMomentFormulations},
                                         <:TrackingRiskMeasure{<:Any, <:Any,
                                                               <:SquaredL2Norm}}
"""
    const CubedRiskMeasures

Union of risk measures whose expected risk is a cubed quantity. When computing risk
contributions via finite differences, the library divides the raw gradient value by three.
"""
const CubedRiskMeasures = Union{<:ThirdCentralMoment, <:Skewness,
                                <:HighOrderMoment{<:Any, <:Any, <:Any, <:ThirdLowerMoment},
                                <:HighOrderMoment{<:Any, <:Any, <:Any,
                                                  <:StandardisedHighOrderMoment{<:Any,
                                                                                <:ThirdLowerMoment}}}
"""
    const FourthPowerRiskMeasures

Union of risk measures whose expected risk is a fourth-power quantity. When computing
risk contributions via finite differences, the library multiplies the raw gradient value
by 0.25.
"""
const FourthPowerRiskMeasures = Union{<:HighOrderMoment{<:Any, <:Any, <:Any,
                                                        <:FourthMoment},
                                      <:HighOrderMoment{<:Any, <:Any, <:Any,
                                                        <:StandardisedHighOrderMoment{<:Any,
                                                                                      <:FourthMoment}},
                                      <:Kurtosis{<:Any, <:Any, <:Any, <:Any, <:Any, <:Any,
                                                 <:QuadSecondMomentFormulations}}
"""
    const DrawdownRiskMeasures

Union of all drawdown-based risk measures.
"""
const DrawdownRiskMeasures = Union{<:DrawdownatRisk, <:RelativeDrawdownatRisk,
                                   <:ConditionalDrawdownatRisk,
                                   <:RelativeConditionalDrawdownatRisk,
                                   <:EntropicDrawdownatRisk,
                                   <:RelativeEntropicDrawdownatRisk,
                                   <:RelativisticDrawdownatRisk,
                                   <:RelativeRelativisticDrawdownatRisk}
"""
    adjust_risk_contribution(r, val::Number, args...)

Adjust the finite-difference gradient value `val` used in risk contribution computation
to account for the mathematical structure of risk measure `r`.

Returns `val` unchanged for most risk measures. Specialisations scale the value
appropriately for [`SquaredRiskMeasures`](@ref) (×0.5), [`CubedRiskMeasures`](@ref)
(÷3), [`FourthPowerRiskMeasures`](@ref) (×0.25), and [`EqualRiskMeasure`](@ref)
(+`delta`).

# Arguments

  - `r`: Risk measure instance.
  - `val::Number`: Raw finite-difference gradient value.

# Returns

  - `Number`: Adjusted gradient value.

# Related

  - [`SquaredRiskMeasures`](@ref)
  - [`CubedRiskMeasures`](@ref)
  - [`FourthPowerRiskMeasures`](@ref)
  - [`EqualRiskMeasure`](@ref)
  - [`risk_contribution`](@ref)
"""
function adjust_risk_contribution(::Any, val::Number, args...)
    return val
end
function adjust_risk_contribution(::SquaredRiskMeasures, val::Number, args...)
    return val * 0.5
end
function adjust_risk_contribution(::CubedRiskMeasures, val::Number, args...)
    return val / 3
end
function adjust_risk_contribution(::FourthPowerRiskMeasures, val::Number, args...)
    return val * 0.25
end
function adjust_risk_contribution(::EqualRiskMeasure, val::Number, delta::Number = 0.0)
    return val + delta
end
