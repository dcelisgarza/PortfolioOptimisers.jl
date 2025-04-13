const SquaredRiskMeasures = Union{<:Variance, <:SemiVariance, <:SquareRootKurtosis,
                                  <:SquareRootSemiKurtosis, <:BrownianDistanceVariance,
                                  <:NegativeQuadraticSkewness,
                                  <:NegativeQuadraticSemiSkewness, <:UncertaintySetVariance}
const CubedRiskMeasures = Union{<:Skewness, <:SemiSkewness}
const FourthPowerRiskMeasures = Union{<:Kurtosis, <:SemiKurtosis}
function adjust_risk_contribution(::Any, val, args...)
    return val
end
function adjust_risk_contribution(::SquaredRiskMeasures, val, args...)
    return val * 0.5
end
function adjust_risk_contribution(::CubedRiskMeasures, val, args...)
    return val / 3
end
function adjust_risk_contribution(::FourthPowerRiskMeasures, val, args...)
    return val * 0.25
end
function adjust_risk_contributions(::EqualRiskMeasure, val, delta::Real = 0.0)
    return val + delta
end
