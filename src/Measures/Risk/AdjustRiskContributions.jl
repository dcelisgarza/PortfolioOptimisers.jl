const SquaredRiskMeasures = Union{<:Variance, <:SemiVariance, <:SquareRootKurtosis,
                                  <:SquareRootSemiKurtosis, <:BrownianDistanceVariance,
                                  <:NegativeQuadraticSkewness,
                                  <:NegativeQuadraticSemiSkewness, <:UncertaintySetVariance}
const CubedRiskMeasures = Union{<:Skewness, <:SemiSkewness}
const FourthPowerRiskMeasures = Union{<:Kurtosis, <:SemiKurtosis}
function adjust_risk_contribution(::Any, val)
    return val
end
function adjust_risk_contribution(::SquaredRiskMeasures, val)
    return val * 0.5
end
function adjust_risk_contribution(::CubedRiskMeasures, val)
    return val / 3
end
function adjust_risk_contribution(::FourthPowerRiskMeasures, val)
    return val * 0.25
end
