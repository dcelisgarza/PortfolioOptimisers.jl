const SquaredRiskMeasures = Union{<:Variance, <:SquareRootKurtosis}
# , <:SemiVariance, <:WorstCaseVariance,
# , <:SquareRootSemiKurtosis,
# <:BrownianDistanceVariance, <:NegativeQuadraticSkewness,
# <:NegativeQuadraticSemiSkewness}
function adjust_risk_contribution(::Any, val)
    return val
end
function adjust_risk_contribution(::SquaredRiskMeasures, val)
    return val * 0.5
end
