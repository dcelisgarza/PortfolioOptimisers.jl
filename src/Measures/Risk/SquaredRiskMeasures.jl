const SquaredRiskMeasures = Union{<:Variance}
# , <:SemiVariance, <:WorstCaseVariance,
# <:SquareRootKurtosis, <:SquareRootSemiKurtosis,
# <:BrownianDistanceVariance, <:NegativeQuadraticSkewness,
# <:NegativeQuadraticSemiSkewness}
function adjust_risk_contribution(::Any, val)
    return val
end
function adjust_risk_contribution(::SquaredRiskMeasures, val)
    return val * 0.5
end