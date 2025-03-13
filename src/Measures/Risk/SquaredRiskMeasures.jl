const SquaredRiskMeasures = Union{<:Variance, <:SquareRootKurtosis,
                                  <:SquareRootSemiKurtosis, <:BrownianDistanceVariance,
                                  <:NegativeQuadraticSkewness,
                                  <:NegativeQuadraticSemiSkewness, <:UncertaintySetVariance}
# , <:SemiVariance, 

function adjust_risk_contribution(::Any, val)
    return val
end
function adjust_risk_contribution(::SquaredRiskMeasures, val)
    return val * 0.5
end
