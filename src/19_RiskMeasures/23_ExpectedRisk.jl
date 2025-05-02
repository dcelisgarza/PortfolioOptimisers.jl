function expected_risk(r::Union{<:WorstRealisation, <:ValueatRisk, <:ValueatRiskRange,
                                <:ConditionalValueatRisk,
                                <:DistributionallyRobustConditionalValueatRisk,
                                <:EntropicValueatRisk, <:EntropicValueatRiskRange,
                                <:RelativisticValueatRisk, <:RelativisticValueatRiskRange,
                                <:DrawdownatRisk, <:MaximumDrawdown, <:AverageDrawdown,
                                <:ConditionalDrawdownatRisk, <:UlcerIndex,
                                <:EntropicDrawdownatRisk, <:RelativisticDrawdownatRisk,
                                <:RelativeDrawdownatRisk, <:RelativeMaximumDrawdown,
                                <:RelativeAverageDrawdown,
                                <:RelativeConditionalDrawdownatRisk, <:RelativeUlcerIndex,
                                <:RelativeEntropicDrawdownatRisk,
                                <:RelativeRelativisticDrawdownatRisk, <:Range,
                                <:ConditionalValueatRiskRange, <:OrderedWeightsArray,
                                <:BrownianDistanceVariance, <:MeanReturn},
                       w::AbstractVector, X::AbstractMatrix,
                       fees::Union{Nothing, <:Fees} = nothing; kwargs...)
    return r(calc_net_returns(w, X, fees))
end
function expected_risk(r::Union{<:LowOrderMoment, <:HighOrderMoment, <:TrackingRiskMeasure,
                                <:SquareRootKurtosis, <:ThirdCentralMoment, <:Skewness},
                       w::AbstractVector, X::AbstractMatrix,
                       fees::Union{Nothing, <:Fees} = nothing; kwargs...)
    return r(w, X, fees)
end
function expected_risk(r::Union{<:StandardDeviation, <:NegativeSkewness,
                                <:TurnoverRiskMeasure, <:Variance, <:UncertaintySetVariance,
                                <:EqualRiskMeasure}, w::AbstractVector, args...; kwargs...)
    return r(w)
end
function number_effective_assets(w::AbstractVector)
    return inv(dot(w, w))
end

export expected_risk, number_effective_assets
