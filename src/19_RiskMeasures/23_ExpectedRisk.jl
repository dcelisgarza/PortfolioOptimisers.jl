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
                                <:OrderedWeightsArrayRange, <:BrownianDistanceVariance,
                                <:MeanReturn}, w::AbstractVector, X::AbstractMatrix,
                       fees::Union{Nothing, <:Fees} = nothing; kwargs...)
    return r(calc_net_returns(w, X, fees))
end
function expected_risk(r::Union{<:LowOrderMoment, <:HighOrderMoment, <:TrackingRiskMeasure,
                                <:RiskTrackingRiskMeasure, <:SquareRootKurtosis,
                                <:ThirdCentralMoment, <:Skewness}, w::AbstractVector,
                       X::AbstractMatrix, fees::Union{Nothing, <:Fees} = nothing; kwargs...)
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
function risk_bounds(r::AbstractBaseRiskMeasure, w1::AbstractVector, w2::AbstractVector,
                     X::AbstractMatrix, fees::Union{Nothing, <:Fees} = nothing; kwargs...)
    r1 = expected_risk(r, w1, X, fees; kwargs...)
    r2 = expected_risk(r, w2, X, fees; kwargs...)
    return r1, r2
end
function risk_contribution(r::AbstractBaseRiskMeasure, w::AbstractVector, X::AbstractMatrix,
                           fees::Union{Nothing, <:Fees} = nothing; delta::Real = 1e-6,
                           marginal::Bool = false, kwargs...)
    N = length(w)
    rc = Vector{eltype(w)}(undef, N)
    ws = Matrix{eltype(w)}(undef, N, 2)
    ws .= w
    id2 = inv(2 * delta)
    for i ∈ eachindex(w)
        ws[i, 1] += delta
        ws[i, 2] -= delta
        r1, r2 = risk_bounds(r, view(ws, :, 1), view(ws, :, 2), X, fees; delta = delta,
                             kwargs...)
        r1 = adjust_risk_contribution(r, r1, delta)
        r2 = adjust_risk_contribution(r, r2, delta)
        rci = (r1 - r2) * id2
        rc[i] = rci
        ws[i, 1] = w[i]
        ws[i, 2] = w[i]
    end
    if !marginal
        rc .*= w
    end
    return rc
end
function factor_risk_contribution(r::AbstractBaseRiskMeasure, w::AbstractVector,
                                  X::AbstractMatrix, fees::Union{Nothing, <:Fees} = nothing;
                                  re::Union{<:RegressionResult,
                                            <:AbstractRegressionEstimator} = StepwiseRegression(),
                                  rd::ReturnsResult = ReturnsResult(), delta::Real = 1e-6,
                                  kwargs...)
    mr = risk_contribution(r, w, X, fees; delta = delta, marginal = true, kwargs...)
    loadings = regression(re, rd.X, rd.F)
    Bt = transpose(loadings.L)
    b2t = transpose(pinv(transpose(nullspace(Bt))))
    b3t = transpose(pinv(b2t))
    rc_f = (Bt * w) .* (transpose(pinv(Bt)) * mr)
    rc_of = sum((b2t * w) .* (b3t * mr))
    rc_f = [rc_f; rc_of]
    return rc_f
end

export number_effective_assets, risk_contribution, factor_risk_contribution
