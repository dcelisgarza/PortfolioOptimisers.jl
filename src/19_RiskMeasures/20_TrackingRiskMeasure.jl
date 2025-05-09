struct TrackingRiskMeasure{T1 <: RiskMeasureSettings, T2 <: AbstractTracking} <: RiskMeasure
    settings::T1
    tracking::T2
end
function TrackingRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             tracking::AbstractTracking)
    return TrackingRiskMeasure{typeof(settings), typeof(tracking)}(settings, tracking)
end
function (r::TrackingRiskMeasure)(w::AbstractVector, X::AbstractMatrix,
                                  fees::Union{Nothing, <:Fees} = nothing)
    benchmark = tracking_benchmark(r.tracking, X)
    return norm(calc_net_returns(w, X, fees) - benchmark) / sqrt(size(X, 1) - 1)
end
function risk_measure_view(r::TrackingRiskMeasure, i::AbstractVector, args...; kwargs...)
    tracking = tracking_view(r.tracking, i)
    return TrackingRiskMeasure(; settings = r.settings, tracking = tracking)
end

export TrackingRiskMeasure
