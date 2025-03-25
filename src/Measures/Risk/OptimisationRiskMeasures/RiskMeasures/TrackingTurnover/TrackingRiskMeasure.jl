struct TrackingRiskMeasure{T1 <: RiskMeasureSettings, T2 <: AbstractTracking} <: RiskMeasure
    settings::T1
    tracking::T2
end
function TrackingRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             tracking::AbstractTracking)
    return TrackingRiskMeasure{typeof(settings), typeof(tracking)}(settings, tracking)
end
function (r::TrackingRiskMeasure)(w::AbstractVector, X::AbstractMatrix, fees::Fees = Fees())
    benchmark = tracking_benchmark(r.tracking, X)
    return norm(calc_net_returns(w, X, fees) - benchmark) / sqrt(size(X, 1) - 1)
end
function cluster_risk_measure_factory(r::TrackingRiskMeasure{<:Any, <:WeightsTracking};
                                      cluster::AbstractVector, kwargs...)
    fees = cluster_fees_factory(r.tracking.fees, cluster)
    turnover = cluster_turnover_factory(r.tracking.turnover, cluster)
    w = view(r.tracking.w, cluster)
    return TrackingRiskMeasure(; settings = r.settings,
                               tracking = WeightsTracking(; fees = fees,
                                                          turnover = turnover, w = w))
end

export TrackingRiskMeasure
