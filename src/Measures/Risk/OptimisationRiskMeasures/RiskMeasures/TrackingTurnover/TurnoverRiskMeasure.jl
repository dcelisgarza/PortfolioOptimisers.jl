struct TurnoverRiskMeasure{T1 <: RiskMeasureSettings, T2 <: AbstractVector{<:Real}} <:
       RiskMeasure
    settings::T1
    w::T2
end
function TurnoverRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             w::AbstractVector{<:Real})
    @smart_assert(!isempty(w))
    return TurnoverRiskMeasure{typeof(settings), typeof(w)}(settings, w)
end
function (r::TurnoverRiskMeasure)(w::AbstractVector)
    return norm(r.w - w, 1)
end
function cluster_risk_measure_factory(r::TurnoverRiskMeasure; cluster::AbstractVector,
                                      kwargs...)
    w = view(r.w, cluster)
    return TurnoverRiskMeasure(; settings = r.settings, w = w)
end

export TurnoverRiskMeasure
