struct TurnoverRiskMeasure{T1,T2} <: RiskMeasure
    settings::T1
    w::T2
end
function TurnoverRiskMeasure(;
    settings::RiskMeasureSettings = RiskMeasureSettings(),
    w::AbstractVector{<:Real},
)
    @argcheck(!isempty(w))
    return TurnoverRiskMeasure(settings, w)
end
function (r::TurnoverRiskMeasure)(w::AbstractVector)
    return norm(r.w - w, 1)
end
function risk_measure_view(r::TurnoverRiskMeasure, i::AbstractVector, args...)
    w = view(r.w, i)
    return TurnoverRiskMeasure(; settings = r.settings, w = w)
end
function factory(r::TurnoverRiskMeasure, w::AbstractVector)
    return TurnoverRiskMeasure(; settings = r.settings, w = w)
end

export TurnoverRiskMeasure
