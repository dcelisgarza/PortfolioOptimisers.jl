struct TurnoverRiskMeasure{T1, T2} <: RiskMeasure
    settings::T1
    w::T2
    function TurnoverRiskMeasure(settings::RiskMeasureSettings, w::NumVec)
        @argcheck(!isempty(w))
        return new{typeof(settings), typeof(w)}(settings, w)
    end
end
function TurnoverRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             w::NumVec)
    return TurnoverRiskMeasure(settings, w)
end
function (r::TurnoverRiskMeasure)(w::NumVec)
    return norm(r.w - w, 1)
end
function risk_measure_view(r::TurnoverRiskMeasure, i::NumVec, args...)
    w = view(r.w, i)
    return TurnoverRiskMeasure(; settings = r.settings, w = w)
end
function factory(r::TurnoverRiskMeasure, w::NumVec)
    return TurnoverRiskMeasure(; settings = r.settings, w = w)
end

export TurnoverRiskMeasure
