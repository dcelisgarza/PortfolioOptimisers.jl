struct TurnoverRiskMeasure{T1, T2, T3} <: RiskMeasure
    settings::T1
    w::T2
    fixed::T3
    function TurnoverRiskMeasure(settings::RiskMeasureSettings, w::VecNum, fixed::Bool)
        @argcheck(!isempty(w))
        return new{typeof(settings), typeof(w), typeof(fixed)}(settings, w, fixed)
    end
end
function TurnoverRiskMeasure(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                             w::VecNum, fixed::Bool = false)
    return TurnoverRiskMeasure(settings, w, fixed)
end
function (r::TurnoverRiskMeasure)(w::VecNum)
    return LinearAlgebra.norm(r.w - w, 1)
end
function risk_measure_view(r::TurnoverRiskMeasure, i, args...)
    w = view(r.w, i)
    return TurnoverRiskMeasure(; settings = r.settings, w = w, fixed = r.fixed)
end
function needs_previous_weights(r::TurnoverRiskMeasure)
    return !r.fixed
end
function factory(r::TurnoverRiskMeasure; w::Option{<:VecNum} = nothing, kwargs...)
    return if r.fixed || isnothing(w)
        r
    else
        TurnoverRiskMeasure(; settings = r.settings, w = w, fixed = r.fixed)
    end
end

export TurnoverRiskMeasure
