struct WorstRealisation{T1 <: RiskMeasureSettings} <: RiskMeasure
    settings::T1
end
function WorstRealisation(; settings::RiskMeasureSettings = RiskMeasureSettings())
    return WorstRealisation{typeof(settings)}(settings)
end
function (::WorstRealisation)(x::AbstractVector)
    return -minimum(x)
end

export WorstRealisation
