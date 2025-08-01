struct WorstRealisation{T1} <: RiskMeasure
    settings::T1
end
function WorstRealisation(; settings::RiskMeasureSettings = RiskMeasureSettings())
    return WorstRealisation(settings)
end
function (::WorstRealisation)(x::AbstractVector)
    return -minimum(x)
end

export WorstRealisation
