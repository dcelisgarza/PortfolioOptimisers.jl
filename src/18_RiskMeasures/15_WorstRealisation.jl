struct WorstRealisation{T1} <: RiskMeasure
    settings::T1
    function WorstRealisation(settings::RiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function WorstRealisation(; settings::RiskMeasureSettings = RiskMeasureSettings())
    return WorstRealisation(settings)
end
function (::WorstRealisation)(x::VecNum)
    return -minimum(x)
end

export WorstRealisation
