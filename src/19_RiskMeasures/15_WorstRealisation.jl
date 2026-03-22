@concrete struct WorstRealisation <: RiskMeasure
    settings
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
