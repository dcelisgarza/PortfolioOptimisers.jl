struct EqualRiskMeasure{T1} <: HierarchicalRiskMeasure
    settings::T1
    function EqualRiskMeasure(settings::HierarchicalRiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function EqualRiskMeasure(;
                          settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings())
    return EqualRiskMeasure(settings)
end
function (::EqualRiskMeasure)(w::NumVec)
    return inv(length(w))
end

export EqualRiskMeasure
