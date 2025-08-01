struct EqualRiskMeasure{T1} <: HierarchicalRiskMeasure
    settings::T1
end
function EqualRiskMeasure(;
                          settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings())
    return EqualRiskMeasure(settings)
end
function (::EqualRiskMeasure)(w::AbstractVector)
    return inv(length(w))
end

export EqualRiskMeasure
