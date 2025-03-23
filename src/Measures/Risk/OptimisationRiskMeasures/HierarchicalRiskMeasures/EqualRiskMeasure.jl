struct EqualRiskMeasure{T1 <: HierarchicalRiskMeasureSettings} <: HierarchicalRiskMeasure
    settings::T1
end
function EqualRiskMeasure(;
                          settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings())
    return EqualRiskMeasure{typeof(settings)}(settings)
end
function (::EqualRiskMeasure)(w::AbstractVector, delta::Real = 0)
    return inv(length(w)) + delta
end

export EqualRiskMeasure
