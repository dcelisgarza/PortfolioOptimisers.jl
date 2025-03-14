mutable struct Equal{T1 <: HierarchicalRiskMeasureSettings} <: HierarchicalRiskMeasure
    settings::T1
end
function Equal(;
               settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings())
    return Equal{typeof(settings)}(settings)
end
function (::Equal)(w::AbstractVector, delta::Real = 0)
    return inv(length(w)) + delta
end
