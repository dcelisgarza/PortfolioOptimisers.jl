struct ValueatRisk{T1 <: HierarchicalRiskMeasureSettings, T2 <: Real} <:
       HierarchicalRiskMeasure
    settings::T1
    alpha::T2
end
function ValueatRisk(;
                     settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                     alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return ValueatRisk{typeof(settings), typeof(alpha)}(settings, alpha)
end
function (r::ValueatRisk)(x::AbstractVector)
    return -partialsort!(x, ceil(Int, r.alpha * length(x)))
end

export ValueatRisk
