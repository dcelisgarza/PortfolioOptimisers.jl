struct ValueatRiskRange{T1 <: HierarchicalRiskMeasureSettings, T2 <: Real, T3 <: Real} <:
       HierarchicalRiskMeasure
    settings::T1
    alpha::T2
    beta::T3
end
function ValueatRiskRange(;
                          settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                          alpha::Real = 0.05, beta::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(beta) < beta < one(beta))
    return ValueatRiskRange{typeof(settings), typeof(alpha), typeof(beta)}(settings, alpha,
                                                                           beta)
end
function (r::ValueatRiskRange)(x::AbstractVector)
    loss = -partialsort!(x, ceil(Int, r.alpha * length(x)))
    gain = partialsort!(x, ceil(Int, r.beta * length(x)); rev = true)
    return loss + gain
end

export ValueatRiskRange
