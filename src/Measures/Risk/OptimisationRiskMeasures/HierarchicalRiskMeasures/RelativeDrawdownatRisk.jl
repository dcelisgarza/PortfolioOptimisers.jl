struct RelativeDrawdownatRisk{T1 <: Real} <: HierarchicalRiskMeasure
    settings::HierarchicalRiskMeasureSettings
    alpha::T1
end
function RelativeDrawdownatRisk(;
                                settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return RelativeDrawdownatRisk{typeof(alpha)}(settings, alpha)
end
function (r::RelativeDrawdownatRisk)(x::AbstractVector)
    x .= pushfirst!(x, 0) .+ one(eltype(x))
    cs = cumprod(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - one(peak)
    end
    popfirst!(dd)
    return -partialsort!(dd, ceil(Int, r.alpha * length(x)))
end

export RelativeDrawdownatRisk
