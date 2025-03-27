struct RelativeConditionalDrawdownatRisk{T1 <: HierarchicalRiskMeasureSettings,
                                         T2 <: Real} <: HierarchicalRiskMeasure
    settings::T1
    alpha::T2
end
function RelativeConditionalDrawdownatRisk(;
                                           settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                           alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return RelativeConditionalDrawdownatRisk{typeof(settings), typeof(alpha)}(settings,
                                                                              alpha)
end
function (r::RelativeConditionalDrawdownatRisk)(x::AbstractVector)
    aT = r.alpha * length(x)
    x .= pushfirst!(x, 0) .+ one(eltype(x))
    cs = cumprod(x)
    peak = typemin(eltype(x))
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - 1
    end
    popfirst!(dd)
    idx = ceil(Int, aT)
    var = -partialsort!(dd, idx)
    sum_var = zero(eltype(x))
    for i ∈ 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / aT
end

export RelativeConditionalDrawdownatRisk
