struct ConditionalDrawdownatRisk{T1 <: RiskMeasureSettings, T2 <: Real} <: RiskMeasure
    settings::T1
    alpha::T2
end
function ConditionalDrawdownatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                   alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return ConditionalDrawdownatRisk{typeof(settings), typeof(alpha)}(settings, alpha)
end
function (r::ConditionalDrawdownatRisk)(x::AbstractVector)
    aT = r.alpha * length(x)
    idx = ceil(Int, aT)
    pushfirst!(x, 1)
    cs = cumsum(x)
    peak = -Inf
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i - peak
    end
    popfirst!(x)
    popfirst!(dd)
    var = -partialsort!(dd, idx)
    sum_var = zero(eltype(x))
    for i ∈ 1:(idx - 1)
        sum_var += dd[i] + var
    end
    return var - sum_var / aT
end

export ConditionalDrawdownatRisk
