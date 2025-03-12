mutable struct ConditionalValueatRisk{T1 <: RiskMeasureSettings, T2 <: Real} <: RiskMeasure
    settings::T1
    alpha::T2
end
function ConditionalValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return ConditionalValueatRisk{typeof(settings), typeof(alpha)}(settings, alpha)
end
function (r::ConditionalValueatRisk)(x::AbstractVector)
    aT = r.alpha * length(x)
    idx = ceil(Int, aT)
    var = -partialsort!(x, idx)
    sum_var = zero(eltype(x))
    for i ∈ 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / aT
end

export ConditionalValueatRisk
