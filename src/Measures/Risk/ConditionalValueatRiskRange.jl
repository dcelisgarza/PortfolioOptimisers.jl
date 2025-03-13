struct ConditionalValueatRiskRange{T1 <: RiskMeasureSettings, T2 <: Real, T3 <: Real} <:
       RiskMeasure
    settings::T1
    alpha::T2
    beta::T3
end
function ConditionalValueatRiskRange(;
                                     settings::RiskMeasureSettings = RiskMeasureSettings(),
                                     alpha::Real = 0.05, beta::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(beta) < beta < one(beta))
    return ConditionalValueatRiskRange{typeof(settings), typeof(alpha), typeof(beta)}(settings,
                                                                                      alpha,
                                                                                      beta)
end
function (r::ConditionalValueatRiskRange)(x::AbstractVector)
    aT = r.alpha * length(x)
    idx1 = ceil(Int, aT)
    var1 = -partialsort!(x, idx1)
    sum_var1 = zero(eltype(x))
    for i ∈ 1:(idx1 - 1)
        sum_var1 += x[i] + var1
    end
    loss = var1 - sum_var1 / aT

    bT = r.beta * length(x)
    idx2 = ceil(Int, bT)
    var2 = -partialsort!(x, idx2; rev = true)
    sum_var2 = zero(eltype(x))
    for i ∈ 1:(idx2 - 1)
        sum_var2 += x[i] + var2
    end
    gain = var2 - sum_var2 / bT

    return loss - gain
end

export ConditionalValueatRiskRange
