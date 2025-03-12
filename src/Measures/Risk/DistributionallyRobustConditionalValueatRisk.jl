struct DistributionallyRobustConditionalValueatRisk{T1 <: RiskMeasureSettings, T2 <: Real,
                                                    T3 <: Real, T4 <: Real} <: RiskMeasure
    settings::T1
    alpha::T2
    l::T3
    r::T4
end
function DistributionallyRobustConditionalValueatRisk(;
                                                      settings::RiskMeasureSettings = RiskMeasureSettings(),
                                                      l::Real = 1.0, alpha::Real = 0.05,
                                                      r::Real = 0.02)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return DistributionallyRobustConditionalValueatRisk{typeof(settings), typeof(alpha),
                                                        typeof(l), typeof(r)}(settings,
                                                                              alpha, l, r)
end
function (r::DistributionallyRobustConditionalValueatRisk)(x::AbstractVector)
    aT = r.alpha * length(x)
    idx = ceil(Int, aT)
    var = -partialsort!(x, idx)
    sum_var = zero(eltype(x))
    for i ∈ 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / aT
end

export DistributionallyRobustConditionalValueatRisk
