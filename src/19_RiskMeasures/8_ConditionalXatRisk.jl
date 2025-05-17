struct ConditionalValueatRisk{T1 <: RiskMeasureSettings, T2 <: Real,
                              T3 <: Union{Nothing, <:AbstractVector}} <: RiskMeasure
    settings::T1
    alpha::T2
    w::T3
end
function ConditionalValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                                alpha::Real = 0.05,
                                w::Union{Nothing, <:AbstractVector} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    if isa(w, AbstractVector)
        @smart_assert(!isempty(w))
    end
    return ConditionalValueatRisk{typeof(settings), typeof(alpha), typeof(w)}(settings,
                                                                              alpha, w)
end
function risk_measure_factory(r::ConditionalValueatRisk, prior::AbstractPriorResult,
                              args...; kwargs...)
    w = risk_measure_nothing_scalar_array_factory(r.w, prior.w)
    return ConditionalValueatRisk(; settings = r.settings, alpha = r.alpha, w = r.w)
end
#! TODO add version of this that uses weights
function (r::ConditionalValueatRisk{<:Any, <:Any, <:Nothing})(x::AbstractVector)
    aT = r.alpha * length(x)
    idx = ceil(Int, aT)
    var = -partialsort!(x, idx)
    sum_var = zero(eltype(x))
    for i ∈ 1:(idx - 1)
        sum_var += x[i] + var
    end
    return var - sum_var / aT
end
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
    peak = typemin(eltype(x))
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

export ConditionalValueatRisk, DistributionallyRobustConditionalValueatRisk,
       ConditionalValueatRiskRange, ConditionalDrawdownatRisk,
       RelativeConditionalDrawdownatRisk
