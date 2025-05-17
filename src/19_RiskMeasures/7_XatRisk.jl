struct ValueatRisk{T1 <: HierarchicalRiskMeasureSettings, T2 <: Real,
                   T3 <: Union{Nothing, <:AbstractWeights}} <: HierarchicalRiskMeasure
    settings::T1
    alpha::T2
    w::T3
end
function ValueatRisk(;
                     settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                     alpha::Real = 0.05, w::Union{Nothing, <:AbstractWeights} = nothing)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    if isa(w, AbstractWeights)
        @smart_assert(!isempty(w))
    end
    return ValueatRisk{typeof(settings), typeof(alpha), typeof(w)}(settings, alpha, w)
end
function risk_measure_factory(r::ValueatRisk, prior::AbstractPriorResult, args...;
                              kwargs...)
    w = risk_measure_nothing_scalar_array_factory(r.w, prior.w)
    return ValueatRisk(; settings = r.settings, alpha = r.alpha, w = w)
end
#TODO: do this for weighted version.
function (r::ValueatRisk)(x::AbstractVector)
    return -partialsort!(x, ceil(Int, r.alpha * length(x)))
end
struct ValueatRiskRange{T1 <: HierarchicalRiskMeasureSettings, T2 <: Real, T3 <: Real,
                        T4 <: Union{Nothing, <:AbstractWeights}} <: HierarchicalRiskMeasure
    settings::T1
    alpha::T2
    beta::T3
    w::T4
end
function ValueatRiskRange(;
                          settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                          alpha::Real = 0.05, beta::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(beta) < beta < one(beta))
    return ValueatRiskRange{typeof(settings), typeof(alpha), typeof(beta), typeof(w)}(settings,
                                                                                      alpha,
                                                                                      beta,
                                                                                      w)
end
function risk_measure_factory(r::ValueatRiskRange, prior::AbstractPriorResult, args...;
                              kwargs...)
    w = risk_measure_nothing_scalar_array_factory(r.w, prior.w)
    return ValueatRiskRange(; settings = r.settings, alpha = r.alpha, beta = r.beta, w = w)
end
#TODO: do this for weighted version.
function (r::ValueatRiskRange)(x::AbstractVector)
    loss = -partialsort!(x, ceil(Int, r.alpha * length(x)))
    gain = partialsort!(x, ceil(Int, r.beta * length(x)); rev = true)
    return loss + gain
end
struct DrawdownatRisk{T1 <: HierarchicalRiskMeasureSettings, T2 <: Real} <:
       HierarchicalRiskMeasure
    settings::T1
    alpha::T2
end
function DrawdownatRisk(;
                        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                        alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return DrawdownatRisk{typeof(settings), typeof(alpha)}(settings, alpha)
end
function (r::DrawdownatRisk)(x::AbstractVector)
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
    popfirst!(dd)
    popfirst!(x)
    return -partialsort!(dd, ceil(Int, r.alpha * length(x)))
end
struct RelativeDrawdownatRisk{T1 <: HierarchicalRiskMeasureSettings, T2 <: Real} <:
       HierarchicalRiskMeasure
    settings::T1
    alpha::T2
end
function RelativeDrawdownatRisk(;
                                settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                alpha::Real = 0.05)
    @smart_assert(zero(alpha) < alpha < one(alpha))
    return RelativeDrawdownatRisk{typeof(settings), typeof(alpha)}(settings, alpha)
end
function (r::RelativeDrawdownatRisk)(x::AbstractVector)
    x .= pushfirst!(x, 0) .+ one(eltype(x))
    cs = cumprod(x)
    peak = typemin(eltype(x))
    dd = similar(cs)
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - one(peak)
    end
    popfirst!(dd)
    popfirst!(x)
    return -partialsort!(dd, ceil(Int, r.alpha * length(x)))
end

export ValueatRisk, ValueatRiskRange, DrawdownatRisk, RelativeDrawdownatRisk
