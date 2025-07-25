abstract type ValueatRiskFormulation <: AbstractAlgorithm end
struct MIPValueatRisk{T1 <: Union{Nothing, <:Real}, T2 <: Union{Nothing, <:Real}} <:
       ValueatRiskFormulation
    b::T1
    s::T2
end
function MIPValueatRisk(; b::Union{Nothing, <:Real} = nothing,
                        s::Union{Nothing, <:Real} = nothing)
    bflag = !isnothing(b)
    sflag = !isnothing(s)
    if bflag
        @smart_assert(b > zero(b))
    end
    if sflag
        @smart_assert(s > zero(s))
    end
    if bflag && sflag
        @smart_assert(b > s)
    end
    return MIPValueatRisk{typeof(b), typeof(s)}(b, s)
end
struct DistributionValueatRisk{T1 <: Union{Nothing, <:AbstractVector},
                               T2 <: Union{Nothing, <:AbstractMatrix},
                               T3 <: Distribution} <: ValueatRiskFormulation
    mu::T1
    sigma::T2
    dist::T3
end
function DistributionValueatRisk(; mu::Union{Nothing, <:AbstractVector} = nothing,
                                 sigma::Union{Nothing, <:AbstractMatrix} = nothing,
                                 dist::Distribution = Normal())
    if !isnothing(mu)
        @smart_assert(!isempty(mu))
    end
    if !isnothing(sigma)
        @smart_assert(!isempty(sigma))
    end
    return DistributionValueatRisk{typeof(mu), typeof(sigma), typeof(dist)}(mu, sigma, dist)
end
struct ValueatRisk{T1 <: RiskMeasureSettings, T2 <: Real,
                   T3 <: Union{Nothing, <:AbstractWeights}, T4 <: ValueatRiskFormulation} <:
       RiskMeasure
    settings::T1
    alpha::T2
    w::T3
    alg::T4
end
function ValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                     alpha::Real = 0.05, w::Union{Nothing, <:AbstractWeights} = nothing,
                     alg::ValueatRiskFormulation = MIPValueatRisk())
    @smart_assert(zero(alpha) < alpha < one(alpha))
    if isa(w, AbstractWeights)
        @smart_assert(!isempty(w))
    end
    return ValueatRisk{typeof(settings), typeof(alpha), typeof(w), typeof(alg)}(settings,
                                                                                alpha, w,
                                                                                alg)
end
function factory(r::ValueatRisk, prior::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_factory(r.w, prior.w)
    return ValueatRisk(; settings = r.settings, alpha = r.alpha, w = w, alg = r.alg)
end
function (r::ValueatRisk{<:Any, <:Any, Nothing})(x::AbstractVector)
    return -partialsort!(x, ceil(Int, r.alpha * length(x)))
end
function (r::ValueatRisk{<:Any, <:Any, <:AbstractWeights})(x::AbstractVector)
    w = r.w
    order = sortperm(x)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, r.alpha)
    idx = ifelse(idx > length(x), idx - 1, idx)
    return -sorted_x[idx]
end
struct ValueatRiskRange{T1 <: RiskMeasureSettings, T2 <: Real, T3 <: Real,
                        T4 <: Union{Nothing, <:AbstractWeights},
                        T5 <: ValueatRiskFormulation} <: RiskMeasure
    settings::T1
    alpha::T2
    beta::T3
    w::T4
    alg::T5
end
function ValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                          alpha::Real = 0.05, beta::Real = 0.05,
                          w::Union{Nothing, <:AbstractWeights} = nothing,
                          alg::ValueatRiskFormulation = MIPValueatRisk())
    @smart_assert(zero(alpha) < alpha < one(alpha))
    @smart_assert(zero(beta) < beta < one(beta))
    if isa(w, AbstractWeights)
        @smart_assert(!isempty(w))
    end
    return ValueatRiskRange{typeof(settings), typeof(alpha), typeof(beta), typeof(w),
                            typeof(alg)}(settings, alpha, beta, w, alg)
end
function factory(r::ValueatRiskRange, prior::AbstractPriorResult, args...; kwargs...)
    w = nothing_scalar_array_factory(r.w, prior.w)
    return ValueatRiskRange(; settings = r.settings, alpha = r.alpha, beta = r.beta, w = w,
                            alg = r.alg)
end
function (r::ValueatRiskRange{<:Any, <:Any, <:Any, Nothing})(x::AbstractVector)
    loss = -partialsort!(x, ceil(Int, r.alpha * length(x)))
    gain = -partialsort!(x, ceil(Int, r.beta * length(x)); rev = true)
    return loss - gain
end
function (r::ValueatRiskRange{<:Any, <:Any, <:Any, <:AbstractWeights})(x::AbstractVector)
    w = r.w
    order = sortperm(x)
    sorted_x = view(x, order)
    sorted_w = view(w, order)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, r.alpha)
    idx = ifelse(idx > length(x), idx - 1, idx)
    loss = -sorted_x[idx]

    sorted_x = reverse(sorted_x)
    sorted_w = reverse(sorted_w)
    cum_w = cumsum(sorted_w)
    idx = searchsortedfirst(cum_w, r.beta)
    idx = ifelse(idx > length(x), idx - 1, idx)
    gain = -sorted_x[idx]
    return loss - gain
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
    for (idx, i) in pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i - peak
    end
    popfirst!(x)
    popfirst!(dd)
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
    for (idx, i) in pairs(cs)
        if i > peak
            peak = i
        end
        dd[idx] = i / peak - one(peak)
    end
    popfirst!(x)
    popfirst!(dd)
    return -partialsort!(dd, ceil(Int, r.alpha * length(x)))
end

export MIPValueatRisk, DistributionValueatRisk, ValueatRisk, ValueatRiskRange,
       DrawdownatRisk, RelativeDrawdownatRisk
