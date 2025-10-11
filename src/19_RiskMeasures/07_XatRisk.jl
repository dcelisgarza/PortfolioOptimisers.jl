abstract type ValueatRiskFormulation <: AbstractAlgorithm end
struct MIPValueatRisk{T1, T2} <: ValueatRiskFormulation
    b::T1
    s::T2
    function MIPValueatRisk(b::Union{Nothing, <:Real}, s::Union{Nothing, <:Real})
        bflag = !isnothing(b)
        sflag = !isnothing(s)
        if bflag
            @argcheck(b > zero(b))
        end
        if sflag
            @argcheck(s > zero(s))
        end
        if bflag && sflag
            @argcheck(b > s)
        end
        return new{typeof(b), typeof(s)}(b, s)
    end
end
function MIPValueatRisk(; b::Union{Nothing, <:Real} = nothing,
                        s::Union{Nothing, <:Real} = nothing)
    return MIPValueatRisk(b, s)
end
struct DistributionValueatRisk{T1, T2, T3} <: ValueatRiskFormulation
    mu::T1
    sigma::T2
    dist::T3
    function DistributionValueatRisk(mu::Union{Nothing, <:AbstractVector},
                                     sigma::Union{Nothing, <:AbstractMatrix},
                                     dist::Distribution)
        if !isnothing(mu)
            @argcheck(!isempty(mu))
        end
        if !isnothing(sigma)
            @argcheck(!isempty(sigma))
        end
        return new{typeof(mu), typeof(sigma), typeof(dist)}(mu, sigma, dist)
    end
end
function DistributionValueatRisk(; mu::Union{Nothing, <:AbstractVector} = nothing,
                                 sigma::Union{Nothing, <:AbstractMatrix} = nothing,
                                 dist::Distribution = Normal())
    return DistributionValueatRisk(mu, sigma, dist)
end
struct ValueatRisk{T1, T2, T3, T4} <: RiskMeasure
    settings::T1
    alpha::T2
    w::T3
    alg::T4
    function ValueatRisk(settings::RiskMeasureSettings, alpha::Real,
                         w::Union{Nothing, <:AbstractWeights}, alg::ValueatRiskFormulation)
        @argcheck(zero(alpha) < alpha < one(alpha))
        if isa(w, AbstractWeights)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(alpha), typeof(w), typeof(alg)}(settings, alpha,
                                                                            w, alg)
    end
end
function ValueatRisk(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                     alpha::Real = 0.05, w::Union{Nothing, <:AbstractWeights} = nothing,
                     alg::ValueatRiskFormulation = MIPValueatRisk())
    return ValueatRisk(settings, alpha, w, alg)
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
struct ValueatRiskRange{T1, T2, T3, T4, T5} <: RiskMeasure
    settings::T1
    alpha::T2
    beta::T3
    w::T4
    alg::T5
    function ValueatRiskRange(settings::RiskMeasureSettings, alpha::Real, beta::Real,
                              w::Union{Nothing, <:AbstractWeights},
                              alg::ValueatRiskFormulation)
        @argcheck(zero(alpha) < alpha < one(alpha))
        @argcheck(zero(beta) < beta < one(beta))
        if isa(w, AbstractWeights)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(alpha), typeof(beta), typeof(w), typeof(alg)}(settings,
                                                                                          alpha,
                                                                                          beta,
                                                                                          w,
                                                                                          alg)
    end
end
function ValueatRiskRange(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                          alpha::Real = 0.05, beta::Real = 0.05,
                          w::Union{Nothing, <:AbstractWeights} = nothing,
                          alg::ValueatRiskFormulation = MIPValueatRisk())
    return ValueatRiskRange(settings, alpha, beta, w, alg)
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
struct DrawdownatRisk{T1, T2} <: HierarchicalRiskMeasure
    settings::T1
    alpha::T2
    function DrawdownatRisk(settings::HierarchicalRiskMeasureSettings, alpha::Real)
        @argcheck(zero(alpha) < alpha < one(alpha))
        return new{typeof(settings), typeof(alpha)}(settings, alpha)
    end
end
function DrawdownatRisk(;
                        settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                        alpha::Real = 0.05)
    return DrawdownatRisk(settings, alpha)
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
struct RelativeDrawdownatRisk{T1, T2} <: HierarchicalRiskMeasure
    settings::T1
    alpha::T2
    function RelativeDrawdownatRisk(settings::HierarchicalRiskMeasureSettings, alpha::Real)
        @argcheck(zero(alpha) < alpha < one(alpha))
        return new{typeof(settings), typeof(alpha)}(settings, alpha)
    end
end
function RelativeDrawdownatRisk(;
                                settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                alpha::Real = 0.05)
    return RelativeDrawdownatRisk(settings, alpha)
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
