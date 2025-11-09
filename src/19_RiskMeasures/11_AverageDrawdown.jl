struct AverageDrawdown{T1, T2} <: RiskMeasure
    settings::T1
    w::T2
    function AverageDrawdown(settings::RiskMeasureSettings, w::Option{<:AbstractWeights})
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(w)}(settings, w)
    end
end
function AverageDrawdown(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         w::Option{<:AbstractWeights} = nothing)
    return AverageDrawdown(settings, w)
end
function (::AverageDrawdown{<:Any, Nothing})(x::VecNum)
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = zero(eltype(x))
    peak = typemin(eltype(x))
    for i in cs
        if i > peak
            peak = i
        end
        dd = peak - i
        if dd > zero(dd)
            val += dd
        end
    end
    popfirst!(x)
    return val / length(x)
end
function (r::AverageDrawdown{<:Any, <:AbstractWeights})(x::VecNum)
    @argcheck(length(r.w) == length(x))
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = zero(eltype(x))
    peak = typemin(eltype(x))
    for (idx, i) in pairs(cs)
        if i > peak
            peak = i
        end
        dd = peak - i
        if dd > zero(dd)
            wi = isone(idx) ? one(eltype(r.w)) : r.w[idx - 1]
            val += dd * wi
        end
    end
    popfirst!(x)
    return val / sum(r.w)
end
struct RelativeAverageDrawdown{T1, T2} <: HierarchicalRiskMeasure
    settings::T1
    w::T2
    function RelativeAverageDrawdown(settings::HierarchicalRiskMeasureSettings,
                                     w::Option{<:AbstractWeights})
        if !isnothing(w)
            @argcheck(!isempty(w))
        end
        return new{typeof(settings), typeof(w)}(settings, w)
    end
end
function RelativeAverageDrawdown(;
                                 settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings(),
                                 w::Option{<:AbstractWeights} = nothing)
    return RelativeAverageDrawdown(settings, w)
end
function (r::RelativeAverageDrawdown{<:Any, Nothing})(x::VecNum)
    x .= pushfirst!(x, 0) .+ one(eltype(x))
    cs = cumprod(x)
    val = zero(eltype(x))
    peak = typemin(eltype(x))
    for i in cs
        if i > peak
            peak = i
        end
        dd = one(eltype(x)) - i / peak
        if dd > zero(dd)
            val += dd
        end
    end
    popfirst!(x)
    return val / length(x)
end
function (r::RelativeAverageDrawdown{<:Any, <:AbstractWeights})(x::VecNum)
    @argcheck(length(r.w) == length(x))
    x .= pushfirst!(x, 0) .+ one(eltype(x))
    cs = cumprod(x)
    val = zero(eltype(x))
    peak = typemin(eltype(x))
    for (idx, i) in pairs(cs)
        if i > peak
            peak = i
        end
        dd = one(eltype(x)) - i / peak
        if dd > zero(dd)
            wi = isone(idx) ? one(eltype(r.w)) : r.w[idx - 1]
            val += dd * wi
        end
    end
    popfirst!(x)
    return val / sum(r.w)
end
for r in (AverageDrawdown, RelativeAverageDrawdown)
    eval(quote
             function factory(r::$(r), prior::AbstractPriorResult, args...; kwargs...)
                 w = nothing_scalar_array_factory(r.w, prior.w)
                 return $(r)(; settings = r.settings, w = w)
             end
         end)
end

export AverageDrawdown, RelativeAverageDrawdown
