struct AverageDrawdown{T1 <: RiskMeasureSettings,
                       T2 <: Union{Nothing, <:AbstractWeights}} <: RiskMeasure
    settings::T1
    w::T2
end
function AverageDrawdown(; settings::RiskMeasureSettings = RiskMeasureSettings(),
                         w::Union{Nothing, <:AbstractWeights} = nothing)
    return AverageDrawdown{typeof(settings), typeof(w)}(settings, w)
end
function (::AverageDrawdown{<:Any, Nothing})(x::AbstractVector)
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = zero(eltype(x))
    peak = typemin(eltype(x))
    for i ∈ cs
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
function (r::AverageDrawdown{<:Any, <:AbstractWeights})(x::AbstractVector)
    @smart_assert(length(r.w) == length(x))
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = zero(eltype(x))
    peak = typemin(eltype(x))
    for (idx, i) ∈ pairs(cs)
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

export AverageDrawdown
