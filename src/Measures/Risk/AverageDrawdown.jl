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
    T = length(x)
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = zero(eltype(x))
    peak = -Inf
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = peak - i
        if dd > 0
            val += dd
        end
    end
    popfirst!(x)
    return val / T
end
function (r::AverageDrawdown{<:Any, <:AbstractWeights})(x::AbstractVector)
    w = r.w
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = zero(eltype(x))
    peak = -Inf
    @smart_assert(length(w) == length(x))
    for (idx, i) ∈ pairs(cs)
        if i > peak
            peak = i
        end
        dd = peak - i
        if dd > 0
            wi = isone(idx) ? 1 : w[idx - 1]
            val += dd * wi
        end
    end
    popfirst!(x)
    return val / sum(w)
end

export AverageDrawdown
