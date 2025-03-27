struct UlcerIndex{T1 <: RiskMeasureSettings} <: RiskMeasure
    settings::T1
end
function UlcerIndex(; settings::RiskMeasureSettings = RiskMeasureSettings())
    return UlcerIndex{typeof(settings)}(settings)
end
function (::UlcerIndex)(x::AbstractVector)
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
            val += dd^2
        end
    end
    popfirst!(x)
    return sqrt(val / length(x))
end
