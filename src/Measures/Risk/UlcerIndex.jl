struct UCI{T1 <: RiskMeasureSettings} <: RiskMeasure
    settings::T1
end
function UCI(; settings::RiskMeasureSettings = RiskMeasureSettings())
    return UCI{typeof(settings)}(settings)
end
function (::UCI)(x::AbstractVector)
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
            val += dd^2
        end
    end
    popfirst!(x)
    return sqrt(val / T)
end
