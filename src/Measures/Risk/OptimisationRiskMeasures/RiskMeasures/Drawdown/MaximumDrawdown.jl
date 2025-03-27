struct MaximumDrawdown{T1 <: RiskMeasureSettings} <: RiskMeasure
    settings::T1
end
function MaximumDrawdown(; settings::RiskMeasureSettings = RiskMeasureSettings())
    return MaximumDrawdown{typeof(settings)}(settings)
end
function (::MaximumDrawdown)(x::AbstractVector)
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = zero(eltype(x))
    peak = typemin(eltype(x))
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = peak - i
        if dd > val
            val = dd
        end
    end
    popfirst!(x)
    return val
end

export MaximumDrawdown
