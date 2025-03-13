struct Range{T1 <: RiskMeasureSettings} <: RiskMeasure
    settings::T1
end
function Range(; settings::RiskMeasureSettings = RiskMeasureSettings())
    return Range{typeof(settings)}(settings)
end
function (::Range)(x::AbstractVector)
    lo, hi = extrema(x)
    return hi - lo
end

export Range
