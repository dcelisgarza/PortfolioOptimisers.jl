struct Range{T1 <: RiskMeasureSettings} <: RiskMeasure
    settings::T1
end
function Range(; settings::RiskMeasureSettings = RiskMeasureSettings())
    return Range{typeof(settings)}(settings)
end
function (::Range)(x::AbstractVector)
    lb, ub = extrema(x)
    return ub - lb
end

export Range
