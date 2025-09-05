struct Range{T1} <: RiskMeasure
    settings::T1
end
function Range(; settings::RiskMeasureSettings = RiskMeasureSettings())
    return Range(settings)
end
function (::Range)(x::AbstractVector)
    lb, ub = extrema(x)
    return ub - lb
end

export Range
