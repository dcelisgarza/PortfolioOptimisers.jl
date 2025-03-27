struct RelativeMaximumDrawdown{T1 <: HierarchicalRiskMeasureSettings} <:
       HierarchicalRiskMeasure
    settings::T1
end
function RelativeMaximumDrawdown(;
                                 settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings())
    return RelativeMaximumDrawdown{typeof(settings)}(settings)
end
function (::RelativeMaximumDrawdown)(x::AbstractVector)
    x .= pushfirst!(x, 0) .+ one(eltype(x))
    cs = cumprod(x)
    val = zero(eltype(x))
    peak = typemin(eltype(x))
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = one(eltype(x)) - i / peak
        if dd > val
            val = dd
        end
    end
    return val
end

export RelativeMaximumDrawdown
