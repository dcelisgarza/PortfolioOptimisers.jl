struct RelativeUlcerIndex{T1 <: HierarchicalRiskMeasureSettings} <: HierarchicalRiskMeasure
    settings::T1
end
function RelativeUlcerIndex(;
                            settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings())
    return RelativeUlcerIndex{typeof(settings)}(settings)
end
function (::RelativeUlcerIndex)(x::AbstractVector)
    x .= pushfirst!(x, 0) .+ one(eltype(x))
    cs = cumprod(x)
    val = zero(eltype(x))
    peak = typemin(eltype(x))
    for i ∈ cs
        if i > peak
            peak = i
        end
        dd = one(eltype(x)) - i / peak
        if dd > zero(dd)
            val += dd^2
        end
    end
    return sqrt(val / length(x))
end

export RelativeUlcerIndex
