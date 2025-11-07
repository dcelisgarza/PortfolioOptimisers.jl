struct UlcerIndex{T1} <: RiskMeasure
    settings::T1
    function UlcerIndex(settings::RiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function UlcerIndex(; settings::RiskMeasureSettings = RiskMeasureSettings())
    return UlcerIndex(settings)
end
function (::UlcerIndex)(x::NumVec)
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = zero(eltype(x))
    peak = typemin(eltype(x))
    for i in cs
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
struct RelativeUlcerIndex{T1} <: HierarchicalRiskMeasure
    settings::T1
    function RelativeUlcerIndex(settings::HierarchicalRiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function RelativeUlcerIndex(;
                            settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings())
    return RelativeUlcerIndex(settings)
end
function (::RelativeUlcerIndex)(x::NumVec)
    x .= pushfirst!(x, 0) .+ one(eltype(x))
    cs = cumprod(x)
    val = zero(eltype(x))
    peak = typemin(eltype(x))
    for i in cs
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

export UlcerIndex, RelativeUlcerIndex
