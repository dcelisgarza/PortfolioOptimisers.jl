struct MaximumDrawdown{T1} <: RiskMeasure
    settings::T1
    function MaximumDrawdown(settings::RiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function MaximumDrawdown(; settings::RiskMeasureSettings = RiskMeasureSettings())
    return MaximumDrawdown(settings)
end
function (::MaximumDrawdown)(x::VecNum)
    #=
    pushfirst!(x, 1)
    cs = cumsum(x)
    val = zero(eltype(x))
    peak = typemin(eltype(x))
    for i in cs
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
    =#
    dd = absolute_drawdown_vec(x)
    return -minimum(dd)
end
struct RelativeMaximumDrawdown{T1} <: HierarchicalRiskMeasure
    settings::T1
    function RelativeMaximumDrawdown(settings::HierarchicalRiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function RelativeMaximumDrawdown(;
                                 settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings())
    return RelativeMaximumDrawdown(settings)
end
function (::RelativeMaximumDrawdown)(x::VecNum)
    #=
    x .= pushfirst!(x, 0) .+ one(eltype(x))
    cs = cumprod(x)
    val = zero(eltype(x))
    peak = typemin(eltype(x))
    for i in cs
        if i > peak
            peak = i
        end
        dd = one(eltype(x)) - i / peak
        if dd > val
            val = dd
        end
    end
    popfirst!(x)
    return val
    =#
    dd = relative_drawdown_vec(x)
    return -minimum(dd)
end

export MaximumDrawdown, RelativeMaximumDrawdown
