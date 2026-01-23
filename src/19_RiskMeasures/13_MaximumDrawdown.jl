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
    dd = relative_drawdown_vec(x)
    return -minimum(dd)
end

export MaximumDrawdown, RelativeMaximumDrawdown
