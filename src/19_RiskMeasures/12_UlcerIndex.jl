@concrete struct UlcerIndex <: RiskMeasure
    settings
    function UlcerIndex(settings::RiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function UlcerIndex(; settings::RiskMeasureSettings = RiskMeasureSettings())
    return UlcerIndex(settings)
end
function (::UlcerIndex)(x::VecNum)
    dd = absolute_drawdown_vec(x)
    return LinearAlgebra.norm(dd, 2) / sqrt(length(x))
end
@concrete struct RelativeUlcerIndex <: HierarchicalRiskMeasure
    settings
    function RelativeUlcerIndex(settings::HierarchicalRiskMeasureSettings)
        return new{typeof(settings)}(settings)
    end
end
function RelativeUlcerIndex(;
                            settings::HierarchicalRiskMeasureSettings = HierarchicalRiskMeasureSettings())
    return RelativeUlcerIndex(settings)
end
function (::RelativeUlcerIndex)(x::VecNum)
    dd = relative_drawdown_vec(x)
    return LinearAlgebra.norm(dd, 2) / sqrt(length(x))
end

export UlcerIndex, RelativeUlcerIndex
