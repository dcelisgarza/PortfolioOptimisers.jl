struct RiskRatioRiskMeasure{T1,T2} <: HierarchicalRiskMeasure
    r1::T1
    r2::T2
end
function RiskRatioRiskMeasure(;
    r1::AbstractBaseRiskMeasure = Variance(),
    r2::AbstractBaseRiskMeasure = ConditionalValueatRisk(),
)
    return RiskRatioRiskMeasure(r1, r2)
end
function factory(r::RiskRatioRiskMeasure, pr::AbstractPriorResult, args...; kwargs...)
    r1 = factory(r.r1, pr, args...; kwargs...)
    r2 = factory(r.r2, pr, args...; kwargs...)
    return RiskRatioRiskMeasure(; r1 = r1, r2 = r2)
end
function factory(r::RiskRatioRiskMeasure, w::AbstractVector)
    return RiskRatioRiskMeasure(; r1 = factory(r.r1, w), r2 = factory(r.r2, w))
end

export RiskRatioRiskMeasure
