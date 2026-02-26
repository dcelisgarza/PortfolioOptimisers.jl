struct RiskRatioRiskMeasure{T1, T2} <: HierarchicalRiskMeasure
    r1::T1
    r2::T2
    function RiskRatioRiskMeasure(r1::OptimisationRiskMeasure, r2::OptimisationRiskMeasure)
        return new{typeof(r1), typeof(r2)}(r1, r2)
    end
end
function RiskRatioRiskMeasure(; r1::OptimisationRiskMeasure = Variance(),
                              r2::OptimisationRiskMeasure = ConditionalValueatRisk())
    return RiskRatioRiskMeasure(r1, r2)
end
function factory(r::RiskRatioRiskMeasure, args...; kwargs...)
    r1 = factory(r.r1, args...; kwargs...)
    r2 = factory(r.r2, args...; kwargs...)
    return RiskRatioRiskMeasure(; r1 = r1, r2 = r2)
end
function factory(r::RiskRatioRiskMeasure, w::VecNum)
    return RiskRatioRiskMeasure(; r1 = factory(r.r1, w), r2 = factory(r.r2, w))
end
struct NonOptimisationRiskRatioRiskMeasure{T1, T2} <: NonOptimisationRiskMeasure
    r1::T1
    r2::T2
    function NonOptimisationRiskRatioRiskMeasure(r1::AbstractBaseRiskMeasure,
                                                 r2::AbstractBaseRiskMeasure)
        return new{typeof(r1), typeof(r2)}(r1, r2)
    end
end
function NonOptimisationRiskRatioRiskMeasure(; r1::AbstractBaseRiskMeasure = Variance(),
                                             r2::AbstractBaseRiskMeasure = ConditionalValueatRisk())
    return NonOptimisationRiskRatioRiskMeasure(r1, r2)
end
function factory(r::NonOptimisationRiskRatioRiskMeasure, args...; kwargs...)
    r1 = factory(r.r1, args...; kwargs...)
    r2 = factory(r.r2, args...; kwargs...)
    return NonOptimisationRiskRatioRiskMeasure(; r1 = r1, r2 = r2)
end
#=
function factory(r::NonOptimisationRiskRatioRiskMeasure, w::VecNum)
    return NonOptimisationRiskRatioRiskMeasure(; r1 = factory(r.r1, w),
                                               r2 = factory(r.r2, w))
end
=#

export RiskRatioRiskMeasure, NonOptimisationRiskRatioRiskMeasure
