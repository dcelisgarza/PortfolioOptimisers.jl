abstract type AbstractBaseRiskMeasure <: AbstractEstimator end

abstract type NoOptimisationRiskMeasure <: AbstractBaseRiskMeasure end
abstract type MuNoOptimisationRiskMeasure <: NoOptimisationRiskMeasure end
abstract type AbstractMomentNoOptimisationRiskMeasure <: MuNoOptimisationRiskMeasure end

abstract type OptimisationRiskMeasure <: AbstractBaseRiskMeasure end
abstract type RiskMeasure <: OptimisationRiskMeasure end
abstract type SigmaRiskMeasure <: RiskMeasure end
abstract type JuMPRiskContributionSigmaRiskMeasure <: SigmaRiskMeasure end
abstract type SolverRiskMeasure <: RiskMeasure end
abstract type AbstractNegativeSkewRiskMeasure <: RiskMeasure end
abstract type SquareRootKurtosisRiskMeasure <: RiskMeasure end
abstract type OrderedWeightsArrayRiskMeasure <: RiskMeasure end
abstract type MuRiskMeasure <: RiskMeasure end
abstract type AbstractMomentRiskMeasure <: MuRiskMeasure end

abstract type HierarchicalRiskMeasure <: OptimisationRiskMeasure end
abstract type SolverHierarchicalRiskMeasure <: HierarchicalRiskMeasure end
abstract type MuHierarchicalRiskMeasure <: HierarchicalRiskMeasure end
abstract type AbstractMomentHierarchicalRiskMeasure <: MuHierarchicalRiskMeasure end

abstract type AbstractRiskMeasureSettings <: AbstractEstimator end
struct RiskMeasureSettings{T1 <: Bool, T2 <: Real, T3 <: Union{Nothing, <:Real}} <:
       AbstractRiskMeasureSettings
    rke::T1
    scale::T2
    ub::T3
end
function RiskMeasureSettings(; rke::Bool = true, scale::Real = 1.0,
                             ub::Union{Nothing, <:Real} = nothing)
    if isa(ub, Real)
        @smart_assert(isfinite(ub) && ub > zero(ub))
    end
    @smart_assert(isfinite(scale))
    return RiskMeasureSettings{typeof(rke), typeof(scale), typeof(ub)}(rke, scale, ub)
end
struct HierarchicalRiskMeasureSettings{T1 <: Real} <: AbstractRiskMeasureSettings
    scale::T1
end
function HierarchicalRiskMeasureSettings(; scale::Real = 1.0)
    return HierarchicalRiskMeasureSettings{typeof(scale)}(scale)
end
const MuRiskMeasures = Union{MuRiskMeasure, MuHierarchicalRiskMeasure,
                             MuNoOptimisationRiskMeasure, SquareRootKurtosisRiskMeasure}
const MomentRiskMeasures = Union{AbstractMomentRiskMeasure,
                                 AbstractMomentHierarchicalRiskMeasure,
                                 AbstractMomentNoOptimisationRiskMeasure}

const SolverRiskMeasures = Union{SolverRiskMeasure, SolverHierarchicalRiskMeasure}
function risk_measure_factory(rs::AbstractBaseRiskMeasure, args...; kwargs...)
    return rs
end
function risk_measure_factory(rs::AbstractVector{<:AbstractBaseRiskMeasure}, args...;
                              kwargs...)
    return risk_measure_factory.(rs, args...; kwargs...)
end
function risk_measure_view(rs::AbstractBaseRiskMeasure, ::Any, ::Any)
    return rs
end
function risk_measure_view(rs::AbstractVector{<:AbstractBaseRiskMeasure}, i::AbstractVector,
                           X::AbstractMatrix)
    return risk_measure_view.(rs, Ref(i), Ref(X))
end
abstract type Scalariser end
struct SumScalariser <: Scalariser end
struct MaxScalariser <: Scalariser end
struct LogSumExpScalariser{T1 <: Real} <: Scalariser
    gamma::T1
end
function LogSumExpScalariser(; gamma::Real = 1.0)
    @smart_assert(gamma > zero(gamma))
    return LogSumExpScalariser{typeof(gamma)}(gamma)
end

export RiskMeasureSettings, HierarchicalRiskMeasureSettings, SumScalariser, MaxScalariser,
       LogSumExpScalariser
