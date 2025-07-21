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
Base.length(::AbstractBaseRiskMeasure) = 1
Base.iterate(rs::AbstractBaseRiskMeasure, state = 1) = state > 1 ? nothing : (rs, state + 1)
struct Frontier{T1 <: Integer, T2 <: Real, T3 <: Bool} <: AbstractAlgorithm
    N::T1
    factor::T2
    flag::T3
end
function Frontier(; N::Integer = 20)
    @smart_assert(N > zero(N))
    factor = 1
    flag = true
    return Frontier{typeof(N), typeof(factor), typeof(flag)}(N, 1, true)
end
function _Frontier(; N::Integer = 20, factor::Real, flag::Bool)
    @smart_assert(N > zero(N))
    @smart_assert(isfinite(factor) && factor > zero(factor))
    return Frontier{typeof(N), typeof(factor), typeof(flag)}(N, factor, flag)
end
struct RiskMeasureSettings{T1 <: Real,
                           T2 <: Union{Nothing, <:Real, <:AbstractVector, <:Frontier},
                           T3 <: Bool} <: AbstractRiskMeasureSettings
    scale::T1
    ub::T2
    rke::T3
end
function RiskMeasureSettings(; scale::Real = 1.0,
                             ub::Union{Nothing, <:Real, <:AbstractVector, <:Frontier} = nothing,
                             rke::Bool = true)
    if isa(ub, Real)
        @smart_assert(isfinite(ub) && ub > zero(ub))
    elseif isa(ub, AbstractVector)
        @smart_assert(!isempty(ub) && all(isfinite, ub) && all(x -> x > zero(x), ub))
    end
    @smart_assert(isfinite(scale))
    return RiskMeasureSettings{typeof(scale), typeof(ub), typeof(rke)}(scale, ub, rke)
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
function factory(rs::AbstractBaseRiskMeasure, args...; kwargs...)
    return rs
end
function factory(rs::AbstractVector{<:AbstractBaseRiskMeasure}, args...; kwargs...)
    return [factory(r, args...; kwargs...) for r in rs]
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
function expected_risk end
function no_bounds_risk_measure end
function no_bounds_no_risk_expr_risk_measure end

export Frontier, RiskMeasureSettings, HierarchicalRiskMeasureSettings, SumScalariser,
       MaxScalariser, LogSumExpScalariser, expected_risk
