abstract type AbstractRiskMeasure end
function risk_measure_factory(r::AbstractRiskMeasure, args...; kwargs...)
    return r
end
function cluster_risk_measure_factory(r::AbstractRiskMeasure, args...; kwargs...)
    return r
end
abstract type OptimisationRiskMeasure <: AbstractRiskMeasure end
abstract type NoOptimisationRiskMeasure <: AbstractRiskMeasure end
abstract type AbstractRiskMeasureSettings end
abstract type RiskMeasure <: OptimisationRiskMeasure end
abstract type HierarchicalRiskMeasure <: OptimisationRiskMeasure end
abstract type OrderedWeightsArrayFormulation end
abstract type SolverRiskMeasure <: RiskMeasure end
abstract type SolverHierarchicalRiskMeasure <: HierarchicalRiskMeasure end
abstract type SigmaRiskMeasure <: RiskMeasure end
abstract type RiskContributionSigmaRiskMeasure <: SigmaRiskMeasure end
abstract type SkewRiskMeasure <: RiskMeasure end
abstract type OrderedWeightsArrayRiskMeasure <: RiskMeasure end
abstract type MuRiskMeasure <: RiskMeasure end
abstract type MuHierarchicalRiskMeasure <: HierarchicalRiskMeasure end
abstract type MuNoOptimisationRiskMeasure <: NoOptimisationRiskMeasure end
abstract type TargetRiskMeasure <: MuRiskMeasure end
abstract type TargetHierarchicalRiskMeasure <: MuHierarchicalRiskMeasure end
struct RiskMeasureSettings{T1 <: Bool, T2 <: Real, T3 <: Real} <:
       AbstractRiskMeasureSettings
    flag::T1
    scale::T2
    ub::T3
end
function RiskMeasureSettings(; flag::Bool = true, scale::Real = 1.0, ub::Real = Inf)
    return RiskMeasureSettings{typeof(flag), typeof(scale), typeof(ub)}(flag, scale, ub)
end
struct HierarchicalRiskMeasureSettings{T1 <: Real} <: AbstractRiskMeasureSettings
    scale::T1
end
function HierarchicalRiskMeasureSettings(; scale::Real = 1.0)
    return HierarchicalRiskMeasureSettings{typeof(scale)}(scale)
end
struct ExactOrderedWeightsArray <: OrderedWeightsArrayFormulation end
struct ApproxOrderedWeightsArray{T1 <: AbstractVector{<:Real}} <:
       OrderedWeightsArrayFormulation
    p::T1
end
function ApproxOrderedWeightsArray(; p::AbstractVector{<:Real} = Float64[2, 3, 4, 10, 50])
    return ApproxOrderedWeightsArray{typeof(p)}(p)
end
const MuRiskMeasures = Union{MuRiskMeasure, MuHierarchicalRiskMeasure,
                             MuNoOptimisationRiskMeasure}
function calc_ret_mu(x::AbstractVector, w::AbstractVector, rm::MuRiskMeasures)
    mu = rm.mu
    return mu = if isnothing(mu) || isempty(mu)
        wi = rm.w
        isnothing(wi) ? mean(x) : mean(x, wi)
    else
        dot(mu, w)
    end
end
const TargetRiskMeasures = Union{TargetRiskMeasure, TargetHierarchicalRiskMeasure}
function calc_target_ret_mu(x::AbstractVector, w::AbstractVector, rm::TargetRiskMeasures)
    target = rm.target
    if isnothing(target) || isa(target, AbstractVector) && isempty(target)
        target = calc_ret_mu(x, w, rm)
    end
    return target
end
const SolverRiskMeasures = Union{SolverRiskMeasure, SolverHierarchicalRiskMeasure}

export RiskMeasureSettings, HierarchicalRiskMeasureSettings, ExactOrderedWeightsArray,
       ApproxOrderedWeightsArray
