abstract type AbstractRiskMeasure end
function risk_measure_factory(r::AbstractRiskMeasure, args...; kwargs...)
    return r
end
function cluster_risk_measure_factory(r::AbstractRiskMeasure, args...; kwargs...)
    return r
end
abstract type AbstractRiskMeasureSettings end
struct RiskMeasureSettings{T1 <: Bool, T2 <: Real, T3 <: Union{Nothing, <:Real}} <:
       AbstractRiskMeasureSettings
    rke::T1
    scale::T2
    ub::T3
end
function RiskMeasureSettings(; rke::Bool = true, scale::Real = 1.0,
                             ub::Union{Nothing, <:Real} = nothing)
    return RiskMeasureSettings{typeof(rke), typeof(scale), typeof(ub)}(rke, scale, ub)
end
struct HierarchicalRiskMeasureSettings{T1 <: Real} <: AbstractRiskMeasureSettings
    scale::T1
end
function HierarchicalRiskMeasureSettings(; scale::Real = 1.0)
    return HierarchicalRiskMeasureSettings{typeof(scale)}(scale)
end

abstract type OptimisationRiskMeasure <: AbstractRiskMeasure end
abstract type RiskMeasure <: OptimisationRiskMeasure end
abstract type HierarchicalRiskMeasure <: OptimisationRiskMeasure end
abstract type OrderedWeightsArrayFormulation end
abstract type SolverRiskMeasure <: RiskMeasure end
abstract type SolverHierarchicalRiskMeasure <: HierarchicalRiskMeasure end
abstract type SigmaRiskMeasure <: RiskMeasure end
abstract type RiskContributionSigmaRiskMeasure <: SigmaRiskMeasure end
abstract type SkewRiskMeasure <: RiskMeasure end
abstract type KurtosisRiskMeasure <: RiskMeasure end
abstract type OrderedWeightsArrayRiskMeasure <: RiskMeasure end
abstract type MuRiskMeasure <: RiskMeasure end
abstract type MuHierarchicalRiskMeasure <: HierarchicalRiskMeasure end
abstract type TargetRiskMeasure <: MuRiskMeasure end
abstract type TargetHierarchicalRiskMeasure <: MuHierarchicalRiskMeasure end
abstract type NoOptimisationRiskMeasure <: AbstractRiskMeasure end
abstract type MuNoOptimisationRiskMeasure <: NoOptimisationRiskMeasure end
abstract type TargetNoOptimisationRiskMeasure <: MuNoOptimisationRiskMeasure end
struct ExactOrderedWeightsArray <: OrderedWeightsArrayFormulation end
struct ApproxOrderedWeightsArray{T1 <: AbstractVector{<:Real}} <:
       OrderedWeightsArrayFormulation
    p::T1
end
function ApproxOrderedWeightsArray(; p::AbstractVector{<:Real} = Float64[2, 3, 4, 10, 50])
    @smart_assert(!isempty(p))
    @smart_assert(all(p .> zero(eltype(p))))
    return ApproxOrderedWeightsArray{typeof(p)}(p)
end
const MuRiskMeasures = Union{MuRiskMeasure, MuHierarchicalRiskMeasure,
                             MuNoOptimisationRiskMeasure, KurtosisRiskMeasure}
function calc_ret_mu(x::AbstractVector, w::AbstractVector, rm::MuRiskMeasures)
    mu = rm.mu
    return mu = if isnothing(mu)
        wi = rm.w
        isnothing(wi) ? mean(x) : mean(x, wi)
    else
        dot(mu, w)
    end
end
const TargetRiskMeasures = Union{TargetRiskMeasure, TargetHierarchicalRiskMeasure,
                                 TargetNoOptimisationRiskMeasure}
function calc_target_ret_mu(x::AbstractVector, w::AbstractVector, rm::TargetRiskMeasures)
    target = rm.target
    if isnothing(target)
        target = calc_ret_mu(x, w, rm)
    end
    return target
end
const SolverRiskMeasures = Union{SolverRiskMeasure, SolverHierarchicalRiskMeasure}

export RiskMeasureSettings, HierarchicalRiskMeasureSettings, ExactOrderedWeightsArray,
       ApproxOrderedWeightsArray
