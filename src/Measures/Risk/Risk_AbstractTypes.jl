abstract type AbstractRiskMeasure end
function risk_measure_factory(r::AbstractRiskMeasure, args...; kwargs...)
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
function risk_measure_nothing_vec_factory(risk_variable::AbstractVector{<:Real},
                                          prior_variable::AbstractVector{<:Real})
    return if !isempty(risk_variable)
        risk_variable
    elseif !isempty(prior_variable)
        prior_variable
    else
        throw(ArgumentError("Both risk_variable and prior_variable are empty."))
    end
end
function risk_measure_nothing_vec_factory(::Any, prior_variable::AbstractVector{<:Real})
    return if !isempty(prior_variable)
        prior_variable
    else
        throw(ArgumentError("prior_variable is empty."))
    end
end
function risk_measure_nothing_matrix_factory(::Any, prior_variable::AbstractMatrix)
    return if !isempty(prior_variable)
        prior_variable
    else
        throw(ArgumentError("prior_variable is empty."))
    end
end
function risk_measure_nothing_matrix_factory(risk_variable::AbstractMatrix{<:Real},
                                             prior_variable::AbstractMatrix{<:Real})
    return if !isempty(risk_variable)
        risk_variable
    elseif !isempty(prior_variable)
        prior_variable
    else
        throw(ArgumentError("Both risk_variable and prior_variable are empty."))
    end
end
function risk_measure_nothing_real_vec_factory(risk_variable::AbstractVector{<:Real},
                                               cluster::AbstractVector)
    return if !isempty(risk_variable)
        view(risk_variable, cluster)
    else
        risk_variable
    end
end
function risk_measure_nothing_real_vec_factory(risk_variable::Any, ::Any)
    return risk_variable
end
function risk_measure_nothing_vec_factory(risk_variable::AbstractVector{<:Real},
                                          prior_variable::AbstractVector{<:Real},
                                          cluster::AbstractVector)
    return if !isempty(risk_variable)
        view(risk_variable, cluster)
    elseif !isempty(prior_variable)
        view(prior_variable, cluster)
    else
        throw(ArgumentError("Both risk_variable and prior_variable are empty."))
    end
end
function risk_measure_nothing_vec_factory(::Any, prior_variable::AbstractVector{<:Real},
                                          cluster::AbstractVector)
    return if !isempty(prior_variable)
        view(prior_variable, cluster)
    else
        throw(ArgumentError("prior_variable is empty."))
    end
end
function risk_measure_nothing_matrix_factory(::Any, prior_variable::AbstractMatrix,
                                             cluster::AbstractVector)
    return if !isempty(prior_variable)
        view(prior_variable, cluster, cluster)
    else
        throw(ArgumentError("prior_variable is empty."))
    end
end
function risk_measure_nothing_matrix_factory(risk_variable::AbstractMatrix{<:Real},
                                             prior_variable::AbstractMatrix{<:Real},
                                             cluster::AbstractVector)
    return if !isempty(risk_variable)
        view(risk_variable, cluster, cluster)
    elseif !isempty(prior_variable)
        view(prior_variable, cluster, cluster)
    else
        throw(ArgumentError("Both risk_variable and prior_variable are empty."))
    end
end
function risk_measure_solver_factory(risk_solvers::Union{<:Solver,
                                                         <:AbstractVector{<:Solver}},
                                     ::Nothing)
    return risk_solvers
end
function risk_measure_solver_factory(::Nothing,
                                     prior_solvers::Union{<:Solver,
                                                          <:AbstractVector{<:Solver}})
    return prior_solvers
end
function risk_measure_solver_factory(::Nothing, ::Nothing)
    throw(ArgumentError("Both risk_solver and prior_solver are nothing, cannot solve JuMP model."))
end
function fourt_moment_cluster_factory(N::Integer, cluster_index::BitVector)
    idx = Int[]
    cluster = findall(cluster_index)
    Nc = length(cluster)
    sizehint!(idx, Nc^2)
    for c ∈ cluster
        append!(idx, (((c - 1) * N + 1):(c * N))[cluster])
    end
    return idx
end

export RiskMeasureSettings, HierarchicalRiskMeasureSettings, ExactOrderedWeightsArray,
       ApproxOrderedWeightsArray
