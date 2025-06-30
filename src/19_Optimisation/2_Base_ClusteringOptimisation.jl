abstract type BaseClusteringOptimisationEstimator <: BaseOptimisationEstimator end
abstract type ClusteringOptimisationEstimator <: OptimisationEstimator end
abstract type ClusteringWeightFinaliser <: AbstractAlgorithm end
Base.length(::BaseOptimisationEstimator) = 1
function Base.iterate(::BaseOptimisationEstimator, state = 1)
    return state > 1 ? nothing : (:BaseOptimisationEstimator, state + 1)
end
struct HeuristicClusteringWeightFiniliser{T1 <: Integer} <: ClusteringWeightFinaliser
    iter::T1
end
function HeuristicClusteringWeightFiniliser(; iter::Integer = 100)
    @smart_assert(iter > 0)
    return HeuristicClusteringWeightFiniliser{typeof(iter)}(iter)
end
abstract type JuMP_ClusteringWeightFiniliserFormulation <: AbstractAlgorithm end
struct RelativeErrorClusteringWeightFiniliser <: JuMP_ClusteringWeightFiniliserFormulation end
struct SquareRelativeErrorClusteringWeightFiniliser <:
       JuMP_ClusteringWeightFiniliserFormulation end
struct AbsoluteErrorClusteringWeightFiniliser <: JuMP_ClusteringWeightFiniliserFormulation end
struct SquareAbsoluteErrorClusteringWeightFiniliser <:
       JuMP_ClusteringWeightFiniliserFormulation end
struct JuMP_ClusteringWeightFiniliser{T1 <: Union{<:Solver, <:AbstractVector{<:Solver}},
                                      T2 <: Real, T3 <: Real,
                                      T4 <: JuMP_ClusteringWeightFiniliserFormulation} <:
       ClusteringWeightFinaliser
    slv::T1
    sc::T2
    so::T3
    alg::T4
end
function JuMP_ClusteringWeightFiniliser(; slv::Union{<:Solver, <:AbstractVector{<:Solver}},
                                        sc::Real = 1.0, so::Real = 1.0,
                                        alg::JuMP_ClusteringWeightFiniliserFormulation = RelativeErrorClusteringWeightFiniliser())
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    @smart_assert(sc > zero(sc))
    @smart_assert(so > zero(so))
    return JuMP_ClusteringWeightFiniliser{typeof(slv), typeof(sc), typeof(so), typeof(alg)}(slv,
                                                                                            sc,
                                                                                            so,
                                                                                            alg)
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::RelativeErrorClusteringWeightFiniliser,
                                              wi::AbstractVector)
    wi[iszero.(wi)] .= eps(eltype(wi))
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
    @variable(model, t)
    @constraint(model,
                [sc * t;
                 sc * (w ⊘ wi .- one(eltype(wi)))] in MOI.NormOneCone(length(w) + 1))
    @objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::SquareRelativeErrorClusteringWeightFiniliser,
                                              wi::AbstractVector)
    wi[iszero.(wi)] .= eps(eltype(wi))
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
    @variable(model, t)
    @constraint(model, [sc * t; sc * (w ⊘ wi .- one(eltype(wi)))] in SecondOrderCone())
    @objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::AbsoluteErrorClusteringWeightFiniliser,
                                              wi::AbstractVector)
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
    @variable(model, t)
    @constraint(model, [sc * t; sc * (w - wi)] in MOI.NormOneCone(length(w) + 1))
    @objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::SquareAbsoluteErrorClusteringWeightFiniliser,
                                              wi::AbstractVector)
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
    @variable(model, t)
    @constraint(model, [sc * t; sc * (w - wi)] in SecondOrderCone())
    @objective(model, Min, so * t)
    return nothing
end
function opt_weight_bounds(cwf::JuMP_ClusteringWeightFiniliser, wb::WeightBoundsResult,
                           wi::AbstractVector)
    lb = wb.lb
    ub = wb.ub
    if !(any(ub .< wi) || any(lb .> wi))
        return wi
    end
    model = JuMP.Model()
    @expression(model, sc, cwf.sc)
    @expression(model, so, cwf.so)
    @variable(model, w[1:length(wi)])
    @constraint(model, sc * (sum(w) - sum(wi)) == 0)
    if !isnothing(lb)
        @constraint(model, sc * (w - lb) >= 0)
    end
    if !isnothing(ub)
        @constraint(model, sc * (w - ub) <= 0)
    end
    set_clustering_weight_finaliser_alg!(model, cwf.alg, wi)
    return if optimise_JuMP_model!(model, cwf.slv).success
        value.(model[:w])
    else
        @warn("Version: $(cwf.alg)\nReverting to Heuristic type.")
        opt_weight_bounds(HeuristicClusteringWeightFiniliser(), wb, wi)
    end
end
function opt_weight_bounds(cwf::HeuristicClusteringWeightFiniliser, wb::WeightBoundsResult,
                           w::AbstractVector)
    lb = wb.lb
    ub = wb.ub
    if !(any(ub .< w) || any(lb .> w))
        return w
    end
    iter = cwf.iter
    s1 = sum(w)
    for _ in 1:iter
        if !(any(ub .< w) || any(lb .> w))
            break
        end
        old_w = copy(w)
        w = max.(min.(w, ub), lb)
        idx = w .< ub .&& w .> lb
        w_add = sum(max.(old_w - ub, zero(eltype(w))))
        w_sub = sum(min.(old_w - lb, zero(eltype(w))))
        delta = w_add + w_sub
        if !iszero(delta)
            w[idx] += delta * w[idx] / sum(w[idx])
        end
        w *= s1 / sum(w)
    end
    return w
end
function clustering_optimisation_result(cwf::ClusteringWeightFinaliser,
                                        wb::WeightBoundsResult, w::AbstractVector)
    w = opt_weight_bounds(cwf, wb, w)
    retcode = if !any(!isfinite, w)
        OptimisationSuccess()
    else
        OptimisationFailure(; res = "Failure to set bounds\n$cwf\n$wb.")
    end
    return retcode, w
end

export HeuristicClusteringWeightFiniliser, RelativeErrorClusteringWeightFiniliser,
       SquareRelativeErrorClusteringWeightFiniliser, AbsoluteErrorClusteringWeightFiniliser,
       SquareAbsoluteErrorClusteringWeightFiniliser, JuMP_ClusteringWeightFiniliser
