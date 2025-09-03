abstract type BaseClusteringOptimisationEstimator <: BaseOptimisationEstimator end
abstract type ClusteringOptimisationEstimator <: OptimisationEstimator end
abstract type WeightFinaliser <: AbstractAlgorithm end
struct IterativeWeightFiniliser{T1} <: WeightFinaliser
    iter::T1
end
function IterativeWeightFiniliser(; iter::Integer = 100)
    @argcheck(iter > 0)
    return IterativeWeightFiniliser(iter)
end
abstract type JuMPWeightFiniliserFormulation <: AbstractAlgorithm end
struct RelativeErrorWeightFiniliser <: JuMPWeightFiniliserFormulation end
struct SquareRelativeErrorWeightFiniliser <: JuMPWeightFiniliserFormulation end
struct AbsoluteErrorWeightFiniliser <: JuMPWeightFiniliserFormulation end
struct SquareAbsoluteErrorWeightFiniliser <: JuMPWeightFiniliserFormulation end
struct JuMPWeightFiniliser{T1, T2, T3, T4} <: WeightFinaliser
    slv::T1
    sc::T2
    so::T3
    alg::T4
end
function JuMPWeightFiniliser(; slv::Union{<:Solver, <:AbstractVector{<:Solver}},
                             sc::Real = 1.0, so::Real = 1.0,
                             alg::JuMPWeightFiniliserFormulation = RelativeErrorWeightFiniliser())
    if isa(slv, AbstractVector)
        @argcheck(!isempty(slv))
    end
    @argcheck(sc > zero(sc))
    @argcheck(so > zero(so))
    return JuMPWeightFiniliser(slv, sc, so, alg)
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::RelativeErrorWeightFiniliser,
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
                                              ::SquareRelativeErrorWeightFiniliser,
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
                                              ::AbsoluteErrorWeightFiniliser,
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
                                              ::SquareAbsoluteErrorWeightFiniliser,
                                              wi::AbstractVector)
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
    @variable(model, t)
    @constraint(model, [sc * t; sc * (w - wi)] in SecondOrderCone())
    @objective(model, Min, so * t)
    return nothing
end
function opt_weight_bounds(cwf::JuMPWeightFiniliser, wb::WeightBounds, wi::AbstractVector)
    lb = wb.lb
    ub = wb.ub
    if !(any(map((x, y) -> x < y, ub, wi)) || any(map((x, y) -> x > y, lb, wi)))
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
        opt_weight_bounds(IterativeWeightFiniliser(), wb, wi)
    end
end
function opt_weight_bounds(cwf::IterativeWeightFiniliser, wb::WeightBounds,
                           w::AbstractVector)
    lb = wb.lb
    ub = wb.ub
    if !(any(map((x, y) -> x < y, ub, w)) || any(map((x, y) -> x > y, lb, w)))
        return w
    end
    iter = cwf.iter
    s1 = sum(w)
    for _ in 1:iter
        if !(any(map((x, y) -> x < y, ub, w)) || any(map((x, y) -> x > y, lb, w)))
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
function clustering_optimisation_result(cwf::WeightFinaliser, wb::WeightBounds,
                                        w::AbstractVector)
    w = opt_weight_bounds(cwf, wb, w)
    retcode = if !any(!isfinite, w)
        OptimisationSuccess()
    else
        OptimisationFailure(; res = "Failure to set bounds\n$cwf\n$wb.")
    end
    return retcode, w
end

export IterativeWeightFiniliser, RelativeErrorWeightFiniliser,
       SquareRelativeErrorWeightFiniliser, AbsoluteErrorWeightFiniliser,
       SquareAbsoluteErrorWeightFiniliser, JuMPWeightFiniliser
