abstract type BaseClusteringOptimisationEstimator <: BaseOptimisationEstimator end
abstract type ClusteringOptimisationEstimator <: OptimisationEstimator end
abstract type JuMPWeightFinaliserFormulation <: AbstractAlgorithm end
struct RelativeErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
struct SquareRelativeErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
struct AbsoluteErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
struct SquareAbsoluteErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
abstract type WeightFinaliser <: AbstractAlgorithm end
struct IterativeWeightFinaliser{T1} <: WeightFinaliser
    iter::T1
    function IterativeWeightFinaliser(iter::Integer)
        @argcheck(iter > 0)
        return new{typeof(iter)}(iter)
    end
end
function IterativeWeightFinaliser(; iter::Integer = 100)
    return IterativeWeightFinaliser(iter)
end
struct JuMPWeightFinaliser{T1, T2, T3, T4} <: WeightFinaliser
    slv::T1
    sc::T2
    so::T3
    alg::T4
    function JuMPWeightFinaliser(slv::Slv_VecSlv, sc::Number, so::Number,
                                 alg::JuMPWeightFinaliserFormulation)
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv))
        end
        @argcheck(sc > zero(sc))
        @argcheck(so > zero(so))
        return new{typeof(slv), typeof(sc), typeof(so), typeof(alg)}(slv, sc, so, alg)
    end
end
function JuMPWeightFinaliser(; slv::Slv_VecSlv, sc::Number = 1.0, so::Number = 1.0,
                             alg::JuMPWeightFinaliserFormulation = RelativeErrorWeightFinaliser())
    return JuMPWeightFinaliser(slv, sc, so, alg)
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::RelativeErrorWeightFinaliser, wi::VecNum)
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
                                              ::SquareRelativeErrorWeightFinaliser,
                                              wi::VecNum)
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
                                              ::AbsoluteErrorWeightFinaliser, wi::VecNum)
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
    @variable(model, t)
    @constraint(model, [sc * t; sc * (w - wi)] in MOI.NormOneCone(length(w) + 1))
    @objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::SquareAbsoluteErrorWeightFinaliser,
                                              wi::VecNum)
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
    @variable(model, t)
    @constraint(model, [sc * t; sc * (w - wi)] in SecondOrderCone())
    @objective(model, Min, so * t)
    return nothing
end
function opt_weight_bounds(cwf::JuMPWeightFinaliser, wb::WeightBounds, wi::VecNum)
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
        opt_weight_bounds(IterativeWeightFinaliser(), wb, wi)
    end
end
function opt_weight_bounds(cwf::IterativeWeightFinaliser, wb::WeightBounds, w::VecNum)
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
function clustering_optimisation_result(cwf::WeightFinaliser, wb::WeightBounds, w::VecNum)
    w = opt_weight_bounds(cwf, wb, w)
    retcode = if !any(!isfinite, w)
        OptimisationSuccess()
    else
        OptimisationFailure(; res = "Failure to set bounds\n$cwf\n$wb.")
    end
    return retcode, w
end

export IterativeWeightFinaliser, RelativeErrorWeightFinaliser,
       SquareRelativeErrorWeightFinaliser, AbsoluteErrorWeightFinaliser,
       SquareAbsoluteErrorWeightFinaliser, JuMPWeightFinaliser
