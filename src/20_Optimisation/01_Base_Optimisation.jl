abstract type AbstractOptimisationEstimator <: AbstractEstimator end
const VecOptE = AbstractVector{<:AbstractOptimisationEstimator}
abstract type BaseOptimisationEstimator <: AbstractOptimisationEstimator end
abstract type OptimisationEstimator <: AbstractOptimisationEstimator end
abstract type NonFiniteAllocationOptimisationEstimator <: OptimisationEstimator end
abstract type OptimisationAlgorithm <: AbstractAlgorithm end
abstract type OptimisationResult <: AbstractResult end
abstract type NonFiniteAllocationOptimisationResult <: OptimisationResult end
const VecOpt = AbstractVector{<:NonFiniteAllocationOptimisationResult}
abstract type OptimisationReturnCode <: AbstractResult end
abstract type OptimisationModelResult <: AbstractResult end
const OptE_Opt = Union{<:NonFiniteAllocationOptimisationEstimator,
                       <:NonFiniteAllocationOptimisationResult}
const VecOptE_Opt = AbstractVector{<:OptE_Opt}
abstract type JuMPWeightFinaliserFormulation <: AbstractAlgorithm end
struct RelativeErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
struct SquaredRelativeErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
struct AbsoluteErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
struct SquaredAbsoluteErrorWeightFinaliser <: JuMPWeightFinaliserFormulation end
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
    JuMP.@variable(model, t)
    JuMP.@constraint(model,
                     [sc * t;
                      sc * (w ⊘ wi .- one(eltype(wi)))] in
                     JuMP.MOI.NormOneCone(length(w) + 1))
    JuMP.@objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::SquaredRelativeErrorWeightFinaliser,
                                              wi::VecNum)
    wi[iszero.(wi)] .= eps(eltype(wi))
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
    JuMP.@variable(model, t)
    JuMP.@constraint(model,
                     [sc * t; sc * (w ⊘ wi .- one(eltype(wi)))] in JuMP.SecondOrderCone())
    JuMP.@objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::AbsoluteErrorWeightFinaliser, wi::VecNum)
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [sc * t; sc * (w - wi)] in JuMP.MOI.NormOneCone(length(w) + 1))
    JuMP.@objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_alg!(model::JuMP.Model,
                                              ::SquaredAbsoluteErrorWeightFinaliser,
                                              wi::VecNum)
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
    JuMP.@variable(model, t)
    JuMP.@constraint(model, [sc * t; sc * (w - wi)] in JuMP.SecondOrderCone())
    JuMP.@objective(model, Min, so * t)
    return nothing
end
function opt_weight_bounds(wf::JuMPWeightFinaliser, wb::WeightBounds, wi::VecNum)
    lb = wb.lb
    ub = wb.ub
    if !(!isnothing(lb) && any(map((x, y) -> x > y, lb, wi)) ||
         !isnothing(ub) && any(map((x, y) -> x < y, ub, wi)))
        return wi
    end
    model = JuMP.Model()
    JuMP.@expression(model, sc, wf.sc)
    JuMP.@expression(model, so, wf.so)
    JuMP.@variable(model, w[1:length(wi)])
    JuMP.@constraint(model, sc * (sum(w) - sum(wi)) == 0)
    if !isnothing(lb)
        JuMP.@constraint(model, sc * (w ⊖ lb) >= 0)
    end
    if !isnothing(ub)
        JuMP.@constraint(model, sc * (w ⊖ ub) <= 0)
    end
    set_clustering_weight_finaliser_alg!(model, wf.alg, wi)
    return if optimise_JuMP_model!(model, wf.slv).success
        JuMP.value.(model[:w])
    else
        @warn("Version: $(wf.alg)\nReverting to Heuristic type.")
        opt_weight_bounds(IterativeWeightFinaliser(), wb, wi)
    end
end
function opt_weight_bounds(wf::IterativeWeightFinaliser, wb::WeightBounds, w::VecNum)
    lb = wb.lb
    ub = wb.ub
    if isnothing(lb)
        lb = typemin(eltype(w))
    end
    if isnothing(ub)
        ub = typemax(eltype(w))
    end
    if !(any(map((x, y) -> x > y, lb, w)) || any(map((x, y) -> x < y, ub, w)))
        return w
    end
    iter = wf.iter
    s1 = sum(w)
    for _ in 1:iter
        if !(any(map((x, y) -> x > y, lb, w)) || any(map((x, y) -> x < y, ub, w)))
            break
        end
        old_w = copy(w)
        w = max.(min.(w, ub), lb)
        idx = w .< ub .&& w .> lb
        w_add = sum(max.(old_w ⊖ ub, zero(eltype(w))))
        w_sub = sum(min.(old_w ⊖ lb, zero(eltype(w))))
        delta = w_add + w_sub
        if !iszero(delta)
            w[idx] += delta * w[idx] / sum(w[idx])
        end
        w *= s1 / sum(w)
    end
    return w
end
function finalise_weight_bounds(wf::WeightFinaliser, wb::WeightBounds, w::VecNum)
    w = opt_weight_bounds(wf, wb, w)
    retcode = if !any(!isfinite, w)
        OptimisationSuccess()
    else
        OptimisationFailure(; res = "Failure to set bounds\n$wf\n$wb.")
    end
    return retcode, w
end
struct OptimisationSuccess{T1} <: OptimisationReturnCode
    res::T1
end
function OptimisationSuccess(; res = nothing)
    return OptimisationSuccess(res)
end
struct OptimisationFailure{T1} <: OptimisationReturnCode
    res::T1
end
function OptimisationFailure(; res = nothing)
    return OptimisationFailure(res)
end
function opt_view(opt::AbstractOptimisationEstimator, args...)
    return opt
end
function opt_view(opt::VecOptE, args...)
    return [opt_view(opti, args...) for opti in opt]
end
function optimise(or::OptimisationResult, args...; kwargs...)
    return or
end
function _optimise end
function optimise(opt::OptimisationEstimator, args...; kwargs...)
    fb = Tuple{OptimisationEstimator, OptimisationResult}[]
    current_opt = opt
    res = nothing
    while true
        res = _optimise(current_opt, args...; kwargs...)
        if isa(res.retcode, OptimisationSuccess) || isnothing(current_opt.fb)
            break
        else
            push!(fb, (current_opt, res))
            current_opt = current_opt.fb
            @warn("Using fallback method. Please ignore previous optimisation failure warnings.")
        end
    end
    return isempty(fb) ? res : factory(res, fb)
end
function assert_internal_optimiser(::NonFiniteAllocationOptimisationResult)
    return nothing
end
function assert_external_optimiser(::NonFiniteAllocationOptimisationResult)
    return nothing
end

export optimise, OptimisationSuccess, OptimisationFailure, IterativeWeightFinaliser,
       RelativeErrorWeightFinaliser, SquaredRelativeErrorWeightFinaliser,
       AbsoluteErrorWeightFinaliser, SquaredAbsoluteErrorWeightFinaliser,
       JuMPWeightFinaliser
