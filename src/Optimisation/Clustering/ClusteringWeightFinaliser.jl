abstract type ClusteringWeightFinaliser end
struct HeuristicClusteringWeightFiniliser{T1 <: Integer} <: ClusteringWeightFinaliser
    iter::T1
end
function HeuristicClusteringWeightFiniliser(; iter::Integer = 100)
    @smart_assert(iter > 0)
    return HeuristicClusteringWeightFiniliser{typeof(iter)}(iter)
end
abstract type JuMP_ClusteringWeightFiniliserVersion end
struct RelativeErrorClusteringWeightFiniliser <: JuMP_ClusteringWeightFiniliserVersion end
struct SquareRelativeErrorClusteringWeightFiniliser <: JuMP_ClusteringWeightFiniliserVersion end
struct AbsoluteErrorClusteringWeightFiniliser <: JuMP_ClusteringWeightFiniliserVersion end
struct SquareAbsoluteErrorClusteringWeightFiniliser <: JuMP_ClusteringWeightFiniliserVersion end
struct JuMP_ClusteringWeightFiniliser{T1 <: Real, T2 <: Real,
                                      T3 <: JuMP_ClusteringWeightFiniliserVersion,
                                      T4 <: Union{<:Solver, AbstractVector{<:Solver}}} <:
       ClusteringWeightFinaliser
    sc::T1
    so::T2
    v::T3
    slv::T4
end
function JuMP_ClusteringWeightFiniliser(; sc::Real = 1.0, so::Real = 1.0,
                                        v::JuMP_ClusteringWeightFiniliserVersion = RelativeErrorClusteringWeightFiniliser(),
                                        slv::Union{Solver, AbstractVector{Solver}})
    return JuMP_ClusteringWeightFiniliser{typeof(sc), typeof(so), typeof(v), typeof(slv)}(sc,
                                                                                          so,
                                                                                          v,
                                                                                          slv)
end
function set_clustering_weight_finaliser_version!(model::JuMP.Model,
                                                  ::RelativeErrorClusteringWeightFiniliser,
                                                  wi::AbstractVector)
    wi[iszero.(wi)] .= eps(eltype(wi))
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
    @variable(model, t)
    @constraint(model,
                [sc * t; sc * (w ./ wi .- one(eltype(wi)))] in
                MOI.NormOneCone(length(w) + 1))
    @objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_version!(model::JuMP.Model,
                                                  ::SquareRelativeErrorClusteringWeightFiniliser,
                                                  wi::AbstractVector)
    wi[iszero.(wi)] .= eps(eltype(wi))
    w = model[:w]
    sc = model[:sc]
    so = model[:so]
    @variable(model, t)
    @constraint(model, [sc * t; sc * (w ./ wi .- one(eltype(wi)))] in SecondOrderCone())
    @objective(model, Min, so * t)
    return nothing
end
function set_clustering_weight_finaliser_version!(model::JuMP.Model,
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
function set_clustering_weight_finaliser_version!(model::JuMP.Model,
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
function opt_weight_bounds(cwf::JuMP_ClusteringWeightFiniliser, wb::WeightBounds,
                           wi::AbstractVector)
    lb = wb.lb
    ub = wb.ub
    if any(ub .< wi) ⊼ any(lb .> wi)
        return wi
    end
    model = JuMP.Model()
    @expression(model, sc, cwf.sc)
    @expression(model, so, cwf.so)
    @variable(model, w[1:length(wi)])
    @constraint(model, sum(w) == sum(wi))
    if !isnothing(lb)
        @constraint(model, sc * w >= sc * lb)
    end
    if !isnothing(ub)
        @constraint(model, sc * w <= sc * ub)
    end
    set_clustering_weight_finaliser_version!(model, cwf.v, wi)
    (; trials, success) = optimise_JuMP_model!(model, cwf.slv)
    return if success
        value.(model[:w])
    else
        @warn("Model could not be optimised satisfactorily.\nVersion: $(cwf.v)\nSolvers: $trials.\nReverting to Heuristic type.")
        opt_weight_bounds(HeuristicClusteringWeightFiniliser(), wb, wi)
    end
end
function opt_weight_bounds(cwf::HeuristicClusteringWeightFiniliser, wb::WeightBounds,
                           w::AbstractVector)
    lb = wb.lb
    ub = wb.ub
    if any(ub .< w) ⊼ any(lb .> w)
        return w
    end
    iter = cwf.iter
    s1 = sum(w)
    for _ ∈ 1:iter
        if !(any(ub .< w) || any(lb .> w))
            break
        end
        old_w = copy(w)
        w = max.(min.(w, ub), lb)
        idx = w .< ub .&& w .> lb
        w_add = sum(max.(old_w - ub, zero(eltype(w))))
        w_sub = sum(min.(old_w - lb, zero(eltype(w))))
        delta = w_add + w_sub
        if delta != 0
            w[idx] += delta * w[idx] / sum(w[idx])
        end
        w *= s1 / sum(w)
    end
    return w
end
function finalise_hierarchical_weights(cwf::ClusteringWeightFinaliser, wb::WeightBounds,
                                       w::AbstractVector)
    w = opt_weight_bounds(cwf, wb, w)
    return w / sum(w)
end
