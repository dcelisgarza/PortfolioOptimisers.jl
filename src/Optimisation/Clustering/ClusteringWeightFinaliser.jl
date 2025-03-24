abstract type ClusteringWeightFinaliser end
struct HeuristicClusteringWeightFiniliser{T1 <: Integer} <: ClusteringWeightFinaliser
    max_iter::T1
end
function HeuristicClusteringWeightFiniliser(; max_iter::Integer = 100)
    return HeuristicClusteringWeightFiniliser{typeof(max_iter)}(max_iter)
end

abstract type JuMP_ClusteringWeightFiniliserVersion end
struct RelativeErrorClusteringWeightFiniliser <: JuMP_ClusteringWeightFiniliserVersion end
struct SquareRelativeErrorClusteringWeightFiniliser <: JuMP_ClusteringWeightFiniliserVersion end
struct AbsoluteErrorClusteringWeightFiniliser <: JuMP_ClusteringWeightFiniliserVersion end
struct SquareAbsoluteErrorClusteringWeightFiniliser <: JuMP_ClusteringWeightFiniliserVersion end
struct JuMP_ClusteringWeightFiniliser{T1 <: JuMP_ClusteringWeightFiniliserVersion} <:
       ClusteringWeightFinaliser
    version::T1
end
function JuMP_ClusteringWeightFiniliser(;
                                        version::JuMP_ClusteringWeightFiniliserVersion = RelativeErrorClusteringWeightFiniliser())
    return JuMP_ClusteringWeightFiniliser{typeof(version)}(version)
end
function finaliser_constraint_obj(::RelativeErrorClusteringWeightFiniliser, model, w,
                                  scale_constr, scale_obj)
    N = length(w)
    w = model[:w]
    @variable(model, t)
    w[iszero.(w)] .= eps(eltype(w))
    @constraint(model,
                [scale_constr * t; scale_constr * (w ./ w .- one(eltype(w)))] in
                MOI.NormOneCone(N + 1))
    @objective(model, Min, scale_obj * t)
    return nothing
end
function finaliser_constraint_obj(::SquareRelativeErrorClusteringWeightFiniliser, model, w,
                                  scale_constr, scale_obj)
    w = model[:w]
    @variable(model, t)
    w[iszero.(w)] .= eps(eltype(w))
    @constraint(model,
                [scale_constr * t; scale_constr * (w ./ w .- one(eltype(w)))] in
                SecondOrderCone())
    @objective(model, Min, scale_obj * t)
    return nothing
end
function finaliser_constraint_obj(::AbsoluteErrorClusteringWeightFiniliser, model, w,
                                  scale_constr, scale_obj)
    N = length(w)
    w = model[:w]
    @variable(model, t)
    @constraint(model,
                [scale_constr * t; scale_constr * (w .- w)] in MOI.NormOneCone(N + 1))
    @objective(model, Min, scale_obj * t)
    return nothing
end
function finaliser_constraint_obj(::SquareAbsoluteErrorClusteringWeightFiniliser, model, w,
                                  scale_constr, scale_obj)
    w = model[:w]
    @variable(model, t)
    @constraint(model, [scale_constr * t; scale_constr * (w .- w)] in SecondOrderCone())
    @objective(model, Min, scale_obj * t)
    return nothing
end
function opt_weight_bounds(port, w_min, w_max, w, finaliser::JuMP_ClusteringWeightFiniliser)
    if !(any(w_max .< w) || any(w_min .> w))
        return w
    end
    scale_constr = port.scale_constr
    scale_obj = port.scale_obj
    solvers = port.solvers
    version = finaliser.version

    model = JuMP.Model()

    N = length(w)
    budget = sum(w)

    @variable(model, w[1:N])
    @constraint(model, sum(w) == budget)
    if all(w .>= 0)
        @constraints(model, begin
                         scale_constr * w .>= 0
                         scale_constr * w .>= scale_constr * w_min
                         scale_constr * w .<= scale_constr * w_max
                     end)
    else
        short_budget = sum(w[w .< zero(eltype(w))])

        @variables(model, begin
                       long_w[1:N] .>= 0
                       short_w[1:N] .<= 0
                   end)

        @constraints(model,
                     begin
                         scale_constr * long_w .<= scale_constr * w_max
                         scale_constr * short_w .>= scale_constr * w_min
                         scale_constr * w .<= scale_constr * long_w
                         scale_constr * w .>= scale_constr * short_w
                         scale_constr * sum(short_w) == scale_constr * short_budget
                         scale_constr * sum(long_w) ==
                         scale_constr * (budget - short_budget)
                     end)
    end
    finaliser_constraint_obj(version, model, w, scale_constr, scale_obj)

    success, solvers_tried = optimise_JuMP_model(model, solvers)

    return if success
        value.(model[:w])
    else
        funcname = "$(fullname(PortfolioOptimiser)[1]).$(nameof(PortfolioOptimiser.opt_weight_bounds))"
        @warn("$funcname: model could not be optimised satisfactorily.\nVersion: $version\nSolvers: $solvers_tried.\nReverting to Heuristic type.")
        opt_weight_bounds(nothing, w_min, w_max, w, HeuristicClusteringWeightFiniliser())
    end
end
function opt_weight_bounds(::Any, w_min, w_max, w,
                           finaliser::HeuristicClusteringWeightFiniliser)
    if !(any(w_max .< w) || any(w_min .> w))
        return w
    end

    max_iter = finaliser.max_iter

    s1 = sum(w)
    for _ ∈ 1:max_iter
        if !(any(w_max .< w) || any(w_min .> w))
            break
        end

        old_w = copy(w)
        w = max.(min.(w, w_max), w_min)

        idx = w .< w_max .&& w .> w_min
        w_add = sum(max.(old_w - w_max, 0.0))
        w_sub = sum(min.(old_w - w_min, 0.0))
        delta = w_add + w_sub

        if delta != 0
            w[idx] += delta * w[idx] / sum(w[idx])
        end
        w .*= s1 / sum(w)
    end

    return w
end
function finalise_hierarchical_weights(port, w, w_min, w_max, finaliser)
    key = type.key == :auto ? String(type) : String(type.key)
    w = opt_weight_bounds(port, w_min, w_max, w, finaliser)
    w ./= sum(w)
    if any(.!isfinite.(w)) || all(iszero.(w))
        fees = calc_fees(w, port.fees)
        if !isempty(fees)
            port.fail[Symbol("$(key)_fees")] = fees
        end
        port.fail[Symbol(key)] = DataFrame(; tickers = port.assets, w = w)
        port.optimal[Symbol(key)] = DataFrame()
    else
        fees = calc_fees(w, port.fees)
        if !isempty(fees)
            port.optimal[Symbol("$(key)_fees")] = fees
        end
        port.optimal[Symbol(key)] = DataFrame(; tickers = port.assets, w = w)
    end
    return port.optimal[Symbol(key)]
end