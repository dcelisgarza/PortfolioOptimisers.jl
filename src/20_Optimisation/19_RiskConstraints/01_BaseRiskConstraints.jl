function set_risk_upper_bound!(args...)
    return nothing
end
#! Using parameters to set the upper bounds would make things more difficult from a user perspective. Keep an eye on this in case things change in the future. We could simplify solve_mean_risk! and solve_noc! for pareto frontiers, we can define ub as a parameter and update it for subsequent solves.
# Solver(; name = :clarabel2,
#        solver = () -> ParametricOptInterface.Optimizer(MOI.instantiate(Clarabel.Optimizer;
#                                                                        with_cache_type = Float64)),
#        check_sol = (; allow_local = true, allow_almost = true),
#        settings = Dict("verbose" => false, "max_step_fraction" => 0.75))
# https://discourse.julialang.org/t/solver-attributes-and-set-optimizer-with-parametricoptinterface-jl-and-jump-jl/129935/8?u=dcelisgarza
function set_risk_upper_bound!(model::JuMP.Model,
                               ::Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting},
                               r_expr::AbstractJuMPScalar,
                               ub::Union{<:AbstractVector, <:Frontier}, key)
    bound_key = Symbol(key, :_ub)
    if !haskey(model, :risk_frontier)
        risk_frontier = @expression(model, risk_frontier,
                                    Pair{Symbol,
                                         Tuple{<:AbstractJuMPScalar,
                                               <:Union{<:AbstractVector, <:Frontier}}}[bound_key => (r_expr,
                                                                                                     ub)])
    else
        risk_frontier = model[:risk_frontier]
        push!(risk_frontier, bound_key => (r_expr, ub))
    end
    return nothing
end
function set_risk_upper_bound!(model::JuMP.Model,
                               ::Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting},
                               r_expr::AbstractJuMPScalar, ub::Real, key)
    k = model[:k]
    sc = model[:sc]
    bound_key = Symbol(key, :_ub)
    model[bound_key] = @constraint(model, sc * (r_expr - ub * k) <= 0)
    return nothing
end
function set_risk_expression!(model::JuMP.Model, r_expr::AbstractJuMPScalar, scale::Real,
                              rke::Bool)
    if !rke
        return nothing
    end
    if !haskey(model, :risk_vec)
        @expression(model, risk_vec, Union{AffExpr, QuadExpr}[])
    end
    risk_vec = model[:risk_vec]
    push!(risk_vec, scale * r_expr)
    return nothing
end
function set_risk_bounds_and_expression!(model::JuMP.Model,
                                         opt::Union{<:MeanRisk, <:NearOptimalCentering,
                                                    <:RiskBudgeting},
                                         r_expr::AbstractJuMPScalar,
                                         settings::RiskMeasureSettings, key)
    set_risk_upper_bound!(model, opt, r_expr, settings.ub, key)
    set_risk_expression!(model, r_expr, settings.scale, settings.rke)
    return nothing
end
function set_drawdown_constraints!(model::JuMP.Model, X::AbstractMatrix)
    if haskey(model, :dd)
        return model[:dd]
    end
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, X)
    T = length(net_X)
    @variable(model, dd[1:(T + 1)])
    @constraints(model, begin
                     cdd_start, sc * dd[1] == 0
                     cdd_geq_0, sc * view(dd, 2:(T + 1)) >= 0
                     cdd, sc * (net_X + view(dd, 2:(T + 1)) - view(dd, 1:T)) >= 0
                 end)
    return dd
end
