const NonFRCJuMPOpt = Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting}
function scalarise_risk_expression!(model::JuMP.Model, ::SumScalariser)
    if !haskey(model, :risk_vec)
        return nothing
    end
    risk_vec = model[:risk_vec]
    if any(x -> isa(x, JuMP.QuadExpr), risk_vec)
        JuMP.@expression(model, risk, zero(JuMP.QuadExpr))
    else
        JuMP.@expression(model, risk, zero(JuMP.AffExpr))
    end
    for risk_i in risk_vec
        JuMP.add_to_expression!(risk, risk_i)
    end
    return nothing
end
function scalarise_risk_expression!(model::JuMP.Model, sca::LogSumExpScalariser)
    if !haskey(model, :risk_vec)
        return nothing
    end
    risk_vec = model[:risk_vec]
    sc = model[:sc]
    N = length(risk_vec)
    gamma = sca.gamma
    JuMP.@variables(model, begin
                        risk
                        u_risk[1:N]
                    end)
    JuMP.@constraints(model,
                      begin
                          u_risk_lse, sc * (sum(u_risk) - 1) <= 0
                          risk_lse[i = 1:N],
                          [sc * gamma * (risk_vec[i] - risk), sc, sc * u_risk[i]] in
                          JuMP.MOI.ExponentialCone()
                      end)
    return nothing
end
function scalarise_risk_expression!(model::JuMP.Model, ::MaxScalariser)
    if !haskey(model, :risk_vec)
        return nothing
    end
    risk_vec = model[:risk_vec]
    JuMP.@variable(model, risk)
    JuMP.@constraint(model, risk_ms, risk .- risk_vec .>= 0)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, r::RiskMeasure,
                               opt::JuMPOptimisationEstimator, pr::AbstractPriorResult,
                               plg::Option{<:PhC_VecPhC}, fees::Option{<:Fees}, args...;
                               kwargs...)
    set_risk_constraints!(model, 1, r, opt, pr, plg, fees, args...; kwargs...)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, rs::VecRM, opt::JuMPOptimisationEstimator,
                               pr::AbstractPriorResult, plg::Option{<:PhC_VecPhC},
                               fees::Option{<:Fees}, args...; kwargs...)
    for (i, r) in enumerate(rs)
        set_risk_constraints!(model, i, r, opt, pr, plg, fees, args...; kwargs...)
    end
    return nothing
end
function set_risk_upper_bound!(args...)
    return nothing
end
#! Using parameters to set the upper bounds would make things more difficult from a user perspective. Keep an eye on this in case things change in the future. We could simplify solve_mean_risk! and solve_noc! for pareto frontiers, we can define ub as a parameter and update it for subsequent solves.
# Solver(; name = :clarabel2,
#        solver = () -> ParametricOptInterface.Optimizer(JuMP.MOI.instantiate(Clarabel.Optimizer;
#                                                                        with_cache_type = Float64)),
#        check_sol = (; allow_local = true, allow_almost = true),
#        settings = Dict("verbose" => false, "max_step_fraction" => 0.75))
# https://discourse.julialang.org/t/solver-attributes-and-set-optimizer-with-parametricoptinterface-jl-and-jump-jl/129935/8?u=dcelisgarza
function set_risk_upper_bound!(model::JuMP.Model, ::NonFRCJuMPOpt,
                               r_expr::JuMP.AbstractJuMPScalar, ub::Front_NumVec, key)
    bound_key = Symbol(key, :_ub)
    if !haskey(model, :risk_frontier)
        risk_frontier = JuMP.@expression(model, risk_frontier,
                                         Pair{Symbol,
                                              Tuple{<:JuMP.AbstractJuMPScalar,
                                                    <:Front_NumVec}}[bound_key => (r_expr,
                                                                                   ub)])
    else
        risk_frontier = model[:risk_frontier]
        push!(risk_frontier, bound_key => (r_expr, ub))
    end
    return nothing
end
function set_risk_upper_bound!(model::JuMP.Model, ::NonFRCJuMPOpt,
                               r_expr::JuMP.AbstractJuMPScalar, ub::Number, key)
    k = model[:k]
    sc = model[:sc]
    bound_key = Symbol(key, :_ub)
    model[bound_key] = JuMP.@constraint(model, sc * (r_expr - ub * k) <= 0)
    return nothing
end
function set_risk_expression!(model::JuMP.Model, r_expr::JuMP.AbstractJuMPScalar,
                              scale::Number, rke::Bool)
    if !rke
        return nothing
    end
    if !haskey(model, :risk_vec)
        JuMP.@expression(model, risk_vec, Union{JuMP.AffExpr, JuMP.QuadExpr}[])
    end
    risk_vec = model[:risk_vec]
    push!(risk_vec, scale * r_expr)
    return nothing
end
function set_risk_bounds_and_expression!(model::JuMP.Model,
                                         opt::RiskJuMPOptimisationEstimator,
                                         r_expr::JuMP.AbstractJuMPScalar,
                                         settings::RiskMeasureSettings, key)
    set_risk_upper_bound!(model, opt, r_expr, settings.ub, key)
    set_risk_expression!(model, r_expr, settings.scale, settings.rke)
    return nothing
end
function set_drawdown_constraints!(model::JuMP.Model, X::MatNum)
    if haskey(model, :dd)
        return model[:dd]
    end
    sc = model[:sc]
    net_X = set_net_portfolio_returns!(model, X)
    T = length(net_X)
    JuMP.@variable(model, dd[1:(T + 1)])
    JuMP.@constraints(model,
                      begin
                          cdd_start, sc * dd[1] == 0
                          cdd_geq_0, sc * view(dd, 2:(T + 1)) >= 0
                          cdd, sc * (net_X + view(dd, 2:(T + 1)) - view(dd, 1:T)) >= 0
                      end)
    return dd
end
