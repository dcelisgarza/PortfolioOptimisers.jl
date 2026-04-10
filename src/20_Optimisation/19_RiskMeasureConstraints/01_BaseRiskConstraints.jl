const NonFRCJuMPOpt = Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Collapse the `risk_vec` expression array stored in `model` into a single scalar
`risk` JuMP expression.

The `SumScalariser` method sums all entries into a linear or quadratic expression. The
`LogSumExpScalariser` method introduces auxiliary variables and exponential cone constraints
to encode a log-sum-exp scalarisation. The `MaxScalariser` method introduces a variable and
linear constraints to encode the maximum over all entries.

# Arguments

  - `model::JuMP.Model`: The JuMP optimisation model.
  - `sca`: Scalariser instance (one of [`SumScalariser`](@ref), [`LogSumExpScalariser`](@ref),
    or [`MaxScalariser`](@ref)).

# Returns

  - `nothing`.

# Related

  - [`SumScalariser`](@ref)
  - [`LogSumExpScalariser`](@ref)
  - [`MaxScalariser`](@ref)
"""
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Dispatch to index-aware `set_risk_constraints!` for a single risk measure or iterate over a
vector of risk measures.

The single-measure overload calls `set_risk_constraints!(model, 1, r, ...)`. The vector
overload calls `set_risk_constraints!(model, i, rs[i], ...)` for each element.

# Arguments

  - `model::JuMP.Model`: The JuMP optimisation model.
  - `r`: A [`RiskMeasure`](@ref) instance, or `rs` a vector of risk measures.
  - `opt`: Optimisation estimator.
  - `pr::AbstractPriorResult`: Prior result.
  - `pl`: Optional phylogeny constraints.
  - `fees`: Optional fees structure.

# Returns

  - `nothing`.

# Related

  - [`RiskMeasure`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, r::RiskMeasure,
                               opt::JuMPOptimisationEstimator, pr::AbstractPriorResult,
                               pl::Option{<:PlC_VecPlC}, fees::Option{<:Fees}, args...;
                               kwargs...)
    set_risk_constraints!(model, 1, r, opt, pr, pl, fees, args...; kwargs...)
    return nothing
end
function set_risk_constraints!(model::JuMP.Model, rs::VecRM, opt::JuMPOptimisationEstimator,
                               pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC},
                               fees::Option{<:Fees}, args...; kwargs...)
    for (i, r) in enumerate(rs)
        set_risk_constraints!(model, i, r, opt, pr, pl, fees, args...; kwargs...)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add an upper-bound constraint on a risk expression to `model`.

The fall-through method (`args...`) does nothing. The `Front_NumVec` overload records the
expression and its frontier bound vector in `model[:risk_frontier]` for later use in Pareto
frontier solves. The `Number` overload adds the constraint `sc * (r_expr - ub * k) <= 0`
directly to the model.

# Arguments

  - `model::JuMP.Model`: The JuMP optimisation model.
  - `r_expr::JuMP.AbstractJuMPScalar`: The risk JuMP expression to bound.
  - `ub`: Upper bound; a scalar number or a frontier specification.
  - `key::Symbol`: Symbol used to name the constraint in the model.

# Returns

  - `nothing`.

# Related

  - [`set_risk_bounds_and_expression!`](@ref)
  - [`set_risk_expression!`](@ref)
"""
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
    bound_var_key = Symbol(key, :_ub_var)
    if !haskey(model, :risk_frontier)
        risk_frontier = JuMP.@expression(model, risk_frontier,
                                         Pair{Tuple{Symbol, Symbol},
                                              Tuple{<:JuMP.AbstractJuMPScalar,
                                                    <:Front_NumVec}}[(bound_var_key, bound_key) => (r_expr,
                                                                                                    ub)])
    else
        risk_frontier = model[:risk_frontier]
        push!(risk_frontier, (bound_var_key, bound_key) => (r_expr, ub))
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Push a scaled risk expression onto the `risk_vec` array in `model`.

If `rke` is `false` the function does nothing. Otherwise it initialises `risk_vec` if needed
and appends `scale * r_expr`.

# Arguments

  - `model::JuMP.Model`: The JuMP optimisation model.
  - `r_expr::JuMP.AbstractJuMPScalar`: The risk JuMP expression to add.
  - `scale::Number`: Scaling factor applied to the expression.
  - `rke::Bool`: When `false` this method is a no-op.

# Returns

  - `nothing`.

# Related

  - [`set_risk_upper_bound!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Apply an upper-bound constraint and register the risk expression for the objective.

Calls [`set_risk_upper_bound!`](@ref) with `settings.ub` and [`set_risk_expression!`](@ref)
with `settings.scale` and `settings.rke`.

# Arguments

  - `model::JuMP.Model`: The JuMP optimisation model.
  - `opt::RiskJuMPOptimisationEstimator`: Optimisation estimator.
  - `r_expr::JuMP.AbstractJuMPScalar`: Risk JuMP expression.
  - `settings::RiskMeasureSettings`: Settings carrying upper bound, scale, and `rke` flag.
  - `key::Symbol`: Symbol used to name constraints in the model.

# Returns

  - `nothing`.

# Related

  - [`set_risk_upper_bound!`](@ref)
  - [`set_risk_expression!`](@ref)
"""
function set_risk_bounds_and_expression!(model::JuMP.Model,
                                         opt::RiskJuMPOptimisationEstimator,
                                         r_expr::JuMP.AbstractJuMPScalar,
                                         settings::RiskMeasureSettings, key)
    set_risk_upper_bound!(model, opt, r_expr, settings.ub, key)
    set_risk_expression!(model, r_expr, settings.scale, settings.rke)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add portfolio drawdown tracking variables and constraints to `model`.

Creates the `dd` variable array (length `T + 1`) together with three constraints:
`cdd_start` (initial drawdown is zero), `cdd_geq_0` (drawdowns are non-negative), and `cdd`
(drawdown recurrence relation). Returns the `dd` array; returns the existing one if already
present in `model`.

# Arguments

  - `model::JuMP.Model`: The JuMP optimisation model.
  - `X::MatNum`: Asset returns matrix (`T × N`).

# Returns

  - `dd`: JuMP variable array of length `T + 1` tracking portfolio drawdowns.

# Related

  - [`set_risk_constraints!`](@ref)
"""
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
