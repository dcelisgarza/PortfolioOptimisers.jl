"""
    const NonFRCJuMPOpt = Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting}

Alias for JuMP optimisers that do not use factor risk contribution.

Matches [`MeanRisk`](@ref), [`NearOptimalCentering`](@ref), or [`RiskBudgeting`](@ref). Used for dispatch in risk constraint generation functions that apply to these optimiser types but not to factor risk contribution.

# Related

  - [`MeanRisk`](@ref)
  - [`NearOptimalCentering`](@ref)
  - [`RiskBudgeting`](@ref)
"""
const NonFRCJuMPOpt = Union{<:MeanRisk, <:NearOptimalCentering, <:RiskBudgeting}
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Collapse the `risk_vec` expression array stored in `model` into a single scalar
`risk` JuMP expression.

The `SumScalariser` method sums all entries into a linear or quadratic expression. The
`LogSumExpScalariser` method introduces auxiliary variables and exponential cone constraints
to encode a log-sum-exp scalarisation. The `MaxScalariser` method introduces a variable and
linear constraints to encode the maximum over all entries.

# Mathematical definition

```math
\\begin{align}
\\mathcal{R}_{\\mathrm{sum}} &= \\sum_k \\mathcal{R}_k\\,, \\\\
\\mathcal{R}_{\\mathrm{lse}} &= \\frac{1}{\\gamma}\\ln\\sum_k e^{\\gamma \\mathcal{R}_k}\\,, \\\\
\\mathcal{R}_{\\mathrm{max}} &= \\max_k \\mathcal{R}_k\\,.
\\end{align}
```

Where:

  - ``\\mathcal{R}_{\\mathrm{sum}}``: Sum scalarisation.
  - ``\\mathcal{R}_{\\mathrm{lse}}``: Log-sum-exp scalarisation.
  - ``\\mathcal{R}_{\\mathrm{max}}``: Maximum scalarisation.
  - ``\\mathcal{R}_k``: ``k``-th risk expression.
  - ``\\gamma``: Temperature parameter for log-sum-exp.

# Arguments

  - $(arg_dict[:model])
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
    sc = get_constraint_scale(model)
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

  - $(arg_dict[:model])
  - `r`: A [`RiskMeasure`](@ref) instance, or `rs` a vector of risk measures.
  - $(arg_dict[:opt_jumpe])
  - $(arg_dict[:pr])
  - $(arg_dict[:pl_opt])
  - $(arg_dict[:fees_opt])

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

The `Nothing` overload does nothing (no bound was requested). The `Front_NumVec` overload
records the expression and its frontier bound vector in `model[:risk_frontier]` for later
use in Pareto frontier solves. The `Number` overload adds the constraint
`sc * (r_expr - ub * k) <= 0` directly to the model. The fall-through method emits a
warning: a non-`nothing` bound with an optimiser outside [`NonFRCJuMPOpt`](@ref) is
ignored, which would otherwise happen silently.

# Arguments

  - $(arg_dict[:model])
  - `r_expr::JuMP.AbstractJuMPScalar`: The risk JuMP expression to bound.
  - `ub`: Upper bound; a scalar number or a frontier specification.
  - `key::Symbol`: Symbol used to name the constraint in the model.
  - `flag::Bool`: If true, sets upper bound; if false sets lower bound (default: `true`).

# Returns

  - `nothing`.

# Related

  - [`set_risk_bounds_and_expression!`](@ref)
  - [`set_risk_expression!`](@ref)
"""
function set_risk_upper_bound!(::JuMP.Model, ::JuMPOptimisationEstimator, r_expr, ::Nothing,
                               key, flag::Bool = true)
    return nothing
end
function set_risk_upper_bound!(::JuMP.Model, opt::JuMPOptimisationEstimator, r_expr, ub,
                               key, flag::Bool = true)
    return @warn("Risk upper bound `settings.ub = $ub` ($key) is not supported by `$(typeof(opt).name.name)` and would be silently ignored. Remove `ub` from the risk measure settings, or use an optimiser that supports risk upper bounds (`MeanRisk`, `NearOptimalCentering`, `RiskBudgeting`).")
end
#! Using parameters to set the upper bounds would make things more difficult from a user perspective. Keep an eye on this in case things change in the future. We could simplify solve_mean_risk! and solve_noc! for pareto frontiers, we can define ub as a parameter and update it for subsequent solves.
# Solver(; name = :clarabel2,
#        solver = () -> ParametricOptInterface.Optimizer(JuMP.MOI.instantiate(Clarabel.Optimizer;
#                                                                        with_cache_type = Float64)),
#        check_sol = (; allow_local = true, allow_almost = true),
#        settings = Dict("verbose" => false, "max_step_fraction" => 0.75))
# https://discourse.julialang.org/t/solver-attributes-and-set-optimizer-with-parametricoptinterface-jl-and-jump-jl/129935/8?u=dcelisgarza
function set_risk_upper_bound!(model::JuMP.Model, ::NonFRCJuMPOpt,
                               r_expr::JuMP.AbstractJuMPScalar, ub::Front_NumVec, key,
                               flag::Bool = true)
    bound_key = Symbol(key, :_ub)
    bound_var_key = Symbol(key, :_ub_var)
    if !haskey(model, :risk_frontier)
        risk_frontier = JuMP.@expression(model, risk_frontier,
                                         Pair{Tuple{Symbol, Symbol},
                                              Tuple{<:JuMP.AbstractJuMPScalar,
                                                    <:Front_NumVec, Bool}}[(bound_var_key, bound_key) => (r_expr,
                                                                                                          ub,
                                                                                                          flag)])
    else
        risk_frontier = model[:risk_frontier]
        push!(risk_frontier, (bound_var_key, bound_key) => (r_expr, ub, flag))
    end
    return nothing
end
function set_risk_upper_bound!(model::JuMP.Model, ::NonFRCJuMPOpt,
                               r_expr::JuMP.AbstractJuMPScalar, ub::Number, key,
                               flag::Bool = true)
    k = get_k(model)
    sc = get_constraint_scale(model)
    bound_key = Symbol(key, :_ub)
    d = ifelse(flag, 1, -1)
    model[bound_key] = JuMP.@constraint(model, d * sc * (r_expr - ub * k) <= 0)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Push a scaled risk expression onto the `risk_vec` array in `model`.

If `rke` is `false` the function does nothing. Otherwise it initialises `risk_vec` if needed
and appends `scale * r_expr`.

# Arguments

  - $(arg_dict[:model])
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

  - $(arg_dict[:model])
  - $(arg_dict[:opt_rjumpe])
  - `r_expr::JuMP.AbstractJuMPScalar`: Risk JuMP expression.
  - `settings::RiskMeasureSettings`: Settings carrying upper bound, scale, and `rke` flag.
  - $(arg_dict[:key_sym])
  - `flag::Bool`: If true, sets upper bound; if false sets lower bound (default: `true`).

# Returns

  - `nothing`.

# Related

  - [`set_risk_upper_bound!`](@ref)
  - [`set_risk_expression!`](@ref)
"""
function set_risk_bounds_and_expression!(model::JuMP.Model,
                                         opt::RiskJuMPOptimisationEstimator,
                                         r_expr::JuMP.AbstractJuMPScalar,
                                         settings::RiskMeasureSettings, key,
                                         flag::Bool = true)
    set_risk_upper_bound!(model, opt, r_expr, settings.ub, key, flag)
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

# Mathematical definition

Drawdown recurrence:

```math
\\begin{align}
dd_0 &= 0\\,, \\\\
dd_t &\\geq 0\\,, \\\\
dd_t &\\geq dd_{t-1} - \\hat{r}_t
\\quad \\Leftrightarrow \\quad dd_t &= \\max_{s \\leq t} V_s - V_t\\,.
\\end{align}
```

Where:

  - ``dd_t``: Portfolio drawdown at time ``t``.
  - ``\\hat{r}_t``: Portfolio return at time ``t``.
  - ``V_t``: Cumulative portfolio wealth at time ``t``.

where ``\\hat{r}_t = \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}`` and ``V_t = k + \\sum_{s=1}^t \\hat{r}_s``.

# Arguments

  - $(arg_dict[:model])
  - `X::MatNum`: Asset returns matrix (`T × N`).

# Returns

  - `dd`: JuMP variable array of length `T + 1` tracking portfolio drawdowns.

# Related

  - [`set_risk_constraints!`](@ref)
"""
function set_drawdown_constraints!(model::JuMP.Model, X::MatNum;
                                   prefix::Symbol = Symbol(""))
    if haskey(model, Symbol(prefix, :dd))
        return model[Symbol(prefix, :dd)]
    end
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    T = length(net_X)
    dd = preg!(model, prefix, :dd, JuMP.@variable(model, [1:(T + 1)]))
    preg!(model, prefix, :cdd_start, JuMP.@constraint(model, sc * dd[1] == 0))
    preg!(model, prefix, :cdd_geq_0, JuMP.@constraint(model, sc * view(dd, 2:(T + 1)) >= 0))
    preg!(model, prefix, :cdd,
          JuMP.@constraint(model, sc * (net_X + view(dd, 2:(T + 1)) - view(dd, 1:T)) >= 0))
    return dd
end
