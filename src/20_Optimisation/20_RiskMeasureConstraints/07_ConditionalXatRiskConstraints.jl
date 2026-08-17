"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add conditional risk constraints (CVaR, DRCVaR, CDaR, and their range/DR variants) to `model`.

Each overload introduces auxiliary non-negative exceedance variables and constructs the
appropriate weighted-sum CVaR (or CDaR) expression. The distributionally robust variants add
infinity-norm cone constraints to handle distributional ambiguity over an `r.r`-radius ball.
Range variants compute the difference between lower-tail and upper-tail conditional
risk expressions.

# Mathematical definition

Rockafellar-Uryasev CVaR linearisation:

```math
\\begin{align}
\\mathrm{CVaR}_\\alpha(\\boldsymbol{w}) &= \\mathrm{VaR} + \\frac{1}{\\alpha T} \\sum_{t=1}^T z_t\\,, \\\\
z_t &\\geq -\\hat{r}_t - \\mathrm{VaR},\\quad z_t \\geq 0\\,.
\\end{align}
```

Where:

  - ``\\mathrm{CVaR}_\\alpha(\\boldsymbol{w})``: Conditional Value-at-Risk.
  - ``\\mathrm{VaR}``: Value-at-Risk auxiliary variable.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - ``z_t \\geq 0``: Auxiliary excess loss variables.
  - ``\\hat{r}_t = \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}``: Portfolio return at time ``t``.

where ``\\hat{r}_t = \\boldsymbol{x}_t^\\intercal \\boldsymbol{w}`` is the net portfolio return at time ``t``.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - $(arg_dict[:r_risk])
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`set_drawdown_constraints!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::ConditionalValueatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; loss::Bool = true, prefix::Symbol = Symbol(""),
                               kwargs...)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, pr.X; prefix = prefix)
    if !loss
        net_X = -net_X
    end
    T = length(net_X)
    var, z_cvar = JuMP.@variables(model, begin
                                      ()
                                      [1:T], (lower_bound = 0)
                                  end)
    state_set!(model, prefix, :var_, i, var)
    state_set!(model, prefix, :z_cvar_, i, z_cvar)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    cvar_risk = if isnothing(wi)
        iat = inv(r.alpha * T)
        JuMP.@expression(model, var + sum(z_cvar) * iat)
    else
        iat = inv(r.alpha * sum(wi))
        JuMP.@expression(model, var + LinearAlgebra.dot(wi, z_cvar) * iat)
    end
    state_set!(model, prefix, :cvar_risk_, i, cvar_risk)
    state_set!(model, prefix, :ccvar_, i,
               JuMP.@constraint(model, sc * ((z_cvar + net_X) .+ var) >= 0))
    set_risk_bounds_and_expression!(model, opt, cvar_risk, r.settings, :cvar_risk_, i;
                                    prefix = prefix)
    return cvar_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `ConditionalValueatRiskRange` to `model`.

Delegates to [`set_range_risk_constraints!`](@ref), which builds the loss tail at `alpha` on
the net portfolio returns and the gain tail at `beta` on their negation, then sums the two
CVaR expressions.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::ConditionalValueatRiskRange`: The CVaR range risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `cvar_range_risk`: The combined `loss + gain` risk expression added to the model.

# Related

  - [`ConditionalValueatRiskRange`](@ref)
  - [`range_tails`](@ref)
  - [`set_range_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::ConditionalValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    return set_range_risk_constraints!(model, i, r, :cvar_range_risk_, opt, pr, args...;
                                       prefix = prefix, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `DistributionallyRobustConditionalValueatRisk` (DR-CVaR)
to `model`.

Adds an infinity-norm cone constraint over an `r.r`-radius Wasserstein ambiguity ball and
auxiliary exceedance variables to encode the distributionally robust CVaR.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::DistributionallyRobustConditionalValueatRisk`: The DR-CVaR risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`DistributionallyRobustConditionalValueatRisk`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::DistributionallyRobustConditionalValueatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; loss::Bool = true, prefix::Symbol = Symbol(""),
                               kwargs...)
    sc = get_constraint_scale(model)
    w = get_w(model, prefix)
    net_X = set_net_portfolio_returns!(model, pr.X; prefix = prefix)
    if !loss
        net_X = -net_X
        X = -pr.X
        prefix = nested_prefix(prefix, :gain_)
    else
        X = pr.X
    end
    Xap1 = set_asset_returns_plus_one!(model, X; prefix = prefix)
    T, N = size(X)

    alpha = r.alpha
    b1 = r.l
    radius = r.r

    a1 = -one(alpha)
    a2 = -one(alpha) - b1 * inv(alpha)
    b2 = b1 * (one(alpha) - inv(alpha))
    lb, tau, s, tu_drcvar, tv_drcvar, u, v = JuMP.@variables(model,
                                                             begin
                                                                 ()
                                                                 ()
                                                                 [1:T]
                                                                 [1:T]
                                                                 [1:T]
                                                                 [1:T, 1:N],
                                                                 (lower_bound = 0)
                                                                 [1:T, 1:N],
                                                                 (lower_bound = 0)
                                                             end)
    state_set!(model, prefix, :lb_drcvar_, i, lb)
    state_set!(model, prefix, :tau_drcvar_, i, tau)
    state_set!(model, prefix, :s_drcvar_, i, s)
    state_set!(model, prefix, :tu_drcvar_, i, tu_drcvar)
    state_set!(model, prefix, :tv_drcvar_, i, tv_drcvar)
    state_set!(model, prefix, :u_drcvar_, i, u)
    state_set!(model, prefix, :v_drcvar_, i, v)
    cu_drcvar, cv_drcvar, cu_drcvar_infnorm, cv_drcvar_infnorm, cu_drcvar_lb, cv_drcvar_lb = JuMP.@constraints(model,
                                                                                                               begin
                                                                                                                   sc *
                                                                                                                   (b1 *
                                                                                                                    tau .+
                                                                                                                    (a1 *
                                                                                                                     net_X +
                                                                                                                     vec(sum(u .*
                                                                                                                             Xap1;
                                                                                                                             dims = 2)) -
                                                                                                                     s)) <=
                                                                                                                   0
                                                                                                                   sc *
                                                                                                                   (b2 *
                                                                                                                    tau .+
                                                                                                                    (a2 *
                                                                                                                     net_X +
                                                                                                                     vec(sum(v .*
                                                                                                                             Xap1;
                                                                                                                             dims = 2)) -
                                                                                                                     s)) <=
                                                                                                                   0
                                                                                                                   [i = 1:T],
                                                                                                                   [sc *
                                                                                                                    tu_drcvar[i]
                                                                                                                    sc *
                                                                                                                    (-view(u,
                                                                                                                           i,
                                                                                                                           :) -
                                                                                                                     a1 *
                                                                                                                     w)] in
                                                                                                                   JuMP.MOI.NormInfinityCone(1 +
                                                                                                                                             N)
                                                                                                                   [i = 1:T],
                                                                                                                   [sc *
                                                                                                                    tv_drcvar[i]
                                                                                                                    sc *
                                                                                                                    (-view(v,
                                                                                                                           i,
                                                                                                                           :) -
                                                                                                                     a2 *
                                                                                                                     w)] in
                                                                                                                   JuMP.MOI.NormInfinityCone(1 +
                                                                                                                                             N)
                                                                                                                   sc *
                                                                                                                   (tu_drcvar .-
                                                                                                                    lb) <=
                                                                                                                   0
                                                                                                                   sc *
                                                                                                                   (tv_drcvar .-
                                                                                                                    lb) <=
                                                                                                                   0
                                                                                                               end)
    state_set!(model, prefix, :cu_drcvar_, i, cu_drcvar)
    state_set!(model, prefix, :cv_drcvar_, i, cv_drcvar)
    state_set!(model, prefix, :cu_drcvar_infnorm_, i, cu_drcvar_infnorm)
    state_set!(model, prefix, :cv_drcvar_infnorm_, i, cv_drcvar_infnorm)
    state_set!(model, prefix, :cu_drcvar_lb_, i, cu_drcvar_lb)
    state_set!(model, prefix, :cv_drcvar_lb_, i, cv_drcvar_lb)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    drcvar_risk = if isnothing(wi)
        JuMP.@expression(model, radius * lb + Statistics.mean(s))
    else
        JuMP.@expression(model, radius * lb + Statistics.mean(s, wi))
    end
    state_set!(model, prefix, :drcvar_risk_, i, drcvar_risk)
    set_risk_bounds_and_expression!(model, opt, drcvar_risk, r.settings, :drcvar_risk_, i;
                                    prefix = prefix)
    return drcvar_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `DistributionallyRobustConditionalValueatRiskRange`
(DR-CVaR range) to `model`.

Delegates to [`set_range_risk_constraints!`](@ref), which builds the loss tail from `alpha`,
`l_a` and `r_a` on the net portfolio returns, and the gain tail from `beta`, `l_b` and `r_b`
on their negation, then sums the two robust CVaR expressions. Each tail brings its own
Wasserstein ambiguity ball.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::DistributionallyRobustConditionalValueatRiskRange`: The DR-CVaR range risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `drcvar_risk_range`: The combined `loss + gain` risk expression added to the model.

# Related

  - [`DistributionallyRobustConditionalValueatRiskRange`](@ref)
  - [`range_tails`](@ref)
  - [`set_range_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::DistributionallyRobustConditionalValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    return set_range_risk_constraints!(model, i, r, :drcvar_risk_range_, opt, pr, args...;
                                       prefix = prefix, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `ConditionalDrawdownatRisk` (CDaR) to `model`.

Introduces a drawdown-at-risk variable and non-negative exceedance variables over the
drawdown series. The CDaR risk expression is the expected shortfall over drawdowns.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::ConditionalDrawdownatRisk`: The CDaR risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`ConditionalDrawdownatRisk`](@ref)
  - [`set_drawdown_constraints!`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::ConditionalDrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    dd = set_drawdown_constraints!(model, pr.X; prefix = prefix)
    T = length(dd) - 1
    iat = inv(r.alpha * T)
    dar, z_cdar = JuMP.@variables(model, begin
                                      ()
                                      [1:T], (lower_bound = 0)
                                  end)
    state_set!(model, prefix, :dar_, i, dar)
    state_set!(model, prefix, :z_cdar_, i, z_cdar)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    cdar_risk = if isnothing(wi)
        iat = inv(r.alpha * T)
        JuMP.@expression(model, dar + sum(z_cdar) * iat)
    else
        iat = inv(r.alpha * sum(wi))
        JuMP.@expression(model, dar + LinearAlgebra.dot(wi, z_cdar) * iat)
    end
    state_set!(model, prefix, :cdar_risk_, i, cdar_risk)
    state_set!(model, prefix, :ccdar_, i,
               JuMP.@constraint(model, sc * ((z_cdar - view(dd, 2:(T + 1))) .+ dar) >= 0))
    set_risk_bounds_and_expression!(model, opt, cdar_risk, r.settings, :cdar_risk_, i;
                                    prefix = prefix)
    return cdar_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `DistributionallyRobustConditionalDrawdownatRisk`
(DR-CDaR) to `model`.

Encodes a distributionally robust CDaR using Wasserstein ambiguity ball constraints
applied to the drawdown series.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::DistributionallyRobustConditionalDrawdownatRisk`: The DR-CDaR risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`DistributionallyRobustConditionalDrawdownatRisk`](@ref)
  - [`set_drawdown_constraints!`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::DistributionallyRobustConditionalDrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    w = get_w(model, prefix)
    X = pr.X
    dd = set_drawdown_constraints!(model, X; prefix = prefix)
    ddap1 = set_portfolio_drawdowns_plus_one!(model, X; prefix = prefix)
    T, N = size(X)

    alpha = r.alpha
    b1 = r.l
    radius = r.r

    a1 = -one(alpha)
    a2 = -one(alpha) - b1 * inv(alpha)
    b2 = b1 * (one(alpha) - inv(alpha))
    lb, tau, s, tu_drcdar, tv_drcdar, u, v = JuMP.@variables(model,
                                                             begin
                                                                 ()
                                                                 ()
                                                                 [1:T]
                                                                 [1:T]
                                                                 [1:T]
                                                                 [1:T, 1:N],
                                                                 (lower_bound = 0)
                                                                 [1:T, 1:N],
                                                                 (lower_bound = 0)
                                                             end)
    state_set!(model, prefix, :lb_drcdar_, i, lb)
    state_set!(model, prefix, :tau_drcdar_, i, tau)
    state_set!(model, prefix, :s_drcdar_, i, s)
    state_set!(model, prefix, :tu_drcdar_, i, tu_drcdar)
    state_set!(model, prefix, :tv_drcdar_, i, tv_drcdar)
    state_set!(model, prefix, :u_drcdar_, i, u)
    state_set!(model, prefix, :v_drcdar_, i, v)
    cu_drcdar, cv_drcdar, cu_drcdar_infnorm, cv_drcdar_infnorm, cu_drcdar_lb, cv_drcdar_lb = JuMP.@constraints(model,
                                                                                                               begin
                                                                                                                   sc *
                                                                                                                   (b1 *
                                                                                                                    tau .+
                                                                                                                    (a1 *
                                                                                                                     -view(dd,
                                                                                                                           2:(T + 1)) +
                                                                                                                     vec(sum(u .*
                                                                                                                             ddap1;
                                                                                                                             dims = 2)) -
                                                                                                                     s)) <=
                                                                                                                   0
                                                                                                                   sc *
                                                                                                                   (b2 *
                                                                                                                    tau .+
                                                                                                                    (a2 *
                                                                                                                     -view(dd,
                                                                                                                           2:(T + 1)) +
                                                                                                                     vec(sum(v .*
                                                                                                                             ddap1;
                                                                                                                             dims = 2)) -
                                                                                                                     s)) <=
                                                                                                                   0
                                                                                                                   [i = 1:T],
                                                                                                                   [sc *
                                                                                                                    tu_drcdar[i]
                                                                                                                    sc *
                                                                                                                    (-view(u,
                                                                                                                           i,
                                                                                                                           :) -
                                                                                                                     a1 *
                                                                                                                     w)] in
                                                                                                                   JuMP.MOI.NormInfinityCone(1 +
                                                                                                                                             N)
                                                                                                                   [i = 1:T],
                                                                                                                   [sc *
                                                                                                                    tv_drcdar[i]
                                                                                                                    sc *
                                                                                                                    (-view(v,
                                                                                                                           i,
                                                                                                                           :) -
                                                                                                                     a2 *
                                                                                                                     w)] in
                                                                                                                   JuMP.MOI.NormInfinityCone(1 +
                                                                                                                                             N)
                                                                                                                   sc *
                                                                                                                   (tu_drcdar .-
                                                                                                                    lb) <=
                                                                                                                   0
                                                                                                                   sc *
                                                                                                                   (tv_drcdar .-
                                                                                                                    lb) <=
                                                                                                                   0
                                                                                                               end)
    state_set!(model, prefix, :cu_drcdar_, i, cu_drcdar)
    state_set!(model, prefix, :cv_drcdar_, i, cv_drcdar)
    state_set!(model, prefix, :cu_drcdar_infnorm_, i, cu_drcdar_infnorm)
    state_set!(model, prefix, :cv_drcdar_infnorm_, i, cv_drcdar_infnorm)
    state_set!(model, prefix, :cu_drcdar_lb_, i, cu_drcdar_lb)
    state_set!(model, prefix, :cv_drcdar_lb_, i, cv_drcdar_lb)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    drcdar_risk = if isnothing(wi)
        JuMP.@expression(model, radius * lb + Statistics.mean(s))
    else
        JuMP.@expression(model, radius * lb + Statistics.mean(s, wi))
    end
    state_set!(model, prefix, :drcdar_risk_, i, drcdar_risk)
    set_risk_bounds_and_expression!(model, opt, drcdar_risk, r.settings, :drcdar_risk_, i;
                                    prefix = prefix)
    return drcdar_risk
end
