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

  - [`risk_series`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::ConditionalValueatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; loss::Bool = true, prefix::Symbol = Symbol(""),
                               kwargs...)
    series, T = risk_series(model, NetReturnsRiskSeries(), pr; loss = loss, prefix = prefix)
    return set_conditional_risk_constraints!(model, i, r, opt, pr, series, T,
                                             (; var = :var_, z = :z_cvar_,
                                              risk = :cvar_risk_, exceedance = :ccvar_);
                                             prefix = prefix)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Encode the Rockafellar-Uryasev programme of `series` and register it under the names in
`keys`.

This is the shared body of `ConditionalValueatRisk` and `ConditionalDrawdownatRisk`. The two
are one linearisation over different series, so [`risk_series`](@ref) chooses the series and
this function writes the exceedance constraint once.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::RiskMeasure`: The conditional risk measure, read for `alpha`, `w` and `settings`.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])
  - `series`: The per-observation return series from [`risk_series`](@ref).
  - `T::Int`: The number of observations.
  - `keys::NamedTuple`: Bare Model State entry names, one per entry this builder registers.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `risk`: The conditional risk expression added to the model.

# Related

  - [`risk_series`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_conditional_risk_constraints!(model::JuMP.Model, i::Any, r::RiskMeasure,
                                           opt::RiskJuMPOptimisationEstimator,
                                           pr::AbstractPriorResult, series, T::Int,
                                           keys::NamedTuple; prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    var, z = JuMP.@variables(model, begin
                                 ()
                                 [1:T], (lower_bound = 0)
                             end)
    state_set!(model, prefix, keys.var, i, var)
    state_set!(model, prefix, keys.z, i, z)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    risk = if isnothing(wi)
        iat = inv(r.alpha * T)
        JuMP.@expression(model, var + sum(z) * iat)
    else
        iat = inv(r.alpha * sum(wi))
        JuMP.@expression(model, var + LinearAlgebra.dot(wi, z) * iat)
    end
    state_set!(model, prefix, keys.risk, i, risk)
    state_set!(model, prefix, keys.exceedance, i,
               JuMP.@constraint(model, sc * ((z + series) .+ var) >= 0))
    set_risk_bounds_and_expression!(model, opt, risk, r.settings, keys.risk, i;
                                    prefix = prefix)
    return risk
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
    w = get_w(model, prefix)
    series, T = risk_series(model, NetReturnsRiskSeries(), pr; loss = loss, prefix = prefix)
    # The gain tail carries its own ambiguity ball, and `:Xap1` is not indexed by measure,
    # so the tail's entries are namespaced rather than allowed to collide with the loss
    # tail's.
    X = loss ? pr.X : -pr.X
    prefix = loss ? prefix : nested_prefix(prefix, :gain_)
    ambiguity = set_asset_returns_plus_one!(model, X; prefix = prefix)
    return set_dr_conditional_risk_constraints!(model, i, r, opt, pr, w, series, ambiguity,
                                                T,
                                                (; lb = :lb_drcvar_, tau = :tau_drcvar_,
                                                 s = :s_drcvar_, tu = :tu_drcvar_,
                                                 tv = :tv_drcvar_, u = :u_drcvar_,
                                                 v = :v_drcvar_, cu = :cu_drcvar_,
                                                 cv = :cv_drcvar_,
                                                 cu_infnorm = :cu_drcvar_infnorm_,
                                                 cv_infnorm = :cv_drcvar_infnorm_,
                                                 cu_lb = :cu_drcvar_lb_,
                                                 cv_lb = :cv_drcvar_lb_,
                                                 risk = :drcvar_risk_); prefix = prefix)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Encode the Wasserstein-robust conditional programme of `series` and register it under the
names in `keys`.

This is the shared body of `DistributionallyRobustConditionalValueatRisk` and
`DistributionallyRobustConditionalDrawdownatRisk`. The two are one ambiguity-ball programme
over different series, so [`risk_series`](@ref) chooses the series and this function writes
the infinity-norm cones once.

`ambiguity` is the per-observation, per-asset matrix the transport cost is measured against:
gross asset returns for the returns twin, drawdowns-plus-one for the drawdown twin.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::RiskMeasure`: The robust conditional risk measure, read for `alpha`, `l`, `r`, `w`
    and `settings`.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])
  - `w`: The portfolio weight variables, read under the *outer* prefix.
  - `series`: The per-observation return series from [`risk_series`](@ref).
  - `ambiguity`: The `T × N` matrix the ambiguity ball is measured against.
  - `T::Int`: The number of observations.
  - `keys::NamedTuple`: Bare Model State entry names, one per entry this builder registers.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `risk`: The robust conditional risk expression added to the model.

# Related

  - [`risk_series`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_dr_conditional_risk_constraints!(model::JuMP.Model, i::Any, r::RiskMeasure,
                                              opt::RiskJuMPOptimisationEstimator,
                                              pr::AbstractPriorResult, w, series, ambiguity,
                                              T::Int, keys::NamedTuple;
                                              prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    N = size(pr.X, 2)
    alpha = r.alpha
    b1 = r.l
    radius = r.r
    a1 = -one(alpha)
    a2 = -one(alpha) - b1 * inv(alpha)
    b2 = b1 * (one(alpha) - inv(alpha))
    lb, tau, s, tu, tv, u, v = JuMP.@variables(model, begin
                                                   ()
                                                   ()
                                                   [1:T]
                                                   [1:T]
                                                   [1:T]
                                                   [1:T, 1:N], (lower_bound = 0)
                                                   [1:T, 1:N], (lower_bound = 0)
                                               end)
    state_set!(model, prefix, keys.lb, i, lb)
    state_set!(model, prefix, keys.tau, i, tau)
    state_set!(model, prefix, keys.s, i, s)
    state_set!(model, prefix, keys.tu, i, tu)
    state_set!(model, prefix, keys.tv, i, tv)
    state_set!(model, prefix, keys.u, i, u)
    state_set!(model, prefix, keys.v, i, v)
    u_cost = vec(sum(u .* ambiguity; dims = 2))
    v_cost = vec(sum(v .* ambiguity; dims = 2))
    state_set!(model, prefix, keys.cu, i,
               JuMP.@constraint(model, sc * (b1 * tau .+ (a1 * series + u_cost - s)) <= 0))
    state_set!(model, prefix, keys.cv, i,
               JuMP.@constraint(model, sc * (b2 * tau .+ (a2 * series + v_cost - s)) <= 0))
    state_set!(model, prefix, keys.cu_infnorm, i,
               JuMP.@constraint(model, [i = 1:T],
                                [sc * tu[i]
                                 sc * (-view(u, i, :) - a1 * w)] in
                                JuMP.MOI.NormInfinityCone(1 + N)))
    state_set!(model, prefix, keys.cv_infnorm, i,
               JuMP.@constraint(model, [i = 1:T],
                                [sc * tv[i]
                                 sc * (-view(v, i, :) - a2 * w)] in
                                JuMP.MOI.NormInfinityCone(1 + N)))
    state_set!(model, prefix, keys.cu_lb, i, JuMP.@constraint(model, sc * (tu .- lb) <= 0))
    state_set!(model, prefix, keys.cv_lb, i, JuMP.@constraint(model, sc * (tv .- lb) <= 0))
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    risk = if isnothing(wi)
        JuMP.@expression(model, radius * lb + Statistics.mean(s))
    else
        JuMP.@expression(model, radius * lb + Statistics.mean(s, wi))
    end
    state_set!(model, prefix, keys.risk, i, risk)
    set_risk_bounds_and_expression!(model, opt, risk, r.settings, keys.risk, i;
                                    prefix = prefix)
    return risk
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
  - [`risk_series`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::ConditionalDrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    series, T = risk_series(model, DrawdownRiskSeries(), pr; prefix = prefix)
    return set_conditional_risk_constraints!(model, i, r, opt, pr, series, T,
                                             (; var = :dar_, z = :z_cdar_,
                                              risk = :cdar_risk_, exceedance = :ccdar_);
                                             prefix = prefix)
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
  - [`risk_series`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::DistributionallyRobustConditionalDrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    w = get_w(model, prefix)
    series, T = risk_series(model, DrawdownRiskSeries(), pr; prefix = prefix)
    ambiguity = set_portfolio_drawdowns_plus_one!(model, pr.X; prefix = prefix)
    return set_dr_conditional_risk_constraints!(model, i, r, opt, pr, w, series, ambiguity,
                                                T,
                                                (; lb = :lb_drcdar_, tau = :tau_drcdar_,
                                                 s = :s_drcdar_, tu = :tu_drcdar_,
                                                 tv = :tv_drcdar_, u = :u_drcdar_,
                                                 v = :v_drcdar_, cu = :cu_drcdar_,
                                                 cv = :cv_drcdar_,
                                                 cu_infnorm = :cu_drcdar_infnorm_,
                                                 cv_infnorm = :cv_drcdar_infnorm_,
                                                 cu_lb = :cu_drcdar_lb_,
                                                 cv_lb = :cv_drcdar_lb_,
                                                 risk = :drcdar_risk_); prefix = prefix)
end
