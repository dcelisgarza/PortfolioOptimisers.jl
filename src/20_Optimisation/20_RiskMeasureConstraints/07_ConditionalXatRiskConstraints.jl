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
    key = Symbol(:cvar_risk_, i)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, pr.X; prefix = prefix)
    if !loss
        net_X = -net_X
    end
    T = length(net_X)
    var, z_cvar = model[Symbol(:var_, i)], model[Symbol(:z_cvar_, i)] = JuMP.@variables(model,
                                                                                        begin
                                                                                            ()
                                                                                            [1:T],
                                                                                            (lower_bound = 0)
                                                                                        end)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    cvar_risk = model[key] = if isnothing(wi)
        iat = inv(r.alpha * T)
        JuMP.@expression(model, var + sum(z_cvar) * iat)
    else
        iat = inv(r.alpha * sum(wi))
        JuMP.@expression(model, var + LinearAlgebra.dot(wi, z_cvar) * iat)
    end
    model[Symbol(:ccvar_, i)] = JuMP.@constraint(model, sc * ((z_cvar + net_X) .+ var) >= 0)
    set_risk_bounds_and_expression!(model, opt, cvar_risk, r.settings, key)
    return cvar_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `ConditionalValueatRiskRange` to `model`.

Introduces lower-tail and upper-tail CVaR variables and auxiliary exceedance variables,
then computes the CVaR range as their difference.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::ConditionalValueatRiskRange`: The CVaR range risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`ConditionalValueatRiskRange`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::ConditionalValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    key = Symbol(:cvar_range_risk_, i)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, pr.X; prefix = prefix)
    T = length(net_X)
    var_l, z_cvar_l, var_h, z_cvar_h = model[Symbol(:var_l_, i)], model[Symbol(:z_cvar_l_, i)], model[Symbol(:var_h_, i)], model[Symbol(:z_cvar_h_, i)] = JuMP.@variables(model,
                                                                                                                                                                          begin
                                                                                                                                                                              ()
                                                                                                                                                                              [1:T],
                                                                                                                                                                              (lower_bound = 0)
                                                                                                                                                                              ()
                                                                                                                                                                              [1:T],
                                                                                                                                                                              (upper_bound = 0)
                                                                                                                                                                          end)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    cvar_risk_l, cvar_risk_h = if isnothing(wi)
        iat = inv(r.alpha * T)
        ibt = inv(r.beta * T)
        model[Symbol(:cvar_risk_l_, i)], model[Symbol(:cvar_risk_h_, i)] = JuMP.@expressions(model,
                                                                                             begin
                                                                                                 var_l +
                                                                                                 sum(z_cvar_l) *
                                                                                                 iat
                                                                                                 var_h +
                                                                                                 sum(z_cvar_h) *
                                                                                                 ibt
                                                                                             end)
    else
        sw = sum(wi)
        iat = inv(r.alpha * sw)
        ibt = inv(r.beta * sw)
        model[Symbol(:cvar_risk_l_, i)], model[Symbol(:cvar_risk_h_, i)] = JuMP.@expressions(model,
                                                                                             begin
                                                                                                 var_l +
                                                                                                 LinearAlgebra.dot(wi,
                                                                                                                   z_cvar_l) *
                                                                                                 iat
                                                                                                 var_h +
                                                                                                 LinearAlgebra.dot(wi,
                                                                                                                   z_cvar_h) *
                                                                                                 ibt
                                                                                             end)
    end
    cvar_range_risk = model[key] = JuMP.@expression(model, cvar_risk_l - cvar_risk_h)
    model[Symbol(:ccvar_l_, i)], model[Symbol(:ccvar_h_, i)] = JuMP.@constraints(model,
                                                                                 begin
                                                                                     sc *
                                                                                     ((z_cvar_l +
                                                                                       net_X) .+
                                                                                      var_l) >=
                                                                                     0
                                                                                     sc *
                                                                                     ((z_cvar_h +
                                                                                       net_X) .+
                                                                                      var_h) <=
                                                                                     0
                                                                                 end)
    set_risk_bounds_and_expression!(model, opt, cvar_range_risk, r.settings, key)
    return cvar_range_risk
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
    key = Symbol(:drcvar_risk_, i)
    sc = get_constraint_scale(model)
    w = get_w(model, prefix)
    X = pr.X
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    if !loss
        net_X = -net_X
    end
    Xap1 = set_portfolio_returns_plus_one!(model, X; prefix = prefix)
    T, N = size(X)

    alpha = r.alpha
    b1 = r.l
    radius = r.r

    a1 = -one(alpha)
    a2 = -one(alpha) - b1 * inv(alpha)
    b2 = b1 * (one(alpha) - inv(alpha))
    lb, tau, s, tu_drcvar, tv_drcvar, u, v = model[Symbol(:lb_drcvar_, i)], model[Symbol(:tau_drcvar_, i)], model[Symbol(:s_drcvar_, i)], model[Symbol(:tu_drcvar_, i)], model[Symbol(:tv_drcvar_, i)], model[Symbol(:u_drcvar_, i)], model[Symbol(:v_drcvar_, i)] = JuMP.@variables(model,
                                                                                                                                                                                                                                                                                     begin
                                                                                                                                                                                                                                                                                         ()
                                                                                                                                                                                                                                                                                         ()
                                                                                                                                                                                                                                                                                         [1:T]
                                                                                                                                                                                                                                                                                         [1:T]
                                                                                                                                                                                                                                                                                         [1:T]
                                                                                                                                                                                                                                                                                         [1:T,
                                                                                                                                                                                                                                                                                          1:N],
                                                                                                                                                                                                                                                                                         (lower_bound = 0)
                                                                                                                                                                                                                                                                                         [1:T,
                                                                                                                                                                                                                                                                                          1:N],
                                                                                                                                                                                                                                                                                         (lower_bound = 0)
                                                                                                                                                                                                                                                                                     end)
    model[Symbol(:cu_drcvar_, i)], model[Symbol(:cv_drcvar_, i)], model[Symbol(:cu_drcvar_infnorm_, i)], model[Symbol(:cv_drcvar_infnorm_, i)], model[Symbol(:cu_drcvar_lb_, i)], model[Symbol(:cv_drcvar_lb_, i)] = JuMP.@constraints(model,
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
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    drcvar_risk = model[key] = if isnothing(wi)
        JuMP.@expression(model, radius * lb + Statistics.mean(s))
    else
        JuMP.@expression(model, radius * lb + Statistics.mean(s, wi))
    end
    set_risk_bounds_and_expression!(model, opt, drcvar_risk, r.settings, key)
    return drcvar_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `DistributionallyRobustConditionalValueatRiskRange`
(DR-CVaR range) to `model`.

Encodes both lower-tail and upper-tail distributionally robust CVaR expressions using
Wasserstein ambiguity ball constraints, then computes their difference as the range risk.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::DistributionallyRobustConditionalValueatRiskRange`: The DR-CVaR range risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`DistributionallyRobustConditionalValueatRiskRange`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::DistributionallyRobustConditionalValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    key = Symbol(:drcvar_risk_range_, i)
    sc = get_constraint_scale(model)
    w = get_w(model, prefix)
    X = pr.X
    net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
    Xap1 = set_portfolio_returns_plus_one!(model, X; prefix = prefix)
    T, N = size(X)

    alpha = r.alpha
    b1_l = r.l_a
    radius_l = r.r_a

    beta = r.beta
    b1_h = r.l_b
    radius_h = r.r_b

    a1_l = -one(alpha)
    a2_l = -one(alpha) - b1_l * inv(alpha)
    b2_l = b1_l * (one(alpha) - inv(alpha))

    a1_h = -one(beta)
    a2_h = -one(beta) - b1_h * inv(beta)
    b2_h = b1_h * (one(beta) - inv(beta))
    lb_l, tau_l, s_l, tu_drcvar_l, tv_drcvar_l, u_l, v_l, lb_h, tau_h, s_h, tu_drcvar_h, tv_drcvar_h, u_h, v_h = model[Symbol(:lb_drcvar_l_, i)], model[Symbol(:tau_drcvar_l_, i)], model[Symbol(:s_drcvar_l_, i)], model[Symbol(:tu_drcvar_l_, i)], model[Symbol(:tv_drcvar_l_, i)], model[Symbol(:u_drcvar_l_, i)], model[Symbol(:v_drcvar_l_, i)], model[Symbol(:lb_drcvar_h_, i)], model[Symbol(:tau_drcvar_h_, i)], model[Symbol(:s_drcvar_h_, i)], model[Symbol(:tu_drcvar_h_, i)], model[Symbol(:tv_drcvar_h_, i)], model[Symbol(:u_drcvar_h_, i)], model[Symbol(:v_drcvar_h_, i)] = JuMP.@variables(model,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            begin
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                [1:T,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 1:N],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                (lower_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                [1:T,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 1:N],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                (lower_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                ()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                [1:T,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 1:N],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                (upper_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                [1:T,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 1:N],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                (upper_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            end)
    model[Symbol(:cu_drcvar_l_, i)], model[Symbol(:cv_drcvar_l_, i)], model[Symbol(:cu_drcvar_infnorm_l_, i)], model[Symbol(:cv_drcvar_infnorm_l_, i)], model[Symbol(:cu_drcvar_lb_l_, i)], model[Symbol(:cv_drcvar_lb_l_, i)], model[Symbol(:cu_drcvar_h_, i)], model[Symbol(:cv_drcvar_h_, i)], model[Symbol(:cu_drcvar_infnorm_h_, i)], model[Symbol(:cv_drcvar_infnorm_h_, i)], model[Symbol(:cu_drcvar_lb_h_, i)], model[Symbol(:cv_drcvar_lb_h_, i)] = JuMP.@constraints(model,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               begin
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   (b1_l *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    tau_l .+
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    (a1_l *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     net_X +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     vec(sum(u_l .*
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             Xap1;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             dims = 2)) -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     s_l)) <=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   (b2_l *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    tau_l .+
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    (a2_l *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     net_X +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     vec(sum(v_l .*
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             Xap1;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             dims = 2)) -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     s_l)) <=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   [i = 1:T],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   [sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    tu_drcvar_l[i]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    (-view(u_l,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           i,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           :) -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     a1_l *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     w)] in
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   JuMP.MOI.NormInfinityCone(1 +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             N)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   [i = 1:T],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   [sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    tv_drcvar_l[i]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    (-view(v_l,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           i,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           :) -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     a2_l *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     w)] in
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   JuMP.MOI.NormInfinityCone(1 +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             N)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   (tu_drcvar_l .-
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    lb_l) <=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   (tv_drcvar_l .-
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    lb_l) <=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   (b1_h *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    tau_h .+
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    (a1_h *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     net_X +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     vec(sum(u_h .*
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             Xap1;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             dims = 2)) -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     s_h)) >=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   (b2_h *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    tau_h .+
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    (a2_h *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     net_X +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     vec(sum(v_h .*
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             Xap1;
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             dims = 2)) -
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     s_h)) >=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   [i = 1:T],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   [-sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    tu_drcvar_h[i]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    (view(u_h,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          i,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          :) +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     a1_h *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     w)] in
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   JuMP.MOI.NormInfinityCone(1 +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             N)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   [i = 1:T],
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   [-sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    tv_drcvar_h[i]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    (view(v_h,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          i,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          :) +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     a2_h *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     w)] in
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   JuMP.MOI.NormInfinityCone(1 +
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             N)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   (tu_drcvar_h .-
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    lb_h) >=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   sc *
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   (tv_drcvar_h .-
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    lb_h) >=
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   0
                                                                                                                                                                                                                                                                                                                                                                                                                                                                               end)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    drcvar_risk_l, drcvar_risk_h = model[Symbol(:drcvar_risk_l_, i)], model[Symbol(:drcvar_risk_h_, i)] = if isnothing(wi)
        JuMP.@expressions(model, begin
                              radius_l * lb_l + Statistics.mean(s_l)
                              radius_h * lb_h + Statistics.mean(s_h)
                          end)
    else
        JuMP.@expressions(model, begin
                              radius_l * lb_l + Statistics.mean(s_l, wi)
                              radius_h * lb_h + Statistics.mean(s_h, wi)
                          end)
    end
    drcvar_risk_range = model[key] = JuMP.@expression(model, drcvar_risk_l - drcvar_risk_h)
    set_risk_bounds_and_expression!(model, opt, drcvar_risk_range, r.settings, key)
    return drcvar_risk_range
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
    key = Symbol(:cdar_risk_, i)
    sc = get_constraint_scale(model)
    dd = set_drawdown_constraints!(model, pr.X; prefix = prefix)
    T = length(dd) - 1
    iat = inv(r.alpha * T)
    dar, z_cdar = model[Symbol(:dar_, i)], model[Symbol(:z_cdar_, i)] = JuMP.@variables(model,
                                                                                        begin
                                                                                            ()
                                                                                            [1:T],
                                                                                            (lower_bound = 0)
                                                                                        end)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    cdar_risk = model[key] = if isnothing(wi)
        iat = inv(r.alpha * T)
        JuMP.@expression(model, dar + sum(z_cdar) * iat)
    else
        iat = inv(r.alpha * sum(wi))
        JuMP.@expression(model, dar + LinearAlgebra.dot(wi, z_cdar) * iat)
    end
    model[Symbol(:ccdar_, i)] = JuMP.@constraint(model,
                                                 sc *
                                                 ((z_cdar - view(dd, 2:(T + 1))) .+ dar) >=
                                                 0)
    set_risk_bounds_and_expression!(model, opt, cdar_risk, r.settings, key)
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
    key = Symbol(:drcvar_risk_, i)
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
    lb, tau, s, tu_drcdar, tv_drcdar, u, v = model[Symbol(:lb_drcdar_, i)], model[Symbol(:tau_drcdar_, i)], model[Symbol(:s_drcdar_, i)], model[Symbol(:tu_drcdar_, i)], model[Symbol(:tv_drcdar_, i)], model[Symbol(:u_drcdar_, i)], model[Symbol(:v_drcdar_, i)] = JuMP.@variables(model,
                                                                                                                                                                                                                                                                                     begin
                                                                                                                                                                                                                                                                                         ()
                                                                                                                                                                                                                                                                                         ()
                                                                                                                                                                                                                                                                                         [1:T]
                                                                                                                                                                                                                                                                                         [1:T]
                                                                                                                                                                                                                                                                                         [1:T]
                                                                                                                                                                                                                                                                                         [1:T,
                                                                                                                                                                                                                                                                                          1:N],
                                                                                                                                                                                                                                                                                         (lower_bound = 0)
                                                                                                                                                                                                                                                                                         [1:T,
                                                                                                                                                                                                                                                                                          1:N],
                                                                                                                                                                                                                                                                                         (lower_bound = 0)
                                                                                                                                                                                                                                                                                     end)
    model[Symbol(:cu_drcdar_, i)], model[Symbol(:cv_drcdar_, i)], model[Symbol(:cu_drcdar_infnorm_, i)], model[Symbol(:cv_drcdar_infnorm_, i)], model[Symbol(:cu_drcdar_lb_, i)], model[Symbol(:cv_drcdar_lb_, i)] = JuMP.@constraints(model,
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
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    drcdar_risk = model[key] = if isnothing(wi)
        JuMP.@expression(model, radius * lb + Statistics.mean(s))
    else
        JuMP.@expression(model, radius * lb + Statistics.mean(s, wi))
    end
    set_risk_bounds_and_expression!(model, opt, drcdar_risk, r.settings, key)
    return drcdar_risk
end
