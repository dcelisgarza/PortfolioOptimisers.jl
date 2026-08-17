"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add Relativistic Value-at-Risk, RLVaR range, or Relativistic Drawdown-at-Risk constraints to
`model`.

Each overload uses power cone constraints (`PowerCone`) to encode the Tsallis entropy-based
risk measure parameterised by `kappa`. Auxiliary variables `t`, `z`, `omega`, `psi`,
`theta`, and `epsilon` are introduced. The range variant encodes both a lower-tail and
upper-tail relativistic expression.

# Mathematical definition

Relativistic Value-at-Risk (Damian et al. 2023):

```math
\\begin{align}
\\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{w}) &= t + c_{\\kappa}(\\alpha)\\, z + \\sum_{t=1}^T (\\psi_t + \\theta_t)\\,, \\\\
c_{\\kappa}(\\alpha) &= \\frac{(\\alpha T)^\\kappa - (\\alpha T)^{-\\kappa}}{2\\kappa}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{w})``: Relativistic Value-at-Risk.
  - ``t``, ``z``, ``\\psi_t``, ``\\theta_t``: Dual variables for the power cone programme.
  - ``c_\\kappa(\\alpha)``: Relativistic scaling coefficient.
  - $(math_dict[:alpha_rm])
  - ``\\kappa``: Relativistic parameter.

encoded via power cones ``\\mathcal{K}_{1/(1+\\kappa)}`` and ``\\mathcal{K}_{1/(1-\\kappa)}``.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r`: Risk measure instance with fields `alpha` and `kappa`.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`set_drawdown_constraints!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticValueatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; loss::Bool = true, prefix::Symbol = Symbol(""),
                               kwargs...)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, pr.X; prefix = prefix)
    if !loss
        net_X = -net_X
    end
    T = length(net_X)
    alpha = r.alpha
    kappa = r.kappa
    t_rlvar, z_rlvar, omega_rlvar, psi_rlvar, theta_rlvar, epsilon_rlvar = JuMP.@variables(model,
                                                                                           begin
                                                                                               ()
                                                                                               (),
                                                                                               (lower_bound = 0)
                                                                                               [1:T]
                                                                                               [1:T]
                                                                                               [1:T]
                                                                                               [1:T]
                                                                                           end)
    state_set!(model, prefix, :t_rlvar_, i, t_rlvar)
    state_set!(model, prefix, :z_rlvar_, i, z_rlvar)
    state_set!(model, prefix, :omega_rlvar_, i, omega_rlvar)
    state_set!(model, prefix, :psi_rlvar_, i, psi_rlvar)
    state_set!(model, prefix, :theta_rlvar_, i, theta_rlvar)
    state_set!(model, prefix, :epsilon_rlvar_, i, epsilon_rlvar)
    ik2 = inv(2 * kappa)
    opk = one(kappa) + kappa
    omk = one(kappa) - kappa
    ik = inv(kappa)
    iopk = inv(opk)
    iomk = inv(omk)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    rlvar_risk = if isnothing(wi)
        iat = inv(alpha * T)
        lnk = (iat^kappa - iat^(-kappa)) * ik2
        JuMP.@expression(model, t_rlvar + lnk * z_rlvar + sum(psi_rlvar + theta_rlvar))
    else
        iat = inv(alpha * sum(wi))
        lnk = (iat^kappa - iat^(-kappa)) * ik2
        JuMP.@expression(model,
                         t_rlvar +
                         lnk * z_rlvar +
                         LinearAlgebra.dot(wi, psi_rlvar + theta_rlvar))
    end
    state_set!(model, prefix, :rlvar_risk_, i, rlvar_risk)
    crlvar_pcone_a, crlvar_pcone_b, crlvar = JuMP.@constraints(model,
                                                               begin
                                                                   [i = 1:T],
                                                                   [sc *
                                                                    z_rlvar *
                                                                    opk *
                                                                    ik2,
                                                                    sc *
                                                                    psi_rlvar[i] *
                                                                    opk *
                                                                    ik,
                                                                    sc * epsilon_rlvar[i]] in
                                                                   JuMP.MOI.PowerCone(iopk)
                                                                   [i = 1:T],
                                                                   [sc *
                                                                    omega_rlvar[i] *
                                                                    iomk,
                                                                    sc *
                                                                    theta_rlvar[i] *
                                                                    ik,
                                                                    -sc * z_rlvar * ik2] in
                                                                   JuMP.MOI.PowerCone(omk)
                                                                   sc * ((epsilon_rlvar +
                                                                          omega_rlvar - net_X) .-
                                                                         t_rlvar) <= 0
                                                               end)
    state_set!(model, prefix, :crlvar_pcone_a_, i, crlvar_pcone_a)
    state_set!(model, prefix, :crlvar_pcone_b_, i, crlvar_pcone_b)
    state_set!(model, prefix, :crlvar_, i, crlvar)
    set_risk_bounds_and_expression!(model, opt, rlvar_risk, r.settings, :rlvar_risk_, i;
                                    prefix = prefix)
    return rlvar_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `RelativisticValueatRiskRange` (RLVaR range) to `model`.

Delegates to [`set_range_risk_constraints!`](@ref), which builds the loss tail from `alpha`
and `kappa_a` on the net portfolio returns, and the gain tail from `beta` and `kappa_b` on
their negation, then sums the two RLVaR expressions. Each tail brings its own pair of power
cones, shaped by *its own* deformation parameter.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::RelativisticValueatRiskRange`: The RLVaR range risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `rlvar_range_risk`: The combined `loss + gain` risk expression added to the model.

# Related

  - [`RelativisticValueatRiskRange`](@ref)
  - [`range_tails`](@ref)
  - [`set_range_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    return set_range_risk_constraints!(model, i, r, :rlvar_range_risk_, opt, pr, args...;
                                       prefix = prefix, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `RelativisticDrawdownatRisk` (RLDaR) to `model`.

Uses power cone constraints applied to the drawdown series to encode the relativistic
drawdown-at-risk parameterised by `kappa` at confidence level `r.alpha`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::RelativisticDrawdownatRisk`: The RLDaR risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`RelativisticDrawdownatRisk`](@ref)
  - [`set_drawdown_constraints!`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticDrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    dd = set_drawdown_constraints!(model, pr.X; prefix = prefix)
    T = length(dd) - 1
    alpha = r.alpha
    kappa = r.kappa
    ik2 = inv(2 * kappa)
    opk = one(kappa) + kappa
    omk = one(kappa) - kappa
    ik = inv(kappa)
    iopk = inv(opk)
    iomk = inv(omk)
    t_rldar, z_rldar, omega_rldar, psi_rldar, theta_rldar, epsilon_rldar = JuMP.@variables(model,
                                                                                           begin
                                                                                               ()
                                                                                               (),
                                                                                               (lower_bound = 0)
                                                                                               [1:T]
                                                                                               [1:T]
                                                                                               [1:T]
                                                                                               [1:T]
                                                                                           end)
    state_set!(model, prefix, :t_rldar_, i, t_rldar)
    state_set!(model, prefix, :z_rldar_, i, z_rldar)
    state_set!(model, prefix, :omega_rldar_, i, omega_rldar)
    state_set!(model, prefix, :psi_rldar_, i, psi_rldar)
    state_set!(model, prefix, :theta_rldar_, i, theta_rldar)
    state_set!(model, prefix, :epsilon_rldar_, i, epsilon_rldar)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    rldar_risk = if isnothing(wi)
        iat = inv(alpha * T)
        lnk = (iat^kappa - iat^(-kappa)) * ik2
        JuMP.@expression(model, t_rldar + lnk * z_rldar + sum(psi_rldar + theta_rldar))
    else
        iat = inv(alpha * sum(wi))
        lnk = (iat^kappa - iat^(-kappa)) * ik2
        JuMP.@expression(model,
                         t_rldar +
                         lnk * z_rldar +
                         LinearAlgebra.dot(wi, psi_rldar + theta_rldar))
    end
    state_set!(model, prefix, :rldar_risk_, i, rldar_risk)
    crldar_pcone_a, crldar_pcone_b, crldar = JuMP.@constraints(model,
                                                               begin
                                                                   [i = 1:T],
                                                                   [sc *
                                                                    z_rldar *
                                                                    opk *
                                                                    ik2,
                                                                    sc *
                                                                    psi_rldar[i] *
                                                                    opk *
                                                                    ik,
                                                                    sc * epsilon_rldar[i]] in
                                                                   JuMP.MOI.PowerCone(iopk)
                                                                   [i = 1:T],
                                                                   [sc *
                                                                    omega_rldar[i] *
                                                                    iomk,
                                                                    sc *
                                                                    theta_rldar[i] *
                                                                    ik,
                                                                    -sc * z_rldar * ik2] in
                                                                   JuMP.MOI.PowerCone(omk)
                                                                   sc * ((epsilon_rldar +
                                                                          omega_rldar +
                                                                          view(dd, 2:(T + 1))) .-
                                                                         t_rldar) <= 0
                                                               end)
    state_set!(model, prefix, :crldar_pcone_a_, i, crldar_pcone_a)
    state_set!(model, prefix, :crldar_pcone_b_, i, crldar_pcone_b)
    state_set!(model, prefix, :crldar_, i, crldar)
    set_risk_bounds_and_expression!(model, opt, rldar_risk, r.settings, :rldar_risk_, i;
                                    prefix = prefix)
    return rldar_risk
end
