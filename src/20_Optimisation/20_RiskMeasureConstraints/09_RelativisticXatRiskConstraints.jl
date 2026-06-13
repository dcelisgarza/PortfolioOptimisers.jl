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
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    key = Symbol(:rlvar_risk_, i)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, pr.X; prefix = prefix)
    T = length(net_X)
    alpha = r.alpha
    kappa = r.kappa
    t_rlvar, z_rlvar, omega_rlvar, psi_rlvar, theta_rlvar, epsilon_rlvar = model[Symbol(:t_rlvar_, i)], model[Symbol(:z_rlvar_, i)], model[Symbol(:omega_rlvar_, i)], model[Symbol(:psi_rlvar_, i)], model[Symbol(:theta_rlvar_, i)], model[Symbol(:epsilon_rlvar_, i)] = JuMP.@variables(model,
                                                                                                                                                                                                                                                                                          begin
                                                                                                                                                                                                                                                                                              ()
                                                                                                                                                                                                                                                                                              (),
                                                                                                                                                                                                                                                                                              (lower_bound = 0)
                                                                                                                                                                                                                                                                                              [1:T]
                                                                                                                                                                                                                                                                                              [1:T]
                                                                                                                                                                                                                                                                                              [1:T]
                                                                                                                                                                                                                                                                                              [1:T]
                                                                                                                                                                                                                                                                                          end)
    ik2 = inv(2 * kappa)
    opk = one(kappa) + kappa
    omk = one(kappa) - kappa
    ik = inv(kappa)
    iopk = inv(opk)
    iomk = inv(omk)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    rlvar_risk = model[key] = if isnothing(wi)
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
    model[Symbol(:crlvar_pcone_a_, i)], model[Symbol(:crlvar_pcone_b_, i)], model[Symbol(:crlvar_, i)] = JuMP.@constraints(model,
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
                                                                                                                                sc *
                                                                                                                                epsilon_rlvar[i]] in
                                                                                                                               JuMP.MOI.PowerCone(iopk)
                                                                                                                               [i = 1:T],
                                                                                                                               [sc *
                                                                                                                                omega_rlvar[i] *
                                                                                                                                iomk,
                                                                                                                                sc *
                                                                                                                                theta_rlvar[i] *
                                                                                                                                ik,
                                                                                                                                -sc *
                                                                                                                                z_rlvar *
                                                                                                                                ik2] in
                                                                                                                               JuMP.MOI.PowerCone(omk)
                                                                                                                               sc *
                                                                                                                               ((epsilon_rlvar +
                                                                                                                                 omega_rlvar -
                                                                                                                                 net_X) .-
                                                                                                                                t_rlvar) <=
                                                                                                                               0
                                                                                                                           end)
    set_risk_bounds_and_expression!(model, opt, rlvar_risk, r.settings, key)
    return rlvar_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `RelativisticValueatRiskRange` (RLVaR range) to `model`.

Uses power cone constraints for both lower-tail and upper-tail Tsallis entropy-based
risk measures parameterised by `kappa`, then computes their difference as the range risk.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::RelativisticValueatRiskRange`: The RLVaR range risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`RelativisticValueatRiskRange`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticValueatRiskRange,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    key = Symbol(:rlvar_range_risk_, i)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, pr.X; prefix = prefix)
    T = length(net_X)
    alpha = r.alpha
    kappa_a = r.kappa_a
    beta = r.beta
    kappa_b = r.kappa_b
    t_rlvar_l, z_rlvar_l, omega_rlvar_l, psi_rlvar_l, theta_rlvar_l, epsilon_rlvar_l, t_rlvar_h, z_rlvar_h, omega_rlvar_h, psi_rlvar_h, theta_rlvar_h, epsilon_rlvar_h = model[Symbol(:t_rlvar_l_, i)], model[Symbol(:z_rlvar_l_, i)], model[Symbol(:omega_rlvar_l_, i)], model[Symbol(:psi_rlvar_l_, i)], model[Symbol(:theta_rlvar_l_, i)], model[Symbol(:epsilon_rlvar_l_, i)], model[Symbol(:t_rlvar_h_, i)], model[Symbol(:z_rlvar_h_, i)], model[Symbol(:omega_rlvar_h_, i)], model[Symbol(:psi_rlvar_h_, i)], model[Symbol(:theta_rlvar_h_, i)], model[Symbol(:epsilon_rlvar_h_, i)] = JuMP.@variables(model,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              begin
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  (),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  (lower_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  ()
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  (),
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  (upper_bound = 0)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  [1:T]
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              end)
    ik2_a = inv(2 * kappa_a)
    ik2_b = inv(2 * kappa_b)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    rlvar_risk_l, rlvar_risk_h = model[Symbol(:rlvar_risk_l_, i)], model[Symbol(:rlvar_risk_h_, i)] = if isnothing(wi)
        iat = inv(alpha * T)
        ibt = inv(beta * T)
        lnk_a = (iat^kappa_a - iat^(-kappa_a)) * ik2_a
        lnk_b = (ibt^kappa_b - ibt^(-kappa_b)) * ik2_b
        JuMP.@expressions(model,
                          begin
                              t_rlvar_l +
                              lnk_a * z_rlvar_l +
                              sum(psi_rlvar_l + theta_rlvar_l)
                              t_rlvar_h +
                              lnk_b * z_rlvar_h +
                              sum(psi_rlvar_h + theta_rlvar_h)
                          end)
    else
        sw = sum(wi)
        iat = inv(alpha * sw)
        ibt = inv(beta * sw)
        lnk_a = (iat^kappa_a - iat^(-kappa_a)) * ik2_a
        lnk_b = (ibt^kappa_b - ibt^(-kappa_b)) * ik2_b
        JuMP.@expressions(model,
                          begin
                              t_rlvar_l +
                              lnk_a * z_rlvar_l +
                              LinearAlgebra.dot(wi, psi_rlvar_l + theta_rlvar_l)
                              t_rlvar_h +
                              lnk_b * z_rlvar_h +
                              LinearAlgebra.dot(wi, psi_rlvar_h + theta_rlvar_h)
                          end)
    end
    opk_a = one(kappa_a) + kappa_a
    omk_a = one(kappa_a) - kappa_a
    ik_a = inv(kappa_a)
    iopk_a = inv(opk_a)
    iomk_a = inv(omk_a)
    opk_b = one(kappa_b) + kappa_b
    omk_b = one(kappa_b) - kappa_b
    ik_b = inv(kappa_b)
    iopk_b = inv(opk_b)
    iomk_b = inv(omk_b)
    rlvar_range_risk = model[Symbol(:rlvar_range_risk_, i)] = JuMP.@expression(model,
                                                                               rlvar_risk_l -
                                                                               rlvar_risk_h)
    model[Symbol(:crlvar_l_pcone_a_, i)], model[Symbol(:crlvar_l_pcone_b_, i)], model[Symbol(:crlvar_l_, i)], model[Symbol(:crlvar_h_pcone_a_, i)], model[Symbol(:crlvar_h_pcone_b_, i)], model[Symbol(:crlvar_h_, i)] = JuMP.@constraints(model,
                                                                                                                                                                                                                                           begin
                                                                                                                                                                                                                                               [i = 1:T],
                                                                                                                                                                                                                                               [sc *
                                                                                                                                                                                                                                                z_rlvar_l *
                                                                                                                                                                                                                                                opk_a *
                                                                                                                                                                                                                                                ik2_a,
                                                                                                                                                                                                                                                sc *
                                                                                                                                                                                                                                                psi_rlvar_l[i] *
                                                                                                                                                                                                                                                opk_a *
                                                                                                                                                                                                                                                ik_a,
                                                                                                                                                                                                                                                sc *
                                                                                                                                                                                                                                                epsilon_rlvar_l[i]] in
                                                                                                                                                                                                                                               JuMP.MOI.PowerCone(iopk_a)
                                                                                                                                                                                                                                               [i = 1:T],
                                                                                                                                                                                                                                               [sc *
                                                                                                                                                                                                                                                omega_rlvar_l[i] *
                                                                                                                                                                                                                                                iomk_a,
                                                                                                                                                                                                                                                sc *
                                                                                                                                                                                                                                                theta_rlvar_l[i] *
                                                                                                                                                                                                                                                ik_a,
                                                                                                                                                                                                                                                sc *
                                                                                                                                                                                                                                                -z_rlvar_l *
                                                                                                                                                                                                                                                ik2_a] in
                                                                                                                                                                                                                                               JuMP.MOI.PowerCone(omk_a)
                                                                                                                                                                                                                                               sc *
                                                                                                                                                                                                                                               ((epsilon_rlvar_l +
                                                                                                                                                                                                                                                 omega_rlvar_l -
                                                                                                                                                                                                                                                 net_X) .-
                                                                                                                                                                                                                                                t_rlvar_l) <=
                                                                                                                                                                                                                                               0
                                                                                                                                                                                                                                               [i = 1:T],
                                                                                                                                                                                                                                               [sc *
                                                                                                                                                                                                                                                -z_rlvar_h *
                                                                                                                                                                                                                                                opk_b *
                                                                                                                                                                                                                                                ik2_b,
                                                                                                                                                                                                                                                sc *
                                                                                                                                                                                                                                                -psi_rlvar_h[i] *
                                                                                                                                                                                                                                                opk_b *
                                                                                                                                                                                                                                                ik_b,
                                                                                                                                                                                                                                                sc *
                                                                                                                                                                                                                                                -epsilon_rlvar_h[i]] in
                                                                                                                                                                                                                                               JuMP.MOI.PowerCone(iopk_b)
                                                                                                                                                                                                                                               [i = 1:T],
                                                                                                                                                                                                                                               [sc *
                                                                                                                                                                                                                                                -omega_rlvar_h[i] *
                                                                                                                                                                                                                                                iomk_b,
                                                                                                                                                                                                                                                sc *
                                                                                                                                                                                                                                                -theta_rlvar_h[i] *
                                                                                                                                                                                                                                                ik_b,
                                                                                                                                                                                                                                                sc *
                                                                                                                                                                                                                                                z_rlvar_h *
                                                                                                                                                                                                                                                ik2_b] in
                                                                                                                                                                                                                                               JuMP.MOI.PowerCone(omk_b)
                                                                                                                                                                                                                                               sc *
                                                                                                                                                                                                                                               ((epsilon_rlvar_h +
                                                                                                                                                                                                                                                 omega_rlvar_h -
                                                                                                                                                                                                                                                 net_X) .-
                                                                                                                                                                                                                                                t_rlvar_h) >=
                                                                                                                                                                                                                                               0
                                                                                                                                                                                                                                           end)
    set_risk_bounds_and_expression!(model, opt, rlvar_range_risk, r.settings, key)
    return rlvar_range_risk
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
    key = Symbol(:rldar_risk_, i)
    sc = get_constraint_scale(model)
    dd = set_drawdown_constraints!(model, pr.X; prefix = prefix)
    T = length(dd) - 1
    alpha = r.alpha
    kappa = r.kappa
    iat = inv(alpha * T)
    ik2 = inv(2 * kappa)
    lnk = (iat^kappa - iat^(-kappa)) * ik2
    opk = one(kappa) + kappa
    omk = one(kappa) - kappa
    ik = inv(kappa)
    iopk = inv(opk)
    iomk = inv(omk)
    t_rldar, z_rldar, omega_rldar, psi_rldar, theta_rldar, epsilon_rldar = model[Symbol(:t_rldar_, i)], model[Symbol(:z_rldar_, i)], model[Symbol(:omega_rldar_, i)], model[Symbol(:psi_rldar_, i)], model[Symbol(:theta_rldar_, i)], model[Symbol(:epsilon_rldar_, i)] = JuMP.@variables(model,
                                                                                                                                                                                                                                                                                          begin
                                                                                                                                                                                                                                                                                              ()
                                                                                                                                                                                                                                                                                              (),
                                                                                                                                                                                                                                                                                              (lower_bound = 0)
                                                                                                                                                                                                                                                                                              [1:T]
                                                                                                                                                                                                                                                                                              [1:T]
                                                                                                                                                                                                                                                                                              [1:T]
                                                                                                                                                                                                                                                                                              [1:T]
                                                                                                                                                                                                                                                                                          end)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    rldar_risk = model[key] = if isnothing(wi)
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
    model[Symbol(:crldar_pcone_a_, i)], model[Symbol(:crldar_pcone_b_, i)], model[Symbol(:crldar_, i)] = JuMP.@constraints(model,
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
                                                                                                                                sc *
                                                                                                                                epsilon_rldar[i]] in
                                                                                                                               JuMP.MOI.PowerCone(iopk)
                                                                                                                               [i = 1:T],
                                                                                                                               [sc *
                                                                                                                                omega_rldar[i] *
                                                                                                                                iomk,
                                                                                                                                sc *
                                                                                                                                theta_rldar[i] *
                                                                                                                                ik,
                                                                                                                                -sc *
                                                                                                                                z_rldar *
                                                                                                                                ik2] in
                                                                                                                               JuMP.MOI.PowerCone(omk)
                                                                                                                               sc *
                                                                                                                               ((epsilon_rldar +
                                                                                                                                 omega_rldar +
                                                                                                                                 view(dd,
                                                                                                                                      2:(T + 1))) .-
                                                                                                                                t_rldar) <=
                                                                                                                               0
                                                                                                                           end)
    set_risk_bounds_and_expression!(model, opt, rldar_risk, r.settings, key)
    return rldar_risk
end
