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
\\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{w}) &= t + \\ln_{\\kappa}\\!\\left(\\frac{1}{\\alpha T}\\right) z + \\sum_{t=1}^T (\\psi_t + \\theta_t)\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{w})``: Relativistic Value-at-Risk.
  - ``t``, ``z``, ``\\psi_t``, ``\\theta_t``: Dual variables for the power cone programme.
  - $(math_dict[:ln_kappa])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - ``\\kappa``: Relativistic parameter.

encoded via power cones ``\\mathcal{K}_{1/(1+\\kappa)}`` and ``\\mathcal{K}_{1/(1-\\kappa)}``.

For observation-weighted samples the weight vector is normalised to ``\\boldsymbol{w}`` with
``\\sum_{t=1}^T w_t = 1``. The Kaniadakis logarithm keeps the argument
``\\frac{1}{\\alpha T}``, and the sum ``\\sum_{t=1}^T (\\psi_t + \\theta_t)`` becomes
``T \\sum_{t=1}^T w_t (\\psi_t + \\theta_t)``.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r`: Risk measure instance with fields `alpha` and `kappa`.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`risk_series`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticValueatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; loss::Bool = true, prefix::Symbol = Symbol(""),
                               kwargs...)
    series, T = risk_series(model, NetReturnsRiskSeries(), pr; loss = loss, prefix = prefix)
    return set_relativistic_risk_constraints!(model, i, r, opt, pr, series, T,
                                              (; t = :t_rlvar_, z = :z_rlvar_,
                                               omega = :omega_rlvar_, psi = :psi_rlvar_,
                                               theta = :theta_rlvar_,
                                               epsilon = :epsilon_rlvar_,
                                               risk = :rlvar_risk_,
                                               pcone_a = :crlvar_pcone_a_,
                                               pcone_b = :crlvar_pcone_b_,
                                               exceedance = :crlvar_); prefix = prefix)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Encode the relativistic tail programme of `series` and register it under the names in
`keys`.

This is the shared body of `RelativisticValueatRisk` and `RelativisticDrawdownatRisk`. The
two are one pair of power cones over different series, so [`risk_series`](@ref) chooses the
series and this function writes the cones once.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::RiskMeasure`: The relativistic risk measure, read for `alpha`, `kappa`, `w` and
    `settings`.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])
  - `series`: The per-observation return series from [`risk_series`](@ref).
  - `T::Int`: The number of observations.
  - `keys::NamedTuple`: Bare Model State entry names, one per entry this builder registers.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `risk`: The relativistic risk expression added to the model.

# Related

  - [`risk_series`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
  - [`kappa_log`](@ref)
"""
function set_relativistic_risk_constraints!(model::JuMP.Model, i::Any, r::RiskMeasure,
                                            opt::RiskJuMPOptimisationEstimator,
                                            pr::AbstractPriorResult, series, T::Int,
                                            keys::NamedTuple; prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    alpha = r.alpha
    kappa = r.kappa
    ik2 = inv(2 * kappa)
    opk = one(kappa) + kappa
    omk = one(kappa) - kappa
    ik = inv(kappa)
    iopk = inv(opk)
    iomk = inv(omk)
    t, z, omega, psi, theta, epsilon = JuMP.@variables(model, begin
                                                           ()
                                                           (), (lower_bound = 0)
                                                           [1:T]
                                                           [1:T]
                                                           [1:T]
                                                           [1:T]
                                                       end)
    state_set!(model, prefix, keys.t, i, t)
    state_set!(model, prefix, keys.z, i, z)
    state_set!(model, prefix, keys.omega, i, omega)
    state_set!(model, prefix, keys.psi, i, psi)
    state_set!(model, prefix, keys.theta, i, theta)
    state_set!(model, prefix, keys.epsilon, i, epsilon)
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    lnk = kappa_log(inv(alpha * T), kappa)
    risk = if isnothing(wi)
        JuMP.@expression(model, t + lnk * z + sum(psi + theta))
    else
        wi /= sum(wi)
        JuMP.@expression(model, t + lnk * z + T * LinearAlgebra.dot(wi, psi + theta))
    end
    state_set!(model, prefix, keys.risk, i, risk)
    pcone_a, pcone_b, exceedance = JuMP.@constraints(model,
                                                     begin
                                                         [i = 1:T],
                                                         [sc * z * opk * ik2,
                                                          sc * psi[i] * opk * ik,
                                                          sc * epsilon[i]] in
                                                         JuMP.MOI.PowerCone(iopk)
                                                         [i = 1:T],
                                                         [sc * omega[i] * iomk,
                                                          sc * theta[i] * ik,
                                                          -sc * z * ik2] in
                                                         JuMP.MOI.PowerCone(omk)
                                                         sc *
                                                         ((epsilon + omega - series) .- t) <=
                                                         0
                                                     end)
    state_set!(model, prefix, keys.pcone_a, i, pcone_a)
    state_set!(model, prefix, keys.pcone_b, i, pcone_b)
    state_set!(model, prefix, keys.exceedance, i, exceedance)
    set_risk_bounds_and_expression!(model, opt, risk, r.settings, keys.risk, i;
                                    prefix = prefix)
    return risk
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
  - [`risk_series`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticDrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    series, T = risk_series(model, DrawdownRiskSeries(), pr; prefix = prefix)
    return set_relativistic_risk_constraints!(model, i, r, opt, pr, series, T,
                                              (; t = :t_rldar_, z = :z_rldar_,
                                               omega = :omega_rldar_, psi = :psi_rldar_,
                                               theta = :theta_rldar_,
                                               epsilon = :epsilon_rldar_,
                                               risk = :rldar_risk_,
                                               pcone_a = :crldar_pcone_a_,
                                               pcone_b = :crldar_pcone_b_,
                                               exceedance = :crldar_); prefix = prefix)
end
