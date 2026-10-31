"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds the Relativistic Value-at-Risk of the net portfolio returns to `model`, and returns its risk expression.

[`set_relativistic_risk_constraints!`](@ref) builds the power cones, and this method gives it the series of the net portfolio returns. A range measure calls this method once for each tail, and `loss = false` gives the gain tail.

# Mathematical definition

Equation 15 of [rlvar](@cite) states the Relativistic Value-at-Risk of the portfolio as a power cone programme:

```math
\\begin{align}
\\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{w}) = \\underset{t,\\, z,\\, \\boldsymbol{\\psi},\\, \\boldsymbol{\\theta},\\, \\boldsymbol{\\epsilon},\\, \\boldsymbol{\\omega}}{\\min} \\quad & t + \\ln_{\\kappa}\\!\\left(\\tfrac{1}{\\alpha T}\\right) z + T \\sum_{i=1}^{T} p_i (\\psi_i + \\theta_i) \\\\
\\mathrm{s.t.} \\quad & -\\hat{r}_i - t + \\epsilon_i + \\omega_i \\leq 0 \\quad \\forall i\\,, \\\\
& \\left(\\tfrac{z(1+\\kappa)}{2\\kappa},\\, \\tfrac{\\psi_i(1+\\kappa)}{\\kappa},\\, \\epsilon_i\\right) \\in \\mathcal{K}_{\\mathrm{pow}}\\!\\left(\\tfrac{1}{1+\\kappa}\\right) \\quad \\forall i\\,, \\\\
& \\left(\\tfrac{\\omega_i}{1-\\kappa},\\, \\tfrac{\\theta_i}{\\kappa},\\, -\\tfrac{z}{2\\kappa}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}(1-\\kappa) \\quad \\forall i\\,, \\\\
& z \\geq 0\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{w})``: Relativistic Value-at-Risk of the portfolio.
  - ``\\hat{r}_i``: Net portfolio return at observation ``i``, so ``-\\hat{r}_i`` is its loss.
  - $(math_dict[:p_i_obs])
  - $(math_dict[:rlvar_t])
  - $(math_dict[:rlvar_z_ge])
  - $(math_dict[:rlvar_aux])
  - $(math_dict[:K_pow])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:T])

Equation 15 of [rlvar](@cite) prints the first row as ``-X - t + \\epsilon + \\omega \\leq 0``, where ``X`` is the loss. The sign is a misprint. Equation 18 of that work writes the loss ``-\\hat{r}_i``, as the row above does. With this row the minimum equals the dual form of Equation 12: 0.0068504374 against 0.0068504424 for six assets, 200 observations, ``\\alpha = 0.05`` and ``\\kappa = 0.3``.

[rlvar](@cite) states the programme for equal observation weights, ``p_i = 1/T``. The weights ``p_i`` are the library's extension, and [`RelativisticValueatRisk`](@ref) states them.

# Algorithm

 1. Read `series`, the net portfolio returns, and `T`, its length, with [`risk_series`](@ref). When `loss` is `false`, `series` is their negation.
 2. Build the programme of `series` with [`set_relativistic_risk_constraints!`](@ref), giving `risk`. Each key names `rlvar`, for example `t_rlvar_` and `crlvar_pcone_a_`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::RelativisticValueatRisk`: The RLVaR risk measure, read for `alpha`, `kappa`, `w` and `settings`.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Keyword arguments

  - `loss::Bool`: `true` reads the net portfolio returns, and `false` reads their negation, the series of a gain tail (default: `true`).
  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `risk`: The RLVaR risk expression added to the model, registered under the key `rlvar_risk_`.

# Related

  - [`RelativisticValueatRisk`](@ref)
  - [`set_relativistic_risk_constraints!`](@ref)
  - [`risk_series`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)

# References

  - $(ref_dict[:rlvar]) Section 4.1, Equations 12 and 15, and Section 4.2, Equation 18.
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticValueatRisk,
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               loss::Bool = true, prefix::Symbol = Symbol(""), kwargs...)
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

Builds the relativistic tail programme of `series` in `model`, and returns its risk expression.

The RLVaR and the RLDaR are one programme over two series. [`risk_series`](@ref) gives the series of each measure, and the keys in `keys` keep the entries of the two measures apart. The variables and the rows cover the observations of positive weight only. An observation of weight zero does not change the measure, but its rows make the programme harder to solve.

# Algorithm

 1. Read `sc`, the constraint scale.
 2. Compute `opk`, `omk`, `ik`, `ik2`, `iopk` and `iomk`, the constants of the cones, from `kappa`.
 3. Read `wi`, the observation weights of `r`, else those of `pr`. Check them against `pr.X`.
 4. Find `rows`, the indices of the observations of positive weight, and `N`, their number. Without `wi`, `rows` holds every observation.
 5. Create `t`, `z`, `omega`, `psi`, `theta` and `epsilon`, and register each under its key.
 6. Compute `lnk`, the Kaniadakis logarithm of ``1/(\\alpha T)``, with [`kappa_log`](@ref).
 7. Build `risk`, and register it under `keys.risk`. When `wi` is stated, divide its entries in `rows` by the sum of `wi` first, giving `wp`.
 8. Register the three rows below under their keys, over the entries of `series` in `rows`.
 9. Bound `risk` and add it to `risk_vec` with [`set_risk_bounds_and_expression!`](@ref), from `r.settings`.
10. Return `risk`.

# JuMP formulation

## Variables

  - `keys.t`: created, the free scalar ``t``. The RLVaR key is `t_rlvar_`, and the RLDaR key is `t_rldar_`.
  - `keys.z`: created, the scalar ``z``, bounded below by zero. The keys are `z_rlvar_` and `z_rldar_`.
  - `keys.psi`, `keys.theta`, `keys.epsilon`, `keys.omega`: created, the free ``N \\times 1`` vectors ``\\boldsymbol{\\psi}``, ``\\boldsymbol{\\theta}``, ``\\boldsymbol{\\epsilon}`` and ``\\boldsymbol{\\omega}``, with one entry for each observation in ``\\mathcal{I}``. The keys are `psi_rlvar_`, `theta_rlvar_`, `epsilon_rlvar_` and `omega_rlvar_`, or the same names with `rldar_`.

The rows read `series`, an expression of the weights that [`risk_series`](@ref) returns.

## Expressions

  - `keys.risk`: ``t + \\ln_{\\kappa}\\!\\left(\\tfrac{1}{\\alpha T}\\right) z + T \\sum_{i \\in \\mathcal{I}} p_i (\\psi_i + \\theta_i)``. The keys are `rlvar_risk_` and `rldar_risk_`.

## Constraints

  - `keys.pcone_a`: ``s_c \\left(\\tfrac{z(1+\\kappa)}{2\\kappa},\\, \\tfrac{\\psi_i(1+\\kappa)}{\\kappa},\\, \\epsilon_i\\right) \\in \\mathcal{K}_{\\mathrm{pow}}\\!\\left(\\tfrac{1}{1+\\kappa}\\right)``, ``\\forall\\, i \\in \\mathcal{I}``. The keys are `crlvar_pcone_a_` and `crldar_pcone_a_`.
  - `keys.pcone_b`: ``s_c \\left(\\tfrac{\\omega_i}{1-\\kappa},\\, \\tfrac{\\theta_i}{\\kappa},\\, -\\tfrac{z}{2\\kappa}\\right) \\in \\mathcal{K}_{\\mathrm{pow}}(1-\\kappa)``, ``\\forall\\, i \\in \\mathcal{I}``. The keys are `crlvar_pcone_b_` and `crldar_pcone_b_`.
  - `keys.exceedance`: ``s_c \\left(\\epsilon_i + \\omega_i - s_i - t\\right) \\leq 0``, ``\\forall\\, i \\in \\mathcal{I}``. The keys are `crlvar_` and `crldar_`.

Each key is a Model State key. A caller reads the entry back under `prefix`, the key and `i`, for example `rlvar_risk_1`.

Where:

  - $(math_dict[:sc_scale])
  - ``s_i``: Entry ``i`` of `series`. It carries the sign of a return, so ``-s_i`` is the loss at observation ``i``.
  - $(math_dict[:I_pos_obs])
  - $(math_dict[:p_i_obs])
  - $(math_dict[:rlvar_t])
  - $(math_dict[:rlvar_z_ge])
  - $(math_dict[:rlvar_aux])
  - $(math_dict[:K_pow])
  - $(math_dict[:ln_kappa])
  - $(math_dict[:alpha_rm])
  - $(math_dict[:kappa_rm])
  - $(math_dict[:T])

## Relaxation

$(val_dict[:relax])

  - `keys.risk` lies at or above the RLVaR of `series`, because the RLVaR is the minimum of the expression over the variables. It is tight when the objective minimises the risk.
  - A bound on `keys.risk` is exact, because the weights satisfy it when the RLVaR of `series` is at most the bound.

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

# Validation

  - The observation weights pass [`checked_observation_weights`](@ref) against `pr.X`. Otherwise that function raises its error.

# Returns

  - `risk`: The relativistic risk expression added to the model.

# Related

  - [`RelativisticValueatRisk`](@ref)
  - [`RelativisticDrawdownatRisk`](@ref)
  - [`risk_series`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
  - [`kappa_log`](@ref)
  - [`checked_observation_weights`](@ref)

# References

  - $(ref_dict[:rlvar]) Section 4.1, Equation 15, and Section 4.4, Equation 25.
"""
function set_relativistic_risk_constraints!(model::JuMP.Model, i::Any, r::RiskMeasure,
                                            opt::RiskConstraintOwner,
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
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = checked_observation_weights(wi, pr.X)
    rows = positive_weight_rows(wi, T)
    N = length(rows)
    t, z, omega, psi, theta, epsilon = JuMP.@variables(model, begin
                                                           ()
                                                           (), (lower_bound = 0)
                                                           [1:N]
                                                           [1:N]
                                                           [1:N]
                                                           [1:N]
                                                       end)
    state_set!(model, prefix, keys.t, i, t)
    state_set!(model, prefix, keys.z, i, z)
    state_set!(model, prefix, keys.omega, i, omega)
    state_set!(model, prefix, keys.psi, i, psi)
    state_set!(model, prefix, keys.theta, i, theta)
    state_set!(model, prefix, keys.epsilon, i, epsilon)
    lnk = kappa_log(inv(alpha * T), kappa)
    risk = if isnothing(wi)
        JuMP.@expression(model, t + lnk * z + sum(psi + theta))
    else
        wp = view(wi, rows) / sum(wi)
        JuMP.@expression(model, t + lnk * z + T * LinearAlgebra.dot(wp, psi + theta))
    end
    state_set!(model, prefix, keys.risk, i, risk)
    sr = view(series, rows)
    pcone_a, pcone_b, exceedance = JuMP.@constraints(model,
                                                     begin
                                                         [k = 1:N],
                                                         [sc * z * opk * ik2,
                                                          sc * psi[k] * opk * ik,
                                                          sc * epsilon[k]] in
                                                         JuMP.MOI.PowerCone(iopk)
                                                         [k = 1:N],
                                                         [sc * omega[k] * iomk,
                                                          sc * theta[k] * ik,
                                                          -sc * z * ik2] in
                                                         JuMP.MOI.PowerCone(omk)
                                                         sc *
                                                         ((epsilon + omega - sr) .- t) <= 0
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

Adds the Relativistic Value-at-Risk range of the net portfolio returns to `model`, and returns its risk expression.

[`range_tails`](@ref) pairs `alpha` with `kappa_a` for the loss tail, and `beta` with `kappa_b` for the gain tail. The RLVaR method of this function builds a pair of power cones for each tail. The tail keys carry `loss_` or `gain_` before `i`, for example `rlvar_risk_loss_1`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{RVaRRange}_{\\alpha,\\kappa_a,\\beta,\\kappa_b}(\\boldsymbol{w}) &= \\mathrm{RLVaR}_{\\alpha,\\kappa_a}(\\hat{\\boldsymbol{r}}) + \\mathrm{RLVaR}_{\\beta,\\kappa_b}(-\\hat{\\boldsymbol{r}})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RVaRRange}_{\\alpha,\\kappa_a,\\beta,\\kappa_b}(\\boldsymbol{w})``: Relativistic VaR range of the portfolio.
  - ``\\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{s})``: The programme of the RLVaR method of this function, with the series ``\\boldsymbol{s}`` in place of the net portfolio returns.
  - $(math_dict[:rhat_net_vec])
  - $(math_dict[:rhat_t_net])
  - ``\\alpha``, ``\\kappa_a``: Significance level and Kaniadakis deformation parameter of the loss tail.
  - ``\\beta``, ``\\kappa_b``: Significance level and Kaniadakis deformation parameter of the gain tail.

Section 7.2.3.5 of [cajas2025](@cite) defines this range. [`RelativisticValueatRiskRange`](@ref) states its sources.

# Algorithm

 1. Build the two tails and their sum with [`set_range_risk_constraints!`](@ref), under the key `rlvar_range_risk_`, giving `range_risk`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::RelativisticValueatRiskRange`: The RLVaR range risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `range_risk`: The combined `loss + gain` risk expression added to the model, registered under the key `rlvar_range_risk_`.

# Related

  - [`RelativisticValueatRiskRange`](@ref)
  - [`range_tails`](@ref)
  - [`set_range_risk_constraints!`](@ref)

# References

  - $(ref_dict[:rlvar]) Section 4.1, Equations 12 and 15.
  - $(ref_dict[:cajas2025]) Section 7.2.3.5, Equations 7.84 and 7.85. $(ref_dict[:no_original_source])
  - $(ref_dict[:owa1])
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticValueatRiskRange,
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    return set_range_risk_constraints!(model, i, r, :rlvar_range_risk_, opt, pr, args...;
                                       prefix = prefix, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Adds the Relativistic Drawdown-at-Risk of the portfolio to `model`, and returns its risk expression.

[`set_relativistic_risk_constraints!`](@ref) builds the power cones, and this method gives it the negated drawdown path, ``-dd_t``. The drawdown rows hold ``dd_t`` at or above the drawdown, and the RLVaR does not fall when a loss rises. So `rldar_risk_` lies at or above the RLDaR. It is tight when the objective minimises the risk, and a bound on it is exact.

# Mathematical definition

```math
\\begin{align}
c_t &= \\sum_{s=1}^{t} \\hat{r}_s\\,, \\quad c_0 = 0\\,, \\\\
d_t &= c_t - \\max_{0 \\leq s \\leq t} c_s \\leq 0\\,, \\\\
\\mathrm{RLDaR}_{\\alpha,\\kappa}(\\boldsymbol{w}) &= \\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{d})\\,.
\\end{align}
```

Where:

  - ``\\mathrm{RLDaR}_{\\alpha,\\kappa}(\\boldsymbol{w})``: Relativistic Drawdown-at-Risk of the portfolio.
  - ``\\mathrm{RLVaR}_{\\alpha,\\kappa}(\\boldsymbol{d})``: The programme of the RLVaR method of this function, with ``d_t`` in place of the net portfolio return ``\\hat{r}_t``.
  - ``\\boldsymbol{d}``: The drawdown series, ``T \\times 1``, with entries ``d_t``.
  - $(math_dict[:rhat_t_net])
  - $(math_dict[:ct])
  - $(math_dict[:dtdd])

Equation 23 of [rlvar](@cite) defines the RLDaR, and Equation 25 states its programme. Its exceedance row reads the drawdown with the sign of a loss, ``-d_t``. Its drawdown rows are the rows of [`set_drawdown_constraints!`](@ref).

# Algorithm

 1. Read `series`, the negated drawdown path, and `T` with [`risk_series`](@ref). The first call builds the drawdown rows.
 2. Build the programme of `series` with [`set_relativistic_risk_constraints!`](@ref), giving `risk`. Each key names `rldar`, for example `t_rldar_` and `crldar_pcone_a_`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::RelativisticDrawdownatRisk`: The RLDaR risk measure, read for `alpha`, `kappa`, `w` and `settings`.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `risk`: The RLDaR risk expression added to the model, registered under the key `rldar_risk_`.

# Related

  - [`RelativisticDrawdownatRisk`](@ref)
  - [`set_relativistic_risk_constraints!`](@ref)
  - [`risk_series`](@ref)
  - [`set_drawdown_constraints!`](@ref)

# References

  - $(ref_dict[:rlvar]) Section 4.3, Equation 23, and Section 4.4, Equation 25.
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::RelativisticDrawdownatRisk,
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
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
