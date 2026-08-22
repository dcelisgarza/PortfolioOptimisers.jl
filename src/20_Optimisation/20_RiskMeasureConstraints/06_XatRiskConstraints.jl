"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add Value-at-Risk, Value-at-Risk range, or Drawdown-at-Risk constraints to `model`.

The MIP overloads introduce binary variables `z_var` and add big-M constraints to encode the
empirical quantile. The distribution overloads use closed-form z-scores computed by
[`compute_value_at_risk_z`](@ref) / [`compute_value_at_risk_cz`](@ref) and add an SOC
constraint. The `DrawdownatRisk` overload applies the MIP approach to the drawdown series.

# Mathematical definition

Empirical (MIP) VaR:

```math
\\begin{align}
z_t &\\in \\{0,1\\}, \\quad \\sum_t z_t \\leq \\alpha T, \\quad \\mathrm{VaR} \\geq -\\hat{r}_t - b\\,z_t \\quad \\forall\\, t\\,.
\\end{align}
```

Where:

  - ``z_t \\in \\{0,1\\}``: Binary indicator for tail losses.
  - $(math_dict[:alpha_rm])
  - $(math_dict[:T])
  - ``\\mathrm{VaR}``: Value-at-Risk variable.
  - ``\\hat{r}_t``: Portfolio return at time ``t``.
  - ``b``: Big-M constant.

Parametric VaR (Normal/t/Laplace):

```math
\\begin{align}
\\mathrm{VaR}_\\alpha(\\boldsymbol{w}) &= -\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} + z_\\alpha \\lVert \\mathbf{G}\\boldsymbol{w} \\rVert_2\\,.
\\end{align}
```

Where:

  - ``\\mathrm{VaR}_\\alpha(\\boldsymbol{w})``: Parametric Value-at-Risk.
  - ``\\boldsymbol{\\mu}``: Expected returns vector.
  - $(math_dict[:w_port])
  - ``z_\\alpha``: Distribution quantile at level ``\\alpha``.
  - ``\\mathbf{G}``: Cholesky factor of the covariance matrix.

where ``z_\\alpha`` is the distribution quantile at level ``\\alpha`` and ``\\mathbf{G}`` is the Cholesky factor of the covariance.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - $(arg_dict[:r_risk])
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`compute_value_at_risk_z`](@ref)
  - [`compute_value_at_risk_cz`](@ref)
  - [`risk_series`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::ValueatRisk{<:Any, <:Any, <:Any, <:MIPValueatRisk},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; loss::Bool = true, prefix::Symbol = Symbol(""),
                               kwargs...)
    b = ifelse(!isnothing(r.alg.b), r.alg.b, 1e3)
    s = ifelse(!isnothing(r.alg.s), r.alg.s, 1e-5)
    series, T = risk_series(model, NetReturnsRiskSeries(), pr; loss = loss, prefix = prefix)
    return set_mip_quantile_risk_constraints!(model, i, r, opt, pr, series, T, b, s,
                                              (; risk = :var_risk_, z = :z_var_,
                                               cardinality = :csvar_, exceedance = :cvar_);
                                              prefix = prefix)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Encode the big-M empirical quantile of `series` and register it under the names in `keys`.

This is the shared body of the MIP `ValueatRisk` and `DrawdownatRisk`. The two are one
big-M programme over different series, so [`risk_series`](@ref) chooses the series and this
function writes the indicator block once.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::RiskMeasure`: The quantile risk measure, read for `alpha`, `w` and `settings`.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])
  - `series`: The per-observation return series from [`risk_series`](@ref).
  - `T::Int`: The number of observations.
  - `b::Number`: Big-M constant.
  - `s::Number`: Cardinality slack.
  - `keys::NamedTuple`: Bare Model State entry names, one per entry this builder registers.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `risk`: The quantile risk variable added to the model.

# Details

The block knows nothing of which tail it builds. [`risk_series`](@ref) negates the series
for the gain tail, and this same programme over that series is the gain tail's quantile, so
the binaries and the cardinality constraint are written once.

# Throws

  - `DomainError`: if `b <= s`.

# Related

  - [`risk_series`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_mip_quantile_risk_constraints!(model::JuMP.Model, i::Any, r::RiskMeasure,
                                            opt::RiskJuMPOptimisationEstimator,
                                            pr::AbstractPriorResult, series, T::Int,
                                            b::Number, s::Number, keys::NamedTuple;
                                            prefix::Symbol = Symbol(""))
    @argcheck(b > s, DomainError("b ($b) must be greater than s ($s)"))
    sc = get_constraint_scale(model)
    risk, z = JuMP.@variables(model, begin
                                  ()
                                  [1:T], (binary = true)
                              end)
    state_set!(model, prefix, keys.risk, i, risk)
    state_set!(model, prefix, keys.z, i, z)
    alpha = r.alpha
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    if isnothing(wi)
        state_set!(model, prefix, keys.cardinality, i,
                   JuMP.@constraint(model, sc * (sum(z) - alpha * T + s * T) <= 0))
    else
        sw = sum(wi)
        state_set!(model, prefix, keys.cardinality, i,
                   JuMP.@constraint(model,
                                    sc * (LinearAlgebra.dot(wi, z) - alpha * sw + s * sw) <=
                                    0))
    end
    state_set!(model, prefix, keys.exceedance, i,
               JuMP.@constraint(model, sc * ((series + b * z) .+ risk) >= 0))
    set_risk_bounds_and_expression!(model, opt, risk, r.settings, keys.risk, i;
                                    prefix = prefix)
    return risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `ValueatRiskRange` using a MIP (big-M) formulation to
`model`.

Delegates to [`set_range_risk_constraints!`](@ref), which builds the loss tail at `alpha` on
the net portfolio returns and the gain tail at `beta` on their negation, then sums the two
VaR expressions. Each tail brings its own binary indicator set and big-M block.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:MIPValueatRisk}`: The VaR range risk
    measure with MIP formulation.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `var_range_risk`: The combined `loss + gain` risk expression added to the model.

# Related

  - [`ValueatRiskRange`](@ref)
  - [`MIPValueatRisk`](@ref)
  - [`range_tails`](@ref)
  - [`set_range_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                                   <:MIPValueatRisk},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    return set_range_risk_constraints!(model, i, r, :var_range_risk_, opt, pr, args...;
                                       prefix = prefix, kwargs...)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `ValueatRisk` using a parametric distribution formulation
to `model`.

Uses the closed-form z-score from `compute_value_at_risk_z` and adds a second-order cone
constraint to bound the portfolio standard deviation. The VaR expression is
`-mu'w + z * g_var`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::ValueatRisk{<:Any, <:Any, <:Any, <:DistributionValueatRisk}`: The VaR risk measure
    with distribution-based formulation.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`ValueatRisk`](@ref)
  - [`DistributionValueatRisk`](@ref)
  - [`compute_value_at_risk_z`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::ValueatRisk{<:Any, <:Any, <:Any,
                                              <:DistributionValueatRisk},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; loss::Bool = true, prefix::Symbol = Symbol(""),
                               kwargs...)
    alg = r.alg
    mu = nothing_scalar_array_selector(alg.mu, pr.mu)
    G = chol_sigma_selector(model, pr, r.alg)
    w = get_w(model, prefix)
    sc = get_constraint_scale(model)
    z = if loss
        compute_value_at_risk_z(r.alg.dist, r.alpha)
    else
        compute_value_at_risk_cz(r.alg.dist, r.alpha)
    end
    g_var = state_set!(model, prefix, :g_var_, i, JuMP.@variable(model))
    var_risk = state_set!(model, prefix, :var_risk_, i,
                          JuMP.@expression(model, -LinearAlgebra.dot(mu, w) + z * g_var))
    state_set!(model, prefix, :cvar_soc_, i,
               JuMP.@constraint(model, [sc * g_var; sc * G * w] in JuMP.SecondOrderCone()))
    set_risk_bounds_and_expression!(model, opt, var_risk, r.settings, :var_risk_, i;
                                    prefix = prefix)
    return loss ? var_risk : -var_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `ValueatRiskRange` using a parametric distribution
formulation to `model`.

Uses closed-form z-scores from `compute_value_at_risk_z` and `compute_value_at_risk_cz`
and adds a second-order cone constraint. The range risk expression is the difference
between the lower-tail and upper-tail VaR expressions.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:DistributionValueatRisk}`: The VaR
    range risk measure with distribution-based formulation.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`ValueatRiskRange`](@ref)
  - [`DistributionValueatRisk`](@ref)
  - [`compute_value_at_risk_z`](@ref)
  - [`compute_value_at_risk_cz`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                                   <:DistributionValueatRisk},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    alg = r.alg
    mu = nothing_scalar_array_selector(alg.mu, pr.mu)
    G = chol_sigma_selector(model, pr, r.alg)
    w = get_w(model, prefix)
    sc = get_constraint_scale(model)
    dist = r.alg.dist
    z_l = compute_value_at_risk_z(dist, r.alpha)
    z_h = compute_value_at_risk_cz(dist, r.beta)
    g_var = state_set!(model, prefix, :g_var_range_, i, JuMP.@variable(model))
    var_range_mu = state_set!(model, prefix, :var_range_mu_, i,
                              JuMP.@expression(model, LinearAlgebra.dot(mu, w)))
    var_risk_l, var_risk_h = JuMP.@expressions(model, begin
                                                   -var_range_mu + z_l * g_var
                                                   -var_range_mu + z_h * g_var
                                               end)
    state_set!(model, prefix, :var_risk_l_, i, var_risk_l)
    state_set!(model, prefix, :var_risk_h_, i, var_risk_h)
    var_range_risk = state_set!(model, prefix, :var_range_risk_, i,
                                JuMP.@expression(model, var_risk_l - var_risk_h))
    state_set!(model, prefix, :cvar_range_soc_, i,
               JuMP.@constraints(model,
                                 begin
                                     [sc * g_var; sc * G * w] in JuMP.SecondOrderCone()
                                 end))
    set_risk_bounds_and_expression!(model, opt, var_range_risk, r.settings,
                                    :var_range_risk_, i; prefix = prefix)
    return var_range_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `DrawdownatRisk` to `model`.

Introduces binary variables and big-M constraints applied to the drawdown series to encode
the empirical drawdown quantile at confidence level `r.alpha`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::DrawdownatRisk`: The drawdown-at-risk risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`DrawdownatRisk`](@ref)
  - [`risk_series`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::DrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    b = ifelse(!isnothing(r.b), r.b, 1e3)
    s = ifelse(!isnothing(r.s), r.s, 1e-5)
    series, T = risk_series(model, DrawdownRiskSeries(), pr; prefix = prefix)
    return set_mip_quantile_risk_constraints!(model, i, r, opt, pr, series, T, b, s,
                                              (; risk = :dar_risk_, z = :z_dar_,
                                               cardinality = :csdar_, exceedance = :cdar_);
                                              prefix = prefix)
end
