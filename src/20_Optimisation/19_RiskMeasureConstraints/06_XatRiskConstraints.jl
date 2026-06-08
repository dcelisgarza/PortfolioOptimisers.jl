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
\\mathrm{VaR}_\\alpha(\\boldsymbol{w}) &= -\\boldsymbol{\\mu}^\\intercal \\boldsymbol{w} + z_\\alpha \\|\\mathbf{G}\\boldsymbol{w}\\|_2\\,.
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
  - [`set_drawdown_constraints!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::ValueatRisk{<:Any, <:Any, <:Any, <:MIPValueatRisk},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    b = ifelse(!isnothing(r.alg.b), r.alg.b, 1e3)
    s = ifelse(!isnothing(r.alg.s), r.alg.s, 1e-5)
    @argcheck(b > s)
    key = Symbol(:var_risk_, i)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    var_risk, z_var = model[key], model[Symbol(:z_var_, i)] = JuMP.@variables(model,
                                                                              begin
                                                                                  ()
                                                                                  [1:T],
                                                                                  (binary = true)
                                                                              end)
    alpha = r.alpha
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    if isnothing(wi)
        model[Symbol(:csvar_, i)] = JuMP.@constraint(model,
                                                     sc *
                                                     (sum(z_var) - alpha * T + s * T) <= 0)
    else
        sw = sum(wi)
        model[Symbol(:csvar_, i)] = JuMP.@constraint(model,
                                                     sc * (LinearAlgebra.dot(wi, z_var) -
                                                           alpha * sw + s * sw) <= 0)
    end
    model[Symbol(:cvar_, i)] = JuMP.@constraint(model,
                                                sc * ((net_X + b * z_var) .+ var_risk) >= 0)
    set_risk_bounds_and_expression!(model, opt, var_risk, r.settings, key)
    return var_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `ValueatRiskRange` using a MIP (big-M) formulation to
`model`.

Introduces binary variables and big-M constraints to encode both the lower-tail and
upper-tail empirical quantiles. The range risk expression is the difference between the
two VaR expressions.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any, <:MIPValueatRisk}`: The VaR range risk
    measure with MIP formulation.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`ValueatRiskRange`](@ref)
  - [`MIPValueatRisk`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any,
                               r::ValueatRiskRange{<:Any, <:Any, <:Any, <:Any,
                                                   <:MIPValueatRisk},
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    b = ifelse(!isnothing(r.alg.b), r.alg.b, 1e3)
    s = ifelse(!isnothing(r.alg.s), r.alg.s, 1e-5)
    @argcheck(b > s)
    key = Symbol(:var_range_risk_, i)
    sc = get_constraint_scale(model)
    net_X = set_net_portfolio_returns!(model, pr.X)
    T = length(net_X)
    var_risk_l, z_var_l, var_risk_h, z_var_h = model[Symbol(:var_risk_l_, i)], model[Symbol(:z_var_l_, i)], model[Symbol(:var_risk_h_, i)], model[Symbol(:z_var_h_, i)] = JuMP.@variables(model,
                                                                                                                                                                                          begin
                                                                                                                                                                                              ()
                                                                                                                                                                                              [1:T],
                                                                                                                                                                                              (binary = true)
                                                                                                                                                                                              ()
                                                                                                                                                                                              [1:T],
                                                                                                                                                                                              (binary = true)
                                                                                                                                                                                          end)
    alpha = r.alpha
    beta = r.beta
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, net_X)
    if isnothing(wi)
        model[Symbol(:csvar_l_, i)], model[Symbol(:csvar_h_, i)] = JuMP.@constraints(model,
                                                                                     begin
                                                                                         sc *
                                                                                         (sum(z_var_l) -
                                                                                          alpha *
                                                                                          T +
                                                                                          s *
                                                                                          T) <=
                                                                                         0
                                                                                         sc *
                                                                                         (sum(z_var_h) -
                                                                                          beta *
                                                                                          T +
                                                                                          s *
                                                                                          T) <=
                                                                                         0
                                                                                     end)
    else
        sw = sum(wi)
        model[Symbol(:csvar_l_, i)], model[Symbol(:csvar_h_, i)] = JuMP.@constraints(model,
                                                                                     begin
                                                                                         sc *
                                                                                         (LinearAlgebra.dot(wi,
                                                                                                            z_var_l) -
                                                                                          alpha *
                                                                                          sw +
                                                                                          s *
                                                                                          sw) <=
                                                                                         0
                                                                                         sc *
                                                                                         (LinearAlgebra.dot(wi,
                                                                                                            z_var_h) -
                                                                                          beta *
                                                                                          sw +
                                                                                          s *
                                                                                          sw) <=
                                                                                         0
                                                                                     end)
    end
    model[Symbol(:cvar_, i)] = JuMP.@constraints(model,
                                                 begin
                                                     sc *
                                                     ((net_X + b * z_var_l) .+ var_risk_l) >=
                                                     0
                                                     sc *
                                                     ((net_X + b * z_var_h) .+ var_risk_h) <=
                                                     0
                                                 end)
    var_range_risk = model[key] = JuMP.@expression(model, var_risk_l - var_risk_h)
    set_risk_bounds_and_expression!(model, opt, var_range_risk, r.settings, key)
    return var_range_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the lower-tail z-score for a parametric VaR at significance level `alpha`.

Returns the complementary quantile for Normal and scaled Student-t distributions, and the
closed-form expression for the Laplace distribution.

# Arguments

  - `dist`: Distribution instance (Normal, TDist, or Laplace).
  - `alpha::Number`: Significance level.

# Returns

  - `z::Number`: Lower-tail z-score for the parametric VaR.

# Related

  - [`compute_value_at_risk_cz`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function compute_value_at_risk_z(dist::Distributions.Normal, alpha::Number)
    return Distributions.cquantile(dist, alpha)
end
function compute_value_at_risk_z(dist::Distributions.TDist, alpha::Number)
    d = StatsAPI.dof(dist)
    @argcheck(d > 2)
    return Distributions.cquantile(dist, alpha) * sqrt((d - 2) / d)
end
function compute_value_at_risk_z(::Distributions.Laplace, alpha::Number)
    return -log(2 * alpha) / sqrt(2)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Compute the upper-tail z-score for a parametric VaR at significance level `alpha`.

Used for the high (upper) bound in VaR range constraints. Returns the lower quantile for
Normal and scaled Student-t distributions, and the closed-form expression for Laplace.

# Arguments

  - `dist`: Distribution instance (Normal, TDist, or Laplace).
  - `alpha::Number`: Significance level.

# Returns

  - `z::Number`: Upper-tail z-score for the parametric VaR.

# Related

  - [`compute_value_at_risk_z`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function compute_value_at_risk_cz(dist::Distributions.Normal, alpha::Number)
    return Statistics.quantile(dist, alpha)
end
function compute_value_at_risk_cz(dist::Distributions.TDist, alpha::Number)
    d = StatsAPI.dof(dist)
    @argcheck(d > 2)
    return Statistics.quantile(dist, alpha) * sqrt((d - 2) / d)
end
function compute_value_at_risk_cz(::Distributions.Laplace, alpha::Number)
    return -log(2 * (one(alpha) - alpha)) / sqrt(2)
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
                               args...; kwargs...)
    alg = r.alg
    mu = nothing_scalar_array_selector(alg.mu, pr.mu)
    G = chol_sigma_selector(model, pr, r.alg)
    w = get_w(model)
    sc = get_constraint_scale(model)
    z = compute_value_at_risk_z(r.alg.dist, r.alpha)
    key = Symbol(:var_risk_, i)
    g_var = model[Symbol(:g_var_, i)] = JuMP.@variable(model)
    var_risk = model[key] = JuMP.@expression(model, -LinearAlgebra.dot(mu, w) + z * g_var)
    model[Symbol(:cvar_soc_, i)] = JuMP.@constraint(model,
                                                    [sc * g_var; sc * G * w] in
                                                    JuMP.SecondOrderCone())
    set_risk_bounds_and_expression!(model, opt, var_risk, r.settings, key)
    return var_risk
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
                               args...; kwargs...)
    alg = r.alg
    mu = nothing_scalar_array_selector(alg.mu, pr.mu)
    G = chol_sigma_selector(model, pr, r.alg)
    w = get_w(model)
    sc = get_constraint_scale(model)
    dist = r.alg.dist
    z_l = compute_value_at_risk_z(dist, r.alpha)
    z_h = compute_value_at_risk_cz(dist, r.beta)
    key = Symbol(:var_range_risk_, i)
    g_var = model[Symbol(:g_var_range_, i)] = JuMP.@variable(model)
    var_range_mu = model[Symbol(:var_range_mu_, i)] = JuMP.@expression(model,
                                                                       LinearAlgebra.dot(mu,
                                                                                         w))
    var_risk_l, var_risk_h = model[Symbol(:var_risk_l_, i)], model[Symbol(:var_risk_h_, i)] = JuMP.@expressions(model,
                                                                                                                begin
                                                                                                                    -var_range_mu +
                                                                                                                    z_l *
                                                                                                                    g_var
                                                                                                                    -var_range_mu +
                                                                                                                    z_h *
                                                                                                                    g_var
                                                                                                                end)
    var_range_risk = model[key] = JuMP.@expression(model, var_risk_l - var_risk_h)
    model[Symbol(:cvar_range_soc_, i)] = JuMP.@constraints(model,
                                                           begin
                                                               [sc * g_var; sc * G * w] in
                                                               JuMP.SecondOrderCone()
                                                           end)
    set_risk_bounds_and_expression!(model, opt, var_range_risk, r.settings, key)
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
  - [`set_drawdown_constraints!`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::DrawdownatRisk,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    b = ifelse(!isnothing(r.b), r.b, 1e3)
    s = ifelse(!isnothing(r.s), r.s, 1e-5)
    @argcheck(b > s)
    key = Symbol(:dar_risk_, i)
    sc = get_constraint_scale(model)
    dd = set_drawdown_constraints!(model, pr.X)
    T = length(dd) - 1
    dar_risk, z_dar = model[key], model[Symbol(:z_dar_, i)] = JuMP.@variables(model,
                                                                              begin
                                                                                  ()
                                                                                  [1:T],
                                                                                  (binary = true)
                                                                              end)
    alpha = r.alpha
    wi = nothing_scalar_array_selector(r.w, pr.w)
    wi = get_observation_weights(wi, pr.X)
    if isnothing(wi)
        model[Symbol(:csdar_, i)] = JuMP.@constraint(model,
                                                     sc *
                                                     (sum(z_dar) - alpha * T + s * T) <= 0)
    else
        sw = sum(wi)
        model[Symbol(:csdar_, i)] = JuMP.@constraint(model,
                                                     sc * (LinearAlgebra.dot(wi, z_dar) -
                                                           alpha * sw + s * sw) <= 0)
    end
    model[Symbol(:cdar_, i)] = JuMP.@constraint(model,
                                                sc *
                                                ((-view(dd, 2:(T + 1)) + b * z_dar) .+
                                                 dar_risk) >= 0)
    set_risk_bounds_and_expression!(model, opt, dar_risk, r.settings, key)
    return dar_risk
end
