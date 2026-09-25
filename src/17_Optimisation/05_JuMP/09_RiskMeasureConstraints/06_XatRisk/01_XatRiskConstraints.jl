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
  - ``b``: Big-M constant, from [`mip_big_m`](@ref).

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
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               loss::Bool = true, prefix::Symbol = Symbol(""), kwargs...)
    b, s = mip_var_bounds(r.alg.b, r.alg.s)
    b = mip_big_m(model, b, s, NetReturnsRiskSeries(), pr; prefix = prefix)
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
  - `b::Number`: Big-M constant, from [`mip_big_m`](@ref).
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

# Related

  - [`risk_series`](@ref)
  - [`mip_big_m`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_mip_quantile_risk_constraints!(model::JuMP.Model, i::Any, r::RiskMeasure,
                                            opt::RiskConstraintOwner,
                                            pr::AbstractPriorResult, series, T::Int,
                                            b::Number, s::Number, keys::NamedTuple;
                                            prefix::Symbol = Symbol(""))
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
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
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
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               loss::Bool = true, prefix::Symbol = Symbol(""), kwargs...)
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
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
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
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    b, s = mip_var_bounds(r.b, r.s)
    b = mip_big_m(model, b, s, DrawdownRiskSeries(), pr; prefix = prefix)
    series, T = risk_series(model, DrawdownRiskSeries(), pr; prefix = prefix)
    return set_mip_quantile_risk_constraints!(model, i, r, opt, pr, series, T, b, s,
                                              (; risk = :dar_risk_, z = :z_dar_,
                                               cardinality = :csdar_, exceedance = :cdar_);
                                              prefix = prefix)
end
"""
    mip_series_spread(alg::AbstractRiskSeriesAlgorithm, X::MatNum) -> Number

Return the largest spread of the losses of one asset over the observations, per unit of weight.

For the net returns it is the range of a column of `X`. For the drawdowns it is the range of the cumulative sum of a column, over a path that starts at zero. Multiplied by a bound on the gross exposure of the weights, it bounds the spread of the losses of the portfolio, as [`mip_big_m`](@ref) states.

# Arguments

  - `alg::AbstractRiskSeriesAlgorithm`: [`NetReturnsRiskSeries`](@ref) or [`DrawdownRiskSeries`](@ref).
  - `X::MatNum`: Asset returns matrix, observations by assets.

# Returns

  - `d::Number`: The largest spread over the assets.

# Related

  - [`mip_big_m`](@ref)
"""
function mip_series_spread(::NetReturnsRiskSeries, X::MatNum)
    return maximum(x -> maximum(x) - minimum(x), eachcol(X))
end
function mip_series_spread(::DrawdownRiskSeries, X::MatNum)
    return maximum(eachcol(X)) do x
        c = cumsum(x)
        return max(zero(eltype(c)), maximum(c)) - min(zero(eltype(c)), minimum(c))
    end
end
"""
    mip_fees_keep_spread(model::JuMP.Model, alg::AbstractRiskSeriesAlgorithm) -> Bool

Return `true` when the fees of `model` do not change the spread of the losses that [`mip_series_spread`](@ref) bounds.

A per period fee is the same at every observation, so it leaves the spread of the net returns unchanged. So does a one-time fee on an [`AmortisedFees`](@ref) clock. A one-time fee on the first observation changes the loss of that observation alone, which changes the spread. A drawdown adds up the fees of its periods, so any fee changes its spread.

# Arguments

  - $(arg_dict[:model])
  - `alg::AbstractRiskSeriesAlgorithm`: [`NetReturnsRiskSeries`](@ref) or [`DrawdownRiskSeries`](@ref).

# Returns

  - `flag::Bool`: `true` when the spread holds with the fees.

# Related

  - [`mip_big_m`](@ref)
  - [`set_net_portfolio_returns!`](@ref)
"""
function mip_fees_keep_spread(model::JuMP.Model, ::NetReturnsRiskSeries)
    return !shared_has(model, :one_time_fees) ||
           isa(shared_get(model, :fee_fa), AmortisedFees)
end
function mip_fees_keep_spread(model::JuMP.Model, ::DrawdownRiskSeries)
    return !shared_has(model, :fees) && !shared_has(model, :one_time_fees)
end
"""
    mip_big_m(model::JuMP.Model, b::Option{<:Number}, s::Number,
              alg::AbstractRiskSeriesAlgorithm, pr::AbstractPriorResult;
              prefix::Symbol = Symbol("")) -> Number

Return the big-M constant of the empirical quantile programme of [`MIPValueatRisk`](@ref).

A stated `b` is returned as it is, after the check that `b > s`. A `nothing` takes the smallest constant that keeps the programme exact for every weight vector that the model admits. Each exceedance row then relaxes its bound by no more than it must, so the integrality tolerance ``\\varepsilon`` of the solver loosens a row by ``b \\varepsilon`` at most.

# Mathematical definition

The programme is exact when ``b`` is at least the largest loss minus the smallest loss, because the risk is never below the smallest loss. For the net returns,

```math
\\begin{align}
\\ell_{t} - \\ell_{t'} &= \\left(\\boldsymbol{X}_{t'} - \\boldsymbol{X}_{t}\\right) \\boldsymbol{w} \\leq \\lVert \\boldsymbol{w} \\rVert_{1} \\max_{i} \\left(\\max_{t} X_{t,\\,i} - \\min_{t} X_{t,\\,i}\\right) \\leq g\\, d\\,.
\\end{align}
```

For the drawdowns, ``0 \\leq \\ell_{t} = \\max_{0 \\leq s \\leq t} c_{s} - c_{t}``, and the same bound holds with the ranges of the cumulative sums ``C_{t,\\,i} = \\sum_{u=1}^{t} X_{u,\\,i}``, ``C_{0,\\,i} = 0``, in place of the ranges of the columns. So

```math
\\begin{align}
b &= g\\, d\\,.
\\end{align}
```

Where:

  - ``\\ell_{t}``: Loss of observation ``t``.
  - ``\\boldsymbol{X}_{t}``: Row ``t`` of the asset returns matrix.
  - $(math_dict[:w_port])
  - $(math_dict[:ct])
  - ``g``: Bound on the gross exposure, the model's `:w_gross_ub`, from [`gross_exposure_bound`](@ref).
  - ``d``: The largest spread per unit of weight, from [`mip_series_spread`](@ref).
  - ``b``: Big-M constant.

The derivation needs the weights in units of the budget, so it holds only when all of these are true:

  - The model's `k` is the number `1`. Under [`MaximumRatio`](@ref) and in [`RiskBudgeting`](@ref) the weights have a free scale.
  - The series reads the head's weights, not weights that a tracking build shifts.
  - The weight builder recorded a finite ``g``.
  - The fees keep the spread, as [`mip_fees_keep_spread`](@ref) states.
  - ``g\\, d`` is finite, which a `NaN` in `X` makes false.

When one is false, the constant is `1000`, the value that Cajas recommends.

# Arguments

  - $(arg_dict[:model])
  - `b::Option{<:Number}`: The stated big-M constant, or `nothing`.
  - `s::Number`: Cardinality slack.
  - `alg::AbstractRiskSeriesAlgorithm`: The loss series, [`NetReturnsRiskSeries`](@ref) or [`DrawdownRiskSeries`](@ref).
  - $(arg_dict[:pr_X])

# Keyword arguments

  - `prefix::Symbol`: Model State namespace (default: empty, i.e. the bare key).

# Returns

  - `b::Number`: The big-M constant.

# Throws

  - `DomainError`: if a stated `b` is not greater than `s`.

# Related

  - [`MIPValueatRisk`](@ref)
  - [`DrawdownatRisk`](@ref)
  - [`mip_var_bounds`](@ref)
  - [`set_mip_quantile_risk_constraints!`](@ref)

# References

  - $(ref_dict[:cajas2025]) Section 7.2.2.3, Equation 7.51.
"""
function mip_big_m(::JuMP.Model, b::Number, s::Number, args...; kwargs...)
    @argcheck(b > s,
              DomainError((b, s),
                          "`b` is $b and `s` is $s. The big-M constant `b` relaxes a bound the slack `s` tightens, so `b > s` must hold."))
    return b
end
function mip_big_m(model::JuMP.Model, ::Nothing, ::Number, alg::AbstractRiskSeriesAlgorithm,
                   pr::AbstractPriorResult; prefix::Symbol = Symbol(""))
    # Every check reads the model without side effects, so all of them can run.
    exact = all((isa(get_k(model), Number), weights_prefix(model, prefix) == Symbol(""),
                 shared_has(model, :w_gross_ub), mip_fees_keep_spread(model, alg)))
    b = exact ? shared_get(model, :w_gross_ub) * mip_series_spread(alg, pr.X) : Inf
    return isfinite(b) ? b : 1e3
end
