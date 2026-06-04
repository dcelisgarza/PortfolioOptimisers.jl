"""
$(DocStringExtensions.TYPEDSIGNATURES)

Retrieve or compute and cache the upper Cholesky factor of the prior covariance matrix.

If `model` does not yet contain a `G` expression, the factor is computed from `pr.chol`
(if available) or by Cholesky-factorising `pr.sigma`, then stored as `model[:G]`.

# Arguments

  - $(arg_dict[:model])
  - `pr::AbstractPriorResult`: Prior result containing `sigma` and optionally `chol`.

# Returns

  - `G::Matrix`: Upper Cholesky factor of the prior covariance matrix.

# Related

  - [`chol_sigma_selector`](@ref)
"""
function get_chol_or_sigma_pm(model::JuMP.Model, pr::AbstractPriorResult)
    if !haskey(model, :G)
        G = isnothing(pr.chol) ? LinearAlgebra.cholesky(pr.sigma).U : pr.chol
        JuMP.@expression(model, G, G)
    end
    return model[:G]
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Select the Cholesky factor to use for the covariance matrix.

Returns the factor from the prior (`get_chol_or_sigma_pm`) when `r.sigma` and `r.chol` are
both `nothing`, the Cholesky of `r.sigma` when `r.chol` is `nothing`, or `r.chol` directly.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:pr])
  - `r::CholRM`: Risk measure carrying optional `sigma` and `chol` fields.

# Returns

  - `G::Matrix`: Upper Cholesky factor of the selected covariance matrix.

# Related

  - [`get_chol_or_sigma_pm`](@ref)
"""
function chol_sigma_selector(model::JuMP.Model, pr::AbstractPriorResult, r::CholRM)
    return if isnothing(r.sigma) && isnothing(r.chol)
        get_chol_or_sigma_pm(model, pr)
    elseif isnothing(r.chol)
        LinearAlgebra.cholesky(r.sigma).U
    else
        r.chol
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Set the upper-bound constraint and register the variance risk expression.

Applies [`set_risk_upper_bound!`](@ref) using `r_expr_ub` and `ub`, then registers `r_expr`
via [`set_risk_expression!`](@ref) according to `settings`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:opt_rjumpe])
  - `r_expr_ub::JuMP.AbstractJuMPScalar`: Expression used for the upper-bound check.
  - `ub`: Upper bound value.
  - $(arg_dict[:key_sym])
  - `r_expr::JuMP.AbstractJuMPScalar`: Risk expression added to the objective.
  - `settings::RiskMeasureSettings`: Settings carrying scale and `rke` flag.
  - `flag::Bool`: If true, sets upper bound; if false sets lower bound.

# Returns

  - `nothing`.

# Related

  - [`set_risk_upper_bound!`](@ref)
  - [`set_risk_expression!`](@ref)
"""
function set_variance_risk_bounds_and_expression!(model::JuMP.Model,
                                                  opt::RiskJuMPOptimisationEstimator,
                                                  r_expr_ub::JuMP.AbstractJuMPScalar,
                                                  ub::Option{<:RkRtBounds}, key::Symbol,
                                                  r_expr::JuMP.AbstractJuMPScalar,
                                                  settings::JuMPRiskMeasureSettings,
                                                  flag::Bool = true)
    set_risk_upper_bound!(model, opt, r_expr_ub, ub, key, flag)
    set_risk_expression!(model, r_expr, settings.scale, settings.rke)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Construct the raw standard-deviation or variance risk JuMP variable and second-order cone
constraint.

For `StandardDeviation`, adds a scalar variable `sd_risk_i` and the SOC constraint
`[sc * sd_risk; sc * G * w] in SecondOrderCone()`. For `Variance`, dispatches to the
appropriate variance formulation (SDP, SOC-squared, or quadratic) and also applies any
risk-contribution constraints.

# Mathematical definition

Standard deviation:

```math
\\begin{align}
\\hat{\\sigma}(\\boldsymbol{w}) &= \\|\\mathbf{G}\\boldsymbol{w}\\|_2\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}(\\boldsymbol{w})``: Portfolio standard deviation.
  - ``\\mathbf{G}``: Upper Cholesky factor of the covariance matrix ``\\boldsymbol{\\Sigma}``.
  - $(math_dict[:w_port])

SDP variance:

```math
\\begin{align}
\\hat{\\sigma}^2(\\boldsymbol{w}) &= \\mathrm{tr}(\\boldsymbol{\\Sigma}\\mathbf{W})\\,.
\\end{align}
```

Where:

  - ``\\hat{\\sigma}^2(\\boldsymbol{w})``: Portfolio variance (SDP formulation).
  - ``\\mathbf{W} = \\boldsymbol{w}\\boldsymbol{w}^\\intercal``: Outer product of portfolio weights.
  - ``\\boldsymbol{\\Sigma}``: Covariance matrix.
  - ``\\mathrm{tr}(\\cdot)``: Matrix trace operator.

where ``\\mathbf{G}`` is the upper Cholesky factor of ``\\boldsymbol{\\Sigma}`` and ``\\mathbf{W}`` is the SDP matrix variable.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r`: Risk measure instance (`StandardDeviation` or `Variance`).
  - $(arg_dict[:opt_jumpe])
  - $(arg_dict[:pr])
  - $(arg_dict[:pl_opt])

# Returns

  - A 2-tuple `(risk_expr, key)` of the JuMP risk expression and its model key.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`set_variance_risk!`](@ref)
"""
function set_risk!(model::JuMP.Model, i::Any, r::StandardDeviation,
                   opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...;
                   kwargs...)
    key = Symbol(:sd_risk_, i)
    sc = model[:sc]
    w = model[:w]
    G = chol_sigma_selector(model, pr, r)
    sd_risk = model[key] = JuMP.@variable(model)
    model[Symbol(:csd_risk_soc_, i)] = JuMP.@constraint(model,
                                                        [sc * sd_risk; sc * G * w] in
                                                        JuMP.SecondOrderCone())
    return sd_risk, key
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add standard-deviation, variance, or uncertainty-set variance risk constraints to `model`.

Each method builds the appropriate JuMP variables and constraints and then calls
[`set_risk_bounds_and_expression!`](@ref) or [`set_variance_risk_bounds_and_expression!`](@ref).
The `Variance` / `NonFRCJuMPOpt` overload automatically chooses between SDP and SOC/quadratic
formulations based on risk-contribution and phylogeny settings.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - $(arg_dict[:r_risk])
  - $(arg_dict[:opt_jumpe])
  - $(arg_dict[:pr])
  - $(arg_dict[:pl_opt])
  - $(arg_dict[:fees_opt])

# Returns

  - `nothing`.

# Related

  - [`set_risk!`](@ref)
  - [`set_variance_risk!`](@ref)
  - [`set_ucs_variance_risk!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::StandardDeviation,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; kwargs...)
    sd_risk, key = set_risk!(model, i, r, opt, pr, args...; kwargs...)
    set_risk_bounds_and_expression!(model, opt, sd_risk, r.settings, key)
    return sd_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether risk-contribution constraints require the SDP variance formulation.

Returns `false` for `Nothing` (no risk-contribution constraints) and `true` for
`LinearConstraint` (risk-contribution constraints are present).

# Arguments

  - $(arg_dict[:model])
  - `opt::NonFRCJuMPOpt`: Optimisation estimator.
  - `rc`: Risk-contribution constraint (`nothing` or `LinearConstraint`).

# Returns

  - `flag::Bool`: Whether risk-contribution constraints require the SDP formulation.

# Related

  - [`sdp_variance_flag!`](@ref)
"""
function sdp_rc_variance_flag!(::JuMP.Model, ::NonFRCJuMPOpt, ::Nothing)
    return false
end
function sdp_rc_variance_flag!(::JuMP.Model, ::NonFRCJuMPOpt, ::LinearConstraint)
    return true
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the [`FrontierBoundEstimator`](@ref) that selects the appropriate variance formulation.

Returns [`LinearBound`](@ref) (SDP formulation) when any of the following hold: `rc_flag` is `true`, `model` already contains a `rc_variance` expression, or `pl` contains a [`SemiDefinitePhylogeny`](@ref) constraint. Returns [`SquareRootBound`](@ref) (SOC formulation) otherwise.

# Arguments

  - $(arg_dict[:model])
  - `rc_flag::Bool`: Whether risk-contribution constraints require the SDP formulation.
  - `pl`: Optional phylogeny constraint(s).

# Returns

  - `bound::FrontierBoundEstimator`: [`LinearBound`](@ref) for SDP; [`SquareRootBound`](@ref) for SOC.

# Related

  - [`sdp_rc_variance_flag!`](@ref)
  - [`set_variance_risk!`](@ref)
  - [`FrontierBoundEstimator`](@ref)
  - [`LinearBound`](@ref)
  - [`SquareRootBound`](@ref)
"""
function sdp_variance_flag!(model::JuMP.Model, rc_flag::Bool, pl::Option{<:PlC_VecPlC})
    return if rc_flag ||
              haskey(model, :rc_variance) ||
              isa(pl, SemiDefinitePhylogeny) ||
              isa(pl, AbstractVector) && any(x -> isa(x, SemiDefinitePhylogeny), pl)
        LinearBound()
    else
        SquareRootBound()
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the variance risk JuMP expression according to the selected formulation.

The flag-dispatching overload routes to either [`set_sdp_variance_risk!`](@ref) (SDP) or the
appropriate SOC/quadratic overload. The `SquaredSOCRiskExpr` overload encodes
variance as the square of an SOC variable. The `QuadRiskExpr` overload encodes variance
directly as ``\\boldsymbol{w}^\\intercal \\Sigma \\boldsymbol{w}``.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::Variance`: Variance risk measure.
  - $(arg_dict[:pr_sigma])
  - `flag`:
      + `::LinearBound`: Use the SDP formulation.
      + `::SquareRootBound`: Use the SOC formulation.
  - $(arg_dict[:key_sym])

# Returns

  - The variance risk JuMP expression.

# Related

  - [`set_sdp_variance_risk!`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_variance_risk!(model::JuMP.Model, i::Any, r::Variance, pr::AbstractPriorResult,
                            ::LinearBound, key::Symbol)
    return set_sdp_variance_risk!(model, i, r, pr, key)
end
function set_variance_risk!(model::JuMP.Model, i::Any, r::Variance, pr::AbstractPriorResult,
                            ::SquareRootBound, key::Symbol)
    return set_variance_risk!(model, i, r, pr, key)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the SDP variance risk expression using the semidefinite matrix `W`.

Computes `sigma_W = sigma * W` and stores `tr(sigma_W)` as the variance risk expression
under `key`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::Variance`: Variance risk measure.
  - $(arg_dict[:pr_sigma])
  - $(arg_dict[:key_sym])

# Returns

  - The variance risk JuMP expression.

# Related

  - [`set_variance_risk!`](@ref)
"""
function set_sdp_variance_risk!(model::JuMP.Model, i::Any, r::Variance,
                                pr::AbstractPriorResult, key::Symbol)
    W = set_sdp_constraints!(model)
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    sigma_W = model[Symbol(:sigma_W_, i)] = JuMP.@expression(model, sigma * W)
    return model[key] = JuMP.@expression(model, LinearAlgebra.tr(sigma_W))
end
function set_variance_risk!(model::JuMP.Model, i::Any,
                            r::Variance{<:Any, <:Any, <:Any, <:Any, <:SquaredSOCRiskExpr},
                            pr::AbstractPriorResult, key::Symbol)
    sc = model[:sc]
    w = model[:w]
    G = chol_sigma_selector(model, pr, r)
    key_dev = Symbol(:dev_, i)
    dev = model[key_dev] = JuMP.@variable(model)
    model[Symbol(key_dev, :_soc)] = JuMP.@constraint(model,
                                                     [sc * dev; sc * G * w] in
                                                     JuMP.SecondOrderCone())
    return model[key] = JuMP.@expression(model, dev^2)
end
function set_variance_risk!(model::JuMP.Model, i::Any,
                            r::Variance{<:Any, <:Any, <:Any, <:Any, <:QuadRiskExpr},
                            pr::AbstractPriorResult, key::Symbol)
    sc = model[:sc]
    w = model[:w]
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    G = chol_sigma_selector(model, pr, r)
    dev = model[Symbol(:dev_, i)] = JuMP.@variable(model)
    model[Symbol(:cdev_soc_, i)] = JuMP.@constraint(model,
                                                    [sc * dev; sc * G * w] in
                                                    JuMP.SecondOrderCone())
    return model[key] = JuMP.@expression(model, LinearAlgebra.dot(w, sigma, w))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the JuMP expression and symbol key used for the variance upper-bound check.

When `flag` is `true` (SDP formulation) the variance expression and `:variance_risk_i` key
are returned; otherwise the standard-deviation variable and `:dev_i` key are returned.

# Arguments

  - $(arg_dict[:model])
  - `i`: Constraint index.
  - `flag`:
      + `::LinearBound`: Use the SDP formulation.
      + `::SquareRootBound`: Use the SOC formulation.

# Returns

  - A 2-tuple `(expr, key)` of the bound expression and its model key.

# Related

  - [`variance_risk_bounds_val`](@ref)
"""
function variance_risk_bounds_expr(model::JuMP.Model, i::Any, ::LinearBound)
    key = Symbol(:variance_risk_, i)
    return model[key], key
end
function variance_risk_bounds_expr(model::JuMP.Model, i::Any, ::SquareRootBound)
    key = Symbol(:dev_, i)
    return model[key], key
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Convert a bound value to the appropriate scale for the selected variance formulation.

Dispatches on the [`FrontierBoundEstimator`](@ref) strategy:

  - [`LinearBound`](@ref): passes the bound through unchanged (variance units → variance units).
  - [`SquareRootBound`](@ref): applies `sqrt` to convert from variance to standard-deviation units.
  - [`SquaredBound`](@ref): applies squaring to convert from linear to squared units.

Returns `nothing` when `ub` is `nothing`.

# Arguments

  - `bound::FrontierBoundEstimator`: Bound-transformation strategy.
  - `ub`: Bound value (scalar, vector, [`Frontier`](@ref), or `nothing`).

# Returns

  - The rescaled bound, or `nothing` when `ub` is `nothing`.

# Related

  - [`FrontierBoundEstimator`](@ref)
  - [`LinearBound`](@ref)
  - [`SquareRootBound`](@ref)
  - [`SquaredBound`](@ref)
  - [`variance_risk_bounds_expr`](@ref)
"""
function variance_risk_bounds_val(bound::FrontierBoundEstimator, ub::Frontier)
    return _Frontier(; N = ub.N, factor = 1, bound = bound)
end
function variance_risk_bounds_val(::LinearBound, ub::Num_VecNum)
    return ub
end
function variance_risk_bounds_val(::SquareRootBound, ub::VecNum)
    return sqrt.(ub)
end
function variance_risk_bounds_val(::SquareRootBound, ub::Number)
    return sqrt(ub)
end
function variance_risk_bounds_val(::SquaredBound, ub::VecNum)
    return ub .^ 2
end
function variance_risk_bounds_val(::SquaredBound, ub::Number)
    return ub^2
end
function variance_risk_bounds_val(::Any, ::Nothing)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add linear risk-contribution constraints on the variance decomposition to `model`.

The fall-through method does nothing. The concrete method extracts the diagonal of the
`sigma_W_i` expression and adds inequality and/or equality constraints of the form
`A_ineq * diag(sigma_W) <= B_ineq * variance_risk` and
`A_eq * diag(sigma_W) == B_eq * variance_risk`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `rc::LinearConstraint`: Linear risk-contribution constraint.
  - `variance_risk::JuMP.AbstractJuMPScalar`: Total variance risk expression.

# Returns

  - `nothing`.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`set_sdp_variance_risk!`](@ref)
"""
function rc_variance_constraints!(args...)
    return nothing
end
function rc_variance_constraints!(model::JuMP.Model, i::Any, rc::LinearConstraint,
                                  variance_risk::JuMP.AbstractJuMPScalar)
    sigma_W = model[Symbol(:sigma_W_, i)]
    sc = model[:sc]
    if !haskey(model, :rc_variance)
        JuMP.@expression(model, rc_variance, true)
    end
    rc_key = Symbol(:rc_variance_, i)
    vsw = vec(LinearAlgebra.diag(sigma_W))
    if !isnothing(rc.A_ineq)
        model[Symbol(rc_key, :_ineq)] = JuMP.@constraint(model,
                                                         sc * (rc.A_ineq * vsw -
                                                               rc.B_ineq * variance_risk) <=
                                                         0)
    end
    if !isnothing(rc.A_eq)
        model[Symbol(rc_key, :_eq)] = JuMP.@constraint(model,
                                                       sc * (rc.A_eq * vsw -
                                                             rc.B_eq * variance_risk) == 0)
    end
    return nothing
end
function set_risk!(model::JuMP.Model, i::Any, r::Variance, opt::NonFRCJuMPOpt,
                   pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, args...; kwargs...)
    rc = linear_constraints(r.rc, opt.opt.sets; datatype = eltype(pr.X),
                            strict = opt.opt.strict)
    rc_flag = sdp_rc_variance_flag!(model, opt, rc)
    sdp_flag = sdp_variance_flag!(model, rc_flag, pl)
    key = Symbol(:variance_risk_, i)
    variance_risk = set_variance_risk!(model, i, r, pr, sdp_flag, key)
    rc_variance_constraints!(model, i, rc, variance_risk)
    return variance_risk, sdp_flag
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `Variance` to `model` using non-factor-risk-contribution
optimisers.

Computes the portfolio variance risk expression and registers the upper-bound constraint
and objective contribution according to the variance risk measure settings.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::Variance`: The variance risk measure.
  - `opt::NonFRCJuMPOpt`: The optimisation estimator.
  - $(arg_dict[:pr])
  - $(arg_dict[:pl_opt])

# Returns

  - `nothing`.

# Related

  - [`Variance`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`set_risk!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::Variance, opt::NonFRCJuMPOpt,
                               pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, args...;
                               kwargs...)
    if !haskey(model, :variance_flag)
        JuMP.@expression(model, variance_flag, true)
    end
    variance_risk, sdp_flag = set_risk!(model, i, r, opt, pr, pl, args...; kwargs...)
    var_bound_expr, var_bound_key = variance_risk_bounds_expr(model, i, sdp_flag)
    ub = variance_risk_bounds_val(sdp_flag, r.settings.ub)
    set_variance_risk_bounds_and_expression!(model, opt, var_bound_expr, ub, var_bound_key,
                                             variance_risk, r.settings)
    return variance_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `Variance` to `model` using a `FactorRiskContribution`
optimiser.

Computes factor-based risk contributions for the portfolio variance and registers the
upper-bound constraint and objective contribution accordingly.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::Variance`: The variance risk measure.
  - `opt::FactorRiskContribution`: The factor risk contribution optimisation estimator.
  - $(arg_dict[:pr])
  - `b1::MatNum`: Factor budget matrix used for risk contribution computations.

# Returns

  - `nothing`.

# Related

  - [`Variance`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`set_risk!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::Variance,
                               opt::FactorRiskContribution, pr::AbstractPriorResult, ::Any,
                               ::Any, b1::MatNum, args...; kwargs...)
    if !haskey(model, :variance_flag)
        JuMP.@expression(model, variance_flag, true)
    end
    rc = linear_constraints(r.rc, opt.sets; datatype = eltype(pr.X),
                            strict = opt.opt.strict)
    key = Symbol(:variance_risk_, i)
    set_sdp_frc_constraints!(model)
    W = model[:frc_W]
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    sigma_W = model[Symbol(:sigma_W_, i)] = JuMP.@expression(model,
                                                             transpose(b1) * sigma * b1 * W)
    variance_risk = model[key] = JuMP.@expression(model, LinearAlgebra.tr(sigma_W))
    rc_variance_constraints!(model, i, rc, variance_risk)
    var_bound_expr, var_bound_key = variance_risk_bounds_expr(model, i, LinearBound())
    ub = variance_risk_bounds_val(LinearBound(), r.settings.ub)
    set_variance_risk_bounds_and_expression!(model, opt, var_bound_expr, ub, var_bound_key,
                                             variance_risk, r.settings)
    return variance_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build an uncertainty-set variance risk expression for box or ellipsoidal uncertainty.

The `BoxUncertaintySet` overload introduces symmetric auxiliary matrices `Au` and `Al` and
encodes the worst-case variance as `tr(Au * ub) - tr(Al * lb)`. The
`EllipsoidalUncertaintySet` overload introduces a PSD matrix `E`, the compound matrix `W + E`,
and adds an SOC constraint to bound the ellipsoidal perturbation term.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `ucs`: Uncertainty set instance (`BoxUncertaintySet` or `EllipsoidalUncertaintySet`).
  - `sigma::MatNum`: Covariance matrix (used by `EllipsoidalUncertaintySet`).

# Returns

  - A 2-tuple `(ucs_variance_risk, key)` of the uncertainty-set variance expression and its model key.

# Related

  - [`set_risk_constraints!`](@ref)
"""
function set_ucs_variance_risk!(model::JuMP.Model, i::Any, ucs::BoxUncertaintySet, args...)
    if !haskey(model, :Au)
        sc = model[:sc]
        W = model[:W]
        N = size(W, 1)
        JuMP.@variables(model, begin
                            Au[1:N, 1:N] >= 0, Symmetric
                            Al[1:N, 1:N] >= 0, Symmetric
                        end)
        JuMP.@constraint(model, cbucs_variance, sc * (Au - Al - W) == 0)
    end
    key = Symbol(:bucs_variance_risk_, i)
    Au = model[:Au]
    Al = model[:Al]
    ub = ucs.ub
    lb = ucs.lb
    ucs_variance_risk = model[key] = JuMP.@expression(model,
                                                      LinearAlgebra.tr(Au * ub) -
                                                      LinearAlgebra.tr(Al * lb))
    return ucs_variance_risk, key
end
function set_ucs_variance_risk!(model::JuMP.Model, i::Any, ucs::EllipsoidalUncertaintySet,
                                sigma::MatNum)
    sc = model[:sc]
    if !haskey(model, :E)
        W = model[:W]
        N = size(W, 1)
        JuMP.@variable(model, E[1:N, 1:N], Symmetric)
        JuMP.@expression(model, WpE, W + E)
        JuMP.@constraint(model, ceucs_variance, sc * E in JuMP.PSDCone())
    end
    key = Symbol(:eucs_variance_risk_, i)
    WpE = model[:WpE]
    k = ucs.k
    G = LinearAlgebra.cholesky(ucs.sigma).U
    t_eucs = model[Symbol(:t_eucs, i)] = JuMP.@variable(model)
    x_eucs, ucs_variance_risk = model[Symbol(:x_eucs, i)], model[key] = JuMP.@expressions(model,
                                                                                          begin
                                                                                              G *
                                                                                              vec(WpE)
                                                                                              LinearAlgebra.tr(sigma *
                                                                                                               WpE) +
                                                                                              k *
                                                                                              t_eucs
                                                                                          end)
    model[Symbol(:ge_soc, i)] = JuMP.@constraint(model,
                                                 [sc * t_eucs; sc * x_eucs] in
                                                 JuMP.SecondOrderCone())
    return ucs_variance_risk, key
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add JuMP risk constraints for `UncertaintySetVariance` to `model`.

Computes portfolio variance using an uncertainty set covariance matrix derived from
the prior or the risk measure's own `ucs` field, and registers the upper-bound constraint
and objective contribution.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::UncertaintySetVariance`: The uncertainty set variance risk measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `nothing`.

# Related

  - [`UncertaintySetVariance`](@ref)
  - [`set_risk_constraints!`](@ref)
  - [`set_ucs_variance_risk!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::UncertaintySetVariance,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; rd::ReturnsResult = ReturnsResult(), kwargs...)
    if !haskey(model, :variance_flag)
        JuMP.@expression(model, variance_flag, true)
    end
    set_sdp_constraints!(model)
    ucs = r.ucs
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    ucs_variance_risk, key = set_ucs_variance_risk!(model, i, sigma_ucs(ucs, rd; kwargs...),
                                                    sigma)
    set_risk_bounds_and_expression!(model, opt, ucs_variance_risk, r.settings, key)
    return ucs_variance_risk
end
