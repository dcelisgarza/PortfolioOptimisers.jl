"""
$(DocStringExtensions.TYPEDSIGNATURES)

Retrieve or compute and cache the upper Cholesky factor of the prior covariance matrix.

If `model` does not yet contain a `G` expression, the factor is computed from `pr.chol`
(if available) or by Cholesky-factorising `pr.sigma`, then stored as the `:G` Model State entry.

# Arguments

  - $(arg_dict[:model])
  - `pr::AbstractPriorResult`: Prior result containing `sigma` and optionally `chol`.

# Returns

  - `G::Matrix`: Upper Cholesky factor of the prior covariance matrix.

# Related

  - [`chol_sigma_selector`](@ref)
"""
function get_chol_or_sigma_pm(model::JuMP.Model, pr::AbstractPriorResult)
    if !shared_has(model, :G)
        G = isnothing(pr.chol) ? LinearAlgebra.cholesky(pr.sigma).U : pr.chol
        JuMP.@expression(model, G, G)
    end
    return shared_get(model, :G)
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
  - `name::Symbol`: Bare Model State entry name seeding the derived bound keys.
  - `i`: Measure index. `name` and `i` are resolved here, so the bound keys cannot drift
    from the key the emitter registered the risk expression under (ADR 0037).
  - `r_expr::JuMP.AbstractJuMPScalar`: Risk expression added to the objective.
  - `settings::RiskMeasureSettings`: Settings carrying scale and `rke` flag.
  - `flag::Bool`: If true, sets upper bound; if false sets lower bound.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace for `name` (default: empty, i.e. the bare key).

# Returns

  - `nothing`.

# Related

  - [`set_risk_upper_bound!`](@ref)
  - [`set_risk_expression!`](@ref)
  - [`state_key`](@ref)
"""
function set_variance_risk_bounds_and_expression!(model::JuMP.Model,
                                                  opt::RiskJuMPOptimisationEstimator,
                                                  r_expr_ub::JuMP.AbstractJuMPScalar,
                                                  ub::Option{<:RkRtBounds}, name::Symbol, i,
                                                  r_expr::JuMP.AbstractJuMPScalar,
                                                  settings::JuMPRiskMeasureSettings,
                                                  flag::Bool = true;
                                                  prefix::Symbol = Symbol(""))
    set_risk_upper_bound!(model, opt, r_expr_ub, ub, state_key(prefix, name, i), flag)
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
\\hat{\\sigma}(\\boldsymbol{w}) &= \\lVert \\mathbf{G}\\boldsymbol{w} \\rVert_2\\,.
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

  - A 2-tuple `(risk_expr, name)` of the JuMP risk expression and the bare Model State
    entry name it was registered under. The caller pairs the name with the same index to
    resolve the key, so no composed key crosses the boundary (ADR 0037).

# Related

  - [`set_risk_constraints!`](@ref)
  - [`set_variance_risk!`](@ref)
"""
function set_risk!(model::JuMP.Model, i::Any, r::StandardDeviation,
                   opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult, args...;
                   prefix::Symbol = Symbol(""), kwargs...)
    sc = get_constraint_scale(model)
    w = get_w(model, prefix)
    G = chol_sigma_selector(model, pr, r)
    sd_risk = state_set!(model, prefix, :sd_risk_, i, JuMP.@variable(model))
    state_set!(model, prefix, :csd_risk_soc_, i,
               JuMP.@constraint(model,
                                [sc * sd_risk; sc * G * w] in JuMP.SecondOrderCone()))
    return sd_risk, :sd_risk_
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
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    sd_risk, name = set_risk!(model, i, r, opt, pr, args...; prefix = prefix, kwargs...)
    set_risk_bounds_and_expression!(model, opt, sd_risk, r.settings, name, i;
                                    prefix = prefix)
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
function sdp_variance_flag!(model::JuMP.Model, rc_flag::Bool, pl::Option{<:PlC_VecPlC};
                            prefix::Symbol = Symbol(""))
    return if rc_flag ||
              state_has(model, prefix, :rc_variance) ||
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
  - $(arg_dict[:ci])

# Returns

  - The variance risk JuMP expression.

# Related

  - [`set_sdp_variance_risk!`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_variance_risk!(model::JuMP.Model, i::Any, r::Variance, pr::AbstractPriorResult,
                            ::LinearBound; prefix::Symbol = Symbol(""))
    return set_sdp_variance_risk!(model, i, r, pr; prefix = prefix)
end
function set_variance_risk!(model::JuMP.Model, i::Any, r::Variance, pr::AbstractPriorResult,
                            ::SquareRootBound; prefix::Symbol = Symbol(""))
    return set_variance_risk!(model, i, r, pr; prefix = prefix)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the SDP variance risk expression using the semidefinite matrix `W`.

Computes `sigma_W = sigma * W` and registers `tr(sigma_W)` as the `:variance_risk_` entry
at index `i`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::Variance`: Variance risk measure.
  - $(arg_dict[:pr_sigma])

# Returns

  - The variance risk JuMP expression.

# Related

  - [`set_variance_risk!`](@ref)
"""
function set_sdp_variance_risk!(model::JuMP.Model, i::Any, r::Variance,
                                pr::AbstractPriorResult; prefix::Symbol = Symbol(""))
    W = set_sdp_constraints!(model; prefix = prefix)
    sigma = nothing_scalar_array_selector(r.sigma, pr.sigma)
    sigma_W = state_set!(model, prefix, :sigma_W_, i, JuMP.@expression(model, sigma * W))
    return state_set!(model, prefix, :variance_risk_, i,
                      JuMP.@expression(model, LinearAlgebra.tr(sigma_W)))
end
function set_variance_risk!(model::JuMP.Model, i::Any,
                            r::Variance{<:Any, <:Any, <:Any, <:Any, <:SquaredSOCRiskExpr},
                            pr::AbstractPriorResult; prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    w = get_w(model, prefix)
    G = chol_sigma_selector(model, pr, r)
    dev = state_set!(model, prefix, :dev_, i, JuMP.@variable(model))
    state_set!(model, prefix, :cdev_soc_, i,
               JuMP.@constraint(model, [sc * dev; sc * G * w] in JuMP.SecondOrderCone()))
    return state_set!(model, prefix, :variance_risk_, i, JuMP.@expression(model, dev^2))
end
function set_variance_risk!(model::JuMP.Model, i::Any,
                            r::Variance{<:Any, <:Any, <:Any, <:Any, <:QuadRiskExpr},
                            pr::AbstractPriorResult; prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    w = get_w(model, prefix)
    sigma = nothing_scalar_array_selector(r.sigma, pr.sigma)
    G = chol_sigma_selector(model, pr, r)
    dev = state_set!(model, prefix, :dev_, i, JuMP.@variable(model))
    state_set!(model, prefix, :cdev_soc_, i,
               JuMP.@constraint(model, [sc * dev; sc * G * w] in JuMP.SecondOrderCone()))
    return state_set!(model, prefix, :variance_risk_, i,
                      JuMP.@expression(model, LinearAlgebra.dot(w, sigma, w)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the JuMP expression and the bare Model State entry name used for the variance
upper-bound check.

When `flag` is `true` (SDP formulation) the variance expression and the `:variance_risk_`
name are returned; otherwise the standard-deviation variable and the `:dev_` name.

# Arguments

  - $(arg_dict[:model])
  - `i`: Constraint index.
  - `flag`:
      + `::LinearBound`: Use the SDP formulation.
      + `::SquareRootBound`: Use the SOC formulation.

# Returns

  - A 2-tuple `(expr, name)` of the bound expression and its bare Model State entry name.

# Related

  - [`variance_risk_bounds_val`](@ref)
"""
function variance_risk_bounds_expr(model::JuMP.Model, i::Any, ::LinearBound;
                                   prefix::Symbol = Symbol(""))
    return state_get(model, prefix, :variance_risk_, i), :variance_risk_
end
function variance_risk_bounds_expr(model::JuMP.Model, i::Any, ::SquareRootBound;
                                   prefix::Symbol = Symbol(""))
    return state_get(model, prefix, :dev_, i), :dev_
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
function rc_variance_constraints!(args...; kwargs...)
    return nothing
end
function rc_variance_constraints!(model::JuMP.Model, i::Any, rc::LinearConstraint,
                                  variance_risk::JuMP.AbstractJuMPScalar;
                                  prefix::Symbol = Symbol(""))
    sigma_W = state_get(model, prefix, :sigma_W_, i)
    sc = get_constraint_scale(model)
    mark_state!(model, prefix, :rc_variance)
    vsw = vec(LinearAlgebra.diag(sigma_W))
    if !isnothing(rc.A_ineq)
        state_set!(model, prefix, :rc_variance_ineq_, i,
                   JuMP.@constraint(model,
                                    sc * (rc.A_ineq * vsw - rc.B_ineq * variance_risk) <= 0))
    end
    if !isnothing(rc.A_eq)
        state_set!(model, prefix, :rc_variance_eq_, i,
                   JuMP.@constraint(model,
                                    sc * (rc.A_eq * vsw - rc.B_eq * variance_risk) == 0))
    end
    return nothing
end
function set_risk!(model::JuMP.Model, i::Any, r::Variance, opt::NonFRCJuMPOpt,
                   pr::AbstractPriorResult, pl::Option{<:PlC_VecPlC}, args...;
                   prefix::Symbol = Symbol(""), kwargs...)
    rc = linear_constraints(r.rc, opt.opt.sets; datatype = eltype(pr.X),
                            strict = opt.opt.strict)
    rc_flag = sdp_rc_variance_flag!(model, opt, rc)
    sdp_flag = sdp_variance_flag!(model, rc_flag, pl; prefix = prefix)
    variance_risk = set_variance_risk!(model, i, r, pr, sdp_flag; prefix = prefix)
    rc_variance_constraints!(model, i, rc, variance_risk; prefix = prefix)
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
                               prefix::Symbol = Symbol(""), kwargs...)
    mark_state!(model, prefix, :variance_flag)
    variance_risk, sdp_flag = set_risk!(model, i, r, opt, pr, pl, args...; prefix = prefix,
                                        kwargs...)
    var_bound_expr, var_bound_name = variance_risk_bounds_expr(model, i, sdp_flag;
                                                               prefix = prefix)
    ub = variance_risk_bounds_val(sdp_flag, r.settings.ub)
    set_variance_risk_bounds_and_expression!(model, opt, var_bound_expr, ub, var_bound_name,
                                             i, variance_risk, r.settings; prefix = prefix)
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
                               ::Any, b1::MatNum, args...; prefix::Symbol = Symbol(""),
                               kwargs...)
    mark_state!(model, prefix, :variance_flag)
    rc = linear_constraints(r.rc, opt.sets; datatype = eltype(pr.X),
                            strict = opt.opt.strict)
    set_sdp_frc_constraints!(model)
    W = shared_get(model, :frc_W)
    sigma = nothing_scalar_array_selector(r.sigma, pr.sigma)
    sigma_W = state_set!(model, prefix, :sigma_W_, i,
                         JuMP.@expression(model, transpose(b1) * sigma * b1 * W))
    variance_risk = state_set!(model, prefix, :variance_risk_, i,
                               JuMP.@expression(model, LinearAlgebra.tr(sigma_W)))
    rc_variance_constraints!(model, i, rc, variance_risk; prefix = prefix)
    var_bound_expr, var_bound_name = variance_risk_bounds_expr(model, i, LinearBound();
                                                               prefix = prefix)
    ub = variance_risk_bounds_val(LinearBound(), r.settings.ub)
    set_variance_risk_bounds_and_expression!(model, opt, var_bound_expr, ub, var_bound_name,
                                             i, variance_risk, r.settings; prefix = prefix)
    return variance_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build an uncertainty-set variance risk expression for box, ellipsoidal, compact or norm-ball uncertainty.

The `BoxUncertaintySet` overload introduces symmetric auxiliary matrices `Au` and `Al` and
encodes the worst-case variance as `tr(Au * ub) - tr(Al * lb)`. The
`EllipsoidalUncertaintySet` overload introduces a PSD matrix `E`, the compound matrix `W + E`,
and adds an SOC constraint to bound the ellipsoidal perturbation term. The
`NormBallUncertaintySet` overload, defined on the covariance tag alone, is the ellipsoid's
lifted form with the set's own map ``\\mathbf{L}^{\\intercal}`` in place of the Cholesky
factor and the cone of the dual norm order in place of the second-order cone, raised by
[`norm_ball_dual_norm_epigraph!`](@ref); it factorises nothing, and a map with no column
raises no cone and leaves `tr(sigma * (W + E))`. These three lift the weights into the
semidefinite matrix `W`, so each calls [`set_sdp_constraints!`](@ref) itself. The
`CompactCovarianceUncertaintySet` overload lifts nothing: it bounds the nominal deviation and
the residual `C .* w - Q * z` with one SOC constraint each, and returns the sum of their
squares. It is the one overload that leaves the programme a second-order cone programme.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `ucs`: Uncertainty set instance (`BoxUncertaintySet`, `EllipsoidalUncertaintySet`, `CompactCovarianceUncertaintySet` or a covariance `NormBallUncertaintySet`).
  - `sigma::MatNum`: Fallback covariance matrix (used by every overload but the box). The set's own `val` field wins over it (ADR 0050). The box overload names no centre at all, so it ignores both.

# Returns

  - A 2-tuple `(ucs_variance_risk, name)` of the uncertainty-set variance expression and its
    bare Model State entry name.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`set_sdp_constraints!`](@ref)
  - [`UncertaintySetVariance`](@ref)
  - [`BoxUncertaintySet`](@ref)
  - [`EllipsoidalUncertaintySet`](@ref)
  - [`CompactCovarianceUncertaintySet`](@ref)
  - [`NormBallUncertaintySet`](@ref)
  - [`norm_ball_dual_norm_epigraph!`](@ref)

# References

  - $(ref_dict[:robustaa])
  - $(ref_dict[:fengpalomar2016])
  - $(ref_dict[:cajas2025]) Section 11.3.
"""
function set_ucs_variance_risk!(model::JuMP.Model, i::Any, ucs::BoxUncertaintySet, args...;
                                prefix::Symbol = Symbol(""))
    set_sdp_constraints!(model; prefix = prefix)
    Au = state_build!(model, prefix, :Au) do
        sc = get_constraint_scale(model)
        W = state_get(model, prefix, :W)
        N = size(W, 1)
        Au = JuMP.@variable(model, [1:N, 1:N], Symmetric, lower_bound = 0)
        Al = state_set!(model, prefix, :Al,
                        JuMP.@variable(model, [1:N, 1:N], Symmetric, lower_bound = 0))
        state_set!(model, prefix, :cbucs_variance,
                   JuMP.@constraint(model, sc * (Au - Al - W) == 0))
        return Au
    end
    Al = state_get(model, prefix, :Al)
    ub = ucs.ub
    lb = ucs.lb
    ucs_variance_risk = state_set!(model, prefix, :bucs_variance_risk_, i,
                                   JuMP.@expression(model,
                                                    LinearAlgebra.tr(Au * ub) -
                                                    LinearAlgebra.tr(Al * lb)))
    return ucs_variance_risk, :bucs_variance_risk_
end
function set_ucs_variance_risk!(model::JuMP.Model, i::Any, ucs::EllipsoidalUncertaintySet,
                                sigma::MatNum; prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    set_sdp_constraints!(model; prefix = prefix)
    state_build!(model, prefix, :E) do
        W = state_get(model, prefix, :W)
        N = size(W, 1)
        E = JuMP.@variable(model, [1:N, 1:N], Symmetric)
        state_set!(model, prefix, :WpE, JuMP.@expression(model, W + E))
        state_set!(model, prefix, :ceucs_variance,
                   JuMP.@constraint(model, sc * E in JuMP.PSDCone()))
        return E
    end
    WpE = state_get(model, prefix, :WpE)
    # The set is a neighbourhood of the covariance it was calibrated on, so it names the
    # centre. The risk measure's field and then the prior are the fallbacks (ADR 0050).
    sigma = something(ucs.val, sigma)
    k = ucs.k
    G = LinearAlgebra.cholesky(ucs.sigma).U
    t_eucs = state_set!(model, prefix, :t_eucs, i, JuMP.@variable(model))
    x_eucs, ucs_variance_risk = JuMP.@expressions(model,
                                                  begin
                                                      G * vec(WpE)
                                                      LinearAlgebra.tr(sigma * WpE) +
                                                      k * t_eucs
                                                  end)
    state_set!(model, prefix, :x_eucs, i, x_eucs)
    state_set!(model, prefix, :eucs_variance_risk_, i, ucs_variance_risk)
    state_set!(model, prefix, :ge_soc, i,
               JuMP.@constraint(model,
                                [sc * t_eucs; sc * x_eucs] in JuMP.SecondOrderCone()))
    return ucs_variance_risk, :eucs_variance_risk_
end
function set_ucs_variance_risk!(model::JuMP.Model, i::Any,
                                ucs::CompactCovarianceUncertaintySet, sigma::MatNum;
                                prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    w = get_w(model, prefix)
    # The set is a neighbourhood of the covariance it was calibrated on, so it names the
    # centre. The risk measure's field and then the prior are the fallbacks (ADR 0050).
    sigma = something(ucs.val, sigma)
    G = LinearAlgebra.cholesky(sigma).U
    dev_cucs = state_set!(model, prefix, :dev_cucs, i, JuMP.@variable(model))
    state_set!(model, prefix, :cdev_cucs_soc, i,
               JuMP.@constraint(model,
                                [sc * dev_cucs; sc * G * w] in JuMP.SecondOrderCone()))
    C = ucs.C
    Q = ucs.Q
    # The basis spans the directions the penalty spares, so a rank of zero leaves the whole
    # of `C .* w` in the residual and needs no coefficient variable.
    x_cucs = if size(Q, 2) > zero(Int)
        z_cucs = state_set!(model, prefix, :z_cucs, i,
                            JuMP.@variable(model, [1:size(Q, 2)]))
        JuMP.@expression(model, C .* w - Q * z_cucs)
    else
        JuMP.@expression(model, C .* w)
    end
    t_cucs = state_set!(model, prefix, :t_cucs, i, JuMP.@variable(model))
    state_set!(model, prefix, :x_cucs, i, x_cucs)
    state_set!(model, prefix, :gc_soc, i,
               JuMP.@constraint(model,
                                [sc * t_cucs; sc * x_cucs] in JuMP.SecondOrderCone()))
    ucs_variance_risk = state_set!(model, prefix, :cucs_variance_risk_, i,
                                   JuMP.@expression(model,
                                                    dev_cucs^2 + ucs.kappa * t_cucs^2))
    return ucs_variance_risk, :cucs_variance_risk_
end
function set_ucs_variance_risk!(model::JuMP.Model, i::Any,
                                ucs::NormBallUncertaintySet{<:Any, <:Any, <:Any,
                                                            <:SigmaUncertaintySetClass},
                                sigma::MatNum; prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    set_sdp_constraints!(model; prefix = prefix)
    state_build!(model, prefix, :E) do
        W = state_get(model, prefix, :W)
        N = size(W, 1)
        E = JuMP.@variable(model, [1:N, 1:N], Symmetric)
        state_set!(model, prefix, :WpE, JuMP.@expression(model, W + E))
        state_set!(model, prefix, :ceucs_variance,
                   JuMP.@constraint(model, sc * E in JuMP.PSDCone()))
        return E
    end
    WpE = state_get(model, prefix, :WpE)
    # The set is a neighbourhood of the covariance it was calibrated on, so it names the
    # centre. The risk measure's field and then the prior are the fallbacks (ADR 0050).
    sigma = something(ucs.val, sigma)
    L = ucs.L
    # A map with no column spans nothing, so the worst case is the nominal variance and no
    # cone is needed.
    ucs_variance_risk = if size(L, 2) > zero(Int)
        x_nbucs = state_set!(model, prefix, :x_nbucs_, i,
                             JuMP.@expression(model, transpose(L) * vec(WpE)))
        t_nbucs = norm_ball_dual_norm_epigraph!(model, prefix, i, x_nbucs, ucs.p)
        JuMP.@expression(model, LinearAlgebra.tr(sigma * WpE) + ucs.kappa * t_nbucs)
    else
        JuMP.@expression(model, LinearAlgebra.tr(sigma * WpE))
    end
    state_set!(model, prefix, :nbucs_variance_risk_, i, ucs_variance_risk)
    return ucs_variance_risk, :nbucs_variance_risk_
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
                               args...; prefix::Symbol = Symbol(""),
                               rd::ReturnsResult = ReturnsResult(), kwargs...)
    mark_state!(model, prefix, :variance_flag)
    # The lift is the set's business: the box and the ellipsoid bound a matrix and raise
    # `W` themselves, and the compact set bounds a quadratic form in `w` and raises none.
    ucs = r.ucs
    sigma = nothing_scalar_array_selector(r.sigma, pr.sigma)
    ucs_variance_risk, name = set_ucs_variance_risk!(model, i,
                                                     sigma_ucs(ucs, rd; kwargs...), sigma;
                                                     prefix = prefix)
    set_risk_bounds_and_expression!(model, opt, ucs_variance_risk, r.settings, name, i;
                                    prefix = prefix)
    return ucs_variance_risk
end
