"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the factor of the prior's covariance matrix, and register it in the model on the first call.

The factor is the prior's `chol` when the prior carries one, and [`covariance_factor`](@ref) of the prior's `sigma` otherwise. The factor of a factor prior is not square, and the second-order cone is well posed on it. `G` is a shared entry, so every measure of the model that falls back to the prior reads one factor, and a later call returns it without a second factorisation.

# JuMP formulation

## Expressions

  - `G`: ``\\mathbf{G}``, registered on the first call.

Where:

  - $(math_dict[:G_cov_factor])

# Arguments

  - $(arg_dict[:model])
  - `pr::AbstractPriorResult`: The prior result. The function reads its `sigma` and its `chol`.

# Returns

  - `G::AbstractMatrix`: The factor of the prior's covariance matrix.

# Related

  - [`covariance_factor`](@ref)
  - [`chol_sigma_selector`](@ref)
"""
function get_chol_or_sigma_pm(model::JuMP.Model, pr::AbstractPriorResult)
    if !shared_has(model, :G)
        G = isnothing(pr.chol) ? covariance_factor(pr.sigma) : pr.chol
        JuMP.@expression(model, G, G)
    end
    return shared_get(model, :G)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return a factor ``\\mathbf{G}`` of a covariance matrix with ``\\mathbf{G}^\\intercal \\mathbf{G} = \\mathbf{\\Sigma}``, for a singular matrix as well as a positive definite one.

A positive definite matrix gives its upper Cholesky factor. A singular positive semidefinite matrix, such as a rank-one estimate, has no Cholesky factor, so it gives the factor of its eigendecomposition. The second-order cone ``\\sigma \\geq \\lVert \\mathbf{G} \\boldsymbol{w} \\rVert_2`` is well posed on either factor.

# Mathematical definition

```math
\\begin{align}
\\mathbf{G} &= \\mathbf{U}\\,, \\quad \\text{when } \\mathbf{\\Sigma} = \\mathbf{U}^\\intercal \\mathbf{U} \\succ 0\\,, \\\\
\\mathbf{G} &= \\max(\\mathbf{\\Lambda}, 0)^{1/2} \\mathbf{V}^\\intercal\\,, \\quad \\text{when } \\mathbf{\\Sigma} = \\mathbf{V} \\mathbf{\\Lambda} \\mathbf{V}^\\intercal \\text{ is singular}\\,.
\\end{align}
```

Where:

  - $(math_dict[:G_cov_factor])
  - ``\\mathbf{U}``: Upper triangular Cholesky factor.
  - ``\\mathbf{V}``, ``\\mathbf{\\Lambda}``: Eigenvectors, and the diagonal matrix of the eigenvalues. The maximum with zero removes a negative eigenvalue of rounding.

# Arguments

  - `sigma::AbstractMatrix`: The covariance matrix, `assets × assets`.

# Validation

  - A matrix that has no Cholesky factor and is not Hermitian raises `LinearAlgebra.PosDefException(-1)`.
  - A matrix whose smallest eigenvalue is below ``-N \\epsilon \\max_i \\lvert \\lambda_i \\rvert`` raises `LinearAlgebra.PosDefException(1)`. Here ``\\epsilon`` is the machine epsilon of the element type. So the function refuses a `BigFloat` copy of a `Float64` matrix that rounding left indefinite, and it accepts the `Float64` matrix.

# Returns

  - `G::AbstractMatrix`: The factor, `assets × assets`.

# Related

  - [`get_chol_or_sigma_pm`](@ref)
  - [`chol_sigma_selector`](@ref)
  - [`RankOneCovariance`](@ref)
"""
function covariance_factor(sigma::AbstractMatrix)
    F = LinearAlgebra.cholesky(sigma; check = false)
    if LinearAlgebra.issuccess(F)
        return F.U
    end
    @argcheck(LinearAlgebra.ishermitian(sigma), LinearAlgebra.PosDefException(-1))
    E = LinearAlgebra.eigen(LinearAlgebra.Symmetric(sigma))
    tol = -length(E.values) * eps(eltype(E.values)) * maximum(abs, E.values)
    @argcheck(minimum(E.values) >= tol, LinearAlgebra.PosDefException(1))
    return LinearAlgebra.Diagonal(sqrt.(max.(E.values, zero(eltype(E.values))))) *
           transpose(E.vectors)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the factor of the covariance matrix that a risk measure reads.

The measure's own matrix wins over the prior's, and a stated factor wins over one computed here:

  - `r.chol`, when the measure states it.
  - [`covariance_factor`](@ref) of `r.sigma`, when the measure states `sigma` alone.
  - [`get_chol_or_sigma_pm`](@ref), the prior's factor, when the measure states neither.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:pr])
  - `r::CholRM`: The measure, or the algorithm of a measure, that carries the optional `sigma` and `chol`.

# Returns

  - `G::AbstractMatrix`: The factor, with ``\\mathbf{G}^\\intercal \\mathbf{G} = \\mathbf{\\Sigma}``.

# Related

  - [`get_chol_or_sigma_pm`](@ref)
  - [`covariance_factor`](@ref)
"""
function chol_sigma_selector(model::JuMP.Model, pr::Option{<:AbstractPriorResult},
                             r::CholRM)
    return if isnothing(r.sigma) && isnothing(r.chol)
        get_chol_or_sigma_pm(model, pr)
    elseif isnothing(r.chol)
        covariance_factor(r.sigma)
    else
        r.chol
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Bound one expression, and add another to the risk that the objective reads.

The row of [`set_risk_upper_bound!`](@ref) multiplies the bound by ``k``, so it holds under a ratio objective only for an expression of degree one in the weights and ``k``. A second moment has degree two. The builder passes the square root, of degree one, as `r_expr_ub` with its bound converted by [`variance_risk_bounds_val`](@ref), and the second moment as `r_expr`. A builder whose risk has degree one passes one expression twice.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:opt_rjumpe])
  - `r_expr_ub::JuMP.AbstractJuMPScalar`: The expression that the bound row acts on.
  - `ub`: The bound, in the units of `r_expr_ub`, or `nothing`.
  - `name::Symbol`: The Model State name of `r_expr_ub`. The keys of the bound derive from it.
  - `i`: The index of the measure. `name` and `i` compose the key here, so the bound keys cannot differ from the key that the builder registered `r_expr_ub` under.
  - `r_expr::JuMP.AbstractJuMPScalar`: The risk expression that the objective reads.
  - `settings::JuMPRiskMeasureSettings`: The settings of the measure, which give the weight `scale` and the flag `rke`.
  - `flag::Bool`: `true` for an upper bound, and `false` for a lower bound (default: `true`).

# Keyword arguments

  - `prefix::Symbol`: Model State namespace for `name` (default: empty, i.e. the bare key).

# Returns

  - `nothing`.

# Related

  - [`set_risk_upper_bound!`](@ref)
  - [`set_risk_expression!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref): The sibling that bounds the risk expression itself.
  - [`state_key`](@ref)
"""
function set_variance_risk_bounds_and_expression!(model::JuMP.Model,
                                                  opt::RiskConstraintOwner,
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

Build the risk expression of a standard deviation or of a variance, and return it with the name or the formulation that its bound reads.

The standard-deviation method bounds the norm of ``\\mathbf{G} \\boldsymbol{w}`` with a second-order cone variable. The variance method chooses the formulation with [`sdp_variance_flag!`](@ref), builds the expression with [`set_variance_risk!`](@ref), and adds the risk-contribution rows of `r.rc` with [`rc_variance_constraints!`](@ref).

# Mathematical definition

```math
\\begin{align}
\\sigma(\\boldsymbol{w}) &= \\lVert \\mathbf{G} \\boldsymbol{w} \\rVert_2 = \\sqrt{\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}}\\,.
\\end{align}
```

Where:

  - ``\\sigma(\\boldsymbol{w})``: Portfolio standard deviation.
  - $(math_dict[:G_cov_factor])
  - $(math_dict[:Sigma_rm])
  - $(math_dict[:w_port])

# JuMP formulation

## Variables

  - `w`: read from the model.
  - `sd_risk_`: ``\\sigma``, created by the standard-deviation method.

## Constraints

  - `csd_risk_soc_`: ``(s_c \\sigma, s_c \\mathbf{G} \\boldsymbol{w}) \\in \\mathcal{K}_{2}``, which is ``\\sigma \\geq \\lVert \\mathbf{G} \\boldsymbol{w} \\rVert_2``.

Each name carries the index `i` of the measure, and the namespace `prefix`.

Where:

  - $(math_dict[:sigma_epi])
  - $(math_dict[:K_q_norm])
  - $(math_dict[:sc_scale])

## Relaxation

$(val_dict[:relax])

  - `sd_risk_` lies at or above ``\\sigma(\\boldsymbol{w})``. It is tight when the objective minimises the risk. A bound on it is exact, because the weights satisfy it when their standard deviation is at most the bound.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r`: The measure, a [`StandardDeviation`](@ref) or a [`Variance`](@ref).
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])
  - $(arg_dict[:pl_opt]) The variance method alone reads it.

# Returns

  - The standard-deviation method returns `(sd_risk, :sd_risk_)`, the variable and the bare Model State name that the method registers it under. The caller pairs the name with the same index to compose the key of the bound.
  - The variance method returns `(variance_risk, flag)`, the variance expression and the [`FrontierBoundEstimator`](@ref) of its formulation: [`LinearBound`](@ref) for the semidefinite one, and [`SquareRootBound`](@ref) for the second-order cone one.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`set_variance_risk!`](@ref)
  - [`chol_sigma_selector`](@ref)
"""
function set_risk!(model::JuMP.Model, i::Any, r::StandardDeviation,
                   opt::RiskConstraintOwner, pr::Option{<:AbstractPriorResult}, args...;
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

Add a standard deviation to `model`, bound it by `settings.ub`, and add it to the risk that the objective reads.

The standard deviation has degree one in the weights, so its bound `settings.ub` is in the units of a standard deviation and acts on `sd_risk_` directly.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::StandardDeviation`: The measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Returns

  - `sd_risk`: The standard-deviation variable.

# Related

  - [`set_risk!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::StandardDeviation,
                               opt::RiskConstraintOwner, pr::Option{<:AbstractPriorResult},
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    sd_risk, name = set_risk!(model, i, r, opt, pr, args...; prefix = prefix, kwargs...)
    set_risk_bounds_and_expression!(model, opt, sd_risk, r.settings, name, i;
                                    prefix = prefix)
    return sd_risk
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return whether the risk-contribution rows of a variance need the semidefinite formulation.

The rows state shares of the lifted variance, so any row needs it. The method on `nothing` returns `false`, and the method on a [`LinearConstraint`](@ref) returns `true`.

# Arguments

  - $(arg_dict[:model])
  - `opt::RiskBoundOwner`: The owner of the constraint.
  - `rc`: The resolved rows, `nothing` or a [`LinearConstraint`](@ref).

# Returns

  - `flag::Bool`: Whether the rows need the semidefinite formulation.

# Related

  - [`sdp_variance_flag!`](@ref)
  - [`risk_contribution_constraints`](@ref)
"""
function sdp_rc_variance_flag!(::JuMP.Model, ::RiskBoundOwner, ::Nothing)
    return false
end
function sdp_rc_variance_flag!(::JuMP.Model, ::RiskBoundOwner, ::LinearConstraint)
    return true
end
"""
    risk_contribution_constraints(r::Variance, opt::NonFRCJuMPOpt, pr::AbstractPriorResult)

The risk-contribution rows of a [`Variance`](@ref) as the [`RiskConstraintOwner`](@ref) resolves them. On a JuMP optimiser they are `r.rc` through [`linear_constraints`](@ref) over the optimiser's `sets` and `strict`, in the element type of the prior's returns. A programme Allocation Set answers its own; see [`ProgrammeAllocationSet`](@ref).

# Related

  - [`RiskConstraintOwner`](@ref)
  - [`set_risk!`](@ref)
  - [`sdp_rc_variance_flag!`](@ref)
"""
function risk_contribution_constraints(r::Variance, opt::NonFRCJuMPOpt,
                                       pr::AbstractPriorResult)
    return linear_constraints(r.rc, opt.opt.sets; datatype = eltype(pr.X),
                              strict = opt.opt.strict)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the [`FrontierBoundEstimator`](@ref) that names the formulation of a variance.

The function returns [`LinearBound`](@ref), the semidefinite formulation, when `rc_flag` is `true`, when the namespace that owns the weights carries the `rc_variance` mark, or when `pl` holds a [`SemiDefinitePhylogeny`](@ref). It returns [`SquareRootBound`](@ref), the second-order cone formulation, otherwise. [`rc_variance_constraints!`](@ref) sets the mark, so a variance built after one with risk-contribution rows on the same weights also takes the semidefinite formulation. A variance built before it keeps the second-order cone, so the formulation of a variance without rows depends on its place in the vector of measures.

# Arguments

  - $(arg_dict[:model])
  - `rc_flag::Bool`: Whether the measure's own rows need the semidefinite formulation.
  - `pl`: Optional phylogeny constraint result or results.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build. The mark is read on its owner, [`weights_prefix`](@ref).

# Returns

  - `bound::FrontierBoundEstimator`: [`LinearBound`](@ref) for the semidefinite formulation, and [`SquareRootBound`](@ref) for the second-order cone formulation.

# Related

  - [`sdp_rc_variance_flag!`](@ref)
  - [`set_variance_risk!`](@ref)
  - [`variance_risk_bounds_expr`](@ref)
  - [`variance_risk_bounds_val`](@ref)
"""
function sdp_variance_flag!(model::JuMP.Model, rc_flag::Bool, pl::Option{<:PlC_VecPlC};
                            prefix::Symbol = Symbol(""))
    return if rc_flag ||
              state_has(model, weights_prefix(model, prefix), :rc_variance) ||
              isa(pl, SemiDefinitePhylogeny) ||
              isa(pl, AbstractVector) && any(x -> isa(x, SemiDefinitePhylogeny), pl)
        LinearBound()
    else
        SquareRootBound()
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the variance expression of the formulation that `flag` or `r.alg` selects.

The method on [`LinearBound`](@ref) builds the semidefinite formulation with [`set_sdp_variance_risk!`](@ref). The method on [`SquareRootBound`](@ref) passes to the method of `r.alg`. Both second-order cone methods build the deviation ``\\sigma`` with its cone, because the bound acts on ``\\sigma`` in the units of a standard deviation. [`SquaredSOCRiskExpr`](@ref) states the variance as ``\\sigma^2``, and [`QuadRiskExpr`](@ref) states it as the quadratic form.

# JuMP formulation

## Variables

  - `w`: read from the model.
  - `dev_`: ``\\sigma``, created by both second-order cone methods.

## Expressions

  - `variance_risk_`: ``\\sigma^2`` under [`SquaredSOCRiskExpr`](@ref), and ``\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w}`` under [`QuadRiskExpr`](@ref).

## Constraints

  - `cdev_soc_`: ``(s_c \\sigma, s_c \\mathbf{G} \\boldsymbol{w}) \\in \\mathcal{K}_{2}``, which is ``\\sigma \\geq \\lVert \\mathbf{G} \\boldsymbol{w} \\rVert_2``.

Each name carries the index `i` of the measure, and the namespace `prefix`.

Where:

  - $(math_dict[:sigma_epi])
  - $(math_dict[:G_cov_factor])
  - $(math_dict[:Sigma_rm])
  - $(math_dict[:w_port])
  - $(math_dict[:K_q_norm])
  - $(math_dict[:sc_scale])

## Relaxation

$(val_dict[:relax])

  - `dev_` lies at or above ``\\lVert \\mathbf{G} \\boldsymbol{w} \\rVert_2``, so under [`SquaredSOCRiskExpr`](@ref) `variance_risk_` lies at or above the variance. Both are tight when the objective minimises the risk.
  - Under [`QuadRiskExpr`](@ref) `variance_risk_` is exact, and `dev_` serves the bound alone.
  - A bound on `dev_` is exact under both, because the weights satisfy it when their variance is at most the square of the bound.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::Variance`: The measure.
  - $(arg_dict[:pr_sigma])
  - `flag`:
      + `::LinearBound`: The semidefinite formulation.
      + `::SquareRootBound`: The second-order cone formulation of `r.alg`.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `variance_risk`: The variance expression.

# Related

  - [`set_sdp_variance_risk!`](@ref)
  - [`sdp_variance_flag!`](@ref)
  - [`set_risk!`](@ref)
"""
function set_variance_risk!(model::JuMP.Model, i::Any, r::Variance,
                            pr::Option{<:AbstractPriorResult}, ::LinearBound;
                            prefix::Symbol = Symbol(""))
    return set_sdp_variance_risk!(model, i, r, pr; prefix = prefix)
end
function set_variance_risk!(model::JuMP.Model, i::Any, r::Variance,
                            pr::Option{<:AbstractPriorResult}, ::SquareRootBound;
                            prefix::Symbol = Symbol(""))
    return set_variance_risk!(model, i, r, pr; prefix = prefix)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Build the semidefinite variance expression, the trace of the covariance matrix times the lifted weight matrix.

The trace has degree one in the weights and ``k``, where the second-order cone expressions have degree two. So under [`MaximumRatio`](@ref) the model maximises the excess return per unit of variance, not the Sharpe ratio. The `## The degree of the risk` subsection of [`MaximumRatio`](@ref) states the rule. The bound acts on the trace in the units of a variance.

# JuMP formulation

## Variables

  - `W`: read, the lifted matrix that [`set_sdp_constraints!`](@ref) builds once for the owner of the weights.

## Expressions

  - `sigma_W_`: ``\\mathbf{\\Sigma} \\mathbf{W}``.
  - `variance_risk_`: ``\\mathrm{Tr}(\\mathbf{\\Sigma} \\mathbf{W})``.

Each name carries the index `i` of the measure, and the namespace `prefix`.

Where:

  - $(math_dict[:W_lift])
  - $(math_dict[:Sigma_rm])
  - $(math_dict[:k_budget])

## Relaxation

$(val_dict[:relax])

  - `variance_risk_` lies at or above ``\\boldsymbol{w}^\\intercal \\mathbf{\\Sigma} \\boldsymbol{w} / k``, because ``\\mathbf{W} \\succeq \\boldsymbol{w} \\boldsymbol{w}^\\intercal / k`` and the covariance matrix is positive semidefinite. It is tight at ``\\mathbf{W} = \\boldsymbol{w} \\boldsymbol{w}^\\intercal / k``, which the objective reaches when it minimises the risk and no other row acts on ``\\mathbf{W}``.
  - A bound on `variance_risk_` alone is exact, because the weights satisfy it when their variance is at most the bound. Risk-contribution rows on ``\\mathbf{W}`` make the bound and the rows one relaxation, which [`rc_variance_constraints!`](@ref) states.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::Variance`: The measure.
  - $(arg_dict[:pr_sigma])

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `variance_risk`: The variance expression.

# Related

  - [`set_variance_risk!`](@ref)
  - [`set_sdp_constraints!`](@ref)
  - [`rc_variance_constraints!`](@ref)
"""
function set_sdp_variance_risk!(model::JuMP.Model, i::Any, r::Variance,
                                pr::Option{<:AbstractPriorResult};
                                prefix::Symbol = Symbol(""))
    W = set_sdp_constraints!(model; prefix = prefix)
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    sigma_W = state_set!(model, prefix, :sigma_W_, i, JuMP.@expression(model, sigma * W))
    return state_set!(model, prefix, :variance_risk_, i,
                      JuMP.@expression(model, LinearAlgebra.tr(sigma_W)))
end
function set_variance_risk!(model::JuMP.Model, i::Any,
                            r::Variance{<:Any, <:Any, <:Any, <:Any, <:SquaredSOCRiskExpr},
                            pr::Option{<:AbstractPriorResult}; prefix::Symbol = Symbol(""))
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
                            pr::Option{<:AbstractPriorResult}; prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    w = get_w(model, prefix)
    sigma = isnothing(r.sigma) ? pr.sigma : r.sigma
    G = chol_sigma_selector(model, pr, r)
    dev = state_set!(model, prefix, :dev_, i, JuMP.@variable(model))
    state_set!(model, prefix, :cdev_soc_, i,
               JuMP.@constraint(model, [sc * dev; sc * G * w] in JuMP.SecondOrderCone()))
    return state_set!(model, prefix, :variance_risk_, i,
                      JuMP.@expression(model, LinearAlgebra.dot(w, sigma, w)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the expression that the bound of a variance acts on, and its bare Model State name.

The method on [`LinearBound`](@ref) returns the trace `variance_risk_` of the semidefinite formulation, and the method on [`SquareRootBound`](@ref) returns the deviation `dev_` of the second-order cone formulation. Both have degree one in the weights and ``k``, so the row of [`set_risk_upper_bound!`](@ref) holds under a ratio objective.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `flag`:
      + `::LinearBound`: The semidefinite formulation.
      + `::SquareRootBound`: The second-order cone formulation.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `(expr, name)`: The expression, and the bare name that its builder registered it under.

# Related

  - [`variance_risk_bounds_val`](@ref)
  - [`sdp_variance_flag!`](@ref)
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

Convert a bound into the units of the expression that it acts on.

A caller states the bound of a second moment in the units of the measure. A formulation whose bound expression is the square root takes the square root of the bound, and a formulation whose bound expression is the square takes its square.

  - [`LinearBound`](@ref): The bound, unchanged.
  - [`SquareRootBound`](@ref): The square root of the bound, entry by entry for a vector.
  - [`SquaredBound`](@ref): The square of the bound, entry by entry for a vector.

A [`Frontier`](@ref) keeps its `N`, and takes `bound` as its [`FrontierBoundEstimator`](@ref) with a `factor` of `1`. The frontier solve converts the ends of its span when it builds the sweep. `nothing` gives `nothing`.

# Arguments

  - `bound::FrontierBoundEstimator`: The conversion.
  - `ub`: The bound, a number, a vector, a [`Frontier`](@ref) or `nothing`.

# Returns

  - The converted bound, or `nothing` when `ub` is `nothing`.

# Related

  - [`FrontierBoundEstimator`](@ref)
  - [`LinearBound`](@ref)
  - [`SquareRootBound`](@ref)
  - [`SquaredBound`](@ref)
  - [`variance_risk_bounds_expr`](@ref)
  - [`_Frontier`](@ref)
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

Add linear risk-contribution rows on the variance decomposition to `model`.

The method on `args...` does nothing, and it runs when the measure has no rows. The method on a [`LinearConstraint`](@ref) reads the diagonal of `sigma_W_`, adds the rows, and sets the `rc_variance` mark on the namespace that owns the weights, which [`sdp_variance_flag!`](@ref) reads.

# JuMP formulation

## Constraints

  - `rc_variance_ineq_`: ``s_c \\left(\\mathbf{A} \\, \\mathrm{diag}(\\mathbf{S}) - \\boldsymbol{b} \\, \\mathrm{Tr}(\\mathbf{S})\\right) \\leq 0``, registered when `rc` holds inequality rows.
  - `rc_variance_eq_`: ``s_c \\left(\\mathbf{C} \\, \\mathrm{diag}(\\mathbf{S}) - \\boldsymbol{d} \\, \\mathrm{Tr}(\\mathbf{S})\\right) = 0``, registered when `rc` holds equality rows.

Where:

  - ``\\mathbf{S}``: The `sigma_W_` expression, read from the model. It is ``\\mathbf{\\Sigma} \\mathbf{W}`` over the asset weights, and ``\\mathbf{B}_1^\\intercal \\mathbf{\\Sigma} \\mathbf{B}_1 \\mathbf{W}_{f}`` over the factor weights of [`FactorRiskContribution`](@ref).
  - ``\\mathbf{W}``, ``\\mathbf{W}_{f}``: The symmetric matrix variables that [`set_sdp_constraints!`](@ref) and [`set_sdp_frc_constraints!`](@ref) bound by ``\\mathbf{W} \\succeq \\boldsymbol{w} \\boldsymbol{w}^\\intercal / k``.
  - ``\\mathbf{A}``, ``\\boldsymbol{b}``: The inequality rows of `rc` and their bounds.
  - ``\\mathbf{C}``, ``\\boldsymbol{d}``: The equality rows of `rc` and their targets.
  - ``\\mathrm{Tr}(\\mathbf{S})``: The `variance_risk` argument.
  - $(math_dict[:w_port])
  - $(math_dict[:w_1_factor]) It takes the place of ``\\boldsymbol{w}`` over the factor weights.
  - $(math_dict[:k_budget])
  - $(math_dict[:sc_scale])

## Relaxation

$(val_dict[:relax])

  - The variance ``\\mathrm{Tr}(\\mathbf{S})`` in `variance_risk_` lies **above** its value at ``\\mathbf{W} = \\boldsymbol{w} \\boldsymbol{w}^\\intercal / k``, because ``\\mathbf{W} \\succeq \\boldsymbol{w} \\boldsymbol{w}^\\intercal / k`` and the covariance matrix is positive semidefinite.
  - The rows bound the shares of ``\\mathrm{diag}(\\mathbf{S})`` in `sigma_W_`, not the risk contributions of the returned weights. Those can lie on either side of a bound.
  - Both are tight when ``\\mathbf{W} = \\boldsymbol{w} \\boldsymbol{w}^\\intercal / k``, a matrix of rank one. No term of the model forces this, so a solve that reports success can return weights whose shares miss the rows. This is formulations 9, 10 and 16 of [sdprp](@cite), which name the rank-one relaxation as their approximation.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `rc::LinearConstraint`: The resolved risk-contribution rows.
  - `variance_risk::JuMP.AbstractJuMPScalar`: The variance expression, the trace of `sigma_W_`.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `nothing`.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`set_sdp_variance_risk!`](@ref)
  - [`Variance`](@ref): Its `## Risk contribution constraints` subsection states how to check a result.

# References

  - $(ref_dict[:sdprp])
"""
function rc_variance_constraints!(args...; kwargs...)
    return nothing
end
function rc_variance_constraints!(model::JuMP.Model, i::Any, rc::LinearConstraint,
                                  variance_risk::JuMP.AbstractJuMPScalar;
                                  prefix::Symbol = Symbol(""))
    sigma_W = state_get(model, prefix, :sigma_W_, i)
    sc = get_constraint_scale(model)
    mark_state!(model, weights_prefix(model, prefix), :rc_variance)
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
function set_risk!(model::JuMP.Model, i::Any, r::Variance, opt::RiskBoundOwner,
                   pr::Option{<:AbstractPriorResult}, pl::Option{<:PlC_VecPlC}, args...;
                   prefix::Symbol = Symbol(""), kwargs...)
    rc = risk_contribution_constraints(r, opt, pr)
    rc_flag = sdp_rc_variance_flag!(model, opt, rc)
    sdp_flag = sdp_variance_flag!(model, rc_flag, pl; prefix = prefix)
    variance_risk = set_variance_risk!(model, i, r, pr, sdp_flag; prefix = prefix)
    rc_variance_constraints!(model, i, rc, variance_risk; prefix = prefix)
    return variance_risk, sdp_flag
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add a variance to `model`, bound it by `settings.ub`, and add it to the risk that the objective reads.

The bound `settings.ub` is in the units of a variance. [`variance_risk_bounds_expr`](@ref) names the expression of degree one that it acts on, and [`variance_risk_bounds_val`](@ref) converts it to the units of that expression.

When the variance is a positive term of the objective's risk, the builder marks `variance_flag` on the namespace that owns the weights, [`mark_objective_variance!`](@ref). A semidefinite phylogeny on the same weights then adds no `p · tr(W)` penalty when the objective minimises the risk. A variance that is only a bound does not mark the flag, and the penalty stays. [`sdp_variance_flag!`](@ref) selects the formulation, and the source of the covariance matrix, the measure or the prior, does not change it. A measure that holds its matrix reads no prior, so a programme Allocation Set that reads no rows passes `pr = nothing`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::Variance`: The measure.
  - `opt::RiskBoundOwner`: The owner of the constraint.
  - `pr`: The prior result, or `nothing` when `r` holds its covariance matrix and has no risk-contribution rows.
  - $(arg_dict[:pl_opt])

# Returns

  - `variance_risk`: The variance expression.

# Related

  - [`Variance`](@ref)
  - [`set_risk!`](@ref)
  - [`set_variance_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::Variance, opt::RiskBoundOwner,
                               pr::Option{<:AbstractPriorResult}, pl::Option{<:PlC_VecPlC},
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    mark_objective_variance!(model, prefix, r.settings)
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

Add a variance to the model of a [`FactorRiskContribution`](@ref), always in the semidefinite formulation over the factor weights.

The model lifts the factor weights ``\\boldsymbol{w}_1`` into ``\\mathbf{W}_f``, and the variance is the trace of the factor covariance ``\\mathbf{B}_1^\\intercal \\mathbf{\\Sigma} \\mathbf{B}_1`` times ``\\mathbf{W}_f``. The risk-contribution rows of `r.rc` then state factor shares. With `flag = true` the asset weights are ``\\mathbf{B}_1 \\boldsymbol{w}_1 + \\mathbf{B}_2 \\boldsymbol{w}_2``, and the expression omits the off-factor weights ``\\mathbf{B}_2 \\boldsymbol{w}_2``. It is then the variance of the factor part alone, which can lie below the variance of the returned weights. The objective and the bound act on the factor part alone.

# JuMP formulation

## Variables

  - `frc_W`: read, ``\\mathbf{W}_f``, the lifted matrix that [`set_sdp_frc_constraints!`](@ref) builds once.

## Expressions

  - `sigma_W_`: ``\\mathbf{B}_1^\\intercal \\mathbf{\\Sigma} \\mathbf{B}_1 \\mathbf{W}_f``.
  - `variance_risk_`: ``\\mathrm{Tr}(\\mathbf{B}_1^\\intercal \\mathbf{\\Sigma} \\mathbf{B}_1 \\mathbf{W}_f)``.

Each name carries the index `i` of the measure, and the namespace `prefix`.

Where:

  - ``\\mathbf{W}_f``: Lifted matrix of the factor weights, symmetric ``N_f \\times N_f``, with ``\\mathbf{W}_f \\succeq \\boldsymbol{w}_1 \\boldsymbol{w}_1^\\intercal / k``.
  - ``\\mathbf{B}_1``: The pseudoinverse of ``\\mathbf{B}^\\intercal``, `b1`.
  - ``\\mathbf{B}_2``: The basis of the off-factor directions.
  - ``\\boldsymbol{w}_2``: The off-factor weights.
  - $(math_dict[:B_loadings])
  - $(math_dict[:w_1_factor])
  - $(math_dict[:Sigma_rm])
  - $(math_dict[:k_budget])

## Relaxation

$(val_dict[:relax])

  - `variance_risk_` lies at or above ``\\boldsymbol{w}_1^\\intercal \\mathbf{B}_1^\\intercal \\mathbf{\\Sigma} \\mathbf{B}_1 \\boldsymbol{w}_1 / k``. It is tight at ``\\mathbf{W}_f = \\boldsymbol{w}_1 \\boldsymbol{w}_1^\\intercal / k``, and [`rc_variance_constraints!`](@ref) states the condition for its rows.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::Variance`: The measure.
  - `opt::FactorRiskContribution`: The optimisation estimator.
  - $(arg_dict[:pr])
  - `b1::MatNum`: ``\\mathbf{B}_1``, the pseudoinverse of the transposed loadings, which maps the factor weights to the asset weights.

# Returns

  - `variance_risk`: The variance expression.

# Related

  - [`Variance`](@ref)
  - [`FactorRiskContribution`](@ref)
  - [`set_factor_risk_contribution_constraints!`](@ref)
  - [`rc_variance_constraints!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::Variance,
                               opt::FactorRiskContribution, pr::AbstractPriorResult, ::Any,
                               ::Any, b1::MatNum, args...; prefix::Symbol = Symbol(""),
                               kwargs...)
    mark_objective_variance!(model, prefix, r.settings)
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

Build the worst-case variance of an uncertainty set over the covariance matrix.

The box, the ellipsoid and the norm ball bound the covariance matrix, so each lifts the weights into ``\\mathbf{W}`` with [`set_sdp_constraints!`](@ref) and states the dual of its worst case over ``\\mathbf{W}``. The compact set bounds a quadratic form in the weights, so it lifts nothing and leaves the programme a second-order cone programme. It factorises its centre with [`covariance_factor`](@ref), so it accepts a singular centre, as [`Variance`](@ref) does.

# Mathematical definition

```math
\\begin{align}
R_{\\mathrm{box}} &= \\underset{\\mathbf{A}_u,\\, \\mathbf{A}_l \\geq 0,\\ \\mathbf{A}_u - \\mathbf{A}_l = \\mathbf{W}}{\\min} \\langle \\mathbf{\\Sigma}_u, \\mathbf{A}_u \\rangle - \\langle \\mathbf{\\Sigma}_l, \\mathbf{A}_l \\rangle\\,, \\\\
R_{\\mathrm{ell}} &= \\underset{\\mathbf{E} \\succeq 0}{\\min} \\langle \\hat{\\mathbf{\\Sigma}}, \\mathbf{W} + \\mathbf{E} \\rangle + k_{e} \\lVert \\mathbf{G}_{\\Omega} \\, \\mathrm{vec}(\\mathbf{W} + \\mathbf{E}) \\rVert_{2}\\,, \\\\
R_{\\mathrm{nb}} &= \\underset{\\mathbf{E} \\succeq 0}{\\min} \\langle \\hat{\\mathbf{\\Sigma}}, \\mathbf{W} + \\mathbf{E} \\rangle + \\kappa_{b} \\lVert \\mathbf{L}^\\intercal \\mathrm{vec}(\\mathbf{W} + \\mathbf{E}) \\rVert_{p^{*}}\\,, \\\\
R_{\\mathrm{cpt}} &= \\lVert \\mathbf{G} \\boldsymbol{w} \\rVert_{2}^{2} + \\kappa \\underset{\\boldsymbol{z}}{\\min} \\lVert \\mathbf{C} \\boldsymbol{w} - \\mathbf{Q} \\boldsymbol{z} \\rVert_{2}^{2}\\,.
\\end{align}
```

Where:

  - ``R_{\\mathrm{box}}``, ``R_{\\mathrm{ell}}``, ``R_{\\mathrm{nb}}``, ``R_{\\mathrm{cpt}}``: Worst-case variance of the box, the ellipsoid, the norm ball and the compact set. The first three state the worst case over ``\\mathbf{W}``, and they give the worst case of ``\\boldsymbol{w}`` at ``\\mathbf{W} = \\boldsymbol{w} \\boldsymbol{w}^\\intercal / k``.
  - ``\\mathbf{\\Sigma}_l``, ``\\mathbf{\\Sigma}_u``: Lower and upper bounds of the box, `lb` and `ub`.
  - ``\\mathbf{A}_l``, ``\\mathbf{A}_u``: Dual matrices of the two sides of the box.
  - ``\\hat{\\mathbf{\\Sigma}}``: Centre of the set, the set's `val` when it states one and `sigma` otherwise.
  - ``\\mathbf{E}``: Dual matrix of the condition that the worst covariance is positive semidefinite.
  - ``k_{e}``, ``\\mathbf{G}_{\\Omega}``: Radius of the ellipsoid, `k`, and the upper Cholesky factor of its matrix ``\\mathbf{\\Omega}``, `sigma`.
  - ``\\kappa_{b}``, ``\\mathbf{L}``, ``p^{*}``: Radius of the norm ball, its map, and the dual order of its norm.
  - ``\\mathbf{C}``, ``\\mathbf{Q}``, ``\\boldsymbol{z}``: Diagonal metric of the compact set, its basis, and the free coefficients of the basis.
  - ``\\langle \\mathbf{X}, \\mathbf{Y} \\rangle = \\mathrm{Tr}(\\mathbf{X}^\\intercal \\mathbf{Y})``: Inner product of two matrices.
  - $(math_dict[:kappa_cpt])
  - $(math_dict[:W_lift])
  - $(math_dict[:G_cov_factor])
  - $(math_dict[:w_port])

# JuMP formulation

## Variables

  - `W`: read by the box, the ellipsoid and the norm ball. `w`: read by the compact set.
  - `Au`, `Al`: created by the box, ``\\mathbf{A}_u`` and ``\\mathbf{A}_l``, symmetric with non-negative entries.
  - `E`: created by the ellipsoid or the norm ball, ``\\mathbf{E}``, symmetric.
  - `t_eucs`: created by the ellipsoid, ``t_e``.
  - `dev_cucs`, `t_cucs`: created by the compact set, ``d`` and ``t``.
  - `z_cucs`: created by the compact set when ``\\mathbf{Q}`` has a column, ``\\boldsymbol{z}``.

## Expressions

  - `bucs_variance_risk_`: ``\\langle \\mathbf{\\Sigma}_u, \\mathbf{A}_u \\rangle - \\langle \\mathbf{\\Sigma}_l, \\mathbf{A}_l \\rangle``.
  - `WpE`: ``\\mathbf{W} + \\mathbf{E}``.
  - `x_eucs`: ``\\mathbf{G}_{\\Omega} \\, \\mathrm{vec}(\\mathbf{W} + \\mathbf{E})``.
  - `eucs_variance_risk_`: ``\\langle \\hat{\\mathbf{\\Sigma}}, \\mathbf{W} + \\mathbf{E} \\rangle + k_{e} t_e``.
  - `x_cucs`: ``\\mathbf{C} \\boldsymbol{w} - \\mathbf{Q} \\boldsymbol{z}``, or ``\\mathbf{C} \\boldsymbol{w}`` when ``\\mathbf{Q}`` has no column.
  - `cucs_variance_risk_`: ``d^{2} + \\kappa t^{2}``.
  - `x_nbucs_`: ``\\mathbf{L}^\\intercal \\mathrm{vec}(\\mathbf{W} + \\mathbf{E})``, registered when ``\\mathbf{L}`` has a column. [`norm_ball_dual_norm_epigraph!`](@ref) bounds its norm by ``t_b``.
  - `nbucs_variance_risk_`: ``\\langle \\hat{\\mathbf{\\Sigma}}, \\mathbf{W} + \\mathbf{E} \\rangle + \\kappa_{b} t_b``, or the first term alone when ``\\mathbf{L}`` has no column.

## Constraints

  - `cbucs_variance`: ``s_c (\\mathbf{A}_u - \\mathbf{A}_l - \\mathbf{W}) = 0``.
  - `ceucs_variance`: ``s_c \\mathbf{E} \\in \\mathcal{S}_{+}^{N}``, registered by the ellipsoid or the norm ball.
  - `ge_soc`: ``(s_c t_e, s_c \\boldsymbol{x}_e) \\in \\mathcal{K}_{2}``, with ``\\boldsymbol{x}_e`` the `x_eucs` expression.
  - `cdev_cucs_soc`: ``(s_c d, s_c \\mathbf{G} \\boldsymbol{w}) \\in \\mathcal{K}_{2}``.
  - `gc_soc`: ``(s_c t, s_c \\boldsymbol{x}_c) \\in \\mathcal{K}_{2}``, with ``\\boldsymbol{x}_c`` the `x_cucs` expression.

`Au`, `Al`, `E`, `WpE`, `cbucs_variance` and `ceucs_variance` are built once for each namespace, and carry the namespace `prefix` alone. Every other name also carries the index `i` of the measure.

Where:

  - ``t_e``, ``t_b``: Epigraph variables of the ellipsoidal and the norm-ball penalty.
  - ``d``, ``t``: Epigraph variables of the two norms of the compact set.
  - ``\\mathcal{S}_{+}^{N}``: The cone of positive semidefinite ``N \\times N`` matrices.
  - $(math_dict[:K_q_norm])
  - $(math_dict[:sc_scale])

## Relaxation

$(val_dict[:relax])

  - `bucs_variance_risk_`, `eucs_variance_risk_` and `nbucs_variance_risk_` lie at or above their worst case at ``\\mathbf{W} = \\boldsymbol{w} \\boldsymbol{w}^\\intercal / k``, and the epigraph variables at or above their norms. They are tight when the objective minimises the risk.
  - Two sets of one kind on the same weights share `Au` and `Al`, or `E`. The box shares them at no cost. For bounds with ``\\mathbf{\\Sigma}_u \\geq \\mathbf{\\Sigma}_l``, the best ``\\mathbf{A}_l`` is the negative part of ``\\mathbf{W}`` for every box. Each ellipsoid or norm ball lies at or above its own worst case, tight when one ``\\mathbf{E}`` serves them all.
  - `dev_cucs` and `t_cucs` lie at or above their norms, so `cucs_variance_risk_` lies at or above the compact worst case. It is tight when the objective minimises the risk.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `ucs`: The uncertainty set, a [`BoxUncertaintySet`](@ref), an [`EllipsoidalUncertaintySet`](@ref), a [`CompactCovarianceUncertaintySet`](@ref) or a covariance [`NormBallUncertaintySet`](@ref).
  - `sigma::MatNum`: The fallback centre. The set's own `val` wins over it, and the box ignores both, because it names no centre.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `(ucs_variance_risk, name)`: The worst-case variance, and the bare Model State name that the method registers it under.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`ucs_variance_risk_bounds!`](@ref)
  - [`set_sdp_constraints!`](@ref)
  - [`UncertaintySetVariance`](@ref)
  - [`ucs_variance`](@ref): The value-level worst case, which omits ``\\mathbf{E}`` and so lies at or above the model's value.
  - [`norm_ball_dual_norm_epigraph!`](@ref)

# References

  - $(ref_dict[:robustaa])
  - $(ref_dict[:fengpalomar2016])
  - $(ref_dict[:cajas2025]) Section 11.3.
"""
function set_ucs_variance_risk!(model::JuMP.Model, i::Any, ucs::BoxUncertaintySet, args...;
                                prefix::Symbol = Symbol(""))
    W = set_sdp_constraints!(model; prefix = prefix)
    Au = state_build!(model, prefix, :Au) do
        sc = get_constraint_scale(model)
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
                                                    LinearAlgebra.dot(transpose(ub), Au) -
                                                    LinearAlgebra.dot(transpose(lb), Al)))
    return ucs_variance_risk, :bucs_variance_risk_
end
function set_ucs_variance_risk!(model::JuMP.Model, i::Any, ucs::EllipsoidalUncertaintySet,
                                sigma::MatNum; prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    W = set_sdp_constraints!(model; prefix = prefix)
    state_build!(model, prefix, :E) do
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
                                                      LinearAlgebra.dot(transpose(sigma),
                                                                        WpE) + k * t_eucs
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
    G = covariance_factor(sigma)
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
        JuMP.@expression(model, C .* w .- Q * z_cucs)
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
    W = set_sdp_constraints!(model; prefix = prefix)
    state_build!(model, prefix, :E) do
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
        JuMP.@expression(model,
                         LinearAlgebra.dot(transpose(sigma), WpE) + ucs.kappa * t_nbucs)
    else
        JuMP.@expression(model, LinearAlgebra.dot(transpose(sigma), WpE))
    end
    state_set!(model, prefix, :nbucs_variance_risk_, i, ucs_variance_risk)
    return ucs_variance_risk, :nbucs_variance_risk_
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the expression that the bound of an uncertainty-set variance acts on, the bound in its units, and the bare Model State name of the expression.

The row of [`set_risk_upper_bound!`](@ref) multiplies the bound by ``k``, so it needs an expression of degree one in the weights and ``k``. The box, the ellipsoid and the norm ball state their worst case over the lifted matrix, which has degree one, so the fall-through method returns the risk expression, the bound and the name unchanged. The compact worst case ``d^2 + \\kappa t^2`` has degree two. Its method adds the square root as a second-order cone variable, and converts the bound with [`variance_risk_bounds_val`](@ref). A compact set without a bound takes the fall-through method and adds nothing.

# JuMP formulation

## Variables

  - `dev_cucs`, `t_cucs`: read, ``d`` and ``t``.
  - `sd_cucs_`: ``s``, created by the compact method.

## Constraints

  - `csd_cucs_soc_`: ``(s_c s, s_c d, s_c \\sqrt{\\kappa} t) \\in \\mathcal{K}_{2}``, which is ``s \\geq \\sqrt{d^{2} + \\kappa t^{2}}``.

Each name carries the index `i` of the measure, and the namespace `prefix`.

Where:

  - ``d``, ``t``: Epigraph variables of the two norms of the compact set, which [`set_ucs_variance_risk!`](@ref) creates.
  - $(math_dict[:kappa_cpt])
  - $(math_dict[:K_q_norm])
  - $(math_dict[:sc_scale])

## Relaxation

$(val_dict[:relax])

  - `sd_cucs_` lies at or above the square root of `cucs_variance_risk_`. The bound on it is exact. The weights satisfy it when their compact worst-case variance is at most the bound, because ``d``, ``t`` and ``\\boldsymbol{z}`` can take their smallest values.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `ucs`: The uncertainty set that [`set_ucs_variance_risk!`](@ref) built.
  - `r_expr::JuMP.AbstractJuMPScalar`: The worst-case variance.
  - `name::Symbol`: The bare Model State name of `r_expr`.
  - `ub`: The bound, in the units of a variance, or `nothing`.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `(expr, ub, name)`: The expression that the bound acts on, the bound in its units, and the bare name of the expression.

# Related

  - [`set_ucs_variance_risk!`](@ref)
  - [`set_variance_risk_bounds_and_expression!`](@ref)
  - [`variance_risk_bounds_val`](@ref)
"""
function ucs_variance_risk_bounds!(::JuMP.Model, ::Any, ::Any,
                                   r_expr::JuMP.AbstractJuMPScalar, name::Symbol, ub;
                                   prefix::Symbol = Symbol(""))
    return r_expr, ub, name
end
function ucs_variance_risk_bounds!(model::JuMP.Model, i::Any,
                                   ucs::CompactCovarianceUncertaintySet,
                                   ::JuMP.AbstractJuMPScalar, ::Symbol, ub::RkRtBounds;
                                   prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    dev_cucs = state_get(model, prefix, :dev_cucs, i)
    t_cucs = state_get(model, prefix, :t_cucs, i)
    sd_cucs = state_set!(model, prefix, :sd_cucs_, i, JuMP.@variable(model))
    state_set!(model, prefix, :csd_cucs_soc_, i,
               JuMP.@constraint(model,
                                [sc * sd_cucs; sc * dev_cucs;
                                 sc * sqrt(ucs.kappa) * t_cucs] in JuMP.SecondOrderCone()))
    return sd_cucs, variance_risk_bounds_val(SquareRootBound(), ub), :sd_cucs_
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the worst-case variance of an uncertainty set to `model`, bound it by `settings.ub`, and add it to the risk that the objective reads.

The set is the measure's own `ucs`, fitted with [`sigma_ucs`](@ref) when it is an estimator. The fit reads the optimisation's returns `rd` and its prior, because an `AbstractPriorUncertaintySetEstimator` fits from the prior result, and an estimator that carries its own `pe` drops it. The centre falls back to the measure's `sigma` and then to the prior's. The bound `settings.ub` is in the units of a variance, and [`ucs_variance_risk_bounds!`](@ref) names the expression of degree one that it acts on.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:ci])
  - `r::UncertaintySetVariance`: The measure.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr])

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).
  - `rd::ReturnsResult`: The optimisation's returns, read by an estimator of the set (default: an empty `ReturnsResult`).
  - `kwargs...`: Passed to [`sigma_ucs`](@ref).

# Returns

  - `ucs_variance_risk`: The worst-case variance expression.

# Related

  - [`UncertaintySetVariance`](@ref)
  - [`set_ucs_variance_risk!`](@ref)
  - [`ucs_variance_risk_bounds!`](@ref)
  - [`sigma_ucs`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, i::Any, r::UncertaintySetVariance,
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""),
                               rd::ReturnsResult = ReturnsResult(), kwargs...)
    mark_objective_variance!(model, prefix, r.settings)
    sigma = nothing_scalar_array_selector(r.sigma, pr.sigma)
    # The prior travels beside the returns, because an `AbstractPriorUncertaintySetEstimator`
    # is fitted from the optimisation's own prior result rather than from returns data.
    ucs = sigma_ucs(r.ucs, rd, pr; kwargs...)
    ucs_variance_risk, name = set_ucs_variance_risk!(model, i, ucs, sigma; prefix = prefix)
    bound_expr, ub, bound_name = ucs_variance_risk_bounds!(model, i, ucs, ucs_variance_risk,
                                                           name, r.settings.ub;
                                                           prefix = prefix)
    set_variance_risk_bounds_and_expression!(model, opt, bound_expr, ub, bound_name, i,
                                             ucs_variance_risk, r.settings; prefix = prefix)
    return ucs_variance_risk
end
