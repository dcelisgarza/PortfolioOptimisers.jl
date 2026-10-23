"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the rows that bound each entry of the distance matrix `Dt` from below by the absolute value of the matching entry of `Dx`.

The formulation selects the rows. Both methods state the same feasible set, ``D_{ij} \\geq |\\Delta_{ij}|`` for every pair of observations, so both give the same optimum. [`NormOneConeBrownianDistanceVariance`](@ref) adds one cone of dimension two for each entry on or below the diagonal. [`IneqBrownianDistanceVariance`](@ref) adds two matrix rows of ``T^2`` linear inequalities each. `Dx` is antisymmetric and `Dt` is symmetric, so the entry ``(j, i)`` of each matrix row repeats the entry ``(i, j)`` of the other.

# JuMP formulation

## Variables

  - `Dt`: read, the symmetric matrix ``\\mathbf{D}``.

## Constraints

  - `cbdvariance_noc`: under [`NormOneConeBrownianDistanceVariance`](@ref), ``(s_c D_{ij},\\, s_c \\Delta_{ij}) \\in \\mathcal{K}_{1}``, which is ``D_{ij} \\geq |\\Delta_{ij}|``, ``\\forall\\, 1 \\leq j \\leq i \\leq T``.
  - `cp_bdvariance`: under [`IneqBrownianDistanceVariance`](@ref), ``s_c (\\mathbf{D} - \\boldsymbol{\\Delta}) \\geq 0``, entry by entry.
  - `cn_bdvariance`: under [`IneqBrownianDistanceVariance`](@ref), ``s_c (\\mathbf{D} + \\boldsymbol{\\Delta}) \\geq 0``, entry by entry.

Each name carries the namespace `prefix`.

Where:

  - ``\\mathbf{D}``: Symmetric ``T \\times T`` matrix variable of the distances between the portfolio returns, with entries ``D_{ij}``.
  - ``\\boldsymbol{\\Delta}``: The expression `Dx`, the ``T \\times T`` matrix of the differences ``\\Delta_{ij} = \\hat{r}_i - \\hat{r}_j`` of the net portfolio returns ``\\hat{r}_i`` in `net_X`.
  - $(math_dict[:T])
  - $(math_dict[:K_q_norm])
  - $(math_dict[:sc_scale])

## Relaxation

$(val_dict[:relax])

  - The rows bound ``D_{ij}`` from below by ``|\\Delta_{ij}|``, so ``\\mathbf{D}`` is the distance matrix of `net_X` only when every row is tight.
  - The risk expression of [`set_brownian_distance_risk_constraint!`](@ref) increases in each entry of a non-negative ``\\mathbf{D}``. So the rows are tight when the objective minimises the risk.
  - Under a bound on the risk and an objective that does not minimise it, an entry of ``\\mathbf{D}`` can exceed ``|\\Delta_{ij}|``. The risk at the true distance matrix is then at or below the risk at ``\\mathbf{D}``, so the bound still holds the value of the functor of [`BrownianDistanceVariance`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `::BrownianDistanceVarianceFormulation`: The formulation, which selects the rows.
  - `Dt::MatNum`: ``\\mathbf{D}``.
  - `Dx::MatNum`: ``\\boldsymbol{\\Delta}``.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `nothing`.

# Related

  - [`set_brownian_distance_risk_constraint!`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_brownian_distance_variance_constraints!(model::JuMP.Model,
                                                     ::NormOneConeBrownianDistanceVariance,
                                                     Dt::MatNum, Dx::MatNum;
                                                     prefix::Symbol = Symbol(""))
    T = size(Dt, 1)
    sc = get_constraint_scale(model)
    state_set!(model, prefix, :cbdvariance_noc,
               JuMP.@constraint(model, [j = 1:T, i = j:T],
                                [sc * Dt[i, j]; sc * Dx[i, j]] in JuMP.MOI.NormOneCone(2)))
    return nothing
end
function set_brownian_distance_variance_constraints!(model::JuMP.Model,
                                                     ::IneqBrownianDistanceVariance,
                                                     Dt::MatNum, Dx::MatNum;
                                                     prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    state_set!(model, prefix, :cp_bdvariance,
               JuMP.@constraint(model, sc * (Dt - Dx) in JuMP.Nonnegatives()))
    state_set!(model, prefix, :cn_bdvariance,
               JuMP.@constraint(model, sc * (Dt + Dx) in JuMP.Nonnegatives()))
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Register the risk expression of [`BrownianDistanceVariance`](@ref) over the distance matrix `Dt`, and return it.

The expression is equation 9 of [bdvar](@cite), the disciplined-convex upper bound on the distance variance that the docstring of [`BrownianDistanceVariance`](@ref) derives. [`QuadRiskExpr`](@ref) states it as a quadratic form in `Dt`. [`RSOCRiskExpr`](@ref) holds each of its two squares with a rotated second-order cone, so the expression is affine in the variables and a bound on the risk is a conic row.

# JuMP formulation

## Variables

  - `Dt`: read, the symmetric matrix ``\\mathbf{D}``.
  - `tDt`: created under [`RSOCRiskExpr`](@ref), the scalar ``t``.
  - `tsum_Dt`: created under [`RSOCRiskExpr`](@ref), the scalar ``u``.

## Expressions

  - `bdvariance_risk`: ``\\frac{1}{T^2} \\left( \\lVert \\mathbf{D} \\rVert_F^2 + \\frac{1}{T^2} \\left( \\boldsymbol{1}^\\intercal \\mathbf{D} \\boldsymbol{1} \\right)^2 \\right)`` under [`QuadRiskExpr`](@ref), and ``\\frac{1}{T^2} (t + u)`` under [`RSOCRiskExpr`](@ref).

## Constraints

  - `rsoc_Dt`: under [`RSOCRiskExpr`](@ref), ``(s_c t,\\, s_c / 2,\\, s_c \\operatorname{vec}(\\mathbf{D})) \\in \\mathcal{K}_{r}``, which is ``t \\geq \\lVert \\mathbf{D} \\rVert_F^2``.
  - `rsoc_sum_Dt`: under [`RSOCRiskExpr`](@ref), ``(s_c u,\\, s_c / 2,\\, s_c \\boldsymbol{1}^\\intercal \\mathbf{D} \\boldsymbol{1} / T) \\in \\mathcal{K}_{r}``, which is ``u \\geq \\left( \\boldsymbol{1}^\\intercal \\mathbf{D} \\boldsymbol{1} / T \\right)^2``. The factor ``1 / T`` keeps ``u`` of the size of ``t``, so both reach the same relative accuracy in the solver.

Each name carries the namespace `prefix`.

Where:

  - ``\\mathbf{D}``: Symmetric ``T \\times T`` matrix variable of the distances between the portfolio returns.
  - $(math_dict[:T])
  - ``\\boldsymbol{1}``: Column vector of ones, ``T \\times 1``.
  - ``\\lVert \\cdot \\rVert_F``: Frobenius norm.
  - ``t``, ``u``: Epigraph variables of the two squares.
  - ``\\mathcal{K}_{r} = \\{(a, b, \\boldsymbol{x}) : 2 a b \\geq \\lVert \\boldsymbol{x} \\rVert_2^2,\\, a \\geq 0,\\, b \\geq 0\\}``: Rotated second-order cone. The model states it with a [JuMP.RotatedSecondOrderCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Rotated-Second-Order-Cone).
  - $(math_dict[:sc_scale])

## Relaxation

$(val_dict[:relax])

  - `bdvariance_risk` lies at or above the distance variance ``\\frac{1}{T^2} \\lVert \\mathbf{C}_T \\mathbf{D} \\mathbf{C}_T \\rVert_F^2`` of ``\\mathbf{D}``, with ``\\mathbf{C}_T`` the centring matrix. The gap is ``\\frac{2}{T^3} \\lVert \\mathbf{D} \\boldsymbol{1} \\rVert_2^2``, which is zero only when every entry of ``\\mathbf{D}`` is zero, that is, for a constant series of portfolio returns. On Gaussian returns at ``T = 50`` the bound is 6.9 times the distance variance.
  - Under [`QuadRiskExpr`](@ref) `bdvariance_risk` is exact in ``\\mathbf{D}``.
  - Under [`RSOCRiskExpr`](@ref) ``t`` and ``u`` lie at or above their squares, so `bdvariance_risk` lies at or above the expression of [`QuadRiskExpr`](@ref). Both are tight when the objective minimises the risk.

# Arguments

  - $(arg_dict[:model])
  - `::QuadRiskExpr` or `::RSOCRiskExpr`: The risk expression formulation.
  - `Dt::MatNum`: ``\\mathbf{D}``.
  - `iT2::Number`: ``1 / T^2``.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `bdvariance_risk`: The risk expression.

# Related

  - [`set_brownian_distance_variance_constraints!`](@ref)
  - [`set_risk_constraints!`](@ref)
"""
function set_brownian_distance_risk_constraint!(model::JuMP.Model, ::QuadRiskExpr,
                                                Dt::MatNum, iT2::Number;
                                                prefix::Symbol = Symbol(""))
    return state_set!(model, prefix, :bdvariance_risk,
                      JuMP.@expression(model,
                                       iT2 * (LinearAlgebra.dot(Dt, Dt) + iT2 * sum(Dt)^2)))
end
function set_brownian_distance_risk_constraint!(model::JuMP.Model, ::RSOCRiskExpr,
                                                Dt::MatNum, iT2::Number;
                                                prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    tDt = state_set!(model, prefix, :tDt, JuMP.@variable(model))
    state_set!(model, prefix, :rsoc_Dt,
               JuMP.@constraint(model,
                                [sc * tDt;
                                 sc * 0.5;
                                 sc * vec(Dt)] in JuMP.RotatedSecondOrderCone()))
    # A quadratic `sum(Dt)^2` in the expression is rank one, and the bridge of a bound on the
    # risk to a second-order cone cannot factor it. The second cone keeps the expression affine.
    # It holds the mean row sum, so `tsum_Dt` is of the size of `tDt` and both reach the same
    # relative accuracy.
    tsum_Dt = state_set!(model, prefix, :tsum_Dt, JuMP.@variable(model))
    state_set!(model, prefix, :rsoc_sum_Dt,
               JuMP.@constraint(model,
                                [sc * tsum_Dt;
                                 sc * 0.5;
                                 sc * sum(Dt) / size(Dt, 1)] in
                                JuMP.RotatedSecondOrderCone()))
    return state_set!(model, prefix, :bdvariance_risk,
                      JuMP.@expression(model, iT2 * (tDt + tsum_Dt)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the [`BrownianDistanceVariance`](@ref) bound of the net portfolio returns to `model`, and return its risk expression.

`r.alg1` selects the risk expression, and `r.alg2` selects the rows that tie the distance matrix to the returns. When the objective minimises the risk, `bdvariance_risk` equals the functor of the measure at the solved weights.

The build is keyed on `prefix` alone, with no measure index. A second call with the same `prefix` uses the registered expression and builds no rows. The settings of the first [`BrownianDistanceVariance`](@ref) take effect, and a second measure logs a warning, because its bound and its scale have no effect.

# Algorithm

 1. When `bdvariance_risk` is registered under `prefix`, go to step 6.
 2. Read the net portfolio return series `net_X` with [`set_net_portfolio_returns!`](@ref), and its length `T`.
 3. Create `Dt`, and register `Dx`, the matrix of the pairwise differences of `net_X`.
 4. Register the risk expression of `r.alg1` with [`set_brownian_distance_risk_constraint!`](@ref).
 5. Add the rows of `r.alg2` with [`set_brownian_distance_variance_constraints!`](@ref).
 6. Add the bound in `r.settings` and the risk expression with [`set_unindexed_risk_bounds_and_expression!`](@ref), and return the risk expression.

# JuMP formulation

## Variables

  - `Dt`: created, the symmetric ``T \\times T`` matrix ``\\mathbf{D}``.

## Expressions

  - `Dx`: ``\\boldsymbol{\\Delta} = \\hat{\\boldsymbol{r}} \\boldsymbol{1}^\\intercal - \\boldsymbol{1} \\hat{\\boldsymbol{r}}^\\intercal``.

The two calls register `bdvariance_risk` and the rows, and their docstrings state them. Each name carries the namespace `prefix`.

Where:

  - ``\\mathbf{D}``: Symmetric matrix variable of the distances between the portfolio returns.
  - ``\\boldsymbol{\\Delta}``: Matrix of the differences ``\\Delta_{ij} = \\hat{r}_i - \\hat{r}_j``.
  - ``\\hat{\\boldsymbol{r}}``: The net portfolio returns `net_X`, ``T \\times 1``.
  - ``\\boldsymbol{1}``: Column vector of ones, ``T \\times 1``.
  - $(math_dict[:T])

## Relaxation

$(val_dict[:relax])

  - `bdvariance_risk` lies at or above ``\\mathrm{BDVar}(\\hat{\\boldsymbol{r}})``, and is tight when the objective minimises the risk. The two calls state the reasons.
  - ``\\mathrm{BDVar}(\\hat{\\boldsymbol{r}})`` lies at or above the distance variance of ``\\hat{\\boldsymbol{r}}``. It is tight only for a constant series.

# Arguments

  - $(arg_dict[:model])
  - `::Any`: The index of the measure in the risk vector. The method does not read it.
  - `r::BrownianDistanceVariance`: The measure, read for `alg1`, `alg2` and `settings`.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `bdvariance_risk`: The risk expression.

# Related

  - [`set_brownian_distance_variance_constraints!`](@ref)
  - [`set_brownian_distance_risk_constraint!`](@ref)
  - [`set_unindexed_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, ::Any, r::BrownianDistanceVariance,
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    bdvariance_risk = state_build!(model, prefix, :bdvariance_risk) do
        X = pr.X
        net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
        T = length(net_X)
        iT2 = inv(T^2)
        ovec = range(one(eltype(X)), one(eltype(X)); length = T)
        Dt = state_set!(model, prefix, :Dt, JuMP.@variable(model, [1:T, 1:T], Symmetric))
        Dx = state_set!(model, prefix, :Dx,
                        JuMP.@expression(model,
                                         net_X * transpose(ovec) - ovec * transpose(net_X)))
        # `set_brownian_distance_risk_constraint!` registers `:bdvariance_risk` itself;
        # the memo re-registers the same value.
        bdvariance_risk = set_brownian_distance_risk_constraint!(model, r.alg1, Dt, iT2;
                                                                 prefix = prefix)
        set_brownian_distance_variance_constraints!(model, r.alg2, Dt, Dx; prefix = prefix)
        return bdvariance_risk
    end
    set_unindexed_risk_bounds_and_expression!(model, opt, bdvariance_risk, r,
                                              :bdvariance_risk; prefix = prefix)
    return bdvariance_risk
end
