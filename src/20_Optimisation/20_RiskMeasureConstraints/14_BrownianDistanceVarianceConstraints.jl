"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add auxiliary BDV constraints linking the symmetric distance matrix `Dt` to portfolio-return
differences `Dx`.

The `NormOneConeBrownianDistanceVariance` overload adds per-element L1-norm cone constraints
`[sc * Dt[i,j]; sc * Dx[i,j]] in NormOneCone(2)` for each upper-triangular entry. The
`IneqBrownianDistanceVariance` overload adds global non-negativity constraints on `Dt - Dx`
and `Dt + Dx`.

# Mathematical definition

```math
\\begin{align}
D_t(i,j) &\\geq |\\hat{r}_i - \\hat{r}_j|\\,, \\\\
\\mathrm{BDV}(\\boldsymbol{w}) &= \\frac{1}{T^2}\\sum_{i,j} D_t(i,j)^2 - \\left(\\frac{1}{T^2}\\sum_{i,j} D_t(i,j)\\right)^2\\,.
\\end{align}
```

Where:

  - ``D_t(i,j)``: Distance auxiliary variable between returns at times ``i`` and ``j``.
  - ``\\hat{r}_i``, ``\\hat{r}_j``: Portfolio returns at times ``i`` and ``j``.
  - ``\\mathrm{BDV}(\\boldsymbol{w})``: Brownian distance variance.
  - $(math_dict[:T])

# Arguments

  - $(arg_dict[:model])
  - `Dt::MatNum`: Symmetric JuMP matrix variable for absolute distances.
  - `Dx::MatNum`: JuMP expression matrix for portfolio-return pairwise differences.

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

Build the Brownian distance variance risk expression.

The `QuadRiskExpr` overload encodes BDV as a quadratic dot product of `Dt` plus a squared
sum term. The `RSOCRiskExpr` overload introduces a scalar variable `tDt` bounded by a
rotated second-order cone on `vec(Dt)`, then builds the same expression with `tDt` in place
of the quadratic dot product.

# Arguments

  - $(arg_dict[:model])
  - `Dt::MatNum`: Symmetric distance matrix variable.
  - `iT2::Number`: Inverse square of the number of observations (`1 / T^2`).

# Returns

  - `bdvariance_risk`: JuMP expression for the Brownian distance variance risk.

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
                                 0.5;
                                 sc * vec(Dt)] in JuMP.RotatedSecondOrderCone()))
    return state_set!(model, prefix, :bdvariance_risk,
                      JuMP.@expression(model, iT2 * (tDt + iT2 * sum(Dt)^2)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add Brownian distance variance risk constraints to `model`.

Constructs the pairwise return difference matrix `Dx`, introduces a symmetric variable matrix
`Dt`, builds the BDV risk expression via [`set_brownian_distance_risk_constraint!`](@ref),
and links `Dt` to `Dx` via [`set_brownian_distance_variance_constraints!`](@ref). Returns
the existing expression if already present.

# Arguments

  - $(arg_dict[:model])
  - `r::BrownianDistanceVariance`: BDV risk measure instance.
  - $(arg_dict[:opt_rjumpe])
  - $(arg_dict[:pr_X])

# Returns

  - `nothing`.

# Related

  - [`set_brownian_distance_variance_constraints!`](@ref)
  - [`set_brownian_distance_risk_constraint!`](@ref)
  - [`set_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, ::Any, r::BrownianDistanceVariance,
                               opt::RiskJuMPOptimisationEstimator, pr::AbstractPriorResult,
                               args...; prefix::Symbol = Symbol(""), kwargs...)
    return state_build!(model, prefix, :bdvariance_risk) do
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
        set_risk_bounds_and_expression!(model, opt, bdvariance_risk, r.settings,
                                        :bdvariance_risk; prefix = prefix)
        return bdvariance_risk
    end
end
