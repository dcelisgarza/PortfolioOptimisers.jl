"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add one distance variable for each pair of observations, and register the differences of the net portfolio returns of the pairs.

A pair is ``(i, j)`` with ``i > j``, so the model holds ``P = T (T - 1) / 2`` pairs, column by column of the strict lower triangle. The diagonal of the distance matrix is zero, and the upper triangle repeats the lower one, so neither gets a variable. The two pair formulations of [`set_brownian_distance_mean!`](@ref) call this function.

# JuMP formulation

## Variables

  - `Dt`: ``\\boldsymbol{d}``, the ``P`` distances of the pairs.

## Expressions

  - `Dx`: ``\\boldsymbol{\\delta}``, with ``\\delta_{p} = \\hat{r}_{i} - \\hat{r}_{j}`` for the pair ``p = (i, j)``.

Each name carries the namespace `prefix`.

Where:

  - ``\\hat{r}_{i}``: The net portfolio return at the period ``i``.
  - $(math_dict[:T])

# Arguments

  - $(arg_dict[:model])
  - `net_X::VecNum`: The net portfolio returns ``\\hat{\\boldsymbol{r}}``.
  - `T::Integer`: The number of observations.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `(Dt, Dx)`: The distance variables and the differences, both of length ``P``.

# Related

  - [`set_brownian_distance_mean!`](@ref)
"""
function set_brownian_distance_pairs!(model::JuMP.Model, net_X::VecNum, T::Integer;
                                      prefix::Symbol = Symbol(""))
    P = div(T * (T - 1), 2)
    I = [i for j in 1:(T - 1) for i in (j + 1):T]
    J = [j for j in 1:(T - 1) for i in (j + 1):T]
    Dt = state_set!(model, prefix, :Dt, JuMP.@variable(model, [1:P]))
    Dx = state_set!(model, prefix, :Dx,
                    JuMP.@expression(model, [p = 1:P], net_X[I[p]] - net_X[J[p]]))
    return Dt, Dx
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the rows of the mean pairwise distance of the net portfolio returns to `model`, and return the expression that bounds it from above.

The formulation selects the rows. The mean distance is ``\\bar{D} = \\frac{2}{T^{2}} \\sum_{i > j} |\\hat{r}_{i} - \\hat{r}_{j}|``, and it is also an ordered weighted average of the returns with nondecreasing weights, see [`brownian_distance_owa_weights`](@ref). So the two pair formulations and the two OWA formulations state the same function, and they give the same optimum. They differ in the number and the type of the rows. The pair formulations add ``O(T^{2})`` variables and rows, [`ExactOrderedWeightsArray`](@ref) adds ``2 T`` variables and ``T^{2}`` rows, and [`SortingNetworkOrderedWeightsArray`](@ref) adds ``O(T \\log^{2} T)`` variables and rows.

# JuMP formulation

## Variables

  - `Dt`: created by [`set_brownian_distance_pairs!`](@ref) under the two pair formulations.
  - `bdvariance_owa_a`, `bdvariance_owa_b`: under [`ExactOrderedWeightsArray`](@ref), the two ``T \\times 1`` variables ``\\boldsymbol{a}`` and ``\\boldsymbol{b}`` of the assignment dual.
  - `bdvariance_sowa_z`: under [`SortingNetworkOrderedWeightsArray`](@ref), the values that the comparators of the network write.

## Expressions

  - `bdvariance_md_ub`: the expression that bounds the mean distance, ``\\frac{2}{T^{2}} \\boldsymbol{1}^\\intercal \\boldsymbol{d}`` under the pair formulations, ``\\boldsymbol{1}^\\intercal (\\boldsymbol{a} + \\boldsymbol{b})`` under [`ExactOrderedWeightsArray`](@ref), and the risk expression of the network under [`SortingNetworkOrderedWeightsArray`](@ref).

## Constraints

  - `cbdvariance_noc`: under [`NormOneConeBrownianDistanceVariance`](@ref), ``(s_c d_{p},\\, s_c \\delta_{p}) \\in \\mathcal{K}_{1}``, which is ``d_{p} \\geq |\\delta_{p}|``, for each pair ``p``.
  - `cp_bdvariance`, `cn_bdvariance`: under [`IneqBrownianDistanceVariance`](@ref), ``s_c (\\boldsymbol{d} - \\boldsymbol{\\delta}) \\geq 0`` and ``s_c (\\boldsymbol{d} + \\boldsymbol{\\delta}) \\geq 0``.
  - `cbdvariance_owa`: under [`ExactOrderedWeightsArray`](@ref), ``s_c (\\hat{r}_{i} \\omega_{j} - a_{j} - b_{i}) \\leq 0`` for each ``i`` and ``j``.
  - `cbdvariance_sowa_eq`, `cbdvariance_sowa_a`, `cbdvariance_sowa_b`: under [`SortingNetworkOrderedWeightsArray`](@ref), the three blocks of rows of [`set_sorting_network_owa!`](@ref).

The two OWA formulations read the net returns from the `owa` variable of [`set_owa_constraints!`](@ref), which a [`OrderedWeightsArray`](@ref) in the same model shares.

Each name carries the namespace `prefix`.

Where:

  - ``\\boldsymbol{d}``, ``\\boldsymbol{\\delta}``: The distances and the differences of the pairs, from [`set_brownian_distance_pairs!`](@ref).
  - ``\\omega_{j}``: The OWA weight of the ``j``-th smallest return, from [`brownian_distance_owa_weights`](@ref).
  - ``\\hat{r}_{i}``: The net portfolio return at the period ``i``.
  - $(math_dict[:T])
  - $(math_dict[:K_q_norm])
  - $(math_dict[:sc_scale])

## Relaxation

$(val_dict[:relax])

  - Each formulation bounds ``\\bar{D}`` from above, and the bound is tight at the minimum of the returned expression. The risk expression of [`set_brownian_distance_risk_constraint!`](@ref) increases in it, so the bound is tight when the objective minimises the risk.
  - Under a bound on the risk and an objective that does not minimise it, the expression can exceed ``\\bar{D}``. The true risk is then at or below the risk of the model, so the bound still holds the value of the functor of [`BrownianDistanceVariance`](@ref).

# Arguments

  - $(arg_dict[:model])
  - The formulation, a member of [`BDVarMeanDistanceFormulations`](@ref).
  - `X::MatNum`: The asset returns matrix. The OWA formulations read it, and the pair formulations do not.
  - `net_X::VecNum`: The net portfolio returns ``\\hat{\\boldsymbol{r}}``. The pair formulations read it.
  - `T::Integer`: The number of observations.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `bdvariance_md_ub`: The affine expression that bounds the mean distance from above.

# Related

  - [`set_brownian_distance_pairs!`](@ref)
  - [`brownian_distance_owa_weights`](@ref)
  - [`set_sorting_network_owa!`](@ref)
  - [`set_risk_constraints!`](@ref)

# References

  - $(ref_dict[:goemans2015])
"""
function set_brownian_distance_mean!(model::JuMP.Model,
                                     ::NormOneConeBrownianDistanceVariance, ::MatNum,
                                     net_X::VecNum, T::Integer; prefix::Symbol = Symbol(""))
    Dt, Dx = set_brownian_distance_pairs!(model, net_X, T; prefix = prefix)
    P = length(Dt)
    sc = get_constraint_scale(model)
    state_set!(model, prefix, :cbdvariance_noc,
               JuMP.@constraint(model, [p = 1:P],
                                [sc * Dt[p]; sc * Dx[p]] in JuMP.MOI.NormOneCone(2)))
    c2 = 2 * inv(T^2)
    return state_set!(model, prefix, :bdvariance_md_ub,
                      JuMP.@expression(model, c2 * sum(Dt)))
end
function set_brownian_distance_mean!(model::JuMP.Model, ::IneqBrownianDistanceVariance,
                                     ::MatNum, net_X::VecNum, T::Integer;
                                     prefix::Symbol = Symbol(""))
    Dt, Dx = set_brownian_distance_pairs!(model, net_X, T; prefix = prefix)
    sc = get_constraint_scale(model)
    state_set!(model, prefix, :cp_bdvariance,
               JuMP.@constraint(model, sc * (Dt - Dx) in JuMP.Nonnegatives()))
    state_set!(model, prefix, :cn_bdvariance,
               JuMP.@constraint(model, sc * (Dt + Dx) in JuMP.Nonnegatives()))
    c2 = 2 * inv(T^2)
    return state_set!(model, prefix, :bdvariance_md_ub,
                      JuMP.@expression(model, c2 * sum(Dt)))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return the OWA weights whose ordered weighted average of the returns is their mean pairwise distance.

The weights increase in ``t``, so the OWA is convex, and the two OWA formulations of [`set_brownian_distance_mean!`](@ref) are exact for them.

# Mathematical definition

```math
\\begin{align}
\\bar{D} &= \\frac{1}{T^{2}} \\sum_{i, j} |\\hat{r}_{i} - \\hat{r}_{j}| = \\frac{2}{T^{2}} \\sum_{i > j} |\\hat{r}_{i} - \\hat{r}_{j}| = \\sum_{t = 1}^{T} \\omega_{t}\\, \\hat{r}_{(t)}\\,, \\\\
\\omega_{t} &= \\frac{2 (2 t - T - 1)}{T^{2}}\\,.
\\end{align}
```

Where:

  - ``\\bar{D}``: The mean pairwise distance of the net portfolio returns.
  - ``\\hat{r}_{(t)}``: The ``t``-th smallest net portfolio return. It is the larger value of ``t - 1`` pairs and the smaller value of ``T - t`` pairs, which gives the factor ``2 t - T - 1``.
  - ``\\omega_{t}``: The OWA weight of ``\\hat{r}_{(t)}``.
  - $(math_dict[:T])

# Arguments

  - `X::MatNum`: The asset returns matrix. The function reads its element type only.
  - `T::Integer`: The number of observations.

# Returns

  - `owa_w::Vector`: The ``T`` weights ``\\omega_{t}``, of the element type of `X`.

# Related

  - [`set_brownian_distance_mean!`](@ref)
  - [`owa_gmd`](@ref)
"""
function brownian_distance_owa_weights(X::MatNum, T::Integer)
    c2 = 2 / (T^2 * one(eltype(X)))
    return [c2 * (2 * t - T - 1) for t in 1:T]
end
function set_brownian_distance_mean!(model::JuMP.Model, ::ExactOrderedWeightsArray,
                                     X::MatNum, ::VecNum, T::Integer;
                                     prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    owa = set_owa_constraints!(model, X; prefix = prefix)
    owa_w = brownian_distance_owa_weights(X, T)
    ovec = range(one(eltype(X)), one(eltype(X)); length = T)
    owa_a = state_set!(model, prefix, :bdvariance_owa_a, JuMP.@variable(model, [1:T]))
    owa_b = state_set!(model, prefix, :bdvariance_owa_b, JuMP.@variable(model, [1:T]))
    state_set!(model, prefix, :cbdvariance_owa,
               JuMP.@constraint(model,
                                sc * (owa * transpose(owa_w) - ovec * transpose(owa_a) -
                                      owa_b * transpose(ovec)) in JuMP.Nonpositives()))
    return state_set!(model, prefix, :bdvariance_md_ub,
                      JuMP.@expression(model, sum(owa_a) + sum(owa_b)))
end
function set_brownian_distance_mean!(model::JuMP.Model,
                                     alg::SortingNetworkOrderedWeightsArray, X::MatNum,
                                     ::VecNum, T::Integer; prefix::Symbol = Symbol(""))
    owa_w = brownian_distance_owa_weights(X, T)
    net = set_sorting_network_owa!(model, alg, X, owa_w; prefix = prefix)
    state_set!(model, prefix, :bdvariance_sowa_z, net.z)
    state_set!(model, prefix, :cbdvariance_sowa_eq, net.c_eq)
    state_set!(model, prefix, :cbdvariance_sowa_a, net.c_a)
    state_set!(model, prefix, :cbdvariance_sowa_b, net.c_b)
    return state_set!(model, prefix, :bdvariance_md_ub, net.risk)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Register the risk expression of [`BrownianDistanceVariance`](@ref), the sum of the squares of the scaled centred returns and of the mean distance, and return it.

The expression is equation 9 of [bdvar](@cite) in its closed form, ``2 \\sigma^{2} + \\bar{D}^{2}``, which the docstring of [`BrownianDistanceVariance`](@ref) derives. The variance term reads the returns directly, so it needs no distance variable. The formulation states the two squares.

# JuMP formulation

## Variables

  - `t_bdvariance_var`, `t_bdvariance_md`: created under [`RSOCRiskExpr`](@ref), the scalars ``t`` and ``u``.
  - `bdvariance_dev`: created under [`SquaredSOCRiskExpr`](@ref) by [`set_brownian_distance_deviation!`](@ref).

## Expressions

  - `bdvariance_risk`: ``\\lVert \\boldsymbol{c} \\rVert_{2}^{2} + m^{2}`` under [`QuadRiskExpr`](@ref), ``t + u`` under [`RSOCRiskExpr`](@ref), and ``\\sigma_{\\mathrm{BD}}^{2}`` under [`SquaredSOCRiskExpr`](@ref).

## Constraints

  - `rsoc_bdvariance_var`: under [`RSOCRiskExpr`](@ref), ``(s_c t,\\, s_c / 2,\\, s_c \\boldsymbol{c}) \\in \\mathcal{K}_{r}``, which is ``t \\geq \\lVert \\boldsymbol{c} \\rVert_{2}^{2}``.
  - `rsoc_bdvariance_md`: under [`RSOCRiskExpr`](@ref), ``(s_c u,\\, s_c / 2,\\, s_c m) \\in \\mathcal{K}_{r}``, which is ``u \\geq m^{2}``.

Each name carries the namespace `prefix`.

Where:

  - ``\\boldsymbol{c} = \\sqrt{2 / T}\\, (\\hat{\\boldsymbol{r}} - \\mu \\boldsymbol{1})``: The scaled centred returns `xc`, so ``\\lVert \\boldsymbol{c} \\rVert_{2}^{2} = 2 \\sigma^{2}``.
  - ``m``: The mean distance variable `md`.
  - ``\\sigma_{\\mathrm{BD}}``: The deviation, ``\\sigma_{\\mathrm{BD}} \\geq \\lVert (\\boldsymbol{c}, m) \\rVert_{2}``.
  - ``t``, ``u``: Epigraph variables of the two squares.
  - ``\\mathcal{K}_{r} = \\{(a, b, \\boldsymbol{x}) : 2 a b \\geq \\lVert \\boldsymbol{x} \\rVert_2^2,\\, a \\geq 0,\\, b \\geq 0\\}``: Rotated second-order cone. The model states it with a [JuMP.RotatedSecondOrderCone](https://jump.dev/JuMP.jl/stable/tutorials/conic/tips_and_tricks/#Rotated-Second-Order-Cone).
  - $(math_dict[:T])
  - $(math_dict[:sc_scale])

## Relaxation

$(val_dict[:relax])

  - `bdvariance_risk` lies at or above ``\\mathrm{BDVar}(\\hat{\\boldsymbol{r}})``, because ``m`` lies at or above the mean distance, see [`set_brownian_distance_mean!`](@ref). The variance term is exact.
  - Under [`RSOCRiskExpr`](@ref) and [`SquaredSOCRiskExpr`](@ref) the epigraph variables lie at or above their squares. All three are tight when the objective minimises the risk.

# Arguments

  - $(arg_dict[:model])
  - `::QuadRiskExpr`, `::RSOCRiskExpr` or `::SquaredSOCRiskExpr`: The risk expression formulation.
  - `xc::VecNum`: ``\\boldsymbol{c}``.
  - `md::JuMP.AbstractJuMPScalar`: ``m``.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `bdvariance_risk`: The risk expression.

# Related

  - [`set_brownian_distance_mean!`](@ref)
  - [`set_brownian_distance_deviation!`](@ref)
  - [`set_risk_constraints!`](@ref)

# References

  - $(ref_dict[:bdvar])
"""
function set_brownian_distance_risk_constraint!(model::JuMP.Model, ::QuadRiskExpr,
                                                xc::VecNum, md::JuMP.AbstractJuMPScalar;
                                                prefix::Symbol = Symbol(""))
    return state_set!(model, prefix, :bdvariance_risk,
                      JuMP.@expression(model, LinearAlgebra.dot(xc, xc) + md^2))
end
function set_brownian_distance_risk_constraint!(model::JuMP.Model, ::RSOCRiskExpr,
                                                xc::VecNum, md::JuMP.AbstractJuMPScalar;
                                                prefix::Symbol = Symbol(""))
    sc = get_constraint_scale(model)
    t_var = state_set!(model, prefix, :t_bdvariance_var, JuMP.@variable(model))
    state_set!(model, prefix, :rsoc_bdvariance_var,
               JuMP.@constraint(model,
                                [sc * t_var; sc * 0.5; sc * xc] in
                                JuMP.RotatedSecondOrderCone()))
    t_md = state_set!(model, prefix, :t_bdvariance_md, JuMP.@variable(model))
    state_set!(model, prefix, :rsoc_bdvariance_md,
               JuMP.@constraint(model,
                                [sc * t_md, sc * 0.5, sc * md] in
                                JuMP.RotatedSecondOrderCone()))
    return state_set!(model, prefix, :bdvariance_risk,
                      JuMP.@expression(model, t_var + t_md))
end
function set_brownian_distance_risk_constraint!(model::JuMP.Model, ::SquaredSOCRiskExpr,
                                                xc::VecNum, md::JuMP.AbstractJuMPScalar;
                                                prefix::Symbol = Symbol(""))
    dev = set_brownian_distance_deviation!(model, xc, md; prefix = prefix)
    return state_set!(model, prefix, :bdvariance_risk, JuMP.@expression(model, dev^2))
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the square root of the risk expression of [`BrownianDistanceVariance`](@ref) to `model` as one second-order cone, and return it.

The risk is the square of the norm of one vector, the scaled centred returns and the mean distance. So one cone of dimension ``T + 2`` bounds its square root. The deviation has degree one in the weights and ``k``, so the bound of the measure acts on it, and [`SquaredSOCRiskExpr`](@ref) squares it to give the risk. A second call with the same `prefix` returns the registered variable and adds no cone.

# JuMP formulation

## Variables

  - `bdvariance_dev`: ``\\sigma_{\\mathrm{BD}}``, the deviation.

## Constraints

  - `cbdvariance_dev_soc`: ``(s_c \\sigma_{\\mathrm{BD}},\\, s_c \\boldsymbol{c},\\, s_c m) \\in \\mathcal{K}_{2}``, which is ``\\sigma_{\\mathrm{BD}} \\geq \\sqrt{\\lVert \\boldsymbol{c} \\rVert_{2}^{2} + m^{2}}``.

Each name carries the namespace `prefix`.

Where:

  - ``\\boldsymbol{c}``, ``m``: The scaled centred returns and the mean distance variable, see [`set_brownian_distance_risk_constraint!`](@ref).
  - $(math_dict[:K_q_norm])
  - $(math_dict[:sc_scale])

## Relaxation

$(val_dict[:relax])

  - `bdvariance_dev` lies at or above the square root of the risk expression. A bound on it is exact, because the weights satisfy it when the square root of their risk is at most the bound.

# Arguments

  - $(arg_dict[:model])
  - `xc::VecNum`: ``\\boldsymbol{c}``.
  - `md::JuMP.AbstractJuMPScalar`: ``m``.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `bdvariance_dev`: The deviation variable.

# Related

  - [`set_brownian_distance_risk_constraint!`](@ref)
  - [`set_brownian_distance_bounds_expr!`](@ref)
"""
function set_brownian_distance_deviation!(model::JuMP.Model, xc::VecNum,
                                          md::JuMP.AbstractJuMPScalar;
                                          prefix::Symbol = Symbol(""))
    return state_build!(model, prefix, :bdvariance_dev) do
        sc = get_constraint_scale(model)
        dev = JuMP.@variable(model)
        state_set!(model, prefix, :cbdvariance_dev_soc,
                   JuMP.@constraint(model,
                                    [sc * dev; sc * xc; sc * md] in JuMP.SecondOrderCone()))
        return dev
    end
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the [`BrownianDistanceVariance`](@ref) bound of the net portfolio returns to `model`, and return its risk expression.

`r.alg2` selects the rows of the mean distance, and `r.alg1` selects the risk expression. When the objective minimises the risk, `bdvariance_risk` equals the functor of the measure at the solved weights.

The build is keyed on `prefix` alone, with no measure index. A second call with the same `prefix` uses the registered expression and builds no rows. The settings of the first [`BrownianDistanceVariance`](@ref) take effect, and a second measure logs a warning, because its bound and its scale have no effect.

# Algorithm

 1. When `bdvariance_risk` is registered under `prefix`, go to step 6.
 2. Read the net portfolio return series `net_X` with [`set_net_portfolio_returns!`](@ref), and the number of observations `T` from `pr.X`.
 3. Add the rows of the mean distance of `r.alg2` with [`set_brownian_distance_mean!`](@ref), and hold the expression that they return in the variable `md`.
 4. Hold the mean of `net_X` in the variable `mu`, and register the scaled centred returns `xc`.
 5. Register the risk expression of `r.alg1` with [`set_brownian_distance_risk_constraint!`](@ref).
 6. Convert the bound in `r.settings` to the units of a deviation with [`variance_risk_bounds_val`](@ref), and take the expression that it acts on from [`set_brownian_distance_bounds_expr!`](@ref). The risk expression has degree two in the weights, and the row of [`set_risk_upper_bound!`](@ref) multiplies the bound by ``k``, so under a ratio objective the bound holds only on an expression of degree one.
 7. Add the bound on the deviation and the risk expression with [`set_unindexed_risk_bounds_and_expression!`](@ref), and return the risk expression.

# JuMP formulation

## Variables

  - `bdvariance_md`: ``m``, the mean distance.
  - `bdvariance_mu`: ``\\mu``, the mean of the net portfolio returns.
  - `bdvariance_dev`: created by [`set_brownian_distance_deviation!`](@ref) when `r.settings` holds a bound or when `r.alg1` is [`SquaredSOCRiskExpr`](@ref), the square root of the risk expression.

## Expressions

  - `bdvariance_xc`: ``\\boldsymbol{c} = \\sqrt{2 / T}\\, (\\hat{\\boldsymbol{r}} - \\mu \\boldsymbol{1})``.

## Constraints

  - `cbdvariance_md`: ``s_c (m - \\bar{D}_{\\mathrm{alg}}) = 0``.
  - `cbdvariance_mu`: ``s_c (\\mu - \\boldsymbol{1}^\\intercal \\hat{\\boldsymbol{r}} / T) = 0``.

The two calls of steps 3 and 5 register their rows and `bdvariance_risk`, and their docstrings state them. Each name carries the namespace `prefix`.

Where:

  - ``\\bar{D}_{\\mathrm{alg}}``: The expression of [`set_brownian_distance_mean!`](@ref), which bounds the mean distance from above.
  - ``\\hat{\\boldsymbol{r}}``: The net portfolio returns `net_X`, ``T \\times 1``.
  - ``\\boldsymbol{1}``: Column vector of ones, ``T \\times 1``.
  - $(math_dict[:T])
  - $(math_dict[:sc_scale])

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

  - [`set_brownian_distance_mean!`](@ref)
  - [`set_brownian_distance_risk_constraint!`](@ref)
  - [`set_unindexed_risk_bounds_and_expression!`](@ref)
"""
function set_risk_constraints!(model::JuMP.Model, ::Any, r::BrownianDistanceVariance,
                               opt::RiskConstraintOwner, pr::AbstractPriorResult, args...;
                               prefix::Symbol = Symbol(""), kwargs...)
    bdvariance_risk = state_build!(model, prefix, :bdvariance_risk) do
        X = pr.X
        T = size(X, 1)
        sc = get_constraint_scale(model)
        net_X = set_net_portfolio_returns!(model, X; prefix = prefix)
        md_expr = set_brownian_distance_mean!(model, r.alg2, X, net_X, T; prefix = prefix)
        md = state_set!(model, prefix, :bdvariance_md, JuMP.@variable(model))
        state_set!(model, prefix, :cbdvariance_md,
                   JuMP.@constraint(model, sc * (md - md_expr) == 0))
        iT = inv(T)
        mu = state_set!(model, prefix, :bdvariance_mu, JuMP.@variable(model))
        state_set!(model, prefix, :cbdvariance_mu,
                   JuMP.@constraint(model, sc * (mu - iT * sum(net_X)) == 0))
        c = sqrt(2 * iT)
        xc = state_set!(model, prefix, :bdvariance_xc,
                        JuMP.@expression(model, c * (net_X .- mu)))
        # `set_brownian_distance_risk_constraint!` registers `:bdvariance_risk` itself;
        # the memo re-registers the same value.
        return set_brownian_distance_risk_constraint!(model, r.alg1, xc, md;
                                                      prefix = prefix)
    end
    ub = variance_risk_bounds_val(SquareRootBound(), r.settings.ub)
    r_expr_ub = set_brownian_distance_bounds_expr!(model, ub, bdvariance_risk;
                                                   prefix = prefix)
    set_unindexed_risk_bounds_and_expression!(model, opt, bdvariance_risk, r,
                                              :bdvariance_risk; prefix = prefix,
                                              r_expr_ub = r_expr_ub, ub = ub)
    return bdvariance_risk
end
"""
    set_brownian_distance_bounds_expr!(model::JuMP.Model, ::Nothing, bdvariance_risk; prefix::Symbol = Symbol("")) -> bdvariance_risk
    set_brownian_distance_bounds_expr!(model::JuMP.Model, ::RkRtBounds, bdvariance_risk; prefix::Symbol = Symbol("")) -> bdvariance_dev

Return the expression that the bound of a [`BrownianDistanceVariance`](@ref) acts on, and build it when the measure has a bound.

Without a bound, the method returns the risk expression and builds nothing. With a bound, it returns the deviation of [`set_brownian_distance_deviation!`](@ref), which reads the registered `bdvariance_xc` and `bdvariance_md`. Under [`SquaredSOCRiskExpr`](@ref) the risk expression already built the deviation, so the bound adds no cone. The bound must be in the units of a deviation, as [`variance_risk_bounds_val`](@ref) gives it.

# Arguments

  - $(arg_dict[:model])
  - `ub`: The bound in the units of a deviation, or `nothing`. The method reads only its type.
  - `bdvariance_risk`: The risk expression.

# Keyword arguments

  - `prefix::Symbol`: Model State namespace of the build (default: empty, i.e. the bare key).

# Returns

  - `bdvariance_risk` without a bound.
  - `bdvariance_dev` with a bound: the deviation variable.

# Related

  - [`set_risk_constraints!`](@ref)
  - [`set_brownian_distance_deviation!`](@ref)
  - [`variance_risk_bounds_val`](@ref)
  - [`variance_risk_bounds_expr`](@ref)
"""
function set_brownian_distance_bounds_expr!(::JuMP.Model, ::Nothing,
                                            bdvariance_risk::JuMP.AbstractJuMPScalar;
                                            prefix::Symbol = Symbol(""))
    return bdvariance_risk
end
function set_brownian_distance_bounds_expr!(model::JuMP.Model, ::RkRtBounds,
                                            ::JuMP.AbstractJuMPScalar;
                                            prefix::Symbol = Symbol(""))
    return set_brownian_distance_deviation!(model, state_get(model, prefix, :bdvariance_xc),
                                            state_get(model, prefix, :bdvariance_md);
                                            prefix = prefix)
end
