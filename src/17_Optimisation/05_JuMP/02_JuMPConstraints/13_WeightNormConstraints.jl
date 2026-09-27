"""
    set_weight_norm_2_constraints!(model::JuMP.Model, val::Number)
    set_weight_norm_2_constraints!(args...)

Bound the 2-norm of the weights above by `val`, a ceiling that spreads the portfolio over more assets as it falls.

The row multiplies `val` by the homogenisation variable `k`, so the ceiling holds on the weights the result reports: on ``\\boldsymbol{w} / k`` under [`MaximumRatio`](@ref), on the renormalised weights under [`RiskBudgeting`](@ref), and on ``\\boldsymbol{w}`` itself under every other objective, where ``k = 1``. The budget does not scale the ceiling. The builder takes a number. The caller-facing slot also takes an [`AbstractNormCeilingCalibrationAlgorithm`](@ref), which computes the ceiling from the universe the prior result carries, and [`assemble_jump_model!`](@ref) resolves it before it calls here. Any other argument, `nothing` included, adds nothing to the model.

# Mathematical definition

```math
\\begin{align}
\\lVert \\boldsymbol{w} \\rVert_{2} &\\leq \\mathrm{val} \\cdot k\\,.
\\end{align}
```

For reported weights that sum to one, the ceiling is a floor on the effective number of assets, and `val = 1 / sqrt(m)` states the floor ``m``:

```math
\\begin{align}
\\lVert \\boldsymbol{w} \\rVert_{2} \\leq \\frac{1}{\\sqrt{m}} &\\iff \\mathrm{ENA}_{2}(\\boldsymbol{w}) \\geq m\\,.
\\end{align}
```

Weights that sum to one have ``\\lVert \\boldsymbol{w} \\rVert_{2} \\geq 1 / \\sqrt{N}``, with equality at equal weights. A ceiling below ``1 / \\sqrt{N}`` admits none of them, and the model is infeasible.

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - ``\\mathrm{val}``: Ceiling on the 2-norm of the weights.
  - $(math_dict[:ENA_p])
  - $(math_dict[:m_ena])
  - $(math_dict[:N])

# JuMP formulation

## Variables

  - `w`: read. The portfolio weights.
  - `k`: read. The homogenisation variable.
  - `l2c`: created. The epigraph ``t`` of the 2-norm.

## Constraints

  - `cl2c_soc`: ``(s_c t, s_c \\boldsymbol{w}) \\in \\mathcal{K}_{2}``, that is ``t \\geq \\lVert \\boldsymbol{w} \\rVert_{2}``.
  - `cl2c`: ``s_c (t - \\mathrm{val} \\cdot k) \\leq 0``.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:K_q_norm])

# Arguments

  - $(arg_dict[:model])
  - `val::Number`: Ceiling on the 2-norm of the weights.

# Validation

  - `isfinite(val)` and `val > 0`, else a `DomainError` that names `l2c`.

# Returns

  - `nothing`.

# Related

  - [`set_weight_norm_p_constraints!`](@ref)
  - [`set_weight_norm_inf_constraints!`](@ref)
  - [`Num_NormCeilCal`](@ref)
  - [`EffectiveAssetFloor`](@ref)
  - [`number_effective_assets`](@ref)
  - [`JuMPOptimiser`](@ref)

# References

  - $(ref_dict[:jump2026cones])
"""
function set_weight_norm_2_constraints!(args...)
    return nothing
end
function set_weight_norm_2_constraints!(model::JuMP.Model, val::Number)
    assert_nonempty_gt0_finite_val(val, :l2c)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    JuMP.@variable(model, l2c)
    JuMP.@constraints(model, begin
                          cl2c_soc, [sc * l2c; sc * w] in JuMP.SecondOrderCone()
                          cl2c, sc * (l2c - val * k) <= 0
                      end)
    return nothing
end
"""
    set_weight_norm_p_constraints!(model::JuMP.Model, lps::LpReg_VecLpReg)
    set_weight_norm_p_constraints!(args...)

Bound the ``p``-norm of the weights above, once for each term of `lps`.

Each term is an [`LpRegularisation`](@ref) that serves as a ceiling: its `p` field is the norm order, and its `val` field is the ceiling. The row multiplies `val` by the homogenisation variable `k`, so the ceiling holds on the weights the result reports, as in [`set_weight_norm_2_constraints!`](@ref). [`norm_ceiling_factory`](@ref) refuses an [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref) in `val`, resolves an [`AbstractNormCeilingCalibrationAlgorithm`](@ref) against the prior result with the term's own norm order, and rebuilds the term. Every `val` this builder sees is therefore a number that the constructor of [`LpRegularisation`](@ref) checked. The entries carry names apart from those of [`set_lp_regularisation!`](@ref), so a model can carry an Lp penalty and a ``p``-norm ceiling at once. Any other argument, `nothing` included, adds nothing to the model.

# Mathematical definition

```math
\\begin{align}
\\lVert \\boldsymbol{w} \\rVert_{p_i} &\\leq \\mathrm{val}_i \\cdot k\\,, \\quad i = 1, \\ldots, n\\,.
\\end{align}
```

For reported weights that sum to one, each ceiling is a floor on the effective number of assets of order ``p_i``, and ``\\mathrm{val}_i = m^{1/p_i - 1}`` states the floor ``m``:

```math
\\begin{align}
\\lVert \\boldsymbol{w} \\rVert_{p} \\leq m^{1/p - 1} &\\iff \\mathrm{ENA}_{p}(\\boldsymbol{w}) \\geq m\\,.
\\end{align}
```

The exponent ``1/p - 1`` tends to ``-1`` as ``p`` grows, which gives the ceiling ``1 / m`` of [`set_weight_norm_inf_constraints!`](@ref). Weights that sum to one have ``\\lVert \\boldsymbol{w} \\rVert_{p} \\geq N^{1/p - 1}``, with equality at equal weights. A ceiling below ``N^{1/p - 1}`` admits none of them, and the model is infeasible.

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - ``p_i > 1``: Norm order of the ``i``-th term, its `p` field.
  - ``\\mathrm{val}_i``: Ceiling of the ``i``-th term, its `val` field.
  - ``n``: Number of terms in `lps`.
  - $(math_dict[:ENA_p])
  - $(math_dict[:m_ena])
  - $(math_dict[:N])

# JuMP formulation

## Variables

  - `w`: read. The portfolio weights.
  - `k`: read. The homogenisation variable.
  - `t_lpc_i`: created. The epigraph ``t_i`` of the ``p_i``-norm.
  - `r_lpc_i`: created. The ``N`` auxiliaries ``r_{i,j}`` of the ``i``-th term, one for each asset.

## Constraints

  - `clpc_i`: ``(s_c r_{i,j}, s_c t_i, s_c w_j) \\in \\mathcal{P}_{1/p_i}`` for ``j = 1, \\ldots, N``, that is ``r_{i,j} \\geq \\lvert w_j \\rvert^{p_i} / t_i^{p_i - 1}``.
  - `cslpc_i`: ``s_c \\left(\\sum_{j} r_{i,j} - t_i\\right) = 0``. With the rows of `clpc_i`, it gives ``t_i^{p_i} \\geq \\sum_{j} \\lvert w_j \\rvert^{p_i}``, that is ``t_i \\geq \\lVert \\boldsymbol{w} \\rVert_{p_i}``.
  - `clpc_bnd_i`: ``s_c (t_i - \\mathrm{val}_i \\cdot k) \\leq 0``.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:P_alpha_power])

# Arguments

  - $(arg_dict[:model])
  - `lps::LpReg_VecLpReg`: One or more ``p``-norm ceilings.

# Returns

  - `nothing`.

# Related

  - [`LpRegularisation`](@ref)
  - [`LpReg_VecLpReg`](@ref)
  - [`set_weight_norm_2_constraints!`](@ref)
  - [`set_weight_norm_inf_constraints!`](@ref)
  - [`norm_ceiling_factory`](@ref)
  - [`assert_norm_ceiling_role`](@ref)
  - [`Num_NormCeilCal`](@ref)
  - [`set_lp_regularisation!`](@ref): the Lp penalty that the same estimator states in the `lp` field.
  - [`JuMPOptimiser`](@ref)

# References

  - $(ref_dict[:jump2026cones])
"""
function set_weight_norm_p_constraints!(args...)
    return nothing
end
function set_weight_norm_p_constraints!(model::JuMP.Model, lps::LpReg_VecLpReg)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    N = length(w)
    for (i, lp) in enumerate(lps)
        val = lp.val
        p_inv = inv(lp.p)
        t_lpc, r_lpc = JuMP.@variables(model, begin
                                           ()
                                           [1:N]
                                       end)
        state_set!(model, Symbol(""), :t_lpc_, i, t_lpc)
        state_set!(model, Symbol(""), :r_lpc_, i, r_lpc)
        clpc, cslpc = JuMP.@constraints(model,
                                        begin
                                            [i = 1:N],
                                            [sc * r_lpc[i], sc * t_lpc, sc * w[i]] in
                                            JuMP.MOI.PowerCone(p_inv)
                                            sc * (sum(r_lpc) - t_lpc) == 0
                                        end)
        state_set!(model, Symbol(""), :clpc_, i, clpc)
        state_set!(model, Symbol(""), :cslpc_, i, cslpc)
        state_set!(model, Symbol(""), :clpc_bnd_, i,
                   JuMP.@constraint(model, sc * (t_lpc - val * k) <= 0))
    end
    return nothing
end
"""
    set_weight_norm_inf_constraints!(model::JuMP.Model, val::Number)
    set_weight_norm_inf_constraints!(args...)

Bound the largest absolute weight above by `val`, the ∞-norm ceiling of the weights.

The row multiplies `val` by the homogenisation variable `k`, so the ceiling holds on the weights the result reports, as in [`set_weight_norm_2_constraints!`](@ref). So `val = 0.2` keeps each reported weight within 20% of a portfolio that sums to one. The builder takes a number. The caller-facing slot also takes an [`AbstractNormCeilingCalibrationAlgorithm`](@ref), which computes the ceiling from the universe the prior result carries, and [`assemble_jump_model!`](@ref) resolves it before it calls here. Any other argument, `nothing` included, adds nothing to the model.

# Mathematical definition

```math
\\begin{align}
\\lVert \\boldsymbol{w} \\rVert_{\\infty} &\\leq \\mathrm{val} \\cdot k\\,.
\\end{align}
```

For reported weights that sum to one, `val = 1 / m` states the floor ``m`` on the effective number of assets of order ``\\infty``:

```math
\\begin{align}
\\lVert \\boldsymbol{w} \\rVert_{\\infty} \\leq \\frac{1}{m} &\\iff \\mathrm{ENA}_{\\infty}(\\boldsymbol{w}) \\geq m\\,.
\\end{align}
```

At least ``m`` of those weights are then non-zero, because ``n`` non-zero weights that sum to one have ``1 \\leq n \\lVert \\boldsymbol{w} \\rVert_{\\infty}``. Weights that sum to one have ``\\lVert \\boldsymbol{w} \\rVert_{\\infty} \\geq 1 / N``, with equality at equal weights. A ceiling below ``1 / N`` admits none of them, and the model is infeasible.

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - ``\\mathrm{val}``: Ceiling on the largest absolute weight.
  - $(math_dict[:ENA_p])
  - $(math_dict[:m_ena])
  - ``n``: Number of non-zero weights.
  - $(math_dict[:N])

# JuMP formulation

## Variables

  - `w`: read. The portfolio weights.
  - `k`: read. The homogenisation variable.
  - `t_linfc`: created. The epigraph ``t`` of the ∞-norm.

## Constraints

  - `clinfc_nic`: ``(s_c t, s_c \\boldsymbol{w}) \\in \\mathcal{K}_{\\infty}``, that is ``t \\geq \\lVert \\boldsymbol{w} \\rVert_{\\infty}``.
  - `clinfc`: ``s_c (t - \\mathrm{val} \\cdot k) \\leq 0``.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:K_q_norm])

# Arguments

  - $(arg_dict[:model])
  - `val::Number`: Ceiling on the largest absolute weight.

# Validation

  - `isfinite(val)` and `val > 0`, else a `DomainError` that names `linfc`.

# Returns

  - `nothing`.

# Related

  - [`set_weight_norm_2_constraints!`](@ref)
  - [`set_weight_norm_p_constraints!`](@ref)
  - [`Num_NormCeilCal`](@ref)
  - [`EffectiveAssetFloor`](@ref)
  - [`set_linf_regularisation!`](@ref): the ∞-norm penalty on the weights.
  - [`JuMPOptimiser`](@ref)
"""
function set_weight_norm_inf_constraints!(args...)
    return nothing
end
function set_weight_norm_inf_constraints!(model::JuMP.Model, val::Number)
    assert_nonempty_gt0_finite_val(val, :linfc)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    JuMP.@variable(model, t_linfc)
    JuMP.@constraint(model, clinfc_nic,
                     [sc * t_linfc;
                      sc * w] in JuMP.MOI.NormInfinityCone(1 + length(w)))
    JuMP.@constraint(model, clinfc, sc * (t_linfc - val * k) <= 0)
    return nothing
end
"""
    norm_ball_dual_norm_epigraph!(model::JuMP.Model, prefix::Symbol, i, x, p::Number)

Register an epigraph variable of the dual norm of `x`, and bound it below by that norm.

The dual order ``q`` picks the cone. At ``q = 2`` it is the second-order cone, as in [`set_weight_norm_2_constraints!`](@ref). At ``q = 1`` (``p = \\infty``) it is the norm-one cone, as in [`set_l1_regularisation!`](@ref). At ``q = \\infty`` (``p = 1``) it is the norm-infinity cone, as in [`set_weight_norm_inf_constraints!`](@ref). At every other order it is one power cone for each entry of `x`, as in [`set_weight_norm_p_constraints!`](@ref). The two norm-ball consumers call it, the mean builder on ``\\mathbf{L}^{\\intercal}\\boldsymbol{w}`` and the covariance builder on ``\\mathbf{L}^{\\intercal}\\operatorname{vec}(\\mathbf{W} + \\mathbf{E})``, so it takes `prefix` and `i` and registers every entry under both. The variable equals the dual norm only where the caller's objective pulls it down.

# Mathematical definition

```math
\\begin{align}
t &\\geq \\lVert \\boldsymbol{x} \\rVert_{q}\\,, \\\\
q &= \\frac{p}{p - 1}\\,.
\\end{align}
```

The dual order is ``q = 1`` at ``p = \\infty``, and ``q = \\infty`` at ``p = 1``.

Where:

  - ``t``: Epigraph of the dual norm.
  - ``\\boldsymbol{x}``: The affine expression whose dual norm is bounded.
  - ``p``: Norm order of the ball.
  - ``q``: Dual norm order of ``p``.

# JuMP formulation

## Variables

  - `t_nbucs_i`: created, under `prefix`. The epigraph ``t``.
  - `r_nbucs_i`: created, under `prefix`, on the power-cone route only. One auxiliary ``r_j`` for each entry of `x`.

## Constraints

  - `nbucs_cone_i`: ``(s_c t, s_c \\boldsymbol{x}) \\in \\mathcal{K}_{q}`` for ``q \\in \\{1, 2, \\infty\\}``. On the power-cone route it holds one row for each entry of `x`, ``(s_c r_j, s_c t, s_c x_j) \\in \\mathcal{P}_{1/q}``, that is ``r_j \\geq \\lvert x_j \\rvert^{q} / t^{q - 1}``.
  - `nbucs_cone_sum_i`: ``s_c \\left(\\sum_{j} r_j - t\\right) = 0``, on the power-cone route only. With the rows of `nbucs_cone_i`, it gives ``t^{q} \\geq \\sum_{j} \\lvert x_j \\rvert^{q}``.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:K_q_norm])
  - $(math_dict[:P_alpha_power])

# Arguments

  - $(arg_dict[:model])
  - `prefix`: Model State prefix that the entries are registered under.
  - `i`: Index of the term, which suffixes every name that the builder registers.
  - `x`: Affine expression, a vector of at least one entry.
  - `p::Number`: Norm order of the ball, `p >= 1`, with `Inf` admitted.

# Returns

  - `t_nbucs`: The epigraph variable.

# Related

  - [`NormBallUncertaintySet`](@ref)
  - [`dual_norm_order`](@ref)
  - [`set_ucs_return_constraints!`](@ref)
  - [`set_ucs_variance_risk!`](@ref)

# References

  - $(ref_dict[:jump2026cones])
"""
function norm_ball_dual_norm_epigraph!(model::JuMP.Model, prefix::Symbol, i, x, p::Number)
    sc = get_constraint_scale(model)
    q = dual_norm_order(p)
    t_nbucs = state_set!(model, prefix, :t_nbucs_, i, JuMP.@variable(model))
    if q == 2
        state_set!(model, prefix, :nbucs_cone_, i,
                   JuMP.@constraint(model,
                                    [sc * t_nbucs; sc * x] in JuMP.SecondOrderCone()))
    elseif isone(q)
        state_set!(model, prefix, :nbucs_cone_, i,
                   JuMP.@constraint(model,
                                    [sc * t_nbucs;
                                     sc * x] in JuMP.MOI.NormOneCone(1 + length(x))))
    elseif isinf(q)
        state_set!(model, prefix, :nbucs_cone_, i,
                   JuMP.@constraint(model,
                                    [sc * t_nbucs;
                                     sc * x] in JuMP.MOI.NormInfinityCone(1 + length(x))))
    else
        q_inv = inv(q)
        r_nbucs = state_set!(model, prefix, :r_nbucs_, i,
                             JuMP.@variable(model, [1:length(x)]))
        # One scalar row per entry rather than a JuMP container, which JET reads through
        # the container closure and reports a builtin call on.
        state_set!(model, prefix, :nbucs_cone_, i,
                   [JuMP.@constraint(model,
                                     [sc * r_nbucs[j], sc * t_nbucs, sc * x[j]] in
                                     JuMP.MOI.PowerCone(q_inv)) for j in eachindex(x)])
        state_set!(model, prefix, :nbucs_cone_sum_, i,
                   JuMP.@constraint(model, sc * (sum(r_nbucs) - t_nbucs) == 0))
    end
    return t_nbucs
end
