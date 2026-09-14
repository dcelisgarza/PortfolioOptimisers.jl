"""
    set_weight_norm_2_constraints!(model::JuMP.Model, val::Number)
    set_weight_norm_2_constraints!(args...)

Constrain the 2-norm of the weights.

`val` is a direct upper bound on ``\\lVert \\boldsymbol{w} \\rVert_2``, expressed relative to the budget `k`: the constraint is ``\\lVert \\boldsymbol{w} \\rVert_2 \\leq \\mathrm{val} \\cdot k``. Smaller `val` forces the weights to spread more evenly across the assets.

The builder takes a number. The caller-facing slot also takes an [`AbstractNormCeilingCalibrationAlgorithm`](@ref), which computes the ceiling from the universe the prior result carries, and [`assemble_jump_model!`](@ref) resolves it before it calls here.

# Mathematical definition

```math
\\begin{align}
\\mathrm{l2c} &\\geq \\lVert \\boldsymbol{w} \\rVert_2\\,, \\\\
\\mathrm{l2c} &\\leq \\mathrm{val} \\cdot k\\,.
\\end{align}
```

Where:

  - ``\\mathrm{l2c}``: Auxiliary variable upper-bounding ``\\lVert \\boldsymbol{w} \\rVert_2``.
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - ``\\mathrm{val}``: Upper bound on the 2-norm of the weights.

# Diversification interpretation

The 2-norm and the effective number of assets are reciprocally related: for a fully invested portfolio (``k = 1``), ``\\mathrm{ENA}(\\boldsymbol{w}) = 1 / \\lVert \\boldsymbol{w} \\rVert_2^2``. To require **at least `m` effective assets**, set `val = 1 / sqrt(m)`:

```math
\\begin{align}
\\lVert \\boldsymbol{w} \\rVert_2 \\leq \\frac{1}{\\sqrt{m}} \\iff \\mathrm{ENA}(\\boldsymbol{w}) = \\frac{1}{\\lVert \\boldsymbol{w} \\rVert_2^2} \\geq m\\,.
\\end{align}
```

# Arguments

  - $(arg_dict[:model])
  - `val::Number`: Upper bound on the 2-norm of the weights.

# Returns

  - `nothing`.

# Details

  - `val::Number`: Introduces the auxiliary variable `l2c`, bounds it below by ``\\lVert \\boldsymbol{w} \\rVert_2`` with a `SecondOrderCone` constraint, and adds the linear constraint `l2c <= val * k`.
  - `args...`: No-op, used when no 2-norm weight constraint is configured.

# Related

  - [`set_weight_norm_p_constraints!`](@ref)
  - [`set_weight_norm_inf_constraints!`](@ref)
  - [`Num_NormCeilCal`](@ref)
  - [`EffectiveAssetFloor`](@ref)
  - [`number_effective_assets`](@ref)
  - [`JuMPOptimiser`](@ref)
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

Constrain the p-norm of the weights.

Generalises [`set_weight_norm_2_constraints!`](@ref) to an arbitrary norm order ``p > 1``. Each term supplies its own norm order and bound, so several may be imposed at once.

Each term is an [`LpRegularisation`](@ref), reused here as a constraint rather than a penalty: its `p` field is the norm order, and its `val` field is a direct upper bound on ``\\lVert \\boldsymbol{w} \\rVert_p``, expressed relative to the budget `k`. Smaller `val` forces a more evenly spread portfolio.

That reuse is why `val` carries two readings and one bound. Here it is a ceiling, so [`norm_ceiling_factory`](@ref) refuses an [`AbstractAmbiguityRadiusCalibrationAlgorithm`](@ref) in it, resolves an [`AbstractNormCeilingCalibrationAlgorithm`](@ref) against the prior result, and hands each term its own norm order first. Every `val` this builder sees is therefore a number.

# Mathematical definition

```math
\\begin{align}
\\mathrm{t}_{p,\\, i} &\\geq \\lVert \\boldsymbol{w} \\rVert_{p_i}\\,, \\\\
\\mathrm{t}_{p,\\, i} &\\leq \\mathrm{val}_i \\cdot k\\,.
\\end{align}
```

Where:

  - ``\\mathrm{t}_{p,\\, i}``: Auxiliary variable upper-bounding ``\\lVert \\boldsymbol{w} \\rVert_{p_i}``.
  - ``p_i``: Norm order of the ``i``-th term, its `p` field.
  - ``\\mathrm{val}_i``: Upper bound on the ``p_i``-norm of the weights, its `val` field.
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])

# Diversification interpretation

For a fully invested portfolio (``k = 1``), the order-``p`` effective number of assets is ``\\mathrm{ENA}_p(\\boldsymbol{w}) = \\left(\\sum_i \\lvert w_i \\rvert^p\\right)^{1/(1 - p)}``. To require **at least `m` order-``p`` effective assets**, set `val = m^(1/p - 1)`:

```math
\\begin{align}
\\lVert \\boldsymbol{w} \\rVert_p \\leq m^{1/p - 1} \\iff \\mathrm{ENA}_p(\\boldsymbol{w}) = \\left(\\sum_i \\lvert w_i \\rvert^p\\right)^{\\frac{1}{1 - p}} \\geq m\\,.
\\end{align}
```

This is [`number_effective_assets`](@ref) taken to an arbitrary order. At ``p = 2`` the two are the same number, and at every order an equal-weight portfolio over ``m`` assets reports exactly ``m``. The exponent is also ``-1/q`` for the conjugate order ``q``, and it tends to ``-1`` as ``p`` grows, which is the ceiling [`set_weight_norm_inf_constraints!`](@ref) states.

# Arguments

  - $(arg_dict[:model])
  - `lps::LpReg_VecLpReg`: One or more p-norm weight constraints.

# Returns

  - `nothing`.

# Details

  - `lps::LpReg_VecLpReg`: For each term, introduces the auxiliary variables `t_lpc_i` and `r_lpc_i`, bounds ``\\lVert \\boldsymbol{w} \\rVert_{p_i}`` above by `t_lpc_i` with a set of `MOI.PowerCone` constraints, and adds the linear constraint `t_lpc_i <= val * k`. Variables and constraints are suffixed by the term's index, so terms do not collide, and are named distinctly from those of [`set_lp_regularisation!`](@ref) so a model may carry both an Lp penalty and a p-norm constraint.
  - `args...`: No-op, used when no p-norm weight constraint is configured.

# Related

  - [`LpRegularisation`](@ref)
  - [`LpReg_VecLpReg`](@ref)
  - [`set_weight_norm_2_constraints!`](@ref)
  - [`set_weight_norm_inf_constraints!`](@ref)
  - [`norm_ceiling_factory`](@ref)
  - [`assert_norm_ceiling_role`](@ref)
  - [`Num_NormCeilCal`](@ref)
  - [`set_lp_regularisation!`](@ref)
  - [`JuMPOptimiser`](@ref)
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

Constrain the ∞-norm of the weights, capping the largest absolute weight.

The limiting case of [`set_weight_norm_p_constraints!`](@ref). `val` is a direct upper bound on the largest absolute weight, expressed relative to the budget `k`: the constraint is ``\\lVert \\boldsymbol{w} \\rVert_\\infty \\leq \\mathrm{val} \\cdot k``. So a fully invested portfolio (``k = 1``) constrained with `val = 0.2` holds no position larger than 20%. Smaller `val` forces a more evenly spread portfolio.

The builder takes a number. The caller-facing slot also takes an [`AbstractNormCeilingCalibrationAlgorithm`](@ref), which computes the ceiling from the universe the prior result carries, and [`assemble_jump_model!`](@ref) resolves it before it calls here.

# Mathematical definition

```math
\\begin{align}
\\mathrm{t}_\\infty &\\geq \\lVert \\boldsymbol{w} \\rVert_\\infty\\,, \\\\
\\mathrm{t}_\\infty &\\leq \\mathrm{val} \\cdot k\\,.
\\end{align}
```

Where:

  - ``\\mathrm{t}_\\infty``: Auxiliary variable upper-bounding ``\\lVert \\boldsymbol{w} \\rVert_\\infty``.
  - ``\\mathrm{val}``: Upper bound on the largest absolute weight.
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])

# Diversification interpretation

Capping the largest weight spreads the portfolio across a minimum number of assets. To spread across **at least `m` assets**, set `val = 1 / m`: no single position can then exceed a ``1 / m`` share of a fully invested portfolio.

# Arguments

  - $(arg_dict[:model])
  - `val::Number`: Upper bound on the largest absolute weight.

# Returns

  - `nothing`.

# Details

  - `val::Number`: Introduces the auxiliary variable `t_linfc`, bounds it below by ``\\lVert \\boldsymbol{w} \\rVert_\\infty`` with a `MOI.NormInfinityCone` constraint, and adds the linear constraint `t_linfc <= val * k`.
  - `args...`: No-op, used when no ∞-norm weight constraint is configured.

# Related

  - [`set_weight_norm_2_constraints!`](@ref)
  - [`set_weight_norm_p_constraints!`](@ref)
  - [`Num_NormCeilCal`](@ref)
  - [`EffectiveAssetFloor`](@ref)
  - [`set_linf_regularisation!`](@ref)
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

Register an epigraph variable of the dual norm of `x`, raising the cone the dual order of `p` names.

The cone follows the dual norm order `q = dual_norm_order(p)`, and the four routes are the ones
[`set_weight_norm_2_constraints!`](@ref), [`set_weight_norm_inf_constraints!`](@ref),
[`set_weight_norm_p_constraints!`](@ref) and the box return builder raise for the weights: a
second-order cone at ``q = 2``, a norm-one cone at ``q = 1`` (``p = \\infty``), a norm-infinity
cone at ``q = \\infty`` (``p = 1``), and one power cone per entry of `x` otherwise. Both norm-ball
consumers call it, the mean builder on ``\\mathbf{L}^{\\intercal}\\boldsymbol{w}`` and the
covariance builder on ``\\mathbf{L}^{\\intercal}\\operatorname{vec}(\\mathbf{W} + \\mathbf{E})``,
so it takes `prefix` and `i` and registers every entry under both.

# JuMP formulation

## Variables

  - `t_nbucs_i`: epigraph of the dual norm, ``t \\geq \\lVert \\boldsymbol{x} \\rVert_{q}``.
  - `r_nbucs_i`: one auxiliary per entry of `x`, on the power-cone route only.

## Constraints

  - `nbucs_cone_i`: ``(s_c t, s_c \\boldsymbol{x}) \\in \\mathcal{K}_{q}``, with ``\\mathcal{K}_{2}`` the second-order cone, ``\\mathcal{K}_{1}`` the norm-one cone and ``\\mathcal{K}_{\\infty}`` the norm-infinity cone. On the power-cone route the entry holds one row per entry of `x`: ``(s_c r_j, s_c t, s_c x_j) \\in \\mathcal{P}_{1/q}``, that is ``r_j^{1/q} t^{1 - 1/q} \\geq \\lvert x_j \\rvert``.
  - `nbucs_cone_sum_i`: ``s_c \\left(\\sum_j r_j - t\\right) = 0``, on the power-cone route only, which closes ``t^{q} \\geq \\sum_j \\lvert x_j \\rvert^{q}``.

Where:

  - ``\\boldsymbol{x}``: The affine expression whose dual norm is bounded.
  - ``q``: Dual norm order of `p`.
  - $(math_dict[:sc_scale])

# Arguments

  - $(arg_dict[:model])
  - `prefix`: Model State prefix the entries are registered under.
  - `i`: Index of the term, which suffixes every name the builder registers.
  - `x`: Affine expression, a vector of at least one entry.
  - `p::Number`: Norm order of the ball, `p >= 1` with `Inf` admitted.

# Returns

  - `t_nbucs`: The epigraph variable.

# Related

  - [`NormBallUncertaintySet`](@ref)
  - [`dual_norm_order`](@ref)
  - [`set_ucs_return_constraints!`](@ref)
  - [`set_ucs_variance_risk!`](@ref)
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
