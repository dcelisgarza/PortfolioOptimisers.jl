"""
    set_weight_norm_2_constraints!(model::JuMP.Model, val::Number)
    set_weight_norm_2_constraints!(args...)

Constrain the 2-norm of the weights, bounding the effective number of assets from below.

Diversification is expressed as a bound on a norm of the weight vector: the smaller the norm, the more evenly the weights spread across the assets. For the 2-norm the reciprocal of the squared norm is the effective number of assets, so bounding the norm above by ``k / \\sqrt{\\mathrm{val}}`` is the same as requiring at least `val` effective assets.

# Mathematical definition

```math
\\begin{align}
\\mathrm{wn2} &\\geq \\lVert \\boldsymbol{w} \\rVert_2\\,, \\\\
\\mathrm{wn2} \\cdot \\sqrt{\\mathrm{val}} &\\leq k\\,.
\\end{align}
```

Which, for a fully invested portfolio (``k = 1``), is equivalent to:

```math
\\begin{align}
\\mathrm{ENA}(\\boldsymbol{w}) &= \\frac{1}{\\lVert \\boldsymbol{w} \\rVert_2^2} \\geq \\mathrm{val}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{wn2}``: Auxiliary variable upper-bounding ``\\lVert \\boldsymbol{w} \\rVert_2``.
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - ``\\mathrm{ENA}(\\boldsymbol{w})``: Effective number of assets.
  - ``\\mathrm{val}``: Minimum required effective number of assets.

# Arguments

  - $(arg_dict[:model])
  - `val::Number`: Minimum required effective number of assets.

# Returns

  - `nothing`.

# Details

  - `val::Number`: Introduces the auxiliary variable `wn2`, bounds it below by ``\\lVert \\boldsymbol{w} \\rVert_2`` with a `SecondOrderCone` constraint, and adds the linear constraint `wn2 * sqrt(val) <= k`.
  - `args...`: No-op, used when no 2-norm weight constraint is configured.

# Related

  - [`set_weight_norm_p_constraints!`](@ref)
  - [`set_weight_norm_inf_constraints!`](@ref)
  - [`number_effective_assets`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_weight_norm_2_constraints!(args...)
    return nothing
end
function set_weight_norm_2_constraints!(model::JuMP.Model, val::Number)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    JuMP.@variable(model, wn2)
    JuMP.@constraints(model, begin
                          cwn2_soc, [sc * wn2; sc * w] in JuMP.SecondOrderCone()
                          cwn2, sc * (wn2 * sqrt(val) - k) <= 0
                      end)
    return nothing
end
"""
    set_weight_norm_p_constraints!(model::JuMP.Model, lps::LpReg_VecLpReg)
    set_weight_norm_p_constraints!(args...)

Constrain the p-norm of the weights, bounding the p-norm effective number of assets from below.

Generalises [`set_weight_norm_2_constraints!`](@ref) to an arbitrary norm order ``p > 1``. Each term supplies its own norm order and bound, so several may be imposed at once.

Each term is an [`LpRegularisation`](@ref), reused here as a constraint rather than a penalty: its `p` field is the norm order, and its `val` field is the *minimum* p-norm effective number of assets, not the maximum norm. Larger `val` forces a more evenly spread portfolio.

# Mathematical definition

```math
\\begin{align}
\\mathrm{t}_{p,\\, i} &\\geq \\lVert \\boldsymbol{w} \\rVert_{p_i}\\,, \\\\
\\mathrm{t}_{p,\\, i} \\cdot \\mathrm{val}_i^{1/p_i} &\\leq k\\,.
\\end{align}
```

Which, for a fully invested portfolio (``k = 1``), is equivalent to:

```math
\\begin{align}
\\mathrm{ENA}_{p_i}(\\boldsymbol{w}) &= \\frac{1}{\\lVert \\boldsymbol{w} \\rVert_{p_i}^{p_i}} \\geq \\mathrm{val}_i\\,.
\\end{align}
```

Where:

  - ``\\mathrm{t}_{p,\\, i}``: Auxiliary variable upper-bounding ``\\lVert \\boldsymbol{w} \\rVert_{p_i}``.
  - ``p_i``: Norm order of the ``i``-th term, its `p` field.
  - ``\\mathrm{val}_i``: Minimum required p-norm effective number of assets of the ``i``-th term, its `val` field.
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - ``\\mathrm{ENA}_{p}(\\boldsymbol{w})``: Effective number of assets under the ``p``-norm.

# Arguments

  - $(arg_dict[:model])
  - `lps::LpReg_VecLpReg`: One or more p-norm weight constraints.

# Returns

  - `nothing`.

# Details

  - `lps::LpReg_VecLpReg`: For each term, introduces the auxiliary variables `t_wnp_i` and `r_wnp_i`, bounds ``\\lVert \\boldsymbol{w} \\rVert_{p_i}`` above by `t_wnp_i` with a set of `MOI.PowerCone` constraints, and adds the linear constraint `t_wnp_i * val^(1 / p) <= k`. Variables and constraints are suffixed by the term's index, so terms do not collide, and are named distinctly from those of [`set_lp_regularisation!`](@ref) so a model may carry both an Lp penalty and a p-norm constraint.
  - `args...`: No-op, used when no p-norm weight constraint is configured.

# Related

  - [`LpRegularisation`](@ref)
  - [`LpReg_VecLpReg`](@ref)
  - [`set_weight_norm_2_constraints!`](@ref)
  - [`set_weight_norm_inf_constraints!`](@ref)
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
        t_wnp, r_wnp = model[Symbol(:t_wnp_, i)], model[Symbol(:r_wnp_, i)] = JuMP.@variables(model,
                                                                                              begin
                                                                                                  ()
                                                                                                  [1:N]
                                                                                              end)
        model[(Symbol(:cwnp_, i))], model[Symbol(:cswnp_, i)] = JuMP.@constraints(model,
                                                                                  begin
                                                                                      [i = 1:N],
                                                                                      [sc *
                                                                                       r_wnp[i],
                                                                                       sc *
                                                                                       t_wnp,
                                                                                       sc *
                                                                                       w[i]] in
                                                                                      JuMP.MOI.PowerCone(p_inv)
                                                                                      sc *
                                                                                      (sum(r_wnp) -
                                                                                       t_wnp) ==
                                                                                      0
                                                                                  end)
        val = lp.p == 3 * one(lp.p) ? cbrt(val) : val^p_inv
        model[Symbol(:cwnp_bnd_, i)] = JuMP.@constraint(model, sc * (t_wnp * val - k) <= 0)
    end
    return nothing
end
"""
    set_weight_norm_inf_constraints!(model::JuMP.Model, linf::Number)
    set_weight_norm_inf_constraints!(args...)

Constrain the ∞-norm of the weights, bounding the largest weight from above.

The limiting case of [`set_weight_norm_p_constraints!`](@ref). `linf` is the *reciprocal* of the largest permitted weight, not the weight itself: a fully invested portfolio constrained with `linf = 5` holds no position larger than `1 / 5`, and so is spread across at least five assets. Larger `linf` forces a more evenly spread portfolio, consistent with the other members of the family.

# Mathematical definition

```math
\\begin{align}
\\mathrm{t}_\\infty &\\geq \\lVert \\boldsymbol{w} \\rVert_\\infty\\,, \\\\
\\mathrm{t}_\\infty \\cdot \\mathrm{linf} &\\leq k\\,.
\\end{align}
```

Which, for a fully invested portfolio (``k = 1``), is equivalent to:

```math
\\begin{align}
\\underset{i}{\\max} \\, |w_i| &\\leq \\frac{1}{\\mathrm{linf}}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{t}_\\infty``: Auxiliary variable upper-bounding ``\\lVert \\boldsymbol{w} \\rVert_\\infty``.
  - ``\\mathrm{linf}``: Reciprocal of the largest permitted weight.
  - ``w_i``: Portfolio weight for asset ``i``.
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])

# Arguments

  - $(arg_dict[:model])
  - `linf::Number`: Reciprocal of the largest permitted weight.

# Returns

  - `nothing`.

# Details

  - `linf::Number`: Introduces the auxiliary variable `t_wninf`, bounds it below by ``\\lVert \\boldsymbol{w} \\rVert_\\infty`` with a `MOI.NormInfinityCone` constraint, and adds the linear constraint `t_wninf * linf <= k`.
  - `args...`: No-op, used when no ∞-norm weight constraint is configured.

# Related

  - [`set_weight_norm_2_constraints!`](@ref)
  - [`set_weight_norm_p_constraints!`](@ref)
  - [`set_linf_regularisation!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_weight_norm_inf_constraints!(args...)
    return nothing
end
function set_weight_norm_inf_constraints!(model::JuMP.Model, linf::Number)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    JuMP.@variable(model, t_wninf)
    JuMP.@constraint(model, cwninf_nic,
                     [sc * t_wninf;
                      sc * w] in JuMP.MOI.NormInfinityCone(1 + length(w)))
    JuMP.@constraint(model, wninf, sc * (t_wninf * linf - k) <= 0)
    return nothing
end
