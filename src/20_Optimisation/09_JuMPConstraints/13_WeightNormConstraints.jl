"""
    set_weight_norm_2_constraints!(model::JuMP.Model, val::Number)
    set_weight_norm_2_constraints!(args...)

Constrain the 2-norm of the weights.

`val` is a direct upper bound on ``\\lVert \\boldsymbol{w} \\rVert_2``, expressed relative to the budget `k`: the constraint is ``\\lVert \\boldsymbol{w} \\rVert_2 \\leq \\mathrm{val} \\cdot k``. Smaller `val` forces the weights to spread more evenly across the assets.

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

For a fully invested portfolio (``k = 1``), the p-norm effective number of assets is ``\\mathrm{ENA}_p(\\boldsymbol{w}) = 1 / \\lVert \\boldsymbol{w} \\rVert_p^p``. To require **at least `m` p-norm effective assets**, set `val = m^(-1/p)`:

```math
\\begin{align}
\\lVert \\boldsymbol{w} \\rVert_p \\leq m^{-1/p} \\iff \\mathrm{ENA}_p(\\boldsymbol{w}) = \\frac{1}{\\lVert \\boldsymbol{w} \\rVert_p^p} \\geq m\\,.
\\end{align}
```

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
        t_lpc, r_lpc = model[Symbol(:t_lpc_, i)], model[Symbol(:r_lpc_, i)] = JuMP.@variables(model,
                                                                                              begin
                                                                                                  ()
                                                                                                  [1:N]
                                                                                              end)
        model[(Symbol(:clpc_, i))], model[Symbol(:cslpc_, i)] = JuMP.@constraints(model,
                                                                                  begin
                                                                                      [i = 1:N],
                                                                                      [sc *
                                                                                       r_lpc[i],
                                                                                       sc *
                                                                                       t_lpc,
                                                                                       sc *
                                                                                       w[i]] in
                                                                                      JuMP.MOI.PowerCone(p_inv)
                                                                                      sc *
                                                                                      (sum(r_lpc) -
                                                                                       t_lpc) ==
                                                                                      0
                                                                                  end)
        model[Symbol(:clpc_bnd_, i)] = JuMP.@constraint(model, sc * (t_lpc - val * k) <= 0)
    end
    return nothing
end
"""
    set_weight_norm_inf_constraints!(model::JuMP.Model, val::Number)
    set_weight_norm_inf_constraints!(args...)

Constrain the ∞-norm of the weights, capping the largest absolute weight.

The limiting case of [`set_weight_norm_p_constraints!`](@ref). `val` is a direct upper bound on the largest absolute weight, expressed relative to the budget `k`: the constraint is ``\\lVert \\boldsymbol{w} \\rVert_\\infty \\leq \\mathrm{val} \\cdot k``. So a fully invested portfolio (``k = 1``) constrained with `val = 0.2` holds no position larger than 20%. Smaller `val` forces a more evenly spread portfolio.

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
  - [`set_linf_regularisation!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_weight_norm_inf_constraints!(args...)
    return nothing
end
function set_weight_norm_inf_constraints!(model::JuMP.Model, val::Number)
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
