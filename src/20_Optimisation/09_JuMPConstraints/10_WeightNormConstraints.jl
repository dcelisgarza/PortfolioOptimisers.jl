"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add an effective number of assets constraint to the JuMP optimisation model.

The fall-through method does nothing. The concrete method introduces an auxiliary variable `wn2` and enforces `‖w‖₂ ≤ wn2` via a SecondOrderCone constraint, combined with `wn2 * √val ≤ k`. This is equivalent to requiring the effective number of assets to be at least `val`.

# Mathematical definition

```math
\\begin{align}
\\mathrm{wn2} \\geq \\|\\boldsymbol{w}\\|_2\\,, \\\\
\\mathrm{wn2} \\cdot \\sqrt{\\mathrm{val}} \\leq k
\\quad \\Leftrightarrow \\quad \\mathrm{ENA}(\\boldsymbol{w}) &= \\frac{1}{\\|\\boldsymbol{w}\\|_2^2} \\geq \\mathrm{val}\\,.
\\end{align}
```

Where:

  - ``\\mathrm{wn2}``: Auxiliary variable upper-bounding ``\\|\\boldsymbol{w}\\|_2``.
  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - ``\\mathrm{ENA}(\\boldsymbol{w})``: Effective number of assets.
  - ``\\mathrm{val}``: Minimum required effective number of assets.

# Arguments

  - $(arg_dict[:model])
  - `val::Number`: Minimum required effective number of assets.

# Returns

  - `nothing`.

# Related

  - [`number_effective_assets`](@ref)
  - [`EqualRisk`](@ref)
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
                          cnea_soc, [sc * wn2; sc * w] in JuMP.SecondOrderCone()
                          cnea, sc * (wn2 * sqrt(val) - k) <= 0
                      end)
    return nothing
end
"""
    set_weight_norm_p_constraints!(args...)
    set_weight_norm_p_constraints!(model::JuMP.Model, lps::LpReg_VecLpReg)

Constrain the P-Norm of the weights to be less than or equal to the value in `lps`. Vararg method is a no-op.

# Arguments

  - $(arg_dict[:model])
  - `lps::LpReg_VecLpReg`: Single instance or vector of [`LpRegularisation`](@ref) objects, one for each P-Norm constraint.

# Returns

  - `nothing`.

# Related

  - [`LpRegularisation`](@ref)
  - [`set_weight_norm_inf_constraints!`](@ref)
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
        t_neap, r_neap = model[Symbol(:t_neap_, i)], model[Symbol(:r_neap_, i)] = JuMP.@variables(model,
                                                                                                  begin
                                                                                                      ()
                                                                                                      [1:N]
                                                                                                  end)
        model[(Symbol(:clp_, i))], model[Symbol(:cslp_, i)] = JuMP.@constraints(model,
                                                                                begin
                                                                                    [i = 1:N],
                                                                                    [sc *
                                                                                     r_neap[i],
                                                                                     sc *
                                                                                     t_neap,
                                                                                     sc *
                                                                                     w[i]] in
                                                                                    JuMP.MOI.PowerCone(p_inv)
                                                                                    sc *
                                                                                    (sum(r_neap) -
                                                                                     t_neap) ==
                                                                                    0
                                                                                end)
        val = lp.p == 3 * one(lp.p) ? cbrt(val) : val^p_inv
        model[Symbol(:cneap_, i)] = JuMP.@constraint(model, sc * (t_neap * val - k) <= 0)
    end
    return nothing
end
"""
    set_weight_norm_inf_constraints!(args...)
    set_weight_norm_inf_constraints!(model::JuMP.Model, linf::Number)

Constrain the L∞-Norm of the weights to be less than or equal to `linf`. Vararg method is a no-op.

# Arguments

  - $(arg_dict[:model])
  - `linf::Number`: Maximum allowed L∞-Norm of the weights.

# Returns

  - `nothing`.

# Related

  - [`set_weight_norm_2_constraints!`](@ref)
  - [`set_weight_norm_p_constraints!`](@ref)
"""
function set_weight_norm_inf_constraints!(args...)
    return nothing
end
function set_weight_norm_inf_constraints!(model::JuMP.Model, linf::Number)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    JuMP.@variable(model, t_neainf)
    JuMP.@constraint(model, cneainf_nic,
                     [sc * t_neainf;
                      sc * w] in JuMP.MOI.NormInfinityCone(1 + length(w)))
    JuMP.@constraint(model, wninf, sc * (t_neainf * linf - k) <= 0)
    return nothing
end
