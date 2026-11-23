"""
    concentration_reading(model::JuMP.Model, pos::AbstractPositionReading, prefix::Symbol)

Return the reading of the weights of the model that `pos` gives, one entry for each asset.

The first block that admits a short position makes the shared parts `lw` and `sw`, see [`set_weight_parts!`](@ref), and a [`RiskBudgeting`](@ref) head declares them itself. When the model has no shared short part, a block on the weights is long-only, and the library reads the weights as their own long part. So every reading is `w` there. A block after a long-only block can make parts of its own for its budgets, see [`weight_parts!`](@ref). The rule does not read them, because the long-only block already gives the reading `w`.

# Algorithm

 1. Read the weights `w` under `prefix`.
 2. When the model has no shared short part `sw`, return `w`.
 3. Otherwise return [`position_reading`](@ref) of `pos`, with `w` and the shared parts `lw` and `sw`.

# Arguments

  - $(arg_dict[:model])
  - `pos`: The reading.
  - $(arg_dict[:prefix_kw])

# Returns

  - `v::AbstractVector`: The reading of each asset.

# Related

  - [`AbstractPositionReading`](@ref)
  - [`position_reading`](@ref)
  - [`set_concentration_constraints!`](@ref)
  - [`set_weight_parts!`](@ref)
"""
function concentration_reading(model::JuMP.Model, pos::AbstractPositionReading,
                               prefix::Symbol)
    w = get_w(model, prefix)
    return if shared_has(model, :sw)
        position_reading(pos, w, shared_get(model, :lw), shared_get(model, :sw))
    else
        w
    end
end
"""
    set_concentration_constraints!(model::JuMP.Model, conc::Nothing, args...; kwargs...)
    set_concentration_constraints!(model::JuMP.Model, conc::Cnl_VecCnl,
                                   wb::WeightBounds, ss::Option{<:Number};
                                   prefix::Symbol = Symbol(""))

Add the concentration rules of a block to the model.

The block builder [`set_block_constraints!`](@ref) calls it after the MIP kinds, so the parts of the weights exist. Rule `i` of the slot registers its entries under the keys `conc_<name>_i`, under `prefix`, so two rules and two blocks do not collide. The form of each rule, its field `alg`, picks its rows, see [`set_concentration_rows!`](@ref). The rules of a vector intersect.

The rows follow the scale of the weights. A ratio objective and a [`RiskBudgeting`](@ref) head solve for the scaled weights ``k \\boldsymbol{w}``, so each number of a rule is multiplied by `κ`, the variable of [`get_k`](@ref), as the weight bounds multiply theirs. The binary of the exact form then gates the product `z κ` through a continuous variable and the big-M expression `ss` of [`set_mip_ss_expr!`](@ref), as [`declare_mip_indicators!`](@ref) does for the held bit. When `κ` is the number 1, the binary gates the rows directly.

# Algorithm

 1. `conc = nothing`: add no row.
 2. For each rule `cl` at index `i`:
     1. Read the weights with [`concentration_reading`](@ref) for `cl.pos`.
     2. Make the positions `x` with [`concentration_positions`](@ref) for `cl.set`, and register them as `conc_x_i`.
     3. Add the rows of the form `cl.alg` with [`set_concentration_rows!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `conc`: The resolved rules of the block, or `nothing`.
  - `wb`: The weight bounds of the block, which size the big-M expression under a free budget.
  - $(arg_dict[:ss_arg])
  - $(arg_dict[:prefix_kw])

# Returns

  - `nothing`.

# Related

  - [`ConcentrationLimit`](@ref)
  - [`set_concentration_rows!`](@ref)
  - [`set_block_constraints!`](@ref)
"""
function set_concentration_constraints!(::JuMP.Model, ::Nothing, args...;
                                        kwargs...)::Nothing
    return nothing
end
function set_concentration_constraints!(model::JuMP.Model, conc::Cnl_VecCnl,
                                        wb::WeightBounds, ss::Option{<:Number};
                                        prefix::Symbol = Symbol(""))::Nothing
    # `vcat` gives a one-rule vector for one rule, so one loop reads both forms.
    for (i, cl) in enumerate(vcat(conc))
        v = concentration_reading(model, cl.pos, prefix)
        x = state_set!(model, prefix, :conc_x_, i,
                       JuMP.@expression(model, concentration_positions(cl.set, v)))
        set_concentration_rows!(model, cl.alg, cl, x, wb, ss, i, prefix)
    end
    return nothing
end
"""
    set_concentration_rows!(model::JuMP.Model, alg::ExactConcentration, cl::ConcentrationLimit,
                            x::AbstractVector, wb::WeightBounds,
                            ss::Option{<:Number}, i::Integer, prefix::Symbol)
    set_concentration_rows!(model::JuMP.Model, alg::SumLargestConcentration,
                            cl::ConcentrationLimit, x::AbstractVector, wb, ss, i::Integer,
                            prefix::Symbol)

Add the rows of one concentration rule in the form `alg`.

# JuMP formulation

## Variables

The exact form:

  - `conc_z_i`: ``M \\times 1``, binary. ``z_j = 1`` says that position ``j`` is over ``t``.
  - `conc_zk_i`: ``M \\times 1``, ``\\geq 0``, under a free budget only. It is the product ``z_j \\kappa``.
  - `conc_y_i`: ``M \\times 1``, ``\\geq 0``. The part of each position that counts towards the sum.

The linear form:

  - `conc_u_i`: scalar, ``\\geq 0``.
  - `conc_v_i`: ``M \\times 1``, ``\\geq 0``.

## Constraints

The exact form, with the gate ``g = z \\kappa`` under a number ``\\kappa``, and ``g =`` `conc_zk_i` otherwise:

  - `conc_cap_i`: ``s_c \\left(\\boldsymbol{x} - t \\kappa - (c - t)\\, \\boldsymbol{g}\\right) \\leq 0``.
  - `conc_ylb_i`: ``s_c \\left(\\boldsymbol{y} - \\boldsymbol{x} + c \\kappa - c\\, \\boldsymbol{g}\\right) \\geq 0``.
  - `conc_sum_i`: ``s_c \\left(\\sum_j y_j - s \\kappa\\right) \\leq 0``.
  - `conc_zk_ub_i`, `conc_zkd_ub_i`, `conc_zkd_lb_i`, under a free budget only: ``g_j \\leq \\kappa``, ``g_j \\leq M_s z_j`` and ``g_j + M_s (1 - z_j) \\geq \\kappa``.

The linear form, [`SumLargestConcentration`](@ref), with ``k`` from [`sum_largest_count`](@ref):

  - `conc_cap_i`: ``s_c \\left(\\boldsymbol{x} - c \\kappa\\right) \\leq 0``.
  - `conc_vlb_i`: ``s_c \\left(\\boldsymbol{v} - \\boldsymbol{x} + u\\right) \\geq 0``.
  - `conc_sum_i`: ``s_c \\left(k u + \\sum_j v_j - s \\kappa\\right) \\leq 0``.

The minimum of ``k u + \\sum_j \\max(x_j - u, 0)`` over ``u \\geq 0`` is the sum of the ``k`` largest positive parts of ``\\boldsymbol{x}``, so the last row bounds that sum. The bound ``u \\geq 0`` makes a negative position add nothing.

Where:

  - $(math_dict[:sc_scale])
  - ``\\boldsymbol{x}``: The positions `conc_x_i`, ``M \\times 1``.
  - ``\\kappa``: The budget scale, [`get_k`](@ref).
  - ``t``, ``c``, ``s``: The threshold, the cap and the limit of the sum of `cl`.
  - ``M_s``: The big-M expression of [`set_mip_ss_expr!`](@ref).

# Arguments

  - $(arg_dict[:model])
  - `alg`: The form of the rule.
  - `cl`: The rule.
  - `x`: The positions of the rule.
  - `wb`: The weight bounds of the block.
  - $(arg_dict[:ss_arg])
  - `i`: The index of the rule in the slot.
  - $(arg_dict[:prefix_kw])

# Returns

  - `nothing`.

# Related

  - [`set_concentration_constraints!`](@ref)
  - [`ExactConcentration`](@ref)
  - [`SumLargestConcentration`](@ref)
"""
function set_concentration_rows!(model::JuMP.Model, ::ExactConcentration,
                                 cl::ConcentrationLimit, x::AbstractVector,
                                 wb::WeightBounds, ss::Option{<:Number}, i::Integer,
                                 prefix::Symbol)::Nothing
    sc = get_constraint_scale(model)
    κ = get_k(model)
    M = length(x)
    (; t, c, s) = cl
    z = state_set!(model, prefix, :conc_z_, i, JuMP.@variable(model, [1:M], binary = true))
    g = if isa(κ, Number)
        state_set!(model, prefix, :conc_g_, i, JuMP.@expression(model, κ * z))
    else
        ssx = set_mip_ss_expr!(model, ss, wb)
        zk = state_set!(model, prefix, :conc_zk_, i,
                        JuMP.@variable(model, [1:M], lower_bound = 0))
        state_set!(model, prefix, :conc_zk_ub_, i,
                   JuMP.@constraint(model, sc * (zk .- κ) <= 0))
        state_set!(model, prefix, :conc_zkd_ub_, i,
                   JuMP.@constraint(model, sc * (zk - ssx * z) <= 0))
        state_set!(model, prefix, :conc_zkd_lb_, i,
                   JuMP.@constraint(model, sc * ((zk + ssx * (1 .- z)) .- κ) >= 0))
        zk
    end
    y = state_set!(model, prefix, :conc_y_, i,
                   JuMP.@variable(model, [1:M], lower_bound = 0))
    state_set!(model, prefix, :conc_cap_, i,
               JuMP.@constraint(model, sc * (x .- t * κ .- (c - t) .* g) <= 0))
    state_set!(model, prefix, :conc_ylb_, i,
               JuMP.@constraint(model, sc * (y .- x .+ c * κ .- c .* g) >= 0))
    state_set!(model, prefix, :conc_sum_, i,
               JuMP.@constraint(model, sc * (sum(y) - s * κ) <= 0))
    return nothing
end
function set_concentration_rows!(model::JuMP.Model, ::SumLargestConcentration,
                                 cl::ConcentrationLimit, x::AbstractVector, ::Any, ::Any,
                                 i::Integer, prefix::Symbol)::Nothing
    sc = get_constraint_scale(model)
    κ = get_k(model)
    M = length(x)
    (; t, c, s) = cl
    k = sum_largest_count(t, s)
    u = state_set!(model, prefix, :conc_u_, i, JuMP.@variable(model, lower_bound = 0))
    v = state_set!(model, prefix, :conc_v_, i,
                   JuMP.@variable(model, [1:M], lower_bound = 0))
    state_set!(model, prefix, :conc_cap_, i,
               JuMP.@constraint(model, sc * (x .- c * κ) <= 0))
    state_set!(model, prefix, :conc_vlb_, i,
               JuMP.@constraint(model, sc * ((v - x) .+ u) >= 0))
    state_set!(model, prefix, :conc_sum_, i,
               JuMP.@constraint(model, sc * (k * u + sum(v) - s * κ) <= 0))
    return nothing
end

public set_concentration_rows!
