"""
$(DocStringExtensions.TYPEDEF)

MIP space of the positions of one concentration rule.

[`mip_wx!`](@ref) gives the positions ``\\boldsymbol{x}`` of the rule, and [`mip_key`](@ref) writes each key as `conc_<name>_i` under `prefix`, so two rules and two blocks do not collide. [`declare_mip_gate!`](@ref) declares the binary "the position is over `t`" in this space. The binary is not the held bit of [`HeldIndicators`](@ref), because "held" and "over `t`" are two facts, so the space registers no bundle in Model State and [`mip_indicators`](@ref) still returns the asset bundle. The space has no method of [`mip_bounds`](@ref), because the rule states its cap on the positions.

The positions are the rows of `cl.set` times the reading `cl.pos` of the weights, see [`concentration_positions`](@ref) and [`position_reading`](@ref). The first block that admits a short position makes the shared parts `lw` and `sw`, see [`set_weight_parts!`](@ref), and a [`RiskBudgeting`](@ref) head declares them itself. When the model has no shared short part, a block on the weights is long-only, and the library reads the weights as their own long part. So every reading is `w` there. A block after a long-only block can make parts of its own for its budgets, see [`weight_parts!`](@ref). The rule does not read them, because the long-only block already gives the reading `w`.

# Fields

  - `cl`: The resolved rule.
  - `i`: Index of the rule in the slot, used in key naming.
  - `prefix::Symbol`: Model State namespace of the block. The weights are read under it.

# Related

  - [`AbstractMIPSpace`](@ref)
  - [`ConcentrationLimit`](@ref)
  - [`set_concentration_constraints!`](@ref)
  - [`declare_mip_gate!`](@ref)
"""
@concrete struct ConcentrationMIPSpace <: AbstractMIPSpace
    cl
    i
    prefix::Symbol
    function ConcentrationMIPSpace(cl::ConcentrationLimit, i::Integer,
                                   prefix::Symbol)::ConcentrationMIPSpace
        return new{typeof(cl), typeof(i)}(cl, i, prefix)
    end
end
function mip_key(sp::ConcentrationMIPSpace, name::Symbol)
    return state_key(sp.prefix, Symbol(:conc_, name, :_), sp.i)
end
function mip_wx!(model::JuMP.Model, sp::ConcentrationMIPSpace)
    (; set, pos) = sp.cl
    w = get_w(model, sp.prefix)
    v = if shared_has(model, :sw)
        position_reading(pos, w, shared_get(model, :lw), shared_get(model, :sw))
    else
        w
    end
    return state_set!(model, Symbol(""), mip_key(sp, :x),
                      JuMP.@expression(model, concentration_positions(set, v)))
end
"""
    set_concentration_constraints!(model::JuMP.Model, conc::Nothing, args...; kwargs...)
    set_concentration_constraints!(model::JuMP.Model, conc::Cnl_VecCnl,
                                   wb::WeightBounds, ss::Option{<:Number};
                                   prefix::Symbol = Symbol(""))

Add the concentration rules of a block to the model.

The block builder [`set_block_constraints!`](@ref) calls it after the MIP kinds, so the parts of the weights exist. Rule `i` of the slot registers its entries under the keys `conc_<name>_i`, under `prefix`, so two rules and two blocks do not collide. The form of each rule, its field `alg`, picks its rows, see [`set_concentration_rows!`](@ref). The rules of a vector intersect.

The rows follow the scale of the weights. A ratio objective and a [`RiskBudgeting`](@ref) head solve for the scaled weights ``k \\boldsymbol{w}``, so each number of a rule is multiplied by `k`, the variable of [`get_k`](@ref), as the weight bounds multiply theirs. The binary of the exact form then gates the rows through the product `z k` that [`declare_mip_gate!`](@ref) declares, as the held bit does. When `k` is the number 1, the binary gates the rows directly.

# Algorithm

 1. `conc = nothing`: add no row.
 2. For each rule `cl` at index `i`:
     1. Make the [`ConcentrationMIPSpace`](@ref) of the rule.
     2. Make the positions `x` with [`mip_wx!`](@ref), which registers them as `conc_x_i`.
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
  - [`ConcentrationMIPSpace`](@ref)
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
        sp = ConcentrationMIPSpace(cl, i, prefix)
        set_concentration_rows!(model, cl.alg, sp, mip_wx!(model, sp), wb, ss)
    end
    return nothing
end
"""
    set_concentration_rows!(model::JuMP.Model, alg::ExactConcentration,
                            sp::ConcentrationMIPSpace, x::AbstractVector,
                            wb::WeightBounds, ss::Option{<:Number})
    set_concentration_rows!(model::JuMP.Model, alg::SumLargestConcentration,
                            sp::ConcentrationMIPSpace, x::AbstractVector, wb, ss)

Add the rows of one concentration rule in the form `alg`.

# JuMP formulation

## Variables

The exact form:

  - `conc_z_i`: ``M \\times 1``, binary. ``z_j = 1`` says that position ``j`` is over ``t``.
  - `conc_g_i`: ``M \\times 1``, ``\\geq 0``, when ``k`` is a variable. It is the gate ``k\\, z_j``.
  - `conc_y_i`: ``M \\times 1``, ``\\geq 0``. The part of each position that counts towards the sum.

The linear form:

  - `conc_u_i`: scalar, ``\\geq 0``.
  - `conc_v_i`: ``M \\times 1``, ``\\geq 0``.

## Constraints

The exact form, with the gate ``\\boldsymbol{g} = k\\, \\boldsymbol{z}`` and the rows of [`declare_mip_gate!`](@ref):

  - `conc_cap_i`: ``s_c \\left(\\boldsymbol{x} - t k - (c - t)\\, \\boldsymbol{g}\\right) \\leq 0``.
  - `conc_ylb_i`: ``s_c \\left(\\boldsymbol{y} - \\boldsymbol{x} + c k - c\\, \\boldsymbol{g}\\right) \\geq 0``.
  - `conc_sum_i`: ``s_c \\left(\\sum_j y_j - s k\\right) \\leq 0``.

The linear form, [`SumLargestConcentration`](@ref), with ``n`` from [`sum_largest_count`](@ref):

  - `conc_cap_i`: ``s_c \\left(\\boldsymbol{x} - c k\\right) \\leq 0``.
  - `conc_vlb_i`: ``s_c \\left(\\boldsymbol{v} - \\boldsymbol{x} + u\\right) \\geq 0``.
  - `conc_sum_i`: ``s_c \\left(n u + \\sum_j v_j - s k\\right) \\leq 0``.

The minimum of ``n u + \\sum_j \\max(x_j - u, 0)`` over ``u \\geq 0`` is the sum of the ``n`` largest positive parts of ``\\boldsymbol{x}``, so the last row bounds that sum. The bound ``u \\geq 0`` makes a negative position add nothing.

Where:

  - $(math_dict[:sc_scale])
  - ``\\boldsymbol{x}``: The positions `conc_x_i`, ``M \\times 1``.
  - $(math_dict[:k_budget])
  - ``t``, ``c``, ``s``: The threshold, the cap and the limit of the sum of `cl`.

# Arguments

  - $(arg_dict[:model])
  - `alg`: The form of the rule.
  - `sp`: The space of the rule. `sp.cl` is the rule, and [`mip_key`](@ref) names the model keys.
  - `x`: The positions of the rule.
  - `wb`: The weight bounds of the block.
  - $(arg_dict[:ss_arg])

# Returns

  - `nothing`.

# Related

  - [`set_concentration_constraints!`](@ref)
  - [`ConcentrationMIPSpace`](@ref)
  - [`declare_mip_gate!`](@ref)
  - [`ExactConcentration`](@ref)
  - [`SumLargestConcentration`](@ref)
"""
function set_concentration_rows!(model::JuMP.Model, ::ExactConcentration,
                                 sp::ConcentrationMIPSpace, x::AbstractVector,
                                 wb::WeightBounds, ss::Option{<:Number})::Nothing
    sc = get_constraint_scale(model)
    k = get_k(model)
    M = length(x)
    (; t, c, s) = sp.cl
    # `mip_key` composes the whole key, so the interface registers it under the empty prefix.
    pfx = Symbol("")
    _, g = declare_mip_gate!(model, sp, :z, :g, M, wb, ss)
    y = state_set!(model, pfx, mip_key(sp, :y),
                   JuMP.@variable(model, [1:M], lower_bound = 0))
    state_set!(model, pfx, mip_key(sp, :cap),
               JuMP.@constraint(model, sc * (x .- t * k .- (c - t) .* g) <= 0))
    state_set!(model, pfx, mip_key(sp, :ylb),
               JuMP.@constraint(model, sc * (y .- x .+ c * k .- c .* g) >= 0))
    state_set!(model, pfx, mip_key(sp, :sum),
               JuMP.@constraint(model, sc * (sum(y) - s * k) <= 0))
    return nothing
end
function set_concentration_rows!(model::JuMP.Model, ::SumLargestConcentration,
                                 sp::ConcentrationMIPSpace, x::AbstractVector, ::Any,
                                 ::Any)::Nothing
    sc = get_constraint_scale(model)
    k = get_k(model)
    M = length(x)
    (; t, c, s) = sp.cl
    n = sum_largest_count(t, s)
    pfx = Symbol("")
    u = state_set!(model, pfx, mip_key(sp, :u), JuMP.@variable(model, lower_bound = 0))
    v = state_set!(model, pfx, mip_key(sp, :v),
                   JuMP.@variable(model, [1:M], lower_bound = 0))
    state_set!(model, pfx, mip_key(sp, :cap),
               JuMP.@constraint(model, sc * (x .- c * k) <= 0))
    state_set!(model, pfx, mip_key(sp, :vlb),
               JuMP.@constraint(model, sc * ((v - x) .+ u) >= 0))
    state_set!(model, pfx, mip_key(sp, :sum),
               JuMP.@constraint(model, sc * (n * u + sum(v) - s * k) <= 0))
    return nothing
end

public ConcentrationMIPSpace, set_concentration_rows!
