"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when the weight bound `wb` contains at least one negative value,
indicating a long-short strategy is required.

# Arguments

  - `wb`: Weight bound. Accepts `nothing`, a scalar `Number`, or a `VecNum`.

# Returns

  - `flag::Bool`: `false` when `wb` is `nothing`; `wb < 0` when `wb` is a scalar; `any(x -> x < 0, wb)` when `wb` is a vector.

# Related

  - [`w_finite_flag`](@ref)
  - [`set_weight_constraints!`](@ref)
  - [`WeightBounds`](@ref)
"""
function w_neg_flag(::Nothing)
    return false
end
function w_neg_flag(wb::Number)
    return wb < zero(wb)
end
function w_neg_flag(wb::VecNum)
    return any(x -> x < zero(x), wb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when `wb` contains at least one finite value, meaning a bound constraint should be added to the model.

# Arguments

  - `wb`: Weight bound. Accepts `nothing`, a scalar `Number`, or a `VecNum`.

# Returns

  - `flag::Bool`: `false` when `wb` is `nothing`; `isfinite(wb)` when `wb` is a scalar; `any(isfinite, wb)` when `wb` is a vector.

# Related

  - [`w_neg_flag`](@ref)
  - [`set_weight_constraints!`](@ref)
  - [`WeightBounds`](@ref)
"""
function w_finite_flag(::Nothing)
    return false
end
function w_finite_flag(wb::Number)
    return isfinite(wb)
end
function w_finite_flag(wb::VecNum)
    return any(isfinite, wb)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add weight bound constraints to the JuMP optimisation model.

The fall-through method does nothing. The concrete method adds lower-bound and upper-bound constraints on the portfolio weight vector `w`, handles long-short decomposition when negative bounds are present via `lw`/`sw` variables, and delegates budget constraints to [`set_budget_constraints!`](@ref).

# Mathematical definition

```math
\\begin{align}
k \\boldsymbol{\\ell} &\\leq \\boldsymbol{w} \\leq k \\boldsymbol{u}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - ``\\boldsymbol{\\ell}``, ``\\boldsymbol{u}``: Lower and upper bound vectors from `wb`.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `bgt`: Optional total budget constraint (number or [`BudgetRange`](@ref)).
  - `sbgt`: Optional short-side budget constraint.
  - `gbgt`: Optional gross (leverage) budget constraint, applied only when the weight bounds admit shorts. See [`set_gross_budget_constraints!`](@ref).
  - `long::Bool = false`: When `true`, raises an error if any bound is negative.

!!! note

    The budgets set here **bound** the realised exposures rather than pinning them, because `lw` and `sw` are upper bounds on the parts of `w`. Pinning them is the `xbgt` option of [`short_mip_threshold_constraints`](@ref), applied later by [`set_mip_constraints!`](@ref).

# Validation

  - Raises an error when `long = true` and any bound is negative.

# Returns

  - `nothing`.

# Related

  - [`w_neg_flag`](@ref)
  - [`w_finite_flag`](@ref)
  - [`WeightBounds`](@ref)
  - [`set_budget_constraints!`](@ref)
"""
function set_weight_constraints!(args...)
    return nothing
end
function set_weight_constraints!(model::JuMP.Model, wb::WeightBounds,
                                 bgt::Option{<:Num_BgtRg}, sbgt::Option{<:Num_BgtRg},
                                 long::Bool = false; gbgt::Option{<:Num_BgtRg} = nothing)
    lb = wb.lb
    ub = wb.ub
    flag = w_neg_flag(lb) || w_neg_flag(ub)
    @argcheck(!(long && flag), "Long-only strategy cannot have negative weight limits")
    w = get_w(model)
    N = length(w)
    k = get_k(model)
    sc = get_constraint_scale(model)
    if w_finite_flag(lb)
        JuMP.@constraint(model, w_lb, sc * (w ⊖ k * lb) >= 0)
    end
    if w_finite_flag(ub)
        JuMP.@constraint(model, w_ub, sc * (w ⊖ k * ub) <= 0)
    end
    set_budget_constraints!(model, bgt, w)
    if flag
        lw, sw = if !haskey(model, :sw)
            JuMP.@variables(model, begin
                                lw[1:N] >= 0
                                sw[1:N] >= 0
                            end)
        else
            model[:lw], model[:sw]
        end
        JuMP.@constraints(model, begin
                              w_lw, sc * (w - lw) <= 0
                              w_sw, sc * (w + sw) >= 0
                          end)
        # Only a declaration when no head made a stronger one: a head that derives `w` from the
        # parts runs this too, and its identity survives these bounds.
        set_decomposition_contract!(model, PartsBoundWeights())
        set_long_short_budget_constraints!(model, bgt, sbgt)
        set_gross_budget_constraints!(model, gbgt)
    elseif !flag && !haskey(model, :lw)
        JuMP.@expression(model, lw, w)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Return `true` when `x` is non-zero.

# Arguments

  - `x`: Value to check. Accepts `nothing`, a scalar `Number`, or a `VecNum`.

# Returns

  - `flag::Bool`: `false` when `x` is `nothing`; `!iszero(x)` for a scalar; `any(!iszero, x)` for a vector.

# Related

  - [`set_linear_weight_constraints!`](@ref)
"""
function non_zero_real_or_vec(::Nothing)
    return false
end
function non_zero_real_or_vec(x::Number)
    return !iszero(x)
end
function non_zero_real_or_vec(x::VecNum)
    return any(!iszero, x)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add linear inequality and equality weight constraints to the JuMP optimisation model.

The fall-through method does nothing. The concrete method iterates over the collection of [`LinearConstraint`](@ref) objects `lcms` and adds `A * w ≤ k * B` (inequality) and `A * w = k * B` (equality) constraints for each entry.

# Mathematical definition

```math
\\begin{align}
\\mathbf{A}_{\\mathrm{ineq}} \\boldsymbol{w} \\leq k \\boldsymbol{B}_{\\mathrm{ineq}}\\,, \\\\
\\mathbf{A}_{\\mathrm{eq}} \\boldsymbol{w} &= k \\boldsymbol{B}_{\\mathrm{eq}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - ``\\mathbf{A}_{\\mathrm{ineq}}``, ``\\mathbf{A}_{\\mathrm{eq}}``: Constraint coefficient matrices for inequality and equality constraints.
  - ``\\boldsymbol{B}_{\\mathrm{ineq}}``, ``\\boldsymbol{B}_{\\mathrm{eq}}``: Constraint response vectors for inequality and equality constraints.

# Arguments

  - $(arg_dict[:model])
  - `lcms`: Collection of [`LinearConstraint`](@ref) objects defining the linear constraints.
  - `key_ineq::Symbol`: Base key for naming inequality constraints in the model.
  - `key_eq::Symbol`: Base key for naming equality constraints in the model.

# Returns

  - `nothing`.

# Related

  - [`non_zero_real_or_vec`](@ref)
  - [`LinearConstraint`](@ref)
  - [`set_weight_constraints!`](@ref)
"""
function set_linear_weight_constraints!(args...)
    return nothing
end
function set_linear_weight_constraints!(model::JuMP.Model, lcms::Lc_VecLc, key_ineq::Symbol,
                                        key_eq::Symbol)
    w = get_w(model)
    k = get_k(model)
    sc = get_constraint_scale(model)
    for (i, lcm) in enumerate(lcms)
        if !isnothing(lcm.ineq)
            A = lcm.ineq.A
            B = lcm.ineq.B
            model[Symbol(key_ineq, i)] = JuMP.@constraint(model, sc * (A * w - k * B) <= 0)
        end
        if !isnothing(lcm.eq)
            A = lcm.eq.A
            B = lcm.eq.B
            model[Symbol(key_eq, i)] = JuMP.@constraint(model, sc * (A * w - k * B) == 0)
        end
    end
    return nothing
end
