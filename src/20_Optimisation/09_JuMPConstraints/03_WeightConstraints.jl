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
k \\boldsymbol{\\ell} \\leq \\boldsymbol{w} \\leq k \\boldsymbol{u}
```

where ``\\boldsymbol{\\ell}`` and ``\\boldsymbol{u}`` are the lower and upper bound vectors from `wb`, and ``k`` is the budget scaling variable.

# Arguments

  - $(arg_dict[:model])
  - $(arg_dict[:wb_arg])
  - `bgt`: Optional total budget constraint (number or [`BudgetRange`](@ref)).
  - `sbgt`: Optional short-side budget constraint.
  - `long::Bool = false`: When `true`, raises an error if any bound is negative.

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
                                 long::Bool = false)
    lb = wb.lb
    ub = wb.ub
    flag = w_neg_flag(lb) || w_neg_flag(ub)
    @argcheck(!(long && flag), "Long-only strategy cannot have negative weight limits")
    w = model[:w]
    N = length(w)
    k = model[:k]
    sc = model[:sc]
    if w_finite_flag(lb)
        JuMP.@constraint(model, w_lb, sc * (w - k * lb) >= 0)
    end
    if w_finite_flag(ub)
        JuMP.@constraint(model, w_ub, sc * (w - k * ub) <= 0)
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
        set_long_short_budget_constraints!(model, bgt, sbgt)
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
\\mathbf{A}_{\\mathrm{ineq}} \\boldsymbol{w} \\leq k \\boldsymbol{B}_{\\mathrm{ineq}}, \\qquad \\mathbf{A}_{\\mathrm{eq}} \\boldsymbol{w} = k \\boldsymbol{B}_{\\mathrm{eq}}
```

where ``k`` is the budget scaling variable.

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
    w = model[:w]
    k = model[:k]
    sc = model[:sc]
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
