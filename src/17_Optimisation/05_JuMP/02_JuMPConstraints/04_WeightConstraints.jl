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

This is the first step of the block builder, [`set_block_constraints!`](@ref), so each [`ConstraintBlock`](@ref) of a model adds its own bounds. The rows go under `prefix`, the namespace of the block, and the builder reads the weights registered under it. The long and short parts belong to the weights. The first block that admits a short position makes them, with their rows `w_lw` and `w_sw`, and a later block reads them. A later block makes parts of its own under its prefix only when the weights already have a long part and no short part, because a block before it is long-only. Its short and gross budgets then read its own parts, as they read the anonymous parts of the set before this change.

**Breaking change.** The positional `long::Bool` went. A long-only head checks its bounds with [`assert_long_only_weight_bounds`](@ref) where it built its weights, and the block builder adds the bounds.

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
  - `bgt`: Optional total budget constraint. A number, a [`BudgetRange`](@ref), or a [`BudgetCostEstimator`](@ref), which is the whole of [`JuMPOptimiser`](@ref)'s own `bgt` bound.
  - `sbgt`: Optional short-side budget constraint. A [`BudgetCostEstimator`](@ref) `bgt` forbids one, so the pair never meets here.
  - `gbgt`: Optional gross (leverage) budget constraint, applied only when the weight bounds admit shorts. See [`set_gross_budget_constraints!`](@ref).
  - $(arg_dict[:prefix_kw])

!!! note

    Optimiser heads do not call this method directly. The block builder [`set_block_constraints!`](@ref) calls it with the net budget of the owner of the block, and with the short and gross budgets of the block, so a budget field cannot reach one head and miss the others.

!!! note

    The budgets set here **bound** the realised exposures rather than pinning them, because `lw` and `sw` are upper bounds on the parts of `w`. Pinning them is the `xbgt` option of [`short_mip_threshold_constraints`](@ref), applied later by [`set_mip_constraints!`](@ref).

# Returns

  - `nothing`.

# Related

  - [`w_neg_flag`](@ref)
  - [`w_finite_flag`](@ref)
  - [`WeightBounds`](@ref)
  - [`set_budget_constraints!`](@ref)
  - [`set_gross_budget_constraints!`](@ref)
  - [`JuMPOptimiser`](@ref)
"""
function set_weight_constraints!(args...; kwargs...)
    return nothing
end
function set_weight_constraints!(model::JuMP.Model, wb::WeightBounds,
                                 bgt::Option{<:Num_BgtCE}, sbgt::Option{<:Num_BgtRg};
                                 gbgt::Option{<:Num_BgtRg} = nothing,
                                 prefix::Symbol = Symbol(""))
    lb = wb.lb
    ub = wb.ub
    flag = w_neg_flag(lb) || w_neg_flag(ub)
    w = get_w(model, prefix)
    N = length(w)
    k = get_k(model)
    sc = get_constraint_scale(model)
    if w_finite_flag(lb)
        state_set!(model, prefix, :w_lb, JuMP.@constraint(model, sc * (w ⊖ k * lb) >= 0))
    end
    if w_finite_flag(ub)
        state_set!(model, prefix, :w_ub, JuMP.@constraint(model, sc * (w ⊖ k * ub) <= 0))
    end
    set_budget_constraints!(model, bgt, w; prefix = prefix)
    set_gross_exposure_bound!(model, gross_exposure_bound(wb, bgt, sbgt, gbgt, N))
    if flag
        set_weight_parts!(model, w, prefix)
        set_long_short_budget_constraints!(model, bgt, sbgt; prefix = prefix)
        set_gross_budget_constraints!(model, gbgt; prefix = prefix)
    elseif !shared_has(model, :lw)
        JuMP.@expression(model, lw, w)
    end
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Add the long and short parts of the weights, `lw` and `sw`, for a block whose bounds admit a short position.

The parts belong to the weights, so they go in the shared Model State once. The first block that admits a short position makes them, or reads the parts that its head made, and adds the rows that tie them to `w`. A later block reads the same parts and adds no row. A head that derives `w` from the parts keeps its identity, because the rows here only bound the parts.

A block before this one can be long-only. Then the weights already have the long part `lw = w` and no short part. This block cannot change that entry, because the rows of the earlier block read it. So the function adds nothing, and a budget of the block that reads the parts makes parts of its own under `prefix` through [`weight_parts!`](@ref). This is the case of a programme set that admits a short position in the model of a long-only leader.

# JuMP formulation

## Variables

  - `lw`, `sw`: ``N \\times 1``, each bounded below by zero, shared.

## Constraints

  - `w_lw`: ``s_c (\\boldsymbol{w} - \\boldsymbol{lw}) \\leq 0``.
  - `w_sw`: ``s_c (\\boldsymbol{w} + \\boldsymbol{sw}) \\geq 0``.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:w_port])
  - $(math_dict[:N])

# Arguments

  - $(arg_dict[:model])
  - `w`: The weights registered under `prefix`.
  - `prefix`: The Model State namespace of the block.

# Returns

  - `nothing`.

# Related

  - [`set_weight_constraints!`](@ref)
  - [`weight_parts!`](@ref)
  - [`set_decomposition_contract!`](@ref)
"""
function set_weight_parts!(model::JuMP.Model, w::AbstractVector, ::Symbol)::Nothing
    if state_has(model, Symbol(""), :w_lw) ||
       (shared_has(model, :lw) && !shared_has(model, :sw))
        return nothing
    end
    N = length(w)
    sc = get_constraint_scale(model)
    lw, sw = if !shared_has(model, :sw)
        JuMP.@variables(model, begin
                            lw[1:N] >= 0
                            sw[1:N] >= 0
                        end)
    else
        shared_get(model, :lw), shared_get(model, :sw)
    end
    JuMP.@constraints(model, begin
                          w_lw, sc * (w - lw) <= 0
                          w_sw, sc * (w + sw) >= 0
                      end)
    # Only a declaration when no head made a stronger one: a head that derives `w` from the
    # parts runs this too, and its identity survives these bounds.
    set_decomposition_contract!(model, PartsBoundWeights())
    return nothing
end
"""
    weight_parts!(model::JuMP.Model, prefix::Symbol) -> Tuple

Return the long and short parts of the weights that the budgets of a block read.

A block reads the shared parts of the weights when they exist, see [`set_weight_parts!`](@ref). When a long-only block came first, the weights have the long part `lw = w` and no short part. Then the first budget of the block that reads the parts makes parts of its own under `prefix`, with their rows, and a later budget of the block reads them. So a block that states no short or gross budget makes no parts.

# JuMP formulation

## Variables

  - `lw`, `sw`: ``N \\times 1``, each bounded below by zero, under `prefix`, when the function makes parts of the block.

## Constraints

  - `w_lw`: ``s_c (\\boldsymbol{w} - \\boldsymbol{lw}) \\leq 0``, under `prefix`.
  - `w_sw`: ``s_c (\\boldsymbol{w} + \\boldsymbol{sw}) \\geq 0``, under `prefix`.

Where:

  - $(math_dict[:sc_scale])
  - $(math_dict[:w_port]) It is the weights registered under `prefix`.
  - $(math_dict[:N])

# Arguments

  - $(arg_dict[:model])
  - `prefix`: The Model State namespace of the block.

# Returns

  - `(lw, sw)::Tuple`: The long and the short parts.

# Related

  - [`set_weight_parts!`](@ref)
  - [`set_long_short_budget_constraints!`](@ref)
  - [`set_gross_budget_constraints!`](@ref)
"""
function weight_parts!(model::JuMP.Model, prefix::Symbol)::Tuple
    if shared_has(model, :sw)
        return shared_get(model, :lw), shared_get(model, :sw)
    end
    if !state_has(model, prefix, :sw)
        w = get_w(model, prefix)
        N = length(w)
        sc = get_constraint_scale(model)
        lw = state_set!(model, prefix, :lw, JuMP.@variable(model, [1:N], lower_bound = 0))
        sw = state_set!(model, prefix, :sw, JuMP.@variable(model, [1:N], lower_bound = 0))
        state_set!(model, prefix, :w_lw, JuMP.@constraint(model, sc * (w - lw) <= 0))
        state_set!(model, prefix, :w_sw, JuMP.@constraint(model, sc * (w + sw) >= 0))
    end
    return state_get(model, prefix, :lw), state_get(model, prefix, :sw)
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Check that weight bounds admit no short position, for a head whose formulation is long-only.

[`RiskBudgeting`](@ref) and [`RelaxedRiskBudgeting`](@ref) take the logarithm or the cone of the asset weights, which holds only for non-negative weights. The head calls this check where it builds its weights. The block builder [`set_block_constraints!`](@ref) then adds the bounds.

# Arguments

  - $(arg_dict[:wb_arg])

# Validation

  - No bound is negative. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`w_neg_flag`](@ref)
  - [`set_weight_constraints!`](@ref)
"""
function assert_long_only_weight_bounds(wb::WeightBounds)::Nothing
    @argcheck(!(w_neg_flag(wb.lb) || w_neg_flag(wb.ub)),
              "Long-only strategy cannot have negative weight limits")
    return nothing
end
"""
    budget_upper_bound(bgt) -> Number

Return the upper bound that a budget puts on a sum of weights, or `Inf` when it puts none.

A number pins the sum, so it is its own bound. A [`BudgetRange`](@ref) bounds the sum by its `ub`. `nothing`, a range without `ub`, and a budget that also pays costs give `Inf`.

# Arguments

  - `bgt`: The budget, as [`set_weight_constraints!`](@ref) receives it.

# Returns

  - `ub::Number`: The upper bound of the sum, in units of ``k``.

# Related

  - [`gross_exposure_bound`](@ref)
"""
function budget_upper_bound(bgt::Number)
    return bgt
end
function budget_upper_bound(bgt::BudgetRange)
    return isnothing(bgt.ub) ? Inf : bgt.ub
end
function budget_upper_bound(args...)
    return Inf
end
"""
    gross_exposure_bound(wb::WeightBounds, bgt, sbgt, gbgt, N::Integer) -> Number

Return an upper bound on the gross exposure of every weight vector that the weight bounds and the budgets admit, or `Inf` when they admit no bound.

The bound is the smallest of the bounds below that apply.

```math
\\begin{align}
\\lVert \\boldsymbol{w} \\rVert_{1} &\\leq \\sum_{i=1}^{N} \\max\\left(\\lvert \\ell_{i} \\rvert,\\, \\lvert u_{i} \\rvert\\right)\\,, \\\\
\\lVert \\boldsymbol{w} \\rVert_{1} &\\leq \\overline{B}\\,, \\quad \\text{if } \\boldsymbol{\\ell} \\geq \\boldsymbol{0}\\,, \\\\
\\lVert \\boldsymbol{w} \\rVert_{1} &\\leq \\min\\left(\\overline{G},\\, \\overline{B} + 2 \\overline{S}\\right)\\,, \\quad \\text{if a bound is negative}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - ``\\boldsymbol{\\ell}``, ``\\boldsymbol{u}``: Lower and upper bound vectors from `wb`.
  - ``\\overline{B}``, ``\\overline{S}``, ``\\overline{G}``: Upper bounds of the budget, the short budget and the gross budget, from [`budget_upper_bound`](@ref).
  - ``w^{+}_{i}``, ``w^{-}_{i}``: Long and short parts of weight ``i``, the model's `lw` and `sw`.
  - $(math_dict[:k_budget])
  - $(math_dict[:N])

The first bound holds for all weights. The second holds because non-negative weights have a gross exposure equal to their sum. The third holds because ``\\lvert w_{i} \\rvert \\leq w^{+}_{i} + w^{-}_{i}``, the long budget bounds the sum of the long parts by ``\\overline{B} + \\overline{S}``, and the short budget bounds the sum of the short parts by ``\\overline{S}``. Each bound is in units of ``k``, as the constraints of [`set_weight_constraints!`](@ref) are.

# Arguments

  - $(arg_dict[:wb_arg])
  - `bgt`: The budget.
  - `sbgt`: The short budget.
  - `gbgt`: The gross budget.
  - `N::Integer`: The number of assets.

# Returns

  - `g::Number`: The bound, or `Inf`.

# Related

  - [`set_weight_constraints!`](@ref)
  - [`set_gross_exposure_bound!`](@ref)
  - [`mip_big_m`](@ref)
"""
function gross_exposure_bound(wb::WeightBounds, bgt, sbgt, gbgt, N::Integer)
    lb = wb.lb
    ub = wb.ub
    g = if isnothing(lb) || isnothing(ub)
        Inf
    else
        m = max.(abs.(lb), abs.(ub))
        isa(m, Number) ? N * m : sum(m)
    end
    return if w_neg_flag(lb) || w_neg_flag(ub)
        min(g, budget_upper_bound(gbgt),
            budget_upper_bound(bgt) + 2 * budget_upper_bound(sbgt))
    elseif !isnothing(lb)
        min(g, budget_upper_bound(bgt))
    else
        g
    end
end
"""
    set_gross_exposure_bound!(model::JuMP.Model, g::Number)

Record `g` as the shared Model State entry `:w_gross_ub`, the bound on the gross exposure of the head's weights.

A second call keeps the smaller bound, because the constraints of both calls hold for the same weights. [`mip_big_m`](@ref) reads the entry.

# Arguments

  - $(arg_dict[:model])
  - `g::Number`: The bound, from [`gross_exposure_bound`](@ref).

# Returns

  - `g::Number`: The bound that the model records.

# Related

  - [`gross_exposure_bound`](@ref)
  - [`mip_big_m`](@ref)
"""
function set_gross_exposure_bound!(model::JuMP.Model, g::Number)
    if shared_has(model, :w_gross_ub)
        g = min(g, shared_get(model, :w_gross_ub))
    end
    return shared_set!(model, :w_gross_ub, g)
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
  - `name_ineq::Symbol`: Bare Model State entry name seeding the indexed inequality
    constraint entries.
  - `name_eq::Symbol`: Bare Model State entry name seeding the indexed equality constraint
    entries.
  - $(arg_dict[:prefix_kw])

# Returns

  - `nothing`.

# Related

  - [`non_zero_real_or_vec`](@ref)
  - [`LinearConstraint`](@ref)
  - [`set_weight_constraints!`](@ref)
"""
function set_linear_weight_constraints!(args...; prefix::Symbol = Symbol(""))
    return nothing
end
function set_linear_weight_constraints!(model::JuMP.Model, lcms::Lc_VecLc,
                                        name_ineq::Symbol, name_eq::Symbol;
                                        prefix::Symbol = Symbol(""))
    w = get_w(model, prefix)
    k = get_k(model)
    sc = get_constraint_scale(model)
    for (i, lcm) in enumerate(lcms)
        if !isnothing(lcm.ineq)
            A = lcm.ineq.A
            B = lcm.ineq.B
            state_set!(model, prefix, name_ineq, i,
                       JuMP.@constraint(model, sc * (A * w - k * B) <= 0))
        end
        if !isnothing(lcm.eq)
            A = lcm.eq.A
            B = lcm.eq.B
            state_set!(model, prefix, name_eq, i,
                       JuMP.@constraint(model, sc * (A * w - k * B) == 0))
        end
    end
    return nothing
end
