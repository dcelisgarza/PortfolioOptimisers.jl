"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for all budget constraint estimators.

# Related

  - [`BudgetEstimator`](@ref)
  - [`BudgetCostEstimator`](@ref)
  - [`BudgetRange`](@ref)
"""
abstract type BudgetConstraintEstimator <: JuMPConstraintEstimator end
"""
    const Num_BgtCE = Union{<:Number, <:BudgetConstraintEstimator}

Union of scalar budget values and [`BudgetConstraintEstimator`](@ref) instances.
"""
const Num_BgtCE = Union{<:Number, <:BudgetConstraintEstimator}
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for estimators that specify the portfolio budget range (sum of
weights).

# Related

  - [`BudgetRange`](@ref)
"""
abstract type BudgetEstimator <: BudgetConstraintEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for estimators that specify transaction cost budgets.

# Related

  - [`BudgetCosts`](@ref)
  - [`BudgetMarketImpact`](@ref)
"""
abstract type BudgetCostEstimator <: BudgetConstraintEstimator end
"""
    set_budget_costs!(args...)

Set transaction cost budget constraints in the JuMP model.

No-op fallback when no cost budget is specified.

# Arguments

  - `args...`: Arguments (ignored).

# Returns

  - `nothing`.

# Related

  - [`set_long_short_budget_constraints!`](@ref)
"""
function set_budget_costs!(args...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Specifies the portfolio budget constraint as a closed interval ``[\\mathrm{lb}, \\mathrm{ub}]``
on the sum of weights. At least one of `lb` or `ub` must be provided.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BudgetRange(; lb::Option{<:Number} = 1.0, ub::Option{<:Number} = 1.0) -> BudgetRange

Keywords correspond to the struct's fields.

## Validation

  - At least one of `lb`, `ub` must not be `nothing`.
  - `lb` and `ub` must be finite.
  - `lb <= ub` when both are provided.

# Related

  - [`BudgetCosts`](@ref)
  - [`BudgetMarketImpact`](@ref)
  - [`set_weight_constraints!`](@ref)
"""
@concrete struct BudgetRange <: BudgetEstimator
    "$(field_dict[:lb])"
    lb
    "$(field_dict[:ub])"
    ub
    function BudgetRange(lb::Option{<:Number}, ub::Option{<:Number})
        lb_flag = isnothing(lb)
        ub_flag = isnothing(ub)
        @argcheck(!(lb_flag && ub_flag))
        if !lb_flag
            @argcheck(isfinite(lb))
        end
        if !ub_flag
            @argcheck(isfinite(ub))
        end
        if !lb_flag && !ub_flag
            @argcheck(lb <= ub)
        end
        return new{typeof(lb), typeof(ub)}(lb, ub)
    end
end
function BudgetRange(; lb::Option{<:Number} = 1.0, ub::Option{<:Number} = 1.0)
    return BudgetRange(lb, ub)
end
"""
    const Num_BgtRg = Union{<:Number, <:BudgetRange}

Alias for a scalar budget value or budget range.

Matches either a plain number (fixed budget, e.g. `1.0`) or a [`BudgetRange`](@ref) (interval budget constraint). Used for dispatch in budget constraint generation.

# Related

  - [`BudgetRange`](@ref)
  - [`set_long_short_budget_constraints!`](@ref)
"""
const Num_BgtRg = Union{<:Number, <:BudgetRange}
"""
    budget_view(bgt, i)

Get a view or subset of the budget constraint for index `i`.

For scalar or [`BudgetRange`](@ref) inputs, returns the input unchanged (budget applies to all assets).

# Arguments

  - `bgt`: Budget value, [`BudgetRange`](@ref), or `nothing`.
  - `i`: Index (ignored for scalar/range budgets).

# Returns

  - The budget unchanged, or `nothing`.

# Related

  - [`BudgetRange`](@ref)
  - [`set_long_short_budget_constraints!`](@ref)
"""
function budget_view(bgt::Num_BgtRg, ::Any)
    return bgt
end
"""
    set_budget_constraints!(args...)
    set_budget_constraints!(model::JuMP.Model, val::Number, w::VecNum)
    set_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, w::VecNum)
    set_budget_constraints!(model::JuMP.Model, bgt::BudgetCosts, w::VecNum)
    set_budget_constraints!(model::JuMP.Model, bgt::BudgetMarketImpact, w::VecNum)

Add budget constraints to the JuMP optimisation model.

The fall-through method does nothing. The concrete methods add the appropriate portfolio budget constraint based on the type of budget specification provided.

# Mathematical definition

Fixed / range:

```math
\\begin{align}
k \\cdot \\mathrm{lb} &\\leq \\sum_i w_i \\leq k \\cdot \\mathrm{ub}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - ``\\mathrm{lb}``, ``\\mathrm{ub}``: Lower and upper budget bounds.

Linear cost budget ([`BudgetCosts`](@ref)):

```math
\\begin{align}
\\sum_i w_i + \\boldsymbol{v}_p^\\intercal \\boldsymbol{w}_p + \\boldsymbol{v}_n^\\intercal \\boldsymbol{w}_n \\in [k\\,\\mathrm{lb},\\; k\\,\\mathrm{ub}]\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - ``\\boldsymbol{w}_p``, ``\\boldsymbol{w}_n``: Positive and negative weight increments.
  - ``\\boldsymbol{v}_p``, ``\\boldsymbol{v}_n``: Cost coefficient vectors for positive and negative changes.
  - ``\\mathrm{lb}``, ``\\mathrm{ub}``: Lower and upper budget bounds.

Power-law market-impact budget ([`BudgetMarketImpact`](@ref)):

```math
\\begin{align}
\\sum_i w_i + \\boldsymbol{v}_p^\\intercal \\boldsymbol{w}_p^\\beta + \\boldsymbol{v}_n^\\intercal \\boldsymbol{w}_n^\\beta \\in [k\\,\\mathrm{lb},\\; k\\,\\mathrm{ub}]\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - ``\\boldsymbol{w}_p``, ``\\boldsymbol{w}_n``: Positive and negative weight increments.
  - ``\\boldsymbol{v}_p``, ``\\boldsymbol{v}_n``: Market-impact cost coefficient vectors.
  - ``\\beta \\in (0, 1]``: Market-impact power exponent.
  - ``\\mathrm{lb}``, ``\\mathrm{ub}``: Lower and upper budget bounds.

# Arguments

  - $(arg_dict[:model])
  - `val::Number`: Fixed budget scalar.
  - `bgt`: Budget constraint specification ([`BudgetRange`](@ref), [`BudgetCosts`](@ref), or [`BudgetMarketImpact`](@ref)).
  - `w`: Portfolio weight vector.

# Returns

  - `nothing`.

# Related

  - [`BudgetRange`](@ref)
  - [`BudgetCosts`](@ref)
  - [`BudgetMarketImpact`](@ref)
  - [`set_weight_constraints!`](@ref)
"""
function set_budget_constraints!(args...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Budget constraint that accounts for linear transaction costs.

# Mathematical definition

Models the portfolio budget as:

```math
\\begin{align}
\\boldsymbol{v}_p^\\intercal \\boldsymbol{w}_p + \\boldsymbol{v}_n^\\intercal \\boldsymbol{w}_n
\\in [\\mathrm{lb}, \\mathrm{ub}]\\,.
\\end{align}
```

Where:

  - ``\\boldsymbol{w}_p``, ``\\boldsymbol{w}_n``: Positive and negative weight increments.
  - ``\\boldsymbol{v}_p``, ``\\boldsymbol{v}_n``: Cost coefficient vectors for positive and negative changes.
  - ``\\mathrm{lb}``, ``\\mathrm{ub}``: Lower and upper budget bounds.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BudgetCosts(;
        bgt::Num_BgtRg = 1.0,
        w::VecNum,
        vp::Num_VecNum = 1.0,
        vn::Num_VecNum = 1.0,
        up::Num_VecNum = 1.0,
        un::Num_VecNum = 1.0
    ) -> BudgetCosts

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(w)`.
  - If `vp` is a vector: `!isempty(vp)` and all elements `>= 0`. If scalar: `>= 0`.
  - If `vn` is a vector: `!isempty(vn)` and all elements `>= 0`. If scalar: `>= 0`.
  - If `up` is a vector: `!isempty(up)` and all elements `>= 0`. If scalar: `>= 0`.
  - If `un` is a vector: `!isempty(un)` and all elements `>= 0`. If scalar: `>= 0`.

# Related

  - [`BudgetRange`](@ref)
  - [`BudgetMarketImpact`](@ref)
"""
@concrete struct BudgetCosts <: BudgetCostEstimator
    "Budget target or range."
    bgt
    "$(field_dict[:pw])"
    w
    "Cost coefficients for positive weight changes. Non-negative."
    vp
    "Cost coefficients for negative weight changes. Non-negative."
    vn
    "Upper limit on positive weight changes. Non-negative."
    up
    "Upper limit on negative weight changes. Non-negative."
    un
    function BudgetCosts(bgt::Num_BgtRg, w::VecNum, vp::Num_VecNum, vn::Num_VecNum,
                         up::Num_VecNum, un::Num_VecNum)
        @argcheck(!isempty(w))
        if isa(vp, VecNum)
            @argcheck(!isempty(vp))
            @argcheck(all(x -> zero(x) <= x, vp))
        else
            @argcheck(vp >= zero(vp))
        end
        if isa(vn, VecNum)
            @argcheck(!isempty(vn))
            @argcheck(all(x -> zero(x) <= x, vn))
        else
            @argcheck(vn >= zero(vn))
        end
        if isa(up, VecNum)
            @argcheck(!isempty(up))
            @argcheck(all(x -> zero(x) <= x, up))
        else
            @argcheck(up >= zero(up))
        end
        if isa(un, VecNum)
            @argcheck(!isempty(un))
            @argcheck(all(x -> zero(x) <= x, un))
        else
            @argcheck(un >= zero(un))
        end
        return new{typeof(bgt), typeof(w), typeof(vp), typeof(vn), typeof(up), typeof(un)}(bgt,
                                                                                           w,
                                                                                           vp,
                                                                                           vn,
                                                                                           up,
                                                                                           un)
    end
end
function BudgetCosts(; bgt::Num_BgtRg = 1.0, w::VecNum, vp::Num_VecNum = 1.0,
                     vn::Num_VecNum = 1.0, up::Num_VecNum = 1.0, un::Num_VecNum = 1.0)
    return BudgetCosts(bgt, w, vp, vn, up, un)
end
function budget_view(bgt::BudgetCosts, i)
    w = view(bgt.w, i)
    vp = nothing_scalar_array_view(bgt.vp, i)
    vn = nothing_scalar_array_view(bgt.vn, i)
    up = nothing_scalar_array_view(bgt.up, i)
    un = nothing_scalar_array_view(bgt.un, i)
    return BudgetCosts(; bgt = bgt.bgt, w = w, vp = vp, vn = vn, up = up, un = un)
end
"""
$(DocStringExtensions.TYPEDEF)

Budget constraint that accounts for non-linear (power-law) market impact costs. Extends
[`BudgetCosts`](@ref) with a `beta` exponent controlling the concavity of the market
impact function.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BudgetMarketImpact(;
        bgt::Num_BgtRg = 1.0,
        w::VecNum,
        vp::Num_VecNum = 1.0,
        vn::Num_VecNum = 1.0,
        up::Num_VecNum = 1.0,
        un::Num_VecNum = 1.0,
        beta::Number = 2/3
    ) -> BudgetMarketImpact

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(w)`.
  - If `vp` is a vector: `!isempty(vp)` and all elements `>= 0`. If scalar: `>= 0`.
  - If `vn` is a vector: `!isempty(vn)` and all elements `>= 0`. If scalar: `>= 0`.
  - If `up` is a vector: `!isempty(up)` and all elements `>= 0`. If scalar: `>= 0`.
  - If `un` is a vector: `!isempty(un)` and all elements `>= 0`. If scalar: `>= 0`.
  - `0 <= beta <= 1`.

# Related

  - [`BudgetRange`](@ref)
  - [`BudgetCosts`](@ref)
"""
@concrete struct BudgetMarketImpact <: BudgetCostEstimator
    "Budget target or range."
    bgt
    "$(field_dict[:pw])"
    w
    "Cost coefficients for positive weight changes. Non-negative."
    vp
    "Cost coefficients for negative weight changes. Non-negative."
    vn
    "Upper limit on positive weight changes. Non-negative."
    up
    "Upper limit on negative weight changes. Non-negative."
    un
    "Market impact exponent in `(0, 1]`."
    beta
    function BudgetMarketImpact(bgt::Num_BgtRg, w::VecNum, vp::Num_VecNum, vn::Num_VecNum,
                                up::Num_VecNum, un::Num_VecNum, beta::Number)
        @argcheck(!isempty(w))
        if isa(vp, VecNum)
            @argcheck(!isempty(vp))
            @argcheck(all(x -> zero(x) <= x, vp))
        else
            @argcheck(vp >= zero(vp))
        end
        if isa(vn, VecNum)
            @argcheck(!isempty(vn))
            @argcheck(all(x -> zero(x) <= x, vn))
        else
            @argcheck(vn >= zero(vn))
        end
        if isa(up, VecNum)
            @argcheck(!isempty(up))
            @argcheck(all(x -> zero(x) <= x, up))
        else
            @argcheck(up >= zero(up))
        end
        if isa(un, VecNum)
            @argcheck(!isempty(un))
            @argcheck(all(x -> zero(x) <= x, un))
        else
            @argcheck(un >= zero(un))
        end
        @argcheck(zero(beta) <= beta <= one(beta), DomainError)
        return new{typeof(bgt), typeof(w), typeof(vp), typeof(vn), typeof(up), typeof(un),
                   typeof(beta)}(bgt, w, vp, vn, up, un, beta)
    end
end
function BudgetMarketImpact(; bgt::Num_BgtRg = 1.0, w::VecNum, vp::Num_VecNum = 1.0,
                            vn::Num_VecNum = 1.0, up::Num_VecNum = 1.0,
                            un::Num_VecNum = 1.0, beta::Number = 2 / 3)
    return BudgetMarketImpact(bgt, w, vp, vn, up, un, beta)
end
function budget_view(bgt::BudgetMarketImpact, i)
    w = view(bgt.w, i)
    vp = nothing_scalar_array_view(bgt.vp, i)
    vn = nothing_scalar_array_view(bgt.vn, i)
    up = nothing_scalar_array_view(bgt.up, i)
    un = nothing_scalar_array_view(bgt.un, i)
    return BudgetMarketImpact(; bgt = bgt.bgt, w = w, vp = vp, vn = vn, up = up, un = un,
                              beta = bgt.beta)
end
function set_budget_constraints!(model::JuMP.Model, val::Number, w::VecNum)
    k = model[:k]
    sc = model[:sc]
    JuMP.@constraint(model, bgt, sc * (sum(w) - k * val) == 0)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, w::VecNum)
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    ub = bgt.ub
    if !isnothing(lb)
        JuMP.@constraint(model, bgt_lb, sc * (sum(w) - k * lb) >= 0)
    end
    if !isnothing(ub)
        JuMP.@constraint(model, bgt_ub, sc * (sum(w) - k * ub) <= 0)
    end
    return nothing
end
"""
    set_long_short_budget_constraints!(args...)

Set budget constraints for long and short portfolio positions in the JuMP model.

Various overloads handle different budget types (fixed, range), dispatching on the presence or absence of long/short budget configurations.

# Arguments

  - `args...`: JuMP model and budget parameters.

# Returns

  - `nothing`.

# Related

  - [`BudgetRange`](@ref)
  - [`set_cost_budget_constraints!`](@ref)
"""
function set_long_short_budget_constraints!(args...)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Number, ::Nothing)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    JuMP.@constraint(model, lbgt, sc * (sum(lw) - k * bgt) == 0)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, ::Nothing, sbgt::Number)
    sw = model[:sw]
    k = model[:k]
    sc = model[:sc]
    JuMP.@constraint(model, sbgt, sc * (sum(sw) - k * sbgt) == 0)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Number, sbgt::Number)
    lw = model[:lw]
    sw = model[:sw]
    k = model[:k]
    sc = model[:sc]
    JuMP.@constraints(model, begin
                          lbgt, sc * (sum(lw) - k * (bgt + sbgt)) == 0
                          sbgt, sc * (sum(sw) - k * sbgt) == 0
                      end)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, ::Nothing)
    lw = model[:lw]
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    if !isnothing(lb)
        JuMP.@constraint(model, lbgt_lb, sc * (sum(lw) - k * lb) >= 0)
    end
    ub = bgt.ub
    if !isnothing(ub)
        JuMP.@constraint(model, lbgt_ub, sc * (sum(lw) - k * ub) <= 0)
    end
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, ::Nothing, sbgt::BudgetRange)
    sw = model[:sw]
    k = model[:k]
    sc = model[:sc]
    lb = sbgt.lb
    if !isnothing(lb)
        JuMP.@constraint(model, sbgt_lb, sc * (sum(sw) - k * lb) >= 0)
    end
    ub = sbgt.ub
    if !isnothing(ub)
        JuMP.@constraint(model, sbgt_ub, sc * (sum(sw) - k * ub) <= 0)
    end
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::BudgetRange,
                                            sbgt::Number)
    lw = model[:lw]
    sw = model[:sw]
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    if !isnothing(lb)
        JuMP.@constraint(model, lbgt_lb, sc * (sum(lw) - k * (lb + sbgt)) >= 0)
    end
    ub = bgt.ub
    if !isnothing(ub)
        JuMP.@constraint(model, lbgt_ub, sc * (sum(lw) - k * (ub + sbgt)) <= 0)
    end
    JuMP.@constraint(model, sbgt, sc * (sum(sw) - k * sbgt) == 0)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Number,
                                            sbgt::BudgetRange)
    lw = model[:lw]
    sw = model[:sw]
    k = model[:k]
    sc = model[:sc]
    lb = sbgt.lb
    if !isnothing(lb)
        JuMP.@constraints(model, begin
                              lbgt_lb, sc * (sum(lw) - k * (lb + bgt)) >= 0
                              sbgt_lb, sc * (sum(sw) - k * lb) >= 0
                          end)
    end
    ub = sbgt.ub
    if !isnothing(ub)
        JuMP.@constraints(model, begin
                              lbgt_ub, sc * (sum(lw) - k * (ub + bgt)) <= 0
                              sbgt_ub, sc * (sum(sw) - k * ub) <= 0
                          end)
    end
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::BudgetRange,
                                            sbgt::BudgetRange)
    lw = model[:lw]
    sw = model[:sw]
    k = model[:k]
    sc = model[:sc]
    lb = bgt.lb
    slb = sbgt.lb
    lb_flag = isnothing(lb)
    slb_flag = isnothing(slb)
    if !lb_flag && !slb_flag
        JuMP.@constraints(model, begin
                              lbgt_lb, sc * (sum(lw) - k * (lb + slb)) >= 0
                              sbgt_lb, sc * (sum(sw) - k * slb) >= 0
                          end)
    elseif !lb_flag && slb_flag
        JuMP.@constraint(model, lbgt_lb, sc * (sum(lw) - k * lb) >= 0)
    elseif lb_flag && !slb_flag
        JuMP.@constraint(model, sbgt_lb, sc * (sum(sw) - k * slb) >= 0)
    end
    ub = bgt.ub
    sub = sbgt.ub
    ub_flag = isnothing(ub)
    sub_flag = isnothing(sub)
    if !ub_flag && !sub_flag
        JuMP.@constraints(model, begin
                              lbgt_ub, sc * (sum(lw) - k * (ub + sub)) <= 0
                              sbgt_ub, sc * (sum(sw) - k * sub) <= 0
                          end)
    elseif !ub_flag && sub_flag
        JuMP.@constraint(model, lbgt_ub, sc * (sum(lw) - k * ub) <= 0)
    elseif ub_flag && !sub_flag
        JuMP.@constraint(model, sbgt_ub, sc * (sum(sw) - k * sub) <= 0)
    end
    return nothing
end
"""
    set_cost_budget_constraints!(model, vp, vn, val_or_bgt, w)

Set cost-budget constraints in the JuMP model.

Various overloads handle different cost types (fixed value or [`BudgetRange`](@ref)).

# Arguments

  - `model`: JuMP optimisation model.
  - `vp`: Positive cost vector or scalar.
  - `vn`: Negative cost vector or scalar.
  - `val_or_bgt`: Fixed budget value or [`BudgetRange`](@ref).
  - `w`: Portfolio weight vector.

# Returns

  - `nothing`.

# Related

  - [`set_budget_costs!`](@ref)
  - [`set_long_short_budget_constraints!`](@ref)
"""
function set_cost_budget_constraints!(model::JuMP.Model, vp::Num_VecNum, vn::Num_VecNum,
                                      val::Number, w::VecNum)
    k = model[:k]
    sc = model[:sc]
    wp = model[:wp]
    wn = model[:wn]
    JuMP.@expression(model, cost_bgt_expr, dot_scalar(vp, wp) + dot_scalar(vn, wn))
    JuMP.@constraint(model, cost_bgt, sc * (sum(w) + cost_bgt_expr - k * val) == 0)
    return nothing
end
function set_cost_budget_constraints!(model::JuMP.Model, vp::Num_VecNum, vn::Num_VecNum,
                                      bgt::BudgetRange, w::VecNum)
    k = model[:k]
    sc = model[:sc]
    wp = model[:wp]
    wn = model[:wn]
    lb = bgt.lb
    ub = bgt.ub
    JuMP.@expression(model, cost_bgt_expr, dot_scalar(vp, wp) + dot_scalar(vn, wn))
    if !isnothing(lb)
        JuMP.@constraint(model, cost_bgt_lb, sc * (sum(w) + cost_bgt_expr - k * lb) >= 0)
    end
    if !isnothing(ub)
        JuMP.@constraint(model, cost_bgt_ub, sc * (sum(w) + cost_bgt_expr - k * ub) <= 0)
    end
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetCosts, w::VecNum)
    wb = bgt.w
    up = bgt.up
    un = bgt.un
    sc = model[:sc]
    N = length(w)
    JuMP.@variables(model, begin
                        wp[1:N], (lower_bound = 0)
                        wn[1:N], (lower_bound = 0)
                    end)
    JuMP.@constraints(model, begin
                          sc * (wp ⊖ up) <= 0
                          sc * (wn ⊖ un) <= 0
                          sc * (w - wb - wp + wn) == 0
                      end)
    set_cost_budget_constraints!(model, bgt.vp, bgt.vn, bgt.bgt, w)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetMarketImpact, w::VecNum)
    wb = bgt.w
    up = bgt.up
    un = bgt.un
    beta = bgt.beta
    sc = model[:sc]
    N = length(w)
    JuMP.@variables(model, begin
                        wp[1:N], (lower_bound = 0)
                        wn[1:N], (lower_bound = 0)
                        wip[1:N]
                        win[1:N]
                    end)
    JuMP.@constraints(model,
                      begin
                          sc * (wp ⊖ up) <= 0
                          sc * (wn ⊖ un) <= 0
                          sc * (w - wb - wp + wn) == 0
                          [i = 1:N],
                          [sc * wip[i], 1, sc * wp[i]] in JuMP.MOI.PowerCone(beta)
                          [i = 1:N],
                          [sc * win[i], 1, sc * wn[i]] in JuMP.MOI.PowerCone(beta)
                      end)
    set_cost_budget_constraints!(model, wip, win, bgt.bgt, w)
    return nothing
end

export BudgetRange, BudgetCosts, BudgetMarketImpact
