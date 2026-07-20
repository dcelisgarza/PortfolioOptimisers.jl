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
    """
    $(field_dict[:lb])
    """
    lb
    """
    $(field_dict[:ub])
    """
    ub
    function BudgetRange(lb::Option{<:Number}, ub::Option{<:Number})
        lb_flag = isnothing(lb)
        ub_flag = isnothing(ub)
        @argcheck(!(lb_flag && ub_flag),
                  ArgumentError("at least one of lb or ub must not be nothing, got lb = $lb, ub = $ub"))
        if !lb_flag
            @argcheck(isfinite(lb), IsNonFiniteError("lb must be finite, got $lb"))
        end
        if !ub_flag
            @argcheck(isfinite(ub), IsNonFiniteError("ub must be finite, got $ub"))
        end
        if !lb_flag && !ub_flag
            @argcheck(lb <= ub, DomainError("lb must be <= ub, got lb = $lb, ub = $ub"))
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

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `w`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `vp`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `vn`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `up`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `un`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Related

  - [`BudgetRange`](@ref)
  - [`BudgetMarketImpact`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct BudgetCosts <: BudgetCostEstimator
    """
    Budget target or range.
    """
    bgt
    """
    $(field_dict[:pw])
    """
    @vprop w
    """
    Cost coefficients for positive weight changes. Non-negative.
    """
    @vprop vp
    """
    Cost coefficients for negative weight changes. Non-negative.
    """
    @vprop vn
    """
    Upper limit on positive weight changes. Non-negative.
    """
    @vprop up
    """
    Upper limit on negative weight changes. Non-negative.
    """
    @vprop un
    function BudgetCosts(bgt::Num_BgtRg, w::VecNum, vp::Num_VecNum, vn::Num_VecNum,
                         up::Num_VecNum, un::Num_VecNum)
        @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
        if isa(vp, VecNum)
            @argcheck(!isempty(vp), IsEmptyError("vp cannot be empty"))
            @argcheck(all(x -> zero(x) <= x, vp),
                      DomainError(vp, "all elements of vp must be >= 0"))
        else
            @argcheck(vp >= zero(vp), DomainError(vp, "vp must be >= 0"))
        end
        if isa(vn, VecNum)
            @argcheck(!isempty(vn), IsEmptyError("vn cannot be empty"))
            @argcheck(all(x -> zero(x) <= x, vn),
                      DomainError(vn, "all elements of vn must be >= 0"))
        else
            @argcheck(vn >= zero(vn), DomainError(vn, "vn must be >= 0"))
        end
        if isa(up, VecNum)
            @argcheck(!isempty(up), IsEmptyError("up cannot be empty"))
            @argcheck(all(x -> zero(x) <= x, up),
                      DomainError(up, "all elements of up must be >= 0"))
        else
            @argcheck(up >= zero(up), DomainError(up, "up must be >= 0"))
        end
        if isa(un, VecNum)
            @argcheck(!isempty(un), IsEmptyError("un cannot be empty"))
            @argcheck(all(x -> zero(x) <= x, un),
                      DomainError(un, "all elements of un must be >= 0"))
        else
            @argcheck(un >= zero(un), DomainError(un, "un must be >= 0"))
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

## View parameters

When [`port_opt_view`](@ref) is called on this type, the following `@vprop`-tagged fields are automatically subset to the selected indices:

  - `w`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `vp`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `vn`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `up`: Sliced to the selected indices via [`port_opt_view`](@ref).
  - `un`: Sliced to the selected indices via [`port_opt_view`](@ref).

# Related

  - [`BudgetRange`](@ref)
  - [`BudgetCosts`](@ref)
  - [`port_opt_view`](@ref)
"""
@propagatable @concrete struct BudgetMarketImpact <: BudgetCostEstimator
    """
    Budget target or range.
    """
    bgt
    """
    $(field_dict[:pw])
    """
    @vprop w
    """
    Cost coefficients for positive weight changes. Non-negative.
    """
    @vprop vp
    """
    Cost coefficients for negative weight changes. Non-negative.
    """
    @vprop vn
    """
    Upper limit on positive weight changes. Non-negative.
    """
    @vprop up
    """
    Upper limit on negative weight changes. Non-negative.
    """
    @vprop un
    """
    Market impact exponent in `(0, 1]`.
    """
    beta
    function BudgetMarketImpact(bgt::Num_BgtRg, w::VecNum, vp::Num_VecNum, vn::Num_VecNum,
                                up::Num_VecNum, un::Num_VecNum, beta::Number)
        @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
        if isa(vp, VecNum)
            @argcheck(!isempty(vp), IsEmptyError("vp cannot be empty"))
            @argcheck(all(x -> zero(x) <= x, vp),
                      DomainError(vp, "all elements of vp must be >= 0"))
        else
            @argcheck(vp >= zero(vp), DomainError(vp, "vp must be >= 0"))
        end
        if isa(vn, VecNum)
            @argcheck(!isempty(vn), IsEmptyError("vn cannot be empty"))
            @argcheck(all(x -> zero(x) <= x, vn),
                      DomainError(vn, "all elements of vn must be >= 0"))
        else
            @argcheck(vn >= zero(vn), DomainError(vn, "vn must be >= 0"))
        end
        if isa(up, VecNum)
            @argcheck(!isempty(up), IsEmptyError("up cannot be empty"))
            @argcheck(all(x -> zero(x) <= x, up),
                      DomainError(up, "all elements of up must be >= 0"))
        else
            @argcheck(up >= zero(up), DomainError(up, "up must be >= 0"))
        end
        if isa(un, VecNum)
            @argcheck(!isempty(un), IsEmptyError("un cannot be empty"))
            @argcheck(all(x -> zero(x) <= x, un),
                      DomainError(un, "all elements of un must be >= 0"))
        else
            @argcheck(un >= zero(un), DomainError(un, "un must be >= 0"))
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
function set_budget_constraints!(model::JuMP.Model, val::Number, w::VecNum)
    k = get_k(model)
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, bgt, sc * (sum(w) - k * val) == 0)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, w::VecNum)
    k = get_k(model)
    sc = get_constraint_scale(model)
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
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that a gross budget is admissible alongside the net and short budgets.

`bgt` and `sbgt` constrain the net and gross exposures only *together*: pinning both gives `1'w == bgt` and `norm(w, 1) == bgt + 2 * sbgt`. `gbgt` exists for the combination they cannot reach — gross pinned with net free — so it is rejected when they already determine the gross exposure, and when the weight bounds forbid short positions (whereupon gross and net coincide and `bgt` already owns the constraint).

Only the statically-decidable contradictions are caught. Ranges may still conflict at solve time, surfacing as infeasibility, and a [`WeightBoundsEstimator`](@ref) is not resolved until then. [`TimeDependent`](@ref) schedules are reached through [`assert_time_dependent_substitution`](@ref).

# Related

  - [`JuMPOptimiser`](@ref)
  - [`set_gross_budget_constraints!`](@ref)
  - [`set_long_short_budget_constraints!`](@ref)
"""
function assert_gross_budget_admissible(bgt, sbgt, gbgt, wb)::Nothing
    if isnothing(gbgt) || isa(gbgt, TimeDependent)
        return nothing
    end
    @argcheck(!(isa(bgt, Number) && isa(sbgt, Number)),
              ArgumentError("gbgt is over-determined: bgt ($bgt) and sbgt ($sbgt) already pin the gross exposure at bgt + 2 * sbgt = $(bgt + 2 * sbgt). Give at most two of bgt, sbgt and gbgt — gbgt exists for the case bgt and sbgt cannot express, a pinned gross exposure with a free net exposure (bgt = nothing)."))
    if isa(wb, WeightBounds)
        @argcheck(w_neg_flag(wb.lb) || w_neg_flag(wb.ub),
                  ArgumentError("gbgt requires weight bounds that admit short positions: with non-negative bounds no short weights exist, so the gross exposure equals the net exposure and bgt already constrains it. Got wb = $wb."))
    end
    return nothing
end
"""
    set_gross_budget_constraints!(model::JuMP.Model, gbgt::Option{<:Num_BgtRg})

Constrain the gross exposure (leverage) `sum(lw) + sum(sw)`, independently of the net exposure.

A number pins it; a [`BudgetRange`](@ref) bounds it on either side. Whether the constraint pins the *realised* gross exposure `norm(w, 1)` or merely bounds it depends on `xbgt` — see [`short_mip_threshold_constraints`](@ref).

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `gbgt`: Gross budget specification, or `nothing` for no constraint.

# Returns

  - `nothing`.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`assert_gross_budget_admissible`](@ref)
  - [`set_long_short_budget_constraints!`](@ref)
"""
function set_gross_budget_constraints!(args...)
    return nothing
end
function set_gross_budget_constraints!(model::JuMP.Model, gbgt::Number)
    lw = shared_get(model, :lw)
    sw = shared_get(model, :sw)
    k = get_k(model)
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, gbgt, sc * (sum(lw) + sum(sw) - k * gbgt) == 0)
    return nothing
end
function set_gross_budget_constraints!(model::JuMP.Model, gbgt::BudgetRange)
    lw = shared_get(model, :lw)
    sw = shared_get(model, :sw)
    k = get_k(model)
    sc = get_constraint_scale(model)
    lb = gbgt.lb
    if !isnothing(lb)
        JuMP.@constraint(model, gbgt_lb, sc * (sum(lw) + sum(sw) - k * lb) >= 0)
    end
    ub = gbgt.ub
    if !isnothing(ub)
        JuMP.@constraint(model, gbgt_ub, sc * (sum(lw) + sum(sw) - k * ub) <= 0)
    end
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Number, ::Nothing)
    lw = shared_get(model, :lw)
    k = get_k(model)
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, lbgt, sc * (sum(lw) - k * bgt) == 0)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, ::Nothing, sbgt::Number)
    sw = shared_get(model, :sw)
    k = get_k(model)
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, sbgt, sc * (sum(sw) - k * sbgt) == 0)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Number, sbgt::Number)
    lw = shared_get(model, :lw)
    sw = shared_get(model, :sw)
    k = get_k(model)
    sc = get_constraint_scale(model)
    JuMP.@constraints(model, begin
                          lbgt, sc * (sum(lw) - k * (bgt + sbgt)) == 0
                          sbgt, sc * (sum(sw) - k * sbgt) == 0
                      end)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, ::Nothing)
    lw = shared_get(model, :lw)
    k = get_k(model)
    sc = get_constraint_scale(model)
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
    sw = shared_get(model, :sw)
    k = get_k(model)
    sc = get_constraint_scale(model)
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
    lw = shared_get(model, :lw)
    sw = shared_get(model, :sw)
    k = get_k(model)
    sc = get_constraint_scale(model)
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
    lw = shared_get(model, :lw)
    sw = shared_get(model, :sw)
    k = get_k(model)
    sc = get_constraint_scale(model)
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
    lw = shared_get(model, :lw)
    sw = shared_get(model, :sw)
    k = get_k(model)
    sc = get_constraint_scale(model)
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
    k = get_k(model)
    sc = get_constraint_scale(model)
    wp = shared_get(model, :wp)
    wn = shared_get(model, :wn)
    JuMP.@expression(model, cost_bgt_expr, dot_scalar(vp, wp) + dot_scalar(vn, wn))
    JuMP.@constraint(model, cost_bgt, sc * (sum(w) + cost_bgt_expr - k * val) == 0)
    return nothing
end
function set_cost_budget_constraints!(model::JuMP.Model, vp::Num_VecNum, vn::Num_VecNum,
                                      bgt::BudgetRange, w::VecNum)
    k = get_k(model)
    sc = get_constraint_scale(model)
    wp = shared_get(model, :wp)
    wn = shared_get(model, :wn)
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
    sc = get_constraint_scale(model)
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
    sc = get_constraint_scale(model)
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

"""
$(DocStringExtensions.TYPEDSIGNATURES)

Pin the long/short decomposition, so the budgets built on `lw`/`sw` hold *exactly*.

`lw` and `sw` carry the long and short parts of the weights, and `bgt`, `sbgt` and `gbgt` are
all stated in terms of them. Whether those budgets *pin* the realised exposure or merely
*bound* it depends on how the head related the parts to the weights, so this dispatches on the
head's [`AbstractDecompositionContract`](@ref). Both methods take the sign from `ind`, whichever
bundle carries one — a dedicated sign bit, or the long/short bits of a builder that already
needed them for thresholds or cardinality.

# Mathematical definition

Both contracts emit the *sign pair*, which forces `sw = 0` on the long side and `lw = 0` on the
short side:

```math
\\begin{align}
lw_i - M\\, g^l_i &\\leq 0\\,, & sw_i - M\\, g^s_i &\\leq 0\\,.
\\end{align}
```

Under [`PartsBoundWeights`](@ref) that leaves the *held* side free to overstate itself (`lw_i`
may exceed `w_i`), so two more constraints close the slack:

```math
\\begin{align}
lw_i - w_i - M(1 - b^l_i) &\\leq 0\\,, & sw_i + w_i - M(1 - b^s_i) &\\leq 0\\,.
\\end{align}
```

Where:

  - ``g^l_i``, ``g^s_i``: Long and short *gates* ([`long_gate`](@ref), [`short_gate`](@ref)), which relax to continuous variables when the budget is free.
  - ``b^l_i``, ``b^s_i``: Long and short *binaries* ([`long_bin`](@ref), [`short_bin`](@ref)). The slack-closing pair keys on these rather than the gates, because it must key on a bit that is exactly 0 or 1 even when the gates relax.
  - ``w_i``: Weight expression for position ``i`` in the space of `sp`.
  - ``M``: Big-M constant.

Together they give `lw_i == max(w_i, 0)` and `sw_i == max(-w_i, 0)`: `b^l_i = 1` turns the
first into `lw_i <= w_i`, which against the head's `lw_i >= w_i` is an equality; both bits zero
gives `lw_i == sw_i == 0`, hence `w_i == 0`.

Under [`WeightsFromParts`](@ref) the pair alone is enough and the slack-closing constraints are
not emitted: `w = lw - sw` is an identity, so `sw_i = 0` already leaves `lw_i == w_i`.

# Arguments

  - $(arg_dict[:model])
  - `sp::AbstractMIPSpace`: Weight space the constraints act on.
  - `ind::AbstractMIPIndicators`: Indicator bundle supplying the sign.
  - `wx::VecNum`: Weight expression in the space of `sp`.
  - `ss`: Big-M expression, as registered by [`set_mip_ss_expr!`](@ref).
  - `dc::AbstractDecompositionContract`: The head's contract. There is deliberately no method for `nothing` — a model with no head contract has no short side, so there is no decomposition to pin and a caller that got here is confused.

# Returns

  - `nothing`.

# Related

  - [`AbstractDecompositionContract`](@ref)
  - [`set_long_short_budget_constraints!`](@ref)
  - [`set_gross_budget_constraints!`](@ref)
  - [`AbstractMIPIndicators`](@ref)
"""
function set_exact_budget_constraints!(model::JuMP.Model, sp::AbstractMIPSpace,
                                       ind::AbstractMIPIndicators, wx::VecNum, ss,
                                       ::WeightsFromParts)
    lw = shared_get(model, :lw)
    sw = shared_get(model, :sw)
    sc = get_constraint_scale(model)
    model[mip_key(sp, :xbgt_lw_ub)] = JuMP.@constraint(model,
                                                       sc * (lw - ss * long_gate(ind)) <= 0)
    model[mip_key(sp, :xbgt_sw_ub)] = JuMP.@constraint(model,
                                                       sc * (sw - ss * short_gate(ind)) <=
                                                       0)
    return nothing
end
function set_exact_budget_constraints!(model::JuMP.Model, sp::AbstractMIPSpace,
                                       ind::AbstractMIPIndicators, wx::VecNum, ss,
                                       ::PartsBoundWeights)
    lw = shared_get(model, :lw)
    sw = shared_get(model, :sw)
    sc = get_constraint_scale(model)
    model[mip_key(sp, :xbgt_lw_ub)] = JuMP.@constraint(model,
                                                       sc * (lw - ss * long_gate(ind)) <= 0)
    model[mip_key(sp, :xbgt_sw_ub)] = JuMP.@constraint(model,
                                                       sc * (sw - ss * short_gate(ind)) <=
                                                       0)
    model[mip_key(sp, :xbgt_lw_eq)] = JuMP.@constraint(model,
                                                       sc *
                                                       (lw - wx - ss * (1 .- long_bin(ind))) <=
                                                       0)
    model[mip_key(sp, :xbgt_sw_eq)] = JuMP.@constraint(model,
                                                       sc * (sw + wx -
                                                             ss * (1 .- short_bin(ind))) <=
                                                       0)
    return nothing
end
function set_exact_budget_constraints!(model::JuMP.Model, sp::AbstractMIPSpace,
                                       ind::AbstractMIPIndicators, wx::VecNum, ss)
    return set_exact_budget_constraints!(model, sp, ind, wx, ss,
                                         decomposition_contract(model))
end

export BudgetRange, BudgetCosts, BudgetMarketImpact
