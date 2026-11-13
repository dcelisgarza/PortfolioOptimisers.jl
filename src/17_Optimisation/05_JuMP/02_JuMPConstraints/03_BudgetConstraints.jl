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
    const Num_BgtCE = Union{Number, BudgetConstraintEstimator}

Union of scalar budget values and [`BudgetConstraintEstimator`](@ref) instances.
"""
const Num_BgtCE = Union{Number, BudgetConstraintEstimator}
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

Each member holds a reference portfolio `w` and a `fixed` flag. [`factory`](@ref factory(bgt::BudgetCosts, w::VecNum)) and [`needs_previous_weights`](@ref needs_previous_weights(bgt::BudgetCostEstimator)) read both.

# Related

  - [`BudgetCosts`](@ref)
  - [`BudgetMarketImpact`](@ref)
"""
abstract type BudgetCostEstimator <: BudgetConstraintEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Bounds the sum of the portfolio weights inside a closed interval, rather than pinning it to one value.

At least one of `lb` or `ub` must be given; the absent side is left unbounded. The same type
serves the net budget `bgt`, the short budget `sbgt` and the gross budget `gbgt`.

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
            @argcheck(lb <= ub,
                      DomainError((lb, ub),
                                  "`BudgetRange.lb` is $lb and `ub` is $ub. A lower budget is at most its upper budget, so `lb <= ub` must hold."))
        end
        return new{typeof(lb), typeof(ub)}(lb, ub)
    end
end
function BudgetRange(; lb::Option{<:Number} = 1.0, ub::Option{<:Number} = 1.0)
    return BudgetRange(lb, ub)
end
"""
    const Num_BgtRg = Union{Number, BudgetEstimator}

Alias for a scalar budget value or budget range.

Matches either a plain number (fixed budget, e.g. `1.0`) or a [`BudgetEstimator`](@ref), such as a [`BudgetRange`](@ref) (interval budget constraint). Used for dispatch in budget constraint generation.

# Related

  - [`BudgetEstimator`](@ref)
  - [`BudgetRange`](@ref)
  - [`set_long_short_budget_constraints!`](@ref)
"""
const Num_BgtRg = Union{Number, BudgetEstimator}
"""
    set_budget_constraints!(args...; kwargs...)
    set_budget_constraints!(model::JuMP.Model, val::Number, w::VecNum; prefix::Symbol = Symbol(""))
    set_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, w::VecNum; prefix::Symbol = Symbol(""))
    set_budget_constraints!(model::JuMP.Model, bgt::BudgetCosts, w::VecNum; prefix::Symbol = Symbol(""))
    set_budget_constraints!(model::JuMP.Model, bgt::BudgetMarketImpact, w::VecNum; prefix::Symbol = Symbol(""))

Add budget constraints to the JuMP optimisation model.

The fall-through method does nothing. The concrete methods add the appropriate portfolio budget constraint based on the type of budget specification provided.

A number and a [`BudgetRange`](@ref) register their rows under `prefix`, the namespace of the block whose owner states the budget. A budget that pays costs registers its trade variables, its cost expression and its rows in the shared Model State, because they measure the trade of the one weight vector. The rows are `wp_ub` and `wn_ub`, the bounds of the trades, `cost_bgt_trade`, the trade of the weights from the reference, and for [`BudgetMarketImpact`](@ref) the power cones `wip_pcone` and `win_pcone`. So it goes on the first block of a model, the block of its owner, and refuses a non-empty `prefix`.

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
\\sum_i w_i + \\boldsymbol{v}_p^\\intercal \\boldsymbol{w}_p^{1/\\beta} + \\boldsymbol{v}_n^\\intercal \\boldsymbol{w}_n^{1/\\beta} \\in [k\\,\\mathrm{lb},\\; k\\,\\mathrm{ub}]\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - ``\\boldsymbol{w}_p``, ``\\boldsymbol{w}_n``: Positive and negative weight increments.
  - ``\\boldsymbol{v}_p``, ``\\boldsymbol{v}_n``: Market-impact cost coefficient vectors.
  - ``\\beta \\in (0, 1)``: Reciprocal of the market impact exponent, so the realised power law is ``1/\\beta``.
  - ``\\mathrm{lb}``, ``\\mathrm{ub}``: Lower and upper budget bounds.

The power is raised through a three-dimensional power cone, one per asset per sign, so the
model stays convex. [`BudgetMarketImpact`](@ref) states the full system.

# Arguments

  - $(arg_dict[:model])
  - `val::Number`: Fixed budget scalar.
  - `bgt`: Budget constraint specification ([`BudgetRange`](@ref), [`BudgetCosts`](@ref), or [`BudgetMarketImpact`](@ref)).
  - `w`: Portfolio weight vector.
  - $(arg_dict[:prefix_kw])

# Validation

  - A [`BudgetCosts`](@ref) or a [`BudgetMarketImpact`](@ref) under a non-empty `prefix`. An `ArgumentError` is thrown.

# Returns

  - `nothing`.

# Related

  - [`BudgetRange`](@ref)
  - [`BudgetCosts`](@ref)
  - [`BudgetMarketImpact`](@ref)
  - [`set_weight_constraints!`](@ref)
"""
function set_budget_constraints!(args...; kwargs...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDEF)

Charges the portfolio budget for transaction costs that grow linearly with the traded volume.

Models the cost of the bid-ask spread, of slippage, of a borrowing charge, or of a management
fee. The cost is charged **against the budget**: the weights and their costs must together
meet `bgt`, so a portfolio that trades more may hold less.

The rows are Equations 1 to 3 of [pogue1969](@cite) with one linear segment on each side of the trade. Exhibit 3 of that work charges the cost against the cash of the revised portfolio, as the first row charges it against the budget.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{1}^\\intercal \\boldsymbol{w} + \\boldsymbol{v}_p^\\intercal \\boldsymbol{w}_p + \\boldsymbol{v}_n^\\intercal \\boldsymbol{w}_n &\\in [k\\,\\mathrm{lb},\\; k\\,\\mathrm{ub}]\\,, \\\\
\\boldsymbol{w} - k\\,\\boldsymbol{w}_0 &= \\boldsymbol{w}_p - \\boldsymbol{w}_n\\,, \\\\
\\boldsymbol{w}_p &\\leq k\\,\\boldsymbol{u}_p\\,, \\\\
\\boldsymbol{w}_n &\\leq k\\,\\boldsymbol{u}_n\\,, \\\\
\\boldsymbol{w}_p,\\; \\boldsymbol{w}_n &\\geq \\boldsymbol{0}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - ``\\boldsymbol{w}_0``: Reference portfolio the trade starts from (`w`). [`factory`](@ref factory(bgt::BudgetCosts, w::VecNum)) replaces it with the weights of the previous fold or step, unless `fixed` is `true`.
  - ``\\boldsymbol{w}_p``, ``\\boldsymbol{w}_n``: Positive and negative weight increments.
  - ``\\boldsymbol{v}_p``, ``\\boldsymbol{v}_n``: Cost coefficient vectors for positive and negative changes.
  - ``\\boldsymbol{u}_p``, ``\\boldsymbol{u}_n``: Upper limits on the positive and negative increments.
  - ``\\mathrm{lb}``, ``\\mathrm{ub}``: Lower and upper budget bounds. A scalar `bgt` pins them together.

Every row is homogeneous in ``(\\boldsymbol{w}, k)``. Under [`MaximumRatio`](@ref), the start portfolio ``\\boldsymbol{w}_0``, the caps and the budget therefore scale with ``k``, and the weights do not depend on `ohf`. Every other objective fixes ``k = 1``.

A pinned `bgt = 1.0` and a [`MaximumReturn`](@ref) objective reproduce the identity to
**7.1e-9** on a five-asset sample: `sum(w) + 0.01 * sum(abs.(w - w0))` lands on
**0.999999992894044**.

!!! warning

    ``\\boldsymbol{w}_p`` and ``\\boldsymbol{w}_n`` are not forced to be complementary, so the
    relaxation is exact only while the objective prefers a **larger** invested budget. Under
    [`MinimumRisk`](@ref) it does not: a larger cost shrinks
    ``\\boldsymbol{1}^\\intercal \\boldsymbol{w}`` and so shrinks the risk, and the solver inflates
    both increments to their `up` and `un` caps. On the sample above, minimising the variance
    gives `sum(w) = 0.9514` with a charged cost of **0.0486** against a traded cost of
    **0.0014**. Give the return term a lower bound through `settings.lb`, as Listing 9.8 of
    [cajas2025](@cite) does, whenever the objective minimises a risk measure.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    BudgetCosts(;
        bgt::Num_BgtRg = 1.0,
        w::VecNum,
        vp::Num_VecNum = 1.0,
        vn::Num_VecNum = 1.0,
        up::Num_VecNum = 1.0,
        un::Num_VecNum = 1.0,
        fixed::Bool = false
    ) -> BudgetCosts

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(w)`.
  - If `vp` is a vector: `!isempty(vp)` and all elements `>= 0`. If scalar: `>= 0`.
  - If `vn` is a vector: `!isempty(vn)` and all elements `>= 0`. If scalar: `>= 0`.
  - If `up` is a vector: `!isempty(up)` and all elements `>= 0`. If scalar: `>= 0`.
  - If `un` is a vector: `!isempty(un)` and all elements `>= 0`. If scalar: `>= 0`.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `w`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `vp`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `vn`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `up`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `un`: Sliced to the selected indices through [`port_opt_view`](@ref).

# Related

  - [`BudgetRange`](@ref)
  - [`BudgetMarketImpact`](@ref)
  - [`port_opt_view`](@ref)
  - [`set_budget_constraints!`](@ref)
  - [`set_cost_budget_constraints!`](@ref)

# References

  - $(ref_dict[:pogue1969]) Pages 11 and 12, Equations 1 to 3, and page 29, Exhibit 3.
  - $(ref_dict[:cajas2025]) Section 9.1, Equations 9.14 and 9.15, and Listing 9.8.
"""
@propagatable @concrete struct BudgetCosts <: BudgetCostEstimator
    """
    $(field_dict[:bgt_cost_target])
    """
    bgt
    """
    $(field_dict[:pw])
    """
    @vprop w
    """
    $(field_dict[:vp_cost])
    """
    @vprop vp
    """
    $(field_dict[:vn_cost])
    """
    @vprop vn
    """
    $(field_dict[:up_cost])
    """
    @vprop up
    """
    $(field_dict[:un_cost])
    """
    @vprop un
    """
    $(field_dict[:fixed])
    """
    fixed
    function BudgetCosts(bgt::Num_BgtRg, w::VecNum, vp::Num_VecNum, vn::Num_VecNum,
                         up::Num_VecNum, un::Num_VecNum, fixed::Bool)
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
        return new{typeof(bgt), typeof(w), typeof(vp), typeof(vn), typeof(up), typeof(un),
                   typeof(fixed)}(bgt, w, vp, vn, up, un, fixed)
    end
end
function BudgetCosts(; bgt::Num_BgtRg = 1.0, w::VecNum, vp::Num_VecNum = 1.0,
                     vn::Num_VecNum = 1.0, up::Num_VecNum = 1.0, un::Num_VecNum = 1.0,
                     fixed::Bool = false)
    return BudgetCosts(bgt, w, vp, vn, up, un, fixed)
end
"""
$(DocStringExtensions.TYPEDEF)

Charges the portfolio budget and the return for market impact costs that follow an empirical power law.

Extends [`BudgetCosts`](@ref) with a power cone, so the charge grows as the traded volume
raised to ``1/\\mathrm{beta}``. Unlike a linear cost, the charge is **also subtracted from the
return expression** of every term whose `settings.mic` is `true`.

Section 6.3 of [mosek2023c](@cite) states this cost, charges it against the budget, and writes it as a power cone. That work names page 452 of [grinoldkahn1999](@cite) as the source of the power law. No copy of that work was found. [cajas2025](@cite) splits the trade into its two signs, as the rows below do.

# Mathematical definition

```math
\\begin{align}
\\boldsymbol{1}^\\intercal \\boldsymbol{w} + \\boldsymbol{v}_p^\\intercal \\boldsymbol{\\iota}_p + \\boldsymbol{v}_n^\\intercal \\boldsymbol{\\iota}_n &\\in [k\\,\\mathrm{lb},\\; k\\,\\mathrm{ub}]\\,, \\\\
(\\iota_{p,i},\\; k,\\; w_{p,i}) &\\in \\mathcal{K}_{\\mathrm{pow}}^\\beta \\quad \\forall i = 1,\\dots,N\\,, \\\\
(\\iota_{n,i},\\; k,\\; w_{n,i}) &\\in \\mathcal{K}_{\\mathrm{pow}}^\\beta \\quad \\forall i = 1,\\dots,N\\,, \\\\
\\boldsymbol{w} - k\\,\\boldsymbol{w}_0 &= \\boldsymbol{w}_p - \\boldsymbol{w}_n\\,, \\\\
\\boldsymbol{w}_p &\\leq k\\,\\boldsymbol{u}_p\\,, \\\\
\\boldsymbol{w}_n &\\leq k\\,\\boldsymbol{u}_n\\,, \\\\
\\boldsymbol{w}_p,\\; \\boldsymbol{w}_n &\\geq \\boldsymbol{0}\\,.
\\end{align}
```

Where:

  - $(math_dict[:w_port])
  - $(math_dict[:k_budget])
  - $(math_dict[:N])
  - ``\\boldsymbol{w}_0``: Reference portfolio the trade starts from (`w`). [`factory`](@ref factory(bgt::BudgetCosts, w::VecNum)) replaces it with the weights of the previous fold or step, unless `fixed` is `true`.
  - ``\\boldsymbol{w}_p``, ``\\boldsymbol{w}_n``: Positive and negative weight increments.
  - ``\\boldsymbol{\\iota}_p``, ``\\boldsymbol{\\iota}_n``: Auxiliary variables the power cone bounds below by ``w_p^{1/\\beta}`` and ``w_n^{1/\\beta}``.
  - ``\\boldsymbol{v}_p``, ``\\boldsymbol{v}_n``: Market impact cost coefficients for positive and negative trades.
  - ``\\boldsymbol{u}_p``, ``\\boldsymbol{u}_n``: Upper limits on the positive and negative increments.
  - ``\\mathcal{K}_{\\mathrm{pow}}^\\beta``: Three-dimensional power cone of exponent ``\\beta``.
  - ``\\mathrm{lb}``, ``\\mathrm{ub}``: Lower and upper budget bounds. A scalar `bgt` pins them together.

Every row is homogeneous in ``(\\boldsymbol{w}, k)``. Under [`MaximumRatio`](@ref), the start portfolio ``\\boldsymbol{w}_0``, the caps and the budget therefore scale with ``k``, and the weights do not depend on `ohf`. Every other objective fixes ``k = 1``.

!!! note

    **`beta` is the reciprocal of the power law's exponent.** The cone gives
    ``\\iota_i \\geq w_i^{1/\\beta}``, so the default `beta = 2/3` charges the volume raised to
    **3/2**. Equation 1 of [toth2011](@cite) is the square-root law of the price impact, and
    the cost is the impact times the traded volume, so it grows as the volume raised to 3/2.
    The cone needs `0 < beta < 1`; the
    constructor refuses both endpoints, because a solver reports `SLOW_PROGRESS` on either.

A pinned `bgt = 1.0` and a [`MaximumReturn`](@ref) objective reproduce the identity to
**9.4e-10** on a five-asset sample: `sum(w) + 0.01 * sum(abs.(w - w0) .^ 1.5)` lands on
**0.9999999990596528**.

!!! warning

    The auxiliary variables carry no upper bound, so the charge is exact only while the
    objective prefers a **larger** invested budget. Under [`MinimumRisk`](@ref) the solver
    treats them as riskless assets and concentrates the whole budget in them: on the sample
    above, minimising the variance gives `sum(w) = 0.0051`. Give the return term a lower bound
    through `settings.lb`, as Section 9.3 of [cajas2025](@cite) instructs, whenever the
    objective minimises a risk measure.

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
        beta::Number = 2/3,
        fixed::Bool = false
    ) -> BudgetMarketImpact

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(w)`.
  - If `vp` is a vector: `!isempty(vp)` and all elements `>= 0`. If scalar: `>= 0`.
  - If `vn` is a vector: `!isempty(vn)` and all elements `>= 0`. If scalar: `>= 0`.
  - If `up` is a vector: `!isempty(up)` and all elements `>= 0`. If scalar: `>= 0`.
  - If `un` is a vector: `!isempty(un)` and all elements `>= 0`. If scalar: `>= 0`.
  - `0 < beta < 1`.

## View parameters

[`port_opt_view`](@ref) takes the view of these `@vprop` fields at the selected indices:

  - `w`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `vp`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `vn`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `up`: Sliced to the selected indices through [`port_opt_view`](@ref).
  - `un`: Sliced to the selected indices through [`port_opt_view`](@ref).

# Related

  - [`BudgetRange`](@ref)
  - [`BudgetCosts`](@ref)
  - [`port_opt_view`](@ref)
  - [`set_budget_constraints!`](@ref)
  - [`add_market_impact_cost!`](@ref)

# References

  - $(ref_dict[:mosek2023c]) Section 6.3, Equations 6.4 to 6.6. $(ref_dict[:no_original_source])
  - $(ref_dict[:grinoldkahn1999])
  - $(ref_dict[:toth2011]) Section 1, Equation 1.
  - $(ref_dict[:cajas2025]) Section 9.3, Equations 9.22 and 9.23.
"""
@propagatable @concrete struct BudgetMarketImpact <: BudgetCostEstimator
    """
    $(field_dict[:bgt_cost_target])
    """
    bgt
    """
    $(field_dict[:pw])
    """
    @vprop w
    """
    $(field_dict[:vp_cost])
    """
    @vprop vp
    """
    $(field_dict[:vn_cost])
    """
    @vprop vn
    """
    $(field_dict[:up_cost])
    """
    @vprop up
    """
    $(field_dict[:un_cost])
    """
    @vprop un
    """
    $(field_dict[:beta_mic])
    """
    beta
    """
    $(field_dict[:fixed])
    """
    fixed
    function BudgetMarketImpact(bgt::Num_BgtRg, w::VecNum, vp::Num_VecNum, vn::Num_VecNum,
                                up::Num_VecNum, un::Num_VecNum, beta::Number, fixed::Bool)
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
        @argcheck(zero(beta) < beta < one(beta),
                  DomainError(beta,
                              "beta must satisfy 0 < beta < 1; the power cone is degenerate at both endpoints, and a solver reports SLOW_PROGRESS. The realised power law exponent is 1/beta, so beta = 2/3 charges the volume raised to 3/2"))
        return new{typeof(bgt), typeof(w), typeof(vp), typeof(vn), typeof(up), typeof(un),
                   typeof(beta), typeof(fixed)}(bgt, w, vp, vn, up, un, beta, fixed)
    end
end
function BudgetMarketImpact(; bgt::Num_BgtRg = 1.0, w::VecNum, vp::Num_VecNum = 1.0,
                            vn::Num_VecNum = 1.0, up::Num_VecNum = 1.0,
                            un::Num_VecNum = 1.0, beta::Number = 2 / 3, fixed::Bool = false)
    return BudgetMarketImpact(bgt, w, vp, vn, up, un, beta, fixed)
end
"""
    factory(bgt::BudgetCosts, w::VecNum)
    factory(bgt::BudgetMarketImpact, w::VecNum)
    factory(bgt::Number, w::VecNum)

Replace the reference portfolio of a cost budget with the weights `w`, unless its `fixed` field is `true`.

A [`BudgetCosts`](@ref) and a [`BudgetMarketImpact`](@ref) charge the cost of the move from their reference `w` against the budget. A fold of a walk-forward, or a step of an online rule, trades from the weights that it holds, and [`factory(opt::JuMPOptimiser, w::AbstractVector)`](@ref) passes these weights to the `bgt` of the optimiser. So the charged cost is the cost of the trade of the fold, as for a [`Turnover`](@ref) on `tn` or on `fees`. A `fixed` estimator keeps the reference that it was built with.

A number holds no reference, so the method on a number returns it unchanged. A [`BudgetRange`](@ref) and `nothing` take the fallback method of [`factory`](@ref), which also returns them unchanged. A [`TimeDependent`](@ref) schedule applies `factory` to each entry.

# Algorithm

 1. If `bgt` is a number, or if `bgt.fixed` is `true`, return `bgt` unchanged.
 2. Else rebuild `bgt` through its keyword constructor with `w` as the reference portfolio. Every other field keeps its value.

# Arguments

  - `bgt`: The net budget of a [`JuMPOptimiser`](@ref).
  - `w`: The weights that the fold or the step trades from.

# Validation

  - Step 2 calls the keyword constructor, so it checks `w` as the constructor does.

# Returns

  - `bgt`: `bgt` unchanged in step 1. Else a new estimator of the same type with `w` as its reference.

# Examples

```jldoctest
julia> bgt = BudgetCosts(; w = [0.2, 0.3, 0.5], vp = 0.01, vn = 0.01);

julia> factory(bgt, [0.1, 0.4, 0.5]).w
3-element Vector{Float64}:
 0.1
 0.4
 0.5

julia> bgt = BudgetCosts(; w = [0.2, 0.3, 0.5], vp = 0.01, vn = 0.01, fixed = true);

julia> factory(bgt, [0.1, 0.4, 0.5]).w
3-element Vector{Float64}:
 0.2
 0.3
 0.5
```

# Related

  - [`BudgetCosts`](@ref)
  - [`BudgetMarketImpact`](@ref)
  - [`needs_previous_weights(bgt::BudgetCostEstimator)`](@ref)
  - [`factory(tn::TurnoverEstimator, w::VecNum)`](@ref): Reads the `fixed` flag in the same way.
"""
function factory(bgt::BudgetCosts, w::VecNum)::BudgetCosts
    return bgt.fixed ? bgt : rebuild_struct(bgt, (; w = w))
end
# `@propagatable` writes `factory(x::BudgetMarketImpact, args...)`, so this method names the
# concrete type to be the more specific one.
function factory(bgt::BudgetMarketImpact, w::VecNum)::BudgetMarketImpact
    return bgt.fixed ? bgt : rebuild_struct(bgt, (; w = w))
end
function factory(bgt::Number, ::VecNum)::Number
    return bgt
end
"""
    needs_previous_weights(bgt::BudgetCostEstimator) -> Bool
    needs_previous_weights(bgt::Number) -> Bool

Tell whether a net budget reads the weights of the previous fold or step.

A cost budget that is not `fixed` measures its cost from the weights that [`factory`](@ref factory(bgt::BudgetCosts, w::VecNum)) gives it, so it needs them. A `fixed` cost budget and a number need none. So a walk-forward that holds a cost budget that is not `fixed` runs its folds in order, as one that holds a [`Turnover`](@ref) does.

# Algorithm

 1. On a [`BudgetCostEstimator`](@ref), return `!bgt.fixed`.
 2. On a number, return `false`.

# Arguments

  - `bgt`: The net budget of a [`JuMPOptimiser`](@ref).

# Returns

  - `Bool`: `true` if `bgt` needs the previous weights, else `false`.

# Related

  - [`factory(bgt::BudgetCosts, w::VecNum)`](@ref)
  - [`needs_previous_weights(tn::TnE_Tn)`](@ref): Reads the same `fixed` flag.
"""
function needs_previous_weights(bgt::BudgetCostEstimator)::Bool
    return !bgt.fixed
end
function needs_previous_weights(::Number)::Bool
    return false
end
function set_budget_constraints!(model::JuMP.Model, val::Number, w::VecNum;
                                 prefix::Symbol = Symbol(""))
    budget_row!(model, prefix, :bgt, val, ==, w)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, w::VecNum;
                                 prefix::Symbol = Symbol(""))
    budget_row!(model, prefix, :bgt_lb, bgt.lb, >=, w)
    budget_row!(model, prefix, :bgt_ub, bgt.ub, <=, w)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Check that a budget that pays costs goes on the first block of a model.

A [`BudgetCosts`](@ref) or a [`BudgetMarketImpact`](@ref) registers the trade variables `wp` and `wn` and the expression `cost_bgt_expr` in the shared Model State, so a second one in the same model collides with the first. The net budget belongs to the owner of the model, whose block is the first, under the empty prefix.

# Arguments

  - `bgt`: The budget that pays costs.
  - `prefix`: The Model State namespace of the block.

# Validation

  - `prefix` is empty. An `ArgumentError` is thrown otherwise.

# Returns

  - `nothing`.

# Related

  - [`set_budget_constraints!`](@ref)
  - [`set_block_constraints!`](@ref)
"""
function assert_cost_budget_prefix(bgt::BudgetCostEstimator, prefix::Symbol)::Nothing
    @argcheck(prefix == Symbol(""),
              ArgumentError("a `$(nameof(typeof(bgt)))` net budget registers its trade variables in the shared Model State, so it goes on the first block of a model, the block of its owner. The block under the prefix `$prefix` must take `bgt = nothing`."))
    return nothing
end
"""
    set_long_short_budget_constraints!(args...; kwargs...)

Set budget constraints for long and short portfolio positions in the JuMP model.

Various overloads handle different budget types (fixed, range), dispatching on the presence or absence of long/short budget configurations. The rows go under `prefix`, the namespace of the block, and they read the parts that [`weight_parts!`](@ref) gives the block. The net budget is the budget of the owner of the block, so the block of a set passes `nothing`.

# Arguments

  - `args...`: JuMP model and budget parameters.
  - $(arg_dict[:prefix_kw])

# Returns

  - `nothing`.

# Related

  - [`BudgetRange`](@ref)
  - [`set_cost_budget_constraints!`](@ref)
  - [`weight_parts!`](@ref)
"""
function set_long_short_budget_constraints!(args...; kwargs...)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that a gross budget is admissible alongside the net and short budgets.

`bgt` and `sbgt` constrain the net and gross exposures only *together*. When both are numbers, the long budget is `sum(lw) == bgt + sbgt` and the short budget is `sum(sw) == sbgt`, so the gross budget `sum(lw) + sum(sw)` is `bgt + 2 * sbgt`, the row that `gbgt` writes. The realised gross exposure `norm(w, 1)` equals it when `xbgt` is `true`, and it can be smaller when `xbgt` is `false`. `gbgt` exists for the combination they cannot reach — gross pinned with net free — so it is rejected when they already determine the gross exposure, and when the weight bounds forbid short positions (whereupon gross and net coincide and `bgt` already owns the constraint). A gross budget whose upper end lies below the magnitude of the net budget is rejected too, see [`assert_gross_budget_reaches_net`](@ref).

Only the statically-decidable contradictions are caught. Ranges may still conflict at solve time, surfacing as infeasibility, and a [`WeightBoundsEstimator`](@ref) is not resolved until then. [`TimeDependent`](@ref) schedules are reached through [`assert_time_dependent_substitution`](@ref).

A [`ProgrammeAllocationSet`](@ref) and the held optimiser of a [`FollowTheLeader`](@ref) rule pass `bgt = 1`, the budget of one that the online head owns.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`set_gross_budget_constraints!`](@ref)
  - [`set_long_short_budget_constraints!`](@ref)
  - [`gross_budget_bounds_msg`](@ref)
  - [`assert_gross_budget_reaches_net`](@ref)
"""
function assert_gross_budget_admissible(bgt, sbgt, gbgt, wb)::Nothing
    @argcheck(!(isa(bgt, Number) && isa(sbgt, Number)),
              ArgumentError("gross budget (gbgt) is over-determined: net budget (bgt = $bgt) and short budget (sbgt = $sbgt) already pin the gross budget at bgt + 2 * sbgt = $(bgt + 2 * sbgt). Give at most two of bgt, sbgt and gbgt. gbgt exists for the case that bgt and sbgt cannot express, a pinned gross budget with a free net exposure (bgt = nothing)."))
    if isa(wb, WeightBounds)
        @argcheck(w_neg_flag(wb.lb) || w_neg_flag(wb.ub),
                  ArgumentError(gross_budget_bounds_msg(wb.lb, wb.ub)))
    end
    assert_gross_budget_reaches_net(bgt, gbgt)
    return nothing
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Assert that the gross budget `gbgt` can reach the magnitude of the net budget `bgt`.

The gross row reads the long and short parts of the weights, and each part bounds its side of the weight, `lw_i >= max(w_i, 0)` and `sw_i >= max(-w_i, 0)`. This holds under both decomposition contracts. So the gross exposure is at least the magnitude of the net exposure, and a gross budget whose upper end lies below the least magnitude that the net budget admits leaves no feasible weights. The bound holds at every `k`, because both rows scale with it.

# Mathematical definition

```math
\\begin{align}
\\sum_i lw_i + \\sum_i sw_i &\\geq \\sum_i \\lvert w_i \\rvert \\geq \\Bigl\\lvert \\sum_i w_i \\Bigr\\rvert\\,.
\\end{align}
```

Where:

  - ``lw_i``, ``sw_i``: The long and short parts of weight ``i``.
  - $(math_dict[:w_port])

# Validation

  - The upper end of `gbgt` is non-negative. An `ArgumentError` is thrown otherwise.
  - The upper end of `gbgt` is at least [`net_budget_floor`](@ref) of `bgt`. An `ArgumentError` is thrown otherwise.

A `gbgt` with no upper end, a `bgt` whose interval holds zero, `nothing`, a [`BudgetCostEstimator`](@ref) and a [`TimeDependent`](@ref) schedule state no static floor or ceiling, so they pass.

# Related

  - [`assert_gross_budget_admissible`](@ref)
  - [`net_budget_floor`](@ref)
  - [`set_gross_budget_constraints!`](@ref)
"""
function assert_gross_budget_reaches_net(bgt, gbgt)::Nothing
    hi = gross_budget_ceiling(gbgt)
    if isnothing(hi)
        return nothing
    end
    @argcheck(zero(hi) <= hi,
              ArgumentError("gross budget (gbgt) has the upper end $hi, below zero: the gross exposure sum(lw) + sum(sw) is at least norm(w, 1), so it is never negative."))
    lo = net_budget_floor(bgt)
    if !isnothing(lo)
        @argcheck(lo <= hi,
                  ArgumentError("gross budget (gbgt) has the upper end $hi, below $lo, the least magnitude of the net budget (bgt = $bgt): the gross exposure sum(lw) + sum(sw) is at least norm(w, 1), which is at least abs(sum(w)), so no weights meet both budgets. Raise gbgt to at least $lo, or let bgt admit a smaller magnitude."))
    end
    return nothing
end
"""
    net_budget_floor(bgt::Number)
    net_budget_floor(bgt::BudgetRange)
    net_budget_floor(bgt)

Returns the least magnitude `abs(sum(w))` that the net budget `bgt` admits, or `nothing` when it states no positive floor.

A number admits its own magnitude. A [`BudgetRange`](@ref) admits the distance from zero to its interval: its lower bound when that is positive, the negative of its upper bound when that is negative, and no floor when the interval holds zero. Every other value, `nothing`, a [`BudgetCostEstimator`](@ref) and a [`TimeDependent`](@ref) schedule among them, states no floor.

# Related

  - [`assert_gross_budget_reaches_net`](@ref)
"""
function net_budget_floor(bgt::Number)
    return abs(bgt)
end
function net_budget_floor(bgt::BudgetRange)
    lb = bgt.lb
    ub = bgt.ub
    return if !isnothing(lb) && lb > zero(lb)
        lb
    elseif !isnothing(ub) && ub < zero(ub)
        -ub
    else
        nothing
    end
end
function net_budget_floor(::Any)
    return nothing
end
"""
    gross_budget_ceiling(gbgt::Number)
    gross_budget_ceiling(gbgt::BudgetRange)
    gross_budget_ceiling(gbgt)

Returns the upper end of the gross budget `gbgt`, or `nothing` when it states none.

A number is its own upper end, and a [`BudgetRange`](@ref) has its `ub`. Every other value, `nothing` and a [`TimeDependent`](@ref) schedule among them, states none.

# Related

  - [`assert_gross_budget_reaches_net`](@ref)
"""
function gross_budget_ceiling(gbgt::Number)
    return gbgt
end
function gross_budget_ceiling(gbgt::BudgetRange)
    return gbgt.ub
end
function gross_budget_ceiling(::Any)
    return nothing
end
# No gross budget, or a schedule that `assert_time_dependent_substitution` reaches, has
# nothing to assert here.
function assert_gross_budget_admissible(::Any, ::Any, ::Union{Nothing, TimeDependent},
                                        ::Any)::Nothing
    return nothing
end
"""
    set_gross_budget_constraints!(model::JuMP.Model, gbgt::Option{<:Num_BgtRg}; prefix::Symbol = Symbol(""))

Constrain the gross exposure (leverage) `sum(lw) + sum(sw)`, independently of the net exposure.

A number pins it; a [`BudgetRange`](@ref) bounds it on either side. Whether the constraint pins the *realised* gross exposure `norm(w, 1)` or merely bounds it depends on `xbgt` — see [`short_mip_threshold_constraints`](@ref). The rows go under `prefix`, the namespace of the block, and they read the parts that [`weight_parts!`](@ref) gives the block.

# Arguments

  - `model::JuMP.Model`: JuMP optimisation model.
  - `gbgt`: Gross budget specification, or `nothing` for no constraint.
  - $(arg_dict[:prefix_kw])

# Returns

  - `nothing`.

# Related

  - [`JuMPOptimiser`](@ref)
  - [`assert_gross_budget_admissible`](@ref)
  - [`set_long_short_budget_constraints!`](@ref)
"""
function set_gross_budget_constraints!(args...; kwargs...)
    return nothing
end
function set_gross_budget_constraints!(model::JuMP.Model, gbgt::Number;
                                       prefix::Symbol = Symbol(""))
    lw, sw = weight_parts!(model, prefix)
    budget_row!(model, prefix, :gbgt, gbgt, ==, lw, sw)
    return nothing
end
function set_gross_budget_constraints!(model::JuMP.Model, gbgt::BudgetRange;
                                       prefix::Symbol = Symbol(""))
    lw, sw = weight_parts!(model, prefix)
    budget_row!(model, prefix, :gbgt_lb, gbgt.lb, >=, lw, sw)
    budget_row!(model, prefix, :gbgt_ub, gbgt.ub, <=, lw, sw)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Number, ::Nothing;
                                            prefix::Symbol = Symbol(""))
    lw, _ = weight_parts!(model, prefix)
    budget_row!(model, prefix, :lbgt, bgt, ==, lw)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, ::Nothing, sbgt::Number;
                                            prefix::Symbol = Symbol(""))
    _, sw = weight_parts!(model, prefix)
    budget_row!(model, prefix, :sbgt, sbgt, ==, sw)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Number, sbgt::Number;
                                            prefix::Symbol = Symbol(""))
    lw, sw = weight_parts!(model, prefix)
    budget_row!(model, prefix, :lbgt, bgt + sbgt, ==, lw)
    budget_row!(model, prefix, :sbgt, sbgt, ==, sw)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::BudgetRange, ::Nothing;
                                            prefix::Symbol = Symbol(""))
    lw, _ = weight_parts!(model, prefix)
    budget_row!(model, prefix, :lbgt_lb, bgt.lb, >=, lw)
    budget_row!(model, prefix, :lbgt_ub, bgt.ub, <=, lw)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, ::Nothing, sbgt::BudgetRange;
                                            prefix::Symbol = Symbol(""))
    _, sw = weight_parts!(model, prefix)
    budget_row!(model, prefix, :sbgt_lb, sbgt.lb, >=, sw)
    budget_row!(model, prefix, :sbgt_ub, sbgt.ub, <=, sw)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::BudgetRange,
                                            sbgt::Number; prefix::Symbol = Symbol(""))
    lw, sw = weight_parts!(model, prefix)
    budget_row!(model, prefix, :lbgt_lb, offset_bound(bgt.lb, sbgt), >=, lw)
    budget_row!(model, prefix, :lbgt_ub, offset_bound(bgt.ub, sbgt), <=, lw)
    budget_row!(model, prefix, :sbgt, sbgt, ==, sw)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::Number,
                                            sbgt::BudgetRange; prefix::Symbol = Symbol(""))
    lw, sw = weight_parts!(model, prefix)
    budget_row!(model, prefix, :lbgt_lb, offset_bound(sbgt.lb, bgt), >=, lw)
    budget_row!(model, prefix, :sbgt_lb, sbgt.lb, >=, sw)
    budget_row!(model, prefix, :lbgt_ub, offset_bound(sbgt.ub, bgt), <=, lw)
    budget_row!(model, prefix, :sbgt_ub, sbgt.ub, <=, sw)
    return nothing
end
function set_long_short_budget_constraints!(model::JuMP.Model, bgt::BudgetRange,
                                            sbgt::BudgetRange; prefix::Symbol = Symbol(""))
    lw, sw = weight_parts!(model, prefix)
    # A bound of the long part needs the bound of the net budget, and it adds the bound of
    # the short budget on the same side when there is one.
    budget_row!(model, prefix, :lbgt_lb, offset_bound(bgt.lb, sbgt.lb), >=, lw)
    budget_row!(model, prefix, :sbgt_lb, sbgt.lb, >=, sw)
    budget_row!(model, prefix, :lbgt_ub, offset_bound(bgt.ub, sbgt.ub), <=, lw)
    budget_row!(model, prefix, :sbgt_ub, sbgt.ub, <=, sw)
    return nothing
end
"""
    budget_row!(model::JuMP.Model, prefix::Symbol, name::Symbol, v::Nothing, sense, xs...)
    budget_row!(model::JuMP.Model, prefix::Symbol, name::Symbol, v::Number, sense, x::AbstractVector)
    budget_row!(model::JuMP.Model, prefix::Symbol, name::Symbol, v::Number, sense, lw::AbstractVector, sw::AbstractVector)

Add one budget row on the sum of one vector or of two, `sc * (sum(x) - k * v)` against zero, under `prefix` and `name`.

`sense` is `==`, `>=` or `<=`, the comparison of the row with zero, and the method states it as the MOI set `EqualTo`, `GreaterThan` or `LessThan` at zero. That set gives the row that the comparison gives. A bound `v = nothing` states no side, so the method adds no row. Every budget builder writes its rows through this function, so each row of a block goes into the namespace of the block.

The method takes the vectors and not their sum, so the whole expression goes inside one `JuMP.@constraint`, which builds it in place. A sum built outside the macro allocates each partial expression. On 50 assets, the gross row built from `s = sum(lw) + sum(sw)` and `e = sc * (s - k * v)` outside the macro allocated 50168 bytes, and the same row inside the macro allocated 38568 bytes.

# Arguments

  - $(arg_dict[:model])
  - `prefix`: The Model State namespace of the block.
  - `name`: The name of the row.
  - `v`: The budget, a number, or `nothing` for no row.
  - `sense`: The comparison with zero.
  - `x`, or `lw` and `sw`: The vector whose sum the budget bounds, or the two vectors whose sums it bounds together.

# Returns

  - `nothing`.

# Related

  - [`set_budget_constraints!`](@ref)
  - [`set_long_short_budget_constraints!`](@ref)
  - [`set_gross_budget_constraints!`](@ref)
"""
function budget_row!(::JuMP.Model, ::Symbol, ::Symbol, ::Nothing, ::Any,
                     ::AbstractVector...)::Nothing
    return nothing
end
function budget_row!(model::JuMP.Model, prefix::Symbol, name::Symbol, v::Number, sense,
                     x::AbstractVector)::Nothing
    k = get_k(model)
    sc = get_constraint_scale(model)
    set = budget_row_set(model, sense)
    state_set!(model, prefix, name, JuMP.@constraint(model, sc * (sum(x) - k * v) in set))
    return nothing
end
function budget_row!(model::JuMP.Model, prefix::Symbol, name::Symbol, v::Number, sense,
                     lw::AbstractVector, sw::AbstractVector)::Nothing
    k = get_k(model)
    sc = get_constraint_scale(model)
    set = budget_row_set(model, sense)
    state_set!(model, prefix, name,
               JuMP.@constraint(model, sc * (sum(lw) + sum(sw) - k * v) in set))
    return nothing
end
"""
    budget_row_set(model::JuMP.Model, ::typeof(==))
    budget_row_set(model::JuMP.Model, ::typeof(>=))
    budget_row_set(model::JuMP.Model, ::typeof(<=))

Return the MOI set at zero, in the value type of `model`, that a comparison of a budget row with zero states.

# Returns

  - `MOI.EqualTo`, `MOI.GreaterThan` or `MOI.LessThan` at zero.

# Related

  - [`budget_row!`](@ref)
"""
function budget_row_set(model::JuMP.Model, ::typeof(==))
    return JuMP.MOI.EqualTo(zero(JuMP.value_type(typeof(model))))
end
function budget_row_set(model::JuMP.Model, ::typeof(>=))
    return JuMP.MOI.GreaterThan(zero(JuMP.value_type(typeof(model))))
end
function budget_row_set(model::JuMP.Model, ::typeof(<=))
    return JuMP.MOI.LessThan(zero(JuMP.value_type(typeof(model))))
end
"""
    offset_bound(b::Nothing, s)
    offset_bound(b::Number, s::Nothing)
    offset_bound(b::Number, s::Number)

Return the bound `b` of the long part of the weights, offset by the short budget `s` on the same side.

The long part holds the net budget plus the short budget, `sum(lw) = sum(w) + sum(sw)`. So a side of the long part is the side of the net budget plus the side of the short budget. A missing side of the net budget gives no row, and a missing side of the short budget adds nothing.

# Returns

  - `nothing` when `b` is `nothing`, `b` when `s` is `nothing`, and `b + s` otherwise.

# Related

  - [`set_long_short_budget_constraints!`](@ref)
"""
function offset_bound(::Nothing, ::Any)::Nothing
    return nothing
end
function offset_bound(b::Number, ::Nothing)::Number
    return b
end
function offset_bound(b::Number, s::Number)::Number
    return b + s
end
"""
    set_cost_budget_constraints!(model, vp, vn, val_or_bgt, w, cp, cn)

Charge the trading costs against the portfolio budget.

Registers the `cost_bgt_expr` expression and the constraint that the weights and their costs
must together meet the budget. Two overloads handle the two budget shapes: a fixed value pins
the sum, and a [`BudgetRange`](@ref) bounds it on either side.

The **charged quantity** is a positional argument, not a field read from the model, because
the two cost families charge different quantities. [`BudgetCosts`](@ref) charges the raw
increments `wp` and `wn`; [`BudgetMarketImpact`](@ref) charges the power-cone variables `wip`
and `win`, which bound the increments raised to `1/beta`. Both pass the same coefficient
vectors.

# Arguments

  - $(arg_dict[:model])
  - `vp`: Cost coefficients for positive trades; a vector or a scalar.
  - `vn`: Cost coefficients for negative trades; a vector or a scalar.
  - `val_or_bgt`: Fixed budget value or [`BudgetRange`](@ref).
  - `w`: Portfolio weight vector.
  - `cp`: The positive-side quantity the coefficients charge.
  - `cn`: The negative-side quantity the coefficients charge.

# Returns

  - `nothing`.

# Related

  - [`set_budget_constraints!`](@ref)
  - [`BudgetCosts`](@ref)
  - [`BudgetMarketImpact`](@ref)
  - [`set_long_short_budget_constraints!`](@ref)
"""
function set_cost_budget_constraints!(model::JuMP.Model, vp::Num_VecNum, vn::Num_VecNum,
                                      val::Number, w::VecNum, cp::VecNum, cn::VecNum)
    k = get_k(model)
    sc = get_constraint_scale(model)
    JuMP.@expression(model, cost_bgt_expr, dot_scalar(vp, cp) + dot_scalar(vn, cn))
    JuMP.@constraint(model, cost_bgt, sc * (sum(w) + cost_bgt_expr - k * val) == 0)
    return nothing
end
function set_cost_budget_constraints!(model::JuMP.Model, vp::Num_VecNum, vn::Num_VecNum,
                                      bgt::BudgetRange, w::VecNum, cp::VecNum, cn::VecNum)
    k = get_k(model)
    sc = get_constraint_scale(model)
    lb = bgt.lb
    ub = bgt.ub
    JuMP.@expression(model, cost_bgt_expr, dot_scalar(vp, cp) + dot_scalar(vn, cn))
    if !isnothing(lb)
        JuMP.@constraint(model, cost_bgt_lb, sc * (sum(w) + cost_bgt_expr - k * lb) >= 0)
    end
    if !isnothing(ub)
        JuMP.@constraint(model, cost_bgt_ub, sc * (sum(w) + cost_bgt_expr - k * ub) <= 0)
    end
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetCosts, w::VecNum;
                                 prefix::Symbol = Symbol(""))
    assert_cost_budget_prefix(bgt, prefix)
    wb = bgt.w
    up = bgt.up
    un = bgt.un
    k = get_k(model)
    sc = get_constraint_scale(model)
    N = length(w)
    JuMP.@variable(model, wp[1:N], lower_bound = 0)
    JuMP.@variable(model, wn[1:N], lower_bound = 0)
    JuMP.@constraint(model, wp_ub, sc * (wp ⊖ up * k) <= 0)
    JuMP.@constraint(model, wn_ub, sc * (wn ⊖ un * k) <= 0)
    JuMP.@constraint(model, cost_bgt_trade, sc * (w - wb * k - wp + wn) == 0)
    set_cost_budget_constraints!(model, bgt.vp, bgt.vn, bgt.bgt, w, wp, wn)
    return nothing
end
function set_budget_constraints!(model::JuMP.Model, bgt::BudgetMarketImpact, w::VecNum;
                                 prefix::Symbol = Symbol(""))
    assert_cost_budget_prefix(bgt, prefix)
    wb = bgt.w
    up = bgt.up
    un = bgt.un
    k = get_k(model)
    beta = bgt.beta
    sc = get_constraint_scale(model)
    N = length(w)
    JuMP.@variable(model, wp[1:N], lower_bound = 0)
    JuMP.@variable(model, wn[1:N], lower_bound = 0)
    JuMP.@variable(model, wip[1:N])
    JuMP.@variable(model, win[1:N])
    JuMP.@constraint(model, wp_ub, sc * (wp ⊖ up * k) <= 0)
    JuMP.@constraint(model, wn_ub, sc * (wn ⊖ un * k) <= 0)
    JuMP.@constraint(model, cost_bgt_trade, sc * (w - wb * k - wp + wn) == 0)
    JuMP.@constraint(model, wip_pcone[i = 1:N],
                     [sc * wip[i], sc * k, sc * wp[i]] in JuMP.MOI.PowerCone(beta))
    JuMP.@constraint(model, win_pcone[i = 1:N],
                     [sc * win[i], sc * k, sc * wn[i]] in JuMP.MOI.PowerCone(beta))
    set_cost_budget_constraints!(model, bgt.vp, bgt.vn, bgt.bgt, w, wip, win)
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

A [`HeldIndicators`](@ref) bundle takes the [`WeightsFromParts`](@ref) method alone. The held
builder pins the decomposition only for that head, because [`run_mip_builder!`](@ref) gives an
explicit `xbgt` on a model with a short side to the sign or the long-short builder. A held bundle
under any other contract is a `TypeError`. So the [`PartsBoundWeights`](@ref) method, and the
method that reads the contract off the model, take the two bundles that carry binaries:
[`LongShortIndicators`](@ref) and [`SignIndicators`](@ref).

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
                                       ind::Union{LongShortIndicators, SignIndicators},
                                       wx::VecNum, ss, ::PartsBoundWeights)
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
                                       ind::Union{LongShortIndicators, SignIndicators},
                                       wx::VecNum, ss)
    return set_exact_budget_constraints!(model, sp, ind, wx, ss,
                                         decomposition_contract(model))
end
function set_exact_budget_constraints!(model::JuMP.Model, sp::AbstractMIPSpace,
                                       ind::HeldIndicators, wx::VecNum, ss)
    # The held builder pins the decomposition only for a `WeightsFromParts` head: an
    # explicit `xbgt` on a model with a short side runs the sign or the long-short builder
    # (`run_mip_builder!`). So the held bundle never meets the slack-closing rows.
    return set_exact_budget_constraints!(model, sp, ind, wx, ss,
                                         decomposition_contract(model)::WeightsFromParts)
end

export BudgetRange, BudgetCosts, BudgetMarketImpact
