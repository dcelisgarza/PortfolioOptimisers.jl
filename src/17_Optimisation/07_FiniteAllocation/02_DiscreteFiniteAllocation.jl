"""
$(DocStringExtensions.TYPEDEF)

Result type for [`DiscreteAllocation`](@ref).

`shares`, `cost` and `w` are signed, so a short position carries a negative share count, a negative cost and a negative weight. `fees` is the charge the two sub-problems paid over the whole horizon, and it is never signed. `retcode` is a failure when either sub-problem failed. `s_retcode` and `l_retcode` carry the return code of the short and of the long side, and `s_model` and `l_model` carry the two JuMP models when `save` is `true`.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DiscreteAllocationResult(;
        retcode::OptimisationReturnCode,
        s_retcode::Option{<:OptimisationReturnCode},
        l_retcode::Option{<:OptimisationReturnCode},
        shares::VecNum,
        cost::VecNum,
        w::VecNum,
        cash::Number,
        fees::Number,
        s_model::Option{<:JuMP.Model},
        l_model::Option{<:JuMP.Model},
        fb::Option{<:FOptE_FOpt_FbChain}
    ) -> DiscreteAllocationResult

Keywords correspond to the struct's fields.

# Related

  - [`DiscreteAllocation`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)
  - [`GreedyAllocationResult`](@ref)

# References

  - $(ref_dict[:martin2021])
"""
@concrete struct DiscreteAllocationResult <: FiniteAllocationOptimisationResult
    """
    $(field_dict[:retcode])
    """
    retcode
    """
    $(field_dict[:s_retcode])
    """
    s_retcode
    """
    $(field_dict[:l_retcode])
    """
    l_retcode
    """
    $(field_dict[:shares])
    """
    shares
    """
    $(field_dict[:cost_alloc])
    """
    cost
    """
    Realised portfolio weights.
    """
    w
    """
    $(field_dict[:cash_alloc])
    """
    cash
    """
    $(field_dict[:fees_alloc])
    """
    fees
    """
    $(field_dict[:s_model])
    """
    s_model
    """
    $(field_dict[:l_model])
    """
    l_model
    """
    $(field_dict[:fb_res])
    """
    fb
    function DiscreteAllocationResult(retcode::OptimisationReturnCode,
                                      s_retcode::Option{<:OptimisationReturnCode},
                                      l_retcode::Option{<:OptimisationReturnCode},
                                      shares::VecNum, cost::VecNum, w::VecNum, cash::Number,
                                      fees::Number, s_model::Option{<:JuMP.Model},
                                      l_model::Option{<:JuMP.Model},
                                      fb::Option{<:FOptE_FOpt_FbChain})
        return new{typeof(retcode), typeof(s_retcode), typeof(l_retcode), typeof(shares),
                   typeof(cost), typeof(w), typeof(cash), typeof(fees), typeof(s_model),
                   typeof(l_model), typeof(fb)}(retcode, s_retcode, l_retcode, shares, cost,
                                                w, cash, fees, s_model, l_model, fb)
    end
end
function DiscreteAllocationResult(; retcode::OptimisationReturnCode,
                                  s_retcode::Option{<:OptimisationReturnCode},
                                  l_retcode::Option{<:OptimisationReturnCode},
                                  shares::VecNum, cost::VecNum, w::VecNum, cash::Number,
                                  fees::Number, s_model::Option{<:JuMP.Model},
                                  l_model::Option{<:JuMP.Model},
                                  fb::Option{<:FOptE_FOpt_FbChain})::DiscreteAllocationResult
    return DiscreteAllocationResult(retcode, s_retcode, l_retcode, shares, cost, w, cash,
                                    fees, s_model, l_model, fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Discrete Allocation portfolio optimiser.

`DiscreteAllocation` turns target weights into whole share counts by solving a Mixed-Integer Programming (MIP) problem. The book tracks the target as closely as the error of `wf` measures it, and leaves as little cash idle as it can.

The long and the short side of a book are two separate sub-problems, each with its own cash and its own budget. Each sub-problem holds a non-negative share vector, and [`optimise`](@ref) negates the short side when it recombines the two. `s_retcode` and `l_retcode` of the result carry the two return codes.

# Mathematical definition

One sub-problem, under the default [`AbsoluteErrorWeightFinaliser`](@ref):

```math
\\begin{align}
\\underset{\\boldsymbol{x} \\in \\mathbb{Z}_{\\geq 0}^{N}}{\\min} \\quad & u + r\\,, \\\\
\\text{s.t.} \\quad & u \\geq \\lVert C \\boldsymbol{w} - \\boldsymbol{x} \\odot \\boldsymbol{p} \\rVert_1\\,, \\\\
& r = C - \\boldsymbol{x}^\\intercal \\boldsymbol{p}\\,, \\\\
& r - F(\\boldsymbol{x}) \\geq 0\\,.
\\end{align}
```

Where:

  - $(math_dict[:x_shares])
  - $(math_dict[:u_alloc_err])
  - $(math_dict[:r_cash_left])
  - $(math_dict[:F_side_fee])
  - $(math_dict[:w_side_target])
  - $(math_dict[:C_side_cash])
  - $(math_dict[:p_prices])
  - ``\\odot``: Element-wise (Hadamard) product.
  - $(math_dict[:N])

``C \\boldsymbol{w}`` is the target money of each position. ``C`` is already the side's share of the cash, so the side's weights are normalised before they are multiplied by it, and the targets of a side sum to its cash.

The fee of [`set_allocation_fees!`](@ref) enters the budget and never the objective. The objective reads *track well, and leave no capital idle*, so a fee added to ``r`` and minimised would reward the model for paying more fees, because a larger fee shrinks the leftover. The budget instead states that the fee must be affordable. The objective still pushes ``\\boldsymbol{x}^\\intercal \\boldsymbol{p}`` up, so every unit of fee competes with a unit of position, and the model drops a position whose fixed fee buys too little tracking.

`wf` selects the row that bounds ``u``, and the error term ``e`` that takes the place of ``u`` in the objective ``e + r``. The integrality and the budget do not change with it.

| `wf`                                          | Row on ``u``                                                                                                          | ``e``   |
|:--------------------------------------------- |:--------------------------------------------------------------------------------------------------------------------- |:------- |
| [`AbsoluteErrorWeightFinaliser`](@ref)        | ``u \\geq \\lVert C \\boldsymbol{w} - \\boldsymbol{x} \\odot \\boldsymbol{p} \\rVert_1``                              | ``u``   |
| [`SquaredAbsoluteErrorWeightFinaliser`](@ref) | ``u \\geq \\lVert C \\boldsymbol{w} - \\boldsymbol{x} \\odot \\boldsymbol{p} \\rVert_2``                              | ``u``   |
| [`RelativeErrorWeightFinaliser`](@ref)        | ``u \\geq \\lVert (\\boldsymbol{x} \\odot \\boldsymbol{p}) \\oslash (C \\boldsymbol{w}) - \\boldsymbol{1} \\rVert_1`` | ``C u`` |
| [`SquaredRelativeErrorWeightFinaliser`](@ref) | ``u \\geq \\lVert (\\boldsymbol{x} \\odot \\boldsymbol{p}) \\oslash (C \\boldsymbol{w}) - \\boldsymbol{1} \\rVert_2`` | ``C u`` |

Where:

  - $(math_dict[:oslash])
  - ``\\boldsymbol{1}``: Vector of ones.

The absolute error is money, and so is ``r``. The relative error is the realised weight of each position over its target weight, less one, and it has no unit. The relative formulations price it in money as ``C u``, so a relative error of one over the whole side is worth the whole side's cash. Without that factor the model gives up any amount of tracking to leave one unit less of cash idle.

!!! note

    The two `Squared` formulations bound the ``\\ell_2`` norm itself, not its square. They build a `JuMP.SecondOrderCone` over ``[u;\\, \\cdot]``. The square is monotonic on a non-negative norm, so the minimiser is the one a squared objective would give, but the objective *value* is the norm. The two relative formulations replace a zero target weight with `eps` so that the division is defined.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    DiscreteAllocation(;
        slv::Slv_VecSlv,
        sc::Number = 1,
        so::Number = 1,
        wf::JuMPWeightFinaliserFormulation = AbsoluteErrorWeightFinaliser(),
        fb::Option{<:FOptE_FOpt} = GreedyAllocation()
    ) -> DiscreteAllocation

Keywords correspond to the struct's fields.

## Validation

  - If `slv` is a vector: `!isempty(slv)`.
  - `sc > 0`, `so > 0`.

# Examples

```jldoctest
julia> DiscreteAllocation(; slv = Solver(; solver = nothing))
DiscreteAllocation
  slv ┼ Solver
      │          name ┼ String: ""
      │        solver ┼ nothing
      │      settings ┼ nothing
      │     check_sol ┼ @NamedTuple{}: NamedTuple()
      │   add_bridges ┴ Bool: true
   sc ┼ Int64: 1
   so ┼ Int64: 1
   wf ┼ AbsoluteErrorWeightFinaliser()
   fb ┼ GreedyAllocation
      │     unit ┼ Int64: 1
      │     args ┼ Tuple{}: ()
      │   kwargs ┼ @NamedTuple{}: NamedTuple()
      │       fb ┴ nothing
```

# Related

  - [`optimise`](@ref)
  - [`DiscreteAllocationResult`](@ref)
  - [`FiniteAllocationOptimisationEstimator`](@ref)
  - [`GreedyAllocation`](@ref)
  - [`set_discrete_error!`](@ref)

# References

  - $(ref_dict[:martin2021])
"""
@concrete struct DiscreteAllocation <: FiniteAllocationOptimisationEstimator
    """
    $(field_dict[:slv])
    """
    slv
    """
    $(field_dict[:sc])
    """
    sc
    """
    $(field_dict[:so])
    """
    so
    """
    $(field_dict[:wf])
    """
    wf
    """
    $(field_dict[:fb])
    """
    fb
    function DiscreteAllocation(slv::Slv_VecSlv, sc::Number, so::Number,
                                wf::JuMPWeightFinaliserFormulation,
                                fb::Option{<:FOptE_FOpt})
        if isa(slv, VecSlv)
            @argcheck(!isempty(slv), IsEmptyError("slv cannot be empty"))
        end
        @argcheck(sc > zero(sc), DomainError(sc, "sc must be > 0"))
        @argcheck(so > zero(so), DomainError(so, "so must be > 0"))
        return new{typeof(slv), typeof(sc), typeof(so), typeof(wf), typeof(fb)}(slv, sc, so,
                                                                                wf, fb)
    end
end
function DiscreteAllocation(; slv::Slv_VecSlv, sc::Number = 1, so::Number = 1,
                            wf::JuMPWeightFinaliserFormulation = AbsoluteErrorWeightFinaliser(),
                            fb::Option{<:FOptE_FOpt} = GreedyAllocation())::DiscreteAllocation
    return DiscreteAllocation(slv, sc, so, wf, fb)
end
"""
    set_discrete_error!(model::JuMP.Model, w::VecNum, p::VecNum, cash::Number,
                        wf::JuMPWeightFinaliserFormulation)

Bound the model's error variable `u` by the allocation error that `wf` selects, and return the error term of the objective.

This row is the one part of the model that separates the four formulations of [`DiscreteAllocation`](@ref). [`finite_sub_allocation`](@ref) creates `x` and `u` before it calls this method, and it builds the objective from the term this method returns.

# JuMP formulation

## Variables

  - `x`: Share count vector, read from the model.
  - `u`: Allocation error bound, read from the model.

## Constraints

One row, which `wf` names:

  - `cabs_err`, under [`AbsoluteErrorWeightFinaliser`](@ref): ``(s_c u,\\, s_c (C \\boldsymbol{w} - \\boldsymbol{x} \\odot \\boldsymbol{p})) \\in \\mathcal{K}_{1}``.
  - `csqabs_err`, under [`SquaredAbsoluteErrorWeightFinaliser`](@ref): ``(s_c u,\\, s_c (C \\boldsymbol{w} - \\boldsymbol{x} \\odot \\boldsymbol{p})) \\in \\mathcal{K}_{2}``.
  - `crel_err`, under [`RelativeErrorWeightFinaliser`](@ref): ``(s_c u,\\, s_c ((\\boldsymbol{x} \\odot \\boldsymbol{p}) \\oslash (C \\tilde{\\boldsymbol{w}}) - \\boldsymbol{1})) \\in \\mathcal{K}_{1}``.
  - `csqrel_err`, under [`SquaredRelativeErrorWeightFinaliser`](@ref): ``(s_c u,\\, s_c ((\\boldsymbol{x} \\odot \\boldsymbol{p}) \\oslash (C \\tilde{\\boldsymbol{w}}) - \\boldsymbol{1})) \\in \\mathcal{K}_{2}``.

Each row is the epigraph of a norm. The objective of [`finite_sub_allocation`](@ref) minimises ``u`` with a positive coefficient, so ``u`` equals the norm at the optimum.

Where:

  - $(math_dict[:x_shares])
  - $(math_dict[:u_alloc_err])
  - $(math_dict[:w_side_target])
  - ``\\tilde{\\boldsymbol{w}}``: ``\\boldsymbol{w}`` with each zero entry replaced by `eps(eltype(w))`, so that the division is defined.
  - $(math_dict[:C_side_cash])
  - $(math_dict[:p_prices])
  - $(math_dict[:sc_scale])
  - $(math_dict[:K_q_norm])
  - ``\\odot``: Element-wise (Hadamard) product.
  - $(math_dict[:oslash])
  - ``\\boldsymbol{1}``: Vector of ones.

# Arguments

  - $(arg_dict[:model])
  - `w::VecNum`: Side target weights, normalised to sum to one. The relative formulations replace a zero entry on a **copy**, so the caller's vector does not change.
  - `p::VecNum`: Asset prices of this side, in the same order as `w`.
  - `cash::Number`: Cash of this side.
  - `wf::JuMPWeightFinaliserFormulation`: Selects the error. See the table in [`DiscreteAllocation`](@ref).

# Returns

  - `err`: The error term of the objective, in money: `u` under an absolute formulation, and `cash * u` under a relative one, whose error has no unit.

# Related

  - [`DiscreteAllocation`](@ref)
  - [`finite_sub_allocation`](@ref)
  - [`JuMPWeightFinaliserFormulation`](@ref)
"""
function set_discrete_error!(model::JuMP.Model, w::VecNum, p::VecNum, cash::Number,
                             ::RelativeErrorWeightFinaliser)
    mask = iszero.(w)
    if any(mask)
        w = copy(w)
        w[mask] .= eps(eltype(w))
    end
    x = model[:x]
    u = model[:u]
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, crel_err,
                     [sc * u
                      sc *
                      ((x .* p) ⊘ (w * cash) .- one(promote_type(eltype(w), eltype(p))))] in
                     JuMP.MOI.NormOneCone(length(x) + 1))
    # A relative error has no unit, and the leftover cash of the objective is money.
    return cash * u
end
function set_discrete_error!(model::JuMP.Model, w::VecNum, p::VecNum, cash::Number,
                             ::SquaredRelativeErrorWeightFinaliser)
    mask = iszero.(w)
    if any(mask)
        w = copy(w)
        w[mask] .= eps(eltype(w))
    end
    x = model[:x]
    u = model[:u]
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, csqrel_err,
                     [sc * u;
                      sc *
                      ((x .* p) ⊘ (w * cash) .- one(promote_type(eltype(w), eltype(p))))] in
                     JuMP.SecondOrderCone())
    return cash * u
end
function set_discrete_error!(model::JuMP.Model, w::VecNum, p::VecNum, cash::Number,
                             ::AbsoluteErrorWeightFinaliser)
    x = model[:x]
    u = model[:u]
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, cabs_err,
                     [sc * u; sc * (w * cash .- x .* p)] in
                     JuMP.MOI.NormOneCone(length(x) + 1))
    return u
end
function set_discrete_error!(model::JuMP.Model, w::VecNum, p::VecNum, cash::Number,
                             ::SquaredAbsoluteErrorWeightFinaliser)
    x = model[:x]
    u = model[:u]
    sc = get_constraint_scale(model)
    JuMP.@constraint(model, csqabs_err,
                     [sc * u;
                      sc * (w * cash .- x .* p)] in JuMP.SecondOrderCone())
    return u
end
"""
    set_allocation_fees!(model::JuMP.Model, p::VecNum, cash::Number, sf::Option{<:NamedTuple})

Write one side's fee in the allocation model's own variables, and return it.

A fee is a cost of the book the allocator buys. The model holds the share vector `x` and the prices `p`, so `x .* p` is the money in each position exactly. Every term is written against that money, and no weight and no price appears on its own.

`sf` is one side's charge, of [`allocation_side_fees`](@ref). A `nothing` `sf` registers only a zero `fee`, so the caller needs no branch.

# Mathematical definition

```math
\\begin{align}
F(\\boldsymbol{x}) &= T \\left( \\boldsymbol{f}_{\\text{p}}^\\intercal \\boldsymbol{m} + \\boldsymbol{f}_{\\text{Tn}}^\\intercal \\lvert \\boldsymbol{m} - \\boldsymbol{m}_{0} \\rvert \\right) + \\boldsymbol{f}_{\\text{f}}^\\intercal \\mathbf{1}[\\boldsymbol{x} > 0] + F_{\\text{lq}}\\,.
\\end{align}
```

Where:

  - $(math_dict[:F_side_fee])
  - $(math_dict[:m_money])
  - ``\\boldsymbol{m}_{0}``: Money in each position before the trade, `sf.prev_money`.
  - ``\\mathbf{1}[\\boldsymbol{x} > 0]``: Indicator vector of a held position.
  - ``T``: Horizon, in periods.
  - ``\\boldsymbol{f}_{\\text{p}},\\, \\boldsymbol{f}_{\\text{Tn}},\\, \\boldsymbol{f}_{\\text{f}}``: Proportional, turnover and fixed rates of this side.
  - ``F_{\\text{lq}}``: Forced exit of [`allocation_liquidation_fee`](@ref), `sf.liq`.

The rates `l`, `s` and `tn` charge on each of the `T` periods, and the fixed amounts `fl` and `fs` charge one time for the whole horizon. That is the rule [`calc_total_fees`](@ref) states, written in the money of the book.

# JuMP formulation

## Variables

  - `x`: Share count vector, read from the model.
  - `t_ftn`: Epigraph of the money traded, created when `sf` states a turnover rate.
  - `b`: Binary indicator of a held position, created when `sf` states a fixed fee.

## Expressions

  - `money`: ``\\boldsymbol{m}``.
  - `fee_prop`: ``\\boldsymbol{f}_{\\text{p}}^\\intercal \\boldsymbol{m}``, when `sf` states a proportional rate.
  - `fee_tn`: ``\\boldsymbol{f}_{\\text{Tn}}^\\intercal \\boldsymbol{t}``, when `sf` states a turnover rate.
  - `fee_fixed`: ``\\boldsymbol{f}_{\\text{f}}^\\intercal \\boldsymbol{b}``, when `sf` states a fixed fee.
  - `fee`: ``T (f_{\\text{p}} + f_{\\text{Tn}}) + f_{\\text{f}} + F_{\\text{lq}}`` over the terms `sf` states, stored under `model[:fee]`. It is a zero expression when `sf` is `nothing`.

## Constraints

  - `cftn_ub`: ``s_c (\\boldsymbol{m} - \\boldsymbol{m}_{0} - \\boldsymbol{t}) \\leq \\boldsymbol{0}``.
  - `cftn_lb`: ``s_c (\\boldsymbol{m}_{0} - \\boldsymbol{m} - \\boldsymbol{t}) \\leq \\boldsymbol{0}``.
  - `cb_ub`: ``s_c (\\boldsymbol{x} - \\lfloor C \\oslash \\boldsymbol{p} \\rfloor \\odot \\boldsymbol{b}) \\leq \\boldsymbol{0}``.
  - `cb_lb`: ``s_c (\\boldsymbol{b} - \\boldsymbol{x}) \\leq \\boldsymbol{0}``.

`x` is integer and non-negative, so `cb_lb` holds ``b_i`` at zero when ``x_i = 0``, and `cb_ub` holds it at one when ``x_i \\geq 1``. `cb_ub` removes no book: the budget of [`finite_sub_allocation`](@ref) holds ``x_i p_i \\leq C`` under a non-negative fee, so ``x_i \\leq \\lfloor C / p_i \\rfloor``. The binary is an exact indicator.

Where:

  - $(math_dict[:x_shares])
  - $(math_dict[:m_money])
  - ``\\boldsymbol{m}_{0}``: Money in each position before the trade, `sf.prev_money`.
  - ``\\boldsymbol{t}``: Epigraph of the money traded, `t_ftn`.
  - ``\\boldsymbol{b}``: Binary indicator of a held position.
  - $(math_dict[:C_side_cash])
  - $(math_dict[:p_prices])
  - ``T``: Horizon, in periods.
  - ``\\boldsymbol{f}_{\\text{p}},\\, \\boldsymbol{f}_{\\text{Tn}},\\, \\boldsymbol{f}_{\\text{f}}``: Proportional, turnover and fixed rates of this side.
  - ``F_{\\text{lq}}``: Forced exit of [`allocation_liquidation_fee`](@ref), `sf.liq`. It is a constant, because the assets it sells are not among the share counts this model solves for.
  - $(math_dict[:sc_scale])
  - ``\\odot``: Element-wise (Hadamard) product.
  - $(math_dict[:oslash])

## Relaxation

$(val_dict[:relax])

`model[:fee]` lies at or above the fee of the book, ``F(\\boldsymbol{x})``, and the quantity bounded is `t_ftn`, which `fee_tn` and `fee` read. The bound is tight where the solver puts ``\\boldsymbol{t}`` at ``\\lvert \\boldsymbol{m} - \\boldsymbol{m}_{0} \\rvert``, and no objective term pulls it there. The set of books the model admits is exact all the same. A book whose exact fee fits the budget admits ``\\boldsymbol{t} = \\lvert \\boldsymbol{m} - \\boldsymbol{m}_{0} \\rvert``. That is why [`finite_sub_allocation`](@ref) reports [`allocation_fee`](@ref) of the realised book, and not the value of `fee`.

# Arguments

  - $(arg_dict[:model])
  - `p::VecNum`: Asset prices of this side.
  - `cash::Number`: Cash of this side. It bounds the shares a binary can switch on.
  - `sf::Option{<:NamedTuple}`: This side's charge, or `nothing`.

# Returns

  - `fee`: The fee expression, also stored under `model[:fee]`.

# Related

  - [`allocation_liquidation_fee`](@ref)
  - [`allocation_side_fees`](@ref)
  - [`finite_sub_allocation`](@ref)
  - [`DiscreteAllocation`](@ref)
  - [`Fees`](@ref)
"""
function set_allocation_fees!(model::JuMP.Model, ::VecNum, ::Number, ::Nothing)
    return model[:fee] = zero(JuMP.AffExpr)
end
function set_allocation_fees!(model::JuMP.Model, p::VecNum, cash::Number, sf::NamedTuple)
    x = model[:x]
    sc = get_constraint_scale(model)
    N = length(p)
    T = sf.T
    # The forced exit of `allocation_liquidation_fee` is a constant: the assets it sells are
    # not among the share counts this model solves for, so it needs no variable. It still
    # enters the budget, which is the whole point of charging it here.
    fee = zero(JuMP.AffExpr) + sf.liq
    # The money in each position, exactly. No weight and no price appears on its own.
    JuMP.@expression(model, money, x .* p)
    prop = sf.prop
    if !isnothing(prop)
        # Per period: the proportional rate of this side.
        JuMP.@expression(model, fee_prop, dot_scalar(prop, money))
        JuMP.add_to_expression!(fee, T, fee_prop)
    end
    tn_val = sf.tn_val
    if !isnothing(tn_val)
        # Per period: the turnover. An absolute value needs an epigraph.
        prev_money = sf.prev_money
        JuMP.@variable(model, t_ftn[1:N] >= 0)
        JuMP.@constraints(model, begin
                              cftn_ub, sc * (money .- prev_money .- t_ftn) .<= 0
                              cftn_lb, sc * (prev_money .- money .- t_ftn) .<= 0
                          end)
        JuMP.@expression(model, fee_tn, dot_scalar(tn_val, t_ftn))
        JuMP.add_to_expression!(fee, T, fee_tn)
    end
    fixed = sf.fixed
    if !isnothing(fixed)
        # One time: a fixed fee is charged per position held, so it needs a bit saying
        # whether the position is there at all. `ub` is the most shares this cash can buy.
        ub = floor.(cash ./ p)
        JuMP.@variable(model, b[1:N], Bin)
        JuMP.@constraints(model, begin
                              cb_ub, sc * (x .- ub .* b) .<= 0
                              cb_lb, sc * (b .- x) .<= 0
                          end)
        JuMP.@expression(model, fee_fixed, dot_scalar(fixed, b))
        JuMP.add_to_expression!(fee, fee_fixed)
    end
    return model[:fee] = fee
end
"""
    finite_sub_allocation(w::VecNum, p::VecNum, cash::Number, bgt::Number,
                          sf::Option{<:NamedTuple}, da::DiscreteAllocation,
                          str_names::Bool = false)

Build and solve the discrete allocation MIP for one side, long or short, of the book.

This is the sub-problem of [`DiscreteAllocation`](@ref). The share vector is integer and non-negative, so a short side must come in with its weights already negated.

# Algorithm

 1. On an empty `w`, return three empty vectors, `cash` less the fee of the empty book, that fee, and `nothing` for the return code and for the model. The fee is [`allocation_fee`](@ref) of the empty book, because a side that buys nothing can still owe a turnover fee or the forced exit.
 2. Normalise `w` to sum to one. A side whose weights sum to zero keeps them.
 3. Build the model of the `# JuMP formulation` below, and solve it with `da.slv`.
 4. Read the share vector back with `round(Int, ...)`, because a MIP solver returns an integer only to within its own tolerance. A model that holds no finite value gives an empty book. A solver that never ran holds no value, and a fee larger than the cash makes the budget infeasible, so neither model has a finite value.
 5. Make the cost `shares .* p`, and the realised weights `cost / sum(cost) * bgt`. The realised weights are all zero when nothing was bought.
 6. Price the fee with [`allocation_fee`](@ref) of the realised shares, and not with the value of `model[:fee]`, which can lie above it. See the `## Relaxation` of [`set_allocation_fees!`](@ref).
 7. Make the leftover cash, `cash` less the cost and the fee.

# JuMP formulation

## Variables

  - `x`: Share count vector, integer and non-negative.
  - `u`: Allocation error bound.

## Expressions

  - `sc`: ``s_c``, `da.sc`.
  - `so`: ``s_o``, `da.so`.
  - `r`: ``r``.

[`set_allocation_fees!`](@ref) registers `fee` and the entries behind it, and [`set_discrete_error!`](@ref) registers the row on ``u`` and returns the error term ``e``.

## Constraints

  - `cr`: ``s_c (r - F(\\boldsymbol{x})) \\geq 0``.

## Objective

  - `Min`: ``s_o (e + r)``.

Where:

  - $(math_dict[:x_shares])
  - $(math_dict[:u_alloc_err])
  - $(math_dict[:r_cash_left])
  - $(math_dict[:F_side_fee])
  - ``e``: Error term of `da.wf`, ``u`` under an absolute formulation and ``C u`` under a relative one.
  - $(math_dict[:C_side_cash])
  - $(math_dict[:p_prices])
  - $(math_dict[:sc_scale])
  - $(math_dict[:so_scale])

# Arguments

  - `w::VecNum`: Target weights of this side, non-negative.
  - `p::VecNum`: Asset prices of this side, in the same order as `w`.
  - `cash::Number`: Cash of this side.
  - `bgt::Number`: Budget of this side, used to rescale the realised weights.
  - `sf::Option{<:NamedTuple}`: This side's charge, of [`allocation_side_fees`](@ref), or `nothing`.
  - `da::DiscreteAllocation`: Allocator carrying the solvers, the scales and the formulation `wf`.
  - `str_names::Bool = false`: Whether the JuMP variables get string names.

# Returns

  - `shares::VecNum`: Share count per asset.
  - `cost::VecNum`: `shares .* p`.
  - `aw::VecNum`: Realised weights, rescaled to sum to `bgt`.
  - `acash::Number`: Cash left over, `cash` less the cost of the shares and the fee.
  - `fee::Number`: The fee this side paid over the whole horizon.
  - `res::OptimisationReturnCode`: An [`OptimisationSuccess`](@ref) or an [`OptimisationFailure`](@ref) carrying the solver trials.
  - `model::JuMP.Model`: The solved model.

# Related

  - [`DiscreteAllocation`](@ref)
  - [`set_discrete_error!`](@ref)
  - [`set_allocation_fees!`](@ref)
  - [`setup_alloc_optim`](@ref)
  - [`adjust_long_cash`](@ref)
"""
function finite_sub_allocation(w::VecNum, p::VecNum, cash::Number, bgt::Number,
                               sf::Option{<:NamedTuple}, da::DiscreteAllocation,
                               str_names::Bool = false)
    if isempty(w)
        # An empty side buys nothing, but it can still owe the forced exit: a delisted asset
        # carries a zero target weight, so it lands on the long side even when that side
        # holds nothing else. `allocation_fee` of the empty book is that charge exactly.
        fee = allocation_fee(sf, p, w)
        return Vector{eltype(w)}(undef, 0), Vector{eltype(w)}(undef, 0),
               Vector{eltype(w)}(undef, 0), cash - fee, fee, nothing, nothing
    end
    # `cash` is this side's cash, so the target money of an asset is its share of the side,
    # `w / sum(w)`, times that cash. A side whose weights are all zero keeps them.
    sw = sum(w)
    w = w / ifelse(iszero(sw), one(sw), sw)
    model = JuMP.Model()
    JuMP.set_string_names_on_creation(model, str_names)
    JuMP.@expression(model, sc, da.sc)
    JuMP.@expression(model, so, da.so)
    N = length(w)
    # x: the share counts. u: the bound on the allocation error.
    JuMP.@variables(model, begin
                        x[1:N] >= 0, Int
                        u
                    end)
    # r: the cash left before the fee.
    JuMP.@expression(model, r, cash - LinearAlgebra.dot(x, p))
    fee = set_allocation_fees!(model, p, cash, sf)
    JuMP.@constraint(model, cr, sc * (r - fee) >= 0)
    err = set_discrete_error!(model, w, p, cash, da.wf)
    JuMP.@objective(model, Min, so * (err + r))
    res = optimise_JuMP_model!(model, da.slv)
    res = if res.success
        OptimisationSuccess(; res = res.trials)
    else
        OptimisationFailure(; res = res.trials)
    end
    # A solver that never ran (no optimiser attached, or `optimize!` threw) holds no values,
    # and reading them raises `OptimizeNotCalled`; `has_values` is false there. A fee larger
    # than the cash makes the budget constraint infeasible, and an infeasible model holds no
    # solution, so its values are not finite. Either way `res` carries the failure, and the
    # book is read as empty rather than raising, so the fallback chain can walk on.
    xv = if JuMP.has_values(model)
        JuMP.value.(x)
    else
        fill(convert(JuMP.value_type(typeof(model)), NaN), N)
    end
    shares = all(isfinite, xv) ? round.(Int, xv) : zeros(Int, N)
    cost = shares .* p
    aw = if any(!iszero, cost)
        cost / sum(cost) * bgt
    else
        range(zero(eltype(w)), zero(eltype(w)); length = N)
    end
    # The reported fee is priced on the realised integer shares, not read off the model.
    # `t_ftn` is an epigraph, and only the budget constraint pushes it down, so the solver
    # leaves it slack whenever the budget does not bind. A slack epigraph only overstates
    # the fee, so the allocation it bought stays affordable.
    afee = allocation_fee(sf, p, shares)
    acash = cash - sum(cost) - afee
    return shares, cost, aw, acash, afee, res, model
end
function _optimise(da::DiscreteAllocation, fai::FiniteAllocationInput;
                   str_names::Bool = false, save::Bool = true, kwargs...)
    w, p, cash, pcash, T, fees = fai.w, fai.prices, fai.cash, fai.prev_cash, fai.horizon,
                                 fai.fees
    bgt, lbgt, sbgt, lidx, sidx, lcash, scash = setup_alloc_optim(w, cash)
    lsf, ssf = allocation_side_fees(fees, T, pcash, lidx, sidx)
    sshares, scost, sw, scash, sfee, sretcode, smodel = finite_sub_allocation(-view(w,
                                                                                    sidx),
                                                                              view(p, sidx),
                                                                              scash, sbgt,
                                                                              ssf, da,
                                                                              str_names)
    lcash = adjust_long_cash(bgt, lcash, scash)
    lshares, lcost, lw, lcash, lfee, lretcode, lmodel = finite_sub_allocation(view(w, lidx),
                                                                              view(p, lidx),
                                                                              lcash, lbgt,
                                                                              lsf, da,
                                                                              str_names)

    res = Matrix{eltype(w)}(undef, length(w), 3)
    res[lidx, 1] = lshares
    res[sidx, 1] = -sshares
    res[lidx, 2] = lcost
    res[sidx, 2] = -scost
    res[lidx, 3] = lw
    res[sidx, 3] = -sw
    retcode = if isa(sretcode, OptimisationFailure) || isa(lretcode, OptimisationFailure)
        if isa(sretcode, OptimisationFailure)
            @warn("Failed to solve sub optimisation problem. Check `s_retcode.res` for details.")
        end
        if isa(lretcode, OptimisationFailure)
            @warn("Failed to solve sub optimisation problem. Check `l_retcode.res` for details.")
        end
        OptimisationFailure()
    else
        OptimisationSuccess()
    end
    return DiscreteAllocationResult(; retcode = retcode, s_retcode = sretcode,
                                    l_retcode = lretcode, shares = view(res, :, 1),
                                    cost = view(res, :, 2), w = view(res, :, 3),
                                    cash = lcash, fees = lfee + sfee,
                                    s_model = ifelse(save, smodel, nothing),
                                    l_model = ifelse(save, lmodel, nothing), fb = nothing)
end
"""
    optimise(da::DiscreteAllocation{<:Any, <:Any, <:Any, <:Any, Nothing},
             fai::FiniteAllocationInput; str_names::Bool = false,
             save::Bool = true, kwargs...) -> DiscreteAllocationResult

Allocate a book into whole shares with a [`DiscreteAllocation`](@ref) that carries no fallback.

This method takes `da.fb === nothing` only. An allocator with a fallback goes through the generic `optimise` of an [`OptimisationEstimator`](@ref), which walks the fallback chain and runs the same steps.

# Algorithm

 1. Split the book into its long and its short side, and share the cash between them, with [`setup_alloc_optim`](@ref).
 2. Split the fee into the charge of each side with [`allocation_side_fees`](@ref).
 3. Solve the short side with [`finite_sub_allocation`](@ref), on its negated weights.
 4. Correct the long side's cash with the cash the short side did not spend, with [`adjust_long_cash`](@ref).
 5. Solve the long side with [`finite_sub_allocation`](@ref).
 6. Recombine the two sides over the whole universe, and negate the short side's shares, cost and weights.
 7. Set `retcode` to an [`OptimisationFailure`](@ref) when either side failed, with one warning per failed side, and to an [`OptimisationSuccess`](@ref) otherwise.

# Arguments

  - `da`: The discrete allocation optimiser.
  - `fai`: The [`FiniteAllocationInput`](@ref) carrying the target weights, the prices, the cash, and the optional horizon and fees.
  - `str_names`: Whether the JuMP variables get string names.
  - `save`: Whether the result keeps the two JuMP models.
  - `kwargs`: Accepted, so that every optimiser takes one call, and ignored.

# Returns

  - `res::DiscreteAllocationResult`: The realised allocation. `cash` is the cash the long side leaves, which holds what the short side did not spend, and `fees` is the sum of the two sides' fees.

# Related

  - [`DiscreteAllocation`](@ref)
  - [`DiscreteAllocationResult`](@ref)
  - [`FiniteAllocationInput`](@ref)
"""
function optimise(da::DiscreteAllocation{<:Any, <:Any, <:Any, <:Any, Nothing},
                  fai::FiniteAllocationInput; str_names::Bool = false, save::Bool = true,
                  kwargs...)
    return _optimise(da, fai; str_names = str_names, save = save, kwargs...)
end

export DiscreteAllocationResult, DiscreteAllocation
