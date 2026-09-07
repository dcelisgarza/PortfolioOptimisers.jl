"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for finite allocation portfolio optimisation estimators.

Finite allocation estimators convert continuous portfolio weights into discrete share quantities given an investment budget and asset prices.

The library ships two: [`DiscreteAllocation`](@ref), which solves a mixed-integer programme, and [`GreedyAllocation`](@ref), which walks the target weights. Both take a [`FiniteAllocationInput`](@ref) as the second argument to [`optimise`](@ref), and both split the portfolio into a long and a short sub-problem.

# Related

  - [`OptimisationEstimator`](@ref)
  - [`DiscreteAllocation`](@ref)
  - [`GreedyAllocation`](@ref)
  - [`FiniteAllocationInput`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)

# References

  - $(ref_dict[:martin2021])
"""
abstract type FiniteAllocationOptimisationEstimator <: OptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for finite allocation optimisation result types.

Every subtype carries `shares`, `cost`, `w`, `cash`, `fees` and a trailing `fb`, in that order, which is what lets the generic [`factory`](@ref) rebuild any of them by swapping the last field alone.

# Related

  - [`OptimisationResult`](@ref)
  - [`DiscreteAllocationResult`](@ref)
  - [`GreedyAllocationResult`](@ref)
  - [`FiniteAllocationOptimisationEstimator`](@ref)
  - [`factory`](@ref)
"""
abstract type FiniteAllocationOptimisationResult <: OptimisationResult end
"""
    const FOptE_FOpt = Union{<:FiniteAllocationOptimisationEstimator,
                             <:FiniteAllocationOptimisationResult}

Alias for a finite allocation optimisation estimator or result.

Matches either a [`FiniteAllocationOptimisationEstimator`](@ref) or a [`FiniteAllocationOptimisationResult`](@ref).

# Related

  - [`FiniteAllocationOptimisationEstimator`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)
"""
const FOptE_FOpt = Union{<:FiniteAllocationOptimisationEstimator,
                         <:FiniteAllocationOptimisationResult}
"""
$(DocStringExtensions.TYPEDEF)

Problem data fed to a finite allocation optimiser.

`FiniteAllocationInput` bundles the inputs shared by every finite allocation optimiser — the target continuous weights, current asset prices, cash budget, the cash held before the trade, and optional time horizon and fees — into a single value passed as the second argument to [`optimise`](@ref). It is consumed by both [`DiscreteAllocation`](@ref) and [`GreedyAllocation`](@ref).

It subtypes [`AbstractEstimator`](@ref) rather than the [`FiniteAllocationOptimisationResult`](@ref) tree: it is the *input* to an allocation, not a computed output, and is deliberately kept clear of the `OptimisationResult` dispatch surface (plotting, result `factory`) that its fields cannot honour. See ADR 0017.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FiniteAllocationInput(;
        w::VecNum,
        prices::VecNum,
        cash::Number = 1e6,
        prev_cash::Number = cash,
        horizon::Option{<:Number} = nothing,
        fees::Option{<:Fees} = nothing
    ) -> FiniteAllocationInput

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(w)`, `!isempty(prices)`.
  - `length(w) == length(prices)`.
  - `cash > 0`.
  - `prev_cash >= 0`.
  - `horizon` must not be `nothing` when `fees` is provided.

# Examples

```jldoctest
julia> FiniteAllocationInput(; w = [0.6, 0.4], prices = [10.0, 20.0], cash = 1000.0)
FiniteAllocationInput
          w ┼ Vector{Float64}: [0.6, 0.4]
     prices ┼ Vector{Float64}: [10.0, 20.0]
       cash ┼ Float64: 1000.0
  prev_cash ┼ Float64: 1000.0
    horizon ┼ nothing
       fees ┴ nothing
```

# Related

  - [`DiscreteAllocation`](@ref)
  - [`GreedyAllocation`](@ref)
  - [`FiniteAllocationOptimisationEstimator`](@ref)
  - [`setup_alloc_optim`](@ref)
  - [`optimise`](@ref)
"""
@concrete struct FiniteAllocationInput <: AbstractEstimator
    """
    Target (continuous) portfolio weights to be discretised.
    """
    w
    """
    Current asset prices, in the same order as `w`.
    """
    prices
    """
    Cash budget available for the allocation.
    """
    cash
    """
    Cash held in the portfolio before the trade. A turnover fee is charged on the money traded, and the money held per asset before the trade is `prev_cash * fees.tn.w`. Defaults to `cash`, which is the caller that states no separate figure.
    """
    prev_cash
    """
    Optional time horizon, in periods. `l`, `s` and `tn` are rates per period, so each of them charges on every one of the `horizon` periods. Required when `fees` is provided.
    """
    horizon
    """
    Optional fees to charge against the allocation over `horizon`.
    """
    fees
    function FiniteAllocationInput(w::VecNum, prices::VecNum, cash::Number,
                                   prev_cash::Number, horizon::Option{<:Number},
                                   fees::Option{<:Fees})
        @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
        @argcheck(!isempty(prices), IsEmptyError("prices cannot be empty"))
        @argcheck(length(w) == length(prices),
                  DimensionMismatch("w ($(length(w))) must match prices ($(length(prices)))"))
        @argcheck(cash > zero(cash), DomainError(cash, "cash must be > 0"))
        @argcheck(prev_cash >= zero(prev_cash),
                  DomainError(prev_cash, "prev_cash must be >= 0"))
        if !isnothing(fees)
            @argcheck(!isnothing(horizon),
                      IsNothingError("horizon cannot be nothing when fees are provided"))
        end
        return new{typeof(w), typeof(prices), typeof(cash), typeof(prev_cash),
                   typeof(horizon), typeof(fees)}(w, prices, cash, prev_cash, horizon, fees)
    end
end
function FiniteAllocationInput(; w::VecNum, prices::VecNum, cash::Number = 1e6,
                               prev_cash::Number = cash,
                               horizon::Option{<:Number} = nothing,
                               fees::Option{<:Fees} = nothing)::FiniteAllocationInput
    return FiniteAllocationInput(w, prices, cash, prev_cash, horizon, fees)
end
export FiniteAllocationInput
"""
    factory(res::FiniteAllocationOptimisationResult, fb::Option{<:FOptE_FOpt})

Rebuild a finite allocation result with an updated fallback optimiser `fb`.

Like the continuous-result generic, every finite allocation result carries `fb` as its last field, so the rebuild copies all fields unchanged except the trailing `fb`. Concrete result types may override this method when rebuilding requires more than swapping `fb`.

# Related

  - [`FOptE_FOpt`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)
"""
function factory(res::FiniteAllocationOptimisationResult, fb::Option{<:FOptE_FOpt})
    flds = ntuple(i -> getfield(res, i), Val(fieldcount(typeof(res))))
    return (typeof(res).name.wrapper)(Base.front(flds)..., fb)
end

"""
    setup_alloc_optim(w::VecNum, cash::Number)

Split a portfolio into its long and its short side, and share the cash between them.

Both finite allocators solve one sub-problem per side. This routine computes the budget of each side, and gives each side the share of the cash its budget calls for.

The routine charges no fee. A fee is a cost of the portfolio the allocator actually buys, so each sub-problem charges its own side inside its own model, on the money it buys. See [`set_allocation_fees!`](@ref) and ADR 0123.

# Arguments

  - `w::VecNum`: Target portfolio weights over the whole universe.
  - `cash::Number`: Cash available.

# Returns

  - `bgt::Number`: Total budget, `sum(w)`.
  - `lbgt::Number`: Long-side budget, the sum of the non-negative weights.
  - `sbgt::Number`: Short-side budget, the **negated** sum of the negative weights, so it is non-negative.
  - `lidx`: Mask of the long side, `w .>= 0`.
  - `sidx`: Mask of the short side. Empty when the portfolio is long only.
  - `lcash::Number`: `cash * lbgt`, before [`adjust_long_cash`](@ref) corrects it.
  - `scash::Number`: `cash * sbgt`. Zero when the portfolio is long only.

# Details

  - A zero weight counts as long, because the test is `w .>= 0`.
  - `lcash` is the long side's share of the cash. It is only correct once the short side has reported what it did not spend, which is why [`adjust_long_cash`](@ref) runs between the two sub-problems.

# Related

  - [`adjust_long_cash`](@ref)
  - [`allocation_side_fees`](@ref)
  - [`finite_sub_allocation`](@ref)
  - [`finite_sub_allocation!`](@ref)
  - [`FiniteAllocationInput`](@ref)
"""
function setup_alloc_optim(w::VecNum, cash::Number)
    bgt = sum(w)
    lidx = w .>= zero(eltype(w))
    long = all(lidx)
    if long
        lbgt = bgt
        sbgt = zero(eltype(w))
        sidx = Vector{eltype(w)}(undef, 0)
        scash = zero(eltype(w))
    else
        sidx = .!lidx
        lbgt = sum(view(w, lidx))
        sbgt = -sum(view(w, sidx))
        scash = cash * sbgt
    end
    lcash = cash * lbgt
    return bgt, lbgt, sbgt, lidx, sidx, lcash, scash
end
"""
    allocation_turnover_money(::Nothing, ::Number, ::Any, ::Bool)
    allocation_turnover_money(tn::Turnover, prev_cash::Number, idx, short::Bool)

Give one side its turnover rate and the money it held per asset before the trade.

`tn.w` is a weight vector, so the money held per asset before the trade is `prev_cash * tn.w`. A short side is allocated with its weights negated, so its money is negated too, and both sides then hold a non-negative figure.

# Arguments

  - `tn`: The turnover of the fee, or `nothing` when the fee states none.
  - `prev_cash::Number`: Cash held in the portfolio before the trade.
  - `idx`: Mask of this side.
  - `short::Bool`: Whether this side is the short one.

# Returns

  - `val`: The turnover rate of this side, or `nothing`.
  - `prev_money`: The money held per asset before the trade, or `nothing`.

# Related

  - [`allocation_side_fees`](@ref)
  - [`Turnover`](@ref)
"""
function allocation_turnover_money(::Nothing, ::Number, ::Any, ::Bool)
    return nothing, nothing
end
function allocation_turnover_money(tn::Turnover, prev_cash::Number, idx, short::Bool)
    sgn = ifelse(short, -one(prev_cash), one(prev_cash))
    return nothing_scalar_array_view(tn.val, idx), sgn * prev_cash * view(tn.w, idx)
end
"""
    allocation_side_fees(::Nothing, ::Option{<:Number}, ::Number, ::Any, ::Any)
    allocation_side_fees(fees::Fees, T::Number, prev_cash::Number, lidx, sidx)

Split a fee into the long side's charge and the short side's charge.

Each sub-problem charges its own side. The long side takes `l` and `fl`, the short side takes `s` and `fs`, and both take the turnover rate and the money they held before the trade. A rate that is a vector is viewed to the side, and a rate that is a scalar is carried through, which is what [`nothing_scalar_array_view`](@ref) does.

# Arguments

  - `fees`: The fee to split, or `nothing` when the caller states none.
  - `T`: Horizon, in periods.
  - `prev_cash::Number`: Cash held in the portfolio before the trade.
  - `lidx`: Mask of the long side.
  - `sidx`: Mask of the short side.

# Returns

  - `lsf`: The long side's charge, or `nothing`.
  - `ssf`: The short side's charge, or `nothing`.

Each charge is a named tuple of `T`, `prop`, `fixed`, `tn_val` and `prev_money`.

# Related

  - [`allocation_turnover_money`](@ref)
  - [`set_allocation_fees!`](@ref)
  - [`setup_alloc_optim`](@ref)
  - [`Fees`](@ref)
"""
function allocation_side_fees(::Nothing, ::Option{<:Number}, ::Number, ::Any, ::Any)
    return nothing, nothing
end
function allocation_side_fees(fees::Fees, T::Number, prev_cash::Number, lidx, sidx)
    ltn_val, lprev = allocation_turnover_money(fees.tn, prev_cash, lidx, false)
    stn_val, sprev = allocation_turnover_money(fees.tn, prev_cash, sidx, true)
    return ((T = T, prop = nothing_scalar_array_view(fees.l, lidx),
             fixed = nothing_scalar_array_view(fees.fl, lidx), tn_val = ltn_val,
             prev_money = lprev),
            (T = T, prop = nothing_scalar_array_view(fees.s, sidx),
             fixed = nothing_scalar_array_view(fees.fs, sidx), tn_val = stn_val,
             prev_money = sprev))
end
"""
    allocation_fee(::Nothing, ::VecNum, shares::VecNum)
    allocation_fee(sf::NamedTuple, p::VecNum, shares::VecNum)

Charge one side's whole fee against a share vector.

`shares .* p` is the money in each position exactly, and every term is charged against that money. No weight and no price appears on its own. This is the rule ADR 0123 states, and it is the number both allocators report.

`sf` is one side's charge, of [`allocation_side_fees`](@ref). A `nothing` `sf` charges nothing.

# Algorithm

 1. Proportional, per period: the rate contracted with the money.
 2. Turnover, per period: the rate contracted with the money traded, `|money - prev_money|`.
 3. Fixed, one time: the rate contracted with the indicator of a non-zero position.
 4. Return `T` times the first two terms, plus the third.

A side that held money before the trade owes a turnover fee even when it buys nothing, because selling out is a trade. That is why both allocators charge this verb against an empty book.

# Arguments

  - `sf`: One side's charge, of [`allocation_side_fees`](@ref), or `nothing`.
  - `p::VecNum`: Asset prices of this side.
  - `shares::VecNum`: Share count per asset.

# Returns

  - `fee::Number`: The whole charge of this side over the horizon.

# Related

  - [`allocation_side_fees`](@ref)
  - [`greedy_fee_delta`](@ref)
  - [`set_allocation_fees!`](@ref)
  - [`finite_sub_allocation`](@ref)
  - [`finite_sub_allocation!`](@ref)
"""
function allocation_fee(::Nothing, p::VecNum, shares::VecNum)
    return zero(promote_type(eltype(p), eltype(shares)))
end
function allocation_fee(sf::NamedTuple, p::VecNum, shares::VecNum)
    money = shares .* p
    fee = zero(promote_type(eltype(p), eltype(shares)))
    prop = sf.prop
    if !isnothing(prop)
        fee += sf.T * dot_scalar(prop, money)
    end
    tn_val = sf.tn_val
    if !isnothing(tn_val)
        fee += sf.T * dot_scalar(tn_val, abs.(money - sf.prev_money))
    end
    fixed = sf.fixed
    if !isnothing(fixed)
        fee += dot_scalar(fixed, .!iszero.(money))
    end
    return fee
end
"""
    permute_side_fees(::Nothing, ::Any)
    permute_side_fees(sf::NamedTuple, idx)

Put one side's charge into the order `idx` names.

[`finite_sub_allocation!`](@ref) sorts its assets by descending target weight, so the rates and the previous money must take the same order. A rate that is a vector is viewed to `idx`, and a rate that is a scalar is carried through, which is what [`nothing_scalar_array_view`](@ref) does.

# Arguments

  - `sf`: One side's charge, of [`allocation_side_fees`](@ref), or `nothing`.
  - `idx`: The order to put it in.

# Returns

  - `sf`: The charge in the order `idx` names, or `nothing`.

# Related

  - [`allocation_side_fees`](@ref)
  - [`finite_sub_allocation!`](@ref)
"""
function permute_side_fees(::Nothing, ::Any)
    return nothing
end
function permute_side_fees(sf::NamedTuple, idx)
    return (T = sf.T, prop = nothing_scalar_array_view(sf.prop, idx),
            fixed = nothing_scalar_array_view(sf.fixed, idx),
            tn_val = nothing_scalar_array_view(sf.tn_val, idx),
            prev_money = nothing_scalar_array_view(sf.prev_money, idx))
end
"""
    adjust_long_cash(bgt::Number, lcash::Number, scash::Number) -> Number

Correct the long side's cash with the cash the short side did not spend.

Runs between the two sub-problems, once the short side has reported its leftover cash. The correction has opposite signs above and below a unit budget, so the long side never spends cash the portfolio does not hold.

# Arguments

  - `bgt::Number`: Total budget, `sum(w)`.
  - `lcash::Number`: Long side's share of the gross cash, from [`setup_alloc_optim`](@ref).
  - `scash::Number`: Cash the short side did not spend.

# Returns

  - `res::Number`: The corrected long-side cash.

# Details

  - `scash == 0`: `lcash` is returned unchanged. A long-only portfolio takes this branch.
  - `bgt >= 1`: `lcash` exceeds the cash actually available, so the unspent short cash is **subtracted**. It is not available to the long side.
  - `bgt < 1`: `lcash` falls short of the cash actually available, so the unspent short cash is **added** without exceeding the true budget.

# Related

  - [`setup_alloc_optim`](@ref)
  - [`finite_sub_allocation`](@ref)
  - [`finite_sub_allocation!`](@ref)
"""
function adjust_long_cash(bgt::Number, lcash::Number, scash::Number)
    if iszero(scash)
        return lcash
    end
    return if bgt >= one(bgt)
        # lcash is more than the actual available cash, so if we want to remain under the available cash, we need to remove any uninvested short cash because it is not available for long positions.
        lcash - scash
    elseif bgt < one(bgt)
        # lcash is less than the actual available cash, so if we have leftover cash from the short allocation we can add it to the long positions without exceeding the actual available cash.
        lcash + scash
    end
end
