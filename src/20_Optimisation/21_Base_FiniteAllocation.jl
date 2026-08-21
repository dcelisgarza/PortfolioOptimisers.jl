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

Every subtype carries `shares`, `cost`, `w`, `cash` and a trailing `fb`, in that order, which is what lets the generic [`factory`](@ref) rebuild any of them by swapping the last field alone.

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

`FiniteAllocationInput` bundles the inputs shared by every finite allocation optimiser — the target continuous weights, current asset prices, cash budget, and optional time horizon and fees — into a single value passed as the second argument to [`optimise`](@ref). It is consumed by both [`DiscreteAllocation`](@ref) and [`GreedyAllocation`](@ref).

It subtypes [`AbstractEstimator`](@ref) rather than the [`FiniteAllocationOptimisationResult`](@ref) tree: it is the *input* to an allocation, not a computed output, and is deliberately kept clear of the `OptimisationResult` dispatch surface (plotting, result `factory`) that its fields cannot honour. See ADR 0017.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    FiniteAllocationInput(;
        w::VecNum,
        prices::VecNum,
        cash::Number = 1e6,
        horizon::Option{<:Number} = nothing,
        fees::Option{<:Fees} = nothing
    ) -> FiniteAllocationInput

Keywords correspond to the struct's fields.

## Validation

  - `!isempty(w)`, `!isempty(prices)`.
  - `length(w) == length(prices)`.
  - `cash > 0`.
  - `horizon` must not be `nothing` when `fees` is provided.

# Examples

```jldoctest
julia> FiniteAllocationInput(; w = [0.6, 0.4], prices = [10.0, 20.0], cash = 1000.0)
FiniteAllocationInput
        w ┼ Vector{Float64}: [0.6, 0.4]
   prices ┼ Vector{Float64}: [10.0, 20.0]
     cash ┼ Float64: 1000.0
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
    Optional time horizon; used to adjust the cash budget for the fees charged over that horizon. Required when `fees` is provided.
    """
    horizon
    """
    Optional fees to charge against the allocation over `horizon`.
    """
    fees
    function FiniteAllocationInput(w::VecNum, prices::VecNum, cash::Number,
                                   horizon::Option{<:Number}, fees::Option{<:Fees})
        @argcheck(!isempty(w), IsEmptyError("w cannot be empty"))
        @argcheck(!isempty(prices), IsEmptyError("prices cannot be empty"))
        @argcheck(length(w) == length(prices),
                  DimensionMismatch("w ($(length(w))) must match prices ($(length(prices)))"))
        @argcheck(cash > zero(cash), DomainError(cash, "cash must be > 0"))
        if !isnothing(fees)
            @argcheck(!isnothing(horizon),
                      IsNothingError("horizon cannot be nothing when fees are provided"))
        end
        return new{typeof(w), typeof(prices), typeof(cash), typeof(horizon), typeof(fees)}(w,
                                                                                           prices,
                                                                                           cash,
                                                                                           horizon,
                                                                                           fees)
    end
end
function FiniteAllocationInput(; w::VecNum, prices::VecNum, cash::Number = 1e6,
                               horizon::Option{<:Number} = nothing,
                               fees::Option{<:Fees} = nothing)::FiniteAllocationInput
    return FiniteAllocationInput(w, prices, cash, horizon, fees)
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
    setup_alloc_optim(w::VecNum, p::VecNum, cash::Number,
                      T::Option{<:Number} = nothing, fees::Option{<:Fees} = nothing)

Split a portfolio into its long and its short side, and share the cash between them.

Both finite allocators solve one sub-problem per side. This routine charges the fees against the cash, computes the budget of each side, and gives each side the share of the cash its budget calls for.

# Arguments

  - `w::VecNum`: Target portfolio weights over the whole universe.
  - `p::VecNum`: Asset prices, in the same order as `w`.
  - `cash::Number`: Cash available before fees.
  - `T::Option{<:Number} = nothing`: Time horizon over which the fees are charged.
  - `fees::Option{<:Fees} = nothing`: Fees to charge. Ignored unless `T` is also given.

# Returns

  - `cash::Number`: Cash after the fees are charged.
  - `bgt::Number`: Total budget, `sum(w)`.
  - `lbgt::Number`: Long-side budget, the sum of the non-negative weights.
  - `sbgt::Number`: Short-side budget, the **negated** sum of the negative weights, so it is non-negative.
  - `lidx`: Mask of the long side, `w .>= 0`.
  - `sidx`: Mask of the short side. Empty when the portfolio is long only.
  - `lcash::Number`: `cash * lbgt`, before [`adjust_long_cash`](@ref) corrects it.
  - `scash::Number`: `cash * sbgt`. Zero when the portfolio is long only.

# Details

  - A zero weight counts as long, because the test is `w .>= 0`.
  - The fees are charged once against the whole `cash`, before the split, so both sides see the net figure.
  - `lcash` is the long side's share of the **gross** cash. It is only correct once the short side has reported what it did not spend, which is why [`adjust_long_cash`](@ref) runs between the two sub-problems.

# Related

  - [`adjust_long_cash`](@ref)
  - [`finite_sub_allocation`](@ref)
  - [`finite_sub_allocation!`](@ref)
  - [`FiniteAllocationInput`](@ref)
"""
function setup_alloc_optim(w::VecNum, p::VecNum, cash::Number,
                           T::Option{<:Number} = nothing, fees::Option{<:Fees} = nothing)
    if !isnothing(T) && !isnothing(fees)
        cash -= calc_fees(w, p, fees) * T
    end
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
    return cash, bgt, lbgt, sbgt, lidx, sidx, lcash, scash
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
