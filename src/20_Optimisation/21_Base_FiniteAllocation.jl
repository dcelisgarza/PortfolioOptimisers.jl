"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for finite allocation portfolio optimisation estimators.

Finite allocation estimators convert continuous portfolio weights into discrete share quantities given an investment budget and asset prices.

# Related Types

  - [`OptimisationEstimator`](@ref)
  - [`DiscreteAllocation`](@ref)
  - [`GreedyAllocation`](@ref)
"""
abstract type FiniteAllocationOptimisationEstimator <: OptimisationEstimator end
"""
$(DocStringExtensions.TYPEDEF)

Abstract supertype for finite allocation optimisation result types.

# Related Types

  - [`OptimisationResult`](@ref)
  - [`DiscreteAllocationResult`](@ref)
  - [`GreedyAllocationResult`](@ref)
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

# Related

  - [`DiscreteAllocation`](@ref)
  - [`GreedyAllocation`](@ref)
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
    setup_alloc_optim(w, p, cash, ...)

Set up the data structures needed for finite allocation optimisation.

Separates the portfolio into long and short positions, computes budgets and cash allocations for each side, and returns indices and cash values for downstream allocation routines.

# Arguments

  - `w`: Portfolio weights vector.
  - `p`: Asset price vector.
  - `cash`: Total cash available.
  - Additional parameters for budget configuration.

# Returns

  - Named tuple or multiple values with allocation setup data.

# Related

  - [`adjust_long_cash`](@ref)
  - [`finite_sub_allocation`](@ref)
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
    adjust_long_cash(bgt, lcash, scash)

Adjust the long-side cash allocation based on budget and short-side cash.

Redistributes cash between long and short portfolio sides to satisfy the overall budget constraint.

# Arguments

  - `bgt`: Portfolio budget (sum of weights target).
  - `lcash`: Long-side cash allocation.
  - `scash`: Short-side cash allocation.

# Returns

  - Adjusted long-side cash.

# Related

  - [`setup_alloc_optim`](@ref)
  - [`finite_sub_allocation`](@ref)
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
