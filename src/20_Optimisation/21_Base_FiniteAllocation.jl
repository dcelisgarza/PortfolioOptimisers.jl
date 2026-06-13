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
