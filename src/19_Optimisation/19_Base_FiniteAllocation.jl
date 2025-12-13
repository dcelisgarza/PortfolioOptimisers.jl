"""
"""
abstract type FiniteAllocationOptimisationEstimator <: OptimisationEstimator end
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
