abstract type BaseAssetAllocationOptimisationEstimator <: BaseOptimisationEstimator end
abstract type AssetAllocationOptimisationAlgorithm <: OptimisationAlgorithm end
struct AllocationResult{T1 <: AbstractVector, T2 <: AbstractVector, T3 <: AbstractVector,
                        T4 <: Real}
    shares::T1
    cost::T2
    w::T3
    cash::T4
end
struct DiscreteAllocation{T1 <: Union{<:Solver, <:AbstractVector{<:Solver}}} <:
       BaseAssetAllocationOptimisationEstimator
    slv::T1
end
function DiscreteAllocation(; slv::Union{<:Solver, <:AbstractVector{<:Solver}})
    if isa(slv, AbstractVector)
        @smart_assert(!isempty(slv))
    end
    return DiscreteAllocation{typeof(slv)}(slv)
end
struct GreedyAllocation{T1 <: Real, T2 <: Tuple, T3 <: NamedTuple} <:
       BaseAssetAllocationOptimisationEstimator
    unit::T1
    args::T2
    kwargs::T3
end
function GreedyAllocation(; unit::Real = 1, args::Tuple = (), kwargs::NamedTuple = (;))
    return GreedyAllocation{typeof(unit), typeof(args), typeof(kwargs)}(unit, args, kwargs)
end
function setup_alloc_optim(w::AbstractVector, p::AbstractVector, T::Integer,
                           fees::Union{Nothing, <:Fees}, cash::Real)
    fees = calc_fees(w, p, fees) * T
    cash -= fees
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
function adjust_long_cash(bgt::Real, lcash::Real, scash::Real)
    if iszero(scash)
        return lcash
    end
    return if bgt >= one(bgt)
        lcash - scash
    elseif bgt < one(bgt)
        lcash + scash
    end
end
function roundmult(val::Real, prec::Real, args...; kwargs...)
    return round(div(val, prec) * prec, args...; kwargs...)
end
function greedy_sub_allocation!(w::AbstractVector, p::AbstractVector, cash::Real,
                                tcash::Real, bgt::Real, ga::GreedyAllocation)
    if isempty(w)
        return Vector{eltype(w)}(undef, 0), Vector{eltype(w)}(undef, 0),
               Vector{eltype(w)}(undef, 0), zero(eltype(w))
    end

    idx = sortperm(w; rev = true)
    w = view(w, idx)
    p = view(p, idx)

    N = length(w)
    acash = cash
    shares = zeros(eltype(w), N)
    w ./= sum(w)
    unit = ga.unit

    # First loop
    for (i, (wi, _pi)) ∈ enumerate(zip(w, p))
        n_shares = roundmult(wi * cash / _pi, unit, ga.args...; ga.kwargs...)
        cost = n_shares * _pi
        if cost > acash
            break
        end
        acash -= cost
        shares[i] = n_shares
    end

    # Second loop
    while acash > 0
        # Calculate equivalent continuous w of what has already been bought.
        current_w = p .* shares
        current_w /= sum(current_w)

        deficit = w - current_w

        # Try to buy tickers whose deficit is the greatest.
        i = argmax(deficit)
        _pi = p[i]

        # If we can't afford it, go through the rest of the tickers from highest deviation to lowest
        while _pi > acash
            deficit[i] = 0
            i = argmax(deficit)
            if deficit[i] <= 0
                break
            end
            _pi = p[i]
        end
        if deficit[i] <= 0
            break
        end
        # Buy one share*unit at a time.
        shares[i] += unit
        acash -= _pi * unit
    end
    cost = p .* shares
    aw = cost / sum(cost) * bgt
    idx = invperm(idx)
    return view(shares, idx), view(cost, idx), view(aw, idx), acash
end
function optimise!(ga::GreedyAllocation, w::AbstractVector, p::AbstractVector, T::Integer,
                   fees::Union{Nothing, <:Fees} = nothing; cash::Real = 1e6)
    @smart_assert(!isempty(w) && !isempty(p) && length(w) == length(p))
    cash, bgt, lbgt, sbgt, lidx, sidx, lcash, scash = setup_alloc_optim(w, p, T, fees, cash)
    sshares, scost, sw, scash = greedy_sub_allocation!(-view(w, sidx), view(p, sidx), scash,
                                                       cash, sbgt, ga)
    lcash = adjust_long_cash(bgt, lcash, scash)
    lshares, lcost, lw, lcash = greedy_sub_allocation!(view(w, lidx), view(p, lidx), lcash,
                                                       cash, lbgt, ga)
    res = Matrix{eltype(w)}(undef, length(w), 3)
    res[lidx, 1] = lshares
    res[sidx, 1] = -sshares
    res[lidx, 2] = lcost
    res[sidx, 2] = -scost
    res[lidx, 3] = lw
    res[sidx, 3] = -sw
    return AllocationResult(view(res, :, 1), view(res, :, 2), view(res, :, 3), lcash)
end

export AllocationResult, DiscreteAllocation, GreedyAllocation
