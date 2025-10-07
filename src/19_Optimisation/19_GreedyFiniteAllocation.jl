struct GreedyAllocationOptimisation{T1, T2, T3, T4, T5, T6} <: OptimisationResult
    oe::T1
    shares::T2
    cost::T3
    w::T4
    retcode::T5
    cash::T6
end
struct GreedyAllocation{T1, T2, T3} <: BaseFiniteAllocationOptimisationEstimator
    unit::T1
    args::T2
    kwargs::T3
    function GreedyAllocation(unit::Real, args::Tuple, kwargs::NamedTuple)
        return new{typeof(unit), typeof(args), typeof(kwargs)}(unit, args, kwargs)
    end
end
function GreedyAllocation(; unit::Real = 1, args::Tuple = (), kwargs::NamedTuple = (;))
    return GreedyAllocation(unit, args, kwargs)
end
function roundmult(val::Real, prec::Real, args...; kwargs...)
    return round(div(val, prec) * prec, args...; kwargs...)
end
function finite_sub_allocation!(w::AbstractVector, p::AbstractVector, cash::Real, bgt::Real,
                                ga::GreedyAllocation, args...)
    if isempty(w)
        return Vector{eltype(w)}(undef, 0), Vector{eltype(w)}(undef, 0),
               Vector{eltype(w)}(undef, 0), cash
    end

    idx = sortperm(w; rev = true)
    w = view(w, idx)
    p = view(p, idx)

    N = length(w)
    acash = cash
    shares = zeros(eltype(w), N)
    w /= sum(w)
    unit = ga.unit

    # First loop
    for (i, (wi, _pi)) in enumerate(zip(w, p))
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
    aw = if any(!iszero, cost)
        cost / sum(cost) * bgt
    else
        range(; start = 0, stop = 0, length = N)
    end
    idx = invperm(idx)
    return view(shares, idx), view(cost, idx), view(aw, idx), acash
end
function optimise(ga::GreedyAllocation, w::AbstractVector, p::AbstractVector,
                  cash::Real = 1e6, T::Union{Nothing, <:Real} = nothing,
                  fees::Union{Nothing, <:Fees} = nothing; kwargs...)
    @argcheck(!isempty(w) && !isempty(p) && length(w) == length(p))
    @argcheck(cash > zero(cash))
    if !isnothing(fees)
        @argcheck(!isnothing(T))
    end
    cash, bgt, lbgt, sbgt, lidx, sidx, lcash, scash = setup_alloc_optim(w, p, cash, T, fees)
    sshares, scost, sw, scash = finite_sub_allocation!(-view(w, sidx), view(p, sidx), scash,
                                                       sbgt, ga)
    lcash = adjust_long_cash(bgt, lcash, scash)
    lshares, lcost, lw, lcash = finite_sub_allocation!(view(w, lidx), view(p, lidx), lcash,
                                                       lbgt, ga)
    res = Matrix{eltype(w)}(undef, length(w), 3)
    res[lidx, 1] = lshares
    res[sidx, 1] = -sshares
    res[lidx, 2] = lcost
    res[sidx, 2] = -scost
    res[lidx, 3] = lw
    res[sidx, 3] = -sw
    return GreedyAllocationOptimisation(typeof(ga), view(res, :, 1), view(res, :, 2),
                                        view(res, :, 3), OptimisationSuccess(nothing),
                                        lcash)
end

export GreedyAllocationOptimisation, GreedyAllocation
