"""
$(DocStringExtensions.TYPEDEF)

Result type for Greedy Allocation portfolio optimisation.

# Fields

$(DocStringExtensions.FIELDS)

# Related

  - [`GreedyAllocation`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)
"""
@concrete struct GreedyAllocationResult <: FiniteAllocationOptimisationResult
    "$(field_dict[:oe])"
    oe
    "$(field_dict[:retcode])"
    retcode
    "$(field_dict[:shares])"
    shares
    "$(field_dict[:cost_alloc])"
    cost
    "Realised portfolio weights."
    w
    "$(field_dict[:cash_alloc])"
    cash
    "$(field_dict[:fb])"
    fb
end
"""
$(DocStringExtensions.TYPEDSIGNATURES)

Rebuild a [`GreedyAllocationResult`](@ref) with an updated fallback optimiser `fb`.
"""
function factory(res::GreedyAllocationResult, fb::Option{<:FOptE_FOpt})
    return GreedyAllocationResult(res.oe, res.retcode, res.shares, res.cost, res.w,
                                  res.cash, fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Greedy Allocation portfolio optimiser.

`GreedyAllocation` converts continuous portfolio weights to discrete share quantities using a greedy two-pass allocation: first round shares to the nearest `unit` multiple, then iteratively buy remaining shares with leftover cash in order of largest weight.

# Mathematical definition

```math
\\begin{align}
x_i^{(0)} &= \\mathrm{round}\\!\\left(\\frac{w_i C}{p_i \\cdot \\mathrm{unit}}\\right) \\cdot \\mathrm{unit}\\,, \\\\
r^{(0)} &= C - \\boldsymbol{x}^{(0)\\intercal} \\boldsymbol{p}\\,.
\\end{align}
```

Then iteratively while ``r > 0``:

```math
\\begin{align}
i^* &= \\underset{i:\\, p_i \\leq r}{\\arg\\max}\\; w_i\\,, \\\\
x_{i^*} &\\leftarrow x_{i^*} + \\mathrm{unit}\\,, \\\\
r &\\leftarrow r - p_{i^*} \\cdot \\mathrm{unit}\\,.
\\end{align}
```

Where:

  - ``x_i^{(0)}``: Initial share allocation for asset ``i``.
  - ``r^{(0)}``: Residual cash after initial allocation.
  - ``\\boldsymbol{w}``: Target weight vector.
  - ``C``: Available cash.
  - ``\\boldsymbol{p}``: Asset price vector.
  - ``\\mathrm{unit}``: Minimum share purchase unit.
  - ``i^*``: Asset with largest weight among those affordable with remaining cash ``r``.
  - ``\\boldsymbol{x}^{(0)}``: Initial share allocation vector.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GreedyAllocation(;
        unit::Number = 1,
        args::Tuple = (),
        kwargs::NamedTuple = (;),
        fb::Option{<:FOptE_FOpt} = nothing
    ) -> GreedyAllocation

Keywords correspond to the struct's fields.

# Examples

```jldoctest
julia> GreedyAllocation()
GreedyAllocation
    unit ┼ Int64: 1
    args ┼ Tuple{}: ()
  kwargs ┼ @NamedTuple{}: NamedTuple()
      fb ┴ nothing
```

# Related

  - [`FiniteAllocationOptimisationEstimator`](@ref)
  - [`DiscreteAllocation`](@ref)
  - [`GreedyAllocationResult`](@ref)
"""
@concrete struct GreedyAllocation <: FiniteAllocationOptimisationEstimator
    "$(field_dict[:unit])"
    unit
    "Additional positional arguments forwarded to `round`."
    args
    "$(field_dict[:kwargs])"
    kwargs
    "$(field_dict[:fb])"
    fb
    function GreedyAllocation(unit::Number, args::Tuple, kwargs::NamedTuple,
                              fb::Option{<:FOptE_FOpt} = nothing)
        return new{typeof(unit), typeof(args), typeof(kwargs), typeof(fb)}(unit, args,
                                                                           kwargs, fb)
    end
end
function GreedyAllocation(; unit::Number = 1, args::Tuple = (), kwargs::NamedTuple = (;),
                          fb::Option{<:FOptE_FOpt} = nothing)::GreedyAllocation
    return GreedyAllocation(unit, args, kwargs, fb)
end
"""
    roundmult(val, prec, args...; kwargs...)

Round a value to the nearest multiple of `prec`.

# Arguments

  - `val`: Value to round.
  - `prec`: Precision (multiple to round to).
  - `args...`: Additional arguments passed to `Base.round`.
  - `kwargs...`: Additional keyword arguments passed to `Base.round`.

# Returns

  - Rounded value.

# Related

  - [`GreedyAllocation`](@ref)
"""
function roundmult(val::Number, prec::Number, args...; kwargs...)
    return round(div(val, prec) * prec, args...; kwargs...)
end
"""
    finite_sub_allocation!(w, p, cash, bgt, ...)

In-place finite allocation for one side (long or short) of the portfolio using the greedy algorithm.

Modifies the allocation in-place, greedily assigning shares to assets to minimise allocation error.

# Arguments

  - `w`: Target portfolio weights (in-place modified).
  - `p`: Asset prices.
  - `cash`: Cash available.
  - `bgt`: Budget target.
  - Additional parameters.

# Returns

  - Modified allocation vector.

# Related

  - [`finite_sub_allocation`](@ref)
  - [`GreedyAllocation`](@ref)
"""
function finite_sub_allocation!(w::VecNum, p::VecNum, cash::Number, bgt::Number,
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
        range(zero(eltype(w)), zero(eltype(w)); length = N)
    end
    idx = invperm(idx)
    return view(shares, idx), view(cost, idx), view(aw, idx), acash
end
function _optimise(ga::GreedyAllocation, w::VecNum, p::VecNum, cash::Number = 1e6,
                   T::Option{<:Number} = nothing, fees::Option{<:Fees} = nothing; kwargs...)
    @argcheck(!isempty(w))
    @argcheck(!isempty(p))
    @argcheck(length(w) == length(p))
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
    return GreedyAllocationResult(typeof(ga), OptimisationSuccess(nothing), view(res, :, 1),
                                  view(res, :, 2), view(res, :, 3), lcash, nothing)
end
"""
    optimise(ga::GreedyAllocation{<:Any, <:Any, <:Any, Nothing}, w::VecNum, p::VecNum,
             cash::Number = 1e6, T::Option{<:Number} = nothing,
             fees::Option{<:Fees} = nothing; kwargs...) -> GreedyAllocationResult

Run the Greedy Allocation portfolio optimisation.

# Arguments

  - `ga`: The greedy allocation optimiser to use.
  - $(arg_dict[:pw])
  - `p`: The prices of the assets in the same order as `w`.
  - `cash`: The initial cash balance.
  - `T`: The time horizon for the optimisation. Used to adjust the initial cash balance according to the fees charged on the portfolio for the time horizon.
  - `fees`: The fees to apply to the portfolio.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Related

  - [`GreedyAllocation`](@ref)
  - [`GreedyAllocationResult`](@ref)
"""
function optimise(ga::GreedyAllocation{<:Any, <:Any, <:Any, Nothing}, w::VecNum, p::VecNum,
                  cash::Number = 1e6, T::Option{<:Number} = nothing,
                  fees::Option{<:Fees} = nothing; kwargs...)
    return _optimise(ga, w, p, cash, T, fees; kwargs...)
end

export GreedyAllocationResult, GreedyAllocation
