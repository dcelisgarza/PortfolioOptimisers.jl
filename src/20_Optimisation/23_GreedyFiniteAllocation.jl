"""
$(DocStringExtensions.TYPEDEF)

Result type for [`GreedyAllocation`](@ref).

`shares`, `cost` and `w` are signed: a short position carries a negative share count, a negative cost and a negative weight. `cash` is the cash left over after the long side is allocated.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GreedyAllocationResult(;
        retcode::OptimisationReturnCode,
        shares::VecNum,
        cost::VecNum,
        w::VecNum,
        cash::Number,
        fb::Option{<:OptE_Opt}
    ) -> GreedyAllocationResult

Keywords correspond to the struct's fields.

# Related

  - [`GreedyAllocation`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)
  - [`DiscreteAllocationResult`](@ref)

# References

  - $(ref_dict[:martin2021])
"""
@concrete struct GreedyAllocationResult <: FiniteAllocationOptimisationResult
    """
    $(field_dict[:retcode])
    """
    retcode
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
    $(field_dict[:fb])
    """
    fb
    function GreedyAllocationResult(retcode::OptimisationReturnCode, shares::VecNum,
                                    cost::VecNum, w::VecNum, cash::Number,
                                    fb::Option{<:OptE_Opt})
        return new{typeof(retcode), typeof(shares), typeof(cost), typeof(w), typeof(cash),
                   typeof(fb)}(retcode, shares, cost, w, cash, fb)
    end
end
function GreedyAllocationResult(; retcode::OptimisationReturnCode, shares::VecNum,
                                cost::VecNum, w::VecNum, cash::Number,
                                fb::Option{<:OptE_Opt})::GreedyAllocationResult
    return GreedyAllocationResult(retcode, shares, cost, w, cash, fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Greedy Allocation portfolio optimiser.

`GreedyAllocation` converts continuous portfolio weights to discrete share quantities using a greedy two-pass allocation: the first pass buys down the target weights in descending order, and the second pass spends the leftover cash on the asset whose realised weight falls furthest short of its target.

The long and the short side of a portfolio are allocated as two separate sub-problems. Each sub-problem receives its own share of the cash, and its own weights are renormalised to sum to one. The definition below describes one such sub-problem, whose targets ``\\boldsymbol{w}`` therefore satisfy ``\\sum_i w_i = 1``. Short shares are negated when the two sides are recombined.

# Mathematical definition

Order the assets so that ``w_1 \\geq w_2 \\geq \\ldots \\geq w_N``. The first pass walks that order and buys

```math
\\begin{align}
x_i &= \\mathrm{round}\\!\\left(\\left\\lfloor \\frac{w_i C}{p_i \\, \\mathrm{unit}} \\right\\rfloor \\mathrm{unit}\\right)\\,, \\\\
r &\\leftarrow r - x_i p_i\\,,
\\end{align}
```

starting from ``r = C``. The pass **stops at the first asset it cannot afford**, so every later asset in the order is left at zero for the second pass to reach.

The second pass repeats, while ``r > 0``,

```math
\\begin{align}
\\boldsymbol{d} &= \\boldsymbol{w} - \\frac{\\boldsymbol{x} \\odot \\boldsymbol{p}}{\\sum_{j=1}^{N} x_j p_j}\\,, \\\\
i^* &= \\underset{i:\\, p_i \\, \\mathrm{unit} \\leq r}{\\arg\\max}\\; d_i\\,, \\\\
x_{i^*} &\\leftarrow x_{i^*} + \\mathrm{unit}\\,, \\\\
r &\\leftarrow r - p_{i^*} \\mathrm{unit}\\,.
\\end{align}
```

The pass stops when no affordable asset has a positive deficit. The selection is by **deficit**, not by target weight: an asset the first pass already filled has a small deficit however large its target weight is.

Where:

  - ``\\boldsymbol{x}``: Share allocation vector.
  - ``r``: Cash not yet spent.
  - ``\\boldsymbol{w}``: Target weight vector of this sub-problem, renormalised to sum to one.
  - ``C``: Cash allocated to this sub-problem.
  - ``\\boldsymbol{p}``: Asset price vector.
  - ``\\mathrm{unit}``: Minimum share purchase unit.
  - ``\\boldsymbol{d}``: Weight deficit, the target weight less the realised weight.
  - ``i^*``: Affordable asset with the largest weight deficit.
  - ``\\odot``: Element-wise (Hadamard) product.
  - ``N``: Number of assets in this sub-problem.

The rounding is a **floor** to a multiple of `unit`, followed by `Base.round` under `args` and `kwargs`. See [`roundmult`](@ref): it is not a round to the nearest multiple.

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

## Validation

  - $(val_dict[:unit])

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

  - [`optimise`](@ref)
  - [`GreedyAllocationResult`](@ref)
  - [`FiniteAllocationOptimisationEstimator`](@ref)
  - [`DiscreteAllocation`](@ref)
  - [`roundmult`](@ref)

# References

  - $(ref_dict[:martin2021])
"""
@concrete struct GreedyAllocation <: FiniteAllocationOptimisationEstimator
    """
    $(field_dict[:unit])
    """
    unit
    """
    Additional positional arguments forwarded to `round`.
    """
    args
    """
    $(field_dict[:kwargs])
    """
    kwargs
    """
    $(field_dict[:fb])
    """
    fb
    function GreedyAllocation(unit::Number, args::Tuple, kwargs::NamedTuple,
                              fb::Option{<:FOptE_FOpt} = nothing)
        @argcheck(unit > zero(unit), DomainError(unit, "`unit` must be positive"))
        return new{typeof(unit), typeof(args), typeof(kwargs), typeof(fb)}(unit, args,
                                                                           kwargs, fb)
    end
end
function GreedyAllocation(; unit::Number = 1, args::Tuple = (), kwargs::NamedTuple = (;),
                          fb::Option{<:FOptE_FOpt} = nothing)::GreedyAllocation
    return GreedyAllocation(unit, args, kwargs, fb)
end
"""
    roundmult(val, prec, args...; kwargs...) -> Number

Truncate `val` towards zero to a multiple of `prec`, then round that multiple with `Base.round`.

Equivalent to `round(div(val, prec) * prec, args...; kwargs...)`. This is **not** a round to the nearest multiple of `prec`: `div` truncates, so `roundmult(7.5, 2)` is `6.0` where the nearest multiple of `2` is `8.0`. The trailing `Base.round` acts on the truncated product, so a `prec` below one can leave a value that is no longer a multiple of `prec` unless `args` or `kwargs` say otherwise. Pass `RoundDown` in `args` to suppress it.

# Arguments

  - `val::Number`: Value to truncate.
  - `prec::Number`: Multiple to truncate to.
  - `args...`: Positional arguments forwarded to `Base.round`, such as a `RoundingMode`.
  - `kwargs...`: Keyword arguments forwarded to `Base.round`, such as `digits` or `sigdigits`.

# Returns

  - `res::Number`: The truncated and rounded value.

# Examples

```jldoctest
julia> PortfolioOptimisers.roundmult(7.5, 2)
6.0

julia> PortfolioOptimisers.roundmult(26.58, 1)
26.0
```

# Related

  - [`GreedyAllocation`](@ref)
"""
function roundmult(val::Number, prec::Number, args...; kwargs...)
    return round(div(val, prec) * prec, args...; kwargs...)
end
"""
    finite_sub_allocation!(w::VecNum, p::VecNum, cash::Number, bgt::Number,
                           ga::GreedyAllocation, args...)

Run the greedy two-pass allocation over one side, long or short, of the portfolio.

Implements the two passes of [`GreedyAllocation`](@ref) for a single side. An empty `w` returns three empty vectors and the untouched `cash`.

# Arguments

  - `w::VecNum`: Target weights of this side. The routine writes `w ./= sum(w)` through the view it is given, so a caller that needs the original weights must pass a copy.
  - `p::VecNum`: Asset prices of this side, in the same order as `w`.
  - `cash::Number`: Cash allocated to this side.
  - `bgt::Number`: Budget of this side, used to rescale the realised weights.
  - `ga::GreedyAllocation`: Allocator carrying `unit`, `args` and `kwargs`.
  - `args...`: Ignored. Present so that both allocators share one call shape.

# Returns

  - `shares::VecNum`: Share count per asset, restored to the caller's asset order.
  - `cost::VecNum`: `shares .* p`, in the caller's order.
  - `aw::VecNum`: Realised weights, rescaled to sum to `bgt`. All zero when nothing was bought.
  - `acash::Number`: Cash left over.

# Details

  - The assets are sorted by descending target weight, and the answer is permuted back before it is returned.
  - The affordability test in the second pass is on `p[i] * unit`, the cost of one purchase, so the pass never spends more cash than it holds.

# Related

  - [`GreedyAllocation`](@ref)
  - [`roundmult`](@ref)
  - [`finite_sub_allocation`](@ref)
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

        # If we can't afford it, go through the rest of the tickers from highest deviation to lowest.
        # The purchase below spends `_pi * unit`, so that is what must be affordable: testing `_pi`
        # alone overdraws the budget whenever `unit > 1`, and refuses affordable buys when `unit < 1`.
        while _pi * unit > acash
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
function _optimise(ga::GreedyAllocation, fai::FiniteAllocationInput; kwargs...)
    w, p, cash, T, fees = fai.w, fai.prices, fai.cash, fai.horizon, fai.fees
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
    return GreedyAllocationResult(; retcode = OptimisationSuccess(),
                                  shares = view(res, :, 1), cost = view(res, :, 2),
                                  w = view(res, :, 3), cash = lcash, fb = nothing)
end
"""
    optimise(ga::GreedyAllocation{<:Any, <:Any, <:Any, Nothing},
             fai::FiniteAllocationInput; kwargs...) -> GreedyAllocationResult

Run the Greedy Allocation portfolio optimisation.

# Arguments

  - `ga`: The greedy allocation optimiser to use.
  - `fai`: The [`FiniteAllocationInput`](@ref) carrying the target weights, prices, cash budget, and optional horizon and fees.
  - `kwargs`: Additional keyword arguments passed to the optimisation function.

# Returns

  - `res::GreedyAllocationResult`: The realised allocation. `retcode` is always an [`OptimisationSuccess`](@ref), because the greedy passes cannot fail.

# Related

  - [`GreedyAllocation`](@ref)
  - [`GreedyAllocationResult`](@ref)
  - [`FiniteAllocationInput`](@ref)
"""
function optimise(ga::GreedyAllocation{<:Any, <:Any, <:Any, Nothing},
                  fai::FiniteAllocationInput; kwargs...)
    return _optimise(ga, fai; kwargs...)
end

export GreedyAllocationResult, GreedyAllocation
