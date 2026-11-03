"""
$(DocStringExtensions.TYPEDEF)

Result type for [`GreedyAllocation`](@ref).

`shares`, `cost` and `w` carry the sign of the position. A short position has a negative share count, a negative cost and a negative weight. `fees` is the charge that the two sides paid over the whole horizon, and the short side's part of it is not negated. `cash` is the cash that the account holds after both sides trade, as the collateral algorithm of the input states it. It includes the cash that no side received, and it is negative when the book borrows. Under the default [`ProceedsCollateral`](@ref), `sum(cost) + cash + fees` is the cash of the input.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GreedyAllocationResult(;
        retcode::OptimisationReturnCode,
        shares::VecNum,
        cost::VecNum,
        w::VecNum,
        cash::Number,
        fees::Number,
        fb::Option{<:FOptE_FOpt_FbChain}
    ) -> GreedyAllocationResult

Keywords correspond to the struct's fields.

# Related

  - [`GreedyAllocation`](@ref)
  - [`FiniteAllocationOptimisationResult`](@ref)
  - [`DiscreteAllocationResult`](@ref)

# References

  - $(ref_dict[:martin2021]) Methods, page 3. $(ref_dict[:no_original_source])
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
    $(field_dict[:fees_alloc])
    """
    fees
    """
    $(field_dict[:fb_res])
    """
    fb
    function GreedyAllocationResult(retcode::OptimisationReturnCode, shares::VecNum,
                                    cost::VecNum, w::VecNum, cash::Number, fees::Number,
                                    fb::Option{<:FOptE_FOpt_FbChain})
        return new{typeof(retcode), typeof(shares), typeof(cost), typeof(w), typeof(cash),
                   typeof(fees), typeof(fb)}(retcode, shares, cost, w, cash, fees, fb)
    end
end
function GreedyAllocationResult(; retcode::OptimisationReturnCode, shares::VecNum,
                                cost::VecNum, w::VecNum, cash::Number, fees::Number,
                                fb::Option{<:FOptE_FOpt_FbChain})::GreedyAllocationResult
    return GreedyAllocationResult(retcode, shares, cost, w, cash, fees, fb)
end
"""
$(DocStringExtensions.TYPEDEF)

Greedy Allocation portfolio optimiser.

`GreedyAllocation` converts continuous portfolio weights to discrete share quantities with a greedy allocation in two passes. [martin2021](@cite) states only that the package it describes converts optimal weights into a discrete allocation, in its Methods on page 3. It does not state the passes. The first pass buys each asset up to its target weight in whole lots, in descending order of target weight. The second pass spends the leftover cash, one lot at a time, on the asset whose realised weight is furthest below its target.

`unit` states the lot unit of each asset, so the share count of an asset is a whole number of lots times its lot unit. The default `LotUnit(1)` buys whole shares. A lot unit below one buys fractional shares, and a lot unit above one buys board lots. A [`LotUnitEstimator`](@ref) resolves the lot unit against `sets`.

[`optimise`](@ref) allocates the long and the short side of a portfolio as two separate sub-problems. Each sub-problem receives its own share of the cash, and renormalises its own weights to sum to one. The definition below describes one sub-problem, so its targets ``\\boldsymbol{w}`` satisfy ``\\sum_i w_i = 1``. A side whose target weights are all zero buys nothing. `optimise` negates the short side's shares when it recombines the two sides.

# Mathematical definition

Order the assets so that ``w_1 \\geq w_2 \\geq \\ldots \\geq w_N``. The first pass goes through that order and buys

```math
\\begin{align}
x_i &= l_i \\left\\lfloor \\frac{w_i C}{p_i l_i} \\right\\rfloor\\,, \\\\
r &\\leftarrow r - x_i p_i - \\Delta F_i\\,,
\\end{align}
```

starting from ``r = C - F(\\boldsymbol{0})``. The pass stops at the first asset it cannot afford. Every later asset in the order then holds zero shares until the second pass.

The second pass repeats, while ``r > 0``,

```math
\\begin{align}
\\boldsymbol{d} &= \\boldsymbol{w} - \\frac{\\boldsymbol{x} \\odot \\boldsymbol{p}}{\\sum_{j=1}^{N} x_j p_j}\\,, \\\\
i^* &= \\underset{i:\\, p_i l_i + \\Delta F_i \\leq r}{\\arg\\max}\\; d_i\\,, \\\\
x_{i^*} &\\leftarrow x_{i^*} + l_{i^*}\\,, \\\\
r &\\leftarrow r - p_{i^*} l_{i^*} - \\Delta F_{i^*}\\,.
\\end{align}
```

While the book holds no shares, ``\\sum_j x_j p_j = 0`` and the realised weight of every asset is zero, so ``\\boldsymbol{d} = \\boldsymbol{w}``. The pass stops when no affordable asset has a positive deficit. It selects by deficit, not by target weight. An asset that the first pass filled has a small deficit, whatever its target weight is. An asset with a zero target weight never has a positive deficit, so neither pass buys it.

Where:

  - ``\\boldsymbol{x}``: Share allocation vector.
  - ``r``: Cash not yet spent.
  - ``\\boldsymbol{w}``: Target weight vector of this sub-problem, renormalised to sum to one.
  - ``C``: Cash allocated to this sub-problem.
  - ``\\boldsymbol{p}``: Asset price vector.
  - $(math_dict[:l_lot_unit])
  - ``\\boldsymbol{d}``: Weight deficit, the target weight less the realised weight.
  - ``F(\\boldsymbol{x})``: Fee of this sub-problem, from [`allocation_fee`](@ref). It is zero when the input states no fee, and it includes the constant forced exit of [`allocation_liquidation_fee`](@ref) when the universe lost an asset.
  - ``\\Delta F_i``: Amount that a purchase of the asset ``i`` adds to the fee, from [`greedy_fee_delta`](@ref).
  - ``i^*``: Affordable asset with the largest weight deficit.
  - ``\\odot``: Element-wise (Hadamard) product.
  - ``N``: Number of assets in this sub-problem.

The first pass floors the lot count, so it never buys more than the target of an asset. Both passes count whole lots and compute each share count as the lot count times the lot unit. So a share count is a multiple of its lot unit to within one rounding of a product, whatever the lot unit is.

# Fields

$(DocStringExtensions.FIELDS)

# Constructors

    GreedyAllocation(;
        unit::LuE_Lu = LotUnit(),
        sets::Option{<:UniverseSets} = nothing,
        strict::Bool = false,
        fb::Option{<:FOptE_FOpt} = nothing
    ) -> GreedyAllocation

Keywords correspond to the struct's fields.

## Validation

  - $(val_dict[:unit])

# Examples

```jldoctest
julia> GreedyAllocation()
GreedyAllocation
    unit ┼ LotUnit
         │   val ┴ Int64: 1
    sets ┼ nothing
  strict ┼ Bool: false
      fb ┴ nothing
```

# Related

  - [`optimise`](@ref)
  - [`GreedyAllocationResult`](@ref)
  - [`FiniteAllocationOptimisationEstimator`](@ref)
  - [`DiscreteAllocation`](@ref)
  - [`LotUnit`](@ref)
  - [`LotUnitEstimator`](@ref)

# References

  - $(ref_dict[:martin2021]) Methods, page 3. $(ref_dict[:no_original_source])
"""
@concrete struct GreedyAllocation <: FiniteAllocationOptimisationEstimator
    """
    $(field_dict[:unit])
    """
    unit
    """
    $(field_dict[:sets_alloc])
    """
    sets
    """
    $(field_dict[:strict_opt])
    """
    strict
    """
    $(field_dict[:fb])
    """
    fb
    function GreedyAllocation(unit::LuE_Lu, sets::Option{<:UniverseSets}, strict::Bool,
                              fb::Option{<:FOptE_FOpt} = nothing)
        if isa(unit, LotUnitEstimator)
            @argcheck(!isnothing(sets),
                      IsNothingError("sets cannot be nothing when unit is a LotUnitEstimator"))
        end
        return new{typeof(unit), typeof(sets), typeof(strict), typeof(fb)}(unit, sets,
                                                                           strict, fb)
    end
end
function GreedyAllocation(; unit::LuE_Lu = LotUnit(),
                          sets::Option{<:UniverseSets} = nothing, strict::Bool = false,
                          fb::Option{<:FOptE_FOpt} = nothing)::GreedyAllocation
    return GreedyAllocation(unit, sets, strict, fb)
end
"""
    greedy_fee_delta(::Nothing, ::VecNum, shares::VecNum, ::Integer, ::Number)
    greedy_fee_delta(sf::NamedTuple, p::VecNum, shares::VecNum, i::Integer, qty::Number)

Charge the amount that a purchase of `qty` shares of asset `i` adds to one side's fee.

The greedy allocator buys one asset at a time against a running cash figure, so it charges the fee purchase by purchase. The delta of each term is exact and needs no loop. The affordability test of both passes compares the cash with the cost of the shares plus this number.

# Algorithm

 1. On a `nothing` charge, return a zero.
 2. Compute `m0 = shares[i] * p[i]`, the money in the position before the purchase, and `m1 = m0 + qty * p[i]`, the money after it.
 3. Proportional term: add `T * prop[i] * (m1 - m0)`.
 4. Turnover term: add `T * tn_val[i] * (abs(m1 - prev) - abs(m0 - prev))`, where `prev` is the money that the position held before the trade. This term is negative when the purchase moves the position towards `prev`.
 5. Fixed term: add `fixed[i]` when `m0` is zero and `m1` is not.
 6. Return `delta`, the sum of the terms.

# Arguments

  - `sf`: One side's charge, from [`allocation_side_fees`](@ref), or `nothing`.
  - `p::VecNum`: Asset prices of this side.
  - `shares::VecNum`: Share count per asset, before the purchase.
  - `i::Integer`: The asset bought.
  - `qty::Number`: The share count bought.

# Returns

  - `delta::Number`: What the purchase adds to the fee.

# Related

  - [`allocation_fee`](@ref)
  - [`finite_sub_allocation!`](@ref)
"""
function greedy_fee_delta(::Nothing, p::VecNum, shares::VecNum, ::Integer, ::Number)
    return zero(promote_type(eltype(p), eltype(shares)))
end
function greedy_fee_delta(sf::NamedTuple, p::VecNum, shares::VecNum, i::Integer,
                          qty::Number)
    pi = p[i]
    m0 = shares[i] * pi
    m1 = m0 + qty * pi
    delta = zero(promote_type(eltype(p), eltype(shares)))
    prop = sf.prop
    if !isnothing(prop)
        delta += sf.T * nothing_scalar_array_getindex(prop, i) * (m1 - m0)
    end
    tn_val = sf.tn_val
    if !isnothing(tn_val)
        prev = sf.prev_money[i]
        delta += sf.T *
                 nothing_scalar_array_getindex(tn_val, i) *
                 (abs(m1 - prev) - abs(m0 - prev))
    end
    fixed = sf.fixed
    if !isnothing(fixed) && iszero(m0) && !iszero(m1)
        delta += nothing_scalar_array_getindex(fixed, i)
    end
    return delta
end
"""
    finite_sub_allocation!(w::VecNum, p::VecNum, lu::Num_VecNum, cash::Number, bgt::Number,
                           sf::Option{<:NamedTuple}, ga::GreedyAllocation)

Run the greedy two-pass allocation over one side, long or short, of the portfolio.

This method runs the two passes of [`GreedyAllocation`](@ref) for a single side. It changes none of its arguments, because it sorts through views and renormalises into a new vector. The affordability test of both passes compares the cash with the cost of the purchase plus the amount of [`greedy_fee_delta`](@ref), so a stated fee cannot overdraw the budget.

# Algorithm

 1. If `w` is empty, charge `fee`, the [`allocation_fee`](@ref) of the empty book. Return three empty vectors, `cash - fee` and `fee`. An empty side can still owe the forced exit of [`allocation_liquidation_fee`](@ref).
 2. Sort the assets by descending target weight into `idx`. View `w`, `p` and a vector `lu` in that order, and put the charge in that order with [`permute_side_fees`](@ref).
 3. Charge `fee`, the fee of the empty book, and start the running cash `acash = cash - fee`. This fee is not zero when the side held money before the trade, because the turnover fee charges the sale, or when the universe lost an asset.
 4. Renormalise `w` to sum to one. A side whose target weights are all zero keeps them at zero.
 5. In the first pass, go through the order and buy `n_lots`, the floor of the lots of the target of each asset. Stop at the first asset whose `cost` is more than `acash`.
 6. In the second pass, while `acash > 0`, compute the `deficit` of each asset. Buy one lot of the affordable asset with the largest positive deficit, and stop when no such asset remains.
 7. Both passes count the lots of each asset, and set its share count to the lot count times its lot unit.
 8. Compute `cost = p .* shares` and the realised weights `aw`, rescaled to sum to `bgt`. Charge `fee`, the fee of the whole book, and compute `acash = cash - sum(cost) - fee`.
 9. Permute `shares`, `cost` and `aw` back to the caller's order with `invperm(idx)`.

# Arguments

  - `w::VecNum`: Target weights of this side.
  - `p::VecNum`: Asset prices of this side, in the same order as `w`.
  - `lu::Num_VecNum`: Lot unit of this side, one scalar for every asset or one value for each asset in the order of `w`.
  - `cash::Number`: Cash allocated to this side.
  - `bgt::Number`: Budget of this side, used to rescale the realised weights.
  - `sf::Option{<:NamedTuple}`: This side's charge, from [`allocation_side_fees`](@ref), or `nothing`.
  - `ga::GreedyAllocation`: Allocator of the side. The method reads none of its fields, because [`allocation_lot_unit`](@ref) resolved its lot unit to `lu`.

# Returns

  - `shares::VecNum`: Share count per asset, in the caller's asset order. Each entry is a whole number of lots of its lot unit.
  - `cost::VecNum`: `shares .* p`, in the caller's order.
  - `aw::VecNum`: Realised weights, rescaled to sum to `bgt`. All zero when nothing was bought.
  - `acash::Number`: Cash left over after the shares and the fee.
  - `fee::Number`: The fee that this side paid over the whole horizon.

# Related

  - [`GreedyAllocation`](@ref)
  - [`allocation_fee`](@ref)
  - [`allocation_liquidation_fee`](@ref)
  - [`greedy_fee_delta`](@ref)
  - [`allocation_lot_unit`](@ref)
  - [`finite_sub_allocation`](@ref)
"""
function finite_sub_allocation!(w::VecNum, p::VecNum, lu::Num_VecNum, cash::Number,
                                bgt::Number, sf::Option{<:NamedTuple}, ga::GreedyAllocation)
    if isempty(w)
        # An empty side buys nothing, but it can still owe the forced exit: a delisted asset
        # carries a zero target weight, so it lands on the long side even when that side
        # holds nothing else. `allocation_fee` of the empty book is that charge exactly.
        fee = allocation_fee(sf, p, w)
        return Vector{eltype(w)}(undef, 0), Vector{eltype(w)}(undef, 0),
               Vector{eltype(w)}(undef, 0), cash - fee, fee
    end

    idx = sortperm(w; rev = true)
    w = view(w, idx)
    p = view(p, idx)
    lu = nothing_scalar_array_view(lu, idx)
    sf = permute_side_fees(sf, idx)

    N = length(w)
    # The passes count whole lots, and each share count is its lot count times its lot unit,
    # so a share count carries one rounding of a product and no sum of rounding errors.
    lots = zeros(Int, N)
    shares = zeros(promote_type(eltype(w), eltype(lu)), N)
    # Selling out is a trade, so a side that held money owes a turnover fee before it buys.
    fee = allocation_fee(sf, p, shares)
    acash = cash - fee
    # A side whose target weights are all zero keeps them at zero, and buys nothing.
    sw = sum(w)
    w /= ifelse(iszero(sw), one(sw), sw)

    # First loop: the whole lots of each target, floored so that no asset buys above it.
    for (i, (wi, _pi)) in enumerate(zip(w, p))
        li = nothing_scalar_array_getindex(lu, i)
        n_lots = floor(Int, wi * cash / (_pi * li))
        n_shares = n_lots * li
        cost = n_shares * _pi + greedy_fee_delta(sf, p, shares, i, n_shares)
        if cost > acash
            break
        end
        acash -= cost
        lots[i] = n_lots
        shares[i] = n_shares
    end

    # Second loop
    while acash > 0
        # Calculate equivalent continuous w of what has already been bought. While nothing is
        # held, that weight is zero, so the deficit is the target and a zero target is never
        # bought.
        current_w = p .* shares
        held = sum(current_w)
        current_w /= ifelse(iszero(held), one(held), held)

        deficit = w - current_w

        # Try to buy tickers whose deficit is the greatest.
        i = argmax(deficit)
        li = nothing_scalar_array_getindex(lu, i)
        cost = p[i] * li + greedy_fee_delta(sf, p, shares, i, li)

        # If we can't afford it, go through the rest of the tickers from highest deviation to lowest.
        # The purchase below spends `p[i] * li` and what it adds to the fee, so that is what must
        # be affordable: testing `p[i]` alone overdraws the budget whenever `li > 1`, and refuses
        # affordable buys when `li < 1`.
        while cost > acash
            deficit[i] = 0
            i = argmax(deficit)
            if deficit[i] <= 0
                break
            end
            li = nothing_scalar_array_getindex(lu, i)
            cost = p[i] * li + greedy_fee_delta(sf, p, shares, i, li)
        end
        if deficit[i] <= 0
            break
        end
        # Buy one lot at a time.
        lots[i] += 1
        shares[i] = lots[i] * li
        acash -= cost
    end
    cost = p .* shares
    aw = if any(!iszero, cost)
        cost / sum(cost) * bgt
    else
        range(zero(eltype(w)), zero(eltype(w)); length = N)
    end
    fee = allocation_fee(sf, p, shares)
    acash = cash - sum(cost) - fee
    idx = invperm(idx)
    return view(shares, idx), view(cost, idx), view(aw, idx), acash, fee
end
function _optimise(ga::GreedyAllocation, fai::FiniteAllocationInput; kwargs...)
    p, cash, ca = fai.prices, fai.cash, fai.ca
    w, lbgt, sbgt, lidx, sidx, lsf, ssf = prepare_finite_allocation(fai)
    lu = allocation_lot_unit(ga.unit, ga.sets, ga.strict, fai)
    sshares, scost, sw, _, sfee = finite_sub_allocation!(-view(w, sidx), view(p, sidx),
                                                         nothing_scalar_array_view(lu,
                                                                                   sidx),
                                                         ca(w, p, cash), sbgt, ssf, ga)
    lcash = ca(w, p, cash, sum(scost), sfee)
    lshares, lcost, lw, _, lfee = finite_sub_allocation!(view(w, lidx), view(p, lidx),
                                                         nothing_scalar_array_view(lu,
                                                                                   lidx),
                                                         lcash, lbgt, lsf, ga)
    # The account keeps the cash that no side received, so the rule states it.
    rcash = ca(w, p, cash, sum(scost), sfee, sum(lcost), lfee)
    # An unpriced asset is on neither side, so its row keeps the zero it starts with.
    res = zeros(eltype(w), length(w), 3)
    res[lidx, 1] = lshares
    res[sidx, 1] = -sshares
    res[lidx, 2] = lcost
    res[sidx, 2] = -scost
    res[lidx, 3] = lw
    res[sidx, 3] = -sw
    return GreedyAllocationResult(; retcode = OptimisationSuccess(),
                                  shares = view(res, :, 1), cost = view(res, :, 2),
                                  w = view(res, :, 3), cash = rcash, fees = lfee + sfee,
                                  fb = nothing)
end
"""
    optimise(ga::GreedyAllocation{<:Any, <:Any, <:Any, Nothing},
             fai::FiniteAllocationInput; kwargs...) -> GreedyAllocationResult

Run the Greedy Allocation portfolio optimisation.

This method takes a `GreedyAllocation` with no fallback. The generic `optimise` of a finite allocation runs one with a fallback through its fallback chain.

# Algorithm

 1. Read the target weights, the long and the short side, and the charge of each side with [`prepare_finite_allocation`](@ref). It removes each Unpriced Asset from its side, and gives its target weight to the rule `fai.upw`.
 2. Resolve the lot unit of each asset with [`allocation_lot_unit`](@ref). Each side views a vector lot unit with its own mask.
 3. Start every share count, cost and weight at zero, so an Unpriced Asset holds zero in each.
 4. Allocate the short side on its negated weights with [`finite_sub_allocation!`](@ref), with the short cash that `fai.ca` gives.
 5. Compute the long side's cash with `fai.ca`, from the money of the shares that the short side sold and the fee that it paid. See [`AbstractCollateralAlgorithm`](@ref).
 6. Allocate the long side with [`finite_sub_allocation!`](@ref).
 7. Negate the short side's shares, costs and weights, and put the two sides into one vector per quantity, in the caller's asset order.
 8. Return the cash that the account holds after the trade as `cash`, from the third method of `fai.ca`, and the sum of the fees of the two sides as `fees`. The cash includes the cash that no side received, such as the weight of an Unpriced Asset under `upw = nothing`.

# Arguments

  - `ga`: The greedy allocation optimiser to use.
  - `fai`: The [`FiniteAllocationInput`](@ref) that holds the target weights, the prices, the cash, the collateral algorithm, and the optional horizon and fees.
  - `kwargs`: Ignored.

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
